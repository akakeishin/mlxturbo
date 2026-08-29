"""Phase 0 regression tests for :mod:`mlxturbo.spec`.

The tests use a deterministic fake model and a small cache implementation.  This keeps the
checks independent of a Qwen checkpoint while exercising the public generation/session contract
and the capture/rollback helpers.  They use plain ``assert`` and ``unittest.SkipTest`` so pytest
is optional, matching the other ``bench/test_*.py`` checks in this repository.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
import traceback
import unittest
from unittest import mock


_IMPORT_ERROR = None
try:
    import mlx.core as mx

    import mlxturbo.spec as spec_module
    from mlxturbo.spec import ChatSession, SpecEngine
except ImportError as exc:  # pragma: no cover - exercised only on hosts without MLX
    if not any(
        marker in str(exc)
        for marker in ("No Metal device available", "No module named 'mlx'")
    ):
        raise
    mx = None
    spec_module = None
    ChatSession = None
    # Keep the fake class definitions importable so the file can report skips on
    # hosts without MLX; every executable test calls _require_mlx() first.
    SpecEngine = object
    _IMPORT_ERROR = exc


def _require_mlx() -> None:
    if _IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"MLX/mlxturbo.spec unavailable: {_IMPORT_ERROR}")


def _expect_raise(fn, expected=(Exception,)):
    try:
        fn()
    except expected:
        return
    except BaseException as exc:  # make an unexpected exception type actionable
        raise AssertionError(
            f"expected {expected}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {expected} to be raised")


def _as_ints(value) -> list[int]:
    """Flatten one of the small MLX arrays used by the fake model."""
    raw = value.tolist()
    while raw and isinstance(raw[0], list):
        raw = raw[0]
    return [int(v) for v in raw]


class _FakeCache:
    """Minimal cache implementing the protocol used by SpecEngine."""

    def __init__(self, name: str = "cache"):
        self.name = name
        self.history: list[int] = []
        self.offset = 0
        self.trim_calls: list[int] = []

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        assert n >= 0
        self.trim_calls.append(n)
        if n:
            self.history = self.history[:-n] if n <= len(self.history) else []
            self.offset = max(0, self.offset - n)
        return n


class _NonTrimmableCache:
    def __init__(self):
        self.trim_called = False

    def is_trimmable(self) -> bool:
        return False

    def trim(self, _n: int) -> None:
        self.trim_called = True


class _FakeEngine(SpecEngine):
    """A tiny greedy model whose next token is always ``token + 1``."""

    vocab_size = 32

    def __init__(self, *, eos_token: int | None = None, fail_forward: bool = False):
        # generate() は _head の引数として self.inner.norm を評価するため、
        # _head をオーバーライドしていても属性自体は必要になる。
        self.inner = SimpleNamespace(norm=None)
        self.mtp = SimpleNamespace(norm=None)
        self.hidden_calls: list[tuple[list[int], bool]] = []
        self.mtp_calls: list[list[int]] = []
        self.head_calls = 0
        self.eos_token = eos_token
        self.fail_forward = fail_forward

        def make_cache():
            return [_FakeCache("text")]

        self.text = SimpleNamespace(make_cache=make_cache)

    def _hidden_forward(self, tokens, caches, capture: bool):
        token_ids = _as_ints(tokens)
        self.hidden_calls.append((token_ids, capture))
        if self.fail_forward and not capture:
            raise RuntimeError("fake forward failure")
        next_ids = [(token + 1) % self.vocab_size for token in token_ids]
        hidden = mx.array(next_ids, dtype=mx.float32).reshape(1, len(next_ids), 1)
        return hidden, []

    def _head(self, hidden, _norm):
        self.head_calls += 1
        ids = hidden[..., 0].astype(mx.int32)
        vocab = mx.arange(self.vocab_size)
        return mx.where(ids[..., None] == vocab, mx.array(0.0), mx.array(-1e9))

    def _mtp_append(self, tok_ids, _hiddens, mtp_cache):
        token_ids = _as_ints(tok_ids)
        self.mtp_calls.append(token_ids)
        if mtp_cache is not None:
            mtp_cache.history.extend(token_ids)
            mtp_cache.offset += len(token_ids)
        next_ids = [(token + 1) % self.vocab_size for token in token_ids]
        return mx.array(next_ids, dtype=mx.float32).reshape(1, len(next_ids), 1)


def _patched_spec() -> ExitStack:
    """Patch the concrete MLX cache type so fake caches exercise protocol dispatch."""
    stack = ExitStack()
    stack.enter_context(mock.patch.object(spec_module, "KVCache", _FakeCache))
    return stack


def test_capture_rejects_sharding_group() -> None:
    _require_mlx()
    engine = SpecEngine.__new__(SpecEngine)
    layer = SimpleNamespace(
        is_linear=True,
        sharding_group=object(),
        linear_attn=SimpleNamespace(sharding_group=object()),
    )
    engine.inner = SimpleNamespace(
        embed_tokens=lambda tokens: mx.zeros((tokens.shape[0], tokens.shape[1], 1)),
        fa_idx=0,
        ssm_idx=0,
        layers=[layer],
        sharding_group=object(),
    )
    cache = _FakeCache("text")

    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                spec_module, "create_attention_mask", lambda _x, _cache: None
            )
        )
        stack.enter_context(
            mock.patch.object(
                spec_module, "create_ssm_mask", lambda _x, _cache: None
            )
        )
        _expect_raise(
            lambda: engine._hidden_forward(mx.array([1]), [cache], capture=True),
            (ValueError, NotImplementedError, RuntimeError),
        )


def test_linear_capture_matches_masked_batched_native_forward() -> None:
    _require_mlx()
    from mlx_lm.models.cache import ArraysCache

    from mlxturbo._mlx_compat import DecoderLayer, TextModelArgs

    args = TextModelArgs(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
    )
    layer = DecoderLayer(args, layer_idx=0)
    layer.eval()
    x = mx.random.normal((2, 4, args.hidden_size)).astype(mx.float16)
    native_cache = ArraysCache(2, left_padding=[0, 2])
    capture_cache = ArraysCache(2, left_padding=[0, 2])
    native_mask = native_cache.make_mask(x.shape[1])
    capture_mask = capture_cache.make_mask(x.shape[1])

    native = layer(x, mask=native_mask, cache=native_cache)
    engine = SpecEngine.__new__(SpecEngine)
    captured = engine._linear_capture(
        layer, x, capture_cache, [], capture_mask
    )
    mx.eval(native, captured, *native_cache.state, *capture_cache.state)

    max_out_diff = mx.abs(
        native.astype(mx.float32) - captured.astype(mx.float32)
    ).max().item()
    assert max_out_diff < 1e-2
    assert mx.abs(native_cache[0] - capture_cache[0]).max().item() < 1e-3
    assert mx.abs(native_cache[1] - capture_cache[1]).max().item() < 1e-3
    assert bool(mx.array_equal(native_cache.left_padding, capture_cache.left_padding))


def test_rollback_uses_trimmable_protocol() -> None:
    _require_mlx()
    engine = SpecEngine.__new__(SpecEngine)
    cache = _FakeCache("protocol")

    engine._rollback([cache], [], total=7, consumed=4)

    assert cache.trim_calls == [3]


def test_rollback_rejects_non_trimmable_cache() -> None:
    _require_mlx()
    engine = SpecEngine.__new__(SpecEngine)
    cache = _NonTrimmableCache()

    _expect_raise(
        lambda: engine._rollback([cache], [], total=7, consumed=4),
        (TypeError, ValueError, RuntimeError, NotImplementedError),
    )
    assert not cache.trim_called


def test_max_tokens_zero_returns_no_token_or_callback() -> None:
    _require_mlx()
    engine = _FakeEngine()
    callbacks: list[list[int]] = []

    with _patched_spec():
        result = engine.generate(
            [1],
            max_tokens=0,
            n_draft=2,
            lookup_len=0,
            on_tokens=lambda tokens: callbacks.append(list(tokens)),
        )

    assert result["tokens"] == []
    assert callbacks == []
    assert engine.head_calls == 0


def test_max_tokens_is_an_exact_output_cap() -> None:
    _require_mlx()
    engine = _FakeEngine()
    callbacks: list[int] = []

    with _patched_spec():
        result = engine.generate(
            [1],
            max_tokens=3,
            n_draft=3,
            lookup_len=0,
            on_tokens=lambda tokens: callbacks.extend(tokens),
        )

    assert result["tokens"] == [2, 3, 4]
    assert len(result["tokens"]) == 3
    assert callbacks == result["tokens"]


def test_eos_in_accepted_draft_keeps_callback_and_session_aligned() -> None:
    _require_mlx()
    engine = _FakeEngine(eos_token=4)
    session = ChatSession()
    callbacks: list[int] = []

    with _patched_spec():
        result = engine.generate(
            [1],
            max_tokens=20,
            n_draft=3,
            lookup_len=0,
            eos_ids=(4,),
            on_tokens=lambda tokens: callbacks.extend(tokens),
            session=session,
        )

    assert result["tokens"] == [2, 3, 4]
    assert callbacks == result["tokens"]
    # EOS is the pending predicted token: it is returned to the caller, but is
    # not fed into the hidden state.  The state/session must stop immediately
    # before EOS and must never retain the speculative tokens after it.
    assert session.processed == [1, 2, 3]
    assert session.mtp_cache.history == [2, 3]


def _prepared_reuse_session() -> ChatSession:
    session = ChatSession()
    session.caches = [_FakeCache("text")]
    session.mtp_cache = _FakeCache("mtp")
    session.mtp_cache.history = [2]
    session.mtp_cache.offset = 1
    session.mtp_valid = True
    session.processed = [1, 2]
    session.h_last = mx.zeros((1, 1, 1))
    return session


def _assert_invalidated(session: ChatSession) -> None:
    assert session.caches is None
    assert session.mtp_cache is None
    assert session.mtp_valid is False
    assert session.processed == []
    assert session.h_last is None


def test_reuse_forward_exception_invalidates_session() -> None:
    _require_mlx()
    engine = _FakeEngine(fail_forward=True)
    session = _prepared_reuse_session()

    with _patched_spec():
        _expect_raise(
            lambda: engine.generate(
                [1, 2, 3],
                max_tokens=1,
                n_draft=1,
                lookup_len=0,
                session=session,
            ),
            (RuntimeError,),
        )

    _assert_invalidated(session)


def test_reuse_on_tokens_exception_invalidates_session() -> None:
    _require_mlx()
    engine = _FakeEngine()
    session = _prepared_reuse_session()

    def fail(_tokens):
        raise RuntimeError("fake callback failure")

    with _patched_spec():
        _expect_raise(
            lambda: engine.generate(
                [1, 2, 3],
                max_tokens=1,
                n_draft=1,
                lookup_len=0,
                on_tokens=fail,
                session=session,
            ),
            (RuntimeError,),
        )

    _assert_invalidated(session)


def test_reuse_keyboard_interrupt_invalidates_session() -> None:
    _require_mlx()
    engine = _FakeEngine()
    session = _prepared_reuse_session()

    def interrupt(_tokens):
        raise KeyboardInterrupt()

    with _patched_spec():
        _expect_raise(
            lambda: engine.generate(
                [1, 2, 3],
                max_tokens=1,
                n_draft=1,
                lookup_len=0,
                on_tokens=interrupt,
                session=session,
            ),
            (KeyboardInterrupt,),
        )

    _assert_invalidated(session)


def _run_one_turn(engine, prompt, session, n_draft):
    with _patched_spec():
        return engine.generate(
            prompt,
            max_tokens=1,
            n_draft=n_draft,
            max_draft=0,
            lookup_len=0,
            session=session,
        )


def test_mtp_reenable_after_off_turn_rebuilds_or_maintains_history() -> None:
    _require_mlx()

    # 0 -> on: the first turn has no MTP entries, while the next turn must create
    # a complete history for the prompt suffix rather than append to stale state.
    engine = _FakeEngine()
    session = ChatSession()
    _run_one_turn(engine, [1, 2], session, n_draft=0)
    off_processed = list(session.processed)
    assert off_processed == [1, 2]
    on_prompt = off_processed + [7]
    _run_one_turn(engine, on_prompt, session, n_draft=1)
    assert session.mtp_cache.history == on_prompt[1:]

    # on -> off -> on: either maintain MTP during the off turn or invalidate and
    # fully rebuild it; both must produce exactly the current prompt suffix.
    engine = _FakeEngine()
    session = ChatSession()
    _run_one_turn(engine, [1, 2], session, n_draft=1)
    _run_one_turn(engine, list(session.processed) + [7], session, n_draft=0)
    on_prompt = list(session.processed) + [8]
    _run_one_turn(engine, on_prompt, session, n_draft=1)
    assert session.mtp_cache.history == on_prompt[1:]


def test_zero_draft_zero_lookup_is_single_token_baseline() -> None:
    _require_mlx()
    engine = _FakeEngine()

    with _patched_spec():
        result = engine.generate(
            [1],
            max_tokens=4,
            n_draft=0,
            max_draft=0,
            lookup_len=0,
        )

    assert result["tokens"] == [2, 3, 4, 5]
    assert engine.mtp_calls == []
    assert all(len(tokens) <= 1 for tokens, _capture in engine.hidden_calls)
    assert all(not capture for _tokens, capture in engine.hidden_calls)


def main() -> None:
    tests = [
        test_capture_rejects_sharding_group,
        test_linear_capture_matches_masked_batched_native_forward,
        test_rollback_uses_trimmable_protocol,
        test_rollback_rejects_non_trimmable_cache,
        test_max_tokens_zero_returns_no_token_or_callback,
        test_max_tokens_is_an_exact_output_cap,
        test_eos_in_accepted_draft_keeps_callback_and_session_aligned,
        test_reuse_forward_exception_invalidates_session,
        test_reuse_on_tokens_exception_invalidates_session,
        test_reuse_keyboard_interrupt_invalidates_session,
        test_mtp_reenable_after_off_turn_rebuilds_or_maintains_history,
        test_zero_draft_zero_lookup_is_single_token_baseline,
    ]
    failures = []
    skipped = 0
    for test in tests:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"[SKIP] {test.__name__}: {exc}")
        except Exception as exc:
            failures.append((test.__name__, exc))
            traceback.print_exc()
        else:
            print(f"[PASS] {test.__name__}")
    if failures:
        names = ", ".join(name for name, _ in failures)
        raise SystemExit(f"{len(failures)} Phase 0 spec test(s) failed: {names}")
    if skipped == len(tests):
        raise SystemExit("Phase 0 spec tests could not run: MLX/Metal unavailable")
    print(f"Phase 0 spec tests passed ({len(tests) - skipped}); skipped={skipped}")


if __name__ == "__main__":
    main()
