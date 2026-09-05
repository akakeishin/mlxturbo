"""Small synthetic checks for the Gemma 4 assistant cache contract."""

import inspect

from types import SimpleNamespace

import pytest

try:
    import mlx.core as mx
except ImportError as exc:
    pytest.skip(f"MLX device unavailable: {exc}", allow_module_level=True)

from mlx_lm.models.cache import KVCache, RotatingKVCache

from mlxturbo.gemma4_mtp import (
    Gemma4AssistantModel,
    Gemma4AssistantRunner,
    _restore_prompt_boundary,
    _rollback,
    _sample,
    _snapshot_prompt_boundary,
)


def _tokens(start, count):
    values = mx.arange(start, start + count, dtype=mx.float32)
    keys = values.reshape(1, 1, count, 1)
    return keys, keys + 100


def test_verify_uses_common_measured_qmm_routes():
    source = inspect.getsource(Gemma4AssistantRunner.generate)
    assert "dispatch_scope(unlisted_small_m=False)" in source
    assert "MLXTURBO_GEMMA_QMM_E120" not in source
    assert "_GEMMA_VERIFY_QMM_ROUTES" not in source


def test_rollback_handles_wrapped_rotating_cache():
    cache = RotatingKVCache(max_size=4, keep=0)
    cache.update_and_fetch(*_tokens(0, 4))
    cache.update_and_fetch(*_tokens(4, 1))
    cache.update_and_fetch(*_tokens(5, 3))

    _rollback([cache], 2)
    mx.eval(cache.keys, cache.values)

    assert cache.offset == 6
    assert cache._idx == 4
    assert cache.keys.shape[-2] == 4
    logical = cache._temporal_order(cache.keys)
    mx.eval(logical)
    assert logical.reshape(-1).tolist() == [2.0, 3.0, 4.0, 5.0]


def test_rollback_trims_contiguous_cache():
    cache = KVCache()
    cache.update_and_fetch(*_tokens(0, 5))
    _rollback([cache], 2)
    mx.eval(cache.keys, cache.values)

    assert cache.offset == 3
    assert cache.state[0].shape[-2] == 3


def test_greedy_sample_skips_full_vocabulary_normalization():
    def stochastic_sampler(_logprobs):
        raise AssertionError("temperature-zero sampling must not call the sampler")

    assert _sample(mx.array([[1.0, 4.0, 2.0]]), stochastic_sampler, greedy=True) == 1


def test_greedy_draft_one_sync_is_default_and_preserves_token_ids(monkeypatch):
    monkeypatch.delenv("MLXTURBO_GEMMA_GREEDY_ONE_SYNC", raising=False)

    def embed(ids):
        return ids.astype(mx.float32)[..., None]

    def forward_one(inputs, _shared_kv, _position, _valid_len):
        hidden = inputs[..., -1:]
        return hidden, hidden

    fake = SimpleNamespace(
        _target_embed=embed,
        _target_embed_scale=1.0,
        forward_one=forward_one,
        model=SimpleNamespace(
            embed_tokens=SimpleNamespace(
                as_linear=lambda _hidden: mx.array([[[0.0, 3.0, 1.0]]])
            )
        ),
    )

    tokens = Gemma4AssistantModel.draft_block(
        fake,
        0,
        mx.zeros((1, 1, 1)),
        {},
        0,
        1,
        4,
        lambda _logits: (_ for _ in ()).throw(AssertionError("sampler called")),
        True,
        [],
        [],
    )

    assert tokens == [1, 1, 1]


def test_prompt_boundary_restores_full_and_wrapped_sliding_caches():
    full = KVCache()
    sliding = RotatingKVCache(max_size=4, keep=0)
    full.update_and_fetch(*_tokens(0, 6))
    sliding.update_and_fetch(*_tokens(0, 6))
    boundary = _snapshot_prompt_boundary([full, sliding])

    # Several later updates force the rotating cache to overwrite its ring;
    # trim alone cannot recover the prompt boundary after this point.
    for start in (6, 8, 10):
        full.update_and_fetch(*_tokens(start, 2))
        sliding.update_and_fetch(*_tokens(start, 2))

    _restore_prompt_boundary([full, sliding], boundary)
    full_keys, _ = full.state
    sliding_keys = sliding._temporal_order(sliding.keys)
    mx.eval(full_keys, sliding_keys)

    assert full.offset == 6
    assert full_keys.reshape(-1).tolist() == list(map(float, range(6)))
    assert sliding.offset == 6
    assert sliding_keys.reshape(-1).tolist() == list(map(float, range(6)))
