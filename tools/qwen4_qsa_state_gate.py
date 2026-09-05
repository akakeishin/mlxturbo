"""実 Qwen3.8 Flash-Next の QSA state adapter 境界を検査する。

これは graphbank や production seam の実装ではない。最初の
``full_attention`` 層の実キャッシュを、固定容量の5葉
``[K, V, array_offset, raw_index, pooled_index]`` へ写し、同じ eager
Attention をその5葉から復元したキャッシュで呼ぶ。したがって、ここで
確認できるのは adapter の pack/unpack と eager 経路の一致であり、
``Attention`` 本体を state-pure な graph にしたことではない。

成功は exit 0、検査失敗は exit 1、Metal/モデル/sidecar が無く実行不能なら
exit 2。stdout は JSON 1行だけにする。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

TASK_ID = "qsa_real_layer_state_gate_impl_0905"
WIDTH = 4
KLD_LIMIT = 0.0005
DEFAULT_MODEL = "/Users/ht/models/ddalcu-mlxlm-head4"
DEFAULT_NGRAM = "~/models/ddalcu-ngram"


@dataclass(frozen=True)
class TensorLeafMeta:
    """固定テンソル葉の形状と論理長。

    ``capacity`` は物理軸の長さ、``logical_length`` は offset から見た
    有効長であり、後者は graph の state leaf にはしない host metadata。
    pooled の有無は、rollback 後に production ``keys`` setter が無効化する
    仕様を表すため ``logical_length=0`` で表す。
    """

    name: str
    axis: int
    capacity: int
    logical_length: int
    dtype: str


@dataclass(frozen=True)
class FixedTensorState:
    """5つの tensor leaves と、pack/unpack 用の非テンソル metadata。"""

    leaves: tuple[Any, Any, Any, Any, Any]
    metadata: tuple[TensorLeafMeta, ...]
    pooled_length: int

    @property
    def offset(self) -> int:
        return int(self.metadata[2].logical_length)


def _round_capacity(length: int, step: int = 256) -> int:
    return max(step, ((length + step - 1) // step) * step)


def _copy_array(value: Any) -> Any:
    if value is None:
        return None
    return mx.contiguous(value)


def _eval_cache(caches: list[Any], *values: Any) -> None:
    pending: list[Any] = [v for v in values if v is not None]
    for cache in caches:
        state = getattr(cache, "state", None)
        if isinstance(state, (list, tuple)):
            pending.extend(v for v in state if v is not None)
        elif state is not None:
            pending.append(state)
        indexer = getattr(cache, "indexer", None)
        if indexer is not None:
            for name in ("_buf", "_pooled"):
                value = getattr(indexer, name, None)
                if value is not None:
                    pending.append(value)
    if pending:
        mx.eval(*pending)


def _clone_caches(caches: list[Any], qwen: Any) -> list[Any]:
    """全層の cache を alias しない形で複製する。

    Adapter は QSA 対象層だけを所有するが、full logits の比較には周囲の
    GDN/PLE cache も同じ状態が必要なので、ここでは診断用に全層を複製する。
    """

    out: list[Any] = []
    for source in caches:
        indexer = getattr(source, "indexer", None)
        if indexer is not None:
            target = qwen._AttnCache()
            target.keys = _copy_array(source.keys)
            target.values = _copy_array(source.values)
            target.offset = source.offset
            target.indexer._buf = _copy_array(indexer._buf)
            target.indexer.offset = indexer.offset
            target.indexer._pooled = _copy_array(indexer._pooled)
            target.indexer._pooled_n = indexer._pooled_n
            target.indexer._bs = _copy_array(getattr(indexer, "_bs", None))
            target.indexer._be = _copy_array(getattr(indexer, "_be", None))
            target.indexer._bs_n = getattr(indexer, "_bs_n", 0)
            target.indexer._pooled_f32 = _copy_array(
                getattr(indexer, "_pooled_f32", None)
            )
            target.indexer._pooled_f32_n = getattr(indexer, "_pooled_f32_n", 0)
            out.append(target)
            continue

        state = getattr(source, "state", None)
        target = qwen.ArraysCache(len(state) if isinstance(state, list) else 0)
        if isinstance(state, list):
            target.state = [_copy_array(value) for value in state]
        else:
            target.state = _copy_array(state)
        for name in ("left_padding", "lengths"):
            if hasattr(source, name):
                value = getattr(source, name)
                setattr(target, name, _copy_array(value))
        out.append(target)
    return out


class FixedStateAdapter:
    """汎用5葉の固定容量契約と、Qwen cache の写像だけを持つ境界。"""

    def __init__(self, qwen: Any, capacity: int, compress_ratio: int):
        self.qwen = qwen
        self.capacity = capacity
        self.compress_ratio = compress_ratio
        self.pool_capacity = (capacity + compress_ratio - 1) // compress_ratio

    def pack(self, cache: Any) -> FixedTensorState:
        """Qwen の可変 cache を5葉の固定容量 state へ写す。"""

        if cache.keys is None or cache.values is None:
            raise ValueError("attention KV cache is empty")
        indexer = cache.indexer
        offset = int(cache.offset)
        if offset > self.capacity or indexer.offset != offset:
            raise ValueError(
                f"cache offset/capacity mismatch: offset={offset} "
                f"capacity={self.capacity} indexer={indexer.offset}"
            )
        k_live = cache.keys[..., :offset, :]
        v_live = cache.values[..., :offset, :]
        raw_live = indexer.keys[:, :offset, :] if indexer.keys is not None else None
        if raw_live is None:
            raise ValueError("indexer raw key cache is empty")
        pooled_n = int(indexer._pooled_n) if indexer._pooled is not None else 0
        if pooled_n > self.pool_capacity:
            raise ValueError(
                f"pooled cache exceeds adapter capacity: {pooled_n} > "
                f"{self.pool_capacity}"
            )

        def pad(value: Any, axis: int, capacity: int) -> Any:
            current = value.shape[axis]
            if current == capacity:
                return mx.contiguous(value)
            shape = list(value.shape)
            shape[axis] = capacity - current
            zeros = mx.zeros(tuple(shape), dtype=value.dtype)
            return mx.concatenate([value, zeros], axis=axis)

        pooled_live = (
            indexer._pooled[:, :pooled_n, :]
            if pooled_n
            else mx.zeros((k_live.shape[0], 0, k_live.shape[-1]), dtype=k_live.dtype)
        )
        leaves = (
            pad(k_live, 2, self.capacity),
            pad(v_live, 2, self.capacity),
            mx.array([offset], dtype=mx.int32),
            pad(raw_live, 1, self.capacity),
            pad(pooled_live, 1, self.pool_capacity),
        )
        metadata = (
            TensorLeafMeta("K", 2, self.capacity, offset, str(leaves[0].dtype)),
            TensorLeafMeta("V", 2, self.capacity, offset, str(leaves[1].dtype)),
            TensorLeafMeta("array_offset", 0, 1, offset, str(leaves[2].dtype)),
            TensorLeafMeta("raw_index", 1, self.capacity, offset, str(leaves[3].dtype)),
            TensorLeafMeta(
                "pooled_index", 1, self.pool_capacity, pooled_n, str(leaves[4].dtype)
            ),
        )
        return FixedTensorState(leaves, metadata, pooled_n)

    def unpack(self, state: FixedTensorState, cache: Any) -> None:
        """5葉を Qwen の可変 cache へ戻す。setter を避け pooled を保つ。"""

        k, v, offset_leaf, raw, pooled = state.leaves
        offset = state.offset
        if tuple(k.shape)[2] != self.capacity or tuple(v.shape)[2] != self.capacity:
            raise ValueError("K/V fixed capacity changed")
        if tuple(raw.shape)[1] != self.capacity:
            raise ValueError("raw index fixed capacity changed")
        cache.keys = k
        cache.values = v
        cache.offset = offset
        indexer = cache.indexer
        indexer._buf = raw
        indexer.offset = offset
        indexer._pooled = (
            pooled[:, : state.pooled_length, :]
            if state.pooled_length
            else None
        )
        indexer._pooled_n = state.pooled_length
        # These are deterministic metadata caches, not additional state leaves.
        indexer._bs = None
        indexer._be = None
        indexer._bs_n = 0
        indexer._pooled_f32 = None
        indexer._pooled_f32_n = 0
        if int(offset_leaf[0].item()) != offset:
            raise ValueError("array_offset leaf disagrees with metadata")

    def rollback(
        self, state: FixedTensorState, keep: int, width: int, base_offset: int
    ) -> FixedTensorState:
        """verify後の ``keep`` 件を残す rollback。

        production の ``spec_flash.rollback`` と同じく keep は verify 幅内の
        件数で、実際の offset は ``base_offset + keep``。keep==width は no-op。
        keep<width では ``_IndexerCache.keys`` setter が pooled を無効化する
        ため、pooled leaf も論理的に空へ戻す。
        """

        if keep < 1 or keep > width:
            raise ValueError(f"keep must be in 1..{width}: {keep}")
        if keep == width:
            return state
        target = base_offset + keep
        k, v, _offset, raw, pooled = state.leaves
        token_pos = mx.arange(self.capacity)
        pool_pos = mx.arange(self.pool_capacity)
        token_live = token_pos < target
        empty_k = mx.zeros(k.shape, dtype=k.dtype)
        empty_v = mx.zeros(v.shape, dtype=v.dtype)
        empty_raw = mx.zeros(raw.shape, dtype=raw.dtype)
        empty_pooled = mx.zeros(pooled.shape, dtype=pooled.dtype)
        leaves = (
            mx.where(token_live[None, None, :, None], k, empty_k),
            mx.where(token_live[None, None, :, None], v, empty_v),
            mx.array([target], dtype=_offset_dtype(state)),
            mx.where(token_live[None, :, None], raw, empty_raw),
            mx.where(
                pool_pos[None, :, None] < (target // self.compress_ratio),
                pooled,
                empty_pooled,
            ),
        )
        metadata = tuple(
            TensorLeafMeta(
                m.name,
                m.axis,
                m.capacity,
                target if m.name != "pooled_index" else 0,
                m.dtype,
            )
            for m in state.metadata
        )
        return FixedTensorState(leaves, metadata, 0)


def _offset_dtype(state: FixedTensorState) -> Any:
    return state.leaves[2].dtype


def _logical_equal(left: FixedTensorState, right: FixedTensorState) -> bool:
    if left.offset != right.offset or left.pooled_length != right.pooled_length:
        return False
    offset = left.offset
    arrays = (
        (left.leaves[0][..., :offset, :], right.leaves[0][..., :offset, :]),
        (left.leaves[1][..., :offset, :], right.leaves[1][..., :offset, :]),
        (left.leaves[3][:, :offset, :], right.leaves[3][:, :offset, :]),
    )
    if left.pooled_length:
        arrays += (
            (
                left.leaves[4][:, : left.pooled_length, :],
                right.leaves[4][:, : right.pooled_length, :],
            ),
        )
    mx.eval(*(a for pair in arrays for a in pair))
    checks = [mx.all(a == b) for a, b in arrays]
    mx.eval(*checks)
    return all(bool(value) for value in checks)


def _max_abs(left: Any, right: Any) -> float:
    diff = mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
    mx.eval(diff)
    return float(mx.max(diff).item())


def _logit_quality(reference: Any, candidate: Any, topk: int = 256) -> dict[str, Any]:
    """bench/quant_eval.py と同じ top-K 近似 KLD を計算する。"""

    p_logits = reference.astype(mx.float32).reshape(-1, reference.shape[-1])
    q_logits = candidate.astype(mx.float32).reshape(-1, candidate.shape[-1])
    logp = p_logits - mx.logsumexp(p_logits, axis=-1, keepdims=True)
    logq = q_logits - mx.logsumexp(q_logits, axis=-1, keepdims=True)
    k = min(topk, logp.shape[-1])
    idx = mx.argpartition(-logp, k - 1, axis=-1)[..., :k]
    top_logp = mx.take_along_axis(logp, idx, axis=-1)
    order = mx.argsort(-top_logp, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    top_logp = mx.take_along_axis(top_logp, order, axis=-1)
    top_logq = mx.take_along_axis(logq, idx, axis=-1)
    argmax_p = mx.argmax(p_logits, axis=-1)
    argmax_q = mx.argmax(q_logits, axis=-1)
    mx.eval(top_logp, top_logq, argmax_p, argmax_q)
    p = np.exp(np.array(top_logp, dtype=np.float64))
    kld = p * (
        np.array(top_logp, dtype=np.float64) - np.array(top_logq, dtype=np.float64)
    )
    agree = np.array(argmax_p) == np.array(argmax_q)
    return {
        "positions": int(kld.size),
        "topk": int(k),
        "kld_mean": float(kld.mean()),
        "kld_max": float(kld.max()),
        "top1_agree_rate": float(agree.mean()),
        "pass": bool(float(kld.mean()) <= KLD_LIMIT),
    }


def _run_step(
    model: Any,
    ids: Any,
    caches: list[Any],
    qwen: Any,
    target_attn: Any,
    capture: bool = True,
) -> tuple[Any, Any | None]:
    """eager model step。対象 Attention 出力も同時に捕まえる。"""

    captured: list[Any] = []
    original = qwen.Attention.__call__

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        if capture and self is target_attn:
            captured.append(result)
        return result

    if capture:
        qwen.Attention.__call__ = wrapped
    try:
        logits = model(ids, cache=caches)
        _eval_cache(caches, logits, *(captured or []))
    finally:
        if capture:
            qwen.Attention.__call__ = original
    return logits, (captured[-1] if captured else None)


def _rollback_native(cache: Any, base_offset: int, keep: int, width: int) -> None:
    if keep == width:
        return
    cache.trim(width - keep)
    if cache.indexer.keys is not None:
        cache.indexer.keys = cache.indexer.keys[:, : base_offset + keep]


def _state_shape(state: FixedTensorState) -> list[list[int]]:
    return [list(value.shape) for value in state.leaves]


def _cache_state_matches(adapter: FixedStateAdapter, state: FixedTensorState, cache: Any) -> bool:
    return _logical_equal(state, adapter.pack(cache))


def _make_ids(tokenizer: Any, requested: int, question: str) -> Any:
    from _bench_text import long_prompts

    body = long_prompts(tokenizer, requested + 256, [question])[0]
    ids = mx.array(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": body}], add_generation_prompt=True
        )
    )[None]
    return ids


def _run_gate(args: argparse.Namespace) -> dict[str, Any]:
    global mx

    if not mx.metal.is_available():
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": "Metal is unavailable",
        }
    model_path = Path(os.path.expanduser(args.model))
    ngram_path = Path(os.path.expanduser(args.ngram)) if args.ngram else None
    if not model_path.exists():
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": f"model path does not exist: {model_path}",
        }
    if ngram_path is None or not ngram_path.exists():
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": f"n-gram sidecar path does not exist: {ngram_path}",
        }

    # The model omits the 32GB n-gram table; load it through the sidecar before
    # importing qwen4_exp, whose module-level flag controls allocation.
    os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")
    mx.set_default_device(mx.gpu)
    import mlxturbo  # noqa: F401  (architecture registry hook)
    from mlx_lm import load
    import mlx_lm.models.qwen4_exp as qwen
    from mlxturbo.ngram_stream import install

    with contextlib.redirect_stdout(io.StringIO()):
        model, tokenizer = load(str(model_path))
        install(model, str(ngram_path))

    layer_types = list(model.args.text.layer_types)
    if "full_attention" not in layer_types:
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": "model has no full_attention layer",
        }
    target_layer_idx = layer_types.index("full_attention")
    target_layer = model.model.layers[target_layer_idx]
    target_attn = target_layer.self_attn
    token_budget = int(target_attn.indexer.token_budget)
    prefix = max(int(args.prefix), token_budget)
    requested = prefix + WIDTH * 2
    ids = _make_ids(tokenizer, requested, args.question)
    if ids.shape[1] < requested:
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": f"prompt has {ids.shape[1]} tokens, needs {requested}",
        }

    # Prefix is evaluated once, in normal eager chunks, and then all branches
    # start from the same complete cache state.
    prefix_cache = model.make_cache()
    chunk = max(1, int(args.chunk))
    with contextlib.redirect_stdout(io.StringIO()):
        for start in range(0, prefix, chunk):
            end = min(prefix, start + chunk)
            _run_step(model, ids[:, start:end], prefix_cache, qwen, target_attn, capture=False)
    _eval_cache(prefix_cache)

    target_prefix = prefix_cache[target_layer_idx]
    capacity = _round_capacity(prefix + WIDTH * 2)
    adapter = FixedStateAdapter(
        qwen, capacity, int(target_attn.indexer.compress_ratio)
    )
    prefix_state = adapter.pack(target_prefix)
    prefix_roundtrip = _clone_caches(prefix_cache, qwen)
    adapter.unpack(prefix_state, prefix_roundtrip[target_layer_idx])
    prefix_state_ok = _cache_state_matches(
        adapter, prefix_state, prefix_roundtrip[target_layer_idx]
    )

    step1_ids = ids[:, prefix : prefix + WIDTH]
    step2_ids = ids[:, prefix + WIDTH : prefix + WIDTH * 2]

    with contextlib.redirect_stdout(io.StringIO()):
        baseline_cache = _clone_caches(prefix_cache, qwen)
        baseline1_logits, baseline1_attn = _run_step(
            model, step1_ids, baseline_cache, qwen, target_attn
        )
        baseline1_state = adapter.pack(baseline_cache[target_layer_idx])
        baseline2_logits, baseline2_attn = _run_step(
            model, step2_ids, baseline_cache, qwen, target_attn
        )
        baseline2_state = adapter.pack(baseline_cache[target_layer_idx])

        # Adapter path: restore the five leaves before each eager step. This is
        # the executable boundary; it deliberately does not claim a pure graph.
        adapter_cache = _clone_caches(prefix_cache, qwen)
        adapter.unpack(prefix_state, adapter_cache[target_layer_idx])
        adapter1_logits, adapter1_attn = _run_step(
            model, step1_ids, adapter_cache, qwen, target_attn
        )
        adapter1_state = adapter.pack(adapter_cache[target_layer_idx])
        adapter1_cache_ok = _cache_state_matches(
            adapter, adapter1_state, adapter_cache[target_layer_idx]
        )
        adapter.unpack(adapter1_state, adapter_cache[target_layer_idx])
        adapter2_logits, adapter2_attn = _run_step(
            model, step2_ids, adapter_cache, qwen, target_attn
        )
        adapter2_state = adapter.pack(adapter_cache[target_layer_idx])

        # Run the same state/input twice from independent restored caches. Both
        # must use the same fixed shape and advance from the prefix offset once.
        replay_a = _clone_caches(prefix_cache, qwen)
        adapter.unpack(prefix_state, replay_a[target_layer_idx])
        replay1_logits, replay1_attn = _run_step(
            model, step1_ids, replay_a, qwen, target_attn
        )
        replay_a_state = adapter.pack(replay_a[target_layer_idx])
        replay_b = _clone_caches(prefix_cache, qwen)
        adapter.unpack(prefix_state, replay_b[target_layer_idx])
        replay2_logits, replay2_attn = _run_step(
            model, step1_ids, replay_b, qwen, target_attn
        )
        replay_b_state = adapter.pack(replay_b[target_layer_idx])

    quality1 = _logit_quality(baseline1_logits, adapter1_logits)
    quality2 = _logit_quality(baseline2_logits, adapter2_logits)
    replay_quality = _logit_quality(replay1_logits, replay2_logits)
    target_output = {
        "step1_max_abs": _max_abs(baseline1_attn, adapter1_attn),
        "step2_max_abs": _max_abs(baseline2_attn, adapter2_attn),
        "replay_max_abs": _max_abs(replay1_attn, replay2_attn),
    }

    state_checks = {
        "prefix_roundtrip": prefix_state_ok,
        "step1_logical_equal": _logical_equal(baseline1_state, adapter1_state),
        "step2_logical_equal": _logical_equal(baseline2_state, adapter2_state),
        "adapter1_matches_cache": adapter1_cache_ok,
        "adapter2_matches_cache": _cache_state_matches(
            adapter, adapter2_state, adapter_cache[target_layer_idx]
        ),
    }

    replay_checks = {
        "same_shape": _state_shape(replay_a_state) == _state_shape(replay_b_state),
        "same_offset": replay_a_state.offset == replay_b_state.offset == prefix + WIDTH,
        "no_stale_offset_on_sequential_step": adapter2_state.offset
        == prefix + WIDTH * 2,
        "logical_equal": _logical_equal(replay_a_state, replay_b_state),
        "logits_quality": replay_quality,
    }

    # Rollback is checked from the full first verify state. keep is relative to
    # the four newly verified positions, as in spec_flash.rollback.
    rollback_checks: dict[str, Any] = {}
    for keep in (1, 3, 4):
        adapter_rb = adapter.rollback(baseline1_state, keep, WIDTH, prefix)
        # baseline_cache is after step2. Restore only the target layer to its
        # first-step state; other layer caches are irrelevant to this local
        # rollback check and need not pay another model forward.
        native_rb_cache = _clone_caches(baseline_cache, qwen)
        adapter.unpack(baseline1_state, native_rb_cache[target_layer_idx])
        _rollback_native(native_rb_cache[target_layer_idx], prefix, keep, WIDTH)
        native_state = adapter.pack(native_rb_cache[target_layer_idx])
        restored_rb_cache = _clone_caches(prefix_cache, qwen)
        adapter.unpack(adapter_rb, restored_rb_cache[target_layer_idx])
        rollback_checks[str(keep)] = {
            "logical_cache_equal": _logical_equal(native_state, adapter_rb),
            "adapter_unpack_equal": _cache_state_matches(
                adapter, adapter_rb, restored_rb_cache[target_layer_idx]
            ),
            "offset": adapter_rb.offset,
            "shape_stable": _state_shape(adapter_rb) == _state_shape(baseline1_state),
        }

    rollback_ok = all(
        item["logical_cache_equal"]
        and item["adapter_unpack_equal"]
        and item["shape_stable"]
        for item in rollback_checks.values()
    )
    quality_ok = (
        quality1["pass"]
        and quality2["pass"]
        and replay_quality["pass"]
        and all(value <= 1e-3 for value in target_output.values())
    )
    replay_ok = all(
        replay_checks[key]
        for key in ("same_shape", "same_offset", "no_stale_offset_on_sequential_step", "logical_equal")
    ) and replay_quality["pass"]
    all_ok = prefix_state_ok and quality_ok and all(state_checks.values()) and replay_ok and rollback_ok
    return {
        "task_id": TASK_ID,
        "status": "PASS" if all_ok else "FAIL",
        "model": str(model_path),
        "target_layer": target_layer_idx,
        "target_layer_type": layer_types[target_layer_idx],
        "width": WIDTH,
        "prefix_tokens": prefix,
        "token_budget": token_budget,
        "fixed_capacity": capacity,
        "state_leaves": ["K", "V", "array_offset", "raw_index", "pooled_index"],
        "quality_rule": {"kld_limit": KLD_LIMIT, "metric": "top-K KL(p||q)"},
        "checks": {
            "output_logits_agreement": {
                "step1": quality1,
                "step2": quality2,
                "attention_output": target_output,
                "pass": quality_ok,
            },
            "logical_cache_state_agreement": {**state_checks, "pass": all(state_checks.values())},
            "same_shape_replay": {**replay_checks, "pass": replay_ok},
            "rollback": {"cases": rollback_checks, "pass": rollback_ok},
        },
        "limitations": [
            "Attention/indexer/KV writes remain the existing eager Python implementation; no mx.compile state-pure real Attention step is claimed.",
            "The adapter owns only the selected full-attention layer; other layer caches are cloned diagnostically to make full-logit comparisons fair.",
            "pooled_index is invalidated on partial rollback to match _IndexerCache.keys setter; a production state plan still needs an explicit pooled-generation contract.",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ngram", default=DEFAULT_NGRAM)
    parser.add_argument("--prefix", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--question", default="上の文書の要点を5つに整理してください。")
    return parser.parse_args()


def main() -> int:
    global mx
    try:
        import mlx.core as mx  # type: ignore[no-redef]
    except Exception as exc:
        print(
            json.dumps(
                {"task_id": TASK_ID, "status": "UNAVAILABLE", "reason": f"MLX import failed: {type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2

    args = _parse_args()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = _run_gate(args)
    except Exception as exc:  # diagnostic result must remain machine-readable
        result = {
            "task_id": TASK_ID,
            "status": "FAIL",
            "reason": f"{type(exc).__name__}: {exc}",
            "limitations": [
                "The real eager gate raised before completing; inspect the exception before treating this as a state-pure result."
            ],
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return {"PASS": 0, "FAIL": 1, "UNAVAILABLE": 2}.get(result["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
