"""実 Qwen3.8 Flash-Next の QSA state-pure component gate。

最初の QSA 対応 ``full_attention`` 層について、固定容量の5葉を入力に
取る同一 ``mx.compile`` callable を実 Attention の QKV/RoPE と照合する。
この最小段階では、mutable cache object を持たない K/V・raw index・pooled
index の更新までを実装する。QSA の block top-k と sparse SDPA は pure な
状態境界を越える別の写像なので実装せず、結果は ``PARTIAL`` とする。

stdout は JSON 1行だけ。成功した component gate は exit 1 (PARTIAL)、
検査失敗も exit 1、Metal/モデル/sidecar が無ければ exit 2。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
for path in (REPO_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import qwen4_qsa_state_gate as B0

TASK_ID = "qsa_pure_component_gate_impl_0905"
WIDTH = 4
DEFAULT_MODEL = B0.DEFAULT_MODEL
DEFAULT_NGRAM = B0.DEFAULT_NGRAM


def _pure_step_factory(attn: Any, qwen: Any, rope: Any, ratio: int):
    """実層の重みだけを閉じ込めた、cache object 無しの width-4 関数。"""

    q_proj = attn.q_proj
    k_proj = attn.k_proj
    v_proj = attn.v_proj
    o_q_proj = attn.indexer.index_qk_proj
    q_norm = attn.q_norm
    k_norm = attn.k_norm
    index_k_norm = attn.indexer.k_layernorm
    n_heads = attn.n_heads
    n_kv_heads = attn.n_kv_heads
    index_heads = attn.indexer.n_heads
    index_dim = attn.indexer.head_dim

    def rope_apply(value: Any, positions: Any) -> Any:
        cos, sin = rope(positions[None])
        return qwen._rope_partial(value, cos[:, None], sin[:, None])

    def put_sequence(base: Any, update: Any, offset: Any, axis: int) -> Any:
        width = update.shape[axis]
        slots = offset + mx.arange(width, dtype=offset.dtype)
        if axis == 2:
            indices = mx.broadcast_to(
                slots.reshape(1, 1, width, 1), update.shape
            )
        else:
            indices = mx.broadcast_to(slots.reshape(1, width, 1), update.shape)
        return mx.put_along_axis(base, indices, update, axis=axis)

    def pure_step(
        x4: Any,
        k_state: Any,
        v_state: Any,
        offset_tensor: Any,
        raw_index: Any,
        pooled_index: Any,
    ):
        # S is intentionally a traced fixed shape. The position is read only
        # from the tensor leaf; no Python offset or cache object is captured.
        batch, steps, _ = x4.shape
        if steps != WIDTH:
            raise ValueError(f"pure gate requires width {WIDTH}, got {steps}")

        qg = q_proj(x4)
        q, gate = mx.split(
            qg.reshape(batch, steps, n_heads, -1), 2, axis=-1
        )
        q = q_norm(q).transpose(0, 2, 1, 3)
        kk = k_proj(x4)
        vv = v_proj(x4)
        k = k_norm(kk.reshape(batch, steps, n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        v = vv.reshape(batch, steps, n_kv_heads, -1).transpose(0, 2, 1, 3)
        positions = offset_tensor + mx.arange(steps, dtype=offset_tensor.dtype)
        q = rope_apply(q, positions)
        k = rope_apply(k, positions)
        k2 = put_sequence(k_state, k, offset_tensor, axis=2)
        v2 = put_sequence(v_state, v, offset_tensor, axis=2)

        index_qk = o_q_proj(x4)
        split = index_heads * index_dim
        raw_new = index_qk[..., split:].reshape(batch, steps, index_dim)
        raw2 = put_sequence(raw_index, raw_new, offset_tensor, axis=1)

        # Width 4 crosses exactly one ratio-4 block for every offset%4 phase.
        # Gather that one block with tensor indices, then write one pooled row.
        old_blocks = offset_tensor // ratio
        block_start = old_blocks * ratio
        block_tokens = block_start + mx.arange(ratio, dtype=offset_tensor.dtype)
        raw_block_idx = mx.broadcast_to(
            block_tokens.reshape(1, ratio, 1), (batch, ratio, index_dim)
        )
        raw_block = mx.take_along_axis(raw2, raw_block_idx, axis=1)
        pooled_new = index_k_norm(
            raw_block.reshape(batch, 1, ratio, index_dim)
            .astype(mx.float32)
            .mean(axis=2)
            .astype(raw2.dtype)
        )
        pooled_positions = block_start.reshape(1, 1)
        cos_p, sin_p = rope(pooled_positions)
        pooled_new = qwen._rope_partial(pooled_new, cos_p, sin_p)
        pooled_idx = mx.broadcast_to(
            old_blocks.reshape(1, 1, 1), (batch, 1, index_dim)
        )
        pooled2 = mx.put_along_axis(
            pooled_index, pooled_idx, pooled_new, axis=1
        )

        # The first result is deliberately the projected query, not a claimed
        # full attention output. Sparse selection + SDPA are the omitted phase.
        del gate
        return q, k2, v2, offset_tensor + mx.array(
            [WIDTH], dtype=offset_tensor.dtype
        ), raw2, pooled2

    return pure_step


def _pure_rollback_factory(capacity: int, pool_capacity: int, ratio: int):
    """固定容量 state の tensor keep rollback。"""

    def rollback(
        k_state: Any,
        v_state: Any,
        offset_tensor: Any,
        raw_index: Any,
        pooled_index: Any,
        keep_tensor: Any,
    ):
        target = offset_tensor - (WIDTH - keep_tensor)
        token_pos = mx.arange(capacity, dtype=target.dtype)
        pool_pos = mx.arange(pool_capacity, dtype=target.dtype)
        token_live = token_pos < target[0]
        pool_live = pool_pos < (target[0] // ratio)
        return (
            mx.where(token_live[None, None, :, None], k_state, mx.zeros_like(k_state)),
            mx.where(token_live[None, None, :, None], v_state, mx.zeros_like(v_state)),
            target,
            mx.where(token_live[None, :, None], raw_index, mx.zeros_like(raw_index)),
            mx.where(pool_live[None, :, None], pooled_index, mx.zeros_like(pooled_index)),
        )

    return rollback


def _state_with_leaves(reference: Any, leaves: tuple[Any, ...]) -> Any:
    return B0.FixedTensorState(leaves, reference.metadata, reference.pooled_length)


def _rollback_reference(state: Any, base_offset: int, keep: int, ratio: int) -> Any:
    """width-4 eager結果から、採用済みtokenだけを残す参照stateを作る。"""

    target = base_offset + keep
    pooled_length = target // ratio
    k, v, offset, raw, pooled = state.leaves

    def zero_tail(value: Any, axis: int, logical: int) -> Any:
        positions = mx.arange(value.shape[axis])
        live = positions < logical
        shape = [1] * value.ndim
        shape[axis] = value.shape[axis]
        return mx.where(live.reshape(shape), value, mx.zeros_like(value))

    leaves = (
        zero_tail(k, 2, target),
        zero_tail(v, 2, target),
        mx.array([target], dtype=offset.dtype),
        zero_tail(raw, 1, target),
        zero_tail(pooled, 1, pooled_length),
    )
    metadata = tuple(
        B0.TensorLeafMeta(
            item.name,
            item.axis,
            item.capacity,
            pooled_length if item.name == "pooled_index" else target,
            item.dtype,
        )
        for item in state.metadata
    )
    return B0.FixedTensorState(leaves, metadata, pooled_length)


def _eager_component_update(
    x4: Any,
    state: Any,
    adapter: Any,
    target_attn: Any,
    rope: Any,
    qwen: Any,
) -> Any:
    """同じhidden入力で既存mutable QSA/KV更新だけを参照実行する。"""

    cache = qwen._AttnCache()
    adapter.unpack(state, cache)
    offset = cache.offset
    positions = (offset + mx.arange(x4.shape[1]))[None]
    target_attn.indexer(x4, rope, cache.indexer, offset, positions)
    target_attn._qkv(x4, positions, rope, cache)
    B0._eval_cache([cache])
    return adapter.pack(cache)


def _canonicalize_pooled(
    state: Any, indexer: Any, qwen: Any, rope: Any, ratio: int
) -> Any:
    """Seed completed pooled rows missing from a cache packed at the budget edge."""

    completed = state.offset // ratio
    if state.pooled_length >= completed:
        return state
    k, v, offset, raw, pooled = state.leaves
    raw_live = raw[:, : completed * ratio, :].reshape(
        raw.shape[0], completed, ratio, raw.shape[-1]
    )
    pooled_live = indexer.k_layernorm(
        raw_live.astype(mx.float32).mean(axis=2).astype(raw.dtype)
    )
    starts = mx.arange(completed, dtype=offset.dtype) * ratio
    cos, sin = rope(starts[None])
    pooled_live = qwen._rope_partial(pooled_live, cos, sin)
    tail = mx.zeros(
        (pooled.shape[0], pooled.shape[1] - completed, pooled.shape[2]),
        dtype=pooled.dtype,
    )
    pooled2 = mx.concatenate([pooled_live, tail], axis=1)
    metadata = list(state.metadata)
    metadata[-1] = B0.TensorLeafMeta(
        metadata[-1].name,
        metadata[-1].axis,
        metadata[-1].capacity,
        completed,
        metadata[-1].dtype,
    )
    return B0.FixedTensorState(
        (k, v, offset, raw, pooled2), tuple(metadata), completed
    )


def _compare_q(pure_q: Any, eager_q: Any) -> dict[str, Any]:
    diff = B0._max_abs(pure_q, eager_q)
    return {"max_abs": diff, "pass": diff <= 1e-3}


def _compare_state(left: Any, right: Any) -> dict[str, Any]:
    """論理領域の不一致を葉単位で返す。"""

    offset = min(left.offset, right.offset)
    arrays = {
        "K": (left.leaves[0][..., :offset, :], right.leaves[0][..., :offset, :]),
        "V": (left.leaves[1][..., :offset, :], right.leaves[1][..., :offset, :]),
        "raw_index": (
            left.leaves[3][:, :offset, :],
            right.leaves[3][:, :offset, :],
        ),
    }
    pooled = min(left.pooled_length, right.pooled_length)
    if pooled:
        arrays["pooled_index"] = (
            left.leaves[4][:, :pooled, :],
            right.leaves[4][:, :pooled, :],
        )
    leaf_diffs = {name: B0._max_abs(a, b) for name, (a, b) in arrays.items()}
    exact = B0._logical_equal(left, right)
    return {
        "left_offset": left.offset,
        "right_offset": right.offset,
        "left_pooled_length": left.pooled_length,
        "right_pooled_length": right.pooled_length,
        "leaf_max_abs": leaf_diffs,
        "pass": exact,
    }


def _eager_qkv_step(
    model: Any,
    ids: Any,
    caches: list[Any],
    qwen: Any,
    target_attn: Any,
) -> tuple[Any, Any]:
    """既存 eager model step と target の q/k/v を一緒に捕まえる。"""

    hidden: list[Any] = []
    captured: list[Any] = []
    original = qwen.Attention._qkv

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        if self is target_attn:
            hidden.append(args[0])
            captured.append(result)
        return result

    qwen.Attention._qkv = wrapped
    try:
        B0._run_step(model, ids, caches, qwen, target_attn, capture=False)
    finally:
        qwen.Attention._qkv = original
    if not captured:
        raise RuntimeError("target Attention._qkv was not called")
    B0._eval_cache(caches, *captured[-1])
    return hidden[-1], captured[-1]


def _run_eager(
    model: Any,
    ids: Any,
    caches: list[Any],
    qwen: Any,
    target_attn: Any,
    adapter: Any,
    layer_idx: int,
) -> tuple[Any, Any, Any]:
    hidden, qkv = _eager_qkv_step(model, ids, caches, qwen, target_attn)
    return hidden, qkv, adapter.pack(caches[layer_idx])


def _load_gate(args: argparse.Namespace, mx_module: Any):
    if not mx_module.metal.is_available():
        return None, {"status": "UNAVAILABLE", "reason": "Metal is unavailable"}
    model_path = Path(os.path.expanduser(args.model))
    ngram_path = Path(os.path.expanduser(args.ngram)) if args.ngram else None
    if not model_path.exists():
        return None, {
            "status": "UNAVAILABLE",
            "reason": f"model path does not exist: {model_path}",
        }
    if ngram_path is None or not ngram_path.exists():
        return None, {
            "status": "UNAVAILABLE",
            "reason": f"n-gram sidecar path does not exist: {ngram_path}",
        }
    os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")
    mx_module.set_default_device(mx_module.gpu)
    import mlxturbo  # noqa: F401
    from mlx_lm import load
    import mlx_lm.models.qwen4_exp as qwen
    from mlxturbo.ngram_stream import install

    with contextlib.redirect_stdout(io.StringIO()):
        model, tokenizer = load(str(model_path))
        install(model, str(ngram_path))
    return (model, tokenizer, qwen, model_path), None


def _run_gate(args: argparse.Namespace, mx_module: Any) -> dict[str, Any]:
    global mx
    mx = mx_module
    B0.mx = mx_module
    loaded, unavailable = _load_gate(args, mx_module)
    if unavailable is not None:
        return {"task_id": TASK_ID, **unavailable}
    model, tokenizer, qwen, model_path = loaded
    layer_types = list(model.args.text.layer_types)
    if "full_attention" not in layer_types:
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": "model has no full_attention layer",
        }
    layer_idx = layer_types.index("full_attention")
    target_attn = model.model.layers[layer_idx].self_attn
    budget = int(target_attn.indexer.token_budget)
    ratio = int(target_attn.indexer.compress_ratio)
    if ratio != WIDTH:
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": f"QSA compress ratio is {ratio}, fixed-M4 gate requires {WIDTH}",
        }
    if getattr(target_attn, "_wide_qkv", None) is not None:
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": "wide fused QKV projection is not represented by this pure component",
        }
    prefix = max(int(args.prefix), budget)
    ids = B0._make_ids(tokenizer, prefix + WIDTH * 2, args.question)
    if ids.shape[1] < prefix + WIDTH * 2:
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": f"prompt has {ids.shape[1]} tokens, needs {prefix + WIDTH * 2}",
        }

    with contextlib.redirect_stdout(io.StringIO()):
        prefix_cache = model.make_cache()
        chunk = max(1, int(args.chunk))
        for start in range(0, prefix, chunk):
            end = min(prefix, start + chunk)
            B0._run_step(model, ids[:, start:end], prefix_cache, qwen, target_attn, capture=False)
        B0._eval_cache(prefix_cache)

    target_prefix = prefix_cache[layer_idx]
    capacity = B0._round_capacity(prefix + WIDTH * 2)
    adapter = B0.FixedStateAdapter(qwen, capacity, ratio)
    state0 = _canonicalize_pooled(
        adapter.pack(target_prefix),
        target_attn.indexer,
        qwen,
        model.model.rope,
        ratio,
    )
    pool_capacity = adapter.pool_capacity

    # Keep the pure callable free of the mutable Attention cache. The rotary
    # callable is a read-only model component captured by the diagnostic.
    pure_fn = _pure_step_factory(target_attn, qwen, model.model.rope, ratio)
    rollback_fn = _pure_rollback_factory(capacity, pool_capacity, ratio)
    compiled_step = mx_module.compile(pure_fn)
    compiled_rollback = mx_module.compile(rollback_fn)

    # Baseline eager sequence supplies real x4/qkv and logical leaves at two
    # consecutive offsets. The compiled callable is reused for both calls.
    with contextlib.redirect_stdout(io.StringIO()):
        eager_cache = B0._clone_caches(prefix_cache, qwen)
        x1, qkv1, eager_state1 = _run_eager(
            model,
            ids[:, prefix : prefix + WIDTH],
            eager_cache,
            qwen,
            target_attn,
            adapter,
            layer_idx,
        )
        x2, qkv2, eager_state2 = _run_eager(
            model,
            ids[:, prefix + WIDTH : prefix + WIDTH * 2],
            eager_cache,
            qwen,
            target_attn,
            adapter,
            layer_idx,
        )

    def evaluate(*values: Any) -> None:
        mx_module.eval(*values)

    def state_from(reference: Any, leaves: Any) -> Any:
        return _state_with_leaves(reference, tuple(leaves))

    # Two consecutive calls use the same compiled callable and carry only the
    # tensor leaves across the boundary.
    pure1 = compiled_step(x1, *state0.leaves)
    evaluate(*pure1)
    pure_state1 = state_from(eager_state1, pure1[1:])
    step1_q = _compare_q(pure1[0], qkv1[0])
    step1_state = B0._logical_equal(pure_state1, eager_state1)

    pure2 = compiled_step(x2, *pure1[1:])
    evaluate(*pure2)
    pure_state2 = state_from(eager_state2, pure2[1:])
    step2_q = _compare_q(pure2[0], qkv2[0])
    step2_state = B0._logical_equal(pure_state2, eager_state2)

    # Replay from the original leaves after the second call. A stale Python
    # offset would write this same-shape replay at the wrong position.
    replay = compiled_step(x1, *state0.leaves)
    evaluate(*replay)
    replay_state = state_from(eager_state1, replay[1:])
    replay_q = _compare_q(replay[0], qkv1[0])
    replay_state_ok = B0._logical_equal(replay_state, eager_state1)

    phase_results: dict[str, Any] = {}
    phase_data: dict[int, tuple[Any, Any, Any, Any]] = {}
    for phase in range(WIDTH):
        with contextlib.redirect_stdout(io.StringIO()):
            phase_cache = B0._clone_caches(prefix_cache, qwen)
            if phase:
                B0._run_step(
                    model,
                    ids[:, prefix : prefix + phase],
                    phase_cache,
                    qwen,
                    target_attn,
                    capture=False,
                )
            B0._eval_cache(phase_cache)
            phase_start = _canonicalize_pooled(
                adapter.pack(phase_cache[layer_idx]),
                target_attn.indexer,
                qwen,
                model.model.rope,
                ratio,
            )
            x_phase, qkv_phase, eager_phase = _run_eager(
                model,
                ids[:, prefix + phase : prefix + phase + WIDTH],
                phase_cache,
                qwen,
                target_attn,
                adapter,
                layer_idx,
            )
        pure_phase = compiled_step(x_phase, *phase_start.leaves)
        evaluate(*pure_phase)
        pure_phase_state = state_from(eager_phase, pure_phase[1:])
        qcheck = _compare_q(pure_phase[0], qkv_phase[0])
        state_check = B0._logical_equal(pure_phase_state, eager_phase)
        offset_mod = (prefix + phase) % WIDTH
        phase_results[str(phase)] = {
            "offset": prefix + phase,
            "offset_mod_4": offset_mod,
            "output": qcheck,
            "state": {"pass": state_check},
            "pass": bool(qcheck["pass"] and state_check),
        }
        phase_data[phase] = (x_phase, qkv_phase, phase_start, eager_phase)

    rollback_results: dict[str, Any] = {}
    for keep in (1, 3, 4):
        if keep == WIDTH:
            x_cont, qkv_cont = x2, qkv2
        else:
            x_cont, qkv_cont, _phase_start, _phase_eager = phase_data[keep]
        rollback_reference = _rollback_reference(eager_state1, prefix, keep, ratio)
        eager_cont_state = _eager_component_update(
            x_cont,
            rollback_reference,
            adapter,
            target_attn,
            model.model.rope,
            qwen,
        )
        keep_leaf = mx_module.array([keep], dtype=state0.leaves[2].dtype)
        rolled_leaves = compiled_rollback(*pure1[1:], keep_leaf)
        evaluate(*rolled_leaves)
        rolled_state = state_from(rollback_reference, rolled_leaves)
        rollback_state = _compare_state(rolled_state, rollback_reference)
        rollback_state_ok = rollback_state["pass"]
        continued = compiled_step(x_cont, *rolled_leaves)
        evaluate(*continued)
        continued_state = state_from(eager_cont_state, continued[1:])
        continued_q = _compare_q(continued[0], qkv_cont[0])
        continuation_state = _compare_state(continued_state, eager_cont_state)
        continued_state_ok = continuation_state["pass"]
        rollback_results[str(keep)] = {
            "keep": keep,
            "rollback_state": rollback_state,
            "continuation_output": continued_q,
            "continuation_state": continuation_state,
            "pass": bool(
                rollback_state_ok
                and continued_q["pass"]
                and continued_state_ok
            ),
        }

    compiled_pass = bool(
        step1_q["pass"]
        and step1_state
        and step2_q["pass"]
        and step2_state
        and replay_q["pass"]
        and replay_state_ok
    )
    phase_pass = all(item["pass"] for item in phase_results.values())
    rollback_pass = all(item["pass"] for item in rollback_results.values())
    status = "PARTIAL" if compiled_pass and phase_pass and rollback_pass else "FAIL"
    return {
        "task_id": TASK_ID,
        "status": status,
        "model": str(model_path),
        "target_layer": layer_idx,
        "target_layer_type": layer_types[layer_idx],
        "width": WIDTH,
        "prefix_tokens": prefix,
        "token_budget": budget,
        "compress_ratio": ratio,
        "fixed_capacity": capacity,
        "fixed_pool_capacity": pool_capacity,
        "state_leaves": [m.name for m in state0.metadata],
        "checks": {
            "compiled_component": {
                "reused_callable_two_offsets": True,
                "step1_output": step1_q,
                "step1_state": {"pass": step1_state},
                "step2_output": step2_q,
                "step2_state": {"pass": step2_state},
                "same_shape_replay": True,
                "replay_output": replay_q,
                "replay_state": {"pass": replay_state_ok},
                "pass": compiled_pass,
            },
            "offset_mod_4_phases": {
                "cases": phase_results,
                "pass": phase_pass,
            },
            "rollback_keep": {
                "cases": rollback_results,
                "pass": rollback_pass,
            },
        },
        "limitations": [
            "The component output is projected Q after QKV and RoPE, not full Attention output or logits.",
            "Pure QSA block top-k selection and sparse SDPA/causal-mask application are not implemented in this diagnostic boundary.",
            "Rollback preserves canonical pooled rows in the fixed leaf; native cache setters invalidate their optional pooled cache and require a production generation contract.",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ngram", default=DEFAULT_NGRAM)
    parser.add_argument("--prefix", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument(
        "--question",
        default="State-pure QSA component gate. Return a short answer.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        import mlx.core as mx_module
    except Exception as exc:
        print(
            json.dumps(
                {
                    "task_id": TASK_ID,
                    "status": "UNAVAILABLE",
                    "reason": f"MLX unavailable: {type(exc).__name__}",
                },
                ensure_ascii=False,
            )
        )
        return 2

    global mx
    mx = mx_module
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = _run_gate(args, mx_module)
    except Exception as exc:
        result = {
            "task_id": TASK_ID,
            "status": "FAIL",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "limitations": [
                "The pure component or its real-model reference failed before the gate completed; no full Attention/logits claim is made."
            ],
        }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result.get("status") == "PASS" else (2 if result.get("status") == "UNAVAILABLE" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
