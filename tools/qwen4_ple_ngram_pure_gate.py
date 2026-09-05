"""Real Qwen4 Flash-Next PLE + n-gram state-pure diagnostic gate.

The target is the model's single PLE layer (configuration layer id 2, Python
index 1).  Its fixed-width pure boundary is::

    F(hidden4, ids4, prev_context, conv_state, embedding4)
        -> (output, new_conv, new_context, conv_full)

The disk-backed sidecar lookup remains outside the compiled boundary, matching
the production PLE-hoist seam: it synchronizes to Python and is therefore not a
traceable MLX operation.  Its fixed ``(B,4,ple_embed_dim)`` result is an input;
all GPU PLE work and both persistent-state updates stay inside the graph.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
for _path in (REPO_ROOT, TOOLS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import qwen4_qsa_state_gate as B0

TASK_ID = "qwen4_ple_ngram_pure_gate_impl_0905"
WIDTH = 4
DEFAULT_MODEL = B0.DEFAULT_MODEL
DEFAULT_NGRAM = B0.DEFAULT_NGRAM

mx = None


def _same(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    mx.eval(left, right)
    equal = mx.all(left == right)
    mx.eval(equal)
    return bool(equal.item())


def _max_abs(left: Any, right: Any) -> float:
    if left is None or right is None or tuple(left.shape) != tuple(right.shape):
        return float("inf")
    diff = mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
    mx.eval(diff)
    return float(mx.max(diff).item())


def _pure_ple_factory(ple: Any):
    """Build a cache-free fixed-width PLE transition.

    The host-side embedding lookup is already complete when this function is
    called.  Every persistent value crossing the boundary is a tensor argument
    or return value; no cache object, offset, or context is read from Python.
    """

    import mlx.nn as nn

    key_proj = ple.key_proj
    value_proj = ple.value_proj
    norm_key = ple.norm_key
    norm_query = ple.norm_query
    norm_conv = ple.norm_conv
    conv1d = ple.conv1d
    hc = int(ple.hc)
    hidden_dim = int(ple.d)
    conv_state_len = int(ple.short_conv_state_len)
    context_len = int(ple.ple_embedding.context_len)

    def pure_step(
        hidden4: Any,
        ids4: Any,
        prev_context: Any,
        conv_state: Any,
        embedding4: Any,
    ):
        if hidden4.shape[1] != WIDTH or ids4.shape[1] != WIDTH:
            raise ValueError(f"fixed PLE gate requires width {WIDTH}")
        if embedding4.shape[1] != WIDTH:
            raise ValueError(f"fixed PLE embedding requires width {WIDTH}")
        emb = embedding4.astype(hidden4.dtype)
        key = norm_key(key_proj(emb)).reshape(-1, WIDTH, hc, hidden_dim)
        value = value_proj(emb)
        query = norm_query(hidden4).reshape(-1, WIDTH, hc, hidden_dim)
        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(hidden_dim)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(-1, WIDTH, hc * hidden_dim)
        conv_input = mx.concatenate([conv_state, norm_conv(gated)], axis=1)
        short = nn.silu(conv1d(conv_input[:, -(conv_state_len + WIDTH) :, :]))
        new_conv = mx.contiguous(conv_input[:, -conv_state_len:, :])
        new_context = mx.concatenate([prev_context, ids4], axis=1)[:, -context_len:]
        return gated + short, new_conv, new_context, conv_input

    return pure_step


def _capture_step(
    model: Any, ids: Any, caches: list[Any], qwen: Any, target: Any
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Run existing capture() and retain target PLE inputs/output."""

    import mlxturbo.spec_flash as sf

    seen: list[tuple[Any, Any, Any, Any]] = []
    output: list[Any] = []
    with sf.capture(model) as cap:
        original = qwen.PLELayer.__call__

        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            value = original(self, *args, **kwargs)
            if self is target:
                seen.append((args[0], args[1], args[2], args[3]))
                output.append(value)
            return value

        qwen.PLELayer.__call__ = wrapped
        try:
            logits = model(ids, cache=caches)
        finally:
            qwen.PLELayer.__call__ = original
    if len(seen) != 1 or len(output) != 1:
        raise RuntimeError(f"target PLE capture count {len(seen)}/{len(output)}")
    record = cap.ple.get(id(target))
    if record is None:
        raise RuntimeError("capture() did not record target PLE conv input")
    B0._eval_cache(caches, logits, *seen[0], output[0], record)
    hidden, step_ids, prev_context, _cache = seen[0]
    return hidden, step_ids, prev_context, output[0], record, cap


def _transition_check(
    pure: tuple[Any, ...], eager_output: Any, eager_cache: list[Any], layer_idx: int,
    record: Any, label: str,
) -> dict[str, Any]:
    conv_full = record
    cache = eager_cache[layer_idx]
    expected_conv, expected_context = cache[2], cache[3]
    checks = {
        "output_exact": _same(pure[0], eager_output),
        "new_conv_exact": _same(pure[1], expected_conv),
        "new_context_exact": _same(pure[2], expected_context),
        "conv_full_exact": _same(pure[3], conv_full),
    }
    return {
        "label": label,
        **checks,
        "pass": all(checks.values()),
        "max_abs_output": _max_abs(pure[0], eager_output),
        "max_abs_conv": _max_abs(pure[1], expected_conv),
    }


def _load(args: argparse.Namespace, mx_module: Any):
    model_path = Path(os.path.expanduser(args.model))
    ngram_path = Path(os.path.expanduser(args.ngram)) if args.ngram else None
    if not mx_module.metal.is_available():
        return None, {"status": "UNAVAILABLE", "reason": "Metal is unavailable"}
    if not model_path.exists():
        return None, {"status": "UNAVAILABLE", "reason": f"model path does not exist: {model_path}"}
    if ngram_path is None or not ngram_path.exists():
        return None, {"status": "UNAVAILABLE", "reason": f"n-gram sidecar path does not exist: {ngram_path}"}
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
    loaded, unavailable = _load(args, mx_module)
    if unavailable is not None:
        return {"task_id": TASK_ID, **unavailable}
    model, tokenizer, qwen, model_path = loaded
    candidates = [
        (index, layer)
        for index, layer in enumerate(model.model.layers)
        if getattr(layer, "ple", None) is not None
    ]
    if len(candidates) != 1:
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": f"expected one PLE layer, found {len(candidates)}",
        }
    layer_idx, layer = candidates[0]
    if layer_idx != 1:
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": f"unique PLE is layer {layer_idx}; expected Python layer 1",
        }
    ple = layer.ple
    if int(ple.short_conv_state_len) != 9 or int(ple.hc) * int(ple.d) != 10240:
        # The model contract is exactly (B,9,10240); do not silently generalize
        # this diagnostic to another PLE kernel shape.
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": f"unexpected PLE conv state length {ple.short_conv_state_len}",
        }
    prefix = max(1, int(args.prefix))
    ids = B0._make_ids(tokenizer, prefix + WIDTH * 2, args.question)
    if ids.shape[1] < prefix + WIDTH * 2:
        return {"task_id": TASK_ID, "status": "UNAVAILABLE", "reason": "prompt is shorter than requested"}

    import mlxturbo.spec_flash as sf

    with contextlib.redirect_stdout(io.StringIO()):
        prefix_cache = model.make_cache()
        chunk = max(1, int(args.chunk))
        for start in range(0, prefix, chunk):
            B0._run_step(
                model, ids[:, start : min(prefix, start + chunk)], prefix_cache,
                qwen, None, capture=False,
            )
        B0._eval_cache(prefix_cache)
        pre_snapshot = sf.snapshot_pre(model, prefix_cache)
        eager_cache = B0._clone_caches(prefix_cache, qwen)
        step1_ids = ids[:, prefix : prefix + WIDTH]
        step2_ids = ids[:, prefix + WIDTH : prefix + WIDTH * 2]
        h1, ids1, ctx1, y1, full1, cap1 = _capture_step(
            model, step1_ids, eager_cache, qwen, ple
        )
        post_cache1 = B0._clone_caches(eager_cache, qwen)
        h2, ids2, ctx2, y2, full2, _cap2 = _capture_step(
            model, step2_ids, eager_cache, qwen, ple
        )
        post_cache2 = B0._clone_caches(eager_cache, qwen)
        B0._eval_cache(eager_cache, h1, ids1, ctx1, y1, full1, h2, ids2, ctx2, y2, full2)

    conv0, context0 = prefix_cache[layer_idx][2], prefix_cache[layer_idx][3]
    if conv0 is None or context0 is None:
        return {"task_id": TASK_ID, "status": "FAIL", "reason": "prefix did not initialize PLE state"}
    if tuple(conv0.shape[1:]) != (9, 10240) or tuple(context0.shape[1:]) != (2,):
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": f"state shapes are conv={conv0.shape}, context={context0.shape}; expected (B,9,10240)/(B,2)",
        }
    context_input_exact = _same(ctx1, context0)
    # StreamNGram deliberately crosses to NumPy/CPU to fetch sparse disk rows.
    # Keep that existing production-hoist seam outside the graph and pass only
    # its fixed tensor result into the state-pure GPU component.
    embedding_lookup = ple.ple_embedding
    emb1 = embedding_lookup(ids1, ctx1).astype(h1.dtype)
    emb2 = embedding_lookup(ids2, ctx2).astype(h2.dtype)
    mx_module.eval(emb1, emb2)
    pure_fn = _pure_ple_factory(ple)
    try:
        compiled = mx_module.compile(pure_fn)
        pure1 = compiled(h1, ids1, ctx1, conv0, emb1)
        mx_module.eval(*pure1)
        pure2 = compiled(h2, ids2, pure1[2], pure1[1], emb2)
        mx_module.eval(*pure2)
        replay = compiled(h1, ids1, ctx1, conv0, emb1)
        mx_module.eval(*replay)
    except Exception as exc:
        return {
            "task_id": TASK_ID,
            "status": "FAIL",
            "reason": f"mx.compile/tensorized PLE execution failed: {type(exc).__name__}: {exc}",
            "host_sidecar_lookup_boundary": "ple_embedding(ids, prev_context) is precomputed through the existing PLE-hoist seam",
            "limitations": ["No eager-only PASS fallback is permitted."],
        }

    step1 = _transition_check(pure1, y1, post_cache1, layer_idx, full1, "step1")
    step2 = _transition_check(pure2, y2, post_cache2, layer_idx, full2, "step2")
    replay_checks = {
        "same_shape": all(tuple(a.shape) == tuple(b.shape) for a, b in zip(pure1, replay)),
        "all_leaves_exact": all(_same(a, b) for a, b in zip(pure1, replay)),
        "context_offset_free": True,
    }

    rollback_cases: dict[str, Any] = {}
    for keep in (1, 3, 4):
        # spec_flash.rollback uses cap.ple's full conv input and pre-snapshot's
        # context, exactly the two values represented by this boundary.
        n = int(ple.short_conv_state_len)
        planned_conv = full1[:, keep : keep + n, :]
        planned_context = mx.concatenate([context0, step1_ids[:, :keep]], axis=1)[:, -2:]
        native = B0._clone_caches(post_cache1, qwen)
        sf.rollback(
            model, native, cap1, pre_snapshot, keep, WIDTH,
            ids_kept=step1_ids[:, :keep],
        )
        B0._eval_cache(native, planned_conv, planned_context)
        rollback_state = {
            "conv_exact": _same(planned_conv, native[layer_idx][2]),
            "context_exact": _same(planned_context, native[layer_idx][3]),
            "shape_stable": tuple(planned_conv.shape) == tuple(conv0.shape)
            and tuple(planned_context.shape) == tuple(context0.shape),
        }
        try:
            continuation_ids = ids[:, prefix + keep : prefix + keep + WIDTH]
            hc, ic, cc, yc, fullc, _ = _capture_step(
                model, continuation_ids, native, qwen, ple
            )
            embc = embedding_lookup(ic, planned_context).astype(hc.dtype)
            mx_module.eval(embc)
            purec = compiled(hc, ic, planned_context, planned_conv, embc)
            mx_module.eval(*purec)
            continuation = _transition_check(
                purec, yc, native, layer_idx, fullc,
                f"rollback-{keep}-continuation",
            )
            continuation["input_context_exact"] = _same(cc, planned_context)
            continuation["pass"] = continuation["pass"] and continuation["input_context_exact"]
        except Exception as exc:
            continuation = {
                "label": f"rollback-{keep}-continuation",
                "pass": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        rollback_cases[str(keep)] = {
            **rollback_state,
            "continuation": continuation,
            "pass": all(rollback_state.values()) and continuation["pass"],
        }

    rollback_ok = all(item["pass"] for item in rollback_cases.values())
    all_ok = (
        context_input_exact
        and step1["pass"]
        and step2["pass"]
        and all(replay_checks.values())
        and rollback_ok
    )
    return {
        "task_id": TASK_ID,
        "status": "PASS" if all_ok else "FAIL",
        "model": str(model_path),
        "target_layer": layer_idx,
        "width": WIDTH,
        "prefix_tokens": prefix,
        "state_shapes": {
            "conv_state": list(conv0.shape),
            "ngram_context": list(context0.shape),
            "hoisted_embedding": list(emb1.shape),
        },
        "checks": {
            "step1": step1,
            "step2": step2,
            "prefix_context_input_exact": context_input_exact,
            "replay": {**replay_checks, "pass": all(replay_checks.values())},
            "rollback": {"cases": rollback_cases, "pass": rollback_ok},
        },
        "limitations": [
            "Only PLE layer 1 is pure; surrounding layer/model execution supplies real hidden inputs.",
            "Disk-backed n-gram row lookup stays outside mx.compile at the existing production PLE-hoist boundary.",
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
    try:
        import mlx.core as mx_module
    except Exception as exc:
        print(json.dumps({"task_id": TASK_ID, "status": "UNAVAILABLE", "reason": f"MLX import failed: {exc}"}, ensure_ascii=False, sort_keys=True))
        return 2
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = _run_gate(_parse_args(), mx_module)
    except Exception as exc:
        result = {
            "task_id": TASK_ID,
            "status": "FAIL",
            "reason": f"{type(exc).__name__}: {exc}",
            "limitations": ["Inspect the diagnostic exception; no eager-only PASS fallback is used."],
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return {"PASS": 0, "FAIL": 1, "UNAVAILABLE": 2}.get(result.get("status"), 1)


if __name__ == "__main__":
    raise SystemExit(main())
