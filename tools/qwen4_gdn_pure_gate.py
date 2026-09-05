"""Real Qwen4 Flash-Next GDN state-pure diagnostic gate.

This is deliberately a diagnostic, not a production graphbank.  It selects the
first PLE-free ``linear_attention`` layer (the expected layer 0) and checks a
fixed-width pure boundary::

    F(x4, conv_state, recurrent_state)
        -> (output, new_conv, new_state, conv_input, states_all)

The callable has no cache object or Python logical offset.  The caller supplies
the two fixed-shape state tensors and the same compiled callable is exercised
twice, replayed, and compared with the existing eager capture/rollback path.
stdout is exactly one JSON line; PASS=0, FAIL=1, unavailable=2.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
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

TASK_ID = "qwen4_gdn_pure_gate_impl_0905"
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


def _pure_gdn_factory(gdn: Any):
    """Make the only state-pure boundary in this file.

    All dimensions below are immutable model metadata.  No cache, offset, or
    history is read from Python; the only carried state is the two tensor
    arguments.  ``x4.shape[1]`` is checked against the fixed graph width before
    tracing and therefore cannot become a dynamic logical length.
    """

    from mlxturbo.kernels import gdn_prework as gp
    from mlxturbo.kernels.gated_delta_states import gated_delta_update_with_states_gb

    project_in = gdn._project_in
    conv_w = gdn.conv1d.weight
    A_log = gdn.A_log
    dt_bias = gdn.dt_bias
    norm = gdn.norm
    out_proj = gdn.out_proj
    n_k, n_v = int(gdn.n_k), int(gdn.n_v)
    dk, dv = int(gdn.dk), int(gdn.dv)
    key_dim, value_dim = int(gdn.key_dim), int(gdn.value_dim)
    conv_kernel_size = int(gdn.conv_kernel_size)
    conv_dim = int(gdn.conv_dim)

    def pure_step(x4: Any, conv_state: Any, recurrent_state: Any):
        if x4.shape[1] != WIDTH:
            raise ValueError(f"fixed GDN gate requires width {WIDTH}, got {x4.shape[1]}")
        mixed_qkv, z, b, a = project_in(x4)
        batch = x4.shape[0]
        z = z.reshape(batch, WIDTH, n_v, dv)
        q, k, v, gate, beta, new_conv = gp.fused_gdn_prework(
            mixed_qkv,
            conv_state,
            conv_w,
            a,
            b,
            A_log,
            dt_bias,
            n_k,
            n_v,
            dk,
            dv,
            key_dim,
            value_dim,
        )
        out, states_all = gated_delta_update_with_states_gb(
            q, k, v, gate, beta, recurrent_state, None
        )
        new_state = states_all[:, -1]
        output = out_proj(norm(out, z).reshape(batch, WIDTH, value_dim))
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        # Keep these shape assertions outside the graph's logical state.  They
        # are fixed model contracts, and make a silent kernel shape change fail.
        if conv_input.shape[1] != conv_kernel_size - 1 + WIDTH:
            raise ValueError("unexpected GDN conv input length")
        if conv_input.shape[2] != conv_dim:
            raise ValueError("unexpected GDN conv input width")
        return output, new_conv, new_state, conv_input, states_all

    return pure_step


def _capture_step(
    model: Any, ids: Any, caches: list[Any], qwen: Any, target: Any
) -> tuple[Any, Any, Any, Any, Any]:
    """Run existing capture() and additionally retain target GDN input/output."""

    import mlxturbo.spec_flash as sf

    seen_x: list[Any] = []
    seen_y: list[Any] = []
    with sf.capture(model) as cap:
        # Production decode fusion replaces each GDN instance with a dynamic
        # subclass.  Patch the actual class, not the vendor base class, so the
        # observed x/y are from the same fused path used by capture().
        target_type = type(target)
        original = target_type.__call__

        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            value = original(self, *args, **kwargs)
            if self is target:
                seen_x.append(args[0])
                seen_y.append(value)
            return value

        target_type.__call__ = wrapped
        try:
            logits = model(ids, cache=caches)
        finally:
            target_type.__call__ = original
    if len(seen_x) != 1 or len(seen_y) != 1:
        raise RuntimeError(f"target GDN capture count {len(seen_x)}/{len(seen_y)}")
    record = cap.gdn.get(id(target))
    if record is None:
        raise RuntimeError("capture() did not record target GDN state")
    B0._eval_cache(caches, logits, seen_x[0], seen_y[0], *record)
    return seen_x[0], seen_y[0], record, logits, cap


def _transition_check(
    pure: tuple[Any, ...], eager_output: Any, eager_cache: list[Any], layer_idx: int,
    record: tuple[Any, Any], label: str,
) -> dict[str, Any]:
    conv_input, states_all = record
    cache = eager_cache[layer_idx]
    expected_conv, expected_state = cache[0], cache[1]
    checks = {
        "output_exact": _same(pure[0], eager_output),
        "new_conv_exact": _same(pure[1], expected_conv),
        "new_state_exact": _same(pure[2], expected_state),
        "conv_input_exact": _same(pure[3], conv_input),
        "states_all_exact": _same(pure[4], states_all),
    }
    return {
        "label": label,
        **checks,
        "pass": all(checks.values()),
        "max_abs_output": _max_abs(pure[0], eager_output),
        "max_abs_state": _max_abs(pure[2], expected_state),
    }


def _rollback_plan(record: tuple[Any, Any], keep: int, width: int) -> tuple[Any, Any]:
    conv_input, states_all = record
    kernel_minus_one = conv_input.shape[1] - width
    committed_conv_input = conv_input[:, : kernel_minus_one + keep]
    committed_conv = committed_conv_input[:, -kernel_minus_one:]
    committed_state = states_all[:, keep - 1]
    return committed_conv, committed_state


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
    from mlxturbo import fused
    from mlxturbo.ngram_stream import install

    with contextlib.redirect_stdout(io.StringIO()):
        model, tokenizer = load(str(model_path))
        install(model, str(ngram_path))
        # This follows the production default (mode 1 unless explicitly set to
        # off) and only sets existing class flags; it adds no production seam.
        fused.enable_gdn_decode_fused(model)
    return (model, tokenizer, qwen, model_path), None


def _run_gate(args: argparse.Namespace, mx_module: Any) -> dict[str, Any]:
    global mx
    mx = mx_module
    B0.mx = mx_module
    loaded, unavailable = _load(args, mx_module)
    if unavailable is not None:
        return {"task_id": TASK_ID, **unavailable}
    model, tokenizer, qwen, model_path = loaded
    layers = list(model.model.layers)
    candidates = [
        (index, layer)
        for index, layer in enumerate(layers)
        if getattr(layer, "linear_attn", None) is not None
        and getattr(layer, "ple", None) is None
    ]
    if not candidates:
        return {"task_id": TASK_ID, "status": "UNAVAILABLE", "reason": "no PLE-free GDN layer"}
    layer_idx, layer = candidates[0]
    if layer_idx != 0:
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": f"first PLE-free GDN is layer {layer_idx}, expected layer 0",
        }
    gdn = layer.linear_attn
    if not getattr(gdn, "_gdn_prework", False):
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": "production GDN prework is disabled (set MLXTURBO_GDN_DECODE_FUSED=1)",
        }
    if int(gdn.conv_kernel_size) < 2:
        return {"task_id": TASK_ID, "status": "UNAVAILABLE", "reason": "invalid GDN conv kernel size"}
    from mlxturbo.kernels import gdn_prework as gp
    import mlxturbo.spec_flash as sf

    prefix = max(1, int(args.prefix))
    ids = B0._make_ids(tokenizer, prefix + WIDTH * 2, args.question)
    if ids.shape[1] < prefix + WIDTH * 2:
        return {"task_id": TASK_ID, "status": "UNAVAILABLE", "reason": "prompt is shorter than requested"}

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
        x1, eager_y1, cap1_record, _, cap1 = _capture_step(
            model, step1_ids, eager_cache, qwen, gdn
        )
        post_cache1 = B0._clone_caches(eager_cache, qwen)
        x2, eager_y2, cap2_record, _, _cap2 = _capture_step(
            model, step2_ids, eager_cache, qwen, gdn
        )
        post_cache2 = B0._clone_caches(eager_cache, qwen)
        B0._eval_cache(eager_cache, x1, eager_y1, x2, eager_y2, *cap1_record, *cap2_record)

    conv0, state0 = prefix_cache[layer_idx][0], prefix_cache[layer_idx][1]
    if conv0 is None or state0 is None:
        return {"task_id": TASK_ID, "status": "FAIL", "reason": "prefix did not initialize GDN state"}
    mixed, _z, b, a = gdn._project_in(x1)
    if not gp.eligible(
        mixed, conv0, gdn.conv1d.weight, a, b, gdn.A_log, gdn.dt_bias,
        gdn.n_k, gdn.n_v, gdn.dk, gdn.key_dim, gdn.value_dim,
    ):
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": "fixed width/state does not satisfy production fused GDN prework eligibility",
        }

    pure_fn = _pure_gdn_factory(gdn)
    try:
        compiled = mx_module.compile(pure_fn)
        pure1 = compiled(x1, conv0, state0)
        mx_module.eval(*pure1)
        pure2 = compiled(x2, pure1[1], pure1[2])
        mx_module.eval(*pure2)
        replay = compiled(x1, conv0, state0)
        mx_module.eval(*replay)
    except Exception as exc:
        return {
            "task_id": TASK_ID,
            "status": "FAIL",
            "reason": f"mx.compile/pure GDN execution failed: {type(exc).__name__}: {exc}",
            "limitations": ["No eager-only PASS fallback is permitted."],
        }

    step1 = _transition_check(pure1, eager_y1, post_cache1, layer_idx, cap1_record, "step1")
    step2 = _transition_check(pure2, eager_y2, post_cache2, layer_idx, cap2_record, "step2")
    replay_checks = {
        "same_shape": all(tuple(a.shape) == tuple(b.shape) for a, b in zip(pure1, replay)),
        "all_leaves_exact": all(_same(a, b) for a, b in zip(pure1, replay)),
        "offset_free_boundary": True,
    }

    rollback_cases = {}
    for keep in (1, 3, 4):
        planned_conv, planned_state = _rollback_plan(cap1_record, keep, WIDTH)
        native = B0._clone_caches(post_cache1, qwen)
        sf.rollback(
            model,
            native,
            cap1,
            pre_snapshot,
            keep,
            WIDTH,
            ids_kept=step1_ids[:, :keep],
        )
        B0._eval_cache(native, planned_conv, planned_state)
        native_conv = native[layer_idx][0]
        native_state = native[layer_idx][1]
        rollback_state = {
            "conv_exact": _same(planned_conv, native_conv),
            "state_exact": _same(planned_state, native_state),
            "shape_stable": tuple(planned_conv.shape) == tuple(conv0.shape)
            and tuple(planned_state.shape) == tuple(state0.shape),
        }
        try:
            continuation_ids = ids[:, prefix + keep : prefix + keep + WIDTH]
            x_cont, eager_y_cont, cap_cont_record, _, _ = _capture_step(
                model, continuation_ids, native, qwen, gdn
            )
            pure_cont = compiled(x_cont, planned_conv, planned_state)
            mx_module.eval(*pure_cont)
            continuation = _transition_check(
                pure_cont,
                eager_y_cont,
                native,
                layer_idx,
                cap_cont_record,
                f"rollback-{keep}-continuation",
            )
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
        step1["pass"]
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
        "fixed_state_shapes": {
            "conv_state": list(conv0.shape),
            "recurrent_state": list(state0.shape),
        },
        "checks": {
            "step1": step1,
            "step2": step2,
            "replay": {**replay_checks, "pass": all(replay_checks.values())},
            "rollback": {
                "cases": rollback_cases,
                "pass": rollback_ok,
            },
        },
        "limitations": [
            "Only the selected layer is made state-pure; the surrounding model is used as the eager source of real layer inputs.",
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
