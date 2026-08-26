"""GPU correctness, occupancy-record, and performance gates for A2 v5.

Run serially: ``uv run python bench/test_qmm_skinny_mma.py``.
Absolute timings are reference observations; final measurement needs a quiet
machine.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from fastmlx.kernels._qmm_skinny_mma_source import SPLIT_K
from fastmlx.kernels.qmm_skinny_mma import M_MAX, M_MIN, qmm_skinny_mma

BF16_CORRECTNESS_THRESHOLD = 8e-3
M8_DEPENDENCY_CHAIN_SPEEDUP = 1.5
M16_OVER_M8_LIMIT = 1.6
CORRECTNESS_SHAPES = ((512, 1024), (5120, 4096))
THREADGROUP_THREADS = 32 * SPLIT_K
GPU_QUEUE_COMMANDS = (
    "uv run python bench/test_qmm_skinny_mma.py --dtype bfloat16 "
    "--correctness-only",
    "uv run python bench/test_qmm_skinny_mma.py --dtype bfloat16 "
    "--json bench/results/qmm-skinny-mma-a2-v5.json",
    "python3 tools/isa/gen_kernels.py",
    "tools/isa/build_air.sh",
    "tools/isa/gpu_probe.sh",
    "python3 tools/isa/gpu_report.py",
)


def _dtype(name):
    return {"bfloat16": mx.bfloat16}[name]


def _make_quantized(n, k, dtype):
    dense = (mx.random.normal((n, k)) * 0.025).astype(dtype)
    w, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    mx.eval(w, scales, biases)
    return w, scales, biases


def _stock(x, q):
    return mx.quantized_matmul(
        x, *q, transpose=True, group_size=64, bits=4
    )


def _normalized_max_error(actual, expected):
    delta = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    scale = mx.maximum(mx.abs(expected.astype(mx.float32)).max(), mx.array(1.0))
    mx.eval(delta, scale)
    return delta.max().item(), (delta.max() / scale).item()


def _median_ms(fn, warmup=3, reps=12):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return sorted(samples)[len(samples) // 2]


def correctness(dtype):
    results = {}
    for k, n in CORRECTNESS_SHAPES:
        q = _make_quantized(n, k, dtype)
        shape_results = {}
        for m in range(M_MIN, M_MAX + 1):
            x = (mx.random.normal((m, k)) * 0.1).astype(dtype)
            expected = _stock(x, q)
            actual = qmm_skinny_mma(x, *q)
            mx.eval(expected, actual)
            max_abs, normalized = _normalized_max_error(actual, expected)
            shape_results[m] = {
                "max_abs": max_abs,
                "normalized_max": normalized,
            }
            assert normalized < BF16_CORRECTNESS_THRESHOLD, (
                f"K={k}, N={n}, M={m}: "
                f"threshold={BF16_CORRECTNESS_THRESHOLD}, {shape_results[m]}"
            )
        results[f"k{k}_n{n}"] = shape_results
    return results


def dependency_chain(dtype):
    k, n = 5120, 17408
    up = _make_quantized(n, k, dtype)
    down = _make_quantized(k, n, dtype)

    def make_input(m):
        return (mx.random.normal((m, k)) * 0.1).astype(dtype)

    def chain(op, x):
        h = op(x, up)
        x = op(h, down)
        h = op(x, up)
        return op(h, down)

    def stock_op(x, q):
        return _stock(x, q)

    def v5_op(x, q):
        return qmm_skinny_mma(x, *q)

    timings = {}
    for m in (8, 16):
        x = make_input(m)
        stock_ms = _median_ms(lambda: chain(stock_op, x))
        v5_ms = _median_ms(lambda: chain(v5_op, x))
        timings[m] = {
            "chain_stock_ms": stock_ms,
            "chain_v5_ms": v5_ms,
            "chain_speedup": stock_ms / v5_ms,
        }
        if m == 8:
            single_stock_ms = _median_ms(lambda: stock_op(x, up))
            single_v5_ms = _median_ms(lambda: v5_op(x, up))
            timings[m].update(
                {
                    "single_stock_ms": single_stock_ms,
                    "single_v5_ms": single_v5_ms,
                    "single_speedup": single_stock_ms / single_v5_ms,
                }
            )

    m16_over_m8 = timings[16]["chain_v5_ms"] / timings[8]["chain_v5_ms"]
    assert timings[8]["chain_speedup"] >= M8_DEPENDENCY_CHAIN_SPEEDUP, timings
    assert m16_over_m8 <= M16_OVER_M8_LIMIT, {
        "m16_over_m8": m16_over_m8,
        "limit": M16_OVER_M8_LIMIT,
        "timings": timings,
    }
    return {
        "by_m": timings,
        "m16_over_m8": m16_over_m8,
        "m8_speedup_minimum": M8_DEPENDENCY_CHAIN_SPEEDUP,
        "m16_over_m8_limit": M16_OVER_M8_LIMIT,
    }


def occupancy_record(v5_max_tptg, reference_max_tptg):
    if v5_max_tptg is None and reference_max_tptg is None:
        return {
            "status": "queued",
            "required_threads_per_threadgroup": THREADGROUP_THREADS,
            "commands": GPU_QUEUE_COMMANDS[2:],
        }
    if v5_max_tptg is None or reference_max_tptg is None:
        raise ValueError(
            "both --v5-max-tptg and --reference-max-tptg are required"
        )
    assert v5_max_tptg >= THREADGROUP_THREADS, {
        "v5_max_tptg": v5_max_tptg,
        "required": THREADGROUP_THREADS,
    }
    return {
        "status": "recorded",
        "v5_max_tptg": v5_max_tptg,
        "reference_max_tptg": reference_max_tptg,
        "v5_over_reference": v5_max_tptg / reference_max_tptg,
        "required_threads_per_threadgroup": THREADGROUP_THREADS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--json", dest="json_out")
    parser.add_argument("--v5-max-tptg", type=int)
    parser.add_argument("--reference-max-tptg", type=int)
    args = parser.parse_args()
    dtype = _dtype(args.dtype)
    mx.random.seed(0)

    result = {
        "dtype": args.dtype,
        "kernel": "A2 v5 direct-load 8x8 MMA",
        "correctness_threshold": BF16_CORRECTNESS_THRESHOLD,
        "correctness": correctness(dtype),
        "occupancy": occupancy_record(
            args.v5_max_tptg, args.reference_max_tptg
        ),
        "gpu_queue_commands": GPU_QUEUE_COMMANDS,
        "timing_note": "reference only; final measurement requires a quiet machine",
    }
    if not args.correctness_only:
        result["dependency_chain"] = dependency_chain(dtype)
    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as file:
            json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()
