"""GPU correctness and dependency-chain acceptance gate for Phase A2.

Run serially: ``uv run python bench/test_qmm_skinny_mma.py``.
Absolute timings are reference observations; final measurement must use a quiet machine.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from fastmlx.kernels.qmm_skinny_mma import qmm_skinny_mma


def _dtype(name):
    return {"float16": mx.float16, "bfloat16": mx.bfloat16}[name]


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
    k, n = 512, 1024
    q = _make_quantized(n, k, dtype)
    results = {}
    for m in range(6, 17):
        x = (mx.random.normal((m, k)) * 0.1).astype(dtype)
        expected = _stock(x, q)
        actual = qmm_skinny_mma(x, *q)
        mx.eval(expected, actual)
        max_abs, normalized = _normalized_max_error(actual, expected)
        results[m] = {"max_abs": max_abs, "normalized_max": normalized}
        assert normalized < 1e-3, f"M={m}: {results[m]}"
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

    def mma_op(x, q):
        return qmm_skinny_mma(x, *q)

    timings = {}
    for m in (8, 16):
        x = make_input(m)
        stock_ms = _median_ms(lambda: chain(stock_op, x))
        mma_ms = _median_ms(lambda: chain(mma_op, x))
        single_stock_ms = _median_ms(lambda: stock_op(x, up))
        single_mma_ms = _median_ms(lambda: mma_op(x, up))
        timings[m] = {
            "chain_stock_ms": stock_ms,
            "chain_mma_ms": mma_ms,
            "chain_speedup": stock_ms / mma_ms,
            "single_stock_ms": single_stock_ms,
            "single_mma_ms": single_mma_ms,
            "single_speedup": single_stock_ms / single_mma_ms,
        }

    assert timings[8]["chain_speedup"] >= 1.5, timings
    assert timings[16]["chain_mma_ms"] / timings[8]["chain_mma_ms"] <= 1.6, timings
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()
    dtype = _dtype(args.dtype)
    mx.random.seed(0)

    result = {
        "dtype": args.dtype,
        "correctness": correctness(dtype),
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
