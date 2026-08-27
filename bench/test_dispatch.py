"""GPU correctness and route-table calibration gate for Phase A3.

Run this script alone.  Timings are reference observations; the final absolute
measurement must be repeated on a quiet machine.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLXLM_FAST_QMM_WIDE", "1")  # dispatch (_load_kernels) と同条件

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx.nn as nn

from fastmlx.kernels.dispatch import (
    DEFAULT_ROUTE_TABLE,
    MMA,
    NOCAP,
    enable,
    quantized_matmul,
    select_route,
)
from fastmlx.fast_qmm import fast_qmm
from fastmlx.kernels.qmv_wide_nocap import qmv_wide_nocap


def _weights(k, n):
    # A packed affine tensor avoids materializing a multi-GB dense lm_head.
    w = mx.random.randint(0, 2**31, shape=(n, k // 8), dtype=mx.uint32)
    scales = (mx.random.uniform(shape=(n, k // 64)) * 0.02).astype(mx.bfloat16)
    biases = (mx.random.uniform(shape=(n, k // 64)) * 0.01 - 0.005).astype(
        mx.bfloat16
    )
    mx.eval(w, scales, biases)
    return w, scales, biases


def _stock(x, q):
    return mx.quantized_matmul(
        x, *q, transpose=True, group_size=64, bits=4, mode="affine"
    )


def _op(route, x, q):
    if route == "stock":
        return _stock(x, q)
    if route == NOCAP:
        return qmv_wide_nocap(x, *q, group_size=64, bits=4)
    if route == MMA:
        # dispatch の MMA 経路の実体は fast_qmm (dispatch._load_kernels と同じ)。
        # 以前はここが v5 skinny を測っていて較正列と実体がズレていた。
        return fast_qmm(x, *q, group_size=64, bits=4)
    if route == "dispatch":
        return quantized_matmul(
            x, *q, group_size=64, bits=4, mode="affine"
        )
    raise ValueError(route)


def _normalized_max(actual, expected):
    actual = actual.astype(mx.float32)
    expected = expected.astype(mx.float32)
    delta = mx.abs(actual - expected).max()
    scale = mx.maximum(mx.abs(expected).max(), mx.array(1.0))
    mx.eval(delta, scale)
    return (delta / scale).item()


def _median_ms(fn, warmup=2, reps=7):
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


def run(calibration_tolerance: float = 1.10):
    results = []
    for (k, n), row in DEFAULT_ROUTE_TABLE.items():
        q = _weights(k, n)
        for m in range(6, 17):
            x = (mx.random.normal((m, k)) * 0.05).astype(mx.bfloat16)
            route = select_route(k, n, m)
            expected = _stock(x, q)
            actual = _op("dispatch", x, q)
            mx.eval(expected, actual)
            error = _normalized_max(actual, expected)
            # MMA は B 断片を bf16 に丸めるため誤差の物理限界が ~6e-3
            # (docs/GATE-RESULTS-A2.md)。nocap/stock は bit-exact 系で 2e-3。
            threshold = 8e-3 if route == MMA else 2e-3
            assert error < threshold, (k, n, m, route, error)

            candidates = ["stock", MMA]
            if m <= 12:
                candidates.append(NOCAP)
            timings = {
                name: _median_ms(lambda name=name: _op(name, x, q))
                for name in candidates
            }
            dispatch_ms = _median_ms(lambda: _op("dispatch", x, q))
            best_name = min(timings, key=timings.get)
            within_best = dispatch_ms / timings[best_name]
            result = {
                "K": k,
                "N": n,
                "M": m,
                "route": route,
                "normalized_max": error,
                "candidate_ms": timings,
                "dispatch_ms": dispatch_ms,
                "best_candidate": best_name,
                "dispatch_over_best": within_best,
            }
            results.append(result)
            # 較正バー。汚れたマシンでは同一経路同士でも 10% を超えるノイズが出る
            # ため、正式較正 (E1 静音プロトコル) までは緩めて実行できる。
            assert within_best <= calibration_tolerance, result
    return results


def enable_integration():
    layer = nn.QuantizedLinear(
        5120,
        12288,
        bias=True,
        group_size=64,
        bits=4,
        mode="affine",
    )
    layer.set_dtype(mx.bfloat16)
    layer.eval()
    x = (mx.random.normal((1, 8, 5120)) * 0.05).astype(mx.bfloat16)
    expected = layer(x)
    layer_id = id(layer)
    enable(layer)
    actual = layer(x)
    mx.eval(expected, actual)
    error = _normalized_max(actual, expected)
    result = {
        "identity_preserved": id(layer) == layer_id,
        "class": type(layer).__name__,
        "normalized_max": error,
    }
    assert result["identity_preserved"], result
    assert result["class"] == "DispatchedQuantizedLinear", result
    # 経路表が MMA を選ぶ shape では bf16 断片丸めが乗るため上限は 8e-3
    assert error < 8e-3, result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", dest="json_out")
    parser.add_argument("--calibration-tolerance", type=float, default=1.10)
    args = parser.parse_args()
    mx.random.seed(0)
    integration = enable_integration()
    results = run(args.calibration_tolerance)
    report = {
        "enable_integration": integration,
        "rows": results,
        "timing_note": "reference only; final measurement requires a quiet machine",
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as file:
            json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
