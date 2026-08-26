"""B1 の per-dispatch オーバーヘッド計測。**このファイルは静音窓で回すこと。**

同じ N ステップの連鎖を 4 経路で流し、N に対する傾き (= 1 呼び出しあたりの
固定費) と切片 (= 1 submit あたりの固定費) を出す。

  mlx        : mx.fast.metal_kernel を N 回呼ぶ (現状の fastmlx)
  bridge     : N dispatch / 1 encoder / 1 command buffer  ← B1 が買いたい形
  bridge-enc : N dispatch / N encoder  / 1 command buffer
  bridge-cb  : N dispatch / N command buffer (MTLEvent で直列化)

kernel=noop なら docs/HYPOTHESES-A2.md の H2 probe (0.064 ms/call) と同じ
「中身のないカーネル」になる。ラッパ税だけを見たいときはこちら。

使い方:
    .venv/bin/python tools/bridge/bench_chain.py --kernel noop  --n 4096
    .venv/bin/python tools/bridge/bench_chain.py --kernel affine --n 262144
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chain_kernels as ck  # noqa: E402
from bridge import (  # noqa: E402
    ORDER_CB,
    SPLIT_CB,
    SPLIT_ENCODER,
    WAIT,
    Bridge,
)

TG = 256


def _fit(xs, ys):
    """最小二乗の (傾き, 切片)。"""
    k = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(v * v for v in xs)
    sxy = sum(a * b for a, b in zip(xs, ys))
    den = k * sxx - sx * sx
    if den == 0:
        return 0.0, sy / k
    slope = (k * sxy - sx * sy) / den
    return slope, (sy - slope * sx) / k


def time_mlx(x, steps, reps, noop):
    mx.eval(ck.mlx_chain(x, min(steps, 2), threadgroup=TG, noop=noop))  # warm (JIT)
    mx.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        y = ck.mlx_chain(x, steps, threadgroup=TG, noop=noop)
        mx.eval(y)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def time_bridge(br, pipe, x, y, steps, n, reps, flags):
    consts = ck.bridge_constants(n)

    def build():
        ds = [br.dispatch(pipe, [x, y], (n, 1, 1), (TG, 1, 1), constants=consts)]
        for _ in range(steps - 1):
            ds.append(br.dispatch(pipe, [y, y], (n, 1, 1), (TG, 1, 1), constants=consts))
        return ds

    ds = build()
    br.submit(ds, flags=flags)  # warm
    best = float("inf")
    best_gpu = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        br.submit(ds, flags=flags)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
        best_gpu = min(best_gpu, br.last_gpu_ms)
    return best, best_gpu


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096, help="要素数")
    ap.add_argument("--kernel", choices=["noop", "affine"], default="noop")
    ap.add_argument("--reps", type=int, default=50, help="各点の試行回数 (最小値を採る)")
    ap.add_argument(
        "--steps",
        type=str,
        default="1,2,4,8,16,32,64,128",
        help="連鎖長のスイープ",
    )
    ap.add_argument("--skip-cb", action="store_true", help="bridge-cb を省く")
    args = ap.parse_args()

    steps_list = [int(s) for s in args.steps.split(",")]
    n = args.n
    noop = args.kernel == "noop"
    fn = "chain_noop" if noop else "chain_affine"

    x = mx.arange(n, dtype=mx.float32) * 0.001
    mx.eval(x)
    y = mx.zeros((n,), dtype=mx.float32)
    mx.eval(y)

    br = Bridge.for_array(x)
    lib = br.add_library(ck.BRIDGE_MSL)
    pipe = br.pipeline(lib, fn)

    routes = [
        ("mlx", None),
        ("bridge", WAIT),
        ("bridge-enc", WAIT | SPLIT_ENCODER),
    ]
    if not args.skip_cb:
        routes.append(("bridge-cb", WAIT | SPLIT_CB | ORDER_CB))

    print(f"device={br.device_name} n={n} kernel={args.kernel} reps={args.reps}")
    print(f"{'N':>5} " + " ".join(f"{name:>12}" for name, _ in routes) + "   (ms, 最小)")

    results = {name: [] for name, _ in routes}
    for steps in steps_list:
        row = []
        for name, flags in routes:
            if name == "mlx":
                ms = time_mlx(x, steps, args.reps, noop)
            else:
                ms, _ = time_bridge(br, pipe, x, y, steps, n, args.reps, flags)
            results[name].append(ms)
            row.append(f"{ms:12.4f}")
        print(f"{steps:>5} " + " ".join(row))

    print("\n傾き = 1 dispatch あたりの固定費 / 切片 = 1 submit あたりの固定費")
    for name, _ in routes:
        slope, icept = _fit(steps_list, results[name])
        print(f"  {name:<12} {slope * 1000:8.1f} us/dispatch   {icept * 1000:8.1f} us/submit")

    base, _ = _fit(steps_list, results["mlx"])
    got, _ = _fit(steps_list, results["bridge"])
    if got > 0:
        print(f"\nラッパ税の圧縮率 (mlx 傾き / bridge 傾き): {base / got:.1f}x")
    br.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
