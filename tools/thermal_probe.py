"""冷え具合の probe: 固定の bf16 行列積を数秒回して TFLOPS を出す。

冷却時間 (5 分か 10 分か) を決めるための減衰曲線を取る道具。ベンチの冷却窓の
0 / 2 / 5 / 10 分に差し込んで、値が冷えた基準値の ±1.5% に入る時刻を見る。
probe 自身も熱を入れるので、長さ (--seconds) と投入点は固定して使う。

  .venv/bin/python tools/thermal_probe.py --seconds 10 --tag "cool=2min" >> bench/results/thermal-probe.csv
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime

import mlx.core as mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=6144, help="正方行列の一辺 (bf16)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--header", action="store_true", help="CSV の見出し行を出す")
    args = ap.parse_args()
    if args.header:
        print("time,tag,n,calls,tflops_median,tflops_p10,tflops_p90")
        return 0
    n = args.n
    a = mx.random.normal((n, n)).astype(mx.bfloat16)
    b = mx.random.normal((n, n)).astype(mx.bfloat16)
    mx.eval(a, b)
    # 温め (コンパイルと最初の投入)
    for _ in range(3):
        mx.eval(a @ b)
    flop = 2.0 * n * n * n
    rates = []
    t_end = time.perf_counter() + args.seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        c = a @ b
        mx.eval(c)
        dt = time.perf_counter() - t0
        rates.append(flop / dt / 1e12)
    rates.sort()
    med = statistics.median(rates)
    p10 = rates[int(0.1 * (len(rates) - 1))]
    p90 = rates[int(0.9 * (len(rates) - 1))]
    print(f"{datetime.now().strftime('%H:%M:%S')},{args.tag},{n},{len(rates)},{med:.2f},{p10:.2f},{p90:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
