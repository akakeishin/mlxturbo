"""GPU の稼働率 (ioreg の IOAccelerator PerformanceStatistics "Device Utilization %") を
一定間隔でサンプリングして CSV に落とす。root 不要、GPU を使わない。

目的: prefill 中に GPU が張り付いているか (泡が無いか) を、xctrace が使えない前提で粗く見る。
100 ms 刻みなのでチャンク境界 (2048 トークンごと、3〜4 s) の泡が 100 ms 級なら見える。

使い方:
  .venv/bin/python tools/gpu_util_sampler.py --out bench/results/gpu-util-17k.csv --interval 0.1 --duration 200
別プロセスで同時に prefill を走らせ、`MLXTURBO_PREFILL_TRACE=1` のログの時刻 (perf_counter) と
CSV の time.time() を突き合わせるには、両方の開始時刻を控える (このツールは wall clock と
perf_counter の両方を書く)。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

_RE = re.compile(r'"Device Utilization %"=(\d+)')


def sample() -> int | None:
    out = subprocess.run(
        ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True
    ).stdout
    m = _RE.search(out)
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--duration", type=float, default=300.0)
    a = ap.parse_args()
    t0 = time.time()
    p0 = time.perf_counter()
    with open(a.out, "w") as f:
        f.write(f"# start wall={t0:.3f} perf={p0:.3f} interval={a.interval}\n")
        f.write("wall,perf,util\n")
        while time.perf_counter() - p0 < a.duration:
            u = sample()
            f.write(f"{time.time():.3f},{time.perf_counter():.3f},{'' if u is None else u}\n")
            f.flush()
            time.sleep(a.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
