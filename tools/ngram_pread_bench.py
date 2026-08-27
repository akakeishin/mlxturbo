"""n-gram サイドカーの行読み出しを mmap gather と 並列 pread で比べる。

モデルを読まずにサイドカー単体 (rows.bin) を触るだけなので biglock は要らない。
スレッド数の当たりはここで付けてから、実機 (tools/decode_profile.py) で
1 回だけ確認する方が速く回る。

行 id は 320M 行から毎回ランダムに引く。同じ行を何度も読むとページキャッシュに
乗ってどちらの経路でも速くなり、mmap のフォールト直列化という本題が見えなく
なるので、試行のたびに新しい行を引いて実ディスクを触らせる。

使い方:
  uv run python tools/ngram_pread_bench.py --sidecar ~/models/qwen38fn-ngram-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def bench_mmap(mm, rows: int, n: int, iters: int, rng: np.random.Generator) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        idx = rng.integers(0, rows, size=n)
        rec = mm[idx]
    dt = time.perf_counter() - t0
    return dt / iters


def bench_pread(
    fd: int, rec_bytes: int, rows: int, n: int, iters: int, threads: int, rng: np.random.Generator
) -> float:
    pool = ThreadPoolExecutor(max_workers=threads)
    buf = np.empty((n, rec_bytes), dtype=np.uint8)

    def read_row(i: int, row_id: int) -> None:
        buf[i] = np.frombuffer(
            os.pread(fd, rec_bytes, int(row_id) * rec_bytes), dtype=np.uint8
        )

    t0 = time.perf_counter()
    for _ in range(iters):
        idx = rng.integers(0, rows, size=n)
        list(pool.map(read_row, range(n), idx))
    dt = time.perf_counter() - t0
    pool.shutdown()
    return dt / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument(
        "--n-rows", type=int, nargs="+", default=[16, 128],
        help="1 回の呼び出しで引く行数。16 は 1 トークン分、128 は S=8 の一括 forward 相当",
    )
    ap.add_argument(
        "--threads", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32],
    )
    args = ap.parse_args()

    sidecar = Path(args.sidecar).expanduser()
    manifest = json.loads((sidecar / "manifest.json").read_text())
    rows, rec = manifest["rows"], manifest["record_bytes"]
    print(f"{sidecar}  rows={rows:,}  record_bytes={rec}")

    mm = np.memmap(sidecar / "rows.bin", dtype=np.uint8, mode="r", shape=(rows, rec))
    fd = os.open(str(sidecar / "rows.bin"), os.O_RDONLY)

    # rng は使い回さずに条件ごとへ渡し切る。同じ乱数系列を条件間で使うと、
    # 先に読んだ経路がページキャッシュを温めてしまい、後で測る経路が
    # (本来はコールドのはずなのに) 不当に速く出る。320M 行から毎回新規に
    # 引けば、条件をまたいだ行の重複はほぼ起きない。
    # 種を固定しない: 固定すると再実行のたびに同じ行を引くことになり、
    # 前回の実行がページキャッシュへ残した跡をそのまま踏んでしまう
    rng = np.random.default_rng()
    for n in args.n_rows:
        print(f"\n=== 1 回で {n} 行 ({args.iters} 回平均) ===")
        t_mmap = bench_mmap(mm, rows, n, args.iters, rng)
        print(f"  {'mmap':10s} {t_mmap * 1000:7.3f} ms/call")
        for th in args.threads:
            t_pread = bench_pread(fd, rec, rows, n, args.iters, th, rng)
            speedup = t_mmap / t_pread
            print(
                f"  pread x{th:<3d} {t_pread * 1000:7.3f} ms/call  "
                f"({speedup:4.2f}x mmap 比)"
            )

    os.close(fd)


if __name__ == "__main__":
    main()
