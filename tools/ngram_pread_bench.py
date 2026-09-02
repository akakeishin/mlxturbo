"""n-gram サイドカーの行読み出しを mmap gather と 並列 pread で比べる。

モデルを読まずにサイドカー単体 (rows.bin) を触るだけなので biglock は要らない。
スレッド数の当たりはここで付けてから、実機 (tools/decode_profile.py) で
1 回だけ確認する方が速く回る。

行 id は 320M 行から毎回ランダムに引く。同じ行を何度も読むとページキャッシュに
乗ってどちらの経路でも速くなり、mmap のフォールト直列化という本題が見えなく
なるので、試行のたびに新しい行を引いて実ディスクを触らせる。

使い方:
  uv run python tools/ngram_pread_bench.py --sidecar ~/models/qwen38fn-ngram-4bit

--preadv: `os.pread` と `os.preadv` (buf 直書き、mlxturbo/ngram_stream.py の
`StreamNGram._use_preadv` と同じ実装) を比べる。**preadv は offset を 1 個
しか取らないので、複数行 (別々のオフセット) を 1 syscall にまとめることは
できない** (readv のベクタ化版で、1 つの連続領域を複数バッファへ分けて読む
ためのもの)。ここで比べているのは「行ごとの malloc+memcpy を 1 回省けるか」
であって「syscall 数を減らせるか」ではない (`StreamNGram` クラス docstring
に詳しい経緯がある)。
  uv run python tools/ngram_pread_bench.py --sidecar ~/models/ddalcu-ngram --preadv
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


def _gather_range_pread(fd, rec_bytes, flat, buf, pool, n_threads) -> None:
    """`mlxturbo/ngram_stream.py` の `_gather_pread` (`batch_min_rows` 以上、
    preadv 無効) と同じスライス分割。"""

    def read_range(lo, hi):
        for i in range(lo, hi):
            row_id = int(flat[i])
            buf[i] = np.frombuffer(
                os.pread(fd, rec_bytes, row_id * rec_bytes), dtype=np.uint8
            )

    n_th = min(n_threads, flat.shape[0])
    step = -(-flat.shape[0] // n_th)
    futures = [
        pool.submit(read_range, lo, min(lo + step, flat.shape[0]))
        for lo in range(0, flat.shape[0], step)
    ]
    for f in futures:
        f.result()


def _gather_range_preadv(fd, rec_bytes, flat, buf, pool, n_threads) -> None:
    """同じスライス分割で `os.preadv` (buf 直書き) を使う版。offset は行ごとに
    1 個 (= 1 syscall/行、syscall 数は pread 版と同じ)。`buf[i]` へ直接書く
    ぶん `os.pread` の bytes 確保+コピーが 1 回減る -- 詳細は
    `mlxturbo/ngram_stream.py` の `StreamNGram` docstring。"""

    def read_range(lo, hi):
        for i in range(lo, hi):
            row_id = int(flat[i])
            os.preadv(fd, [buf[i]], row_id * rec_bytes)

    n_th = min(n_threads, flat.shape[0])
    step = -(-flat.shape[0] // n_th)
    futures = [
        pool.submit(read_range, lo, min(lo + step, flat.shape[0]))
        for lo in range(0, flat.shape[0], step)
    ]
    for f in futures:
        f.result()


def bench_pread_vs_preadv(
    fd: int, rec_bytes: int, rows: int, n: int, iters: int, threads: int, rng: np.random.Generator
) -> tuple[list[float], list[float]]:
    """`os.pread` と `os.preadv` (buf 直書き) を同じスライス分割・同じスレッド数
    で interleave して比べる。呼び出しごとに引く行は毎回別 (ページキャッシュの
    温まりが片方だけに偏らないよう、条件の実行順も交互に入れ替える --
    先に読んだ側だけが後続の温まりの恩恵を受け続けると差が過大/過小に出る)。
    戻り値は (pread の ms リスト, preadv の ms リスト) -- 呼び出し側で中央値を取る。
    """
    pool = ThreadPoolExecutor(max_workers=threads)
    buf_p = np.empty((n, rec_bytes), dtype=np.uint8)
    buf_v = np.empty((n, rec_bytes), dtype=np.uint8)
    ts_pread: list[float] = []
    ts_preadv: list[float] = []
    for i in range(iters):
        pread_first = i % 2 == 0
        order = ("pread", "preadv") if pread_first else ("preadv", "pread")
        for which in order:
            ids = rng.integers(0, rows, size=n).astype(np.int64)
            t0 = time.perf_counter()
            if which == "pread":
                _gather_range_pread(fd, rec_bytes, ids, buf_p, pool, threads)
            else:
                _gather_range_preadv(fd, rec_bytes, ids, buf_v, pool, threads)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            (ts_pread if which == "pread" else ts_preadv).append(dt_ms)
    pool.shutdown()
    return ts_pread, ts_preadv


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
    ap.add_argument(
        "--preadv", action="store_true",
        help="mmap/pread の代わりに os.pread と os.preadv (buf 直書き) を"
             "interleave で比べる (StreamNGram._use_preadv と同じ実装)",
    )
    ap.add_argument(
        "--preadv-iters", type=int, default=10,
        help="--preadv 時、行数・スレッド数の組ごとに何回 interleave するか"
             "(片側 iters 回、pread/preadv 合わせて 2x iters 回の読み出し)",
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

    if args.preadv:
        if not hasattr(os, "preadv"):
            raise SystemExit("この環境に os.preadv が無い (macOS 11 未満相当)")
        import statistics

        for n in args.n_rows:
            print(f"\n=== 1 回で {n} 行 (pread vs preadv, 片側 {args.preadv_iters} 回 interleave) ===")
            for th in args.threads:
                ts_p, ts_v = bench_pread_vs_preadv(
                    fd, rec, rows, n, args.preadv_iters, th, rng
                )
                med_p, med_v = statistics.median(ts_p), statistics.median(ts_v)
                speedup = med_p / med_v if med_v else float("nan")
                print(
                    f"  x{th:<3d} pread {med_p:8.3f} ms  preadv {med_v:8.3f} ms  "
                    f"({speedup:4.2f}x, {'preadv 勝ち' if speedup > 1 else 'pread 勝ち'})"
                )
        os.close(fd)
        return

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
