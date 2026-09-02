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

--mmap: `mlxturbo/ngram_stream.py` の `StreamNGram` を直に使い、まず 3 経路
(pread / preadv / mmap、先読み無し) を 1 チャンク相当の行数 (既定 32768 =
2048 トークン x 16) で温 (同じ行を 2 回目) / 冷 (別の行) 比べたあと、
`_madvise_prefetch` の丸め粒度 (`--madvise-chunk-bytes`、既定 16KB/64KB/256KB)
を振って、粒度ごとに区間数 (syscall 回数) / 発行そのものの時間 (submit) /
`--prefetch-delay` 秒待ってからの冷取得時間を出す (mmap backend の
`prefetch()` は背景スレッド化されている -- `_madvise_prefetch` を直接
同期呼び出しで測らないと、この発行コストは見えなくなる)。sysctl vm.swapusage
を前後で出す (32GB のファイルを mmap しても常駐はページキャッシュ任せで
RSS/swap を無条件には食わないはずなのを見るため)。
  uv run python tools/ngram_pread_bench.py --sidecar ~/models/ddalcu-ngram --mmap
  uv run python tools/ngram_pread_bench.py --sidecar ~/models/ddalcu-ngram --mmap \
      --madvise-chunk-bytes 16384 65536 262144 1048576
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _swapusage() -> str:
    """`sysctl vm.swapusage` の出力をそのまま返す (macOS のみ)。mmap で
    32GB のファイルを開いても常駐はページキャッシュ任せ (RSS を無条件に
    食うわけではない) はずだが、それを確かめたいので `--mmap` の前後で
    呼んで報告する。"""
    try:
        out = subprocess.run(
            ["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or out.stderr.strip()
    except Exception as e:  # pragma: no cover - 環境依存の best-effort
        return f"(sysctl 失敗: {e})"


def bench_madvise_granularity(
    mmap_s, rows: int, rec: int, n: int, prefetch_delay: float, granularities: list[int],
) -> None:
    """`_madvise_prefetch` の丸め粒度 (`chunk_bytes`) を振って、発行そのものの
    費用 (syscall 回数 x 1 回あたり 5-10us) と、粒度を上げて読み過ぎることで
    冷取得がどう変わるかを見る。行 id はハッシュでほぼ一様に散っているため、
    粒度を上げても行の集約自体はあまり進まない (むしろ 1 区間あたりの
    読み過ぎ量が増えるだけ) と予想されるが、それを実測で確かめる。

    `prefetch()` ではなく `_madvise_prefetch` を直接・同期で呼ぶ -- mmap
    backend の `prefetch()` は 2026-09-03 に背景スレッド化されたので、
    `prefetch()` 経由で測るとスレッド起動コストしか見えず、知りたい
    「madvise 発行そのものの費用」が測れなくなったため。
    """
    rng = np.random.default_rng()  # 種固定なし

    def fresh_ids() -> np.ndarray:
        return rng.integers(0, rows, size=n).astype(np.int64)

    print(
        f"\n=== madvise 粒度スイープ (chunk_bytes): 1 回で {n} 行, "
        f"待ち {prefetch_delay:.1f}s (record_bytes={rec}) ==="
    )
    print(f"{'chunk_bytes':>12s} {'区間数':>8s} {'submit(ms)':>12s} {'冷取得(ms)':>12s}")

    real_madvise = mmap_s._madvise_fn
    for chunk in granularities:
        count = 0

        def counting_madvise(option, start, length, _real=real_madvise):
            nonlocal count
            count += 1
            _real(option, start, length)

        mmap_s._madvise_fn = counting_madvise
        ids = fresh_ids()
        t0 = time.perf_counter()
        mmap_s._madvise_prefetch(ids, chunk_bytes=chunk)
        submit_ms = (time.perf_counter() - t0) * 1000.0
        mmap_s._madvise_fn = real_madvise

        time.sleep(prefetch_delay)
        t0 = time.perf_counter()
        mmap_s._gather_mmap(ids)
        cold_ms = (time.perf_counter() - t0) * 1000.0

        print(f"{chunk:>12,d} {count:>8,d} {submit_ms:>12.2f} {cold_ms:>12.2f}")


def bench_mmap_backend(
    sidecar: Path,
    rows: int,
    rec: int,
    n: int,
    threads: int,
    prefetch_delay: float,
    madvise_granularities: list[int],
) -> None:
    """StreamNGram の 3 経路 (pread / preadv / mmap、先読み無し) を、1 チャンク
    相当 (既定 32768 行) で温/冷 それぞれ比べたあと、madvise の粒度スイープ
    (`bench_madvise_granularity`) を続けて出す。

    実装をそのまま使う (再実装しない) ので、ここで出る数字は
    `mlxturbo/ngram_stream.py` の `_gather_pread` / `_gather_mmap` /
    `_madvise_prefetch` を直に呼んだもの。`StreamNGram.__call__` の
    dequantize (mx 側) は含まない -- 比べたいのは行取得 (ディスク/mmap)
    だけなので、そこを混ぜると差が見えにくくなる。

    温: 同じ行集合を 1 回捨て読みしてから 2 回目を計測 (ページキャッシュに
    乗った状態)。
    冷: 毎回新しく引いた行集合を初回だけ計測 (ページキャッシュに乗っていない
    見込みが高い状態。320M 行からランダムに引くので条件間の重複はほぼ無い)。
    """
    from mlxturbo.ngram_stream import StreamNGram

    print(f"\n=== backend 比較 (mmap): 1 回で {n} 行, pread/preadv は {threads} スレッド ===")
    print("計測前:", _swapusage())

    os.environ["MLXTURBO_NGRAM_PREFETCH"] = "1"  # madvise 先読みを有効にする
    pread_s = StreamNGram(sidecar, backend="pread", n_threads=threads)
    mmap_s = StreamNGram(sidecar, backend="mmap")
    assert mmap_s.prefetch_enabled, "MLXTURBO_NGRAM_PREFETCH=1 が効いていない"

    rng = np.random.default_rng()  # 種固定なし (前回実行のページキャッシュの跡を踏まない)

    def fresh_ids() -> np.ndarray:
        return rng.integers(0, rows, size=n).astype(np.int64)

    results: dict[str, dict[str, float]] = {}

    # --- pread (preadv 無効) ---
    pread_s._use_preadv = False
    warm_ids = fresh_ids()
    pread_s._gather_pread(warm_ids)  # 捨て読みで温める
    t0 = time.perf_counter()
    pread_s._gather_pread(warm_ids)
    warm_ms = (time.perf_counter() - t0) * 1000.0
    cold_ids = fresh_ids()
    t0 = time.perf_counter()
    pread_s._gather_pread(cold_ids)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    results["pread"] = dict(warm=warm_ms, cold=cold_ms)

    # --- preadv ---
    pread_s._use_preadv = True
    warm_ids = fresh_ids()
    pread_s._gather_pread(warm_ids)
    t0 = time.perf_counter()
    pread_s._gather_pread(warm_ids)
    warm_ms = (time.perf_counter() - t0) * 1000.0
    cold_ids = fresh_ids()
    t0 = time.perf_counter()
    pread_s._gather_pread(cold_ids)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    results["preadv"] = dict(warm=warm_ms, cold=cold_ms)

    # --- mmap (先読み無し) --- 粒度スイープの前に測る。スイープの各区間は
    # それぞれ prefetch_delay 秒の sleep を挟むので、直前の重い区間からの
    # I/O queue の持ち越しは起きにくい (それでも一番重いのはこの区間自体
    # なので、これより後に何かを測るなら間に置かないよう注意)
    warm_ids = fresh_ids()
    mmap_s._gather_mmap(warm_ids)
    t0 = time.perf_counter()
    mmap_s._gather_mmap(warm_ids)
    warm_ms = (time.perf_counter() - t0) * 1000.0
    cold_ids = fresh_ids()
    t0 = time.perf_counter()
    mmap_s._gather_mmap(cold_ids)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    results["mmap"] = dict(warm=warm_ms, cold=cold_ms)

    print(f"\n{'経路':<16s} {'温 (ms)':>10s} {'冷 (ms)':>10s}")
    for name in ("pread", "preadv", "mmap"):
        r = results[name]
        print(f"{name:<16s} {r['warm']:>10.2f} {r['cold']:>10.2f}")

    bench_madvise_granularity(mmap_s, rows, rec, n, prefetch_delay, madvise_granularities)

    pread_s.close()
    mmap_s.close()
    print("\n計測後:", _swapusage())


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
    ap.add_argument(
        "--mmap", action="store_true",
        help="StreamNGram の 3 経路 (pread / preadv / mmap、先読み無し) を "
             "1 チャンク相当の行数で温/冷 比べたあと、madvise の粒度スイープを "
             "出す (mlxturbo/ngram_stream.py の実装をそのまま使う)。--mmap-rows / "
             "--mmap-threads / --prefetch-delay / --madvise-chunk-bytes で調整できる",
    )
    ap.add_argument(
        "--mmap-rows", type=int, default=32768,
        help="--mmap 時に 1 回で読む行数。既定 32768 = 1 チャンク 2048 トークン x "
             "16 (ngram_heads) 分",
    )
    ap.add_argument(
        "--mmap-threads", type=int, default=12,
        help="--mmap 時の pread/preadv 側のスレッド数",
    )
    ap.add_argument(
        "--prefetch-delay", type=float, default=3.0,
        help="--mmap 時、madvise 粒度スイープの各条件で madvise を打ってから"
             "読むまで待つ秒数 (先読みが間に合っているかを見る)",
    )
    ap.add_argument(
        "--madvise-chunk-bytes", type=int, nargs="+", default=[16384, 65536, 262144],
        help="--mmap 時、_madvise_prefetch の丸め粒度スイープに使う chunk_bytes "
             "の候補 (既定 16KB/64KB/256KB)。大きくすると読み過ぎる代わりに"
             "区間数 (=syscall 回数) が減る",
    )
    args = ap.parse_args()

    sidecar = Path(args.sidecar).expanduser()
    manifest = json.loads((sidecar / "manifest.json").read_text())
    rows, rec = manifest["rows"], manifest["record_bytes"]
    print(f"{sidecar}  rows={rows:,}  record_bytes={rec}")

    if args.mmap:
        bench_mmap_backend(
            sidecar, rows, rec, args.mmap_rows, args.mmap_threads, args.prefetch_delay,
            args.madvise_chunk_bytes,
        )
        return

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
