"""n-gram サイドカーの行取得まわりの CPU / I/O 費用を、モデル無しで測る。

測るもの (どれも実サイドカー `rows.bin` を使う):

1. `_gather_pread` の 1 チャンク (32,768 行 = 2048 トークン x 16 head) の壁時計。
   ページキャッシュ有り (通常 fd) と無し (`F_NOCACHE`) の両方。**冷の条件は
   `sudo purge` が使えないので `F_NOCACHE` (macOS の fcntl) で作る。**
2. `_gather_cached` のヒット経路 (キャッシュ全ヒット) の CPU 費用。今の実装は
   行ごとの Python ループ (`out[i] = buf[slot]`) なので、ここが無視できない
   なら先読みで I/O を隠しても CPU が残る。
3. `_cache_put` の CPU 費用 (先読みスレッドがここで GIL を握る)。
4. **GIL 泥棒の実測**: 背景スレッドが `_gather_pread` + `_cache_put` を回して
   いる間、主スレッドの純 Python 仕事 (= prefill のグラフ構築の代役) が
   どれだけ遅くなるか。先読みが「重なる」ためにはここが軽い必要がある。

使い方 (biglock 経由。モデルを読まないので prio 2):
    tools/biglock.sh .venv/bin/python tools/ngram_fetch_micro.py \\
        --sidecar ~/models/ddalcu-ngram --out bench/results/ngram-fetch-micro.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

F_NOCACHE = 48  # <sys/fcntl.h> (macOS)


def _med(f, reps):
    xs = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        xs.append((time.perf_counter() - t) * 1e3)
    return statistics.median(xs), xs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", default="~/models/ddalcu-ngram")
    ap.add_argument("--rows", type=int, default=32768, help="1 チャンク相当")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlxturbo.ngram_stream import StreamNGram, _NGramCacheGen

    sc = Path(os.path.expanduser(args.sidecar))
    st = StreamNGram(sc)
    assert st.backend == "pread", st.backend
    print(f"sidecar={sc} rows={st.rows:,} rec={st.rec}B threads={st.n_threads} "
          f"preadv={st._use_preadv}", flush=True)

    rng = np.random.default_rng(args.seed)
    res: dict = {"sidecar": str(sc), "rows_per_call": args.rows,
                 "threads": st.n_threads, "rec_bytes": st.rec}

    # --- 1. fetch: 温 (通常 fd) / 冷 (F_NOCACHE) -----------------------------
    # 毎回ちがう行 id を使う (同じ行を測り直すとページキャッシュに乗る)
    def fetch_run(tag, nocache):
        old_fd = st._fd
        fd = os.open(str(sc / "rows.bin"), os.O_RDONLY)
        if nocache:
            import fcntl
            fcntl.fcntl(fd, F_NOCACHE, 1)
        st._fd = fd
        try:
            xs = []
            for _ in range(args.reps):
                ids = rng.integers(0, st.rows, args.rows, dtype=np.int64)
                t = time.perf_counter()
                st._gather_pread(ids)
                xs.append((time.perf_counter() - t) * 1e3)
        finally:
            st._fd = old_fd
            os.close(fd)
        m = statistics.median(xs)
        print(f"  fetch {tag}: median {m:.1f} ms ({m * 1e3 / args.rows:.2f} us/行)"
              f"  all={[round(x) for x in xs]}", flush=True)
        return {"median_ms": m, "us_per_row": m * 1e3 / args.rows, "all_ms": xs}

    print("1. _gather_pread (1 チャンク相当)", flush=True)
    res["fetch_pagecache"] = fetch_run("ページキャッシュ有", False)
    res["fetch_nocache"] = fetch_run("F_NOCACHE", True)
    # 同じ行をもう一度 (完全に温い場合の下限)
    ids_warm = rng.integers(0, st.rows, args.rows, dtype=np.int64)
    st._gather_pread(ids_warm)
    m, xs = _med(lambda: st._gather_pread(ids_warm), args.reps)
    print(f"  fetch 同じ行の再読み: median {m:.1f} ms "
          f"({m * 1e3 / args.rows:.2f} us/行)", flush=True)
    res["fetch_same_rows"] = {"median_ms": m, "us_per_row": m * 1e3 / args.rows}

    # --- 2/3. キャッシュのヒット経路と _cache_put ---------------------------
    print("2/3. 行キャッシュの CPU 費用", flush=True)
    st.prefetch_enabled = True
    st._cache_gen = _NGramCacheGen(st._cache_cap, st.rec)
    ids_c = rng.integers(0, st.rows, args.rows, dtype=np.int64)
    data = st._gather_pread(ids_c)
    m_put, _ = _med(lambda: st._cache_put(ids_c, data), args.reps)  # 2 回目以降は全部既存
    # 初回 (実際に書く) を別に測る
    st._cache_gen = _NGramCacheGen(st._cache_cap, st.rec)
    t = time.perf_counter()
    st._cache_put(ids_c, data)
    put_first = (time.perf_counter() - t) * 1e3
    m_hit, xs_hit = _med(lambda: st._gather_cached(ids_c), args.reps)
    print(f"  _cache_put 初回 (全部新規): {put_first:.1f} ms", flush=True)
    print(f"  _cache_put 2 回目以降 (全部既存): {m_put:.1f} ms", flush=True)
    print(f"  _gather_cached 全ヒット: median {m_hit:.1f} ms "
          f"({m_hit * 1e3 / args.rows:.2f} us/行)  all={[round(x, 1) for x in xs_hit]}",
          flush=True)
    res["cache_put_first_ms"] = put_first
    res["cache_put_existing_ms"] = m_put
    res["gather_cached_hit"] = {"median_ms": m_hit,
                                "us_per_row": m_hit * 1e3 / args.rows}

    # --- 4. GIL 泥棒 --------------------------------------------------------
    # 主スレッドの純 Python 仕事 (グラフ構築の代役) の速度を、背景で先読みが
    # 走っているときと走っていないときで比べる
    print("4. 背景先読み中の主スレッド (GIL) の減速", flush=True)

    def busy(n=400_000):
        s = 0
        for i in range(n):
            s += i * 3 % 7
        return s

    base, _ = _med(busy, 5)
    stop = threading.Event()
    bg_ids = [rng.integers(0, st.rows, args.rows, dtype=np.int64) for _ in range(40)]

    def bg():
        i = 0
        while not stop.is_set() and i < len(bg_ids):
            st._prefetch_worker(bg_ids[i])
            i += 1

    st._cache_gen = _NGramCacheGen(st._cache_cap, st.rec)
    th = threading.Thread(target=bg, daemon=True)
    th.start()
    time.sleep(0.05)
    with_bg, _ = _med(busy, 5)
    stop.set()
    th.join(timeout=30)
    print(f"  主スレッドの Python 仕事: 素 {base:.1f} ms -> 先読み中 {with_bg:.1f} ms "
          f"({with_bg / base:.2f}x)", flush=True)
    res["gil_main_alone_ms"] = base
    res["gil_main_with_prefetch_ms"] = with_bg
    res["gil_slowdown"] = with_bg / base

    st.close()
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
