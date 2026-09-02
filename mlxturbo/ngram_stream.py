"""Evict the n-gram hash tables from RAM and fetch only the needed rows from disk.

Flash-Next's n-gram tables are 51.2B params: 102GB in bf16, and still 19.2GB
even crushed down to 2bit. They are the single largest block, eating a quarter
of the whole model, and yet **only 16 rows are touched per token**
(ngram_heads = (ngram_size-1) * heads_per_ngram = 16). It is a lookup, not a
matmul, so there is almost no point in keeping it resident.

Measurements (docs/STATUS.md Phase Q):
  - Crushing the n-gram tables all the way to 2bit (relative error 0.36) left
    top1 agreement unchanged
  - The error source that stands out is the experts instead
So the capacity that was being spent on the n-gram tables is better handed to
the experts. Putting them on disk brings the RAM cost to zero, and on top of
that the precision can be raised (relative error 0.081 at 4bit).

The sidecar format is raw binary, so it is easy to memmap. With safetensors you
would end up holding a single 25GB tensor, with an extra layer in the way of
row-granular extraction.

    <dir>/manifest.json   row count, dimension, bits, group_size
    <dir>/rows.bin        (rows, record_bytes) uint8

**One row is laid out contiguously as one record.** Putting weight / scales /
biases in separate files would mean touching 3 places to fetch one row, i.e.
16 rows x 3 = 48 random accesses per token. Measured, that was costing 23.9ms
per token (generation 21.3 -> 14.1 tok/s). With contiguous layout the faults
drop to a third, and since about 40 rows fit in a 4KB page you get locality too.

The 128 shards are logically just row blocks laid end to end, so they are
concatenated and held as a single flat table. That way the shard arithmetic
disappears from the lookup path.
"""

from __future__ import annotations

import atexit
import json
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .kernels import _fire

_SHARD_RE = "ngram_embedding.shard_"


# --------------------------------------------------------- building the sidecar


def _iter_shard_tensors(src: Path):
    """Yield the n-gram shards from the bf16 archive in shard_0, shard_1, ...
    order."""

    found: dict[int, tuple[Path, int, int, list[int]]] = {}
    for shard in sorted(src.glob("model-*.safetensors")):
        with open(shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        base = 8 + hlen
        for name, info in header.items():
            if _SHARD_RE not in name:
                continue
            i = int(name.split(_SHARD_RE)[1].split(".")[0])
            a, b = info["data_offsets"]
            found[i] = (shard, base + a, base + b, info["shape"])
    for i in sorted(found):
        yield i, found[i]


def build_sidecar(
    src: Path, out: Path, bits: int = 4, group_size: int = 32, layout: str = "interleaved"
) -> dict:
    """Build a quantized sidecar from the bf16 archive.

    Reading and writing happens in row blocks, so this only uses a few hundred
    MB of memory.

    layout="interleaved" lays out one row contiguously as one record. This is
    for fetching while the data stays on disk, and touches only one page per
    row.
    layout="separate" puts weight/scales/biases in separate files. This is for
    holding the table in RAM and fetching with `mx.take`, which needs the 3
    arrays as-is.
    """

    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    out.mkdir(parents=True, exist_ok=True)
    shards = list(_iter_shard_tensors(src))
    if not shards:
        raise SystemExit("n-gram のシャードが見つからない")

    dim = shards[0][1][3][-1]
    if dim % group_size:
        raise SystemExit(f"dim {dim} が group_size {group_size} で割り切れない")
    total_rows = sum(s[1][3][0] for s in shards)
    n_groups = dim // group_size
    n_packed = dim * bits // 32

    print(f"{len(shards)} シャード / {total_rows:,} 行 x {dim} 次元 -> {bits}bit")
    wb, sb = n_packed * 4, n_groups * 2
    rec_bytes = wb + sb * 2
    sep = layout == "separate"
    if sep:
        fw = open(out / "weight.bin", "wb")
        fs = open(out / "scales.bin", "wb")
        fb = open(out / "biases.bin", "wb")
    else:
        frows = open(out / "rows.bin", "wb")
    BLOCK = 1 << 16
    for i, (path, a, b, shape) in shards:
        rows, _ = shape
        mm = np.memmap(path, dtype=np.uint16, mode="r", offset=a, shape=(rows, dim))
        for lo in range(0, rows, BLOCK):
            blk = np.ascontiguousarray(mm[lo : lo + BLOCK])
            w = mx.array(blk).view(mx.bfloat16)
            q, sc, bi = mx.quantize(w, group_size=group_size, bits=bits)
            mx.eval(q, sc, bi)
            n = q.shape[0]
            if sep:
                fw.write(np.array(q, copy=False).tobytes())
                fs.write(np.array(sc.view(mx.uint16), copy=False).tobytes())
                fb.write(np.array(bi.view(mx.uint16), copy=False).tobytes())
                continue
            buf = np.empty((n, rec_bytes), dtype=np.uint8)
            buf[:, :wb] = np.array(q, copy=False).view(np.uint8).reshape(n, wb)
            buf[:, wb : wb + sb] = (
                np.array(sc.view(mx.uint16), copy=False).view(np.uint8).reshape(n, sb)
            )
            buf[:, wb + sb :] = (
                np.array(bi.view(mx.uint16), copy=False).view(np.uint8).reshape(n, sb)
            )
            frows.write(buf.tobytes())
        del mm
        if (i + 1) % 16 == 0:
            print(f"  shard {i + 1}/{len(shards)}", flush=True)
    if sep:
        fw.close(), fs.close(), fb.close()
    else:
        frows.close()

    manifest = {
        "rows": total_rows,
        "rows_per_shard": shards[0][1][3][0],
        "n_shards": len(shards),
        "dim": dim,
        "bits": bits,
        "group_size": group_size,
        "packed_per_row": n_packed,
        "groups_per_row": n_groups,
        "layout": layout,
        "record_bytes": rec_bytes,
        "weight_bytes": wb,
        "scale_bytes": sb,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    size = sum(
        (out / f).stat().st_size
        for f in (("weight.bin", "scales.bin", "biases.bin") if sep else ("rows.bin",))
    )
    print(f"wrote {out}  {size / 1e9:.1f} GB")
    return manifest


# ------------------------------------------------------------ runtime lookups


class _NGramCacheGen:
    """行キャッシュ 1 世代ぶんの入れ物 (辞書 + スロット配列)。

    満杯になったら世代ごと差し替える (このオブジェクトを作り直して
    `StreamNGram._cache_gen` に付け替える)。既存の世代を in-place で
    使い回さないのは、読み手が世代への参照を 1 回だけ取ってから読む限り、
    書き手が別スレッドで新しい行を足したり世代を丸ごと差し替えたりしても
    読み手の見ているスロットの中身が後から書き換わることが無いようにする
    ため (満杯 -> 全消し のときに起こりうる)。"""

    __slots__ = ("idx", "buf", "n")

    def __init__(self, cap: int, rec: int):
        self.idx: dict[int, int] = {}
        self.buf = np.empty((cap, rec), dtype=np.uint8)
        self.n = 0


class StreamNGram:
    """Fetch only the needed rows from the sidecar. Same calling convention as
    `_ShardedEmbedding`.

    This is not an nn.Module. It holds no parameters at all, so it must not show
    up in the model's `parameters()` (if it did, it would become a target for
    saving and dtype conversion).

    A fancy index into an mmap looks like a single gather, but what actually
    happens is a page fault per row, processed serially inside the kernel.
    Measured, fetching 16 rows (one token's worth) took 1.5-2ms, which was
    7-9% of the entire decode. ddalcu/mlx-serve (a Zig implementation) hit the
    same problem with the same design; it says "serial mmap faults were ~5ms of
    every decode step".

    `os.pread` releases the GIL, so dispatching each row on a separate thread in
    parallel makes the serialization of the faults go away. In the microbenchmark
    (tools/ngram_pread_bench.py), fetching 16 rows is 5-7x faster than mmap, and
    128 rows (equivalent to a batched forward) is 10-12x faster. The thread count
    is essentially flat between 8 and 24, and matching the performance core count
    (12 on this machine) looked like the safe choice, so that is the default.

    The mmap path is kept so it can be reverted to if this regresses. Switch with
    `FASTMLX_NGRAM_BACKEND=mmap` or `backend="mmap"`.

    `_gather_pread` の中身自体も `os.preadv` (macOS 11+) が使えるなら使う
    (既定 on)。`os.preadv(fd, buffers, offset)` は offset を 1 個しか取らない
    readv のベクタ化版で、**1 つの連続領域を複数バッファへ分けて読む**もの
    であり、別々のオフセットにある行を 1 回の呼び出しでまとめて読むことは
    できない (試すと `preadv(fd, [buf_a, buf_b], off)` は off から連続する
    2 レコード分を返すだけで、2 つ目のバッファに別の行が入ったりはしない --
    2026-09-03 に 10 行の合成ファイルで確認済み)。n-gram の行 id はハッシュ
    出力で 320M 行にほぼ一様に散らばるため、隣接ペアの期待値は
    32,768 行あたり約 3 組 (birthday paradox) しかなく、束ねる土台がほぼ無い。
    したがって「複数行を 1 syscall にまとめる」効果は無い。使っているのは
    別の効果で、`os.pread` は毎回新しい `bytes` を確保してから `buf[i]` へ
    コピーする一方、`os.preadv` は呼び出し側のバッファへ直接書けるので、
    行ごとの「malloc + memcpy」が 1 回減る (syscall 数は行数のまま変わらない)。
    CPU micro (`tools/ngram_pread_bench.py --preadv`、実サイドカー、
    pread/preadv を毎回 interleave で交互に測り、片方だけがページキャッシュの
    温まりの恩恵を受け続けないよう試行ごとに実行順も入れ替える、2026-09-03)
    では 1 チャンク相当 (32768 行、12 スレッド) で 10 回中 10 回 preadv が
    勝ち、152ms->137ms (中央値、-11%)。8/16 スレッドでも同様に勝ち
    (-12%/-3%)。decode 相当 (48 行) でも 0.50ms->0.46ms (-9%) で勝ち。
    最初にスレッド順を固定して測ったとき (interleave なし) は後に測った側
    (=先に測った側がページキャッシュを温めていた側) が過大に有利に出ていた
    ので、この数字は interleave 版のみを信用すること。ビット一致は確認済み。
    `FASTMLX_NGRAM_PREADV=0` で旧経路 (`os.pread`) に戻せる。
    `hasattr(os, "preadv")` が False の環境 (Python 3.7 未満相当) では
    自動的に旧経路。

    prefetch: prefill プロンプトの行 id は最初から全部わかっているので、
    `prefetch()` に渡すと専用のバックグラウンドスレッドが行キャッシュ
    (`_NGramCacheGen`) を埋めておく。`__call__` はキャッシュに乗っている行を
    そのまま返し、乗っていない行だけ pread する。呼び出しは即時に返り、キャッシュ
    への書き込みは `_cache_lock` で守られているので `__call__` と並走しても壊れない
    (同じ行を二重に読んでも結果は同じなので、そこは守らない)。
    `MLXTURBO_NGRAM_PREFETCH=1` で有効化できる (既定 off。backend=mmap では
    常に無効)。

    呼び出し側 (`mlxturbo/spec_flash.py` の `_prefetch_ngram_span`) は
    プロンプト全体を一度にではなく、**次の 1 eval 境界ぶんだけ**、直前境界の
    GPU 実行に重ねて `prefetch()` を呼ぶ。最初の実装 (`_prefetch_ngram_rows`、
    削除済み) は `generate_stream` のループへ入る前に `ids` 全体を一度に
    渡していて、まだどの GPU 実行も投入されていない状態でバックグラウンド
    pread が始まっていた。重ねる相手が無いので実質先出し同期待ちにしかならず、
    その上 `__call__` 側の on-demand フォールバック (`_gather_pread`) と
    背景スレッドの `_gather_pread` が同じ `self._pool` (pread 用スレッド
    プール) を取り合って競合していた。17k の in-model A/B
    (`tools/decode_ab.py --knob ngram-prefetch`) で先読みの取り分がほぼ 0%
    (-0.9%) だったのはこの 2 つが原因 (`bench/results/ple-split.json` の
    `prefetch_rows=0` は別の計測ツール (`tools/ple_split.py`、prefetch を
    そもそも有効にしない) の結果で、直接の証拠ではない — 内訳の切り分けに
    使っただけ)。2026-09-03 に呼び出し側を「次の 1 境界だけ、直前境界の
    `mx.eval` 投入直前に」呼ぶ形へ直し、on-demand 側が動く時間帯 (境界の
    graph 構築中) と背景スレッドが動く時間帯 (前の境界の GPU 実行中) が
    重ならないようにした。
    """

    def __init__(
        self,
        sidecar: Path,
        backend: str | None = None,
        n_threads: int | None = None,
        cache_rows: int | None = None,
    ):
        self.dir = Path(sidecar)
        m = json.loads((self.dir / "manifest.json").read_text())
        self.rows = m["rows"]
        self.dim = m["dim"]
        self.bits = m["bits"]
        self.group_size = m["group_size"]
        self.npack, self.ngrp = m["packed_per_row"], m["groups_per_row"]
        self.rec = m.get("record_bytes", self.npack * 4 + self.ngrp * 4)
        self.wb = m.get("weight_bytes", self.npack * 4)
        self.sb = m.get("scale_bytes", self.ngrp * 2)
        rows_bin = self.dir / "rows.bin"
        self.mm = np.memmap(rows_bin, dtype=np.uint8, mode="r", shape=(self.rows, self.rec))

        self.backend = backend or os.environ.get("FASTMLX_NGRAM_BACKEND", "pread")
        if self.backend not in ("mmap", "pread"):
            raise ValueError(f"backend は mmap/pread のどちらか ({self.backend})")
        # `_gather_pread` の「64 行未満は行ごと submit / それ以上はスライス
        # 分割」の閾値。tools/decode_ab.py の ngram-batch knob が
        # `10**9` に上げて常に行ごと経路 (旧経路) を踏ませる A/B に使う
        self.batch_min_rows = 64
        if self.backend == "pread":
            self.n_threads = n_threads or int(
                os.environ.get("FASTMLX_NGRAM_THREADS", "12")
            )
            # Do not create this per call. Creating the pool takes a few ms
            # including thread startup, and creating one per token would undo
            # the point of parallelizing
            self._pool = ThreadPoolExecutor(max_workers=self.n_threads)
            self._fd = os.open(str(rows_bin), os.O_RDONLY)
            # `_gather_pread` の読み方 (os.preadv/os.pread の選択)。クラス
            # docstring 参照。preadv が無い環境では自動的に旧経路
            self._use_preadv = hasattr(os, "preadv") and (
                os.environ.get("FASTMLX_NGRAM_PREADV", "1") != "0"
            )

            # 行キャッシュ。既定 4M 行 = 400MB (50k プロンプト = 80万行が
            # 入る)。満杯になったら世代ごと全消し (単純さ優先)。
            # ここでは確保しない -- prefetch を使わない構成 (既定) でも
            # この 419MB を無条件に確保していたのが B-7。実際に使う時点
            # (`_ensure_cache_gen`) まで遅延し、prefetch が有効にならない
            # 限り作らない。
            self._cache_cap = cache_rows or int(
                os.environ.get("FASTMLX_NGRAM_CACHE_ROWS", str(1 << 22))
            )
            self._cache_lock = threading.Lock()
            self._cache_gen: "_NGramCacheGen | None" = None

        self.prefetch_enabled = self.backend == "pread" and (
            os.environ.get("MLXTURBO_NGRAM_PREFETCH", "0") == "1"
        )
        self.reset_stats()

        # `install()` は今のこのインスタンスへの参照を呼び手に返さない (現状
        # 戻り値は無し) ので、後始末をその戻り値に頼らずここで自前に予約する。
        # ThreadPoolExecutor のワーカースレッドは非 daemon (Python の仕様。
        # `concurrent.futures.thread` が自前の atexit でジョインしようとする)
        # ため、明示的に shutdown しないままだと interpreter shutdown 時の
        # スレッド join 待ちでプロセスが残ることがある (計測ツールで 1 時間
        # 以上プロセスが残り、Metal のバッファも握ったままだった実測がある)。
        # ここで登録した close() は同モジュール import 時に登録済みの
        # `_python_exit` より後に登録されるので、atexit の LIFO 順で
        # `_python_exit` より先に呼ばれ、join 待ちが始まる前にプールを
        # shutdown できる
        self._closed = False
        atexit.register(self.close)

    def close(self) -> None:
        """pool のワーカースレッドと prefetch スレッドの後始末をする。

        多重呼び出しに耐える (atexit と __del__ の両方から呼ばれ得る)。
        interpreter shutdown 中に呼ばれる可能性があるので、内部で例外が
        出ても外には投げない。
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        t = getattr(self, "_prefetch_thread", None)
        if t is not None and t.is_alive():
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        pool = getattr(self, "_pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def reset_stats(self) -> None:
        """発火カウンタを初期化しなおす。A/B の条件ごとの頭で呼ぶ。"""
        self.stats: dict[str, float] = dict(
            calls=0, rows=0, hits=0, misses=0, prefetch_rows=0,
            prefetch_done=0, sync_ms=0.0, fetch_ms=0.0,
        )

    def stats_line(self) -> str:
        """`self.stats` を 1 行にまとめる (ログ用)。"""
        s = self.stats
        total = s["hits"] + s["misses"]
        hit_rate = s["hits"] / total * 100 if total else 0.0
        return (
            f"ngram calls={s['calls']} rows={s['rows']} hits={s['hits']} "
            f"misses={s['misses']} hit_rate={hit_rate:.1f}% "
            f"prefetch_rows={s['prefetch_rows']} prefetch_done={s['prefetch_done']} "
            f"sync_ms={s['sync_ms']:.2f} fetch_ms={s['fetch_ms']:.2f}"
        )

    def _gather_pread(self, flat: np.ndarray) -> np.ndarray:
        """Take an array of row ids and fill in the corresponding records with
        parallel preads.

        `self.batch_min_rows` (既定 64) 未満は従来どおり行ごとに future を 1 つ
        submit する (decode の 48 行はこちらの並列度が効く)。それ以上は `flat`
        を `n_threads` 個の連続スライスに割り、各 future が自分のスライスを
        ループで pread する -- 行ごとの submit/result にかかる Python 側費用
        (実測 7.6us/行、prefill 1 チャンク 32768 行で約 250ms) を消すため。
        結果は従来と完全に同じ `buf` (bit 一致)。`tools/decode_ab.py` の
        ngram-batch knob は `batch_min_rows` を `10**9` にして常にこの行ごと
        経路 (旧経路) を踏ませ、バッチ化そのものの取り分を切り分ける。
        """

        n = flat.shape[0]
        rec_bytes = self.rec
        buf = np.empty((n, rec_bytes), dtype=np.uint8)
        fd = self._fd
        use_preadv = self._use_preadv

        if n < self.batch_min_rows:
            if use_preadv:
                def read_one(i: int, row_id) -> None:
                    # os.preadv releases the GIL like os.pread, and writes
                    # straight into buf[i] instead of allocating a new bytes
                    # object first (class docstring has the measurements)
                    os.preadv(fd, [buf[i]], int(row_id) * rec_bytes)
            else:
                def read_one(i: int, row_id) -> None:
                    # os.pread releases the GIL, so the disk I/O really does run in
                    # parallel here. The destination buf[i] is disjoint per row, so
                    # there is no contention
                    buf[i] = np.frombuffer(
                        os.pread(fd, rec_bytes, int(row_id) * rec_bytes), dtype=np.uint8
                    )

            futures = [
                self._pool.submit(read_one, i, row_id) for i, row_id in enumerate(flat)
            ]
            for f in futures:
                f.result()
            return buf

        if use_preadv:
            def read_range(lo: int, hi: int) -> None:
                for i in range(lo, hi):
                    row_id = int(flat[i])
                    os.preadv(fd, [buf[i]], row_id * rec_bytes)
        else:
            def read_range(lo: int, hi: int) -> None:
                for i in range(lo, hi):
                    row_id = int(flat[i])
                    buf[i] = np.frombuffer(
                        os.pread(fd, rec_bytes, row_id * rec_bytes), dtype=np.uint8
                    )

        n_th = min(self.n_threads, n)
        step = -(-n // n_th)  # ceil div
        futures = [
            self._pool.submit(read_range, lo, min(lo + step, n))
            for lo in range(0, n, step)
        ]
        for f in futures:
            f.result()
        return buf

    def _ensure_cache_gen(self):
        """`_cache_gen` を必要になった時点で遅延生成する。

        `prefetch_enabled` が False のままなら作らない (呼び手は None を
        「キャッシュ無し」として扱う) -- 既定の prefetch 無効構成で
        419MB (4M 行) を無条件に確保していたのが B-7。ロック無しで読んで
        から要るときだけロックの下で作り直す (double-checked) ので、既に
        出来ている通常時はロックを取らない。
        """
        gen = self._cache_gen
        if gen is not None or not self.prefetch_enabled:
            return gen
        with self._cache_lock:
            gen = self._cache_gen
            if gen is None:
                gen = _NGramCacheGen(self._cache_cap, self.rec)
                self._cache_gen = gen
            return gen

    def _cache_put(self, rows: np.ndarray, data: np.ndarray) -> None:
        """`rows`/`data` (`_gather_pread` の戻り値) をキャッシュへ書き込む。

        prefetch が無効で `_cache_gen` がまだ (これからも) 無いときは何も
        しない (B-7)。それ以外はスロットの割り当てとキャッシュ辞書への
        公開をロックの下で行う。`buf[slot] = data` を辞書への登録より先に
        済ませてから登録するので、ロック無しで読む `_gather_cached` は
        「辞書にあれば buf の中身も揃っている」を常に見られる。満杯なら
        世代ごと差し替える (既存の読み手がまだ古い世代を掴んでいても、
        その世代は書き換えない)。
        """
        if self._ensure_cache_gen() is None:
            return
        with self._cache_lock:
            gen = self._cache_gen
            for i in range(rows.shape[0]):
                row = int(rows[i])
                if row in gen.idx:
                    continue
                if gen.n >= self._cache_cap:
                    gen = _NGramCacheGen(self._cache_cap, self.rec)
                    self._cache_gen = gen
                gen.buf[gen.n] = data[i]
                gen.idx[row] = gen.n
                gen.n += 1

    def _gather_cached(self, flat: np.ndarray) -> np.ndarray:
        """`flat` をキャッシュ済み/未キャッシュに分け、未キャッシュ分だけ
        `_gather_pread` する。ヒット分はキャッシュのスロット配列から集める。
        戻り値は従来 (`_gather_pread(flat)` だけ) と完全に同じ `buf`。

        キャッシュが空 (`gen is None` または `gen.n == 0`) のときは全行が
        確実に miss なので、1 行ずつ `idx.get` する意味が無い。素通しで
        `_gather_pread` に渡す (無駄な dict 参照を消す)。`gen is None`
        (prefetch 無効) のときは `_cache_put` も何もしないので、キャッシュは
        育たないまま (B-7)。
        """
        gen = self._cache_gen  # 1 回だけ読む: 以降はこの世代の idx/buf で通す
        n = flat.shape[0]
        if gen is None or gen.n == 0:
            out = self._gather_pread(flat)
            self._cache_put(flat, out)
            self.stats["misses"] += n
            _fire.bump("ngram_misses", n)
            return out
        idx, buf = gen.idx, gen.buf
        out = np.empty((n, self.rec), dtype=np.uint8)
        miss_pos: list[int] = []
        miss_rows: list[int] = []
        for i in range(n):
            row = int(flat[i])
            slot = idx.get(row)
            if slot is None:
                miss_pos.append(i)
                miss_rows.append(row)
            else:
                out[i] = buf[slot]
        if miss_rows:
            miss_arr = np.asarray(miss_rows, dtype=np.int64)
            got = self._gather_pread(miss_arr)
            for j, i in enumerate(miss_pos):
                out[i] = got[j]
            self._cache_put(miss_arr, got)
        n_miss = len(miss_rows)
        n_hit = n - n_miss
        self.stats["hits"] += n_hit
        self.stats["misses"] += n_miss
        _fire.bump("ngram_hits", n_hit)
        _fire.bump("ngram_misses", n_miss)
        return out

    def prefetch(self, flat_ids: np.ndarray) -> None:
        """`flat_ids` のうち未キャッシュの行をバックグラウンドスレッドで
        取り込む。呼び出しは即時に返る。backend=mmap または
        `prefetch_enabled=False` のときは何もしない。"""
        if not self.prefetch_enabled:
            return
        ids64 = np.asarray(flat_ids, dtype=np.int64).reshape(-1)
        if ids64.size == 0:
            return
        self.stats["prefetch_rows"] += ids64.size
        t = threading.Thread(target=self._prefetch_worker, args=(ids64,), daemon=True)
        self._prefetch_thread = t
        t.start()

    def _prefetch_worker(self, ids64: np.ndarray) -> None:
        # prefetch() は呼び出し時点で prefetch_enabled を確認済みだが、
        # ここが最初のキャッシュ利用なら `_cache_gen` はまだ無いので
        # 遅延生成する (B-7: 419MB は実際に要るときまで確保しない)。
        gen = self._ensure_cache_gen()
        if gen is None:
            self.stats["prefetch_done"] += 1
            return
        idx = gen.idx
        uniq = np.unique(ids64)
        # 事前に重複/既キャッシュ分を削る (無くても正しさは変わらないが、
        # 同じ行を何度も pread しないほうが速い)
        missing = [int(r) for r in uniq.tolist() if int(r) not in idx]
        if not missing:
            self.stats["prefetch_done"] += 1
            return
        miss_arr = np.asarray(missing, dtype=np.int64)
        got = self._gather_pread(miss_arr)
        self._cache_put(miss_arr, got)
        self.stats["prefetch_done"] += 1

    def __call__(self, gid):
        import mlx.core as mx

        t0 = time.perf_counter()
        # ここの np.array(gid.reshape(-1)) が GPU->CPU 同期そのもの
        flat = np.array(gid.reshape(-1), copy=False).astype(np.int64)
        t1 = time.perf_counter()
        if self.backend == "pread":
            rec = self._gather_cached(flat)
        else:
            # numpy's fancy index collects everything at once on the C side. A
            # per-row Python loop would show up during generation, so always
            # settle this with a single gather. Since one row is one contiguous
            # record, this also touches only one page per row
            rec = self.mm[flat]
        t2 = time.perf_counter()
        sync_ms = (t1 - t0) * 1000.0
        fetch_ms = (t2 - t1) * 1000.0
        self.stats["calls"] += 1
        self.stats["rows"] += flat.shape[0]
        self.stats["sync_ms"] += sync_ms
        self.stats["fetch_ms"] += fetch_ms
        _fire.bump("ngram_sync_ms", sync_ms)
        n = rec.shape[0]
        w = mx.array(rec[:, : self.wb].copy().view(np.uint32).reshape(n, self.npack))
        s = mx.array(
            rec[:, self.wb : self.wb + self.sb].copy().view(np.uint16).reshape(n, self.ngrp)
        ).view(mx.bfloat16)
        b = mx.array(
            rec[:, self.wb + self.sb :].copy().view(np.uint16).reshape(n, self.ngrp)
        ).view(mx.bfloat16)
        out = mx.dequantize(w, s, b, group_size=self.group_size, bits=self.bits)
        return out.reshape(*gid.shape, self.dim)


def install(model, sidecar: str | Path) -> None:
    """Swap the n-gram tables of an already-loaded model for sidecar lookups.

    At quantization/conversion time `_ShardedEmbedding` was made an empty
    parameter-free implementation, so the checkpoint contains no n-gram tensors.
    This is where the actual data gets supplied.

    サイドカーの manifest が layout=separate なら RAM 常駐 (RamNGram) に
    振り分ける。interleaved はディスク参照 (StreamNGram)。どちらを使うかは
    サイドカーを焼いた時点で決まっている、という一点に寄せる。
    """

    manifest = json.loads((Path(sidecar) / "manifest.json").read_text())
    if manifest.get("layout") == "separate":
        return install_ram(model, sidecar)

    stream = StreamNGram(Path(sidecar))
    n = 0
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        emb = ple.ple_embedding
        if emb.ngram_embedding.dim != stream.dim:
            raise ValueError(
                f"次元が合わない: モデル {emb.ngram_embedding.dim} / "
                f"サイドカー {stream.dim}"
            )
        emb.ngram_embedding = stream
        n += 1
    if n == 0:
        raise ValueError("PLE 層が見つからない")
    # The backend can also be decided by an environment variable, so always
    # print which one was taken. If FASTMLX_NGRAM_BACKEND=mmap is left in the
    # environment, fetching 16 rows becomes 5-7x slower. The output is identical,
    # so if it runs silently the only thing that shifts is the numbers we publish
    if stream.backend == "pread":
        how = f"backend=pread threads={stream.n_threads}"
    else:
        how = "backend=mmap (既定は pread。FASTMLX_NGRAM_BACKEND を確認すること)"
    print(
        f"[mlxturbo] n-gram をサイドカー参照に差し替えた "
        f"({n} 層, {stream.bits}bit, RAM 0, {how}) <- {stream.dir}"
    )


def warn_if_not_installed(model) -> bool:
    """Raise the alarm at startup when the n-gram tables have not been swapped
    for the sidecar.

    Forgetting to pass `--ngram` means `FASTMLX_NGRAM_DISK` is never set, and
    qwen4_exp's `_ShardedEmbedding` ends up allocating the tables itself. A
    checkpoint whose n-gram data was split out into a sidecar contains no n-gram
    tensors, so those tables stay at their initial values. Generation itself
    runs to completion, so if nothing says anything this slips into both
    conversations and measurements.

    Does nothing for models with no PLE layers (Llama and the like). Returns
    True if the swap has been done.
    """

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return False
    stale = total = 0
    for layer in layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        total += 1
        if not isinstance(ple.ple_embedding.ngram_embedding, (StreamNGram, RamNGram)):
            stale += 1
    if total == 0:
        return False
    if stale:
        print(
            f"[mlxturbo] 警告: n-gram 表がサイドカーに差し替わっていない "
            f"({stale}/{total} 層)。--ngram <サイドカー> を渡していないなら、"
            "出力は初期値の表で生成されたものになる"
        )
        return False
    return True


__all__ = [
    "RamNGram",
    "StreamNGram",
    "build_sidecar",
    "install",
    "install_ram",
    "warn_if_not_installed",
]


class RamNGram:
    """Hold the concatenated n-gram table in RAM and fetch with a single
    `mx.take`.

    The stock `_ShardedEmbedding` holds the 128 shards separately, so every
    lookup descends to the host via `np.unique` (= a GPU sync every token) and
    round-trips between numpy and MLX once per shard touched. Measured, that was
    costing 11-30ms per token (38.5% of the entire decode, docs/STATUS.md).

    The 128 shards are logically just row blocks laid end to end, so once
    concatenated neither the shard arithmetic nor the branching is needed. It
    comes down to 4 ops — 3 gathers plus a dequantize — and both the sync and
    the Python loop disappear. Memory usage is the same as before concatenation.
    """

    def __init__(self, sidecar: Path):
        import mlx.core as mx

        self.dir = Path(sidecar)
        m = json.loads((self.dir / "manifest.json").read_text())
        if m.get("layout") != "separate":
            raise ValueError(
                f"RAM 常駐には layout=separate のサイドカーが要る "
                f"(このサイドカーは {m.get('layout')})"
            )
        self.dim, self.bits = m["dim"], m["bits"]
        self.group_size = m["group_size"]
        rows, npack, ngrp = m["rows"], m["packed_per_row"], m["groups_per_row"]

        def load(name, dtype, cols):
            a = np.fromfile(self.dir / name, dtype=dtype).reshape(rows, cols)
            return mx.array(a)

        self.w = load("weight.bin", np.uint32, npack)
        self.s = load("scales.bin", np.uint16, ngrp).view(mx.bfloat16)
        self.b = load("biases.bin", np.uint16, ngrp).view(mx.bfloat16)
        mx.eval(self.w, self.s, self.b)

    @property
    def nbytes(self) -> int:
        return self.w.nbytes + self.s.nbytes + self.b.nbytes

    def __call__(self, gid):
        import mlx.core as mx

        flat = gid.reshape(-1)
        out = mx.dequantize(
            mx.take(self.w, flat, axis=0),
            mx.take(self.s, flat, axis=0),
            mx.take(self.b, flat, axis=0),
            group_size=self.group_size,
            bits=self.bits,
        )
        return out.reshape(*gid.shape, self.dim)


def install_ram(model, sidecar: str | Path) -> None:
    """Swap the n-gram tables for a RAM-resident concatenated table (the
    speed-first path)."""

    table = RamNGram(Path(sidecar))
    n = 0
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        if ple.ple_embedding.ngram_embedding.dim != table.dim:
            raise ValueError("次元が合わない")
        ple.ple_embedding.ngram_embedding = table
        n += 1
    if n == 0:
        raise ValueError("PLE 層が見つからない")
    gb = table.nbytes / 1e9
    print(
        f"n-gram を連結テーブルに差し替えた "
        f"({n} 層, {table.bits}bit, RAM {gb:.1f}GB, gather 1 回)"
    )
    # **長い文脈で落ちる構成なので、起動時に言う。**2026-09-02 に実際に踏んだ:
    # モデル 91GB + このテーブル 32GB = 123GB / 128GB で、50k のプロンプトが
    # `[METAL] Insufficient Memory` になる。サーバーは 200 を返してから
    # 31 秒後にストリームへエラーを流すので、**使う側は原因が分からない。**
    # interleaved のサイドカーなら同じデータをディスク参照 (RAM 0) で読み、
    # 50k が通る。17k の decode 実測でも 44.4 vs 43.8 tok/s とほとんど差が無い。
    if gb >= 8.0:
        print(
            f"n-gram: 警告 -- このサイドカーは layout=separate で RAM に "
            f"{gb:.0f}GB 常駐する。長い文脈でメモリ不足になりやすい "
            f"(91GB のモデルと合わせて 128GB 機で 50k が通らない実測がある)。"
            f"\nn-gram:   layout=interleaved のサイドカーを使うと RAM 0 で、"
            f"decode の差はほとんど無い (17k で 44.4 vs 43.8 tok/s)。"
            f"\nn-gram:   **--max-batch-spec も効かなくなる。**この常駐のせいで"
            f" rows_fit が通らず、_admit_next が落ちて全要求が単独経路に倒れる"
            f" (2026-09-02 実測: separate では joins=0、interleaved では毎回"
            f" joins=1 で 1880 トークン x 2 本が -21%)。"
        )
