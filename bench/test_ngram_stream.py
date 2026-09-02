"""StreamNGram の高速化 (バッチ pread + prefetch 行キャッシュ) がビット一致を
崩していないかを確認する。

モデルは読まない。合成サイドカー (rows=10000, dim=160, bits=4, group_size=32,
interleaved) を tmp_path に焼き、StreamNGram.__call__ の出力を
`_RefStreamNGram` (変更前の実装の写し。行ごとの pread、キャッシュ無し) と
比較する。GPU は使わない (mx.set_default_device(mx.cpu))。
"""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mlxturbo.ngram_stream import StreamNGram

ROWS = 10000
DIM = 160
BITS = 4
GROUP_SIZE = 32


def _build_synthetic_sidecar(
    tmp_path, rows: int = ROWS, dim: int = DIM, bits: int = BITS, group_size: int = GROUP_SIZE
):
    npack = dim * bits // 32
    ngrp = dim // group_size
    wb = npack * 4
    sb = ngrp * 2
    rec = wb + sb * 2
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(rows, rec), dtype=np.uint8)
    out = tmp_path / "sidecar"
    out.mkdir()
    (out / "rows.bin").write_bytes(data.tobytes())
    manifest = {
        "rows": rows,
        "rows_per_shard": rows,
        "n_shards": 1,
        "dim": dim,
        "bits": bits,
        "group_size": group_size,
        "packed_per_row": npack,
        "groups_per_row": ngrp,
        "layout": "interleaved",
        "record_bytes": rec,
        "weight_bytes": wb,
        "scale_bytes": sb,
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out, rec


class _RefStreamNGram:
    """変更前の StreamNGram (行ごとに future を 1 つ submit、キャッシュ無し)
    の写し。ビット一致を確認するための参照実装で、以降は触らない。"""

    def __init__(self, sidecar):
        self.dir = sidecar
        m = json.loads((self.dir / "manifest.json").read_text())
        self.rows = m["rows"]
        self.dim = m["dim"]
        self.bits = m["bits"]
        self.group_size = m["group_size"]
        self.npack, self.ngrp = m["packed_per_row"], m["groups_per_row"]
        self.rec = m.get("record_bytes", self.npack * 4 + self.ngrp * 4)
        self.wb = m.get("weight_bytes", self.npack * 4)
        self.sb = m.get("scale_bytes", self.ngrp * 2)
        self._fd = os.open(str(self.dir / "rows.bin"), os.O_RDONLY)
        self._pool = ThreadPoolExecutor(max_workers=4)

    def _gather_pread(self, flat: np.ndarray) -> np.ndarray:
        n = flat.shape[0]
        buf = np.empty((n, self.rec), dtype=np.uint8)
        rec_bytes = self.rec

        def read_one(i: int, row_id) -> None:
            buf[i] = np.frombuffer(
                os.pread(self._fd, rec_bytes, int(row_id) * rec_bytes), dtype=np.uint8
            )

        futures = [self._pool.submit(read_one, i, row_id) for i, row_id in enumerate(flat)]
        for f in futures:
            f.result()
        return buf

    def __call__(self, gid):
        flat = np.array(gid.reshape(-1), copy=False).astype(np.int64)
        rec = self._gather_pread(flat)
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


def _assert_bit_identical(a, b) -> None:
    # 合成サイドカーは重み/スケール/バイアスにランダムバイトを使っているため
    # bfloat16 の NaN パターンが普通に出る (NaN != NaN で mx.array_equal が
    # 使えない)。生バイト比較のほうが「ビット単位で同じ」の定義そのものなので
    # こちらだけを見る (tools/verify_ngram_pread.py と同じ流儀の後半部分)
    ab = np.array(a.view(mx.uint16), copy=False).tobytes()
    bb = np.array(b.view(mx.uint16), copy=False).tobytes()
    assert ab == bb


def test_gather_pread_batched_matches_naive(tmp_path):
    sidecar, rec = _build_synthetic_sidecar(tmp_path, rows=2000)
    stream = StreamNGram(sidecar, backend="pread", n_threads=4)
    rng = np.random.default_rng(1)
    # 64 行の閾値をまたぐ行数を含めて確認する
    for n in (1, 5, 63, 64, 65, 500, 1999):
        ids = rng.integers(0, 2000, size=n).astype(np.int64)
        got = stream._gather_pread(ids)
        want = np.empty((n, rec), dtype=np.uint8)
        for i, row in enumerate(ids):
            want[i] = np.frombuffer(
                os.pread(stream._fd, rec, int(row) * rec), dtype=np.uint8
            )
        assert got.tobytes() == want.tobytes(), n


def test_call_matches_reference_without_prefetch(tmp_path):
    sidecar, _ = _build_synthetic_sidecar(tmp_path)
    ref = _RefStreamNGram(sidecar)
    new = StreamNGram(sidecar, backend="pread", n_threads=4)
    rng = np.random.default_rng(2)
    # decode 1 ラウンド相当 (48 行) と prefill 1 チャンク相当 (数千行) の両方
    for shape in ((1, 3, 16), (1, 2000)):
        ids = rng.integers(0, ROWS, size=int(np.prod(shape))).astype(np.int64)
        gid = mx.array(ids.reshape(shape))
        _assert_bit_identical(new(gid), ref(gid))


def test_call_matches_reference_after_prefetch_completes(tmp_path):
    sidecar, _ = _build_synthetic_sidecar(tmp_path)
    ref = _RefStreamNGram(sidecar)
    new = StreamNGram(sidecar, backend="pread", n_threads=4)
    new.prefetch_enabled = True  # 既定 off (2026-09-02)。ここは先読み自体を確認する
    rng = np.random.default_rng(3)
    ids = rng.integers(0, ROWS, size=3000).astype(np.int64)
    new.prefetch(ids)
    new._prefetch_thread.join()
    assert new._cache_gen.n > 0  # 実際にキャッシュへ入った
    gid = mx.array(ids.reshape(1, -1))
    _assert_bit_identical(new(gid), ref(gid))


def test_call_matches_reference_during_concurrent_prefetch(tmp_path):
    """prefetch が背後で走っている最中 (未済/進行中/済のどれもあり得る) に
    __call__ を何度も叩いても、結果は常にビット一致する。"""
    sidecar, _ = _build_synthetic_sidecar(tmp_path)
    ref = _RefStreamNGram(sidecar)
    new = StreamNGram(sidecar, backend="pread", n_threads=4)
    new.prefetch_enabled = True  # 既定 off (2026-09-02)。ここは先読み自体を確認する
    rng = np.random.default_rng(4)

    all_ids = rng.integers(0, ROWS, size=20000).astype(np.int64)
    new.prefetch(all_ids)  # 即時に返る。バックグラウンドで取り込み中のはず

    mismatches: list[np.ndarray] = []
    lock = threading.Lock()

    def worker(seed: int) -> None:
        r = np.random.default_rng(seed)
        for _ in range(15):
            n = int(r.integers(1, 200))
            ids = r.integers(0, ROWS, size=n).astype(np.int64)
            gid = mx.array(ids.reshape(1, -1))
            got = new(gid)
            want = ref(gid)
            ok = np.array(got.view(mx.uint16), copy=False).tobytes() == np.array(
                want.view(mx.uint16), copy=False
            ).tobytes()
            if not ok:
                with lock:
                    mismatches.append(ids)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not mismatches


def test_prefetch_disabled_is_noop(tmp_path):
    sidecar, _ = _build_synthetic_sidecar(tmp_path)
    ref = _RefStreamNGram(sidecar)
    new = StreamNGram(sidecar, backend="pread", n_threads=4)
    new.prefetch_enabled = False
    rng = np.random.default_rng(5)
    ids = rng.integers(0, ROWS, size=500).astype(np.int64)
    new.prefetch(ids)
    # prefetch が無効なら行キャッシュ (既定 4M 行 = 419MB) は一度も確保しない
    # (B-7)。以前は空の生成物 (n==0) を無条件に確保していた。
    assert new._cache_gen is None
    # backend=mmap のときも常に off (StreamNGram.__init__ の契約)
    mmap_stream = StreamNGram(sidecar, backend="mmap")
    assert mmap_stream.prefetch_enabled is False

    gid = mx.array(ids.reshape(1, -1))
    _assert_bit_identical(new(gid), ref(gid))
    # 通常の __call__ 経由でもキャッシュは育たない (prefetch 無効のまま)
    assert new._cache_gen is None


def test_cache_full_clears_and_stays_correct(tmp_path):
    """容量を極小にして、1 回の __call__ の中で複数回「満杯 -> 全消し」を
    踏ませても、以降の参照が壊れないことを確認する。"""
    sidecar, _ = _build_synthetic_sidecar(tmp_path)
    ref = _RefStreamNGram(sidecar)
    new = StreamNGram(sidecar, backend="pread", n_threads=4, cache_rows=50)
    new.prefetch_enabled = True  # 既定 off (B-7 以降、行キャッシュは prefetch
    # が有効なときだけ育つ)。ここは「満杯 -> 全消し」の世代差し替えそのものを
    # 確認したいので、通常の __call__ でもキャッシュへ書き込ませる
    rng = np.random.default_rng(6)
    for _ in range(30):
        ids = rng.integers(0, ROWS, size=200).astype(np.int64)  # cap の 4 倍
        gid = mx.array(ids.reshape(1, -1))
        _assert_bit_identical(new(gid), ref(gid))
    assert new._cache_gen.n <= 50


def test_stats_counters(tmp_path):
    """発火カウンタ (calls/rows/hits/misses/prefetch_rows/prefetch_done/
    sync_ms/fetch_ms) が実際に積まれ、reset_stats() で初期化されることを
    確認する。ngram-prefetch / ngram-batch A/B の「発火してるか」の土台。"""
    sidecar, _ = _build_synthetic_sidecar(tmp_path)
    new = StreamNGram(sidecar, backend="pread", n_threads=4)
    new.prefetch_enabled = True  # 既定 off (2026-09-02)。ここは prefetch カウンタを確認する

    # 初期状態は全部 0
    s0 = new.stats
    assert s0 == dict(calls=0, rows=0, hits=0, misses=0, prefetch_rows=0,
                      prefetch_done=0, sync_ms=0.0, fetch_ms=0.0)
    assert "hits=0" in new.stats_line() and "misses=0" in new.stats_line()

    rng = np.random.default_rng(7)
    ids = rng.integers(0, ROWS, size=200).astype(np.int64)
    gid = mx.array(ids.reshape(1, -1))

    # 1 回目: キャッシュが空なので全部 miss
    new(gid)
    assert new.stats["calls"] == 1
    assert new.stats["rows"] == 200
    assert new.stats["misses"] == 200
    assert new.stats["hits"] == 0
    assert new.stats["sync_ms"] >= 0.0
    assert new.stats["fetch_ms"] >= 0.0

    # 2 回目: 同じ行なので全部キャッシュ hit (misses は増えない)
    new(gid)
    assert new.stats["calls"] == 2
    assert new.stats["rows"] == 400
    assert new.stats["hits"] == 200
    assert new.stats["misses"] == 200

    # prefetch: prefetch_rows は呼び出し即座に積まれ、prefetch_done は
    # スレッド完了後に +1 される
    more_ids = rng.integers(0, ROWS, size=64).astype(np.int64)
    new.prefetch(more_ids)
    assert new.stats["prefetch_rows"] == 64
    new._prefetch_thread.join()
    assert new.stats["prefetch_done"] == 1

    line = new.stats_line()
    assert "calls=2" in line and "rows=400" in line and "hits=200" in line

    new.reset_stats()
    assert new.stats == dict(calls=0, rows=0, hits=0, misses=0, prefetch_rows=0,
                             prefetch_done=0, sync_ms=0.0, fetch_ms=0.0)


def test_batch_min_rows_default_and_override(tmp_path):
    """`batch_min_rows` (既定 64) が `_gather_pread` の分岐閾値そのもので
    あることを確認する。ngram-batch knob は B 側でこれを `10**9` にして
    常に行ごとの旧経路を踏ませる。"""
    sidecar, rec = _build_synthetic_sidecar(tmp_path, rows=2000)
    stream = StreamNGram(sidecar, backend="pread", n_threads=4)
    assert stream.batch_min_rows == 64

    rng = np.random.default_rng(8)
    ids = rng.integers(0, 2000, size=200).astype(np.int64)
    want = np.empty((200, rec), dtype=np.uint8)
    for i, row in enumerate(ids):
        want[i] = np.frombuffer(
            os.pread(stream._fd, rec, int(row) * rec), dtype=np.uint8
        )

    # 既定 (64): 200 行はスライス分割経路を通る
    got_default = stream._gather_pread(ids)
    assert got_default.tobytes() == want.tobytes()

    # 10**9: 200 行でも常に行ごと submit の旧経路を通る (結果は同じはず)
    stream.batch_min_rows = 10**9
    got_forced_naive = stream._gather_pread(ids)
    assert got_forced_naive.tobytes() == want.tobytes()


@pytest.mark.skipif(not hasattr(os, "preadv"), reason="os.preadv が無い環境")
def test_gather_pread_preadv_matches_legacy_pread(tmp_path):
    """`_use_preadv` (os.preadv で buf に直書き) と旧経路 (os.pread +
    np.frombuffer) が、行ごと submit 経路 (< batch_min_rows) とスライス分割
    経路 (>= batch_min_rows) の両方でビット一致することを確認する。

    `StreamNGram` はこの環境 (os.preadv あり) では既定で `_use_preadv=True`
    になるはず -- それも合わせて確認する。
    """
    sidecar, rec = _build_synthetic_sidecar(tmp_path, rows=2000)
    stream = StreamNGram(sidecar, backend="pread", n_threads=4)
    assert stream._use_preadv is True  # この環境の既定 (FASTMLX_NGRAM_PREADV 未設定)

    rng = np.random.default_rng(9)
    for n in (1, 5, 63, 64, 65, 500, 1999):  # batch_min_rows=64 をまたぐ行数
        ids = rng.integers(0, 2000, size=n).astype(np.int64)

        stream._use_preadv = False
        want = stream._gather_pread(ids)

        stream._use_preadv = True
        got = stream._gather_pread(ids)

        assert got.tobytes() == want.tobytes(), n


def test_preadv_disabled_by_env_var(tmp_path, monkeypatch):
    """`FASTMLX_NGRAM_PREADV=0` で `_use_preadv` が False になり、
    `os.preadv` がこの環境に無いふりをしても (hasattr 相当) 同様に False に
    なることを確認する (どちらも旧経路へのフォールバック)。"""
    sidecar, _ = _build_synthetic_sidecar(tmp_path, rows=100)

    monkeypatch.setenv("FASTMLX_NGRAM_PREADV", "0")
    stream_off = StreamNGram(sidecar, backend="pread", n_threads=2)
    assert stream_off._use_preadv is False

    monkeypatch.delenv("FASTMLX_NGRAM_PREADV", raising=False)
    stream_default = StreamNGram(sidecar, backend="pread", n_threads=2)
    assert stream_default._use_preadv == hasattr(os, "preadv")


def test_prefetch_ngram_span_matches_real_forward(tmp_path):
    """`mlxturbo/spec_flash.py` の `_prefetch_ngram_span` (次の 1 eval 境界
    ぶんの n-gram 先読み) が計算する行 id が、実際の prefill フォワード
    (直前文脈をキャッシュ `pc[3]` から引く経路) がその境界で要求する行 id と
    ビット一致することを確認する。

    `_prefetch_ngram_span` は「次の境界の GPU 実行がまだ始まっていない
    (直前境界の分までしかキャッシュが進んでいない) 時点」に呼ぶ都合上、
    直前文脈を `pc[3]` ではなく `ids` 自身の `[start-context_len, start)`
    から直接切り出して計算する。この一致が崩れていると、先読みで温めた
    キャッシュが実フォワードの要求と一度も噛み合わず、キャッシュヒットが
    常に 0 になる (正しさ自体は壊れない -- miss してその場で同期 pread に
    落ちるだけだが、先読みの効果が丸ごと消える。2026-09-02 の 17k A/B が
    ほぼ 0% だった不具合と同じ症状)。

    合成した小さい Flash-Next (`tools/verify_batch_cache.TINY`、PLE 層が
    1 つ、ngram_size=3 -> context_len=2) を CPU で使う。実モデルは読まない。
    """
    import mlxturbo.ngram_stream as NS
    import mlxturbo.spec_flash as SF
    from verify_batch_cache import TINY, build

    model = build(8)  # indexer_budget=8 (TINY の既定の 1 つ)
    ple_layer_idx = model.model.ple_layers[0]
    ple_emb = model.model.layers[ple_layer_idx].ple.ple_embedding
    dim = ple_emb.ngram_embedding.dim

    # group_size=dim にしておけば、TINY の dim (32/heads_per_ngram) が
    # 何であっても必ず割り切れる
    sidecar, _ = _build_synthetic_sidecar(
        tmp_path, rows=200_000, dim=dim, bits=4, group_size=dim
    )
    NS.install(model, sidecar)
    stream = ple_emb.ngram_embedding
    assert isinstance(stream, NS.StreamNGram)
    stream.prefetch_enabled = True  # 既定 off。ここは先読み自体を確認する

    ctx_len = ple_emb.context_len  # TINY: ngram_size=3 -> 2
    start, length, total_len = 5, 4, 12
    assert start >= ctx_len and start + length <= total_len
    rng = np.random.default_rng(11)
    ids_np = rng.integers(0, TINY["vocab_size"], size=total_len).astype(np.int64)
    ids = mx.array(ids_np[None])

    # -- 基準: start までを実フォワードで流し、実際の pc[3] (直前文脈) から
    #    次の区間の行 id を計算する (本番の _group_prefill_forward /
    #    NGramEmbedding.__call__ と同じ経路)。
    cache = model.make_cache()
    model.model(ids[:, :start], cache=cache)
    prev_from_cache = cache[ple_layer_idx][3]
    expected_history = mx.concatenate(
        [prev_from_cache, ids[:, start : start + length]], axis=1
    )
    expected_gid = ple_emb.ngram_ids(expected_history)[:, -length:]
    mx.eval(expected_gid)
    expected_flat = np.array(expected_gid.reshape(-1), copy=False).astype(np.int64)

    # -- _prefetch_ngram_span: ids から直接切り出して計算する経路 (キャッシュは
    #    見ない)。stream.prefetch に渡された行 id をそのまま捕まえる。
    captured: dict = {}
    orig_prefetch = stream.prefetch

    def capture_prefetch(flat_ids: np.ndarray) -> None:
        captured["flat"] = np.array(flat_ids, copy=True)
        orig_prefetch(flat_ids)

    stream.prefetch = capture_prefetch
    try:
        SF._prefetch_ngram_span(model, ids, start, length)
    finally:
        stream.prefetch = orig_prefetch

    assert "flat" in captured, "stream.prefetch が呼ばれなかった"
    assert np.array_equal(captured["flat"], expected_flat)

    # 先読みしたキャッシュが実際に消費されることも合わせて確認する (行 id が
    # 一致していても、_gather_cached が引かなければ意味が無い)
    if stream._prefetch_thread is not None:
        stream._prefetch_thread.join()
    stream.reset_stats()
    got = stream(expected_gid)
    assert stream.stats["hits"] == expected_flat.size
    assert stream.stats["misses"] == 0
    mx.eval(got)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
