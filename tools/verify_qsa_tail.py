"""QSA の可視集合を HF 参照の規則 (numpy) と突き合わせる (CPU、GPU 不要)。

## 何を確かめるか

HF transformers main の `models/qwen4_exp/modeling_qwen4_exp.py`
(`Qwen4ExpTextQSAIndexer.forward`) は、クエリ ``q`` の可視集合をこう決める:

1. ``visible = 0..q`` (causal)
2. ``num_complete_blocks = (q+1) // cr``、ブロック ``i`` は列 ``cr*i .. cr*i+cr-1``
3. スコア上位 ``min(block_topk, num_complete_blocks)`` 個のブロックの列
4. **tail** = ``visible[num_complete_blocks*cr:]`` = 列 ``[cr*floor((q+1)/cr), q]``
   (自分自身を含む 0〜cr-1 列)。**クエリごと**

このツールは 3 の「どのブロックを選ぶか」は検証しない (スコアの再現は
別の話)。``QSAIndexer._pooled_and_top`` が返した ``keep_block`` を
**そのまま参照側にも食わせて**、2 の候補判定と 4 の tail 展開だけを比べる。
つまり見ているのは「ブロック選択をトークン幅へ展開する規則」 ---
`MLXTURBO_QSA_TAIL` が変えるのはそこだけ。

比べる相手は ``QSAIndexer.__call__`` の返り値を ``Attention._final_mask``
に通したもの (``mask=None`` なので「sparse があれば causal を捨てて sparse
だけを見る」規約どおり sparse そのもの)。これが実際に sdpa へ渡る集合。

## 期待する結果

- ``MLXTURBO_QSA_TAIL=query``: 全部の (S, kv) で 100% 一致
- ``MLXTURBO_QSA_TAIL=global``: 不一致の**セル数を予言して**突き合わせる。
  行 ``q`` の global tail は ``[cr*n_blocks, q]``、query tail は ``[cr*floor((q+1)/cr), q]``
  なので、

  * ``q >= cr*n_blocks`` (端数域にいる行、S=1 の decode はここ) では両者が一致
  * それ以外の行は global が **``q % cr + 1`` 列** (自分自身を含む) を落とす。
    ``q % cr == cr-1`` の行だけは query 側も空なので落とさない

  つまり S=1 は常に一致、prefill 幅では 3/4 の行が自分自身を見ていない。
  この予言と実測がずれたら NG。

## 使い方

    .venv/bin/python tools/verify_qsa_tail.py

合成の重みと合成の隠れ状態でだけ動く (実モデルは要らない)。既定は
S ∈ {1, 2, 3, 2048} x kv ∈ {2049, 4096, 8191, 17000}。
"""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx
import numpy as np

import mlxturbo  # noqa: F401  (meta_path フックを入れる)
from mlxturbo import qsa_tail as QT

import mlx_lm.models.qwen4_exp as Q


CR = 4
BUDGET = 2048


def _make_indexer(hidden: int = 64, head_dim: int = 32, n_heads: int = 2):
    args = Q.TextArgs(
        hidden_size=hidden,
        indexer_n_heads=n_heads,
        indexer_kv_heads=1,
        indexer_head_dim=head_dim,
        indexer_budget=BUDGET,
        indexer_compress_ratio=CR,
        rms_norm_eps=1e-6,
    )
    idx = Q.QSAIndexer(args)
    mx.eval(idx.parameters())
    rope = Q.RotaryEmbedding(dim=head_dim // 2, base=10000.0)
    return idx, rope, hidden


def _run(idx, rope, hidden: int, S: int, kv: int, seed: int):
    """(keep_block, keep_mask, offset) を返す。keep_mask は最終的な可視集合。"""
    offset = kv - S
    assert offset >= 0
    mx.random.seed(seed)
    cache = Q._IndexerCache()
    if offset:
        # キャッシュを offset 列ぶん先に埋める (raw キーは中身が何でもよい ---
        # 見ているのは展開規則だけ)。
        cache.update(mx.random.normal((1, offset, idx.head_dim)))
    x = mx.random.normal((1, S, hidden))

    res = idx._pooled_and_top(x, rope, cache, offset)
    assert res is not None, f"kv={kv} で疎化が起きていない (budget={BUDGET})"
    keep_block, n_blocks, kv_len, _ = res
    assert kv_len == kv, (kv_len, kv)

    # `__call__` はもう一度 `_pooled_and_top` を呼ぶので、キャッシュを同じ
    # 状態に戻してから通す (update が offset を進めてしまうため)。
    cache2 = Q._IndexerCache()
    cache2.keys = cache.keys[:, :offset] if offset else cache.keys[:, :0]
    sparse = idx(x, rope, cache2, offset)
    assert sparse is not None
    # 実際に sdpa へ渡る形。mask=None なので sparse がそのまま返る規約。
    final = Q.Attention._final_mask(None, None, sparse, None, S, mx.bfloat16)
    mx.eval(keep_block, final)
    return np.array(keep_block)[0], np.array(final)[0, 0], offset, n_blocks


def _reference(keep_block: np.ndarray, offset: int, kv: int, n_blocks: int):
    """HF の規則で (S, kv) の可視集合を組む。

    ``keep_block`` は (S, n_blocks) --- ブロック選択そのものは共有し、
    「候補判定 + トークン幅への展開 + tail」だけを参照側で組み直す。
    """
    S = keep_block.shape[0]
    q = offset + np.arange(S)

    # 2: 完全ブロックの範囲。選ばれたブロックが全部この範囲に入っていること
    # (`block_end <= q` と `i < (q+1)//cr` が同じであることの確認も兼ねる)。
    ncomp = (q + 1) // CR
    bad = np.nonzero(keep_block & (np.arange(n_blocks)[None, :] >= ncomp[:, None]))
    if bad[0].size:
        raise AssertionError(
            f"完全ブロックでないブロックが選ばれている: 行 {bad[0][:5]} "
            f"ブロック {bad[1][:5]}"
        )

    ref = np.zeros((S, kv), dtype=bool)
    ref[:, : n_blocks * CR] = np.repeat(keep_block, CR, axis=1)

    # 4: クエリごとの tail
    own = ((q + 1) // CR) * CR
    cols = np.arange(kv)[None, :]
    ref |= (cols >= own[:, None]) & (cols <= q[:, None])
    return ref


def _global_gap(offset: int, S: int, n_blocks: int) -> int:
    """global tail が HF 参照に対して落とすセル数 (予言値)。

    行 ``q`` が端数域 (``q >= cr*n_blocks``) にいれば global tail は query tail と
    同じ区間になる。それ以外の行では global tail が空で、query tail の
    ``q % cr + 1`` 列 (``q % cr == cr-1`` の行は 0 列) がまるごと落ちる。
    """
    q = offset + np.arange(S)
    per = np.where((q >= CR * n_blocks) | (q % CR == CR - 1), 0, q % CR + 1)
    return int(per.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,3,2048",
                    help="S (クエリ行数) のカンマ区切り")
    ap.add_argument("--kv", default="2049,4096,8191,17000",
                    help="kv 長のカンマ区切り")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    mx.set_default_device(mx.cpu)
    rows = [int(v) for v in a.rows.split(",")]
    kvs = [int(v) for v in a.kv.split(",")]

    idx, rope, hidden = _make_indexer()
    orig_mode = QT.MODE

    print(f"{'mode':>6} {'S':>5} {'kv':>6} {'tail':>4} "
          f"{'不一致行':>8} {'不一致セル':>10} {'予言セル':>9}  判定")
    print("-" * 72)
    fail = 0
    try:
        for mode in ("query", "global"):
            QT.MODE = mode
            for S in rows:
                for kv in kvs:
                    if kv - S < CR - 1:
                        continue
                    kb, got, offset, n_blocks = _run(
                        idx, rope, hidden, S, kv, a.seed)
                    ref = _reference(kb, offset, kv, n_blocks)
                    diff = got != ref
                    nrow = int(diff.any(axis=1).sum())
                    ncell = int(diff.sum())
                    want = 0 if mode == "query" else _global_gap(
                        offset, S, n_blocks)
                    ok = ncell == want
                    fail += not ok
                    print(f"{mode:>6} {S:>5} {kv:>6} {kv % CR:>4} "
                          f"{nrow:>8} {ncell:>10} {want:>9}  "
                          f"{'OK' if ok else 'NG'}")
    finally:
        QT.MODE = orig_mode

    print()
    if fail:
        print(f"NG が {fail} 件")
        return 1
    print("全件 期待どおり (query = HF 参照と 100% 一致、"
          "global = 予言どおりのセル数だけ落ちる)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
