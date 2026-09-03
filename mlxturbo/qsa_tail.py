"""QSA の未完成ブロック (tail) の可視規則を選ぶ knob。

HF transformers の参照実装 (`models/qwen4_exp/modeling_qwen4_exp.py` の
`Qwen4ExpTextQSAIndexer.forward`) は、クエリ ``q`` の可視集合を

    「完全ブロック (block_end <= q) から top-k」 ∪ 「tail」

とし、**tail をクエリごとに** 決める --- ``visible = 0..q`` を
``compress_ratio`` 個ずつに切った余り、つまり列
``[cr*floor((q+1)/cr), q]`` (自分自身を含む 0〜cr-1 列) である。
``q % cr == cr-1`` の行では空になり、そのときその行のブロックは完全ブロック
として top-k の候補に入る。

こちらの実装は長らく **global tail** ---「kv 長の端数列 (``cr*n_blocks``
以降) を因果性の範囲で見せる」--- だった。最終行 (S=1 の decode) では両者は
一致するが、``S > 1`` の呼び出し (prefill の全行、投機検証の非最終行) では
``(q+1) % cr != 0`` の行が **自分自身と直前 0〜cr-2 トークンを見ていない**。

``MODE``
    ``"global"`` (既定、従来の挙動) か ``"query"`` (HF 参照と同じ規則)。
    環境変数 ``MLXTURBO_QSA_TAIL`` で決まる。実装は
    `mlxturbo/_vendor/qwen4_exp.py` (``QSAIndexer.__call__`` と
    ``Attention._gather_tile_attn``)、`mlxturbo/batch.py` の
    ``indexer_call``、`mlxturbo/kernels/prefill_attn.py` のカーネルの
    4 か所にあり、全部がここを見る。

``TIEBREAK``
    ブロックスコアの同点処理。HF は ``torch.topk`` で同点のとき小さい添字を
    残す。mlx の ``argpartition`` にその保証は無いので、``scores`` から
    ``block_idx * TIEBREAK_EPS`` を引いて小さい添字を優先させる
    (mlx-serve が HF との比較で使っているのと同じ bias)。環境変数
    ``MLXTURBO_QSA_TIEBREAK=1``、既定 off。tail の規則とは独立に切れる
    (どちらの効果かを分けて測るため)。

どちらも **モジュール属性として読むこと** (``qsa_tail.MODE``)。
`tools/decode_ab.py` の A/B は 1 プロセス内でここを書き換えて切り替える。
"""

from __future__ import annotations

import os

TIEBREAK_EPS = 1e-7

MODE = os.environ.get("MLXTURBO_QSA_TAIL", "global")
if MODE not in ("query", "global"):
    raise ValueError(
        f"MLXTURBO_QSA_TAIL={MODE!r} は不正 (query か global)"
    )

TIEBREAK = os.environ.get("MLXTURBO_QSA_TIEBREAK", "0") == "1"


__all__ = ["MODE", "TIEBREAK", "TIEBREAK_EPS"]
