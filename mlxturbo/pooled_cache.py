"""段 X1: QSA pooled キーの増分キャッシュ (`docs/research/KERNEL-PROGRAM.md` 段 X1)。

`_vendor/qwen4_exp.py` の `QSAIndexer._pooled_and_top` は、フォワードのたびに
pooled キー (mean pooling + k_layernorm + rope) を全ブロック作り直していた。
`tools/micro_indexer.py` の内訳計測 (17k・幅 2) では、この再構築 (27.4%) と
そこへの rope (18.1%) を合わせて **indexer の 45.5%** を占めていた。

ブロックは compress_ratio トークン分が揃った時点で内容が確定し (mean・
k_layernorm はブロック内で閉じている)、以後は不変。rope の角度も
block_starts (ブロック開始位置) だけで決まり、ブロックが確定した時点で
固定される。よって新しく完成したブロックぶんだけ計算して足せば、毎回
全ブロックを作り直すのとビット一致する (段 X1 の反転条件どおり、値が
変わるならそれは実装が間違っている)。

実体は `_vendor/qwen4_exp.py` の `_IndexerCache.pooled` /
`QSAIndexer._pooled_and_top`。増分ぶんの計算対象は毎更新で追記される
raw キー (`_IndexerCache.update`) の末尾のみで、cache が `None` (キャッシュ
無しの単発呼び出し) のときは従来どおり毎回全部作り直す。

``_IndexerCache.keys`` の setter (trim / rollback / batch の filter・
extend・extract・merge・state 復元がどれもここを通る) は、縮み・
並べ替えのどちらもあり得るので pooled キャッシュを無条件で捨てる
(古いブロックを静かに使い回すのが最悪という判断)。通常の追記
(``update``) だけが増分で伸ばす。

ここは `gather_attn.py` と同じ形の有効化・無効化関数を置くだけ:

    from mlxturbo import pooled_cache
    pooled_cache.disable_pooled_cache(model)  # A/B の B 側 (毎回作り直し)
    pooled_cache.enable_pooled_cache(model)   # 既定 (A 側、増分)

既定 on (`QSAIndexer.__init__` の `self._pooled_cache = True`)。ビット
不変なので、既定を on にするのに品質判定は要らない -- 出力が動いたら
それは実装のバグ。A/B は `tools/decode_ab.py --knob pooled-cache`
(値の一致を対照にしつつ ms/token の改善を見る)。合成モデルでの正しさ
確認は `tools/verify_pooled_cache.py`。
"""

from __future__ import annotations


def _each_layer(model, mtp=None):
    for layer in model.model.layers:
        yield layer
    if mtp is not None:
        for layer in mtp.layers:
            yield layer


def enable_pooled_cache(model, mtp=None) -> int:
    """レイヤーの `QSAIndexer` に増分キャッシュを立てる (既定と同じ状態)。

    `indexer` を持つ層 (self_attn がある層) にだけ `_pooled_cache = True` を
    立てる。GDN 層 (linear_attn) には触らない。戻り値は適用した層数。
    """
    n = 0
    for layer in _each_layer(model, mtp):
        sa = getattr(layer, "self_attn", None)
        idx = getattr(sa, "indexer", None) if sa is not None else None
        if idx is not None:
            idx._pooled_cache = True
            n += 1
    return n


def disable_pooled_cache(model, mtp=None) -> int:
    """`enable_pooled_cache` を打ち消す (= 毎回全ブロックを作り直す旧経路)。

    A/B で交互に測るために要る。戻り値は外した数。
    """
    n = 0
    for layer in _each_layer(model, mtp):
        sa = getattr(layer, "self_attn", None)
        idx = getattr(sa, "indexer", None) if sa is not None else None
        if idx is not None:
            idx._pooled_cache = False
            n += 1
    return n
