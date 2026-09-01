"""標準的な mlx_lm モデル (``model.model.layers`` を持ち、各層を
``layer(h, mask=mask, cache=c)`` で回す形) 向けの、アーキテクチャ非依存の
段階投入 (staged submission)。

spec_flash.py の ``_staged_forward`` (Flash-Next / qwen4_exp 型、hyper
connection mixer あり) と同じ手法をここに一般化した: 全 layer ぶんの
遅延グラフを組み切ってから流すと、グラフ構築中 (7.3ms/round、spec_flash.py
側の実測。docs/research/KERNEL-BRIEF-DECODE-BW.md 参照) の間 GPU が遊ぶ。
``every`` 層ごとに ``mx.async_eval(h)`` で
途中結果を先に投入すれば、GPU がそこから計算を始める一方で CPU は残りの
グラフを組み続けられる。``async_eval`` は値を変えずスケジューリングだけを
変えるので、``every`` の値によらず計算内容は同一 (出力トークン列が正しさの
基準)。

適用範囲: 対応するのは ``mlx_lm.models.qwen3_5.Qwen3_5TextModel`` と同じ
呼び出し規約の層構成 --- ``embed_tokens`` で埋め込み、``layer.is_linear``
で linear-attention 層 (GDN) と full-attention 層を判別して ``ssm_mask``/
``fa_mask`` を使い分け、各層は ``layer(h, mask=mask, cache=c)`` で呼ぶ形。
mlxturbo/spec.py の ``SpecEngine._hidden_forward`` (capture=False 分岐) は
この形そのものなので、そこでの決め打ちの構造として利用している。

最終 RMSNorm・lm_head はここでは適用しない (呼び手側にすでにある
tied-embedding 分岐や量子化 dispatch_scope の処理を重複させないため)。
``_hidden_forward`` と同じく、戻り値は「最終 RMSNorm 適用前の hidden
state」。
"""

from __future__ import annotations

import mlx.core as mx

from ._mlx_compat import create_attention_mask, create_ssm_mask


def staged_forward(model, ids: mx.array, caches, every: int = 2) -> mx.array:
    """qwen3_5 型モデルの層ループを ``every`` 層ごとに ``mx.async_eval``
    しながら実行する。

    ``model``: ``embed_tokens``/``layers``/``fa_idx``/``ssm_idx`` を持つ
    オブジェクト (``mlx_lm.models.qwen3_5.Qwen3_5TextModel`` 相当。
    mlxturbo.spec.SpecEngine では ``self.inner`` がこれにあたる)。
    ``ids``: ``(1, S)`` の token id (mlx_lm の呼び出し規約と同じく、埋め込み
    前に batch 軸を持つ)。
    ``caches``: ``model.make_cache()`` が返す形。linear 層には
    ``ArraysCache`` 相当、full-attention 層には ``KVCache`` 相当が並ぶ。
    ``every``: 0 で段階投入を無効化 (一括構築、従来と完全に同一の計算順)。

    計算内容は ``Qwen3_5TextModel.__call__`` の本体 (最終 ``self.norm``
    適用前まで) の写しであることが正しさの根拠 -- 本家を変えるときは
    ここも変えること。
    """
    h = model.embed_tokens(ids)
    fa_mask = create_attention_mask(h, caches[model.fa_idx])
    ssm_mask = create_ssm_mask(h, caches[model.ssm_idx])

    layers = model.layers
    n = len(layers)
    for i, (layer, c) in enumerate(zip(layers, caches)):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=c)
        if every and (i + 1) % every == 0 and i < n - 1:
            mx.async_eval(h)
    return h
