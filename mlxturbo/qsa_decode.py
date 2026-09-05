"""MLXTURBO_QSA_DECODE_KERNEL: decode / verify 幅の QSA attention を自前の
2 カーネルで走らせる (段 K2c)。

段 K2a (`mlxturbo/kernels/qsa_select.py`) がブロック top-k 選択を 1 dispatch に、
段 K2b (`mlxturbo/kernels/qsa_attn_decode.py`) が「bool マスクの実体化 +
``ceil(S*gqa/32)`` 回の sdpa 呼び」を MLX の 2-pass vector の写し 1 本に畳む。
ここはその 2 つを `Attention.__call__` に繋ぐ口
(`Attention._decode_qsa_forward`、`mlxturbo/_vendor/qwen4_exp.py`) を
有効化・無効化するだけで、`gather_attn.py` / `indexer_lean.py` と同じ形。

同じ``blocks``なら**出力はビット一致する。**選ぶ集合 (同点は添字の昇順) も演算の順 (逐次
online softmax、`fast::exp`、bf16 partials、pass 2) も本家の写しで、
`tools/verify_qsa_attn_decode.py` が S∈{1..6} × kv 2k〜50k × スコア 4 種の
120 通りで `mx.array_equal` を取っている。取り分は消える op の本数 ---
argpartition (GPU では全ソート) とマスク組みで 25〜40 本、sdpa 側は
「全キーの mask バイトを 1 個ずつ待つ」直列ループが候補判定だけになる。
冷たい 12 層連鎖では 17k S=2 で 281→139 us/層、50k S=6 で 1278→481 us/層。

**``MLXTURBO_QSA_TAIL=query`` が要る** (2026-09-03 の commit 11790ee で
**これが既定になった**ので、通常は何もしなくてよい)。K2b が写した参照は HF と
同じ per-query tail 1 本だけで、旧既定の global tail は実装していない
(`mlxturbo/qsa_tail.py`)。global に戻して有効化しても
`Attention._decode_qsa_forward` が発火せず、`qsa_attn_decode` が理由を
1 行表示して素の経路に落ちる。

    from mlxturbo import qsa_decode
    qsa_decode.enable_qsa_decode_kernel(model)    # A 側
    qsa_decode.disable_qsa_decode_kernel(model)   # B 側 (既定)

既定 off。in-model A/B は `tools/decode_ab.py --knob qsa-decode-kernel`。
17k の実測 (2026-09-03、`bench/results/qsa-decode-kernel-17k-v2.json`、
depth 混合がそろった 10 行): ms/round -4.1%、ms/tok -4.5%、出力はビット一致。
製品既定ではさらにKV長18,000以下だけ``blocks=64``へ縮めるため本家とは丸めが変わる。
17kの長文課題は非退行、50kは従来表へ戻る。``MLXTURBO_QSA_BLOCKS64=0``で無効化できる。
発火は
`mlxturbo.kernels._fire.snapshot()` の ``qsa_decode_kernel``
(選択側は ``qsa_select``、attention 側は ``qsa_attn_decode``)。
"""

from __future__ import annotations

import os


def _each_layer(model, mtp=None):
    # 層の列挙は `fused._model_layers` に寄せる (族ごとにラッパの形が違う。
    # qwen4_exp は `model.model.layers`、qwen3_5 (27B) は
    # `model.language_model.model.layers`)。見つからなければ 0 層で、
    # 呼び手は何もせず 0 を返す。
    from .fused import _model_layers

    for layer in _model_layers(model):
        yield layer
    if mtp is not None:
        for layer in mtp.layers:
            yield layer


def enable_qsa_decode_kernel(model, mtp=None) -> int:
    """レイヤーの ``Attention`` に ``_qsa_decode = True`` を立てる。

    `indexer` を持つ層 (= full attention 層) にだけ立てる。GDN 層
    (`linear_attn`) には触らない。戻り値は適用した層数。
    """
    n = 0
    for layer in _each_layer(model, mtp):
        sa = getattr(layer, "self_attn", None)
        if sa is not None and hasattr(sa, "indexer"):
            sa._qsa_decode = True
            n += 1
    return n


def disable_qsa_decode_kernel(model, mtp=None) -> int:
    """`enable_qsa_decode_kernel` を打ち消す (既定と同じ状態に戻す)。
    A/B で交互に測るために要る。戻り値は外した数。"""
    n = 0
    for layer in _each_layer(model, mtp):
        sa = getattr(layer, "self_attn", None)
        if sa is not None and getattr(sa, "_qsa_decode", False):
            sa._qsa_decode = False
            n += 1
    return n


def enable_qsa_decode_kernel_default(model, mtp=None, log_prefix: str = "") -> int:
    """`mlxturbo/runner.py` の `enable_default_fusions` から無条件に呼ばれる
    自己ゲート版 (`enable_indexer_lean_default` と同じ作法)。環境変数
    **既定 on** (2026-09-03 18:15、17k ms/round -4.1%、ビット一致)。``MLXTURBO_QSA_DECODE_KERNEL=0`` で off。

    ``MLXTURBO_QSA_TAIL`` が ``query`` でないときも属性は立てるが、その場合は
    `Attention._decode_qsa_forward` が毎回退く (理由は 1 度だけ表示される)。
    ここで警告しておくのは「有効にしたのに発火 0」を取り違えないため。
    """
    if os.environ.get("MLXTURBO_QSA_DECODE_KERNEL", "1") == "0":
        return 0
    n = enable_qsa_decode_kernel(model, mtp)
    if log_prefix:
        print(f"{log_prefix} QSA decode カーネル有効 (段 K2c、{n} 層、"
              "decode/verify 幅 S<=8 かつ B=1 のみ。"
              "MLXTURBO_QSA_DECODE_KERNEL=1)")
        from . import qsa_tail as _qt

        if _qt.MODE != "query":
            print(f"{log_prefix}   ただし MLXTURBO_QSA_TAIL={_qt.MODE} なので"
                  " 1 回も発火しない (query の参照だけを写してある)")
    return n
