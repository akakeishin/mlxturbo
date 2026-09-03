"""MLXTURBO_INDEXER_LEAN: decode/verify 幅 (S<=8) の QSAIndexer 費用を減らす。

`docs/research/SESSION-2026-09-02-CATCHUP.md` 末尾「decode 幅 S=2 の
attention 部品、連鎖版」: S=2 の attention 1 層で indexer
(`QSAIndexer._pooled_and_top` + `__call__`) が 228us (kv=4k) 〜 307us
(kv=50k) と、sdpa 本体 (71us) の 3 倍。12 層で 2.7 ms/round (decode ラウンド
38〜43 ms の 6〜7%)。FLOP はほぼゼロ (n_blocks 個の pooled key との内積) な
ので、op 数 (小さいカーネルの dispatch レイテンシ) の問題。

ここで削るのは、decode の毎ラウンド「値は変わらないのに毎回作り直して
いた」もの 2 つ (実体は `_vendor/qwen4_exp.py` の
`_IndexerCache.block_grid` / `.pooled_fp32`、`QSAIndexer._pooled_and_top`
の `lean` 分岐。QSAIndexer 自体の I/F (`__call__`/`select_blocks` の返り値
の形) は変えていない):

    (1) block_starts / block_end (= `arange(n_blocks) * compress_ratio` と
        その `+ compress_ratio - 1`)。n_blocks だけで決まる決定的な計算で、
        ブロックが新しく確定した回 (compress_ratio/S 回に 1 回) だけ伸びる。
        それ以外の毎ラウンドでは arange + 乗算 (+ block_end の加算) の
        計 3 op をキャッシュ参照 (0 op) にする。`block_starts` は n_blocks
        の**接頭辞として安定** (小さい n_blocks の値は大きい n_blocks で
        計算したときの先頭部分と完全に一致する) なので、縮んだ呼び出しが
        来てもスライスするだけで正しく、明示的な無効化は要らない
        (`_IndexerCache.block_grid` の docstring 参照)。

    (2) pooled の fp32 キャスト (einsum の入力用)。`pooled` 自体は段 X1の
        増分キャッシュ (`mlxturbo/pooled_cache.py`、既定 on) のおかげで
        ブロックが新しく確定した回だけ差し替わる。fp32 キャストも同じ
        頻度でしか要らないのに、毎ラウンド `pooled.astype(mx.float32)` を
        やり直していた。`_pooled_n` が前回と同じならキャストし直さない。

**値は変えない。**どちらも「同じ入力なら毎回同じ結果になる決定的な計算」
を、値が変わらない回に限ってやり直さないだけ。KLD を動かす類の変更
(bf16 スコアなど) はここには含まない。

decode/verify 幅 (S<=8) だけに絞ってある (`QSAIndexer._pooled_and_top` の
`lean` 判定)。prefill 幅 (S>8、既定チャンク 2048) はこの knob を on にしても
経路が変わらない --- `tools/decode_ab.py --knob indexer-lean` を
`DECODE_ONLY_KNOBS` に入れられる理由もここ。

    from mlxturbo import indexer_lean
    indexer_lean.enable_indexer_lean(model)   # A 側
    indexer_lean.disable_indexer_lean(model)  # B 側 (既定)

既定は off (`QSAIndexer` に `_indexer_lean` 属性が無ければ
`getattr(self, "_indexer_lean", False)` で off 相当)。in-model A/B
(`tools/decode_ab.py --knob indexer-lean`) は親が行う (このファイルは
CPU の一次検査 (`tools/vendor_fingerprint.py`、`tools/indexer_ops.py`) の
範囲でしか検証していない)。
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


def enable_indexer_lean(model, mtp=None) -> int:
    """レイヤーの `QSAIndexer` に `_indexer_lean = True` を立てる (段 3(b)/
    `pooled_cache.py` と同じ形)。`indexer` を持つ層 (= full attention 層)
    にだけ立てる。戻り値は適用した層数。"""
    n = 0
    for layer in _each_layer(model, mtp):
        sa = getattr(layer, "self_attn", None)
        idx = getattr(sa, "indexer", None) if sa is not None else None
        if idx is not None:
            idx._indexer_lean = True
            n += 1
    return n


def disable_indexer_lean(model, mtp=None) -> int:
    """`enable_indexer_lean` を打ち消す (既定と同じ状態に戻す)。A/B で
    交互に測るために要る。戻り値は外した数。"""
    n = 0
    for layer in _each_layer(model, mtp):
        sa = getattr(layer, "self_attn", None)
        idx = getattr(sa, "indexer", None) if sa is not None else None
        if idx is not None:
            idx._indexer_lean = False
            n += 1
    return n


def enable_indexer_lean_default(model, mtp=None, log_prefix: str = "") -> int:
    """`mlxturbo/runner.py` の `enable_default_fusions` から無条件に呼ばれる
    自己ゲート版 (`enable_moe_verify_gather` / `enable_fast_rope` と同じ
    作法)。この関数を呼ぶだけでは何も起きない --- 環境変数
    `MLXTURBO_INDEXER_LEAN=1` が立っているときだけ `enable_indexer_lean` を
    呼ぶ (呼び出し側が env var を忘れても既定 off が保たれるように、ゲートを
    関数自身の中に持たせている)。戻り値は適用した層数 (0 なら未適用)。
    """
    if os.environ.get("MLXTURBO_INDEXER_LEAN") != "1":
        return 0
    n = enable_indexer_lean(model, mtp)
    if log_prefix:
        print(f"{log_prefix} indexer lean 有効 ({n} 層、decode/verify 幅"
              " S<=8 のみ。MLXTURBO_INDEXER_LEAN=1)")
    return n
