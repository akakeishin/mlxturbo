"""decode 経路の A/B を 1 プロセス内で交互に測る (knob 差し替え式)。

A = 新しい側、B = 比較対象 (多くは修正前 / 既定 off)。knob ごとに
「どう切り替えるか」と「何をもって合格とするか」を `KNOBS` に書いてある。

## 共通の作法 (CLAUDE.md の計測の作法に従う)

- 1 プロセス内でプロンプトごとに A→B→B→A。線形の熱ドリフトを相殺する。
- **プロセスの最初の 1 本は捨てる。**実測で最初だけ 19.19 ms/tok
  (以降 16.45)、長文脈の TTFT は 73s (以降 35s) になる。混ぜると数 % の
  嘘の差が出る。温まってからの繰り返しは 0.2% 以内。
- 長時間回し続けると熱で 25% 落ちる (16.9k TTFT 35s -> 45s の実測)。
  絶対値を語るなら冷ましてから短く測る。ここで信じるのは A/B の差だけ。
- 生成長は全条件そろえる (既定 512)。長文脈は実文書から切った窓を使う
  (繰り返し文字列だと n-gram と MTP が当てすぎて受理率が嘘になる)。
- **長文脈では `--prefill-once`。**50k の prefill は 100 秒級で、A/B x 回文
  だと 12 回踏むことになる (実測: 50k の 1 knob で 30 分)。knob が prefill に
  効かないなら 1 回で足りる -- エンジンが checkpoint 復帰用に持っている
  `resume` をそのまま使い、控えたキャッシュから decode だけを流す。
  各条件がまったく同じ状態から始まるので**制御はむしろ強くなる**。
  prefill に効く knob (`prefill-group` / `stage-every`) では使えない (弾く)。

## knob

`qsa-tail`   QSA の端数ブロック因果性 (5d1e1c5)。B は返り値の端数列を
             True に戻す薄い包みで再現する (写しは作らない)。
             合格条件: 短文脈 (QSA 不活性) で A/B の出力が完全一致すること。
             ここが割れたら測定は無効。長文脈は合否を付けない (正しさの
             修正なので遅くても戻さない) が、tok/round の相対低下が 5% を
             超えたら代償として目立つ形で報告する。
             結果 (2026-09-01): tok/round +2.1%、ms/token は符号が揃わず。
             結果 (moe-verify, 2026-09-01): 短 decode 3 本とも +46〜52% 遅く、
             既定 off で据え置き。

`depth`      MTP 投機の深さ (spec_flash.MTP_DEPTH、既定 2)。1/2/3 を回文順で
             回す。貪欲なので品質は不変、判定は ms/token だけで行う。
             合格条件: 既定 2 より速い深さがあれば、短・長の両方で改善して
             いることを確かめてから既定を動かす。片方だけなら文脈長で
             切り替える話になるので、その場で決めない。

`indexer-cache` QSA の生鍵キャッシュを確保方式にした件 (2026-09-01)。
             B は毎更新 concat の旧実装。値はビット不変なので、対照 (短文脈で
             出力一致) がそのまま効く。合格条件: 長文脈で ms/token が改善する
             こと (17k で 52MB/フォワードの読み書きが消える見込み)。

`pooled-cache` QSA の pooled キー (段 X1、`docs/research/KERNEL-PROGRAM.md`)。
             A は増分キャッシュ (既定)、B は毎回全ブロック作り直し (旧経路)。
             値はビット不変 (mean・k_layernorm・rope はどれもブロック内で
             閉じているので、増分で作っても全部作り直しても同じ)。対照
             (出力一致) がそのまま効く。合格条件: 長文脈で ms/token が
             改善すること (`tools/micro_indexer.py` の内訳で pooled 再構築 +
             rope が indexer の 45.5% を占めていた)。

`stage-every` 段階投入の間隔 (既定 2)。1/2/4 を回文順で。値は変わらず
             スケジューリングだけが変わるので、対照 (出力一致) が効く。
             合格条件: 短・長の両方で ms/token が改善すること。

`prefill-group` layer-major prefill のグループ幅 (既定 4)。判定は
             **prefill_s** で見る (decode には効かない)。代償は checkpoint
             粒度が粗くなること。

`bool-mask`  sdpa に渡す疎マスクを bool にする (現行は fp32 の加算マスク)。
             **集合が同じなので品質は変わらない**。合格条件: 17k の ms/token が
             改善すること。改善したら既定を bool に変える。

`prefill-attn` prefill の gather + softmax を 1 本の Metal カーネルに畳む
             (段 P1、MLXTURBO_PREFILL_ATTN、既定 off)。A がカーネル、
             B が現行の汎用 op 2 段。**注意する集合は同じ**で、加算順と
             スケーリングの順だけが変わる。判定は **prefill_s** で見る
             (decode 幅では比の判定が先に効いて両者とも従来経路へ落ちる)。

`moe-verify` 共有タイル gather v2 (MLXTURBO_MOE_VERIFY、既定 off)。
             verify 幅の MoE だけを差し替える。
             合格条件: **ms/token が短・長の両方で改善すること。**
             どちらかで悪化したら既定 off のまま据え置く。出力は
             累積順が変わるので一致を要求しない (tok/round の変化は
             テキスト運と区別できないので、判定は ms/token で行う)。

    tools/biglock.sh .venv/bin/python tools/decode_ab.py --knob moe-verify \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep

`gdn-prework` GatedDeltaNet の前処理 (conv1d -> silu -> q/k の
             rms_norm+スケール -> 次段 conv 状態の書き出し -> g -> beta) を
             1 dispatch に畳む (MLXTURBO_GDN_PREWORK、既定 off、
             mlxturbo/kernels/gdn_prework.py)。decode/verify 幅のみが対象
             (gdn_prework.MAX_S / MAX_M)。**silu と beta の sigmoid は
             参照とビット一致しない** (hyper_connection.py と同じ bf16 の
             1 ulp 制約) ので出力一致は要求しない。
             合格条件: **ms/token が短・長の両方で改善すること。**
             どちらかで悪化したら既定 off のまま据え置く。

`gdn-blocked` GDN の再帰を長さ C のブロックに切り、ブロック内を行列積
             (単位下三角の連立) にまとめる (MLXTURBO_GDN_BLOCKED、既定 off、
             mlxturbo/kernels/gated_delta_blocked.py)。**prefill 幅
             (T >= 64) のみが対象**で、decode/verify 幅は両者とも逐次
             カーネルに落ちる -- **したがって --long で prefill を見ないと
             差が出ない。**加算順が変わるので出力一致は要求しない。
             合格条件: prefill_s の改善に加え、tok/round と KLD が悪化
             しないこと。

`hc-write`   hyper-connection の書き戻し (`DecoderLayer._combine`、
             MLXTURBO_HC_WRITE、既定 off) を mx.compile で 1 kernel に畳む。
             読み側 (`enable_hyper_connection_kernel`) とは別のディスパッチで、
             層あたり 2 回、48 層で計 96 回。演算順を変えないのでビット同一。
             合格条件: 出力一致 (対照) に加え、短・長の両方で ms/token が
             改善すること。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 長文脈の素材は tools/_bench_text.py の池から切る (実文。繰り返し文字列で
# 埋めると n-gram と MTP が当てすぎて受理率が嘘になる)
LONG_QUESTIONS = [
    "上の文書の要点を、初めて読む人向けに 5 つに整理してください。",
    "上の文書から、判断の根拠になっている数字だけを抜き出して並べてください。",
    "Summarize the document above, then list the open questions it leaves.",
]
SHORT_PROMPTS = [
    "分散システムにおける結果整合性について説明してください。",
    "Explain why speculative decoding helps when decoding is dispatch bound.",
    "Python でリストの重複を順序を保って除去する関数を書いてください。",
]


def _knob_qsa_tail(ctx):
    """A = 現行 (因果性で切る) / B = 修正前 (端数を常に可視)。

    B は `QSAIndexer.__call__` の返り値の端数列を True に戻す薄い包みで作る。
    写しを作らないので、A 側の実装がこの先変わっても B の意味はずれない。
    """
    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q

    orig = Q.QSAIndexer.__call__

    def stock_tail(self, x, rope, cache, offset, positions=None):
        keep = orig(self, x, rope, cache, offset, positions)
        if keep is None:
            return None
        kv_len = keep.shape[-1]
        tail = kv_len % self.compress_ratio
        if not tail:
            return keep
        ones = mx.ones(keep.shape[:-1] + (tail,), dtype=keep.dtype)
        return mx.concatenate([keep[..., : kv_len - tail], ones], axis=-1)

    def apply(variant):
        Q.QSAIndexer.__call__ = orig if variant == "A" else stock_tail

    return apply


def _knob_moe_verify(ctx):
    """A = 共有タイル gather v2 on / B = off (既定)。"""
    import os

    from mlxturbo import fused

    os.environ["MLXTURBO_MOE_VERIFY"] = "1"  # enable 側のゲートを開ける

    def apply(variant):
        if variant == "A":
            fused.enable_moe_verify_gather()
        else:
            fused.disable_moe_verify_gather()

    return apply


def _knob_gdn_prework(ctx):
    """A = GDN 前処理の融合カーネル on / B = off (既定)。"""
    import os

    from mlxturbo import fused

    os.environ["MLXTURBO_GDN_PREWORK"] = "1"  # enable 側のゲートを開ける

    def apply(variant):
        if variant == "A":
            fused.enable_gdn_prework_kernel()
        else:
            fused.disable_gdn_prework_kernel()

    return apply


def _knob_gdn_blocked(ctx):
    """A = GDN の再帰をブロック化スキャンで解く / B = 逐次カーネル (既定)。

    A は再帰を長さ C のブロックに切り、ブロック内を行列積 (単位下三角の連立)
    にまとめる (`mlxturbo/kernels/gated_delta_blocked.py`)。**prefill 幅
    (T >= 64) だけ**が対象で、decode/verify 幅は両者とも逐次カーネルに落ちる。

    合格条件: **17k の prefill 壁時計 (prefill_s)。**加算順が変わるので出力は
    一致せず、`tok/round` と KLD も併せて見る (prefill の数値を変えた
    カーネルが受理率を落として差し引きで負けた前例がある)。

    発火の確認: `mlxturbo.kernels._fire.snapshot()` の `gdn_blocked`。
    """
    import os

    from mlxturbo import fused

    os.environ["MLXTURBO_GDN_BLOCKED"] = "1"  # enable 側のゲートを開ける

    def apply(variant):
        if variant == "A":
            fused.enable_gdn_blocked_kernel()
        else:
            fused.disable_gdn_blocked_kernel()

    return apply


def _knob_hc_write(ctx):
    """A = 融合 (mx.compile) / C = 差し替えの機構だけ (融合なし) / B = 素 (既定)。

    `DecoderLayer._combine` は素のまま層あたり 2 回、48 層で 96 回呼ばれる。
    融合は multiply→add をまとめるだけで演算順を変えないのでビット同一。

    **C は対照。**2026-09-01 の A/B で、この knob が長文脈で +5.2%、
    gdn-prework が +5.3% 遅くなった。後者は**一度も発火していない**のに遅く、
    原因は `eligible()` の評価そのものだった。つまり knob を有効にする機構に
    費用がある。C は同じ差し替えをして同じ per-call の Python の仕事をし、
    compile を通さない素の式を呼ぶ。**A-C が融合の取り分、C-B が機構の費用。**

    合格条件: 出力一致 (対照)。
    """
    from mlxturbo import fused

    def apply(variant):
        fused.disable_hc_write()
        if variant == "A":
            fused.enable_hc_write()
        elif variant == "C":
            fused.enable_hc_write_nofuse()

    return apply


def _knob_depth(ctx):
    """MTP 投機の深さ。既定は 2 (spec_flash.MTP_DEPTH)。

    値は 2 つに限らないので、順序バイアスは 1,2,3,3,2,1 の回文で相殺する。
    貪欲なので深さを変えても出力トークン列は原則変わらない (verify は本体の
    argmax と一致したときだけ受理する)。よって品質は不変で、判定は速度だけ。
    厳密一致は要求しない -- 深さが変わると verify の幅が変わり、
    `mx.quantized_matmul` の幅依存の丸めで argmax がまれに割れる
    (spec_flash の注記と同じ性質)。
    """

    def apply(variant):
        eng = ctx["eng"]
        eng.depth = int(variant)
        # 既定では文脈長が indexer_budget を超えると engine 自身が depth 1 に
        # 落とす (spec_flash._depth_ctx_limit)。掃引でそれを効かせると、境界の
        # 向こう側では全条件が depth 1 になって**同じものを測ってしまう**
        # (実際 2.6k で 1 と 2 の tok/round が 3 桁一致して気づいた)。
        # ここでは自動切り替えを外し、指定した深さそのものを測る。
        eng.depth_ctx_limit = 1 << 30

    return apply



def _knob_rms_norm_gated(ctx):
    """A = 融合カーネル / C = 機構だけ (eligible まで評価して素へ) / B = 素 (既定)。

    `RMSNormGated.__call__` は素で 6 op、GDN を持つ 36 層で 1 回ずつ。
    `runner.py` には「実測で空振り」と記録されているが、**その測定は機構の
    費用を含んでいる。**2026-09-01 に `gdn_prework` が一度も発火しないまま
    長文脈で +5.3% 遅いことが分かり、原因が `eligible()` の評価だと判明した。
    C を挟めば A-C (カーネルの取り分) と C-B (機構の費用) に分けられる。

    カーネルはビット一致 (量子化重みを読まず、gate は fp32 のまま)。
    合格条件: 出力一致 (対照)。
    """
    from mlxturbo import fused

    def apply(variant):
        fused.disable_rms_norm_gated()
        if variant == "A":
            fused.enable_rms_norm_gated()
        elif variant == "C":
            fused.enable_rms_norm_gated_nofuse()

    return apply


def _knob_moe_route(ctx):
    """A = ルーティング融合 / C = 機構だけ / B = 素 (既定)。

    `SparseMoeBlock.__call__` の top-k と softmax は素で 7 op、48 層すべてで
    走る。`docs/BACKLOG.md` に「やって純損 (+0.34ms)」と記録されているが、
    **その +0.34ms が機構の費用と同じ桁**なので、分けて測り直す。

    ビット一致はしない (加算順が変わる)。合格条件は出力一致を要求しない。
    """
    from mlxturbo import fused

    def apply(variant):
        fused.disable_moe_route()
        if variant == "A":
            fused.enable_moe_route()
        elif variant == "C":
            fused.enable_moe_route_nofuse()

    return apply



def _knob_null(ctx):
    """A も B も**何もしない**。ハーネス自身のばらつきを測るための対照。

    2026-09-01 に、一度も発火していない `gdn-prework` を 2 回測って
    **-0.9% と +5.3%** が出た。中身が同じものが 6 ポイント振れたことになる。
    `eligible()` の費用も測ったが、17k decode 全体で 1.4ms + 0.8ms (0.01%) で
    説明にならない。

    **つまりハーネスの雑音が疑わしい。**この knob は差し替えを一切せず、
    回文順・温め捨て・prefill 使い回しまで本番の A/B とまったく同じ手順を
    踏む。ここで出る差が、**今日のすべての判定の下限**になる。

    合格条件: 出力一致 (対照)。ここが一致しないならハーネス自体が壊れている。
    """

    def apply(variant):
        return

    return apply


def _knob_indexer_cache(ctx):
    """A = 確保方式 (現行) / B = 毎更新 concat (2026-09-01 以前)。

    B は `_IndexerCache.update` を旧実装に戻すだけ。`keys` はプロパティなので、
    代入すればバッファごと置き換わり、当時と同じ「毎回全長を読み書きし直す」
    挙動になる。
    """
    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q

    new_update = Q._IndexerCache.update

    def old_update(self, k):
        self.keys = k if self.keys is None else mx.concatenate([self.keys, k], axis=1)
        return self.keys

    def apply(variant):
        Q._IndexerCache.update = new_update if variant == "A" else old_update

    return apply


def _knob_pooled_cache(ctx):
    """A = pooled キーの増分キャッシュ (段 X1、既定) / B = 毎回全ブロック作り直し。

    `_IndexerCache.pooled` / `QSAIndexer._pooled_and_top` (`mlxturbo/pooled_cache.py`
    の `enable_pooled_cache` / `disable_pooled_cache` を素通しするだけ)。値は
    ビット不変 (mean・k_layernorm・rope はどれもブロック内で閉じている) なので
    対照 (出力一致) がそのまま効く。
    """
    from mlxturbo.pooled_cache import disable_pooled_cache, enable_pooled_cache

    model = ctx["eng"].model

    def apply(variant):
        if variant == "A":
            enable_pooled_cache(model)
        else:
            disable_pooled_cache(model)

    return apply


def _knob_stage_every(ctx):
    """段階投入の間隔 (`spec_flash._STAGE_EVERY`)。既定 2。

    掃引が 16→2 で単調改善のまま端点で打ち切られていて **1 が未測**。しかも
    短 decode の probe でしか測っておらず 17k は未測 (fable-advisor 指摘)。
    0 は無効化 (層ループ中に async_eval を挟まない)。
    """
    import mlxturbo.spec_flash as SF

    def apply(variant):
        SF._STAGE_EVERY = int(variant)

    return apply


def _knob_prefill_group(ctx):
    """layer-major prefill のグループ幅 (`spec_flash._PREFILL_GROUP`)。既定 4。

    gather_qmm の効率は行数/expert に単調 (r=40/80/160 で 7.5/8.9/9.8 TFLOPS、
    密上限 11.2) なので、G を上げると MoE の効率は上がりうる。代償は
    checkpoint 粒度が g*2048 に粗くなること (2 ターン目の再 prefill が増える)。
    判定は **prefill_s** で見ること (decode には効かない)。
    """
    import mlxturbo.spec_flash as SF

    def apply(variant):
        SF._PREFILL_GROUP = int(variant)

    return apply


def _knob_qsa(ctx):
    """A = QSA 有効 (既定) / B = 無効 (素の causal)。

    17k の解剖で indexer が 3.80ms (ラウンドの 9.3%、長文ペナルティの 43%) と
    出た。しかも sdpa は加算マスクを渡されても**全 KV を読んで全スコアを
    計算する**ので、QSA の疎性は sdpa 側の節約になっていない疑いがある。
    だとすると長文の QSA は「費用だけ払って得をしていない」ことになる。

    B は `indexer_budget` を巨大にして `QSAIndexer.__call__` の早期 return に
    落とす (kv 長が budget 以下なら None を返す = 素の causal)。**出力は
    変わる** (QSA は full attention の近似で、切ると近似が外れる方向) ので、
    速度が勝っても採否は品質 (KLD) を測ってから。
    """
    eng = ctx["eng"]
    args_text = eng.model.args.text
    real = args_text.indexer_budget

    def apply(variant):
        args_text.indexer_budget = real if variant == "A" else 1 << 30
        for layer in eng.model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn.indexer.token_budget = args_text.indexer_budget

    return apply


def _knob_wide(ctx):
    """A = 連結射影 on / B = off (既定)。

    GDN が帯域下限の 50% しか出ていない (実測 6.56ms / 下限 3.25ms)。
    連結射影は射影 4 本を 1 本の qmm にまとめる。既定 off の理由は
    「連結で N が変わると qmv のカーネル変種が変わり、加算順の違いが
    最終 ulp を動かす疑い (tok/step 2.44 -> 2.23 の低下と時期が一致)」で、
    **単独 A/B の記録は無い**。出力が変わりうるので対照は要求しない。
    """
    from mlxturbo import fused

    eng = ctx["eng"]
    applied = {"on": False}

    def apply(variant):
        if variant == "A" and not applied["on"]:
            fused.enable_wide_projections(eng.model)
            applied["on"] = True
        elif variant == "B" and applied["on"]:
            fused.disable_wide_projections(eng.model)
            applied["on"] = False

    return apply


def _knob_bool_mask(ctx):
    """A = bool マスク / B = 加算マスク (現行)。

    `Attention._final_mask` はいま `where(sparse, 0, -inf)` で fp32 の加算
    マスクを作って sdpa に渡している。MLX の sdpa は **bool マスクも取れる**
    (`mx.fast.scaled_dot_product_attention` の docstring: "If the mask is an
    array it can be a boolean or additive mask")。

    **集合はまったく同じなので品質は変わらない。**変わるのはマスクの表現だけ。
    17k で attention が帯域下限の 25% しか出ていない理由が「加算マスクを
    実体化して渡すと融合経路から落ちる」なら、これだけで戻る。

    QSA を切る実験 (`--knob qsa`) と同じ「マスクが重いのか」を測るが、
    こちらは**品質が動かないのでそのまま採用できる**。
    """
    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q

    orig = Q.Attention._final_mask

    def bool_mask(self, mask, sparse, cache, S, dtype):
        if sparse is None:
            return mask
        if mask is None or isinstance(mask, str):
            return sparse
        # 左パディング等の bool マスクが来たら連言。加算に落とさない
        return mask & sparse

    def apply(variant):
        Q.Attention._final_mask = bool_mask if variant == "A" else orig

    return apply


def _knob_gather_attn(ctx):
    """A = gather 経路 (段 3(b)) / B = 現行の加算マスク。

    選ばれたブロックの和集合だけ KV を集めてから dense sdpa に渡す。
    **注意する集合は同じなので品質は変わらない** (加算順が変わるだけ、
    合成モデルで max|diff|=1.8e-7)。合格条件: 17k / 50k の ms/token が
    改善すること。改善したら既定 on にする。
    """
    from mlxturbo.gather_attn import disable_gather_attn, enable_gather_attn

    model = ctx["eng"].model

    def apply(variant):
        if variant == "A":
            enable_gather_attn(model)
        else:
            disable_gather_attn(model)

    return apply


def _knob_gather_tile(ctx):
    """gather attention のタイル幅 (段 P1)。0 = タイルなし (= 従来の 1 回)。

    prefill (S=2048) は untiled だと union がほぼ全ブロックになって効かない。
    クエリ行をタイルに切ると、隣接クエリの選択が相関するぶん union が縮む。
    代償はタイルごとの K/V 重複読み。**判定は prefill の壁時計**で、
    union 比 (`_gather_stats`) は理由を読むためだけに使う。

    gather 経路そのものも一緒に有効化する (タイルは gather の中の話なので)。
    """
    from mlxturbo.gather_attn import disable_gather_attn, enable_gather_attn

    model = ctx["eng"].model

    def apply(variant):
        disable_gather_attn(model)
        t = int(variant)
        if t >= 0:
            enable_gather_attn(model, tile=t)

    return apply


def _knob_prefill_attn(ctx):
    """A = 融合カーネル (段 P1) / B = 現行の gather (汎用 op 2 段)。

    A は選ばれたブロックの gather と softmax を 1 本の Metal カーネルに畳む
    (`mlxturbo/kernels/prefill_attn.py`)。B は `take_along_axis` で
    `k_sel`/`v_sel` を実体化してから union 幅の bool マスク付き sdpa に渡す
    現行経路。**注意する集合は同じ**で、変わるのは加算順とスケーリングの順。

    合格条件: **17k の prefill 壁時計 (prefill_s)** が改善すること。
    decode 幅では `_gather_forward` の比の判定が先に効いて両者とも従来経路に
    落ちるので、この knob は prefill でしか差が出ない。
    """
    from mlxturbo.gather_attn import (
        disable_prefill_attn,
        enable_gather_attn,
        enable_prefill_attn,
    )

    model = ctx["eng"].model

    def apply(variant):
        if variant == "A":
            enable_prefill_attn(model)
        else:
            disable_prefill_attn(model)
            enable_gather_attn(model)

    return apply


def _knob_prefill_pipeline(ctx):
    """group prefill の境界同期を非同期にする (段 D5)。A = 非同期 / B = 現行。

    グループ境界の `mx.eval` + `clear_cache` が完全同期で、次のグループの
    グラフ構築中に GPU が遊ぶ。全部使うグラフなので「作って捨てる」禁則には
    当たらない。**2 グループぶんの中間が同時に生きるので OOM 側の危険がある。**
    判定は prefill_s。
    """
    import mlxturbo.spec_flash as SF

    def apply(variant):
        SF._PREFILL_PIPELINE = variant == "A"

    return apply


KNOBS = {
    # name: (setup(ctx) -> apply(variant), variants, 出力一致を要求するか,
    #        まとめで基準にする variant)
    "qsa-tail": (_knob_qsa_tail, ["A", "B"], True, "B"),
    "moe-verify": (_knob_moe_verify, ["A", "B"], False, "B"),
    "gdn-prework": (_knob_gdn_prework, ["A", "B"], False, "B"),
    "gdn-blocked": (_knob_gdn_blocked, ["A", "B"], False, "B"),
    "hc-write": (_knob_hc_write, ["A", "C", "B"], True, "B"),
    "rms-norm-gated": (_knob_rms_norm_gated, ["A", "C", "B"], True, "B"),
    "moe-route": (_knob_moe_route, ["A", "C", "B"], False, "B"),
    "null": (_knob_null, ["A", "B"], True, "B"),
    "indexer-cache": (_knob_indexer_cache, ["A", "B"], True, "B"),
    "pooled-cache": (_knob_pooled_cache, ["A", "B"], True, "B"),
    "stage-every": (_knob_stage_every, ["1", "2", "4"], True, "2"),
    "prefill-group": (_knob_prefill_group, ["2", "4", "8"], True, "4"),
    "prefill-pipeline": (_knob_prefill_pipeline, ["A", "B"], True, "B"),
    "qsa": (_knob_qsa, ["A", "B"], False, "A"),
    "bool-mask": (_knob_bool_mask, ["A", "B"], False, "B"),
    "gather-attn": (_knob_gather_attn, ["A", "B"], False, "B"),
    # -1 は gather 自体を切る (現行既定)。0 はタイルなしの gather
    "gather-tile": (_knob_gather_tile, ["-1", "0", "256"], False, "-1"),
    "prefill-attn": (_knob_prefill_attn, ["A", "B"], False, "B"),
    "wide": (_knob_wide, ["A", "B"], False, "B"),
    "depth": (_knob_depth, ["1", "2", "3"], False, "2"),
}


def _snapshot(caches):
    """キャッシュの参照と offset を控える。MLX の配列は不変なので、
    参照を控えておけば付け替えで完全に戻せる
    (`spec_flash._pipeline_snapshot` と同じ理屈)。"""
    st = []
    for c in caches:
        if hasattr(c, "keys"):
            st.append(("a", c.keys, c.values, c.offset,
                       c.indexer.keys, c.indexer.offset))
        else:
            st.append(("l", [c[i] for i in range(4)]))
    return st


def _restore(caches, st):
    for c, rec in zip(caches, st):
        if rec[0] == "a":
            _, c.keys, c.values, c.offset, ik, io = rec
            c.indexer.keys = ik
            c.indexer.offset = io
        else:
            for i, v in enumerate(rec[1]):
                c[i] = v


def prefill_once(eng, ids, eos_ids):
    """prefill を 1 回だけ流し、(caches, snapshot, resume, 先頭トークン) を返す。

    50k の prefill は 100 秒級で、A/B x 回文だと 12 回踏むことになる
    (実測: 50k の 1 knob で 30 分)。**knob が prefill に効かないなら
    prefill は 1 回で足りる。**エンジンは checkpoint 復帰のために
    `resume` を既に持っているので、それをそのまま使う
    (`generate_stream` の `use_resume` 分岐、`ids` が 0 幅のときだけ有効)。

    副産物として、各条件がまったく同じ状態から decode を始めることになるので、
    prefill をやり直すより**むしろ制御が効く**。

    **prefill に効く knob (prefill-group、stage-every) には使わないこと。**
    """
    import mlx.core as mx

    caches = eng.model.make_cache()
    mx.clear_cache()
    gen = eng.generate_stream(ids, 1, caches=caches, eos_ids=eos_ids)
    first = []
    try:
        while True:
            first.extend(next(gen))
    except StopIteration as e:
        resume = e.value[2]
    return caches, _snapshot(caches), resume, first


def run_resumed(eng, caches, snap, resume, base_pos, n_tokens, eos_ids):
    """控えた状態から decode だけを流す。返り値は run_once と同じ形。"""
    import mlx.core as mx

    _restore(caches, snap)
    empty = mx.zeros((1, 0), dtype=mx.int32)
    t0 = time.perf_counter()
    gen = eng.generate_stream(empty, n_tokens, caches=caches, eos_ids=eos_ids,
                              resume=resume, base_pos=base_pos)
    out = []
    try:
        while True:
            out.extend(next(gen))
    except StopIteration as e:
        accepted, rounds = e.value[0], e.value[1]
    return out, 0.0, time.perf_counter() - t0, accepted, rounds


def run_once(eng, ids, n_tokens, eos_ids):
    """1 本流して (トークン列, prefill 秒, decode 秒, accepted, rounds) を返す。"""
    import mlx.core as mx

    caches = eng.model.make_cache()
    mx.clear_cache()
    t0 = time.perf_counter()
    gen = eng.generate_stream(ids, n_tokens, caches=caches, eos_ids=eos_ids)
    out, t_prefill = [], None
    try:
        while True:
            toks = next(gen)
            if t_prefill is None:
                t_prefill = time.perf_counter() - t0
                t_dec0 = time.perf_counter()
            out.extend(toks)
    except StopIteration as e:
        val = e.value
        accepted, rounds = val[0], val[1]
    t_dec = time.perf_counter() - t_dec0
    return out, t_prefill, t_dec, accepted, rounds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", required=True, choices=sorted(KNOBS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None,
                    help="既定は --model の中の mtp.safetensors")
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--out", default=None, help="結果 JSON の書き出し先")
    ap.add_argument("--only", choices=("both", "short", "long"), default="both",
                    help="長さの片方だけ回す (交差点探しで短文脈を省くため)")
    ap.add_argument("--variants", default=None,
                    help="knob の値をカンマ区切りで絞る (既定は KNOBS の全部)")
    ap.add_argument("--prefill-once", action="store_true",
                    help="長文脈で prefill を 1 回に畳む (prefill に効く knob "
                         "= prefill-group / stage-every には使わないこと)")
    args = ap.parse_args()

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401
    from mlxturbo import mtp_flash, spec_flash

    model_path = os.path.expanduser(args.model)
    model, tok = load(model_path)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    # 出荷経路と同じ融合を当てる。以前ここが
    # enable_hyper_connection_kernel() だけで、gather のソート (既定 16) が
    # 入らないまま測っていた (fable-advisor 指摘)。同一ハーネス内の相対比較
    # なら符号は生き残るが、閾値や交差点は構成で動く。
    from mlxturbo.kernels import _fire
    from mlxturbo.runner import enable_default_fusions

    enable_default_fusions(model, log_prefix="[decode_ab]")
    mtp_path = args.mtp or os.path.join(model_path, "mtp.safetensors")
    q = {"group_size": 64, "bits": args.mtp_bits} if args.mtp_bits else None
    mtp = mtp_flash.load_flash_mtp(os.path.expanduser(mtp_path),
                                   model.args.text, quantize=q)
    mx.eval(mtp.parameters())
    eng = spec_flash.FlashSpecEngine(model, mtp)

    eos = tok.eos_token_ids if hasattr(tok, "eos_token_ids") else ()
    eos_ids = tuple(eos) if eos else ()

    # ---- プロンプトを組む -------------------------------------------
    from _bench_text import long_prompts

    try:
        longs = (
            long_prompts(tok, args.ctx, LONG_QUESTIONS)
            if args.only != "short" else []
        )
    except ValueError as e:
        print(e)
        return 1

    def to_ids(text):
        return mx.array(tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True))[None]

    cases = []
    if args.only in ("both", "short"):
        cases += [("short", to_ids(p)) for p in SHORT_PROMPTS]
    if args.only in ("both", "long"):
        cases += [("long", to_ids(p)) for p in longs]

    setup, variants, control_identical, baseline = KNOBS[args.knob]
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",")]
        if baseline not in variants:
            baseline = variants[0]
    set_variant = setup({"eng": eng})
    order = variants + variants[::-1]

    print(f"knob={args.knob}  判定基準はモジュール docstring のとおり"
          " (測る前に宣言済み)。")
    print(f"生成長 {args.tokens} トークンで全条件そろえる。"
          " 最初の 1 本は温めなので捨てる。\n")

    # 温め: 最初の 1 本は必ず遅いので、計測に混ぜず先に捨てる。短文脈だけ
    # 温めても長文脈の初回は重みのページインで 2 倍かかる (73s vs 35s の実測)
    # ので、長い方も 1 本捨てる
    set_variant(variants[0])
    for want in ("short", "long"):
        for kind, ids in cases:
            if kind == want:
                run_once(eng, ids, 32, eos_ids)
                break

    # **ホワイトリストにすること。**`--prefill-once` は共有 prefill を
    # `variants[0]` で 1 回だけ組み、他の変種はそこから decode を再開する。
    # **prefill に影響する knob では比較が成立しない** (別の変種で組んだ
    # キャッシュから再開することになる)。
    #
    # 以前はブラックリストで、`wide` が漏れていた。その結果 **段 4 が
    # 「ms/round +495%」という嘘の数字を出していた** (2026-09-02 に再測して
    # 発覚。基準そのものが 4 倍ずれていた: 199.8 vs 48.2 ms/round)。
    # **漏れても誰も気づかない形なので、許可する側を列挙する。**
    #
    # ここに足すときは「その knob が prefill 幅で何もしないこと」をコードで
    # 確かめること。`gdn-prework` は S<=9 の適格判定があるので prefill 幅では
    # 発火しない。`depth` / `null` は生成の設定だけ。
    DECODE_ONLY_KNOBS = {
        "hc-write", "moe-verify", "gdn-prework", "depth", "null",
        "rms-norm-gated", "moe-route",
    }
    if args.prefill_once and args.knob not in DECODE_ONLY_KNOBS:
        print(f"knob={args.knob} は prefill に影響しうるので --prefill-once は"
              f"使えない (decode 専用と確認済みなのは"
              f" {sorted(DECODE_ONLY_KNOBS)})")
        return 1

    rows = []
    for kind, ids in cases:
        n = ids.shape[1]
        print(f"--- {kind} ctx={n} ---", flush=True)
        shared = None
        if args.prefill_once:
            t0 = time.perf_counter()
            caches, snap, resume, _first = prefill_once(eng, ids, eos_ids)
            shared = (caches, snap, resume)
            print(f"  prefill 1 回だけ流した ({time.perf_counter() - t0:.1f}s)。"
                  " 以降はここから decode のみ", flush=True)
        # **文脈グループごとに 1 本捨てる。**冒頭の温めは kind ごとに 1 回、
        # しかも最初のケースにしか当たっていなかった。グループが変わると
        # キャッシュを組み直すので 1 本目だけ一回きりの費用を払い、
        # **回文順 (A,B,B,A) はそれを相殺できない** -- 位置 1 の段差は
        # 線形のドリフトではないうえ、A が必ず位置 1 に来るため。
        #
        # 2026-09-01 に null knob (A も B も何もしない) で長文脈 +5.6% が出て
        # 判明した。実際の並びは 13.90 / 12.58 / 12.64 / 12.6x で、1 本目だけ
        # 段差になっている。**この日の長文脈の A/B は全部 A 側に約 5% の
        # 下駄を履かせていた。**
        set_variant(baseline)
        if shared is None:
            run_once(eng, ids, 32, eos_ids)
        else:
            run_resumed(eng, *shared, base_pos=n, n_tokens=32, eos_ids=eos_ids)
        for v in order:
            set_variant(v)
            # カーネルの発火回数を条件ごとに数え直す。適格判定は条件を外すと
            # 黙って False を返すので、「効果ゼロ」が遅いのか届いていないのかを
            # 区別する手が要る (2026-09-01 に GDN 前処理で実際に空振りした)。
            _fire.reset()
            if shared is None:
                out, tp, td, acc, rounds = run_once(eng, ids, args.tokens, eos_ids)
            else:
                out, tp, td, acc, rounds = run_resumed(
                    eng, *shared, base_pos=n, n_tokens=args.tokens,
                    eos_ids=eos_ids)
            ms = td / max(len(out), 1) * 1000
            tpr = len(out) / max(rounds, 1)
            # ms/token は ms/round と tok/round の比なので、**費用と受理が
            # 混ざる**。出力が変わりうる knob では ms/round を見ないと、
            # テキスト運による受理の増減を実装の速さと取り違える
            rows.append(dict(kind=kind, ctx=n, variant=v, n_out=len(out),
                             prefill_s=tp, decode_s=td, ms_per_tok=ms,
                             ms_per_round=td / max(rounds, 1) * 1000,
                             accepted=acc, rounds=rounds, tok_per_round=tpr,
                             head=out[:24]))
            fired = _fire.snapshot()
            rows[-1]["fired"] = fired
            fired_s = ("  発火 " + " ".join(f"{k}={n}" for k, n in
                                            sorted(fired.items()))) if fired else ""
            print(f"  {v}: prefill {tp:6.2f}s  decode {td:6.2f}s  "
                  f"{ms:6.2f} ms/tok  tok/round {tpr:.3f}  "
                  f"({acc}/{rounds}){fired_s}", flush=True)
    set_variant(baseline)

    # ---- まとめ -------------------------------------------------------
    print("\n=== まとめ ===")
    ok = True
    for kind in ("short", "long"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        for metric in ("ms_per_tok", "ms_per_round", "tok_per_round",
                       "prefill_s"):
            means = {}
            for v in variants:
                vals = [r[metric] for r in sub if r["variant"] == v]
                means[v] = sum(vals) / len(vals)
            base = means[baseline]
            if base == 0:
                # --prefill-once のとき prefill_s は全条件 0 になる (prefill は
                # 1 回しか流していない)。比を取る意味が無いので飛ばす
                continue
            cells = "  ".join(
                f"{v}={means[v]:8.3f}({(means[v] - base) / base * 100:+5.1f}%)"
                for v in variants
            )
            print(f"  {kind:5s} {metric:14s} {cells}   [基準 {baseline}]")
            if kind == "long" and metric == "tok_per_round":
                worst = min((means[v] - base) / base * 100 for v in variants)
                if worst < -5:
                    print("    ** tok/round が 5% 超落ちた条件がある **")

    if control_identical:
        # 対照: 短文脈は A と B で出力が完全一致するはず
        for c in sorted({r["ctx"] for r in rows if r["kind"] == "short"}):
            sub = [r for r in rows if r["ctx"] == c]
            if len({tuple(r["head"]) for r in sub}) != 1:
                ok = False
                print(f"  対照 NG: ctx={c} で条件間の出力が食い違う"
                      " (一致するはずの領域。測定は無効)")
        if ok:
            print("  対照 OK: 短文脈は条件間で出力が一致")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"\n書き出し: {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
