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

`bool-mask`  sdpa に渡す疎マスク。A は bool (2026-09-02 に既定化)、B は旧
             fp32 加算マスクの再現。**集合が同じなので品質は変わらない**。
             結果 (2026-09-01): 17k で ms/token -7% (`bench/results/bool-mask-17k.json`)。

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

`gdn-metal`  GDN の再帰を oMLX (jundot/oMLX) 移植の blocked-sequential Metal
             カーネルで解く (MLXTURBO_GDN_METAL、既定 on、2026-09-02〜、
             mlxturbo/kernels/gdn_blocked_metal.py)。`gdn-blocked` (行列積へ
             作り替え) とは別物で、逐次版と**同じ再帰をそのまま**計算する
             (k/q の threadgroup ステージングと状態のレジスタ常駐だけが違う)。
             **prefill 幅 (T >= 64)、Dk == 128、Dv % 32 == 0 のみが対象**で、
             decode/verify 幅は両者とも逐次カーネルに落ちる --
             **したがって --long で prefill を見ないと差が出ない。**
             加算順が変わるので出力一致は要求しない。
             合格条件: prefill_s の改善に加え、tok/round と KLD が悪化
             しないこと。
             結果 (2026-09-02): 17k prefill_s -1.3〜-4.5%、KLD 0.01312 ->
             0.01326 (受け入れ幅 +0.0005 の中)。既定 on にした。

             発火の確認: `mlxturbo.kernels._fire.snapshot()` の `gdn_metal`。

`fast-rope`  QK-norm 後の rope (cos/sin 生成 + `_rope_partial` x2、層あたり
             cos 1 / sin 1 / concatenate 6) を `mx.fast.rope` 1 dispatch x2
             (q/k) に畳む (MLXTURBO_FAST_ROPE、既定 off、
             `Attention._qkv` の `_fast_rope` 分岐)。CPU 上の合成入力では
             offset+arange(S) の位置に対して浮動小数の丸み差だけで一致
             することを確認済み (`docs/research/KERNEL-BRIEF-DECODE-BW.md`)。
             出力はビット不一致 (積和の順が変わる) なので出力一致は要求
             しない。バッチ経路 (`Attention._positions` が差し替わっている
             間) は実行時ガードで素の経路に落ちるため、この knob は
             batch/batch_spec 経路には効かない。
             合格条件: **ms/token が短・長の両方で改善すること。**
             どちらかで悪化したら既定 off のまま据え置く。

`mtp-append` 独立レビュー A-1 の修正 (MLXTURBO_MTP_CACHE_APPEND、既定 on、
             `spec_flash._prime_accepted_gap`)。A = 検証で確定した中間
             トークンを MTP キャッシュへ積む (既定) / B = 積まない
             (修正前、`_draft_chain` が毎ラウンド cur 1 列まで trim して
             戻すぶんが埋まらないまま)。B では MTP の offset (RoPE 位置)
             が毎ラウンド hit ぶん遅れ、受理率が生成長に比例して落ちる。
             出力トークン列は不変のはず (greedy はトランクの検証 logits
             からしか出ない。合成モデルの CPU テストで確認: 同じ ids で
             A/B を回してトークン列が一致すること)。ここでは
             `control_identical=False` -- draft 自体は変わるので、途中の
             `nxt_all`/`dv` 比較 (`_verify` の一致判定はドラフト側の値も
             使う) はビット同一を要求しない。
             合格条件: **tok/round (複数プロンプト x 512 の平均) が
             改善すること。**上がらなければ既定 off に戻す。ms/token も
             併せて見る。
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


def _knob_fast_rope(ctx):
    """A = QK-norm 後の rope を mx.fast.rope 1 dispatch x2 (q/k) に畳む
    (MLXTURBO_FAST_ROPE、既定 off) / B = 素の cos/sin 生成 + _rope_partial x2
    (既定)。CPU 上の合成入力では offset+arange(S) の位置に対して浮動小数の
    丸み差だけで一致することを確認済み (`docs/research/KERNEL-BRIEF-DECODE-BW.md`)。
    出力はビット不一致 (積和の順が変わる) なので出力一致は要求しない。
    合格条件: **ms/token が短・長の両方で改善すること。**どちらかで悪化
    したら既定 off のまま据え置く。tok/round も併せて見る (受理率が動く
    経路ではないはずだが、他の knob で受理率が変わって差し引き負けた前例
    が複数あるため)。"""
    import os

    from mlxturbo import fused

    os.environ["MLXTURBO_FAST_ROPE"] = "1"  # enable 側のゲートを開ける
    eng = ctx["eng"]

    def apply(variant):
        if variant == "A":
            fused.enable_fast_rope(eng.model)
        else:
            fused.disable_fast_rope(eng.model)

    return apply


def _knob_gdn_prework(ctx):
    """A = GDN 前処理の融合カーネル on / B = off (既定)。

    model を渡すのは、A_log/dt_bias が bf16 で読み込まれた実モデルでも
    fp32 の写しを作って eligible() の dtype 判定を通すため
    (mlxturbo/fused.py の enable_gdn_prework_kernel docstring 参照。
    2026-09-02、実機ログで発火 0 が判明した分の修正)。"""
    import os

    from mlxturbo import fused

    os.environ["MLXTURBO_GDN_PREWORK"] = "1"  # enable 側のゲートを開ける
    eng = ctx["eng"]

    def apply(variant):
        if variant == "A":
            fused.enable_gdn_prework_kernel(eng.model)
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


def _knob_gdn_metal(ctx):
    """2026-09-02 に既定 on にした (17k prefill_s -1.3〜-4.5%、KLD +0.00014)。
    A = GDN の再帰を oMLX 移植の blocked-seq Metal カーネルで解く (既定) /
    B = 逐次カーネル (旧既定)。

    A は oMLX (jundot/oMLX) の `gated_delta_blocked_seq` (kernel S) を移植
    したもの (`mlxturbo/kernels/gdn_blocked_metal.py`)。`gdn-blocked` (行列積
    への作り替え) と違い、逐次版と**同じ再帰をそのまま**計算する。
    **prefill 幅 (T >= 64)、Dk == 128、Dv % 32 == 0 のみが対象**で、
    decode/verify 幅は両者とも逐次カーネルに落ちる -- **したがって --long で
    prefill を見ないと差が出ない。**加算順が変わるので出力一致は要求しない。

    合格条件: **17k の prefill 壁時計 (prefill_s)。**tok/round と KLD も
    併せて見る (prefill の数値を変えたカーネルが受理率を落として差し引きで
    負けた前例がある)。

    発火の確認: `mlxturbo.kernels._fire.snapshot()` の `gdn_metal`。
    """
    import os

    from mlxturbo import fused

    def apply(variant):
        if variant == "A":
            os.environ.pop("MLXTURBO_GDN_METAL", None)  # 既定 on のゲートを開けたまま
            fused.enable_gdn_metal_kernel()
        else:
            os.environ["MLXTURBO_GDN_METAL"] = "0"  # off 側のゲートを閉じる
            fused.disable_gdn_metal_kernel()

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



def _knob_hc_prefill(ctx):
    """A = prefill 用の 1 ディスパッチ版 / C = 機構だけ / B = 素 (既定)。

    `mlxturbo/kernels/hyper_connection.py` の `fused_gated_residual_prefill` は
    prefill 幅で hyper-connection の読みを 1 ディスパッチに畳む。**既定 off で、
    `CLAUDE.md` に「in-model 実測で負けたから off」と記録されている。**

    **その判定はハーネスを直す前のもの。**2026-09-02 に長文脈の A/B が A 側に
    +5.6% の下駄を履かせていたことが分かり、`rms_norm_gated` と `moe_route` も
    同じ理由で測り直した (どちらも結論は変わらなかったが、値は動いた)。
    ここも同じ扱いで測り直す。

    段 P0 の実測では **HC は prefill の 10%、効率 52-55% で prefill 最低。**
    伸びしろは 17k 全体で 1.65s。

    有効化のゲートは `enable_hyper_connection_kernel` が起動時に 1 回だけ
    `MLXTURBO_HC_PREFILL` を読む形なので、env を差し替えてから貼り直す。
    C は env を off にしたまま貼り直すだけで、**差し替えの機構は A と同じ**。

    prefill に効くので `--prefill-once` は使えない (`DECODE_ONLY_KNOBS` の外)。
    見るのは `prefill_s`。
    """
    import os

    from mlxturbo import fused

    def apply(variant):
        fused.disable_hyper_connection_kernel()
        os.environ["MLXTURBO_HC_PREFILL"] = "1" if variant == "A" else "0"
        if variant != "B":
            fused.enable_hyper_connection_kernel()
        # B は読み側の融合カーネルごと外した素の経路

    return apply



def _knob_pipeline(ctx):
    """A = 楽観パイプライン (次ラウンドの draft を先に組む) / B = 無効 (既定)。

    `spec_flash.generate_stream` が毎ラウンド `MLXTURBO_PIPELINE` を読むので、
    env を差し替えるだけで切り替わる (1=通常、2=組むが毎回捨てる切り分け用、
    0=無効)。

    **`CLAUDE.md` に「楽観先組みは 3 回失敗して棄却済み」と記録されている。**
    「作って捨てる」遅延グラフは規模を問わず MLX の暗黙 eval に罰される、という
    決定則の根拠になっている項目。**その判定は 2026-09-02 に直したハーネス
    (長文脈で A 側に +5.6% の下駄) より前のもの**なので測り直す。

    ただし前提が構造的 (捨てるグラフを MLX が罰する) なので、下駄の 5.6% で
    ひっくり返る種類ではない可能性が高い。**それでも数字を持っておく価値はある。**

    貪欲なので出力は変わらない。判定は ms/round と tok/round。
    """
    import os

    def apply(variant):
        os.environ["MLXTURBO_PIPELINE"] = "1" if variant == "A" else "0"

    return apply



def _knob_fast_qmm(ctx):
    """A = 検証フォワードの密 qmm を MMA タイルへ / B = stock (既定)。

    `mlxturbo/fast_qmm.py`。記録は **「M=3 で 164GB/s、fast_qmm が適格なのに
    in-model で負ける」という未解決の謎** (`docs/BACKLOG.md`)。単体では
    qkv -22% / lm_head -33% が出るのに、モデルに入れると負ける。

    **そして前提が変わっている。**記録に「depth 既定が 1 に変わって M=2 に
    なったので前提そのものが変わっている」とある。窓の下限は `M_MIN=3` で、
    **M=2 なら fast_qmm はそもそも発火しない。**

    つまり見るべきは 2 つ:
    1. いまの既定 (17k で depth 1 = M 2) で**発火するのか** — しないなら
       この knob は何もしない knob で、A/B は 0 になるはず
    2. `M_MIN=2` まで下げたときに勝つのか — 記録は「M=2 は stock が勝つ」

    ここでは 1 だけを見る。**A が 0 なら「発火していない」ことの確認**で、
    2 の測定は別途 `MLXTURBO_QMM_M_MIN` を振る話になる。

    貪欲なので出力は変わらない。判定は ms/round。
    """
    from mlxturbo import fast_qmm as fq

    def apply(variant):
        fq.disable()
        if variant == "A":
            fq.enable(ctx["eng"].model)

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
    """2026-09-02 に既定を bool にした。A = bool (新既定) / B = 加算 (旧経路の再現)。

    `Attention._final_mask` は `where(sparse, 0, -inf)` で fp32 の加算マスクを
    作って sdpa に渡していたが、MLX 0.32.2 の sdpa vector カーネルは加算値が
    `finfo.min` (= finite_min) だと `>=` 判定でキー読み込みをスキップできず、
    bool マスクの `bmask[0]` 判定だけがスキップする (17k で全キーを読んでいた)。
    **集合はまったく同じなので品質は変わらない。**変わるのはマスクの表現だけ。

    QSA を切る実験 (`--knob qsa`) と同じ「マスクが重いのか」を測るが、
    こちらは**品質が動かないのでそのまま採用できる**。
    """
    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q

    orig = Q.Attention._final_mask  # 既定 = bool (2026-09-02〜)

    def additive_mask(self, mask, sparse, cache, S, dtype):
        # 旧経路の再現: fp32 の加算マスク (0 / finfo.min)
        if sparse is None:
            return mask
        neg = mx.finfo(dtype).min if hasattr(mx, "finfo") else -1e9
        add = mx.where(sparse, mx.array(0, dtype), mx.array(neg, dtype))
        if mask is None or isinstance(mask, str):
            return add
        return mask + add

    def apply(variant):
        Q.Attention._final_mask = orig if variant == "A" else additive_mask

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


def _knob_fold_tail(ctx):
    """端数チャンクをレイヤー主導グループに畳み込むか (b80d7e2)。
    A = 畳む (既定) / B = 畳まない (commit 前の chunk-major 単独)。

    17k (16869 tok) のフェーズ別トレースで、端数チャンク 485 tok が
    4.49 ms/tok (グループは 2.45-2.79 ms/tok) だったのが動機。畳み込みは
    チャンク境界 (grid) を変えないので出力はビット一致するはず。
    判定は prefill_s。
    """
    import mlxturbo.spec_flash as SF

    def apply(variant):
        SF._PREFILL_FOLD_TAIL = variant == "A"

    return apply


def _knob_mtp_append(ctx):
    """独立レビュー A-1 の修正。A = 検証で確定した中間トークンを MTP
    キャッシュへ積む (既定) / B = 積まない (修正前の挙動)。

    `spec_flash._MTP_CACHE_APPEND` を直接差し替える (env は import 時にしか
    読まれないので、他の module-level knob と同じくここで属性を書き換える)。
    """
    import mlxturbo.spec_flash as SF

    def apply(variant):
        SF._MTP_CACHE_APPEND = variant == "A"

    return apply


def _knob_ngram_layout(ctx):
    """n-gram サイドカーのレイアウト。A = interleaved (`StreamNGram`、ディスク
    参照) / B = separate (`RamNGram`、RAM 常駐)。

    本番 (HTTP サーバー、50k を通す構成) は interleaved。`StreamNGram.__call__`
    は先頭で `np.array(gid.reshape(-1))` が GPU→CPU 同期を起こし、行ごとに
    `os.pread` を ThreadPoolExecutor に投げる。`RamNGram` は連結テーブルを
    RAM に置いて `mx.take` 1 回で読む (GPU 上の gather)。decode の差は未測
    (`docs/BACKLOG.md` の「n-gram サイドカーのレイアウト選択が 50k を殺して
    いた」)。

    A 側は `--ngram` (layout=interleaved)、B 側は `--ngram-b`
    (既定 `~/models/ddalcu-ngram-sep`、layout=separate)。どちらも同じ 4bit
    行を dequantize するだけなので、manifest の bits/group_size が一致して
    いれば**値はビット一致するはず** (`control_identical=True` で短文脈の
    出力一致を検査する)。一致しなければ setup 時点でヘッダに警告を出す。

    切り替えは `mlxturbo.ngram_stream.install(model, path)` を呼んで PLE 層の
    `ngram_embedding` を差し替える (install は再呼び出し可能: 単に
    `emb.ngram_embedding = ...` を代入するだけ)。install/install_ram は内部で
    `StreamNGram(path)`/`RamNGram(path)` を毎回素で作り直すので、そのまま
    呼ぶと B に切り替えるたびに 32GB を読み直してしまう。**両方のインスタンス
    を控えて使い回す**ために、コンストラクタだけをパスごとに使い回す
    薄いサブクラスに差し替える (install 自体の中身・print・戻り値は変えない)。
    B への初回切り替えだけ 32GB の読み込みで 20 秒級、以降 (A に戻すときも
    含め) は代入だけなので即時。

    合格条件: **ms/token の差が A/B で 3% 以上なら、interleaved の同期
    (`StreamNGram.__call__` の `np.array(gid)`) を疑う。tok/round は同一の
    はず** (どちらも同じ行を dequantize するだけで、受理率には関与しない)。
    """
    import json
    from pathlib import Path

    from mlxturbo import ngram_stream as NS

    args = ctx["args"]
    model = ctx["eng"].model
    if not args.ngram:
        raise ValueError("ngram-layout には --ngram (A 側、layout=interleaved) が要る")
    path_a = Path(os.path.expanduser(args.ngram))
    path_b = Path(os.path.expanduser(args.ngram_b))

    man_a = json.loads((path_a / "manifest.json").read_text())
    man_b = json.loads((path_b / "manifest.json").read_text())
    for key in ("bits", "group_size"):
        if man_a.get(key) != man_b.get(key):
            print(f"[ngram-layout] 警告: manifest の {key} が A/B で異なる"
                  f" (A={man_a.get(key)!r} B={man_b.get(key)!r})。"
                  " 出力一致の対照は成立しない可能性がある")

    # パスごとに 1 インスタンスだけ作って使い回す。__new__ でキャッシュを
    # 引き、__init__ は初回だけ本体を走らせる (2 回目以降は何もしない)。
    # サブクラスなので isinstance(x, StreamNGram/RamNGram) は変わらず通る。
    cache: dict[Path, object] = {}

    class _CachedStream(NS.StreamNGram):
        def __new__(cls, sidecar, *a, **kw):
            hit = cache.get(Path(sidecar))
            return hit if hit is not None else super().__new__(cls)

        def __init__(self, sidecar, *a, **kw):
            p = Path(sidecar)
            if p in cache:
                return
            super().__init__(sidecar, *a, **kw)
            cache[p] = self

    class _CachedRam(NS.RamNGram):
        def __new__(cls, sidecar, *a, **kw):
            hit = cache.get(Path(sidecar))
            return hit if hit is not None else super().__new__(cls)

        def __init__(self, sidecar, *a, **kw):
            p = Path(sidecar)
            if p in cache:
                return
            super().__init__(sidecar, *a, **kw)  # B の初回だけ 32GB を読む
            cache[p] = self

    NS.StreamNGram = _CachedStream
    NS.RamNGram = _CachedRam

    def apply(variant):
        NS.install(model, path_a if variant == "A" else path_b)

    return apply


def _knob_ngram_prefetch(ctx):
    """n-gram サイドカーの先読み (`StreamNGram.prefetch`)。A = 有効 (既定) /
    B = 無効。

    prefill はプロンプト全体の n-gram 行 id が最初から全部わかっている。
    GPU が chunk 0 を計算している間に CPU 側でディスクから先読みしておけば
    (`mlxturbo/spec_flash.py` の `_prefetch_ngram_rows`、`generate_stream` の
    prefill ループ直前で 1 回呼ぶ)、後続チャンクの `StreamNGram.__call__` が
    キャッシュヒットで待たずに返る。温キャッシュの pread 自体も 17k で
    約 2.5s、50k で約 7s が GPU が止まったまま CPU で消えている実測がある
    (行 1 つ 7.6us、prefill 1 チャンク 32768 行で約 250ms)。

    環境変数 (`MLXTURBO_NGRAM_PREFETCH`) は使わない。あちらは `StreamNGram`
    インスタンスの初期値を決めるだけで、1 プロセス内で A/B を交互に取る
    このハーネスからは (インストール後に) 直接インスタンスの
    `prefetch_enabled` を切り替えるほうが確実。**`--ngram` で install 済みの
    StreamNGram を直接触るので、B 側でもプロセスは張り直さない。**

    prefetch はキャッシュを温めるだけで `__call__` が返す値は変えない
    (ヒットでもミスでも同じ行を返す) ので出力は一致するはず
    (`control_identical=True`)。判定は **prefill_s** (prefill-group knob と
    同じ扱い)。`--prefill-once` とは併用できない (prefill 自体を先読みごと
    畳んでしまうと A/B の差が消える) ので `DECODE_ONLY_KNOBS` には入れない。

    `StreamNGram.prefetch_enabled` の既定は off (`MLXTURBO_NGRAM_PREFETCH=1`
    で明示的に有効化しない限り on にならない)。17k の in-model A/B で先読みの
    取り分が 0% だったため (2026-09-02)。
    """
    model = ctx["eng"].model
    if not ctx["args"].ngram:
        raise ValueError("ngram-prefetch には --ngram (StreamNGram の install) が要る")
    streams = []
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        stream = ple.ple_embedding.ngram_embedding
        if not hasattr(stream, "prefetch_enabled"):
            raise ValueError(
                "ngram_embedding が StreamNGram ではない (--ngram の manifest が"
                " layout=separate だと RamNGram になり prefetch を持たない)"
            )
        streams.append(stream)
    if not streams:
        raise ValueError("PLE 層が見つからない")

    def apply(variant):
        for s in streams:
            s.prefetch_enabled = variant == "A"

    return apply


def _knob_ngram_batch(ctx):
    """`StreamNGram._gather_pread` のバッチ化 pread 自体の取り分。
    A = 既定 (`batch_min_rows=64`、64 行以上はスライス分割 + 並列 pread) /
    B = `batch_min_rows=10**9` (常に行ごとに future を 1 つ submit する
    旧経路)。

    17k の ngram-prefetch A/B で差が 0% だった (B もバッチ化 pread を含む
    ため)。prefetch の発火有無とバッチ化自体の取り分が分かれていなかった
    ので、こちらでバッチ化だけを切り出す。prefetch は両側とも `--ngram`
    install 時点の既定 (on) のまま触らない。

    行の中身は変えないので出力は一致するはず (`control_identical=True`)。
    判定は prefill_s (n-gram lookup は prefill のチャンクで大量の行数を
    まとめて叩くので、バッチ化の差はここに出るはず。decode の 48 行は
    どちらの分岐でも同じ「行ごと submit」経路を通るので差が出ない)。
    """
    model = ctx["eng"].model
    if not ctx["args"].ngram:
        raise ValueError("ngram-batch には --ngram (StreamNGram の install) が要る")
    streams = []
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        stream = ple.ple_embedding.ngram_embedding
        if not hasattr(stream, "batch_min_rows"):
            raise ValueError(
                "ngram_embedding が StreamNGram ではない (--ngram の manifest が"
                " layout=separate だと RamNGram になり batch_min_rows を持たない)"
            )
        streams.append(stream)
    if not streams:
        raise ValueError("PLE 層が見つからない")

    def apply(variant):
        for s in streams:
            s.batch_min_rows = 64 if variant == "A" else 10**9

    return apply


KNOBS = {
    # name: (setup(ctx) -> apply(variant), variants, 出力一致を要求するか,
    #        まとめで基準にする variant)
    "qsa-tail": (_knob_qsa_tail, ["A", "B"], True, "B"),
    "moe-verify": (_knob_moe_verify, ["A", "B"], False, "B"),
    "fast-rope": (_knob_fast_rope, ["A", "B"], False, "B"),
    "gdn-prework": (_knob_gdn_prework, ["A", "B"], False, "B"),
    "gdn-blocked": (_knob_gdn_blocked, ["A", "B"], False, "B"),
    "gdn-metal": (_knob_gdn_metal, ["A", "B"], False, "B"),
    "hc-write": (_knob_hc_write, ["A", "C", "B"], True, "B"),
    "rms-norm-gated": (_knob_rms_norm_gated, ["A", "C", "B"], True, "B"),
    "moe-route": (_knob_moe_route, ["A", "C", "B"], False, "B"),
    "hc-prefill": (_knob_hc_prefill, ["A", "C", "B"], False, "C"),
    "pipeline": (_knob_pipeline, ["A", "B"], False, "B"),
    "fast-qmm": (_knob_fast_qmm, ["A", "B"], False, "B"),
    "null": (_knob_null, ["A", "B"], True, "B"),
    "indexer-cache": (_knob_indexer_cache, ["A", "B"], True, "B"),
    "pooled-cache": (_knob_pooled_cache, ["A", "B"], True, "B"),
    "stage-every": (_knob_stage_every, ["1", "2", "4"], True, "2"),
    "prefill-group": (_knob_prefill_group, ["2", "4", "8"], True, "4"),
    "prefill-pipeline": (_knob_prefill_pipeline, ["A", "B"], True, "B"),
    "fold-tail": (_knob_fold_tail, ["A", "B"], True, "A"),
    "qsa": (_knob_qsa, ["A", "B"], False, "A"),
    "bool-mask": (_knob_bool_mask, ["A", "B"], False, "B"),
    "gather-attn": (_knob_gather_attn, ["A", "B"], False, "B"),
    # -1 は gather 自体を切る (現行既定)。0 はタイルなしの gather
    "gather-tile": (_knob_gather_tile, ["-1", "0", "256"], False, "-1"),
    "prefill-attn": (_knob_prefill_attn, ["A", "B"], False, "B"),
    "wide": (_knob_wide, ["A", "B"], False, "B"),
    "depth": (_knob_depth, ["1", "2", "3"], False, "2"),
    "mtp-append": (_knob_mtp_append, ["A", "B"], False, "B"),
    # A = interleaved (本番既定) を基準に、B = separate (RAM 常駐) と比べる
    "ngram-layout": (_knob_ngram_layout, ["A", "B"], True, "A"),
    # A = 先読み有効 (既定) / B = 無効。判定は prefill_s
    "ngram-prefetch": (_knob_ngram_prefetch, ["A", "B"], True, "A"),
    # A = batch_min_rows=64 (既定) / B = 10**9 (常に行ごと旧経路)。判定は prefill_s
    "ngram-batch": (_knob_ngram_batch, ["A", "B"], True, "A"),
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


# ngram-prefetch / ngram-batch のときだけ StreamNGram.stats を run ごとに
# 拾う。install() は PLE 層全部に同じ 1 インスタンスを配るので、最初の 1 個
# を掴めば足りる
NGRAM_STATS_KNOBS = {"ngram-prefetch", "ngram-batch"}


def _ngram_stream_instance(model):
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        emb = ple.ple_embedding.ngram_embedding
        if hasattr(emb, "reset_stats"):
            return emb
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", required=True, choices=sorted(KNOBS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--ngram-b", default="~/models/ddalcu-ngram-sep",
                    help="ngram-layout knob の B 側 (layout=separate、RAM 常駐)。"
                         "A 側は --ngram (layout=interleaved) を使う")
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
    set_variant = setup({"eng": eng, "args": args})
    order = variants + variants[::-1]

    # ngram-prefetch / ngram-batch: 発火が実際にあるか (キャッシュ hit 率、
    # バッチ pread の分岐) を数字で見る。StreamNGram.stats を run ごとに
    # reset_stats() してから拾う
    ngram_stream = (
        _ngram_stream_instance(eng.model) if args.knob in NGRAM_STATS_KNOBS else None
    )

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
        # MLXTURBO_PIPELINE は generate_stream の decode ループの中でしか
        # 読まれない (spec_flash.py:1264)。prefill には触らない。
        "pipeline",
        # fast_qmm は検証フォワードの密 qmm だけを差し替える。prefill は
        # fast_qmm 自身の窓判定で素通りする (M_MIN=3..8 の窓)。
        "fast-qmm",
        # ngram-layout は PLE の n-gram 参照バックエンドを差し替えるだけ。
        # prefill 幅でも呼ばれるが、返す値は manifest が一致していればビット
        # 一致 (対照が効く) なので prefill_s は動かない。ステップ数に比例する
        # decode の呼び出し回数のところで初めて差が出る。
        "ngram-layout",
        # mtp-append (独立レビュー A-1) は generate_stream の decode ループ
        # 内、_verify の後でしか _MTP_CACHE_APPEND を読まない
        # (spec_flash._prime_accepted_gap の呼び出し口)。_prime_draft_cache
        # (prefill 側の priming) は触らない。
        "mtp-append",
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
            if ngram_stream is not None:
                ngram_stream.reset_stats()
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
            ngram_s = ""
            if ngram_stream is not None:
                rows[-1]["ngram"] = dict(ngram_stream.stats)
                ngram_s = "  " + ngram_stream.stats_line()
            print(f"  {v}: prefill {tp:6.2f}s  decode {td:6.2f}s  "
                  f"{ms:6.2f} ms/tok  tok/round {tpr:.3f}  "
                  f"({acc}/{rounds}){fired_s}{ngram_s}", flush=True)
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

    # ngram-prefetch / ngram-batch: StreamNGram.stats の合計 (発火の確認)。
    # hit/miss は「先読みが実際に効いているか」、sync_ms/fetch_ms は
    # 「バッチ化自体の取り分」を切り分けるためのもの
    if ngram_stream is not None:
        for kind in ("short", "long"):
            sub = [r for r in rows if r["kind"] == kind and "ngram" in r]
            if not sub:
                continue
            for metric in ("hits", "misses", "sync_ms", "fetch_ms"):
                totals = {}
                for v in variants:
                    vals = [r["ngram"][metric] for r in sub if r["variant"] == v]
                    totals[v] = sum(vals)
                base = totals[baseline]
                if base == 0:
                    cells = "  ".join(f"{v}={totals[v]:10.2f}" for v in variants)
                    print(f"  {kind:5s} ngram_{metric:9s} {cells}"
                          f"   [基準 {baseline} が 0 なので比は無し]")
                    continue
                cells = "  ".join(
                    f"{v}={totals[v]:10.2f}"
                    f"({(totals[v] - base) / base * 100:+5.1f}%)"
                    for v in variants
                )
                print(f"  {kind:5s} ngram_{metric:9s} {cells}   [基準 {baseline}]")

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
    # 計測ツールなので destructor (スレッドプール等の後始末) に用は無い。
    # interpreter shutdown 待ちでプロセスが Metal のメモリを握ったまま
    # 1 時間以上残った実測があるので、結果を書き終えたら即 _exit で落とす
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
