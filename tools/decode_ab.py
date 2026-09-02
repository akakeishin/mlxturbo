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

`indexer-lean` QSA indexer の decode/verify 幅 (S<=8) 費用を減らす
             (MLXTURBO_INDEXER_LEAN、既定 off、`mlxturbo/indexer_lean.py`)。
             `docs/research/SESSION-2026-09-02-CATCHUP.md` 末尾: S=2 の
             attention 1 層で indexer が 228〜307us と sdpa 本体 (71us) の
             3 倍、12 層で 2.7 ms/round。`_IndexerCache.block_grid`/
             `.pooled_fp32` が、ブロックが新しく確定した回だけ変わる 2 つ
             (block_starts/block_end、pooled の fp32 キャスト) を n_blocks
             (または `_pooled_n`) が前回と同じならキャッシュから返す。
             `tools/indexer_ops.py` (CPU、合成モデル) の実測: 定常状態
             (ブロック境界をまたがない decode ラウンド) で 1 層あたり
             mx/mx.array ディスパッチ -5 (-8〜8.5%、内訳は arange 1 +
             array.__mul__ 1 + array.__add__ 1 + array.__sub__ 1 +
             array.astype 1)。境界をまたぐ回 (compress_ratio/S 回に 1 回)
             は base と同数 (delta 0、退化なし)。**値は変えない**
             (`indexer_ops.py` が lean on/off の `__call__` 返り値とキャッシュ
             状態のビット一致を検査する)。prefill 幅 (S>8) では `lean` が
             常に False になるので経路が変わらない (`DECODE_ONLY_KNOBS`)。
             合格条件: **ms/token が短・長の両方で改善すること** (出力は
             ビット同一のはずなので対照が効く --- control_identical=True)。
             in-model の壁時計 A/B は未実施 (このファイルの変更時点では
             CPU 検査のみ)。

`draft-rerank` レーン11 仮説5 の裏取り: MLXTURBO_DRAFT_RERANK (既定 1、
             `spec_flash.FlashSpecEngine._build_rerank`) が受理率を静かに
             削っていないか。A = trunk lm_head の 2bit 粗ヘッドで全語彙を
             読んでから正確な top-32 だけ再採点する (既定)。B = 粗ヘッドを
             経由せず trunk ヘッドの argmax をそのまま draft にする
             (MLXTURBO_DRAFT_RERANK=0 相当)。env var はエンジン構築時にしか
             読まれないので、A/B は `eng._rerank` を直接付け替えて切る
             (`depth` knob と同じ流儀 --- 粗ヘッドの構築はやり直さず、A の
             タプルを退避して挿し戻すだけ)。draft の argmax が top-32 の
             外れで trunk 直読みと食い違いうるので厳密一致は要求しない
             (`control_identical=False`)。合格条件: **tok/round** (複数
             プロンプト x 512 の平均) が rerank off で有意に上がったら、
             rerank が受理率を削っている証拠。`--draft-trace` を足すと
             `MLXTURBO_DRAFT_TRACE=1` が立ち、1 段目の draft top-1/top-2 と
             検証済みの真の次トークンを突き合わせた hit@1/hit@2 が
             `fired` (`draft_trace_rounds`/`draft_trace_hit1`/
             `draft_trace_hit2`) に乗る --- こちらは仮説7 (木化ドラフトの
             上限 = hit@2 - hit@1) の裏取り用で、rerank の on/off どちらでも
             測れる (top-2 の取り方が rerank の有無で変わるだけ、
             `spec_flash.FlashSpecEngine._draft_argmax` 参照)。

`temp`       レーン11 仮説6 の裏取り: 現行の temp>0 投機 (`spec_flash._verify`
             の `temp > 0 or sampler is not None` 分岐 --- verify の logits
             からサンプルし、greedy な draft と一致したら採用) は分布としては
             正しい (`verify_spec_sampling.py` が検証済み) が、真の棄却
             サンプリング (受理確率 min(1, p/q)、棄却時は残差分布から再サンプル)
             より受理率が低いはず、という仮説。variant の値をそのまま
             `generate_stream(..., temp=...)` の温度として使う (`--temp`
             フラグ自体は無視され、variant が勝つ)。既定の対照 (`--variants`
             省略時) は 0.0 (greedy) 対 0.7。**サンプリングなので出力は
             temp>0 側で毎回変わりうる --- `control_identical=False`。**
             `--seed` で `mx.random.seed` を固定すれば同じ乱数列にはなるが、
             それでも greedy 側と一致することは期待しない。合格条件は無い
             (探索用の道具): **tok/round** (複数プロンプト x 512 の平均) を
             `--knob null --temp 0.7` (ハーネス自身のばらつき、両側 temp 0.7)
             と比べて、`temp` knob の 0.0→0.7 の落ち幅がそれより大きいかを見る。
             prefill には触らない (`_sample` は verify 後の 1 トークン目/2
             トークン目だけで、prefill のチャンクループはそれを通らない) ので
             `DECODE_ONLY_KNOBS` に入れてある --- `--prefill-once` が使える。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
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


def _knob_ple_hoist(ctx):
    """PLE (n-gram) 埋め込みの層ループ前一括計算 (`MLXTURBO_PLE_HOIST`、既定
    off)。A = 有効 / B = 無効 (既定)。

    PLE の入力は隠れ状態 h に依らず ids と直前文脈だけで決まる
    (`mlxturbo/_vendor/qwen4_exp.py` の `PLELayer.__call__`) ので、48 層の
    ループへ入る前にまとめて計算できる。素の経路は PLE 層 (Flash-Next で
    5 層) それぞれの `ple_embedding(ids, prev_ctx)` 呼び出しが n-gram
    テーブル側の GPU->CPU 同期 (`StreamNGram.__call__` の
    `np.array(gid.reshape(-1))`) を挟み、それが `_staged_forward` の 2 層
    ごとの async_eval 投入をその境界で断ち切っていた。まとめて計算すれば、
    サイドカーが全 PLE 層で共有されている構成 (`--ngram` install 後) では
    その同期は 1 forward で 1 回になる。

    `ngram-prefetch` (先読みでキャッシュを温める) とは別の仕組み --- あちらは
    ディスク I/O の待ちを隠す狙いで、17k の in-model A/B で取り分 0% だった
    (`docs/research/SESSION-2026-09-02-CATCHUP.md`)。こちらは I/O 待ちでは
    なく、同期そのものが `_staged_forward` の async_eval 投入を断ち切る回数を
    減らす狙いなので、別の勝ち筋のはず --- ただし実測は本 knob で取ること。

    テーブル呼び出しの入出力は素の経路と同一の値を通すだけなので出力は
    一致するはず (`control_identical=True`)。prefill (chunk-major の末尾
    チャンク、`Qwen4ExpModel.__call__` 経由) にも効くので `DECODE_ONLY_KNOBS`
    には入れない (`_group_prefill_forward` の layer-major 経路は `_prelude`
    を呼ばないので対象外のまま、ビットは動かない)。
    """
    from mlxturbo import fused

    os.environ["MLXTURBO_PLE_HOIST"] = "1"  # enable 側のゲートを開ける
    eng = ctx["eng"]

    def apply(variant):
        if variant == "A":
            n = fused.enable_ple_hoist(eng.model)
            if n == 0:
                raise ValueError("ple-hoist: PLE 層が見つからない (PLE 無しのモデル?)")
        else:
            fused.disable_ple_hoist(eng.model)

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


def _knob_depth_adapt(ctx):
    """A = 受理率 EMA で depth を選ぶ適応 (`MLXTURBO_DEPTH_ADAPT=1`
    相当) / B = 既定の静的規則 (`choose_depth`、文脈長だけで決める)。

    レーン10 (docs/research/LANES-2026-09.md)。貪欲なので出力トークン列は
    A/B で原則一致するはず -- verify は本体の argmax と一致したときだけ
    受理するので、depth の選び方自体は採否の基準を変えない。ただし `depth`
    knob と同じ理由で、verify 幅が変わると `mx.quantized_matmul` の幅依存の
    丸めで argmax がまれに割れる (厳密一致は要求しない)。

    ラウンドごとに選んだ depth の分布は `mlxturbo.kernels._fire`
    (`depth_adapt_<m>`) に積まれ、結果 JSON の `fired` にそのまま出る
    (harness 側の仕組みをそのまま使う。既存の knob と同じ)。

    A に切り替えるたびに controller を作り直す (`DepthController` は
    エンジンの生存期間中ずっと学習を持ち回る設計だが、A/B の回文順で前の
    A 実行の学習が残ると 2 回目の A が不公平に有利になるので、ここでは
    `_fire.reset()` と同じ「実行のたびに測り直す」に合わせてある)。
    """
    from mlxturbo.spec_flash import DepthController

    def apply(variant):
        eng = ctx["eng"]
        eng._depth_adapt = variant == "A"
        if eng._depth_adapt:
            eng._depth_controller = DepthController()

    return apply


def _knob_draft_rerank(ctx):
    """A = draft-rerank on (既定、trunk lm_head の 2bit 粗ヘッドで全語彙を
    読んでから正確な top-32 だけ再採点) / B = off
    (MLXTURBO_DRAFT_RERANK=0 相当、trunk ヘッドの argmax をそのまま draft
    にする)。

    `MLXTURBO_DRAFT_RERANK` は `FlashSpecEngine.__init__` (`_build_rerank`)
    でしか読まれず、粗ヘッドの構築 (dequantize→requantize) はやり直すと
    重いので、env ではなく `eng._rerank` を直接付け替える (`depth` knob と
    同じ流儀)。A のタプルは構築済みのものを退避して挿し戻すだけ --- B は
    `_rerank = None` にして `_draft_argmax` の rerank なし分岐 (trunk
    ヘッドの argmax) へ落とす。

    draft の argmax が変わりうる (rerank の粗い top-32 に真の argmax が
    入らない語で trunk 直読みと食い違いうる) ので厳密一致は要求しない。
    判定は **tok/round** (受理率、複数プロンプト x 512 の平均) --- rerank
    off で有意に上がるなら、rerank が受理率を静かに削っている証拠
    (レーン11 仮説5)。
    """
    eng = ctx["eng"]
    saved = eng._rerank

    def apply(variant):
        if variant == "A" and saved is None:
            raise ValueError(
                "draft-rerank: eng._rerank が構築されていない"
                " (lm_head 無し=tie埋め込みか非量子化パック?)"
            )
        eng._rerank = saved if variant == "A" else None

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



def _knob_temp(ctx):
    """variant の値をそのまま温度として使う (`--knob temp --variants 0.0,0.7`)。

    詳細はモジュール docstring の `temp` 節を参照。`--temp` フラグの値は
    ここでは使わない (常に variant で上書きされる) --- `run_once`/`run_resumed`
    は呼び出し側 (`main`) が `ctx["args"].temp` を読んで
    `generate_stream(..., temp=...)` に渡す。
    """
    args = ctx["args"]

    def apply(variant):
        args.temp = float(variant)

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



def _knob_hc_kernel(ctx):
    """A = hyper-connection の読み側 (`GatedResidual.__call__`) を融合 Metal
    カーネルに畳む (既定 off) / B = 素の実装 (既定)。

    `mlxturbo/kernels/hyper_connection.py` の `fused_gated_residual`
    (`fused.enable_hyper_connection_kernel`)。診断
    (scratchpad/hc_fire_diag.py、実モデル ~/models/ddalcu-mlxlm、S=1) で、
    97 層の `GatedResidual` のうち 96 層は `block_inject_weight` が
    QuantizedLinear に変換されず bf16 の `nn.Linear` のまま残っていて、
    `fused.py` の `_pack_quantized` が None を返すため inject の量子化
    5-tuple パックに失敗し、**down/up が量子化で問題なくてもカーネル全体が
    毎回素の実装へ落ちていた**(発火は inject の無い 1 層 (mixer) だけ)。
    `kernels/hyper_connection.py` に非量子化 bf16 inject をそのまま読む分岐
    (`_pre_source` の "bf16" ケース、`eligible`/`fused_gated_residual` の
    `inject_kind`) を足し、`fused.py` の `_pack_inject_bf16` がそれを渡す
    ようにしたことで 97 層全部が発火するはず
    (bench/test_hc_kernel_inject.py がモデル無しの合成入力で素の計算との
    一致を確認済み。最大絶対誤差は bf16 の丸め幅 1.5e-2 以内 -- sigmoid が
    ビット単位で再現できないのはこのカーネルの既知の性質で、
    hyper_connection.py 冒頭の説明を参照)。

    **decode 専用ではない。**`hck.eligible` は M (行数) を一切見ないので、
    `MLXTURBO_HC_PREFILL` が既定 off でも `enable_hyper_connection_kernel`
    を呼べば、prefill 幅の呼び出しもこの同じ decode 幅カーネル (pre_tg=21 /
    post_tg=80 の設計、`_prefill_source` の docstring が言う「M が大きい
    prefill では threadgroup 数は M だけで GPU を埋められる」の経路) を通る。
    つまりこの knob は prefill にも効くので **`--prefill-once` は使えない**
    (`DECODE_ONLY_KNOBS` には入れない -- `bool-mask`/`gdn-metal` もそちらの
    リストには入っていない。prefill 幅専用の 1 ディスパッチ版は別の knob
    (`hc-prefill`) が担当する)。

    合格条件: **ms/token が短・長の両方で改善すること。**出力は sigmoid の
    1 ulp 差で厳密には一致しないので一致は要求しない。
    発火の確認: `mlxturbo.kernels._fire.snapshot()` の `hc_kernel`
    (層数ぶん、既定モデルなら 97 になるはず)。
    """
    from mlxturbo import fused

    def apply(variant):
        fused.disable_hyper_connection_kernel()
        if variant == "A":
            fused.enable_hyper_connection_kernel()

    return apply



def _knob_hc_compiled(ctx):
    """A = hyper-connection の読み側を mx.compile で 1 グラフに畳む
    (`fused.enable_hyper_connection`、`MLXTURBO_HC=compiled` 相当) / B = 素の
    実装 (既定は `hc-kernel` の融合 Metal カーネルだが、この knob の B は
    `disable_hyper_connection` と `disable_hyper_connection_kernel` を両方
    呼んで `GatedResidual.__call__` を素の実装に戻した状態)。

    `hc-kernel` (Metal カーネル、sigmoid が bf16 1 ulp でビット一致しない) と
    違い、compiled 版は重みを引数で渡す mx.compile の記録なので**素の実装と
    ビット同一**(`enable_hyper_connection` の docstring)。対照 (出力一致) が
    そのまま効く。

    **decode 専用ではない。**`patched` は M (行数) を一切見ないので prefill
    幅の呼び出しにも同じグラフがかかる。つまりこの knob は prefill にも
    効くので **`--prefill-once` は使えない** (`DECODE_ONLY_KNOBS` には入れ
    ない -- `hc-kernel` と同じ理由)。

    合格条件: **ms/token が短・長の両方で改善すること。**出力はビット同一
    なので一致することを確かめる (割れたら測定は無効)。

    `enable_hyper_connection` は `_ORIG_HC is not None` なら何もしないので、
    切替のたびに両方の disable を先に呼んでから A なら enable する
    (`hc-kernel` の `apply` と同じ書き方)。
    """
    from mlxturbo import fused

    def apply(variant):
        fused.disable_hyper_connection()
        fused.disable_hyper_connection_kernel()
        if variant == "A":
            fused.enable_hyper_connection()

    return apply


def _knob_hc_prefill_compile(ctx):
    """A = hyper-connection の読み側のうち、GEMM 2 本 (`input_mix_weight_down`/
    `up`、+ inject の `block_inject_weight`) を除いた elementwise 部分だけを
    prefill 幅 (行数 >= `MLXTURBO_HC_COMPILE_MIN_ROWS`、既定 64) に限って
    `mx.compile(shapeless=True)` に畳む
    (`fused.enable_hyper_connection_prefill_compiled`、
    `MLXTURBO_HC_PREFILL_COMPILE`、既定 off) / B = 素の実装 (`hc-compiled` の
    B と同じ --- `disable_hyper_connection`/`disable_hyper_connection_kernel`
    を両方呼んで `GatedResidual.__call__` を vendor 実装に戻した状態)。

    `hc-compiled` (`enable_hyper_connection`、GEMM も含めて全体を 1 グラフに
    する版) との違いは、GEMM 2 本を `mx.compile` の外に出したまま
    `mx.quantized_matmul` を直接呼ぶこと (`_hc_prefill_compile_pre`/
    `_hc_prefill_compile_post` の 2 グラフ + down/up の間の `silu(.../hc)` 1
    op だけが素のまま挟まる)。decode/verify 幅 (行数 < しきい値) は常に
    vendor 実装にそのまま落ちるので、`hc-kernel`/`hc-compiled` と違って
    decode 幅では A/B とも同じコードパスを踏む。

    既存の融合 Metal カーネル (`hc-kernel`、`enable_hyper_connection_kernel`)
    とは同じ `GatedResidual.__call__` を取り合うため同時に使えない
    (`enable_hyper_connection_prefill_compiled` 自身が `_ORIG_HC_KERNEL is
    not None` で例外を出す)。A を貼る前に必ず
    `disable_hyper_connection_kernel()` を呼ぶこと。

    prefill にしか効かない (decode/verify 幅は行数ゲートで必ず素の実装へ
    落ちる) ので **`--prefill-once` は使えない** (`DECODE_ONLY_KNOBS` には
    入れない)。見るのは `prefill_s`。

    合格条件: prefill_s が短・長の両方で改善すること。**出力はビット同一
    ではない** (`control_identical=False`)。当初は `hc-compiled` と同じ理由
    (GEMM の呼び出し順・量子化経路を変えていない) でビット同一のはずと
    見込んでいたが、CPU 上の合成モデルで実測すると同一ではなかった
    (`tools/vendor_fingerprint.py` の md5 が on/off で変わる、logits の
    max|diff|=1.04e-7)。原因は `enable_hyper_connection_prefill_compiled`
    の reshape 戦略 -- `mx.compile(shapeless=True)` は先頭の可変長軸を
    Python の `.shape` 展開で reshape の target に焼き込むと壊れる (最初の
    trace の行数が別の行数の呼び出しに使い回されて reshape が失敗する、
    実測で確認)。回避のため先頭 (B, T) を 1 本の可変長行に潰してから
    `hc_norm`/`RMSNorm` のグループ計算をしており、この潰し方が vendor の
    `RMSNorm.__call__` (先頭を展開したまま `(B, T, hc, d)` で保持する) と
    reshape の経路が違う。1.04e-7 は float32 の丸み込みの水準
    (`tools/vendor_fingerprint.py` の `group_prefill_forward` が記録している
    8e-7 と同じ桁) で、正しさの問題ではなく mx.compile の演算順の違いと
    見て良い。短文脈で出力一致を要求せず、tok/round の悪化が無いことだけ
    確認すること。
    """
    import os

    from mlxturbo import fused

    def apply(variant):
        fused.disable_hyper_connection_prefill_compiled()
        fused.disable_hyper_connection_kernel()
        fused.disable_hyper_connection()
        os.environ["MLXTURBO_HC_PREFILL_COMPILE"] = "1" if variant == "A" else "0"
        if variant == "A":
            fused.enable_hyper_connection_prefill_compiled()
        # B は読み側の融合をすべて外した素の経路 (hc-compiled の B と同じ)

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


def _knob_indexer_lean(ctx):
    """A = block_starts/block_end + pooled fp32 キャストをキャッシュ (既定 off、
    2026-09-03) / B = 毎回作り直し (現行既定)。

    `mlxturbo.indexer_lean.enable_indexer_lean`/`disable_indexer_lean` を
    素通しするだけ (`pooled-cache` と同じ形)。値はビット不変 (`_IndexerCache
    .block_grid`/`.pooled_fp32` はどちらも「決定的な計算を、値が変わらない
    回にはやり直さない」だけ) なので対照 (出力一致) がそのまま効く。
    decode/verify 幅 (S<=8) だけが対象で prefill 幅では経路が変わらない
    (`QSAIndexer._pooled_and_top` の `lean` 判定)。
    """
    from mlxturbo.indexer_lean import disable_indexer_lean, enable_indexer_lean

    model = ctx["eng"].model

    def apply(variant):
        if variant == "A":
            enable_indexer_lean(model)
        else:
            disable_indexer_lean(model)

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

    `SL.SwitchGLU.__call__` の統合ディスパッチ (`fused.py` の `dispatched`)
    は verify -> glu -> gather_sort -> wide -> stock の優先順で、
    `gather_sort` は「素通し条件を持たない」(有効なら常にそこで確定する)。
    既定は `MLXTURBO_SORT_MIN=16` 経由で `_MOE_DISPATCH_SORT_MIN` が
    None でなくなっているので、`enable_wide_projections` で連結を仕込んでも
    `wide()` には一度も到達しない -- 上の「単独 A/B の記録は無い」はこれが
    理由 (measured のではなく測れていなかった)。A はこの gather_sort 分岐を
    一時的に退避して `wide()` (連結 + `wide()` 自身の内部ソート、閾値は
    `_MOE_DISPATCH_WIDE_SORT_MIN` 既定 64) を通す。B は連結を外し
    (`disable_wide_projections`、D-2 の修正込みで `_fused_w/_fused_s/_fused_b`
    も外れる)、退避した gather_sort を戻す -- つまり比較は「連結 + 内部
    ソート」対「素の 3 gather + gather_sort(16)」になる。`MLXTURBO_SORT_MIN=0`
    のような環境変数はプロセス全体の既定を変えてしまうので使わず、この
    knob の中だけでモジュール属性を退避・復元する。
    """
    from mlxturbo import fused

    eng = ctx["eng"]
    applied = {"on": False}
    stashed_sort_min = {"value": None}

    def apply(variant):
        if variant == "A" and not applied["on"]:
            stashed_sort_min["value"] = fused._MOE_DISPATCH_SORT_MIN
            fused._MOE_DISPATCH_SORT_MIN = None
            fused.enable_wide_projections(eng.model)
            applied["on"] = True
        elif variant == "B" and applied["on"]:
            fused.disable_wide_projections(eng.model)
            fused._MOE_DISPATCH_SORT_MIN = stashed_sort_min["value"]
            applied["on"] = False

    return apply


def _knob_wide_attn(ctx):
    """A = 連結射影を attention の qkv(+gate) だけに絞って on / B = off (既定)。

    `wide` knob (上) は 4 種類 (gdn/attn/shared/experts) まとめてで、experts の
    gather 連結を含んでいたため 17k prefill で +62% と大きく負けて棄却された
    (`docs/research/KERNEL-BRIEF-DECODE-BW.md`)。attention 単独の数字は無かった。

    `docs/research/SESSION-2026-09-02-CATCHUP.md` の「QSA attention 1 層の
    内訳」(2026-09-02 実測、実モデル) では、prefill 幅 S=2048 の qkv 射影が
    1 層 14ms・75 GFLOP で 5 TFLOPS 相当と低い。q_proj (2560->6144, gate 込み)
    / k_proj (2560->512) / v_proj (2560->512) を個別の 4-bit qmm 3 本で呼んで
    いるところを 1 本の 2560->7168 に連結すれば、M=2048 での効率が上がる
    見込み (-2% prefill)。

    `fused.enable_wide_projections(model, scope={"attn"})` で attn 以外
    (gdn/shared/experts) には一切触れない。連結した射影は
    `Attention._wide_qkv` に置かれ、`_vendor/qwen4_exp.py` の `_qkv` が
    行数 (B*S) >= `_wide_min_rows` (既定 64、`MLXTURBO_WIDE_MIN_ROWS`) の
    ときだけそれを使う -- decode 幅 (S=1..4 程度) は必ず個別 3 射影に落ちる
    ので、この knob は実質 prefill (チャンク幅 2048) だけを動かす。

    連結は「同じ入力に掛かる量子化行列を出力次元で連結するだけ」で、各出力行
    の量子化パラメータは行単位のため個々の出力は他の行が何本連結されても
    変わらないはず。CPU の合成量子化モデルで qkv 3 本 -> 1 本の呼び出し回数の
    減少 (S=128 で quantized_matmul 呼び出し数が層あたり -2) と、出力の
    ビット一致 (max|diff|=0.0) を確認した上での想定
    (`control_identical=True`)。実測で崩れたらここに記録して False にする。
    """
    from mlxturbo import fused

    eng = ctx["eng"]
    applied = {"on": False}

    def apply(variant):
        if variant == "A" and not applied["on"]:
            fused.enable_wide_projections(eng.model, scope={"attn"})
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


def _knob_sdpa_split(ctx):
    """A = sdpa 幅分割 (既定) / B = 単発呼び出し (旧経路)。

    Flash-Next は Hq=24/Hk=2 (gqa_factor=12)。MLX 0.32.2 の sdpa vector
    カーネルは `query_sequence_length * gqa_factor > 32` だと適格から
    外れ (`mlx/backend/metal/scaled_dot_product_attention.cpp:703`)、
    全 KV を読んで全スコアを実体化する経路に落ちる。verify 幅 (S>=3) は
    常にこれを越える。A は `Attention.__call__` / `_gather_tile_attn` の
    sdpa 呼び出しを、q と mask を幅 `max(1, 32 // gqa_factor)`
    (Flash-Next では 2) で S 軸に割って複数回呼び、concatenate で戻す
    (`mlxturbo/_vendor/qwen4_exp.py`、`_sdpa_split_width` がゲート)。
    合成マイクロでは 1 層あたり最大 9 倍 (4k/17k/50k で幅 2 の分割が
    3〜9 倍速い、`bench/test_sdpa_split.py`)。

    出力一致は要求しない -- 幅を割ったほうは vector カーネル、割らない
    ほうは materialize 経路に落ちる、選ばれるカーネル自体が違うので
    丸めが ulp オーダーでずれうる (`bench/test_sdpa_split.py` が bf16
    丸め (1e-2 以内) に収まることを別途、モデル無しで確認する)。

    発火の確認: `mlxturbo.kernels._fire.snapshot()` の `sdpa_split`。
    """
    import os

    from mlxturbo import fused

    def apply(variant):
        if variant == "A":
            os.environ.pop("MLXTURBO_SDPA_SPLIT", None)  # 既定 on のゲートを開けたまま
            fused.enable_sdpa_split()
        else:
            os.environ["MLXTURBO_SDPA_SPLIT"] = "0"  # off 側のゲートを閉じる
            fused.disable_sdpa_split()

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
    呼び出し側 (`mlxturbo/spec_flash.py` の `_prefetch_ngram_span`、
    `generate_stream` のチャンクループ内 2 箇所) は、いま組み立て終えた
    eval 境界 (グループ or 単独チャンク) の `mx.eval`/`mx.async_eval` を
    投入する**直前**に、**次の 1 境界ぶんだけ**先読みを呼ぶ。そうすると
    CPU 側の pread が直前境界の GPU 実行の壁時計 (2048 トークンのチャンクで
    3.4〜3.8s) に重なり、次の境界の `StreamNGram.__call__` がキャッシュ
    ヒットで待たずに返る。温キャッシュの pread 自体も 17k で約 2.5s、50k で
    約 7s が (重ねなければ) GPU が止まったまま CPU で消えている実測がある
    (行 1 つ 7.6us、prefill 1 チャンク 32768 行で約 250ms)。

    2026-09-02 の 17k A/B で取り分が 0% (-0.9%) だったのは、当時の実装
    (`_prefetch_ngram_rows`、削除済み) が `generate_stream` のループへ
    入る**前**に `ids` 全体を一度にまとめて先読みしていたため。まだどの
    GPU 実行も投入されていない時点でバックグラウンド pread が始まるので
    重ねる相手が無く、かつ最初の境界自身の `StreamNGram.__call__`
    (on-demand フォールバック、`self._pool` 使用) と背景スレッドの
    `_gather_pread` (同じ `self._pool`) が競合していた
    (`mlxturbo/ngram_stream.py` の `StreamNGram` docstring 参照)。
    2026-09-03 に「次の 1 境界だけ、直前境界の GPU 実行に重ねて」呼ぶ形へ
    直したので、この A/B は改めて取り直すこと。

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
    で明示的に有効化しない限り on にならない)。2026-09-02 時点の値 (0%) は
    上記のとおり旧実装のもので、直し後の値はまだ取っていない --
    `--knob ngram-prefetch --only long --ctx 8000/17000` で取り直すこと。
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


def _knob_prime_window(ctx):
    """prefill 末尾で MTP head を暖める窓幅 (`spec_flash.PRIME_WINDOW`)。既定 2048。

    `_prime_draft_cache` は末尾 min(PRIME_WINDOW, len) トークンぶんだけ MTP head を
    先回りして回し、decode 開始時点で draft キャッシュを空でなくしておく
    (docstring の言葉で言えば「窓で費用を文脈長に依らなくする」)。窓を狭めれば
    prefill 末尾の費用 (17k 冷 prefill で 0.25s、4k なら TTFT の 3.6%) は減るが、
    暖める文脈が短くなるぶん最初の数ラウンドの受理率が落ちうる。**prefill に
    効くので `DECODE_ONLY_KNOBS` には入れない** (`--prefill-once` は使えない)。

    出力は変わりうる (`control_identical=False`) -- draft の暖まり方が変わると
    受理・棄却の分岐点が動き、以降の生成トークン列が分岐しうるため。

    判定の物差し: **prefill_s (= TTFT) が減り、かつ tok/round の低下が 2%
    未満なら、小さい窓を既定にする。**どちらか片方しか満たさない場合は
    その場で決めない (`depth` knob と同じ扱い)。
    """
    import mlxturbo.spec_flash as SF

    def apply(variant):
        SF.PRIME_WINDOW = int(variant)

    return apply


def _knob_moe_combine(ctx):
    """A = ルータ重みを down_proj の入力 (SwiGLU 出力) に先掛けしてから
    down_proj を通し、top_k 軸の和を出力側で取る (MLXTURBO_MOE_COMBINE_FOLD、
    既定 on、`SparseMoeBlock._combine_fold_min_s`) / B = 素の経路
    (switch_mlp の出力 (rows, top_k, hidden_size) を実体化してから w を
    掛けて sum、MLXTURBO_MOE_COMBINE_FOLD=0)。

    動機: prefill 8k の内訳 (`tools/prefill_anatomy.py --ctx 8000`、
    `docs/research/SESSION-2026-09-02-CATCHUP.md` の「prefill 短文脈の内訳、
    8k」) で MoE 48 層中「ルータ重み + top-K 縮約」が 142ms/チャンク
    (効率 9.9%) と最大だった。down_proj は bias 無しの線形写像なので A/B は
    数式上は同じ値になるはずだが、w を掛ける位置が変わるぶん量子化 4bit +
    bf16 の積和順が動く (`control_identical=False`)。

    **行数ゲート**: 初回の in-model A/B (2026-09-03、行数ゲート無し) は
    prefill 8k -2.2%/17k -2.5% と勝った一方、decode は 8k +1.3%/17k +0.6%/
    短文脈 ms/round +1.4% (tok/round -4.0%) と負けた。行数が少ないと
    gate_proj/up_proj/down_proj を個別に呼ぶディスパッチ増分が乗算削減分を
    上回る。そこで行数 (B×S) が `MLXTURBO_MOE_COMBINE_FOLD_MIN_S`
    (既定 64) 未満のときは A 側でも必ず B と同じ素の経路に落ちる
    (decode/verify 幅 S<=8 は必ずここに入る -- `SparseMoeBlock.__call__` の
    行数ゲート、`mlxturbo/_vendor/qwen4_exp.py`)。
    `bench/test_moe_combine_fold.py` が合成モデルで、閾値以上は A/B の出力
    の RMS 相対誤差が 1e-2 以内、閾値未満はビット一致であることを確認済み。

    有効な間 (行数が閾値以上) は switch_mlp.__call__
    (mlx_lm.models.switch_layers.SwitchGLU) を経由しない (gate_proj/
    up_proj/down_proj を直接呼ぶ) ため、同じ SwitchGLU.__call__ に載って
    いる他の knob (`wide`/`moe-verify` など) の効果を素通りする --
    単独で測ること。

    **prefill に効くので `DECODE_ONLY_KNOBS` には入れない**
    (`--prefill-once` は使えない)。

    合格条件 (行数ゲート込みで再測定): **17k / 8k の prefill 壁時計
    (prefill_s) が縮み、かつ短文脈 decode の tok/round・ms/round が
    悪化しないこと。**KLD も併せて見る (積和順が変わるカーネルが受理率を
    落として差し引きで負けた前例が複数あるため)。
    `MLXTURBO_MOE_COMBINE_FOLD_MIN_S` を動かす実験をするときは、prefill
    側だけでなく短文脈 decode 側も必ず併せて見ること (閾値を下げすぎると
    decode/verify 幅まで fold が効いて短文脈がまた負ける)。
    """
    import os

    from mlxturbo import fused

    eng = ctx["eng"]

    def apply(variant):
        if variant == "A":
            os.environ.pop("MLXTURBO_MOE_COMBINE_FOLD", None)  # 既定 on のゲートを開けたまま
            fused.enable_moe_combine_fold(eng.model)
        else:
            os.environ["MLXTURBO_MOE_COMBINE_FOLD"] = "0"  # off 側のゲートを閉じる
            fused.disable_moe_combine_fold(eng.model)

    return apply


# ------------------------------------------------------------------------
# 天井スタブ (ceiling stub) knob 5 種。「その部品を 0 費用にしたら壁時計は
# どれだけ減るか」を in-model で測るための道具。出力の正しさは問わない --
# ms/round / prefill_s だけを見る。触るのは decode_ab.py の中だけ
# (`_vendor/qwen4_exp.py` と `spec_flash.py` は編集しない。差し替えは
# monkeypatch で knob の apply/呼び出し口から行う)。
# ------------------------------------------------------------------------


def _knob_stub_draft(ctx):
    """天井スタブ (D1)。A = MTP のドラフト生成を丸ごとスキップし、直前
    トークン (`cur`) の繰り返しを固定 draft として `depth` 個返す /
    B = 素 (既定)。

    `FlashSpecEngine._draft_chain` を丸ごと差し替え、`self.mtp.layers[0]`
    の forward を一切呼ばない (embed / combine / hyper_connection_mixer も
    呼ばない --- CPU 構築も GPU 実行も 0)。返す `drafts` は「長さ depth の
    mx.array (1,1) のリスト」という呼び手側の形だけ保つ (`generate_stream`
    の `mx.concatenate([cur] + drafts, axis=1)` / `mx.async_eval(drafts)`
    (2236〜2247 行、`next_drafts` 経由の 2349〜2358 行) はそのまま通る)。

    `_prime_accepted_gap` も no-op にする。素の `_draft_chain` はキャッシュを
    `keep=cache.size()+1` まで trim して cur 自身の 1 枠だけ残す規約になって
    いて、`_prime_accepted_gap` はその残り (`toks[0..len-2]`) を埋める役目
    --- この knob はその 1 枠すら書かない (MTP を一切呼ばないため) ので、
    `_prime_accepted_gap` を生かしたままだと毎ラウンド cur の分だけ書き損ね
    が積もり、MTP キャッシュの offset がラウンドを追うごとにずれ続ける
    (`oracle-draft` の docstring 参照 --- あちらは逆に「cur の 1 枠だけは
    本物の forward で書く」ことでこの積み残りを避けている)。ここでは MTP
    キャッシュの整合性そのものに用が無いので、両方まとめて無効化するのが
    安全。verify 幅 (depth) は呼び手からそのまま受け取って個数だけ揃える
    --- **変えない**。

    出力の正しさは問わない (draft はほぼ毎回外れる想定で、trunk の verify
    が「+1 ボーナストークン」だけを確定させる形に近づく)。目的は「draft の
    CPU 構築 + GPU 実行を 0 費用にしたら壁時計がどれだけ減るか」の天井
    (D1)。**判定は ms/round だけ**(tok_per_round/受理率は意味を持たないので
    無視すること)。
    """
    from mlxturbo.spec_flash import FlashSpecEngine

    orig_chain = FlashSpecEngine._draft_chain
    orig_prime = FlashSpecEngine._prime_accepted_gap

    def stub_chain(self, cur, hyper_prev, cache, depth, trace_top2=False):
        return [cur for _ in range(depth)]

    def stub_prime(self, toks, hypers, cache):
        return None

    def apply(variant):
        if variant == "A":
            FlashSpecEngine._draft_chain = stub_chain
            FlashSpecEngine._prime_accepted_gap = stub_prime
        else:
            FlashSpecEngine._draft_chain = orig_chain
            FlashSpecEngine._prime_accepted_gap = orig_prime

    return apply


def _knob_stub_indexer_topk(ctx):
    """天井スタブ (D6)。A = decode/verify 幅 (S<=4) の QSA top-k 選択
    (`QSAIndexer._pooled_and_top` の `mx.argpartition` 呼び出し 1 箇所、
    `_vendor/qwen4_exp.py:474`) を「先頭 k ブロックの固定 slice」に
    置き換える / B = 素 (既定)。**prefill 幅 (S>=64) は触らない**
    (`_pooled_and_top` に入った時点で S を見て素の経路へ丸ごと逃がす)。

    `_pooled_and_top` (370〜481 行) は pooled key の作成からブロック選択
    まで 1 関数にまとまっていて、argpartition の 1 行だけを外から差し替える
    フックが無い。関数を丸ごと写すと `_vendor` 側の変更に追従できなくなる
    ので、代わりに **呼び出しの瞬間だけ** `mx.argpartition` をこの関数専用の
    固定 slice にすり替え、`_pooled_and_top` 本体 (`orig`) を呼んだ直後に
    元へ戻す。MLX はグラフを Python 側で同期的に組むだけ (GPU 実行は非同期
    でも、この 1 メソッド呼び出しの間に他の Python コードが割り込むことは
    無い) なので、この間だけの差し替えは `_pooled_and_top` 以外の
    argpartition 呼び出し (MoE routing、draft-rerank の top-2 選定など) には
    一切触れない。

    固定 slice は `mx.arange(n_blocks, dtype=mx.uint32)` を
    `mx.argpartition` と同じ形・同じ dtype (uint32) にブロードキャストした
    もの --- スライス `[..., :k]` を取ると常に「ブロック 0..k-1」になる
    (`visible` によるマスク処理・softmax 相当の重み付けなど後続の演算は
    素のまま通す)。出力の正しさ (選ばれるブロックが実際に近いかどうか) は
    問わない。目的は「top-k 選択そのものの費用が 0 だったら壁時計がどれだけ
    減るか」の天井 (D6)。**判定は ms/round だけ**(短文脈 decode で見る --
    `--only short` で十分。長文脈でも発火はするが目的は decode 幅の費用な
    ので `--only short` を推奨)。
    """
    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q

    orig = Q.QSAIndexer._pooled_and_top

    def _fixed_argpartition(a, kth, axis=-1):
        n = a.shape[axis]
        idx = mx.arange(n, dtype=mx.uint32)
        shape = [1] * a.ndim
        shape[axis] = n
        return mx.broadcast_to(idx.reshape(shape), a.shape)

    def stub(self, x, rope, cache, offset, positions=None):
        B, S, _ = x.shape
        if S > 4:
            return orig(self, x, rope, cache, offset, positions)
        real_argpartition = mx.argpartition
        mx.argpartition = _fixed_argpartition
        try:
            return orig(self, x, rope, cache, offset, positions)
        finally:
            mx.argpartition = real_argpartition

    def apply(variant):
        Q.QSAIndexer._pooled_and_top = stub if variant == "A" else orig

    return apply


def _knob_stub_gdn_scan(ctx):
    """天井スタブ (P1)。A = prefill 幅 (T>=64) の GDN 再帰スキャンを、v を
    そのまま返す no-op (state も同じ形のまま素通し) に差し替える /
    B = 素 (既定)。

    prefill 幅で実際に発火する経路は、既定 (`MLXTURBO_GDN_METAL=1`) では
    `GatedDeltaNet.__call__` (`_vendor/qwen4_exp.py:1292〜1298`) の
    `_gdn_metal` 分岐 -- `gdn_blocked_metal.gated_delta_update_blocked_metal`
    が呼ぶ `gated_delta_blocked_seq` (同ファイル 271 行、実際の Metal
    カーネル dispatch)。`gbm.eligible()` (Dk==128 / Dv%32==0 / T>=64 など)
    で外れた場合の素の経路は `mlx_lm.models.gated_delta.gated_delta_update`
    (`_vendor/qwen4_exp.py:1311〜1323`、`from .gated_delta import
    gated_delta_update` でモジュール名前空間に束ねてある)。**この 2 つを
    両方 no-op にする**ことで、どちらの経路が発火していても scan の費用を
    消せる (`gdn-blocked` は既定 off で対象外のまま --- 発火していないことは
    `mlxturbo.kernels._fire.snapshot()` に `gdn_blocked` が出ないことで確認
    できる)。

    no-op は「g/beta の計算と `_fire.bump` は素のまま (費用として残る --
    再帰そのものではないので)、実際の再帰スキャンだけを飛ばして v を
    そのまま出力・state を素通し (無ければ fp32 ゼロ) で返す」。
    出力の正しさは問わない (v をそのまま出すだけで decode 後段は破綻した
    値のまま流れる)。目的は「scan を丸ごと 0 費用にしたら prefill_s が
    どれだけ縮むか」の天井 (P1)。**判定は prefill_s だけ**(生成トークン
    自体は捨てる -- `--only long --ctx 8000 --tokens 8` 程度で十分)。
    """
    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo.kernels import gdn_blocked_metal as gbm

    orig_seq = gbm.gated_delta_blocked_seq
    orig_ref = Q.gated_delta_update

    def stub_seq(q, k, v, g, beta, state=None, block_t=None):
        B, T, Hk, Dk = q.shape
        Hv, Dv = v.shape[2:]
        if state is None:
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        return v, state

    def stub_ref(q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
                 use_kernel=True):
        B, T, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        if state is None:
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        return v, state

    def apply(variant):
        if variant == "A":
            gbm.gated_delta_blocked_seq = stub_seq
            Q.gated_delta_update = stub_ref
        else:
            gbm.gated_delta_blocked_seq = orig_seq
            Q.gated_delta_update = orig_ref

    return apply


def _knob_stub_moe_single_expert(ctx):
    """天井スタブ (P3/P4)。A = `SparseMoeBlock.__call__`
    (`_vendor/qwen4_exp.py:1422` 付近) の routing 結果 `idx` を全行
    「専門家 0..top_k-1 の固定」に差し替える (top_k はそのまま、全行同じ
    top_k 個の専門家) / B = 素 (既定)。gather_qmm はソート済み経路のまま
    触らない。

    `stub-indexer-topk` と同じ「呼び出しの瞬間だけ `mx.argpartition` を
    固定 slice にすり替える」手口 (`idx = mx.argpartition(-lr_or_logits,
    top_k-1, axis=-1)[..., :top_k]` が r513/素の両分岐にあるが、どちらも
    `[..., :top_k]` を取る前の argpartition だけを差し替えれば済む)。
    全行が同じ `[0..top_k-1]` を選ぶので、後続の softmax 重み
    (`take_along_axis(logits, idx)`) は実在する値のまま計算され、
    `_moe_combine_fold`/`switch_mlp` のソート済み経路 (`mx.argsort(idx_flat)`)
    に渡ると **行またぎのセグメントが専門家境界をまたがない最良ケース**
    になる (全トークンが同じ専門家集合を引くため)。

    幅によるゲートは無い (decode/verify 幅の MoE 呼び出しにも同じ差し替えが
    掛かる) --- 測定対象が prefill_s だけなので実害は無い。目的は
    「gather_qmm がセグメント straddle 無しの最大効率で走ったときの
    prefill_s」の天井 (P3/P4)。**判定は prefill_s だけ**
    (`--only long --ctx 8000 --tokens 8` 程度で十分、出力の正しさは問わない
    --- 全トークンが専門家 0..top_k-1 だけで処理されるので中身は破綻する)。
    """
    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q

    orig_call = Q.SparseMoeBlock.__call__

    def _fixed_argpartition(a, kth, axis=-1):
        n = a.shape[axis]
        idx = mx.arange(n, dtype=mx.uint32)
        shape = [1] * a.ndim
        shape[axis] = n
        return mx.broadcast_to(idx.reshape(shape), a.shape)

    def stub_call(self, x):
        real_argpartition = mx.argpartition
        mx.argpartition = _fixed_argpartition
        try:
            return orig_call(self, x)
        finally:
            mx.argpartition = real_argpartition

    def apply(variant):
        Q.SparseMoeBlock.__call__ = stub_call if variant == "A" else orig_call

    return apply


def _knob_oracle_draft(ctx):
    """D3 の損益 (仮説裏取り、天井スタブではない)。`--oracle-out <json>`
    (decode_ab の `--save-out` 出力、`rows[].out`/`rows[].prompt_ids`) を
    読み、各ラウンドで「現在位置の真の次 N-1 トークン」を draft にして
    verify 幅 S=N で流す (受理はほぼ 100% になるはず)。variant がその S
    そのもの --- `"6"` は真の次 5 トークンを draft (depth=5)、`"2"` は
    真の次 1 トークンだけを draft (depth=1、既定の depth=2 相当の対照より
    さらに絞った下限)。目的は **S=6 の ms/round が S=2 の何倍か**
    (2.4 倍を超えたら畳む、というのが仮説の判定線)。

    `--oracle-out` の JSON は **`--only long` で作った (かつこの knob の
    run 自身も `--only long` でなければならない)** --- `rows` の中で
    `kind=="long"` の行を `prompt_ids` の初出順に 0,1,2,... と番号付けし、
    それが decode_ab 自身の `case_idx` (`cases` を `--only long` で作った
    ときの並び) と一致する前提で対応付けている。`--only both` で作った
    JSON でも `kind=="long"` の行だけを拾うので使えるが、**この knob の run
    自身は必ず `--only long`**(`_knob_oracle_draft` の setup で検査し、
    違えば例外)。

    `_draft_chain` を差し替える (呼び手は変えない)。現在の生成位置は
    エンジン側の状態を直接読むのではなく、この knob 自身が
    `eng.depth_trace_prompt_id` (`f"{kind}:{case_idx}"`、main() が
    ラウンドとは無関係に毎回立てている識別子) で「どの oracle 系列を
    追っているか」を特定し、**oracle の受理が 100% だった場合に engine が
    たどるはずの受理済みトークン数**を `apply()` 呼び出し (= 1 回の
    generate_stream 開始) のたびに 0 から数え直して求める --- 1 ラウンド
    ごとに `depth+1` (draft 全部命中 + trunk 自身のボーナス 1 トークン)
    ずつ進む想定。受理が 100% から外れた場合はこのカウンタが実際の位置から
    ずれていくので、その回以降は「真の続き」ではなくなる (出力の正しさは
    問わない --- ms/round の計測が目的で、tok/round は参考程度)。

    cur 自身の MTP キャッシュへの書き込みだけは本物の forward で行う
    (embed -> combine -> `mtp.layers[0]` を 1 段だけ、素の `_draft_chain`
    の step 0 と全く同じ形。予測結果は使わず捨てる)。**これは
    `stub-draft` と違って `_prime_accepted_gap` を無効化しないため**
    --- `_prime_accepted_gap` は検証で確定した中間トークンを毎ラウンド
    MTP キャッシュへ書き戻す本物の処理で、S が広いほど書き戻す本数が
    増える (これ自体が S=6 vs S=2 の実コスト差の一部なので、消してはいけ
    ない)。それが要求する「呼び出し前の cache は常にクリーン」という
    不変条件を満たすには、cur 自身の 1 枠だけは `_draft_chain` が書く
    という規約を保つ必要がある -- なので step 0 だけは本物を通す
    (2 段目以降 --- 素の経路なら投機的に組んでトリムで捨てられるだけの
    段 --- はスキップして oracle の値をそのまま使う。捨てられる計算を
    そもそもしないだけなので、cache の整合性に影響しない)。

    `eng.depth_ctx_limit` を `1<<30` に上げ、`eng._depth_adapt` を False に
    する (`_knob_depth`/`_knob_depth_adapt` と同じ理由 --- 17k は
    `MLXTURBO_DEPTH_ADAPT` の既定 on が効く範囲なので、それを無効化しないと
    指定した S が長文脈の途中で上書きされる)。

    `DECODE_ONLY_KNOBS` に入れてある (`_draft_chain` は decode ループの中
    でしか呼ばれない) ので **`--prefill-once` が使える** --- 17k の prefill
    を 4 回 (variants=["6","2"] の回文) 払わずに済む。
    """
    import json
    from pathlib import Path

    import mlx.core as mx
    from mlxturbo import spec_flash
    from mlxturbo.spec_flash import FlashSpecEngine, trim_attn_cache

    args = ctx["args"]
    eng = ctx["eng"]
    if args.only != "long":
        raise ValueError(
            "oracle-draft: --only long でだけ使うこと (case_idx の対応付けが "
            "--oracle-out の JSON と decode_ab 自身の long 系列内の出現順で "
            "決まるため -- --only both だと global な case_idx がずれる)"
        )
    if not args.oracle_out:
        raise ValueError("oracle-draft: --oracle-out <json> が必須")

    saved = json.loads(Path(args.oracle_out).expanduser().read_text())
    seen: dict[tuple, int] = {}
    by_idx: dict[int, list] = {}
    for row in saved:
        if row.get("kind") != "long":
            continue
        if "out" not in row or "prompt_ids" not in row:
            raise ValueError(
                f"oracle-draft: {args.oracle_out} に out/prompt_ids が無い"
                " (元の run で --save-out を付け忘れていないか確認すること)"
            )
        key = tuple(row["prompt_ids"])
        idx = seen.setdefault(key, len(seen))
        by_idx.setdefault(idx, row["out"])
    if not by_idx:
        raise ValueError(f"oracle-draft: {args.oracle_out} に kind=long の行が無い")

    state = {"fresh": True, "seq": None, "pos": 0}

    def _resolve():
        pid = getattr(eng, "depth_trace_prompt_id", "") or ""
        head, sep, tail = pid.rpartition(":")
        case_idx = int(tail) if sep and head == "long" and tail.isdigit() else None
        state["seq"] = by_idx.get(case_idx)
        state["pos"] = 0
        state["fresh"] = False

    def stub_chain(self, cur, hyper_prev, cache, depth, trace_top2=False):
        if state["fresh"]:
            _resolve()
        # cur 自身の MTP キャッシュ書き込みだけは本物 (素の _draft_chain の
        # step 0 と同じ形)。予測結果は使わず捨てる -- docstring 参照。
        Q = spec_flash._arch()
        keep = cache.size() + 1
        emb = self.model.model.embed_tokens(cur)
        mask = Q.create_attention_mask(emb, None)
        x = self.mtp.combine(emb, hyper_prev)
        self.mtp.layers[0](x, self.rope, mask, None, cache, cache.indexer, None, None)
        trim_attn_cache(cache, keep)

        seq = state["seq"]
        pos = state["pos"]
        if seq is None:
            # oracle 系列が引けない (warmup、または case 不一致)。安全な
            # フォールバックとして stub-draft と同じ「cur の繰り返し」
            drafts = [cur for _ in range(depth)]
        else:
            last = len(seq) - 1
            drafts = []
            for i in range(1, depth + 1):
                j = pos + i if pos + i <= last else last
                drafts.append(mx.array([[seq[j]]], dtype=cur.dtype))
        state["pos"] = pos + depth + 1
        return drafts

    FlashSpecEngine._draft_chain = stub_chain

    def apply(variant):
        depth = int(variant) - 1
        eng.depth = depth
        eng.depth_ctx_limit = 1 << 30
        eng._depth_adapt = False
        state["fresh"] = True

    return apply


KNOBS = {
    # name: (setup(ctx) -> apply(variant), variants, 出力一致を要求するか,
    #        まとめで基準にする variant)
    "qsa-tail": (_knob_qsa_tail, ["A", "B"], True, "B"),
    "moe-verify": (_knob_moe_verify, ["A", "B"], False, "B"),
    "fast-rope": (_knob_fast_rope, ["A", "B"], False, "B"),
    # A = PLE 埋め込みを層ループ前に一括計算 / B = 素 (既定)。判定は決めていない
    # (第 1 段は実測前)。出力一致は要求する (control_identical=True)。
    "ple-hoist": (_knob_ple_hoist, ["A", "B"], True, "B"),
    "gdn-prework": (_knob_gdn_prework, ["A", "B"], False, "B"),
    "gdn-blocked": (_knob_gdn_blocked, ["A", "B"], False, "B"),
    "gdn-metal": (_knob_gdn_metal, ["A", "B"], False, "B"),
    "hc-write": (_knob_hc_write, ["A", "C", "B"], True, "B"),
    "rms-norm-gated": (_knob_rms_norm_gated, ["A", "C", "B"], True, "B"),
    "moe-route": (_knob_moe_route, ["A", "C", "B"], False, "B"),
    "hc-prefill": (_knob_hc_prefill, ["A", "C", "B"], False, "C"),
    "hc-kernel": (_knob_hc_kernel, ["A", "B"], False, "B"),
    "hc-compiled": (_knob_hc_compiled, ["A", "B"], True, "B"),
    "hc-prefill-compile": (_knob_hc_prefill_compile, ["A", "B"], False, "B"),
    "pipeline": (_knob_pipeline, ["A", "B"], False, "B"),
    "fast-qmm": (_knob_fast_qmm, ["A", "B"], False, "B"),
    "null": (_knob_null, ["A", "B"], True, "B"),
    "temp": (_knob_temp, ["0.0", "0.7"], False, "0.0"),
    "indexer-cache": (_knob_indexer_cache, ["A", "B"], True, "B"),
    "pooled-cache": (_knob_pooled_cache, ["A", "B"], True, "B"),
    "indexer-lean": (_knob_indexer_lean, ["A", "B"], True, "B"),
    "stage-every": (_knob_stage_every, ["1", "2", "4"], True, "2"),
    "prefill-group": (_knob_prefill_group, ["2", "4", "8"], True, "4"),
    "prefill-pipeline": (_knob_prefill_pipeline, ["A", "B"], True, "B"),
    "fold-tail": (_knob_fold_tail, ["A", "B"], True, "A"),
    "qsa": (_knob_qsa, ["A", "B"], False, "A"),
    "bool-mask": (_knob_bool_mask, ["A", "B"], False, "B"),
    "sdpa-split": (_knob_sdpa_split, ["A", "B"], False, "B"),
    "gather-attn": (_knob_gather_attn, ["A", "B"], False, "B"),
    # -1 は gather 自体を切る (現行既定)。0 はタイルなしの gather
    "gather-tile": (_knob_gather_tile, ["-1", "0", "256"], False, "-1"),
    "prefill-attn": (_knob_prefill_attn, ["A", "B"], False, "B"),
    "wide": (_knob_wide, ["A", "B"], False, "B"),
    "wide-attn": (_knob_wide_attn, ["A", "B"], True, "B"),
    "depth": (_knob_depth, ["1", "2", "3"], False, "2"),
    "depth-adapt": (_knob_depth_adapt, ["A", "B"], False, "B"),
    "draft-rerank": (_knob_draft_rerank, ["A", "B"], False, "B"),
    "mtp-append": (_knob_mtp_append, ["A", "B"], False, "B"),
    # A = interleaved (本番既定) を基準に、B = separate (RAM 常駐) と比べる
    "ngram-layout": (_knob_ngram_layout, ["A", "B"], True, "A"),
    # A = 先読み有効 (既定) / B = 無効。判定は prefill_s
    "ngram-prefetch": (_knob_ngram_prefetch, ["A", "B"], True, "A"),
    # A = batch_min_rows=64 (既定) / B = 10**9 (常に行ごと旧経路)。判定は prefill_s
    "ngram-batch": (_knob_ngram_batch, ["A", "B"], True, "A"),
    "prime-window": (_knob_prime_window, ["2048", "512"], False, "2048"),
    "moe-combine": (_knob_moe_combine, ["A", "B"], False, "B"),
    # 天井スタブ 5 種 (docs 参照は各 knob 関数の docstring)。出力の正しさは
    # 問わないので control_identical は全部 False
    "stub-draft": (_knob_stub_draft, ["A", "B"], False, "B"),
    "stub-indexer-topk": (_knob_stub_indexer_topk, ["A", "B"], False, "B"),
    "stub-gdn-scan": (_knob_stub_gdn_scan, ["A", "B"], False, "B"),
    "stub-moe-single-expert": (
        _knob_stub_moe_single_expert, ["A", "B"], False, "B"),
    "oracle-draft": (_knob_oracle_draft, ["6", "2"], False, "2"),
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


def run_resumed(eng, caches, snap, resume, base_pos, n_tokens, eos_ids, temp=0.0):
    """控えた状態から decode だけを流す。返り値は run_once と同じ形。

    ``temp`` は `spec_flash.generate_stream` にそのまま渡す (既定 0.0 = 現行の
    greedy)。temp>0 だと draft は greedy のまま、verify の logits からだけ
    温度付きサンプリングする (`spec_flash._verify` の分岐)。"""
    import mlx.core as mx

    _restore(caches, snap)
    empty = mx.zeros((1, 0), dtype=mx.int32)
    t0 = time.perf_counter()
    gen = eng.generate_stream(empty, n_tokens, caches=caches, eos_ids=eos_ids,
                              resume=resume, base_pos=base_pos, temp=temp)
    out = []
    try:
        while True:
            out.extend(next(gen))
    except StopIteration as e:
        accepted, rounds = e.value[0], e.value[1]
    return out, 0.0, time.perf_counter() - t0, accepted, rounds


def run_once(eng, ids, n_tokens, eos_ids, temp=0.0):
    """1 本流して (トークン列, prefill 秒, decode 秒, accepted, rounds) を返す。

    ``temp`` は `run_resumed` と同じ意味 (既定 0.0 = greedy)。"""
    import mlx.core as mx

    caches = eng.model.make_cache()
    mx.clear_cache()
    t0 = time.perf_counter()
    gen = eng.generate_stream(ids, n_tokens, caches=caches, eos_ids=eos_ids, temp=temp)
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
    ap.add_argument("--temp", type=float, default=0.0,
                    help="サンプリング温度 (既定 0.0 = greedy)。"
                         "generate_stream(..., temp=...) にそのまま渡り、"
                         "draft は greedy のまま verify の logits だけ温度付きで"
                         "サンプルする (spec_flash._verify)。"
                         "temp>0 では毎ラウンドの採否が乱数で変わるので、"
                         "control_identical=True の knob でも短文脈の出力一致"
                         "対照は成立しない (対照 NG は測定破綻ではなく "
                         "temp>0 の性質)。temp 自体を A/B の軸にしたいときは "
                         "--knob temp --variants 0.0,0.7 を使う (このフラグは "
                         "そちらでは無視され、variant の値が温度になる)")
    ap.add_argument("--seed", type=int, default=0,
                    help="mx.random.seed に渡す。temp>0 の乱数列を固定して "
                         "run 間で比較・再現できるようにする (既定 0)")
    ap.add_argument("--out", default=None, help="結果 JSON の書き出し先")
    ap.add_argument("--only", choices=("both", "short", "long"), default="both",
                    help="長さの片方だけ回す (交差点探しで短文脈を省くため)")
    ap.add_argument("--variants", default=None,
                    help="knob の値をカンマ区切りで絞る (既定は KNOBS の全部)")
    ap.add_argument("--prefill-once", action="store_true",
                    help="長文脈で prefill を 1 回に畳む (prefill に効く knob "
                         "= prefill-group / stage-every には使わないこと)")
    ap.add_argument("--round-trace", action="store_true",
                    help="MLXTURBO_ROUND_TRACE=1 を立て、ラウンドごとの "
                         "peak_delta_mb (verify までのピークメモリ増分、KV "
                         "全長コピー調査用) を rows に記録する。"
                         "spec_flash.FlashSpecEngine.last_round_trace を "
                         "run_once/run_resumed のたびに読み出す。"
                         "--prefill-once と組み合わせるときは注意: "
                         "prefill_once/run_resumed の _snapshot/_restore が "
                         "caches の keys/values を直接掴んだまま次の decode を "
                         "始めるので、各 variant の最初のラウンドだけ "
                         "update_and_fetch が donation できずコピーになる "
                         "(本番の継続 decode には無い、このハーネス自身の "
                         "アーティファクト)。1 ラウンド目を除いて見ること。")
    ap.add_argument("--draft-trace", action="store_true",
                    help="MLXTURBO_DRAFT_TRACE=1 を立てる。_draft_chain の "
                         "1 段目の draft top-1/top-2 と _verify が確定させた "
                         "真の次トークンを突き合わせ、hit@1/hit@2 を "
                         "mlxturbo.kernels._fire (draft_trace_rounds/"
                         "draft_trace_hit1/draft_trace_hit2) に積む。既存の "
                         "fired 収集 (_fire.snapshot()) にそのまま乗るので "
                         "結果 JSON の fired に出る。木化ドラフト (レーン11 "
                         "仮説7) の上限 = hit@2 - hit@1 を見るための道具。")
    ap.add_argument("--depth", type=int, default=None,
                    help="`--knob depth` を経由せず、engine の投機深さを "
                         "指定した値に固定する (どの knob と組み合わせても "
                         "効く)。`eng.depth` を書き換え、`_knob_depth` と "
                         "同じ理由で `eng.depth_ctx_limit` も 1<<30 に "
                         "上げるので、文脈長によらず指定した深さのまま "
                         "(MLXTURBO_DEPTH_ADAPT=1 のときは、それでも "
                         "depth_adapt_min_pos を超えた位置では controller が "
                         "上書きする -- 完全に固定したいなら "
                         "MLXTURBO_DEPTH_ADAPT=0 も一緒に渡すこと)。"
                         "MLXTURBO_DEPTH_TRACE の trace 採取を `--knob null` "
                         "と組み合わせて 1 本だけ流すために足した")
    ap.add_argument("--save-out", action="store_true",
                    help="既定では rows に生成トークンの先頭 24 個 "
                         "(head=out[:24]) しか残さない。このフラグを立てると "
                         "各 row に生成トークン id の全列 (`out`、list[int]) "
                         "と、プロンプトのトークン id 列 (`prompt_ids`、"
                         "長文脈だと 17k 個規模になるのでこのフラグの "
                         "ときだけ) を追加で持たせる。既定 off のときの "
                         "結果 JSON は今までと完全に同一 (行を追加するだけで "
                         "既存キーは変えない)。")
    ap.add_argument("--oracle-out", default=None,
                    help="`--knob oracle-draft` 専用。decode_ab の "
                         "`--save-out` 出力 (JSON、`rows[].out`/"
                         "`rows[].prompt_ids` を使う) へのパス。真の続き "
                         "トークン列を draft として流す (`_knob_oracle_draft` "
                         "の docstring 参照)。他の knob では無視される。")
    args = ap.parse_args()

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")
    if args.round_trace:
        os.environ["MLXTURBO_ROUND_TRACE"] = "1"
    if args.draft_trace:
        os.environ["MLXTURBO_DRAFT_TRACE"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    mx.random.seed(args.seed)

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
    from mlxturbo.runner import enable_default_fusions, set_wired_limit_default

    enable_default_fusions(model, log_prefix="[decode_ab]")
    # engine を直叩きなので server.py の _load() を経由しない --
    # 常駐条件を本番と揃えるため、ここで自前で wire する
    # (mlxturbo/runner.py の set_wired_limit_default 参照)。
    set_wired_limit_default(log_prefix="[decode_ab]")
    mtp_path = args.mtp or os.path.join(model_path, "mtp.safetensors")
    q = {"group_size": 64, "bits": args.mtp_bits} if args.mtp_bits else None
    mtp = mtp_flash.load_flash_mtp(os.path.expanduser(mtp_path),
                                   model.args.text, quantize=q)
    mx.eval(mtp.parameters())
    eng = spec_flash.FlashSpecEngine(model, mtp)
    if args.depth is not None:
        # `_knob_depth` と同じ理由 (このファイル内のコメント参照): ここで
        # ctx_limit も上げておかないと、文脈長が indexer_budget を越えた
        # 位置で engine 自身が depth を 1 に落としてしまい、指定した深さの
        # まま全位置を観測できない。
        eng.depth = args.depth
        eng.depth_ctx_limit = 1 << 30

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
                eng.depth_trace_prompt_id = f"warmup:{kind}"
                run_once(eng, ids, 32, eos_ids, temp=args.temp)
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
        # depth-adapt は _effective_depth 経由でしか呼ばれず、それ自体
        # decode ループの中 (generate/generate_stream の while) でしか
        # 呼ばれない (prefill の forward は素の model(...) 呼び出しで、
        # draft/verify を挟まない)。"depth" knob と同じ理由。
        "depth-adapt",
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
        # sdpa-split は `1 < S <= 8` のときしか分岐に入らない。prefill の
        # チャンク幅 (既定 2048、`mlxturbo/spec.py` の PREFILL_STEP_SIZE)
        # は常にこれを大きく越えるので、prefill 幅では発火しない。
        "sdpa-split",
        # indexer-lean (`QSAIndexer._pooled_and_top` の `lean` 判定) は
        # `S <= 8` をコードで明示的に要求する。prefill のチャンク幅は常に
        # それを越えるので、on にしても prefill 幅の経路は 1 op も変わらない
        # (`pooled-cache`/`indexer-cache` と違い、こちらは decode/verify 幅
        # 限定の枝を新設した knob なので prefill には触れようがない)。
        "indexer-lean",
        # draft-rerank (`eng._rerank`) は `FlashSpecEngine._draft_argmax`
        # からしか読まれず、それを呼ぶのは `_draft_chain` (decode ループの
        # 中だけ) だけ。prefill の priming (`_prime_draft_cache`) は
        # `_draft_argmax` を経由しない素の `self.mtp(...)` forward。
        "draft-rerank",
        # temp は generate_stream の _sample (verify 後の 1/2 トークン目) に
        # しか渡らない。prefill のチャンクループはトークンをサンプルせず、
        # 与えられた ids をそのまま処理するだけ (_sample を通らない)。
        "temp",
        # stub-draft (`_draft_chain`/`_prime_accepted_gap`) は decode ループ
        # の中でしか呼ばれない。prefill のチャンクループは MTP に一切触れ
        # ない (素の model(...) 呼び出しのみ)。
        "stub-draft",
        # stub-indexer-topk は `_pooled_and_top` に入った時点で S<=4 を見て
        # 分岐する (S>=64 の prefill 幅は orig にそのまま逃がす、knob 自身の
        # docstring 参照)。prefill 幅は 1 op も変わらない。
        "stub-indexer-topk",
        # oracle-draft (`_draft_chain` の差し替え) も decode ループの中でしか
        # 呼ばれない。17k の prefill を variants=["6","2"] の回文 4 回払わず
        # 済むよう、ここに入れておく (knob 自身の docstring 参照)。
        "oracle-draft",
    }
    if args.prefill_once and args.knob not in DECODE_ONLY_KNOBS:
        print(f"knob={args.knob} は prefill に影響しうるので --prefill-once は"
              f"使えない (decode 専用と確認済みなのは"
              f" {sorted(DECODE_ONLY_KNOBS)})")
        return 1

    rows = []
    for case_idx, (kind, ids) in enumerate(cases):
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
        eng.depth_trace_prompt_id = f"warmup:{kind}:{case_idx}"
        if shared is None:
            run_once(eng, ids, 32, eos_ids, temp=args.temp)
        else:
            run_resumed(eng, *shared, base_pos=n, n_tokens=32, eos_ids=eos_ids,
                        temp=args.temp)
        for v in order:
            set_variant(v)
            # カーネルの発火回数を条件ごとに数え直す。適格判定は条件を外すと
            # 黙って False を返すので、「効果ゼロ」が遅いのか届いていないのかを
            # 区別する手が要る (2026-09-01 に GDN 前処理で実際に空振りした)。
            _fire.reset()
            if ngram_stream is not None:
                ngram_stream.reset_stats()
            # MLXTURBO_DEPTH_TRACE の prompt_id フィールド用 (未設定 = トレース
            # 無効時は engine 側で読まれないだけなので、常に立てて害はない)。
            eng.depth_trace_prompt_id = f"{kind}:{case_idx}"
            if shared is None:
                out, tp, td, acc, rounds = run_once(
                    eng, ids, args.tokens, eos_ids, temp=args.temp)
            else:
                out, tp, td, acc, rounds = run_resumed(
                    eng, *shared, base_pos=n, n_tokens=args.tokens,
                    eos_ids=eos_ids, temp=args.temp)
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
            if args.save_out:
                # 既定 (off) の JSON は上の rows.append(...) までで変わらない。
                # --save-out のときだけ、生成トークン id の全列と (長文脈だと
                # 17k 個規模になる) プロンプトのトークン id 列を追加で持たせる。
                rows[-1]["out"] = out
                rows[-1]["prompt_ids"] = ids[0].tolist()
            fired = _fire.snapshot()
            rows[-1]["fired"] = fired
            fired_s = ("  発火 " + " ".join(f"{k}={n}" for k, n in
                                            sorted(fired.items()))) if fired else ""
            ngram_s = ""
            if ngram_stream is not None:
                rows[-1]["ngram"] = dict(ngram_stream.stats)
                ngram_s = "  " + ngram_stream.stats_line()
            peak_s = ""
            if args.round_trace:
                trace = list(getattr(eng, "last_round_trace", None) or [])
                rows[-1]["peak_delta_mb"] = trace
                if trace:
                    rows[-1]["peak_delta_mb_median"] = statistics.median(trace)
                    rows[-1]["peak_delta_mb_max"] = max(trace)
                    # --prefill-once のとき、run_resumed の _restore が
                    # caches の keys/values を直接掴んだ snap を毎 variant
                    # の頭で挿し戻す (このファイルの _snapshot/_restore の
                    # docstring 参照)。そのため 1 ラウンド目だけ
                    # update_and_fetch が donation できずコピーになるのは
                    # ハーネス自身のアーティファクトで、本番の継続 decode
                    # には無い。2 ラウンド目以降だけの値も別に残す。
                    rest = trace[1:]
                    if rest:
                        rows[-1]["peak_delta_mb_median_wo_r1"] = statistics.median(rest)
                        rows[-1]["peak_delta_mb_max_wo_r1"] = max(rest)
                    peak_s = (f"  peak_delta_mb median={rows[-1]['peak_delta_mb_median']:.1f}"
                              f" max={rows[-1]['peak_delta_mb_max']:.1f}"
                              f" (r1={trace[0]:.1f})")
            print(f"  {v}: prefill {tp:6.2f}s  decode {td:6.2f}s  "
                  f"{ms:6.2f} ms/tok  tok/round {tpr:.3f}  "
                  f"({acc}/{rounds}){fired_s}{ngram_s}{peak_s}", flush=True)
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
