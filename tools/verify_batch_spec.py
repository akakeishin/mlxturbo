"""`mlxturbo.batch_spec` (バッチ x 投機の中核部品) の正しさハーネス。

91GB の実モデルは使わない。合成した小さい Flash-Next を乱数初期化し、
CPU だけで数値を比較する (GPU の時間計測はしない -- 正しさのみ)。

## 判定基準

B=3 行それぞれについて、

    (a) 単独:  mlxturbo.spec_flash の capture()/rollback()/snapshot_pre()
               (無変更) で 1 行ずつ検証ラウンドを回す
    (b) バッチ: mlxturbo.batch_spec の RaggedLedger/RaggedAttnCache/
               batched_rollback で B 行同時に同じラウンドを回す

に、**全く同じトークン列** (プロンプト + 各ラウンドの draft) を与え、
最終 logits とキャッシュ状態 (GDN 状態・GDN conv window・PLE conv window・
n-gram 文脈・full attention の生きている KV) が一致することを確認する。

dead slot の mask 除外は数学的には単独実行と同じ集合に対する softmax なので、
丸め (float32) 以外は一致するはず。判定は許容誤差つきの allclose。

各ラウンドの draft は「その行の本当の greedy 継続 (oracle)」から作る:
最初に oracle (drafting なしの素の 1 トークンずつの greedy 継続) を計算して
おき、ラウンドごとに oracle の次 T トークンを draft として使う。全部正しい
draft を使えば受理数 keep=T+1 (満額受理) になるが、意図的に途中から
「間違った」トークンに差し替えることで keep を 1..T+1 の好きな値に制御できる
(位置 j より前の draft が正しい限り、その回で検証フォワードに実際に流れる
トークン列は oracle の履歴と完全に一致するので、その回の argmax は
oracle の「本当の次のトークン」と数学的に一致するはず -- 途中に何を
drafted しても、受理された分の出力トークンは oracle と一致する)。
台本どおりの受理数になるのは QSA が不活性なときだけなので (下の「QSA が
活性な構成」参照)、実際の受理数は solo で出たものを記録して batch に
渡し、**両者が一致すること自体を判定に含める**。

途中 (2 ラウンド消化後) で 1 回 ``ledger.compact()`` を呼び、その前後を
通して一致することも確認する (「compaction 前後の一致」のケース)。

## 実行結果 (記録、2026-09-01、CPU)

    .venv/bin/python tools/verify_batch_spec.py

B=3 行 x (受理数 1/2/3 を満遍なく踏むスクリプト、途中で compaction を 1 回
挟む) 全ケース一致。最大誤差は attention の KV で 2.4e-6、GDN 状態で
1.8e-8、n-gram 文脈は完全一致 (0.0、整数の切り出しなので丸めが乗らない)。
tools/verify_batch_cache.py が単体で確認済みの float32 丸め幅 (1.5e-7) と
桁で見て同オーダー -- 破損 (5 桁以上の差) は出ていない。

## QSA が活性な構成 (追記 2026-09-02)

同じ検査を ``indexer_budget=8`` に落としてもう一度回す。そこでは prefill の
直後からブロック選択が働き、行ごとに dead slot の入り方が違う状態で
``_ragged_indexer_call`` (行ごとのブロック境界) が回る。

**ただし QSA が活性だと「素の貪欲デコード (oracle)」は参照に使えない。**
完成したブロックに入った列はブロック選択の支配下に入るので、1 回の
フォワードで何列進めるかで見える集合が変わる。単独の投機経路
(``FlashSpecRunner.generate``) ですら token-by-token と食い違う (この合成
モデルで実測)。本家 QSA の性質で、``tools/vendor_fingerprint.py`` の
「chunk の割り方が変われば選ばれるブロックも変わる」と同じ現象。

そこで QSA 活性側の参照は次の 2 つにする:

- ``check_rounds``: 同じ draft 列・同じラウンド幅で 1 行ずつ回した**単独の
  検証フォワード** (本家 QSA) と、最終 logits・全キャッシュ・受理数まで
  一致すること。判定基準の (a)/(b) そのもの。
- ``check_qsa_row_invariance``: 同じ機構を B=1 で回したものと、B=3 で
  同時に回したものが一致すること (行数だけが違う比較)。走行中の join と
  compaction を挟んだ場合も見る。
- ``check_qsa_mixed_budget`` (B-1、追記 2026-09-02): budget を跨ぐ長い行と
  跨がない短い行を同居させ、短い行の出力が単独実行と一致すること。プロンプト
  長を固定して最初のラウンドから確実に踏ませる (受理/棄却の巡り合わせに
  頼る ``check_qsa_row_invariance`` では、短い行が長い行に巻き込まれる瞬間を
  安定して踏めない)。

## 既知の未対応

- 段 3(b) の gather 経路 (``MLXTURBO_GATHER_ATTN``) は行別境界に未対応で、
  バッチ経路の間は通常経路 (加算マスク) に落ちる。
- 温度 > 0 のサンプリングは検証していない (貪欲のみ)。

## スケジューラ (追記 2026-09-02、chunked prefill に組み替え)

末尾の 4 つがスケジューラ側を見る。どれも「貪欲の投機は出力を変えない」ことを
使って、素の貪欲継続 (oracle) と突き合わせる。

- ``check_chunked_prefill``: プロンプトを chunk 幅 2/3/一括で流して、どれでも
  同じ列が出ること。2 回目以降のチャンクは「生きている過去 + 新規列の因果」の
  マスクで走るので、ここが合えば刻んだ prefill の帳簿・再帰状態の持ち越し・
  priming 窓の尻尾がそろっている。
- ``check_join_midflight``: 走行中のバッチに後から行を足しても両方が正しい列を
  出すこと (0 ラウンド後と 3 ラウンド後 = dead slot が溜まった状態の両方)。
  **前回「実機でしか確かめられない」として見送った箇所がここ** -- 新入りの KV を
  走行中の物理列数へ左詰めで揃えたときの RoPE の角度とマスクの整合、再帰系の
  バッチ軸連結、priming 窓の幅が行ごとに違う場合。
- ``check_coordinator``: 同時 3 本 (chunk 512 と 2 の両方) と、走行中に 3 本目を
  投げる場合。行ごとの eos / max_tokens で順に抜けること、走行中の join が
  実際に起きていること (``coord.joins``)、1 本しか無いときは単独経路
  (``FlashSpecRunner.generate``) へ落ちること (``coord.solo_runs``)。
- ``check_preemption``: 空きの見積もりを差し替えて退避を起こし、生成済みを
  保持したまま復帰した列が oracle と一致すること。

### 実行結果 (記録、2026-09-02、CPU)

全ケース一致。退避は 2 回発生し、退避後の列も oracle と一致した。

### 実行結果 (記録、2026-09-02、QSA 活性を足したあと、CPU)

全ケース一致。QSA 活性 (budget=8) の ``check_rounds`` の最大誤差は attention の
KV で 2.3e-6、logits で 7.7e-7 で、QSA 不活性のときと同オーダー。
``check_qsa_row_invariance`` は同時 3 本・走行中 join とも 12 トークン完全一致。
行別経路が実際に走っていることは分岐カウンタで確認済み
(RaggedAttnCache 116 回 / RaggedDraftCache 44 回 / 本家へ素通し 143 回)。

``--gpu`` (Metal、tools/biglock.sh 経由) でも全ケース一致。QSA 活性の
``check_rounds`` は KV が 6.3e-7、logits が 4.2e-7 と CPU より小さい
(cumsum / put_along_axis / argpartition / take_along_axis の Metal 実装でも
同じ列が選ばれている、ということ)。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# 実モデルの比率をなるべく保った縮小版 (tools/verify_batch_cache.py の TINY を
# 踏襲しつつ、依頼どおり層数・hidden をもう少し大きく)。linear_*_head_dim は
# gated_delta の Metal カーネルが Dk % 32 == 0 を要求するので 32 で止める
# (CPU では素の mx 経路にフォールバックするので実害は無いが、実装を合わせる)
TINY = dict(
    model_type="qwen4_exp_text",
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=32,
    vocab_size=256,
    rms_norm_eps=1e-6,
    full_attention_interval=2,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    hc_count=2,
    hc_lowrank=32,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=32,
    # 既定では QSA を発火させない。活性側は build(budget=8) で別に見る
    indexer_budget=4096,
    indexer_compress_ratio=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=1024,
    make_ngram_vocab_size_divisible_by=8,
    split_ngram_parts=4,
    ple_embed_dim=32,
    # layer_types (interval=2, layers=4) = [linear, full, linear, full].
    # PLE は linear_attention 層にしか付けられない (PLELayer._short_conv が
    # ArraysCache 前提の cache[2]/cache[3] を触るため) ので、layer_idx=2
    # (linear) に対応する 3 を選ぶ
    ple_layer_ids=[3],
    ple_conv_kernel_size=4,
    seed=0,
    eos_token_id=255,
    partial_rotary_factor=0.25,
    rope_theta=10000.0,
    tie_word_embeddings=False,
)

VOCAB = TINY["vocab_size"]
PROMPT_LEN = 6
B = 3

# 各行・各ラウンド (T=2) の受理数スクリプト。1..T+1=3 を満遍なく踏む
KEEP_SCRIPT = {
    0: [3, 3, 1, 3],
    1: [2, 3, 2, 1],
    2: [1, 2, 3, 3],
}
DEPTH = 2  # T
N_ROUNDS = 4
COMPACT_AFTER_ROUND = 2  # 2 ラウンド消化後に compaction をはさむ
EXTRA_ROUNDS = 2  # rollback 後も生成が続くことの確認 (T=1)


def build(budget: int | None = None):
    """``budget`` を渡すと ``indexer_budget`` だけ差し替える (QSA を早く
    発火させるための小さい値を入れる用)。重みは seed 固定なので、budget を
    変えても同じ重みの同じモデルになる。"""
    from mlx.utils import tree_map
    from mlx_lm.models import qwen4_exp as Q

    cfg = dict(TINY)
    if budget is not None:
        cfg["indexer_budget"] = budget
    mx.random.seed(0)
    model = Q.Model(Q.ModelArgs(model_type="qwen4_exp", text_config=cfg))
    model.update(
        tree_map(
            lambda a: mx.random.normal(a.shape) * 0.05 if a.dtype == mx.float32 else a,
            model.parameters(),
        )
    )
    mx.eval(model.parameters())
    model.eval()
    return model


def oracle_continue(model, prompt, n):
    """drafting なしの素の greedy 継続。行ごとの「本当の答え」を作るためだけ
    に使う使い捨てのキャッシュ (以後は破棄する)。"""
    cache = model.make_cache()
    logits = model(prompt, cache=cache)
    mx.eval(logits)
    cur = int(mx.argmax(logits[0, -1]))
    seq = [cur]
    cur_t = mx.array([[cur]])
    for _ in range(n - 1):
        logits = model(cur_t, cache=cache)
        mx.eval(logits)
        cur = int(mx.argmax(logits[0, -1]))
        seq.append(cur)
        cur_t = mx.array([[cur]])
    return seq


def build_drafts(oracle_seq, ptr, depth, keep):
    """oracle の次 depth トークンから draft を作る。keep より後ろの位置を
    わざと間違ったトークンに差し替えて、受理数を keep に制御する。"""
    true_next = oracle_seq[ptr + 1 : ptr + 1 + depth]
    drafts = list(true_next)
    for j in range(keep - 1, depth):
        drafts[j] = (drafts[j] + 1) % VOCAB
    return drafts


def compute_keep(nxt_row, draft_row):
    """argmax 列 (depth+1 個) と draft 列 (depth 個) を比べて受理数を返す。"""
    hit = 0
    while hit < len(draft_row) and nxt_row[hit] == draft_row[hit]:
        hit += 1
    return hit + 1


# --------------------------------------------------------------- solo 経路


def run_solo(model, prompt, oracle_seq, keep_script):
    from mlxturbo.spec_flash import capture, rollback, snapshot_pre

    cache = model.make_cache()
    with capture(model) as cap:
        logits = model(prompt, cache=cache)
        mx.eval(logits)
    cur = mx.argmax(logits[:, -1], axis=-1).reshape(1, 1)
    assert int(cur.item()) == oracle_seq[0], "oracle と prefill の初手が食い違う"
    ptr = 0
    keeps_used = []
    drafts_used = []

    def do_round(depth, keep_target):
        nonlocal cur, ptr
        drafts_list = build_drafts(oracle_seq, ptr, depth, keep_target)
        drafts = [mx.array([[d]]) for d in drafts_list]
        pair = mx.concatenate([cur] + drafts, axis=1)
        total = pair.shape[1]
        pre = snapshot_pre(model, cache)
        with capture(model) as cap:
            lg = model(pair, cache=cache)
            mx.eval(lg)
        nxt = mx.argmax(lg, axis=-1)
        keep = compute_keep(nxt[0].tolist(), drafts_list)
        rollback(model, cache, cap, pre, keep=keep, total=total,
                 ids_kept=pair[:, :keep])
        cur = nxt[:, keep - 1 : keep]
        ptr += keep
        keeps_used.append(keep)
        drafts_used.append(drafts_list)
        return lg

    lg = None
    for r in range(N_ROUNDS):
        lg = do_round(DEPTH, keep_script[r])
    for _ in range(EXTRA_ROUNDS):
        lg = do_round(1, 2)  # T=1

    # 台本 (`KEEP_SCRIPT`) は draft の作り方を決めるだけで、受理数そのものは
    # ここで実際に出たものを使う。QSA が活性な構成では oracle (1 トークンずつ)
    # と検証フォワード (T+1 列を同時に) でブロック格子が違うため、台本どおりの
    # 受理数にならないことがある (`tools/vendor_fingerprint.py` の「chunk の
    # 割り方が変われば選ばれるブロックも変わる」と同じ現象)。バッチ側には
    # ここで出た draft 列をそのまま流し、受理数が一致することも判定に含める。
    return cache, cur, lg, ptr, keeps_used, drafts_used


# ------------------------------------------------------------- batched 経路


def run_batched(model, prompt_batch, solo_keeps, solo_drafts):
    """``run_solo`` が実際に流した draft 列を、そのまま B 行同時に流す。
    受理数が行ごとに solo と一致することも判定に含める。"""
    from mlxturbo.batch_spec import (
        batched_capture,
        batched_rollback,
        make_ragged_cache,
        snapshot_pre_ctx,
    )

    caches, ledger = make_ragged_cache(model, B)
    # prefill 幅のフォワードなので light=True (F1、Opus 正しさレビュー指摘)。
    # この cap は下で rollback に使わない (commit_round だけで確定するので
    # states_all は要らない) -- batched_capture の docstring 参照。
    with batched_capture(model, light=True) as cap:
        logits = model(prompt_batch, cache=caches)
        mx.eval(logits)
    ledger.commit_round([PROMPT_LEN] * B, PROMPT_LEN)
    cur = mx.argmax(logits[:, -1], axis=-1).reshape(B, 1)
    ptrs = [0] * B

    def do_round(r, depth, compact_before=False):
        nonlocal cur
        if compact_before:
            ledger.compact(model, caches)
        drafts_lists = [solo_drafts[b][r] for b in range(B)]
        assert all(len(d) == depth for d in drafts_lists)
        drafts = [
            mx.array([[drafts_lists[b][j]] for b in range(B)]) for j in range(depth)
        ]
        pair = mx.concatenate([cur] + drafts, axis=1)
        total = pair.shape[1]
        pre_ctx = snapshot_pre_ctx(model, caches)
        with batched_capture(model) as cap:
            lg = model(pair, cache=caches)
            mx.eval(lg)
        nxt = mx.argmax(lg, axis=-1)
        nxt_l = nxt.tolist()
        keeps = [
            compute_keep(nxt_l[b], drafts_lists[b]) for b in range(B)
        ]
        for b in range(B):
            assert keeps[b] == solo_keeps[b][r], (
                f"row{b} round{r}: 受理数が solo と食い違う "
                f"(batch={keeps[b]} solo={solo_keeps[b][r]})"
            )
        batched_rollback(model, caches, cap, keeps, pre_ctx=pre_ctx, pair=pair)
        ledger.commit_round(keeps, total)
        idx = (mx.array(keeps) - 1)[:, None]
        cur = mx.take_along_axis(nxt, idx, axis=1)
        for b in range(B):
            ptrs[b] += keeps[b]
        return lg

    lg = None
    for r in range(N_ROUNDS):
        lg = do_round(r, DEPTH, compact_before=(r == COMPACT_AFTER_ROUND))
    for r in range(N_ROUNDS, N_ROUNDS + EXTRA_ROUNDS):
        lg = do_round(r, 1)

    return caches, ledger, cur, lg, ptrs


# ----------------------------------------------------------------- 比較


def _row_live_kv(cache, ledger, b):
    idx = [i for i, a in enumerate(ledger._alive[b]) if a]
    idx_arr = mx.array(idx)
    keys = mx.take(cache.keys[b], idx_arr, axis=1)[None]
    values = mx.take(cache.values[b], idx_arr, axis=1)[None]
    return keys, values


def _report(label, a, b, atol=1e-4):
    diff = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
    ok = diff <= atol or bool(mx.allclose(a, b, atol=atol, rtol=1e-3))
    print(f"  {'OK' if ok else 'NG'} {label}: max|diff|={diff:.3e}")
    return ok


def check_generator(model) -> bool:
    """`BatchSpecGenerator` の出力が、1 本ずつの貪欲デコードと一致するか。

    **貪欲の投機は出力を変えない**のが最も強い検査になる -- verify は本体の
    argmax と一致したときだけ受理するので、ドラフトの当たり外れに関係なく
    出力は素の貪欲デコードと同じ列になる。ここが割れたら、dead slot の
    マスク・行別 rollback・右パディングのどれかが壊れている。

    プロンプト長は「そろっている」と「不揃い」の両方を見る。不揃いは
    右パディングを踏み、再帰系の窓 (`_tail_window`) と入力マスク
    (`conv_mask`) の両方が効いていないと合わない。
    """
    from mlxturbo.batch_spec import BatchSpecGenerator
    from mlxturbo.mtp_flash import FlashMTPModule
    from mlxturbo.spec_flash import FlashSpecEngine

    mx.random.seed(0)
    mtp = FlashMTPModule(model.args.text, variant="lane")
    mx.eval(mtp.parameters())
    eng = FlashSpecEngine(model, mtp)
    n = 12
    cases = {
        "そろい": [[3, 11, 27, 5, 9, 41, 8], [7, 2, 19, 33, 4, 15, 22]],
        "不揃い": [[3, 11, 27, 5, 9, 41, 8], [7, 2, 19, 33, 4], [12, 45, 6]],
    }
    ok = True
    print("\n--- BatchSpecGenerator == 1 本ずつの貪欲 ---")
    for name, prompts in cases.items():
        ref = [
            oracle_continue(model, mx.array(p)[None], n)[:n] for p in prompts
        ]
        gen = BatchSpecGenerator(eng, prompts)
        got = gen.generate(n)
        for b, (r, g) in enumerate(zip(ref, got)):
            same = r == g
            ok &= same
            head = next((i for i, (x, y) in enumerate(zip(r, g)) if x != y), n)
            print(f"  {'OK' if same else 'NG'} {name} row{b}: 先頭一致 {head}/{n}")
            if not same:
                print(f"     solo ={r}\n     batch={g}")
    return ok


def _engine(model):
    from mlxturbo.mtp_flash import FlashMTPModule
    from mlxturbo.spec_flash import FlashSpecEngine

    mx.random.seed(0)
    mtp = FlashMTPModule(model.args.text, variant="lane")
    mx.eval(mtp.parameters())
    return FlashSpecEngine(model, mtp)


def check_chunked_prefill(model) -> bool:
    """prefill を刻んでも、一括で流したのと同じ列が出ること。

    `SpecPrefillLane` を chunk 幅 2 で回し、1 個目のトークンと、そこから
    `BatchSpecGenerator` で続けた列を素の貪欲継続 (oracle) と突き合わせる。
    2 回目以降のチャンクは「生きている過去 + 新規列の因果」のマスクで走る
    (`RaggedLedger.next_round_mask`) ので、ここが合っていれば刻んだ prefill の
    帳簿・再帰状態の持ち越し・priming 窓の尻尾が全部そろっていることになる。
    """
    from mlxturbo.batch_spec import BatchSpecGenerator, SpecPrefillLane

    eng = _engine(model)
    prompt = [3, 11, 27, 5, 9, 41, 8, 17, 2]
    n = 10
    ref = oracle_continue(model, mx.array(prompt)[None], n)

    print("\n--- chunked prefill ---")
    ok = True
    for chunk in (2, 3, len(prompt)):
        lane = SpecPrefillLane(eng, prompt)
        steps = 0
        while not lane.finished:
            lane.advance(min(chunk, lane.remaining))
            steps += 1
        gen = BatchSpecGenerator.from_prefilled(eng, [lane.result()])
        while len(gen.out[0]) < n:
            gen.step()
        got = gen.out[0][:n]
        same = got == ref
        ok &= same
        print(f"  {'OK' if same else 'NG'} chunk={chunk} ({steps} 回に分割): {n} トークン")
        if not same:
            print(f"     want={ref}\n     got ={got}")
    return ok


def check_join_midflight(model) -> bool:
    """走行中のバッチに後から行を足しても、両方の行が正しい列を出すこと。

    行 0 を数ラウンド走らせてから (物理列が dead slot を含む状態にしてから)
    行 1 を join する。ここが前回「実機でしか確かめられない」として見送った
    箇所で、確かめるのは 3 点:

    - 新入りの KV を走行中の物理列数へ左詰めで揃えたときに、RoPE の角度と
      マスクが食い違わないこと (食い違えば出力がずれる)。
    - 再帰系 (GDN/PLE/n-gram) をバッチ軸に連結しただけで正しいこと。
    - priming 窓の幅が行ごとに違っても、MTP のドラフトが壊れないこと
      (壊れても出力は変わらないが、`_pad` の帳簿が壊れていれば `trim` の
      基準がずれて落ちる)。
    """
    from mlxturbo.batch_spec import BatchSpecGenerator, SpecPrefillLane

    eng = _engine(model)
    # 長さをわざと変える (priming 窓の幅と物理列数の両方がずれる)
    prompts = [[3, 11, 27, 5, 9, 41, 8], [7, 2, 19, 33, 4, 15, 22, 6, 31, 12, 45]]
    n = 12
    ref = [oracle_continue(model, mx.array(p)[None], n) for p in prompts]

    print("\n--- 走行中の join ---")
    ok = True
    for after in (0, 3):
        lane0 = SpecPrefillLane(eng, prompts[0])
        while not lane0.finished:
            lane0.advance(4)
        gen = BatchSpecGenerator.from_prefilled(eng, [lane0.result()])
        for _ in range(after):
            gen.step()
        lane1 = SpecPrefillLane(eng, prompts[1])
        while not lane1.finished:
            lane1.advance(4)
        gen.join([lane1.result()])
        while min(len(o) for o in gen.out) < n:
            gen.step()
        for b in range(2):
            got = gen.out[b][:n]
            same = got == ref[b][:n]
            ok &= same
            print(
                f"  {'OK' if same else 'NG'} {after} ラウンド後に join row{b}:"
                f" {n} トークン"
            )
            if not same:
                print(f"     want={ref[b][:n]}\n     got ={got}")
    return ok


def check_preemption(model) -> bool:
    """メモリが足りなくなったら退避し、生成済みを保持したまま復帰すること。

    空きの見積もり (`_free_bytes`) を差し替えて、走り出してから足りなくなる
    状況を作る。退避された行は「プロンプト + 生成済み」で prefill をやり直す
    ので、貪欲なら列は変わらないはず -- これが崩れると、退避が「静かに別の
    文章になる」形で出る。
    """
    import concurrent.futures

    from mlxturbo.batch_spec import BatchSpecCoordinator
    from mlxturbo.runner import FlashSpecRunner, start_batched_spec_generation

    eng = _engine(model)
    runner = FlashSpecRunner(eng)
    prompts = [[3, 11, 27, 5, 9, 41, 8], [7, 2, 19, 33, 4, 15, 22]]
    n = 14
    ref = [oracle_continue(model, mx.array(p)[None], n) for p in prompts]

    print("\n--- preemption ---")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        coord = BatchSpecCoordinator(runner, executor, max_batch=4, eos_ids=set(),
                                     wait_ms=200)
        calls = [0]

        def room():
            # 最初は 2 行とも通し、走り出してから 1 行ぶんしか無いことにする
            calls[0] += 1
            return 10 ** 12 if calls[0] <= 4 else 1
        coord._free_bytes = room
        futures = [
            start_batched_spec_generation(coord, p, n, 0.0, set(), None, None, None)
            for p in prompts
        ]
        got = [f.result(timeout=300)["tokens"] for f in futures]
    finally:
        executor.shutdown(wait=True)

    ok = coord.preemptions > 0
    print(f"  {'OK' if ok else 'NG'} 退避が起きた: {coord.preemptions} 回")
    for b in range(2):
        same = got[b] == ref[b]
        ok &= same
        print(f"  {'OK' if same else 'NG'} row{b}: {len(got[b])} トークン")
        if not same:
            print(f"     want={ref[b]}\n     got ={got[b]}")
    return ok


def check_coordinator(model) -> bool:
    """`BatchSpecCoordinator` (サーバー配線側のスケジューラ) の検査。

    見るのは 3 つ。どれも「貪欲の投機は出力を変えない」ことを使って、
    素の貪欲継続 (oracle) と突き合わせる。

    1. 同時に来た複数の要求が同じバッチで回り、それぞれ正しい列を返す。
    2. max_tokens と eos が行ごとに効き、先に終わった行が落ちても
       (`BatchSpecGenerator.retire`) 残りの行の出力が変わらない。
    3. 1 本しか無いときは単独経路 (`FlashSpecRunner.generate`) に落ちる。
    """
    import concurrent.futures

    from mlxturbo.batch_spec import BatchSpecCoordinator
    from mlxturbo.mtp_flash import FlashMTPModule
    from mlxturbo.runner import FlashSpecRunner, start_batched_spec_generation
    from mlxturbo.spec_flash import FlashSpecEngine

    mx.random.seed(0)
    mtp = FlashMTPModule(model.args.text, variant="lane")
    mx.eval(mtp.parameters())
    runner = FlashSpecRunner(FlashSpecEngine(model, mtp))

    prompts = [[3, 11, 27, 5, 9, 41, 8], [7, 2, 19, 33, 4], [12, 45, 6]]
    n = 16
    oracle = [oracle_continue(model, mx.array(p)[None], n) for p in prompts]
    # 行 0 が 6 個目で止まるように eos を選ぶ (行ごとに止まる位置が違う状態を
    # 作って、先に終わった行の retire を踏ませる)
    eos = {oracle[0][5]}

    def expected(seq, max_tokens):
        out = []
        for t in seq[:max_tokens]:
            out.append(t)
            if t in eos:
                break
        return out

    print("\n--- BatchSpecCoordinator ---")
    ok = True
    cases = (
        # (ラベル, max_tokens, 3 本目を後から投げるか, chunk 幅)
        ("同時 3 本", [12, 7, 12], False, 512),
        ("同時 3 本 (chunk 2)", [12, 7, 12], False, 2),
        ("走行中に 3 本目", [12, 7, 12], True, 2),
        ("単独 1 本", [9], False, 512),
    )
    for label, max_tokens_list, staggered, chunk in cases:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            coord = BatchSpecCoordinator(
                runner, executor, max_batch=4, eos_ids=eos, wait_ms=200,
                prefill_chunk=chunk,
            )

            def submit(b):
                return start_batched_spec_generation(
                    coord, prompts[b], max_tokens_list[b], 0.0, eos, None, None, None
                )

            n_first = len(max_tokens_list) - 1 if staggered else len(max_tokens_list)
            futures = [submit(b) for b in range(n_first)]
            if staggered:
                # 先の 2 本が走り出したところに 3 本目を入れる。join に間に
                # 合わなければ次のバッチで走るだけで、出す列は変わらない
                time.sleep(0.05)
                futures.append(submit(len(max_tokens_list) - 1))
            got = [f.result(timeout=300)["tokens"] for f in futures]
            joins, solos = coord.joins, coord.solo_runs
        finally:
            executor.shutdown(wait=True)
        for b, mt in enumerate(max_tokens_list):
            want = expected(oracle[b], mt)
            same = got[b] == want
            ok &= same
            print(f"  {'OK' if same else 'NG'} {label} row{b}: {len(got[b])} トークン")
            if not same:
                print(f"     want={want}\n     got ={got[b]}")
        print(f"     走行中の join {joins} 回 / 単独経路 {solos} 回")
        if len(max_tokens_list) == 1:
            # 1 本しか無いときはバッチ機構に触らせない (B=1 無劣化の線)
            ok &= joins == 0 and solos == 1
        else:
            ok &= joins > 0
    return ok


def check_qsa_row_invariance(model) -> bool:
    """QSA が活性なとき、行を**同時に**回した結果が、同じ行を 1 行だけで
    回した結果と一致すること。

    ここだけ参照が「素の貪欲デコード (oracle)」ではない。**QSA が活性だと
    oracle は参照に使えない** -- 完成したブロックに入った列はブロック選択の
    支配下に入るので、1 回のフォワードで何列進めるかで見える集合が変わる。
    単独の投機経路 (`FlashSpecRunner.generate`) ですら token-by-token と
    食い違う (budget=8 の合成モデルで実測)。これは本家 QSA の性質で、
    `tools/vendor_fingerprint.py` が「chunk の割り方が変われば選ばれる
    ブロックも変わる」と書いているのと同じ現象。

    そこで参照を「同じ機構を B=1 で回したもの」にする。車線 (`SpecPrefillLane`)
    で prefill すると priming 窓も draft も受理数もラウンド幅も行ごとに閉じる
    ので、**行数だけが違う**比較になる。ここが割れたら、他の行の dead slot が
    自分の行に見えているか、行別のブロック境界がずれている。
    """
    from mlxturbo.batch_spec import BatchSpecGenerator, SpecPrefillLane

    eng = _engine(model)
    # 長さをわざと不揃いにする (行ごとに実在ブロック数と端数の幅が変わる)
    prompts = [
        [3, 11, 27, 5, 9, 41, 8, 17, 2],
        [7, 2, 19, 33, 4, 15, 22],
        [12, 45, 6, 31, 88, 4, 7, 19, 55, 2, 61, 30],
    ]
    n = 12

    def prefilled(p):
        lane = SpecPrefillLane(eng, p)
        while not lane.finished:
            lane.advance(min(4, lane.remaining))
        return lane.result()

    def drive(gen, rows, compact=False):
        while min(len(gen.out[b]) for b in range(len(rows))) < n:
            gen.step()
            if compact:
                # dead slot を物理的に詰めても出力は変わらないこと (論理列の
                # 並びは compaction で動かない、が行別ブロック境界の前提)
                gen.maybe_compact(waste_ratio=1.0)
        return [gen.out[b][:n] for b in range(len(rows))]

    print("\n--- QSA 活性: 同時に回しても 1 行ずつと同じ ---")
    ref = []
    for p in prompts:
        g = BatchSpecGenerator.from_prefilled(eng, [prefilled(p)])
        ref.append(drive(g, [0])[0])

    ok = True
    g = BatchSpecGenerator.from_prefilled(eng, [prefilled(p) for p in prompts])
    got = drive(g, prompts, compact=True)
    for b in range(len(prompts)):
        same = got[b] == ref[b]
        ok &= same
        print(f"  {'OK' if same else 'NG'} 同時 3 本 row{b}: {n} トークン")
        if not same:
            print(f"     alone={ref[b]}\n     batch={got[b]}")

    # 走行中の join (dead slot が溜まった物理列へ左詰めで入る)
    g = BatchSpecGenerator.from_prefilled(eng, [prefilled(prompts[0])])
    for _ in range(3):
        g.step()
    g.join([prefilled(prompts[1]), prefilled(prompts[2])])
    got = drive(g, prompts, compact=True)
    for b in range(len(prompts)):
        same = got[b] == ref[b]
        ok &= same
        print(f"  {'OK' if same else 'NG'} 3 ラウンド後に join row{b}: {n} トークン")
        if not same:
            print(f"     alone={ref[b]}\n     batch={got[b]}")
    return ok


def check_qsa_mixed_budget(model) -> bool:
    """B-1: budget を跨ぐ長い行と跨がない短い行が同居するとき、短い行の出力が
    単独実行と一致すること (17k 級の長い行 + 短いリクエストの同居を、合成
    モデルの縮尺で再現したケース)。

    ``_ragged_indexer_call`` はどれか 1 行でも論理 kv 長が ``token_budget`` を
    超えたら**全行**をブロック選択経路に落とす (帳簿の ``qsa_max_len`` は
    行ごとの最大)。修正前は budget 以下の短い行もそこに巻き込まれ、クエリ
    自身が属するブロックが ``block_end <= q_col`` を満たせず候補にすら
    入らないので、直近の列が不可視になっていた (B-1)。

    長い行のプロンプト長を budget 超えに、短い行を budget 以下に固定して
    join するので、**最初の検証ラウンドから確実にこの経路を踏む**
    (`check_qsa_row_invariance` のような自然な受理/棄却の巡り合わせに頼らない)。
    """
    from mlxturbo.batch_spec import BatchSpecGenerator, SpecPrefillLane

    eng = _engine(model)
    budget = model.args.text.indexer_budget
    long_prompt = [3 + i % 200 for i in range(budget + 8)]  # kv 長 > budget
    short_prompt = [3, 11, 27, 5, 9, 41, 8, 17]  # kv 長 8 <= budget
    n = 6

    def prefilled(p):
        lane = SpecPrefillLane(eng, p)
        while not lane.finished:
            lane.advance(min(8, lane.remaining))
        return lane.result()

    print(f"\n--- QSA 活性 (budget={budget}): budget を跨ぐ行との同居 (B-1) ---")
    ref_gen = BatchSpecGenerator.from_prefilled(eng, [prefilled(short_prompt)])
    while len(ref_gen.out[0]) < n:
        ref_gen.step()
    ref = ref_gen.out[0][:n]

    mixed = BatchSpecGenerator.from_prefilled(
        eng, [prefilled(long_prompt), prefilled(short_prompt)]
    )
    while min(len(o) for o in mixed.out) < n:
        mixed.step()
    got = mixed.out[1][:n]

    ok = got == ref
    print(f"  {'OK' if ok else 'NG'} 短い行 (単独 vs 長い行との同居): {n} トークン")
    if not ok:
        print(f"     alone={ref}\n     mixed={got}")
    return ok


def check_rounds(model, label: str) -> bool:
    """検証ラウンドを B 行同時に回した結果が、1 行ずつ回した結果と一致するか
    (最終 logits と全キャッシュ配列)。判定基準の (a)/(b) そのもの。"""

    print(f"\n########## 検証ラウンド ({label}) ##########")
    mx.random.seed(1)
    prompt_batch = mx.random.randint(0, VOCAB, (B, PROMPT_LEN))
    mx.eval(prompt_batch)

    oracle_seqs = [
        oracle_continue(model, prompt_batch[b : b + 1], 20) for b in range(B)
    ]

    solo_results = []
    for b in range(B):
        solo_results.append(
            run_solo(model, prompt_batch[b : b + 1], oracle_seqs[b], KEEP_SCRIPT[b])
        )

    caches, ledger, cur_batched, lg_batched, ptrs_batched = run_batched(
        model,
        prompt_batch,
        [r[4] for r in solo_results],
        [r[5] for r in solo_results],
    )

    ok = True
    for b in range(B):
        cache_b, cur_b, lg_b, ptr_b = solo_results[b][:4]
        print(f"\n### row {b}")
        ok &= ptr_b == ptrs_batched[b]
        print(f"  {'OK' if ptr_b == ptrs_batched[b] else 'NG'} ptr: solo={ptr_b} batch={ptrs_batched[b]}")
        ok &= int(cur_b.item()) == int(cur_batched[b, 0].item())
        print(
            f"  {'OK' if int(cur_b.item()) == int(cur_batched[b, 0].item()) else 'NG'} "
            f"cur: solo={int(cur_b.item())} batch={int(cur_batched[b, 0].item())}"
        )
        ok &= _report(f"logits[row{b}]", lg_b[0, -1], lg_batched[b, -1])

        for i, (layer, c_solo) in enumerate(zip(model.model.layers, cache_b)):
            c_batch = caches[i]
            if layer.layer_type == "full_attention":
                bk, bv = _row_live_kv(c_batch, ledger, b)
                # KVCache は self.step=256 刻みで確保するので、生の
                # .keys/.values は末尾に未使用領域を持つ -- .state で
                # offset まで切ったものを比べる
                sk, sv = c_solo.state
                ok &= _report(f"layer{i} attn.keys", sk, bk)
                ok &= _report(f"layer{i} attn.values", sv, bv)
                continue
            if c_solo[0] is not None:
                ok &= _report(f"layer{i} gdn.conv", c_solo[0], c_batch[0][b : b + 1])
            if c_solo[1] is not None:
                ok &= _report(f"layer{i} gdn.state", c_solo[1], c_batch[1][b : b + 1])
            if layer.ple is not None:
                if c_solo[2] is not None:
                    ok &= _report(f"layer{i} ple.conv", c_solo[2], c_batch[2][b : b + 1])
                if c_solo[3] is not None:
                    ok &= _report(f"layer{i} ngram.ctx", c_solo[3], c_batch[3][b : b + 1])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)

    import mlxturbo  # noqa: F401  (arch registry の副作用のためだけに import)

    model = build()
    ok = check_rounds(model, "QSA 不活性")
    ok &= check_generator(model)
    ok &= check_chunked_prefill(model)
    ok &= check_join_midflight(model)
    ok &= check_coordinator(model)
    ok &= check_preemption(model)

    # QSA が活性な構成でもう一度。budget を 8 に落とすと prefill の直後から
    # ブロック選択が働き、行ごとに dead slot の入り方が違う状態で
    # `_ragged_indexer_call` が回る。ここが割れたら、論理列の組み直し
    # (ブロックの中身・rope の角度・top-k の候補) のどれかが壊れている。
    qsa = build(budget=8)
    print("\n\n########## QSA 活性 (indexer_budget=8) ##########")
    ok &= check_rounds(qsa, "QSA 活性")
    ok &= check_qsa_row_invariance(qsa)

    # B-1: 短い行が budget を跨ぐ長い行と同居するケース。budget=8 だと
    # cr=4 の短い行を「跨がない」まま数ラウンド維持する余地が狭いので、
    # 別途 budget=32 で確保する (17k 級の長い行 + 短いリクエストの縮尺)。
    qsa32 = build(budget=32)
    print("\n\n########## QSA 活性 (indexer_budget=32) ##########")
    ok &= check_qsa_mixed_budget(qsa32)

    print("\n=== 全ケース一致 ===" if ok else "\n=== 不一致あり ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
