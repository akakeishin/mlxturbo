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

途中 (2 ラウンド消化後) で 1 回 ``ledger.compact()`` を呼び、その前後を
通して一致することも確認する (「compaction 前後の一致」のケース)。

## 実行結果 (記録、2026-09-01、CPU)

    .venv/bin/python tools/verify_batch_spec.py

B=3 行 x (受理数 1/2/3 を満遍なく踏むスクリプト、途中で compaction を 1 回
挟む) 全ケース一致。最大誤差は attention の KV で 2.4e-6、GDN 状態で
1.8e-8、n-gram 文脈は完全一致 (0.0、整数の切り出しなので丸めが乗らない)。
tools/verify_batch_cache.py が単体で確認済みの float32 丸め幅 (1.5e-7) と
桁で見て同オーダー -- 破損 (5 桁以上の差) は出ていない。

## 既知の未対応

- indexer / QSA はバッチ対象外 (mlxturbo/batch_spec.py の docstring 参照)。
  この検証では indexer_budget を kv 長より十分大きく取って、そもそも
  QSA が発火しない構成だけを見ている。
- サーバー配線・admission・スケジューラはここには無い (依頼どおり)。
- 温度 > 0 のサンプリングは検証していない (貪欲のみ)。
"""

from __future__ import annotations

import argparse
import sys
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
    # QSA を発火させない (このモジュールの対象外なので構成側で避ける)
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
EXTRA_ROUNDS = 2  # rollback 後も生成が続くことの確認 (T=1、常に満額受理)


def build():
    from mlx.utils import tree_map
    from mlx_lm.models import qwen4_exp as Q

    mx.random.seed(0)
    model = Q.Model(Q.ModelArgs(model_type="qwen4_exp", text_config=dict(TINY)))
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
        assert keep == keep_target, (keep, keep_target)
        rollback(model, cache, cap, pre, keep=keep, total=total,
                 ids_kept=pair[:, :keep])
        cur = nxt[:, keep - 1 : keep]
        ptr += keep
        keeps_used.append(keep)
        return lg

    lg = None
    for r in range(N_ROUNDS):
        lg = do_round(DEPTH, keep_script[r])
    for _ in range(EXTRA_ROUNDS):
        lg = do_round(1, 2)  # T=1, 常に満額受理

    return cache, cur, lg, ptr


# ------------------------------------------------------------- batched 経路


def run_batched(model, prompt_batch, oracle_seqs):
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
    for b in range(B):
        assert int(cur[b, 0].item()) == oracle_seqs[b][0]
    ptrs = [0] * B

    def do_round(depth, keep_targets, compact_before=False):
        nonlocal cur
        if compact_before:
            ledger.compact(model, caches)
        drafts_lists = [
            build_drafts(oracle_seqs[b], ptrs[b], depth, keep_targets[b])
            for b in range(B)
        ]
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
            assert keeps[b] == keep_targets[b], (b, keeps[b], keep_targets[b])
        batched_rollback(model, caches, cap, keeps, pre_ctx=pre_ctx, pair=pair)
        ledger.commit_round(keeps, total)
        idx = (mx.array(keeps) - 1)[:, None]
        cur = mx.take_along_axis(nxt, idx, axis=1)
        for b in range(B):
            ptrs[b] += keeps[b]
        return lg

    lg = None
    for r in range(N_ROUNDS):
        keep_targets = [KEEP_SCRIPT[b][r] for b in range(B)]
        lg = do_round(DEPTH, keep_targets, compact_before=(r == COMPACT_AFTER_ROUND))
    for _ in range(EXTRA_ROUNDS):
        lg = do_round(1, [2] * B)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)

    import mlxturbo  # noqa: F401  (arch registry の副作用のためだけに import)

    model = build()

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
        model, prompt_batch, oracle_seqs
    )

    ok = True
    for b in range(B):
        cache_b, cur_b, lg_b, ptr_b = solo_results[b]
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

    print("\n=== 全ケース一致 ===" if ok else "\n=== 不一致あり ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
