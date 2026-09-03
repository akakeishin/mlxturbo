"""自前の Metal カーネル (`mx.fast.metal_kernel`) 1 回あたりの実行費用を、

直列依存の連鎖で N 回呼んでから 1 回だけ `mx.eval` し、eval 時間を N で割って
求める。`tools/micro_kernel_latency.py` は 1 回呼んで毎回 eval する形
(`_bench_fixed` / `_bench_fixed_ab` / `_bench_chained_hc_ab`) なので、
eval のたびに同期 (~200us) が「1 回の費用」に混ざって読めない。ここでは
N=200 個を「前ステップの出力を次ステップの入力にする」依存連鎖でまず
Python 側だけで組み (どの eval も呼ばない)、最後に 1 回だけ `mx.eval` して、
(eval 時間) / N を 1 回あたりの費用とする。連鎖にする理由は decode の実態
(層 N の起動は層 N-1 の出力に本当に依存していて、GPU のコマンドキューが
勝手に並べ替えたり重ねたりできない) を壊さないため。

チェーンに乗らない出力 (例: GDN 前処理の g・beta) を放置すると、MLX の
遅延評価がそれを計算しない枝として刈ってしまい、素の実装側だけ実計算を
サボって見える (詳細は各 build_*_items 関数の docstring)。そのため各ステップ
関数は `(next_state, extra_leaves)` を返し、`extra_leaves` も毎回集めて
最後の 1 回の eval に含める。

構築 (Python でグラフを組む時間) と eval (GPU + 同期) を分けて両方出す。
fused/plain は CLAUDE.md の作法通り 1 プロセス内で ABBA 交互に測り、中央値を
取る (既定 n_pairs=3 で ABBA x 3 = 各側 6 試行)。

対象 (等価な plain 実装は多くを tools/micro_kernel_latency.py から流用):
  1. HC (hyper_connection.fused_gated_residual) vs plain_gated_residual
  2. GDN recurrent step: gated_delta_update_with_states vs
     mlx_lm.models.gated_delta.gated_delta_update (use_kernel=True)
     -- どちらも Metal カーネル。「plain」は素の op 列ではなく mlx_lm 自身の
     カーネル実装 (micro_kernel_latency.run_gdn_recurrent と同じ組み合わせ)。
  3. GDN prework: fused_gdn_prework vs plain_gdn_prework
  4. RMSNormGated: rms_norm_gated vs 素の (rms_norm して sigmoid(gate) を掛ける) 実装
  5. 参考の床 (カーネル起動そのものの固定費の目安。fused/plain の対がないので
     単独の us/call として出す): mx.fast.rms_norm 単体、`y + 1.0` の
     elementwise 単体、`mx.fast.metal_kernel` で書いた 2560 要素の加算。

**重みは既定で `--weight-sets` 組 (項目ごとの層数、hc=97/gdn=36) を巡回する
冷キャッシュ**で読む (各ステップ i が `weights[i % N]` を使う、各組別々の乱数)。
もともとは重み 1 組 (HC は down/up/inject 約 7 MB) を N=200 回読み回す温キャッシュ
だったため、並列度の低い自前カーネルが冷の DRAM レイテンシを隠せない負けが
見えていなかった (HC 融合: 温 +13 us -> 冷 +78 us/回 x 97 層 = +7.6 ms/forward、
docs/research/SESSION-2026-09-02-CATCHUP.md の「2026-09-03 12:00 custom kernel が
decode の in-model で負ける理由」節)。**判定ゲートは「重みを 100 MB 超巡回させた
冷の連鎖で素の op 列より速いこと」**に移した。`--weight-sets 1` で旧来の温キャッシュ
に戻せる。GDN 系の重み (A_log/dt_bias/conv1d.weight) は元々小さく既定の組数でも
100 MB に届かないため、起動時にその旨の警告が出る (HC のような量子化重み行列を
持たないという正しい結果)。

モデルは読まない (重みは乱数の量子化重みで代用)。`--count-forward` だけ例外で、
実モデル (既定 ~/models/ddalcu-mlxlm) を読んで S=1 decode forward を 1 回走らせ、
`mx.fast.metal_kernel` が返す呼び出し可能オブジェクトを薄いカウンタで包んで、
forward 1 回あたりのカスタムカーネル呼び出し回数を名前別に数える (モデル読み込みに
1 分以上かかるので既定 off、使うときも 1 回だけにすること)。

    uv run python tools/kernel_chain_cost.py --out bench/results/kernel-chain-cost.json
    uv run python tools/kernel_chain_cost.py --count-forward --model ~/models/ddalcu-mlxlm

最後に os._exit(0) で落ちる。
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
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# 実寸・plain 実装は micro_kernel_latency.py のものをそのまま流用する
# (写しを作ると shape/式がずれる)。
from micro_kernel_latency import (  # noqa: E402
    CONV_DIM,
    CONV_KERNEL,
    DK,
    DV,
    HC,
    HC_LOWRANK,
    HIDDEN,
    KEY_DIM,
    N_K,
    N_V,
    RMS_EPS,
    VALUE_DIM,
    WEIGHT_SET_DEFAULTS,
    _combine,
    _cycle_index,
    _quant_linear,
    _report_weight_sets,
    _weight_bytes,
    plain_gated_residual,
    plain_gdn_prework,
)

WARMUP = 20
N_CHAIN = 200
N_PAIRS = 3  # ABBA x n_pairs = 側ごと 2*n_pairs 試行 (既定 6)


# ----------------------------------------------------------------- 汎用計測


def _chain_once(step_fn, init_carry, n):
    """依存連鎖を n 個、1 回も eval せずに組んでから 1 回だけ mx.eval する。

    step_fn(carry) -> (next_carry, extra_leaves) の形。extra_leaves は連鎖
    (carry) には乗らないが、乗せないと MLX の遅延評価に刈られて実際には
    計算されなくなる出力 (chain に無関係な副出力) をここへ集める。carry・
    extra ともに、この関数の最後で 1 回だけまとめて mx.eval する。
    戻り値は (build_s, eval_s, final_carry)。
    """
    import mlx.core as mx

    carry = init_carry
    to_eval: list = []
    t0 = time.perf_counter()
    for _ in range(n):
        carry, extra = step_fn(carry)
        to_eval.extend(extra)
    t1 = time.perf_counter()
    to_eval.extend(carry if isinstance(carry, (tuple, list)) else (carry,))
    mx.eval(*to_eval)
    t2 = time.perf_counter()
    return (t1 - t0), (t2 - t1), carry


def _warmup_step(step_fn, init_carry, n):
    """カーネルの初回コンパイル等を計測窓の外に出すため、個別 eval しながら
    n 回使い捨てる (計測対象外)。"""
    import mlx.core as mx

    carry = init_carry
    for _ in range(n):
        carry, extra = step_fn(carry)
        leaves = list(extra) + list(carry if isinstance(carry, (tuple, list)) else (carry,))
        mx.eval(*leaves)
    return carry


def _pack(build_us: list, eval_us: list, n_chain: int) -> dict:
    return {
        "n": len(eval_us),
        "build_us_median": round(statistics.median(build_us), 3),
        "build_us_per_call_median": round(statistics.median(build_us) / n_chain, 4),
        "eval_us_median": round(statistics.median(eval_us), 3),
        "us_per_call_median": round(statistics.median(eval_us) / n_chain, 4),
        "eval_us_min": round(min(eval_us), 3),
        "eval_us_max": round(max(eval_us), 3),
    }


def _bench_pair(label_a, step_a, init_a, label_b, step_b, init_b, n_chain, n_pairs, warmup):
    """A/B chain 比較: warmup 回を個別 eval して捨てたあと、ABBA を n_pairs 回
    (側ごと 2*n_pairs 回のチェーン試行)。各試行は毎回 init_a()/init_b() から
    改めて N=n_chain 本の連鎖を組んで 1 回 eval する (試行間で状態を持ち越さない)。
    """
    _warmup_step(step_a, init_a(), warmup)
    _warmup_step(step_b, init_b(), warmup)

    build_a: list = []
    eval_a: list = []
    build_b: list = []
    eval_b: list = []
    for _ in range(n_pairs):
        for step_fn, init_fn, b_list, e_list in (
            (step_a, init_a, build_a, eval_a),
            (step_b, init_b, build_b, eval_b),
            (step_b, init_b, build_b, eval_b),
            (step_a, init_a, build_a, eval_a),
        ):
            b_s, e_s, _ = _chain_once(step_fn, init_fn(), n_chain)
            b_list.append(b_s * 1e6)
            e_list.append(e_s * 1e6)

    a_summary = _pack(build_a, eval_a, n_chain)
    b_summary = _pack(build_b, eval_b, n_chain)
    return {
        label_a: a_summary,
        label_b: b_summary,
        "ratio_a_over_b": round(
            a_summary["us_per_call_median"] / b_summary["us_per_call_median"], 4
        ),
    }


def _bench_floor_items(items: dict, n_chain: int, n_pairs: int, warmup: int) -> dict:
    """fused/plain の対がない単独項目 (「参考の床」) を、1 プロセス内で
    総当たり順 (パリンドローム掃引、`verify_width_cost.sweep_order` と同じ理屈)
    に交互測定する。items: {label: (step_fn, init_fn)}。
    """
    for step_fn, init_fn in items.values():
        _warmup_step(step_fn, init_fn(), warmup)

    build_by_label = {label: [] for label in items}
    eval_by_label = {label: [] for label in items}
    forward_order = list(items.items())
    backward_order = list(reversed(forward_order))
    for _ in range(n_pairs):
        for label, (step_fn, init_fn) in forward_order + backward_order:
            b_s, e_s, _ = _chain_once(step_fn, init_fn(), n_chain)
            build_by_label[label].append(b_s * 1e6)
            eval_by_label[label].append(e_s * 1e6)

    return {
        label: {
            "n": len(eval_by_label[label]),
            "build_us_per_call_median": round(
                statistics.median(build_by_label[label]) / n_chain, 4
            ),
            "eval_us_median": round(statistics.median(eval_by_label[label]), 3),
            "us_per_call_median": round(
                statistics.median(eval_by_label[label]) / n_chain, 4
            ),
        }
        for label in items
    }


# --------------------------------------------------------------- 各項目の構築


def build_hc_items(n_sets: int | None = None):
    """1) HC (hyper-connections)。連鎖は「出力 hyper を次の入力 hyper にする」形
    (micro_kernel_latency._bench_chained_hc_ab と同じ組み方)。mixed と inject の
    両方を _combine が直接消費するので、余計な extra_leaves は要らない。

    `n_sets` 組の重み (norm_weight/down/up/inject、各組別々の乱数) を起動時に
    まとめて作り、連鎖のステップ i が `weights[i % n_sets]` を巡回する
    (fused/plain は独立カウンタ)。既定は `WEIGHT_SET_DEFAULTS["hc"]` (97、HC の
    発火回数、docs/research/SESSION-2026-09-02-CATCHUP.md の「2026-09-03 12:00」
    節)。`n_sets=1` で旧来通り重み 1 組 (約 7 MB) を 200 回使い回す温キャッシュに
    戻る -- この温キャッシュこそが、並列度の低い自前カーネルが冷の DRAM
    レイテンシを隠せない負けを見えなくしていた当人 (HC 融合: 温 +13 us -> 冷
    +78 us/回)。判定ゲートは「重みを 100 MB 超巡回させた冷の連鎖」。
    """
    import mlx.core as mx

    from mlxturbo.kernels.hyper_connection import fused_gated_residual

    n_sets = n_sets or WEIGHT_SET_DEFAULTS["hc"]
    dtype = mx.bfloat16

    weight_sets = []
    for _ in range(n_sets):
        norm_weight = mx.zeros(HC * HIDDEN, dtype=dtype)
        down = _quant_linear(HC_LOWRANK, HC * HIDDEN, dtype)
        up = _quant_linear(HC * HIDDEN, HC_LOWRANK, dtype)
        inject = _quant_linear(HC, HC * HIDDEN, dtype)
        mx.eval(norm_weight, down, up, inject)
        weight_sets.append((norm_weight, down, up, inject))

    init_hyper = mx.random.normal((1, HC * HIDDEN)).astype(dtype)
    mx.eval(init_hyper)

    per_set_bytes = sum(_weight_bytes(t) for t in weight_sets[0])
    meta = _report_weight_sets("hc_gated_residual", per_set_bytes, n_sets)

    def _make_step(fn):
        idx = _cycle_index(n_sets)

        def step(hyper):
            norm_weight, down, up, inject = weight_sets[next(idx)]
            mixed, inj = fn(hyper, norm_weight, RMS_EPS, HC, HIDDEN, down, up, inject)
            return _combine(hyper, mixed, inj), []

        return step

    fused_step = _make_step(fused_gated_residual)
    plain_step = _make_step(plain_gated_residual)
    return fused_step, plain_step, (lambda: init_hyper), meta


def build_gdn_recurrent_items(n_sets: int | None = None):
    """2) GDN recurrent step (S=1)。連鎖は状態 (state, fp32) だけを次段へ渡す。
    q/k/v/a/b は固定 (HC の重みや floor 項目の weight と同じ「据え置き入力」
    扱い)。どちらの実装も 1 回の Metal 起動で y と state を一緒に出すので、
    y を使わなくても kernel の起動そのものは省略されない (extra_leaves 不要)。

    `n_sets` 組の A_log/dt_bias (実体は per-layer 学習パラメータ、「重み」に
    相当) を巡回する (fused/plain は独立カウンタ)。q/k/v/a/b は本物の decode
    でも毎トークン計算し直す活性化であって重みではないので、従来通り固定入力
    のまま (巡回しない)。既定は `WEIGHT_SET_DEFAULTS["gdn"]` (36、GDN 層数)。
    A_log/dt_bias は 1 組あたり数百バイトしかないので、既定の組数を渡しても
    100 MB には遠く届かず起動時に警告が出る -- それ自体が「GDN recurrent には
    HC のような大きい重み (量子化行列) を読む冷キャッシュ問題が無い」という
    正しい結果 (CATCHUP の敗因整理でも GDN recurrent は custom kernel 同士の
    比較で、HC のような custom-vs-plain の負けの対象に含まれていない)。
    """
    import mlx.core as mx

    from mlx_lm.models.gated_delta import gated_delta_update
    from mlxturbo.kernels.gated_delta_states import gated_delta_update_with_states

    n_sets = n_sets or WEIGHT_SET_DEFAULTS["gdn"]
    dtype = mx.bfloat16
    q = mx.random.normal((1, 1, N_K, DK)).astype(dtype)
    k = mx.random.normal((1, 1, N_K, DK)).astype(dtype)
    v = mx.random.normal((1, 1, N_V, DV)).astype(dtype)
    a = mx.random.normal((1, 1, N_V)).astype(dtype)
    b = mx.random.normal((1, 1, N_V)).astype(dtype)
    mx.eval(q, k, v, a, b)

    weight_sets = []
    for _ in range(n_sets):
        A_log = mx.random.normal((N_V,)).astype(mx.float32)
        dt_bias = mx.random.normal((N_V,)).astype(mx.float32)
        mx.eval(A_log, dt_bias)
        weight_sets.append((A_log, dt_bias))

    init_state = mx.random.normal((1, N_V, DV, DK)).astype(mx.float32)
    mx.eval(init_state)

    per_set_bytes = sum(_weight_bytes(t) for t in weight_sets[0])
    meta = _report_weight_sets("gdn_recurrent", per_set_bytes, n_sets)

    idx_fused = _cycle_index(n_sets)
    idx_plain = _cycle_index(n_sets)

    def fused_step(state):
        A_log, dt_bias = weight_sets[next(idx_fused)]
        _out, states_all = gated_delta_update_with_states(
            q, k, v, a, b, A_log, dt_bias, state, None
        )
        return states_all[:, -1], []

    def plain_step(state):
        A_log, dt_bias = weight_sets[next(idx_plain)]
        _out, state_out = gated_delta_update(
            q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=True
        )
        return state_out, []

    return fused_step, plain_step, (lambda: init_state), meta


def build_gdn_prework_items(n_sets: int | None = None):
    """3) GDN 前処理。連鎖は (mixed_qkv, conv_state) の 2 つを次段へ渡す。

    plain_gdn_prework は「conv1d -> silu -> rms_norm」の枝と「concat + slice
    (次段 conv_state)」の枝が MLX のグラフ上は独立している (新しい conv_state
    は conv1d の出力を一切参照しない)。conv_state だけを chain に乗せると、
    plain 側は高い方の枝 (実際に重い conv1d/rms_norm) を一切計算しない graph
    になってしまい、素の実装が不当に速く見える。これを避けるため、
    q/k/v を conv_dim 幅に詰め直して次段の mixed_qkv として連鎖させる
    (KEY_DIM + KEY_DIM + VALUE_DIM == CONV_DIM なので形が合う) --
    これで conv1d 以降の全計算が次段の入力の本物の祖先になる。
    g/beta はどちらの chain 変数にも乗らないので extra_leaves で明示的に
    毎回 eval 対象へ入れる (fused は 1 dispatch なのでどのみち計算されるが、
    plain は g/beta が完全に独立した op 列なので、入れないと一度も実行されない)。

    `n_sets` 組の conv1d.weight (CONV_DIM x CONV_KERNEL、per-layer 学習
    パラメータ) と A_log/dt_bias を巡回する (fused/plain は独立カウンタ)。
    mixed_qkv/conv_state は chain の carry (前段の出力)、a/b は毎トークンの
    活性化なのでどちらも巡回しない。既定は `WEIGHT_SET_DEFAULTS["gdn"]`
    (36、GDN 層数)。conv1d 重みは 1 組 ~80 KB しかないので、既定の組数を
    渡しても 100 MB には遠く届かず起動時に警告が出る -- build_gdn_recurrent_items
    と同じ理由 (HC のような大きい量子化重みが無い) で正しい結果。
    """
    import mlx.core as mx
    import mlx.nn as nn

    from mlxturbo.kernels.gdn_prework import fused_gdn_prework

    n_sets = n_sets or WEIGHT_SET_DEFAULTS["gdn"]
    dtype = mx.bfloat16
    a = mx.random.normal((1, 1, N_V)).astype(dtype)
    b = mx.random.normal((1, 1, N_V)).astype(dtype)
    mx.eval(a, b)

    weight_sets = []
    for _ in range(n_sets):
        conv1d = nn.Conv1d(CONV_DIM, CONV_DIM, kernel_size=CONV_KERNEL, groups=CONV_DIM, bias=False)
        conv1d.weight = mx.random.normal((CONV_DIM, CONV_KERNEL, 1)).astype(dtype)
        A_log = mx.random.normal((N_V,)).astype(mx.float32)
        dt_bias = mx.random.normal((N_V,)).astype(mx.float32)
        mx.eval(conv1d.weight, A_log, dt_bias)
        weight_sets.append((conv1d, conv1d.weight, A_log, dt_bias))

    init_mixed_qkv = mx.random.normal((1, 1, CONV_DIM)).astype(dtype)
    init_conv_state = mx.random.normal((1, CONV_KERNEL - 1, CONV_DIM)).astype(dtype)
    mx.eval(init_mixed_qkv, init_conv_state)

    per_set_bytes = sum(_weight_bytes(t) for t in weight_sets[0][1:])  # conv1d モジュール自体は数えない
    meta = _report_weight_sets("gdn_prework", per_set_bytes, n_sets)

    def _repack_qkv(q, k, v):
        return mx.concatenate(
            [q.reshape(1, 1, KEY_DIM), k.reshape(1, 1, KEY_DIM), v.reshape(1, 1, VALUE_DIM)],
            axis=-1,
        )

    idx_fused = _cycle_index(n_sets)
    idx_plain = _cycle_index(n_sets)

    def fused_step(carry):
        mixed_qkv, conv_state = carry
        _conv1d, conv_w, A_log, dt_bias = weight_sets[next(idx_fused)]
        q, k, v, g, beta, conv_state_out = fused_gdn_prework(
            mixed_qkv, conv_state, conv_w, a, b, A_log, dt_bias,
            N_K, N_V, DK, DV, KEY_DIM, VALUE_DIM, RMS_EPS,
        )
        return (_repack_qkv(q, k, v), conv_state_out), [g, beta]

    def plain_step(carry):
        mixed_qkv, conv_state = carry
        conv1d, _conv_w, A_log, dt_bias = weight_sets[next(idx_plain)]
        q, k, v, g, beta, conv_state_out = plain_gdn_prework(
            mixed_qkv, conv_state, conv1d, a, b, A_log, dt_bias,
            N_K, N_V, DK, DV, KEY_DIM, VALUE_DIM,
        )
        return (_repack_qkv(q, k, v), conv_state_out), [g, beta]

    init = lambda: (init_mixed_qkv, init_conv_state)  # noqa: E731
    return fused_step, plain_step, init, meta


def _plain_rms_norm_gated(x, weight, gate, eps):
    """`RMSNormGated.__call__` の素の 3 op 列 (rms_norm_gated.py の docstring
    の「元の狙い」の式そのもの)。gate は fp32 に上げてから sigmoid する
    (bf16 の丸めが gate 経路に無いのは参照と同じ)。"""
    import mlx.core as mx

    out = mx.fast.rms_norm(x, weight, eps)
    g = mx.sigmoid(gate.astype(mx.float32))
    return (g * out.astype(mx.float32)).astype(x.dtype)


def build_rms_norm_gated_items(n_sets: int | None = None):
    """4) RMSNormGated。GDN の実寸 (48 head x 128) で、rms_norm_gated.py の
    docstring 通り x=(B,S,n_v,dv)。連鎖は出力を次の x にする (gate は固定の
    据え置き入力)。

    `n_sets` 組の weight ((DV,)、per-layer 学習パラメータ) を巡回する
    (fused/plain は独立カウンタ)。gate は毎トークンの活性化なので巡回しない。
    既定は `WEIGHT_SET_DEFAULTS["gdn"]` (36、GDN 層数)。weight は 1 組 256 B
    しかないので、既定の組数を渡しても 100 MB には遠く届かず起動時に警告が
    出る -- build_gdn_recurrent_items と同じ理由で正しい結果。
    """
    import mlx.core as mx

    from mlxturbo.kernels.rms_norm_gated import rms_norm_gated

    n_sets = n_sets or WEIGHT_SET_DEFAULTS["gdn"]
    dtype = mx.bfloat16

    weight_sets = []
    for _ in range(n_sets):
        w = mx.random.normal((DV,)).astype(dtype)
        mx.eval(w)
        weight_sets.append(w)

    gate = mx.random.normal((1, 1, N_V, DV)).astype(dtype)
    init_x = mx.random.normal((1, 1, N_V, DV)).astype(dtype)
    mx.eval(gate, init_x)

    per_set_bytes = _weight_bytes(weight_sets[0])
    meta = _report_weight_sets("rms_norm_gated", per_set_bytes, n_sets)

    idx_fused = _cycle_index(n_sets)
    idx_plain = _cycle_index(n_sets)

    def fused_step(x):
        weight = weight_sets[next(idx_fused)]
        return rms_norm_gated(x, weight, gate, RMS_EPS, "sigmoid"), []

    def plain_step(x):
        weight = weight_sets[next(idx_plain)]
        return _plain_rms_norm_gated(x, weight, gate, RMS_EPS), []

    return fused_step, plain_step, (lambda: init_x), meta


def build_rms_norm_floor_item():
    """5a) mx.fast.rms_norm 単体 (MLX 組み込みの融合 op、1 dispatch)。"""
    import mlx.core as mx

    dtype = mx.bfloat16
    weight = mx.ones((HIDDEN,), dtype=dtype)
    init_x = mx.random.normal((1, HIDDEN)).astype(dtype)
    mx.eval(weight, init_x)

    def step(x):
        return mx.fast.rms_norm(x, weight, RMS_EPS), []

    return step, (lambda: init_x)


def build_elementwise_floor_item():
    """5b) `y = x + 1.0` 単体 (MLX 組み込みの素朴な elementwise、1 dispatch)。"""
    import mlx.core as mx

    dtype = mx.bfloat16
    init_x = mx.zeros((1, HIDDEN), dtype=dtype)
    mx.eval(init_x)

    def step(x):
        return x + 1.0, []

    return step, (lambda: init_x)


_FLOOR_ADD_KERNEL = None


def _get_floor_add_kernel():
    """5c) 手書きの最小 `mx.fast.metal_kernel` (2560 要素の加算)。組み込み op
    (5a/5b) と比べて、自前カーネル起動そのものの固定費がどれだけ上乗せに
    なるかの目安。"""
    global _FLOOR_ADD_KERNEL
    if _FLOOR_ADD_KERNEL is None:
        import mlx.core as mx

        _FLOOR_ADD_KERNEL = mx.fast.metal_kernel(
            name="kernel_chain_cost_floor_add",
            input_names=["x"],
            output_names=["y"],
            source=f"""
                uint i = thread_position_in_grid.x;
                if (i < {HIDDEN}) {{
                    float v = (float)x[i];
                    y[i] = (T)(v + 1.0f);
                }}
            """,
        )
    return _FLOOR_ADD_KERNEL


def build_metal_kernel_floor_item():
    import mlx.core as mx

    dtype = mx.bfloat16
    kern = _get_floor_add_kernel()
    init_x = mx.zeros((HIDDEN,), dtype=dtype)
    mx.eval(init_x)

    def step(x):
        (y,) = kern(
            inputs=[x],
            template=[("T", dtype)],
            grid=(HIDDEN, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(HIDDEN,)],
            output_dtypes=[dtype],
        )
        return y, []

    return step, (lambda: init_x)


# --------------------------------------------------------------- --count-forward


def run_count_forward(args) -> dict:
    """~/models/ddalcu-mlxlm (既定) を読み、S=1 decode forward を 1 回走らせて、
    `mx.fast.metal_kernel` が返す呼び出し可能オブジェクトをカーネル名別に
    数える。読み方は tools/forward_split.py の名前空間差し替え (`_profiled_build`)
    と同じ発想: 対象そのもの (`mx.fast.metal_kernel`) を薄いクロージャで包み、
    呼ばれるたびにカウントしてから本体を呼ぶ。

    素の nanobind オブジェクトの `__call__` をインスタンス属性として差し替えても
    暗黙呼び出し (`kern(...)`) には効かない (Python の特殊メソッド解決は型を
    見る) ため、ファクトリ自体を差し替えて「カウントする Python クロージャ」を
    返す形にしている -- 効果としては「返ってきた呼び出し可能オブジェクトを包む」
    のと同じ。

    各モジュールはカーネルオブジェクトを `_KERNELS` 系の辞書に一度作ったら
    使い回すので、**このプロセスで最初にどれか 1 つでもカーネルが作られる前に
    パッチを入れる必要がある** (後から入れると、それより前に作られたカーネルは
    パッチを素通りしてしまい、二度と数えられなくなる)。そのためパッチは
    prefill を含む全体にかけたうえで、測りたい decode forward の直前に
    `counts` を空にしてから、その forward 1 回だけを数える。
    """
    import mlx.core as mx

    from decode_ab import _restore, prefill_once
    from mlxturbo.spec_flash import capture
    from verify_width_cost import build_pair, build_prompt_ids, build_runner

    counts: dict[str, int] = {}
    orig_metal_kernel = mx.fast.metal_kernel

    def counting_factory(*a, **kw):
        name = kw.get("name")
        if name is None and a:
            name = a[0]
        name = name or "unknown"
        kern = orig_metal_kernel(*a, **kw)

        def counted(*ia, **ikw):
            counts[name] = counts.get(name, 0) + 1
            return kern(*ia, **ikw)

        return counted

    mx.fast.metal_kernel = counting_factory
    try:
        runner_args = argparse.Namespace(
            model=args.model, ngram=args.ngram, mtp=args.mtp, mtp_bits=args.mtp_bits,
        )
        eng, model, tok, eos_ids = build_runner(runner_args)
        ids = build_prompt_ids(tok, 0)
        caches, snap, resume, _first = prefill_once(eng, ids, eos_ids)
        pair, _cur = build_pair(eng, resume, 1)  # depth=0 なので MTP forward は挟まらない

        counts.clear()  # prefill などで先に作られた分の呼び出しを捨て、ここから数える
        _restore(caches, snap)
        with capture(model) as _cap:
            lg = model(pair, cache=caches)
        mx.eval(lg)
    finally:
        mx.fast.metal_kernel = orig_metal_kernel

    total = sum(counts.values())
    return {
        "total_custom_kernel_calls": total,
        "by_name": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }


# --------------------------------------------------------------------- 表示


def _print_pair_row(title, pair_result, label_a="fused", label_b="plain"):
    a = pair_result[label_a]
    b = pair_result[label_b]
    print(f"-- {title} --")
    print(
        f"  {label_a:32s}  {a['us_per_call_median']:9.3f} us/call"
        f"  (build {a['build_us_per_call_median']:.3f})"
    )
    print(
        f"  {label_b:32s}  {b['us_per_call_median']:9.3f} us/call"
        f"  (build {b['build_us_per_call_median']:.3f})"
    )
    print(f"  ratio {label_a}/{label_b}: {pair_result['ratio_a_over_b']:.3f}")


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bench/results/kernel-chain-cost.json")
    ap.add_argument("--n-chain", type=int, default=N_CHAIN, help="直列依存チェーンの長さ (既定 200)")
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS, help="ABBA の対数 (既定 3 = 側ごと 6 試行)")
    ap.add_argument("--warmup", type=int, default=WARMUP, help="個別 eval で捨てる warmup 回数 (既定 20)")
    ap.add_argument(
        "--count-forward", action="store_true",
        help="重い (モデル読み込みに 1 分以上)。既定 off。実モデルで decode S=1 "
             "forward 1 回のカスタムカーネル呼び出し回数を名前別に数える。",
    )
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm", help="--count-forward 用")
    ap.add_argument("--ngram", default=None, help="--count-forward 用 (既定 未使用)")
    ap.add_argument("--mtp", default=None, help="--count-forward 用 (既定はモデルディレクトリ内の mtp.safetensors)")
    ap.add_argument("--mtp-bits", type=int, default=4, help="--count-forward 用")
    ap.add_argument(
        "--weight-sets", type=int, default=None,
        help=(
            "kernels (hc_gated_residual/gdn_recurrent/gdn_prework/rms_norm_gated) の"
            "連鎖が巡回する重みの組数。既定は項目ごとの層数 "
            f"(hc={WEIGHT_SET_DEFAULTS['hc']} / gdn={WEIGHT_SET_DEFAULTS['gdn']}、"
            "冷キャッシュ)。1 を指定すると旧来の温キャッシュ (重み 1 組を使い回す) "
            "に戻る。floor 項目 (rms_norm_alone/elementwise_add/"
            "metal_kernel_add_2560) は対象外 (重み自体を持たない/無視できるほど"
            "小さい起動費の床の参照値なので巡回しても意味がない)。"
        ),
    )
    args = ap.parse_args()

    n_chain, n_pairs, warmup = args.n_chain, args.n_pairs, args.warmup

    print(
        f"N_CHAIN={n_chain}  N_PAIRS={n_pairs} (ABBA x n_pairs = 側ごと {2 * n_pairs} 試行)"
        f"  WARMUP={warmup}\n"
        f"--weight-sets={args.weight_sets if args.weight_sets is not None else '(既定、項目ごと)'}"
        f"  (既定値: {WEIGHT_SET_DEFAULTS})\n"
    )

    result: dict = {
        "meta": {
            "note": (
                "N 本を eval せずに直列依存で組んでから 1 回だけ mx.eval し、"
                "eval 時間 / N を 1 回あたりの費用とする。tools/micro_kernel_latency.py "
                "は毎回 eval するため同期 (~200us) が混ざって 1 回の費用が読めない、その補い。"
                "絶対値は熱・キャッシュ状態に依存する目安 (CLAUDE.md の計測の作法参照)。"
                "採否は必ず in-model A/B で決めること。既定は --weight-sets で重みを"
                "層数ぶん巡回させる冷キャッシュ (判定ゲート: 重み 100 MB 超巡回の冷の"
                "連鎖で素の op 列より速いこと)。--weight-sets 1 で旧来の温キャッシュに戻る。"
            ),
            "n_chain": n_chain,
            "n_pairs": n_pairs,
            "warmup": warmup,
            "weight_set_defaults": WEIGHT_SET_DEFAULTS,
            "dims": {
                "hidden": HIDDEN, "hc": HC, "hc_lowrank": HC_LOWRANK,
                "n_k": N_K, "n_v": N_V, "dk": DK, "dv": DV,
                "key_dim": KEY_DIM, "value_dim": VALUE_DIM, "conv_dim": CONV_DIM,
                "conv_kernel": CONV_KERNEL,
            },
        },
        "kernels": {},
        "floor": {},
    }
    weight_meta: dict = {}

    fused_step, plain_step, init, wm = build_hc_items(args.weight_sets)
    weight_meta["hc_gated_residual"] = wm
    hc = _bench_pair("fused", fused_step, init, "plain", plain_step, init, n_chain, n_pairs, warmup)
    result["kernels"]["hc_gated_residual"] = hc
    _print_pair_row("hc_gated_residual (fused_gated_residual, pre+post 2 kernel)", hc)

    fused_step, plain_step, init, wm = build_gdn_recurrent_items(args.weight_sets)
    weight_meta["gdn_recurrent"] = wm
    gdn_rec = _bench_pair(
        "gated_delta_update_with_states", fused_step, init,
        "mlx_lm_gated_delta_update", plain_step, init, n_chain, n_pairs, warmup,
    )
    result["kernels"]["gdn_recurrent"] = gdn_rec
    _print_pair_row(
        "gdn_recurrent", gdn_rec,
        "gated_delta_update_with_states", "mlx_lm_gated_delta_update",
    )

    fused_step, plain_step, init, wm = build_gdn_prework_items(args.weight_sets)
    weight_meta["gdn_prework"] = wm
    gdn_pre = _bench_pair("fused", fused_step, init, "plain", plain_step, init, n_chain, n_pairs, warmup)
    result["kernels"]["gdn_prework"] = gdn_pre
    _print_pair_row("gdn_prework (fused_gdn_prework)", gdn_pre)

    fused_step, plain_step, init, wm = build_rms_norm_gated_items(args.weight_sets)
    weight_meta["rms_norm_gated"] = wm
    rmsg = _bench_pair("fused", fused_step, init, "plain", plain_step, init, n_chain, n_pairs, warmup)
    result["kernels"]["rms_norm_gated"] = rmsg
    _print_pair_row("rms_norm_gated", rmsg)

    result["meta"]["weight_sets"] = weight_meta

    # 床の 3 項目は --weight-sets の対象外 (rms_norm_alone の weight は HIDDEN 分
    # ~5KB、他 2 つは重みを持たない。起動費そのものの床であって weight-sets が
    # 検証したい「量子化重み行列の DRAM 読み出し」とは無関係)。
    rms_step, rms_init = build_rms_norm_floor_item()
    add_step, add_init = build_elementwise_floor_item()
    mk_step, mk_init = build_metal_kernel_floor_item()
    floor = _bench_floor_items(
        {
            "rms_norm_alone": (rms_step, rms_init),
            "elementwise_add": (add_step, add_init),
            "metal_kernel_add_2560": (mk_step, mk_init),
        },
        n_chain, n_pairs, warmup,
    )
    result["floor"] = floor
    print("\n-- 参考の床 (カーネル起動そのものの固定費の目安) --")
    for label, s in floor.items():
        print(
            f"  {label:24s}  {s['us_per_call_median']:9.3f} us/call"
            f"  (build {s['build_us_per_call_median']:.3f})"
        )

    print(
        "\n=== まとめ (項目 / fused|A us/call / plain|B us/call / 比) ==="
    )
    for title, pr, la, lb in (
        ("hc_gated_residual", hc, "fused", "plain"),
        ("gdn_recurrent", gdn_rec, "gated_delta_update_with_states", "mlx_lm_gated_delta_update"),
        ("gdn_prework", gdn_pre, "fused", "plain"),
        ("rms_norm_gated", rmsg, "fused", "plain"),
    ):
        a_us = pr[la]["us_per_call_median"]
        b_us = pr[lb]["us_per_call_median"]
        print(f"  {title:20s}  {a_us:9.3f}  {b_us:9.3f}  {pr['ratio_a_over_b']:6.3f}")

    if args.count_forward:
        print(
            "\n--count-forward: モデルを読んで decode S=1 forward 1 回のカスタム"
            "カーネル呼び出し回数を数える (重い)。"
        )
        cf = run_count_forward(args)
        result["count_forward"] = cf
        print(f"  合計 {cf['total_custom_kernel_calls']} 回 (S=1 decode forward 1 回)")
        for name, n in cf["by_name"].items():
            print(f"    {n:6d}  {name}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out_path}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
