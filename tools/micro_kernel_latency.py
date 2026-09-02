"""decode S=1 の固定費 (行数にも重み読み出しにも依らない部分) を、モデルを
読まずに合成テンソルだけで測る単体マイクロ。

うちの decode S=1 の固定費が相手より約 6ms 大きい件の仮説検証用:
融合カーネル 1 発の起動+同期コスト (HC・GDN 前処理) が相手より重く、
小さいカーネルの直列レイテンシが積もっているのではないか、という筋を見る。

CLAUDE.md の「温キャッシュのマイクロを信じない」は帯域 (実効 GB/s) の話。
ここで測るのは**起動と直列依存のレイテンシ**で、性質が違う道具なので目安として
使ってよいが、採否は必ず in-model A/B で決めること (micro で勝って in-model で
負けた前例が複数ある: moe_glu、moe_route、HC prefill 融合)。

    uv run python tools/micro_kernel_latency.py --out bench/results/micro-kernel-latency.json

モデルはロードしない (合成テンソルのみ)。最後に os._exit(0) で落ちる。
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

# 実寸 (mlxturbo/_vendor/qwen4_exp.py: TextArgs の既定値と同じ)。
HIDDEN = 2560
HC = 4
HC_LOWRANK = 320
N_K = 16
N_V = 48
DK = 128
DV = 128
KEY_DIM = N_K * DK  # 2048
VALUE_DIM = N_V * DV  # 6144
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 10240
CONV_KERNEL = 4
RMS_EPS = 1e-6
VOCAB = 248320
QBITS = 8
QGROUP = 64

WARMUP = 20
N_OP_CHAIN = 2000
TRIALS_OP = 50
N_HC = 200
N_GDN_PREWORK = 200
N_GDN_RECUR = 200
N_LM_HEAD = 100

OP_CHAIN_SHAPES = [(1, HIDDEN), (1, CONV_DIM), (1, N_V, DK)]


# --------------------------------------------------------------------- 補助


def _qmm_linear(x, w):
    """`mlxturbo/fused.py::_build` 内の qmm と同じ量子化線形。

    w は `(weight, scales, biases, group_size, bits)` (`fused._pack_quantized`
    や `mlxturbo/kernels/hyper_connection.py` の down/up/inject と同じ形)。
    """
    import mlx.core as mx

    wt, sc, bi, gs, bits = w
    return mx.quantized_matmul(x, wt, scales=sc, biases=bi, transpose=True, group_size=gs, bits=bits)


def _quant_linear(n_out, n_in, dtype, group_size=QGROUP, bits=QBITS):
    """`n_out x n_in` の量子化線形を合成データで作る (nn.Linear + 量子化と同じ形)。

    HC の down/up/inject と lm_head は `docs/research/KERNEL-HANDOFF-HC.md:316-318`
    (「HC の 3 層はレシピで明示的に 8bit を割り当てられている」) と
    `tools/bench_lm_head.py` の docstring (「8bit / group_size 64」) の通り、
    どちらも 8bit / group_size 64 なので既定値を揃えてある。
    """
    import mlx.core as mx

    w = mx.random.normal((n_out, n_in)).astype(dtype)
    wq, sc, bi = mx.quantize(w, group_size=group_size, bits=bits)
    return (wq, sc, bi, group_size, bits)


def _combine(hyper, x, inject):
    """`DecoderLayer._combine` (mlxturbo/_vendor/qwen4_exp.py:1521-1526) と同じ式。

    合成専用の写し (このファイルは既存のシームを一切変更しない)。HC カーネルの
    出力を次の hyper へ連鎖させ、固定入力を使い回す donation 等のアーチファクトを
    避けるために使う。
    """
    return hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)


def plain_gated_residual(hyper, norm_weight, eps, hc, d, down, up, inject):
    """`GatedResidual.__call__` + `RMSNorm.__call__` (group_size=d のレーンごと
    正規化、mlxturbo/_vendor/qwen4_exp.py:139-150, 1248-1257) を量子化タプルで
    組んだ素の MLX op 列。融合カーネル (`fused_gated_residual`) との比較対照。

    戻り値は `fused_gated_residual` の生の契約 (combine 有りなら `(mixed, inject)`)
    に揃える。呼び出し側 (`fused.enable_hyper_connection_kernel`) が付け足す
    `hyper` のパススルーはベンチには不要なので含めない。
    """
    import mlx.core as mx
    import mlx.nn as nn

    shape = hyper.shape
    x = hyper.reshape(*shape[:-1], hc, d)
    x = mx.fast.rms_norm(x, None, eps).reshape(shape)
    normed = x * (1.0 + norm_weight)
    w = nn.silu(_qmm_linear(normed, down) / hc)
    w = mx.sigmoid(_qmm_linear(w, up))
    w = w.reshape(*w.shape[:-1], hc, d)
    mixed = (w * normed.reshape(*normed.shape[:-1], hc, d)).mean(axis=-2)
    if inject is None:
        return mixed
    inj = 2 * mx.sigmoid(_qmm_linear(normed, inject) / hc)
    return mixed, inj


def plain_gdn_prework(mixed_qkv, conv_state, conv1d, a, b, A_log, dt_bias,
                       n_k, n_v, dk, dv, key_dim, value_dim):
    """`GatedDeltaNet.__call__` の conv1d 以降 (`mlxturbo/spec_flash.py` の
    capture 版 `gdn()`、mask なし分岐: conv_input の組み立てから q/k の
    rms_norm+スケールまで) に、g/beta の計算 (`mlx_lm.models.gated_delta` の
    `compute_g` / `sigmoid(b)`。`fused_gdn_prework` も同じ 2 つを出力するので
    揃える) を足した、素の MLX op 列。`fused_gdn_prework` との比較対照。

    戻り値は `fused_gdn_prework` と同じ順序 `(q, k, v, g, beta, conv_state_out)`。
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import compute_g

    B, S, _ = mixed_qkv.shape
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    conv_out = nn.silu(conv1d(conv_input))

    q, k, v = mx.split(conv_out, [key_dim, 2 * key_dim], axis=-1)
    q = q.reshape(B, S, n_k, dk)
    k = k.reshape(B, S, n_k, dk)
    v = v.reshape(B, S, n_v, dv)

    inv_scale = dk ** -0.5
    q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    g = compute_g(A_log, a, dt_bias)
    beta = mx.sigmoid(b)

    K = conv1d.weight.shape[1]
    new_conv_state = mx.contiguous(conv_input[:, -(K - 1):, :])
    return q, k, v, g, beta, new_conv_state


# --------------------------------------------------------------------- 計測


def _median_us(samples, warmup):
    kept = samples[warmup:]
    if not kept:
        raise ValueError(f"warmup={warmup} >= サンプル数={len(samples)}")
    return statistics.median(kept), kept


def _chain_trial(shape, n, dtype):
    """依存連鎖 1 試行: x = x + 1 を n 回つないでから 1 回だけ eval し、1 op あたりの us。"""
    import mlx.core as mx

    x = mx.zeros(shape, dtype=dtype)
    t0 = time.perf_counter()
    for _ in range(n):
        x = x + 1
    mx.eval(x)
    return (time.perf_counter() - t0) / n * 1e6


def _independent_trial(shape, n, dtype):
    """依存なし 1 試行: 同じ入力から n 個の独立な op をまとめて 1 回 eval し、1 op あたりの us。"""
    import mlx.core as mx

    base = mx.zeros(shape, dtype=dtype)
    t0 = time.perf_counter()
    outs = [base + 1 for _ in range(n)]
    mx.eval(outs)
    return (time.perf_counter() - t0) / n * 1e6


def _bench_fixed(fn, n):
    """同じ入力で fn() を n 回、毎回 eval して us を記録する (帯域ではなく起動+同期のレイテンシ)。"""
    import mlx.core as mx

    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - t0) * 1e6)
    return samples


def _bench_fixed_ab(fn_a, fn_b, n_pairs):
    """A/B 比較 (CLAUDE.md: 1 プロセス内で交互に測る): ABBA を n_pairs 回繰り返し、
    fn_a/fn_b それぞれ同じ入力で毎回 eval して us を記録する。ブロック測定
    (A を n 回 → B を n 回) だと熱・キャッシュ状態のドリフトが A/B 間で
    ずれるので、呼び出し順序を交互にして打ち消す。戻り値は
    (samples_a, samples_b) (各 2*n_pairs 個)。
    """
    import mlx.core as mx

    samples_a: list[float] = []
    samples_b: list[float] = []
    for _ in range(n_pairs):
        for fn, samples in (
            (fn_a, samples_a),
            (fn_b, samples_b),
            (fn_b, samples_b),
            (fn_a, samples_a),
        ):
            t0 = time.perf_counter()
            mx.eval(fn())
            samples.append((time.perf_counter() - t0) * 1e6)
    return samples_a, samples_b


def _bench_chained_hc_ab(step_fn_a, step_fn_b, init_hyper, n_pairs):
    """HC 用 A/B 比較: 各側は出力 (mixed, inject) を `_combine` で自分の hyper
    に連鎖させながら (固定入力の使い回しによるアーチファクトを避ける)、
    呼び出し順序だけ ABBA (CLAUDE.md: 1 プロセス内で交互) にする。

    `_combine` の結果は次周回の入力になるが、eval せずに返すと次周回の
    `step_fn` 呼び出し + eval がこの `_combine` の遅延グラフも一緒に評価する
    ことになり、前周回の後始末が次周回の計測窓に漏れる。ここでは
    `_combine` 直後に eval して窓の外で確定させてから次周回へ渡す。
    戻り値は (samples_a, samples_b) (各 2*n_pairs 個)。
    """
    import mlx.core as mx

    def _step(step_fn, hyper, samples):
        t0 = time.perf_counter()
        mixed, inj = step_fn(hyper)
        mx.eval(mixed, inj)
        samples.append((time.perf_counter() - t0) * 1e6)
        hyper = _combine(hyper, mixed, inj)
        mx.eval(hyper)  # 次入力の準備 (計測窓の外で確定させる)
        return hyper

    hyper_a = init_hyper
    hyper_b = init_hyper
    samples_a: list[float] = []
    samples_b: list[float] = []
    for _ in range(n_pairs):
        hyper_a = _step(step_fn_a, hyper_a, samples_a)
        hyper_b = _step(step_fn_b, hyper_b, samples_b)
        hyper_b = _step(step_fn_b, hyper_b, samples_b)
        hyper_a = _step(step_fn_a, hyper_a, samples_a)
    return samples_a, samples_b


def _summarize(samples, warmup):
    median_us, kept = _median_us(samples, warmup)
    return {
        "n": len(samples),
        "warmup": warmup,
        "kept": len(kept),
        "median_us": round(median_us, 3),
        "min_us": round(min(kept), 3),
        "max_us": round(max(kept), 3),
    }


# ------------------------------------------------------------------ 各項目


def run_op_chain():
    """1) 依存連鎖 vs 独立 op の 1 op あたり費用 (直列化の罰)。"""
    import mlx.core as mx

    out = {}
    for shape in OP_CHAIN_SHAPES:
        chain_samples = [
            _chain_trial(shape, N_OP_CHAIN, mx.bfloat16) for _ in range(TRIALS_OP)
        ]
        indep_samples = [
            _independent_trial(shape, N_OP_CHAIN, mx.bfloat16) for _ in range(TRIALS_OP)
        ]
        chain = _summarize(chain_samples, WARMUP)
        indep = _summarize(indep_samples, WARMUP)
        out[str(shape)] = {
            "n_ops": N_OP_CHAIN,
            "chain": chain,
            "independent": indep,
            "penalty_ratio": round(chain["median_us"] / indep["median_us"], 4),
        }
    return out


def run_hc_kernel():
    """2) HC 融合カーネル (fused_gated_residual) vs 素の GatedResidual 相当。"""
    import mlx.core as mx

    from mlxturbo.kernels.hyper_connection import fused_gated_residual

    dtype = mx.bfloat16
    norm_weight = mx.zeros(HC * HIDDEN, dtype=dtype)
    down = _quant_linear(HC_LOWRANK, HC * HIDDEN, dtype)
    up = _quant_linear(HC * HIDDEN, HC_LOWRANK, dtype)
    inject = _quant_linear(HC, HC * HIDDEN, dtype)
    init_hyper = mx.random.normal((1, HC * HIDDEN)).astype(dtype)
    mx.eval(norm_weight, down, up, inject, init_hyper)

    def fused_step(hyper):
        return fused_gated_residual(hyper, norm_weight, RMS_EPS, HC, HIDDEN, down, up, inject)

    def plain_step(hyper):
        return plain_gated_residual(hyper, norm_weight, RMS_EPS, HC, HIDDEN, down, up, inject)

    fused_samples, plain_samples = _bench_chained_hc_ab(fused_step, plain_step, init_hyper, N_HC // 2)
    fused = _summarize(fused_samples, WARMUP)
    plain = _summarize(plain_samples, WARMUP)
    return {
        "fused": fused,
        "plain": plain,
        "ratio_fused_over_plain": round(fused["median_us"] / plain["median_us"], 4),
    }


def run_gdn_prework():
    """3) GDN 前処理融合カーネル (fused_gdn_prework) vs 素の op 列。"""
    import mlx.core as mx
    import mlx.nn as nn

    from mlxturbo.kernels.gdn_prework import fused_gdn_prework

    dtype = mx.bfloat16
    mixed_qkv = mx.random.normal((1, 1, CONV_DIM)).astype(dtype)
    conv_state = mx.random.normal((1, CONV_KERNEL - 1, CONV_DIM)).astype(dtype)
    a = mx.random.normal((1, 1, N_V)).astype(dtype)
    b = mx.random.normal((1, 1, N_V)).astype(dtype)
    A_log = mx.zeros(N_V, dtype=mx.float32)
    dt_bias = mx.ones(N_V, dtype=mx.float32)

    conv1d = nn.Conv1d(CONV_DIM, CONV_DIM, kernel_size=CONV_KERNEL, groups=CONV_DIM, bias=False)
    conv1d.weight = mx.random.normal((CONV_DIM, CONV_KERNEL, 1)).astype(dtype)
    mx.eval(mixed_qkv, conv_state, a, b, A_log, dt_bias, conv1d.weight)

    def fused_step():
        return fused_gdn_prework(
            mixed_qkv, conv_state, conv1d.weight, a, b, A_log, dt_bias,
            N_K, N_V, DK, DV, KEY_DIM, VALUE_DIM,
        )

    def plain_step():
        return plain_gdn_prework(
            mixed_qkv, conv_state, conv1d, a, b, A_log, dt_bias,
            N_K, N_V, DK, DV, KEY_DIM, VALUE_DIM,
        )

    fused_samples, plain_samples = _bench_fixed_ab(fused_step, plain_step, N_GDN_PREWORK // 2)
    fused = _summarize(fused_samples, WARMUP)
    plain = _summarize(plain_samples, WARMUP)
    return {
        "fused": fused,
        "plain": plain,
        "ratio_fused_over_plain": round(fused["median_us"] / plain["median_us"], 4),
    }


def run_gdn_recurrent():
    """4) GDN 再帰カーネル (S=1): gated_delta_update_with_states vs
    mlx_lm.models.gated_delta.gated_delta_update(use_kernel=True)。"""
    import mlx.core as mx

    from mlx_lm.models.gated_delta import gated_delta_update
    from mlxturbo.kernels.gated_delta_states import gated_delta_update_with_states

    dtype = mx.bfloat16
    q = mx.random.normal((1, 1, N_K, DK)).astype(dtype)
    k = mx.random.normal((1, 1, N_K, DK)).astype(dtype)
    v = mx.random.normal((1, 1, N_V, DV)).astype(dtype)
    a = mx.random.normal((1, 1, N_V)).astype(dtype)
    b = mx.random.normal((1, 1, N_V)).astype(dtype)
    A_log = mx.zeros(N_V, dtype=mx.float32)
    dt_bias = mx.ones(N_V, dtype=mx.float32)
    state = mx.random.normal((1, N_V, DV, DK)).astype(mx.float32)
    mx.eval(q, k, v, a, b, A_log, dt_bias, state)

    def with_states_step():
        return gated_delta_update_with_states(q, k, v, a, b, A_log, dt_bias, state, None)

    def mlx_lm_step():
        return gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=True)

    with_states_samples, mlx_lm_samples = _bench_fixed_ab(with_states_step, mlx_lm_step, N_GDN_RECUR // 2)
    with_states = _summarize(with_states_samples, WARMUP)
    mlx_lm_summary = _summarize(mlx_lm_samples, WARMUP)
    return {
        "gated_delta_update_with_states": with_states,
        "mlx_lm_gated_delta_update": mlx_lm_summary,
        "ratio_with_states_over_mlx_lm": round(
            with_states["median_us"] / mlx_lm_summary["median_us"], 4
        ),
    }


def run_lm_head(lm_head_bits=QBITS):
    """5) lm_head の quantized_matmul (帯域の目安。融合との比較対象はない)。

    ビット数は `--lm-head-bits` で選ぶ (既定 8: 本番モデル
    `~/models/ddalcu-mlxlm` の config が lm_head 8bit/g64。
    `convert_flash.RECIPES["v-fast6"]` は 6bit なので、そちらを見たいときは
    `--lm-head-bits 6` を渡す)。帯域床とバイト数はここで渡すビット数から
    計算する。
    """
    import mlx.core as mx

    dtype = mx.bfloat16
    wq, sc, bi, gs, bits = _quant_linear(VOCAB, HIDDEN, dtype, bits=lm_head_bits)
    x = mx.random.normal((1, HIDDEN)).astype(dtype)
    mx.eval(wq, sc, bi, x)

    def step():
        return mx.quantized_matmul(x, wq, scales=sc, biases=bi, transpose=True, group_size=gs, bits=bits)

    samples = _bench_fixed(step, N_LM_HEAD)
    summary = _summarize(samples, WARMUP)

    k_words = wq.shape[1]
    k = k_words * 32 // bits
    nbytes = k * VOCAB * bits / 8 + sc.size * 2 + bi.size * 2
    floor_400gbps_us = nbytes / 400e9 * 1e6
    summary["weight_bytes"] = int(nbytes)
    summary["effective_gbps"] = round(nbytes / summary["median_us"] / 1000, 1)
    summary["bandwidth_floor_us_400gbps"] = round(floor_400gbps_us, 1)
    return summary


# ------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="bench/results/micro-kernel-latency.json")
    ap.add_argument(
        "--lm-head-bits", type=int, default=QBITS,
        help="lm_head の量子化ビット数 (既定 8。v-fast6 レシピを見たいときは 6)。"
             " HC の down/up/inject は対象外 (常に 8bit のまま)。",
    )
    args = ap.parse_args()

    result = {
        "meta": {
            "note": "起動+直列依存のレイテンシの目安。帯域のマイクロとは別物 (採否は in-model A/B)。",
            "dims": {
                "hidden": HIDDEN, "hc": HC, "hc_lowrank": HC_LOWRANK,
                "n_k": N_K, "n_v": N_V, "dk": DK, "dv": DV,
                "key_dim": KEY_DIM, "value_dim": VALUE_DIM, "conv_dim": CONV_DIM,
                "conv_kernel": CONV_KERNEL, "vocab": VOCAB,
                "qbits": QBITS, "qgroup": QGROUP,
                "lm_head_bits": args.lm_head_bits,
            },
            "warmup_discarded": WARMUP,
        },
        "op_chain": run_op_chain(),
        "hc_kernel": run_hc_kernel(),
        "gdn_prework": run_gdn_prework(),
        "gdn_recurrent": run_gdn_recurrent(),
        "lm_head": run_lm_head(args.lm_head_bits),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out_path}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
