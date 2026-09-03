"""prefill の dense 射影を 4-bit qmm のままにするか、bf16 に逆量子化して
`mx.matmul` で回すかを決める micro (P8, `docs/research/IDEAS-2026-09-03.md:258`)。

## 目的

GDN / attention / HC の dense 射影は現状すべて 4-bit affine group_size=64 の
`mx.quantized_matmul` (逆量子化込みで大きい M でも 10.3〜11.3 TFLOPS が上限)。
M3 Max の bf16 `mx.matmul` は 12〜13 TFLOPS 級と見込まれ、当たれば
prefill が 15〜20% 速くなる可能性がある。重みを bf16 へ戻す費用
(`mx.dequantize`、48 層分で 5 GB ≈ 12 ms/チャンクの見込み) を無視できるかも
含めて、層ごと dequant する価値があるかをこの micro で決める。
モデルは読まない (形だけ合わせた乱数の重み)。

## 形 (Flash-Next / qwen4_exp、bf16 activations)

`~/models/ddalcu-mlxlm/config.json` の `text_config` と
`mlxturbo/_vendor/qwen4_exp.py` の実装から実際の入出力幅を計算する
(nn.Linear と同じ (out_dim, in_dim) の重みで測る):

  - **gdn_in_proj**: 2560 -> 16480。`GatedDeltaNet.__init__` (qwen4_exp.py:1228)
    は `in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a` の 4 本の nn.Linear に
    分かれているが、`_project_in` のコメント (qwen4_exp.py:1249) 「4 本の
    入力射影...1 本の qmm に連結される」のとおり概念上は 1 本の射影で、
    `fused.enable_wide_projections` が実際にその連結を仕込む (fused.py:1244)。
    出力幅は conv_dim(10240) + value_dim(6144) + n_v(48) + n_v(48) = 16480
    (conv_dim = key_dim*2 + value_dim、key_dim = 128*16 = 2048、
    value_dim = 128*48 = 6144)。
  - **gdn_out_proj**: 6144 -> 2560 (`out_proj`: value_dim -> hidden_size)。
  - **attn_q_proj**: 2560 -> 12288 (`Attention.__init__` (qwen4_exp.py:615)、
    `q_proj` は output gate を抱えるので n_heads*head_dim*2 = 24*256*2)。
  - **attn_o_proj**: 6144 -> 2560 (n_heads*head_dim -> hidden_size)。
  - **hc_mix_down** / **hc_mix_up**: 10240 <-> 320
    (`GatedResidual.__init__` (qwen4_exp.py:1548)、
    hc_dim = hc_count(4)*hidden_size(2560) = 10240、hc_lowrank = 320。
    down は 10240->320、up は 320->10240)。

M は {2048, 8192} (prefill チャンクの典型幅)。

## 計測

1プロセス内で ABAB (× `--reps`、既定 5、ウォームアップ後の中央値) で
以下 3 通りを交互に測る:

  1. **qmm**: `mx.quantized_matmul(x, w_q, scales, biases, transpose=True,
     group_size=64, bits=4)`。
  2. **dequant込み**: `mx.dequantize(...)` を毎回 (ウォームアップ含め全反復で)
     呼んでから `mx.matmul` するところまでを時間に含める
     (層ごとに毎チャンク逆量子化する場合の実費用)。
  3. **bf16常駐**: `mx.dequantize` をあらかじめ 1 回だけ (計測外で) 済ませた
     bf16 の重みを使い回して `mx.matmul` だけを測る (逆量子化の固定費を
     償却できた場合の下限)。

TFLOPS は `2 * M * in_dim * out_dim / 時間` (積和を 2 flops として換算)。
数値差は、**同じ量子化重み** (`w_q, scales, biases` を 1 回だけ逆量子化した
`bf16常駐` の重み) を使った bf16 matmul と、qmm 内部で逆量子化した結果を
比べる。同じ重みの積和なので丸めの差だけのはず (`max|diff|` と相対 RMS)。

## 判定基準

形×M ごとに:

  - **bf16常駐が qmm の 0.85 倍以下**、かつ
  - **dequant込みが qmm の 0.90 倍以下**

の両方を満たせば「層ごと dequant を検討する価値あり」。どちらか一方でも
満たさなければ qmm のまま見送り (bf16 常駐だけ速くても、逆量子化の固定費が
乗ると prefill では取り返せない可能性が高い)。

## 使い方 (GPU 必須。このファイルを書いた時点では実行していない。
   親が `tools/biglock.sh` 経由で流す)

    tools/biglock.sh .venv/bin/python tools/dequant_gemm_micro.py \
        --json bench/results/dequant-gemm-micro.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import mlx.core as mx

GROUP_SIZE, BITS = 64, 4

# ~/models/ddalcu-mlxlm/config.json の text_config (Flash-Next / qwen4_exp)。
HIDDEN = 2560                      # hidden_size
LINEAR_KEY_HEAD_DIM = 128          # linear_key_head_dim
LINEAR_VALUE_HEAD_DIM = 128        # linear_value_head_dim
LINEAR_NUM_KEY_HEADS = 16          # linear_num_key_heads
LINEAR_NUM_VALUE_HEADS = 48        # linear_num_value_heads
N_HEADS = 24                       # num_attention_heads
HEAD_DIM = 256                     # head_dim
HC_COUNT = 4                       # hc_count
HC_LOWRANK = 320                   # hc_lowrank

# GatedDeltaNet.__init__ (_vendor/qwen4_exp.py:1228) の派生形。
KEY_DIM = LINEAR_KEY_HEAD_DIM * LINEAR_NUM_KEY_HEADS         # 2048
VALUE_DIM = LINEAR_VALUE_HEAD_DIM * LINEAR_NUM_VALUE_HEADS   # 6144
CONV_DIM = KEY_DIM * 2 + VALUE_DIM                             # 10240
# in_proj_qkv/z/b/a (4 本) を連結した後の出力幅 (fused.enable_wide_projections
# / qwen4_exp.py:1249 のコメントのとおり、概念上は 1 本の射影)。
GDN_IN_PROJ_N = CONV_DIM + VALUE_DIM + LINEAR_NUM_VALUE_HEADS * 2   # 16480
GDN_OUT_PROJ_K = VALUE_DIM                                            # 6144

# Attention.__init__ (_vendor/qwen4_exp.py:615)。q_proj は output gate 込み。
ATTN_Q_PROJ_N = N_HEADS * HEAD_DIM * 2   # 12288
ATTN_O_PROJ_K = N_HEADS * HEAD_DIM       # 6144

# GatedResidual.__init__ (_vendor/qwen4_exp.py:1548、hyper-connections)。
HC_DIM = HC_COUNT * HIDDEN               # 10240

# (name, in_dim, out_dim) -- nn.Linear(in_dim, out_dim) と同じ向き。
SHAPES: list[tuple[str, int, int]] = [
    ("gdn_in_proj", HIDDEN, GDN_IN_PROJ_N),
    ("gdn_out_proj", GDN_OUT_PROJ_K, HIDDEN),
    ("attn_q_proj", HIDDEN, ATTN_Q_PROJ_N),
    ("attn_o_proj", ATTN_O_PROJ_K, HIDDEN),
    ("hc_mix_down", HC_DIM, HC_LOWRANK),
    ("hc_mix_up", HC_LOWRANK, HC_DIM),
]

M_VALUES = (2048, 8192)

# (json 用キー, 表示用ラベル)
VARIANTS = [("qmm", "qmm"), ("dequant_incl", "dequant込み"), ("bf16_resident", "bf16常駐")]


def make_weight(seed: int, out_dim: int, in_dim: int):
    """nn.Linear と同じ (out_dim, in_dim) の bf16 重みを作り 4bit 量子化する。

    返り値の `w_resident` は `wq/scales/biases` を 1 回だけ逆量子化した
    bf16 の重み (「bf16常駐」ケースおよび数値差の比較対象はこれを使う --
    qmm と同じ量子化パラメータから来ているので、差は丸めの差だけになる)。
    """
    mx.random.seed(seed)
    w = (mx.random.normal((out_dim, in_dim)) * 0.02).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    del w
    w_resident = mx.dequantize(
        wq, sc, bi, group_size=GROUP_SIZE, bits=BITS, mode="affine", dtype=mx.bfloat16
    )
    mx.eval(wq, sc, bi, w_resident)
    return w_resident, wq, sc, bi


def qmm(x, wq, sc, bi):
    return mx.quantized_matmul(
        x, wq, scales=sc, biases=bi, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, mode="affine",
    )


def dequant_then_matmul(x, wq, sc, bi):
    """毎回逆量子化してから matmul。層ごとに毎チャンク dequant する場合の実費用。"""
    w = mx.dequantize(
        wq, sc, bi, group_size=GROUP_SIZE, bits=BITS, mode="affine", dtype=mx.bfloat16
    )
    return mx.matmul(x, w.T)


def bf16_matmul(x, w_resident):
    return mx.matmul(x, w_resident.T)


def bench_abab(cases: dict, reps: int, warmup: int = 2) -> dict[str, float]:
    """1 プロセス内で交互に測る。各ケースを個別にウォームアップしたあと、
    rep ごとに順方向/逆方向を入れ替えながら壁時計 (ms) を取り、中央値を返す。
    """
    names = list(cases)
    for n in names:
        for _ in range(warmup):
            mx.eval(cases[n]())
    samples: dict[str, list[float]] = {n: [] for n in names}
    for r in range(reps):
        order = names if r % 2 == 0 else list(reversed(names))
        for n in order:
            t0 = time.perf_counter()
            mx.eval(cases[n]())
            samples[n].append((time.perf_counter() - t0) * 1e3)
    return {n: statistics.median(v) for n, v in samples.items()}


def numeric_diff(x, w_resident, wq, sc, bi) -> tuple[float, float]:
    """同じ量子化重みを使った bf16 matmul と qmm の出力差 (max|diff|、相対 RMS)。"""
    out_bf16 = mx.matmul(x, w_resident.T).astype(mx.float32)
    out_qmm = qmm(x, wq, sc, bi).astype(mx.float32)
    mx.eval(out_bf16, out_qmm)
    diff = out_bf16 - out_qmm
    max_abs = float(mx.max(mx.abs(diff)))
    denom = float(mx.sqrt(mx.mean(out_bf16 ** 2)))
    rel_rms = float(mx.sqrt(mx.mean(diff ** 2))) / (denom + 1e-12)
    return max_abs, rel_rms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5, help="ウォームアップ後の反復数 (ABAB)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise SystemExit(
            "GPU が既定デバイスでない。qmm/dequant/matmul の TFLOPS を測るものなので "
            "GPU 専用 (CPU では絶対値が参考にならない)"
        )

    rows: list[dict] = []
    print(f"reps={args.reps} (ABAB, ウォームアップ後の中央値)  group_size={GROUP_SIZE} bits={BITS}")
    print(
        f"{'shape':<14s} {'M':>6s} {'in':>7s} {'out':>7s} {'variant':<14s} "
        f"{'ms':>10s} {'TFLOPS':>8s} {'比(qmm=1)':>10s}"
    )

    for shape_idx, (name, in_dim, out_dim) in enumerate(SHAPES):
        w_resident, wq, sc, bi = make_weight(args.seed + shape_idx * 1_000_000, out_dim, in_dim)

        # 数値差は M に依らない (要素ごとの丸め差) ので、最大の M で 1 回だけ確認する。
        mx.random.seed(args.seed + shape_idx * 1000 + 999)
        x_num = (mx.random.normal((max(M_VALUES), in_dim)) * 0.02).astype(mx.bfloat16)
        mx.eval(x_num)
        max_abs, rel_rms = numeric_diff(x_num, w_resident, wq, sc, bi)
        del x_num

        for m in M_VALUES:
            mx.random.seed(args.seed + shape_idx * 1000 + m)
            x = (mx.random.normal((m, in_dim)) * 0.02).astype(mx.bfloat16)
            mx.eval(x)

            cases = {
                "qmm": lambda: qmm(x, wq, sc, bi),
                "dequant_incl": lambda: dequant_then_matmul(x, wq, sc, bi),
                "bf16_resident": lambda: bf16_matmul(x, w_resident),
            }
            ms = bench_abab(cases, args.reps)

            flops = 2.0 * m * in_dim * out_dim
            row = {"shape": name, "M": m, "in_dim": in_dim, "out_dim": out_dim}
            for key, label in VARIANTS:
                t_ms = ms[key]
                tflops = flops / (t_ms * 1e-3) / 1e12
                ratio = t_ms / ms["qmm"]
                row[f"{key}_ms"] = t_ms
                row[f"{key}_tflops"] = tflops
                row[f"{key}_ratio_vs_qmm"] = ratio
                print(
                    f"{name:<14s} {m:6d} {in_dim:7d} {out_dim:7d} {label:<14s} "
                    f"{t_ms:10.3f} {tflops:8.2f} {ratio:10.3f}"
                )

            bf16_ratio = row["bf16_resident_ratio_vs_qmm"]
            dequant_ratio = row["dequant_incl_ratio_vs_qmm"]
            pass_bar = bf16_ratio <= 0.85 and dequant_ratio <= 0.90
            row["max_abs_diff"] = max_abs
            row["rel_rms_diff"] = rel_rms
            row["pass_bar"] = pass_bar
            verdict = "層ごと dequant を検討する価値あり" if pass_bar else "見送り (qmm のまま)"
            print(
                f"  判定: bf16常駐/qmm={bf16_ratio:.3f} (基準 <=0.85 かつ) "
                f"dequant込み/qmm={dequant_ratio:.3f} (基準 <=0.90) -> {verdict}"
            )
            print(f"  数値差 (同じ量子化重み): max|diff|={max_abs:.6g}  相対RMS={rel_rms:.6g}")
            rows.append(row)
            del x

        del w_resident, wq, sc, bi

    n_pass = sum(1 for r in rows if r["pass_bar"])
    print(f"\n総合: {n_pass}/{len(rows)} 形×M が基準 (bf16常駐<=0.85 かつ dequant込み<=0.90) を満たす")
    if n_pass == len(rows):
        overall = "全形で層ごと dequant が有望 -> prefill 側で本実装を検討"
    elif n_pass == 0:
        overall = "全形で qmm のまま -> 層ごと dequant は見送り"
    else:
        overall = "形依存 -> 基準を満たした形 (行の pass_bar) だけ dequant を検討"
    print(f"総合判定: {overall}")

    if args.json:
        out_dir = os.path.dirname(args.json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(
                {
                    "note": __doc__,
                    "group_size": GROUP_SIZE,
                    "bits": BITS,
                    "reps": args.reps,
                    "rows": rows,
                    "n_pass": n_pass,
                    "n_total": len(rows),
                    "overall_verdict": overall,
                },
                f,
                ensure_ascii=False,
                indent=1,
            )
        print("書き出し:", args.json)

    os._exit(0)


if __name__ == "__main__":
    main()
