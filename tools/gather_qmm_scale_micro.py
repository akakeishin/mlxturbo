"""D5 (MoE の専門家共有) の proof-of-life。

## 目的

decode 幅の `mx.gather_qmm` (`sorted_indices=True`) の時間が「行数」で
スケールするのか「異なる専門家の数」でスケールするのかを、モデルを読まずに
測る。行数でスケールするなら、複数トークンが同じ専門家を選んだときにその
専門家ぶんの計算をまとめる案 (D5、専門家共有) は効かない (行を増やしても
減らしても時間が変わらないだけで、専門家をまとめる意味が無い)。逆に
専門家数でスケールするなら、同じ専門家を選ぶトークンをまとめて 1 回の
呼び出しに畳むことで時間を削れる可能性がある。

## 重みの形

Flash-Next (qwen4_exp) の `SparseMoeBlock`/`SwitchGLU`
(`mlxturbo/_vendor/qwen4_exp.py`, `mlx_lm.models.switch_layers.SwitchGLU`) と
同じ形: 512 専門家 x (hidden_size=2560 -> moe_intermediate_size=640) の
gate_proj/up_proj と、640 -> 2560 の down_proj。4-bit, group_size 64
(`QuantizedSwitchLinear` と同じ `mx.quantize` の呼び方)。

## rhs_indices の形について (過去に shape エラーが出た点)

`mx.gather_qmm` は「x の最後の 2 軸より前の軸」をバッチ軸として扱い、
`rhs_indices` はそのバッチ軸への添字を取る。x を `(rows, in_dim)`
(バッチ軸が空) のまま渡すと、rows 個の添字それぞれに対して x 全体
(rows 行) を掛けてしまい `(len(indices), rows, out_dim)` という総当たりの
出力になる (行ごとの 1 対 1 対応にならない)。`QuantizedSwitchLinear.__call__`
と同じく x を `(rows, 1, in_dim)` にしてから `rhs_indices` を `(rows,)` の
flat な 1 次元配列として渡すと、`(rows, 1, out_dim)` の行ごとの出力になる
(CPU の極小形状で実測して確認済み)。本ファイルはこの `(rows, 1, in_dim)`
+ flat `(rows,)` の形で統一する。

## ケース

行は専門家 id で昇順ソート済みとする (`sorted_indices=True` の前提)。

  - (rows=22, distinct=20): 22 行のうち 2 専門家だけ 2 行、残り 18 専門家は 1 行
  - (rows=22, distinct=11): 22 行を 11 専門家に 2 行ずつ均等配分
  - (rows=11, distinct=11): 1 専門家 1 行
  - (rows=1,  distinct=1) : 最小形

## 計測

gate_proj/up_proj/down_proj (SwiGLU 1 段) を 1 セットとして `--chain` 回
(既定 50)、前のセットの出力 (down_proj の出力) の一部を次のセットの入力に
混ぜて直列に連ね、1 回の `mx.eval` の壁時計を計る。ウォームアップ 2 回、
本番 `--reps` 回 (既定 5) の中央値を `--chain` で割り、1 セットあたりの us
を出す。

## 判定基準

  - (22,20) と (22,11) の差 (対称相対差 = |a-b| / ((a+b)/2)) が 10% 未満
    -> 行数でスケール。専門家共有 (D5) は死んでいる
  - 25% 以上 -> 専門家数でスケール。D5 に進む価値あり
  - どちらでもない -> 判定保留

## 使い方 (GPU 必須。このファイルを書いた時点では実行していない)

    tools/biglock.sh .venv/bin/python tools/gather_qmm_scale_micro.py \
        --json bench/results/gather-qmm-scale-micro.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

# Flash-Next (qwen4_exp) の SparseMoeBlock/SwitchGLU と同じ形
# (mlxturbo/_vendor/qwen4_exp.py の SparseMoeBlock、
# mlx_lm.models.switch_layers.QuantizedSwitchLinear と同じ量子化引数)。
NUM_EXPERTS = 512
HIDDEN = 2560
INTER = 640
GROUP_SIZE, BITS = 64, 4

CASES = [
    (22, 20),
    (22, 11),
    (11, 11),
    (1, 1),
]


def _make_case_indices(rows: int, distinct: int) -> list[int]:
    """`rows` 行を `distinct` 個の専門家 (id は 0 起点の連番) へできるだけ
    均等に割り当て、昇順ソート済みの添字列を返す。余りは前方の専門家に 1 行
    ずつ足す (例: rows=22,distinct=20 -> 先頭 2 専門家だけ 2 行)。
    """
    if not (1 <= distinct <= rows):
        raise ValueError(f"distinct={distinct} は 1..rows={rows} の範囲でない")
    base, rem = divmod(rows, distinct)
    counts = [base + (1 if i < rem else 0) for i in range(distinct)]
    idx: list[int] = []
    for expert_id, c in enumerate(counts):
        idx.extend([expert_id] * c)
    assert len(idx) == rows
    return idx


def _quantize_expert_weight(seed: int, out_dim: int, in_dim: int):
    mx.random.seed(seed)
    w = mx.random.uniform(
        low=-0.02, high=0.02, shape=(NUM_EXPERTS, out_dim, in_dim)
    ).astype(mx.bfloat16)
    return mx.quantize(w, group_size=GROUP_SIZE, bits=BITS, mode="affine")


def make_weights(seed: int = 0):
    """gate_proj/up_proj (2560->640) と down_proj (640->2560) を量子化して作る。"""
    gate = _quantize_expert_weight(seed, INTER, HIDDEN)
    up = _quantize_expert_weight(seed + 1, INTER, HIDDEN)
    down = _quantize_expert_weight(seed + 2, HIDDEN, INTER)
    return gate, up, down


def _gather(x, w, idx):
    wq, sc, bi = w
    return mx.gather_qmm(
        x, wq, sc, bi, rhs_indices=idx, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, mode="affine", sorted_indices=True,
    )


def moe_set(x, idx, gate, up, down):
    """gate/up/down の SwiGLU 1 段。x: (rows,1,HIDDEN) -> (rows,1,HIDDEN)。"""
    g = _gather(x, gate, idx)
    u = _gather(x, up, idx)
    act = (nn.silu(g) * u).astype(mx.bfloat16)
    return _gather(act, down, idx)


def bench_case(
    rows: int, distinct: int, gate, up, down, chain: int, reps: int, seed: int
) -> float:
    """`chain` 回の直列連鎖を 1 回の eval で計り、1 セットあたりの us (中央値) を返す。"""
    idx = mx.array(_make_case_indices(rows, distinct))
    mx.random.seed(seed + rows * 10_000 + distinct)
    x0 = mx.random.normal((rows, 1, HIDDEN)).astype(mx.bfloat16)
    mixes = [mx.random.normal((rows, 1, HIDDEN)).astype(mx.bfloat16) for _ in range(chain)]
    mx.eval(idx, x0, mixes)

    def go():
        x = x0
        for i in range(chain):
            out = moe_set(x, idx, gate, up, down)
            # 前セットの出力の一部を次の入力に混ぜる (最適化で消えないように)
            x = (0.1 * out + 0.9 * mixes[i]).astype(mx.bfloat16)
        return x

    for _ in range(2):
        mx.eval(go())

    samples_us = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(go())
        samples_us.append((time.perf_counter() - t0) * 1e6 / chain)
    return statistics.median(samples_us)


def _sym_diff_pct(a: float, b: float) -> float:
    """対称相対差 (%)。閾値判定は基準を固定しないほうが頑健なのでこちらを使う。"""
    denom = (a + b) / 2
    if denom == 0:
        return 0.0
    return abs(a - b) / denom * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", type=int, default=50, help="1 回の eval に積むセット数")
    ap.add_argument("--reps", type=int, default=5, help="ウォームアップ後の反復数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise SystemExit(
            "GPU が既定デバイスでない。gather_qmm の decode 幅の挙動を測るものなので "
            "GPU 専用 (CPU では絶対値もスケーリングも参考にならない)"
        )

    gate, up, down = make_weights(args.seed)
    mx.eval(gate, up, down)

    rows_us: dict[tuple[int, int], float] = {}
    print(f"chain={args.chain} reps={args.reps}")
    print(f"{'rows':>5s} {'distinct':>9s} {'us/set':>10s}")
    for rows, distinct in CASES:
        us = bench_case(rows, distinct, gate, up, down, args.chain, args.reps, args.seed)
        rows_us[(rows, distinct)] = us
        print(f"{rows:5d} {distinct:9d} {us:10.2f}")

    us_22_20 = rows_us[(22, 20)]
    us_22_11 = rows_us[(22, 11)]
    diff_pct = _sym_diff_pct(us_22_20, us_22_11)
    if diff_pct < 10.0:
        verdict = "行数でスケール: 専門家共有 (D5) は死んでいる"
    elif diff_pct >= 25.0:
        verdict = "専門家数でスケール: D5 に進む価値あり"
    else:
        verdict = "判定保留: 10-25% の間 (--chain/--reps を増やすか再計測)"

    print(
        f"判定: (22,20)={us_22_20:.2f}us vs (22,11)={us_22_11:.2f}us の対称相対差 "
        f"= {diff_pct:.1f}% (10% 未満なら行数でスケール、25% 以上なら専門家数でスケール)"
        f" -> {verdict}"
    )

    if args.json:
        out_dir = os.path.dirname(args.json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(
                {
                    "note": __doc__,
                    "chain": args.chain,
                    "reps": args.reps,
                    "rows_us": {f"{r}/{d}": us for (r, d), us in rows_us.items()},
                    "diff_pct_22_20_vs_22_11": diff_pct,
                    "verdict": verdict,
                },
                f,
                ensure_ascii=False,
                indent=1,
            )
        print("書き出し:", args.json)

    os._exit(0)


if __name__ == "__main__":
    main()
