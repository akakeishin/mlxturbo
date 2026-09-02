"""P3 (MoE grouped GEMM 自前カーネル) の入場料チェック: BM=16 タイルの
straddle (専門家境界がタイルの途中に来て K ループをやり直す) を、行の水増し
だけで潰した対比較。

## 背景

`docs/research/IDEAS-2026-09-03.md` の Challenge 表、P3 行:

    入場料: steel の平坦化で dense qmm の効率に届いた実績が無い。BM=16 の
    straddle は超過の 6〜8 割で、現実的な天井は 17k prefill -3〜4%
    天井スタブ: 全行を 1 専門家に流す (P4)。既製 gather_qmm でセグメントを
    16 行の倍数にダミー行で揃える対比較
    判定線: 揃えても dense の 1.10 倍以上なら主因は straddle でなくタイル形
    + DRAM 床

この機 (M3 Max、Apple Silicon の NAX 拡張なし) の `mx.gather_qmm`
(`sorted_indices=True`) は `affine_gather_qmm_rhs` (非 nax 版、
`gather_qmm_rhs_nax` の M/E 依存 bm=32/64 とは別物) を通り、BM=16 固定。
タイル内で専門家境界をまたぐと (straddle)、その境界ごとに K ループを
やり直す (Challenger の確認)。本ファイルは Metal を書かず、既製の
`mx.gather_qmm` のまま**各専門家のセグメントを 16 行 (または 32 行) の
倍数にダミー行で切り上げて straddle を物理的に無くす**とどこまで dense 比
(= 同じ有効行数を単一の重みで `mx.quantized_matmul` した場合との比) が
下がるかを測る。

## 重みの形

Flash-Next (qwen4_exp) の `SparseMoeBlock`/`SwitchGLU` と同じ形
(`gather_qmm_scale_micro.py` と揃える): 512 専門家 x
(hidden_size=2560 -> moe_intermediate_size=640) の gate_proj/up_proj と
640 -> 2560 の down_proj。4-bit、group_size 64、`mode="affine"`。
`x` は `(rows, 1, in_dim)` + `rhs_indices` は flat `(rows,)`
(`gather_qmm_scale_micro.py` の docstring で確認済みの形)。

## ルーティングの合成 (実測分布に寄せる)

総行数は 81920 (chunk 幅 8192 相当、行/専門家の平均=160) と 20480
(chunk 幅 2048 相当、行/専門家の平均=40)。専門家ごとの行数は一様でなく、
`tools/moe_routing_skew.py` が実モデルから採った層平均
(`bench/results/moe-routing-skew.json`) に近い偏りを、Dirichlet(alpha) で
作った専門家選択確率から multinomial サンプリングして再現する::

    実測 (avg, 48層平均):
      rows=20480 (M/E=40): median 11.8-16.5  p90 100.8-104.3  max 800-925
                            zero_experts 72.6-99.4 (/512)  p10 0.02-0.19
      rows=81920 (M/E=160): median 57.4-62.8  p90 401.4-412.6  max 3264-3379
                            zero_experts 36.5-37.7 (/512)  p10 1.7-2.1

alpha は 0.2〜0.7 を試し、両方の総行数で median と zero_experts が良く
揃う alpha=0.4 (既定) を選んだ (下記、seed=0 での実測)::

    rows=20480: median=16.0  p90=106.9  max=560  zero=78  p10=0.0
    rows=81920: median=62.0  p90=423.7  max=2358  zero=45  p10=1.0

max だけは実測よりやや小さい (2358 対 3264〜3379 など) が、水増し実験で
効くのは行数の少ない専門家 (16/32 の切り上げで相対的に無駄が大きい) の
分布なので、max のずれは判定への影響が小さいと判断した。

## ケース

行は専門家 id で昇順ソート済み (`sorted_indices=True` の前提)。

  - (a) そのまま (無加工)
  - (b) 各専門家のセグメントを 16 行の倍数に切り上げ、ダミー行 (x=0、
    rhs_indices はその専門家の id) を足す
  - (c) 同じく 32 行の倍数
  - (d) dense: 同じ有効行数ぶんを単一の量子化重みで `mx.quantized_matmul`
    (gather なし、専門家スイッチなし。straddle も含めた全部のせの理想床)

(b)(c) はダミー行の分だけ (a) より総行数が増える。ms はそのまま比べつつ、
「有効行あたり」= ms を (有効行数/総行数) で按分して水増し分を割り引いた
値も出す (dense 比の分母は常に「有効行数で測った dense」)。

## 計測

各ケース gate/up/down (SwiGLU 1 段、3 回の量子化行列積) を 1 セットとして
1 回の `mx.eval`。ウォームアップ 2 回のあと、`--reps` 回 (既定 10) を
ABBA 式カウンターバランス (奇数ラウンドは正順、偶数ラウンドは逆順、
`two_stream_micro.py` と同じ一般化) で回し、ケースごとの中央値を取る
(CLAUDE.md 「A/B は 1 プロセス内で交互に測る」)。

## 判定基準

16 行揃え (b) の「有効行あたり dense 比」で判定する
(Challenger の判定線をそのまま使う):

  - 1.10 以上 -> straddle は主因でない。P3 (タイル形まで直す自前カーネル)
    でないと届かない
  - 1.05 以下 -> straddle が主因。既製 gather_qmm + パディングだけでも
    かなりの部分が取れる
  - どちらでもない -> 判定保留

## 使い方 (GPU 必須。このファイルを書いた時点では実行していない)

    tools/biglock.sh .venv/bin/python tools/gather_qmm_pad_micro.py \
        --json bench/results/gather-qmm-pad-micro.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

import mlx.core as mx
import mlx.nn as nn

# Flash-Next (qwen4_exp) の SparseMoeBlock/SwitchGLU と同じ形
# (gather_qmm_scale_micro.py と揃える)。
NUM_EXPERTS = 512
HIDDEN = 2560
INTER = 640
GROUP_SIZE, BITS = 64, 4

# `tools/moe_routing_skew.py` の実測 (bench/results/moe-routing-skew.json)
# に median/zero_experts を近づけるよう選んだ Dirichlet 集中度 (モジュール
# docstring 参照)。
DEFAULT_ALPHA = 0.4

CASE_ORDER = ["raw", "pad16", "pad32", "dense"]
CASE_LABEL = {"raw": "そのまま", "pad16": "16行揃え", "pad32": "32行揃え", "dense": "dense"}
CASE_MULTIPLE = {"raw": None, "pad16": 16, "pad32": 32}
# raw/pad16/pad32 で x の乱数列をずらすための固定オフセット (hash() はプロ
# セスごとに変わりうるので使わない)。
CASE_SEED_OFFSET = {"raw": 0, "pad16": 1, "pad32": 2}


# ---------------------------------------------------------------------------
# ルーティングの合成
# ---------------------------------------------------------------------------


def gen_routing_counts(total_rows: int, num_experts: int, alpha: float, seed: int) -> np.ndarray:
    """Dirichlet(alpha) で専門家ごとの選択確率を作り、multinomial で
    `total_rows` 個の行を専門家に配る (専門家 id は 0..num_experts-1)。
    """
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(num_experts, alpha))
    return rng.multinomial(total_rows, p)


def routing_stats(counts: np.ndarray) -> dict:
    return {
        "min": int(counts.min()),
        "p10": float(np.percentile(counts, 10)),
        "median": float(np.percentile(counts, 50)),
        "p90": float(np.percentile(counts, 90)),
        "max": int(counts.max()),
        "zero_experts": int((counts == 0).sum()),
    }


def pad_counts(counts: np.ndarray, multiple: int | None) -> np.ndarray:
    """専門家ごとの行数を `multiple` の倍数へ切り上げる (0 行の専門家は 0
    のまま、切り上げない)。`multiple=None` なら無加工のコピーを返す。
    """
    if not multiple:
        return counts.copy()
    return np.where(counts > 0, ((counts + multiple - 1) // multiple) * multiple, 0)


def build_indices_and_mask(counts: np.ndarray, padded_counts: np.ndarray):
    """専門家 id 昇順にソート済みの flat な `rhs_indices` (int32) と、
    「その行が本物 (True) かダミー (False、切り上げで足した 0 埋め行)」を
    示す bool マスクを返す。専門家 e のブロックは `padded_counts[e]` 行、
    先頭 `counts[e]` 行が本物。
    """
    idx_parts, mask_parts = [], []
    for e, (c, pc) in enumerate(zip(counts, padded_counts)):
        if pc <= 0:
            continue
        idx_parts.append(np.full(int(pc), e, dtype=np.int32))
        m = np.zeros(int(pc), dtype=bool)
        m[: int(c)] = True
        mask_parts.append(m)
    if not idx_parts:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=bool)
    return np.concatenate(idx_parts), np.concatenate(mask_parts)


# ---------------------------------------------------------------------------
# 重み
# ---------------------------------------------------------------------------


def _quantize_weight(seed: int, out_dim: int, in_dim: int, num_experts: int | None = None):
    mx.random.seed(seed)
    shape = (num_experts, out_dim, in_dim) if num_experts else (out_dim, in_dim)
    w = mx.random.uniform(low=-0.02, high=0.02, shape=shape).astype(mx.bfloat16)
    return mx.quantize(w, group_size=GROUP_SIZE, bits=BITS, mode="affine")


def make_weights(seed: int = 0):
    """gate/up/down の専門家別重みと、同じ形の単一 (dense) 重みを作る。"""
    gate = _quantize_weight(seed, INTER, HIDDEN, NUM_EXPERTS)
    up = _quantize_weight(seed + 1, INTER, HIDDEN, NUM_EXPERTS)
    down = _quantize_weight(seed + 2, HIDDEN, INTER, NUM_EXPERTS)
    gate_d = _quantize_weight(seed + 3, INTER, HIDDEN)
    up_d = _quantize_weight(seed + 4, INTER, HIDDEN)
    down_d = _quantize_weight(seed + 5, HIDDEN, INTER)
    return (gate, up, down), (gate_d, up_d, down_d)


# ---------------------------------------------------------------------------
# SwiGLU 1 段 (gather / dense)
# ---------------------------------------------------------------------------


def _gather(x, w, idx):
    wq, sc, bi = w
    return mx.gather_qmm(
        x, wq, sc, bi, rhs_indices=idx, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, mode="affine", sorted_indices=True,
    )


def moe_set(x, idx, gate, up, down):
    """x: (rows,1,HIDDEN) -> (rows,1,HIDDEN)。"""
    g = _gather(x, gate, idx)
    u = _gather(x, up, idx)
    act = (nn.silu(g) * u).astype(mx.bfloat16)
    return _gather(act, down, idx)


def _dense_mm(x, w):
    wq, sc, bi = w
    return mx.quantized_matmul(
        x, wq, sc, bi, transpose=True, group_size=GROUP_SIZE, bits=BITS, mode="affine"
    )


def dense_set(x, gate_d, up_d, down_d):
    """x: (rows,HIDDEN) -> (rows,HIDDEN)。gather なしの単一重み版。"""
    g = _dense_mm(x, gate_d)
    u = _dense_mm(x, up_d)
    act = (nn.silu(g) * u).astype(mx.bfloat16)
    return _dense_mm(act, down_d)


# ---------------------------------------------------------------------------
# ケースの構築 / 計測
# ---------------------------------------------------------------------------


def build_case_go(kind: str, counts: np.ndarray, weights, dense_weights, seed: int):
    """`kind` (CASE_ORDER のいずれか) の go() クロージャと
    (総行数, 有効行数) を返す。"""
    gate, up, down = weights
    gate_d, up_d, down_d = dense_weights
    valid_rows = int(counts.sum())

    if kind == "dense":
        mx.random.seed(seed + 80_000)
        x = mx.random.normal((valid_rows, HIDDEN)).astype(mx.bfloat16)
        mx.eval(x)

        def go():
            return dense_set(x, gate_d, up_d, down_d)

        return go, valid_rows, valid_rows

    multiple = CASE_MULTIPLE[kind]
    padded_counts = pad_counts(counts, multiple)
    idx_np, mask_np = build_indices_and_mask(counts, padded_counts)
    rows_total = int(idx_np.shape[0])
    idx = mx.array(idx_np)

    mx.random.seed(seed + 90_000 + CASE_SEED_OFFSET[kind])
    x_all = mx.random.normal((rows_total, HIDDEN)).astype(mx.bfloat16)
    mask_mx = mx.array(mask_np.astype(np.float32)).reshape(rows_total, 1)
    x = (x_all * mask_mx).astype(mx.bfloat16).reshape(rows_total, 1, HIDDEN)
    mx.eval(idx, x)

    def go():
        return moe_set(x, idx, gate, up, down)

    return go, rows_total, valid_rows


def bench_once(go) -> float:
    t0 = time.perf_counter()
    mx.eval(go())
    return (time.perf_counter() - t0) * 1e3


def run_scale(rows_scale: int, weights, dense_weights, alpha: float, seed: int, reps: int) -> dict:
    counts = gen_routing_counts(rows_scale, NUM_EXPERTS, alpha, seed)
    stats = routing_stats(counts)

    builders = {}
    rows_info = {}
    for kind in CASE_ORDER:
        go, rows_total, rows_valid = build_case_go(kind, counts, weights, dense_weights, seed)
        builders[kind] = go
        rows_info[kind] = (rows_total, rows_valid)

    # ウォームアップ (捨てる)
    for kind in CASE_ORDER:
        for _ in range(2):
            bench_once(builders[kind])

    # ABBA 式カウンターバランス (two_stream_micro.py と同じ一般化)
    samples: dict[str, list[float]] = {kind: [] for kind in CASE_ORDER}
    for r in range(reps):
        order = CASE_ORDER if r % 2 == 0 else CASE_ORDER[::-1]
        for kind in order:
            samples[kind].append(bench_once(builders[kind]))

    medians = {kind: statistics.median(v) for kind, v in samples.items()}
    dense_ms = medians["dense"]

    results = []
    for kind in CASE_ORDER:
        rows_total, rows_valid = rows_info[kind]
        ms = medians[kind]
        dense_ratio = ms / dense_ms if dense_ms else None
        eff_ms = ms * (rows_valid / rows_total) if rows_total else ms
        eff_ratio = eff_ms / dense_ms if dense_ms else None
        results.append({
            "case": kind, "label": CASE_LABEL[kind],
            "rows": rows_total, "rows_valid": rows_valid,
            "ms": ms, "dense_ratio": dense_ratio, "eff_dense_ratio": eff_ratio,
        })

    return {
        "rows_scale": rows_scale, "alpha": alpha,
        "routing_stats": stats, "samples_ms": samples,
        "median_ms": medians, "results": results,
    }


def print_table(res: dict) -> None:
    st = res["routing_stats"]
    print(f"[rows_scale={res['rows_scale']}] routing (alpha={res['alpha']}): "
          f"min={st['min']} p10={st['p10']:.1f} median={st['median']:.1f} "
          f"p90={st['p90']:.1f} max={st['max']} zero_experts={st['zero_experts']}"
          f" (/{NUM_EXPERTS})")
    print(f"  {'ケース':10s}{'行数':>8s}{'有効行':>8s}{'ms':>10s}"
          f"{'dense比':>9s}{'有効行dense比':>13s}")
    for r in res["results"]:
        print(f"  {r['label']:10s}{r['rows']:8d}{r['rows_valid']:8d}{r['ms']:10.3f}"
              f"{r['dense_ratio']:9.3f}{r['eff_dense_ratio']:13.3f}")


def verdict_for(eff_ratio_pad16: float) -> str:
    if eff_ratio_pad16 >= 1.10:
        return "straddle は主因でない (P3 はタイル形まで直さないと届かない)"
    if eff_ratio_pad16 <= 1.05:
        return "straddle が主因 (既製 gather_qmm + パディングでかなり取れる)"
    return "判定保留 (1.05〜1.10 の間、--reps を増やすか再計測)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="81920,20480",
                     help="合成する総行数。カンマ区切り (既定: 81920,20480)")
    ap.add_argument("--reps", type=int, default=10, help="ABBA カウンターバランスのラウンド数")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                     help="ルーティング合成の Dirichlet 集中度 (既定 0.4、モジュール docstring 参照)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise SystemExit(
            "GPU が既定デバイスでない。gather_qmm の BM=16 straddle を測る"
            "ものなので GPU 専用 (CPU では絶対値もタイル挙動も参考にならない)"
        )

    rows_modes = [int(v) for v in args.rows.split(",") if v.strip() != ""]

    weights, dense_weights = make_weights(args.seed)
    mx.eval(*weights, *dense_weights)

    print(f"reps={args.reps} alpha={args.alpha} rows={rows_modes}")
    print()

    all_results = []
    for rows_scale in rows_modes:
        res = run_scale(rows_scale, weights, dense_weights, args.alpha, args.seed, args.reps)
        all_results.append(res)
        print_table(res)
        print()

    print("判定 (16 行揃えの有効行あたり dense 比。"
          "1.10 以上で straddle 否定 / 1.05 以下で straddle 肯定):")
    verdicts = {}
    for res in all_results:
        pad16 = next(r for r in res["results"] if r["case"] == "pad16")
        eff = pad16["eff_dense_ratio"]
        v = verdict_for(eff)
        verdicts[res["rows_scale"]] = {"eff_dense_ratio_pad16": eff, "verdict": v}
        print(f"  rows={res['rows_scale']:6d}: 有効行あたりdense比={eff:.3f} -> {v}")

    if args.json:
        out_dir = os.path.dirname(args.json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(
                {
                    "note": __doc__,
                    "reps": args.reps,
                    "alpha": args.alpha,
                    "rows_modes": rows_modes,
                    "cases": all_results,
                    "verdicts": verdicts,
                },
                f,
                ensure_ascii=False,
                indent=1,
            )
        print("\n書き出し:", args.json)

    # 計測ツールなので destructor 待ちでプロセスが Metal のメモリを握ったまま
    # 残る前例がある (moe_routing_skew.py と同じ理由) -- 即 _exit で落とす
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
