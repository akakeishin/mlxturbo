"""MoE grouped GEMM レーン (P3) の proof-of-life。

## 第 1 段 (`--stage dense`)

`mlxturbo/kernels/_steel_flat.py` に平坦化した steel が、素の
`mx.quantized_matmul(transpose=True)` と

  - **ビット一致する** (`mx.array_equal`。近似では不可)
  - **1.10 倍以内の時間で回る** (同一プロセス内 ABBA、ウォームアップ後の中央値)

かを見る。どちらかが落ちたら `mx.fast.metal_kernel` 経路の固定費で勝ち目が
無いということなので、セグメント対応 (第 2 段) には進まない
(反転条件は `docs/research/IDEAS-2026-09-03.md` の P3)。

形は Flash-Next の MoE そのもの (512 専門家 x 2560 -> 640):

  - gate/up: M=20480 (2048 tok x top-k 10), K=2560, N=640
  - down   : M=20480,                        K=640,  N=2560

## 第 2 段 (`--stage segmented`)

`qmm_segmented` を、素の `mx.gather_qmm(sorted_indices=True)` と

  - **ビット一致する** (合成ルーティングで)
  - 同じ総行数の `mx.quantized_matmul` (dense) に対する比が、素の
    gather_qmm の dense 比より小さい

かで見る。判定線は r=160 (81920 行) で dense 比 1.14 未満 (`IDEAS` の P3)。

ルーティングは 2 種類:

  - `skew-r40` / `skew-r160`: `bench/results/moe-routing-skew.json` の
    実測分布の形 (2048 tok で median 20 / p90 90 / max ~800 / 行を持たない
    専門家 67、8192 tok はその 4 倍) を順序統計量から起こしたもの
  - `flat20` / `flat32` / `flat160`: 全専門家が同じ行数。**端数タイルの
    費用 c (フルタイル比) を直接出すため**にある。flat20 と flat32 は
    どちらもタイル数が E ちょうどで、片方だけが端数なので、
    c = t(flat20) / t(flat32)。flat160 はフルタイルだけの対照
    (1 タイルあたりの時間が flat32 と揃うことを見る)

## 使い方 (GPU。必ず lock 経由で)

    tools/biglock.sh .venv/bin/python tools/moe_grouped_gemm_micro.py \
        --stage dense --json bench/results/moe-grouped-gemm-dense.json
    tools/biglock.sh .venv/bin/python tools/moe_grouped_gemm_micro.py \
        --stage segmented --json bench/results/moe-grouped-gemm-segmented.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable

import mlx.core as mx
import numpy as np

from mlxturbo.kernels.moe_grouped_gemm import (
    BITS,
    BM,
    ENV_KNOB,
    GROUP_SIZE,
    dense_eligible,
    enabled,
    is_nax_device,
    n_tiles_max,
    qmm_dense_clone,
    qmm_segmented,
    segment_tables,
    segmented_eligible,
)

# (名前, M, K, N)。M は 2048 tok x top-k 10 の行数
SHAPES = [
    ("gate/up", 20480, 2560, 640),
    ("down", 20480, 640, 2560),
]


def _timed(fn: Callable[[], mx.array]) -> float:
    """1 回の呼びの壁時計 (ms)。"""
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    return (time.perf_counter() - t0) * 1e3


def _abba(
    stock: Callable[[], mx.array],
    clone: Callable[[], mx.array],
    rounds: int,
    warmup: int,
) -> tuple[list[float], list[float]]:
    """1 プロセス内で A B B A の順に交互に測る (熱とキャッシュを揃える)。"""
    for _ in range(warmup):
        mx.eval(stock())
        mx.eval(clone())

    a: list[float] = []
    b: list[float] = []
    for _ in range(rounds):
        a.append(_timed(stock))
        b.append(_timed(clone))
        b.append(_timed(clone))
        a.append(_timed(stock))
    return a, b


def _first_diff(ref: mx.array, got: mx.array) -> dict:
    """最初に食い違う要素の位置と値 (ビット一致しなかったときの手掛かり)。"""
    ne = ref != got
    flat = mx.flatten(ne)
    idx = int(mx.argmax(flat.astype(mx.int32)).item())
    row, col = divmod(idx, ref.shape[1])
    return {
        "flat_index": idx,
        "row": row,
        "col": col,
        "ref": float(ref[row, col].item()),
        "got": float(got[row, col].item()),
        "n_diff": int(mx.sum(ne.astype(mx.int32)).item()),
    }


def run_dense(rounds: int, warmup: int, seed: int) -> list[dict]:
    results = []
    for name, M, K, N in SHAPES:
        mx.random.seed(seed)
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        w_fp = mx.random.normal((N, K)).astype(mx.bfloat16)
        wq, scales, biases = mx.quantize(w_fp, group_size=GROUP_SIZE, bits=BITS)
        mx.eval(x, wq, scales, biases)
        del w_fp

        assert dense_eligible(x, wq, scales, biases), f"{name} の形が適格でない"

        def stock() -> mx.array:
            return mx.quantized_matmul(
                x,
                wq,
                scales,
                biases,
                transpose=True,
                group_size=GROUP_SIZE,
                bits=BITS,
            )

        def clone() -> mx.array:
            return qmm_dense_clone(x, wq, scales, biases)

        y_ref = stock()
        y_got = clone()
        mx.eval(y_ref, y_got)
        exact = bool(mx.array_equal(y_ref, y_got).item())
        max_diff = float(
            mx.max(mx.abs(y_ref.astype(mx.float32) - y_got.astype(mx.float32))).item()
        )
        entry: dict = {
            "shape": name,
            "M": M,
            "K": K,
            "N": N,
            "bit_exact": exact,
            "max_abs_diff": max_diff,
        }
        if not exact:
            entry["first_diff"] = _first_diff(y_ref, y_got)
        del y_ref, y_got

        a, b = _abba(stock, clone, rounds, warmup)
        stock_ms = statistics.median(a)
        clone_ms = statistics.median(b)
        entry.update(
            {
                "stock_ms": stock_ms,
                "clone_ms": clone_ms,
                "ratio": clone_ms / stock_ms,
                "stock_samples": len(a),
                "clone_samples": len(b),
            }
        )
        results.append(entry)

        print(
            f"{name:8s} M={M} K={K} N={N}  "
            f"stock {stock_ms:7.3f} ms  clone {clone_ms:7.3f} ms  "
            f"比 {clone_ms / stock_ms:5.3f}  "
            f"ビット一致 {'yes' if exact else 'NO'}  max|diff| {max_diff:g}"
        )
        if not exact:
            print(f"         最初の差: {entry['first_diff']}")

        del x, wq, scales, biases
    return results


# ---------------------------------------------------------------------------
# 第 2 段 (--stage segmented)
# ---------------------------------------------------------------------------

NUM_EXPERTS = 512

# (名前, K, N)。本番の SwitchGLU の 3 本 (gate と up は同じ形)
SEG_SHAPES = [
    ("gate/up", 2560, 640),
    ("down", 640, 2560),
]


def _rand_u32(shape: tuple[int, ...]) -> mx.array:
    """32 bit 全域の乱数 (mlx 0.32.2 に `mx.random.bits` が無いので 16 bit x 2)。

    重みの中身は比較にも時間にも効かない (素とこちらが同じ配列を読む) が、
    上位 16 bit を 0 にすると 4 bit の値が半分ゼロに偏るので繋いでおく。
    """

    hi = mx.random.randint(0, 1 << 16, shape, dtype=mx.uint32)
    lo = mx.random.randint(0, 1 << 16, shape, dtype=mx.uint32)
    return mx.left_shift(hi, mx.array(16, dtype=mx.uint32)) | lo


def _inv_cdf(q: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(q)


def skew_counts(
    num_experts: int, total: int, n_zero: int, median: float, p90: float
) -> np.ndarray:
    """実測の偏りの形に合わせた専門家ごとの行数を作る。

    `bench/results/moe-routing-skew.json` が持っているのは分位点の要約
    (min / p10 / median / p90 / max / 行を持たない専門家の数) だけなので、
    分布族を当てるのではなく**順序統計量を直に置く**: 下から `n_zero` 本は
    0、残りは対数正規の分位点を並べ、`median` と `p90` の 2 点で
    (尺度, 対数の標準偏差) を決める。最後に合計が `total` になるよう
    揃える。乱数を使わないので seed で揺れない。
    """

    nz = num_experts - n_zero
    if nz <= 2:
        raise ValueError("行を持つ専門家が少なすぎる")

    def q_at(pos: float) -> float:
        # 全 num_experts 本を昇順に並べたときの位置 pos が、0 でない側の
        # どの分位点に当たるか
        return (pos - n_zero + 0.5) / nz

    z_med = _inv_cdf(q_at((num_experts - 1) / 2.0))
    z_p90 = _inv_cdf(q_at(0.9 * (num_experts - 1)))
    sigma = np.log(p90 / median) / (z_p90 - z_med)
    scale = median / np.exp(sigma * z_med)

    counts = np.zeros(num_experts, dtype=np.int64)
    qs = (np.arange(nz) + 0.5) / nz
    zs = np.array([_inv_cdf(float(q)) for q in qs])
    vals = scale * np.exp(sigma * zs)
    counts[n_zero:] = np.maximum(1, np.round(vals * total / vals.sum())).astype(
        np.int64
    )

    # 合計を total ぴったりに寄せる (余りは一番大きい専門家で吸う)
    diff = total - int(counts.sum())
    counts[-1] += diff
    if counts[-1] < 1:
        raise ValueError("total が小さすぎて形が保てない")
    return counts


def counts_stats(counts: np.ndarray, bm: int) -> dict:
    nz = counts[counts > 0]
    tiles = np.ceil(counts / bm).astype(np.int64)
    full = (counts // bm).sum()
    return {
        "rows": int(counts.sum()),
        "experts_active": int((counts > 0).sum()),
        "zero_experts": int((counts == 0).sum()),
        "median": float(np.median(counts)),
        "p90": float(np.percentile(counts, 90)),
        "max": int(counts.max()),
        "tiles_total": int(tiles.sum()),
        "tiles_full": int(full),
        "tiles_partial": int(tiles.sum() - full),
        "tile_inflation": float(tiles.sum() * bm / counts.sum()) if len(nz) else 0.0,
    }


def _interleave(
    fns: list[tuple[str, Callable[[], mx.array]]],
    rounds: int,
    warmup: int,
) -> dict[str, list[float]]:
    """1 プロセス内で順方向 -> 逆方向に回して測る (ABC CBA を rounds 回)。"""

    for _ in range(warmup):
        for _, fn in fns:
            mx.eval(fn())

    out: dict[str, list[float]] = {name: [] for name, _ in fns}
    for _ in range(rounds):
        for name, fn in fns:
            out[name].append(_timed(fn))
        for name, fn in reversed(fns):
            out[name].append(_timed(fn))
    return out


def _first_diff_seg(ref: mx.array, got: mx.array, row_start: np.ndarray) -> dict:
    """最初に食い違う要素の (タイル, 行, 列) と値。"""

    ne = ref != got
    idx = int(mx.argmax(mx.flatten(ne).astype(mx.int32)).item())
    row, col = divmod(idx, ref.shape[1])
    e = int(np.searchsorted(row_start, row, side="right") - 1)
    return {
        "n_diff": int(mx.sum(ne.astype(mx.int32)).item()),
        "row": row,
        "col": col,
        "expert": e,
        "row_in_expert": row - int(row_start[e]),
        "tile_in_expert": (row - int(row_start[e])) // BM,
        "rows_of_expert": int(row_start[e + 1] - row_start[e]),
        "ref": float(ref[row, col].item()),
        "got": float(got[row, col].item()),
    }


def run_segmented(rounds: int, warmup: int, seed: int) -> list[dict]:
    E = NUM_EXPERTS
    cases: list[tuple[str, np.ndarray]] = [
        # 2048 tok の実測の形 (layer 1 @point 0: median 20 / p90 90.9 /
        # max 820 / zero 67、平均 40)
        ("skew-r40", skew_counts(E, 20480, 67, 20.0, 90.0)),
        # 8192 tok は「2048 tok の 4 倍」(IDEAS の P3 の想定)
        ("skew-r160", skew_counts(E, 81920, 67, 80.0, 360.0)),
        # 端数タイルの費用 c を出すための対照
        ("flat20", np.full(E, 20, dtype=np.int64)),
        ("flat32", np.full(E, 32, dtype=np.int64)),
        ("flat160", np.full(E, 160, dtype=np.int64)),
    ]

    rng = np.random.default_rng(seed)
    prepared = []
    for name, counts in cases:
        counts = counts.copy()
        rng.shuffle(counts)  # 行数の大小が専門家番号順に並ばないようにする
        ids = np.repeat(np.arange(E, dtype=np.uint32), counts)
        prepared.append((name, counts, ids))
        st16 = counts_stats(counts, 16)
        st32 = counts_stats(counts, 32)
        print(
            f"{name:10s} rows={st32['rows']:6d} active={st32['experts_active']:3d} "
            f"median={st32['median']:6.1f} p90={st32['p90']:7.1f} "
            f"max={st32['max']:5d}  "
            f"tiles(bm32)={st32['tiles_total']:5d} "
            f"(full {st32['tiles_full']} / 端数 {st32['tiles_partial']})  "
            f"水増し bm16={st16['tile_inflation']:.3f} bm32={st32['tile_inflation']:.3f}"
        )

    results = []
    for shape_name, K, N in SEG_SHAPES:
        mx.random.seed(seed)
        # 重みは中身を見ないので、逆量子化して有限に収まる範囲で乱数で作る
        # (E x N x K の bf16 を実体化しないで済ませるため)
        wq = _rand_u32((E, N, K * BITS // 32))
        groups = K // GROUP_SIZE
        scales = (mx.random.normal((E, N, groups)) * 0.02).astype(mx.bfloat16)
        biases = (mx.random.normal((E, N, groups)) * 0.02).astype(mx.bfloat16)
        w0 = mx.contiguous(wq[0])
        s0 = mx.contiguous(scales[0])
        b0 = mx.contiguous(biases[0])
        mx.eval(wq, scales, biases, w0, s0, b0)

        for name, counts, ids_np in prepared:
            M = int(counts.sum())
            x = mx.random.normal((M, K)).astype(mx.bfloat16)
            idx = mx.array(ids_np)
            x3 = x.reshape(M, 1, K)
            tables = segment_tables(mx.array(counts.astype(np.int32)))
            mx.eval(x, idx, *tables)

            assert segmented_eligible(x, wq, scales, biases), f"{name} の形が不適格"
            assert dense_eligible(x, w0, s0, b0)

            def stock() -> mx.array:
                return mx.gather_qmm(
                    x3,
                    wq,
                    scales,
                    biases,
                    rhs_indices=idx,
                    transpose=True,
                    group_size=GROUP_SIZE,
                    bits=BITS,
                    sorted_indices=True,
                ).reshape(M, N)

            def seg() -> mx.array:
                return qmm_segmented(x, wq, scales, biases, None, tables=tables)

            def seg_fragskip() -> mx.array:
                # 端数タイルで MMA を frag 単位に間引く版 (既定 off)。
                # 出力は seg と同じで、間引きの分岐が得か損かだけを見る
                return qmm_segmented(
                    x, wq, scales, biases, None, tables=tables, frag_skip=True
                )

            def dense() -> mx.array:
                return mx.quantized_matmul(
                    x,
                    w0,
                    s0,
                    b0,
                    transpose=True,
                    group_size=GROUP_SIZE,
                    bits=BITS,
                )

            y_ref = stock()
            y_got = seg()
            mx.eval(y_ref, y_got)
            y_ref2 = y_ref
            exact = bool(mx.array_equal(y_ref, y_got).item())
            max_diff = float(
                mx.max(
                    mx.abs(y_ref.astype(mx.float32) - y_got.astype(mx.float32))
                ).item()
            )
            st = counts_stats(counts, BM)
            entry: dict = {
                "shape": shape_name,
                "case": name,
                "M": M,
                "K": K,
                "N": N,
                "bit_exact": exact,
                "max_abs_diff": max_diff,
                "routing": st,
                "grid_tiles_max": n_tiles_max(M, E),
            }
            if not exact:
                row_start = np.concatenate([[0], np.cumsum(counts)])
                entry["first_diff"] = _first_diff_seg(y_ref, y_got, row_start)
            del y_got

            y_fs = seg_fragskip()
            mx.eval(y_fs)
            entry_fragskip_exact = bool(mx.array_equal(y_ref2, y_fs).item())
            del y_fs

            times = _interleave(
                [
                    ("stock", stock),
                    ("seg", seg),
                    ("fragskip", seg_fragskip),
                    ("dense", dense),
                ],
                rounds,
                warmup,
            )
            med = {k: statistics.median(v) for k, v in times.items()}
            entry.update(
                {
                    "stock_ms": med["stock"],
                    "seg_ms": med["seg"],
                    "seg_fragskip_ms": med["fragskip"],
                    "dense_ms": med["dense"],
                    "stock_over_dense": med["stock"] / med["dense"],
                    "seg_over_dense": med["seg"] / med["dense"],
                    "seg_fragskip_over_dense": med["fragskip"] / med["dense"],
                    "seg_over_stock": med["seg"] / med["stock"],
                    "fragskip_bit_exact": entry_fragskip_exact,
                    "samples": len(times["stock"]),
                }
            )
            results.append(entry)

            print(
                f"{shape_name:8s} {name:10s} M={M:6d}  "
                f"stock {med['stock']:8.3f}  seg {med['seg']:8.3f}  "
                f"fragskip {med['fragskip']:8.3f}  dense {med['dense']:8.3f} ms  |  "
                f"dense 比 stock {med['stock'] / med['dense']:5.3f} "
                f"seg {med['seg'] / med['dense']:5.3f} "
                f"fragskip {med['fragskip'] / med['dense']:5.3f}  "
                f"ビット一致 {'yes' if exact else 'NO'}"
                f"{'' if entry_fragskip_exact else ' (fragskip NO)'}"
            )
            if not exact:
                print(f"           最初の差: {entry['first_diff']}")

            del x, x3, idx, tables, y_ref, y_ref2
        del wq, scales, biases, w0, s0, b0

    _report_partial_cost(results)
    return results


def _report_partial_cost(results: list[dict]) -> None:
    """flat20 / flat32 から端数タイルの費用 c (フルタイル比) を出す。

    どちらもタイル数は E ちょうど (20 行も 32 行も 1 専門家 1 タイル) なので、
    総時間の比がそのまま 1 タイルあたりの費用比になる。既定 (frag 間引き
    off) では 20 行タイルも 32 行ぶんの MMA を回すので、c ~ 1 が予想値。
    flat160 は
    フルタイルだけ (1 専門家 5 タイル) なので、1 タイルあたりが flat32 と
    揃うかの対照に使う。
    """

    by = {(r["shape"], r["case"]): r for r in results}
    print()
    for shape_name, _, _ in SEG_SHAPES:
        f20 = by.get((shape_name, "flat20"))
        f32 = by.get((shape_name, "flat32"))
        f160 = by.get((shape_name, "flat160"))
        if not (f20 and f32 and f160):
            continue
        c = f20["seg_ms"] / f32["seg_ms"]
        c_fragskip = f20["seg_fragskip_ms"] / f32["seg_ms"]
        per_full_32 = f32["seg_ms"] / f32["routing"]["tiles_total"]
        per_full_160 = f160["seg_ms"] / f160["routing"]["tiles_total"]
        f20["partial_tile_cost_c"] = c
        f20["partial_tile_cost_c_fragskip"] = c_fragskip
        f20["per_tile_ms_flat32"] = per_full_32
        f20["per_tile_ms_flat160"] = per_full_160
        print(
            f"{shape_name:8s} 端数タイル (20 行) の費用 c = {c:5.3f} "
            f"(frag 間引き {c_fragskip:5.3f}) フルタイル比  "
            f"| 1 タイルあたり flat32 {per_full_32 * 1e3:6.2f} us "
            f"flat160 {per_full_160 * 1e3:6.2f} us "
            f"(比 {per_full_32 / per_full_160:5.3f})"
        )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dense", "segmented"], default="dense")
    ap.add_argument("--rounds", type=int, default=10, help="ABBA の回数 (A/B 各 20 本)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    info = (getattr(mx, "device_info", None) or mx.metal.device_info)()
    print(
        f"device: {info.get('device_name')} "
        f"arch={info.get('architecture')} "
        f"nax={is_nax_device()} "
        f"{ENV_KNOB}={'on' if enabled() else 'off'}"
    )

    if args.stage == "dense":
        results = run_dense(args.rounds, args.warmup, args.seed)
    else:
        results = run_segmented(args.rounds, args.warmup, args.seed)

    if args.json:
        payload = {
            "stage": args.stage,
            "device": {k: str(v) for k, v in info.items()},
            "nax": is_nax_device(),
            "knob_on": enabled(),
            "rounds": args.rounds,
            "results": results,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
