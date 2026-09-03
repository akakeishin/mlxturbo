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
  - `flat16` / `flat20` / `flat32` / `flat160`: 全専門家が同じ行数。
    **タイル 1 枚あたりの費用を形ごとに直接出すため**にある。
    flat20 と flat32 はどちらもタイル数が E ちょうどで片方だけが端数なので
    c = t(flat20) / t(flat32)。flat32 は 16 行タイルだと**同じ行を 2 枚で
    覆う**ので、t(flat32, bm16) / 2 / t(flat32, bm32) が 16 行タイル 1 枚の
    相対費用。flat160 はフルタイルだけの対照

比べるケース (全部 1 プロセス内の interleave、全部ビット一致を確認する):

  - `stock`    : `mx.gather_qmm(sorted_indices=True)` (既製、BM=16 で straddle あり)
  - `seg32`    : 自前カーネル、32 行タイル / WM=2 (128 スレッド、既定)
  - `seg16`    : 自前カーネル、16 行タイル / WM=1 (64 スレッド)
  - `seg32w1`  : 自前カーネル、32 行タイル / WM=1 (64 スレッド)。混合が
                 32 行タイルを 64 スレッドで回す分の損を切り分ける対照
  - `mix<t>`   : 専門家ごとに、行数 < t なら 16 行タイル、それ以外 32 行タイル
                 (1 dispatch。threadgroup を揃えるため WM は必ず 1)
  - `dense`    : 同じ総行数の `mx.quantized_matmul` (床)
  - `pad16`    : 既製 `gather_qmm` に、各専門家のセグメントを 16 行の倍数へ
                 ダミー行で切り上げて流したもの (`gather_qmm_pad_micro.py` と
                 同じ作り方)。**この変種の判定線**

## 第 3 段の付帯費用 (`--stage tables`)

行列積そのものではなく、**in-model で行列積の外側に乗るもの**を測る。

  - `seg`: `counts_from_sorted_ids` -> `segment_tables` (scatter-add 1 +
    cumsum 2 + concat 2)。host 同期なし、MoE 層 1 つにつき 1 回
  - `pad16`: 16 行揃えの `(src, idx_pad, keep)`。水増し後の行数を Python の
    int にする **host 同期が 1 回**入る (MLX の配列は形が静的なので、
    これを避けるには静的上限で張るしかなく、そうすると r=40 で行数 +40%
    になって勝ちが消える)

## 使い方 (GPU。必ず lock 経由で)

    tools/biglock.sh .venv/bin/python tools/moe_grouped_gemm_micro.py \
        --stage dense --json bench/results/moe-grouped-gemm-dense.json
    tools/biglock.sh .venv/bin/python tools/moe_grouped_gemm_micro.py \
        --stage segmented --json bench/results/moe-grouped-gemm-segmented.json
    tools/biglock.sh .venv/bin/python tools/moe_grouped_gemm_micro.py \
        --stage tables --json bench/results/moe-grouped-gemm-tables.json

BM=16 と混合を含む形 (閾値の掃引つき):

    tools/biglock.sh .venv/bin/python tools/moe_grouped_gemm_micro.py \
        --stage segmented --rounds 8 --mix-thresholds 24,32,40,48,64,96 \
        --json bench/results/moe-grouped-gemm-bm16.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from functools import partial
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlxturbo.kernels.moe_grouped_gemm import (
    BITS,
    BM,
    ENV_KNOB,
    GROUP_SIZE,
    counts_from_sorted_ids,
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


def pad16_layout(counts: np.ndarray, multiple: int = 16):
    """16 行揃えのダミー行を足した layout を作る (`gather_qmm_pad_micro.py` と同じ)。

    返るのは ``(src, keep, ids_pad)``:

      - ``src``  : 水増し後の各行が元の x の何行目を読むか (ダミーは 0)
      - ``keep`` : 水増し後のどの行が本物か (bool)
      - ``ids_pad``: 水増し後の専門家添字

    行数が 0 の専門家は 0 のまま (切り上げない)。
    """

    padded = np.where(counts > 0, ((counts + multiple - 1) // multiple) * multiple, 0)
    m_pad = int(padded.sum())
    src = np.zeros(m_pad, dtype=np.int32)
    keep = np.zeros(m_pad, dtype=bool)
    ids_pad = np.zeros(m_pad, dtype=np.uint32)
    o_pad = 0
    o_src = 0
    for e, (c, pc) in enumerate(zip(counts, padded)):
        c = int(c)
        pc = int(pc)
        if pc == 0:
            continue
        ids_pad[o_pad : o_pad + pc] = e
        src[o_pad : o_pad + c] = np.arange(o_src, o_src + c, dtype=np.int32)
        keep[o_pad : o_pad + c] = True
        o_pad += pc
        o_src += c
    return src, keep, ids_pad


def mixed_stats(counts: np.ndarray, thresh: int) -> dict:
    """混合モード (行数 < thresh なら 16 行タイル、それ以外 32 行タイル) の枚数。"""

    bm_e = np.where(counts < thresh, 16, 32)
    tiles = np.ceil(counts / np.maximum(bm_e, 1)).astype(np.int64)
    tiles = np.where(counts > 0, tiles, 0)
    rows_cov = int((tiles * bm_e).sum())
    return {
        "threshold": int(thresh),
        "tiles_total": int(tiles.sum()),
        "tiles_bm16": int(tiles[bm_e == 16].sum()),
        "tiles_bm32": int(tiles[bm_e == 32].sum()),
        "experts_bm16": int(((bm_e == 16) & (counts > 0)).sum()),
        "row_inflation": rows_cov / float(counts.sum()),
    }


def run_segmented(
    rounds: int,
    warmup: int,
    seed: int,
    mix_thresholds: list[int],
    with_fragskip: bool = False,
    bns: list[int] | None = None,
    bks: list[int] | None = None,
) -> list[dict]:
    E = NUM_EXPERTS
    cases: list[tuple[str, np.ndarray]] = [
        # 2048 tok の実測の形 (layer 1 @point 0: median 20 / p90 90.9 /
        # max 820 / zero 67、平均 40)
        ("skew-r40", skew_counts(E, 20480, 67, 20.0, 90.0)),
        # 8192 tok は「2048 tok の 4 倍」(IDEAS の P3 の想定)
        ("skew-r160", skew_counts(E, 81920, 67, 80.0, 360.0)),
        # 端数タイルの費用 c と 16 行タイル 1 枚の費用を出すための対照。
        # flat16 は bm16 でちょうど満タン、bm32 で全部端数。
        # flat32 は bm32 で満タン 1 枚 = bm16 で満タン 2 枚 (同じ行を覆う)
        ("flat16", np.full(E, 16, dtype=np.int64)),
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
        # 掃引は skew だけ。flat は先頭の閾値 1 つで足りる
        th = list(mix_thresholds) if name.startswith("skew") else mix_thresholds[:1]
        prepared.append((name, counts, ids, th))
        st16 = counts_stats(counts, 16)
        st32 = counts_stats(counts, 32)
        print(
            f"{name:10s} rows={st32['rows']:6d} active={st32['experts_active']:3d} "
            f"median={st32['median']:6.1f} p90={st32['p90']:7.1f} "
            f"max={st32['max']:5d}  "
            f"tiles bm16={st16['tiles_total']:5d} bm32={st32['tiles_total']:5d}  "
            f"水増し bm16={st16['tile_inflation']:.3f} "
            f"bm32={st32['tile_inflation']:.3f} "
            + " ".join(
                f"mix{t}={mixed_stats(counts, t)['row_inflation']:.3f}" for t in th
            )
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

        for name, counts, ids_np, thresholds in prepared:
            M = int(counts.sum())
            x = mx.random.normal((M, K)).astype(mx.bfloat16)
            idx = mx.array(ids_np)
            x3 = x.reshape(M, 1, K)
            counts_mx = mx.array(counts.astype(np.int32))
            tab32 = segment_tables(counts_mx, bm=32)
            tab16 = segment_tables(counts_mx, bm=16)
            tab8 = segment_tables(counts_mx, bm=8)
            tab_mix = {
                t: segment_tables(counts_mx, mix_threshold=t) for t in thresholds
            }
            mx.eval(x, idx, *tab32, *tab16, *[a for v in tab_mix.values() for a in v])

            # 16 行揃えの既製 gather_qmm (P3 の入場料チェックと同じ作り方)
            src_np, keep_np, ids_pad_np = pad16_layout(counts)
            m_pad = int(src_np.shape[0])
            keep_idx = mx.array(np.nonzero(keep_np)[0].astype(np.int32))
            x_pad = (
                x[mx.array(src_np)]
                * mx.array(keep_np.astype(np.float32)).reshape(m_pad, 1)
            ).astype(x.dtype)
            x_pad3 = x_pad.reshape(m_pad, 1, K)
            idx_pad = mx.array(ids_pad_np)
            mx.eval(x_pad3, idx_pad, keep_idx)

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

            def pad16() -> mx.array:
                return mx.gather_qmm(
                    x_pad3,
                    wq,
                    scales,
                    biases,
                    rhs_indices=idx_pad,
                    transpose=True,
                    group_size=GROUP_SIZE,
                    bits=BITS,
                    sorted_indices=True,
                ).reshape(m_pad, N)

            def seg32() -> mx.array:
                return qmm_segmented(x, wq, scales, biases, None, tables=tab32)

            def seg16() -> mx.array:
                return qmm_segmented(
                    x, wq, scales, biases, None, tables=tab16, bm=16
                )

            def seg32w1() -> mx.array:
                # 混合の対照: 32 行タイルを 64 スレッド (WM=1) で回す
                return qmm_segmented(
                    x, wq, scales, biases, None, tables=tab32, bm=32, wm=1
                )

            def make_mix(t: int):
                def mix() -> mx.array:
                    return qmm_segmented(
                        x,
                        wq,
                        scales,
                        biases,
                        None,
                        tables=tab_mix[t],
                        mix_threshold=t,
                    )

                return mix

            def seg_fragskip() -> mx.array:
                return qmm_segmented(
                    x, wq, scales, biases, None, tables=tab32, frag_skip=True
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

            variants: list[tuple[str, Callable[[], mx.array]]] = [
                ("stock", stock),
                ("seg32", seg32),
                ("seg16", seg16),
                ("seg32w1", seg32w1),
            ]
            for t in thresholds:
                variants.append((f"mix{t}", make_mix(t)))

            # 列タイル幅 (BN) の掃引 (P7 第 3 段)。BN を広げると 1 枚の
            # threadgroup が受け持つ出力が増え、タイル 1 枚あたりの固定費
            # (dispatch と x タイルの読み) が薄まる。行数の少ない専門家が
            # 多い r=40 で効くかを見る。表 (row_start/tile_prefix) は BN に
            # 依らないので使い回せる
            def make_shape(fn_name: str, bn: int = 32, bk: int = 32):
                if fn_name == "mix":
                    kw = dict(tables=tab_mix[thresholds[0]],
                              mix_threshold=thresholds[0])
                elif fn_name == "seg16":
                    kw = dict(tables=tab16, bm=16)
                elif fn_name == "seg8":
                    kw = dict(tables=tab8, bm=8)
                else:
                    kw = dict(tables=tab32, bm=32, wm=1)

                def run() -> mx.array:
                    return qmm_segmented(x, wq, scales, biases, None,
                                         bn=bn, bk=bk, **kw)

                return run

            variants.append(("seg8", make_shape("seg8")))
            for bn in bns or []:
                if N % bn != 0:
                    continue
                for base in ("mix", "seg16", "seg32w1"):
                    label = (f"mix{thresholds[0]}" if base == "mix" else base)
                    variants.append(
                        (f"{label}-bn{bn}", make_shape(base, bn=bn)))
            for bk in bks or []:
                if K % bk != 0 or GROUP_SIZE % bk != 0:
                    continue
                for base in ("mix", "seg16", "seg8", "seg32w1"):
                    label = (f"mix{thresholds[0]}" if base == "mix" else base)
                    variants.append(
                        (f"{label}-bk{bk}", make_shape(base, bk=bk)))
            if with_fragskip:
                variants.append(("fragskip", seg_fragskip))
            variants += [("dense", dense), ("pad16", pad16)]

            # ビット一致 (全ケース)。pad16 は本物の行だけ取り出して比べる
            y_ref = stock()
            mx.eval(y_ref)
            exact: dict[str, bool] = {}
            max_diff: dict[str, float] = {}
            for vname, fn in variants:
                if vname in ("stock", "dense"):
                    continue
                y = fn()
                if vname == "pad16":
                    y = y[keep_idx]
                mx.eval(y)
                exact[vname] = bool(mx.array_equal(y_ref, y).item())
                max_diff[vname] = float(
                    mx.max(
                        mx.abs(y_ref.astype(mx.float32) - y.astype(mx.float32))
                    ).item()
                )
                del y

            st = counts_stats(counts, BM)
            entry: dict = {
                "shape": shape_name,
                "case": name,
                "M": M,
                "M_pad16": m_pad,
                "K": K,
                "N": N,
                "bit_exact": exact,
                "max_abs_diff": max_diff,
                "routing": st,
                "routing_bm16": counts_stats(counts, 16),
                "mixed": {str(t): mixed_stats(counts, t) for t in thresholds},
                "grid_tiles_max": n_tiles_max(M, E),
            }
            del y_ref

            times = _interleave(variants, rounds, warmup)
            med = {k: statistics.median(v) for k, v in times.items()}
            dense_ms = med["dense"]
            entry["ms"] = med
            entry["dense_ratio"] = {k: v / dense_ms for k, v in med.items()}
            entry["samples"] = len(times["stock"])
            # 旧いキー (下流の _report_partial_cost と過去の json の互換)
            entry["stock_ms"] = med["stock"]
            entry["seg_ms"] = med["seg32"]
            entry["dense_ms"] = dense_ms
            entry["seg_over_dense"] = med["seg32"] / dense_ms
            results.append(entry)

            order = [v for v, _ in variants]
            print(
                f"{shape_name:8s} {name:10s} M={M:6d}  "
                + "  ".join(f"{v} {med[v]:7.3f}" for v in order)
                + f"  ms | dense 比 "
                + " ".join(f"{v} {med[v] / dense_ms:5.3f}" for v in order)
            )
            bad = [v for v, ok in exact.items() if not ok]
            if bad:
                print(f"           ビット不一致: {bad} (max|diff| {max_diff})")

            del x, x3, idx, tab32, tab16, tab8, tab_mix, counts_mx
            del x_pad, x_pad3, idx_pad, keep_idx
        del wq, scales, biases, w0, s0, b0

    _report_partial_cost(results)
    _report_layer_totals(results, mix_thresholds)
    return results


def _report_layer_totals(results: list[dict], mix_thresholds: list[int]) -> None:
    """MoE 層 1 つぶん (gate + up + down = gate/up x2 + down) の合計を出す。

    P3 の判定線がこの単位 (`gather_qmm_pad_micro.py` の 22.61 ms/層) なので、
    そこと直に比べられる形にしておく。
    """

    by = {(r["shape"], r["case"]): r for r in results}
    print()
    print("MoE 層 1 つぶんの合計 (gate/up x2 + down)")
    for case in ("skew-r40", "skew-r160"):
        gu = by.get(("gate/up", case))
        dn = by.get(("down", case))
        if not (gu and dn):
            continue
        names = [k for k in gu["ms"] if k in dn["ms"]]
        total = {k: 2 * gu["ms"][k] + dn["ms"][k] for k in names}
        d = total["dense"]
        print(f"  {case}:")
        for k in names:
            print(f"    {k:10s} {total[k]:8.3f} ms  dense 比 {total[k] / d:5.3f}")
        best_mix = min(
            (f"mix{t}" for t in mix_thresholds if f"mix{t}" in total),
            key=lambda k: total[k],
            default=None,
        )
        gu["layer_total_ms"] = total
        gu["layer_dense_ratio"] = {k: total[k] / d for k in names}
        gu["layer_best_mix"] = best_mix
        if best_mix:
            print(
                f"    -> 最良の混合 {best_mix} {total[best_mix]:.3f} ms "
                f"(pad16 {total['pad16']:.3f} / seg32 {total['seg32']:.3f} / "
                f"seg16 {total['seg16']:.3f})"
            )


def _report_partial_cost(results: list[dict]) -> None:
    """flat の対照から、タイル 1 枚あたりの費用を 16 行と 32 行で突き合わせる。

      - 端数タイルの費用 c (32 行タイル比) = t(flat20, bm32) / t(flat32, bm32)。
        どちらもタイル数は E ちょうどなので、総時間の比がそのまま 1 枚の費用比
      - 16 行タイル 1 枚の費用 (32 行タイル比) = t(flat32, bm16) / 2 /
        t(flat32, bm32)。flat32 は bm16 でちょうど 2 枚 / bm32 で 1 枚と、
        **同じ行を覆う**のでタイル形の差だけが出る
      - flat16 は bm16 で満タン 1 枚、bm32 で全部端数。既製の BM=16 が
        なぜ効くかの直接の対照
    """

    by = {(r["shape"], r["case"]): r for r in results}
    print()
    for shape_name, _, _ in SEG_SHAPES:
        f16 = by.get((shape_name, "flat16"))
        f20 = by.get((shape_name, "flat20"))
        f32 = by.get((shape_name, "flat32"))
        f160 = by.get((shape_name, "flat160"))
        if not (f20 and f32 and f160):
            continue
        c = f20["ms"]["seg32"] / f32["ms"]["seg32"]
        tile16 = f32["ms"]["seg16"] / 2.0 / f32["ms"]["seg32"]
        per_full_32 = f32["ms"]["seg32"] / f32["routing"]["tiles_total"]
        per_full_160 = f160["ms"]["seg32"] / f160["routing"]["tiles_total"]
        f20["partial_tile_cost_c"] = c
        f32["bm16_tile_cost_vs_bm32"] = tile16
        f20["per_tile_ms_flat32"] = per_full_32
        f20["per_tile_ms_flat160"] = per_full_160
        line = (
            f"{shape_name:8s} 端数 (20 行) タイルの費用 c = {c:5.3f}  "
            f"| 16 行タイル 1 枚 = 32 行タイルの {tile16:5.3f} 倍  "
            f"| 1 タイルあたり flat32 {per_full_32 * 1e3:6.2f} us "
            f"flat160 {per_full_160 * 1e3:6.2f} us "
            f"(比 {per_full_32 / per_full_160:5.3f})"
        )
        if f16:
            line += (
                f"  | flat16 bm16 {f16['ms']['seg16']:6.3f} "
                f"bm32 {f16['ms']['seg32']:6.3f} "
                f"stock {f16['ms']['stock']:6.3f} ms"
            )
        print(line)


# ---------------------------------------------------------------------------
# SwiGLU を up GEMM の store に畳む (--stage glu、P7 第 3 段)
# ---------------------------------------------------------------------------


@partial(mx.compile, shapeless=True)
def _swiglu_compiled(gate: mx.array, up: mx.array) -> mx.array:
    """`nn.silu(gate) * up` を 1 カーネルに畳む。

    素は `nn.silu` (それ自体 compile 済み = 1 本) と乗算の **2 本**で、
    26 MB の往復が 1 回余計に要る (M=20480, N=640, bf16)。
    """

    return (gate * mx.sigmoid(gate)) * up


def run_glu(rounds: int, warmup: int, seed: int, mix_threshold: int) -> list[dict]:
    """gate GEMM -> up GEMM -> `silu(gate) * up` を、SwiGLU の作り方だけ
    変えて比べる。

    素の `nn.silu(gate) * up` は 2 本 (`nn.silu` 自体は `mx.compile` 済みで
    1 本 + 乗算 1 本) で、26 MB (M=20480 x N=640 bf16) の往復が 1 回余計に
    要る。1 本に畳んだ `compiled` は **素とビット一致**する。

    実測 (2026-09-03、M3 Max): 素 15.130 / compiled 15.062 ms (-0.5% =
    -0.068 ms/層)。**up GEMM の store に畳む変種 (TGLU) は取り除いた** --
    -0.123 ms/層と少し速いがビット一致せず、frag の並び (8 行 x 2 要素) で
    読むので coalescing が悪く、in-model 8k では ±0.3% (揺れの中) だった
    (`docs/research/SESSION-2026-09-02-CATCHUP.md` の 2026-09-03 20:55)。
    """

    E = NUM_EXPERTS
    K, N = 2560, 640
    counts = skew_counts(E, 20480, 67, 20.0, 90.0)
    rng = np.random.default_rng(seed)
    rng.shuffle(counts)
    ids = np.repeat(np.arange(E, dtype=np.uint32), counts)
    M = int(counts.sum())

    mx.random.seed(seed)
    x = mx.random.normal((M, K)).astype(mx.bfloat16)
    wq_g = _rand_u32((E, N, K * BITS // 32))
    wq_u = _rand_u32((E, N, K * BITS // 32))
    groups = K // GROUP_SIZE
    sc_g = (mx.random.normal((E, N, groups)) * 0.02).astype(mx.bfloat16)
    bi_g = (mx.random.normal((E, N, groups)) * 0.02).astype(mx.bfloat16)
    sc_u = (mx.random.normal((E, N, groups)) * 0.02).astype(mx.bfloat16)
    bi_u = (mx.random.normal((E, N, groups)) * 0.02).astype(mx.bfloat16)
    counts_mx = mx.array(counts.astype(np.int32))
    tab = segment_tables(counts_mx, mix_threshold=mix_threshold)
    mx.eval(x, wq_g, wq_u, sc_g, bi_g, sc_u, bi_u, tab)
    del ids

    kw = dict(tables=tab, mix_threshold=mix_threshold)

    def stock() -> mx.array:
        g = qmm_segmented(x, wq_g, sc_g, bi_g, None, **kw)
        u = qmm_segmented(x, wq_u, sc_u, bi_u, None, **kw)
        return nn.silu(g) * u

    def compiled() -> mx.array:
        g = qmm_segmented(x, wq_g, sc_g, bi_g, None, **kw)
        u = qmm_segmented(x, wq_u, sc_u, bi_u, None, **kw)
        return _swiglu_compiled(g, u)

    ys = {}
    for nm, fn in (("stock", stock), ("compiled", compiled)):
        ys[nm] = fn()
    mx.eval(list(ys.values()))
    base = ys["stock"].astype(mx.float32)
    scale = float(mx.mean(mx.abs(base)).item())
    diff = {
        nm: {
            "mean_abs_vs_stock": float(
                mx.mean(mx.abs(ys[nm].astype(mx.float32) - base)).item()),
            "max_abs_vs_stock": float(
                mx.max(mx.abs(ys[nm].astype(mx.float32) - base)).item()),
            "bit_exact_vs_stock": bool(
                mx.array_equal(ys["stock"], ys[nm]).item()),
        }
        for nm in ("compiled",)
    }
    del ys, base

    times = _interleave(
        [("stock", stock), ("compiled", compiled)], rounds, warmup)
    med = {k: statistics.median(v) for k, v in times.items()}
    entry = {
        "shape": "gate/up+swiglu",
        "M": M,
        "K": K,
        "N": N,
        "mix_threshold": mix_threshold,
        "ms": med,
        "ratio": {k: v / med["stock"] for k, v in med.items()},
        "mean_abs_stock": scale,
        "vs_stock": diff,
        "samples": len(times["stock"]),
    }
    print(
        f"gate/up+swiglu M={M}  "
        + "  ".join(f"{k} {v:7.3f}" for k, v in med.items())
        + "  ms | 比 "
        + " ".join(f"{k} {v:5.3f}" for k, v in entry["ratio"].items())
    )
    print(f"  素との差 (素の平均絶対値 {scale:.4g}): "
          + " ".join(
              f"{k} mean {v['mean_abs_vs_stock']:.3e} "
              f"max {v['max_abs_vs_stock']:.3e}" for k, v in diff.items()))
    return [entry]


# ---------------------------------------------------------------------------
# 第 3 段の付帯費用 (--stage tables)
# ---------------------------------------------------------------------------

# 本番の MoE 層数 (qwen4_exp、Flash-Next)
N_LAYERS = 48


def run_tables(rounds: int, warmup: int, seed: int) -> list[dict]:
    """テーブル構築の小 op 列だけを 48 層ぶん測る。

    in-model では行列積そのものの取り分に**これが乗る**。2 種類:

      - ``seg``  : `counts_from_sorted_ids` -> `segment_tables`
                   (scatter-add 1 + cumsum 2 + concat 2)。host 同期なし。
                   MoE 層 1 つにつき 1 回 (gate/up/down で使い回す)
      - ``pad16``: 16 行揃えの `(src, idx_pad, keep)`。**host 同期が 1 回**
                   入る (水増し後の行数を Python の int にしないと配列が
                   作れない)。同じく MoE 層 1 つにつき 1 回

    ``seg`` は同期が無いので 48 本まとめて 1 回の `mx.eval` で測る
    (in-model でもレイヤー間に他の仕事が挟まるので、これが上限ではなく
    「投げ切るのに要る時間」の目安)。``pad16`` は同期が構造上入るので
    48 回逐次で測る。
    """

    E = NUM_EXPERTS
    from mlxturbo import fused

    cases = [
        ("2048tok(r=40)", skew_counts(E, 20480, 67, 20.0, 90.0)),
        ("8192tok(r=160)", skew_counts(E, 81920, 67, 80.0, 360.0)),
    ]
    rng = np.random.default_rng(seed)
    results = []
    for name, counts in cases:
        counts = counts.copy()
        rng.shuffle(counts)
        M = int(counts.sum())
        ids = mx.array(np.repeat(np.arange(E, dtype=np.uint32), counts))
        mx.eval(ids)

        def build_seg() -> float:
            t0 = time.perf_counter()
            outs = []
            for _ in range(N_LAYERS):
                rs, tp = segment_tables(counts_from_sorted_ids(ids, E))
                outs.append(rs)
                outs.append(tp)
            mx.eval(*outs)
            return (time.perf_counter() - t0) * 1e3

        def build_pad() -> float:
            t0 = time.perf_counter()
            for _ in range(N_LAYERS):
                fused._MOE_GEMM_PAD_TABLES = None  # キャッシュを外して毎回作らせる
                tabs = fused._moe_gemm_pad_tables(ids, E)
                mx.eval(*tabs)
            fused._MOE_GEMM_PAD_TABLES = None
            return (time.perf_counter() - t0) * 1e3

        for _ in range(warmup):
            build_seg()
            build_pad()
        seg_ms = []
        pad_ms = []
        for _ in range(rounds):
            seg_ms.append(build_seg())
            pad_ms.append(build_pad())
            pad_ms.append(build_pad())
            seg_ms.append(build_seg())

        seg_med = statistics.median(seg_ms)
        pad_med = statistics.median(pad_ms)
        # 水増し後の行数 (host 側で同じ式を回して確かめる)
        m_pad = int((np.ceil(counts / 16) * 16).sum())
        entry = {
            "case": name,
            "M": M,
            "M_pad16": m_pad,
            "pad_row_inflation": m_pad / M,
            "layers": N_LAYERS,
            "seg_tables_ms_48": seg_med,
            "seg_tables_us_per_layer": seg_med / N_LAYERS * 1e3,
            "pad16_tables_ms_48": pad_med,
            "pad16_tables_us_per_layer": pad_med / N_LAYERS * 1e3,
            "samples": len(seg_ms),
        }
        results.append(entry)
        print(
            f"{name:16s} M={M:6d}  seg テーブル {seg_med:7.3f} ms/48 層 "
            f"({seg_med / N_LAYERS * 1e3:6.1f} us/層)  |  "
            f"pad16 テーブル {pad_med:7.3f} ms/48 層 "
            f"({pad_med / N_LAYERS * 1e3:6.1f} us/層、host 同期込み)  |  "
            f"水増し後 {m_pad} 行 ({m_pad / M:5.3f} 倍)"
        )
        del ids
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",
                    choices=["dense", "segmented", "tables", "glu"],
                    default="dense")
    ap.add_argument("--rounds", type=int, default=10, help="ABBA の回数 (A/B 各 20 本)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--mix-thresholds",
        default="48",
        help="混合モードの閾値 (行数 < 閾値 なら 16 行タイル)。カンマ区切りで掃引",
    )
    ap.add_argument(
        "--fragskip",
        action="store_true",
        help="端数タイルの frag 間引き版も interleave に入れる (既定 off、負け済み)",
    )
    ap.add_argument(
        "--bn",
        default="",
        help="列タイル幅 BN の掃引 (カンマ区切り、既定 32 は常に入っている)。"
             "例 --bn 64",
    )
    ap.add_argument(
        "--bk",
        default="",
        help="K タイル幅 BK の掃引 (カンマ区切り、既定 32)。例 --bk 64",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    mix_thresholds = [int(v) for v in args.mix_thresholds.split(",") if v.strip()]
    bns = [int(v) for v in args.bn.split(",") if v.strip()]
    bks = [int(v) for v in args.bk.split(",") if v.strip()]

    info = (getattr(mx, "device_info", None) or mx.metal.device_info)()
    print(
        f"device: {info.get('device_name')} "
        f"arch={info.get('architecture')} "
        f"nax={is_nax_device()} "
        f"{ENV_KNOB}={'on' if enabled() else 'off'}"
    )

    if args.stage == "dense":
        results = run_dense(args.rounds, args.warmup, args.seed)
    elif args.stage == "tables":
        results = run_tables(args.rounds, args.warmup, args.seed)
    elif args.stage == "glu":
        results = run_glu(args.rounds, args.warmup, args.seed,
                          mix_thresholds[0])
    else:
        results = run_segmented(
            args.rounds,
            args.warmup,
            args.seed,
            mix_thresholds,
            with_fragskip=args.fragskip,
            bns=bns,
            bks=bks,
        )

    if args.json:
        payload = {
            "stage": args.stage,
            "device": {k: str(v) for k, v in info.items()},
            "nax": is_nax_device(),
            "knob_on": enabled(),
            "rounds": args.rounds,
            "mix_thresholds": mix_thresholds,
            "results": results,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
