"""MoE の行の並べ替え (argsort 一式 vs 計数ソート) を測る micro。モデル不要。

置き換える相手は `_moe_combine_fold` (`mlxturbo/_vendor/qwen4_exp.py`) の

    order   = mx.argsort(idx_flat)          # (M,) = 20480、値は 0..511
    idx_s   = idx_flat[order]
    row_src = order // top_k
    inv     = zeros(M).at[order].add(arange(M))     # fused._inv_perm

の 4 本 (+ `_moe_gemm_tables` の `counts_from_sorted_ids`)。計数ソート
(`mlxturbo/kernels/moe_counting_sort.py`) はこれを 2 カーネル + cumsum で作る。

入力は 80 KB しかないので冷やす必要は無いが、**同じ配列の使い回しにならない
ように 48 層ぶん (`--copies`) の別入力を巡回する** (CLAUDE.md)。

等価性は「専門家ごとの行の集合」と「専門家の境界 (row_start)」で見る。
専門家内の並びは一致しなくてよい (行ごとの GEMM は位置に依らず、combine は
k の固定順で足すので、モデルの出力はビット一致する)。

使い方:

    tools/biglock.sh .venv/bin/python tools/moe_sort_micro.py \\
        --json bench/results/moe-sort-micro.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlxturbo.kernels import moe_counting_sort as cs  # noqa: E402
from mlxturbo.kernels import moe_grouped_gemm as mgg  # noqa: E402

ROWS = 2048
E = 512
TOP_K = 10
BM = 32
MIX = 48


def _timed(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    return (time.perf_counter() - t0) * 1e3


def _interleave(fns, rounds: int, warmup: int) -> dict[str, list[float]]:
    """A B C ... C B A の順で交互に回す (熱とキャッシュを揃える)。"""

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


def _med(xs: list[float]) -> float:
    return statistics.median(xs)


def _cycle(n: int):
    i = 0

    def nxt() -> int:
        nonlocal i
        v = i
        i = (i + 1) % n
        return v

    return nxt


def _make_idx(rows: int, top_k: int, experts: int) -> mx.array:
    """本番と同じ作り方 (router の logits -> argpartition) で添字を作る。"""

    lg = mx.random.normal((rows, experts))
    idx = mx.argpartition(-lg, top_k - 1, axis=-1)[..., :top_k]
    return idx.flatten().astype(mx.uint32)


# ------------------------------------------------------------------- 等価性


def _check(idx_flat: mx.array, top_k: int, experts: int) -> dict:
    """argsort 版と計数版で「専門家ごとの集合」と境界が一致するか。"""

    order = mx.argsort(idx_flat)
    idx_s = idx_flat[order]
    row_src = order // top_k
    n = order.shape[0]
    inv = mx.zeros((n,), dtype=mx.uint32).at[order].add(
        mx.arange(n, dtype=mx.uint32))
    counts = mgg.counts_from_sorted_ids(idx_s, experts)
    ref_tab = mgg.segment_tables(counts, bm=BM, mix_threshold=MIX)

    k_order, k_idx_s, k_row_src, k_inv, k_tab = cs.sort_rows(
        idx_flat, experts, top_k, bm=BM, mix_threshold=MIX)
    mx.eval(order, idx_s, row_src, inv, ref_tab,
            k_order, k_idx_s, k_row_src, k_inv, k_tab)

    # 1. ソート後の専門家添字は完全一致 (専門家内の並びに依らない)
    idx_same = bool(mx.all(idx_s == k_idx_s).item())
    # 2. 境界表 (row_start / tile_prefix) が `segment_tables` と一致
    rs_same = bool(mx.all(ref_tab[0] == k_tab[0]).item())
    tp_same = bool(mx.all(ref_tab[1] == k_tab[1]).item())
    # 3. order は置換になっている (集合として 0..M-1)
    perm_ok = bool(mx.all(mx.sort(k_order) ==
                          mx.arange(n, dtype=mx.uint32)).item())
    # 4. `k_order` が指す先の専門家が `k_idx_s` と合っている。1 と 3 と
    #    合わせると「専門家ごとの行の集合が argsort 版と一致」が言える
    #    (同じ大きさの部分集合どうしなので)
    set_same = bool(mx.all(idx_flat[k_order] == k_idx_s).item())
    # 5. row_src / inv が order と整合
    src_same = bool(mx.all(k_row_src == k_order // top_k).item())
    inv_ok = bool(mx.all(
        mx.zeros((n,), dtype=mx.uint32).at[k_order].add(
            mx.arange(n, dtype=mx.uint32)) == k_inv).item())
    # 6. 計数版の並びは走行ごとに同じか (決定性)
    k2 = cs.sort_rows(idx_flat, experts, top_k, bm=BM, mix_threshold=MIX)[0]
    mx.eval(k2)
    deterministic = bool(mx.all(k2 == k_order).item())
    return {
        "idx_s_same": idx_same,
        "row_start_same": rs_same,
        "tile_prefix_same": tp_same,
        "order_is_perm": perm_ok,
        "expert_sets_same": set_same,
        "row_src_ok": src_same,
        "inv_ok": inv_ok,
        "deterministic": deterministic,
        "intra_expert_order_same": bool(mx.all(order == k_order).item()),
    }


# --------------------------------------------------------------------- 計測


def run(rounds: int, warmup: int, copies: int, seed: int, rows: int,
        layers: int) -> dict:
    """1 回の eval に `layers` 層ぶんを積んで測る (us/層)。

    op 1 本を単独で eval すると、投入と同期の往復 (この機体で ~165 us、
    `bench/results/moe-route-micro.json` の `counts_only` がその床) が
    そのまま乗る。本番は 48 層が 1 本のコマンドバッファに並ぶので、
    層ぶんを積んでから 1 回だけ eval したほうが実費に近い。
    """

    mx.random.seed(seed)
    idxs = [_make_idx(rows, TOP_K, E) for _ in range(copies)]
    mx.eval(idxs)
    nx = _cycle(copies)
    n = rows * TOP_K

    def argsort_only():
        return mx.argsort(idxs[nx()])

    def stock_sort():
        """`_moe_combine_fold` の 3 本 (order / idx_s / row_src)。"""
        idx_flat = idxs[nx()]
        order = mx.argsort(idx_flat)
        return order, idx_flat[order], order // TOP_K

    def stock_full():
        """3 本 + `fused._inv_perm` + `counts_from_sorted_ids` の表。"""
        idx_flat = idxs[nx()]
        order = mx.argsort(idx_flat)
        idx_s = idx_flat[order]
        row_src = order // TOP_K
        inv = mx.zeros((n,), dtype=mx.uint32).at[order].add(
            mx.arange(n, dtype=mx.uint32))
        tables = mgg.segment_tables(
            mgg.counts_from_sorted_ids(idx_s, E), bm=BM, mix_threshold=MIX)
        return order, idx_s, row_src, inv, tables

    def counting():
        """計数ソートの 4 本 (order / idx_s / row_src / inv)。表は捨てる。"""
        return cs.sort_rows(idxs[nx()], E, TOP_K, bm=BM,
                            mix_threshold=MIX)[:4]

    def counting_full():
        """4 本 + 表 (表も計数ソートのカーネルが作る)。"""
        return cs.sort_rows(idxs[nx()], E, TOP_K, bm=BM, mix_threshold=MIX)

    ones = [
        ("argsort_only", argsort_only),
        ("stock_sort", stock_sort),
        ("stock_full", stock_full),
        ("counting", counting),
        ("counting_full", counting_full),
    ]
    fns = [(name, (lambda f=fn: [f() for _ in range(layers)]))
           for name, fn in ones]
    ms = _interleave(fns, rounds, warmup)
    solo = _interleave(ones, max(2, rounds // 3), 1)
    return {
        "us_per_layer": {k: _med(v) * 1e3 / layers for k, v in ms.items()},
        "us_per_solo_call": {k: _med(v) * 1e3 for k, v in solo.items()},
        "layers": layers,
        "raw": ms,
        "raw_solo": solo,
        "equiv": _check(idxs[0], TOP_K, E),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--copies", type=int, default=48)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    out = {
        "shape": {"rows": args.rows, "experts": E, "top_k": TOP_K,
                  "M": args.rows * TOP_K},
        "copies": args.copies,
        "rounds": args.rounds,
        "block": {"tg": cs.TG, "cpt": cs.CPT, "block": cs.BLOCK},
    }
    out["sort"] = run(args.rounds, args.warmup, args.copies, args.seed,
                      args.rows, args.layers)
    print(f"== moe sort micro (rows={args.rows}, M={args.rows * TOP_K}, "
          f"{args.layers} 層/eval) ==")
    solo = out["sort"]["us_per_solo_call"]
    for k, v in out["sort"]["us_per_layer"].items():
        print(f"  {k:16s} {v:8.1f} us/層   (単発 {solo[k]:7.1f} us)")
    print("  equiv:")
    for k, v in out["sort"]["equiv"].items():
        print(f"    {k:24s} {v}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
