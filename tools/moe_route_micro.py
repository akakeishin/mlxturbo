"""prefill 幅の MoE で「行列積以外」に残っている 4 つを部品ごとに測る (P7 第 3 段)。

第 2 段で combine (1.95 ms/層) と x の gather (0.49) は回収した
(`docs/research/SESSION-2026-09-02-CATCHUP.md` の 18:35 / 18:44)。残りは
`bench/results/moe-split.json` (層 20、M=2048、8k) の

    router 0.82   x.astype(fp32) -> 4bit qmm 2560->512
    topk   0.31   argpartition + take_along_axis + softmax(precise)
    sort   0.34   argsort(idx) + idx の並べ替え (+ row_src / inv_perm / 表)
    swiglu 0.40   silu(gate) * up  (M=20480, H=640)

の 4 つ、合わせて 1.9 ms/層 = 8k prefill の約 3%。

このツールはモデルを読まない。本番と同じ形 (rows=2048、E=512、top_k=10、
K=2560、H=640、4bit/gs64) を乱数で作り、部品ごとに変種を 1 プロセス内で
交互に測る。**重みと活性は `--copies` 組を巡回する** (CLAUDE.md の
「連鎖 micro は重みを巡回させて冷やす」)。router の重みは 1 組 0.75 MB
しかないので巡回しても冷えないが、logits (4.2 MB) と SwiGLU の入力
(2 x 26 MB) は巡回で効く。

注意 (CLAUDE.md): ここの絶対値は案の優劣の目安にだけ使い、採否は in-model
A/B で決める。

使い方:

    tools/biglock.sh .venv/bin/python tools/moe_route_micro.py \\
        --stage all --json bench/results/moe-route-micro.json
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
import mlx.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlxturbo.kernels import moe_grouped_gemm as mgg  # noqa: E402
from mlxturbo.kernels import moe_route as mr  # noqa: E402

ROWS = 2048
E = 512
TOP_K = 10
K = 2560
H = 640
GS = 64
BITS = 4


def _timed(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    return (time.perf_counter() - t0) * 1e3


def _interleave(
    fns: list[tuple[str, Callable[[], object]]], rounds: int, warmup: int
) -> dict[str, list[float]]:
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


def _quant(shape: tuple[int, ...], dtype=mx.bfloat16):
    w = mx.random.normal(shape).astype(dtype)
    return mx.quantize(w, group_size=GS, bits=BITS)


def _cycle(n: int):
    i = 0

    def nxt() -> int:
        nonlocal i
        v = i
        i = (i + 1) % n
        return v

    return nxt


# ---------------------------------------------------------------- router


def run_router(rounds: int, warmup: int, copies: int, seed: int) -> dict:
    """`x.astype(fp32) -> 4bit qmm 2560->512` の代わりになる形を測る。

    現行 (fp32) は逆量子化も積和も fp32 で回る。bf16 で回すと積和は
    同じ fp32 の累算器 (MLX の qmm は float で累算) だが、**重みの
    逆量子化結果と出力が bf16 に丸まる**。top-k の並びが変わりうるので
    採否は KLD と in-model で決める。ここで見るのは時間だけ。
    """

    mx.random.seed(seed)
    xs = [mx.random.normal((ROWS, K)).astype(mx.bfloat16) for _ in range(copies)]
    wq = [_quant((E, K)) for _ in range(copies)]
    wb = [
        (w, s.astype(mx.bfloat16), b.astype(mx.bfloat16)) for (w, s, b) in wq
    ]
    mx.eval(xs, [t for t in wq for t in t], [t for t in wb for t in t])

    nx = _cycle(copies)

    def fp32_qmm():
        i = nx()
        w, s, b = wq[i]
        return mx.quantized_matmul(
            xs[i].astype(mx.float32), w, s.astype(mx.float32),
            b.astype(mx.float32), transpose=True, group_size=GS, bits=BITS)

    def bf16_qmm():
        i = nx()
        w, s, b = wb[i]
        return mx.quantized_matmul(
            xs[i], w, s, b, transpose=True, group_size=GS, bits=BITS)

    def bf16_qmm_f32():
        i = nx()
        w, s, b = wb[i]
        return mx.quantized_matmul(
            xs[i], w, s, b, transpose=True, group_size=GS, bits=BITS
        ).astype(mx.float32)

    def f16_qmm_f32():
        i = nx()
        w, s, b = wq[i]
        return mx.quantized_matmul(
            xs[i].astype(mx.float16), w, s.astype(mx.float16),
            b.astype(mx.float16), transpose=True, group_size=GS, bits=BITS
        ).astype(mx.float32)

    def cast_only():
        return xs[nx()].astype(mx.float32)

    # x を先に fp32 にしておいて行列積だけを測る (cast と行列積の切り分け)。
    # 本番は cast も毎層払うので「現行 = fp32_qmm」だが、cast を消せたら
    # どこまで下がるかの下限がこれ
    xs32 = [v.astype(mx.float32) for v in xs]
    mx.eval(xs32)

    def fp32_qmm_precast():
        i = nx()
        w, s, b = wq[i]
        return mx.quantized_matmul(
            xs32[i], w, s.astype(mx.float32), b.astype(mx.float32),
            transpose=True, group_size=GS, bits=BITS)

    def fp32_qmm_bf16sb():
        # 本番の形: x は fp32、scales/biases は bf16 のまま
        i = nx()
        w, s, b = wb[i]
        return mx.quantized_matmul(
            xs[i].astype(mx.float32), w, s, b, transpose=True,
            group_size=GS, bits=BITS)

    fns = [
        ("fp32_qmm", fp32_qmm),
        ("fp32_qmm_bf16sb", fp32_qmm_bf16sb),
        ("fp32_qmm_precast", fp32_qmm_precast),
        ("bf16_qmm", bf16_qmm),
        ("bf16_qmm_f32", bf16_qmm_f32),
        ("f16_qmm_f32", f16_qmm_f32),
        ("cast_only", cast_only),
    ]
    ms = _interleave(fns, rounds, warmup)

    # 数値の食い違い (同じ入力・同じ重みで 1 回だけ)
    w, s, b = wq[0]
    ref = mx.quantized_matmul(
        xs[0].astype(mx.float32), w, s.astype(mx.float32),
        b.astype(mx.float32), transpose=True, group_size=GS, bits=BITS)
    wbq = wb[0]
    got = mx.quantized_matmul(
        xs[0], wbq[0], wbq[1], wbq[2], transpose=True, group_size=GS,
        bits=BITS).astype(mx.float32)
    mx.eval(ref, got)
    rel = float(mx.mean(mx.abs(got - ref) / (mx.abs(ref) + 1e-6)).item())
    i_ref = mx.argpartition(-ref, TOP_K - 1, axis=-1)[..., :TOP_K]
    i_got = mx.argpartition(-got, TOP_K - 1, axis=-1)[..., :TOP_K]
    same = mx.sum(
        (mx.sort(i_ref, axis=-1) == mx.sort(i_got, axis=-1)).astype(mx.int32))
    return {
        "ms": {k: _med(v) for k, v in ms.items()},
        "raw": ms,
        "logit_rel_err_bf16": rel,
        "topk_slots_same": int(same.item()),
        "topk_slots_total": ROWS * TOP_K,
    }


# ------------------------------------------------------------ topk + sort


def _stock_topk(logits: mx.array):
    idx = mx.argpartition(-logits, TOP_K - 1, axis=-1)[..., :TOP_K]
    w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1,
                   precise=True)
    return idx, w


def run_topk(rounds: int, warmup: int, copies: int, seed: int) -> dict:
    mx.random.seed(seed)
    lg = [mx.random.normal((ROWS, E)) for _ in range(copies)]
    mx.eval(lg)
    nx = _cycle(copies)

    def stock():
        return _stock_topk(lg[nx()])

    def kernel():
        return mr.route(lg[nx()], TOP_K)

    fns = [("stock", stock), ("kernel", kernel)]
    ms = _interleave(fns, rounds, warmup)

    i0, w0 = _stock_topk(lg[0])
    i1, w1 = mr.route(lg[0], TOP_K)
    mx.eval(i0, w0, i1, w1)
    same = mx.sum(
        (mx.sort(i0.astype(mx.uint32), axis=-1) == mx.sort(i1, axis=-1))
        .astype(mx.int32))
    return {
        "ms": {k: _med(v) for k, v in ms.items()},
        "raw": ms,
        "topk_slots_same": int(same.item()),
        "topk_slots_total": ROWS * TOP_K,
    }


def run_sort(rounds: int, warmup: int, copies: int, seed: int) -> dict:
    """`_moe_combine_fold` の並べ替え一式 + `_moe_gemm_tables` の表作り。"""

    mx.random.seed(seed)
    lg = [mx.random.normal((ROWS, E)) for _ in range(copies)]
    idxs = []
    for l in lg:
        i, _ = _stock_topk(l)
        idxs.append(i.flatten().astype(mx.uint32))
    mx.eval(idxs)
    nx = _cycle(copies)

    def argsort_only():
        return mx.argsort(idxs[nx()])

    def stock_all():
        idx_flat = idxs[nx()]
        order = mx.argsort(idx_flat)
        idx_s = idx_flat[order]
        row_src = order // TOP_K
        return idx_s, row_src, order

    def inv_perm():
        order = mx.argsort(idxs[nx()])
        n = order.shape[0]
        return mx.zeros((n,), dtype=mx.uint32).at[order].add(
            mx.arange(n, dtype=mx.uint32))

    def tables():
        idx_flat = idxs[nx()]
        order = mx.argsort(idx_flat)
        idx_s = idx_flat[order]
        counts = mgg.counts_from_sorted_ids(idx_s, E)
        return mgg.segment_tables(counts, bm=32, mix_threshold=48)

    def counts_only():
        return mgg.counts_from_sorted_ids(idxs[nx()], E)

    fns = [
        ("argsort_only", argsort_only),
        ("stock_all", stock_all),
        ("inv_perm", inv_perm),
        ("tables", tables),
        ("counts_only", counts_only),
    ]
    ms = _interleave(fns, rounds, warmup)
    return {"ms": {k: _med(v) for k, v in ms.items()}, "raw": ms}


# ---------------------------------------------------------------- swiglu


def run_swiglu(rounds: int, warmup: int, copies: int, seed: int) -> dict:
    mx.random.seed(seed)
    M = ROWS * TOP_K
    gs = [mx.random.normal((M, H)).astype(mx.bfloat16) for _ in range(copies)]
    us = [mx.random.normal((M, H)).astype(mx.bfloat16) for _ in range(copies)]
    mx.eval(gs, us)
    nx = _cycle(copies)

    def stock():
        i = nx()
        return nn.silu(gs[i]) * us[i]

    def compiled():
        i = nx()
        return _compiled_swiglu(gs[i], us[i])

    _compiled_swiglu = mx.compile(lambda g, u: nn.silu(g) * u)

    fns = [("stock", stock), ("compiled", compiled)]
    ms = _interleave(fns, rounds, warmup)
    return {"ms": {k: _med(v) for k, v in ms.items()}, "raw": ms}


# ------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "router", "topk", "sort", "swiglu"])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--copies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    out: dict = {
        "shape": {"rows": ROWS, "experts": E, "top_k": TOP_K, "K": K, "H": H},
        "copies": args.copies,
        "rounds": args.rounds,
    }
    stages = (["router", "topk", "sort", "swiglu"] if args.stage == "all"
              else [args.stage])
    fn = {"router": run_router, "topk": run_topk, "sort": run_sort,
          "swiglu": run_swiglu}
    for st in stages:
        out[st] = fn[st](args.rounds, args.warmup, args.copies, args.seed)
        print(f"== {st} ==")
        for k, v in out[st]["ms"].items():
            print(f"  {k:16s} {v:8.3f} ms")
        for k, v in out[st].items():
            if k not in ("ms", "raw"):
                print(f"  {k}: {v}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
