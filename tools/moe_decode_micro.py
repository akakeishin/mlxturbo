"""decode / verify 幅 (S=1..8) の MoE を「冷えた重み」で測る連鎖 micro。

CLAUDE.md の計測の作法どおり、重み 1 組を繰り返し読む温の連鎖では
「並列度の足りない自前カーネルが DRAM レイテンシを隠せない負け」が見えない。
ここでは E=512 の専門家を本番と同じ形 (2560 -> 640 / 640 -> 2560、Q4 g64) で
`--sets` 組ぶん確保し (1 組 1.42 GB)、呼び出しごとに乱数の添字で引くので、
同じ専門家が続けて当たる確率は 10/512 ≈ 2% しかない = 常に冷えた読み。

測るのは MoE 1 層ぶん (gate + up + SwiGLU + down + ルータ重み和) の us:

  - `base`  : 本番と同じ並び (`_gather_sort` -> gather_qmm x3 -> unsort -> 和)
  - `fused` : 専門家ごとに 1 回だけ重みを読む自前カーネル (union 読み)

添字は本番の重複率に合わせて作る (実測: S=3 の top-10 union は 21〜22.5/30)。
`--share` が 1 行あたり「これまでの union から借りる専門家の数」で、既定 4 は
S=3 で union 22/30 になる。

使い方:
  tools/biglock.sh .venv/bin/python tools/moe_decode_micro.py --mode base
  tools/biglock.sh .venv/bin/python tools/moe_decode_micro.py --mode both --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
from mlx_lm.models import switch_layers as SL  # noqa: E402

GROUP_SIZE = 64
BITS = 4


# ---------------------------------------------------------------- 重みを作る
def _rand_bits(shape) -> mx.array:
    """4bit を詰めた uint32 の乱数。値そのものは意味を持たない (両経路で同じ
    バッファを使うので比較は成立する)。実重みを quantize すると 3 GB の
    一時 fp32 が要るので、詰めた形を直接作る。"""
    hi = mx.random.randint(0, 1 << 16, shape=shape, dtype=mx.uint32)
    lo = mx.random.randint(0, 1 << 16, shape=shape, dtype=mx.uint32)
    return (hi * 65536) + lo


def make_expert_set(E: int, out_dim: int, in_dim: int):
    w = _rand_bits((E, out_dim, in_dim // 8))
    ngroups = in_dim // GROUP_SIZE
    s = (mx.random.uniform(shape=(E, out_dim, ngroups)) * 0.015 + 0.005).astype(
        mx.bfloat16
    )
    b = (-7.5 * s).astype(mx.bfloat16)
    return w, s, b


def make_sets(n_sets: int, E: int, hidden: int, inter: int):
    sets = []
    for _ in range(n_sets):
        gate = make_expert_set(E, inter, hidden)
        up = make_expert_set(E, inter, hidden)
        down = make_expert_set(E, hidden, inter)
        sets.append((gate, up, down))
    mx.eval([a for s in sets for m in s for a in m])
    return sets


# ------------------------------------------------------------------ 添字作り
def make_indices(S: int, topk: int, E: int, share: int, seed: int) -> mx.array:
    """行ごとに topk 個の相異なる専門家。行 i>0 は「これまでの union」から
    `share` 個を借り、残りを新規に引く (本番の重複率を再現するため)。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    union: list[int] = []
    rows = []
    for i in range(S):
        borrowed = []
        if i > 0 and share > 0 and union:
            k = min(share, len(union), topk)
            borrowed = list(rng.choice(union, size=k, replace=False))
        need = topk - len(borrowed)
        pool = [e for e in range(E) if e not in set(borrowed)]
        fresh = list(rng.choice(pool, size=need, replace=False))
        row = borrowed + fresh
        rng.shuffle(row)
        rows.append(row)
        for e in row:
            if e not in union:
                union.append(int(e))
    return mx.array(np.asarray(rows, dtype=np.uint32)), len(union)


# --------------------------------------------------------------- 素の並び
def moe_base(x, idx, rw, gate, up, down, sort_min: int):
    """本番 (`mlxturbo.fused.gather_sort`) と同じ並び。x は (S, K) bf16、
    idx は (S, topk) uint32、rw は (S, topk) bf16 のルータ重み。"""
    S, K = x.shape
    topk = idx.shape[-1]
    xx = mx.expand_dims(x.reshape(1, S, K), (-2, -3))     # (1,S,1,1,K)
    do_sort = idx.size >= sort_min
    ii = idx.reshape(1, S, topk)
    inv_order = None
    if do_sort:
        xx, ii, inv_order = SL._gather_sort(xx, ii)
    kw = dict(transpose=True, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    x_up = mx.gather_qmm(xx, *up, rhs_indices=ii, sorted_indices=do_sort, **kw)
    x_gate = mx.gather_qmm(xx, *gate, rhs_indices=ii, sorted_indices=do_sort, **kw)
    h = mx.sigmoid(x_gate) * x_gate * x_up
    out = mx.gather_qmm(h, *down, rhs_indices=ii, sorted_indices=do_sort, **kw)
    if do_sort:
        out = SL._scatter_unsort(out, inv_order, (1, S, topk))
    out = out.squeeze(-2)                                  # (1,S,topk,K)
    return (out * rw.reshape(1, S, topk, 1)).sum(-2).reshape(S, K)


# --------------------------------------------------------------- 自前の並び
def moe_split(x, idx, rw, gate, up, down, rmax: int = 0):
    """gate と up を別カーネルにした分割版 (threadgroup 数を素と同じに保つ)。"""
    from mlxturbo.kernels import moe_decode_fused as mdf

    S, K = x.shape
    topk = idx.shape[-1]
    H = int(gate[1].shape[-2]) if isinstance(gate, (tuple, list)) \
        else int(gate.scales.shape[-2])
    idx_flat = idx.reshape(-1)
    g = mdf.single(x, idx_flat, gate, topk, H, rmax)
    u = mdf.single(x, idx_flat, up, topk, H, rmax)
    h = (mx.sigmoid(g) * g * u).astype(mx.bfloat16)
    y = mdf.single(h, idx_flat, down, 1, K, rmax)
    return (y.reshape(S, topk, K) * rw.reshape(S, topk, 1)).sum(-2)


def moe_fused(x, idx, rw, gate, up, down, rmax: int = 0):
    from mlxturbo.kernels import moe_decode_fused as mdf

    S, K = x.shape
    topk = idx.shape[-1]
    idx_flat = idx.reshape(-1)
    h = mdf.gate_up(x, idx_flat, gate, up, topk, rmax)     # (S*topk, inter)
    y = mdf.down(h, idx_flat, down, topk, K, rmax)         # (S*topk, K)
    return (y.reshape(S, topk, K) * rw.reshape(S, topk, 1)).sum(-2)


# ------------------------------------------------------------------ 計測
def chain(fn, x0, layers):
    """N 層ぶんを 1 グラフに繋いで 1 回だけ eval する (連鎖 micro)。

    層をまたいで tanh を挟むのは (a) 依存を作って本番と同じ直列にするため、
    (b) 乱数重みで値が発散しないため。1 dispatch/層 は本番の残差 + norm 相当。"""
    x = x0
    for (idx, rw, g, u, d) in layers:
        y = fn(x, idx, rw, g, u, d)
        x = mx.tanh(y).astype(mx.bfloat16)
    return x


def bench_chain(fn, x0, layers, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        mx.eval(chain(fn, x0, layers))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(chain(fn, x0, layers))
    mx.synchronize()
    return (time.perf_counter() - t0) / iters / len(layers) * 1e6   # us/層


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, default=2)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--inter", type=int, default=640)
    ap.add_argument("--s", type=str, default="1,3,6")
    ap.add_argument("--share", type=str, default="4")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=2, help="ABBA の往復数")
    ap.add_argument("--mode", choices=["base", "fused", "both"], default="base")
    ap.add_argument("--variants", type=str, default="",
                    help="例 base,fused:1,fused:2,fused:4 (--mode より優先)")
    ap.add_argument("--sort-min", type=int, default=16)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    mx.random.seed(0)
    print(f"重み {a.sets} 組 x E={a.experts} ({a.hidden}->{a.inter}) を作る...",
          flush=True)
    t0 = time.perf_counter()
    sets = make_sets(a.sets, a.experts, a.hidden, a.inter)
    bytes_per_set = 3 * (a.experts * a.inter * a.hidden / 2
                         + a.experts * a.inter * (a.hidden // GROUP_SIZE) * 4)
    print(f"  {time.perf_counter()-t0:.1f}s、{a.sets*bytes_per_set/1e9:.2f} GB、"
          f"1 パスの読み {a.layers*10*a.inter*a.hidden/2*3/1e9:.2f} GB", flush=True)

    base_fn = (lambda x, i, r, g, u, d, sm=a.sort_min:
               moe_base(x, i, r, g, u, d, sm))
    fns = {}
    if a.variants:
        for v in a.variants.split(","):
            if v == "base":
                fns["base"] = base_fn
            elif v.startswith("fused") or v.startswith("split"):
                r = int(v.split(":")[1]) if ":" in v else 0
                f = moe_split if v.startswith("split") else moe_fused
                fns[v] = (lambda x, i, rw, g, u, d, _r=r, _f=f:
                          _f(x, i, rw, g, u, d, _r))
            else:
                raise SystemExit(f"未知の variant: {v}")
    else:
        if a.mode in ("base", "both"):
            fns["base"] = base_fn
        if a.mode in ("fused", "both"):
            fns["fused"] = moe_fused

    results = []
    for share in [int(v) for v in a.share.split(",")]:
        for S in [int(v) for v in a.s.split(",")]:
            layers = []
            unions = []
            for li in range(a.layers):
                idx, u = make_indices(S, a.topk, a.experts, share,
                                      seed=100000 * S + 1000 * share + li)
                unions.append(u)
                rw = mx.softmax(mx.random.normal((S, a.topk)),
                                axis=-1).astype(mx.bfloat16)
                g, up, dn = sets[li % a.sets]
                layers.append((idx, rw, g, up, dn))
            x0 = (mx.random.normal((S, a.hidden)) * 0.1).astype(mx.bfloat16)
            mx.eval([x0] + [t for l in layers for t in l[:2]])
            uavg = sum(unions) / len(unions)

            row = {"S": S, "share": share, "union": uavg, "pairs": S * a.topk}
            order = list(fns) + list(reversed(list(fns)))     # ABBA
            acc = {k: [] for k in fns}
            for _ in range(a.reps):
                for k in order:
                    acc[k].append(bench_chain(fns[k], x0, layers,
                                              a.iters, a.warmup))
            for k, v in acc.items():
                row[k] = sum(v) / len(v)
                row[k + "_all"] = [round(t, 1) for t in v]
            if "base" in row:
                for k in list(row):
                    if (k.startswith("fused") or k.startswith("split")) \
                            and not k.endswith("_all"):
                        row["ratio_" + k] = row[k] / row["base"]
            results.append(row)
            msg = f"share={share} S={S:<2} union {uavg:5.1f}/{S*a.topk:<2}"
            for k in fns:
                msg += f"  {k} {row[k]:7.1f}"
                if "ratio_" + k in row:
                    msg += f" ({row['ratio_' + k]:.3f})"
            print(msg, flush=True)

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"args": vars(a), "results": results}, indent=1, ensure_ascii=False))
        print("書いた:", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
