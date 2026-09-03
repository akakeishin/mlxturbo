"""P5 (dense sdpa の行タイル分割) の proof-of-life。モデル無し。

`docs/research/IDEAS-2026-09-03.md` の P5: MLX 0.32.2 の sdpa は head_dim=256・
S>8 では steel ではなく fallback (matmul -> where -> softmax -> matmul、
`~/dev/mlx-serve/lib/mlx-src/mlx/fast.cpp:835-886`) に落ち、タイルを 1 つも
飛ばさない。現チャンク (S=2048) の上三角 = kv 1024 列ぶん = 5.1 ms/層/チャンクを
毎チャンク無駄に計算している。

ここでは `Attention.__call__` の dense 分岐 (`_vendor/qwen4_exp.py` 1110 行付近)
と同じ形を、モデル無しで再現する: q を R 行ずつのタイルに割り、タイル t は
K/V を `[0, offset+(t+1)R)` に、マスクを `mask[..., tR:(t+1)R, :offset+(t+1)R]`
に絞って sdpa を呼び、`concatenate(axis=2)` で戻す (`mask=="causal"` のときは
各タイルにも "causal" を渡す -- fallback は `offset = kL - qL` で対角を出すので、
タイルごとに正しい対角になる)。QSA の可視集合は `block_end <= q_col` と
tail `col <= q_col` なので、各行タイルの可視 key は `[0, offset+(t+1)R)` に
全部入る (近似無し)。

判定線 (kv=2048 の削減、S=2048 全体):
    R=512 で -3.9 ms 級 / R=256 で -4.6 ms 級 / R=128 で -4.9 ms 級。

1 プロセス内で whole (単発 sdpa) と各 R を ABAB... (reps 回) で交互に測る。
GPU が空いているときに走らせること (`tools/biglock.sh` 経由)。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import mlx.core as mx


def qsa_like_mask(S: int, kv: int, block: int = 4, keep: int = 512) -> mx.array:
    """QSA 風の bool マスク: causal & (行ごとにランダムな keep ブロックだけ可視)。

    `tools/sdpa_headdim_micro.py` の同名関数と揃えてある (block=4, keep=512
    で過去 512 ブロック x 4 = 2048 列がランダムに可視)。
    """
    offset = kv - S
    rows = mx.arange(S)[:, None] + offset
    cols = mx.arange(kv)[None, :]
    causal = cols <= rows
    n_blocks = (kv + block - 1) // block
    if n_blocks <= keep:
        return causal[None, None]
    scores = mx.random.uniform(shape=(S, n_blocks))
    top = mx.argpartition(-scores, kth=keep - 1, axis=-1)[:, :keep]
    vis = mx.zeros((S, n_blocks), dtype=mx.bool_)
    vis = mx.put_along_axis(vis, top, mx.ones((S, keep), dtype=mx.bool_), axis=-1)
    vis = mx.repeat(vis, block, axis=-1)[:, :kv]
    return (causal & vis)[None, None]


def rowtile_sdpa(q: mx.array, k: mx.array, v: mx.array, mask, scale: float, rows: int) -> mx.array:
    """`Attention.__call__` の dense 分岐を行タイルに割った形 (段 P5)。

    `mlxturbo/fused.py` の `enable_sdpa_rowtile` と同じ算法。q/k/v は
    (B, H, S/kv, D)。K/V は前方 `[0, kv_end)` だけに絞る (可視集合は不変)。
    """
    S = q.shape[2]
    kv_len = k.shape[2]
    offset = kv_len - S
    outs = []
    t0 = 0
    while t0 < S:
        t1 = min(t0 + rows, S)
        kv_end = offset + t1
        k_t = k[:, :, :kv_end, :]
        v_t = v[:, :, :kv_end, :]
        q_t = q[:, :, t0:t1, :]
        m_t = mask if isinstance(mask, str) else mask[..., t0:t1, :kv_end]
        outs.append(
            mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale, mask=m_t)
        )
        t0 = t1
    return mx.concatenate(outs, axis=2)


def bench(fn, reps: int, warm: int = 2) -> float:
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=2048)
    ap.add_argument("--kvs", default="2048,4096,8192")
    ap.add_argument("--rows", default="512,256,128")
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json", default=None, help="JSON 書き出し先 (省略なら書かない)")
    a = ap.parse_args()

    S = a.S
    Hq, Hk = a.heads, a.kv_heads
    D = 256
    scale = D ** -0.5
    row_widths = [int(r) for r in a.rows.split(",")]

    res = []
    for kv in (int(x) for x in a.kvs.split(",")):
        q = mx.random.normal((1, Hq, S, D)).astype(mx.bfloat16)
        k = mx.random.normal((1, Hk, kv, D)).astype(mx.bfloat16)
        v = mx.random.normal((1, Hk, kv, D)).astype(mx.bfloat16)
        mx.eval(q, k, v)

        if kv == S:
            mask = "causal"
        else:
            mask = qsa_like_mask(S, kv)
            mx.eval(mask)

        cases = {
            "whole": lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        }
        for r in row_widths:
            cases[f"R{r}"] = (lambda r=r: rowtile_sdpa(q, k, v, mask, scale, r))

        names = list(cases)
        samples = {n: [] for n in names}
        outs = {n: None for n in names}
        for n in names:
            outs[n] = cases[n]()
            mx.eval(outs[n])
        for r in range(a.reps):
            order = names if r % 2 == 0 else names[::-1]
            for n in order:
                t0 = time.perf_counter()
                mx.eval(cases[n]())
                samples[n].append((time.perf_counter() - t0) * 1e3)

        row = {"kv": kv, "S": S}
        for n in names:
            row[n] = statistics.median(samples[n])
        diffs = {}
        for r in row_widths:
            name = f"R{r}"
            row[f"{name}_diff_ms"] = row["whole"] - row[name]
            d = float(mx.abs(outs["whole"].astype(mx.float32) - outs[name].astype(mx.float32)).max())
            diffs[name] = d
            row[f"{name}_max_abs_diff"] = d
        res.append(row)

        pieces = " ".join(
            f"R={r:3d} {row[f'R{r}']:7.2f}ms (diff {row[f'R{r}_diff_ms']:+6.2f}ms, "
            f"max|d|={row[f'R{r}_max_abs_diff']:.4f})"
            for r in row_widths
        )
        print(f"kv={kv:6d}  whole {row['whole']:7.2f}ms | {pieces}", flush=True)
        del q, k, v, mask, outs, cases

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump({"note": __doc__, "rows": res}, f, ensure_ascii=False, indent=1)
        print("書き出し:", a.json)
    os._exit(0)


if __name__ == "__main__":
    main()
