"""head_dim 256 の prefill attention が MLX でどの経路に落ちるかを費用で見る。

MLX 0.32.2 の `ScaledDotProductAttention::use_fallback` は S>8 で head_dim が
192/256 のとき常に true (NAX 機を除く)。つまり Flash-Next (head_dim 256) の
prefill attention は融合 (steel) ではなく、スコア [H, S, kv] を実体化する
素の経路で走る。ここでは同じ FLOP・同じバイト数の head_dim 128 (融合経路) と
並べ、融合カーネルを書いたときの見込みを出す。1 プロセス内で交互 (ABAB)。
モデルは読まない。GPU が空いているときに走らせること。
"""
from __future__ import annotations
import argparse, json, os, statistics, time
import mlx.core as mx


def qsa_like_mask(S: int, kv: int, block: int = 4, keep: int = 512) -> mx.array:
    """QSA 風の bool マスク: causal & (行ごとにランダムな keep ブロックだけ可視)。"""
    offset = kv - S
    rows = mx.arange(S)[:, None] + offset
    cols = mx.arange(kv)[None, :]
    causal = cols <= rows
    n_blocks = (kv + block - 1) // block
    if n_blocks <= keep:
        return causal[None, None]
    # 行ごとに乱数スコアの上位 keep ブロックを可視にする
    scores = mx.random.uniform(shape=(S, n_blocks))
    top = mx.argpartition(-scores, kth=keep - 1, axis=-1)[:, :keep]
    vis = mx.zeros((S, n_blocks), dtype=mx.bool_)
    vis = mx.put_along_axis(vis, top, mx.ones((S, keep), dtype=mx.bool_), axis=-1)
    vis = mx.repeat(vis, block, axis=-1)[:, :kv]
    return (causal & vis)[None, None]


def bench(fn, reps: int, warm: int = 2) -> float:
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=2048)
    ap.add_argument("--kvs", default="2048,8192,16896")
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="bench/results/sdpa-headdim-micro.json")
    a = ap.parse_args()
    S, Hq, Hk = a.S, a.heads, a.kv_heads
    res = []
    for kv in (int(x) for x in a.kvs.split(",")):
        mask = qsa_like_mask(S, kv); mx.eval(mask)
        cases = {}
        # 現行: head_dim 256、bool 配列マスク (fallback 経路)
        q = mx.random.normal((1, Hq, S, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, Hk, kv, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, Hk, kv, 256)).astype(mx.bfloat16)
        mx.eval(q, k, v)
        cases["d256_mask"] = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=256 ** -0.5, mask=mask)
        cases["d256_causal"] = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=256 ** -0.5, mask="causal")
        # 同じ FLOP・同じバイト数を head_dim 128 x 2 倍の head で (融合 steel 経路)
        q2 = mx.random.normal((1, Hq * 2, S, 128)).astype(mx.bfloat16)
        k2 = mx.random.normal((1, Hk * 2, kv, 128)).astype(mx.bfloat16)
        v2 = mx.random.normal((1, Hk * 2, kv, 128)).astype(mx.bfloat16)
        mx.eval(q2, k2, v2)
        cases["d128x2_mask"] = lambda: mx.fast.scaled_dot_product_attention(q2, k2, v2, scale=128 ** -0.5, mask=mask)
        cases["d128x2_causal"] = lambda: mx.fast.scaled_dot_product_attention(q2, k2, v2, scale=128 ** -0.5, mask="causal")
        # 交互に測る (ABCD を reps 回)
        names = list(cases)
        samples = {n: [] for n in names}
        for n in names:
            bench(cases[n], 1, warm=2)
        for r in range(a.reps):
            order = names if r % 2 == 0 else names[::-1]
            for n in order:
                t0 = time.perf_counter(); mx.eval(cases[n]()); samples[n].append((time.perf_counter() - t0) * 1e3)
        row = {"S": S, "kv": kv, **{n: statistics.median(v) for n, v in samples.items()}}
        row["fallback_over_fused_mask"] = row["d256_mask"] / row["d128x2_mask"]
        res.append(row)
        print(f"S={S} kv={kv:6d}  d256 mask {row['d256_mask']:7.1f} ms  causal {row['d256_causal']:7.1f} ms | "
              f"d128x2 (融合) mask {row['d128x2_mask']:7.1f} ms  causal {row['d128x2_causal']:7.1f} ms | "
              f"fallback/融合 = {row['fallback_over_fused_mask']:.2f}x", flush=True)
        del q, k, v, q2, k2, v2, mask
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"note": __doc__, "rows": res}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("書き出し:", a.out)
    os._exit(0)


if __name__ == "__main__":
    main()
