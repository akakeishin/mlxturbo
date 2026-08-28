"""stock の `mx.gather_qmm` が、デコード形状で実際に何バイト動かしているかを測る。

親の指示 (docs/KERNEL-BRIEF-HC.md の MoE 節の訂正) による。別実装
(ddalcu/mlx-serve) が「take 経路の 3 倍ではなく理想の量を動かす専用 gather-qmv」
を持っており、こちらの `tools/micro_moe_gdn.py` のビット掃引の切片
(48 層で 6.7ms = gather_qmm 1 回あたり 46us) とも符合する。

## 何で切り分けるか (2)

**同じバイト数を動かす密な量子化行列積と比べる。**top_k 個ぶんの行を連結した
密な重み (top_k*640 x 2560) は、理想の 8.19MB ちょうどを動かす。gather_qmm が
その何倍かかるかが、そのまま「余分に動かしているか / 散らばりで遅いか」の
合計の罰則になる。帯域の絶対値を較正しなくて済む。

## 何で切り分けるか (1)

**バンクのエキスパート数を変えて時間が変わるかを見る。**top_k=10 を固定して
バンクを 64 -> 512 に増やしたとき、

- 時間が平ら       -> 選ばれた 10 個だけを読んでいる (理想)
- 時間が N に比例  -> バンク全体を読んでいる
- その間          -> 余分に読んでいる

帯域の較正が要らないので、環境差や熱の影響を受けにくい。比較対象として
「take で 10 個を実体化してから密な量子化行列積」も測る。

    uv run python tools/probe_gather_qmm.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
import numpy as np

D = 2560
INTER = 640
TOP_K = 10


# `mx.eval` 1 回には約 160us の固定費が乗る (セッション冒頭の実測)。1 呼び出し
# ごとに eval すると、測りたい 30-80us がその中に埋もれる。**CHAIN 回を直列に
# 積んでから 1 回だけ eval** して割る。呼び出しごとに添字を変えて CSE を避ける。
CHAIN = 48


def bench_chain(make, n=8, reps=5) -> float:
    """make(i) が i 番目の呼び出しを作る。CHAIN 本積んで 1 回 eval。"""
    def go():
        return [make(i) for i in range(CHAIN)]

    for _ in range(2):
        mx.eval(go())
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(n):
            mx.eval(go())
        out.append((time.perf_counter() - t) / n * 1e6 / CHAIN)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--experts", default="64,128,256,512")
    ap.add_argument("--topk", default="1,2,5,10,20",
                    help="top_k を振って傾きを見る (固定費に依らない)")
    args = ap.parse_args()

    gs = 64
    bits = args.bits
    sizes = [int(v) for v in args.experts.split(",")]
    n_max = max(sizes)

    mx.random.seed(0)
    dense = (mx.random.normal((n_max, INTER, D)) * 0.02).astype(mx.bfloat16)
    w, s, b = mx.quantize(dense, group_size=gs, bits=bits)
    del dense
    mx.eval(w, s, b)

    x = (mx.random.normal((1, 1, D)) * 1.0).astype(mx.bfloat16)
    mx.eval(x)

    ideal = TOP_K * INTER * D * bits / 8
    print(f"bank=[{n_max}, {INTER}, {D}] {bits}bit gs={gs}"
          f"   常駐 {mx.get_peak_memory() / 1e9:.2f}GB")
    print(f"理想 (top_k={TOP_K} 個ぶんだけ読む) = {ideal / 1e6:.2f} MB/投影\n")
    print(f"{'エキスパート数':>12s} {'gather_qmm':>12s} {'実効 GB/s':>10s}"
          f" {'take+qmm':>10s} {'比':>6s}")

    base = None
    for n in sizes:
        wn, sn, bn = w[:n], s[:n], b[:n]
        mx.eval(wn, sn, bn)
        # 呼び出しごとに別のエキスパート組を引く (CSE 回避 + 実機に近い散らばり)
        rng = np.random.default_rng(0)
        idxs = [mx.array(rng.choice(n, TOP_K, replace=False).reshape(1, 1, TOP_K)
                         .astype(np.uint32)) for _ in range(CHAIN)]
        mx.eval(idxs)

        def g(i):
            return mx.gather_qmm(
                x, wn, sn, bn, rhs_indices=idxs[i], transpose=True,
                group_size=gs, bits=bits,
            )

        us = bench_chain(g)

        # 比較: 10 個を実体化してから密な量子化行列積を回す
        def t(i):
            flat = idxs[i].reshape(-1)
            ws = mx.take(wn, flat, axis=0)
            ss = mx.take(sn, flat, axis=0)
            bs = mx.take(bn, flat, axis=0)
            return mx.quantized_matmul(
                x.reshape(1, 1, D), ws, scales=ss, biases=bs, transpose=True,
                group_size=gs, bits=bits,
            )

        try:
            us_t = bench_chain(t)
            tt = f"{us_t:10.1f}"
        except Exception as e:
            us_t = float("nan")
            tt = "       n/a"

        if base is None:
            base = us
        print(f"{n:12d} {us:12.1f} {ideal / us / 1000:10.1f} {tt} {us / base:6.2f}x")

    print("\n時間がエキスパート数に比例して伸びるなら、選ばれた 10 個以外も"
          "読んでいる。平らなら理想に近い。")

    # --- 密な等価物との比較 ---
    print(f"\n=== 同じバイト数を動かす密な行列積との比較 (bank={n_max}) ===")
    print(f"{'top_k':>6s} {'gather_qmm':>12s} {'密な行列積':>12s} {'倍率':>8s}"
          f" {'理想MB':>8s}")
    rng = np.random.default_rng(1)
    for tk in [int(v) for v in args.topk.split(",")]:
        if tk > n_max:
            continue
        ideal_tk = tk * INTER * D * bits / 8
        idxs = [mx.array(rng.choice(n_max, tk, replace=False)
                         .reshape(1, 1, tk).astype(np.uint32))
                for _ in range(CHAIN)]
        mx.eval(idxs)

        def g(i, tk=tk, idxs=idxs):
            return mx.gather_qmm(x, w, s, b, rhs_indices=idxs[i], transpose=True,
                                 group_size=gs, bits=bits)

        us_g = bench_chain(g)

        # 密: top_k*INTER 行を 1 本の重みにまとめる。動くバイト数は理想ちょうど
        dw = (mx.random.normal((tk * INTER, D)) * 0.02).astype(mx.bfloat16)
        qw, qs, qb = mx.quantize(dw, group_size=gs, bits=bits)
        del dw
        mx.eval(qw, qs, qb)
        xs = [(mx.random.normal((1, 1, D))).astype(mx.bfloat16) for _ in range(CHAIN)]
        mx.eval(xs)

        def d(i, qw=qw, qs=qs, qb=qb, xs=xs):
            return mx.quantized_matmul(xs[i], qw, scales=qs, biases=qb,
                                       transpose=True, group_size=gs, bits=bits)

        us_d = bench_chain(d)
        print(f"{tk:6d} {us_g:12.1f} {us_d:12.1f} {us_g / us_d:7.2f}x"
              f" {ideal_tk / 1e6:8.2f}")

    print("\n倍率が 1 に近ければ gather_qmm は理想に近い。3 前後なら、"
          "余分に読んでいるか散らばりで遅いかで、専用カーネルの余地がある。")


if __name__ == "__main__":
    main()
