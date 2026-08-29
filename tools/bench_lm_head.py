"""lm_head (2560 -> 248320) に削る余地があるかを直接測る。

`tools/ablate_gdn.py` の lm_head 段は壊れていた (71.78ms という、帯域下限を
下回る 6.03ms が出た)。ablation ではなく**その層だけを直接回す**。

重みは 8bit / group_size 64 で 636MB/token。400GB/s でも 1.59ms、実効
250GB/s なら 2.54ms。**重み読みは融合では削れない**ので、見るべきは

1. いま何 GB/s 出ているか (stock の実効帯域)
2. mlxturbo の自作経路 (nocap / mma) が M=1 で stock に勝てるか
   -> 経路表 `mlxturbo/kernels/dispatch.py` に (2560, 248320) は無い。
      既存の較正は M=5..16 (投機検証の幅) 向けで、M=1 は stock 固定
3. 4bit に落としたらどれだけ縮むか (レシピ側の判断材料。品質は別途 KLD)

    tools/biglock.sh uv run python tools/bench_lm_head.py \
        --model ~/models/qwen38fn-mlx-v-stream --ngram ~/models/qwen38fn-ngram-4bit

**--ngram を忘れると load が落ちる。**vendored arch は import 時に
FASTMLX_NGRAM_DISK を読み、立っていないと n-gram を本体持ちとして組むので
「Missing 128 parameters」になる (このセッションで 2 回踏んだ)。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def bench(fn, n=50, reps=5) -> float:
    import mlx.core as mx

    out = []
    for _ in range(reps):
        for _ in range(5):
            mx.eval(fn())
        t = time.perf_counter()
        for _ in range(n):
            mx.eval(fn())
        out.append((time.perf_counter() - t) / n * 1e6)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None,
                    help="n-gram サイドカー。ngram_disk で焼いたモデルには必須")
    ap.add_argument("--widths", default="1,2,4,8")
    args = ap.parse_args()

    import os

    # arch の import より先に立てること (docstring 参照)
    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    model, _ = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)
    head = model.lm_head
    w, s, b = head["weight"], head["scales"], head["biases"]
    gs, bits = head.group_size, head.bits
    n_out, k_words = w.shape
    k = k_words * 32 // bits
    nbytes = k * n_out * bits / 8 + s.size * 2 + b.size * 2
    print(f"lm_head: {k} -> {n_out}  {bits}bit gs={gs}  重み {nbytes / 1e6:.0f}MB")

    # 参考用に 4bit / 6bit へ打ち直したものも作る (品質は見ない、速度だけ)
    deq = mx.dequantize(w, s, b, group_size=gs, bits=bits)
    alts = {}
    for nb in (4, 6):
        if nb >= bits:
            continue
        q2, s2, b2 = mx.quantize(deq, group_size=64, bits=nb)
        mx.eval(q2, s2, b2)
        alts[nb] = (q2, s2, b2, 64, nb)
    del deq

    from mlxturbo.kernels.dispatch import quantized_matmul as dispatch_qmm

    for m in [int(v) for v in args.widths.split(",")]:
        x = mx.random.normal((m, k)).astype(mx.bfloat16)
        mx.eval(x)
        print(f"\n--- M={m} ---")

        def stock():
            return mx.quantized_matmul(
                x, w, scales=s, biases=b, transpose=True, group_size=gs, bits=bits
            )

        us = bench(stock)
        print(f"  stock ({bits}bit)      {us:8.2f} us  実効 {nbytes / us / 1000:6.1f} GB/s")

        # mlxturbo の経路表を (k, n_out) について強制的に有効化して比べる
        for route in ("nocap", "mma"):
            table = {(k, n_out): tuple([route] * 17)}

            def routed(r=route, t=table):
                return dispatch_qmm(
                    x, w, s, b, group_size=gs, bits=bits, table=t
                )

            try:
                us_r = bench(routed)
                print(f"  経路 {route:5s}          {us_r:8.2f} us  ({us / us_r:.2f}x)")
            except Exception as e:  # 適格外なら内部で stock に落ちる
                print(f"  経路 {route:5s}          失敗: {str(e)[:60]}")

        for nb, (q2, s2, b2, g2, _) in alts.items():
            nb_bytes = k * n_out * nb / 8 + s2.size * 2 + b2.size * 2

            def lower(q2=q2, s2=s2, b2=b2, g2=g2, nb=nb):
                return mx.quantized_matmul(
                    x, q2, scales=s2, biases=b2, transpose=True,
                    group_size=g2, bits=nb,
                )

            us_l = bench(lower)
            print(f"  {nb}bit に落とす      {us_l:8.2f} us  ({us / us_l:.2f}x)"
                  f"  重み {nb_bytes / 1e6:.0f}MB  実効 {nb_bytes / us_l / 1000:6.1f} GB/s")


if __name__ == "__main__":
    main()
