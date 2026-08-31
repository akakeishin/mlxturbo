"""糊 (モジュール間) の 3 容疑者を実重み・実形状で測る。

  1. router: 量子化 gate に fp32 入力を食わせる現行 vs bf16 入力 vs 逆量子化済み密 fp32
  2. 残差合成 hyper + (x*inject): 素の 3-4 op vs mx.compile
  3. GDN のキャッシュ更新 (concat + slice + contiguous) の単体コスト

    uv run python tools/glue_probe.py --model <path> --ngram <sidecar>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def med_ms(fn, reps=50):
    import mlx.core as mx

    for _ in range(5):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t) * 1000)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--width", type=int, default=3)
    args = ap.parse_args()
    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx

    import mlxturbo  # noqa: F401
    from mlx_lm import load

    model, tok = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)

    T = args.width
    layers = model.model.layers
    gates = [l.mlp.gate for l in layers]
    x32 = mx.random.normal((1, T, 2560)).astype(mx.float32)
    xbf = x32.astype(mx.bfloat16)
    dense = [mx.dequantize(g.weight, g.scales, g.biases,
                           group_size=g.group_size, bits=g.bits).astype(mx.float32)
             for g in gates]
    mx.eval(dense)

    r_f32 = med_ms(lambda: [g(x32) for g in gates])
    r_bf = med_ms(lambda: [g(xbf).astype(mx.float32) for g in gates])
    r_dense = med_ms(lambda: [x32 @ w.T for w in dense])
    print(f"router x48  fp32入力qmm={r_f32:.2f}ms  bf16入力qmm={r_bf:.2f}ms  密fp32={r_dense:.2f}ms")

    hc, d = 3, 2560
    hyper = mx.random.normal((1, T, hc * d)).astype(mx.bfloat16)
    xx = mx.random.normal((1, T, d)).astype(mx.bfloat16)
    inj = mx.random.uniform(shape=(1, T, hc)).astype(mx.bfloat16)

    def combine_eager(h_, x_, i_):
        return h_ + (x_[..., None, :] * i_[..., None]).reshape(*x_.shape[:-1], -1)

    combine_c = mx.compile(combine_eager)
    mx.eval(combine_c(hyper, xx, inj))
    e = med_ms(lambda: [combine_eager(hyper, xx, inj) for _ in range(96)])
    c = med_ms(lambda: [combine_c(hyper, xx, inj) for _ in range(96)])
    print(f"残差合成 x96  素={e:.2f}ms  compile={c:.2f}ms")

    conv_dim = 10240
    state = mx.random.normal((1, 3, conv_dim)).astype(mx.bfloat16)
    qkv = mx.random.normal((1, T, conv_dim)).astype(mx.bfloat16)

    def cache_ops():
        outs = []
        for _ in range(36):
            ci = mx.concatenate([state, qkv], axis=1)
            outs.append(mx.contiguous(ci[:, -3:, :]))
        return outs

    print(f"GDN cache更新 x36  {med_ms(cache_ops):.2f}ms")


if __name__ == "__main__":
    main()
