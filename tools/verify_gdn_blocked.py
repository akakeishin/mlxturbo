"""GDN のブロック化スキャンを逐次版と突き合わせる (合成テンソル、実モデル不要)。

`mlxturbo/kernels/gated_delta_blocked.py` は加算順が逐次版と違うのでビット一致
しない。**どこまで違ってよいかを、逐次版自身の誤差で測る。**

三者を比べる:

- `ops`     : `gated_delta_ops` を fp32 入力で回したもの。位置ごとに MLX の
              op で更新するだけなので、これを基準にする
- `seq`     : `gated_delta_kernel` (本番の逐次 Metal カーネル)。基準との差が
              「今すでに許容されている誤差」
- `blocked` : この変更で入るブロック化スキャン

**合格条件は「blocked の対 ops 誤差が seq の対 ops 誤差と同じ桁に収まること」。**
絶対値の閾値を先に決めても、入力の大きさが変わると意味が変わる。

    tools/biglock.sh .venv/bin/python tools/verify_gdn_blocked.py
"""

from __future__ import annotations

import argparse
import math

import mlx.core as mx
import mlx.nn as nn

import mlxturbo  # noqa: F401  (qwen4_exp の解決と mlx 契約の検査)
from mlx_lm.models.gated_delta import (
    compute_g,
    gated_delta_kernel,
    gated_delta_ops,
)
from mlxturbo.kernels import gated_delta_blocked as gdb

# Qwen3.8-Flash-Next の GDN の形
N_K, N_V, DK, DV = 16, 48, 128, 128


def _make_inputs(B, T, dtype, seed):
    """本番の `GatedDeltaNet.__call__` が渡すのと同じ性質の入力を作る。

    q/k は rms_norm 済み + スケール済み、v は silu 出力相当、a/b は素の射影。
    """
    mx.random.seed(seed)
    q = mx.random.normal((B, T, N_K, DK)).astype(dtype)
    k = mx.random.normal((B, T, N_K, DK)).astype(dtype)
    v = mx.random.normal((B, T, N_V, DV)).astype(dtype)
    inv = DK**-0.5
    q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv * mx.fast.rms_norm(k, None, 1e-6)
    v = nn.silu(v)
    a = mx.random.normal((B, T, N_V)).astype(dtype)
    b = mx.random.normal((B, T, N_V)).astype(dtype)
    A_log = mx.random.normal((N_V,)) * 0.5
    dt_bias = mx.ones((N_V,))
    state = mx.random.normal((B, N_V, DV, DK)) * 0.1
    return q, k, v, a, b, A_log, dt_bias, state.astype(mx.float32)


def _rel(x, ref):
    """相対誤差 (最大 / RMS)。ref の RMS で割る。"""
    x = x.astype(mx.float32)
    ref = ref.astype(mx.float32)
    d = x - ref
    scale = math.sqrt(float(mx.mean(ref * ref)))
    if scale == 0:
        scale = 1.0
    return (
        float(mx.max(mx.abs(d))) / scale,
        math.sqrt(float(mx.mean(d * d))) / scale,
    )


def run(B, T, dtype, seed, block, ref_ops):
    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(B, T, dtype, seed)
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)

    y_seq, s_seq = gated_delta_kernel(q, k, v, g, beta, state, None)
    mx.eval(y_seq, s_seq)

    y_blk, s_blk = gdb.gated_delta_update_blocked(
        q, k, v, a, b, A_log, dt_bias, state, block
    )
    mx.eval(y_blk, s_blk)

    print(f"\n== B={B} T={T} dtype={dtype} block={block or gdb.BLOCK} seed={seed}")
    if ref_ops:
        # fp32 入力で位置ごとに回した参照 (T に比例して遅いので短いときだけ)
        y_ref, s_ref = gated_delta_ops(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            g, beta, state, None,
        )
        mx.eval(y_ref, s_ref)
        for name, y_, s_ in (("seq", y_seq, s_seq), ("blocked", y_blk, s_blk)):
            ym, yr = _rel(y_, y_ref)
            sm, sr = _rel(s_, s_ref)
            print(f"  {name:8s} vs ops : y max={ym:.3e} rms={yr:.3e} | "
                  f"state max={sm:.3e} rms={sr:.3e}")
    ym, yr = _rel(y_blk, y_seq)
    sm, sr = _rel(s_blk, s_seq)
    print(f"  blocked  vs seq : y max={ym:.3e} rms={yr:.3e} | "
          f"state max={sm:.3e} rms={sr:.3e}")
    return yr, sr


def bench(B, T, blocks, reps=5):
    """ブロック長を選ぶための温キャッシュのマイクロ。

    **絶対値は信じないこと** (CLAUDE.md「計測の作法」)。ここで見るのは
    ブロック長どうしの順序だけで、採否は in-model の実測で決める。
    """
    import time

    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(B, T, mx.bfloat16, 7)
    g = compute_g(A_log, a, dt_bias)
    beta = mx.sigmoid(b)

    def timeit(fn):
        for _ in range(2):
            mx.eval(*fn())
        ts = []
        for _ in range(reps):
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(*fn())
            mx.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return min(ts)

    print(f"\n== micro (1 層ぶん、B={B} T={T}、最小値 / {reps} 回)")
    base = timeit(lambda: gated_delta_kernel(q, k, v, g, beta, state, None))
    print(f"  seq (逐次カーネル)   {base:8.3f} ms")
    for blk in blocks:
        t = timeit(lambda blk=blk: gdb.gated_delta_update_blocked(
            q, k, v, a, b, A_log, dt_bias, state, blk))
        print(f"  blocked C={blk:<4d}      {t:8.3f} ms   x{base / t:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="16,32,64,128",
                    help="試すブロック長 (カンマ区切り)")
    ap.add_argument("--bench", action="store_true",
                    help="ブロック長を選ぶマイクロだけ回す")
    args = ap.parse_args()
    blocks = [int(x) for x in args.blocks.split(",")]

    if args.bench:
        bench(1, 2048, blocks)
        return

    # 1. 参照つき (短い T)。逐次版自身の誤差と並べて読む
    for blk in blocks:
        run(1, 256, mx.bfloat16, 0, blk, ref_ops=True)
    # 2. C の倍数でない T (末尾のパディング経路)
    run(1, 300, mx.bfloat16, 1, 64, ref_ops=True)
    # 3. fp32 入力 (丸めの寄与を切り離す)
    run(1, 256, mx.float32, 2, 64, ref_ops=True)
    # 4. 本番の prefill 幅
    for blk in blocks:
        run(1, 2048, mx.bfloat16, 3, blk, ref_ops=False)
    # 5. state=None (最初のチャンク)
    q, k, v, a, b, A_log, dt_bias, _ = _make_inputs(1, 256, mx.bfloat16, 4)
    y0, s0 = gdb.gated_delta_update_blocked(q, k, v, a, b, A_log, dt_bias, None, 64)
    y1, s1 = gated_delta_kernel(
        q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b),
        mx.zeros((1, N_V, DV, DK), dtype=mx.float32), None,
    )
    mx.eval(y0, s0, y1, s1)
    print("\n== state=None")
    print(f"  blocked vs seq : y {_rel(y0, y1)} | state {_rel(s0, s1)}")


if __name__ == "__main__":
    main()
