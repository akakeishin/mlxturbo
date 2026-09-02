"""oMLX 移植の GDN blocked-seq Metal カーネルを逐次版と突き合わせる。

**GPU 必須。**`mlxturbo/kernels/gdn_blocked_metal.py` の kernel S は Metal
専用で CPU では動かない (このリポジトリでの一次検査は
`tools/vendor_fingerprint.py` 止まり -- そちらは既定 off の分岐に触れないので
CPU で走る。この検査は実際にカーネルを起動する)。

`gated_delta_blocked.py` (行列積版) と違い、こちらは逐次版と**同じ再帰**を
計算するだけ (チャンク分解も WY 表現も無い)。ビット一致は要求しない
(加算順が違う) が、逐次版どうしの差より 1 桁大きくなっていないかを見る。
fp32 state に対して相対誤差 1e-3 を合格の目安とし、外れたら
`** 1e-3 超え **` を付けて目立たせる。

    .venv/bin/python tools/verify_gdn_metal.py
    .venv/bin/python tools/verify_gdn_metal.py --bench-only
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import mlx.nn as nn

import mlxturbo  # noqa: F401  (qwen4_exp の解決と mlx 契約の検査)
from mlx_lm.models.gated_delta import gated_delta_update
from mlxturbo.kernels import gdn_blocked_metal as gbm

# Qwen3.8-Flash-Next の GDN の形 (Dk==128, Dv%32==0 の kernel S 制約を満たす)
N_K, N_V, DK, DV = 16, 48, 128, 128


def _make_inputs(B, T, dtype, seed, with_state):
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
    state = None
    if with_state:
        state = (mx.random.normal((B, N_V, DV, DK)) * 0.1).astype(mx.float32)
    return q, k, v, a, b, A_log, dt_bias, state


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


def run(B, T, dtype, seed, with_state, tol=1e-3):
    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(B, T, dtype, seed, with_state)

    # 逐次版 (本番の Metal カーネル、mlx_lm 標準)
    y_seq, s_seq = gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=True
    )
    mx.eval(y_seq, s_seq)

    # oMLX 移植の blocked-seq Metal カーネル
    y_met, s_met = gbm.gated_delta_update_blocked_metal(
        q, k, v, a, b, A_log, dt_bias, state
    )
    mx.eval(y_met, s_met)

    ym, yr = _rel(y_met, y_seq)
    sm, sr = _rel(s_met, s_seq)
    tag = "state あり" if with_state else "state=None"
    flag = "  ** 1e-3 超え **" if max(yr, sr) > tol else ""
    print(
        f"B={B} T={T} dtype={dtype} {tag}: "
        f"y max={ym:.3e} rel={yr:.3e} | state max={sm:.3e} rel={sr:.3e}{flag}"
    )
    return yr, sr


def bench(T=2048, reps=20, warmup=3):
    """T=2048 の壁時計。1 プロセス内で逐次版と交互に測る
    (CLAUDE.md の計測の作法: プロセスを分けた比較は熱・キャッシュで数 % ずれる)。

    **絶対値は信じないこと。**ここで見るのは 2 つの実装の比較だけで、
    採否は `tools/decode_ab.py --knob gdn-metal` の in-model A/B で決める。
    """
    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(1, T, mx.bfloat16, 9, True)

    def call_seq():
        return gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=True)

    def call_met():
        return gbm.gated_delta_update_blocked_metal(q, k, v, a, b, A_log, dt_bias, state)

    seq_ts: list[float] = []
    met_ts: list[float] = []
    for i in range(warmup + reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(*call_seq())
        mx.synchronize()
        dt_seq = (time.perf_counter() - t0) * 1e3

        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(*call_met())
        mx.synchronize()
        dt_met = (time.perf_counter() - t0) * 1e3

        if i >= warmup:
            seq_ts.append(dt_seq)
            met_ts.append(dt_met)

    def stats(ts):
        return min(ts), sum(ts) / len(ts)

    seq_min, seq_mean = stats(seq_ts)
    met_min, met_mean = stats(met_ts)
    print(
        f"\n== 壁時計 (T={T}、1 プロセス内で交互、{reps} 回、最初 {warmup} 回捨て)"
    )
    print(f"  seq   (逐次カーネル)      min={seq_min:8.3f} ms  mean={seq_mean:8.3f} ms")
    print(f"  metal (blocked-seq 移植)  min={met_min:8.3f} ms  mean={met_mean:8.3f} ms"
          f"   x{seq_min / met_min:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-only", action="store_true", help="壁時計だけ回す")
    ap.add_argument("--tol", type=float, default=1e-3, help="相対誤差の合格目安")
    args = ap.parse_args()

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        print("[verify_gdn_metal] GPU が既定デバイスでない。このスクリプトは"
              " GPU 必須 (Metal カーネル)。中断する。")
        return

    if not args.bench_only:
        for T in (64, 128, 2048):
            for with_state in (False, True):
                seed = T * 10 + (1 if with_state else 0)
                run(1, T, mx.bfloat16, seed, with_state, args.tol)

    bench()


if __name__ == "__main__":
    main()
