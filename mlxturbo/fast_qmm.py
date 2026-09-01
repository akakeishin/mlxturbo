# vendored from https://github.com/avlp12/mlx-lm/blob/main/mlx_lm/fast_qmm.py
# License checked (2026-08-26): the fork's LICENSE is MIT (Copyright Apple Inc.).
# Use, modification and redistribution are permitted under the MIT terms
# (retaining the license text).
# Measured on M3 Max: on a dependent chain, m=8 is 1.57x mx.quantized_matmul and
# m=6 is 1.25x. The real cause of the m-curve tax (the 5-row cap in qmv_wide,
# which re-reads the weights ceil(M/5) times) is avoided here with an 8x8 MMA
# tile + dequantization per quantization group + split-K.
# Copyright © 2026 Apple Inc.

"""Small-M quantized matmul that amortizes the weight read.

`mx.quantized_matmul` grows almost linearly in M for M in 2..8 while the dense
bf16 path is flat — the weight read is not amortized across rows until roughly
M >= 16 (ml-explore/mlx#4265). Everything that verifies several tokens at once
lives in that window: speculative decoding, MTP, small-batch serving.

This kernel puts the multiply on `simdgroup_matrix`. An 8x8 MMA tile covers
M <= 8 exactly, so each quantized weight group is read once and reused by every
row. Dequantization happens per quantization group (64 elements) rather than per
MMA tile (8), which cuts the barrier count eightfold — that single change took
the kernel from 0.29 ms to 0.15 ms at M=8.

Measured on M3 Ultra, (M,5120) @ (17408,5120)^T, 4-bit / group 64:

    M      MLX      ours    ratio
    1   0.042     0.136    0.31x
    4   0.122     0.141    0.86x
    7   0.204     0.149    1.37x
    8   0.232     0.152    1.52x

So MLX wins below M=4 (its GEMV path is at roofline) and loses above it. The
dispatch below reflects that: this is a supplement to `quantized_matmul`, not a
replacement.
"""

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

KC = 128           # x staging chunk (4KB) — total threadgroup use 12KB
NPT = 64           # output columns per threadgroup — amortizes the x staging
TGT = 256          # 8 simdgroups
M_MIN = 5          # measured crossover on a *dependent* chain (see below)
# Re-measured 2026-08-27 (after vectorizing the B staging and caching _zpad):
# M=5 wins on all 4 shapes in both dependent-chain runs (0.91-0.96x vs stock).
# At M=4 stock still wins, at 1.07-1.26x (qmv reads the weights once and is close
# to bandwidth). The observation that justified the old crossover of M=6 — "M=4 is
# 0.98-1.10x, inside the noise, and it slowed MTP k=3 down in the model" — still
# holds for M=4. As before, the verdict is based on dependent-chain latency, not
# on one-shot throughput.
M_MAX = 8          # one MMA tile
# Below this the grid is ceil(N/64) threadgroups and the GPU sits idle; the
# model has 96 layers with N=48, which would each get a single threadgroup.
N_MIN = 4096

_SRC = r"""
    const int K = KD, N = ND, M = MD;
    const int KPS = KD / 8;                 // K span per simdgroup

    uint tid  = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg   = tid >> 5;
    uint lane = tid & 31;

    int n0 = (int)tgid * 8;                 // one threadgroup covers 8 output columns

    // x is read directly by MMA from device memory (bf16 input, float accumulation).
    // With no staging, threadgroup memory drops to 10KB, and more importantly the
    // critical path gets shorter.
    threadgroup bfloat16_t bs[8 * 512];     // per simdgroup: 64k x 8n, 8KB
    threadgroup float red[8 * 64];          // cross-simdgroup reduction, 2KB

    simdgroup_matrix<float, 8, 8> C = simdgroup_matrix<float, 8, 8>(0);
    threadgroup bfloat16_t* bt = bs + sg * 512;

    // ── split-K: 8 simdgroups split K into 8 parts, each running its own short
    //    serial loop. (the earlier version had one threadgroup walk all of K,
    //    which put 160 barrier pairs on the critical path)
    int kbeg = (int)sg * KPS;
    for (int kk = 0; kk < KPS; kk += 64) {
        int ka = kbeg + kk;
        int j  = (int)(lane & 7);
        int kq = (int)(lane >> 3);
        int n  = n0 + j;
        // The B slab is column-major, bt[j*64 + k]. A lane's 16 elements then sit
        // contiguously and can be written with 4 vec4 stores (this retired the
        // 16 scalar stores at stride 8).
        threadgroup bfloat16_t* bc = bt + j * 64 + kq * 16;
        if (n < N) {
            int g = ka >> 6;
            float s  = (float)sc[(size_t)n * (K / 64) + g];
            float bb = (float)bi[(size_t)n * (K / 64) + g];
            const device uint* wr = w + (size_t)n * (K / 8) + (ka >> 3) + kq * 2;
            uint p0 = wr[0], p1 = wr[1];
            vec<bfloat16_t, 4> v0, v1, v2, v3;
            for (int t = 0; t < 4; ++t)
                v0[t] = (bfloat16_t)((float)((p0 >> (4 * t)) & 15u) * s + bb);
            for (int t = 0; t < 4; ++t)
                v1[t] = (bfloat16_t)((float)((p0 >> (4 * (t + 4))) & 15u) * s + bb);
            for (int t = 0; t < 4; ++t)
                v2[t] = (bfloat16_t)((float)((p1 >> (4 * t)) & 15u) * s + bb);
            for (int t = 0; t < 4; ++t)
                v3[t] = (bfloat16_t)((float)((p1 >> (4 * (t + 4))) & 15u) * s + bb);
            ((threadgroup vec<bfloat16_t, 4>*)bc)[0] = v0;
            ((threadgroup vec<bfloat16_t, 4>*)bc)[1] = v1;
            ((threadgroup vec<bfloat16_t, 4>*)bc)[2] = v2;
            ((threadgroup vec<bfloat16_t, 4>*)bc)[3] = v3;
        } else {
            vec<bfloat16_t, 4> z = vec<bfloat16_t, 4>(bfloat16_t(0));
            for (int t = 0; t < 4; ++t) ((threadgroup vec<bfloat16_t, 4>*)bc)[t] = z;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<bfloat16_t, 8, 8> A, B;
        for (int kt = 0; kt < 8; ++kt) {
            simdgroup_load(A, x + ka + kt * 8, K);   // x[0:8, ka+8kt ..] — padded to 8 rows
            simdgroup_load(B, bt + kt * 8, 64, ulong2(0, 0), true);  // read the column-major slab transposed
            simdgroup_multiply_accumulate(C, A, B, C);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(C, red + sg * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // sum the 8 partial results and write them out.
    for (int i = (int)tid; i < 64; i += 256) {
        int m = i >> 3, j = i & 7;
        int n = n0 + j;
        if (m < M && n < N) {
            float v = 0.0f;
            for (int q = 0; q < 8; ++q) v += red[q * 64 + i];
            out[(size_t)m * N + n] = (bfloat16_t)v;
        }
    }
"""


M_WIDE_MAX = 16    # up to the second MMA tile — the weight read is identical to M=8
# On stock, verification widths 9-16 cost 3.8x M=8 ([I128]). The B tile is
# already resident in threadgroup memory, so adding one more accumulator for
# rows 8-15 lets that width share M=8's weight-read cost outright. The
# in-window (M<=8) path is left untouched.

# Wide (M=9..16) の既定は off。有効化は `MLXLM_FAST_QMM_WIDE=1` (公開の環境
# 変数) か、この内部フラグの直接代入 (kernels/dispatch.py の MMA 経路が使う)
# のどちらか -- dispatch.py は自分の経路だけ wide を強制したいので、
# os.environ を書き換えてプロセス全体の既定を横取りする代わりにこのフラグを
# 立てる (A2, Opus 設計レビュー指摘)。
_WIDE_FORCE_ON = False


def _wide_enabled() -> bool:
    return _WIDE_FORCE_ON or os.environ.get("MLXLM_FAST_QMM_WIDE") == "1"


_SRC_WIDE = r"""
    const int K = KD, N = ND, M = MD;
    const int KPS = KD / 8;

    uint tid  = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg   = tid >> 5;
    uint lane = tid & 31;

    int n0 = (int)tgid * 8;

    threadgroup bfloat16_t bs[8 * 512];     // B tile — shared by both row-tiles (that's the whole point)
    threadgroup float red[8 * 128];         // 8 simdgroups x 2 row-tiles x 64, 4KB

    simdgroup_matrix<float, 8, 8> C0 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> C1 = simdgroup_matrix<float, 8, 8>(0);
    threadgroup bfloat16_t* bt = bs + sg * 512;

    int kbeg = (int)sg * KPS;
    for (int kk = 0; kk < KPS; kk += 64) {
        int ka = kbeg + kk;
        int j  = (int)(lane & 7);
        int kq = (int)(lane >> 3);
        int n  = n0 + j;
        // The B slab is column-major, bt[j*64 + k]. A lane's 16 elements then sit
        // contiguously and can be written with 4 vec4 stores (this retired the
        // 16 scalar stores at stride 8).
        threadgroup bfloat16_t* bc = bt + j * 64 + kq * 16;
        if (n < N) {
            int g = ka >> 6;
            float s  = (float)sc[(size_t)n * (K / 64) + g];
            float bb = (float)bi[(size_t)n * (K / 64) + g];
            const device uint* wr = w + (size_t)n * (K / 8) + (ka >> 3) + kq * 2;
            uint p0 = wr[0], p1 = wr[1];
            vec<bfloat16_t, 4> v0, v1, v2, v3;
            for (int t = 0; t < 4; ++t)
                v0[t] = (bfloat16_t)((float)((p0 >> (4 * t)) & 15u) * s + bb);
            for (int t = 0; t < 4; ++t)
                v1[t] = (bfloat16_t)((float)((p0 >> (4 * (t + 4))) & 15u) * s + bb);
            for (int t = 0; t < 4; ++t)
                v2[t] = (bfloat16_t)((float)((p1 >> (4 * t)) & 15u) * s + bb);
            for (int t = 0; t < 4; ++t)
                v3[t] = (bfloat16_t)((float)((p1 >> (4 * (t + 4))) & 15u) * s + bb);
            ((threadgroup vec<bfloat16_t, 4>*)bc)[0] = v0;
            ((threadgroup vec<bfloat16_t, 4>*)bc)[1] = v1;
            ((threadgroup vec<bfloat16_t, 4>*)bc)[2] = v2;
            ((threadgroup vec<bfloat16_t, 4>*)bc)[3] = v3;
        } else {
            vec<bfloat16_t, 4> z = vec<bfloat16_t, 4>(bfloat16_t(0));
            for (int t = 0; t < 4; ++t) ((threadgroup vec<bfloat16_t, 4>*)bc)[t] = z;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<bfloat16_t, 8, 8> A0, A1, B;
        for (int kt = 0; kt < 8; ++kt) {
            simdgroup_load(B, bt + kt * 8, 64, ulong2(0, 0), true);  // read the column-major slab transposed
            simdgroup_load(A0, x + ka + kt * 8, K);              // rows 0-7
            simdgroup_multiply_accumulate(C0, A0, B, C0);
            simdgroup_load(A1, x + (size_t)8 * K + ka + kt * 8, K);  // rows 8-15
            simdgroup_multiply_accumulate(C1, A1, B, C1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(C0, red + sg * 128, 8);
    simdgroup_store(C1, red + sg * 128 + 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = (int)tid; i < 128; i += 256) {
        int m = i >> 3, j = i & 7;          // m 0-15 (0-7 is C0's range, 8-15 is C1's)
        int mm = (m < 8) ? m : (m - 8);
        int slot = (m < 8) ? (mm * 8 + j) : (64 + mm * 8 + j);
        int mrow = (m < 8) ? m : m;
        int n = n0 + j;
        if (mrow < M && n < N) {
            float v = 0.0f;
            for (int q = 0; q < 8; ++q) v += red[q * 128 + slot];
            out[(size_t)mrow * N + n] = (bfloat16_t)v;
        }
    }
"""

_ZPAD: dict = {}


def _zpad(rows: int, k: int):
    """Constant array used for zero padding, cached so we don't dispatch mx.zeros every time.

    The alternative of synthesizing the non-existent rows inside the kernel with a
    lane-side mask was rejected on dependent-chain measurements (building A per
    lane as scalars raised the dispatch cost by 20-45% and it even lost to nocap).
    Keeping the padding on the host side is what currently wins.
    """
    key = (rows, k)
    z = _ZPAD.get(key)
    if z is None:
        z = mx.zeros((rows, k), dtype=mx.bfloat16)
        mx.eval(z)
        _ZPAD[key] = z
    return z


_KERNEL_WIDE = mx.fast.metal_kernel(
    name="qmm_mma4_wide",
    input_names=["x", "w", "sc", "bi"],
    output_names=["out"],
    source=_SRC_WIDE,
    ensure_row_contiguous=True,
)


_KERNEL = mx.fast.metal_kernel(
    name="qmm_mma4",
    input_names=["x", "w", "sc", "bi"],
    output_names=["out"],
    source=_SRC,
    # raw linear indexing in the source requires contiguous rows; pin the
    # guarantee instead of relying on the MLX default
    ensure_row_contiguous=True,
)


def _eligible(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    K: int,
    N: int,
) -> bool:
    return (
        bits == 4
        and group_size == 64
        # split-K carves K into 8 simdgroup regions walked in 64-wide groups,
        # so K must divide by 512 (K%128 alone lets regions overlap, e.g. K=640)
        and K % 512 == 0
        and N >= N_MIN
        and x.dtype == mx.bfloat16
        # the kernel indexes packed operands with these exact layouts; anything
        # else must fall back rather than read out of bounds
        and w.dtype == mx.uint32
        and w.shape == (N, K // 8)
        and scales.shape == (N, K // 64)
        and biases.shape == (N, K // 64)
        and mx.default_device() == mx.gpu
    )


def _wide_qmm(x, w, scales, biases, *, M: int, K: int, N: int):
    """M in (8, 16] — two row-tiles. Uses the same shape convention as the in-window path."""
    flat = x.reshape(-1, K)
    if M < 16:                       # the kernel reads a 16-row tile directly from device memory
        flat = mx.concatenate([flat, _zpad(16 - M, K)], axis=0)
    (out,) = _KERNEL_WIDE(
        inputs=[flat, w, scales, biases],
        template=[("KD", K), ("ND", N), ("MD", M)],
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
        grid=(((N + 7) // 8) * TGT, 1, 1),
        threadgroup=(TGT, 1, 1),
    )
    return out.reshape(*x.shape[:-1], N)


# --- width histogram (diagnostics only, enable with MLXLM_QMM_HIST=1) ---
_HIST_ON = os.environ.get("MLXLM_QMM_HIST") == "1"
_WIDTH_HIST: dict = {}


def width_histogram():
    """Call count by M (verification width), and which path each one took. Diagnostics only."""
    return dict(sorted(_WIDTH_HIST.items()))


def fast_qmm(x, w, scales, biases, *, group_size: int, bits: int):
    """Drop-in for `mx.quantized_matmul(..., transpose=True)` in the small-M window.

    Falls back whenever the shape or dtype is outside what the kernel handles, so
    callers never have to check.
    """
    K = x.shape[-1]
    M = 1
    for d in x.shape[:-1]:
        M *= d
    N = w.shape[0]
    if _HIST_ON:
        _WIDTH_HIST[M] = _WIDTH_HIST.get(M, 0) + 1
    if (
        M_MAX < M <= M_WIDE_MAX
        and _wide_enabled()
        and _eligible(x, w, scales, biases, group_size, bits, K, N)
    ):
        return _wide_qmm(x, w, scales, biases, M=M, K=K, N=N)
    if not (
        M_MIN <= M <= M_MAX
        and _eligible(x, w, scales, biases, group_size, bits, K, N)
    ):
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
    flat = x.reshape(M, K)
    if M < 8:  # the kernel reads an 8-row MMA tile directly from device memory — this keeps it from reading out of bounds
        flat = mx.concatenate([flat, _zpad(8 - M, K)], axis=0)
    (out,) = _KERNEL(
        inputs=[flat, w, scales, biases],
        template=[("KD", K), ("ND", N), ("MD", M)],
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
        grid=(((N + 7) // 8) * TGT, 1, 1),
        threadgroup=(TGT, 1, 1),
    )
    return out.reshape(*x.shape[:-1], N)


_ORIGINAL_CALL = None
_ORIGINAL_SHARDED_CALLS: dict = {}


def _qmm_or_fallback(self, x, original):
    """fast_qmm for the affine window, otherwise just the original GEMM path (no communication)."""
    if "biases" not in self or getattr(self, "mode", "affine") != "affine":
        return None  # caller falls back to the full original __call__
    return fast_qmm(
        x,
        self["weight"],
        self["scales"],
        self["biases"],
        group_size=self.group_size,
        bits=self.bits,
    )


def enable(model: Any = None) -> None:
    """Route quantized linears through `fast_qmm`.

    Patching the class rather than each instance keeps this reversible and keeps
    quantized layers created later (draft models, adapters) on the same path.
    The TP sharded variants (`Quantized{AllToSharded,ShardedToAll}Linear`) are
    NOT subclasses of `nn.QuantizedLinear`, so they are patched separately with
    their communication step (sum_gradients / all_sum) preserved verbatim —
    without this, tensor-parallel runs silently fall back to the small-M slope
    the kernel exists to fix.
    Set `MLXLM_NO_FAST_QMM=1` to opt out without touching call sites.
    """
    global _ORIGINAL_CALL
    if _ORIGINAL_CALL is not None or os.environ.get("MLXLM_NO_FAST_QMM") == "1":
        return
    _ORIGINAL_CALL = nn.QuantizedLinear.__call__

    def __call__(self, x):
        # the kernel is affine(scales+biases)-only — layers in other modes such as
        # nvfp4/mxfp4 have no biases and would die with a KeyError (confirmed in
        # practice during a community build's KL measurements).
        y = _qmm_or_fallback(self, x, _ORIGINAL_CALL)
        if y is None:
            return _ORIGINAL_CALL(self, x)
        if "bias" in self:
            y = y + self["bias"]
        return y

    nn.QuantizedLinear.__call__ = __call__

    try:
        from mlx.nn.layers.distributed import (
            QuantizedAllToShardedLinear,
            QuantizedShardedToAllLinear,
            sum_gradients,
        )
    except ImportError:
        return

    _ORIGINAL_SHARDED_CALLS[QuantizedAllToShardedLinear] = (
        QuantizedAllToShardedLinear.__call__
    )
    _ORIGINAL_SHARDED_CALLS[QuantizedShardedToAllLinear] = (
        QuantizedShardedToAllLinear.__call__
    )

    def __call_a2s__(self, x):
        orig = _ORIGINAL_SHARDED_CALLS[QuantizedAllToShardedLinear]
        x = sum_gradients(self.group)(x)
        y = _qmm_or_fallback(self, x, orig)
        if y is None:
            return orig(self, x)
        if "bias" in self:
            y = y + self["bias"]
        return y

    def __call_s2a__(self, x):
        orig = _ORIGINAL_SHARDED_CALLS[QuantizedShardedToAllLinear]
        y = _qmm_or_fallback(self, x, orig)
        if y is None:
            return orig(self, x)
        y = mx.distributed.all_sum(y, group=self.group)
        if "bias" in self:
            y = y + self["bias"]
        return y

    QuantizedAllToShardedLinear.__call__ = __call_a2s__
    QuantizedShardedToAllLinear.__call__ = __call_s2a__


def disable() -> None:
    global _ORIGINAL_CALL
    if _ORIGINAL_CALL is not None:
        nn.QuantizedLinear.__call__ = _ORIGINAL_CALL
        _ORIGINAL_CALL = None
    for cls, call in _ORIGINAL_SHARDED_CALLS.items():
        cls.__call__ = call
    _ORIGINAL_SHARDED_CALLS.clear()
