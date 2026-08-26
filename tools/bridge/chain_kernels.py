"""正しさテストと計測で共有する「連鎖カーネル」の定義。

同じ計算 (y = x * 2 + 1 を N 回) を 2 経路で走らせるためのもの:
  (a) mx.fast.metal_kernel を N 回呼ぶ  -> MLX のラッパを毎回通る
  (b) fastmlx bridge で N dispatch を 1 command buffer に積む

閉形式は y_N = 2^N * x + (2^N - 1)。x を 0.25 刻みの小さい値に取り、
N <= 16 に抑えれば float32 で厳密に一致する (最大 327679 < 2^24)。
"""

from __future__ import annotations

import struct

import mlx.core as mx

# --- (b) bridge が自前で MTLLibrary にコンパイルする MSL ---------------------
BRIDGE_MSL = r"""
#include <metal_stdlib>
using namespace metal;

kernel void chain_affine(device const float* inp [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& n [[buffer(2)]],
                         uint elem [[thread_position_in_grid]]) {
    if (elem >= n) { return; }
    out[elem] = inp[elem] * 2.0f + 1.0f;
}

// ラッパ税だけを見るための空カーネル。docs/HYPOTHESES-A2.md の H2 probe
// (0.064 ms/call) と同じ形。バッファは束縛だけして触らない。
kernel void chain_noop(device const float* inp [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       uint elem [[thread_position_in_grid]]) {
}
"""


def bridge_constants(n_elem: int) -> bytes:
    return struct.pack("<I", n_elem)


# --- (a) MLX 側 -------------------------------------------------------------

_mlx_cache: dict[tuple[str, int], object] = {}


def mlx_chain_kernel(n_elem: int, noop: bool = False):
    """mx.fast.metal_kernel 版。source は上の MSL と同じ本体。"""
    key = ("noop" if noop else "affine", n_elem)
    if key in _mlx_cache:
        return _mlx_cache[key]
    if noop:
        source = f"""
            uint elem = thread_position_in_grid.x;
            if (elem >= {n_elem}u) {{ return; }}
        """
        name = "fmb_chain_noop"
    else:
        source = f"""
            uint elem = thread_position_in_grid.x;
            if (elem >= {n_elem}u) {{ return; }}
            out[elem] = inp[elem] * 2.0f + 1.0f;
        """
        name = "fmb_chain_affine"
    kernel = mx.fast.metal_kernel(
        name=name,
        input_names=["inp"],
        output_names=["out"],
        source=source,
    )
    _mlx_cache[key] = kernel
    return kernel


def mlx_chain(x: mx.array, steps: int, threadgroup: int = 256, noop: bool = False):
    """(a) 経路: metal_kernel を steps 回チェーンする。"""
    n = x.size
    kernel = mlx_chain_kernel(n, noop=noop)
    y = x
    for _ in range(steps):
        y = kernel(
            inputs=[y],
            grid=(n, 1, 1),
            threadgroup=(threadgroup, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]
    return y


def closed_form(x, steps: int):
    """y_N = 2^N * x + (2^N - 1)。numpy 配列を受けて numpy を返す。"""
    f = float(2**steps)
    return x * f + (f - 1.0)
