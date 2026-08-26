"""Clean-room 8x8 MMA kernel for skinny affine-4bit quantized matmul.

The implementation follows only ``docs/PLAN.md`` Phase A2 and the public
Metal SIMD-group matrix contract.  It does not copy or depend on
``fastmlx.fast_qmm``.
"""

from typing import Any

from ._qmm_skinny_mma_source import (
    BITS,
    GROUP_SIZE,
    M_MAX,
    M_MIN,
    SPLIT_K,
    build_source,
    eligible_layout,
)

_KERNELS: dict[tuple[int, bool], Any] = {}
_METAL_HEADER = "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"


def _load_mx():
    import mlx.core as mx

    return mx


def _eligible(mx, x, w, scales, biases, group_size: int, bits: int) -> bool:
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    if x.dtype not in (mx.float16, mx.bfloat16):
        return False
    if w.dtype != mx.uint32 or scales.dtype != x.dtype or biases.dtype != x.dtype:
        return False
    if x.ndim != 2 or w.ndim != 2 or scales.ndim != 2 or biases.ndim != 2:
        return False
    m, k = x.shape
    n = w.shape[0]
    return eligible_layout(
        m,
        k,
        n,
        tuple(w.shape),
        tuple(scales.shape),
        tuple(biases.shape),
        group_size,
        bits,
    )


def _get_kernel(mx, m: int, fp16_input: bool):
    key = (m, fp16_input)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"fastmlx_qmm_skinny_mma_m{m}_{'f16' if fp16_input else 'bf16'}",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=build_source(m, fp16_input=fp16_input),
            header=_METAL_HEADER,
        )
        _KERNELS[key] = kernel
    return kernel


def qmm_skinny_mma(
    x,
    w,
    scales,
    biases,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
):
    """Compute ``x[M,K] @ dequant(w[N,K]).T`` or use MLX stock fallback."""

    mx = _load_mx()
    if not _eligible(mx, x, w, scales, biases, group_size, bits):
        return mx.quantized_matmul(
            x,
            w,
            scales,
            biases,
            transpose=True,
            group_size=group_size,
            bits=bits,
        )

    m, k = x.shape
    n = w.shape[0]
    kernel = _get_kernel(mx, m, x.dtype == mx.float16)
    (y,) = kernel(
        inputs=[x, w, scales, biases],
        template=[("T", x.dtype), ("K", k), ("N", n)],
        grid=(32, SPLIT_K, n // 8),
        threadgroup=(32, SPLIT_K, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    return y


__all__ = ["M_MIN", "M_MAX", "qmm_skinny_mma"]
