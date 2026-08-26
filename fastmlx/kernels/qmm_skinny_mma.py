# Ported from Layr-Labs/qwen-3.8-mtp-challenge Qwen35.swift E120 QMV.
# Copyright (c) 2026 Layr Labs, Inc. Licensed under the MIT License; see
# tools/reference/e120/LICENSE.
"""E120 register-only QMV for skinny BF16 affine-4/group-64 matmul."""

from typing import Any

from ._qmm_skinny_mma_source import (
    BITS,
    GROUP_SIZE,
    METAL_HEADER,
    M_MAX,
    M_MIN,
    THREADGROUP,
    active_input_groups,
    build_source,
    eligible_layout,
)

_KERNEL: Any | None = None


def _load_mx():
    import mlx.core as mx

    return mx


def _eligible(mx, x, w, scales, biases, group_size: int, bits: int) -> bool:
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    # E120's arithmetic and vector loads are explicitly bfloat16_t.  Other
    # dtypes remain on the exact stock path rather than instantiating a variant.
    if x.dtype != mx.bfloat16:
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


def _get_kernel(mx):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="fastmlx_e120_affine4_g64_qmv_v3",
            input_names=["w", "scales", "biases", "x"],
            output_names=["y"],
            source=build_source(),
            header=METAL_HEADER,
            ensure_row_contiguous=True,
        )
    return _KERNEL


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

    m = x.shape[0]
    n = w.shape[0]
    kernel = _get_kernel(mx)
    (y,) = kernel(
        inputs=[w, scales, biases, x],
        grid=(active_input_groups(m) * 32, (n // 8) * 2, 1),
        threadgroup=THREADGROUP,
        output_shapes=[(m, n)],
        output_dtypes=[mx.bfloat16],
    )
    return y


__all__ = ["M_MIN", "M_MAX", "qmm_skinny_mma"]
