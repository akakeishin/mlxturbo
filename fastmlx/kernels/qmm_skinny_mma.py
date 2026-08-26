# Ported from Layr-Labs/qwen-3.8-mtp-challenge Qwen35.swift E120 QMV.
# Copyright (c) 2026 Layr Labs, Inc. Licensed under the MIT License; see
# tools/reference/e120/LICENSE.
"""E120 register-only QMV for skinny BF16 affine-4/group-64 matmul."""

from typing import Any

from ._qmm_skinny_mma_source import (
    BITS,
    BLOCK_SIZE,
    GROUP_SIZE,
    METAL_HEADER,
    M_MAX,
    M_MIN,
    MINIMUM_TABLE_WIDTH,
    TABLE_METAL_HEADER,
    THREADGROUP,
    XSUMS_SOURCE,
    XSUMS_THREADGROUP,
    active_input_groups,
    build_source,
    build_table_source,
    eligible_layout,
    sums_stride,
)

_KERNEL: Any | None = None
_TABLE_KERNEL: Any | None = None
_XSUMS_KERNEL: Any | None = None


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


def _get_table_kernel(mx):
    global _TABLE_KERNEL
    if _TABLE_KERNEL is None:
        _TABLE_KERNEL = mx.fast.metal_kernel(
            name="fastmlx_e120_affine4_g64_qmv_table_v4",
            input_names=["w", "scales", "biases", "x", "xsums"],
            output_names=["y"],
            source=build_table_source(),
            header=TABLE_METAL_HEADER,
            ensure_row_contiguous=True,
        )
    return _TABLE_KERNEL


def _get_xsums_kernel(mx):
    global _XSUMS_KERNEL
    if _XSUMS_KERNEL is None:
        _XSUMS_KERNEL = mx.fast.metal_kernel(
            name="fastmlx_e120_affine4_g64_xsums_v4",
            input_names=["x"],
            output_names=["xsums"],
            source=XSUMS_SOURCE,
            ensure_row_contiguous=True,
        )
    return _XSUMS_KERNEL


def qmm_skinny_mma(
    x,
    w,
    scales,
    biases,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
    *,
    use_table: bool | None = None,
):
    """Compute ``x[M,K] @ dequant(w[N,K]).T`` or use MLX stock fallback.

    The E120 xsums table is selected by default for M >= 4.  ``use_table`` is
    an A/B gate for bit-exact GPU validation; requesting it below M=4 keeps the
    reference no-table path.
    """

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
    table_enabled = m >= MINIMUM_TABLE_WIDTH and use_table is not False
    inputs = [w, scales, biases, x]
    if table_enabled:
        xsums_kernel = _get_xsums_kernel(mx)
        (xsums,) = xsums_kernel(
            inputs=[x],
            grid=(32, k // BLOCK_SIZE, m),
            threadgroup=XSUMS_THREADGROUP,
            output_shapes=[(k // BLOCK_SIZE * 32 * sums_stride(m),)],
            output_dtypes=[mx.float32],
        )
        inputs.append(xsums)
        kernel = _get_table_kernel(mx)
    else:
        kernel = _get_kernel(mx)
    (y,) = kernel(
        inputs=inputs,
        grid=(active_input_groups(m) * 32, (n // 8) * 2, 1),
        threadgroup=THREADGROUP,
        output_shapes=[(m, n)],
        output_dtypes=[mx.bfloat16],
    )
    return y


__all__ = ["M_MIN", "M_MAX", "MINIMUM_TABLE_WIDTH", "qmm_skinny_mma"]
