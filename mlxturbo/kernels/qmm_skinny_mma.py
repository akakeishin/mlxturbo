"""Skinny BF16 quantized matmul kernels with explicit implementations."""

from typing import Any

from . import _qmm_e120_source as _e120
from ._qmm_skinny_mma_source import (
    BITS,
    GROUP_SIZE,
    METAL_HEADER,
    M_MAX,
    M_MIN,
    SPLIT_K,
    THREADGROUP,
    build_source,
    eligible_layout,
)

V5_IMPLEMENTATION = "v5"
E120_V4_IMPLEMENTATION = "e120_v4"
_IMPLEMENTATIONS = frozenset((V5_IMPLEMENTATION, E120_V4_IMPLEMENTATION))

_V5_KERNELS: dict[int, Any] = {}
_E120_KERNEL: Any | None = None
_E120_TABLE_KERNEL: Any | None = None
_E120_XSUMS_KERNEL: Any | None = None


def _load_mx():
    import mlx.core as mx

    return mx


def _stock(mx, x, w, scales, biases, group_size: int, bits: int):
    return mx.quantized_matmul(
        x,
        w,
        scales,
        biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )


def _common_eligible(mx, x, w, scales, biases) -> bool:
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    if x.dtype != mx.bfloat16:
        return False
    if (
        w.dtype != mx.uint32
        or scales.dtype != x.dtype
        or biases.dtype != x.dtype
    ):
        return False
    return x.ndim == w.ndim == scales.ndim == biases.ndim == 2


def _v5_eligible(mx, x, w, scales, biases, group_size: int, bits: int) -> bool:
    if not _common_eligible(mx, x, w, scales, biases):
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


def _e120_eligible(mx, x, w, scales, biases, group_size: int, bits: int) -> bool:
    if not _common_eligible(mx, x, w, scales, biases):
        return False
    m, k = x.shape
    n = w.shape[0]
    return _e120.eligible_layout(
        m,
        k,
        n,
        tuple(w.shape),
        tuple(scales.shape),
        tuple(biases.shape),
        group_size,
        bits,
    )


def _get_v5_kernel(mx, m: int):
    kernel = _V5_KERNELS.get(m)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"fastmlx_qmm_skinny_mma_v5_m{m}_bf16",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=build_source(m),
            header=METAL_HEADER,
            ensure_row_contiguous=True,
        )
        _V5_KERNELS[m] = kernel
    return kernel


def _get_e120_kernel(mx):
    global _E120_KERNEL
    if _E120_KERNEL is None:
        _E120_KERNEL = mx.fast.metal_kernel(
            name="fastmlx_e120_affine4_g64_qmv_v3",
            input_names=["w", "scales", "biases", "x"],
            output_names=["y"],
            source=_e120.build_source(),
            header=_e120.METAL_HEADER,
            ensure_row_contiguous=True,
        )
    return _E120_KERNEL


def _get_e120_table_kernel(mx):
    global _E120_TABLE_KERNEL
    if _E120_TABLE_KERNEL is None:
        _E120_TABLE_KERNEL = mx.fast.metal_kernel(
            name="fastmlx_e120_affine4_g64_qmv_table_v4",
            input_names=["w", "scales", "biases", "x", "xsums"],
            output_names=["y"],
            source=_e120.build_table_source(),
            header=_e120.TABLE_METAL_HEADER,
            ensure_row_contiguous=True,
        )
    return _E120_TABLE_KERNEL


def _get_e120_xsums_kernel(mx):
    global _E120_XSUMS_KERNEL
    if _E120_XSUMS_KERNEL is None:
        _E120_XSUMS_KERNEL = mx.fast.metal_kernel(
            name="fastmlx_e120_affine4_g64_xsums_v4",
            input_names=["x"],
            output_names=["xsums"],
            source=_e120.XSUMS_SOURCE,
            ensure_row_contiguous=True,
        )
    return _E120_XSUMS_KERNEL


def _run_v5(mx, x, w, scales, biases, group_size: int, bits: int):
    if not _v5_eligible(mx, x, w, scales, biases, group_size, bits):
        return _stock(mx, x, w, scales, biases, group_size, bits)
    m, _ = x.shape
    n = w.shape[0]
    kernel = _get_v5_kernel(mx, m)
    (y,) = kernel(
        inputs=[x, w, scales, biases],
        grid=(32, SPLIT_K, n // 8),
        threadgroup=THREADGROUP,
        output_shapes=[(m, n)],
        output_dtypes=[mx.bfloat16],
    )
    return y


def _run_e120_v4(
    mx,
    x,
    w,
    scales,
    biases,
    group_size: int,
    bits: int,
    use_table: bool | None,
):
    if not _e120_eligible(mx, x, w, scales, biases, group_size, bits):
        return _stock(mx, x, w, scales, biases, group_size, bits)
    m, k = x.shape
    n = w.shape[0]
    table_enabled = m >= _e120.MINIMUM_TABLE_WIDTH and use_table is not False
    inputs = [w, scales, biases, x]
    if table_enabled:
        xsums_kernel = _get_e120_xsums_kernel(mx)
        (xsums,) = xsums_kernel(
            inputs=[x],
            grid=(32, k // _e120.BLOCK_SIZE, m),
            threadgroup=_e120.XSUMS_THREADGROUP,
            output_shapes=[
                (k // _e120.BLOCK_SIZE * 32 * _e120.sums_stride(m),)
            ],
            output_dtypes=[mx.float32],
        )
        inputs.append(xsums)
        kernel = _get_e120_table_kernel(mx)
    else:
        kernel = _get_e120_kernel(mx)
    (y,) = kernel(
        inputs=inputs,
        grid=(
            _e120.active_input_groups(m) * 32,
            (n // 8) * 2,
            1,
        ),
        threadgroup=_e120.THREADGROUP,
        output_shapes=[(m, n)],
        output_dtypes=[mx.bfloat16],
    )
    return y


def qmm_skinny_mma(
    x,
    w,
    scales,
    biases,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
    *,
    implementation: str = V5_IMPLEMENTATION,
    use_table: bool | None = None,
):
    """Compute ``x[M,K] @ dequant(w[N,K]).T`` or use stock fallback.

    ``v5`` is the direct-call default and accepts BF16 M=6..16. ``e120_v4``
    accepts compatible affine q4/group64 layouts at M=2..9; production use is
    selected independently by the measured shape table in ``dispatch.py``.
    Its ``use_table`` switch retains the table/no-table diagnostic path.
    """

    if implementation not in _IMPLEMENTATIONS:
        choices = ", ".join(sorted(_IMPLEMENTATIONS))
        raise ValueError(
            f"implementation must be one of {choices}, got {implementation!r}"
        )
    if implementation == V5_IMPLEMENTATION and use_table is not None:
        raise ValueError("use_table is only valid with implementation='e120_v4'")
    mx = _load_mx()
    if implementation == E120_V4_IMPLEMENTATION:
        return _run_e120_v4(
            mx,
            x,
            w,
            scales,
            biases,
            group_size,
            bits,
            use_table,
        )
    return _run_v5(mx, x, w, scales, biases, group_size, bits)


__all__ = [
    "E120_V4_IMPLEMENTATION",
    "M_MAX",
    "M_MIN",
    "V5_IMPLEMENTATION",
    "qmm_skinny_mma",
]
