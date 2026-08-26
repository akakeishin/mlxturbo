"""Shape-by-M routing for affine quantized linear layers.

Only shapes observed in the target Qwen3.8 model are eligible for custom
kernels.  Unknown shapes and unsupported quantization modes always retain the
MLX implementation.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from math import prod
from typing import Any

STOCK = "stock"
NOCAP = "nocap"
MMA = "mma"
_ROUTES = frozenset((STOCK, NOCAP, MMA))


def _routes(*, mma_max: int = 16) -> tuple[str, ...]:
    """Build an M-indexed route row; indices outside 6..16 are stock.

    2026-08-26 の依存チェーン実測 (docs/GATE-RESULTS-A2.md) で MMA v2 は
    m=8 で 0.82x、m=16 で 0.99x と全域で最速にならなかったため、
    現行表は nocap (bit-exact、m=6..12 で 1.0-1.1x) を採り、MMA は
    経路として残すが v3 が実測で勝つまで表からは外す。
    """

    del mma_max
    row = [STOCK] * 17
    # 2026-08-26 静音窓の全点実測 (bench/results/dispatch-a3-quiet.json):
    # 全 4 shape で一貫して M=6..9 は mma (v3)、M=10..11 は nocap、12+ は stock。
    for m in range(6, 10):
        row[m] = MMA
    for m in range(10, 12):
        row[m] = NOCAP
    return tuple(row)


# Explicit candidate table for the real model shapes recorded by
# bench/op_curve.py.  The wide lm_head retains stock for M=13..16 because its
# output dimension makes the second MMA tile a distinct regime.  A3's GPU gate
# compares every selected entry with both alternatives before acceptance.
DEFAULT_ROUTE_TABLE: dict[tuple[int, int], tuple[str, ...]] = {
    (5120, 17408): _routes(),       # MLP up/gate
    (17408, 5120): _routes(),       # MLP down
    (5120, 12288): _routes(),       # attention q
    (5120, 248320): _routes(mma_max=12),  # lm_head
}

_DISPATCHED_CLASS = None
_DISPATCH_ACTIVE = ContextVar("fastmlx_quantized_dispatch_active", default=False)


def select_route(
    k: int,
    n: int,
    m: int,
    table: dict[tuple[int, int], tuple[str, ...]] | None = None,
) -> str:
    """Return ``stock``, ``nocap``, or ``mma`` for one flattened matmul."""

    route_table = DEFAULT_ROUTE_TABLE if table is None else table
    row = route_table.get((k, n))
    if row is None or m < 0 or m >= len(row):
        return STOCK
    route = row[m]
    if route not in _ROUTES:
        raise ValueError(f"invalid quantized-matmul route {route!r}")
    return route


def _load_mx():
    import mlx.core as mx

    return mx


def _load_nn():
    import mlx.nn as nn

    return nn


def _load_kernels():
    from .qmm_skinny_mma import qmm_skinny_mma
    from .qmv_wide_nocap import qmv_wide_nocap

    return qmv_wide_nocap, qmm_skinny_mma


def quantized_matmul(
    x,
    w,
    scales,
    biases,
    *,
    group_size: int,
    bits: int,
    mode: str = "affine",
    table: dict[tuple[int, int], tuple[str, ...]] | None = None,
):
    """Dispatch a QuantizedLinear matmul while preserving stock fallback."""

    mx = _load_mx()

    def stock():
        return mx.quantized_matmul(
            x,
            w,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

    if mode != "affine" or biases is None or x.ndim < 2 or w.ndim != 2:
        return stock()

    k = x.shape[-1]
    n = w.shape[0]
    m = prod(x.shape[:-1])
    route = select_route(k, n, m, table)
    if route == STOCK:
        return stock()

    flat = x if x.ndim == 2 else x.reshape((m, k))
    nocap, mma = _load_kernels()
    if route == NOCAP:
        out = nocap(flat, w, scales, biases, group_size=group_size, bits=bits)
    else:
        out = mma(flat, w, scales, biases, group_size=group_size, bits=bits)
    if x.ndim == 2:
        return out
    return out.reshape((*x.shape[:-1], n))


def _get_dispatched_class():
    global _DISPATCHED_CLASS
    if _DISPATCHED_CLASS is None:
        nn = _load_nn()

        class DispatchedQuantizedLinear(nn.QuantizedLinear):
            """QuantizedLinear with fastmlx's shape-by-M matmul routing."""

            def __call__(self, x):
                active = self._fastmlx_dispatch_always or _DISPATCH_ACTIVE.get()
                out = quantized_matmul(
                    x,
                    self["weight"],
                    self["scales"],
                    self.get("biases"),
                    group_size=self.group_size,
                    bits=self.bits,
                    mode=self.mode,
                    table=None if active else {},
                )
                if "bias" in self:
                    out = out + self["bias"]
                return out

        DispatchedQuantizedLinear.__name__ = "DispatchedQuantizedLinear"
        DispatchedQuantizedLinear.__module__ = __name__
        _DISPATCHED_CLASS = DispatchedQuantizedLinear
    return _DISPATCHED_CLASS


@contextmanager
def dispatch_scope():
    """Temporarily activate verification-only dispatched layers."""

    token = _DISPATCH_ACTIVE.set(True)
    try:
        yield
    finally:
        _DISPATCH_ACTIVE.reset(token)


def enable(model: Any, *, active: bool = True):
    """Enable dispatch in-place for every QuantizedLinear below ``model``.

    Changing only ``__class__`` preserves module identity, parameter names,
    frozen state, and all loaded arrays.  Repeated calls are idempotent.
    ``active=False`` installs verification-only routing for use with
    :func:`dispatch_scope`.
    """

    nn = _load_nn()
    dispatched = _get_dispatched_class()
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear) and not isinstance(
            module, dispatched
        ):
            module.__class__ = dispatched
        if isinstance(module, dispatched):
            module._fastmlx_dispatch_always = active
    return model


__all__ = [
    "DEFAULT_ROUTE_TABLE",
    "MMA",
    "NOCAP",
    "STOCK",
    "dispatch_scope",
    "enable",
    "quantized_matmul",
    "select_route",
]
