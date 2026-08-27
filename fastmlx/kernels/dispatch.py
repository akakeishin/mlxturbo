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


def _routes(overrides: dict[int, str] | None = None) -> tuple[str, ...]:
    """Build an M-indexed route row; indices outside 6..16 are stock.

    基本形: M=6..8 MMA (fast_qmm)、M=9..11 nocap、12+ stock。形状ごとの
    上書きは 2026-08-27 依存チェーン較正による (bench/results/
    calib-chain-a/b.json、2 ラン勝者一致 + 両ランで現経路比 5% 超の行のみ
    反転)。B ステージングのベクトル化と _zpad キャッシュ後の fast_qmm は
    M=6..16 のほぼ全域で nocap/stock に勝つ (勝ち幅 1.06-1.59x)。
    単発レイテンシではなく依存チェーンで判定している (BRIEF の規律)。
    M=5 は 2026-08-27 の追加 2 ラン (bench/results/smallm-a/b.json) で
    up/gate・q・lm_head の 3 形状が MMA 勝ち (5.6-8.8%)。down は 4.0-5.0% で
    バー未達につき stock 維持。M=2..4 は stock 維持 (qmv が優位)。
    """

    row = [STOCK] * 17
    for m in range(6, 9):
        row[m] = MMA
    for m in range(9, 12):
        row[m] = NOCAP
    if overrides:
        for m, route in overrides.items():
            row[m] = route
    return tuple(row)


# Explicit candidate table for the real model shapes recorded by
# bench/op_curve.py.  A3's GPU gate compares every selected entry with both
# alternatives before acceptance.
DEFAULT_ROUTE_TABLE: dict[tuple[int, int], tuple[str, ...]] = {
    (5120, 17408): _routes({5: MMA} | {m: MMA for m in range(9, 17)}),   # MLP up/gate
    # M=9 は 2 ラン較正で勝者が割れたため nocap (基本形) を維持
    (17408, 5120): _routes({m: MMA for m in range(10, 17)}),  # MLP down
    (5120, 12288): _routes({5: MMA} | {m: MMA for m in range(9, 17)}),   # attention q
    (5120, 248320): _routes({5: MMA} | {m: MMA for m in range(9, 17)}),  # lm_head
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


_KERNELS = None


def _load_kernels():
    # quantized_matmul は呼び出しごとにここを通るので、import と closure
    # 生成を初回だけにする (単発レイテンシ計測で wrapper 税として観測された)
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import os

    # fast_qmm (MIT、ライセンス確認済み 2026-08-26) が依存チェーン実測で
    # m=8: 1.57x と自作 v3/v4/v5 (1.16-1.18x) を上回るため、MMA 経路の実体は
    # fast_qmm にする。wide (m=9..16) も有効化。適格外は内部で stock へ落ちる。
    os.environ.setdefault("MLXLM_FAST_QMM_WIDE", "1")
    from ..fast_qmm import fast_qmm
    from .qmv_wide_nocap import qmv_wide_nocap

    def mma(flat, w, scales, biases, *, group_size, bits):
        return fast_qmm(flat, w, scales, biases, group_size=group_size, bits=bits)

    _KERNELS = (qmv_wide_nocap, mma)
    return _KERNELS


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
