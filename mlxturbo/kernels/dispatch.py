"""Shape-by-M routing for affine quantized linear layers.

Only shapes observed in the target Qwen3.8 model are eligible for custom
kernels.  Unknown shapes and unsupported quantization modes always retain the
MLX implementation.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from math import prod
from typing import Any

STOCK = "stock"
NOCAP = "nocap"
MMA = "mma"
SMALLM = "small_m"
_ROUTES = frozenset((STOCK, NOCAP, MMA, SMALLM))


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
_SMALL_M_KERNEL = None


def _load_kernels():
    # quantized_matmul は呼び出しごとにここを通るので、import と closure
    # 生成を初回だけにする (単発レイテンシ計測で wrapper 税として観測された)
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS

    # fast_qmm (MIT、ライセンス確認済み 2026-08-26) が依存チェーン実測で
    # m=8: 1.57x と自作 v3/v4/v5 (1.16-1.18x) を上回るため、MMA 経路の実体は
    # fast_qmm にする。wide (m=9..16) も有効化。適格外は内部で stock へ落ちる。
    # `MLXLM_FAST_QMM_WIDE` は fast_qmm.py 側で「既定 off」と読める公開の
    # 環境変数なので、ここで os.environ.setdefault してプロセス全体の既定を
    # 横取りしない -- fast_qmm() の呼び出し単位の `force_wide=True` で、この
    # MMA 経路の呼び出しだけ wide を有効化する (A2、Opus 設計レビュー指摘)。
    # 以前はモジュールグローバルなフラグを立てて済ませていたが、それだと
    # ここで一度立てるだけでプロセス全体の fast_qmm() 呼び出し (例:
    # fast_qmm.enable() が全 QuantizedLinear に配る側) にまで wide が
    # 伝染していた (D-3)。
    from ..fast_qmm import fast_qmm
    from .qmv_wide_nocap import qmv_wide_nocap

    def mma(flat, w, scales, biases, *, group_size, bits):
        return fast_qmm(
            flat, w, scales, biases, group_size=group_size, bits=bits,
            force_wide=True,
        )

    _KERNELS = (qmv_wide_nocap, mma)
    return _KERNELS


def _load_small_m():
    """小 M 経路 (`SMALLM`) の実体。

    **`_load_kernels()` の戻り値には混ぜない。**あちらは
    `bench/test_dispatch_static.py` が `(nocap, mma)` の 2 タプルに差し替えて
    経路の振り分けだけを検査する口で、要素を増やすと偽物側が壊れる
    (2026-09-04 に実際に落とした)。既定 off の経路をわざわざ既定の口に
    背負わせない。
    """
    global _SMALL_M_KERNEL
    if _SMALL_M_KERNEL is not None:
        return _SMALL_M_KERNEL

    from .qmv_small_m import qmv_small_m

    def small_m(flat, w, scales, biases, *, group_size, bits):
        return qmv_small_m(flat, w, scales, biases,
                           group_size=group_size, bits=bits)

    _SMALL_M_KERNEL = small_m
    return _SMALL_M_KERNEL


# --------------------------------------------------------------------------
# 検証幅 M=2..5 の経路 (MLXTURBO_SMALL_M_ROUTE、既定 auto = 非 NAX で small_m、2026-09-04)
# --------------------------------------------------------------------------
#
# 上の `DEFAULT_ROUTE_TABLE` は M=6..8 を MMA、M=9..11 を nocap にしてあり、
# **M=2..5 は stock のまま**。理由は 2026-08-27 の較正 (「M=2..4 は stock 維持
# (qmv が優位)」) だが、あれは Flash-Next の形での依存チェーンの判定だった。
#
# 27B (qwen3_5) の投機デコードでは検証幅 S の 98% が 3/4/5 に入る (S=4 65%、
# S=3 24%、S=5 9%) うえ、実機の内訳 (`tools/verify_width_cost_27b.py`、
# `bench/results/width-cost-27b-0904.json`) で **超線形の出所がこの帯の
# 量子化 dense 行列積そのもの**だと分かった:
#
#   mlp_all (64 層の MLP): M=1 25.8 / M=2 30.8 / M=3 44.1 / M=4 44.4 ms
#   行列積でない部品 (conv, rms, layernorm) は M に対して平坦
#
# ここはその帯だけを別のカーネルに回す口。**既定は auto** (値が空・auto なら
# 非 NAX 機で `small_m`、NAX 機では素のまま。自前カーネルは NAX 機で auto=off の
# 方針)。`0` / `off` / `stock` で切り、`small_m` / `nocap` / `mma` で経路を固定する。
# 2026-09-04 の 27B in-model (512 × 短 3 本 × 2 回文 + 4k、`bench/results/
# ab-smallm-3way-{short,4k}-0904.json`): small_m 短 -1.9% / 4k -4.6%、nocap 短 -1.9%
# / 4k +0.8%、mma 短 +3.1% / 4k +5.2%。fp32 参照への距離は small_m / nocap が素と
# 同じ (RMS 相対 1.65e-3)、mma は素の 1.3〜1.7 倍遠い → 既定は small_m。
# 形は `_SMALL_M_N_MIN` / `_SMALL_M_K_MIN` 以上のものだけ (GDN の
# in_proj_a/b のような N=48 の射影は自前カーネルの並列度に届かない)。
#
# 数値: `small_m` は **行ごとに幅 1 の素とビット一致**する (mlx の qmv_fast の
# 縮約順そのもの。`bench/test_qmm_smallm.py`)。`nocap` は mlx の `qmv_wide` の
# 移植で M<=5 なら素とビット一致する (mlx 側も 1 タイル = 同じ縮約順)。`mma` は
# MMA の別の縮約順なのでビット一致しない -- そちらを採るなら tok/round と KLD で
# 判定すること。
_SMALL_M_ENV = "MLXTURBO_SMALL_M_ROUTE"
_SMALL_M_MIN = 2
_SMALL_M_MAX = 5
_SMALL_M_N_MIN = 1024
_SMALL_M_K_MIN = 1024
_SMALL_M_ROUTE: str | None = None       # 現在有効な上書き (None = 素のまま)
_SMALL_M_ROWS: dict[tuple[int, int], tuple[str, ...]] = {}
_MMA_M_MIN = 5                          # `mlxturbo/fast_qmm.py` の M_MIN


def refresh_small_m_route() -> str | None:
    """`MLXTURBO_SMALL_M_ROUTE` を読み直す。

    行列積 1 本ごとに `os.environ` を引くと 1 round で 500 回になるので、
    読むのは `dispatch_scope()` の入口 (検証フォワードと lm_head で 1 回ずつ)
    だけにする。1 プロセス内 A/B (`tools/decode_ab_generic.py`) は round を
    またいで env を振るので、これで追従できる。
    """
    global _SMALL_M_ROUTE, _SMALL_M_ROWS

    val = os.environ.get(_SMALL_M_ENV, "").strip().lower()
    if val in ("", "auto"):
        from .moe_grouped_gemm import is_nax_device
        val = None if is_nax_device() else SMALLM
    elif val in ("0", "off", "stock"):
        val = None
    elif val not in (NOCAP, MMA, SMALLM):
        raise ValueError(
            f"{_SMALL_M_ENV}={val!r} は不明 ("
            f"auto / {SMALLM!r} / {NOCAP!r} / {MMA!r} / off のいずれか)")
    if val != _SMALL_M_ROUTE:
        _SMALL_M_ROUTE = val
        _SMALL_M_ROWS = {}
    return _SMALL_M_ROUTE


def _small_m_row(k: int, n: int) -> tuple[str, ...] | None:
    """小 M の上書き込みの経路行 (上書きが無い形なら None)。"""

    row = _SMALL_M_ROWS.get((k, n))
    if row is not None:
        return row
    if n < _SMALL_M_N_MIN or k < _SMALL_M_K_MIN:
        return None
    base = DEFAULT_ROUTE_TABLE.get((k, n))
    new = list(base) if base is not None else [STOCK] * 17
    for mm in range(_SMALL_M_MIN, _SMALL_M_MAX + 1):
        new[mm] = _SMALL_M_ROUTE
    row = tuple(new)
    _SMALL_M_ROWS[(k, n)] = row
    return row


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
    route = None
    if (
        _SMALL_M_ROUTE is not None
        and table is None
        and _SMALL_M_MIN <= m <= _SMALL_M_MAX
    ):
        row = _small_m_row(k, n)
        if row is not None:
            route = row[m]
    if route is None:
        route = select_route(k, n, m, table)
    if route == STOCK:
        return stock()

    flat = x if x.ndim == 2 else x.reshape((m, k))
    if route == SMALLM:
        # 行ごとに幅 1 の素とビット一致する (kernels/qmv_small_m.py)。
        # 形が外れたら関数の中で `mx.quantized_matmul` に落ちる。
        out = _load_small_m()(flat, w, scales, biases,
                              group_size=group_size, bits=bits)
        return out if x.ndim == 2 else out.reshape((*x.shape[:-1], n))

    nocap, mma = _load_kernels()
    if route == NOCAP:
        # 既定の窓は M=6..12。小 M の上書きのときだけ下限を開ける
        # (カーネルは 1 タイル = M 全体なので M=2..5 でも正しく動く)。
        out = nocap(flat, w, scales, biases, group_size=group_size, bits=bits,
                    m_min=_SMALL_M_MIN)
    elif m < _MMA_M_MIN:
        # fast_qmm の窓は M=5..16。M=2..4 をそのまま渡すと**黙って stock に
        # 落ちる**ので、5 行に 0 詰めしてから渡して戻りを切る。カーネルは
        # 内部でどのみち 8 行の MMA タイルに詰めるので費用は M=5 と同じ
        # (`mlxturbo/fast_qmm.py` は Flash-Next と共有なので触らない)。
        pad = mx.zeros((_MMA_M_MIN - m, k), dtype=flat.dtype)
        out = mma(mx.concatenate([flat, pad], axis=0), w, scales, biases,
                  group_size=group_size, bits=bits)[:m]
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
            """QuantizedLinear with mlxturbo's shape-by-M matmul routing."""

            def __call__(self, x):
                active = self._fastmlx_dispatch_always or _DISPATCH_ACTIVE.get()
                if not active:
                    # 非活性 (`dispatch_scope()` の外) では基底クラスに委ねる。
                    # ここで自前の「素と同じ写し」を通すと、基底の
                    # `nn.QuantizedLinear.__call__` に当たっている差し替え
                    # (`fused.enable_qmm_wide` / `enable_hc_qmm_wide` の
                    # `_qmm_wide_dispatch`) を丸ごとシャドーしてしまう。
                    # `SpecEngine.__init__` が `enable(..., active=False)` で
                    # 全射影のクラスを差し替えるので、27B では prefill/decode の
                    # 全部がこの枝を通る (2026-09-04 の回帰: 印は 368 本
                    # 付いているのに `_fire` の `qmm_wide_*` が 0 だった)。
                    return super().__call__(x)
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

    refresh_small_m_route()
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
    "SMALLM",
    "STOCK",
    "dispatch_scope",
    "enable",
    "quantized_matmul",
    "refresh_small_m_route",
    "select_route",
]
