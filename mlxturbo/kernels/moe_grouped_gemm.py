"""MoE の grouped GEMM を自前 Metal カーネルで書くレーン (P3)。

設計は `docs/research/IDEAS-2026-09-03.md` の P3。狙いは prefill の
`mx.gather_qmm(sorted_indices=True)` が払っている二重の損:

  1. タイル内に専門家境界が来ると、そのタイルは**セグメントごとに
     フル BM x BN x K をやり直す** (`quantized.h:2517` のループ)
  2. 非 NAX 機の gather 経路のタイルは BM=16 / simdgroup 2 本
     (`quantized.cpp:1652`) で、dense (BM=32 / 4 本) より逆量子化の
     償却が悪い

どちらもタイルを専門家境界で切れば消える。ただし自前 simdgroup_matrix を
一から書いた前例 (A2 v5) が目標に届かなかったので、**MMA は steel の丸写し**
で行く (`_steel_flat.py`)。

## 第 1 段 (このファイルの現状)

セグメントに入る前に、写した steel が素の dense と同じ速さで回ることを
確かめる。:func:`qmm_dense_clone` は `mx.quantized_matmul(transpose=True)`
の非 NAX 経路 (`quantized.cpp:1058-1065` の bm=bn=32 / wm=wn=2 と
`quantized.h:1197-1199` の BK=32) と同じタイル・同じ frag 順・同じ累算順を
使う。ここが**ビット一致 + 1.10 倍以内**で通らなければ、
`mx.fast.metal_kernel` 経路の固定費が勝ち目を消しているということなので、
第 2 段には進まない (反転条件、IDEAS の P3)。

計測は `tools/moe_grouped_gemm_micro.py --stage dense`。

## NAX 機 (M5 以降) の扱い

カーネル本体は NAX 専用の intrinsic を **1 つも使わない**。写した範囲は
steel の `simdgroup_matrix` 経路だけで、`nax.h` / `gemm_nax.h` /
`quantized_nax.h` は入れていない (`_steel_flat.py` の生成時に検査している)。
したがって NAX 機でも同じコードがそのまま動く。

ただし NAX 機では MLX 自身が別物のカーネル (`gather_qmm_rhs_nax`、
bm 32/64 + simdgroup 単位の `sg_active` スキップ) を持っていて、相手の形が
こちらと違う。**NAX 機での A/B はまだ取っていない** (この開発機は
applegpu_g15s = gen 15 で NAX 非対応)。**M5 系が手に入ったら
`tools/decode_ab.py` の knob として配線した上で取り直すこと**
(配線は第 2 段で入れる)。それまでは
既定 (`auto`) で NAX 機だけ off にしておく --- 未計測のものを既定で
入れないため。`MLXTURBO_MOE_GEMM=on` にすれば NAX 機でも強制的に使える。

機種の判定は :func:`is_nax_device` の 1 箇所だけ。MLX 本体
(`metal/device.cpp:947-965`) と同じく architecture 文字列の世代 (末尾 3 文字
の手前 2 桁) で見る。MLX 側はこれに macOS 26.2 以降の条件も掛けているので、
古い OS の M5 では MLX が非 NAX 経路に落ちる一方こちらは `auto` で off に
なる。取りこぼす側なので害は無い。
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

from . import _fire
from ._steel_flat import STEEL_HEADER

# steel の dense qmm_t と同じタイル (quantized.cpp:1062-1065 / quantized.h:1197)
BM = 32
BN = 32
BK = 32
WM = 2
WN = 2
THREADS = WM * WN * 32  # 1 threadgroup 128 スレッド

# 本番の重みの量子化 (QuantizedSwitchLinear と同じ)
GROUP_SIZE = 64
BITS = 4

_KERNELS: dict[tuple, Any] = {}

# 発火の 3 値 knob。`auto` は NAX 機だけ off (上の「NAX 機の扱い」を参照)
ENV_KNOB = "MLXTURBO_MOE_GEMM"


def is_nax_device() -> bool:
    """この機が MLX の NAX カーネルを持つ世代か。**機種判定はここだけ。**

    MLX 本体 (`metal/device.cpp:594-601, 947-965`) と同じ読み方:
    architecture 文字列 (`applegpu_g15s` など) の末尾から 3 文字目と 2 文字目を
    世代の 10 の位・1 の位として読み、末尾が `p` (phone) なら 18、それ以外は
    17 以上を NAX 世代とする。
    """

    try:
        info = getattr(mx, "device_info", None) or mx.metal.device_info
        arch = str(info().get("architecture", ""))
    except Exception:
        return False
    if len(arch) < 3 or not arch[-3:-1].isdigit():
        return False
    gen = int(arch[-3:-1])
    return gen >= (18 if arch[-1] == "p" else 17)


def enabled() -> bool:
    """`MLXTURBO_MOE_GEMM` (auto|on|off) を解く。既定は `auto`。"""

    mode = os.environ.get(ENV_KNOB, "auto").strip().lower()
    if mode in ("on", "1", "true", "yes"):
        return True
    if mode in ("off", "0", "false", "no"):
        return False
    # auto: 非 NAX 機だけ on。NAX 機は A/B が未実施なので既定では入れない
    return not is_nax_device()


# `qmm_t_impl` をそのまま呼ぶだけの本体。threadgroup メモリの宣言と
# 次元の受け渡し以外は書かない (写しの外側に自分の算数を持ち込まない)。
_DENSE_SOURCE = """
  constexpr int BM = %(bm)d;
  constexpr int BN = %(bn)d;
  constexpr int BK = %(bk)d;
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  // dims = (K, N, M)。3 要素なので constant アドレス空間に載り、
  // 素の `const constant int&` と同じ渡し方になる
  qmm_t_impl<T, GROUP_SIZE, BITS, ALIGNED_N, BM, BK, BN>(
      w,
      scales,
      biases,
      x,
      y,
      Xs,
      Ws,
      dims[0],
      dims[1],
      dims[2],
      dims[0],
      threadgroup_position_in_grid,
      thread_index_in_threadgroup,
      simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup);
""" % {"bm": BM, "bn": BN, "bk": BK}


def _get_dense_kernel():
    kernel = _KERNELS.get("dense")
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="mlxturbo_steel_qmm_t_clone",
            input_names=["w", "scales", "biases", "x", "dims"],
            output_names=["y"],
            source=_DENSE_SOURCE,
            header=STEEL_HEADER,
            ensure_row_contiguous=True,
        )
        _KERNELS["dense"] = kernel
    return kernel


def dense_eligible(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
) -> bool:
    """:func:`qmm_dense_clone` が素とビット一致する形か。"""

    if group_size != GROUP_SIZE or bits != BITS:
        return False
    if x.ndim != 2 or w.ndim != 2 or scales.ndim != 2 or biases.ndim != 2:
        return False
    if x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if scales.dtype != x.dtype or biases.dtype != x.dtype:
        return False
    if w.dtype != mx.uint32:
        return False
    K = x.shape[1]
    N = w.shape[0]
    if K % BK != 0 or K % group_size != 0:
        return False
    if w.shape[1] != K * bits // 32:
        return False
    groups = K // group_size
    return scales.shape == (N, groups) and biases.shape == (N, groups)


def qmm_dense_clone(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
) -> mx.array:
    """`mx.quantized_matmul(x, w, scales, biases, transpose=True)` の写し。

    ``x`` は (M, K)、``w`` は (N, K*bits/32) の packed uint32、
    ``scales``/``biases`` は (N, K/group_size)。戻り値は (M, N)。

    素との違いは起動の仕組み (``dispatch_threads`` と
    ``dispatch_threadgroups``) だけで、タイル・ローダー・MMA・store は
    `_steel_flat.STEEL_HEADER` の写しをそのまま呼んでいる。
    """

    _fire.bump("moe_grouped_gemm_dense")
    M, K = x.shape
    N = w.shape[0]
    aligned_n = (N % BN) == 0

    dims = mx.array([K, N, M], dtype=mx.int32)
    kernel = _get_dense_kernel()
    (out,) = kernel(
        inputs=[w, scales, biases, x, dims],
        template=[
            ("T", x.dtype),
            ("GROUP_SIZE", group_size),
            ("BITS", bits),
            ("ALIGNED_N", aligned_n),
        ],
        # 素は dispatch_threadgroups((N/BN, M/BM, 1), (32, WN, WM))。
        # custom kernel は dispatch_threads なので x に 128 を掛ける
        grid=(THREADS * ((N + BN - 1) // BN), (M + BM - 1) // BM, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )
    return out


def qmm_segmented(*args, **kwargs):
    """第 2 段 (専門家セグメント対応) の入り口。まだ中身が無い。

    ここに来るのは第 1 段 (:func:`qmm_dense_clone`) がビット一致と
    1.10 倍以内を通ってから。入る形は
    ``(x_sorted, w, scales, biases, tile_prefix, row_start)`` で、
    カーネル側が tile id から (専門家, 行ブロック) を 2 分探索する。
    """

    raise NotImplementedError(
        "moe_grouped_gemm の第 2 段 (セグメント対応) は未実装"
    )


__all__ = [
    "ENV_KNOB",
    "dense_eligible",
    "enabled",
    "is_nax_device",
    "qmm_dense_clone",
    "qmm_segmented",
]
