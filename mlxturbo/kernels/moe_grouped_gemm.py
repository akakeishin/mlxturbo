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
結果: gate/up 1.000 倍、down 1.006 倍、どちらもビット一致
(`bench/results/moe-grouped-gemm-dense.json`)。

## 第 2 段 (:func:`qmm_segmented`)

:func:`qmm_dense_clone` のタイルを**専門家境界で切る**。素の
`mx.gather_qmm(sorted_indices=True)` は BM=16 のタイルを行の並びだけで切る
ので、境界をまたいだタイルはセグメントごとにフル BM x BN x K をやり直す
(`quantized.h:2517`)。こちらはタイル 1 枚が必ず 1 専門家に収まるので
やり直しが無く、タイルの形も dense と同じ BM=32 / 128 スレッドになる。

タイルから (専門家, 行ブロック) を引くのに host 同期は要らない。専門家ごとの
行数 `counts` から

  row_start[e]   = counts[:e].sum()                 (行の先頭)
  tile_prefix[e] = ceil(counts[:e] / BM).sum()      (タイルの通し番号)

を mx op で作り、カーネルが自分の threadgroup id を `tile_prefix` の中で
2 分探索する。grid は静的上限 (`_n_tiles_max`) で張って、余った
threadgroup は即 return する。

端数タイル (行数 t < BM) は
  - X をそのまま読む (次の専門家の行を読むだけで、結果は store しない)。
    バッファ末尾に掛かるタイルだけ `load_safe` に落とす
  - store は `store_result_safe`
の 2 つで畳む。W の逆量子化は端数でもフル幅ぶん (BN x BK) 要る。

**MMA の frag 単位の間引き (`mlxturbo_mma_rows`、`frag_skip=True`) は
計測で負けたので既定 off。** 20 行タイルの費用 (32 行タイル比) は
間引き無しで 0.975 / 0.980 (gate/up / down)、間引き有りで 1.078 / 1.103。
削れる MMA は 4 frag 中 1 枚 (25%) だけなのに、分岐が入ると
simdgroup_multiply_accumulate 4 本の並びが崩れて 10% 損する
(`bench/results/moe-grouped-gemm-segmented.json`)。

つまり**端数タイルの費用 c はほぼ 1** で、時間はタイル枚数でほぼ決まる:

    seg / dense ~= Σ_e ceil(rows_e / 32) * 32 / M  x (1 + 0.015)

実測 (r=40 で水増し 1.375 -> 1.391、r=160 で 1.085 -> 1.095、
flat20 で 1.600 -> 1.602) がこの式に 1 % 以内で乗る。静的上限で余らせた
threadgroup (flat20 では 832 枚中 320 枚が空振り) は事実上ただ。

計測は `tools/moe_grouped_gemm_micro.py --stage segmented`。

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

# 第 2 段の変種で使う小さいタイル。既製の非 NAX gather_qmm_rhs と同じ形
# (quantized.cpp:1652 の bm=16 / wm=1 / wn=2 = 64 スレッド)
BM16 = 16
WM16 = 1

# 混合モードで「小さい専門家」と見なす行数の既定の境目。48 前後を掃引した
# 結果はレーンの記録に残す (micro の `--stage segmented --mix-sweep`)
MIX_THRESHOLD = 48

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


# ---------------------------------------------------------------------------
# 第 2 段: 専門家セグメント対応
# ---------------------------------------------------------------------------

# steel の写しに足す 2 つだけ。ここより下の算数は全部写しに任せる。
_SEGMENTED_HEADER = r"""
// ---- mlxturbo: 専門家セグメント対応 (moe_grouped_gemm 第 2 段) ----

// prefix[0..E] (prefix[0] == 0、単調非減少) の中で prefix[e] <= t を満たす
// 最大の e を返す。E = 512 なら 9 段。threadgroup 内の全スレッドが同じ
// 探索をするが、読むのは 2 KB の同じ配列なのでキャッシュに載る。
METAL_FUNC int mlxturbo_seg_search(const device int* prefix, int E, int t) {
  int lo = 0;
  int hi = E - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (prefix[mid] <= t) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

// `mlx::steel::BlockMMA::mma` (mma.h) の写し。違いは末尾の tile_matmad を
// 展開して「この frag が受け持つ 8 行が丸ごと row_lim の外なら simdgroup の
// MMA ごと飛ばす」条件を 1 つ足したところだけ。累算の順序 (kk -> m -> n の
// serpentine -> 1 回の simdgroup_multiply_accumulate) は写しのまま。
//
// frag (m, *) が触る行は tm + m*TM_stride .. +8 で、tm は
// simdgroup_index_in_threadgroup だけで決まる = simdgroup 内で一様。
// simdgroup_multiply_accumulate は simdgroup 全体の命令なので、条件は
// simdgroup 一様でなければならない (行ごとに落とすのは不可)。飛ばした frag の
// 累算器は 0 のままだが、その 8 行は store されないので出力に出ない。
template <typename T, typename mma_t, int BK, int WM, int WN>
METAL_FUNC void mlxturbo_mma_rows(
    thread mma_t& mma_op,
    const threadgroup T* As,
    const threadgroup T* Bs,
    const short tm,
    const short row_lim) {
  using frag_t = typename mma_t::MMAFrag_acc_t;

  As += mma_op.As_offset;
  Bs += mma_op.Bs_offset;

  STEEL_PRAGMA_UNROLL
  for (short kk = 0; kk < BK; kk += mma_t::kFragSize) {
    simdgroup_barrier(mem_flags::mem_none);

    mma_op.Atile.template load<T, WM, 1, mma_t::A_str_m, mma_t::A_str_k>(As);

    simdgroup_barrier(mem_flags::mem_none);

    mma_op.Btile.template load<T, 1, WN, mma_t::B_str_k, mma_t::B_str_n>(Bs);

    simdgroup_barrier(mem_flags::mem_none);

    STEEL_PRAGMA_UNROLL
    for (short m = 0; m < mma_t::TM; ++m) {
      if ((tm + m * mma_t::TM_stride) < row_lim) {
        STEEL_PRAGMA_UNROLL
        for (short n = 0; n < mma_t::TN; ++n) {
          short n_serp = (m % 2) ? (mma_t::TN - 1 - n) : n;
          frag_t::mma(
              mma_op.Ctile.frag_at(m, n_serp),
              mma_op.Atile.frag_at(m, 0),
              mma_op.Btile.frag_at(0, n_serp),
              mma_op.Ctile.frag_at(m, n_serp));
        }
      }
    }

    As += mma_t::tile_stride_a;
    Bs += mma_t::tile_stride_b;
  }
}

// タイル 1 枚 (専門家 e の行 rs..re、列 y_col..y_col+BN) を計算して store する。
// BM と WM を template で受けるので、16 行 (WM=1、64 スレッド) と 32 行
// (WM=2 の既定、または混合モードの WM=1) を同じ本文で回せる。中身は
// もともと本文に直書きしてあったものをそのまま関数に移しただけ。
template <
    typename T,
    int TBM,
    int TBN,
    int TBK,
    int TWM,
    int TWN,
    int TGS,
    int TBITS,
    bool TSKIP>
METAL_FUNC void mlxturbo_seg_tile(
    const device T* x,
    const device uint8_t* w8,
    const device T* scales,
    const device T* biases,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int M,
    const int e,
    const int rs,
    const int re,
    const int y_col,
    const ushort simd_group_id,
    const ushort simd_lane_id) {
  constexpr short BK_padded = (TBK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, TBM, TBN, TBK, TWM, TWN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, TBM, TBK, BK_padded, 1, TWM * TWN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      TBN,
      TBK,
      BK_padded,
      1,
      TWM * TWN * SIMD_SIZE,
      TGS,
      TBITS>;

  constexpr int pack_factor = get_pack_factor<TBITS, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<TBITS>();

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / TGS;
  const int K_it = K / TBK;
  const short num_els = short(re - rs);

  const size_t stride_w = size_t(N) * size_t(K_w);
  const size_t stride_s = size_t(N) * size_t(K_g);

  const device T* xp = x + size_t(rs) * size_t(K);
  device T* yp = y + size_t(rs) * size_t(N) + size_t(y_col);
  const device uint8_t* wl =
      w8 + size_t(e) * stride_w + size_t(y_col) * size_t(K_w);
  const device T* sp =
      scales + size_t(e) * stride_s + size_t(y_col) * size_t(K_g);
  const device T* bp =
      biases + size_t(e) * stride_s + size_t(y_col) * size_t(K_g);

  loader_x_t loader_x(xp, K, Xs, simd_group_id, simd_lane_id);
  loader_w_t loader_w(wl, sp, bp, K, Ws, simd_group_id, simd_lane_id);
  mma_t mma_op(simd_group_id, simd_lane_id);

  if (num_els == TBM) {
    // 満タンのタイル。dense クローンの内側そのもの
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.store_result(yp, N);
    return;
  }

  // 端数タイル。X のはみ出しぶんは「次の専門家の行」なので、x の末尾に
  // 掛かっていない限りそのまま読んでよい (読んだ結果は store しない)。
  const short tm = short(mma_t::kFragSize * (simd_group_id / TWN));
  if (rs + TBM <= M) {
    for (int k = 0; k < K_it; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (TSKIP) {
        mlxturbo_mma_rows<T, mma_t, TBK, TWM, TWN>(mma_op, Xs, Ws, tm, num_els);
      } else {
        mma_op.mma(Xs, Ws);
      }
      loader_x.next();
      loader_w.next();
    }
  } else {
    // x の最後のタイルだけ。全 dispatch で高々 1 枚
    for (int k = 0; k < K_it; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(short(TBK), num_els));
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (TSKIP) {
        mlxturbo_mma_rows<T, mma_t, TBK, TWM, TWN>(mma_op, Xs, Ws, tm, num_els);
      } else {
        mma_op.mma(Xs, Ws);
      }
      loader_x.next();
      loader_w.next();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mma_op.store_result_safe(yp, N, short2(short(TBN), num_els));
}
"""

# 本体。タイルの引き当てと端数の分岐だけを書き、ローダー・MMA・store は
# 写しをそのまま呼ぶ (dense クローンと同じ方針)。
# 本体。タイルの引き当てだけを書き、ローダー・MMA・store は写しをそのまま
# 呼ぶ (dense クローンと同じ方針)。TILE_M / TILE_WM は template 引数なので、
# 32 行 (WM=2、128 スレッド) と 16 行 (WM=1、64 スレッド) が同じ本文で回る。
_SEGMENTED_HEAD = """
  constexpr int BN = %(bn)d;
  constexpr int BK = %(bk)d;
  constexpr int WN = %(wn)d;
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  // 混合モードでは同じ dispatch に 32 行タイルも来るので、threadgroup は
  // いつも 32 行ぶんで張る (16 行タイルは前半しか使わない)
  threadgroup T Xs[%(bm_max)d * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  // dims = (K, N, M, E, MIX_THRESHOLD)。5 要素なので constant アドレス空間
  const int K = dims[0];
  const int N = dims[1];
  const int M = dims[2];
  const int E = dims[3];

  // 自分のタイル。grid は静的上限で張ってあるので余りが来る
  const int t = int(threadgroup_position_in_grid.y);
  if (t >= tile_prefix[E]) {
    return;
  }

  // タイル -> (専門家、行ブロック)。tile_prefix は counts から作った
  // 「専門家 e までのタイル数」の累積和なので、2 分探索 1 回で引ける
  const int e = mlxturbo_seg_search(tile_prefix, E, t);
  const int y_col = int(threadgroup_position_in_grid.x) * BN;
  const device uint8_t* w8 = (const device uint8_t*)w;
""" % {"bn": BN, "bk": BK, "wn": WN, "bm_max": BM}

_SEGMENTED_SOURCE = _SEGMENTED_HEAD + """
  const int rs = row_start[e] + (t - tile_prefix[e]) * TILE_M;
  const int re = min(row_start[e + 1], rs + TILE_M);
  mlxturbo_seg_tile<T, TILE_M, BN, BK, TILE_WM, WN, GROUP_SIZE, BITS, SKIP_ROWS>(
      x, w8, scales, biases, y, Xs, Ws, K, N, M, e, rs, re, y_col,
      simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

# 混合モード。専門家の行数が閾値未満なら 16 行タイル、それ以外は 32 行タイル。
# threadgroup は 64 スレッドで、32 行タイルも WM=1 (TM=4) で回す --- 1 回の
# dispatch に 2 種のタイルを混ぜるには threadgroup の形を揃えるしかないため。
# 32 行 / WM=1 が 32 行 / WM=2 (既定) より遅いなら、その差は micro の
# `bm32w1` ケース (混合の対照) に出る。
_SEGMENTED_MIXED_SOURCE = _SEGMENTED_HEAD + """
  const int MIX_THRESHOLD = dims[4];
  const int rows_e = row_start[e + 1] - row_start[e];
  if (rows_e < MIX_THRESHOLD) {
    const int rs = row_start[e] + (t - tile_prefix[e]) * %(bm16)d;
    const int re = min(row_start[e + 1], rs + %(bm16)d);
    mlxturbo_seg_tile<T, %(bm16)d, BN, BK, 1, WN, GROUP_SIZE, BITS, SKIP_ROWS>(
        x, w8, scales, biases, y, Xs, Ws, K, N, M, e, rs, re, y_col,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
  } else {
    const int rs = row_start[e] + (t - tile_prefix[e]) * %(bm)d;
    const int re = min(row_start[e + 1], rs + %(bm)d);
    mlxturbo_seg_tile<T, %(bm)d, BN, BK, 1, WN, GROUP_SIZE, BITS, SKIP_ROWS>(
        x, w8, scales, biases, y, Xs, Ws, K, N, M, e, rs, re, y_col,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
  }
""" % {"bm16": BM16, "bm": BM}


def _get_segmented_kernel(mixed: bool = False):
    key = "segmented_mixed" if mixed else "segmented"
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="mlxturbo_steel_qmm_seg_mixed" if mixed else
            "mlxturbo_steel_qmm_segmented",
            input_names=[
                "x",
                "w",
                "scales",
                "biases",
                "row_start",
                "tile_prefix",
                "dims",
            ],
            output_names=["y"],
            source=_SEGMENTED_MIXED_SOURCE if mixed else _SEGMENTED_SOURCE,
            header=STEEL_HEADER + _SEGMENTED_HEADER,
            ensure_row_contiguous=True,
        )
        _KERNELS[key] = kernel
    return kernel


def counts_from_sorted_ids(ids: mx.array, num_experts: int) -> mx.array:
    """専門家添字 (ソート済みでなくてもよい) から専門家ごとの行数を数える。

    host 同期はしない。返るのは (num_experts,) の int32。
    """

    flat = ids.reshape(-1).astype(mx.int32)
    zeros = mx.zeros((num_experts,), dtype=mx.int32)
    return zeros.at[flat].add(mx.ones(flat.shape, dtype=mx.int32))


def segment_tables(
    counts: mx.array,
    bm: int = BM,
    mix_threshold: int | None = None,
) -> tuple[mx.array, mx.array]:
    """`counts` (E,) から ``(row_start, tile_prefix)`` を作る。どちらも (E+1,)。

    ``row_start[e]`` は専門家 e の行の先頭、``tile_prefix[e]`` は専門家 e より
    前のタイル数の合計。カーネルはこの 2 本だけで tile -> (専門家, 行ブロック)
    を引く。**host 同期は入らない** (`counts` の中身を Python 側で見ない)。

    ``mix_threshold`` を渡すと混合モードの表になる: 行数がその値未満の
    専門家は 16 行タイル、それ以外は 32 行タイルで数える。カーネル側も
    同じ比較 (`rows_e < MIX_THRESHOLD`) をするので、両者は必ず一致する。
    """

    counts = counts.astype(mx.int32)
    zero = mx.zeros((1,), dtype=mx.int32)
    row_start = mx.concatenate([zero, mx.cumsum(counts)])
    if mix_threshold is None:
        tiles = (counts + (bm - 1)) // bm
    else:
        bm_e = mx.where(counts < mix_threshold, BM16, BM)
        tiles = (counts + bm_e - 1) // bm_e
    tile_prefix = mx.concatenate([zero, mx.cumsum(tiles)])
    return row_start, tile_prefix


def n_tiles_max(num_rows: int, num_experts: int, bm: int = BM) -> int:
    """タイル数の静的上限。grid をこれで張る。

    Σ_e ceil(c_e / bm) <= Σ_e floor(c_e / bm) + (行を持つ専門家の数)
                       <= floor(M / bm) + min(E, M)

    混合モードは専門家ごとに 16 か 32 なので、最悪 (全部 16) の
    ``bm=BM16`` で張る。
    """

    return num_rows // bm + min(num_experts, num_rows)


def segmented_eligible(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
) -> bool:
    """:func:`qmm_segmented` が `mx.gather_qmm` とビット一致する形か。

    N と K を BN / BK / group_size で割り切れる形に限る (本番の
    2560 -> 640 と 640 -> 2560 はどちらも通る)。端数の N・K を許すと
    ローダー側の分岐が 2 倍になるうえ、本番に無い経路の面倒を見ることに
    なるので入れていない。
    """

    if group_size != GROUP_SIZE or bits != BITS:
        return False
    if x.ndim != 2 or w.ndim != 3 or scales.ndim != 3 or biases.ndim != 3:
        return False
    if x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if scales.dtype != x.dtype or biases.dtype != x.dtype:
        return False
    if w.dtype != mx.uint32:
        return False
    K = x.shape[1]
    E, N = w.shape[0], w.shape[1]
    # row_start / tile_prefix は E+1 要素。MLX は 8 要素未満の入力を
    # constant アドレス空間で渡す (metal_kernel.cpp:18) ので、そこを
    # 跨ぐと `const device int*` を取るカーネル側と食い違う
    if E < 8:
        return False
    if N % BN != 0:
        return False
    if K % BK != 0 or K % group_size != 0:
        return False
    if w.shape[2] != K * bits // 32:
        return False
    groups = K // group_size
    return scales.shape == (E, N, groups) and biases.shape == (E, N, groups)


def qmm_segmented(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    segments: mx.array,
    *,
    num_experts: int | None = None,
    segments_are: str = "auto",
    tables: tuple[mx.array, mx.array] | None = None,
    frag_skip: bool = False,
    bm: int = BM,
    wm: int | None = None,
    mix_threshold: int | None = None,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
) -> mx.array:
    """`mx.gather_qmm(..., transpose=True, sorted_indices=True)` の写し。

    ``x`` は (M, K) の**専門家順にソート済み**の行、``w`` は
    (E, N, K*bits/32) の packed uint32、``scales``/``biases`` は
    (E, N, K/group_size)。戻り値は (M, N) でソート順のまま。

    ``segments`` は専門家ごとの行数 (E,) か、ソート済みの専門家添字 (M,)。
    どちらかは ``segments_are`` (``"auto"`` / ``"counts"`` / ``"ids"``) で
    決める。``auto`` は要素数で見て、M と E が同じときは ids を採る。
    ``tables`` に :func:`segment_tables` の結果を渡せば作り直さない
    (gate / up / down の 3 本で使い回す用)。

    ``frag_skip`` は端数タイルで MMA を frag (8 行) 単位に間引くか。
    どちらでも出力は同じ (間引くのは元々 store しない行だけ)。**既定 off --
    計測で 10 % 負けたから** (上の「第 2 段」を参照)。残してあるのは
    別の機種で取り直せるようにするためで、既定を戻すなら
    `--stage segmented` の seg / fragskip を測り直してからにすること。

    ``bm`` はタイルの行数 (16 か 32、既定 32)。``wm`` を渡すと simdgroup の
    行方向の本数を上書きできる (既定は bm=32 で 2、bm=16 で 1)。
    ``mix_threshold`` を渡すと**専門家ごとに** 16 行タイルと 32 行タイルを
    選ぶ混合モードになり、``bm`` / ``wm`` は無視される (混合は 1 dispatch に
    2 種のタイルを混ぜるので threadgroup を 64 スレッドに揃える必要があり、
    WM は必ず 1)。``tables`` を渡すときは同じ ``bm`` /
    ``mix_threshold`` で作った表であること。
    """

    _fire.bump("moe_grouped_gemm_segmented")
    M, K = x.shape
    E, N = w.shape[0], w.shape[1]
    if num_experts is not None and num_experts != E:
        raise ValueError(f"num_experts={num_experts} が w の {E} と合わない")

    if tables is None:
        kind = segments_are
        if kind == "auto":
            if segments.size == M:
                kind = "ids"
            elif segments.size == E:
                kind = "counts"
            else:
                raise ValueError(
                    f"segments の要素数 {segments.size} が M={M} とも E={E} とも違う"
                )
        if kind == "ids":
            counts = counts_from_sorted_ids(segments, E)
        elif kind == "counts":
            counts = segments
        else:
            raise ValueError(f"segments_are={segments_are!r} が不正")
        tables = segment_tables(counts, bm=bm, mix_threshold=mix_threshold)
    row_start, tile_prefix = tables

    mixed = mix_threshold is not None
    if mixed:
        tile_wm = 1
        grid_bm = BM16  # 最悪 (全専門家が 16 行タイル) で grid を張る
    else:
        if bm not in (BM16, BM):
            raise ValueError(f"bm={bm} は 16 か 32 のみ")
        tile_wm = (WM16 if bm == BM16 else WM) if wm is None else wm
        grid_bm = bm
    threads = tile_wm * WN * 32

    dims = mx.array(
        [K, N, M, E, int(mix_threshold or 0)], dtype=mx.int32
    )
    kernel = _get_segmented_kernel(mixed)
    template = [
        ("T", x.dtype),
        ("GROUP_SIZE", group_size),
        ("BITS", bits),
        ("SKIP_ROWS", bool(frag_skip)),
    ]
    if not mixed:
        template += [("TILE_M", int(bm)), ("TILE_WM", int(tile_wm))]
    (out,) = kernel(
        inputs=[x, w, scales, biases, row_start, tile_prefix, dims],
        template=template,
        grid=(threads * (N // BN), n_tiles_max(M, E, grid_bm), 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )
    return out


__all__ = [
    "ENV_KNOB",
    "counts_from_sorted_ids",
    "dense_eligible",
    "enabled",
    "is_nax_device",
    "qmm_dense_clone",
    "qmm_segmented",
    "segment_tables",
    "segmented_eligible",
]
