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

## 第 2 段の変種: BM=16 タイルと、専門家ごとの混合 (2026-09-03)

`qmm_segmented` の ``bm`` / ``wm`` / ``mix_threshold`` がこれ。既製の非 NAX
`gather_qmm_rhs` と同じ 16 行タイル (WM=1、64 スレッド) を足し、さらに
**専門家ごとに** 16 行と 32 行を選べるようにした (1 dispatch。閾値は
`MIX_THRESHOLD`)。全ケース `mx.gather_qmm` とビット一致 (80/80)。

micro (`--stage segmented`、M3 Max、MoE 層 1 つ = gate/up x2 + down、dense 比):

| ケース | r=40 ms (比) | r=160 ms (比) |
|---|---|---|
| stock (既製) | 27.02 (1.408) | 89.39 (1.163) |
| seg32 (BM=32 / WM=2、既定) | 26.30 (1.370) | 84.66 (1.101) |
| seg16 (BM=16 / WM=1) | 23.81 (1.241) | 85.43 (1.111) |
| seg32w1 (BM=32 / WM=1) | 25.61 (1.334) | 80.89 (1.052) |
| **mix48 (閾値 48)** | **23.69 (1.235)** | **80.12 (1.042)** |
| pad16 (既製 + ダミー行) | 23.85 (1.243) | 85.90 (1.117) |
| dense | 19.19 (1.000) | 76.87 (1.000) |

読み方 (取り分の出どころが 2 つある):

- **16 行タイル 1 枚の費用は 32 行タイルの 0.52 倍** (flat32 を bm16 で
  2 枚に割った実測: gate/up 0.528 / down 0.517)。1 行あたりでは 32 行タイルの
  1.05 倍なので、**行数の少ない専門家だけ** 16 行にするのが正しい。
  閾値は 40〜96 でほぼ平ら (r=40 で 23.69〜23.79)、最良は 48。24 / 32 は
  やや損 (24.04 / 24.24) で、これは「32 行タイルは 16 行タイル 1.93 枚ぶん」
  という費用比から出る損益分岐 (~48 行) と一致する。
- **WM=1 (64 スレッド、TM=4) が WM=2 (128 スレッド、TM=2) より速い。**
  同じ 32 行タイルで seg32 -> seg32w1 が r=40 -2.6% / r=160 -4.5%。r=160 の
  取り分はほぼ全部これで、16 行タイルの寄与は 1% 未満。**steel の既定
  (dense qmm と同じ WM=2) がこの機で最良とは限らない**という結果で、
  flat32 / flat160 では seg32w1 が `mx.quantized_matmul` の dense すら
  下回る (dense 比 0.951〜0.986)。dense クローン側の WM は測っていない。

pad16 (既製 + 16 行揃えのダミー行) との比較は、**同じ合成ルーティング・
同じプロセスで測った 23.85 ms が判定線**。`gather_qmm_pad_micro.py` の
22.61 ms は別のルーティング (Dirichlet、median 16 / zero 78) の数字で、
そちらは dense の床も 17.70 と違うので直接は比べられない。同一条件では
mix48 が pad16 を r=40 で -0.7%、r=160 で -6.7% 下回る。しかも pad16 は
行の水増し (r=40 で +17%) と**リクエストごとの host 同期 1 回**が要るのに対し、
混合は表の作り直し (`mx.where` 1 つ増えるだけ、同期なし) で済む。

計測は `tools/moe_grouped_gemm_micro.py --stage segmented --mix-thresholds
24,32,40,48,64,96` (interleave 1 本に stock / seg32 / seg16 / seg32w1 /
mix<t> / dense / pad16 を全部入れて、順方向と逆方向で 8 ラウンド)。

**まだ in-model では測っていない。**`fused.enable_moe_grouped_gemm` は
seg32 を呼ぶままで、既定 (`bm=32` / `wm=None` / `mix_threshold=None`) も
変えていない。in-model の見込みは prefill の MoE が 43% (8k の内訳) なので、
seg32 -> mix48 で prefill -2.3%、既製 -> mix48 で -4.5%。

## 第 3 段: 行列積以外を GEMM の口に畳む (P7 第 2 段、2026-09-03)

`qmm_segmented` の ``row_dst`` / ``row_scale`` / ``row_src`` がこれ。MoE の
prefill で残っていた「行列積以外」(`bench/results/moe-split.json`、層 20、
M=2048、ms/層) のうち 3 つを GEMM の store と loader に畳む:

  - ``row_dst`` + ``row_scale``: 行 s の結果を `row_dst[s]` の位置へ
    `row_scale[s]` 倍して書く。これで combine の weight_mul_cast (0.87) と
    unsort (`out[inv_order]` の gather) が消え、`inv_order` を作る 2 本目の
    argsort (sort 0.34 の半分) も要らなくなる。掛け算は累算器 (fp32) の
    ままなので、**丸めは現行より 1 回少ない** (ビット一致しない)
  - ``row_src``: 行 s が読む x を `x[row_src[s]]` にする。これで x の
    gather (0.49、16384x2560 = 84MB の実体化) が消える。読む値は同じなので
    「gather してから GEMM」と**ビット一致する**。micro で gate/up 2 本が
    11.35 -> 10.79 ms (-0.56 ms/層)

畳めないのは top_k 軸の和だけ (専門家をまたぐ行の和なのでタイル内では
取れず、atomic は非決定になる)。配線は `fused.enable_moe_down_epilogue`
(vendor `_moe_combine_fold` のフック 1 個)、A/B は
`tools/decode_ab.py --knob moe-down-epi`。

**既定に入ったのは ``row_src`` と `kernels/moe_combine.py` の組**
(in-model 8k で prefill -3.8%)。``row_dst`` / ``row_scale`` (epilogue) は
combine と同着 (micro -0.97 対 -0.98) だったので、down GEMM を触らない
combine のほうを採った。`mode="epi"` として残してある。

## NAX 機 (M5 以降) の扱い

カーネル本体は NAX 専用の intrinsic を **1 つも使わない**。写した範囲は
steel の `simdgroup_matrix` 経路だけで、`nax.h` / `gemm_nax.h` /
`quantized_nax.h` は入れていない (`_steel_flat.py` の生成時に検査している)。
したがって NAX 機でも同じコードがそのまま動く。

ただし NAX 機では MLX 自身が別物のカーネル (`gather_qmm_rhs_nax`、
bm 32/64 + simdgroup 単位の `sg_active` スキップ) を持っていて、相手の形が
こちらと違う。**NAX 機での A/B はまだ取っていない** (この開発機は
applegpu_g15s = gen 15 で NAX 非対応)。**M5 以降の実機が手に入ったら
`tools/decode_ab.py` の knob として配線した上で取り直すこと**
(配線は第 2 段で入れる)。それまでは
既定 (`auto`) で NAX 機だけ off にしておく --- 未計測のものを既定で
入れないため。`MLXTURBO_MOE_GEMM=on` にすれば NAX 機でも強制的に使える。

機種の判定は :func:`is_nax_device` の 1 箇所だけ。MLX 本体
(`metal/device.cpp:947-965`) と同じく architecture 文字列の世代 (末尾 3 文字
の手前 2 桁) で見る。M6 は M5 の後継として同じ保守側へ入る。MLX 側はこれに
macOS 26.2 以降の条件も掛けているので、古い OS の M5 では MLX が非 NAX 経路に
落ちる一方こちらは `auto` で off に
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


def apple_gpu_family() -> tuple[int, bool] | None:
    """Return ``(generation, is_phone)`` from MLX's architecture string."""

    try:
        info = getattr(mx, "device_info", None) or mx.metal.device_info
        arch = str(info().get("architecture", ""))
    except Exception:
        return None
    if len(arch) < 3 or not arch[-3:-1].isdigit():
        return None
    return int(arch[-3:-1]), arch[-1] == "p"


def is_nax_device() -> bool:
    """この機が MLX の NAX カーネルを持つ世代か。**機種判定はここだけ。**

    MLX 本体 (`metal/device.cpp:594-601, 947-965`) と同じ読み方:
    architecture 文字列 (`applegpu_g15s` など) の末尾から 3 文字目と 2 文字目を
    世代の 10 の位・1 の位として読み、末尾が `p` (phone) なら 18、それ以外は
    17 以上を NAX 世代とする。上限を置かないため、M6 以降も M5 と同じ
    保守側へ入る。
    """

    family = apple_gpu_family()
    if family is None:
        return False
    gen, is_phone = family
    return gen >= (18 if is_phone else 17)


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

// epilogue 版の store (P7 第 2 段)。`BlockMMA::store_result` の写しだが、
// 行を「タイル内の並び」ではなく `row_dst[行]` の位置へ書き (scatter)、
// 書く直前に `row_scale[行]` を掛ける。frag (i, j) が持つのは
// 行 sm + i*TM_stride、列 sn + j*TN_stride の 2 要素 (kElemsPerFrag=2、
// `_steel_flat.py` の BaseMMAFrag<T,8,8>) なので、行は frag ごとに 1 本しか
// 無く、行ごとの分岐で足りる (simdgroup 一様である必要も無い -- ここには
// simdgroup 命令が無い)。
//
// 端数タイルでは row_lim 未満の行だけ書く (store_result_safe と同じ条件)。
// 掛け算は累算器 (fp32) のまま行って最後に T へ丸めるので、素の
// 「bf16 に丸めてから掛ける」よりも丸めが 1 回少ない。
template <typename T, typename mma_t>
METAL_FUNC void mlxturbo_seg_store_scatter(
    thread mma_t& mma_op,
    device T* y,
    const int N,
    const int y_col,
    const device uint* row_dst,
    const device float* row_scale,
    const int rs,
    const short row_lim) {
  constexpr short kelems = mma_t::MMAFrag_acc_t::kElemsPerFrag;

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < mma_t::TM; i++) {
    const short r = mma_op.sm + i * mma_t::TM_stride;
    if (r >= row_lim) {
      continue;
    }
    device T* drow = y + size_t(row_dst[rs + r]) * size_t(N)
        + size_t(y_col) + size_t(mma_op.sn);
    const float s = row_scale[rs + r];
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < mma_t::TN; j++) {
      thread const auto& accum = mma_op.Ctile.frag_at(i, j);
      const int off = j * mma_t::TN_stride;
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < kelems; k++) {
        drow[off + k] = static_cast<T>(accum[k] * s);
      }
    }
  }
}

// タイル 1 枚 (専門家 e の行 rs..re、列 y_col..y_col+BN) を計算して store する。
// BM と WM を template で受けるので、16 行 (WM=1、64 スレッド) と 32 行
// (WM=2 の既定、または混合モードの WM=1) を同じ本文で回せる。中身は
// もともと本文に直書きしてあったものをそのまま関数に移しただけ。
//
// TEPI が true のときだけ store が scatter + 重み掛けになる (row_dst /
// row_scale を読む)。false の側は `nullptr` が渡り、分岐は template 定数
// なので消える -- 行列積とローダーは 2 つの経路で完全に同じ命令列。
template <
    typename T,
    int TBM,
    int TBN,
    int TBK,
    int TWM,
    int TWN,
    int TGS,
    int TBITS,
    bool TSKIP,
    bool TEPI = false,
    bool TSRC = false>
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
    const ushort simd_lane_id,
    const device uint* row_dst = nullptr,
    const device float* row_scale = nullptr,
    const device uint* row_src = nullptr) {
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

  // TSRC のときは x の行を `row_src` 経由で読む (MoE の x gather を GEMM に
  // 畳む)。BlockLoader はコンストラクタで `src + bi*src_ld + bj` を作るので、
  // ここで先に bi ぶんを引いておけば、あとは素の loader がそのまま使える。
  // bi はスレッド id と compile-time 定数だけで決まる
  // (`_steel_flat.py` の BlockLoader: TCOLS = BCOLS / n_reads、
  // bi = thread_idx / TCOLS)。本番の 3 構成はどれも 1 スレッド 1 行
  // (n_rows == 1) なので、行は 1 本引くだけで足りる。
  static_assert(!TSRC || loader_x_t::n_rows == 1,
                "TSRC は 1 スレッド 1 行の loader 構成でだけ正しい");
  const device T* xp;
  if (TSRC) {
    constexpr short xl_tcols = TBK / loader_x_t::vec_size;
    const short xl_bi =
        short(simd_group_id * SIMD_SIZE + simd_lane_id) / xl_tcols;
    // 端数タイルでは rs + xl_bi が M を超えうる。読んだ結果は store しない
    // ので、クランプして手前の行を読ませるだけでよい
    const int src_row = int(row_src[min(rs + int(xl_bi), M - 1)]);
    xp = x + (long(src_row) - long(xl_bi)) * long(K);
  } else {
    xp = x + size_t(rs) * size_t(K);
  }
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
    if (TEPI) {
      mlxturbo_seg_store_scatter<T, mma_t>(
          mma_op, y, N, y_col, row_dst, row_scale, rs, TBM);
    } else {
      mma_op.store_result(yp, N);
    }
    return;
  }

  // 端数タイル。X のはみ出しぶんは「次の専門家の行」なので、x の末尾に
  // 掛かっていない限りそのまま読んでよい (読んだ結果は store しない)。
  const short tm = short(mma_t::kFragSize * (simd_group_id / TWN));
  // TSRC のときは行の添字をクランプ済みなので、末尾のタイルでも x の外は
  // 読まない (load_safe に落とす必要が無い)
  if (TSRC || rs + TBM <= M) {
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
  if (TEPI) {
    mlxturbo_seg_store_scatter<T, mma_t>(
        mma_op, y, N, y_col, row_dst, row_scale, rs, num_els);
  } else {
    mma_op.store_result_safe(yp, N, short2(short(TBN), num_els));
  }
}
"""

# 本体。タイルの引き当てと端数の分岐だけを書き、ローダー・MMA・store は
# 写しをそのまま呼ぶ (dense クローンと同じ方針)。
# 本体。タイルの引き当てだけを書き、ローダー・MMA・store は写しをそのまま
# 呼ぶ (dense クローンと同じ方針)。TILE_M / TILE_WM は template 引数なので、
# 32 行 (WM=2、128 スレッド) と 16 行 (WM=1、64 スレッド) が同じ本文で回る。
#
# ``bn`` (列タイル幅) だけは呼び出しごとに変えられるようにしてある (P7 第 3 段
# の掃引用)。既定の 32 では生成される文字列が以前と 1 文字も変わらない
# (`_SEGMENTED_HEAD` 以下の 3 つの定数がそれを固定している)。
_SEG_HEAD_TMPL = """
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
"""


def _seg_head(bn: int = BN, bk: int = BK) -> str:
    return _SEG_HEAD_TMPL % {"bn": bn, "bk": bk, "wn": WN, "bm_max": BM}


_SEGMENTED_HEAD = _seg_head(BN)

_SEG_BODY_ONE = """
  const int rs = row_start[e] + (t - tile_prefix[e]) * TILE_M;
  const int re = min(row_start[e + 1], rs + TILE_M);
  mlxturbo_seg_tile<T, TILE_M, BN, BK, TILE_WM, WN, GROUP_SIZE, BITS, SKIP_ROWS>(
      x, w8, scales, biases, y, Xs, Ws, K, N, M, e, rs, re, y_col,
      simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_SEGMENTED_SOURCE = _SEGMENTED_HEAD + _SEG_BODY_ONE

# 混合モード。専門家の行数が閾値未満なら 16 行タイル、それ以外は 32 行タイル。
# threadgroup は 64 スレッドで、32 行タイルも WM=1 (TM=4) で回す --- 1 回の
# dispatch に 2 種のタイルを混ぜるには threadgroup の形を揃えるしかないため。
# 32 行 / WM=1 が 32 行 / WM=2 (既定) より遅いなら、その差は micro の
# `bm32w1` ケース (混合の対照) に出る。
_SEG_MIXED_BODY = """
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

_SEGMENTED_MIXED_SOURCE = _SEGMENTED_HEAD + _SEG_MIXED_BODY

# P7 第 2 段の 2 つの変種。本文は上の 2 つと同じで、`mlxturbo_seg_tile` の
# template 定数と末尾の引数だけが違う:
#
#   epilogue (TEPI): store を「`row_dst[行]` の位置へ `row_scale[行]` 倍して
#                    書く」に替える (MoE の combine を down GEMM に畳む)
#   src      (TSRC): x の行を `row_src[行]` 経由で読む (MoE の x gather を
#                    gate/up GEMM に畳む)。行数 M は row_src の長さ
#
# 素の 2 つ (`_SEGMENTED_SOURCE` / `_SEGMENTED_MIXED_SOURCE`) は 1 文字も
# 動かさない (既定の経路のコンパイル結果を変えないため)。
def _seg_variant_source(mixed: bool, epi: bool, src: bool,
                        bn: int = BN, bk: int = BK) -> str:
    tail = ", true" if epi else ", false"
    tail += ", true" if src else ""
    args = ",\n      row_dst, row_scale" if epi else ""
    if src:
        args = ",\n      " + ("row_dst, row_scale, " if epi else
                              "nullptr, nullptr, ") + "row_src"
    body_one = """
  const int rs = row_start[e] + (t - tile_prefix[e]) * %(bm)s;
  const int re = min(row_start[e + 1], rs + %(bm)s);
  mlxturbo_seg_tile<T, %(bm)s, BN, BK, %(wm)s, WN, GROUP_SIZE, BITS,
                    SKIP_ROWS%(tail)s>(
      x, w8, scales, biases, y, Xs, Ws, K, N, M, e, rs, re, y_col,
      simdgroup_index_in_threadgroup, thread_index_in_simdgroup%(args)s);
"""
    head_str = _seg_head(bn, bk)
    if not mixed:
        return head_str + body_one % {
            "bm": "TILE_M", "wm": "TILE_WM", "tail": tail, "args": args}
    head = """
  const int MIX_THRESHOLD = dims[4];
  const int rows_e = row_start[e + 1] - row_start[e];
  if (rows_e < MIX_THRESHOLD) {"""
    small = body_one % {"bm": str(BM16), "wm": "1", "tail": tail, "args": args}
    big = body_one % {"bm": str(BM), "wm": "1", "tail": tail, "args": args}
    return (head_str + head + small + "  } else {" + big + "  }\n")


def _get_segmented_kernel(mixed: bool = False, epilogue: bool = False,
                          src: bool = False, bn: int = BN, bk: int = BK):
    shaped = bn != BN or bk != BK
    key = "segmented" + ("_epi" if epilogue else "") + ("_src" if src else "") \
        + ("_mixed" if mixed else "") \
        + ("" if not shaped else f"_bn{bn}_bk{bk}")
    kernel = _KERNELS.get(key)
    if kernel is None:
        names = [
            "x",
            "w",
            "scales",
            "biases",
            "row_start",
            "tile_prefix",
            "dims",
        ]
        if epilogue:
            names += ["row_dst", "row_scale"]
        if src:
            names += ["row_src"]
        if epilogue or src:
            source = _seg_variant_source(mixed, epilogue, src, bn, bk)
        elif shaped:
            source = _seg_head(bn, bk) + (_SEG_MIXED_BODY if mixed
                                          else _SEG_BODY_ONE)
        else:
            source = _SEGMENTED_MIXED_SOURCE if mixed else _SEGMENTED_SOURCE
        if epilogue or src or shaped:
            name = "mlxturbo_steel_qmm_" + key
        else:
            name = "mlxturbo_steel_qmm_seg_mixed" if mixed else \
                "mlxturbo_steel_qmm_segmented"
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=names,
            output_names=["y"],
            source=source,
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
    row_dst: mx.array | None = None,
    row_scale: mx.array | None = None,
    row_src: mx.array | None = None,
    bn: int = BN,
    bk: int = BK,
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

    ``row_dst`` / ``row_scale`` (両方セットで渡す、P7 第 2 段の epilogue) を
    渡すと、行 r の結果を ``row_dst[r]`` の位置へ ``row_scale[r]`` 倍して
    書く。MoE の combine (unsort + ルータ重み掛け) を down GEMM の store に
    畳むためのもの。``row_dst`` は uint32 の (M,)、``row_scale`` は float32 の
    (M,)。**``row_dst`` は [0, M) の置換であること** -- 出力は初期化せずに
    タイルが書くだけなので、書かれない行があると未初期化のまま残る。
    掛け算は累算器 (fp32) のまま行い、丸めは store の 1 回だけ。

    ``row_src`` (uint32 の (M,)) を渡すと、GEMM の行 r が読むのは
    ``x[row_src[r]]`` になる (MoE の x gather を GEMM に畳む)。このとき
    ``x`` の行数は M と無関係でよく、**M は ``row_src`` の長さ**になる。

    ``bn`` は列タイルの幅 (既定 32 = steel の dense qmm_t と同じ)。64 に
    すると 1 threadgroup の受け持つ出力が倍になり、タイル 1 枚あたりの
    固定費 (dispatch と x タイルの読み) が半分の枚数に薄まる。行数の少ない
    専門家が多いとき (r=40) に効くかを見るための掃引用で、``N`` が ``bn`` で
    割り切れることが要る (`segmented_eligible` の判定も同じ ``bn`` で行う
    こと)。
    """

    _fire.bump("moe_grouped_gemm_segmented")
    K = x.shape[1]
    M = x.shape[0] if row_src is None else row_src.shape[0]
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
        # 8 は掃引用 (P7 第 3 段)。行数の少ない専門家が多いとき、タイルの
        # 端数で捨てる行が減る。WM は必ず 1 (TM = 8/(8*WM) が 1 以上)
        if bm not in (8, BM16, BM):
            raise ValueError(f"bm={bm} は 8 / 16 / 32 のみ")
        tile_wm = (WM if bm == BM else WM16) if wm is None else wm
        grid_bm = bm
    threads = tile_wm * WN * 32

    dims = mx.array(
        [K, N, M, E, int(mix_threshold or 0)], dtype=mx.int32
    )
    epilogue = row_dst is not None or row_scale is not None
    if epilogue:
        if row_dst is None or row_scale is None:
            raise ValueError("row_dst と row_scale は両方セットで渡すこと")
        if row_dst.shape != (M,) or row_scale.shape != (M,):
            raise ValueError(
                f"row_dst/row_scale の形は (M,)={M} のみ"
                f" (来たのは {row_dst.shape} / {row_scale.shape})")
        if row_dst.dtype != mx.uint32:
            row_dst = row_dst.astype(mx.uint32)
        if row_scale.dtype != mx.float32:
            row_scale = row_scale.astype(mx.float32)
    if row_src is not None and row_src.dtype != mx.uint32:
        row_src = row_src.astype(mx.uint32)

    if N % bn != 0:
        raise ValueError(f"N={N} が bn={bn} で割り切れない")
    if K % bk != 0 or group_size % bk != 0:
        raise ValueError(f"K={K} / group_size={group_size} が bk={bk} と合わない")
    kernel = _get_segmented_kernel(
        mixed, epilogue, row_src is not None, bn, bk)
    template = [
        ("T", x.dtype),
        ("GROUP_SIZE", group_size),
        ("BITS", bits),
        ("SKIP_ROWS", bool(frag_skip)),
    ]
    if not mixed:
        template += [("TILE_M", int(bm)), ("TILE_WM", int(tile_wm))]
    ins = [x, w, scales, biases, row_start, tile_prefix, dims]
    if epilogue:
        ins += [row_dst, row_scale]
    if row_src is not None:
        ins += [row_src]
    (out,) = kernel(
        inputs=ins,
        template=template,
        grid=(threads * (N // bn), n_tiles_max(M, E, grid_bm), 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )
    return out


__all__ = [
    "ENV_KNOB",
    "apple_gpu_family",
    "counts_from_sorted_ids",
    "dense_eligible",
    "enabled",
    "is_nax_device",
    "qmm_dense_clone",
    "qmm_segmented",
    "segment_tables",
    "segmented_eligible",
]
