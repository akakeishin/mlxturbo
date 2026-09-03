"""prefill の dense 射影用に、4-bit のまま逆量子化を償却する自前 qmm (P10)。

台帳は `docs/research/IDEAS-2026-09-03.md` の P10。狙いは 1 つだけ:

MLX 0.32.2 の dense qmm_t (`quantized.cpp:1058-1065` の bm=bn=32 / wm=wn=2 と
`quantized.h:1197` の BK=32) は、**重みタイルの逆量子化を BM=32 行ごとに
払い直している**。grid は (N/BN, M/BM) なので、W の device 読みと
`QuantizedBlockLoader` の逆量子化 ALU は両方とも M/BM に比例する。BM を 64 に
すれば、同じ W タイル 1 枚が 64 行を養うので**逆量子化の総量が半分**になる。
MMA の総量は変わらない。

BM を上げるのに `_steel_flat.STEEL_HEADER` の `qmm_t_impl` はそのままでは
使えない。あちらは `constexpr int WM = 2; constexpr int WN = 2;` を**本体に
直書き**していて (`_steel_flat.py:1440-1441`)、BM=128 や 256 スレッドの
配置が作れないため。そこでこのファイルは `qmm_t_impl` の写しを 1 つだけ足す
(`_WIDE_HEADER` の `mlxturbo_qmm_t_wide`)。写しに対する変更は 3 点だけ:

  1. `WM` / `WN` を本体の constexpr から template 引数に上げた
  2. X 側 `BlockLoader` の `n_reads` を template 引数 (`XREADS`) にした
     (BM=64 / BK=64 で既定の n_reads が 16〜32 まで膨らみ、1 スレッドが
     32〜64 バイトのベクタ読みをする形になるので、8 で頭打ちにできるように)
  3. W 側 `load_safe` の引数の順を直した (下の「N の端」)
  4. 端数タイルの分岐 (`num_els < BM`) と store は写しのまま

## N の端 (BN != BK のタイルで写しをそのまま使うと壊れる)

`QuantizedBlockLoader::load_safe` は `reduction_dim == 1` のとき
`bi >= src_tile_dim.x` で 0 埋めするが、この `bi` は **N の添字** (BROWS = BN)
なので、比べる相手は N の残り (`num_outs`) でなければならない。本家の
`qmm_t_impl` は `short2(BK, num_outs)` を渡していて `.x` が BK になっている。
本家は BN == BK == 32 しか作らないので `bi >= BK` は決して立たず、N の端の
W タイルは 0 埋めされずに**範囲外を読む** (読んだ値は `store_result_safe` の
マスクで捨てられるので結果は正しい)。

このファイルは BN=64 のタイルを作るので、そのまま写すと `bi >= 32` が
**有効な行 (bi = 32..63) にも立ち**、N の端のタイルで出力の後ろが 0 になる
(N=300 / BN=64 で最後の 12 列が 0 になるのを 2026-09-04 に観測)。そこで
呼び出しを `short2(num_outs, BK)` にして、`.x` に N の残りを入れる。
BN == BK のタイルでは結果は変わらない (0 埋めか範囲外読みかの違いで、どちらも
store のマスクで捨てられる列) が、範囲外読みは無くなる。

MMA・ローダー・dequantize・store は `STEEL_HEADER` の写しをそのまま呼ぶ。

## 数値

K の縮約順は **タイル形を変えても変わらない**。`BlockMMA::mma` は
threadgroup タイルの中を `kFragSize = 8` 刻みで昇順に回るだけで、BM / BN は
どの出力要素をどのスレッドが持つかを決めるだけ、BK は 1 回の外側ループで
何刻み進むかを決めるだけだから。したがって**全変種がお互いにビット一致する**
のが期待値で、変種どうしが割れたらタイルの張り方を間違えている合図になる
(計測より先に効く検査として使う)。

`mx.quantized_matmul(transpose=True)` とのビット一致は、**素が `qmm_t` を
選ぶ形でだけ**成り立つ。MLX 0.32.2 は同じ呼び出しを 3 つの実装に振り分ける
(`libmlx.dylib` のカーネル名 `qmv_fast` / `qmm_t` / `qmm_t_splitk`)。

  - ``M < 13``: `qmv` 系。1 行を複数スレッドで割る別の縮約順。
  - ``K >= 128`` かつ出力タイル数 ``ceil(M/32) * ceil(N/32) < 256``:
    **`qmm_t_splitk`**。K を分割して部分和を出し、あとで足す。並列度を稼ぐ
    代わりに部分和が出力 dtype (bf16) を経由するので、**素の側が真値から離れる**。
  - それ以外: `qmm_t`。ここでだけ写しと**ビット一致**する。

`stock_bit_matches` がこの振り分けを写している。境界の 13 / 256 / 128 は
`tools/qmm_wide_shape_micro.py` の掃引で当てた実測値 (M3 Max / MLX 0.32.2。
タイル数の 256 は N=320 / 640 / 1280 の 3 本、M の 13 は N=320 と N=12288 の
2 本で一致) であって MLX の契約ではない。

**splitk 帯での食い違いは写しの欠陥ではない。**同じ掃引で、素と同形のタイル
`m32n32k32w2x2` も BM=64 のタイルとまったく同じだけ素から外れ、両者は
お互いにビット一致し、float32 の参照に対しては**どちらも素より近い**
(HC の down, K=10240 N=320, M=256 で 素 0.074 に対し写しは 0.031)。
つまり `qmm_wide` は splitk 帯でも `qmm_t` の答えをそのまま出しているだけで、
比較相手のほうが M で入れ替わっている。

## 現状

proof-of-life。まだ本番の呼び手に配線していない (`fused.py` の
`enable_wide_projections` の口に入れるかは `tools/qmm_wide_micro.py` の
判定次第)。判定線は形 (a)(b) の M=2048 で素の 0.87 倍以下。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from . import _fire
from ._steel_flat import STEEL_HEADER

# 本番の重みの量子化 (QuantizedLinear と同じ)
GROUP_SIZE = 64
BITS = 4

SIMD_SIZE = 32
FRAG = 8  # steel の kFragSize

_KERNELS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# `qmm_t_impl` の写し (WM / WN / XREADS を template に上げただけ)
# ---------------------------------------------------------------------------

_WIDE_HEADER = r"""
// ---- mlxturbo: quantized.h:1192-1318 (qmm_t_impl) の写し。
// 変更点は (1) WM / WN を template に上げた (2) X 側 BlockLoader の n_reads を
// template (XREADS) にした (3) W 側 load_safe の short2 の順を直した (BN != BK の
// タイルで N の端が壊れるため。上の「N の端」)、の 3 つだけ。
// それ以外は 1 文字も変えていない。
template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM,
    const int BK,
    const int BN,
    const int WM,
    const int WN,
    const int XREADS>
METAL_FUNC void mlxturbo_qmm_t_wide(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& K_eff,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::
      BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, XREADS>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        // mlxturbo: 本家は short2(BK, num_outs)。reduction_dim == 1 の
        // QuantizedBlockLoader は .x を N の添字と比べるので、BN != BK だと
        // 有効な行まで 0 埋めされる (ヘッダ冒頭の「N の端」)
        loader_w.load_safe(short2(num_outs, BK));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(num_outs, BK));  // mlxturbo: 上と同じ
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}
"""


# 呼ぶだけの本体。threadgroup メモリの宣言と次元の受け渡し以外は書かない。
_WIDE_SOURCE = """
  constexpr int BM = @BM@;
  constexpr int BN = @BN@;
  constexpr int BK = @BK@;
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  // dims = (K, N, M)。3 要素なので constant アドレス空間に載り、
  // 素の `const constant int&` と同じ渡し方になる (dense clone と同じ)
  mlxturbo_qmm_t_wide<T, GROUP_SIZE, BITS, ALIGNED_N, BM, BK, BN,
                      @WM@, @WN@, @XREADS@>(
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
"""


# ---------------------------------------------------------------------------
# タイル
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    """1 threadgroup が受け持つ出力タイルと simdgroup の配置。

    ``xreads`` は X 側 `BlockLoader` の 1 スレッドあたりの要素数。0 なら
    steel の既定 ``(BK * BM) / threads`` をそのまま使う。
    """

    bm: int
    bn: int
    bk: int
    wm: int
    wn: int
    xreads: int = 0

    @property
    def threads(self) -> int:
        return self.wm * self.wn * SIMD_SIZE

    @property
    def x_reads(self) -> int:
        return self.xreads or ((self.bk * self.bm) // self.threads)

    @property
    def name(self) -> str:
        base = f"m{self.bm}n{self.bn}k{self.bk}w{self.wm}x{self.wn}"
        return base if not self.xreads else f"{base}r{self.xreads}"

    @property
    def tgp_bytes(self) -> int:
        """bf16 のときの threadgroup メモリ (Xs + Ws)。上限は 32 KB。"""
        bk_padded = self.bk + 8
        return (self.bm + self.bn) * bk_padded * 2

    @property
    def accum_regs(self) -> int:
        """1 スレッドが持つ fp32 の累算レジスタ数 (Ctile ぶん)。"""
        tm = self.bm // (FRAG * self.wm)
        tn = self.bn // (FRAG * self.wn)
        return tm * tn * 2


# 素の dense qmm_t と同じ形。写しが素と一致することを毎回確かめる基準線
STOCK = Tile(32, 32, 32, 2, 2)

# 掃引する候補。BM を上げる (逆量子化の償却) のが主眼で、BN / BK / スレッド数は
# threadgroup メモリとレジスタの当たり方を見るための振り幅
TILES: dict[str, Tile] = {t.name: t for t in [
    STOCK,
    Tile(64, 32, 32, 2, 2),            # BM だけ 2 倍 (P10 の本命、128 スレッド)
    Tile(64, 32, 32, 2, 2, xreads=8),  # 同上、X の 32 B ベクタ読みを 16 B に
    Tile(64, 32, 32, 4, 2),            # 同じタイルを 256 スレッドで
    Tile(64, 64, 32, 2, 2),            # X 側の帯域も半分。累算 32 レジスタ
    Tile(64, 64, 32, 2, 4),            # 同じタイルを 256 スレッドで
    Tile(64, 64, 32, 4, 2),
    Tile(128, 32, 32, 4, 2),           # BM 4 倍。Xs 10 KB
    Tile(128, 64, 32, 4, 2),           # Xs+Ws 15 KB、累算 32 レジスタ
    Tile(64, 32, 64, 2, 2, xreads=8),  # K を 2 倍に刻む (barrier 半分)
    Tile(64, 64, 64, 2, 4, xreads=8),
]}


def tile_ok(tile: Tile, K: int, N: int, group_size: int = GROUP_SIZE,
            bits: int = BITS) -> str | None:
    """このタイルが steel の写しで成立するか。駄目なら理由を返す。"""

    if tile.bm % (FRAG * tile.wm) or tile.bn % (FRAG * tile.wn):
        return "BM/BN が simdgroup frag (8) x WM/WN で割り切れない"
    if tile.bk < SIMD_SIZE or tile.bk % SIMD_SIZE:
        return "BK が SIMD_SIZE の倍数でない"
    if tile.bk > group_size or group_size % tile.bk:
        return "BK が group_size を超える / 割り切らない"
    if K % tile.bk:
        return f"K={K} が BK={tile.bk} で割り切れない"
    pack_factor = 8 // bits
    bcols_packed = tile.bk // pack_factor
    if (bcols_packed * tile.bn) % tile.threads:
        return "W ローダーの (BCOLS_PACKED * BN) がスレッド数で割り切れない"
    if bcols_packed * tile.bn < tile.threads:
        return "W タイルがスレッド数より小さい (n_reads=1 の分岐に落ちる)"
    if (tile.bk * tile.bm) % tile.threads:
        return "X タイルがスレッド数で割り切れない"
    if tile.bk % tile.x_reads or (tile.threads * tile.x_reads) % tile.bk:
        return "XREADS が BK を割り切らない"
    if tile.tgp_bytes > 32 * 1024:
        return f"threadgroup メモリ {tile.tgp_bytes} B が 32 KB を超える"
    if tile.threads > 1024:
        return "threadgroup が 1024 スレッドを超える"
    return None


# 素が `qmm_t` を選ぶ帯 (= 写しとビット一致する帯) の判定。上の「数値」の節。
_STOCK_QMV_MAX_ROWS = 13   # M < これ は qmv 系
_STOCK_SPLITK_TILES = 256  # 出力タイル数がこれ未満なら qmm_t_splitk
_STOCK_SPLITK_MIN_K = 128  # K がこれ未満なら分割しない


def stock_bit_matches(M: int, K: int, N: int) -> bool:
    """`mx.quantized_matmul(transpose=True)` がこの形で `qmm_t` を選ぶか。

    ``True`` の帯でだけ :func:`qmm_wide` は素とビット一致する。``False`` の帯
    (小さい M、または出力タイルが少ない細長い形) では素が `qmv` / `qmm_t_splitk`
    に振れるので、素との差は写しの欠陥ではない (「数値」の節)。

    :func:`eligible` には入れていない。splitk 帯でも写しの答えは `qmm_t` の答え
    そのもので、float32 参照に対しては素より近い。ここで弾くと、正しくて速い形を
    落とすことになる。**素とのビット一致を検査に使うとき**にだけ使う判定。
    """

    if M < _STOCK_QMV_MAX_ROWS:
        return False
    if K < _STOCK_SPLITK_MIN_K:
        return True
    tiles = ((M + 31) // 32) * ((N + 31) // 32)
    return tiles >= _STOCK_SPLITK_TILES


def _get_kernel(tile: Tile):
    kernel = _KERNELS.get(tile.name)
    if kernel is None:
        source = (
            _WIDE_SOURCE.replace("@BM@", str(tile.bm))
            .replace("@BN@", str(tile.bn))
            .replace("@BK@", str(tile.bk))
            .replace("@WM@", str(tile.wm))
            .replace("@WN@", str(tile.wn))
            .replace("@XREADS@", str(tile.x_reads))
        )
        kernel = mx.fast.metal_kernel(
            name=f"mlxturbo_qmm_wide_{tile.name}",
            input_names=["w", "scales", "biases", "x", "dims"],
            output_names=["y"],
            source=source,
            header=STEEL_HEADER + _WIDE_HEADER,
            ensure_row_contiguous=True,
        )
        _KERNELS[tile.name] = kernel
    return kernel


def eligible(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    tile: Tile,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
) -> bool:
    """:func:`qmm_wide` が素と同じ結果を出す形か。"""

    if bits != BITS:
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
    if K % group_size:
        return False
    if w.shape[1] != K * bits // 32:
        return False
    groups = K // group_size
    if scales.shape != (N, groups) or biases.shape != (N, groups):
        return False
    return tile_ok(tile, K, N, group_size, bits) is None


def qmm_wide(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    tile: Tile = Tile(64, 32, 32, 2, 2),
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
) -> mx.array:
    """`mx.quantized_matmul(x, w, scales, biases, transpose=True)` の広タイル版。

    ``x`` は (M, K)、``w`` は (N, K*bits/32) の packed uint32、
    ``scales``/``biases`` は (N, K/group_size)。戻り値は (M, N)。
    """

    why = tile_ok(tile, x.shape[1], w.shape[0], group_size, bits)
    if why is not None:
        raise ValueError(f"タイル {tile.name} は成立しない: {why}")

    _fire.bump(f"qmm_wide_{tile.name}")
    M, K = x.shape
    N = w.shape[0]
    aligned_n = (N % tile.bn) == 0

    dims = mx.array([K, N, M], dtype=mx.int32)
    kernel = _get_kernel(tile)
    (out,) = kernel(
        inputs=[w, scales, biases, x, dims],
        template=[
            ("T", x.dtype),
            ("GROUP_SIZE", group_size),
            ("BITS", bits),
            ("ALIGNED_N", aligned_n),
        ],
        # 素は dispatch_threadgroups((N/BN, M/BM, 1), (32, WN, WM))。
        # custom kernel は dispatch_threads なので x にスレッド数を掛ける
        grid=(
            tile.threads * ((N + tile.bn - 1) // tile.bn),
            (M + tile.bm - 1) // tile.bm,
            1,
        ),
        threadgroup=(tile.threads, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )
    return out


__all__ = [
    "BITS",
    "GROUP_SIZE",
    "STOCK",
    "TILES",
    "Tile",
    "eligible",
    "qmm_wide",
    "stock_bit_matches",
    "tile_ok",
]
