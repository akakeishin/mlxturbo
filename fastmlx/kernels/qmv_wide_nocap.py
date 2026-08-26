"""mlx の `qmv_wide` カーネル (kernels/quantized.h の `qmv_wide_impl`) から
「タイルは5ベクトルで頭打ち」の上限を外した移植版。

mx.quantized_matmul(x, w, scales, biases, transpose=True) は M(検証幅・行数)が
6〜12 のとき、mlx v0.32.2 のディスパッチ (backend/metal/quantized.cpp の
`qmv_wide`) で

    n_tiles = ceil(M / 5)              // タイル本数
    vecs_per_tg = ceil(M / n_tiles)    // タイルあたりの行数 (5 が上限)

としてグリッドを組む。各タイルは量子化重み(w/scales/biases)を丸ごと読み直すため、
M=6 は 2 タイル (3+3)、M=10 は 2 タイル (5+5)、M=11 は 3 タイル (4+4+3) に割れ、
その回数だけ帯域を食う重み読みが発生する (M=5→6, 10→11 の段差として実測で確認済み。
README.md の「m カーブ税の真因」参照)。

このモジュールはカーネル本体 (`qmv_wide_impl`) をほぼそのまま
`mx.fast.metal_kernel` へ移植しつつ、タイルの5本上限を外して「1タイル=M全体
(6..12)」にする。結果として重み読みは M によらず常に1回になり、増えるのは
x の読み (M 行ぶん) だけになる。x は重みよりずっと小さいので割に合う。

対応範囲: 4bit affine, group_size=64, transpose=True の
    out[M, N] = x[M, K] @ dequant(w[N, K])^T
のみ。x の dtype は bfloat16 / float16 の両対応 (テンプレート型 T)。
それ以外の形状・dtype・bits・group_size は `mx.quantized_matmul` にそのまま
委譲する (呼び出し側は形状を気にしなくてよい drop-in)。

## M ループを Python 側で完全展開している理由 (最初の移植でハマった点)

元の qmv_wide_impl は `vecs_per_tg` (=M) 本のアキュムレータ配列
`U result[vecs_per_tg]` を持ち、`for (int v = 0; v < vecs_per_tg; v++)` で
回す。これを素直に「M をテンプレート定数にした C++ for ループ」のまま移植する
と、M=6,8 では mx.quantized_matmul とほぼ互角なのに、M=10 で 0.6x、M=12 で
0.37x まで悪化する逆転現象が実測で出た (`#pragma unroll` の有無は無関係)。

原因は配列 `result[M]` / `xv[M]` への動的インデックス (`result[v]`) が、
Metal のコンパイラが実際にはループを完全展開しなかった場合にレジスタでは
なく private (device 相当) メモリへ push されること。M が大きいほどこの
スピル往復が線形以上に効いてくる。テンプレート定数として M を渡しても
(コンパイル時に確定していても) この判断はコンパイラの unroll ヒューリス
ティック任せになり、信頼できなかった。

対策は `result[M]` 配列そのものをやめ、Python 側で M 本ぶんの
`result0, result1, ..., result{M-1}` という独立したスカラー変数と、対応する
展開済みコード片を M ごとに文字列生成すること。これで配列・動的インデックス
が完全になくなり、コンパイラの unroll 判断に依存せず全レーンがレジスタに
載る。x 側のポインタもレジスタ確保用の変数にキャッシュせず、都度
`x + v*K + k0` を計算する形に変えると (`xv[M]` 配列も無くす)、さらに安定した
(M=10, M=12 でキャッシュ版よりレジスタ圧が下がり、独立呼び出し計測で
0.8x 台 → 0.9x 台まで改善)。

M は 6..12 の 7 通りしかないので、この生成コストは無視できる。
"""

from typing import Any, Dict

import mlx.core as mx

GROUP_SIZE = 64
BITS = 4

# qmv_wide_impl は元々 M<=5 のタイル向けだが、このカーネルは常に「1タイル=M
# 全体」なので原理上は任意の M で動く。M=6..12 が mlx の税が生じる窓 (README の
# m カーブ) なのでそこに絞ってフォールバック境界を引く。M<6 は mlx の qmv/qmv_fast
# が既に帯域に近く、M>12 はレジスタ圧 (result0..result{M-1} が線形に伸びる) が
# 割に合わなくなる領域。
M_MIN = 6
M_MAX = 12

# 元カーネルの構造を保つ定数 (quantized.h の qmv_wide_impl と同じ値)。
K_LANES = 8  # affine mode の元の分割 (mode=="fp" 側は 16 だが今回は affine のみ)
NUM_SIMDGROUPS = 2
RESULTS_PER_SIMDGROUP = 32 // K_LANES  # 4
ROWS_PER_TG = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS  # 8 output rows / threadgroup
SUB = 8  # サブチャンクの要素数 (= 4bit で 4 byte = 8 値、byte 境界に揃う)


def _build_source(m: int) -> str:
    """M=m 用のカーネル本体を生成する。

    quantized.h の qmv_wide_impl との対応:
      - vecs_per_tg (テンプレート引数) -> m (このファイルでは 1 タイル = M 全体)
      - vec0 / xv[v] = x + min(vec0+v, M-1)*in_vec_size のクランプ -> 不要
        (タイルが M を跨がないので毎回 v < M が保証される。原型のクランプと
        末尾の `if (vec0+v < M)` ガードを削除)
      - dequantize<U, sub, bits>(...) は mlx 内部ヘッダの関数で
        mx.fast.metal_kernel からは呼べないため、bits=4 の分岐だけを直接
        展開してインライン化した (ロジックは同一)
      - simd_shuffle_down によるラダー縮約・8要素サブチャンクでの dequant
        再利用・k_lanes=8 で群を分担する構造はそのまま
      - `result[vecs_per_tg]` 配列は M 本の独立したスカラー変数
        (result0..result{m-1}) に展開している (モジュール docstring 参照:
        配列 + 動的インデックスだと M>=9 あたりで Metal コンパイラがレジスタに
        載せ切れず大きく劣化したため)
    """
    result_decls = "\n".join(f"    U result{v} = 0;" for v in range(m))
    acc_body = "\n".join(
        f"""
            {{
                const device T* xc = x + (size_t){v} * K + k0;
                U acc = 0;
                #pragma unroll
                for (int i = 0; i < sub; i++) {{
                    acc += (U)xc[i] * w_dq[i];
                }}
                result{v} += acc;
            }}"""
        for v in range(m)
    )
    shuffle_body = "\n".join(
        f"""
    result{v} += simd_shuffle_down(result{v}, 4);
    result{v} += simd_shuffle_down(result{v}, 2);
    result{v} += simd_shuffle_down(result{v}, 1);"""
        for v in range(m)
    )
    write_body = "\n".join(
        f"        y[(size_t){v} * N + out_row] = (T)result{v};" for v in range(m)
    )

    return f"""
    constexpr int k_lanes = {K_LANES};
    constexpr int num_simdgroups = {NUM_SIMDGROUPS};
    constexpr int results_per_sg = 32 / k_lanes;  // {RESULTS_PER_SIMDGROUP}
    constexpr int sub = {SUB};

    typedef float U;

    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint tgy = threadgroup_position_in_grid.y;

    short k_lane = (short)(simd_lid % k_lanes);
    short sg_row = (short)(simd_lid / k_lanes);

    int out_row = (int)tgy * (results_per_sg * num_simdgroups) +
        results_per_sg * (int)simd_gid + sg_row;
    int row = min(out_row, N - 1);

    constexpr int in_vec_size_w = K * {BITS} / 8;      // bytes/row (bits={BITS} 固定)
    constexpr int in_vec_size_g = K / {GROUP_SIZE};    // groups/row (group_size={GROUP_SIZE} 固定)

    const device uint8_t* wrow = (const device uint8_t*)w + (size_t)row * in_vec_size_w;
    const device T* srow = scales + (size_t)row * in_vec_size_g;
    const device T* brow = biases + (size_t)row * in_vec_size_g;

    // M={m} 本のアキュムレータ。docstring 参照: 配列でなく展開済みスカラーにする
    // ことでレジスタ確保をコンパイラの unroll 判断に依存させない。
{result_decls}

    // 各レーンが群 (group) を k_lanes 個おきに分担し、群を8要素サブチャンク
    // 単位で dequant してから M 本の入力ベクトルへ使い回す (元コードと同じ)。
    for (int g = k_lane; g < in_vec_size_g; g += k_lanes) {{
        float s = (float)srow[g];
        float b = (float)brow[g];
        float s1 = s / 16.0f;

        #pragma unroll
        for (int sc = 0; sc < {GROUP_SIZE} / sub; sc++) {{
            int k0 = g * {GROUP_SIZE} + sc * sub;
            const device uint8_t* wc = wrow + (k0 * {BITS}) / 8;

            // quantized.h の dequantize<U, 8, 4> をインライン展開したもの:
            //   sc[0]=s, sc[1]=s/16; w_local[2i]=sc[0]*(w[i]&0x0f)+b;
            //   w_local[2i+1]=sc[1]*(w[i]&0xf0)+b;  (シフトなし、係数側で補正)
            U w_dq[sub];
            #pragma unroll
            for (int i = 0; i < sub / 2; i++) {{
                w_dq[2 * i]     = (U)(s  * (float)(wc[i] & 0x0f) + b);
                w_dq[2 * i + 1] = (U)(s1 * (float)(wc[i] & 0xf0) + b);
            }}
{acc_body}
        }}
    }}

    // k_lanes=8 本の部分和をシャッフルラダーで縮約 (results_per_sg=4 の各行が
    // 独立した8レーン組で reduce される。simd_sum は使えない: 1 simdgroup に
    // results_per_sg 行が同居しているため)。
{shuffle_body}

    if (k_lane == 0 && out_row < N) {{
{write_body}
    }}
"""


_KERNELS: Dict[int, Any] = {}


def _get_kernel(m: int) -> Any:
    if not mx.metal.is_available():
        return None
    kernel = _KERNELS.get(m)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"qmv_wide_nocap_m{m}",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_build_source(m),
        )
        _KERNELS[m] = kernel
    return kernel


def _eligible(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
) -> bool:
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    if group_size != GROUP_SIZE or bits != BITS:
        return False
    if x.dtype not in (mx.float16, mx.bfloat16):
        return False
    if scales.dtype != x.dtype or biases.dtype != x.dtype:
        return False
    if w.dtype != mx.uint32:
        return False
    if x.ndim != 2 or w.ndim != 2 or scales.ndim != 2 or biases.ndim != 2:
        return False

    M, K = x.shape
    N = w.shape[0]
    if not (M_MIN <= M <= M_MAX):
        return False
    if K % GROUP_SIZE != 0:
        return False
    if w.shape[1] != K * BITS // 32:
        return False
    n_groups = K // GROUP_SIZE
    if scales.shape != (N, n_groups) or biases.shape != (N, n_groups):
        return False
    return True


def qmv_wide_nocap(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """`mx.quantized_matmul(x, w, scales, biases, transpose=True, ...)` の drop-in。

    out[M, N] = x[M, K] @ dequant(w[N, K])^T を、量子化重みを1タイルにつき
    1回だけ読む移植カーネルで計算する。対応範囲 (4bit affine, group_size=64,
    M in [6, 12], dtype が bfloat16/float16 で x/scales/biases が揃っている)
    の外では `mx.quantized_matmul` にそのまま委譲する。

    Shapes:
      - x: [M, K]
      - w: [N, K * bits / 32] (uint32, mx.quantize の出力そのまま)
      - scales, biases: [N, K / group_size]
    Returns:
      - out: [M, N]、dtype は x と同じ
    """
    if not _eligible(x, w, scales, biases, group_size, bits):
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=group_size, bits=bits
        )

    M, K = x.shape
    N = w.shape[0]
    kernel = _get_kernel(M)
    tg_y = (N + ROWS_PER_TG - 1) // ROWS_PER_TG

    (y,) = kernel(
        inputs=[x, w, scales, biases],
        template=[("T", x.dtype), ("K", K), ("N", N)],
        grid=(32, NUM_SIMDGROUPS * tg_y, 1),
        threadgroup=(32, NUM_SIMDGROUPS, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )
    return y
