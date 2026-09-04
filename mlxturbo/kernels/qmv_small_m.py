"""検証幅 M=1..8 の量子化 dense 射影 (4bit affine)。**行ごとに mlx の `qmv_fast`
とビット一致**しつつ、重みは 1 回しか読まない。

## なぜ要るか

投機デコードの検証フォワードは、同じ重みに **M 行 (= 検証幅)** を掛ける。MLX
0.32.2 はこれを M=1 なら `qmv_fast`、M>=2 なら `qmv_wide` (1 タイル最大 5 行) に
振り、M が増えるほど**実効帯域が落ちる**。27B (qwen3_5) の実機で測った MLP
64 層ぶんの実効帯域は

    M=1 372 GB/s / M=2 312 / M=3 218 / M=4 216

で、重みの読みは M<=5 ならどれも 1 回なのに **M=4 は M=1 の 1.72 倍**かかる
(`tools/verify_width_cost_27b.py`、`bench/results/width-cost-27b-0904.json`)。
27B の本番の検証幅は 98% が 3/4/5 なので、これがそのまま round の値段になる。
既存の逃げ道はどれも届かない (`tools/qmv_small_m_micro.py`): `qmv_wide_nocap`
は stock と同速、`fast_qmm` は天井が 209 GB/s、`qmm_wide` は遅延律速。

## 作り (mlx の `qmv_fast_impl` の構造をそのまま、行だけ M 本にする)

`mlx/backend/metal/kernels/quantized.h` の `qmv_fast_impl` は 4bit のとき

    pack_factor = 8, packs_per_thread = 2  -> values_per_thread = 16
    block_size = values_per_thread * SIMD_SIZE = 512
    scale_step_per_thread = group_size / values_per_thread = 4

で、1 スレッドが「K のうち連続 16 値」を持ち、512 値ごとに次の塊へ進む。
1 出力行の 32 レーンぶんの部分和を最後に `simd_sum` で畳む。

ここは **`results_per_simdgroup` を 1 に落として** (1 simdgroup = 出力 1 行)、
空いたレジスタに **M 行ぶんのアキュムレータ**を置く。`results_per_simdgroup`
は「1 スレッドが何本の出力行を担当するか」でしかなく、**ある 1 行の K 方向の
足し込み順には関係しない**ので、1 に落としても 1 行あたりの答えは変わらない。

したがって

    out[v] == mx.quantized_matmul(x[v:v+1], ...)   (v = 0..M-1、ビット一致)

が成り立つ。**検証幅が変わっても 1 行の答えが動かない**ので、投機の検証で
丸めが変わって生成列が分岐する問題も起きない。

読み方も `qmv_fast` と同じ: 重みは 1 スレッドあたり 8 バイト連続
(`uint2` 相当) で、simdgroup 32 レーンで 256 バイト連続 = 完全にコアレス。
x は M 行ぶんを 4 値ずつ読み、`load_vector` と同じく 1/16 / 1/256 / 1/4096 に
割ってからニブル (シフトしていない生の 0x00f0 等) と掛ける。

## 対応範囲

4bit affine、`group_size` が `values_per_thread`(=16) の倍数、K が
`block_size`(=512) の倍数、x は bf16 / fp16 の 2 次元 (M<=8)。
**この条件は mlx が `qmv_fast` を選ぶ条件と同じ**で、外れると mlx 側は
`qmv` / `qmv_wide` の別の順になるためビット一致の前提が崩れる。外れたら
`mx.quantized_matmul` にそのまま委譲する (呼び手は形を気にしなくてよい)。
bits=8 はまだ (mlx 側の `qdot` の形が違う) -- 委譲する。
"""

from typing import Any, Dict, Tuple

import mlx.core as mx

GROUP_SIZE = 64
BITS = 4

# mlx の qmv_fast_impl (bits=4) と同じ定数
PACK_FACTOR = 8                       # 32 / bits
PACKS_PER_THREAD = 2
VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD   # 16
SIMD_SIZE = 32
BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE           # 512

M_MIN = 1
M_MAX = 8
NUM_SIMDGROUPS = 2                    # threadgroup あたりの simdgroup 数 (mlx と同じ)
RESULTS_PER_SIMDGROUP = 4             # 1 スレッドが持つ出力行数 (mlx と同じ)

_KERNELS: Dict[Tuple[int, int, int], Any] = {}


def _build_source(m: int, nsg: int, rps: int) -> str:
    """M=m / simdgroup 数 nsg / 1 スレッドが持つ出力行数 rps 用の本体を生成する。

    ``rps`` は mlx の `results_per_simdgroup` そのもの (本家は 4)。**1 行の
    K 方向の足し込み順には効かない**ので、いくつにしてもビット一致は保たれる。
    効くのは x の読み直しで、rps 行ぶんで 1 回ぶんの x 読みを分け合う ---
    rps=1 にすると x の load が rps 倍に膨らんで、M が増えるほど負ける
    (2026-09-04 の micro で確認)。

    M x rps 本のアキュムレータは配列でなく展開済みスカラーにする。配列 +
    動的インデックスだと Metal のコンパイラが unroll を諦めたときに private
    メモリへ落ちる (`kernels/qmv_wide_nocap.py` の docstring と同じ理由)。
    """

    decls = []
    for r in range(rps):
        for v in range(m):
            decls.append(f"    U res{r}_{v} = 0; U acc{r}_{v} = 0;")
    for v in range(m):
        decls.append(f"    U sum{v} = 0;")
    acc_decl = "\n".join(decls)

    zero = "\n".join(
        ["        " + " ".join(f"acc{r}_{v} = 0;" for v in range(m))
         for r in range(rps)]
        + ["        " + " ".join(f"sum{v} = 0;" for v in range(m))])

    # i 番目の 4 値ぶんの x を **rps 行で共有**して読む (mlx の load_vector が
    # x_thread を 1 回作って results_per_simdgroup 行に使い回すのと同じ)。
    # **一致の要**: `sum += x[i] + x[i+1] + x[i+2] + x[i+3];` は sum が float
    # でも右辺が **T (bf16/fp16) のまま**足されてから U に上がる (MSL の
    # bfloat/half 演算は結果が同じ型に丸まる)。float に上げてから足すと
    # `bias * sum` の項が 1 ulp ずれ、出力の 3〜6% の要素が割れる
    # (2026-09-04 実測: bf16 M=4 で 4016/69632 要素、max|d| 7.8e-3)。
    xload = "\n".join(f"""
            const device T* xp{v} = xb + {v} * K + 4 * i;
            U t{v}_0 = static_cast<U>(xp{v}[0]);
            U t{v}_1 = static_cast<U>(xp{v}[1]) / 16.0f;
            U t{v}_2 = static_cast<U>(xp{v}[2]) / 256.0f;
            U t{v}_3 = static_cast<U>(xp{v}[3]) / 4096.0f;
            sum{v} += static_cast<U>(xp{v}[0] + xp{v}[1] + xp{v}[2] + xp{v}[3]);"""
                      for v in range(m))

    # 1 語 (uint16 = 4 値) を rps 行 x M 行ぶん流す。mlx の qdot と同じ式・同じ順:
    #   accum += (x0*(w&0x000f) + x1*(w&0x00f0) + x2*(w&0x0f00) + x3*(w&0xf000))
    rows = []
    for r in range(rps):
        macs = "\n".join(
            f"""                acc{r}_{v} += (t{v}_0 * n0 + t{v}_1 * n1 +
                                t{v}_2 * n2 + t{v}_3 * n3);"""
            for v in range(m))
        rows.append(f"""
            {{
                uint16_t wi = wq{r}[i];
                U n0 = static_cast<U>(wi & 0x000f);
                U n1 = static_cast<U>(wi & 0x00f0);
                U n2 = static_cast<U>(wi & 0x0f00);
                U n3 = static_cast<U>(wi & 0xf000);
{macs}
            }}""")
    row_body = "".join(rows)

    # 行のずらしは**コンパイル時の定数**にする (mlx の qmv_fast の
    # `ws + row * in_vec_size_w` と同じ)。実行時の値にするとアドレス計算が
    # ブロックごとに残って、M=1 でも本家の 1.9 倍まで落ちた (2026-09-04)。
    # 端の行が出ないことは呼び手が保証する (N % (nsg*rps) == 0)。
    wptrs = "\n".join(
        f"        const device uint16_t* wq{r} = "
        f"(const device uint16_t*)(ws + {r} * in_vec_size_w);" for r in range(rps))
    sload = "\n".join(
        f"        U s{r} = static_cast<U>(sl[{r} * in_vec_size_g]);"
        f" U b{r} = static_cast<U>(bl[{r} * in_vec_size_g]);" for r in range(rps))
    fold = "\n".join(
        f"        res{r}_{v} += s{r} * acc{r}_{v} + b{r} * sum{v};"
        for r in range(rps) for v in range(m))
    reduce_ = "\n".join(
        f"    res{r}_{v} = simd_sum(res{r}_{v});"
        for r in range(rps) for v in range(m))
    write = "\n".join(
        f"        y[(size_t){v} * N + out_row + {r}] = static_cast<T>(res{r}_{v});"
        for r in range(rps) for v in range(m))

    return f"""
    typedef float U;

    constexpr int pack_factor = {PACK_FACTOR};
    constexpr int packs_per_thread = {PACKS_PER_THREAD};
    constexpr int bytes_per_pack = 4;
    constexpr int values_per_thread = {VALUES_PER_THREAD};
    constexpr int block_size = {BLOCK_SIZE};
    constexpr int num_simdgroups = {nsg};
    constexpr int results_per_simdgroup = {rps};
    constexpr int scale_step_per_thread = {GROUP_SIZE} / values_per_thread;

    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint tgy = threadgroup_position_in_grid.y;

    int out_row = (int)tgy * (num_simdgroups * results_per_simdgroup)
        + (int)simd_gid * results_per_simdgroup;

    constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;  // K/2 バイト
    constexpr int in_vec_size_g = K / {GROUP_SIZE};

    const device uint8_t* ws = (const device uint8_t*)w
        + (size_t)out_row * in_vec_size_w
        + simd_lid * packs_per_thread * bytes_per_pack;
    const device T* sl = scales + (size_t)out_row * in_vec_size_g
        + simd_lid / scale_step_per_thread;
    const device T* bl = biases + (size_t)out_row * in_vec_size_g
        + simd_lid / scale_step_per_thread;
    const device T* xb = x + simd_lid * values_per_thread;

{acc_decl}

    for (int k = 0; k < K; k += block_size) {{
{wptrs}
{sload}
{zero}

        #pragma unroll
        for (int i = 0; i < values_per_thread / 4; i++) {{
{xload}
{row_body}
        }}

{fold}

        ws += block_size * bytes_per_pack / pack_factor;
        sl += block_size / {GROUP_SIZE};
        bl += block_size / {GROUP_SIZE};
        xb += block_size;
    }}

{reduce_}

    if (simd_lid == 0) {{
{write}
    }}
"""


def _get_kernel(m: int, nsg: int, rps: int):
    if not mx.metal.is_available():
        return None
    key = (m, nsg, rps)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"qmv_small_m_m{m}_sg{nsg}_r{rps}",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_build_source(m, nsg, rps),
        )
        _KERNELS[key] = kernel
    return kernel


def eligible(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    m_min: int = M_MIN,
    m_max: int = M_MAX,
) -> bool:
    """このカーネルが「行ごとに mlx の qmv_fast とビット一致」を保てる形か。

    条件は mlx が `qmv_fast` を選ぶ条件と同じにしてある (K が block_size の
    倍数、group_size が values_per_thread の倍数)。外れると mlx 側が別の
    縮約順のカーネルに振れるので、一致の前提が消える。
    """

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    if bits != BITS or group_size != GROUP_SIZE:
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
    if not (m_min <= M <= m_max):
        return False
    if group_size % VALUES_PER_THREAD:
        return False
    if K % BLOCK_SIZE:
        return False
    # mlx が出力行を丸ごと qmv_fast で回す条件 (num_simdgroups=2 x
    # results_per_simdgroup=4)。割り切れないと mlx 側は端の行だけ別扱いに
    # なり、そこだけ 1 ulp ずれる (N=1025 で 3075 要素中 1 つ、2026-09-04)。
    # 本番の N (17408 / 5120 / 10240 / 6144 / 12288 / 248320 / 2560) は全部
    # 8 の倍数なので、ここで弾いても当たる形は減らない。
    if N % (2 * 4):
        return False
    if w.shape[1] != K * BITS // 32:
        return False
    n_groups = K // group_size
    if scales.shape != (N, n_groups) or biases.shape != (N, n_groups):
        return False
    return True


def qmv_small_m(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
    nsg: int = NUM_SIMDGROUPS,
    rps: int = RESULTS_PER_SIMDGROUP,
) -> mx.array:
    """`mx.quantized_matmul(x, w, scales, biases, transpose=True, ...)` の drop-in。

    Shapes:
      - x: [M, K] (M <= 8)
      - w: [N, K * bits / 32] (uint32、`mx.quantize` の出力そのまま)
      - scales, biases: [N, K / group_size]
    Returns:
      - out: [M, N]、dtype は x と同じ。**各行が M=1 の素の呼び出しとビット一致**
    """

    if not eligible(x, w, scales, biases, group_size, bits):
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=group_size, bits=bits
        )

    M, K = x.shape
    N = w.shape[0]
    rows_per_tg = nsg * rps
    if N % rows_per_tg:
        # 端の threadgroup が出る組み合わせは当てない (カーネルは行の端を
        # 弾かない = コンパイル時定数の代金)。既定の 2x4 = 8 は本番の N を
        # 全部割り切る。
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
    tg_y = N // rows_per_tg
    kernel = _get_kernel(M, nsg, rps)

    (y,) = kernel(
        inputs=[x, w, scales, biases],
        template=[("T", x.dtype), ("K", K), ("N", N)],
        grid=(SIMD_SIZE, nsg * tg_y, 1),
        threadgroup=(SIMD_SIZE, nsg, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )
    return y


__all__ = ["M_MAX", "M_MIN", "NUM_SIMDGROUPS", "RESULTS_PER_SIMDGROUP",
           "eligible", "qmv_small_m"]
