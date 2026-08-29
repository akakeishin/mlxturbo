"""Small-M quantized matmul, B fragment built by per-lane direct reads.

fast_qmm (mlxturbo/fast_qmm.py) の B ステージング (threadgroup スラブ +
simdgroup_barrier 2 本 + ストライド 8 のスカラー store 16 本) が hot ループの
約 250 命令 / 412 を占める (docs/ISA-DIFF.md §2)。この実装は ISA 検証済みの
v_direct probe (tools/isa/variants.py、hot 314 = fast_qmm 比 -24%) の構造を
製品側に持ち込む。

構造 (docs/ISA-DIFF.md §6 の v5 設計差分):

1. B 断片はレーン直読み。`simdgroup_matrix<T,8,8>` の 1 レーンは
   (frag_row, frag_col) と (frag_row, frag_col+1) だけを持ち、1 つの k タイル
   では 1 列の全 8 行が 1 語の packed word に入っている。レーンは自分の
   2 列ぶんの語を uint4 で読み、frag_row で nibble を選ぶ。shuffle も
   threadgroup スラブも不要。
2. threadgroup メモリは split-K 縮約の partials だけ (2KB、wide 4KB)。
   hot ループ内の barrier は 0。
3. 重み読みは 1 回。threadgroup が出力 8 列を持ち、K を 8 simdgroup で分ける。
4. A 断片は M=8 (と wide の下タイル) は device から simdgroup_load 直読み。
   M<8 は frag_row < M のマスク付きレーン直読みでゼロ行を作るので、
   ホスト側のゼロ埋めコピー (mx.zeros + mx.concatenate、fast_qmm の
   M=6/7 コピー税) が消える。wide の M=9..15 も同様にパディング不要。
5. M ごとの特殊化はテンプレート引数 MD で行う。

frag_row / frag_col のレーン所有式は出荷済み _qmm_skinny_mma_source.py と
同一で、bench/test_dispatch.py の数値ゲートを通過済みのもの。

2026-08-27 依存チェーン実測 (M3 Max、chain_ab): 全 4 形状・全 M で fast_qmm に
敗退 (M=8 で 5-12% 遅、wide 帯で 20-60% 遅)。ISA-DIFF §6 の反転条件のうち
「同じ packed 32B を 8 レーンが読む L1 冗長」「maxReg 52 の occupancy」が
実際に効いたと推定。B ステージング (threadgroup スラブ) 維持が正解と確定。
経路には接続しない。masked A 構築 (frag_row < M) の数値正当性の実証と
負の結果の記録としてファイルは残す。
"""

import os

import mlx.core as mx

TGT = 256          # 8 simdgroups; threadgroup = 8 出力列 x split-K 8
N_MIN = 4096       # これ未満は grid が痩せて GPU が遊ぶ (fast_qmm と同じ判断)
M_WIDE_MAX = 16    # C タイル 2 枚の上限

_SRC = r"""
    const int K = KD, N = ND, M = MD;
    constexpr int CT = (MD <= 8) ? 1 : 2;   // C タイル数 (行 0-7 / 8-15)

    uint tid  = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg   = tid >> 5;
    uint lane = tid & 31;

    int n0 = (int)tgid * 8;                 // threadgroup 1 つが出力 8 列

    // レーンが持つ C 断片は (frag_row, frag_col) と (frag_row, frag_col+1)。
    // この式は _qmm_skinny_mma_source.py で数値ゲート通過済み。
    const ushort quad = lane / 4;
    const ushort frag_row = (quad & 4) + (lane / 2) % 4;
    const ushort frag_col = (quad & 2) * 2 + (lane % 2) * 2;

    const int groups = K / 64;
    const int packed_stride = K / 8;

    threadgroup float red[8 * CT * 64];     // split-K 縮約のみ (2KB / wide 4KB)

    simdgroup_matrix<float, 8, 8> C0 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> C1 = simdgroup_matrix<float, 8, 8>(0);

    const int ca = n0 + frag_col;           // レーン担当の 2 出力列
    const int cb = ca + 1;
    const bool va = ca < N;
    const bool vb = cb < N;

    // ── split-K: 8 simdgroup が量子化グループ (64 要素) を 8 本おきに歩く。
    //    B 断片はレーン直読みで作るので hot ループに barrier が無い。
    for (int g = (int)sg; g < groups; g += 8) {
        uint4 pa_lo = 0, pa_hi = 0, pb_lo = 0, pb_hi = 0;
        float sa = 0.0f, ba = 0.0f, sb = 0.0f, bb = 0.0f;
        if (va) {
            const device uint4* wa = (const device uint4*)(
                w + (size_t)ca * packed_stride + g * 8);
            pa_lo = wa[0];
            pa_hi = wa[1];
            sa = (float)sc[(size_t)ca * groups + g];
            ba = (float)bi[(size_t)ca * groups + g];
        }
        if (vb) {
            const device uint4* wb = (const device uint4*)(
                w + (size_t)cb * packed_stride + g * 8);
            pb_lo = wb[0];
            pb_hi = wb[1];
            sb = (float)sc[(size_t)cb * groups + g];
            bb = (float)bi[(size_t)cb * groups + g];
        }

        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            // unroll 後は kt が定数になり uint4 成分の直接参照に畳まれる
            // (スカラー配列 wa_words[8] は mov を積むので使わない、ISA-DIFF §6-7)
            const uint wda = (kt < 4) ? pa_lo[kt & 3] : pa_hi[kt & 3];
            const uint wdb = (kt < 4) ? pb_lo[kt & 3] : pb_hi[kt & 3];

            simdgroup_matrix<bfloat16_t, 8, 8> B;
            B.thread_elements()[0] =
                (bfloat16_t)((float)((wda >> (4 * frag_row)) & 0xFu) * sa + ba);
            B.thread_elements()[1] =
                (bfloat16_t)((float)((wdb >> (4 * frag_row)) & 0xFu) * sb + bb);

            const int kbase = g * 64 + kt * 8;

            simdgroup_matrix<bfloat16_t, 8, 8> A0;
            if (MD >= 8) {
                // 行 0-7 が全部実在するので hardware block load
                simdgroup_load(A0, x + kbase, K);
            } else {
                // M<8: 実在しない行はレーン側で 0 を作る。隣接 2 要素は
                // コンパイラが 32bit ロード 1 本に併合する (ISA-NOTES §3)。
                A0.thread_elements()[0] = frag_row < M
                    ? x[(size_t)frag_row * K + kbase + frag_col]
                    : bfloat16_t(0);
                A0.thread_elements()[1] = frag_row < M
                    ? x[(size_t)frag_row * K + kbase + frag_col + 1]
                    : bfloat16_t(0);
            }
            simdgroup_multiply_accumulate(C0, A0, B, C0);

            if (CT == 2) {
                simdgroup_matrix<bfloat16_t, 8, 8> A1;
                if (MD == 16) {
                    simdgroup_load(A1, x + (size_t)8 * K + kbase, K);
                } else {
                    A1.thread_elements()[0] = 8 + frag_row < M
                        ? x[(size_t)(8 + frag_row) * K + kbase + frag_col]
                        : bfloat16_t(0);
                    A1.thread_elements()[1] = 8 + frag_row < M
                        ? x[(size_t)(8 + frag_row) * K + kbase + frag_col + 1]
                        : bfloat16_t(0);
                }
                simdgroup_multiply_accumulate(C1, A1, B, C1);
            }
        }
    }

    // ── split-K 縮約。partials を置いて sg0 が 8 本を足す。
    red[(sg * CT) * 64 + lane * 2] = C0.thread_elements()[0];
    red[(sg * CT) * 64 + lane * 2 + 1] = C0.thread_elements()[1];
    if (CT == 2) {
        red[(sg * CT + 1) * 64 + lane * 2] = C1.thread_elements()[0];
        red[(sg * CT + 1) * 64 + lane * 2 + 1] = C1.thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        if (frag_row < M) {
            float t0 = 0.0f, t1 = 0.0f;
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                t0 += red[(s * CT) * 64 + lane * 2];
                t1 += red[(s * CT) * 64 + lane * 2 + 1];
            }
            if (va) out[(size_t)frag_row * N + ca] = (bfloat16_t)t0;
            if (vb) out[(size_t)frag_row * N + cb] = (bfloat16_t)t1;
        }
        if (CT == 2 && 8 + frag_row < M) {
            float t0 = 0.0f, t1 = 0.0f;
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                t0 += red[(s * CT + 1) * 64 + lane * 2];
                t1 += red[(s * CT + 1) * 64 + lane * 2 + 1];
            }
            if (va) out[(size_t)(8 + frag_row) * N + ca] = (bfloat16_t)t0;
            if (vb) out[(size_t)(8 + frag_row) * N + cb] = (bfloat16_t)t1;
        }
    }
"""

_KERNEL = mx.fast.metal_kernel(
    name="qmm_direct",
    input_names=["x", "w", "sc", "bi"],
    output_names=["out"],
    source=_SRC,
    # 素の線形添字で読むので行連続を固定する (fast_qmm と同じ判断)
    ensure_row_contiguous=True,
)


def eligible(x, w, scales, biases, group_size: int, bits: int, K: int, N: int) -> bool:
    """fast_qmm と同じ適格条件。M 窓の判断は呼び出し側が持つ。"""
    return (
        bits == 4
        and group_size == 64
        # split-K は 64 要素グループを 8 simdgroup で等分に歩く
        and K % 512 == 0
        and N >= N_MIN
        and x.dtype == mx.bfloat16
        and w.dtype == mx.uint32
        and w.shape == (N, K // 8)
        and scales.shape == (N, K // 64)
        and biases.shape == (N, K // 64)
        and mx.default_device() == mx.gpu
    )


def qmm_direct(x, w, scales, biases, *, group_size: int, bits: int):
    """Drop-in for `mx.quantized_matmul(..., transpose=True)`, M in 1..16.

    ゼロ埋めパディングは不要 (カーネルが M をテンプレートで持ち、実在しない
    行はレーン側で 0 を作る)。適格外は stock へ落ちる。
    """
    K = x.shape[-1]
    M = 1
    for d in x.shape[:-1]:
        M *= d
    N = w.shape[0]
    if not (
        1 <= M <= M_WIDE_MAX
        and eligible(x, w, scales, biases, group_size, bits, K, N)
    ):
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
    flat = x.reshape(M, K)
    (out,) = _KERNEL(
        inputs=[flat, w, scales, biases],
        template=[("KD", K), ("ND", N), ("MD", M)],
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
        grid=(((N + 7) // 8) * TGT, 1, 1),
        threadgroup=(TGT, 1, 1),
    )
    return out.reshape(*x.shape[:-1], N)
