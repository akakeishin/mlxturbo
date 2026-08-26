#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

typedef bfloat bfloat16_t;


// Port of Layr-Labs/qwen-3.8-mtp-challenge E120 QMV.
// Copyright (c) 2026 Layr Labs, Inc. MIT License; see vendored LICENSE.
template <int NA>
inline void fastmlx_e120_qmv_wide(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const device bfloat16_t* x,
    device bfloat16_t* y,
    const int in_vec_size,
    const int out_vec_size,
    int first_m,
    int out_row,
    uint simd_lid
) {
    typedef vec<float, NA> VF;
    constexpr int rows_per_simd = 4;
    constexpr int values_per_thread = 16;
    constexpr int block_size = values_per_thread * 32;
    constexpr int bytes_per_lane = 8;
    const int in_vec_size_w = in_vec_size / 2;
    const int in_vec_size_g = in_vec_size / 64;

    VF acc[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
        acc[r] = VF(0.0f);
    }

    for (int k = 0; k < in_vec_size; k += block_size) {
        thread uint16_t packed[rows_per_simd][4];
        thread float scale_local[rows_per_simd];
        thread float bias_local[rows_per_simd];
        for (int r = 0; r < rows_per_simd; r++) {
            const int row = out_row + r;
            const device uint16_t* ws =
                reinterpret_cast<const device uint16_t*>(
                    reinterpret_cast<const device uint8_t*>(w) +
                    row * in_vec_size_w + k / 2 +
                    simd_lid * bytes_per_lane);
            for (int i = 0; i < 4; i++) {
                packed[r][i] = ws[i];
            }
            const int group_index =
                row * in_vec_size_g + k / 64 + int(simd_lid) / 4;
            scale_local[r] = scales[group_index];
            bias_local[r] = biases[group_index];
        }

        VF sums = VF(0.0f);
        VF partial[rows_per_simd];
        for (int r = 0; r < rows_per_simd; r++) {
            partial[r] = VF(0.0f);
        }
        for (int i = 0; i < 4; i++) {
            VF a0, a1, a2, a3;
            for (int m = 0; m < NA; m++) {
                const device bfloat16_t* xm =
                    x + (first_m + m) * in_vec_size + k +
                    simd_lid * values_per_thread + 4 * i;
                const vec<bfloat16_t, 4> xv =
                    *reinterpret_cast<const device vec<bfloat16_t, 4>*>(xm);
                a0[m] = static_cast<float>(xv[0]);
                a1[m] = static_cast<float>(xv[1]);
                a2[m] = static_cast<float>(xv[2]);
                a3[m] = static_cast<float>(xv[3]);
                sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
            }
            for (int r = 0; r < rows_per_simd; r++) {
                partial[r] += (a0 * (packed[r][i] & 0x000f) +
                               a1 * ((packed[r][i] >> 4) & 0x000f) +
                               a2 * ((packed[r][i] >> 8) & 0x000f) +
                               a3 * ((packed[r][i] >> 12) & 0x000f));
            }
        }
        for (int r = 0; r < rows_per_simd; r++) {
            acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
        }
    }

    for (int r = 0; r < rows_per_simd; r++) {
        for (int m = 0; m < NA; m++) {
            const float reduced = simd_sum(acc[r][m]);
            if (simd_lid == 0) {
                y[(first_m + m) * out_vec_size + out_row + r] =
                    static_cast<bfloat16_t>(reduced);
            }
        }
    }
}

template <int M, int IPG>
inline void fastmlx_e120_qmv_m(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const device bfloat16_t* x,
    device bfloat16_t* y,
    const int in_vec_size,
    const int out_vec_size,
    int group_x,
    int out_row,
    uint simd_lid
) {
    static_assert(M % IPG != 1, "a one-input tail group is not built");
    constexpr int TAIL = M % IPG;
    const int first_m = group_x * IPG;
    if (first_m >= M) {
        return;
    }
    if (TAIL == 0 || M - first_m >= IPG) {
        fastmlx_e120_qmv_wide<IPG>(
            w, scales, biases, x, y, in_vec_size, out_vec_size,
            first_m, out_row, simd_lid);
    } else {
        fastmlx_e120_qmv_wide<(TAIL >= 2 ? TAIL : 2)>(
            w, scales, biases, x, y, in_vec_size, out_vec_size,
            first_m, out_row, simd_lid);
    }
}

[[kernel]] void custom_kernel_current_qmm_skinny(
  const device uint32_t* w [[buffer(0)]],
  const constant int* w_shape [[buffer(1)]],
  const device bfloat16_t* scales [[buffer(2)]],
  const device bfloat16_t* biases [[buffer(3)]],
  const device bfloat16_t* x [[buffer(4)]],
  const constant int* x_shape [[buffer(5)]],
  const constant int& x_ndim [[buffer(6)]],
  device bfloat16_t* y [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    const int qmv_m = x_shape[x_ndim - 2];
    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    const uint3 qmv_tid = threadgroup_position_in_grid;
    const uint qmv_lid = thread_index_in_simdgroup;
    const uint qmv_sgid = simdgroup_index_in_threadgroup;
    const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
    const int qmv_gx = int(qmv_tid.x);
    switch (qmv_m) {
        case 2:
            fastmlx_e120_qmv_m<2, 2>(
                w, scales, biases, x, y, qmv_k, qmv_n,
                qmv_gx, qmv_out_row, qmv_lid);
            break;
        case 3:
            fastmlx_e120_qmv_m<3, 3>(
                w, scales, biases, x, y, qmv_k, qmv_n,
                qmv_gx, qmv_out_row, qmv_lid);
            break;
        case 4:
            fastmlx_e120_qmv_m<4, 4>(
                w, scales, biases, x, y, qmv_k, qmv_n,
                qmv_gx, qmv_out_row, qmv_lid);
            break;
        case 5:
            fastmlx_e120_qmv_m<5, 5>(
                w, scales, biases, x, y, qmv_k, qmv_n,
                qmv_gx, qmv_out_row, qmv_lid);
            break;
        case 6:
            fastmlx_e120_qmv_m<6, 3>(
                w, scales, biases, x, y, qmv_k, qmv_n,
                qmv_gx, qmv_out_row, qmv_lid);
            break;
        case 7:
            fastmlx_e120_qmv_m<7, 4>(
                w, scales, biases, x, y, qmv_k, qmv_n,
                qmv_gx, qmv_out_row, qmv_lid);
            break;
        case 8:
            fastmlx_e120_qmv_m<8, 4>(
                w, scales, biases, x, y, qmv_k, qmv_n,
                qmv_gx, qmv_out_row, qmv_lid);
            break;
        case 9:
            fastmlx_e120_qmv_m<9, 3>(
                w, scales, biases, x, y, qmv_k, qmv_n,
                qmv_gx, qmv_out_row, qmv_lid);
            break;
        default:
            break;
    }

}
