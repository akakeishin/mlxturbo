#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

typedef bfloat bfloat16_t;


template <typename T, int K, int N>
[[kernel]] void custom_kernel_v_e120_r2na8_m8_bf16(
  const device bfloat16_t* x [[buffer(0)]],
  const constant int* x_shape [[buffer(1)]],
  const constant int& x_ndim [[buffer(2)]],
  const device uint32_t* w [[buffer(3)]],
  const constant int* w_shape [[buffer(4)]],
  const device bfloat16_t* scales [[buffer(5)]],
  const device bfloat16_t* biases [[buffer(6)]],
  device bfloat16_t* y [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    // E120 QMV, width 8: inputs-per-group 8, 2 output rows
    // per simdgroup, so the weight column is read 1x for this width.
    constexpr int NA = 8;
    constexpr int rows_per_simd = 2;
    constexpr int values_per_thread = 16;
    constexpr int block_size = values_per_thread * 32;
    constexpr int bytes_per_lane = 8;
    typedef vec<float, NA> VF;

    const int in_vec_size = x_shape[x_ndim - 1];
    const int out_vec_size = w_shape[0];
    const int in_vec_size_w = in_vec_size / 2;
    const int in_vec_size_g = in_vec_size / 64;

    const uint simd_lid = thread_index_in_simdgroup;
    const uint sgid = simdgroup_index_in_threadgroup;
    const uint3 e_tid = threadgroup_position_in_grid;
    const int out_row = int(e_tid.y) * (rows_per_simd * 2)
        + int(sgid) * rows_per_simd;
    const int first_m = int(e_tid.x) * NA;
    const device float* e_xsums = reinterpret_cast<const device float*>(x);
    if (first_m >= 8) {
        return;
    }

    VF acc[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
        acc[r] = VF(0.0f);
    }

    for (int k = 0; k < in_vec_size; k += block_size) {
        thread uint16_t packed[rows_per_simd][4];
        thread float scale_local[rows_per_simd];
        thread float bias_local[rows_per_simd];
        #pragma unroll
        for (int r = 0; r < rows_per_simd; r++) {
            const int row = out_row + r;
            const device uint16_t* ws =
                reinterpret_cast<const device uint16_t*>(
                    reinterpret_cast<const device uint8_t*>(w) +
                    row * in_vec_size_w + k / 2 +
                    simd_lid * bytes_per_lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                packed[r][i] = ws[i];
            }
            const int group_index =
                row * in_vec_size_g + k / 64 + int(simd_lid) / 4;
            scale_local[r] = (float)scales[group_index];
            bias_local[r] = (float)biases[group_index];
        }

        VF sums;
        const device float* e_st = e_xsums
            + ((k / block_size) * 32 + int(simd_lid)) * 8 + first_m;
        #pragma unroll
        for (int mi = 0; mi < NA; mi++) {
            sums[mi] = e_st[mi];
        }
        VF partial[rows_per_simd];
        #pragma unroll
        for (int r = 0; r < rows_per_simd; r++) {
            partial[r] = VF(0.0f);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            VF a0, a1, a2, a3;
            #pragma unroll
            for (int mi = 0; mi < NA; mi++) {
                const device bfloat16_t* xm =
                    (const device bfloat16_t*)x
                    + (first_m + mi) * in_vec_size + k
                    + simd_lid * values_per_thread + 4 * i;
                const vec<bfloat16_t, 4> xv =
                    *reinterpret_cast<const device vec<bfloat16_t, 4>*>(xm);
                a0[mi] = static_cast<float>(xv[0]);
                a1[mi] = static_cast<float>(xv[1]);
                a2[mi] = static_cast<float>(xv[2]);
                a3[mi] = static_cast<float>(xv[3]);
            }
            #pragma unroll
            for (int r = 0; r < rows_per_simd; r++) {
                partial[r] += (a0 * (packed[r][i] & 0x000f) +
                               a1 * ((packed[r][i] >> 4) & 0x000f) +
                               a2 * ((packed[r][i] >> 8) & 0x000f) +
                               a3 * ((packed[r][i] >> 12) & 0x000f));
            }
        }
        #pragma unroll
        for (int r = 0; r < rows_per_simd; r++) {
            acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
        }
    }

    for (int r = 0; r < rows_per_simd; r++) {
        for (int mi = 0; mi < NA; mi++) {
            const float reduced = simd_sum(acc[r][mi]);
            if (simd_lid == 0) {
                y[(first_m + mi) * out_vec_size + out_row + r] =
                    static_cast<T>(reduced);
            }
        }
    }

}

template [[host_name("custom_kernel_v_e120_r2na8_m8_bf16_bfloat16_t_5120_17408")]] [[kernel]] decltype(custom_kernel_v_e120_r2na8_m8_bf16<bfloat16_t, 5120, 17408>) custom_kernel_v_e120_r2na8_m8_bf16<bfloat16_t, 5120, 17408>;
