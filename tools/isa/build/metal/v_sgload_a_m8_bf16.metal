#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

typedef bfloat bfloat16_t;


template <typename T, int K, int N>
[[kernel]] void custom_kernel_v_sgload_a_m8_bf16(
  const device bfloat16_t* x [[buffer(0)]],
  const device uint32_t* w [[buffer(1)]],
  const device bfloat16_t* scales [[buffer(2)]],
  const device bfloat16_t* biases [[buffer(3)]],
  device bfloat16_t* y [[buffer(4)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    constexpr int QGROUP = 64;
    constexpr int QBITS = 4;
    constexpr int SPLITS = 8;
    constexpr int C_TILES = 1;

    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint n0 = threadgroup_position_in_grid.z * 8;

    threadgroup float partials[SPLITS * C_TILES * 64];

    simdgroup_matrix<float, 8, 8> c0;
    c0.thread_elements()[0] = 0.0f;
    c0.thread_elements()[1] = 0.0f;

    const ushort quad = lane / 4;
    const ushort frag_row = (quad & 4) + (lane / 2) % 4;
    const ushort frag_col = (quad & 2) * 2 + (lane % 2) * 2;

    const int groups = K / QGROUP;
    const int packed_stride = K / 8;
    const int scale_stride = groups;

    for (int group = (int)sg; group < groups; group += SPLITS) {
        float scale_by_col = 0.0f;
        float bias_by_col = 0.0f;
        if (lane < 8) {
            scale_by_col = (float)scales[(size_t)(n0 + lane) * scale_stride + group];
            bias_by_col = (float)biases[(size_t)(n0 + lane) * scale_stride + group];
        }
        const float scale0 = simd_shuffle(scale_by_col, frag_col);
        const float scale1 = simd_shuffle(scale_by_col, frag_col + 1);
        const float bias0 = simd_shuffle(bias_by_col, frag_col);
        const float bias1 = simd_shuffle(bias_by_col, frag_col + 1);

        #pragma unroll
        for (int kt = 0; kt < QGROUP / 8; ++kt) {
            const int k_base = group * QGROUP + kt * 8;
            uint packed_by_col = 0;
            if (lane < 8) {
                packed_by_col = w[(size_t)(n0 + lane) * packed_stride + k_base / 8];
            }
            const uint packed0 = simd_shuffle(packed_by_col, frag_col);
            const uint packed1 = simd_shuffle(packed_by_col, frag_col + 1);

            simdgroup_matrix<bfloat16_t, 8, 8> bmat;
            bmat.thread_elements()[0] = bfloat16_t(
                scale0 * (float)((packed0 >> (QBITS * frag_row)) & 0xF) + bias0);
            bmat.thread_elements()[1] = bfloat16_t(
                scale1 * (float)((packed1 >> (QBITS * frag_row)) & 0xF) + bias1);

            simdgroup_matrix<bfloat16_t, 8, 8> a0;
            simdgroup_load(a0, x + k_base, K);
            simdgroup_multiply_accumulate(c0, a0, bmat, c0);

        }
    }

    partials[(sg * C_TILES * 64) + lane * 2] = c0.thread_elements()[0];
    partials[(sg * C_TILES * 64) + lane * 2 + 1] = c0.thread_elements()[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        if (frag_row < 8) {
            float total0 = 0.0f;
            float total1 = 0.0f;
            #pragma unroll
            for (int split = 0; split < SPLITS; ++split) {
                total0 += partials[(split * C_TILES * 64) + lane * 2];
                total1 += partials[(split * C_TILES * 64) + lane * 2 + 1];
            }
            y[(size_t)frag_row * N + n0 + frag_col] = (T)total0;
            y[(size_t)frag_row * N + n0 + frag_col + 1] = (T)total1;
        }
    }

}

template [[host_name("custom_kernel_v_sgload_a_m8_bf16_bfloat16_t_5120_17408")]] [[kernel]] decltype(custom_kernel_v_sgload_a_m8_bf16<bfloat16_t, 5120, 17408>) custom_kernel_v_sgload_a_m8_bf16<bfloat16_t, 5120, 17408>;
