#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

typedef bfloat bfloat16_t;


template <typename T, int K, int N>
[[kernel]] void custom_kernel_v_uint4_m16_bf16(
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
    constexpr int C_TILES = 2;

    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint n0 = threadgroup_position_in_grid.z * 8;

    threadgroup float partials[SPLITS * C_TILES * 64];

    simdgroup_matrix<float, 8, 8> c0;
    c0.thread_elements()[0] = 0.0f;
    c0.thread_elements()[1] = 0.0f;
    simdgroup_matrix<float, 8, 8> c1;
    c1.thread_elements()[0] = 0.0f;
    c1.thread_elements()[1] = 0.0f;

    const ushort quad = lane / 4;
    const ushort frag_row = (quad & 4) + (lane / 2) % 4;
    const ushort frag_col = (quad & 2) * 2 + (lane % 2) * 2;

    const int groups = K / QGROUP;
    const int packed_stride = K / 8;
    const int scale_stride = groups;

    for (int group = (int)sg; group < groups; group += SPLITS) {
        float scale_by_col = 0.0f;
        float bias_by_col = 0.0f;
        uint4 packed_lo = uint4(0);
        uint4 packed_hi = uint4(0);
        if (lane < 8) {
            scale_by_col = (float)scales[(size_t)(n0 + lane) * scale_stride + group];
            bias_by_col = (float)biases[(size_t)(n0 + lane) * scale_stride + group];
            const device uint4* wv = (const device uint4*)(
                w + (size_t)(n0 + lane) * packed_stride + group * (QGROUP / 8));
            packed_lo = wv[0];
            packed_hi = wv[1];
        }
        const float scale0 = simd_shuffle(scale_by_col, frag_col);
        const float scale1 = simd_shuffle(scale_by_col, frag_col + 1);
        const float bias0 = simd_shuffle(bias_by_col, frag_col);
        const float bias1 = simd_shuffle(bias_by_col, frag_col + 1);

        uint words[8];
        words[0] = packed_lo.x; words[1] = packed_lo.y;
        words[2] = packed_lo.z; words[3] = packed_lo.w;
        words[4] = packed_hi.x; words[5] = packed_hi.y;
        words[6] = packed_hi.z; words[7] = packed_hi.w;

        #pragma unroll
        for (int kt = 0; kt < QGROUP / 8; ++kt) {
            const int k_base = group * QGROUP + kt * 8;
            const uint packed0 = simd_shuffle(words[kt], frag_col);
            const uint packed1 = simd_shuffle(words[kt], frag_col + 1);

            simdgroup_matrix<half, 8, 8> bmat;
            bmat.thread_elements()[0] = half(
                scale0 * (float)((packed0 >> (QBITS * frag_row)) & 0xF) + bias0);
            bmat.thread_elements()[1] = half(
                scale1 * (float)((packed1 >> (QBITS * frag_row)) & 0xF) + bias1);

            simdgroup_matrix<half, 8, 8> a0;
            simdgroup_matrix<half, 8, 8> a1;

            a0.thread_elements()[0] = frag_row < 8
                ? half(x[(size_t)(0 + frag_row) * K + k_base + frag_col])
                : half(0);
            a0.thread_elements()[1] = frag_row < 8
                ? half(x[(size_t)(0 + frag_row) * K + k_base + frag_col + 1])
                : half(0);

            a1.thread_elements()[0] = frag_row < 8
                ? half(x[(size_t)(8 + frag_row) * K + k_base + frag_col])
                : half(0);
            a1.thread_elements()[1] = frag_row < 8
                ? half(x[(size_t)(8 + frag_row) * K + k_base + frag_col + 1])
                : half(0);
            simdgroup_multiply_accumulate(c0, a0, bmat, c0);
            simdgroup_multiply_accumulate(c1, a1, bmat, c1);
        }
    }

    partials[(sg * C_TILES * 64) + lane * 2] = c0.thread_elements()[0];
    partials[(sg * C_TILES * 64) + lane * 2 + 1] = c0.thread_elements()[1];
    partials[((sg * C_TILES + 1) * 64) + lane * 2] = c1.thread_elements()[0];
    partials[((sg * C_TILES + 1) * 64) + lane * 2 + 1] = c1.thread_elements()[1];
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
        const uint row1 = 8 + frag_row;
        if (row1 < 16) {
            float total0 = 0.0f;
            float total1 = 0.0f;
            #pragma unroll
            for (int split = 0; split < SPLITS; ++split) {
                total0 += partials[((split * C_TILES + 1) * 64) + lane * 2];
                total1 += partials[((split * C_TILES + 1) * 64) + lane * 2 + 1];
            }
            y[(size_t)row1 * N + n0 + frag_col] = (T)total0;
            y[(size_t)row1 * N + n0 + frag_col + 1] = (T)total1;
        }
    }

}

template [[host_name("custom_kernel_v_uint4_m16_bf16_bfloat16_t_5120_17408")]] [[kernel]] decltype(custom_kernel_v_uint4_m16_bf16<bfloat16_t, 5120, 17408>) custom_kernel_v_uint4_m16_bf16<bfloat16_t, 5120, 17408>;
