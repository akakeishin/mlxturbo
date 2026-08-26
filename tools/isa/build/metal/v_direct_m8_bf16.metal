#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

typedef bfloat bfloat16_t;


template <typename T, int K, int N>
[[kernel]] void custom_kernel_v_direct_m8_bf16(
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
        const uint col_a = n0 + frag_col;
        const uint col_b = n0 + frag_col + 1;
        const device uint4* wa = (const device uint4*)(
            w + (size_t)col_a * packed_stride + group * (QGROUP / 8));
        const device uint4* wb = (const device uint4*)(
            w + (size_t)col_b * packed_stride + group * (QGROUP / 8));
        const uint4 pa_lo = wa[0];
        const uint4 pa_hi = wa[1];
        const uint4 pb_lo = wb[0];
        const uint4 pb_hi = wb[1];
        const float scale0 = (float)scales[(size_t)col_a * scale_stride + group];
        const float scale1 = (float)scales[(size_t)col_b * scale_stride + group];
        const float bias0 = (float)biases[(size_t)col_a * scale_stride + group];
        const float bias1 = (float)biases[(size_t)col_b * scale_stride + group];

        uint wa_words[8];
        wa_words[0] = pa_lo.x; wa_words[1] = pa_lo.y;
        wa_words[2] = pa_lo.z; wa_words[3] = pa_lo.w;
        wa_words[4] = pa_hi.x; wa_words[5] = pa_hi.y;
        wa_words[6] = pa_hi.z; wa_words[7] = pa_hi.w;
        uint wb_words[8];
        wb_words[0] = pb_lo.x; wb_words[1] = pb_lo.y;
        wb_words[2] = pb_lo.z; wb_words[3] = pb_lo.w;
        wb_words[4] = pb_hi.x; wb_words[5] = pb_hi.y;
        wb_words[6] = pb_hi.z; wb_words[7] = pb_hi.w;

        #pragma unroll
        for (int kt = 0; kt < QGROUP / 8; ++kt) {
            const int k_base = group * QGROUP + kt * 8;
            simdgroup_matrix<bfloat16_t, 8, 8> bmat;
            bmat.thread_elements()[0] = bfloat16_t(
                scale0 * (float)((wa_words[kt] >> (QBITS * frag_row)) & 0xF)
                + bias0);
            bmat.thread_elements()[1] = bfloat16_t(
                scale1 * (float)((wb_words[kt] >> (QBITS * frag_row)) & 0xF)
                + bias1);

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

template [[host_name("custom_kernel_v_direct_m8_bf16_bfloat16_t_5120_17408")]] [[kernel]] decltype(custom_kernel_v_direct_m8_bf16<bfloat16_t, 5120, 17408>) custom_kernel_v_direct_m8_bf16<bfloat16_t, 5120, 17408>;
