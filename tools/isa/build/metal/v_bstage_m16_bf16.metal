#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

typedef bfloat bfloat16_t;


template <typename T, int K, int N>
[[kernel]] void custom_kernel_v_bstage_m16_bf16(
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

    threadgroup bfloat16_t bslab[SPLITS * QGROUP * 8];
    threadgroup bfloat16_t* bt = bslab + sg * (QGROUP * 8);

    for (int group = (int)sg; group < groups; group += SPLITS) {
        const uint col = lane & 7;      // output column within the 8-wide tile
        const uint half_sel = lane >> 3;  // 4 lanes each cover 16 k-values
        {
            const float s = (float)scales[
                (size_t)(n0 + col) * scale_stride + group];
            const float b = (float)biases[
                (size_t)(n0 + col) * scale_stride + group];
            const device uint* wr = w + (size_t)(n0 + col) * packed_stride
                + group * (QGROUP / 8) + half_sel * 2;
            const uint p0 = wr[0];
            const uint p1 = wr[1];
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                bt[(half_sel * 16 + t) * 8 + col] =
                    bfloat16_t((float)((p0 >> (QBITS * t)) & 0xFu) * s + b);
            }
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                bt[(half_sel * 16 + 8 + t) * 8 + col] =
                    bfloat16_t((float)((p1 >> (QBITS * t)) & 0xFu) * s + b);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int kt = 0; kt < QGROUP / 8; ++kt) {
            const int k_base = group * QGROUP + kt * 8;
            simdgroup_matrix<bfloat16_t, 8, 8> bmat;
            simdgroup_load(bmat, bt + kt * 64, 8);
            simdgroup_matrix<bfloat16_t, 8, 8> a0;
            simdgroup_load(a0, x + k_base, K);
            simdgroup_multiply_accumulate(c0, a0, bmat, c0);
            simdgroup_matrix<bfloat16_t, 8, 8> a1;
            simdgroup_load(a1, x + (size_t)8 * K + k_base, K);
            simdgroup_multiply_accumulate(c1, a1, bmat, c1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
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

template [[host_name("custom_kernel_v_bstage_m16_bf16_bfloat16_t_5120_17408")]] [[kernel]] decltype(custom_kernel_v_bstage_m16_bf16<bfloat16_t, 5120, 17408>) custom_kernel_v_bstage_m16_bf16<bfloat16_t, 5120, 17408>;
