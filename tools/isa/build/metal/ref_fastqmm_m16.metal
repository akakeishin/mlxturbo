#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

typedef bfloat bfloat16_t;


template <int KD, int ND, int MD>
[[kernel]] void custom_kernel_ref_fastqmm_m16(
  const device bfloat16_t* x [[buffer(0)]],
  const device uint32_t* w [[buffer(1)]],
  const device bfloat16_t* sc [[buffer(2)]],
  const device bfloat16_t* bi [[buffer(3)]],
  device bfloat16_t* out [[buffer(4)]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    const int K = KD, N = ND, M = MD;
    const int KPS = KD / 8;

    uint tid  = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg   = tid >> 5;
    uint lane = tid & 31;

    int n0 = (int)tgid * 8;

    threadgroup bfloat16_t bs[8 * 512];     // B 타일 — 두 행-타일이 공유한다(요점)
    threadgroup float red[8 * 128];         // 심드그룹 8 x 행타일 2 x 64, 4KB

    simdgroup_matrix<float, 8, 8> C0 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> C1 = simdgroup_matrix<float, 8, 8>(0);
    threadgroup bfloat16_t* bt = bs + sg * 512;

    int kbeg = (int)sg * KPS;
    for (int kk = 0; kk < KPS; kk += 64) {
        int ka = kbeg + kk;
        int j  = (int)(lane & 7);
        int kq = (int)(lane >> 3);
        int n  = n0 + j;
        if (n < N) {
            int g = ka >> 6;
            float s  = (float)sc[(size_t)n * (K / 64) + g];
            float bb = (float)bi[(size_t)n * (K / 64) + g];
            const device uint* wr = w + (size_t)n * (K / 8) + (ka >> 3) + kq * 2;
            uint p0 = wr[0], p1 = wr[1];
            for (int t = 0; t < 8; ++t)
                bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)((float)((p0 >> (4 * t)) & 15u) * s + bb);
            for (int t = 0; t < 8; ++t)
                bt[(kq * 16 + 8 + t) * 8 + j] = (bfloat16_t)((float)((p1 >> (4 * t)) & 15u) * s + bb);
        } else {
            for (int t = 0; t < 16; ++t) bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)0;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<bfloat16_t, 8, 8> A0, A1, B;
        for (int kt = 0; kt < 8; ++kt) {
            simdgroup_load(B,  bt + kt * 64, 8);
            simdgroup_load(A0, x + ka + kt * 8, K);              // 행 0-7
            simdgroup_multiply_accumulate(C0, A0, B, C0);
            simdgroup_load(A1, x + (size_t)8 * K + ka + kt * 8, K);  // 행 8-15
            simdgroup_multiply_accumulate(C1, A1, B, C1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(C0, red + sg * 128, 8);
    simdgroup_store(C1, red + sg * 128 + 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = (int)tid; i < 128; i += 256) {
        int m = i >> 3, j = i & 7;          // m 0-15 (0-7 은 C0, 8-15 는 C1 구간)
        int mm = (m < 8) ? m : (m - 8);
        int slot = (m < 8) ? (mm * 8 + j) : (64 + mm * 8 + j);
        int mrow = (m < 8) ? m : m;
        int n = n0 + j;
        if (mrow < M && n < N) {
            float v = 0.0f;
            for (int q = 0; q < 8; ++q) v += red[q * 128 + slot];
            out[(size_t)mrow * N + n] = (bfloat16_t)v;
        }
    }

}

template [[host_name("custom_kernel_ref_fastqmm_m16_5120_17408_16")]] [[kernel]] decltype(custom_kernel_ref_fastqmm_m16<5120, 17408, 16>) custom_kernel_ref_fastqmm_m16<5120, 17408, 16>;
