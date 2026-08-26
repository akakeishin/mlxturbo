#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

typedef bfloat bfloat16_t;


template <int KD, int ND, int MD>
[[kernel]] void custom_kernel_ref_fastqmm_m8(
  const device bfloat16_t* x [[buffer(0)]],
  const device uint32_t* w [[buffer(1)]],
  const device bfloat16_t* sc [[buffer(2)]],
  const device bfloat16_t* bi [[buffer(3)]],
  device bfloat16_t* out [[buffer(4)]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

    const int K = KD, N = ND, M = MD;
    const int KPS = KD / 8;                 // 심드그룹당 K 구간

    uint tid  = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg   = tid >> 5;
    uint lane = tid & 31;

    int n0 = (int)tgid * 8;                 // threadgroup 하나가 출력열 8개

    // x 는 device 에서 직접 MMA 로 읽는다(bf16 입력 + float 누산). 스테이징이 없어져
    // threadgroup 메모리가 10KB 로 내려가고, 무엇보다 임계경로가 짧아진다.
    threadgroup bfloat16_t bs[8 * 512];     // 심드그룹별 64k x 8n, 8KB
    threadgroup float red[8 * 64];          // 심드그룹 간 합산, 2KB

    simdgroup_matrix<float, 8, 8> C = simdgroup_matrix<float, 8, 8>(0);
    threadgroup bfloat16_t* bt = bs + sg * 512;

    // ── split-K: 8개 심드그룹이 K 를 8등분해 각자 짧은 직렬 루프를 돈다.
    //    (종전 판본은 한 threadgroup 이 K 전체를 순회해 배리어 160쌍이 임계경로에 놓였다)
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

        simdgroup_matrix<bfloat16_t, 8, 8> A, B;
        for (int kt = 0; kt < 8; ++kt) {
            simdgroup_load(A, x + ka + kt * 8, K);   // x[0:8, ka+8kt ..] — 패딩된 8행
            simdgroup_load(B, bt + kt * 64, 8);
            simdgroup_multiply_accumulate(C, A, B, C);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(C, red + sg * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 8개 부분합을 더해 쓴다.
    for (int i = (int)tid; i < 64; i += 256) {
        int m = i >> 3, j = i & 7;
        int n = n0 + j;
        if (m < M && n < N) {
            float v = 0.0f;
            for (int q = 0; q < 8; ++q) v += red[q * 64 + i];
            out[(size_t)m * N + n] = (bfloat16_t)v;
        }
    }

}

template [[host_name("custom_kernel_ref_fastqmm_m8_5120_17408_8")]] [[kernel]] decltype(custom_kernel_ref_fastqmm_m8<5120, 17408, 8>) custom_kernel_ref_fastqmm_m8<5120, 17408, 8>;
