# Ported from Layr-Labs/qwen-3.8-mtp-challenge Qwen35.swift E120 QMV.
# Copyright (c) 2026 Layr Labs, Inc. Licensed under the MIT License; see
# tools/reference/e120/LICENSE.
"""Legacy v4 Metal source and host geometry for the E120 affine-4/group-64 QMV."""

GROUP_SIZE = 64
BITS = 4
M_MIN = 2
M_MAX = 9
MINIMUM_TABLE_WIDTH = 4
ROWS_PER_SIMD = 4
VALUES_PER_THREAD = 16
BLOCK_SIZE = VALUES_PER_THREAD * 32
THREADGROUP = (32, 2, 1)
XSUMS_THREADGROUP = (32, 1, 1)

# Faithful transcription of Qwen35CustomQMV.activeInputGroups.  M is split
# across independent input groups; each SIMD group still owns four output rows.
INPUTS_PER_GROUP = {
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 3,
    7: 4,
    8: 4,
    9: 3,
}
ACTIVE_INPUT_GROUPS = {
    m: (m + inputs_per_group - 1) // inputs_per_group
    for m, inputs_per_group in INPUTS_PER_GROUP.items()
}


def active_input_groups(m: int) -> int:
    """Return the exact E120 X-grid threadgroup count for one width."""

    try:
        return ACTIVE_INPUT_GROUPS[m]
    except KeyError as exc:
        raise ValueError(f"E120 QMV has no width plan for M={m}") from exc


def sums_stride(m: int) -> int:
    """Return E120's cache-line-padded xsums row stride, in floats."""

    return 8 if m <= 8 else 16


def eligible_layout(
    m: int,
    k: int,
    n: int,
    w_shape: tuple[int, ...],
    scales_shape: tuple[int, ...],
    biases_shape: tuple[int, ...],
    group_size: int,
    bits: int,
) -> bool:
    """Check the E120 shape/packing contract without importing MLX."""

    if group_size != GROUP_SIZE or bits != BITS:
        return False
    if m not in INPUTS_PER_GROUP or k <= 0 or n <= 0:
        return False
    if k % BLOCK_SIZE != 0 or n % 8 != 0:
        return False
    if w_shape != (n, k * bits // 32):
        return False
    groups = k // group_size
    return scales_shape == (n, groups) and biases_shape == (n, groups)


# The v3 no-table source stays byte-for-byte independent of the v4 table
# source.  K and N are absent from every template parameter list: E120 recorded
# a compiler miscompile when K was templated.
METAL_HEADER = r"""
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
"""


# E120 USE_TABLE path.  Its only arithmetic difference from METAL_HEADER is
# that each lane's activation sum is loaded from xsums instead of recomputed
# once for every four-output-row block.
TABLE_METAL_HEADER = r"""
// Port of Layr-Labs/qwen-3.8-mtp-challenge E120 QMV USE_TABLE path.
// Copyright (c) 2026 Layr Labs, Inc. MIT License; see vendored LICENSE.
template <int NA>
inline void fastmlx_e120_qmv_wide_table(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const device bfloat16_t* x,
    const device float* xsums,
    device bfloat16_t* y,
    const int in_vec_size,
    const int out_vec_size,
    const int sums_stride,
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
        const device float* st =
            xsums + ((k / block_size) * 32 + int(simd_lid)) *
            sums_stride + first_m;
        for (int m = 0; m < NA; m++) {
            sums[m] = st[m];
        }
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
inline void fastmlx_e120_qmv_m_table(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const device bfloat16_t* x,
    const device float* xsums,
    device bfloat16_t* y,
    const int in_vec_size,
    const int out_vec_size,
    const int sums_stride,
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
        fastmlx_e120_qmv_wide_table<IPG>(
            w, scales, biases, x, xsums, y, in_vec_size, out_vec_size,
            sums_stride, first_m, out_row, simd_lid);
    } else {
        fastmlx_e120_qmv_wide_table<(TAIL >= 2 ? TAIL : 2)>(
            w, scales, biases, x, xsums, y, in_vec_size, out_vec_size,
            sums_stride, first_m, out_row, simd_lid);
    }
}
"""


XSUMS_SOURCE = r"""
    const int xs_m = x_shape[x_ndim - 2];
    const int xs_k = x_shape[x_ndim - 1];
    const int xs_stride = xs_m <= 8 ? 8 : 16;
    const uint3 xs_gid = thread_position_in_grid;
    const int xs_lane = int(xs_gid.x);
    const int xs_kb = int(xs_gid.y);
    const int xs_row = int(xs_gid.z);
    const device bfloat16_t* xm =
        x + xs_row * xs_k + xs_kb * 512 + xs_lane * 16;
    float s = 0.0f;
    for (int i = 0; i < 4; i++) {
        const vec<bfloat16_t, 4> xv =
            *reinterpret_cast<const device vec<bfloat16_t, 4>*>(xm + 4 * i);
        s += xv[0] + xv[1] + xv[2] + xv[3];
    }
    xsums[(xs_kb * 32 + xs_lane) * xs_stride + xs_row] = s;
"""


def build_source() -> str:
    """Return the shared no-table body for every E120-supported M."""

    cases = "\n".join(
        f"""        case {m}:
            fastmlx_e120_qmv_m<{m}, {inputs_per_group}>(
                w, scales, biases, x, y, qmv_k, qmv_n,
                qmv_gx, qmv_out_row, qmv_lid);
            break;"""
        for m, inputs_per_group in INPUTS_PER_GROUP.items()
    )
    return f"""
    const int qmv_m = x_shape[x_ndim - 2];
    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    const uint3 qmv_tid = threadgroup_position_in_grid;
    const uint qmv_lid = thread_index_in_simdgroup;
    const uint qmv_sgid = simdgroup_index_in_threadgroup;
    const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
    const int qmv_gx = int(qmv_tid.x);
    switch (qmv_m) {{
{cases}
        default:
            break;
    }}
"""


def build_table_source() -> str:
    """Return the E120 body that consumes one precomputed xsums table."""

    cases = "\n".join(
        f"""        case {m}:
            fastmlx_e120_qmv_m_table<{m}, {inputs_per_group}>(
                w, scales, biases, x, xsums, y, qmv_k, qmv_n, qmv_stride,
                qmv_gx, qmv_out_row, qmv_lid);
            break;"""
        for m, inputs_per_group in INPUTS_PER_GROUP.items()
    )
    return f"""
    const int qmv_m = x_shape[x_ndim - 2];
    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    const int qmv_stride = qmv_m <= 8 ? 8 : 16;
    const uint3 qmv_tid = threadgroup_position_in_grid;
    const uint qmv_lid = thread_index_in_simdgroup;
    const uint qmv_sgid = simdgroup_index_in_threadgroup;
    const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
    const int qmv_gx = int(qmv_tid.x);
    switch (qmv_m) {{
{cases}
        default:
            break;
    }}
"""


__all__ = [
    "ACTIVE_INPUT_GROUPS",
    "BITS",
    "BLOCK_SIZE",
    "GROUP_SIZE",
    "INPUTS_PER_GROUP",
    "METAL_HEADER",
    "M_MAX",
    "M_MIN",
    "MINIMUM_TABLE_WIDTH",
    "ROWS_PER_SIMD",
    "TABLE_METAL_HEADER",
    "THREADGROUP",
    "VALUES_PER_THREAD",
    "XSUMS_SOURCE",
    "XSUMS_THREADGROUP",
    "active_input_groups",
    "build_source",
    "build_table_source",
    "eligible_layout",
    "sums_stride",
]
