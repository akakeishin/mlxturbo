# SNAPSHOT -- do not edit.
# fastmlx/kernels/_qmm_skinny_mma_source.py as of the commit that produced the
# numbers in docs/GATE-RESULTS-A2.md (m=8 bf16 = 1.04x vs mlx).  The working
# copy of that file was replaced by a different kernel design while this ISA
# study was running, so the study keeps its own copy to stay reproducible.
"""Pure source builder for the clean-room skinny quantized MMA kernel.

This file is derived from the Phase A2 design in ``docs/PLAN.md`` and the
public Metal SIMD-group matrix API.  It deliberately has no dependency on the
vendored ``fast_qmm.py`` implementation.
"""

GROUP_SIZE = 64
BITS = 4
M_MIN = 6
M_MAX = 16
MMA_TILE = 8
SPLIT_K = 8


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
    """Check the shape/quantization contract without importing MLX."""

    if group_size != GROUP_SIZE or bits != BITS:
        return False
    if not (M_MIN <= m <= M_MAX) or k <= 0 or n <= 0:
        return False
    if k % GROUP_SIZE != 0 or n % MMA_TILE != 0:
        return False
    if w_shape != (n, k * bits // 32):
        return False
    groups = k // group_size
    return scales_shape == (n, groups) and biases_shape == (n, groups)


def _fragment_load(
    name: str,
    m: int,
    row_base: int,
    fp16_input: bool,
) -> str:
    """Build a full fp16 simdgroup_load or guarded direct device loads."""

    rows = min(MMA_TILE, max(0, m - row_base))
    if rows == MMA_TILE and fp16_input:
        return (
            f"            simdgroup_load({name}, "
            f"x + (size_t){row_base} * K + k_base, K);"
        )
    return f"""
            // Read x straight from device memory.  BF16 input is exactly
            // representable as half for the normed activation range, while
            // half preserves three more dequantized-weight mantissa bits.
            {name}.thread_elements()[0] = frag_row < {rows}
                ? half(x[(size_t)({row_base} + frag_row) * K + k_base + frag_col])
                : half(0);
            {name}.thread_elements()[1] = frag_row < {rows}
                ? half(x[(size_t)({row_base} + frag_row) * K + k_base + frag_col + 1])
                : half(0);"""


def build_source(m: int, fp16_input: bool = False) -> str:
    """Return a Metal body specialized for one M in [6, 16]."""

    if not M_MIN <= m <= M_MAX:
        raise ValueError(f"M must be in [{M_MIN}, {M_MAX}], got {m}")
    c_tiles = 1 if m <= MMA_TILE else 2
    a0_load = _fragment_load("a0", m, 0, fp16_input)
    a1_decl = (
        "            simdgroup_matrix<half, 8, 8> a1;"
        if c_tiles == 2
        else ""
    )
    a1_load = (
        _fragment_load("a1", m, MMA_TILE, fp16_input) if c_tiles == 2 else ""
    )
    c1_decl = """
    simdgroup_matrix<float, 8, 8> c1;
    c1.thread_elements()[0] = 0.0f;
    c1.thread_elements()[1] = 0.0f;""" if c_tiles == 2 else ""
    c1_mma = (
        "            simdgroup_multiply_accumulate(c1, a1, bmat, c1);"
        if c_tiles == 2
        else ""
    )
    c1_partial = """
    partials[((sg * C_TILES + 1) * 64) + lane * 2] =
        c1.thread_elements()[0];
    partials[((sg * C_TILES + 1) * 64) + lane * 2 + 1] =
        c1.thread_elements()[1];""" if c_tiles == 2 else ""
    c1_reduce = f"""
        const uint row1 = 8 + frag_row;
        if (row1 < {m}) {{
            float total0 = 0.0f;
            float total1 = 0.0f;
            #pragma unroll
            for (int split = 0; split < SPLITS; ++split) {{
                total0 += partials[((split * C_TILES + 1) * 64) + lane * 2];
                total1 += partials[((split * C_TILES + 1) * 64) + lane * 2 + 1];
            }}
            y[(size_t)row1 * N + n0 + frag_col] = (T)total0;
            y[(size_t)row1 * N + n0 + frag_col + 1] = (T)total1;
        }}""" if c_tiles == 2 else ""

    return f"""
    constexpr int QGROUP = {GROUP_SIZE};
    constexpr int QBITS = {BITS};
    constexpr int SPLITS = {SPLIT_K};
    constexpr int C_TILES = {c_tiles};

    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint n0 = threadgroup_position_in_grid.z * 8;

    threadgroup float partials[SPLITS * C_TILES * 64];

    simdgroup_matrix<float, 8, 8> c0;
    c0.thread_elements()[0] = 0.0f;
    c0.thread_elements()[1] = 0.0f;
{c1_decl}

    // Public row-major Metal 8x8 fragment lane mapping: each lane owns two
    // adjacent columns in one row.
    const ushort quad = lane / 4;
    const ushort frag_row = (quad & 4) + (lane / 2) % 4;
    const ushort frag_col = (quad & 2) * 2 + (lane % 2) * 2;

    const int groups = K / QGROUP;
    const int packed_stride = K / 8;
    const int scale_stride = groups;

    // Eight simdgroups split K by whole quantization groups.  Only lanes 0..7
    // read one output column's packed word and affine pair; shuffles distribute
    // them to the matrix-fragment lane mapping without threadgroup B staging.
    for (int group = (int)sg; group < groups; group += SPLITS) {{
        float scale_by_col = 0.0f;
        float bias_by_col = 0.0f;
        if (lane < 8) {{
            scale_by_col = (float)scales[
                (size_t)(n0 + lane) * scale_stride + group
            ];
            bias_by_col = (float)biases[
                (size_t)(n0 + lane) * scale_stride + group
            ];
        }}
        const float scale0 = simd_shuffle(scale_by_col, frag_col);
        const float scale1 = simd_shuffle(scale_by_col, frag_col + 1);
        const float bias0 = simd_shuffle(bias_by_col, frag_col);
        const float bias1 = simd_shuffle(bias_by_col, frag_col + 1);

        #pragma unroll
        for (int kt = 0; kt < QGROUP / 8; ++kt) {{
            const int k_base = group * QGROUP + kt * 8;
            uint packed_by_col = 0;
            if (lane < 8) {{
                packed_by_col = w[
                    (size_t)(n0 + lane) * packed_stride + k_base / 8
                ];
            }}
            const uint packed0 = simd_shuffle(packed_by_col, frag_col);
            const uint packed1 = simd_shuffle(packed_by_col, frag_col + 1);

            simdgroup_matrix<half, 8, 8> bmat;
            bmat.thread_elements()[0] = half(
                scale0 * (float)((packed0 >> (QBITS * frag_row)) & 0xF) + bias0
            );
            bmat.thread_elements()[1] = half(
                scale1 * (float)((packed1 >> (QBITS * frag_row)) & 0xF) + bias1
            );

            simdgroup_matrix<half, 8, 8> a0;
{a1_decl}
{a0_load}
{a1_load}
            simdgroup_multiply_accumulate(c0, a0, bmat, c0);
{c1_mma}
        }}
    }}

    // Preserve the native fragment layout in split-K scratch.  SIMD-group 0
    // reduces the same two elements from all eight workers and writes them
    // directly, avoiding matrix store/load transforms on the critical path.
    partials[(sg * C_TILES * 64) + lane * 2] = c0.thread_elements()[0];
    partials[(sg * C_TILES * 64) + lane * 2 + 1] = c0.thread_elements()[1];
{c1_partial}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {{
        if (frag_row < {min(m, 8)}) {{
            float total0 = 0.0f;
            float total1 = 0.0f;
            #pragma unroll
            for (int split = 0; split < SPLITS; ++split) {{
                total0 += partials[(split * C_TILES * 64) + lane * 2];
                total1 += partials[(split * C_TILES * 64) + lane * 2 + 1];
            }}
            y[(size_t)frag_row * N + n0 + frag_col] = (T)total0;
            y[(size_t)frag_row * N + n0 + frag_col + 1] = (T)total1;
        }}
{c1_reduce}
    }}
"""
