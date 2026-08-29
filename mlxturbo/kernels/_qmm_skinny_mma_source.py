"""Production source builder for the clean-room A2 v5 skinny MMA kernel.

The implementation follows ``docs/ISA-DIFF.md`` and the ``v_direct`` probe in
``tools/isa/variants.py``.  It is independent of ``mlxturbo.fast_qmm``.
"""

GROUP_SIZE = 64
BITS = 4
M_MIN = 6
M_MAX = 16
MMA_TILE = 8
SPLIT_K = 8
THREADGROUP = (32, SPLIT_K, 1)

METAL_HEADER = "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"


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
    """Check the BF16 affine-4/group-64 v5 layout without importing MLX."""

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


def _fragment_load(name: str, m: int, row_base: int) -> str:
    """Load one A tile directly from device, guarding only a partial M tile."""

    rows = min(MMA_TILE, max(0, m - row_base))
    if rows == MMA_TILE:
        return (
            f"            simdgroup_load({name}, "
            f"x + (size_t){row_base} * qmm_k + k_base, qmm_k);"
        )
    return f"""
            {name}.thread_elements()[0] = frag_row < {rows}
                ? x[(size_t)({row_base} + frag_row) * qmm_k + k_base + frag_col]
                : bfloat16_t(0.0f);
            {name}.thread_elements()[1] = frag_row < {rows}
                ? x[(size_t)({row_base} + frag_row) * qmm_k + k_base + frag_col + 1]
                : bfloat16_t(0.0f);"""


def _mma_step(kt: int, m: int, c_tiles: int) -> str:
    """Emit one fully scalarized K tile with direct uint4 component access."""

    vector = "pa_lo" if kt < 4 else "pa_hi"
    vector_b = "pb_lo" if kt < 4 else "pb_hi"
    component = ("x", "y", "z", "w")[kt % 4]
    a0_load = _fragment_load("a0", m, 0)
    a1 = ""
    if c_tiles == 2:
        a1_load = _fragment_load("a1", m, MMA_TILE)
        a1 = f"""
            simdgroup_matrix<bfloat16_t, 8, 8> a1;
{a1_load}
            simdgroup_multiply_accumulate(c1, a1, bmat, c1);"""
    return f"""
        {{
            const int k_base = group * QGROUP + {kt * MMA_TILE};
            const uint packed0 = {vector}.{component};
            const uint packed1 = {vector_b}.{component};
            simdgroup_matrix<bfloat16_t, 8, 8> bmat;
            bmat.thread_elements()[0] = bfloat16_t(
                scale0 * (float)((packed0 >> (QBITS * frag_row)) & 0xFu)
                + bias0);
            bmat.thread_elements()[1] = bfloat16_t(
                scale1 * (float)((packed1 >> (QBITS * frag_row)) & 0xFu)
                + bias1);

            simdgroup_matrix<bfloat16_t, 8, 8> a0;
{a0_load}
            simdgroup_multiply_accumulate(c0, a0, bmat, c0);{a1}
        }}"""


def _epilogue(m: int, c_tiles: int) -> str:
    c1_store = ""
    c1_reduce = ""
    if c_tiles == 2:
        c1_store = """
    partials[((sg * C_TILES + 1) * 64) + lane * 2] =
        c1.thread_elements()[0];
    partials[((sg * C_TILES + 1) * 64) + lane * 2 + 1] =
        c1.thread_elements()[1];"""
        c1_reduce = f"""
        const uint row1 = MMA_TILE + frag_row;
        if (row1 < {m}) {{
            float total0 = 0.0f;
            float total1 = 0.0f;
            #pragma unroll
            for (int split = 0; split < SPLITS; ++split) {{
                total0 += partials[
                    ((split * C_TILES + 1) * 64) + lane * 2];
                total1 += partials[
                    ((split * C_TILES + 1) * 64) + lane * 2 + 1];
            }}
            y[(size_t)row1 * qmm_n + n0 + frag_col] = bfloat16_t(total0);
            y[(size_t)row1 * qmm_n + n0 + frag_col + 1] = bfloat16_t(total1);
        }}"""
    return f"""
    partials[(sg * C_TILES * 64) + lane * 2] = c0.thread_elements()[0];
    partials[(sg * C_TILES * 64) + lane * 2 + 1] =
        c0.thread_elements()[1];{c1_store}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {{
        if (frag_row < {min(m, MMA_TILE)}) {{
            float total0 = 0.0f;
            float total1 = 0.0f;
            #pragma unroll
            for (int split = 0; split < SPLITS; ++split) {{
                total0 += partials[(split * C_TILES * 64) + lane * 2];
                total1 += partials[(split * C_TILES * 64) + lane * 2 + 1];
            }}
            y[(size_t)frag_row * qmm_n + n0 + frag_col] = bfloat16_t(total0);
            y[(size_t)frag_row * qmm_n + n0 + frag_col + 1] =
                bfloat16_t(total1);
        }}{c1_reduce}
    }}
"""


def build_source(m: int = MMA_TILE) -> str:
    """Return a Metal body specialized for one M in [6, 16].

    The default M=8 keeps the existing ISA tooling's ``build_source()`` probe
    useful without making K or N compile-time constants.
    """

    if not M_MIN <= m <= M_MAX:
        raise ValueError(f"M must be in [{M_MIN}, {M_MAX}], got {m}")
    c_tiles = 1 if m <= MMA_TILE else 2
    c1 = ""
    if c_tiles == 2:
        c1 = """
    simdgroup_matrix<float, 8, 8> c1;
    c1.thread_elements()[0] = 0.0f;
    c1.thread_elements()[1] = 0.0f;"""
    mma_steps = "".join(_mma_step(kt, m, c_tiles) for kt in range(8))
    return f"""
    constexpr int QGROUP = {GROUP_SIZE};
    constexpr int QBITS = {BITS};
    constexpr int SPLITS = {SPLIT_K};
    constexpr int MMA_TILE = {MMA_TILE};
    constexpr int C_TILES = {c_tiles};

    const int qmm_k = x_shape[x_ndim - 1];
    const int qmm_n = w_shape[0];
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint n0 = threadgroup_position_in_grid.z * MMA_TILE;

    threadgroup float partials[SPLITS * C_TILES * 64];

    simdgroup_matrix<float, 8, 8> c0;
    c0.thread_elements()[0] = 0.0f;
    c0.thread_elements()[1] = 0.0f;{c1}

    const ushort quad = lane / 4;
    const ushort frag_row = (quad & 4) + (lane / 2) % 4;
    const ushort frag_col = (quad & 2) * 2 + (lane % 2) * 2;

    const int groups = qmm_k / QGROUP;
    const int packed_stride = qmm_k / 8;
    const int scale_stride = groups;

    for (int group = (int)sg; group < groups; group += SPLITS) {{
        const uint col_a = n0 + frag_col;
        const uint col_b = col_a + 1;
        const device uint4* wa = reinterpret_cast<const device uint4*>(
            w + (size_t)col_a * packed_stride + group * (QGROUP / 8));
        const device uint4* wb = reinterpret_cast<const device uint4*>(
            w + (size_t)col_b * packed_stride + group * (QGROUP / 8));
        const uint4 pa_lo = wa[0];
        const uint4 pa_hi = wa[1];
        const uint4 pb_lo = wb[0];
        const uint4 pb_hi = wb[1];
        const float scale0 =
            (float)scales[(size_t)col_a * scale_stride + group];
        const float scale1 =
            (float)scales[(size_t)col_b * scale_stride + group];
        const float bias0 =
            (float)biases[(size_t)col_a * scale_stride + group];
        const float bias1 =
            (float)biases[(size_t)col_b * scale_stride + group];
{mma_steps}
    }}
{_epilogue(m, c_tiles)}
"""


__all__ = [
    "BITS",
    "GROUP_SIZE",
    "METAL_HEADER",
    "MMA_TILE",
    "M_MAX",
    "M_MIN",
    "SPLIT_K",
    "THREADGROUP",
    "build_source",
    "eligible_layout",
]
