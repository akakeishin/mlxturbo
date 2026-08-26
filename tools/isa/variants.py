"""Probe variants of the skinny MMA body, for A/B at the instruction level.

Each entry changes exactly one mechanism against the shipped baseline so an
instruction-count delta can be attributed.  Nothing here is imported by
``fastmlx`` -- these bodies exist only to be compiled and disassembled.

Hypothesis mapping (docs/HYPOTHESES-A2.md):
  v_uint4        H4  packed weight read width: 8x uint32 -> 2x uint4 per group
  v_sgload_a     H4  A fragment: 2 scalar device loads per lane per MMA
                     -> one simdgroup_load per MMA
  v_uint4_sgload H4  both of the above
  v_bstage       H1  dequant the whole 64-wide group into a threadgroup B slab
                     once, then simdgroup_load it (this is fast_qmm's shape)
"""

from __future__ import annotations

GROUP_SIZE = 64
BITS = 4
SPLIT_K = 8


def _prologue(c_tiles: int, m: int) -> str:
    c1_decl = (
        """
    simdgroup_matrix<float, 8, 8> c1;
    c1.thread_elements()[0] = 0.0f;
    c1.thread_elements()[1] = 0.0f;"""
        if c_tiles == 2
        else ""
    )
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
    c0.thread_elements()[1] = 0.0f;{c1_decl}

    const ushort quad = lane / 4;
    const ushort frag_row = (quad & 4) + (lane / 2) % 4;
    const ushort frag_col = (quad & 2) * 2 + (lane % 2) * 2;

    const int groups = K / QGROUP;
    const int packed_stride = K / 8;
    const int scale_stride = groups;
"""


def _epilogue(c_tiles: int, m: int) -> str:
    c1_partial = (
        """
    partials[((sg * C_TILES + 1) * 64) + lane * 2] = c1.thread_elements()[0];
    partials[((sg * C_TILES + 1) * 64) + lane * 2 + 1] = c1.thread_elements()[1];"""
        if c_tiles == 2
        else ""
    )
    c1_reduce = (
        f"""
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
        }}"""
        if c_tiles == 2
        else ""
    )
    return f"""
    partials[(sg * C_TILES * 64) + lane * 2] = c0.thread_elements()[0];
    partials[(sg * C_TILES * 64) + lane * 2 + 1] = c0.thread_elements()[1];{c1_partial}
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
        }}{c1_reduce}
    }}
"""


def _scalar_a_load(name: str, m: int, row_base: int) -> str:
    rows = min(8, max(0, m - row_base))
    return f"""
            {name}.thread_elements()[0] = frag_row < {rows}
                ? half(x[(size_t)({row_base} + frag_row) * K + k_base + frag_col])
                : half(0);
            {name}.thread_elements()[1] = frag_row < {rows}
                ? half(x[(size_t)({row_base} + frag_row) * K + k_base + frag_col + 1])
                : half(0);"""


def uint4_body(m: int) -> str:
    """H4: one uint4 pair per group replaces eight scalar uint32 loads."""

    c_tiles = 1 if m <= 8 else 2
    a1_decl = "            simdgroup_matrix<half, 8, 8> a1;" if c_tiles == 2 else ""
    a1_load = _scalar_a_load("a1", m, 8) if c_tiles == 2 else ""
    c1_mma = (
        "            simdgroup_multiply_accumulate(c1, a1, bmat, c1);"
        if c_tiles == 2
        else ""
    )
    return (
        _prologue(c_tiles, m)
        + f"""
    for (int group = (int)sg; group < groups; group += SPLITS) {{
        float scale_by_col = 0.0f;
        float bias_by_col = 0.0f;
        uint4 packed_lo = uint4(0);
        uint4 packed_hi = uint4(0);
        if (lane < 8) {{
            scale_by_col = (float)scales[(size_t)(n0 + lane) * scale_stride + group];
            bias_by_col = (float)biases[(size_t)(n0 + lane) * scale_stride + group];
            const device uint4* wv = (const device uint4*)(
                w + (size_t)(n0 + lane) * packed_stride + group * (QGROUP / 8));
            packed_lo = wv[0];
            packed_hi = wv[1];
        }}
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
        for (int kt = 0; kt < QGROUP / 8; ++kt) {{
            const int k_base = group * QGROUP + kt * 8;
            const uint packed0 = simd_shuffle(words[kt], frag_col);
            const uint packed1 = simd_shuffle(words[kt], frag_col + 1);

            simdgroup_matrix<half, 8, 8> bmat;
            bmat.thread_elements()[0] = half(
                scale0 * (float)((packed0 >> (QBITS * frag_row)) & 0xF) + bias0);
            bmat.thread_elements()[1] = half(
                scale1 * (float)((packed1 >> (QBITS * frag_row)) & 0xF) + bias1);

            simdgroup_matrix<half, 8, 8> a0;
{a1_decl}
{_scalar_a_load("a0", m, 0)}
{a1_load}
            simdgroup_multiply_accumulate(c0, a0, bmat, c0);
{c1_mma}
        }}
    }}
"""
        + _epilogue(c_tiles, m)
    )


def sgload_a_body(m: int) -> str:
    """H4: A fragment through simdgroup_load instead of scalar device loads.

    Requires both MMA operands to share a type, so B becomes bfloat16_t here.
    Precision differs from the baseline; this variant exists to price the A
    load path, not to be shipped.
    """

    c_tiles = 1 if m <= 8 else 2
    a1 = (
        """            simdgroup_matrix<bfloat16_t, 8, 8> a1;
            simdgroup_load(a1, x + (size_t)8 * K + k_base, K);
            simdgroup_multiply_accumulate(c1, a1, bmat, c1);"""
        if c_tiles == 2
        else ""
    )
    return (
        _prologue(c_tiles, m)
        + f"""
    for (int group = (int)sg; group < groups; group += SPLITS) {{
        float scale_by_col = 0.0f;
        float bias_by_col = 0.0f;
        if (lane < 8) {{
            scale_by_col = (float)scales[(size_t)(n0 + lane) * scale_stride + group];
            bias_by_col = (float)biases[(size_t)(n0 + lane) * scale_stride + group];
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
                packed_by_col = w[(size_t)(n0 + lane) * packed_stride + k_base / 8];
            }}
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
{a1}
        }}
    }}
"""
        + _epilogue(c_tiles, m)
    )


def uint4_sgload_body(m: int) -> str:
    """H4: both weight-read width and A load path changed."""

    c_tiles = 1 if m <= 8 else 2
    a1 = (
        """            simdgroup_matrix<bfloat16_t, 8, 8> a1;
            simdgroup_load(a1, x + (size_t)8 * K + k_base, K);
            simdgroup_multiply_accumulate(c1, a1, bmat, c1);"""
        if c_tiles == 2
        else ""
    )
    return (
        _prologue(c_tiles, m)
        + f"""
    for (int group = (int)sg; group < groups; group += SPLITS) {{
        float scale_by_col = 0.0f;
        float bias_by_col = 0.0f;
        uint4 packed_lo = uint4(0);
        uint4 packed_hi = uint4(0);
        if (lane < 8) {{
            scale_by_col = (float)scales[(size_t)(n0 + lane) * scale_stride + group];
            bias_by_col = (float)biases[(size_t)(n0 + lane) * scale_stride + group];
            const device uint4* wv = (const device uint4*)(
                w + (size_t)(n0 + lane) * packed_stride + group * (QGROUP / 8));
            packed_lo = wv[0];
            packed_hi = wv[1];
        }}
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
        for (int kt = 0; kt < QGROUP / 8; ++kt) {{
            const int k_base = group * QGROUP + kt * 8;
            const uint packed0 = simd_shuffle(words[kt], frag_col);
            const uint packed1 = simd_shuffle(words[kt], frag_col + 1);

            simdgroup_matrix<bfloat16_t, 8, 8> bmat;
            bmat.thread_elements()[0] = bfloat16_t(
                scale0 * (float)((packed0 >> (QBITS * frag_row)) & 0xF) + bias0);
            bmat.thread_elements()[1] = bfloat16_t(
                scale1 * (float)((packed1 >> (QBITS * frag_row)) & 0xF) + bias1);

            simdgroup_matrix<bfloat16_t, 8, 8> a0;
            simdgroup_load(a0, x + k_base, K);
            simdgroup_multiply_accumulate(c0, a0, bmat, c0);
{a1}
        }}
    }}
"""
        + _epilogue(c_tiles, m)
    )


def bstage_body(m: int) -> str:
    """H1: whole-group dequant into a threadgroup B slab, then simdgroup_load.

    This is the structure fast_qmm uses.  It replaces the per-MMA shuffle-and-
    dequant with one dequant pass per group plus a threadgroup round trip.
    """

    c_tiles = 1 if m <= 8 else 2
    a1 = (
        """            simdgroup_matrix<bfloat16_t, 8, 8> a1;
            simdgroup_load(a1, x + (size_t)8 * K + k_base, K);
            simdgroup_multiply_accumulate(c1, a1, bmat, c1);"""
        if c_tiles == 2
        else ""
    )
    return (
        _prologue(c_tiles, m)
        + f"""
    threadgroup bfloat16_t bslab[SPLITS * QGROUP * 8];
    threadgroup bfloat16_t* bt = bslab + sg * (QGROUP * 8);

    for (int group = (int)sg; group < groups; group += SPLITS) {{
        const uint col = lane & 7;      // output column within the 8-wide tile
        const uint half_sel = lane >> 3;  // 4 lanes each cover 16 k-values
        {{
            const float s = (float)scales[
                (size_t)(n0 + col) * scale_stride + group];
            const float b = (float)biases[
                (size_t)(n0 + col) * scale_stride + group];
            const device uint* wr = w + (size_t)(n0 + col) * packed_stride
                + group * (QGROUP / 8) + half_sel * 2;
            const uint p0 = wr[0];
            const uint p1 = wr[1];
            #pragma unroll
            for (int t = 0; t < 8; ++t) {{
                bt[(half_sel * 16 + t) * 8 + col] =
                    bfloat16_t((float)((p0 >> (QBITS * t)) & 0xFu) * s + b);
            }}
            #pragma unroll
            for (int t = 0; t < 8; ++t) {{
                bt[(half_sel * 16 + 8 + t) * 8 + col] =
                    bfloat16_t((float)((p1 >> (QBITS * t)) & 0xFu) * s + b);
            }}
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int kt = 0; kt < QGROUP / 8; ++kt) {{
            const int k_base = group * QGROUP + kt * 8;
            simdgroup_matrix<bfloat16_t, 8, 8> bmat;
            simdgroup_load(bmat, bt + kt * 64, 8);
            simdgroup_matrix<bfloat16_t, 8, 8> a0;
            simdgroup_load(a0, x + k_base, K);
            simdgroup_multiply_accumulate(c0, a0, bmat, c0);
{a1}
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }}
"""
        + _epilogue(c_tiles, m)
    )


VARIANTS = {
    "v_uint4": uint4_body,
    "v_sgload_a": sgload_a_body,
    "v_uint4_sgload": uint4_sgload_body,
    "v_bstage": bstage_body,
}
