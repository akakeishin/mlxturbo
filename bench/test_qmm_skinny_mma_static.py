"""Metal-free source, host-geometry, eligibility, and fallback tests for A2 v4."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmlx.kernels._qmm_skinny_mma_source import (
    ACTIVE_INPUT_GROUPS,
    BITS,
    BLOCK_SIZE,
    GROUP_SIZE,
    INPUTS_PER_GROUP,
    METAL_HEADER,
    M_MAX,
    M_MIN,
    MINIMUM_TABLE_WIDTH,
    ROWS_PER_SIMD,
    TABLE_METAL_HEADER,
    THREADGROUP,
    VALUES_PER_THREAD,
    XSUMS_SOURCE,
    XSUMS_THREADGROUP,
    active_input_groups,
    build_source,
    build_table_source,
    eligible_layout,
    sums_stride,
)
from fastmlx.kernels import qmm_skinny_mma as qmv_module

REPO_ROOT = Path(__file__).resolve().parent.parent
SWIFT_REFERENCE = REPO_ROOT / "tools/reference/e120/Qwen35.swift"


def test_host_constants_and_width_plan_match_e120():
    assert ROWS_PER_SIMD == 4
    assert VALUES_PER_THREAD == 16
    assert BLOCK_SIZE == 512
    assert THREADGROUP == (32, 2, 1)
    assert XSUMS_THREADGROUP == (32, 1, 1)
    assert (M_MIN, M_MAX) == (2, 9)
    assert MINIMUM_TABLE_WIDTH == 4
    assert [sums_stride(m) for m in range(M_MIN, M_MAX + 1)] == [
        8, 8, 8, 8, 8, 8, 8, 16
    ]
    assert INPUTS_PER_GROUP == {
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 3,
        7: 4,
        8: 4,
        9: 3,
    }
    assert ACTIVE_INPUT_GROUPS == {
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2,
        9: 3,
    }
    for m, groups in ACTIVE_INPUT_GROUPS.items():
        assert active_input_groups(m) == groups
    for m in (1, 10):
        try:
            active_input_groups(m)
        except ValueError:
            pass
        else:
            raise AssertionError(f"M={m} must not have an E120 width plan")


def test_python_constants_are_witnessed_in_vendored_swift():
    swift = SWIFT_REFERENCE.read_text()
    license_text = (SWIFT_REFERENCE.parent / "LICENSE").read_text()
    for relative in (
        "fastmlx/kernels/_qmm_skinny_mma_source.py",
        "fastmlx/kernels/qmm_skinny_mma.py",
    ):
        port = (REPO_ROOT / relative).read_text()
        assert "Layr-Labs/qwen-3.8-mtp-challenge" in port
        assert "Copyright (c) 2026 Layr Labs" in port
        assert "MIT License" in port
    assert "MIT License" in license_text
    assert "Copyright (c) 2026 Layr Labs, Inc." in license_text
    assert "constexpr int rows_per_simd = 4;" in swift
    assert "constexpr int values_per_thread = 16;" in swift
    assert "constexpr int block_size = values_per_thread * 32;" in swift
    assert "threadGroup: (32, 2, 1)" in swift
    assert "grid: (Self.activeInputGroups(cell.m) * 32, (cell.n / 8) * 2, 1)" in swift
    assert "public static let minimumTableWidth = 4" in swift
    assert "grid: (32, kBlocks, m)" in swift
    for m, inputs_per_group in INPUTS_PER_GROUP.items():
        assert f"case {m}: inputsPerGroup = {inputs_per_group}" in swift


def test_source_is_e120_no_table_and_respects_prohibitions():
    header = METAL_HEADER
    source = build_source()
    joined = header + source

    assert "Copyright (c) 2026 Layr Labs" in header
    assert "MIT License" in header
    assert "typedef vec<float, NA> VF" in header
    assert "constexpr int rows_per_simd = 4" in header
    assert "constexpr int values_per_thread = 16" in header
    assert "constexpr int block_size = values_per_thread * 32" in header
    assert "simd_sum(acc[r][m])" in header

    # Runtime K/N contract. M/NA/IPG may be template constants; K and N may not.
    assert "qmv_k = x_shape[x_ndim - 1]" in source
    assert "qmv_n = w_shape[0]" in source
    template_lists = re.findall(r"template\s*<([^>]*)>", joined)
    assert all(not re.search(r"\b[KN]\b", params) for params in template_lists)

    # Phase 2 and every empirically failed design stay absent.
    assert "USE_TABLE" not in joined
    assert "xsums" not in joined
    assert "simdgroup_matrix" not in joined
    assert "threadgroup_barrier" not in joined
    assert "threadgroup float" not in joined
    assert "split_k" not in joined.lower()
    assert "SPLITS" not in joined

    # The widest specialization is NA=5, hence 4*5=20 accumulators/thread.
    assert ROWS_PER_SIMD * max(INPUTS_PER_GROUP.values()) == 20
    assert ROWS_PER_SIMD * max(INPUTS_PER_GROUP.values()) <= 24


def test_table_sources_match_e120_and_respect_prohibitions():
    header = TABLE_METAL_HEADER
    source = build_table_source()
    joined = header + source + XSUMS_SOURCE

    assert "const device float* xsums" in header
    assert "sums[m] = st[m]" in header
    assert "sums[m] +=" not in header
    assert "xsums[(xs_kb * 32 + xs_lane) * xs_stride + xs_row] = s" in XSUMS_SOURCE
    assert "s += xv[0] + xv[1] + xv[2] + xv[3]" in XSUMS_SOURCE
    assert "qmv_stride = qmv_m <= 8 ? 8 : 16" in source

    # Runtime K/N is the vendored E120 and KERNEL-INTEL contract.
    assert "qmv_k = x_shape[x_ndim - 1]" in source
    assert "qmv_n = w_shape[0]" in source
    template_lists = re.findall(r"template\s*<([^>]*)>", joined)
    assert all(not re.search(r"\b[KN]\b", params) for params in template_lists)

    assert "simdgroup_matrix" not in joined
    assert "threadgroup_barrier" not in joined
    assert "threadgroup float" not in joined
    assert "split_k" not in joined.lower()
    assert "SPLITS" not in joined
    assert ROWS_PER_SIMD * max(INPUTS_PER_GROUP.values()) == 20


def test_runtime_width_switch_matches_active_group_plan():
    source = build_source()
    for m, inputs_per_group in INPUTS_PER_GROUP.items():
        assert f"case {m}:" in source
        assert f"fastmlx_e120_qmv_m<{m}, {inputs_per_group}>" in source
    assert source.count("fastmlx_e120_qmv_m<") == len(INPUTS_PER_GROUP)


def test_gpu_gate_has_v3_acceptance_thresholds():
    gate = (REPO_ROOT / "bench/test_qmm_skinny_mma.py").read_text()
    assert "BF16_CORRECTNESS_THRESHOLD = 8e-3" in gate
    assert "CORRECTNESS_SHAPES = ((512, 1024), (5120, 4096))" in gate
    assert 'timings[8]["chain_speedup"] >= 1.5' in gate
    assert "range(M_MIN, M_MAX + 1)" in gate
    assert "mx.array_equal(actual, no_table)" in gate
    assert '"table_on_off_bit_exact"' in gate


def test_layout_eligibility_boundaries():
    def eligible(m=8, k=5120, n=17408, group=64, bits=4):
        return eligible_layout(
            m,
            k,
            n,
            (n, k * bits // 32),
            (n, k // group),
            (n, k // group),
            group,
            bits,
        )

    assert eligible(m=2)
    assert eligible(m=9)
    assert not eligible(m=1)
    assert not eligible(m=10)
    assert not eligible(group=128)
    assert not eligible(bits=8)
    assert not eligible(k=5184)
    assert not eligible(n=17409)
    assert not eligible_layout(
        8,
        5120,
        17408,
        (17408, 639),
        (17408, 80),
        (17408, 80),
        GROUP_SIZE,
        BITS,
    )


class _FakeArray:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)


class _FakeMX:
    cpu = object()
    gpu = object()
    float16 = object()
    float32 = object()
    bfloat16 = object()
    uint32 = object()

    def __init__(self, gpu_available=False):
        self._device = self.gpu if gpu_available else self.cpu
        self.metal = SimpleNamespace(is_available=lambda: gpu_available)
        self.fallback_calls = []
        self.kernel_builds = []
        self.kernel_calls = []
        self.fast = SimpleNamespace(metal_kernel=self._metal_kernel)

    def default_device(self):
        return self._device

    def quantized_matmul(self, *args, **kwargs):
        self.fallback_calls.append((args, kwargs))
        return "stock-fallback"

    def _metal_kernel(self, **kwargs):
        self.kernel_builds.append(kwargs)

        def kernel(**call_kwargs):
            self.kernel_calls.append(call_kwargs)
            return (f"{kwargs['output_names'][0]}-output",)

        return kernel


def _arrays(fake_mx, m=8, k=5120, n=17408, dtype=None):
    dtype = dtype or fake_mx.bfloat16
    return (
        _FakeArray((m, k), dtype),
        _FakeArray((n, k // 8), fake_mx.uint32),
        _FakeArray((n, k // 64), dtype),
        _FakeArray((n, k // 64), dtype),
    )


def test_non_metal_and_non_bf16_paths_call_stock_exactly_once():
    for fake_mx, dtype in (
        (_FakeMX(), None),
        (_FakeMX(gpu_available=True), "float16"),
    ):
        old_loader = qmv_module._load_mx
        qmv_module._load_mx = lambda fake_mx=fake_mx: fake_mx
        try:
            chosen_dtype = fake_mx.float16 if dtype == "float16" else None
            x, w, scales, biases = _arrays(fake_mx, dtype=chosen_dtype)
            out = qmv_module.qmm_skinny_mma(x, w, scales, biases)
        finally:
            qmv_module._load_mx = old_loader

        assert out == "stock-fallback"
        assert len(fake_mx.fallback_calls) == 1
        args, kwargs = fake_mx.fallback_calls[0]
        assert args == (x, w, scales, biases)
        assert kwargs == {"transpose": True, "group_size": 64, "bits": 4}


def test_eligible_path_builds_once_and_launches_e120_geometry():
    fake_mx = _FakeMX(gpu_available=True)
    old_loader = qmv_module._load_mx
    qmv_module._load_mx = lambda: fake_mx
    qmv_module._KERNEL = None
    qmv_module._TABLE_KERNEL = None
    qmv_module._XSUMS_KERNEL = None
    try:
        outputs = []
        arrays_by_m = {}
        for m in (2, 8, 9):
            x, w, scales, biases = _arrays(fake_mx, m=m)
            arrays_by_m[m] = (x, w, scales, biases)
            outputs.append(qmv_module.qmm_skinny_mma(x, w, scales, biases))
    finally:
        qmv_module._load_mx = old_loader
        qmv_module._KERNEL = None
        qmv_module._TABLE_KERNEL = None
        qmv_module._XSUMS_KERNEL = None

    assert outputs == ["y-output", "y-output", "y-output"]
    assert not fake_mx.fallback_calls
    assert len(fake_mx.kernel_builds) == 3
    builds = {build["output_names"][0]: build for build in fake_mx.kernel_builds}
    assert builds["y"]["ensure_row_contiguous"] is True
    assert any(
        build["input_names"] == ["w", "scales", "biases", "x"]
        and build["header"] == METAL_HEADER
        and build["source"] == build_source()
        for build in fake_mx.kernel_builds
    )
    assert any(
        build["input_names"] == ["w", "scales", "biases", "x", "xsums"]
        and build["header"] == TABLE_METAL_HEADER
        and build["source"] == build_table_source()
        for build in fake_mx.kernel_builds
    )
    xsums_build = builds["xsums"]
    assert xsums_build["input_names"] == ["x"]
    assert xsums_build["source"] == XSUMS_SOURCE
    assert xsums_build["ensure_row_contiguous"] is True

    assert len(fake_mx.kernel_calls) == 5
    m2, xsums8, m8, xsums9, m9 = fake_mx.kernel_calls
    x2, w2, scales2, biases2 = arrays_by_m[2]
    assert m2["inputs"] == [w2, scales2, biases2, x2]
    assert m2["grid"] == (32, 4352, 1)
    assert m2["threadgroup"] == (32, 2, 1)
    assert m2["output_shapes"] == [(2, 17408)]
    x8, w8, scales8, biases8 = arrays_by_m[8]
    assert xsums8["inputs"] == [x8]
    assert xsums8["grid"] == (32, 10, 8)
    assert xsums8["threadgroup"] == (32, 1, 1)
    assert xsums8["output_shapes"] == [(2560,)]
    assert xsums8["output_dtypes"] == [fake_mx.float32]
    assert m8["inputs"] == [w8, scales8, biases8, x8, "xsums-output"]
    assert m8["grid"] == (64, 4352, 1)
    assert m8["threadgroup"] == (32, 2, 1)
    assert m8["output_shapes"] == [(8, 17408)]
    assert "template" not in m8
    x9, w9, scales9, biases9 = arrays_by_m[9]
    assert xsums9["inputs"] == [x9]
    assert xsums9["grid"] == (32, 10, 9)
    assert xsums9["output_shapes"] == [(5120,)]
    assert m9["inputs"] == [w9, scales9, biases9, x9, "xsums-output"]
    assert m9["grid"] == (96, 4352, 1)
    assert m9["threadgroup"] == (32, 2, 1)
    assert m9["output_shapes"] == [(9, 17408)]
    assert "template" not in m9


def main():
    tests = [
        test_host_constants_and_width_plan_match_e120,
        test_python_constants_are_witnessed_in_vendored_swift,
        test_source_is_e120_no_table_and_respects_prohibitions,
        test_table_sources_match_e120_and_respect_prohibitions,
        test_runtime_width_switch_matches_active_group_plan,
        test_gpu_gate_has_v3_acceptance_thresholds,
        test_layout_eligibility_boundaries,
        test_non_metal_and_non_bf16_paths_call_stock_exactly_once,
        test_eligible_path_builds_once_and_launches_e120_geometry,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


if __name__ == "__main__":
    main()
