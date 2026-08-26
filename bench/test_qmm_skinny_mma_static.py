"""Metal-free structural, eligibility, and fallback tests for Phase A2."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmlx.kernels._qmm_skinny_mma_source import (
    BITS,
    GROUP_SIZE,
    M_MAX,
    M_MIN,
    build_source,
    eligible_layout,
)
from fastmlx.kernels import qmm_skinny_mma as mma_module


def test_all_m_specializations_have_required_structure():
    for m in range(M_MIN, M_MAX + 1):
        source = build_source(m)
        assert "simdgroup_matrix<float, 8, 8> c0" in source
        assert "group += SPLITS" in source
        assert "constexpr int SPLITS = 8" in source
        assert "constexpr int QGROUP = 64" in source
        assert "simdgroup_matrix<half, 8, 8> bmat" in source
        assert "threadgroup T b_tiles" not in source
        assert "scale0 = simd_shuffle" in source
        assert "packed0 = simd_shuffle" in source
        assert "threadgroup float partials" in source
        assert "for (int split = 0; split < SPLITS; ++split)" in source


def test_partial_m_tiles_are_guarded_device_loads():
    m6 = build_source(6)
    assert "a0.thread_elements()[0]" in m6
    assert "frag_row < 6" in m6
    assert "simdgroup_matrix<float, 8, 8> c1" not in m6

    m8 = build_source(8)
    assert "a0.thread_elements()[0]" in m8
    m8_fp16 = build_source(8, fp16_input=True)
    assert "simdgroup_load(a0, x + (size_t)0 * K + k_base, K)" in m8_fp16

    m9 = build_source(9)
    assert "a0.thread_elements()[0]" in m9
    assert "a1.thread_elements()[0]" in m9
    assert "frag_row < 1" in m9
    assert m9.count("simdgroup_multiply_accumulate") == 2

    m16 = build_source(16, fp16_input=True)
    assert "simdgroup_load(a0" in m16
    assert "simdgroup_load(a1" in m16
    assert m16.count("simdgroup_multiply_accumulate") == 2


def test_fragment_lane_mapping_covers_each_matrix_element_once():
    coordinates = []
    for lane in range(32):
        quad = lane // 4
        row = (quad & 4) + (lane // 2) % 4
        col = (quad & 2) * 2 + (lane % 2) * 2
        coordinates.extend(((row, col), (row, col + 1)))
    assert len(coordinates) == 64
    assert set(coordinates) == {
        (row, col) for row in range(8) for col in range(8)
    }


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

    assert eligible(m=6)
    assert eligible(m=16)
    assert not eligible(m=5)
    assert not eligible(m=17)
    assert not eligible(group=128)
    assert not eligible(bits=8)
    assert not eligible(k=5100)
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
            return ("mma-output",)

        return kernel


def test_non_metal_path_calls_stock_exactly_once():
    fake_mx = _FakeMX()
    old_loader = mma_module._load_mx
    mma_module._load_mx = lambda: fake_mx
    try:
        x = _FakeArray((8, 5120), fake_mx.bfloat16)
        w = _FakeArray((17408, 640), fake_mx.uint32)
        scales = _FakeArray((17408, 80), fake_mx.bfloat16)
        biases = _FakeArray((17408, 80), fake_mx.bfloat16)
        out = mma_module.qmm_skinny_mma(x, w, scales, biases)
    finally:
        mma_module._load_mx = old_loader

    assert out == "stock-fallback"
    assert len(fake_mx.fallback_calls) == 1
    args, kwargs = fake_mx.fallback_calls[0]
    assert args == (x, w, scales, biases)
    assert kwargs == {"transpose": True, "group_size": 64, "bits": 4}


def test_eligible_path_builds_and_launches_expected_grid():
    fake_mx = _FakeMX(gpu_available=True)
    old_loader = mma_module._load_mx
    mma_module._load_mx = lambda: fake_mx
    mma_module._KERNELS.clear()
    try:
        x = _FakeArray((9, 5120), fake_mx.bfloat16)
        w = _FakeArray((17408, 640), fake_mx.uint32)
        scales = _FakeArray((17408, 80), fake_mx.bfloat16)
        biases = _FakeArray((17408, 80), fake_mx.bfloat16)
        out = mma_module.qmm_skinny_mma(x, w, scales, biases)
    finally:
        mma_module._load_mx = old_loader
        mma_module._KERNELS.clear()

    assert out == "mma-output"
    assert not fake_mx.fallback_calls
    assert len(fake_mx.kernel_builds) == 1
    assert fake_mx.kernel_builds[0]["header"] == (
        "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"
    )
    assert len(fake_mx.kernel_calls) == 1
    launch = fake_mx.kernel_calls[0]
    assert launch["grid"] == (32, 8, 2176)
    assert launch["threadgroup"] == (32, 8, 1)
    assert launch["output_shapes"] == [(9, 17408)]


def main():
    tests = [
        test_all_m_specializations_have_required_structure,
        test_partial_m_tiles_are_guarded_device_loads,
        test_fragment_lane_mapping_covers_each_matrix_element_once,
        test_layout_eligibility_boundaries,
        test_non_metal_path_calls_stock_exactly_once,
        test_eligible_path_builds_and_launches_expected_grid,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


if __name__ == "__main__":
    main()
