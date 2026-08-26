"""Metal-free route, fallback, reshape, and enable tests for Phase A3."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmlx.kernels import dispatch


class _Array:
    def __init__(self, shape, label="array"):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.label = label

    def reshape(self, shape):
        return _Array(shape, f"reshape({self.label})")


class _FakeMX:
    def __init__(self):
        self.calls = []

    def quantized_matmul(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _Array((*args[0].shape[:-1], args[1].shape[0]), "stock")


def test_shape_by_m_table_and_unknown_fallback():
    assert dispatch.select_route(5120, 17408, 6) == dispatch.NOCAP
    assert dispatch.select_route(5120, 17408, 8) == dispatch.MMA
    assert dispatch.select_route(17408, 5120, 16) == dispatch.MMA
    assert dispatch.select_route(5120, 248320, 12) == dispatch.MMA
    assert dispatch.select_route(5120, 248320, 13) == dispatch.STOCK
    assert dispatch.select_route(4096, 4096, 8) == dispatch.STOCK
    assert dispatch.select_route(5120, 17408, 5) == dispatch.STOCK
    assert dispatch.select_route(5120, 17408, 17) == dispatch.STOCK


def test_stock_path_preserves_full_contract():
    fake_mx = _FakeMX()
    old_mx = dispatch._load_mx
    dispatch._load_mx = lambda: fake_mx
    try:
        x = _Array((1, 5, 5120), "x")
        w = _Array((17408, 640), "w")
        scales = _Array((17408, 80), "scales")
        biases = _Array((17408, 80), "biases")
        out = dispatch.quantized_matmul(
            x,
            w,
            scales,
            biases,
            group_size=64,
            bits=4,
            mode="affine",
        )
    finally:
        dispatch._load_mx = old_mx

    assert out.shape == (1, 5, 17408)
    assert len(fake_mx.calls) == 1
    args, kwargs = fake_mx.calls[0]
    assert args == (x, w)
    assert kwargs == {
        "scales": scales,
        "biases": biases,
        "transpose": True,
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
    }


def test_custom_routes_flatten_and_restore_prefix_shape():
    fake_mx = _FakeMX()
    calls = []

    def kernel(name):
        def run(x, w, scales, biases, **kwargs):
            calls.append((name, x.shape, kwargs))
            return _Array((x.shape[0], w.shape[0]), name)

        return run

    old_mx = dispatch._load_mx
    old_kernels = dispatch._load_kernels
    dispatch._load_mx = lambda: fake_mx
    dispatch._load_kernels = lambda: (kernel("nocap"), kernel("mma"))
    try:
        w = _Array((17408, 640))
        scales = biases = _Array((17408, 80))
        out6 = dispatch.quantized_matmul(
            _Array((1, 6, 5120)),
            w,
            scales,
            biases,
            group_size=64,
            bits=4,
        )
        out8 = dispatch.quantized_matmul(
            _Array((2, 4, 5120)),
            w,
            scales,
            biases,
            group_size=64,
            bits=4,
        )
    finally:
        dispatch._load_mx = old_mx
        dispatch._load_kernels = old_kernels

    assert out6.shape == (1, 6, 17408)
    assert out8.shape == (2, 4, 17408)
    assert calls == [
        ("nocap", (6, 5120), {"group_size": 64, "bits": 4}),
        ("mma", (8, 5120), {"group_size": 64, "bits": 4}),
    ]
    assert not fake_mx.calls


class _QuantizedLinear:
    def __init__(self):
        self.marker = object()


class _Other:
    pass


class _Model:
    def __init__(self):
        self.linear = _QuantizedLinear()
        self.other = _Other()

    def named_modules(self):
        return [("", self), ("linear", self.linear), ("other", self.other)]


def test_enable_is_in_place_identity_preserving_and_idempotent():
    fake_nn = SimpleNamespace(QuantizedLinear=_QuantizedLinear)
    old_nn = dispatch._load_nn
    old_class = dispatch._DISPATCHED_CLASS
    dispatch._load_nn = lambda: fake_nn
    dispatch._DISPATCHED_CLASS = None
    try:
        model = _Model()
        linear_id = id(model.linear)
        marker = model.linear.marker
        assert dispatch.enable(model) is model
        dispatched_class = type(model.linear)
        assert dispatched_class.__name__ == "DispatchedQuantizedLinear"
        assert id(model.linear) == linear_id
        assert model.linear.marker is marker
        assert type(model.other) is _Other
        assert dispatch.enable(model) is model
        assert type(model.linear) is dispatched_class
        assert model.linear._fastmlx_dispatch_always is True
        dispatch.enable(model, active=False)
        assert model.linear._fastmlx_dispatch_always is False
        assert dispatch._DISPATCH_ACTIVE.get() is False
        with dispatch.dispatch_scope():
            assert dispatch._DISPATCH_ACTIVE.get() is True
        assert dispatch._DISPATCH_ACTIVE.get() is False
    finally:
        dispatch._load_nn = old_nn
        dispatch._DISPATCHED_CLASS = old_class


def main():
    tests = [
        test_shape_by_m_table_and_unknown_fallback,
        test_stock_path_preserves_full_contract,
        test_custom_routes_flatten_and_restore_prefix_shape,
        test_enable_is_in_place_identity_preserving_and_idempotent,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


if __name__ == "__main__":
    main()
