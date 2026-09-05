"""Metal-free route, fallback, reshape, and enable tests for Phase A3."""

import pytest
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlxturbo.kernels import dispatch


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
    assert dispatch.select_route(5120, 17408, 6) == dispatch.MMA
    assert dispatch.select_route(5120, 17408, 8) == dispatch.MMA
    assert dispatch.select_route(17408, 5120, 16) == dispatch.MMA
    assert dispatch.select_route(5120, 248320, 12) == dispatch.MMA
    assert dispatch.select_route(5120, 248320, 13) == dispatch.MMA
    assert dispatch.select_route(17408, 5120, 9) == dispatch.NOCAP
    assert dispatch.select_route(4096, 4096, 8) == dispatch.STOCK
    assert dispatch.select_route(5120, 17408, 4) == dispatch.STOCK
    assert dispatch.select_route(5120, 17408, 5) == dispatch.MMA
    assert dispatch.select_route(17408, 5120, 5) == dispatch.STOCK
    assert dispatch.select_route(21504, 5376, 4) == dispatch.E120
    assert dispatch.select_route(21504, 5376, 3) == dispatch.STOCK
    assert dispatch.select_route(21504, 5376, 5) == dispatch.STOCK
    assert dispatch.select_route(5120, 17408, 17) == dispatch.STOCK


@pytest.fixture(autouse=True)
def _small_m_off(monkeypatch):
    """この file は偽の mx で素の契約を見る。小 M の既定 (auto = 非 NAX で
    small_m) が入ると M=2..5 が自前カーネルの形検査 (実 mx 前提) に流れるので、
    経路を off に固定しておく。"""
    monkeypatch.setenv("MLXTURBO_SMALL_M_ROUTE", "off")
    dispatch.refresh_small_m_route()
    yield
    monkeypatch.delenv("MLXTURBO_SMALL_M_ROUTE", raising=False)
    dispatch.refresh_small_m_route()


def test_stock_path_preserves_full_contract():
    fake_mx = _FakeMX()
    old_mx = dispatch._load_mx
    dispatch._load_mx = lambda: fake_mx
    try:
        # M=4: stock 帯 (M=5 は 2026-08-27 較正で MMA に移った)
        x = _Array((1, 4, 5120), "x")
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

    assert out.shape == (1, 4, 17408)
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
        ("mma", (6, 5120), {"group_size": 64, "bits": 4}),
        ("mma", (8, 5120), {"group_size": 64, "bits": 4}),
    ]
    assert not fake_mx.calls


def test_e120_default_table_uses_only_selected_shape_and_width():
    fake_mx = _FakeMX()
    calls = []

    def e120(x, w, scales, biases, **kwargs):
        calls.append((x.shape, w.shape, kwargs))
        return _Array((x.shape[0], w.shape[0]), "e120")

    old_mx = dispatch._load_mx
    old_e120 = dispatch._load_e120
    dispatch._load_mx = lambda: fake_mx
    dispatch._load_e120 = lambda: e120
    try:
        w = _Array((5376, 2688))
        scales = biases = _Array((5376, 336))
        out = dispatch.quantized_matmul(
            _Array((1, 4, 21504)),
            w,
            scales,
            biases,
            group_size=64,
            bits=4,
        )
        dispatch.quantized_matmul(
            _Array((1, 3, 21504)),
            w,
            scales,
            biases,
            group_size=64,
            bits=4,
        )
    finally:
        dispatch._load_mx = old_mx
        dispatch._load_e120 = old_e120

    assert out.shape == (1, 4, 5376)
    assert calls == [
        ((4, 21504), (5376, 2688), {"group_size": 64, "bits": 4})
    ]
    assert len(fake_mx.calls) == 1


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


def test_e120_route_is_generic_and_has_common_rollback(monkeypatch):
    assert dispatch.select_route(21504, 5376, 4) == dispatch.E120

    monkeypatch.setenv("MLXTURBO_SMALL_M_ROUTE", "small_m")
    dispatch.refresh_small_m_route()
    row = dispatch._small_m_row(21504, 5376)
    assert row[3] == dispatch.SMALLM
    assert row[4] == dispatch.E120
    assert row[5] == dispatch.SMALLM

    monkeypatch.setenv("MLXTURBO_QMM_E120", "0")
    with dispatch.dispatch_scope(unlisted_small_m=False):
        assert dispatch._E120_ACTIVE.get() is False
        assert dispatch._UNLISTED_SMALL_M_ACTIVE.get() is False
    assert dispatch._E120_ACTIVE.get() is True
    assert dispatch._UNLISTED_SMALL_M_ACTIVE.get() is True


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ((13, False), False),  # M1: 未測定
        ((14, False), False),  # M2: 未測定
        ((15, False), True),   # M3: この開発機で実測
        ((16, False), True),   # M4: E120公開計測あり
        ((17, False), False),  # M5/NAX: 未測定
        ((18, False), False),  # M6: M5の保守側を継承
        (None, False),
    ],
)
def test_e120_auto_uses_only_measured_gpu_families(
    monkeypatch, family, expected
):
    monkeypatch.delenv("MLXTURBO_QMM_E120", raising=False)
    monkeypatch.setattr(dispatch, "_apple_gpu_family", lambda: family)
    assert dispatch._e120_enabled() is expected

    monkeypatch.setenv("MLXTURBO_QMM_E120", "force")
    assert dispatch._e120_enabled() is True

    monkeypatch.setenv("MLXTURBO_QMM_E120", "off")
    assert dispatch._e120_enabled() is False


def test_enable_refreshes_e120_policy_for_always_active(monkeypatch):
    fake_nn = SimpleNamespace(QuantizedLinear=_QuantizedLinear)
    old_nn = dispatch._load_nn
    old_class = dispatch._DISPATCHED_CLASS
    token = dispatch._E120_ACTIVE.set(True)
    dispatch._load_nn = lambda: fake_nn
    dispatch._DISPATCHED_CLASS = None
    monkeypatch.setenv("MLXTURBO_QMM_E120", "off")
    try:
        dispatch.enable(_Model())
        assert dispatch._E120_ACTIVE.get() is False
    finally:
        dispatch._load_nn = old_nn
        dispatch._DISPATCHED_CLASS = old_class
        dispatch._E120_ACTIVE.reset(token)


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
