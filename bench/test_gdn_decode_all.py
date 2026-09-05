"""Qwen4-Exp の decode-only GDN 全段融合の admission 契約を検査する。

Metal の実行はこのテストの責務ではない。CPU でも、形・dtype・実行条件を
外れたときに理由付きで素の経路へ戻せることと、カーネルの dispatch 形を検査する。
"""

from __future__ import annotations

from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("mlx")

import mlx.core as mx  # noqa: E402

from mlxturbo.kernels import gdn_decode_all as fused_gdn  # noqa: E402


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


def production_values(dtype=mx.bfloat16):
    return {
        "qkv": FakeArray((1, 1, fused_gdn.CONV_DIM), dtype),
        "z": FakeArray((1, 1, fused_gdn.VALUE_DIM), dtype),
        "beta": FakeArray((1, 1, fused_gdn.NUM_VALUE_HEADS), dtype),
        "alpha": FakeArray((1, 1, fused_gdn.NUM_VALUE_HEADS), dtype),
        "conv_state": FakeArray(
            (1, fused_gdn.CONV_KERNEL - 1, fused_gdn.CONV_DIM), dtype
        ),
        "recurrent_state": FakeArray(
            (1, fused_gdn.NUM_VALUE_HEADS, fused_gdn.VALUE_HEAD_DIM,
             fused_gdn.KEY_HEAD_DIM),
            mx.float32,
        ),
        "conv_weight": FakeArray(
            (fused_gdn.CONV_DIM, fused_gdn.CONV_KERNEL, 1), dtype
        ),
        "a_log": FakeArray((fused_gdn.NUM_VALUE_HEADS,), mx.float32),
        "dt_bias": FakeArray((fused_gdn.NUM_VALUE_HEADS,), dtype),
        "norm_weight": FakeArray((fused_gdn.VALUE_HEAD_DIM,), dtype),
    }


def admission(**overrides):
    values = production_values()
    options = {
        "mask": None,
        "cache_lengths": None,
        "record_rollback": False,
        "training": False,
        "sharded": False,
        "num_key_heads": fused_gdn.NUM_KEY_HEADS,
        "num_value_heads": fused_gdn.NUM_VALUE_HEADS,
        "key_head_dim": fused_gdn.KEY_HEAD_DIM,
        "value_head_dim": fused_gdn.VALUE_HEAD_DIM,
        "conv_kernel": fused_gdn.CONV_KERNEL,
        "gate_activation": "sigmoid",
    }
    for key in tuple(options):
        if key in overrides:
            options[key] = overrides.pop(key)
    values.update(overrides)
    return fused_gdn.admit_qwen4_fused_gdn_decode(
        **values,
        **options,
    )


def test_exact_single_token_production_geometry_is_admitted():
    result = admission()
    assert result.accepted, result.reason
    assert result.reason == "eligible"


@pytest.mark.parametrize(
    ("field", "shape"),
    [
        ("qkv", (2, 1, fused_gdn.CONV_DIM)),
        ("qkv", (1, 2, fused_gdn.CONV_DIM)),
        ("z", (1, 1, fused_gdn.VALUE_DIM - 1)),
        ("alpha", (1, 1, fused_gdn.NUM_VALUE_HEADS - 1)),
        ("conv_state", (1, fused_gdn.CONV_KERNEL, fused_gdn.CONV_DIM)),
        ("recurrent_state", (1, fused_gdn.NUM_VALUE_HEADS,
                              fused_gdn.VALUE_HEAD_DIM, fused_gdn.KEY_HEAD_DIM - 1)),
    ],
)
def test_non_single_token_or_wrong_shapes_fall_back(field, shape):
    values = production_values()
    values[field] = FakeArray(shape, values[field].dtype)
    result = admission(**{field: values[field]})
    assert not result.accepted
    assert "shape" in result.reason


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"mask": object()}, "masked decode"),
        ({"cache_lengths": object()}, "ragged cache lengths"),
        ({"record_rollback": True}, "speculative rollback"),
        ({"training": True}, "training"),
        ({"sharded": True}, "distributed sharding"),
        ({"gate_activation": "silu"}, "output gate 'silu'"),
    ],
)
def test_runtime_flags_always_fall_back_with_reason(kwargs, reason):
    values = production_values()
    args = {
        **values,
        "mask": None,
        "cache_lengths": None,
        "record_rollback": False,
        "training": False,
        "sharded": False,
        "num_key_heads": fused_gdn.NUM_KEY_HEADS,
        "num_value_heads": fused_gdn.NUM_VALUE_HEADS,
        "key_head_dim": fused_gdn.KEY_HEAD_DIM,
        "value_head_dim": fused_gdn.VALUE_HEAD_DIM,
        "conv_kernel": fused_gdn.CONV_KERNEL,
        "gate_activation": "sigmoid",
    }
    args.update(kwargs)
    result = fused_gdn.admit_qwen4_fused_gdn_decode(**args)
    assert not result.accepted
    assert result.reason == reason


def test_dtype_and_geometry_are_strict():
    assert not admission(a_log=FakeArray((fused_gdn.NUM_VALUE_HEADS,), mx.float16)).accepted
    assert not admission(z=FakeArray((1, 1, fused_gdn.VALUE_DIM), mx.float16)).accepted
    assert not admission(
        recurrent_state=FakeArray(
            (1, fused_gdn.NUM_VALUE_HEADS, fused_gdn.VALUE_HEAD_DIM,
             fused_gdn.KEY_HEAD_DIM),
            mx.bfloat16,
        )
    ).accepted
    result = admission(num_key_heads=fused_gdn.NUM_KEY_HEADS + 1)
    assert not result.accepted
    assert "unsupported geometry" in result.reason


def test_kernel_dispatch_shape_is_fixed_and_threadgroup_is_validated():
    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return [
            FakeArray(shape, dtype)
            for shape, dtype in zip(
                kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
            )
        ]

    values = production_values()
    with patch.object(fused_gdn, "_kernel", return_value=fake_kernel):
        outputs = fused_gdn.qwen4_fused_gdn_decode(
            values["qkv"], values["z"], values["beta"], values["alpha"],
            values["conv_state"], values["conv_weight"], values["a_log"],
            values["dt_bias"], values["recurrent_state"], values["norm_weight"],
            1.0e-6, threadgroup_y=16,
        )

    assert calls[0]["grid"] == (32, 16, fused_gdn.NUM_VALUE_HEADS)
    assert calls[0]["threadgroup"] == (32, 16, 1)
    assert outputs[0].shape == (1, 1, fused_gdn.VALUE_DIM)
    assert outputs[1].shape == (1, fused_gdn.CONV_KERNEL - 1, fused_gdn.CONV_DIM)
    assert outputs[2].shape == (
        1, fused_gdn.NUM_VALUE_HEADS, fused_gdn.VALUE_HEAD_DIM,
        fused_gdn.KEY_HEAD_DIM,
    )
    assert ("RATIO", fused_gdn.NUM_VALUE_HEADS // fused_gdn.NUM_KEY_HEADS) in calls[0]["template"]

    with pytest.raises(ValueError, match="unsupported threadgroup_y"):
        fused_gdn.qwen4_fused_gdn_decode(
            values["qkv"], values["z"], values["beta"], values["alpha"],
            values["conv_state"], values["conv_weight"], values["a_log"],
            values["dt_bias"], values["recurrent_state"], values["norm_weight"],
            1.0e-6, threadgroup_y=2,
        )


def test_runtime_capability_and_probe_fail_closed_without_metal(monkeypatch):
    assert isinstance(fused_gdn.fused_gdn_runtime_supported(), bool)
    monkeypatch.setattr(fused_gdn, "_PROBE_COMPLETE", False)
    monkeypatch.setattr(fused_gdn, "_PROBED_THREADGROUP_Y", None)
    monkeypatch.setattr(fused_gdn, "_PROBE_LOCK", Lock())
    monkeypatch.setattr(fused_gdn, "fused_gdn_runtime_supported", lambda: False)
    assert fused_gdn.probe_qwen4_fused_gdn_decode(mx.bfloat16) is None


class _Cache:
    def __init__(self, conv_state, recurrent_state, *, rollback=False):
        self.values = [conv_state, recurrent_state]
        self.record_rollback = rollback
        self.advanced = 0

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def advance(self, amount):
        self.advanced += amount


class _Callable:
    def __init__(self, fn, **attrs):
        self._fn = fn
        for name, value in attrs.items():
            setattr(self, name, value)

    def __call__(self, *args):
        return self._fn(*args)


def _stub_gdn(sequence_length=1, *, rollback=False):
    dtype = mx.bfloat16
    conv_dim = fused_gdn.CONV_DIM
    value_dim = fused_gdn.VALUE_DIM
    n_v = fused_gdn.NUM_VALUE_HEADS
    dv = fused_gdn.VALUE_HEAD_DIM
    state = mx.zeros(
        (1, n_v, dv, fused_gdn.KEY_HEAD_DIM), dtype=mx.float32
    )
    conv_state = mx.zeros(
        (1, fused_gdn.CONV_KERNEL - 1, conv_dim), dtype=dtype
    )

    def project(_x):
        return (
            mx.zeros((1, sequence_length, conv_dim), dtype=dtype),
            mx.zeros((1, sequence_length, value_dim), dtype=dtype),
            mx.zeros((1, sequence_length, n_v), dtype=dtype),
            mx.zeros((1, sequence_length, n_v), dtype=dtype),
        )

    def store_conv(cache, conv_input):
        cache[0] = mx.contiguous(conv_input[:, -fused_gdn.CONV_KERNEL + 1 :, :])

    norm = _Callable(
        lambda out, _gate: out,
        weight=mx.ones((dv,), dtype=dtype),
        eps=1.0e-6,
        activation="sigmoid",
    )
    stub = SimpleNamespace(
        n_v=n_v,
        n_k=fused_gdn.NUM_KEY_HEADS,
        dk=fused_gdn.KEY_HEAD_DIM,
        dv=dv,
        key_dim=fused_gdn.KEY_DIM,
        value_dim=value_dim,
        conv_kernel_size=fused_gdn.CONV_KERNEL,
        conv_dim=conv_dim,
        training=False,
        A_log=mx.zeros((n_v,), dtype=mx.float32),
        dt_bias=mx.zeros((n_v,), dtype=dtype),
        _gdn_decode_all=True,
        _project_in=project,
        _store_conv_state=store_conv,
        in_proj_qkv=object(),
        in_proj_z=object(),
        in_proj_b=object(),
        in_proj_a=object(),
        conv1d=_Callable(
            lambda conv_input: mx.zeros(
                (1, sequence_length, conv_dim), dtype=dtype
            ),
            weight=mx.zeros((conv_dim, fused_gdn.CONV_KERNEL, 1), dtype=dtype),
        ),
        norm=norm,
        out_proj=_Callable(lambda value: value),
    )
    return stub, _Cache(conv_state, state, rollback=rollback)


def test_plain_ar_s1_calls_fused_and_updates_cache(monkeypatch):
    from mlx_lm.models import qwen4_exp as Q

    stub, cache = _stub_gdn()
    fused_output = mx.zeros((1, 1, fused_gdn.VALUE_DIM), dtype=mx.bfloat16)
    next_conv = mx.ones(
        (1, fused_gdn.CONV_KERNEL - 1, fused_gdn.CONV_DIM), dtype=mx.bfloat16
    )
    next_state = mx.ones(
        (1, fused_gdn.NUM_VALUE_HEADS, fused_gdn.VALUE_HEAD_DIM,
         fused_gdn.KEY_HEAD_DIM),
        dtype=mx.float32,
    )
    execute = patch.object(
        fused_gdn,
        "qwen4_fused_gdn_decode",
        return_value=(fused_output, next_conv, next_state),
    )
    monkeypatch.setattr(fused_gdn, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(fused_gdn, "probe_qwen4_fused_gdn_decode", lambda _dtype: 16)
    with execute as run:
        result = Q.GatedDeltaNet.__call__(
            stub, mx.zeros((1, 1, 2560), dtype=mx.bfloat16), None, cache
        )

    assert result is fused_output
    assert cache[0] is next_conv
    assert cache[1] is next_state
    assert cache.advanced == 1
    assert run.call_count == 1


@pytest.mark.parametrize(
    ("sequence_length", "rollback"),
    [(2, False), (1, True)],
    ids=["prefill-width", "rollback-capture"],
)
def test_prefill_and_rollback_fall_back_to_stock(sequence_length, rollback):
    from mlx_lm.models import qwen4_exp as Q

    stub, cache = _stub_gdn(sequence_length, rollback=rollback)
    stock_output = mx.zeros(
        (1, sequence_length, fused_gdn.NUM_VALUE_HEADS,
         fused_gdn.VALUE_HEAD_DIM),
        dtype=mx.bfloat16,
    )

    def stock(*args, **kwargs):
        state = args[7]
        return stock_output, state

    with (
        patch.object(fused_gdn, "fused_gdn_runtime_supported", lambda: True),
        patch.object(fused_gdn, "probe_qwen4_fused_gdn_decode", lambda _dtype: 16),
        patch.object(
            fused_gdn,
            "qwen4_fused_gdn_decode",
            side_effect=AssertionError("fallback must not execute fused kernel"),
        ),
        patch.object(Q, "gated_delta_update", side_effect=stock) as stock_call,
    ):
        result = Q.GatedDeltaNet.__call__(
            stub,
            mx.zeros((1, sequence_length, 2560), dtype=mx.bfloat16),
            None,
            cache,
        )

    assert result.shape == (1, sequence_length, fused_gdn.VALUE_DIM)
    assert mx.all(result == stock_output.reshape(1, sequence_length, -1)).item()
    assert stock_call.call_count == 1
    assert cache.advanced == sequence_length


def test_uninitialized_cache_falls_back_to_stock():
    from mlx_lm.models import qwen4_exp as Q

    stub, cache = _stub_gdn()
    cache[0] = None
    cache[1] = None

    def stock(*args, **kwargs):
        return mx.zeros(
            (1, 1, fused_gdn.NUM_VALUE_HEADS, fused_gdn.VALUE_HEAD_DIM),
            dtype=mx.bfloat16,
        ), None

    with (
        patch.object(fused_gdn, "fused_gdn_runtime_supported", lambda: True),
        patch.object(fused_gdn, "probe_qwen4_fused_gdn_decode", lambda _dtype: 16),
        patch.object(
            fused_gdn,
            "qwen4_fused_gdn_decode",
            side_effect=AssertionError("uninitialized cache must use stock path"),
        ),
        patch.object(Q, "gated_delta_update", side_effect=stock) as stock_call,
    ):
        Q.GatedDeltaNet.__call__(
            stub, mx.zeros((1, 1, 2560), dtype=mx.bfloat16), None, cache
        )

    assert stock_call.call_count == 1
    assert cache.advanced == 1


def test_capture_replaces_entrypoint_and_restores_it():
    from mlx_lm.models import qwen4_exp as Q
    from mlxturbo.spec_flash import capture

    model = SimpleNamespace(model=SimpleNamespace(hyper_connection_mixer=object()))
    original = Q.GatedDeltaNet.__call__
    with capture(model):
        assert Q.GatedDeltaNet.__call__ is not original
    assert Q.GatedDeltaNet.__call__ is original


def test_fused_enable_is_default_on_and_disable_is_reversible(monkeypatch):
    from mlx_lm.models import qwen4_exp as Q
    from mlxturbo import fused

    monkeypatch.delenv("MLXTURBO_GDN_DECODE_ALL", raising=False)
    assert fused.enable_gdn_decode_all() == 1
    assert Q.GatedDeltaNet._gdn_decode_all is True

    monkeypatch.setenv("MLXTURBO_GDN_DECODE_ALL", "0")
    assert fused.enable_gdn_decode_all() == 0
    assert Q.GatedDeltaNet._gdn_decode_all is False
    fused.disable_gdn_decode_all()
    assert Q.GatedDeltaNet._gdn_decode_all is False


def test_existing_default_fusion_hook_honors_full_decode_opt_in(monkeypatch):
    from mlx_lm.models import qwen4_exp as Q
    from mlxturbo import fused

    monkeypatch.setenv("MLXTURBO_GDN_DECODE_ALL", "1")
    monkeypatch.setenv("MLXTURBO_GDN_DECODE_FUSED", "0")
    fused.enable_gdn_decode_fused()
    assert Q.GatedDeltaNet._gdn_decode_all is True
    fused.disable_gdn_decode_all()
