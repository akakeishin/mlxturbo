"""MoE compileがqwen4_exp以外を有効と誤表示しない契約を確認する。"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlxturbo import fused


def _block_type(name):
    def __init__(self):
        self.gate = object()
        self.switch_mlp = object()

    def __call__(self, x):
        return x

    return type(name, (), {"__init__": __init__, "__call__": __call__})


def _model(*blocks):
    return SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(mlp=block) for block in blocks]
        )
    )


@pytest.fixture(autouse=True)
def _restore_compile_state():
    old_on = fused._MOE_COMPILE_ON
    old_orig = fused._MOE_COMPILE_ORIG
    try:
        fused.disable_moe_block_compile()
        # 別ファイルが先に実クラスへcompileを有効化していても、このテストが
        # 差し替える契約用クラスを独立に検査できるようにする。
        fused._MOE_COMPILE_ORIG = None
        yield
    finally:
        fused._MOE_COMPILE_ON = old_on
        fused._MOE_COMPILE_ORIG = old_orig


def test_non_qwen4_structural_moe_is_not_counted(monkeypatch):
    from mlx_lm.models import qwen4_exp

    qwen4_type = _block_type("_ContractQwen4Moe")
    other_type = _block_type("_ContractQwen3NextMoe")
    monkeypatch.setattr(qwen4_exp, "SparseMoeBlock", qwen4_type)

    assert fused.enable_moe_block_compile(_model(other_type()), mode="auto") == 0
    assert other_type.__call__ is not fused._moe_compile_call
    assert not fused._MOE_COMPILE_ON


def test_unsupported_draft_keeps_target_compile(monkeypatch):
    """対象外draftを後から走査してもtargetのMoE compileを保つ。"""
    from mlx_lm.models import qwen4_exp

    qwen4_type = _block_type("_ContractQwen4Moe")
    other_type = _block_type("_ContractOtherMoe")
    monkeypatch.setattr(qwen4_exp, "SparseMoeBlock", qwen4_type)

    assert fused.enable_moe_block_compile(
        _model(qwen4_type()), mode="auto"
    ) == 1
    assert fused.enable_moe_block_compile(
        _model(other_type()), mode="auto"
    ) == 0
    assert fused._MOE_COMPILE_ON is True


def test_qwen4_blocks_are_counted_and_wrapped(monkeypatch):
    from mlx_lm.models import qwen4_exp

    qwen4_type = _block_type("_ContractQwen4Moe")
    original = qwen4_type.__call__
    monkeypatch.setattr(qwen4_exp, "SparseMoeBlock", qwen4_type)
    monkeypatch.setattr(mx, "compile", lambda fn: fn)
    blocks = [qwen4_type(), qwen4_type()]

    try:
        assert fused.enable_moe_block_compile(_model(*blocks), mode="auto") == 2
        assert qwen4_type.__call__ is fused._moe_compile_call
        assert blocks[0](SimpleNamespace(shape=(1, 1), dtype="bf16")) is not None
        fused.disable_moe_block_compile()
        assert blocks[0]("plain") == "plain"
    finally:
        qwen4_type.__call__ = original
