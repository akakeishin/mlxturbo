"""Executable startup contract for fastmlx's audited mlx-lm internals."""

from fastmlx._mlx_compat import (
    MLX_LM_MAX_EXCLUSIVE,
    MLX_LM_MIN,
    MLX_MAX_EXCLUSIVE,
    MLX_MIN,
    QWEN35_SHIFTED_NORM_SUFFIXES,
    validate_mlx_contract,
)


def test_supported_contract():
    validate_mlx_contract()
    assert MLX_MIN == (0, 32, 2)
    assert MLX_MAX_EXCLUSIVE == (0, 33, 0)
    assert MLX_LM_MIN == (0, 31, 3)
    assert MLX_LM_MAX_EXCLUSIVE == (0, 32, 0)
    assert QWEN35_SHIFTED_NORM_SUFFIXES == (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )


if __name__ == "__main__":
    test_supported_contract()
    print("[test_mlx_compat] contract passed")

