"""The supported mlx / mlx-lm compatibility boundary for mlxturbo.

mlxturbo intentionally relies on a small set of mlx-lm implementation details.  Keep
those imports and the executable version/signature checks here so an upstream change
fails loudly instead of silently changing speculative decoding semantics.
"""

from __future__ import annotations

import inspect
from importlib.metadata import version
from pathlib import Path

from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gated_delta import (
    compute_g,
    gated_delta_kernel,
    gated_delta_ops,
    gated_delta_update,
)
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs
from mlx_lm.utils import (
    create_model_card,
    get_total_parameters,
    hf_repo_to_path,
    load as mlx_lm_load,
    make_shards,
    quantize_model,
    save_config,
)

MLX_MIN = (0, 32, 2)
MLX_MAX_EXCLUSIVE = (0, 33, 0)
MLX_LM_MIN = (0, 31, 3)
MLX_LM_MAX_EXCLUSIVE = (0, 32, 0)
AFFINE_GROUP_SIZES = frozenset((32, 64, 128))
AFFINE_BITS = frozenset((2, 3, 4, 5, 6, 8))

# qwen3_5.TextModel.sanitize() shifts these zero-centred raw RMSNorm weights
# by +1 whenever mtp.* is present.  convert.py reverses exactly this set while
# saving so reloading the combined artifact applies the shift only once.
QWEN35_SHIFTED_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    out = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits or 0))
    return tuple((out + [0, 0, 0])[:3])


def _require_signature(fn, expected: tuple[str, ...]) -> None:
    actual = tuple(inspect.signature(fn).parameters)
    if actual[: len(expected)] != expected:
        raise RuntimeError(
            f"unsupported mlx-lm signature for {fn.__module__}.{fn.__name__}: "
            f"expected prefix {expected}, got {actual}"
        )


def validate_mlx_contract() -> None:
    """Fail fast when the installed upstream is outside the audited contract."""

    mlx_v = _version_tuple(version("mlx"))
    mlx_lm_v = _version_tuple(version("mlx-lm"))
    if not MLX_MIN <= mlx_v < MLX_MAX_EXCLUSIVE:
        raise RuntimeError(
            f"unsupported mlx version {version('mlx')}; expected >=0.32.2,<0.33"
        )
    if not MLX_LM_MIN <= mlx_lm_v < MLX_LM_MAX_EXCLUSIVE:
        raise RuntimeError(
            f"unsupported mlx-lm version {version('mlx-lm')}; "
            "expected >=0.31.3,<0.32"
        )

    _require_signature(
        gated_delta_update,
        ("q", "k", "v", "a", "b", "A_log", "dt_bias", "state", "mask"),
    )
    _require_signature(
        gated_delta_kernel,
        ("q", "k", "v", "g", "beta", "state", "mask"),
    )
    _require_signature(create_attention_mask, ("h", "cache"))
    _require_signature(create_ssm_mask, ("h", "cache"))
    _require_signature(
        quantize_model,
        ("model", "config", "group_size", "bits", "mode", "quant_predicate"),
    )
    _require_signature(
        mlx_lm_load,
        ("path_or_hf_repo", "tokenizer_config", "model_config", "adapter_path"),
    )
    for method in ("is_trimmable", "trim"):
        if not callable(getattr(KVCache, method, None)):
            raise RuntimeError(f"mlx-lm KVCache no longer implements {method}()")


def validate_affine_quantization(group_size: int, bits: int) -> None:
    if group_size not in AFFINE_GROUP_SIZES:
        raise ValueError(
            f"unsupported affine group_size={group_size}; "
            f"expected one of {sorted(AFFINE_GROUP_SIZES)}"
        )
    if bits not in AFFINE_BITS:
        raise ValueError(
            f"unsupported affine bits={bits}; expected one of {sorted(AFFINE_BITS)}"
        )


def validate_spec_model_contract(model) -> None:
    """Validate the Qwen3.5 model surface used by SpecEngine."""

    try:
        text = model.language_model
        inner = text.model
        layers = inner.layers
        inner.embed_tokens
        inner.norm
        inner.fa_idx
        inner.ssm_idx
        text.args.tie_word_embeddings
        text.make_cache
    except AttributeError as exc:
        raise TypeError(
            "SpecEngine requires the audited mlx-lm Qwen3.5 model contract"
        ) from exc
    if not layers:
        raise ValueError("SpecEngine requires at least one decoder layer")
    caches = text.make_cache()
    if len(caches) != len(layers):
        raise RuntimeError(
            f"mlx-lm cache/layer count mismatch: {len(caches)} != {len(layers)}"
        )
    for index, (layer, cache) in enumerate(zip(layers, caches)):
        for name in (
            "is_linear",
            "input_layernorm",
            "post_attention_layernorm",
            "mlp",
        ):
            if not hasattr(layer, name):
                raise RuntimeError(
                    f"mlx-lm layer {index} is missing the required {name} contract"
                )
        if layer.is_linear:
            for name in ("advance", "__getitem__", "__setitem__"):
                if not callable(getattr(cache, name, None)):
                    raise RuntimeError(
                        f"linear cache {index} is missing required {name}()"
                    )
            for name in (
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "conv1d",
                "A_log",
                "dt_bias",
                "norm",
                "out_proj",
                "sharding_group",
            ):
                if not hasattr(layer.linear_attn, name):
                    raise RuntimeError(
                        f"linear attention layer {index} is missing {name}"
                    )
        else:
            for name in ("is_trimmable", "trim"):
                if not callable(getattr(cache, name, None)):
                    raise RuntimeError(
                        f"attention cache {index} is missing required {name}()"
                    )


def resolve_local_model_path(path_or_repo: str) -> Path:
    """Return the already-loaded artifact's local directory without downloading."""

    path = Path(path_or_repo)
    return path if path.exists() else Path(hf_repo_to_path(path_or_repo))


validate_mlx_contract()
