"""The MTP (multi-token prediction) head of Qwen3.8.

Assembles a DeepSeek-V3 style single-block draft head from the checkpoint's
mtp.* weights. mlx-lm throws these weights away at load time, so we read them
separately here.

The norm convention: plain RMSNorm in qwen3_5 is stored zero-centered (mlx-lm's
sanitize adds +1 on the main model side). We apply the same +1 to the 7 plain
norms on the mtp side. The evidence for this convention is that the raw mean of
the pre_fc norm is negative.
"""

import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ._mlx_compat import (
    DecoderLayer,
    TextModelArgs,
    create_attention_mask,
    validate_affine_quantization,
)


class MTPModule(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        # Pass a layer_idx that lands on a full_attention position
        self.layers = [DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, embeds: mx.array, hiddens: mx.array, cache=None) -> mx.array:
        """embeds, hiddens: (B, S, D). Returns the hidden before the final norm, (B, S, D).

        The input at position i is (embed(t_{i+1}), h_i). From the output, lm_head
        predicts t_{i+2}.
        """
        x = self.fc(
            mx.concatenate(
                [self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(hiddens)],
                axis=-1,
            )
        )
        mask = create_attention_mask(x, cache)
        return self.layers[0](x, mask=mask, cache=cache)

    def head(self, hidden: mx.array, lm_head) -> mx.array:
        return lm_head(self.norm(hidden))


_SHIFT_NORM_KEYS = (
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
)


def load_mtp(
    original_repo_path: str, args: TextModelArgs, quantize: dict | None = None
) -> MTPModule:
    snap = Path(original_repo_path)
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    shards = sorted(
        {v for k, v in index["weight_map"].items() if k.startswith("mtp.")}
    )
    weights = {}
    for shard in shards:
        for k, v in mx.load(str(snap / shard)).items():
            if k.startswith("mtp."):
                weights[k.removeprefix("mtp.")] = v

    for k in list(weights):
        if any(k.endswith(sfx) for sfx in _SHIFT_NORM_KEYS) and weights[k].ndim == 1:
            weights[k] = weights[k] + 1.0

    mtp = MTPModule(args)
    mtp.load_weights(list(weights.items()))
    if quantize:
        group_size = quantize.get("group_size", 64)
        bits = quantize.get("bits", 4)
        validate_affine_quantization(group_size, bits)
        incompatible = [
            (name, value.shape[-1])
            for name, value in tree_flatten(mtp.parameters())
            if name.endswith(".weight")
            and value.ndim == 2
            and value.shape[-1] % group_size
        ]
        if incompatible:
            details = ", ".join(f"{name}:K={size}" for name, size in incompatible)
            raise ValueError(
                f"MTP Linear input dimensions must be divisible by group_size="
                f"{group_size}: {details}"
            )
        nn.quantize(
            mtp,
            group_size=group_size,
            bits=bits,
            mode="affine",
            class_predicate=lambda _, m: isinstance(m, nn.Linear),
        )
    mtp.eval()
    return mtp


def load_mtp_file(
    path: str, args: TextModelArgs, quantize: dict | None = None
) -> MTPModule:
    """Load a trained head (the artifact produced by train_mtp.py).

    The norm +1 shift was already applied at save time, so unlike load_mtp we do
    not shift at all here (avoiding the double-application landmine).
    """
    weights = dict(mx.load(str(path)).items())
    mtp = MTPModule(args)
    mtp.load_weights(list(weights.items()))
    if quantize:
        group_size = quantize.get("group_size", 64)
        bits = quantize.get("bits", 4)
        validate_affine_quantization(group_size, bits)
        nn.quantize(
            mtp,
            group_size=group_size,
            bits=bits,
            mode="affine",
            class_predicate=lambda _, m: isinstance(m, nn.Linear),
        )
    mtp.eval()
    return mtp


def find_snapshot(repo_id: str) -> str:
    pat = (
        Path.home()
        / ".cache/huggingface/hub"
        / ("models--" + repo_id.replace("/", "--"))
        / "snapshots/*"
    )
    matches = glob.glob(str(pat))
    if not matches:
        raise FileNotFoundError(f"no local snapshot for {repo_id}")
    return matches[0]
