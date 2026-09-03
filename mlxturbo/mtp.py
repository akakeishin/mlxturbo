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


_QUANT_SUFFIXES = (".scales", ".biases")


def _read_sidecar(path: Path) -> tuple[dict, dict | None]:
    """Read an MTP sidecar's weights and, when there is one, its config.json.

    Accepts a single safetensors file or a directory holding config.json plus
    one or more safetensors shards. For a file we still look for a config.json
    next to it, so ``--mtp DIR`` and ``--mtp DIR/model.safetensors`` behave the
    same way.
    """
    if path.is_dir():
        index = path / "model.safetensors.index.json"
        if index.exists():
            shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
            files = [path / name for name in shards]
        else:
            files = sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors file under {path}")
        config_path = path / "config.json"
    else:
        files = [path]
        config_path = path.parent / "config.json"
    weights: dict = {}
    for f in files:
        weights.update(mx.load(str(f)))
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    return weights, config


def _packed_quantization(
    weights: dict, config: dict | None, args: TextModelArgs
) -> dict | None:
    """Quantization parameters of an already-quantized sidecar, None if it is bf16.

    A sidecar is already quantized when it carries .scales/.biases. group_size
    and bits come from config.json when there is one; otherwise they are read
    off the fc shapes (fc's input is 2 * hidden_size, so group_size is that
    divided by the number of scale columns, and bits is 32 * packed columns
    divided by it).
    """
    if not any(k.endswith(_QUANT_SUFFIXES) for k in weights):
        return None
    q = None
    if config:
        q = config.get("quantization") or config.get("quantization_config")
    if q and "group_size" in q and "bits" in q:
        return {
            "group_size": int(q["group_size"]),
            "bits": int(q["bits"]),
            "mode": q.get("mode", "affine"),
        }
    if "fc.weight" not in weights or "fc.scales" not in weights:
        raise ValueError(
            "quantized MTP sidecar without a config.json quantization block, and "
            "fc.weight/fc.scales are missing so it cannot be inferred either"
        )
    in_features = 2 * args.hidden_size
    return {
        "group_size": in_features // weights["fc.scales"].shape[-1],
        "bits": 32 * weights["fc.weight"].shape[-1] // in_features,
        "mode": "affine",
    }


def load_mtp_file(
    path: str, args: TextModelArgs, quantize: dict | None = None
) -> MTPModule:
    """Load a trained head: train_mtp.py's artifact, or an already-quantized
    MTP sidecar pack (a directory, or the model.safetensors inside it).

    The norm +1 shift was already applied at save time in both cases, so unlike
    load_mtp we do not shift at all here (avoiding the double-application
    landmine).

    An already-quantized sidecar has to be quantized *before* load_weights: the
    bf16 skeleton has no .scales/.biases for those tensors to land on, so
    load_weights rejects them. In that case the sidecar's own group_size/bits
    win and ``quantize`` is ignored. A bf16 sidecar keeps the original order
    (load, then quantize according to ``quantize``).
    """
    weights, config = _read_sidecar(Path(path))
    packed = _packed_quantization(weights, config, args)
    mtp = MTPModule(args)
    if packed is not None:
        if packed["mode"] == "affine":
            validate_affine_quantization(packed["group_size"], packed["bits"])
        nn.quantize(
            mtp,
            group_size=packed["group_size"],
            bits=packed["bits"],
            mode=packed["mode"],
            class_predicate=lambda _, m: isinstance(m, nn.Linear),
        )
        mtp.load_weights(list(weights.items()))
        mtp.eval()
        return mtp
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
