"""Qwen3.8 の MTP (multi-token prediction) ヘッド。

チェックポイントの mtp.* 重みから DeepSeek-V3 型の 1 ブロック draft ヘッドを
組み立てる。mlx-lm はこの重みを読み込み時に捨てるので、ここで別途読む。

norm の規約: qwen3_5 の plain RMSNorm は zero-centered で保存されている
(mlx-lm の sanitize が本体側へ +1 している)。mtp 側の plain norm 7 本にも
同じ +1 を適用する。pre_fc norm の生平均が負であることがこの規約の証拠。
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
        # layer_idx は full_attention になる位置を渡す
        self.layers = [DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, embeds: mx.array, hiddens: mx.array, cache=None) -> mx.array:
        """embeds, hiddens: (B, S, D)。戻り値は最終 norm 前の hidden (B, S, D)。

        位置 i の入力は (embed(t_{i+1}), h_i)。出力から lm_head で t_{i+2} を予測する。
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
