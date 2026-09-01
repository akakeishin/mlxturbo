"""MTP head for Qwen3.8-Flash-Next (qwen4_exp).

`mlxturbo/mtp.py` is for qwen3_5 (27B) and is built out of
`mlx_lm.models.qwen3_5.DecoderLayer`. Flash-Next has a different structure, so it
gets its own module here. The 27B side is left untouched.

## Structure read off the sidecar weights (31 tensors, bf16 5.21GB)

    pre_fc_norm_embedding   (2560,)          norm on the embedding side
    pre_fc_norm_hidden      (10240,)         **norm of the hyper state (4 lanes)**
    fc_embedding            (2560, 2560)     unlike 27B's single fc, there are two
    fc_hidden               (2560, 2560)
    layers.0                a complete DecoderLayer (full_attention + 512-expert MoE
                            + 2 hyper_connection + indexer)
    hyper_connection_mixer  GatedResidual(use_combine=False)

**It can be assembled from the same classes as the model proper.** q_proj
(12288, 2560) matches `n_heads(24) * head_dim(256) * 2` (output gate included),
k/v (512) matches `n_kv_heads(2) * 256`, and o_proj (2560, 6144) matches
`n_heads * head_dim`. The problem seen with 27B, where mlx_lm's sanitize inserts a
+1 into the norms, does not occur here (the vendored qwen4_exp's RMSNorm carries
`(1 + weight)` itself).

The norm convention is the same as the model proper: `(1 + weight)`. The grounds are
that the mean of `pre_fc_norm_embedding` is -0.764 and that of `pre_fc_norm_hidden`
is -0.328, i.e. both negative (the same way of establishing evidence that the 27B
`mlxturbo/mtp.py` used).

## How to combine was decided by measurement -> ``lane``

`pre_fc_norm_hidden` is 10240, yet the input of `fc_hidden` is 2560, so the weight
shapes alone do not pin down the combination. Two versions were implemented and
judged on real data (`tools/mtp_flash_probe.py`, v-l + 6 prompts).

- ``lane`` (the default): apply `fc_hidden` per lane to build the 10240, and add the
  output of `fc_embedding` replicated across the 4 lanes. Symmetric with the model
  proper's `h = mx.tile(h, (1, 1, hc))`
- ``mean``: collapse hyper to 2560 by averaging over the lanes, add, then replicate
  the result

| variant | bits | t+2 hit rate | mean logprob |
|---|---|---|---|
| **lane** | bf16 | 0.499 | **-3.2172** |
| mean | bf16 | 0.501 | -4.4764 |
| lane | 4 | 0.489 | -3.3411 |
| mean | 4 | 0.441 | -4.4526 |

**Even though the hit rates are almost the same, the log-likelihood differs by
1.26.** Looking only at argmax would not have distinguished them. That `mean`
collapses badly under quantization (0.501 -> 0.441) is also the brittleness of the
side whose formulation is wrong.

For reference: the model proper's hit rate for t+1 is 0.566. **It hits 2 tokens
ahead with accuracy close to what the model proper achieves 1 token ahead.**

## 4bit is enough for quantization

| bits | t+2 hit rate | mean logprob | size |
|---|---|---|---|
| bf16 | 0.499 | -3.2172 | 4.9GiB |
| 8 | 0.496 | -3.2627 | 2.4GiB |
| 6 | 0.491 | -3.2303 | 1.8GiB |
| **4** | **0.489** | **-3.3411** | **1.2GiB** |

Relative to bf16, the hit rate is -0.010 and logprob -0.124. **At 4bit, even added
on top of v-xl (90.8GiB), 4GiB of headroom remains against the 96GiB ceiling.**

Note that MTP is a draft, so **even if you lower the bits, the correctness of the
output is guaranteed by the model proper's verification.** What drops is only the
acceptance rate (= speed), not quality.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# spec_flash.py / batch_spec.py と 3 重定義されていたヘルパーを
# mlxturbo/arch.py に一本化 (qwen4_exp 固有のままにしてある -- 詳細は
# arch.py のモジュール docstring 参照)。
from .arch import qwen4_arch as _arch

VARIANTS = ("lane", "mean")


class FlashMTPModule(nn.Module):
    """A single-block draft head.

    The input at position i is `(embed(t_{i+1}), hyper_i)`, and passing the output
    through lm_head yields the prediction of `t_{i+2}` (DeepSeek-V3 style, the same
    convention as 27B).

    `hyper_i` is **the model proper's hyper state before it passes through the final
    mixer** (B, S, hc*d). That `pre_fc_norm_hidden` is 10240 is the grounds for it.
    """

    def __init__(self, args, variant: str = "lane"):
        super().__init__()
        Q = _arch()
        if variant not in VARIANTS:
            raise ValueError(f"variant は {VARIANTS} のいずれか: {variant!r}")
        self.variant = variant
        self.hc = args.hc_count
        self.d = args.hidden_size
        self.pre_fc_norm_embedding = Q.RMSNorm(self.d, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = Q.RMSNorm(self.hc * self.d, eps=args.rms_norm_eps)
        self.fc_embedding = nn.Linear(self.d, self.d, bias=False)
        self.fc_hidden = nn.Linear(self.d, self.d, bias=False)
        # Pass the layer number that becomes full_attention. Since
        # ple_layer_ids=[2], layer_idx=3 has no PLE (there are no PLE weights on
        # the MTP side)
        self.layers = [Q.DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]
        self.hyper_connection_mixer = Q.GatedResidual(args, use_combine=False)

    def combine(self, embeds: mx.array, hyper: mx.array) -> mx.array:
        """(embeds, hyper) -> the hyper state (B, S, hc*d) passed to layers[0]."""
        shape = embeds.shape[:-1]
        e = self.fc_embedding(self.pre_fc_norm_embedding(embeds))
        h = self.pre_fc_norm_hidden(hyper)
        if self.variant == "lane":
            lanes = self.fc_hidden(h.reshape(*shape, self.hc, self.d))
            return mx.tile(e, (1,) * len(shape) + (self.hc,)) + lanes.reshape(
                *shape, self.hc * self.d
            )
        mixed = h.reshape(*shape, self.hc, self.d).mean(axis=-2)
        return mx.tile(e + self.fc_hidden(mixed), (1,) * len(shape) + (self.hc,))

    def __call__(self, embeds, hyper, rope, mask=None, cache=None, idx_cache=None):
        x = self.combine(embeds, hyper)
        # A layer that has no ple, so ids / prev_ctx / conv_mask go unused
        x = self.layers[0](x, rope, mask, None, cache, idx_cache, None, None)
        return self.hyper_connection_mixer(x)


def _sanitize(weights: dict) -> dict:
    """Strip `mtp.` and split the experts across SwitchGLU's two sheets.

    Same convention as the model proper's `Qwen4ExpModel.sanitize` (in the fused
    (E, 2*inter, H), the first half of the rows is gate and the second half is up).
    """
    out = {}
    for k, v in weights.items():
        k = k[len("mtp."):] if k.startswith("mtp.") else k
        if k.endswith("mlp.experts.gate_up_proj"):
            base = k[: -len("experts.gate_up_proj")] + "switch_mlp."
            gate, up = mx.split(v, 2, axis=1)
            out[base + "gate_proj.weight"] = gate
            out[base + "up_proj.weight"] = up
            continue
        if k.endswith("mlp.experts.down_proj"):
            out[k[: -len("experts.down_proj")] + "switch_mlp.down_proj.weight"] = v
            continue
        out[k] = v
    return out


def load_flash_mtp(path: str | None, args, variant: str = "lane",
                   quantize: dict | None = None, weights: dict | None = None) -> FlashMTPModule:
    """Assemble from the extracted sidecar (the output of `convert_flash extract-mtp`).

    `quantize` is `{"group_size": 64, "bits": 4}`. MTP is a draft, so even if you
    lower the bits, **the correctness of the output is guaranteed by the model
    proper's verification**. What drops is only the acceptance rate (= speed), not
    quality.

    `weights` (dict | None): if given, `path` is not read at all and this dict
    (raw, un-sanitized tensors carrying the `mtp.` prefix) is used as-is — an entry
    point for collecting just the `mtp.*` keys out of the model proper's safetensors
    shards and handing them in (addition only: when it is not given, the behavior
    that goes through `path` is unchanged down to the last bit).
    """
    raw_weights = dict(weights) if weights is not None else dict(mx.load(str(Path(path))).items())
    sanitized_weights = _sanitize(raw_weights)
    mtp = FlashMTPModule(args, variant=variant)
    # **Load first, then quantize.** Quantizing first would leave it in a shape that
    # carries scales/biases, and the bf16 weights would not fit (the same order as
    # 27B's load_mtp)
    mtp.load_weights(list(sanitized_weights.items()))
    if quantize:
        gs = quantize.get("group_size", 64)
        bad = [
            (n, v.shape[-1])
            for n, v in tree_flatten(mtp.parameters())
            if n.endswith(".weight") and v.ndim >= 2 and v.shape[-1] % gs
        ]
        if bad:
            raise ValueError(
                "入力次元が group_size で割り切れない: "
                + ", ".join(f"{n}:K={k}" for n, k in bad)
            )
        # Not passing class_predicate = SwitchLinear (512 experts, the bulk of the
        # 5GB of bf16) gets quantized too. 27B narrows this down to nn.Linear only,
        # but for Flash-Next the experts are the main body, so narrowing would be
        # pointless
        nn.quantize(mtp, group_size=gs, bits=quantize.get("bits", 4), mode="affine")
    mtp.eval()
    return mtp


__all__ = ["FlashMTPModule", "VARIANTS", "load_flash_mtp"]
