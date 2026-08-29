"""Qwen3.8-Flash-Next (qwen4_exp) の MTP ヘッド。

`fastmlx/mtp.py` は qwen3_5 (27B) 用で、`mlx_lm.models.qwen3_5.DecoderLayer`
で組んでいる。Flash-Next は構造が違うので別に持つ。27B 側は触らない。

## サイドカーの重みから読み取った構造 (31 テンソル、bf16 5.21GB)

    pre_fc_norm_embedding   (2560,)          埋め込み側の norm
    pre_fc_norm_hidden      (10240,)         **hyper 状態 (4 レーン) の norm**
    fc_embedding            (2560, 2560)     27B の 1 本の fc とは違い 2 本ある
    fc_hidden               (2560, 2560)
    layers.0                完全な DecoderLayer (full_attention + 512 expert MoE
                            + hyper_connection 2 個 + indexer)
    hyper_connection_mixer  GatedResidual(use_combine=False)

**本体と同じクラスで組める。**q_proj (12288, 2560) は
`n_heads(24) * head_dim(256) * 2` (出力ゲート込み) と一致し、k/v (512) は
`n_kv_heads(2) * 256`、o_proj (2560, 6144) は `n_heads * head_dim` と一致する。
27B のように mlx_lm の sanitize が norm に +1 を入れる問題は起きない
(vendored qwen4_exp の RMSNorm が `(1 + weight)` を自分で持っている)。

norm の規約は本体と同じ `(1 + weight)`。根拠は `pre_fc_norm_embedding` の
平均が -0.764、`pre_fc_norm_hidden` が -0.328 と負であること (27B の
`fastmlx/mtp.py` が使ったのと同じ証拠の立て方)。

## 合成の仕方は実測で決めた -> ``lane``

`pre_fc_norm_hidden` は 10240 なのに `fc_hidden` の入力は 2560 で、重みの形
だけでは合成が決まらない。2 通り実装して実データで判定した
(`tools/mtp_flash_probe.py`、v-l + 6 プロンプト)。

- ``lane`` (既定): レーンごとに `fc_hidden` を適用して 10240 を作り、
  `fc_embedding` の出力を 4 レーンに複製して足す。本体の
  `h = mx.tile(h, (1, 1, hc))` と対称
- ``mean``: hyper をレーン平均で 2560 に潰してから足し、結果を複製する

| 変種 | bits | t+2 的中率 | 平均 logprob |
|---|---|---|---|
| **lane** | bf16 | 0.499 | **-3.2172** |
| mean | bf16 | 0.501 | -4.4764 |
| lane | 4 | 0.489 | -3.3411 |
| mean | 4 | 0.441 | -4.4526 |

**的中率はほぼ同じでも対数尤度が 1.26 違う。**argmax だけ見ていたら見分け
られなかった。`mean` が量子化で大きく崩れる (0.501 -> 0.441) のも、
定式化が間違っている側の脆さ。

参考: 本体が t+1 を当てる率は 0.566。**2 トークン先を、本体が 1 トークン先を
当てるのに近い精度で当てている。**

## 量子化は 4bit で十分

| bits | t+2 的中率 | 平均 logprob | サイズ |
|---|---|---|---|
| bf16 | 0.499 | -3.2172 | 4.9GiB |
| 8 | 0.496 | -3.2627 | 2.4GiB |
| 6 | 0.491 | -3.2303 | 1.8GiB |
| **4** | **0.489** | **-3.3411** | **1.2GiB** |

bf16 比で的中率 -0.010、logprob -0.124。**4bit なら v-xl (90.8GiB) に足しても
上限 96GiB に対して余白 4GiB 残る。**

なお MTP は draft なので、**ビットを下げても出力の正しさは本体の検証が
保証する。**落ちるのは受理率 (= 速度) だけで、品質ではない。
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

VARIANTS = ("lane", "mean")


def _arch():
    """vendored された qwen4_exp を返す (install-arch 済みの site-packages)。"""
    import mlx_lm.models.qwen4_exp as Q

    return Q


class FlashMTPModule(nn.Module):
    """1 ブロックの draft ヘッド。

    位置 i の入力は `(embed(t_{i+1}), hyper_i)` で、出力を lm_head に通すと
    `t_{i+2}` の予測になる (DeepSeek-V3 型、27B と同じ規約)。

    `hyper_i` は**本体の最終 mixer を通す前の hyper 状態** (B, S, hc*d)。
    `pre_fc_norm_hidden` が 10240 なのがその根拠。
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
        # full_attention になる層番号を渡す。ple_layer_ids=[2] なので
        # layer_idx=3 は PLE を持たない (MTP 側に PLE の重みは無い)
        self.layers = [Q.DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]
        self.hyper_connection_mixer = Q.GatedResidual(args, use_combine=False)

    def combine(self, embeds: mx.array, hyper: mx.array) -> mx.array:
        """(embeds, hyper) -> layers[0] へ渡す hyper 状態 (B, S, hc*d)。"""
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
        # ple を持たない層なので ids / prev_ctx / conv_mask は使われない
        x = self.layers[0](x, rope, mask, None, cache, idx_cache, None, None)
        return self.hyper_connection_mixer(x)


def _sanitize(weights: dict) -> dict:
    """`mtp.` を剥がし、expert を SwitchGLU の 2 枚へ割る。

    本体の `Qwen4ExpModel.sanitize` と同じ規約 (融合 (E, 2*inter, H) の
    行の前半が gate、後半が up)。
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


def load_flash_mtp(path: str, args, variant: str = "lane",
                   quantize: dict | None = None) -> FlashMTPModule:
    """抽出済みサイドカー (`convert_flash extract-mtp` の出力) から組む。

    `quantize` は `{"group_size": 64, "bits": 4}`。MTP は draft なので、
    ビットを下げても**出力の正しさは本体の検証が保証する**。落ちるのは
    受理率 (= 速度) だけで、品質ではない。
    """
    weights = _sanitize(dict(mx.load(str(Path(path))).items()))
    mtp = FlashMTPModule(args, variant=variant)
    # **読んでから量子化する。**先に量子化すると scales/biases を持つ形に
    # なってしまい、bf16 の重みが入らない (27B の load_mtp と同じ順序)
    mtp.load_weights(list(weights.items()))
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
        # class_predicate を渡さない = SwitchLinear (512 expert、bf16 5GB の
        # 大半) も量子化される。27B は nn.Linear だけに絞っているが、
        # Flash-Next は expert が本体なので絞ると意味が無い
        nn.quantize(mtp, group_size=gs, bits=quantize.get("bits", 4), mode="affine")
    mtp.eval()
    return mtp


__all__ = ["FlashMTPModule", "VARIANTS", "load_flash_mtp"]
