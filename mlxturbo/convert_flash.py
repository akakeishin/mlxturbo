"""Per-class recipe conversion driver for Qwen3.8-Flash-Next (qwen4_exp).

Phase Q (docs/STATUS.md). Bakes a mixed-precision MLX checkpoint that fits on a
128GB Mac from the bf16 archive (on an external SSD). The recipes are justified
by the byte-budget table in the tensor ledger (the Phase Q section of STATUS),
and are updated by the sensitivity scans that follow.

Subcommands:
  estimate      Estimate the post-recipe size without reading the weights
                (headers only)
  extract-mtp   Extract the mtp.* tensors into a sidecar safetensors file
                (left as bf16 — quantization is left to --mtp-bits at engine
                load time, the same convention as 27B)
  convert       Convert the model proper with mlx_lm.convert plus a per-class
                quant_predicate
  build-ngram   Quantize the n-gram tables and write them to a sidecar (for
                disk-resident operation)

The qwen4_exp model class becomes resolvable automatically as soon as
`mlxturbo` is imported (`mlxturbo._arch_registry`). The `install-arch`
subcommand that used to exist (physically copying the vendored qwen4_exp.py
into the user's site-packages) has been removed — it had the side effect of
rewriting the user's mlx_lm package. See the module docstring of
_arch_registry.py for details.

Usage (examples):
  uv run python -m mlxturbo.convert_flash estimate --recipe v0-95
  uv run python -m mlxturbo.convert_flash extract-mtp \
      --src ~/models/Qwen3.8-Flash-Next \
      --out ~/models/qwen38fn-mtp.safetensors
  uv run python -m mlxturbo.convert_flash convert --recipe v0-95 \
      --src ~/models/Qwen3.8-Flash-Next \
      --out ~/models/qwen38fn-mlx-v0-95
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------
# Class detection uses the same rules for both module paths (at quantize time)
# and tensor names (at estimate time). The qwen4_exp path structure:
#   experts:  model.layers.N.mlp.switch_mlp.{gate_up_proj,down_proj}
#   router:   model.layers.N.mlp.gate  (not quantized — same as the model default)
#   shared:   model.layers.N.mlp.shared_expert.* / shared_expert_gate
#   ngram:    model.layers.N.ple_embedding.ngram_embedding.shard_i
#   gdn:      model.layers.N.linear_attn.*
#   qsa:      model.layers.N.self_attn.* (including the indexer)
#   embed:    model.embed_tokens / lm_head

# Per-layer override: the experts of the layers listed in "experts_hi_layers"
# are baked at the "experts_hi" bit width (this started from the folklore that
# the entry and exit layers are the sensitive ones; to be replaced by per-layer
# KLD from the sensitivity scan). 48 layers (0..47).
_FIRST5_LAST5 = list(range(5)) + list(range(43, 48))
_FIRST6_LAST6 = list(range(6)) + list(range(42, 48))


def _spread(n: int) -> list[int]:
    """Pick n layers out of 48 at even intervals (not biased toward the entry
    and exit layers).

    Up through v-exp6 the picks were clustered at the ends, following the
    folklore that "the entry and exit layers are what matter", but per-layer
    sensitivity has never been measured. Beyond 10 layers the justification for
    clustering at the ends gets even thinner, so for configurations that put
    many layers at 6bit, use unbiased even spacing.
    """

    return sorted({round(i * 48 / n) for i in range(n)})

RECIPES: dict[str, dict] = {
    # Everyday use (~96GB): fits within the default GPU wired limit, KV included
    "v0-95": {
        "experts": {"bits": 4, "group_size": 64},
        "ngram": {"bits": 3, "group_size": 32},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # +6.4GB spent on the n-gram tables (~102GB). One half of the equal-byte
    # A/B against v-exp6
    "v0-105": {
        "experts": {"bits": 4, "group_size": 64},
        "ngram": {"bits": 4, "group_size": 32},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # Spends the same +6.3GB on 6bit experts in 10 layers instead (~102GB).
    # Compared against v0-105 by KLD at equal budget, to decide whether to pile
    # the bits onto the n-gram tables or onto the experts
    "v-exp6": {
        "experts": {"bits": 4, "group_size": 64},
        "experts_hi": {"bits": 6, "group_size": 64},
        "experts_hi_layers": _FIRST5_LAST5,
        "ngram": {"bits": 3, "group_size": 32},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # Configuration that evicts the n-gram tables from RAM (~98GB). The tables
    # live in a sidecar, and the whole 19.2GB freed up goes to the experts,
    # raising the number of 6bit layers from 10 to 40.
    # To use it, build the sidecar with build-ngram and, after loading, call
    # mlxturbo.ngram_stream.install(model, <sidecar>)
    "v-stream": {
        "experts": {"bits": 4, "group_size": 64},
        "experts_hi": {"bits": 6, "group_size": 64},
        "experts_hi_layers": _spread(40),
        "ngram": False,
        "ngram_disk": True,
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # Starting from v-stream, drops only the read-heavy classes to 6bit. There
    # was no justification at the time for choosing 8bit as the default; it was
    # just everything other than the experts lumped together.
    #
    # The bit widths were decided by measuring the cost of each step one at a
    # time in a cumulative rebit sweep
    # (bench/results/quant-eval/sweep-vstream-6bit.json). KLD increase from a
    # base of 0.00260:
    #   gdn 8->6     +0.00063   reads -0.554 GB  <- the best deal
    #   head 8->6    +0.00041   reads -0.169 GB
    #   attn 8->6    +0.00082   reads -0.159 GB
    #   shared 8->6  -0.00015   reads -0.063 GB  (noise; no sensitivity)
    #   hc 8->4      +0.01337   <- too expensive to take. Stays at 8bit
    #
    # hyper-connections cannot be set to 6bit because the fused kernel only
    # accepts 4/8. 4bit triples the KLD in a single step, so it stays at 8bit.
    # The default class (norm/embed_tokens/PLE/indexer) is barely read at all
    # per token, so lowering it does nothing for speed. Keeping it at 8bit also
    # avoids mixing in a change that was never swept
    "v-fast6": {
        "experts": {"bits": 4, "group_size": 64},
        "experts_hi": {"bits": 6, "group_size": 64},
        "experts_hi_layers": _spread(40),
        "ngram": False,
        "ngram_disk": True,
        "router": False,
        "gdn": {"bits": 6, "group_size": 64},
        "head": {"bits": 6, "group_size": 64},
        "attn": {"bits": 6, "group_size": 64},
        "shared": {"bits": 6, "group_size": 64},
        "hc": {"bits": 8, "group_size": 64},
        "default": {"bits": 8, "group_size": 64},
    },
    # Starting from v-stream, drops only the GDN projections to 4bit. The GDN
    # projections are the single biggest item, accounting for 34.3% of the bytes
    # read per token (tools/byte_budget.py), and since the marginal cost here is
    # bandwidth itself, whatever is cut translates directly into time. The
    # advance estimate from rebit is -3.27 ms/token (19.55 -> 20.88 tok/s).
    # Check quality by KLD before baking
    "v-fast": {
        "experts": {"bits": 4, "group_size": 64},
        "experts_hi": {"bits": 6, "group_size": 64},
        "experts_hi_layers": _spread(40),
        "ngram": False,
        "ngram_disk": True,
        "router": False,
        "gdn": {"bits": 4, "group_size": 64},
        "default": {"bits": 8, "group_size": 64},
    },
    # For 96GB Macs. Assuming the n-gram tables can be evicted to disk, this
    # keeps the experts at 4bit and drops the default from 8 to 4bit to fit.
    #
    # hc and gdn are not set to 4bit, though. In the sweep, hc 8->4 stands out
    # at KLD +0.01337 (larger than everything else combined) while its storage
    # is only 0.7GB, so there is nothing to gain in capacity. For gdn as well,
    # 8->4 costs +0.00663 whereas 8->6 costs +0.00063 — paying +0.55GB keeps it
    # to a tenth
    "v-96": {
        "experts": {"bits": 4, "group_size": 64},
        "ngram": False,
        "ngram_disk": True,
        "router": False,
        "hc": {"bits": 8, "group_size": 64},
        "gdn": {"bits": 6, "group_size": 64},
        "default": {"bits": 4, "group_size": 64},
    },
    # For 64GB Macs (~48GB). Puts half the experts at 3bit and half at 2bit.
    # The experts stand out as the dominant error source, so quality degrades
    # substantially. This configuration prioritizes running at all, and ships
    # with quality numbers attached.
    # Since cutting the experts is the whole point, the places outside the
    # experts where bits are cheap and effective are restored. hc at 8bit and
    # gdn at 6bit cost +0.9GB (2% of 48GB). Looking at the costs from the sweep,
    # putting these at 4bit would be far too expensive a purchase for the
    # capacity it buys
    "v-64": {
        "experts": {"bits": 2, "group_size": 64},
        "experts_hi": {"bits": 3, "group_size": 64},
        "experts_hi_layers": _spread(24),
        "ngram": False,
        "ngram_disk": True,
        "router": False,
        "hc": {"bits": 8, "group_size": 64},
        "gdn": {"bits": 6, "group_size": 64},
        "default": {"bits": 4, "group_size": 64},
    },
    # For validating n-gram at 2bit (~99GB). Keeps the experts identical to
    # v-exp6 and drops only the n-gram bits from 3 to 2. If the KLD does not
    # degrade much from v-exp6 (0.00181), then 2bit is enough for the n-gram
    # tables and 6.4GB can be handed to the experts
    "v-ng2": {
        "experts": {"bits": 4, "group_size": 64},
        "experts_hi": {"bits": 6, "group_size": 64},
        "experts_hi_layers": _FIRST5_LAST5,
        "ngram": {"bits": 2, "group_size": 32},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # The main candidate if v-ng2 passes (~112GB): hold the n-gram tables at
    # 2bit and hand all the freed capacity to the experts, raising the number of
    # 6bit layers from 10 to 32
    "v-exp-max": {
        "experts": {"bits": 4, "group_size": 64},
        "experts_hi": {"bits": 6, "group_size": 64},
        "experts_hi_layers": _spread(32),
        "ngram": {"bits": 2, "group_size": 32},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
    # The right-at-the-edge configuration (~112GB): n-gram at 4bit + experts at
    # 6bit in 12 layers. Assumes a raised iogpu.wired_limit_mb and a dedicated
    # machine. The 115GB range starts to squeeze the OS, so this is the ceiling.
    # The number of 6bit layers was tightened from 16 to 12 in order to absorb,
    # within that ceiling, the +3.2GB that came from having to put the n-gram
    # tables at group_size=32 (head_dim=160 is not divisible by 64)
    "v-max-112": {
        "experts": {"bits": 4, "group_size": 64},
        "experts_hi": {"bits": 6, "group_size": 64},
        "experts_hi_layers": _FIRST6_LAST6,
        "ngram": {"bits": 4, "group_size": 32},
        "router": False,
        "default": {"bits": 8, "group_size": 64},
    },
}

_LAYER_RE = None


def _layer_index(path: str) -> int | None:
    global _LAYER_RE
    if _LAYER_RE is None:
        import re

        _LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    m = _LAYER_RE.search(path)
    return int(m.group(1)) if m else None


def resolve_rule(recipe: dict, path: str):
    """Path -> the quantization rule to apply to that tensor/module.

    The finer-grained classes (gdn/attn/head/hc) fall back to default if the
    recipe does not mention them, so the behavior of existing recipes is
    unchanged.
    """

    c = classify(path)
    if c == "experts":
        hi_layers = recipe.get("experts_hi_layers")
        if hi_layers is not None and _layer_index(path) in hi_layers:
            return recipe["experts_hi"]
    if c in recipe:
        return recipe[c]
    return recipe["default"]


def classify(path: str) -> str:
    """Module path / tensor name -> recipe class.

    The classification is kept aligned with tools/byte_budget.py. Bytes read per
    token break down as GDN projections 34.3% / experts 28.1% /
    hyper-connections 10.5% / lm_head 10.5%, and everything other than the
    experts used to land in default (8bit).
    """

    if "ngram" in path:
        return "ngram"
    if path.endswith("mlp.gate") or ".mlp.gate." in path:
        return "router"
    if "switch_mlp" in path or ".experts." in path:
        return "experts"
    if "shared_expert" in path:
        return "shared"
    if "linear_attn" in path:
        return "gdn"
    if "self_attn" in path:
        return "attn"
    if "hyper_connection" in path:
        return "hc"
    if "lm_head" in path:
        return "head"
    return "default"


def validate_recipe(recipe_name: str) -> None:
    """Before baking, reject settings that do not mesh with the other lanes.

    The fused hyper-connections kernel (mlxturbo/kernels/hyper_connection.py)
    restricts bits to 4/8 in `eligible()`. Specifying 6bit simply falls back to
    the stock implementation without raising or warning, and the 16ms of
    hyper-connections comes straight back. That would leave you hunting for why
    the finished bake is slow, so stop it here.
    """

    recipe = RECIPES[recipe_name]
    # Resolve through an actual path, so that the case of implicitly falling
    # through from default is caught too
    hc = resolve_rule(recipe, "model.layers.0.attn_hyper_connection.input_mix_weight_down")
    if isinstance(hc, dict) and hc.get("bits") not in (4, 8):
        raise SystemExit(
            f"レシピ {recipe_name}: hyper-connections に bits={hc.get('bits')} は"
            "使えない。融合カーネルが 4/8 しか受けないので、素の実装に落ちて "
            "hyper-connections の 16ms が戻る。\n"
            '  レシピに "hc": {"bits": 8, "group_size": 64} を明示して、'
            "default から外すこと"
        )


def build_predicate(recipe_name: str):
    recipe = RECIPES[recipe_name]

    def predicate(path, module, *_config):
        del module
        return resolve_rule(recipe, path)

    return predicate


# ---------------------------------------------------------------------------
# Reading safetensors headers (without reading the weights themselves)
# ---------------------------------------------------------------------------


def iter_tensor_headers(src: Path):
    """Enumerate (tensor_name, dtype, shape, shard_path) across all shards."""

    for shard in sorted(src.glob("model-*.safetensors")):
        with open(shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            yield name, info["dtype"], info["shape"], shard


_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1}


def cmd_estimate(args):
    recipe = RECIPES[args.recipe]
    totals: dict[str, float] = {}
    for name, dtype, shape, _ in iter_tensor_headers(Path(args.src)):
        if name.startswith(("mtp.", "vision_tower.", "model.visual.")):
            continue
        n = 1
        for d in shape:
            n *= d
        c = classify(name)
        if c == "ngram" and recipe.get("ngram_disk"):
            continue  # goes into the sidecar, so it is not part of the model
        rule = resolve_rule(recipe, name)
        if rule is False or len(shape) < 2 or shape[-1] % rule["group_size"]:
            # Things that stay unquantized: router / norm / 1-D tensors, plus
            # anything whose input dimension is not divisible by group_size.
            # mlx's quantize silently passes those through (we hit this with the
            # n-gram head_dim=160). That would make the ledger diverge from
            # measurement, so apply the same rule here as well
            b = n * _DTYPE_BYTES.get(dtype, 2)
        else:
            # effective bits/weight = bits + 16*2/group_size (scale+bias are bf16)
            eff = rule["bits"] + 32 / rule["group_size"]
            b = n * eff / 8
        totals[c] = totals.get(c, 0.0) + b
    for c, b in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{c:10s} {b / 1e9:8.1f} GB")
    print(f"{'TOTAL':10s} {sum(totals.values()) / 1e9:8.1f} GB")


def cmd_extract_mtp(args):
    import mlx.core as mx

    # Put this on the CPU for the same reason as convert (reading an mmap on the
    # external SSD from the GPU makes the command buffer hit the watchdog
    # timeout)
    mx.set_default_device(mx.cpu)
    src = Path(args.src)
    picked: dict[str, object] = {}
    shard_names: dict[Path, list[str]] = {}
    for name, _, _, shard in iter_tensor_headers(src):
        if name.startswith("mtp."):
            shard_names.setdefault(shard, []).append(name)
    for shard, names in sorted(shard_names.items()):
        tensors = mx.load(str(shard))
        for n in names:
            picked[n] = tensors[n]
        del tensors
    if not picked:
        raise SystemExit("mtp.* テンソルが見つからない")
    mx.save_safetensors(args.out, picked)
    total = sum(v.nbytes for v in picked.values())
    print(f"extracted {len(picked)} tensors, {total / 1e9:.2f} GB -> {args.out}")


def cmd_convert(args):
    import os

    validate_recipe(args.recipe)
    if RECIPES[args.recipe].get("ngram_disk"):
        # The vendored arch reads this flag at import time. Set it before
        # touching mlx_lm
        os.environ["FASTMLX_NGRAM_DISK"] = "1"
        print("n-gram はディスク運用: 本体には入れない")

    import mlx.core as mx
    from mlx_lm.convert import convert

    if args.device == "cpu":
        # CPU is the default. The weights are an mmap on the external SSD, so
        # quantizing on the GPU means page-ins during kernel execution go over
        # USB, and Metal's command buffer dies on the watchdog timeout
        # (kIOGPUCommandBufferCallbackErrorTimeout, reproduced while saving
        # shard 2). The CPU has no such watchdog
        mx.set_default_device(mx.cpu)

    # Class resolution for qwen4_exp always goes through mlxturbo._arch_registry
    # via sys.meta_path, reading the vendored file (mlxturbo/_vendor/qwen4_exp.py)
    # directly. Since there is no copy in site-packages, mixing up copies cannot
    # happen in the first place (the accident from the old install-arch days,
    # where a stale copy stuck around, FASTMLX_NGRAM_DISK never took effect, and
    # the bake ballooned to 169GB)
    convert(
        hf_path=args.src,
        mlx_path=args.out,
        quantize=True,
        # Before calling the predicate, mlx_lm.quantize_model applies a cutoff:
        #   module.weight.shape[-1] % <this group_size> != 0 -> do not quantize
        # If this is 64, the n-gram tables (head_dim=160) get dropped before
        # they ever reach the recipe's group_size=32 and stay bf16 (51B params
        # were left as-is, the result came to 178GB, and it OOM'd). Pass the
        # smallest value present in the recipe.
        q_group_size=min(
            r["group_size"]
            for r in RECIPES[args.recipe].values()
            if isinstance(r, dict)
        ),
        q_bits=4,
        quant_predicate=build_predicate(args.recipe),
    )
    print(f"converted -> {args.out}")


def cmd_build_ngram(args):
    from .ngram_stream import build_sidecar

    build_sidecar(
        Path(args.src), Path(args.out), bits=args.bits, group_size=32,
        layout=args.layout,
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("estimate")
    p.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    p.add_argument("--src", required=True)
    p.set_defaults(fn=cmd_estimate)

    p = sub.add_parser("extract-mtp")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_extract_mtp)

    p = sub.add_parser("build-ngram")
    p.add_argument("--src", required=True, help="bf16 アーカイブ")
    p.add_argument("--out", required=True, help="サイドカーの出力先")
    p.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 5, 6, 8))
    p.add_argument(
        "--layout", default="interleaved", choices=("interleaved", "separate"),
        help="interleaved=ディスク常駐向け / separate=RAM 常駐向け",
    )
    p.set_defaults(fn=cmd_build_ngram)

    p = sub.add_parser("convert")
    p.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    p.set_defaults(fn=cmd_convert)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
