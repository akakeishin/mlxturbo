"""Re-quantize parts of an already-loaded model to a different bit width in place.

A tool for trying out bit allocations without re-baking. The internal disk has
only 64GB free and each bake is 92GB, so we cannot casually add more of them;
this lets us settle the verdict first.

Re-quantizing goes 8bit -> dequantize -> 4bit, i.e. double quantization, so it
always comes out worse than baking to 4bit directly from bf16. That means it is
only usable for verdicts in the direction of "if quality holds up here, it will
hold up in the real bake". The converse does not follow.

RAM moves downward (the original weights are dropped before the new ones are
put in place).

    from mlxturbo import rebit
    rebit.apply(model, "gdn=4,hc=4")

The class names line up with the categories in byte_budget.py:
    gdn   GDN projections (in_proj_qkv/z/b/a, out_proj). The largest reader per token
    hc    the 3 hyper-connections
    attn  q/k/v/o of full attention (12 layers)
    head  lm_head
    router the MoE gate
    shared the MoE shared expert
"""

from __future__ import annotations

CLASSES = ("gdn", "hc", "attn", "head", "router", "shared")


def _targets(model, cls: str):
    """Enumerate (parent module, attribute name, quantized linear)."""

    def q(mod, attr):
        child = getattr(mod, attr, None)
        if child is not None and hasattr(child, "scales"):
            yield mod, attr, child

    if cls == "head":
        yield from q(model, "lm_head")
        return

    for layer in model.model.layers:
        if cls == "gdn":
            la = getattr(layer, "linear_attn", None)
            if la is None:
                continue
            for attr in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                         "out_proj"):
                yield from q(la, attr)
        elif cls == "hc":
            for hc_attr in ("attn_hyper_connection", "mlp_hyper_connection"):
                hc = getattr(layer, hc_attr, None)
                if hc is None:
                    continue
                for attr in ("input_mix_weight_down", "input_mix_weight_up",
                             "block_inject_weight"):
                    yield from q(hc, attr)
        elif cls == "attn":
            sa = getattr(layer, "self_attn", None)
            if sa is None:
                continue
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                yield from q(sa, attr)
        elif cls == "router":
            yield from q(layer.mlp, "gate")
        elif cls == "shared":
            se = getattr(layer.mlp, "shared_expert", None)
            if se is not None:
                for attr in ("gate_proj", "up_proj", "down_proj"):
                    yield from q(se, attr)
            yield from q(layer.mlp, "shared_expert_gate")


def _requantize(lin, bits: int, group_size: int):
    import mlx.core as mx
    import mlx.nn as nn

    w = mx.dequantize(
        lin.weight, lin.scales, lin.biases,
        group_size=lin.group_size, bits=lin.bits,
    )
    out_dims, in_dims = w.shape
    # If head_dim is not divisible by group_size, mlx silently skips the
    # quantization. That is the same trap we hit with 7.942 bpw, so fail here.
    if in_dims % group_size:
        raise ValueError(
            f"in_dims={in_dims} が group_size={group_size} で割り切れない"
        )
    new = nn.QuantizedLinear(
        in_dims, out_dims, bias=("bias" in lin),
        group_size=group_size, bits=bits,
    )
    new.weight, new.scales, *bi = mx.quantize(w, group_size=group_size, bits=bits)
    new.biases = bi[0] if bi else None
    if "bias" in lin:
        new.bias = lin.bias
    del w
    return new


def parse(spec: str) -> dict[str, tuple[int, int]]:
    """"gdn=4,hc=4:32" -> {"gdn": (4, 64), "hc": (4, 32)}"""

    out: dict[str, tuple[int, int]] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        cls, _, rhs = part.partition("=")
        bits, _, gs = rhs.partition(":")
        if cls not in CLASSES:
            raise ValueError(f"未知のクラス {cls!r} (使えるのは {CLASSES})")
        out[cls] = (int(bits), int(gs) if gs else 64)
    return out


def apply(model, spec: str, verbose: bool = True) -> None:
    import mlx.core as mx

    for cls, (bits, gs) in parse(spec).items():
        n = 0
        saved = 0
        for parent, attr, lin in list(_targets(model, cls)):
            if lin.bits == bits and lin.group_size == gs:
                continue
            before = lin.weight.nbytes + lin.scales.nbytes
            setattr(parent, attr, _requantize(lin, bits, gs))
            new = getattr(parent, attr)
            saved += before - (new.weight.nbytes + new.scales.nbytes)
            n += 1
        mx.clear_cache()
        if verbose:
            print(f"  rebit {cls}: {n} 本を {bits}bit/gs{gs} へ "
                  f"({saved / 1e9:+.2f} GB)", flush=True)


__all__ = ["CLASSES", "apply", "parse"]
