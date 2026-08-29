"""読み込み済みモデルの一部を、その場で別のビット数に打ち直す。

焼き直さずにビット配分を試すための道具。内蔵ディスクの空きが 64GB しかなく、
1 本 92GB の焼きを気軽に増やせないので、判定だけ先にこれで済ませる。

打ち直しは 8bit -> 逆量子化 -> 4bit の二重量子化なので、bf16 から直接 4bit に
焼いたものより必ず悪く出る。つまり「これで品質が保てるなら本番の焼きでも
保てる」という向きの判定にしか使わない。逆は言えない。

RAM は減る方向に動く (元の重みを捨ててから新しいのを置く)。

    from fastmlx import rebit
    rebit.apply(model, "gdn=4,hc=4")

クラス名は byte_budget.py の分類と揃えてある:
    gdn   GDN の投影 (in_proj_qkv/z/b/a, out_proj)。1 トークンあたり最大の読み手
    hc    hyper-connections の 3 本
    attn  full attention (12 層) の q/k/v/o
    head  lm_head
    router MoE のゲート
    shared MoE の共有エキスパート
"""

from __future__ import annotations

CLASSES = ("gdn", "hc", "attn", "head", "router", "shared")


def _targets(model, cls: str):
    """(親モジュール, 属性名, 量子化線形) を列挙する。"""

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
    # head_dim が group_size で割り切れないと mlx は黙って量子化を飛ばす。
    # 7.942 bpw を踏んだのと同じ罠なので、ここで落とす
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
