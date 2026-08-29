"""焼く前にレシピの容量と KLD を見積もる (experts の層別を含む)。

`mlxturbo/rebit.py` は gdn/hc/attn/head/router/shared しか打ち直せない。
**experts が無い。**experts は v-fast6 の 91GB のうち 85.6GB を占めるので、
容量の話をするならここを動かせないと始まらない。experts は `SwitchLinear` の
3 次元バンク (512, out, in) で `nn.QuantizedLinear` ではないため、rebit の
`_requantize` は使えない。ここで別に持つ。

打ち直しは二重量子化なので、bf16 から直接焼いたものより悪く出る。STATUS の
実測で **rebit の予測は 14% 悲観側** (v-fast6 予測 0.00432 -> 実測 0.00378)。
つまり「これで保てるなら焼いても保てる」の向きに使う。

## 層の選び方も試せる

`--hi-layers` に spread / edge / first / last / mid を指定できる。同じ層数で
選び方だけ変えれば、**層別の配分に意味があるか**が同じ予算で比べられる
(`_spread` も `_FIRST5_LAST5` も folklore が根拠で、感度は未測定 — STATUS の
積み残し)。

    tools/biglock.sh uv run python tools/predict_recipe.py \\
        --model ~/models/qwen38fn-mlx-v-fast6 --ngram ~/models/qwen38fn-ngram-4bit \\
        --experts 4 --tag pred-v96ish --size-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)


def hi_layers(spec: str, n_layers: int) -> list[int]:
    """"spread:20" / "edge:10" / "first:10" / "last:10" / "mid:10" /
    "range:0-7" / "list:0,3,5" / "none"。"""
    if spec in ("none", ""):
        return []
    kind, _, num = spec.partition(":")
    if kind == "range":
        a, _, b = num.partition("-")
        return list(range(int(a), min(int(b) + 1, n_layers)))
    if kind == "list":
        return sorted({int(v) for v in num.split(",") if v.strip() != ""})
    n = int(num) if num else 0
    if kind == "spread":
        return sorted({round(i * n_layers / n) for i in range(n)}) if n else []
    if kind == "first":
        return list(range(min(n, n_layers)))
    if kind == "last":
        return list(range(max(0, n_layers - n), n_layers))
    if kind == "edge":
        h = n // 2
        return list(range(h)) + list(range(n_layers - (n - h), n_layers))
    if kind == "mid":
        s = (n_layers - n) // 2
        return list(range(s, s + n))
    raise ValueError(f"未知の層指定 {spec!r}")


def _expert_banks(model):
    """(層番号, 親, 属性名, 量子化された 3 次元バンク) を列挙する。"""
    for i, layer in enumerate(model.model.layers):
        sm = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if sm is None:
            continue
        for attr in ("gate_proj", "up_proj", "down_proj"):
            child = getattr(sm, attr, None)
            if child is not None and hasattr(child, "scales") and child.weight.ndim == 3:
                yield i, sm, attr, child


def requantize_experts(model, bits: int, group_size: int, layers, verbose=True):
    """指定した層の expert バンクを打ち直す。1 本ずつ捨てながら進める。"""
    import mlx.core as mx

    want = set(layers)
    n = 0
    saved = 0
    for idx, parent, attr, lin in list(_expert_banks(model)):
        if idx not in want:
            continue
        if lin.bits == bits and lin.group_size == group_size:
            continue
        if lin.bits < bits:
            # 上げ直しは情報を復元しない。黙って通すと誤った予測になるので飛ばす
            continue
        before = lin.weight.nbytes + lin.scales.nbytes + lin.biases.nbytes
        w = mx.dequantize(lin.weight, lin.scales, lin.biases,
                          group_size=lin.group_size, bits=lin.bits)
        q, s, b = mx.quantize(w, group_size=group_size, bits=bits)
        mx.eval(q, s, b)
        del w
        lin.weight, lin.scales, lin.biases = q, s, b
        lin.bits, lin.group_size = bits, group_size
        saved += before - (q.nbytes + s.nbytes + b.nbytes)
        n += 1
        if n % 12 == 0:
            mx.clear_cache()
    mx.clear_cache()
    if verbose:
        print(f"  experts: {n} 本を {bits}bit/gs{group_size} へ "
              f"({-saved / 1e9:+.2f} GB)", flush=True)


def total_bytes(model) -> float:
    from mlx.utils import tree_flatten

    return sum(a.nbytes for _, a in tree_flatten(model.parameters())
               if hasattr(a, "nbytes"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--experts", type=int, default=None,
                    help="全層の expert をこのビットへ")
    ap.add_argument("--experts-hi", type=int, default=None,
                    help="--hi-layers の層だけこのビットへ (--experts の後に適用)")
    ap.add_argument("--hi-layers", default="none",
                    help="spread:N / edge:N / first:N / last:N / mid:N / none")
    ap.add_argument("--lo-layers", default=None,
                    help="この層だけ --experts のビットへ落とし、他は触らない。"
                         "感度地図を取るときに使う (例 range:0-7)")
    ap.add_argument("--rebit", default=None, help="他クラス (例 head=4,attn=4)")
    ap.add_argument("--tag", default=None, help="KLD も測る場合のタグ")
    ap.add_argument("--size-only", action="store_true")
    args = ap.parse_args()

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    t0 = time.perf_counter()
    model, _ = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)
    n_layers = len(model.model.layers)
    before = total_bytes(model)
    print(f"読み込み {time.perf_counter() - t0:.0f}s  元の容量 {before / 1e9:.1f} GB"
          f"  ({n_layers} 層)")

    # **hi 層は「落とさない」ことで実現する。**先に全層を落としてから hi 層を
    # 上げ直すと 6bit -> 4bit -> 6bit の二重劣化になり、4bit の情報を 6bit の
    # 器に入れ直すだけになる (最初これをやって、hi 層を足したのに KLD が
    # 悪化するという結果を出した)。打ち直しは常にビットを下げる向きだけ。
    if args.lo_layers is not None:
        # 感度地図: 指定した層だけ落とす。他は v-fast6 のまま
        lo = hi_layers(args.lo_layers, n_layers)
        print(f"  落とす層 ({args.lo_layers}) = {lo}")
        requantize_experts(model, args.experts, 64, lo)
        after = total_bytes(model)
        print(f"\n予測容量 {after / 1e9:.1f} GB  ({(after - before) / 1e9:+.1f} GB)")
        if not args.size_only and args.tag:
            _evaluate(model, args, after)
        return

    sel = set(hi_layers(args.hi_layers, n_layers)) if args.experts_hi is not None else set()
    if args.experts_hi is not None:
        print(f"  hi 層 ({args.hi_layers}) = {sorted(sel)} は {args.experts_hi}bit 側")
    if args.experts is not None:
        requantize_experts(model, args.experts, 64,
                           [i for i in range(n_layers) if i not in sel])
    if args.experts_hi is not None:
        # hi 層は元が experts_hi 以上のときだけ落とす (上げ直さない)
        requantize_experts(model, args.experts_hi, 64, sorted(sel))
    if args.rebit:
        from mlxturbo import rebit

        rebit.apply(model, args.rebit)

    after = total_bytes(model)
    print(f"\n予測容量 {after / 1e9:.1f} GB  ({(after - before) / 1e9:+.1f} GB)"
          f"  128GB 機での空き 約 {128 - after / 1e9 - 3:.0f} GB")

    if args.size_only or not args.tag:
        return
    _evaluate(model, args, after)


def _evaluate(model, args, after):
    import json

    import numpy as np  # noqa: F401
    import quant_eval

    cont = json.loads((REPO_ROOT / "bench/results/qe-cont.json").read_text())
    ref = np.load(REPO_ROOT / "bench/results/qe-ref-bf16.npz")
    per_prompt = quant_eval.evaluate(model, cont, ref)
    summary = quant_eval.summarize(per_prompt)
    print(f"\n予測 KLD {summary['kld_mean']:.5f}  top1 {summary['top1_agree_mean']:.4f}"
          f"   (rebit は約 14% 悲観側。焼くと少し良く出る)")
    out = REPO_ROOT / f"bench/results/quant-eval/predict-{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "predict-recipe", "tag": args.tag, "model": args.model,
        "experts": args.experts, "experts_hi": args.experts_hi,
        "hi_layers": args.hi_layers, "rebit": args.rebit,
        "bytes": after, "gb": after / 1e9, **summary,
        "per_prompt": per_prompt,
    }, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
