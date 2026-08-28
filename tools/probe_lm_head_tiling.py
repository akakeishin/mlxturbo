"""lm_head の巨大 N (248320) が、タイル分割の問題かどうかを直接見る。

親の指示 (別実装が `lm_head_n == 248320` の分岐を持ち「巨大な N では細かい
タイルの split-K グリッドがスケジューラを取り合って潰し合う」と書いている)
の検証。

**N を分割して同じバイト数を動かし、速くなるかを見る。**1 本の
(248320 x 2560) と、k 本に割った (248320/k x 2560) x k は動くバイト数が同じ。
分割した方が速いなら、巨大 N のタイル問題が実在する。起動回数は増えるので、
それでも速いなら効果は本物。

既定ではモデルを載せない (lm_head の重みだけ読む)。`--with-model` を付けると
本体 91GB を常駐させた状態で同じ掃引をする。**タイル問題がメモリ圧の下でだけ
出る可能性**を潰すため。非常駐で分割が効かず、常駐でも効かないなら、
遅さの原因はタイルではなく圧。

    uv run python tools/probe_lm_head_tiling.py --model ~/models/qwen38fn-mlx-v-fast6
    tools/biglock.sh uv run python tools/probe_lm_head_tiling.py \
        --model ~/models/qwen38fn-mlx-v-fast6 --with-model \
        --ngram ~/models/qwen38fn-ngram-4bit
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

CHAIN = 16


def bench_chain(make, n=10, reps=5) -> float:
    """CHAIN 本積んで 1 回 eval。mx.eval 1 回の固定費 (約 160us) を償却する。"""
    def go():
        return [make(i) for i in range(CHAIN)]

    for _ in range(2):
        mx.eval(go())
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(n):
            mx.eval(go())
        out.append((time.perf_counter() - t) / n * 1e6 / CHAIN)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits", default="1,2,4,8,16")
    ap.add_argument("--with-model", action="store_true",
                    help="本体を常駐させた状態で測る (メモリ圧の下での挙動)")
    ap.add_argument("--ngram", default=None)
    args = ap.parse_args()

    keep = None
    if args.with_model:
        import os
        import sys

        if args.ngram:
            os.environ["FASTMLX_NGRAM_DISK"] = "1"
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from mlx_lm import load

        keep, _ = load(args.model)
        if args.ngram:
            from fastmlx.ngram_stream import install

            install(keep, args.ngram)
        mx.eval(keep.parameters())
        print(f"本体を常駐させた: peak={mx.get_peak_memory() / 1e9:.1f}GB")

    root = Path(args.model).expanduser()
    cfg = json.loads((root / "config.json").read_text())
    q = cfg["quantization"].get("lm_head", cfg["quantization"])
    bits, gs = q["bits"], q["group_size"]

    want = {"lm_head.weight", "lm_head.scales", "lm_head.biases"}
    found = {}
    for f in sorted(glob.glob(str(root / "*.safetensors"))):
        d = mx.load(f)
        for k in want & set(d):
            found[k] = d[k]
        if want <= set(found):
            break
    w, s, b = found["lm_head.weight"], found["lm_head.scales"], found["lm_head.biases"]
    mx.eval(w, s, b)
    n_out, k_words = w.shape
    k = k_words * 32 // bits
    nbytes = k * n_out * bits / 8 + s.size * 2 + b.size * 2
    print(f"lm_head {k} -> {n_out}  {bits}bit gs={gs}  {nbytes / 1e6:.0f}MB"
          f"  常駐 {mx.get_peak_memory() / 1e9:.2f}GB\n")

    xs = [(mx.random.normal((1, 1, k))).astype(mx.bfloat16) for _ in range(CHAIN)]
    mx.eval(xs)

    print(f"{'分割数':>6s} {'1 本あたりの N':>14s} {'合計時間':>10s} {'実効 GB/s':>10s} {'比':>7s}")
    base = None
    for nsp in [int(v) for v in args.splits.split(",")]:
        if n_out % nsp:
            continue
        per = n_out // nsp
        parts = [(w[i * per:(i + 1) * per], s[i * per:(i + 1) * per],
                  b[i * per:(i + 1) * per]) for i in range(nsp)]
        mx.eval([a for p in parts for a in p])

        def go(i, parts=parts):
            # 同じ x に対して分割した重みを順に掛ける (合計バイト数は同じ)
            return [mx.quantized_matmul(xs[i], pw, scales=ps, biases=pb,
                                        transpose=True, group_size=gs, bits=bits)
                    for pw, ps, pb in parts]

        us = bench_chain(go)
        if base is None:
            base = us
        print(f"{nsp:6d} {per:14d} {us:9.1f}us {nbytes / us / 1000:10.1f}"
              f" {base / us:6.2f}x")

    print("\n分割した方が速いなら、巨大 N のタイル問題が実在する"
          "(起動回数は増えているのに速い、ということなので)。")


if __name__ == "__main__":
    main()
