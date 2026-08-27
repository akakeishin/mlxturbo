"""lm_head の実効帯域 109GB/s が、カーネルの問題かメモリ圧の問題かを切り分ける。

98GB のモデルを載せた状態で測ると 109GB/s しか出ない。同じマシンで
hyper-connections のカーネルは 227GB/s 出ているので、半分しか使えていない。

**モデルを載せずに lm_head の重みだけを読み込んで測る。**帯域が跳ねれば
原因はメモリ圧 (98GB 常駐によるページフォルト)、変わらなければカーネル側。

    uv run python tools/probe_lm_head_bw.py --model ~/models/qwen38fn-mlx-v-fast6
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def bench(fn, n=50, reps=5) -> float:
    import mlx.core as mx

    out = []
    for _ in range(reps):
        for _ in range(5):
            mx.eval(fn())
        t = time.perf_counter()
        for _ in range(n):
            mx.eval(fn())
        out.append((time.perf_counter() - t) / n * 1e6)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    import mlx.core as mx

    root = Path(args.model).expanduser()
    cfg = json.loads((root / "config.json").read_text())
    q = cfg["quantization"].get("lm_head", cfg["quantization"])
    bits, gs = q["bits"], q["group_size"]

    # lm_head のテンソルだけを拾う (モデル全体は載せない)
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
    print(f"lm_head だけを読み込んだ: {k} -> {n_out} {bits}bit gs={gs}"
          f"  重み {nbytes / 1e6:.0f}MB  常駐 {mx.get_peak_memory() / 1e9:.2f}GB")

    for m in (1, 2, 4):
        x = mx.random.normal((m, k)).astype(mx.bfloat16)
        mx.eval(x)
        us = bench(lambda: mx.quantized_matmul(
            x, w, scales=s, biases=b, transpose=True, group_size=gs, bits=bits))
        print(f"  M={m}  {us:8.2f} us  実効 {nbytes / us / 1000:6.1f} GB/s")

    # 比較: 同じ量のデータを素直に流したときに何 GB/s 出るか (帯域の天井)
    n_el = int(nbytes // 2)
    big = mx.zeros((n_el,), dtype=mx.bfloat16)
    mx.eval(big)
    us = bench(lambda: mx.sum(big), n=20, reps=3)
    print(f"\n  参考: 同量 ({nbytes / 1e6:.0f}MB) の総和  {us:8.2f} us"
          f"  実効 {nbytes / us / 1000:6.1f} GB/s  <- このマシンの実力")


if __name__ == "__main__":
    main()
