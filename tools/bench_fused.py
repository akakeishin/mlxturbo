"""融合カーネルを積み上げてデコード速度を測る。

同じプロセスの中で構成を切り替えて交互に測るので、マシンのドリフトに強い
(別々の時刻に測った数字を比べてはいけない: docs/KERNEL-HANDOFF-HC.md)。
数値は同じ入力の logits 相対誤差で見る。

    tools/biglock.sh uv run python tools/bench_fused.py \
        --model ~/models/qwen38fn-mlx-v-fast6 --ngram ~/models/qwen38fn-ngram-4bit
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def measure(model, ids, n=25) -> float:
    import mlx.core as mx

    cache = model.make_cache()
    logits = model(mx.array(ids)[None], cache=cache)
    cur = int(mx.argmax(logits[0, -1], axis=-1))
    for _ in range(3):
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
    t0 = time.perf_counter()
    for _ in range(n):
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rebit", default=None,
                    help="測定の途中でビットを打ち直して前後を比べる (例 head=4)。"
                         "同一プロセス内なのでマシンのドリフトに強い")
    args = ap.parse_args()

    import os

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    from fastmlx import fused

    model, tok = load(args.model)
    if args.ngram:
        from fastmlx.ngram_stream import install

        install(model, args.ngram)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて説明してください。"}],
        add_generation_prompt=True,
    )
    full = mx.array(ids)[None]

    def logits():
        out = model(full, cache=model.make_cache())[0, -1].astype(mx.float32)
        mx.eval(out)
        return np.array(out)

    def off():
        fused.disable_hyper_connection_kernel()
        fused.disable_rms_norm_gated()

    CONFIGS = [
        ("融合なし", lambda: None),
        ("+ hyper-connections", fused.enable_hyper_connection_kernel),
        ("+ RMSNormGated", fused.enable_rms_norm_gated),
    ]

    print(f"peak={mx.get_peak_memory() / 1e9:.1f}GB  reps={args.reps}")
    off()
    base = logits()

    def report(name, arr):
        rel = np.linalg.norm(arr - base) / max(np.linalg.norm(base), 1e-9)
        top1 = "一致" if arr.argmax() == base.argmax() else "不一致"
        print(f"  {name:24s} 相対誤差 {rel:.6f}  top1 {top1}", flush=True)

    # 数値は**単独で**見る。積み上げだと、どのカーネルがずらしたのか分からない
    print("\n=== 数値: 単独 (一括 forward、最終位置の logits) ===")
    for name, enable in CONFIGS[1:]:
        off()
        enable()
        report(name.replace("+ ", "") + " のみ", logits())
    print("\n=== 数値: 積み上げ ===")
    off()
    for name, enable in CONFIGS:
        enable()
        report(name, logits())
    off()

    samples: dict[str, list[float]] = {n: [] for n, _ in CONFIGS}
    for r in range(args.reps):
        off()
        for name, enable in CONFIGS:
            enable()
            samples[name].append(measure(model, ids))
        print(f"  速度 ラウンド {r + 1} 完了", flush=True)
    off()

    print("\n=== 速度 (積み上げ、中央値) ===")
    prev = None
    for name, _ in CONFIGS:
        med = statistics.median(samples[name])
        spread = max(samples[name]) - min(samples[name])
        delta = "" if prev is None else f"  {med - prev:+6.2f} ms"
        print(f"  {name:24s} {med:6.2f} ms/token ({1000 / med:5.2f} tok/s)"
              f"  (振れ {spread:4.2f}){delta}")
        prev = med

    if not args.rebit:
        return

    # ビットを打ち直して同じ構成をもう一度測る。ロードし直さないので
    # マシンの状態が揃ったまま前後を比べられる
    from fastmlx import rebit

    before = statistics.median(samples[CONFIGS[-1][0]])
    off()
    for _, enable in CONFIGS:
        enable()
    rebit.apply(model, args.rebit)
    after = [measure(model, ids) for _ in range(args.reps)]
    med = statistics.median(after)
    print(f"\n=== rebit {args.rebit} (全カーネル有効のまま打ち直し) ===")
    print(f"  打ち直し前 {before:6.2f} ms/token ({1000 / before:5.2f} tok/s)")
    print(f"  打ち直し後 {med:6.2f} ms/token ({1000 / med:5.2f} tok/s)"
          f"  (振れ {max(after) - min(after):4.2f})  {med - before:+6.2f} ms")
    off()


if __name__ == "__main__":
    main()
