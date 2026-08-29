"""MTP 投機デコードが (1) 同じ出力を出すか (2) 速いか を測る。

受理は「draft が本体の argmax と一致したとき」だけなので、出力は貪欲生成と
一致するはず。ただし `mx.quantized_matmul` はバッチ長依存の丸めをするので、
幅 2 で計算した logits の argmax が幅 1 と稀に食い違う (spec.py の注記と同じ
性質)。**完全一致でなくても、大きく割れなければ想定内。**

    tools/biglock.sh uv run python tools/spec_flash_bench.py \\
        --model ~/models/qwen38fn-mlx-v-l --ngram ~/models/qwen38fn-ngram-4bit \\
        --mtp "/Volumes/Mobile SSD/models/qwen38fn-mtp.safetensors"
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

PROMPTS = [
    "分散システムにおける結果整合性について説明してください。",
    "Explain why speculative decoding helps when decoding is dispatch bound.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    import os

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    from fastmlx import fused, mtp_flash, spec_flash

    model, tok = load(args.model)
    if args.ngram:
        from fastmlx.ngram_stream import install

        install(model, args.ngram)
    fused.enable_hyper_connection_kernel()
    quant = {"group_size": 64, "bits": args.mtp_bits} if args.mtp_bits else None
    mtp = mtp_flash.load_flash_mtp(args.mtp, model.args.text, quantize=quant)
    eng = spec_flash.FlashSpecEngine(model, mtp)
    print(f"MTP {args.mtp_bits or 'bf16'}bit  peak={mx.get_peak_memory() / 1e9:.1f}GB")

    def greedy(ids, n):
        caches = model.make_cache()
        lg = model(ids, cache=caches)
        cur = mx.argmax(lg[:, -1], axis=-1).reshape(1, 1)
        out = []
        for _ in range(n):
            out.append(int(cur.item()))
            lg = model(cur, cache=caches)
            cur = mx.argmax(lg[:, -1], axis=-1).reshape(1, 1)
        return out

    for text in PROMPTS:
        ids = mx.array(tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True))[None]
        g = greedy(ids, args.tokens)
        s, acc, rounds = eng.generate(ids, args.tokens)
        same = sum(1 for x, y in zip(g, s) if x == y)
        first_diff = next((i for i, (x, y) in enumerate(zip(g, s)) if x != y), None)
        gt, st = [], []
        for _ in range(args.reps):
            t = time.perf_counter(); greedy(ids, args.tokens); gt.append(time.perf_counter() - t)
            t = time.perf_counter(); eng.generate(ids, args.tokens); st.append(time.perf_counter() - t)
        gm, sm = statistics.median(gt), statistics.median(st)

        # 1 反復の内訳を測る。深さを上げる価値はここで決まる:
        # draft が高いなら、深さを増やしても本体 forward の安さを活かせない
        import mlx.core as mx as _mx  # noqa

        print(f"\n--- {text[:38]} ---")
        print(f"  出力一致 {same}/{len(g)}"
              + (f" (最初の相違 {first_diff} 文字目)" if first_diff is not None else " (完全一致)"))
        print(f"  受理率 {acc}/{rounds} = {acc / max(rounds, 1):.3f}"
              f"   1 反復あたり {(len(s)) / max(rounds, 1):.2f} トークン")
        print(f"  貪欲 {args.tokens / gm:6.2f} tok/s   投機 {args.tokens / sm:6.2f} tok/s"
              f"   **{gm / sm:.2f}x**")
        # 状態が壊れていれば文章が崩れる。数値の一致率だけでは見えない
        print(f"  貪欲: {tok.decode(g)[:180]}")
        print(f"  投機: {tok.decode(s)[:180]}")


if __name__ == "__main__":
    main()
