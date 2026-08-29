"""焼いた段が実際に文章を生成できるかを目で見る。

KLD は teacher-forced の forward で測っているので、**実際の生成の質は見ていない**。
特に低ビット段は、KLD が許容範囲でも生成が崩れることがある (STATUS の
「RMSNorm の +1 欠落」は生成が無意味な反復になったが活性の大きさは
それらしいままだった)。数字だけで出荷しないための確認。

    tools/biglock.sh uv run python tools/smoke_generate.py --model ~/models/... --ngram ...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROMPTS = [
    "分散システムにおける結果整合性を、3 文で説明してください。",
    "次の関数のバグを指摘してください:\n\ndef mean(xs):\n    return sum(xs) / len(xs)",
    "What is the capital of Australia? Answer in one word.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--max-tokens", type=int, default=120)
    args = ap.parse_args()

    import os

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate

    from mlxturbo import fused

    model, tok = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)
    fused.enable_hyper_connection_kernel()

    for i, p in enumerate(PROMPTS):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True
        )
        t0 = time.perf_counter()
        out = generate(model, tok, prompt=tok.decode(ids), max_tokens=args.max_tokens,
                       verbose=False)
        dt = time.perf_counter() - t0
        print(f"\n--- プロンプト {i + 1} ({dt:.1f}s) ---\n{p}\n>>> {out.strip()[:600]}",
              flush=True)
    print(f"\npeak={mx.get_peak_memory() / 1e9:.1f}GB")


if __name__ == "__main__":
    main()
