"""受理率を言語・課題別に、意味のあるサンプル数で測る。

`tools/spec_flash_bench.py` は 1 プロンプト 48 トークン (反復 30 前後) しか
回さないので、受理率の標準誤差が ±0.09 になる。日本語 0.741 と英語 0.516 の
差は**その幅に埋もれる**ので、本物かどうかはこれで測る。

    tools/biglock.sh uv run python tools/spec_flash_accept.py \\
        --model ~/models/qwen38fn-mlx-v-l --ngram ~/models/qwen38fn-ngram-4bit \\
        --mtp "/Volumes/Mobile SSD/models/qwen38fn-mtp.safetensors" --tokens 160
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASKS = [
    ("ja", "説明", "分散システムにおける結果整合性について説明してください。"),
    ("ja", "説明", "機械学習の過学習とは何か、対策とあわせて説明してください。"),
    ("ja", "手順", "Python の仮想環境を作って依存を入れる手順を書いてください。"),
    ("ja", "自由", "秋の京都を旅行する人向けに、二泊三日の案を書いてください。"),
    ("en", "説明", "Explain eventual consistency in distributed systems."),
    ("en", "説明", "Explain what overfitting is in machine learning and how to avoid it."),
    ("en", "手順", "Describe the steps to set up a Python virtual environment."),
    ("en", "自由", "Write a short travel plan for three days in Kyoto in autumn."),
    ("code", "コード", "Write a Python function that flattens a nested list."),
    ("code", "コード", "Python でリストの重複を順序を保って除去する関数を書いてください。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=160)
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
    q = {"group_size": 64, "bits": args.mtp_bits} if args.mtp_bits else None
    mtp = mtp_flash.load_flash_mtp(args.mtp, model.args.text, quantize=q)
    eng = spec_flash.FlashSpecEngine(model, mtp)

    by_lang, by_kind = {}, {}
    print(f"{'言語':5s} {'種別':6s} {'受理':>9s} {'率':>7s} {'±95%':>7s}  プロンプト")
    for lang, kind, text in TASKS:
        ids = mx.array(tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True))[None]
        _, acc, rounds = eng.generate(ids, args.tokens)
        p = acc / max(rounds, 1)
        ci = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / max(rounds, 1))
        print(f"  {lang:4s} {kind:6s} {acc:4d}/{rounds:<4d} {p:7.3f} {ci:7.3f}  {text[:36]}")
        for d, k in ((by_lang, lang), (by_kind, kind)):
            a, r = d.get(k, (0, 0))
            d[k] = (a + acc, r + rounds)

    def summarize(name, d):
        print(f"\n=== {name} ===")
        for k, (a, r) in sorted(d.items()):
            p = a / max(r, 1)
            ci = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / max(r, 1))
            print(f"  {k:6s} {a:4d}/{r:<4d}  {p:.3f} ± {ci:.3f}")

    summarize("言語別", by_lang)
    summarize("課題別", by_kind)


if __name__ == "__main__":
    main()
