"""焼いた段を同じ課題で生成させ、横並びで比べる。

KLD は teacher-forced の forward で測るので**生成の質は見ていない**。
STATUS の「RMSNorm の +1 欠落」は、活性の大きさはそれらしいまま情報だけ
壊れて生成が無意味な反復になった。数字だけで出荷しないための確認。

温度 0 (貪欲) で回す。段の違いだけを見たいので、サンプリングの揺れを入れない。
出力は bench/results/gen-compare-<tag>.md に落として横並びで読む。

    tools/biglock.sh uv run python tools/gen_compare.py \\
        --model ~/models/qwen38fn-mlx-v-l --ngram ~/models/qwen38fn-ngram-4bit --tag v-l
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASKS = [
    ("短答", "オーストラリアの首都は? 単語ひとつで答えてください。"),
    ("事実+説明", "分散システムの結果整合性を、3 文で説明してください。"),
    ("バグ指摘", "次の関数の問題点を指摘してください:\n\n"
                 "def mean(xs):\n    return sum(xs) / len(xs)"),
    ("コード生成", "Python で、リストを受け取り重複を保ちつつ順序を維持して"
                   "ユニーク化する関数を書いてください。"),
    ("論理", "ある箱に赤玉 3 個と白玉 2 個が入っています。2 個続けて取り出す"
             "とき、2 個とも赤である確率を求め、式も示してください。"),
    ("要約", "次を 1 文で要約してください: 量子化はモデルの重みを低いビット幅で"
             "表現する技術で、メモリ使用量と帯域を減らせる一方、表現できる値が"
             "粗くなるため出力分布が元のモデルからずれる。どのテンソルに何ビットを"
             "配るかで、同じ容量でも品質が大きく変わる。"),
    ("翻訳", "次を自然な日本語に訳してください: "
             "The model is quantized per tensor class, so the same total size "
             "can yield very different quality."),
    ("指示追従", "3 つの箇条書きで、MLX と PyTorch の違いを述べてください。"
                 "各項目は 20 字以内にしてください。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    import os

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler

    from mlxturbo import fused

    model, tok = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)
    fused.enable_hyper_connection_kernel()
    sampler = make_sampler(temp=0.0)

    rows = []
    for name, p in TASKS:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True
        )
        t0 = time.perf_counter()
        out = generate(model, tok, prompt=tok.decode(ids),
                       max_tokens=args.max_tokens, sampler=sampler, verbose=False)
        dt = time.perf_counter() - t0
        rows.append({"task": name, "prompt": p, "out": out.strip(), "sec": dt})
        print(f"\n### {name} ({dt:.1f}s)\n{out.strip()[:700]}", flush=True)

    out_dir = REPO_ROOT / "bench/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"gen-compare-{args.tag}.json").write_text(
        json.dumps({"model": args.model, "tag": args.tag, "rows": rows},
                   ensure_ascii=False, indent=1))
    print(f"\npeak={mx.get_peak_memory() / 1e9:.1f}GB  "
          f"wrote bench/results/gen-compare-{args.tag}.json")


if __name__ == "__main__":
    main()
