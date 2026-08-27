"""n-gram をディスクに置いたモデルを読み込んで動かす。

`FASTMLX_NGRAM_DISK=1` で焼いたチェックポイントは n-gram 表を持たないので、
読み込んだあと `fastmlx.ngram_stream.install` でサイドカーを結びつける必要が
ある。mlx_lm.generate はその手順を知らないので、ここを入口にする。

使い方:
  FASTMLX_NGRAM_DISK=1 uv run python tools/load_stream.py \
      --model ~/models/qwen38fn-mlx-v-stream \
      --ngram ~/models/qwen38fn-ngram-4bit \
      --prompt "日本の首都はどこですか。一文で答えてください。"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_streamed(model_path: str, ngram_dir: str | None, mode: str = "disk"):
    """n-gram をサイドカー参照にしたモデルとトークナイザを返す。"""

    # vendored arch は import 時にこの旗を読む。mlx_lm を触る前に立てる
    if ngram_dir:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"
    from mlx_lm import load

    model, tok = load(model_path)
    if ngram_dir:
        from fastmlx.ngram_stream import install, install_ram

        (install_ram if mode == "ram" else install)(model, ngram_dir)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram-mode", default="disk", choices=("disk","ram"),
                    help="disk=memmap から引く / ram=連結テーブルを常駐")
    ap.add_argument("--ngram", default=None, help="省略すると素の読み込み (比較用)")
    ap.add_argument("--prompt", default="日本の首都はどこですか。一文で答えてください。")
    ap.add_argument("--max-tokens", type=int, default=60)
    args = ap.parse_args()

    import mlx.core as mx

    t0 = time.time()
    model, tok = load_streamed(args.model, args.ngram, args.ngram_mode)
    print(f"読み込み {time.time() - t0:.0f}s  peak={mx.get_peak_memory() / 1e9:.1f}GB")

    ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True
    )
    cache = model.make_cache()
    t0 = time.time()
    logits = model(mx.array(ids)[None], cache=cache)
    cur = int(mx.argmax(logits[0, -1], axis=-1))
    prefill = time.time() - t0
    out, eos = [], {tok.eos_token_id}
    t0 = time.time()
    while len(out) < args.max_tokens and cur not in eos:
        out.append(cur)
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
    dt = time.time() - t0

    print("=" * 60)
    print(tok.decode(out))
    print("=" * 60)
    print(f"prompt {len(ids)} tok / {len(ids) / prefill:.1f} tok/s")
    print(f"生成 {len(out)} tok / {len(out) / dt:.2f} tok/s")
    print(f"peak={mx.get_peak_memory() / 1e9:.2f}GB")


if __name__ == "__main__":
    main()
