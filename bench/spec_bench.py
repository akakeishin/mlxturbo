"""MTP 自己投機 vs 素の mlx-lm の比較。

greedy 同士なので出力トークン列は完全一致するはず。一致確認と同時に
decode tok/s と受理率を測る。
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.qwen3_5 import TextModelArgs
from mlx_lm.sample_utils import make_sampler

from fastmlx.mtp import find_snapshot, load_mtp
from fastmlx.spec import SpecEngine

PROMPTS = {
    "code": "Pythonで、ディレクトリ以下の全ファイルをSHA-256でハッシュ化して"
    "重複ファイルを検出するスクリプトを書いてください。",
    "prose": "分散システムにおける結果整合性と強整合性の違いを、具体例を"
    "挙げながら詳しく説明してください。",
}


def stock_generate(model, tokenizer, prompt_ids, max_tokens):
    sampler = make_sampler(temp=0.0)
    tokens = []
    last = None
    t0 = time.perf_counter()
    for resp in stream_generate(
        model, tokenizer, prompt_ids, max_tokens=max_tokens, sampler=sampler
    ):
        tokens.append(resp.token)
        last = resp
    wall = time.perf_counter() - t0
    return {
        "tokens": tokens,
        "decode_tps": last.generation_tps,
        "wall_s": wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lmstudio-community/Qwen3.8-27B-MLX-4bit")
    ap.add_argument("--original", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n-draft", default="3", help="comma separated sweep, e.g. 3,5,7")
    ap.add_argument("--max-draft", type=int, default=0, help="adaptive depth cap")
    ap.add_argument("--mtp-bits", type=int, default=0, help="quantize MTP (0=bf16)")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    text_args = TextModelArgs.from_dict(model.args.text_config)
    quant = {"bits": args.mtp_bits, "group_size": 64} if args.mtp_bits else None
    mtp = load_mtp(find_snapshot(args.original), text_args, quantize=quant)
    mx.eval(mtp.parameters())
    engine = SpecEngine(model, mtp)

    eos_ids = {tokenizer.eos_token_id}
    results = {}
    for name, prompt in PROMPTS.items():
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )

        stock = stock_generate(model, tokenizer, prompt_ids, args.max_tokens)
        results[name] = {"stock_decode_tps": stock["decode_tps"], "sweep": {}}
        for nd in [int(x) for x in str(args.n_draft).split(",")]:
            spec = engine.generate(
                prompt_ids,
                max_tokens=args.max_tokens,
                n_draft=nd,
                max_draft=args.max_draft,
                eos_ids=eos_ids,
            )
            n = min(len(stock["tokens"]), len(spec["tokens"]))
            match = next(
                (i for i in range(n) if stock["tokens"][i] != spec["tokens"][i]), n
            )
            results[name]["sweep"][nd] = {
                "spec_decode_tps": spec["decode_tps"],
                "speedup": spec["decode_tps"] / stock["decode_tps"],
                "identical": match == n,
                "compared": n,
                "mean_accepted": spec["mean_accepted"],
                "tokens_per_step": spec["tokens_per_step"],
                "accept_hist": spec["accept_hist"],
            }
        print(name, json.dumps(results[name], indent=2))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
