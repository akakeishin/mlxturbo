"""L0: 素の mlx-lm decode をルーフラインと比べる。

達成帯域 ~= 重み実バイト x decode tok/s をチップ公称帯域と比べ、
カーネル層の伸び代を判定する。KV 読みは短文脈では重み比で誤差なので
初版では無視する(長文脈の測定は別スクリプトで行う)。
"""

import argparse
import json
import platform
import subprocess
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate


def param_bytes(model) -> int:
    return sum(v.nbytes for _, v in tree_flatten(model.parameters()))


def chip_name() -> str:
    out = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
    )
    return out.stdout.strip()


def build_prompt(tokenizer, n_tokens: int) -> str:
    base = (
        "以下は分散システムの設計判断についての長い技術文書である。"
        "一貫性と可用性のトレードオフ、リーダー選出、ログ複製、"
        "スナップショットの取り方について詳細に述べる。"
    )
    text = base
    while len(tokenizer.encode(text)) < n_tokens:
        text += base
    ids = tokenizer.encode(text)[:n_tokens]
    return tokenizer.decode(ids)


def run_once(model, tokenizer, prompt: str, gen_tokens: int) -> dict:
    ttft = None
    t0 = time.perf_counter()
    last = None
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=gen_tokens):
        if ttft is None:
            ttft = time.perf_counter() - t0
        last = resp
    return {
        "ttft_s": ttft,
        "prompt_tps": last.prompt_tps,
        "decode_tps": last.generation_tps,
        "gen_tokens": last.generation_tokens,
        "peak_mem_gb": last.peak_memory,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--bandwidth-gbs", type=float, default=400.0, help="chip peak GB/s")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    wbytes = param_bytes(model)
    prompt = build_prompt(tokenizer, args.prompt_tokens)

    run_once(model, tokenizer, prompt, 8)  # warmup

    runs = [run_once(model, tokenizer, prompt, args.gen_tokens) for _ in range(args.runs)]
    best = max(runs, key=lambda r: r["decode_tps"])

    achieved_gbs = best["decode_tps"] * wbytes / 1e9
    result = {
        "model": args.model,
        "chip": chip_name(),
        "mlx_version": mx.__version__,
        "weight_gb": wbytes / 1e9,
        "prompt_tokens": args.prompt_tokens,
        "runs": runs,
        "best_decode_tps": best["decode_tps"],
        "best_prompt_tps": best["prompt_tps"],
        "achieved_gbs": achieved_gbs,
        "peak_bandwidth_gbs": args.bandwidth_gbs,
        "roofline_ratio": achieved_gbs / args.bandwidth_gbs,
        "theoretical_ceiling_tps": args.bandwidth_gbs * 1e9 / wbytes,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
