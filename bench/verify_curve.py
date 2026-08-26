"""L0: 検証幅 m のコストカーブ。

m 個のトークンを 1 回の forward で処理する時間が m=1 の何倍かを測る。
decode が帯域律速なら、重み読みは m で償却されるので曲線は平坦に近いはず。
平坦な区間の広さが投機デコードの利得上限を決める。
"""

import argparse
import json
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def time_forward(model, tokens: mx.array, cache, reps: int) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        logits = model(tokens[None], cache=cache)
        mx.eval(logits)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--widths", default="1,2,4,8,16,24,32,48,64")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    widths = [int(w) for w in args.widths.split(",")]

    prompt = mx.array(tokenizer.encode("システム設計の要点を述べる。" * 200)[: args.prompt_tokens])
    cache = make_prompt_cache(model)
    mx.eval(model(prompt[None], cache=cache))

    tok = mx.array([1000])
    time_forward(model, tok, cache, 5)  # warmup

    results = []
    for m in widths:
        tokens = mx.array([1000 + i for i in range(m)])
        t = time_forward(model, tokens, cache, args.reps)
        results.append({"m": m, "step_s": t})

    base = results[0]["step_s"]
    for r in results:
        r["ratio_vs_m1"] = r["step_s"] / base
        r["tokens_per_s_if_all_accepted"] = r["m"] / r["step_s"]

    out = {"model": args.model, "prompt_tokens": args.prompt_tokens, "curve": results}
    print(json.dumps(out, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
