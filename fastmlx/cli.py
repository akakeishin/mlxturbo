"""fastmlx 対話 CLI。

uv run fastmlx                       # チャット REPL
uv run fastmlx --prompt "..."        # ワンショット
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.qwen3_5 import TextModelArgs

from .mtp import find_snapshot, load_mtp
from .spec import SpecEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lmstudio-community/Qwen3.8-27B-MLX-4bit")
    ap.add_argument("--original", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--n-draft", type=int, default=3)
    ap.add_argument("--max-draft", type=int, default=8)
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    quant = {"bits": args.mtp_bits, "group_size": 64} if args.mtp_bits else None
    text_args = TextModelArgs.from_dict(model.args.text_config)
    mtp = load_mtp(find_snapshot(args.original), text_args, quantize=quant)
    mx.eval(mtp.parameters())
    engine = SpecEngine(model, mtp)
    print(f"[fastmlx] loaded in {time.perf_counter() - t0:.1f}s: {args.model}")

    eos_ids = {tokenizer.eos_token_id}

    def run_turn(messages):
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        detok = tokenizer.detokenizer
        detok.reset()

        def on_tokens(toks):
            for t in toks:
                if t in eos_ids:
                    continue
                detok.add_token(t)
            print(detok.last_segment, end="", flush=True)

        res = engine.generate(
            prompt_ids,
            max_tokens=args.max_tokens,
            n_draft=args.n_draft,
            max_draft=args.max_draft,
            temp=args.temp,
            eos_ids=eos_ids,
            on_tokens=on_tokens,
        )
        detok.finalize()
        print(detok.last_segment, flush=True)
        print(
            f"\n[{res['decode_tps']:.1f} tok/s | {res['tokens_per_step']:.2f} tok/step"
            f" | ttft {res['ttft_s']:.1f}s]"
        )
        return tokenizer.decode(
            [t for t in res["tokens"] if t not in eos_ids]
        )

    if args.prompt is not None:
        run_turn([{"role": "user", "content": args.prompt}])
        return

    messages = []
    print("[fastmlx] チャット開始。exit で終了。")
    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": user})
        reply = run_turn(messages)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
