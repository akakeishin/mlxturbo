"""Flash-Next の通常 AR で S=1 full-GDN 融合を同一モデル ABBA する。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace


PROMPTS = (
    "分散システムにおける結果整合性について説明してください。",
    "Explain why speculative decoding helps when decoding is dispatch bound.",
    "Python でリストの重複を順序を保って除去する関数を書いてください。",
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--out")
    parser.add_argument(
        "--quality",
        action="store_true",
        help="速度の代わりに step=1 KLD を融合 off/on の順で測る",
    )
    parser.add_argument("--continuations", default="bench/results/qe-cont.json")
    parser.add_argument("--ref-dump", default="bench/results/qe-ref.npz")
    parser.add_argument("--quality-tag", default="gdn-decode-all")
    return parser


def _set_mode(enabled: bool) -> None:
    from mlxturbo import fused

    os.environ["MLXTURBO_GDN_DECODE_ALL"] = "1" if enabled else "0"
    fused.enable_gdn_decode_all()


def _run(model, tokenizer, prompt_ids, tokens: int) -> dict:
    from mlx_lm.generate import stream_generate

    generated = []
    last = None
    for response in stream_generate(
        model, tokenizer, prompt_ids, max_tokens=tokens
    ):
        generated.append(int(response.token))
        last = response
    if last is None:
        raise RuntimeError("AR generation returned no response")
    return {
        "tokens": generated,
        "n_tokens": len(generated),
        "decode_tps": float(last.generation_tps),
        "prompt_tps": float(last.prompt_tps),
        "peak_memory_gb": float(last.peak_memory),
    }


def run_with_model(argv, bundle) -> int:
    args = _parser().parse_args(argv)
    if args.quality:
        from bench.quant_eval import compare_with_model

        previous = os.environ.get("MLXTURBO_GDN_DECODE_ALL")
        try:
            for enabled, suffix in ((False, "off"), (True, "on")):
                _set_mode(enabled)
                compare_with_model(
                    bundle.model,
                    SimpleNamespace(
                        continuations=args.continuations,
                        ref_dump=args.ref_dump,
                        tag=f"{args.quality_tag}-{suffix}",
                        fusions=True,
                        model=bundle.model_path,
                        rebit=None,
                        step=1,
                    ),
                )
        finally:
            if previous is None:
                os.environ.pop("MLXTURBO_GDN_DECODE_ALL", None)
                _set_mode(False)
                os.environ.pop("MLXTURBO_GDN_DECODE_ALL", None)
            else:
                os.environ["MLXTURBO_GDN_DECODE_ALL"] = previous
                from mlxturbo import fused
                fused.enable_gdn_decode_all()
        return 0

    if not args.out:
        raise SystemExit("速度測定では --out が必要")
    if args.tokens < 2 or args.warmup_tokens < 1:
        raise SystemExit("--tokens は2以上、--warmup-tokens は1以上")

    tokenizer = bundle.tokenizer
    model = bundle.model
    prompt_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        for prompt in PROMPTS
    ]
    previous = os.environ.get("MLXTURBO_GDN_DECODE_ALL")
    rows = []
    try:
        for enabled in (False, True):
            _set_mode(enabled)
            _run(model, tokenizer, prompt_ids[0], args.warmup_tokens)

        for case_index, ids in enumerate(prompt_ids):
            order = (True, False, False, True)
            for enabled in order:
                _set_mode(enabled)
                row = _run(model, tokenizer, ids, args.tokens)
                row.update(case_index=case_index, variant="A" if enabled else "B")
                rows.append(row)
                print(
                    f"case={case_index} {'A' if enabled else 'B'} "
                    f"{row['decode_tps']:.3f} tok/s n={row['n_tokens']}",
                    flush=True,
                )
    finally:
        if previous is None:
            os.environ.pop("MLXTURBO_GDN_DECODE_ALL", None)
            _set_mode(False)
            os.environ.pop("MLXTURBO_GDN_DECODE_ALL", None)
        else:
            os.environ["MLXTURBO_GDN_DECODE_ALL"] = previous
            from mlxturbo import fused
            fused.enable_gdn_decode_all()

    mismatches = []
    for case_index in range(len(prompt_ids)):
        case_rows = [row for row in rows if row["case_index"] == case_index]
        baseline = case_rows[1]["tokens"]
        if any(row["tokens"] != baseline for row in case_rows):
            mismatches.append(case_index)
    means = {
        variant: statistics.mean(
            row["decode_tps"] for row in rows if row["variant"] == variant
        )
        for variant in ("A", "B")
    }
    summary = {
        "mean_decode_tps": means,
        "speedup_percent": 100.0 * (means["A"] / means["B"] - 1.0),
        "token_mismatch_cases": mismatches,
    }
    payload = {"summary": summary, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not mismatches else 1


def main() -> int:
    raise SystemExit(
        "この道具は tools/biglock.sh 経由の常駐 A/B worker で実行する"
    )


if __name__ == "__main__":
    main()
