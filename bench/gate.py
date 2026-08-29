"""One-shot greedy identity and speculative mismatch gate.

The correctness reference is a raw-model manual loop using ``make_cache``,
``model(...)``, and greedy ``argmax``.  It intentionally does not use mlx-lm's
generation pipeline because that pipeline's execution environment has already
been isolated as a separate divergence source.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


PROMPTS = {
    "code": "Pythonで、ディレクトリ以下の全ファイルをSHA-256でハッシュ化して"
    "重複ファイルを検出するスクリプトを書いてください。",
    "prose": "分散システムにおける結果整合性と強整合性の違いを、具体例を"
    "挙げながら詳しく説明してください。",
    "edit": "次の関数に型ヒントとエラーハンドリングを追加してください。\n"
    "```python\ndef load_json(path):\n    import json\n"
    "    return json.loads(open(path).read())\n```",
}


def mismatch_positions(reference, actual):
    common = min(len(reference), len(actual))
    positions = [i for i in range(common) if reference[i] != actual[i]]
    positions.extend(range(common, max(len(reference), len(actual))))
    return positions


def mismatch_records(reference, actual, positions, tokenizer, radius=8):
    records = []
    for index in positions[:5]:
        start = max(0, index - radius)
        ref_context = reference[start : index + radius + 1]
        actual_context = actual[start : index + radius + 1]
        records.append(
            {
                "index": index,
                "reference_token": reference[index] if index < len(reference) else None,
                "actual_token": actual[index] if index < len(actual) else None,
                "reference_context_tokens": ref_context,
                "actual_context_tokens": actual_context,
                "reference_context": tokenizer.decode(ref_context),
                "actual_context": tokenizer.decode(actual_context),
            }
        )
    return records


def manual_greedy(model, prompt_ids, max_tokens, eos_ids, mx):
    """Generate with the audited raw-model cache/model/argmax loop."""

    start = time.perf_counter()
    cache = model.make_cache()
    prompt = mx.array(prompt_ids)
    logits = model(prompt[None], cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1)
    mx.eval(token)
    tokens = []
    while len(tokens) < max_tokens:
        value = int(token.item())
        tokens.append(value)
        if value in eos_ids or len(tokens) == max_tokens:
            break
        logits = model(token[None], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1)
        mx.eval(token)
    wall = time.perf_counter() - start
    return {"tokens": tokens, "wall_s": wall}


def _comparison(reference, actual, tokenizer):
    positions = mismatch_positions(reference, actual)
    return {
        "identical": not positions,
        "compared": max(len(reference), len(actual)),
        "n_mismatch": len(positions),
        "first_mismatches": mismatch_records(
            reference, actual, positions, tokenizer
        ),
    }


def main():
    import mlx.core as mx

    from mlxturbo._mlx_compat import TextModelArgs, mlx_lm_load
    from mlxturbo.mtp import find_snapshot, load_mtp
    from mlxturbo.spec import SpecEngine

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="lmstudio-community/Qwen3.8-27B-MLX-4bit"
    )
    parser.add_argument("--original", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--n-draft", type=int, default=3)
    parser.add_argument("--max-draft", type=int, default=0)
    parser.add_argument("--lookup-len", type=int, default=16)
    parser.add_argument("--mtp-bits", type=int, default=0)
    parser.add_argument("--prompts", default="code,prose,edit")
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()
    if args.n_draft <= 0 and args.max_draft <= 0 and args.lookup_len <= 0:
        parser.error("speculative-on settings must enable at least one draft source")

    model, tokenizer = mlx_lm_load(args.model)
    selected = [name for name in args.prompts.split(",") if name]
    unknown = set(selected) - PROMPTS.keys()
    if unknown:
        parser.error(f"unknown prompts: {', '.join(sorted(unknown))}")
    prompt_ids = {
        name: tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPTS[name]}],
            add_generation_prompt=True,
        )
        for name in selected
    }
    eos_ids = {tokenizer.eos_token_id}

    # Run every reference before SpecEngine installs verification-only classes.
    references = {
        name: manual_greedy(
            model, prompt_ids[name], args.max_tokens, eos_ids, mx
        )
        for name in selected
    }

    text_args = TextModelArgs.from_dict(model.args.text_config)
    quant = {"bits": args.mtp_bits, "group_size": 64} if args.mtp_bits else None
    mtp = load_mtp(find_snapshot(args.original), text_args, quantize=quant)
    mx.eval(mtp.parameters())
    engine = SpecEngine(model, mtp)

    results = {}
    baseline_failed = False
    for name in selected:
        baseline = engine.generate(
            prompt_ids[name],
            max_tokens=args.max_tokens,
            n_draft=0,
            max_draft=0,
            lookup_len=0,
            temp=0.0,
            eos_ids=eos_ids,
        )
        speculative = engine.generate(
            prompt_ids[name],
            max_tokens=args.max_tokens,
            n_draft=args.n_draft,
            max_draft=args.max_draft,
            lookup_len=args.lookup_len,
            temp=0.0,
            eos_ids=eos_ids,
        )
        reference_tokens = references[name]["tokens"]
        baseline_check = _comparison(
            reference_tokens, baseline["tokens"], tokenizer
        )
        speculative_check = _comparison(
            reference_tokens, speculative["tokens"], tokenizer
        )
        if not speculative_check["identical"]:
            # 合格基準は baseline のみ。spec の不一致は bf16 完全同点が
            # バッチ形状の縮約順で割れる同点 flip でも起きる (bench/tie_flip_probe.py
            # で実証、docs/STATUS.md の同点 flip 節)。ここの False 単体を
            # バグの証拠として扱わないこと。
            speculative_check["note"] = (
                "not a gate criterion; may be bf16 tie flips from batched "
                "verification (see docs/STATUS.md)"
            )
        baseline_failed |= not baseline_check["identical"]
        results[name] = {
            "reference": {
                "kind": "raw_model_manual_loop",
                "wall_s": references[name]["wall_s"],
                "n_tokens": len(reference_tokens),
            },
            "baseline": {
                "settings": {"n_draft": 0, "max_draft": 0, "lookup_len": 0},
                **baseline_check,
                "decode_tps": baseline["decode_tps"],
            },
            "speculative": {
                "settings": {
                    "n_draft": args.n_draft,
                    "max_draft": args.max_draft,
                    "lookup_len": args.lookup_len,
                },
                **speculative_check,
                "decode_tps": speculative["decode_tps"],
                "accept_hist": speculative["accept_hist"],
                "phase_s": speculative["phase_s"],
            },
        }

    report = {
        "identity_reference": "raw_model_manual_loop",
        "baseline_all_identical": not baseline_failed,
        "results": results,
        "timing_note": "reference only; final measurement requires a quiet machine",
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as file:
            json.dump(report, file, indent=2)
    if baseline_failed:
        raise SystemExit("baseline identity gate failed")


if __name__ == "__main__":
    main()
