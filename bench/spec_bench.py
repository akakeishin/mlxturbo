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

_EDIT_SNIPPET = '''
import json
import sqlite3
from pathlib import Path


def load_records(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, name, score, created_at FROM records ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"count": 0, "mean": None, "best": None}
    scores = [r["score"] for r in records]
    best = max(records, key=lambda r: r["score"])
    return {
        "count": len(records),
        "mean": sum(scores) / len(scores),
        "best": {"id": best["id"], "name": best["name"], "score": best["score"]},
    }


def export_summary(db_path: str, out_path: str) -> None:
    records = load_records(db_path)
    summary = summarize(records)
    Path(out_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    export_summary(sys.argv[1], sys.argv[2])
'''

PROMPTS = {
    "code": "Pythonで、ディレクトリ以下の全ファイルをSHA-256でハッシュ化して"
    "重複ファイルを検出するスクリプトを書いてください。",
    "prose": "分散システムにおける結果整合性と強整合性の違いを、具体例を"
    "挙げながら詳しく説明してください。",
    "edit": "次のPythonコードの各関数にエラーハンドリングを追加して、"
    "全体を書き直してください。他は変えないでください。\n```python"
    + _EDIT_SNIPPET
    + "```",
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
    ap.add_argument("--prompts", default="", help="comma separated subset of prompt names")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    text_args = TextModelArgs.from_dict(model.args.text_config)
    quant = {"bits": args.mtp_bits, "group_size": 64} if args.mtp_bits else None
    mtp = load_mtp(find_snapshot(args.original), text_args, quantize=quant)
    mx.eval(mtp.parameters())
    engine = SpecEngine(model, mtp)

    eos_ids = {tokenizer.eos_token_id}
    selected = (
        {k: PROMPTS[k] for k in args.prompts.split(",")} if args.prompts else PROMPTS
    )
    results = {}
    for name, prompt in selected.items():
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
                "src_hist": spec["src_hist"],
            }
        print(name, json.dumps(results[name], indent=2))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
