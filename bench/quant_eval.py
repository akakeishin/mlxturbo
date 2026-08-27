"""Phase Q: 量子化変種の品質・速度スイート (モデル非依存)。

多数の変種を繰り返し測る前提の設計:
  参照 logits を一度だけダンプして保存し、各変種はダンプとの比較だけを行う。
  kld_probe.py (27B 用) は参照モデルを毎回ロードするが、Flash-Next の参照
  (bf16 360GB) はメモリに載らないため、この分離が必須になる。

レーンは 2 本:
  相対レーン (今すぐ使える): 参照 = 走る中で最大の変種 (例 v-max-112)。
    変種間の優劣 (等バイト A/B など) はこれで決着する
  絶対レーン (後日): 参照 = bf16/FP8 をシャードストリーミングで 1 パスだけ
    流す teacher ダンプ。実装は初回起動後 (subcommand `teacher` は未実装)

KLD は参照 top-K (既定 256) + 裾の質量補正による近似。裾質量は通常 <0.1% で、
変種間比較には十分。モデルカードには近似である旨を書く。

使い方 (すべて GPU 実行):
  # 1. 正準継続を作る (参照モデルで 1 回)
  uv run python bench/quant_eval.py continuations --model <ref> \
      --out bench/results/qe-cont.json
  # 2. 参照 logits をダンプ (1 回)
  uv run python bench/quant_eval.py dump --model <ref> \
      --continuations bench/results/qe-cont.json --out bench/results/qe-ref.npz
  # 3. 各変種を比較 + 速度 (変種ごと)
  uv run python bench/quant_eval.py compare --model <variant> \
      --continuations bench/results/qe-cont.json \
      --ref-dump bench/results/qe-ref.npz --tag v0-95
  uv run python bench/quant_eval.py speed --model <variant> --tag v0-95
  # 4. 一覧表
  uv run python bench/quant_eval.py report
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results" / "quant-eval"

# 感度が出やすい軸を広めに: 事実想起 (n-gram テーブル直撃)、コード、数学、
# 反復構造、多言語、構造化出力、長め文脈の要約
CALIB_PROMPTS: dict[str, str] = {
    "ja-explain": "分散システムにおける結果整合性と強整合性の違いを、具体例を挙げながら詳しく説明してください。",
    "ja-fact": "鎌倉幕府の成立から滅亡までの主要な出来事を、年号付きで時系列に列挙してください。",
    "en-prose": "Explain why the sky is blue during the day but red at sunset, in a way a curious teenager would enjoy.",
    "en-fact": "List the chemical elements discovered in the 20th century, with the year and discoverer for each.",
    "code-py": "Pythonで、ディレクトリ以下の全ファイルをSHA-256でハッシュ化して重複ファイルを検出するスクリプトを書いてください。",
    "code-rust": "Write a Rust function that parses an ISO-8601 timestamp without external crates, returning a struct with year, month, day, hour, minute, second.",
    "math": "3桁の整数のうち、各桁の数字の和が10になるものは何個あるか。途中の考え方も含めて答えてください。",
    "translate": "次の文を自然な英語に翻訳してください:「量子化は精度と引き換えにメモリと帯域を節約する技術であり、その配分には測定に基づく判断が必要である。」",
    "zh": "请用中文解释一下什么是投机解码（speculative decoding），以及它为什么能加速大语言模型的推理。",
    "json-struct": "架空の書店の在庫管理APIのレスポンス例をJSONで作成してください。書籍5冊分、各書籍にはISBN、タイトル、著者、価格、在庫数を含めてください。",
    "repeat-edit": "次の関数に型ヒントとdocstringを追加してください。他は変えないでください。\n```python\ndef merge(a, b):\n    out = dict(a)\n    for k, v in b.items():\n        if k in out and isinstance(out[k], dict) and isinstance(v, dict):\n            out[k] = merge(out[k], v)\n        else:\n            out[k] = v\n    return out\n```",
    "summarize": "次の主張を3行で要約してください: 大規模言語モデルの推論速度はメモリ帯域に律速されることが多い。重みを低ビットに量子化すると読み出し量が減って速度が上がるが、精度が犠牲になる。投機デコードは複数トークンを一括検証することで、帯域あたりの生成トークン数を増やす。この2つは独立に効くため併用できるが、量子化はドラフトの受理率を下げる方向にも働くため、併用時の利得は単純な掛け算にはならない。",
}


def _load(model_ref: str):
    from mlx_lm import load

    return load(model_ref)


def _prompt_ids(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True
        )


def _teacher_logits(model, full_ids):
    import mlx.core as mx

    cache = model.make_cache()
    logits = model(mx.array(full_ids)[None], cache=cache)
    mx.eval(logits)
    return logits[0]


def cmd_continuations(args):
    import mlx.core as mx

    model, tokenizer = _load(args.model)
    eos = {tokenizer.eos_token_id}
    out = {"model": args.model, "gen_tokens": args.gen_tokens, "prompts": {}}
    for key, text in CALIB_PROMPTS.items():
        ids = _prompt_ids(tokenizer, text)
        cache = model.make_cache()
        logits = model(mx.array(ids)[None], cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
        cont = []
        while len(cont) < args.gen_tokens and cur not in eos:
            cont.append(cur)
            logits = model(mx.array([[cur]]), cache=cache)
            cur = int(mx.argmax(logits[0, -1], axis=-1))
        out["prompts"][key] = {"prompt_ids": ids, "continuation_ids": cont}
        print(f"[cont] {key}: {len(cont)} tokens")
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}")


def cmd_dump(args):
    import mlx.core as mx

    model, _ = _load(args.model)
    cont = json.loads(Path(args.continuations).read_text())
    arrays = {}
    meta = {"model": args.model, "topk": args.topk, "prompts": {}}
    for key, entry in cont["prompts"].items():
        full = entry["prompt_ids"] + entry["continuation_ids"]
        start = len(entry["prompt_ids"]) - 1
        logits = _teacher_logits(model, full)[start:-1].astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        idx = mx.argpartition(-logp, args.topk - 1, axis=-1)[:, : args.topk]
        top = mx.take_along_axis(logp, idx, axis=-1)
        mx.eval(idx, top)
        idx_np = np.array(idx, dtype=np.int32)
        top_np = np.array(top, dtype=np.float32)
        tail = np.log1p(-np.minimum(np.exp(top_np).sum(axis=-1), 1 - 1e-9))
        arrays[f"{key}.idx"] = idx_np
        arrays[f"{key}.logp"] = top_np
        arrays[f"{key}.tail"] = tail.astype(np.float32)
        meta["prompts"][key] = {"positions": int(idx_np.shape[0])}
        print(f"[dump] {key}: {idx_np.shape[0]} positions")
        del logits, logp, idx, top
        mx.clear_cache()
    np.savez_compressed(args.out, **arrays)
    Path(str(args.out) + ".meta.json").write_text(json.dumps(meta))
    print(f"wrote {args.out}")


def cmd_compare(args):
    import mlx.core as mx

    model, _ = _load(args.model)
    cont = json.loads(Path(args.continuations).read_text())
    ref = np.load(args.ref_dump)
    per_prompt = {}
    for key, entry in cont["prompts"].items():
        full = entry["prompt_ids"] + entry["continuation_ids"]
        start = len(entry["prompt_ids"]) - 1
        logits = _teacher_logits(model, full)[start:-1].astype(mx.float32)
        logq_all = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        idx = mx.array(ref[f"{key}.idx"])
        logq = mx.take_along_axis(logq_all, idx, axis=-1)
        argmax_q = mx.argmax(logits, axis=-1)
        mx.eval(logq, argmax_q)
        logq_np = np.array(logq, dtype=np.float64)
        logp_np = ref[f"{key}.logp"].astype(np.float64)
        p = np.exp(logp_np)
        # top-K 近似 KLD: 裾は「参照裾質量が変種でも同じ集合に落ちる」仮定の
        # 下限側近似。裾質量 <0.1% なら変種間比較への影響は無視できる
        kld_pos = (p * (logp_np - logq_np)).sum(axis=-1)
        ref_top1 = ref[f"{key}.idx"][:, 0]
        agree = (np.array(argmax_q) == ref_top1).mean()
        # 変種自身の継続 PPL (正準トークン列に対する)
        tgt = np.array(full[start + 1 :], dtype=np.int64)
        tgt_logq = np.array(
            mx.take_along_axis(logq_all, mx.array(tgt)[:, None], axis=-1)[:, 0],
            dtype=np.float64,
        )
        per_prompt[key] = {
            "kld_mean": float(kld_pos.mean()),
            "kld_p95": float(np.quantile(kld_pos, 0.95)),
            "top1_agree": float(agree),
            "ppl": float(np.exp(-tgt_logq.mean())),
            "positions": int(kld_pos.shape[0]),
        }
        print(f"[cmp] {key}: kld={per_prompt[key]['kld_mean']:.5f} agree={agree:.3f}")
        del logits, logq_all
        mx.clear_cache()
    klds = [v["kld_mean"] for v in per_prompt.values()]
    agrees = [v["top1_agree"] for v in per_prompt.values()]
    result = {
        "kind": "compare",
        "tag": args.tag,
        "model": args.model,
        "ref_dump": str(args.ref_dump),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kld_mean": float(np.mean(klds)),
        "kld_worst_prompt": max(per_prompt, key=lambda k: per_prompt[k]["kld_mean"]),
        "top1_agree_mean": float(np.mean(agrees)),
        "per_prompt": per_prompt,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"compare-{args.tag}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}: kld_mean={result['kld_mean']:.5f} agree={result['top1_agree_mean']:.3f}")


def cmd_speed(args):
    import mlx.core as mx

    t0 = time.perf_counter()
    model, tokenizer = _load(args.model)
    load_s = time.perf_counter() - t0
    rows = {}
    for key in args.prompts.split(","):
        text = CALIB_PROMPTS[key]
        ids = _prompt_ids(tokenizer, text)
        cache = model.make_cache()
        t0 = time.perf_counter()
        logits = model(mx.array(ids)[None], cache=cache)
        cur = mx.argmax(logits[0, -1], axis=-1).reshape(1)
        mx.eval(cur)
        prefill_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        n = 0
        while n < args.gen_tokens:
            logits = model(cur[None], cache=cache)
            cur = mx.argmax(logits[0, -1], axis=-1).reshape(1)
            mx.eval(cur)
            n += 1
        decode_s = time.perf_counter() - t0
        rows[key] = {
            "prefill_tps": len(ids) / prefill_s,
            "decode_tps": n / decode_s,
        }
        print(f"[speed] {key}: {rows[key]['decode_tps']:.1f} tok/s")
    result = {
        "kind": "speed",
        "tag": args.tag,
        "model": args.model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "load_s": load_s,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "decode_tps_mean": float(np.mean([r["decode_tps"] for r in rows.values()])),
        "per_prompt": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"speed-{args.tag}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}: decode_mean={result['decode_tps_mean']:.1f} tok/s peak={result['peak_memory_gb']:.1f}GB")


def cmd_report(args):
    del args
    rows = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        tag = d["tag"]
        rows.setdefault(tag, {})
        if d["kind"] == "compare":
            rows[tag].update(
                kld=d["kld_mean"],
                agree=d["top1_agree_mean"],
                worst=d["kld_worst_prompt"],
            )
        else:
            rows[tag].update(
                tps=d["decode_tps_mean"], mem=d["peak_memory_gb"], load=d["load_s"]
            )
    if not rows:
        print("結果なし")
        return
    print(f"{'tag':12s} {'KLD':>9s} {'top1一致':>8s} {'decode':>8s} {'peak GB':>8s}  worst prompt")
    for tag, r in sorted(rows.items()):
        print(
            f"{tag:12s} {r.get('kld', float('nan')):9.5f} {r.get('agree', float('nan')):8.3f} "
            f"{r.get('tps', float('nan')):8.1f} {r.get('mem', float('nan')):8.1f}  {r.get('worst', '-')}"
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("continuations")
    p.add_argument("--model", required=True)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_continuations)

    p = sub.add_parser("dump")
    p.add_argument("--model", required=True)
    p.add_argument("--continuations", required=True)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_dump)

    p = sub.add_parser("compare")
    p.add_argument("--model", required=True)
    p.add_argument("--continuations", required=True)
    p.add_argument("--ref-dump", required=True)
    p.add_argument("--tag", required=True)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("speed")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--prompts", default="ja-explain,code-py,repeat-edit")
    p.set_defaults(fn=cmd_speed)

    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
