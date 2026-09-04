"""qwen3_5族のMTP設定と真のARを同一prefill cacheから回文比較する。

長文脈ではverify幅ぶん同じKVを読む費用が受理利得を上回りうる。環境変数の
knobでは表せない生成設定を同一processの回文順で測る。`cap3`はSDPAの
query-row境界を越える幅5以上を避ける候補（draft最大3、lookupなし）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import decode_ab_generic as G


SETTINGS = {
    "mtp": {"n_draft": 3, "max_draft": 8, "lookup_len": 16},
    "cap3": {"n_draft": 3, "max_draft": 3, "lookup_len": 0},
    "ar": {"n_draft": 0, "max_draft": 0, "lookup_len": 0},
}


def build_parser():
    ap = argparse.ArgumentParser(
        description="qwen3_5族のMTP設定と真ARを同一cacheで回文比較する"
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--ctxs", default="17000,25000,32000,40000,50000")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--prompts", type=int, default=1)
    ap.add_argument("--variants", default="mtp,ar")
    ap.add_argument(
        "--warmup-tokens", type=int, default=32,
        help="各context・promptで両variantを捨て走行する長さ",
    )
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ignore-eos", action="store_true",
        help="速度比較の生成長を揃えるためEOSで停止しない",
    )
    ap.add_argument("--out", required=True)
    ap.set_defaults(no_mtp=False)
    return ap


def _run(eng, ids, session, snapshot, n_tokens, eos_ids, variant, temp=0.0,
         seed=None):
    if seed is not None:
        import mlx.core as mx

        mx.random.seed(seed)
    cfg = SETTINGS[variant]
    row = G.run_resumed(
        eng, ids, session, snapshot, n_tokens, eos_ids,
        cfg["n_draft"], cfg["max_draft"], lookup_len=cfg["lookup_len"],
        temp=temp,
    )
    row["variant"] = variant
    return row


def _first_divergence(a, b):
    pos = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), None)
    if pos is None and len(a) != len(b):
        return min(len(a), len(b))
    return pos


def _summarize(
    rows, requested_ctx, actual_ctx, prefill_s, case_idx=0,
    variants=("mtp", "ar"),
):
    means = {
        variant: statistics.mean(
            row["ms_per_tok"] for row in rows if row["variant"] == variant
        )
        for variant in variants
    }
    mtp_rows = [row for row in rows if row["variant"] == "mtp"]
    ar_rows = [row for row in rows if row["variant"] == "ar"]
    out = {
        "requested_ctx": requested_ctx,
        "case_idx": case_idx,
        "ctx": actual_ctx,
        "prefill_s": prefill_s,
        "variants": {
            variant: {
                "mean_ms_per_tok": means[variant],
                "mean_tok_per_round": statistics.mean(
                    row["tok_per_round"] for row in rows
                    if row["variant"] == variant
                ),
            }
            for variant in variants
        },
    }
    if "ar" in means:
        for variant in variants:
            out["variants"][variant]["vs_ar_pct"] = (
                means[variant] / means["ar"] - 1
            ) * 100
    if mtp_rows and ar_rows:
        out.update(
            mtp_mean_ms_per_tok=means["mtp"],
            ar_mean_ms_per_tok=means["ar"],
            mtp_vs_ar_pct=(means["mtp"] / means["ar"] - 1) * 100,
            mtp_mean_tok_per_round=out["variants"]["mtp"]["mean_tok_per_round"],
            ar_mean_tok_per_round=out["variants"]["ar"]["mean_tok_per_round"],
            first_divergence=_first_divergence(
                mtp_rows[0]["tokens"], ar_rows[0]["tokens"]
            ),
        )
    return out


def main() -> int:
    args = build_parser().parse_args()
    ctxs = [int(x) for x in args.ctxs.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    if not ctxs or any(x < 0 for x in ctxs):
        raise SystemExit("--ctxs は0以上の整数を1つ以上指定する")
    if len(variants) < 2 or len(set(variants)) != len(variants):
        raise SystemExit("--variants は重複しない2種以上にする")
    unknown = [variant for variant in variants if variant not in SETTINGS]
    if unknown:
        raise SystemExit(f"未知のvariant: {','.join(unknown)}")
    if args.prompts <= 0 or args.prompts > 3:
        raise SystemExit("--prompts は1〜3にする")
    if args.warmup_tokens <= 0:
        raise SystemExit("--warmup-tokens は正の整数にする")
    if args.temp < 0:
        raise SystemExit("--temp は0以上にする")

    import mlx.core as mx

    model, tok, eng, eos_ids, _guard = G.load_model(args)
    if args.ignore_eos:
        eos_ids = ()
    mx.random.seed(args.seed)
    short_ids = G.build_cases(tok, 0)[0][1]
    for variant in variants:
        cfg = SETTINGS[variant]
        G.run_once(
            eng, short_ids, 32, eos_ids, cfg["n_draft"], cfg["max_draft"],
            lookup_len=cfg["lookup_len"], temp=args.temp,
        )

    rows = []
    summaries = []
    for requested_ctx in ctxs:
        cases = G.build_cases(tok, requested_ctx, long_count=args.prompts)
        if requested_ctx == 0:
            cases = cases[:args.prompts]
        for case_idx, (_, ids) in enumerate(cases):
            session, snapshot, prefill_s = G.prefill_once(eng, ids, 3, 8)
            print(
                f"\nctx={len(ids)} case={case_idx} prefill={prefill_s:.2f}s",
                flush=True,
            )
            for variant in variants:
                _run(
                    eng, ids, session, snapshot, args.warmup_tokens,
                    eos_ids, variant, temp=args.temp,
                    seed=args.seed + case_idx,
                )

            forward = variants if case_idx % 2 == 0 else list(reversed(variants))
            palindrome = forward + list(reversed(forward))
            for variant in (palindrome * args.reps):
                row = _run(
                    eng, ids, session, snapshot, args.tokens, eos_ids, variant,
                    temp=args.temp, seed=args.seed + case_idx,
                )
                row.update(
                    ctx=len(ids), requested_ctx=requested_ctx,
                    case_idx=case_idx,
                )
                rows.append(row)
                print(
                    f"  {variant:>3s}: {row['ms_per_tok']:.2f} ms/tok  "
                    f"{row['ms_per_round']:.2f} ms/round  "
                    f"tok/round {row['tok_per_round']:.3f}",
                    flush=True,
                )

            sub = [
                row for row in rows
                if row["requested_ctx"] == requested_ctx
                and row["case_idx"] == case_idx
            ]
            summary = _summarize(
                sub, requested_ctx, len(ids), prefill_s, case_idx=case_idx,
                variants=variants,
            )
            summaries.append(summary)
            cells = []
            for variant in variants:
                stats = summary["variants"][variant]
                delta = (f" / AR {stats['vs_ar_pct']:+.1f}%"
                         if "vs_ar_pct" in stats else "")
                cells.append(
                    f"{variant} {stats['mean_ms_per_tok']:.3f} ms/tok{delta}"
                )
            print("  mean " + " | ".join(cells), flush=True)

    output = {
        "meta": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "model": args.model,
            "mtp": args.mtp,
            "mtp_bits": args.mtp_bits,
            "tokens": args.tokens,
            "reps": args.reps,
            "prompts": args.prompts,
            "warmup_tokens": args.warmup_tokens,
            "variants": variants,
            "temp": args.temp,
            "ignore_eos": args.ignore_eos,
            "settings": SETTINGS,
            "source_sha256": {
                str(path.relative_to(REPO_ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in (
                    Path(__file__).resolve(),
                    Path(G.__file__).resolve(),
                    REPO_ROOT / "mlxturbo/spec.py",
                )
            },
        },
        "summaries": summaries,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
