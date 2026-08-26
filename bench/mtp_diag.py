"""D0 診断: MTP 連鎖の位置別受理率と hidden RMS ドリフトの実測。

spec.py の生成ループを固定深度 K・greedy・lookup 無効で再現し、
1 ステップごとに次を記録する。

- 位置別一致: 検証 forward の予測 preds[k-1] と draft window[k] の一致
  (prefix の受理可否によらない位置単位の一致 = match)、および prefix 込みの
  連鎖受理 (chain)。conditional は chain[k]/chain[k-1]。
- RMS ドリフト: 連鎖深度 k の MTP 出力 ĥ_k の RMS と、同位置の真の
  backbone hidden hs[:, k-1] の RMS の比。膨張していれば Attention Drift。

`--rescale` で深度 2 以降の連鎖入力 dh をスカラー倍してから流せる
(推論時再スケールの訓練ゼロ試行)。値は深度 2..K 用のカンマ区切り。

使い方 (GPU 作業。受理率と RMS は correctness 系なので静音窓は不要):
  uv run python bench/mtp_diag.py --json bench/results/mtp-diag-d0.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate import PROMPTS  # noqa: E402


def _rms(x):
    import mlx.core as mx

    return mx.sqrt(mx.mean(mx.square(x.astype(mx.float32))))


def _cos(a, b):
    import mlx.core as mx

    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    denom = mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf))
    return mx.sum(af * bf) / mx.maximum(denom, 1e-12)


def diag_generate(engine, prompt_ids, max_tokens, n_draft, eos_ids, rescale,
                  base_variant="pre", chain_variant="pre"):
    """固定深度 greedy ループ。spec.py generate() の mtp 経路を写して計測する。"""
    import mlx.core as mx

    from fastmlx._mlx_compat import KVCache

    eos = set(eos_ids)
    caches = engine.text.make_cache()
    mtp_cache = KVCache()
    prompt = mx.array(list(prompt_ids))

    def _base(h):
        return engine.inner.norm(h) if base_variant == "post" else h
    def _chain(h):
        return engine.mtp.norm(h) if chain_variant == "post" else h

    h_all, _ = engine._hidden_forward(prompt, caches, capture=False)
    if prompt.shape[0] > 1:
        engine._mtp_append(prompt[1:], _base(h_all[:, :-1]), mtp_cache)
    h_last = h_all[:, -1:]
    y = mx.argmax(engine._head(h_last, engine.inner.norm), axis=-1).reshape(1)
    mx.eval(y)

    out_tokens = [int(y.item())]
    # depth k (1 始まり) ごとの試行数・一致数・連鎖受理数
    attempts = [0] * (n_draft + 1)
    match = [0] * (n_draft + 1)
    chain = [0] * (n_draft + 1)
    top2_hit = 0
    accept_trace = []
    # RMS 記録: ステップごとに (rms_h_last, [rms_hat_k], [rms_true_k])
    rms_rows = []

    while len(out_tokens) < max_tokens and out_tokens[-1] not in eos:
        proposal_cap = max_tokens - len(out_tokens) - 1
        depth = min(n_draft, max(proposal_cap, 0))
        mtp_off0 = mtp_cache.offset

        drafts = []
        rms_hat = []
        hat_states = []
        dh, dtok = _base(h_last), y
        for k in range(depth):
            if k > 0 and rescale is not None:
                dh = dh * rescale[k - 1]
            h_mtp = engine._mtp_append(dtok, dh, mtp_cache)
            rms_hat.append(_rms(h_mtp[:, -1:]))
            hat_states.append(h_mtp[:, -1:])
            mtp_logits = engine._head(h_mtp[:, -1:], engine.mtp.norm)
            d = mx.argmax(mtp_logits, axis=-1).reshape(1)
            if k == 0:
                # ミニ木 (第 1 リンク幅) の採算判定用: MTP の第 2 候補。
                # greedy の被覆は排反なので p(top1) + p(top2) がそのまま効く。
                masked = mx.where(
                    mx.arange(mtp_logits.shape[-1]) == d[0], -mx.inf, mtp_logits[0, -1]
                )
                second = mx.argmax(masked).reshape(1)
            drafts.append(d)
            dh, dtok = _chain(h_mtp[:, -1:]), d
        window = mx.concatenate([y] + drafts) if drafts else y

        hs, sink = engine._hidden_forward(
            window, caches, capture=window.shape[0] > 1
        )
        logits = engine._head(hs, engine.inner.norm)
        preds = mx.argmax(logits, axis=-1)[0]
        rms_true = [_rms(hs[:, k : k + 1]) for k in range(depth)]
        cos_vals = [
            _cos(hat_states[k], hs[:, k : k + 1]) for k in range(depth)
        ]
        rms_h_last = _rms(h_last)
        mx.eval(preds, window, *rms_hat, *rms_true, *cos_vals, rms_h_last)

        mx.eval(second)
        preds_l, window_l = preds.tolist(), window.tolist()
        if preds_l[0] == int(second.item()):
            top2_hit += 1
        accepted_eos = None
        a = 0
        while a < depth and preds_l[a] == window_l[a + 1]:
            a += 1
            if window_l[a] in eos:
                accepted_eos = a
                break

        for k in range(1, depth + 1):
            attempts[k] += 1
            if preds_l[k - 1] == window_l[k]:
                match[k] += 1
            if a >= k:
                chain[k] += 1
        accept_trace.append(a)
        rms_rows.append(
            (
                float(rms_h_last.item()),
                [float(v.item()) for v in rms_hat],
                [float(v.item()) for v in rms_true],
                [float(v.item()) for v in cos_vals],
            )
        )

        consumed = a if accepted_eos is not None else 1 + a
        if window.shape[0] > 1:
            engine._rollback(caches, sink, len(window_l), consumed)
        mtp_cache.trim(mtp_cache.offset - mtp_off0)
        true_hiddens = mx.concatenate([h_last, hs[:, : consumed - 1]], axis=1)
        engine._mtp_append(window[:consumed], _base(true_hiddens), mtp_cache)
        h_last = hs[:, consumed - 1 : consumed]

        if accepted_eos is not None:
            step_tokens = window_l[1 : a + 1]
        else:
            next_tok = preds_l[a]
            y = mx.array([next_tok])
            step_tokens = window_l[1:consumed] + [next_tok]
        out_tokens.extend(step_tokens)

    return {
        "tokens": out_tokens,
        "attempts": attempts,
        "match": match,
        "chain": chain,
        "top2_hit": top2_hit,
        "accept_trace": accept_trace,
        "rms_rows": rms_rows,
    }


def _mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def summarize(per_prompt, n_draft):
    attempts = [0] * (n_draft + 1)
    match = [0] * (n_draft + 1)
    chain = [0] * (n_draft + 1)
    trace = []
    ratio_by_depth = [[] for _ in range(n_draft + 1)]
    rms_hat_by_depth = [[] for _ in range(n_draft + 1)]
    rms_true_by_depth = [[] for _ in range(n_draft + 1)]
    cos_by_depth = [[] for _ in range(n_draft + 1)]
    top2_hit = 0
    for r in per_prompt.values():
        for k in range(n_draft + 1):
            attempts[k] += r["attempts"][k]
            match[k] += r["match"][k]
            chain[k] += r["chain"][k]
        top2_hit += r.get("top2_hit", 0)
        trace.extend(r["accept_trace"])
        for _, rms_hat, rms_true, cos_vals in r["rms_rows"]:
            for k, (h_hat, h_true, c) in enumerate(
                zip(rms_hat, rms_true, cos_vals), start=1
            ):
                rms_hat_by_depth[k].append(h_hat)
                rms_true_by_depth[k].append(h_true)
                cos_by_depth[k].append(c)
                if h_true > 0:
                    ratio_by_depth[k].append(h_hat / h_true)

    steps = len(trace)
    depths = {}
    for k in range(1, n_draft + 1):
        n_k = attempts[k]
        prev = chain[k - 1] if k > 1 else n_k
        r_mean, r_std = _mean_std(ratio_by_depth[k])
        hat_mean, _ = _mean_std(rms_hat_by_depth[k])
        true_mean, _ = _mean_std(rms_true_by_depth[k])
        cos_mean, cos_std = _mean_std(cos_by_depth[k])
        depths[k] = {
            "attempts": n_k,
            "match_rate": match[k] / n_k if n_k else None,
            "chain_rate": chain[k] / n_k if n_k else None,
            "conditional_rate": chain[k] / prev if prev else None,
            "rms_hat_mean": hat_mean,
            "rms_true_mean": true_mean,
            "rms_ratio_mean": r_mean,
            "rms_ratio_std": r_std,
            "cos_mean": cos_mean,
            "cos_std": cos_std,
        }
    mean_accepted = sum(trace) / steps if steps else 0.0
    p1 = match[1] / attempts[1] if attempts[1] else None
    p2 = top2_hit / attempts[1] if attempts[1] else None
    return {
        "steps": steps,
        "top1_rate_k1": p1,
        "top2_rate_k1": p2,
        "top2_coverage_k1": (p1 + p2) if p1 is not None else None,
        "mean_accepted": mean_accepted,
        "tokens_per_step": 1.0 + mean_accepted,
        "depths": depths,
    }


def main():
    import mlx.core as mx

    from fastmlx._mlx_compat import TextModelArgs, mlx_lm_load
    from fastmlx.mtp import find_snapshot, load_mtp
    from fastmlx.spec import SpecEngine

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="lmstudio-community/Qwen3.8-27B-MLX-4bit"
    )
    parser.add_argument("--original", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--n-draft", type=int, default=3)
    parser.add_argument("--mtp-bits", type=int, default=0)
    parser.add_argument("--prompts", default="code,prose,edit")
    parser.add_argument("--base-variant", default="pre", choices=["pre", "post"])
    parser.add_argument("--chain-variant", default="pre", choices=["pre", "post"])
    parser.add_argument(
        "--rescale",
        default=None,
        help="深度2..K の連鎖入力に掛けるスカラー (カンマ区切り、例 0.8,0.7)",
    )
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()

    rescale = None
    if args.rescale:
        rescale = [float(v) for v in args.rescale.split(",")]
        if len(rescale) != args.n_draft - 1:
            parser.error(
                f"--rescale must have n_draft-1={args.n_draft - 1} values"
            )

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

    text_args = TextModelArgs.from_dict(model.args.text_config)
    quant = {"bits": args.mtp_bits, "group_size": 64} if args.mtp_bits else None
    mtp = load_mtp(find_snapshot(args.original), text_args, quantize=quant)
    mx.eval(mtp.parameters())
    engine = SpecEngine(model, mtp)

    per_prompt = {}
    for name in selected:
        per_prompt[name] = diag_generate(
            engine,
            prompt_ids[name],
            args.max_tokens,
            args.n_draft,
            eos_ids,
            rescale,
            base_variant=args.base_variant,
            chain_variant=args.chain_variant,
        )

    summary = summarize(per_prompt, args.n_draft)
    report = {
        "settings": {
            "model": args.model,
            "original": args.original,
            "max_tokens": args.max_tokens,
            "n_draft": args.n_draft,
            "mtp_bits": args.mtp_bits,
            "prompts": selected,
            "rescale": rescale,
        },
        "summary": summary,
        "per_prompt": {
            name: {
                "n_tokens": len(r["tokens"]),
                "summary": summarize({name: r}, args.n_draft),
            }
            for name, r in per_prompt.items()
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
