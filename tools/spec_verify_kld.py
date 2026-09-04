"""Teacher-forced quality check for Qwen3.6 verification widths.

The harness deliberately keeps one autoregressive continuation fixed.  It
prefills each prompt once, generates that continuation with the true AR path,
then restores the same prefill snapshot before feeding the continuation through
width 1, 4, and 9 target forwards.  The first (prompt-next) distribution is
not part of the comparison: it is independent of verification width.

This is a quality harness rather than a throughput benchmark.  Model imports
are lazy so the helpers and parser can be tested on CPU without MLX/GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


DEFAULT_WIDTHS = (1, 4, 9)
DEFAULT_TOPK = 256
VERDICT_THRESHOLD = 0.0005
TAIL_THRESHOLD = 0.001


# ---------------------------------------------------------------------------
# Pure helpers (also used by the CPU/stub test suite)


def partition_tokens(tokens: Iterable[int], width: int) -> list[list[int]]:
    """Partition a fixed token history into consecutive verification chunks.

    The final chunk may be shorter than ``width``.  It still belongs to the
    requested width run, so the caller can exercise the capture path for that
    run without padding or changing the token history.
    """

    if width <= 0:
        raise ValueError("verification width must be positive")
    values = list(tokens)
    return [values[start : start + width] for start in range(0, len(values), width)]


def _flatten_rows(value) -> list[list[float]]:
    """Flatten ``(B, S, V)``/``(S, V)`` nested values into ``(rows, V)``."""

    if hasattr(value, "tolist"):
        value = value.tolist()

    def visit(node):
        if not isinstance(node, (list, tuple)):
            return [[float(node)]]
        if not node:
            return []
        if isinstance(node[0], (list, tuple)):
            rows = []
            for child in node:
                rows.extend(visit(child))
            return rows
        return [[float(x) for x in node]]

    return visit(value)


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logits row is empty")
    peak = max(values)
    return peak + math.log(sum(math.exp(x - peak) for x in values))


def _scaled_row(values: Sequence[float], temperature: float) -> list[float]:
    if temperature > 0.0:
        return [float(x) / temperature for x in values]
    return [float(x) for x in values]


def _python_kld_summary(logits_p, logits_q, topk: int, temperature: float) -> dict:
    rows_p = _flatten_rows(logits_p)
    rows_q = _flatten_rows(logits_q)
    if len(rows_p) != len(rows_q):
        raise ValueError(
            f"logit row count differs ({len(rows_p)} != {len(rows_q)})"
        )
    if not rows_p:
        return _empty_summary(topk)
    vocab = len(rows_p[0])
    if vocab == 0 or any(len(row) != vocab for row in rows_p + rows_q):
        raise ValueError("logit rows must have one common non-empty vocabulary")
    k = min(topk, vocab)

    klds = []
    p_tails = []
    q_tails = []
    agreements = []
    for p_raw, q_raw in zip(rows_p, rows_q):
        p = _scaled_row(p_raw, temperature)
        q = _scaled_row(q_raw, temperature)
        lp = _logsumexp(p)
        lq = _logsumexp(q)
        order = sorted(range(vocab), key=p.__getitem__, reverse=True)[:k]
        p_log = [p[i] - lp for i in order]
        q_log = [q[i] - lq for i in order]
        p_mass = sum(math.exp(x) for x in p_log)
        q_mass = sum(math.exp(x) for x in q_log)
        klds.append(sum(math.exp(a) * (a - b) for a, b in zip(p_log, q_log)))
        p_tails.append(max(0.0, min(1.0, 1.0 - p_mass)))
        q_tails.append(max(0.0, min(1.0, 1.0 - q_mass)))
        agreements.append(int(max(range(vocab), key=p.__getitem__) == max(range(vocab), key=q.__getitem__)))

    return _make_summary(k, klds, agreements, p_tails, q_tails)


def _is_mlx_array(value) -> bool:
    module = type(value).__module__
    return module.startswith("mlx.")


def _mlx_kld_summary(logits_p, logits_q, topk: int, temperature: float) -> dict | None:
    """Use MLX's top-k operations for real model tensors, if available."""

    if not (_is_mlx_array(logits_p) and _is_mlx_array(logits_q)):
        return None
    try:
        import mlx.core as mx
    except ImportError:
        return None

    if logits_p.shape[:-1] != logits_q.shape[:-1]:
        raise ValueError(
            f"logit shape differs ({logits_p.shape} != {logits_q.shape})"
        )
    vocab = int(logits_p.shape[-1])
    if vocab <= 0:
        return _empty_summary(topk)
    k = min(topk, vocab)
    p = logits_p.astype(mx.float32).reshape(-1, vocab)
    q = logits_q.astype(mx.float32).reshape(-1, vocab)
    if temperature > 0.0:
        p = p / temperature
        q = q / temperature
    logp = p - mx.logsumexp(p, axis=-1, keepdims=True)
    logq = q - mx.logsumexp(q, axis=-1, keepdims=True)
    idx = mx.argpartition(-logp, k - 1, axis=-1)[..., :k]
    top_logp = mx.take_along_axis(logp, idx, axis=-1)
    order = mx.argsort(-top_logp, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    top_logp = mx.take_along_axis(top_logp, order, axis=-1)
    top_logq = mx.take_along_axis(logq, idx, axis=-1)
    p_tail = mx.maximum(0.0, 1.0 - mx.sum(mx.exp(top_logp), axis=-1))
    q_tail = mx.maximum(0.0, 1.0 - mx.sum(mx.exp(top_logq), axis=-1))
    kld = mx.sum(mx.exp(top_logp) * (top_logp - top_logq), axis=-1)
    agree = mx.argmax(p, axis=-1) == mx.argmax(q, axis=-1)
    mx.eval(kld, agree, p_tail, q_tail)
    return _make_summary(
        k,
        [float(x) for x in kld.tolist()],
        [int(x) for x in agree.tolist()],
        [float(x) for x in p_tail.tolist()],
        [float(x) for x in q_tail.tolist()],
    )


def _empty_summary(topk: int) -> dict:
    return _make_summary(topk, [], [], [], [])


def _make_summary(topk: int, klds, agreements, p_tails, q_tails) -> dict:
    positions = len(klds)
    if positions:
        kld_mean = statistics.fmean(klds)
        kld_max = max(klds)
        top1 = statistics.fmean(agreements)
        p_tail = statistics.fmean(p_tails)
        q_tail = statistics.fmean(q_tails)
        p_tail_max = max(p_tails)
        q_tail_max = max(q_tails)
    else:
        kld_mean = kld_max = top1 = p_tail = q_tail = 0.0
        p_tail_max = q_tail_max = 0.0
    return {
        "positions": positions,
        "topk": int(topk),
        "kld_mean": float(kld_mean),
        "kld_max": float(kld_max),
        "kld_per_position": [float(x) for x in klds],
        "top1_agreement": float(top1),
        "top1_agree": float(top1),
        "top1_agreement_rate": float(top1),
        # The candidate tail is measured on the reference top-k support.  This
        # is the same support used by the approximate KLD, so the two masses
        # remain comparable even if the candidate's native top-k set differs.
        "tail_mass": float(p_tail),
        "reference_tail_mass": float(p_tail),
        "candidate_tail_mass": float(q_tail),
        "reference_tail_mass_max": float(p_tail_max),
        "candidate_tail_mass_max": float(q_tail_max),
        "tail_mass_delta": float(q_tail - p_tail),
        "tail_mass_per_position": [float(x) for x in p_tails],
        "candidate_tail_mass_per_position": [float(x) for x in q_tails],
    }


def summarize_kld(
    logits_p,
    logits_q,
    topk: int = DEFAULT_TOPK,
    temperature: float = 0.0,
) -> dict:
    """Return project-compatible top-k approximate ``KL(P || Q)`` statistics.

    ``P`` supplies the top-k support.  KLD is therefore the same lower-side
    approximation used by ``bench/quant_eval.py``: ``sum(p_k * (log p_k -
    log q_k))``.  Positive temperature scales both logit arrays before
    normalisation; it never changes the fixed token history.
    """

    if topk <= 0:
        raise ValueError("topk must be positive")
    if temperature < 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and non-negative")
    result = _mlx_kld_summary(logits_p, logits_q, topk, temperature)
    return result if result is not None else _python_kld_summary(
        logits_p, logits_q, topk, temperature
    )


def aggregate_width_summaries(case_results: Sequence[dict], widths: Sequence[int]) -> dict:
    """Aggregate per-case width summaries with equal case weighting."""

    out = {}
    for width in widths:
        entries = []
        for case in case_results:
            by_width = case.get("widths", case)
            item = by_width.get(str(width), by_width.get(width))
            if item is not None and item.get("positions", 0) > 0:
                entries.append(item)
        if not entries:
            out[str(width)] = _empty_summary(DEFAULT_TOPK)
            continue
        # Per-prompt means are the project's aggregate convention.  Keep the
        # total positions as evidence, while avoiding a long-context case
        # silently dominating the shorter prompts.
        klds = [item["kld_mean"] for item in entries]
        tops = [item.get("top1_agreement", item.get("top1_agree", 0.0)) for item in entries]
        p_tails = [item.get("reference_tail_mass", item.get("tail_mass", 0.0)) for item in entries]
        q_tails = [item.get("candidate_tail_mass", 0.0) for item in entries]
        merged = dict(_empty_summary(entries[0].get("topk", DEFAULT_TOPK)))
        merged.update(
            positions=sum(int(item.get("positions", 0)) for item in entries),
            cases=len(entries),
            kld_mean=float(statistics.fmean(klds)),
            kld_max=float(max(item.get("kld_max", item["kld_mean"]) for item in entries)),
            top1_agreement=float(statistics.fmean(tops)),
            top1_agree=float(statistics.fmean(tops)),
            top1_agreement_rate=float(statistics.fmean(tops)),
            tail_mass=float(statistics.fmean(p_tails)),
            reference_tail_mass=float(statistics.fmean(p_tails)),
            candidate_tail_mass=float(statistics.fmean(q_tails)),
            reference_tail_mass_max=float(max(
                item.get("reference_tail_mass_max", item.get("reference_tail_mass", 0.0))
                for item in entries
            )),
            candidate_tail_mass_max=float(max(
                item.get("candidate_tail_mass_max", item.get("candidate_tail_mass", 0.0))
                for item in entries
            )),
            tail_mass_delta=float(statistics.fmean(q - p for p, q in zip(p_tails, q_tails))),
        )
        out[str(width)] = merged
    return out


def quality_verdict(
    cap3_kld: float,
    current_kld: float,
    threshold: float = VERDICT_THRESHOLD,
    reference_tail_mass_max: float = 0.0,
    tail_threshold: float = TAIL_THRESHOLD,
) -> dict:
    """Apply the primary cap3-vs-current KLD acceptance rule."""

    delta = float(cap3_kld) - float(current_kld)
    kld_pass = delta <= threshold
    tail_pass = reference_tail_mass_max <= tail_threshold
    passed = kld_pass and tail_pass
    return {
        "cap3_kld": float(cap3_kld),
        "current_kld": float(current_kld),
        "delta": delta,
        "threshold": float(threshold),
        "reference_tail_mass_max": float(reference_tail_mass_max),
        "tail_threshold": float(tail_threshold),
        "kld_pass": bool(kld_pass),
        "tail_pass": bool(tail_pass),
        "cap3_kld_minus_current_kld": delta,
        "cap3_le_current": bool(cap3_kld <= current_kld),
        "pass": bool(passed),
        "verdict": "PASS" if passed else "FAIL",
    }


# ---------------------------------------------------------------------------
# Real-model path (imports intentionally remain lazy)


def _eval_logits(mx, logits, sink) -> None:
    """Materialise logits and capture state before the next chunk."""

    pending = [logits]
    if sink:
        for item in sink:
            if isinstance(item, (tuple, list)):
                # GDN capture records put states_all and conv_input at slots 1
                # and 2.  Evaluating them makes the cache boundary explicit;
                # unknown sink records are simply ignored.
                pending.extend(x for x in item[1:3] if _is_mlx_array(x))
    if pending and any(_is_mlx_array(x) for x in pending):
        mx.eval(*pending)


def teacher_force(engine, caches, continuation: Sequence[int], width: int) -> list:
    """Run fixed ``continuation[:-1]`` through one target verify width.

    The first continuation token is already the prompt-next sample, so feeding
    it produces the distribution for token two.  This intentionally excludes
    the width-independent prompt-next distribution from all quality metrics.
    """

    chunks = partition_tokens(continuation[:-1], width)
    if not chunks:
        return []
    # Importing ``mlx.core`` itself aborts on a headless host (rather than
    # consistently raising ImportError), so do not probe it for stub engines.
    # A real SpecEngine is defined in ``mlxturbo.spec`` and already implies
    # that the MLX runtime is available.
    if type(engine).__module__.startswith("mlxturbo."):
        try:
            import mlx.core as mx
        except ImportError:
            mx = None
    else:
        mx = None
    logits = []
    for chunk in chunks:
        token_arg = (
            mx.array(chunk, dtype=mx.int32) if mx is not None else list(chunk)
        )
        if len(chunk) == 1:
            hidden, sink = engine._hidden_forward(token_arg, caches, capture=False, staged=True)
        else:
            hidden, sink = engine._hidden_forward(token_arg, caches, capture=True)
        row_logits = engine._head(hidden, engine.inner.norm)
        if mx is not None:
            _eval_logits(mx, row_logits, sink)
        logits.append(row_logits)
    return logits


def _concat_logits(logits):
    if not logits:
        return []
    if all(_is_mlx_array(item) for item in logits):
        import mlx.core as mx

        result = mx.concatenate(logits, axis=1)
        mx.eval(result)
        return result
    # Stub engines often return nested Python values.  Concatenating the row
    # dimension here preserves the same shape contract as real MLX tensors.
    rows = []
    for item in logits:
        rows.extend(_flatten_rows(item))
    return rows


def _source_hashes() -> dict:
    paths = (
        Path(__file__).resolve(),
        TOOLS_DIR / "decode_ab_generic.py",
        REPO_ROOT / "mlxturbo/spec.py",
    )
    return {
        str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.exists()
    }


def _parse_csv_ints(parser: argparse.ArgumentParser, raw: str, name: str, minimum: int) -> list[int]:
    values = [part.strip() for part in raw.split(",")]
    if not values or any(not part for part in values):
        parser.error(f"--{name} は空でない整数をカンマ区切りで指定する")
    try:
        parsed = [int(part) for part in values]
    except ValueError:
        parser.error(f"--{name} は整数を指定する")
    if any(value < minimum for value in parsed):
        parser.error(f"--{name} は {minimum} 以上を指定する")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.6 SpecEngine の検証幅ごとの top-256 KLD を測る"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--mtp", default=None)
    parser.add_argument("--mtp-bits", type=int, default=4)
    parser.add_argument("--ctxs", default="0,4000,17000")
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument(
        "--widths", "--verify-widths", dest="widths", default="1,4,9",
        help="固定 verify 幅 (1 を必ず含める。既定: 1,4,9)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--out", required=True)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.model.strip():
        parser.error("--model は空にできない")
    if args.mtp_bits <= 0:
        parser.error("--mtp-bits は正の整数を指定する")
    if args.prompts < 1 or args.prompts > 3:
        parser.error("--prompts は1〜3を指定する")
    if args.tokens < 2:
        parser.error("--tokens は2以上を指定する")
    if args.temp < 0.0 or not math.isfinite(args.temp):
        parser.error("--temp は有限の0以上を指定する")
    if args.topk <= 0:
        parser.error("--topk は正の整数を指定する")
    args.ctx_values = _parse_csv_ints(parser, args.ctxs, "ctxs", 0)
    args.width_values = _parse_csv_ints(parser, args.widths, "widths", 1)
    if len(set(args.width_values)) != len(args.width_values):
        parser.error("--widths に同じ幅を重複指定しない")
    required_widths = {1, 4, 9}
    missing_widths = sorted(required_widths - set(args.width_values))
    if missing_widths:
        parser.error(
            "--widths は参照幅1・cap3幅4・現行MTP幅9を必ず含める"
            f"（不足: {','.join(map(str, missing_widths))}）"
        )
    return args


def _run_case(args, engine, requested_ctx: int, case_idx: int, ids: Sequence[int]) -> dict:
    import decode_ab_generic as generic

    # No MTP state is needed for an AR continuation, but using the shared
    # helper ensures the exact production prefill/chunking path and snapshot
    # semantics are exercised.
    session, snapshot, prefill_s = generic.prefill_once(engine, ids, 0, 0)
    import mlx.core as mx

    mx.random.seed(args.seed + case_idx)
    generated = generic.run_resumed(
        engine,
        ids,
        session,
        snapshot,
        args.tokens,
        (),
        0,
        0,
        lookup_len=0,
        temp=args.temp,
    )
    continuation = list(generated.get("tokens", ()))
    width_results = {}
    reference = None
    # Width 1 is the reference even when a caller lists configurable widths in
    # another order.  Candidate runs still use the caller's requested set.
    run_widths = [1] + [width for width in args.width_values if width != 1]
    for width in run_widths:
        # Every width starts from the identical prefill snapshot.  No generated
        # candidate is ever fed back into another width run.
        generic._restore_session(session, snapshot)
        logs = teacher_force(engine, session.caches, continuation, width)
        joined = _concat_logits(logs)
        if width == 1:
            reference = joined
            width_results[str(width)] = summarize_kld(
                reference, reference, topk=args.topk, temperature=args.temp
            ) if continuation[1:] else _empty_summary(args.topk)
        else:
            width_results[str(width)] = summarize_kld(
                reference, joined, topk=args.topk, temperature=args.temp
            ) if continuation[1:] else _empty_summary(args.topk)
        if _is_mlx_array(joined):
            del joined
            mx.clear_cache()

    verdict = None
    if "4" in width_results and "9" in width_results:
        verdict = quality_verdict(
            width_results["4"]["kld_mean"],
            width_results["9"]["kld_mean"],
            reference_tail_mass_max=width_results["4"]["reference_tail_mass_max"],
        )

    return {
        "requested_ctx": requested_ctx,
        "ctx": len(ids),
        "case_idx": case_idx,
        "prefill_s": float(prefill_s),
        "continuation_tokens": continuation,
        "positions": max(len(continuation) - 1, 0),
        "initial_prompt_next_omitted": True,
        "widths": width_results,
        "verdict": verdict,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    import decode_ab_generic as generic

    _model, tokenizer, engine, _eos_ids, _guard = generic.load_model(args)
    cases = []
    for requested_ctx in args.ctx_values:
        built = generic.build_cases(tokenizer, requested_ctx, long_count=args.prompts)
        if requested_ctx == 0:
            built = built[: args.prompts]
        for case_idx, (_kind, ids) in enumerate(built):
            print(
                f"ctx={len(ids)} requested_ctx={requested_ctx} case={case_idx}",
                flush=True,
            )
            case = _run_case(args, engine, requested_ctx, case_idx, ids)
            cases.append(case)
            print(
                "  "
                + " | ".join(
                    f"w{width} KLD={case['widths'][str(width)]['kld_mean']:.6f}"
                    for width in args.width_values
                ),
                flush=True,
            )

    aggregate = aggregate_width_summaries(cases, args.width_values)
    if "4" not in aggregate or "9" not in aggregate:
        raise RuntimeError("widths 1,4,9 are required for the cap3/current verdict")
    verdict = quality_verdict(
        aggregate["4"]["kld_mean"], aggregate["9"]["kld_mean"],
        reference_tail_mass_max=aggregate["4"]["reference_tail_mass_max"],
    )
    by_context = {}
    for requested_ctx in args.ctx_values:
        context_cases = [
            case for case in cases if case["requested_ctx"] == requested_ctx
        ]
        context_aggregate = aggregate_width_summaries(
            context_cases, args.width_values
        )
        context_verdict = quality_verdict(
            context_aggregate["4"]["kld_mean"],
            context_aggregate["9"]["kld_mean"],
            reference_tail_mass_max=context_aggregate["4"][
                "reference_tail_mass_max"
            ],
        )
        by_context[str(requested_ctx)] = {
            "aggregate": context_aggregate,
            "verdict": context_verdict,
        }
    verdict["all_contexts_pass"] = all(
        item["verdict"]["pass"] for item in by_context.values()
    )
    verdict["all_cases_pass"] = all(
        case["verdict"] is not None and case["verdict"]["pass"]
        for case in cases
    )
    # CLAUDE.mdの既定線は全体の現行比ΔKLD <= +0.0005。context/case別は
    # 平均に隠れた偏りを読む診断として残すが、後から強い採用条件へ変えない。
    verdict["aggregate_pass"] = verdict["pass"]
    output = {
        "meta": {
            "kind": "spec-verify-kld",
            "version": "v1",
            "argv": sys.argv,
            "model": args.model,
            "mtp": args.mtp,
            "mtp_bits": args.mtp_bits,
            "ctxs": args.ctx_values,
            "prompts": args.prompts,
            "tokens": args.tokens,
            "temp": args.temp,
            "topk": args.topk,
            "widths": args.width_values,
            "seed": args.seed,
            "initial_prompt_next_omitted": True,
            "ignore_eos_for_fixed_length": True,
            "prefill_snapshot_shared": True,
            "source_sha256": _source_hashes(),
        },
        "cases": cases,
        "aggregate": aggregate,
        "by_context": by_context,
        "verdict": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(output, ensure_ascii=False, indent=1))
    print(
        f"aggregate cap3={aggregate['4']['kld_mean']:.6f} "
        f"current={aggregate['9']['kld_mean']:.6f} "
        f"delta={verdict['delta']:+.6f} verdict={verdict['verdict']}",
        flush=True,
    )
    print(f"書き出し: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
