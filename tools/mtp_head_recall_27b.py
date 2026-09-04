"""27B MTP proposal head の q2 top-k recall を、生成を変えずに測る。

exact q4 head の argmax をそのまま draft に使い続け、その横で q4 を一度
dequantize→q2へ再量子化した粗ヘッドの上位候補だけを記録する。生成前後の
token列も比較する。採択前の診断なので速度値は判定に使わない。

    BIGLOCK_NO_WORKER=1 tools/biglock.sh .venv/bin/python \
      tools/mtp_head_recall_27b.py \
      --model ~/models/qwen38-27b-4bit --mtp ~/models/qwen38-27b-mtp \
      --tokens 512 --top 32 --out bench/results/mtp-head-q2-recall-27b-0904.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from tools import decode_ab_generic


class ProposalRecall:
    """``SpecEngine._head`` のMTP呼び出しだけを横から観測する。"""

    def __init__(self, engine, top: int):
        import mlx.core as mx

        self.engine = engine
        self.top = top
        self.enabled = False
        self.records: list[int | None] = []
        self.rerank_matches: list[bool] = []
        self._head = engine._head
        lm = engine.text.lm_head
        if not all(hasattr(lm, name) for name in ("weight", "scales", "biases")):
            raise RuntimeError("量子化 lm_head が無いため q2 recall を測れない")
        t0 = time.perf_counter()
        w = mx.dequantize(
            lm.weight,
            lm.scales,
            lm.biases,
            group_size=lm.group_size,
            bits=lm.bits,
        )
        self.weight, self.scales, self.biases = mx.quantize(
            w, group_size=64, bits=2
        )
        self.lm = lm
        mx.eval(self.weight, self.scales, self.biases)
        self.build_s = time.perf_counter() - t0
        self.nbytes = sum(
            a.nbytes for a in (self.weight, self.scales, self.biases)
        )
        del w
        mx.clear_cache()
        engine._head = self.call

    def reset(self) -> None:
        self.records = []
        self.rerank_matches = []

    def call(self, h_prenorm, norm):
        import mlx.core as mx

        exact = self._head(h_prenorm, norm)
        if not self.enabled or norm is not self.engine.mtp.norm:
            return exact
        row = norm(h_prenorm)[:, -1]
        coarse = mx.quantized_matmul(
            row,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=64,
            bits=2,
        )
        pool = mx.argpartition(-coarse, self.top - 1, axis=-1)[..., : self.top]
        vals = mx.take_along_axis(coarse, pool, axis=-1)
        ranked = mx.take_along_axis(pool, mx.argsort(-vals, axis=-1), axis=-1)
        rows = mx.dequantize(
            self.lm.weight[pool[0]],
            self.lm.scales[pool[0]],
            self.lm.biases[pool[0]],
            group_size=self.lm.group_size,
            bits=self.lm.bits,
        )
        scores = row.astype(rows.dtype) @ rows.T
        best = mx.argmax(scores, axis=-1, keepdims=True)
        rerank_tok = mx.take_along_axis(pool, best, axis=-1)
        exact_tok = mx.argmax(exact[:, -1], axis=-1)
        mx.eval(ranked, exact_tok, rerank_tok)
        ids = ranked[0].tolist()
        tok = int(exact_tok[0].item())
        self.records.append(ids.index(tok) + 1 if tok in ids else None)
        self.rerank_matches.append(int(rerank_tok[0, 0].item()) == tok)
        return exact


def _summary(
    ranks: list[int | None], rerank_matches: list[bool], ks=(1, 4, 8, 16, 32)
) -> dict:
    n = len(ranks)
    hits = {str(k): sum(r is not None and r <= k for r in ranks) for k in ks}
    return {
        "n": n,
        "hits": hits,
        "recall": {k: (v / n if n else 0.0) for k, v in hits.items()},
        "misses_top32": sum(r is None for r in ranks),
        "rerank_agree": (
            sum(rerank_matches) / len(rerank_matches) if rerank_matches else 0.0
        ),
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="27B MTP q2 coarse head の recall 診断")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--top", type=int, default=32)
    ap.add_argument("--n-draft", type=int, default=3)
    ap.add_argument("--max-draft", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.top < 1 or args.top > 256:
        ap.error("--top は 1..256")
    if args.no_mtp:
        ap.error("proposal recall には MTP が必要")
    return args


def main(argv=None) -> int:
    import mlx.core as mx

    args = parse_args(argv)
    # この道具の基準は常に従来の全語彙q4 proposal。本番のrerank既定化後も
    # observerが `_head` 呼び出しを拾えるよう、engine構築前に明示して切る。
    os.environ["MLXTURBO_DRAFT_RERANK"] = "0"
    model, tok, engine, eos_ids, _guard = decode_ab_generic.load_model(args)
    tracer = ProposalRecall(engine, args.top)
    print(
        f"[recall] q2 head {tracer.nbytes / 2**20:.1f} MiB、"
        f"構築 {tracer.build_s:.2f}s",
        flush=True,
    )
    cases = decode_ab_generic.build_cases(tok, 0)
    rows = []
    for case_idx, (_kind, ids) in enumerate(cases):
        # exact側とtrace側の双方で初回段差を捨てる。traceの記録も捨てる。
        tracer.enabled = False
        decode_ab_generic.run_once(
            engine, ids, 32, eos_ids, args.n_draft, args.max_draft
        )
        tracer.enabled = True
        tracer.reset()
        decode_ab_generic.run_once(
            engine, ids, 32, eos_ids, args.n_draft, args.max_draft
        )

        tracer.enabled = False
        base = decode_ab_generic.run_once(
            engine, ids, args.tokens, eos_ids, args.n_draft, args.max_draft
        )
        tracer.enabled = True
        tracer.reset()
        traced = decode_ab_generic.run_once(
            engine, ids, args.tokens, eos_ids, args.n_draft, args.max_draft
        )
        if traced["tokens"] != base["tokens"]:
            raise RuntimeError(f"case {case_idx}: recall trace が生成列を変えた")
        summary = _summary(tracer.records, tracer.rerank_matches)
        rows.append(
            {
                "case_idx": case_idx,
                "ctx": len(ids),
                "n_out": traced["n_out"],
                "summary": summary,
                "ranks": tracer.records,
                "rerank_matches": tracer.rerank_matches,
                "tokens": traced["tokens"],
            }
        )
        print(
            f"[recall] case={case_idx} ctx={len(ids)} proposals={summary['n']} "
            + " ".join(
                f"R@{k}={summary['recall'][str(k)] * 100:.3f}%"
                for k in (1, 4, 8, 16, 32)
            )
            + f" rerank一致={summary['rerank_agree'] * 100:.3f}%",
            flush=True,
        )

    ranks = [rank for row in rows for rank in row["ranks"]]
    rerank_matches = [
        match for row in rows for match in row["rerank_matches"]
    ]
    overall = _summary(ranks, rerank_matches)
    payload = {
        "version": "mtp-head-q2-recall/v1",
        "model": os.path.realpath(os.path.expanduser(args.model)),
        "mtp": os.path.realpath(os.path.expanduser(args.mtp)),
        "bits": 2,
        "group_size": 64,
        "top": args.top,
        "build_s": tracer.build_s,
        "resident_bytes": tracer.nbytes,
        "overall": overall,
        "rows": rows,
        "source_sha256": {
            "tools/mtp_head_recall_27b.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "mlxturbo/spec.py": hashlib.sha256(
                (REPO_ROOT / "mlxturbo/spec.py").read_bytes()
            ).hexdigest(),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(
        "[recall] overall "
        + " ".join(
            f"R@{k}={overall['recall'][str(k)] * 100:.3f}%"
            for k in (1, 4, 8, 16, 32)
        )
        + f" rerank一致={overall['rerank_agree'] * 100:.3f}%"
        + f" / 出力 {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
