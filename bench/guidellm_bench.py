#!/usr/bin/env python3
"""GuideLLM 0.7.3 の公開比較用 scenario を固定して実行する。"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDELLM = ROOT / "tools" / "guidellm.sh"


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("1 以上を指定してください")
    return parsed


def _streams(value: str) -> list[int]:
    try:
        parsed = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("例: 1,2,4,8") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("各 stream は 1 以上にしてください")
    return parsed


def build_scenario(args: argparse.Namespace, outputs: dict[str, Path]) -> dict:
    profile: dict = {
        "kind": "synchronous",
        "warmup": {"mode": "requests", "value": args.warmup_requests},
    }
    if args.mode == "concurrent":
        profile = {
            "kind": "concurrent",
            "streams": args.streams,
            "warmup": {"mode": "requests", "value": args.warmup_requests},
        }
    elif args.mode == "sweep":
        profile = {
            "kind": "sweep",
            "sweep_size": args.sweep_size,
            "max_concurrency": args.max_concurrency,
            "warmup": {"mode": "requests", "value": args.warmup_requests},
        }

    data: dict = {
        "kind": "synthetic_text",
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
    }
    if args.mode == "prefix":
        data["prefix_buckets"] = [
            {
                "bucket_weight": 100,
                "prefix_count": args.prefix_count,
                "prefix_tokens": args.prefix_tokens,
            }
        ]

    body = {"temperature": args.temperature}
    if args.thinking == "off":
        body["reasoning_effort"] = "none"

    labels = {
        "engine": args.engine,
        "mode": args.mode,
        "guidellm": "0.7.3",
        "cooling": args.cooling,
    }
    for item in args.label:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise ValueError(f"--label は key=value で指定してください: {item!r}")
        labels[key] = value

    return {
        "metadata": {"labels": labels},
        "spec": {
            "backend": {
                "kind": "openai_http",
                "target": args.target,
                "model": args.model,
                "request_format": "/v1/chat/completions",
                "http2": False,
                "timeout": args.timeout,
                "stream": True,
                "extras": {"body": body},
            },
            "profile": profile,
            "constraints": [{"kind": "max_requests", "count": args.requests}],
            "tokenizer": {"kind": "hf_auto", "model": args.tokenizer},
            "data": [data],
            "seed": {"kind": "static", "value": args.seed},
            "outputs": [
                {"kind": "json", "path": str(outputs["json"])},
                {"kind": "csv", "path": str(outputs["csv"])},
                {"kind": "html", "path": str(outputs["html"])},
            ],
            "metrics": {"kind": "generative", "sample_size": args.sample_size},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="サーバーへ送る served model id")
    parser.add_argument("--tokenizer", required=True, help="同じ tokenizer のローカルpathかHF id")
    parser.add_argument("--engine", required=True, help="結果へ記録するエンジン名")
    parser.add_argument(
        "--mode", choices=("latency", "prefix", "concurrent", "sweep"), default="latency"
    )
    parser.add_argument("--prompt-tokens", type=_positive, default=512)
    parser.add_argument("--output-tokens", type=_positive, default=256)
    parser.add_argument("--requests", type=_positive, default=12)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--prefix-tokens", type=_positive, default=4000)
    parser.add_argument("--prefix-count", type=_positive, default=1)
    parser.add_argument("--streams", type=_streams, default=[1, 2, 4, 8])
    parser.add_argument("--sweep-size", type=_positive, default=6)
    parser.add_argument("--max-concurrency", type=_positive, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=("off", "default"), default="off")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--cooling", default="unspecified")
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("bench/results/guidellm"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup_requests < 0 or args.warmup_requests >= args.requests:
        raise SystemExit("--warmup-requests は 0 以上、--requests 未満にしてください")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or (
        f"{args.engine}-{args.mode}-p{args.prompt_tokens}-o{args.output_tokens}"
    )
    outputs = {suffix: out_dir / f"{name}.{suffix}" for suffix in ("json", "csv", "html")}
    scenario_path = out_dir / f"{name}.scenario.json"
    scenario = build_scenario(args, outputs)
    scenario_path.write_text(json.dumps(scenario, ensure_ascii=False, indent=2) + "\n")

    command = [str(GUIDELLM), "run", "--config", str(scenario_path)]
    print(f"scenario: {scenario_path}")
    print(f"command:  {shlex.join(command)}")
    if args.dry_run:
        return 0

    env = os.environ.copy()
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
