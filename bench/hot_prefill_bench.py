#!/usr/bin/env python3
"""Flash-Nextのhot prefillを固定差分で測る。

通常のchatベンチは生成文を再tokenizeするため、warm側の新規tokenが16〜274で
揺れる。この道具は`/v1/completions`へtoken ID列を直接送り、1回目を1 token
生成で止める。FlashSpecRunnerはその場合promptだけをsessionへpublishするので、
次のrequestの差分を0/16/64/256 tokenへ正確に固定できる。

`pure_append`は前promptへ指定数を足す。`retokenized`は末尾を既定8 token
書き換えてから指定数を足し、chat template/BPE境界が変わる場合のcheckpoint
復元を再現する。各系列は差分を累積するため、cold prefillは文脈・反復・mode
ごとに1回だけで済む。出力には実際のLCP/reused/new、pool bytes、MLX memoryを
残す。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from _bench_text import text_pool  # noqa: E402
from vs_mlx_serve import SHORT, Server, install_term_handler, model_id  # noqa: E402


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _stream_completion(port: int, model: str, prompt_ids: list[int]) -> dict:
    body = json.dumps({
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            item = json.loads(payload)
            if "error" in item:
                raise RuntimeError(item["error"])
            choices = item.get("choices") or []
            if choices and ttft is None:
                # 通常は最初のtoken chunk。EOSが最初なら生成完了chunkになるが、
                # max_tokens=1なので、どちらもprefill+first-token完了時点を表す。
                ttft = time.perf_counter() - started
            if item.get("usage") is not None:
                usage = item["usage"]
    wall = time.perf_counter() - started
    if ttft is None:
        ttft = wall
    if usage is None:
        raise RuntimeError("stream usageが返らなかった")
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    return {"ttft_s": ttft, "wall_s": wall, "cached_tokens": cached, "usage": usage}


def _lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _replacement_tail(current: list[int], width: int, palette: list[int]) -> list[int]:
    if width < 1 or width >= len(current):
        raise ValueError("rewrite tailは1以上、prompt長未満が必要")
    out = []
    for i, old in enumerate(current[-width:]):
        candidate = next(
            (palette[(i + offset) % len(palette)] for offset in range(len(palette))
             if palette[(i + offset) % len(palette)] != old),
            None,
        )
        if candidate is None:
            raise ValueError("書き換えtokenを作れなかった")
        out.append(candidate)
    return out


def _next_prompt(
    current: list[int], suffix_tokens: int, palette: list[int],
    mode: str, rewrite_tail: int,
) -> tuple[list[int], int]:
    suffix = [palette[i % len(palette)] for i in range(suffix_tokens)]
    if mode == "pure_append":
        return current + suffix, len(current)
    if mode != "retokenized":
        raise ValueError(f"未知のmode: {mode}")
    rewritten = _replacement_tail(current, rewrite_tail, palette)
    prompt = current[:-rewrite_tail] + rewritten + suffix
    expected_lcp = len(current) - rewrite_tail
    if _lcp(current, prompt) != expected_lcp:
        raise AssertionError("retokenized promptのLCPが意図した位置と違う")
    return prompt, expected_lcp


def _non_special_token(start: int, vocab_size: int, special: set[int]) -> int:
    for offset in range(vocab_size):
        candidate = (start + offset) % vocab_size
        if candidate not in special:
            return candidate
    raise ValueError("通常tokenが無いtokenizer")


def _base_prompt(
    pool_ids: list[int], short_ids: list[int], ctx: int, case_id: int,
    vocab_size: int, special: set[int],
) -> list[int]:
    if ctx == 0:
        ids = list(short_ids)
    else:
        if len(pool_ids) < ctx:
            raise ValueError(f"text poolが{ctx} tokenに足りない")
        ids = list(pool_ids[:ctx])
    if len(ids) < 2:
        raise ValueError("promptは2 token以上必要")
    # case間の先頭LCPを0にし、前のsessionをcold requestが再利用しないようにする。
    ids[0] = _non_special_token(10_000 + case_id, vocab_size, special)
    return ids


def _selection(status: dict) -> dict:
    telemetry = status.get("session_telemetry") or {}
    selected = telemetry.get("last_selection")
    if not isinstance(selected, dict):
        raise RuntimeError("/api/statusにlast_selectionが無い")
    return selected


def main() -> int:
    install_term_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ngram", default=None)
    parser.add_argument("--mtp", default=None)
    parser.add_argument("--port", type=int, default=8164)
    parser.add_argument("--ctxs", default="0,4000,17000,25000,32000,50000")
    parser.add_argument("--suffixes", default="0,16,64,256")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--rewrite-tail", type=int, default=8)
    parser.add_argument("--server-log", default="scratchpad/log-hot-prefill-bench.txt")
    parser.add_argument("--out", default="bench/results/hot-prefill-bench.json")
    parser.add_argument("--turbo-extra", default=None)
    args = parser.parse_args()

    ctxs = [int(value) for value in args.ctxs.split(",")]
    suffixes = [int(value) for value in args.suffixes.split(",")]
    if args.reps < 1 or any(value < 0 for value in ctxs + suffixes):
        raise SystemExit("ctx/suffixは0以上、repsは1以上が必要")
    if suffixes != sorted(suffixes) or not suffixes or suffixes[0] != 0:
        raise SystemExit("suffixesは0から始まる昇順にする")

    from transformers import AutoTokenizer

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pool_ids = list(tokenizer.encode(text_pool(), add_special_tokens=False))
    short_ids = list(tokenizer.encode(SHORT, add_special_tokens=False))
    special = set(tokenizer.all_special_ids)
    vocab_size = len(tokenizer)
    palette = [token for token in pool_ids[-1024:] if token not in special]
    if len(set(palette)) < 2:
        raise SystemExit("suffix/rewrite用の通常tokenが足りない")

    argv = [sys.executable, "-m", "mlxturbo.server", "--model", model_path,
            "--host", "127.0.0.1", "--port", str(args.port)]
    if args.ngram:
        argv += ["--ngram", os.path.expanduser(args.ngram)]
    if args.mtp:
        argv += ["--mtp", os.path.expanduser(args.mtp)]
    if args.turbo_extra:
        argv += shlex.split(args.turbo_extra)
    argv += ["--max-sessions", "1", "--log-level", "debug"]

    rows = []
    case_id = 0
    started = time.time()

    def log(message: str) -> None:
        elapsed = time.time() - started
        print(f"[{time.strftime('%H:%M:%S')}] (+{elapsed:6.1f}s) {message}", flush=True)

    log("server起動開始")
    with Server("mlxturbo-hot-prefill", argv, args.port, log_path=args.server_log):
        mid = model_id(args.port)
        status_url = f"http://127.0.0.1:{args.port}/api/status"
        initial = _get_json(status_url)
        if initial.get("runner_kind") != "flash_spec":
            raise RuntimeError(f"FlashSpecRunnerが必要: {initial.get('runner_kind')}")
        log("server起動完了")

        # 測定promptと先頭tokenが違う短いrequestで初回compileだけを済ませる。
        warm_ids = list(short_ids)
        warm_ids[0] = _non_special_token(9_000, vocab_size, special)
        _stream_completion(args.port, mid, warm_ids)
        log("compile warmup完了")

        for ctx in ctxs:
            for rep in range(args.reps):
                for mode in ("pure_append", "retokenized"):
                    case_id += 1
                    base = _base_prompt(
                        pool_ids, short_ids, ctx, case_id, vocab_size, special
                    )
                    cold = _stream_completion(args.port, mid, base)
                    cold_status = _get_json(status_url)
                    cold_selection = _selection(cold_status)
                    if cold_selection["match_kind"] != "miss" or cold["cached_tokens"] != 0:
                        raise RuntimeError(
                            f"coldがcache hitした: ctx={ctx} rep={rep} mode={mode} "
                            f"selection={cold_selection} usage={cold['usage']}"
                        )

                    current = base
                    for suffix in suffixes:
                        prompt, expected_lcp = _next_prompt(
                            current, suffix, palette, mode, args.rewrite_tail
                        )
                        measured = _stream_completion(args.port, mid, prompt)
                        status = _get_json(status_url)
                        selected = _selection(status)
                        cached = measured["cached_tokens"]
                        if cached != selected["reused_tokens"]:
                            raise RuntimeError(
                                f"usage/statusのreuse不一致: {cached} != {selected}"
                            )
                        row = {
                            "ctx": ctx,
                            "rep": rep,
                            "mode": mode,
                            "requested_suffix_tokens": suffix,
                            "prompt_tokens": len(prompt),
                            "expected_lcp": expected_lcp,
                            "ttft_s": measured["ttft_s"],
                            "wall_s": measured["wall_s"],
                            "cached_tokens": cached,
                            "selection": selected,
                            "pool_allocated_bytes": status["session_telemetry"].get(
                                "pool_allocated_bytes"
                            ),
                            "pool_known_allocated_bytes": status["session_telemetry"].get(
                                "pool_known_allocated_bytes"
                            ),
                            "active_memory_bytes": status.get("active_memory_bytes"),
                            "cache_memory_bytes": status.get("cache_memory_bytes"),
                            "rss_bytes": status.get("rss_bytes"),
                        }
                        rows.append(row)
                        current = prompt
                        log(
                            f"ctx={ctx} rep={rep} {mode} suffix={suffix} "
                            f"TTFT={row['ttft_s']:.3f}s reused/new="
                            f"{selected['reused_tokens']}/{selected['new_tokens']}"
                        )

    output = {
        "engine": "mlxturbo",
        "runner": "flash_spec",
        "model": model_path,
        "ctxs": ctxs,
        "suffixes": suffixes,
        "reps": args.reps,
        "rewrite_tail": args.rewrite_tail,
        "max_sessions": 1,
        "rows": rows,
        "argv": argv,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"書き出し: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
