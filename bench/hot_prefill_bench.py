#!/usr/bin/env python3
"""Flash-Nextのhot prefillを固定差分で測る。

通常のchatベンチは生成文を再tokenizeするため、warm側の新規tokenが16〜274で
揺れる。この道具は`/v1/completions`へtoken ID列を直接送り、1回目を1 token
生成で止める。FlashSpecRunnerはその場合promptだけをsessionへpublishするので、
次のrequestの差分を0/16/64/256 tokenへ正確に固定できる。

`pure_append`は共通のcold baseへ指定数の合成token-IDを足す。`synthetic_tail_rewrite`
は共通のcold baseの末尾を合成token-IDで既定8 token書き換えてから指定数を足し、
chat template/BPE境界が変わる場合のcheckpoint復元を再現する（実際の再tokenizeではない）。
各warm系列は同じbaseから独立に構築し、非空suffixの先頭へ系列ごとに異なる合成token-ID
を置くため、直前のsuffixを誤って選択しない。suffix 0以外の各測定前にはbaseを
republishする未計測のreset requestを送り、直前の枝のsession状態を持ち越さない。
出力には実際のLCP/reused/new、pool bytes、MLX memoryを残す。
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


def _suffix_tokens(suffix_tokens: int, palette: list[int], branch_id: int) -> list[int]:
    """独立したsuffix用に、枝ごとに異なる合成token-ID列を作る。"""

    if suffix_tokens < 0:
        raise ValueError("suffix長は0以上が必要")
    if suffix_tokens == 0:
        return []
    if not palette:
        raise ValueError("suffix用の通常tokenが無い")
    markers = list(dict.fromkeys(palette))
    if branch_id < 0 or branch_id >= len(markers):
        raise ValueError("独立suffix枝用の異なる通常tokenが足りない")
    # branch 0 は従来の並びを保ち、branch 1 以降だけ先頭markerを変える。
    return [markers[branch_id]] + [
        palette[i % len(palette)] for i in range(1, suffix_tokens)
    ]


def _next_prompt(
    current: list[int], suffix_tokens: int, palette: list[int],
    mode: str, rewrite_tail: int, branch_id: int = 0,
) -> tuple[list[int], int]:
    suffix = _suffix_tokens(suffix_tokens, palette, branch_id)
    if mode == "pure_append":
        prompt = current + suffix
        expected_lcp = len(current)
        if _lcp(current, prompt) != expected_lcp:
            raise AssertionError("pure append promptのLCPが意図した位置と違う")
        return prompt, expected_lcp
    if mode != "synthetic_tail_rewrite":
        raise ValueError(f"未知のmode: {mode}")
    rewritten = _replacement_tail(current, rewrite_tail, palette)
    prompt = current[:-rewrite_tail] + rewritten + suffix
    expected_lcp = len(current) - rewrite_tail
    if _lcp(current, prompt) != expected_lcp:
        raise AssertionError("synthetic tail rewrite promptのLCPが意図した位置と違う")
    return prompt, expected_lcp


def _expected_selection(
    prompt: list[int], expected_lcp: int, reused_tokens: int
) -> dict[str, int]:
    """LCPとusageの再利用数からselectionのtoken数を求める。"""

    if not 0 <= expected_lcp <= len(prompt):
        raise ValueError("expected_lcpはprompt内でなければならない")
    if not 0 <= reused_tokens <= len(prompt):
        raise ValueError("reused_tokensはprompt内でなければならない")
    return {
        "lcp": expected_lcp,
        "reused_tokens": reused_tokens,
        "new_tokens": len(prompt) - reused_tokens,
    }


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


def _input_tps(tokens: int, ttft_s: float) -> float:
    """TTFTに含まれる入力処理の粗い tok/s。

    first-tokenの費用も含むのでkernel単体のprefill tok/sではない。
    HTTPから見えるcold/warmを同じ定義で継続比較するための値。
    """
    return tokens / ttft_s if ttft_s > 0 else 0.0


def _flash_recurrent_state_bytes(text_config: dict) -> int:
    """Flash-Nextのlive GDN/PLE/ngram state 1組の理論byte数。"""

    layer_types = text_config["layer_types"]
    linear_layers = sum(kind == "linear_attention" for kind in layer_types)
    bf16_bytes = 2
    fp32_bytes = 4
    int32_bytes = 4
    n_key_heads = int(text_config["linear_num_key_heads"])
    n_value_heads = int(text_config["linear_num_value_heads"])
    key_dim = int(text_config["linear_key_head_dim"])
    value_dim = int(text_config["linear_value_head_dim"])
    conv_width = int(text_config["linear_conv_kernel_dim"]) - 1
    conv_dim = 2 * n_key_heads * key_dim + n_value_heads * value_dim
    per_linear = (
        conv_width * conv_dim * bf16_bytes
        + n_value_heads * value_dim * key_dim * fp32_bytes
    )

    ple_ids = text_config.get("ple_layer_ids") or []
    ple_width = (
        (int(text_config.get("ple_conv_kernel_size", 4)) - 1)
        * int(text_config.get("ngram_size", 3))
    )
    ple_state = (
        len(ple_ids)
        * ple_width
        * int(text_config.get("hc_count", 4))
        * int(text_config["hidden_size"])
        * bf16_bytes
    )
    # n-gram文脈は最初のPLE layerのcacheへ1組だけ保持される。
    ngram_state = (
        (int(text_config.get("ngram_size", 3)) - 1) * int32_bytes
        if ple_ids else 0
    )
    return linear_layers * per_linear + ple_state + ngram_state


def _flash_expected_core_components(
    text_config: dict, processed_tokens: int, checkpoint_count: int
) -> dict[str, int]:
    """statusのsession/indexer/checkpointと同じ範囲の理論allocated bytes。"""

    if processed_tokens < 0 or checkpoint_count < 0:
        raise ValueError("token/checkpoint数は0以上が必要")
    layer_types = text_config["layer_types"]
    full_layers = sum(kind == "full_attention" for kind in layer_types)
    capacity = (
        ((processed_tokens + 255) // 256) * 256 if processed_tokens else 0
    )
    bf16_bytes = 2
    kv_heads = int(text_config["num_key_value_heads"])
    head_dim = int(text_config["head_dim"])
    attention_kv = (
        full_layers * capacity * kv_heads * (head_dim + head_dim) * bf16_bytes
    )

    indexer_heads = int(text_config.get("indexer_kv_heads", 1))
    indexer_dim = int(text_config["indexer_head_dim"])
    raw_indexer = (
        full_layers * capacity * indexer_heads * indexer_dim * bf16_bytes
    )
    compress_ratio = int(text_config.get("indexer_compress_ratio", 4))
    indexer_budget = int(text_config.get("indexer_budget", 2048))
    pooled_blocks = (
        processed_tokens // compress_ratio if processed_tokens > indexer_budget else 0
    )
    pooled_indexer = (
        full_layers * pooled_blocks * indexer_heads * indexer_dim * bf16_bytes
    )

    recurrent = _flash_recurrent_state_bytes(text_config)
    return {
        "session_cache": attention_kv + recurrent,
        "indexer": raw_indexer + pooled_indexer,
        # 最終checkpointはlive cacheと同じ配列を指す。telemetryもidentityで
        # 重複排除するため、追加割当はそれより古いentryだけ。
        "checkpoints": max(checkpoint_count - 1, 0) * recurrent,
    }


def _byte_acceptance(
    status: dict, text_config: dict, tolerance_pct: float
) -> dict:
    """実配列と理論capacity modelを同じcore範囲で比較する。"""

    telemetry = status.get("session_telemetry") or {}
    unknown = telemetry.get("pool_unknown_sessions")
    actual_all = telemetry.get("pool_allocated_bytes_by_component")
    if unknown != 0 or not isinstance(actual_all, dict):
        raise RuntimeError(
            f"pool byte内訳が確定していない: unknown={unknown} components={actual_all}"
        )
    processed = telemetry.get("pool_processed_tokens")
    checkpoints = telemetry.get("pool_checkpoint_count")
    if not isinstance(processed, int) or not isinstance(checkpoints, int):
        raise RuntimeError(
            f"pool token/checkpoint数が無い: processed={processed} checkpoints={checkpoints}"
        )
    expected = _flash_expected_core_components(text_config, processed, checkpoints)
    scope = tuple(expected)
    actual = {name: int(actual_all.get(name, 0)) for name in scope}
    actual_total = sum(actual.values())
    expected_total = sum(expected.values())
    error_pct = (
        abs(actual_total - expected_total) / expected_total * 100
        if expected_total else (0.0 if actual_total == 0 else float("inf"))
    )
    result = {
        "scope": list(scope),
        "excluded_components": ["mtp_cache", "h_last", "tail"],
        "processed_tokens": processed,
        "checkpoint_count": checkpoints,
        "actual_components": actual,
        "expected_components": expected,
        "actual_bytes": actual_total,
        "expected_bytes": expected_total,
        "absolute_error_bytes": abs(actual_total - expected_total),
        "error_pct": error_pct,
        "tolerance_pct": tolerance_pct,
        "passes": error_pct <= tolerance_pct,
    }
    if not result["passes"]:
        raise RuntimeError(f"pool byte理論値との差が許容外: {result}")
    return result


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
    parser.add_argument("--byte-tolerance-pct", type=float, default=5.0)
    parser.add_argument("--server-log", default="scratchpad/log-hot-prefill-bench.txt")
    parser.add_argument("--out", default="bench/results/hot-prefill-bench.json")
    parser.add_argument("--turbo-extra", default=None)
    args = parser.parse_args()

    ctxs = [int(value) for value in args.ctxs.split(",")]
    suffixes = [int(value) for value in args.suffixes.split(",")]
    if args.reps < 1 or any(value < 0 for value in ctxs + suffixes):
        raise SystemExit("ctx/suffixは0以上、repsは1以上が必要")
    if args.byte_tolerance_pct < 0:
        raise SystemExit("byte toleranceは0以上が必要")
    if suffixes != sorted(suffixes) or not suffixes or suffixes[0] != 0:
        raise SystemExit("suffixesは0から始まる昇順にする")

    from transformers import AutoTokenizer

    model_path = os.path.expanduser(args.model)
    model_config = json.loads((Path(model_path) / "config.json").read_text())
    if model_config.get("model_type") != "qwen4_exp":
        raise SystemExit("byte capacity modelはqwen4_exp専用")
    text_config = model_config.get("text_config") or {}
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pool_ids = list(tokenizer.encode(text_pool(), add_special_tokens=False))
    short_ids = list(tokenizer.encode(SHORT, add_special_tokens=False))
    special = set(tokenizer.all_special_ids)
    vocab_size = len(tokenizer)
    palette = [token for token in pool_ids[-1024:] if token not in special]
    if len(set(palette)) < 2:
        raise SystemExit("suffix/rewrite用の通常tokenが足りない")
    non_empty_suffixes = sum(value > 0 for value in suffixes)
    if len(set(palette)) < non_empty_suffixes:
        raise SystemExit("独立suffix枝用の異なる通常tokenが足りない")

    argv = [sys.executable, "-m", "mlxturbo.server", "--model", model_path,
            "--host", "127.0.0.1", "--port", str(args.port)]
    if args.ngram:
        argv += ["--ngram", os.path.expanduser(args.ngram)]
    if args.mtp:
        argv += ["--mtp", os.path.expanduser(args.mtp)]
    if args.turbo_extra:
        argv += shlex.split(args.turbo_extra)
    argv += ["--max-sessions", "1", "--log-level", "debug"]

    cold_rows = []
    rows = []
    case_id = 0
    reset_request_count = 0
    total_http_request_count = 0
    started = time.time()

    def completion(prompt_ids: list[int]) -> dict:
        nonlocal total_http_request_count
        total_http_request_count += 1
        return _stream_completion(args.port, mid, prompt_ids)

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
        completion(warm_ids)
        log("compile warmup完了")

        for ctx in ctxs:
            for rep in range(args.reps):
                for mode in ("pure_append", "synthetic_tail_rewrite"):
                    case_id += 1
                    base = _base_prompt(
                        pool_ids, short_ids, ctx, case_id, vocab_size, special
                    )
                    cold = completion(base)
                    cold_status = _get_json(status_url)
                    cold_selection = _selection(cold_status)
                    cold_telemetry = cold_status.get("session_telemetry") or {}
                    expected_cold = _expected_selection(base, 0, 0)
                    if (
                        cold_selection["match_kind"] != "miss"
                        or any(
                            cold_selection.get(key) != value
                            for key, value in expected_cold.items()
                        )
                        or cold["cached_tokens"] != expected_cold["reused_tokens"]
                        or cold_telemetry.get("pool_unknown_sessions") != 0
                    ):
                        raise RuntimeError(
                            f"coldがcache hitした: ctx={ctx} rep={rep} mode={mode} "
                            f"selection={cold_selection} telemetry={cold_telemetry} "
                            f"usage={cold['usage']}"
                        )
                    cold_row = {
                        "ctx": ctx,
                        "rep": rep,
                        "mode": mode,
                        "prompt_tokens": len(base),
                        "ttft_s": cold["ttft_s"],
                        "wall_s": cold["wall_s"],
                        "input_tokens_per_ttft_s": _input_tps(
                            len(base), cold["ttft_s"]
                        ),
                        "selection": cold_selection,
                        "pool_allocated_bytes": cold_status["session_telemetry"].get(
                            "pool_allocated_bytes"
                        ),
                        "pool_known_allocated_bytes": cold_status[
                            "session_telemetry"
                        ].get("pool_known_allocated_bytes"),
                        "active_memory_bytes": cold_status.get("active_memory_bytes"),
                        "cache_memory_bytes": cold_status.get("cache_memory_bytes"),
                        "rss_bytes": cold_status.get("rss_bytes"),
                        "byte_acceptance": _byte_acceptance(
                            cold_status, text_config, args.byte_tolerance_pct
                        ),
                    }
                    cold_rows.append(cold_row)
                    log(
                        f"ctx={ctx} rep={rep} {mode} cold "
                        f"TTFT={cold_row['ttft_s']:.3f}s "
                        f"input={cold_row['input_tokens_per_ttft_s']:.1f}tok/s"
                    )

                    branch_id = 0
                    previous_prompt = base
                    for suffix_index, suffix in enumerate(suffixes):
                        if suffix_index > 0:
                            reset = completion(base)
                            reset_status = _get_json(status_url)
                            reset_selection = _selection(reset_status)
                            reset_cached = reset["cached_tokens"]
                            reset_telemetry = reset_status.get("session_telemetry") or {}
                            if reset_cached is None:
                                raise RuntimeError(
                                    f"reset usageにcached_tokensが無い: ctx={ctx} "
                                    f"rep={rep} mode={mode} suffix={suffix}"
                                )
                            expected_reset_lcp = _lcp(previous_prompt, base)
                            expected_reset = _expected_selection(
                                base, expected_reset_lcp, reset_cached
                            )
                            if (
                                any(
                                    reset_selection.get(key) != value
                                    for key, value in expected_reset.items()
                                )
                                or reset_telemetry.get("pool_unknown_sessions") != 0
                            ):
                                raise RuntimeError(
                                    f"base reset不一致: ctx={ctx} rep={rep} "
                                    f"mode={mode} suffix={suffix} "
                                    f"expected={expected_reset} "
                                    f"telemetry={reset_telemetry} "
                                    f"selected={reset_selection}"
                                )
                            reset_request_count += 1
                        prompt, expected_lcp = _next_prompt(
                            base, suffix, palette, mode, args.rewrite_tail, branch_id
                        )
                        measured = completion(prompt)
                        status = _get_json(status_url)
                        selected = _selection(status)
                        cached = measured["cached_tokens"]
                        if cached is None:
                            raise RuntimeError(
                                f"warm usageにcached_tokensが無い: ctx={ctx} "
                                f"rep={rep} mode={mode} suffix={suffix}"
                            )
                        expected = _expected_selection(prompt, expected_lcp, cached)
                        if (
                            any(
                                selected.get(key) != value
                                for key, value in expected.items()
                            )
                            or cached != expected["reused_tokens"]
                        ):
                            raise RuntimeError(
                                f"warm selection不一致: expected={expected} "
                                f"cached={cached} selected={selected}"
                            )
                        row = {
                            "ctx": ctx,
                            "rep": rep,
                            "mode": mode,
                            "branch_id": branch_id,
                            "requested_suffix_tokens": suffix,
                            "prompt_tokens": len(prompt),
                            "expected_lcp": expected_lcp,
                            "expected_reused_tokens": expected["reused_tokens"],
                            "expected_new_tokens": expected["new_tokens"],
                            "ttft_s": measured["ttft_s"],
                            "wall_s": measured["wall_s"],
                            "cached_tokens": cached,
                            "new_input_tokens_per_ttft_s": _input_tps(
                                selected["new_tokens"], measured["ttft_s"]
                            ),
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
                            "byte_acceptance": _byte_acceptance(
                                status, text_config, args.byte_tolerance_pct
                            ),
                        }
                        rows.append(row)
                        previous_prompt = prompt
                        log(
                            f"ctx={ctx} rep={rep} {mode} suffix={suffix} "
                            f"TTFT={row['ttft_s']:.3f}s reused/new="
                            f"{selected['reused_tokens']}/{selected['new_tokens']}"
                        )
                        if suffix > 0:
                            branch_id += 1

    output = {
        "engine": "mlxturbo",
        "runner": "flash_spec",
        "model": model_path,
        "ctxs": ctxs,
        "suffixes": suffixes,
        "reps": args.reps,
        "rewrite_tail": args.rewrite_tail,
        "byte_tolerance_pct": args.byte_tolerance_pct,
        "byte_model": (
            "qwen4_exp bf16 attention/indexer capacity plus live recurrent state "
            "and retained recurrent checkpoints (latest aliases live state); "
            "MTP, h_last and tail excluded"
        ),
        "mode_descriptions": {
            "pure_append": "independent synthetic token-ID suffix from the common cold base",
            "synthetic_tail_rewrite": (
                "synthetic token-ID tail rewrite plus an independent suffix; "
                "not tokenizer retokenization"
            ),
        },
        "measurement_request_count": len(cold_rows) + len(rows),
        "reset_request_count": reset_request_count,
        "total_http_request_count": total_http_request_count,
        "http_request_count_scope": "completion POSTs, including compile warmup; status polls excluded",
        "max_sessions": 1,
        "cold_rows": cold_rows,
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
