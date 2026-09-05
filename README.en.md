# mlxturbo

English | [日本語](README.md)

A local inference engine for Apple Silicon (MLX), plus an HTTP server compatible with the OpenAI, Anthropic, and Responses APIs.

**Only two specific architectures are fast.** Every other model runs at the same speed as plain [mlx-lm](https://github.com/ml-explore/mlx-lm) (see the compatibility table below). This is not a "general-purpose acceleration runtime."

If you're coming from Ollama or LM Studio, the first difference you'll hit: **one process holds exactly one model.** Switching models means restarting the process (hosting multiple models at once from a single server is not supported). See "Constraints" below for details.

## What this is

- An engine that loads a model into one long-lived process and speeds up decode with speculative decoding (self-speculation: the model's own MTP head plus context-based suffix lookup)
- An HTTP server (`mlxturbo-serve`) that exposes that engine over both the OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`) and the Anthropic-compatible API (`/v1/messages`)
- Unsupported models still work (fallback), but in that case you get plain, unspeculated mlx-lm speed

## Compatibility table (the honest version)

| Route | Target | Speculation | Measured speedup (vs. mlx-lm, see reproduction commands below) |
|---|---|---|---|
| `flash_spec` | Qwen3.8-Flash-Next (`qwen4_exp` architecture) + MTP sidecar | Yes (MTP depth 1 + fused hyper-connections kernel) | ~1.25x-1.39x (depends on task type, see below) |
| `spec` | Models satisfying the `qwen3_5` contract (e.g. the Qwen3.8-27B family) | Yes (MTP chain + suffix-lookup hybrid) | 1.3x-2.2x (depends on prompt content, see below) |
| `fallback` | Everything else | No | 1.0x (same speed as plain mlx-lm) |

You can see which route was selected in the startup log and in `GET /health`'s `runner` / `fallback_reason` fields. If you want to guard against silently falling back when speculation should be working, start with `--require-runner flash_spec` (or `spec`): the server refuses to start at all if the conditions aren't met. Details in [`docs/SERVER.md`](docs/SERVER.md).

## Conditions for speculation to kick in

- `flash_spec` requires the model's `model_type` to be `qwen4_exp`, and MTP weights to be found (explicitly via `--mtp`, bundled in the main shards, or auto-discovered as a sidecar). If MTP isn't found, `flash_spec` does not degrade to a non-speculative mode — the route simply doesn't come up at all. Measurements are in [`docs/MTP-FLASH.md`](docs/MTP-FLASH.md)
- `spec` requires the model to match the `qwen3_5` contract (the layer structure and attention shape `SpecEngine` expects). If it doesn't match, the server falls back to `fallback`
- On both routes, `temp=0` (greedy) runs under exact-match verification against the true output distribution, and `temp>0` uses rejection sampling that keeps the distribution exactly identical too. However, sampling parameters that alter the distribution — `top_p<1.0` or `repetition_penalty!=1.0` — get a 400. The speculative block-verification assumes exact sampling from the target distribution, so distorting that distribution with sampling parameters breaks the verification's premise. This restriction does not apply to the `fallback` route

## Installation

Requires an Apple Silicon Mac (macOS, with MLX/Metal available).

```
git clone <this repository>
cd mlxturbo
uv sync
```

## Getting a model

- `fallback` / `spec` routes: point directly at a normal mlx-lm-compatible checkpoint (a Hugging Face repo ID or a local path)
- `flash_spec` route (Qwen3.8-Flash-Next): requires converting from the original checkpoint and extracting MTP. See [`docs/MTP-FLASH.md`](docs/MTP-FLASH.md) and `mlxturbo/convert_flash.py --help` (subcommands `estimate` / `extract-mtp` / `convert`). The `qwen4_exp` architecture isn't in mlx-lm upstream, but importing `mlxturbo` resolves it automatically — nothing is written to your mlx-lm package

## Running it

Interactive CLI (`spec` route, aimed at the 27B family):

```
uv run mlxturbo --model <path-or-repo-id> --prompt "hello"
```

HTTP server:

```
uv run mlxturbo-serve --model <path-or-repo-id> --served-model-name mymodel --port 8000
```

Connection methods, the full option list, API key auth, and connection examples for opencode / Codex CLI / Claude Code / Chatbox are all in [`docs/SERVER.md`](docs/SERVER.md).

## Constraints

- **Text only. No images, audio or video.** The original checkpoint is a VLM, but this conversion carries no vision tower (zero vision tensors in the converted weights). Requests containing non-text blocks (`image_url`, `image`, `input_audio`, ...) are rejected with a 400. See §1 of [`docs/BACKLOG.md`](docs/BACKLOG.md) for what adding it would take
- **One process, one model.** A model is loaded exactly once at startup and stays resident; switching models means starting a different process
- **Requests are serialized by default.** `--max-batch N` enables continuous batching, but the default is 1 (serial). Even when enabled, speculative routes are excluded — only requests running on `FallbackRunner` get pooled. Once the queue exceeds `--max-queue`, new requests get a 503
- **Non-identity sampling parameters downgrade that one request to non-speculative** on the `spec` / `flash_spec` routes, because speculative block-verification assumes exact distribution matching. The downgrade is reported in the response's `downgrade_reason` and in the server log
- **logprobs only on non-speculative routes.** Requesting them on a speculative route downgrades the request the same way. Combining them with streaming returns a 400
- For the full constraint list (the scope of determinism, context-length guards, etc.), see the "Constraints" section of [`docs/SERVER.md`](docs/SERVER.md)

## Measured numbers and how to reproduce them

Raw result JSON files and the `docs/` write-ups linked below live in the GitHub repository, not in the PyPI package (the wheel/sdist ship only the `mlxturbo` library) — follow the links from [github.com/akakeishin/mlxturbo](https://github.com/akakeishin/mlxturbo) if you installed via `pip`.

All of the following were measured on an M3 Max 128GB / macOS 26.4 / mlx 0.32.2. Numbers shift across hardware generations (see "judgments that expire when the hardware generation changes" in [`docs/research/ROOFLINE-2026-08-26.md`](docs/research/ROOFLINE-2026-08-26.md)).

### `spec` route (Qwen3.8-27B-4bit, greedy, 512 tok)

| Condition | decode tok/s | vs. mlx-lm |
|---|---|---|
| Plain mlx-lm (fallback-equivalent) | 21-23 | 1.0x |
| Self-speculation, sustained hard content (code) | 31.9 | 1.49x |
| Self-speculation, sustained hard content (prose) | 28.3 | 1.32x |

Reproduce: `uv run mlxturbo --model <model> --prompt "<prompt>"` (it prints `decode tok/s` automatically after generation). Comparison against plain mlx-lm: `uv run python bench/baseline.py <model-id>`.

### `flash_spec` route (Qwen3.8-Flash-Next, v-l recipe + MTP 4bit, greedy, 48 tok)

| Prompt | Acceptance rate | Greedy tok/s | Speculative tok/s | Speedup |
|---|---|---|---|---|
| Prose (English 0.720 ± 0.046 / Japanese 0.682 ± 0.047) | ~0.70 | ~1.39x |
| Code | 0.564 ± 0.068 | ~1.25x |

Acceptance rate also varies by task type (code / prose / step-by-step instructions, etc). Details and reproduction commands are in the "tools" section of [`docs/MTP-FLASH.md`](docs/MTP-FLASH.md) (`tools/spec_flash_bench.py` and others). See also [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for exactly which result file each number in this README traces back to, including one row whose source file we could not track down and one table whose own source document later flags it as not statistically significant.

## Other documentation

- [`docs/README.md`](docs/README.md) — index of all documentation
- [`docs/SERVER.md`](docs/SERVER.md) — server startup, options, connection methods, constraints
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — running a public instance: reverse proxy, `/health` monitoring, log reading, launchd
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — which reproduction command and result file backs each published number
- [`docs/MTP-FLASH.md`](docs/MTP-FLASH.md) — design of Flash-Next's MTP speculative decoding
- [`docs/BACKLOG.md`](docs/BACKLOG.md) — things worth doing that haven't been started yet
- [`docs/RELEASE.md`](docs/RELEASE.md) — what to do before publishing

Everything under `docs/` besides this README's translation is Japanese-only for now.

## License

mlxturbo is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for copyright and license notices covering incorporated third-party code.
