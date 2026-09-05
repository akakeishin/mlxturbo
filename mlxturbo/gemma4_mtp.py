"""B=1 Gemma 4 assistant-backed speculative decoding.

Gemma 4's assistant is not an ordinary ``mlx_lm`` draft model.  It consumes
the target model's last sliding/full attention K/V banks and feeds its own
post-projection hidden state back into the next draft step.  This module keeps
that contract local to the Gemma family; other models continue to use the
existing runners unchanged.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .runner import FallbackSession, PREFILL_STEP_SIZE
from .spec import restore_untrimmable_caches, snapshot_untrimmable_caches
from .kernels.dispatch import dispatch_scope


GEMMA4_ASSISTANT_KIND = "gemma4_assistant_spec"
DEFAULT_DRAFT_BLOCK_SIZE = 4
ALLOWED_DRAFT_BLOCK_SIZES = frozenset((2, 4, 6, 8))


class Gemma4AssistantSession(FallbackSession):
    """Target prompt cache used by the B=1 Gemma assistant runner."""


class _AssistantInner(nn.Module):
    """Gemma text layers without the target-only ``previous_kvs`` wiring."""

    def __init__(self, config):
        super().__init__()
        from mlx_lm.models.gemma4_text import DecoderLayer

        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        self.layers = [
            DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def _read_config(path: str | Path) -> dict:
    with (Path(path).expanduser() / "config.json").open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("config.json must contain an object")
    return value


def _target_inner(model):
    language_model = getattr(model, "language_model", None)
    inner = getattr(language_model, "model", None)
    if inner is None or not hasattr(inner, "layers") or not hasattr(inner, "embed_tokens"):
        raise ValueError("target is not an mlx-lm Gemma 4 text model")
    return language_model, inner


def _validate_pair(target_model, assistant_config: dict) -> dict:
    language_model, target_inner = _target_inner(target_model)
    target_args = getattr(target_model, "args", None)
    target_type = getattr(target_args, "model_type", None)
    if target_type != "gemma4":
        raise ValueError(f"target model_type must be gemma4, got {target_type!r}")
    if assistant_config.get("model_type") != "gemma4_assistant":
        raise ValueError(
            "assistant model_type must be gemma4_assistant, "
            f"got {assistant_config.get('model_type')!r}"
        )
    text = assistant_config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("assistant text_config is missing")
    if int(assistant_config.get("backbone_hidden_size", -1)) != int(
        target_inner.config.hidden_size
    ):
        raise ValueError("assistant backbone_hidden_size does not match target")
    if int(target_inner.config.hidden_size) != 5376 or int(
        target_inner.config.num_hidden_layers
    ) != 60:
        raise ValueError("target is not the Gemma 4 dense 31B shape")
    if int(getattr(target_inner.config, "num_kv_shared_layers", 0) or 0) != 0:
        raise ValueError("target must own its Gemma K/V banks")
    if (
        not bool(getattr(target_inner.config, "attention_k_eq_v", False))
        or int(target_inner.config.num_attention_heads) != 32
        or int(target_inner.config.num_global_key_value_heads) != 4
        or int(target_inner.config.head_dim) != 256
        or int(target_inner.config.global_head_dim) != 512
        or int(target_inner.config.sliding_window) != 1024
    ):
        raise ValueError("target attention shape is not Gemma 4 dense 31B")
    if int(text.get("vocab_size", -1)) != int(target_inner.config.vocab_size):
        raise ValueError("assistant vocab_size does not match target")
    if (
        not bool(text.get("attention_k_eq_v", False))
        or int(text.get("num_key_value_heads", -1)) != 16
        or text.get("rope_parameters") != target_inner.config.rope_parameters
        or not bool(assistant_config.get("tie_word_embeddings", False))
        or not bool(getattr(language_model, "tie_word_embeddings", False))
    ):
        raise ValueError("assistant attention/RoPE/embedding contract does not match target")
    if int(text.get("hidden_size", -1)) <= 0 or int(text.get("num_hidden_layers", -1)) <= 0:
        raise ValueError("assistant text_config has invalid dimensions")
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != int(text["num_hidden_layers"]):
        raise ValueError("assistant layer_types does not match num_hidden_layers")
    if any(kind not in ("sliding_attention", "full_attention") for kind in layer_types):
        raise ValueError("assistant layer_types contains an unsupported attention kind")
    target_types = set(getattr(target_inner.config, "layer_types", ()))
    if not set(layer_types).issubset(target_types):
        raise ValueError("assistant attention kinds are absent from target")
    if int(text.get("num_kv_shared_layers", 0) or 0) != int(text["num_hidden_layers"]):
        raise ValueError("Gemma assistant must use shared K/V in every assistant layer")
    return text


class Gemma4AssistantModel(nn.Module):
    """The four-layer assistant loaded from ``gemma4_assistant`` weights."""

    def __init__(self, config: dict):
        super().__init__()
        from mlx_lm.models.gemma4_text import ModelArgs

        text = config["text_config"]
        text_args = ModelArgs.from_dict(text)
        self.config = config
        self.model = _AssistantInner(text_args)
        self.pre_projection = nn.Linear(
            2 * int(config["backbone_hidden_size"]), text_args.hidden_size, bias=False
        )
        self.post_projection = nn.Linear(
            text_args.hidden_size, int(config["backbone_hidden_size"]), bias=False
        )
        self._target_embed = None
        self._target_embed_scale = 1.0

    def bind(self, target_model) -> None:
        _, target_inner = _target_inner(target_model)
        self._target_embed = target_inner.embed_tokens
        self._target_embed_scale = float(getattr(target_inner, "embed_scale", 1.0))

    def forward_one(self, inputs_embeds: mx.array, shared_kv: dict, position: int, valid_len: int):
        text_cfg = self.model.config
        h = self.pre_projection(inputs_embeds)
        query_len = h.shape[1]
        masks = _assistant_masks(
            shared_kv,
            query_len=query_len,
            query_offset=position,
            window=int(text_cfg.sliding_window),
            valid_len=valid_len,
            dtype=h.dtype,
        )
        offset = mx.array(position)
        for layer in self.model.layers:
            h, _, _ = layer(
                h,
                mask=masks[layer.layer_type],
                cache=None,
                shared_kv=shared_kv[layer.layer_type],
                offset=offset,
            )
        h = self.model.norm(h)
        return self.post_projection(h), h

    def draft_block(
        self,
        last_token: int,
        hidden: mx.array,
        shared_kv: dict,
        position: int,
        valid_len: int,
        block_size: int,
        sampler,
        greedy: bool,
        processors,
        history: list[int],
    ) -> list[int]:
        if self._target_embed is None:
            raise RuntimeError("assistant is not bound to the target embedding")
        one_sync = (
            greedy
            and not processors
            and os.environ.get("MLXTURBO_GEMMA_GREEDY_ONE_SYNC", "1") != "0"
        )
        token = int(last_token)
        token_array = None
        h_prev = hidden
        out: list[int] = []
        out_arrays = []
        for _ in range(block_size - 1):
            ids = (
                token_array
                if one_sync and token_array is not None
                else mx.array([[token]], dtype=mx.uint32)
            )
            token_embed = self._target_embed(ids)
            token_embed = token_embed * self._target_embed_scale
            inputs = mx.concatenate([token_embed.astype(h_prev.dtype), h_prev], axis=-1)
            h_prev, h_norm = self.forward_one(inputs, shared_kv, position, valid_len)
            logits = self.model.embed_tokens.as_linear(h_norm)
            if one_sync:
                token_array = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
                out_arrays.append(token_array)
                continue
            logits = _apply_processors(logits[:, -1, :], processors, history + out)
            token = _sample(logits, sampler, greedy=greedy)
            out.append(token)
        if one_sync and out_arrays:
            packed = mx.concatenate(out_arrays, axis=1)
            mx.eval(packed)
            return [int(value) for value in packed.reshape(-1).tolist()]
        return out


def _assistant_masks(shared_kv, *, query_len, query_offset, window, valid_len, dtype):
    # The production drafter predicts one token from the target's last valid
    # position.  Its shared banks contain no padded/future slots and the
    # sliding bank is already capped to the window, so both additive masks
    # would contain only zeroes.  Keep the explicit construction as a safe
    # fallback for any future multi-token or over-allocated cache caller.
    if query_len == 1 and query_offset == valid_len - 1:
        all_visible = all(
            int(keys.shape[-2]) <= valid_len
            and (
                kind == "full_attention"
                or (kind == "sliding_attention" and int(keys.shape[-2]) <= window)
            )
            for kind, (keys, _) in shared_kv.items()
        )
        if all_visible:
            return {kind: None for kind in shared_kv}
    masks = {}
    for kind, (keys, _) in shared_kv.items():
        kv_len = int(keys.shape[-2])
        if kind == "full_attention":
            if kv_len <= valid_len:
                masks[kind] = None
            else:
                inside = mx.arange(kv_len) < valid_len
                masks[kind] = mx.where(
                    inside,
                    mx.array(0.0, dtype=dtype),
                    mx.array(-mx.inf, dtype=dtype),
                )[None, None, None, :]
            continue
        # The shared sliding bank is in temporal order and contains at most
        # the target's sliding window.  The assistant attends bidirectionally
        # around the target's current position, as in the reference drafter.
        key_start = max(valid_len - kv_len, 0)
        q = mx.arange(query_offset, query_offset + query_len)[:, None]
        k = mx.arange(key_start, key_start + kv_len)[None, :]
        inside = (q - k > -window) & (q - k < window) & (k < valid_len)
        masks[kind] = mx.where(
            inside,
            mx.array(0.0, dtype=dtype),
            mx.array(-mx.inf, dtype=dtype),
        )[None, None, :, :]
    return masks


def _sample(logits, sampler, *, greedy: bool = False) -> int:
    # Normalizing 262k logits cannot change argmax.  Avoid one full-vocabulary
    # logsumexp for every draft and verifier row on the temperature-zero path.
    token = mx.argmax(logits, axis=-1) if greedy else sampler(
        logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    )
    mx.eval(token)
    return int(token.reshape(-1).item())


def _copy_rng_state():
    return [mx.array(value) for value in mx.random.state]


def _restore_rng_state(state) -> None:
    for index, value in enumerate(state):
        mx.random.state[index][:] = value


def _apply_processors(logits, processors, history):
    if not processors:
        return logits
    tokens = mx.array(history, dtype=mx.uint32)
    for processor in processors:
        logits = processor(tokens, logits)
    return logits


def _cache_offset(caches) -> int:
    for cache in caches:
        offset = getattr(cache, "offset", None)
        if isinstance(offset, int):
            return offset
        if isinstance(offset, mx.array):
            return int(offset.max().item())
    return 0


def _snapshot_prompt_boundary(caches):
    """Capture one transient boundary so generated tokens need not be retained.

    Gemma's sliding caches cannot be trimmed after their ring has wrapped. A
    deep snapshot is therefore needed for those layers; ordinary full-attention
    caches only need their logical offsets recorded. The snapshot is restored
    before the request publishes its session, so it is never retained as a
    second copy in the session pool.
    """

    snapshot = snapshot_untrimmable_caches(caches, deep=True)
    snap_indices = {entry[0] for entry in snapshot}
    offsets = {}
    for index, cache in enumerate(caches):
        if index in snap_indices:
            continue
        offset = getattr(cache, "offset", None)
        if isinstance(offset, mx.array):
            offset = int(offset.max().item())
        if not isinstance(offset, int) or not callable(getattr(cache, "trim", None)):
            raise RuntimeError(
                f"Gemma target cache {type(cache).__name__} has no restorable boundary"
            )
        offsets[index] = offset
    return offsets, snapshot


def _restore_prompt_boundary(caches, boundary) -> None:
    offsets, snapshot = boundary
    trims = {}
    for index, saved_offset in offsets.items():
        cache = caches[index]
        offset = getattr(cache, "offset", None)
        if isinstance(offset, mx.array):
            offset = int(offset.max().item())
        if not isinstance(offset, int) or offset < saved_offset:
            raise RuntimeError(
                f"Gemma target cache {type(cache).__name__} moved before its prompt boundary"
            )
        trims[index] = offset - saved_offset

    for index, count in trims.items():
        if caches[index].trim(count) != count:
            raise RuntimeError(
                f"Gemma target cache {type(caches[index]).__name__} could not restore prompt boundary"
            )
    restore_untrimmable_caches(caches, snapshot)


def _temporal_state(cache):
    state = getattr(cache, "state", None)
    if state is None or len(state) < 2:
        return None
    keys, values = state[:2]
    temporal = getattr(cache, "_temporal_order", None)
    if callable(temporal):
        keys = temporal(keys)
        values = temporal(values)
    max_size = getattr(cache, "max_size", None)
    if isinstance(max_size, int) and keys.shape[-2] > max_size:
        keys = keys[..., -max_size:, :]
        values = values[..., -max_size:, :]
    return keys, values


def _shared_kv_from_cache(inner, caches) -> dict:
    selected = {}
    # Assistant が読むのは各 attention 種別の最後の1バンクだけ。先頭から
    # 全60層を走査すると、途中の値は直後に上書きされるのに、rotating cacheの
    # `_temporal_order`（concatを含みうる）を毎round組み立ててしまう。
    # 逆順で最初に見つかった2種だけを取り、従来と同じ最終バンクを返す。
    for layer, cache in zip(reversed(inner.layers), reversed(caches)):
        if layer.layer_type in selected:
            continue
        state = _temporal_state(cache)
        if state is not None:
            selected[layer.layer_type] = state
            if "sliding_attention" in selected and "full_attention" in selected:
                break
    if "sliding_attention" not in selected or "full_attention" not in selected:
        raise RuntimeError("target cache did not produce both Gemma attention K/V banks")
    return selected


def _target_logits(language_model, inner, hidden):
    logits = inner.embed_tokens.as_linear(hidden)
    softcap = getattr(language_model, "final_logit_softcapping", None)
    if softcap is not None:
        logits = mx.tanh(logits / softcap) * softcap
    return logits


def _target_forward(
    target_model, tokens, caches, *, return_logits=True, return_shared_kv=True
):
    language_model, inner = _target_inner(target_model)
    h = inner.embed_tokens(tokens) * inner.embed_scale
    masks = inner._make_masks(h, caches)
    intermediates = [(None, None)] * len(inner.layers)
    for idx, (layer, cache, mask, previous) in enumerate(
        zip(inner.layers, caches, masks, inner.previous_kvs)
    ):
        shared, offset = intermediates[previous]
        h, shared, offset = layer(
            h,
            mask=mask,
            cache=cache,
            shared_kv=shared,
            offset=offset,
        )
        intermediates[idx] = (shared, offset)
    hidden = h
    norm_hidden = inner.norm(hidden) if return_logits else None
    logits = _target_logits(language_model, inner, norm_hidden) if return_logits else None
    shared_kv = _shared_kv_from_cache(inner, caches) if return_shared_kv else None
    return hidden, norm_hidden, logits, shared_kv


def _prefill(target_model, tokens, caches):
    """Prefill all but the final token in bounded chunks, then return its logits."""
    ids = list(tokens)
    if not ids:
        raise ValueError("Gemma generation requires at least one prompt token")
    pos = 0
    while len(ids) - pos > 1:
        end = min(pos + PREFILL_STEP_SIZE, len(ids) - 1)
        _target_forward(
            target_model,
            mx.array([ids[pos:end]], dtype=mx.uint32),
            caches,
            return_logits=False,
            return_shared_kv=False,
        )
        mx.eval([c.state for c in caches])
        mx.clear_cache()
        pos = end
    return _target_forward(
        target_model,
        mx.array([[ids[-1]]], dtype=mx.uint32),
        caches,
    )


def _rollback(caches, count: int) -> None:
    if count <= 0:
        return
    for cache in caches:
        state = getattr(cache, "state", None)
        if not isinstance(state, tuple) or len(state) < 2:
            raise RuntimeError(f"Gemma target cache {type(cache).__name__} cannot rollback")
        keys, values = state[:2]
        if keys is None or values is None:
            raise RuntimeError(f"Gemma target cache {type(cache).__name__} is empty")

        # RotatingKVCache.trim() only advertises itself as trimmable before the
        # ring wraps.  Verification can append a multi-token suffix after the
        # wrap, so restore a temporal prefix explicitly and keep its absolute
        # offset.  KVCache has no ring metadata and follows the same path.
        if hasattr(cache, "_temporal_order"):
            keys = cache._temporal_order(cache.keys)
            values = cache._temporal_order(cache.values)
        keep = max(int(keys.shape[-2]) - count, 0)
        keys = keys[..., :keep, :]
        values = values[..., :keep, :]
        offset = getattr(cache, "offset", 0)
        if isinstance(offset, mx.array):
            offset = int(offset.max().item())
        offset = max(int(offset) - count, 0)

        if hasattr(cache, "_idx"):
            max_size = int(getattr(cache, "max_size", keep))
            if keep > max_size:
                keys = keys[..., -max_size:, :]
                values = values[..., -max_size:, :]
            cache.keys = keys
            cache.values = values
            cache.offset = offset
            cache._idx = keys.shape[-2]
        elif hasattr(cache, "offset"):
            # KVCache keeps an over-allocated backing buffer; changing only
            # the logical offset avoids a large reallocation on every reject.
            cache.offset = offset
        else:
            cache.state = (keys, values)


def load_gemma4_assistant(path: str | Path, target_model) -> Gemma4AssistantModel:
    """Load and validate an explicit ``gemma4_assistant`` checkpoint."""
    path = Path(path).expanduser()
    config = _read_config(path)
    text = _validate_pair(target_model, config)
    if (
        int(text["num_hidden_layers"]) != 4
        or int(text["hidden_size"]) != 1024
        or int(text["num_attention_heads"]) != 32
        or int(text["num_global_key_value_heads"]) != 4
        or int(text["head_dim"]) != 256
        or int(text["global_head_dim"]) != 512
        or int(text["sliding_window"]) != 1024
    ):
        raise ValueError("this lane only supports the Gemma 4 31B assistant shape")
    weights_path = path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    model = Gemma4AssistantModel(config)
    model.bind(target_model)
    model.load_weights(list(mx.load(str(weights_path)).items()), strict=True)
    model.eval()
    return model


class Gemma4AssistantRunner:
    """B=1 target-verified assistant speculation for Gemma 4 31B."""

    KIND = GEMMA4_ASSISTANT_KIND
    SUPPORTED_SAMPLING_PARAMS = frozenset(
        {
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "seed",
        }
    )
    SUPPORTS_LOGPROBS = False
    SUPPORTS_TTFT_PHASES = False

    def __init__(self, target_model, tokenizer, assistant, block_size=DEFAULT_DRAFT_BLOCK_SIZE):
        if block_size not in ALLOWED_DRAFT_BLOCK_SIZES:
            raise ValueError(
                f"draft block size must be one of {sorted(ALLOWED_DRAFT_BLOCK_SIZES)}"
            )
        self.model = target_model
        self.tokenizer = tokenizer
        self.assistant = assistant
        self.block_size = block_size
        self.fallback_reason = None

    def generate(
        self,
        prompt_ids,
        max_tokens,
        temp,
        eos_ids,
        on_tokens,
        session,
        top_p=0.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        logit_bias=None,
        seed=None,
        **extra,
    ):
        del extra
        if seed is not None:
            mx.random.seed(seed)
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
        greedy = temp == 0
        processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        t0 = time.perf_counter()
        cache = None
        reused = 0
        if session is not None and session.cache is not None:
            processed = session.processed
            n = min(len(processed), len(prompt_ids))
            while reused < n and processed[reused] == prompt_ids[reused]:
                reused += 1
            if reused == len(processed) and reused < len(prompt_ids):
                cache = session.cache
                session.invalidate()
            else:
                reused = 0
        if cache is None:
            cache = self.model.make_cache()
        if max_tokens <= 0:
            return {
                "tokens": [],
                "ttft_s": 0.0,
                "decode_tps": 0.0,
                "prefill_reused": reused,
                "prefill_new": len(prompt_ids) - reused,
                "tokens_per_step": 0.0,
            }

        _, norm_hidden, logits, shared_kv = _prefill(
            self.model, prompt_ids[reused:], cache
        )
        first_logits = _apply_processors(logits[:, -1, :], processors, prompt_ids[: len(prompt_ids)])
        bonus = _sample(first_logits, sampler, greedy=greedy)
        target_rng_state = _copy_rng_state()
        draft_rng_state = _copy_rng_state()
        tokens = [bonus]
        if on_tokens:
            on_tokens([bonus], self.tokenizer.decode([bonus]))
        ttft = time.perf_counter() - t0
        if bonus in eos_ids or max_tokens == 1:
            if session is not None:
                # The cache contains exactly the prompt here; keep that useful
                # boundary even when the first sampled token is EOS.
                session.publish(cache, list(prompt_ids))
            return {
                "tokens": tokens,
                "ttft_s": ttft,
                "decode_tps": 0.0,
                "prefill_reused": reused,
                "prefill_new": len(prompt_ids) - reused,
                "tokens_per_step": 0.0,
            }

        prompt_boundary = (
            _snapshot_prompt_boundary(cache) if session is not None else None
        )

        inner_hidden = norm_hidden[:, -1:, :]
        target_inner = _target_inner(self.model)[1]
        hidden = inner_hidden
        pending = bonus
        rounds = 0
        history = list(prompt_ids) + tokens
        while len(tokens) < max_tokens:
            rounds += 1
            draft_count = min(self.block_size - 1, max(max_tokens - len(tokens) - 1, 0))
            _restore_rng_state(draft_rng_state)
            draft = self.assistant.draft_block(
                pending,
                hidden,
                shared_kv,
                _cache_offset(cache) - 1,
                _cache_offset(cache),
                draft_count + 1,
                sampler,
                greedy,
                processors,
                history,
            )
            draft_rng_state = _copy_rng_state()
            _restore_rng_state(target_rng_state)
            verify_input = [pending, *draft]
            # E120 は共通のshape routeとして使う。Gemmaで未計測の他shapeまで
            # broad small-M fallbackへ広げない。
            with dispatch_scope(unlisted_small_m=False):
                raw_hidden, _, verify_logits, _ = _target_forward(
                    self.model,
                    mx.array([verify_input], dtype=mx.uint32),
                    cache,
                    return_shared_kv=False,
                )
            target_tokens = []
            accepted = 0
            emitted = []
            verify_history = list(history)
            one_sync = (
                greedy
                and not processors
                and os.environ.get("MLXTURBO_GEMMA_GREEDY_ONE_SYNC", "1") != "0"
            )
            if one_sync:
                greedy_tokens = mx.argmax(verify_logits, axis=-1)
                mx.eval(raw_hidden, greedy_tokens)
                greedy_values = [
                    int(value) for value in greedy_tokens.reshape(-1).tolist()
                ]
            else:
                mx.eval(raw_hidden, verify_logits)
                greedy_values = []
            for row in range(len(verify_input)):
                if one_sync:
                    target_token = greedy_values[row]
                else:
                    row_logits = _apply_processors(
                        verify_logits[:, row, :], processors, verify_history
                    )
                    target_token = _sample(row_logits, sampler, greedy=greedy)
                target_tokens.append(target_token)
                verify_history.append(target_token)
                if row < len(draft) and target_token == draft[row]:
                    accepted += 1
                    emitted.append(target_token)
                    continue
                emitted.append(target_token)
                break
            target_rng_state = _copy_rng_state()
            consumed = min(accepted + 1, len(verify_input))
            _rollback(cache, len(verify_input) - consumed)
            shared_kv = _shared_kv_from_cache(target_inner, cache)
            hidden = target_inner.norm(raw_hidden[:, accepted : accepted + 1, :])
            pending = target_tokens[accepted]
            emitted = emitted[: max_tokens - len(tokens)]
            eos_seen = False
            for index, token in enumerate(emitted):
                tokens.append(token)
                if on_tokens:
                    on_tokens([token], self.tokenizer.decode([token]))
                if token in eos_ids:
                    eos_seen = True
                    emitted = emitted[: index + 1]
                    break
            history.extend(emitted)
            if eos_seen:
                if session is not None:
                    _restore_prompt_boundary(cache, prompt_boundary)
                    session.publish(cache, list(prompt_ids))
                return {
                    "tokens": tokens,
                    "ttft_s": ttft,
                    "decode_tps": max(len(tokens) - 1, 0)
                    / max(time.perf_counter() - t0 - ttft, 1e-12),
                    "prefill_reused": reused,
                    "prefill_new": len(prompt_ids) - reused,
                    "tokens_per_step": max(len(tokens) - 1, 0) / rounds,
                }
            if len(tokens) >= max_tokens:
                break

        if session is not None:
            _restore_prompt_boundary(cache, prompt_boundary)
            session.publish(cache, list(prompt_ids))
        elapsed = time.perf_counter() - t0
        return {
            "tokens": tokens,
            "ttft_s": ttft,
            "decode_tps": max(len(tokens) - 1, 0) / max(elapsed - ttft, 1e-12),
            "prefill_reused": reused,
            "prefill_new": len(prompt_ids) - reused,
            "tokens_per_step": max(len(tokens) - 1, 0) / rounds if rounds else 0.0,
        }


def build_gemma4_runner(target_model, tokenizer, assistant_path, block_size=None):
    """Build the explicitly requested assistant runner.

    An explicit checkpoint is authoritative: an incompatible or corrupt pair
    is an error, not a reason to silently benchmark the non-assistant runner.
    """
    assistant = load_gemma4_assistant(assistant_path, target_model)
    from .kernels.dispatch import enable as enable_quantized_dispatch

    enable_quantized_dispatch(target_model, active=False)
    size = DEFAULT_DRAFT_BLOCK_SIZE if block_size is None else int(block_size)
    runner = Gemma4AssistantRunner(target_model, tokenizer, assistant, size)
    print(
        f"[mlxturbo] Gemma 4 assistant 投機有効 (B=1, block_size={size}, "
        f"assistant={assistant_path})"
    )
    return runner


__all__ = [
    "ALLOWED_DRAFT_BLOCK_SIZES",
    "DEFAULT_DRAFT_BLOCK_SIZE",
    "GEMMA4_ASSISTANT_KIND",
    "Gemma4AssistantModel",
    "Gemma4AssistantRunner",
    "Gemma4AssistantSession",
    "build_gemma4_runner",
    "load_gemma4_assistant",
]
