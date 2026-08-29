"""Model-agnostic speculative decoding path that uses n-gram lookup (SAM) only.

Kimi K3 review item 12. Speculation in ``mlxturbo.spec.SpecEngine`` is built out of
two parts (a draft from MTP, and a draft from n-gram lookup (SAM)), but the latter
is pure string (token ID sequence) matching that only asks "among the tokens
actually emitted/read so far, has the same run as the prefix of the current
continuation appeared before?" — it never looks at the model's weights or its layer
structure. ``SuffixAutomaton`` in ``mlxturbo/sam.py`` (an existing general-purpose
utility, not rewritten at all by this change) performs that matching in amortized
O(1).

Since ``mlxturbo/spec.py`` is under a do-not-touch constraint (it is the engine
proper, already verified by measurement), the lookup part alone is rebuilt directly
on top of ``SuffixAutomaton`` and written here as an independent runner. The scope
is limited to "models whose KV cache can be ``trim()``ed": a linear state such as
GDN cannot be rewound to a midway position (the same constraint as the
``ChatSession`` docstring in ``mlxturbo/spec.py``), but a model that is attention
(KV only) in every layer can be rewound straightforwardly with
``mlx_lm.models.cache.trim_prompt_cache``.

Only greedy (temperature 0) is supported. temperature > 0 requires a derivation for
"resampling correctly from the verifier-side distribution when the draft misses"
(``mlx_lm.generate.speculative_generate_step``, which
``mlxturbo.runner.DraftSpecRunner`` uses, solves this with a single sampler call on
the verifier side, but that depends on mlx_lm's design in which "the verifier model
itself samples independently of the draft, and if they agree, that sample is used",
and on the lookup-draft side there is nothing corresponding to that "verifier
model" — the draft itself is string matching "outside the model", so the same
construction is not possible). We do not step into it here and leave it
unsupported; when ``temp > 0`` (or when a logits processor that changes the output
even under greedy, such as repetition_penalty, is requested) we delegate to
``FallbackRunner`` on the spot (a demotion inside ``generate()`` — this avoids
adding routing on the server.py side; see the decision at the end of this module
docstring).
"""

from __future__ import annotations

import time

import mlx.core as mx
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

from .runner import FallbackRunner
from .sam import SuffixAutomaton
from .spec import PREFILL_STEP_SIZE


def _prefill(model, cache, y: mx.array, step: int) -> mx.array:
    """Same shape as ``mlx_lm.generate.speculative_generate_step._prefill``
    (an independent implementation written on top of the public API
    ``mlx_lm.models.cache`` — ``mlxturbo/spec.py`` is neither read nor
    referenced). Leaves just the last token un-fed (the caller uses it as the
    first "already decided but not yet fed" pending token)."""

    while y.size > 1:
        n = min(step, y.size - 1)
        model(y[:n][None], cache=cache)
        mx.eval([c.state for c in cache])
        y = y[n:]
    return y


def _needs_logits_processors(
    repetition_penalty: float | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
    logit_bias: dict | None,
) -> bool:
    """Whether the values are identity values (defaults that change neither the
    distribution nor the greedy result). Uses the same set of values as
    ``_IDENTITY_SAMPLING_VALUES`` in server.py (we do not want this module to
    depend on server.py, so the values themselves are duplicated here — if you
    change one, fix both)."""

    if logit_bias:
        return True
    if repetition_penalty not in (None, 0.0, 1.0):
        return True
    if presence_penalty not in (None, 0.0):
        return True
    if frequency_penalty not in (None, 0.0):
        return True
    return False


class LookupSpecRunner:
    """A runner that speculates using n-gram lookup (SAM) only. It never looks
    at the model architecture at all, so it is layered over models that
    ``mlxturbo.runner.build_runner`` judged not to satisfy the spec/flash_spec
    contract (= normally ``FallbackRunner``); see the ``--lookup-spec`` branch
    in ``build_runner``.

    ``SUPPORTED_SAMPLING_PARAMS``: declares all the same keys as
    ``FallbackRunner`` — this class merely picks, internally and based on the
    combination of values, between "n-gram lookup speculation" and "plain
    (non-speculative) generation", and neither path changes the final output
    distribution (the plain path delegates as-is to the ``FallbackRunner``
    instance it holds internally; see ``generate()`` below). Which of the two
    is chosen for a given request is invisible to the caller, and does not need
    to be visible.
    """

    KIND = "lookup_spec"
    SUPPORTED_SAMPLING_PARAMS = FallbackRunner.SUPPORTED_SAMPLING_PARAMS

    def __init__(self, model, tokenizer, max_draft: int = 8, min_match: int = 2):
        self.model = model
        self.tokenizer = tokenizer
        self.max_draft = max_draft
        self.min_match = min_match
        self.fallback_reason = None
        # For delegating to the plain (non-speculative) path. When a request
        # with a session falls through to here, FallbackRunner's own LCP reuse
        # works as-is (per the FallbackSession contract) — no need to duplicate
        # the implementation, delegation alone is enough.
        self._fallback = FallbackRunner(model, tokenizer)
        # Whether the KV cache can be trimmed is determined by the model's
        # layer structure (whether GDN is mixed in) and does not change per
        # request, so decide it once at construction time. What is actually
        # allocated is a dummy empty cache (make_prompt_cache here only builds
        # Python objects and involves no GPU computation).
        probe_cache = make_prompt_cache(model)
        self.trimmable = can_trim_prompt_cache(probe_cache)

    def generate(
        self,
        prompt_ids,
        max_tokens,
        temp,
        eos_ids,
        on_tokens,
        session,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict | None = None,
        seed: int | None = None,
        **extra,
    ):
        plain_only = (
            not self.trimmable
            or temp > 0.0
            or _needs_logits_processors(
                repetition_penalty, presence_penalty, frequency_penalty, logit_bias
            )
        )
        if plain_only:
            # When temp==0, mlx_lm.sample_utils.make_sampler itself
            # short-circuits to argmax and ignores top_p/top_k/min_p (line 46 of
            # make_sampler in mlx_lm/sample_utils.py), so there is no need to
            # care about them here — the only things to care about are the
            # logits_processors side (repetition_penalty etc.) and logit_bias.
            return self._fallback.generate(
                prompt_ids,
                max_tokens,
                temp,
                eos_ids,
                on_tokens,
                session,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                seed=seed,
                **extra,
            )
        if seed is not None:
            # Greedy, so randomness does not affect the output, but since we
            # accepted it as an argument just like SpecRunner/FallbackRunner, we
            # consume it anyway (avoids the caller's "I passed a seed and it was
            # silently ignored").
            mx.random.seed(seed)
        return self._lookup_generate(prompt_ids, max_tokens, eos_ids, on_tokens)

    def _lookup_generate(self, prompt_ids, max_tokens, eos_ids, on_tokens) -> dict:
        """The greedy n-gram lookup speculation proper.

        Per round: (a) if the same run as the prefix decided up to now has
        appeared earlier somewhere in the history, propose its continuation as
        ``draft`` (``SuffixAutomaton.draft``); (b) forward the pending token +
        draft together in a single pass (teacher forcing); (c) accept for as
        long as the argmax at each position matches the next element of the
        draft, and emit the argmax at the first mismatching position as a
        "bonus token"; (d) for the part of the draft that was not accepted,
        rewind the KV with ``trim_prompt_cache``.

        If there is no draft / it is empty, (b) becomes a forward of just one
        token, with the same cost and the same output as ordinary
        one-token-at-a-time greedy decoding — this is the grounds for it not
        getting slower in situations where no match occurs (measurements are
        separate; anything under docs/ is outside the scope of this change, so
        it is written here).

        session (per-conversation prompt cache reuse) is not handled here — a
        cache dedicated to that request is created anew every time (the caller's
        session is neither read nor written. On the next turn it simply
        re-prefills everything as usual; it does not misbehave).
        """

        t0 = time.perf_counter()
        cache = make_prompt_cache(self.model)
        ids = mx.array(prompt_ids, dtype=mx.uint32)
        y = _prefill(self.model, cache, ids, PREFILL_STEP_SIZE)

        sam = SuffixAutomaton()
        sam.extend_all(prompt_ids)

        tokens: list[int] = []
        ttft: float | None = None
        rounds = 0
        stop = False
        while len(tokens) < max_tokens and not stop:
            rounds += 1
            budget_left = max_tokens - len(tokens)
            # Always leave exactly 1 slot for the bonus token (even if the
            # whole draft is accepted, exactly 1 "new" token comes out at the
            # end — the same structure as DraftSpecRunner /
            # mlx_lm.speculative_generate_step).
            draft_cap = max(0, min(self.max_draft, budget_left - 1))
            draft = sam.draft(draft_cap, min_len=self.min_match) if draft_cap > 0 else None
            cand = y.tolist() + (draft or [])
            cand_arr = mx.array(cand, dtype=mx.uint32)
            logits = self.model(cand_arr[None], cache=cache)
            mx.eval(logits)
            if ttft is None:
                ttft = time.perf_counter() - t0
            preds = mx.argmax(logits[0], axis=-1).tolist()

            m = len(draft) if draft else 0
            accepted = 0
            while accepted < m and preds[accepted] == cand[accepted + 1]:
                accepted += 1
            bonus = preds[accepted]
            rejected = m - accepted
            if rejected > 0:
                trim_prompt_cache(cache, rejected)

            emit = (draft[:accepted] if draft else []) + [bonus]
            batch: list[int] = []
            for t in emit:
                if len(tokens) >= max_tokens:
                    break
                tokens.append(t)
                sam.extend(t)
                batch.append(t)
                if t in eos_ids:
                    stop = True
                    break
            if batch and on_tokens:
                on_tokens(batch)
            y = mx.array([bonus], dtype=mx.uint32)

        decode_time = time.perf_counter() - t0 - (ttft or 0.0)
        n_decode = max(len(tokens) - 1, 0)
        return {
            "tokens": tokens,
            "ttft_s": ttft or 0.0,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            "prefill_reused": 0,
            "prefill_new": len(prompt_ids),
            "tokens_per_step": (n_decode / rounds) if rounds else 0.0,
        }
