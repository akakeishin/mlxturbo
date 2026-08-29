"""MTP self-speculative decoding.

draft: chain the MTP block (~1.5% of the whole model) to produce n_draft candidates.
verify: verify n_draft+1 tokens together in a single forward pass of the main model.
An m=2 verification costs only 1.04x an m=1 one (measured), so whatever gets
accepted translates almost directly into the speed multiplier. Under greedy
decoding the output matches non-speculative decoding exactly.

Rollback across the 48 linear-attention layers: the projections and the conv run
over all m tokens at once (keeping the bandwidth amortization), and only the
recurrent update is looped per position so that the state of every position is
retained. On rejection we merely swap in the state and the conv window for that
position -- nothing is recomputed.

Phase D general-purpose pack (docs/RESEARCH.md, docs/KERNEL-INTEL.md, "depth
control" section): D1 replaced the MTP chain's depth ladder with an AdaEDL-style
confidence + per-position acceptance-rate EMA continue/stop gate; D3 turned the
context lookup into a suffix automaton (mlxturbo/sam.py) and replaced the
arbitration with ReSpec-style entropy triggering + a candidate EMA; D4 replaced
the temp>0 rejection sampling with Block Verification (arXiv:2403.10444).
None of them require training and none of them change the distribution guarantee
(D1/D3 only change the depth and the choice of proposal source, so they are
irrelevant to correctness; D4 has been confirmed by proof and by measurement to
have exactly the same distribution as rejection sampling. Details in
docs/STATUS.md, "Phase D general-purpose pack").
"""

import math
import time
from collections import Counter
from contextlib import nullcontext

import mlx.core as mx
import mlx.nn as nn

from ._mlx_compat import (
    KVCache,
    create_attention_mask,
    create_ssm_mask,
    validate_spec_model_contract,
)
from .kernels.gated_delta_states import gated_delta_update_with_states
from .kernels.dispatch import (
    dispatch_scope,
    enable as enable_quantized_dispatch,
    quantized_matmul as dispatched_quantized_matmul,
)
from .sam import SuffixAutomaton

# ---------- D1: confidence-gated chaining ----------
#
# Take AdaEDL's (arXiv:2410.18351) entropy lower bound 1-sqrt(gamma*H) as the
# per-link confidence signal, and use it together with the per-position
# acceptance-rate EMA + expected-gain threshold from KERNEL-INTEL.md's "depth
# control" (reach *= p_d, threshold = h*(1+expected)/(1+d*h)) as an equally
# weighted average. gamma=0.2 is the paper's default value. h=0.19 is the median
# of the 0.18-0.20 range for checkpoint-style rollback (mlxturbo retains the
# state of every position).
ADAEDL_GAMMA = 0.2
GATE_ROLLBACK_COST = 0.19
GATE_EMA_ALPHA = 0.2
# Confidence-weighted blend warmup: a gate-truncated step can only update
# the EMA for positions it actually verified, so an under-observed position
# must not be dominated by the (possibly-pessimistic) EMA/prior yet -- see
# _gate_depth docstring.
GATE_EMA_WARMUP = 5
# Typical per-position acceptance rates for a vanilla MTP chain (the FastMTP
# measurements quoted in RESEARCH.md: k=1 ~70% / k=2 ~11% / k=3 ~2%). Used as the
# prior until the EMA warms up.
_POS_ACCEPT_PRIOR = {1: 0.70, 2: 0.11, 3: 0.02}


def _pos_accept_prior(d: int) -> float:
    if d in _POS_ACCEPT_PRIOR:
        return _POS_ACCEPT_PRIOR[d]
    return 0.02 * (0.3 ** (d - 3))


# ---------- D3: ReSpec arbitration ----------
#
# The entropy threshold adopts ReSpec's (arXiv:2511.01282) experimental values
# theta_entropy=1.5 and maximum lookback length l=3 as they are (there is room
# for recalibration since the vocabulary size differs; noted in docs/STATUS.md).
# lambda_e (the entropy/length tradeoff), the EMA update rate alpha, and the
# quality threshold theta_score=0.5 are calibration values chosen on the mlxturbo
# side, because the body of the paper states no concrete values for them.
RESPEC_LOOKBACK = 3
RESPEC_LAMBDA_E = 1.0
RESPEC_ENTROPY_THETA = 1.5
RESPEC_EMA_ALPHA = 0.3
RESPEC_SCORE_THETA = 0.5
RESPEC_SCORE_PRIOR = 0.5
RESPEC_BUCKET_WIDTH = 4
RESPEC_MAX_BUCKET = 8

# ---------- prefill chunking ----------
#
# Feeding a new prompt into _hidden_forward in one shot allocates the attention
# score matrix as heads * S^2 * 2 bytes, which exceeds Metal's single-buffer
# limit (measured at a little over 86GB) somewhere around S ~ 43,000-52,000
# tokens and dies in [metal::malloc] (measured: 42,688 passes, while about
# 57,000 fails at 103,639,939,200 = 16 * 57,000^2 * 2). If we feed
# PREFILL_STEP_SIZE tokens at a time, that chunk's attention score matrix is
# heads * chunk * S * 2 (linear in S) and fits.
#
# mlxturbo/runner.py's FallbackRunner (mlx_lm.generate.stream_generate) already
# splits prefill at this value, and the Flash-Next shipping path goes through it.
# Giving the SpecEngine side a different step size would mean manufacturing a
# "same prompt, yet the output differs depending on the path" problem for
# ourselves, so the value lives in this one place only and mlxturbo/runner.py
# imports this constant directly so that both paths share it.
#
# Scope of determinism: mx.quantized_matmul returns different rounding when the
# batch length (M) differs, even for identical row data (confirmed by
# measurement -- the output of a single layer of a 4-bit quantized model alone
# already shows ~0.4% relative difference, which grows to ~30% after 40 layers).
# Therefore "changing the chunk width changes the output" -- it does not match
# the unchunked result. What we guarantee is two things: that it is
# "deterministic as long as the chunk width and the configuration are identical",
# and that speculative decoding's acceptance decision is consistent with the
# target logits that were actually computed (= spec-on and spec-off agree under
# the same chunk width). The former is a matter of bit-exact invariance that
# vLLM/llama.cpp do not provide either, and we do not require it. Details in
# docs/PREFILL-CHUNKING-DETERMINISM.md.
PREFILL_STEP_SIZE = 2048


# ---------- checkpoints for layers that cannot be rolled back ----------
#
# Unlike attention's KVCache, the linear layers of the GDN hybrid
# (mlx_lm.models.cache.ArraysCache: recurrent state + conv window) have no means
# of trimming back to an arbitrary position (is_trimmable() is always False).
# Under the policy that whatever cannot be rolled back can simply be held as a
# snapshot, _prefill_hidden saves the state of just those layers at every prefill
# chunk boundary (the same step size as PREFILL_STEP_SIZE at the top of this
# module; we do not create a new step size). Then, even if the next turn's new
# prompt diverges in the middle of the already-processed sequence (before a chunk
# boundary), we can restore back to the most recent checkpoint and re-prefill
# only the difference from there (consumed by
# _try_checkpoint_restore_session_cache in mlxturbo/server.py). Trimmable layers
# (KVCache) need no snapshot -- .trim() takes them back to any position.
#
# How many we retain, and the reasoning: in the measured text_config (a
# configuration confirmed in the docs/KERNEL-BRIEF-MOE-GDN.md series), 30 of the
# 40 layers are linear_attention. One checkpoint is ~66MB (recurrent state:
# linear_num_value_heads=32 * linear_key_head_dim=128
# * linear_value_head_dim=128, ~2.1MB per layer at mamba_ssm_dtype=float32,
# + conv state: ~131KB per layer at linear_conv_kernel_dim=4, totaled over the 30
# layers). We keep only the most recent CHECKPOINT_RETENTION=8 (a number aligned
# with the default of STATE.max_sessions -- purely so that the retention cost
# stays in the same order of magnitude even when 8 sessions hold their maximum
# simultaneously; there is no meaning to it beyond that): at most ~528MB per
# session, and ~4.2GB even if all 8 sessions (the pool limit) hold the maximum at
# the same time -- a rounding error on a 128GB machine. 8 * PREFILL_STEP_SIZE =
# 16,384 tokens' worth is reliably covered within the most recent turn, so the
# symptom we actually want to fix (a slight drift of the trailing few tokens
# caused by a thinking marker being reopened) is reliably within range. If we
# need to go back further than that (into the range where the old checkpoints
# have been pushed out), no matching checkpoint is simply found and we fall over
# to a fresh slot (the safe side).
CHECKPOINT_RETENTION = 8


def snapshot_untrimmable_caches(caches) -> list[tuple[int, object, object, object]]:
    """Save the state of only those layers of ``caches`` that cannot be trimmed
    (``is_trimmable()`` is False -- ArraysCache in the GDN hybrid). Trimmable
    layers (KVCache and the like) can be taken back to any position with
    ``.trim()``, so they are left untouched here.

    When the value returned by ``c.state`` is a list (as it is for ArraysCache),
    that getter hands back the internal mutable list itself -- if some other slot
    is updated later, as in ``cache[i] = new_array``, holding only the reference
    means what was supposed to be a "snapshot" gets swapped out for the latest
    state (each individual mx.array is itself immutable, but the list slot
    pointing at it does get replaced). So here we ``list(state)`` to copy the list
    itself, without copying the elements (mx.array is immutable unless explicitly
    overwritten via ``__setitem__``, so sharing the elements is safe).
    """

    snapshot = []
    for i, c in enumerate(caches):
        if c.is_trimmable():
            continue
        state = c.state
        if isinstance(state, list):
            state = list(state)
        snapshot.append(
            (i, state, getattr(c, "left_padding", None), getattr(c, "lengths", None))
        )
    return snapshot


def restore_untrimmable_caches(caches, snapshot) -> None:
    """Write back the state saved by ``snapshot_untrimmable_caches``."""

    for i, state, left_padding, lengths in snapshot:
        c = caches[i]
        c.state = state
        if left_padding is not None:
            c.left_padding = left_padding
        if lengths is not None:
            c.lengths = lengths


class ChatSession:
    """A container for carrying KV and linear state across turns.

    If the new prompt is a pure append to the already-processed sequence, only
    the difference is prefilled. If it is not an append (the template rewrote the
    history, etc.), then if ``checkpoints`` holds a snapshot of a recent prefill
    chunk boundary we restore back to it and re-prefill only the difference (see
    _select_session in mlxturbo/server.py). If neither is usable, we fall back to
    a full rebuild.
    """

    def __init__(self):
        self.caches = None
        self.mtp_cache = None
        self.mtp_valid = False
        self.processed = []
        self.h_last = None
        # [(position, snapshot), ...] in ascending order of position. snapshot
        # is the return value of snapshot_untrimmable_caches(). Only the most
        # recent CHECKPOINT_RETENTION entries are retained (trimmed on the
        # _prefill_hidden side).
        self.checkpoints: list[tuple[int, list]] = []

    def invalidate(self):
        """Drop every published field before aliased caches are mutated."""

        self.caches = None
        self.mtp_cache = None
        self.mtp_valid = False
        self.processed = []
        self.h_last = None
        self.checkpoints = []

    def publish(self, caches, mtp_cache, mtp_valid, processed, h_last, checkpoints=None):
        """Publish one internally consistent session snapshot after success."""

        self.caches = caches
        self.mtp_cache = mtp_cache
        self.mtp_valid = mtp_valid
        self.processed = processed
        self.h_last = h_last
        self.checkpoints = checkpoints if checkpoints is not None else []


class SpecEngine:
    def __init__(self, model, mtp, prefill_step_size: int = PREFILL_STEP_SIZE):
        validate_spec_model_contract(model)
        self.text = model.language_model
        self.inner = self.text.model
        self.mtp = mtp
        self.prefill_step_size = prefill_step_size
        # Install the replacement without touching prefill/draft behavior.
        # Dispatch is activated only by the verification scopes below.
        enable_quantized_dispatch(self.text, active=False)

    def _head(self, h_prenorm: mx.array, norm) -> mx.array:
        out = norm(h_prenorm)
        if self.text.args.tie_word_embeddings:
            embedding = self.inner.embed_tokens
            if (
                hasattr(embedding, "group_size")
                and hasattr(embedding, "bits")
                and "scales" in embedding
            ):
                return dispatched_quantized_matmul(
                    out,
                    embedding["weight"],
                    embedding["scales"],
                    embedding.get("biases"),
                    group_size=embedding.group_size,
                    bits=embedding.bits,
                    mode=embedding.mode,
                )
            return embedding.as_linear(out)
        with dispatch_scope():
            return self.text.lm_head(out)

    # ---------- main-model forward ----------

    def _hidden_forward(self, tokens: mx.array, caches, capture: bool):
        """tokens: (S,). Returns: (hidden before the final norm (1,S,D), rollback info for the linear layers)"""
        if capture and any(
            layer.is_linear and layer.linear_attn.sharding_group is not None
            for layer in self.inner.layers
        ):
            raise NotImplementedError(
                "SpecEngine capture does not support sharded GatedDeltaNet layers"
            )
        x = self.inner.embed_tokens(tokens[None])
        fa_mask = create_attention_mask(x, caches[self.inner.fa_idx])
        ssm_mask = create_ssm_mask(x, caches[self.inner.ssm_idx])
        sink = []
        h = x
        scope = dispatch_scope() if capture else nullcontext()
        with scope:
            for layer, c in zip(self.inner.layers, caches):
                if layer.is_linear:
                    if capture:
                        h = self._linear_capture(layer, h, c, sink, ssm_mask)
                    else:
                        h = layer(h, mask=ssm_mask, cache=c)
                else:
                    h = layer(h, mask=fa_mask, cache=c)
        return h, sink

    def _prefill_hidden(
        self,
        tokens: mx.array,
        caches,
        checkpoints: list | None = None,
        base_pos: int = 0,
    ) -> mx.array:
        """Forward the new-prompt portion in chunks (used only for the initial
        bulk prefill in generate()).

        Passing the whole of tokens to _hidden_forward at once allocates the
        attention score matrix as S^2 and exceeds Metal's limit (see the
        PREFILL_STEP_SIZE docstring at the top of this module). We always feed
        PREFILL_STEP_SIZE tokens at a time (even a short prompt goes through as
        a single chunk), doing mx.eval + mx.clear_cache() per chunk before moving
        on -- the same shape and the same step size as the mlx_lm.generate
        prefill loop that mlxturbo/runner.py's FallbackRunner uses. caches carries
        state across chunks through the ordinary capture=False path
        (KVCache.update_and_fetch / GDN's cache.advance).

        The numerics of the hidden states can differ depending on whether we
        chunk or not (because mx.quantized_matmul rounds in a batch-length-
        dependent way; see the PREFILL_STEP_SIZE docstring). What is guaranteed
        here is two things -- determinism at the same chunk width, and that
        speculation's acceptance decision is consistent with the target logits
        that were actually computed -- not bit-exact agreement with unchunked
        processing.

        ``checkpoints`` (None when omitted): if given, a state snapshot of the
        layers that cannot be rolled back (see snapshot_untrimmable_caches) is
        appended in-place to this list at every chunk boundary. The position is
        absolute, i.e. with ``base_pos`` added (the starting position of this
        call from the caller's point of view = the number of tokens the session
        has already reused). Once there are more than CHECKPOINT_RETENTION
        entries, the oldest are evicted.
        """
        n = tokens.shape[0]
        step = getattr(self, "prefill_step_size", PREFILL_STEP_SIZE)
        chunks = []
        i = 0
        while i < n:
            j = min(i + step, n)
            h_chunk, _ = self._hidden_forward(tokens[i:j], caches, capture=False)
            chunks.append(h_chunk)
            mx.eval(h_chunk)
            for c in caches:
                state = getattr(c, "state", None)
                if state is not None:
                    mx.eval(state)
            mx.clear_cache()
            i = j
            if checkpoints is not None:
                checkpoints.append((base_pos + i, snapshot_untrimmable_caches(caches)))
                del checkpoints[:-CHECKPOINT_RETENTION]
        return chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=1)

    def _linear_capture(self, layer, x, cache, sink, mask=None):
        """Perform the same computation as GatedDeltaNet while retaining the per-position recurrent state."""
        la = layer.linear_attn
        if la.sharding_group is not None:
            raise NotImplementedError(
                "SpecEngine capture does not support sharded GatedDeltaNet layers"
            )
        xin = layer.input_layernorm(x)
        B, S, _ = xin.shape

        qkv = la.in_proj_qkv(xin)
        z = la.in_proj_z(xin).reshape(B, S, la.num_v_heads, la.head_v_dim)
        b = la.in_proj_b(xin)
        a = la.in_proj_a(xin)

        conv_state = cache[0]
        if conv_state is None:
            conv_state = mx.zeros(
                (B, la.conv_kernel_size - 1, la.conv_dim), dtype=xin.dtype
            )
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        conv_out = nn.silu(la.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h_, d)
            for t, h_, d in zip(
                mx.split(conv_out, [la.key_dim, 2 * la.key_dim], -1),
                [la.num_k_heads, la.num_k_heads, la.num_v_heads],
                [la.head_k_dim, la.head_k_dim, la.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, states_all = gated_delta_update_with_states(
            q, k, v, a, b, la.A_log, la.dt_bias, cache[1], mask
        )

        n_keep = la.conv_kernel_size - 1
        old_lengths = cache.lengths
        old_left_padding = cache.left_padding
        if old_lengths is not None:
            ends = mx.clip(old_lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        cache[1] = states_all[:, -1]
        cache.advance(S)
        sink.append(
            (
                cache,
                states_all,
                conv_input,
                la.conv_kernel_size,
                old_lengths,
                old_left_padding,
            )
        )

        r = la.out_proj(la.norm(out, z).reshape(B, S, -1))
        h = x + r
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def _rollback(self, caches, sink, total: int, consumed: int):
        if consumed == total:
            return
        if not 0 < consumed < total:
            raise ValueError(
                f"rollback requires 0 < consumed < total; got {consumed}/{total}"
            )
        linear_cache_ids = {id(item[0]) for item in sink}
        for c in caches:
            if id(c) in linear_cache_ids:
                continue
            is_trimmable = getattr(c, "is_trimmable", None)
            trim = getattr(c, "trim", None)
            if not callable(is_trimmable) or not callable(trim) or not is_trimmable():
                raise TypeError(
                    f"verification cache {type(c).__name__} does not implement "
                    "the trimmable cache protocol"
                )
            requested = total - consumed
            trimmed = trim(requested)
            if trimmed != requested:
                raise RuntimeError(
                    f"verification cache trimmed {trimmed} tokens, expected {requested}"
                )
        for (
            cache,
            states_all,
            conv_input,
            kernel,
            old_lengths,
            old_left_padding,
        ) in sink:
            cache[1] = states_all[:, consumed - 1]
            n_keep = kernel - 1
            if old_lengths is not None:
                ends = mx.clip(old_lengths, 0, consumed)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(
                    conv_input[:, consumed : consumed + n_keep, :]
                )
            cache.lengths = (
                old_lengths - consumed if old_lengths is not None else None
            )
            cache.left_padding = (
                old_left_padding - consumed
                if old_left_padding is not None
                else None
            )

    # ---------- MTP ----------

    def _mtp_base(self, hiddens: mx.array) -> mx.array:
        """Apply the main model's final norm, to match the checkpoint contract
        base_hidden_variant=post_norm. Measured 2x2
        (bench/results/mtp-2x2-*.json): post/post wins at every depth."""
        return self.inner.norm(hiddens)

    def _mtp_append(self, tok_ids: mx.array, hiddens: mx.array, mtp_cache) -> mx.array:
        """Feed the per-position (embed(t_{i+1}), h_i) pairs to the MTP and accumulate K/V."""
        e = self.inner.embed_tokens(tok_ids[None])
        return self.mtp(e, hiddens, cache=mtp_cache)

    # ---------- D1: confidence-gated chaining ----------

    @staticmethod
    def _adaedl_bound(entropy: float, gamma: float = ADAEDL_GAMMA) -> float:
        """AdaEDL's (arXiv:2410.18351) acceptance lower bound 1 - sqrt(gamma * H).

        H is the Shannon entropy (natural log) of the draft distribution (this
        link's softmax). The low-confidence region, where the lower bound can go
        negative, is clipped to 0.
        """
        return max(0.0, 1.0 - math.sqrt(max(gamma, 0.0) * max(entropy, 0.0)))

    @staticmethod
    def _expected_future_gain(pos_accept_ema: dict, d: int, cap: int) -> float:
        """The expected number of additional tokens that can be accepted if we
        continue deeper than d.

        Sums the cumulative products of the per-position EMA (with the FastMTP
        measurements as the prior if it has not warmed up) over d+1..cap (the
        standard expected-acceptance-length convolution).
        """
        expected = 0.0
        running = 1.0
        for k in range(d + 1, cap + 1):
            running *= pos_accept_ema.get(k, _pos_accept_prior(k))
            expected += running
        return expected

    @classmethod
    def _gate_depth(
        cls,
        entropies: list,
        pos_accept_ema: dict,
        pos_obs_count: dict | None = None,
        gamma: float = ADAEDL_GAMMA,
        h: float = GATE_ROLLBACK_COST,
    ) -> int:
        """Decide, with a confidence gate, how much of the MTP chain is actually
        sent to verification.

        For each link d, update reach (=prod p_d) with a p_d that blends the
        AdaEDL lower bound and the per-position acceptance-rate EMA, and compare
        it against KERNEL-INTEL.md's threshold = h*(1+expected)/(1+d*h). We keep
        everything up to and including the link at which reach fell below the
        threshold (that position itself is still worth verifying) and truncate
        only what is deeper than it.

        The EMA's weight is ramped up by the observation count
        (w = n/(n+GATE_EMA_WARMUP)). A step where the gate stopped at a shallow
        position cannot update the EMA for links at deeper positions (what was
        cut off was never verified = there is no ground truth), so blending at a
        fixed 50/50 falls into a starvation state where "a prior that once came
        out low keeps taking effect while still under-observed and reproduces the
        shallow verdict". While observations are few we judge from AdaEDL's
        instantaneous confidence alone, and let the EMA take effect only once
        measurements have accumulated.
        """
        pos_obs_count = pos_obs_count or {}
        cap = len(entropies)
        reach = 1.0
        keep = 0
        for d in range(1, cap + 1):
            bound = cls._adaedl_bound(entropies[d - 1], gamma)
            n_obs = pos_obs_count.get(d, 0)
            w_ema = n_obs / (n_obs + GATE_EMA_WARMUP)
            ema = pos_accept_ema.get(d, _pos_accept_prior(d))
            p_d = w_ema * ema + (1 - w_ema) * bound
            reach *= max(0.0, min(1.0, p_d))
            keep = d
            expected = cls._expected_future_gain(pos_accept_ema, d, cap)
            threshold = h * (1 + expected) / (1 + d * h)
            if reach <= threshold:
                break
        return keep

    # ---------- D3: context lookup (SAM) + ReSpec arbitration ----------

    @staticmethod
    def _respec_trigger(
        entropy_hist: list,
        lookback: int = RESPEC_LOOKBACK,
        lambda_e: float = RESPEC_LAMBDA_E,
        theta_entropy: float = RESPEC_ENTROPY_THETA,
    ) -> bool:
        """The entropy-guided trigger of ReSpec (arXiv:2511.01282) Algorithm 1.

        Over the target-distribution entropies of the most recent confirmed
        tokens, compute for each lookback length k=1..lookback the mean entropy
        H_k and the confidence score C_k = H_k + lambda_e/k, and choose the k*
        that minimizes C_k. If that H_k* is at or below theta_entropy, the
        context counts as "easy to predict" and retrieval is triggered.
        """
        n = min(lookback, len(entropy_hist))
        if n == 0:
            return False
        tail = entropy_hist[-n:]
        best_c = None
        best_h = None
        running_sum = 0.0
        for k in range(1, n + 1):
            running_sum += tail[n - k]
            h_k = running_sum / k
            c_k = h_k + lambda_e / k
            if best_c is None or c_k < best_c:
                best_c, best_h = c_k, h_k
        return best_h is not None and best_h <= theta_entropy

    @staticmethod
    def _respec_bucket(
        match_len: int,
        width: int = RESPEC_BUCKET_WIDTH,
        cap: int = RESPEC_MAX_BUCKET,
    ) -> int:
        """Quantize SAM's match length into a bucket for the EMA ranking.

        mlxturbo's lookup candidate is the single candidate SAM returns (the most
        recent occurrence of the longest match), so ReSpec's "EMA per match
        position" is mapped here onto "EMA per match-length bucket": the longer
        the match, the higher the chance that the recurrence is not coincidental,
        and learning the confidence per bucket is the natural single-candidate
        analogue.
        """
        return min(match_len // width, cap)

    # ---------- D4: Block Verification ----------

    @staticmethod
    def _block_verify_tau(p_l: list, u_l: list, n_avail: int) -> int:
        """Block Verification (Sun et al., arXiv:2403.10444, Algorithm 2 /
        Eq. 4-5) specialized to a deterministic (delta) draft proposal.

        mlxturbo's draft tokens (MTP argmax chain and SAM lookup) are always
        a fixed, deterministic proposal, i.e. Ms(x | c, X^i) = 1 for the
        drafted token and 0 otherwise. Substituting this delta distribution
        into the paper's general formulas collapses the running joint-
        probability ratio p_i (Algorithm 2 Line 4) to a plain cumulative
        product of ``p_l`` (already the per-position target probability of
        the drafted token, exactly what the previous sequential rejection
        sampler used), and both the block acceptance probability (Eq. 5)
        and the residual distribution (Eq. 4) reduce to the same
        "renormalize target logits excluding the next drafted token" step
        the sequential sampler already performs -- only *how many* tokens
        that step accepts changes. See docs/STATUS.md Phase D notes for the
        full derivation and the Monte Carlo check against the sequential
        sampler (bench/test_block_verify.py).

        Unlike sequential rejection sampling, which stops at the first
        rejected position (``a`` = length of the run of consecutive
        successes from the start), block verification evaluates every
        candidate sub-block length and keeps the *longest* one whose
        acceptance draw succeeds, even if a shorter prefix's draw failed:
        ``tau = max{L : u_l[L-1] <= h_block(L)}`` (0 if none succeed). This
        is what lets it accept at least as many tokens in expectation
        without ever accepting fewer (paper Theorem 2); for a genuinely
        stochastic draft distribution this raises expected accepted length,
        while for mlxturbo's deterministic draft the two samplers are
        provably (and empirically, see the Monte Carlo test) identically
        distributed, since a degenerate proposal leaves no joint-coupling
        slack for block verification to exploit -- see docs/STATUS.md for
        the derivation of why this is expected, not a bug.
        """
        tau = 0
        p_cum = 1.0
        for length in range(1, n_avail + 1):
            p_cum *= p_l[length - 1]
            if length < n_avail:
                q_next = p_l[length]
                numer = p_cum * (1.0 - q_next)
                denom = numer + (1.0 - p_cum)
                h_block = numer / denom if denom > 0.0 else 0.0
            else:
                h_block = p_cum
            if u_l[length - 1] <= h_block:
                tau = length
        return tau

    # ---------- generation ----------

    def generate(
        self,
        prompt_ids,
        max_tokens: int = 256,
        n_draft: int = 3,
        max_draft: int = 0,
        lookup_len: int = 16,
        lookup_ngram: int = 4,
        temp: float = 0.0,
        eos_ids=(),
        on_tokens=None,
        session: ChatSession | None = None,
        fly_theta: float = 0.0,
        fly_window: int = 6,
    ):
        """The depth of the MTP chain is decided by a confidence gate and the
        lookup trigger by a ReSpec-style entropy threshold, each on a per-step
        basis (Phase D1/D3).

        MTP chain (source="mtp"): a reach/threshold gate that uses the AdaEDL
        entropy lower bound together with the per-position acceptance-rate EMA on
        each link decides how many links are actually sent to verification
        (``_gate_depth``). The depth cap reuses
        ``max_draft if max_draft > 0 else n_draft``.

        Context lookup (source="lookup"): the suffix automaton in
        mlxturbo/sam.py tracks the longest match in O(1) amortized. It triggers
        only when the target entropy history of the most recent confirmed tokens
        satisfies ReSpec's entropy-guided trigger (``_respec_trigger``) and the
        acceptance-rate EMA of the match-length bucket is at or above the
        threshold (``_respec_bucket``). If it misses, we fall back to MTP.

        Verification for temp>0 is by Block Verification (arXiv:2403.10444,
        ``_block_verify_tau``), which makes the accepted length monotonically
        non-decreasing while keeping exactly the same distribution as sequential
        rejection sampling (see docs/STATUS.md).

        ``n_draft=0, max_draft=0, lookup_len=0`` is the non-speculative
        baseline contract used by ``bench/gate.py``.
        """
        if min(max_tokens, n_draft, max_draft, lookup_len, lookup_ngram) < 0:
            raise ValueError("generation limits must be non-negative")
        eos = set(eos_ids)
        prompt_ids = list(prompt_ids)
        if self.mtp is None:
            # A configuration with no MTP checkpoint (when load_cli_mtp in
            # cli.py returned None). Cut the MTP chain entirely and keep
            # speculating with lookup (D3, SAM) alone. Zeroing out cap_base here
            # means the subsequent draft loop and the D7 extension branch (both
            # of which call self.mtp) never fire.
            n_draft = 0
            max_draft = 0
        use_mtp = n_draft > 0 or max_draft > 0

        caches = mtp_cache = None
        reused = 0
        reused_h_last = None
        # For calls with no session (bench/gate.py and other paths that hit
        # SpecEngine.generate directly) there is no notion of a next turn at all,
        # so we do not track checkpoints either (with None, _prefill_hidden skips
        # snapshotting entirely). If there is a session we always track them,
        # whether the slot is new or continued (they are published at the end of
        # generate(), via session.publish).
        checkpoints: list | None = [] if session is not None else None
        if session is not None and session.caches is not None:
            pl = session.processed
            n = min(len(pl), len(prompt_ids))
            lcp = 0
            while lcp < n and pl[lcp] == prompt_ids[lcp]:
                lcp += 1
            if lcp == len(pl) and lcp < len(prompt_ids):
                # Reuse of the KV/GDN state is decoupled from whether the MTP
                # chain survives -- carrying over an mtp_cache with no
                # corresponding h_last makes the later concat fail (in a session
                # where mtp_valid is False, h_last/mtp_cache do not correspond to
                # the position after rollback; see
                # _try_checkpoint_restore_session_cache in mlxturbo/server.py).
                # So MTP is carried over only when session.mtp_valid, and
                # otherwise the use_mtp block below naturally falls into the same
                # path as "no session reuse" (rebuilding from prompt[1:]).
                caches = session.caches
                checkpoints = session.checkpoints
                reused = lcp
                if session.mtp_valid:
                    mtp_cache = session.mtp_cache
                    reused_h_last = session.h_last
                # The local variables now own these mutable caches.  Any error,
                # including KeyboardInterrupt or a callback failure, leaves the
                # public session invalid instead of half-published.
                session.invalidate()
        if caches is None:
            caches = self.text.make_cache()
        if mtp_cache is None:
            mtp_cache = KVCache()

        prompt = mx.array(prompt_ids[reused:])

        t0 = time.perf_counter()
        h_all = self._prefill_hidden(prompt, caches, checkpoints=checkpoints, base_pos=reused)
        if use_mtp:
            if reused and reused_h_last is not None:
                mtp_hiddens = mx.concatenate(
                    [reused_h_last, h_all[:, :-1]], axis=1
                )
                self._mtp_append(prompt, self._mtp_base(mtp_hiddens), mtp_cache)
            elif prompt.shape[0] > 1:
                self._mtp_append(prompt[1:], self._mtp_base(h_all[:, :-1]), mtp_cache)
        h_last = h_all[:, -1:]

        if max_tokens == 0:
            mx.eval(h_last)
            ttft = time.perf_counter() - t0
            if session is not None:
                session.publish(
                    caches, mtp_cache, use_mtp, list(prompt_ids), h_last, checkpoints
                )
            return {
                "prefill_reused": reused,
                "prefill_new": len(prompt_ids) - reused,
                "tokens": [],
                "ttft_s": ttft,
                "decode_tps": 0.0,
                "accept_hist": {},
                "accept_trace": [],
                "src_hist": {"lookup": {}, "mtp": {}},
                "phase_s": {"draft": 0.0, "verify": 0.0, "maint": 0.0},
                "steps": 0,
                "mean_accepted": 0.0,
                "tokens_per_step": 0.0,
            }

        y_logits = self._head(h_last, self.inner.norm)
        if temp > 0:
            y = mx.random.categorical(
                y_logits[0].astype(mx.float32) / temp
            ).reshape(1)
        else:
            y = mx.argmax(y_logits, axis=-1).reshape(1)
        mx.eval(y)
        ttft = time.perf_counter() - t0

        out_tokens = [int(y.item())]
        if on_tokens:
            on_tokens(out_tokens[:])
        accept_hist = Counter()
        accept_trace = []
        fed_gen = []
        src_hist = {"lookup": Counter(), "mtp": Counter()}
        lookup_ext_hits = 0
        # D6 (FLy) is explicitly opt-in (fly_theta > 0). With a temperature it
        # is D4's closed form that guarantees distributional exactness, so we
        # apply this to greedy only.
        fly_active = fly_theta > 0.0 and temp == 0
        fly_defer_accepts = 0
        phase = {"draft": 0.0, "verify": 0.0, "maint": 0.0}
        t1 = time.perf_counter()

        # D3: dynamic suffix automaton replaces the O(n) backward scan;
        # seed with the full (session-reused-or-not) prompt so cross-turn
        # repeats are visible too. Skipped entirely when lookup is off, so
        # the baseline/no-lookup path pays nothing extra.
        track_lookup = lookup_len > 0
        sam = SuffixAutomaton() if track_lookup else None
        if sam is not None:
            sam.extend_all(prompt_ids)
            sam.extend(out_tokens[0])
        # ReSpec-style arbitration state (fresh per generate() call, same
        # lifetime as the old depth/lookup_cool/lookup_cur locals it
        # replaces): per-position MTP acceptance-rate EMA (D1), per-match-
        # length-bucket lookup quality EMA (D3), and a short rolling history
        # of confirmed-token target entropies driving the D3 trigger.
        pos_accept_ema: dict = {}
        pos_obs_count: dict = {}
        quality_ema: dict = {}
        entropy_hist: list = []
        cap_base = max_draft if max_draft > 0 else n_draft
        while len(out_tokens) < max_tokens and out_tokens[-1] not in eos:
            ts = time.perf_counter()
            mtp_off0 = mtp_cache.offset
            lk = None
            lookup_bucket = None
            chain_head = None
            proposal_cap = max_tokens - len(out_tokens) - 1
            cap = min(cap_base, proposal_cap)
            triggered = (
                proposal_cap > 0
                and track_lookup
                and self._respec_trigger(entropy_hist)
            )
            if triggered:
                match_len, _ = sam.longest_match()
                if match_len >= lookup_ngram:
                    lookup_bucket = self._respec_bucket(match_len)
                    score = quality_ema.get(lookup_bucket, RESPEC_SCORE_PRIOR)
                    if score >= RESPEC_SCORE_THETA:
                        cand_len = max(1, round(lookup_len * score))
                        lk = sam.draft(
                            min(cand_len, proposal_cap), min_len=lookup_ngram
                        )
                if lk is None:
                    lookup_bucket = None
            if lk is None and triggered and cap >= 1 and proposal_cap >= 2:
                # D7 (an adaptation of LogitSpec arXiv:2507.01449): only when
                # the direct match missed, extend the search key by one with the
                # token predicted by the MTP's first link and query SAM again.
                # On a miss the first link is reused as-is for the first entry of
                # the MTP chain, so the extra GPU cost is one sync.
                # The quality EMA is learned separately under ("ext", bucket)
                # because its population differs from that of the direct key (by
                # analogy with ReSpec's source-aware verification).
                dh, dtok = self._mtp_base(h_last), y
                h_mtp = self._mtp_append(dtok, dh, mtp_cache)
                d_logits = self._head(h_mtp[:, -1:], self.mtp.norm)[0, -1]
                d_probs = mx.softmax(d_logits.astype(mx.float32), axis=-1)
                conf1 = -mx.sum(d_probs * mx.log(mx.maximum(d_probs, 1e-12)))
                d1 = mx.argmax(d_logits, axis=-1).reshape(1)
                mx.eval(d1)
                m1 = int(d1.item())
                ext_len, _ = sam.peek_match(m1)
                if ext_len >= lookup_ngram:
                    ext_bucket = ("ext", self._respec_bucket(ext_len))
                    score = quality_ema.get(ext_bucket, RESPEC_SCORE_PRIOR)
                    if score >= RESPEC_SCORE_THETA:
                        cand_len = max(1, round(lookup_len * score))
                        _, cont = sam.draft_after(
                            m1,
                            min(cand_len, proposal_cap - 1),
                            min_len=lookup_ngram,
                        )
                        if cont:
                            lk = [m1] + cont
                            lookup_bucket = ext_bucket
                            lookup_ext_hits += 1
                if lk is None:
                    chain_head = (conf1, d1, self.mtp.norm(h_mtp[:, -1:]))
            if lk:
                source = "lookup"
                window = mx.concatenate([y, mx.array(lk)])
            else:
                source = "mtp"
                lookup_bucket = None
                drafts = []
                confidences = []
                if chain_head is not None:
                    conf1, d1, dh = chain_head
                    confidences.append(conf1)
                    drafts.append(d1)
                    dtok = d1
                else:
                    dh, dtok = self._mtp_base(h_last), y
                for _ in range(max(cap - len(drafts), 0)):
                    h_mtp = self._mtp_append(dtok, dh, mtp_cache)
                    d_logits = self._head(h_mtp[:, -1:], self.mtp.norm)[0, -1]
                    d_probs = mx.softmax(d_logits.astype(mx.float32), axis=-1)
                    confidences.append(
                        -mx.sum(d_probs * mx.log(mx.maximum(d_probs, 1e-12)))
                    )
                    d = mx.argmax(d_logits, axis=-1).reshape(1)
                    drafts.append(d)
                    dh, dtok = self.mtp.norm(h_mtp[:, -1:]), d
                if confidences:
                    # D1: one combined sync for the whole confidence vector
                    # (never per-link) decides how many of the already-
                    # drafted links are worth sending to verification.
                    mx.eval(*confidences)
                    keep = self._gate_depth(
                        [float(c.item()) for c in confidences],
                        pos_accept_ema,
                        pos_obs_count,
                    )
                    drafts = drafts[:keep]
                window = mx.concatenate([y] + drafts)
            mx.async_eval(window)
            phase["draft"] += time.perf_counter() - ts

            ts = time.perf_counter()
            # A one-token window cannot require rollback.  Use the native path
            # so n_draft=0 + lookup_len=0 is a true non-speculative baseline.
            hs, sink = self._hidden_forward(
                window, caches, capture=window.shape[0] > 1
            )
            logits = self._head(hs, self.inner.norm)
            accepted_eos = None
            ent_l = None
            if temp == 0:
                preds = mx.argmax(logits, axis=-1)[0]
                if track_lookup or fly_active:
                    ent_probs = mx.softmax(logits[0].astype(mx.float32), axis=-1)
                    ent_row = -mx.sum(
                        ent_probs * mx.log(mx.maximum(ent_probs, 1e-12)), axis=-1
                    )
                    mx.eval(preds, window, ent_row)
                    ent_l = ent_row.tolist()
                else:
                    mx.eval(preds, window)
                preds_l, window_l = preds.tolist(), window.tolist()
                n_avail = len(window_l) - 1
                a = 0
                while a < n_avail and preds_l[a] == window_l[a + 1]:
                    a += 1
                if fly_active and ent_l is not None:
                    # D6 (FLy, arXiv:2511.22972; the two-stage mechanism applied
                    # to greedy, opt-in): at the first mismatch j, if the
                    # target's normalized entropy h_j >= theta (= the target
                    # itself is unsure), do not reject immediately but defer, and
                    # accept j's draft token only if all fly_window following
                    # tokens match (= the model agrees with the alternative
                    # phrasing and the continuation does not diverge). A mismatch
                    # with h_j < theta is a genuine error and is rejected
                    # immediately. The output distribution is no longer exact
                    # (off by default; the gate and the official benchmarks stay
                    # exact).
                    log_v = math.log(logits.shape[-1])
                    while a < n_avail:
                        if ent_l[a] / log_v < fly_theta:
                            break
                        end = a + 1 + fly_window
                        if end > n_avail:
                            break
                        if any(
                            preds_l[i] != window_l[i + 1]
                            for i in range(a + 1, end)
                        ):
                            break
                        fly_defer_accepts += 1
                        a = end
                        while a < n_avail and preds_l[a] == window_l[a + 1]:
                            a += 1
                for i in range(1, a + 1):
                    if window_l[i] in eos:
                        a = i
                        accepted_eos = i
                        break
                if accepted_eos is None:
                    next_tok = preds_l[a]
            else:
                # D4: Block Verification (arXiv:2403.10444) replaces
                # sequential rejection sampling. The draft is a deterministic
                # proposal (a delta distribution), so the closed form for the
                # accepted length tau can be written with p_l alone (the same
                # sequence of target probabilities as sequential rejection); the
                # derivation is in the _block_verify_tau docstring and
                # docs/STATUS.md. The output distribution is exactly identical to
                # sequential rejection.
                lg = logits[0].astype(mx.float32) / temp
                probs = mx.softmax(lg, axis=-1)
                nw = window.shape[0] - 1
                p_draft = mx.take_along_axis(
                    probs[:nw], window[1:, None], axis=-1
                )[:, 0]
                u = mx.random.uniform(shape=(nw,))
                if track_lookup:
                    nat_probs = mx.softmax(logits[0].astype(mx.float32), axis=-1)
                    ent_row = -mx.sum(
                        nat_probs * mx.log(mx.maximum(nat_probs, 1e-12)), axis=-1
                    )
                    mx.eval(p_draft, u, window, ent_row)
                    ent_l = ent_row.tolist()
                else:
                    mx.eval(p_draft, u, window)
                window_l = window.tolist()
                p_l, u_l = p_draft.tolist(), u.tolist()
                n_avail = nw
                a = self._block_verify_tau(p_l, u_l, n_avail)
                # A block-accepted length can still contain EOS; truncate at
                # the first one exactly like the sequential walk used to
                # (can't emit past EOS even though block verification
                # doesn't itself break on it).
                for i in range(1, a + 1):
                    if window_l[i] in eos:
                        a = i
                        accepted_eos = i
                        break
                if accepted_eos is None:
                    row = lg[a]
                    if a < n_avail:
                        rejected = mx.arange(row.shape[-1]) == window_l[a + 1]
                        row = mx.where(rejected, -mx.inf, row)
                    next_tok = int(mx.random.categorical(row).item())
            if accepted_eos is not None:
                consumed = a
            else:
                consumed = 1 + a
            phase["verify"] += time.perf_counter() - ts

            ts = time.perf_counter()
            accept_hist[a] += 1
            accept_trace.append(a)
            src_hist[source][a] += 1
            if source == "lookup":
                # D3: ReSpec feedback -- EMA the observed acceptance rate of
                # this match-length bucket (Eq. 5: R = accepted/proposed).
                r = a / len(lk)
                quality_ema[lookup_bucket] = (
                    1 - RESPEC_EMA_ALPHA
                ) * quality_ema.get(
                    lookup_bucket, RESPEC_SCORE_PRIOR
                ) + RESPEC_EMA_ALPHA * r
            elif drafts:
                # D1: per-position acceptance-rate EMA feeding _gate_depth.
                for d in range(1, len(drafts) + 1):
                    observed = 1.0 if a >= d else 0.0
                    pos_accept_ema[d] = (
                        1 - GATE_EMA_ALPHA
                    ) * pos_accept_ema.get(
                        d, _pos_accept_prior(d)
                    ) + GATE_EMA_ALPHA * observed
                    pos_obs_count[d] = pos_obs_count.get(d, 0) + 1

            self._rollback(caches, sink, len(window_l), consumed)
            if use_mtp:
                mtp_cache.trim(mtp_cache.offset - mtp_off0)
                true_hiddens = mx.concatenate(
                    [h_last, hs[:, : consumed - 1]], axis=1
                )
                self._mtp_append(window[:consumed], self._mtp_base(true_hiddens), mtp_cache)

            h_last = hs[:, consumed - 1 : consumed]
            if accepted_eos is not None:
                step_tokens = window_l[1 : a + 1]
            else:
                y = mx.array([next_tok])
                step_tokens = window_l[1:consumed] + [next_tok]
            out_tokens.extend(step_tokens)
            if sam is not None:
                sam.extend_all(step_tokens)
            if ent_l is not None:
                entropy_hist.extend(ent_l[: len(step_tokens)])
            fed_gen.extend(window_l[:consumed])
            if on_tokens:
                on_tokens(step_tokens)
            phase["maint"] += time.perf_counter() - ts

        decode_time = time.perf_counter() - t1
        n_decode = len(out_tokens) - 1
        steps = sum(accept_hist.values())
        if session is not None:
            session.publish(
                caches,
                mtp_cache,
                use_mtp,
                prompt_ids + fed_gen,
                h_last,
                checkpoints,
            )
        return {
            "prefill_reused": reused,
            "prefill_new": len(prompt_ids) - reused,
            "tokens": out_tokens,
            "ttft_s": ttft,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            "accept_hist": dict(sorted(accept_hist.items())),
            "accept_trace": accept_trace,
            "src_hist": {k: dict(sorted(v.items())) for k, v in src_hist.items()},
            "lookup_ext_hits": lookup_ext_hits,
            "fly_defer_accepts": fly_defer_accepts,
            "phase_s": phase,
            "steps": steps,
            "mean_accepted": (
                sum(k * v for k, v in accept_hist.items()) / steps if steps else 0.0
            ),
            "tokens_per_step": (n_decode / steps if steps else 0.0),
        }
