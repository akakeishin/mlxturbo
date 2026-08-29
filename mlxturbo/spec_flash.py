"""The foundation of Qwen3.8-Flash-Next speculative decoding: state capture and rollback.

`mlxturbo/spec.py` is tightly coupled to the 27B (qwen3_5) configuration, so we
keep this separately.

## Why capture is necessary

Speculation is "mix in a draft, verify it in one batch, and throw it away if it
misses". But 36 of Flash-Next's layers are GatedDeltaNet, and **the recurrent
state cannot be rolled back**. It is not something you can fix by cutting off the
tail as with KV, so we retain **the state immediately after finishing each
position** of the verification forward and adopt as much of it as was accepted.

There are 4 things that need rollback:

| Target | Layers | How to roll back |
|---|---|---|
| GDN recurrent state `cache[1]` | 36 | `states_all[:, keep-1]` (captured) |
| GDN conv window `cache[0]` | 36 | `conv_input[:, keep : keep+K-1]` (captured) |
| PLE conv window `cache[2]` | 1 | same as above |
| full attention KV and indexer | 12 | `KVCache.trim()` and truncating keys |
| n-gram context `cache[3]` | 1 | save the pre-forward value |

`ArraysCache.advance` only touches `lengths`/`left_padding` (both None for a
single sequence), so no offset rollback is needed on the GDN side.

## Design: do not replace the main model

We temporarily replace only `GatedDeltaNet.__call__` and `PLELayer._short_conv`,
and let everything else go through the main implementation as it is. Because we
**do not transcribe the forward path**, there is little room for the capturing
version and the plain version to diverge (transcribe it and they will drift
somewhere, without fail).
"""

from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn

# mlxturbo-serve wiring (added 2026-08-29): share the constant used to chunk
# prefill at the same width as FallbackRunner/SpecEngine (see the
# PREFILL_STEP_SIZE docstring in mlxturbo/spec.py -- for the same reason, that a
# width differing per path makes the output diverge even for the same prompt, we
# reuse the value from that one place here too).
#
# thinking support (added 2026-08-29): partial restoration via session-reuse
# checkpoints also reuses the same machinery as the spec.py side
# (ChatSession.checkpoints / _prefill_hidden / CHECKPOINT_RETENTION). The state
# of the GDN/PLE/n-gram layers that cannot be rolled back rides on the same
# ArraysCache.state (list) on either path, so spec.py's
# snapshot_untrimmable_caches/restore_untrimmable_caches are model-independent
# (they look only at caches' is_trimmable()/state) -- we reuse them as they are.
# spec.py is only read, never modified.
from .spec import CHECKPOINT_RETENTION, PREFILL_STEP_SIZE, snapshot_untrimmable_caches


def _arch():
    import mlx_lm.models.qwen4_exp as Q

    return Q


class Capture:
    """The records needed to roll back one verification forward."""

    def __init__(self):
        self.gdn = {}      # id(module) -> (conv_input, states_all)
        self.ple = {}      # id(module) -> full
        self.hyper = None  # the hyper state right before entering the final mixer
        self.pre = {}      # cache state before the forward (KV offset, etc.)


@contextmanager
def capture(model, light: bool = False):
    """A context that runs the verification forward while leaving behind the
    records needed for rollback.

    ``light=True`` (an addition, False by default): record only
    ``GatedResidual`` (``cap.hyper``) and let ``GatedDeltaNet``/``PLELayer`` go
    through their plain forward (which does not capture state).

    The reason: the ``states_all`` returned by
    ``gated_delta_update_with_states`` is ``(B, T, Hv, Dv, Dk)`` fp32 --
    ``Hv*Dv*Dk*4`` bytes per layer per token (~3MiB in this model's
    configuration). For the decode loop's verification forward ``T<=2``, so the
    value is negligible, but for ``generate_stream``'s final prefill chunk ``T``
    can be the chunk width itself (up to ``PREFILL_STEP_SIZE``). There, only
    ``cap.hyper[:, -1:]`` (the hyper state at the last position) is used, yet
    ``states_all`` was being allocated and retained unconditionally for all 36
    layers (the linear_attention layer count), which demanded hundreds of GB of
    memory at around T=2000 and got the whole process killed by macOS's
    memorystatus killer (measured; this was the cause of the symptom where the
    process vanished with no traceback). Even when ``GatedDeltaNet.__call__``/
    ``PLELayer._short_conv`` go through as the plain implementation, the cache
    updates (``cache[0]``/``cache[1]``/``cache[2]``/``cache.advance``) are
    performed by the main model with the same logic, so cache consistency is
    unchanged. The only thing that changes is that ``cap.gdn``/``cap.ple``
    (unused by this caller) stay empty. Existing calls (``light`` omitted =
    False: ``generate()``, and the verification forward inside
    ``generate_stream``'s decode loop) have their behavior completely unchanged
    -- this is an addition only.
    """

    Q = _arch()
    from .kernels.gated_delta_states import gated_delta_update_with_states

    cap = Capture()
    orig_gdn = Q.GatedDeltaNet.__call__
    orig_ple = Q.PLELayer._short_conv
    orig_hc = Q.GatedResidual.__call__
    mixer = model.model.hyper_connection_mixer

    def gdn(self, x, mask, cache):
        # Transcribe the main GatedDeltaNet.__call__ as-is, replacing only the
        # kernel with the one that returns states. Do not change the logic
        B, S, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.n_v, self.dv)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        conv_state = (
            cache[0]
            if (cache is not None and cache[0] is not None)
            else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.n_k, self.dk)
        k = k.reshape(B, S, self.n_k, self.dk)
        v = v.reshape(B, S, self.n_v, self.dv)

        inv_scale = self.dk**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        state = cache[1] if cache is not None else None
        out, states_all = gated_delta_update_with_states(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask
        )
        cap.gdn[id(self)] = (conv_input, states_all)
        if cache is not None:
            cache[1] = states_all[:, -1]
            cache.advance(S)
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))

    def ple_conv(self, x, cache):
        n = self.short_conv_state_len
        S = x.shape[1]
        prev = (
            cache[2]
            if (cache is not None and cache[2] is not None)
            else mx.zeros((x.shape[0], n, x.shape[-1]), dtype=x.dtype)
        )
        full = mx.concatenate([prev, x], axis=1)
        cap.ple[id(self)] = full
        if cache is not None:
            cache[2] = mx.contiguous(full[:, -n:, :])
        return nn.silu(self.conv1d(full[:, -(n + S) :, :]))

    def hc(self, hyper):
        if self is mixer:
            cap.hyper = hyper
        return orig_hc(self, hyper)

    if not light:
        Q.GatedDeltaNet.__call__ = gdn
        Q.PLELayer._short_conv = ple_conv
    Q.GatedResidual.__call__ = hc
    try:
        yield cap
    finally:
        if not light:
            Q.GatedDeltaNet.__call__ = orig_gdn
            Q.PLELayer._short_conv = orig_ple
        Q.GatedResidual.__call__ = orig_hc


def snapshot_pre(model, caches) -> dict:
    """**Before** the forward, note down what capture cannot restore."""
    pre = {"kv": [], "ctx": []}
    for layer, c in zip(model.model.layers, caches):
        if layer.layer_type == "full_attention":
            # _AttnCache derives from KVCache and is not an ArraysCache (it has no .cache)
            keys = c.indexer.keys
            pre["kv"].append((c.offset, None if keys is None else keys.shape[1]))
            pre["ctx"].append(None)
        else:
            pre["kv"].append(None)
            pre["ctx"].append(c[3])
    return pre


def rollback(model, caches, cap: Capture, pre: dict, keep: int, total: int,
             ids_kept=None):
    """Of the `total` tokens advanced by the verification forward, keep only the
    leading `keep`.

    `ids_kept` is the adopted token sequence (B, keep). The n-gram context has to
    be a value advanced up to "the position that does not include what was thrown
    away", so we rebuild it from the pre-forward context (restoring the
    pre-forward value as-is would roll back `keep` tokens too far).
    """
    if keep == total:
        return
    drop = total - keep
    for i, (layer, c) in enumerate(zip(model.model.layers, caches)):
        if layer.layer_type == "full_attention":
            c.trim(drop)
            if c.indexer.keys is not None:
                old_len = pre["kv"][i][1] or 0
                c.indexer.keys = c.indexer.keys[:, : old_len + keep]
            continue
        la = layer.linear_attn
        conv_input, states_all = cap.gdn[id(la)]
        k = la.conv_kernel_size
        # conv window after processing keep tokens = conv_input[:, keep : keep+k-1]
        c[0] = mx.contiguous(conv_input[:, keep : keep + k - 1, :])
        c[1] = states_all[:, keep - 1] if keep > 0 else None
        if layer.ple is not None:
            full = cap.ple.get(id(layer.ple))
            if full is not None:
                n = layer.ple.short_conv_state_len
                c[2] = mx.contiguous(full[:, keep : keep + n, :])

    if ids_kept is None:
        return
    ctx_len = model.args.text.ngram_size - 1
    for layer, c, ctx in zip(model.model.layers, caches, pre["ctx"]):
        if layer.layer_type == "full_attention" or ctx is None:
            continue
        c[3] = mx.concatenate([ctx, ids_kept], axis=1)[:, -ctx_len:]


class FlashSpecEngine:
    """Depth-1 speculative decoding that uses the MTP as the draft.

    Decoding is dispatch-bound: a batched forward costs only 1.17x the S=1 case
    even at S=16 (docs/STATUS.md). **A width-2 verification costs roughly the
    price of a single token**, so on acceptance one forward advances 2 tokens.

    Invariant: `cur` is the token to be fed next, the caches are processed up to
    just before it, and `hyper_prev` is the hyper state at the position that
    produced `cur`.
    """

    def __init__(self, model, mtp):
        self.model = model
        self.mtp = mtp
        self.rope = model.model.rope

    def _draft(self, cur, hyper_prev):
        Q = _arch()
        emb = self.model.model.embed_tokens(cur)
        cache = Q._AttnCache()
        mask = Q.create_attention_mask(emb, None)
        out = self.mtp(emb, hyper_prev, self.rope, mask, cache, cache.indexer)
        return mx.argmax(self.model.lm_head(out)[:, -1], axis=-1).reshape(1, 1)

    def generate(self, ids, max_tokens: int, caches=None):
        """Greedy generation. Returns (token sequence, accepted count, round count)."""
        model = self.model
        caches = caches or model.make_cache()
        with capture(model) as cap:
            logits = model(ids, cache=caches)
            mx.eval(logits)
        hyper_prev = cap.hyper[:, -1:]
        cur = mx.argmax(logits[:, -1], axis=-1).reshape(1, 1)

        # **Include the first token produced by prefill in the output too.**
        # `cur` is both "the token to be fed next" and "the most recently
        # generated token", so dropping it here shifts everything by one
        out, accepted, rounds = [int(cur.item())], 0, 0
        while len(out) < max_tokens:
            draft = self._draft(cur, hyper_prev)
            pair = mx.concatenate([cur, draft], axis=1)
            pre = snapshot_pre(model, caches)
            with capture(model) as cap:
                lg = model(pair, cache=caches)
                mx.eval(lg)
            nxt = mx.argmax(lg[:, 0], axis=-1).reshape(1, 1)
            out.append(int(nxt.item()))
            rounds += 1
            if int(nxt.item()) == int(draft.item()):
                # the draft hit -> the logits at position 1 are the next token as-is
                nxt2 = mx.argmax(lg[:, 1], axis=-1).reshape(1, 1)
                out.append(int(nxt2.item()))
                accepted += 1
                cur, hyper_prev = nxt2, cap.hyper[:, 1:2]
                # keep == total, so no rollback needed
            else:
                rollback(model, caches, cap, pre, keep=1, total=2, ids_kept=cur)
                cur, hyper_prev = nxt, cap.hyper[:, 0:1]
        return out[:max_tokens], accepted, rounds

    # ---------- mlxturbo-serve wiring (added 2026-08-29, additions only) ----------
    #
    # What follows is a new path added without changing ``generate()`` at all.
    # The reason is to absolutely not change the behavior of the existing
    # ``generate()``/``capture``/``rollback``/``snapshot_pre``, which are tied to
    # the measurements in docs/MTP-FLASH.md -- even where logic could be shared,
    # we chose to accept a little duplication over rewriting existing methods.

    @staticmethod
    def _sample(logits_row: mx.array, temp: float) -> mx.array:
        """Choose the next token from one position's worth of logits ((1, vocab)).

        temp<=0 is greedy (argmax, numerically identical to the existing
        generate()). temp>0 samples with temperature via
        ``mx.random.categorical`` (docs/MTP-FLASH.md, "sampling" section: as far
        as sampling from the verification-side logits goes there is no
        approximation in the correctness of the distribution -- the draft stays
        greedy). Returns (1, 1).
        """
        if temp > 0:
            return mx.random.categorical(logits_row.astype(mx.float32) / temp).reshape(1, 1)
        return mx.argmax(logits_row, axis=-1).reshape(1, 1)

    def generate_stream(
        self,
        ids: mx.array,
        max_tokens: int,
        caches=None,
        temp: float = 0.0,
        eos_ids=(),
        checkpoints: list | None = None,
        base_pos: int = 0,
    ):
        """The token-by-token version of ``generate()`` (for mlxturbo-serve's
        streaming).

        Yields the list of new tokens confirmed in one round (1 or 2 of them)
        each time, and returns ``(accepted, rounds)`` at the end of generation
        (the expectation is that you drive ``next()`` manually and pick it up
        from ``StopIteration.value`` -- ``generate()`` itself does not consume
        this generator; it is a completely independent path).

        There are 3 differences from ``generate()``, all of them additions that
        simply use the existing rollback machinery (``rollback``) as it is:

        1. When temp>0, sample with temperature from the logits at positions 0/1
           of the verification forward (the draft (MTP) itself stays greedy --
           as designed in docs/MTP-FLASH.md). Only when the sample matches the
           draft do we also sample from position 1 and advance 2 tokens. When it
           does not match, the position-1 logits are discarded outright (they are
           conditioned incorrectly, so we do not use them).
        2. Stop at a token matching ``eos_ids``. ``generate()`` does not look at
           eos at all, so this is the first place it is handled.
        3. When the number of new tokens one round produces (1 or 2) exceeds the
           remainder of ``max_tokens`` or an eos boundary, the excess is reliably
           thrown away by ``rollback`` (we just pass ``keep`` matching the number
           actually adopted --- ``rollback`` itself still has its existing branch
           that returns early when ``keep == total``, so an ordinary accept round
           (keep=total=2) is effectively a no-op). This means that even if the
           caller reuses the ``caches`` at the end of this generation as the next
           turn's session, the number of processed positions in ``caches`` always
           matches the number of tokens actually returned/yielded (it does not
           break the engine's invariant "cur is the token to be fed next, and the
           caches are processed up to just before it", even when we stop in the
           middle of a round).

        Prefill is chunked at the same width as
        ``mlxturbo.spec.PREFILL_STEP_SIZE`` (to avoid exceeding Metal's
        single-buffer limit; the same reason and the same width as
        ``_prefill_hidden`` in spec.py -- a width differing per path makes the
        output diverge even for the same prompt). Only the last chunk is
        forwarded with ``capture`` to obtain the hyper state (``hyper_prev`` uses
        only the last position, so there is no need to capture intermediate
        chunks). Intermediate chunks are forwarded with ``model.model(...)``
        (hidden only, not going through lm_head) and merely advance the cache --
        we do not repeat a vocabulary-sized matmul we will not use on every
        chunk. When the whole prompt fits in one chunk (which is nearly always
        the case for real conversational turns), this chunking is numerically
        identical to calling ``model(ids, cache=caches)`` once with capture
        (because no chunk boundary arises at all) -- it goes through the very
        same path as the existing ``generate()``.

        ``checkpoints`` (None when omitted; only FlashSpecRunner in
        mlxturbo/server.py passes it): if given, a snapshot of the layers that
        cannot be rolled back (GDN recurrent state, conv window, PLE conv window,
        n-gram context -- all of which ride on the same list returned by
        ``ArraysCache.state``) is appended in-place to this list at every chunk
        boundary. The position is absolute, i.e. with ``base_pos`` added (the
        starting position of this call from the caller's point of view = the
        number of tokens the session has already reused). Once there are more
        than ``mlxturbo.spec.CHECKPOINT_RETENTION`` entries, the oldest are
        evicted -- the same machinery and the same step size as
        ``mlxturbo.spec.ChatSession``/``_prefill_hidden`` (the prefill chunk
        boundaries themselves; we do not create a new step size). KV/indexer
        (full attention) is trimmable, so it needs no snapshot -- the restore
        side (``_try_checkpoint_restore_session_cache`` in mlxturbo/server.py)
        handles it with ``.trim()`` and by following along the indexer keys.
        """
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        eos = set(eos_ids)
        model = self.model
        caches = caches if caches is not None else model.make_cache()

        n = ids.shape[1]
        step = PREFILL_STEP_SIZE
        i = 0
        logits = None
        cap = None
        while i < n:
            j = min(i + step, n)
            chunk = ids[:, i:j]
            if j == n:
                # light=True: this chunk uses only cap.hyper[:, -1:] (referenced
                # right below). Full capture (cap.gdn/cap.ple) unconditionally
                # allocated memory proportional to T (this chunk's length, at
                # most PREFILL_STEP_SIZE) for all 36 layers, and OOMed the actual
                # machine at a few thousand tokens (see the docstring for
                # capture()'s light argument). The decode loop's verification
                # forward (below, T<=2) stays on full capture as before
                with capture(model, light=True) as cap:
                    logits = model(chunk, cache=caches)
                    mx.eval(logits)
            else:
                h = model.model(chunk, cache=caches)
                mx.eval(h)
                for c in caches:
                    state = getattr(c, "state", None)
                    if state is not None:
                        mx.eval(state)
                mx.clear_cache()
            i = j
            if checkpoints is not None:
                checkpoints.append((base_pos + i, snapshot_untrimmable_caches(caches)))
                del checkpoints[:-CHECKPOINT_RETENTION]
        hyper_prev = cap.hyper[:, -1:]
        if max_tokens == 0:
            # Keep the successfully prefetched cache, but do not expose the
            # first sampled ``cur``.  This mirrors SpecEngine.generate(0) and
            # lets FlashSpecRunner publish exactly the prompt as processed.
            return 0, 0
        cur = self._sample(logits[:, -1], temp)

        first = int(cur.item())
        out = [first]
        yield [first]
        accepted, rounds = 0, 0
        if first in eos:
            return accepted, rounds

        while len(out) < max_tokens:
            draft = self._draft(cur, hyper_prev)
            pair = mx.concatenate([cur, draft], axis=1)
            pre = snapshot_pre(model, caches)
            with capture(model) as cap:
                lg = model(pair, cache=caches)
                mx.eval(lg)
            rounds += 1
            nxt = self._sample(lg[:, 0], temp)
            if int(nxt.item()) == int(draft.item()):
                accepted += 1
                nxt2 = self._sample(lg[:, 1], temp)
                toks = [nxt, nxt2]
                hypers = [cap.hyper[:, 0:1], cap.hyper[:, 1:2]]
            else:
                toks = [nxt]
                hypers = [cap.hyper[:, 0:1]]

            vals = [int(t.item()) for t in toks]
            cut = next((k for k, v in enumerate(vals) if v in eos), None)
            if cut is not None:
                toks, hypers, vals = toks[: cut + 1], hypers[: cut + 1], vals[: cut + 1]
            remaining = max_tokens - len(out)
            if len(vals) > remaining:
                toks, hypers, vals = toks[:remaining], hypers[:remaining], vals[:remaining]

            # When keep==total (an ordinary accept round with no truncation),
            # rollback() itself returns early, so it is fine to always call it.
            rollback(model, caches, cap, pre, keep=len(vals), total=2, ids_kept=cur)
            out.extend(vals)
            yield vals
            cur, hyper_prev = toks[-1], hypers[-1]
            if cut is not None:
                break

        return accepted, rounds


__all__ = ["Capture", "FlashSpecEngine", "capture", "rollback", "snapshot_pre"]
