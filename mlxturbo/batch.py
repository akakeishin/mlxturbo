"""Monkey patches that make Flash-Next (`qwen4_exp`) work under batch generation.

`mlx_lm.generate.BatchGenerator` bundles per-sequence caches together with
`merge`, drops rows with `filter`, pulls single sequences back out with
`extract`, and appends with `extend`. Among Flash-Next's caches, the GDN side
(`ArraysCache`) has all of these, but the full-attention side does not. This
module injects what is missing straight into the classes, without touching the
venv.

    from mlxturbo import batch
    batch.enable_batch_cache()

## What was missing

1. `_IndexerCache` (the raw keys of the QSA indexer) has no `filter`/`merge`/
   `extract`.

2. `_AttnCache` inherits from `KVCache`, so it does "have" a `merge`. But
   `KVCache.merge` returns a `BatchKVCache` — a **different class**, one that
   carries no indexer. `Qwen4ExpModel.__call__` picks the indexer up via
   `hasattr(c, "indexer")`, so the indexer vanishes entirely without so much as
   an exception. Meanwhile `is_batchable` on the `mlx_lm` side
   (`hasattr(c, "merge")`) passes just fine.

3. `QSAIndexer` and `Attention` use `cache.offset` as a Python integer.
   `BatchKVCache.offset` is a `(B,)` array, so `mx.arange` blows up. It blows up
   even at B=1 (because passing through merge turns the cache into a
   `BatchKVCache`).

4. Left padding has no effect anywhere. `Qwen4ExpModel.__call__` passes a
   **list**, `create_attention_mask(h, [attn_cache])`, but this version of
   `create_attention_mask` inspects a single cache and calls `make_mask` on it.
   A list has no `make_mask`, so the result never gets past "causal" and the
   left-padded columns sail right through. `conv_mask` is worse still: it is
   hardcoded to `None`.

5. The GDN convolution state, the PLE convolution state, and the n-gram context
   all saved their state by slicing off the "last k columns". During prefill
   with right padding the tail is padding, so the recurrent state of a short
   sequence got filled with padding. `qwen3_next` avoids this by pointing at
   the right position via `cache.lengths` (models/qwen3_next.py:263-269).
   `qwen4_exp` had nothing of the sort — it does now (`_tail_window` in the
   vendored module), so this module no longer patches anything for it.

6. Rope positions are counted by column index. Prefill runs with right padding,
   so the keys are rotated by "the position within that sequence". Once finalize
   has rearranged the layout to left padding, rotating the decode queries by
   column index shifts the origin by the amount of padding and breaks rope's
   relative property. The true per-sequence position is already held by
   `BatchKVCache.offset` (an array), so use that.

7. `ArraysCache.extract` crashes on empty slots (on the mlx_lm side).
   Flash-Next's GDN layers use 2 of the 4 slots (PLE / n-gram) only in PLE
   layers, so every time a single sequence finishes you get a `TypeError`.

## How it is patched (seams, not copies)

Everything above used to be fixed by copying the forward passes of
`Attention` / `GatedDeltaNet` / `PLELayer` / `Qwen4ExpModel` into this module
and editing a few lines. The copies drifted: fusions that landed in the
vendored module (`_wide_qkv`, `_project_in`) never reached the batch path, and
nothing said so out loud. The vendored module now carries the hooks instead
(`Attention._positions` / `_final_mask`, `Qwen4ExpModel._make_masks`), and
this module replaces only those. The recurrent state windows needed no hook at
all in the end: `_tail_window` in the vendored module reads `cache.lengths`
itself, which is the upstream convention (`qwen3_next`) and is a no-op for a
cache that has no per-row lengths. The forward passes
themselves live in one place. `QSAIndexer.__call__` is still replaced whole:
its left-padding differences run through the middle of the block-grid
computation, so a hook would have to expose the internals.

The rule of thumb (docs/BACKLOG.md): if the overrides for one class go past 3,
the hooks have started leaking internal structure and copying is the better
trade again.

## What was left alone

When there is no padding (B=1, or a batch of equal lengths) the same graph runs
as in the stock implementation. Both the mask construction and the indexer's
visibility test are written so that if `left_padding` is all zeros they are
character-for-character identical to stock.

## What used to be changed here on purpose

QSA's "the sub-block tail is always visible" was made causal in this module
first, because `BatchGenerator` splits the prompt into (n-1, 1) and then
advances in `prefill_step_size` increments, which made the leak visible as an
output that moved with the split. That fix now lives in the vendored module,
where every path gets it. What is left here is the left-padding half of
`indexer_call`.

## Remaining limitation

At lengths where QSA kicks in (kv_len > indexer_budget), and with uneven prompt
lengths, the batched output does not match the output of running the sequences
one at a time. Block boundaries are determined by the column index within the
cache, whereas in the reference implementation the boundaries are anchored at
the start of the sequence, so unless the left padding is a multiple of
`indexer_compress_ratio` the grid shifts and the top-k selection changes. The
thing to be careful about is that this is not "broken": causality holds, and
padding columns are always dropped. Only the set of blocks chosen changes, and
since QSA is itself an approximation the output remains valid. Getting bit-exact
agreement would require shifting and repacking the indexer keys per sequence,
which amounts to a gather over the full kv length and is not worth the cost.

Verification lives in `tools/verify_batch_cache.py` (synthetic model, CPU only).
"""

from __future__ import annotations

_ENABLED = False
_ORIG = {}


# --------------------------------------------------------------- cache


def _install_indexer_cache_methods():
    """Graft the 4 `ArraysCache`-equivalent methods onto `_IndexerCache`.

    `self.keys` is `(B, T, head_dim)`. `update` concatenates along `axis=1`, so
    T is the column (token) axis and the batch axis is axis 0.
    """

    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q

    C = Q._IndexerCache

    def empty(self):
        return self.keys is None

    def filter(self, batch_indices):
        """Keep only the surviving rows (in-place)."""
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
        if getattr(self, "left_padding", None) is not None:
            self.left_padding = self.left_padding[batch_indices]

    def extract(self, idx, start=0, end=None):
        """Pull out a single sequence, slicing columns to `[start, end)` (which
        drops the left padding)."""
        out = C()
        if self.keys is not None:
            end = self.keys.shape[1] if end is None else end
            out.keys = mx.contiguous(self.keys[idx : idx + 1, start:end])
        return out

    def extend(self, other):
        """Stack rows vertically, aligned on the right edge (the newest token)
        (in-place)."""
        a, b = self.keys, other.keys
        if a is None and b is None:
            return
        ref = a if a is not None else b
        na = a.shape[1] if a is not None else 0
        nb = b.shape[1] if b is not None else 0
        n = max(na, nb)
        # The row count for the empty side is left behind by the caller
        # (`BatchAttnCache.extend`)
        ba = a.shape[0] if a is not None else getattr(self, "_pending_batch", 1)
        bb = b.shape[0] if b is not None else getattr(other, "_pending_batch", 1)

        def pad(x, bsz, length):
            if x is None:
                return mx.zeros((bsz, n) + ref.shape[2:], ref.dtype)
            if length < n:
                x = mx.pad(x, [(0, 0), (n - length, 0)] + [(0, 0)] * (x.ndim - 2))
            return x

        self.keys = mx.concatenate([pad(a, ba, na), pad(b, bb, nb)], axis=0)

    @classmethod
    def merge(cls, caches):
        """Bundle per-sequence caches together, aligned with left padding.

        The right edge is aligned under the same convention as
        `BatchKVCache.merge`. If column j on the KV side and column j of the
        indexer did not point at the same token, the block boundaries would
        shift and QSA's selection would break outright.
        """
        # Return early if everything is empty (same as ArraysCache.merge)
        if all(c is None or c.keys is None for c in caches):
            return cls()

        ref = next(c.keys for c in caches if c is not None and c.keys is not None)
        lengths = [
            0 if (c is None or c.keys is None) else c.keys.shape[1] for c in caches
        ]
        n = max(lengths)
        B = len(caches)
        keys = mx.zeros((B, n) + ref.shape[2:], ref.dtype)
        for i, (c, l) in enumerate(zip(caches, lengths)):
            if l:
                keys[i : i + 1, n - l :] = c.keys
        out = cls()
        out.keys = keys
        return out

    C.empty = empty
    C.filter = filter
    C.extract = extract
    C.extend = extend
    C.merge = merge
    C.left_padding = None


def _fix_arrays_cache_extract():
    """Fix `ArraysCache.extract` crashing on empty slots.

    Flash-Next's GDN layers use `ArraysCache(4)`, but the PLE and n-gram context
    (slots 2 and 3) are only used by the layers in `ple_layer_ids`. In the
    remaining layers those two stay `None`. `ArraysCache.extract` indexes into
    them unconditionally, so every time a single sequence finishes you get a
    `TypeError` (generate.py:1442 always calls `extract_cache` for each
    completed sequence).
    """

    from mlx_lm.models.cache import ArraysCache

    def extract(self, idx):
        out = ArraysCache(len(self.cache))
        out.cache = [None if c is None else c[idx : idx + 1] for c in self.cache]
        return out

    ArraysCache.extract = extract


def _make_batch_attn_cache():
    """Build a `BatchKVCache` that carries the indexer along with it.

    Every method of `BatchKVCache` only looks at keys/values, so this is a thin
    wrapper that applies the same operations to the indexer keys as well. How
    the columns are handled (roll / left-justify / zero-fill) must be exactly
    identical to the KV side.
    """

    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q
    from mlx_lm.models.cache import BatchKVCache, dynamic_roll

    class BatchAttnCache(BatchKVCache):
        def __init__(self, left_padding):
            super().__init__(left_padding)
            self.indexer = Q._IndexerCache()
            self._max_left_pad = max(left_padding) if len(left_padding) else 0
            if len(left_padding):
                self._sync_padding()

        # ---- left-padding sync --------------------------------------
        def _sync_padding(self):
            """Bring the max left padding down to the Python side and hand it
            to the indexer.

            Hitting `.item()` on every forward would insert a sync on every
            single token. Only count when the structure changes (prepare /
            finalize / filter / extend / merge).
            """
            self._max_left_pad = int(self.left_padding.max())
            if self._max_left_pad > 0:
                self.indexer.left_padding = self.left_padding
                k = self.indexer.keys
                if k is not None:
                    # Wipe the debris that `roll` wrapped around into the
                    # padding region. Leaving it there makes the QSA pool pick
                    # up keys from earlier tokens
                    live = mx.arange(k.shape[1])[None, :, None] >= (
                        self.left_padding[:, None, None]
                    )
                    self.indexer.keys = mx.where(live, k, 0)
            else:
                self.indexer.left_padding = None

        # ---- batch operations ---------------------------------------
        def prepare(self, **kwargs):
            super().prepare(**kwargs)
            self._sync_padding()

        def finalize(self):
            padding = self._right_padding
            super().finalize()
            if padding is not None and self.indexer.keys is not None:
                # KV is (B,H,T,D) with axis=2; the indexer is (B,T,D) with
                # axis=1. `dynamic_roll` derives the rank of `shifts` from the
                # axis, so pass it as 1-D rather than the KV side's
                # `padding[:, None]`
                self.indexer.keys = dynamic_roll(self.indexer.keys, padding, axis=1)
            self._sync_padding()

        def filter(self, batch_indices):
            before = self._idx
            super().filter(batch_indices)
            shift = before - self._idx  # columns dropped by left-justifying
            self.indexer.filter(batch_indices)
            if shift and self.indexer.keys is not None:
                self.indexer.keys = self.indexer.keys[:, shift:]
            self._sync_padding()

        def extend(self, other):
            # Hand over the batch width to use when filling the empty side
            self.indexer._pending_batch = self.left_padding.shape[0]
            other.indexer._pending_batch = other.left_padding.shape[0]
            super().extend(other)
            self.indexer.extend(other.indexer)
            self._trim_indexer()
            self._sync_padding()

        def _trim_indexer(self):
            """Match the indexer's column count to `_idx` (the KV side has
            slack because it is allocated in chunks)."""
            k = self.indexer.keys
            if k is None:
                return
            if k.shape[1] > self._idx:
                self.indexer.keys = k[:, k.shape[1] - self._idx :]
            elif k.shape[1] < self._idx:
                pad = self._idx - k.shape[1]
                self.indexer.keys = mx.pad(
                    k, [(0, 0), (pad, 0)] + [(0, 0)] * (k.ndim - 2)
                )

        def extract(self, idx):
            """Pull out a single sequence, converting it back to a plain
            `_AttnCache` (with its indexer attached)."""
            out = Q._AttnCache()
            padding = int(self.left_padding[idx])
            out.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding : self._idx])
            out.values = mx.contiguous(
                self.values[idx : idx + 1, :, padding : self._idx]
            )
            out.offset = out.keys.shape[2]
            out.indexer = self.indexer.extract(idx, padding, self._idx)
            return out

        @classmethod
        def merge(cls, caches):
            base = BatchKVCache.merge(caches)
            lengths = [c.size() for c in caches]
            padding = [max(lengths) - l for l in lengths]
            out = cls(padding)
            out.keys, out.values = base.keys, base.values
            out.offset, out.left_padding = base.offset, base.left_padding
            out._idx = base._idx
            out.indexer = Q._IndexerCache.merge([c.indexer for c in caches])
            out._trim_indexer()
            out._sync_padding()
            return out

        @property
        def state(self):
            return super().state + (self.indexer.keys,)

        @state.setter
        def state(self, v):
            *rest, ik = v
            BatchKVCache.state.fset(self, tuple(rest))
            self.indexer.keys = ik

    return BatchAttnCache


# --------------------------------------------------------- forward monkey patches


def _install_model_patches(BatchAttnCache):
    import math

    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q

    # ---- _AttnCache.merge --------------------------------------------
    # `KVCache.merge` returns a `BatchKVCache`. That is a different class with
    # no indexer, so the QSA state disappears the moment the caches are bundled.
    # This one spot has to be replaced. `filter` / `extract` never reach a
    # single-sequence cache (they are batch-side operations), and `KVCache` does
    # not have them either, so there is no risk of one silently passing through
    # via inheritance.
    @classmethod
    def _attn_merge(cls, caches):
        return BatchAttnCache.merge(caches)

    Q._AttnCache.merge = _attn_merge

    # ---- QSAIndexer -------------------------------------------------
    def indexer_call(self, x, rope, cache, offset: int, positions=None):
        """`offset` is the column position (for the visibility test);
        `positions` is the true position (for rope).

        When there is left padding the two do not agree. Block boundaries are
        determined by column, whereas the rope angle is determined by the
        position counted from the start of that sequence.
        """
        B, S, _ = x.shape
        qk = self.index_qk_proj(x)
        split = self.n_heads * self.head_dim
        q = qk[..., :split].reshape(B, S, self.n_heads, self.head_dim)
        raw_k = qk[..., split:].reshape(B, S, self.head_dim)

        if cache is not None:
            raw_k = cache.update(raw_k)
        kv_len = raw_k.shape[1]

        if kv_len <= self.token_budget:
            return None

        n_blocks = kv_len // self.compress_ratio
        pooled = raw_k[:, : n_blocks * self.compress_ratio].reshape(
            B, n_blocks, self.compress_ratio, self.head_dim
        )
        pooled = self.k_layernorm(
            pooled.astype(mx.float32).mean(axis=2).astype(raw_k.dtype)
        )

        left_pad = getattr(cache, "left_padding", None) if cache is not None else None
        block_starts = mx.arange(n_blocks) * self.compress_ratio
        block_pos = (
            block_starts[None, :]
            if left_pad is None
            else block_starts[None, :] - left_pad[:, None]
        )
        cos_k, sin_k = rope(block_pos)
        pooled = Q._rope_partial(pooled, cos_k, sin_k)

        q_col = mx.arange(offset, offset + S)
        q_pos = q_col[None, :] if positions is None else positions
        cos_q, sin_q = rope(q_pos)
        q = self.q_layernorm(q)
        q = Q._rope_partial(q, cos_q[:, :, None, :], sin_q[:, :, None, :])

        scores = mx.einsum(
            "bshd,bnd->bsnh", q.astype(mx.float32), pooled.astype(mx.float32)
        )
        scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(self.head_dim)

        block_end = block_starts + self.compress_ratio - 1
        visible = block_end[None, None, :] <= q_col[None, :, None]
        if left_pad is not None:
            # `block_starts >= left_pad` drops not just blocks that are
            # entirely padding but also any block whose start still straddles
            # the padding boundary (block_starts < left_pad <= block_end) --
            # safe side, since a partially-padded block would still mix
            # padding keys into `pooled`. Without this exclusion these blocks
            # would eat up the budget, and `pooled` would get contaminated
            # with padding keys
            visible = visible & (block_starts[None, None, :] >= left_pad[:, None, None])
        scores = mx.where(visible, scores, -mx.inf)

        k = min(self.block_topk, n_blocks)
        top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]

        keep_block = mx.zeros((B, S, n_blocks + 1), dtype=mx.bool_)
        top = mx.where(mx.take_along_axis(visible, top, axis=-1), top, n_blocks)
        keep_block = mx.put_along_axis(keep_block, top, mx.array(True), axis=-1)[
            ..., :n_blocks
        ]

        keep = mx.repeat(keep_block, self.compress_ratio, axis=-1)
        tail = kv_len - n_blocks * self.compress_ratio
        if tail:
            # Same as the vendored module: the incomplete tail is visible only
            # as far as causality allows (see the comment there for why the
            # stock `ones` leaks the future).
            tail_col = n_blocks * self.compress_ratio + mx.arange(tail)
            keep = mx.concatenate(
                [keep, mx.broadcast_to(
                    tail_col[None, None, :] <= q_col[None, :, None], (B, S, tail)
                )],
                axis=-1,
            )
        return keep[:, None]

    # ---- Attention: positions and mask only --------------------------
    #
    # The forward pass itself stays in the vendored module; only these two
    # seams differ. That way batch keeps whatever lands there (the `_wide_qkv`
    # fusion, the sdpa width-wall split).
    def attn_positions(self, cache, S):
        """Separate the column position from the true position; with left
        padding the two diverge.

          - column position (`size()`): which column of the cache this is.
            QSA's block boundaries and visibility test use this one
          - true position (`offset`): the number of tokens counted from the
            start of that sequence. The rope angle uses this one.
            `BatchKVCache.offset` is a (B,) array and is exactly the
            per-sequence true position

        Keys written during prefill are rotated by "the position within that
        sequence" (prefill runs with right padding, and finalize only
        rearranges the layout). Rotating the decode queries by column position
        would shift the origin by the amount of padding and break rope's
        relative property.
        """
        if cache is None:
            return 0, mx.arange(S)[None]
        if isinstance(cache.offset, mx.array):
            return cache.size(), cache.offset[:, None] + mx.arange(S)
        return cache.size(), mx.arange(cache.offset, cache.offset + S)[None]

    def attn_final_mask(self, mask, sparse, cache, S, dtype):
        if sparse is None or mask is None or isinstance(mask, str):
            # Nothing to combine: fall back to the stock rule (when sparse is
            # present, causal is thrown away)
            return _ORIG["Attention._final_mask"](self, mask, sparse, cache, S, dtype)
        # A bool mask that carries left padding; take the conjunction here.
        # 2026-09-02: stays bool (stock rule no longer converts to additive).
        return mask & sparse

    # ---- Qwen4ExpModel: actually distribute the mask ------------------
    def make_masks(self, h, cache):
        full_idx = [
            i for i, l in enumerate(self.layers) if l.layer_type == "full_attention"
        ]
        attn_cache = cache[full_idx[0]] if full_idx else None
        if attn_cache is not None and getattr(attn_cache, "_max_left_pad", 0) > 0:
            # Only build an array mask when there is left padding. Otherwise
            # take the stock path through unchanged (the "causal" string /
            # None)
            mask = attn_cache.make_mask(h.shape[1])
        else:
            mask, _ = _ORIG["Qwen4ExpModel._make_masks"](self, h, cache)

        lin_idx = [
            i for i, l in enumerate(self.layers) if l.layer_type == "linear_attention"
        ]
        lin_cache = cache[lin_idx[0]] if lin_idx else None
        lengths = getattr(lin_cache, "lengths", None) if lin_cache is not None else None
        conv_mask = None
        if lengths is not None:
            # Prefill under right padding. Keep padding columns out of the
            # recurrent state
            conv_mask = mx.arange(h.shape[1])[None] < lengths[:, None]
        return mask, conv_mask

    _ORIG["QSAIndexer"] = Q.QSAIndexer.__call__
    _ORIG["Attention._final_mask"] = Q.Attention._final_mask
    _ORIG["Attention._positions"] = Q.Attention._positions
    _ORIG["Qwen4ExpModel._make_masks"] = Q.Qwen4ExpModel._make_masks

    Q.QSAIndexer.__call__ = indexer_call
    Q.Attention._positions = attn_positions
    Q.Attention._final_mask = attn_final_mask
    Q.Qwen4ExpModel._make_masks = make_masks


# ------------------------------------------------------------------ activation

BatchAttnCache = None


def enable_batch_cache() -> None:
    """Make Flash-Next work under batch generation. A no-op after the first
    call."""

    global _ENABLED, BatchAttnCache
    if _ENABLED:
        return

    _install_indexer_cache_methods()
    _fix_arrays_cache_extract()
    BatchAttnCache = _make_batch_attn_cache()
    _install_model_patches(BatchAttnCache)
    _ENABLED = True


def disable_batch_cache() -> None:
    global _ENABLED
    if not _ENABLED:
        return
    import mlx_lm.models.qwen4_exp as Q

    Q.QSAIndexer.__call__ = _ORIG["QSAIndexer"]
    Q.Attention._positions = _ORIG["Attention._positions"]
    Q.Attention._final_mask = _ORIG["Attention._final_mask"]
    Q.Qwen4ExpModel._make_masks = _ORIG["Qwen4ExpModel._make_masks"]
    # Leaving `merge` in place would mean building indexer-carrying batch
    # caches while the forward pass has been restored to stock. Fall back to
    # the inherited one (KVCache.merge)
    del Q._AttnCache.merge
    _ENABLED = False


# ------------------------------------------------------------- coordinator
#
# Wires the above patches into server.py as opt-in continuous batching
# (BACKLOG.md §2, Kimi K3 review item 11). Restricted to the FallbackRunner
# path only: SpecRunner/FlashSpecRunner keep their own draft-token state
# machine, which this module has never touched and which server.py forbids
# editing for this change (spec.py / spec_flash.py / mtp*.py are the
# measured, verified speculative engines). Whether continuous batching can
# coexist with speculative drafting is unmeasured, so it simply isn't
# attempted here — a request only ever reaches this coordinator when the
# runner resolved for it is a FallbackRunner (see runner.should_batch).
#
# Bit-exact agreement between batched and solo generation is not the bar.
# `mx.quantized_matmul` rounds differently depending on the total batch
# length passed to one call — measured independently in this repo for
# prefill chunking (commit 963c868) and confirmed here for request batching
# (tools/verify_batch_real.py --mode kld): even equal-length, padding-free
# batches (`short-eq`) diverge from the solo reference after a few dozen
# greedy-decoded tokens. This is an MLX property, not a bug in this module,
# and neither vLLM nor llama.cpp promise cross-configuration bit-exactness
# either. What this module guarantees instead:
#
#   1. Determinism within one fixed batch composition (same inputs, same
#      co-resident sequences -> same output, every time).
#   2. Fluent output (no word-level corruption, no language switching, no
#      repeat loops) on every configuration, including the QSA/uneven-length
#      one below.
#   3. next-token KLD against solo generation, while both paths still agree
#      (i.e. still conditioned on an identical history), at the same order
#      of magnitude as this project's own quantization noise floor (v-fast6:
#      0.00378) for every configuration this module will ever actually
#      admit into a shared batch.
#
# `classify()` below only ever sees one request at a time: `prompt_len +
# max_tokens` against `indexer_budget`. What actually determines whether QSA
# can activate for a shared batch is the batch's *physical* column count
# (`BatchKVCache._idx` = the longest co-resident prompt plus elapsed decode
# steps) — two requests can each individually classify "pool" (own
# prompt_len + max_tokens <= budget) and still, once co-resident, push
# max(prompt_len) + max(max_tokens) across the batch past budget — e.g.
# A = 2040+8 and B = 8+2040 against a 2048 budget: each is "pool" alone, but
# sharing a batch gives a physical column count of 2040 + 2040 = 4080 > 2048
# (B-3). `classify()` stays the per-request heuristic it always was — it
# cannot see who else will be resident. The actual guarantee (max(prompt_len)
# + max(max_tokens) across the co-resident set never exceeds budget) is
# enforced where that set is actually known: `_drive()`'s pool admission loop
# (`_pool_admission_fits`) refuses to add a "pool" admission whenever doing so
# would push the combined figure over budget, leaving it in `pending_pool`
# until enough of the live pool has drained.
#
# QSA's block grid is cut by absolute column position, so an unequal-length,
# left-padded batch can select a different set of blocks than solo
# generation would (mlxturbo/batch.py's own "Remaining limitation" section,
# above) — not corruption, but a real selection difference, and the one
# case in tools/verify_batch_real.py --mode kld where the aligned-window KLD
# ran measurably above the noise floor (long-uneq, B=4: up to ~4.4x
# 0.00378 for one sequence). Everything that provably can never reach QSA
# (`pool` tier) may share a batch freely up to --max-batch; the "solo" tier
# is deliberately more conservative than the strict correctness condition
# (which only forbids *unequal*-length QSA-risk combinations, not equal-length
# ones) because tracking "would this still be equal-length by the time X
# joins" across a continuously-refilling batch is a correctness-sensitive
# bookkeeping problem this pass chose not to take on; the cost is one
# forgone speed opportunity (equal-length long prompts always run one at a
# time here), not a correctness gap — classify()'s solo/pool split is sound
# again now that the co-resident-budget gap above is closed at admission time.
#
# ---- admission timing (docs/research/LANES-2026-09.md レーン 5, 2026-09-02) --
#
# There has never been an artificial "wait for a companion" window here (that
# exists only in `batch_spec.py`'s `BatchSpecCoordinator`, and for a
# different reason — see that module). The first admission of either tier
# starts on the very next `_drive()` iteration, no delay. What changed in
# lane 5 is how often a *later* same-tier arrival gets a chance to join an
# already-running generator: `_drive()` used to hand control to
# `gen.next_generated()`, which internally loops `BatchGenerator._next()`
# until it has produced at least one decode token — during that internal
# loop (which can span several prefill chunks while the first row's own
# prompt is still being processed) the inbox is not visited. `_drive()` now
# calls `gen.next()` directly, one `_next()` tick per outer-loop iteration,
# so the inbox — and therefore admission of a newly-arrived same-tier row —
# is checked every tick, including ticks that are pure prefill and produce
# no decode token yet. `_next()` itself already interleaves one decode round
# for whatever is in `_generation_batch` with one `prefill_step_size`-sized
# prefill chunk for whatever is in `_prompt_batch` (mlx_lm's own design, see
# `PromptProcessingBatch`/`GenerationBatch` in `mlx_lm.generate`); this
# module does not reimplement that interleaving, only exposes its natural
# tick boundary to the scheduler above. `default_join_prefill_chunk()` below
# controls how big that per-tick prefill chunk is (512 by default,
# `MLXTURBO_JOIN_PREFILL_CHUNK` to override) — smaller than
# `spec.PREFILL_STEP_SIZE` (2048) because this chunk size doubles as "how
# long a newcomer's own prefill can hold up the next decode round for every
# other co-resident row".

import os
import queue as _queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


def default_join_prefill_chunk() -> int:
    """Default ``prefill_step_size`` for :class:`BatchCoordinator`'s own
    ``BatchGenerator`` (docs/research/LANES-2026-09.md レーン 5, 2026-09-02).

    ``mlxturbo.spec.PREFILL_STEP_SIZE`` (2048) is tuned for a single prompt's
    own prefill throughput and was, until this change, also what this
    coordinator used. Under continuous batching that value doubles as the
    chunk size a newly-joining row's own prefill is cut into (see `_drive`'s
    per-tick loop below) -- a large chunk here means a newcomer's first
    prefill step can hold up every already-decoding row's next token for a
    correspondingly long stretch. 512 (vllm-mlx's own default for this same
    role) keeps that stall bounded while still being large enough that a
    short prompt still finishes in one chunk. ``MLXTURBO_JOIN_PREFILL_CHUNK``
    overrides it -- the one new knob this change adds (CLAUDE.md)."""

    try:
        v = int(os.environ.get("MLXTURBO_JOIN_PREFILL_CHUNK", "512"))
    except ValueError:
        v = 512
    return max(1, v)


def _indexer_budget(model) -> int | None:
    """None means "this architecture has no QSA / no such notion" (anything
    other than qwen4_exp) — every request is then always poolable."""

    from .arch import indexer_budget

    return indexer_budget(model)


def classify(model, prompt_len: int, max_tokens: int) -> str:
    """"solo": this request's own kv length could exceed indexer_budget
    before it is done, so QSA could activate for it — it must never share a
    batch with anything else (see the module docstring above). "pool": QSA
    cannot activate from this request's own length alone.

    Per-request only (B-3, see the module docstring above): two "pool"
    requests can still push a shared batch's physical column count (longest
    co-resident prompt + elapsed steps) past `indexer_budget` once they are
    actually co-resident. `classify()` cannot see who else will be resident,
    so it does not by itself rule that out — `BatchCoordinator._drive()`'s
    pool admission loop (`_pool_admission_fits`) enforces the actual
    guarantee where the co-resident set is known."""

    budget = _indexer_budget(model)
    if budget is None:
        return "pool"
    return "solo" if (prompt_len + max_tokens) > budget else "pool"


def _pool_admission_fits(
    candidate: "Admission", resident: list, budget: int | None
) -> bool:
    """B-3: whether adding ``candidate`` to the already-live ``resident``
    pool admissions keeps the co-resident set's guarantee intact —
    max(prompt_len) + max(max_tokens), taken across ``resident + [candidate]``,
    must stay within ``budget`` (``None`` means the architecture has no QSA,
    so anything fits). ``classify()`` already proved this for ``candidate``
    alone; this is the check it cannot make on its own (no visibility into
    who else is resident)."""

    if budget is None or not resident:
        return True
    rows = resident + [candidate]
    longest_prompt = max(len(a.prompt_ids) for a in rows)
    longest_tokens = max(a.max_tokens for a in rows)
    return longest_prompt + longest_tokens <= budget


@dataclass
class Admission:
    """One request waiting on (or being served by) the coordinator.

    ``on_tokens``: called once per generated token, on the coordinator's
    worker thread, exactly like a synchronous ``Runner.generate``'s own
    on_tokens callback (``None`` for a non-streaming caller). It is expected
    to be cheap, thread-safe, and never touch MLX arrays (the existing
    server.py closure only appends to plain lists/queues and reads a
    ``threading.Event`` — see ``_start_generation``).

    ``on_done``: called exactly once, on the coordinator's worker thread,
    with ``("done", res)`` / ``("cancelled", None)`` / ``("error", exc)`` —
    the same three-outcome protocol server.py's own worker() already speaks
    over its queue. ``None`` for a non-streaming caller, which reads the
    outcome off ``future`` directly instead.

    ``future``: a plain ``concurrent.futures.Future`` (thread-safe to
    complete from the coordinator's worker thread and observe via
    ``asyncio.wrap_future`` on the caller's own event loop). Always resolved
    with the ``res`` dict (or ``None`` on cancel/error) — never with an
    exception, matching ``_start_generation``'s existing worker(), whose own
    Future never raises either (every outcome, including errors, is
    reported through the queue instead so a caller waiting on the Future
    only ever needs to know "the request is done").
    """

    prompt_ids: list[int]
    max_tokens: int
    sampler: Callable
    logits_processors: list
    tier: str
    on_tokens: Callable[[list[int]], None] | None
    on_done: Callable[[str, Any], None] | None
    cancel_event: "threading.Event | None"
    future: "Any"  # concurrent.futures.Future
    uid: int | None = None
    tokens: list = field(default_factory=list)
    t0: float | None = None
    ttft: float | None = None


class BatchCoordinator:
    """Continuous-batching worker. Owns nothing until the first admission;
    from then on it runs its drive loop on ``executor`` (the single MLX
    worker thread — see the docstring at the top of server.py) only while it
    has live or pending work, so it never starves the other things already
    submitted to that same executor (session-pool lookups, non-batched
    generate calls for solo-tier or spec-path requests).
    """

    def __init__(self, model, executor, max_batch: int, prefill_step_size: int, eos_ids):
        self.model = model
        self.executor = executor
        self.max_batch = max(1, max_batch)
        self.prefill_step_size = prefill_step_size
        self.eos_ids = list(eos_ids)
        self._inbox: "_queue.SimpleQueue[Admission]" = _queue.SimpleQueue()
        self._guard = threading.Lock()
        self._active = False

    def submit(self, admission: Admission) -> None:
        """Thread-safe; callable from any asyncio task. Enqueues the
        admission and, if no drive loop is currently running, starts one."""

        self._inbox.put(admission)
        with self._guard:
            if not self._active:
                self._active = True
                self.executor.submit(self._drive)

    # ---- runs on the single MLX worker thread ---------------------------

    def _complete(self, adm: "Admission", res=None, cancelled=False, error=None) -> None:
        if adm.on_done is not None:
            # Streaming: errors/cancellation are reported through the queue
            # (mirroring _start_generation's worker() exactly), so the
            # Future itself never raises — it exists only so the caller
            # knows the worker has finished.
            if error is not None:
                adm.on_done("error", error)
            elif cancelled:
                adm.on_done("cancelled", None)
            else:
                adm.on_done("done", res)
            adm.future.set_result(res)
        else:
            # Non-streaming: there is no queue, so the Future is the only
            # channel — an error must actually raise for the caller (e.g.
            # _run_generate_batched) to see it, exactly as a real exception
            # from runner.generate() would propagate through
            # loop.run_in_executor on the non-batched path.
            if error is not None:
                adm.future.set_exception(error)
            else:
                adm.future.set_result(res)

    def _build_res(self, adm: "Admission") -> dict:
        decode_time = 0.0
        if adm.t0 is not None and adm.ttft is not None:
            decode_time = max(0.0, time.perf_counter() - adm.t0 - adm.ttft)
        n_decode = max(len(adm.tokens) - 1, 0)
        # tokens_per_step は mlxturbo.spec.SpecEngine / FlashSpecSession と
        # 同じ定義 (n_decode / steps) で運ぶ。今この Admission には steps を
        # 数える者がいない — 現行の BatchCoordinator は FallbackRunner の
        # continuous batching のみを駆動し (投機なし、1 トークン = 1
        # ラウンド)、admission/スケジューリング側はこの変更の対象外なので
        # ここでは増分しない。将来ここに投機エンジンを配線する変更が
        # Admission に `steps` を足して forward パスのラウンドごとに
        # 増分するようになれば、getattr 経由でそのまま実測の tok/step が
        # 出るようになる。今は存在しないので 1.0 (= stock の 1 トークン/
        # ラウンドと正しく一致する値) にフォールバックする。
        steps = getattr(adm, "steps", None)
        tokens_per_step = (n_decode / steps) if steps else 1.0
        return {
            "tokens": adm.tokens,
            "ttft_s": adm.ttft or 0.0,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            "prefill_reused": 0,
            "prefill_new": len(adm.prompt_ids),
            "tokens_per_step": tokens_per_step,
        }

    def _deliver_token(self, adm: "Admission", token: int) -> None:
        if adm.ttft is None:
            adm.ttft = time.perf_counter() - (adm.t0 or time.perf_counter())
        adm.tokens.append(token)
        if adm.on_tokens is not None:
            adm.on_tokens([token])

    def _drive(self) -> None:
        from mlx_lm.generate import BatchGenerator

        gen = None
        live: dict[int, Admission] = {}
        mode: str | None = None  # None | "solo" | "pool"
        pending_solo: list[Admission] = []
        pending_pool: list[Admission] = []

        def new_gen():
            return BatchGenerator(
                self.model,
                stop_tokens=[[t] for t in self.eos_ids],
                prefill_step_size=self.prefill_step_size,
            )

        def admit(adm: Admission, tier: str) -> None:
            nonlocal gen, mode
            if adm.cancel_event is not None and adm.cancel_event.is_set():
                self._complete(adm, cancelled=True)
                return
            if gen is None:
                gen = new_gen()
            uid = gen.insert(
                [adm.prompt_ids],
                [adm.max_tokens],
                samplers=[adm.sampler],
                logits_processors=[adm.logits_processors],
            )[0]
            adm.uid = uid
            adm.t0 = time.perf_counter()
            live[uid] = adm
            mode = tier

        try:
            while True:
                while True:
                    try:
                        adm = self._inbox.get_nowait()
                    except _queue.Empty:
                        break
                    (pending_solo if adm.tier == "solo" else pending_pool).append(adm)

                if mode in (None, "pool"):
                    # B-3: 各候補は「今すでに live な pool 行 + 自分」の
                    # max(prompt_len) + max(max_tokens) が budget に収まる
                    # ときだけ入れる (classify() は自分の値しか見ないので、
                    # 同居する相手を知らない)。収まらない候補は先頭で止めて
                    # pending_pool に残す -- pool が空くまで待たせれば、次に
                    # 試すときは resident が減って通る可能性が出る。
                    budget = _indexer_budget(self.model)
                    resident = list(live.values())
                    while (
                        pending_pool
                        and len(live) < self.max_batch
                        and _pool_admission_fits(pending_pool[0], resident, budget)
                    ):
                        adm = pending_pool.pop(0)
                        admit(adm, "pool")
                        resident.append(adm)
                if mode is None and not live and pending_solo:
                    admit(pending_solo.pop(0), "solo")

                if not live:
                    if pending_solo or pending_pool:
                        # Something is waiting for the other tier to fully
                        # drain (or for cancellation to clear it above) —
                        # spin once more rather than exiting.
                        continue
                    break

                # 1 tick = BatchGenerator._next() 1 回。すでに `mode` を
                # "pool"/"solo" に固定して以降の毎ラウンド、上のブロックで
                # inbox を非ブロッキングで見て新入りを admit している。ここを
                # `gen.next_generated()` にすると、その内部の while ループが
                # 「生成トークンが出るまで」何回でも `_next()` を回してしまい
                # (最初の 1 行がまだ複数チャンクの prefill 中でトークンが
                # 1 つも出ていない間) inbox に戻れない。`gen.next()` を直接
                # 1 tick だけ呼ぶことで、prefill 中の tick でも (生成
                # トークンが 0 本の ROUND でも) 上のループに戻って inbox を
                # 見られるようにする -- vllm-mlx 型の「毎 tick の途中参加」。
                # `_next()` 自体は decode 1 ラウンドと prefill 最大
                # `prefill_step_size` トークンを 1 回の呼び出しに同居させる
                # ので、interleave の中身自体は元から BatchGenerator 任せで
                # 変わっていない (mlxturbo/batch.py が新たに書いたコードでは
                # ない)。
                _prompt_responses, generation_responses = gen.next()
                for r in generation_responses:
                    adm = live.get(r.uid)
                    if adm is None:
                        continue
                    try:
                        if r.finish_reason != "stop":
                            self._deliver_token(adm, r.token)
                    except BaseException as exc:  # noqa: BLE001 - isolate one uid's callback failure from the rest of the batch
                        del live[r.uid]
                        gen.remove([r.uid])
                        cancelled = adm.cancel_event is not None and adm.cancel_event.is_set()
                        self._complete(adm, cancelled=cancelled, error=None if cancelled else exc)
                        continue
                    if r.finish_reason is not None:
                        del live[r.uid]
                        self._complete(adm, res=self._build_res(adm))

                if not live:
                    mode = None
        except BaseException as exc:  # noqa: BLE001 - never leave a caller's Future unresolved on an internal bug
            for adm in list(live.values()) + pending_solo + pending_pool:
                self._complete(adm, error=exc)
        finally:
            if gen is not None:
                gen.close()
            with self._guard:
                self._active = False
            # A late arrival could have landed between the last empty-check
            # above and clearing _active; re-check under the lock so it is
            # never stranded in the inbox with nothing to ever pick it up.
            if not self._inbox.empty():
                with self._guard:
                    if not self._active:
                        self._active = True
                        self.executor.submit(self._drive)


__all__ = [
    "Admission",
    "BatchCoordinator",
    "classify",
    "disable_batch_cache",
    "enable_batch_cache",
]
