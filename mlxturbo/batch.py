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
   all save their state by slicing off the "last k columns". During prefill with
   right padding the tail is padding, so the recurrent state of a short sequence
   gets filled with padding. `qwen3_next` avoids this by pointing at the right
   position via `cache.lengths` (models/qwen3_next.py:263-269). `qwen4_exp` has
   nothing of the sort.

6. Rope positions are counted by column index. Prefill runs with right padding,
   so the keys are rotated by "the position within that sequence". Once finalize
   has rearranged the layout to left padding, rotating the decode queries by
   column index shifts the origin by the amount of padding and breaks rope's
   relative property. The true per-sequence position is already held by
   `BatchKVCache.offset` (an array), so use that.

7. `ArraysCache.extract` crashes on empty slots (on the mlx_lm side).
   Flash-Next's GDN layers use 2 of the 4 slots (PLE / n-gram) only in PLE
   layers, so every time a single sequence finishes you get a `TypeError`.

## What was left alone

When there is no padding (B=1, or a batch of equal lengths) the same graph runs
as in the stock implementation. Both the mask construction and the indexer's
visibility test are written so that if `left_padding` is all zeros they are
character-for-character identical to stock.

## What was changed on purpose (output differs from stock)

QSA's "the sub-block tail is always visible" was made causal (there is a comment
at the relevant spot in `indexer_call`). In the stock implementation this region
is visible to every query, so the output changes depending on how prefill is
split. `BatchGenerator` always splits the prompt into (n-1, 1), and on top of
that it advances each sequence in `prefill_step_size` increments, so the split
cannot be pinned down. This is why it is not enabled by default (only a process
that calls `enable_batch_cache()` is affected).

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
    import mlx.nn as nn
    import mlx_lm.models.qwen4_exp as Q
    from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

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
            # Exclude blocks that are entirely padding from the candidates.
            # Without this they eat up the budget, and on top of that `pooled`
            # gets contaminated with padding keys
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
            # The stock implementation sets this to `ones` (always visible). The
            # intent is that the sub-block tail is "always visible", but when
            # `sparse` is non-None, Attention throws away the causal mask and
            # uses only `sparse` (qwen4_exp.py:313-318), so up to
            # compress_ratio-1 trailing columns become visible to every query.
            # That is why splitting prefill as 19+1 changes the output — and
            # BatchGenerator always splits as (n-1, 1). Fold causality in here
            # so that `sparse` is causal on its own.
            tail_col = n_blocks * self.compress_ratio + mx.arange(tail)
            keep = mx.concatenate(
                [keep, mx.broadcast_to(
                    tail_col[None, None, :] <= q_col[None, :, None], (B, S, tail)
                )],
                axis=-1,
            )
        return keep[:, None]

    # ---- Attention ---------------------------------------------------
    def attention_call(self, x, rope, mask, cache, idx_cache):
        B, S, _ = x.shape
        # Separate the column position from the true position; with left
        # padding the two diverge.
        #   - column position (`size()`): which column of the cache this is.
        #     QSA's block boundaries and visibility test use this one
        #   - true position (`offset`): the number of tokens counted from the
        #     start of that sequence. The rope angle uses this one.
        #     `BatchKVCache.offset` is a (B,) array and is exactly the
        #     per-sequence true position
        #
        # Keys written during prefill are rotated by "the position within that
        # sequence" (prefill runs with right padding, and finalize only
        # rearranges the layout). Rotating the decode queries by column position
        # would shift the origin by the amount of padding and break rope's
        # relative property
        offset = cache.size() if cache is not None else 0
        if cache is None:
            positions = mx.arange(S)[None]
        elif isinstance(cache.offset, mx.array):
            positions = cache.offset[:, None] + mx.arange(S)
        else:
            positions = mx.arange(cache.offset, cache.offset + S)[None]

        sparse = self.indexer(x, rope, idx_cache, offset, positions)

        q, gate = mx.split(self.q_proj(x).reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        cos, sin = rope(positions)
        cos, sin = cos[:, None], sin[:, None]
        q, k = Q._rope_partial(q, cos, sin), Q._rope_partial(k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if sparse is not None:
            neg = mx.finfo(q.dtype).min if hasattr(mx, "finfo") else -1e9
            zero, negv = mx.array(0, q.dtype), mx.array(neg, q.dtype)
            if mask is None or isinstance(mask, str):
                # Same as the stock implementation: when sparse is present,
                # throw away causal
                mask = mx.where(sparse, zero, negv)
            else:
                # A bool mask that carries left padding; take the conjunction
                # here
                mask = mx.where(mask & sparse, zero, negv)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))

    # ---- GatedDeltaNet: slice the conv state at the real length -------
    def gdn_call(self, x, mask, cache):
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
            n_keep = self.conv_kernel_size - 1
            lengths = getattr(cache, "lengths", None)
            if lengths is None:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
            else:
                # Under right padding the tail is padding. Point at the n_keep
                # columns immediately following the real length
                ends = mx.clip(lengths, 0, S)
                pos = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, pos, axis=1)
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.n_k, self.dk)
        k = k.reshape(B, S, self.n_k, self.dk)
        v = v.reshape(B, S, self.n_v, self.dv)

        inv_scale = self.dk**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        state = cache[1] if cache is not None else None
        out, state = Q.gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))

    # ---- PLELayer: slice the short-conv state at the real length too --
    def ple_short_conv(self, x, cache):
        S = x.shape[1]
        n = self.short_conv_state_len
        state = (
            cache[2]
            if (cache is not None and cache[2] is not None)
            else mx.zeros((x.shape[0], n, x.shape[-1]), dtype=x.dtype)
        )
        full = mx.concatenate([state, x], axis=1)
        if cache is not None:
            lengths = getattr(cache, "lengths", None)
            if lengths is None:
                cache[2] = mx.contiguous(full[:, -n:, :])
            else:
                ends = mx.clip(lengths, 0, S)
                pos = (ends[:, None] + mx.arange(n))[..., None]
                cache[2] = mx.take_along_axis(full, pos, axis=1)
        return nn.silu(self.conv1d(full[:, -(n + S) :, :]))

    # ---- Qwen4ExpModel: actually distribute the mask ------------------
    def model_call(self, ids, cache=None, input_embeddings=None):
        h = self.embed_tokens(ids) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)

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
            mask = create_attention_mask(
                h, [attn_cache] if attn_cache is not None else None
            )

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

        prev_ctx = None
        if self.ple_layers:
            ctx_len = self.args.ngram_size - 1
            eos = self.args.eos_token_id
            eos = eos[0] if isinstance(eos, list) else eos
            pc = cache[self.ple_layers[0]]
            prev = pc[3] if pc is not None else None
            prev_ctx = (
                prev
                if prev is not None
                else mx.full((ids.shape[0], ctx_len), eos, ids.dtype)
            )
            if pc is not None:
                cat = mx.concatenate([prev_ctx, ids], axis=1)
                pc_lengths = getattr(pc, "lengths", None)
                if pc_lengths is None:
                    pc[3] = cat[:, -ctx_len:]
                else:
                    ends = mx.clip(pc_lengths, 0, ids.shape[1])
                    pos = ends[:, None] + mx.arange(ctx_len)
                    pc[3] = mx.take_along_axis(cat, pos, axis=1)

        h = mx.tile(h, (1, 1, self.hc))
        for layer, c in zip(self.layers, cache):
            idx_c = c.indexer if (c is not None and hasattr(c, "indexer")) else None
            h = layer(h, self.rope, mask, conv_mask, c, idx_c, ids, prev_ctx)
        return self.hyper_connection_mixer(h)

    _ORIG["QSAIndexer"] = Q.QSAIndexer.__call__
    _ORIG["Attention"] = Q.Attention.__call__
    _ORIG["GatedDeltaNet"] = Q.GatedDeltaNet.__call__
    _ORIG["PLELayer"] = Q.PLELayer._short_conv
    _ORIG["Qwen4ExpModel"] = Q.Qwen4ExpModel.__call__

    Q.QSAIndexer.__call__ = indexer_call
    Q.Attention.__call__ = attention_call
    Q.GatedDeltaNet.__call__ = gdn_call
    Q.PLELayer._short_conv = ple_short_conv
    Q.Qwen4ExpModel.__call__ = model_call


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
    Q.Attention.__call__ = _ORIG["Attention"]
    Q.GatedDeltaNet.__call__ = _ORIG["GatedDeltaNet"]
    Q.PLELayer._short_conv = _ORIG["PLELayer"]
    Q.Qwen4ExpModel.__call__ = _ORIG["Qwen4ExpModel"]
    # Leaving `merge` in place would mean building indexer-carrying batch
    # caches while the forward pass has been restored to stock. Fall back to
    # the inherited one (KVCache.merge)
    del Q._AttnCache.merge
    _ENABLED = False


__all__ = ["disable_batch_cache", "enable_batch_cache"]
