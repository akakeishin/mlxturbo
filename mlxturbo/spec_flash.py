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

import os
import time
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
from .prefill_common import split_and_checkpoint_tail
from . import arch as _archmod
from .arch import qwen4_arch as _arch


# How many trailing prompt positions are fed to the MTP head before decoding
# starts (see FlashSpecEngine._prime_draft_cache). Acceptance comes from recent
# context, so a window buys most of the gain while keeping the cost independent
# of prompt length -- this model's ceiling is 262144 tokens, where priming the
# whole prompt would cost both minutes of TTFT and gigabytes of retained hyper
# state. Measured at 2048: 32k prompt, acceptance 0.574 -> 0.827.
PRIME_WINDOW = 2048
# Trailing prefill chunks whose hyper state generate_stream retains. Two chunks
# of PREFILL_STEP_SIZE always cover PRIME_WINDOW+1 positions.
HYPER_KEEP_CHUNKS = 2

# ドラフトを 1 ラウンドで何トークン引くか。1 = ヘッドを 1 回だけ回す。
# 2 以上ではヘッド自身の hyper 状態を次段に渡して連鎖させる (_draft_chain)。
# 受理率 r で depth d なら 1 ラウンドの期待トークン数は (1-r^(d+1))/(1-r) で、
# d=1 は r=0.83 でも 1.83 が上限になる。
#
# 既定 2 (2026-08-31 の掃引、ddalcu 一律 4bit・複数プロンプト x 512 トークン):
# depth 1/2/3/4 の短文脈 decode は 46.8 / 53.1-53.6 / 47-48 / 45.1。
# depth 3 は 3 本目の的中が verify の位置追加費用 (~5ms) を償却しない。
MTP_DEPTH = 2

# ここを越えたら depth を 1 に落とす。深くすると検証フォワードの位置数が増え、
# その費用は文脈長に比例する (indexer のブロック選択が長いキャッシュ全体に
# 対して位置ごとに走るため) ので、長文では利得を食い潰して逆に遅くなる。
#
# v-l / M3 Max のサーバー実測 (tok/s、温まった状態):
#
#   文脈    depth 1   depth 2   depth 3
#    1k      45.4       —        51.5
#    4k      37.0       —        45.5
#    6k      36.3      39.7      45.7
#    8k      38.1      36.6      35.4     <- ここで反転済み
#   12k      37.3      29.8      33.3
#   48k      30.8       —        17.6
#
# 反転は 6k と 8k の間。勝っている側の内側を採って 6144 (= 3 チャンク) に置く。
#
# 2026-09-01 再較正 (1): 6144 は sdpa の 32 行の壁 (qL>=3 が未融合経路に
# 落ちる) を避けるための遺物と判断し、いったん実質無効 (262144) に置いた。
#
# 2026-09-01 再較正 (2、こちらが現行): 複数プロンプト x 512 の回文順掃引で
# 測り直したところ、**長文では depth 1 が勝つ**とはっきり出た
# (tools/decode_ab.py --knob depth、bench/results/depth-*.json)。
# ms/token、depth 2 を基準にした差:
#
#   文脈    depth 1   depth 2   depth 3
#    65 tok  +5.6%     基準     +11.7%
#   900      +5.1%     基準      +2.6%
#   2.6k     -3.3%     基準      +9.9%
#   4k       -3.1%     基準      +8.2%
#   17k     -10.9%     基準      +3.2%
#
# 反転は 900 と 2.6k の間にあり、そこには QSA が働き始める境界
# (indexer_budget = 2048) がある。機構としても符合する — QSA が活性だと
# 検証フォワードに 1 位置足す費用にブロック選択と疎マスクが乗り、受理が
# 増えるぶんを償却しなくなる。tok/round は深いほど上がり続ける (17k で
# 1.64 / 1.97 / 2.33) のに壁時計は逆、というのがこの現象の形。
#
# よって既定はモデルの indexer_budget にする (下の _depth_ctx_limit)。
# 定数を持たない族では境界が無いので切り替えない。env で上書きできる。
DEPTH_CONTEXT_LIMIT = int(os.environ.get("MLXTURBO_DEPTH_CTX_LIMIT", "0")) or None
_DEPTH_CTX_LIMIT_FALLBACK = 262144


def _depth_ctx_limit(model) -> int:
    """このモデルで depth を 1 に落とす文脈長。

    env の指定が最優先。無ければ疎注意の境界 (indexer_budget) を使う。
    境界を持たない族では切り替えない (モデルの文脈上限に置く)。
    """
    if DEPTH_CONTEXT_LIMIT:
        return DEPTH_CONTEXT_LIMIT
    from .arch import indexer_budget

    return indexer_budget(model) or _DEPTH_CTX_LIMIT_FALLBACK


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
        mixed_qkv, z, b, a = self._project_in(x)
        z = z.reshape(B, S, self.n_v, self.dv)

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




# 段階投入の間隔 (層数)。0 で無効 = 一括構築。2026-08-31 の掃引: 16/12/8/6/4/3/2
# で 53.9/55.0/57.6/57.8/58.3/58.7 tok/s (probe 実測、出力は全て一致)。既定 2。
_STAGE_EVERY = int(os.environ.get("MLXTURBO_STAGE_EVERY", "2") or 0)

# MLXTURBO_ROUND_TRACE=1: ラウンド内の CPU 側区間を ms で刻む (調査用)
_ROUND_TRACE = os.environ.get("MLXTURBO_ROUND_TRACE") == "1"


def _staged_forward(model, ids, caches):
    """Model.__call__ と同じ計算を、途中の hidden を async_eval しながら組む。

    グラフを 48 層ぶん組み切ってから投げると、構築中 (7.3ms、xctrace 実測の
    泡。docs/research/KERNEL-BRIEF-DECODE-BW.md 参照) GPU が遊ぶ。既定の
    `_STAGE_EVERY=2` (2 層ごと) で投入すれば GPU は先頭から走り出し、
    CPU は残りを組み続けられる。計算内容は Qwen4ExpModel.__call__ +
    lm_head と完全に同一。

    前段 (mask 生成と PLE/n-gram の直前文脈) は本家の `_prelude` を呼ぶので
    写しではない。写しなのは層ループの骨格だけで、そこが本物の差分
    (2 層ごとの async_eval は本家に無い制御フロー)。
    """
    m = model.model
    h = m.embed_tokens(ids)
    cache = caches

    mask, conv_mask, prev_ctx = m._prelude(ids, h, cache)

    h = mx.tile(h, (1, 1, m.hc))
    step = _STAGE_EVERY
    for i, (layer, c) in enumerate(zip(m.layers, cache)):
        idx_c = c.indexer if (c is not None and hasattr(c, "indexer")) else None
        h = layer(h, m.rope, mask, conv_mask, c, idx_c, ids, prev_ctx)
        if step and (i + 1) % step == 0 and i < len(m.layers) - 1:
            mx.async_eval(h)
    out = m.hyper_connection_mixer(h)
    if model.args.text.tie_word_embeddings:
        return m.embed_tokens.as_linear(out)
    return model.lm_head(out)


# prefill の中間チャンクを何個まとめてレイヤー主導で流すか。attention/indexer は
# チャンク幅 (2048) のまま、MoE だけグループぶんの行を concat して 1 回で流す。
# gather_qmm (affine_gather_qmm_rhs) は BM 行タイル内の expert 境界ごとに
# フル GEMM をやり直すため、効率は行数/expert に単調 (r=40/80/160 で
# 7.5/8.9/9.8 TFLOPS、密上限 11.2)。チャンク幅そのものを上げる案 (下の
# MLXTURBO_PREFILL_CHUNK) は attention/indexer の一時増と相殺したが、
# こちらは MoE の行だけ太らせるので相殺しない (in-model 実測: chunk 4096 で
# MoE 部分時間 18.4 -> 14.9s)。gather_qmm は行独立で、BM=32/64 をまたぐ
# 分割でもビット一致 (micro 確認済み) — 出力はトークン列まで不変が要件。
_PREFILL_GROUP = int(os.environ.get("MLXTURBO_PREFILL_GROUP", "4") or 0)


def _group_prefill_forward(model, chunks, caches):
    """中間 prefill チャンクのグループをレイヤー主導で流す。

    計算内容は「各チャンクを順に Qwen4ExpModel.__call__ に通す」のと完全に
    同一で、違いは MoE (layer.mlp) だけグループ内チャンクの行を concat して
    1 回で呼ぶこと (行独立なのでビット一致)。mixer は通さない — chunk-major
    でも中間チャンクの mixer 出力は捨てられており、呼び手が使う cap.hyper は
    mixer の入力 (= 最終レイヤー出力) だから、それをチャンクごとに返す。

    層の中身は本家の `DecoderLayer.pre_mlp` と `_combine` を呼ぶので写しでは
    ない。ここに残る差分は「二重ループ (レイヤー主導 x G チャンク) と MoE の
    呼び出し粒度」だけで、それが layer-major prefill の本体。

    キャッシュ整合性: レイヤー主導でも「レイヤー i がチャンク c を処理する
    時点のキャッシュ i の中身」は chunk-major と一致する (レイヤー i の
    キャッシュを進めるのはレイヤー i 自身だけなので、処理順の入れ替えは
    各キャッシュから見ると無関係)。mask も同じ理由で per-chunk に 1 個。
    """
    Q = _arch()
    m = model.model
    G = len(chunks)
    hs = [mx.tile(m.embed_tokens(ch), (1, 1, m.hc)) for ch in chunks]
    conv_mask = None

    masks = [None] * G  # 最初の full attention 層の走査中に生成する

    # per-chunk の prev_ctx: 本家はモデル呼び出しごとに pc[3] を進めるので、
    # チャンク列に対して同じ更新を先に畳んでおく
    prev_ctxs = [None] * G
    if m.ple_layers:
        ctx_len = m.args.ngram_size - 1
        eos_id = m.args.eos_token_id
        eos_id = eos_id[0] if isinstance(eos_id, list) else eos_id
        pc = caches[m.ple_layers[0]]
        prev = pc[3] if pc is not None else None
        if prev is None:
            prev = mx.full((chunks[0].shape[0], ctx_len), eos_id, chunks[0].dtype)
        for ci, ch in enumerate(chunks):
            prev_ctxs[ci] = prev
            prev = mx.concatenate([prev, ch], axis=1)[:, -ctx_len:]
        if pc is not None:
            pc[3] = prev

    step = _STAGE_EVERY
    for li, (layer, c) in enumerate(zip(m.layers, caches)):
        idx_c = c.indexer if (c is not None and hasattr(c, "indexer")) else None
        posts = []
        for ci in range(G):
            if layer.layer_type != "linear_attention" and masks[ci] is None:
                # 本家は層ごとに作るが、同じチャンクなら層をまたいで同一
                # (mask はチャンク幅とそのキャッシュのオフセットだけで決まり、
                # レイヤー i のキャッシュを進めるのはレイヤー i 自身だけ)。
                masks[ci] = Q.create_attention_mask(hs[ci], [c])
            posts.append(layer.pre_mlp(
                hs[ci], m.rope, masks[ci], conv_mask, c, idx_c,
                chunks[ci], prev_ctxs[ci],
            ))
        xcat = mx.concatenate([p[0] for p in posts], axis=1)
        ycat = layer.mlp(xcat)
        offs, acc = [], 0
        for p in posts[:-1]:
            acc += p[0].shape[1]
            offs.append(acc)
        ys = mx.split(ycat, offs, axis=1)
        for ci in range(G):
            _, hyper, inject = posts[ci]
            hs[ci] = layer._combine(hyper, ys[ci], inject)
        if step and (li + 1) % step == 0 and li < len(m.layers) - 1:
            mx.async_eval(*hs)
    return hs


def _pipeline_snapshot(model, caches, mtp_cache):
    """楽観先組み (次ラウンドのグラフを結果を知らずに組む) 用の浅い退避。

    MLX の配列は不変ノードで、キャッシュの「更新」は Python 参照の付け替え
    (KVCache の setitem もオブジェクトを新ノードへ束縛し直すだけ) なので、
    参照とオフセットを控えておけば付け替えで完全に戻せる。飛行中のグラフは
    古いノードを掴んでいるため影響を受けない。"""
    st = []
    for layer, c in zip(model.model.layers, caches):
        if layer.layer_type == "full_attention":
            st.append(("a", c.keys, c.values, c.offset, c.indexer.keys))
        else:
            # ArraysCache は 4 スロットの参照だけ (offset を持たない。
            # advance() は lengths/left_padding 用で decode では両方 None)
            st.append(("l", c[0], c[1], c[2], c[3]))
    st.append(("m", mtp_cache.keys, mtp_cache.values, mtp_cache.offset,
               mtp_cache.indexer.keys))
    return st


def _pipeline_restore(model, caches, mtp_cache, st) -> None:
    for (layer, c), rec in zip(zip(model.model.layers, caches), st[:-1]):
        if rec[0] == "a":
            _, c.keys, c.values, c.offset, c.indexer.keys = rec
        else:
            _, c0, c1, c2, c3 = rec
            c[0], c[1], c[2], c[3] = c0, c1, c2, c3
    rec = st[-1]
    _, mtp_cache.keys, mtp_cache.values, mtp_cache.offset, mtp_cache.indexer.keys = rec


def snapshot_pre(model, caches) -> dict:
    """**Before** the forward, note down what capture cannot restore.

    n-gram context の取り出しは、直添字 (``c[3]``) ではなく
    ``arch.recurrent_layers`` の名前付き ``ngram`` スロットを経由する
    (mlxturbo/arch.py 参照 -- 族が変わってもスロット番号を汎用コード側に
    ハードコードしないため)。
    """
    slots = {rl.index: rl for rl in _archmod.recurrent_layers(model)}
    pre = {"kv": [], "ctx": []}
    for i, (layer, c) in enumerate(zip(model.model.layers, caches)):
        if layer.layer_type == "full_attention":
            # _AttnCache derives from KVCache and is not an ArraysCache (it has no .cache)
            keys = c.indexer.keys
            pre["kv"].append((c.offset, None if keys is None else keys.shape[1]))
            pre["ctx"].append(None)
        else:
            pre["kv"].append(None)
            rl = slots.get(i)
            pre["ctx"].append(c[rl.ngram] if (rl is not None and rl.ngram is not None) else None)
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
    # full attention 側の trim/indexer は族ごとに有無が違う別能力
    # (arch.has_indexer) なので、再帰状態の巻き戻し (arch.rollback_recurrent)
    # には混ぜない -- ここで直接扱う。
    for i, layer in enumerate(model.model.layers):
        if layer.layer_type != "full_attention":
            continue
        c = caches[i]
        c.trim(drop)
        if _archmod.has_indexer(c):
            old_len = pre["kv"][i][1] or 0
            c.indexer.keys = c.indexer.keys[:, : old_len + keep]

    # GDN 状態・conv 窓・PLE conv・n-gram 文脈の巻き戻しは
    # mlxturbo/arch.py の名前付きスロット経由の共通部品に畳んである
    # (batch_spec.batched_rollback と共有)。
    _archmod.rollback_recurrent(
        model, caches, cap, keep, ngram_ctx=pre["ctx"], ids_kept=ids_kept
    )


def trim_attn_cache(cache, keep: int) -> None:
    """MTP のドラフトキャッシュを先頭 ``keep`` 件まで縮める。"""
    drop = cache.offset - keep
    if drop <= 0:
        return
    cache.trim(drop)
    if cache.indexer.keys is not None:
        cache.indexer.keys = cache.indexer.keys[:, :keep]


def snapshot_mtp_cache(cache):
    """Copy the MTP draft cache so it can be handed to a later resumed call.

    ``KVCache`` preallocates and writes into ``self.keys[..., i:i+n, :]`` in
    place, so the live cache cannot simply be aliased -- the decode loop would
    overwrite the copy. ``_IndexerCache`` concatenates instead (a fresh array
    per update), so its keys are safe to share. About 5MB at PRIME_WINDOW with
    this model's 2 KV heads.
    """
    if cache is None or cache.offset == 0:
        return None
    n = cache.offset
    return (
        mx.contiguous(cache.keys[..., :n, :]),
        mx.contiguous(cache.values[..., :n, :]),
        n,
        cache.indexer.keys,
    )


def restore_mtp_cache(snap):
    """Rebuild the cache saved by ``snapshot_mtp_cache`` (None -> empty)."""
    cache = _arch()._AttnCache()
    if snap is None:
        return cache
    keys, values, n, idx = snap
    cache.keys, cache.values, cache.offset = keys, values, n
    cache.indexer.keys = idx
    return cache


class FlashSpecEngine:
    """Depth-1 speculative decoding that uses the MTP as the draft.

    Decoding is dispatch-bound: a batched forward costs only 1.17x the S=1 case
    even at S=16 (docs/STATUS.md). **A width-2 verification costs roughly the
    price of a single token**, so on acceptance one forward advances 2 tokens.

    Invariant: `cur` is the token to be fed next, the caches are processed up to
    just before it, and `hyper_prev` is the hyper state at the position that
    produced `cur`.
    """

    def __init__(self, model, mtp, depth: int = MTP_DEPTH):
        self.model = model
        self.mtp = mtp
        self.rope = model.model.rope
        self.depth = max(1, int(depth))
        self.depth_ctx_limit = _depth_ctx_limit(model)
        # draft-rerank (mlx-serve の設計の移植): trunk lm_head の 2bit 再量子化で
        # 全語彙を粗く読み、正確な top-32 だけを trunk のヘッドの行で再採点する。
        # 粗い top-32 に真の argmax が入っている限り draft は trunk と一致し、
        # 検証は常に trunk ヘッドなので出力分布は無条件で無傷。
        # MLXTURBO_DRAFT_RERANK=0 で無効。
        self._rerank = None
        if os.environ.get("MLXTURBO_DRAFT_RERANK", "1") != "0":
            self._build_rerank()

    RERANK_BITS = 2
    RERANK_TOP = 32

    def _build_rerank(self) -> None:
        lm = self.model.lm_head
        if not hasattr(lm, "scales"):
            return
        w = mx.dequantize(lm.weight, lm.scales, lm.biases,
                          group_size=lm.group_size, bits=lm.bits)
        cw, cs, cb = mx.quantize(w, group_size=64, bits=self.RERANK_BITS)
        # 再採点用に trunk の行を bf16 で引けるよう、逆量子化した行列も保持
        # ... はメモリを食い過ぎる (2.5GB)。行の gather は量子化のまま行い、
        # その場で 32 行だけ逆量子化する。
        mx.eval(cw, cs, cb)
        self._rerank = (cw, cs, cb)
        del w
        mx.clear_cache()

    def _draft_argmax(self, out) -> mx.array:
        """draft 用の次トークン。(1, 1) を返す。

        rerank あり: 2bit 粗ヘッドで全語彙 -> top-32 -> trunk の該当行を
        逆量子化して再採点 -> argmax。無し: trunk ヘッドで argmax。
        """
        lm = self.model.lm_head
        row = out[:, -1]
        if self._rerank is None:
            return mx.argmax(lm(out)[:, -1], axis=-1).reshape(1, 1)
        cw, cs, cb = self._rerank
        coarse = mx.quantized_matmul(
            row, cw, scales=cs, biases=cb, transpose=True,
            group_size=64, bits=self.RERANK_BITS)
        top = mx.argpartition(-coarse, self.RERANK_TOP - 1, axis=-1)[..., : self.RERANK_TOP]
        rows = mx.dequantize(
            lm.weight[top[0]], lm.scales[top[0]], lm.biases[top[0]],
            group_size=lm.group_size, bits=lm.bits)
        scores = (row.astype(rows.dtype) @ rows.T)
        best = mx.argmax(scores, axis=-1, keepdims=True)
        return mx.take_along_axis(top, best, axis=-1)

    def _effective_depth(self, pos: int) -> int:
        """この位置で引くドラフト数。長い文脈では 1 に落とす
        (DEPTH_CONTEXT_LIMIT の注記を参照)。"""
        return 1 if pos >= self.depth_ctx_limit else self.depth

    def _draft_chain(self, cur, hyper_prev, cache, depth: int):
        """``self.depth`` トークンをまとめて引く。

        ヘッドは (embed(t), hyper) を受けて、mixer で潰す前に **hyper 形状の
        状態を自分で作る**。それを次段に渡すことで、1 つのヘッドで t+2 より
        先まで届く (DeepSeek-V3 と同じ形)。

        確定した (トークン, hyper) の対は 1 段目だけなので、戻る前にキャッシュを
        その 1 件まで縮める。**このキャッシュに投機的なものを入れない**という
        不変条件が、ラウンドを跨いで持ち回れる根拠になっている。
        """
        Q = _arch()
        keep = cache.offset + 1
        drafts = []
        tok, hyper = cur, hyper_prev
        for step in range(depth):
            emb = self.model.model.embed_tokens(tok)
            mask = Q.create_attention_mask(emb, None)
            x = self.mtp.combine(emb, hyper)
            x = self.mtp.layers[0](
                x, self.rope, mask, None, cache, cache.indexer, None, None
            )
            out = self.mtp.hyper_connection_mixer(x)
            tok = self._draft_argmax(out)
            drafts.append(tok)
            hyper = x
            if step < depth - 1:
                # 段を組み終えるごとに投入する。GPU がこの段を回している間に
                # CPU は次の段 (と rerank) を組む。呼び出し側は末尾で全体を
                # async_eval するので、廃棄されるグラフは無い
                mx.async_eval(tok)
        trim_attn_cache(cache, keep)
        return drafts

    def _verify(self, cap, lg, drafts, temp, precomputed=None, sampler=None):
        """検証フォワードの結果から、採用するトークンと hyper を取り出す。

        ``pair`` は [cur, d1, ..., dk]。位置 j の logits は pair[j] の次の
        トークンを与える。d1 から順に一致する限り採用し、外れたところで
        打ち切ってその位置のトークンを代わりに出す。最後まで当たれば k+1 個出る。
        """
        if not drafts:
            toks = [self._sample(lg[:, 0], temp, sampler)]
            return toks, [cap.hyper[:, 0:1]], 0
        if temp > 0 or sampler is not None:
            # 位置 j のサンプルは lg[:, j] にしか依存しない (j-1 の採否は
            # 「どこで打ち切るか」を決めるだけ) ので、全位置を先に引いて
            # 一致プレフィックスだけ採用しても分布は逐次版と同一。
            # 同期が位置ごと (最大 depth+1 回) から 1 回になる。
            #
            # **この独立性はサンプラーの形に依らない。**top_p / top_k / min_p /
            # logit_bias はどれも「その位置の logits だけを見る変換」なので、
            # 受理判定 (samples[0..j-1] にしか依存しない) で条件付けても位置 j の
            # 分布は歪まない。よって投機ありでも逐次サンプリングと厳密一致する。
            # 履歴依存のもの (repetition_penalty / presence / frequency) は
            # この形では正しく載らないので、サーバー側で非投機に降ろしてある
            # (mlxturbo/runner.py の FlashSpecRunner を参照)。
            k = len(drafts)
            if sampler is not None:
                samp = sampler(lg.reshape(k + 1, -1)).reshape(1, k + 1)
            else:
                samp = mx.random.categorical(
                    lg.astype(mx.float32) / temp).reshape(1, k + 1)
            dv = mx.concatenate(drafts, axis=1)
            mx.eval(samp, dv)
            vals = samp[0].tolist()
            dvals = dv[0].tolist()
            hit = 0
            while hit < k and vals[hit] == dvals[hit]:
                hit += 1
            toks = [samp[:, j:j + 1] for j in range(hit + 1)]
            hypers = [cap.hyper[:, j:j + 1] for j in range(hit + 1)]
            return toks, hypers, hit

        # greedy はドラフト位置ごとに .item() で同期せず、argmax と一致判定を
        # まとめて 1 回の同期で取る (1 ラウンドあたり最大 depth+1 回 -> 1 回)。
        # 呼び出し側が verify 本体と同じ eval で評価済みなら precomputed で
        # 受け取り、この同期も消える
        k = len(drafts)
        if precomputed is not None and precomputed[0] is not None:
            nxt_all, dv = precomputed
        else:
            nxt_all = mx.argmax(lg, axis=-1)          # (1, k+1)
            dv = mx.concatenate(drafts, axis=1)       # (1, k)
            mx.eval(nxt_all, dv)
        vals = nxt_all[0].tolist()
        dvals = dv[0].tolist()
        hit = 0
        while hit < k and vals[hit] == dvals[hit]:
            hit += 1
        toks = [nxt_all[:, j:j + 1] for j in range(hit + 1)]
        hypers = [cap.hyper[:, j:j + 1] for j in range(hit + 1)]
        return toks, hypers, hit

    def _prime_draft_cache(self, ids, hyper):
        """Run the tail of the prompt through the MTP head once ("priming"),
        so the first ``_draft_chain()`` of a generation already has real history.

        ``ids`` and ``hyper`` must cover the same trailing positions of the
        prompt. Only the last ``PRIME_WINDOW`` pairs are fed in: acceptance
        comes from recent context, and a window keeps the cost independent of
        prompt length (this model goes to 262144 tokens).

        The pairing follows ``_draft_chain()``: at position k the head takes
        ``(embed(t_k), hyper_{k-1})``. The prompt supplies every such pair for
        k = 1 .. N-1, and the first real draft continues at k = N, so
        there is no gap and no duplicate.

        No rollback on a rejected round: every pair fed here or by ``_draft_chain()``
        is built from values the target model's verification forward already
        confirmed. The rejected guess's embedding never enters this cache --
        only its argmax is compared. Nor can this cache change what is emitted:
        the output tokens come from the target model's own logits, which never
        read it. It moves the acceptance rate, nothing else.
        """
        Q = _arch()
        cache = Q._AttnCache()
        n = min(ids.shape[1], hyper.shape[1])
        if n > PRIME_WINDOW + 1:
            n = PRIME_WINDOW + 1
            ids, hyper = ids[:, -n:], hyper[:, -n:]
        n_pairs = n - 1
        if n_pairs < 1:
            return cache
        embeds = self.model.model.embed_tokens(ids[:, 1:])
        hyper_ctx = hyper[:, :-1]
        i = 0
        while i < n_pairs:
            j = min(i + PREFILL_STEP_SIZE, n_pairs)
            chunk = embeds[:, i:j]
            out = self.mtp(
                chunk, hyper_ctx[:, i:j], self.rope,
                Q.create_attention_mask(chunk, None), cache, cache.indexer,
            )
            mx.eval(out)
            mx.clear_cache()
            i = j
        return cache

    def generate(self, ids, max_tokens: int, caches=None):
        """Greedy generation. Returns (token sequence, accepted count, round count)."""
        model = self.model
        caches = caches or model.make_cache()
        with capture(model) as cap:
            logits = model(ids, cache=caches)
            mx.eval(logits)
        hyper_prev = cap.hyper[:, -1:]
        mtp_cache = self._prime_draft_cache(ids, cap.hyper)
        cur = mx.argmax(logits[:, -1], axis=-1).reshape(1, 1)

        # **Include the first token produced by prefill in the output too.**
        # `cur` is both "the token to be fed next" and "the most recently
        # generated token", so dropping it here shifts everything by one
        out, accepted, rounds = [int(cur.item())], 0, 0
        while len(out) < max_tokens:
            drafts = self._draft_chain(
                cur, hyper_prev, mtp_cache,
                self._effective_depth(ids.shape[1] + len(out)),
            )
            pair = mx.concatenate([cur] + drafts, axis=1)
            total = pair.shape[1]
            pre = snapshot_pre(model, caches)
            with capture(model) as cap:
                lg = model(pair, cache=caches)
                mx.eval(lg)
            rounds += 1
            toks, hypers, hit = self._verify(cap, lg, drafts, 0.0)
            accepted += hit
            keep = len(toks)
            rollback(model, caches, cap, pre, keep=keep, total=total,
                     ids_kept=pair[:, :keep])
            vals = [int(t.item()) for t in toks][: max_tokens - len(out)]
            out.extend(vals)
            cur, hyper_prev = toks[-1], hypers[-1]
        return out[:max_tokens], accepted, rounds

    # ---------- mlxturbo-serve wiring (added 2026-08-29, additions only) ----------
    #
    # What follows is a new path added without changing ``generate()`` at all.
    # The reason is to absolutely not change the behavior of the existing
    # ``generate()``/``capture``/``rollback``/``snapshot_pre``, which are tied to
    # the measurements in docs/MTP-FLASH.md -- even where logic could be shared,
    # we chose to accept a little duplication over rewriting existing methods.

    @staticmethod
    def _sample(logits_row: mx.array, temp: float, sampler=None) -> mx.array:
        """Choose the next token from one position's worth of logits ((1, vocab)).

        temp<=0 is greedy (argmax, numerically identical to the existing
        generate()). temp>0 samples with temperature via
        ``mx.random.categorical`` (docs/MTP-FLASH.md, "sampling" section: as far
        as sampling from the verification-side logits goes there is no
        approximation in the correctness of the distribution -- the draft stays
        greedy). Returns (1, 1).

        ``sampler`` (省略可) は「1 位置の生 logits (N, vocab) を受けてトークン
        (N,) を返す」関数。top_p/top_k/min_p/logit_bias のような**位置局所な
        変換**を載せるための口で、渡されたときだけこちらを使う (省略時は既存の
        経路が 1 ビットも変わらない)。履歴依存のもの (repetition_penalty 系) は
        ここに載せてはいけない -- 下の `_verify` が全位置を先に引くため、
        位置 j のペナルティが「j-1 までを含む履歴」で計算できない。
        """
        if sampler is not None:
            return sampler(logits_row).reshape(1, 1)
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
        resume: tuple | None = None,
        sampler=None,
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

        ``resume`` (mlxturbo-serve wiring, added 2026-08-30): a
        ``(logits_last, hyper_prev)`` pair captured by a previous call at
        *exactly* this same position (see the return value below), for when
        ``ids`` has zero new tokens (a prompt that matches a session's cache
        down to the very last position -- a resend of the same prompt, a
        regenerate). The chunk loop above never executes for an empty
        ``ids`` and ``cap``/``logits`` would stay unset, so this path skips
        the loop outright and resumes decoding straight from the saved
        state; the MTP draft cache starts cold (empty) since there is no
        freshly-prefilled tail to prime it from -- this costs a little draft
        acceptance on the first few rounds, not correctness (the target
        model's own verification is what the output distribution rests on).
        Ignored (falls back to the normal loop) unless ``ids`` is actually
        empty -- a mismatched caller never gets a shortcut.

        Returns/yields the same as before, except the final ``(accepted,
        rounds)`` is now a 3-tuple ``(accepted, rounds, (logits_last,
        hyper_prev))`` -- the pair at the prefill/decode boundary of *this*
        call (untouched by the decode loop below, unlike the ``hyper_prev``
        local variable that keeps advancing), passed straight through
        unchanged when ``resume`` was used. Callers thread it back in via
        ``resume`` on a session's next call (mlxturbo/runner.py's
        ``FlashSpecRunner``).
        """
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        eos = set(eos_ids)
        model = self.model
        caches = caches if caches is not None else model.make_cache()

        n = ids.shape[1]
        use_resume = resume is not None and n == 0
        if use_resume:
            logits_tail, hyper_tail0, mtp_snap = resume
            # The primed draft cache is carried across too: without it a
            # resumed call would draft from an empty cache and give back the
            # acceptance that priming buys (measured: decode -7%).
            mtp_cache = restore_mtp_cache(mtp_snap)
        else:
            step = PREFILL_STEP_SIZE
            # 中間チャンク幅の knob。sorted gather_qmm の GEMM 効率は
            # 1 エキスパートあたりの行数に単調 (S=2048: 7.5 TFLOPS ->
            # S=8192: 9.8、密上限 11.2) だが、in-model では attention/indexer の
            # 一時テンソル増と相殺して 17k TTFT -2% 止まり、しかもチャンク割りで
            # 境界の丸めが動き出力が揺れる。既定は従来幅のまま。8192 は
            # n-gram RAM 常駐 (wired 108GiB 張り付き) でサーバーが Metal OOM。
            # prefill の本命は MoE gather の segment-aware GEMM カーネル
            # (KERNEL-BRIEF-DECODE-BW.md)。
            big = int(os.environ.get("MLXTURBO_PREFILL_CHUNK", "0") or 0) or step
            i = 0
            logits = None
            cap = None
            # The trailing prefill chunks' hyper state (B, chunk_len, hc*d),
            # kept to prime the MTP's own cache below. Only the last
            # HYPER_KEEP_CHUNKS are retained: priming reads at most
            # PRIME_WINDOW+1 positions, and at this model's 262144-token
            # ceiling holding all of them would be gigabytes.
            # capture(light=True) records only GatedResidual's hyper, so this
            # does not reintroduce the OOM that `light` exists to avoid.
            hyper_chunks = []
            while i < n:
                remaining = n - i
                # 前方の等長 2048 チャンクだけレイヤー主導でグループ処理する
                # (チャンク境界は従来と同一 grid なので出力はビット一致)。
                # 末尾側 (端数チャンクと最終チャンク) は従来経路のまま —
                # checkpoint の粒度が要るのは分岐が起きやすい末尾だから。
                # MLXTURBO_PREFILL_CHUNK 指定時は旧 knob を優先して無効化。
                if _PREFILL_GROUP > 1 and big == step and remaining - step >= 2 * step:
                    g = min(_PREFILL_GROUP, (remaining - step) // step)
                    group_chunks = [
                        ids[:, i + k * step : i + (k + 1) * step] for k in range(g)
                    ]
                    hys = _group_prefill_forward(model, group_chunks, caches)
                    hys = hys[-HYPER_KEEP_CHUNKS:]
                    mx.eval(*hys)
                    for c in caches:
                        state = getattr(c, "state", None)
                        if state is not None:
                            mx.eval(state)
                    mx.clear_cache()
                    hyper_chunks.extend(hys)
                    del hyper_chunks[:-HYPER_KEEP_CHUNKS]
                    i += g * step
                    if checkpoints is not None:
                        checkpoints.append(
                            (base_pos + i, snapshot_untrimmable_caches(caches))
                        )
                        del checkpoints[:-CHECKPOINT_RETENTION]
                    continue
                if remaining > step:
                    j = i + min(big, remaining - step)
                else:
                    j = n
                chunk = ids[:, i:j]
                if j == n:
                    # BPE 境界 checkpoint (共有ヘルパー: prefill_common.py の
                    # split_and_checkpoint_tail、詳しい背景はそちらの
                    # docstring 参照)。checkpoints が有効かつ chunk が 2
                    # トークン以上のときだけ、直前 1 トークンを切り離して
                    # 手前 (head) にも checkpoint を積む -- そうしないと
                    # 会話 2 ターン目の retemplate で末尾トークンが BPE
                    # マージにより化け、LCP が checkpoint のちょうど 1
                    # トークン手前に落ちてセッション全体が使い捨てになる
                    # (実測: 診断で確認)。no-op 時 (checkpoints=None または
                    # chunk 長 1 以下、generate()/検証プローブがこちら) は
                    # head_result が空 tuple で返り、tail_split は False の
                    # まま従来の分岐に合流する。副次効果として分割時は最終
                    # チャンクの lm_head が 1 トークン分に縮む (従来は
                    # チャンク全幅の logits を作って末尾だけ使っていた)。
                    def _forward_head(head):
                        with capture(model, light=True) as cap0:
                            h0 = model.model(head, cache=caches)
                        return h0, cap0.hyper

                    chunk, head_result = split_and_checkpoint_tail(
                        chunk,
                        checkpoints,
                        base_pos + i,
                        caches,
                        CHECKPOINT_RETENTION,
                        snapshot_untrimmable_caches,
                        _forward_head,
                    )
                    tail_split = bool(head_result)
                    # light=True: this chunk uses only cap.hyper[:, -1:]
                    # (referenced right below). Full capture (cap.gdn/cap.ple)
                    # unconditionally allocated memory proportional to T (this
                    # chunk's length, at most PREFILL_STEP_SIZE) for all 36
                    # layers, and OOMed the actual machine at a few thousand
                    # tokens (see the docstring for capture()'s light
                    # argument). The decode loop's verification forward
                    # (below, T<=2) stays on full capture as before
                    with capture(model, light=True) as cap:
                        logits = model(chunk, cache=caches)
                        mx.eval(logits)
                    if tail_split:
                        cap.hyper = mx.concatenate([head_result[1], cap.hyper], axis=1)
                else:
                    # light=True (added for MTP priming): only the cheap
                    # GatedResidual hook runs, so this branch's computation and
                    # cache updates are unchanged -- we additionally record
                    # cap.hyper for this chunk.
                    with capture(model, light=True) as cap:
                        h = model.model(chunk, cache=caches)
                        mx.eval(h)
                    for c in caches:
                        state = getattr(c, "state", None)
                        if state is not None:
                            mx.eval(state)
                    mx.clear_cache()
                hyper_chunks.append(cap.hyper)
                del hyper_chunks[:-HYPER_KEEP_CHUNKS]
                i = j
                if checkpoints is not None:
                    checkpoints.append((base_pos + i, snapshot_untrimmable_caches(caches)))
                    del checkpoints[:-CHECKPOINT_RETENTION]
            # mx.contiguous, not a bare slice: a slice keeps its parent
            # buffer alive, and these two outlive the call (they are published
            # as the session's tail for a later diff-0 resume). The last
            # chunk's ``logits`` is (1, chunk_len, vocab) -- 2GB at
            # PREFILL_STEP_SIZE with this vocabulary -- so holding a view of it
            # in every pooled session would retain gigabytes per slot.
            hyper_tail0 = mx.contiguous(cap.hyper[:, -1:])
            logits_tail = mx.contiguous(logits[:, -1])
            if max_tokens == 0:
                # Keep the successfully prefetched cache, but do not expose the
                # first sampled ``cur``.  This mirrors SpecEngine.generate(0) and
                # lets FlashSpecRunner publish exactly the prompt as processed.
                return 0, 0, (logits_tail, hyper_tail0, None)
            hyper_tail = (
                hyper_chunks[0] if len(hyper_chunks) == 1
                else mx.concatenate(hyper_chunks, axis=1)
            )
            mtp_cache = self._prime_draft_cache(ids[:, -hyper_tail.shape[1]:], hyper_tail)
            mtp_snap = snapshot_mtp_cache(mtp_cache)

        if max_tokens == 0:
            # Only reachable via ``use_resume`` -- the normal path already
            # returned above.
            return 0, 0, (logits_tail, hyper_tail0, mtp_snap)

        hyper_prev = hyper_tail0
        cur = self._sample(logits_tail, temp, sampler)

        first = int(cur.item())
        out = [first]
        yield [first]
        accepted, rounds = 0, 0
        if first in eos:
            return accepted, rounds, (logits_tail, hyper_tail0, mtp_snap)

        # 計測モード (MLXTURBO_PHASE_TIMERS=1): フェーズ境界で強制 eval して
        # draft / verify / post の実時間を分ける。強制 eval 自体が同期を増やす
        # ので、絶対値は少し膨らむ。比率を見るためのもの。既定では完全に素通り。
        timers = os.environ.get("MLXTURBO_PHASE_TIMERS") == "1"
        phase = {"draft": 0.0, "verify": 0.0, "post": 0.0, "rollback": 0.0}
        self.last_phase = phase if timers else None
        # 楽観パイプライン: verify の GPU 実行中に「全採用だった場合の次ラウンド」
        # のグラフを CPU で先に組む。全採用なら rollback は no-op なので先組みが
        # そのまま正しく、外れたら _pipeline_restore で参照を戻して組み直す。
        # GPU トレース実測でラウンド毎に ~7ms の泡 (グラフ構築中の GPU アイドル)
        # があり、全採用率 ~0.7 との積で泡の大半が消える。greedy のみ。
        pending = None
        next_drafts = None
        # 1=通常の楽観パイプライン, 2=組むが毎回捨てる (切り分け用), 0=無効
        pipeline = int(os.environ.get("MLXTURBO_PIPELINE", "0") or 0)
        while len(out) < max_tokens:
            if timers:
                ts = time.perf_counter()
            if pending is not None:
                drafts, pair, total, pre, cap, lg, pipe_snap = pending
                pending = None
                pipe_snap = None
            else:
                if next_drafts is not None:
                    drafts = next_drafts        # 前ラウンド末尾で構築・投機済み
                    next_drafts = None
                else:
                    drafts = self._draft_chain(
                        cur, hyper_prev, mtp_cache,
                        self._effective_depth(base_pos + n + len(out)),
                    )
                    # draft は必ず使うので先に投げる。GPU が draft チェーンを
                    # 回している間に、CPU は下の検証フォワードのグラフを組む
                    mx.async_eval(drafts)
                pair = mx.concatenate([cur] + drafts, axis=1)
                total = pair.shape[1]
                pre = snapshot_pre(model, caches)
                with capture(model) as cap:
                    lg = _staged_forward(model, pair, caches)
            if timers:
                mx.eval(drafts)
                phase["draft"] += time.perf_counter() - ts
                ts = time.perf_counter()
            next_pending = None
            if pipeline > 0 and temp <= 0 and len(out) + total < max_tokens:
                mx.async_eval(lg)
                snap2 = _pipeline_snapshot(model, caches, mtp_cache)
                cur2 = mx.argmax(lg[:, total - 1], axis=-1).reshape(1, 1)
                hyper2 = cap.hyper[:, total - 1: total]
                drafts2 = self._draft_chain(
                    cur2, hyper2, mtp_cache,
                    self._effective_depth(base_pos + n + len(out) + total),
                )
                pair2 = mx.concatenate([cur2] + drafts2, axis=1)
                pre2 = snapshot_pre(model, caches)
                with capture(model) as cap2:
                    lg2 = model(pair2, cache=caches)
                next_pending = (drafts2, pair2, pair2.shape[1], pre2, cap2,
                                lg2, snap2)
            if _ROUND_TRACE:
                _rt = [("built", time.perf_counter())]
            if temp <= 0 and drafts:
                nxt_all = mx.argmax(lg, axis=-1)
                dv = mx.concatenate(drafts, axis=1)
                mx.eval(lg, nxt_all, dv)
            else:
                nxt_all = dv = None
                mx.eval(lg)
            if _ROUND_TRACE:
                _rt.append(("eval_done", time.perf_counter()))
            rounds += 1
            if timers:
                phase["verify"] += time.perf_counter() - ts
                ts = time.perf_counter()
            toks, hypers, hit = self._verify(cap, lg, drafts, temp, sampler=sampler,
                                             precomputed=(nxt_all, dv))
            accepted += hit

            vals = [int(t.item()) for t in toks]
            cut = next((k for k, v in enumerate(vals) if v in eos), None)
            if cut is not None:
                toks, hypers, vals = toks[: cut + 1], hypers[: cut + 1], vals[: cut + 1]
            remaining = max_tokens - len(out)
            if len(vals) > remaining:
                toks, hypers, vals = toks[:remaining], hypers[:remaining], vals[:remaining]

            # When keep==total (an ordinary accept round with no truncation),
            # rollback() itself returns early, so it is fine to always call it.
            if timers:
                phase["post"] += time.perf_counter() - ts
                ts = time.perf_counter()
            full_accept = (len(vals) == total and cut is None)
            if next_pending is not None:
                if full_accept and pipeline == 1:
                    pending = next_pending
                else:
                    _pipeline_restore(model, caches, mtp_cache, next_pending[6])
            cur, hyper_prev = toks[-1], hypers[-1]
            # 次ラウンドの draft をここで組んで投げる (rollback / yield などの
            # CPU 後処理を draft の GPU 実行に隠す)。draft はトランクの
            # キャッシュに触れず、rollback は MTP のキャッシュに触れないので
            # 順序を入れ替えても意味は変わらない。最終ラウンドでは無駄になるが
            # 1 リクエスト 1 回きり。
            if _ROUND_TRACE:
                _rt.append(("verify_done", time.perf_counter()))
            next_drafts = None
            if (pending is None and cut is None
                    and len(out) + len(vals) < max_tokens):
                next_drafts = self._draft_chain(
                    cur, hyper_prev, mtp_cache,
                    self._effective_depth(base_pos + n + len(out) + len(vals)),
                )
                mx.async_eval(next_drafts)
            if _ROUND_TRACE:
                _rt.append(("drafts_submitted", time.perf_counter()))
            rollback(model, caches, cap, pre, keep=len(vals), total=total,
                     ids_kept=pair[:, : len(vals)])
            if timers:
                mx.eval([c.state for c in caches if getattr(c, "state", None) is not None])
                phase["rollback"] += time.perf_counter() - ts
            out.extend(vals)
            if _ROUND_TRACE:
                _rt.append(("rollback_done", time.perf_counter()))
                base_t = _rt[0][1]
                print(f"[round] t={base_t * 1e3:.2f}", " ".join(
                    f"{k}={(t - base_t) * 1e3:.2f}" for k, t in _rt[1:]), flush=True)
            yield vals
            if cut is not None:
                break

        return accepted, rounds, (logits_tail, hyper_tail0, mtp_snap)


__all__ = ["Capture", "FlashSpecEngine", "capture", "rollback", "snapshot_pre"]
