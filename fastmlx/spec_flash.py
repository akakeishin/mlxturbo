"""Qwen3.8-Flash-Next の投機デコードの土台: 状態の捕獲と巻き戻し。

`fastmlx/spec.py` は 27B (qwen3_5) 構成に強く結びついているので別に持つ。

## なぜ捕獲が要るか

投機は「draft を混ぜて一括で検証し、外れたら捨てる」。ところが Flash-Next の
36 層は GatedDeltaNet で、**再帰状態は巻き戻せない**。KV のように末尾を
切れば済むものではないので、検証 forward の**各位置を処理し終えた直後の
状態**を残しておき、受理した長さの分だけ採用する。

巻き戻しが要るものは 4 つ:

| 対象 | 層数 | 戻し方 |
|---|---|---|
| GDN の再帰状態 `cache[1]` | 36 | `states_all[:, keep-1]` (捕獲) |
| GDN の conv 窓 `cache[0]` | 36 | `conv_input[:, keep : keep+K-1]` (捕獲) |
| PLE の conv 窓 `cache[2]` | 1 | 同上 |
| full attention の KV と indexer | 12 | `KVCache.trim()` と keys の切り詰め |
| n-gram の文脈 `cache[3]` | 1 | forward 前の値を保存しておく |

`ArraysCache.advance` は `lengths`/`left_padding` しか触らない (単一系列では
どちらも None) ので、GDN 側に offset の巻き戻しは要らない。

## 設計: 本体は差し替えない

`GatedDeltaNet.__call__` と `PLELayer._short_conv` だけを一時的に差し替え、
それ以外は本体の実装をそのまま通す。**forward の経路を書き写さない**ので、
捕獲版と素の版が食い違う余地が小さい (書き写すと必ずどこかでずれる)。
"""

from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn


def _arch():
    import mlx_lm.models.qwen4_exp as Q

    return Q


class Capture:
    """検証 forward 1 回ぶんの、巻き戻しに必要な記録。"""

    def __init__(self):
        self.gdn = {}      # id(module) -> (conv_input, states_all)
        self.ple = {}      # id(module) -> full
        self.hyper = None  # 最終 mixer に入る直前の hyper 状態
        self.pre = {}      # forward 前のキャッシュ状態 (KV offset など)


@contextmanager
def capture(model):
    """検証 forward を、巻き戻しに必要な記録を残しながら回す文脈。"""

    Q = _arch()
    from .kernels.gated_delta_states import gated_delta_update_with_states

    cap = Capture()
    orig_gdn = Q.GatedDeltaNet.__call__
    orig_ple = Q.PLELayer._short_conv
    orig_hc = Q.GatedResidual.__call__
    mixer = model.model.hyper_connection_mixer

    def gdn(self, x, mask, cache):
        # 本体の GatedDeltaNet.__call__ をそのまま写し、状態を返すカーネルに
        # だけ差し替える。ロジックを変えないこと
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

    Q.GatedDeltaNet.__call__ = gdn
    Q.PLELayer._short_conv = ple_conv
    Q.GatedResidual.__call__ = hc
    try:
        yield cap
    finally:
        Q.GatedDeltaNet.__call__ = orig_gdn
        Q.PLELayer._short_conv = orig_ple
        Q.GatedResidual.__call__ = orig_hc


def snapshot_pre(model, caches) -> dict:
    """forward の**前**に、捕獲では戻せないものを控える。"""
    pre = {"kv": [], "ctx": []}
    for layer, c in zip(model.model.layers, caches):
        if layer.layer_type == "full_attention":
            # _AttnCache は KVCache 派生で ArraysCache ではない (.cache を持たない)
            keys = c.indexer.keys
            pre["kv"].append((c.offset, None if keys is None else keys.shape[1]))
            pre["ctx"].append(None)
        else:
            pre["kv"].append(None)
            pre["ctx"].append(c[3])
    return pre


def rollback(model, caches, cap: Capture, pre: dict, keep: int, total: int,
             ids_kept=None):
    """検証 forward で進めた `total` トークンのうち、先頭 `keep` だけ残す。

    `ids_kept` は採用したトークン列 (B, keep)。n-gram の文脈は「捨てた分を
    含まない位置」まで進めた値でなければならないので、forward 前の文脈から
    作り直す (forward 前の値をそのまま戻すと keep 個ぶん巻き戻しすぎる)。
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
        # keep トークン処理後の conv 窓 = conv_input[:, keep : keep+k-1]
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
    """MTP を draft に使う深さ 1 の投機デコード。

    デコードはディスパッチ律速で、一括 forward は S=16 でも S=1 の 1.17 倍しか
    かからない (docs/STATUS.md)。**幅 2 の検証はほぼ 1 トークンぶんの値段**なので、
    受理すれば 1 回の forward で 2 トークン進む。

    不変条件: `cur` が次に流すトークン、キャッシュはその手前まで処理済み、
    `hyper_prev` は `cur` を生んだ位置の hyper 状態。
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
        """貪欲生成。戻り値は (トークン列, 受理数, 反復数)。"""
        model = self.model
        caches = caches or model.make_cache()
        with capture(model) as cap:
            logits = model(ids, cache=caches)
            mx.eval(logits)
        hyper_prev = cap.hyper[:, -1:]
        cur = mx.argmax(logits[:, -1], axis=-1).reshape(1, 1)

        # **プレフィルが生んだ最初のトークンも出力に入れる。**`cur` は「次に
        # 流すトークン」であると同時に「生成済みの最新トークン」なので、
        # ここを落とすと丸ごと 1 つずれる
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
                # draft が当たった -> 位置 1 の logits がそのまま次のトークン
                nxt2 = mx.argmax(lg[:, 1], axis=-1).reshape(1, 1)
                out.append(int(nxt2.item()))
                accepted += 1
                cur, hyper_prev = nxt2, cap.hyper[:, 1:2]
                # keep == total なので巻き戻し不要
            else:
                rollback(model, caches, cap, pre, keep=1, total=2, ids_kept=cur)
                cur, hyper_prev = nxt, cap.hyper[:, 0:1]
        return out[:max_tokens], accepted, rounds


__all__ = ["Capture", "FlashSpecEngine", "capture", "rollback", "snapshot_pre"]
