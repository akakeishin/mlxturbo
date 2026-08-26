"""MTP 自己投機デコード。

draft: MTP ブロック(全体の ~1.5%)を連鎖させて n_draft 個の候補を出す。
verify: 本体 1 回の forward で n_draft+1 トークンをまとめて検証する。
m=2 の検証は m=1 の 1.04 倍しかかからない(実測)ので、受理された分が
ほぼそのまま速度倍率になる。greedy では出力は非投機と完全一致する。

線形アテンション 48 層の巻き戻し: 射影と conv は m トークン一括(帯域償却を
維持)、再帰更新だけ位置ごとに回して全位置の状態を保持する。棄却時は
該当位置の状態と conv 窓を差し替えるだけで再計算しない。
"""

import time
from collections import Counter

import mlx.core as mx
import mlx.nn as nn

from ._mlx_compat import (
    KVCache,
    create_attention_mask,
    create_ssm_mask,
    validate_spec_model_contract,
)
from .kernels.gated_delta_states import gated_delta_update_with_states


class ChatSession:
    """ターンをまたいで KV と線形状態を持ち越すための入れ物。

    新プロンプトが処理済み列の純粋な追記なら差分だけ prefill する。
    追記でなければ (テンプレートが履歴を書き換えた等) 全再構築に落ちる。
    線形アテンションの状態は途中位置へ巻き戻せないため、この二択になる。
    """

    def __init__(self):
        self.caches = None
        self.mtp_cache = None
        self.mtp_valid = False
        self.processed = []
        self.h_last = None

    def invalidate(self):
        """Drop every published field before aliased caches are mutated."""

        self.caches = None
        self.mtp_cache = None
        self.mtp_valid = False
        self.processed = []
        self.h_last = None

    def publish(self, caches, mtp_cache, mtp_valid, processed, h_last):
        """Publish one internally consistent session snapshot after success."""

        self.caches = caches
        self.mtp_cache = mtp_cache
        self.mtp_valid = mtp_valid
        self.processed = processed
        self.h_last = h_last


class SpecEngine:
    def __init__(self, model, mtp):
        validate_spec_model_contract(model)
        self.text = model.language_model
        self.inner = self.text.model
        self.mtp = mtp

    def _head(self, h_prenorm: mx.array, norm) -> mx.array:
        out = norm(h_prenorm)
        if self.text.args.tie_word_embeddings:
            return self.inner.embed_tokens.as_linear(out)
        return self.text.lm_head(out)

    # ---------- 本体 forward ----------

    def _hidden_forward(self, tokens: mx.array, caches, capture: bool):
        """tokens: (S,)。戻り値: (最終 norm 前 hidden (1,S,D), 線形層の巻き戻し情報)"""
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
        for layer, c in zip(self.inner.layers, caches):
            if layer.is_linear:
                if capture:
                    h = self._linear_capture(layer, h, c, sink, ssm_mask)
                else:
                    h = layer(h, mask=ssm_mask, cache=c)
            else:
                h = layer(h, mask=fa_mask, cache=c)
        return h, sink

    def _linear_capture(self, layer, x, cache, sink, mask=None):
        """GatedDeltaNet と同じ計算を、位置ごとの再帰状態を残しながら行う。"""
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

    def _mtp_append(self, tok_ids: mx.array, hiddens: mx.array, mtp_cache) -> mx.array:
        """位置ごとの (embed(t_{i+1}), h_i) ペアを MTP に流し K/V を積む。"""
        e = self.inner.embed_tokens(tok_ids[None])
        return self.mtp(e, hiddens, cache=mtp_cache)

    # ---------- 文脈 lookup draft ----------

    @staticmethod
    def _lookup_draft(ctx: list, ngram: int, max_len: int):
        """ctx 末尾 ngram の直近の再出現を探し、その続きを draft にする。

        コード・編集・引用のような写経率の高い生成で受理長が伸びる。
        線形走査だが ctx 数千トークンなら 1ms 未満。
        """
        if len(ctx) < ngram + 1:
            return None
        key = ctx[-ngram:]
        for i in range(len(ctx) - ngram - 1, -1, -1):
            if ctx[i : i + ngram] == key:
                cont = ctx[i + ngram : i + ngram + max_len]
                if cont:
                    return cont
        return None

    # ---------- 生成 ----------

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
    ):
        """max_draft > 0 で受理適応の可変深度になる。

        直前ステップの受理数 a に応じて次の深度を決める:
        全受理なら +2 (上限 max_draft)、全棄却なら 1、部分受理なら a。
        外れ区間の検証税を m=2 の 1.04 倍まで下げ、当たり区間だけ深掘りする。

        draft 源は 2 系統: 文脈 lookup (suffix 再出現の続き、深さ lookup_len) が
        当たればそれを優先し、外れたら MTP 連鎖。lookup が全棄却だったら
        数ステップ休ませる。

        ``n_draft=0, max_draft=0, lookup_len=0`` is the non-speculative
        baseline contract used by ``bench/gate.py``.
        """
        if min(max_tokens, n_draft, max_draft, lookup_len, lookup_ngram) < 0:
            raise ValueError("generation limits must be non-negative")
        eos = set(eos_ids)
        prompt_ids = list(prompt_ids)
        use_mtp = n_draft > 0 or max_draft > 0

        caches = mtp_cache = None
        reused = 0
        reused_h_last = None
        if session is not None and session.caches is not None:
            pl = session.processed
            n = min(len(pl), len(prompt_ids))
            lcp = 0
            while lcp < n and pl[lcp] == prompt_ids[lcp]:
                lcp += 1
            mtp_reusable = not use_mtp or session.mtp_valid
            if lcp == len(pl) and lcp < len(prompt_ids) and mtp_reusable:
                caches, mtp_cache = session.caches, session.mtp_cache
                reused_h_last = session.h_last
                reused = lcp
                # The local variables now own these mutable caches.  Any error,
                # including KeyboardInterrupt or a callback failure, leaves the
                # public session invalid instead of half-published.
                session.invalidate()
        if caches is None:
            caches = self.text.make_cache()
            mtp_cache = KVCache()
        elif mtp_cache is None:
            mtp_cache = KVCache()

        prompt = mx.array(prompt_ids[reused:])

        t0 = time.perf_counter()
        h_all, _ = self._hidden_forward(prompt, caches, capture=False)
        if use_mtp:
            if reused:
                mtp_hiddens = mx.concatenate(
                    [reused_h_last, h_all[:, :-1]], axis=1
                )
                self._mtp_append(prompt, mtp_hiddens, mtp_cache)
            elif prompt.shape[0] > 1:
                self._mtp_append(prompt[1:], h_all[:, :-1], mtp_cache)
        h_last = h_all[:, -1:]

        if max_tokens == 0:
            mx.eval(h_last)
            ttft = time.perf_counter() - t0
            if session is not None:
                session.publish(
                    caches, mtp_cache, use_mtp, list(prompt_ids), h_last
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
        ctx = list(prompt_ids) + out_tokens
        accept_hist = Counter()
        accept_trace = []
        fed_gen = []
        src_hist = {"lookup": Counter(), "mtp": Counter()}
        phase = {"draft": 0.0, "verify": 0.0, "maint": 0.0}
        t1 = time.perf_counter()

        depth = n_draft
        lookup_cool = 0
        lookup_cur = min(6, lookup_len)
        while len(out_tokens) < max_tokens and out_tokens[-1] not in eos:
            ts = time.perf_counter()
            mtp_off0 = mtp_cache.offset
            lk = None
            proposal_cap = max_tokens - len(out_tokens) - 1
            if proposal_cap > 0 and lookup_len > 0 and lookup_cool == 0:
                lk = self._lookup_draft(
                    ctx, lookup_ngram, min(lookup_cur, proposal_cap)
                )
            if lk:
                source = "lookup"
                window = mx.concatenate([y, mx.array(lk)])
            else:
                source = "mtp"
                lookup_cool = max(0, lookup_cool - 1)
                drafts = []
                dh, dtok = h_last, y
                for _ in range(min(depth, proposal_cap)):
                    h_mtp = self._mtp_append(dtok, dh, mtp_cache)
                    d = mx.argmax(
                        self._head(h_mtp[:, -1:], self.mtp.norm), axis=-1
                    ).reshape(1)
                    drafts.append(d)
                    dh, dtok = h_mtp[:, -1:], d
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
            if temp == 0:
                preds = mx.argmax(logits, axis=-1)[0]
                mx.eval(preds, window)
                preds_l, window_l = preds.tolist(), window.tolist()
                n_avail = len(window_l) - 1
                a = 0
                while a < n_avail and preds_l[a] == window_l[a + 1]:
                    a += 1
                    if window_l[a] in eos:
                        accepted_eos = a
                        break
                if accepted_eos is None:
                    next_tok = preds_l[a]
            else:
                # 棄却サンプリング (draft は決定的提案 = delta 分布)。
                # 受理確率 p_target(d)、棄却時は d を除いた残差から引き直す。
                # 出力分布は非投機の temp サンプリングと厳密に一致する。
                lg = logits[0].astype(mx.float32) / temp
                probs = mx.softmax(lg, axis=-1)
                nw = window.shape[0] - 1
                p_draft = mx.take_along_axis(
                    probs[:nw], window[1:, None], axis=-1
                )[:, 0]
                u = mx.random.uniform(shape=(nw,))
                mx.eval(p_draft, u, window)
                window_l = window.tolist()
                p_l, u_l = p_draft.tolist(), u.tolist()
                n_avail = nw
                a = 0
                while a < n_avail and u_l[a] < p_l[a]:
                    a += 1
                    if window_l[a] in eos:
                        accepted_eos = a
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
                if a == 0:
                    lookup_cool = 4
                    lookup_cur = min(6, lookup_len)
                elif a == n_avail:
                    lookup_cur = min(lookup_cur * 2, lookup_len)
            elif max_draft > 0:
                if a == depth:
                    depth = min(depth + 2, max_draft)
                elif a == 0:
                    depth = 1
                else:
                    depth = a

            self._rollback(caches, sink, len(window_l), consumed)
            if use_mtp:
                mtp_cache.trim(mtp_cache.offset - mtp_off0)
                true_hiddens = mx.concatenate(
                    [h_last, hs[:, : consumed - 1]], axis=1
                )
                self._mtp_append(window[:consumed], true_hiddens, mtp_cache)

            h_last = hs[:, consumed - 1 : consumed]
            if accepted_eos is not None:
                step_tokens = window_l[1 : a + 1]
            else:
                y = mx.array([next_tok])
                step_tokens = window_l[1:consumed] + [next_tok]
            out_tokens.extend(step_tokens)
            ctx.extend(step_tokens)
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
            "phase_s": phase,
            "steps": steps,
            "mean_accepted": (
                sum(k * v for k, v in accept_hist.items()) / steps if steps else 0.0
            ),
            "tokens_per_step": (n_decode / steps if steps else 0.0),
        }
