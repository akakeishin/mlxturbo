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
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gated_delta import gated_delta_update


class SpecEngine:
    def __init__(self, model, mtp):
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
        x = self.inner.embed_tokens(tokens[None])
        fa_mask = create_attention_mask(x, caches[self.inner.fa_idx])
        ssm_mask = create_ssm_mask(x, caches[self.inner.ssm_idx])
        sink = []
        h = x
        for layer, c in zip(self.inner.layers, caches):
            if layer.is_linear:
                if capture:
                    h = self._linear_capture(layer, h, c, sink)
                else:
                    h = layer(h, mask=ssm_mask, cache=c)
            else:
                h = layer(h, mask=fa_mask, cache=c)
        return h, sink

    def _linear_capture(self, layer, x, cache, sink):
        """GatedDeltaNet と同じ計算を、位置ごとの再帰状態を残しながら行う。"""
        la = layer.linear_attn
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

        state = cache[1]
        outs, states = [], []
        for p in range(S):
            o, state = gated_delta_update(
                q[:, p : p + 1],
                k[:, p : p + 1],
                v[:, p : p + 1],
                a[:, p : p + 1],
                b[:, p : p + 1],
                la.A_log,
                la.dt_bias,
                state,
                None,
            )
            outs.append(o)
            states.append(state)
        out = outs[0] if S == 1 else mx.concatenate(outs, axis=1)

        n_keep = la.conv_kernel_size - 1
        cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        cache[1] = state
        sink.append((cache, states, conv_input, la.conv_kernel_size))

        r = la.out_proj(la.norm(out, z).reshape(B, S, -1))
        h = x + r
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def _rollback(self, caches, sink, total: int, consumed: int):
        if consumed == total:
            return
        for c in caches:
            if isinstance(c, KVCache):
                c.trim(total - consumed)
        for cache, states, conv_input, kernel in sink:
            cache[1] = states[consumed - 1]
            cache[0] = mx.contiguous(conv_input[:, consumed : consumed + kernel - 1, :])

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
    ):
        """max_draft > 0 で受理適応の可変深度になる。

        直前ステップの受理数 a に応じて次の深度を決める:
        全受理なら +2 (上限 max_draft)、全棄却なら 1、部分受理なら a。
        外れ区間の検証税を m=2 の 1.04 倍まで下げ、当たり区間だけ深掘りする。

        draft 源は 2 系統: 文脈 lookup (suffix 再出現の続き、深さ lookup_len) が
        当たればそれを優先し、外れたら MTP 連鎖。lookup が全棄却だったら
        数ステップ休ませる。
        """
        eos = set(eos_ids)
        caches = self.text.make_cache()
        mtp_cache = KVCache()
        prompt = mx.array(prompt_ids)

        t0 = time.perf_counter()
        h_all, _ = self._hidden_forward(prompt, caches, capture=False)
        y_logits = self._head(h_all[:, -1:], self.inner.norm)
        if temp > 0:
            y = mx.random.categorical(
                y_logits[0].astype(mx.float32) / temp
            ).reshape(1)
        else:
            y = mx.argmax(y_logits, axis=-1).reshape(1)
        if prompt.shape[0] > 1:
            self._mtp_append(prompt[1:], h_all[:, :-1], mtp_cache)
        h_last = h_all[:, -1:]
        mx.eval(y)
        ttft = time.perf_counter() - t0

        out_tokens = [int(y.item())]
        if on_tokens:
            on_tokens(out_tokens[:])
        ctx = list(prompt_ids) + out_tokens
        accept_hist = Counter()
        accept_trace = []
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
            if lookup_len > 0 and lookup_cool == 0:
                lk = self._lookup_draft(ctx, lookup_ngram, lookup_cur)
            if lk:
                source = "lookup"
                window = mx.concatenate([y, mx.array(lk)])
            else:
                source = "mtp"
                lookup_cool = max(0, lookup_cool - 1)
                drafts = []
                dh, dtok = h_last, y
                for _ in range(depth):
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
            hs, sink = self._hidden_forward(window, caches, capture=True)
            logits = self._head(hs, self.inner.norm)
            if temp == 0:
                preds = mx.argmax(logits, axis=-1)[0]
                mx.eval(preds, window)
                preds_l, window_l = preds.tolist(), window.tolist()
                n_avail = len(window_l) - 1
                a = 0
                while a < n_avail and preds_l[a] == window_l[a + 1]:
                    a += 1
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
                row = lg[a]
                if a < n_avail:
                    rejected = mx.arange(row.shape[-1]) == window_l[a + 1]
                    row = mx.where(rejected, -mx.inf, row)
                next_tok = int(mx.random.categorical(row).item())
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
            mtp_cache.trim(mtp_cache.offset - mtp_off0)
            true_hiddens = mx.concatenate(
                [h_last, hs[:, : consumed - 1]], axis=1
            )
            self._mtp_append(window[:consumed], true_hiddens, mtp_cache)

            h_last = hs[:, consumed - 1 : consumed]
            y = mx.array([next_tok])
            step_tokens = []
            for t in window_l[1:consumed] + [next_tok]:
                step_tokens.append(t)
                if t in eos:
                    break
            out_tokens.extend(step_tokens)
            ctx.extend(step_tokens)
            if on_tokens:
                on_tokens(step_tokens)
            phase["maint"] += time.perf_counter() - ts

        decode_time = time.perf_counter() - t1
        n_decode = len(out_tokens) - 1
        steps = sum(accept_hist.values())
        return {
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
