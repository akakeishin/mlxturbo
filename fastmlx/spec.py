"""MTP 自己投機デコード。

draft: MTP ブロック(全体の ~1.5%)を連鎖させて n_draft 個の候補を出す。
verify: 本体 1 回の forward で n_draft+1 トークンをまとめて検証する。
m=2 の検証は m=1 の 1.04 倍しかかからない(実測)ので、受理された分が
ほぼそのまま速度倍率になる。greedy では出力は非投機と完全一致する。

線形アテンション 48 層の巻き戻し: 射影と conv は m トークン一括(帯域償却を
維持)、再帰更新だけ位置ごとに回して全位置の状態を保持する。棄却時は
該当位置の状態と conv 窓を差し替えるだけで再計算しない。

Phase D 汎用パック (docs/RESEARCH.md, docs/KERNEL-INTEL.md「深度制御」節):
D1 は MTP 連鎖の深度ラダーを AdaEDL 型の確信度 + 位置別受理率 EMA の
継続/停止ゲートへ置換、D3 は文脈 lookup を suffix automaton (fastmlx/sam.py)
化して仲裁を ReSpec 流のエントロピー起動 + 候補 EMA へ置換、D4 は temp>0 の
棄却サンプリングを Block Verification (arXiv:2403.10444) へ置換した。
いずれも訓練不要・分布保証は不変 (D1/D3 は深度と提案元の選び方だけを変える
ので正しさに無関係、D4 は棄却サンプリングと厳密同一分布であることを
証明・実測で確認済み。詳細は docs/STATUS.md「Phase D 汎用パック」)。
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

# ---------- D1: 確信度ゲート連鎖 ----------
#
# AdaEDL (arXiv:2410.18351) のエントロピー下界 1-sqrt(gamma*H) をリンクごとの
# 確信度信号にし、KERNEL-INTEL.md「深度制御」の位置別受理率 EMA + 期待利得
# 閾値 (reach *= p_d, threshold = h*(1+expected)/(1+d*h)) と等重み平均で
# 併用する。gamma=0.2 は論文の既定値。h=0.19 は checkpoint 型巻き戻し
# (fastmlx の全位置状態保持) のレンジ 0.18-0.20 の中央値。
ADAEDL_GAMMA = 0.2
GATE_ROLLBACK_COST = 0.19
GATE_EMA_ALPHA = 0.2
# Confidence-weighted blend warmup: a gate-truncated step can only update
# the EMA for positions it actually verified, so an under-observed position
# must not be dominated by the (possibly-pessimistic) EMA/prior yet -- see
# _gate_depth docstring.
GATE_EMA_WARMUP = 5
# vanilla MTP 連鎖の位置別受理率の相場 (RESEARCH.md 引用の FastMTP 実測:
# k=1 ~70% / k=2 ~11% / k=3 ~2%)。EMA が温まるまでの事前分布として使う。
_POS_ACCEPT_PRIOR = {1: 0.70, 2: 0.11, 3: 0.02}


def _pos_accept_prior(d: int) -> float:
    if d in _POS_ACCEPT_PRIOR:
        return _POS_ACCEPT_PRIOR[d]
    return 0.02 * (0.3 ** (d - 3))


# ---------- D3: ReSpec 仲裁 ----------
#
# エントロピー閾値は ReSpec (arXiv:2511.01282) 実験値 theta_entropy=1.5,
# 最大遡り長 l=3 をそのまま採用 (語彙サイズが異なるため較正の余地あり、
# docs/STATUS.md に注記)。lambda_e (エントロピーと長さのトレードオフ) と
# EMA 更新率 alpha, 品質閾値 theta_score=0.5 は論文本文に具体値の記載が
# なかったため fastmlx 側で選んだ較正値。
RESPEC_LOOKBACK = 3
RESPEC_LAMBDA_E = 1.0
RESPEC_ENTROPY_THETA = 1.5
RESPEC_EMA_ALPHA = 0.3
RESPEC_SCORE_THETA = 0.5
RESPEC_SCORE_PRIOR = 0.5
RESPEC_BUCKET_WIDTH = 4
RESPEC_MAX_BUCKET = 8


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

    def _mtp_base(self, hiddens: mx.array) -> mx.array:
        """checkpoint 契約 base_hidden_variant=post_norm に合わせて本体最終 norm を適用。
        2x2 実測 (bench/results/mtp-2x2-*.json): post/post が全深度で優位。"""
        return self.inner.norm(hiddens)

    def _mtp_append(self, tok_ids: mx.array, hiddens: mx.array, mtp_cache) -> mx.array:
        """位置ごとの (embed(t_{i+1}), h_i) ペアを MTP に流し K/V を積む。"""
        e = self.inner.embed_tokens(tok_ids[None])
        return self.mtp(e, hiddens, cache=mtp_cache)

    # ---------- D1: 確信度ゲート連鎖 ----------

    @staticmethod
    def _adaedl_bound(entropy: float, gamma: float = ADAEDL_GAMMA) -> float:
        """AdaEDL (arXiv:2410.18351) の受理下界 1 - sqrt(gamma * H)。

        H はドラフト分布 (このリンクの softmax) のシャノンエントロピー
        (自然対数)。下界が負になり得る低確信域は 0 にクリップする。
        """
        return max(0.0, 1.0 - math.sqrt(max(gamma, 0.0) * max(entropy, 0.0)))

    @staticmethod
    def _expected_future_gain(pos_accept_ema: dict, d: int, cap: int) -> float:
        """d より深く続けた場合に追加で受理できるトークン数の期待値。

        位置別 EMA (温まっていなければ FastMTP 実測を事前分布に) の
        累積積を d+1..cap で足し合わせる (標準的な期待受理長の畳み込み)。
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
        """MTP 連鎖のうち実際に検証へ回す長さを確信度ゲートで決める。

        リンク d ごとに AdaEDL 下界と位置別受理率 EMA を混ぜた p_d で
        reach (=prod p_d) を更新し、KERNEL-INTEL.md の
        threshold = h*(1+expected)/(1+d*h) と比較する。reach が閾値を
        下回った時点のリンクまでは残し (その位置自体は依然検証する価値が
        ある)、それより深い分だけ切り捨てる。

        EMA の重みは観測回数で立ち上げる (w = n/(n+GATE_EMA_WARMUP))。
        ゲートが浅い位置で止まった回よりリンクは深い位置の EMA を
        更新できないので (打ち切った先は検証していない = 正解が無い)、
        固定 50/50 で混ぜると「一度低く出た事前分布が観測不足のまま
        効き続けて浅い判定を再生産する」飢餓状態に陥る。観測が少ない
        うちは AdaEDL の瞬時確信度だけで判断し、EMA は実測が積み上がって
        から効かせる。
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

    # ---------- D3: 文脈 lookup (SAM) + ReSpec 仲裁 ----------

    @staticmethod
    def _respec_trigger(
        entropy_hist: list,
        lookback: int = RESPEC_LOOKBACK,
        lambda_e: float = RESPEC_LAMBDA_E,
        theta_entropy: float = RESPEC_ENTROPY_THETA,
    ) -> bool:
        """ReSpec (arXiv:2511.01282) Algorithm 1 の entropy-guided trigger。

        直近 confirmed トークンの target 分布エントロピーについて、
        遡り長 k=1..lookback それぞれの平均エントロピー H_k と
        confidence score C_k = H_k + lambda_e/k を計算し、C_k を最小化する
        k* を選ぶ。その H_k* が theta_entropy 以下なら "予測しやすい文脈"
        として retrieval を起動する。
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
        """SAM の一致長を EMA ランキングのバケットへ量子化する。

        fastmlx の lookup 候補は SAM が返す単一候補 (最長一致の最新出現) な
        ので、ReSpec の「一致位置ごとの EMA」を「一致長バケットごとの EMA」
        に対応させる: 一致が長いほど再出現が偶然でない可能性が高く、
        バケット単位で信頼度を学習するのが自然な単一候補版の類推になる。
        """
        return min(match_len // width, cap)

    # ---------- D4: Block Verification ----------

    @staticmethod
    def _block_verify_tau(p_l: list, u_l: list, n_avail: int) -> int:
        """Block Verification (Sun et al., arXiv:2403.10444, Algorithm 2 /
        Eq. 4-5) specialized to a deterministic (delta) draft proposal.

        fastmlx's draft tokens (MTP argmax chain and SAM lookup) are always
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
        while for fastmlx's deterministic draft the two samplers are
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
        """MTP 連鎖の深度は確信度ゲートで、lookup 起動は ReSpec 流の
        エントロピー閾値で、それぞれステップごとに決める (Phase D1/D3)。

        MTP 連鎖 (source="mtp"): リンクごとに AdaEDL エントロピー下界と
        位置別受理率 EMA を併用した reach/threshold ゲートで、実際に検証へ
        送るリンク数を決める (``_gate_depth``)。深度上限は
        ``max_draft if max_draft > 0 else n_draft`` を流用する。

        文脈 lookup (source="lookup"): fastmlx/sam.py の suffix automaton が
        O(1) 償却で最長一致を追跡する。直近 confirmed トークンの target
        エントロピー履歴が ReSpec の entropy-guided trigger
        (``_respec_trigger``) を満たし、かつ一致長バケットの受理率 EMA が
        閾値以上のときだけ起動する (``_respec_bucket``)。外れれば MTP へ
        フォールバックする。

        temp>0 の検証は Block Verification (arXiv:2403.10444,
        ``_block_verify_tau``) で、逐次棄却サンプリングと厳密同一分布の
        まま受理長が単調非減少になる (docs/STATUS.md 参照)。

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
                self._mtp_append(prompt, self._mtp_base(mtp_hiddens), mtp_cache)
            elif prompt.shape[0] > 1:
                self._mtp_append(prompt[1:], self._mtp_base(h_all[:, :-1]), mtp_cache)
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
        accept_hist = Counter()
        accept_trace = []
        fed_gen = []
        src_hist = {"lookup": Counter(), "mtp": Counter()}
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
            proposal_cap = max_tokens - len(out_tokens) - 1
            if proposal_cap > 0 and track_lookup and self._respec_trigger(entropy_hist):
                match_len, _ = sam.longest_match()
                if match_len >= lookup_ngram:
                    lookup_bucket = self._respec_bucket(match_len)
                    score = quality_ema.get(lookup_bucket, RESPEC_SCORE_PRIOR)
                    if score >= RESPEC_SCORE_THETA:
                        cand_len = max(1, round(lookup_len * score))
                        lk = sam.draft(
                            min(cand_len, proposal_cap), min_len=lookup_ngram
                        )
            if lk:
                source = "lookup"
                window = mx.concatenate([y, mx.array(lk)])
            else:
                source = "mtp"
                lookup_bucket = None
                cap = min(cap_base, proposal_cap)
                drafts = []
                confidences = []
                dh, dtok = self._mtp_base(h_last), y
                for _ in range(max(cap, 0)):
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
                if track_lookup:
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
                    if window_l[a] in eos:
                        accepted_eos = a
                        break
                if accepted_eos is None:
                    next_tok = preds_l[a]
            else:
                # D4: Block Verification (arXiv:2403.10444) replaces
                # sequential rejection sampling. Draft は決定的提案 (delta
                # 分布) なので受理長 tau の閉形式が p_l (逐次棄却と同じ
                # target 確率列) だけで書ける (_block_verify_tau docstring
                # と docs/STATUS.md に導出)。出力分布は逐次棄却と厳密同一。
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
