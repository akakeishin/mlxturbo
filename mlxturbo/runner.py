"""Common entry point for SpecEngine (speculative decoding) and, for models it
cannot handle, ordinary generation.

The SpecEngine in mlxturbo/spec.py assumes the GDN-hybrid-specific shape that
validate_spec_model_contract in mlxturbo/_mlx_compat.py requires
(fa_idx/ssm_idx on model.language_model.model, is_linear on each layer,
advance/trim on the linear cache, and so on). Construction fails with a
TypeError or the like not only for Llama/Gemma/dense Qwen, but also for GDN
hybrids whose layout differs from the mlx-lm shape that existed when
mlxturbo/spec.py was written (the qwen3_5 language_model wrapper). The case
confirmed here is the qwen4_exp architecture, which uses hyper-connections: it
has no model.language_model, its layers have no is_linear/input_layernorm, it
has no final norm, and so on — this is not a mere attribute-name mismatch but
a mismatch with the reimplementation of _hidden_forward/_linear_capture
itself.

``build_runner`` tries to construct a SpecEngine exactly once at startup and,
if that fails, drops to ordinary (non-speculative) generation via
mlx_lm.generate.stream_generate. On either path the shape seen by callers
(cli.py / server.py) — the dict returned by ``Runner.generate(...)`` — is
identical.

``build_runner`` also enables mlxturbo.fused.enable_hyper_connection_kernel()
here (on by default, disabled with ``no_fused=True``). It only swaps the
qwen4_exp (hyper-connections) architecture's `GatedResidual.__call__` for a
Metal kernel at the class level, so it has no effect on the SpecEngine path
(spec.py reimplements forward on its own and never goes through this swap)
and takes effect only on the FallbackRunner path (which calls the model's own
__call__ — that is, the only road Flash-Next/qwen4_exp actually travels).
moe_route and rms_norm_gated came out empty-handed in measurements (the
former is +0.34ms slower; see tools/ablate_moe.py), so they are not enabled.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Protocol

import mlx.core as mx

from ._mlx_compat import TextModelArgs, resolve_local_model_path
from .spec import PREFILL_STEP_SIZE, ChatSession, SpecEngine


class Runner(Protocol):
    """``SUPPORTED_SAMPLING_PARAMS`` (class attribute, set of str) declares
    which of top_p/top_k/min_p/repetition_penalty/presence_penalty/
    frequency_penalty/logit_bias/seed server.py is allowed to pass to this
    runner via ``**sampling_kwargs``. A request that specifies a key not in
    the declaration is rejected by server.py with a 400 before the generate
    call (see the class docstrings of SpecRunner/FallbackRunner).

    ``fallback_reason`` (str | None) is the reason build_runner chose this
    runner as the fallback path. On the non-fallback paths
    (SpecRunner/FlashSpecRunner) it is always None. server.py's /health
    reports it verbatim — it exists to prevent "silently dropping to the
    fallback with no way to notice" (see build_runner's class docstring).
    """

    SUPPORTED_SAMPLING_PARAMS: frozenset
    # logprobs を自分で出せるか (server.py が降格の判定に使う)。宣言しない
    # runner は False 扱い = logprobs 付きのリクエストは非投機へ降ろされる。
    SUPPORTS_LOGPROBS: bool
    fallback_reason: str | None

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        temp: float,
        eos_ids: set,
        on_tokens,
        session: ChatSession | None,
        **sampling_kwargs,
    ) -> dict: ...


class SpecRunner:
    """Speculative decoding path. Uses mlxturbo.spec.SpecEngine directly.

    fly_theta/fly_window are optional keywords for cli.py's
    --fly-theta/--fly-window; they are forwarded verbatim to
    SpecEngine.generate via **extra (when unspecified, SpecEngine's own
    defaults of 0.0/6 apply). server.py does not pass them.

    ``SUPPORTED_SAMPLING_PARAMS``: the declaration server.py uses to decide
    which of a request's sampling parameters (top_p/top_k/min_p/
    repetition_penalty/presence_penalty/frequency_penalty/logit_bias/seed)
    may be forwarded to this path. Here it is ``seed`` only. Why:
    SpecEngine.generate guarantees, for temp>0, a distribution strictly
    identical to rejection sampling, via Block Verification
    (arXiv:2403.10444, spec.py ``_block_verify_tau``). That guarantee stands
    on a closed-form derivation of the acceptance length (docs/STATUS.md)
    which depends on "both the draft's proposal distribution and the
    verifier's target distribution being the raw ``softmax(logits/temp)``";
    truncating the logits with top_p/top_k/min_p, or rewriting them with
    repetition_penalty and the like, changes the target distribution itself,
    so the closed form for the acceptance length becomes a different
    equation and the distribution guarantee silently breaks unless the
    implementation is reworked. Correctly re-deriving and implementing that
    was judged out of scope for this task, so these parameters take option
    (b): server.py rejects them with a 400. ``seed``, on the other hand,
    only changes the initial state of the RNG and not the distribution
    itself, so it can be passed straight through with no effect
    (engine.generate itself takes no seed argument, so it is consumed here by
    calling mx.random.seed()).
    """

    KIND = "spec"
    SUPPORTED_SAMPLING_PARAMS = frozenset({"seed"})

    def __init__(self, engine: SpecEngine, n_draft: int, max_draft: int):
        self.engine = engine
        self.n_draft = n_draft
        self.max_draft = max_draft
        self.fallback_reason = None

    def generate(
        self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, seed=None, **extra
    ):
        if seed is not None:
            mx.random.seed(seed)
        return self.engine.generate(
            prompt_ids,
            max_tokens=max_tokens,
            n_draft=self.n_draft,
            max_draft=self.max_draft,
            temp=temp,
            eos_ids=eos_ids,
            on_tokens=on_tokens,
            session=session,
            **extra,
        )


def _position_local_sampler(temp, top_p, top_k, min_p, logit_bias):
    """位置ごとの logits だけで決まるサンプラーを組む。恒等なら None を返す
    (呼び手は既存経路をそのまま通り、1 ビットも変わらない)。

    返す関数は「生 logits (N, vocab) -> トークン (N,)」。mlx_lm の
    ``make_sampler`` は log 確率を受ける約束なので、ここで正規化してから渡す。
    ``logit_bias`` は語彙長のベクトルに畳んで加算する (位置に依らない定数なので
    位置局所の条件を満たす)。
    """
    if temp <= 0:
        return None
    if not (
        (0 < top_p < 1.0) or top_k > 0 or min_p != 0.0 or logit_bias
    ):
        return None

    from mlx_lm.sample_utils import make_sampler

    base = make_sampler(temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)
    bias_idx = bias_val = None
    if logit_bias:
        bias_idx = mx.array([int(k) for k in logit_bias])
        bias_val = mx.array([float(v) for v in logit_bias.values()])

    def sampler(logits: mx.array) -> mx.array:
        x = logits.astype(mx.float32)
        if bias_idx is not None:
            add = mx.zeros(x.shape[-1], dtype=mx.float32)
            add[bias_idx] = bias_val
            x = x + add
        logprobs = x - mx.logsumexp(x, axis=-1, keepdims=True)
        return base(logprobs)

    return sampler


class FlashSpecRunner:
    """Speculative decoding path for Qwen3.8-Flash-Next (``qwen4_exp``).

    Uses ``mlxturbo.spec_flash.FlashSpecEngine`` (depth-1 speculation with
    MTP as the draft, plus capture/rewind of the GatedDeltaNet state)
    directly. It is a different thing from
    ``SpecRunner``/``mlxturbo.spec.SpecEngine``, which serve the 27B
    (``qwen3_5``) — the model's layer structure (hyper-connections, GDN,
    512-expert MoE) is completely different, so instead of unifying them they
    are split into separate classes (see docs/MTP-FLASH.md).

    For session it reuses ``FallbackSession`` (singular ``.cache`` /
    ``.processed``) as-is. Flash-Next is also a GDN hybrid and cannot rewind
    the linear state to an intermediate position, so the reuse decision this
    function itself makes is still the same two-way choice as
    ``FallbackRunner``: "reuse only when the new prompt is a pure append to
    the processed sequence, otherwise build a new one from scratch" — but the
    range that can actually be reused is wider than that. Because
    ``FallbackSession`` was given ``.checkpoints`` in the same shape as
    ``mlxturbo.spec.ChatSession`` (added 2026-08-29, for thinking support),
    ``server.py``'s ``_select_session``/``_try_checkpoint_restore_session_
    cache`` restore up to the most recent checkpoint even for a prompt that
    branched partway through the processed sequence, and only then hand over
    ``session``. This function knows nothing at all about whether that
    restore happened — as long as ``_select_session`` hands over a restored
    ``session.processed``, the LCP check here always passes it through as a
    "full match" (``_select_session`` guarantees that the post-restore
    ``processed`` is a true prefix of the new prompt). Recording the
    ``checkpoints`` themselves happens at each prefill chunk boundary inside
    ``FlashSpecEngine.generate_stream``
    (reusing ``mlxturbo.spec.CHECKPOINT_RETENTION``/
    ``snapshot_untrimmable_caches`` as-is; the KV/indexer caches are
    trimmable so they need no recording — see the ``generate_stream``
    docstring in spec_flash.py). Because ``KIND`` is not ``"spec"``, the
    ``session_factory`` selection in ``server.py``/``cli.py``
    (``ChatSession if KIND == "spec" else FallbackSession``) is unchanged
    here and still picks ``FallbackSession``.

    ``SUPPORTED_SAMPLING_PARAMS``: ``seed`` に加えて **位置局所な logits 変換**
    (top_p / top_k / min_p / logit_bias) を受ける。**履歴依存のもの
    (repetition_penalty / presence_penalty / frequency_penalty) は受けない。**
    線引きの理由は param 名ではなく形にある:

    ``FlashSpecEngine._verify`` は 1 ラウンドで検証フォワードの全位置を先に
    サンプルし、draft と一致したプレフィックスだけ採用する。位置 j の
    サンプルは ``lg[:, j]`` にしか依存せず、受理判定は samples[0..j-1] にしか
    依存しないので、条件付けても位置 j の分布は歪まない。**この独立性は
    サンプラーの形に依らない**ので、位置局所な変換なら投機ありでも逐次
    サンプリングと厳密一致する (近似ではない)。

    一方ペナルティ系は「それまでに出たトークン列」に依存するため、全位置を
    先に引く形では位置 j のペナルティを j-1 までの履歴で計算できない。載せると
    静かに分布が変わるので、これらは従来どおり server.py が非投機に降ろす。

    2026-09-01 以前はここが ``seed`` だけで、top_p が付いたリクエストは
    まるごと非投機に降格していた。実クライアント (opencode、OpenAI SDK) は
    top_p を既定で送るので、**実トラフィックのほぼ全部が投機の 1.26-1.44x を
    静かに捨てていた**。降格は理論の壁ではなく実装の穴だった。

    ``logprobs`` は今も降格の引き金として残る (検証側 logits から出せる
    見込みはあるが未実装 -- docs/research/IMPROVEMENT-QUEUE.md の D1)。
    """

    KIND = "flash_spec"
    # 位置局所な変換だけ。履歴依存 (repetition/presence/frequency) は入れない
    SUPPORTED_SAMPLING_PARAMS = frozenset(
        {"seed", "top_p", "top_k", "min_p", "logit_bias"}
    )
    # 検証フォワードの logits から出せる。採用位置 j の logits は pair[:j+1] に
    # 正しく条件付いていて、棄却位置のものは採用側に混ざらない。先頭トークン
    # だけ出所が prefill 末尾の logits_tail。
    SUPPORTS_LOGPROBS = True

    def __init__(self, engine, tokenizer=None):
        self.engine = engine
        # logprobs のトークン文字列化にだけ使う (無ければ logprobs は
        # server.py 側が降格させる。build_runner は必ず渡す)
        self.tokenizer = tokenizer
        self.fallback_reason = None

    def generate(
        self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session,
        seed=None, top_p: float = 0.0, top_k: int = 0, min_p: float = 0.0,
        logit_bias: dict | None = None, logprobs: bool = False,
        top_logprobs: int = 0, **extra
    ):
        if seed is not None:
            mx.random.seed(seed)
        sampler = _position_local_sampler(temp, top_p, top_k, min_p, logit_bias)
        logprob_rows: list | None = [] if logprobs else None
        logprob_entries: list = []

        # The same LCP (longest common prefix) contract as
        # FallbackRunner.generate: this function itself only looks for "the
        # whole of processed is a prefix of the new prompt". In practice
        # server.py's _select_session may already have restored processed to
        # an intermediate position via a checkpoint (see this class's
        # docstring), but the caller guarantees that the post-restore
        # processed is a true prefix of the new prompt, so the check here
        # always catches it in this branch.
        #
        # ``checkpoints`` follows the same style as
        # ``mlxturbo.spec.ChatSession`` (inherit the reference directly, do
        # not copy) — ``FallbackSession.invalidate()`` merely **rebinds**
        # ``self.checkpoints`` to a new empty list and does not rewrite the
        # old list that this already-grabbed local variable points at. When
        # the cache is rebuilt (no session / a new one / not an append) there
        # is no point inheriting the old records, so it is emptied.
        prompt_cache = None
        reused = 0
        checkpoints: list | None = None
        resume = None
        if session is not None:
            checkpoints = session.checkpoints
            if session.cache is not None:
                pl = session.processed
                n = min(len(pl), len(prompt_ids))
                lcp = 0
                while lcp < n and pl[lcp] == prompt_ids[lcp]:
                    lcp += 1
                if lcp == len(pl) and lcp <= len(prompt_ids):
                    prompt_cache = session.cache
                    reused = lcp
                    tail = session.tail
                    # From here on this local variable owns session.cache. An
                    # exception during generation leaves the published
                    # session invalid from this point until publish() is
                    # reached (the same reason as FallbackRunner).
                    session.invalidate()
                    if reused > 0 and reused == len(prompt_ids):
                        # Diff-0 reuse: _select_session (mlxturbo/server.py)
                        # only widens its cap to let `reused` reach
                        # len(prompt_ids) exactly when it also found a
                        # session.tail stamped at this same position, so this
                        # should always resolve. If that invariant is ever
                        # violated regardless, feeding generate_stream an
                        # empty ``ids`` with no ``resume`` is not something it
                        # supports -- fall back to leaving one token to
                        # prefill, the cap this module used before the tail
                        # mechanism existed.
                        if tail is not None and tail[0] == reused:
                            resume = tail[1]
                        else:
                            reused -= 1
            if prompt_cache is None:
                prompt_cache = self.engine.model.make_cache()
                checkpoints = []

        remaining_prompt = prompt_ids[reused:]
        ids = mx.array(remaining_prompt)[None]

        tokens: list[int] = []
        t0 = time.perf_counter()
        ttft = None
        accepted = rounds = 0
        tail_out = None
        gen = self.engine.generate_stream(
            ids,
            max_tokens,
            caches=prompt_cache,
            temp=temp,
            eos_ids=eos_ids,
            checkpoints=checkpoints,
            base_pos=reused,
            resume=resume,
            sampler=sampler,
            logprob_rows=logprob_rows,
            **extra,
        )
        try:
            while True:
                step_tokens = next(gen)
                if ttft is None:
                    ttft = time.perf_counter() - t0
                tokens.extend(step_tokens)
                if logprob_rows is not None:
                    # ラウンドごとに畳む。語彙長のベクトルを最後まで溜めると
                    # 512 トークンで数百 MB になる
                    for tok, row in zip(step_tokens, logprob_rows):
                        logprob_entries.append(
                            _logprob_entry_from_row(
                                tok, row, self.tokenizer, top_logprobs
                            )
                        )
                    logprob_rows.clear()
                if on_tokens:
                    on_tokens(step_tokens)
        except StopIteration as stop:
            if stop.value is not None:
                accepted, rounds, tail_out = stop.value
        decode_time = time.perf_counter() - t0 - (ttft or 0.0)
        n_decode = max(len(tokens) - 1, 0)
        if session is not None:
            # FlashSpecEngine's invariant is that the final ``cur`` has been
            # produced/yielded but has not yet been fed into ``prompt_cache``.
            # Publish only the prefix the cache actually contains; the next
            # turn will prefill the trailing cur together with its new suffix.
            # ``checkpoints`` was extended in-place by ``generate_stream``
            # (or reset to ``[]`` above when this call built a fresh cache) —
            # publish it alongside so the next turn's _select_session can
            # restore to it (the same shape as ChatSession.publish in
            # mlxturbo/spec.py). ``tail_out`` (the logits/hyper pair at this
            # call's prefill/decode boundary, from generate_stream) is stamped
            # with the position it belongs to (this call's own full prompt
            # length) so it doubles as the (position, payload) shape
            # ChatSession.tail and _select_session's ``_reuse_cap_for``
            # already use.
            session.publish(
                prompt_cache,
                list(prompt_ids) + tokens[:-1],
                checkpoints=checkpoints,
                tail=(len(prompt_ids), tail_out) if tail_out is not None else None,
            )
        result = {
            "tokens": tokens,
            "ttft_s": ttft or 0.0,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            "prefill_reused": reused,
            "prefill_new": len(prompt_ids) - reused,
            # The same definition as mlxturbo.spec.SpecEngine.generate
            # (n_decode/steps): the effective number of tokens per
            # iteration, excluding the first token that prefill produced.
            # When rounds==0 (max_tokens<=1, so the loop never runs once),
            # report 0.0, the same as SpecEngine.
            "tokens_per_step": (n_decode / rounds) if rounds else 0.0,
        }
        if logprob_rows is not None:
            result["logprobs"] = logprob_entries
        # MLXTURBO_PHASE_TIMERS=1 のときだけ engine が埋める (spec_flash.py)。
        if getattr(self.engine, "last_phase", None):
            result["phase"] = {**self.engine.last_phase, "rounds": rounds}
        return result


class FallbackSession:
    """Per-conversation container for the mlx_lm prompt_cache used by
    FallbackRunner.

    It applies the same contract as spec.ChatSession (if the new prompt is a
    pure append to the previously processed sequence, prefill only the delta;
    otherwise rebuild everything) to the generic KV cache of
    mlx_lm.models.cache. The only model FallbackRunner actually travels with
    (the qwen4_exp/Flash-Next family) is a GDN hybrid whose linear state
    cannot be rewound to an intermediate position (the same constraint as the
    ChatSession docstring in mlxturbo/spec.py), so here too no partial rewind
    (trim) is performed and it falls back to the two-way choice of append or
    full rebuild.

    ``processed``: the token sequence actually fed into this cache so far
    (prompt + the tokens generated up to that point).
    mlx_lm.generate.stream_generate feeds a yielded token into the cache as
    the input of the next step before emitting it (see the analysis at the
    FallbackRunner.generate call site), so whether generation ends early on
    EOS or is cut off at max_tokens, the sequence gathered in ``tokens``
    matches exactly the sequence already fed into the cache.

    ``checkpoints``: the same shape as
    ``mlxturbo.spec.ChatSession.checkpoints`` (``[(position, snapshot),
    ...]``, ascending by position) — so that even a new prompt that branched
    partway through the processed sequence (e.g. because a thinking marker
    was reopened) can be restored up to the most recent prefill chunk
    boundary and then have only the delta prefilled again (consumed by
    ``_try_checkpoint_restore_session_cache`` in mlxturbo/server.py; see the
    ``FlashSpecRunner`` docstring in mlxturbo/runner.py). ``FallbackRunner``
    (this class's other user) uses ``mlx_lm.generate.stream_generate``
    directly and does not take snapshots at chunk boundaries itself, so for
    it this stays empty forever — ``_try_checkpoint_restore_session_cache``
    treats an empty list as falsy and gives up immediately, so as far as that
    path is concerned, the addition of this field itself changes nothing from
    the previous two-way choice of "full match or a new slot".
    """

    def __init__(self):
        self.cache = None
        self.processed: list[int] = []
        self.checkpoints: list = []
        # (position, (logits_last, hyper_prev)) -- see ChatSession.tail in
        # mlxturbo/spec.py and FlashSpecEngine.generate_stream's ``resume``
        # in mlxturbo/spec_flash.py. FallbackRunner never sets this (it has
        # no notion of logits/hyper state to save), so for it this stays
        # None forever, same as ``checkpoints`` for that path.
        self.tail = None

    def invalidate(self):
        """Drop the published cache before it is aliased and mutated in place."""

        self.cache = None
        self.processed = []
        self.checkpoints = []
        self.tail = None

    def publish(self, cache, processed, checkpoints=None, tail=None):
        self.cache = cache
        self.processed = processed
        self.checkpoints = checkpoints if checkpoints is not None else []
        self.tail = tail


def _logprob_entry_from_row(token: int, row, tokenizer, top_n: int) -> dict:
    """1 トークンぶんの logprobs レコードを、log 確率の行から組む。

    ``_logprob_entry`` は mlx_lm の ``GenerationResponse`` を受ける形なので
    投機経路からは使えない。中身は同じで入口だけ「トークン id と log 確率の
    行」にしたもの。**経路が違っても同じリクエストには同じ答えを返す**のが
    要件なので、両者は必ず同じ規約 (log_softmax の段、top_n の取り方) で
    書くこと。
    """
    entry = {
        "token": tokenizer.decode([token]),
        "logprob": float(row[token]),
        "token_id": token,
    }
    top_n = max(top_n, 0)
    if top_n > 0:
        k = min(top_n, row.shape[-1])
        top_idx = mx.argsort(row)[::-1][:k].tolist()
        entry["top_logprobs"] = [
            {
                "token": tokenizer.decode([idx]),
                "logprob": float(row[idx]),
                "token_id": idx,
            }
            for idx in top_idx
        ]
    return entry


def _logprob_entry(resp, tokenizer, top_n: int) -> dict:
    """Convert one ``GenerationResponse`` from mlx_lm into an OpenAI-shaped
    logprobs record (Kimi K3 review item 17; for ``FallbackRunner.generate``
    only).

    ``resp.logprobs`` is the log_softmax vector over the whole vocabulary for
    that decode step (``logits - logsumexp(logits)``; see mlx_lm/generate.py
    lines 420/549), computed by ``mlx_lm.generate.generate_step``/
    ``speculative_generate_step`` immediately before sampling. Both the
    logprob of the token actually sampled and the top-N alternative
    candidates are obtained by simply reading from this same vector — no
    additional forward pass is needed.
    """

    token_logprob = float(resp.logprobs[resp.token])
    entry = {
        "token": tokenizer.decode([resp.token]),
        "logprob": token_logprob,
        "token_id": resp.token,
    }
    top_n = max(top_n, 0)
    if top_n > 0:
        vocab = resp.logprobs.shape[-1]
        k = min(top_n, vocab)
        top_idx = mx.argsort(resp.logprobs)[::-1][:k].tolist()
        entry["top_logprobs"] = [
            {
                "token": tokenizer.decode([idx]),
                "logprob": float(resp.logprobs[idx]),
                "token_id": idx,
            }
            for idx in top_idx
        ]
    else:
        entry["top_logprobs"] = []
    return entry


class FallbackRunner:
    """Ordinary (non-speculative) generation path for models SpecEngine does
    not accept.

    Uses mlx_lm.generate.stream_generate directly. If a session
    (FallbackSession) is passed, the mlx_lm prompt_cache is handed over
    as-is and only the delta is prefilled, but only when the LCP (longest
    common prefix) with the previously processed sequence matches that
    session's entire processed sequence. If it does not match (the
    conversation switched, the template rewrote the history, etc.), it
    silently creates a new prompt_cache and runs the whole prompt through
    again — no partial rewind is performed (see the FallbackSession
    docstring). If session is None, this mechanism itself is bypassed and,
    as before, the temporary cache on the mlx_lm side is relied upon (full
    prefill every turn). n_draft/max_draft/fly_* are also for the
    speculative paths only, so they are merely received via **extra and
    ignored.

    ``SUPPORTED_SAMPLING_PARAMS``: this path has no need to care about a
    speculative distribution guarantee, so everything mlx_lm.sample_utils
    supports is passed straight through.

    ``logprobs``/``top_logprobs`` (Kimi K3 review item 17): collected only
    when requested (``logprobs=True``) — always collecting them would cost
    memory and speed, so the default is not to collect (``logprobs=False``,
    and the ``"logprobs"`` key does not even appear in the result dict).
    When collected, ``res["logprobs"]`` is a ``list[dict]`` in 1:1
    correspondence with the generated token sequence ``res["tokens"]`` (see
    ``_logprob_entry`` for each element). This is not subject to the
    identity-value check on ``SUPPORTED_SAMPLING_PARAMS`` — server.py
    decides the routing via the separate ``logprobs_requested`` argument of
    ``_resolve_runner_for_request`` (if logprobs are requested on a
    speculative path, it demotes to this runner — the speculation in
    SpecRunner/FlashSpecRunner/DraftSpecRunner either does not assume logit
    processing in its distribution-guarantee math, for the same reason as
    top_p and friends, or does assume it but with a different closed form, so
    this non-speculative path is the only one where, after generation, we can
    assert "this really is the logprob of the distribution that was
    sampled").

    ``fallback_reason``: the reason (str) build_runner chose this runner.
    Passed only when build_runner constructs it directly — the existing usage
    where unit tests call ``FallbackRunner(model, tokenizer)`` directly is
    not broken, staying None when omitted.
    """

    KIND = "fallback"
    SUPPORTED_SAMPLING_PARAMS = frozenset(
        {
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "seed",
        }
    )
    SUPPORTS_LOGPROBS = True

    def __init__(self, model, tokenizer, fallback_reason: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.fallback_reason = fallback_reason

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
        logprobs: bool = False,
        top_logprobs: int = 0,
        **extra,
    ):
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        if seed is not None:
            mx.random.seed(seed)

        sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        # If a session was passed, reuse prompt_cache only when the LCP
        # (longest common prefix) matches the session's entire processed
        # sequence (see the FallbackSession docstring). If session is None,
        # as before we do not pass prompt_cache at all — stream_generate/
        # generate_step builds an internal temporary cache each time, exactly
        # the same path as before (this branch also exists so that unit tests
        # hitting this method directly with session=None do not break).
        prompt_cache = None
        reused = 0
        if session is not None:
            if session.cache is not None:
                pl = session.processed
                n = min(len(pl), len(prompt_ids))
                lcp = 0
                while lcp < n and pl[lcp] == prompt_ids[lcp]:
                    lcp += 1
                if lcp == len(pl) and lcp < len(prompt_ids):
                    prompt_cache = session.cache
                    reused = lcp
                    # From here on this local variable owns session.cache.
                    # An exception during generation (including
                    # KeyboardInterrupt) leaves the published session invalid
                    # from this point until publish() is reached (so that a
                    # cache being rewritten in place is never published in a
                    # half-finished state — the same reason as
                    # spec.ChatSession).
                    session.invalidate()
            if prompt_cache is None:
                from mlx_lm.models.cache import make_prompt_cache

                prompt_cache = make_prompt_cache(self.model)

        remaining_prompt = prompt_ids[reused:]

        tokens: list[int] = []
        # Collected only when requested (item 17; see this class's docstring)
        # — if it stays None nothing is ever appended below and the
        # "logprobs" key itself does not appear in the returned dict (so that
        # always collecting does not cost memory and speed).
        collected_logprobs: list[dict] | None = [] if logprobs else None
        t0 = time.perf_counter()
        ttft = None
        stream_kwargs = {}
        if prompt_cache is not None:
            stream_kwargs["prompt_cache"] = prompt_cache
        # stream_generate yields exactly one GenerationResponse per generated
        # token (the very last one is folded into the finish_reason-carrying
        # wrap-up response instead of a plain per-step one, see its source),
        # so collecting .token across every yielded response is lossless: no
        # token is skipped or duplicated regardless of why generation stopped.
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            remaining_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            # Explicitly shared with mlxturbo.spec.PREFILL_STEP_SIZE. Not
            # passing this value would just mean silently riding on
            # mlx_lm.generate's own default (2048, which happens to be the
            # same), and changing either one would make the prefill step size
            # diverge between the paths — the same prompt would be processed
            # with a different chunk width depending on the path, which is
            # creating a bug where the outputs disagree with our own hands.
            prefill_step_size=PREFILL_STEP_SIZE,
            **stream_kwargs,
        ):
            if ttft is None:
                ttft = time.perf_counter() - t0
            tokens.append(resp.token)
            if collected_logprobs is not None:
                collected_logprobs.append(_logprob_entry(resp, self.tokenizer, top_logprobs))
            if on_tokens:
                # stream_generate already ran this token through its own
                # internal detokenizer to produce resp.text (correctly
                # excluding eos, handling multi-byte/BPE trailing-space
                # merges). Re-detokenizing the raw id through a second,
                # independent detokenizer instance server-side was pure
                # waste at best; pass the already-correct text through so
                # there is exactly one detokenizer in the loop for this path.
                on_tokens([resp.token], resp.text)
        decode_time = time.perf_counter() - t0 - (ttft or 0.0)
        n_decode = max(len(tokens) - 1, 0)
        if session is not None:
            # generate_step feeds a yielded token into the cache as the input
            # of the next step before emitting it (and likewise when it ends
            # early on EOS), so tokens matches exactly the generated sequence
            # already fed into prompt_cache.
            session.publish(prompt_cache, list(prompt_ids) + tokens)
        result = {
            "tokens": tokens,
            "ttft_s": ttft or 0.0,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            "prefill_reused": reused,
            "prefill_new": len(prompt_ids) - reused,
            # No speculation, so 1 step = 1 token, fixed. Filled in so that
            # cli.py's display line can assume the same res.keys() on either
            # path.
            "tokens_per_step": 1.0,
        }
        if collected_logprobs is not None:
            result["logprobs"] = collected_logprobs
        return result


# ---------------------------------------------------------- continuous batching
#
# BACKLOG.md §2 / Kimi K3 review item 11. Restricted to FallbackRunner: the
# batch coordinator (mlxturbo.batch.BatchCoordinator) drives
# mlx_lm.generate.BatchGenerator directly against the model's own __call__ —
# exactly the path FallbackRunner.generate already takes (via
# stream_generate), and exactly what mlxturbo.batch's monkey patches exist
# for. SpecRunner/FlashSpecRunner/DraftSpecRunner/LookupSpecRunner keep their
# own draft-token state machines untouched; whether continuous batching can
# coexist with any of them is unmeasured, so server.py only ever routes a
# request here when the runner resolved for it (after any per-request
# downgrade) is a plain FallbackRunner.

BATCH_SUPPORTED_SAMPLING_PARAMS = FallbackRunner.SUPPORTED_SAMPLING_PARAMS


def can_batch(resolved_runner) -> bool:
    """True only for a plain ``FallbackRunner`` — see the module note above."""

    return type(resolved_runner) is FallbackRunner


def batch_tier(coordinator, prompt_ids: list[int], max_tokens: int) -> str:
    """"solo" or "pool" — see mlxturbo.batch.classify's docstring for what
    the two mean and why "solo" is deliberately more conservative than the
    strict correctness condition."""

    from . import batch as _batch

    return _batch.classify(coordinator.model, len(prompt_ids), max_tokens)


def start_batched_generation(
    coordinator,
    prompt_ids: list[int],
    max_tokens: int,
    temp: float,
    on_tokens,
    on_done,
    cancel_event,
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
    """Build one ``mlxturbo.batch.Admission`` and submit it to ``coordinator``.

    Mirrors ``FallbackRunner.generate``'s own sampler/logits_processors
    construction exactly (same ``make_sampler``/``make_logits_processors``
    calls, same ``seed`` handling via ``mx.random.seed``), so the two paths
    sample identically for the same parameters — only *how* the forward pass
    is driven differs (a private ``BatchGenerator`` per request there, one
    shared continuously-refilled ``BatchGenerator`` here).

    ``on_tokens``/``on_done``/``cancel_event`` follow the same three-argument
    contract server.py's own ``_build_streaming_pipeline`` produces
    (``on_tokens(toks)`` per generated token, ``on_done(kind, val)`` exactly
    once with ``"done"``/``"cancelled"``/``"error"``); a non-streaming caller
    passes ``on_tokens=None, on_done=None, cancel_event=None``.

    ``logprobs``/``top_logprobs``/``n_draft``/``max_draft``/``fly_*`` are
    accepted via ``**extra`` and ignored — a logprobs request is excluded
    from batch eligibility upstream (server.py's ``_resolve_runner_for_request``
    already demotes it to non-speculative for the same reason SpecRunner
    does; batching does not yet collect logprobs, so such a request simply
    never reaches this function), and the rest are speculative-only knobs
    that do not apply to FallbackRunner in the first place.

    Returns the ``Admission``'s ``concurrent.futures.Future`` (always
    resolved with a ``res`` dict — see mlxturbo.batch.Admission's docstring
    for the streaming/non-streaming difference in how errors surface).
    """

    import concurrent.futures

    import mlx.core as mx
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    from . import batch as _batch

    if seed is not None:
        mx.random.seed(seed)

    sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
    logits_processors = make_logits_processors(
        logit_bias=logit_bias,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    tier = _batch.classify(coordinator.model, len(prompt_ids), max_tokens)

    future: "concurrent.futures.Future" = concurrent.futures.Future()
    admission = _batch.Admission(
        prompt_ids=list(prompt_ids),
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        tier=tier,
        on_tokens=on_tokens,
        on_done=on_done,
        cancel_event=cancel_event,
        future=future,
    )
    coordinator.submit(admission)
    return future


class DraftSpecRunner:
    """Architecture-independent speculative path that uses mlx_lm's own
    draft-model speculative decoding (``mlx_lm.generate.
    speculative_generate_step``, ``stream_generate(..., draft_model=...)``)
    as-is (Kimi K3 review item 13).

    ``mlxturbo.spec.SpecEngine``/``mlxturbo.spec_flash.FlashSpecEngine`` are
    both dedicated implementations that depend heavily on a specific model
    shape (the GDN hybrids qwen3_5/qwen4_exp, the layer structure that
    ``validate_spec_model_contract`` inspects) and do nothing whatsoever for
    a plain dense transformer such as Llama/Gemma/Mistral/Mixtral (see the
    docstring at the top of this module). Independently of that, mlx_lm
    already has classical speculative decoding — "propose with a small draft
    model, verify with the real model" — as its ``draft_model`` argument
    (``speculative_generate_step`` in ``mlx_lm/generate.py``) — and since it
    never looks at the architecture, it can deliver speculation to any
    HF/MLX-converted model without building a compatibility table. There is
    no need to write our own kernels or verification logic; the upstream
    implementation is used as-is.

    ``SUPPORTED_SAMPLING_PARAMS``: declares the same full set of keys as
    ``FallbackRunner`` (top_p/top_k/min_p/repetition_penalty/
    presence_penalty/frequency_penalty/logit_bias/seed). The reason
    ``SpecRunner``/``FlashSpecRunner`` reject top_p and friends (the
    speculative side derives a closed-form acceptance length itself, on the
    premise that the raw softmax(logits/temp) is the distribution being
    verified, so interposing logit processing breaks that premise) does not
    exist here. Reading ``speculative_generate_step`` (mlx_lm/generate.py
    from line 473), ``sampler``/``logits_processors`` are passed as the same
    closure into both the draft side's ``_step(draft_model, draft_cache, y)``
    and the verification side's ``_step(model, model_cache, y,
    num_draft_tokens + 1)``, and the acceptance test only looks at whether
    "the result the verification side actually sampled independently (with
    the same sampler) against the true prefix at that point happens to match
    the draft side's value" (line 626, ``if tn != dtn: break``). On a match
    the draft's value is emitted as-is, but since it is equal to the sample
    the verification side produced independently, it is itself a sample from
    the target distribution; on a mismatch, the ``tokens[n]`` that the
    verification side sampled is emitted on the spot (line 634) — that is, in
    either branch the emitted token remains "one sample from the verification
    model, after the sampler is applied, against the true prefix at that
    point". Generation keeps coming from the target distribution whether the
    draft hit or missed, so deforming the logits with
    top_p/top_k/repetition_penalty and the like does not break the
    distribution guarantee (it does not rely on a closed-form
    acceptance-length derivation the way SpecRunner/FlashSpecRunner do). The
    state that ``repetition_penalty``/``presence_penalty``/
    ``frequency_penalty`` need — the "generated sequence so far"
    (``prev_tokens``) — is also already handled on the library side, which
    trims what was eaten ahead during the draft stage just before
    verification so it is not double-counted (line 616,
    ``prev_tokens = prev_tokens[: ...]``).

    All of the above is a judgement based on reading the behavior of the
    mlx_lm-side implementation (not written in this task, only read), and
    that reading has not itself been demonstrated in this task — what was
    confirmed on real hardware is only that "the output does not come out
    broken" (greedy match, self-draft configuration); the strict distribution
    match under non-greedy sampling has not been measured (anything under
    docs/ is outside the scope of this change, so the basis for this
    judgement is written here).

    session (per-conversation prompt cache reuse) is not attempted for now:
    it would require carrying two KV caches, draft and target, around as one
    unit, which does not fit the contract of ``FallbackSession`` (singular
    ``.cache``). A session, if passed, is ignored and both models are fully
    prefilled every time — slower, but not incorrect.
    """

    KIND = "draft_spec"
    SUPPORTED_SAMPLING_PARAMS = FallbackRunner.SUPPORTED_SAMPLING_PARAMS

    def __init__(self, model, draft_model, tokenizer, num_draft_tokens: int):
        self.model = model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_draft_tokens = num_draft_tokens
        self.fallback_reason = None

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
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        if seed is not None:
            mx.random.seed(seed)

        sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        tokens: list[int] = []
        t0 = time.perf_counter()
        ttft = None
        n_from_draft = 0
        # stream_generate yields exactly one GenerationResponse per generated
        # token (the same premise as FallbackRunner.generate; see its
        # docstring).
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt_ids,
            max_tokens=max_tokens,
            draft_model=self.draft_model,
            num_draft_tokens=self.num_draft_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=PREFILL_STEP_SIZE,
        ):
            if ttft is None:
                ttft = time.perf_counter() - t0
            tokens.append(resp.token)
            if resp.from_draft:
                n_from_draft += 1
            if on_tokens:
                on_tokens([resp.token], resp.text)
        decode_time = time.perf_counter() - t0 - (ttft or 0.0)
        n_decode = max(len(tokens) - 1, 0)
        # mlx_lm's speculative_generate_step yields, per round, zero or more
        # tokens with from_draft=True (draft hits) and at most one with
        # from_draft=False (the closing token of that round, sampled by the
        # verification model itself). The exact round count is not visible
        # from outside stream_generate, so the number of from_draft=False
        # tokens is used as an approximation of the round count (a value for
        # observability that approximates the same "effective tokens per
        # round" as tokens_per_step in SpecRunner/FlashSpecRunner; it has no
        # bearing on correctness).
        n_verify_rounds = max(len(tokens) - n_from_draft, 1)
        return {
            "tokens": tokens,
            "ttft_s": ttft or 0.0,
            "decode_tps": n_decode / decode_time if decode_time > 0 else 0.0,
            "prefill_reused": 0,
            "prefill_new": len(prompt_ids),
            "tokens_per_step": (n_decode / n_verify_rounds) if tokens else 0.0,
        }


class _FlashMTPDiscoveryError(RuntimeError):
    """An auto-discovery candidate exists but cannot be inspected safely."""


def _discover_flash_mtp_source(model_dir: Path) -> tuple[str, dict | str] | None:
    """MTP auto-discovery for qwen4_exp (Flash-Next) when ``--mtp`` is not
    specified.

    Its purpose is to make it so that "a specialized model carries its own
    accelerator around with it", closing the hole of "forgot to pass it, so
    it fell back". Priority order:

    1. Inside the model's own safetensors shards — if the ``weight_map`` in
       ``model.safetensors.index.json`` has keys starting with ``mtp.``, read
       only the shards containing them with ``mx.load``, gather just the
       ``mtp.*`` keys and return them (not all shards are read — it is
       narrowed to the relevant shards only). In quantized distributions it
       is the norm for the MTP weights to be bundled into the model itself
    2. For a single-file model with no index, ``model.safetensors`` itself
    3. An ``mtp.safetensors`` sidecar directly under the model directory

    When found, returns ``(source_label, spec)``. ``spec`` is a ``dict`` for
    cases 1 and 2 (passed straight to ``weights=`` of
    ``mtp_flash.load_flash_mtp``) and a ``str`` for case 3 (that same
    function's ``path``). If none are found, ``None``.
    """

    discovery_errors: list[str] = []
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text())
            if not isinstance(index_data, dict):
                raise ValueError("top level must be a JSON object")
            weight_map = index_data.get("weight_map", {})
            if not isinstance(weight_map, dict):
                raise ValueError("'weight_map' must be a JSON object")

            shards: set[str] = set()
            for key, shard in weight_map.items():
                if not isinstance(key, str) or not key.startswith("mtp."):
                    continue
                if not isinstance(shard, str) or not shard:
                    raise ValueError(f"weight_map[{key!r}] must name a shard")
                shard_path = Path(shard)
                if shard_path.is_absolute() or ".." in shard_path.parts:
                    raise ValueError(f"unsafe shard path for {key!r}: {shard!r}")
                shards.add(shard)

            if shards:
                collected: dict = {}
                for shard in sorted(shards):
                    shard_path = model_dir / shard
                    if not shard_path.exists():
                        raise FileNotFoundError(f"index references missing shard: {shard_path}")
                    for key, value in mx.load(str(shard_path)).items():
                        if key.startswith("mtp."):
                            collected[key] = value
                if not collected:
                    raise ValueError("indexed MTP shards contain no mtp.* tensors")
                return ("モデル内蔵", collected)
        except Exception as exc:
            discovery_errors.append(
                f"{index_path.name}: {type(exc).__name__}: {exc}"
            )

    # Standard unsharded Hugging Face/MLX layout.  mx.load is lazy for
    # safetensors, so filtering the mapping does not materialize the unrelated
    # base-model tensors on the GPU.
    unsharded = model_dir / "model.safetensors"
    if unsharded.exists():
        try:
            collected = {
                key: value
                for key, value in mx.load(str(unsharded)).items()
                if key.startswith("mtp.")
            }
            if collected:
                return ("モデル内蔵 (model.safetensors)", collected)
        except Exception as exc:
            discovery_errors.append(
                f"{unsharded.name}: {type(exc).__name__}: {exc}"
            )

    sidecar = model_dir / "mtp.safetensors"
    if sidecar.exists():
        return ("サイドカー (mtp.safetensors)", str(sidecar))

    if discovery_errors:
        raise _FlashMTPDiscoveryError("; ".join(discovery_errors))
    return None


#: Every value of Runner.KIND that build_runner can return. server.py's
#: --require-runner uses this set as its choices (the single definition site,
#: so that the KIND strings do not get scattered between here and
#: build_runner's actual branches). "lookup_spec"
#: (mlxturbo.lookup_spec.LookupSpecRunner) is added here as a string literal —
#: to avoid a circular import (lookup_spec.py imports FallbackRunner from
#: runner.py, so the reverse import is not possible).
RUNNER_KINDS = frozenset(
    {
        SpecRunner.KIND,
        FlashSpecRunner.KIND,
        FallbackRunner.KIND,
        DraftSpecRunner.KIND,
        "lookup_spec",
    }
)


def enable_default_fusions(model, log_prefix: str = "", no_fused: bool = False) -> None:
    """出荷経路で有効になる融合・置き換えを全部当てる。

    `build_runner` が起動時に通す設定そのもの。**ベンチや A/B ハーネスは
    必ずここを通すこと** -- 以前 `tools/decode_ab.py` が
    `enable_hyper_connection_kernel()` だけを呼んでいて、gather のソート
    (既定 16) が入らないまま測っていた。同一ハーネス内の相対比較なら符号は
    生き残るが、閾値や交差点は構成で動く。

    env で切り替わるもの (MLXTURBO_HC / _WIDE / _SORT_MIN / _MOE_GLU /
    _MOE_VERIFY / _FAST_QMM) の既定はここが唯一の出どころ。
    """
    from . import fused

    if no_fused:
        print(f"{log_prefix} --no-fused: hyper-connections 融合カーネルと連結射影を無効化")
    else:
        # HC の実装は MLXTURBO_HC で選ぶ: kernel (既定) / compiled / off。
        # kernel は 2 ディスパッチだが sigmoid が bf16 とビット一致しない
        # (kernels/hyper_connection.py の精度の節)。compiled は op 単位で
        # 素と同じ計算の記録なのでビット同一のまま起動回数だけ減る。
        hc_mode = os.environ.get("MLXTURBO_HC", "kernel")
        if hc_mode == "compiled":
            fused.enable_hyper_connection()
            print(f"{log_prefix} hyper-connections: mx.compile 版 (ビット同一)")
        elif hc_mode == "off":
            print(f"{log_prefix} hyper-connections: 素の実装 (MLXTURBO_HC=off)")
        else:
            fused.enable_hyper_connection_kernel()
            print(f"{log_prefix} hyper-connections 融合カーネル有効 (moe_route/rms_norm_gated は実測で"
                  " 空振りのため無効のまま)")
        # enable_moe_shared_fold は実測で逆効果 (verify +1.8ms) につき呼ばない。
        # 連結射影も既定 OFF: 連結で N が変わると qmv のカーネル変種が変わり、
        # 加算順の違いが最終 ulp を動かす疑いがある (tok/step 2.44 -> 2.23 の
        # 低下と時期が一致)。MLXTURBO_WIDE=1 で実験的に有効化。
        if os.environ.get("MLXTURBO_WIDE") == "1":
            wide = fused.enable_wide_projections(model)
            print(f"{log_prefix} 連結射影有効: gdn={wide['gdn']} attn={wide['attn']}"
                  f" shared={wide['shared']} experts={wide['experts']} 層")
        # gather のソート閾値は既定 16 (値は不変で、検証幅 22..44 の添字が
        # ソートされて同じエキスパートの読みが隣接する)。0 で無効化。
        sort_min = int(os.environ.get("MLXTURBO_SORT_MIN", "16"))
        if sort_min:
            fused.enable_gather_sort(sort_min)
            print(f"{log_prefix} gather のソート閾値 {sort_min} (既定 16、値は不変)")
        if os.environ.get("MLXTURBO_MOE_GLU") == "1":
            fused.enable_moe_glu()
            print(f"{log_prefix} moe_glu カーネル有効 (gate+up+silu*mul を 1 ディスパッチ)")
        # enable_moe_verify_gather 自身が MLXTURBO_MOE_VERIFY=1 をゲートに
        # 持っているので、ここでは呼ぶだけで安全 (既定 off が保たれる)。
        # 以前はこの呼び出し自体が無く、環境変数を立ててもサーバーでは
        # 何も起きなかった (B1、Opus 設計レビュー指摘。統合ディスパッチ
        # (C1) 済みなので、他の 3 経路と掛け順で衝突する心配は無い)。
        fused.enable_moe_verify_gather()
        if os.environ.get("MLXTURBO_MOE_VERIFY") == "1":
            print(f"{log_prefix} moe_verify_gather カーネル有効 (verify 幅の gate+up 融合 + down)")
        if os.environ.get("MLXTURBO_FAST_QMM") == "1":
            # 検証フォワード (M=3..8) の密 qmm を 8x8 MMA タイルに通す。
            # stock qmv は M にほぼ比例して重みを読み直すが、MMA タイルは
            # 1 回で済む (Flash-Next 形状の実測: M=3 で qkv -22% / lm_head -33%)。
            # M=2 は stock が勝つので窓の下限は 3。M=1 (draft) と prefill は
            # fast_qmm 自身の窓判定で素通り。
            from . import fast_qmm

            fast_qmm.M_MIN = int(os.environ.get("MLXTURBO_QMM_M_MIN", "3"))
            fast_qmm.enable()
            print(f"{log_prefix} fast_qmm 有効 (M={fast_qmm.M_MIN}..8 を MMA タイルへ)")



def build_runner(
    model,
    tokenizer,
    config,
    args,
    n_draft: int = 3,
    max_draft: int = 8,
    log_prefix: str = "[mlxturbo]",
) -> Runner:
    """A thin wrapper that handles the two of
    ``args.draft_model``/``args.lookup_spec`` ahead of the existing
    spec/flash_spec/fallback selection (``_build_base_runner``, this
    function's former body) (Kimi K3 review items 13/12).

    - Only when ``args.draft_model`` (str | None) is specified does it enter
      the ``DraftSpecRunner`` path, and it does not touch the branches of
      ``_build_base_runner`` (the choice between qwen4_exp/SpecEngine/
      FallbackRunner) at all — when unspecified, the same selection order as
      before is followed unchanged. The draft model is loaded inside this
      function with ``mlx_lm.load`` (a second model, loaded independently of
      the main one). If the tokenizer's vocab_size disagrees with the main
      model's, it raises ``SystemExit(1)`` for the same reason as the check
      that ``mlx_lm.generate``'s own CLI performs (the draft is assumed to
      use the same tokenizer as the main model) — the same "no escape hatch"
      treatment as an explicit ``--mtp``. Self-drafting, i.e. specifying the
      same model as the draft (which will not make things faster but does
      verify the wiring), also passes this check (because the vocab_size
      matches).
    - ``args.lookup_spec`` (bool) swaps in
      ``mlxturbo.lookup_spec.LookupSpecRunner`` only when what
      ``_build_base_runner`` selected is a ``FallbackRunner`` (= a model that
      does not fit the spec/flash_spec contract) (Kimi K3 review item 12). If
      spec/flash_spec was selected it does nothing — model-specific
      speculation is already in effect, so there is no motive to stack an
      n-gram lookup on top.

    ``args`` is an argparse.Namespace carrying ``model``/``original``/
    ``mtp_bits``/``no_mtp``/``no_fused``/``draft_model``/
    ``num_draft_tokens``/``lookup_spec``/``lookup_max_draft``/
    ``lookup_min_match`` (the arguments of both cli.py and server.py have
    this shape. cli.py does not have these newer flags, so
    ``getattr(..., None)``/``getattr(..., False)`` tolerate their absence).
    """

    draft_model_path = getattr(args, "draft_model", None)
    if draft_model_path:
        from mlx_lm import load as mlx_lm_load

        num_draft_tokens = getattr(args, "num_draft_tokens", None) or 4
        print(
            f"{log_prefix} --draft-model {draft_model_path}: mlx_lm draft-model 投機"
            f" (アーキテクチャ非依存、DraftSpecRunner) を構築します"
            f" (num_draft_tokens={num_draft_tokens})"
        )
        draft_model, draft_tokenizer = mlx_lm_load(draft_model_path)
        model_vocab = getattr(tokenizer, "vocab_size", None)
        draft_vocab = getattr(draft_tokenizer, "vocab_size", None)
        if draft_vocab != model_vocab:
            print(
                f"{log_prefix} --draft-model {draft_model_path} のトークナイザが本体"
                f" モデルと一致しません (vocab_size {draft_vocab} != {model_vocab})。"
                " draft モデルは本体と同じトークナイザ (同系統モデル、self-draft を"
                " 含む) である必要があります。終了します。"
            )
            raise SystemExit(1)
        print(
            f"{log_prefix} draft-model 投機デコード有効 (DraftSpecRunner, "
            f"draft={draft_model_path})"
        )
        return DraftSpecRunner(model, draft_model, tokenizer, num_draft_tokens=num_draft_tokens)

    resolved = _build_base_runner(
        model, tokenizer, config, args, n_draft=n_draft, max_draft=max_draft, log_prefix=log_prefix
    )

    if getattr(args, "lookup_spec", False) and resolved.KIND == FallbackRunner.KIND:
        from .lookup_spec import LookupSpecRunner

        wrapped = LookupSpecRunner(
            model,
            tokenizer,
            max_draft=getattr(args, "lookup_max_draft", None) or 8,
            min_match=getattr(args, "lookup_min_match", None) or 2,
        )
        # Carry over the reason the original FallbackRunner held ("why
        # spec/flash_spec was not selected") — /health's fallback_reason
        # stays meaningful information on this path too (the policy of
        # preventing "silently dropping to the fallback with no way to
        # notice"; see build_runner's original docstring).
        wrapped.fallback_reason = resolved.fallback_reason
        note = (
            "有効"
            if wrapped.trimmable
            else "無効 (このモデルの KV キャッシュが trim 不可のため、通常生成のまま)"
        )
        print(f"{log_prefix} n-gram lookup 投機 (LookupSpecRunner) を有効化: {note}")
        return wrapped

    return resolved


def maybe_build_batch_coordinator(
    resolved_runner,
    model,
    executor,
    max_batch: int,
    eos_ids,
    prefill_step_size: int | None = None,
    log_prefix: str = "[mlxturbo]",
    primary_runner=None,
):
    """``None`` unless both hold: ``--max-batch`` asked for more than 1, and
    ``resolved_runner`` is a plain ``FallbackRunner`` (see ``can_batch``). The
    default (``--max-batch`` omitted or 1) must produce ``None`` here so that
    server.py's existing fully-serial behavior is exactly unchanged when the
    flag is not used — this function is the single place that decides "does
    continuous batching exist on this server at all".

    ``resolved_runner`` is not necessarily server.py's primary ``STATE.runner``
    — when the primary is speculative (spec/flash_spec/draft_spec/
    lookup_spec), server.py passes ``STATE.downgrade_runner`` instead (the
    same plain ``FallbackRunner``, wrapping the identical model, that
    per-request downgrades for non-identity sampling/logprobs already route
    to today — see ``_resolve_runner_for_request`` in server.py). Either way
    this never touches the speculative runner itself: an ordinary request
    served by spec/flash_spec keeps going through STATE.lock exactly as
    before, unaffected by --max-batch.

    Also calls ``mlxturbo.batch.enable_batch_cache()`` when building one —
    harmless to call even when the served model is not qwen4_exp (the patches
    only touch qwen4_exp's own classes; see that module's docstring), and
    necessary when it is.

    ``primary_runner`` (省略可) は server.py の実際の ``STATE.runner``
    (上の段落で説明した ``resolved_runner`` とは別物) で、これを渡すのは
    次の 1 組み合わせだけを警告するため: Flash-Next (``qwen4_exp``) を
    ``FlashSpecRunner`` (MTP 投機) で提供しつつ同時に ``--max-batch`` も
    有効、というケース。``enable_batch_cache()`` は qwen4_exp 自身の
    メソッドをプロセス全体で差し替える (``mlxturbo/batch.py`` のモジュール
    docstring 参照) ため、FlashSpecEngine 自体は batch coordinator に
    降格されなくても、その decode はこの差し替え後のメソッドを踏む。
    現時点で数値は壊れないが、``MLXTURBO_WIDE=1`` の ``_wide_qkv`` 分岐
    (``fused.enable_wide_projections``) は batch 側の写しに未実装なので、
    この組み合わせで有効にすると投機 decode の速度だけが (--max-batch
    無しの場合と違う形で) 変わりうる (恒久解は写しのシーム化 --
    BACKLOG 段 1-2、ここでは対象外)。「黙って別構成に落ちて気づけない」を
    避けるこの repo の規律を、単体では例外にならないこのケースに適用した
    ものが以下の警告ログ (C2、Opus 設計レビュー指摘)。
    """

    if max_batch <= 1 or not can_batch(resolved_runner):
        return None

    from . import batch as _batch
    from .spec import PREFILL_STEP_SIZE

    if (
        primary_runner is not None
        and getattr(primary_runner, "KIND", None) == FlashSpecRunner.KIND
        and getattr(getattr(model, "args", None), "model_type", None) == "qwen4_exp"
    ):
        print(
            f"{log_prefix} 警告: 主 runner が FlashSpecRunner (Flash-Next MTP 投機) で"
            " --max-batch も有効です。enable_batch_cache() はプロセス全体の"
            " qwen4_exp 用メソッドを差し替えます (現時点で数値は壊れませんが、"
            " MLXTURBO_WIDE=1 の連結射影分岐は batch 側の写しに未実装のため"
            " 有効にすると投機 decode の速度だけが変わりえます。恒久解は写しの"
            " シーム化 -- BACKLOG 段 1-2)"
        )

    _batch.enable_batch_cache()
    coordinator = _batch.BatchCoordinator(
        model,
        executor,
        max_batch=max_batch,
        prefill_step_size=prefill_step_size or PREFILL_STEP_SIZE,
        eos_ids=eos_ids,
    )
    print(
        f"{log_prefix} 継続バッチング有効 (--max-batch {max_batch}, FallbackRunner 限定)。"
        " QSA が有効になりうるリクエスト (プロンプト長 + max_tokens が"
        " indexer_budget を超えうるもの) は自動で単独実行に倒します"
        " (mlxturbo/batch.py の classify() 参照)"
    )
    return coordinator


def _build_base_runner(
    model,
    tokenizer,
    config,
    args,
    n_draft: int = 3,
    max_draft: int = 8,
    log_prefix: str = "[mlxturbo]",
) -> Runner:
    """Try to construct a SpecEngine and drop to ordinary generation if the
    model's shape does not fit.

    The body that chooses between qwen4_exp/SpecEngine/FallbackRunner, called
    from ``build_runner`` (this function's original name; it was split out
    because a thin wrapper that intercepts ``--draft-model``/
    ``--lookup-spec`` was added).

    ``args`` is an argparse.Namespace carrying ``model``/``original``/
    ``mtp_bits``/``no_mtp``/``no_fused`` (the arguments of both cli.py and
    server.py have this shape).

    Dropping to the fallback is narrowed to only the cases where we can judge
    that "this model's layout does not fit SpecEngine's contract". If genuine
    failures such as corrupt weights or a Metal allocation failure were also
    disguised as "unsupported architecture" and silently fell back, the cause
    would become impossible to determine:

    - the absence of ``model.args.text_config`` (= not even in the VLM
      wrapper form to begin with) is an ``AttributeError`` — this is a
      legitimate "out of scope" verdict
    - ``load_cli_mtp`` absorbs missing weights itself and returns ``None``
      (designed not to propagate failures this far). An exception that
      propagates here anyway is a genuine bug
    - ``mx.eval(mtp.parameters())`` is not caught here. If it fails it is a
      Metal allocation failure or corrupt weights, which is real damage
      rather than a fallback case, so let it fail loudly
    - only the ``TypeError``/``ValueError``/``RuntimeError`` thrown by
      ``validate_spec_model_contract`` while constructing
      ``SpecEngine(model, mtp)`` is the formal signal for "contract mismatch
      = out of scope"

    The MTP for qwen4_exp (Flash-Next) is looked for in 3 stages (in priority
    order; see ``_discover_flash_mtp_source``):

    1. an explicit ``--mtp PATH`` — highest priority. Because it is the
       operator's explicit specification, starting up silently in a slow
       (non-speculative) configuration when it cannot be read is treated as
       something other than an "out of scope" verdict. The load is retried
       exactly once (assuming a transient GPU Timeout caused by an external
       SSD), and if it still cannot be read it does not fall back but exits
       with ``SystemExit(1)`` — there is no escape-hatch flag
    2. inside the model's own safetensors shards — if the ``weight_map`` in
       ``model.safetensors.index.json`` has keys starting with ``mtp.``, read
       only those shards, gather the ``mtp.*`` tensors and use them (in
       quantized distributions it is the norm for the MTP weights to be
       bundled into the model itself)
    3. an ``mtp.safetensors`` sidecar directly under the model directory

    Failures of 2 and 3 (auto-discovery) are not an explicit specification,
    so they do not exit — the retry applies to 2 and 3 as well, but if they
    still fail it falls back. If none are found it falls back as before.

    A runner that fell back (``FallbackRunner``) carries the reason in
    ``fallback_reason`` (str): one of ``"MTP が見つからない (...)"`` /
    ``"MTP 自動発見 (<出典>) の読み込みに失敗: <理由>"`` /
    ``"spec 契約検証に失敗: <理由>"``. On runners other than the fallback,
    ``fallback_reason`` is always None (see the Runner Protocol).
    """

    from .cli import load_cli_mtp
    from .ngram_stream import warn_if_not_installed

    # If --ngram is forgotten, the n-gram table is used for generation still
    # at its initial values. The output comes out to the very end regardless,
    # so unless the alarm is sounded here nobody notices
    warn_if_not_installed(model)

    enable_default_fusions(model, log_prefix, args.no_fused)

    # Qwen3.8-Flash-Next (qwen4_exp) + MTP (explicit or auto-discovered) ->
    # the FlashSpecEngine path (docs/MTP-FLASH.md). The 27B (qwen3_5) has a
    # different model_type, so it never enters the branch below at all and
    # passes through the rest of this function (the existing
    # SpecEngine/FallbackRunner branches) unchanged — the 27B-side path has
    # not been altered. For cli.py calls without ``args.mtp`` (it has no
    # --mtp argument), getattr returns None so it naturally lands on the
    # auto-discovery side.
    mtp_path = getattr(args, "mtp", None)
    if getattr(model.args, "model_type", None) == "qwen4_exp":
        from . import mtp_flash, spec_flash
        from mlx.utils import tree_flatten

        mtp_bits = getattr(args, "mtp_bits", None)
        quant = {"group_size": 64, "bits": mtp_bits} if mtp_bits else None

        explicit = bool(mtp_path)
        if explicit:
            source_label = "明示指定 (--mtp)"
            load_path, load_weights = mtp_path, None
        else:
            # ``args.model`` may be a Hugging Face repo id.  The model has
            # already been loaded by this point, so resolve the corresponding
            # local snapshot instead of treating ``org/repo`` as a relative
            # filesystem path and silently missing its embedded MTP tensors.
            model_ref = getattr(args, "model", "") or "."
            try:
                model_dir = resolve_local_model_path(model_ref)
            except Exception:
                # ``build_runner`` is also a public/testable boundary and can
                # be called with an already-constructed model plus a synthetic
                # non-cached name.  Preserve the old "no candidate" behavior
                # there; normal CLI/server flow has loaded the repo already.
                model_dir = Path(model_ref)
            try:
                discovered = _discover_flash_mtp_source(model_dir)
            except _FlashMTPDiscoveryError as exc:
                reason = f"MTP 自動発見に失敗: {exc}"
                print(f"{log_prefix} {reason} — 通常生成にフォールバックします")
                return FallbackRunner(model, tokenizer, fallback_reason=reason)
            if discovered is None:
                reason = (
                    "MTP が見つからない (--mtp で指定するか、モデルディレクトリに"
                    " mtp.safetensors を置く)"
                )
                print(f"{log_prefix} {reason} — 通常生成にフォールバックします")
                return FallbackRunner(model, tokenizer, fallback_reason=reason)
            source_label, spec = discovered
            if isinstance(spec, dict):
                load_path, load_weights = None, spec
            else:
                load_path, load_weights = spec, None
            print(f"{log_prefix} MTP を自動発見: {source_label}")

        mtp = None
        last_exc: Exception | None = None
        # The load is attempted at most twice (retried exactly once). There
        # was a real case of a GPU Timeout right before this (a transient
        # failure caused by mmap reads from an external SSD); the main cause
        # was eliminated by evaluating tensor by tensor, but this is
        # insurance against the transient failures that can still happen.
        # The same retry applies not only to the explicit specification
        # (--mtp) but also to auto-discovery (bundled in the model /
        # sidecar). If the second attempt fails too, it is deemed a permanent
        # rather than a transient failure (wrong path, format mismatch, etc.).
        for attempt in range(2):
            try:
                mtp = mtp_flash.load_flash_mtp(
                    load_path, model.args.text, quantize=quant, weights=load_weights
                )
                # Corrupt weights / a Metal allocation failure fail loudly
                # here (deliberately). But not as a single bulk mx.eval: the
                # sidecar lazily mmap-reads 5.2GB from an external SSD, and
                # piling every quantization kernel into one command buffer
                # stalls the GPU on USB page faults and hits Metal's watchdog
                # (GPU Timeout Error) every single time. Cutting the eval per
                # tensor keeps each buffer short, so the slowness of the read
                # merely turns into waiting time.
                for _name, p in tree_flatten(mtp.parameters()):
                    mx.eval(p)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                mtp = None
                if attempt == 0:
                    print(
                        f"{log_prefix} MTP ロードに失敗、再試行します: "
                        f"{type(exc).__name__}: {exc}"
                    )

        if last_exc is not None:
            if explicit:
                # ``--mtp`` is the operator's explicit specification, so
                # starting up silently in a slow (non-speculative)
                # configuration when it cannot be read is treated as
                # something other than an "unsupported architecture" verdict
                # — do not fall back; state the reason and exit. There is no
                # escape-hatch flag.
                print(
                    f"{log_prefix} --mtp {mtp_path} (qwen4_exp) を再試行しても"
                    f"読み込めません ({type(last_exc).__name__}: {last_exc})。"
                    " この --mtp を外せば、モデル内蔵/サイドカーの MTP を"
                    " 自動で探す運用に切り替わります (見つからなければ MTP"
                    " 無しで起動)。終了します。"
                )
                raise SystemExit(1)
            # A failure of auto-discovery (bundled in the model / sidecar) is
            # not an explicit specification, so do not exit — tip over to the
            # fallback instead.
            reason = (
                f"MTP 自動発見 ({source_label}) の読み込みに失敗: "
                f"{type(last_exc).__name__}: {last_exc}"
            )
            print(f"{log_prefix} {reason}; 通常生成にフォールバックします")
            return FallbackRunner(model, tokenizer, fallback_reason=reason)

        # --mtp-depth の既定は None (未指定)。その場合はモジュールの既定を使う。
        depth = getattr(args, "mtp_depth", None) or spec_flash.MTP_DEPTH
        engine = spec_flash.FlashSpecEngine(model, mtp, depth=depth)
        bits_note = f"{mtp_bits}bit" if mtp_bits else "bf16"
        print(
            f"{log_prefix} Flash-Next 投機デコード有効 (FlashSpecEngine, MTP: あり"
            f" [{source_label}], {bits_note})"
        )
        return FlashSpecRunner(engine, tokenizer)

    try:
        text_args = TextModelArgs.from_dict(model.args.text_config)
    except AttributeError as exc:
        reason = f"spec 契約検証に失敗: text_config なし ({type(exc).__name__}: {exc})"
        print(
            f"{log_prefix} 非対応モデルにつき通常生成にフォールバック "
            f"(text_config なし: {type(exc).__name__}: {exc})"
        )
        return FallbackRunner(model, tokenizer, fallback_reason=reason)

    mtp = load_cli_mtp(args.model, config, text_args, args.original, args.mtp_bits, args.no_mtp)
    if mtp is not None:
        # Corrupt weights / a Metal allocation failure fail loudly here
        # (deliberately).
        mx.eval(mtp.parameters())

    try:
        engine = SpecEngine(model, mtp)
    except (TypeError, ValueError, RuntimeError) as exc:
        reason = f"spec 契約検証に失敗: {type(exc).__name__}: {exc}"
        print(
            f"{log_prefix} 非対応モデルにつき通常生成にフォールバック "
            f"(SpecEngine 契約検証エラー: {type(exc).__name__}: {exc})"
        )
        return FallbackRunner(model, tokenizer, fallback_reason=reason)

    mtp_note = "MTP: なし" if mtp is None else "MTP: あり"
    print(f"{log_prefix} 投機デコード有効 ({mtp_note} / lookup: 有効)")
    return SpecRunner(engine, n_draft=n_draft, max_draft=max_draft)
