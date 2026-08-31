"""OpenAI-compatible / Anthropic-compatible HTTP server.

Uses the existing speculative decoding engine (mlxturbo.spec.SpecEngine) as-is.
The model is loaded exactly once at startup and kept resident, and requests are
serialized with a global lock: the premise is a 91GB-class model on a 128GB
machine, so concurrent execution (parallel batching) is not done yet (a
BatchGenerator-based scheduler is the next step). Waiting requests keep their
connection open too (they simply await on the asyncio lock) until the lock
frees up.

Because both the OpenAI and Anthropic APIs are stateless, the client resends
the whole conversation history every turn, so the session (mlxturbo.spec.ChatSession
on the SpecEngine path, mlxturbo.runner.FallbackSession on the FallbackRunner
path) is looked up per request from ``STATE.session_pool`` (a per-conversation
LRU pool, see ``_select_session``), which owns them for the duration of the
request. If the head of the new prompt matches the entire already-processed
token sequence of an existing slot (i.e. it is a pure append), that slot is
picked up and its prefill reused. If nothing matches, a new slot is allocated
(and if the pool has reached its limit, the least recently used slot is
discarded with ``popitem(last=False)``). That limit exists so per-conversation
KV does not pile up without bound on top of a 91GB-class model; the field names
differ, but the same pool is used on both the SpecEngine and FallbackRunner
paths (``STATE.session_factory`` decides at startup which class gets stacked
into it).

The serialization lock (``STATE.lock``) has not been removed in this pass — for
now it remains the only mechanism that guarantees "one request = one
generation", but because session ownership is now per-request, merely removing
the lock would no longer let simultaneous generations for different
conversations destroy each other's session (removing the lock itself is work
for the next scheduler pass). Concurrent access to ``STATE.session_pool``
itself (two requests selecting or evicting slots at the same time) is still
assumed to be protected by this lock, so when the lock is removed the pool
operations themselves will need separate mutual exclusion.

For models SpecEngine does not accept (Llama/Gemma/dense Qwen, or GDN hybrids
whose layout differs), mlxturbo.runner.build_runner automatically falls back at
startup to ordinary (non-speculative) generation via
mlx_lm.generate.stream_generate. Both paths look identical from the HTTP layer
(Runner.generate returns the same dict), and a one-line log at startup says
which path is active. FallbackRunner also reuses mlx_lm's prompt_cache through
the session (under the same LCP contract as SpecEngine). See mlxturbo/runner.py
for details.

MLX's computation graph (including the model weights and the KV cache) is bound
to the thread that loaded it. Escaping to another thread via asyncio.to_thread
or a general-purpose thread pool crashes with "There is no Stream(gpu, N) in
current thread" (confirmed by measurement: doing the load and the forward on
the same thread works, on different threads it does not). Therefore both the
model load and the generate calls are pinned to a **dedicated single worker
thread** (``STATE.executor``, ``max_workers=1``). Requests were already
designed to be serialized by the global lock, so concentrating them on a single
thread loses no concurrency.

The streaming path holds the Future from ``executor.submit(worker)`` inside the
lock, and even when it exits early because of a client disconnect
(StreamingResponse calling ``aclose()`` on the generator = ``GeneratorExit``)
or an error, it waits for that Future in ``finally`` before releasing the lock.
Otherwise the next request could take the lock while a still-generating worker
is touching the session assigned to this request, and two generations would be
rewriting the same session at once.

For streaming responses, validation (every check that could yield a 400) is
completed inside the endpoint function before the StreamingResponse is
assembled; after that, the first SSE event (the role delta chunk for OpenAI,
message_start for Anthropic) is emitted immediately without waiting for
generation to start (lock acquisition, worker submission) — in workloads where
TTFT (prefill) dominates, the time until the client receives its first byte is
itself the real cause of disconnects. There is no path where something
400-worthy is discovered after 200 has been returned (every check that could
discover one completes before StreamingResponse is returned).

Handling of thinking (the reasoning process): only OpenAI's
``reasoning_effort`` and Anthropic's ``thinking`` are read (there are no
mlxturbo-specific fields). The value is mapped to a token budget, and
``ThinkingRouter`` splits the raw token stream into the two channels
reasoning and content. The markers are taken from mlx_lm.TokenizerWrapper's
public API (``has_thinking``/``think_start_tokens``/``think_end_tokens``); for
models where they cannot be obtained, everything stays on a single content
channel (no separation). When the budget is exceeded there is no in-place
restart that "forcibly closes the thinking block and makes the model continue
with the body" (because neither SpecEngine nor mlx_lm.stream_generate permits
interrupting a generation in progress) — instead the cut-off is observable:
output past that point is not forwarded to the client. See the ThinkingRouter
docstring for details.
"""

import argparse
import asyncio
import concurrent.futures
import functools
import hashlib
import hmac
import itertools
import json
import os
import queue
import secrets
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from importlib import metadata as _importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterator

import mlx.core as mx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ._mlx_compat import mlx_lm_load
from .cli import MTP_PATH_ENV
from .runner import (
    RUNNER_KINDS,
    FallbackRunner,
    FallbackSession,
    Runner,
    build_runner,
    can_batch,
    maybe_build_batch_coordinator,
    start_batched_generation,
)
from .runner import batch_tier as _runner_batch_tier
from .spec import PREFILL_STEP_SIZE, ChatSession, restore_untrimmable_caches

app = FastAPI()


def _mlxturbo_version() -> str:
    """The version of this distribution. Returns ``[project] version`` from
    pyproject.toml verbatim via ``importlib.metadata`` (when running in an
    environment where the package is installed) — no second copy maintained in
    a separate file. Running uninstalled (rare) falls back."""

    try:
        return _importlib_metadata.version("mlxturbo")
    except _importlib_metadata.PackageNotFoundError:
        return "0.0.0-unknown"


_FASTMLX_VERSION = _mlxturbo_version()

# Paths that --api-key / graceful shutdown let straight through. They exist for
# monitoring and connectivity checks, so they always return 200 even when a key
# is configured and even during shutdown (the policy described in main()'s
# docstring).
_UNGATED_PATHS = frozenset({"/health", "/api/hello"})

# Set by main() on the first SIGTERM/SIGINT. The gate middleware watches this
# and refuses new requests with 503 (separately from uvicorn's own graceful
# shutdown waiting for in-flight requests to finish, the ASGI layer also has to
# stop accepting new ones — requests arriving over an existing keep-alive
# connection cannot be turned away by uvicorn merely closing the socket).
_SHUTTING_DOWN = False


def _protocol_for_path(path: str) -> str:
    return "anthropic" if path == "/v1/messages" else "openai"


def _busy_response(protocol: str, message: str) -> JSONResponse:
    """Return a 503 (queue limit reached / new request during graceful
    shutdown) in the shape of the given protocol. Attaches Retry-After so the
    client can try again."""

    if protocol == "anthropic":
        resp = _anthropic_error(message, status=503, err_type="overloaded_error")
    else:
        resp = _openai_error(message, status=503, err_type="server_error", code="server_busy")
    resp.headers["Retry-After"] = "1"
    return resp


def _extract_api_key(request: Request) -> str | None:
    """Accept both ``Authorization: Bearer <key>`` and ``x-api-key: <key>`` on
    either path (OpenAI-style/Anthropic-style), because client implementations
    vary — following the policy at the top of server.py."""

    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth[len("bearer ") :].strip()
        if token:
            return token
    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        return x_api_key
    return None


def _unauthorized_response(protocol: str) -> JSONResponse:
    if protocol == "anthropic":
        return _anthropic_error(
            "invalid x-api-key", status=401, err_type="authentication_error"
        )
    return _openai_error(
        "Incorrect API key provided.",
        status=401,
        err_type="invalid_request_error",
        code="invalid_api_key",
    )


@app.middleware("http")
async def _gate_requests(request: Request, call_next):
    """Handle authentication (``--api-key``) and refusal of new requests during
    graceful shutdown in one place, at the entrance to every path except
    ``/health`` and ``/api/hello``.

    Both act as the first gate deciding whether a request is accepted at all,
    ahead of the per-endpoint validation (the 400 family) inside each endpoint.
    The keys come from ``STATE.api_keys`` (treated as empty before startup has
    completed = no authentication, which does not change the default behavior).
    """

    if request.url.path in _UNGATED_PATHS:
        return await call_next(request)

    protocol = _protocol_for_path(request.url.path)

    if _SHUTTING_DOWN:
        return _busy_response(protocol, "server is shutting down")

    api_keys = STATE.api_keys if STATE is not None else frozenset()
    if api_keys:
        supplied = _extract_api_key(request)
        if supplied is None or not any(
            secrets.compare_digest(supplied, k) for k in api_keys
        ):
            return _unauthorized_response(protocol)

    return await call_next(request)


def _add_cors_middleware(fastapi_app: FastAPI, allowed_origins: list[str]) -> None:
    """Called from main() only when ``--allowed-origins`` is given. By default
    (not specified) it is never called at all = no CORSMiddleware is installed,
    and cross-origin fetches from a browser stay blocked (local use only). Use
    it when you want to call this server directly from a browser-based UI such
    as Open WebUI.
    """

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@dataclass
class ModelState:
    runner: Runner
    tokenizer: object
    # LRU pool of per-conversation sessions (ChatSession/FallbackSession). The
    # key is not the identity of the conversation but merely a serial number
    # for insertion-order bookkeeping (see _select_session) — selection is done
    # by LCP against each slot's processed sequence, not by the key.
    session_pool: "OrderedDict[int, object]"
    # Factory for new slots, decided at startup according to the kind of runner
    # (ChatSession for SpecRunner, FallbackSession for FallbackRunner).
    session_factory: Callable[[], object]
    lock: asyncio.Lock
    executor: concurrent.futures.ThreadPoolExecutor
    model_name: str  # served id: GET /v1/models and the "model" field of every response
    eos_ids: set
    max_tokens_cap: int
    default_temp: float
    created_ts: int
    # Raw --model value at startup, as opposed to model_name (the served
    # name). Only GET /api/status reads it.
    model_path: str = ""
    max_sessions: int = 8  # Limit on session_pool (LRU). It exists so that
    # per-conversation KV does not pile up without bound on top of a 91GB-class
    # model, and it can be changed with --max-sessions.
    # Upper limit on prompt length (in tokens). By default it automatically
    # takes the smaller of the model config's max_position_embeddings and the
    # value derived from the actual limit Metal can allocate in one go
    # (_resolve_default_max_context_tokens); --max-context-tokens overrides it.
    # None means nothing was detected and the guard is disabled.
    # _check_context_length consults it on all 4 paths
    # (chat/anthropic/completions/responses).
    max_context_tokens: int | None = None
    session_key_seq: Iterator[int] = field(default_factory=lambda: itertools.count())
    # A model mismatch is a 404 by default (a deliberate, OpenAI-conforming
    # design; see _check_model_openai/_check_model_anthropic). Clients
    # sometimes send a different, smaller model name for background work (e.g.
    # Claude Code generating a conversation title), and when that gets rejected
    # too, this set lets you allow it explicitly by passing --model-alias at
    # startup (repeatable). The default is empty = nothing changes from the
    # existing behavior.
    model_aliases: frozenset[str] = frozenset()
    # --api-key (repeatable). An empty set (the default) = no authentication,
    # keeping the previous local-only behavior. _gate_requests consults it and
    # lets a request through if either Authorization: Bearer or x-api-key
    # matches (compared with secrets.compare_digest).
    api_keys: frozenset[str] = frozenset()
    # Upper limit on the serialization lock's wait queue (--max-queue, default
    # 8). A request arriving when queue_depth has reached this is refused with a
    # 503 (carrying Retry-After)
    # (see _try_reserve_queue_slot/_release_queue_slot).
    max_queue: int = 8
    queue_depth: int = 0
    version: str = ""
    # When STATE.runner is a speculative one (SpecRunner/FlashSpecRunner), only
    # requests that ask for non-identity sampling parameters or logprobs are
    # downgraded on the spot to this runner (the non-speculative
    # FallbackRunner). Instead of rejecting them with a 400, they are served
    # while the speculative decoder's distribution guarantee is upheld by "not
    # using speculation" (see _resolve_runner_for_request). When STATE.runner is
    # already the fallback (build_runner fell back automatically at startup), no
    # downgrade is needed, so this stays None.
    downgrade_runner: "Runner | None" = None
    # LRU of responses stored by /v1/responses with store:true (item 15). The
    # key is the response id, the value a _ResponseRecord. Nothing is persisted
    # — everything is lost on process restart (this server was never designed
    # to persist responses; see the docstring at the top of server.py). The
    # limit is --max-stored-responses.
    response_store: "OrderedDict[str, object]" = field(default_factory=OrderedDict)
    max_stored_responses: int = 50
    # --max-batch (item 11 / BACKLOG.md §2). None unless continuous batching
    # was actually built for this server (max_batch>1 AND the resolved
    # runner is a plain FallbackRunner — see
    # mlxturbo.runner.maybe_build_batch_coordinator). Every one of the 8
    # generation call sites checks this (via _resolve_batch_tier) before
    # deciding whether to route around STATE.lock into the coordinator;
    # None here always means "route through STATE.lock exactly as before",
    # so the default (--max-batch omitted, or spec/flash_spec active) is
    # provably unchanged.
    batch_coordinator: "Any" = None
    max_batch: int = 1


STATE: ModelState | None = None

_THINKING_SIGNATURE_PREFIX = "mlxturbo_v1:"
_THINKING_SIGNATURE_KEY = os.urandom(32)
_THINKING_SIGNATURE_KEY_ID = hashlib.sha256(_THINKING_SIGNATURE_KEY).hexdigest()[:16]


def _thinking_signature(text: str) -> str:
    """Process-local integrity token for Anthropic thinking block round-trips."""

    digest = hmac.new(
        _THINKING_SIGNATURE_KEY, text.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{_THINKING_SIGNATURE_PREFIX}{_THINKING_SIGNATURE_KEY_ID}:{digest}"


def _validate_local_thinking_signature(text: str, signature) -> bool:
    """Validate signatures issued here; leave opaque foreign signatures alone."""

    if not isinstance(signature, str) or not signature.startswith(_THINKING_SIGNATURE_PREFIX):
        return True
    try:
        key_id, _ = signature[len(_THINKING_SIGNATURE_PREFIX) :].split(":", 1)
    except ValueError:
        return False
    # A conversation resumed after a server restart carries a signature from
    # a no-longer-available process key.  Treat it like another provider's
    # opaque signature instead of breaking an otherwise valid history.
    if key_id != _THINKING_SIGNATURE_KEY_ID:
        return True
    return hmac.compare_digest(signature, _thinking_signature(text))


def _try_trim_session_cache(sess, n_trim: int) -> bool:
    """Check whether ``sess``'s (ChatSession/FallbackSession) KV cache can be
    rewound by ``n_trim`` tokens, and actually rewind it if it can.

    This function itself knows nothing about the individual cache
    implementations (mlx_lm.models.cache's ``KVCache`` can be rewound; the
    ``ArraysCache`` used for the linear layers of a GDN hybrid cannot, because
    there is no way to return its recurrent state to an intermediate position —
    see the ChatSession docstring in mlxturbo/spec.py and the FallbackSession
    docstring in mlxturbo/runner.py). Both the decision and the execution are
    delegated straight to mlx_lm.models.cache's
    ``can_trim_prompt_cache``/``trim_prompt_cache`` — it trims for real only
    when every constituent layer declares ``is_trimmable()``, and if even one
    layer cannot, it changes nothing and returns False (leaving behind no
    half-rewritten, broken state).

    Sessions with ``mtp_valid`` set (ChatSession only) are excluded: the
    ``h_last``/``mtp_cache`` used for the MTP chain hold state for only the one
    final position of the session, so even after rewinding the KV to an
    intermediate position there is no corresponding h_last, and MTP
    continuation at the rewound position cannot be guaranteed. In
    configurations where the caller may hand us a session with ``mtp_valid``
    set (GDN hybrid + MTP, which is exactly this server's current
    configuration), an ``ArraysCache`` is mixed in to begin with, so
    ``can_trim_prompt_cache`` should return False and reject it naturally;
    rather than rely on that, we reject it explicitly here as well (belt and
    braces).
    """

    if getattr(sess, "mtp_valid", False):
        return False
    cache_list = getattr(sess, "caches", None)
    if cache_list is None:
        cache_list = getattr(sess, "cache", None)
    # The real ChatSession.caches/FallbackSession.cache is always a list of
    # mlx_lm.models.cache._BaseCache instances, but this function does not know
    # which kind of runner it is dealing with (lightweight fakes in the tests
    # sometimes make cache a contentless sentinel object), so if it is not in
    # that shape we err on the safe side and treat it as un-trimmable.
    if not isinstance(cache_list, list) or not cache_list:
        return False
    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

    if not can_trim_prompt_cache(cache_list):
        return False
    trimmed = trim_prompt_cache(cache_list, n_trim)
    return trimmed == n_trim


def _try_checkpoint_restore_session_cache(sess, lcp: int) -> int | None:
    """Alternative path for configurations where ``_try_trim_session_cache``
    always comes up empty (a GDN hybrid with ArraysCache mixed in — exactly
    this server's production configuration).

    Among the checkpoints ``sess`` has left behind (``sess.checkpoints`` —
    snapshots of only the un-rewindable layers, one per prefill chunk boundary)
    partway through ``mlxturbo.spec._prefill_hidden`` (the SpecEngine path) or
    ``mlxturbo.spec_flash.FlashSpecEngine.generate_stream`` (the Flash-Next
    path), pick the closest position that is at most ``lcp`` and restore to it:
    layers that can be trimmed (the attention KVCache) are ``.trim()``-ed down
    to that checkpoint position, and layers that cannot (ArraysCache) get the
    snapshot state written straight back. (``sess`` may be either a ChatSession
    or a FallbackSession — this function only duck-types
    ``.checkpoints``/``.caches`` or ``.cache``/``.processed`` and does not know
    the kind of runner.) The gap between the checkpoint position and ``lcp`` is
    filled in naturally by forward computation on the ordinary (chunked)
    prefill path of the subsequent generate(), because this function's caller
    truncates ``sess.processed`` to the checkpoint position — here we only
    rewind the position and perform no forward at all.

    When the trimmable layer is Flash-Next's ``_AttnCache`` (derived from
    KVCache plus an ``.indexer``; see the "KV and indexer" table at the top of
    mlxturbo/spec_flash.py), ``.trim()`` itself only rewinds the KV offset and
    does not touch the indexer's raw keys (``.indexer.keys``, which QSAIndexer
    merely appends to with one ``.update()`` per forward, and which has neither
    trim nor advance). However, the indexer is always updated by the same
    length in the same forward call as the KV (qwen4_exp's
    ``Attention.__call__`` — for a single sequence the offset and the indexer
    length always agree), so simply re-truncating the indexer keys to the same
    length as that layer's ``.offset`` after the trim brings KV and indexer
    back in sync (the same operation ``rollback()`` in
    mlxturbo/spec_flash.py performs for one round of verification forward,
    done here at checkpoint granularity). In configurations without an
    ``.indexer`` (the 27B/qwen3_5 that ChatSession uses in production) this is
    simply a no-op.

    Unlike ``_try_trim_session_cache``, this does not reject on
    ``sess.mtp_valid``. In this server's production use, where MTP is always
    on, a published session is almost always ``mtp_valid=True`` right after
    generation (see ChatSession.publish in mlxturbo/spec.py), so rejecting
    there would make this path permanently useless in production. Instead,
    because the h_last/mtp_cache used for the MTP chain hold no state
    corresponding to the restored position (the same reason as in
    ``_try_trim_session_cache``), this function never touches them at all, and
    on success the caller clears ``sess.mtp_valid`` — the subsequent generate()
    sees that and rebuilds the MTP chain from scratch (the KV/GDN reuse itself
    is not lost; see the generate() docstring in mlxturbo/spec.py).

    Before restoring, it first confirms that ``n_trim <= offset`` holds for
    every trimmable layer (= the trim is certain to succeed by exactly the
    requested amount), and only then begins making actual changes. If even one
    layer fails the requirement it changes nothing and returns None — leaving
    behind no slot in a half trimmed/restored state.

    On success it returns the restored position (the position of the chosen
    checkpoint; note that this is not ``lcp`` itself — the caller must truncate
    ``sess.processed`` to this return value), and None when there is no usable
    checkpoint.
    """

    checkpoints = getattr(sess, "checkpoints", None)
    if not checkpoints:
        return None
    # ChatSession uses the plural .caches, FallbackSession (which
    # FlashSpecRunner uses) the singular .cache — the same fallback as in
    # _try_trim_session_cache.
    cache_list = getattr(sess, "caches", None)
    if cache_list is None:
        cache_list = getattr(sess, "cache", None)
    if not isinstance(cache_list, list) or not cache_list:
        return None
    processed_len = len(sess.processed)
    if not (0 <= lcp < processed_len):
        return None

    cp_pos = None
    cp_snapshot = None
    for pos, snapshot in checkpoints:
        if pos <= lcp and (cp_pos is None or pos > cp_pos):
            cp_pos = pos
            cp_snapshot = snapshot
    if not cp_pos:  # None or 0 (0 means "no progress at all" = nothing to gain)
        return None

    trimmable_idx = [i for i, c in enumerate(cache_list) if c.is_trimmable()]
    non_trimmable_idx = {i for i in range(len(cache_list)) if i not in trimmable_idx}
    snapshot_idx = {i for i, *_ in cp_snapshot}
    # If the set of layers the snapshot covers disagrees with how the current
    # caches divide into trimmable and non-trimmable (the configuration should
    # never change, but still), err on the safe side and do not restore.
    if snapshot_idx != non_trimmable_idx:
        return None

    n_trim = processed_len - cp_pos
    for i in trimmable_idx:
        offset = getattr(cache_list[i], "offset", None)
        if offset is None or offset < n_trim:
            return None

    # From here on, the trim is confirmed to succeed as requested on every layer.
    for i in trimmable_idx:
        c = cache_list[i]
        trimmed = c.trim(n_trim)
        if trimmed != n_trim:
            # This contradicts the earlier offset check = an unexpected cache
            # implementation. This layer has already been trimmed, but
            # processed/checkpoints have not been rewritten yet, so we report a
            # miss to the caller (this particular slot may no longer be safe to
            # reuse, but that is not outside what the caller absorbs by falling
            # back to a new slot — and given the pre-checks up to this point it
            # should never actually be hit).
            return None
        # Flash-Next's _AttnCache carries an .indexer (raw keys, with no trim).
        # KV and indexer always advance by the same length, so re-truncating the
        # indexer keys to match the post-trim offset brings them back in sync
        # (see this function's docstring). In configurations without .indexer this
        # is a no-op.
        indexer = getattr(c, "indexer", None)
        if indexer is not None and getattr(indexer, "keys", None) is not None:
            indexer.keys = indexer.keys[:, : c.offset]

    restore_untrimmable_caches(cache_list, cp_snapshot)
    return cp_pos


def _select_session(prompt_ids: list[int]):
    """Pick the session (ChatSession/FallbackSession) to use for a new prompt
    out of ``STATE.session_pool``.

    The existing safe path is tried first, with top priority: if there is a
    slot whose entire already-processed token sequence is a prefix of the new
    prompt (= a pure append), it is used as-is without touching the cache at
    all. Rather than guessing the identity of the conversation from the content
    of the messages (hashing the system prompt, etc.), this merely widens to
    the whole pool the LCP test that SpecEngine's ChatSession already performs
    for a single session (mlxturbo/spec.py), so even when a dynamically
    changing system prompt (an embedded current timestamp, say) is involved, it
    degrades naturally to "if it does not match, go to the next candidate".
    Where several slots could match, the longest match wins.

    If there is no whole-sequence match, reuse is attempted in two stages over
    the partial matches (only part of the processed sequence is a prefix of the
    new prompt), in order of decreasing LCP (the same idea as llama.cpp: reuse
    up to the longest common prefix, then roll back):

    1. Can the KV cache actually be rewound to that LCP
       (``_try_trim_session_cache``)? This succeeds only when every constituent
       layer is trimmable, and on success ``processed`` can be truncated to
       exactly the LCP length (the cheapest path, requiring no differential
       prefill at all). In configurations with GDN hybrid layers (ArraysCache)
       mixed in, it always comes up empty (see that docstring above).
    2. If 1 comes up empty, try ``_try_checkpoint_restore_session_cache`` to
       roll back to the most recent checkpoint at or below the LCP. Layers that
       can be trimmed are trimmed to that checkpoint position, and layers that
       cannot (ArraysCache) get the snapshot written back — the idea being that
       what cannot be rewound is instead held as a snapshot (see
       CHECKPOINT_RETENTION in mlxturbo/spec.py). Since we can only go back as
       far as the checkpoint position, ``processed`` is truncated to that
       position (not to the LCP itself), and the gap up to the LCP is filled in
       by forward computation in the ordinary chunked prefill of the subsequent
       generate(). Because the h_last/mtp_cache used for the MTP chain hold no
       state corresponding to the restored position, they are always discarded
       on success (and ``mtp_valid`` is cleared too) — the KV/GDN reuse is not
       lost, but the MTP chain is rebuilt from scratch on the next turn.

    If both come up empty (and there is no usable checkpoint either), this
    candidate is abandoned in favour of the next LCP candidate, and when those
    run out we fall back to a new slot as before. There is no way to grab a
    slot in a wrong/half-finished state and break.

    The LCP itself is normally capped one token short of the new prompt (see
    ``reuse_cap`` below), but is widened to the full prompt length for a
    session whose ``.tail`` was stamped at exactly that position — letting
    generate()/generate_stream() resume decoding from the saved state there
    instead of prefilling. See ``_reuse_cap_for``.

    If there is no candidate at all, a new slot is allocated when the pool is
    below its limit, and when it is at the limit the least recently used slot
    (the head = LRU) is evicted first and then a new slot allocated.

    The caller must invoke this while holding ``STATE.lock`` — neither this
    function nor the session.publish/invalidate that follows performs any
    mutual exclusion on reads and writes of the pool itself (the assumption is
    that the serialization lock protects it). When the lock is removed (the
    next scheduler pass), separate mutual exclusion will be needed here too.
    """

    pool = STATE.session_pool
    # The generate paths need at least one token left to prefill: with a
    # zero-length delta, generate_stream never enters its chunk loop and the
    # hyper state it reads right after stays unset. So reuse is capped one
    # token short of the prompt, the same thing llama.cpp does. Without the cap
    # the best case of all -- the new prompt being a prefix of what this slot
    # already processed (the same prompt sent again, a regenerate, a
    # conversation stepped back a turn) -- was excluded by both passes below
    # and fell through to a fresh slot and a full prefill.
    #
    # Widened to len(prompt_ids) itself (2026-08-30) for a session that holds
    # a ``.tail`` -- a (position, state) pair a previous generate()/
    # generate_stream() call saved at *exactly* the position where its own
    # prefill ended (ChatSession.tail in mlxturbo/spec.py,
    # FallbackSession.tail in mlxturbo/runner.py) -- stamped at this same
    # len(prompt_ids). Only then can generate()/generate_stream() resume
    # decoding straight from that saved state instead of entering their
    # prefill loop with nothing left to feed it (the very crash this cap
    # exists to avoid). Every other session -- no tail, or one stamped at a
    # different position -- keeps the one-token-short cap unchanged, so this
    # is additive: it only ever widens what a specific, already-resumable
    # session can reach.
    reuse_cap = len(prompt_ids) - 1

    def _reuse_cap_for(sess) -> int:
        tail = getattr(sess, "tail", None)
        if tail is not None and tail[0] == len(prompt_ids):
            return len(prompt_ids)
        return reuse_cap

    def _lcp(pl: list[int], sess) -> int:
        n = min(len(pl), len(prompt_ids))
        i = 0
        while i < n and pl[i] == prompt_ids[i]:
            i += 1
        return min(i, _reuse_cap_for(sess))

    # 1st pass: whole-sequence match (a pure append) — the existing safe path,
    # which does not touch the cache.
    best_key = None
    best_lcp = -1
    for key, sess in pool.items():
        pl = sess.processed
        if not pl:
            continue
        lcp = _lcp(pl, sess)
        if lcp == len(pl) and lcp > best_lcp:
            best_lcp = lcp
            best_key = key

    if best_key is not None:
        pool.move_to_end(best_key)
        return pool[best_key]

    # 2nd pass: partial matches. Try candidates in order of decreasing LCP
    # until a slot that can actually be rewound is found.
    partial = []
    for key, sess in pool.items():
        pl = sess.processed
        if not pl:
            continue
        lcp = _lcp(pl, sess)
        if 0 < lcp < len(pl):
            partial.append((lcp, key, sess))
    partial.sort(key=lambda c: -c[0])

    for lcp, key, sess in partial:
        if _try_trim_session_cache(sess, len(sess.processed) - lcp):
            sess.processed = sess.processed[:lcp]
            if hasattr(sess, "mtp_cache"):
                sess.mtp_cache = None
            if hasattr(sess, "h_last"):
                sess.h_last = None
            if hasattr(sess, "tail") and (sess.tail is None or sess.tail[0] != lcp):
                sess.tail = None
            pool.move_to_end(key)
            return sess

        cp_pos = _try_checkpoint_restore_session_cache(sess, lcp)
        if cp_pos is not None:
            sess.processed = sess.processed[:cp_pos]
            if hasattr(sess, "checkpoints"):
                sess.checkpoints = [c for c in sess.checkpoints if c[0] <= cp_pos]
            if hasattr(sess, "mtp_cache"):
                sess.mtp_cache = None
            if hasattr(sess, "h_last"):
                sess.h_last = None
            if hasattr(sess, "mtp_valid"):
                sess.mtp_valid = False
            if hasattr(sess, "tail") and (sess.tail is None or sess.tail[0] != cp_pos):
                sess.tail = None
            pool.move_to_end(key)
            return sess

    if len(pool) >= STATE.max_sessions:
        pool.popitem(last=False)
    key = next(STATE.session_key_seq)
    session = STATE.session_factory()
    pool[key] = session
    return session


async def _select_session_on_executor(prompt_ids: list[int]):
    """Run ``_select_session`` on ``STATE.executor`` (the same dedicated worker
    thread that loaded the model).

    The invariant stated in ``_run_generate``'s docstring and the comment just
    above it (MLX's computation graph and KV cache are bound to the thread that
    loaded them, so touching them from another thread crashes with "There is no
    Stream(gpu, N) in current thread"; confirmed by measurement) is not only
    about ``STATE.runner.generate`` — the 1st pass for a whole-sequence match
    (a pure append) merely reads ``sess.processed`` (a Python list of ints) and
    is therefore harmless, but as of 2026-08-29 the 2nd pass that
    ``_try_trim_session_cache``/``_try_checkpoint_restore_session_cache``
    actually touch (``cache.trim()``, slicing ``indexer.keys``, the
    ``c.state = state`` assignment in ``restore_untrimmable_caches``) consists
    of real MLX array operations in the same sense, and running this function
    directly on the caller's event-loop thread crashes with the same error
    (measured: it reproduced on the 2nd turn with thinking enabled, when the
    history diverged partway through the processed sequence and the 2nd pass's
    checkpoint restore actually fired). Paths that only ever go through the 1st
    pass (most of this server's existing tests) never touched an MLX array in
    the first place, so the symptom did not appear and this was overlooked.

    Call this while still holding ``STATE.lock`` (the same premise as
    ``_select_session``'s own docstring) — it is not re-acquired here.
    """

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(STATE.executor, _select_session, prompt_ids)


def _try_reserve_queue_slot() -> bool:
    """Called by the 4 paths that use ``STATE.lock`` (= that generate) before
    they actually queue up. ``STATE.queue_depth`` is the sum of "waiting for
    the lock + currently being processed" (since the serialization lock has a
    capacity of 1, this single number is enough to express the queue depth).
    If the limit (``--max-queue``, default 8) has been reached, no slot is
    reserved and False is returned (the caller responds with 503).

    The matching ``_release_queue_slot`` is called in the endpoint function's
    own finally for non-streaming, and in the finally of the generating side
    (``_openai_stream`` and friends) for streaming — streaming reserves the
    slot before the StreamingResponse is assembled (immediately after this
    function), but the release can only be detected on the generator side,
    where generation actually finishes or the connection is dropped.
    """

    if STATE.queue_depth >= STATE.max_queue:
        return False
    STATE.queue_depth += 1
    return True


def _release_queue_slot() -> None:
    STATE.queue_depth = max(0, STATE.queue_depth - 1)


async def _queue_owned_stream(stream):
    """Make one stream the sole owner of one already-reserved queue slot.

    The wrapper enters its cleanup region before requesting the inner stream's
    first item.  Consequently a disconnect immediately after a protocol
    preamble still releases the reservation; inner generators never release
    queue slots themselves.
    """

    try:
        async for item in stream:
            yield item
    finally:
        try:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
        finally:
            _release_queue_slot()


_SSE_KEEPALIVE_INTERVAL = 15.0
_SSE_KEEPALIVE_LINE = ": keepalive\n\n"


async def _await_with_keepalive(coro, interval: float = _SSE_KEEPALIVE_INTERVAL):
    """An async generator that, while waiting for ``coro`` to complete, yields
    an SSE keepalive comment line (``("keepalive", line)``) every ``interval``
    seconds, and once it completes yields ``("result", return value)`` last and
    finishes.

    This stops real clients from timing out and dropping the connection while
    waiting for the first token in a prefill-dominated workload (measured:
    about 3 minutes for Claude Code's 97k tokens; see the docstring at the top
    of server.py). SSE comment lines (lines beginning with ``:``) are ignored
    by clients per the specification, so it is safe to mix them in anywhere
    during streaming.

    If the caller (the generator returned by StreamingResponse) is
    ``aclose()``-d (client disconnect), ``GeneratorExit`` arrives here. Along
    with ``CancelledError`` it is caught as ``BaseException`` so that ``task``
    is reliably cancelled before re-raising — swallowing it and moving on would
    leave ``task`` (waiting on a lock acquisition or on ``queue.Queue.get``)
    orphaned.
    """

    task = asyncio.ensure_future(coro)
    try:
        while True:
            done, _pending = await asyncio.wait({task}, timeout=interval)
            if task in done:
                break
            yield ("keepalive", _SSE_KEEPALIVE_LINE)
    except BaseException:
        if not task.done():
            task.cancel()
        raise
    yield ("result", task.result())


async def _acquire_lock_with_keepalive(
    lock: asyncio.Lock,
    owned: list[bool],
    interval: float = _SSE_KEEPALIVE_INTERVAL,
):
    """Acquire ``lock`` while yielding keepalives, with explicit ownership.

    ``asyncio.Lock.acquire()`` can finish while the caller is suspended at a
    yielded keepalive.  If the stream is closed in that window, merely
    cancelling a still-pending task is insufficient: the completed task has
    already taken the lock.  This helper records ownership before handing
    control back after a successful acquire, and releases a completed-but-not-
    transferred acquire when the async generator is closed.
    """

    task = asyncio.ensure_future(lock.acquire())
    try:
        while True:
            done, _pending = await asyncio.wait({task}, timeout=interval)
            if task in done:
                break
            yield _SSE_KEEPALIVE_LINE
        task.result()
        # No await/yield occurs between this assignment and generator return,
        # so the caller's outer finally can now be the sole release owner.
        owned[0] = True
    except BaseException:
        if not task.done():
            task.cancel()
        elif not task.cancelled() and task.exception() is None and task.result():
            lock.release()
        raise


def _requeue_front(q: "queue.Queue", item) -> None:
    """A small trick that puts back at the head of ``q``, order preserved, the
    first item that was peeked at with ``q.get()`` in order to drive
    keepalives. It lets the next ``q.get()`` fetch the same element again
    without changing the body of the existing
    ``while True: kind, payload = await asyncio.to_thread(q.get)`` loop at all.
    The ``worker()`` thread only ever touches the queue through ``q.put``, so
    taking ``q.mutex`` for this consumer-side-only operation avoids any
    race."""

    with q.mutex:
        q.queue.appendleft(item)


# ---------- Normalization and validation of input ----------


class ContentNormalizationError(ValueError):
    """Base class for client-side input defects found while normalizing
    content. A marker that makes them come back as 400 (in each protocol's
    error format) instead of 5xx."""


class MultimodalContentError(ContentNormalizationError):
    """The content contained a non-text block (image_url/image/input_audio and
    the like). mlxturbo is text-only (convert_flash.py drops
    vision_tower.*/model.visual.* during conversion), so silently skipping such
    a block would leave the client believing the image was read and continuing
    the conversation on that basis. Say so explicitly with a 400.
    """

    def __init__(self, block_type):
        self.block_type = block_type
        super().__init__(
            f"this model is text-only; unsupported content block type: {block_type!r}"
        )


class InvalidContentError(ContentNormalizationError):
    """The shape of the content itself is broken (a missing key, a type
    mismatch, and so on). Silently coercing it to an empty string or to str()
    would let the broken input through as an empty prompt, so it is rejected
    here with a 400."""


def _content_to_text(content) -> str:
    """Reduce either the OpenAI or the Anthropic content format (a string, or a
    list of blocks) to plain text, because tokenizer.apply_chat_template only
    expects string content.

    A block list consisting solely of ``type: "text"`` blocks is joined into
    the same result as a single string. If any other type (image_url/image/
    input_audio and the like) is found, ``MultimodalContentError`` is raised.
    Any other broken shape (content missing/null, a block without a type, a
    text block whose text is not a string, content that is neither a string nor
    a list) raises ``InvalidContentError`` rather than being silently coerced
    to "" or str(...): letting broken input through as an empty prompt does not
    produce a 500, but it does produce behavior whose cause cannot be worked
    out.
    """

    if content is None:
        raise InvalidContentError("'content' is required and must not be null")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                raise InvalidContentError(
                    f"each content block must be an object, got {type(block).__name__}"
                )
            if "type" not in block:
                raise InvalidContentError("each content block must have a 'type'")
            btype = block["type"]
            if btype == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise InvalidContentError(
                        "content block of type 'text' must have a string 'text' field"
                    )
                parts.append(text)
                continue
            raise MultimodalContentError(btype)
        return "".join(parts)
    raise InvalidContentError(
        f"'content' must be a string or a list of content blocks, got {type(content).__name__}"
    )


def _validate_messages_shape(messages) -> str | None:
    """If messages is a list of strings or numbers, the subsequent m.get(...)
    raises AttributeError and turns into a 500. Check just the shape up front
    and reject with a 400."""

    if not isinstance(messages, list):
        return "'messages' must be a list"
    for m in messages:
        if not isinstance(m, dict):
            return "each item in 'messages' must be an object with 'role' and 'content'"
    return None


# ---------- tool calling: normalizing history (tool_calls / tool_result) ----------
#
# Whichever protocol they came from, the messages handed to
# apply_chat_template are unified into the OpenAI format
# (assistant: {"role","content","tool_calls":[{"id","type":
# "function","function":{"name","arguments": <dict>}}]} / tool result:
# {"role":"tool","tool_call_id","content"}), because the chat template itself
# assumes the OpenAI/Qwen-family tool calling convention (just as mlx_lm's
# automatic tool_parser selection does). arguments is passed through as a dict
# (mlx_lm.server.process_message_content likewise runs json.loads before
# handing it to the template — templates usually re-serialize it with tojson,
# so the string must not be encoded twice).


def _normalize_openai_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        tool_calls = m.get("tool_calls")
        if role == "assistant" and tool_calls:
            raw_content = m.get("content")
            text = "" if raw_content is None else _content_to_text(raw_content)
            converted_calls = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    raise InvalidContentError("each item in 'tool_calls' must be an object")
                func = tc.get("function")
                if not isinstance(func, dict) or not isinstance(func.get("name"), str):
                    raise InvalidContentError(
                        "'tool_calls[].function' must be an object with a string 'name'"
                    )
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        args_obj = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError as exc:
                        raise InvalidContentError(
                            f"'tool_calls[].function.arguments' is not valid JSON: {exc}"
                        ) from exc
                elif isinstance(raw_args, dict):
                    args_obj = raw_args
                else:
                    raise InvalidContentError(
                        "'tool_calls[].function.arguments' must be a JSON string or an object"
                    )
                converted_calls.append(
                    {
                        "id": tc.get("id") or f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {"name": func["name"], "arguments": args_obj},
                    }
                )
            out.append({"role": "assistant", "content": text, "tool_calls": converted_calls})
            continue
        if role == "tool":
            template_msg = {"role": "tool", "content": _content_to_text(m.get("content"))}
            tcid = m.get("tool_call_id")
            if tcid is not None:
                template_msg["tool_call_id"] = tcid
            name = m.get("name")
            if name is not None:
                template_msg["name"] = name
            out.append(template_msg)
            continue
        out.append({"role": role, "content": _content_to_text(m.get("content"))})
    return out


def _normalize_anthropic_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str) or content is None:
            out.append({"role": role, "content": _content_to_text(content)})
            continue
        if not isinstance(content, list):
            raise InvalidContentError(
                f"'content' must be a string or a list of content blocks, got {type(content).__name__}"
            )
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_result_msgs: list[dict] = []
        for block in content:
            if not isinstance(block, dict) or "type" not in block:
                raise InvalidContentError("each content block must be an object with a 'type'")
            btype = block["type"]
            if btype == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise InvalidContentError(
                        "content block of type 'text' must have a string 'text' field"
                    )
                text_parts.append(text)
            elif btype == "tool_use":
                if role != "assistant":
                    raise InvalidContentError(
                        "'tool_use' content blocks are only valid in assistant messages"
                    )
                name = block.get("name")
                if not isinstance(name, str) or not name:
                    raise InvalidContentError("'tool_use' block must have a non-empty string 'name'")
                tool_calls.append(
                    {
                        "id": block.get("id") or f"toolu_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {"name": name, "arguments": block.get("input", {}) or {}},
                    }
                )
            elif btype == "tool_result":
                if role != "user":
                    raise InvalidContentError(
                        "'tool_result' content blocks are only valid in user messages"
                    )
                tuid = block.get("tool_use_id")
                result_text = _content_to_text(block.get("content", "") or "")
                tool_result_msgs.append(
                    {"role": "tool", "content": result_text, "tool_call_id": tuid}
                )
            elif btype in ("thinking", "redacted_thinking"):
                # The Anthropic convention for extended thinking + tool use:
                # the previous turn's thinking/redacted_thinking blocks must be
                # sent back verbatim as part of the next turn's history (stated
                # explicitly in the official documentation). This server itself
                # returns such blocks when thinking is enabled (see where a
                # "thinking" block is assembled in this module), so a client
                # that sends back a multi-turn conversation using tools (e.g.
                # Claude Code) will certainly include this type in the history.
                # Verify that a thinking signature issued by mlxturbo itself
                # has not been tampered with (opaque signatures from other
                # providers cannot be interpreted, so they are accepted as-is).
                if btype == "thinking":
                    thinking = block.get("thinking")
                    if not isinstance(thinking, str):
                        raise InvalidContentError(
                            "content block of type 'thinking' must have a string 'thinking' field"
                        )
                    if not _validate_local_thinking_signature(
                        thinking, block.get("signature")
                    ):
                        raise InvalidContentError(
                            "thinking block was modified after its signature was issued"
                        )
                # There is no need to make the model read the contents again
                # (a thinking-aware chat template is expected to drop the think
                # contents of settled past turns from the history — see the
                # _apply_template docstring), so rather than returning a 400 we
                # simply skip it here.
                pass
            else:
                raise MultimodalContentError(btype)
        # tool_result (the previous turn's results) is laid out as
        # role: "tool" ahead of this message's ordinary text. Text and
        # tool_result rarely coexist within the same user message, but even
        # then the order of appearance is preserved (the tool_result group ->
        # this message's own text).
        out.extend(tool_result_msgs)
        joined_text = "".join(text_parts)
        if role == "assistant" and tool_calls:
            out.append({"role": "assistant", "content": joined_text, "tool_calls": tool_calls})
        elif joined_text or not tool_result_msgs:
            out.append({"role": role, "content": joined_text})
    return out


def _naive_prompt_ids(messages: list[dict]):
    """A naive fallback for models whose tokenizer has no chat template.

    It merely lines up "role: content" into a plain string. No Qwen-family-only
    assumption is brought in (if gemma/kimi/glm are served in the future, their
    chat template is used whenever there is one, and only otherwise do we land
    here).
    """

    lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    lines.append("assistant:")
    return STATE.tokenizer.encode("\n".join(lines))


def _apply_template(
    messages: list[dict], enable_thinking: bool | None = None, tools: list | None = None
):
    """``enable_thinking`` is the value already resolved from the standard
    fields (reasoning_effort/thinking). When omitted (None) the template's own
    default is left in charge — if enable_thinking is not passed, mlx_lm's
    TokenizerWrapper.apply_chat_template fills in ``self.has_thinking`` itself
    (thinking-capable models default to on).

    Some thinking-enabled templates drop <think>...</think> when folding a
    settled past turn's assistant message into the history (because there is no
    need to make the model read it again). The next turn's prompt is therefore
    shorter than the token sequence that was actually generated, and
    ChatSession's LCP reuse (which requires the previously processed sequence
    to be a prefix of the new prompt as-is) can no longer hold even in
    principle. Passing enable_thinking=False (reasoning_effort: "none", etc.)
    brings this kind of model back into scope for reuse.

    ``tools`` is the OpenAI-format tools array after tool_choice has been
    resolved (None means no tools are shown to the model this turn). The
    premise is that the caller has already checked
    ``tokenizer.has_tool_calling`` via ``_check_tool_calling_support`` — here
    the tools that were passed in are simply forwarded to
    ``apply_chat_template(tools=...)`` (HF's apply_chat_template accepts tools
    as a first-class kwarg, so it will not be silently ignored — if it were
    ignored, that would be a model where has_tool_calling is not being picked
    up, and making that determination is the caller's responsibility).
    """

    kwargs = {"add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    if tools is not None:
        kwargs["tools"] = tools
    try:
        return STATE.tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return STATE.tokenizer.apply_chat_template(messages, **kwargs)
        except ValueError:
            return _naive_prompt_ids(messages)
    except ValueError:
        # A model with no chat template (chat_template is unset).
        return _naive_prompt_ids(messages)


# Rough mapping from reasoning_effort to a thinking token budget. mlxturbo
# cannot adjust the depth of thinking directly, so the value is mapped to a
# budget (the upper limit on the number of generated tokens ThinkingRouter
# counts) to make it actually take effect. An unknown value is treated as
# medium rather than rejected with a 400 (so that adding effort values in the
# future does not break anything).
_REASONING_EFFORT_BUDGET = {
    "minimal": 512,
    "low": 2048,
    "medium": 8192,
    "high": 32768,
    "xhigh": 65536,
    "max": 131072,
}
_REASONING_EFFORT_DEFAULT_BUDGET = _REASONING_EFFORT_BUDGET["medium"]


def _resolve_thinking(body: dict, protocol: str) -> tuple[bool | None, int | None, str | None]:
    """Read only the standard fields and return (kwarg for enable_thinking,
    budget, error). There are no mlxturbo-specific fields — if nothing is
    specified, this returns (None, None, None) (leave it to the template, no
    budget enforced).

    Budget: None = not enforced (wait for it to end naturally), 0 = thinking
    completely off, a positive integer = ThinkingRouter cuts it off after that
    many tokens. The caller must clamp it against max_tokens (which is not
    known here).
    """

    if protocol == "openai":
        if "reasoning_effort" not in body:
            return None, None, None
        value = body["reasoning_effort"]
        key = value.lower() if isinstance(value, str) else None
        if key == "none":
            return False, 0, None
        budget = _REASONING_EFFORT_BUDGET.get(key, _REASONING_EFFORT_DEFAULT_BUDGET)
        return True, budget, None

    # anthropic
    if "thinking" not in body:
        return None, None, None
    value = body["thinking"]
    if not isinstance(value, dict):
        return None, None, "'thinking' must be an object with a 'type' field"
    t = value.get("type")
    if t == "disabled":
        return False, 0, None
    if t == "enabled":
        budget = value.get("budget_tokens")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
            return None, None, "'thinking.budget_tokens' must be a non-negative integer"
        return True, budget, None
    if t == "adaptive":
        output_config = body.get("output_config", {})
        if not isinstance(output_config, dict):
            return None, None, "'output_config' must be an object"
        effort = output_config.get("effort", "high")
        if effort not in {"low", "medium", "high", "xhigh", "max"}:
            return None, None, (
                "'output_config.effort' must be 'low', 'medium', 'high', "
                f"'xhigh', or 'max', got {effort!r}"
            )
        # Local Qwen templates expose a binary enable_thinking switch rather
        # than Anthropic's model-side adaptive controller.  Preserve the wire
        # contract by accepting adaptive mode and map its soft effort hint to
        # the same bounded router budget used by reasoning_effort.  The endpoint
        # clamps this to max_tokens before generation.
        return True, _REASONING_EFFORT_BUDGET[effort], None
    return None, None, (
        f"'thinking.type' must be 'enabled', 'adaptive', or 'disabled', got {t!r}"
    )


# ---------- tool calling: request side (validating/resolving tools/tool_choice) ----------
#
# The format passed to tokenizer.apply_chat_template's ``tools=`` is unified on
# the OpenAI format
# (``[{"type": "function", "function": {"name", "description", "parameters"}}]``).
# Anthropic's ``input_schema`` format is converted to the OpenAI format here
# before being passed along (the jinja template assumes Qwen-family json_tools
# anyway, so keeping the shape handed to the template down to a single variant
# is less likely to fall apart). tokenizer.tool_parser (the model-specific
# parser mlx_lm selects automatically; qwen3_coder, for example, looks up
# argument types from the tools' parameters.properties) also expects tools in
# this OpenAI format, so the same conversion result can be reused for it.


def _validate_openai_tools(tools) -> str | None:
    if not isinstance(tools, list) or not tools:
        return "'tools' must be a non-empty array"
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            return 'each item in \'tools\' must be an object with "type": "function"'
        func = t.get("function")
        if not isinstance(func, dict) or not isinstance(func.get("name"), str) or not func.get("name"):
            return "each tool's 'function' must be an object with a non-empty string 'name'"
    return None


def _validate_anthropic_tools(tools) -> str | None:
    if not isinstance(tools, list) or not tools:
        return "'tools' must be a non-empty array"
    for t in tools:
        if not isinstance(t, dict) or not isinstance(t.get("name"), str) or not t.get("name"):
            return "each item in 'tools' must be an object with a non-empty string 'name'"
        if "input_schema" in t and not isinstance(t["input_schema"], dict):
            return "'tools[].input_schema' must be an object"
    return None


def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.get("name"),
                "description": t.get("description", "") or "",
                "parameters": t.get("input_schema", {}) or {},
            },
        }
        for t in tools
    ]


def _resolve_tool_choice_openai(body: dict) -> tuple[list | None, str | None]:
    """Read ``tools``/``tool_choice`` (OpenAI format) and resolve the tools to
    hand to apply_chat_template (still in OpenAI format). If tools is
    missing/empty, return (None, None) (tool_choice is ignored — the real API
    behaves the same way).

    ``tool_choice: "none"`` is handled simply by not passing tools at all (no
    tools are shown to the model this turn; the tools that were passed in are
    themselves ignored). ``"required"`` and naming a specific function are
    rejected with a 400, because there is no way to force the model's hand (we
    do not silently treat them as auto).
    """

    tools = body.get("tools")
    if not tools:
        return None, None
    shape_err = _validate_openai_tools(tools)
    if shape_err is not None:
        return None, shape_err
    choice = body.get("tool_choice", "auto")
    if choice is None:
        choice = "auto"
    if choice == "none":
        return None, None
    if choice == "auto":
        return tools, None
    if choice == "required":
        return None, (
            "'tool_choice: \"required\"' is not supported: this server has no "
            "mechanism to force the model to call a tool (it can only detect "
            "tool calls the model chooses to emit on its own)"
        )
    if isinstance(choice, dict):
        return None, (
            "'tool_choice' selecting a specific function is not supported: this "
            "server has no mechanism to force the model to call a particular tool"
        )
    return None, f"'tool_choice' must be 'auto', 'none', 'required', or a function object; got {choice!r}"


def _resolve_tool_choice_anthropic(body: dict) -> tuple[list | None, str | None]:
    """The Anthropic-format version. The returned tools are not left in
    Anthropic format but have already been converted to the OpenAI format by
    ``_anthropic_tools_to_openai`` — the caller can pass them straight to
    either apply_chat_template or tool_parser.
    """

    tools = body.get("tools")
    if not tools:
        return None, None
    shape_err = _validate_anthropic_tools(tools)
    if shape_err is not None:
        return None, shape_err
    choice = body.get("tool_choice")
    if choice is None:
        return _anthropic_tools_to_openai(tools), None
    if not isinstance(choice, dict):
        return None, "'tool_choice' must be an object with a 'type' field"
    t = choice.get("type")
    if t == "none":
        return None, None
    if t == "auto":
        return _anthropic_tools_to_openai(tools), None
    if t in ("any", "tool"):
        return None, (
            f"'tool_choice.type': {t!r} (forced tool calling) is not supported: this "
            "server has no mechanism to force the model to call a tool"
        )
    return None, f"'tool_choice.type' must be 'auto', 'none', 'any', or 'tool'; got {t!r}"


def _check_tool_calling_support(resolved_tools) -> str | None:
    """If resolved_tools (after tool_choice resolution; None means no tools are
    shown this time) is not None, check whether the tokenizer has tool_call
    markers (``has_tool_calling``, which mlx_lm's TokenizerWrapper detects
    automatically from the chat template string — the same mechanism as
    has_thinking). If it does not, return a 400 rather than silently ignoring
    the request.
    """

    if resolved_tools is None:
        return None
    if not getattr(STATE.tokenizer, "has_tool_calling", False):
        return (
            "this model does not support tool calling: no tool-call marker is "
            "configured for its tokenizer/chat template (tokenizer.has_tool_calling "
            "is False)"
        )
    return None


def _prompt_already_thinking(prompt_ids: list[int]) -> bool:
    """Determine whether the prompt rendered by ``_apply_template`` ends with an
    unclosed thinking block (= whether the chat template itself has already
    opened ``<think>`` at the end of the generation prompt).

    A bug confirmed on real models (Qwen3.6 / Flash-Next): these templates end
    the prompt with ``<think>`` already open by the time it is handed to the
    model, as in ``<|im_start|>assistant\\n<think>\\n``. The model itself does
    not generate ``<think>`` (it will not emit an already-opened marker a
    second time). ``ThinkingRouter``, meanwhile, always starts from
    ``phase="detect"`` (waiting to see whether the beginning matches
    think_start), so it misjudges "the beginning does not match = it is not
    thinking" and pours the entire output into content; the reasoning then
    contaminates the body and only the ``</think>`` is left behind as raw
    closing text.

    The test is "there is no think_end_tokens after the last occurrence of
    think_start_tokens, and everything after that start is whitespace" = the
    end of the generation prompt is a thinking block left open. We do not look
    only for an exact match at the very end, so as not to miss templates where
    a newline token or similar follows immediately after ``<think>``.
    Conversely, between a literal ``<think>`` inside the user/history text and
    the assistant generation suffix there is always a role delimiter that is
    not whitespace, so such a case is not mistaken for the start of thinking.

    Always False when ``has_thinking`` is False (this model does not separate
    thinking at all).
    """

    tokenizer = STATE.tokenizer
    if not getattr(tokenizer, "has_thinking", False):
        return False
    start = tuple(getattr(tokenizer, "think_start_tokens", None) or ())
    if not start:
        return False
    end = tuple(getattr(tokenizer, "think_end_tokens", None) or ())

    def _rfind(seq, sub) -> int:
        n = len(sub)
        if n == 0 or n > len(seq):
            return -1
        for i in range(len(seq) - n, -1, -1):
            if tuple(seq[i : i + n]) == sub:
                return i
        return -1

    start_idx = _rfind(prompt_ids, start)
    if start_idx < 0:
        return False
    end_idx = _rfind(prompt_ids, end) if end else -1
    if end_idx >= start_idx:
        return False

    # A literal unmatched marker can occur in user/history text.  A template-
    # opened thinking block is specifically a generation suffix: only ignorable
    # whitespace may follow the last start marker.  Looking at the whole prompt
    # without this suffix check routes an ordinary answer into reasoning whenever
    # a user happens to type ``<think>``.
    tail = prompt_ids[start_idx + len(start) :]
    if tail:
        decoder = getattr(tokenizer, "decode", None)
        if not callable(decoder):
            return False
        try:
            tail_text = decoder(tail)
        except Exception:
            return False
        if not isinstance(tail_text, str) or tail_text.strip():
            return False
    return True


class ThinkingRouter:
    """Route the model's raw token stream into the two channels reasoning
    (thinking) and content. The markers are not guessed but taken from
    mlx_lm.TokenizerWrapper's public API
    (``has_thinking``/``think_start_tokens``/``think_end_tokens``, which covers
    Qwen's ``<think>``/``</think>`` and channel-style models through the same
    interface). For models where they cannot be obtained (``has_thinking`` is
    False) there is always a single content channel — meaning "thinking cannot
    be separated for this model", and the whole output goes into the body as
    before.

    To decode correctly across marker boundaries, separate streaming
    detokenizers are used for reasoning and for content (BPE's trailing-space
    merging and the like then stay contained within each channel). To avoid
    false detections spanning a marker token sequence (which can be several
    tokens long), a marker's worth of raw tokens is always buffered for
    lookahead, and only the portion known not to match is committed to one
    detokenizer or the other.

    budget (int | None): the upper limit on tokens permitted inside a thinking
    block. None is unlimited (wait for it to end naturally). 0 means "thinking
    completely off", in which case no routing happens at all.

    The limit of what can be done when the budget is exceeded: neither
    SpecEngine (spec.py) nor mlx_lm.generate.stream_generate offers a live
    interface for interrupting a generation from outside to "cut off here and
    rebuild the rest from a different prompt" (the return value of on_tokens is
    not looked at — changing the caller alone cannot achieve it without making
    that change inside the existing speculative decoding engine). So what is
    implemented here goes only as far as "tokens after the budget is reached
    are not forwarded to the client"; there is no in-place restart that
    "forcibly closes the thinking block and makes the model continue with the
    body". It can be detected via the budget_exceeded flag, which the caller
    reflects into finish_reason/stop_reason.
    (The two-stage restart itself is not technically impossible: on reaching
    the budget, give up on the current generate() call and call generate()
    again with prompt_ids + the tokens generated so far + </think> as the new
    prompt; ChatSession's LCP reuse would then avoid recomputing the prefill.
    However, the implementation cost and bug surface of making this chunk split
    consistent across both the streaming and non-streaming paths and across
    both SpecRunner and FallbackRunner were judged not to be worth the value of
    the feature, so it was deferred this time.)

    tool_calling_enabled (bool): whether this generation should also detect
    tool calls (corresponding to whether the caller passed resolved_tools =
    whether tool_choice was not "none" and tools were actually specified). It
    is unconditionally disabled if the tokenizer has no tool_call markers (the
    same pattern as ``has_thinking``; see ``self.tool_enabled``).

    When enabled, occurrences of ``tokenizer.tool_call_start_tokens`` are
    watched with lookahead during the content phase (the same rolling-window
    scheme by which the thinking phase waits for its end marker), and on a
    match it enters the "tool" phase and returns the text up to
    ``tokenizer.tool_call_end_tokens`` as the "tool" channel (the markers
    themselves are included in neither channel — the same treatment as mlx_lm's
    ToolCallFormatter/_process_control_tokens). Tool calls can occur several
    times within one generation, so content <-> tool can go back and forth any
    number of times. Entering and leaving the "tool" phase always returns
    ("tool_start", "") / ("tool_end", matched: bool) exactly once each, even
    when the text is empty, so the caller can use those as boundaries for
    grouping the individual calls (because feed() omits empty segments, without
    making the boundaries themselves independent events, "two calls that
    happened to have empty text around them" would be wrongly merged into one).
    The bool in ``tool_end`` says "did it really close with the end marker" —
    False indicates it was forcibly cut off by max_tokens or similar, and the
    caller uses it to decide not to append the end-marker string when
    reconstructing.

    ``already_thinking`` (bool): pass True when the prompt rendered by
    ``_apply_template`` already ends with an unclosed ``<think>`` (see
    ``_prompt_already_thinking``). In that case the model itself does not
    generate the think_start marker (the template has already emitted it), so
    starting from ``phase="detect"`` and waiting for the beginning to match
    think_start would misjudge it as "not thinking". When True, ``phase``
    starts directly at ``"thinking"`` (skipping detect) — budget accounting,
    the streaming and non-streaming paths, and tool calling all branch solely
    on the value of ``phase``, so changing this initial value alone takes
    effect consistently on every path.
    """

    def __init__(
        self,
        tokenizer,
        budget: int | None,
        eos_ids: set,
        tool_calling_enabled: bool = False,
        already_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.budget = budget
        self.eos_ids = eos_ids
        self.enabled = bool(getattr(tokenizer, "has_thinking", False)) and budget != 0
        self.tool_enabled = tool_calling_enabled and bool(
            getattr(tokenizer, "has_tool_calling", False)
        )
        if not self.enabled:
            self.phase = "content"
        elif already_thinking:
            # The template ended the prompt with <think> already open. The
            # model will not re-emit think_start, so skip detect and start
            # directly in the thinking phase.
            self.phase = "thinking"
        else:
            self.phase = "detect"
        self.buf: list[int] = []
        self.tool_buf: list[int] = []
        self.think_detok = tokenizer.detokenizer
        self.content_detok = tokenizer.detokenizer
        self.tool_detok = tokenizer.detokenizer
        self.thinking_token_count = 0
        self.budget_exceeded = False
        # Qwen-family models generate ``\n\n`` right after the thinking end
        # marker as a separator from the answer. Do not mix this framing into
        # the visible body. Leading newlines in a response that never went
        # through thinking are model output proper, though, so leave them
        # alone.
        self._strip_post_thinking_newlines = False
        self._start_tokens = list(tokenizer.think_start_tokens) if self.enabled else []
        self._end_tokens = list(tokenizer.think_end_tokens) if self.enabled else []
        self._tool_start = (
            list(tokenizer.tool_call_start_tokens) if self.tool_enabled else []
        )
        self._tool_end = (
            list(tokenizer.tool_call_end_tokens) if self.tool_enabled else []
        )

    def _clean_content_segment(self, segment: str) -> str:
        """Drop only leading CR/LF framing immediately after a thinking block."""

        if not segment or not self._strip_post_thinking_newlines:
            return segment
        cleaned = segment.lstrip("\r\n")
        if cleaned:
            self._strip_post_thinking_newlines = False
        return cleaned

    def _feed_thinking_token(self, token: int) -> bool:
        """Decode one allowed thinking token; return False at the budget boundary."""

        if self.budget is not None and self.thinking_token_count >= self.budget:
            self.budget_exceeded = True
            return False
        self.think_detok.add_token(token)
        self.thinking_token_count += 1
        return True

    def _feed_content_token(self, t: int, out: list) -> None:
        """Process one token of the content phase. If tool routing is disabled,
        just decode it immediately (as before). If it is enabled, keep a
        lookahead buffer for ``tool_call_start`` (a rolling window of the same
        form by which the "thinking" phase waits for its end marker)."""

        if not self.tool_enabled:
            self.content_detok.add_token(t)
            seg = self._clean_content_segment(self.content_detok.last_segment)
            if seg:
                out.append(("content", seg))
            return
        self.tool_buf.append(t)
        if len(self.tool_buf) <= len(self._tool_start):
            if self.tool_buf == self._tool_start:
                self.phase = "tool"
                self.tool_buf = []
                self._strip_post_thinking_newlines = False
                out.append(("tool_start", ""))
            return
        cut = len(self.tool_buf) - len(self._tool_start)
        flush, self.tool_buf = self.tool_buf[:cut], self.tool_buf[cut:]
        for ft in flush:
            self.content_detok.add_token(ft)
        seg = self._clean_content_segment(self.content_detok.last_segment)
        if seg:
            out.append(("content", seg))
        if self.tool_buf == self._tool_start:
            self.phase = "tool"
            self.tool_buf = []
            self._strip_post_thinking_newlines = False
            out.append(("tool_start", ""))

    def _feed_tool_token(self, t: int, out: list) -> None:
        self.tool_buf.append(t)
        if len(self.tool_buf) <= len(self._tool_end):
            if self.tool_buf == self._tool_end:
                self.phase = "content"
                self.tool_buf = []
                out.append(("tool_end", True))
            return
        cut = len(self.tool_buf) - len(self._tool_end)
        flush, self.tool_buf = self.tool_buf[:cut], self.tool_buf[cut:]
        for ft in flush:
            self.tool_detok.add_token(ft)
        seg = self.tool_detok.last_segment
        if seg:
            out.append(("tool", seg))
        if self.tool_buf == self._tool_end:
            self.phase = "content"
            self.tool_buf = []
            out.append(("tool_end", True))

    def feed(self, toks) -> list[tuple[str, object]]:
        out: list[tuple[str, object]] = []
        for t in toks:
            if t in self.eos_ids:
                continue
            if self.budget_exceeded:
                continue
            if self.phase == "detect":
                self.buf.append(t)
                if len(self.buf) < len(self._start_tokens):
                    continue
                if self.buf == self._start_tokens:
                    self.phase = "thinking"
                    self.buf = []
                else:
                    # The beginning does not match think_start = this turn is
                    # not thinking. Re-feed everything buffered so far into the
                    # content phase logic (including tool_call_start
                    # watching) — pouring it unconditionally into
                    # content_detok here would miss the case where the
                    # beginning happened to coincide with the head of
                    # tool_call_start.
                    self.phase = "content"
                    pending, self.buf = self.buf, []
                    for pt in pending:
                        self._feed_content_token(pt, out)
                continue
            if self.phase == "thinking":
                self.buf.append(t)
                if len(self.buf) <= len(self._end_tokens):
                    if self.buf == self._end_tokens:
                        self.phase = "content"
                        self.buf = []
                        self._strip_post_thinking_newlines = True
                    continue
                # Commit to thinking only the amount by which the window
                # exceeds the end marker length (always keeping a marker's
                # worth at the tail to continue the lookahead).
                cut = len(self.buf) - len(self._end_tokens)
                flush, self.buf = self.buf[:cut], self.buf[cut:]
                for ft in flush:
                    if not self._feed_thinking_token(ft):
                        break
                seg = self.think_detok.last_segment
                if seg:
                    out.append(("reasoning", seg))
                if self.budget_exceeded:
                    return out
                if self.buf == self._end_tokens:
                    self.phase = "content"
                    self.buf = []
                    self._strip_post_thinking_newlines = True
                continue
            if self.phase == "content":
                self._feed_content_token(t, out)
                continue
            # phase == "tool"
            self._feed_tool_token(t, out)
        return out

    def finalize(self) -> list[tuple[str, object]]:
        out: list[tuple[str, object]] = []
        if self.budget_exceeded:
            return out
        if self.phase == "detect":
            # Generation ended before reaching the marker length (an extremely
            # small max_tokens, for example). Err on the safe side and treat it
            # as content (including tool_call_start watching).
            pending, self.buf = self.buf, []
            for t in pending:
                self._feed_content_token(t, out)
        elif self.phase == "thinking":
            # max_tokens/eos was reached without emitting </think>. Commit the
            # uncommitted tokens left in the buffer as thinking.
            for t in self.buf:
                if not self._feed_thinking_token(t):
                    break
            self.buf = []
            self.think_detok.finalize()
            seg = self.think_detok.last_segment
            if seg:
                out.append(("reasoning", seg))
        elif self.phase == "tool":
            # max_tokens/eos was reached without emitting </tool_call>. Commit
            # the uncommitted tokens left in the buffer as tool, and report
            # "tool_end" as False (since it was not a real end marker).
            for t in self.tool_buf:
                self.tool_detok.add_token(t)
            self.tool_buf = []
            self.tool_detok.finalize()
            seg = self.tool_detok.last_segment
            if seg:
                out.append(("tool", seg))
            out.append(("tool_end", False))
        # Whatever was still in tool_call_start lookahead in the content phase
        # (a token sequence not yet confirmed to be a marker) when generation
        # ended is committed and dropped into content (a safety valve against
        # losing output).
        for t in self.tool_buf:
            self.content_detok.add_token(t)
        self.tool_buf = []
        self.content_detok.finalize()
        seg = self._clean_content_segment(self.content_detok.last_segment)
        if seg:
            out.append(("content", seg))
        return out


def _parse_positive_int(raw, cap: int, field_name: str) -> tuple[int, str | None]:
    """Convert to int and clamp into 1..cap. If it cannot be converted, or is 0
    or less, return an error string destined for a 400 (the caller wraps it in
    the protocol's format)."""

    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0, f"'{field_name}' must be an integer"
    if isinstance(raw, bool):  # bool is a subclass of int, so reject it explicitly
        return 0, f"'{field_name}' must be an integer"
    if value < 1:
        return 0, f"'{field_name}' must be a positive integer"
    return min(value, cap), None


def _resolve_max_tokens_openai(body: dict, cap: int) -> tuple[int, str | None]:
    """In OpenAI, ``max_tokens`` is deprecated and newer SDKs send
    ``max_completion_tokens``. Read both, preferring the latter. OpenAI treats
    0 as invalid (unlike Anthropic there is no special handling of
    ``max_tokens: 0``), so it goes straight through _parse_positive_int (below
    1 is a 400)."""

    raw = body.get("max_completion_tokens")
    if raw is None:
        raw = body.get("max_tokens")
    if raw is None:
        return cap, None
    return _parse_positive_int(raw, cap, "max_tokens")


def _parse_anthropic_max_tokens(raw, cap: int) -> tuple[int, str | None]:
    """Anthropic's real API accepts ``max_tokens: 0`` as legitimate input (it
    returns a special response that ends without generating). Unlike the
    OpenAI-side ``_parse_positive_int``, 0 alone is let through, while negative
    numbers and non-integers become a 400."""

    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0, "'max_tokens' must be an integer"
    if isinstance(raw, bool):
        return 0, "'max_tokens' must be an integer"
    if value < 0:
        return 0, "'max_tokens' must be a non-negative integer"
    return min(value, cap), None


def _parse_temperature(body: dict, default: float) -> tuple[float, str | None]:
    raw = body.get("temperature")
    if raw is None:
        return default, None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0, "'temperature' must be a number"
    if value < 0:
        return 0.0, "'temperature' must be non-negative"
    return value, None


def _parse_optional_float(
    body: dict, field: str, lo: float | None = None, hi: float | None = None
) -> tuple[float | None, str | None]:
    raw = body.get(field)
    if raw is None:
        return None, None
    if isinstance(raw, bool):
        return None, f"'{field}' must be a number"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"'{field}' must be a number"
    if lo is not None and value < lo:
        return None, f"'{field}' must be at least {lo}"
    if hi is not None and value > hi:
        return None, f"'{field}' must be at most {hi}"
    return value, None


def _parse_optional_int(
    body: dict, field: str, lo: int | None = None
) -> tuple[int | None, str | None]:
    raw = body.get(field)
    if raw is None:
        return None, None
    # bool is a subclass, so reject it explicitly. Floats are rejected here even
    # when they hold a true integer value (5.0 and the like): int(1.5) raises no
    # exception and silently truncates to 1, so something meant as "'field' must
    # be an integer" would quietly let a different value through.
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None, f"'{field}' must be an integer"
    if lo is not None and raw < lo:
        return None, f"'{field}' must be at least {lo}"
    return raw, None


def _parse_logit_bias(body: dict) -> tuple[dict[int, float] | None, str | None]:
    raw = body.get("logit_bias")
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "'logit_bias' must be an object mapping token id to bias"
    try:
        parsed = {int(k): float(v) for k, v in raw.items()}
    except (TypeError, ValueError):
        return None, "'logit_bias' must be an object mapping token id (string) to a numeric bias"
    return parsed, None


# OpenAI and Anthropic both use the same field names (top_p/top_k/min_p/
# repetition_penalty/presence_penalty/frequency_penalty/logit_bias/seed), so a
# single shared parser with no protocol branching suffices (only things that go
# by different names, such as thinking/stop_sequences, need per-protocol
# handling).
#
# Keys that were not specified are left out of the returned dict — so that
# Runner.generate's own default arguments (= the disabled values) take over
# unchanged.
def _parse_sampling_params(body: dict) -> tuple[dict, str | None]:
    params: dict = {}

    top_p, err = _parse_optional_float(body, "top_p", 0.0, 1.0)
    if err is not None:
        return {}, err
    if top_p is not None:
        params["top_p"] = top_p

    top_k, err = _parse_optional_int(body, "top_k", -1)
    if err is not None:
        return {}, err
    if top_k is not None:
        params["top_k"] = top_k

    min_p, err = _parse_optional_float(body, "min_p", 0.0, 1.0)
    if err is not None:
        return {}, err
    if min_p is not None:
        params["min_p"] = min_p

    repetition_penalty, err = _parse_optional_float(body, "repetition_penalty", 0.0, None)
    if err is not None:
        return {}, err
    if repetition_penalty is not None:
        params["repetition_penalty"] = repetition_penalty

    presence_penalty, err = _parse_optional_float(body, "presence_penalty")
    if err is not None:
        return {}, err
    if presence_penalty is not None:
        params["presence_penalty"] = presence_penalty

    frequency_penalty, err = _parse_optional_float(body, "frequency_penalty")
    if err is not None:
        return {}, err
    if frequency_penalty is not None:
        params["frequency_penalty"] = frequency_penalty

    logit_bias, err = _parse_logit_bias(body)
    if err is not None:
        return {}, err
    if logit_bias is not None:
        params["logit_bias"] = logit_bias

    seed, err = _parse_optional_int(body, "seed")
    if err is not None:
        return {}, err
    if seed is not None:
        params["seed"] = seed

    return params, None


# The list of "identity values" for sampling parameters (default values that do
# not change the distribution at all). top_p=1.0, for example, leaves 100% of
# the probability mass, so it does not change the distribution. Even for keys
# that SpecRunner does not list in SUPPORTED_SAMPLING_PARAMS, if the value that
# actually arrived is one of the identity values enumerated here it threatens
# nothing about the speculative decoder's distribution guarantee, so
# _resolve_runner_for_request does not reject it with a 400 (when it is not an
# identity value, the request is downgraded per request; see item 7). This
# keeps the magic numbers gathered here instead of scattered through
# conditionals. For logit_bias the identity value is "None or an empty dict",
# so it is special-cased inside _is_identity_sampling_value rather than by
# value.
#
# -1 for top_k is treated, like 0, as an identity value that "disables top-k".
# mlx_lm.make_sampler, which FallbackRunner uses, also enables the filter only
# when ``top_k > 0``, so passing this value straight through does not change
# the distribution.
_IDENTITY_SAMPLING_VALUES: dict[str, tuple] = {
    "top_p": (0.0, 1.0),
    "top_k": (0, -1),
    "min_p": (0.0,),
    "frequency_penalty": (0.0,),
    "presence_penalty": (0.0,),
    "repetition_penalty": (0.0, 1.0),
}


def _is_identity_sampling_value(name: str, value) -> bool:
    """True if ``value`` is an identity value (one that does not change the
    distribution at all) for the sampling parameter ``name``. By design
    _parse_sampling_params never puts an unspecified value (None) into the
    params dict, but defensively it is let through here as well."""

    if value is None:
        return True
    if name == "logit_bias":
        return not value  # None or {} (an empty dict) means no bias
    return value in _IDENTITY_SAMPLING_VALUES.get(name, ())


def _resolve_runner_for_request(
    params: dict, *, logprobs_requested: bool = False
) -> tuple[object, str | None, str | None]:
    """Decide which runner this request actually uses. The return value is
    ``(runner, downgrade_reason, error)``.

    When non-identity sampling parameters (top_p=0.9, say) or logprobs are
    specified that the current ``STATE.runner`` (SpecRunner/FlashSpecRunner)
    does not accept, they used to be rejected with a 400 (Kimi K3 review item
    7). Real clients (opencode, the OpenAI SDK, and so on) routinely send top_p
    and friends with non-default values, so the experience was that merely
    being a speculative-decoding-capable model got you rejected out of hand.

    Here, instead of a 400, if ``STATE.downgrade_runner`` (the non-speculative
    FallbackRunner main() prepared at startup; see runner.py) is available, this
    request alone is downgraded to it. The speculative decoder's distribution
    guarantee (see the SpecRunner/FlashSpecRunner docstring) is trivially upheld
    by "not using speculation", so the request can be served with the values
    themselves correctly honored — and rather than downgrading silently, the
    reason is written to the server log (the caller must also put it into the
    response). This follows this repository's overall policy of preventing
    "silently falling back to the fallback with no way to notice" — the same
    idea as build_runner's fallback_reason.

    When no downgrade happens, the side effect is the same as in the original
    _check_and_strip_sampling_params: ``params`` is mutated in place, deleting
    the keys that the current runner does not support but that are let through
    because they are identity values (this prevents type-mismatched arguments
    leaking into a generate() that has no **kwargs, such as
    SpecEngine.generate()).

    Only when a non-identity value arrives and a downgrade is also impossible
    (= there is no STATE.downgrade_runner; a model that was already on the
    fallback at startup always takes this path) is a 400-worthy error string
    returned as before — this normally does not happen (the fallback runner
    declares support for every key), but it is kept as a safety net for when
    other runner kinds are added in the future.
    """

    supported = getattr(STATE.runner, "SUPPORTED_SAMPLING_PARAMS", frozenset())
    unrecognized = set(params) - supported
    non_identity = sorted(
        name for name in unrecognized if not _is_identity_sampling_value(name, params[name])
    )
    needs_downgrade = bool(non_identity) or logprobs_requested
    if needs_downgrade and STATE.downgrade_runner is not None:
        triggers = list(non_identity)
        if logprobs_requested:
            triggers.append("logprobs")
        kind = getattr(STATE.runner, "KIND", type(STATE.runner).__name__)
        reason = (
            f"non-default sampling parameter(s)/logprobs ({', '.join(triggers)}) are "
            f"incompatible with the '{kind}' speculative-decoding runner's exact-"
            "distribution guarantee; this request was transparently downgraded to "
            "non-speculative ('fallback') generation instead of being rejected"
        )
        print(f"[mlxturbo-serve] request downgraded to non-speculative generation: {reason}")
        return STATE.downgrade_runner, reason, None
    if not unrecognized:
        return STATE.runner, None, None
    if non_identity:
        kind = getattr(STATE.runner, "KIND", type(STATE.runner).__name__)
        return (
            STATE.runner,
            None,
            f"this model is served via the '{kind}' runner, which does not support "
            f"non-default values for: {', '.join(non_identity)}. The speculative-decoding "
            "runner only supports 'seed' (plus identity values that leave the sampling "
            "distribution unchanged, e.g. top_p=1.0 or frequency_penalty=0.0) among the "
            "extended sampling parameters, because its correctness guarantee (rejection "
            "sampling against the exact target distribution) assumes temperature-only "
            "sampling; changing the base distribution would silently break that guarantee.",
        )
    # Everything left here is a key that "the runner does not support but that
    # is an identity value". It does not change the distribution, so it is not
    # rejected; but drop it before passing it on, so that runner.generate()
    # does not receive it as a keyword argument it does not know.
    for name in unrecognized:
        del params[name]
    return STATE.runner, None, None


def _parse_logprobs_openai_chat(body: dict) -> tuple[bool, int, str | None]:
    """The logprobs contract of OpenAI chat-completions: ``logprobs: bool``
    (default false) plus ``top_logprobs: int`` (0-20), which is only meaningful
    when ``logprobs: true``. Kimi K3 review item 17."""

    raw = body.get("logprobs", False)
    if not isinstance(raw, bool):
        return False, 0, "'logprobs' must be a boolean"
    top_n, err = _parse_optional_int(body, "top_logprobs", 0)
    if err is not None:
        return False, 0, err
    if top_n is not None:
        if top_n > 20:
            return False, 0, "'top_logprobs' must be at most 20"
        if not raw:
            return False, 0, "'top_logprobs' requires 'logprobs': true"
    return raw, (top_n or 0), None


def _parse_logprobs_openai_legacy(body: dict) -> tuple[bool, int, str | None]:
    """The logprobs contract of OpenAI's legacy ``/v1/completions``:
    ``logprobs: int | null`` itself doubles as the top-N count (there is no
    separate bool as in the chat version)."""

    top_n, err = _parse_optional_int(body, "logprobs", 0)
    if err is not None:
        return False, 0, err
    if top_n is None:
        return False, 0, None
    if top_n > 20:
        return False, 0, "'logprobs' must be at most 20"
    return True, top_n, None


def _utf8_bytes_or_none(token: str) -> list[int] | None:
    try:
        return list(token.encode("utf-8"))
    except Exception:
        return None


def _build_chat_logprobs(entries: list[dict] | None) -> dict | None:
    """Convert the ``res["logprobs"]`` returned by ``FallbackRunner.generate``
    (see ``_logprob_entry`` in mlxturbo/runner.py; it corresponds 1:1 with the
    generated token sequence) into the ``choices[].logprobs`` shape of OpenAI
    chat-completions. If it was not requested, or the runner does not support
    it, the caller does not call this at all (``res`` has no ``"logprobs"`` key
    in the first place)."""

    if not entries:
        return None
    content = []
    for e in entries:
        item = {
            "token": e["token"],
            "logprob": e["logprob"],
            "bytes": _utf8_bytes_or_none(e["token"]),
            "top_logprobs": [
                {
                    "token": t["token"],
                    "logprob": t["logprob"],
                    "bytes": _utf8_bytes_or_none(t["token"]),
                }
                for t in e.get("top_logprobs", [])
            ],
        }
        content.append(item)
    return {"content": content}


def _build_legacy_logprobs(entries: list[dict] | None) -> dict | None:
    """Convert the same ``res["logprobs"]`` into the ``choices[].logprobs``
    shape of the legacy ``/v1/completions`` (the parallel arrays
    ``tokens``/``token_logprobs``/``top_logprobs``/``text_offset``)."""

    if not entries:
        return None
    tokens = [e["token"] for e in entries]
    token_logprobs = [e["logprob"] for e in entries]
    top_logprobs = [
        {t["token"]: t["logprob"] for t in e.get("top_logprobs", [])} for e in entries
    ]
    text_offset = []
    acc = 0
    for t in tokens:
        text_offset.append(acc)
        acc += len(t)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }


def _stop_sequences(body: dict) -> list[str]:
    """Accept both OpenAI's ``stop`` (a string or an array) and Anthropic's
    ``stop_sequences`` (an array). The same handling is used on either endpoint
    (it depends only on the shape of the request body)."""

    raw = body.get("stop")
    if raw is None:
        raw = body.get("stop_sequences")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str) and s]
    return []


def _find_stop(text: str, stops: list[str]) -> tuple[int, str] | None:
    """Return the match among `stops` that starts earliest in `text`, as
    (start position, matched string)."""

    best = None
    for s in stops:
        idx = text.find(s)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, s)
    return best


def _check_response_format(body: dict) -> str | None:
    """``response_format`` (OpenAI's JSON mode) is not silently ignored, since
    there is no constrained-decoding implementation (Kimi K3 review item 8).
    This prevents the accident where an agentic client expecting JSON receives
    plain text and fails to parse it, following the same "explicitly refuse
    what is not supported" policy as previous_response_id (the existing 400 in
    this file). ``{"type": "text"}`` (equivalent to not specifying it) involves
    no transformation, so it is let through as-is."""

    rf = body.get("response_format")
    if rf is None:
        return None
    if not isinstance(rf, dict) or "type" not in rf:
        return "'response_format' must be an object with a 'type' field"
    rf_type = rf["type"]
    if rf_type == "text":
        return None
    return (
        f"'response_format' of type {rf_type!r} is not supported: this server has no "
        "constrained-decoding implementation, so it cannot guarantee the output actually "
        "conforms to the requested format. Omit 'response_format' (or pass "
        "{\"type\": \"text\"}) and parse/validate the model's plain-text output yourself."
    )


def _downgrade_headers(downgrade_reason: str | None) -> dict[str, str] | None:
    """An HTTP header that makes a per-request downgrade (item 7) observable in
    streaming responses too. Each SSE chunk follows the standard OpenAI/
    Anthropic schema, so mixing a custom field into it could break an agentic
    client's schema validation — a header conveys the same information without
    changing the per-chunk JSON shape at all. None when no downgrade occurred
    (StreamingResponse treats headers=None as "leave the defaults alone")."""

    if downgrade_reason is None:
        return None
    return {"X-Mlxturbo-Downgrade-Reason": downgrade_reason}


def _attach_downgrade_reason(body: dict, downgrade_reason: str | None) -> None:
    """Add the reason for a per-request downgrade (item 7) verbatim to the JSON
    body of a non-streaming response. Does nothing if no downgrade occurred
    (the key itself is omitted — the same manner as /health's
    fallback_reason). It is an extra field not present in the standard schema,
    but just as with `/health`'s fallback_reason, this repository prioritizes
    "never downgrade silently" over the shape of the response."""

    if downgrade_reason is not None:
        body["downgrade_reason"] = downgrade_reason


def _openai_error(
    message: str,
    status: int = 400,
    err_type: str = "invalid_request_error",
    code: str | None = None,
):
    # OpenAI's error object carries nullable ``param`` and ``code`` keys even
    # when a validation failure is not tied to one request field.
    err = {"message": message, "type": err_type, "param": None, "code": code}
    return JSONResponse(status_code=status, content={"error": err})


def _anthropic_error(message: str, status: int = 400, err_type: str = "invalid_request_error"):
    request_id = f"req_{uuid.uuid4().hex}"
    return JSONResponse(
        status_code=status,
        content={
            "type": "error",
            "error": {"type": err_type, "message": message},
            "request_id": request_id,
        },
        headers={"request-id": request_id},
    )


def _model_name_allowed(requested: str) -> bool:
    """Let a request through if it matches either the served name itself or an
    alias explicitly permitted with --model-alias at startup. By default
    (--model-alias not specified) STATE.model_aliases is the empty set, so this
    remains the exact match that permits only the served name, as before — the
    deliberate, OpenAI-conforming design of "a mismatch is a 404" is left
    intact, and it never turns into silently accepting anything."""

    return requested == STATE.model_name or requested in STATE.model_aliases


def _check_model_openai(body: dict):
    """If the request's model differs from the served model (or from an alias
    permitted with --model-alias), return 404 (the equivalent of OpenAI's
    model_not_found). Omitting it is allowed. Whether or not it matches, the
    response's "model" field always carries STATE.model_name (the served name)
    — the string the client sent (aliases included) is never echoed back."""

    requested = body.get("model")
    if requested is not None and not _model_name_allowed(requested):
        return _openai_error(
            f"The model `{requested}` does not exist or you do not have access to it.",
            status=404,
            err_type="invalid_request_error",
            code="model_not_found",
        )
    return None


def _check_model_anthropic(body: dict):
    requested = body.get("model")
    if requested is not None and not _model_name_allowed(requested):
        return _anthropic_error(
            f"model: {requested} not found", status=404, err_type="not_found_error"
        )
    return None


def _check_context_length(prompt_ids: list[int], protocol: str):
    """Return 400 if the prompt exceeds the limit decided at startup (the
    smaller of the model config's max_position_embeddings and the value derived
    from the actual limit Metal can allocate in one go; overridable with
    ``--max-context-tokens``; see ``_resolve_default_max_context_tokens``).

    This is an up-front check that prevents the accident where Metal cannot
    allocate the attention matrix in one go, dies with [metal::malloc], and
    turns into a 500 (measured to occur from around ~57,000 tokens). SpecEngine
    forwards a new prompt in one go (chunk splitting was dropped because, on
    4-bit quantized models, mx.quantized_matmul was measured to return
    different rounding depending on the batch length, so the generated token
    sequence no longer matched between the split and unsplit cases), so this
    limit is effectively the only guard."""

    limit = STATE.max_context_tokens
    if limit is None or len(prompt_ids) <= limit:
        return None
    message = (
        f"This model's maximum context length is {limit} tokens. "
        f"However, your messages resulted in {len(prompt_ids)} tokens. "
        "Please reduce the length of the messages."
    )
    if protocol == "anthropic":
        return _anthropic_error(message, status=400, err_type="invalid_request_error")
    return _openai_error(
        message, status=400, err_type="invalid_request_error", code="context_length_exceeded"
    )


def _finish_reason_openai(tokens: list[int]) -> str:
    return "stop" if tokens and tokens[-1] in STATE.eos_ids else "length"


def _stop_reason_anthropic(tokens: list[int]) -> str:
    return "end_turn" if tokens and tokens[-1] in STATE.eos_ids else "max_tokens"


def _responses_terminal_state(
    tokens: list[int], budget_exceeded: bool, has_tool_calls: bool
) -> tuple[str, str | None]:
    """Return the Responses status/reason from the same terminal facts as Chat.

    A fully parsed tool call is a successful assistant turn even when the model
    does not append EOS.  Chat Completions and Anthropic already give tool calls
    precedence over the token cap; Responses must not describe the same output
    as incomplete merely because its adapter looked only at the last token.
    """

    if budget_exceeded:
        return "incomplete", "max_output_tokens"
    if has_tool_calls or (tokens and tokens[-1] in STATE.eos_ids):
        return "completed", None
    return "incomplete", "max_output_tokens"


def _usage_dict(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> dict:
    """OpenAI-format usage. ``prompt_tokens_details.cached_tokens`` carries the
    measured prefill reuse of the ChatSession (``res["prefill_reused"]``)
    verbatim (the same shape as mlx_lm:1339-1346 / 1567-1575). The key is
    emitted even when it is 0 (no reuse, or a path such as FallbackRunner that
    has no prefill reuse mechanism) — the shape of the response does not
    distinguish "supported but zero this time" from "not supported" (mlx_lm
    does the same, emitting the field whenever prompt_cache_count is 0 or
    more).
    """

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens >= 0:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return usage


def _anthropic_usage(prompt_tokens: int, completion_tokens: int, res: dict) -> dict:
    """Anthropic-format usage. Maps the measured prefill reuse
    (``res["prefill_reused"]``/``res["prefill_new"]``, the same numbers as
    cli.py's display line and ``_log_gen_stats``) onto Anthropic's cache-related
    fields (Kimi K3 review item 16). The ``cache_control`` that Claude Code and
    others send is itself not read — it is an instruction about "how far to
    cache", but this server merely decides the amount of reuse mechanically per
    request from the session's LCP and has no notion of block-level cache
    boundaries, so it accepts and ignores it.

    Aligned with the meaning in Anthropic's real API: ``input_tokens`` is the
    number of tokens "actually processed" excluding what was read from the
    cache (= prefill_new), and ``cache_read_input_tokens`` is the reused part
    (= prefill_reused). ``cache_creation_input_tokens`` takes the newly
    processed part written into the session (= prefill_new) verbatim — this
    server's session stacks up KV every turn, so in principle everything newly
    processed becomes a candidate for reuse on subsequent turns (there is no
    ephemeral 5m/1h breakdown, only a flat total). Even for a runner whose
    ``res`` carries no prefill information (normally there is none, but
    defensively the default is 0), the fields themselves are always emitted —
    the same policy as ``_usage_dict``: the shape of the response does not
    distinguish 0 from "not supported".
    """

    prefill_reused = res.get("prefill_reused", 0)
    prefill_new = res.get("prefill_new", max(prompt_tokens - prefill_reused, 0))
    return {
        "input_tokens": prefill_new,
        "output_tokens": completion_tokens,
        "cache_creation_input_tokens": prefill_new,
        "cache_read_input_tokens": prefill_reused,
    }


def _log_gen_stats(res: dict) -> None:
    """Whether prefill reuse is working is a number that does not exist in the
    OpenAI/Anthropic response formats, so the same content as cli.py's display
    line is written to a one-line log for the operator."""

    line = (
        f"[mlxturbo-serve] prefill reused={res.get('prefill_reused', 0)} "
        f"new={res.get('prefill_new', 0)} decode={res.get('decode_tps', 0.0):.1f}tok/s"
        f" tok/step={res.get('tokens_per_step', 0.0):.2f}"
        f" ttft={res.get('ttft_s', 0.0):.2f}s"
    )
    ph = res.get("phase")
    if ph:
        rounds = ph.get("rounds") or 1
        parts = " ".join(
            f"{k}={ph[k] * 1000 / rounds:.1f}ms"
            for k in ("draft", "verify", "post", "rollback") if k in ph
        )
        line += f" | phase/round: {parts} (rounds={rounds})"
    print(line)


# ---------- generation plumbing ----------
#
# on_tokens hands over several tokens at once, batched by what speculation
# accepted. The raw token sequence is always passed (SpecRunner has nothing
# else, and FallbackRunner always passes toks in on_tokens(toks, text)), so
# ThinkingRouter can split reasoning/content with the same logic on either
# path. When thinking separation is involved, FallbackRunner's precomputed text
# (a workaround to avoid double decoding) is not used and we always
# re-detokenize from the raw ids here — because the raw tokens are needed to
# judge across marker boundaries, and because reasoning and content need
# separate detokenizers.
#
# STATE.runner.generate is synchronous and long-running, so it is always
# submitted to the same dedicated worker thread that loaded the model
# (STATE.executor). Escaping to another thread via asyncio's general-purpose
# thread pool (asyncio.to_thread's default executor) or threading.Thread
# crashes with "There is no Stream(gpu, N) in current thread", because the
# model weights and the KV cache are bound to the thread that loaded them
# (confirmed by measurement: see the docstring in server.py).


class _GenerationCancelled(Exception):
    """Private cooperative-stop signal raised from a runner callback."""


def _resolve_batch_tier(
    gen_runner, prompt_ids: list[int], max_tokens: int, logprobs_requested: bool = False
) -> str | None:
    """``None`` means "not batch-eligible — route through STATE.lock exactly
    as before" (this is also what makes --max-batch's default of 1, and any
    server not passing it at all, provably identical to before this change:
    STATE.batch_coordinator is None, so every call short-circuits here).

    Otherwise ``"pool"``/``"solo"`` (mlxturbo.batch.classify, via
    runner.batch_tier) — see the ModelState.batch_coordinator field
    docstring and mlxturbo/batch.py's module docstring for what the two mean.

    Excludes logprobs requests: batching does not collect logprobs yet (see
    runner.start_batched_generation's docstring), so such a request simply
    keeps going through the existing STATE.lock + FallbackRunner.generate
    path, identical to how it is served today.
    """

    coordinator = STATE.batch_coordinator
    if coordinator is None or logprobs_requested:
        return None
    if not can_batch(gen_runner):
        return None
    return _runner_batch_tier(coordinator, prompt_ids, max_tokens)


async def _run_generate_batched(
    prompt_ids, max_tokens, temp, on_tokens=None, **sampling_kwargs
) -> dict:
    """The batched-path analogue of ``_run_generate`` — used instead of it
    (never both) whenever ``_resolve_batch_tier`` returned non-``None``.

    No session is passed (batched requests always do a fresh prefill; see
    mlxturbo/batch.py's module docstring — the same simplification already
    used for a per-request-downgraded request, see the callers'
    ``session=None`` comment). No ``STATE.lock`` either: concurrency between
    admissions is the entire point, and mutual exclusion around the model
    forward pass now lives inside ``BatchCoordinator`` itself.

    Cancellation: mirrors ``_run_generate``'s own shield/defer pattern. A
    non-streaming caller has no ``cancel_event`` to signal early, so — same
    as the pre-existing non-batched non-streaming path — a client
    disconnect does not stop generation early, only defers when the
    ``CancelledError`` is allowed to propagate until the Future is done.
    """

    future = start_batched_generation(
        STATE.batch_coordinator, prompt_ids, max_tokens, temp, on_tokens, None, None,
        **sampling_kwargs,
    )
    wrapped = asyncio.wrap_future(future)
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(wrapped)
        except asyncio.CancelledError as exc:
            if wrapped.cancelled():
                raise
            cancelled = exc
            continue
        except Exception:
            if cancelled is not None:
                raise cancelled
            raise
        if cancelled is not None:
            raise cancelled
        return result


def _start_batched_generation(
    prompt_ids,
    max_tokens,
    temp,
    thinking_budget,
    tool_calling_enabled: bool = False,
    tools_for_parsing=None,
    **sampling_kwargs,
):
    """The batched-path analogue of ``_start_generation`` — same external
    contract (``q, future, cancel_event, raw_token_count``), so every
    streaming call site can swap one for the other without any other change:
    the SSE-side consumption code (``_await_with_keepalive``,
    ``asyncio.to_thread(q.get)``, ``_await_worker``) neither knows nor cares
    which one produced its queue.
    """

    q, cancel_event, raw_token_count, on_tokens, on_done = _build_streaming_pipeline(
        prompt_ids, thinking_budget, tool_calling_enabled, tools_for_parsing
    )
    future = start_batched_generation(
        STATE.batch_coordinator,
        prompt_ids,
        max_tokens,
        temp,
        on_tokens,
        on_done,
        cancel_event,
        **sampling_kwargs,
    )
    # A plain concurrent.futures.Future, exactly like _start_generation's own
    # STATE.executor.submit(worker) return value — _await_worker wraps it
    # with asyncio.wrap_future itself.
    return q, future, cancel_event, raw_token_count


async def _run_generate(
    prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, runner=None, **sampling_kwargs
):
    """Omitting ``runner`` means ``STATE.runner`` (the normal path). For a
    request that was downgraded per request (item 7; see
    ``_resolve_runner_for_request``), the caller explicitly passes
    ``STATE.downgrade_runner``."""

    loop = asyncio.get_running_loop()
    fn = functools.partial(
        (runner or STATE.runner).generate,
        prompt_ids,
        max_tokens=max_tokens,
        temp=temp,
        eos_ids=eos_ids,
        on_tokens=on_tokens,
        session=session,
        **sampling_kwargs,
    )
    future = loop.run_in_executor(STATE.executor, fn)
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError as exc:
            if future.cancelled():
                raise
            # A client cancellation must not release STATE.lock while this
            # synchronous worker still owns/mutates its selected session.
            # Defer propagating cancellation until the executor future ends;
            # the loop also tolerates a repeated cancellation request.
            cancelled = exc
            continue
        except Exception:
            if cancelled is not None:
                raise cancelled
            raise
        if cancelled is not None:
            raise cancelled
        return result


def _chunk_string(s: str, size: int = 24) -> list[str]:
    """Split the arguments JSON string into fixed-length pieces. In both the
    OpenAI and Anthropic streaming conventions it is customary for "arguments
    to stream in pieces", but in reality the whole text of a tool call can only
    be interpreted as JSON after it has been buffered until the marker closes
    (partial JSON cannot be streamed mid-flight, because we do not fabricate
    anything if it turns out to be broken). So once parsing has settled, this
    function splits it artificially into events — concatenating them always
    reproduces the original JSON string, so this does not conflict with the
    client-side implementation that "concatenates the fragments and then
    parses".
    """

    if not s:
        return [""]
    return [s[i : i + size] for i in range(0, len(s), size)]


def _parse_tool_calls_text(raw: str, tools_for_parsing) -> list[dict] | None:
    """Convert the raw text that was enclosed between tool_call markers into
    one or more structured tool calls. It uses ``tokenizer.tool_parser`` as-is
    (the model-specific parser mlx_lm selects automatically from the chat
    template string — for Qwen's naive JSON format that is ``json.loads``
    itself, and for Qwen3-Coder's ``<function=...>`` XML-like format a
    dedicated parser) — detection of the marker strings has already been done
    by ThinkingRouter, so only the syntactic parsing of the contents is needed
    here.

    To support parsers that can extract several calls from a single marker
    region (pythonic and the like), a list return value is expanded as-is (the
    same treatment as ToolCallFormatter in mlx_lm.server).

    If parsing fails (broken JSON, a missing name, and so on) None is returned
    — the caller uses this as the trigger for the fallback of "do not fabricate
    a tool call; return the raw text as content instead".
    """

    parser = getattr(STATE.tokenizer, "tool_parser", None)
    if parser is None:
        return None
    try:
        parsed = parser(raw, tools_for_parsing)
    except Exception as exc:  # noqa: BLE001 - model output drives third-party parsers
        print(
            f"[mlxturbo-serve] tool call の解析に失敗 ({type(exc).__name__}: {exc})。"
            " テキストとしてそのまま返す"
        )
        return None
    if not isinstance(parsed, list):
        parsed = [parsed]
    calls = []
    for tc in parsed:
        if not isinstance(tc, dict):
            return None
        name = tc.get("name")
        if not isinstance(name, str) or not name:
            return None
        arguments = tc.get("arguments", {})
        if not isinstance(arguments, dict):
            return None
        try:
            # Protocol responses require a JSON object.  Some mlx_lm parsers
            # use ast.literal_eval and can therefore produce sets/tuples or
            # non-finite floats that json.dumps would reject (or encode as
            # non-standard JSON) only after streaming has already started.
            json.dumps(arguments, allow_nan=False)
        except (TypeError, ValueError):
            return None
        raw_id = tc.get("id")
        tc_id = raw_id if isinstance(raw_id, str) and raw_id else f"call_{uuid.uuid4().hex}"
        calls.append({"name": name, "arguments": arguments, "id": tc_id})
    return calls


class SegmentAssembler:
    """Convert the (channel, payload) stream returned by ThinkingRouter into
    the high-level events the caller uses: ``("reasoning_delta", text)`` /
    ``("content_delta", text)`` / ``("tool_call", {"id","name","arguments"})``.

    The "tool"/"tool_start"/"tool_end" channels buffer the raw text of the
    marker region and only run it through ``_parse_tool_calls_text`` once the
    region closes ("tool_end"): on success it returns tool_call events (several
    of them if there were several calls), and on failure it returns, as a
    content_delta, the raw text including the marker strings, following the
    server-wide policy of "do not fabricate; return it as text" (if tool_end
    did not close with a real marker, the end-marker string is not appended —
    so that it is visible that it was cut off by max_tokens or similar).

    The same instance can be reused as-is for both streaming (pushing each time
    from on_tokens) and non-streaming (pushing everything at once after
    generation) — the only state is tool_buf (the uncommitted tool text
    fragments).
    """

    def __init__(self, tools_for_parsing):
        self.tools_for_parsing = tools_for_parsing
        self._tool_buf: list[str] | None = None

    def push(self, channel: str, payload) -> list[tuple[str, object]]:
        if channel == "reasoning":
            return [("reasoning_delta", payload)] if payload else []
        if channel == "content":
            return [("content_delta", payload)] if payload else []
        if channel == "tool_start":
            self._tool_buf = []
            return []
        if channel == "tool":
            if payload:
                self._tool_buf.append(payload)
            return []
        if channel == "tool_end":
            matched = bool(payload)
            raw = "".join(self._tool_buf) if self._tool_buf is not None else ""
            self._tool_buf = None
            # A JSON-looking prefix is not a completed tool call until the
            # model emits the closing marker.  Parsing an unterminated block
            # fabricates a structured call from truncated model output.
            calls = _parse_tool_calls_text(raw, self.tools_for_parsing) if matched else None
            if calls:
                return [("tool_call", c) for c in calls]
            start_m = getattr(STATE.tokenizer, "tool_call_start", None) or ""
            end_m = (getattr(STATE.tokenizer, "tool_call_end", None) or "") if matched else ""
            text = f"{start_m}{raw}{end_m}"
            return [("content_delta", text)] if text else []
        return []


def _start_generation(
    prompt_ids,
    max_tokens: int,
    temp: float,
    thinking_budget: int | None,
    tool_calling_enabled: bool = False,
    tools_for_parsing=None,
    session=None,
    runner=None,
    **sampling_kwargs,
):
    """Submit the worker to STATE.executor and return (queue, Future, stop
    Event, raw_token_count).

    ``session`` is the session (ChatSession/FallbackSession) that
    ``_select_session`` assigned for this request — the caller selects it
    inside ``STATE.lock`` and passes it in (there is no global STATE.session).

    Omitting ``runner`` means ``STATE.runner``. For a request that was
    downgraded per request (item 7; see ``_resolve_runner_for_request``), the
    caller explicitly passes ``STATE.downgrade_runner``.

    ``raw_token_count`` is a one-element ``[int]`` list (a mutable box visible
    to the caller) — ``on_tokens`` adds to it the number of raw tokens actually
    processed each time. When a stop string matches and generation is cut off
    early via cancel_event.set() (item 18), runner.generate() does not return a
    normal res dict but yields ``("cancelled", None)``, so this stands in for
    usage's completion_tokens.

    The elements pushed onto the queue: ``("reasoning_delta", text)`` /
    ``("content_delta", text)`` / ``("tool_call", {"id","name","arguments"})``
    (one successfully parsed tool call) / ``("budget_exceeded", None)`` (once,
    only on the round where the budget overrun is detected) /
    ``("done", res)`` / ``("cancelled", None)`` / ``("error", exc)``.

    The caller must always wait for this Future to the very end (on every path:
    normal completion, error, and client disconnect). Otherwise the next
    request could take the lock while a still-running worker is touching this
    ``session``, producing a race in which two generations rewrite the same
    session at once.
    """

    q, cancel_event, raw_token_count, on_tokens, on_done = _build_streaming_pipeline(
        prompt_ids, thinking_budget, tool_calling_enabled, tools_for_parsing
    )

    def worker():
        try:
            res = (runner or STATE.runner).generate(
                prompt_ids,
                max_tokens=max_tokens,
                temp=temp,
                eos_ids=STATE.eos_ids,
                on_tokens=on_tokens,
                session=session,
                **sampling_kwargs,
            )
            on_done("done", res)
        except _GenerationCancelled:
            # Cancelling the asyncio Task around ``to_thread(q.get)`` cannot
            # cancel the underlying blocking thread.  Wake that orphaned get
            # even though the SSE consumer itself is already gone.
            on_done("cancelled", None)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an SSE error event
            on_done("error", exc)

    # Submit to the generation-only thread (executor). The async side picks up
    # results through the queue with asyncio.to_thread(q.get) (that is only a
    # queue.Queue wait, unrelated to MLX's thread pinning, so a general-purpose
    # thread pool is fine there).
    future = STATE.executor.submit(worker)
    return q, future, cancel_event, raw_token_count


def _build_streaming_pipeline(
    prompt_ids, thinking_budget, tool_calling_enabled, tools_for_parsing
):
    """The part of ``_start_generation`` that has nothing to do with *how*
    generation actually runs: the queue, the cancellation flag, the raw token
    counter, and the ThinkingRouter/SegmentAssembler pipeline that turns a raw
    token stream into the ``(kind, val)`` events the SSE layer consumes.

    Shared verbatim by ``_start_generation`` (submits a single blocking
    ``runner.generate`` call to ``STATE.executor``) and
    ``_start_batched_generation`` (submits an ``Admission`` to
    ``STATE.batch_coordinator`` instead — see mlxturbo/batch.py) so the two
    only differ in how the token stream is actually produced, never in how it
    is turned into SSE events.

    Returns ``(q, cancel_event, raw_token_count, on_tokens, on_done)``.
    ``on_done(kind, val)`` is the three-outcome completion callback
    (``"done"``/``"cancelled"``/``"error"``) both callers invoke exactly
    once; for ``"done"`` it runs the router's finalize pass (flushing any
    still-buffered reasoning/content/tool-call text) before pushing the
    terminal event, exactly as the original inline ``worker()`` did — for
    ``"cancelled"``/``"error"`` it does not, also unchanged from before.
    """

    q: queue.Queue = queue.Queue()
    cancel_event = threading.Event()
    eos_ids = STATE.eos_ids
    router = ThinkingRouter(
        STATE.tokenizer,
        thinking_budget,
        eos_ids,
        tool_calling_enabled=tool_calling_enabled,
        already_thinking=_prompt_already_thinking(prompt_ids),
    )
    assembler = SegmentAssembler(tools_for_parsing)
    signaled = [False]
    raw_token_count = [0]

    def on_tokens(toks, text=None):
        if cancel_event.is_set():
            # All runner families invalidate aliased session state before
            # mutation and publish only after successful completion.  Raising
            # here therefore stops the next decode round without exposing a
            # half-mutated cache.  Prefill itself remains non-preemptible.
            raise _GenerationCancelled()
        raw_token_count[0] += len(toks)
        for channel, payload in router.feed(toks):
            for kind, val in assembler.push(channel, payload):
                q.put((kind, val))
        if router.budget_exceeded and not signaled[0]:
            q.put(("budget_exceeded", None))
            signaled[0] = True

    def on_done(kind, val):
        if kind == "done":
            if not router.budget_exceeded:
                for channel, payload in router.finalize():
                    for k2, v2 in assembler.push(channel, payload):
                        q.put((k2, v2))
            q.put(("done", val))
        else:
            q.put((kind, val))

    return q, cancel_event, raw_token_count, on_tokens, on_done


async def _await_worker(future) -> None:
    """Wait for the worker Future. Exceptions inside the worker are caught and
    swallowed by worker() itself via q.put(("error", ...)), so the only things
    raised here are unexpected problems around the creation/cancellation of the
    future itself. Since this is called from a generator's finally, ordinary
    exceptions are swallowed. Cancellation of the calling Task, on the other
    hand, is deferred until the worker finishes no matter how many times it
    arrives, and re-raised only afterwards."""

    wrapped = asyncio.wrap_future(future)
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(wrapped)
        except asyncio.CancelledError as exc:
            if wrapped.cancelled():
                break
            # A second cancellation must not let the generator's outer
            # finally release STATE.lock while the synchronous worker still
            # mutates its session.  Defer it until the worker is terminal.
            cancelled = exc
            continue
        except Exception:
            break
        else:
            break
    if cancelled is not None:
        raise cancelled


def _collect_events(
    prompt_ids: list[int],
    tokens: list[int],
    budget: int | None,
    tool_calling_enabled: bool,
    tools_for_parsing,
) -> tuple[list[tuple[str, object]], bool]:
    """For the non-streaming case: run the final token sequence through
    ThinkingRouter + SegmentAssembler in one go and return the order-preserving
    high-level event stream (the same vocabulary as ``feed()``/``push()``:
    reasoning_delta/content_delta/tool_call) together with budget_exceeded.
    This is the very same conversion that the streaming path
    (_start_generation) performs incrementally on every on_tokens, just done in
    one batch after generation for the non-streaming case.

    ``prompt_ids`` is the actual prompt rendered by ``_apply_template`` —
    ``_prompt_already_thinking`` uses it to decide whether the template has
    already opened ``<think>``, and that is reflected into ``ThinkingRouter``'s
    initial phase (see the ``_prompt_already_thinking`` docstring in
    server.py)."""

    router = ThinkingRouter(
        STATE.tokenizer,
        budget,
        STATE.eos_ids,
        tool_calling_enabled=tool_calling_enabled,
        already_thinking=_prompt_already_thinking(prompt_ids),
    )
    parts = router.feed(tokens) + router.finalize()
    assembler = SegmentAssembler(tools_for_parsing)
    events: list[tuple[str, object]] = []
    for ch, payload in parts:
        events.extend(assembler.push(ch, payload))
    return events, router.budget_exceeded


def _split_response_final(
    prompt_ids: list[int],
    tokens: list[int],
    budget: int | None,
    tool_calling_enabled: bool = False,
    tools_for_parsing=None,
) -> tuple[str, str, list[dict], bool]:
    """For the OpenAI non-streaming case: return (reasoning_text,
    content_text, tool_calls, budget_exceeded). tool_calls contains only the
    calls that were parsed successfully (those that failed to parse are
    included on the content_text side as raw text, markers and all — the
    policy of not fabricating anything). ``prompt_ids`` is passed straight
    through to ``_collect_events`` (for the already_thinking decision)."""

    events, budget_exceeded = _collect_events(
        prompt_ids, tokens, budget, tool_calling_enabled, tools_for_parsing
    )
    reasoning_text = "".join(v for k, v in events if k == "reasoning_delta")
    content_text = "".join(v for k, v in events if k == "content_delta")
    tool_calls = [v for k, v in events if k == "tool_call"]
    return reasoning_text, content_text, tool_calls, budget_exceeded


def _truncate_content_events(
    events: list[tuple[str, object]], cut_pos: int
) -> list[tuple[str, object]]:
    """For the Anthropic non-streaming case: when a stop_sequence matched at
    ``cut_pos`` in the concatenation of the content_delta strings, truncate
    every event from there on. This function is not called when tool_calls are
    involved (the caller's branching guarantees that — the combination of
    stop_sequence and tool calling is out of scope, a known limitation), so it
    is safe to assume that only reasoning_delta/content_delta arrive here."""

    out: list[tuple[str, object]] = []
    acc = 0
    for k, v in events:
        if k != "content_delta":
            out.append((k, v))
            continue
        if acc >= cut_pos:
            break
        remaining = cut_pos - acc
        if len(v) <= remaining:
            out.append((k, v))
            acc += len(v)
        else:
            if remaining > 0:
                out.append((k, v[:remaining]))
            break
    return out


def _anthropic_blocks_from_events(events: list[tuple[str, object]]) -> list[dict]:
    """Assemble the order-preserving high-level event stream into Anthropic's
    content block list. Consecutive content_deltas are merged into a single
    text block, and each tool_call becomes an independent tool_use block (in
    Anthropic's real API too, interposing a tool_use splits the text blocks).
    reasoning_delta is not handled here (the caller, as before, always
    assembles the thinking block separately and puts it first)."""

    blocks: list[dict] = []

    def push_text(t: str) -> None:
        if not t:
            return
        if blocks and blocks[-1]["type"] == "text":
            blocks[-1]["text"] += t
        else:
            blocks.append({"type": "text", "text": t})

    for kind, val in events:
        if kind == "content_delta":
            push_text(val)
        elif kind == "tool_call":
            blocks.append(
                {"type": "tool_use", "id": val["id"], "name": val["name"], "input": val["arguments"]}
            )
    return blocks


# ---------- OpenAI-compatible ----------


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": STATE.model_name,
                "object": "model",
                "created": STATE.created_ts,
                "owned_by": "mlxturbo",
            }
        ],
        "version": STATE.version,
    }


@app.get("/health")
async def health():
    """Returns more than mlx_lm's /health (which is just ``{"status": "ok"}``):
    the model name, whether it is loaded, which runner (speculative or ordinary
    generation), whether a request is being processed, the queue depth, and the
    version. Whether it is processing is simply ``STATE.lock.locked()`` (given
    the serialized design, a free lock is equivalent to idle and a held lock to
    busy).

    ``fallback_reason`` is added only when the runner is the fallback
    (non-speculative) one (mlxturbo.runner.build_runner attaches the reason
    string to the runner — it exists to prevent "silently falling back with no
    way to notice"). When it is not the fallback, the key itself is omitted.
    """

    if STATE is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "loaded": False, "version": _FASTMLX_VERSION},
        )
    body = {
        "status": "ok",
        "model": STATE.model_name,
        "loaded": True,
        "runner": getattr(STATE.runner, "KIND", type(STATE.runner).__name__),
        "busy": STATE.lock.locked(),
        "queue_depth": STATE.queue_depth,
        "version": STATE.version,
    }
    fallback_reason = getattr(STATE.runner, "fallback_reason", None)
    if fallback_reason is not None:
        body["fallback_reason"] = fallback_reason
    return body


def _read_rss_bytes() -> int | None:
    """Current (not peak) RSS of this process, in bytes.

    ``getrusage`` only reports the high-water mark, so it can't answer "how
    much is resident right now". macOS-only, so ``ps -o rss=`` (KiB there) is
    enough. Best-effort: None rather than failing the request.
    """

    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return int(out.stdout.strip()) * 1024
    except Exception:
        return None


@app.get("/api/status")
async def api_status():
    """Polling target for the menu-bar app in app/.

    Separate from /health so that endpoint's shape and callers stay untouched.
    """

    if STATE is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "loaded": False},
        )
    return {
        "model_name": STATE.model_name,
        "model_path": STATE.model_path,
        "runner_kind": getattr(STATE.runner, "KIND", type(STATE.runner).__name__),
        "fallback_reason": getattr(STATE.runner, "fallback_reason", None),
        "rss_bytes": _read_rss_bytes(),
        "peak_memory_bytes": mx.get_peak_memory(),
        "uptime_s": time.time() - STATE.created_ts,
        "n_sessions": len(STATE.session_pool),
        "queue_depth": STATE.queue_depth,
        "max_context_tokens": STATE.max_context_tokens,
    }


@app.get("/api/hello")
@app.head("/api/hello")
async def api_hello():
    """For Claude Code's connectivity check (GET/HEAD /api/hello). Without an
    implementation it 404s and, although there is no real harm (the
    conversation itself still works), it puts noise in the log right after
    startup. The body means nothing; the only requirement is to return 200."""

    return {"status": "ok"}


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Kimi K3 review item 14: state explicitly with a 501 that this is not
    supported.

    The conclusion after investigating: not "impossible", but the judgment that
    "there is no guarantee it can be implemented correctly in this server's
    current configuration, so do not force it". There are two reasons:

    1. What this server loads is a generation model (Qwen3.6-35B-A3B /
       Flash-Next family), not a dedicated embedding model. Techniques that
       pool a generation model's final hidden state (just before lm_head) and
       use it in place of an embedding do exist (LLM2Vec, GritLM, and so on),
       but they are dedicated implementations that involve selecting and
       validating a pooling scheme suited to the target model; quality cannot
       be guaranteed by merely "forwarding and pooling and calling it a day".

    2. The forward call needed to extract hidden states depends on the model's
       wrapper structure (``model.language_model.model`` in the VLM format,
       cache construction for a GDN hybrid, and so on) — an implementation that
       correctly accounts for that structure is territory that SpecEngine in
       mlxturbo/spec.py already covers, and bringing in a separate
       implementation of it under this pass's constraint of touching only
       server.py would mean exposing it to review carrying the risk of
       diverging from spec.py's implementation and with no way to validate it.

    Therefore, rather than silently returning some (wrong) embedding, we chose
    to say so explicitly with a 501. If real embeddings are needed, the sound
    arrangement should be to load a dedicated embedding model (mlx-embeddings
    or similar) separately and run both, but that is a change touching the very
    design of this server's model management (one resident model, pinned to a
    dedicated thread) and goes beyond the scope of this change.
    """

    return _openai_error(
        "'/v1/embeddings' is not implemented: this server only loads a generation "
        "model (not a dedicated embedding model), and deriving embeddings from its "
        "hidden states would require architecture-specific pooling logic this pass "
        "did not implement or validate. Run a dedicated embedding server for this.",
        status=501,
        err_type="server_error",
        code="not_implemented",
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _openai_error("request body must be valid JSON")
    if not isinstance(body, dict):
        return _openai_error("request body must be a JSON object")

    model_err = _check_model_openai(body)
    if model_err is not None:
        return model_err

    messages = body.get("messages")
    if not messages:
        return _openai_error("'messages' is required")
    shape_err = _validate_messages_shape(messages)
    if shape_err is not None:
        return _openai_error(shape_err)

    try:
        norm_messages = _normalize_openai_messages(messages)
    except ContentNormalizationError as exc:
        return _openai_error(str(exc))

    resolved_tools, tool_err = _resolve_tool_choice_openai(body)
    if tool_err is not None:
        return _openai_error(tool_err)
    unsupported_tools_err = _check_tool_calling_support(resolved_tools)
    if unsupported_tools_err is not None:
        return _openai_error(unsupported_tools_err)
    tool_enabled = resolved_tools is not None

    enable_thinking, thinking_budget, err = _resolve_thinking(body, "openai")
    if err is not None:
        return _openai_error(err)
    try:
        prompt_ids = _apply_template(norm_messages, enable_thinking, tools=resolved_tools)
    except Exception as exc:
        return _openai_error(f"failed to render chat template: {exc}")
    ctx_err = _check_context_length(prompt_ids, "openai")
    if ctx_err is not None:
        return ctx_err

    max_tokens, err = _resolve_max_tokens_openai(body, STATE.max_tokens_cap)
    if err is not None:
        return _openai_error(err)
    if thinking_budget is not None:
        thinking_budget = min(thinking_budget, max_tokens)
    temp, err = _parse_temperature(body, STATE.default_temp)
    if err is not None:
        return _openai_error(err)
    sampling_params, err = _parse_sampling_params(body)
    if err is not None:
        return _openai_error(err)
    response_format_err = _check_response_format(body)
    if response_format_err is not None:
        return _openai_error(response_format_err)
    logprobs_requested, top_logprobs_n, lp_err = _parse_logprobs_openai_chat(body)
    if lp_err is not None:
        return _openai_error(lp_err)
    stream = bool(body.get("stream", False))
    if logprobs_requested and stream:
        # Item 17 (Kimi K3 review): logprobs is not implemented for streaming
        # (correlating per-chunk incremental logprobs token by token with
        # ThinkingRouter/SegmentAssembler's channel branching and buffering is
        # difficult, and it was deferred as out of scope for this task —
        # refusing explicitly with a 400 is more honest than silently returning
        # null, which would amount to "you asked and got nothing back").
        # Non-streaming requests are handled normally below.
        return _openai_error(
            "'logprobs' is not supported together with 'stream': true in this server "
            "(yet); request without streaming to get logprobs"
        )
    gen_runner, downgrade_reason, unsupported_err = _resolve_runner_for_request(
        sampling_params, logprobs_requested=logprobs_requested
    )
    if unsupported_err is not None:
        return _openai_error(unsupported_err)
    if logprobs_requested and gen_runner.KIND == FallbackRunner.KIND:
        # logprobs is a separate flag outside the SUPPORTED_SAMPLING_PARAMS
        # identity-value check (the logprobs_requested argument of
        # _resolve_runner_for_request), so it is only added to sampling_params
        # here. It is added only when it was requested and the runner that was
        # actually resolved is FallbackRunner (non-speculative) —
        # SpecRunner/FlashSpecRunner/DraftSpecRunner/LookupSpecRunner may have
        # a generate() that does not know this keyword (SpecEngine.generate/
        # generate_stream have a fixed signature with no **kwargs). A request
        # asking for logprobs is always downgraded to the fallback by
        # _resolve_runner_for_request (and only when there is nowhere to
        # downgrade to is STATE.runner itself already the fallback), so the
        # runner arriving here should always be FallbackRunner; even so, we
        # double-check.
        sampling_params["logprobs"] = True
        sampling_params["top_logprobs"] = top_logprobs_n
    stops = _stop_sequences(body)

    model_id = STATE.model_name
    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    include_usage = False
    if stream:
        stream_options = body.get("stream_options")
        if stream_options is None:
            stream_options = {}
        elif not isinstance(stream_options, dict):
            return _openai_error("'stream_options' must be a JSON object")
        include_usage = bool(stream_options.get("include_usage", False))

    if not _try_reserve_queue_slot():
        return _busy_response("openai", "server is busy: too many queued requests")

    if stream:
        # _openai_stream's finally releases this request's queue slot —
        # StreamingResponse is guaranteed to read the generator to the end or
        # aclose it, so releasing here is unnecessary (and would in fact be a
        # double release).
        return StreamingResponse(
            _queue_owned_stream(
                _openai_stream(
                    prompt_ids,
                    max_tokens,
                    temp,
                    req_id,
                    created,
                    model_id,
                    stops,
                    include_usage,
                    thinking_budget,
                    sampling_params,
                    tool_enabled,
                    resolved_tools,
                    runner=gen_runner,
                )
            ),
            media_type="text/event-stream",
            headers=_downgrade_headers(downgrade_reason),
        )

    batch_tier = _resolve_batch_tier(
        gen_runner, prompt_ids, max_tokens, logprobs_requested=logprobs_requested
    )
    try:
        if batch_tier is not None:
            # Continuous batching (--max-batch): no STATE.lock, no session
            # (see mlxturbo/batch.py's module docstring) — concurrency with
            # other in-flight requests is the entire point.
            res = await _run_generate_batched(
                prompt_ids, max_tokens, temp, None, **sampling_params
            )
        else:
            async with STATE.lock:
                # A downgraded request does not touch the session pool:
                # FallbackRunner expects a FallbackSession (singular .cache), but
                # if STATE.runner is a spec runner the pool holds ChatSessions
                # (plural .caches) and the types do not match. Passing session=None
                # makes FallbackRunner.generate fall naturally onto the path where
                # it simply builds a fresh prompt_cache every time (see the
                # _resolve_runner_for_request docstring).
                session = None if downgrade_reason is not None else await _select_session_on_executor(
                    prompt_ids
                )
                res = await _run_generate(
                    prompt_ids,
                    max_tokens,
                    temp,
                    STATE.eos_ids,
                    None,
                    session,
                    runner=gen_runner,
                    **sampling_params,
                )
    except Exception as exc:
        return _openai_error(str(exc), status=500, err_type="server_error")
    finally:
        _release_queue_slot()
    _log_gen_stats(res)

    reasoning_text, content_text, tool_calls, budget_exceeded = _split_response_final(
        prompt_ids, res["tokens"], thinking_budget, tool_enabled, resolved_tools
    )
    finish_reason = _finish_reason_openai(res["tokens"])
    content_truncated_by_stop = False
    if budget_exceeded:
        finish_reason = "length"
    elif tool_calls:
        finish_reason = "tool_calls"
    elif stops:
        hit = _find_stop(content_text, stops)
        if hit is not None:
            content_text = content_text[: hit[0]]
            finish_reason = "stop"
            content_truncated_by_stop = True

    # res["logprobs"] (from FallbackRunner.generate; see mlxturbo/runner.py)
    # corresponds 1:1 with the raw generated token sequence res["tokens"]. When
    # reasoning/tool_calls are involved, or the content was truncated at a
    # character position by a stop string, correctly re-deriving "which raw
    # tokens actually remained in the content" would require a change giving
    # _split_response_final token-level split information, which goes beyond
    # this task's scope (the judgment being that null is more honest than
    # returning fabricated data — the reason for deferring it is recorded
    # here). Otherwise (a plain generation with no reasoning, no tool_calls and
    # no stop truncation) res["logprobs"] corresponds 1:1 with every token of
    # the content as-is, so it is used directly.
    choice_logprobs = None
    if logprobs_requested and not (reasoning_text or tool_calls or content_truncated_by_stop):
        choice_logprobs = _build_chat_logprobs(res.get("logprobs"))

    message = {"role": "assistant", "content": content_text}
    if reasoning_text:
        message["reasoning_content"] = reasoning_text
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                },
            }
            for tc in tool_calls
        ]
        if not content_text:
            message["content"] = None

    out = {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": choice_logprobs,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage_dict(
            len(prompt_ids), len(res["tokens"]), res.get("prefill_reused", 0)
        ),
    }
    _attach_downgrade_reason(out, downgrade_reason)
    return out


async def _openai_stream(
    prompt_ids,
    max_tokens,
    temp,
    req_id,
    created,
    model_id,
    stops,
    include_usage,
    thinking_budget,
    sampling_params: dict | None = None,
    tool_enabled: bool = False,
    tools_for_parsing=None,
    runner=None,
):
    # Acceptance (validation) has already completed on the caller's side,
    # before the StreamingResponse was assembled. Emitting the first event
    # without waiting for generation to start (lock acquisition, worker
    # submission) returns "the first byte" quickly even in workloads where TTFT
    # dominates (see the docstring in server.py). Beyond this point there is no
    # path on which something 400-worthy could still be discovered.
    first = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    if include_usage:
        first["usage"] = None
    yield f"data: {json.dumps(first)}\n\n"

    # Keep sending keepalives while waiting for the lock too (when the queue is
    # backed up, this can effectively account for most of the "time until the
    # first token"). Ownership of the queue slot is centralized in
    # _queue_owned_stream, and ownership of the lock in the owned flag below.
    owned = [False]
    batch_tier = _resolve_batch_tier(runner or STATE.runner, prompt_ids, max_tokens)
    try:
        if batch_tier is not None:
            # Continuous batching (--max-batch): no STATE.lock (owned stays
            # False, so the function's own final `if owned[0]:` is already a
            # correct no-op), no session (see mlxturbo/batch.py's module
            # docstring — batched requests always do a fresh prefill).
            session = None
            reused_at_select = 0
            q, future, cancel_event, raw_token_count = _start_batched_generation(
                prompt_ids,
                max_tokens,
                temp,
                thinking_budget,
                tool_enabled,
                tools_for_parsing,
                **(sampling_params or {}),
            )
        else:
            async for keepalive in _acquire_lock_with_keepalive(STATE.lock, owned):
                yield keepalive

            # A downgraded request (whose runner differs from STATE.runner =
            # STATE.downgrade_runner) does not touch the session pool — the types
            # do not match (see the _resolve_runner_for_request docstring).
            downgraded = runner is not None and runner is not STATE.runner
            session = None if downgraded else await _select_session_on_executor(prompt_ids)
            # By the time ``_select_session`` has chosen a slot it has already
            # truncated the processed sequence to exactly "the length that will
            # actually be reused" (unchanged for a whole-sequence match, the
            # post-trim/post-checkpoint-restore length for a partial match), so the
            # value read here matches the prefill_reused that runner.generate()
            # computes internally. When a stop string matches and generation is cut
            # off early (cancel_event.set(), item 18), the runner does not return a
            # res dict ("cancelled"), so usage's cached_tokens falls back to this
            # precomputed value.
            reused_at_select = len(session.processed) if session is not None else 0
            q, future, cancel_event, raw_token_count = _start_generation(
                prompt_ids,
                max_tokens,
                temp,
                thinking_budget,
                tool_enabled,
                tools_for_parsing,
                session=session,
                runner=runner,
                **(sampling_params or {}),
            )
        try:
            # Starting generation (submitting the worker) finishes immediately,
            # but keepalives are likewise sent until the first queue element
            # arrives (= during prefill). The first item that was peeked at is
            # put back at the head of q with _requeue_front, so that the while
            # loop below processes it again on the same path without being
            # changed at all.
            first_item = None
            async for _ka_kind, _ka_val in _await_with_keepalive(asyncio.to_thread(q.get)):
                if _ka_kind == "keepalive":
                    yield _ka_val
                else:
                    first_item = _ka_val
            _requeue_front(q, first_item)

            finish_reason = "length"
            n_completion = 0
            cached_tokens = 0
            acc_text = ""
            stopped = False  # Once a stop string matches, only the forwarding
            # to the client stops; the queue keeps being drained until
            # generation actually ends ("done") (to get accurate usage and to
            # not break the premise of waiting for the Future). budget_exceeded
            # takes the same "stop only the forwarding" form.
            budget_exceeded = False
            made_tool_call = False
            tool_call_index = 0
            while True:
                kind, payload = await asyncio.to_thread(q.get)
                if kind == "tool_call":
                    if stopped:
                        continue
                    made_tool_call = True
                    idx = tool_call_index
                    tool_call_index += 1
                    name_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": idx,
                                            "id": payload["id"],
                                            "type": "function",
                                            "function": {"name": payload["name"], "arguments": ""},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    if include_usage:
                        name_chunk["usage"] = None
                    yield f"data: {json.dumps(name_chunk)}\n\n"
                    args_str = json.dumps(payload["arguments"], ensure_ascii=False)
                    for piece in _chunk_string(args_str):
                        args_chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [{"index": idx, "function": {"arguments": piece}}]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        if include_usage:
                            args_chunk["usage"] = None
                        yield f"data: {json.dumps(args_chunk)}\n\n"
                elif kind == "reasoning_delta":
                    if stopped or payload == "":
                        continue
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": payload},
                                "finish_reason": None,
                            }
                        ],
                    }
                    if include_usage:
                        chunk["usage"] = None
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif kind == "content_delta":
                    if stopped:
                        continue
                    visible = payload
                    if stops:
                        new_acc = acc_text + payload
                        hit = _find_stop(new_acc, stops)
                        if hit is not None:
                            idx, _matched = hit
                            keep_len = idx - len(acc_text)
                            visible = payload[:keep_len] if keep_len > 0 else ""
                            stopped = True
                            # Item 18: cut generation itself off the instant it
                            # matches. Previously only the forwarding stopped
                            # here, and the actual generation kept running in
                            # the background all the way to max_tokens (wasted
                            # decoding). The computation of visible (= the
                            # string shown to the client) has already settled
                            # above this point, so cutting off does not change
                            # the output content — the only difference is the
                            # speed gained by skipping the remaining decoding.
                            # The session is not published afterwards (the next
                            # turn gets a fresh prefill, per cancel_event's
                            # existing contract).
                            cancel_event.set()
                        acc_text = new_acc
                    if visible:
                        chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [
                                {"index": 0, "delta": {"content": visible}, "finish_reason": None}
                            ],
                        }
                        if include_usage:
                            chunk["usage"] = None
                        yield f"data: {json.dumps(chunk)}\n\n"
                elif kind == "budget_exceeded":
                    budget_exceeded = True
                elif kind == "done":
                    if budget_exceeded:
                        finish_reason = "length"
                    elif made_tool_call:
                        finish_reason = "tool_calls"
                    else:
                        finish_reason = "stop" if stopped else _finish_reason_openai(payload["tokens"])
                    n_completion = len(payload["tokens"])
                    cached_tokens = payload.get("prefill_reused", 0)
                    _log_gen_stats(payload)
                    break
                elif kind == "cancelled":
                    # Only the early cut-off caused by a stop string match (the
                    # result of cancel_event.set() in the content_delta branch
                    # above) reaches here — a client disconnect aborts this
                    # loop itself through the async generator's GeneratorExit
                    # and never gets here. The runner returns no res dict, so
                    # usage falls back to the raw_token_count that on_tokens
                    # counted and the reuse figure from when the session was
                    # selected.
                    finish_reason = "stop"
                    n_completion = raw_token_count[0]
                    cached_tokens = reused_at_select
                    print(
                        f"[mlxturbo-serve] stop string matched — cancelled early after "
                        f"{n_completion} decoded tokens (prefill reused={cached_tokens})"
                    )
                    break
                else:  # error
                    err = {
                        "error": {
                            "message": str(payload),
                            "type": "server_error",
                            "param": None,
                            "code": "server_error",
                        }
                    }
                    yield f"data: {json.dumps(err)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            final_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
            if include_usage:
                final_chunk["usage"] = None
            yield f"data: {json.dumps(final_chunk)}\n\n"

            if include_usage:
                usage_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [],
                    "usage": _usage_dict(len(prompt_ids), n_completion, cached_tokens),
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"

            yield "data: [DONE]\n\n"
        finally:
            # Including the case where we arrive here through a client
            # disconnect (GeneratorExit), stop cooperatively at the next token
            # callback and do not release the lock until the worker has
            # actually finished (prefill and an in-flight kernel cannot be
            # interrupted).
            cancel_event.set()
            await _await_worker(future)
    finally:
        if owned[0]:
            STATE.lock.release()


# ---------- Anthropic-compatible ----------


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _anthropic_error("request body must be valid JSON")
    if not isinstance(body, dict):
        return _anthropic_error("request body must be a JSON object")

    model_err = _check_model_anthropic(body)
    if model_err is not None:
        return model_err

    if "max_tokens" not in body:
        return _anthropic_error("'max_tokens' is required")
    messages = body.get("messages")
    if not messages:
        return _anthropic_error("'messages' is required")
    shape_err = _validate_messages_shape(messages)
    if shape_err is not None:
        return _anthropic_error(shape_err)

    try:
        # Some clients send role: "system" items mixed into ``messages`` in
        # addition to the top-level ``system`` (a real example: Claude Code —
        # in a captured body the order was ['user', 'system'], with system
        # last. Anthropic's published specification does not permit
        # role: "system" inside messages, but real clients send it routinely).
        # The chat template (Qwen family) dies with "System message must be at
        # the beginning" unless the system message comes first, so both the
        # top-level system and the system roles inside messages are gathered to
        # the front and concatenated into a single system message. The
        # concatenation order preserves the original order, "top level -> the
        # order they appeared in messages" (neither is disturbed). Messages
        # other than system keep their original relative order.
        normalized = _normalize_anthropic_messages(messages)
        system_parts: list[str] = []
        top_system = body.get("system")
        if top_system:
            system_parts.append(_content_to_text(top_system))
        norm_messages = []
        for m in normalized:
            if m.get("role") == "system":
                text = m.get("content") or ""
                if text:
                    system_parts.append(text)
                continue
            norm_messages.append(m)
        if system_parts:
            norm_messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    except ContentNormalizationError as exc:
        return _anthropic_error(str(exc))

    resolved_tools, tool_err = _resolve_tool_choice_anthropic(body)
    if tool_err is not None:
        return _anthropic_error(tool_err)
    unsupported_tools_err = _check_tool_calling_support(resolved_tools)
    if unsupported_tools_err is not None:
        return _anthropic_error(unsupported_tools_err)
    tool_enabled = resolved_tools is not None

    enable_thinking, thinking_budget, err = _resolve_thinking(body, "anthropic")
    if err is not None:
        return _anthropic_error(err)
    try:
        prompt_ids = _apply_template(norm_messages, enable_thinking, tools=resolved_tools)
    except Exception as exc:
        return _anthropic_error(f"failed to render chat template: {exc}")
    ctx_err = _check_context_length(prompt_ids, "anthropic")
    if ctx_err is not None:
        return ctx_err

    max_tokens, err = _parse_anthropic_max_tokens(body["max_tokens"], STATE.max_tokens_cap)
    if err is not None:
        return _anthropic_error(err)
    if thinking_budget is not None:
        thinking_budget = min(thinking_budget, max_tokens)
    temp, err = _parse_temperature(body, STATE.default_temp)
    if err is not None:
        return _anthropic_error(err)
    sampling_params, err = _parse_sampling_params(body)
    if err is not None:
        return _anthropic_error(err)
    gen_runner, downgrade_reason, unsupported_err = _resolve_runner_for_request(sampling_params)
    if unsupported_err is not None:
        return _anthropic_error(unsupported_err)
    stops = _stop_sequences(body)

    model_id = STATE.model_name
    stream = bool(body.get("stream", False))
    msg_id = f"msg_{uuid.uuid4().hex}"

    if max_tokens == 0:
        # Aligned with Anthropic's real API: 0 is valid input meaning "end
        # without generating", but it makes no sense for a stream, so return a
        # 400.
        if stream:
            return _anthropic_error("'max_tokens: 0' is not supported when 'stream' is true")
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model_id,
            "content": [],
            "stop_reason": "max_tokens",
            "stop_sequence": None,
            "usage": {"input_tokens": len(prompt_ids), "output_tokens": 0},
        }

    if not _try_reserve_queue_slot():
        return _busy_response("anthropic", "server is busy: too many queued requests")

    if stream:
        # _anthropic_stream's finally releases this request's queue slot (the
        # same reason as chat_completions on the openai path).
        return StreamingResponse(
            _queue_owned_stream(
                _anthropic_stream(
                    prompt_ids,
                    max_tokens,
                    temp,
                    msg_id,
                    model_id,
                    stops,
                    thinking_budget,
                    sampling_params,
                    tool_enabled,
                    resolved_tools,
                    runner=gen_runner,
                )
            ),
            media_type="text/event-stream",
            headers=_downgrade_headers(downgrade_reason),
        )

    batch_tier = _resolve_batch_tier(gen_runner, prompt_ids, max_tokens)
    try:
        if batch_tier is not None:
            res = await _run_generate_batched(
                prompt_ids, max_tokens, temp, None, **sampling_params
            )
        else:
            async with STATE.lock:
                # A downgraded request does not touch the session pool (the same
                # reason as chat_completions; see the _resolve_runner_for_request
                # docstring).
                session = None if downgrade_reason is not None else await _select_session_on_executor(
                    prompt_ids
                )
                res = await _run_generate(
                    prompt_ids,
                    max_tokens,
                    temp,
                    STATE.eos_ids,
                    None,
                    session,
                    runner=gen_runner,
                    **sampling_params,
                )
    except Exception as exc:
        return _anthropic_error(str(exc), status=500, err_type="server_error")
    finally:
        _release_queue_slot()
    _log_gen_stats(res)

    events, budget_exceeded = _collect_events(
        prompt_ids, res["tokens"], thinking_budget, tool_enabled, resolved_tools
    )
    reasoning_text = "".join(v for k, v in events if k == "reasoning_delta")
    tool_calls = [v for k, v in events if k == "tool_call"]
    stop_reason = _stop_reason_anthropic(res["tokens"])
    matched_stop = None
    if budget_exceeded:
        stop_reason = "max_tokens"
    elif tool_calls:
        stop_reason = "tool_use"
    elif stops:
        # stop_sequence support when tool_calls are involved is deferred as a
        # known limitation (never reached, thanks to the elif) — see the
        # _truncate_content_events docstring for details.
        content_text_concat = "".join(v for k, v in events if k == "content_delta")
        hit = _find_stop(content_text_concat, stops)
        if hit is not None:
            events = _truncate_content_events(events, hit[0])
            stop_reason = "stop_sequence"
            matched_stop = hit[1]

    content_blocks = []
    if reasoning_text:
        content_blocks.append(
            {
                "type": "thinking",
                "thinking": reasoning_text,
                "signature": _thinking_signature(reasoning_text),
            }
        )
    content_blocks.extend(_anthropic_blocks_from_events(events))
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    out = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": matched_stop,
        "usage": _anthropic_usage(len(prompt_ids), len(res["tokens"]), res),
    }
    _attach_downgrade_reason(out, downgrade_reason)
    return out


async def _anthropic_stream(
    prompt_ids,
    max_tokens,
    temp,
    msg_id,
    model_id,
    stops,
    thinking_budget,
    sampling_params: dict | None = None,
    tool_enabled: bool = False,
    tools_for_parsing=None,
    runner=None,
):
    def sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    # Acceptance (validation) has already completed on the caller's side,
    # before the StreamingResponse was assembled. Emitting message_start
    # without waiting for generation to start (lock acquisition, worker
    # submission) returns "the first byte" quickly even in workloads where TTFT
    # dominates (see the docstring in server.py). Beyond this point there is no
    # path on which something 400-worthy could still be discovered.
    yield sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model_id,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": len(prompt_ids), "output_tokens": 0},
            },
        },
    )
    yield sse("ping", {"type": "ping"})

    owned = [False]
    batch_tier = _resolve_batch_tier(runner or STATE.runner, prompt_ids, max_tokens)
    try:
        if batch_tier is not None:
            session = None
            reused_at_select = 0
            q, future, cancel_event, raw_token_count = _start_batched_generation(
                prompt_ids,
                max_tokens,
                temp,
                thinking_budget,
                tool_enabled,
                tools_for_parsing,
                **(sampling_params or {}),
            )
        else:
            async for keepalive in _acquire_lock_with_keepalive(STATE.lock, owned):
                yield keepalive

            downgraded = runner is not None and runner is not STATE.runner
            session = None if downgraded else await _select_session_on_executor(prompt_ids)
            reused_at_select = len(session.processed) if session is not None else 0
            q, future, cancel_event, raw_token_count = _start_generation(
                prompt_ids,
                max_tokens,
                temp,
                thinking_budget,
                tool_enabled,
                tools_for_parsing,
                session=session,
                runner=runner,
                **(sampling_params or {}),
            )
        try:
            first_item = None
            async for _ka_kind, _ka_val in _await_with_keepalive(asyncio.to_thread(q.get)):
                if _ka_kind == "keepalive":
                    yield _ka_val
                else:
                    first_item = _ka_val
            _requeue_front(q, first_item)

            # content_block_start is emitted only after the first delta
            # arrives, according to its channel (whether thinking happened),
            # because it cannot be known in advance (the real API also emits
            # content_block_start immediately before sending the first block,
            # so the ordering is the same).

            n_out = 0
            prefill_reused = 0
            prefill_new = 0
            stop_reason = "end_turn"
            matched_stop = None
            acc_text = ""
            stopped = False
            budget_exceeded = False
            made_tool_call = False
            current_block: str | None = None  # None | "reasoning" | "content"
            any_block_emitted = False  # current_block alone cannot distinguish
            # "finished emitting a tool_use and reset to None" from "nothing
            # has been emitted yet", so this is kept as a separate flag.
            next_index = 0
            block_index: dict[str, int] = {}
            thinking_text_by_index: dict[int, str] = {}

            def open_block(channel: str):
                nonlocal next_index
                idx = next_index
                next_index += 1
                block_index[channel] = idx
                if channel == "reasoning":
                    thinking_text_by_index[idx] = ""
                    content_block = {"type": "thinking", "thinking": "", "signature": ""}
                else:
                    content_block = {"type": "text", "text": ""}
                return idx, sse(
                    "content_block_start",
                    {"type": "content_block_start", "index": idx, "content_block": content_block},
                )

            def close_block(channel: str) -> list[str]:
                idx = block_index[channel]
                events = []
                if channel == "reasoning":
                    events.append(
                        sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {
                                    "type": "signature_delta",
                                    "signature": _thinking_signature(
                                        thinking_text_by_index.get(idx, "")
                                    ),
                                },
                            },
                        )
                    )
                events.append(
                    sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": idx},
                    )
                )
                return events

            while True:
                kind, payload = await asyncio.to_thread(q.get)
                if kind == "tool_call":
                    if stopped:
                        continue
                    made_tool_call = True
                    any_block_emitted = True
                    if current_block is not None:
                        for event in close_block(current_block):
                            yield event
                        current_block = None
                    idx = next_index
                    next_index += 1
                    yield sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": payload["id"],
                                "name": payload["name"],
                                "input": {},
                            },
                        },
                    )
                    args_str = json.dumps(payload["arguments"], ensure_ascii=False)
                    for piece in _chunk_string(args_str):
                        yield sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "input_json_delta", "partial_json": piece},
                            },
                        )
                    yield sse("content_block_stop", {"type": "content_block_stop", "index": idx})
                elif kind in ("reasoning_delta", "content_delta"):
                    if stopped:
                        continue
                    channel = "reasoning" if kind == "reasoning_delta" else "content"
                    visible = payload
                    if channel == "content" and stops:
                        new_acc = acc_text + payload
                        hit = _find_stop(new_acc, stops)
                        if hit is not None:
                            idx, matched = hit
                            keep_len = idx - len(acc_text)
                            visible = payload[:keep_len] if keep_len > 0 else ""
                            stopped = True
                            matched_stop = matched
                            # Item 18: cut generation itself off the instant it
                            # matches (the same reason as the identical change
                            # in _openai_stream).
                            cancel_event.set()
                        acc_text = new_acc
                    if not visible:
                        continue
                    if current_block != channel:
                        if current_block is not None:
                            for event in close_block(current_block):
                                yield event
                        idx, start_evt = open_block(channel)
                        yield start_evt
                        current_block = channel
                        any_block_emitted = True
                    delta_field = (
                        {"type": "thinking_delta", "thinking": visible}
                        if channel == "reasoning"
                        else {"type": "text_delta", "text": visible}
                    )
                    if channel == "reasoning":
                        idx = block_index[channel]
                        thinking_text_by_index[idx] += visible
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_index[channel],
                            "delta": delta_field,
                        },
                    )
                elif kind == "budget_exceeded":
                    budget_exceeded = True
                elif kind == "done":
                    n_out = len(payload["tokens"])
                    prefill_reused = payload.get("prefill_reused", 0)
                    prefill_new = payload.get("prefill_new", 0)
                    if budget_exceeded:
                        stop_reason = "max_tokens"
                    elif made_tool_call:
                        stop_reason = "tool_use"
                    else:
                        stop_reason = (
                            "stop_sequence" if stopped else _stop_reason_anthropic(payload["tokens"])
                        )
                    _log_gen_stats(payload)
                    break
                elif kind == "cancelled":
                    # Early cut-off caused by a stop string match (the same
                    # reason and the same contract as the identical branch in
                    # _openai_stream).
                    n_out = raw_token_count[0]
                    prefill_reused = reused_at_select
                    prefill_new = max(len(prompt_ids) - reused_at_select, 0)
                    stop_reason = "stop_sequence"
                    print(
                        f"[mlxturbo-serve] stop string matched — cancelled early after "
                        f"{n_out} decoded tokens (prefill reused={prefill_reused})"
                    )
                    break
                else:  # error
                    yield sse(
                        "error",
                        {"type": "error", "error": {"type": "server_error", "message": str(payload)}},
                    )
                    return

            if current_block is not None:
                for event in close_block(current_block):
                    yield event
            elif not any_block_emitted:
                # Nothing was generated (e.g. max_tokens is extremely small).
                # So as not to break the premise that "there is at least one
                # content_block", open an empty text block and close it
                # immediately. The case where one or more tool_use blocks have
                # been emitted and current_block has merely returned to None
                # does not reach here (any_block_emitted tells them apart).
                idx, start_evt = open_block("content")
                yield start_evt
                yield sse("content_block_stop", {"type": "content_block_stop", "index": idx})

            yield sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": matched_stop},
                    # input_tokens stays the naive len(prompt_ids) that was
                    # streamed out immediately in message_start (before
                    # generation began), so it is not overwritten — here we add
                    # only cache_read/cache_creation_input_tokens, the measured
                    # cache figures (item 16) that are not known until after
                    # generation.
                    "usage": {
                        "output_tokens": n_out,
                        "cache_creation_input_tokens": prefill_new,
                        "cache_read_input_tokens": prefill_reused,
                    },
                },
            )
            yield sse("message_stop", {"type": "message_stop"})
        finally:
            cancel_event.set()
            await _await_worker(future)
    finally:
        if owned[0]:
            STATE.lock.release()


# ---------- OpenAI-compatible (legacy /v1/completions) ----------
#
# A legacy endpoint that does not go through the chat template and feeds the
# prompt it was given straight into generation. Thinking is not separated
# (there is no turn structure in a raw completion, so there is no way to give
# reasoning_effort any meaning) — passing thinking_budget=0 to
# _start_generation/_split_response_final forces ThinkingRouter into
# content-only mode (budget=0 means enabled=False regardless of the value of
# has_thinking) and reuses the existing on_tokens wiring as-is. Tool calling is
# likewise not enabled (tool_calling_enabled is simply not passed and stays at
# its default of False).


def _prompt_to_ids(prompt) -> tuple[list[int] | None, str | None]:
    """``prompt`` is accepted either as a string (run through
    tokenizer.encode) or as a pre-tokenized array of ints. OpenAI's legacy
    completions also permits an array of strings or an array of token arrays (a
    batch), but mlxturbo is designed to serialize requests (one request = one
    generation), so batches are not handled.
    """

    if isinstance(prompt, str):
        if not prompt:
            return None, "'prompt' must not be empty"
        return list(STATE.tokenizer.encode(prompt)), None
    if isinstance(prompt, list):
        if not prompt:
            return None, "'prompt' must not be empty"
        if all(isinstance(t, int) and not isinstance(t, bool) for t in prompt):
            return list(prompt), None
        return None, "'prompt' array must contain only integers (pre-tokenized ids)"
    return None, "'prompt' must be a string or an array of token ids"


@app.post("/v1/completions")
async def completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _openai_error("request body must be valid JSON")
    if not isinstance(body, dict):
        return _openai_error("request body must be a JSON object")

    model_err = _check_model_openai(body)
    if model_err is not None:
        return model_err

    if "prompt" not in body:
        return _openai_error("'prompt' is required")
    prompt_ids, err = _prompt_to_ids(body["prompt"])
    if err is not None:
        return _openai_error(err)
    ctx_err = _check_context_length(prompt_ids, "openai")
    if ctx_err is not None:
        return ctx_err

    max_tokens, err = _resolve_max_tokens_openai(body, STATE.max_tokens_cap)
    if err is not None:
        return _openai_error(err)
    temp, err = _parse_temperature(body, STATE.default_temp)
    if err is not None:
        return _openai_error(err)
    sampling_params, err = _parse_sampling_params(body)
    if err is not None:
        return _openai_error(err)
    logprobs_requested, top_logprobs_n, lp_err = _parse_logprobs_openai_legacy(body)
    if lp_err is not None:
        return _openai_error(lp_err)
    stream = bool(body.get("stream", False))
    if logprobs_requested and stream:
        # The same reason as chat_completions (see the item 17 docstring):
        # per-chunk logprobs support for streaming was deferred this time.
        return _openai_error(
            "'logprobs' is not supported together with 'stream': true in this server "
            "(yet); request without streaming to get logprobs"
        )
    gen_runner, downgrade_reason, unsupported_err = _resolve_runner_for_request(
        sampling_params, logprobs_requested=logprobs_requested
    )
    if unsupported_err is not None:
        return _openai_error(unsupported_err)
    if logprobs_requested and gen_runner.KIND == FallbackRunner.KIND:
        sampling_params["logprobs"] = True
        sampling_params["top_logprobs"] = top_logprobs_n
    stops = _stop_sequences(body)

    model_id = STATE.model_name
    req_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    include_usage = False
    if stream:
        stream_options = body.get("stream_options")
        if stream_options is None:
            stream_options = {}
        elif not isinstance(stream_options, dict):
            return _openai_error("'stream_options' must be a JSON object")
        include_usage = bool(stream_options.get("include_usage", False))

    if not _try_reserve_queue_slot():
        return _busy_response("openai", "server is busy: too many queued requests")

    if stream:
        # _completions_stream's finally releases this request's queue slot (the
        # same reason as the openai chat path).
        return StreamingResponse(
            _queue_owned_stream(
                _completions_stream(
                    prompt_ids,
                    max_tokens,
                    temp,
                    req_id,
                    created,
                    model_id,
                    stops,
                    include_usage,
                    sampling_params,
                    runner=gen_runner,
                )
            ),
            media_type="text/event-stream",
            headers=_downgrade_headers(downgrade_reason),
        )

    batch_tier = _resolve_batch_tier(
        gen_runner, prompt_ids, max_tokens, logprobs_requested=logprobs_requested
    )
    try:
        if batch_tier is not None:
            res = await _run_generate_batched(
                prompt_ids, max_tokens, temp, None, **sampling_params
            )
        else:
            async with STATE.lock:
                session = None if downgrade_reason is not None else await _select_session_on_executor(
                    prompt_ids
                )
                res = await _run_generate(
                    prompt_ids,
                    max_tokens,
                    temp,
                    STATE.eos_ids,
                    None,
                    session,
                    runner=gen_runner,
                    **sampling_params,
                )
    except Exception as exc:
        return _openai_error(str(exc), status=500, err_type="server_error")
    finally:
        _release_queue_slot()
    _log_gen_stats(res)

    reasoning_text, text, tool_calls, _budget_exceeded = _split_response_final(
        prompt_ids, res["tokens"], 0
    )
    finish_reason = _finish_reason_openai(res["tokens"])
    content_truncated_by_stop = False
    if stops:
        hit = _find_stop(text, stops)
        if hit is not None:
            text = text[: hit[0]]
            finish_reason = "stop"
            content_truncated_by_stop = True

    # The same limitation as chat_completions (see its docstring): if
    # reasoning/tool_call markers were mixed in, or a stop string truncated at
    # a character position, the correspondence between res["logprobs"] (1:1
    # with the raw token sequence) and text breaks down, so it is null.
    choice_logprobs = None
    if logprobs_requested and not (reasoning_text or tool_calls or content_truncated_by_stop):
        choice_logprobs = _build_legacy_logprobs(res.get("logprobs"))

    out = {
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": choice_logprobs,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage_dict(
            len(prompt_ids), len(res["tokens"]), res.get("prefill_reused", 0)
        ),
    }
    _attach_downgrade_reason(out, downgrade_reason)
    return out


async def _completions_stream(
    prompt_ids,
    max_tokens,
    temp,
    req_id,
    created,
    model_id,
    stops,
    include_usage,
    sampling_params: dict | None = None,
    runner=None,
):
    owned = [False]
    batch_tier = _resolve_batch_tier(runner or STATE.runner, prompt_ids, max_tokens)
    try:
        if batch_tier is not None:
            session = None
            reused_at_select = 0
            # thinking_budget=0 pins ThinkingRouter to content-only (regardless of
            # has_thinking), so reasoning_delta can never arrive.
            q, future, cancel_event, raw_token_count = _start_batched_generation(
                prompt_ids, max_tokens, temp, 0, **(sampling_params or {})
            )
        else:
            async for keepalive in _acquire_lock_with_keepalive(STATE.lock, owned):
                yield keepalive

            downgraded = runner is not None and runner is not STATE.runner
            session = None if downgraded else await _select_session_on_executor(prompt_ids)
            reused_at_select = len(session.processed) if session is not None else 0
            # thinking_budget=0 pins ThinkingRouter to content-only (regardless of
            # has_thinking), so reasoning_delta can never arrive.
            q, future, cancel_event, raw_token_count = _start_generation(
                prompt_ids, max_tokens, temp, 0, session=session, runner=runner, **(sampling_params or {})
            )
        try:
            first_item = None
            async for _ka_kind, _ka_val in _await_with_keepalive(asyncio.to_thread(q.get)):
                if _ka_kind == "keepalive":
                    yield _ka_val
                else:
                    first_item = _ka_val
            _requeue_front(q, first_item)

            finish_reason = "length"
            n_completion = 0
            cached_tokens = 0
            acc_text = ""
            stopped = False
            while True:
                kind, payload = await asyncio.to_thread(q.get)
                if kind == "content_delta":
                    if stopped:
                        continue
                    visible = payload
                    if stops:
                        new_acc = acc_text + payload
                        hit = _find_stop(new_acc, stops)
                        if hit is not None:
                            idx, _matched = hit
                            keep_len = idx - len(acc_text)
                            visible = payload[:keep_len] if keep_len > 0 else ""
                            stopped = True
                            cancel_event.set()  # Item 18: early cut-off
                        acc_text = new_acc
                    if visible:
                        chunk = {
                            "id": req_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": visible,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        if include_usage:
                            chunk["usage"] = None
                        yield f"data: {json.dumps(chunk)}\n\n"
                elif kind in ("reasoning_delta", "budget_exceeded", "tool_call"):
                    # With thinking_budget=0 and tool_calling_enabled=False
                    # (the default) these should never arrive here, but just in
                    # case we only ignore them (safer than crashing).
                    continue
                elif kind == "done":
                    finish_reason = "stop" if stopped else _finish_reason_openai(payload["tokens"])
                    n_completion = len(payload["tokens"])
                    cached_tokens = payload.get("prefill_reused", 0)
                    _log_gen_stats(payload)
                    break
                elif kind == "cancelled":
                    # Early cut-off caused by a stop string match (the same
                    # reason and the same contract as _openai_stream).
                    finish_reason = "stop"
                    n_completion = raw_token_count[0]
                    cached_tokens = reused_at_select
                    print(
                        f"[mlxturbo-serve] stop string matched — cancelled early after "
                        f"{n_completion} decoded tokens (prefill reused={cached_tokens})"
                    )
                    break
                else:  # error
                    err = {"error": {"message": str(payload), "type": "server_error"}}
                    yield f"data: {json.dumps(err)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            final_chunk = {
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {"index": 0, "text": "", "logprobs": None, "finish_reason": finish_reason}
                ],
            }
            if include_usage:
                final_chunk["usage"] = None
            yield f"data: {json.dumps(final_chunk)}\n\n"

            if include_usage:
                usage_chunk = {
                    "id": req_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [],
                    "usage": _usage_dict(len(prompt_ids), n_completion, cached_tokens),
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"

            yield "data: [DONE]\n\n"
        finally:
            cancel_event.set()
            await _await_worker(future)
    finally:
        if owned[0]:
            STATE.lock.release()


# ---------- OpenAI-compatible (Responses API, /v1/responses) ----------
#
# Codex CLI (from 2026-02-01 onwards) accepts only "responses" as its wire_api
# (the old "chat" has been removed), so Codex does not work unless this is
# implemented. The structure differs from Chat Completions (input rather than
# messages, an output array rather than choices, flat rather than nested
# tools), but generation itself, thinking separation, tool call parsing,
# sampling parameters, session selection and stop detection are shared
# completely with the Chat Completions/Anthropic paths — the only things
# written anew here are "normalize input into the existing internal messages
# format" and "assemble the event stream (the same vocabulary as
# _collect_events) into an output array / SSE events"; the generation logic is
# not duplicated as a third protocol.
#
# Server-side conversation continuation via store / previous_response_id is
# implemented purely with an in-memory LRU (STATE.response_store) — nothing is
# persisted (everything is lost on process restart; the design stated in the
# module docstring at the top, that "the client resends the whole conversation
# history every turn", is itself unchanged. previous_response_id is no more
# than a thin optimization that "keeps the same history on the server side
# too", saving the effort of resending everything). See
# _resolve_previous_response/_store_response_if_requested for details.


def _responses_content_to_text(content) -> str:
    """Reduce one Responses API item's content (a string, or an array of typed
    parts such as input_text/output_text/refusal) to plain text. Same policy as
    ``_content_to_text`` (for Chat Completions/Anthropic): image/audio/file
    parts are not silently skipped but turned into
    ``MultimodalContentError`` (400) (because mlxturbo only serves text-only
    models). Broken shapes (a part with no type, a text that is not a string,
    and so on) likewise become ``InvalidContentError`` (400).
    """

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                raise InvalidContentError(
                    f"each content part must be an object, got {type(part).__name__}"
                )
            if "type" not in part:
                raise InvalidContentError("each content part must have a 'type'")
            ptype = part["type"]
            if ptype in ("input_text", "output_text", "text"):
                text = part.get("text")
                if not isinstance(text, str):
                    raise InvalidContentError(
                        f"content part of type {ptype!r} must have a string 'text' field"
                    )
                parts.append(text)
                continue
            if ptype == "refusal":
                # An echo of a refusal block the model itself produced. It is
                # not skipped but treated as text (this is not fabrication: the
                # client is sending back verbatim the string it actually
                # received on the previous turn).
                refusal = part.get("refusal")
                if isinstance(refusal, str):
                    parts.append(refusal)
                continue
            raise MultimodalContentError(ptype)
        return "".join(parts)
    raise InvalidContentError(
        f"'content' must be a string or a list of content parts, got {type(content).__name__}"
    )


def _normalize_responses_input(input_value, instructions) -> list[dict]:
    """Convert the Responses API's ``input`` (plus ``instructions``) into the
    existing internal messages format that ``_apply_template`` accepts (the
    same shape as ``_normalize_openai_messages``: role/content, plus
    ``tool_calls`` for assistant and role: "tool" + ``tool_call_id`` for tool
    results). Converging on the existing format here lets everything after this
    point (apply_chat_template onwards) be shared completely with Chat
    Completions/Anthropic.

    Each item in ``input`` branches on its type (a missing type is treated as
    "message" — to accommodate clients such as Codex CLI that send the naive
    ``{"role":..., "content":...}`` shape):

    - ``message``: role (user/assistant/system/developer) + content.
      developer is treated the same as system (there is no point in this server
      having two kinds of system-equivalent role).
    - ``function_call``: a tool call the model made on a previous turn.
      Consecutive function_calls are gathered into the ``tool_calls`` of a
      single assistant message (the same shape as Chat Completions'
      assistant.tool_calls).
    - ``function_call_output``: the result of executing a tool. Converted into
      a role: "tool" message.
    - ``reasoning``/``item_reference``: a previous turn's reasoning summary, or
      a reference item (which presumes a previous_response_id this server does
      not support). There is no need to make the model read them again, so they
      are skipped (the same policy as bug 2: the handling of thinking blocks).
    - Any other unknown type becomes a 400 rather than being silently ignored
      (in line with this server's overall policy of not letting broken or
      unsupported input through as an empty prompt).

    In addition to the top-level ``instructions``, Codex CLI mixes a
    role: "developer" message in near the head of ``input`` (confirmed on a
    captured body). Since developer is treated the same as system, leaving it
    alone would line up two system-equivalent messages, non-consecutively or in
    a position other than the head, and the chat template would die for the
    same reason as the bug on the Anthropic path (System message must be at the
    beginning). All system/developer content is collected, concatenated into a
    single system message in the order ``instructions`` -> the order it
    appeared in ``input``, and always placed first.
    """

    system_parts: list[str] = []
    if instructions:
        system_parts.append(_content_to_text(instructions))

    def _finish(out: list[dict]) -> list[dict]:
        if system_parts:
            out.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        return out

    if input_value is None:
        raise InvalidContentError("'input' is required")
    if isinstance(input_value, str):
        return _finish([{"role": "user", "content": input_value}])
    if not isinstance(input_value, list):
        raise InvalidContentError(
            f"'input' must be a string or an array of items, got {type(input_value).__name__}"
        )

    out: list[dict] = []
    pending_calls: list[dict] | None = None

    def flush_pending() -> None:
        nonlocal pending_calls
        if pending_calls:
            out.append({"role": "assistant", "content": "", "tool_calls": pending_calls})
            pending_calls = None

    for item in input_value:
        if not isinstance(item, dict):
            raise InvalidContentError("each item in 'input' must be an object")
        itype = item.get("type", "message")

        if itype == "message":
            role = item.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                raise InvalidContentError(
                    "message item 'role' must be one of 'user'/'assistant'/'system'/"
                    f"'developer', got {role!r}"
                )
            flush_pending()
            text = _responses_content_to_text(item.get("content"))
            if role in ("system", "developer"):
                if text:
                    system_parts.append(text)
                continue
            out.append({"role": role, "content": text})
            continue

        if itype == "function_call":
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise InvalidContentError(
                    "'function_call' item must have a non-empty string 'name'"
                )
            raw_args = item.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args_obj = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError as exc:
                    raise InvalidContentError(
                        f"'function_call.arguments' is not valid JSON: {exc}"
                    ) from exc
            elif isinstance(raw_args, dict):
                args_obj = raw_args
            else:
                raise InvalidContentError(
                    "'function_call.arguments' must be a JSON string or an object"
                )
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
            if pending_calls is None:
                pending_calls = []
            pending_calls.append(
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": args_obj}}
            )
            continue

        # When an item other than a function_call arrives, commit the group of
        # function_calls buffered up to that point (the same treatment as for a
        # message item).
        flush_pending()

        if itype == "function_call_output":
            call_id = item.get("call_id")
            output_text = _responses_content_to_text(item.get("output", "") or "")
            out.append({"role": "tool", "content": output_text, "tool_call_id": call_id})
            continue
        if itype in ("reasoning", "item_reference"):
            continue
        raise InvalidContentError(f"unsupported 'input' item type: {itype!r}")

    flush_pending()
    return _finish(out)


def _flatten_responses_tools(tools: list) -> list:
    """Handle the non-``function`` types Codex CLI mixes into ``tools``.

    Two kinds confirmed on actually captured bodies:

    - ``{"type": "namespace", "tools": [...]}``: a container that groups
      subagent-related tools; it is not itself a callable tool. Its inner
      ``tools`` (whose contents are ordinary ``{"type": "function", ...}``) are
      expanded and lifted to the top level. The expansion is recursive, in case
      they are nested.
    - ``{"type": "web_search", ...}``: this server has nothing that performs
      web search (mlxturbo has nothing but a simple forward into a local
      model), so it cannot execute it even if it is passed. Swallowing it
      silently would leave the client proceeding in the belief that search is
      available, so it is removed after recording in the operator-facing log
      that it was dropped (the same one-line ``[mlxturbo-serve]`` log style as
      _log_gen_stats).

    ``function`` types pass straight through (the subsequent
    ``_validate_responses_tools`` validates their shape itself).
    """

    flat: list = []
    for t in tools:
        if not isinstance(t, dict):
            flat.append(t)
            continue
        ttype = t.get("type")
        if ttype == "namespace":
            nested = t.get("tools")
            if isinstance(nested, list):
                flat.extend(_flatten_responses_tools(nested))
            continue
        if ttype != "function":
            print(
                f"[mlxturbo-serve] dropping unsupported tool in 'tools': "
                f"type={ttype!r} name={t.get('name')!r} — this server has no "
                "execution backend for it"
            )
            continue
        flat.append(t)
    return flat


def _validate_responses_tools(tools) -> str | None:
    if not isinstance(tools, list) or not tools:
        return "'tools' must be a non-empty array"
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            return 'each item in \'tools\' must be an object with "type": "function"'
        if not isinstance(t.get("name"), str) or not t.get("name"):
            return "each tool must have a non-empty string 'name'"
    return None


def _responses_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert the Responses API's flat tool definitions
    (``{"type":"function","name":..., "parameters":...}``) into the nested
    OpenAI (Chat Completions) format
    (``{"type":"function","function":{"name",...}}``) that
    ``apply_chat_template``/``tool_parser`` expect."""

    return [
        {
            "type": "function",
            "function": {
                "name": t.get("name"),
                "description": t.get("description", "") or "",
                "parameters": t.get("parameters", {}) or {},
            },
        }
        for t in tools
    ]


def _resolve_tool_choice_responses(body: dict) -> tuple[list | None, str | None]:
    """The Responses API version of ``_resolve_tool_choice_openai``. Apart from
    the tools being flat in shape, the resolution logic
    (auto/none/required/naming a specific function) is exactly the same."""

    tools = body.get("tools")
    if not tools:
        return None, None
    if isinstance(tools, list):
        tools = _flatten_responses_tools(tools)
        if not tools:
            return None, None
    shape_err = _validate_responses_tools(tools)
    if shape_err is not None:
        return None, shape_err
    choice = body.get("tool_choice", "auto")
    if choice is None:
        choice = "auto"
    if choice == "none":
        return None, None
    if choice == "auto":
        return _responses_tools_to_openai(tools), None
    if choice == "required":
        return None, (
            "'tool_choice: \"required\"' is not supported: this server has no "
            "mechanism to force the model to call a tool (it can only detect "
            "tool calls the model chooses to emit on its own)"
        )
    if isinstance(choice, dict):
        return None, (
            "'tool_choice' selecting a specific function is not supported: this "
            "server has no mechanism to force the model to call a particular tool"
        )
    return None, f"'tool_choice' must be 'auto', 'none', 'required', or a function object; got {choice!r}"


def _resolve_max_output_tokens(body: dict, cap: int) -> tuple[int, str | None]:
    raw = body.get("max_output_tokens")
    if raw is None:
        return cap, None
    return _parse_positive_int(raw, cap, "max_output_tokens")


def _resolve_thinking_responses(body: dict) -> tuple[bool | None, int | None, str | None]:
    """The Responses API version of ``_resolve_thinking``. The field read is
    ``reasoning: {"effort": ...}`` — the mapping onto a budget
    (`_REASONING_EFFORT_BUDGET`) is shared with the OpenAI side."""

    value = body.get("reasoning")
    if not value:
        return None, None, None
    if not isinstance(value, dict):
        return None, None, "'reasoning' must be an object"
    effort = value.get("effort")
    if effort is None:
        return None, None, None
    key = effort.lower() if isinstance(effort, str) else None
    if key == "none":
        return False, 0, None
    budget = _REASONING_EFFORT_BUDGET.get(key, _REASONING_EFFORT_DEFAULT_BUDGET)
    return True, budget, None


def _responses_output_items_from_events(events: list[tuple[str, object]]) -> list[dict]:
    """Assemble the order-preserving high-level event stream (the same
    vocabulary as ``_collect_events``: reasoning_delta/content_delta/tool_call)
    into the Responses API's ``output`` array (typed items). The Responses
    version of ``_anthropic_blocks_from_events`` — consecutive
    reasoning_deltas and content_deltas are each merged into a single
    reasoning/message item, and each tool_call becomes an independent
    function_call item."""

    items: list[dict] = []
    text_buf: list[str] = []
    reasoning_buf: list[str] = []

    def flush_text() -> None:
        if text_buf:
            items.append(
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "".join(text_buf), "annotations": []}
                    ],
                }
            )
            text_buf.clear()

    def flush_reasoning() -> None:
        if reasoning_buf:
            items.append(
                {
                    "type": "reasoning",
                    "id": f"rs_{uuid.uuid4().hex}",
                    "summary": [{"type": "summary_text", "text": "".join(reasoning_buf)}],
                }
            )
            reasoning_buf.clear()

    for kind, val in events:
        if kind == "reasoning_delta":
            flush_text()
            if val:
                reasoning_buf.append(val)
        elif kind == "content_delta":
            flush_reasoning()
            if val:
                text_buf.append(val)
        elif kind == "tool_call":
            flush_reasoning()
            flush_text()
            items.append(
                {
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex}",
                    "call_id": val["id"],
                    "name": val["name"],
                    "arguments": json.dumps(val["arguments"], ensure_ascii=False),
                    "status": "completed",
                }
            )
    flush_reasoning()
    flush_text()
    return items


# ---------- previous_response_id / store (item 15) ----------
#
# The conversation is continued purely with an in-memory LRU
# (STATE.response_store). Nothing is persisted — everything is lost if the
# process restarts (this server was never designed to persist responses; see
# the docstring at the top of the module). What is stored is "the raw input
# item sequence actually used to generate that response + that response's own
# output item sequence" — both in the shape _normalize_responses_input accepts
# as-is (the Responses API's raw item shape), so the next turn can reconstruct
# the whole conversation simply by concatenating it in front of ``input`` and
# running it through _normalize_responses_input once more. No separate new
# conversion logic is kept.


def _resolve_previous_response(
    body: dict,
) -> tuple[list, str | None, "JSONResponse | None"]:
    """Resolve ``previous_response_id``. The return value is
    ``(prior_items, effective_instructions, error_response)``.

    If it is not found, return 404 (OpenAI-conforming, in the same manner as
    _check_model_openai). Since this server has nothing but an LRU that is lost
    entirely on restart, "not found" does not distinguish between "it was never
    stored with store:true in the first place", "it was evicted from the LRU"
    and "the process restarted" — the 404 message explains as much.
    """

    previous_response_id = body.get("previous_response_id")
    if previous_response_id is None:
        return [], body.get("instructions"), None
    if not isinstance(previous_response_id, str):
        return [], None, _openai_error("'previous_response_id' must be a string")
    record = STATE.response_store.get(previous_response_id)
    if record is None:
        return (
            [],
            None,
            _openai_error(
                f"'previous_response_id' {previous_response_id!r} was not found. This "
                "server keeps only an in-memory LRU of responses created with "
                "'store: true' (see --max-stored-responses); it is never persisted to "
                "disk, so a process restart, an evicted entry, or a response that was "
                "never stored will all surface as this same error.",
                status=404,
                code="response_not_found",
            ),
        )
    STATE.response_store.move_to_end(previous_response_id)
    effective_instructions = body.get("instructions")
    if effective_instructions is None:
        effective_instructions = record["instructions"]
    return record["items"], effective_instructions, None


def _combine_responses_input(raw_input, prior_items: list) -> tuple[list | None, str | None]:
    """Concatenate this turn's ``input`` after ``prior_items`` (the raw item
    sequence restored from previous_response_id, empty when there is none). If
    this turn's part is a string, it is first expanded into the same shape as
    ``_normalize_responses_input``'s "single user message when input is a
    string" conversion and only then concatenated — so that it can be mixed
    with prior_items (a list). On the normal path where there is no
    previous_response_id (prior_items is empty), expanding the string here
    still ends up handing _normalize_responses_input exactly the same message
    sequence as before (it merely goes through the list branch).
    """

    if raw_input is None:
        return None, "'input' is required"
    if isinstance(raw_input, str):
        new_items = [{"type": "message", "role": "user", "content": raw_input}]
    elif isinstance(raw_input, list):
        new_items = raw_input
    else:
        return None, f"'input' must be a string or an array of items, got {type(raw_input).__name__}"
    return prior_items + new_items, None


def _store_response_if_requested(
    store: bool,
    resp_id: str,
    effective_instructions: str | None,
    combined_input: list | None,
    output_items: list[dict],
) -> None:
    """Only when ``store: true``, push this response into STATE.response_store
    (an LRU with a limit of STATE.max_stored_responses; when it is exceeded the
    least recently used entry is discarded). ``combined_input`` (the raw item
    sequence actually used this time, including what was restored from the
    previous turn) + ``output_items`` (this response's own output, kept whole
    including the reasoning items — harmless, since the reading side,
    _normalize_responses_input, skips reasoning/item_reference) are stored in a
    form the next turn can concatenate as-is."""

    if not store:
        return
    STATE.response_store[resp_id] = {
        "items": list(combined_input or []) + output_items,
        "instructions": effective_instructions,
    }
    STATE.response_store.move_to_end(resp_id)
    while len(STATE.response_store) > STATE.max_stored_responses:
        STATE.response_store.popitem(last=False)


@app.post("/v1/responses")
async def responses_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _openai_error("request body must be valid JSON")
    if not isinstance(body, dict):
        return _openai_error("request body must be a JSON object")

    # previous_response_id/store (item 15): the server keeps an in-memory LRU
    # (STATE.response_store, main()'s --max-stored-responses; nothing is
    # persisted) so that a response saved with store:true can be reached from a
    # later turn's previous_response_id. Previously both were refused with a
    # 400 rather than silently ignored (see the older version of this comment
    # and the _resolve_previous_response docstring), but this was implemented
    # after it was pointed out that OpenAI's official agentic example (sending
    # only the previous turn's id and not resending the full text) does not
    # work as-is.
    prior_items, effective_instructions, prev_err = _resolve_previous_response(body)
    if prev_err is not None:
        return prev_err
    store = bool(body.get("store", False))

    model_err = _check_model_openai(body)
    if model_err is not None:
        return model_err

    combined_input, input_err = _combine_responses_input(body.get("input"), prior_items)
    if input_err is not None:
        return _openai_error(input_err)
    try:
        norm_messages = _normalize_responses_input(combined_input, effective_instructions)
    except ContentNormalizationError as exc:
        return _openai_error(str(exc))

    resolved_tools, tool_err = _resolve_tool_choice_responses(body)
    if tool_err is not None:
        return _openai_error(tool_err)
    unsupported_tools_err = _check_tool_calling_support(resolved_tools)
    if unsupported_tools_err is not None:
        return _openai_error(unsupported_tools_err)
    tool_enabled = resolved_tools is not None

    enable_thinking, thinking_budget, err = _resolve_thinking_responses(body)
    if err is not None:
        return _openai_error(err)
    try:
        prompt_ids = _apply_template(norm_messages, enable_thinking, tools=resolved_tools)
    except Exception as exc:
        return _openai_error(f"failed to render chat template: {exc}")
    ctx_err = _check_context_length(prompt_ids, "openai")
    if ctx_err is not None:
        return ctx_err

    max_tokens, err = _resolve_max_output_tokens(body, STATE.max_tokens_cap)
    if err is not None:
        return _openai_error(err)
    if thinking_budget is not None:
        thinking_budget = min(thinking_budget, max_tokens)
    temp, err = _parse_temperature(body, STATE.default_temp)
    if err is not None:
        return _openai_error(err)
    sampling_params, err = _parse_sampling_params(body)
    if err is not None:
        return _openai_error(err)
    gen_runner, downgrade_reason, unsupported_err = _resolve_runner_for_request(sampling_params)
    if unsupported_err is not None:
        return _openai_error(unsupported_err)

    model_id = STATE.model_name
    stream = bool(body.get("stream", False))
    resp_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())

    if not _try_reserve_queue_slot():
        return _busy_response("openai", "server is busy: too many queued requests")

    if stream:
        # _responses_stream's finally releases this request's queue slot (the
        # same reason as the other openai paths).
        return StreamingResponse(
            _queue_owned_stream(
                _responses_stream(
                    prompt_ids,
                    max_tokens,
                    temp,
                    resp_id,
                    created,
                    model_id,
                    thinking_budget,
                    sampling_params,
                    tool_enabled,
                    resolved_tools,
                    runner=gen_runner,
                    store=store,
                    combined_input=combined_input,
                    effective_instructions=effective_instructions,
                )
            ),
            media_type="text/event-stream",
            headers=_downgrade_headers(downgrade_reason),
        )

    batch_tier = _resolve_batch_tier(gen_runner, prompt_ids, max_tokens)
    try:
        if batch_tier is not None:
            res = await _run_generate_batched(
                prompt_ids, max_tokens, temp, None, **sampling_params
            )
        else:
            async with STATE.lock:
                session = None if downgrade_reason is not None else await _select_session_on_executor(
                    prompt_ids
                )
                res = await _run_generate(
                    prompt_ids,
                    max_tokens,
                    temp,
                    STATE.eos_ids,
                    None,
                    session,
                    runner=gen_runner,
                    **sampling_params,
                )
    except Exception as exc:
        return _openai_error(str(exc), status=500, err_type="server_error")
    finally:
        _release_queue_slot()
    _log_gen_stats(res)

    events, budget_exceeded = _collect_events(
        prompt_ids, res["tokens"], thinking_budget, tool_enabled, resolved_tools
    )
    output_items = _responses_output_items_from_events(events)
    status, incomplete_reason = _responses_terminal_state(
        res["tokens"],
        budget_exceeded,
        any(kind == "tool_call" for kind, _ in events),
    )

    out = {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model_id,
        "output": output_items,
        "usage": {
            "input_tokens": len(prompt_ids),
            "output_tokens": len(res["tokens"]),
            "total_tokens": len(prompt_ids) + len(res["tokens"]),
        },
    }
    if incomplete_reason is not None:
        out["incomplete_details"] = {"reason": incomplete_reason}
    _attach_downgrade_reason(out, downgrade_reason)
    _store_response_if_requested(store, resp_id, effective_instructions, combined_input, output_items)
    return out


async def _responses_stream(
    prompt_ids,
    max_tokens,
    temp,
    resp_id,
    created,
    model_id,
    thinking_budget,
    sampling_params: dict | None = None,
    tool_enabled: bool = False,
    tools_for_parsing=None,
    runner=None,
    store: bool = False,
    combined_input: list | None = None,
    effective_instructions: str | None = None,
):
    """Emit the Responses API's main lifecycle event sequence: response.created
    -> response.output_item.added -> response.output_text.delta /
    response.reasoning_summary_text.delta / response.function_call_arguments.delta
    -> the respective done events -> response.output_item.done ->
    response.completed / response.incomplete / response.failed.

    Acceptance (validation) has already completed on the caller's side, before
    the StreamingResponse was assembled — following the same policy as the
    existing OpenAI/Anthropic streaming paths (see the docstring in server.py),
    response.created is streamed out immediately without waiting for generation
    to start.
    """

    sequence_number = 0

    def sse(event: str, data: dict) -> str:
        nonlocal sequence_number
        payload = dict(data)
        payload.setdefault("sequence_number", sequence_number)
        sequence_number += 1
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    yield sse(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": resp_id,
                "object": "response",
                "created_at": created,
                "status": "in_progress",
                "model": model_id,
                "output": [],
                "usage": None,
            },
        },
    )

    owned = [False]
    batch_tier = _resolve_batch_tier(runner or STATE.runner, prompt_ids, max_tokens)
    try:
        if batch_tier is not None:
            session = None
            q, future, cancel_event, _raw_token_count = _start_batched_generation(
                prompt_ids,
                max_tokens,
                temp,
                thinking_budget,
                tool_enabled,
                tools_for_parsing,
                **(sampling_params or {}),
            )
        else:
            async for keepalive in _acquire_lock_with_keepalive(STATE.lock, owned):
                yield keepalive

            downgraded = runner is not None and runner is not STATE.runner
            session = None if downgraded else await _select_session_on_executor(prompt_ids)
            q, future, cancel_event, _raw_token_count = _start_generation(
                prompt_ids,
                max_tokens,
                temp,
                thinking_budget,
                tool_enabled,
                tools_for_parsing,
                session=session,
                runner=runner,
                **(sampling_params or {}),
            )
        try:
            first_item = None
            async for _ka_kind, _ka_val in _await_with_keepalive(asyncio.to_thread(q.get)):
                if _ka_kind == "keepalive":
                    yield _ka_val
                else:
                    first_item = _ka_val
            _requeue_front(q, first_item)

            output_items: list[dict] = []
            next_index = 0
            current_kind: str | None = None  # None | "reasoning" | "content"
            current_item: dict | None = None
            current_index = 0
            budget_exceeded = False
            final_tokens: list[int] = []
            saw_tool_call = False

            def close_current():
                nonlocal current_item, current_kind
                if current_item is None:
                    return []
                item = current_item
                events = []
                if item["type"] == "message":
                    item["status"] = "completed"
                    events.append(
                        sse(
                            "response.output_text.done",
                            {
                                "type": "response.output_text.done",
                                "output_index": current_index,
                                "item_id": item["id"],
                                "content_index": 0,
                                "text": item["content"][0]["text"],
                                "logprobs": [],
                            },
                        )
                    )
                else:
                    events.append(
                        sse(
                            "response.reasoning_summary_text.done",
                            {
                                "type": "response.reasoning_summary_text.done",
                                "output_index": current_index,
                                "item_id": item["id"],
                                "summary_index": 0,
                                "text": item["summary"][0]["text"],
                            },
                        )
                    )
                output_items.append(item)
                events.append(
                    sse(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": current_index,
                            "item": item,
                        },
                    )
                )
                current_item = None
                current_kind = None
                return events

            while True:
                kind, payload = await asyncio.to_thread(q.get)
                if kind == "tool_call":
                    saw_tool_call = True
                    for closing in close_current():
                        yield closing
                    idx = next_index
                    next_index += 1
                    call_item = {
                        "type": "function_call",
                        "id": f"fc_{uuid.uuid4().hex}",
                        "call_id": payload["id"],
                        "name": payload["name"],
                        "arguments": "",
                        "status": "in_progress",
                    }
                    yield sse(
                        "response.output_item.added",
                        {"type": "response.output_item.added", "output_index": idx, "item": dict(call_item)},
                    )
                    args_str = json.dumps(payload["arguments"], ensure_ascii=False)
                    for piece in _chunk_string(args_str):
                        yield sse(
                            "response.function_call_arguments.delta",
                            {
                                "type": "response.function_call_arguments.delta",
                                "output_index": idx,
                                "item_id": call_item["id"],
                                "delta": piece,
                            },
                        )
                    call_item["arguments"] = args_str
                    call_item["status"] = "completed"
                    yield sse(
                        "response.function_call_arguments.done",
                        {
                            "type": "response.function_call_arguments.done",
                            "output_index": idx,
                            "item_id": call_item["id"],
                            "name": call_item["name"],
                            "arguments": args_str,
                        },
                    )
                    output_items.append(call_item)
                    yield sse(
                        "response.output_item.done",
                        {"type": "response.output_item.done", "output_index": idx, "item": call_item},
                    )
                elif kind in ("reasoning_delta", "content_delta"):
                    if not payload:
                        continue
                    channel = "reasoning" if kind == "reasoning_delta" else "content"
                    if current_kind != channel:
                        for closing in close_current():
                            yield closing
                        idx = next_index
                        next_index += 1
                        current_index = idx
                        current_kind = channel
                        if channel == "reasoning":
                            current_item = {
                                "type": "reasoning",
                                "id": f"rs_{uuid.uuid4().hex}",
                                "summary": [{"type": "summary_text", "text": ""}],
                            }
                        else:
                            current_item = {
                                "type": "message",
                                "id": f"msg_{uuid.uuid4().hex}",
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [{"type": "output_text", "text": "", "annotations": []}],
                            }
                        yield sse(
                            "response.output_item.added",
                            {"type": "response.output_item.added", "output_index": idx, "item": dict(current_item)},
                        )
                    if channel == "reasoning":
                        current_item["summary"][0]["text"] += payload
                        yield sse(
                            "response.reasoning_summary_text.delta",
                            {
                                "type": "response.reasoning_summary_text.delta",
                                "output_index": current_index,
                                "item_id": current_item["id"],
                                "summary_index": 0,
                                "delta": payload,
                            },
                        )
                    else:
                        current_item["content"][0]["text"] += payload
                        yield sse(
                            "response.output_text.delta",
                            {
                                "type": "response.output_text.delta",
                                "output_index": current_index,
                                "item_id": current_item["id"],
                                "content_index": 0,
                                "logprobs": [],
                                "delta": payload,
                            },
                        )
                elif kind == "budget_exceeded":
                    budget_exceeded = True
                elif kind == "done":
                    final_tokens = payload["tokens"]
                    _log_gen_stats(payload)
                    break
                else:  # error
                    response_error = {
                        "code": "server_error",
                        "message": str(payload),
                        "param": None,
                    }
                    yield sse(
                        "error",
                        {"type": "error", **response_error},
                    )
                    failed_response = {
                        "id": resp_id,
                        "object": "response",
                        "created_at": created,
                        "status": "failed",
                        "model": model_id,
                        "output": output_items,
                        "error": response_error,
                        "usage": None,
                    }
                    yield sse(
                        "response.failed",
                        {"type": "response.failed", "response": failed_response},
                    )
                    return

            for closing in close_current():
                yield closing

            status, incomplete_reason = _responses_terminal_state(
                final_tokens, budget_exceeded, saw_tool_call
            )

            final_response = {
                "id": resp_id,
                "object": "response",
                "created_at": created,
                "status": status,
                "model": model_id,
                "output": output_items,
                "usage": {
                    "input_tokens": len(prompt_ids),
                    "output_tokens": len(final_tokens),
                    "total_tokens": len(prompt_ids) + len(final_tokens),
                },
            }
            if incomplete_reason is not None:
                final_response["incomplete_details"] = {"reason": incomplete_reason}
            # Item 15: if store:true, save here. Even when the status is
            # "incomplete" (cut off by max_tokens, and so on), the output is
            # saved as-is, the same as in the non-streaming case — because
            # being able to continue from a partial response via
            # previous_response_id is more useful. "failed" has already left
            # through the return above and is therefore not saved (a broken
            # response is not carried over to the next turn).
            _store_response_if_requested(
                store, resp_id, effective_instructions, combined_input, output_items
            )
            terminal_event = (
                "response.completed" if status == "completed" else "response.incomplete"
            )
            yield sse(
                terminal_event,
                {"type": terminal_event, "response": final_response},
            )
        finally:
            cancel_event.set()
            await _await_worker(future)
    finally:
        if owned[0]:
            STATE.lock.release()


# ---------- startup ----------


def _resolve_model_max_context(config: dict) -> int | None:
    """Take the context-length limit the model itself declares from the raw
    config loaded at startup (the return value of
    ``mlx_lm_load(..., return_config=True)``). Nothing is hardcoded —
    ``--max-context-tokens`` takes precedence, and this is used only as the
    fallback when it is not specified.

    In the VLM wrapper format (Qwen3.6-35B-A3B and the like)
    ``max_position_embeddings`` is nested under ``text_config`` rather than at
    the top level, so if it is not found we descend exactly one level and look
    there. If it is still not found, None is returned and
    ``_check_context_length`` keeps running with the guard disabled
    (unlimited).
    """

    if isinstance(config.get("max_position_embeddings"), int):
        return config["max_position_embeddings"]
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and isinstance(
        text_config.get("max_position_embeddings"), int
    ):
        return text_config["max_position_embeddings"]
    return None


def _metal_safe_prefill_limit(config: dict) -> int | None:
    """The token-count limit derived from Metal's actual allocation ceiling, up
    to which SpecEngine can forward a new prompt.

    SpecEngine (mlxturbo/spec.py) forwards a new prompt split into chunks of
    PREFILL_STEP_SIZE (2048 by default) tokens
    (``SpecEngine._prefill_hidden``). The attention score matrix allocated in
    one forward is therefore ``num_attention_heads * PREFILL_STEP_SIZE * T *
    bytes_per_elem`` (T being the total number of processed tokens, largest on
    the final chunk) — down from the pre-split
    ``num_attention_heads * T^2 * bytes_per_elem`` (quadratic in T) to linear
    in T. Even so, exceeding Metal's single-buffer limit
    (``mx.device_info()["max_buffer_length"]``, 86,586,540,032 bytes on the
    actual machine) makes the request fail with ``[metal::malloc]``.

    Looking only at ``max_position_embeddings`` (the training-time context
    length the model declares) can return a value of an order that is
    realistically never reached, so that value is not used as the limit
    directly; instead the smaller of the value computed here and the value from
    ``_resolve_model_max_context`` is used as the actual limit
    (``_resolve_default_max_context_tokens``). For most models
    max_position_embeddings binds first — because the post-split wall is very
    high (over 1 million tokens for this model).

    The theoretical T that exactly fills one buffer (the boundary at which
    exceeding it fails immediately) is multiplied by 0.9 to err on the safe
    side — because buffers other than the attention matrix are allocated at the
    same time, so it cannot be called safe right up to the theoretical value.
    """

    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        return None
    n_heads = text_config.get("num_attention_heads")
    if not isinstance(n_heads, int) or n_heads <= 0:
        return None
    dtype = str(text_config.get("dtype") or config.get("dtype") or "bfloat16")
    bytes_per_elem = 2 if "16" in dtype else 4
    try:
        max_buffer_bytes = mx.device_info()["max_buffer_length"]
    except Exception:
        return None
    if not isinstance(max_buffer_bytes, int) or max_buffer_bytes <= 0:
        return None
    theoretical = int(max_buffer_bytes / (n_heads * PREFILL_STEP_SIZE * bytes_per_elem))
    return int(theoretical * 0.9)


def _resolve_default_max_context_tokens(config: dict) -> int | None:
    """The default when ``--max-context-tokens`` is not specified: the smaller
    of the limit the model declares (``_resolve_model_max_context``) and the
    value derived from what Metal can actually allocate
    (``_metal_safe_prefill_limit``). None if neither can be obtained (the guard
    is disabled)."""

    candidates = [
        v
        for v in (_resolve_model_max_context(config), _metal_safe_prefill_limit(config))
        if v is not None
    ]
    return min(candidates) if candidates else None


def _install_graceful_shutdown(server_obj: "uvicorn.Server") -> None:
    """On the first SIGTERM/SIGINT: set ``_SHUTTING_DOWN`` (so that
    ``_gate_requests`` refuses subsequent new requests with 503) but do not yet
    set uvicorn's own ``should_exit``.

    The reason, confirmed on the actual machine: uvicorn's
    ``Server.shutdown()`` closes the listening socket as its very first move
    (``for server in self.servers: server.close()``). That runs almost
    simultaneously with the end of ``main_loop`` right after ``should_exit`` is
    set, so within a few hundred milliseconds of the signal a new TCP
    connection itself becomes "connection refused", and everything ends without
    the ASGI layer (``_gate_requests``) ever getting a chance to return a 503
    (measured: a new request 0.2 seconds after sending SIGTERM was confirmed to
    fail at connection time, without even reaching the HTTP level).

    So ``_drain_and_exit`` (a watchdog task made resident by wrapping
    ``startup``) keeps the listener open and waits until ``STATE.queue_depth``
    (the number of generation requests being processed) reaches 0, so that new
    requests during that window can be refused by ``_gate_requests`` as real
    503 responses. Only once the queue is empty is ``should_exit`` set, and
    uvicorn's own ordinary shutdown (close the listener and wait for the
    remaining connections) begins.

    A second signal (of any kind) terminates immediately via ``os._exit``.
    Exiting via uvicorn's own ``should_exit``/``force_exit`` is not
    "immediate" when the worker thread that is generating (``STATE.executor``,
    non-daemon by default) is executing MLX's synchronous generation code,
    because the ordinary interpreter shutdown waits for that thread to finish
    (confirmed by measurement: the process exited only after a 500-token
    generation had completed)."""

    original_startup = server_obj.startup
    signal_count = {"n": 0}

    def handle_exit(sig, frame):
        global _SHUTTING_DOWN
        signal_count["n"] += 1
        if signal_count["n"] == 1:
            _SHUTTING_DOWN = True
            print(
                "[mlxturbo-serve] シグナル受信: graceful shutdown 開始 "
                "(新規リクエストは 503、処理中のリクエストは完了を待つ)"
            )
            # Do not call original_handle_exit here (= do not yet set
            # uvicorn's own should_exit). See the docstring above —
            # _drain_and_exit sets it after seeing the queue drain.
        else:
            # Drop to an immediate OS-level exit via os._exit (see the
            # docstring for the reason). The ASGI lifespan's shutdown does not
            # run at all after this either.
            print("[mlxturbo-serve] 2 度目のシグナルを受信: 即時終了します", flush=True)
            server_obj.force_exit = True
            os._exit(1)

    async def startup_with_drain_watcher(sockets=None):
        await original_startup(sockets=sockets)
        asyncio.create_task(_drain_and_exit(server_obj))

    server_obj.handle_exit = handle_exit
    server_obj.startup = startup_with_drain_watcher


async def _drain_and_exit(server_obj: "uvicorn.Server") -> None:
    """The watchdog task ``_install_graceful_shutdown`` makes resident at
    startup.

    It waits until ``_SHUTTING_DOWN`` is set (the first signal), then waits
    until ``STATE.queue_depth`` (the number of in-flight requests across the 4
    generating paths) reaches 0, and only then sets uvicorn's own
    ``should_exit`` (= makes it close the listening socket). If a second signal
    (``force_exit``) arrives first, it returns immediately regardless of the
    state of the queue."""

    while not _SHUTTING_DOWN and not server_obj.force_exit:
        await asyncio.sleep(0.1)
    if server_obj.force_exit:
        return
    while STATE is not None and STATE.queue_depth > 0 and not server_obj.force_exit:
        await asyncio.sleep(0.1)
    server_obj.should_exit = True


def _enforce_required_runner(
    runner: Runner, required_kind: str | None, log_prefix: str = "[mlxturbo-serve]"
) -> None:
    """If ``--require-runner`` was specified but the resolved runner's ``KIND``
    does not match, do not fall back: state the reason explicitly and raise
    ``SystemExit(1)``.

    When it is not specified (``required_kind is None``) this does nothing —
    falling back is silently permitted as before, but a fallback is visible via
    ``fallback_reason`` on ``/health`` (see the class docstring of
    mlxturbo.runner.build_runner).
    """

    if required_kind is None:
        return
    resolved_kind = getattr(runner, "KIND", type(runner).__name__)
    if resolved_kind == required_kind:
        return
    reason = getattr(runner, "fallback_reason", None)
    detail = f" ({reason})" if reason else ""
    print(
        f"{log_prefix} --require-runner {required_kind} が指定されたが、解決された"
        f" runner は {resolved_kind}{detail}。--mtp のパスと重みを確認せよ"
    )
    raise SystemExit(1)


def _positive_int(value: str) -> int:
    """argparse type for capacities that must never be zero."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--version",
        action="version",
        version=f"mlxturbo-serve {_FASTMLX_VERSION}",
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--original", default="Qwen/Qwen3.8-27B")
    ap.add_argument(
        "--served-model-name",
        default=None,
        help="GET /v1/models とレスポンスの \"model\" 欄で名乗る id。既定は --model の"
        " basename (例: ~/models/qwen38fn-mlx-v-fast6 なら qwen38fn-mlx-v-fast6)。"
        " リクエストの model がこれと違えば 404 (OpenAI の model_not_found と同じ)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--max-tokens", type=int, default=4096, help="1 リクエストあたりの max_tokens 上限"
    )
    ap.add_argument(
        "--temp", type=float, default=0.7, help="リクエストで temperature 省略時の既定値"
    )
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument(
        "--mtp-depth",
        type=int,
        default=None,
        help="1 ラウンドで引くドラフトの数 (既定は mlxturbo.spec_flash.MTP_DEPTH)。"
        "2 以上ではヘッド自身の hyper 状態を次段に渡して連鎖させる。深くすると "
        "1 ラウンドの採用数は増えるが、検証フォワードの位置数も増える",
    )
    ap.add_argument(
        "--no-mtp", action="store_true", help="MTP を読み込まず lookup (SAM) のみで投機する"
    )
    ap.add_argument(
        "--draft-model",
        default=None,
        metavar="PATH",
        help="Kimi K3 レビュー項目 13: mlx_lm 自身の draft-model 投機デコード"
        " (mlx_lm.generate.speculative_generate_step) をアーキテクチャ非依存で使う。"
        " 指定すると mlxturbo.runner.DraftSpecRunner が有効になり、spec/flash_spec/"
        " fallback の既存の選択順序 (qwen3_5/qwen4_exp 専用) には一切入らない —"
        " Llama/Gemma/Mistral/Mixtral のような素の dense モデルにも投機を届けるための"
        " 経路。ドラフトはこのパスから mlx_lm.load で別途読み込む (本体より小さい"
        " モデルを想定するが、同じモデルを指定する self-draft でも動く — 速くは"
        " ならないが配線の確認にはなる)。トークナイザの vocab_size が本体と食い違う"
        " 場合は起動を中止する (exit 1)",
    )
    ap.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help="--draft-model 指定時、1 ラウンドでドラフトモデルに生成させるトークン数"
        " (mlx_lm.generate の num_draft_tokens そのまま)。既定 4",
    )
    ap.add_argument(
        "--lookup-spec",
        action="store_true",
        help="Kimi K3 レビュー項目 12: n-gram lookup (SAM, mlxturbo/sam.py) だけを使う"
        " モデル非依存の投機 (mlxturbo.lookup_spec.LookupSpecRunner) を有効化する。"
        " spec/flash_spec の契約を満たさないモデル (= 通常なら FallbackRunner) に"
        " だけかぶせる — 既に専用の投機があるモデルには何もしない。KV キャッシュが"
        " trim 不可なモデル、または temperature>0/repetition_penalty 等が要求された"
        " リクエストでは、内部で黙って通常生成にフォールバックする (壊れた出力を"
        " 出すよりは速度で妥協する判断、mlxturbo/lookup_spec.py 参照)。--draft-model"
        " と同時指定した場合は --draft-model が優先され、この経路には入らない",
    )
    ap.add_argument(
        "--lookup-max-draft",
        type=int,
        default=8,
        help="--lookup-spec 指定時、1 ラウンドで提案する n-gram ドラフトの最大長。既定 8",
    )
    ap.add_argument(
        "--lookup-min-match",
        type=int,
        default=2,
        help="--lookup-spec 指定時、ドラフトを提案するのに必要な最小の一致 (n-gram) 長。"
        " 既定 2 (2 トークン以上の繰り返しが無ければドラフトしない)",
    )
    ap.add_argument(
        "--ngram",
        default=None,
        help="n-gram (PLE) 表を外部サイドカーへ分離してある変換の場合、そのディレクトリを指定する"
        " (mlxturbo/ngram_stream.py)。cli.py の --ngram と同じ",
    )
    ap.add_argument(
        "--rebit",
        default=None,
        help="読み込み後にクラス単位でビットを打ち直す (例 hc=4,router=4)。"
        " mlxturbo/rebit.py の spec と同じ。パックを焼き直さずに"
        " ビット配分やカーネル適格性を A/B するための道具",
    )
    ap.add_argument(
        "--no-fused",
        action="store_true",
        help="hyper-connections 融合カーネルを無効化する (既定は有効)。"
        "qwen4_exp (Flash-Next 系) の通常生成経路にのみ効く",
    )
    ap.add_argument(
        "--allowed-origins",
        default=None,
        help="ブラウザからのクロスオリジン fetch を許可する Origin をカンマ区切りで"
        " 指定する (例: http://localhost:3000,https://my-ui.example)。'*' で全許可。"
        " 既定 (未指定) では CORS ヘッダを一切付けない = ローカル専用のまま"
        " (Open WebUI 等、ブラウザで動く UI から直接叩きたい場合に指定する)",
    )
    ap.add_argument(
        "--max-sessions",
        type=_positive_int,
        default=8,
        help="会話ごとの session (KV/prompt cache) を同時に保持する上限 (LRU、"
        " 超えたら最も長く未使用のものを捨てる)。91GB 級モデルの上に会話ごとの"
        " KV を無制限に積まないための上限",
    )
    ap.add_argument(
        "--max-stored-responses",
        type=_positive_int,
        default=50,
        help="/v1/responses に store: true が指定されたときサーバー側で保持する"
        " 応答の上限 (LRU、超えたら最も長く未使用のものを捨てる)。永続化はしない"
        " (プロセス再起動で全て失われる)。previous_response_id の解決に使う",
    )
    ap.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="1 リクエストのプロンプト長 (トークン数) の上限。超えたリクエストは"
        " 400 (invalid_request_error) で弾く。既定 (未指定) はモデルの config"
        " (max_position_embeddings 等) と、Metal が一括確保できる実際の上限"
        " から逆算した値のうち小さい方を自動で使う"
        " (_resolve_default_max_context_tokens 参照)。どちらも取れなければ"
        " ガード無効 (無制限)",
    )
    ap.add_argument(
        "--model-alias",
        action="append",
        default=[],
        metavar="NAME",
        help="--served-model-name 以外にも、リクエストの model 欄にこの名前が"
        " 来たら 404 にせず受け付ける (繰り返し指定可)。クライアントが裏方"
        " 処理 (例: Claude Code の会話タイトル生成) で別の小さいモデル名"
        " (例: claude-3-5-haiku-20241022) を送ってくる場合に使う。既定"
        " (未指定) では従来どおりサーブ中の名前と厳密一致しないと 404 の"
        " まま — 黙って何でも受け付ける形にはしない",
    )
    ap.add_argument(
        "--api-key",
        action="append",
        default=[],
        metavar="KEY",
        help="このサーバーを叩ける API キー (繰り返し指定可)。OpenAI 系"
        " (/v1/chat/completions, /v1/completions, /v1/responses, /v1/models)"
        " は Authorization: Bearer <key>、Anthropic 系 (/v1/messages) は"
        " x-api-key: <key> で送る (どちらのヘッダもどちらの経路でも受け付ける)。"
        " 既定 (未指定) では認証なし — ローカル専用の従来挙動のまま変えない。"
        " /health と /api/hello は鍵の有無に関わらず常に認証なしで通す",
    )
    ap.add_argument(
        "--max-queue",
        type=int,
        default=8,
        help="直列化ロックの待ち行列の上限 (既定 8)。生成系 4 経路 (chat/"
        "completions/messages/responses) でこれに達したリクエストは"
        " 503 (Retry-After 付き) で断る。/health の queue_depth で現在値を見れる",
    )
    ap.add_argument(
        "--max-batch",
        type=int,
        default=1,
        metavar="N",
        help="継続バッチング (BACKLOG.md §2) を有効にし、同時に処理する"
        " リクエスト数の上限を N にする。既定 (未指定、または 1) では"
        " 従来どおり全リクエストを直列実行する — この既定の挙動は変えない。"
        " 効くのは FallbackRunner を実際に使うリクエストだけ: runner が"
        " 元から非投機ならすべてのリクエストに、投機系 (spec/flash_spec/"
        "draft_spec/lookup_spec) が主経路なら非恒等サンプリング/logprobs で"
        " fallback へ降格されたリクエストだけに効く (投機経路自体とバッチ化"
        " の相性は未検証なので、通常の投機リクエストは従来どおり直列)。"
        " /health の queue_depth はどちらの経路でも共通の待ち行列を報告する。"
        " QSA が有効になりうるリクエスト (プロンプト長 + max_tokens が"
        " indexer_budget を超えうるもの) は正しさのため常に単独実行に自動で"
        " 倒す (mlxturbo/batch.py 参照)。ビット一致は保証しない"
        " (mx.quantized_matmul のバッチ長依存の丸めのため。"
        " tools/verify_batch_real.py --mode kld 参照) — 保証するのは"
        " 同一バッチ構成内の決定性",
    )
    ap.add_argument(
        "--mtp",
        default=None,
        metavar="PATH",
        help="MTP (multi-token prediction) ヘッドを単一 safetensors サイドカー"
        " から読み込む場合のパス。指定すると --original の生チェックポイント"
        " からの探索やバンドル済みアーティファクトより優先する。qwen4_exp"
        " (Flash-Next) 系では、明示指定したのに読み込めない場合 (1 回だけ"
        " 自動リトライした後も失敗) はフォールバックせず、理由を表示して"
        " 起動を中止する (exit 1) — 逃げ道のフラグは無い。未指定の場合は"
        " モデル本体の safetensors (mtp.* テンソル) → モデルディレクトリの"
        " mtp.safetensors サイドカーの順に自動発見する (見つかった場合は"
        " その旨をログに出す。この自動発見の失敗は exit せずフォールバック"
        " する)。どれも見つからなければフォールバックする (exit はしない —"
        " /health の fallback_reason で理由が見える)。27B (--original) 側は"
        " 従来どおり、重み欠損を吸収して MTP 無しで起動する",
    )
    ap.add_argument(
        "--require-runner",
        default=None,
        choices=sorted(RUNNER_KINDS),
        metavar="KIND",
        help="起動時に解決された runner (mlxturbo.runner の KIND: "
        + "/".join(sorted(RUNNER_KINDS))
        + ") がこれと一致しなければ、フォールバックせず理由を表示して起動を"
        " 中止する (exit 1)。未指定 (既定) では従来どおり黙って fallback を"
        " 許す — その場合も /health の fallback_reason で理由が見える",
    )
    args = ap.parse_args()

    global STATE

    served_name = args.served_model_name or Path(args.model).name

    # The model weights and the KV cache are bound to the thread that loaded
    # them (see the docstring). Every subsequent generate call is pinned to
    # this thread too, so the load itself is done here. With max_workers=1
    # there is only one thread, reused for the lifetime of the process.
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="mlxturbo-mlx"
    )

    def _load():
        t0 = time.perf_counter()
        if args.ngram:
            # NGRAM_ON_DISK in qwen4_exp.py is evaluated at module import time,
            # so it has to be set before the load call (the same reason as in
            # cli.py).
            os.environ["FASTMLX_NGRAM_DISK"] = "1"
        if args.mtp:
            # build_runner (mlxturbo/runner.py) calls load_cli_mtp with fixed
            # positional arguments only, so there is no way to pass the --mtp
            # path directly. cli.py's load_cli_mtp reads this environment
            # variable (see MTP_PATH_ENV) — the same trick as --ngram for
            # wiring things up without editing the caller.
            os.environ[MTP_PATH_ENV] = args.mtp
        model, tokenizer, config = mlx_lm_load(args.model, return_config=True)
        # 重みを GPU に wire する。mlx_lm の stream_generate は生成のたびに
        # wired_limit() で巻くが、FlashSpecEngine/SpecEngine の経路はそれを
        # 通らないので、ここで一度だけ恒久的に設定する。wire しないと macOS が
        # ページを退避・圧縮でき、読み出しが劣化する (docs/research/
        # KERNEL-BRIEF-MOE-GDN.md: lm_head 常駐 109GB/s vs 非常駐 315GB/s)。
        try:
            rec = mx.metal.device_info()["max_recommended_working_set_size"]
            mx.set_wired_limit(rec)
            print(f"[mlxturbo-serve] wired limit を {rec / 2**30:.0f}GiB に設定")
        except Exception as e:  # noqa: BLE001  wire できない環境でも起動は続ける
            print(f"[mlxturbo-serve] wired limit 設定失敗 (続行): {e}")
        if args.ngram:
            from .ngram_stream import install

            install(model, args.ngram)
        if args.rebit:
            from . import rebit

            rebit.apply(model, args.rebit)
        print(f"[mlxturbo-serve] loaded in {time.perf_counter() - t0:.1f}s: {args.model}")
        runner = build_runner(model, tokenizer, config, args, log_prefix="[mlxturbo-serve]")
        _enforce_required_runner(runner, args.require_runner)
        if args.max_context_tokens is not None:
            max_context_tokens, source = args.max_context_tokens, "--max-context-tokens"
        else:
            from_config = _resolve_model_max_context(config)
            from_metal = _metal_safe_prefill_limit(config)
            if from_config is None and from_metal is None:
                max_context_tokens, source = None, None
            elif from_metal is None or (from_config is not None and from_config <= from_metal):
                max_context_tokens, source = from_config, "config"
            else:
                max_context_tokens, source = from_metal, "Metal 一括確保上限から逆算"
        # If the runner is already non-speculative (the fallback) there is no
        # room to downgrade, so nothing is constructed. Only for the
        # speculative ones (spec/flash_spec) do we keep a second FallbackRunner
        # pointing at the same model/tokenizer, for this per-request downgrade
        # (item 7). Construction merely takes references and involves no GPU
        # computation, so it would also be safe to create it outside the
        # dedicated thread (after receiving this _load()'s return value).
        downgrade_runner = None if runner.KIND == FallbackRunner.KIND else FallbackRunner(
            model, tokenizer
        )
        # --max-batch (default 1 = off) builds mlxturbo.batch.BatchCoordinator
        # only when there is a plain FallbackRunner in play somewhere — see
        # maybe_build_batch_coordinator's docstring for why it must be that
        # exact class. That FallbackRunner is either the primary `runner`
        # (this model does not fit the spec/flash_spec contract at all) or,
        # when the primary is speculative, `downgrade_runner` — the same
        # object per-request downgrades (item 7) already route non-identity-
        # sampling/logprobs requests to today, wrapping the identical
        # `model`. Either way this never touches the speculative runner
        # itself: a spec/flash_spec-served request that is not downgraded
        # keeps going through STATE.lock exactly as before, unaffected by
        # --max-batch. Built inside _load() (this same executor thread)
        # because BatchGenerator's own construction touches MLX (the
        # wired-limit call), which is subject to the same thread-pinning
        # rule as everything else here.
        fallback_candidate = runner if runner.KIND == FallbackRunner.KIND else downgrade_runner
        eos_ids = set(tokenizer.eos_token_ids)
        batch_coordinator = maybe_build_batch_coordinator(
            fallback_candidate, model, executor, args.max_batch, eos_ids,
            log_prefix="[mlxturbo-serve]",
        )
        return runner, tokenizer, max_context_tokens, source, downgrade_runner, batch_coordinator

    (
        runner,
        tokenizer,
        max_context_tokens,
        source,
        downgrade_runner,
        batch_coordinator,
    ) = executor.submit(_load).result()
    if max_context_tokens is not None:
        print(
            f"[mlxturbo-serve] プロンプト長の上限: {max_context_tokens} トークン ({source})"
        )
    else:
        print(
            "[mlxturbo-serve] プロンプト長の上限: 検出できず (ガード無効 — config に"
            " max_position_embeddings が見当たらず、Metal 側の逆算もできなかった。"
            " --max-context-tokens で指定可)"
        )

    # Decide, according to the kind of runner, which class the per-conversation
    # slots hold. ChatSession (for SpecEngine) for SpecRunner, and otherwise
    # (FallbackRunner) FallbackSession (for mlx_lm's prompt_cache) — both have
    # .processed, and _select_session looks only at that, so it does not care
    # about the rest.
    session_factory = ChatSession if getattr(runner, "KIND", None) == "spec" else FallbackSession

    STATE = ModelState(
        runner=runner,
        tokenizer=tokenizer,
        session_pool=OrderedDict(),
        session_factory=session_factory,
        lock=asyncio.Lock(),
        executor=executor,
        model_name=served_name,
        model_path=args.model,
        eos_ids=set(tokenizer.eos_token_ids),
        max_tokens_cap=args.max_tokens,
        default_temp=args.temp,
        created_ts=int(time.time()),
        max_sessions=args.max_sessions,
        max_context_tokens=max_context_tokens,
        model_aliases=frozenset(args.model_alias),
        api_keys=frozenset(args.api_key),
        max_queue=args.max_queue,
        version=_FASTMLX_VERSION,
        downgrade_runner=downgrade_runner,
        max_stored_responses=args.max_stored_responses,
        batch_coordinator=batch_coordinator,
        max_batch=args.max_batch,
    )
    print(f"[mlxturbo-serve] version {_FASTMLX_VERSION}")
    if batch_coordinator is not None:
        if runner.KIND == FallbackRunner.KIND:
            print(f"[mlxturbo-serve] 継続バッチング: 有効 (--max-batch {args.max_batch})")
        else:
            print(
                f"[mlxturbo-serve] 継続バッチング: 有効 (--max-batch {args.max_batch})、"
                f" ただし runner={runner.KIND} なので効くのは非恒等サンプリング/logprobs で"
                " fallback へ降格されたリクエストだけ (投機経路の通常リクエストは"
                " 従来どおり直列)"
            )
    elif args.max_batch > 1:
        print(
            f"[mlxturbo-serve] --max-batch {args.max_batch} が指定されましたが無効"
            " (継続バッチングが使える runner が見当たらない)"
        )
    if downgrade_runner is not None:
        print(
            "[mlxturbo-serve] 非恒等サンプリングパラメータ/logprobs 要求時のリクエスト単位"
            " 降格: 有効 (fallback runner へ、400 で拒否する代わりに処理する)"
        )
    else:
        print(
            "[mlxturbo-serve] リクエスト単位降格: 対象外 (runner が既に非投機のため)"
        )
    print(
        f"[mlxturbo-serve] served model name: {served_name} "
        f"(session pool: {session_factory.__name__}, max {args.max_sessions} 会話)"
    )
    print(f"[mlxturbo-serve] 待ち行列の上限 (--max-queue): {args.max_queue}")
    if args.api_key:
        print(f"[mlxturbo-serve] API キー認証: 有効 ({len(args.api_key)} 件)")
    else:
        print("[mlxturbo-serve] API キー認証: 無効 (既定、--api-key 未指定)")
    if args.model_alias:
        print(f"[mlxturbo-serve] model alias (404 を回避): {', '.join(args.model_alias)}")
    if getattr(tokenizer, "has_thinking", False):
        print(
            f"[mlxturbo-serve] thinking マーカー検出: {tokenizer.think_start!r} / "
            f"{tokenizer.think_end!r} (reasoning_content/thinking ブロック分離が有効)"
        )
    else:
        print("[mlxturbo-serve] thinking マーカー検出なし (このモデルでは分離しない)")
    if getattr(tokenizer, "has_tool_calling", False):
        print(
            f"[mlxturbo-serve] tool call マーカー検出: {tokenizer.tool_call_start!r} / "
            f"{tokenizer.tool_call_end!r} (tools/tool_choice に対応)"
        )
    else:
        print(
            "[mlxturbo-serve] tool call マーカー検出なし (このモデルは tool calling 非対応: "
            "tools を渡すリクエストは 400 になる)"
        )

    if args.allowed_origins:
        origins = [o.strip() for o in args.allowed_origins.split(",") if o.strip()]
        _add_cors_middleware(app, origins)
        print(f"[mlxturbo-serve] CORS 許可 origin: {', '.join(origins)}")
    else:
        print("[mlxturbo-serve] CORS 無効 (既定、ローカル専用)")

    import uvicorn

    config = uvicorn.Config(app, host=args.host, port=args.port)
    server_obj = uvicorn.Server(config)
    _install_graceful_shutdown(server_obj)
    server_obj.run()


if __name__ == "__main__":
    main()
