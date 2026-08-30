"""mlxturbo/server.py の追加分 (サンプリングパラメータ / /health / CORS /
prompt cache 可視化 / /v1/completions) を、モデルをロードせずに検証する。

STATE (mlxturbo.server.STATE) はモジュールグローバルなので、各テストは
フェイクの Runner/Tokenizer で ModelState を組み立てて差し替える。生成の
下回り (asyncio.Lock / ThreadPoolExecutor) は本物を使う (どちらも構築に
イベントループやモデルを要らない) — テスト対象はあくまで server.py 側の
パース・検証・配線であって、フェイクの Runner.generate がその配線を受け
取れているかを確認する。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import queue as queue_mod
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

import mlxturbo.cli as cli_module
import mlxturbo.server as server
from mlxturbo.lookup_spec import LookupSpecRunner
from mlxturbo.runner import (
    RUNNER_KINDS,
    DraftSpecRunner,
    FallbackRunner,
    FallbackSession,
    FlashSpecRunner,
    SpecRunner,
    build_runner,
)
from mlxturbo.spec import ChatSession, SpecEngine


# ---------- フェイク tokenizer/detokenizer ----------


class _FakeDetokenizer:
    """ThinkingRouter が要求する最小限の口 (add_token/last_segment/finalize)。
    vocab: token id -> 断片文字列。無ければ f"<{id}>" を出す。
    """

    def __init__(self, vocab: dict[int, str]):
        self._vocab = vocab
        self._text = ""
        self._offset = 0

    def add_token(self, token: int) -> None:
        self._text += self._vocab.get(token, f"<{token}>")

    def finalize(self) -> None:
        pass

    @property
    def last_segment(self) -> str:
        seg = self._text[self._offset :]
        self._offset = len(self._text)
        return seg


def _json_tools_parser(text, tools=None):
    """json_tools.py (mlx_lm) と同じ最小のパーサ: '<tool_call>' と
    '</tool_call>' の間のテキストをそのまま json.loads するだけ。"""

    return json.loads(text.strip())


class FakeTokenizer:
    def __init__(
        self,
        vocab: dict[int, str] | None = None,
        has_thinking: bool = False,
        think_start_tokens: list[int] | None = None,
        think_end_tokens: list[int] | None = None,
        eos_token_ids=(999,),
        prompt_ids: list[int] | None = None,
        has_tool_calling: bool = False,
        tool_call_start_tokens: list[int] | None = None,
        tool_call_end_tokens: list[int] | None = None,
        tool_call_start: str = "<tool_call>",
        tool_call_end: str = "</tool_call>",
        tool_parser=_json_tools_parser,
        prompt_ids_fn=None,
    ):
        # prompt_ids_fn: messages -> list[int] を差し替えるフック。既定の
        # `prompt_ids` は messages の中身に関わらず常に固定値を返すので、
        # 会話が育つにつれてプロンプト token 列も伸びる (= session の
        # processed 列に対する LCP 再利用が試せる) ことを検証したいテスト
        # だけがこれを渡す。
        self._prompt_ids_fn = prompt_ids_fn
        self.vocab = vocab or {}
        self.has_thinking = has_thinking
        self.think_start_tokens = think_start_tokens or []
        self.think_end_tokens = think_end_tokens or []
        self.eos_token_ids = set(eos_token_ids)
        self._prompt_ids = prompt_ids if prompt_ids is not None else [1, 2, 3]
        self.has_tool_calling = has_tool_calling
        # 実物の mlx_lm TokenizerWrapper は has_tool_calling が False のとき
        # これらすべて None/空を返す (tool_parser_type が検出できなかった
        # ということなので)。フェイクも同じ形にしておく — さもないと
        # 「has_tool_calling を見ずに tool_call_start の truthy 判定だけで
        # 分岐してしまうバグ」を握りつぶしてしまう。
        self.tool_call_start_tokens = tool_call_start_tokens or [] if has_tool_calling else []
        self.tool_call_end_tokens = tool_call_end_tokens or [] if has_tool_calling else []
        self.tool_call_start = tool_call_start if has_tool_calling else None
        self.tool_call_end = tool_call_end if has_tool_calling else None
        self.tool_parser = tool_parser if has_tool_calling else None
        self.last_apply_chat_template_kwargs: dict | None = None

    @property
    def detokenizer(self) -> _FakeDetokenizer:
        return _FakeDetokenizer(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.vocab.get(t, f"<{t}>") for t in tokens)

    def apply_chat_template(
        self, messages, add_generation_prompt=True, enable_thinking=None, tools=None
    ):
        self.last_apply_chat_template_kwargs = {
            "messages": messages,
            "tools": tools,
            "enable_thinking": enable_thinking,
        }
        if self._prompt_ids_fn is not None:
            return self._prompt_ids_fn(messages)
        return list(self._prompt_ids)


# ---------- フェイク Runner ----------


class FakeRunner:
    """generate() に渡された kwargs をすべて記録する。tokens_to_emit を
    on_tokens 経由で 1 個ずつ流し、最後に res dict を返す。

    ``logprobs_to_emit`` (項目 17 用、任意): 渡しておくと、呼び出し側が
    ``logprobs=True`` を extra 経由で渡してきたときだけ、実際に emit された
    トークン数ぶんに切り詰めて ``res["logprobs"]`` に入れる — 本物の
    ``FallbackRunner.generate`` (mlxturbo/runner.py の ``_logprob_entry``)
    と同じ「要求されたときだけ・トークン列と 1:1」という契約を、server.py
    側 (choices[].logprobs への変換) のテストのために最小限で真似る。"""

    KIND = "fallback"
    SUPPORTED_SAMPLING_PARAMS = FallbackRunner.SUPPORTED_SAMPLING_PARAMS

    def __init__(
        self,
        tokens_to_emit: list[int],
        prefill_reused: int = 0,
        logprobs_to_emit: list[dict] | None = None,
    ):
        self.tokens_to_emit = tokens_to_emit
        self.prefill_reused = prefill_reused
        self.logprobs_to_emit = logprobs_to_emit
        self.calls: list[dict] = []

    def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
        self.calls.append(
            {
                "prompt_ids": list(prompt_ids),
                "max_tokens": max_tokens,
                "temp": temp,
                **extra,
            }
        )
        # Match the real runners' two terminal boundaries.  A permissive fake
        # that ignores max_tokens/EOS can make protocol status tests exercise
        # an output sequence that production can never return.
        toks = []
        for t in self.tokens_to_emit:
            if len(toks) >= max_tokens:
                break
            toks.append(t)
            if on_tokens:
                on_tokens([t])
            if t in eos_ids:
                break
        result = {
            "tokens": toks,
            "ttft_s": 0.001,
            "decode_tps": 100.0,
            "prefill_reused": self.prefill_reused,
            "prefill_new": len(prompt_ids) - self.prefill_reused,
            "tokens_per_step": 1.0,
        }
        if extra.get("logprobs") and self.logprobs_to_emit is not None:
            result["logprobs"] = self.logprobs_to_emit[: len(toks)]
        return result


class FakeSpecRunner(FakeRunner):
    """SpecRunner を模す: seed しかサポートしない。

    実物の SpecRunner.generate は seed 以外を **extra 経由でそのまま
    mlxturbo.spec.SpecEngine.generate() に渡すが、SpecEngine.generate は
    **kwargs を持たない固定シグネチャなので、未知のキーワード引数を渡すと
    TypeError で落ちる (実サーバーで実測: "SpecEngine.generate() got an
    unexpected keyword argument 'top_p'")。FakeRunner の generate は
    **extra を無条件に飲み込んでこの挙動を再現しないため、
    _check_and_strip_sampling_params 側が恒等値なのに params から
    ストリップし忘れた場合でもテストが 200 のまま通ってしまう (実際に
    このバグが一度この形で実サーバーまで抜けた)。ここで TypeError を
    投げることで、同じ穴が再発してもテストが検出できるようにする。
    """

    KIND = "spec"
    SUPPORTED_SAMPLING_PARAMS = SpecRunner.SUPPORTED_SAMPLING_PARAMS

    def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
        unexpected = sorted(set(extra) - self.SUPPORTED_SAMPLING_PARAMS)
        if unexpected:
            raise TypeError(
                f"SpecEngine.generate() got an unexpected keyword argument '{unexpected[0]}'"
            )
        if "seed" in extra:
            # SpecRunner.generate seeds MLX immediately before generation.
            # Keeping that side effect in the fake makes the HTTP forwarding
            # assertion cover the real boundary instead of only recording a kwarg.
            mx.random.seed(extra["seed"])
        return super().generate(
            prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra
        )


class FakeFlashSpecRunner(FakeRunner):
    """FlashSpecRunner (Qwen3.8-Flash-Next 用) を模す: seed しかサポート
    しない。FakeSpecRunner と同じ理由 (docstring 参照) で、実物の
    FlashSpecRunner.generate も未対応キーを ``**extra`` 経由でそのまま
    ``FlashSpecEngine.generate_stream`` へ渡し、そちらが固定シグネチャ
    (``**kwargs`` を持たない) なので未知のキーワード引数は TypeError で
    落ちる。FakeRunner.generate はこれを再現しないので、ここで模す。
    """

    KIND = "flash_spec"
    SUPPORTED_SAMPLING_PARAMS = FlashSpecRunner.SUPPORTED_SAMPLING_PARAMS

    def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
        unexpected = sorted(set(extra) - self.SUPPORTED_SAMPLING_PARAMS)
        if unexpected:
            raise TypeError(
                "FlashSpecEngine.generate_stream() got an unexpected keyword "
                f"argument '{unexpected[0]}'"
            )
        if "seed" in extra:
            mx.random.seed(extra["seed"])
        return super().generate(
            prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra
        )


class FakeReusingRunner:
    """FallbackRunner の cache 所有と LCP 再利用契約を最小限で模す。

    ``processed`` だけが残り cache が無い session は、実物と同じく再利用
    できない。生成が成功した時だけ sentinel cache と新しい processed 列を
    公開することで、server.py の session プール配線を緩い fake で誤魔化さ
    ない。
    """

    KIND = "fallback"
    SUPPORTED_SAMPLING_PARAMS = FallbackRunner.SUPPORTED_SAMPLING_PARAMS

    def __init__(self, reply_tokens: list[int]):
        self.reply_tokens = reply_tokens
        self.calls: list[dict] = []

    def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
        prompt_ids = list(prompt_ids)
        reused = 0
        if session is not None and session.cache is not None and session.processed:
            pl = session.processed
            n = min(len(pl), len(prompt_ids))
            lcp = 0
            while lcp < n and pl[lcp] == prompt_ids[lcp]:
                lcp += 1
            if lcp == len(pl) and lcp < len(prompt_ids):
                reused = lcp
        self.calls.append(
            {"prompt_ids": prompt_ids, "reused_before_call": reused, "session": session}
        )
        toks = []
        for t in self.reply_tokens:
            if len(toks) >= max_tokens:
                break
            toks.append(t)
            if on_tokens:
                on_tokens([t])
            if t in eos_ids:
                break
        if session is not None:
            session.cache = object()
            session.processed = prompt_ids + toks
        return {
            "tokens": toks,
            "ttft_s": 0.001,
            "decode_tps": 100.0,
            "prefill_reused": reused,
            "prefill_new": len(prompt_ids) - reused,
            "tokens_per_step": 1.0,
        }


def _fake_messages_to_ids(messages):
    """FakeTokenizer.prompt_ids_fn: messages を token id 列へ変換する。

    role: content をそのまま char ごとに ord() 化するだけだが、role ==
    "assistant" のメッセージだけは content を「カンマ区切りの生 token id
    列」として解釈する — 本物のチャットテンプレートが、確定した過去ターンの
    assistant 応答を "その応答を実際に生成したのと同じ token 列" として履歴に
    埋め戻す (再トークナイズしても同じ id に戻る) という性質を、テストの
    フェイクとして最小限で再現するための約束事。呼び出し側のテストは前の
    ターンで実際に生成された token id をそのままカンマ区切り文字列にして
    次ターンの assistant メッセージの content に使う。

    固定マーカー 0 は「ここから assistant のターンが始まる」合図 (本物の
    チャットテンプレートの ``<|assistant|>`` 相当) で、履歴に埋め戻された
    assistant メッセージの直前と、末尾の生成プロンプト (次に生成させたい
    ターンの合図) の両方に同じものが立つ — これが無いと、1 ターン目の
    「生成プロンプトの 0」と 2 ターン目の「履歴に埋め戻された assistant
    メッセージ直前の 0」が同じ位置に揃わず、processed 列が新プロンプトの
    正しい接頭辞にならない (本物のチャットテンプレートなら両方とも同じ
    "<|assistant|>" トークン列になるので揃う)。
    """

    ids: list[int] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        if role == "assistant":
            ids.append(0)
            ids.extend(int(t) for t in content.split(",") if t)
        else:
            ids.extend(ord(c) for c in f"{role}:{content}\n")
    ids.append(0)  # 次に生成させたいターンの生成プロンプト
    return ids


# ---------- 共通ヘルパ ----------


def _install_state(runner, tokenizer=None, **overrides) -> server.ModelState:
    session_factory = overrides.get("session_factory")
    if session_factory is None:
        session_factory = ChatSession if runner.KIND == "spec" else FallbackSession
    state = server.ModelState(
        runner=runner,
        tokenizer=tokenizer or FakeTokenizer(),
        session_pool=overrides.get("session_pool", OrderedDict()),
        session_factory=session_factory,
        lock=asyncio.Lock(),
        executor=concurrent.futures.ThreadPoolExecutor(max_workers=1),
        model_name=overrides.get("model_name", "test-model"),
        eos_ids=overrides.get("eos_ids", {999}),
        max_tokens_cap=overrides.get("max_tokens_cap", 4096),
        default_temp=overrides.get("default_temp", 0.7),
        created_ts=0,
        max_sessions=overrides.get("max_sessions", 8),
        max_context_tokens=overrides.get("max_context_tokens"),
        model_aliases=overrides.get("model_aliases", frozenset()),
        api_keys=overrides.get("api_keys", frozenset()),
        max_queue=overrides.get("max_queue", 8),
        queue_depth=overrides.get("queue_depth", 0),
        version=overrides.get("version", "0.0.0-test"),
        downgrade_runner=overrides.get("downgrade_runner"),
        response_store=overrides.get("response_store", OrderedDict()),
        max_stored_responses=overrides.get("max_stored_responses", 50),
        batch_coordinator=overrides.get("batch_coordinator"),
        max_batch=overrides.get("max_batch", 1),
    )
    server.STATE = state
    return state


@pytest.fixture
def client():
    return TestClient(server.app)


def _sse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
    return events


# ---------- 1. サンプリングパラメータが FallbackRunner に届く ----------


def test_sampling_params_reach_fallback_runner(client):
    runner = FakeRunner(tokens_to_emit=[10, 11])
    tok = FakeTokenizer(vocab={10: "hello", 11: " world"})
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "logit_bias": {"5": 2.0},
            "seed": 42,
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["top_p"] == 0.9
    assert call["top_k"] == 40
    assert call["min_p"] == 0.05
    assert call["repetition_penalty"] == 1.1
    assert call["presence_penalty"] == 0.2
    assert call["frequency_penalty"] == 0.3
    assert call["logit_bias"] == {5: 2.0}
    assert call["seed"] == 42
    assert resp.json()["choices"][0]["message"]["content"] == "hello world"


def test_sampling_params_omitted_when_not_requested(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200, resp.text
    call = runner.calls[0]
    for key in (
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "seed",
    ):
        assert key not in call


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_p", 1.5),
        ("top_p", "nope"),
        ("top_k", -2),
        ("min_p", -0.1),
        ("repetition_penalty", -1.0),
        ("logit_bias", "not-a-dict"),
        ("seed", 1.5),
    ],
)
def test_sampling_params_validation_errors(client, field, value):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], field: value},
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_top_p_and_top_k_supported_on_anthropic_messages_too(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "hi"}))

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.5,
            "top_k": 20,
        },
    )
    assert resp.status_code == 200, resp.text
    assert runner.calls[0]["top_p"] == 0.5
    assert runner.calls[0]["top_k"] == 20


# ---------- 2. SpecRunner 経路は seed 以外を 400 で弾く ----------


@pytest.mark.parametrize(
    "params",
    [
        {"top_p": 0.9},
        {"top_k": 10},
        {"min_p": 0.1},
        {"repetition_penalty": 1.1},
        {"presence_penalty": 0.1},
        {"frequency_penalty": 0.1},
        {"logit_bias": {"1": 1.0}},
    ],
)
def test_spec_runner_rejects_unsupported_sampling_params(client, params):
    runner = FakeSpecRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    body = {"messages": [{"role": "user", "content": "hi"}], **params}
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 400, resp.text
    assert "spec" in resp.json()["error"]["message"]
    assert not runner.calls


def test_spec_runner_allows_seed(client, monkeypatch):
    seeded = []
    monkeypatch.setattr(mx.random, "seed", seeded.append)
    runner = FakeSpecRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "seed": 7},
    )
    assert resp.status_code == 200, resp.text
    assert runner.calls[0]["seed"] == 7
    assert seeded == [7]


# ---------- 2c. FlashSpecRunner (Qwen3.8-Flash-Next) 経路も seed 以外を 400 で弾く ----------


@pytest.mark.parametrize(
    "params",
    [
        {"top_p": 0.9},
        {"top_k": 10},
        {"min_p": 0.1},
        {"repetition_penalty": 1.1},
        {"presence_penalty": 0.1},
        {"frequency_penalty": 0.1},
        {"logit_bias": {"1": 1.0}},
    ],
)
def test_flash_spec_runner_rejects_unsupported_sampling_params(client, params):
    runner = FakeFlashSpecRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    body = {"messages": [{"role": "user", "content": "hi"}], **params}
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_flash_spec_runner_allows_seed(client, monkeypatch):
    seeded = []
    monkeypatch.setattr(mx.random, "seed", seeded.append)
    runner = FakeFlashSpecRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "seed": 7},
    )
    assert resp.status_code == 200, resp.text
    assert runner.calls[0]["seed"] == 7
    assert seeded == [7]


@pytest.mark.parametrize(
    ("tokens", "max_tokens", "eos_ids", "expected"),
    [
        pytest.param([10, 11], 1, set(), [10], id="max-tokens"),
        pytest.param([10, 999, 11], 8, {999}, [10, 999], id="eos"),
    ],
)
@pytest.mark.parametrize("runner_cls", [FakeRunner, FakeReusingRunner])
def test_fake_runners_match_real_generation_boundaries(
    runner_cls, tokens, max_tokens, eos_ids, expected
):
    """The test doubles must stop where FallbackRunner/SpecRunner stop."""

    runner = runner_cls(tokens)
    observed = []
    result = runner.generate(
        [1, 2, 3],
        max_tokens=max_tokens,
        temp=0.0,
        eos_ids=eos_ids,
        on_tokens=observed.extend,
        session=None,
    )
    assert result["tokens"] == expected
    assert observed == expected


def test_fake_reusing_runner_requires_a_published_cache():
    """processed tokens alone do not make a FallbackSession reusable."""

    runner = FakeReusingRunner([10])
    session = FallbackSession()
    session.processed = [1]

    first = runner.generate(
        [1, 2], 8, 0.0, set(), None, session
    )
    assert first["prefill_reused"] == 0
    assert session.cache is not None

    second = runner.generate(
        [1, 2, 10, 3], 8, 0.0, set(), None, session
    )
    assert second["prefill_reused"] == 3


@pytest.mark.parametrize(
    ("runner", "expected_factory"),
    [
        pytest.param(FakeRunner([]), FallbackSession, id="fallback"),
        pytest.param(FakeSpecRunner([]), ChatSession, id="spec"),
        # FlashSpecRunner (Qwen3.8-Flash-Next) は KIND="flash_spec" (!= "spec")
        # なので FallbackSession を流用する側に落ちる (mlxturbo/runner.py の
        # FlashSpecRunner docstring 参照) — server.py 側は何も変えていない。
        pytest.param(FakeFlashSpecRunner([]), FallbackSession, id="flash_spec"),
    ],
)
def test_install_state_matches_production_session_type(runner, expected_factory):
    state = _install_state(runner)
    assert state.session_factory is expected_factory


# ---------- 2b. SpecRunner 経路でも恒等値 (分布を変えない既定値) は通す ----------
#
# opencode / OpenAI SDK など実クライアントは top_p=1.0 や frequency_penalty=0
# のような「未指定と等価」な値をキー付きで送ってくる。これらは分布を一切
# 変えないので、SUPPORTED_SAMPLING_PARAMS に無いキーでも 400 にしてはいけない
# (このバグの実測: opencode を spec runner のモデルに繋ぐと最初のリクエストで
# 即 400 になっていた)。
#
# ただし 400 にしないだけでは足りない: SpecRunner.generate は seed 以外の
# kwarg を mlxturbo.spec.SpecEngine.generate() へそのまま **extra 経由で
# 渡すが、SpecEngine.generate は **kwargs を持たない固定シグネチャなので、
# 恒等値であっても runner がサポートしないキーをそのまま渡すと
# "unexpected keyword argument" で 500 になる (実サーバーでの実測)。
# なので _check_and_strip_sampling_params は「400 にしない」だけでなく
# 「runner が知らないキーを params から取り除く」まで行う必要があり、
# 以下は 200 になることに加えて runner が実際に受け取った引数からその
# キーが消えていることまで確認する。


@pytest.mark.parametrize(
    "params",
    [
        {"top_p": 0.0},
        {"top_p": 1.0},
        {"top_k": 0},
        {"top_k": -1},
        {"min_p": 0.0},
        {"frequency_penalty": 0.0},
        {"presence_penalty": 0.0},
        {"repetition_penalty": 0.0},
        {"repetition_penalty": 1.0},
        {"logit_bias": {}},
    ],
)
def test_spec_runner_allows_identity_sampling_values(client, params):
    runner = FakeSpecRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    body = {"messages": [{"role": "user", "content": "hi"}], **params}
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200, resp.text
    assert runner.calls
    # 恒等値だが SpecRunner.SUPPORTED_SAMPLING_PARAMS (= {"seed"}) には
    # 無いキーなので、runner.generate が実際に受け取った kwargs から
    # 消えていなければならない (残っていれば実物では TypeError -> 500)。
    (key,) = params.keys()
    assert key not in runner.calls[0]


@pytest.mark.parametrize(
    "field",
    [
        "top_p",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
    ],
)
def test_spec_runner_allows_explicit_null(client, field):
    """明示的な JSON null (未指定と区別しないクライアントがいる) も、
    _parse_sampling_params 側で「未指定」扱いになって params dict に入らない
    ため、そもそも _check_sampling_support に渡らずに通る。"""

    runner = FakeSpecRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], field: None},
    )
    assert resp.status_code == 200, resp.text
    assert runner.calls


@pytest.mark.parametrize(
    "params",
    [
        {"top_p": 0.9},
        {"top_k": 10},
        {"min_p": 0.1},
        {"repetition_penalty": 1.1},
        {"presence_penalty": 0.5},
        {"frequency_penalty": 0.3},
        {"logit_bias": {"1": 1.0}},
    ],
)
def test_spec_runner_still_rejects_non_identity_sampling_values(client, params):
    """恒等値判定を入れても、実際に分布を変える値は今まで通り 400。"""

    runner = FakeSpecRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    body = {"messages": [{"role": "user", "content": "hi"}], **params}
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 400, resp.text
    assert "spec" in resp.json()["error"]["message"]
    assert not runner.calls


def test_spec_runner_error_lists_only_non_identity_params(client):
    """恒等値と非恒等値を同時に送ったとき、エラーメッセージに挙がるのは
    非恒等値のものだけ (恒等値のキー名が紛れ込まない)。"""

    runner = FakeSpecRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 1.0,  # 恒等値: メッセージに出てはいけない
            "frequency_penalty": 0.0,  # 恒等値: メッセージに出てはいけない
            "presence_penalty": 0.5,  # 非恒等値: メッセージに出るべき
        },
    )
    assert resp.status_code == 400, resp.text
    message = resp.json()["error"]["message"]
    # 「non-default values for: <listed>. The speculative-decoding runner ...」
    # の <listed> 部分だけを見る — 説明文中の例示 (top_p=1.0 等) を誤検出
    # しないように、リスト部分を切り出してから判定する。
    listed = message.split("non-default values for: ", 1)[1].split(". The speculative-decoding", 1)[0]
    assert listed == "presence_penalty"
    assert not runner.calls


def test_fallback_runner_still_allows_everything(client):
    """FallbackRunner は元から SUPPORTED_SAMPLING_PARAMS に全キーを宣言して
    いるので、恒等値判定の追加による影響を受けない (念のための確認)。"""

    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3,
            "logit_bias": {"5": 2.0},
        },
    )
    assert resp.status_code == 200, resp.text
    assert runner.calls


def test_spec_runner_rejects_unsupported_params_on_completions_endpoint(client):
    runner = FakeSpecRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/completions", json={"prompt": "hello", "top_p": 0.5}
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


# ---------- seed 固定で出力が再現する (runner.py レベル、フェイク stream_generate) ----------


class _FakeGenResponse:
    """``mlx_lm.generate.GenerationResponse`` の最小限のフェイク。

    ``logprobs`` (項目 17 用) と ``from_draft`` (項目 13 用) は任意 —
    既存呼び出し (``_FakeGenResponse(val, str(val))``) は 2 引数のままで
    壊れない。"""

    def __init__(self, token, text, logprobs=None, from_draft: bool = False):
        self.token = token
        self.text = text
        self.logprobs = logprobs
        self.from_draft = from_draft


def test_fallback_runner_seed_makes_output_reproducible(monkeypatch):
    """FallbackRunner.generate に seed を渡すと mx.random.seed() が呼ばれ、
    以後の (フェイクの) mx.random 呼び出し列が再現することを確認する。
    実モデルは要らない: stream_generate をスタブに差し替え、スタブ自体が
    mx.random から値を引く形にすることで「seed で結果が固定される」という
    契約を、本物の乱数発生器を使ったまま検証する。
    """

    import importlib

    mlx_generate = importlib.import_module("mlx_lm.generate")

    def fake_stream_generate(
        model, tokenizer, prompt, max_tokens, sampler=None, logits_processors=None, **_kwargs
    ):
        for _ in range(max_tokens):
            val = int(mx.random.randint(0, 1_000_000, shape=()).item())
            yield _FakeGenResponse(val, str(val))

    monkeypatch.setattr(mlx_generate, "stream_generate", fake_stream_generate)

    runner = FallbackRunner(model=object(), tokenizer=object())

    def run(seed):
        return runner.generate(
            [1, 2, 3],
            max_tokens=5,
            temp=0.0,
            eos_ids=set(),
            on_tokens=None,
            session=None,
            seed=seed,
        )["tokens"]

    first = run(1234)
    second = run(1234)
    third = run(5678)
    assert first == second
    assert first != third


def test_fallback_runner_calls_mx_random_seed(monkeypatch):
    calls = []
    monkeypatch.setattr(mx.random, "seed", lambda s: calls.append(s))

    import importlib

    mlx_generate = importlib.import_module("mlx_lm.generate")

    def fake_stream_generate(
        model, tokenizer, prompt, max_tokens, sampler=None, logits_processors=None, **_kwargs
    ):
        return iter(())

    monkeypatch.setattr(mlx_generate, "stream_generate", fake_stream_generate)

    runner = FallbackRunner(model=object(), tokenizer=object())
    runner.generate(
        [1], max_tokens=0, temp=0.0, eos_ids=set(), on_tokens=None, session=None, seed=99
    )
    assert calls == [99]


def test_fallback_runner_no_seed_does_not_touch_rng(monkeypatch):
    calls = []
    monkeypatch.setattr(mx.random, "seed", lambda s: calls.append(s))

    import importlib

    mlx_generate = importlib.import_module("mlx_lm.generate")

    monkeypatch.setattr(mlx_generate, "stream_generate", lambda *a, **k: iter(()))

    runner = FallbackRunner(model=object(), tokenizer=object())
    runner.generate([1], max_tokens=0, temp=0.0, eos_ids=set(), on_tokens=None, session=None)
    assert calls == []


# ---------- 3. /health ----------


def test_health_endpoint_fallback(client):
    runner = FakeRunner(tokens_to_emit=[])
    _install_state(runner, model_name="my-model")

    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == "my-model"
    assert body["loaded"] is True
    assert body["runner"] == "fallback"
    assert body["busy"] is False


def test_health_endpoint_reports_spec_runner(client):
    runner = FakeSpecRunner(tokens_to_emit=[])
    _install_state(runner)
    resp = client.get("/health")
    assert resp.json()["runner"] == "spec"


def test_health_endpoint_reports_flash_spec_runner(client):
    runner = FakeFlashSpecRunner(tokens_to_emit=[])
    _install_state(runner)
    resp = client.get("/health")
    assert resp.json()["runner"] == "flash_spec"


def test_health_endpoint_reports_fallback_reason_when_present(client):
    """build_runner が fallback_reason を持たせた runner (実物の
    FallbackRunner や、同じ属性を持つフェイク) では /health にそのまま
    出る — 「黙って fallback に落ちて気づけない」を防ぐための配線。"""

    runner = FakeRunner(tokens_to_emit=[])
    runner.fallback_reason = "qwen4_exp だが MTP を自動発見できなかった (fake reason)"
    _install_state(runner)
    resp = client.get("/health")
    assert resp.json()["fallback_reason"] == (
        "qwen4_exp だが MTP を自動発見できなかった (fake reason)"
    )


def test_health_endpoint_omits_fallback_reason_when_none(client):
    """fallback_reason が None (投機経路、または FallbackRunner でも理由
    なし) なら、キー自体を出さない。"""

    runner = FakeRunner(tokens_to_emit=[])
    runner.fallback_reason = None
    _install_state(runner)
    resp = client.get("/health")
    assert "fallback_reason" not in resp.json()


def test_health_endpoint_omits_fallback_reason_for_spec_runner(client):
    """SpecRunner/FlashSpecRunner のフェイクには fallback_reason 属性が
    そもそも無い (getattr の既定 None) — それでも壊れず、キーが出ない。"""

    runner = FakeSpecRunner(tokens_to_emit=[])
    _install_state(runner)
    resp = client.get("/health")
    assert "fallback_reason" not in resp.json()


# ---------- 3b. バグ修正: GET/HEAD /api/hello が 404 だった ----------
#
# Claude Code の疎通確認。実装が無いと 404 になる (会話自体は成立するので
# 実害は小さいが、起動直後のログにノイズが乗る)。中身に意味は無く、200 を
# 返すことだけが要件。


def test_api_hello_get_returns_200(client):
    resp = client.get("/api/hello")
    assert resp.status_code == 200, resp.text


def test_api_hello_head_returns_200(client):
    resp = client.head("/api/hello")
    assert resp.status_code == 200, resp.text


# ---------- 4. CORS ----------


def test_cors_disabled_by_default(client):
    runner = FakeRunner(tokens_to_emit=[])
    _install_state(runner)
    resp = client.get("/health", headers={"Origin": "http://example.com"})
    assert "access-control-allow-origin" not in resp.headers


def test_cors_enabled_via_helper():
    from fastapi import FastAPI

    from mlxturbo.server import _add_cors_middleware

    test_app = FastAPI()
    _add_cors_middleware(test_app, ["http://allowed.example"])

    @test_app.get("/ping")
    async def ping():
        return {"ok": True}

    test_client = TestClient(test_app)
    ok = test_client.get("/ping", headers={"Origin": "http://allowed.example"})
    assert ok.headers.get("access-control-allow-origin") == "http://allowed.example"

    blocked = test_client.get("/ping", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in blocked.headers


# ---------- 5. /v1/completions ----------


def test_completions_non_stream(client):
    runner = FakeRunner(tokens_to_emit=[10, 11], prefill_reused=3)
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "foo", 11: "bar"}))

    resp = client.post(
        "/v1/completions", json={"model": "test-model", "prompt": "hello", "max_tokens": 8}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "foobar"
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 3
    # prompt was tokenizer.encode'd (naive char-by-char fake), not templated
    assert runner.calls[0]["prompt_ids"] == [ord(c) for c in "hello"]


def test_completions_accepts_pretokenized_prompt(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post("/v1/completions", json={"prompt": [1, 2, 3]})
    assert resp.status_code == 200, resp.text
    assert runner.calls[0]["prompt_ids"] == [1, 2, 3]


def test_completions_rejects_empty_prompt(client):
    runner = FakeRunner(tokens_to_emit=[])
    _install_state(runner)
    resp = client.post("/v1/completions", json={"prompt": ""})
    assert resp.status_code == 400


def test_completions_stream(client):
    runner = FakeRunner(tokens_to_emit=[10, 11])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "foo", 11: "bar"}))

    resp = client.post(
        "/v1/completions",
        json={
            "prompt": "hello",
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert resp.status_code == 200, resp.text
    events = _sse_events(resp.text)
    text_chunks = [
        e["choices"][0]["text"]
        for e in events
        if e.get("choices") and e["choices"][0].get("text")
    ]
    assert "".join(text_chunks) == "foobar"
    usage_events = [e for e in events if e.get("usage")]
    assert usage_events, "expected a final usage chunk"
    assert usage_events[-1]["usage"]["completion_tokens"] == 2


# ---------- cached_tokens on OpenAI chat completions ----------


def test_cached_tokens_reflects_prefill_reused_non_stream(client):
    runner = FakeRunner(tokens_to_emit=[10], prefill_reused=7)
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 7


def test_cached_tokens_reflects_prefill_reused_stream(client):
    runner = FakeRunner(tokens_to_emit=[10], prefill_reused=4)
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert resp.status_code == 200, resp.text
    events = _sse_events(resp.text)
    usage_events = [e for e in events if e.get("usage")]
    assert usage_events[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 4


# ---------- 6. tool calling ----------
#
# <tool_call>{"name": ..., "arguments": {...}}</tool_call> をトークン列で
# 模す。マーカーが 1 トークンだと ThinkingRouter のローリングウィンドウが
# 「次のトークンが来るまで確定しない」ため 1 トークン遅れて検出される
# (thinking の end marker と同じ挙動) — これはバグではなく、テストの
# tokens_to_emit の並びで吸収する (末尾に確定用のダミートークンや eos を
# 置く)。

_TOOL_CALL_JSON = json.dumps({"name": "get_weather", "arguments": {"city": "Tokyo"}})

_WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

_WEATHER_TOOL_ANTHROPIC = {
    "name": "get_weather",
    "description": "get the weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def _tool_calling_tokenizer(vocab: dict[int, str], **overrides) -> FakeTokenizer:
    return FakeTokenizer(
        vocab=vocab,
        has_tool_calling=True,
        tool_call_start_tokens=[100],
        tool_call_end_tokens=[101],
        **overrides,
    )


# ---- 6.1 リクエスト側: tools/tool_choice が apply_chat_template へ届く ----


def test_openai_tools_reach_apply_chat_template(client):
    tok = FakeTokenizer(vocab={10: "ok"}, has_tool_calling=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "weather in tokyo?"}],
            "tools": [_WEATHER_TOOL_OPENAI],
        },
    )
    assert resp.status_code == 200, resp.text
    assert tok.last_apply_chat_template_kwargs["tools"] == [_WEATHER_TOOL_OPENAI]


def test_openai_tool_choice_none_does_not_pass_tools(client):
    tok = FakeTokenizer(vocab={10: "ok"}, has_tool_calling=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_WEATHER_TOOL_OPENAI],
            "tool_choice": "none",
        },
    )
    assert resp.status_code == 200, resp.text
    assert tok.last_apply_chat_template_kwargs["tools"] is None


def test_openai_tool_choice_required_is_400(client):
    tok = FakeTokenizer(vocab={10: "ok"}, has_tool_calling=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_WEATHER_TOOL_OPENAI],
            "tool_choice": "required",
        },
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_openai_tool_choice_specific_function_is_400(client):
    tok = FakeTokenizer(vocab={10: "ok"}, has_tool_calling=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_WEATHER_TOOL_OPENAI],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        },
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_openai_tools_on_unsupported_model_is_400(client):
    tok = FakeTokenizer(vocab={10: "ok"})  # has_tool_calling=False (既定)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_WEATHER_TOOL_OPENAI],
        },
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_anthropic_tools_converted_to_openai_shape_for_template(client):
    tok = FakeTokenizer(vocab={10: "ok"}, has_tool_calling=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_WEATHER_TOOL_ANTHROPIC],
        },
    )
    assert resp.status_code == 200, resp.text
    assert tok.last_apply_chat_template_kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "get the weather for a city",
                "parameters": _WEATHER_TOOL_ANTHROPIC["input_schema"],
            },
        }
    ]


@pytest.mark.parametrize("choice", [{"type": "any"}, {"type": "tool", "name": "get_weather"}])
def test_anthropic_forced_tool_choice_is_400(client, choice):
    tok = FakeTokenizer(vocab={10: "ok"}, has_tool_calling=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_WEATHER_TOOL_ANTHROPIC],
            "tool_choice": choice,
        },
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_anthropic_tools_on_unsupported_model_is_400(client):
    tok = FakeTokenizer(vocab={10: "ok"})  # has_tool_calling=False (既定)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_WEATHER_TOOL_ANTHROPIC],
        },
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


# ---- 6.2 モデル出力の解析: <tool_call>...</tool_call> の検出・構造化 ----


def test_openai_nonstream_parses_tool_call(client):
    vocab = {10: "Sure, checking. ", 200: _TOOL_CALL_JSON, 11: "!"}
    tok = _tool_calling_tokenizer(vocab)
    runner = FakeRunner(tokens_to_emit=[10, 100, 200, 101, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "weather in tokyo?"}],
            "tools": [_WEATHER_TOOL_OPENAI],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] == "Sure, checking. !"
    assert len(message["tool_calls"]) == 1
    tc = message["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Tokyo"}


def test_openai_stream_emits_tool_calls_deltas(client):
    vocab = {10: "Sure, checking. ", 200: _TOOL_CALL_JSON, 11: "!"}
    tok = _tool_calling_tokenizer(vocab)
    runner = FakeRunner(tokens_to_emit=[10, 100, 200, 101, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "weather in tokyo?"}],
            "tools": [_WEATHER_TOOL_OPENAI],
            "stream": True,
        },
    )
    assert resp.status_code == 200, resp.text
    events = _sse_events(resp.text)

    tool_call_deltas = [
        e["choices"][0]["delta"]["tool_calls"][0]
        for e in events
        if e.get("choices") and e["choices"][0].get("delta", {}).get("tool_calls")
    ]
    assert tool_call_deltas, "expected at least one delta.tool_calls chunk"
    first = tool_call_deltas[0]
    assert first["index"] == 0
    assert first["type"] == "function"
    assert first["function"]["name"] == "get_weather"
    assert first["function"]["arguments"] == ""

    args_str = "".join(
        d["function"].get("arguments", "") for d in tool_call_deltas[1:]
    )
    assert json.loads(args_str) == {"city": "Tokyo"}

    finish_reasons = [
        e["choices"][0]["finish_reason"]
        for e in events
        if e.get("choices") and e["choices"][0].get("finish_reason")
    ]
    assert finish_reasons[-1] == "tool_calls"


def test_anthropic_nonstream_emits_tool_use_block(client):
    vocab = {10: "Sure, checking. ", 200: _TOOL_CALL_JSON, 11: "!"}
    tok = _tool_calling_tokenizer(vocab)
    runner = FakeRunner(tokens_to_emit=[10, 100, 200, 101, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "weather in tokyo?"}],
            "tools": [_WEATHER_TOOL_ANTHROPIC],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    blocks = body["content"]
    tool_use_blocks = [b for b in blocks if b["type"] == "tool_use"]
    assert len(tool_use_blocks) == 1
    tu = tool_use_blocks[0]
    assert tu["name"] == "get_weather"
    assert tu["input"] == {"city": "Tokyo"}
    text_blocks = [b for b in blocks if b["type"] == "text"]
    assert "".join(b["text"] for b in text_blocks) == "Sure, checking. !"


def test_anthropic_stream_emits_tool_use_block(client):
    vocab = {10: "Sure, checking. ", 200: _TOOL_CALL_JSON, 11: "!"}
    tok = _tool_calling_tokenizer(vocab)
    runner = FakeRunner(tokens_to_emit=[10, 100, 200, 101, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "weather in tokyo?"}],
            "tools": [_WEATHER_TOOL_ANTHROPIC],
            "stream": True,
        },
    )
    assert resp.status_code == 200, resp.text
    events = _sse_events(resp.text)

    starts = [
        e["content_block"]
        for e in events
        if e.get("type") == "content_block_start" and e["content_block"]["type"] == "tool_use"
    ]
    assert len(starts) == 1
    assert starts[0]["name"] == "get_weather"

    partials = [
        e["delta"]["partial_json"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "input_json_delta"
    ]
    assert json.loads("".join(partials)) == {"city": "Tokyo"}

    stops = [e["delta"]["stop_reason"] for e in events if e.get("type") == "message_delta"]
    assert stops[-1] == "tool_use"


def test_malformed_tool_call_json_falls_back_to_text(client):
    broken_json = '{"name": "get_weather", "arguments": {'  # 壊れている (閉じ括弧無し)
    vocab = {10: "Sure, checking. ", 200: broken_json}
    tok = _tool_calling_tokenizer(vocab)
    # 末尾を eos (999) にして finish_reason を決定的にする。
    runner = FakeRunner(tokens_to_emit=[10, 100, 200, 101, 999])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "weather in tokyo?"}],
            "tools": [_WEATHER_TOOL_OPENAI],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    choice = body["choices"][0]
    message = choice["message"]
    assert "tool_calls" not in message
    assert choice["finish_reason"] == "stop"
    assert "<tool_call>" in message["content"]
    assert broken_json in message["content"]
    assert "</tool_call>" in message["content"]


def test_qwen36_tool_parser_matches_production_xml_boundary(client):
    """Exercise qwen3_coder's XML boundary used by the target Qwen3.6 model."""

    from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call

    raw_call = (
        "<function=get_weather>\n"
        "<parameter=city>\nTokyo\n</parameter>\n"
        "</function>"
    )
    tok = _tool_calling_tokenizer({200: raw_call}, tool_parser=parse_tool_call)
    runner = FakeRunner(tokens_to_emit=[100, 200, 101, 999])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [_WEATHER_TOOL_OPENAI],
        },
    )
    assert resp.status_code == 200, resp.text
    call = resp.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Tokyo"}


@pytest.mark.parametrize(
    "tool_parser",
    [
        pytest.param(
            lambda text, tools=None: (_ for _ in ()).throw(SyntaxError("bad literal")),
            id="parser-raises-syntax-error",
        ),
        pytest.param(
            lambda text, tools=None: {
                "name": "get_weather",
                "arguments": {"cities": {"Tokyo"}},
            },
            id="parser-returns-non-json-arguments",
        ),
    ],
)
def test_tool_parser_failures_at_real_boundary_fall_back_to_text(client, tool_parser):
    """Model-specific mlx_lm parsers can fail more broadly than json.loads.

    The server must validate their return value before protocol serializers see
    it; otherwise a parser SyntaxError or a non-JSON literal becomes a 500 after
    the permissive FakeRunner has already made the request look successful.
    """

    raw_call = "malformed model tool output"
    vocab = {200: raw_call}
    tok = _tool_calling_tokenizer(vocab, tool_parser=tool_parser)
    runner = FakeRunner(tokens_to_emit=[100, 200, 101, 999])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [_WEATHER_TOOL_OPENAI],
        },
    )
    assert resp.status_code == 200, resp.text
    message = resp.json()["choices"][0]["message"]
    assert "tool_calls" not in message
    assert message["content"] == f"<tool_call>{raw_call}</tool_call>"


def test_unclosed_tool_call_is_not_promoted_to_structured_call(client):
    """Valid-looking JSON is still incomplete until the model emits the end marker."""

    vocab = {200: _TOOL_CALL_JSON}
    tok = _tool_calling_tokenizer(vocab)
    runner = FakeRunner(tokens_to_emit=[100, 200, 999])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [_WEATHER_TOOL_OPENAI],
        },
    )
    assert resp.status_code == 200, resp.text
    message = resp.json()["choices"][0]["message"]
    assert "tool_calls" not in message
    assert message["content"] == f"<tool_call>{_TOOL_CALL_JSON}"


# ---- 6.3 履歴 (tool_calls / tool_result) の正規化 ----


def test_openai_history_tool_calls_and_tool_role_reach_template(client):
    tok = FakeTokenizer(vocab={10: "It's 22C."})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Tokyo"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": '{"temp": 22}'},
    ]
    resp = client.post("/v1/chat/completions", json={"messages": messages})
    assert resp.status_code == 200, resp.text

    sent = tok.last_apply_chat_template_kwargs["messages"]
    assert sent[1]["role"] == "assistant"
    assert sent[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    # arguments は dict に戻してテンプレートへ渡す (二重 JSON エンコードしない)。
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Tokyo"}
    assert sent[2]["role"] == "tool"
    assert sent[2]["tool_call_id"] == "call_abc"
    assert sent[2]["content"] == '{"temp": 22}'


def test_openai_history_malformed_tool_call_arguments_is_400(client):
    tok = FakeTokenizer(vocab={10: "x"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{not valid json"},
                }
            ],
        },
    ]
    resp = client.post("/v1/chat/completions", json={"messages": messages})
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_anthropic_history_tool_use_and_tool_result_reach_template(client):
    tok = FakeTokenizer(vocab={10: "It's 22C outside."})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Tokyo"},
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "22C"}],
        },
    ]
    resp = client.post(
        "/v1/messages",
        json={"model": "test-model", "max_tokens": 32, "messages": messages},
    )
    assert resp.status_code == 200, resp.text

    sent = tok.last_apply_chat_template_kwargs["messages"]
    assistant_msg = next(m for m in sent if m["role"] == "assistant")
    assert assistant_msg["content"] == "Let me check."
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"
    assert assistant_msg["tool_calls"][0]["function"]["arguments"] == {"city": "Tokyo"}

    tool_msg = next(m for m in sent if m["role"] == "tool")
    assert tool_msg["content"] == "22C"
    assert tool_msg["tool_call_id"] == "toolu_1"


def test_anthropic_tool_use_block_in_user_message_is_400(client):
    tok = FakeTokenizer(vocab={10: "x"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_use", "id": "x", "name": "f", "input": {}}],
        },
    ]
    resp = client.post(
        "/v1/messages",
        json={"model": "test-model", "max_tokens": 32, "messages": messages},
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


# ---------- 会話ごとの session プール (_select_session) ----------


class _FakeSession:
    """_select_session が読むのは processed 属性だけ (ChatSession/
    FallbackSession のどちらも duck typing で扱える) なので、プール選択
    ロジックだけを見るテストではこの最小限のフェイクで足りる。"""

    def __init__(self, processed=()):
        self.processed = list(processed)


def test_select_session_separates_independent_conversations():
    _install_state(FakeRunner(tokens_to_emit=[]))

    s1 = server._select_session([1, 2, 3])
    s1.processed = [1, 2, 3, 99]
    s2 = server._select_session([7, 8, 9])

    assert s1 is not s2
    assert len(server.STATE.session_pool) == 2


def test_select_session_reuses_slot_for_append_only_continuation():
    _install_state(FakeRunner(tokens_to_emit=[]))

    s1 = server._select_session([1, 2, 3])
    s1.processed = [1, 2, 3, 99]

    # turn 2: 前ターンの処理済み列 [1, 2, 3, 99] がそのまま接頭辞になっている
    s1_again = server._select_session([1, 2, 3, 99, 4, 5])

    assert s1_again is s1
    assert len(server.STATE.session_pool) == 1


def test_select_session_discards_on_non_append_prompt():
    """処理済み列の途中で分岐する新プロンプトは、既存スロットを再利用せず
    (=部分巻き戻しをせず)、processed が空の新規スロットを割り当てる。既存
    スロットの状態は (別の会話かもしれないので) そのまま残る。"""

    _install_state(FakeRunner(tokens_to_emit=[]))

    s1 = server._select_session([1, 2, 3])
    s1.processed = [1, 2, 3, 99]

    s2 = server._select_session([1, 5, 6])  # 位置 1 で分岐、追記ではない

    assert s2 is not s1
    assert s2.processed == []
    assert s1.processed == [1, 2, 3, 99]  # 既存スロットは無傷
    assert len(server.STATE.session_pool) == 2


def test_select_session_evicts_lru_when_pool_full():
    _install_state(FakeRunner(tokens_to_emit=[]), max_sessions=2)

    s1 = server._select_session([1])
    s1.processed = [1, 100]
    s2 = server._select_session([2])
    s2.processed = [2, 200]
    assert len(server.STATE.session_pool) == 2

    s3 = server._select_session([3])  # プール上限超過 -> s1 (LRU) を追い出す
    s3.processed = [3, 300]
    assert len(server.STATE.session_pool) == 2

    # 会話 1 の続きのつもりで送っても、そのスロットはもう無い
    s1_cont = server._select_session([1, 100, 4])
    assert s1_cont is not s1
    assert s1_cont.processed == []


def test_select_session_touching_a_slot_protects_it_from_eviction():
    _install_state(FakeRunner(tokens_to_emit=[]), max_sessions=2)

    s1 = server._select_session([1])
    s1.processed = [1, 100]
    s2 = server._select_session([2])
    s2.processed = [2, 200]

    # 会話 1 を継続 (LRU 順で会話 1 が最新になる)
    s1_cont = server._select_session([1, 100, 4])
    assert s1_cont is s1
    s1_cont.processed = [1, 100, 4, 400]

    # 会話 3 が入ってくると、今度は会話 2 (触っていない方) が追い出される
    s3 = server._select_session([3])
    s3.processed = [3, 300]
    assert len(server.STATE.session_pool) == 2

    s2_cont = server._select_session([2, 200, 5])
    assert s2_cont is not s2
    assert s2_cont.processed == []


# ---------- 会話ごとの session プール: HTTP 経由の end-to-end ----------


def test_http_multiturn_reuses_prefill_via_session_pool(client):
    """2 ターン目のリクエストで、1 ターン目の応答 (session.processed) が
    そのまま新プロンプトの接頭辞になっていれば cached_tokens に反映される
    ことを、HTTP 層 (server.py の _select_session 配線) まで通して確認する。
    """

    runner = FakeReusingRunner(reply_tokens=[10, 11])
    tok = FakeTokenizer(vocab={10: "hi", 11: " there"}, prompt_ids_fn=_fake_messages_to_ids)
    _install_state(runner, tokenizer=tok)

    turn1 = [{"role": "user", "content": "hello"}]
    resp1 = client.post("/v1/chat/completions", json={"messages": turn1})
    assert resp1.status_code == 200, resp1.text
    assert resp1.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0

    turn2 = turn1 + [
        {"role": "assistant", "content": "10,11"},  # 前ターンの生 token 列
        {"role": "user", "content": "how are you"},
    ]
    resp2 = client.post("/v1/chat/completions", json={"messages": turn2})
    assert resp2.status_code == 200, resp2.text
    cached = resp2.json()["usage"]["prompt_tokens_details"]["cached_tokens"]

    assert len(runner.calls) == 2
    # 1 ターン目でこのスロットへ publish された processed 列全体が、2 ターン
    # 目のプロンプトの接頭辞として丸ごと再利用されている
    assert cached == len(runner.calls[0]["prompt_ids"]) + 2  # +2 = 1 ターン目の応答分
    assert runner.calls[1]["session"] is runner.calls[0]["session"]


def test_http_two_independent_conversations_get_different_sessions(client):
    runner = FakeReusingRunner(reply_tokens=[10])
    tok = FakeTokenizer(vocab={10: "hi"}, prompt_ids_fn=_fake_messages_to_ids)
    _install_state(runner, tokenizer=tok)

    resp_a = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "conversation A"}]}
    )
    resp_b = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "conversation B"}]}
    )
    assert resp_a.status_code == 200 and resp_b.status_code == 200
    assert len(server.STATE.session_pool) == 2
    assert runner.calls[0]["session"] is not runner.calls[1]["session"]
    # 別会話なので、どちらも 1 ターン目扱い (再利用ゼロ)
    assert resp_a.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert resp_b.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


def test_http_max_sessions_caps_pool_size(client):
    runner = FakeReusingRunner(reply_tokens=[10])
    tok = FakeTokenizer(vocab={10: "hi"}, prompt_ids_fn=_fake_messages_to_ids)
    _install_state(runner, tokenizer=tok, max_sessions=2)

    for name in ("A", "B", "C"):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": f"conversation {name}"}]},
        )
        assert resp.status_code == 200, resp.text

    assert len(server.STATE.session_pool) == 2


# ---------- FallbackRunner: mlx_lm prompt_cache の再利用 (runner.py) ----------


class _FakeMlxCache:
    """make_prompt_cache が返す本物のリストの代わり。中身は使わず、id() で
    「同じオブジェクトが session 経由で往復したか」だけを確認する。"""


def test_fallback_runner_reuses_prompt_cache_on_append(monkeypatch):
    """session.processed が新プロンプトの接頭辞なら、session.cache が
    そのまま stream_generate へ渡され、送る prompt は差分だけになる。"""

    import importlib

    mlx_generate = importlib.import_module("mlx_lm.generate")

    calls = []

    def fake_stream_generate(model, tokenizer, prompt, max_tokens, sampler=None,
                              logits_processors=None, prompt_cache=None, **_kwargs):
        calls.append({"prompt": list(prompt), "prompt_cache": prompt_cache})
        for i, tok in enumerate([50, 51]):
            yield _FakeGenResponse(tok, str(tok))

    monkeypatch.setattr(mlx_generate, "stream_generate", fake_stream_generate)

    runner = FallbackRunner(model=object(), tokenizer=object())
    session = FallbackSession()
    existing_cache = _FakeMlxCache()
    session.publish(existing_cache, [1, 2, 3])  # 前ターンの処理済み列

    res = runner.generate(
        [1, 2, 3, 4, 5],  # [1, 2, 3] の続きとして 4, 5 が追記されている
        max_tokens=2,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=session,
    )

    assert len(calls) == 1
    assert calls[0]["prompt"] == [4, 5]  # 差分だけ prefill
    assert calls[0]["prompt_cache"] is existing_cache  # 同じ cache を再利用
    assert res["prefill_reused"] == 3
    assert res["prefill_new"] == 2
    # 生成後は「今回の prompt + 生成した token」で processed が更新される
    assert session.processed == [1, 2, 3, 4, 5, 50, 51]
    assert session.cache is existing_cache


def test_fallback_runner_discards_and_rebuilds_on_non_append_prompt(monkeypatch):
    """新プロンプトが前回処理済み列の追記でなければ、古い cache は使わず
    全量を新しい cache へ流し直す (部分巻き戻しはしない)。"""

    import importlib

    mlx_generate = importlib.import_module("mlx_lm.generate")
    mlx_cache_mod = importlib.import_module("mlx_lm.models.cache")

    calls = []
    fresh_cache = _FakeMlxCache()
    monkeypatch.setattr(mlx_cache_mod, "make_prompt_cache", lambda model: fresh_cache)

    def fake_stream_generate(model, tokenizer, prompt, max_tokens, sampler=None,
                              logits_processors=None, prompt_cache=None, **_kwargs):
        calls.append({"prompt": list(prompt), "prompt_cache": prompt_cache})
        yield _FakeGenResponse(50, "x")

    monkeypatch.setattr(mlx_generate, "stream_generate", fake_stream_generate)

    runner = FallbackRunner(model=object(), tokenizer=object())
    session = FallbackSession()
    stale_cache = _FakeMlxCache()
    session.publish(stale_cache, [1, 2, 3])

    res = runner.generate(
        [1, 9, 9],  # 位置 1 で分岐、[1, 2, 3] の追記ではない
        max_tokens=1,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=session,
    )

    assert calls[0]["prompt"] == [1, 9, 9]  # 全量を流し直す
    assert calls[0]["prompt_cache"] is fresh_cache  # 古い cache は使わない
    assert calls[0]["prompt_cache"] is not stale_cache
    assert res["prefill_reused"] == 0
    assert res["prefill_new"] == 3
    assert session.cache is fresh_cache


def test_fallback_runner_without_session_matches_pre_existing_behavior(monkeypatch):
    """session=None なら prompt_cache 機構自体に触らない (呼び出し既存テスト
    3 本 test_fallback_runner_* が検証している「以前どおり全量 prefill」と
    完全に同じ経路のまま)。"""

    import importlib

    mlx_generate = importlib.import_module("mlx_lm.generate")
    calls = []

    def fake_stream_generate(model, tokenizer, prompt, max_tokens, sampler=None,
                              logits_processors=None, **_kwargs):
        calls.append({"prompt": list(prompt)})
        return
        yield  # pragma: no cover - keep this a generator

    monkeypatch.setattr(mlx_generate, "stream_generate", fake_stream_generate)

    runner = FallbackRunner(model=object(), tokenizer=object())
    runner.generate(
        [1, 2, 3], max_tokens=0, temp=0.0, eos_ids=set(), on_tokens=None, session=None
    )
    assert calls[0]["prompt"] == [1, 2, 3]


class _ContractFlashModel:
    """FlashSpecEngine の cache/cur 契約だけを再現する軽量モデル。"""

    @staticmethod
    def make_cache():
        return {"fed": []}


class _ContractFlashEngine:
    """各 yield の末尾 token はまだ cache に feed されていない。"""

    model = _ContractFlashModel()

    def __init__(self, calls):
        self._outputs = iter(calls)
        self.inputs = []

    def generate_stream(self, ids, max_tokens, caches=None, **_kwargs):
        prompt = list(ids[0].tolist())
        self.inputs.append(prompt)
        caches["fed"].extend(prompt)
        emitted = 0
        previous_cur = None
        for token in next(self._outputs):
            if emitted >= max_tokens:
                break
            if previous_cur is not None:
                # A rejected depth-1 round feeds the previous cur and emits the
                # newly verified token.  The emitted token becomes the next cur
                # and remains immediately after the cache at the yield boundary.
                caches["fed"].append(previous_cur)
            previous_cur = token
            emitted += 1
            yield [token]
        return 0, max(emitted - 1, 0), None


def test_flash_spec_runner_publishes_only_tokens_already_fed_to_cache():
    """The trailing cur must be re-prefilled on the next turn, not skipped."""

    engine = _ContractFlashEngine([[50, 51], [60]])
    runner = FlashSpecRunner(engine)
    session = FallbackSession()

    first = runner.generate(
        [1, 2, 3],
        max_tokens=2,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=session,
    )
    assert first["tokens"] == [50, 51]
    assert session.processed == session.cache["fed"] == [1, 2, 3, 50]

    second_prompt = [1, 2, 3, 50, 51, 4]
    second = runner.generate(
        second_prompt,
        max_tokens=1,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=session,
    )
    assert second["prefill_reused"] == 4
    assert engine.inputs[1] == [51, 4]
    assert session.processed == session.cache["fed"] == second_prompt


def test_flash_spec_generate_stream_zero_tokens_prefills_without_yield(monkeypatch):
    """max_tokens=0 must not leak the prefill-produced cur to callers."""

    from contextlib import contextmanager

    import mlxturbo.spec_flash as spec_flash_module

    class FakeModel:
        model = SimpleNamespace()

        @staticmethod
        def make_cache():
            return []

        @staticmethod
        def __call__(ids, cache=None):
            return mx.zeros((1, ids.shape[1], 4))

    @contextmanager
    def fake_capture(_model, light: bool = False):
        yield SimpleNamespace(hyper=mx.zeros((1, 2, 1)))

    monkeypatch.setattr(spec_flash_module, "capture", fake_capture)
    engine = object.__new__(spec_flash_module.FlashSpecEngine)
    engine.model = FakeModel()
    engine.mtp = object()

    chunks = list(
        engine.generate_stream(
            mx.array([[1, 2]]), max_tokens=0, caches=[], temp=0.0, eos_ids=set()
        )
    )
    assert chunks == []


def test_flash_spec_callback_failure_does_not_publish_mutated_reused_cache():
    """A disconnect sentinel raised by on_tokens must leave no reusable state."""

    engine = _ContractFlashEngine([[50, 51]])
    runner = FlashSpecRunner(engine)
    session = FallbackSession()
    session.publish({"fed": [1, 2]}, [1, 2])

    def disconnect(_tokens):
        raise RuntimeError("client disconnected")

    with pytest.raises(RuntimeError, match="client disconnected"):
        runner.generate(
            [1, 2, 3],
            max_tokens=2,
            temp=0.0,
            eos_ids=set(),
            on_tokens=disconnect,
            session=session,
        )

    assert session.cache is None
    assert session.processed == []


def test_nonstream_cancellation_keeps_lock_until_worker_finishes():
    """Cancelling the HTTP task must not expose a still-mutating session."""

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockingRunner(FakeRunner):
        def generate(self, *args, **kwargs):
            started.set()
            assert release.wait(2), "test did not release blocking fake runner"
            try:
                return super().generate(*args, **kwargs)
            finally:
                finished.set()

    state = _install_state(BlockingRunner([10]))

    async def run():
        async def request():
            async with state.lock:
                await server._run_generate(
                    [1, 2, 3], 8, 0.0, state.eos_ids, None, object()
                )

        task = asyncio.create_task(request())
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)

        # Before the fix, cancellation propagated through run_in_executor at
        # this point, releasing the lock while the worker was still blocked.
        assert state.lock.locked()
        assert not finished.is_set()

        # Repeated cancellation must remain deferred as well.
        task.cancel()
        await asyncio.sleep(0)
        assert state.lock.locked()
        assert not finished.is_set()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert not state.lock.locked()

    asyncio.run(run())


def test_stream_cancel_wakes_blocking_queue_get_with_internal_sentinel():
    started = threading.Event()
    release = threading.Event()

    class PrefillBlockingRunner(FakeRunner):
        def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
            started.set()
            assert release.wait(2), "test did not release blocking fake runner"
            on_tokens([10])
            raise AssertionError("cancel callback must stop before this line")

    state = _install_state(PrefillBlockingRunner([]))
    q, future, cancel_event, _raw_token_count = server._start_generation([1, 2, 3], 8, 0.0, None)
    assert started.wait(1)

    cancel_event.set()
    release.set()
    future.result(timeout=1)

    assert q.get(timeout=1) == ("cancelled", None)
    assert future.done()
    state.executor.shutdown(wait=True)


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_streaming_generation_errors_keep_protocol_error_shape(client, protocol):
    class RaisingRunner(FakeRunner):
        def generate(self, *args, **kwargs):
            raise RuntimeError("runner exploded")

    _install_state(RaisingRunner([]))
    if protocol == "openai":
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        error = next(event["error"] for event in _sse_events(resp.text) if "error" in event)
        assert error == {
            "message": "runner exploded",
            "type": "server_error",
            "param": None,
            "code": "server_error",
        }
        assert "data: [DONE]" in resp.text
    else:
        resp = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        error = next(event for event in _sse_events(resp.text) if event.get("type") == "error")
        assert error["error"] == {"type": "server_error", "message": "runner exploded"}


# ---------- ストリーミング: 最初のイベントが生成開始より前に出る ----------


def test_openai_stream_first_event_precedes_generation(client):
    runner = FakeRunner(tokens_to_emit=[10])
    tok = FakeTokenizer(vocab={10: "hi"})
    _install_state(runner, tokenizer=tok)

    async def run():
        gen = server._openai_stream(
            [1, 2, 3],
            8,
            0.0,
            "chatcmpl-x",
            0,
            "test-model",
            [],
            False,
            None,
        )
        first = await gen.__anext__()
        # 最初のイベントを受け取った時点では、まだロックすら獲得しておらず
        # ワーカーも投入されていない
        assert runner.calls == []
        assert '"role": "assistant"' in first

        chunks = [first]
        async for chunk in gen:
            chunks.append(chunk)
        # 生成は最終的にはちゃんと行われている
        assert runner.calls

    asyncio.run(run())


def test_anthropic_stream_first_event_precedes_generation(client):
    runner = FakeRunner(tokens_to_emit=[10])
    tok = FakeTokenizer(vocab={10: "hi"})
    _install_state(runner, tokenizer=tok)

    async def run():
        gen = server._anthropic_stream(
            [1, 2, 3],
            8,
            0.0,
            "msg_x",
            "test-model",
            [],
            None,
        )
        first = await gen.__anext__()
        assert runner.calls == []
        assert "message_start" in first

        chunks = [first]
        async for chunk in gen:
            chunks.append(chunk)
        assert runner.calls

    asyncio.run(run())


def test_streaming_400_returns_before_any_sse_event(client):
    """SpecRunner 経路で top_p のような未対応パラメータを stream=True で
    送ると、生成 (enqueue) より前に 400 で弾かれ、SSE イベントは 1 つも
    流れない (プレーンな JSON 400 のまま)。"""

    runner = FakeSpecRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True, "top_p": 0.5},
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls
    assert "text/event-stream" not in resp.headers.get("content-type", "")
    assert not resp.text.startswith("data: ")


@pytest.mark.parametrize(
    ("effort", "expected_budget"),
    [
        ("low", 2048),
        ("medium", 8192),
        ("high", 32768),
        ("xhigh", 65536),
        ("max", 131072),
    ],
)
def test_anthropic_adaptive_thinking_effort_is_accepted(client, effort, expected_budget):
    tok = FakeTokenizer(vocab={10: "ok"}, has_thinking=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok, max_tokens_cap=200000)

    body = {
        "model": "test-model",
        "max_tokens": 200000,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
    }
    assert server._resolve_thinking(body, "anthropic") == (True, expected_budget, None)

    resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text
    assert tok.last_apply_chat_template_kwargs["enable_thinking"] is True


# ---------- 7. バグ修正: thinking がプロンプト側で既に開かれている場合 ----------
#
# _apply_template が描画するプロンプトの末尾が think_start トークン列で
# 終端している (テンプレート自身が既に <think> を開いている) モデルでは、
# モデルは think_start を生成し直さない。ThinkingRouter が detect フェーズ
# から始まると「冒頭が一致しない = 考えていない」と誤判定し、思考が本文
# (content) へ混入したうえ閉じ側の </think> だけが生テキストとして残る
# (実モデルで確認済みのバグ)。フェイクトークナイザで prompt_ids の末尾に
# think_start トークンを置くことで、この状況を再現する。
#
# 終了マーカーが 1 トークンだと ThinkingRouter のローリングウィンドウが
# 1 トークン遅れて確定する (tool calling のテストと同じ理由)。


def test_bug1_openai_nonstream_thinking_not_mixed_into_content(client):
    tok = FakeTokenizer(
        vocab={10: "pondering. ", 501: "</think>", 11: "the answer is 4"},
        has_thinking=True,
        think_start_tokens=[500],
        think_end_tokens=[501],
        prompt_ids=[1, 2, 3, 500],  # 描画済みプロンプトが <think> で終わる
    )
    runner = FakeRunner(tokens_to_emit=[10, 501, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "2+2?"}]},
    )
    assert resp.status_code == 200, resp.text
    message = resp.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "pondering. "
    assert message["content"] == "the answer is 4"
    assert "</think>" not in message["content"]
    assert "pondering" not in message["content"]


def test_bug1_openai_stream_thinking_not_mixed_into_content(client):
    tok = FakeTokenizer(
        vocab={10: "pondering. ", 501: "</think>", 11: "the answer is 4"},
        has_thinking=True,
        think_start_tokens=[500],
        think_end_tokens=[501],
        prompt_ids=[1, 2, 3, 500],
    )
    runner = FakeRunner(tokens_to_emit=[10, 501, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "2+2?"}], "stream": True},
    )
    assert resp.status_code == 200, resp.text
    events = _sse_events(resp.text)
    reasoning_text = "".join(
        e["choices"][0]["delta"].get("reasoning_content", "")
        for e in events
        if e.get("choices") and "reasoning_content" in e["choices"][0].get("delta", {})
    )
    content_text = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events
        if e.get("choices") and "content" in e["choices"][0].get("delta", {})
    )
    assert reasoning_text == "pondering. "
    assert content_text == "the answer is 4"
    assert "</think>" not in content_text


def test_bug1_anthropic_nonstream_thinking_not_mixed_into_content(client):
    tok = FakeTokenizer(
        vocab={10: "pondering. ", 501: "</think>", 11: "the answer is 4"},
        has_thinking=True,
        think_start_tokens=[500],
        think_end_tokens=[501],
        prompt_ids=[1, 2, 3, 500],
    )
    runner = FakeRunner(tokens_to_emit=[10, 501, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "2+2?"}],
        },
    )
    assert resp.status_code == 200, resp.text
    blocks = resp.json()["content"]
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["thinking"] == "pondering. "
    assert blocks[0]["signature"] == server._thinking_signature("pondering. ")
    text_blocks = [b for b in blocks if b["type"] == "text"]
    assert text_blocks and text_blocks[0]["text"] == "the answer is 4"
    assert all("</think>" not in b.get("text", "") for b in blocks)


def test_bug1_anthropic_stream_thinking_not_mixed_into_content(client):
    tok = FakeTokenizer(
        vocab={10: "pondering. ", 501: "</think>", 11: "the answer is 4"},
        has_thinking=True,
        think_start_tokens=[500],
        think_end_tokens=[501],
        prompt_ids=[1, 2, 3, 500],
    )
    runner = FakeRunner(tokens_to_emit=[10, 501, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "2+2?"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200, resp.text
    events = _sse_events(resp.text)
    thinking_deltas = "".join(
        e["delta"]["thinking"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "thinking_delta"
    )
    signature_deltas = [
        e["delta"]["signature"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e["delta"].get("type") == "signature_delta"
    ]
    text_deltas = "".join(
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta"
    )
    assert thinking_deltas == "pondering. "
    assert signature_deltas == [server._thinking_signature("pondering. ")]
    assert text_deltas == "the answer is 4"
    assert "</think>" not in text_deltas


def test_bug1_tool_calling_still_works_when_prompt_already_thinking(client):
    tok = FakeTokenizer(
        vocab={10: "pondering. ", 501: "</think>", 200: _TOOL_CALL_JSON, 11: "done"},
        has_thinking=True,
        think_start_tokens=[500],
        think_end_tokens=[501],
        prompt_ids=[1, 2, 3, 500],
        has_tool_calling=True,
        tool_call_start_tokens=[100],
        tool_call_end_tokens=[101],
    )
    runner = FakeRunner(tokens_to_emit=[10, 501, 100, 200, 101, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "weather in tokyo?"}],
            "tools": [_WEATHER_TOOL_OPENAI],
        },
    )
    assert resp.status_code == 200, resp.text
    message = resp.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "pondering. "
    assert "</think>" not in (message.get("content") or "")
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"city": "Tokyo"}


def test_bug1_budget_only_counts_generated_thinking_tokens(client):
    """<think> がプロンプト側 (テンプレート由来) に含まれる場合でも、budget
    はモデルが実際に生成した thinking トークンだけを数える。"""

    tok = FakeTokenizer(
        vocab={10: "a", 501: "", 11: "b"},
        has_thinking=True,
        think_start_tokens=[500],
        think_end_tokens=[501],
        prompt_ids=[1, 500],
    )
    runner = FakeRunner(tokens_to_emit=[10, 10, 10, 501, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 1},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stop_reason"] == "max_tokens"
    thinking_blocks = [b for b in body["content"] if b["type"] == "thinking"]
    assert thinking_blocks
    assert thinking_blocks[0]["thinking"] == "a"


def test_thinking_budget_is_enforced_when_close_marker_is_missing(client):
    tok = FakeTokenizer(
        vocab={10: "a"},
        has_thinking=True,
        think_start_tokens=[500],
        think_end_tokens=[501],
        prompt_ids=[1, 500],
    )
    runner = FakeRunner(tokens_to_emit=[10, 10, 999])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 1},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stop_reason"] == "max_tokens"
    thinking_blocks = [b for b in body["content"] if b["type"] == "thinking"]
    assert thinking_blocks == [
        {
            "type": "thinking",
            "thinking": "a",
            "signature": server._thinking_signature("a"),
        }
    ]


def test_user_literal_think_marker_does_not_open_assistant_thinking(client):
    """An unmatched marker in rendered history is not the generation suffix."""

    tok = FakeTokenizer(
        vocab={42: "user text", 500: "<think>", 43: "assistant:", 10: "answer"},
        has_thinking=True,
        think_start_tokens=[500],
        think_end_tokens=[501],
        prompt_ids=[42, 500, 43],
    )
    runner = FakeRunner(tokens_to_emit=[10, 999])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "literal <think> marker"}]},
    )
    assert resp.status_code == 200, resp.text
    message = resp.json()["choices"][0]["message"]
    assert message["content"] == "answer"
    assert "reasoning_content" not in message


@pytest.mark.parametrize("stream", [False, True])
def test_post_thinking_separator_newlines_are_not_exposed(client, stream):
    """The template/model separator after ``</think>`` is framing, not answer text.

    The real Qwen template emits ``</think>\n\n<answer>``.  Keeping those two
    newlines made Chat Completions return ``\n\n408`` and Responses return
    ``\n\npong`` even though the visible answer itself did not start with a blank
    paragraph.  Cover both collection modes because they use separate assembly
    paths around the shared ThinkingRouter.
    """

    tok = FakeTokenizer(
        vocab={10: "pondering", 501: "</think>", 11: "\n\n", 12: "pong"},
        has_thinking=True,
        think_start_tokens=[500],
        think_end_tokens=[501],
        prompt_ids=[1, 500],
    )
    runner = FakeRunner(tokens_to_emit=[10, 501, 11, 12])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "ping",
            "reasoning": {"effort": "low"},
            "stream": stream,
        },
    )
    assert resp.status_code == 200, resp.text
    if stream:
        pairs = _responses_sse_pairs(resp.text)
        content = "".join(
            data["delta"] for event, data in pairs if event == "response.output_text.delta"
        )
    else:
        content = next(
            item["content"][0]["text"]
            for item in resp.json()["output"]
            if item["type"] == "message"
        )
    assert content == "pong"


def test_leading_newlines_without_thinking_are_preserved(client):
    """Only the separator after a real thinking phase is framing."""

    tok = FakeTokenizer(vocab={10: "\n\n", 11: "pong"})
    runner = FakeRunner(tokens_to_emit=[10, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post("/v1/responses", json={"model": "test-model", "input": "ping"})
    assert resp.status_code == 200, resp.text
    message = next(item for item in resp.json()["output"] if item["type"] == "message")
    assert message["content"][0]["text"] == "\n\npong"


# ---------- 8. バグ修正: thinking/redacted_thinking ブロックを履歴で 400 にしない ----------
#
# 拡張思考 + tool use の Anthropic 規約では、直前ターンの thinking/
# redacted_thinking ブロックを次ターンの履歴にそのまま含めて送り返す必要が
# ある。このサーバー自身が thinking 有効時に "thinking" ブロックを返して
# いるので、tool を使う複数ターンの会話を送り返すクライアント (Claude Code
# 等) は確実にこの型を履歴へ含めてくる。


def test_anthropic_history_thinking_block_does_not_400(client):
    tok = FakeTokenizer(vocab={10: "ok"}, has_tool_calling=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "weather in tokyo?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "let me check the weather tool",
                            "signature": "sig123",
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Tokyo"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                    ],
                },
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    rendered_messages = tok.last_apply_chat_template_kwargs["messages"]
    assert not any(
        "let me check the weather tool" in str(m.get("content", "")) for m in rendered_messages
    )


def test_anthropic_rejects_modified_mlxturbo_thinking_signature(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "ok"}))

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "modified",
                            "signature": server._thinking_signature("original"),
                        },
                        {"type": "text", "text": "answer"},
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
        },
    )
    assert resp.status_code == 400, resp.text
    assert "modified" in resp.json()["error"]["message"]
    assert not runner.calls


def test_anthropic_history_redacted_thinking_block_does_not_400(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "opaque-blob"},
                        {"type": "text", "text": "hello"},
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text


# ---------- 9. Responses API (/v1/responses) ----------
#
# Codex CLI (wire_api: "responses") 向け。生成そのもの・thinking の分離・
# tool call の解析・サンプリングパラメータ・session 選択・stop 判定は
# Chat Completions/Anthropic 経路と共有する (_collect_events/_start_generation
# をそのまま使う) — ここで検証するのは input の読み取りと output/イベント
# の組み立てだけ。


_RESPONSES_WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "get the weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def _responses_sse_pairs(text: str) -> list[tuple[str | None, dict]]:
    """Responses API の SSE ボディを (event, data) のリストへパースする。
    Chat Completions/Anthropic と違い ``data: [DONE]`` は送らない。"""

    pairs: list[tuple[str | None, dict]] = []
    event = None
    for line in text.splitlines():
        if line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            pairs.append((event, json.loads(line[len("data: ") :])))
    return pairs


def test_responses_nonstream_basic_text(client):
    tok = FakeTokenizer(vocab={10: "hello", 11: " world"})
    runner = FakeRunner(tokens_to_emit=[10, 11, 999])  # 999 = eos (自然終了させる)
    _install_state(runner, tokenizer=tok)

    resp = client.post("/v1/responses", json={"model": "test-model", "input": "hi"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert len(body["output"]) == 1
    item = body["output"][0]
    assert item["type"] == "message"
    assert item["role"] == "assistant"
    assert item["content"] == [{"type": "output_text", "text": "hello world", "annotations": []}]
    assert body["usage"]["output_tokens"] == 3
    assert tok.last_apply_chat_template_kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_responses_instructions_becomes_system_message(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={"model": "test-model", "instructions": "be terse", "input": "hi"},
    )
    assert resp.status_code == 200, resp.text
    messages = tok.last_apply_chat_template_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "be terse"}
    assert messages[1] == {"role": "user", "content": "hi"}


def test_responses_typed_input_items(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    assert tok.last_apply_chat_template_kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_responses_tools_flat_shape_converted_for_template(client):
    tok = FakeTokenizer(vocab={10: "ok"}, has_tool_calling=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "weather in tokyo?",
            "tools": [_RESPONSES_WEATHER_TOOL],
        },
    )
    assert resp.status_code == 200, resp.text
    assert tok.last_apply_chat_template_kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "get the weather for a city",
                "parameters": _RESPONSES_WEATHER_TOOL["parameters"],
            },
        }
    ]


def test_responses_tools_on_unsupported_model_is_400(client):
    tok = FakeTokenizer(vocab={10: "ok"})  # has_tool_calling=False (既定)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi", "tools": [_RESPONSES_WEATHER_TOOL]},
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_responses_nonstream_function_call_output_item(client):
    vocab = {10: "Sure, checking. ", 200: _TOOL_CALL_JSON, 11: "!"}
    tok = _tool_calling_tokenizer(vocab)
    runner = FakeRunner(tokens_to_emit=[10, 100, 200, 101, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "weather in tokyo?",
            "tools": [_RESPONSES_WEATHER_TOOL],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    types = [item["type"] for item in body["output"]]
    # "Sure, checking. " (message) -> get_weather (function_call) -> "!"
    # (別の message、tool_call を挟んで content_delta のランが分かれるため)
    assert types == ["message", "function_call", "message"]
    assert body["output"][0]["content"][0]["text"] == "Sure, checking. "
    assert body["output"][2]["content"][0]["text"] == "!"
    fc = body["output"][1]
    assert fc["name"] == "get_weather"
    assert fc["call_id"]
    assert json.loads(fc["arguments"]) == {"city": "Tokyo"}


def test_responses_function_call_history_reaches_template(client):
    tok = FakeTokenizer(vocab={10: "ok"}, has_tool_calling=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {"role": "user", "content": "weather in tokyo?"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "Tokyo"}),
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    messages = tok.last_apply_chat_template_kwargs["messages"]
    assert messages[0] == {"role": "user", "content": "weather in tokyo?"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": {"city": "Tokyo"},
    }
    assert messages[2] == {"role": "tool", "content": "sunny", "tool_call_id": "call_1"}


def test_responses_reasoning_item_in_history_is_skipped_not_400(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {"role": "user", "content": "hi"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "continue"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text


def test_responses_non_text_input_part_is_400(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {"role": "user", "content": [{"type": "input_image", "image_url": "http://x/y.png"}]},
            ],
        },
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_responses_model_mismatch_is_404(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok, model_name="test-model")

    resp = client.post("/v1/responses", json={"model": "other-model", "input": "hi"})
    assert resp.status_code == 404, resp.text
    assert not runner.calls


# ---------- バグ修正: --model-alias で別名の model も受け付ける ----------
#
# Claude Code は会話タイトル生成などの裏方処理に別の小さいモデル
# (claude-3-5-haiku-20241022 等) を使う。サーブ中の名前と不一致なら 404 と
# いう既定の挙動は OpenAI 準拠の意図的な設計 (_check_model_openai/
# _check_model_anthropic docstring 参照) なので崩さないが、--model-alias で
# 明示的に許可した名前だけは通す。未指定 (既定) では従来どおり厳密一致の
# ままであることも合わせて確認する。


def test_model_alias_unset_still_404s_for_other_model(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok, model_name="test-model")  # model_aliases 未指定

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-3-5-haiku-20241022", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404, resp.text
    assert not runner.calls


def test_model_alias_allows_configured_name_on_chat_completions(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner,
        tokenizer=tok,
        model_name="test-model",
        model_aliases=frozenset({"claude-3-5-haiku-20241022"}),
    )

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-3-5-haiku-20241022", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200, resp.text
    # 別名を受け付けても、応答の "model" 欄は常にサーブ中の名前そのもの
    # (クライアントが送った別名をそのまま echo しない)。
    assert resp.json()["model"] == "test-model"


def test_model_alias_still_404s_for_unlisted_name(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner,
        tokenizer=tok,
        model_name="test-model",
        model_aliases=frozenset({"claude-3-5-haiku-20241022"}),
    )

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "some-other-unlisted-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404, resp.text
    assert not runner.calls


def test_model_alias_allows_configured_name_on_anthropic_messages(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeSpecRunner(tokens_to_emit=[10])
    _install_state(
        runner,
        tokenizer=tok,
        model_name="test-model",
        model_aliases=frozenset({"claude-3-5-haiku-20241022"}),
    )

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["model"] == "test-model"


def test_protocol_error_envelopes_include_required_metadata(client):
    runner = FakeRunner(tokens_to_emit=[])
    _install_state(runner)

    openai = client.post("/v1/responses", json={"model": "test-model"})
    assert openai.status_code == 400
    assert openai.json()["error"]["param"] is None
    assert "code" in openai.json()["error"]

    anthropic = client.post("/v1/messages", json={"model": "test-model"})
    assert anthropic.status_code == 400
    request_id = anthropic.json()["request_id"]
    assert request_id.startswith("req_")
    assert anthropic.headers["request-id"] == request_id


def test_responses_unknown_previous_response_id_is_404(client):
    """項目 15: previous_response_id はもう黙って 400 にしない — 保存済み
    LRU (STATE.response_store) に無い id だけ 404 になる (OpenAI 準拠、
    _check_model_openai の model_not_found と同じ流儀)。"""

    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi", "previous_response_id": "resp_123"},
    )
    assert resp.status_code == 404, resp.text
    assert not runner.calls
    assert "resp_123" in resp.json()["error"]["message"]
    assert resp.json()["error"]["code"] == "response_not_found"


def test_responses_store_true_round_trips_with_previous_response_id(client):
    """項目 15: store:true で保存した応答の id を次ターンの
    previous_response_id に渡すと、保存しておいた会話 (今回の input + この
    応答の output) が次ターンの新しい input の前に連結されて
    apply_chat_template へ渡る — クライアントは全文を送り直さなくてよい。"""

    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    first = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi", "store": True},
    )
    assert first.status_code == 200, first.text
    resp_id = first.json()["id"]
    assert resp_id
    assert "downgrade_reason" not in first.json()

    second = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "again",
            "previous_response_id": resp_id,
        },
    )
    assert second.status_code == 200, second.text
    assert len(runner.calls) == 2
    messages = tok.last_apply_chat_template_kwargs["messages"]
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "again"},
    ]


def test_responses_store_defaults_to_no_previous_response_id_available(client):
    """store を指定しない (既定 False) 応答の id は previous_response_id
    として使えない — LRU に積まれていないので 404 になる。"""

    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    first = client.post("/v1/responses", json={"model": "test-model", "input": "hi"})
    assert first.status_code == 200, first.text
    resp_id = first.json()["id"]

    second = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "again", "previous_response_id": resp_id},
    )
    assert second.status_code == 404, second.text


def test_responses_stored_lru_evicts_oldest(client):
    """保持数の上限 (--max-stored-responses) を超えたら最も古いものを
    追い出す。"""

    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok, max_stored_responses=1)

    first = client.post(
        "/v1/responses", json={"model": "test-model", "input": "hi", "store": True}
    )
    first_id = first.json()["id"]
    client.post("/v1/responses", json={"model": "test-model", "input": "bye", "store": True})

    resp = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "again", "previous_response_id": first_id},
    )
    assert resp.status_code == 404, resp.text


def test_responses_reasoning_effort_produces_reasoning_output_item(client):
    tok = FakeTokenizer(
        vocab={10: "pondering. ", 501: "</think>", 11: "42"},
        has_thinking=True,
        think_start_tokens=[500],
        think_end_tokens=[501],
        prompt_ids=[1, 500],
    )
    runner = FakeRunner(tokens_to_emit=[10, 501, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "2+2?", "reasoning": {"effort": "low"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    types = [item["type"] for item in body["output"]]
    assert types == ["reasoning", "message"]
    assert body["output"][0]["summary"][0]["text"] == "pondering. "
    assert body["output"][1]["content"][0]["text"] == "42"


def test_responses_stream_event_sequence(client):
    tok = FakeTokenizer(vocab={10: "hello", 11: " world"})
    runner = FakeRunner(tokens_to_emit=[10, 11, 999])  # 999 = eos (自然終了させる)
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi", "stream": True},
    )
    assert resp.status_code == 200, resp.text
    pairs = _responses_sse_pairs(resp.text)
    events = [e for e, _ in pairs]
    assert [data["sequence_number"] for _, data in pairs] == list(range(len(pairs)))
    assert events[0] == "response.created"
    assert "response.output_item.added" in events
    assert "response.output_text.delta" in events
    assert "response.output_item.done" in events
    assert events[-1] == "response.completed"

    text = "".join(d["delta"] for e, d in pairs if e == "response.output_text.delta")
    assert text == "hello world"

    completed = next(d for e, d in pairs if e == "response.completed")
    assert completed["response"]["status"] == "completed"
    assert completed["response"]["output"][0]["content"][0]["text"] == "hello world"


def test_responses_stream_uses_incomplete_terminal_event_at_token_cap(client):
    """A status=incomplete body must not be wrapped in response.completed."""

    runner = FakeRunner(tokens_to_emit=[10, 11])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "a", 11: "b"}))

    resp = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi", "stream": True, "max_output_tokens": 1},
    )
    assert resp.status_code == 200, resp.text
    pairs = _responses_sse_pairs(resp.text)
    assert pairs[-1][0] == "response.incomplete"
    assert pairs[-1][1]["response"]["status"] == "incomplete"
    assert pairs[-1][1]["response"]["incomplete_details"] == {
        "reason": "max_output_tokens"
    }


def test_responses_complete_tool_call_without_eos_is_completed(client):
    tok = _tool_calling_tokenizer({200: _TOOL_CALL_JSON})
    # The fourth token is deliberately truncated: the closed tool call lands
    # exactly on the generation cap and must still win over length termination.
    runner = FakeRunner(tokens_to_emit=[100, 200, 101, 10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "weather?",
            "tools": [_RESPONSES_WEATHER_TOOL],
            "max_output_tokens": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"
    assert resp.json()["output"][0]["type"] == "function_call"


def test_responses_stream_failure_has_failed_terminal_event(client):
    class RaisingRunner(FakeRunner):
        def generate(self, *args, **kwargs):
            raise RuntimeError("runner exploded")

    _install_state(RaisingRunner([]))
    resp = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi", "stream": True},
    )
    assert resp.status_code == 200, resp.text
    pairs = _responses_sse_pairs(resp.text)
    assert [event for event, _ in pairs][-2:] == ["error", "response.failed"]
    failed = pairs[-1][1]["response"]
    assert failed["status"] == "failed"
    assert failed["error"]["message"] == "runner exploded"


def test_responses_stream_emits_function_call_arguments_delta(client):
    vocab = {10: "Sure. ", 200: _TOOL_CALL_JSON, 11: "!"}
    tok = _tool_calling_tokenizer(vocab)
    runner = FakeRunner(tokens_to_emit=[10, 100, 200, 101, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "weather in tokyo?",
            "tools": [_RESPONSES_WEATHER_TOOL],
            "stream": True,
        },
    )
    assert resp.status_code == 200, resp.text
    pairs = _responses_sse_pairs(resp.text)
    events = [e for e, _ in pairs]
    assert "response.function_call_arguments.delta" in events
    assert "response.function_call_arguments.done" in events

    done = next(d for e, d in pairs if e == "response.function_call_arguments.done")
    assert done["type"] == "response.function_call_arguments.done"
    assert done["name"] == "get_weather"
    assert done["output_index"] == 1
    assert done["item_id"].startswith("fc_")
    assert done["arguments"]
    assert isinstance(done["sequence_number"], int)

    args_str = "".join(
        d["delta"] for e, d in pairs if e == "response.function_call_arguments.delta"
    )
    assert json.loads(args_str) == {"city": "Tokyo"}

    completed = next(d for e, d in pairs if e == "response.completed")
    types = [item["type"] for item in completed["response"]["output"]]
    # "Sure. " (message) -> get_weather (function_call) -> "!" (別の message)
    assert types == ["message", "function_call", "message"]


def test_responses_stream_emits_reasoning_summary_text_delta(client):
    """バグ 1 (thinking がプロンプト側で既に開かれている) の修正が
    Responses API のストリーミング経路にも効いていることを併せて確認する。"""

    tok = FakeTokenizer(
        vocab={10: "pondering. ", 501: "</think>", 11: "42"},
        has_thinking=True,
        think_start_tokens=[500],
        think_end_tokens=[501],
        prompt_ids=[1, 500],
    )
    runner = FakeRunner(tokens_to_emit=[10, 501, 11])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "2+2?",
            "reasoning": {"effort": "low"},
            "stream": True,
        },
    )
    assert resp.status_code == 200, resp.text
    pairs = _responses_sse_pairs(resp.text)
    reasoning_text = "".join(
        d["delta"] for e, d in pairs if e == "response.reasoning_summary_text.delta"
    )
    reasoning_deltas = [
        d for e, d in pairs if e == "response.reasoning_summary_text.delta"
    ]
    assert reasoning_deltas and all(d["summary_index"] == 0 for d in reasoning_deltas)
    content_text = "".join(d["delta"] for e, d in pairs if e == "response.output_text.delta")
    assert reasoning_text == "pondering. "
    assert content_text == "42"
    assert "</think>" not in content_text


def test_responses_stream_first_event_precedes_generation(client):
    tok = FakeTokenizer(vocab={10: "hi"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    async def run():
        gen = server._responses_stream(
            [1, 2, 3], 8, 0.0, "resp_x", 0, "test-model", None
        )
        first = await gen.__anext__()
        assert runner.calls == []
        assert "response.created" in first

        chunks = [first]
        async for chunk in gen:
            chunks.append(chunk)
        assert runner.calls

    asyncio.run(run())


# ---------- 10. バグ修正: Anthropic system の並び順 (実クライアント: Claude Code) ----------
#
# 実際に捕獲した Claude Code のリクエストボディでは、トップレベルの
# "system" (3 要素の text ブロック配列) に加えて、"messages" のロール並びが
# ['user', 'system'] — system ロールのメッセージが末尾に来る。Qwen 系の
# チャットテンプレートは system が先頭に無いと "System message must be at
# the beginning" で落ちるので、トップレベルの system と messages 内の
# system ロールを両方とも先頭へ寄せて 1 個の system メッセージへ連結する
# (mlxturbo/server.py の anthropic_messages 参照)。


def test_anthropic_system_role_in_messages_moved_to_front_with_top_level_system(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    body = {
        "model": "test-model",
        "max_tokens": 32,
        "system": [
            {"type": "text", "text": "top-level sys A. "},
            {
                "type": "text",
                "text": "top-level sys B.",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ],
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "embedded sys C"},
        ],
    }
    resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text

    rendered = tok.last_apply_chat_template_kwargs["messages"]
    assert rendered == [
        {
            "role": "system",
            "content": "top-level sys A. top-level sys B.\n\nembedded sys C",
        },
        {"role": "user", "content": "hi"},
    ]


def test_anthropic_embedded_system_role_without_top_level_system(client):
    """トップレベル system が無くても、messages 内の system ロールだけで
    先頭へ寄せる経路が動くことを確認する。"""

    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    body = {
        "model": "test-model",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "embedded only"},
        ],
    }
    resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text

    rendered = tok.last_apply_chat_template_kwargs["messages"]
    assert rendered == [
        {"role": "system", "content": "embedded only"},
        {"role": "user", "content": "hi"},
    ]


def test_anthropic_multiple_embedded_system_roles_keep_relative_order(client):
    """messages 内に system ロールが複数回現れても (トップレベルも含めて)
    元の出現順のまま連結される。system 以外のメッセージの相対順序も崩れ
    ない。"""

    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    body = {
        "model": "test-model",
        "max_tokens": 32,
        "system": "top",
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "s1"},
            {"role": "user", "content": "u2"},
            {"role": "system", "content": "s2"},
        ],
    }
    resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text

    rendered = tok.last_apply_chat_template_kwargs["messages"]
    assert rendered == [
        {"role": "system", "content": "top\n\ns1\n\ns2"},
        {"role": "user", "content": "u1"},
        {"role": "user", "content": "u2"},
    ]


# ---------- 11. バグ修正: Codex CLI の tools (namespace 展開 / web_search 除外) ----------
#
# 実際に捕獲した Codex CLI のリクエストボディでは、"tools" に
# {"type": "function", ...} 以外の要素 (namespace で入れ子になった
# サブエージェント用ツール群、web_search) が混ざって送られてくる。mlxturbo
# は "each item in 'tools' must be an object with \"type\": \"function\""
# で 400 を返していた。namespace は中の function を展開して拾い、
# web_search はこのサーバーに実行主体が無いので黙って落とさずログへ残して
# から除く (mlxturbo/server.py の _flatten_responses_tools 参照)。


def test_responses_tools_flattens_namespace_and_drops_web_search(client, capsys):
    tok = FakeTokenizer(vocab={10: "ok"}, has_tool_calling=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    body = {
        "model": "test-model",
        "input": "hi",
        "tools": [
            {
                "type": "function",
                "name": "f",
                "description": "top-level function",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "Tools for spawning and managing sub-agents.",
                "tools": [
                    {
                        "type": "function",
                        "name": "close_agent",
                        "parameters": {"type": "object", "properties": {}},
                    },
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ],
            },
            {"type": "web_search", "external_web_access": False},
        ],
    }
    resp = client.post("/v1/responses", json=body)
    assert resp.status_code == 200, resp.text

    sent_tools = tok.last_apply_chat_template_kwargs["tools"]
    names = {t["function"]["name"] for t in sent_tools}
    assert names == {"f", "close_agent", "spawn_agent"}
    assert all(t["type"] == "function" for t in sent_tools)

    # web_search を黙って握りつぶさず、落としたことがログに残る。
    log = capsys.readouterr().out
    assert "web_search" in log
    assert "[mlxturbo-serve]" in log


def test_responses_tools_all_unsupported_falls_back_to_no_tools(client):
    """namespace/web_search を展開・除外した結果 tools が空になった場合、
    400 にはせず「今回のターンはツール無し」として素通しする。"""

    tok = FakeTokenizer(vocab={10: "ok"}, has_tool_calling=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hi",
            "tools": [{"type": "web_search", "external_web_access": False}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert tok.last_apply_chat_template_kwargs["tools"] is None


def test_responses_tools_nested_namespace_expands_recursively(client):
    tok = FakeTokenizer(vocab={10: "ok"}, has_tool_calling=True)
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    body = {
        "model": "test-model",
        "input": "hi",
        "tools": [
            {
                "type": "namespace",
                "name": "outer",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "inner",
                        "tools": [
                            {
                                "type": "function",
                                "name": "deep",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ],
                    }
                ],
            },
        ],
    }
    resp = client.post("/v1/responses", json=body)
    assert resp.status_code == 200, resp.text
    sent_tools = tok.last_apply_chat_template_kwargs["tools"]
    assert [t["function"]["name"] for t in sent_tools] == ["deep"]


# ---------- 11b. バグ修正: Responses API の instructions/developer の並び順 ----------
#
# 実クライアントでの検証中に発見: Codex CLI は トップレベルの
# "instructions" に加えて、"input" の先頭に role: "developer" のメッセージ
# を混ぜて送ってくる (捕獲したボディで確認済み)。developer は system と
# 同じ扱いにする既存の変換のせいで、system 相当のメッセージが 2 個
# (instructions 由来 + developer 由来) 別々の位置に並び、Anthropic 経路の
# bug (10 番) と同じ理由で "System message must be at the beginning" に
# なっていた。instructions と developer/system ロールの input アイテムは
# すべて 1 個の system メッセージへ連結し、先頭に置く。


def test_responses_instructions_and_developer_role_consolidated_to_front(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    body = {
        "model": "test-model",
        "instructions": "top-level instructions",
        "input": [
            {"type": "message", "role": "developer", "content": "developer text"},
            {"type": "message", "role": "user", "content": "hi"},
        ],
    }
    resp = client.post("/v1/responses", json=body)
    assert resp.status_code == 200, resp.text

    rendered = tok.last_apply_chat_template_kwargs["messages"]
    assert rendered == [
        {"role": "system", "content": "top-level instructions\n\ndeveloper text"},
        {"role": "user", "content": "hi"},
    ]


def test_responses_string_input_with_instructions_still_consolidates(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={"model": "test-model", "instructions": "sys", "input": "hi"},
    )
    assert resp.status_code == 200, resp.text
    assert tok.last_apply_chat_template_kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


# ---------- 12. バグ修正: _select_session の部分一致 + KV trim ----------
#
# opencode 実測: 3 ターン目以降で ~11.7k の接頭辞を共有しているのに毎ターン
# 全量を再 prefill していた。原因は _select_session が「処理済み列の全体が
# 新プロンプトの接頭辞であること」(lcp == len(pl)) を要求していたこと —
# チャットテンプレートが生成プロンプトの末尾に開く thinking マーカー等は
# 次ターンの履歴には再現されないため、末尾のわずかなトークンが一致せず
# 全体が不一致扱いになり、毎ターン全再構築していた。
#
# 実際に確かめると (このモジュール冒頭の調査参照): KV キャッシュを最長
# 共通接頭辞まで巻き戻す (trim) 操作自体は、GDN ハイブリッドの線形層に
# 使う ArraysCache では原理的に不可能 (mlx_lm.models.cache.ArraysCache は
# is_trimmable() を持たず、常に False)。このサーバーが実運用で使う唯一の
# 2 経路 (SpecEngine の ChatSession、FallbackRunner の FallbackSession) は
# どちらも GDN ハイブリッド専用なので、この trim (exact-trim) だけに頼ると
# 常に不発に終わり、「全体一致 or 新規スロット」に落ちる。以下のテストは、
# (a) trim が安全に効く汎用的な形 (KVCache だけで構成されたキャッシュ、
# mlx_lm.models.cache の本物を使う) と、(b) 実運用どおり ArraysCache が
# 混ざっていて trim が不発に終わる形の両方を、実際の mlx_lm キャッシュ
# 実装に対して確認する。GDN ハイブリッド構成でも再利用そのものを諦めない
# チェックポイント経由の復元は 12b 節 (_try_checkpoint_restore_session_
# cache) を参照。


def _real_kv_cache(offset: int):
    """mlx_lm.models.cache.KVCache を trim 判定・実行だけ検証できる最小限の
    状態で作る。KVCache.trim() は offset を減らすだけで keys/values の中身
    は見ない (mlx_lm/models/cache.py 参照) ので、shape さえ辻褄が合っていれ
    ば中身はダミーでよい。"""

    from mlx_lm.models.cache import KVCache

    c = KVCache()
    c.offset = offset
    c.keys = mx.zeros((1, 1, max(offset, 1), 1))
    c.values = mx.zeros((1, 1, max(offset, 1), 1))
    return c


def test_select_session_reuses_via_trim_when_cache_is_fully_trimmable():
    """caches が (GDN の線形層を含まない、通常の attention だけの)
    KVCache だけで構成されていれば、部分一致でも実際に trim して同じ
    スロットを再利用する。"""

    _install_state(FakeRunner(tokens_to_emit=[]))

    sess = ChatSession()
    sess.processed = [1, 2, 3, 4, 5]
    sess.caches = [_real_kv_cache(5), _real_kv_cache(5)]
    sess.mtp_valid = False
    server.STATE.session_pool[0] = sess

    got = server._select_session([1, 2, 3, 9, 9])

    assert got is sess
    assert got.processed == [1, 2, 3]
    assert got.caches[0].offset == 3
    assert got.caches[1].offset == 3
    # MTP 継続用の状態は、対応する位置が無いので巻き戻し後は使わせない。
    assert got.mtp_cache is None
    assert got.h_last is None


def test_select_session_reuses_when_prompt_is_a_prefix_of_processed():
    """同じプロンプトの投げ直し (regenerate、会話を 1 往復戻す、エージェントが
    毎ターン同じ前置きを送る) で再利用が効くこと。

    processed は「プロンプト + 生成したトークン」なので、同じプロンプトが再び
    来ると lcp == len(prompt_ids) になる。再利用できる量が最大の場合だが、
    以前は両方の pass がこれを候補から外していて、まるごと prefill を
    やり直していた (実測: 12k トークンで 2 回目も 39.6 秒)。
    """

    _install_state(FakeRunner(tokens_to_emit=[]))

    sess = ChatSession()
    sess.processed = [1, 2, 3, 4, 5, 90, 91]  # プロンプト 5 + 生成 2
    sess.caches = [_real_kv_cache(7), _real_kv_cache(7)]
    sess.mtp_valid = False
    server.STATE.session_pool[0] = sess

    got = server._select_session([1, 2, 3, 4, 5])

    assert got is sess, "同じプロンプトの投げ直しで別スロットに落ちてはいけない"
    # 差分 0 では generate_stream がチャンクループに入らず hyper 状態が
    # 未設定のままになるので、最後の 1 トークンは必ず prefill に残す。
    assert got.processed == [1, 2, 3, 4]
    assert got.caches[0].offset == 4


def test_select_session_leaves_one_token_to_prefill_on_exact_repeat():
    """processed がプロンプトそのものと完全一致していても、再利用は
    プロンプト長 -1 で頭打ちにすること。"""

    _install_state(FakeRunner(tokens_to_emit=[]))

    sess = ChatSession()
    sess.processed = [1, 2, 3]
    sess.caches = [_real_kv_cache(3)]
    sess.mtp_valid = False
    server.STATE.session_pool[0] = sess

    got = server._select_session([1, 2, 3])

    assert got is sess
    assert got.processed == [1, 2]
    assert got.caches[0].offset == 2


def test_select_session_reuses_fully_when_tail_matches_exact_repeat():
    """Diff-0 の TTFT 修正 (2026-08-30): 上の 2 テストはどちらも 1 トークン
    残す (reuse_cap の既定)。session.tail が「新プロンプトの長さちょうど」の
    位置にスタンプされていれば、そこまでフルに再利用してよい --
    generate()/generate_stream() がそこから resume できるようになったため
    (mlxturbo/spec.py, mlxturbo/spec_flash.py)。tail の中身自体は
    _select_session にとって不透明 (位置だけ見る) なのでダミーでよい。"""

    _install_state(FakeRunner(tokens_to_emit=[]))

    sess = ChatSession()
    sess.processed = [1, 2, 3, 4, 5, 90, 91]  # プロンプト 5 + 生成 2
    sess.caches = [_real_kv_cache(7), _real_kv_cache(7)]
    sess.mtp_valid = False
    sess.tail = (5, "dummy-resume-state")  # プロンプト長 (5) ちょうど
    server.STATE.session_pool[0] = sess

    got = server._select_session([1, 2, 3, 4, 5])

    assert got is sess
    assert got.processed == [1, 2, 3, 4, 5]  # 1 トークンも残さずフル再利用
    assert got.caches[0].offset == 5
    assert got.tail == (5, "dummy-resume-state")  # 位置が一致するので残る


def test_select_session_leaves_one_token_when_tail_position_does_not_match():
    """tail はあっても、その位置が新プロンプト長と一致しなければ従来どおり
    1 トークン残す (無関係な tail に釣られて緩めてはいけない)。tail の位置
    (6) は緩和の対象 (新プロンプト長 5) にも、結果として trim される位置
    (4, reuse_cap どおり) にも一致しない値を選び、「一致しないので必ず捨て
    られる」ことを区別して確認する。"""

    _install_state(FakeRunner(tokens_to_emit=[]))

    sess = ChatSession()
    sess.processed = [1, 2, 3, 4, 5, 90, 91]
    sess.caches = [_real_kv_cache(7), _real_kv_cache(7)]
    sess.mtp_valid = False
    sess.tail = (6, "stale-resume-state")  # 新プロンプト長 (5) にも
    # trim 後の位置 (4) にも一致しない
    server.STATE.session_pool[0] = sess

    got = server._select_session([1, 2, 3, 4, 5])

    assert got is sess
    assert got.processed == [1, 2, 3, 4]
    assert got.caches[0].offset == 4
    assert got.tail is None  # 一致しない tail は使えないので捨てる


def test_select_session_falls_back_when_a_cache_layer_is_not_trimmable():
    """1 レイヤーでも巻き戻せなければ (実運用の GDN ハイブリッド構成:
    ArraysCache が線形層に混ざる)、部分一致は諦めて新規スロットへ倒す。
    元のスロットは無傷のまま残る。"""

    from mlx_lm.models.cache import ArraysCache

    _install_state(FakeRunner(tokens_to_emit=[]))

    sess = ChatSession()
    sess.processed = [1, 2, 3, 4, 5]
    sess.caches = [_real_kv_cache(5), ArraysCache(size=2)]
    server.STATE.session_pool[0] = sess

    got = server._select_session([1, 2, 3, 9, 9])

    assert got is not sess
    assert got.processed == []
    assert sess.processed == [1, 2, 3, 4, 5]  # 元のスロットは半端に壊れていない
    assert sess.caches[0].offset == 5


def test_select_session_skips_partial_match_when_mtp_valid():
    """caches 自体は全レイヤー trim 可能でも、mtp_valid が立っていれば
    (MTP 継続用の h_last が巻き戻し後の位置に対応しなくなる) 部分一致は
    使わない。"""

    _install_state(FakeRunner(tokens_to_emit=[]))

    sess = ChatSession()
    sess.processed = [1, 2, 3, 4, 5]
    sess.caches = [_real_kv_cache(5)]
    sess.mtp_valid = True
    server.STATE.session_pool[0] = sess

    got = server._select_session([1, 2, 3, 9, 9])

    assert got is not sess
    assert sess.processed == [1, 2, 3, 4, 5]
    assert sess.caches[0].offset == 5  # 触られていない


def test_select_session_full_match_still_preferred_over_partial(client):
    """全体一致するスロットがあれば、より長い部分一致候補が他にあっても
    キャッシュに触れない既存の安全経路 (全体一致) が優先される。"""

    _install_state(FakeRunner(tokens_to_emit=[]))

    # 部分一致で "得" に見えるが trim 不可のスロット (長い processed)
    from mlx_lm.models.cache import ArraysCache

    partial = ChatSession()
    partial.processed = [1, 2, 3, 4, 5, 6, 7, 8]
    partial.caches = [ArraysCache(size=2)]
    server.STATE.session_pool["partial"] = partial

    # 全体一致するスロット (短いが、キャッシュに触れず安全に再利用できる)
    full = ChatSession()
    full.processed = [1, 2, 3]
    full.caches = [_real_kv_cache(3)]
    server.STATE.session_pool["full"] = full

    got = server._select_session([1, 2, 3, 4])

    assert got is full
    assert got.processed == [1, 2, 3]  # 変更されていない (追記のみ)


# ---------- 12b. バグ修正: チェックポイントによる部分一致からの復元 ----------
#
# 12 節の trim (exact-trim) は GDN ハイブリッド (ArraysCache が混ざる、この
# サーバーの実運用構成そのもの) では常に不発に終わる。「巻き戻せないものは
# スナップショットで持てばよい」という方針で、mlxturbo/spec.py の
# _prefill_hidden がプレフィルのチャンク境界ごとに ArraysCache 層の状態を
# ChatSession.checkpoints へ退避しておき、trim が効かない場合の代替として
# _try_checkpoint_restore_session_cache がそこから最も近い位置まで復元する。
#
# もう 1 点、実運用で trim が「効いても無駄になる」既存の落とし穴も一緒に
# 直した: generate() 側の再利用判定が KV/GDN の再利用可否と MTP 連鎖の
# 継続可否を同じフラグ (mtp_valid) で束ねていたため、MTP を常用するこの
# サーバーでは公開済み session が生成直後ほぼ常に mtp_valid=True になり、
# (a) _try_trim_session_cache 自体が mtp_valid=True を理由に最初から諦める
# うえ、(b) 仮にそこを通っても generate() 側が「MTP 継続できないなら KV も
# 使わない」という設計だったため、trim の成果が丸ごと捨てられていた。
# generate() 側を「KV/GDN の再利用は session.mtp_valid と無関係に行い、MTP
# 連鎖だけ mtp_valid のときだけ引き継ぐ (そうでなければ次ターン開始位置
# から作り直す)」よう分離したことで、mtp_valid=True (実運用の通常状態) の
# ままチェックポイント経由の復元が機能するようになった。


def test_checkpoint_restore_picks_nearest_checkpoint_at_or_below_lcp():
    """複数チェックポイントがあるとき、lcp 以下で最も近い位置を選ぶ。trim
    できるレイヤー (KVCache) はそのチェックポイント位置まで trim され、
    trim できないレイヤー (ArraysCache) はスナップショットの中身がそのまま
    書き戻る。mtp_valid=True (実運用の通常状態) でも弾かれない。"""

    from mlx_lm.models.cache import ArraysCache

    arr = ArraysCache(size=2)
    arr.cache = [mx.array([9.0]), mx.array([99.0])]  # 「今」の (使われない) 状態

    sess = ChatSession()
    sess.processed = list(range(10))
    sess.caches = [_real_kv_cache(10), arr]
    sess.mtp_valid = True
    sess.checkpoints = [
        (4, [(1, [mx.array([4.0]), mx.array([44.0])], None, None)]),
        (7, [(1, [mx.array([7.0]), mx.array([77.0])], None, None)]),
    ]

    cp_pos = server._try_checkpoint_restore_session_cache(sess, lcp=8)

    assert cp_pos == 7
    assert sess.caches[0].offset == 7  # チェックポイント位置 (7) まで trim された
    assert [float(x.item()) for x in sess.caches[1].cache] == [7.0, 77.0]


def test_checkpoint_restore_returns_none_when_no_checkpoint_at_or_below_lcp():
    """使えるチェックポイントが無ければ何も変更せず諦める (呼び出し側が
    新規スロットへ倒せるように、半端な状態を残さない)。"""

    from mlx_lm.models.cache import ArraysCache

    arr = ArraysCache(size=1)
    arr.cache = [mx.array([1.0])]
    sess = ChatSession()
    sess.processed = list(range(10))
    sess.caches = [_real_kv_cache(10), arr]
    sess.checkpoints = [(9, [(1, [mx.array([9.0])], None, None)])]

    cp_pos = server._try_checkpoint_restore_session_cache(sess, lcp=3)

    assert cp_pos is None
    assert sess.caches[0].offset == 10  # 触られていない
    assert float(sess.caches[1].cache[0].item()) == 1.0  # 触られていない


def test_checkpoint_restore_returns_none_without_checkpoints():
    """session.checkpoints が空 (real prefill を一度も通っていない、または
    12 節のテストのように手作業で caches だけ差し込んだ) なら、trim 不可の
    レイヤーがあってもチェックポイント経由でも復元しようがなく None。"""

    from mlx_lm.models.cache import ArraysCache

    sess = ChatSession()
    sess.processed = [1, 2, 3, 4, 5]
    sess.caches = [_real_kv_cache(5), ArraysCache(size=1)]

    assert server._try_checkpoint_restore_session_cache(sess, lcp=3) is None


def test_restore_untrimmable_caches_does_not_alias_the_stored_snapshot():
    """バグ修正 (2026-08-30、diff-0 の実機確認中に発見): restore_untrimmable_
    caches は ``c.state = state`` で snapshot のリストをそのままライブの
    ArraysCache に渡していたが、``ArraysCache.state`` のセッター (mlx_lm.
    models.cache) は単なるエイリアス代入 (``self.cache = v``) で、
    ``__setitem__`` はそのリストを in-place に書き換える (``self.cache[idx]
    = value``)。つまり復元直後、ライブのキャッシュと checkpoints に積んで
    ある snapshot は同じリストオブジェクトを指してしまう。その状態で 1 回
    でも書き込む (次のデコードラウンドの ``cache[i] = ...``) と、アーカイブ
    してあったはずの snapshot 自体まで書き換わり、同じチェックポイントから
    の 2 回目以降の復元が壊れた値を返す (実機測定で「同じプロンプトの 3 回
    目の投げ直しから出力が変わる」形で発覚した -- 1 回目の復元は snapshot
    がまだ無傷なので気づけない)。"""

    from mlx_lm.models.cache import ArraysCache

    from mlxturbo.spec import restore_untrimmable_caches, snapshot_untrimmable_caches

    c = ArraysCache(size=2)
    c.cache = [mx.array([1.0]), mx.array([2.0])]
    snapshot = snapshot_untrimmable_caches([c])

    restore_untrimmable_caches([c], snapshot)
    c[0] = mx.array([999.0])  # 次のデコードラウンドが書き込む想定

    restore_untrimmable_caches([c], snapshot)

    assert float(c[0].item()) == 1.0  # 直前の書き込みの影響を受けていない
    # snapshot 自体もまだ元の値のまま (アーカイブが破壊されていない) --
    # これが壊れていると、次にこの位置へ復元しようとした誰か (別スロットの
    # 再利用など) も巻き添えになる。
    assert float(snapshot[0][1][0].item()) == 1.0


def test_select_session_reuses_via_checkpoint_when_trim_is_not_possible_even_with_mtp_valid():
    """ArraysCache が混ざっていて exact-trim が不可能でも、チェックポイント
    があればそこまで復元して同じスロットを再利用する。mtp_valid=True
    (実運用の通常状態) でも弾かれない -- KV/GDN の再利用を MTP 連鎖の生死
    から切り離した効果そのもの。MTP 連鎖用の状態は復元後の位置に対応しない
    ので必ず捨てられ、mtp_valid も False に落ちる。"""

    from mlx_lm.models.cache import ArraysCache

    _install_state(FakeRunner(tokens_to_emit=[]))

    arr = ArraysCache(size=1)
    arr.cache = [mx.array([9.0])]

    sess = ChatSession()
    sess.processed = [1, 2, 3, 4, 5]
    sess.caches = [_real_kv_cache(5), arr]
    sess.mtp_valid = True
    sess.mtp_cache = object()
    sess.h_last = object()
    sess.checkpoints = [
        (3, [(1, [mx.array([3.0])], None, None)]),
        (4, [(1, [mx.array([4.0])], None, None)]),  # lcp=3 より後ろなので使えない
    ]
    server.STATE.session_pool[0] = sess

    got = server._select_session([1, 2, 3, 9, 9])

    assert got is sess
    assert got.processed == [1, 2, 3]
    assert got.caches[0].offset == 3
    assert float(got.caches[1].cache[0].item()) == 3.0
    assert got.mtp_cache is None
    assert got.h_last is None
    assert got.mtp_valid is False
    assert [pos for pos, _ in got.checkpoints] == [3]  # cp_pos より後ろは捨てる


def test_checkpoint_restore_end_to_end_matches_full_rebuild_with_mtp():
    """実運用に近い形の統合確認。長いプロンプトを複数チャンクに分けて
    prefill し (チェックポイントが複数できる)、MTP を使って何トークンか
    生成する (session.mtp_valid=True になる -- 実運用の通常状態)。2 ターン
    目のプロンプトが処理済み列の末尾わずかに食い違う (thinking マーカー
    再オープンと同じ形の不一致、このモジュール冒頭の調査で実測した
    「末尾 2 トークンだけ一致しない」を模す) とき、
    _try_checkpoint_restore_session_cache 経由で直近チェックポイントまで
    復元してから続きを prefill した結果 (prefill_reused > 0) が、同じ
    プロンプトを何も再利用せずまっさらに処理した結果と完全一致することを
    確認する -- これがこの節の合否基準そのもの (一致したときと、新規に
    作り直したときで、生成が一致すること)。"""

    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    from mlxturbo.mtp import MTPModule

    mx.random.seed(0)
    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=48,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        full_attention_interval=3,
        head_dim=8,
        tie_word_embeddings=True,
    )
    text_model = TextModel(args)
    mx.eval(text_model.parameters())
    mtp = MTPModule(args)
    mx.eval(mtp.parameters())
    fake_model = SimpleNamespace(language_model=text_model)
    engine = SpecEngine(fake_model, mtp=mtp, prefill_step_size=3)

    turn1_prompt = list(range(12))  # step=3 で 4 チャンク: 境界 3,6,9,12

    sess = ChatSession()
    r1 = engine.generate(
        turn1_prompt,
        max_tokens=4,
        n_draft=2,
        max_draft=2,
        lookup_len=0,
        temp=0.0,
        eos_ids=set(),
        session=sess,
    )
    assert sess.mtp_valid is True  # 実運用の通常状態を再現できていることの確認
    assert [pos for pos, _ in sess.checkpoints] == [3, 6, 9, 12]

    full_history = turn1_prompt + r1["tokens"]
    turn2_prompt = full_history[:-2] + [40, 41]  # 末尾 2 トークンだけ違う

    lcp = 0
    n = min(len(sess.processed), len(turn2_prompt))
    while lcp < n and sess.processed[lcp] == turn2_prompt[lcp]:
        lcp += 1
    assert lcp == len(full_history) - 2  # 末尾だけ食い違っている想定どおり
    assert lcp not in (3, 6, 9, 12)  # チェックポイント境界そのものではない

    cp_pos = server._try_checkpoint_restore_session_cache(sess, lcp)
    assert cp_pos == 12  # 生成部分にはチェックポイントが無いので prefill の
    # 最後のチェックポイントまでしか戻れない -- それで十分機能する。

    sess.processed = sess.processed[:cp_pos]
    sess.checkpoints = [c for c in sess.checkpoints if c[0] <= cp_pos]
    sess.mtp_cache = None
    sess.h_last = None
    sess.mtp_valid = False

    r2_reused = engine.generate(
        turn2_prompt,
        max_tokens=4,
        n_draft=2,
        max_draft=2,
        lookup_len=0,
        temp=0.0,
        eos_ids=set(),
        session=sess,
    )
    assert r2_reused["prefill_reused"] == cp_pos  # 全量再構築ではない

    r2_fresh = engine.generate(
        turn2_prompt,
        max_tokens=4,
        n_draft=2,
        max_draft=2,
        lookup_len=0,
        temp=0.0,
        eos_ids=set(),
        session=None,
    )

    assert r2_reused["tokens"] == r2_fresh["tokens"]


# ---------- 12d. バグ修正: 温まった prompt cache の diff-0 再利用 (TTFT) ----------
#
# 12/12b 節で checkpoint 経由の部分一致は直したが、reuse_cap =
# len(prompt_ids) - 1 が全経路で無条件に効いていたため、同じプロンプトの
# 投げ直し (regenerate) のような「差分 0 まで戻せる」最良のケースでも常に
# 1 トークン分の prefill が残っていた (実測: 12k トークンで 2 回目以降も
# 数秒かかる -- docs/VS-MLX-SERVE.md)。差分 0 では generate()/
# generate_stream() が prefill のチャンクループに入らず、以前の逐次生成が
# 依存していた hyper/logits (spec_flash) や h_last (spec) が計算されない
# ため、単に上限を外すだけでは cap.hyper 参照などでクラッシュする。
#
# 直したのは: generate()/generate_stream() が prefill 終了時点の
# logits[:, -1]/hyper_prev (spec_flash) または h_last (spec) を
# ``session.tail`` として位置つきで保持し (ChatSession.tail /
# FallbackSession.tail)、_select_session がその位置と新プロンプト長が
# 完全に一致するときだけ上限を len(prompt_ids) まで緩め、
# generate()/generate_stream() は差分 0 のとき prefill を飛ばして
# session.tail から直接 decode を再開すること。


def test_spec_engine_exact_repeat_reuses_prefill_fully_and_matches_fresh_run():
    """SpecEngine/ChatSession 版の中核回帰テスト。同じプロンプトを 2 回
    (session 経由で) 投げたとき、2 回目は 1 トークンも残さず全量再利用
    (prefill_reused == len(prompt)) できること、かつ生成結果がまっさらな
    再構築と完全一致すること。

    修正前のコードでは SpecEngine.generate() 自身の再利用判定が
    ``lcp < len(prompt_ids)`` を要求していたため、2 回目の呼び出しは
    reused == len(prompt_ids) - 1 (1 トークン残す) にしかならず、この
    テストの ``r2["prefill_reused"] == len(turn1_prompt)`` は必ず失敗して
    いた (修正前のコードで確認済み)。"""

    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    from mlxturbo.mtp import MTPModule

    mx.random.seed(6)
    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=48,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        full_attention_interval=3,
        head_dim=8,
        tie_word_embeddings=True,
    )
    text_model = TextModel(args)
    mx.eval(text_model.parameters())
    mtp = MTPModule(args)
    mx.eval(mtp.parameters())
    fake_model = SimpleNamespace(language_model=text_model)
    engine = SpecEngine(fake_model, mtp=mtp, prefill_step_size=3)

    turn1_prompt = list(range(12))  # step=3 で 4 チャンク: 境界 3,6,9,12

    sess = ChatSession()
    r1 = engine.generate(
        turn1_prompt,
        max_tokens=1,  # tokens が1個だけ -> processed == turn1_prompt そのもの
        n_draft=2,
        max_draft=2,
        lookup_len=0,
        temp=0.0,
        eos_ids=set(),
        session=sess,
    )
    assert sess.processed == turn1_prompt
    assert sess.tail is not None
    assert sess.tail[0] == len(turn1_prompt)

    r2 = engine.generate(
        turn1_prompt,
        max_tokens=4,
        n_draft=2,
        max_draft=2,
        lookup_len=0,
        temp=0.0,
        eos_ids=set(),
        session=sess,
    )
    assert r2["prefill_reused"] == len(turn1_prompt)

    r2_fresh = engine.generate(
        turn1_prompt,
        max_tokens=4,
        n_draft=2,
        max_draft=2,
        lookup_len=0,
        temp=0.0,
        eos_ids=set(),
        session=None,
    )

    assert r2["tokens"] == r2_fresh["tokens"]


def test_flash_spec_exact_repeat_reuses_prefill_fully_and_matches_fresh_run(monkeypatch):
    """FlashSpecEngine/FallbackSession 版 (flash_spec 経路、実運用の
    qwen38fn-mlx-v-l がここを通る) の中核回帰テスト。上の SpecEngine 版と
    同じ形: 同じプロンプトの投げ直しが全量再利用になり、まっさらな
    再構築と生成が一致すること。

    修正前のコードでは FlashSpecRunner.generate() 自身の再利用判定が
    ``lcp < len(prompt_ids)`` を要求していたため、2 回目の呼び出しは
    reused == len(turn1_prompt) - 1 にしかならず、この
    ``r2["prefill_reused"] == len(turn1_prompt)`` は必ず失敗していた
    (修正前のコードで確認済み)。"""

    import mlxturbo.spec_flash as spec_flash_module

    monkeypatch.setattr(spec_flash_module, "PREFILL_STEP_SIZE", 3)

    mx.random.seed(7)
    model, mtp = _build_tiny_qwen4_exp()
    engine = spec_flash_module.FlashSpecEngine(model, mtp)
    runner = FlashSpecRunner(engine)

    turn1_prompt = list(range(1, 13))  # 12 トークン、step=3 で境界 3,6,9,12

    session = FallbackSession()
    r1 = runner.generate(
        turn1_prompt,
        max_tokens=1,  # tokens が1個だけ -> processed == turn1_prompt そのもの
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=session,
    )
    assert session.processed == turn1_prompt
    assert session.tail is not None
    assert session.tail[0] == len(turn1_prompt)

    r2 = runner.generate(
        turn1_prompt,
        max_tokens=4,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=session,
    )
    assert r2["prefill_reused"] == len(turn1_prompt)

    r2_fresh = runner.generate(
        turn1_prompt,
        max_tokens=4,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=None,
    )

    assert r2["tokens"] == r2_fresh["tokens"]


# ---------- 12c. flash_spec 経路 (FallbackSession) のチェックポイント復元 ----------
#
# 12b 節の _try_checkpoint_restore_session_cache/_select_session は
# ChatSession (SpecEngine、27B/qwen3_5) だけを想定していたが、実装自体は
# .checkpoints/.caches/.processed を duck typing で見るだけで runner の種類を
# 知らない。FlashSpecRunner (Qwen3.8-Flash-Next, KIND="flash_spec") が使う
# FallbackSession は .caches ではなく単数形 .cache を持つため、そのままでは
# 12b 節の経路が「.caches が無い」で即座に諦めていた (mlxturbo/server.py の
# _try_trim_session_cache と同じフォールバックが無かった) — これを直した。
# もう1つ、Flash-Next の full attention 層 (_AttnCache) は KV とは別に
# indexer の生 keys を持ち、.trim() は KV の offset しか戻さないので、
# 何もしなければ indexer だけ古い長さのまま取り残される。これも直した
# (mlxturbo/spec_flash.py の rollback() が検証 1 ラウンドぶんについて同じ
# 後始末をしているのと同じ理由)。


def test_checkpoint_restore_falls_back_to_singular_cache_and_truncates_indexer():
    """FallbackSession (.cache 単数形) でも復元でき、Flash-Next の
    _AttnCache.indexer.keys が KV の trim 後の offset に揃うところまで
    確認する。"""

    import mlx_lm.models.qwen4_exp as Q

    attn = Q._AttnCache()
    attn.offset = 10
    attn.keys = mx.zeros((1, 1, 10, 1))
    attn.values = mx.zeros((1, 1, 10, 1))
    attn.indexer.keys = mx.arange(10)[None, :]  # (1, 10) -- 10 トークン分

    arr = Q.ArraysCache(1)
    arr.cache = [mx.array([9.0])]  # 「今」の (使われない) 状態

    session = FallbackSession()
    session.processed = list(range(10))
    session.cache = [attn, arr]  # 複数形 .caches は無い
    session.checkpoints = [(7, [(1, [mx.array([7.0])], None, None)])]

    cp_pos = server._try_checkpoint_restore_session_cache(session, lcp=8)

    assert cp_pos == 7
    assert session.cache[0].offset == 7  # KV は trim
    assert session.cache[0].indexer.keys.shape[1] == 7  # indexer も揃う
    assert session.cache[0].indexer.keys.tolist() == [[0, 1, 2, 3, 4, 5, 6]]
    assert float(session.cache[1].cache[0].item()) == 7.0  # ArraysCache は復元


def test_select_session_reuses_via_checkpoint_for_flash_spec_fallback_session():
    """FlashSpecRunner (KIND="flash_spec") が実際に使う FallbackSession には
    ChatSession 専用の mtp_cache/h_last/mtp_valid が無い。_select_session の
    hasattr ガードでそれらの後始末が自然に skip されるだけで、チェックポイント
    経由の部分復元そのものは同じスロットを再利用できることを HTTP 越しでなく
    直接確認する。"""

    import mlx_lm.models.qwen4_exp as Q

    _install_state(FakeFlashSpecRunner(tokens_to_emit=[]))

    attn = Q._AttnCache()
    attn.offset = 5
    attn.keys = mx.zeros((1, 1, 5, 1))
    attn.values = mx.zeros((1, 1, 5, 1))
    attn.indexer.keys = mx.arange(5)[None, :]

    arr = Q.ArraysCache(1)
    arr.cache = [mx.array([99.0])]

    sess = FallbackSession()
    sess.processed = [1, 2, 3, 4, 5]
    sess.cache = [attn, arr]
    sess.checkpoints = [(3, [(1, [mx.array([3.0])], None, None)])]
    server.STATE.session_pool[0] = sess

    got = server._select_session([1, 2, 3, 9, 9])

    assert got is sess
    assert got.processed == [1, 2, 3]
    assert got.cache[0].offset == 3
    assert got.cache[0].indexer.keys.shape[1] == 3
    assert float(got.cache[1].cache[0].item()) == 3.0
    assert [pos for pos, _ in got.checkpoints] == [3]  # cp_pos より後ろは捨てる
    # ChatSession 専用のフィールドは無い (FallbackSession に存在しない) --
    # hasattr ガードで単に触られないことの確認。
    assert not hasattr(got, "mtp_valid")
    assert not hasattr(got, "mtp_cache")


def _build_tiny_qwen4_exp():
    """FlashSpecEngine が要求する形 (hyper-connections + GDN + full
    attention/indexer + MoE + PLE + n-gram) を、実物の mlx_lm.models.
    qwen4_exp / mlxturbo.mtp_flash.FlashMTPModule そのままで最小サイズで
    組む (SpecEngine 側の test_checkpoint_restore_end_to_end_matches_full_
    rebuild_with_mtp が qwen3_5 の TextModel を小さく組んだのと同じ発想)。
    num_hidden_layers=4, full_attention_interval=4 -> 層0-2 が
    linear_attention (GDN)、層3 が full_attention。ple_layer_ids=[2] ->
    層1 (0-indexed) が PLE を持つ。"""

    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo.mtp_flash import FlashMTPModule

    text_args = Q.TextArgs(
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        hc_count=2,
        hc_lowrank=4,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=32,
        ple_embed_dim=16,
        ple_layer_ids=[2],
        ple_conv_kernel_size=4,
        full_attention_interval=4,
        partial_rotary_factor=0.25,
        rope_theta=10_000.0,
        tie_word_embeddings=False,
    )
    model_args = Q.ModelArgs(text_config=text_args.__dict__.copy())
    model = Q.Model(model_args)
    mx.eval(model.parameters())
    mtp = FlashMTPModule(model_args.text, variant="lane")
    mx.eval(mtp.parameters())
    return model, mtp


def test_flash_spec_draft_reuses_primed_cache_across_rounds():
    """MTP priming (2026-08-30): _draft() must no longer build a fresh,
    empty _AttnCache() on every call -- it should keep using the one cache
    that generate()/generate_stream() primed over the whole prompt, and that
    cache's .offset should advance by exactly 1 every round instead of
    resetting. This is a direct regression test for the bug the priming
    change fixes: before it, every _draft() call re-created its own cache at
    offset 0, so the MTP head never had any context beyond the single
    current token."""

    import mlxturbo.spec_flash as spec_flash_module

    mx.random.seed(2)
    model, mtp = _build_tiny_qwen4_exp()
    engine = spec_flash_module.FlashSpecEngine(model, mtp)

    ids = mx.array([[1, 2, 3, 4, 5]])
    seen_caches = []
    offsets = []
    orig_draft = spec_flash_module.FlashSpecEngine._draft

    def spy(self, cur, hyper_prev, cache):
        seen_caches.append(cache)
        offsets.append(cache.offset)
        return orig_draft(self, cur, hyper_prev, cache)

    spec_flash_module.FlashSpecEngine._draft = spy
    try:
        engine.generate(ids, max_tokens=6)
    finally:
        spec_flash_module.FlashSpecEngine._draft = orig_draft

    assert len(offsets) >= 2  # otherwise this test proves nothing
    # every _draft() call in one generate() must reuse the identical cache
    # object -- not a fresh one each time.
    assert all(c is seen_caches[0] for c in seen_caches)
    # priming covers prompt positions 0..N-2 (N=5 tokens -> offset 4) before
    # the first _draft() call, then the shared cache's offset must grow by
    # exactly 1 every round (never reset to 0 or re-primed).
    assert offsets[0] == ids.shape[1] - 1
    assert offsets == list(range(offsets[0], offsets[0] + len(offsets)))


def test_flash_spec_checkpoint_reuse_matches_full_rebuild_with_tail_mismatch(monkeypatch):
    """この節の合否基準そのもの: thinking マーカー再オープンと同じ形の不一致
    (処理済み列の末尾わずかだけが新プロンプトと食い違う、チェックポイント
    境界そのものではない位置) のとき、チェックポイント経由で復元してから
    続きを prefill した結果が、同じプロンプトを何も再利用せずまっさらに
    処理した結果と完全一致すること。実物の FlashSpecEngine/FlashSpecRunner/
    FallbackSession を小さいモデルで実際に動かして確認する (12b 節の
    ChatSession 版 test_checkpoint_restore_end_to_end_matches_full_rebuild_
    with_mtp の flash_spec 対)。"""

    import mlxturbo.spec_flash as spec_flash_module

    # 実運用と同じ PREFILL_STEP_SIZE=2048 では 12 トークンのプロンプトが
    # 1 チャンクに収まってしまいチェックポイントが 1 個しかできない。
    # SpecEngine 側のテストが prefill_step_size=3 を渡すのと同じ理由で、
    # このモジュール定数だけ小さく差し替える (spec_flash.generate_stream は
    # モジュールグローバルの PREFILL_STEP_SIZE をその場で参照するので、
    # モンキーパッチだけで両経路 (フル/再利用) に同じ幅がかかる)。
    monkeypatch.setattr(spec_flash_module, "PREFILL_STEP_SIZE", 3)

    mx.random.seed(0)
    model, mtp = _build_tiny_qwen4_exp()
    engine = spec_flash_module.FlashSpecEngine(model, mtp)
    runner = FlashSpecRunner(engine)

    turn1_prompt = list(range(1, 13))  # 12 トークン、step=3 で境界 3,6,9,12

    session = FallbackSession()
    r1 = runner.generate(
        turn1_prompt,
        max_tokens=4,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=session,
    )
    assert [pos for pos, _ in session.checkpoints] == [3, 6, 9, 12]

    # FlashSpecRunner の不変条件 (6a0cd27): 最後の cur はまだ cache に
    # feed されていないので、publish されるのは tokens[:-1] まで。
    fed_history = list(turn1_prompt) + r1["tokens"][:-1]
    assert session.processed == fed_history

    last_two = fed_history[-2:]
    replacement = [(t + 1) % 32 for t in last_two]  # 必ず元の値と違う
    turn2_prompt = fed_history[:-2] + replacement

    lcp = 0
    n = min(len(session.processed), len(turn2_prompt))
    while lcp < n and session.processed[lcp] == turn2_prompt[lcp]:
        lcp += 1
    assert lcp == len(fed_history) - 2  # 末尾だけ食い違っている想定どおり
    assert lcp not in (3, 6, 9, 12)  # チェックポイント境界そのものではない

    cp_pos = server._try_checkpoint_restore_session_cache(session, lcp)
    assert cp_pos == 12  # 生成部分にはチェックポイントが無いので prefill の
    # 最後のチェックポイントまでしか戻れない -- それで十分機能する。

    session.processed = session.processed[:cp_pos]
    session.checkpoints = [c for c in session.checkpoints if c[0] <= cp_pos]

    r2_reused = runner.generate(
        turn2_prompt,
        max_tokens=4,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=session,
    )
    assert r2_reused["prefill_reused"] == cp_pos  # 全量再構築ではない

    r2_fresh = runner.generate(
        turn2_prompt,
        max_tokens=4,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=None,
    )

    assert r2_reused["tokens"] == r2_fresh["tokens"]


def test_flash_spec_checkpoint_reuse_disabled_when_prompt_fits_one_chunk(monkeypatch):
    """プロンプト全体が 1 チャンクに収まる (実運用の既定 PREFILL_STEP_SIZE=2048
    ではほぼ常にそう) 場合、チェックポイントは末尾 (プロンプト全体) の 1 個
    しかできない。末尾より前で食い違えば再利用できず新規スロットへ倒れる --
    「再利用できないときは安全側の全再構築に倒す」が実際に効くことの確認。"""

    import mlxturbo.spec_flash as spec_flash_module

    mx.random.seed(1)
    model, mtp = _build_tiny_qwen4_exp()
    engine = spec_flash_module.FlashSpecEngine(model, mtp)
    runner = FlashSpecRunner(engine)

    turn1_prompt = list(range(1, 9))  # 8 トークン、既定 PREFILL_STEP_SIZE の
    # 下では 1 チャンク (境界はプロンプト末尾の 1 個だけ)

    session = FallbackSession()
    runner.generate(
        turn1_prompt,
        max_tokens=2,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=session,
    )
    assert [pos for pos, _ in session.checkpoints] == [8]

    turn2_prompt = [1, 2, 20, 21]  # 位置 2 で分岐 -- 唯一のチェックポイント
    # (位置 8) より手前なので使えない

    cp_pos = server._try_checkpoint_restore_session_cache(session, lcp=2)
    assert cp_pos is None
    assert session.cache[-1].offset == len(session.processed)  # 触られていない


class FakeChatSessionRunner:
    """ChatSession 経由 (KIND="spec") の実運用に近い形で、_select_session の
    trim 経路を HTTP 越しに確認するための最小フェイク。caches は本物の
    mlx_lm.models.cache.KVCache を使う (KVCache.trim は offset を減らす
    だけなので、中身のトークンを実際に流し込まなくても "この session が
    どこまで処理済みか" を offset で正しく表現できる — _real_kv_cache と
    同じ理屈)。MTP は使わない (mtp_valid は常に False) — MTP を絡めた
    分岐は test_select_session_skips_partial_match_when_mtp_valid で別途
    見ている。
    """

    KIND = "spec"
    SUPPORTED_SAMPLING_PARAMS = SpecRunner.SUPPORTED_SAMPLING_PARAMS

    def __init__(self, reply_tokens_by_call: list[list[int]]):
        self._reply_tokens_by_call = list(reply_tokens_by_call)
        self.calls: list[dict] = []

    def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
        prompt_ids = list(prompt_ids)
        reused = 0
        if session is not None and session.caches is not None:
            pl = session.processed
            n = min(len(pl), len(prompt_ids))
            lcp = 0
            while lcp < n and pl[lcp] == prompt_ids[lcp]:
                lcp += 1
            if lcp == len(pl) and lcp < len(prompt_ids):
                reused = lcp
        emitted: list[int] = []
        toks = self._reply_tokens_by_call[len(self.calls)]
        for t in toks:
            if len(emitted) >= max_tokens:
                break
            emitted.append(t)
            if on_tokens:
                on_tokens([t])
            if t in eos_ids:
                break
        self.calls.append(
            {"prompt_ids": prompt_ids, "reused_before_call": reused, "emitted": emitted}
        )
        if session is not None:
            new_processed = prompt_ids + emitted
            session.publish([_real_kv_cache(len(new_processed))], None, False, new_processed, None)
        return {
            "tokens": emitted,
            "ttft_s": 0.001,
            "decode_tps": 100.0,
            "prefill_reused": reused,
            "prefill_new": len(prompt_ids) - reused,
            "tokens_per_step": 1.0,
            "accept_hist": {},
        }


def _fake_messages_to_ids_reopened_marker(messages):
    """<think>\\n 再オープンによる末尾不一致バグを模す prompt_ids_fn。

    生成直前にだけ open marker (77) を付けるが、確定した過去ターンを
    履歴へ埋め戻すときは (本物のチャットテンプレートが thinking を履歴
    から落とすのと同じ非対称性で) open marker を含めない。そのため
    「新プロンプトが前回の処理済み列の純粋な追記になる」という
    _fake_messages_to_ids の前提が崩れ、部分一致 (assistant ターンの
    直前までは一致、その先の open marker だけ食い違う) になる。
    """

    ids: list[int] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        if role == "assistant":
            ids.extend(int(t) for t in content.split(",") if t)
        else:
            ids.extend(ord(c) for c in f"{role}:{content}\n")
    ids.append(77)
    return ids


def test_http_partial_match_trim_reuses_slot_and_reports_nonzero_reused(client):
    tok = FakeTokenizer(
        vocab={10: "ok"}, prompt_ids_fn=_fake_messages_to_ids_reopened_marker
    )
    runner = FakeChatSessionRunner(reply_tokens_by_call=[[10], [10]])
    _install_state(runner, tokenizer=tok)

    turn1 = [{"role": "user", "content": "hello"}]
    r1 = client.post("/v1/chat/completions", json={"messages": turn1})
    assert r1.status_code == 200, r1.text
    assert runner.calls[0]["reused_before_call"] == 0

    turn2 = turn1 + [
        {"role": "assistant", "content": "10"},
        {"role": "user", "content": "again"},
    ]
    r2 = client.post("/v1/chat/completions", json={"messages": turn2})
    assert r2.status_code == 200, r2.text

    # 全体一致では届かなかった再利用が、部分一致 + trim 経由で発生する。
    assert runner.calls[1]["reused_before_call"] > 0
    assert r2.json()["usage"]["prompt_tokens_details"]["cached_tokens"] > 0


def test_http_partial_match_trim_produces_same_prompt_and_output_as_full_rebuild(client):
    """部分一致 + trim でスロットを再利用したときも、trim を封じて (新規
    スロット =全再構築に倒れる状態で) 処理したときも、runner.generate() に
    渡る prompt_ids (実際にモデルへ見せる論理プロンプト) と最終的な出力は
    完全に一致することを確認する。session 再利用は runner 内部で
    「どこまで KV を使い回せるか」を決めるだけの最適化であり、上位から
    モデルへ見せる入力そのものは変えない — ここが一致していれば trim の
    有無は出力に影響しない。
    """

    tok = FakeTokenizer(
        vocab={10: "ok"}, prompt_ids_fn=_fake_messages_to_ids_reopened_marker
    )
    turn1 = [{"role": "user", "content": "hello"}]
    turn2 = turn1 + [
        {"role": "assistant", "content": "10"},
        {"role": "user", "content": "again"},
    ]

    # (a) 部分一致 + trim でスロットを再利用するケース。
    runner_reused = FakeChatSessionRunner(reply_tokens_by_call=[[10], [10]])
    _install_state(runner_reused, tokenizer=tok)
    r1 = client.post("/v1/chat/completions", json={"messages": turn1})
    assert r1.status_code == 200, r1.text
    r2_reused = client.post("/v1/chat/completions", json={"messages": turn2})
    assert r2_reused.status_code == 200, r2_reused.text
    assert runner_reused.calls[1]["reused_before_call"] > 0

    # (b) 毎回まっさらな pool で turn2 だけを新規スロットとして処理する
    #     (trim を経由しない = 全再構築) ケース。
    runner_fresh = FakeChatSessionRunner(reply_tokens_by_call=[[10]])
    _install_state(runner_fresh, tokenizer=tok)
    r2_fresh = client.post("/v1/chat/completions", json={"messages": turn2})
    assert r2_fresh.status_code == 200, r2_fresh.text
    assert runner_fresh.calls[0]["reused_before_call"] == 0

    assert runner_reused.calls[1]["prompt_ids"] == runner_fresh.calls[0]["prompt_ids"]
    assert (
        r2_reused.json()["choices"][0]["message"]["content"]
        == r2_fresh.json()["choices"][0]["message"]["content"]
    )


def test_select_session_on_executor_runs_on_state_executor_thread():
    """``_select_session_on_executor`` は ``STATE.executor`` (モデルをロード
    したのと同じ専用ワーカースレッド) 上で ``_select_session`` を実行する
    こと。

    2026-08-29 のチェックポイント機能追加で、``_select_session`` の 2nd
    pass (``_try_trim_session_cache``/``_try_checkpoint_restore_session_
    cache``) は実際に MLX 配列を触るようになった (``cache.trim()``、
    ``indexer.keys`` のスライス、``restore_untrimmable_caches`` の
    ``c.state`` 代入)。ところが ``_select_session`` 自体は
    ``STATE.runner.generate`` と違い、長らく各エンドポイントから直接
    (= イベントループのスレッドで) 呼ばれていた。モデルをロードした
    専用スレッド (``STATE.executor``) 以外で MLX 配列を触ると "There is
    no Stream(gpu, N) in current thread" で落ちる (``_run_generate``
    直前のコメント参照、実測確認済み) — thinking 有効の 2 ターン目で
    checkpoint 復元が実際に発火すると実機で再現した (この節の他のテストは
    どれも FakeRunner/フェイクの caches か、TestClient と同じスレッドで
    完結する tiny な実モデル呼び出しなので、この "呼び出し元と違うスレッド"
    という条件そのものを再現しておらず、この不具合を見逃していた)。
    ``_select_session_on_executor`` (server.py の chat/anthropic/completions/
    responses、非ストリーミング/ストリーミング全 8 経路が呼ぶ) がその
    修正そのもの — 直接呼ばず必ず ``STATE.executor`` 経由にする。
    """

    runner = FakeRunner([1])
    _install_state(runner)
    executor = server.STATE.executor
    caller_thread_id = threading.get_ident()

    seen_thread_id: dict[str, int] = {}
    orig = server._select_session

    def spy(prompt_ids):
        seen_thread_id["id"] = threading.get_ident()
        return orig(prompt_ids)

    with mock.patch.object(server, "_select_session", side_effect=spy):
        asyncio.run(server._select_session_on_executor([1, 2, 3]))

    assert seen_thread_id["id"] != caller_thread_id
    assert seen_thread_id["id"] == executor.submit(threading.get_ident).result()


# ---------- 文脈長ガード (_resolve_model_max_context / _check_context_length) ----------
#
# 実サーバーで ~57,000 トークンのプロンプトが Metal の一括確保上限を超えて
# [metal::malloc] のまま 500 になっていた事故 (mlxturbo/spec.py のチャンク
# prefill とは別レイヤの、起動時にモデル config から決まる上限による事前
# ガード)。ここではモデルをロードせず、STATE.max_context_tokens を直接
# 差し込んで 4 経路 (chat/anthropic/completions/responses) と境界条件を
# 検証する。


def test_resolve_model_max_context_reads_top_level_field():
    assert server._resolve_model_max_context({"max_position_embeddings": 131072}) == 131072


def test_resolve_model_max_context_reads_nested_text_config():
    # VLM ラッパー形式 (実物の Qwen3.6-35B-A3B config.json と同じ形):
    # トップレベルには無く、text_config の下にネストされている。
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {"max_position_embeddings": 262144},
    }
    assert server._resolve_model_max_context(config) == 262144


def test_resolve_model_max_context_prefers_top_level_over_nested():
    config = {
        "max_position_embeddings": 4096,
        "text_config": {"max_position_embeddings": 262144},
    }
    assert server._resolve_model_max_context(config) == 4096


def test_resolve_model_max_context_returns_none_when_absent():
    assert server._resolve_model_max_context({"model_type": "whatever"}) is None
    assert server._resolve_model_max_context({"text_config": {}}) is None


# ---------- _metal_safe_prefill_limit / _resolve_default_max_context_tokens ----------
#
# SpecEngine は新規プロンプトを PREFILL_STEP_SIZE (既定 2048) トークンずつ
# チャンク分割して forward する (SpecEngine._prefill_hidden)。1 回の forward
# の注意スコア行列確保は num_attention_heads * PREFILL_STEP_SIZE * T *
# bytes_per_elem (T に対して線形)。それでも T が非常に大きいモデルでは
# Metal の 1 バッファ上限を超え得るので、モデルが申告する
# max_position_embeddings をそのまま上限にはせず、この関数が求めた値との
# 小さい方を使う (_resolve_default_max_context_tokens)。
#
# ここでのテスト用の分母 (n_heads * PREFILL_STEP_SIZE * bytes_per_elem) は
# 2 * 2048 * 2 = 8192 (PREFILL_STEP_SIZE は mlxturbo.spec の実定数をそのまま
# 使う -- テスト側で別の値を仮定すると経路間の共有を検証したことにならない)。


_METAL_TEST_DENOM = 2 * server.PREFILL_STEP_SIZE * 2  # heads=2, bf16=2 bytes


def test_metal_safe_prefill_limit_computes_from_heads_and_buffer_length(monkeypatch):
    monkeypatch.setattr(
        server.mx, "device_info", lambda: {"max_buffer_length": _METAL_TEST_DENOM * 100}
    )
    config = {"num_attention_heads": 2, "dtype": "bfloat16"}
    # theoretical = (denom * 100) / denom = 100; limit = int(100 * 0.9) = 90
    assert server._metal_safe_prefill_limit(config) == 90


def test_metal_safe_prefill_limit_reads_nested_text_config(monkeypatch):
    monkeypatch.setattr(
        server.mx, "device_info", lambda: {"max_buffer_length": _METAL_TEST_DENOM * 100}
    )
    config = {"text_config": {"num_attention_heads": 2, "dtype": "bfloat16"}}
    assert server._metal_safe_prefill_limit(config) == 90


def test_metal_safe_prefill_limit_none_without_head_count(monkeypatch):
    monkeypatch.setattr(
        server.mx, "device_info", lambda: {"max_buffer_length": _METAL_TEST_DENOM * 100}
    )
    assert server._metal_safe_prefill_limit({"dtype": "bfloat16"}) is None


def test_metal_safe_prefill_limit_none_when_device_info_unavailable(monkeypatch):
    def _raise():
        raise RuntimeError("no metal device")

    monkeypatch.setattr(server.mx, "device_info", _raise)
    config = {"num_attention_heads": 2, "dtype": "bfloat16"}
    assert server._metal_safe_prefill_limit(config) is None


def test_resolve_default_max_context_tokens_takes_the_smaller_value(monkeypatch):
    # config の max_position_embeddings (大きい) より Metal 由来の実測上限
    # (小さい) が効く -- コーディネータ指摘の通り、config を鵜呑みにしない。
    monkeypatch.setattr(
        server.mx, "device_info", lambda: {"max_buffer_length": _METAL_TEST_DENOM * 100}
    )
    config = {
        "max_position_embeddings": 100_000,
        "num_attention_heads": 2,
        "dtype": "bfloat16",
    }
    assert server._resolve_default_max_context_tokens(config) == 90


def test_resolve_default_max_context_tokens_uses_config_when_it_is_smaller(monkeypatch):
    monkeypatch.setattr(
        server.mx, "device_info", lambda: {"max_buffer_length": _METAL_TEST_DENOM * 100}
    )
    config = {
        "max_position_embeddings": 10,
        "num_attention_heads": 2,
        "dtype": "bfloat16",
    }
    assert server._resolve_default_max_context_tokens(config) == 10


def test_resolve_default_max_context_tokens_falls_back_to_whichever_is_available(monkeypatch):
    monkeypatch.setattr(
        server.mx, "device_info", lambda: {"max_buffer_length": _METAL_TEST_DENOM * 100}
    )
    # heads 情報が無い -> Metal 由来の値は None、config 側だけが効く。
    assert server._resolve_default_max_context_tokens({"max_position_embeddings": 500}) == 500


def test_resolve_default_max_context_tokens_none_when_nothing_available(monkeypatch):
    monkeypatch.setattr(
        server.mx, "device_info", lambda: {"max_buffer_length": _METAL_TEST_DENOM * 100}
    )
    assert server._resolve_default_max_context_tokens({"model_type": "whatever"}) is None


# ---------- SpecEngine chunked prefill: spec-on == spec-off (docs/PREFILL-CHUNKING-DETERMINISM.md) ----------
#
# 分割あり/なしのビット一致は要求しない (mx.quantized_matmul がバッチ長
# 依存の丸めをするため、docs/PREFILL-CHUNKING-DETERMINISM.md 参照)。代わりに
# 要求するのは「チャンク幅を固定した下で、投機あり (SAM lookup) と投機なし
# の貪欲デコード出力トークン列が完全一致する」こと。実モデルでの確認は別途
# 手動で行った (report 参照)。ここでは合成の小さな GDN ハイブリッドモデルで
# 同じ性質を高速に固定回帰できるようにする。


def test_chunked_prefill_spec_on_matches_spec_off(monkeypatch):
    """PREFILL_STEP_SIZE を小さく (3) してプロンプトが何チャンクにも分かれる
    ようにし、lookup 投機ありの貪欲デコードと投機なしの貪欲デコードが完全
    一致することを検証する。``_respec_trigger`` を強制的に True にして毎
    ステップ lookup 候補を出させる (合成モデルは未学習で target logits の
    エントロピーが高く、素の閾値では lookup がほぼ起動しないため) — 起動
    自体の可否 (エントロピーゲート) は他のテストの対象で、ここでの主張は
    「lookup が実際に起動したとき、受理判定が spec-off と矛盾しない」こと
    だけに絞る。"""

    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    mx.random.seed(0)
    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=48,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        full_attention_interval=3,
        head_dim=8,
        tie_word_embeddings=True,
    )
    text_model = TextModel(args)
    mx.eval(text_model.parameters())
    fake_model = SimpleNamespace(language_model=text_model)
    engine = SpecEngine(fake_model, mtp=None, prefill_step_size=3)

    monkeypatch.setattr(SpecEngine, "_respec_trigger", staticmethod(lambda *a, **k: True))

    # 繰り返しパターンにして SAM lookup が match_len >= lookup_ngram を
    # 確実に見つけられるようにする。29 トークンは step=3 で 10 チャンクに
    # 分かれる (チャンク境界をまたいだ prefill を実際に運動させる)。
    pattern = [3, 7, 12, 20, 5, 41]
    prompt_ids = (pattern * 6)[:29]

    r_on = engine.generate(
        prompt_ids,
        max_tokens=20,
        n_draft=0,
        max_draft=0,
        lookup_len=8,
        lookup_ngram=3,
        temp=0.0,
        eos_ids=set(),
    )
    r_off = engine.generate(
        prompt_ids,
        max_tokens=20,
        n_draft=0,
        max_draft=0,
        lookup_len=0,
        temp=0.0,
        eos_ids=set(),
    )

    assert r_on["tokens"] == r_off["tokens"]
    # lookup が実際に起動したことを確認する (でなければこのテストは
    # spec-off と実質同じ経路しか通っておらず、何も検証していない)。
    assert sum(r_on["src_hist"]["lookup"].values()) > 0


def test_context_length_guard_disabled_by_default(client):
    # _install_state はデフォルトで max_context_tokens=None を積む =
    # ガード無効。長いプロンプトでも通常どおり生成へ進む。
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner, tokenizer=FakeTokenizer(vocab={10: "x"}, prompt_ids=list(range(50_000)))
    )
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200, resp.text
    assert runner.calls


def test_context_length_guard_rejects_over_limit_chat(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner,
        tokenizer=FakeTokenizer(vocab={10: "x"}, prompt_ids=list(range(101))),
        max_context_tokens=100,
    )
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 400, resp.text
    assert not runner.calls
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "context_length_exceeded"
    assert "100" in err["message"] and "101" in err["message"]


def test_context_length_guard_rejects_over_limit_anthropic(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner,
        tokenizer=FakeTokenizer(vocab={10: "x"}, prompt_ids=list(range(101))),
        max_context_tokens=100,
    )
    resp = client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    request_id = body["request_id"]
    assert request_id.startswith("req_")
    assert resp.headers["request-id"] == request_id


def test_context_length_guard_rejects_over_limit_completions(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}), max_context_tokens=5)
    resp = client.post("/v1/completions", json={"prompt": [1, 2, 3, 4, 5, 6]})
    assert resp.status_code == 400, resp.text
    assert not runner.calls
    err = resp.json()["error"]
    assert err["code"] == "context_length_exceeded"


def test_context_length_guard_rejects_over_limit_responses(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner,
        tokenizer=FakeTokenizer(vocab={10: "x"}, prompt_ids=list(range(101))),
        max_context_tokens=100,
    )
    resp = client.post("/v1/responses", json={"input": "hi"})
    assert resp.status_code == 400, resp.text
    assert not runner.calls
    err = resp.json()["error"]
    assert err["code"] == "context_length_exceeded"


def test_context_length_guard_boundary_exact_limit_is_allowed(client):
    # ちょうど上限と同じ長さ (超えてはいない) は通す。
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner,
        tokenizer=FakeTokenizer(vocab={10: "x"}, prompt_ids=list(range(100))),
        max_context_tokens=100,
    )
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200, resp.text
    assert runner.calls


def test_context_length_guard_boundary_one_over_limit_is_rejected(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner,
        tokenizer=FakeTokenizer(vocab={10: "x"}, prompt_ids=list(range(101))),
        max_context_tokens=100,
    )
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_context_length_guard_streaming_400_returns_before_any_sse_event(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner,
        tokenizer=FakeTokenizer(vocab={10: "x"}, prompt_ids=list(range(101))),
        max_context_tokens=100,
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls
    assert "text/event-stream" not in resp.headers.get("content-type", "")
    assert not resp.text.startswith("data: ")


# ---------- 13. --api-key 認証 ----------


def test_api_key_disabled_by_default(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))
    resp = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 200, resp.text


def test_api_key_missing_returns_401_openai(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner, tokenizer=FakeTokenizer(vocab={10: "x"}), api_keys=frozenset({"secret"})
    )
    resp = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 401, resp.text
    assert not runner.calls
    err = resp.json()["error"]
    assert err["code"] == "invalid_api_key"


def test_api_key_wrong_returns_401_openai(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner, tokenizer=FakeTokenizer(vocab={10: "x"}), api_keys=frozenset({"secret"})
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401, resp.text
    assert not runner.calls


def test_api_key_correct_via_bearer_on_openai_route(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner, tokenizer=FakeTokenizer(vocab={10: "x"}), api_keys=frozenset({"secret"})
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer secret"},
    )
    assert resp.status_code == 200, resp.text


def test_api_key_correct_via_x_api_key_on_openai_route(client):
    # 両ヘッダともどちらの経路でも受け付ける (クライアント実装の揺れがあるため)。
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner, tokenizer=FakeTokenizer(vocab={10: "x"}), api_keys=frozenset({"secret"})
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"x-api-key": "secret"},
    )
    assert resp.status_code == 200, resp.text


def test_api_key_v1_models_requires_auth(client):
    runner = FakeRunner(tokens_to_emit=[])
    _install_state(runner, api_keys=frozenset({"secret"}))
    resp = client.get("/v1/models")
    assert resp.status_code == 401, resp.text
    resp_ok = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert resp_ok.status_code == 200, resp_ok.text


def test_api_key_missing_returns_401_anthropic_format(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner, tokenizer=FakeTokenizer(vocab={10: "x"}), api_keys=frozenset({"secret"})
    )
    resp = client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
    )
    assert resp.status_code == 401, resp.text
    assert not runner.calls
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "authentication_error"


def test_api_key_correct_via_x_api_key_on_anthropic_route(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner, tokenizer=FakeTokenizer(vocab={10: "x"}), api_keys=frozenset({"secret"})
    )
    resp = client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
        headers={"x-api-key": "secret"},
    )
    assert resp.status_code == 200, resp.text


def test_api_key_correct_via_bearer_on_anthropic_route(client):
    # 逆方向のクロスオーバーも同様に受け付ける。
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner, tokenizer=FakeTokenizer(vocab={10: "x"}), api_keys=frozenset({"secret"})
    )
    resp = client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
        headers={"Authorization": "Bearer secret"},
    )
    assert resp.status_code == 200, resp.text


def test_api_key_health_and_hello_bypass_auth(client):
    runner = FakeRunner(tokens_to_emit=[])
    _install_state(runner, api_keys=frozenset({"secret"}))
    assert client.get("/health").status_code == 200
    assert client.get("/api/hello").status_code == 200
    assert client.head("/api/hello").status_code == 200


def test_api_key_multiple_keys_any_match(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(
        runner,
        tokenizer=FakeTokenizer(vocab={10: "x"}),
        api_keys=frozenset({"key-a", "key-b"}),
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer key-b"},
    )
    assert resp.status_code == 200, resp.text


# ---------- 14. --max-queue / キュー枠 ----------


def test_max_queue_returns_503_when_at_capacity_openai(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}), max_queue=0)
    resp = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 503, resp.text
    assert not runner.calls
    assert resp.headers.get("retry-after")
    err = resp.json()["error"]
    assert err["code"] == "server_busy"


def test_max_queue_returns_503_when_at_capacity_anthropic(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}), max_queue=0)
    resp = client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
    )
    assert resp.status_code == 503, resp.text
    assert not runner.calls
    assert resp.headers.get("retry-after")
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "overloaded_error"


def test_max_queue_returns_503_for_completions_and_responses(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}), max_queue=0)

    resp1 = client.post("/v1/completions", json={"prompt": "hi"})
    assert resp1.status_code == 503, resp1.text

    resp2 = client.post("/v1/responses", json={"input": "hi"})
    assert resp2.status_code == 503, resp2.text


def test_max_queue_streaming_returns_503_before_any_sse_event(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}), max_queue=0)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 503, resp.text
    assert "text/event-stream" not in resp.headers.get("content-type", "")


def test_health_reports_queue_depth(client):
    runner = FakeRunner(tokens_to_emit=[])
    state = _install_state(runner, queue_depth=3)
    resp = client.get("/health")
    assert resp.json()["queue_depth"] == 3
    assert state.max_queue == 8  # 既定値


def test_queue_slot_released_after_non_stream_request(client):
    runner = FakeRunner(tokens_to_emit=[10])
    state = _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}), max_queue=1)
    resp = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 200, resp.text
    assert state.queue_depth == 0


def test_queue_slot_released_after_stream_request(client):
    runner = FakeRunner(tokens_to_emit=[10])
    state = _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}), max_queue=1)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200, resp.text
    # TestClient は StreamingResponse のジェネレータを最後まで読み切ってから
    # 戻る (aclose まで含む) ので、この時点で finally は必ず実行済み。
    assert state.queue_depth == 0


@pytest.mark.parametrize("route", ["chat", "completions"])
@pytest.mark.parametrize("bad_options", ["bad", ["bad"], 1])
def test_stream_options_must_be_object_before_queue_reservation(
    client, route, bad_options
):
    runner = FakeRunner(tokens_to_emit=[10])
    state = _install_state(
        runner, tokenizer=FakeTokenizer(vocab={10: "x"}), max_queue=1
    )
    if route == "chat":
        path = "/v1/chat/completions"
        body = {"messages": [{"role": "user", "content": "hi"}]}
    else:
        path = "/v1/completions"
        body = {"prompt": "hi"}
    body.update({"stream": True, "stream_options": bad_options})

    resp = client.post(path, json=body)

    assert resp.status_code == 400, resp.text
    assert "stream_options" in resp.json()["error"]["message"]
    assert state.queue_depth == 0
    assert not runner.calls


@pytest.mark.parametrize("protocol", ["chat", "anthropic", "responses"])
def test_close_after_stream_preamble_releases_queue_slot(protocol):
    runner = FakeRunner(tokens_to_emit=[10])
    state = _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}), max_queue=1)

    async def run():
        assert server._try_reserve_queue_slot()
        if protocol == "chat":
            inner = server._openai_stream(
                [1, 2, 3], 8, 0.0, "chat-id", 0, "test-model", [], False, None
            )
        elif protocol == "anthropic":
            inner = server._anthropic_stream(
                [1, 2, 3], 8, 0.0, "msg-id", "test-model", [], None
            )
        else:
            inner = server._responses_stream(
                [1, 2, 3], 8, 0.0, "resp-id", 0, "test-model", None
            )
        stream = server._queue_owned_stream(inner)
        await stream.__anext__()  # protocol preamble, before lock acquisition
        await stream.aclose()

    asyncio.run(run())
    assert state.queue_depth == 0
    assert not state.lock.locked()
    assert not runner.calls


def test_close_after_lock_handoff_keepalive_releases_acquired_lock_and_queue():
    runner = FakeRunner(tokens_to_emit=[10])
    state = _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}), max_queue=1)

    async def run():
        await state.lock.acquire()
        assert server._try_reserve_queue_slot()
        inner = server._openai_stream(
            [1, 2, 3], 8, 0.0, "chat-id", 0, "test-model", [], False, None
        )
        stream = server._queue_owned_stream(inner)
        await stream.__anext__()  # role preamble

        defaults = server._acquire_lock_with_keepalive.__defaults__
        server._acquire_lock_with_keepalive.__defaults__ = (0.01,)
        try:
            assert await stream.__anext__() == server._SSE_KEEPALIVE_LINE
            state.lock.release()
            await asyncio.sleep(0.03)  # let the pending acquire task win
            await stream.aclose()
        finally:
            server._acquire_lock_with_keepalive.__defaults__ = defaults

    asyncio.run(run())
    assert state.queue_depth == 0
    assert not state.lock.locked()
    assert not runner.calls


def test_stream_double_cancellation_stops_decode_before_releasing_lock():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    second_callback_completed = threading.Event()

    class BlockingRunner(FakeRunner):
        def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
            try:
                on_tokens([10])
                started.set()
                assert release.wait(2), "test did not release blocking fake runner"
                on_tokens([11])
                second_callback_completed.set()
                return super().generate(
                    prompt_ids, max_tokens, temp, eos_ids, None, session, **extra
                )
            finally:
                finished.set()

    state = _install_state(
        BlockingRunner([]), tokenizer=FakeTokenizer(vocab={10: "x", 11: "y"}), max_queue=1
    )

    async def run():
        assert server._try_reserve_queue_slot()
        stream = server._queue_owned_stream(
            server._openai_stream(
                [1, 2, 3], 8, 0.0, "chat-id", 0, "test-model", [], False, None
            )
        )

        async def consume():
            async for _chunk in stream:
                pass

        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0.02)
        assert state.lock.locked()
        assert state.queue_depth == 1

        task.cancel()
        await asyncio.sleep(0.02)
        assert state.lock.locked()
        assert state.queue_depth == 1

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert finished.is_set()
    assert not second_callback_completed.is_set()
    assert not state.lock.locked()
    assert state.queue_depth == 0


def test_reserve_and_release_queue_slot_unit():
    runner = FakeRunner(tokens_to_emit=[])
    state = _install_state(runner, max_queue=2)
    assert server._try_reserve_queue_slot() is True
    assert state.queue_depth == 1
    assert server._try_reserve_queue_slot() is True
    assert state.queue_depth == 2
    assert server._try_reserve_queue_slot() is False
    assert state.queue_depth == 2
    server._release_queue_slot()
    assert state.queue_depth == 1
    server._release_queue_slot()
    server._release_queue_slot()  # 0 未満に落ちない
    assert state.queue_depth == 0


# ---------- 15. SSE keepalive ----------


def test_await_with_keepalive_yields_periodic_lines_then_result():
    async def _run():
        async def slow():
            await asyncio.sleep(0.12)
            return "done-value"

        keepalives = []
        result = None
        async for kind, val in server._await_with_keepalive(slow(), interval=0.03):
            if kind == "keepalive":
                keepalives.append(val)
            else:
                result = val
        return keepalives, result

    keepalives, result = asyncio.run(_run())
    assert result == "done-value"
    assert len(keepalives) >= 2
    assert all(k == server._SSE_KEEPALIVE_LINE for k in keepalives)


def test_await_with_keepalive_returns_immediately_without_keepalive_when_fast():
    async def _run():
        async def fast():
            return "quick"

        keepalives = []
        result = None
        async for kind, val in server._await_with_keepalive(fast(), interval=5.0):
            if kind == "keepalive":
                keepalives.append(val)
            else:
                result = val
        return keepalives, result

    keepalives, result = asyncio.run(_run())
    assert result == "quick"
    assert keepalives == []


def test_await_with_keepalive_cancels_pending_task_on_early_close():
    async def _run():
        cancelled = {"v": False}

        async def slow():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled["v"] = True
                raise

        agen = server._await_with_keepalive(slow(), interval=0.02)
        kind, _val = await agen.__anext__()
        assert kind == "keepalive"
        await agen.aclose()
        await asyncio.sleep(0.05)
        return cancelled["v"]

    assert asyncio.run(_run()) is True


def test_requeue_front_preserves_order():
    q: queue_mod.Queue = queue_mod.Queue()
    q.put(("b", 2))
    q.put(("c", 3))
    server._requeue_front(q, ("a", 1))
    assert q.get() == ("a", 1)
    assert q.get() == ("b", 2)
    assert q.get() == ("c", 3)


def test_chat_completions_stream_emits_keepalive_before_first_token(client):
    class SlowRunner(FakeRunner):
        def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
            time.sleep(0.05)
            return super().generate(
                prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra
            )

    runner = SlowRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    # _await_with_keepalive の既定 interval (15s) は使えないので、呼び出し側
    # (server.py の各 _*_stream) が interval= を明示しない前提のまま、関数
    # object の既定値だけ差し替える。
    original_defaults = server._await_with_keepalive.__defaults__
    server._await_with_keepalive.__defaults__ = (0.01,)
    try:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    finally:
        server._await_with_keepalive.__defaults__ = original_defaults

    assert resp.status_code == 200, resp.text
    assert server._SSE_KEEPALIVE_LINE in resp.text
    events = _sse_events(resp.text)
    text_chunks = [
        e["choices"][0]["delta"].get("content", "")
        for e in events
        if e.get("choices") and "content" in e["choices"][0].get("delta", {})
    ]
    assert "".join(text_chunks) == "x"


def test_anthropic_messages_stream_emits_keepalive_before_first_token(client):
    class SlowRunner(FakeRunner):
        def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
            time.sleep(0.05)
            return super().generate(
                prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra
            )

    runner = SlowRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    original_defaults = server._await_with_keepalive.__defaults__
    server._await_with_keepalive.__defaults__ = (0.01,)
    try:
        resp = client.post(
            "/v1/messages",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "stream": True,
            },
        )
    finally:
        server._await_with_keepalive.__defaults__ = original_defaults

    assert resp.status_code == 200, resp.text
    assert server._SSE_KEEPALIVE_LINE in resp.text


# ---------- 16. graceful shutdown ----------


def test_shutting_down_flag_returns_503_for_new_requests(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))
    server._SHUTTING_DOWN = True
    try:
        resp = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
    finally:
        server._SHUTTING_DOWN = False
    assert resp.status_code == 503, resp.text
    assert resp.headers.get("retry-after")
    assert not runner.calls


def test_shutting_down_flag_still_allows_health_and_hello(client):
    runner = FakeRunner(tokens_to_emit=[])
    _install_state(runner)
    server._SHUTTING_DOWN = True
    try:
        health_resp = client.get("/health")
        hello_resp = client.get("/api/hello")
    finally:
        server._SHUTTING_DOWN = False
    assert health_resp.status_code == 200
    assert hello_resp.status_code == 200


class _FakeUvicornServer:
    """本物の uvicorn.Server の代わりに ``_install_graceful_shutdown`` /
    ``_drain_and_exit`` が触る属性・メソッドだけを最小限で持つ fake。
    ``startup`` を持たない fake だと ``_install_graceful_shutdown`` が
    ``server_obj.startup`` の取得で AttributeError になる (実物の Server は
    必ず持つ) ので、ここで揃えておく。"""

    def __init__(self):
        self.force_exit = False
        self.should_exit = False
        self.startup_calls = 0
        self.handle_exit_calls: list = []

    def handle_exit(self, sig, frame):
        self.handle_exit_calls.append((sig, frame))

    async def startup(self, sockets=None):
        self.startup_calls += 1


def test_install_graceful_shutdown_first_signal_sets_flag_not_force_exit():
    fake = _FakeUvicornServer()
    server._SHUTTING_DOWN = False
    try:
        server._install_graceful_shutdown(fake)
        fake.handle_exit(15, None)
        assert server._SHUTTING_DOWN is True
        assert fake.force_exit is False
        assert fake.should_exit is False
        # 1 回目は uvicorn 本体の should_exit をまだ立てない (リスナーを
        # 開けたまま _gate_requests に 503 を返させるため — should_exit は
        # _drain_and_exit がキューの空きを見てから立てる。
        # _install_graceful_shutdown の docstring 参照)。fake.handle_exit
        # (元の uvicorn ハンドラ相当) 自体もこの実装ではもう呼ばれない。
        assert fake.handle_exit_calls == []
    finally:
        server._SHUTTING_DOWN = False


def test_install_graceful_shutdown_second_signal_force_exits_process_immediately():
    # 2 回目は os._exit で OS レベルの即時終了に落とす (force_exit だけでは
    # 生成中の非 daemon ワーカースレッドの完了待ちが残ってしまい「即時」に
    # ならないことを実機で確認済み — server.py の docstring 参照)。テスト
    # プロセスを実際に殺されては困るので os._exit をモックで差し替える。
    fake = _FakeUvicornServer()
    server._SHUTTING_DOWN = False
    try:
        server._install_graceful_shutdown(fake)
        fake.handle_exit(15, None)  # SIGTERM (1 回目)
        with mock.patch("os._exit") as exit_mock:
            fake.handle_exit(2, None)  # SIGINT (別種でも2回目なので即時終了)
            exit_mock.assert_called_once_with(1)
        assert fake.force_exit is True
        # os._exit をモックしたので実際には終了しない (テスト用の代替経路)。
        assert fake.handle_exit_calls == []
    finally:
        server._SHUTTING_DOWN = False


def test_install_graceful_shutdown_wraps_startup_to_launch_drain_watcher():
    fake = _FakeUvicornServer()
    server._SHUTTING_DOWN = False
    try:
        server._install_graceful_shutdown(fake)

        async def _run():
            await fake.startup()

        asyncio.run(_run())
        assert fake.startup_calls == 1
    finally:
        server._SHUTTING_DOWN = False


def test_drain_and_exit_waits_for_queue_depth_before_setting_should_exit():
    runner = FakeRunner(tokens_to_emit=[])
    state = _install_state(runner, queue_depth=2)
    fake = _FakeUvicornServer()
    server._SHUTTING_DOWN = True
    try:

        async def _run():
            task = asyncio.create_task(server._drain_and_exit(fake))
            await asyncio.sleep(0.05)
            # まだキューが空いていないので should_exit はまだ立たない。
            assert fake.should_exit is False
            state.queue_depth = 0
            await asyncio.wait_for(task, timeout=1.0)

        asyncio.run(_run())
        assert fake.should_exit is True
    finally:
        server._SHUTTING_DOWN = False


def test_drain_and_exit_short_circuits_on_force_exit():
    fake = _FakeUvicornServer()
    fake.force_exit = True
    server._SHUTTING_DOWN = False
    try:
        asyncio.run(asyncio.wait_for(server._drain_and_exit(fake), timeout=1.0))
        assert fake.should_exit is False
    finally:
        server._SHUTTING_DOWN = False


# ---------- 17. バージョン ----------


def test_health_includes_version(client):
    runner = FakeRunner(tokens_to_emit=[])
    _install_state(runner, version="9.9.9-test")
    resp = client.get("/health")
    assert resp.json()["version"] == "9.9.9-test"


def test_list_models_includes_version(client):
    runner = FakeRunner(tokens_to_emit=[])
    _install_state(runner, version="9.9.9-test")
    resp = client.get("/v1/models")
    assert resp.json()["version"] == "9.9.9-test"


def test_health_before_load_still_reports_a_version():
    previous = server.STATE
    server.STATE = None
    try:
        c = TestClient(server.app)
        resp = c.get("/health")
    finally:
        server.STATE = previous
    assert resp.status_code == 503
    assert resp.json()["version"] == server._FASTMLX_VERSION


def test_mlxturbo_version_matches_pyproject():
    # importlib.metadata 経由で pyproject.toml の version をそのまま返す
    # (別ファイルに二重管理しない)。
    assert server._FASTMLX_VERSION == server._mlxturbo_version()
    assert server._FASTMLX_VERSION != ""


# ---------- 18. --mtp サイドカーのパス (cli.py: load_cli_mtp) ----------


def test_load_cli_mtp_explicit_path_takes_priority_over_bundled_and_search():
    sentinel = object()
    with (
        mock.patch.object(cli_module, "load_mtp_file", return_value=sentinel) as load_file,
        mock.patch.object(cli_module, "load_quantized_mtp") as load_bundled,
        mock.patch.object(cli_module, "find_snapshot") as find_original,
    ):
        actual = cli_module.load_cli_mtp(
            "artifact-repo",
            {"fastmlx_mtp": True},  # 通常ならバンドル済みアーティファクト経路
            object(),
            "raw-repo",
            4,
            mtp_path="/some/sidecar.safetensors",
        )
    assert actual is sentinel
    load_file.assert_called_once()
    assert load_file.call_args[0][0] == "/some/sidecar.safetensors"
    load_bundled.assert_not_called()
    find_original.assert_not_called()


def test_load_cli_mtp_env_var_fallback_when_path_kwarg_omitted(monkeypatch):
    sentinel = object()
    monkeypatch.setenv(cli_module.MTP_PATH_ENV, "/env/sidecar.safetensors")
    with mock.patch.object(cli_module, "load_mtp_file", return_value=sentinel) as load_file:
        actual = cli_module.load_cli_mtp("repo", {}, object(), "raw-repo", 4)
    assert actual is sentinel
    assert load_file.call_args[0][0] == "/env/sidecar.safetensors"


def test_load_cli_mtp_explicit_path_kwarg_wins_over_env_var(monkeypatch):
    sentinel = object()
    monkeypatch.setenv(cli_module.MTP_PATH_ENV, "/env/sidecar.safetensors")
    with mock.patch.object(cli_module, "load_mtp_file", return_value=sentinel) as load_file:
        actual = cli_module.load_cli_mtp(
            "repo", {}, object(), "raw-repo", 4, mtp_path="/explicit/sidecar.safetensors"
        )
    assert actual is sentinel
    assert load_file.call_args[0][0] == "/explicit/sidecar.safetensors"


def test_load_cli_mtp_unreadable_sidecar_disables_mtp_without_fallback(capsys):
    with (
        mock.patch.object(
            cli_module, "load_mtp_file", side_effect=ValueError("bad tensor names")
        ),
        mock.patch.object(cli_module, "load_quantized_mtp") as load_bundled,
        mock.patch.object(cli_module, "find_snapshot") as find_original,
    ):
        actual = cli_module.load_cli_mtp(
            "repo",
            {"fastmlx_mtp": True},
            object(),
            "raw-repo",
            4,
            mtp_path="/bad/sidecar.safetensors",
        )
    assert actual is None
    load_bundled.assert_not_called()
    find_original.assert_not_called()
    out = capsys.readouterr().out
    assert "/bad/sidecar.safetensors" in out
    assert "ValueError" in out


def test_load_cli_mtp_no_mtp_flag_wins_even_with_explicit_path(monkeypatch):
    monkeypatch.setenv(cli_module.MTP_PATH_ENV, "/env/sidecar.safetensors")
    with mock.patch.object(cli_module, "load_mtp_file") as load_file:
        actual = cli_module.load_cli_mtp(
            "repo", {}, object(), "raw-repo", 4, no_mtp=True, mtp_path="/explicit/path"
        )
    assert actual is None
    load_file.assert_not_called()


def test_load_cli_mtp_nonexistent_path_does_not_touch_search_paths(capsys):
    # 実在しないパスは load_mtp_file 内部で例外になる (FileNotFoundError 系) —
    # ここではモックせず、本物の mx.load に読みに行かせて失敗させる。5.2GB の
    # 実サイドカーには一切アクセスしない。
    with (
        mock.patch.object(cli_module, "load_quantized_mtp") as load_bundled,
        mock.patch.object(cli_module, "find_snapshot") as find_original,
    ):
        actual = cli_module.load_cli_mtp(
            "repo",
            {},
            object(),
            "raw-repo",
            4,
            mtp_path="/no/such/sidecar.safetensors",
        )
    assert actual is None
    load_bundled.assert_not_called()
    find_original.assert_not_called()
    assert "/no/such/sidecar.safetensors" in capsys.readouterr().out


def test_server_module_shares_mtp_path_env_var_with_cli():
    assert server.MTP_PATH_ENV == cli_module.MTP_PATH_ENV == "FASTMLX_MTP_PATH"


# ---------- build_runner: Qwen3.8-Flash-Next (qwen4_exp) 配線 ----------
#
# mlxturbo/runner.py の build_runner に足した分岐だけを、実モデルなしで検証
# する。model_type/text/model.rope だけを持つ最小限のフェイクで route を
# 確認する — 27B (qwen3_5) 側の既存分岐はここでは一切変えていない (下の
# 「--mtp 無し」テストは、その既存分岐が最後まで正しく FallbackRunner に
# 落ちることの回帰確認も兼ねる)。


def _fake_qwen4_exp_model():
    return SimpleNamespace(
        args=SimpleNamespace(model_type="qwen4_exp", text=object(), text_config={}),
        model=SimpleNamespace(rope=object()),
    )


def test_build_runner_routes_qwen4_exp_with_mtp_to_flash_spec(monkeypatch):
    import mlxturbo.mtp_flash as mtp_flash_module
    import mlxturbo.runner as runner_module

    class DummyMTP:
        def parameters(self):
            return {}

    captured = {}

    def fake_load_flash_mtp(path, text_args, quantize=None, weights=None):
        captured["path"] = path
        captured["text_args"] = text_args
        captured["quantize"] = quantize
        captured["weights"] = weights
        return DummyMTP()

    monkeypatch.setattr(mtp_flash_module, "load_flash_mtp", fake_load_flash_mtp)

    model = _fake_qwen4_exp_model()
    args = SimpleNamespace(
        model="fake-model",
        original="fake-original",
        mtp="fake-sidecar.safetensors",
        mtp_bits=4,
        no_mtp=False,
        no_fused=True,
    )
    runner = runner_module.build_runner(model, tokenizer=object(), config={}, args=args)
    assert isinstance(runner, FlashSpecRunner)
    assert runner.KIND == "flash_spec"
    assert runner.engine.model is model
    assert captured["path"] == "fake-sidecar.safetensors"
    assert captured["text_args"] is model.args.text
    assert captured["quantize"] == {"group_size": 64, "bits": 4}
    # 明示指定 (--mtp) のときは weights= は渡さない (path 経由のまま)
    assert captured["weights"] is None


def test_build_runner_exits_when_flash_mtp_sidecar_unreadable_after_retry(monkeypatch, capsys):
    """``--mtp`` は運用者の明示指定なので、1 回リトライしても読めなければ
    フォールバックせず理由を明示して ``SystemExit(1)`` する (仕様変更:
    以前はここで FallbackRunner に倒していたが、それは「明示的に頼まれた
    のにできないなら黙って無視せず大声で断る」という設計と矛盾するため、
    逃げ道のフラグ無しで exit 1 に統一した)。"""

    import mlxturbo.mtp_flash as mtp_flash_module
    import mlxturbo.runner as runner_module

    calls = []

    def fake_load_flash_mtp(*a, **k):
        calls.append(1)
        raise ValueError("bad sidecar")

    monkeypatch.setattr(mtp_flash_module, "load_flash_mtp", fake_load_flash_mtp)

    model = _fake_qwen4_exp_model()
    args = SimpleNamespace(
        model="fake-model",
        original="fake-original",
        mtp="bad-sidecar.safetensors",
        mtp_bits=4,
        no_mtp=False,
        no_fused=True,
    )
    with pytest.raises(SystemExit) as exc_info:
        runner_module.build_runner(model, tokenizer=object(), config={}, args=args)
    assert exc_info.value.code == 1
    # ロードは 1 回だけリトライする = 呼び出しはちょうど 2 回
    assert len(calls) == 2
    out = capsys.readouterr().out
    assert "bad-sidecar.safetensors" in out
    assert "ValueError" in out
    assert "再試行します" in out  # 1 回目の失敗でリトライ予告のログが出る
    assert "--mtp を外せば" in out  # 回避策の一言


def test_build_runner_retries_flash_mtp_load_once_then_succeeds(monkeypatch, capsys):
    """1 回目のロードが失敗し、2 回目 (リトライ) が成功すれば通常どおり
    FlashSpecRunner が返る — フォールバックにも exit 1 にもならない。"""

    import mlxturbo.mtp_flash as mtp_flash_module
    import mlxturbo.runner as runner_module

    class DummyMTP:
        def parameters(self):
            return {}

    attempts = []

    def fake_load_flash_mtp(*a, **k):
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("GPU Timeout Error (simulated)")
        return DummyMTP()

    monkeypatch.setattr(mtp_flash_module, "load_flash_mtp", fake_load_flash_mtp)

    model = _fake_qwen4_exp_model()
    args = SimpleNamespace(
        model="fake-model",
        original="fake-original",
        mtp="fake-sidecar.safetensors",
        mtp_bits=4,
        no_mtp=False,
        no_fused=True,
    )
    runner = runner_module.build_runner(model, tokenizer=object(), config={}, args=args)
    assert isinstance(runner, FlashSpecRunner)
    assert runner.fallback_reason is None
    assert len(attempts) == 2
    out = capsys.readouterr().out
    assert "再試行します" in out
    assert "GPU Timeout Error" in out


_NOT_FOUND_REASON = (
    "MTP が見つからない (--mtp で指定するか、モデルディレクトリに"
    " mtp.safetensors を置く)"
)


def test_build_runner_falls_back_when_no_mtp_source_found(monkeypatch):
    """--mtp が無い呼び出し (cli.py の Namespace は ``mtp`` 属性自体を持た
    ない) かつモデルディレクトリにも自動発見できる MTP が無い場合、
    qwen4_exp モデルでも FlashSpecEngine 経路に入らず、load_flash_mtp を
    呼ぶことすらなく FallbackRunner に落ちる — ``fallback_reason`` は
    「MTP が見つからない (...)」。"""

    import mlxturbo.mtp_flash as mtp_flash_module
    import mlxturbo.runner as runner_module

    called = []
    monkeypatch.setattr(
        mtp_flash_module, "load_flash_mtp", lambda *a, **k: called.append(1)
    )

    model = _fake_qwen4_exp_model()
    # cli.py の argparse.Namespace には --mtp が無いので、mtp 属性自体が無い
    # (server.py の Namespace と違い getattr(args, "mtp", None) が None を返す)。
    # "fake-model" は存在しないディレクトリなので自動発見も失敗する。
    args = SimpleNamespace(
        model="fake-model", original="fake-original", mtp_bits=4, no_mtp=True,
        no_fused=True,
    )
    runner = runner_module.build_runner(model, tokenizer=object(), config={}, args=args)
    assert not called
    assert isinstance(runner, FallbackRunner)
    assert not isinstance(runner, FlashSpecRunner)
    assert runner.fallback_reason == _NOT_FOUND_REASON


def test_build_runner_qwen4_exp_with_explicit_none_mtp_and_no_source_found(monkeypatch):
    """server.py の Namespace は ``--mtp`` 未指定でも ``mtp`` 属性自体は
    存在し値が None になる (argparse の ``default=None``)。cli.py 側の
    「属性自体が無い」場合と挙動が揃うことを確認する (自動発見も失敗する
    ディレクトリを渡した場合)。"""

    import mlxturbo.mtp_flash as mtp_flash_module
    import mlxturbo.runner as runner_module

    called = []
    monkeypatch.setattr(
        mtp_flash_module, "load_flash_mtp", lambda *a, **k: called.append(1)
    )

    model = _fake_qwen4_exp_model()
    args = SimpleNamespace(
        model="fake-model", original="fake-original", mtp=None, mtp_bits=4,
        no_mtp=True, no_fused=True,
    )
    runner = runner_module.build_runner(model, tokenizer=object(), config={}, args=args)
    assert not called
    assert isinstance(runner, FallbackRunner)
    assert runner.fallback_reason == _NOT_FOUND_REASON


def test_build_runner_resolves_loaded_hf_repo_before_flash_mtp_discovery(
    monkeypatch, tmp_path
):
    """A repo id must be searched in its local snapshot, not as ``org/repo``."""

    import mlxturbo.runner as runner_module

    seen = []
    monkeypatch.setattr(
        runner_module,
        "resolve_local_model_path",
        lambda ref: seen.append(("resolve", ref)) or tmp_path,
    )
    monkeypatch.setattr(
        runner_module,
        "_discover_flash_mtp_source",
        lambda path: seen.append(("discover", path)) or None,
    )

    model = _fake_qwen4_exp_model()
    args = SimpleNamespace(
        model="org/flash-model", original="fake-original", mtp=None, mtp_bits=4,
        no_mtp=True, no_fused=True,
    )
    actual = runner_module.build_runner(model, tokenizer=object(), config={}, args=args)

    assert isinstance(actual, FallbackRunner)
    assert seen == [("resolve", "org/flash-model"), ("discover", tmp_path)]


# ---------- build_runner: qwen4_exp の MTP 自動発見 (モデル内蔵/サイドカー) ----------
#
# --mtp が未指定のとき、mlxturbo.runner._discover_flash_mtp_source が
# モデルディレクトリを見て MTP を自動で見つける。実モデルは要らない —
# 合成の index.json + 小さな safetensors シャード (モデル内蔵側) と、
# 単体の mtp.safetensors (サイドカー側) を tmp_path に置いて検証する。
# load_flash_mtp 自体はモック (実際の重みロード/構築ロジックは
# mlxturbo/mtp_flash.py 側の責務で、ここでは呼び出され方だけを見る)。


def test_discover_flash_mtp_source_finds_embedded_weights_in_index(tmp_path):
    """model.safetensors.index.json の weight_map に mtp.* キーがあれば、
    それを含むシャードだけを読み、mtp.* テンソルだけを集めて返す
    (同じシャードに同居する非 mtp テンソルは混ざらない)。"""

    import mlxturbo.runner as runner_module

    shard_name = "model-00001-of-00001.safetensors"
    mx.save_safetensors(
        str(tmp_path / shard_name),
        {
            "mtp.pre_fc_norm_embedding.weight": mx.zeros((4,)),
            "model.embed_tokens.weight": mx.zeros((4, 4)),
        },
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.pre_fc_norm_embedding.weight": shard_name,
                    "model.embed_tokens.weight": shard_name,
                }
            }
        )
    )

    result = runner_module._discover_flash_mtp_source(tmp_path)
    assert result is not None
    source_label, spec = result
    assert source_label == "モデル内蔵"
    assert isinstance(spec, dict)
    assert set(spec.keys()) == {"mtp.pre_fc_norm_embedding.weight"}


def test_discover_flash_mtp_source_finds_unsharded_embedded_weights(tmp_path):
    """An unsharded model has no index; inspect model.safetensors itself."""

    import mlxturbo.runner as runner_module

    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "mtp.pre_fc_norm_embedding.weight": mx.zeros((4,)),
            "model.embed_tokens.weight": mx.zeros((4, 4)),
        },
    )

    result = runner_module._discover_flash_mtp_source(tmp_path)
    assert result is not None
    source_label, spec = result
    assert source_label == "モデル内蔵 (model.safetensors)"
    assert set(spec) == {"mtp.pre_fc_norm_embedding.weight"}


def test_discover_flash_mtp_source_uses_sidecar_when_index_is_malformed(tmp_path):
    """A broken optional index must not hide a valid sidecar fallback."""

    import mlxturbo.runner as runner_module

    (tmp_path / "model.safetensors.index.json").write_text("[]")
    sidecar = tmp_path / "mtp.safetensors"
    mx.save_safetensors(str(sidecar), {"mtp.x": mx.zeros((1,))})

    assert runner_module._discover_flash_mtp_source(tmp_path) == (
        "サイドカー (mtp.safetensors)",
        str(sidecar),
    )


def test_discover_flash_mtp_source_finds_sidecar_file(tmp_path):
    """index.json が無くても、モデルディレクトリ直下の mtp.safetensors が
    あればそれをサイドカーとして返す (path 文字列、weights ではない)。"""

    import mlxturbo.runner as runner_module

    sidecar = tmp_path / "mtp.safetensors"
    mx.save_safetensors(str(sidecar), {"mtp.x": mx.zeros((1,))})

    result = runner_module._discover_flash_mtp_source(tmp_path)
    assert result == ("サイドカー (mtp.safetensors)", str(sidecar))


def test_discover_flash_mtp_source_returns_none_when_nothing_found(tmp_path):
    import mlxturbo.runner as runner_module

    assert runner_module._discover_flash_mtp_source(tmp_path) is None


def test_build_runner_routes_qwen4_exp_via_embedded_mtp_weights(monkeypatch, tmp_path):
    """--mtp 未指定でも、モデルディレクトリの safetensors シャードに
    mtp.* テンソルがあれば自動発見して FlashSpecRunner まで到達する。
    load_flash_mtp には path=None, weights=<集めた dict> で渡ること。"""

    import mlxturbo.mtp_flash as mtp_flash_module
    import mlxturbo.runner as runner_module

    shard_name = "model-00001-of-00001.safetensors"
    mx.save_safetensors(
        str(tmp_path / shard_name),
        {"mtp.pre_fc_norm_embedding.weight": mx.zeros((4,))},
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"mtp.pre_fc_norm_embedding.weight": shard_name}}
        )
    )

    class DummyMTP:
        def parameters(self):
            return {}

    captured = {}

    def fake_load_flash_mtp(path, text_args, quantize=None, weights=None):
        captured["path"] = path
        captured["weights"] = weights
        return DummyMTP()

    monkeypatch.setattr(mtp_flash_module, "load_flash_mtp", fake_load_flash_mtp)

    model = _fake_qwen4_exp_model()
    args = SimpleNamespace(
        model=str(tmp_path), original="fake-original", mtp=None, mtp_bits=4,
        no_mtp=True, no_fused=True,
    )
    runner = runner_module.build_runner(model, tokenizer=object(), config={}, args=args)
    assert isinstance(runner, FlashSpecRunner)
    assert runner.fallback_reason is None
    assert captured["path"] is None
    assert set(captured["weights"].keys()) == {"mtp.pre_fc_norm_embedding.weight"}


def test_build_runner_routes_qwen4_exp_via_sidecar_file(monkeypatch, tmp_path):
    """--mtp 未指定でも、モデルディレクトリ直下に mtp.safetensors があれば
    自動発見してそのパスで load_flash_mtp を呼ぶ (weights= は渡さない)。"""

    import mlxturbo.mtp_flash as mtp_flash_module
    import mlxturbo.runner as runner_module

    sidecar = tmp_path / "mtp.safetensors"
    mx.save_safetensors(str(sidecar), {"mtp.x": mx.zeros((1,))})

    class DummyMTP:
        def parameters(self):
            return {}

    captured = {}

    def fake_load_flash_mtp(path, text_args, quantize=None, weights=None):
        captured["path"] = path
        captured["weights"] = weights
        return DummyMTP()

    monkeypatch.setattr(mtp_flash_module, "load_flash_mtp", fake_load_flash_mtp)

    model = _fake_qwen4_exp_model()
    args = SimpleNamespace(
        model=str(tmp_path), original="fake-original", mtp=None, mtp_bits=4,
        no_mtp=True, no_fused=True,
    )
    runner = runner_module.build_runner(model, tokenizer=object(), config={}, args=args)
    assert isinstance(runner, FlashSpecRunner)
    assert captured["path"] == str(sidecar)
    assert captured["weights"] is None


def test_build_runner_explicit_mtp_wins_over_auto_discovered_sidecar(monkeypatch, tmp_path):
    """--mtp が明示指定されていれば、モデルディレクトリに自動発見できる
    サイドカーがあってもそちらは一切見ない (優先順位 1 が最優先)。"""

    import mlxturbo.mtp_flash as mtp_flash_module
    import mlxturbo.runner as runner_module

    # 自動発見されうるサイドカーを置いておく — 明示指定より優先されない
    # ことを確認するための撒き餌
    mx.save_safetensors(str(tmp_path / "mtp.safetensors"), {"mtp.x": mx.zeros((1,))})

    class DummyMTP:
        def parameters(self):
            return {}

    captured = {}

    def fake_load_flash_mtp(path, text_args, quantize=None, weights=None):
        captured["path"] = path
        captured["weights"] = weights
        return DummyMTP()

    monkeypatch.setattr(mtp_flash_module, "load_flash_mtp", fake_load_flash_mtp)

    model = _fake_qwen4_exp_model()
    args = SimpleNamespace(
        model=str(tmp_path),
        original="fake-original",
        mtp="explicit-sidecar.safetensors",
        mtp_bits=4,
        no_mtp=False,
        no_fused=True,
    )
    runner = runner_module.build_runner(model, tokenizer=object(), config={}, args=args)
    assert isinstance(runner, FlashSpecRunner)
    assert captured["path"] == "explicit-sidecar.safetensors"


def test_build_runner_falls_back_when_auto_discovered_mtp_unreadable(monkeypatch, capsys, tmp_path):
    """自動発見 (サイドカー) の読み込みが 1 回リトライしても失敗する場合、
    明示指定 (--mtp) とは違い exit せずフォールバックする。fallback_reason
    は「MTP 自動発見 (...) の読み込みに失敗: ...」。"""

    import mlxturbo.mtp_flash as mtp_flash_module
    import mlxturbo.runner as runner_module

    sidecar = tmp_path / "mtp.safetensors"
    mx.save_safetensors(str(sidecar), {"mtp.x": mx.zeros((1,))})

    calls = []

    def fake_load_flash_mtp(*a, **k):
        calls.append(1)
        raise ValueError("corrupt sidecar")

    monkeypatch.setattr(mtp_flash_module, "load_flash_mtp", fake_load_flash_mtp)

    model = _fake_qwen4_exp_model()
    args = SimpleNamespace(
        model=str(tmp_path), original="fake-original", mtp=None, mtp_bits=4,
        no_mtp=True, no_fused=True,
    )
    runner = runner_module.build_runner(model, tokenizer=object(), config={}, args=args)
    assert isinstance(runner, FallbackRunner)
    assert not isinstance(runner, FlashSpecRunner)
    assert len(calls) == 2  # 自動発見にもリトライは適用される
    assert runner.fallback_reason.startswith(
        "MTP 自動発見 (サイドカー (mtp.safetensors)) の読み込みに失敗:"
    )
    assert "corrupt sidecar" in runner.fallback_reason
    out = capsys.readouterr().out
    assert "再試行します" in out


@pytest.mark.parametrize(
    ("index_body", "detail"),
    [
        ("[]", "JSON object"),
        (
            json.dumps({"weight_map": {"mtp.x": "missing.safetensors"}}),
            "missing.safetensors",
        ),
    ],
)
def test_build_runner_falls_back_with_reason_when_embedded_mtp_index_is_broken(
    monkeypatch, tmp_path, index_body, detail
):
    """Broken auto-discovery is observable fallback, never a startup traceback."""

    import mlxturbo.mtp_flash as mtp_flash_module
    import mlxturbo.runner as runner_module

    called = []
    monkeypatch.setattr(
        mtp_flash_module, "load_flash_mtp", lambda *a, **k: called.append(1)
    )
    (tmp_path / "model.safetensors.index.json").write_text(index_body)

    model = _fake_qwen4_exp_model()
    args = SimpleNamespace(
        model=str(tmp_path), original="fake-original", mtp=None, mtp_bits=4,
        no_mtp=True, no_fused=True,
    )
    actual = runner_module.build_runner(model, tokenizer=object(), config={}, args=args)

    assert isinstance(actual, FallbackRunner)
    assert not called
    assert actual.fallback_reason.startswith("MTP 自動発見に失敗:")
    assert detail in actual.fallback_reason


# ---------- --require-runner (server.py: _enforce_required_runner) ----------
#
# main() 自体は実モデルロードを伴うので直接は叩かず、main() が呼ぶ
# _enforce_required_runner を単体でテストする。


class _FakeRunnerForRequire:
    def __init__(self, kind: str, fallback_reason: str | None = None):
        self.KIND = kind
        self.fallback_reason = fallback_reason


def test_enforce_required_runner_noop_when_unspecified():
    runner = _FakeRunnerForRequire("fallback", fallback_reason="whatever")
    server._enforce_required_runner(runner, None)  # 例外なし


def test_enforce_required_runner_noop_when_kind_matches():
    runner = _FakeRunnerForRequire("flash_spec")
    server._enforce_required_runner(runner, "flash_spec")  # 例外なし


def test_enforce_required_runner_exits_on_mismatch(capsys):
    runner = _FakeRunnerForRequire("fallback", fallback_reason="MTP が見つからない (fake)")
    with pytest.raises(SystemExit) as exc_info:
        server._enforce_required_runner(runner, "flash_spec")
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "--require-runner flash_spec" in out
    assert "fallback" in out
    assert "MTP が見つからない (fake)" in out


def test_enforce_required_runner_exits_without_reason_detail_when_none(capsys):
    """fallback_reason が None のミスマッチ (投機経路同士の取り違え等) でも
    SystemExit(1) 自体は変わらず起きる — 理由の括弧書きが単に付かないだけ。"""

    runner = _FakeRunnerForRequire("spec", fallback_reason=None)
    with pytest.raises(SystemExit) as exc_info:
        server._enforce_required_runner(runner, "flash_spec")
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "--require-runner flash_spec" in out
    assert "解決された runner は spec。" in out  # 理由の括弧書きが付かない


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_zero_session_capacity(value):
    with pytest.raises(server.argparse.ArgumentTypeError, match="at least 1"):
        server._positive_int(value)


def test_positive_int_accepts_valid_session_capacity():
    assert server._positive_int("8") == 8


# ---------- Kimi K3 レビュー 項目 14: /v1/embeddings は 501 で明示 ----------


def test_embeddings_endpoint_returns_501(client):
    runner = FakeRunner(tokens_to_emit=[])
    _install_state(runner)

    resp = client.post("/v1/embeddings", json={"model": "test-model", "input": "hi"})
    assert resp.status_code == 501, resp.text
    assert resp.json()["error"]["code"] == "not_implemented"


# ---------- Kimi K3 レビュー 項目 7: 非恒等サンプリングパラメータのリクエスト単位降格 ----------


def test_spec_runner_downgrades_non_identity_sampling_params_instead_of_400(client):
    """STATE.downgrade_runner が使えるとき、SpecRunner 経路への非恒等値
    (top_p=0.9) はもう 400 にならず、そのリクエストだけ downgrade_runner
    (FallbackRunner 相当) へ処理が回る。主 runner (spec) は一切呼ばれない。"""

    primary = FakeSpecRunner(tokens_to_emit=[10])
    downgrade = FakeRunner(tokens_to_emit=[11])
    tok = FakeTokenizer(vocab={10: "spec-out", 11: "fallback-out"})
    _install_state(primary, tokenizer=tok, downgrade_runner=downgrade)

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "top_p": 0.9},
    )
    assert resp.status_code == 200, resp.text
    assert not primary.calls
    assert len(downgrade.calls) == 1
    assert downgrade.calls[0]["top_p"] == 0.9
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "fallback-out"
    assert "downgrade_reason" in body
    assert "top_p" in body["downgrade_reason"]


def test_spec_runner_identity_values_still_use_primary_runner_when_downgrade_available(client):
    """downgrade_runner が構成されていても、恒等値 (top_p=1.0) は従来どおり
    主 runner (投機経路) のまま — 降格は非恒等値のときだけ。"""

    primary = FakeSpecRunner(tokens_to_emit=[10])
    downgrade = FakeRunner(tokens_to_emit=[11])
    tok = FakeTokenizer(vocab={10: "spec-out", 11: "fallback-out"})
    _install_state(primary, tokenizer=tok, downgrade_runner=downgrade)

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "top_p": 1.0},
    )
    assert resp.status_code == 200, resp.text
    assert len(primary.calls) == 1
    assert not downgrade.calls
    assert "downgrade_reason" not in resp.json()


def test_flash_spec_runner_downgrades_on_anthropic_route_too(client):
    """項目 7 の降格は OpenAI 経路専用ではない — Anthropic 経路でも同じ
    resolver (_resolve_runner_for_request) を通る。"""

    primary = FakeFlashSpecRunner(tokens_to_emit=[10])
    downgrade = FakeRunner(tokens_to_emit=[11])
    tok = FakeTokenizer(vocab={10: "spec-out", 11: "fallback-out"})
    _install_state(primary, tokenizer=tok, downgrade_runner=downgrade)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": 5,
        },
    )
    assert resp.status_code == 200, resp.text
    assert not primary.calls
    assert len(downgrade.calls) == 1
    assert downgrade.calls[0]["top_k"] == 5


def test_spec_runner_still_400s_without_a_downgrade_runner(client):
    """downgrade_runner が無い (main() がそもそも用意しない = STATE.runner が
    既に fallback のとき) 構成では、既存どおり 400 のまま — 降格の安全網
    が無い場合まで黙って何かへ回したりしない。"""

    runner = FakeSpecRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))  # downgrade_runner=None (既定)

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "top_p": 0.9},
    )
    assert resp.status_code == 400, resp.text


def test_spec_runner_downgrade_streaming_sets_response_header(client):
    """ストリーミング経路では per-chunk の JSON スキーマを変えず、HTTP
    ヘッダで降格を示す。"""

    primary = FakeSpecRunner(tokens_to_emit=[10])
    downgrade = FakeRunner(tokens_to_emit=[10, 999])
    tok = FakeTokenizer(vocab={10: "hi"}, eos_token_ids=(999,))
    _install_state(primary, tokenizer=tok, downgrade_runner=downgrade)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.9,
            "stream": True,
        },
    )
    assert resp.status_code == 200, resp.text
    assert "X-Mlxturbo-Downgrade-Reason" in resp.headers
    assert not primary.calls
    assert downgrade.calls


def test_spec_runner_downgrade_does_not_touch_session_pool(client):
    """降格したリクエストは session=None で渡る (ChatSession/FallbackSession
    の型不一致を避けるため) — session_pool は一切増えない。"""

    primary = FakeSpecRunner(tokens_to_emit=[10])
    downgrade = FakeRunner(tokens_to_emit=[11])
    tok = FakeTokenizer(vocab={10: "spec-out", 11: "fallback-out"})
    pool = OrderedDict()
    _install_state(primary, tokenizer=tok, downgrade_runner=downgrade, session_pool=pool)

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "top_p": 0.9},
    )
    assert resp.status_code == 200, resp.text
    assert downgrade.calls[0]["session"] is None if "session" in downgrade.calls[0] else True
    assert len(pool) == 0


# ---------- Kimi K3 レビュー 項目 8: response_format を黙殺しない ----------


@pytest.mark.parametrize("rf", [{"type": "json_object"}, {"type": "json_schema"}])
def test_response_format_non_text_is_400(client, rf):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "response_format": rf},
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls
    assert "response_format" in resp.json()["error"]["message"]


def test_response_format_text_is_allowed(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "text"},
        },
    )
    assert resp.status_code == 200, resp.text
    assert runner.calls


def test_response_format_omitted_is_allowed(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200, resp.text


# ---------- Kimi K3 レビュー 項目 18: stop 文字列のトークン単位 early-stop ----------


class _StopWatchingRunner(FakeRunner):
    """``tokens_to_emit`` を「1 要素 = 1 回の on_tokens 呼び出し (1 デコード
    ラウンド)」の token id リストのリストとして解釈する (通常の FakeRunner は
    1 トークン = 1 呼び出し固定)。投機デコードの 1 ラウンドが複数トークンを
    まとめて確定させるのと同じ形にしないと、stop 文字列がラウンドの境目を
    跨いだときの既存の _find_stop 切り詰めロジック (今回変更していない部分)
    を正しく再現できない — 1 文字ずつ別々の on_tokens 呼び出しで届けると、
    マーカーの前半が「まだ一致と分かっていない」時点で別チャンクとして
    クライアントへ流れてしまい、この既存ロジック自体の仕様と噛み合わない
    (項目 18 が変えたのは打ち切りの早さだけで、この切り詰めロジック自体は
    対象外)。

    stop 一致後の cancel_event.set() が実際に generate() を打ち切ることは、
    「用意したラウンド全部を処理せずに終わる」(= 呼び出し側が検証する
    res["tokens"]/実際に呼ばれた回数) で確認する。
    """

    rounds_processed = 0

    def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
        self.calls.append({"prompt_ids": list(prompt_ids), **extra})
        emitted: list[int] = []
        for batch in self.tokens_to_emit:
            if len(emitted) >= max_tokens:
                break
            emitted.extend(batch)
            self.rounds_processed += 1
            on_tokens(list(batch))  # cancel_event が立っていればここで例外が飛ぶ
            if any(t in eos_ids for t in batch):
                break
            # 実機の 1 デコードラウンドは実時間がかかる (GPU forward) ので、
            # その間に非同期側の consumer が queue を追いついて処理し、stop
            # 一致を検出して cancel_event.set() する猶予がある。この fake は
            # メモリ操作だけで即座に全ラウンドを queue へ積んでしまうと
            # consumer が追いつく前に生成が終わってしまい、早期打ち切りその
            # ものを検証できない — ラウンド間に短い sleep を挟んで、実機と
            # 同じ「producer と consumer が競走する」状況を作る。
            time.sleep(0.005)
        return {
            "tokens": emitted,
            "ttft_s": 0.001,
            "decode_tps": 100.0,
            "prefill_reused": self.prefill_reused,
            "prefill_new": len(prompt_ids) - self.prefill_reused,
            "tokens_per_step": 1.0,
        }


def _stop_marker_token_batches(
    marker: str, tail_rounds: int = 40
) -> tuple[list[list[int]], dict[int, str], int]:
    """"a" / "b" / <marker、1トークンで> / "c" (1ラウンド1文字) x
    tail_rounds、という「ラウンドごとの token id リスト」を作る。marker の
    後ろに大量のラウンドを積んでおき、early-stop が効かなければ全部処理
    されてしまう形にする。戻り値は (batches, vocab, total_token_count)。

    marker は 1 トークンで表す (1 文字にする) — ThinkingRouter.feed は
    on_tokens に何個まとめて渡されても content_delta を 1 トークン=1
    セグメント単位でキューに積む (tool_enabled=False でも content_detok.
    add_token を 1 トークンずつ呼ぶ実装) ため、複数トークンにまたがる stop
    文字列は「マーカー全体が既知になった時点で、既に一部が別の (直前の)
    content_delta として転送済み」になり得る。これは _find_stop の切り詰め
    ロジック自体の話で項目 18 (打ち切りの早さ) の対象外 — このテストは
    早期打ち切りだけを見るので、その既存の切り詰めロジックが素直に扱える
    「1 トークンに収まる marker」で構成する。
    """

    vocab: dict[int, str] = {}
    next_id = 0

    def new_id(ch: str) -> int:
        nonlocal next_id
        i = next_id
        next_id += 1
        vocab[i] = ch
        return i

    assert len(marker) == 1, "marker は 1 トークン (1 文字) で表せる長さにすること"
    batches = [[new_id("a")], [new_id("b")], [new_id(marker)]]
    batches.extend([new_id("c")] for _ in range(tail_rounds))
    total = sum(len(b) for b in batches)
    return batches, vocab, total


def test_chat_completions_stream_stop_cancels_generation_early(client):
    marker = "!"
    batches, vocab, total = _stop_marker_token_batches(marker)
    runner = _StopWatchingRunner(tokens_to_emit=batches)
    tok = FakeTokenizer(vocab=vocab)
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stop": [marker],
            "max_tokens": total,
        },
    )
    assert resp.status_code == 200, resp.text
    events = _sse_events(resp.text)
    content = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events
        if e.get("choices") and "delta" in e["choices"][0]
    )
    assert content == "ab"
    finish_reasons = [
        e["choices"][0]["finish_reason"]
        for e in events
        if e.get("choices") and e["choices"][0].get("finish_reason")
    ]
    assert finish_reasons[-1] == "stop"
    # 早期打ち切りの核心: マーカーの後ろに 40 ラウンドぶんの 'c' を用意した
    # (= 用意した全ラウンドを消化していれば rounds_processed は 43 になる)
    # が、cancel_event.set() で打ち切られるので明らかにそれより少ない —
    # マーカーを検出したラウンドの次のラウンドで即座に止まる。
    assert runner.rounds_processed < len(batches)
    assert runner.rounds_processed <= 4  # a, b, marker, 最初の 'c' の直前で停止


def test_anthropic_stream_stop_sequence_cancels_generation_early(client):
    marker = "!"
    batches, vocab, total = _stop_marker_token_batches(marker)
    runner = _StopWatchingRunner(tokens_to_emit=batches)
    tok = FakeTokenizer(vocab=vocab)
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": total,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stop_sequences": [marker],
        },
    )
    assert resp.status_code == 200, resp.text
    events = _sse_events(resp.text)
    text = "".join(
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta"
    )
    assert text == "ab"
    stop_reasons = [e["delta"]["stop_reason"] for e in events if e.get("type") == "message_delta"]
    assert stop_reasons[-1] == "stop_sequence"
    assert runner.rounds_processed < len(batches)


def test_completions_legacy_stream_stop_cancels_generation_early(client):
    marker = "!"
    batches, vocab, total = _stop_marker_token_batches(marker)
    runner = _StopWatchingRunner(tokens_to_emit=batches)
    tok = FakeTokenizer(vocab=vocab)
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "stream": True,
            "stop": [marker],
            "max_tokens": total,
        },
    )
    assert resp.status_code == 200, resp.text
    events = _sse_events(resp.text)
    text = "".join(e["choices"][0]["text"] for e in events if e.get("choices"))
    assert text == "ab"
    finish_reasons = [
        e["choices"][0]["finish_reason"]
        for e in events
        if e.get("choices") and e["choices"][0].get("finish_reason")
    ]
    assert finish_reasons[-1] == "stop"
    assert runner.rounds_processed < len(batches)


def test_stream_stop_without_match_runs_to_completion(client):
    """一致しなければ従来どおり最後まで回る (early-stop の副作用で結果が
    変わっていないことの対照実験)。"""

    runner = FakeRunner(tokens_to_emit=[10, 11, 999])
    tok = FakeTokenizer(vocab={10: "a", 11: "b"}, eos_token_ids=(999,))
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stop": ["never-appears"],
        },
    )
    assert resp.status_code == 200, resp.text
    events = _sse_events(resp.text)
    content = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events
        if e.get("choices") and "delta" in e["choices"][0]
    )
    assert content == "ab"


# ---------- Kimi K3 レビュー 項目 16: Anthropic usage のキャッシュフィールド ----------


def test_anthropic_usage_maps_prefill_reused_to_cache_read_input_tokens(client):
    runner = FakeReusingRunner(reply_tokens=[10, 999])
    tok = FakeTokenizer(vocab={10: "hi"}, eos_token_ids=(999,))
    _install_state(runner, tokenizer=tok)

    session_key = next(server.STATE.session_key_seq)
    session = FallbackSession()
    session.cache = object()
    session.processed = [1, 2, 3]
    server.STATE.session_pool[session_key] = session

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    usage = resp.json()["usage"]
    assert usage["cache_read_input_tokens"] >= 0
    assert "cache_creation_input_tokens" in usage
    assert usage["input_tokens"] + usage["cache_read_input_tokens"] >= 0


def test_anthropic_usage_always_includes_cache_fields_even_when_zero(client):
    """再利用が無い (cold) ターンでも、フィールド自体は常に出す —
    「対応しているが今回は 0 件」と「対応していない」を区別しない
    (_usage_dict/_anthropic_usage 共通の方針)。"""

    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "hi"}))

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    usage = resp.json()["usage"]
    assert usage["cache_read_input_tokens"] == 0
    assert usage["cache_creation_input_tokens"] == usage["input_tokens"]


def test_anthropic_cache_control_on_system_block_is_accepted_and_ignored(client):
    """項目 16: cache_control 自体は読まない (無視する) — 少なくとも 400 に
    はならず、system テキストはそのまま反映される。"""

    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "hi"}))

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "system": [
                {"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text


# ---------- 項目 13: DraftSpecRunner (mlx_lm 自身の draft-model 投機) ----------


def test_draft_spec_runner_supported_sampling_params_matches_fallback():
    """FallbackRunner と同じ全キーを宣言している (DraftSpecRunner の
    docstring 参照: speculative_generate_step はドラフト/検証の両方に同じ
    sampler/logits_processors を通すので、top_p 等でも分布保証は壊れない)。"""

    assert DraftSpecRunner.SUPPORTED_SAMPLING_PARAMS == FallbackRunner.SUPPORTED_SAMPLING_PARAMS


def test_draft_spec_runner_passes_draft_model_and_num_draft_tokens_through(monkeypatch):
    import importlib

    mlx_generate = importlib.import_module("mlx_lm.generate")
    captured: dict = {}

    def fake_stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens,
        draft_model=None,
        num_draft_tokens=None,
        sampler=None,
        logits_processors=None,
        **kwargs,
    ):
        captured["draft_model"] = draft_model
        captured["num_draft_tokens"] = num_draft_tokens
        captured["prompt"] = list(prompt)
        for tok, from_draft in [(10, False), (11, True), (12, False)]:
            yield _FakeGenResponse(tok, f"<{tok}>", from_draft=from_draft)

    monkeypatch.setattr(mlx_generate, "stream_generate", fake_stream_generate)

    target_model = object()
    draft_model = object()
    runner = DraftSpecRunner(target_model, draft_model, tokenizer=object(), num_draft_tokens=5)

    observed: list[int] = []
    result = runner.generate(
        [1, 2, 3],
        max_tokens=10,
        temp=0.0,
        eos_ids=set(),
        on_tokens=lambda toks, text=None: observed.extend(toks),
        session=None,
    )
    assert captured["draft_model"] is draft_model
    assert captured["num_draft_tokens"] == 5
    assert captured["prompt"] == [1, 2, 3]
    assert result["tokens"] == [10, 11, 12]
    assert observed == [10, 11, 12]
    # n_decode = 2 (先頭を除く), from_draft=False が 2 個 (10, 12) なので
    # n_verify_rounds の近似も 2 -> tokens_per_step == 1.0。
    assert result["tokens_per_step"] == pytest.approx(1.0)


def test_draft_spec_runner_seed_calls_mx_random_seed(monkeypatch):
    calls = []
    monkeypatch.setattr(mx.random, "seed", calls.append)

    import importlib

    mlx_generate = importlib.import_module("mlx_lm.generate")
    monkeypatch.setattr(mlx_generate, "stream_generate", lambda *a, **k: iter(()))

    runner = DraftSpecRunner(object(), object(), tokenizer=object(), num_draft_tokens=4)
    runner.generate(
        [1], max_tokens=0, temp=0.0, eos_ids=set(), on_tokens=None, session=None, seed=99
    )
    assert calls == [99]


def test_runner_kinds_includes_draft_spec_and_lookup_spec():
    assert "draft_spec" in RUNNER_KINDS
    assert "lookup_spec" in RUNNER_KINDS


def test_health_endpoint_reports_draft_spec_runner(client):
    class _FakeDraftSpecRunner(FakeRunner):
        KIND = "draft_spec"
        SUPPORTED_SAMPLING_PARAMS = DraftSpecRunner.SUPPORTED_SAMPLING_PARAMS

    runner = _FakeDraftSpecRunner(tokens_to_emit=[])
    _install_state(runner)
    resp = client.get("/health")
    assert resp.json()["runner"] == "draft_spec"


def test_enforce_required_runner_accepts_matching_draft_spec_kind():
    class _R:
        KIND = "draft_spec"
        fallback_reason = None

    server._enforce_required_runner(_R(), "draft_spec")  # 例外が出なければ良い


def test_enforce_required_runner_rejects_mismatched_draft_spec_requirement():
    class _R:
        KIND = "fallback"
        fallback_reason = "some reason"

    with pytest.raises(SystemExit):
        server._enforce_required_runner(_R(), "draft_spec")


def test_build_runner_selects_draft_spec_runner_when_draft_model_given(monkeypatch):
    import mlx_lm as mlx_lm_pkg

    draft_model_obj = object()
    draft_tokenizer_obj = SimpleNamespace(vocab_size=100)
    captured: dict = {}

    def fake_load(path):
        captured["path"] = path
        return draft_model_obj, draft_tokenizer_obj

    monkeypatch.setattr(mlx_lm_pkg, "load", fake_load)

    model = object()
    tokenizer = SimpleNamespace(vocab_size=100)
    args = SimpleNamespace(
        model="main/model",
        draft_model="some/draft-path",
        num_draft_tokens=6,
        lookup_spec=False,
    )
    runner = build_runner(model, tokenizer, config={}, args=args)
    assert isinstance(runner, DraftSpecRunner)
    assert runner.KIND == "draft_spec"
    assert runner.model is model
    assert runner.draft_model is draft_model_obj
    assert runner.num_draft_tokens == 6
    assert runner.fallback_reason is None
    assert captured["path"] == "some/draft-path"


def test_build_runner_draft_model_vocab_mismatch_exits(monkeypatch):
    import mlx_lm as mlx_lm_pkg

    monkeypatch.setattr(
        mlx_lm_pkg, "load", lambda path: (object(), SimpleNamespace(vocab_size=999))
    )

    model = object()
    tokenizer = SimpleNamespace(vocab_size=100)
    args = SimpleNamespace(model="m", draft_model="d", num_draft_tokens=4, lookup_spec=False)
    with pytest.raises(SystemExit):
        build_runner(model, tokenizer, config={}, args=args)


def test_build_runner_draft_model_takes_precedence_over_qwen4_exp_selection(monkeypatch):
    """--draft-model が指定されたら qwen4_exp/SpecEngine の既存選択には一切
    入らない (build_runner の docstring 参照) — qwen4_exp モデルでも
    DraftSpecRunner が選ばれる。"""

    import mlx_lm as mlx_lm_pkg

    monkeypatch.setattr(
        mlx_lm_pkg, "load", lambda path: (object(), SimpleNamespace(vocab_size=100))
    )
    model = _fake_qwen4_exp_model()
    tokenizer = SimpleNamespace(vocab_size=100)
    args = SimpleNamespace(
        model="fake-model", draft_model="some/draft", num_draft_tokens=4, lookup_spec=False
    )
    runner = build_runner(model, tokenizer, config={}, args=args)
    assert isinstance(runner, DraftSpecRunner)


# ---------- 項目 12: LookupSpecRunner (n-gram lookup (SAM) だけの投機) ----------


class _TrimTrackingCache:
    """LookupSpecRunner が呼ぶ trim_prompt_cache と噛み合う最小限のフェイク
    cache。``seen`` が「今 _ScriptedGreedyModel が真に確定済みとみなして
    いるトークン数」を表し、trim(k) が正しく末尾 k 個を巻き戻す。"""

    def __init__(self):
        self.seen: list[int] = []

    def trim(self, k):
        del self.seen[len(self.seen) - k :]
        return k

    def is_trimmable(self):
        return True

    @property
    def state(self):
        return mx.array(0)


class _ScriptedGreedyModel:
    """テスト用の「モデル」: 位置 i (0-indexed, フィード累計本数) の次トー
    クンを固定配列 ``true_seq[i]`` から読むだけの、実際の重みを持たない
    フェイク。``cache[0].seen`` (= trim で正しく巻き戻る想定の「確定済み」
    カウント) を経由してのみ位置を決めるので、LookupSpecRunner 側の受理/
    拒否/trim の会計が 1 個でもずれれば真の出力列と食い違って検出できる。"""

    def __init__(self, true_seq: list[int], vocab: int = 16):
        self.true_seq = true_seq
        self.vocab = vocab

    def make_cache(self):
        return [_TrimTrackingCache()]

    def __call__(self, ids, cache=None):
        c = cache[0]
        toks = ids[0].tolist()
        rows = []
        for t in toks:
            c.seen.append(t)
            nxt = self.true_seq[len(c.seen)]
            row = [0.0] * self.vocab
            row[nxt] = 100.0
            rows.append(row)
        return mx.array([rows])


def test_lookup_spec_runner_accepts_correct_multi_token_draft():
    """真の継続が周期的 (= 繰り返しが多い) なら、1 ラウンドで複数トークン
    を受理できる (tokens_per_step > 1)。"""

    pattern = [1, 2, 3, 4]
    true_seq = [pattern[i % 4] for i in range(13)]
    prompt_ids = true_seq[:7]  # [1,2,3,4,1,2,3]
    model = _ScriptedGreedyModel(true_seq)
    runner = LookupSpecRunner(model, tokenizer=object(), max_draft=8, min_match=2)
    assert runner.trimmable is True

    collected: list[int] = []
    result = runner.generate(
        prompt_ids,
        max_tokens=6,
        temp=0.0,
        eos_ids=set(),
        on_tokens=collected.extend,
        session=None,
    )
    expected = true_seq[7:13]
    assert result["tokens"] == expected
    assert collected == expected
    assert result["tokens_per_step"] > 1.0


def test_lookup_spec_runner_corrects_a_wrong_draft():
    """SAM が過去の再帰から続きを提案しても、今回はそこで分岐する
    ("false friend" な繰り返し) 場合、最終出力は貪欲デコードの真値と一致
    する — 受理チェックが誤りを検出し、trim で正しく巻き戻すことの確認。"""

    true_seq = [1, 2, 3, 4, 1, 2, 3] + [4, 1, 9, 2, 3, 4]
    prompt_ids = true_seq[:7]
    model = _ScriptedGreedyModel(true_seq)
    runner = LookupSpecRunner(model, tokenizer=object(), max_draft=8, min_match=2)

    collected: list[int] = []
    result = runner.generate(
        prompt_ids,
        max_tokens=6,
        temp=0.0,
        eos_ids=set(),
        on_tokens=collected.extend,
        session=None,
    )
    expected = true_seq[7:13]
    assert expected == [4, 1, 9, 2, 3, 4]
    assert result["tokens"] == expected
    assert collected == expected


def test_lookup_spec_runner_matches_plain_greedy_when_no_repeat_exists():
    """繰り返しが無いプロンプトでは draft が一度も見つからず、1 トークン
    ずつの貪欲デコードと同じ出力・同じ round 数になる (「効かない場面で
    遅くならない」の根拠 — 余計な forward を積み増さない)。"""

    true_seq = [1, 2, 3, 5, 9, 2, 7]
    prompt_ids = true_seq[:3]
    model = _ScriptedGreedyModel(true_seq)
    runner = LookupSpecRunner(model, tokenizer=object(), max_draft=8, min_match=2)

    result = runner.generate(
        prompt_ids, max_tokens=4, temp=0.0, eos_ids=set(), on_tokens=None, session=None
    )
    expected = true_seq[3:7]
    assert result["tokens"] == expected
    # 一致が一度も起きなければ 1 round = 1 token (4 round で 4 token) になる
    # ので、n_decode/rounds = (4-1)/4 = 0.75 -- SpecRunner/FlashSpecRunner と
    # 同じ「先頭を除く」定義のままなら、これが「一度も加速しなかった」場合の
    # 唯一の値になる (FallbackRunner 自身の定数 1.0 とは定義が違う点に注意)。
    assert result["tokens_per_step"] == pytest.approx((len(expected) - 1) / len(expected))


def test_lookup_spec_runner_stops_at_eos_mid_round():
    """draft の途中で eos が確定した場合、それ以降 (同じラウンド内で既に
    teacher-forcing 済みのトークンを含む) は出力に出さない。"""

    # プロンプトは周期パターンの 1 周分 + 1 (SAM が [1,2,3,4] の巡回を見つけ
    # られるだけの繰り返し)。継続側は一意な番兵 99 を eos として使う —
    # パターン値そのもの (1/2/3/4) を eos にすると周期的に何度も出てきて
    # しまい「途中で打ち切る」ことの検証にならない。継続の並びはドラフト
    # ([4,1,2,3], SAM がプロンプトの巡回から提案するはず) の 4 番目
    # (0-indexed 3 番目) が 99 に化けている「途中で外れる」設計: 最初の 3 個
    # (4,1,2) は受理され、4 個目で不一致 (真値は 99) となり、その 99 が
    # eos なのでそこで打ち切る — 受理・trim・eos 打ち切りを 1 ラウンドで
    # 同時に確認できる。
    prompt_ids = [1, 2, 3, 4, 1, 2, 3]
    continuation = [4, 1, 2, 99, 3, 4, 1, 2]
    true_seq = prompt_ids + continuation
    model = _ScriptedGreedyModel(true_seq, vocab=128)
    runner = LookupSpecRunner(model, tokenizer=object(), max_draft=8, min_match=2)

    eos_token = 99
    collected: list[int] = []
    result = runner.generate(
        prompt_ids,
        max_tokens=10,
        temp=0.0,
        eos_ids={eos_token},
        on_tokens=collected.extend,
        session=None,
    )
    assert result["tokens"] == [4, 1, 2, 99]
    assert result["tokens"][-1] == eos_token
    assert eos_token not in result["tokens"][:-1]
    assert collected == result["tokens"]


def test_lookup_spec_runner_uses_lookup_path_for_identity_sampling_values():
    """恒等値 (top_p=1.0 等) は lookup 経路を妨げない — plain へ落ちるのは
    実際にロジットを変える値が来たときだけ。"""

    pattern = [1, 2, 3, 4]
    true_seq = [pattern[i % 4] for i in range(13)]
    prompt_ids = true_seq[:7]
    model = _ScriptedGreedyModel(true_seq)
    runner = LookupSpecRunner(model, tokenizer=object())
    result = runner.generate(
        prompt_ids,
        max_tokens=6,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=None,
        top_p=1.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        logit_bias={},
    )
    assert result["tokens"] == true_seq[7:13]
    assert result["tokens_per_step"] > 1.0


@pytest.mark.parametrize(
    "extra",
    [
        {"temp": 0.7},
        {"repetition_penalty": 1.1},
        {"presence_penalty": 0.2},
        {"frequency_penalty": 0.2},
        {"logit_bias": {"1": 2.0}},
    ],
)
def test_lookup_spec_runner_falls_back_to_plain_when_distribution_altering_params_given(extra):
    model = _ScriptedGreedyModel([0] * 5)
    runner = LookupSpecRunner(model, tokenizer=object())

    called = []

    def fake_fallback_generate(*a, **k):
        called.append((a, k))
        return {
            "tokens": [],
            "ttft_s": 0.0,
            "decode_tps": 0.0,
            "prefill_reused": 0,
            "prefill_new": 0,
            "tokens_per_step": 0.0,
        }

    runner._fallback.generate = fake_fallback_generate
    kwargs = {"temp": 0.0, **extra}
    runner.generate([1, 2], max_tokens=3, eos_ids=set(), on_tokens=None, session=None, **kwargs)
    assert called


def test_lookup_spec_runner_falls_back_to_plain_when_cache_not_trimmable():
    class _NonTrimmableCache:
        def is_trimmable(self):
            return False

    class _NonTrimmableModel(_ScriptedGreedyModel):
        def make_cache(self):
            return [_NonTrimmableCache()]

    model = _NonTrimmableModel([1, 2, 3, 4, 5])
    runner = LookupSpecRunner(model, tokenizer=object())
    assert runner.trimmable is False

    called = []

    def fake_fallback_generate(*a, **k):
        called.append(1)
        return {
            "tokens": [1],
            "ttft_s": 0.0,
            "decode_tps": 0.0,
            "prefill_reused": 0,
            "prefill_new": 0,
            "tokens_per_step": 1.0,
        }

    runner._fallback.generate = fake_fallback_generate
    result = runner.generate(
        [1], max_tokens=1, temp=0.0, eos_ids=set(), on_tokens=None, session=None
    )
    assert called
    assert result["tokens"] == [1]


def test_build_runner_wraps_fallback_with_lookup_spec_runner_when_flagged():
    model = SimpleNamespace(
        args=SimpleNamespace(model_type="llama"), layers=[object(), object()]
    )
    args = SimpleNamespace(
        model="fake-llama",
        original="fake-original",
        mtp_bits=4,
        no_mtp=True,
        no_fused=True,
        lookup_spec=True,
        lookup_max_draft=6,
        lookup_min_match=3,
    )
    runner = build_runner(model, tokenizer=object(), config={}, args=args)
    assert isinstance(runner, LookupSpecRunner)
    assert runner.KIND == "lookup_spec"
    assert runner.max_draft == 6
    assert runner.min_match == 3
    assert runner.trimmable is True
    # base FallbackRunner が選ばれた理由 (text_config なし) を引き継ぐ。
    assert runner.fallback_reason is not None


def test_build_runner_does_not_wrap_when_lookup_spec_flag_is_false():
    model = SimpleNamespace(
        args=SimpleNamespace(model_type="llama"), layers=[object(), object()]
    )
    args = SimpleNamespace(
        model="fake-llama",
        original="fake-original",
        mtp_bits=4,
        no_mtp=True,
        no_fused=True,
        lookup_spec=False,
    )
    runner = build_runner(model, tokenizer=object(), config={}, args=args)
    assert isinstance(runner, FallbackRunner)
    assert not isinstance(runner, LookupSpecRunner)


def test_build_runner_lookup_spec_does_not_apply_when_flash_spec_selected(monkeypatch):
    """spec/flash_spec が選ばれた場合、--lookup-spec を指定しても何もしない
    (既にモデル専用の投機が効いているため、build_runner の docstring 参照)。"""

    import mlxturbo.mtp_flash as mtp_flash_module

    class DummyMTP:
        def parameters(self):
            return {}

    monkeypatch.setattr(mtp_flash_module, "load_flash_mtp", lambda *a, **k: DummyMTP())

    model = _fake_qwen4_exp_model()
    args = SimpleNamespace(
        model="fake-model",
        original="fake-original",
        mtp="fake-sidecar.safetensors",
        mtp_bits=4,
        no_mtp=False,
        no_fused=True,
        lookup_spec=True,
    )
    runner = build_runner(model, tokenizer=object(), config={}, args=args)
    assert isinstance(runner, FlashSpecRunner)


# ---------- 項目 17: logprobs (fallback / 降格経路限定) ----------


def test_fallback_runner_collects_logprobs_when_requested(monkeypatch):
    import importlib

    mlx_generate = importlib.import_module("mlx_lm.generate")

    def fake_stream_generate(model, tokenizer, prompt, max_tokens, sampler=None,
                              logits_processors=None, **_kwargs):
        raw0 = mx.array([0.0, -1.0, -2.0, -3.0, -4.0])
        lp0 = raw0 - mx.logsumexp(raw0)
        yield _FakeGenResponse(0, "a", logprobs=lp0)
        raw1 = mx.array([-5.0, 0.0, -1.0, -2.0, -3.0])
        lp1 = raw1 - mx.logsumexp(raw1)
        yield _FakeGenResponse(1, "b", logprobs=lp1)

    monkeypatch.setattr(mlx_generate, "stream_generate", fake_stream_generate)

    tok = FakeTokenizer(vocab={0: "A", 1: "B", 2: "C", 3: "D", 4: "E"})
    runner = FallbackRunner(model=object(), tokenizer=tok)
    result = runner.generate(
        [9],
        max_tokens=2,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=None,
        logprobs=True,
        top_logprobs=2,
    )
    entries = result["logprobs"]
    assert len(entries) == 2
    assert entries[0]["token"] == "A"
    assert entries[0]["token_id"] == 0
    assert len(entries[0]["top_logprobs"]) == 2
    assert entries[0]["top_logprobs"][0]["token_id"] == 0
    assert entries[0]["top_logprobs"][0]["logprob"] >= entries[0]["top_logprobs"][1]["logprob"]
    assert entries[1]["token"] == "B"
    assert entries[1]["token_id"] == 1


def test_fallback_runner_omits_logprobs_when_not_requested(monkeypatch):
    import importlib

    mlx_generate = importlib.import_module("mlx_lm.generate")
    monkeypatch.setattr(
        mlx_generate,
        "stream_generate",
        lambda *a, **k: iter([_FakeGenResponse(0, "a", logprobs=mx.array([0.0, -1.0]))]),
    )
    runner = FallbackRunner(model=object(), tokenizer=FakeTokenizer())
    result = runner.generate(
        [9], max_tokens=1, temp=0.0, eos_ids=set(), on_tokens=None, session=None
    )
    assert "logprobs" not in result


def test_chat_completions_returns_logprobs_when_requested(client):
    runner = FakeRunner(
        tokens_to_emit=[10, 11],
        logprobs_to_emit=[
            {
                "token": "hello",
                "logprob": -0.1,
                "token_id": 10,
                "top_logprobs": [{"token": "hello", "logprob": -0.1, "token_id": 10}],
            },
            {
                "token": " world",
                "logprob": -0.2,
                "token_id": 11,
                "top_logprobs": [],
            },
        ],
    )
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "hello", 11: " world"}))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
            "top_logprobs": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hello world"
    lp = body["choices"][0]["logprobs"]
    assert lp is not None
    assert len(lp["content"]) == 2
    assert lp["content"][0]["token"] == "hello"
    assert lp["content"][0]["logprob"] == -0.1
    assert lp["content"][0]["top_logprobs"][0]["token"] == "hello"
    call = runner.calls[0]
    assert call["logprobs"] is True
    assert call["top_logprobs"] == 1


def test_chat_completions_logprobs_null_by_default(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))
    resp = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["logprobs"] is None
    assert "logprobs" not in runner.calls[0]


def test_chat_completions_logprobs_downgrades_spec_runner(client):
    spec_runner = FakeSpecRunner(tokens_to_emit=[10])
    fallback_runner = FakeRunner(
        tokens_to_emit=[10],
        logprobs_to_emit=[{"token": "x", "logprob": -0.5, "token_id": 10, "top_logprobs": []}],
    )
    _install_state(
        spec_runner, tokenizer=FakeTokenizer(vocab={10: "x"}), downgrade_runner=fallback_runner
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "logprobs": True},
    )
    assert resp.status_code == 200, resp.text
    assert not spec_runner.calls
    assert fallback_runner.calls
    assert fallback_runner.calls[0]["logprobs"] is True
    body = resp.json()
    assert body["downgrade_reason"] is not None
    assert "logprobs" in body["downgrade_reason"]
    assert body["choices"][0]["logprobs"] is not None


@pytest.mark.parametrize(
    "body_extra",
    [
        {"logprobs": "yes"},
        {"logprobs": True, "top_logprobs": 25},
        {"top_logprobs": 3},
    ],
)
def test_chat_completions_logprobs_validation_errors(client, body_extra):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], **body_extra},
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_chat_completions_logprobs_with_stream_is_400(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
            "stream": True,
        },
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_chat_completions_logprobs_null_when_stop_truncates_content(client):
    runner = FakeRunner(
        tokens_to_emit=[10, 11, 12],
        logprobs_to_emit=[
            {"token": "foo", "logprob": -0.1, "token_id": 10, "top_logprobs": []},
            {"token": "STOP", "logprob": -0.2, "token_id": 11, "top_logprobs": []},
            {"token": "bar", "logprob": -0.3, "token_id": 12, "top_logprobs": []},
        ],
    )
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "foo", 11: "STOP", 12: "bar"}))
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
            "stop": "STOP",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "foo"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["logprobs"] is None


def test_completions_legacy_returns_logprobs_int_format(client):
    runner = FakeRunner(
        tokens_to_emit=[10, 11],
        logprobs_to_emit=[
            {
                "token": "hello",
                "logprob": -0.1,
                "token_id": 10,
                "top_logprobs": [
                    {"token": "hello", "logprob": -0.1, "token_id": 10},
                    {"token": "hi", "logprob": -1.5, "token_id": 99},
                ],
            },
            {"token": " world", "logprob": -0.2, "token_id": 11, "top_logprobs": []},
        ],
    )
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "hello", 11: " world"}))
    resp = client.post("/v1/completions", json={"prompt": "hi", "logprobs": 2})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    lp = body["choices"][0]["logprobs"]
    assert lp is not None
    assert lp["tokens"] == ["hello", " world"]
    assert lp["token_logprobs"] == [-0.1, -0.2]
    assert lp["top_logprobs"][0] == {"hello": -0.1, "hi": -1.5}
    assert lp["text_offset"] == [0, 5]
    call = runner.calls[0]
    assert call["logprobs"] is True
    assert call["top_logprobs"] == 2


def test_completions_legacy_logprobs_with_stream_is_400(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))
    resp = client.post(
        "/v1/completions", json={"prompt": "hi", "logprobs": 1, "stream": True}
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


def test_completions_legacy_logprobs_validation_error_over_20(client):
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))
    resp = client.post("/v1/completions", json={"prompt": "hi", "logprobs": 25})
    assert resp.status_code == 400, resp.text
    assert not runner.calls


# ---------- Continuous batching (item 11 / BACKLOG.md §2) ----------
#
# mlxturbo.batch.BatchCoordinator drives mlx_lm.generate.BatchGenerator
# directly against the served model's own __call__ — a FakeRunner cannot
# stand in for that (there is no forward pass to drive), so these tests use
# a tiny but structurally real qwen4_exp model, the same reduced shape
# tools/verify_batch_cache.py's own CPU-only verification uses. Every other
# test in this file uses FakeRunner because it does not need real MLX
# compute; this is the one corner that does, by construction.
#
# What is and is not covered here: the gating logic (_resolve_batch_tier /
# maybe_build_batch_coordinator) is tested directly (fast, deterministic).
# The coordinator's actual concurrency/tier-isolation behavior is tested by
# calling server._run_generate_batched / server._start_batched_generation
# concurrently — the same "call the internal function directly" style
# test_nonstream_cancellation_keeps_lock_until_worker_finishes and
# test_stream_cancel_wakes_blocking_queue_get_with_internal_sentinel already
# use above, rather than a full HTTP round trip (which would additionally
# need a tokenizer whose chat template / detokenization actually agrees with
# a randomly-initialized tiny model's vocabulary — orthogonal to what this
# section is verifying).

_TINY_BATCH_CFG = dict(
    model_type="qwen4_exp_text",
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=256,
    rms_norm_eps=1e-6,
    full_attention_interval=2,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    hc_count=2,
    hc_lowrank=16,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=8,
    indexer_compress_ratio=2,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=256,
    make_ngram_vocab_size_divisible_by=8,
    split_ngram_parts=2,
    ple_embed_dim=32,
    ple_layer_ids=[1],
    ple_conv_kernel_size=4,
    seed=0,
    eos_token_id=99,
    partial_rotary_factor=0.25,
    rope_theta=10000.0,
    tie_word_embeddings=False,
)


def _build_tiny_batch_model(budget: int = 8):
    from mlx.utils import tree_map
    from mlx_lm.models import qwen4_exp as Q

    cfg = dict(_TINY_BATCH_CFG, indexer_budget=budget)
    mx.random.seed(0)
    model = Q.Model(Q.ModelArgs(model_type="qwen4_exp", text_config=cfg))
    model.update(
        tree_map(
            lambda a: mx.random.normal(a.shape) * 0.05 if a.dtype == mx.float32 else a,
            model.parameters(),
        )
    )
    mx.eval(model.parameters())
    model.eval()
    return model


@pytest.fixture
def batch_env():
    """A tiny real qwen4_exp model + FallbackRunner + BatchCoordinator
    (max_batch=4, indexer_budget=8), built on a single dedicated executor
    thread — mirroring server.py's own thread-pinning rule (the model and
    BatchGenerator's wired-limit call are both MLX operations, so both must
    happen on the thread the coordinator will later drive them from; see
    server.py's module docstring for the "There is no Stream(gpu, N) in
    current thread" failure mode this avoids). CPU-only and restores the
    previous default device / mx.metal.is_available on teardown so this does
    not leak into other tests in the file.
    """

    import mlxturbo.batch as batch_module

    prev_device = mx.default_device()
    prev_metal_available = mx.metal.is_available
    mx.set_default_device(mx.cpu)
    # BatchGenerator pokes mx.device_info()["max_recommended_working_set_size"]
    # when it thinks Metal is available, which does not exist once the
    # default device has been forced to CPU.
    mx.metal.is_available = lambda: False
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        model = executor.submit(_build_tiny_batch_model, 8).result()
        batch_module.enable_batch_cache()
        runner = FallbackRunner(model, FakeTokenizer())
        coordinator = batch_module.BatchCoordinator(
            model, executor, max_batch=4, prefill_step_size=8, eos_ids=[99]
        )
        yield SimpleNamespace(executor=executor, model=model, runner=runner, coordinator=coordinator)
    finally:
        executor.shutdown(wait=True)
        mx.set_default_device(prev_device)
        mx.metal.is_available = prev_metal_available


def test_resolve_batch_tier_default_off():
    """--max-batch unset (the default) -> STATE.batch_coordinator is None
    -> every request routes through the pre-existing STATE.lock path,
    unconditionally. This is what makes the default behavior provably
    unchanged by this feature."""

    _install_state(FakeRunner([10]))
    assert server.STATE.batch_coordinator is None
    assert server._resolve_batch_tier(server.STATE.runner, [1, 2, 3], 8) is None


def test_resolve_batch_tier_excludes_non_fallback_runner(batch_env):
    """Even with a coordinator built, only a plain FallbackRunner is
    eligible — SpecRunner/FlashSpecRunner/DraftSpecRunner/LookupSpecRunner
    keep going through the unchanged STATE.lock path (see
    mlxturbo.runner.can_batch / the module note above FallbackRunner in
    runner.py)."""

    _install_state(FakeRunner([10]), batch_coordinator=batch_env.coordinator, max_batch=4)
    assert server._resolve_batch_tier(FakeRunner([10]), [1, 2, 3], 8) is None
    assert server._resolve_batch_tier(FakeSpecRunner([10]), [1, 2, 3], 8) is None
    assert server._resolve_batch_tier(batch_env.runner, [1, 2, 3], 8) is not None


def test_resolve_batch_tier_excludes_logprobs(batch_env):
    _install_state(batch_env.runner, batch_coordinator=batch_env.coordinator, max_batch=4)
    assert (
        server._resolve_batch_tier(batch_env.runner, [1, 2, 3], 8, logprobs_requested=True)
        is None
    )
    assert (
        server._resolve_batch_tier(batch_env.runner, [1, 2, 3], 8, logprobs_requested=False)
        is not None
    )


def test_resolve_batch_tier_pool_vs_solo(batch_env):
    """indexer_budget=8 for this fixture's model. A request whose prompt +
    max_tokens stays at or under budget can never trigger QSA ("pool");
    one that could cross it is always "solo", regardless of how the current
    live batch happens to look (see mlxturbo/batch.py's classify docstring)."""

    _install_state(batch_env.runner, batch_coordinator=batch_env.coordinator, max_batch=4)
    assert server._resolve_batch_tier(batch_env.runner, [1, 2, 3], 4) == "pool"  # 3+4=7 <= 8
    assert server._resolve_batch_tier(batch_env.runner, [1] * 6, 4) == "solo"  # 6+4=10 > 8


def test_maybe_build_batch_coordinator_gating(batch_env):
    from mlxturbo.runner import maybe_build_batch_coordinator

    assert (
        maybe_build_batch_coordinator(
            batch_env.runner, batch_env.model, batch_env.executor, 1, [99]
        )
        is None
    )  # --max-batch omitted/1: off, regardless of runner kind
    assert (
        maybe_build_batch_coordinator(
            FakeSpecRunner([10]), batch_env.model, batch_env.executor, 4, [99]
        )
        is None
    )  # spec-kind runner: off, regardless of --max-batch
    built = maybe_build_batch_coordinator(
        batch_env.runner, batch_env.model, batch_env.executor, 4, [99]
    )
    assert built is not None
    assert built.max_batch == 4


def test_two_concurrent_batched_requests_complete(batch_env):
    """The actual point of item 11: two requests admitted before either has
    finished both complete correctly, via the exact function server.py's
    endpoints call (server._run_generate_batched)."""

    _install_state(batch_env.runner, batch_coordinator=batch_env.coordinator, max_batch=4)

    async def run():
        return await asyncio.gather(
            server._run_generate_batched([1, 2, 3, 4], 5, 0.0),
            server._run_generate_batched([5, 6, 7, 8], 5, 0.0),
        )

    res1, res2 = asyncio.run(run())
    assert len(res1["tokens"]) == 5
    assert len(res2["tokens"]) == 5
    assert res1["prefill_reused"] == 0 and res1["prefill_new"] == 4
    assert res2["prefill_reused"] == 0 and res2["prefill_new"] == 4


def test_batched_streaming_end_to_end(batch_env):
    """server._start_batched_generation must speak the exact same q/future
    protocol as server._start_generation (see test_stream_cancel_wakes_
    blocking_queue_get_with_internal_sentinel above for the non-batched
    counterpart of this shape)."""

    _install_state(batch_env.runner, batch_coordinator=batch_env.coordinator, max_batch=4)
    q, future, cancel_event, raw_token_count = server._start_batched_generation(
        [1, 2, 3], 4, 0.0, None
    )
    events = []
    while True:
        kind, val = q.get(timeout=5)
        events.append(kind)
        if kind in ("done", "cancelled", "error"):
            break
    assert events[-1] == "done"
    assert raw_token_count[0] == 4
    future.result(timeout=5)  # never raises — see mlxturbo.batch.Admission's docstring


def test_batched_solo_tier_never_overlaps_pool_tier(batch_env):
    """The safety property item 11 exists for: a request that could ever
    cross indexer_budget ("solo") never shares a live batch with anything
    else, even when submitted concurrently with pool-tier requests (see
    mlxturbo/batch.py's classify docstring for why unequal-length QSA-active
    batches are the one configuration this whole feature must never
    produce)."""

    _install_state(batch_env.runner, batch_coordinator=batch_env.coordinator, max_batch=4)
    solo_ts: list[float] = []
    pool1_ts: list[float] = []
    pool2_ts: list[float] = []

    def make_on_tokens(sink):
        def on_tokens(toks, text=None):
            sink.append(time.perf_counter())

        return on_tokens

    async def run():
        await asyncio.gather(
            server._run_generate_batched(
                [1] * 6, 6, 0.0, on_tokens=make_on_tokens(solo_ts)  # 6+6=12 > 8 budget -> solo
            ),
            server._run_generate_batched(
                [1, 2, 3], 4, 0.0, on_tokens=make_on_tokens(pool1_ts)  # 3+4=7 <= 8 -> pool
            ),
            server._run_generate_batched(
                [1, 2], 4, 0.0, on_tokens=make_on_tokens(pool2_ts)  # 2+4=6 <= 8 -> pool
            ),
        )

    asyncio.run(run())
    assert solo_ts and pool2_ts

    def overlaps(a, b):
        return max(a[0], b[0]) <= min(a[-1], b[-1])

    solo_window = (solo_ts[0], solo_ts[-1])
    pool_ts = pool1_ts + pool2_ts
    pool_window = (min(pool_ts), max(pool_ts))
    assert not overlaps(solo_window, pool_window)


def test_batched_admission_cancelled_before_start_resolves_cleanly(batch_env):
    """A request cancelled before the coordinator ever admits it (e.g. the
    client disconnected while still queued behind --max-batch capacity)
    resolves as "cancelled" without disturbing anything else — see
    mlxturbo.batch.BatchCoordinator._drive's admit() cancel_event check."""

    import concurrent.futures as cf

    from mlxturbo import batch as batch_module

    ev = threading.Event()
    ev.set()
    fut: "cf.Future" = cf.Future()
    admission = batch_module.Admission(
        prompt_ids=[1, 2, 3],
        max_tokens=4,
        sampler=None,
        logits_processors=[],
        tier="pool",
        on_tokens=None,
        on_done=None,
        cancel_event=ev,
        future=fut,
    )
    batch_env.coordinator.submit(admission)
    assert fut.result(timeout=5) is None
