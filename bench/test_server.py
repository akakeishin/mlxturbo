"""fastmlx/server.py の追加分 (サンプリングパラメータ / /health / CORS /
prompt cache 可視化 / /v1/completions) を、モデルをロードせずに検証する。

STATE (fastmlx.server.STATE) はモジュールグローバルなので、各テストは
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

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

import fastmlx.server as server
from fastmlx.runner import FallbackRunner, SpecRunner
from fastmlx.spec import ChatSession


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


class FakeTokenizer:
    def __init__(
        self,
        vocab: dict[int, str] | None = None,
        has_thinking: bool = False,
        think_start_tokens: list[int] | None = None,
        think_end_tokens: list[int] | None = None,
        eos_token_ids=(999,),
        prompt_ids: list[int] | None = None,
    ):
        self.vocab = vocab or {}
        self.has_thinking = has_thinking
        self.think_start_tokens = think_start_tokens or []
        self.think_end_tokens = think_end_tokens or []
        self.eos_token_ids = set(eos_token_ids)
        self._prompt_ids = prompt_ids if prompt_ids is not None else [1, 2, 3]

    @property
    def detokenizer(self) -> _FakeDetokenizer:
        return _FakeDetokenizer(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=None):
        return list(self._prompt_ids)


# ---------- フェイク Runner ----------


class FakeRunner:
    """generate() に渡された kwargs をすべて記録する。tokens_to_emit を
    on_tokens 経由で 1 個ずつ流し、最後に res dict を返す。"""

    KIND = "fallback"
    SUPPORTED_SAMPLING_PARAMS = FallbackRunner.SUPPORTED_SAMPLING_PARAMS

    def __init__(self, tokens_to_emit: list[int], prefill_reused: int = 0):
        self.tokens_to_emit = tokens_to_emit
        self.prefill_reused = prefill_reused
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
        toks = self.tokens_to_emit
        if on_tokens:
            for t in toks:
                on_tokens([t])
        return {
            "tokens": list(toks),
            "ttft_s": 0.001,
            "decode_tps": 100.0,
            "prefill_reused": self.prefill_reused,
            "prefill_new": len(prompt_ids) - self.prefill_reused,
            "tokens_per_step": 1.0,
        }


class FakeSpecRunner(FakeRunner):
    """SpecRunner を模す: seed しかサポートしない。"""

    KIND = "spec"
    SUPPORTED_SAMPLING_PARAMS = SpecRunner.SUPPORTED_SAMPLING_PARAMS


# ---------- 共通ヘルパ ----------


def _install_state(runner, tokenizer=None, **overrides) -> server.ModelState:
    state = server.ModelState(
        runner=runner,
        tokenizer=tokenizer or FakeTokenizer(),
        session=ChatSession(),
        lock=asyncio.Lock(),
        executor=concurrent.futures.ThreadPoolExecutor(max_workers=1),
        model_name=overrides.get("model_name", "test-model"),
        eos_ids=overrides.get("eos_ids", {999}),
        max_tokens_cap=overrides.get("max_tokens_cap", 4096),
        default_temp=overrides.get("default_temp", 0.7),
        created_ts=0,
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
        ("top_k", -1),
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


def test_spec_runner_allows_seed(client):
    runner = FakeSpecRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=FakeTokenizer(vocab={10: "x"}))

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "seed": 7},
    )
    assert resp.status_code == 200, resp.text
    assert runner.calls[0]["seed"] == 7


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
    def __init__(self, token, text):
        self.token = token
        self.text = text


def test_fallback_runner_seed_makes_output_reproducible(monkeypatch):
    """FallbackRunner.generate に seed を渡すと mx.random.seed() が呼ばれ、
    以後の (フェイクの) mx.random 呼び出し列が再現することを確認する。
    実モデルは要らない: stream_generate をスタブに差し替え、スタブ自体が
    mx.random から値を引く形にすることで「seed で結果が固定される」という
    契約を、本物の乱数発生器を使ったまま検証する。
    """

    import importlib

    mlx_generate = importlib.import_module("mlx_lm.generate")

    def fake_stream_generate(model, tokenizer, prompt, max_tokens, sampler=None, logits_processors=None):
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

    def fake_stream_generate(model, tokenizer, prompt, max_tokens, sampler=None, logits_processors=None):
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


# ---------- 4. CORS ----------


def test_cors_disabled_by_default(client):
    runner = FakeRunner(tokens_to_emit=[])
    _install_state(runner)
    resp = client.get("/health", headers={"Origin": "http://example.com"})
    assert "access-control-allow-origin" not in resp.headers


def test_cors_enabled_via_helper():
    from fastapi import FastAPI

    from fastmlx.server import _add_cors_middleware

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
