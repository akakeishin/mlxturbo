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
from collections import OrderedDict

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

import fastmlx.server as server
from fastmlx.runner import FallbackRunner, FallbackSession, SpecRunner
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


def _default_qwen_tool_parser(text, tools=None):
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
        tool_parser=_default_qwen_tool_parser,
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
    """SpecRunner を模す: seed しかサポートしない。

    実物の SpecRunner.generate は seed 以外を **extra 経由でそのまま
    fastmlx.spec.SpecEngine.generate() に渡すが、SpecEngine.generate は
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
        return super().generate(
            prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra
        )


class FakeReusingRunner:
    """本物の Runner (SpecEngine/FallbackRunner) が持つ「渡された session の
    processed 列に対する LCP が session.processed 全体と一致するときだけ
    再利用扱いにする」契約だけを最小限で模す。ChatSession/FallbackSession の
    publish() シグネチャはそれぞれ違う (前者は 5 引数、後者は 2 引数) ので、
    どちらにも依存しないよう ``session.processed`` を直接書き換える (両方
    ただの list 属性)。session.py/runner.py の本物の実装とは独立に、
    server.py の session プール選択 (_select_session) が正しい session
    オブジェクトを渡せているかどうかだけを検証する目的。
    """

    KIND = "fallback"
    SUPPORTED_SAMPLING_PARAMS = FallbackRunner.SUPPORTED_SAMPLING_PARAMS

    def __init__(self, reply_tokens: list[int]):
        self.reply_tokens = reply_tokens
        self.calls: list[dict] = []

    def generate(self, prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **extra):
        prompt_ids = list(prompt_ids)
        reused = 0
        if session is not None and session.processed:
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
        toks = list(self.reply_tokens)
        if on_tokens:
            for t in toks:
                on_tokens([t])
        if session is not None:
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
    state = server.ModelState(
        runner=runner,
        tokenizer=tokenizer or FakeTokenizer(),
        session_pool=overrides.get("session_pool", OrderedDict()),
        session_factory=overrides.get("session_factory", ChatSession),
        lock=asyncio.Lock(),
        executor=concurrent.futures.ThreadPoolExecutor(max_workers=1),
        model_name=overrides.get("model_name", "test-model"),
        eos_ids=overrides.get("eos_ids", {999}),
        max_tokens_cap=overrides.get("max_tokens_cap", 4096),
        default_temp=overrides.get("default_temp", 0.7),
        created_ts=0,
        max_sessions=overrides.get("max_sessions", 8),
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


# ---------- 2b. SpecRunner 経路でも恒等値 (分布を変えない既定値) は通す ----------
#
# opencode / OpenAI SDK など実クライアントは top_p=1.0 や frequency_penalty=0
# のような「未指定と等価」な値をキー付きで送ってくる。これらは分布を一切
# 変えないので、SUPPORTED_SAMPLING_PARAMS に無いキーでも 400 にしてはいけない
# (このバグの実測: opencode を spec runner のモデルに繋ぐと最初のリクエストで
# 即 400 になっていた)。
#
# ただし 400 にしないだけでは足りない: SpecRunner.generate は seed 以外の
# kwarg を fastmlx.spec.SpecEngine.generate() へそのまま **extra 経由で
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
        {"top_p": 1.0},
        {"top_k": 0},
        {"min_p": 0.0},
        {"frequency_penalty": 0.0},
        {"presence_penalty": 0.0},
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
                              logits_processors=None, prompt_cache=None):
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
                              logits_processors=None, prompt_cache=None):
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
                              logits_processors=None):
        calls.append({"prompt": list(prompt)})
        return
        yield  # pragma: no cover - keep this a generator

    monkeypatch.setattr(mlx_generate, "stream_generate", fake_stream_generate)

    runner = FallbackRunner(model=object(), tokenizer=object())
    runner.generate(
        [1, 2, 3], max_tokens=0, temp=0.0, eos_ids=set(), on_tokens=None, session=None
    )
    assert calls[0]["prompt"] == [1, 2, 3]


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
    assert blocks[0] == {"type": "thinking", "thinking": "pondering. "}
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
    text_deltas = "".join(
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta"
    )
    assert thinking_deltas == "pondering. "
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
    assert thinking_blocks[0]["thinking"] == "aa"


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


def test_responses_previous_response_id_is_400(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi", "previous_response_id": "resp_123"},
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls
    assert "previous_response_id" in resp.json()["error"]["message"]


def test_responses_store_true_is_400(client):
    tok = FakeTokenizer(vocab={10: "ok"})
    runner = FakeRunner(tokens_to_emit=[10])
    _install_state(runner, tokenizer=tok)

    resp = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi", "store": True},
    )
    assert resp.status_code == 400, resp.text
    assert not runner.calls


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
