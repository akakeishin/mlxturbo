"""OpenAI 互換 / Anthropic 互換 HTTP サーバー。

既存の投機デコードエンジン (fastmlx.spec.SpecEngine) をそのまま使う。モデルは
起動時に 1 回だけロードして常駐させ、リクエストはグローバルなロックで直列化
する: 91GB 級のモデルを 128GB 機に載せている前提なので、同時実行 (並列
バッチング) はしない。待機中のリクエストもコネクションは保ったまま
(asyncio のロック待ちで単に await するだけ)、ロックが空くまで待たせる。

会話履歴はクライアントが毎ターン全文を送り直す (OpenAI/Anthropic どちらの
API も無状態) ので、cli.py の run_turn と同じやり方で使う: 単一の
ChatSession をプロセス全体で使い回し、新しいプロンプトの先頭が前回処理した
トークン列と一致すれば SpecEngine 側が自動で prefill を再利用する
(fastmlx/spec.py の ChatSession.publish/invalidate)。一致しなければ黙って
全再構築にフォールバックするだけなので、複数の会話が入れ替わり立ち替わり
来ても壊れない (遅くなるだけ)。

SpecEngine が受け付けないモデル (Llama/Gemma/dense Qwen や、GDN ハイブリッド
でもレイアウトが異なるもの) では、起動時に fastmlx.runner.build_runner が
mlx_lm.generate.stream_generate による普通の (非投機) 生成に自動でフォール
バックする。どちらの経路でも HTTP 層から見た形は同一 (Runner.generate が
同じ dict を返す) で、起動時にどちらの経路が有効かを一行ログで出す。詳細は
fastmlx/runner.py を参照。

MLX の計算グラフ (モデルの重み・KV キャッシュを含む) はロードしたスレッドに
紐づく。asyncio.to_thread や汎用スレッドプールで別スレッドへ逃がすと
"There is no Stream(gpu, N) in current thread" で落ちる (実測で確認済み:
ロードと forward を同一スレッドで行えば通り、別スレッドだと通らない)。その
ため、モデルのロードも生成呼び出しも**専用の単一ワーカースレッド**
(``STATE.executor``, ``max_workers=1``) に固定して行う。requests はもともと
グローバルロックで直列化する設計なので、単一スレッドに寄せても並行性は
失わない。

ストリーミング経路はロックの中で ``executor.submit(worker)`` の Future を
保持し、クライアント切断 (StreamingResponse がジェネレータを ``aclose()``
する = ``GeneratorExit``) やエラーで早期に抜ける場合も、``finally`` でその
Future を待ってからロックを解放する。そうしないと、まだ生成中のワーカーが
共有の ``STATE.session`` (ChatSession) を触っている間に次のリクエストが
ロックを取れてしまい、同じ session を 2 つの生成が同時に書き換える。

thinking (推論過程) の扱い: OpenAI の ``reasoning_effort`` / Anthropic の
``thinking`` だけを読む (fastmlx 独自フィールドは無い)。値はトークン予算
(budget) に写し、``ThinkingRouter`` が生トークン列を reasoning/content の
2 チャンネルへ振り分ける。マーカーは mlx_lm.TokenizerWrapper の公開 API
(``has_thinking``/``think_start_tokens``/``think_end_tokens``) から引き、
引けないモデルでは常に content 一本 (分離しない)。予算超過時は「思考を
強制的に閉じて本文を続けさせる」その場再開はしていない (SpecEngine/
mlx_lm.stream_generate のどちらも生成途中の割り込みを許さないため) —
それ以降の出力をクライアントへ転送しないという、観測可能な形で打ち切る。
詳細は ThinkingRouter の docstring を参照。
"""

import argparse
import asyncio
import concurrent.futures
import functools
import json
import os
import queue
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ._mlx_compat import mlx_lm_load
from .runner import Runner, build_runner
from .spec import ChatSession

app = FastAPI()


def _add_cors_middleware(fastapi_app: FastAPI, allowed_origins: list[str]) -> None:
    """``--allowed-origins`` が指定されたときだけ main() から呼ぶ。既定
    (未指定) では一切呼ばれない = CORSMiddleware 自体が付かず、ブラウザ
    からのクロスオリジン fetch は常にブロックされたまま (ローカル専用)。
    Open WebUI 等、ブラウザで動く UI から直接叩きたい場合に使う。
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
    session: ChatSession
    lock: asyncio.Lock
    executor: concurrent.futures.ThreadPoolExecutor
    model_name: str  # served id: GET /v1/models と全レスポンスの "model" 欄
    eos_ids: set
    max_tokens_cap: int
    default_temp: float
    created_ts: int


STATE: ModelState | None = None


# ---------- 入力の正規化・検証 ----------


class ContentNormalizationError(ValueError):
    """content の正規化中に見つかったクライアント側の入力不備の基底クラス。
    5xx ではなく 400 (各プロトコルのエラー形式) で返すためのマーカー。"""


class MultimodalContentError(ContentNormalizationError):
    """content にテキスト以外のブロック (image_url/image/input_audio 等) が
    含まれていた。fastmlx はテキスト専用 (convert_flash.py が変換時に
    vision_tower.*/model.visual.* を落としている) なので、黙って読み飛ばすと
    クライアントは画像が読まれたと思って会話を続けてしまう。400 で明示する。
    """

    def __init__(self, block_type):
        self.block_type = block_type
        super().__init__(
            f"this model is text-only; unsupported content block type: {block_type!r}"
        )


class InvalidContentError(ContentNormalizationError):
    """content の形そのものが壊れている (キー欠落・型不一致等)。黙って空文字や
    str() に丸めると壊れた入力がそのまま空プロンプトとして通ってしまうので、
    ここで弾いて 400 にする。"""


def _content_to_text(content) -> str:
    """OpenAI/Anthropic どちらの content 形式 (文字列 or ブロックのリスト) も
    プレーンテキストへ落とす。tokenizer.apply_chat_template は文字列の
    content しか想定していないため。

    ``type: "text"`` だけのブロック列は単一文字列と同じ結果に結合する。
    それ以外の type (image_url/image/input_audio 等) を見つけたら
    ``MultimodalContentError`` を送出する。それ以外の壊れた形 (content が
    無い/null、ブロックに type が無い、text ブロックの text が文字列でない、
    content 自体が文字列でもリストでもない) は黙って "" や str(...) に丸め
    ず ``InvalidContentError`` を送出する: 壊れた入力を空プロンプトとして
    通すと 500 にはならないが原因が分からない挙動になる。
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
    """messages が文字列や数値のリストだったりすると、後続の m.get(...) が
    AttributeError で 500 になる。形だけ先に確認して 400 で弾く。"""

    if not isinstance(messages, list):
        return "'messages' must be a list"
    for m in messages:
        if not isinstance(m, dict):
            return "each item in 'messages' must be an object with 'role' and 'content'"
    return None


# ---------- tool calling: 履歴 (tool_calls / tool_result) の正規化 ----------
#
# apply_chat_template に渡す messages は、どちらのプロトコルから来ても
# OpenAI 形式 (assistant: {"role","content","tool_calls":[{"id","type":
# "function","function":{"name","arguments": <dict>}}]} / tool 結果:
# {"role":"tool","tool_call_id","content"}) に揃える。チャットテンプレート
# 自体が (mlx_lm の tool_parser 自動選択と同様に) OpenAI/Qwen 系の tool
# calling 規約を前提にしているため。arguments は dict のまま渡す
# (mlx_lm.server.process_message_content も同様に json.loads してから
# テンプレートへ渡している — テンプレートは大抵 tojson でシリアライズし
# 直すので、文字列を二重にエンコードしてはいけない)。


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
            else:
                raise MultimodalContentError(btype)
        # tool_result (前ターンの結果) はこのメッセージの通常テキストより
        # 先に role: "tool" として並べる。同じ user メッセージ内でテキスト
        # と tool_result が混在するのは稀だが、その場合も出現順を保つ
        # (tool_result 群 -> このメッセージ自身のテキスト)。
        out.extend(tool_result_msgs)
        joined_text = "".join(text_parts)
        if role == "assistant" and tool_calls:
            out.append({"role": "assistant", "content": joined_text, "tool_calls": tool_calls})
        elif joined_text or not tool_result_msgs:
            out.append({"role": role, "content": joined_text})
    return out


def _naive_prompt_ids(messages: list[dict]):
    """tokenizer にチャットテンプレートが無いモデル向けの素朴なフォールバック。

    role: content を並べて素の文字列にするだけ。Qwen 系限定の前提を持ち
    込まない (将来 gemma/kimi/glm を載せても、チャットテンプレートさえ
    あればそちらを使い、無ければここに落ちる)。
    """

    lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    lines.append("assistant:")
    return STATE.tokenizer.encode("\n".join(lines))


def _apply_template(
    messages: list[dict], enable_thinking: bool | None = None, tools: list | None = None
):
    """``enable_thinking`` は標準フィールド (reasoning_effort/thinking) から
    解決済みの値。省略時 (None) はテンプレートの既定に任せる — mlx_lm の
    TokenizerWrapper.apply_chat_template は enable_thinking が渡されなければ
    ``self.has_thinking`` を自分で補う (thinking 対応モデルは既定 on)。

    thinking が有効なテンプレートの一部は、確定した過去ターンの assistant
    メッセージを履歴へ組み込む際に <think>...</think> を落とす (再度読ませる
    必要がないため)。そのため実際に生成されたトークン列より次ターンの
    prompt が短くなり、ChatSession の LCP 再利用 (前回処理列がそのまま新
    prompt の接頭辞であることを要求する) が原理的に成立しなくなる。
    enable_thinking=False (reasoning_effort: "none" 等) を渡せばこの型の
    モデルでも再利用の対象に戻る。

    ``tools`` は tool_choice 解決後の OpenAI 形式の tools 配列 (None なら
    このターンはツールを見せない)。呼び出し側が既に
    ``_check_tool_calling_support`` で ``tokenizer.has_tool_calling`` を
    確認済みである前提 — ここでは渡された tools をそのまま
    ``apply_chat_template(tools=...)`` へ転送するだけ (HF の
    apply_chat_template は tools を第一級の kwarg として受け付けるので、
    黙って無視されることはない — 無視されるとしたらそれは
    has_tool_calling が拾えていないモデルであり、その判定は呼び出し側の
    責任)。
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
        # チャットテンプレートを持たないモデル (chat_template が未設定)。
        return _naive_prompt_ids(messages)


# reasoning_effort -> thinking トークン予算の目安。fastmlx は思考の深さを
# 直接は調整できないので、予算 (ThinkingRouter が数える生成済みトークン数の
# 上限) に写して実際に効かせる。未知の値は 400 にせず medium 相当とする
# (将来 effort の値が増えても壊れない)。
_REASONING_EFFORT_BUDGET = {
    "minimal": 512,
    "low": 2048,
    "medium": 8192,
    "high": 32768,
}
_REASONING_EFFORT_DEFAULT_BUDGET = _REASONING_EFFORT_BUDGET["medium"]


def _resolve_thinking(body: dict, protocol: str) -> tuple[bool | None, int | None, str | None]:
    """標準フィールドだけを読み、(enable_thinking 用の kwarg, 予算, エラー)
    を返す。fastmlx 独自フィールドは無い — 何も指定が無ければ
    (None, None, None) (テンプレート任せ・予算強制なし)。

    予算: None = 強制なし (自然に終わるまで待つ)、0 = thinking 完全オフ、
    正の整数 = ThinkingRouter がその数のトークンで打ち切る。呼び出し側で
    max_tokens に対して clamp すること (ここでは知らないため)。
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
    return None, None, f"'thinking.type' must be 'enabled' or 'disabled', got {t!r}"


# ---------- tool calling: リクエスト側 (tools/tool_choice の検証・解決) ----------
#
# tokenizer.apply_chat_template の ``tools=`` に渡す形式は OpenAI 形式
# (``[{"type": "function", "function": {"name", "description", "parameters"}}]``)
# に統一する。Anthropic の ``input_schema`` 形式はここで OpenAI 形式へ変換
# してから渡す (jinja 側のテンプレートはどのみち Qwen 系の json_tools 前提
# なので、テンプレートへ渡す形は 1 通りに揃えておいたほうが破綻しにくい)。
# tokenizer.tool_parser (mlx_lm 側で自動選択されるモデル固有パーサ、例:
# qwen3_coder は tools の parameters.properties から引数の型を引く) もこの
# OpenAI 形式の tools を期待するので、同じ変換結果を使い回せる。


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
    """``tools``/``tool_choice`` (OpenAI 形式) を読み、apply_chat_template へ
    渡す tools (OpenAI 形式のまま) を解決する。tools が無い/空なら
    (None, None) (tool_choice は無視する — 実 API も同様)。

    ``tool_choice: "none"`` は「tools を渡さない」で済ませる (このターンは
    ツールを見せない、渡された tools 自体は無視)。``"required"`` と特定
    関数指定は、モデル側を強制する手段が無いので 400 にする (黙って auto
    として扱わない)。
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
    """Anthropic 形式版。戻り値の tools は (Anthropic 形式のままではなく)
    ``_anthropic_tools_to_openai`` で OpenAI 形式へ変換済みのもの — 呼び出し
    側は apply_chat_template/tool_parser のどちらにもこれをそのまま渡せる。
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
    """resolved_tools (tool_choice 解決後、None なら今回はツールを見せない)
    が None でなければ、tokenizer が tool_call マーカーを持つか
    (``has_tool_calling``、mlx_lm の TokenizerWrapper がチャットテンプレート
    文字列から自動検出する — has_thinking と同じ仕組み) を確認する。持た
    なければ黙って無視せず 400 にする。
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


class ThinkingRouter:
    """モデルの生トークン列を reasoning (thinking) / content の 2 チャンネル
    へ振り分ける。マーカーは推測せず、mlx_lm.TokenizerWrapper の公開 API
    (``has_thinking``/``think_start_tokens``/``think_end_tokens``、Qwen の
    ``<think>``/``</think>`` や channel 方式のモデルも同じ口でカバーされる)
    から引く。引けないモデル (``has_thinking`` が False) では常に content
    一本になる — 「このモデルでは thinking を分離できない」という意味で、
    全文が従来どおり本文に入る。

    マーカー境界をまたいで正しくデコードするため、reasoning と content で
    別々のストリーミング detokenizer を使う (BPE の trailing-space マージ
    等は各チャンネル内で完結する)。マーカートークン列 (複数トークンのことが
    ある) を跨いだ誤検出を避けるため、マーカー長ぶんの生トークンを常に
    バッファして先読みし、一致しないと分かった分だけ確定でどちらかの
    detokenizer へ流す。

    budget (int | None): thinking ブロック内で許すトークン数の上限。None は
    無制限 (自然に終わるまで待つ)。0 は「thinking 完全オフ」で、そもそも
    ルーティングしない。

    予算超過時にできることの限界: SpecEngine (spec.py) も
    mlx_lm.generate.stream_generate も、生成の途中で外部から割り込んで
    「ここで打ち切り、続きを別のプロンプトから作り直す」ライブな口を持たな
    い (on_tokens の戻り値は見ていない — 呼び出し元を変えても、既存の
    投機デコードエンジンにその変更を入れない限り実現できない)。そのため
    ここで実装しているのは「予算に達した後のトークンはクライアントへ転送
    しない」までで、「思考を強制的に閉じて本文を続けさせる」その場再開は
    やっていない。budget_exceeded フラグで検知でき、呼び出し側はこれを
    finish_reason/stop_reason に反映する。
    (2 段構えの再開そのものは技術的に不可能ではない: 予算に達した時点で
    現在の generate() 呼び出しを見限り、prompt_ids + それまでに生成された
    トークン + </think> を新しい prompt として generate() をもう一度呼べば、
    ChatSession の LCP 再利用によって prefill の再計算は避けられる。ただし
    ストリーミング/非ストリーミング両経路・SpecRunner/FallbackRunner 両方で
    このチャンク分割を整合させる実装コストとバグ面積が、この機能の価値に
    見合わないと判断し、今回は見送った。)

    tool_calling_enabled (bool): この生成でツール呼び出し検出も行うかどうか
    (呼び出し側が resolved_tools を渡した = tool_choice が "none" ではなく
    tools が実際に指定されたか、に対応)。tokenizer 側が tool_call マーカーを
    持たない場合は無条件で無効になる (``has_thinking`` と同じパターン、
    ``self.tool_enabled`` 参照)。

    有効なとき、content フェーズの間 ``tokenizer.tool_call_start_tokens``
    の出現を先読みで監視し (thinking フェーズが end marker を待つのと同じ
    ローリングウィンドウ方式)、一致したら "tool" フェーズへ入って
    ``tokenizer.tool_call_end_tokens`` までのテキストを "tool" チャンネルと
    して返す (マーカー自体はどちらのチャンネルにも含まれない — mlx_lm の
    ToolCallFormatter/_process_control_tokens と同じ扱い)。1 回の生成内で
    ツール呼び出しは複数回起こりうるので content <-> tool は何度でも往復
    する。"tool" フェーズへの出入りはテキストが空でも必ず ("tool_start", "")
    / ("tool_end", matched: bool) を 1 回ずつ返すので、呼び出し側はそれを
    境界として個々の呼び出しをグルーピングできる (feed() が空セグメントを
    省略する都合上、境界自体を独立したイベントにしないと "たまたま前後の
    テキストが空だった 2 回の呼び出し" を誤って 1 回に結合してしまう)。
    ``tool_end`` の bool は「本当に終了マーカーで閉じたか」— False は
    max_tokens 等で強制的に打ち切られたことを示し、呼び出し側は再構成時に
    終了マーカー文字列を足さない判断に使う。
    """

    def __init__(
        self,
        tokenizer,
        budget: int | None,
        eos_ids: set,
        tool_calling_enabled: bool = False,
    ):
        self.tokenizer = tokenizer
        self.budget = budget
        self.eos_ids = eos_ids
        self.enabled = bool(getattr(tokenizer, "has_thinking", False)) and budget != 0
        self.tool_enabled = tool_calling_enabled and bool(
            getattr(tokenizer, "has_tool_calling", False)
        )
        self.phase = "detect" if self.enabled else "content"
        self.buf: list[int] = []
        self.tool_buf: list[int] = []
        self.think_detok = tokenizer.detokenizer
        self.content_detok = tokenizer.detokenizer
        self.tool_detok = tokenizer.detokenizer
        self.thinking_token_count = 0
        self.budget_exceeded = False
        self._start_tokens = list(tokenizer.think_start_tokens) if self.enabled else []
        self._end_tokens = list(tokenizer.think_end_tokens) if self.enabled else []
        self._tool_start = (
            list(tokenizer.tool_call_start_tokens) if self.tool_enabled else []
        )
        self._tool_end = (
            list(tokenizer.tool_call_end_tokens) if self.tool_enabled else []
        )

    def _feed_content_token(self, t: int, out: list) -> None:
        """content フェーズの 1 トークンを処理する。tool ルーティングが
        無効なら即デコードするだけ (従来どおり)。有効なら
        ``tool_call_start`` への先読みバッファを持つ ("thinking" フェーズが
        end marker を待つのと同型のローリングウィンドウ)。"""

        if not self.tool_enabled:
            self.content_detok.add_token(t)
            seg = self.content_detok.last_segment
            if seg:
                out.append(("content", seg))
            return
        self.tool_buf.append(t)
        if len(self.tool_buf) <= len(self._tool_start):
            if self.tool_buf == self._tool_start:
                self.phase = "tool"
                self.tool_buf = []
                out.append(("tool_start", ""))
            return
        cut = len(self.tool_buf) - len(self._tool_start)
        flush, self.tool_buf = self.tool_buf[:cut], self.tool_buf[cut:]
        for ft in flush:
            self.content_detok.add_token(ft)
        seg = self.content_detok.last_segment
        if seg:
            out.append(("content", seg))
        if self.tool_buf == self._tool_start:
            self.phase = "tool"
            self.tool_buf = []
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
                    # 冒頭が think_start と一致しない = このターンは考えて
                    # いない。貯めていた分は content フェーズのロジック
                    # (tool_call_start 監視込み) へそのまま再投入する —
                    # ここで無条件に content_detok へ流すと、たまたま冒頭が
                    # tool_call_start の先頭と重なっていた場合を取りこぼす。
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
                    continue
                # ウィンドウが end marker 長を超えた分だけ確定で thinking へ
                # 流す (末尾は常に marker 長ぶん保持して先読みを続ける)。
                cut = len(self.buf) - len(self._end_tokens)
                flush, self.buf = self.buf[:cut], self.buf[cut:]
                for ft in flush:
                    self.think_detok.add_token(ft)
                    self.thinking_token_count += 1
                    if self.budget is not None and self.thinking_token_count > self.budget:
                        self.budget_exceeded = True
                        break
                seg = self.think_detok.last_segment
                if seg:
                    out.append(("reasoning", seg))
                if self.budget_exceeded:
                    return out
                if self.buf == self._end_tokens:
                    self.phase = "content"
                    self.buf = []
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
            # マーカー長に届かないまま生成が終わった (極端に短い max_tokens
            # 等)。安全側で content 扱いにする (tool_call_start 監視込み)。
            pending, self.buf = self.buf, []
            for t in pending:
                self._feed_content_token(t, out)
        elif self.phase == "thinking":
            # </think> を出さないまま max_tokens/eos に達した。バッファに
            # 残る未確定トークンは thinking として確定させる。
            for t in self.buf:
                self.think_detok.add_token(t)
                self.thinking_token_count += 1
            self.buf = []
            self.think_detok.finalize()
            seg = self.think_detok.last_segment
            if seg:
                out.append(("reasoning", seg))
        elif self.phase == "tool":
            # </tool_call> を出さないまま max_tokens/eos に達した。バッファに
            # 残る未確定トークンは tool として確定させ、"tool_end" は
            # (本当の終了マーカーではないので) False で知らせる。
            for t in self.tool_buf:
                self.tool_detok.add_token(t)
            self.tool_buf = []
            self.tool_detok.finalize()
            seg = self.tool_detok.last_segment
            if seg:
                out.append(("tool", seg))
            out.append(("tool_end", False))
        # content フェーズで tool_call_start の先読み中 (まだ marker と
        # 確定していないトークン列) のまま生成が終わった分は、確定させて
        # content 側へ落とす (取りこぼし防止の安全弁)。
        for t in self.tool_buf:
            self.content_detok.add_token(t)
        self.tool_buf = []
        self.content_detok.finalize()
        seg = self.content_detok.last_segment
        if seg:
            out.append(("content", seg))
        return out


def _parse_positive_int(raw, cap: int, field_name: str) -> tuple[int, str | None]:
    """int に変換して 1..cap へ収める。変換できない・0 以下は 400 対象の
    エラー文字列を返す (呼び出し側が protocol 形式へ包む)。"""

    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0, f"'{field_name}' must be an integer"
    if isinstance(raw, bool):  # bool は int のサブクラスなので明示的に弾く
        return 0, f"'{field_name}' must be an integer"
    if value < 1:
        return 0, f"'{field_name}' must be a positive integer"
    return min(value, cap), None


def _resolve_max_tokens_openai(body: dict, cap: int) -> tuple[int, str | None]:
    """OpenAI では ``max_tokens`` は非推奨、新しい SDK は ``max_completion_tokens``
    を送る。両方読み、後者を優先する。OpenAI は 0 を不正とする (Anthropic と
    違い ``max_tokens: 0`` の特別扱いは無い) ので、そのまま _parse_positive_int
    (1 未満は 400) に通す。"""

    raw = body.get("max_completion_tokens")
    if raw is None:
        raw = body.get("max_tokens")
    if raw is None:
        return cap, None
    return _parse_positive_int(raw, cap, "max_tokens")


def _parse_anthropic_max_tokens(raw, cap: int) -> tuple[int, str | None]:
    """Anthropic の実 API は ``max_tokens: 0`` を正当な入力として許容する
    (生成せずに終わる特別なレスポンスを返す)。OpenAI 側の
    ``_parse_positive_int`` と違い 0 だけは通し、負数と非整数は 400 にする。"""

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
    # bool はサブクラスなので明示的に弾く。float はここでは真の整数値
    # (5.0 等) でも拒否する: int(1.5) は例外を出さず黙って 1 に切り捨てる
    # ため、"'field' must be an integer" のつもりが静かに違う値を通してしまう。
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


# OpenAI と Anthropic のどちらも同じフィールド名 (top_p/top_k/min_p/
# repetition_penalty/presence_penalty/frequency_penalty/logit_bias/seed) を
# 使うので、プロトコル分岐なしで共通のパーサ 1 つで足りる (thinking/
# stop_sequences のように呼び名が違うものだけプロトコル別処理が必要)。
#
# 指定されなかったキーは戻り値の dict に含めない — Runner.generate 側の
# デフォルト引数 (= 無効化された値) にそのまま委ねるため。
def _parse_sampling_params(body: dict) -> tuple[dict, str | None]:
    params: dict = {}

    top_p, err = _parse_optional_float(body, "top_p", 0.0, 1.0)
    if err is not None:
        return {}, err
    if top_p is not None:
        params["top_p"] = top_p

    top_k, err = _parse_optional_int(body, "top_k", 0)
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


def _check_sampling_support(params: dict) -> str | None:
    """現在の Runner (SpecRunner/FallbackRunner) が受け付けないサンプリング
    パラメータが指定されていれば、理由付きのエラー文字列を返す。SpecRunner
    が seed 以外を弾く理由は fastmlx/runner.py の SpecRunner docstring
    (Block Verification の分布保証との関係) を参照。
    """

    if not params:
        return None
    supported = getattr(STATE.runner, "SUPPORTED_SAMPLING_PARAMS", frozenset())
    unsupported = sorted(set(params) - supported)
    if not unsupported:
        return None
    kind = getattr(STATE.runner, "KIND", type(STATE.runner).__name__)
    return (
        f"this model is served via the '{kind}' runner, which does not support: "
        f"{', '.join(unsupported)}. The speculative-decoding runner only supports "
        "'seed' among the extended sampling parameters, because its correctness "
        "guarantee (rejection sampling against the exact target distribution) "
        "assumes temperature-only sampling; changing the base distribution would "
        "silently break that guarantee."
    )


def _stop_sequences(body: dict) -> list[str]:
    """OpenAI の ``stop`` (文字列または配列) と Anthropic の
    ``stop_sequences`` (配列) の両方を受け付ける。どちらのエンドポイントでも
    同じ処理を使う (リクエストの本体形式にしか依らないため)。"""

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
    """`stops` のうち `text` 中で最も早く始まる一致を返す (開始位置, 一致文字列)。"""

    best = None
    for s in stops:
        idx = text.find(s)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, s)
    return best


def _openai_error(message: str, status: int = 400, err_type: str = "invalid_request_error", code: str | None = None):
    err = {"message": message, "type": err_type}
    if code is not None:
        err["code"] = code
    return JSONResponse(status_code=status, content={"error": err})


def _anthropic_error(message: str, status: int = 400, err_type: str = "invalid_request_error"):
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": err_type, "message": message}},
    )


def _check_model_openai(body: dict):
    """リクエストの model がサーブ中のモデルと違えば 404 (OpenAI の
    model_not_found 相当)。省略時は許容。一致してもしなくても、応答の
    "model" 欄には常に STATE.model_name (サーブ中の名前) を入れる — クライ
    アントが送った文字列をそのまま echo しない。"""

    requested = body.get("model")
    if requested is not None and requested != STATE.model_name:
        return _openai_error(
            f"The model `{requested}` does not exist or you do not have access to it.",
            status=404,
            err_type="invalid_request_error",
            code="model_not_found",
        )
    return None


def _check_model_anthropic(body: dict):
    requested = body.get("model")
    if requested is not None and requested != STATE.model_name:
        return _anthropic_error(
            f"model: {requested} not found", status=404, err_type="not_found_error"
        )
    return None


def _finish_reason_openai(tokens: list[int]) -> str:
    return "stop" if tokens and tokens[-1] in STATE.eos_ids else "length"


def _stop_reason_anthropic(tokens: list[int]) -> str:
    return "end_turn" if tokens and tokens[-1] in STATE.eos_ids else "max_tokens"


def _usage_dict(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> dict:
    """OpenAI 形式の usage。``prompt_tokens_details.cached_tokens`` は
    ChatSession の prefill 再利用実測 (``res["prefill_reused"]``) をそのまま
    載せる (mlx_lm:1339-1346 / 1567-1575 と同じ形)。0 のとき (再利用なし、
    または FallbackRunner のように prefill 再利用機構を持たない経路) も
    キー自体は出す — 「対応しているが今回は 0 件」と「対応していない」を
    レスポンス形状からは区別しない (mlx_lm も同様、prompt_cache_count が
    0 以上ならフィールドを出す)。
    """

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens >= 0:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return usage


def _log_gen_stats(res: dict) -> None:
    """prefill 再利用が効いているかは OpenAI/Anthropic のレスポンス形式には
    無い数値なので、cli.py の表示行と同じ内容を運用者向けに一行ログへ出す。"""

    print(
        f"[fastmlx-serve] prefill reused={res.get('prefill_reused', 0)} "
        f"new={res.get('prefill_new', 0)} decode={res.get('decode_tps', 0.0):.1f}tok/s"
    )


# ---------- 生成の下回り ----------
#
# on_tokens は投機の受理まとめて複数トークンを一度に渡してくる。生トークン
# 列は常に渡ってくる (SpecRunner はそれしか持たず、FallbackRunner も
# on_tokens(toks, text) の toks は必ず渡す) ので、ThinkingRouter はどちらの
# 経路でも同じロジックで reasoning/content を振り分けられる。thinking の
# 分離が絡む場合は FallbackRunner の precomputed text (二重デコード回避策)
# を使わず、ここで raw id から必ず再デトケナイズする — マーカー境界をまたぐ
# 判定に生トークンが要るのと、reasoning/content で detokenizer を分ける
# 必要があるため。
#
# STATE.runner.generate は同期・長時間実行なので、モデルをロードしたのと
# 同じ専用ワーカースレッド (STATE.executor) に必ず投げる。asyncio の汎用
# スレッドプール (asyncio.to_thread の既定 executor) や threading.Thread で
# 別スレッドに逃がすと、モデルの重み・KV キャッシュがロード時のスレッドに
# 紐づいているため "There is no Stream(gpu, N) in current thread" で落ちる
# (実測で確認済み: server.py の docstring 参照)。


async def _run_generate(
    prompt_ids, max_tokens, temp, eos_ids, on_tokens, session, **sampling_kwargs
):
    loop = asyncio.get_running_loop()
    fn = functools.partial(
        STATE.runner.generate,
        prompt_ids,
        max_tokens=max_tokens,
        temp=temp,
        eos_ids=eos_ids,
        on_tokens=on_tokens,
        session=session,
        **sampling_kwargs,
    )
    return await loop.run_in_executor(STATE.executor, fn)


def _chunk_string(s: str, size: int = 24) -> list[str]:
    """引数 JSON 文字列を固定長で分割する。OpenAI/Anthropic どちらの
    ストリーミング規約も「引数は分割して流れる」形が慣例だが、実際には
    tool call のテキスト全体をマーカーが閉じるまでバッファしてから初めて
    JSON として解釈できる (壊れていたら捏造しないため、途中経過のまま
    部分 JSON を流すことはできない)。そのため、パース確定後にこの関数で
    人為的に分割してイベント化する — 結合すれば必ず元の JSON 文字列に戻る
    ので、クライアント側の「断片を連結してから parse する」実装と矛盾しない。
    """

    if not s:
        return [""]
    return [s[i : i + size] for i in range(0, len(s), size)]


def _parse_tool_calls_text(raw: str, tools_for_parsing) -> list[dict] | None:
    """tool_call マーカーの間に挟まれていた生テキストを 1 個以上の構造化
    tool call へ変換する。``tokenizer.tool_parser`` (mlx_lm がチャット
    テンプレート文字列から自動選択するモデル固有パーサ、例えば Qwen の
    素朴な JSON 形式なら ``json.loads`` そのもの、Qwen3-Coder の
    ``<function=...>`` XML 風なら専用パーサ) をそのまま使う — マーカー
    文字列の検出は ThinkingRouter が既に済ませているので、ここでは中身の
    構文解析だけでよい。

    1 個のマーカー区間から複数の呼び出しが取れるパーサ (pythonic 等) にも
    対応するため、戻り値が list ならそのまま展開する (mlx_lm.server の
    ToolCallFormatter と同じ扱い)。

    解析に失敗した場合 (JSON 壊れ・name 欠落等) は None を返す — 呼び出し
    側はこれを「tool call を捏造せず、生テキストをそのまま content として
    返す」フォールバックのトリガーに使う。
    """

    parser = getattr(STATE.tokenizer, "tool_parser", None)
    if parser is None:
        return None
    try:
        parsed = parser(raw, tools_for_parsing)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        print(
            f"[fastmlx-serve] tool call の解析に失敗 ({type(exc).__name__}: {exc})。"
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
        tc_id = tc.get("id") or f"call_{uuid.uuid4().hex}"
        calls.append({"name": name, "arguments": arguments, "id": tc_id})
    return calls


class SegmentAssembler:
    """ThinkingRouter が返す (channel, payload) 列を、呼び出し側が使う
    高レベルイベント ``("reasoning_delta", text)`` / ``("content_delta",
    text)`` / ``("tool_call", {"id","name","arguments"})`` へ変換する。

    "tool"/"tool_start"/"tool_end" チャンネルはマーカー区間の生テキストを
    バッファし、区間が閉じた ("tool_end") 時点で初めて
    ``_parse_tool_calls_text`` に通す: 成功すれば tool_call イベントを
    (複数呼び出しなら複数個) 返し、失敗すれば "捏造せず、テキストとして
    そのまま返す" というサーバー全体の方針どおり、マーカー文字列を含めた
    生テキストを content_delta として返す (tool_end が実マーカーで閉じて
    いなければ終了マーカー文字列は付けない — max_tokens 等で打ち切られた
    ことが分かるように)。

    ストリーミング (on_tokens から都度 push する) ・非ストリーミング
    (生成後にまとめて push する) のどちらでも同じインスタンスをそのまま
    使い回せる — 状態は tool_buf (未確定の tool テキスト断片) だけ。
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
            calls = _parse_tool_calls_text(raw, self.tools_for_parsing)
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
    **sampling_kwargs,
):
    """ワーカーを STATE.executor へ投げ、(キュー, Future) を返す。

    キューに積まれる要素: ``("reasoning_delta", text)`` / ``("content_delta",
    text)`` / ``("tool_call", {"id","name","arguments"})`` (成功裏に解析
    できた tool call 1 個ぶん) / ``("budget_exceeded", None)`` (予算超過を
    検知した回だけ 1 回) / ``("done", res)`` / ``("error", exc)``。

    呼び出し側は必ずこの Future を最後まで待つこと (正常終了・エラー・
    クライアント切断のどの経路でも)。そうしないと、まだ実行中のワーカーが
    共有の STATE.session を触っている間に次のリクエストがロックを取れて
    しまい、同じ ChatSession を 2 つの生成が同時に書き換えるレースになる。
    """

    q: queue.Queue = queue.Queue()
    eos_ids = STATE.eos_ids
    router = ThinkingRouter(
        STATE.tokenizer, thinking_budget, eos_ids, tool_calling_enabled=tool_calling_enabled
    )
    assembler = SegmentAssembler(tools_for_parsing)
    signaled = [False]

    def on_tokens(toks, text=None):
        for channel, payload in router.feed(toks):
            for kind, val in assembler.push(channel, payload):
                q.put((kind, val))
        if router.budget_exceeded and not signaled[0]:
            q.put(("budget_exceeded", None))
            signaled[0] = True

    def worker():
        try:
            res = STATE.runner.generate(
                prompt_ids,
                max_tokens=max_tokens,
                temp=temp,
                eos_ids=eos_ids,
                on_tokens=on_tokens,
                session=STATE.session,
                **sampling_kwargs,
            )
            if not router.budget_exceeded:
                for channel, payload in router.finalize():
                    for kind, val in assembler.push(channel, payload):
                        q.put((kind, val))
            q.put(("done", res))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an SSE error event
            q.put(("error", exc))

    # 生成専用スレッド (executor) へ投げる。結果はキュー経由で非同期側が
    # asyncio.to_thread(q.get) で拾う (queue.Queue の待ち合わせだけなので
    # MLX のスレッド固定とは無関係、こちらは汎用スレッドプールで構わない)。
    future = STATE.executor.submit(worker)
    return q, future


async def _await_worker(future) -> None:
    """ワーカー Future を待つ。ワーカー内の例外は worker() 自身が
    q.put(("error", ...)) で拾って握り潰しているので、ここで raise される
    のは future 自体の生成/キャンセル絡みの想定外のみ。ジェネレータの
    finally から呼ぶので、ここでの例外がクリーンアップを壊さないよう飲む。"""

    try:
        await asyncio.wrap_future(future)
    except Exception:
        pass


def _collect_events(
    tokens: list[int],
    budget: int | None,
    tool_calling_enabled: bool,
    tools_for_parsing,
) -> tuple[list[tuple[str, object]], bool]:
    """非ストリーム向け: 最終トークン列をまとめて ThinkingRouter +
    SegmentAssembler に通し、順序を保った高レベルイベント列 (``feed()``/
    ``push()`` と同じ語彙: reasoning_delta/content_delta/tool_call) と
    budget_exceeded を返す。ストリーミング経路 (_start_generation) が
    on_tokens のたびに逐次行っているのと同じ変換を、非ストリームでは
    生成後に一括で行うだけ。"""

    router = ThinkingRouter(
        STATE.tokenizer, budget, STATE.eos_ids, tool_calling_enabled=tool_calling_enabled
    )
    parts = router.feed(tokens) + router.finalize()
    assembler = SegmentAssembler(tools_for_parsing)
    events: list[tuple[str, object]] = []
    for ch, payload in parts:
        events.extend(assembler.push(ch, payload))
    return events, router.budget_exceeded


def _split_response_final(
    tokens: list[int],
    budget: int | None,
    tool_calling_enabled: bool = False,
    tools_for_parsing=None,
) -> tuple[str, str, list[dict], bool]:
    """OpenAI 非ストリーム向け: (reasoning_text, content_text, tool_calls,
    budget_exceeded) を返す。tool_calls は成功裏に解析できた呼び出しだけ
    (解析に失敗したものは content_text 側にマーカーごと生テキストとして
    含まれる — 捏造しない方針)。"""

    events, budget_exceeded = _collect_events(
        tokens, budget, tool_calling_enabled, tools_for_parsing
    )
    reasoning_text = "".join(v for k, v in events if k == "reasoning_delta")
    content_text = "".join(v for k, v in events if k == "content_delta")
    tool_calls = [v for k, v in events if k == "tool_call"]
    return reasoning_text, content_text, tool_calls, budget_exceeded


def _truncate_content_events(
    events: list[tuple[str, object]], cut_pos: int
) -> list[tuple[str, object]]:
    """Anthropic 非ストリーム向け: stop_sequence が content_delta の連結
    文字列上の ``cut_pos`` で一致したとき、それ以降のイベントを切り捨てる。
    tool_calls が絡む場合はこの関数を呼ばない (呼び出し側の分岐で保証する
    — stop_sequence と tool calling の組み合わせは対応範囲外、既知の制限)
    ので、ここでは reasoning_delta/content_delta しか来ない前提でよい。"""

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
    """順序を保った高レベルイベント列を Anthropic の content ブロック列へ
    組み立てる。連続する content_delta は 1 個の text ブロックへ結合し、
    tool_call は独立した tool_use ブロックにする (Anthropic の実 API も
    tool_use を挟むと text ブロックが分かれる)。reasoning_delta はここでは
    扱わない (呼び出し側が既存どおり thinking ブロックを別途・常に先頭に
    組み立てる)。"""

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


# ---------- OpenAI 互換 ----------


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": STATE.model_name,
                "object": "model",
                "created": STATE.created_ts,
                "owned_by": "fastmlx",
            }
        ],
    }


@app.get("/health")
async def health():
    """mlx_lm の /health (``{"status": "ok"}`` だけ) より詳しく返す: モデル名・
    ロード済みかどうか・どちらの runner (投機 or 通常生成) か・リクエストを
    処理中かどうか。処理中かどうかは ``STATE.lock.locked()`` を見るだけ
    (直列化の設計上、ロックが空いていれば idle、取られていれば busy と等価)。
    """

    if STATE is None:
        return JSONResponse(
            status_code=503, content={"status": "loading", "loaded": False}
        )
    return {
        "status": "ok",
        "model": STATE.model_name,
        "loaded": True,
        "runner": getattr(STATE.runner, "KIND", type(STATE.runner).__name__),
        "busy": STATE.lock.locked(),
    }


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
    unsupported_err = _check_sampling_support(sampling_params)
    if unsupported_err is not None:
        return _openai_error(unsupported_err)
    stops = _stop_sequences(body)

    model_id = STATE.model_name
    stream = bool(body.get("stream", False))
    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if stream:
        stream_options = body.get("stream_options") or {}
        include_usage = bool(stream_options.get("include_usage", False))
        return StreamingResponse(
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
            ),
            media_type="text/event-stream",
        )

    try:
        async with STATE.lock:
            res = await _run_generate(
                prompt_ids,
                max_tokens,
                temp,
                STATE.eos_ids,
                None,
                STATE.session,
                **sampling_params,
            )
    except Exception as exc:
        return _openai_error(str(exc), status=500, err_type="server_error")
    _log_gen_stats(res)

    reasoning_text, content_text, tool_calls, budget_exceeded = _split_response_final(
        res["tokens"], thinking_budget, tool_enabled, resolved_tools
    )
    finish_reason = _finish_reason_openai(res["tokens"])
    if budget_exceeded:
        finish_reason = "length"
    elif tool_calls:
        finish_reason = "tool_calls"
    elif stops:
        hit = _find_stop(content_text, stops)
        if hit is not None:
            content_text = content_text[: hit[0]]
            finish_reason = "stop"

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

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage_dict(
            len(prompt_ids), len(res["tokens"]), res.get("prefill_reused", 0)
        ),
    }


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
):
    async with STATE.lock:
        q, future = _start_generation(
            prompt_ids,
            max_tokens,
            temp,
            thinking_budget,
            tool_enabled,
            tools_for_parsing,
            **(sampling_params or {}),
        )
        try:
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

            finish_reason = "length"
            n_completion = 0
            cached_tokens = 0
            acc_text = ""
            stopped = False  # stop 文字列に一致してからはクライアントへの
            # 転送だけ止め、実際の生成が終わる ("done") まではキューを
            # 空読みし続ける (正確な usage を得るのと、Future を待つ前提を
            # 崩さないため)。budget_exceeded も同様の「転送だけ止める」形。
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
                else:  # error
                    err = {"error": {"message": str(payload), "type": "server_error"}}
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
            # クライアント切断 (GeneratorExit) でここに来た場合も含め、
            # ワーカーが実際に終わるまでロックを離さない。
            await _await_worker(future)


# ---------- Anthropic 互換 ----------


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
        norm_messages = []
        system = body.get("system")
        if system:
            norm_messages.append({"role": "system", "content": _content_to_text(system)})
        norm_messages.extend(_normalize_anthropic_messages(messages))
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
    unsupported_err = _check_sampling_support(sampling_params)
    if unsupported_err is not None:
        return _anthropic_error(unsupported_err)
    stops = _stop_sequences(body)

    model_id = STATE.model_name
    stream = bool(body.get("stream", False))
    msg_id = f"msg_{uuid.uuid4().hex}"

    if max_tokens == 0:
        # Anthropic の実 API に合わせる: 0 は「生成せずに終わる」有効な入力
        # だが、ストリームでは意味を成さないので 400 にする。
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

    if stream:
        return StreamingResponse(
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
            ),
            media_type="text/event-stream",
        )

    try:
        async with STATE.lock:
            res = await _run_generate(
                prompt_ids,
                max_tokens,
                temp,
                STATE.eos_ids,
                None,
                STATE.session,
                **sampling_params,
            )
    except Exception as exc:
        return _anthropic_error(str(exc), status=500, err_type="server_error")
    _log_gen_stats(res)

    events, budget_exceeded = _collect_events(
        res["tokens"], thinking_budget, tool_enabled, resolved_tools
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
        # tool_calls が絡む場合の stop_sequence 対応は既知の制限として
        # 見送っている (elif で never reached) — 詳細は
        # _truncate_content_events の docstring を参照。
        content_text_concat = "".join(v for k, v in events if k == "content_delta")
        hit = _find_stop(content_text_concat, stops)
        if hit is not None:
            events = _truncate_content_events(events, hit[0])
            stop_reason = "stop_sequence"
            matched_stop = hit[1]

    content_blocks = []
    if reasoning_text:
        content_blocks.append({"type": "thinking", "thinking": reasoning_text})
    content_blocks.extend(_anthropic_blocks_from_events(events))
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": matched_stop,
        "usage": {
            "input_tokens": len(prompt_ids),
            "output_tokens": len(res["tokens"]),
        },
    }


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
):
    def sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async with STATE.lock:
        q, future = _start_generation(
            prompt_ids,
            max_tokens,
            temp,
            thinking_budget,
            tool_enabled,
            tools_for_parsing,
            **(sampling_params or {}),
        )
        try:
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
            # content_block_start は最初のデルタが来てから、その channel
            # (thinking が起きたかどうか) に応じて出す。事前には分からない
            # ため (実 API も、最初のブロックを送る直前に content_block_start
            # を出す形なので、順序としては同じ)。

            n_out = 0
            stop_reason = "end_turn"
            matched_stop = None
            acc_text = ""
            stopped = False
            budget_exceeded = False
            made_tool_call = False
            current_block: str | None = None  # None | "reasoning" | "content"
            any_block_emitted = False  # current_block だけだと「tool_use を
            # 出し終えて None に戻した」のと「まだ何も出していない」を
            # 区別できないので、別フラグで持つ。
            next_index = 0
            block_index: dict[str, int] = {}

            def open_block(channel: str):
                nonlocal next_index
                idx = next_index
                next_index += 1
                block_index[channel] = idx
                if channel == "reasoning":
                    content_block = {"type": "thinking", "thinking": ""}
                else:
                    content_block = {"type": "text", "text": ""}
                return idx, sse(
                    "content_block_start",
                    {"type": "content_block_start", "index": idx, "content_block": content_block},
                )

            while True:
                kind, payload = await asyncio.to_thread(q.get)
                if kind == "tool_call":
                    if stopped:
                        continue
                    made_tool_call = True
                    any_block_emitted = True
                    if current_block is not None:
                        yield sse(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": block_index[current_block]},
                        )
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
                        acc_text = new_acc
                    if not visible:
                        continue
                    if current_block != channel:
                        if current_block is not None:
                            yield sse(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": block_index[current_block]},
                            )
                        idx, start_evt = open_block(channel)
                        yield start_evt
                        current_block = channel
                        any_block_emitted = True
                    delta_field = (
                        {"type": "thinking_delta", "thinking": visible}
                        if channel == "reasoning"
                        else {"type": "text_delta", "text": visible}
                    )
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
                else:  # error
                    yield sse(
                        "error",
                        {"type": "error", "error": {"type": "server_error", "message": str(payload)}},
                    )
                    return

            if current_block is not None:
                yield sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": block_index[current_block]},
                )
            elif not any_block_emitted:
                # 何も生成されなかった (例: max_tokens が極端に小さい)。
                # 「content_block が最低 1 つはある」前提を壊さないよう、
                # 空の text ブロックを開いてすぐ閉じる。tool_use ブロックを
                # 1 個以上出し終えて current_block が None に戻っている
                # だけの場合はここには来ない (any_block_emitted で判別)。
                idx, start_evt = open_block("content")
                yield start_evt
                yield sse("content_block_stop", {"type": "content_block_stop", "index": idx})

            yield sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": matched_stop},
                    "usage": {"output_tokens": n_out},
                },
            )
            yield sse("message_stop", {"type": "message_stop"})
        finally:
            await _await_worker(future)


# ---------- OpenAI 互換 (legacy /v1/completions) ----------
#
# chat template を通さず、渡された prompt をそのまま生成に流すレガシー
# エンドポイント。thinking の分離は行わない (raw completion にターン構造が
# 無いので reasoning_effort の意味を持たせようがない) — thinking_budget=0 を
# _start_generation/_split_response_final に渡すことで ThinkingRouter を
# 強制的に content-only モードにし (budget=0 は has_thinking の値に関わらず
# enabled=False)、既存の on_tokens 配線をそのまま再利用する。tool calling
# も同様に有効化しない (tool_calling_enabled は既定 False のまま渡さない)。


def _prompt_to_ids(prompt) -> tuple[list[int] | None, str | None]:
    """``prompt`` は文字列 (tokenizer.encode に通す) か、事前トークナイズ済み
    の int 配列のどちらかを受け付ける。OpenAI の legacy completions は
    文字列配列やトークン配列の配列 (バッチ) も許すが、fastmlx はリクエストを
    直列化する設計 (1 リクエスト = 1 生成) なのでバッチは扱わない。
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

    max_tokens, err = _resolve_max_tokens_openai(body, STATE.max_tokens_cap)
    if err is not None:
        return _openai_error(err)
    temp, err = _parse_temperature(body, STATE.default_temp)
    if err is not None:
        return _openai_error(err)
    sampling_params, err = _parse_sampling_params(body)
    if err is not None:
        return _openai_error(err)
    unsupported_err = _check_sampling_support(sampling_params)
    if unsupported_err is not None:
        return _openai_error(unsupported_err)
    stops = _stop_sequences(body)

    model_id = STATE.model_name
    stream = bool(body.get("stream", False))
    req_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if stream:
        stream_options = body.get("stream_options") or {}
        include_usage = bool(stream_options.get("include_usage", False))
        return StreamingResponse(
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
            ),
            media_type="text/event-stream",
        )

    try:
        async with STATE.lock:
            res = await _run_generate(
                prompt_ids,
                max_tokens,
                temp,
                STATE.eos_ids,
                None,
                STATE.session,
                **sampling_params,
            )
    except Exception as exc:
        return _openai_error(str(exc), status=500, err_type="server_error")
    _log_gen_stats(res)

    _reasoning_text, text, _tool_calls, _budget_exceeded = _split_response_final(res["tokens"], 0)
    finish_reason = _finish_reason_openai(res["tokens"])
    if stops:
        hit = _find_stop(text, stops)
        if hit is not None:
            text = text[: hit[0]]
            finish_reason = "stop"

    return {
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage_dict(
            len(prompt_ids), len(res["tokens"]), res.get("prefill_reused", 0)
        ),
    }


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
):
    async with STATE.lock:
        # thinking_budget=0: ThinkingRouter を content-only に固定する
        # (has_thinking に関わらず) ので reasoning_delta は絶対に来ない。
        q, future = _start_generation(
            prompt_ids, max_tokens, temp, 0, **(sampling_params or {})
        )
        try:
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
                    # thinking_budget=0・tool_calling_enabled=False (既定)
                    # なのでここには来ないはずだが、念のため無視するだけに
                    # しておく (クラッシュより安全)。
                    continue
                elif kind == "done":
                    finish_reason = "stop" if stopped else _finish_reason_openai(payload["tokens"])
                    n_completion = len(payload["tokens"])
                    cached_tokens = payload.get("prefill_reused", 0)
                    _log_gen_stats(payload)
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
            await _await_worker(future)


# ---------- 起動 ----------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--original", default="Qwen/Qwen3.8-27B")
    ap.add_argument(
        "--served-model-name",
        default=None,
        help="GET /v1/models とレスポンスの \"model\" 欄で名乗る id。既定は --model の"
        " basename (例: /Users/ht/models/qwen38fn-mlx-v-fast6 なら qwen38fn-mlx-v-fast6)。"
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
        "--no-mtp", action="store_true", help="MTP を読み込まず lookup (SAM) のみで投機する"
    )
    ap.add_argument(
        "--ngram",
        default=None,
        help="n-gram (PLE) 表を外部サイドカーへ分離してある変換の場合、そのディレクトリを指定する"
        " (fastmlx/ngram_stream.py)。cli.py の --ngram と同じ",
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
    args = ap.parse_args()

    global STATE

    served_name = args.served_model_name or Path(args.model).name

    # モデルの重み・KV キャッシュはロードしたスレッドに紐づく (docstring
    # 参照)。以後すべての生成呼び出しもこのスレッドに固定するので、ロード
    # 自体もここで行う。max_workers=1 でスレッドは 1 本だけ、プロセス生涯
    # 使い回す。
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="fastmlx-mlx"
    )

    def _load():
        t0 = time.perf_counter()
        if args.ngram:
            # qwen4_exp.py の NGRAM_ON_DISK はモジュール import 時に評価される
            # ので、読み込み呼び出しより前に立てておく必要がある (cli.py と
            # 同じ理由)。
            os.environ["FASTMLX_NGRAM_DISK"] = "1"
        model, tokenizer, config = mlx_lm_load(args.model, return_config=True)
        if args.ngram:
            from .ngram_stream import install

            install(model, args.ngram)
        print(f"[fastmlx-serve] loaded in {time.perf_counter() - t0:.1f}s: {args.model}")
        runner = build_runner(model, tokenizer, config, args, log_prefix="[fastmlx-serve]")
        return runner, tokenizer

    runner, tokenizer = executor.submit(_load).result()

    STATE = ModelState(
        runner=runner,
        tokenizer=tokenizer,
        session=ChatSession(),
        lock=asyncio.Lock(),
        executor=executor,
        model_name=served_name,
        eos_ids=set(tokenizer.eos_token_ids),
        max_tokens_cap=args.max_tokens,
        default_temp=args.temp,
        created_ts=int(time.time()),
    )
    print(f"[fastmlx-serve] served model name: {served_name}")
    if getattr(tokenizer, "has_thinking", False):
        print(
            f"[fastmlx-serve] thinking マーカー検出: {tokenizer.think_start!r} / "
            f"{tokenizer.think_end!r} (reasoning_content/thinking ブロック分離が有効)"
        )
    else:
        print("[fastmlx-serve] thinking マーカー検出なし (このモデルでは分離しない)")
    if getattr(tokenizer, "has_tool_calling", False):
        print(
            f"[fastmlx-serve] tool call マーカー検出: {tokenizer.tool_call_start!r} / "
            f"{tokenizer.tool_call_end!r} (tools/tool_choice に対応)"
        )
    else:
        print(
            "[fastmlx-serve] tool call マーカー検出なし (このモデルは tool calling 非対応: "
            "tools を渡すリクエストは 400 になる)"
        )

    if args.allowed_origins:
        origins = [o.strip() for o in args.allowed_origins.split(",") if o.strip()]
        _add_cors_middleware(app, origins)
        print(f"[fastmlx-serve] CORS 許可 origin: {', '.join(origins)}")
    else:
        print("[fastmlx-serve] CORS 無効 (既定、ローカル専用)")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
