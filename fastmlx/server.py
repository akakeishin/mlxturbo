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
from fastapi.responses import JSONResponse, StreamingResponse

from ._mlx_compat import mlx_lm_load
from .runner import Runner, build_runner
from .spec import ChatSession

app = FastAPI()


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
    default_enable_thinking: bool | None
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


def _naive_prompt_ids(messages: list[dict]):
    """tokenizer にチャットテンプレートが無いモデル向けの素朴なフォールバック。

    role: content を並べて素の文字列にするだけ。Qwen 系限定の前提を持ち
    込まない (将来 gemma/kimi/glm を載せても、チャットテンプレートさえ
    あればそちらを使い、無ければここに落ちる)。
    """

    lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    lines.append("assistant:")
    return STATE.tokenizer.encode("\n".join(lines))


def _apply_template(messages: list[dict], enable_thinking: bool | None = None):
    """``enable_thinking`` is an optional passthrough for reasoning-model chat
    templates (Qwen3 系など)。省略時はテンプレートの既定 (通常 thinking 有効)
    に任せる。cli.py の --no-think と同じキーワードで、対応しないテンプレート
    では単に無視される (下の TypeError フォールバック)。

    thinking が有効なテンプレートの一部は、確定した過去ターンの assistant
    メッセージを履歴へ組み込む際に <think>...</think> を落とす (再度読ませる
    必要がないため)。そのため実際に生成されたトークン列より次ターンの
    prompt が短くなり、ChatSession の LCP 再利用 (前回処理列がそのまま新
    prompt の接頭辞であることを要求する) が原理的に成立しなくなる。
    enable_thinking=False を明示すればこの型のモデルでも再利用の対象に戻る。
    """

    kwargs = {"add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
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


def _effective_enable_thinking(body: dict) -> bool | None:
    """リクエストが ``enable_thinking`` を明示すればそれを最優先。省略時は
    ``--no-think`` で決めたサーバー既定 (未指定ならテンプレート任せの None)
    に落ちる。OpenAI/Anthropic 系クライアントの多くはこのフィールドを送って
    こないので、既定を運用者側で選べるようにしておく必要がある。"""

    requested = body.get("enable_thinking")
    if requested is not None:
        return requested
    return STATE.default_enable_thinking


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


def _log_gen_stats(res: dict) -> None:
    """prefill 再利用が効いているかは OpenAI/Anthropic のレスポンス形式には
    無い数値なので、cli.py の表示行と同じ内容を運用者向けに一行ログへ出す。"""

    print(
        f"[fastmlx-serve] prefill reused={res.get('prefill_reused', 0)} "
        f"new={res.get('prefill_new', 0)} decode={res.get('decode_tps', 0.0):.1f}tok/s"
    )


# ---------- 生成の下回り ----------
#
# on_tokens は投機の受理まとめて複数トークンを一度に渡してくる。SpecRunner
# (SpecEngine) はテキストを持たず生トークンしか渡してこないので、その場合は
# ここの detok がトークンごとに分解し直さず detokenizer.last_segment を
# そのまま 1 個の delta として流す。FallbackRunner (mlx_lm.stream_generate)
# は自分の内部 detokenizer で resp.text をすでに正しく計算済みなので、それを
# 渡してもらったときはここでの再デトケナイズをスキップする (二重デコード
# だと BPE の trailing-space マージ等が二重に走り、先頭空白の扱いが崩れうる)。
#
# STATE.runner.generate は同期・長時間実行なので、モデルをロードしたのと
# 同じ専用ワーカースレッド (STATE.executor) に必ず投げる。asyncio の汎用
# スレッドプール (asyncio.to_thread の既定 executor) や threading.Thread で
# 別スレッドに逃がすと、モデルの重み・KV キャッシュがロード時のスレッドに
# 紐づいているため "There is no Stream(gpu, N) in current thread" で落ちる
# (実測で確認済み: server.py の docstring 参照)。


async def _run_generate(prompt_ids, max_tokens, temp, eos_ids, on_tokens, session):
    loop = asyncio.get_running_loop()
    fn = functools.partial(
        STATE.runner.generate,
        prompt_ids,
        max_tokens=max_tokens,
        temp=temp,
        eos_ids=eos_ids,
        on_tokens=on_tokens,
        session=session,
    )
    return await loop.run_in_executor(STATE.executor, fn)


def _start_generation(prompt_ids, max_tokens: int, temp: float):
    """ワーカーを STATE.executor へ投げ、(キュー, Future) を返す。

    呼び出し側は必ずこの Future を最後まで待つこと (正常終了・エラー・
    クライアント切断のどの経路でも)。そうしないと、まだ実行中のワーカーが
    共有の STATE.session を触っている間に次のリクエストがロックを取れて
    しまい、同じ ChatSession を 2 つの生成が同時に書き換えるレースになる。
    """

    q: queue.Queue = queue.Queue()
    detok = STATE.tokenizer.detokenizer
    detok.reset()
    eos_ids = STATE.eos_ids

    def on_tokens(toks, text=None):
        if text is not None:
            if text:
                q.put(("delta", text))
            return
        for t in toks:
            if t in eos_ids:
                continue
            detok.add_token(t)
        seg = detok.last_segment
        if seg:
            q.put(("delta", seg))

    def worker():
        try:
            res = STATE.runner.generate(
                prompt_ids,
                max_tokens=max_tokens,
                temp=temp,
                eos_ids=eos_ids,
                on_tokens=on_tokens,
                session=STATE.session,
            )
            detok.finalize()
            seg = detok.last_segment
            if seg:
                q.put(("delta", seg))
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
        norm_messages = [
            {"role": m.get("role"), "content": _content_to_text(m.get("content"))}
            for m in messages
        ]
    except ContentNormalizationError as exc:
        return _openai_error(str(exc))
    try:
        prompt_ids = _apply_template(norm_messages, _effective_enable_thinking(body))
    except Exception as exc:
        return _openai_error(f"failed to render chat template: {exc}")

    max_tokens, err = _resolve_max_tokens_openai(body, STATE.max_tokens_cap)
    if err is not None:
        return _openai_error(err)
    temp, err = _parse_temperature(body, STATE.default_temp)
    if err is not None:
        return _openai_error(err)
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
                prompt_ids, max_tokens, temp, req_id, created, model_id, stops, include_usage
            ),
            media_type="text/event-stream",
        )

    try:
        async with STATE.lock:
            res = await _run_generate(
                prompt_ids, max_tokens, temp, STATE.eos_ids, None, STATE.session
            )
    except Exception as exc:
        return _openai_error(str(exc), status=500, err_type="server_error")
    _log_gen_stats(res)

    text = STATE.tokenizer.decode([t for t in res["tokens"] if t not in STATE.eos_ids])
    finish_reason = _finish_reason_openai(res["tokens"])
    if stops:
        hit = _find_stop(text, stops)
        if hit is not None:
            text = text[: hit[0]]
            finish_reason = "stop"

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(res["tokens"]),
            "total_tokens": len(prompt_ids) + len(res["tokens"]),
        },
    }


async def _openai_stream(
    prompt_ids, max_tokens, temp, req_id, created, model_id, stops, include_usage
):
    async with STATE.lock:
        q, future = _start_generation(prompt_ids, max_tokens, temp)
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
            acc_text = ""
            stopped = False  # stop 文字列に一致してからはクライアントへの
            # 転送だけ止め、実際の生成が終わる ("done") まではキューを
            # 空読みし続ける (正確な usage を得るのと、Future を待つ前提を
            # 崩さないため)。
            while True:
                kind, payload = await asyncio.to_thread(q.get)
                if kind == "delta":
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
                elif kind == "done":
                    finish_reason = "stop" if stopped else _finish_reason_openai(payload["tokens"])
                    n_completion = len(payload["tokens"])
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
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": n_completion,
                        "total_tokens": len(prompt_ids) + n_completion,
                    },
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
        for m in messages:
            norm_messages.append(
                {"role": m.get("role"), "content": _content_to_text(m.get("content"))}
            )
    except ContentNormalizationError as exc:
        return _anthropic_error(str(exc))

    try:
        prompt_ids = _apply_template(norm_messages, _effective_enable_thinking(body))
    except Exception as exc:
        return _anthropic_error(f"failed to render chat template: {exc}")

    max_tokens, err = _parse_anthropic_max_tokens(body["max_tokens"], STATE.max_tokens_cap)
    if err is not None:
        return _anthropic_error(err)
    temp, err = _parse_temperature(body, STATE.default_temp)
    if err is not None:
        return _anthropic_error(err)
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
            _anthropic_stream(prompt_ids, max_tokens, temp, msg_id, model_id, stops),
            media_type="text/event-stream",
        )

    try:
        async with STATE.lock:
            res = await _run_generate(
                prompt_ids, max_tokens, temp, STATE.eos_ids, None, STATE.session
            )
    except Exception as exc:
        return _anthropic_error(str(exc), status=500, err_type="server_error")
    _log_gen_stats(res)

    text = STATE.tokenizer.decode([t for t in res["tokens"] if t not in STATE.eos_ids])
    stop_reason = _stop_reason_anthropic(res["tokens"])
    matched_stop = None
    if stops:
        hit = _find_stop(text, stops)
        if hit is not None:
            text = text[: hit[0]]
            stop_reason = "stop_sequence"
            matched_stop = hit[1]

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": matched_stop,
        "usage": {
            "input_tokens": len(prompt_ids),
            "output_tokens": len(res["tokens"]),
        },
    }


async def _anthropic_stream(prompt_ids, max_tokens, temp, msg_id, model_id, stops):
    def sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async with STATE.lock:
        q, future = _start_generation(prompt_ids, max_tokens, temp)
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
            yield sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            yield sse("ping", {"type": "ping"})

            n_out = 0
            stop_reason = "end_turn"
            matched_stop = None
            acc_text = ""
            stopped = False
            while True:
                kind, payload = await asyncio.to_thread(q.get)
                if kind == "delta":
                    if stopped:
                        continue
                    visible = payload
                    if stops:
                        new_acc = acc_text + payload
                        hit = _find_stop(new_acc, stops)
                        if hit is not None:
                            idx, matched = hit
                            keep_len = idx - len(acc_text)
                            visible = payload[:keep_len] if keep_len > 0 else ""
                            stopped = True
                            matched_stop = matched
                        acc_text = new_acc
                    if visible:
                        yield sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": visible},
                            },
                        )
                elif kind == "done":
                    n_out = len(payload["tokens"])
                    stop_reason = (
                        "stop_sequence" if stopped else _stop_reason_anthropic(payload["tokens"])
                    )
                    _log_gen_stats(payload)
                    break
                else:  # error
                    yield sse(
                        "error",
                        {
                            "type": "error",
                            "error": {"type": "server_error", "message": str(payload)},
                        },
                    )
                    return

            yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})
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
        "--no-think",
        action="store_true",
        help="リクエストが enable_thinking を明示しない場合の既定を thinking 無効にする"
        " (cli.py の --no-think と同じ)。リクエスト側の enable_thinking が優先される",
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
        default_enable_thinking=False if args.no_think else None,
        created_ts=int(time.time()),
    )
    print(f"[fastmlx-serve] served model name: {served_name}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
