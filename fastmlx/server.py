"""OpenAI 互換 / Anthropic 互換 HTTP サーバー。

既存の投機デコードエンジン (fastmlx.spec.SpecEngine) をそのまま使う。モデルは
起動時に 1 回だけロードして常駐させ、リクエストはグローバルなロックで直列化
する: 91GB 級のモデルを 128GB 機に載せている前提なので、同時実行 (並列
バッチング) はまだしない (BatchGenerator ベースのスケジューラは次段)。待機中
のリクエストもコネクションは保ったまま (asyncio のロック待ちで単に await
するだけ)、ロックが空くまで待たせる。

会話履歴はクライアントが毎ターン全文を送り直す (OpenAI/Anthropic どちらの
API も無状態) ので、session (SpecEngine 経路なら fastmlx.spec.ChatSession、
FallbackRunner 経路なら fastmlx.runner.FallbackSession) はリクエスト単位で
所有権を持つ ``STATE.session_pool`` (会話ごとの LRU プール、``_select_session``
参照) から都度引き当てる。新しいプロンプトの先頭が既存スロットの処理済み
トークン列全体と一致すれば (= 純粋な追記なら) そのスロットを引き当てて
prefill を再利用する。一致しなければ新規スロットを割り当てる (プールが
上限に達していれば最も長く使われていないスロットを ``popitem(last=False)``
で捨てる)。91GB 級モデルの上に会話ごとの KV を無制限に積み上げないための
上限であり、フィールド名は違うが SpecEngine/FallbackRunner どちらの経路でも
同じプールを使う (``STATE.session_factory`` が起動時にどちらのクラスを
積むか決める)。

直列化ロック (``STATE.lock``) は今回はまだ外していない — 現状は「1 リクエスト
= 1 生成」を保証する唯一の仕組みであり続けるが、session の所有権がリクエスト
単位になったことで、ロックさえ外せば異なる会話を同時に生成しても互いの
session を破壊しなくなる (ロックを外すこと自体は次のスケジューラの回でやる)。
``STATE.session_pool`` 自体への並行アクセス (2 リクエストが同時にスロットを
選ぶ・追い出す) はまだこのロックに守られている前提のままなので、ロックを
外す際はプール操作自体にも別途排他が要る。

SpecEngine が受け付けないモデル (Llama/Gemma/dense Qwen や、GDN ハイブリッド
でもレイアウトが異なるもの) では、起動時に fastmlx.runner.build_runner が
mlx_lm.generate.stream_generate による普通の (非投機) 生成に自動でフォール
バックする。どちらの経路でも HTTP 層から見た形は同一 (Runner.generate が
同じ dict を返す) で、起動時にどちらの経路が有効かを一行ログで出す。
FallbackRunner も (SpecEngine と同じ LCP 契約で) mlx_lm の prompt_cache を
session 経由で再利用する。詳細は fastmlx/runner.py を参照。

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
このリクエストへ引き当てた session を触っている間に次のリクエストがロックを
取れてしまい、同じ session を 2 つの生成が同時に書き換える。

ストリーミング応答は、検証 (400 になりうるチェック全て) をエンドポイント
関数内で StreamingResponse を組み立てる前に済ませたあと、生成の開始 (ロック
獲得・ワーカー投入) を待たずに最初の SSE イベント (OpenAI なら role delta の
チャンク、Anthropic なら message_start) を即座に流す — TTFT (prefill) が
支配項のワークロードでは、クライアントが最初の 1 バイトを受け取るまでの
時間そのものが切断の実因になるため。200 を返してから 400 相当が判明する
経路にはなっていない (判明し得るチェックは全て StreamingResponse を返す前
に完了している)。

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
import hashlib
import hmac
import itertools
import json
import os
import queue
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import mlx.core as mx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ._mlx_compat import mlx_lm_load
from .runner import FallbackSession, Runner, build_runner
from .spec import PREFILL_STEP_SIZE, ChatSession

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
    # 会話ごとの session (ChatSession/FallbackSession) の LRU プール。キーは
    # 会話の身元そのものではなく単なる挿入順管理用の連番 (_select_session
    # 参照) — 選択はキーではなく各スロットの processed 列との LCP で行う。
    session_pool: "OrderedDict[int, object]"
    # runner の種類 (SpecRunner なら ChatSession、FallbackRunner なら
    # FallbackSession) に応じて起動時に決める、新規スロット用のファクトリ。
    session_factory: Callable[[], object]
    lock: asyncio.Lock
    executor: concurrent.futures.ThreadPoolExecutor
    model_name: str  # served id: GET /v1/models と全レスポンスの "model" 欄
    eos_ids: set
    max_tokens_cap: int
    default_temp: float
    created_ts: int
    max_sessions: int = 8  # session_pool の上限 (LRU)。91GB 級モデルの上に
    # 会話ごとの KV を無制限に積まないための上限で、--max-sessions で変えられる。
    # プロンプト長 (トークン数) の上限。既定はモデル config の
    # max_position_embeddings と、Metal が一括確保できる実際の上限から
    # 逆算した値の小さい方を自動で取る (_resolve_default_max_context_tokens)、
    # --max-context-tokens で上書き可。None なら未検出でガード無効。
    # _check_context_length が全 4 経路 (chat/anthropic/completions/responses)
    # で参照する。
    max_context_tokens: int | None = None
    session_key_seq: Iterator[int] = field(default_factory=lambda: itertools.count())


STATE: ModelState | None = None

_THINKING_SIGNATURE_PREFIX = "fastmlx_v1:"
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
    """``sess`` (ChatSession/FallbackSession) の KV キャッシュを ``n_trim``
    トークン分だけ巻き戻せるか確認し、できれば実際に巻き戻す。

    個々のキャッシュ実装 (mlx_lm.models.cache の ``KVCache`` は巻き戻せる。
    GDN ハイブリッドの線形層に使う ``ArraysCache`` は再帰状態を途中位置へ
    戻す手段が無く巻き戻せない — fastmlx/spec.py の ChatSession docstring、
    fastmlx/runner.py の FallbackSession docstring 参照) をこの関数自身は
    一切知らない。判定・実行はどちらも mlx_lm.models.cache の
    ``can_trim_prompt_cache``/``trim_prompt_cache`` にそのまま委ねる —
    構成する全レイヤーが ``is_trimmable()`` を申告したときだけ実際に trim
    し、1 レイヤーでも無理なら何もせず False を返す (半端に書き換えて
    壊れた状態を残さない)。

    ``mtp_valid`` (ChatSession のみ) が立っている session は対象外にする:
    MTP 連鎖用の ``h_last``/``mtp_cache`` はセッションの最終位置 1 点分の
    状態しか持たず、途中位置まで KV を巻き戻しても対応する h_last が
    無いため、巻き戻した位置での MTP 継続を保証できない。呼び出し側が
    ``mtp_valid`` を立てて渡してくる可能性がある構成 (GDN ハイブリッド +
    MTP、このサーバーの現行構成そのもの) では、そもそも ``ArraysCache``
    が混ざっているため ``can_trim_prompt_cache`` が False を返して自然に
    弾かれるはずだが、それに依存せずここでも明示的に弾く (二重の安全策)。
    """

    if getattr(sess, "mtp_valid", False):
        return False
    cache_list = getattr(sess, "caches", None)
    if cache_list is None:
        cache_list = getattr(sess, "cache", None)
    # 実際の ChatSession.caches/FallbackSession.cache は常に
    # mlx_lm.models.cache._BaseCache のインスタンスのリストだが、この
    # 関数は runner の種類を知らない (テストの軽量フェイクは cache を
    # 中身の無い sentinel オブジェクトにすることがある) ので、その形に
    # なっていなければ trim できないものとして安全側に倒す。
    if not isinstance(cache_list, list) or not cache_list:
        return False
    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

    if not can_trim_prompt_cache(cache_list):
        return False
    trimmed = trim_prompt_cache(cache_list, n_trim)
    return trimmed == n_trim


def _select_session(prompt_ids: list[int]):
    """新しいプロンプトに対して使う session (ChatSession/FallbackSession) を
    ``STATE.session_pool`` から選ぶ。

    まず既存の安全な経路を最優先で試す: 「既存スロットの処理済みトークン列
    全体が新プロンプトの接頭辞になっている (= 純粋な追記)」スロットが
    あれば、キャッシュには一切触れずそのまま使う。会話の身元を messages
    の内容から推測する (システムプロンプトのハッシュ等) のではなく、
    SpecEngine の ChatSession が単一セッションで既にやっている LCP 判定
    (fastmlx/spec.py) をプール全体に広げただけなので、動的に変化しうる
    システムプロンプト (現在時刻の埋め込み等) が混ざっていても「一致
    しなければ次点へ」に自然に倒れる。一致するスロットが複数ありうる
    状況では最長一致を優先する。

    全体一致が無ければ、部分一致 (処理済み列の一部だけが新プロンプトの
    接頭辞) の中で LCP が長い順に、KV キャッシュをその LCP まで実際に
    巻き戻せるか試す (llama.cpp と同様、最長共通接頭辞まで再利用してから
    trim する発想)。巻き戻し可否の判定は ``_try_trim_session_cache`` に
    委ねており、巻き戻せた場合だけそのスロットを使い、``processed`` を
    LCP 長へ切り詰める。1 つも巻き戻せなければ (GDN ハイブリッド層を含む
    構成では常にこうなる — 上記 docstring 参照) 部分一致は諦め、従来
    どおり新規スロットへ倒す。誤った/半端な状態のスロットを掴んで壊れる
    ことは無い。

    候補が 1 つも無ければ、プールが上限未満なら新規スロットを、上限なら
    最も長く未使用のスロット (先頭 = LRU) を追い出して新規スロットを
    割り当てる。

    呼び出し側は ``STATE.lock`` を保持した状態でこれを呼ぶこと — プール自体
    への読み書きはこの関数もその後の session.publish/invalidate も排他して
    いない (直列化ロックに守られている前提)。ロックを外す (次のスケジューラ
    の回) ときは、ここにも別途排他が要る。
    """

    pool = STATE.session_pool

    def _lcp(pl: list[int]) -> int:
        n = min(len(pl), len(prompt_ids))
        i = 0
        while i < n and pl[i] == prompt_ids[i]:
            i += 1
        return i

    # 1st pass: 全体一致 (純粋な追記) — キャッシュに触れない、既存の安全経路。
    best_key = None
    best_lcp = -1
    for key, sess in pool.items():
        pl = sess.processed
        if not pl:
            continue
        lcp = _lcp(pl)
        if lcp == len(pl) and lcp < len(prompt_ids) and lcp > best_lcp:
            best_lcp = lcp
            best_key = key

    if best_key is not None:
        pool.move_to_end(best_key)
        return pool[best_key]

    # 2nd pass: 部分一致。LCP が長い候補から順に、実際に巻き戻せるスロット
    # が見つかるまで試す。
    partial = []
    for key, sess in pool.items():
        pl = sess.processed
        if not pl:
            continue
        lcp = _lcp(pl)
        if 0 < lcp < len(pl) and lcp < len(prompt_ids):
            partial.append((lcp, key, sess))
    partial.sort(key=lambda c: -c[0])

    for lcp, key, sess in partial:
        if _try_trim_session_cache(sess, len(sess.processed) - lcp):
            sess.processed = sess.processed[:lcp]
            if hasattr(sess, "mtp_cache"):
                sess.mtp_cache = None
            if hasattr(sess, "h_last"):
                sess.h_last = None
            pool.move_to_end(key)
            return sess

    if len(pool) >= STATE.max_sessions:
        pool.popitem(last=False)
    key = next(STATE.session_key_seq)
    session = STATE.session_factory()
    pool[key] = session
    return session


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
            elif btype in ("thinking", "redacted_thinking"):
                # 拡張思考 + tool use の Anthropic 規約: 直前ターンの
                # thinking/redacted_thinking ブロックは次ターンの履歴に
                # そのまま含めて送り返す必要がある (公式ドキュメントで
                # 明記)。このサーバー自身が thinking 有効時にこのブロックを
                # 返している (このモジュール内で "thinking" ブロックを組み
                # 立てている箇所を参照) ので、tool を使う複数ターンの会話を
                # 送り返すクライアント (例: Claude Code) は確実にこの型を
                # 履歴へ含めてくる。fastmlx 自身が発行した thinking signature
                # は改変されていないことを確認する (他 provider の opaque
                # signature は解釈できないので、そのまま受け付ける)。
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
                # 中身をモデルに再度読ませる必要は無い
                # (thinking 対応チャットテンプレートは確定した過去ターンの
                # think 内容を履歴から落とす想定 — _apply_template の
                # docstring 参照) ので、ここでは 400 にせず読み飛ばすだけ
                # にする。
                pass
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
    "xhigh": 65536,
    "max": 131072,
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


def _prompt_already_thinking(prompt_ids: list[int]) -> bool:
    """``_apply_template`` が描画したプロンプトが、未閉鎖の thinking ブロック
    で終わっているか (= チャットテンプレート自身が生成プロンプトの末尾で
    既に ``<think>`` を開いているか) を判定する。

    実モデル (Qwen3.6 / Flash-Next) で確認済みのバグ: これらのテンプレートは
    ``<|im_start|>assistant\\n<think>\\n`` のように、モデルへ渡す時点で既に
    ``<think>`` を開いた状態でプロンプトを終える。モデル自身は (開き済みの
    マーカーをもう一度は出さないので) ``<think>`` を生成しない。一方
    ``ThinkingRouter`` は常に ``phase="detect"`` (冒頭が think_start と一致
    するかをこれから待つ) から始まるため、「冒頭が一致しない = 考えていない」
    と誤判定して全文を content へ流してしまい、思考が本文に混入したうえ
    ``</think>`` だけが閉じ側の生テキストとして残る。

    判定は「think_start_tokens の最後の出現位置より後ろに think_end_tokens
    が無く、start より後ろが空白だけ」= 生成プロンプトの末尾が開きっぱなし
    の thinking ブロック、で行う。末尾の完全一致だけを見ないのは、
    ``<think>`` の直後に改行トークン等が続くテンプレートで取りこぼさない
    ため。逆に user/history 内のリテラルな ``<think>`` から assistant の
    generation suffix までには空白以外の role delimiter が続くので、誤って
    thinking 開始扱いにしない。

    ``has_thinking`` が False (このモデルではそもそも thinking を分離
    しない) なら常に False。
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

    ``already_thinking`` (bool): ``_apply_template`` が描画したプロンプトの
    末尾が既に未閉鎖の ``<think>`` で終わっている場合 (``_prompt_already_thinking``
    参照) に True を渡す。この場合モデル自身は think_start マーカーを生成
    しない (テンプレート側が既に出している) ので、``phase="detect"`` から
    始めて先頭が think_start と一致するのを待つと「考えていない」と誤判定
    してしまう。True なら ``phase`` をいきなり ``"thinking"`` から始める
    (detect をスキップする) — budget の計上・ストリーミング/非ストリーミング・
    tool calling の各経路は全て ``phase`` の値だけで分岐しているので、この
    初期値を変えるだけで全経路に一貫して効く。
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
            # テンプレートが既に <think> を開いた状態でプロンプトを終えて
            # いる。モデルは think_start を生成し直さないので detect を
            # 経由せず、いきなり thinking フェーズから始める。
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
        # Qwen 系は thinking 終了マーカーの直後に回答との区切りとして
        # ``\n\n`` を生成する。この framing を可視本文へ混ぜない。ただし
        # thinking を通らない応答の先頭改行はモデル出力そのものなので触らない。
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
        """content フェーズの 1 トークンを処理する。tool ルーティングが
        無効なら即デコードするだけ (従来どおり)。有効なら
        ``tool_call_start`` への先読みバッファを持つ ("thinking" フェーズが
        end marker を待つのと同型のローリングウィンドウ)。"""

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
                        self._strip_post_thinking_newlines = True
                    continue
                # ウィンドウが end marker 長を超えた分だけ確定で thinking へ
                # 流す (末尾は常に marker 長ぶん保持して先読みを続ける)。
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
            # マーカー長に届かないまま生成が終わった (極端に短い max_tokens
            # 等)。安全側で content 扱いにする (tool_call_start 監視込み)。
            pending, self.buf = self.buf, []
            for t in pending:
                self._feed_content_token(t, out)
        elif self.phase == "thinking":
            # </think> を出さないまま max_tokens/eos に達した。バッファに
            # 残る未確定トークンは thinking として確定させる。
            for t in self.buf:
                if not self._feed_thinking_token(t):
                    break
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
        seg = self._clean_content_segment(self.content_detok.last_segment)
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


# サンプリングパラメータの「恒等値」(分布を一切変えない既定値) の一覧。
# 例えば top_p=1.0 は確率質量を 100% 残すので分布を変えない。SpecRunner が
# SUPPORTED_SAMPLING_PARAMS に挙げていないキーでも、実際に渡ってきた値が
# ここに列挙された恒等値なら投機デコードの分布保証を何も脅かさないので、
# _check_and_strip_sampling_params は 400 で弾かない。マジックナンバーを条件式へ散らさず
# ここへ集約する。logit_bias は「None または空 dict」が恒等値なので、
# 値そのものではなく _is_identity_sampling_value 側で特別扱いする。
#
# top_k の -1 は 0 と同じく「top-k を無効化する」恒等値として扱う。
# FallbackRunner が使う mlx_lm.make_sampler も ``top_k > 0`` のときだけ
# フィルタを有効化するため、この値をそのまま渡しても分布は変わらない。
_IDENTITY_SAMPLING_VALUES: dict[str, tuple] = {
    "top_p": (0.0, 1.0),
    "top_k": (0, -1),
    "min_p": (0.0,),
    "frequency_penalty": (0.0,),
    "presence_penalty": (0.0,),
    "repetition_penalty": (0.0, 1.0),
}


def _is_identity_sampling_value(name: str, value) -> bool:
    """``value`` が ``name`` というサンプリングパラメータにとって恒等値
    (分布を一切変えない値) なら True。未指定 (None) は _parse_sampling_params
    が params dict に入れない設計だが、防御的にここでも通す。"""

    if value is None:
        return True
    if name == "logit_bias":
        return not value  # None または {} (空 dict) はバイアス無し
    return value in _IDENTITY_SAMPLING_VALUES.get(name, ())


def _check_and_strip_sampling_params(params: dict) -> str | None:
    """現在の Runner (SpecRunner/FallbackRunner) が受け付けないサンプリング
    パラメータが、かつ分布を変える値で指定されていれば、理由付きのエラー
    文字列を返す。値が恒等値 (例: top_p=1.0, frequency_penalty=0.0) なら
    キーが SUPPORTED_SAMPLING_PARAMS に無くても分布を変えないため 400 には
    しない — opencode や OpenAI SDK など、既定値をキー付きで送ってくる実
    クライアントが spec runner のモデルを使えなくなることを避けるため。
    SpecRunner が非恒等値を弾く理由は fastmlx/runner.py の SpecRunner
    docstring (Block Verification の分布保証との関係) を参照。

    副作用: エラーを返さずに通す場合、``params`` を破壊的に書き換えて、
    「runner がサポートしないが恒等値なので通したキー」を削除する。
    SpecEngine.generate() のように **kwargs を持たない generate() へ
    そのまま渡すと未知のキーワード引数で TypeError (-> 500) になるため —
    「無変換」という意味を、そのキーを一切渡さないことで正確に表現する。
    呼び出し側は 4 箇所あるが、ここで dict を直接変更することで各所に
    ストリップ処理を書かずに済ませる。
    """

    if not params:
        return None
    supported = getattr(STATE.runner, "SUPPORTED_SAMPLING_PARAMS", frozenset())
    unrecognized = set(params) - supported
    if not unrecognized:
        return None
    non_identity = sorted(
        name for name in unrecognized if not _is_identity_sampling_value(name, params[name])
    )
    if non_identity:
        kind = getattr(STATE.runner, "KIND", type(STATE.runner).__name__)
        return (
            f"this model is served via the '{kind}' runner, which does not support "
            f"non-default values for: {', '.join(non_identity)}. The speculative-decoding "
            "runner only supports 'seed' (plus identity values that leave the sampling "
            "distribution unchanged, e.g. top_p=1.0 or frequency_penalty=0.0) among the "
            "extended sampling parameters, because its correctness guarantee (rejection "
            "sampling against the exact target distribution) assumes temperature-only "
            "sampling; changing the base distribution would silently break that guarantee."
        )
    # ここに残っているのは全て「runner がサポートしないが恒等値」のキー。
    # 分布を変えないので拒否はしないが、runner.generate() が知らない
    # キーワード引数として受け取らないよう、渡す前に落としておく。
    for name in unrecognized:
        del params[name]
    return None


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


def _check_context_length(prompt_ids: list[int], protocol: str):
    """プロンプトが起動時に決めた上限 (モデル config の max_position_embeddings
    と、Metal が一括確保できる実際の上限から逆算した値のうち小さい方。
    ``--max-context-tokens`` で上書き可。``_resolve_default_max_context_tokens``
    参照) を超えていれば 400 を返す。

    Metal がアテンション行列を一括確保できず [metal::malloc] で落ちて
    そのまま 500 になる事故 (実測で ~57,000 トークン付近から発生) を防ぐ
    ための事前チェック。SpecEngine は新規プロンプトを一括で forward する
    (チャンク分割は 4-bit 量子化モデルで mx.quantized_matmul がバッチ長に
    応じて異なる丸めを返し、分割の有無で生成トークン列が一致しなくなる
    ことを実測で確認したため見送った) ので、この上限が実質的な唯一の
    ガードになる。"""

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
    except Exception as exc:  # noqa: BLE001 - model output drives third-party parsers
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
    **sampling_kwargs,
):
    """ワーカーを STATE.executor へ投げ、(キュー, Future) を返す。

    ``session`` はこのリクエスト用に ``_select_session`` が引き当てた
    session (ChatSession/FallbackSession) — 呼び出し側が ``STATE.lock`` の
    中で選んで渡す (グローバル状態の STATE.session は無い)。

    キューに積まれる要素: ``("reasoning_delta", text)`` / ``("content_delta",
    text)`` / ``("tool_call", {"id","name","arguments"})`` (成功裏に解析
    できた tool call 1 個ぶん) / ``("budget_exceeded", None)`` (予算超過を
    検知した回だけ 1 回) / ``("done", res)`` / ``("error", exc)``。

    呼び出し側は必ずこの Future を最後まで待つこと (正常終了・エラー・
    クライアント切断のどの経路でも)。そうしないと、まだ実行中のワーカーが
    この ``session`` を触っている間に次のリクエストがロックを取れてしまい、
    同じ session を 2 つの生成が同時に書き換えるレースになる。
    """

    q: queue.Queue = queue.Queue()
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
                session=session,
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
    prompt_ids: list[int],
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
    生成後に一括で行うだけ。

    ``prompt_ids`` は ``_apply_template`` が描画した実際のプロンプト —
    ``_prompt_already_thinking`` でテンプレートが既に ``<think>`` を開いた
    状態かどうかを判定し、``ThinkingRouter`` の初期 phase に反映する
    (server.py の ``_prompt_already_thinking`` docstring 参照)。"""

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
    """OpenAI 非ストリーム向け: (reasoning_text, content_text, tool_calls,
    budget_exceeded) を返す。tool_calls は成功裏に解析できた呼び出しだけ
    (解析に失敗したものは content_text 側にマーカーごと生テキストとして
    含まれる — 捏造しない方針)。``prompt_ids`` は ``_collect_events`` へ
    そのまま渡す (already_thinking 判定用)。"""

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
    unsupported_err = _check_and_strip_sampling_params(sampling_params)
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
            session = _select_session(prompt_ids)
            res = await _run_generate(
                prompt_ids,
                max_tokens,
                temp,
                STATE.eos_ids,
                None,
                session,
                **sampling_params,
            )
    except Exception as exc:
        return _openai_error(str(exc), status=500, err_type="server_error")
    _log_gen_stats(res)

    reasoning_text, content_text, tool_calls, budget_exceeded = _split_response_final(
        prompt_ids, res["tokens"], thinking_budget, tool_enabled, resolved_tools
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
    # 受付 (検証) は呼び出し側で StreamingResponse を組み立てる前に完了して
    # いる。生成の開始 (ロック獲得・ワーカー投入) を待たずに最初のイベントを
    # 出すことで、TTFT が支配項のワークロードでも「最初の 1 バイト」を早く
    # 返す (server.py の docstring 参照)。ここから先で 400 相当が判明する
    # 経路は無い。
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

    async with STATE.lock:
        session = _select_session(prompt_ids)
        q, future = _start_generation(
            prompt_ids,
            max_tokens,
            temp,
            thinking_budget,
            tool_enabled,
            tools_for_parsing,
            session=session,
            **(sampling_params or {}),
        )
        try:
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
        # トップレベルの ``system`` に加えて、``messages`` 内に role: "system"
        # の要素が混ざって送られてくるクライアントがいる (実例: Claude Code
        # — 捕獲したボディでは ['user', 'system'] の順で、system が末尾に来る。
        # Anthropic の公開仕様は role: "system" を messages に許していないが、
        # 実クライアントは普通に送ってくる)。チャットテンプレート
        # (Qwen 系) は system メッセージが先頭に無いと
        # "System message must be at the beginning" で落ちるので、トップ
        # レベルの system と messages 内の system ロールを両方とも先頭へ
        # 寄せて 1 個の system メッセージへ連結する。連結順は「トップ
        # レベル -> messages に現れた順」で元の順序を保つ (どちらも
        # 崩さない)。system 以外のメッセージは元の相対順序のまま残す。
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
    unsupported_err = _check_and_strip_sampling_params(sampling_params)
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
            session = _select_session(prompt_ids)
            res = await _run_generate(
                prompt_ids,
                max_tokens,
                temp,
                STATE.eos_ids,
                None,
                session,
                **sampling_params,
            )
    except Exception as exc:
        return _anthropic_error(str(exc), status=500, err_type="server_error")
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

    # 受付 (検証) は呼び出し側で StreamingResponse を組み立てる前に完了して
    # いる。生成の開始 (ロック獲得・ワーカー投入) を待たずに message_start を
    # 出すことで、TTFT が支配項のワークロードでも「最初の 1 バイト」を早く
    # 返す (server.py の docstring 参照)。ここから先で 400 相当が判明する
    # 経路は無い。
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

    async with STATE.lock:
        session = _select_session(prompt_ids)
        q, future = _start_generation(
            prompt_ids,
            max_tokens,
            temp,
            thinking_budget,
            tool_enabled,
            tools_for_parsing,
            session=session,
            **(sampling_params or {}),
        )
        try:
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
                for event in close_block(current_block):
                    yield event
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
    unsupported_err = _check_and_strip_sampling_params(sampling_params)
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
            session = _select_session(prompt_ids)
            res = await _run_generate(
                prompt_ids,
                max_tokens,
                temp,
                STATE.eos_ids,
                None,
                session,
                **sampling_params,
            )
    except Exception as exc:
        return _openai_error(str(exc), status=500, err_type="server_error")
    _log_gen_stats(res)

    _reasoning_text, text, _tool_calls, _budget_exceeded = _split_response_final(
        prompt_ids, res["tokens"], 0
    )
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
        session = _select_session(prompt_ids)
        # thinking_budget=0: ThinkingRouter を content-only に固定する
        # (has_thinking に関わらず) ので reasoning_delta は絶対に来ない。
        q, future = _start_generation(
            prompt_ids, max_tokens, temp, 0, session=session, **(sampling_params or {})
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


# ---------- OpenAI 互換 (Responses API, /v1/responses) ----------
#
# Codex CLI (2026-02-01 以降) は wire_api として "responses" しか受け付け
# なくなった (旧 "chat" は削除済み) ので、これを実装しない限り Codex は
# 動かない。Chat Completions とは構造が別物 (messages ではなく input、
# choices ではなく output 配列、tools はネストではなくフラット) だが、
# 生成そのもの・thinking の分離・tool call の解析・サンプリングパラメータ・
# session 選択・stop 判定は Chat Completions/Anthropic 経路と完全に共有
# する — ここで新しく書くのは「input を既存の内部 messages 形式へ正規化
# する」「イベント列 (_collect_events と同じ語彙) を output 配列/SSE
# イベントへ組み立てる」の 2 つだけで、生成ロジックを 3 つ目のプロトコル
# として複製しない。
#
# store / previous_response_id によるサーバー側での会話継続は実装しない
# (このサーバーは元々レスポンスを一切永続化しない設計 — 会話履歴は毎ターン
# クライアントが全文を送り直す前提、モジュール冒頭の docstring 参照)。
# previous_response_id が来たら黙って無視せず 400 で明示する。


def _responses_content_to_text(content) -> str:
    """Responses API の 1 アイテムぶんの content (文字列、または
    input_text/output_text/refusal などの型付きパーツの配列) をプレーン
    テキストへ落とす。``_content_to_text`` (Chat Completions/Anthropic 用)
    と同じ方針: image/audio/file 系のパーツは黙って読み飛ばさず
    ``MultimodalContentError`` (400) にする (fastmlx はテキスト専用モデル
    しか served しないため)。壊れた形 (パーツに type が無い、text が
    文字列でない等) も同様に ``InvalidContentError`` (400) にする。
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
                # モデル自身が出した refusal ブロックの echo。読み飛ばさず
                # テキストとして扱う (捏造ではなく、クライアントが実際に
                # 前ターンで受け取った文字列をそのまま送り返してくる)。
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
    """Responses API の ``input`` (+ ``instructions``) を、``_apply_template``
    が受け付ける既存の内部 messages 形式 (``_normalize_openai_messages`` と
    同じ形: role/content の他、assistant は ``tool_calls``、tool 結果は
    role: "tool" + ``tool_call_id``) へ変換する。ここで既存形式へ寄せる
    ことで、これより後ろの経路 (apply_chat_template 以降) を Chat
    Completions/Anthropic と完全に共有できる。

    ``input`` の各アイテムは type で分岐する (type 省略時は "message" 扱い
    — Codex CLI 等、素朴な ``{"role":..., "content":...}`` 形で送ってくる
    クライアントに対応するため):

    - ``message``: role (user/assistant/system/developer) + content。
      developer は system と同じ扱いにする (このサーバーに system 相当が
      2 種類ある意味は無い)。
    - ``function_call``: 前ターンでモデルが行った tool 呼び出し。連続する
      function_call は 1 個の assistant メッセージの ``tool_calls`` へ
      まとめる (Chat Completions の assistant.tool_calls と同じ形)。
    - ``function_call_output``: tool 実行結果。role: "tool" のメッセージへ
      変換する。
    - ``reasoning``/``item_reference``: 前ターンの reasoning 要約や
      (このサーバーが対応しない previous_response_id 前提の) 参照アイテム。
      再度モデルに読ませる必要は無いので読み飛ばす (bug 2: thinking ブロック
      の扱いと同じ方針)。
    - それ以外の未知の type は黙って無視せず 400 にする (壊れた/対応外の
      入力を空プロンプトとして通さない、というこのサーバー全体の方針に
      揃える)。

    Codex CLI は ``instructions`` (トップレベル) に加えて、``input`` の
    先頭付近に role: "developer" のメッセージを混ぜてくる (捕獲したボディ
    で確認済み)。developer は system と同じ扱いにするため、そのままだと
    system 相当のメッセージが 2 個、非連続または「先頭以外」の位置に並ぶ
    ことになり、Anthropic 経路の bug (System message must be at the
    beginning) と同じ理由でチャットテンプレートが落ちる。system/developer
    の内容はすべて集めて、``instructions`` -> ``input`` に現れた順で
    1 個の system メッセージへ連結し、必ず先頭へ置く。
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

        # function_call 以外のアイテムが来たら、直前まで貯めていた
        # function_call 群をここで確定させる (message アイテムと同じ扱い)。
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
    """Codex CLI が ``tools`` に混ぜてくる非 ``function`` 型を処理する。

    実際に捕獲したボディで確認した 2 種類:

    - ``{"type": "namespace", "tools": [...]}``: サブエージェント関連の
      ツール群をまとめる入れ物で、それ自体は呼び出し可能なツールではない。
      中の ``tools`` (中身は通常の ``{"type": "function", ...}``) を
      展開してトップレベルへ引き上げる。入れ子になっている場合に備えて
      再帰的に展開する。
    - ``{"type": "web_search", ...}``: このサーバーには web 検索を実行する
      主体が無い (fastmlx はローカルモデルへの単純な forward しか持たない)
      ので、渡されても実行できない。黙って握りつぶすとクライアントは
      検索が使えると思ったまま話を進めてしまうので、落としたことを
      運用者向けログへ残してから除く (_log_gen_stats と同じ
      ``[fastmlx-serve]`` 一行ログの作法)。

    ``function`` 型はそのまま素通しする (後続の ``_validate_responses_tools``
    がその形自体を検証する)。
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
                f"[fastmlx-serve] dropping unsupported tool in 'tools': "
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
    """Responses API のフラットな tool 定義 (``{"type":"function","name":...,
    "parameters":...}``) を、``apply_chat_template``/``tool_parser`` が
    期待する OpenAI (Chat Completions) 形式のネスト
    (``{"type":"function","function":{"name",...}}``) へ変換する。"""

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
    """``_resolve_tool_choice_openai`` の Responses API 版。tools の形が
    フラットな点以外は解決ロジック (auto/none/required/特定関数指定) は
    完全に同じ。"""

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
    """``_resolve_thinking`` の Responses API 版。読むフィールドは
    ``reasoning: {"effort": ...}`` — budget への写し方 (`_REASONING_EFFORT_BUDGET`)
    は OpenAI 側と共有する。"""

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
    """順序を保った高レベルイベント列 (``_collect_events`` と同じ語彙:
    reasoning_delta/content_delta/tool_call) を Responses API の ``output``
    配列 (型付きアイテム) へ組み立てる。``_anthropic_blocks_from_events``
    の Responses 版 — 連続する reasoning_delta/content_delta はそれぞれ
    1 個の reasoning/message アイテムへ結合し、tool_call は独立した
    function_call アイテムにする。"""

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


@app.post("/v1/responses")
async def responses_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _openai_error("request body must be valid JSON")
    if not isinstance(body, dict):
        return _openai_error("request body must be a JSON object")

    if body.get("previous_response_id") is not None:
        return _openai_error(
            "'previous_response_id' is not supported: this server does not persist "
            "responses server-side (see server.py docstring — sessions are keyed by "
            "prefix match on the conversation you resend, not by a stored response "
            "id). Send the full conversation in 'input' each turn instead."
        )
    if body.get("store") is True:
        return _openai_error(
            "'store: true' is not supported: this server never persists responses "
            "server-side, so a stored response would not be retrievable afterwards."
        )

    model_err = _check_model_openai(body)
    if model_err is not None:
        return model_err

    try:
        norm_messages = _normalize_responses_input(body.get("input"), body.get("instructions"))
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
    unsupported_err = _check_and_strip_sampling_params(sampling_params)
    if unsupported_err is not None:
        return _openai_error(unsupported_err)

    model_id = STATE.model_name
    stream = bool(body.get("stream", False))
    resp_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())

    if stream:
        return StreamingResponse(
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
            ),
            media_type="text/event-stream",
        )

    try:
        async with STATE.lock:
            session = _select_session(prompt_ids)
            res = await _run_generate(
                prompt_ids,
                max_tokens,
                temp,
                STATE.eos_ids,
                None,
                session,
                **sampling_params,
            )
    except Exception as exc:
        return _openai_error(str(exc), status=500, err_type="server_error")
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
):
    """Responses の主要な lifecycle イベント列を出す: response.created ->
    response.output_item.added -> response.output_text.delta /
    response.reasoning_summary_text.delta / response.function_call_arguments.delta
    -> 各 done イベント -> response.output_item.done ->
    response.completed / response.incomplete / response.failed。

    受付 (検証) は呼び出し側で StreamingResponse を組み立てる前に完了して
    いる — OpenAI/Anthropic の既存ストリーミング経路 (server.py の docstring
    参照) と同じ方針で、生成の開始を待たずに response.created を即座に
    流す。
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

    async with STATE.lock:
        session = _select_session(prompt_ids)
        q, future = _start_generation(
            prompt_ids,
            max_tokens,
            temp,
            thinking_budget,
            tool_enabled,
            tools_for_parsing,
            session=session,
            **(sampling_params or {}),
        )
        try:
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
            terminal_event = (
                "response.completed" if status == "completed" else "response.incomplete"
            )
            yield sse(
                terminal_event,
                {"type": terminal_event, "response": final_response},
            )
        finally:
            await _await_worker(future)


# ---------- 起動 ----------


def _resolve_model_max_context(config: dict) -> int | None:
    """モデル自身が申告する文脈長上限を、起動時にロードした生の config
    (``mlx_lm_load(..., return_config=True)`` の戻り値) から取る。
    ハードコードしない — ``--max-context-tokens`` が優先され、これは
    未指定時のフォールバックにだけ使う。

    VLM ラッパー形式 (Qwen3.6-35B-A3B 等) は ``max_position_embeddings`` が
    トップレベルではなく ``text_config`` の下にネストされているので、
    見つからなければ 1 段だけ潜って探す。見つからなければ None を返し、
    ``_check_context_length`` はガード無効 (無制限) のまま動く。
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
    """SpecEngine が新規プロンプトを forward できる、Metal の実際の確保上限
    から逆算したトークン数上限。

    SpecEngine (fastmlx/spec.py) は新規プロンプトを PREFILL_STEP_SIZE
    (既定 2048) トークンずつチャンク分割して forward する
    (``SpecEngine._prefill_hidden``)。そのため 1 回の forward で確保される
    注意スコア行列は ``num_attention_heads * PREFILL_STEP_SIZE * T *
    bytes_per_elem`` (T は処理済みトークン数の総量、最後のチャンクで最大)
    — 分割前の ``num_attention_heads * T^2 * bytes_per_elem`` (T に対して
    二次) から、T に対して線形に下がっている。それでも Metal の 1 バッファ
    上限 (``mx.device_info()["max_buffer_length"]``、実機で 86,586,540,032
    バイト) を超えると ``[metal::malloc]`` でリクエストが失敗する。

    ``max_position_embeddings`` (モデルが申告する学習時文脈長) だけを見ると
    現実的にはまず届かない桁の値を返すこともあるので、その値をそのまま
    上限にはせず、ここで求めた値と ``_resolve_model_max_context`` の値の
    小さい方を実際の上限として使う (``_resolve_default_max_context_tokens``)。
    多くのモデルでは max_position_embeddings の方が先に効く — 分割後の壁は
    非常に大きい (このモデルでは 100 万トークン超) ため。

    理論上ちょうど 1 バッファを埋め切る T (割ると即失敗する境界) に 0.9 を
    掛けて安全側に倒す — アテンション行列以外にも同時に確保されるバッファ
    がある分、理論値ちょうどまでは安全とは言えないため。
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
    """``--max-context-tokens`` 未指定時の既定値: モデルが申告する上限
    (``_resolve_model_max_context``) と、Metal が実際に確保できる上限から
    逆算した値 (``_metal_safe_prefill_limit``) の小さい方。どちらも取れな
    ければ None (ガード無効)。"""

    candidates = [
        v
        for v in (_resolve_model_max_context(config), _metal_safe_prefill_limit(config))
        if v is not None
    ]
    return min(candidates) if candidates else None


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
    ap.add_argument(
        "--max-sessions",
        type=int,
        default=8,
        help="会話ごとの session (KV/prompt cache) を同時に保持する上限 (LRU、"
        " 超えたら最も長く未使用のものを捨てる)。91GB 級モデルの上に会話ごとの"
        " KV を無制限に積まないための上限",
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
        return runner, tokenizer, max_context_tokens, source

    runner, tokenizer, max_context_tokens, source = executor.submit(_load).result()
    if max_context_tokens is not None:
        print(
            f"[fastmlx-serve] プロンプト長の上限: {max_context_tokens} トークン ({source})"
        )
    else:
        print(
            "[fastmlx-serve] プロンプト長の上限: 検出できず (ガード無効 — config に"
            " max_position_embeddings が見当たらず、Metal 側の逆算もできなかった。"
            " --max-context-tokens で指定可)"
        )

    # runner の種類に応じて会話ごとのスロットが積むクラスを決める。
    # SpecRunner なら SpecEngine 用の ChatSession、それ以外 (FallbackRunner)
    # なら mlx_lm prompt_cache 用の FallbackSession — どちらも .processed を
    # 持ち、_select_session はこれを見るだけなので中身は気にしない。
    session_factory = ChatSession if getattr(runner, "KIND", None) == "spec" else FallbackSession

    STATE = ModelState(
        runner=runner,
        tokenizer=tokenizer,
        session_pool=OrderedDict(),
        session_factory=session_factory,
        lock=asyncio.Lock(),
        executor=executor,
        model_name=served_name,
        eos_ids=set(tokenizer.eos_token_ids),
        max_tokens_cap=args.max_tokens,
        default_temp=args.temp,
        created_ts=int(time.time()),
        max_sessions=args.max_sessions,
        max_context_tokens=max_context_tokens,
    )
    print(
        f"[fastmlx-serve] served model name: {served_name} "
        f"(session pool: {session_factory.__name__}, max {args.max_sessions} 会話)"
    )
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
