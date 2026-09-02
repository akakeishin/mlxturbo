"""エンジンアダプタ: 起動 argv・ready 判定・キャッシュ信号の取り方をエンジンごとに閉じ込める。

`bench/vs_mlx_serve.py` の `Server` / `wait_ready` / `model_id` / `stream_once` /
`install_term_handler` をそのまま再利用する (作り直さない)。ここで足すのは
「エンジンごとに違う部分」の抽象化だけ: argv の組み方、ready 判定の URL、
そしてログ/レスポンスから接頭辞キャッシュ命中を読む方法。

## 接頭辞キャッシュ命中は HTTP レスポンスから読む (ログ scraping ではない)

mlxturbo (`mlxturbo/server.py:2530` `_usage_dict`) と mlx-serve
(`~/dev/mlx-serve/src/server.zig:9222` `formatChatUsage`) は、どちらも
OpenAI 標準の `usage.prompt_tokens_details.cached_tokens` を実装している。
リクエストに `stream_options: {"include_usage": true}` を付ければ、
ストリームの最終チャンクにこの値が乗る。**両エンジンが同じ JSON 形で
answers する**ので、ログの正規表現に頼るより堅く、かつエンジン非依存の
契約として扱える。`docs/NEXT-SESSION-PROMPT.md` が言う「相手のログを読む」
(`[hot-cache] reused`) は mlx-serve 固有のログ形式なので、こちらは
フォールバック (ログが取れているときだけ使う二次確認) に位置づける。

新しいエンジンを足すときは、そのエンジンが `cached_tokens` を実装していない
場合がある。その場合 `parse_cache_hit_from_log` 側で拾えるようにするか、
「キャッシュ命中は未検証」として結果に明記すること (無いものを揃わせない)。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "bench", REPO_ROOT / "tools"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

# 既存の道具をそのまま使う。作り直さない。
from vs_mlx_serve import (  # noqa: E402
    Server, install_term_handler, model_id, wait_ready,
)

__all__ = [
    "Server", "install_term_handler", "model_id", "wait_ready",
    "EngineAdapter", "MlxturboAdapter", "MlxServeAdapter",
    "OMlxAdapter", "LlamaCppAdapter", "MlxLmAdapter",
    "ENGINE_REGISTRY", "stream_with_usage",
]


def stream_with_usage(
    port: int, messages: list, n_tokens: int, model: str = "x",
    extra_body: dict | None = None,
):
    """`vs_mlx_serve.stream_once` と同じ計時に加え、末尾の usage チャンクを拾う。

    返り値: (ttft 秒, decode 秒, チャンク数, 本文, usage dict | None)。

    `usage` は `stream_options: {"include_usage": true}` を付けたときだけ
    OpenAI 互換サーバーが送ってくる最終チャンクに乗る。両エンジンとも実装
    済み (モジュール docstring 参照)。`prompt_tokens_details.cached_tokens`
    が「その文脈をどれだけ接頭辞再利用できたか」で、**「冷」と称した
    リクエストが実は温まっていた**ことを検出する主手段になる
    (`cached_tokens` が 0 でなければ冷えていない)。
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": n_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_body:
        payload.update(extra_body)
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n = 0
    parts: list[str] = []
    usage: dict | None = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            choices = d.get("choices") or [{}]
            delta = (choices[0] if choices else {}).get("delta") or {}
            # vs_mlx_serve.stream_once と同じ理由で思考も数える
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    t_dec = time.perf_counter()
                n += 1
                parts.append(piece)
    if ttft is None:
        return float("nan"), float("nan"), 0, "", usage
    return ttft, time.perf_counter() - t_dec, n, "".join(parts), usage


@dataclass
class LogSignals:
    """エンジンのログ/レスポンスから読み取れる、判定に使う信号。

    値が取れなければ None のままにする (「揃わない」を「0」で誤魔化さない)。
    """
    cached_tokens: int | None = None       # usage.prompt_tokens_details.cached_tokens
    thinking_on: bool | None = None        # ログの thinking=true/false 等
    spec_mode: str | None = None           # 発火した投機モード名 (取れれば)
    tokens_per_step: float | None = None   # ラウンドあたり受理トークン数 (取れれば)


class EngineAdapter(ABC):
    """エンジンごとの違いを閉じ込める基底クラス。

    `name` は結果 JSON とレポートの列見出しに使う識別子 (例: "mlxturbo",
    "mlx-serve")。`kind` はアダプタの実装クラス種別 (`ENGINE_REGISTRY` の鍵)。
    """

    name: str
    kind: str

    @abstractmethod
    def build_argv(self, host: str, port: int) -> list[str]:
        """サーバー起動コマンドの argv を返す (実行はしない)。"""

    def is_available(self) -> bool:
        """このアダプタが今すぐ実行可能か (バイナリ/モデルが手元にあるか)。

        既定では常に True (具象クラスで上書きする)。stub アダプタは False。
        """
        return True

    def unavailable_reason(self) -> str | None:
        """`is_available()` が False のときの理由 (日本語、レポートに出す用)。"""
        return None

    def ready_url(self, host: str, port: int) -> str:
        """起動完了を判定する URL。既定は OpenAI 互換の `/v1/models`。"""
        return f"http://{host}:{port}/v1/models"

    def wait_ready(self, port: int, timeout: float = 900.0) -> bool:
        return wait_ready(port, timeout=timeout)

    def model_id(self, port: int) -> str:
        return model_id(port)

    def parse_log_signals(self, log_text: str) -> LogSignals:
        """サーバーログ (`--server-log` で保存したもの) から二次確認の信号を拾う。

        既定は「何も拾えない」。エンジンごとに正規表現を持つ具象クラスで
        上書きする。usage.cached_tokens が一次情報で、これは二次確認。
        """
        return LogSignals()

    # 停止は Server コンテキストマネージャに任せる (作り直さない)。
    # 呼び出し側は `with Server(adapter.name, adapter.build_argv(...), port,
    # log_path=...) as srv: ...` の形で使う。


@dataclass
class MlxturboAdapter(EngineAdapter):
    """`python -m mlxturbo.server`。argv は `vs_mlx_serve.py` / `self_snapshot.py`
    と同じ組み方 (差し替えるとハーネス間で数字が食い違う)。
    """

    model: str
    ngram: str | None = None
    mtp: str | None = None
    extra: str | None = None  # shlex.split して argv 末尾に足す
    name: str = "mlxturbo"
    kind: str = "mlxturbo"

    def build_argv(self, host: str, port: int) -> list[str]:
        import shlex
        argv = [sys.executable, "-m", "mlxturbo.server",
                "--model", os.path.expanduser(self.model),
                "--host", host, "--port", str(port)]
        if self.ngram:
            argv += ["--ngram", os.path.expanduser(self.ngram)]
        if self.mtp:
            argv += ["--mtp", os.path.expanduser(self.mtp)]
        if self.extra:
            argv += shlex.split(self.extra)
        return argv

    def is_available(self) -> bool:
        return Path(os.path.expanduser(self.model)).exists()

    def unavailable_reason(self) -> str | None:
        if self.is_available():
            return None
        return f"モデルが見つからない: {self.model}"

    def parse_log_signals(self, log_text: str) -> LogSignals:
        sig = LogSignals()
        m = re.search(r"prefill reused=(\d+)", log_text)
        if m:
            sig.cached_tokens = int(m.group(1))
        return sig


@dataclass
class MlxServeAdapter(EngineAdapter):
    """mlx-serve (Zig バイナリ)。argv は `bench/self_snapshot.py` の
    `--serve-bin` 経路と同じ組み方。
    """

    binary: str
    model: str
    mtp: bool = True
    log_level: str = "info"
    extra: str | None = None
    name: str = "mlx-serve"
    kind: str = "mlx-serve"

    def build_argv(self, host: str, port: int) -> list[str]:
        import shlex
        argv = [os.path.expanduser(self.binary), "--serve",
                "--model", os.path.expanduser(self.model),
                "--host", host, "--port", str(port),
                "--log-level", self.log_level]
        if self.mtp:
            argv.append("--mtp")
        if self.extra:
            argv += shlex.split(self.extra)
        return argv

    def is_available(self) -> bool:
        return (Path(os.path.expanduser(self.binary)).exists()
                and Path(os.path.expanduser(self.model)).exists())

    def unavailable_reason(self) -> str | None:
        b, m = os.path.expanduser(self.binary), os.path.expanduser(self.model)
        if not Path(b).exists():
            return f"バイナリが見つからない: {b}"
        if not Path(m).exists():
            return f"モデルが見つからない: {m}"
        return None

    def parse_log_signals(self, log_text: str) -> LogSignals:
        sig = LogSignals()
        m = re.search(r"reused (\d+)/(\d+)", log_text)
        if m:
            sig.cached_tokens = int(m.group(1))
        m = re.search(r"thinking=(true|false)", log_text)
        if m:
            sig.thinking_on = m.group(1) == "true"
        # mlx-serve の tests/bench.sh が使っているのと同じパターン
        modes = re.findall(r"\[spec-stats\] mode=(\w+)", log_text)
        if modes:
            sig.spec_mode = max(set(modes), key=modes.count)
        return sig


@dataclass
class _StubAdapter(EngineAdapter):
    """未実装エンジンの拡張点。`is_available()` は常に False —
    「アダプタは存在するが今は走らない」ことをスケジューラに伝える。

    実装するときにやること (このクラスを埋める形で):
      1. `build_argv` — 起動コマンド
      2. `ready_url` / `wait_ready` — health チェックの流儀が違えば上書き
      3. `parse_log_signals` — そのエンジンのログ形式
      4. `is_available` — バイナリ/モデルの実在チェックに差し替える
    """

    name: str = "stub"
    kind: str = "stub"
    note: str = "未実装"

    def build_argv(self, host: str, port: int) -> list[str]:
        raise NotImplementedError(
            f"{self.kind} アダプタは未実装 ({self.note})。"
            " engines.py の _StubAdapter docstring を参照して実装すること。")

    def is_available(self) -> bool:
        return False

    def unavailable_reason(self) -> str | None:
        return f"未実装アダプタ ({self.note})"


@dataclass
class OMlxAdapter(_StubAdapter):
    """oMLX (dflash-mlx 系譜)。`docs/research/LANES-2026-09.md` の調査記録を
    参照。CLI の起動引数と ready 判定を埋めれば `MlxServeAdapter` とほぼ同型
    になるはず (どちらも OpenAI 互換 HTTP を出す想定)。
    """

    name: str = "oMLX"
    kind: str = "omlx"
    note: str = "起動 argv 未調査"


@dataclass
class LlamaCppAdapter(_StubAdapter):
    """llama.cpp (`llama-server`)。**量子化方式が違う (K-quant)** ので、
    速度だけでなく KLD も必ず並記すること (`docs/BENCHMARKS.md` の量子化差の
    扱いを踏襲)。ready 判定は `/health`、usage の `cached_tokens` 相当は
    `--prompt-cache` 系のログか `timings.prompt_cache_hit_tokens`
    (llama.cpp 側の実装次第、要確認)。
    """

    name: str = "llama.cpp"
    kind: str = "llama.cpp"
    note: str = "起動 argv・量子化方式差の明記が未着手"


@dataclass
class MlxLmAdapter(_StubAdapter):
    """mlx-lm 素の `mlx_lm.server`。MTP も n-gram も無い素の下限として、
    Gemma 4 系 (mlxturbo が対応しない qwen4_exp 以外のモデル) の
    もう一方のベースラインに使える (`~/dev/mlx-serve/tests/bench.sh` の
    モデル行列を参照)。
    """

    name: str = "mlx-lm"
    kind: str = "mlx-lm"
    note: str = "起動 argv 未実装"


ENGINE_REGISTRY: dict[str, type[EngineAdapter]] = {
    "mlxturbo": MlxturboAdapter,
    "mlx-serve": MlxServeAdapter,
    "omlx": OMlxAdapter,
    "llama.cpp": LlamaCppAdapter,
    "mlx-lm": MlxLmAdapter,
}
