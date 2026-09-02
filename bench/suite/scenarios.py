"""シナリオ定義: 一点突破 (point) / エージェント型 (agent) / コード編集ループ
(code-edit) / RAG (rag) / 並列 (parallel、定義のみ・既定では走らせない)。

どのシナリオも「ターンのテンプレート列」として表現する。各ターンは
`content_fn(mctx)` が実際の本文を組み立て、`run.py` 側が前ターンの応答を
履歴に積んでから次のターンを送る (`bench/self_snapshot.py` の
「追記ターン: 実クライアントは履歴をまるごと送り直す」と同じ流儀)。

長文脈の本文は必ず `tools/_bench_text.py` の実文プール (docs の Markdown +
自前のソース) から、**互いに重ならない窓**を切って作る。プロセス内で
`PoolCursor` を 1 つだけ共有し、シナリオをまたいでもオフセットを進める
(`bench/self_snapshot.py` が文脈・繰り返しをまたいで守っているのと同じ規律)。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "bench", REPO_ROOT / "tools"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from vs_mlx_serve import QUESTIONS, SHORT  # noqa: E402

__all__ = [
    "MaterializeCtx", "PoolCursor", "TurnTemplate", "Scenario",
    "build_scenario", "SCENARIO_NAMES", "DEFAULT_CTXS", "DEFAULT_CONCURRENCY",
]

SCENARIO_NAMES = ("point", "agent", "code-edit", "rag", "parallel")
DEFAULT_CTXS = (0, 4000, 17000, 25000, 32000, 50000)
DEFAULT_CONCURRENCY = (1, 2, 4, 8)

RAG_QUESTIONS = [
    "検索結果のうち、主張の根拠になっている記述を1つ引用して答えてください。",
    "検索結果の内容が互いに矛盾していないか確認してください。",
    "検索結果に基づいて、次に確認すべき点を1つ挙げてください。",
]

AGENT_TASKS = [
    "このリポジトリのベンチ道具を1つ選んで、何を測っているか要約して。"
    "まず対象ファイルの中身を確認して。",
]
AGENT_FOLLOWUPS = [
    "この内容を踏まえて、次に確認すべきことを1つだけ挙げて。",
    "了解。それを踏まえた結果は以上。次にすべきことは？",
    "続けて。最後に変更点を3行で要約して。",
]

CODE_EDIT_ROUNDS = [
    "このファイルの docstring を、何をしているか一目で分かるように書き直して。"
    "コード全体を、変更後の完全な内容で返して。",
    "さっきの変更に加えて、型ヒントが抜けている関数があれば補って。"
    "また全体を返して。",
    "エラーハンドリングが薄い箇所があれば指摘して直して。全体を返して。",
]


class PoolCursor:
    """`tools/_bench_text.py` の実文プールから、重ならない窓を順に切り出す。

    プロセス内で 1 つだけ作り、全シナリオ・全 rep で使い回すこと。
    足りなくなったら (窓を切り尽くしたら) `ValueError` — 繰り返しで
    埋めてはいけない (受理率が嘘になる。CLAUDE.md 参照)。
    """

    def __init__(self, tok) -> None:
        from _bench_text import text_pool  # 遅延 import (トークナイザ依存)
        self.tok = tok
        self.ids: list[int] = tok.encode(text_pool())
        self.offset = 0

    def take(self, n_tokens: int) -> str:
        n_tokens = max(n_tokens, 16)
        if self.offset + n_tokens > len(self.ids):
            raise ValueError(
                f"実文プールが足りない (要求 {n_tokens} tok, 残り "
                f"{len(self.ids) - self.offset} tok)。シナリオの文脈設定を"
                " 見直すこと (繰り返しで埋めない)")
        chunk = self.tok.decode(self.ids[self.offset: self.offset + n_tokens])
        self.offset += n_tokens
        return chunk


@dataclass
class MaterializeCtx:
    """ターンの `content_fn` に渡す実行時コンテキスト。"""

    tok: object
    pool: PoolCursor
    rng: object  # random.Random
    # point シナリオ用: このシナリオ実行が対象とする文脈トークン数の目安
    target_ctx: int = 0


@dataclass
class TurnTemplate:
    """1 ターンぶんの「作り方」。実際の送信・応答受信は `run.py` が行う。"""

    label: str
    content_fn: Callable[[MaterializeCtx], str]
    max_tokens: int
    reset_history: bool = False
    note: str = ""


@dataclass
class Scenario:
    """シナリオ 1 つの定義。"""

    name: str
    description: str
    tool_calls: bool
    prompt_source: str
    turns: list[TurnTemplate] = field(default_factory=list)
    concurrency_levels: list[int] | None = None  # None = 単一ストリーム
    enabled_by_default: bool = True
    disabled_reason: str = ""


# ── point ──────────────────────────────────────────────────────────────

def _point_cold(mctx: MaterializeCtx) -> str:
    c = mctx.target_ctx
    if c == 0:
        return SHORT
    win = max(c - 200, 16)
    q = QUESTIONS[mctx.rng.randrange(len(QUESTIONS))]
    return f"{mctx.pool.take(win)}\n\n---\n\n{q}"


def _point_warm(mctx: MaterializeCtx) -> str:
    return "続けて。"


def build_point_scenario(ctx: int, tokens: int) -> Scenario:
    """一点突破: 単一ストリームで冷 TTFT・温 TTFT・decode を測る。

    `bench/self_snapshot.py` の測り方 (冷 1 ターン → 履歴を丸ごと送り直す
    追記ターンで温 TTFT) をそのまま踏襲する。文脈ごとに `Scenario` を作る
    (窓幅が `ctx` に依存するため)。
    """
    return Scenario(
        name=f"point@{ctx}",
        description=(
            f"文脈 {ctx} トークンの単一ストリーム。冷 TTFT → 追記ターンで"
            " 温 TTFT → decode。tool call なし。"),
        tool_calls=False,
        prompt_source="tools/_bench_text.py の実文プール (ctx=0 は固定短文)",
        turns=[
            TurnTemplate("cold", _point_cold, tokens,
                         note="このターンの TTFT が「冷 TTFT」"),
            TurnTemplate("warm", _point_warm, 8,
                         note="履歴を丸ごと送り直す追記ターン。TTFT が「温 TTFT」"),
        ],
    )


# ── agent ──────────────────────────────────────────────────────────────

def _agent_task(mctx: MaterializeCtx) -> str:
    return AGENT_TASKS[0]


def _agent_tool_result(i: int, followup: str) -> Callable[[MaterializeCtx], str]:
    def _f(mctx: MaterializeCtx) -> str:
        chunk = mctx.pool.take(600)
        return f"[ツール実行結果 {i}]\n{chunk}\n\n{followup}"
    return _f


def build_agent_scenario(tokens: int = 128) -> Scenario:
    """エージェント型マルチターン: タスク依頼 → 疑似ツール結果の注入 → 短い応答、
    を繰り返す。**本物の function calling API (`tools`/`tool_calls`) は
    使わない** — 目的はエージェント運用に特有の「大きな外部テキストが
    ターンごとに履歴へ追加され、各ターンの生成は短い」という文脈成長の
    形を再現することで、ツール呼び出し自体の正しさを測ることではない。

    履歴は毎ターン累積 (`reset_history=False`) — 実クライアントと同じく
    前のやり取り全体を送り直すので、接頭辞キャッシュの再利用がターンを
    追うごとに効いてくるはずの形になっている。
    """
    turns = [TurnTemplate("task", _agent_task, tokens,
                          note="タスク依頼 (ツール呼び出しを促す短い発話)")]
    for i, followup in enumerate(AGENT_FOLLOWUPS, start=1):
        turns.append(TurnTemplate(
            f"tool_result_{i}", _agent_tool_result(i, followup), tokens,
            note="疑似ツール結果 (実文プールから ~600 tok) + 短い追加指示"))
    return Scenario(
        name="agent",
        description=(
            "5 ターン。タスク依頼 1 回 + 疑似ツール結果注入 3 回 + 要約 1 回。"
            " 各ターンの生成は短め、履歴は累積。"),
        tool_calls=False,
        prompt_source="固定タスク文 + tools/_bench_text.py の実文プール (擬似ツール出力)",
        turns=turns,
    )


# ── code-edit ──────────────────────────────────────────────────────────

def _code_edit_paste(mctx: MaterializeCtx) -> str:
    target = REPO_ROOT / "tools" / "_bench_text.py"
    body = target.read_text(errors="ignore")
    return f"```python\n{body}\n```\n\n{CODE_EDIT_ROUNDS[0]}"


def _code_edit_round(text: str) -> Callable[[MaterializeCtx], str]:
    def _f(mctx: MaterializeCtx) -> str:
        return text
    return _f


def build_code_edit_scenario(tokens: int = 512) -> Scenario:
    """コード編集ループ: 実ファイル全体を貼って複数ラウンド編集依頼を重ねる。

    毎ラウンド「変更後のファイル全体を返して」と頼む設計なので、生成は
    長め (既定 512 トークン)。対象ファイルはリポジトリ内の実ファイル
    (`tools/_bench_text.py`) — 繰り返し文字列は使わない (CLAUDE.md の
    n-gram/MTP が当たりすぎる罠を避ける規律を踏襲)。
    """
    turns = [TurnTemplate("paste_file", _code_edit_paste, tokens,
                          note="実ファイル全体の貼り付け + 最初の編集依頼")]
    for i, text in enumerate(CODE_EDIT_ROUNDS[1:], start=2):
        turns.append(TurnTemplate(f"edit_round_{i}", _code_edit_round(text),
                                  tokens, note="追加の編集依頼 (新規プール消費なし)"))
    return Scenario(
        name="code-edit",
        description=f"{len(turns)} ラウンドのコード編集ループ。生成は毎回ファイル全体。",
        tool_calls=False,
        prompt_source="tools/_bench_text.py (自己言及: 対象ファイルそのもの)",
        turns=turns,
    )


# ── rag ────────────────────────────────────────────────────────────────

def _rag_fresh_turn(i: int) -> Callable[[MaterializeCtx], str]:
    def _f(mctx: MaterializeCtx) -> str:
        chunks = "\n\n".join(f"[検索結果 {j}]\n{mctx.pool.take(500)}"
                             for j in range(1, 5))
        q = RAG_QUESTIONS[(i - 1) % len(RAG_QUESTIONS)]
        return f"{chunks}\n\n質問: {q}"
    return _f


def _rag_shared_first(mctx: MaterializeCtx) -> str:
    chunks = "\n\n".join(f"[検索結果 {j}]\n{mctx.pool.take(500)}"
                         for j in range(1, 5))
    return f"{chunks}\n\n質問: {RAG_QUESTIONS[0]}"


def _rag_shared_followup(q: str) -> Callable[[MaterializeCtx], str]:
    def _f(mctx: MaterializeCtx) -> str:
        return q
    return _f


def build_rag_scenario(mode: Literal["fresh", "shared"] = "fresh",
                       tokens: int = 256, n_queries: int = 4) -> Scenario:
    """RAG: 検索結果チャンクを注入してから質問する。

    - `fresh` (既定): 毎回新しい検索結果 (新規窓) を注入する独立クエリを
      `n_queries` 回。**接頭辞はほぼ効かない** — RAG の現実的な悪いケース
      (検索結果が毎回変わる) を再現する。各ターンは `reset_history=True`
      (前ターンの履歴を引きずらない、独立リクエスト)。
    - `shared`: 最初のターンだけ検索結果チャンクを注入し、以降は同じ文脈に
      短い追加質問を重ねる (履歴累積)。**接頭辞キャッシュが効くはずの形**
      で、`fresh` との対比になる。

    チャンクは 4 個 x 500 トークン程度 (実文プールの新規窓)。tool call なし
    (検索そのものはハーネス側でテキストとして注入するだけで、検索ツールを
    実際には呼ばない)。
    """
    if mode == "fresh":
        turns = [TurnTemplate(f"query_{i}", _rag_fresh_turn(i), tokens,
                              reset_history=True,
                              note="新規検索結果 4 チャンク + 質問 (独立リクエスト)")
                 for i in range(1, n_queries + 1)]
        desc = f"{n_queries} 回の独立クエリ。毎回新規チャンクで接頭辞は効かない想定。"
    elif mode == "shared":
        turns = [TurnTemplate("query_1", _rag_shared_first, tokens,
                              note="検索結果 4 チャンク + 最初の質問")]
        for i, q in enumerate(RAG_QUESTIONS[1:], start=2):
            turns.append(TurnTemplate(f"query_{i}", _rag_shared_followup(q),
                                      tokens,
                                      note="同じ検索結果への追加質問 (履歴累積)"))
        desc = "同じ検索結果に対する追加質問。履歴累積で接頭辞が効くはずの対比条件。"
    else:
        raise ValueError(f"未知の rag mode: {mode!r}")
    return Scenario(
        name=f"rag-{mode}",
        description=desc,
        tool_calls=False,
        prompt_source="tools/_bench_text.py の実文プール (検索結果チャンクとして注入)",
        turns=turns,
    )


# ── parallel (定義のみ) ───────────────────────────────────────────────

def build_parallel_scenario(concurrency: tuple[int, ...] = DEFAULT_CONCURRENCY) -> Scenario:
    """同時 1/2/4/8 の並列デコード。**定義のみ、既定では実行しない。**

    mlxturbo の並列デコード (continuous batching) 経路は
    `docs/research/KERNEL-BRIEF-DECODE-BW.md` の「バッチ x 投機の判定」節で
    宣言していた反転条件 (B=4 で 1.6 倍未満) を満たせず畳まれている
    (`mlxturbo/batch_spec.py` は残るが既定では誰も呼ばない)。並列デコードを
    直すまでは、この経路をフェアな対戦材料として使わない。

    実装するときの指標: 集計 decode tok/s (総トークン数 / 壁時計) と
    per-request レイテンシの p50/p95。単一ストリームの `point` と同じ
    プロンプト源を使い回せる設計にすること。
    """
    return Scenario(
        name="parallel",
        description=f"同時 {list(concurrency)} の並列デコード。",
        tool_calls=False,
        prompt_source="point と同じ (未実装)",
        concurrency_levels=list(concurrency),
        enabled_by_default=False,
        disabled_reason=(
            "mlxturbo の並列デコードが直っていないためフェアでない"
            " (docs/research/KERNEL-BRIEF-DECODE-BW.md 参照)"),
    )


def build_scenario(name: str, **kwargs) -> Scenario:
    """シナリオ名からファクトリを呼ぶディスパッチャ。

    `point` は `ctx` (必須) と `tokens` (既定 512)、`rag` は `mode`
    (既定 "fresh")、`parallel` は `concurrency` を受け取る。
    """
    if name == "point":
        return build_point_scenario(kwargs["ctx"], kwargs.get("tokens", 512))
    if name == "agent":
        return build_agent_scenario(kwargs.get("tokens", 128))
    if name == "code-edit":
        return build_code_edit_scenario(kwargs.get("tokens", 512))
    if name == "rag":
        return build_rag_scenario(kwargs.get("mode", "fresh"),
                                  kwargs.get("tokens", 256))
    if name == "parallel":
        return build_parallel_scenario(
            tuple(kwargs.get("concurrency", DEFAULT_CONCURRENCY)))
    raise ValueError(f"未知のシナリオ: {name!r} (候補: {SCENARIO_NAMES})")
