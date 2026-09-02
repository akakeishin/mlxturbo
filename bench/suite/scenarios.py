"""シナリオ定義: 一点突破 (point) / エージェント型 (agent) / コード編集ループ
(code-edit) / RAG (rag) / 並列 (parallel、定義のみ・既定では走らせない)。

どのシナリオも「ターンのテンプレート列」として表現する。各ターンは
`content_fn(mctx)` が実際の本文を組み立て、`run.py` 側が前ターンの応答を
履歴に積んでから次のターンを送る (`bench/self_snapshot.py` の
「追記ターン: 実クライアントは履歴をまるごと送り直す」と同じ流儀)。

長文脈の本文は `PROMPT_POOLS` (下記) の実文プールから、**互いに重ならない
窓**を切って作る。`point` シナリオは種類の違うプールを軸に振る (投機デコード
の受理率はテキストの性質で大きく動くので、1 つの池だけで測ると片方に有利な
数字が固定される — `docs/research/BENCH-DESIGN-2026-09.md` (c) 節)。
`agent`/`code-edit`/`rag` は従来通り `"default"` プール
(`tools/_bench_text.py` の実文プール、docs Markdown + 自前ソース) を使う。

`PoolCursor` はプール 1 つぶんの読み位置を持つ。`PoolRegistry` がプロセス内で
プールごとに 1 つずつ (使うものだけ遅延生成して) 共有し、シナリオ・文脈・rep
をまたいでもオフセットを進める (`bench/self_snapshot.py` が文脈ごとの offset
で守っている規律をプール単位に一般化した形)。
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
    "MaterializeCtx", "PoolCursor", "PoolRegistry", "PoolSpec",
    "PROMPT_POOLS", "POOL_ORDER", "POOL_TOKEN_BUDGET", "TurnTemplate",
    "Scenario", "build_scenario", "SCENARIO_NAMES", "DEFAULT_CTXS",
    "DEFAULT_CONCURRENCY",
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
    """実文プール 1 つぶんの読み位置。重ならない窓を順に切り出す。

    プールごとに 1 つだけ作り (`PoolRegistry` 経由)、そのプールを使う
    全シナリオ・全 rep で使い回すこと。足りなくなったら (窓を切り尽くしたら)
    `ValueError` — 繰り返しで埋めてはいけない (受理率が嘘になる。CLAUDE.md
    参照)。
    """

    def __init__(self, tok, text: str) -> None:
        self.tok = tok
        self.ids: list[int] = tok.encode(text)
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


def _read_files(paths: list[Path], max_chars: int | None = None) -> str:
    """複数ファイルを連結して読む。存在しないものは黙って飛ばす
    (機体によって `~/dev/mlx-serve` や `.venv` の中身が違うため、
    「無ければ次の情報源」という設計そのものが前提にしている)。
    """
    parts: list[str] = []
    total = 0
    for p in paths:
        if not p.exists() or not p.is_file():
            continue
        try:
            t = p.read_text(errors="ignore")
        except OSError:
            continue
        parts.append(t)
        total += len(t)
        if max_chars is not None and total >= max_chars:
            break
    return "\n\n".join(parts)


def _pool_default() -> str:
    """`agent`/`code-edit`/`rag` が使う既定プール。`tools/_bench_text.py` の
    実文プールをそのまま使う (この骨組みの前バージョンからの挙動を変えない)。
    """
    from _bench_text import text_pool  # 遅延 import (トークナイザ非依存だが軽くはない)
    return text_pool()


def _pool_ja_prose() -> str:
    """(a) 日本語散文。`docs/**/*.md` — このリポジトリの研究ログ・設計文書
    (ほぼ日本語)。"""
    files = sorted(REPO_ROOT.glob("docs/**/*.md"))
    return _read_files(files)


def _pool_en_prose() -> str:
    """(b) 英語散文。`README.en.md` を核に、mlx-serve のドキュメント一式
    (`~/dev/mlx-serve/docs/**/*.md` — トップレベルの API/CLI 文書に加えて
    `docs/gotchas/*.md` が本文の大半を占める、英語) を再帰的に足す。それでも
    足りなさそうなら (目安 20 万文字未満)、`.venv` に入っているパッケージの
    英語 README/RST/METADATA で埋める (ユーザー指定の優先順位)。
    """
    paths = [REPO_ROOT / "README.en.md"]
    mlx_serve_docs = Path.home() / "dev" / "mlx-serve" / "docs"
    if mlx_serve_docs.exists():
        paths += sorted(mlx_serve_docs.rglob("*.md"))
    text = _read_files(paths)
    if len(text) < 200_000:
        venv_lib = REPO_ROOT / ".venv" / "lib"
        extra = (sorted(venv_lib.glob("**/README*"))
                + sorted(venv_lib.glob("**/METADATA"))) if venv_lib.exists() else []
        text += "\n\n" + _read_files(extra, max_chars=1_500_000)
    return text


def _pool_source_code() -> str:
    """(c) ソースコード。python (mlxturbo/tools/bench の自前ソース) と、
    あれば zig (`~/dev/mlx-serve/src/*.zig`) を混ぜる。zig が無い機体では
    python だけになる (無いものを埋めない — その場合は窓が短くなるだけ)。
    """
    files = (sorted(REPO_ROOT.glob("mlxturbo/*.py"))
             + sorted(REPO_ROOT.glob("mlxturbo/kernels/*.py"))
             + sorted(REPO_ROOT.glob("tools/*.py"))
             + sorted(REPO_ROOT.glob("bench/*.py")))
    zig_dir = Path.home() / "dev" / "mlx-serve" / "src"
    if zig_dir.exists():
        files += sorted(zig_dir.glob("*.zig"))
    return _read_files(files)


def _pool_structured_data() -> str:
    """(d) 構造化データ。`bench/results/*.json` (計測結果の生 JSON) +
    `bench/results/logs/*.log` (サーバーログ)。JSON/ログはプローズと語の
    分布が大きく違う (キー名の反復、数値の連続、タイムスタンプ) — n-gram/MTP
    の効き方を試す別極として入れる。
    """
    results_dir = REPO_ROOT / "bench" / "results"
    files = (sorted(results_dir.glob("*.json"))
             + sorted((results_dir / "logs").glob("*.log")))
    return _read_files(files, max_chars=2_000_000)


_REPETITIVE_FILES = (
    # 表行 (`^|`) + 箇条書き行の比率が高い順に選んだ、このリポジトリの
    # 実在する文書 (`grep -c` で数えて選定。数値は BENCH-DESIGN-2026-09.md
    # (c) 節に記録してある)。
    "docs/README.md",
    "docs/research/ROOFLINE-2026-08-26.md",
    "docs/research/ENGINE-COMPARISON.md",
    "docs/research/D2-RESULTS.md",
    "docs/research/BAKE-RESULTS.md",
    "docs/research/KERNEL-INTEL.md",
    "docs/research/LANES-2026-09.md",
    "docs/research/ARCH-BETS.md",
    "docs/research/EXPANSION-BOTTLENECKS.md",
    "docs/COMPATIBILITY.md",
    "docs/MTP-FLASH.md",
    "docs/research/ANE-PREFILL-BRIEF.md",
    "docs/research/PLAN.md",
    "docs/research/MLX-SERVE-PACK-FORMAT.md",
    "docs/research/DECODE-ANATOMY-2026-08-31.md",
    "docs/research/GATE-RESULTS-A2.md",
    "docs/research/HYPOTHESES-A2.md",
    "docs/research/SDPA-WIDTH-WALL.md",
    "docs/research/PREFILL-CHUNKING-DETERMINISM.md",
    "docs/research/KERNEL-BRIEF-HC.md",
    "docs/research/KERNEL-BRIEF-MOE-GDN.md",
    "docs/research/KERNEL-BRIEF.md",
    "docs/research/ISA-QUEUE.md",
    "docs/research/OFFLOAD-RESEARCH.md",
)


def _pool_repetitive() -> str:
    """(f) 反復の多いテキスト。表・箇条書きが密な Markdown を選んだ
    (`_REPETITIVE_FILES` — `docs/**/*.md` 全体のうち表行 (`^|`) と箇条書き行
    の比率を数えて上位に来たもの)。n-gram が当たりやすい極端側を明示的に
    用意する狙いで、繰り返し文字列を自分で作るのではなく、**実在する
    表・箇条書き密度の高い文書**を選ぶことで「繰り返しで長さを作らない」
    規律を保ったまま反復性の高い実文を確保する。
    """
    files = [REPO_ROOT / p for p in _REPETITIVE_FILES]
    return _read_files(files)


def _pool_conversation() -> str:
    """(e) 会話履歴。**注意: 本物の保存済み会話ログではない** — このリポジトリ
    には再利用できる形の会話ログが無い (`bench/opencode_e2e.py` は実サーバーと
    やり取りする e2e ドライバで、静的なテキストとしては読めない)。代わりに、
    ja-prose と source-code の実文プールから段落/関数単位の断片を切り出し、
    `User:` / `Assistant:` の役割プレフィックスを挟んで積んだ、トランスクリプト
    "形" の文字列を作る。文そのものは毎回違う実文から取るので、同じ短い
    文字列を繰り返すわけではない — 作っているのは「役割プレフィックスと
    細かい段落区切りが挟まる」という、単一の長い説明文とは違うトークン分布
    であって、本物の会話内容ではないことを明記する。
    """
    ja = _pool_ja_prose()
    code = _pool_source_code() or ja
    ja_paras = [p for p in ja.split("\n\n") if len(p) > 40]
    code_paras = [p for p in code.split("\n\n") if len(p) > 40]
    n = min(len(ja_paras), len(code_paras))
    turns = []
    for i in range(n):
        turns.append(f"User: {ja_paras[i][:400]}")
        turns.append(f"Assistant: {code_paras[i][:400]}")
    return "\n\n".join(turns)


@dataclass
class PoolSpec:
    """実文プール 1 つの定義。`build()` はテキストを返すだけで、
    トークナイザには触れない (遅延評価: dry-run では絶対に呼ばれない)。
    """

    key: str
    category: str  # 設計書 (c) 節のラベル ("(a) 日本語散文" 等)
    description: str
    build: Callable[[], str]


PROMPT_POOLS: dict[str, PoolSpec] = {
    "default": PoolSpec("default", "(既定)",
                        "tools/_bench_text.py の実文プール (docs + 自前ソース)。"
                        " agent/code-edit/rag が使う", _pool_default),
    "ja-prose": PoolSpec("ja-prose", "(a) 日本語散文",
                         "docs/**/*.md (このリポジトリの研究ログ・設計文書)",
                         _pool_ja_prose),
    "en-prose": PoolSpec("en-prose", "(b) 英語散文",
                         "README.en.md + ~/dev/mlx-serve/docs/*.md"
                         " (無ければ .venv の英語 README/RST で補う)",
                         _pool_en_prose),
    "source-code": PoolSpec("source-code", "(c) ソースコード",
                            "mlxturbo/tools/bench の python + "
                            "~/dev/mlx-serve/src/*.zig (あれば)",
                            _pool_source_code),
    "structured-data": PoolSpec("structured-data", "(d) 構造化データ",
                                "bench/results/*.json + bench/results/logs/*.log",
                                _pool_structured_data),
    "conversation": PoolSpec("conversation", "(e) 会話履歴 (構成物、注意書き参照)",
                             "ja-prose / source-code の断片を role プレフィックス"
                             " 付きで積んだトランスクリプト形。本物の会話ログではない",
                             _pool_conversation),
    "repetitive": PoolSpec("repetitive", "(f) 反復の多いテキスト",
                           "表・箇条書きが密な Markdown (MTP-FLASH.md 等、"
                           " scenarios.py の _pool_repetitive docstring参照)",
                           _pool_repetitive),
}
# point で振る 6 種 (a)-(f)。"default" は agent/code-edit/rag 専用なので含めない。
POOL_ORDER: tuple[str, ...] = (
    "ja-prose", "en-prose", "source-code", "structured-data",
    "conversation", "repetitive",
)

# 実測トークン数のスナップショット (2026-09-02、~/models/ddalcu-mlxlm の
# トークナイザで `PoolSpec.build()` を実際にエンコードして測った値。
# `run.py --dry-run` はこれを使ってプール残量が足りるかを見積もる —
# dry-run 自体はトークナイザを読まないので、ここで固定値として持つ。
# プールの情報源 (docs/**/*.md 等) が増減したら実測し直すこと (単に
# `PROMPT_POOLS[key].build()` をトークナイズして `len()` を見ればよい)。
POOL_TOKEN_BUDGET: dict[str, int] = {
    "default": 1_070_797,
    "ja-prose": 267_148,
    "en-prose": 318_066,
    "source-code": 4_108_940,
    "structured-data": 2_094_687,
    "conversation": 398_099,
    "repetitive": 59_574,  # 6 種のうち最小 — 文脈点を選ぶときの律速
}


class PoolRegistry:
    """必要になったプールだけを遅延構築して使い回すキャッシュ。

    6 種類のプールを毎回全部読んでトークナイズすると (特に structured-data
    や source-code は数百万文字あるので) 無駄が大きい。実際に `point` が
    指定したプールだけを、初回アクセス時に `PoolCursor` へ変換する。
    """

    def __init__(self, tok) -> None:
        self.tok = tok
        self._cursors: dict[str, PoolCursor] = {}

    def get(self, key: str) -> PoolCursor:
        if key not in self._cursors:
            spec = PROMPT_POOLS[key]
            self._cursors[key] = PoolCursor(self.tok, spec.build())
        return self._cursors[key]


@dataclass
class MaterializeCtx:
    """ターンの `content_fn` に渡す実行時コンテキスト。"""

    tok: object
    pools: PoolRegistry
    rng: object  # random.Random
    # point シナリオ用: このシナリオ実行が対象とする文脈トークン数の目安
    target_ctx: int = 0

    @property
    def pool(self) -> PoolCursor:
        """後方互換の別名。`"default"` プールを指す
        (agent/code-edit/rag の既存コードがこれを使う)。"""
        return self.pools.get("default")


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

def _point_cold(pool_key: str) -> Callable[[MaterializeCtx], str]:
    def _f(mctx: MaterializeCtx) -> str:
        c = mctx.target_ctx
        if c == 0:
            return SHORT
        win = max(c - 200, 16)
        q = QUESTIONS[mctx.rng.randrange(len(QUESTIONS))]
        return f"{mctx.pools.get(pool_key).take(win)}\n\n---\n\n{q}"
    return _f


def _point_warm(mctx: MaterializeCtx) -> str:
    return "続けて。"


def build_point_scenario(ctx: int, tokens: int, pool: str = "default") -> Scenario:
    """一点突破: 単一ストリームで冷 TTFT・温 TTFT・decode を測る。

    `bench/self_snapshot.py` の測り方 (冷 1 ターン → 履歴を丸ごと送り直す
    追記ターンで温 TTFT) をそのまま踏襲する。文脈・プールごとに `Scenario`
    を作る (窓幅は `ctx`、窓の中身は `pool` に依存するため)。`pool` は
    `PROMPT_POOLS` の鍵 (既定 `"default"` — 骨組みの前バージョンと同じ挙動。
    入力の多様性を測るときは `POOL_ORDER` の 6 種を明示的に指定する)。
    ctx=0 はどのプールでも `vs_mlx_serve.SHORT` の固定短文になる (プールを
    引く必要が無いほど短いため) — プール間の差は文脈が付くところから出る。
    """
    pool_label = "" if pool == "default" else f":{pool}"
    return Scenario(
        name=f"point@{ctx}{pool_label}",
        description=(
            f"文脈 {ctx} トークン、プール={pool} の単一ストリーム。冷 TTFT →"
            " 追記ターンで温 TTFT → decode。tool call なし。"),
        tool_calls=False,
        prompt_source=f"PROMPT_POOLS[{pool!r}] ({PROMPT_POOLS[pool].description})"
        if pool in PROMPT_POOLS else f"未知のプール: {pool}",
        turns=[
            TurnTemplate("cold", _point_cold(pool), tokens,
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

    `point` は `ctx` (必須)・`tokens` (既定 512)・`pool` (既定 "default")、
    `rag` は `mode` (既定 "fresh")、`parallel` は `concurrency` を受け取る。
    """
    if name == "point":
        return build_point_scenario(kwargs["ctx"], kwargs.get("tokens", 512),
                                    kwargs.get("pool", "default"))
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
