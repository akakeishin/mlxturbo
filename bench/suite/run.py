#!/usr/bin/env python3
"""公開ベンチマークの実行ドライバ。

やること (`docs/research/BENCH-DESIGN-2026-09.md` の (d)(h) 節が仕様):

  1. シナリオ x 文脈 x エンジンの全組を「ブロック」に展開する
     (1 ブロック = 1 回のサーバー起動。文脈ごとに交互に起動する、というのは
     「両サーバーを同時にメモリへ載せない」制約から来る)。`point` シナリオは
     さらにブロックの中で「セル」(プール x 出力長 x thinking の組) を振る
     (`--tier` で既定の掃引を選ぶ、`--pools`/`--tokens-set`/`--thinking-set`
     で個別に上書きできる)
  2. ブロックの実行順・ブロック内のセル順をシャッフルする (乱数種はログに
     残す。再現できるランダム化であって、無作為でごまかしているわけではない)
  3. ブロックごとに: 起動 → 2 段ウォームアップ → 全セル x `reps` 回の反復測定
     → 停止 → 残留プロセス確認 → 冷却 (既定 180 秒)
  4. 生ログを JSONL で `bench/results/suite/<run-id>/raw.jsonl` に 1 ブロック
     ずつ追記する (`--resume` で中断から再開できる)

**GPU を使う工程 (サーバー起動・推論) は `--dry-run` では一切呼ばない。**
`--dry-run` は「どの順で何を起動するか」「プールの残量が足りるか」
「所要時間の見積もり」だけを表示する (モデルもトークナイザも読まない)。

    uv run python bench/suite/run.py --dry-run
        # tier=quick (既定): Flash-Next, mlxturbo vs mlx-serve, point x 文脈
        # 6 点、プールは "default" 1 種のみ、各 3 反復、ランダム順、冷却 180 秒
    uv run python bench/suite/run.py --dry-run --tier standard
    uv run python bench/suite/run.py --dry-run --tier overnight

3 段の tier (docs/research/BENCH-DESIGN-2026-09.md (h) 節):

  - quick:     今までの既定計画そのまま。約 2 時間
  - standard:  池 (6 種) x 出力長 (128/1024) x thinking (off/on) を反復 1 回で
               全部回す。文脈点は最小のプール (repetitive, 実測 154,676 トー
               クン、in-repo 保証値) に収まるよう (0, 4000, 8000) に絞って
               ある。数時間
  - overnight: standard と同じ池 x 出力長 x thinking の全部の組を反復 3 回
               以上で回し (p50/p95 が出せる)、quick と同じ長文脈ラダー
               (プール "default" のみ) も反復 3 回以上で回し、さらに
               「mlxturbo が実際に負けている 17k 以上でも池差は出るか」を
               見るための long-diversity 軸 (17k x 池 6 種 x 出力長 2 種、
               thinking off、反復 1) を足す。三つを合わせて一晩

実行 (GPU を使う。このスクリプト自身は対戦の判定をしない — 判定は
CLAUDE.md の「計測の判定と commit は親が行う」を読む人間が行う):

    tools/biglock.sh uv run python bench/suite/run.py --tier overnight
    # 途中で止まったら (電源、SIGTERM 等) 同じ out-dir を指定して再開:
    tools/biglock.sh uv run python bench/suite/run.py --resume bench/results/suite/<run-id>
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "bench", REPO_ROOT / "bench" / "suite",
           REPO_ROOT / "tools"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from engines import (  # noqa: E402
    ENGINE_REGISTRY, EngineAdapter, LlamaCppAdapter, MlxLmAdapter,
    MlxServeAdapter, MlxturboAdapter, OMlxAdapter, Server,
    install_term_handler, stream_with_usage,
)
from scenarios import (  # noqa: E402
    DEFAULT_CTXS, POOL_ORDER, POOL_TOKEN_BUDGET, PROMPT_POOLS,
    SCENARIO_NAMES, Scenario, build_scenario,
)

# ── 既定プリセット: Flash-Next, mlxturbo vs mlx-serve ────────────────────
# 値の根拠は docs/NEXT-SESSION-PROMPT.md の再現コマンドと
# bench/results/logs/turbo-0902c.log (n-gram/MTP はサイドカー自動発見な
# ので --ngram / --mtp は不要、というログ上の事実)。
DEFAULT_MLXTURBO = dict(model="~/models/ddalcu-mlxlm", ngram=None, mtp=None)
DEFAULT_MLXSERVE = dict(
    binary="~/dev/mlx-serve/zig-out/bin/mlx-serve",
    model="~/models/ddalcu-flashnext-serve-4bit", mtp=True)

DEFAULT_HOST = "127.0.0.1"
ENGINE_PORTS = {"mlxturbo": 8151, "mlx-serve": 8161,
                "omlx": 8171, "llama.cpp": 8181, "mlx-lm": 8191}


def default_engines() -> dict[str, EngineAdapter]:
    """既定の設定を持つ全アダプタ。stub (omlx/llama.cpp/mlx-lm) も含める —
    「未登録エンジン」(CLI のタイプミス) と「まだ実装していないエンジン」
    (`is_available()` が False で理由を名乗る) を区別するため。
    """
    return {
        "mlxturbo": MlxturboAdapter(**DEFAULT_MLXTURBO),
        "mlx-serve": MlxServeAdapter(**DEFAULT_MLXSERVE),
        "omlx": OMlxAdapter(),
        "llama.cpp": LlamaCppAdapter(),
        "mlx-lm": MlxLmAdapter(),
    }


# ── 池 x 出力長 x thinking の掃引軸 (tier) ────────────────────────────

@dataclass(frozen=True)
class Cell:
    """`point` の 1 測定条件。他シナリオは `pool="default"` の 1 セルだけを
    使う (`--tier`/`--pools`/`--tokens-set`/`--thinking-set` は point 専用 —
    「点ごとに」というユーザー要件のスコープをそのまま反映している)。
    """

    pool: str
    tokens: int
    thinking: str
    reps: int
    extra_kwargs: tuple = ()  # (key, value) のタプル。rag の mode 等

    def kwargs(self) -> dict:
        return dict(self.extra_kwargs)

    def label(self) -> str:
        return f"{self.pool}/{self.tokens}tok/think={self.thinking}"


@dataclass(frozen=True)
class AxisConfig:
    """掃引軸 1 グループ。`ctxs x pools x tokens_set x thinking_set` の
    直積が、そのグループが生む `point` セル群になる。

    `tag` は同じ `(シナリオ, 文脈)` に複数の軸が重なったときに**ブロックを
    分けるための識別子**。例えば overnight は ctx=17000 に「長文脈ラダー」
    (`tag="ladder"`) と「17k 池掃引」(`tag="long-diversity"`) の 2 軸が
    重なるが、同じブロックに混ぜてしまうとブロック内でシャッフルした
    セルのどれが「最初 (真の冷)」になるか運任せになり、ラダー側の
    見出し数字 (quick と同じ意味を持たせたい) が汚れる。軸ごとに別ブロック
    (別起動) にすることで、両方が自分の「最初のセルの rep=0」を持てる。
    """

    ctxs: tuple[int, ...]
    pools: tuple[str, ...]
    tokens_set: tuple[int, ...]
    thinking_set: tuple[str, ...]
    reps: int
    tag: str = "axis"


# tier=standard/overnight の文脈点は当てずっぽうではない — POOL_TOKEN_BUDGET
# の実測値 (最小のプール repetitive で 154,676 トークン、in-repo 保証値) から
# 「池 x 出力長 x thinking x reps」の総消費量が予算に収まる上限を逆算して
# 選んだ。overnight の 3 本目の軸 (17k の long-diversity) は、mlxturbo が
# 実際に負けている文脈 (17k 以上の prefill/decode) でも池間のばらつきを
# 見るために追加した — 短文脈 (0,4000) だけの掃引では、優劣の勝負どころで
# 池差が出るかどうかが分からないままだった。計算は `pool_demand_report`
# と対応する (`--dry-run` が同じ計算をして表示する)。
TIERS: dict[str, dict] = {
    "quick": dict(
        axes=(AxisConfig(ctxs=DEFAULT_CTXS, pools=("default",),
                         tokens_set=(512,), thinking_set=("off",), reps=3,
                         tag="ladder"),),
        cooldown=180.0,
        summary="既定計画。文脈 6 点 x プール 1 種 (default) x 反復 3。約 2 時間",
    ),
    "standard": dict(
        axes=(AxisConfig(ctxs=(0, 4000, 8000), pools=POOL_ORDER,
                         tokens_set=(128, 1024), thinking_set=("off", "on"),
                         reps=1, tag="diversity"),),
        cooldown=180.0,
        summary=("池 (6 種) x 出力長 (128/1024) x thinking (off/on) を全部、"
                 " 反復 1 回で回す (分布は出ない)。文脈は (0,4000,8000)。数時間"),
    ),
    "overnight": dict(
        axes=(
            AxisConfig(ctxs=DEFAULT_CTXS, pools=("default",),
                      tokens_set=(512,), thinking_set=("off",), reps=3,
                      tag="ladder"),
            AxisConfig(ctxs=(0, 4000), pools=POOL_ORDER,
                      tokens_set=(128, 1024), thinking_set=("off", "on"),
                      reps=3, tag="diversity-short"),
            # long-diversity: mlxturbo が負けている 17k 以上の文脈でこそ
            # 池間のばらつきを見たい、というのが短文脈掃引だけでは埋まらない
            # 穴だった。17k 1 点・thinking off 固定・反復 1 で 6 池 x 出力長
            # 2 種を振る (文脈 x thinking まで掛けると 154,676 トークンの
            # 保証予算を超えるので、ここだけは割り切って反復 1・thinking off
            # 固定にしてある)。`tag` を "ladder" と別にして、ctx=17000 で
            # ラダー軸とブロックが混ざらないようにしてある (`tag` の docstring
            # 参照 — 混ざるとラダー側の「真の冷」がセルのシャッフル次第になる)。
            AxisConfig(ctxs=(17000,), pools=POOL_ORDER,
                      tokens_set=(128, 1024), thinking_set=("off",), reps=1,
                      tag="long-diversity"),
        ),
        cooldown=180.0,
        summary=("quick と同じ長文脈ラダー (default プール) + 池 x 出力長 x"
                 " thinking の全部の組 (文脈 0,4000、反復 3 以上) + 池ごとの"
                 " 17k 掃引 (出力長 2 種、thinking off、反復 1、独立ブロック)"
                 " を合わせて回す。一晩"),
    ),
}


# ── 所要時間の見積もり (dry-run 表示専用。実測ではない) ──────────────────
# 2026-09-02 の「直したハーネスで測った現在地」(SESSION-2026-09-02-CATCHUP.md)
# を較正点にした線形補間。文脈 25000/32000 は 17000-50000 間の補間で、
# 実測ではなく見積もりであることを常に明記する。
# **thinking="on" とプール差の較正データは無い** — この表は thinking="off"、
# プール="default" 相当の実測しか元にしていない。on / 他プールの推定は
# この表をそのまま流用した「較正なしの近似」であることを `note` に明記する。
_CALIB = {
    "mlxturbo": {
        "cold_ttft": {0: 0.49, 4000: 8.28, 17000: 37.6, 50000: 163.0},
        "decode_tps": {0: 47.5, 4000: 48.8, 17000: 41.0, 50000: 34.4},
        "warm_ttft": {0: 0.45, 4000: 0.46, 17000: 0.51, 50000: 1.33},
        "boot_s": 180.0,
    },
    "mlx-serve": {
        "cold_ttft": {0: 0.17, 4000: 5.77, 17000: 29.2, 50000: 108.0},
        "decode_tps": {0: 53.7, 4000: 55.3, 17000: 56.0, 50000: 30.8},
        "warm_ttft": {0: 0.72, 4000: 0.87, 17000: 0.92, 50000: 22.8},
        "boot_s": 180.0,
    },
}


def _interp(table: dict[int, float], x: int) -> float:
    xs = sorted(table)
    if x <= xs[0]:
        return table[xs[0]]
    if x >= xs[-1]:
        lo, hi = xs[-2], xs[-1]
    else:
        lo = max(v for v in xs if v <= x)
        hi = min(v for v in xs if v >= x)
        if lo == hi:
            return table[lo]
    t = (x - lo) / (hi - lo)
    return table[lo] + t * (table[hi] - table[lo])


def estimate_block_seconds(engine_kind: str, ctx: int | None,
                           cells: list[Cell], cooldown: float) -> dict:
    """1 ブロック (起動→全セル測定→冷却) の推定秒数。較正表が無いエンジン
    (stub) は `None` を返す — 「わからない」を 0 で埋めない。
    """
    calib = _CALIB.get(engine_kind)
    c = ctx if ctx is not None else 4000  # point 以外は較正点が無いので代用
    if calib is None:
        return {"boot_s": None, "measure_s": None, "cooldown_s": cooldown,
                "total_s": None, "note": f"{engine_kind} の較正データが無い"}
    boot_s = calib["boot_s"]
    cold = _interp(calib["cold_ttft"], c)
    tps = _interp(calib["decode_tps"], c)
    warm = _interp(calib["warm_ttft"], c)
    measure_s = 0.0
    for cell in cells:
        decode_s = cell.tokens / tps if tps > 0 else 0.0
        measure_s += (cold + decode_s + warm) * cell.reps
    warmup_s = 20.0  # 2 段ウォームアップ (短文 + 長文プロンプト 1 回ずつ)
    total = boot_s + warmup_s + measure_s + cooldown
    note = f"ctx={c} 較正点からの補間 (実測ではない)"
    if any(cell.thinking == "on" or cell.pool != "default" for cell in cells):
        note += "。thinking=on / プール差の較正データは無く、off/default の値を流用"
    return {"boot_s": boot_s, "warmup_s": warmup_s, "measure_s": measure_s,
            "cooldown_s": cooldown, "total_s": total, "note": note}


# ── セル展開・ブロック構築 ────────────────────────────────────────────

def expand_cells(scenario_names: list[str], axes: tuple[AxisConfig, ...],
                 default_tokens: int, default_thinking: str,
                 ) -> list[tuple[str, int | None, str, Cell]]:
    """`--scenarios` x tier の軸から `(シナリオ名, ctx, 軸タグ, Cell)` の
    平坦な列を作る。

    池 x 出力長 x thinking の掃引は **`point` にだけ**適用する
    (ユーザー要件が「point の各文脈点で」と明示しているスコープ)。
    `agent`/`code-edit`/`rag`/`parallel` は 1 セルだけ (`pool="default"`、
    `tokens`/`thinking` は CLI の既定値) を使い、挙動を変えない
    (軸タグは `"single"` — 他の軸と混ざる余地が無いので固定)。
    """
    flat: list[tuple[str, int | None, str, Cell]] = []
    for name in scenario_names:
        if name == "point":
            for axis in axes:
                for ctx in axis.ctxs:
                    for pool in axis.pools:
                        for tokens in axis.tokens_set:
                            for thinking in axis.thinking_set:
                                flat.append(("point", ctx, axis.tag, Cell(
                                    pool=pool, tokens=tokens,
                                    thinking=thinking, reps=axis.reps)))
        elif name == "rag":
            for mode in ("fresh", "shared"):
                flat.append(("rag", None, "single", Cell(
                    pool="default", tokens=default_tokens,
                    thinking=default_thinking, reps=axes[0].reps,
                    extra_kwargs=(("mode", mode),))))
        else:
            # agent / code-edit / parallel: シナリオ 1 つに 1 セル
            flat.append((name, None, "single", Cell(
                pool="default", tokens=default_tokens,
                thinking=default_thinking, reps=axes[0].reps)))
    return flat


@dataclass
class Block:
    """1 回のサーバー起動で完結する測定単位。`cells` は同じ
    (シナリオ, 文脈, 軸タグ) に属する全セル (プール x 出力長 x thinking の
    組)。tier=quick では常に 1 セル、standard/overnight では複数になる。

    軸タグ (`axis_tag`) でブロックを分ける理由は `AxisConfig.tag` の
    docstring を参照 — 同じ文脈に複数の軸が重なっても (例: overnight の
    ctx=17000 はラダー軸と long-diversity 軸が両方触る)、軸ごとに別ブロック
    (別起動) にして、それぞれが自分の「最初のセルの rep=0 (真の冷)」を
    持てるようにする。
    """

    index: int
    scenario_name: str
    engine_kind: str
    ctx: int | None
    axis_tag: str
    cells: list[Cell]
    block_seed: int

    def label(self) -> str:
        c = f"ctx={self.ctx}" if self.ctx is not None else "ctx=-"
        return (f"{self.scenario_name:10s} {c:10s} [{self.axis_tag}] "
               f"{self.engine_kind:10s} ({len(self.cells)} セル)")


def group_into_blocks(flat_cells: list[tuple[str, int | None, str, Cell]],
                      engine_kinds: list[str], seed: int) -> list[Block]:
    """`(シナリオ, 文脈, 軸タグ)` ごとにセルをまとめ、エンジンごとに
    1 ブロックを作る。

    「文脈ブロックごとに交互に起動する」の実装: グループの出現順そのものを
    シャッフルし、各グループでどちらのエンジンを先に起動するかも毎回
    コインを振る。ブロック内のセル順も (dry-run が見せる実行順そのものと
    して) シャッフルする — ブロック内で最初に来たセルの最初の rep だけが
    「真にフレッシュな冷」になる (`docs/research/BENCH-DESIGN-2026-09.md`
    (i) 節)。**軸タグをグループ鍵に含める** ことで、同じ文脈に複数の軸が
    重なっても (`AxisConfig.tag` 参照) 別ブロック (別起動) になり、
    軸ごとに独立した「最初のセルの rep=0」を持てる。
    """
    rng = random.Random(seed)
    groups: dict[tuple[str, int | None, str], list[Cell]] = {}
    order: list[tuple[str, int | None, str]] = []
    for name, ctx, tag, cell in flat_cells:
        key = (name, ctx, tag)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(cell)
    rng.shuffle(order)

    blocks: list[Block] = []
    idx = 0
    for name, ctx, tag in order:
        cells = list(groups[(name, ctx, tag)])
        rng.shuffle(cells)
        pair = list(engine_kinds)
        rng.shuffle(pair)
        for eng in pair:
            blocks.append(Block(index=idx, scenario_name=name, engine_kind=eng,
                               ctx=ctx, axis_tag=tag, cells=list(cells),
                               block_seed=rng.randrange(1 << 30)))
            idx += 1
    return blocks


def pool_demand_report(blocks: list[Block]) -> list[dict]:
    """プールごとの推定必要トークン数と実測予算を比べる。

    `point` の窓は `max(ctx-200, 16)` トークンで、セルごと・rep ごとに
    重ならない窓を消費する (`PoolCursor` の規律)。**エンジンごとには数えない**
    — `run_block` は各ブロックで `PoolRegistry` をブロック内だけの資源として
    フレッシュに作り直す (offset 0 から)。同じ `(シナリオ, 文脈)` のセル列は
    エンジン間で同じ順に並べてある (`group_into_blocks`) ので、mlxturbo と
    mlx-serve は同じ窓を独立に引く (= 同じ本文を両エンジンに送る、比較の
    公平性のため) だけであって、1 つの共有プールを取り合っているわけではない。
    そのため `(シナリオ, 文脈)` の組ごとに 1 回だけ数える。

    予算を超えるプールは `over=True` — 実行すると後半のどこかで
    `PoolCursor.take()` が `ValueError` を投げてそのセルが失敗する
    (run.py はブロック単位で捕まえて続行するが、`--dry-run` の時点で
    気づけた方がよい)。
    """
    demand: dict[str, int] = {}
    seen: set[tuple[str, int | None, str]] = set()
    for b in blocks:
        key = (b.scenario_name, b.ctx, b.axis_tag)
        if b.scenario_name != "point" or b.ctx in (None, 0) or key in seen:
            continue
        seen.add(key)
        win = max(b.ctx - 200, 16)
        for cell in b.cells:
            demand[cell.pool] = demand.get(cell.pool, 0) + win * cell.reps
    rows = []
    for pool, need in sorted(demand.items()):
        budget = POOL_TOKEN_BUDGET.get(pool)
        over = budget is not None and need > budget
        rows.append(dict(pool=pool, need_tokens=need, budget_tokens=budget,
                         over_budget=over))
    return rows


# ── 残留プロセス検査 (実行時のみ。dry-run では絶対に呼ばない) ────────────

def check_residual_processes(patterns: list[str]) -> list[str]:
    """`pkill` 対象になりうるプロセスと `vm.swapusage` を確認する。

    NEXT-SESSION-PROMPT.md「守ること」2. の実装: サーバー停止後に
    Metal のメモリを握ったまま残るプロセスが無いか、スワップが
    異常に膨らんでいないかを見る。警告文字列のリストを返すだけで、
    実際に kill するかは呼び出し側 (run.py の実行ループ) が判断する。
    """
    warnings: list[str] = []
    for pat in patterns:
        try:
            out = subprocess.run(["pgrep", "-fl", pat], capture_output=True,
                                 text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                warnings.append(f"残留プロセスの疑い ({pat}):\n{out.stdout.strip()}")
        except (OSError, subprocess.TimeoutExpired) as e:
            warnings.append(f"pgrep 失敗 ({pat}): {e}")
    try:
        out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                             text=True, timeout=10)
        if out.returncode == 0:
            warnings.append(f"[参考] {out.stdout.strip()}")
    except (OSError, subprocess.TimeoutExpired) as e:
        warnings.append(f"vm.swapusage 取得失敗: {e}")
    return warnings


# ── dry-run 表示 ─────────────────────────────────────────────────────

def print_plan(blocks: list[Block], seed: int, cooldown: float, tier: str,
              out_dir: Path) -> dict:
    """コマンド列・順序・プール残量・推定所要時間を表示する。**副作用は無い**
    (サーバー起動・HTTP・ファイル書き込みはこの関数の外)。
    """
    engines = default_engines()
    print(f"=== ベンチ計画 (dry-run, tier={tier}, seed={seed}) ===")
    print(TIERS[tier]["summary"])
    print(f"ブロック数: {len(blocks)}  冷却: {cooldown:.0f}s  出力先: {out_dir}\n")

    demand_rows = pool_demand_report(blocks)
    if demand_rows:
        print("--- プール残量チェック (実測トークン数の予算に対して) ---")
        for r in demand_rows:
            mark = "OVER" if r["over_budget"] else "OK"
            print(f"  {r['pool']:16s} 必要 {r['need_tokens']:>9d} tok  "
                 f"予算 {r['budget_tokens']:>9} tok  [{mark}]")
        if any(r["over_budget"] for r in demand_rows):
            print("  ※ OVER のプールは実行中に PoolCursor.take() が"
                 " ValueError で止まる可能性がある"
                 " (繰り返しで埋めない設計を守っているため。"
                 " --ctxs/--pools/--reps を絞ること)。")
        print()

    total = 0.0
    unknown = 0
    rows = []
    for b in blocks:
        adapter = engines.get(b.engine_kind)
        if adapter is None:
            avail, reason = False, f"未登録エンジン (候補: {list(ENGINE_REGISTRY)})"
        else:
            avail = adapter.is_available()
            reason = adapter.unavailable_reason()
        est = estimate_block_seconds(b.engine_kind, b.ctx, b.cells, cooldown)
        if est["total_s"] is None:
            unknown += 1
        else:
            total += est["total_s"]
        if adapter is None:
            argv = ["<未登録エンジン>"]
        elif avail:
            argv = adapter.build_argv(DEFAULT_HOST,
                                      ENGINE_PORTS.get(b.engine_kind, 8200))
        else:
            argv = [f"<未実装: {reason}>"]
        avail_mark = "OK" if avail else f"SKIP ({reason})"
        print(f"[{b.index:03d}] {b.label():48s} 見積={_fmt_s(est['total_s'])}"
             f"  [{avail_mark}]")
        print(f"       argv: {' '.join(argv)}")
        cell_labels = [c.label() + (f" x{c.reps}" if c.reps != 1 else "") for c in b.cells]
        shown = cell_labels[:6]
        more = f" (+{len(cell_labels) - 6})" if len(cell_labels) > 6 else ""
        print(f"       cells: {shown}{more}")
        rows.append(dict(index=b.index, scenario=b.scenario_name, ctx=b.ctx,
                         axis_tag=b.axis_tag, engine=b.engine_kind, argv=argv,
                         cells=[dict(pool=c.pool, tokens=c.tokens,
                                    thinking=c.thinking, reps=c.reps,
                                    extra=dict(c.extra_kwargs)) for c in b.cells],
                         available=avail, unavailable_reason=reason,
                         estimate=est))

    print(f"\n合計推定 (較正できたブロックのみ): {_fmt_s(total)}"
         f"  (較正データ無しのブロック: {unknown} 件、時間見積もりから除外)")
    print("見積もりは 2026-09-02 の実測 (SESSION-2026-09-02-CATCHUP.md) からの"
         " 線形補間であり、実測ではない。熱・残留プロセス・機体差で外れる。")
    return dict(seed=seed, cooldown=cooldown, tier=tier, out_dir=str(out_dir),
               total_estimate_s=total, unknown_blocks=unknown,
               pool_demand=demand_rows, blocks=rows)


def _fmt_s(s: float | None) -> str:
    if s is None:
        return "?"
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


# ── 実行 (GPU を使う。dry-run では呼ばれない) ────────────────────────

def run_block(block: Block, adapter: EngineAdapter, thinking_map: dict,
             server_log_dir: Path) -> dict:
    """1 ブロックを実際に走らせる。起動 → ウォームアップ → 全セル x `reps`
    回の測定 → 停止 → 残留プロセス確認、まで面倒を見る。

    冷 TTFT の信頼できる値は **ブロック内で最初に来るセルの rep=0**
    (フレッシュ起動直後の最初のリクエスト) だけである。それ以外はサーバーは
    起動済みだが GPU が温まっていくので、絶対値としては当てにならない
    (`docs/research/BENCH-DESIGN-2026-09.md` (i) 節)。`report.py` が
    `is_fresh_boot` でこれを見分ける。
    """
    from transformers import AutoTokenizer  # 遅延 import (dry-run を汚さない)
    from scenarios import MaterializeCtx, PoolRegistry

    port = ENGINE_PORTS.get(block.engine_kind, 8200)
    log_path = server_log_dir / f"{block.index:03d}-{block.engine_kind}.log"

    rng = random.Random(block.block_seed)
    tok = AutoTokenizer.from_pretrained(_tokenizer_source(adapter))
    pools = PoolRegistry(tok)

    cells_out = []
    is_first_measurement = True
    with Server(adapter.name, adapter.build_argv(DEFAULT_HOST, port), port,
               log_path=str(log_path)):
        mid = adapter.model_id(port)
        # ウォームアップ 1 段目 (短文、カーネルの初回コンパイル)
        stream_with_usage(port, [{"role": "user", "content": "こんにちは。"}],
                          8, mid, extra_body=thinking_map["off"])
        for cell in block.cells:
            kwargs = dict(ctx=block.ctx, tokens=cell.tokens, pool=cell.pool,
                          **cell.kwargs())
            scenario = build_scenario(block.scenario_name, **kwargs)
            thinking_extra = thinking_map[cell.thinking]
            reps_out = []
            for r in range(cell.reps):
                mctx = MaterializeCtx(tok=tok, pools=pools, rng=rng,
                                      target_ctx=block.ctx or 0)
                history: list[dict] = []
                turn_results = []
                for turn in scenario.turns:
                    content = turn.content_fn(mctx)
                    if turn.reset_history:
                        history = []
                    messages = history + [{"role": "user", "content": content}]
                    ttft, dec_s, n, reply, usage = stream_with_usage(
                        port, messages, turn.max_tokens, mid,
                        extra_body=thinking_extra)
                    turn_results.append(dict(
                        label=turn.label, ttft_s=ttft, decode_s=dec_s,
                        n_tokens=n, cached_tokens=(usage or {})
                        .get("prompt_tokens_details", {}).get("cached_tokens"),
                        reply_head=reply[:160]))
                    history = messages + [{"role": "assistant", "content": reply}]
                reps_out.append(dict(rep=r, is_fresh_boot=is_first_measurement and r == 0,
                                     turns=turn_results))
                is_first_measurement = False
            cells_out.append(dict(pool=cell.pool, tokens=cell.tokens,
                                  thinking=cell.thinking, extra=dict(cell.extra_kwargs),
                                  scenario=scenario.name, reps=reps_out))
        warnings = check_residual_processes(
            [adapter.name, "mlxturbo.server", "mlx-serve --serve"])

    return dict(block=dict(index=block.index, scenario_name=block.scenario_name,
                           engine_kind=block.engine_kind, ctx=block.ctx,
                           axis_tag=block.axis_tag),
               cells=cells_out, residual_warnings=warnings,
               server_log=str(log_path))


def _tokenizer_source(adapter: EngineAdapter) -> str:
    import os
    if isinstance(adapter, MlxturboAdapter):
        return os.path.expanduser(adapter.model)
    if isinstance(adapter, MlxServeAdapter):
        return os.path.expanduser(adapter.model)
    raise NotImplementedError(
        "このアダプタ種別のトークナイザ取得元が未定義 (engines.py を見て追加すること)")


# ── CLI ───────────────────────────────────────────────────────────────

def _resolve_axes(args) -> tuple[AxisConfig, ...]:
    """`--pools`/`--tokens-set`/`--thinking-set`/`--ctxs` が明示されたら、
    tier の中身を無視して単一の `AxisConfig` に差し替える (「自分で軸を
    指定した」という明確な意思表示として扱う)。何も指定しなければ
    `--tier` のプリセットをそのまま使う。
    """
    overridden = any(v is not None for v in
                     (args.pools, args.tokens_set, args.thinking_set, args.ctxs))
    if not overridden:
        return TIERS[args.tier]["axes"]
    base = TIERS[args.tier]["axes"][0]
    return (AxisConfig(
        ctxs=tuple(int(c) for c in args.ctxs.split(",")) if args.ctxs else base.ctxs,
        pools=tuple(args.pools.split(",")) if args.pools else base.pools,
        tokens_set=tuple(int(t) for t in args.tokens_set.split(","))
        if args.tokens_set else base.tokens_set,
        thinking_set=tuple(args.thinking_set.split(","))
        if args.thinking_set else base.thinking_set,
        reps=args.reps if args.reps is not None else base.reps,
        tag="custom",
    ),)


def _load_completed_indices(raw_path: Path) -> set[int]:
    done = set()
    if not raw_path.exists():
        return done
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # 中断で壊れた最終行は無視 (その分は再実行される)
            done.add(rec["block"]["index"])
    return done


def main() -> int:
    install_term_handler()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="point",
                    help=f"カンマ区切り。候補: {','.join(SCENARIO_NAMES)}"
                         " (既定 point のみ)")
    ap.add_argument("--engines", default="mlxturbo,mlx-serve",
                    help=f"カンマ区切り。候補: {','.join(ENGINE_REGISTRY)}")
    ap.add_argument("--tier", choices=tuple(TIERS), default="quick",
                    help="池 x 出力長 x thinking の掃引プリセット"
                         " (quick/standard/overnight)。既定 quick")
    ap.add_argument("--ctxs", default=None,
                    help="point の文脈トークン数リスト。指定すると tier の"
                         " 軸を上書きする (未指定なら tier の既定)")
    ap.add_argument("--pools", default=None,
                    help=f"point が振るプールのカンマ区切り。候補: default,"
                         f"{','.join(POOL_ORDER)}。指定すると tier を上書きする")
    ap.add_argument("--tokens-set", default=None,
                    help="point の生成トークン数のカンマ区切り (例 128,1024)。"
                         " 指定すると tier を上書きする")
    ap.add_argument("--thinking-set", default=None,
                    help="point の thinking off/on のカンマ区切り (例 off,on)。"
                         " 指定すると tier を上書きする")
    ap.add_argument("--tokens", type=int, default=512,
                    help="point 以外のシナリオ (agent/code-edit/rag/parallel)"
                         " の生成トークン数。各シナリオは指定が無ければ自分の"
                         " 既定値を使う (この値は明示指定があるときだけ使う)")
    ap.add_argument("--reps", type=int, default=None,
                    help="ブロックあたりの反復。指定すると tier の既定 reps"
                         " を上書きする (未指定なら tier の既定: quick=3,"
                         " standard=1, overnight=3)")
    ap.add_argument("--cooldown", type=float, default=None,
                    help="ブロック間の冷却秒数。既定は tier の値 (今のところ"
                         " どの tier も 180 秒)")
    ap.add_argument("--seed", type=int, default=None,
                    help="ブロック順のシャッフルに使う乱数種。省略時は生成して表示する"
                         " (dry-run を再現したいときはここに渡し直す)")
    ap.add_argument("--out-dir", default=None,
                    help="既定: bench/results/suite/<timestamp>/")
    ap.add_argument("--resume", default=None, metavar="OUT_DIR",
                    help="既存の out-dir を指定して中断から再開する。"
                         " plan.json からシード/tier/軸を復元し、raw.jsonl に"
                         " 記録済みのブロックはスキップする")
    ap.add_argument("--thinking", choices=("off", "on", "default"), default="off",
                    help="point 以外のシナリオの thinking (point は"
                         " --thinking-set / tier で決める)")
    ap.add_argument("--dry-run", action="store_true",
                    help="コマンド列・順序・プール残量・所要時間見積もりを"
                         " 表示するだけ。サーバー起動・HTTP・モデル読み込みを"
                         " 一切行わない")
    args = ap.parse_args()

    thinking_map = {
        "off": {"reasoning_effort": "none"},
        "on": {"reasoning_effort": "medium"},
        "default": None,
    }

    if args.resume:
        out_dir = Path(args.resume)
        plan_path = out_dir / "plan.json"
        if not plan_path.exists():
            print(f"--resume: {plan_path} が無い (このディレクトリは"
                 " run.py が作ったものではない?)", file=sys.stderr)
            return 2
        saved = json.loads(plan_path.read_text())
        seed = saved["seed"]
        tier = saved["tier"]
        args.scenarios = saved["scenario_names"]
        args.engines = saved["engine_kinds"]
        axes = tuple(AxisConfig(**a) for a in saved["axes"])
        cooldown = saved["cooldown"]
        scenario_names = args.scenarios
        engine_kinds = args.engines
        print(f"--resume: {out_dir} を復元 (tier={tier}, seed={seed})")
    else:
        seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 30)
        scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        for s in scenario_names:
            if s not in SCENARIO_NAMES:
                print(f"未知のシナリオ: {s} (候補: {SCENARIO_NAMES})", file=sys.stderr)
                return 2
        engine_kinds = [e.strip() for e in args.engines.split(",") if e.strip()]
        for e in engine_kinds:
            if e not in ENGINE_REGISTRY:
                print(f"未知のエンジン: {e} (候補: {list(ENGINE_REGISTRY)})", file=sys.stderr)
                return 2
        for key in (args.pools.split(",") if args.pools else ()):
            if key not in PROMPT_POOLS:
                print(f"未知のプール: {key} (候補: {list(PROMPT_POOLS)})", file=sys.stderr)
                return 2
        axes = _resolve_axes(args)
        cooldown = args.cooldown if args.cooldown is not None else TIERS[args.tier]["cooldown"]
        tier = args.tier
        out_dir = Path(args.out_dir) if args.out_dir else (
            REPO_ROOT / "bench" / "results" / "suite" / time.strftime("%Y%m%d-%H%M%S"))

    flat = expand_cells(scenario_names, axes, args.tokens, args.thinking)
    blocks = group_into_blocks(flat, engine_kinds, seed)

    if args.dry_run:
        plan = print_plan(blocks, seed, cooldown, tier, out_dir)
        plan["scenario_names"] = scenario_names
        plan["engine_kinds"] = engine_kinds
        plan["axes"] = [dict(ctxs=list(a.ctxs), pools=list(a.pools),
                             tokens_set=list(a.tokens_set),
                             thinking_set=list(a.thinking_set), reps=a.reps,
                             tag=a.tag)
                        for a in axes]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2))
        print(f"\n計画を書き出した (dry-run。何も起動していない): {out_dir / 'plan.json'}")
        return 0

    # ── ここから先は GPU を使う実行 (このエージェントは呼ばない) ──────
    engines = default_engines()
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log_dir = out_dir / "logs"
    server_log_dir.mkdir(exist_ok=True)
    plan = print_plan(blocks, seed, cooldown, tier, out_dir)
    plan["scenario_names"] = scenario_names
    plan["engine_kinds"] = engine_kinds
    plan["axes"] = [dict(ctxs=list(a.ctxs), pools=list(a.pools),
                         tokens_set=list(a.tokens_set),
                         thinking_set=list(a.thinking_set), reps=a.reps,
                         tag=a.tag)
                    for a in axes]
    if not args.resume:
        (out_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))

    raw_path = out_dir / "raw.jsonl"
    already_done = _load_completed_indices(raw_path)
    if already_done:
        print(f"再開: {len(already_done)}/{len(blocks)} ブロックは記録済み。スキップする")

    for block in blocks:
        if block.index in already_done:
            continue
        adapter = engines.get(block.engine_kind)
        if adapter is None or not adapter.is_available():
            reason = adapter.unavailable_reason() if adapter else "未登録エンジン"
            print(f"[{block.index:03d}] SKIP {block.label()}: {reason}")
            continue
        print(f"[{block.index:03d}] 実行: {block.label()}", flush=True)
        try:
            result = run_block(block, adapter, thinking_map, server_log_dir)
        except Exception as e:  # noqa: BLE001 — 一晩走らせるジョブを 1 ブロックの
            # 例外 (プール枯渇の ValueError 等) で丸ごと落とさない。失敗を
            # 記録して次のブロックへ進む (`--resume` で後から追える)。
            print(f"  [失敗] {block.label()}: {e!r}")
            with open(raw_path, "a") as f:
                f.write(json.dumps(dict(
                    block=dict(index=block.index, scenario_name=block.scenario_name,
                              engine_kind=block.engine_kind, ctx=block.ctx,
                              axis_tag=block.axis_tag),
                    failed=True, error=repr(e)), ensure_ascii=False) + "\n")
            continue
        with open(raw_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        if result["residual_warnings"]:
            for w in result["residual_warnings"]:
                print(f"  [警告] {w}")
        if block.index != blocks[-1].index:
            print(f"  冷却 {cooldown:.0f}s...", flush=True)
            time.sleep(cooldown)

    print(f"\n書き出し: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
