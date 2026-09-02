"""`run.py` が書き出した生ログ (`raw.jsonl`) を Markdown の表にする。

集計方針 (`docs/research/BENCH-DESIGN-2026-09.md` (b)(c)(g) 節が仕様):

- **冷 TTFT の見出し数字はブロック内で最初に来たセルの rep=0 (フレッシュ
  起動直後) だけ**を使う。ブロック内の他のセル・rep>=1 は GPU が温まって
  いくため、絶対値としては当てにならない (CLAUDE.md の熱の作法)。
- 各条件 (シナリオ x 文脈 x プール x 出力長 x thinking x エンジン) について
  **min / p50 / p95 / max と変動係数 (CV = 標本標準偏差 / 平均)** を出す。
  **反復が 1-2 回の条件は「分布なし」と明記する** (3 点未満で percentile を
  語るのは統計として無理がある — 数字は出すが「分布」とは呼ばない)。
- decode tok/s は `(n-1)/経過` を rep ごとに計算し、その分布を報告する。
- `usage.prompt_tokens_details.cached_tokens` が「冷」ターンで 0 でなければ、
  その rep は接頭辞キャッシュに当たっていた = 真の意味で冷えていない。
  headline から除外し、警告として明記する (無かったことにしない)。
- **池間のばらつきを別表で出す** (`pool_variance_table`)。同じ
  (シナリオ, 文脈, 出力長, thinking, エンジン) で池だけ変えたときの
  decode tok/s と冷 TTFT の中央値の散らばりを見て、`(max-min)/中央値` が
  10% を超えたら印を付ける。**池を混ぜた平均だけを出すことはしない**
  (投機デコードの受理率はテキストの性質で大きく動くので、平均は片方の
  プール分布に有利不利が出ても隠してしまう)。
- **数字が無い欄は「未計測」と出す** — 空欄や 0 で埋めない (無いものを
  埋めない、CLAUDE.md/BENCH-DESIGN 共通の規律)。

`--out` は詳細版 (この docstring の集計方針をそのまま出す、内部向け診断用)。
`--markdown` は公開版 (`docs/BENCHMARKS-2026-09.md` に貼る想定の「大本営発表」
下地) — 見出し表 / 池ごとの差 / 出力長 x thinking / 分布 / 手順と環境 /
注意書きの 6 節に分け、**そのデータが無い節は表を省略してその旨を明記する**
(quick tier だけで作った raw.jsonl なら (b)(c) が省略される、という具合)。

`from_llmprobe_json` は `~/dev/mlx-serve/tests/bench.sh` が使う
`llmprobe --bench-only --save <path>.json` の出力を、この報告と同じ行の
形に正規化する。mlxturbo が対応しないモデル (Gemma 4 系など) を、相手側の
自己申告値と並べて出すための互換列。
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def collect_environment_meta() -> dict:
    """機種・OS・mlx バージョン・両リポジトリのコミットハッシュ。

    公開物には必ずこれを添える (docs/BENCHMARKS.md が既にやっている
    「計測環境（共通）」欄と同じ理由 — ハードウェア世代が変わると数字も
    変わる)。GPU は使わない (すべて CLI 問い合わせ)。
    """
    def _run(argv: list[str]) -> str:
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else f"<失敗: {out.stderr.strip()}>"
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"<取得失敗: {e}>"

    def _git_commit(repo: Path) -> str:
        if not repo.exists():
            return "<リポジトリなし>"
        return _run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"])

    return dict(
        machine=_run(["sysctl", "-n", "hw.model"]),
        macos=_run(["sw_vers", "-productVersion"]),
        mlxturbo_commit=_git_commit(REPO_ROOT),
        mlx_serve_commit=_git_commit(Path.home() / "dev" / "mlx-serve"),
        # mlx のバージョンはトークナイザ同様に軽いが、import mlx.core は
        # Metal デバイスへ触れるので、ここでは pip 経由の静的な問い合わせに
        # 留める (GPU に触らない)。
        mlx_version=_run([sys.executable, "-c",
                          "import importlib.metadata as m; print(m.version('mlx'))"]),
    )


def load_raw(path: Path) -> list[dict]:
    """`raw.jsonl` (1 ブロック 1 行) を読む。`--resume` で中断・再開した
    ジョブでも、最後の行が壊れていれば無視して読み進める (途中で
    プロセスが落ちたときの半端な行に備える)。
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_plan(path: Path) -> dict | None:
    """`run.py` が書く `plan.json` (実行順・シード・tier・軸・argv) を読む。

    無ければ None を返すだけで、呼び出し側 (公開版レポート) はそれを
    「未計測」の根拠として扱う — 無いことをエラーにしない。
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# ── 統計 ─────────────────────────────────────────────────────────────

def _percentile(sorted_vals: list[float], p: float) -> float:
    """線形補間の分位点 (0<=p<=1)。標本 1 点なら常にその値を返す。"""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_vals[0]
    idx = p * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _stats(samples: list[float]) -> dict:
    """min/p50/p95/max と変動係数 (CV)。**反復 3 回未満は `has_distribution`
    を False にする** — 数字は出すが「分布」を語らないための印。
    """
    vals = sorted(v for v in samples if v == v)  # NaN 除去
    n = len(vals)
    if n == 0:
        return dict(n=0, min=None, p50=None, p95=None, max=None, cv=None,
                   has_distribution=False)
    mean = statistics.mean(vals)
    cv = (statistics.pstdev(vals) / mean) if (n >= 2 and mean) else None
    return dict(n=n, min=vals[0], p50=_percentile(vals, 0.5),
               p95=_percentile(vals, 0.95), max=vals[-1], cv=cv,
               has_distribution=n >= 3)


def _cold_headline(reps: list[dict]) -> dict:
    """ブロック内で最初に来たセルの rep=0 (フレッシュ起動直後) の
    "cold" ラベルターンを取り出す。ブロック内の他セルにはこれが無い
    (`run_block` がブロックあたり 1 回しか `is_fresh_boot=True` を立てない)。
    """
    for rep in reps:
        if rep.get("is_fresh_boot"):
            for turn in rep.get("turns", []):
                if turn["label"] == "cold":
                    return turn
    return {}


def _decode_tps(turn: dict) -> float:
    n, dec_s = turn.get("n_tokens", 0), turn.get("decode_s", 0.0)
    if dec_s and dec_s > 0 and n and n > 1:
        return (n - 1) / dec_s
    return float("nan")


AggKey = tuple  # (scenario, ctx, pool, tokens, thinking)


def aggregate(raw: list[dict]) -> dict[AggKey, dict[str, dict]]:
    """`(シナリオ, 文脈, プール, 出力長, thinking)` ごとに、エンジン別の
    集計行を作る。`run.py` が記録した失敗ブロック (`failed: true`、プール
    枯渇の `ValueError` 等) は集計から外す (無かったことにはしないが、
    数字としては混ぜない — `failures` に理由を残す)。
    """
    out: dict[AggKey, dict[str, dict]] = {}
    failures: list[dict] = []
    for block_result in raw:
        b = block_result["block"]
        if block_result.get("failed"):
            failures.append(dict(scenario=b["scenario_name"], ctx=b["ctx"],
                                 engine=b["engine_kind"],
                                 error=block_result.get("error")))
            continue
        for cell in block_result.get("cells", []):
            key = (b["scenario_name"], b["ctx"], cell["pool"], cell["tokens"],
                  cell["thinking"])
            row = out.setdefault(key, {})
            reps = cell["reps"]
            cold = _cold_headline(reps)
            all_cold_ttft = [t["ttft_s"] for rep in reps for t in rep["turns"]
                             if t["label"] == "cold"]
            all_warm_ttft = [t["ttft_s"] for rep in reps for t in rep["turns"]
                             if t["label"] == "warm"]
            # decode tok/s は本文の生成 (実質的な各ターン) だけから取る。
            # point の "warm" ターンは温 TTFT を測るためだけの 8 トークンの
            # 埋め草で、これを混ぜると decode の分布に n=8 の極端に短い
            # サンプルが混入し、CV や p95 を歪める
            # (bench/self_snapshot.py も warm 側は decode tok/s に数えない
            # のと同じ convention — 実測で気づいた不整合をここで揃える)。
            decode_samples = [_decode_tps(t) for rep in reps for t in rep["turns"]
                              if t["label"] != "warm"]
            cold_cache_hits = [t.get("cached_tokens") for rep in reps
                              for t in rep["turns"]
                              if t["label"] == "cold" and t.get("cached_tokens")]
            row[b["engine_kind"]] = dict(
                cold_ttft_fresh_s=cold.get("ttft_s"),
                cold_ttft_stats=_stats(all_cold_ttft),
                warm_ttft_stats=_stats(all_warm_ttft),
                decode_tps_stats=_stats(decode_samples),
                reps=len(reps),
                cold_cache_hit_warning=(
                    f"「冷」ターンで cached_tokens>0 が {len(cold_cache_hits)} 件"
                    f" (値: {cold_cache_hits}) — 真に冷えていなかった可能性"
                    if cold_cache_hits else None),
                residual_warnings=block_result.get("residual_warnings") or [],
            )
    out["__failures__"] = failures  # type: ignore[assignment]
    return out


def pool_variance_table(agg: dict[AggKey, dict[str, dict]]) -> list[dict]:
    """同じ (シナリオ, 文脈, 出力長, thinking, エンジン) で池だけ変えたときの
    ばらつきを見る。**池を混ぜた平均は出さない** — ここは個々のプールの
    中央値をそのまま並べ、`(max-min)/中央値` が 10% を超えたものに印を付ける。
    """
    groups: dict[tuple, dict[str, dict]] = {}
    for key, engines in agg.items():
        if key == "__failures__":
            continue
        scenario, ctx, pool, tokens, thinking = key
        for engine, row in engines.items():
            gkey = (scenario, ctx, tokens, thinking, engine)
            groups.setdefault(gkey, {})[pool] = row
    rows = []
    for gkey, by_pool in groups.items():
        if len(by_pool) < 2:
            continue  # プールが 1 つしか無ければ「ばらつき」は語れない
        for metric, getter in (
            ("decode_tps_p50", lambda r: r["decode_tps_stats"]["p50"]),
            ("cold_ttft_p50", lambda r: r["cold_ttft_stats"]["p50"]),
        ):
            vals = {p: getter(r) for p, r in by_pool.items() if getter(r) is not None}
            if len(vals) < 2:
                continue
            lo, hi = min(vals.values()), max(vals.values())
            med = statistics.median(vals.values())
            spread = (hi - lo) / med if med else 0.0
            rows.append(dict(
                scenario=gkey[0], ctx=gkey[1], tokens=gkey[2], thinking=gkey[3],
                engine=gkey[4], metric=metric, min=lo, max=hi, median=med,
                spread_pct=spread * 100, flagged=spread > 0.10,
                per_pool={p: round(v, 3) for p, v in vals.items()}))
    return rows


def from_llmprobe_json(path: Path, engine_label: str) -> dict:
    """mlx-serve の `tests/bench.sh` / `llmprobe --bench-only --save` 出力を
    この報告のスキーマに正規化する。

    llmprobe の JSON は `bench.{decodeTokPerSec,prefillTokPerSec}.median` と
    `bench.speculative.tokensPerStep` を持つ (`~/dev/mlx-serve/tests/bench.sh`
    末尾の Python 抽出コードを参照)。**cold/warm TTFT の区別は無く、
    投機モードはサーバー自身のログを別途読まないと分からない** — 対応する
    列は None のままにし、無いものを埋めない。
    """
    data = json.loads(path.read_text())
    bench = (data or {}).get("bench") or {}
    decode = (bench.get("decodeTokPerSec") or {}).get("median")
    prefill = (bench.get("prefillTokPerSec") or {}).get("median")
    tps_step = (bench.get("speculative") or {}).get("tokensPerStep")
    return dict(engine=engine_label, source="llmprobe --bench-only",
               decode_tps_median=decode, prefill_tps_median=prefill,
               tokens_per_step=tps_step,
               cold_ttft_fresh_s=None, warm_ttft_median_s=None,
               note="llmprobe 由来: cold/warm TTFT の区別なし")


def _fmt(v) -> str:
    """**数字が無ければ「未計測」** — 空欄や 0 で埋めない。"""
    if v is None:
        return "未計測"
    if isinstance(v, float):
        if v != v:  # NaN
            return "未計測"
        return f"{v:.2f}"
    return str(v)


def _main_table_lines(agg: dict[AggKey, dict[str, dict]]) -> list[str]:
    """詳細版・公開版の両方が使う「分布」表 (min/p50/p95/max/CV)。"""
    header = ("シナリオ", "文脈", "プール", "出力長", "think", "エンジン",
             "冷TTFT(fresh)", "冷TTFT p50/p95", "decode p50/p95", "CV(decode)",
             "reps")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    entries = [(k, v) for k, v in agg.items() if k != "__failures__"]
    if not entries:
        return ["_(データが無い)_"]
    for key, engines in sorted(
        entries, key=lambda kv: (kv[0][0], kv[0][1] or -1, kv[0][2], kv[0][3], kv[0][4]),
    ):
        scenario, ctx, pool, tokens, thinking = key
        for eng, row in sorted(engines.items()):
            cts = row["cold_ttft_stats"]
            dts = row["decode_tps_stats"]
            dist_note = "" if cts["has_distribution"] else " (分布なし)"
            lines.append("| " + " | ".join(str(v) for v in (
                scenario, ctx if ctx is not None else "-", pool, tokens, thinking,
                eng, _fmt(row["cold_ttft_fresh_s"]),
                f"{_fmt(cts['p50'])}/{_fmt(cts['p95'])}{dist_note}",
                f"{_fmt(dts['p50'])}/{_fmt(dts['p95'])}",
                _fmt(dts["cv"]), row["reps"])) + " |")
            if row.get("cold_cache_hit_warning"):
                lines.append(f"| | | | | | | **警告**: {row['cold_cache_hit_warning']} ||||")
            for w in row.get("residual_warnings") or []:
                lines.append(f"| | | | | | | **警告 (残留プロセス)**: {w} ||||")
    return lines


def render_markdown(agg: dict[AggKey, dict[str, dict]], meta: dict | None = None,
                    extra_rows: list[dict] | None = None) -> str:
    """詳細版 (内部診断用)。この docstring 冒頭の集計方針をそのまま出す。"""
    lines = ["# ベンチ結果 (詳細版)", ""]
    if meta:
        lines.append("計測環境: " + ", ".join(f"{k}={v}" for k, v in meta.items()))
        lines.append("")
    lines.append("見出し数字はブロック内で最初に来たセルの rep=0"
                 " (フレッシュ起動直後) のみ。他のセル・rep は GPU が"
                 " 温まった状態での分布として別列に出す。**反復 1-2 回の"
                 " 条件は「分布なし」** (percentile を出すには足りない)。")
    lines.append("")
    lines += _main_table_lines(agg)

    failures = agg.get("__failures__") or []
    if failures:
        lines.append("")
        lines.append("## 失敗したブロック (集計から除外)")
        lines.append("")
        lines.append("| シナリオ | 文脈 | エンジン | エラー |")
        lines.append("|---|---|---|---|")
        for f in failures:
            lines.append(f"| {f['scenario']} | {f['ctx']} | {f['engine']} | {f['error']} |")

    pv = pool_variance_table(agg)
    if pv:
        lines.append("")
        lines.append("## 池間のばらつき (10% を超える差に印)")
        lines.append("")
        lines.append("池を混ぜた平均ではなく、各条件で池ごとの中央値をそのまま並べる。"
                     " `spread%` = (max-min)/中央値。")
        lines.append("")
        lines.append("| シナリオ | 文脈 | 出力長 | think | エンジン | 指標 |"
                     " min | 中央値 | max | spread% | 印 | 池ごとの値 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(pv, key=lambda r: -r["spread_pct"]):
            mark = "**>10%**" if r["flagged"] else ""
            lines.append("| " + " | ".join(str(v) for v in (
                r["scenario"], r["ctx"], r["tokens"], r["thinking"], r["engine"],
                r["metric"], _fmt(r["min"]), _fmt(r["median"]), _fmt(r["max"]),
                f"{r['spread_pct']:.1f}", mark, r["per_pool"])) + " |")

    if extra_rows:
        lines.append("")
        lines.append("## 互換列 (llmprobe --bench-only など、他ツール由来)")
        lines.append("")
        lines.append("| エンジン | 出典 | decode(中央値) | prefill(中央値) | tok/step |")
        lines.append("|---|---|---|---|---|")
        for r in extra_rows:
            lines.append("| " + " | ".join(str(v) for v in (
                r["engine"], r["source"], _fmt(r.get("decode_tps_median")),
                _fmt(r.get("prefill_tps_median")), _fmt(r.get("tokens_per_step")))) + " |")
    return "\n".join(lines) + "\n"


# ── 公開版 (「大本営発表」下地。docs/BENCHMARKS-2026-09.md に貼る想定) ───

def headline_table(agg: dict[AggKey, dict[str, dict]]) -> tuple[bool, list[str]]:
    """(a) 見出し表: プール `default`、thinking off、既定の出力長で
    エンジン x 文脈点の冷/温 TTFT と decode tok/s を並べる。

    `default` プールは `quick` tier (と `overnight` のラダー軸) だけが作る
    ので、`standard` tier 単体の raw.jsonl ではこの表は無い — その場合は
    省略してその旨を書く (無いものを埋めない)。
    """
    candidates: dict[int, set] = {}
    for key in agg:
        if key == "__failures__":
            continue
        scenario, ctx, pool, tokens, thinking = key
        if scenario == "point" and pool == "default" and thinking == "off":
            candidates.setdefault(tokens, set()).add(ctx)
    if not candidates:
        return False, ["_(省略: プール `default` x thinking off の `point` "
                       "データが無い。quick tier または overnight tier の"
                       "ラダー軸を含む raw.jsonl が要る)_"]
    tokens = max(candidates, key=lambda t: (len(candidates[t]), -t))
    ctxs = sorted((c for c in candidates[tokens] if c is not None),
                 key=lambda c: c)
    if 0 in candidates[tokens]:
        ctxs = [0] + ctxs if 0 not in ctxs else ctxs
    lines = [f"プール `default`、出力長 {tokens}、thinking off。",
            "", "| 文脈 | エンジン | 冷 TTFT (fresh) | 温 TTFT (p50) |"
                " decode tok/s (p50) | reps |",
            "|---|---|---|---|---|---|"]
    for ctx in sorted(candidates[tokens], key=lambda c: (c is None, c)):
        key = ("point", ctx, "default", tokens, "off")
        row = agg.get(key, {})
        for eng in sorted(row):
            r = row[eng]
            lines.append(f"| {ctx} | {eng} | {_fmt(r['cold_ttft_fresh_s'])} | "
                         f"{_fmt(r['warm_ttft_stats']['p50'])} | "
                         f"{_fmt(r['decode_tps_stats']['p50'])} | {r['reps']} |")
    return True, lines


def pool_table(agg: dict[AggKey, dict[str, dict]]) -> tuple[bool, list[str]]:
    """(b) 池ごとの差。`pool_variance_table` の各グループを、プール別 1 行
    ずつに展開した読みやすい表にする (dict をセルに詰めない)。
    """
    pv = pool_variance_table(agg)
    if not pv:
        return False, ["_(省略: 同一条件で 2 種以上のプールを比較できる"
                       "データが無い。standard/overnight tier が要る)_"]
    lines = ["同じ条件でプールだけ変えたときの値。`spread%` = "
            "(max-min)/中央値、10% 超に印。**池を混ぜた平均は出さない**"
            " (`(c)` 節参照)。", "",
            "| シナリオ | 文脈 | 出力長 | think | エンジン | 指標 | プール |"
            " 値 | spread% | 印 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(pv, key=lambda r: -r["spread_pct"]):
        mark = "**>10%**" if r["flagged"] else ""
        for pool, v in sorted(r["per_pool"].items(), key=lambda kv: kv[1]):
            lines.append(f"| {r['scenario']} | {r['ctx']} | {r['tokens']} |"
                         f" {r['thinking']} | {r['engine']} | {r['metric']} |"
                         f" {pool} | {_fmt(v)} | {r['spread_pct']:.1f} | {mark} |")
    return True, lines


def tokens_thinking_table(agg: dict[AggKey, dict[str, dict]]) -> tuple[bool, list[str]]:
    """(c) 出力長 x thinking。同じ (シナリオ, 文脈, プール, エンジン) で
    出力長・thinking を複数振ったデータがあるときだけ出す (quick tier は
    出力長・thinking とも 1 点しか無いので、この表は省略になる)。
    """
    groups: dict[tuple, dict[tuple, dict]] = {}
    for key, engines in agg.items():
        if key == "__failures__":
            continue
        scenario, ctx, pool, tokens, thinking = key
        for eng, row in engines.items():
            gkey = (scenario, ctx, pool, eng)
            groups.setdefault(gkey, {})[(tokens, thinking)] = row
    usable = {k: v for k, v in groups.items() if len(v) >= 2}
    if not usable:
        return False, ["_(省略: 出力長/thinking を複数振ったデータが無い — "
                       "quick tier は出力長・thinking とも 1 点のみ。"
                       "standard/overnight tier が要る)_"]
    lines = ["| シナリオ | 文脈 | プール | エンジン | 出力長 | think |"
            " 冷TTFT(p50) | decode(p50) |",
            "|---|---|---|---|---|---|---|---|"]
    for gkey in sorted(usable, key=lambda k: (k[0], k[1] or -1, k[2], k[3])):
        scenario, ctx, pool, eng = gkey
        for (tokens, thinking), row in sorted(usable[gkey].items()):
            lines.append(f"| {scenario} | {ctx} | {pool} | {eng} | {tokens} |"
                         f" {thinking} | {_fmt(row['cold_ttft_stats']['p50'])} |"
                         f" {_fmt(row['decode_tps_stats']['p50'])} |")
    return True, lines


_KNOWN_PACKS = {
    "ddalcu-mlxlm": "qwen4_exp 4bit affine group64 (mlxturbo 側、Flash-Next)",
    "ddalcu-flashnext-serve-4bit": "qwen4_exp 4bit affine group64 (mlx-serve 側、Flash-Next)",
}


def _extract_model(argv: list[str] | None) -> str | None:
    if not argv:
        return None
    for i, tok in enumerate(argv):
        if tok == "--model" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _pack_note(model_path: str | None) -> str:
    if not model_path:
        return "未計測"
    for key, note in _KNOWN_PACKS.items():
        if key in model_path:
            return note
    return "未計測 (既知のモデルパスと一致せず、量子化方式を自動判定できない)"


def procedure_and_env_section(plan: dict | None, meta: dict | None) -> list[str]:
    """(e) 手順と環境。`plan.json` (実行順・シード・冷却・argv) と
    `collect_environment_meta()` (機種・OS・mlx バージョン・コミット) を
    合わせる。どちらか無ければ、その項目だけ「未計測」にする。
    """
    lines: list[str] = []
    if plan:
        lines.append(f"- tier: `{plan.get('tier', '未計測')}`")
        lines.append(f"- 乱数種 (`--seed`): `{plan.get('seed', '未計測')}`")
        lines.append(f"- 冷却: {plan.get('cooldown', '未計測')} 秒")
        lines.append("- シナリオ: " + (", ".join(plan.get("scenario_names") or []) or "未計測"))
        lines.append("- エンジン: " + (", ".join(plan.get("engine_kinds") or []) or "未計測"))
        models: dict[str, str] = {}
        for row in plan.get("blocks") or []:
            eng, m = row.get("engine"), _extract_model(row.get("argv"))
            if eng and m and eng not in models:
                models[eng] = m
        if models:
            for eng, m in sorted(models.items()):
                lines.append(f"- {eng} モデル: `{m}` — 量子化: {_pack_note(m)}")
        else:
            lines.append("- モデルパス・量子化: 未計測 (plan.json に argv が無い)")
    else:
        lines.append("- tier・乱数種・冷却・モデル・量子化: 未計測"
                     " (plan.json が見つからなかった。`--plan` で指定するか"
                     " `raw.jsonl` と同じディレクトリに置くこと)")
    if meta:
        lines.append(f"- 機種: {meta.get('machine', '未計測')}")
        lines.append(f"- macOS: {meta.get('macos', '未計測')}")
        lines.append(f"- mlx バージョン: {meta.get('mlx_version', '未計測')}")
        lines.append(f"- mlxturbo コミット: {meta.get('mlxturbo_commit', '未計測')}")
        lines.append(f"- mlx-serve コミット: {meta.get('mlx_serve_commit', '未計測')}")
    else:
        lines.append("- 機種・OS・mlx バージョン・コミット: 未計測"
                     " (`--no-meta` が指定されたか、収集に失敗した)")
    return lines


def caveats_section() -> list[str]:
    """(f) 注意書き。BENCH-DESIGN の (a)(d)(i) 節から、読者が数字を見る前に
    知っておくべき点だけを抜き出した固定文 (データに依存しないので常に出す)。
    """
    return [
        "熱: 連続測定で GPU 温度が上がると同じ文脈でも遅くなる"
        " (実測例: 17k prefill が 50 分の連続稼働で 37→57 秒、"
        " `docs/research/SESSION-2026-09-02-CATCHUP.md`)。見出しの冷 TTFT は"
        " 各ブロックで最初に来たセルの rep=0 のみを使い、それ以外は"
        " 「分布」の節に別掲している。",
        "",
        "接頭辞キャッシュの検知: 「冷」と称した計測で"
        " `usage.prompt_tokens_details.cached_tokens > 0` の rep があれば、"
        " 詳細版レポートに警告が出る。真に冷えていなかった可能性がある"
        " (この公開版では該当条件だけ除いて集計している)。",
        "",
        "thinking: off/on を分けて測る。on でも `max_tokens` は off と同じ値"
        " を送り、on 側にだけ生成予算を足して有利にすることはしない。"
        " `reasoning_content` も生成トークンとして数える。",
        "",
        "生成長: シナリオごとに固定の `max_tokens` を使い、エンジン間で"
        " 変えない。",
        "",
        "既知の限界: 並列 (同時 N リクエスト) デコードはこの版の対象外"
        " (mlxturbo の並列デコード経路が未修正のため、"
        " `docs/research/KERNEL-BRIEF-DECODE-BW.md`)。量子化は MLX 系"
        " (mlxturbo/mlx-serve とも qwen4_exp 4bit affine group64) のみを"
        " 揃えて比較しており、llama.cpp 等の K-quant 系との比較ではない。"
        " 反復 1-2 回の条件は分布を語らない (「分布なし」の印を参照)。",
        "",
        "設計の全体像・シナリオの定義・tier の内訳は"
        " `docs/research/BENCH-DESIGN-2026-09.md` を参照。",
    ]


def render_public_markdown(agg: dict[AggKey, dict[str, dict]],
                           plan: dict | None = None,
                           meta: dict | None = None) -> str:
    """公開版 (「大本営発表」下地)。存在するデータの節だけを実表にし、
    無い節は省略してその旨を書く (`docs/BENCHMARKS-2026-09.md` に貼る想定)。
    """
    lines = ["# mlxturbo vs mlx-serve — ベンチ結果", ""]
    lines.append(f"生成日時: {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
                 " (UTC)。`bench/suite/report.py --markdown` の出力。")
    if plan:
        lines.append(f"tier: `{plan.get('tier', '未計測')}`")
    lines.append("")

    lines.append("## (a) 見出し")
    lines.append("")
    _ok, sec = headline_table(agg)
    lines += sec
    lines.append("")

    lines.append("## (b) 池ごとの差 (10% 超に印)")
    lines.append("")
    _ok, sec = pool_table(agg)
    lines += sec
    lines.append("")

    lines.append("## (c) 出力長 x thinking")
    lines.append("")
    _ok, sec = tokens_thinking_table(agg)
    lines += sec
    lines.append("")

    lines.append("## (d) 分布 (min/p50/p95/max、反復 3 未満は「分布なし」)")
    lines.append("")
    lines += _main_table_lines(agg)
    lines.append("")

    failures = agg.get("__failures__") or []
    if failures:
        lines.append("**失敗したブロック (上の表には含まれない):**")
        lines.append("")
        lines.append("| シナリオ | 文脈 | エンジン | エラー |")
        lines.append("|---|---|---|---|")
        for f in failures:
            lines.append(f"| {f['scenario']} | {f['ctx']} | {f['engine']} | {f['error']} |")
        lines.append("")

    lines.append("## (e) 手順と環境")
    lines.append("")
    lines += procedure_and_env_section(plan, meta)
    lines.append("")

    lines.append("## (f) 注意書き")
    lines.append("")
    lines += caveats_section()

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="raw_path", required=True,
                    help="run.py が書き出した raw.jsonl")
    ap.add_argument("--out", default=None,
                    help="詳細版 Markdown の書き出し先 (内部診断用。"
                         " 省略すると詳細版は書かない)")
    ap.add_argument("--markdown", default=None,
                    help="公開版 Markdown の書き出し先"
                         " (`docs/BENCHMARKS-2026-09.md` に貼る想定)")
    ap.add_argument("--plan", default=None,
                    help="run.py が書いた plan.json のパス。省略すると"
                         " `--in` と同じディレクトリの plan.json を探す"
                         " (無ければ (e) 節は未計測になる)")
    ap.add_argument("--llmprobe", action="append", default=[],
                    metavar="ENGINE=PATH",
                    help="llmprobe --bench-only の JSON を互換列として足す"
                         " (複数指定可、例: --llmprobe mlx-serve=/path/to.json)")
    ap.add_argument("--no-meta", action="store_true",
                    help="計測環境メタデータの収集をしない (git/sysctl も呼ばない)")
    args = ap.parse_args()

    if not args.out and not args.markdown:
        print("--out か --markdown の少なくとも一方を指定すること", file=sys.stderr)
        return 2

    raw_path = Path(args.raw_path)
    raw = load_raw(raw_path)
    agg = aggregate(raw)
    extra_rows = []
    for item in args.llmprobe:
        engine, _, path = item.partition("=")
        extra_rows.append(from_llmprobe_json(Path(path), engine))
    meta = None if args.no_meta else collect_environment_meta()

    if args.out:
        md = render_markdown(agg, meta, extra_rows)
        Path(args.out).write_text(md)
        print(f"書き出し (詳細版): {args.out}")

    if args.markdown:
        plan_path = Path(args.plan) if args.plan else raw_path.parent / "plan.json"
        plan = load_plan(plan_path)
        if plan is None:
            print(f"[注意] plan.json が見つからない ({plan_path}) — "
                 "(e) 手順と環境の一部が「未計測」になる", file=sys.stderr)
        md = render_public_markdown(agg, plan, meta)
        Path(args.markdown).write_text(md)
        print(f"書き出し (公開版): {args.markdown}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
