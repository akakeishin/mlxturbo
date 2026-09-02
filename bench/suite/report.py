"""`run.py` が書き出した生 JSON を Markdown の表にする。

集計方針 (`docs/research/BENCH-DESIGN-2026-09.md` (b)(g) 節が仕様):

- **冷 TTFT の見出し数字は各ブロックの rep=0 (フレッシュ起動直後) だけ**を使う。
  rep>=1 は同じ起動セッション内で GPU が温まっていくため、絶対値としては
  rep=0 より当てにならない (CLAUDE.md の熱の作法)。rep>=1 は分散の参考値として
  別列に出す。
- decode tok/s は `(n-1)/経過` を rep ごとに計算し、中央値を報告する
  (これは同一ブロック内の反復なので rep=0 に限定する理由が無い)。
- `usage.prompt_tokens_details.cached_tokens` が「冷」ターンで 0 でなければ、
  その rep は接頭辞キャッシュに当たっていた = 真の意味で冷えていない。
  headline から除外し、警告として明記する (無かったことにしない)。

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
    return json.loads(path.read_text())


def _cold_headline(block_result: dict) -> dict:
    """rep=0 (フレッシュ起動) の "cold" ラベルターンを取り出す。"""
    for rep in block_result.get("reps", []):
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


def aggregate(raw: list[dict]) -> dict:
    """`(scenario, ctx)` ごとに、エンジン別の集計行を作る。"""
    out: dict[tuple[str, int | None], dict[str, dict]] = {}
    for block_result in raw:
        b = block_result["block"]
        key = (b["scenario_name"], b["ctx"])
        row = out.setdefault(key, {})
        cold = _cold_headline(block_result)
        all_cold_ttft = [
            t["ttft_s"] for rep in block_result["reps"]
            for t in rep["turns"] if t["label"] == "cold"
        ]
        all_warm_ttft = [
            t["ttft_s"] for rep in block_result["reps"]
            for t in rep["turns"] if t["label"] == "warm"
        ]
        decode_samples = [
            _decode_tps(t) for rep in block_result["reps"] for t in rep["turns"]
        ]
        decode_samples = [d for d in decode_samples if d == d]  # NaN 除去
        cold_cache_hits = [
            t.get("cached_tokens") for rep in block_result["reps"]
            for t in rep["turns"] if t["label"] == "cold" and t.get("cached_tokens")
        ]
        row[b["engine_kind"]] = dict(
            cold_ttft_fresh_s=cold.get("ttft_s"),
            cold_ttft_all_median_s=(statistics.median(all_cold_ttft)
                                    if all_cold_ttft else None),
            warm_ttft_median_s=(statistics.median(all_warm_ttft)
                                if all_warm_ttft else None),
            decode_tps_median=(statistics.median(decode_samples)
                               if decode_samples else None),
            reps=len(block_result["reps"]),
            cold_cache_hit_warning=(
                f"「冷」ターンで cached_tokens>0 が {len(cold_cache_hits)} 件"
                f" (値: {cold_cache_hits}) — 真に冷えていなかった可能性"
                if cold_cache_hits else None),
            residual_warnings=block_result.get("residual_warnings") or [],
        )
    return out


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


def render_markdown(agg: dict, meta: dict | None = None,
                    extra_rows: list[dict] | None = None) -> str:
    lines = ["# ベンチ結果", ""]
    if meta:
        lines.append("計測環境: " + ", ".join(f"{k}={v}" for k, v in meta.items()))
        lines.append("")
    lines.append("見出し数字は各ブロックの rep=0 (フレッシュ起動直後) のみ。"
                 "全 rep の中央値は別列 (熱で膨らんでいる可能性がある)。")
    lines.append("")
    header = ("シナリオ", "文脈", "エンジン", "冷TTFT(fresh)",
             "冷TTFT(全rep中央値)", "温TTFT(中央値)", "decode(中央値)", "reps")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for (scenario, ctx), engines in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1)):
        for eng, row in sorted(engines.items()):
            lines.append("| " + " | ".join(str(v) for v in (
                scenario, ctx if ctx is not None else "-", eng,
                _fmt(row["cold_ttft_fresh_s"]), _fmt(row["cold_ttft_all_median_s"]),
                _fmt(row["warm_ttft_median_s"]), _fmt(row["decode_tps_median"]),
                row["reps"])) + " |")
            if row.get("cold_cache_hit_warning"):
                lines.append(f"|  |  |  | **警告**: {row['cold_cache_hit_warning']} |||||")
            for w in row.get("residual_warnings") or []:
                lines.append(f"|  |  |  | **警告 (残留プロセス)**: {w} |||||")
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


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="raw_path", required=True,
                    help="run.py が書き出した raw.json")
    ap.add_argument("--out", required=True, help="書き出す Markdown のパス")
    ap.add_argument("--llmprobe", action="append", default=[],
                    metavar="ENGINE=PATH",
                    help="llmprobe --bench-only の JSON を互換列として足す"
                         " (複数指定可、例: --llmprobe mlx-serve=/path/to.json)")
    ap.add_argument("--no-meta", action="store_true",
                    help="計測環境メタデータの収集をしない (git/sysctl も呼ばない)")
    args = ap.parse_args()

    raw = load_raw(Path(args.raw_path))
    agg = aggregate(raw)
    extra_rows = []
    for item in args.llmprobe:
        engine, _, path = item.partition("=")
        extra_rows.append(from_llmprobe_json(Path(path), engine))
    meta = None if args.no_meta else collect_environment_meta()
    md = render_markdown(agg, meta, extra_rows)
    Path(args.out).write_text(md)
    print(f"書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
