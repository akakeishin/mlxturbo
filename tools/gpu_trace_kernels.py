"""decode forward 1 回の GPU カーネル内訳を測る (Metal System Trace)。

うちの forward が GPU 時間 24.0ms、mlx-serve が 18.1ms で、差 6ms がどの
カーネルにあるかをカーネル区間の突き合わせで特定するための道具。

xctrace の作法 (docs/research/KERNEL-BRIEF-DECODE-BW.md の記録どおり):
  record: `xcrun xctrace record --template 'Metal System Trace' --attach <pid>`
          (既定 20 秒。--attach の代わりにコマンドを渡すと --launch で起動する)
  export: `xcrun xctrace export --xpath
          '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'`
          schema 名に確信が無ければ `toc` サブコマンドで一覧を見る。
          export の XML は同じ値を id/ref で重複排除するので、export は
          これを解決してからカーネル区間 (name, start, duration) を拾う。
          列名 (どのタグが名前/開始/長さか) は実物の trace でしか確定
          できないため、先頭行のタグ一覧を毎回 stderr に出す。ヒューリス
          ティックが外れたら --name-field/--start-field/--duration-field
          で上書きする。

record と export は GPU と xctrace を実際に動かす。**別計測が GPU を
使っている間は叩かないこと。** --dry-run を付けると xctrace に渡す
コマンド列を表示するだけで実行しない。summarize と diff は JSON だけを
扱うので GPU を使わない。

    uv run python tools/gpu_trace_kernels.py record --attach 12345 \\
        --tag decode-ours --time-limit 20s
    uv run python tools/gpu_trace_kernels.py record --tag decode-serve -- \\
        .venv/bin/python tools/verify_width_cost.py --widths 1 --reps 40
    uv run python tools/gpu_trace_kernels.py export \\
        --trace bench/results/traces/decode-ours.trace --window
    uv run python tools/gpu_trace_kernels.py diff \\
        --a bench/results/traces/decode-ours-summary.json \\
        --b bench/results/traces/decode-serve-summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRACES_DIR = REPO_ROOT / "bench" / "results" / "traces"

_START_CANDIDATES = ("start-time", "start_time", "start", "timestamp", "time")
_DURATION_CANDIDATES = ("duration", "dur")
_NAME_CANDIDATES = ("name", "label", "kernel")

_UNIT_NS = {"ns": 1.0, "us": 1e3, "µs": 1e3, "ms": 1e6, "s": 1e9}
_TIME_FMT_RE = re.compile(r"([\d.]+)\s*(ns|us|µs|ms|s)\b")

# diff でカーネル名を突き合わせるための正規化: テンプレート/引数と末尾の
# 番号を落として「同じカーネルの別インスタンス」をまとめる。
_TEMPLATE_OR_ARGS_RE = re.compile(r"[<(].*$")
_TRAILING_NUM_RE = re.compile(r"[_\d]+$")

BIG_GAP_MS = 1.0  # summarize (b) のアイドル分布しきい値


# --------------------------------------------------------------------------
# xctrace コマンド組み立て (record / export / toc は実行するだけで、この
# ツール自体はパース済みの結果しか解釈しない)
# --------------------------------------------------------------------------


def build_record_cmd(out: Path, template: str, time_limit: str,
                      attach: int | None, command: list[str],
                      env: list[str]) -> list[str]:
    cmd = ["xcrun", "xctrace", "record",
           "--template", template,
           "--time-limit", time_limit,
           "--output", str(out),
           "--no-prompt"]
    if attach is not None:
        cmd += ["--attach", str(attach)]
    else:
        cmd += ["--launch", "--", *command]
    for kv in env:
        cmd += ["--env", kv]
    return cmd


def build_export_cmd(trace: Path, xpath: str, xml_out: Path) -> list[str]:
    return ["xcrun", "xctrace", "export",
            "--input", str(trace), "--xpath", xpath, "--output", str(xml_out)]


def build_toc_cmd(trace: Path) -> list[str]:
    return ["xcrun", "xctrace", "export", "--input", str(trace), "--toc"]


def _print_cmd(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(c) for c in cmd))


# --------------------------------------------------------------------------
# export: xctrace の XML (id/ref 重複排除つき) をカーネル区間に変換
# --------------------------------------------------------------------------


def _load_id_registry(root: ET.Element) -> dict[str, ET.Element]:
    """文書全体を 1 回舐めて id -> 要素 の対応表を作る。

    xctrace の export XML は同じ値の 2 回目以降を `ref="N"` で参照し、
    最初の出現だけが `id="N"` を持つ。id は定義が参照より先に出る前提
    (実測どおりだが未確認) なので、先に全体を registry 化してから各行を
    読む形にして順序に依存しないようにしてある。
    """
    registry: dict[str, ET.Element] = {}
    for elem in root.iter():
        eid = elem.get("id")
        if eid is not None:
            registry[eid] = elem
    return registry


def _resolve(elem: ET.Element, registry: dict[str, ET.Element]) -> ET.Element:
    ref = elem.get("ref")
    if ref is None:
        return elem
    target = registry.get(ref)
    if target is None:
        raise ValueError(f"未解決の ref={ref!r} (<{elem.tag}>): registry に無い")
    return target


def _elem_value(elem: ET.Element) -> tuple[str, str | None]:
    """(生テキスト, fmt 属性) を返す。fmt は人間可読の表示値。"""
    text = (elem.text or "").strip()
    if not text:
        for sub in elem.iter():
            if sub is not elem and sub.text and sub.text.strip():
                text = sub.text.strip()
                break
    return text, elem.get("fmt")


def _parse_time_ns(text: str, fmt: str | None) -> float | None:
    """start-time/duration の列を ns の float に変換する。

    xctrace の実測では生テキストが ns 整数、fmt が "12.3 ms" のような
    表示用文字列という組み合わせが通例 (未確認、次回の実物 trace で要検証)。
    生テキストが数値ならそれを ns として採用し、ダメなら fmt から
    単位付きで読み直す。
    """
    if text:
        try:
            return float(text)
        except ValueError:
            pass
    for s in (fmt, text):
        if not s:
            continue
        m = _TIME_FMT_RE.search(s)
        if m:
            return float(m.group(1)) * _UNIT_NS[m.group(2)]
    return None


def _row_columns(row: ET.Element, registry: dict[str, ET.Element]
                  ) -> list[tuple[str, str, str | None]]:
    cols = []
    for child in list(row):
        resolved = _resolve(child, registry)
        text, fmt = _elem_value(resolved)
        cols.append((resolved.tag, text, fmt))
    return cols


def _pick_numeric(cols: list[tuple[str, str, str | None]],
                   override: str | None, candidates: tuple[str, ...]
                   ) -> float | None:
    if override:
        for tag, text, fmt in cols:
            if tag == override:
                return _parse_time_ns(text, fmt)
        return None
    for cand in candidates:
        for tag, text, fmt in cols:
            if cand in tag.lower():
                v = _parse_time_ns(text, fmt)
                if v is not None:
                    return v
    return None


def _pick_name(cols: list[tuple[str, str, str | None]],
                override: str | None) -> str | None:
    if override:
        for tag, text, fmt in cols:
            if tag == override:
                return fmt or text or None
    for cand in _NAME_CANDIDATES:
        for tag, text, fmt in cols:
            if cand in tag.lower() and (fmt or text):
                return fmt or text
    # 専用の name/label 列が無い場合: <string> 型の列のうち最後のもの
    # (track/queue/thread の後に具体的なカーネル名が来ることが多い、という
    # ヒューリスティック。外れたら --name-field で上書き)
    strings = [(fmt or text) for tag, text, fmt in cols
               if tag.lower() == "string" and (fmt or text)]
    return strings[-1] if strings else None


def parse_kernel_intervals(
    xml_path: Path,
    name_field: str | None = None,
    start_field: str | None = None,
    duration_field: str | None = None,
) -> tuple[list[dict], list[str]]:
    """export 済み XML からカーネル区間 [{name, start_ms, duration_ms}] を読む。

    戻り値の 2 つ目は先頭行の列タグ一覧 (診断用)。空リストなら <table><row>
    が見つかっていない = xpath / schema を間違えている。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    registry = _load_id_registry(root)

    table = root.find(".//table")
    if table is None:
        raise ValueError("<table> が見つからない (--schema / --xpath を確認)")
    rows = table.findall("row")
    if not rows:
        rows = list(table.iter("row"))

    intervals: list[dict] = []
    sample_tags: list[str] = []
    for i, row in enumerate(rows):
        cols = _row_columns(row, registry)
        if i == 0:
            sample_tags = [c[0] for c in cols]
        start_ns = _pick_numeric(cols, start_field, _START_CANDIDATES)
        dur_ns = _pick_numeric(cols, duration_field, _DURATION_CANDIDATES)
        name = _pick_name(cols, name_field)
        if start_ns is None or dur_ns is None or name is None:
            continue
        intervals.append({
            "name": name,
            "start_ms": start_ns / 1e6,
            "duration_ms": dur_ns / 1e6,
        })
    return intervals, sample_tags


# --------------------------------------------------------------------------
# summarize: (a) カーネル別集計 (b) アイドル (c) GPU/壁時計比 (d) window 分解
# --------------------------------------------------------------------------


def summarize(intervals: list[dict], gap_ms: float = 3.0, window: bool = False,
              wall_ms: float | None = None, top_n: int = 40) -> dict:
    if not intervals:
        return {"n_intervals": 0, "total_gpu_ms": 0.0, "top_kernels": [],
                 "idle": {"total_idle_ms": 0.0, "n_gaps": 0,
                          "n_gaps_gt_1ms": 0, "mean_gap_gt_1ms_ms": 0.0}}

    ivs = sorted(intervals, key=lambda x: x["start_ms"])
    total_gpu_ms = sum(iv["duration_ms"] for iv in ivs)
    span_start = ivs[0]["start_ms"]
    span_end = max(iv["start_ms"] + iv["duration_ms"] for iv in ivs)
    span_ms = span_end - span_start

    if wall_ms is None:
        wall = span_ms
        print("wall_ms 未指定: カーネル区間の span で代用 (--wall-ms で正確な"
              "壁時計を渡すとよい)", file=sys.stderr)
    else:
        wall = wall_ms

    top_kernels = _kernel_breakdown(ivs, top_n)

    gaps = []
    cur_end = span_start
    for iv in ivs:
        gap = iv["start_ms"] - cur_end
        if gap > 0:
            gaps.append(gap)
        cur_end = max(cur_end, iv["start_ms"] + iv["duration_ms"])
    big_gaps = [g for g in gaps if g > BIG_GAP_MS]
    idle = {
        "total_idle_ms": sum(gaps),
        "n_gaps": len(gaps),
        "n_gaps_gt_1ms": len(big_gaps),
        "mean_gap_gt_1ms_ms": (sum(big_gaps) / len(big_gaps)) if big_gaps else 0.0,
    }

    result = {
        "n_intervals": len(ivs),
        "total_gpu_ms": total_gpu_ms,
        "span_ms": span_ms,
        "wall_ms": wall,
        "gpu_wall_ratio": (total_gpu_ms / wall) if wall else None,
        "top_kernels": top_kernels,
        "idle": idle,
    }
    if window:
        result["window"] = _window_breakdown(ivs, gap_ms, top_n)
    return result


def _kernel_breakdown(ivs: list[dict], top_n: int) -> list[dict]:
    by_name: dict[str, list[float]] = defaultdict(list)
    for iv in ivs:
        by_name[iv["name"]].append(iv["duration_ms"])
    rows = [{"name": n, "total_ms": sum(d), "count": len(d), "mean_ms": sum(d) / len(d)}
            for n, d in by_name.items()]
    rows.sort(key=lambda r: -r["total_ms"])
    return rows[:top_n]


def _window_breakdown(ivs: list[dict], gap_ms: float, top_n: int) -> dict:
    """隙間が gap_ms 以上でセグメントに区切り、中央値セグメントと同じ
    カーネル数を持つセグメント群の平均を「1 forward」の内訳として返す。
    """
    segments: list[list[dict]] = [[ivs[0]]]
    for prev, cur in zip(ivs, ivs[1:]):
        gap = cur["start_ms"] - (prev["start_ms"] + prev["duration_ms"])
        if gap >= gap_ms:
            segments.append([cur])
        else:
            segments[-1].append(cur)

    seg_stats = []
    for seg in segments:
        s0 = seg[0]["start_ms"]
        e0 = max(iv["start_ms"] + iv["duration_ms"] for iv in seg)
        gpu_ms = sum(iv["duration_ms"] for iv in seg)
        seg_stats.append({
            "start_ms": s0, "end_ms": e0, "n_kernels": len(seg),
            "gpu_ms": gpu_ms, "idle_ms": max(0.0, (e0 - s0) - gpu_ms),
        })

    order = sorted(range(len(seg_stats)),
                    key=lambda i: (seg_stats[i]["end_ms"] - seg_stats[i]["start_ms"],
                                   seg_stats[i]["n_kernels"]))
    median_idx = order[len(order) // 2]
    typical_n = seg_stats[median_idx]["n_kernels"]
    typical_idx = [i for i, s in enumerate(seg_stats) if s["n_kernels"] == typical_n]

    sums: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, list[int]] = defaultdict(list)
    for i in typical_idx:
        by_name: dict[str, list[float]] = defaultdict(list)
        for iv in segments[i]:
            by_name[iv["name"]].append(iv["duration_ms"])
        for n, d in by_name.items():
            sums[n].append(sum(d))
            counts[n].append(len(d))
    per_forward = [{"name": n, "mean_total_ms": sum(v) / len(v),
                     "mean_count": sum(counts[n]) / len(counts[n])}
                    for n, v in sums.items()]
    per_forward.sort(key=lambda r: -r["mean_total_ms"])

    return {
        "gap_ms": gap_ms,
        "n_segments": len(segments),
        "segments": seg_stats,
        "median_segment_index": median_idx,
        "typical_kernel_count": typical_n,
        "n_typical_segments": len(typical_idx),
        "per_forward_kernels": per_forward[:top_n],
    }


# --------------------------------------------------------------------------
# diff: 2 つの summary の top_kernels をカーネル名 (完全一致 -> 正規化名) で
# 突き合わせる
# --------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    n = _TEMPLATE_OR_ARGS_RE.sub("", name)
    n = _TRAILING_NUM_RE.sub("", n)
    n = n.strip().rstrip(":_")
    return n or name


def diff_summaries(summary_a: dict, summary_b: dict, top_n: int = 40) -> dict:
    ka = {r["name"]: r["total_ms"] for r in summary_a.get("top_kernels", [])}
    kb = {r["name"]: r["total_ms"] for r in summary_b.get("top_kernels", [])}

    matched = set(ka) & set(kb)
    rows = [{"name": n, "total_a_ms": ka[n], "total_b_ms": kb[n],
              "delta_ms": ka[n] - kb[n], "names_a": [n], "names_b": [n]}
             for n in matched]

    def bucket(d: dict[str, float], used: set[str]) -> dict[str, list[tuple[str, float]]]:
        out: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for n, v in d.items():
            if n in used:
                continue
            out[normalize_name(n)].append((n, v))
        return out

    ba = bucket(ka, matched)
    bb = bucket(kb, matched)
    for norm in set(ba) | set(bb):
        a_items = ba.get(norm, [])
        b_items = bb.get(norm, [])
        total_a = sum(v for _, v in a_items)
        total_b = sum(v for _, v in b_items)
        rows.append({
            "name": norm, "total_a_ms": total_a, "total_b_ms": total_b,
            "delta_ms": total_a - total_b,
            "names_a": [n for n, _ in a_items], "names_b": [n for n, _ in b_items],
        })

    rows.sort(key=lambda r: -abs(r["delta_ms"]))
    only_a = [r["name"] for r in rows if r["total_b_ms"] == 0 and r["total_a_ms"] > 0]
    only_b = [r["name"] for r in rows if r["total_a_ms"] == 0 and r["total_b_ms"] > 0]

    total_a = summary_a.get("total_gpu_ms")
    total_b = summary_b.get("total_gpu_ms")
    return {
        "total_gpu_ms_a": total_a,
        "total_gpu_ms_b": total_b,
        "delta_total_ms": (total_a - total_b) if (total_a is not None and total_b is not None) else None,
        "kernels": rows[:top_n],
        "only_a": only_a,
        "only_b": only_b,
    }


# --------------------------------------------------------------------------
# サブコマンド
# --------------------------------------------------------------------------


def cmd_record(args: argparse.Namespace) -> None:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if (args.attach is None) == (not command):
        raise SystemExit("record には --attach <pid> か `-- コマンド...` の"
                          "どちらか一方が要る (両方無し/両方ありは不可)")
    if args.attach is not None and args.env:
        print("--env は --attach では使われない (xctrace の仕様。--launch の"
              "ときだけ有効)", file=sys.stderr)

    tag = args.tag or (f"attach{args.attach}" if args.attach is not None
                        else Path(command[0]).stem)
    out = Path(args.out) if args.out else TRACES_DIR / f"{tag}.trace"

    cmd = build_record_cmd(out, args.template, args.time_limit, args.attach,
                            command, args.env)
    _print_cmd(cmd)
    if args.dry_run:
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} が既にある (--force で消して録り直す。"
                              f"--append-run は未対応)")
        if out.is_dir():
            shutil.rmtree(out)
        else:
            out.unlink()
    subprocess.run(cmd, check=True)
    print(f"trace -> {out}")


def cmd_export(args: argparse.Namespace) -> None:
    trace = Path(args.trace)
    xpath = args.xpath or (
        f'/trace-toc/run[@number="{args.run}"]/data/table[@schema="{args.schema}"]')
    xml_out = Path(args.xml_out) if args.xml_out else trace.parent / f"{trace.stem}-{args.schema}.xml"

    cmd = build_export_cmd(trace, xpath, xml_out)
    _print_cmd(cmd)
    if args.dry_run:
        return

    xml_out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)

    intervals, sample_tags = parse_kernel_intervals(
        xml_out, name_field=args.name_field, start_field=args.start_field,
        duration_field=args.duration_field)
    print(f"先頭行の列タグ: {sample_tags}", file=sys.stderr)
    print(f"カーネル区間 {len(intervals)} 件をパース", file=sys.stderr)
    if not intervals:
        print("0 件: --name-field/--start-field/--duration-field を先頭行の"
              "列タグに合わせて指定し直すこと", file=sys.stderr)

    summary = summarize(intervals, gap_ms=args.gap_ms, window=args.window,
                         wall_ms=args.wall_ms, top_n=args.top_n)
    summary["trace"] = str(trace)
    summary["schema"] = args.schema

    out_json = Path(args.out_json) if args.out_json else trace.parent / f"{trace.stem}-summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"summary -> {out_json}")

    if args.raw_json:
        raw_path = Path(args.raw_json)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(intervals, ensure_ascii=False, indent=1))
        print(f"raw intervals -> {raw_path}")


def cmd_summarize(args: argparse.Namespace) -> None:
    intervals = json.loads(Path(args.raw_json).read_text())
    summary = summarize(intervals, gap_ms=args.gap_ms, window=args.window,
                         wall_ms=args.wall_ms, top_n=args.top_n)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"summary -> {out_json}")


def cmd_diff(args: argparse.Namespace) -> None:
    summary_a = json.loads(Path(args.a).read_text())
    summary_b = json.loads(Path(args.b).read_text())
    result = diff_summaries(summary_a, summary_b, top_n=args.top_n)
    result["a"] = args.a
    result["b"] = args.b
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out_json:
        Path(args.out_json).write_text(text)
        print(f"diff -> {args.out_json}")
    else:
        print(text)


def cmd_toc(args: argparse.Namespace) -> None:
    cmd = build_toc_cmd(Path(args.trace))
    _print_cmd(cmd)
    if args.dry_run:
        return
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if args.out_xml:
        Path(args.out_xml).write_text(proc.stdout)
    schemas = sorted(set(re.findall(r'schema="([^"]+)"', proc.stdout)))
    print("schema 一覧:")
    for s in schemas:
        print(f"  {s}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GPU カーネル単位のプロファイル道具 (xctrace / Metal System Trace)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("record", help="xctrace で Metal System Trace を録る (attach か launch)")
    p.add_argument("--attach", type=int, default=None, help="既存プロセスの PID")
    p.add_argument("command", nargs=argparse.REMAINDER,
                    help="--attach を使わないときの起動コマンド。先頭に -- を "
                         "置く (例: -- .venv/bin/python tools/foo.py --x)")
    p.add_argument("--template", default="Metal System Trace")
    p.add_argument("--time-limit", default="20s",
                    help="xctrace の <time[ms|s|m|h]> 形式")
    p.add_argument("--tag", default=None, help="出力ファイル名に使う (既定は pid か起動コマンド名)")
    p.add_argument("--out", default=None, help="既定 bench/results/traces/<tag>.trace")
    p.add_argument("--env", action="append", default=[],
                    help="KEY=VALUE。--launch のときだけ有効。繰り返し可")
    p.add_argument("--force", action="store_true", help="既存の .trace を消して録り直す")
    p.add_argument("--dry-run", action="store_true", help="xctrace コマンドを表示するだけ")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("export", help="trace からカーネル区間を書き出し、summarize まで自動で流す")
    p.add_argument("--trace", required=True)
    p.add_argument("--schema", default="metal-gpu-intervals")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--xpath", default=None, help="既定は --run / --schema から組む")
    p.add_argument("--xml-out", default=None, help="書き出した生 XML の保存先 (既定 trace 隣)")
    p.add_argument("--out-json", default=None, help="summary の書き出し先 (既定 trace 隣)")
    p.add_argument("--raw-json", default=None, help="生のカーネル区間も別途保存 (再 summarize 用)")
    p.add_argument("--name-field", default=None, help="ヒューリスティックが外れたときの列タグ上書き")
    p.add_argument("--start-field", default=None)
    p.add_argument("--duration-field", default=None)
    p.add_argument("--gap-ms", type=float, default=3.0, help="--window のセグメント区切り")
    p.add_argument("--window", action="store_true", help="1 forward 単位への分解も出す")
    p.add_argument("--wall-ms", type=float, default=None, help="壁時計 (未指定ならカーネル span で代用)")
    p.add_argument("--top-n", type=int, default=40)
    p.add_argument("--dry-run", action="store_true", help="xctrace export コマンドを表示するだけ")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("summarize", help="export が吐いた raw intervals JSON から summary を作り直す")
    p.add_argument("--raw-json", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--gap-ms", type=float, default=3.0)
    p.add_argument("--window", action="store_true")
    p.add_argument("--wall-ms", type=float, default=None)
    p.add_argument("--top-n", type=int, default=40)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("diff", help="2 つの summary JSON をカーネル名で突き合わせる")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out-json", default=None)
    p.add_argument("--top-n", type=int, default=40)
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("toc", help="trace の table 一覧 (schema 名) を確認する")
    p.add_argument("--trace", required=True)
    p.add_argument("--out-xml", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_toc)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
