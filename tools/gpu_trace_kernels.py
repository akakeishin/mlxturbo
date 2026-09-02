"""decode forward 1 回の GPU 区間内訳を測る (Metal System Trace)。

うちの forward が GPU 時間 24.0ms、mlx-serve が 18.1ms で、差 6ms がどこに
あるかを GPU 区間の突き合わせで特定するための道具。

xctrace の作法 (docs/research/KERNEL-BRIEF-DECODE-BW.md の記録どおり):
  record: `xcrun xctrace record --template 'Metal System Trace' --attach <pid>`
          (既定 20 秒。--attach の代わりにコマンドを渡すと --launch で起動する)
  export: `xcrun xctrace export --xpath
          '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'`
          schema 名に確信が無ければ `toc` サブコマンドで一覧を見る。

実物の export XML (2026-09-02、実 trace で確認済み。以下は前提ではなく事実):
  ルートは <trace-query-result> -> <node xpath='...'> -> <schema name=
  "metal-gpu-intervals"> (列定義は <col><mnemonic>...</mnemonic>...</col> の
  並びで、これが行内の列順そのもの) と、続く <row> 群。<table> 要素は無い。
  <row> の子は型名の要素で列順に並ぶが、`duration` 型のタグが 2 回
  (区間長そのものと CPU→GPU の start-latency) 出るなど、**タグ名だけでは
  列を特定できない**。schema の <col> 順で列インデックスを引く。
  同じ値は初出で `id="N"`、以後 `ref="N"` で参照される (ref 解決が必須)。
  カーネル名 (shader 名) はこの schema には無い
  (metal-shader-profiler-intervals は別 schema で 0 行のことがある)。
  取れるのは「プロセス x チャネル (Vertex/Fragment/Compute) x 区間」で、
  区間には event-depth (ネストの深さ) が付く。

record と export (xctrace 実行あり) は GPU と xctrace を実際に動かす。
**別計測が GPU を使っている間は叩かないこと。** --dry-run を付けると
xctrace に渡すコマンド列を表示するだけで実行しない。既に export 済みの XML
があるなら export に --xml で渡せば xctrace を一切呼ばない (この経路は GPU
を使わない)。summarize と diff は JSON だけを扱うので GPU を使わない。

    uv run python tools/gpu_trace_kernels.py record --attach 12345 \\
        --tag decode-ours --time-limit 20s
    uv run python tools/gpu_trace_kernels.py export \\
        --trace bench/results/traces/decode-ours.trace
    uv run python tools/gpu_trace_kernels.py export \\
        --xml bench/results/traces/decode-ours-metal-gpu-intervals.xml \\
        --process python --channel Compute
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
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRACES_DIR = REPO_ROOT / "bench" / "results" / "traces"

_UNIT_NS = {"ns": 1.0, "us": 1e3, "µs": 1e3, "ms": 1e6, "s": 1e9}
_TIME_FMT_RE = re.compile(r"([\d.]+)\s*(ns|us|µs|ms|s)\b")

# metal-gpu-intervals の schema で今回使う列 (mnemonic 名、col の並び順で
# 引く。タグ名では duration が 2 回出るなど区別が付かないため)。
MNEM_START = "start"
MNEM_DURATION = "duration"
MNEM_CHANNEL = "channel-name"
MNEM_DEPTH = "event-depth"
MNEM_PROCESS = "process"
MNEM_ENCODER = "encoder-id"
MNEM_CMDBUFFER = "cmdbuffer-id"
REQUIRED_MNEMONICS = (MNEM_START, MNEM_DURATION, MNEM_CHANNEL, MNEM_DEPTH,
                       MNEM_PROCESS, MNEM_ENCODER, MNEM_CMDBUFFER)

IDLE_GAP_MS = 0.5   # summarize (b) のアイドル分布しきい値 (固定、仕様どおり)
DEFAULT_GAP_MS = 2.0  # summarize (c) のセグメント区切り既定値


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
# export: xctrace の XML (id/ref 重複排除つき) を GPU 区間に変換
# --------------------------------------------------------------------------


def _load_id_registry(root: ET.Element) -> dict[str, ET.Element]:
    """文書全体を 1 回舐めて id -> 要素 の対応表を作る。

    xctrace の export XML は同じ値の 2 回目以降を `ref="N"` で参照し、
    最初の出現だけが `id="N"` を持つ。id は定義が参照より先に出る
    (実 trace で確認済み) が、念のため先に全体を registry 化してから
    各行を読む形にして順序に依存しないようにしてある。
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
    """start/duration 列を ns の float に変換する。

    実 trace では生テキストが ns 整数、fmt が "12.3 ms" のような表示用
    文字列という組み合わせ (確認済み)。生テキストが数値ならそれを ns
    として採用し、ダメなら fmt から単位付きで読み直す (保険)。
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


def _find_data_node(root: ET.Element, schema_name: str | None) -> ET.Element:
    """<node xpath='...'><schema name="...">...</schema><row>...</row>... を
    持つ <node> を返す。schema_name が指定されていればそれに一致するものを
    優先し、無ければ最初に見つかった <node> を使う。
    """
    nodes = root.findall(".//node")
    if not nodes:
        raise ValueError("<node> が見つからない (export 済み XML か確認)")
    if schema_name:
        for node in nodes:
            schema = node.find("schema")
            if schema is not None and schema.get("name") == schema_name:
                return node
    return nodes[0]


def parse_intervals(
    xml_path: Path, schema_name: str = "metal-gpu-intervals",
) -> tuple[list[dict], list[str]]:
    """export 済み XML から GPU 区間を読む。

    戻り値の 1 つ目は各区間 dict のリスト:
      start_ms, duration_ms, channel, depth (event-depth, int),
      pid (int|None), process (表示名、例 "Claude Helper (1635)"),
      encoder_id, cmdbuffer_id (どちらも fmt 表示、例 "0x8cd6da407")
    2 つ目は schema の列 mnemonic 一覧 (診断用)。

    列はタグ名ではなく schema の <col> 並び順 (mnemonic) で特定する。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    node = _find_data_node(root, schema_name)

    schema = node.find("schema")
    if schema is None:
        raise ValueError("<schema> が見つからない (export 済み XML か確認)")
    mnemonics = [c.findtext("mnemonic") for c in schema.findall("col")]
    missing = [m for m in REQUIRED_MNEMONICS if m not in mnemonics]
    if missing:
        raise ValueError(f"必要な列が無い: {missing} (実際の列: {mnemonics})")
    idx = {m: i for i, m in enumerate(mnemonics)}

    registry = _load_id_registry(root)
    rows = node.findall("row")

    intervals: list[dict] = []
    n_bad_rows = 0
    for row in rows:
        children = list(row)
        if len(children) != len(mnemonics):
            n_bad_rows += 1
            continue  # 列数が schema と合わない行は壊れているとみなして飛ばす

        start_el = _resolve(children[idx[MNEM_START]], registry)
        dur_el = _resolve(children[idx[MNEM_DURATION]], registry)
        chan_el = _resolve(children[idx[MNEM_CHANNEL]], registry)
        depth_el = _resolve(children[idx[MNEM_DEPTH]], registry)
        proc_el = _resolve(children[idx[MNEM_PROCESS]], registry)
        enc_el = _resolve(children[idx[MNEM_ENCODER]], registry)
        cmdbuf_el = _resolve(children[idx[MNEM_CMDBUFFER]], registry)

        start_ns = _parse_time_ns(*_elem_value(start_el))
        dur_ns = _parse_time_ns(*_elem_value(dur_el))
        if start_ns is None or dur_ns is None:
            continue

        chan_text, chan_fmt = _elem_value(chan_el)
        channel = chan_fmt or chan_text or None

        depth_text, _ = _elem_value(depth_el)
        try:
            depth = int(depth_text)
        except ValueError:
            depth = None

        pid_el = proc_el.find("pid")
        pid = None
        if pid_el is not None:
            pid_text, _ = _elem_value(pid_el)
            try:
                pid = int(pid_text)
            except ValueError:
                pid = None
        process = proc_el.get("fmt")

        enc_text, enc_fmt = _elem_value(enc_el)
        encoder_id = enc_fmt or enc_text or None
        cmdbuf_text, cmdbuf_fmt = _elem_value(cmdbuf_el)
        cmdbuffer_id = cmdbuf_fmt or cmdbuf_text or None

        intervals.append({
            "start_ms": start_ns / 1e6,
            "duration_ms": dur_ns / 1e6,
            "channel": channel,
            "depth": depth,
            "pid": pid,
            "process": process,
            "encoder_id": encoder_id,
            "cmdbuffer_id": cmdbuffer_id,
        })

    if n_bad_rows:
        print(f"列数が schema と合わない行を {n_bad_rows} 件飛ばした", file=sys.stderr)
    return intervals, mnemonics


def filter_intervals(intervals: list[dict], pid: int | None = None,
                      process: str | None = None,
                      channel: str | None = "Compute") -> list[dict]:
    """pid (完全一致) / process (部分一致、大小無視) / channel (完全一致、
    大小無視) で区間を絞る。どれも None/空なら素通し。
    """
    out = intervals
    if channel:
        needle = channel.lower()
        out = [iv for iv in out if (iv.get("channel") or "").lower() == needle]
    if pid is not None:
        out = [iv for iv in out if iv.get("pid") == pid]
    if process:
        needle = process.lower()
        out = [iv for iv in out if needle in (iv.get("process") or "").lower()]
    return out


# --------------------------------------------------------------------------
# summarize: (a) 区間数/busy/span/busy比、(b) 隙間分布、(c) セグメント
# (「forward 1 回」候補) ごとの中央値と中央値セグメントの duration 分布。
# 重なり (nesting) は event-depth 0 だけを数える。
# --------------------------------------------------------------------------


def _empty_summary(gap_ms: float) -> dict:
    return {
        "n_intervals": 0,
        "total_busy_ms": 0.0,
        "span_ms": 0.0,
        "busy_span_ratio": None,
        "idle": {"total_idle_ms": 0.0, "n_gaps": 0,
                 "n_gaps_gt_idle_ms": 0, "mean_gap_gt_idle_ms": 0.0,
                 "max_gap_gt_idle_ms": 0.0},
        "segments": {"gap_ms": gap_ms, "n_segments": 0},
    }


def summarize(intervals: list[dict], gap_ms: float = DEFAULT_GAP_MS,
              top_n: int = 20) -> dict:
    n_total = len(intervals)
    depth0 = sorted((iv for iv in intervals if iv.get("depth") == 0),
                     key=lambda x: x["start_ms"])
    n_skipped = n_total - len(depth0)
    if n_skipped:
        print(f"event-depth != 0 (ネスト) を {n_skipped} 件除外して集計",
              file=sys.stderr)

    if not depth0:
        return _empty_summary(gap_ms)

    total_busy_ms = sum(iv["duration_ms"] for iv in depth0)
    span_start = depth0[0]["start_ms"]
    span_end = max(iv["start_ms"] + iv["duration_ms"] for iv in depth0)
    span_ms = span_end - span_start
    busy_span_ratio = (total_busy_ms / span_ms) if span_ms > 0 else None

    # (b) 隙間: 前の区間の end から次の区間の start までが正の分だけ
    gaps: list[float] = []
    cur_end = span_start
    for iv in depth0:
        gap = iv["start_ms"] - cur_end
        if gap > 0:
            gaps.append(gap)
        cur_end = max(cur_end, iv["start_ms"] + iv["duration_ms"])
    big_gaps = [g for g in gaps if g > IDLE_GAP_MS]
    idle = {
        "total_idle_ms": sum(gaps),
        "n_gaps": len(gaps),
        "n_gaps_gt_idle_ms": len(big_gaps),
        "mean_gap_gt_idle_ms": (sum(big_gaps) / len(big_gaps)) if big_gaps else 0.0,
        "max_gap_gt_idle_ms": max(big_gaps) if big_gaps else 0.0,
        "idle_threshold_ms": IDLE_GAP_MS,
    }

    segments = _segment_breakdown(depth0, gap_ms, top_n)

    return {
        "n_intervals": len(depth0),
        "total_busy_ms": total_busy_ms,
        "span_ms": span_ms,
        "busy_span_ratio": busy_span_ratio,
        "idle": idle,
        "segments": segments,
    }


def _segment_breakdown(depth0: list[dict], gap_ms: float, top_n: int) -> dict:
    """隙間が gap_ms 以上でセグメントに区切る (= 「forward 1 回」の候補)。
    セグメントごとの (区間数, busy ms, span ms) の中央値と、span_ms が
    その中央値に最も近い「中央値セグメント」の区間 duration 分布 (上位
    top_n 件、降順) を返す。
    """
    segments: list[list[dict]] = [[depth0[0]]]
    for prev, cur in zip(depth0, depth0[1:]):
        gap = cur["start_ms"] - (prev["start_ms"] + prev["duration_ms"])
        if gap >= gap_ms:
            segments.append([cur])
        else:
            segments[-1].append(cur)

    seg_stats = []
    for seg in segments:
        s0 = seg[0]["start_ms"]
        e0 = max(iv["start_ms"] + iv["duration_ms"] for iv in seg)
        busy_ms = sum(iv["duration_ms"] for iv in seg)
        seg_stats.append({
            "start_ms": s0, "end_ms": e0,
            "n_intervals": len(seg), "busy_ms": busy_ms, "span_ms": e0 - s0,
        })

    median_n = statistics.median(s["n_intervals"] for s in seg_stats)
    median_busy = statistics.median(s["busy_ms"] for s in seg_stats)
    median_span = statistics.median(s["span_ms"] for s in seg_stats)

    # 「中央値セグメント」= span_ms が中央値に最も近いもの (同着は先に出た方)
    median_idx = min(range(len(seg_stats)),
                      key=lambda i: (abs(seg_stats[i]["span_ms"] - median_span), i))
    median_seg_durations = sorted(
        (iv["duration_ms"] for iv in segments[median_idx]), reverse=True)

    return {
        "gap_ms": gap_ms,
        "n_segments": len(segments),
        "median_n_intervals": median_n,
        "median_busy_ms": median_busy,
        "median_span_ms": median_span,
        "median_segment_index": median_idx,
        "median_segment": seg_stats[median_idx],
        "median_segment_top_duration_ms": median_seg_durations[:top_n],
    }


# --------------------------------------------------------------------------
# diff: カーネル名が無いので、2 つの summary のセグメント統計 (busy/span/
# 隙間/区間数) を突き合わせる。
# --------------------------------------------------------------------------


def _get_path(d: dict, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


_DIFF_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("n_intervals", ("n_intervals",)),
    ("total_busy_ms", ("total_busy_ms",)),
    ("span_ms", ("span_ms",)),
    ("busy_span_ratio", ("busy_span_ratio",)),
    ("idle_total_idle_ms", ("idle", "total_idle_ms")),
    ("idle_n_gaps", ("idle", "n_gaps")),
    ("idle_n_gaps_gt_idle_ms", ("idle", "n_gaps_gt_idle_ms")),
    ("idle_mean_gap_gt_idle_ms", ("idle", "mean_gap_gt_idle_ms")),
    ("idle_max_gap_gt_idle_ms", ("idle", "max_gap_gt_idle_ms")),
    ("segments_n_segments", ("segments", "n_segments")),
    ("segments_median_n_intervals", ("segments", "median_n_intervals")),
    ("segments_median_busy_ms", ("segments", "median_busy_ms")),
    ("segments_median_span_ms", ("segments", "median_span_ms")),
)


def diff_summaries(summary_a: dict, summary_b: dict) -> dict:
    rows = []
    for label, path in _DIFF_METRICS:
        va = _get_path(summary_a, *path)
        vb = _get_path(summary_b, *path)
        delta = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = va - vb
        rows.append({"metric": label, "a": va, "b": vb, "delta": delta})
    return {"metrics": rows}


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
    if args.xml:
        xml_out = Path(args.xml)
        if not xml_out.exists():
            raise SystemExit(f"{xml_out} が無い")
        trace = Path(args.trace) if args.trace else xml_out
        if args.dry_run:
            print(f"--xml {xml_out} 指定: xctrace export は呼ばない "
                  f"(--dry-run は無視、このままパースする)", file=sys.stderr)
    else:
        if not args.trace:
            raise SystemExit("--trace か --xml のどちらかが要る")
        trace = Path(args.trace)
        xpath = args.xpath or (
            f'/trace-toc/run[@number="{args.run}"]/data/table[@schema="{args.schema}"]')
        xml_out = (Path(args.xml_out) if args.xml_out
                    else trace.parent / f"{trace.stem}-{args.schema}.xml")

        cmd = build_export_cmd(trace, xpath, xml_out)
        _print_cmd(cmd)
        if args.dry_run:
            return

        xml_out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)

    intervals, mnemonics = parse_intervals(xml_out, schema_name=args.schema)
    print(f"schema の列: {mnemonics}", file=sys.stderr)
    print(f"GPU 区間 {len(intervals)} 件をパース", file=sys.stderr)

    filtered = filter_intervals(intervals, pid=args.pid, process=args.process,
                                 channel=args.channel)
    print(f"フィルタ後 (channel={args.channel!r} pid={args.pid} "
          f"process={args.process!r}): {len(filtered)} 件", file=sys.stderr)
    if not filtered:
        print("0 件: --pid/--process/--channel を見直すこと (--process には "
              "\"python\" や \"mlx-serve\" のような部分文字列を渡す)", file=sys.stderr)

    summary = summarize(filtered, gap_ms=args.gap_ms, top_n=args.top_n)
    summary["trace"] = str(trace)
    summary["xml"] = str(xml_out)
    summary["schema"] = args.schema
    summary["filter"] = {"pid": args.pid, "process": args.process,
                          "channel": args.channel}

    out_json = (Path(args.out_json) if args.out_json
                else xml_out.parent / f"{xml_out.stem}-summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"summary -> {out_json}")

    if args.raw_json:
        raw_path = Path(args.raw_json)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=1))
        print(f"raw intervals (フィルタ後) -> {raw_path}")


def cmd_summarize(args: argparse.Namespace) -> None:
    intervals = json.loads(Path(args.raw_json).read_text())
    summary = summarize(intervals, gap_ms=args.gap_ms, top_n=args.top_n)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"summary -> {out_json}")


def cmd_diff(args: argparse.Namespace) -> None:
    summary_a = json.loads(Path(args.a).read_text())
    summary_b = json.loads(Path(args.b).read_text())
    result = diff_summaries(summary_a, summary_b)
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
        description="GPU 区間単位のプロファイル道具 (xctrace / Metal System Trace)")
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

    p = sub.add_parser("export", help="trace (または既存 XML) から GPU 区間を書き出し、summarize まで自動で流す")
    p.add_argument("--trace", default=None, help="--xml 未指定なら必須。--xml 指定時は summary の trace 欄に残すだけ")
    p.add_argument("--xml", default=None,
                    help="xctrace export 済みの XML を直接渡す。指定時は xctrace を一切呼ばない (GPU 不使用)")
    p.add_argument("--schema", default="metal-gpu-intervals")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--xpath", default=None, help="既定は --run / --schema から組む (--xml 指定時は無視)")
    p.add_argument("--xml-out", default=None, help="書き出した生 XML の保存先 (既定 trace 隣、--xml 指定時は無視)")
    p.add_argument("--out-json", default=None, help="summary の書き出し先 (既定 xml 隣)")
    p.add_argument("--raw-json", default=None, help="フィルタ後の生区間も別途保存 (再 summarize 用)")
    p.add_argument("--pid", type=int, default=None, help="この pid だけに絞る (完全一致)")
    p.add_argument("--process", default=None, help="プロセス名の部分一致 (例 \"python\" / \"mlx-serve\")")
    p.add_argument("--channel", default="Compute", help="gpu-channel-name の完全一致 (既定 Compute。空文字で無効化)")
    p.add_argument("--gap-ms", type=float, default=DEFAULT_GAP_MS, help="forward 1 回とみなすセグメント区切りの隙間しきい値")
    p.add_argument("--top-n", type=int, default=20, help="中央値セグメントの duration 上位何件を出すか")
    p.add_argument("--dry-run", action="store_true", help="xctrace export コマンドを表示するだけ (--xml 指定時は無視)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("summarize", help="export が吐いた raw intervals JSON から summary を作り直す")
    p.add_argument("--raw-json", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--gap-ms", type=float, default=DEFAULT_GAP_MS)
    p.add_argument("--top-n", type=int, default=20)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("diff", help="2 つの summary JSON のセグメント統計 (busy/span/隙間/区間数) を突き合わせる")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out-json", default=None)
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
