"""17k prefill の GPU 稼働率と泡 (アイドル区間) を、xctrace の export XML と
MLXTURBO_PREFILL_TRACE のログを突き合わせて出す。

背景・目的:
    冷 prefill が mlx-serve に 5% 負けている残差が、
    (a) チャンク/グループ境界 (mx.eval + clear_cache) や層の間の「泡」
        (GPU に何も入っていない区間) なのか、
    (b) カーネルが GPU に張り付いている演算そのもの (busy 時間の絶対量) なのか
    を、GPU タイムラインの busy/idle 分布と MLXTURBO_PREFILL_TRACE の境界
    時刻を突き合わせて切り分けるための道具。decode のラウンド間の泡 7.3ms を
    xctrace で特定したのと同じ手口 (docs/research/KERNEL-BRIEF-DECODE-BW.md
    2026-08-31 の記録) を prefill に向ける。

    録画は tools/prefill_gpu_trace.sh (同じ tools/ 配下、別ファイル) が行う。
    この道具は録れた .trace / export 済み XML と、そのとき一緒に残る
    MLXTURBO_PREFILL_TRACE のログを読むだけで、GPU は使わない。

**未検証の懸念 (2026-09-02 14:30 の実測、
docs/research/SESSION-2026-09-02-CATCHUP.md「xctrace は今回は使えなかった」
節):** 同じ Metal System Trace / --launch を decode ループの python プロセスに
向けたとき、metal-gpu-intervals に python (MLX) の Compute 区間がほぼ出な
かった (49k 行中 87 行、大半は Claude Helper の描画)。prefill は GPU 稼働が
連続的で decode のラウンド境界より掴みやすい可能性はあるが、**この道具が
同じ結果 (ほぼ空) になる可能性は残っている。**busy_span_ratio が 0 に近い、
または区間数が極端に少ない場合は、まずこの前例を疑うこと (このスクリプトは
その旨の警告を stderr に出す)。

xctrace の export XML の形式は tools/gpu_trace_kernels.py が 2026-09-02 に
実 trace で確認済みのものをそのまま前提にしている (ルートは
<trace-query-result> -> <node> -> <schema> + <row>*、列は schema の
<col><mnemonic> 順、同じ値は初出が id=N、以後は ref=N で参照される)。
そちらは xml.etree.ElementTree.parse で XML 全体を一度に読むが、17k prefill
は GPU 区間数が桁違いに多くなりうる (2026-09-02 の decode 32 秒トレースで
既に 19GB の export が後処理不能になった実測がある。同じ CATCHUP の節)。
**このためここでは xml.etree.ElementTree.iterparse + 逐次 clear() で処理し、
DOM 全体をメモリに載せない** (行を読むたびに使い終えた要素を捨てる。id/ref の
対応表だけは文書全体で保持するが、これは重複値の再利用のための小さな表で
あって行数には比例しない -- xctrace 自身がその dedup のために id/ref を
使っている)。

xctrace の作法 (tools/gpu_trace_kernels.py と同じ、記録は docs/research/
KERNEL-BRIEF-DECODE-BW.md の実測どおり):
    テーブル名 (schema) の確認:
        xcrun xctrace export --input <trace> --toc
        (または `tools/gpu_trace_kernels.py toc --trace <trace>` -- 同じ
        リポジトリの既存道具で、export 済み XML の schema 一覧を出すだけ)
    export そのもの (.trace を読むだけ。GPU は使わない):
        xcrun xctrace export --input <trace> \\
            --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \\
            --output <xml>
    --trace を渡せばこのスクリプトが上の export を自分で呼ぶ (GPU 不使用)。
    既に export 済みなら --xml で直接渡せて xctrace を一切呼ばない。
    schema 名は "metal-gpu-intervals" を既定にしてあるが、実物と違えば
    --schema で差し替える (toc の一覧から選ぶ)。

使い方:
    # .trace から一気に (xctrace export は呼ぶが GPU は使わない)
    .venv/bin/python tools/parse_gpu_trace.py \\
        --trace bench/results/traces/prefill-17000-20260903-101500.trace \\
        --pf-log bench/results/logs/prefill-17000-20260903-101500.log

    # 既に export 済みの XML から (xctrace を一切呼ばない)
    .venv/bin/python tools/parse_gpu_trace.py \\
        --xml bench/results/traces/prefill-17000-...-metal-gpu-intervals.xml \\
        --pf-log bench/results/logs/prefill-17000-...log

    # export コマンドを見るだけ (実行しない)
    .venv/bin/python tools/parse_gpu_trace.py --trace <trace> --dry-run

出すもの (JSON。--out-json 省略時は <xml のあるディレクトリ>/<stem>-analysis.json
に書き、要約を stdout にも出す):
    (a) prefill 全体の GPU busy 率 (busy_span_ratio。depth==0 の区間の
        duration 合計 / (最後の区間の終わり - 最初の区間の始まり))
    (b) 泡 (idle) の分布と、--min-gap-ms (既定 5.0ms) 以上の泡の一覧
        (時刻付き、"gaps")
    (c) --pf-log を渡した場合、MLXTURBO_PREFILL_TRACE のログの境界時刻
        (チャンク/グループの build/eval/clear_cache/tail split 等) と
        泡を突き合わせ、それぞれの泡を「境界の泡 (boundary)」「層内の泡
        (in-layer、ログ上は 1 区間の途中なのに GPU が止まっている)」
        「unclassified (ログの区間外)」に分ける ("classified_gaps")

(c) の時刻の突き合わせについて (calibration):
    xctrace の区間の時刻 (start_ms) はこの export のトレース基準時刻からの
    経過 ms で、MLXTURBO_PREFILL_TRACE のログの t= は _PrefillTracer 生成時
    (perf_counter()) からの経過 ms -- **2 つの時計は基準点が違う。**
    このスクリプトは (1) xctrace 側の区間を大きな隙間 (既定 2000ms、
    --segment-gap-ms) で「occurrence」に分割し、(2) ログ側も prefill 1 回分
    ("[prefill] total dur_sum=..." 行で区切られる。tools/prefill_gpu_trace.sh
    の docstring にあるとおり 1 プロセス内で prefill が複数回起きる) に分割
    したうえで、span (継続時間) が一番近い組を自動で選び、その 2 点
    (区間の開始/終了、ログの最初/最後のイベント) から線形の
    offset + scale を最小限のフィッティングで決める。**これは推定であり、
    自動一致の質 (span_match_delta_pct) を出力に必ず含めるので、大きい
    (目安 10% 超) ときは自動一致を信用せず、"xctrace_segments" と
    "pf_occurrences" を見比べて --segment-index / --pf-occurrence で
    手で選び直すこと。**

    "prime chunk ci=..." で始まるログ行は _prime_draft_cache 内のローカル
    perf_counter() (spec_flash.py:1783 の _pf_t0) を使っており、上記の
    _PrefillTracer の共有タイムラインとは基準点が別物なので、突き合わせ
    (classify_gaps) には使わない (別途 "prime_chunks" として出すだけ)。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(TOOLS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 既に実 trace で確認済みの xctrace export XML の読み方 (id/ref 解決、
# ns への時刻変換、xctrace コマンドの組み立て) を再利用する。ここで
# 二重に実装して食い違いが起きるより、検証済みの1本を共有するほうが安全
# (tools/gpu_trace_kernels.py は変更しない -- そちらの DOM 一括パース
# parse_intervals() 自体は 17k 向けに使わず、コマンド組み立てと時刻変換
# だけを借りる)。
import gpu_trace_kernels as GTK  # noqa: E402

TRACES_DIR = REPO_ROOT / "bench" / "results" / "traces"

DEFAULT_SCHEMA = "metal-gpu-intervals"
DEFAULT_MIN_GAP_MS = 5.0
DEFAULT_SEGMENT_GAP_MS = 2000.0
DEFAULT_BOUNDARY_TOL_MS = 3.0

_STRUCTURAL_TAGS = {"row", "schema", "node", "trace-query-result", "col",
                    "mnemonic", "name", "trace-toc", "run", "data", "table"}


# --------------------------------------------------------------------------
# ストリーム export: iterparse + 逐次 clear() で XML 全体を DOM に載せない。
# --------------------------------------------------------------------------


def _leaf_value(elem: ET.Element) -> dict:
    """id/ref で参照されうる 1 要素の値を、Element 参照を残さず抜き出す。

    gpu_trace_kernels._elem_value 相当 (生テキストが無ければ子孫を探す) に
    加えて、process 列が持つ <pid> 子要素のテキストも一緒に取る (process は
    fmt = 表示名、text/pid 子 = 数値 pid という構造。gpu_trace_kernels.
    parse_intervals の process 抽出と同じ前提)。
    """
    text = (elem.text or "").strip()
    if not text:
        for sub in elem.iter():
            if sub is not elem and sub.text and sub.text.strip():
                text = sub.text.strip()
                break
    pid_el = elem.find("pid")
    pid_text = None
    if pid_el is not None:
        pid_text = (pid_el.text or "").strip()
    return {"text": text, "fmt": elem.get("fmt"), "pid_text": pid_text}


def _resolve_child(elem: ET.Element, registry: dict[str, dict]) -> dict:
    ref = elem.get("ref")
    if ref is not None:
        val = registry.get(ref)
        if val is None:
            raise ValueError(
                f"未解決の ref={ref!r} (<{elem.tag}>): registry に無い。"
                " xctrace の export XML は id が参照より先に出る前提"
                " (tools/gpu_trace_kernels.py の _load_id_registry と同じ"
                " 前提) だが、このファイルではそれが崩れている")
    else:
        val = _leaf_value(elem)
        eid = elem.get("id")
        if eid is not None:
            registry[eid] = val
    return val


def _process_row(row_elem: ET.Element, idx: dict[str, int],
                  registry: dict[str, dict]) -> dict | None:
    children = list(row_elem)
    if len(children) != len(idx):
        return None  # schema と列数が合わない壊れた行

    def get(mnem: str) -> dict:
        return _resolve_child(children[idx[mnem]], registry)

    start_v = get(GTK.MNEM_START)
    dur_v = get(GTK.MNEM_DURATION)
    start_ns = GTK._parse_time_ns(start_v["text"], start_v["fmt"])
    dur_ns = GTK._parse_time_ns(dur_v["text"], dur_v["fmt"])
    if start_ns is None or dur_ns is None:
        return None

    chan_v = get(GTK.MNEM_CHANNEL)
    channel = chan_v["fmt"] or chan_v["text"] or None

    depth_v = get(GTK.MNEM_DEPTH)
    try:
        depth = int(depth_v["text"])
    except ValueError:
        depth = None

    proc_v = get(GTK.MNEM_PROCESS)
    process = proc_v["fmt"]
    pid = None
    if proc_v["pid_text"]:
        try:
            pid = int(proc_v["pid_text"])
        except ValueError:
            pid = None

    enc_v = get(GTK.MNEM_ENCODER)
    encoder_id = enc_v["fmt"] or enc_v["text"] or None
    cmdbuf_v = get(GTK.MNEM_CMDBUFFER)
    cmdbuffer_id = cmdbuf_v["fmt"] or cmdbuf_v["text"] or None

    return {
        "start_ms": start_ns / 1e6,
        "duration_ms": dur_ns / 1e6,
        "channel": channel,
        "depth": depth,
        "pid": pid,
        "process": process,
        "encoder_id": encoder_id,
        "cmdbuffer_id": cmdbuffer_id,
    }


def stream_filtered_intervals(
    xml_path: Path, schema_name: str | None, channel: str | None,
    process_substr: str | None,
) -> tuple[list[dict], list[str] | None, int]:
    """xctrace export XML をストリームで読み、depth==0 かつ channel/process
    が一致する区間だけを集めて返す ([区間], mnemonics, 壊れた行数)。

    メモリに残るのは (1) id/ref の対応表 (重複値の dedup 用で行数に比例
    しない)、(2) フィルタを通った区間のリスト (全区間よりずっと少ない
    はず) だけ -- <row> は処理し終えるたびに clear() し、DOM を溜め込まない。
    """
    registry: dict[str, dict] = {}
    mnemonics: list[str] | None = None
    idx: dict[str, int] | None = None
    current_schema_name: str | None = None
    n_bad = 0
    out: list[dict] = []

    context = ET.iterparse(str(xml_path), events=("end",))
    for _event, elem in context:
        tag = elem.tag
        if tag == "row":
            if idx is not None and (
                schema_name is None or current_schema_name == schema_name
            ):
                rec = _process_row(elem, idx, registry)
                if rec is None:
                    n_bad += 1
                elif rec["depth"] == 0 and (
                    not channel
                    or (rec["channel"] or "").lower() == channel.lower()
                ) and (
                    not process_substr
                    or process_substr.lower() in (rec["process"] or "").lower()
                ):
                    out.append(rec)
            elem.clear()
            continue
        if tag == "schema":
            current_schema_name = elem.get("name")
            if schema_name is None or current_schema_name == schema_name:
                cur_mnemonics = [c.findtext("mnemonic") for c in elem.findall("col")]
                missing = [m for m in GTK.REQUIRED_MNEMONICS if m not in cur_mnemonics]
                if not missing:
                    mnemonics = cur_mnemonics
                    idx = {m: i for i, m in enumerate(mnemonics)}
            continue
        if tag in _STRUCTURAL_TAGS:
            continue
        eid = elem.get("id")
        if eid is not None:
            registry[eid] = _leaf_value(elem)
        # id を持たない葉要素はどこからも参照されないので、値も取らずに
        # 素通りしてよい (row 処理側が直接その場で読む)

    if idx is None:
        raise ValueError(
            f"schema={schema_name!r} の <schema> が見つからなかった"
            " (xpath や --schema を見直すこと。"
            " `xcrun xctrace export --input <trace> --toc` で一覧を確認できる)")
    return out, mnemonics, n_bad


# --------------------------------------------------------------------------
# (a)(b): busy 率と、隙間 (泡) の一覧 (時刻付き)
# --------------------------------------------------------------------------


def compute_busy_and_gaps(intervals: list[dict], min_gap_ms: float) -> dict:
    if not intervals:
        return {
            "n_intervals": 0, "total_busy_ms": 0.0, "span_ms": 0.0,
            "busy_span_ratio": None, "span_start_ms": None, "span_end_ms": None,
            "min_gap_ms": min_gap_ms, "n_gaps": 0, "gaps": [],
        }
    ordered = sorted(intervals, key=lambda iv: iv["start_ms"])
    total_busy = sum(iv["duration_ms"] for iv in ordered)
    span_start = ordered[0]["start_ms"]
    span_end = max(iv["start_ms"] + iv["duration_ms"] for iv in ordered)
    span = span_end - span_start

    gaps: list[dict] = []
    cur_end = span_start
    for iv in ordered:
        s = iv["start_ms"]
        if s > cur_end:
            g = s - cur_end
            if g >= min_gap_ms:
                gaps.append({
                    "gap_start_ms": cur_end, "gap_end_ms": s, "duration_ms": g,
                })
        cur_end = max(cur_end, iv["start_ms"] + iv["duration_ms"])

    return {
        "n_intervals": len(ordered),
        "total_busy_ms": total_busy,
        "span_ms": span,
        "busy_span_ratio": (total_busy / span) if span > 0 else None,
        "span_start_ms": span_start,
        "span_end_ms": span_end,
        "min_gap_ms": min_gap_ms,
        "n_gaps": len(gaps),
        "gaps": gaps,
    }


def split_segments(intervals: list[dict], segment_gap_ms: float) -> list[dict]:
    """大きな隙間 (既定 2000ms) で区間列を「occurrence」候補に割る。

    温めの prefill / 共有 prefill / decode だけの区間などが、この粒度の
    隙間で分かれて出てくることを期待している (経験的な閾値。実データを見て
    --segment-gap-ms で調整すること)。
    """
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv["start_ms"])
    segs: list[list[dict]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        gap = cur["start_ms"] - (prev["start_ms"] + prev["duration_ms"])
        if gap >= segment_gap_ms:
            segs.append([cur])
        else:
            segs[-1].append(cur)
    out = []
    for seg in segs:
        s0 = seg[0]["start_ms"]
        e0 = max(iv["start_ms"] + iv["duration_ms"] for iv in seg)
        out.append({
            "start_ms": s0, "end_ms": e0, "span_ms": e0 - s0,
            "busy_ms": sum(iv["duration_ms"] for iv in seg),
            "n_intervals": len(seg),
        })
    return out


# --------------------------------------------------------------------------
# MLXTURBO_PREFILL_TRACE ログの読み取り
# --------------------------------------------------------------------------

_EVENT_RE = re.compile(r"^\[prefill\] (?P<label>.+?) t=(?P<t>[\d.]+) dur=(?P<dur>[\d.]+)\s*$")
_TOTAL_RE = re.compile(
    r"^\[prefill\] total dur_sum=(?P<dur_sum>[\d.]+) wall=(?P<wall>[\d.]+) "
    r"gap=(?P<gap>-?[\d.]+)\s*$")


def parse_pf_log(path: Path) -> list[dict]:
    """MLXTURBO_PREFILL_TRACE=1 の (stdout+stderr 結合) ログを、prefill 1 回
    ("[prefill] total dur_sum=..." 行まで) ごとの occurrence に分けて返す。

    "prime chunk ci=..." 行は _prime_draft_cache 独自のローカル perf_counter
    (spec_flash.py:1783) を使っており、他の [prefill] 行 (_PrefillTracer の
    共有タイムライン) と基準点が違うので、"events" ではなく別枠
    "prime_chunks" に入れる (calibrate/classify_gaps はこちらを見ない)。
    """
    occurrences: list[dict] = []
    cur_events: list[dict] = []
    cur_prime_chunks: list[dict] = []

    def flush_incomplete():
        if cur_events or cur_prime_chunks:
            occurrences.append({
                "events": cur_events, "prime_chunks": cur_prime_chunks,
                "dur_sum_ms": None, "wall_ms": None, "unaccounted_gap_ms": None,
                "incomplete": True,
            })

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m_total = _TOTAL_RE.match(line)
            if m_total:
                occurrences.append({
                    "events": cur_events, "prime_chunks": cur_prime_chunks,
                    "dur_sum_ms": float(m_total["dur_sum"]),
                    "wall_ms": float(m_total["wall"]),
                    "unaccounted_gap_ms": float(m_total["gap"]),
                    "incomplete": False,
                })
                cur_events = []
                cur_prime_chunks = []
                continue
            m = _EVENT_RE.match(line)
            if not m:
                continue
            label = m["label"]
            t_end = float(m["t"])
            dur = float(m["dur"])
            rec = {"label": label, "t_end_ms": t_end, "dur_ms": dur,
                   "t_start_ms": t_end - dur}
            if label.startswith("prime chunk"):
                cur_prime_chunks.append(rec)
            else:
                cur_events.append(rec)
    flush_incomplete()
    return occurrences


# --------------------------------------------------------------------------
# (c) 時刻の突き合わせ: 2 つの時計 (xctrace のトレース基準 / ログの
# perf_counter 基準) を線形 (offset + scale) で対応付け、泡を分類する
# --------------------------------------------------------------------------


def calibrate(xctrace_segments: list[dict], occurrence: dict,
              segment_index: int | None = None) -> dict | None:
    events = occurrence.get("events") or []
    if not events:
        return None
    pf_first = min(e["t_start_ms"] for e in events)
    pf_last = max(e["t_end_ms"] for e in events)
    pf_span = pf_last - pf_first
    if pf_span <= 0:
        return None

    if segment_index is not None:
        if not (0 <= segment_index < len(xctrace_segments)):
            raise ValueError(
                f"--segment-index {segment_index} は範囲外"
                f" (0..{len(xctrace_segments) - 1})")
        chosen_idx = segment_index
    else:
        if not xctrace_segments:
            return None
        wall = occurrence.get("wall_ms") or pf_span
        chosen_idx = min(
            range(len(xctrace_segments)),
            key=lambda i: abs(xctrace_segments[i]["span_ms"] - wall))
    seg = xctrace_segments[chosen_idx]

    scale = seg["span_ms"] / pf_span if pf_span > 0 else 1.0
    offset = seg["start_ms"] - pf_first * scale
    wall = occurrence.get("wall_ms")
    delta_pct = (
        abs(seg["span_ms"] - wall) / wall * 100.0
        if wall else None
    )
    return {
        "segment_index": chosen_idx, "segment": seg,
        "pf_first_ms": pf_first, "pf_last_ms": pf_last, "pf_span_ms": pf_span,
        "scale": scale, "offset_ms": offset,
        "span_match_delta_pct": delta_pct,
    }


def _pf_to_x(t_pf_ms: float, calib: dict) -> float:
    return calib["offset_ms"] + calib["scale"] * t_pf_ms


def _x_to_pf(t_x_ms: float, calib: dict) -> float:
    return (t_x_ms - calib["offset_ms"]) / calib["scale"]


def classify_gaps(gaps: list[dict], occurrence: dict, calib: dict,
                   boundary_tol_ms: float) -> list[dict]:
    events = occurrence.get("events") or []
    boundaries_pf: list[float] = []
    for e in events:
        boundaries_pf.append(e["t_start_ms"])
        boundaries_pf.append(e["t_end_ms"])

    out = []
    for g in gaps:
        mid_x = (g["gap_start_ms"] + g["gap_end_ms"]) / 2.0
        pf_mid = _x_to_pf(mid_x, calib)
        nearest = min((abs(pf_mid - b) for b in boundaries_pf), default=None)
        if nearest is not None and nearest <= boundary_tol_ms:
            cls = "boundary"
        elif any(e["t_start_ms"] < pf_mid < e["t_end_ms"] for e in events):
            cls = "in-layer"
        else:
            cls = "unclassified"  # ログの区間外 (prime chunk の区間や末尾の
            # 未計測 gap=... の間である可能性。occurrence["unaccounted_gap_ms"]
            # と付き合わせること)
        rec = dict(g)
        rec["pf_mid_ms_est"] = pf_mid
        rec["nearest_boundary_delta_ms"] = nearest
        rec["classification"] = cls
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------


def _default_out_json(xml_path: Path) -> Path:
    stem = xml_path.stem
    for suffix in ("-metal-gpu-intervals", f"-{DEFAULT_SCHEMA}"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return xml_path.parent / f"{stem}-prefill-gpu-analysis.json"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="17k prefill の GPU busy/idle を xctrace export XML と "
                     "MLXTURBO_PREFILL_TRACE のログから出す (GPU 不使用)")
    ap.add_argument("--trace", default=None,
                     help="xctrace の .trace。--xml 未指定ならこちらから export する")
    ap.add_argument("--xml", default=None,
                     help="export 済みの XML を直接渡す (xctrace を一切呼ばない)")
    ap.add_argument("--schema", default=DEFAULT_SCHEMA)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--xpath", default=None,
                     help="既定は --run/--schema から組む (--xml 指定時は無視)")
    ap.add_argument("--xml-out", default=None,
                     help="export した生 XML の保存先 (既定 trace の隣)")
    ap.add_argument("--process", default="python",
                     help="process 列の部分一致フィルタ (既定 python)")
    ap.add_argument("--channel", default="Compute",
                     help="channel-name の完全一致 (既定 Compute。空文字で無効化)")
    ap.add_argument("--min-gap-ms", type=float, default=DEFAULT_MIN_GAP_MS,
                     help="この時間以上の隙間だけを泡の一覧に出す (既定 5.0ms)")
    ap.add_argument("--segment-gap-ms", type=float, default=DEFAULT_SEGMENT_GAP_MS,
                     help="xctrace 側の区間を occurrence に割る隙間しきい値 (既定 2000ms)")
    ap.add_argument("--pf-log", default=None,
                     help="MLXTURBO_PREFILL_TRACE=1 の結合ログ (省略時は (c) の"
                          " 境界突き合わせをせず (a)(b) だけ出す)")
    ap.add_argument("--pf-occurrence", type=int, default=-1,
                     help="--pf-log 中の何回目の prefill を使うか (既定 -1 = 最後"
                          " = --prefill-once の共有 prefill。"
                          " tools/prefill_gpu_trace.sh の docstring 参照)")
    ap.add_argument("--segment-index", type=int, default=None,
                     help="自動選択された xctrace セグメントを手で上書きする")
    ap.add_argument("--boundary-tol-ms", type=float, default=DEFAULT_BOUNDARY_TOL_MS,
                     help="泡の中心がログの境界からこの ms 以内なら「境界の泡」"
                          "とみなす (既定 3.0ms)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--dry-run", action="store_true",
                     help="xctrace export コマンドを表示するだけ (--xml 指定時は無視)")
    args = ap.parse_args()

    if args.xml:
        xml_path = Path(args.xml)
        if not xml_path.exists():
            print(f"{xml_path} が無い", file=sys.stderr)
            return 1
        trace_repr = args.trace or str(xml_path)
    else:
        if not args.trace:
            print("--trace か --xml のどちらかが要る", file=sys.stderr)
            return 1
        trace = Path(args.trace)
        xpath = args.xpath or (
            f'/trace-toc/run[@number="{args.run}"]/data/table[@schema="{args.schema}"]')
        xml_path = (Path(args.xml_out) if args.xml_out
                    else trace.parent / f"{trace.stem}-{args.schema}.xml")
        cmd = GTK.build_export_cmd(trace, xpath, xml_path)
        GTK._print_cmd(cmd)
        if args.dry_run:
            return 0
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
        trace_repr = str(trace)

    print(f"XML をストリーム解析中: {xml_path}", file=sys.stderr)
    intervals, mnemonics, n_bad = stream_filtered_intervals(
        xml_path, args.schema, args.channel or None, args.process or None)
    print(f"schema の列: {mnemonics}", file=sys.stderr)
    if n_bad:
        print(f"列数が schema と合わない行を {n_bad} 件飛ばした", file=sys.stderr)
    print(
        f"depth==0 / channel={args.channel!r} / process*={args.process!r} で "
        f"{len(intervals)} 区間", file=sys.stderr)

    if not intervals:
        print(
            "0 区間: --process/--channel を見直すこと。2026-09-02 に同じ"
            " xctrace で python の Compute 区間がほぼ拾えなかった実測がある"
            " (docs/research/SESSION-2026-09-02-CATCHUP.md の「xctrace は"
            " 今回は使えなかった」節) -- この前例に当たった可能性を疑うこと",
            file=sys.stderr)

    busy = compute_busy_and_gaps(intervals, args.min_gap_ms)
    if busy["busy_span_ratio"] is not None and busy["busy_span_ratio"] < 0.05:
        print(
            f"busy_span_ratio={busy['busy_span_ratio']:.4f} が極端に低い。"
            " 上記 CATCHUP の前例 (python の Compute 区間がほぼ出ない) を"
            " 疑うこと (雑音チャネル/他プロセスだけを拾っている可能性)",
            file=sys.stderr)

    xctrace_segments = split_segments(intervals, args.segment_gap_ms)

    result: dict = {
        "trace": trace_repr,
        "xml": str(xml_path),
        "schema": args.schema,
        "filter": {"process": args.process, "channel": args.channel},
        "mnemonics": mnemonics,
        "n_bad_rows": n_bad,
        "busy": busy,
        "xctrace_segments": xctrace_segments,
    }

    if args.pf_log:
        pf_path = Path(args.pf_log)
        if not pf_path.exists():
            print(f"{pf_path} が無い (--pf-log)", file=sys.stderr)
            return 1
        occurrences = parse_pf_log(pf_path)
        result["pf_log"] = str(pf_path)
        result["pf_occurrences"] = occurrences
        print(f"MLXTURBO_PREFILL_TRACE のログから prefill {len(occurrences)} 回分を検出",
              file=sys.stderr)
        if not occurrences:
            print("[prefill] 行が 1 行も無かった。--pf-log の中身と"
                  " MLXTURBO_PREFILL_TRACE=1 が実際に効いていたかを確認すること",
                  file=sys.stderr)
        else:
            try:
                occurrence = occurrences[args.pf_occurrence]
            except IndexError:
                print(f"--pf-occurrence {args.pf_occurrence} は範囲外"
                      f" (occurrence 数 {len(occurrences)})", file=sys.stderr)
                return 1
            result["pf_occurrence_index"] = (
                args.pf_occurrence if args.pf_occurrence >= 0
                else len(occurrences) + args.pf_occurrence)
            result["pf_occurrence"] = occurrence

            calib = calibrate(xctrace_segments, occurrence, args.segment_index)
            if calib is None:
                print("時刻の突き合わせ (calibrate) ができなかった"
                      " (イベントが無いか xctrace セグメントが無い)。"
                      " (a)(b) だけを見ること", file=sys.stderr)
            else:
                result["calibration"] = calib
                delta = calib["span_match_delta_pct"]
                if delta is not None and delta > 10.0:
                    print(
                        f"calibration の span 一致度が甘い"
                        f" (span_match_delta_pct={delta:.1f}%)。"
                        " xctrace_segments / pf_occurrences を見比べて"
                        " --segment-index / --pf-occurrence で選び直すことを"
                        " 検討すること", file=sys.stderr)
                classified = classify_gaps(
                    busy["gaps"], occurrence, calib, args.boundary_tol_ms)
                result["classified_gaps"] = classified
                n_boundary = sum(1 for g in classified if g["classification"] == "boundary")
                n_inlayer = sum(1 for g in classified if g["classification"] == "in-layer")
                n_unclassified = len(classified) - n_boundary - n_inlayer
                result["classified_gaps_summary"] = {
                    "boundary": n_boundary, "in-layer": n_inlayer,
                    "unclassified": n_unclassified,
                }
                print(
                    f"泡 {len(classified)} 件: boundary={n_boundary} "
                    f"in-layer={n_inlayer} unclassified={n_unclassified}",
                    file=sys.stderr)

    out_json = Path(args.out_json) if args.out_json else _default_out_json(xml_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"analysis -> {out_json}")

    if busy["busy_span_ratio"] is not None:
        print(f"busy_span_ratio = {busy['busy_span_ratio']:.4f} "
              f"(busy {busy['total_busy_ms']:.1f}ms / span {busy['span_ms']:.1f}ms)")
    print(f"泡 (>= {args.min_gap_ms}ms) = {busy['n_gaps']} 件")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
