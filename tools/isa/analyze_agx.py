#!/usr/bin/env python3
"""Disassemble the extracted AGX programs and summarise them.

Reports, per kernel: instruction histogram, register high-water mark, spill
traffic, and the per-MMA cost implied by the unroll factor.

    python3 tools/isa/analyze_agx.py [--arch applegpu_g13g] [--full name]

Decoding is dougallj/applegpu, which covers the G13 (M1) encoding.  Run
``agx_build.sh apple7 applegpu_g13g`` for a decodable target; newer slices
(G15/M3) translate fine but mostly fail to decode, so for those only the
``__text`` byte size is meaningful.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import re
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
APPLEGPU = HERE / "applegpu"

MMA = "simd_matrix_fmadd"
SPILL = ("stack_load", "stack_store", "stack_get_ptr")
LOADS = ("device_load", "threadgroup_load", "uniform_load")
STORES = ("device_store", "threadgroup_store", "uniform_store")

REG_RE = re.compile(r"\br(\d+)(?:[lh])?\b")
UREG_RE = re.compile(r"\bu(\d+)(?:[lh])?\b")
LINE_RE = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)\s*(.*)$")


def disassemble(path: pathlib.Path, offset: int) -> str:
    sys.path.insert(0, str(APPLEGPU))
    import disassemble as dis  # type: ignore

    code = path.read_bytes()[offset:]
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        dis.disassemble(code, code_offset=offset)
    except Exception as exc:  # applegpu raises on unknown G15 encodings
        print(f"; disassembly aborted: {type(exc).__name__}: {exc}")
    finally:
        sys.stdout = old
    return buf.getvalue()


JMP_BACK_RE = re.compile(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+jmp_exec_any\s+0x([0-9A-Fa-f]+)")


def _hot_loop_range(text: str) -> tuple[int, int] | None:
    """Address range of the outermost backward branch: the steady-state body.

    The kt loop is fully unrolled by the backend, so the only remaining cycle
    is the quantization-group loop.  Everything outside it runs once per
    threadgroup and is not what the m=8 fixed cost is made of.
    """

    best: tuple[int, int] | None = None
    for line in text.splitlines():
        m = JMP_BACK_RE.match(line)
        if not m:
            continue
        here = int(m.group(1), 16)
        target = int(m.group(2), 16)
        if target < here and (best is None or (here - target) > (best[1] - best[0])):
            best = (target, here)
    return best


def summarise(text: str) -> dict:
    hist: Counter = Counter()
    maxreg = -1
    maxureg = -1
    failed = 0
    total = 0
    hot = _hot_loop_range(text)
    hot_hist: Counter = Counter()
    hot_total = 0
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        mnem, rest = m.group(3), m.group(4)
        if mnem == "<disassembly":
            failed += 1
            continue
        total += 1
        hist[mnem] += 1
        if hot and hot[0] <= addr <= hot[1]:
            hot_total += 1
            hot_hist[mnem] += 1
        for rm in REG_RE.finditer(rest):
            maxreg = max(maxreg, int(rm.group(1)))
        for um in UREG_RE.finditer(rest):
            maxureg = max(maxureg, int(um.group(1)))

    mma = sum(v for k, v in hist.items() if k.startswith(MMA))
    hot_mma = sum(v for k, v in hot_hist.items() if k.startswith(MMA))
    return {
        "instructions": total,
        "undecoded": failed,
        "hot_instructions": hot_total,
        "hot_mma": hot_mma,
        "hot_shuffle": hot_hist.get("simd_shuffle", 0),
        "hot_loads": sum(v for k, v in hot_hist.items() if k.startswith(LOADS)),
        "hot_wait": hot_hist.get("wait", 0),
        "hot_hist": dict(hot_hist.most_common()),
        "max_reg": maxreg,
        "max_ureg": maxureg,
        "mma": mma,
        "spill": sum(hist.get(k, 0) for k in SPILL),
        "loads": sum(v for k, v in hist.items() if k.startswith(LOADS)),
        "stores": sum(v for k, v in hist.items() if k.startswith(STORES)),
        "simd_shuffle": hist.get("simd_shuffle", 0),
        "wait": hist.get("wait", 0),
        "icmpsel": hist.get("icmpsel", 0),
        "mov": hist.get("mov", 0) + hist.get("mov_imm", 0),
        "convert": hist.get("convert", 0),
        "fmadd32": hist.get("fmadd32", 0),
        "fadd32": hist.get("fadd32", 0),
        "barrier": hist.get("threadgroup_barrier", 0),
        "hist": dict(hist.most_common()),
    }


COLS = [
    ("instr", "instructions"),
    ("undec", "undecoded"),
    ("hotInstr", "hot_instructions"),
    ("hotMMA", "hot_mma"),
    ("instr/mma", None),
    ("hotShuf", "hot_shuffle"),
    ("hotLoad", "hot_loads"),
    ("hotWait", "hot_wait"),
    ("maxReg", "max_reg"),
    ("spill", "spill"),
    ("icmpsel", "icmpsel"),
    ("mov", "mov"),
    ("barrier", "barrier"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="applegpu_g13g")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--full", default=None, help="print full disassembly for this kernel")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    d = pathlib.Path(args.dir) if args.dir else HERE / "build" / "agx" / args.arch
    extracts = sorted(d.glob("*.extract.txt"))
    if not extracts:
        print(f"nothing under {d}; run tools/isa/agx_build.sh first", file=sys.stderr)
        return 1

    results: dict[str, dict] = {}
    for ex in extracts:
        name = ex.name.replace(".extract.txt", "")
        line = ex.read_text().strip()
        if not line:
            continue
        binpath = pathlib.Path(line.split()[0])
        off = 0
        m = re.search(r"_agc\.main@(0x[0-9a-f]+)", line)
        if m:
            off = int(m.group(1), 0)
        text = disassemble(binpath, off)
        (d / f"{name}.asm").write_text(text)
        results[name] = summarise(text)
        results[name]["text_bytes"] = binpath.stat().st_size

        if args.full and args.full in name:
            print(text)

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=2))

    print(f"== {args.arch} ==")
    hdr = f"{'kernel':<26}" + "".join(f"{c:>10}" for c, _ in COLS)
    print(hdr)
    for name, r in results.items():
        cells = []
        for label, key in COLS:
            if key is None:
                v = (
                    round(r["hot_instructions"] / r["hot_mma"], 1)
                    if r["hot_mma"]
                    else 0
                )
            else:
                v = r[key]
            cells.append(f"{v:>10}")
        print(f"{name:<26}" + "".join(cells))
    print(f"\nasm written next to the extracts in {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
