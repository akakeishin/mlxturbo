#!/usr/bin/env python3
"""Count AIR (LLVM IR) operations per loop nest for the generated kernels.

Answers the questions that do not need a GPU:
  * did ``#pragma unroll`` actually unroll?
  * how many ops does one dequant + MMA step cost, and of what kind?
  * how wide are the device loads, and were they hoisted out of the inner loop?

    python3 tools/isa/analyze_air.py [--dir build/air] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent

LABEL_RE = re.compile(r"^(\d+):")
BR_RE = re.compile(r"\bbr\s+(?:i1\s+[^,]+,\s+)?label %(\d+)(?:,\s*label %(\d+))?")
DEFINE_RE = re.compile(r"^define\s+.*@([A-Za-z0-9_.$]+)\(")
LOAD_RE = re.compile(r"=\s*load\s+([^,]+),\s*([^*]+)\*")
STORE_RE = re.compile(r"^\s*store\s+([^,]+?)\s+[^,]+,\s*([^*]+)\*")
CALL_RE = re.compile(r"call\s+[^@]*@([A-Za-z0-9_.$]+)")

# AIR address spaces: 1 = device, 3 = threadgroup, 0 = thread/private.
ADDRSPACE_RE = re.compile(r"addrspace\((\d+)\)")

DEQUANT_OPS = ("lshr", "ashr", "and", "or", "shl")
CONVERT_CALLS = ("air.convert.",)
MMA_CALL = "air.simdgroup_matrix_8x8_multiply_accumulate"
SG_LOAD_CALLS = ("air.simdgroup_matrix_8x8_load", "air.simdgroup_matrix_8x8_store")
SHUFFLE_CALL = "air.simd_shuffle"
BARRIER_CALLS = ("air.wg.barrier", "air.simdgroup.barrier", "air.mem.barrier")


def _addrspace(text: str) -> str:
    m = ADDRSPACE_RE.search(text)
    if not m:
        return "thread"
    return {"1": "device", "3": "threadgroup", "0": "thread"}.get(m.group(1), m.group(1))


class Block:
    __slots__ = ("label", "lines", "succs")

    def __init__(self, label: str) -> None:
        self.label = label
        self.lines: list[str] = []
        self.succs: list[str] = []


def parse_function(lines: list[str]) -> list[Block]:
    blocks: list[Block] = [Block("entry")]
    for raw in lines:
        line = raw.rstrip()
        m = LABEL_RE.match(line)
        if m:
            blocks.append(Block(m.group(1)))
            continue
        blocks[-1].lines.append(line)
        if "br " in line:
            bm = BR_RE.search(line)
            if bm:
                blocks[-1].succs.extend(g for g in bm.groups() if g)
            elif "switch" not in line:
                for t in re.findall(r"label %(\d+)", line):
                    blocks[-1].succs.append(t)
    return blocks


def find_loops(blocks: list[Block]) -> list[tuple[int, int]]:
    """Return (header_index, latch_index) for each back edge, outermost first."""

    index = {b.label: i for i, b in enumerate(blocks)}
    loops: list[tuple[int, int]] = []
    for i, b in enumerate(blocks):
        for s in b.succs:
            j = index.get(s)
            if j is not None and j <= i:
                loops.append((j, i))
    loops.sort(key=lambda t: (t[0], -t[1]))
    return loops


def count_ops(lines: list[str]) -> Counter:
    c: Counter = Counter()
    for line in lines:
        s = line.strip()
        if not s or s.startswith(";"):
            continue
        c["_ir_lines"] += 1

        lm = LOAD_RE.search(s)
        if lm:
            ty = lm.group(1).strip()
            c[f"load[{_addrspace(s)}:{ty}]"] += 1
            c["_loads"] += 1
        sm = STORE_RE.search(s)
        if sm:
            c[f"store[{_addrspace(s)}:{sm.group(1).strip()}]"] += 1
            c["_stores"] += 1

        cm = CALL_RE.search(s)
        if cm:
            fn = cm.group(1)
            if fn.startswith(MMA_CALL):
                c["mma"] += 1
            elif fn.startswith(SG_LOAD_CALLS):
                c["simdgroup_matrix_ldst"] += 1
            elif fn.startswith(SHUFFLE_CALL):
                c["simd_shuffle"] += 1
            elif fn.startswith(BARRIER_CALLS):
                c["barrier"] += 1
            elif fn.startswith(CONVERT_CALLS):
                c[f"convert[{fn.split('air.convert.')[1]}]"] += 1
                c["_converts"] += 1
            elif fn.startswith("llvm.fmuladd"):
                c["fmuladd"] += 1
            elif fn.startswith("llvm.") or fn.startswith("air."):
                c[f"call[{fn}]"] += 1

        for op in DEQUANT_OPS:
            if re.search(rf"=\s*{op}\s", s):
                c[f"int:{op}"] += 1
        for op in ("fmul", "fadd", "fsub", "fptrunc", "fpext", "uitofp", "sitofp"):
            if re.search(rf"=\s*(?:tail call\s+)?{op}\s", s) or re.search(
                rf"=\s*{op}\s", s
            ):
                c[op] += 1
        if re.search(r"=\s*(insertelement|extractelement)\s", s):
            c["vec_element"] += 1
        if re.search(r"=\s*phi\s", s):
            c["phi"] += 1
            vm = re.search(r"=\s*phi\s+<(\d+) x ([a-z0-9]+)>", s)
            if vm:
                c[f"phi_vec[<{vm.group(1)} x {vm.group(2)}>]"] += 1
        if re.search(r"=\s*alloca\s", s):
            c["alloca"] += 1
    return c


def analyze_file(path: pathlib.Path) -> dict:
    text = path.read_text().splitlines()
    funcs: dict[str, list[str]] = {}
    cur: str | None = None
    for line in text:
        dm = DEFINE_RE.match(line)
        if dm:
            cur = dm.group(1)
            funcs[cur] = []
            continue
        if cur is not None:
            if line.startswith("}"):
                cur = None
                continue
            funcs[cur].append(line)

    out: dict = {"file": path.name, "functions": {}}
    for fname, flines in funcs.items():
        blocks = parse_function(flines)
        loops = find_loops(blocks)
        whole = count_ops([l for b in blocks for l in b.lines])
        finfo: dict = {
            "total": dict(whole),
            "blocks": len(blocks),
            "loops": [],
        }
        for header, latch in loops:
            body = [l for b in blocks[header : latch + 1] for l in b.lines]
            c = count_ops(body)
            finfo["loops"].append(
                {
                    "header_block": blocks[header].label,
                    "latch_block": blocks[latch].label,
                    "n_blocks": latch - header + 1,
                    "counts": dict(c),
                }
            )
        out["functions"][fname] = finfo
    return out


SUMMARY_KEYS = [
    "mma",
    "simdgroup_matrix_ldst",
    "simd_shuffle",
    "barrier",
    "_loads",
    "_stores",
    "_converts",
    "fmuladd",
    "fmul",
    "fadd",
    "fptrunc",
    "fpext",
    "int:lshr",
    "int:and",
    "phi",
    "alloca",
    "_ir_lines",
]


def innermost_with_mma(finfo: dict) -> dict | None:
    cands = [lp for lp in finfo["loops"] if lp["counts"].get("mma")]
    if not cands:
        return None
    return min(cands, key=lambda lp: lp["n_blocks"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "build" / "air"))
    ap.add_argument("--json", default=str(HERE / "build" / "air-stats.json"))
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    files = sorted(d.glob("*.ll"))
    if not files:
        print(f"no .ll under {d}; run tools/isa/build_air.sh first", file=sys.stderr)
        return 1

    results = [analyze_file(f) for f in files]
    pathlib.Path(args.json).write_text(json.dumps(results, indent=2))

    short = {
        "simdgroup_matrix_ldst": "sgmat",
        "simd_shuffle": "shuffle",
        "_ir_lines": "irLines",
        "_loads": "loads",
        "_stores": "stores",
        "_converts": "cvt",
    }
    labels = [short.get(k, k.replace("int:", "")) for k in SUMMARY_KEYS]
    hdr = f"{'kernel':<26}" + "".join(f"{lab:>9}" for lab in labels)
    print("== innermost loop containing the MMA (per iteration) ==")
    print(hdr)
    for r in results:
        for fname, finfo in r["functions"].items():
            lp = innermost_with_mma(finfo)
            if lp is None:
                continue
            c = lp["counts"]
            name = r["file"].replace(".ll", "")
            print(f"{name:<26}" + "".join(f"{c.get(k, 0):>9}" for k in SUMMARY_KEYS))

    print()
    print("== whole kernel ==")
    print(hdr)
    for r in results:
        for fname, finfo in r["functions"].items():
            c = finfo["total"]
            name = r["file"].replace(".ll", "")
            print(f"{name:<26}" + "".join(f"{c.get(k, 0):>9}" for k in SUMMARY_KEYS))

    print()
    print("== device loads by type, whole kernel ==")
    for r in results:
        for fname, finfo in r["functions"].items():
            c = finfo["total"]
            items = sorted(k for k in c if k.startswith("load[device"))
            desc = "  ".join(f"{k.split(':',1)[1][:-1]}x{c[k]}" for k in items)
            print(f"{r['file'].replace('.ll',''):<26} {desc}")

    print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
