#!/usr/bin/env python3
"""Summarise what ``gpu_probe.sh`` recorded, and disassemble its archives.

CPU only -- run this after the GPU-queue step has produced
``tools/isa/build/gpu/``.

    python3 tools/isa/gpu_report.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
EXTRACTOR = HERE / "applegpu" / "compiler_explorer_tools" / "metal-archive-extractor"


def disasm_archive(archive: pathlib.Path, outdir: pathlib.Path) -> str | None:
    """Run the applegpu compiler_explorer pipeline over one binary archive."""

    if not EXTRACTOR.exists():
        return None
    try:
        out = subprocess.run(
            [sys.executable, str(HERE / "applegpu" / "compiler_explorer.py"), str(archive)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:  # noqa: BLE001
        return f"; failed: {exc}"
    text = out.stdout or out.stderr
    (outdir / f"{archive.stem}.asm").write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "build" / "gpu"))
    ap.add_argument("--disasm", action="store_true")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    rows = []
    for f in sorted(d.glob("*.json")):
        for row in json.loads(f.read_text()):
            rows.append((f.stem.replace(".json", ""), row))
    if not rows:
        print(f"nothing under {d}; run tools/isa/gpu_probe.sh on a Metal machine")
        return 1

    print(f"{'kernel':<30}{'maxTPTG':>9}{'execW':>7}{'tgMem':>8}  device")
    for name, r in rows:
        print(
            f"{name:<30}{r['maxTotalThreadsPerThreadgroup']:>9}"
            f"{r['threadExecutionWidth']:>7}{r['staticThreadgroupMemoryLength']:>8}"
            f"  {r['device']}"
        )
    print(
        "\nmaxTPTG is the driver's register-footprint verdict: 1024 means the "
        "function fits the widest threadgroup, lower values mean the register "
        "allocation forced a cap."
    )

    if args.disasm:
        for archive in sorted(d.glob("*.archive.bin")):
            text = disasm_archive(archive, d)
            head = (text or "").splitlines()[:3]
            print(f"\n{archive.name}: " + (" / ".join(head) if head else "no output"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
