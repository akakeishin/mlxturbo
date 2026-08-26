#!/bin/bash
# GPU-QUEUE STEP.  Needs a Metal device; everything else in tools/isa does not.
#
#   tools/isa/gpu_probe.sh
#
# For every generated kernel: build the real compute pipeline on this machine's
# GPU and record what the driver decides -- maxTotalThreadsPerThreadgroup (the
# register footprint expressed as an occupancy limit), threadExecutionWidth and
# the static threadgroup allocation -- then serialize a MTLBinaryArchive of the
# native code for disassembly.
#
# Results land in tools/isa/build/gpu/.  Feed them back with
#   python3 tools/isa/gpu_report.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$HERE/build/pipeline_probe"
LIB="$HERE/build/metallib"
OUT="$HERE/build/gpu"
mkdir -p "$OUT"

if [ ! -x "$PROBE" ]; then
  clang "$HERE/pipeline_probe.m" -O2 -framework Metal -framework Foundation \
        -fobjc-arc -o "$PROBE"
fi

shopt -s nullglob
for f in "$LIB"/*.metallib; do
  name="$(basename "$f" .metallib)"
  "$PROBE" --archive "$OUT/$name.archive.bin" --json "$OUT/$name.json" "$f"
done

echo "-> $OUT"
