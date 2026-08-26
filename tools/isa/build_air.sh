#!/bin/bash
# Compile every generated .metal to AIR text, AIR bitcode and a .metallib.
# CPU only: no Metal device is created, so this runs anywhere xcrun works.
#
#   tools/isa/build_air.sh [std] [math-mode]
#
# Defaults match what mlx 0.32.2 asks the runtime compiler for: latest language
# version, math mode "safe" (mx.fast.metal_kernel's documented default).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STD="${1:-metal3.2}"
MATH="${2:-safe}"

SRC="$HERE/build/metal"
OUT="$HERE/build/air"
LIB="$HERE/build/metallib"
mkdir -p "$OUT" "$LIB"

FLAGS=(-std="$STD" -fmetal-math-mode="$MATH" -O2)

shopt -s nullglob
for f in "$SRC"/*.metal; do
  name="$(basename "$f" .metal)"
  xcrun -sdk macosx metal "${FLAGS[@]}" -S -o "$OUT/$name.ll" "$f"
  xcrun -sdk macosx metal "${FLAGS[@]}" -c -o "$OUT/$name.air" "$f"
  xcrun -sdk macosx metallib -o "$LIB/$name.metallib" "$OUT/$name.air"
  printf '%-28s ll=%-6s metallib=%s\n' "$name" \
    "$(wc -l < "$OUT/$name.ll" | tr -d ' ')" \
    "$(stat -f%z "$LIB/$name.metallib")"
done

echo "std=$STD math=$MATH -> $OUT , $LIB"
