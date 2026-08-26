#!/bin/bash
# Translate every .metallib to native Apple GPU code, offline.
#
#   tools/isa/agx_build.sh [gpu-family] [arch]
#     gpu-family  apple7 (G13/M1) | apple8 (G14/M2) | apple9 (G15/M3) ...
#     arch        slice to thin out, e.g. applegpu_g13g / applegpu_g15s
#
# No Metal device is involved: metal-tt drives the same AIR native translator
# the driver uses, from a pipeline script naming the compute function.
# Default is apple7/applegpu_g13g because that is the only target the public
# dougallj/applegpu disassembler decodes; the G15 slice is produced too, and
# its __text size is comparable across variants even undecoded.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY="${1:-apple7}"
ARCH="${2:-applegpu_g13g}"

TT="$(xcrun -sdk macosx -f metal-tt 2>/dev/null || true)"
if [ -z "$TT" ]; then
  BIN="$(dirname "$(xcrun -sdk macosx -f metal)")"
  TT="$BIN/metal-tt"
  LIPO="$BIN/metal-lipo"
else
  LIPO="$(xcrun -sdk macosx -f metal-lipo)"
fi

LIB="$HERE/build/metallib"
OUT="$HERE/build/agx/$ARCH"
TMP="$HERE/build/agx/tmp"
mkdir -p "$OUT" "$TMP"

shopt -s nullglob
for f in "$LIB"/*.metallib; do
  name="$(basename "$f" .metallib)"
  fn="$(xcrun -sdk macosx metal-nm "$f" 2>/dev/null | awk '$2=="T"{print $3}' | head -1)"
  if [ -z "$fn" ]; then
    fn="$(strings -a "$f" | grep -m1 '^custom_kernel_')"
  fi
  if [ -z "$fn" ]; then
    echo "$name: could not find the kernel symbol, skipped" >&2
    continue
  fi
  cat > "$TMP/$name.mtlp-json" <<EOF
{
  "pipelines": {
    "compute_pipelines": [
      { "compute_function": "$fn" }
    ]
  }
}
EOF
  "$TT" -gpu-family "$FAMILY" -platform_version macos 26.0 26.0 \
        -o "$TMP/$name.gpubin" "$f" "$TMP/$name.mtlp-json"
  "$LIPO" -thin "$ARCH" -output "$TMP/$name.$ARCH.bin" "$TMP/$name.gpubin"
  python3 "$HERE/agx_extract.py" "$TMP/$name.$ARCH.bin" -o "$OUT" > "$OUT/$name.extract.txt"
  sz="$(awk '{for(i=1;i<=NF;i++) if ($i ~ /^text=/) print $i}' "$OUT/$name.extract.txt")"
  printf '%-28s %-14s %s  fn=%s\n' "$name" "$ARCH" "$sz" "$fn"
done

echo "-> $OUT"
