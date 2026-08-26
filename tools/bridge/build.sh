#!/usr/bin/env bash
# fastmlx bridge のビルド。Xcode Command Line Tools 以外に依存しない。
# 出力: tools/bridge/libfastmlx_bridge.dylib
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/fastmlx_bridge.mm"
OUT="$HERE/libfastmlx_bridge.dylib"

xcrun clang++ \
  -std=c++17 \
  -fobjc-arc \
  -O2 \
  -fvisibility=hidden \
  -Wall -Wextra -Wno-unused-parameter \
  -dynamiclib \
  -install_name "@rpath/libfastmlx_bridge.dylib" \
  -framework Metal -framework Foundation \
  -o "$OUT" "$SRC"

echo "built: $OUT"
