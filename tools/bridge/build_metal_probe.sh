#!/usr/bin/env bash
# metal_probe のビルド。Xcode Command Line Tools 以外に依存しない。
# 出力: tools/bridge/libmetal_probe.dylib
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/metal_probe.mm"
OUT="$HERE/libmetal_probe.dylib"

# ARC は使わない (swizzle した IMP を素の関数ポインタとして呼ぶため、
# 所有権の移動を ARC に推測させない)。
xcrun clang++ \
  -std=c++17 \
  -fno-objc-arc \
  -O2 \
  -fvisibility=hidden \
  -Wall -Wextra -Wno-unused-parameter \
  -dynamiclib \
  -install_name "@rpath/libmetal_probe.dylib" \
  -framework Metal -framework Foundation \
  -o "$OUT" "$SRC"

echo "built: $OUT"
