#!/bin/zsh
# MLXTurbo.app を組み立てる。
#
#   ./build.sh              debug ビルドして .app を作る
#   ./build.sh release      release ビルド
#   NOTARIZE=1 ./build.sh release   公証まで通す (Developer ID が要る)
#
# 署名は、あるものを上から順に使う:
#   Developer ID Application  … 配布できる (公証の前提)
#   Apple Development         … 自分の機械では動く
#   ad-hoc (-)                … 最後の手段
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-debug}"
APP="build/MLXTurbo.app"

swift build -c "$CONFIG"
BIN="$(swift build -c "$CONFIG" --show-bin-path)/MLXTurbo"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/MLXTurbo"
cp Resources/Info.plist "$APP/Contents/Info.plist"

pick_identity() {
  local want line
  for want in "Developer ID Application" "Apple Development"; do
    line=$(security find-identity -v -p codesigning 2>/dev/null | grep "$want" | head -1) || true
    if [[ -n "$line" ]]; then
      print -r -- "${${line#*\"}%\"}"
      return
    fi
  done
  print -r -- "-"
}

IDENTITY="${IDENTITY:-$(pick_identity)}"
echo "署名: $IDENTITY"
# Hardened Runtime は公証の必須要件。ad-hoc でも付けて挙動を揃える。
codesign --force --options runtime --timestamp --sign "$IDENTITY" "$APP" 2>/dev/null \
  || codesign --force --options runtime --sign "$IDENTITY" "$APP"

codesign --verify --strict --verbose=2 "$APP" 2>&1 | sed 's/^/  /'
echo "→ $(pwd)/$APP"

if [[ "${NOTARIZE:-0}" == "1" ]]; then
  case "$IDENTITY" in
    "Developer ID Application"*) ;;
    *) echo "公証には Developer ID Application が要ります (いまは $IDENTITY)"; exit 1 ;;
  esac
  ditto -c -k --keepParent "$APP" build/MLXTurbo.zip
  # 資格情報は keychain のプロファイルから読む。ここには書かない:
  #   xcrun notarytool store-credentials mlxturbo --apple-id ... --team-id ...
  xcrun notarytool submit build/MLXTurbo.zip --keychain-profile mlxturbo --wait
  xcrun stapler staple "$APP"
  spctl -a -vvv "$APP"
fi
