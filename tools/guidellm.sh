#!/bin/zsh
# GuideLLM を再現可能な版と macOS 向けの安全な multiprocessing 設定で動かす。

set -eu

ROOT="${0:a:h:h}"
VENV="$ROOT/tools/compare/guidellm-venv"
BIN="$VENV/bin/guidellm"
VERSION="0.7.3"

if [ "${1:-}" = "setup" ]; then
  uv venv "$VENV" --python "$ROOT/.venv/bin/python"
  uv pip install --python "$VENV/bin/python" "guidellm==$VERSION"
  "$BIN" --version
  exit 0
fi

if [ ! -x "$BIN" ]; then
  echo "GuideLLM が未導入です。先に tools/guidellm.sh setup を実行してください。" >&2
  exit 1
fi

# GuideLLM 0.7.3 の既定 fork は、macOS + torch で worker が SIGSEGV する。
# spawn と main-process data loader なら同じ実エンドポイント煙試験が通る。
if [ "${1:-}" = "run" ]; then
  shift
  exec env GUIDELLM__MP_CONTEXT_TYPE=spawn "$BIN" run \
    --data-loader kind=pytorch,num_workers=0 "$@"
fi

exec "$BIN" "$@"
