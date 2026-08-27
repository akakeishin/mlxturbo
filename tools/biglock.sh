#!/bin/zsh
# 98GB のモデルを読む実行を直列化する。
#
# このマシン (128GB) では 98GB のモデルは同時に 1 本しか載らない。親と
# カーネルセッションが両方読むと、後から来た方ではなく**先に走っていた方**が
# メモリ圧で落ちる。実際に sweep が 2 回落ちた (30 分 x 2)。
#
# 使い方:
#   tools/biglock.sh uv run python bench/quant_eval.py sweep --model ...
#
# 先に走っているものが終わるまで待ってから、ロックを取って実行する。
# ロックはプロセスが死んでも残らない (PID を書いて生存確認する)。

set -e

LOCK="${TMPDIR:-/tmp}/fastmlx-bigmodel.lock"
WAITED=0

# 1. 既存のロックが生きているか確認し、生きていれば待つ
while [ -f "$LOCK" ]; do
  OWNER=$(cat "$LOCK" 2>/dev/null | head -1)
  if [ -z "$OWNER" ] || ! kill -0 "$OWNER" 2>/dev/null; then
    echo "biglock: 死んだロック (pid=$OWNER) を掃除する" >&2
    rm -f "$LOCK"
    break
  fi
  if [ "$WAITED" -eq 0 ]; then
    echo "biglock: pid=$OWNER が 98GB を抱えている。空くまで待つ" >&2
  fi
  sleep 15
  WAITED=$((WAITED + 15))
done

# 2. ロックを持たない相手 (まだこの仕組みを使っていない側) も見る
while true; do
  OTHER=$(pgrep -f "\.venv/bin/python3? (tools|bench)/" | head -1)
  [ -z "$OTHER" ] && break
  if [ "$WAITED" -eq 0 ]; then
    echo "biglock: ロック無しの pid=$OTHER が走っている。空くまで待つ" >&2
  fi
  sleep 15
  WAITED=$((WAITED + 15))
done

[ "$WAITED" -gt 0 ] && echo "biglock: ${WAITED}s 待った。開始する" >&2

echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

"$@"
