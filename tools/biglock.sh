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

# ロックを取るまで回る。取得は noclobber で不可分にする。`[ -f ]` で見てから
# 書く形だと、同時に待っていた 2 本が両方通って両方 98GB を読みに行く
while true; do
  # 1. 既存のロックが生きているか。死んでいれば掃除する
  if [ -f "$LOCK" ]; then
    OWNER=$(head -1 "$LOCK" 2>/dev/null)
    if [ -z "$OWNER" ] || ! kill -0 "$OWNER" 2>/dev/null; then
      echo "biglock: 死んだロック (pid=$OWNER) を掃除する" >&2
      rm -f "$LOCK"
    else
      [ "$WAITED" -eq 0 ] && \
        echo "biglock: pid=$OWNER が 98GB を抱えている。空くまで待つ" >&2
      sleep 15; WAITED=$((WAITED + 15)); continue
    fi
  fi

  # 2. ロックを持たない相手 (まだこの仕組みを使っていない側) も見る
  OTHER=$(pgrep -f "\.venv/bin/python3? (tools|bench)/" | head -1)
  if [ -n "$OTHER" ]; then
    [ "$WAITED" -eq 0 ] && \
      echo "biglock: ロック無しの pid=$OTHER が走っている。空くまで待つ" >&2
    sleep 15; WAITED=$((WAITED + 15)); continue
  fi

  # 3. 取得。既に誰かが作っていれば失敗するので、その場合は待ちに戻る
  if (set -o noclobber; echo $$ > "$LOCK") 2>/dev/null; then
    break
  fi
  sleep 5; WAITED=$((WAITED + 5))
done

[ "$WAITED" -gt 0 ] && echo "biglock: ${WAITED}s 待った。開始する" >&2
trap 'rm -f "$LOCK"' EXIT INT TERM

"$@"
