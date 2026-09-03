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

LOCK="${TMPDIR:-/tmp}/mlxturbo-bigmodel.lock"
WAITED=0

# ---- 優先レーン ------------------------------------------------------------
#
# モデルを読まない数分の micro が、モデルを読む 20 分の A/B の後ろで 40 分待つ
# のは無駄 (エージェントがその間止まる)。micro は札 (ticket) を出し、札が
# 生きている間は普通の待ち手がロックを取らない。micro どうしは先着順。
# 自動判定: コマンドに _micro.py / kernel_chain_cost.py / micro_kernel_latency.py /
# verify_*.py / smoke_*.py が含まれれば優先。BIGLOCK_PRIO=1/0 で明示できる。
PRIO_DIR="${TMPDIR:-/tmp}/mlxturbo-biglock-prio"
mkdir -p "$PRIO_DIR"
PRIO="${BIGLOCK_PRIO:-}"
if [ -z "$PRIO" ]; then
  case "$*" in
    *_micro.py*|*kernel_chain_cost.py*|*micro_kernel_latency.py*|*/verify_*.py*|*/smoke_*.py*|*[[:space:]]verify_*.py*|*[[:space:]]smoke_*.py*) PRIO=1 ;;
    *) PRIO=0 ;;
  esac
fi
TICKET="$PRIO_DIR/$$"
if [ "$PRIO" = 1 ]; then
  echo $$ > "$TICKET"
  POLL=3
else
  POLL=15
fi
trap 'rm -f "$TICKET"' EXIT INT TERM
# 生きている優先札 (自分以外) があるか。死んだ札は掃除する
live_prio() {
  local t o
  for t in "$PRIO_DIR"/*(N); do
    o=$(basename "$t")
    [ "$o" = "$$" ] && continue
    if kill -0 "$o" 2>/dev/null; then return 0; else rm -f "$t"; fi
  done
  return 1
}

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
      sleep $POLL; WAITED=$((WAITED + POLL)); continue
    fi
  fi

  # 2. ロックを持たない相手 (まだこの仕組みを使っていない側) も見る。
  #
  # **待っている biglock.sh 自身を数えないこと。**待機中の wrapper の
  # コマンドラインには引数として `.venv/bin/python tools/...` が丸ごと入って
  # いるので、素の pgrep だと待ち合わせている者どうしが互いを「ロック無しで
  # 走っている python」と見なして永久に待つ (実際に 4 本並べて 47 分空転した)。
  OTHER=""
  for _pid in $(pgrep -f "\.venv/bin/python3? (tools|bench)/"); do
    [ "$_pid" = "$$" ] && continue
    case "$(ps -o command= -p "$_pid" 2>/dev/null)" in
      *biglock.sh*) continue ;;
    esac
    OTHER=$_pid
    break
  done
  if [ -n "$OTHER" ]; then
    [ "$WAITED" -eq 0 ] && \
      echo "biglock: ロック無しの pid=$OTHER が走っている。空くまで待つ" >&2
    sleep 15; WAITED=$((WAITED + 15)); continue
  fi

  # 3. 優先札が生きていれば、普通の待ち手は譲る
  if [ "$PRIO" != 1 ] && live_prio; then
    [ "$WAITED" -eq 0 ] && echo "biglock: 優先の micro が待っている。先に通す" >&2
    sleep $POLL; WAITED=$((WAITED + POLL)); continue
  fi

  # 4. 取得。既に誰かが作っていれば失敗するので、その場合は待ちに戻る
  if (set -o noclobber; echo $$ > "$LOCK") 2>/dev/null; then
    rm -f "$TICKET"
    break
  fi
  sleep 5; WAITED=$((WAITED + 5))
done

[ "$WAITED" -gt 0 ] && echo "biglock: ${WAITED}s 待った。開始する" >&2

# ---- メモリが空くのを待つ ----------------------------------------------
#
# 91GB のモデル + n-gram の RAM テーブル 32GB で 123GB / 128GB。**直前の
# 実行のページが返る前に次を始めると、丸ごと圧縮領域に落ちてスラッシングする。**
# 実測: 空き 14MB / 圧縮 62GB / 伸長 21 億回まで行き、13 分間 CPU を回して
# 何も進まなかった。プロセスを落としたら空き 116GB に戻ったので、待てば済む。
#
# 他のプロセス (Claude Code の多重起動を含む) がメモリを握っている場合も
# 同じ待ちで守れる。ここは 91GB を読む全経路が通る唯一の場所。
MEM_NEED_GB="${MLXTURBO_MIN_FREE_GB:-100}"
MEM_WAITED=0
while [ "$MEM_WAITED" -lt 600 ]; do
  # 空き + 非活性 (回収可能) を見る。圧縮済みは「使用中」なので数えない
  FREE_GB=$(vm_stat | awk '
    /Pages free/        {f=$3}
    /Pages inactive/    {i=$3}
    /Pages speculative/ {s=$3}
    END {gsub(/\./,"",f); gsub(/\./,"",i); gsub(/\./,"",s);
         printf "%d", (f+i+s)*16384/1073741824}')
  [ "$FREE_GB" -ge "$MEM_NEED_GB" ] && break
  [ "$MEM_WAITED" -eq 0 ] &&     echo "biglock: 回収可能メモリ ${FREE_GB}GB < ${MEM_NEED_GB}GB。空くまで待つ" >&2
  sleep 15
  MEM_WAITED=$((MEM_WAITED + 15))
done
if [ "$MEM_WAITED" -ge 600 ]; then
  echo "biglock: 警告 -- 10 分待ってもメモリが空かない (${FREE_GB}GB)。" >&2
  echo "biglock:   このまま始めるとスワップで空転する可能性が高い。" >&2
  echo "biglock:   他のプロセス (Claude Code の多重起動など) を確認すること。" >&2
elif [ "$MEM_WAITED" -gt 0 ]; then
  echo "biglock: メモリ待ち ${MEM_WAITED}s (回収可能 ${FREE_GB}GB)。開始する" >&2
fi
trap 'rm -f "$LOCK" "$TICKET"' EXIT INT TERM

"$@"
