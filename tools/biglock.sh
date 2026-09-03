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

# ---- レーン (3 段) ---------------------------------------------------------
#
# 方針: 親の A/B (run_chainNN.sh から流す長い列) が 1 本終わったら、エージェントの
# GPU 待ちを全部先に流してから次の A/B に進む。エージェントどうしでは、モデルを
# 読まない数分の micro を、モデルを読む A/B より先に通す。
#
#   2 = micro  : コマンドに _micro.py / kernel_chain_cost.py / micro_kernel_latency.py /
#                verify_*.py / smoke_*.py が含まれる (自動)
#   1 = normal : それ以外 (既定。エージェントの in-model A/B など)
#   0 = bg     : 祖先に run_chain*.sh がいる親の列 (自動)
#
# BIGLOCK_PRIO=0/1/2 で明示できる。各待ち手は札 (PRIO_DIR/<pid>、中身が段) を出し、
# 自分より高い段の札が生きている間はロックを取らない。同じ段は先着順。
PRIO_DIR="${TMPDIR:-/tmp}/mlxturbo-biglock-prio"
mkdir -p "$PRIO_DIR"
in_chain() {
  local p=$$ c
  while [ -n "$p" ] && [ "$p" -gt 1 ]; do
    c=$(ps -o command= -p "$p" 2>/dev/null)
    case "$c" in
      *zsh\ /*/run_chain*.sh*|*zsh\ run_chain*.sh*|*zsh\ ./run_chain*.sh*) return 0 ;;
    esac
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  done
  return 1
}
PRIO="${BIGLOCK_PRIO:-}"
if [ -z "$PRIO" ]; then
  case "$*" in
    *_micro.py*|*kernel_chain_cost.py*|*micro_kernel_latency.py*|*/verify_*.py*|*/smoke_*.py*|*[[:space:]]verify_*.py*|*[[:space:]]smoke_*.py*) PRIO=2 ;;
    *) if in_chain; then PRIO=0; else PRIO=1; fi ;;
  esac
fi
TICKET="$PRIO_DIR/$$"
echo "$PRIO" > "$TICKET"
case "$PRIO" in 2) POLL=3 ;; 1) POLL=5 ;; *) POLL=15 ;; esac
trap 'rm -f "$TICKET"' EXIT
trap 'rm -f "$TICKET"; exit 143' INT TERM

# ---- 常駐 worker が居れば、その 1 本の列に乗せる --------------------------
#
# worker (tools/ab_daemon.py) は 98GB を読んだまま常駐していて、decode_ab は
# in-process、モデルを読まない道具 (micro / verify / smoke / 連鎖) は
# subprocess で回す。**ここを通すので、chain スクリプトもエージェントも
# 呼び方を変えずに全部その 1 本の列に乗る。**段と先着順はそのまま渡す。
#
# 乗らないもの (self_snapshot、mlx-serve、worker が別のモデルを抱えている、
# 道具がまだ run_with_model を持っていない等) は 64 が返るので、そのまま
# 従来どおり自分でロックを取る (その場合は下で worker に「降りろ」を伝える)。
#
# **委譲する前に自分の札を消すこと。**残すと worker の live_higher が
# 「同じ段の古い札」としてこちらを待ち、こちらは worker の完了を待つ
# (両すくみ)。載らなかったら並び直す。
# BIGLOCK_NO_WORKER=1 で従来の経路に固定できる。
AB_PID_FILE="${MLXTURBO_AB_PID:-${TMPDIR:-/tmp}/mlxturbo-ab-daemon.pid}"
AB_STOP_FILE="${AB_PID_FILE%.pid}.stop"
AB_DIR="${0:a:h}"
AB_PY="${AB_DIR:h}/.venv/bin/python"
HEAVY=0
case "$*" in *tools/decode_ab.py*|*tools/longctx_quality.py*|*tools/moe_split.py*|*bench/quant_eval.py*) HEAVY=1 ;; esac
# worker が居れば routable なもの全部、居なくてもモデルを読む routable なもの (ab_submit が worker を立てる) は worker へ
if [ -z "$BIGLOCK_NO_WORKER" ] && [ -x "$AB_PY" ] && { [ -f "$AB_PID_FILE" ] || [ "$HEAVY" = 1 ]; }; then
  rm -f "$TICKET"
  AB_RC=0
  "$AB_PY" "$AB_DIR/ab_submit.py" --from-biglock --prio "$PRIO" -- "$@" || AB_RC=$?
  if [ "$AB_RC" -ne 64 ]; then
    exit $AB_RC
  fi
  echo "$PRIO" > "$TICKET"
fi
# 自分より先に通すべき札 (自分以外) が生きているか: 上の段、または同じ段で自分より古い札。
# 同じ段を先着順にしないと、poll の短い新しい待ち手が古い待ち手を追い越し続ける (2 時間待ちの前例)。
# 死んだ札は掃除する
live_higher() {
  local t o lv
  for t in "$PRIO_DIR"/*(N); do
    o=$(basename "$t")
    [ "$o" = "$$" ] && continue
    if kill -0 "$o" 2>/dev/null; then
      lv=$(cat "$t" 2>/dev/null); [ -z "$lv" ] && lv=1
      [ "$lv" -gt "$PRIO" ] && return 0
      [ "$lv" -eq "$PRIO" ] && [ "$t" -ot "$TICKET" ] && return 0
    else
      rm -f "$t"
    fi
  done
  return 1
}
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
      # 常駐 worker (tools/ab_daemon.py) は 98GB を抱えたまま何時間も居るが、
      # ジョブを走らせる間だけこの LOCK を正規に取る。投入側 (ab_submit.py)
      # は待っているだけ。**どちらも「ロック無しで走っている python」では
      # ない** -- 数えると全員が永久に待つ (待機中の wrapper と同じ罠)。
      *ab_daemon.py*|*ab_submit.py*) continue ;;
    esac
    OTHER=$_pid
    break
  done
  if [ -n "$OTHER" ]; then
    [ "$WAITED" -eq 0 ] && \
      echo "biglock: ロック無しの pid=$OTHER が走っている。空くまで待つ" >&2
    sleep 15; WAITED=$((WAITED + 15)); continue
  fi

  # 3. 自分より高い段の待ち手がいれば譲る
  if live_higher; then
    [ "$WAITED" -eq 0 ] && echo "biglock: 段 $PRIO。上の段の待ち手を先に通す" >&2
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

# ---- 常駐 worker (tools/ab_daemon.py) に 98GB を返させる ----------------
#
# daemon は 98GB を抱えたまま居座る。**ここを通る段 0/1 は「モデルを読む」
# 仕事**なので (小ベンチ / フルベンチ、mlx-serve、quant_eval、longctx_quality)、
# そのままだと下のメモリ待ちが必ず 10 分空振りして、そのあとスラッシングする。
# ロックを取った直後 (メモリ待ちの前) に降りるよう伝え、pid が消えるまで待つ。
#
# daemon は実行中のジョブがあればそれを終えてから降りる。ただしジョブ実行中は
# daemon がこの LOCK を持っているので、ここに来た時点で相手は待機中か
# ロック待ちのどちらかで、実際にはすぐ降りる。次に ab_submit が来たら
# daemon は自分で立ち上がり直す。
# 段 2 (micro) は通らない -- あちらはモデルを読まないので追い出す理由が無い。
if [ "$PRIO" -lt 2 ] && [ -f "$AB_PID_FILE" ]; then
  AB_PID=$(sed -n 's/.*"pid"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$AB_PID_FILE" | head -1)
  if [ -n "$AB_PID" ] && kill -0 "$AB_PID" 2>/dev/null; then
    echo "biglock: 常駐 worker (pid=$AB_PID) に 98GB を返させる" >&2
    : > "$AB_STOP_FILE"
    AB_WAITED=0
    while kill -0 "$AB_PID" 2>/dev/null; do
      if [ "$AB_WAITED" -ge 1800 ]; then
        echo "biglock: 警告 -- worker (pid=$AB_PID) が 30 分たっても降りない。" >&2
        echo "biglock:   このまま進むとメモリ待ちで空転する。手で止めること:" >&2
        echo "biglock:   .venv/bin/python tools/ab_submit.py --stop" >&2
        break
      fi
      sleep 2; AB_WAITED=$((AB_WAITED + 2))
    done
    rm -f "$AB_STOP_FILE"
    [ "$AB_WAITED" -gt 0 ] && echo "biglock: worker が降りるのを ${AB_WAITED}s 待った" >&2
  fi
fi

# ---- メモリが空くのを待つ ----------------------------------------------
#
# 91GB のモデル + n-gram の RAM テーブル 32GB で 123GB / 128GB。**直前の
# 実行のページが返る前に次を始めると、丸ごと圧縮領域に落ちてスラッシングする。**
# 実測: 空き 14MB / 圧縮 62GB / 伸長 21 億回まで行き、13 分間 CPU を回して
# 何も進まなかった。プロセスを落としたら空き 116GB に戻ったので、待てば済む。
#
# 他のプロセス (Claude Code の多重起動を含む) がメモリを握っている場合も
# 同じ待ちで守れる。ここは 91GB を読む全経路が通る唯一の場所。
# 段 2 (micro) だけは 8GB でよい。モデルを読まない数分の仕事なのに、常駐
# worker (tools/ab_daemon.py) が 98GB を抱えている間は 100GB が空くことは
# 無く、毎回 10 分待って警告付きで始まることになる。段 0/1 (98GB を読む側)
# は 100 のまま -- あちらは本当に空きが要る。
if [ "$PRIO" -ge 2 ]; then
  MEM_NEED_GB="${MLXTURBO_MIN_FREE_GB:-8}"
else
  MEM_NEED_GB="${MLXTURBO_MIN_FREE_GB:-100}"
fi
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
trap 'rm -f "$LOCK" "$TICKET"' EXIT
trap 'rm -f "$LOCK" "$TICKET"; exit 143' INT TERM

"$@"
