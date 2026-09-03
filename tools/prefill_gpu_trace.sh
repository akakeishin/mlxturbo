#!/bin/zsh
# 17k prefill を xctrace (Metal System Trace) で録る。
#
# 目的: 冷 prefill が mlx-serve に 5% 負けている残差が「チャンク/グループ境界
# (mx.eval + clear_cache) や層の間の泡 (GPU に何も入っていない区間)」なのか
# 「カーネルが GPU に張り付いている演算そのもの」なのかを、GPU タイムラインの
# busy/idle で切り分ける。decode のラウンド間の泡 7.3ms を xctrace で特定した
# のと同じ手口 (docs/research/KERNEL-BRIEF-DECODE-BW.md 2026-08-31 の記録) を
# prefill に向ける。解析は同じ tools/ の parse_gpu_trace.py (別ファイル) が行う。
# このスクリプトは「録る」ところまで。
#
# **注意 (2026-09-02 14:30 の実測、docs/research/SESSION-2026-09-02-CATCHUP.md
# 「xctrace は今回は使えなかった」節):** 同じ xctrace (Metal System Trace、
# --launch、32 秒) を decode ループの python (MLX) プロセスに向けたとき、
# metal-gpu-intervals に python の Compute 区間がほぼ出なかった (49k 行中 87
# 行、大半は Claude Helper の描画)。--attach は uv の python に "Cannot find
# process" で失敗した。90 秒版は 19GB になり後処理が終わらなかった。
# **この道具が同じ壁に当たる可能性は高い。**prefill は decode のラウンド境界
# より GPU 稼働が連続的で長いので結果が変わるかもしれないが、未検証。
# busy_span_ratio が 0 に近い、または区間数が極端に少ない結果が出たら、まず
# この前例を疑うこと (parse_gpu_trace.py もこの旨の警告を出す)。空振りなら
# 道具として残すだけで、読解ベースの切り分けに戻る (CLAUDE.md の「触ると
# 壊れるもの」節、prefill 側の写しの調査と同じやり方)。
#
# 事前確認 (このリポジトリにまだ無い情報。実行前に必ず確認すること):
#   1. この Mac にある xctrace テンプレート一覧:
#        xcrun xctrace list templates
#      "Metal System Trace" が無ければ近いもの ("Game Performance" など) を
#      --template で指定し直す。
#   2. テーブル (schema) 名の確認は録ったあとに:
#        xcrun xctrace export --input <trace> --toc
#      (または tools/gpu_trace_kernels.py toc --trace <trace>)
#      "metal-gpu-intervals" を前提にしているが (tools/gpu_trace_kernels.py が
#      2026-09-02 に実 trace で確認済みの名前)、無ければ toc の一覧から選び
#      直して parse_gpu_trace.py の --schema に渡す。
#   3. 初回は --ctx を小さく (例 4000) して 1 回動かし、.trace の大きさと
#      export にかかる時間を見てから 17k に上げる (上の「19GB で後処理が
#      終わらなかった」を踏まないため)。
#
# 使い方 (GPU を使う。他の計測やダウンロードと並走させないこと):
#
#   tools/prefill_gpu_trace.sh --model ~/models/ddalcu-mlxlm \
#       --ngram ~/models/ddalcu-ngram
#
#   小さい文脈で下見してから:
#   tools/prefill_gpu_trace.sh --model ~/models/ddalcu-mlxlm \
#       --ngram ~/models/ddalcu-ngram --ctx 4000 --time-limit 60s
#
#   このスクリプト自身が既定で tools/biglock.sh 経由で起動する (98GB モデルの
#   同時ロードを避けるため)。外側で既に biglock を掛けて呼ぶ場合は二重がけに
#   なるので --no-biglock を付けること (biglock.sh は「ロック無しで走っている
#   python」を pgrep で探すので、待機中の biglock.sh 同士が誤検知しうる --
#   biglock.sh 冒頭のコメント参照)。
#
# 出力:
#   bench/results/traces/<tag>.trace   -- xctrace の録画本体 (ディレクトリ)
#   bench/results/logs/<tag>.log       -- MLXTURBO_PREFILL_TRACE=1 の
#       stdout+stderr 結合ログ (境界時刻の突き合わせに使う)。[prefill] 行は
#       print() の既定で stdout に出るが、[transformers]/[decode_ab]/biglock:
#       等は stderr のことがあるので、内部で 2>&1 にまとめて 1 ファイルへ
#       落とす (xctrace --launch 経由でも子プロセスの標準出力の転送に頼らず、
#       起動コマンド自体をシェルでラップしてリダイレクトする -- --launch が
#       子の stdio をどこまで親に転送するかは未確認のため、依存しない作り
#       にしてある)。
#
# --prefill-once について (decode_ab.py 側の既定の挙動、触っていない):
#   1 プロセス内で prefill が **2 回** 起きる。(1) 温めの 1 回
#   (run_once(eng, ids, 32, ...) が独自に prefill してから 32 トークン decode
#   する。tools/decode_ab.py の `for want in ("short","long")` ループ、
#   --prefill-once の有無に関係なく毎回走る)、(2) 比較用の共有 prefill
#   (--prefill-once が無ければ ABBA の回文順 (variants+variants[::-1]) で
#   4 回に増える。decode_ab.py に --reps は無い。--only long で 1 プロンプト
#   に絞っても A,B,B,A の 4 回が既定 -- --prefill-once はそのうち 3 回を
#   「共有 prefill から decode だけ再開」に変えて実質 1 回に潰す)。
#   なので --prefill-once を付けても [prefill] のログは 2 回分 (温め +
#   共有) 出る。parse_gpu_trace.py は "[prefill] total dur_sum=..." 行で
#   区切って複数回に対応しており、--pf-occurrence で選べる (既定 -1 = 最後 =
#   共有 prefill。過去の解析 (SESSION-2026-09-02-CATCHUP.md の「4 本目」) も
#   同じ流儀で最後の occurrence を読んでいる)。

set -e

REPO_ROOT="${0:A:h:h}"
cd "$REPO_ROOT"

MODEL=""
NGRAM=""
CTX=17000
TOKENS=8
TEMPLATE="Metal System Trace"
TIME_LIMIT="200s"
USE_BIGLOCK=1
DRY_RUN=0
TAG=""
MODE="launch"   # launch (既定、推奨) / attach (過去に失敗した実績あり)
ATTACH_PID=""

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --ngram) NGRAM="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --tokens) TOKENS="$2"; shift 2 ;;
    --template) TEMPLATE="$2"; shift 2 ;;
    --time-limit) TIME_LIMIT="$2"; shift 2 ;;
    --no-biglock) USE_BIGLOCK=0; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --attach-pid) ATTACH_PID="$2"; MODE="attach"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,80p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "未知の引数: $1 (--help でヘルプ)" >&2
      exit 1
      ;;
  esac
done

if [ -z "$MODEL" ]; then
  echo "--model が要る (例 --model ~/models/ddalcu-mlxlm)" >&2
  exit 1
fi
if [ "$MODE" != "launch" ] && [ "$MODE" != "attach" ]; then
  echo "--mode は launch か attach (指定: $MODE)" >&2
  exit 1
fi
if [ "$MODE" = "attach" ] && [ -z "$ATTACH_PID" ]; then
  echo "--mode attach には --attach-pid PID が要る" >&2
  exit 1
fi

DATE_TAG="$(date +%Y%m%d-%H%M%S)"
[ -z "$TAG" ] && TAG="prefill-${CTX}-${DATE_TAG}"

TRACES_DIR="bench/results/traces"
LOGS_DIR="bench/results/logs"
mkdir -p "$TRACES_DIR" "$LOGS_DIR"

TRACE_OUT="${TRACES_DIR}/${TAG}.trace"
LOG_OUT="${LOGS_DIR}/${TAG}.log"

if [ -e "$TRACE_OUT" ]; then
  echo "$TRACE_OUT が既にある。--tag を変えるか消してから流すこと" >&2
  exit 1
fi

CMD_PY=(.venv/bin/python tools/decode_ab.py
  --knob null --model "$MODEL"
  --only long --ctx "$CTX" --tokens "$TOKENS" --prefill-once)
[ -n "$NGRAM" ] && CMD_PY+=(--ngram "$NGRAM")

if [ "$USE_BIGLOCK" = "1" ]; then
  CMD=(tools/biglock.sh "${CMD_PY[@]}")
else
  CMD=("${CMD_PY[@]}")
fi

echo "録る対象コマンド (MLXTURBO_PREFILL_TRACE=1 付き):" >&2
echo "  env MLXTURBO_PREFILL_TRACE=1 ${CMD[*]}" >&2
echo "ログ  -> $LOG_OUT" >&2
echo "trace -> $TRACE_OUT" >&2

if [ "$MODE" = "attach" ]; then
  echo "--mode attach: 2026-09-02 に uv の python への --attach が" >&2
  echo "  \"Cannot find process\" で失敗した実測がある。--attach-pid には" >&2
  echo "  ラッパー (biglock.sh) ではなく実際の python インタプリタの PID を" >&2
  echo "  渡すこと (--no-biglock でラッパーを外すと分かりやすい)。" >&2
  XCTRACE_CMD=(xcrun xctrace record --template "$TEMPLATE"
    --time-limit "$TIME_LIMIT" --output "$TRACE_OUT" --no-prompt
    --attach "$ATTACH_PID")
  echo "\$ ${XCTRACE_CMD[*]}"
  if [ "$DRY_RUN" = "1" ]; then
    echo "(--dry-run: このプロセスの起動と xctrace の実行はしない)"
    exit 0
  fi
  # attach は既に生きているプロセスを観測するだけなので、対象は自前で
  # バックグラウンド起動して stdout+stderr を直接リダイレクトできる
  # (launch と違いシェルラップは不要)。
  env MLXTURBO_PREFILL_TRACE=1 "${CMD[@]}" > "$LOG_OUT" 2>&1 &
  PY_BG_PID=$!
  "${XCTRACE_CMD[@]}"
  wait "$PY_BG_PID"
else
  # launch は xctrace 自身がプロセスを起こす。子の stdout/stderr が xctrace
  # 経由でどこまで親に転送されるか確認していないので、それに頼らず
  # 起動コマンド自体を zsh -c でラップしてリダイレクトを内包させる
  # (${(q)a} で 1 引数ずつ安全にクォートする)。
  INNER_ARGS=(env "MLXTURBO_PREFILL_TRACE=1" "${CMD[@]}")
  INNER_STR=""
  for a in "${INNER_ARGS[@]}"; do
    INNER_STR+="${(q)a} "
  done
  INNER_STR+="> ${(q)LOG_OUT} 2>&1"

  XCTRACE_CMD=(xcrun xctrace record --template "$TEMPLATE"
    --time-limit "$TIME_LIMIT" --output "$TRACE_OUT" --no-prompt
    --launch -- /bin/zsh -c "$INNER_STR")
  echo "\$ ${XCTRACE_CMD[*]}"
  if [ "$DRY_RUN" = "1" ]; then
    echo "(--dry-run: xctrace は実行しない)"
    exit 0
  fi
  "${XCTRACE_CMD[@]}"
fi

echo "録画完了。次は:" >&2
echo "  xcrun xctrace export --input $TRACE_OUT --toc   # テーブル名の確認" >&2
echo "  .venv/bin/python tools/parse_gpu_trace.py --trace $TRACE_OUT --pf-log $LOG_OUT" >&2
