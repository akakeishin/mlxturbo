#!/bin/zsh
# 27B の 5 エンジン基準測定 (同じ重み・同じ池・同じ harness)。
#
#   同じ重み: ~/models/qwen38-27b-4bit (+ MTP サイドカー ~/models/qwen38-27b-mtp)
#   同じ経路: bench/bench_http_engine.py (OpenAI 互換 SSE)
#   測るもの: 文脈 4000 / 17000 / 32000 の 冷 TTFT / 温 TTFT / decode tok/s
#
# 使い方 (**biglock は外で 1 回だけ取る。**この中では取らない —
# 冷却窓ごと直列にしたいので、列の途中で他の GPU 仕事に割り込まれては困る):
#
#   BIGLOCK_NO_WORKER=1 tools/biglock.sh bench/run_27b_baseline.sh <tag>
#   BIGLOCK_NO_WORKER=1 tools/biglock.sh bench/run_27b_baseline.sh <tag> reverse
#
# **2 周目は `reverse` で回す。**エンジンの順序は熱の偏りをそのまま順位に変える
# (1 番目が一番冷えている)。正順と逆順の 2 周で打ち消すこと。
#
# 出力: bench/results/baseline-27b-<engine>-<tag>.json
#       bench/results/logs/baseline-27b-<engine>-<tag>.{out,log}
#       bench/results/thermal-probe.csv (追記)
#
# 前提 (満たしていないと先頭で止まる):
#   - ~/.omlx/model_settings.json に oMLX の MTP ペア付けが書いてある
#   - ~/models/qwen38-27b-mtp-prefixed/mtp.safetensors (mtp. 接頭辞の写し) がある
#   - ~/models/{omlx-27b,mlxserve-27b,mtplx-27b} の symlink ディレクトリがある
#   詳しくは docs/research/COMPARE-QUEUE.md の「27B の 5 者の起動コマンドと状態」。

set -e
cd /Users/ht/dev/fastmlx

TAG="${1:?使い方: bench/run_27b_baseline.sh <tag> [reverse]}"
ORDER="${2:-forward}"

P=/Users/ht/dev/fastmlx/.venv/bin/python
H=/Users/ht/dev/fastmlx/bench/bench_http_engine.py
M=$HOME/models
L=bench/results/logs
PROBE=bench/results/thermal-probe.csv
CTXS="${CTXS:-4000,17000,32000}"
TOKENS="${TOKENS:-256}"
REPS="${REPS:-1}"
WARM_LONG="${WARM_LONG:-4000}"       # 測定前に長い prefill を 1 本流して重みをページインする
GAP_S="${GAP_S:-180}"                # エンジン間の間隔 (3 分)
export HF_HUB_OFFLINE=1              # mlx_lm.server / oMLX が HF に取りに行かないように
export PATH="/opt/homebrew/bin:$PATH"  # omlx

mkdir -p $L bench/results

# ---- 前提の確認 -----------------------------------------------------------
fail() { echo "前提が満たされていない: $1" >&2; exit 1; }
[ -f "$HOME/.omlx/model_settings.json" ] || fail "~/.omlx/model_settings.json (oMLX の MTP ペア付け)"
grep -q vlm_mtp_draft_model "$HOME/.omlx/model_settings.json" || fail "~/.omlx/model_settings.json に vlm_mtp_draft_model が無い"
[ -f "$M/qwen38-27b-mtp-prefixed/mtp.safetensors" ] || fail "$M/qwen38-27b-mtp-prefixed/mtp.safetensors (mtp. 接頭辞の写し)"
for d in omlx-27b mlxserve-27b mtplx-27b; do [ -d "$M/$d" ] || fail "$M/$d"; done
[ -e "$M/mlxserve-27b/mtp.safetensors" ] || fail "$M/mlxserve-27b/mtp.safetensors"
[ -e "$M/mtplx-27b/mtp.safetensors" ] || fail "$M/mtplx-27b/mtp.safetensors"

# ---- 常駐 worker を降ろす (98GB を抱えたままだと 15GB のエンジンが載らない) ----
pkill -f "mlxturbo.server" 2>/dev/null || true
$P tools/ab_submit.py --stop 2>/dev/null || true
for i in {1..30}; do pgrep -f "tools/ab_daemon.py" >/dev/null || break; sleep 2; done
if pgrep -f "tools/ab_daemon.py" >/dev/null; then echo "警告: worker が残っている" >&2; fi

# ---- エンジンごとの起動 argv ------------------------------------------------
# 中身と根拠は docs/research/COMPARE-QUEUE.md。ここを直したらあちらも直す。
engine_spec() {
  case "$1" in
    mlxturbo)
      PORT=8151; MODEL_ID=""; THINK=reasoning_effort
      ARGV="$P -m mlxturbo.server --model $M/qwen38-27b-4bit --mtp $M/qwen38-27b-mtp --host 127.0.0.1 --port 8151"
      ;;
    mlx-lm)
      # /v1/models が HF キャッシュを全部並べるので model-id を固定する
      PORT=8152; MODEL_ID="$M/qwen38-27b-4bit"; THINK=chat_template_kwargs
      ARGV="/Users/ht/dev/fastmlx/.venv/bin/mlx_lm.server --model $M/qwen38-27b-4bit --host 127.0.0.1 --port 8152 --log-level INFO"
      ;;
    omlx)
      PORT=8153; MODEL_ID="qwen38-27b-4bit"; THINK=chat_template_kwargs
      ARGV="/opt/homebrew/bin/omlx serve --model-dir $M/omlx-27b --host 127.0.0.1 --port 8153 --no-hf-cache --log-level info"
      ;;
    mtplx)
      # mtplx quickstart --model $M/mtplx-27b --mtp --dry-run --json の server_command と同じ。
      # **--chat-template-profile だけ tokenizer に戻してある** (quickstart は MTP 時に
      # local_qwen36 を選ぶが、それだと 5 者でレンダリング後のプロンプトが揃わない)。
      PORT=8154; MODEL_ID="mtplx-27b"; THINK=chat_template_kwargs
      ARGV="/Users/ht/dev/fastmlx/tools/compare/mtplx-venv/bin/python -m mtplx.server.openai --model $M/mtplx-27b --backend-id qwen3_next --host 127.0.0.1 --port 8154 --depth 3 --generation-mode mtp --profile sustained --reasoning-mode auto --preserve-thinking auto --verify-strategy capture_commit --verify-core linear-gdn-from-conv-tape --draft-lm-head-bits 4 --draft-lm-head-group-size 64 --draft-lm-head-mode affine --rate-limit 0 --stream-interval 1 --scheduler-mode serial --batching-preset latency --mtp-batch-numerics throughput --warmup-tokens 16 --model-id mtplx-27b --paged-kv-quantization off --fan-mode default --retrieval-max-resident 2 --ssd-session-cache on --ssd-session-cache-max-size 100GB --ssd-session-cache-min-prefix-tokens 512 --draft-temperature 0.6 --draft-top-p 0.95 --draft-top-k 20 --draft-sampler-source default --tool-prompt-mode hybrid --chat-template-profile tokenizer --temperature 0.6 --top-p 0.95 --top-k 20 --reasoning-parser qwen3 --reasoning-effort auto --no-stats-footer"
      ;;
    mlx-serve)
      PORT=8155; MODEL_ID=""; THINK=reasoning_effort
      ARGV="/Users/ht/dev/mlx-serve/zig-out/bin/mlx-serve --serve --model $M/mlxserve-27b --host 127.0.0.1 --port 8155 --mtp --log-level debug"
      ;;
    *) echo "未知のエンジン: $1" >&2; exit 1 ;;
  esac
}

ENGINES=(mlxturbo mlx-lm omlx mtplx mlx-serve)
if [ "$ORDER" = "reverse" ]; then
  ENGINES=(mlx-serve mtplx omlx mlx-lm mlxturbo)
fi

# ---- 冷却 10 分 (probe を 0/2/5/10 分に差す) --------------------------------
echo "== 冷却開始 $(date +%H:%M:%S)  tag=$TAG order=$ORDER"
[ -f $PROBE ] || $P tools/thermal_probe.py --header > $PROBE
$P tools/thermal_probe.py --seconds 10 --tag "$TAG 27b cool=0" >> $PROBE
for m in 2 5 10; do
  case $m in 2) s=110 ;; 5) s=170 ;; 10) s=290 ;; esac
  sleep $s
  $P tools/thermal_probe.py --seconds 10 --tag "$TAG 27b cool=${m}min" >> $PROBE
done
echo "== 冷却終了 $(date +%H:%M:%S)"; sysctl vm.swapusage

# ---- 本編 -----------------------------------------------------------------
first=1
for e in $ENGINES; do
  if [ $first -eq 0 ]; then
    echo "== 間隔 ${GAP_S}s $(date +%H:%M:%S)"
    sleep $GAP_S
    $P tools/thermal_probe.py --seconds 10 --tag "$TAG 27b before=$e" >> $PROBE
  fi
  first=0
  engine_spec $e
  OUT=bench/results/baseline-27b-$e-$TAG.json
  echo "== $e 開始 $(date +%H:%M:%S) -> $OUT"
  MID_ARG=()
  if [ -n "$MODEL_ID" ]; then MID_ARG=(--model-id "$MODEL_ID"); fi
  $P $H --engine $e --port $PORT --tokenizer $M/qwen38-27b-4bit \
      --argv "$ARGV" $MID_ARG \
      --thinking-how $THINK \
      --ctxs $CTXS --tokens $TOKENS --reps $REPS --warm-long $WARM_LONG \
      --server-log $L/baseline-27b-$e-$TAG.log \
      --out $OUT > $L/baseline-27b-$e-$TAG.out 2>&1 \
    || echo "!! $e が失敗した (続行): $L/baseline-27b-$e-$TAG.out" >&2
  grep "文脈" $L/baseline-27b-$e-$TAG.out | cut -c1-160 || true
  echo "== $e 終了 $(date +%H:%M:%S)"
done

echo "== 全部終わり $(date +%H:%M:%S)  tag=$TAG order=$ORDER"
echo "   結果: bench/results/baseline-27b-*-$TAG.json"
echo "   まだ 1 周。**逆順の 2 周目を回してから読むこと** (順序が熱の偏りを順位に変える)。"
