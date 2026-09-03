# 比較実行キュー（Phase E2 準備）

docs/research/PLAN.md Phase E2「mlx-lm 素 vs fastmlx vs MTPLX の同一マシン・同一プロンプト比較」
と Phase C1（KLD 品質ゲート）の実行準備。**このファイル自体は準備記録であって、
実行はしていない**（インストール・DL開始・ハーネス作成のみ。GPU ベンチ実行禁止の
指示のもとで作成）。

## やったこと

1. **MTPLX インストール**: `tools/compare/mtplx-venv/`（独立 venv、Python 3.13）
   に PyPI 版 `mtplx==2.9.2` を `[server]` extra 込みで pip install した。
   `/private/tmp/.../scratchpad/mtplx/` の clone とバージョン一致を確認済み
   （どちらも 2.9.2。clone は参照のみ、コードはコピーしていない）。
   fastmlx 本体の `.venv` には一切触れていない。
   動作確認は CPU のみ: `mtplx --help`, `mtplx hardware`, `mtplx ask --help`,
   `mtplx inspect --help`, `mtplx tune --help`, `mtplx quickstart --help`。
   `mtplx hardware` の出力はこのマシンの実チップ情報を返す
   （chip=Apple M3 Max, macOS=26.4, MLX=0.32.2 — README の実測環境と一致）。
2. **モデル取得を開始**（バックグラウンド、完了は待っていない）:
   `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`（HF API で実在確認済み。
   4-bit dynamic quant、base_model=Qwen/Qwen3.8-27B、約20.4GB、18 ファイル）。
   ログ: `tools/compare/hf-download-optimized-speed.log`。
   進捗確認:
   ```bash
   tail -20 ~/dev/fastmlx/tools/compare/hf-download-optimized-speed.log
   ps aux | grep "hf download" | grep -v grep
   du -sh ~/.cache/huggingface/hub/models--Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed 2>/dev/null
   ```
   完了後の確認（ローカル完結、ネットワークに出ない）:
   ```bash
   uv run --project ~/dev/fastmlx python -c "
   from huggingface_hub import snapshot_download
   print(snapshot_download('Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed', local_files_only=True))
   "
   ```
3. **比較ハーネス** `bench/compare_engines.py`（新規）。
4. **KLD 計測下ごしらえ** `bench/kld_probe.py`（新規、実行はしていない）。
5. このファイル。

## 見つけたリポジトリ ID・バージョン情報

- MTPLX Optimized-Speed: `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`
  （mtplx clone の `TROUBLESHOOTING.md`・`tests/test_tail_profile_truth.py` 等
  複数箇所で一致。HF API `/api/models/...` でも実在・非 gated を確認済み）
- 同 FP16 版: `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16`（M1/M2 用。今回は未使用）
- PyPI `mtplx` 最新版 = 2.9.2 = clone の `pyproject.toml` と同一

## 注意: docs/research/KERNEL-INTEL.md に「KLD 計測レーン」という見出しは無かった

依頼文にあった「docs/research/KERNEL-INTEL.md の『KLD 計測レーン』の注意（token_ids を
同一性レーンに使う等）」を探したが、2026-08-26 時点の同ファイルにその見出しは
存在しない。最も近い記述は docs/research/PLAN.md Phase C1:

> 品質ゲートは KLD を主指標にする: bf16 参照に対する出力分布の KL divergence を
> 固定評価セットで測り、閾値超えは速度がどうであれ不合格。greedy 一致率と logit
> 差は補助指標にする

`bench/kld_probe.py` はこの記述と teacher forcing の一般原則から実装した:
bf16 参照モデル自身の greedy 継続を「正解の token_ids」として固定し、bf16・
量子化モデルの両方に**同じ** token_ids を teacher force する（各モデルが自分の
argmax で分岐すると、KLD が分布差ではなく文脈差を測ってしまうため）。この設計が
依頼者の想定と違う場合は要修正。

## GPU キューの衝突に注意

`docs/research/STATUS.md` の「GPU gate queue」に Phase A2/A3 の未実行ゲート
（例: `bench/test_qmm_skinny_mma.py --dtype bfloat16 ...`）が並んでいる。
GPU は同時 1 プロセストラフィックが前提（docs/research/PLAN.md 契約 5）。
下記の比較コマンドを実行する前に、STATUS.md の GPU gate queue を先に片付けるか、
少なくとも同時に走らせないこと。

## 私（ht）が実行する正確なコマンド列

### 0. 予備戦（GPU を使うが軽い。動作確認用）

まず `--dry-run` でコマンド列だけを確認する（GPU 不使用）:

```bash
cd ~/dev/fastmlx
uv run python bench/compare_engines.py --dry-run \
  --gpu-note "dry-run: GPU未使用" --prompts code
```

`mtplx inspect` は GPU を使わない（metadata のみ）ので先に単独で流してよい:

```bash
tools/compare/mtplx-venv/bin/mtplx inspect lmstudio-community/Qwen3.8-27B-MLX-4bit --json
tools/compare/mtplx-venv/bin/mtplx inspect Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed --json
```

モデル DL が終わっていること、GPU が空いていることを確認したら、
1 プロンプトだけで各エンジンが動くか予備戦する（cooldown 込み、~10-20分想定):

```bash
cd ~/dev/fastmlx
uv run python bench/compare_engines.py \
  --modes same-quant \
  --engines mlx-lm fastmlx mtplx \
  --prompts code \
  --max-tokens 64 \
  --cooldown-sec 60 \
  --gpu-note "予備戦: 直前は他プロセスなしを確認済み" \
  --output bench/results/compare-engines-smoke.json
```

`bench/results/compare-engines-smoke.json` を見て、3 エンジンとも
`decode_tok_s` / `ttft_s` が埋まっているか、`error` が無いかを確認すること。
MTPLX の same-quant 行はモデル未対応で失敗する可能性がある
（lmstudio 4bit に MTPLX が認識する MTP ヘッドが無いための `--no-mtp` 固定。
`mtplx inspect` の事前結果で `AR-only` 系分類が出ていれば正常）。

### 1. 正式実行（静音プロトコル後）

docs/research/PLAN.md Phase E1「正式ベンチプロトコル」（再起動直後・Spotlight 静止・
電源接続・他プロセス最小）はまだ `bench/PROTOCOL.md` として明文化されていない。
それが無い間は最低限、以下を実行前に確認すること:

```bash
pmset -g batt          # AC Power になっているか
mdutil -s /             # Indexing enabled/disabled。実行中なら手動 index が走り得る
ps aux | grep -i mtplx  # 前回の mtplx デーモンが残っていないか
```

本番実行（全モード・全プロンプト・512トークン。所要時間はロードの重い
MTPLX Optimized-Speed 込みで長め。エンジン間 60 秒冷却込み）:

```bash
cd ~/dev/fastmlx
uv run python bench/compare_engines.py \
  --modes same-quant recommended \
  --engines mlx-lm fastmlx mtplx \
  --prompts all \
  --max-tokens 512 \
  --cooldown-sec 60 \
  --depth 3 \
  --gpu-note "静音プロトコル: 再起動直後 / AC電源 / Spotlight確認済み / 他プロセスなし" \
  --output bench/results/compare-engines-$(date -u +%Y%m%dT%H%M%SZ).json
```

長時間ベンチ後は帯域が半減しうる（README の計測規律）。連続実行せず、
必要なら `pmset -g thermlog` 等でスロットリングの有無も併記すること。

### 2. KLD（実行しないこと。参考のみ）

`bench/kld_probe.py` は bf16 の Qwen/Qwen3.8-27B（既にローカルキャッシュ済みを
確認済み）を丸ごとロードする重い処理。ローカルキャッシュの存在だけ確認したい
場合は `--dry-run`（GPU 不使用、トークナイザのみ）:

```bash
cd ~/dev/fastmlx
uv run python bench/kld_probe.py \
  --quant-model lmstudio-community/Qwen3.8-27B-MLX-4bit --dry-run
```

実行する場合（GPU 作業。上記の比較ベンチとは同時に走らせないこと）:

```bash
cd ~/dev/fastmlx
uv run python bench/kld_probe.py \
  --ref-model Qwen/Qwen3.8-27B \
  --quant-model lmstudio-community/Qwen3.8-27B-MLX-4bit \
  --prompts all --gen-tokens 128 \
  --output bench/results/kld-lmstudio-4bit-$(date -u +%Y%m%dT%H%M%SZ).json

# Optimized-Speed の DL が終わっていれば同じ継続キャッシュを使い回して比較できる
uv run python bench/kld_probe.py \
  --ref-model Qwen/Qwen3.8-27B \
  --quant-model Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed \
  --prompts all --gen-tokens 128 \
  --output bench/results/kld-mtplx-optimized-speed-$(date -u +%Y%m%dT%H%M%SZ).json
```

2 回目の実行は `bench/results/kld-continuations.json` にキャッシュされた
bf16 継続トークン列を再利用するので、bf16 側の逐次貪欲デコードは 1 回で済む
（bf16 モデルのロードと teacher-forced 順伝播は毎回必要）。
KERNEL-INTEL.md の Phase C 初期レシピが引用している MTPLX Optimized-Speed の
実測 KLD 0.022 は比較の目安になる。

## 既知の注意点（実測前に把握しておくこと）

- TTFT として JSON に入れている値は、3 エンジンとも実体は prefill（プロンプト
  処理）にかかった時間で揃えた。mlx-lm は `prompt_tokens/prompt_tps` の商、
  fastmlx は CLI がそのまま出す `ttft_s`、MTPLX は JSON の `prompt_eval_time_s`
  を使っている。ただしこの値にプロセス起動からモデルロード完了までの時間は
  入らない。subprocess で毎回コールドスタートするぶん `wall_time_s`（壁時計
  合計）のほうが長く出るので、TTFT と壁時計合計は別の数字として読むこと。
- fastmlx 側の生成トークン数は実測ではなく概算になる。`fastmlx/cli.py`
  （既存ファイルなので今回は変更していない）が生成トークン総数を出力しない
  ため、`wall_time_s - load_s - ttft_s` を `decode_tok_s` で割り戻した値を
  `generated_tokens_estimate` として JSON に入れた。正確な数を取りたければ
  fastmlx/cli.py に `--json` 出力を足す作業が別途要る。
- same-quant 行の MTPLX は投機デコードを比較していない。lmstudio 4bit には
  MTPLX が認識する MTP ヘッドが同梱されていない（mlx-lm の sanitize がロード
  時に `mtp.*` を捨てる件、README/KERNEL-INTEL.md に記載のとおり）ため、この
  行の MTPLX は `--no-mtp` 固定で走らせている。ここで比べているのは MTPLX の
  AR カーネル実装であって、MTPLX の看板機能である投機デコードではない。投機
  込みの MTPLX を見るなら recommended 行（Optimized-Speed, `--depth 3 --mtp`）
  を見る。
- **mlx-lm の `stream_generate` は既知の非決定 quirk がある**
  （`docs/research/STATUS.md` Phase B2 実測）: 手動 greedy ループとは位置 49 あたりで
  準同点 argmax が入れ替わることがあり、原因は `stream_generate` 側の
  専用 stream/wired_limit/非同期パイプラインで fastmlx 側の不具合ではないと
  切り分け済み。速度比較には影響しないが、出力テキストが 1 トークン単位で
  ずれても「バグ」ではなくこの既知 quirk の可能性が高い。
- **量子化モデルは lmstudio-community/Qwen3.8-27B-MLX-4bit を既定にした**
  （fastmlx/cli.py の既定と同じ）。同一 checkpoint を 3 エンジンに揃えるための
  選択で、この ID は既にローカル HF キャッシュに存在することを確認済み
  （`~/.cache/huggingface/hub/models--lmstudio-community--Qwen3.8-27B-MLX-4bit`）。
- 温度は 3 エンジンとも `--temp 0.0` の greedy に揃えてある。決定的な出力で
  比較したいのでこれを既定にしたが、MTPLX の厳密棄却サンプリングの主張
  （temp>0 でも分布厳密同一）を確かめたいなら `--temp 0.6`（MTPLX README の
  推奨値）に変えて流す。

## ライセンス

MTPLX は Apache-2.0 + NOTICE。`tools/compare/mtplx-venv/` は pip 経由のインストール
であり、fastmlx リポジトリへのコード取り込みではない。fastmlx 側で MTPLX を
「使った」と主張・公開する場合は NOTICE の attribution 要求
（"Powered by MTPLX" の表示）を満たす必要がある — これは配布物側の話で、
このリポジトリのコードには影響しない。

---

# 27B の 5 者の起動コマンドと状態 (2026-09-04)

対象は **同じ重み**: `~/models/qwen38-27b-4bit` (mlx-community の Qwen3.8-27B 4bit、
dense・64 層・hidden 5120・head_dim 256・vocab 248320) と MTP サイドカー
`~/models/qwen38-27b-mtp` (同 MTP-4bit、238 MB)。計測は `bench/bench_http_engine.py`
(新規) で 5 者とも同じ経路 (OpenAI 互換 SSE、`bench/vs_mlx_serve.py` の `stream_once`)。

## ハーネスの使い方

```bash
BIGLOCK_NO_WORKER=1 tools/biglock.sh .venv/bin/python bench/bench_http_engine.py \
    --engine <名前> --port <N> \
    --tokenizer ~/models/qwen38-27b-4bit \
    --argv '<サーバーの起動コマンド 1 行>' \
    --thinking-how {reasoning_effort,chat_template_kwargs,prompt,none} \
    --ctxs 0,4000 --tokens 64 --reps 1 \
    --server-log scratchpad/log-<名前>.txt \
    --out bench/results/smoke-27b-<名前>-0904.json
```

行の形は `bench/self_snapshot.py` と同じ (`ctx` / `cold_ttft` / `warm_ttft` /
`decode_tps` / `n_tokens` / `colds` / `warms` / `decs` / `ntoks` / `prompts`)。
`--model-id` は `/v1/models` の名乗りを使わずに送る id を固定するためのもの
(mlx-lm と oMLX で要る、下記)。

**`n_tokens` はチャンク数ではなく本文をトークナイザで数え直した値。**
oMLX は 1 チャンクに 4 トークンほど詰めて流すので、チャンクで割ると decode が
1/4 に見える (64 トークンが 16 チャンク)。チャンク基準の値も
`decode_tps_chunks` / `n_chunks` に残してあるので、1 チャンク = 1 トークンの
エンジンでは両者の一致を突き合わせに使える。

## 5 者の起動コマンドと状態

| エンジン | 起動 | 投機 (MTP) | 証拠 |
|---|---|---|---|
| mlxturbo | OK | **効く** | `投機デコード有効 (MTP: あり / lookup: 有効)`、`tok/step=2.33〜3.50` |
| mlx-lm 0.31.3 | OK | 無し | `qwen3_5_mtp` を読めない (`--draft-model` は別モデル用) |
| oMLX 0.6.4 | OK | **効く** | `VLM MTP enabled for qwen38-27b-4bit, drafter=qwen38-27b-mtp`、`tokens_per_round=1.75` |
| MTPLX 2.9.2 | OK | 無し (AR) | quickstart が自分で `--depth 0 --generation-mode ar --no-load-mtp` を選ぶ |
| mlx-serve 26.9.1-dev | OK | 無し (PLD のみ) | `[mtp] no head found: ... — MTP off` |

```bash
# 1) mlxturbo  (port 8151)
.venv/bin/python -m mlxturbo.server --model ~/models/qwen38-27b-4bit \
    --mtp ~/models/qwen38-27b-mtp --host 127.0.0.1 --port 8151
# thinking: --thinking-how reasoning_effort ("none" が enable_thinking=False に落ちる)

# 2) mlx-lm  (port 8152)
.venv/bin/mlx_lm.server --model ~/models/qwen38-27b-4bit --host 127.0.0.1 --port 8152
# thinking: --thinking-how chat_template_kwargs (server.py:547 が per-request で読む)
# **--model-id ~/models/qwen38-27b-4bit が要る。** mlx_lm.server の /v1/models は
# HF キャッシュのモデルを全部並べるので、名乗りの先頭を拾うと別モデル (gemma-4-12B) を
# 取りに行って 404 になる。HF_HUB_OFFLINE=1 も付けること。

# 3) oMLX  (port 8153)
omlx serve --model-dir ~/models/omlx-27b --host 127.0.0.1 --port 8153 \
    --no-hf-cache --log-level info
# --model-dir は 27B と mtp だけを symlink した専用ディレクトリ:
#   ~/models/omlx-27b/qwen38-27b-4bit -> ~/models/qwen38-27b-4bit
#   ~/models/omlx-27b/qwen38-27b-mtp  -> ~/models/qwen38-27b-mtp
# --no-hf-cache が無いと HF キャッシュの他モデルも並ぶ。--model-id qwen38-27b-4bit。
# thinking: --thinking-how chat_template_kwargs

# 4) MTPLX  (port 8154) -- quickstart --dry-run --json が吐く server_command をそのまま使う
tools/compare/mtplx-venv/bin/python -m mtplx.server.openai \
    --model ~/models/qwen38-27b-4bit --backend-id qwen3_next --host 127.0.0.1 --port 8154 \
    --depth 0 --generation-mode ar --profile sustained ... --no-load-mtp ...
# thinking: --thinking-how chat_template_kwargs (openai.py:24563 が読む)

# 5) mlx-serve  (port 8155)
~/dev/mlx-serve/zig-out/bin/mlx-serve --serve --model ~/models/mlxserve-27b \
    --host 127.0.0.1 --port 8155 --mtp --log-level debug
# --model は 27B の各ファイルを symlink し、mtp.safetensors を足した専用ディレクトリ:
#   ~/models/mlxserve-27b/* -> ~/models/qwen38-27b-4bit/*
#   ~/models/mlxserve-27b/mtp.safetensors -> ~/models/qwen38-27b-mtp/model.safetensors
# thinking: --thinking-how reasoning_effort
```

## MTP のペア付け

- **oMLX**: CLI ではなく `~/.omlx/model_settings.json` (無ければ新規作成)。
  **サーバー起動前に書くこと** — 設定は起動時にメモリへ読み込まれる。
  ```json
  {"version": 1, "models": {"qwen38-27b-4bit": {
      "vlm_mtp_enabled": true, "vlm_mtp_draft_model": "qwen38-27b-mtp"}}}
  ```
  admin の `PUT /admin/api/models/{id}/settings` でも同じ場所に書けるが、
  セッション cookie 認証が要る (`omlx/admin/auth.py:257`)。
  ドラフタは `qwen3_5_mtp` として自動判別され (`mlx_vlm.speculative.drafters`)、
  失敗しても警告だけで本体は起動する (fail-soft) ので、**ログを見ないと
  効いていないことに気付けない。**
- **mlx-serve**: **今の構成では MTP が付かない。**サイドカーのキーが接頭辞なし
  (`fc.weight` / `layers.0.*` / `pre_fc_norm_*`) で、mlx-serve の marker gate
  (`src/mtp.zig:926`) が探すのは `mtp.fc.weight` / `language_model.mtp.fc.weight` /
  `mtp.eh_proj.weight`。ログは
  `skipping mtp.safetensors: not named by model.safetensors.index.json` と
  `[mtp] no head found: no mtp/ sidecar and no [language_model.]mtp.* keys resolvable
  from the index — MTP off`。
  量子化サイドカー自体は受け付ける設計 (`loadLinear` が `<prefix>.{weight,scales,biases}`
  を読み、bits/group_size を scales の形から導く。`src/mtp.zig:395-410`)。
  **足りないのはキーの接頭辞だけ**なので、`mtp.` を付けた写しを作れば通る見込み。
  作るかどうか (同じ重みの比較として妥当か) は未決。
  推奨パック `ddalcu/Qwen3.8-27B-MLX-Serve-4bit` (18.2 GB、draft head baked in) は
  ローカルに無く、ダウンロードしていない。
- **MTPLX**: `mtplx inspect` が `runtime_compatibility: native-ar-only-missing-mtp`
  (base 単体) / `invalid-mtp-tensor-layout` (mtp.safetensors を置いた構成) を返す。
  MTPLX も `mtp.` 接頭辞のキーを期待する (`mtplx/artifacts.py:71` の
  `MTP_KEY_PREFIXES`)。quickstart は自動で AR (`--depth 0 --no-load-mtp`) に落ちるので、
  **起動はするが投機は効かない。**別配布の `Youssofal/...MTPLX-Optimized-Speed` は
  重みが違うので「同じ重みの比較」には使えない。
- **mlx-lm**: `qwen3_5_mtp` を読めない。投機無しが正しい姿。

## 煙試験 (2026-09-04 08:16-08:19、`--ctxs 0,4000 --tokens 64 --reps 1`、冷却無し)

| エンジン | 冷 TTFT (ctx0) | decode (ctx0) | 冷 TTFT (4k) | 温 TTFT (4k) | decode (4k) |
|---|---|---|---|---|---|
| mlxturbo | 0.24s | 28.1 tok/s | 17.52s | **0.27s** | 27.3 tok/s |
| mlx-lm | 0.43s | 21.8 tok/s | 18.92s | **0.44s** | 20.9 tok/s |
| oMLX | 0.45s | **33.7 tok/s** | 18.98s | 20.24s | **32.8 tok/s** |
| MTPLX | 0.29s | 21.4 tok/s | 18.54s | 21.40s | 20.6 tok/s |
| mlx-serve | 0.32s | 24.3 tok/s | **15.88s** | **0.74s** | 25.3 tok/s |

**これは煙試験であって比較ではない** (1 本ずつ、冷却なし、`--warm-long` 無し、
プロセス起動直後の段差も消していない)。判定に使わないこと。
結果 JSON: `bench/results/smoke-27b-<engine>-0904.json`。

本番計測の前に決めること:

- **oMLX と MTPLX は 4k の温 TTFT が冷とほぼ同じ** (20.24s / 21.40s) = 接頭辞を
  再利用していない。oMLX は `--paged-ssd-cache-dir` を渡さないと接頭辞キャッシュが
  有効にならない。mlxturbo・mlx-lm・mlx-serve は既定で効いている。
  **揃えないと温 TTFT の比較は「機能の有無」を測ることになる。**
- MTPLX の quickstart 既定は `--ssd-session-cache on --ssd-session-cache-max-size 100GB`。
  ディスクに書く。揃えるかどうか。
- 4000 の冷 TTFT が 5 者とも 15.9〜19.0s (prefill 200〜240 tok/s) に固まっている。
  27B の prefill でここまで揃うのは、どのエンジンでも同じ律速に当たっている可能性がある。
  本番では `--warm-long` を入れて重みのページインを分離すること。
