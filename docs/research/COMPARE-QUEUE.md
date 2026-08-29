# 比較実行キュー（Phase E2 準備）

docs/PLAN.md Phase E2「mlx-lm 素 vs fastmlx vs MTPLX の同一マシン・同一プロンプト比較」
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
   tail -20 /Users/ht/dev/fastmlx/tools/compare/hf-download-optimized-speed.log
   ps aux | grep "hf download" | grep -v grep
   du -sh ~/.cache/huggingface/hub/models--Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed 2>/dev/null
   ```
   完了後の確認（ローカル完結、ネットワークに出ない）:
   ```bash
   uv run --project /Users/ht/dev/fastmlx python -c "
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

## 注意: docs/KERNEL-INTEL.md に「KLD 計測レーン」という見出しは無かった

依頼文にあった「docs/KERNEL-INTEL.md の『KLD 計測レーン』の注意（token_ids を
同一性レーンに使う等）」を探したが、2026-08-26 時点の同ファイルにその見出しは
存在しない。最も近い記述は docs/PLAN.md Phase C1:

> 品質ゲートは KLD を主指標にする: bf16 参照に対する出力分布の KL divergence を
> 固定評価セットで測り、閾値超えは速度がどうであれ不合格。greedy 一致率と logit
> 差は補助指標にする

`bench/kld_probe.py` はこの記述と teacher forcing の一般原則から実装した:
bf16 参照モデル自身の greedy 継続を「正解の token_ids」として固定し、bf16・
量子化モデルの両方に**同じ** token_ids を teacher force する（各モデルが自分の
argmax で分岐すると、KLD が分布差ではなく文脈差を測ってしまうため）。この設計が
依頼者の想定と違う場合は要修正。

## GPU キューの衝突に注意

`docs/STATUS.md` の「GPU gate queue」に Phase A2/A3 の未実行ゲート
（例: `bench/test_qmm_skinny_mma.py --dtype bfloat16 ...`）が並んでいる。
GPU は同時 1 プロセストラフィックが前提（docs/PLAN.md 契約 5）。
下記の比較コマンドを実行する前に、STATUS.md の GPU gate queue を先に片付けるか、
少なくとも同時に走らせないこと。

## 私（ht）が実行する正確なコマンド列

### 0. 予備戦（GPU を使うが軽い。動作確認用）

まず `--dry-run` でコマンド列だけを確認する（GPU 不使用）:

```bash
cd /Users/ht/dev/fastmlx
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
cd /Users/ht/dev/fastmlx
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

docs/PLAN.md Phase E1「正式ベンチプロトコル」（再起動直後・Spotlight 静止・
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
cd /Users/ht/dev/fastmlx
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
cd /Users/ht/dev/fastmlx
uv run python bench/kld_probe.py \
  --quant-model lmstudio-community/Qwen3.8-27B-MLX-4bit --dry-run
```

実行する場合（GPU 作業。上記の比較ベンチとは同時に走らせないこと）:

```bash
cd /Users/ht/dev/fastmlx
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
  （`docs/STATUS.md` Phase B2 実測）: 手動 greedy ループとは位置 49 あたりで
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
