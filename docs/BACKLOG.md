# BACKLOG — やりたいが未着手のもの (2026-08-29 更新)

ここには「やる価値はあるが手を付けていない」ものを、着手前に分かっている根拠つきで置く。決着したものは末尾の「済み」へ移した。

## 1. マルチモーダル対応 (画像 → 音声・動画)

元 checkpoint は VLM (`Qwen4ExpForConditionalGeneration`)。1,658 キー中 **333 が vision 系** (`model.visual.blocks.*`) だが、変換で意図的に捨てている。変換後の v 系成果物に vision 系は **0 キー**。元 checkpoint (360GB / 131 シャード) は外付け SSD 配下の `models/Qwen3.8-Flash-Next` に再取得済みなので、素材はいつでも読める。

落としている箇所:
- `mlxturbo/convert_flash.py:331` — `mtp.` / `vision_tower.` / `model.visual.` を skip
- `tools/vendor/qwen4_exp.py:849` — mlx-lm 側の `sanitize` でも同じものを skip (`# text-only pour l'instant`)

必要な作業は 3 段。

1. 変換で `model.visual.*` を残す (`convert_flash.py:331`)
2. **mlx-lm 側に vision タワーのクラスを書く。** 現状 `qwen4_exp.py` に vision/visual の言及は skip の 2 行だけで、クラス自体が存在しない。**ここが一番重い**
3. エンジンが埋め込み入力を受け取れるようにする。`mlxturbo/spec.py:164` が `self.inner.embed_tokens(tokens[None])` と自前で埋め込んでいるのを、外から渡せるようにする。モデルの `__call__` は既に `input_embeddings` を受ける口を持っている (`qwen4_exp.py:746`)。`spec_flash.py` 側も同様の口が要る

投機デコード固有の注意点として、**n-gram lookup はトークン ID 上で動くので、画像プレースホルダ (同じ ID が数百個並ぶ) の区間で誤マッチを量産する。** lookup の対象から画像区間を外す処理が要る。MTP 側はドラフトが常にテキストなので影響しない。

音声・動画は扱う場所自体が無い。ViT のぶん重みも増えるので、91GB がさらに膨らむ点も込みで判断すること。

## 2. サーバーの並列化 (継続バッチング)

**測定は済んでいて、実装も opt-in の形で存在する。配線だけしていない。**

- 取り分の実測: gemma4-26B で B=4 実測 2.10x。Flash-Next はエキスパート和集合からの予測で 2.05〜2.19x (`bench/results/batch-flashnext-expert-union.json`)。「512 エキスパートなら飽和しにくい」は誤りで、飽和は (B × top_k) / num_experts で決まる
- prefill には一切効かない (B に線形)。cold 32k は TTFT 112 秒で壁時計の 84% を prefill が占める (`bench/results/batch-flashnext-prefill-decode.json`)。**体感を狙うなら prefill 側が先**
- 実装: `mlxturbo/batch.py` に `enable_batch_cache()` (既定 off、未配線)。合成モデルの CPU 検証で QSA off と長さ揃いは B=1/2/4 完全一致。**残る制限**: QSA 有効 (2048 超) かつ長さ不揃いでバッチ出力が単体と一致しない (破損ではなく QSA のブロックグリッドが絶対列で切られるため。パディングを揃えれば 3.7e-8)。実モデル検証 `tools/verify_batch_real.py` は未実行
- やる根拠は速度でなく**可用性**: 直列サーバーだとサブエージェントを流しっぱなしにしたまま別の作業ができない。B=4 でレイテンシは 0.53x に悪化するので、対話 1 本の用途では逆効果

継続バッチングを投機デコードと GDN の再帰状態の上に載せるのは片手間の規模ではない。着手するなら `verify_batch_real.py` の実行が最初の一歩。

## 3. 他モデル対応で投機を効かせる (一部済み)

契約に合わないモデルでも「載せれば喋る」状態は元からある (fallback、理由は `/health` の
`fallback_reason` に出る)。2026-08-30 に、投機を効かせる汎用経路を 2 つ足した。どちらも
既定 off。

`--draft-model` (`DraftSpecRunner`) は mlx_lm 自身の `speculative_generate_step` を包む
だけなので、アーキテクチャに依存しない。ただし**適合する小型ドラフトが要る**。手元に
無く、self-draft では 367 → 248 tok/s と当然ながら悪化した。真の高速化ケースは未測定で、
ここが残っている最大の穴。

`--lookup-spec` (`LookupSpecRunner`) は n-gram lookup だけを使う。trim 可能なキャッシュ
かつ貪欲限定。繰り返しの多い入力で 361 → 427 tok/s (+18%) だが、**繰り返しの無い自然文
では 367 → 245 tok/s (-32%) と遅くなる**。自前ループが mlx_lm の `generate_step` が持つ
`async_eval` 二重バッファを持たないため。ここを埋めれば常時 on にできる可能性がある。

残るのは、`spec` / `flash_spec` 級の速度 (1.25-1.39x) を他アーキテクチャで出すこと。
それには MTP 相当のドラフトヘッドか、アーキテクチャ固有の状態捕獲が要る。

## 4. `n` (複数候補)

直列のままだと生成時間が候補数倍になるので §2 のバッチ化が前提。同一プロンプトの n 候補は
§2 のバッチにそのまま載る。

`logprobs` は 2026-08-30 に fallback / 降格経路限定で実装した。投機経路で要求されたら
自動で非投機に降格する。ストリーミングとの併用は 400 で断っている — ThinkingRouter や
tool_calls が絡むとトークン列と content の対応が崩れ、正しく対応付ける実装が要るため。
そこを埋めるのは残件。

## 5. prefill の高速化 (体感の本丸)

prefill は 290〜325 tok/s しか出ない (デコードの約 11 倍でしかない)。Claude Code は毎ターン 54k 超を送るので、初回 TTFT を支配する。prompt cache の再利用 (fadadf7) で 2 ターン目以降は 9 割超を飛ばせるようになったが、**初回とキャッシュミス時はまるごと食らう**。バッチ化では縮まない (B に線形)。カーネル側の課題として未着手。

## 6. LoRA アダプタ対応 (`adapter_path`)

**mlx_lm 側の受け口は既に揃っている。fastmlx/mlxturbo 側は未配線**。以下は調査のみで、実装はしていない (`runner.py` / `server.py` が別レーンで編集中のため)。

mlx_lm の受け口:

`mlx_lm/utils.py` の `load(path_or_hf_repo, tokenizer_config=None, model_config=None, adapter_path=None, lazy=False, return_config=False, revision=None)` (453-502行) は `adapter_path` をキーワード引数として直接受け取り、非 None なら 492-494行で `model = load_adapters(model, adapter_path); model.eval()` を呼ぶ。`load_adapters` (utils.py 423-426行) は `mlx_lm/tuner/utils.py` の実体 (113-138行) へのラッパーで、`adapter_path/adapter_config.json` を読んで `fine_tune_type` (既定 `"lora"`) を見つつ `linear_to_lora_layers()` (同ファイル 38-110行) でモデルの `Linear`/`Embedding`/`SwitchLinear` を `LoRALinear` 系へ差し替え、その後 `model.load_weights(adapter_path/adapters.safetensors, strict=False)` で重みを流し込む。ディレクトリ構成は `adapter_config.json` + `adapters.safetensors` の組で固定。`mlx_lm/generate.py` の CLI は `--adapter-path` フラグ (81-83行) からそのまま `load(model_path, adapter_path=args.adapter_path, ...)` (2008-2012行) へ渡すだけ。

fastmlx/mlxturbo 側の現状:

`mlxturbo/_mlx_compat.py:90-93` の契約チェック (`_require_signature`) が `mlx_lm_load` のシグネチャに `adapter_path` が含まれることを検査しているだけで、実際に値を渡している呼び出しはリポジトリ内のどこにも無い。`mlxturbo/cli.py:122`、`mlxturbo/server.py:101,4792` の `mlx_lm_load(args.model, return_config=True)` はいずれも `adapter_path` 未指定。`grep -rn -i "adapter\|lora" mlxturbo/*.py` でヒットするのはこの契約チェックの1箇所と、無関係な英語コメント2箇所 (`fast_qmm.py:377` の「adapters」は量子化レイヤー再利用の話、`server.py:1962` の「adapter」は一般名詞) のみ。`runner.py` (`build_runner` 以下) にも adapter_path/LoRA を受け取る引数・分岐は無い。

配線先と見積もり:

`--adapter-path` フラグ自体は `cli.py`/`server.py` どちらの argparse にも追加できる (前者は編集可能、後者は今回のレーン境界で触れない)。値を実際に効かせるには、`mlx_lm_load(args.model, adapter_path=..., return_config=True)` という形で呼び出しに1引数足すだけで済む — この呼び出しは `build_runner` より前、まだモデルをロードしている段階 (`cli.py:122` / `server.py:101,4792`) にあるので、`build_runner` 自体のシグネチャを変える必要は無い。ただし `server.py` 側は今回のレーン境界外なので着手できない。`cli.py` 側だけなら技術的には可能だが、対話 CLI だけ先に対応してもサーバー経由の運用 (本来の主用途) には効かないため、今回は見送って調査止まりとした。

分岐点として、**実行時に `adapter_path` を渡すだけ (LoRALinear のまま使う) か、事前に `mlx_lm` 付属の `fuse.py` でオフライン fuse 済みのモデルディレクトリを用意してそれを読ませるか**で影響範囲が変わる。前者は `SpecEngine`/`FlashSpecEngine` が構築される前にモデルの `named_modules` 構造が `<parent>.linear.weight` 型に変わる (`LoRALinear.from_base()` が元の `Linear` を `self.linear` として内包する) ため、`_mlx_compat.py` の契約チェックや `spec_flash.py` の MTP サイドカーのテンソル名突合に影響しないか未検証。後者 (`mlx_lm.tuner.fuse`) は `LoRALinear.fuse()` でマージ後の平のテンソルに戻してから保存するため (`fuse.py:60-75`)、テンソル名・形状は元のモデルと同じに戻り、既存の契約チェック・MTP 探索への影響はほぼ無いはず。着手するなら後者 (オフライン fuse → 通常のモデルとして読み込む) の方が安全側で、見積もりも軽い (`cli.py`/`server.py` へのフラグ追加 + 呼び出し1箇所の引数追加のみ)。前者 (実行時 LoRA、fuse 無し) は `_mlx_compat.py` の契約チェックと `spec_flash.py` 側の名前突合を実地で確認する一手間が追加で要る。

---

## 済み (BACKLOG から出たもの)

- **tool calling** → 73e061a で実装。opencode / Codex / Claude Code の 3 クライアントで実機検証済み。懸念だった「モデルが構文を安定して出すか」は Qwen3.6 / Flash-Next とも問題なし
- **MTP の価値の決着** → 決着した。27B の数字から引いた「現 base では 1.2 倍遅くなる」は Flash-Next には当てはまらず、専用エンジン (`spec_flash.py`、深さ 1) で **1.26〜1.44 倍**。MTP 重みは元 checkpoint から抽出したサイドカー (5.2GB、4 段の量子化成果物で共用)。経緯と実測は `docs/MTP-FLASH.md`
- **公開の前提** → `fastmlx` から `mlxturbo` へ改名 (PyPI/GitHub の `fastmlx` は競合の既存プロジェクト)、MIT ライセンス、README の書き直し (「汎用高速推論ランタイム」とは名乗らない)、docs の整理、CI、運用手引き、個人パスの除去、コメントの英語化。手順は `docs/RELEASE.md`
- **プロトコル層の穴** → 非恒等サンプリングの 400 をリクエスト単位の非投機降格に変更 (実クライアントは `top_p` を既定で送るので、看板構成が最初の 1 発で 400 を返していた)、`response_format` の 400 明示、stop の早期打ち切り (ストリームで 11 倍)、Responses の `store`/`previous_response_id`、Anthropic の usage キャッシュ量、`/v1/embeddings` の 501 明示
- **サーバーの配布準備** → 認証・キュー上限・SSE keepalive・graceful shutdown・文脈長ガード・prompt cache 再利用・MTP 自動発見・`--require-runner`。接続手順は `docs/SERVER.md`
