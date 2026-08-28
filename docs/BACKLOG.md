# BACKLOG — やりたいが未着手のもの (2026-08-28)

速度の決着は [[docs/KERNEL-BRIEF-MOE-GDN.md]] と直近のコミットで一区切り。ここには「やる価値はあるが手を付けていない」ものを、着手前に分かっている根拠つきで置く。

## 1. マルチモーダル対応 (画像 → 音声・動画)

元 checkpoint は VLM (`Qwen4ExpForConditionalGeneration`)。1,658 キー中 **333 が vision 系** (`model.visual.blocks.*`) だが、変換で意図的に捨てている。変換後の v-fast6 に vision 系は **0 キー**。

落としている箇所:
- `fastmlx/convert_flash.py:331` — `mtp.` / `vision_tower.` / `model.visual.` を skip
- `tools/vendor/qwen4_exp.py:849` — mlx-lm 側の `sanitize` でも同じものを skip (`# text-only pour l'instant`)

必要な作業は 3 段。

1. 変換で `model.visual.*` を残す (`convert_flash.py:331`)
2. **mlx-lm 側に vision タワーのクラスを書く。** 現状 `qwen4_exp.py` に vision/visual の言及は skip の 2 行だけで、クラス自体が存在しない。**ここが一番重い**
3. `SpecEngine` が埋め込み入力を受け取れるようにする。`fastmlx/spec.py:164` が `self.inner.embed_tokens(tokens[None])` と自前で埋め込んでいるのを、外から渡せるようにする。モデルの `__call__` は既に `input_embeddings` を受ける口を持っている (`qwen4_exp.py:746`) ので、そこは流用できる

投機デコード固有の注意点として、**n-gram lookup はトークン ID 上で動くので、画像プレースホルダ (同じ ID が数百個並ぶ) の区間で誤マッチを量産する。** lookup の対象から画像区間を外す処理が要る。MTP 側はドラフトが常にテキストなので影響しない。

音声・動画は扱う場所自体が無い。ViT のぶん重みも増えるので、91GB がさらに膨らむ点も込みで判断すること。

## 2. MTP の価値を決着させる

現状の 30.61 tok/s は **MTP 無し**。そして今の構成では、有効にしても遅くなる公算が高い。

`bench/results/mtp-cost.json` の実測から損益分岐を引くと、verify は m1 が 51.85ms、1 トークン増えるごとに約 +15.2ms、draft 1 トークンが 4.72ms。

| | 旧 base (51.85ms) | 現 base (約 32.7ms) |
|---|---|---|
| n_draft=3 の分岐点 | 平均受理 1.17 | 平均受理 **約 2.8** |
| n_draft=1 の分岐点 | 0.42 | 約 0.61 |

実測受理は `bench/results/mtp-diag-d0.json` で n_draft=3 のとき平均 1.38 (深度別の連鎖受理 0.69 / 0.43 / 0.26)。旧 base では得だったが、**現 base では 2.8 に届かず概算 1.2 倍遅くなる**。n_draft=1 なら深度1の 0.69 が分岐点 0.61 を上回るので 5% 前後の得は残る。

自分で base を速くしたぶん、投機が回収すべき固定費が減って損益分岐が上がった、という構図。

着手するなら順番は:

1. **Flash-Next の MTP 重みを再取得** (28/131 シャードに `mtp.*` が散っている。現在スナップショットはメタデータのみで実体ゼロ)
2. Flash-Next での深度別受理率を測る (`spec.py` の accept_trace)。上の数字は全て **Qwen3.8-27B** のもので、Flash-Next では未測定
3. verify コストを現構成で測り直す (上の値は最適化前)
4. n_draft=1 に絞って損益を判定

見込みの取り分が 5% 前後なので、数十 GB の再取得に見合うかは微妙。

## 3. サーバーの並列化 (継続バッチング)

直列サーバーは実装済み。並列化の取り分は MoE の算数から 2〜3 倍ある (512 experts に 10 routed なので、B=8 で 1 トークンあたりの読み出しが 4.7GB → 約 2.0GB)。

ただし着手前に確かめること:

- **prefill が許容範囲か。** README の実測で 219 tok/s。1 万トークンの文脈なら最初のトークンまで 45 秒で、バッチ化しても縮まない。ここが辛さの本体なら並列化は的外れ
- **並列リクエストが実際に発生しているか。** 1 人が 1 問ずつ待つ使い方ならバッチ化はレイテンシを悪化させるだけ。効くのはエージェントが並列にサブエージェントを投げるとき
- メモリ。91GB 常駐に KV を 1 本あたり数百 MB (8K 文脈で約 450MB) 積むので、同時 4〜8 本が現実的な上限

継続バッチングを投機デコードと GDN の再帰状態の上に載せるのは片手間の規模ではない。

## 4. 他モデル対応 (gemma / kimi / glm)

サーバーには「`validate_spec_model_contract` が通らなければ通常生成にフォールバック」を入れてあるので、**載せれば喋る**状態にはなる。ただし投機デコードは効かない。

`SpecEngine` は `fastmlx/_mlx_compat.py:111` で GDN ハイブリッド固有の構造 (`fa_idx` / `ssm_idx` / 層ごとの `is_linear` / linear cache の `advance`) を要求するため、他アーキテクチャで投機を効かせるには contract の一般化が要る。lookup (SAM) 側はモデル非依存なので、そちらだけ先に切り出す手はある。

この Mac では VRAM が先に効くので、載せられるサイズが実質的な制約になる。

## 5. サーバーの残タスク (mlx_lm との差分調査より、2026-08-28)

サンプリングパラメータ (top_p/top_k/min_p/repetition_penalty/presence_penalty/frequency_penalty/seed/logit_bias)・`/health`・CORS・`prompt_tokens_details.cached_tokens`・`/v1/completions` は実装済み (`fastmlx/server.py`, `fastmlx/runner.py`)。以下は mlx_lm/server.py にあって fastmlx には無いが、意図的に見送ったもの。

- **`--decode-concurrency` / `--prompt-concurrency` (並列スロット)。** fastmlx サーバーは `asyncio.Lock` + 単一ワーカースレッドで直列化する設計 (server.py 冒頭 docstring)。91GB 級モデルを 128GB 機に載せている前提なので、複数リクエストを同時にバッチングする余地が薄い (BACKLOG §3 参照: prefill の遅さが先に効く可能性が高く、並列化の是非自体がまだ判定できていない)。並列スロットを足すには直列化の設計そのものをやめる必要があり、この差分調査の範囲を超える。
- **`--draft-model` / `--num-draft-tokens` (mlx_lm 側の投機デコード機構)。** fastmlx は自前の投機エンジン (`fastmlx/spec.py` の `SpecEngine`: MTP 連鎖 + n-gram lookup + Block Verification) を持っており、mlx_lm の draft-model 方式とは別の実装。二重に投機機構を持つ意味が無い。
- **`n` (1 リクエストで複数候補を生成)。** 91GB 級モデルを直列で回す構成では、候補数だけ生成コストが単純倍増する。現実的な用途 (best-of-n 選択など) が出てから改めて検討する。
- **`logprobs` / `top_logprobs`。** 実装コストに対して需要が低いと判断。加えて SpecRunner 経路では意味付けが難しい: Block Verification は受理側 (target) の分布からサンプリングするが、実際に出力されたトークンが「ドラフト由来でそのまま受理されたもの」か「棄却後に target から再サンプリングしたもの」かでどの分布の logprob を返すべきかが変わり、素朴に「最終ロジットの softmax」を返すと投機デコードをしていないかのような値になる (受理された draft トークンの真の logprob は verify 時点のロジットから取れるので不可能ではないが、実装・検証コストが見合わない)。

## 6. tool calling (次に着手する候補として最有力)

コーディングエージェント (opencode 等) をこのサーバーに繋ぐには必須の機能。今回は実装していないが、規模の見積もりと方針を書いておく。

### mlx_lm 側の実装のかたち

参考にした箇所 (`.venv/lib/python3.13/site-packages/mlx_lm/server.py`):

- `146-147`: `tool_call_start` / `tool_call_end` をモデルごとの文字列 (例 Qwen なら `<tool_call>` / `</tool_call>`) として `TokenizerWrapper` に渡す。渡された文字列はトークナイズして `_tool_call_start_tokens` / `_tool_call_end_tokens` として保持する (fastmlx の `think_start_tokens` / `think_end_tokens` と全く同じパターン)。
- `537`: `ToolCallFormatter` がモデル固有の tool call 構文をパースする本体。generation_config やモデル名からパーサ (`tool_parser`) を選ぶ。
- `668`: 生成中の状態機械が `normal` / `reasoning` / `tool` の 3 状態を持ち、`tool_call_start` トークン列を検出すると `tool` 状態に遷移する。これは fastmlx の `ThinkingRouter` が `detect` / `thinking` / `content` の 3 状態を持つのと同型 (第 4 の状態 `tool` を足す形になる)。
- `1359-1360`, `1486`, `1511`: 状態が `tool` の間はテキストをそのままクライアントへ流さず貯め、`tool` から抜けたタイミングで `tool_calls` 配列に確定させてから OpenAI 形式の `message.tool_calls` に載せる。
- `1540`: `finish_reason` が `tool_calls` になる分岐 (通常の `stop` の代わり)。

つまりモデル固有の構文解析は「開始/終了マーカー文字列 → トークン列 → 状態機械」という、fastmlx がすでに thinking で持っている仕組みの再利用で足りる。ゼロから構文パーサを書く必要は無い。

### fastmlx のどこに入るか

- `fastmlx/server.py` の `ThinkingRouter` を拡張して 4 状態 (`detect` / `thinking` / `tool` / `content`) にするか、`ToolCallRouter` として並列に持つか。マーカー検出ロジック (バッファ先読み・境界またぎ処理) は thinking と共通化できるので、共通クラスから両方を作る形が筋が良い。
- マーカー文字列の出どころ: mlx_lm は `TokenizerWrapper` に無い (fastmlx 独自に付与が必要)。Qwen3.5/Qwen4 系なら `<tool_call>` / `</tool_call>` 固定でおそらく足りるが、他アーキテクチャ対応時にモデルごとの一覧を持つ必要がある (BACKLOG §4 の「他モデル対応」と絡む)。
- パース結果 (関数名・引数 JSON) を OpenAI 形式 (`message.tool_calls[].function.{name,arguments}`) と Anthropic 形式 (`content` 内の `tool_use` ブロック) の両方に変換する層が要る。現状 `chat_completions`/`anthropic_messages` がそれぞれ独立にレスポンスを組み立てているのと同じ場所に、プロトコルごとの変換を足す。
- リクエスト側: `body["tools"]` (OpenAI) / `body["tools"]` (Anthropic、形は微妙に違う) を読み、chat template の `tools=` 引数へ渡す配線が要る (`_apply_template` の拡張)。

### 分からないこと・要検証

- **SpecRunner 経路との相性。** ドラフト (MTP/lookup) が tool call のマーカートークン列を跨いで投機した場合、Block Verification 自体は受理判定に影響しない (マーカーもただのトークン列なので) はずだが、`ThinkingRouter`/`ToolCallRouter` の状態機械が「複数トークンをまとめて受理した瞬間」に正しく動くかは要確認 (`on_tokens` は投機の受理まとめて複数トークンを一度に渡してくる、既存の thinking 実装がこれに対応済みなのでおそらく同じ形で対応できる)。
- **モデルが実際に tool call 構文を安定して出すか。** Flash-Next / Qwen3.8-27B が tool calling についてどの程度チューニングされているかは未確認。構文が安定しないと状態機械側でどれだけ頑張っても意味が無いので、着手前にモデル側の tool call 出力を手動で数サンプル確認するべき。
- 規模感: thinking 分離の実装 (`ThinkingRouter` 本体 + OpenAI/Anthropic 両プロトコルへの配線) が現状のコード量の目安になる。tool calling は状態がもう 1 つ増える・JSON 引数のパース/エラー処理・プロトコル別のレスポンス形式変換が追加で乗るので、体感でその 1.5〜2 倍程度の実装量になると見ている。
