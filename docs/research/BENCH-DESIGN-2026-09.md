# BENCH-DESIGN-2026-09 — 公開ベンチマークの設計

**状態: 設計 + 骨組み (2026-09-02)。実行はしていない。** 骨組みは
`bench/suite/`(`engines.py` / `scenarios.py` / `run.py` / `report.py`)。
`--dry-run` の動作確認と構文チェックのみ済み (GPU 実行なし)。実際の対戦
(数字を取って判定する) は次のセッションの作業。

このリポジトリには対戦の一次データ・作法がすでにある
(`bench/self_snapshot.py`、`bench/vs_mlx_serve.py`、`docs/VS-MLX-SERVE.md`、
`docs/BENCHMARKS.md`、CLAUDE.md の「計測の作法」)。ここではそれを**作り直さず**、
(1) 複数エンジン・複数モデルへ広げられる形に一般化し、(2) 「一点突破」だけで
なく実利用に近いシナリオも同じページに並べ、(3) 公開に耐える再現手順として
固定する。

## (a) 目的と非目的

**目的**:

- 最初に回すのは **Qwen3.8-Flash-Next の mlxturbo 対 mlx-serve**。ただし
  他エンジン (oMLX、llama.cpp、mlx-lm) と他モデル (Qwen3.8-27B dense + MTP、
  Gemma 4 31B、Gemma 4 26B-A4B) をアダプタ追加で載せられる設計にする
  (`bench/suite/engines.py` の `EngineAdapter` / `ENGINE_REGISTRY`)。
- 「一点突破」の表 (冷 TTFT / decode を文脈別に単一ストリームで) と、
  「実利用に近い」表 (エージェント型マルチターン、コード編集ループ、RAG) と、
  品質 (KLD / top-1 一致) を **同じページに置く**。速さだけを見せない。
- 熱・接頭辞キャッシュ・thinking 不一致の罠を、手順ではなくハーネスの構造で
  防ぐ (下記 (d))。

**非目的 (今回やらないこと、理由込み)**:

- **並列 (同時 1/2/4/8) を既定では走らせない。** シナリオとしては定義する
  (`scenarios.py` の `build_parallel_scenario`) が、`enabled_by_default=False`。
  理由: mlxturbo の並列デコード (continuous batching) 経路は
  `docs/research/KERNEL-BRIEF-DECODE-BW.md` の「バッチ x 投機の判定」節で
  宣言していた反転条件 (B=4 で 1.6 倍未満なら畳む) を満たせず、既に畳まれて
  いる (`mlxturbo/batch_spec.py` は残るが誰も呼ばない)。直っていない経路を
  対戦材料にしない。
- **量子化方式の違いを揃えない、代わりに明記する。** MLX 系 (mlxturbo /
  mlx-serve / oMLX / mlx-lm) は同じパック (4bit affine group 64、
  `qwen4_exp`) を共有できるが、llama.cpp は K-quant で方式が違う。速度だけ
  並べると量子化の緩さの差を速度の差として誤読させる。**方式差を表に明記し、
  KLD を必ず併記する** (`docs/VS-MLX-SERVE.md` の「相手が公言している代償」
  節、CLAUDE.md の「品質を売って速度を買わない」を踏襲)。
- **「大本営発表」にしない。** 一点突破で勝っている文脈だけを載せて実利用の
  表を省く、逆に実利用の表だけで一点突破の弱さを隠す、どちらもしない。
  3 種類の表 (一点突破・実利用・品質) を常にセットで出す。

## (b) 指標の定義

すべて `bench/vs_mlx_serve.py` / `bench/self_snapshot.py` の既存の数え方と
一致させる (数え方を変えると過去の数字と比較できなくなる)。

| 指標 | 定義 | 備考 |
|---|---|---|
| 冷 TTFT | そのプロンプトを初めて見たときの最初の content/reasoning_content チャンクまでの壁時計秒 | 思考 (`reasoning_content`) も数える。本文だけだと長文脈で thinking が予算を食い尽くし 0 tok になる罠がある (`vs_mlx_serve.stream_once` docstring) |
| 温 TTFT | 「前のやり取り + 新しい発言」をまるごと送り直した 2 ターン目の TTFT | 実クライアントの挙動そのもの。接頭辞再利用が効いているかがここに出る |
| decode tok/s | `(n-1) / decode_s`。`n`=受信チャンク数、`decode_s`=最初のチャンク後の経過秒 | `n<=1` または `decode_s<=0` は 0.0 (NaN にしない。集計で自然に無視されるようにする一方、レポートでは NaN として除外する場所もあるので用途で使い分ける) |
| prefill tok/s | プロンプトのトークン数 / 冷 TTFT | 「冷」でないと接頭辞再利用ぶんだけ過大評価になる。`usage.cached_tokens` で確認すること (下記) |
| tok/round | 投機のラウンドあたり受理トークン数。両エンジンともログに出る (mlx-serve: `avg_per_round` / `[spec-stats]`、mlxturbo: `res["tokens"]` の長さ / ラウンド数) | 受理率そのものではなく「1 回のフォワードで何トークン進んだか」。壁時計の差の内訳を見るときに使う (積算しない。CLAUDE.md「無効化の積み上げで部品時間を見積もらない」と同じ理由で、ラウンド費用は単独では読めても足し算しない) |
| p50/p95 | 同一条件の反復 (既定 3 回以上) から取る中央値と 95 パーセンタイル | 反復 1-2 回では出さない。「3 回以上」の反復が無い指標には出さない |
| KLD | `bench/quant_eval.py` の近似 KLD (参照 top-K + 裾質量補正) | 受け入れ幅は現行比 +0.0005 (CLAUDE.md)。エンジン間比較では「両者とも同じ量子化方式か」を必ず明記する |
| top-1 一致 | `bench/quant_eval.py` の `top1_agree_mean` | KLD と並記。KLD が小さくても top-1 が動くケースの検出用 |

**接頭辞キャッシュ命中は HTTP レスポンスの `usage` から読む。** mlxturbo
(`mlxturbo/server.py:2530` `_usage_dict`) と mlx-serve
(`~/dev/mlx-serve/src/server.zig:9222` `formatChatUsage`) は、どちらも
OpenAI 標準 `usage.prompt_tokens_details.cached_tokens` を実装している
(ソースを実読して確認した事実で、推測ではない)。リクエストに
`stream_options: {"include_usage": true}` を付ければストリーム最終チャンクに
乗る。「冷 TTFT」と称した計測で `cached_tokens > 0` なら、**そのリクエストは
冷えていなかった** — レポートはこれを headline から自動的に警告として出す
(`bench/suite/report.py` の `cold_cache_hit_warning`)。ログの正規表現
(`[hot-cache] reused`、mlxturbo の `prefill reused=`) は二次確認に使う
(engine-agnostic ではないため)。

## (c) シナリオ

すべて `bench/suite/scenarios.py` に定義済み。ターンは「テンプレート列」で
表現し、実際の送受信は `run.py` が履歴を積みながら行う (実クライアントが
毎ターン「前のやり取り + 新しい発言」を丸ごと送るのと同じ流儀。
`bench/self_snapshot.py` の追記ターンを一般化した形)。

### point (一点突破)

プロンプトは `tools/_bench_text.py` の実文プール (docs の Markdown + 自前
ソース、約 145 万トークン相当) から切り出す窓で作り、ctx=0 だけは固定短文
(`vs_mlx_serve.SHORT`) を使う。文脈は既定で 6 点
(`0, 4000, 17000, 25000, 32000, 50000`、`docs/NEXT-SESSION-PROMPT.md` の
再現コマンドと同じ点) を振り、各点を冷ターン 1 回・追記の温ターン 1 回の
計 2 ターンで測る。tool call は使わない。生成長は既定 512 トークンとした。
CLAUDE.md が「tok/step の比較は複数プロンプト x 512 トークンの平均でだけ
行う」と定めているのに合わせた値で、短い決め打ちが要る場面は `--tokens`
で変えられる。

### agent (エージェント型マルチターン)

プロンプトは固定のタスク文 1 つと、`tools/_bench_text.py` の実文プールから
注入する疑似ツール出力で組み立てる。疑似ツール結果は 1 ターンあたり
~600 トークンあり、4 回の注入を経て文脈は最終ターンまでに 2k〜3k トークン
程度まで育つ。ターン構成はタスク依頼 1 回、疑似ツール結果と追加指示を重ねる
3 回、最後に要約を求める 1 回の計 5 ターンで、履歴は毎ターン累積する。
tool call は本物の function calling API を使わず、ツールの実行結果を人間が
転記した体で `user` ロールに注入する簡略化にとどめた。狙いは「大きな
外部テキストが繰り返し履歴へ追加され、各ターンの生成は短い」という
エージェント運用特有の文脈成長パターンの再現であって、tool_calls 自体の
正しさを検証することではない。生成長は既定 128 トークンとした。エージェント
は通常、構造化された短い応答を返すため。

### code-edit (コード編集ループ)

プロンプトはリポジトリ内の実ファイル (既定は `tools/_bench_text.py`
自身) をそのまま貼り付けて作る。繰り返し文字列は使わない — CLAUDE.md が
「繰り返し文字列で長さを作ってはいけない (n-gram/MTP が当たりすぎる)」と
定めているのを踏襲した。ターン構成はファイル全体を貼って編集を依頼する
1 回目に、追加の編集依頼を 2 回重ねる計 3 ラウンドで、履歴は累積するので
ラウンドを追うごとに直前の応答ぶん文脈が伸びる。tool call は使わない。
生成長は既定 512 トークンとした。毎ラウンド「変更後のファイル全体を返して」
と頼む設計なので、長めに取ってある。

### rag (検索拡張)

プロンプトは `tools/_bench_text.py` の実文プールから切り出す検索結果
チャンクと質問文で組み立てる。サブモードを 2 つ用意した。fresh (既定) は
毎回新しい検索結果を 4 チャンク (各 ~500 トークンの新規窓) 注入する独立
クエリを 4 回送る設計で、各ターンは履歴を引きずらない独立リクエスト
(`reset_history=True`) にしてあるため接頭辞はほぼ効かない — 検索結果が
毎回変わるという RAG の現実的な悪いケースを再現する狙いがある。shared は
最初のターンだけ検索結果チャンクを注入し、以降は同じ文脈へ短い追加質問を
重ねる (履歴累積) 設計で、接頭辞キャッシュが効くはずの対比条件になる。
どちらも tool call は使わない (検索そのものはハーネス側でテキストとして
注入するだけで、検索ツールを実際には呼ばない)。生成長は既定 256 トークン
とした。

### parallel (定義のみ、既定では走らせない)

同時 1/2/4/8 のリクエストを送り、集計 decode tok/s (総トークン数/壁時計) と
per-request レイテンシの p50/p95 を測る設計にした。ただし `(a)` 節の非目的
に書いた通り、既定では走らせない。実装するときは point と同じプロンプト源を
使い回す設計にすること — 指標だけは `build_parallel_scenario` の docstring
に書いてある。

## (d) 実行手順

**単位はブロック = 1 回のサーバー起動** (`(scenario, ctx, engine)` の組)。
「両サーバーを同時にメモリへ載せない (68GB + 91GB は 128GB に収まらない)」
制約から、そもそも 1 プロセスしか起動できない。ブロック内で `reps`
(既定 3) 回反復する (`self_snapshot.py` の反復と同じ — 起動は 1 回、
中で複数回測る)。

1. **スケジュール構築**: シナリオ x 文脈の全点を列挙し、**列挙順そのものを
   シャッフル**する。各点で「どちらのエンジンを先に起動するか」も毎回
   コインを振る (`build_schedule` の `rng.shuffle(pair)`)。これが
   「文脈ブロックごとに交互に起動する」の実装で、単純な A→B→B→A の
   固定パターンよりバイアス除去が広い (`vs_mlx_serve.py` の既存 ABBA 手法を
   包含しつつ、毎回どちらが先かをランダム化する)。
2. **乱数種はログに残す** (`--seed`)。再現したい dry-run / 実行はこの種を
   渡し直せば同じ順序になる。
3. **ブロックごとに**:
   - 起動 (`Server` コンテキストマネージャ、`vs_mlx_serve.py` を再利用)
   - ウォームアップ 2 段 (短文 → 対象文脈と重ならない長文) — カーネルの
     初回コンパイルと専門家重み/n-gram 行のページインを、測定と別のプロンプト
     で先に済ませる (`self_snapshot.py` と同じ理由: 同じプロンプトで
     温めると次の「冷」が冷えなくなる)
   - `reps` 回の反復測定。**rep=0 (フレッシュ起動直後) だけが真の「冷」**。
     rep>=1 は起動済みプロセスの中で GPU が温まっていくので、絶対値としては
     rep=0 より当てにならない。ただし同一起動セッション内で真に冷え直る
     わけではないので、`docs/NEXT-SESSION-PROMPT.md`「守ること」1. の
     「A/B は 2 本目以降で判定」とは意味が違う: あちらは同一プロセスを
     長時間使い回す decode_ab.py の話で「1 本目が暴走する」、こちらは
     ブロックごとに毎回フレッシュ起動するので **rep=0 が最も信頼できる**。
     この違いを混同しないこと (レポートの `cold_ttft_fresh_s` と
     `cold_ttft_all_median_s` を両方出すのはこの区別を可視化するため)。
   - 停止 (`Server.__exit__`: SIGTERM → 60 秒待って SIGKILL)
   - 停止した後で `pgrep -fl` を対象パターンに対して走らせ、
     `sysctl vm.swapusage` を記録する (`check_residual_processes`)。
     何か引っかかれば、`docs/NEXT-SESSION-PROMPT.md`「守ること」2. が
     指摘しているのと同じ「サーバーが Metal のメモリを握ったまま残る」
     症状を疑う
   - **冷却** (既定 180 秒)。`docs/research/SESSION-2026-09-02-CATCHUP.md`
     の実測 (GPU を 50 分回すと 17k prefill が 37→57s) を踏まえた既定値。
4. **ログ保存**: `--server-log` 相当でサーバーの stdout/stderr を
   `bench/results/suite/<run-id>/logs/` に残す (`self_snapshot.py --server-log`
   を踏襲)。キャッシュヒットの一次情報は `usage.cached_tokens` だが、
   ログはそれを裏取りする二次情報として保存する。
5. **thinking を揃える**: 既定 `--thinking off` (両エンジンに
   `reasoning_effort: "none"` を送る)。mlx-serve は qwen4_exp で thinking
   既定 off、mlxturbo はテンプレート既定で on — 揃えないと生成する文の種類が
   違い、MTP の受理率も比較にならない (`SESSION-2026-09-02-CATCHUP.md`)。
6. **窓を重ねない**: `PoolCursor` がプロセス内で 1 つの読み位置を共有し、
   シナリオ・文脈・rep をまたいでも同じ実文プールの窓を再利用しない
   (`self_snapshot.py` が同じ規律を文脈ごとの offset で実装しているのを
   プロセス全体に一般化した形)。
7. **生成長を揃える**: シナリオごとに固定の `max_tokens` を使い、エンジン間
   で変えない (CLAUDE.md「生成長を揃えて比較する」)。

## (e) エンジンアダプタの契約

`bench/suite/engines.py` の `EngineAdapter` (ABC)。新しいエンジンを足すときに
実装すること:

| メソッド | 役割 | 既定実装 |
|---|---|---|
| `build_argv(host, port)` | 起動コマンド (実行はしない) | 抽象。必須実装 |
| `is_available()` | バイナリ/モデルが手元にあるか | 常に True (具象クラスで Path 存在チェックに差し替える) |
| `unavailable_reason()` | 上が False のときの日本語理由 | None |
| `ready_url(host, port)` | 起動完了判定の URL | `/v1/models` (OpenAI 互換共通) |
| `wait_ready(port, timeout)` | 起動完了までポーリング | `vs_mlx_serve.wait_ready` をそのまま使う |
| `model_id(port)` | `/v1/models` が名乗る id (リクエストの `"model"` に使う) | `vs_mlx_serve.model_id` をそのまま使う。**プレースホルダを送らないこと** — 両エンジンとも完全一致以外は 404 にする (`vs_mlx_serve.py` の実際の失敗例) |
| `parse_log_signals(log_text)` | ログからの二次確認 (cache hit、thinking、spec mode) | 何も拾わない (エンジンごとに正規表現を持つ) |

停止は作り直さない — `Server` コンテキストマネージャ
(`vs_mlx_serve.Server`) をそのまま使う。SIGTERM → 60 秒待って SIGKILL、
`start_new_session=True` で切り離した子を `install_term_handler` の
SystemExit 経由で確実に道連れにする (`vs_mlx_serve.py` の実装意図をそのまま
引き継ぐ)。

**リクエストの投げ方**は `stream_with_usage` (`engines.py`) — 既存の
`vs_mlx_serve.stream_once` に `stream_options: {"include_usage": true}` を
足しただけの薄い拡張で、TTFT/decode の数え方 (思考も数える) は変えていない。

実装済み: `MlxturboAdapter`、`MlxServeAdapter` (どちらも既存スクリプトの
argv 組み立てと 1 対 1 対応)。**未実装 (スタブ)**: `OMlxAdapter`、
`LlamaCppAdapter`、`MlxLmAdapter` — `is_available()` が常に False を返し、
`build_argv` は `NotImplementedError` (dry-run はこれを SKIP として表示する
だけでクラッシュしない)。埋めるときにやることは各クラスの docstring に
書いてある。

## (f) モデルとパックの対応表

| モデル | mlxturbo | mlx-serve | 量子化 | 備考 |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | `~/models/ddalcu-mlxlm` (n-gram/MTP はサイドカー自動発見、`--ngram`/`--mtp` 不要 — `bench/results/logs/turbo-0902c.log` で確認済み) | `~/models/ddalcu-flashnext-serve-4bit` (`--mtp` 明示要) | 両者とも `qwen4_exp` 4bit affine group64 | 最初に回す対戦。`bench/suite/run.py` の既定プリセット |
| Qwen3.8-27B dense + MTP | 対応 (別モデルディレクトリ、`bench/spec_bench.py` 系列) | `~/.mlx-serve/models/ddalcu/Qwen3.8-27B-MLX-Serve-4bit` (`~/dev/mlx-serve/tests/bench.sh` の `qwen38-27b` 行) | 同上 | mlxturbo 側のモデルパスは環境ごとに違うので `--engines` の設定で渡す。アダプタの型は同じ `MlxturboAdapter` |
| Gemma 4 31B | **非対応** (mlxturbo は `_vendor/qwen4_exp.py` 系のみ、`CLAUDE.md` の「触ると壊れるもの」参照) | `$LMS_DIR/mlx-community/gemma-4-31b-it-4bit` | mlx-serve は 4bit | mlxturbo 列は空欄にする (無いものを埋めない)。もう一方のベースラインは `mlx-lm` アダプタ (未実装) |
| Gemma 4 26B-A4B (MoE) | **非対応** | `$LMS_DIR/mlx-community/gemma-4-26B-A4B-it-qat-4bit` | QAT 4bit | 同上 |
| llama.cpp 系 (将来) | — | — | **K-quant (方式が違う)** | 速度だけ並べない。KLD を必ず併記し、表の脚注に量子化方式の違いを明記する |

`$LMS_DIR` / `$MD` は `~/dev/mlx-serve/tests/bench.sh` のモデル行列と同じ
環境変数 (`~/.lmstudio/models`、`~/.mlx-serve/models`)。パスは機体依存なので
既定値は `bench/suite/run.py` の `DEFAULT_MLXTURBO` / `DEFAULT_MLXSERVE` に
定数として置き、他モデルは CLI 引数で差し替える設計 (アダプタのフィールドが
そのまま argparse の入力になる)。

## (g) 公開物

1. **生 JSON**: `bench/suite/run.py` が書く `bench/results/suite/<run-id>/raw.json`
   (ブロックごとの全 rep・全ターンの生値。`usage.cached_tokens` を含む) と
   `plan.json` (実行順・シード・見積もり)。`bench/results/` はリポジトリの
   `.gitignore` 対象 — 配布はリリース物として別途まとめる
   (`docs/BENCHMARKS.md` が既にやっている「元データは配布しない、
   コマンドで再現可能にする」方針を踏襲)。
2. **集計表 (Markdown)**: `bench/suite/report.py` が `raw.json` → 表に変換
   (`--in raw.json --out REPORT.md`)。冷 TTFT は `fresh (rep=0)` と
   `全 rep 中央値` を両方出す (熱の影響を隠さない)。
3. **スクリプト自体**: `bench/suite/*.py`。読者が自分の機体で再現できることが
   目的 (`docs/BENCHMARKS.md` の「その数字、自分の機械でも出るのか」と同じ
   立場)。
4. **機種/OS/コミットのメタ情報**: `report.py` の `collect_environment_meta()`
   — `hw.model`、`sw_vers`、mlxturbo と mlx-serve 双方の git コミット
   (別リポジトリなので個別に `git -C <path> rev-parse` する)、mlx のバージョン
   (import はせず `importlib.metadata` で問い合わせる — GPU に触れない)。
   `docs/BENCHMARKS.md` の「計測環境（共通）」欄に相当するものを、手で
   書き写す代わりに自動収集する。
5. **互換列**: `report.py --llmprobe <engine>=<path>` で
   `~/dev/mlx-serve/tests/bench.sh` (`llmprobe --bench-only --save`) の出力を
   同じ表に混ぜられる。mlxturbo が対応しないモデル (Gemma 4 系) でも、
   相手の自己申告値を並べて出せる。**cold/warm TTFT の区別が無い**など
   出典が違う値であることを表の脚注に明記する (無いものを揃わせない)。

## (h) 所要時間の見積もり

`bench/suite/run.py` の `estimate_block_seconds` が
`SESSION-2026-09-02-CATCHUP.md` の実測値 (文脈 0/4000/17000/50000 の冷
TTFT・decode tok/s・温 TTFT) を較正点にした線形補間で見積もる
(25000/32000 は 17000-50000 間の補間)。**実測ではない、大まかな時間予算**。

既定プリセット (Flash-Next、mlxturbo vs mlx-serve、point x 文脈 6 点、
各 3 反復、冷却 180 秒) の `--dry-run` 実測 (このドキュメントの検証時、
`--seed 42`):

```
ブロック数: 12 (文脈 6 点 x エンジン 2)
合計推定: 1h56m49s
```

内訳の目安 (1 ブロック = 起動 180s + ウォームアップ 20s + 測定 (反復 3) +
冷却 180s):

| 文脈 | 1 ブロックの推定 (mlx-serve / mlxturbo) |
|---|---|
| 0 | 約 6m51s / 6m55s |
| 4000 | 約 7m07s / 7m17s |
| 17000 | 約 8m17s / 8m51s |
| 25000 | 約 9m34s / 10m25s |
| 32000 | 約 10m42s / 11m46s |
| 50000 | 約 13m42s / 15m17s |

固定費 (起動 180s + 冷却 180s = 360s) が 12 ブロックで 4320s (72 分) を占める
— 文脈が短いほど固定費の割合が支配的になる。ブロック粒度を「文脈単位で
毎回起動し直す」設計にした代償で、(d) 節にある「rep=0 だけが真の冷」という
性質を得るためのトレードオフ。

シナリオを増やすと (`--scenarios point,agent,code-edit,rag`)、agent/code-edit/
rag は較正点が無いため見積もりに含まれない (「較正データ無しのブロック」と
して件数だけ表示される) — 較正のやり直しには実測が要る。

## (i) 判定が反転する条件 (測る前に宣言する)

- **「冷」と称した rep で `usage.cached_tokens > 0`** → その rep は真に
  冷えていなかった。headline から除外し、`report.py` の警告として明記する
  (`vs-mlx-serve` の 17k 冷 TTFT が 0.21s と出て気づいた実例と同じ罠 —
  今回はログではなく `usage` で機械的に検出できる)。
- **rep=0 と rep>=1 の冷 TTFT 差が 10% を超える** → そのブロックは熱の
  立ち上がりの影響下にある可能性が高い (`SESSION-2026-09-02-CATCHUP.md`
  「1 本目が熱の立ち上がりで 37.7 → 60.7s と暴走」と同型の現象)。
  再測定するか、rep=0 を「参考値」に格下げして report に明記する。
- **prefill の差が 1.1 倍未満** → 前セッションで観測された大きな差は
  ハーネスの産物 (接頭辞キャッシュ、thinking 不一致) だった前例
  (`SESSION-2026-09-02-CATCHUP.md`) と同型。「差は実在しない」と判定し、
  そのシナリオ/文脈点のレーンは畳む。
- **同一条件・同一エンジンでの反復 3 回の分散が中央値の 5% を超える**
  → 「テキスト運」(受理率がプロンプト依存で振れる、
  `docs/BENCHMARKS.md` の flash_spec 受理率の自己訂正と同型) の疑い。
  反復を増やすかプロンプト集合を広げてから判定する。単一プロンプトの
  値では判定しない (CLAUDE.md)。
- **エンジン間で生成トークン数 (`n_tokens`) が max_tokens 未満で揃わない**
  (片方だけ早期に EOS で止まる等) → decode tok/s の比較が生成長不一致の
  産物になる (CLAUDE.md「生成長を揃えて比較する」の逆パターン)。揃わない
  ペアは decode 比較から除外し、理由を明記する。
- **並列シナリオを走らせる判断をするとき** → mlxturbo の並列デコードが
  実際に直り、`docs/research/KERNEL-BRIEF-DECODE-BW.md` のバッチ判定が
  再訪されて反転したときだけ、`enabled_by_default=True` に変える。

## 付録: 骨組みのファイル一覧

```
bench/suite/
  engines.py    EngineAdapter 基底 + MlxturboAdapter/MlxServeAdapter (実装済み)
                + OMlxAdapter/LlamaCppAdapter/MlxLmAdapter (スタブ)。
                stream_with_usage() で usage.cached_tokens を拾う。
  scenarios.py  Scenario / TurnTemplate / PoolCursor。
                point / agent / code-edit / rag(fresh|shared) / parallel(定義のみ)。
  run.py        スケジュール構築 (シャッフル・冷却)・所要時間見積もり・
                --dry-run・実行ループ (起動→測定→残留確認→冷却)。
  report.py     raw.json → Markdown。cached_tokens 警告、環境メタ情報、
                llmprobe 互換列。
```

**再現 (dry-run、GPU 不使用)**:

```bash
.venv/bin/python bench/suite/run.py --dry-run
# 既定: Flash-Next, mlxturbo vs mlx-serve, point x 文脈 6 点, 各 3 反復,
# ランダム順, 冷却 180 秒。コマンド列・ターン構成・所要時間見積もりを表示し、
# bench/results/suite/<timestamp>/plan.json に書き出す (サーバー起動・HTTP・
# モデル読み込みは一切行わない)。
```

**実行 (GPU 使用。本ドキュメントでは実行していない)**:

```bash
tools/biglock.sh uv run python bench/suite/run.py
uv run python bench/suite/report.py --in bench/results/suite/<run-id>/raw.json \
    --out bench/results/suite/<run-id>/REPORT.md
```
