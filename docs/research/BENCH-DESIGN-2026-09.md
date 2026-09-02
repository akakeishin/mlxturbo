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
| min/p50/p95/max/CV | 同一条件の反復から取る最小・中央値・95 パーセンタイル・最大・変動係数 (標本標準偏差/平均) | 数字自体は反復 1-2 回でも出すが、**3 回未満は「分布なし」と明記する** (`report.py` の `has_distribution`) — percentile を語るには足りないという注記であって、数字を隠すことではない |
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

### 入力の多様性: `point` が振る 6 種のプール

`tools/_bench_text.py` の実文プール 1 つだけで文脈窓を切っていると、
パターンが少なくばらつきが見えない。投機デコード (MTP/n-gram) の受理率は
テキストの性質で大きく動くので、1 つの池だけで測ると片方のエンジンに
有利な数字が固定される恐れがある。そこで `point` は文脈窓の中身を 6 種類の
実文プールに分け、同じ文脈点でプールを替えて測れるようにした
(`bench/suite/scenarios.py` の `PROMPT_POOLS`)。繰り返し文字列で長さを
作らない規律 (CLAUDE.md) はどのプールでも守っていて、すべて実在する
ファイルの連結から窓を切り出す。実測トークン数は `~/models/ddalcu-mlxlm`
のトークナイザで実際にエンコードして測った値 (`scenarios.py` の
`POOL_TOKEN_BUDGET`、2026-09-02 時点):

| プール | 分類 | 出所 | 実測トークン数 |
|---|---|---|---|
| `ja-prose` | (a) 日本語散文 | `docs/**/*.md` (このリポジトリの研究ログ・設計文書) | 267,148 |
| `en-prose` | (b) 英語散文 | `README.en.md` + `~/dev/mlx-serve/docs/**/*.md` (英語。無ければ `.venv` の英語 README/RST/METADATA で補う) | 318,066 |
| `source-code` | (c) ソースコード | python (mlxturbo/tools/bench の自前ソース) + zig (`~/dev/mlx-serve/src/*.zig`、あれば) | 4,108,940 |
| `structured-data` | (d) 構造化データ | `bench/results/*.json` (計測結果の生 JSON) + `bench/results/logs/*.log` | 2,094,687 |
| `conversation` | (e) 会話履歴 (構成物) | ja-prose と source-code の断片を `User:`/`Assistant:` プレフィックスで積んだトランスクリプト形。**本物の保存済み会話ログではない** (下記注記) | 398,099 |
| `repetitive` | (f) 反復の多いテキスト | 表・箇条書きが密な Markdown 24 本 (`docs/README.md`、`docs/MTP-FLASH.md` 等。選定基準は `scenarios.py` の `_REPETITIVE_FILES` docstring) | 59,574 |

`conversation` プールは注意が要る。このリポジトリには再利用できる形の
保存済み会話ログが無い (`bench/opencode_e2e.py` は実サーバーとやり取りする
e2e ドライバで、静的なテキストとしては読めない)。代わりに ja-prose と
source-code の実文プールから段落・関数単位の断片を切り出し、役割
プレフィックスを挟んで積んだ「トランスクリプト形」の文字列を作っている。
文そのものは毎回違う実文から取るので同じ短い文字列を繰り返してはいないが、
**内容として本物の対話ではない** — 作っているのは「役割プレフィックスと
細かい段落区切りが挟まる」という、単一の長い説明文とは違うトークン分布
であって、対話の意味的な整合性を測るものではないことを明記しておく。

6 種のうち `repetitive` が最小 (59,574 トークン) で、これが `point` の
文脈点をどこまで広げられるかの律速になる。`(h)` 節の tier 設計はこの
実測値から逆算してある。`--dry-run` は毎回、その回の計画がどのプールを
どれだけ消費するかを実測予算と突き合わせて表示する
(`bench/suite/run.py` の `pool_demand_report`)。予算を超えるプールがあれば
`OVER` と表示し、実行すると `PoolCursor.take()` が `ValueError` で止まる
(窓を使い果たしたら繰り返しで埋めるのではなく、素直に止まる設計)。

結果は**プールを混ぜた平均だけを出さない**。`report.py` は
`(シナリオ, 文脈, プール, 出力長, thinking)` の組ごとに集計し、同じ条件で
プールだけ変えたときのばらつきを別表に出す (`(i)` 節、`(g)` 節)。

### point (一点突破)

プロンプトは上記 6 種のプール (既定では `"default"` — `tools/_bench_text.py`
の実文プールそのもの、後方互換のための既定値) から切り出す窓で作り、ctx=0
だけは固定短文 (`vs_mlx_serve.SHORT`) を使う。文脈は既定で 6 点
(`0, 4000, 17000, 25000, 32000, 50000`、`docs/NEXT-SESSION-PROMPT.md` の
再現コマンドと同じ点) を振り、各点を冷ターン 1 回・追記の温ターン 1 回の
計 2 ターンで測る。tool call は使わない。

**出力長と thinking も軸になる。** 生成長は短 (128) と長 (1024) の 2 段を
既定の行列に入れ (CLAUDE.md の「tok/step の比較は複数プロンプト x 512
トークンの平均」を踏まえつつ、短文応答と長文応答で受理率の挙動が違う点を
軸として明示的に測る)、thinking は off/on の 2 値を軸にする。on のときは
`reasoning_content` も生成トークンとして数え (`stream_with_usage` が両サーバー
で同じ扱いをする)、**両エンジンに同じ `reasoning_effort` を送る**
(`docs/research/SESSION-2026-09-02-CATCHUP.md` の thinking 不一致の罠を
踏まない)。on のときの生成長は思考込みで揃える — `max_tokens` の値自体は
off/on で変えない (thinking がその予算の中で思考と本文を取り合う、という
現実の制約をそのまま測る設計で、on 側にだけ予算を足して有利にしない)。

池 x 出力長 x thinking を全部振ると行列が大きくなるので、既定の掃引は
`--tier` (quick/standard/overnight) で選ぶ (`(h)` 節)。`--pools`/
`--tokens-set`/`--thinking-set`/`--ctxs` で個別に上書きもできる。

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
制約から、そもそも 1 プロセスしか起動できない。ブロックの中には**セル**
(プール x 出力長 x thinking の組。`point` 以外は 1 セルだけ) が 1 つ以上
入っていて、各セルを `reps` 回ずつ反復する (`self_snapshot.py` の反復と
同じ — 起動は 1 回、中で複数回測る。セルの直積は `(h)` 節の tier が決める)。

1. **スケジュール構築**: `(シナリオ, 文脈)` の組を列挙し、**列挙順そのものを
   シャッフル**する。各点で「どちらのエンジンを先に起動するか」も毎回
   コインを振り、その点に属するセルの順序もシャッフルする
   (`group_into_blocks` の `rng.shuffle`)。これが「文脈ブロックごとに
   交互に起動する」の実装で、単純な A→B→B→A の固定パターンよりバイアス
   除去が広い (`vs_mlx_serve.py` の既存 ABBA 手法を包含しつつ、毎回どちらが
   先かをランダム化する)。
2. **乱数種はログに残す** (`--seed`)。再現したい dry-run / 実行はこの種を
   渡し直せば同じ順序になる (`--resume` もこの種を `plan.json` から復元して
   同じスケジュールを再構成する)。
3. **プール残量を dry-run の時点で確認する**: `point` のセルが要求する
   窓の合計トークン数を、`(c)` 節の実測プール予算と突き合わせる
   (`pool_demand_report`)。超えているプールは `OVER` と表示し、実行すると
   `PoolCursor.take()` が `ValueError` で止まることを事前に警告する。
4. **ブロックごとに**:
   - 起動 (`Server` コンテキストマネージャ、`vs_mlx_serve.py` を再利用)
   - ウォームアップ 2 段 (短文 → 対象文脈と重ならない長文) — カーネルの
     初回コンパイルと専門家重み/n-gram 行のページインを、測定と別のプロンプト
     で先に済ませる (`self_snapshot.py` と同じ理由: 同じプロンプトで
     温めると次の「冷」が冷えなくなる)
   - ブロック内の全セル x `reps` 回の反復測定。**ブロックで最初に来た
     セルの rep=0 (フレッシュ起動直後) だけが真の「冷」**。それ以外
     (同じセルの rep>=1、2 番目以降のセル) は起動済みプロセスの中で GPU が
     温まっていくので、絶対値としては当てにならない。ただし同一起動
     セッション内で真に冷え直るわけではないので、
     `docs/NEXT-SESSION-PROMPT.md`「守ること」1. の「A/B は 2 本目以降で
     判定」とは意味が違う: あちらは同一プロセスを長時間使い回す
     decode_ab.py の話で「1 本目が暴走する」、こちらはブロックごとに毎回
     フレッシュ起動するので **ブロック最初のセルの rep=0 が最も信頼できる**。
     この違いを混同しないこと (レポートの `cold_ttft_fresh_s` と
     `cold_ttft_stats` を両方出すのはこの区別を可視化するため)。
   - 停止 (`Server.__exit__`: SIGTERM → 60 秒待って SIGKILL)
   - 停止した後で `pgrep -fl` を対象パターンに対して走らせ、
     `sysctl vm.swapusage` を記録する (`check_residual_processes`)。
     何か引っかかれば、`docs/NEXT-SESSION-PROMPT.md`「守ること」2. が
     指摘しているのと同じ「サーバーが Metal のメモリを握ったまま残る」
     症状を疑う
   - **冷却** (既定 180 秒)。`docs/research/SESSION-2026-09-02-CATCHUP.md`
     の実測 (GPU を 50 分回すと 17k prefill が 37→57s) を踏まえた既定値。
5. **ログを 1 ブロック終わるたびに追記する**: `raw.jsonl` (JSON Lines、
   1 行 1 ブロック) に書き、`--server-log` 相当でサーバーの stdout/stderr も
   `bench/results/suite/<run-id>/logs/` に残す (`self_snapshot.py
   --server-log` を踏襲)。キャッシュヒットの一次情報は `usage.cached_tokens`
   だが、ログはそれを裏取りする二次情報として保存する。
6. **中断と再開**: `--tier overnight` のような長時間ジョブが電源断や
   SIGTERM で止まっても、`--resume <out-dir>` で `plan.json` からシード/
   tier/軸を復元し、`raw.jsonl` に記録済みのブロック index をスキップして
   続きから走る。1 ブロックの失敗 (プール枯渇の `ValueError` 等) はジョブ
   全体を落とさず、`failed: true` の行として記録して次のブロックへ進む
   (`report.py` は失敗ブロックを集計から外し、別表で理由を残す)。
7. **thinking を揃える**: `point` の thinking は tier/`--thinking-set` の
   軸そのもの (off/on 両方測る)、それ以外のシナリオは既定 `--thinking off`
   (両エンジンに `reasoning_effort: "none"` を送る)。mlx-serve は qwen4_exp
   で thinking 既定 off、mlxturbo はテンプレート既定で on — 揃えないと
   生成する文の種類が違い、MTP の受理率も比較にならない
   (`SESSION-2026-09-02-CATCHUP.md`)。on のときも off と同じ `max_tokens`
   を送る (`(c)` 節、on 側に予算を足して有利にしない)。
8. **窓を重ねない**: `PoolRegistry`/`PoolCursor` がブロック内でプールごとに
   1 つの読み位置を共有し、セル・rep をまたいでも同じ窓を再利用しない
   (`self_snapshot.py` が同じ規律を文脈ごとの offset で実装しているのを
   プール単位に一般化した形)。**エンジンをまたぐときは意図的に例外**:
   mlxturbo と mlx-serve は同じ `(シナリオ, 文脈)` に属するセルを同じ順で
   持つので、それぞれ独立にフレッシュな `PoolRegistry` (offset 0) から
   引く — これは「同じ本文を両エンジンに送って比較する」ための意図した
   一致であって、プール予算の消費としては 1 回分としてしか数えない
   (`(c)` 節の実測予算チェックはこの前提で計算している)。
9. **生成長を揃える**: シナリオごとに固定の `max_tokens` を使い、エンジン間
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

1. **生ログ**: `bench/suite/run.py` が 1 ブロックずつ追記で書く
   `bench/results/suite/<run-id>/raw.jsonl` (JSON Lines。ブロックごとの
   全セル・全 rep・全ターンの生値。`usage.cached_tokens` を含む) と
   `plan.json` (実行順・シード・tier・軸・見積もり)。JSONL にしたのは
   `--resume` (中断再開、(d) 節) のため — 1 ブロック終わるたびに 1 行
   追記されるので、途中で落ちても直前までの記録が残る。`bench/results/`
   はリポジトリの `.gitignore` 対象 — 配布はリリース物として別途まとめる
   (`docs/BENCHMARKS.md` が既にやっている「元データは配布しない、
   コマンドで再現可能にする」方針を踏襲)。
2. **集計表 (Markdown)**: `bench/suite/report.py` が `raw.jsonl` → 表に変換
   (`--in raw.jsonl --out REPORT.md`)。`(シナリオ, 文脈, プール, 出力長,
   thinking)` ごとにエンジン別の行を作り、冷 TTFT は `fresh` (ブロック最初の
   セルの rep=0) と `min/p50/p95/max + 変動係数` を両方出す (熱の影響を
   隠さない、反復 1-2 回は「分布なし」と明記する)。**池を混ぜた平均は
   出さず**、池間のばらつきを別表にする ((c) 節)。
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
TTFT・decode tok/s・温 TTFT、thinking off・プール default) を較正点にした
線形補間で見積もる (25000/32000 は 17000-50000 間の補間)。**thinking=on と
プール差 (default 以外) の較正データは無い** — この表は thinking=off・
プール=default の実測値をそのまま流用した近似で、それを `--dry-run` の
出力にも明記する。**実測ではない、大まかな時間予算。**

`--tier` は 3 段。tier=standard/overnight の文脈点は当てずっぽうではなく、
`(c)` 節の実測プール予算 (最小のプール `repetitive`、59,574 トークン) から
「池 x 出力長(2) x thinking(2) x reps の合計消費量が予算に収まる上限」を
逆算して選んである — 計算は `bench/suite/run.py` の `pool_demand_report`
と一致する。

### quick (既定、約 2 時間)

`point` x 文脈 6 点 (`0, 4000, 17000, 25000, 32000, 50000`) x プール 1 種
(`default`) x 反復 3。骨組みの前バージョンから変えていない。`--dry-run
--tier quick` (`--seed` はどれでも合計推定は変わらない — ブロックの集合が
同じなら順序を変えても合計は不変):

```
ブロック数: 12 (文脈 6 点 x エンジン 2)
合計推定: 1h56m49s
```

### standard (`--tier standard`、数時間)

`point` x 文脈 3 点 (`0, 4000, 8000`) x プール 6 種 (`POOL_ORDER`) x
出力長 2 種 (`128, 1024`) x thinking 2 値 (`off, on`) x 反復 1 (**分布は
出ない** — `report.py` が「分布なし」と明記する)。文脈点をここまで絞った
理由: 6 種のうち最小の `repetitive` (59,574 トークン) を基準に、文脈
(0,4000,8000) x セル 24 種 (プール 6 x 出力長 2 x thinking 2) x 反復 1 の
消費量が 46,400 トークン (予算の 77.9%) に収まるよう選んだ。もう 1 点
足すと超える。

```
ブロック数: 6 (文脈 3 点 x エンジン 2)
合計推定: 1h24m41s
```

### overnight (`--tier overnight`、反復 3 回以上で p50/p95、一晩)

standard と同じ「池 x 出力長 x thinking の全部」の組を反復 3 回以上で回し
(p50/p95 が出せる)、かつ quick と同じ長文脈ラダー (プール `default` のみ、
これは 1,070,797 トークンの予算があるので長文脈でも余裕がある) も反復 3
回以上で回す。両者を `(シナリオ, 文脈)` の組で合流させる (`0` と `4000` は
2 つの掃引軸のセルが 1 ブロックに同居する)。池の掃引側は反復 3 回になった
ぶん、文脈点をさらに `(0, 4000)` の 2 点に絞ってある —
`repetitive` の消費量は 45,600 トークン (予算の 76.5%) で、`(4000, 8000)`
の 2 点はそのままでは反復 3 回に耐えない (`(0,4000,8000)` x 反復 3 だと
139,200 トークン要求、予算の 2.3 倍で確実に `ValueError` になる)。

```
ブロック数: 12 (長文脈ラダー 6 点 + 池掃引 2 点、うち 0/4000 は合流)
合計推定: 3h11m34s
```

3 時間強は「一晩」と呼ぶには短いが、**反復 3 回以上で全条件の分布が出る**
という、このドキュメントの実データ制約下で正直に出せる最大の掃引がこれ。
もっと長く回したければ `--reps` を上げるか (実データの予算が許す範囲で)
`--scenarios` を足す — `--tier overnight --resume` は中断しても続きから
やり直せるので、複数晩に分けて伸ばすこともできる。

### 内訳の目安 (quick の各文脈点、1 ブロック = 起動 180s + ウォームアップ 20s + 測定 (反復 3) + 冷却 180s)

| 文脈 | 1 ブロックの推定 (mlx-serve / mlxturbo) |
|---|---|
| 0 | 約 6m51s / 6m55s |
| 4000 | 約 7m07s / 7m17s |
| 17000 | 約 8m17s / 8m51s |
| 25000 | 約 9m34s / 10m25s |
| 32000 | 約 10m42s / 11m46s |
| 50000 | 約 13m42s / 15m17s |

固定費 (起動 180s + 冷却 180s = 360s) が quick の 12 ブロックで 4320s
(72 分) を占める — 文脈が短いほど固定費の割合が支配的になる。ブロック
粒度を「文脈単位で毎回起動し直す」設計にした代償で、(d) 節にある
「ブロック最初のセルの rep=0 だけが真の冷」という性質を得るための
トレードオフ。standard/overnight は 1 ブロックに複数セルを詰めることで、
この固定費をセル数で割って薄めている (standard は 1 ブロックあたり 24
セルで、固定費の比率が quick よりずっと小さい)。

`--scenarios` に `agent`/`code-edit`/`rag`/`parallel` を足すと、これらは
tier の池/出力長/thinking 軸を使わない (`(c)` 節のスコープ通り、掃引は
`point` だけ) ので 1 セルのブロックとして増える。較正点が無い (`agent`
等の較正はまだ実測していない) ため、`ctx=4000` 相当を代用した近似で
見積もりに加算される — `--dry-run` の合計に含まれるが精度は低い。

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
- **同じ条件でプールだけ変えたときの spread (`(max-min)/中央値`) が 10%
  を超える** → `report.py` の「池間のばらつき」表で印が付く。これは
  「エンジンの優劣」ではなく「投機デコードの受理率がテキストの性質に
  依存する」ことの直接証拠として扱う — 片方のプールだけを見出しに使わない。
  10% を超えた指標は、6 プール全部の値を併記しないと結論を出さない
  (単一プールの数字だけを headline にしない)。
- **反復が 1-2 回の条件 (`standard` tier や `--reps` を絞った実行) を
  percentile 付きで語ろうとしたとき** → `report.py` は `has_distribution`
  を見て「分布なし」と明記する。min/p50/max の数字自体は出すが、
  「p95 が◯◯だった」のような強い主張はしない。分布が要る結論は
  `overnight` (反復 3 回以上) の結果を待つ。
- **プール残量チェックで `OVER` と出た組み合わせを走らせたとき** →
  実行中のどこかで `PoolCursor.take()` が `ValueError` を投げてそのセルが
  失敗する。`run.py` はブロック単位で捕まえてジョブ全体は継続するので、
  「一部のセルが失敗ブロックとして記録され、集計から外れる」のは想定内の
  劣化であって異常終了ではない。ただし本番の判定に使う組み合わせは
  `OVER` が出ない範囲に収めること (既定の `standard`/`overnight` は
  収まるように選んである)。

## 付録: 骨組みのファイル一覧

```
bench/suite/
  engines.py    EngineAdapter 基底 + MlxturboAdapter/MlxServeAdapter (実装済み)
                + OMlxAdapter/LlamaCppAdapter/MlxLmAdapter (スタブ)。
                stream_with_usage() で usage.cached_tokens を拾う。
  scenarios.py  Scenario / TurnTemplate / PoolCursor / PoolRegistry。
                PROMPT_POOLS (6 種、POOL_TOKEN_BUDGET に実測トークン数)。
                point(ctx, tokens, pool) / agent / code-edit / rag(fresh|shared)
                / parallel(定義のみ)。
  run.py        Cell/AxisConfig/TIERS (quick/standard/overnight)・
                セル展開 (expand_cells)・ブロック構築 (group_into_blocks、
                シャッフル・プール残量チェック)・所要時間見積もり・
                --dry-run・実行ループ (起動→全セル測定→残留確認→冷却)・
                --resume (中断再開)。
  report.py     raw.jsonl → Markdown。min/p50/p95/max・変動係数、
                「分布なし」明記、池間ばらつき表 (10% 印)、cached_tokens
                警告、環境メタ情報、llmprobe 互換列。
```

**再現 (dry-run、GPU 不使用)**:

```bash
.venv/bin/python bench/suite/run.py --dry-run
# tier=quick (既定): Flash-Next, mlxturbo vs mlx-serve, point x 文脈 6 点,
# プール "default" 1 種, 各 3 反復, ランダム順, 冷却 180 秒。

.venv/bin/python bench/suite/run.py --dry-run --tier standard
.venv/bin/python bench/suite/run.py --dry-run --tier overnight
# どちらもプール残量チェック・コマンド列・セル構成・所要時間見積もりを
# 表示し、bench/results/suite/<timestamp>/plan.json に書き出す
# (サーバー起動・HTTP・モデル読み込みは一切行わない)。

.venv/bin/python bench/suite/run.py --dry-run --tier standard \
    --pools repetitive,ja-prose --tokens-set 128 --thinking-set off --reps 2
# --pools/--tokens-set/--thinking-set/--ctxs を指定すると tier の軸を
# 独自の組に差し替える。
```

**実行 (GPU 使用。本ドキュメントでは実行していない)**:

```bash
tools/biglock.sh uv run python bench/suite/run.py --tier overnight
# 途中で止まったら (電源、SIGTERM 等) 同じ out-dir で再開:
tools/biglock.sh uv run python bench/suite/run.py --resume bench/results/suite/<run-id>

uv run python bench/suite/report.py --in bench/results/suite/<run-id>/raw.jsonl \
    --out bench/results/suite/<run-id>/REPORT.md
```
