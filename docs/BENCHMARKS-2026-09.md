# BENCHMARKS-2026-09 — mlxturbo vs mlx-serve (Flash-Next)

**下書き。数字はまだ無い。** `bench/suite/` (`docs/research/BENCH-DESIGN-2026-09.md`
が設計) の overnight tier を実走してから、`bench/suite/report.py --markdown`
の出力をこの文書の表の節にそのまま貼って完成させる。今回はその貼り込み場所と、
生成物では書けない文章 (何を主張するか・しないか、比較対象の版と起動オプション、
再現コマンド、免責) を先に固めておく回。

`docs/BENCHMARKS.md` (旧、`bench/spec_bench.py` 系のアドホック計測) とは別文書。
古い数字はここに混ぜない — 対戦相手も測り方も違うので、比較すること自体が
誤読を生む。旧文書の書式 (「その数字、自分の機械でも出るのか」を確認したい
懐疑的な読者向け、測定 JSON は配布せず再現コマンドで示す) は踏襲する。

## 目的と非目的

主張するのはこれだけ: 同一機・同一 HTTP 経路・同一プロンプトでの
mlxturbo と mlx-serve の Flash-Next 対戦結果、文脈点別の冷/温 TTFT と
decode tok/s、入力の性質 (プール) による受理率のばらつき、出力長と
thinking の影響。いずれも `bench/suite/` のスクリプトで再現できるコマンド
付きで示す。

主張しないのはこれだけ: 並列 (同時 N リクエスト) デコードの優劣 (mlxturbo
の並列デコード経路が未修正のため測っていない)、mlxturbo 以外のモデル
(Qwen3.8-27B dense、Gemma 4 系) での結果 (アダプタは用意してあるが、この
版ではまだ回していない)、量子化方式をまたいだ優劣 (MLX 系の同一パック
どうしの比較で、llama.cpp の K-quant とは比べていない)。「一点突破で勝って
いる文脈だけ載せる」「実利用の表だけで一点突破の弱さを隠す」はしない —
一点突破・池間のばらつき・出力長 x thinking の 3 種の表を必ずセットで出す。

## 比較対象

| | mlxturbo | mlx-serve |
|---|---|---|
| モデル | `~/models/ddalcu-mlxlm` (Qwen3.8-Flash-Next) | `~/models/ddalcu-flashnext-serve-4bit` |
| 量子化 | qwen4_exp 4bit affine group64 | qwen4_exp 4bit affine group64 |
| 起動コマンド | `python -m mlxturbo.server --model <path> --host 127.0.0.1 --port <port>` (n-gram/MTP はサイドカー自動発見、`--ngram`/`--mtp` 不要) | `mlx-serve --serve --model <path> --host 127.0.0.1 --port <port> --log-level info --mtp` |
| コミット | (overnight 実走時に `git rev-parse --short HEAD` で記録。`bench/suite/report.py` の `collect_environment_meta()` が自動収集する) | 同左 (`~/dev/mlx-serve` 側) |

両者とも同じ `qwen4_exp` 4bit affine group64 パックで、それぞれの推奨構成
どうしの比較 (同一重みの比較ではない — `docs/VS-MLX-SERVE.md` の注記と同じ
立場)。起動フラグの正本は `bench/suite/run.py` の `DEFAULT_MLXTURBO` /
`DEFAULT_MLXSERVE`。

## 計測環境

(overnight tier の実走後に `bench/suite/report.py --markdown` の
「(e) 手順と環境」節をここに差し替える。機種・macOS バージョン・mlx
バージョン・両リポジトリのコミット・乱数種・冷却秒数が入る。)

## 結果

以下は `bench/suite/report.py --markdown` が生成する 4 つの表の置き場所。
**今はまだ空— overnight tier の実走後に生成物で丸ごと差し替える。**
`--dry-run` の時点で分かっている一部の見積もり数値のみ、参考として仮置きする。

### (a) 見出し: エンジン x 文脈点 (プール `default`、出力長 512、thinking off)

(overnight tier の実走後に report.py --markdown で差し替え)

想定する文脈点: `0, 4000, 17000, 25000, 32000, 50000` (quick tier / overnight
のラダー軸と同じ)。列: 冷 TTFT (fresh) / 温 TTFT (p50) / decode tok/s (p50)。

### (b) 池ごとの差 (入力の性質による受理率のばらつき、10% 超に印)

(overnight tier の実走後に report.py --markdown で差し替え)

6 種のプール (日本語散文・英語散文・ソースコード・構造化データ・会話履歴・
反復の多いテキスト、定義は `docs/research/BENCH-DESIGN-2026-09.md` (c) 節)
を、短文脈 (0, 4000) と mlxturbo が実際に負けている長文脈 (17000) の両方で
振る。**プールを混ぜた平均は出さない** — 同一条件でプールだけ変えたときの
値をそのまま並べ、投機デコードの受理率がテキストの性質にどれだけ左右
されるかを見せる。

### (c) 出力長 x thinking

(overnight tier の実走後に report.py --markdown で差し替え)

出力長 128/1024、thinking off/on の組。thinking on でも `max_tokens` は
off と同じ値を送る (on 側にだけ生成予算を足して有利にしない)。

### (d) 分布 (min/p50/p95/max、反復 3 未満は「分布なし」)

(overnight tier の実走後に report.py --markdown で差し替え)

反復 3 回以上の条件だけ percentile を語る。反復 1-2 回 (standard tier 相当
の条件が混ざる場合) は「分布なし」と明記されたまま残す — 数字を隠さないが、
統計として語れないことも隠さない。

## 再現コマンド

```bash
# 1. 計画の確認 (GPU 不使用、モデルもトークナイザも読まない)
uv run python bench/suite/run.py --dry-run --tier overnight

# 2. 実走 (GPU 使用。数時間かかる。tools/biglock.sh で他のロック済み計測と競合しない)
tools/biglock.sh uv run python bench/suite/run.py --tier overnight

# 3. 中断したら同じ out-dir で再開
tools/biglock.sh uv run python bench/suite/run.py --resume bench/results/suite/<run-id>

# 4. この文書に貼る表を生成
uv run python bench/suite/report.py \
    --in bench/results/suite/<run-id>/raw.jsonl \
    --markdown docs/BENCHMARKS-2026-09.md.generated \
    --out bench/results/suite/<run-id>/detailed.md
# --markdown の出力から「結果」節の (a)-(d) を、この文書へ手で貼り込む
# (自動貼り込みはしない — 貼る前に数字を目で確認する余地を残すため)。
```

縮小版 (quick tier、約 2 時間) で「大本営発表にしない」形の速報を先に出す
こともできる。この場合 `bench/suite/report.py --markdown` は (b)(c) の表を
自動的に省略し、その旨を明記する (プール/出力長/thinking の掃引データが
無いため) — 省略された節がある版だと分かるように、この文書にも
「quick tier 版 (縮小版)」と明記して貼ること。

```bash
uv run python bench/suite/run.py --dry-run --tier quick
tools/biglock.sh uv run python bench/suite/run.py --tier quick
uv run python bench/suite/report.py --in bench/results/suite/<run-id>/raw.jsonl \
    --markdown bench/results/suite/<run-id>/BENCHMARKS-quick-preview.md
```

## 免責と既知の限界

数字は `bench/suite/report.py --markdown` の「(f) 注意書き」節がそのまま
本文に入る (熱・接頭辞キャッシュ検知・thinking の揃え方・生成長の揃え方・
既知の限界)。要点だけ先に書いておく:

冷 TTFT は各ブロックで最初に来たセルの rep=0 だけを信頼する。連続測定は
GPU が温まるため、それ以外の値は絶対値としては当てにならない (実測例:
17k prefill が 50 分の連続稼働で 37→57 秒)。「冷」と称した計測で接頭辞
キャッシュに当たっていた rep は `usage.prompt_tokens_details.cached_tokens`
で機械的に検出し、該当条件は除いて集計する。並列デコードは対象外、
量子化方式をまたいだ比較はしない (この版では MLX 系どうしのみ)。設計の
全体像とシナリオの定義は `docs/research/BENCH-DESIGN-2026-09.md` を参照。

## 更新履歴

- 2026-09-02: 下書き作成 (骨組みのみ、数字なし)。`bench/suite/` の
  overnight tier 実走後に表を差し替える。
