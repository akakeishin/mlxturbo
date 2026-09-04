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

## 2026-09-03 のフルベンチ (self_snapshot、公開ベンチの suite ではなく単発の点計測)

条件: M3 Max 128GB、macOS 26.4、mlx 0.32.2。両エンジン同日・同冷却 (10 分冷却後に起動)、反復 1、生成 256 トークン、
thinking off、Qwen3.8-Flash-Next 4-bit (mlxturbo: `ddalcu-mlxlm` (lm_head 8-bit) + n-gram interleaved、mlx-serve:
`ddalcu-flashnext-serve-4bit` `--mtp`、origin/main 8058076)。数字は `docs/research/SESSION-2026-09-02-CATCHUP.md`
の 2026-09-03 12:20 の節。

| 文脈 | 冷 TTFT serve / turbo | 温 TTFT serve / turbo | decode tok/s serve / turbo |
|---|---|---|---|
| 4k | 5.70 / 6.88 | 0.85 / 0.15 | 55.7 / 51.4 |
| 17k | 27.8 / 31.6 | 0.91 / 0.20 | 49.0 / 45.3 |
| 25k | 41.3 / 46.6 | 0.89 / 0.23 | 60.8 / 48.0 |
| 32k | 51.6 / 58.3 | 0.90 / 0.25 | 47.4 / 44.9 |
| 50k | 82.1 / 93.1 | 15.9 / 0.93 | 45.9 / 46.9 |

主張すること: 温 TTFT (接頭辞キャッシュが当たる 2 ターン目以降) は mlxturbo が 4〜17 倍速い。冷 prefill は mlx-serve が
1.13〜1.21 倍速い。decode は ±10% で、反復 1 回の揺れの範囲。主張しないこと: 品質の優劣 (同じ 4-bit パックだが
lm_head のビットが違う)、25k の decode 差 (tok/step の当たり)。

## 2026-09-04 の常時冷却フルベンチ (self_snapshot、反復2)

ユーザーが普段使う常時冷却を固定し、追加の強冷却は使わなかった。生成256 token、
thinking off、文脈0/4k/17k/25k/32k/50k、各2回の中央値。mlxturboは
`ddalcu-mlxlm-head4` + n-gram sidecar、mlx-serveは
`ddalcu-flashnext-serve-4bit --mtp`。JSONは
`bench/results/self-snapshot-full-normalcool-{turbo,serve}-0904.json`。

| 文脈 | 冷TTFT turbo / serve | 温TTFT turbo / serve | 温の倍率 | decode turbo / serve |
|---:|---:|---:|---:|---:|
| 0 | 0.174 / 0.185 s | 0.143 / 0.730 s | **5.11倍** | **53.55** / 48.46 tok/s |
| 4k | 6.111 / 5.861 s | 0.160 / 0.865 s | **5.39倍** | 52.66 / **55.51** tok/s |
| 17k | 28.791 / 28.388 s | 0.207 / 0.891 s | **4.30倍** | **49.89** / 48.37 tok/s |
| 25k | **40.424** / 43.632 s | **0.488** / 0.902 s | **1.85倍** | **51.86** / 47.43 tok/s |
| 32k | **52.733** / 55.968 s | **0.752** / 0.923 s | **1.23倍** | **51.56** / 49.21 tok/s |
| 50k | **84.315** / 89.843 s | **0.600** / 16.871 s | **28.10倍** | **45.11** / 41.58 tok/s |

冷TTFTは0〜17kでほぼ互角、25k以降はmlxturboが6.1〜7.9%速い。decodeは
6条件中5条件で3.1〜10.5%速く、4kだけ5.1%遅い。ただし反復2なのでp95を主張せず、
4k serveの52.46/58.55 tok/sや25k turbo温TTFTの0.235/0.741秒という振れも
隠さない。

温TTFTは単なる「50k prefillが0.6秒」ではない。直前のassistant出力を含む履歴を
送り直し、token IDのLCPを再利用した値である。mlxturboの50kは一方が49,826 tokenを
再利用して274 tokenを新規処理、もう一方が50,088 tokenを再利用して16 tokenを処理した。
mlx-serveは2GiBのhot-cache上限で40,960/50,105 tokenしか残らず、9,145 tokenを
再処理して16.7秒掛かった。長文会話ではこの容量差がdecodeの数%差より大きい。

この結果は内部診断の速報で、公開用の正本はGuideLLM 0.7.3へ移す。
`docs/GUIDELLM-BENCHMARK.md`の固定scenarioでrequest数を増やし、TTFT/ITLの
p50/p95、実output token数、冷却、server側reused/newを併記する。

## 2026-09-04 のQwen3.6-35B-A3B強冷却フルベンチ

本体4bit + MTP-5bit、thinking off、生成256 token、文脈0/4k/17k/25k/32k/50kを
各2回。開始前に10分以上休止し、固定10秒GEMMが12.76 TFLOPS（冷えた基準
12.78比-0.16%）だったことを確認した。JSONは
`bench/results/qwen36-full-strongcool-tokenfix-0904.json`。

| 文脈 | 冷TTFT | 温TTFT | decode | 短文比 | input tok/s |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.082 s | 0.254 s | 117.98 tok/s | — | 231 |
| 4k | 2.310 s | 0.286 s | **120.16 tok/s** | +1.8% | 1,652 |
| 17k | 13.490 s | 0.425 s | 104.68 tok/s | -11.3% | 1,246 |
| 25k | 21.155 s | 0.499 s | 86.64 tok/s | -26.6% | 1,173 |
| 32k | 28.216 s | 0.542 s | 82.54 tok/s | -30.0% | 1,128 |
| 50k | 52.886 s | 0.691 s | 65.56 tok/s | **-44.4%** | 942 |

SSE deltaの個数ではなく、結合した本文を同じtokenizerで数え直した値をdecodeの分母にした。
旧chunk方式は条件ごとに2.4〜5.0%過小評価していた。修正後も長文脈低下は残り、サーバー内部値とも
概ね一致する。tok/stepは短2.36〜2.38、50k 2.20〜3.00で採択率だけの崩壊ではなく、40層中
10個のfull-attention層とMTP側1層が長いKVを読むverify/draft/repair stepの費用増が主因。
50k同一process ABBAでは汎用SDPA幅分割autoがoff比でms/token -19.1%、ms/round -19.9%、
tok/round -0.9%。正式fullはauto込みであり、offなら概算57 tok/sまで落ちる。

同じ既定でround anatomyを取ると、短文3本平均20.10 ms/roundに対して50kは39.03 ms。
内蔵phaseのverifyは17.49→34.53 msで、総増分18.93 msの約90%を占めた。draftは
2.34→3.23 ms、maintは0.26→0.31 ms。Metal probeでもdispatchは約2,132→2,167回/roundと
ほぼ同じだが、GPU和集合は17.81→34.39 ms、平均kernelは8.3→15.9 usへ増えた。したがって
長文脈低下はPython糊やdispatch増ではなく、主にfull-attention verifyのKV帯域費用である。

### 2026-09-05 条件付き幅4の採用と50k参考値

temp=0.7の同一process回文A/Bでは、入力3,830 token以上を幅4以下・lookupなしにすると、現行の
3/8/16比で4k/17k/50kが9/9非退行、平均ms/token -10.4/-9.4/-8.9%だった。50kは
62.7→68.8 tok/s（+9.8%）。幅4と現行上限幅9のΔKLDは+0.000043で、受け入れ幅+0.0005内。
短文は1/3 promptが+8.9%悪化したため、3,830未満にはlookupを残す。

製品配線後、旧fullと同一の50k prompt 2本を最初に再生した参考値はcold TTFT 54.226秒、
warm TTFT 0.617秒、decode 69.06 tok/s（個別65.23/72.90）。ただし固定GEMMは前12.74から
直後12.01 TFLOPSへ低下し、前後±1.5%の熱ゲートに不合格だった。この絶対値を公開比較には使わず、
採用判断と改善率は熱条件を交互化した上記A/Bを正本とする。
