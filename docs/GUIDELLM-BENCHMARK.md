# GuideLLM 公開ベンチマーク

API経由の速度はGuideLLM 0.7.3で測る。TTFT、ITL、request latency、request/token
throughputを同じOpenAI互換経路から取り、JSON・CSV・HTMLを一度に残す。内部の
`self_snapshot`は実装診断、GuideLLMは外へ示す比較という役割分担にする。

## 導入

```bash
tools/guidellm.sh setup
tools/guidellm.sh --version
```

`tools/compare/guidellm-venv/`へ隔離して0.7.3を固定する。このディレクトリは
gitignore済み。macOSではGuideLLM既定の`fork` workerがtorch初期化後にSIGSEGVする
ことがあるため、wrapperは`spawn`と`num_workers=0`を強制する。この条件で
mlxturboの実エンドポイントへ2 requestを送り、JSON出力まで確認済み。

## 正式プロトコル

サーバーは別ターミナルで起動し、モデルを使うサーバー側だけを`tools/biglock.sh`
で包む。比較対象も同じモデル名、thinking off、temperature 0、streaming SSE、
同じtokenizer、同じ冷却条件に揃える。

GuideLLMが固定出力長のため送る`ignore_eos: true`は、chat/completionsのstream・
non-stream全4経路で解釈する。これを明示したrequestはEOSで止めず、指定した
`max_completion_tokens`まで生成する。起動時にEOS方針を固定するbatch coordinatorへは
流さず、request単位の直列経路を使う。明示的なstop文字列、context上限、エラーは別なので、
公開表には要求長に加えてGuideLLM JSONの実`output_tokens`も必ず載せる。

```bash
BIGLOCK_PRIO=0 BIGLOCK_NO_WORKER=1 tools/biglock.sh \
  .venv/bin/python -m mlxturbo.server \
  --model /path/to/model --served-model-name bench-model \
  --host 127.0.0.1 --port 8000 --require-runner flash_spec
```

単独requestのTTFT/ITLは文脈長ごとに別cellで取る。次は4k入力、512出力、
1回のwarmupを除いた11 requestの例。

```bash
.venv/bin/python bench/guidellm_bench.py \
  --engine mlxturbo --model bench-model --tokenizer /path/to/model \
  --mode latency --prompt-tokens 4000 --output-tokens 512 \
  --requests 12 --warmup-requests 1 --cooling always-on \
  --name mlxturbo-latency-4k
```

共有prefixのhot prefillは、4k prefixを全requestで共有し、64 tokenだけを
requestごとに変える。`prefix-count`を増やすと複数の共有prefixがLRU内で競合する。

```bash
.venv/bin/python bench/guidellm_bench.py \
  --engine mlxturbo --model bench-model --tokenizer /path/to/model \
  --mode prefix --prefix-tokens 4000 --prefix-count 1 \
  --prompt-tokens 64 --output-tokens 256 --requests 12 \
  --cooling always-on --name mlxturbo-prefix-4k
```

同時接続は固定stream列と自動sweepを分ける。固定列は回帰比較、自動sweepは
飽和点の発見に使う。

```bash
.venv/bin/python bench/guidellm_bench.py \
  --engine mlxturbo --model bench-model --tokenizer /path/to/model \
  --mode concurrent --streams 1,2,4,8 --prompt-tokens 512 \
  --output-tokens 256 --requests 24 --cooling always-on

.venv/bin/python bench/guidellm_bench.py \
  --engine mlxturbo --model bench-model --tokenizer /path/to/model \
  --mode sweep --sweep-size 6 --max-concurrency 8 \
  --prompt-tokens 512 --output-tokens 256 --requests 24 \
  --cooling always-on
```

## 公開時に必ず添えるもの

- GuideLLM、mlxturbo、比較対象、MLXの版とgit commit。
- 機種、RAM、OS、電源、冷却条件、測定順。
- prompt/output token分布、request数、warmup除外数、seed。
- p50とp95。反復2回の値を分布として扱わない。
- server logの`prefill reused/new`。hot値をcold値のように見せない。
- JSONを正本とし、CSV/HTMLは閲覧用とする。

結果は既定で`bench/results/guidellm/`へ出る。公開前には`--dry-run`で生成された
`*.scenario.json`を目視し、比較する全エンジンで同一scenarioになっていることを
確認する。
