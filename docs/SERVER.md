# mlxturbo-serve — 配布・接続ガイド

`mlxturbo-serve` は OpenAI 互換 (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`) と
Anthropic 互換 (`/v1/messages`) の両方を話す HTTP サーバー。モデルは起動時に 1 回だけロードして常駐させ、
リクエストは内部で直列化する (`mlxturbo/server.py` 冒頭の docstring 参照)。このドキュメントは配布物として
接続方法・運用上の制約をまとめる。既定の `127.0.0.1` はローカル利用向け。LAN/WAN に公開する場合は、
後述のとおり TLS を終端する reverse proxy または SSH tunnel を必ず併用する。

## 起動

```
uv run mlxturbo-serve --model mlx-community/Qwen3.6-35B-A3B-4bit --served-model-name qwen36 --port 8765
```

`--model` 以外は全て省略可。ロードには数秒〜十数秒かかる (モデルサイズ依存)。起動ログに
バージョン・待ち行列上限・API キー認証の有無・thinking/tool calling 対応の検出結果が出る。

Flash-Next の投機デコードが必須なら、黙って通常生成へ落ちないよう起動条件も固定する:

```
uv run mlxturbo-serve --model /path/to/qwen38fn-mlx-v-l --ngram /path/to/qwen38fn-ngram-4bit \
  --served-model-name qwen38fn --require-runner flash_spec --port 8765
```

別ターミナルから次を実行し、`runner` と `fallback_reason` を確認してからクライアントを繋ぐ:

```
curl -sS http://127.0.0.1:8765/health
curl -sS http://127.0.0.1:8765/v1/models
```

### オプション一覧

`uv run mlxturbo-serve --help` が正。以下は主要オプションの抜粋:

| オプション | 既定 | 内容 |
|---|---|---|
| `--model` | (必須) | サーブするモデル (パスまたは HF リポジトリ) |
| `--original` | `Qwen/Qwen3.8-27B` | MTP 探索・投機デコードの契約判定に使う生チェックポイント |
| `--served-model-name` | `--model` の basename | `GET /v1/models` とレスポンスの `model` 欄で名乗る id |
| `--host` | `127.0.0.1` | bind するホスト |
| `--port` | `8000` | bind するポート |
| `--max-tokens` | `4096` | 1 リクエストあたりの `max_tokens` 上限 |
| `--temp` | `0.7` | `temperature` 省略時の既定値 |
| `--mtp-bits` | `4` | MTP ヘッドの量子化ビット数 |
| `--no-mtp` | 無効 | MTP を読み込まず lookup (SAM) のみで投機する |
| `--mtp PATH` | 無効 | MTP ヘッドを単一 safetensors サイドカーから読み込む (後述) |
| `--ngram DIR` | 無効 | n-gram (PLE) 表を外部サイドカーから読み込む |
| `--no-fused` | 無効 | hyper-connections 融合カーネルを無効化する |
| `--allowed-origins` | 無効 (CORS 無し) | ブラウザからのクロスオリジン fetch を許可する Origin (カンマ区切り、`*` で全許可) |
| `--max-sessions` | `8` | 会話ごとの session (KV/prompt cache) を同時保持する上限 (LRU、1以上) |
| `--max-context-tokens` | 自動検出 | 1 リクエストのプロンプト長上限。超えたら 400 |
| `--model-alias NAME` | なし | `--served-model-name` 以外にもこの名前を 404 にせず受け付ける (繰り返し指定可) |
| `--api-key KEY` | 無効 (認証なし) | このサーバーを叩ける API キー (繰り返し指定可) |
| `--max-queue N` | `8` | 直列化ロックの待ち行列上限。超えたら 503 |
| `--require-runner KIND` | なし | 解決 runner が `flash_spec` / `spec` / `fallback` の指定値と違えば exit 1 |
| `--version` | — | バージョンを表示して終了 |

## API キー認証 (`--api-key`)

**未指定なら今までどおり認証なし** (ローカル専用の既定を変えない)。

```
uv run mlxturbo-serve --model ... --api-key sk-mlxturbo-xxxxxxxx
```

- OpenAI 系 (`/v1/chat/completions` / `/v1/completions` / `/v1/responses` / `/v1/models`) は
  `Authorization: Bearer <key>`、Anthropic 系 (`/v1/messages`) は `x-api-key: <key>` を見る。
  **どちらのヘッダもどちらの経路でも受け付ける** (クライアント実装の揺れがあるため)。
- 不一致は 401。エラー形式はプロトコルに合わせる (OpenAI は `invalid_api_key`、Anthropic は
  `authentication_error`)。キーの比較は `secrets.compare_digest` (タイミング攻撃対策)。
- `/health` と `/api/hello` は監視・疎通用なので鍵の有無に関わらず常に 200 を返す。
- `--api-key` は複数回指定でき、どれか 1 つに一致すれば通る (ローテーション用)。

`--api-key` は認証だけを行い、通信を暗号化しない。`http://` のまま別ホストへ公開すると、キーと会話本文を
経路上から読まれる。`--host 0.0.0.0` を使う場合は nginx/Caddy 等で HTTPS を終端するか、サーバー自体は
`127.0.0.1` のまま SSH tunnel 越しに接続すること。また、現在は key-file / 環境変数専用オプションがなく、
コマンドライン引数は shell history や同一ホストの process listing に残り得る。共有ホストではこの制約を
理解したうえで reverse proxy 側の認証を使う。

## 待ち行列上限 (`--max-queue`)

直列化ロック (1 リクエスト = 1 生成) の待ち行列が無制限だと、公開サーバーでは詰まりの温床になる。
既定 8。ロック待ち + 処理中の合計がこれに達すると、新規リクエストは **503** (`Retry-After: 1` 付き) を
即座に返す — 生成を試みることすらしない。`GET /health` の `queue_depth` で現在の待ち行列の深さを見れる。

## SSE keepalive

長いプレフィル (実測: Claude Code から 97k トークンのプロンプトを送ったケースで約 3 分) の間、
ストリーミング接続に何もデータが流れないと、クライアントやその手前のプロキシがタイムアウトで
コネクションを切ることがある。ストリーミング中、**最初のトークンが出るまでの間は 15 秒ごとに
SSE コメント行 (`: keepalive`) を送る** (OpenAI / Anthropic / Responses の全経路)。SSE コメント行は
仕様上クライアントに無視されるため、通常のイベント処理には影響しない。

クライアント切断後は、次の token callback で decode を協調停止する。既に実行中の長い prefill や Metal
kernel 自体は割り込めないため、その区間だけは終了を待ってからロックとキュー枠を解放する。

## graceful shutdown

SIGTERM/SIGINT を受けると、新規リクエストは 503 で断りつつ、**処理中のリクエスト (ストリーミング中の
ものを含む) は完了させてから**終了する。もう一度シグナルを送る (種類は問わない) と即時終了する。

## Flash-Next の MTP 自動発見と `--mtp PATH`

Flash-Next (`qwen4_exp`) では、次の優先順で MTP を解決する。

1. `--mtp PATH` の明示指定
2. モデル本体の `model.safetensors.index.json` が指す `mtp.*` シャード、または index のない
   `model.safetensors` 内の `mtp.*` テンソル
3. モデルディレクトリ直下の `mtp.safetensors` サイドカー

`--model` が Hugging Face repo ID の場合も、ロード済みローカル snapshot を探索する。サイドカーを別の
場所に置く場合だけ `--mtp` を使う:

```
uv run mlxturbo-serve --model ... --mtp "/Volumes/Mobile SSD/models/qwen38fn-mtp.safetensors"
```

- `--mtp` を明示したのに2回のロード試行とも失敗した場合は、通常生成へ落とさず理由を出して **exit 1** する。
- 自動発見候補の壊れた index・欠損シャード・読み込み失敗は通常生成へフォールバックし、`/health` の
  `fallback_reason` に理由を出す。有効なサイドカーが別にあれば、壊れた index よりそちらを使う。
- runner を運用条件にしたい場合は `--require-runner flash_spec` を付ける。MTP が見つからない場合も exit 1
  になるため、配布環境で性能低下を見逃さない。

## クライアント接続例

いずれも `--served-model-name` に指定した id (例では `qwen36`) をクライアント側の `model` に使う。

### opencode

`opencode.json` (プロジェクト直下、または `~/.config/opencode/opencode.json`) に
`@ai-sdk/openai-compatible` プロバイダを追加する。baseURL は **`/v1` まで**含める:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "mlxturbo/qwen36",
  "provider": {
    "mlxturbo": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "mlxturbo (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8765/v1",
        "apiKey": "{env:MLXTURBO_API_KEY}"
      },
      "models": {
        "qwen36": {
          "name": "Qwen3.6-35B-A3B (mlxturbo)"
        }
      }
    }
  }
}
```

`--api-key` を立てていなければ `apiKey` は何を入れても無視される (サーバー側が見ない)。`/models` で
選択できるようになる。

```
MLXTURBO_API_KEY=dummy opencode run --pure "Return exactly: mlxturbo-ok"
```

### Codex CLI

Codex CLI (2026-02-01 以降) は `wire_api` として `"responses"` しか受け付けない (`"chat"` は削除済み) —
mlxturbo-serve は `/v1/responses` を実装しているのでこれで問題ない。`model_providers` はユーザーレベルの
`~/.codex/config.toml` でのみ有効 (プロジェクトローカルの `.codex/config.toml` では無視される)。

他の Codex 設定と混ぜたくない場合は `CODEX_HOME` で丸ごと隔離できる:

```
export CODEX_HOME=/tmp/codex-mlxturbo
mkdir -p "$CODEX_HOME"
```

`$CODEX_HOME/config.toml`:

```toml
model = "qwen36"
model_provider = "mlxturbo"

[model_providers.mlxturbo]
name = "mlxturbo (local)"
base_url = "http://127.0.0.1:8765/v1"
env_key = "MLXTURBO_API_KEY"
wire_api = "responses"
```

`--api-key` を立てていない場合も `env_key` に何かしらの環境変数を割り当てておく (Codex は
`Authorization: Bearer <値>` を必ず送るが、mlxturbo 側は鍵未設定なら見ないので値は何でもよい)。

```
CODEX_HOME=/tmp/codex-mlxturbo MLXTURBO_API_KEY=dummy \
  codex exec --skip-git-repo-check "Return exactly: mlxturbo-ok" < /dev/null
```

### Claude Code

`ANTHROPIC_BASE_URL` を `/v1/messages` の手前 (ベース URL) まで、`ANTHROPIC_AUTH_TOKEN` に鍵を設定する:

```
export ANTHROPIC_BASE_URL=http://127.0.0.1:8765
export ANTHROPIC_AUTH_TOKEN=sk-mlxturbo-xxxxxxxx   # --api-key 未設定なら任意の値でよい
export ANTHROPIC_MODEL=qwen36
```

```
claude -p "Return exactly: mlxturbo-ok" --model qwen36
```

Claude Code は会話タイトル生成などの裏方処理で、メインモデルと別の小さいモデル名
(例: `claude-3-5-haiku-20241022`) を送ってくることがある。指定していないモデル名は既定で 404 になる
ため、これを許可するには起動時に `--model-alias claude-3-5-haiku-20241022` を足す (複数回指定可)。

`CLAUDE_CODE_MAX_CONTEXT_TOKENS` に注意: Claude Code は非 Anthropic モデルの実際の文脈長を検出できない
ため、この環境変数で申告しないと既定の 200k を前提に auto-compaction 等が動く。mlxturbo-serve 側の
`--max-context-tokens` (または自動検出値、起動ログに出る) と数値を揃えておくこと — ここがずれると、
Claude Code 側が「まだ十分空きがある」と判断したのに mlxturbo-serve 側は 400 (`context_length_exceeded`)
を返す、という不整合が起こる。

### Chatbox

設定画面で API の種類を「OpenAI API 互換」、API ホストを `http://127.0.0.1:8765/v1` に、モデル名を
`qwen36` に設定する。API キー欄は `--api-key` を立てていなければダミーの値で構わない (サーバー側が
見ない)。`--api-key` を立てている場合はそのキーをそのまま入れる。

## 制約

- **リクエストは直列処理。** `mlxturbo-serve` は 91GB 級のモデルを 128GB 機に載せている前提で、1 リクエスト
  = 1 生成に直列化する設計 (継続バッチングは未実装)。複数クライアントを同時に繋ぐ場合、後着のリクエスト
  は先着の生成が終わるまで待たされる (`--max-queue` を超えると 503)。
- **token logprobs は未対応。** Chat/Completions/Responses のレスポンス中の `logprobs` は `null` または空配列
  で、リクエストの `logprobs` / `top_logprobs` を計算しない。候補のrerank・信頼度表示・評価用途で必要な
  クライアントは、値が返る前提で使わないこと。
- **spec runner (投機デコード) 経路は恒等値以外のサンプリングパラメータを 400 にする。** `top_p=1.0` や
  `repetition_penalty=1.0` のような「分布を変えない既定値」は通すが、それ以外 (`top_p=0.9` 等) を投機
  デコード経路 (`SpecRunner`) へ渡すと 400 になる。投機デコードのブロック検証は「対象分布からの厳密な
  サンプリング一致」を前提にしており、サンプリングパラメータで分布を歪めると検証の意味が壊れるため。
  通常生成 (`FallbackRunner`) 経路ではこの制限はない。
- **決定性は同一構成内でのみ保証する。** `temp=0` (greedy) は完全一致検証で出力分布が厳密に同一になるが、
  これは同じサーバー構成 (同じチャンク幅 `PREFILL_STEP_SIZE`、同じ量子化、同じ融合カーネルの有無) の
  範囲内での話。チャンク幅が変わると `mx.quantized_matmul` がバッチ長に応じて異なる丸めを返し得るため、
  構成が変わればビット一致は保証されない。
- **文脈長ガードと `--max-context-tokens`。** プロンプトが上限を超えると 400 (`context_length_exceeded`)
  で弾く。既定はモデルの config (`max_position_embeddings` 等) と、Metal が一括確保できる実際の上限
  から逆算した値のうち小さい方を自動で使う (起動ログに出る)。手元で上限を明示したい場合は
  `--max-context-tokens` で上書きする。
