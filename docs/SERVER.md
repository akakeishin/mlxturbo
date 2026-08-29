# fastmlx-serve — 配布・接続ガイド

`fastmlx-serve` は OpenAI 互換 (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`) と
Anthropic 互換 (`/v1/messages`) の両方を話す HTTP サーバー。モデルは起動時に 1 回だけロードして常駐させ、
リクエストは内部で直列化する (`fastmlx/server.py` 冒頭の docstring 参照)。このドキュメントは配布物として
他人のネットワークで動かすときの起動方法・クライアント設定・制約をまとめる。

## 起動

```
uv run fastmlx-serve --model mlx-community/Qwen3.6-35B-A3B-4bit --served-model-name qwen36 --port 8765
```

`--model` 以外は全て省略可。ロードには数秒〜十数秒かかる (モデルサイズ依存)。起動ログに
バージョン・待ち行列上限・API キー認証の有無・thinking/tool calling 対応の検出結果が出る。

### オプション一覧

`uv run fastmlx-serve --help` が正。以下は抜粋 (この作業で追加したものは ← で示す):

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
| `--mtp PATH` | 無効 ← | MTP ヘッドを単一 safetensors サイドカーから読み込む (後述) |
| `--ngram DIR` | 無効 | n-gram (PLE) 表を外部サイドカーから読み込む |
| `--no-fused` | 無効 | hyper-connections 融合カーネルを無効化する |
| `--allowed-origins` | 無効 (CORS 無し) | ブラウザからのクロスオリジン fetch を許可する Origin (カンマ区切り、`*` で全許可) |
| `--max-sessions` | `8` | 会話ごとの session (KV/prompt cache) を同時保持する上限 (LRU) |
| `--max-context-tokens` | 自動検出 | 1 リクエストのプロンプト長上限。超えたら 400 |
| `--model-alias NAME` | なし | `--served-model-name` 以外にもこの名前を 404 にせず受け付ける (繰り返し指定可) |
| `--api-key KEY` | 無効 (認証なし) ← | このサーバーを叩ける API キー (繰り返し指定可) |
| `--max-queue N` | `8` ← | 直列化ロックの待ち行列上限。超えたら 503 |
| `--version` | — ← | バージョンを表示して終了 |

## API キー認証 (`--api-key`)

配布して他人のネットワークで動かす前提の機能。**未指定なら今までどおり認証なし** (ローカル専用の既定を変えない)。

```
uv run fastmlx-serve --model ... --api-key sk-fastmlx-xxxxxxxx
```

- OpenAI 系 (`/v1/chat/completions` / `/v1/completions` / `/v1/responses` / `/v1/models`) は
  `Authorization: Bearer <key>`、Anthropic 系 (`/v1/messages`) は `x-api-key: <key>` を見る。
  **どちらのヘッダもどちらの経路でも受け付ける** (クライアント実装の揺れがあるため)。
- 不一致は 401。エラー形式はプロトコルに合わせる (OpenAI は `invalid_api_key`、Anthropic は
  `authentication_error`)。キーの比較は `secrets.compare_digest` (タイミング攻撃対策)。
- `/health` と `/api/hello` は監視・疎通用なので鍵の有無に関わらず常に 200 を返す。
- `--api-key` は複数回指定でき、どれか 1 つに一致すれば通る (ローテーション用)。

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

## graceful shutdown

SIGTERM/SIGINT を受けると、新規リクエストは 503 で断りつつ、**処理中のリクエスト (ストリーミング中の
ものを含む) は完了させてから**終了する。もう一度シグナルを送る (種類は問わない) と即時終了する。

## `--mtp PATH`: MTP サイドカーの指定

Flash-Next 系の MTP ヘッドが単一の safetensors ファイル (サイドカー) として別途存在する場合、そのパスを
直接指定できる。

```
uv run fastmlx-serve --model ... --mtp "/Volumes/Mobile SSD/models/qwen38fn-mtp.safetensors"
```

- 指定すると、`--original` の生チェックポイントからの探索やバンドル済みアーティファクトより **優先**
  する。
- サイドカーの中身の形式 (テンソル名など) が既存のロード処理と合わない場合は、無理に合わせず
  **読めなかった旨をログに出して MTP 無しで起動する** (通常の「重みが無ければ無効化」と同じ姿勢)。
- サーバー側はパスを渡す配線だけに徹している。実際のロード・エンジン側の統合は別レーンの担当。

## クライアント接続例

いずれも `--served-model-name` に指定した id (例では `qwen36`) をクライアント側の `model` に使う。

### opencode

`opencode.json` (プロジェクト直下、または `~/.config/opencode/opencode.json`) に
`@ai-sdk/openai-compatible` プロバイダを追加する。baseURL は **`/v1` まで**含める:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "fastmlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "fastmlx (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8765/v1",
        "apiKey": "{env:FASTMLX_API_KEY}"
      },
      "models": {
        "qwen36": {
          "name": "Qwen3.6-35B-A3B (fastmlx)"
        }
      }
    }
  }
}
```

`--api-key` を立てていなければ `apiKey` は何を入れても無視される (サーバー側が見ない)。`/models` で
選択できるようになる。

### Codex CLI

Codex CLI (2026-02-01 以降) は `wire_api` として `"responses"` しか受け付けない (`"chat"` は削除済み) —
fastmlx-serve は `/v1/responses` を実装しているのでこれで問題ない。`model_providers` はユーザーレベルの
`~/.codex/config.toml` でのみ有効 (プロジェクトローカルの `.codex/config.toml` では無視される)。

他の Codex 設定と混ぜたくない場合は `CODEX_HOME` で丸ごと隔離できる:

```
export CODEX_HOME=/tmp/codex-fastmlx
mkdir -p "$CODEX_HOME"
```

`$CODEX_HOME/config.toml`:

```toml
model = "qwen36"
model_provider = "fastmlx"

[model_providers.fastmlx]
name = "fastmlx (local)"
base_url = "http://127.0.0.1:8765/v1"
env_key = "FASTMLX_API_KEY"
wire_api = "responses"
```

`--api-key` を立てていない場合も `env_key` に何かしらの環境変数を割り当てておく (Codex は
`Authorization: Bearer <値>` を必ず送るが、fastmlx 側は鍵未設定なら見ないので値は何でもよい)。

### Claude Code

`ANTHROPIC_BASE_URL` を `/v1/messages` の手前 (ベース URL) まで、`ANTHROPIC_AUTH_TOKEN` に鍵を設定する:

```
export ANTHROPIC_BASE_URL=http://127.0.0.1:8765
export ANTHROPIC_AUTH_TOKEN=sk-fastmlx-xxxxxxxx   # --api-key 未設定なら任意の値でよい
export ANTHROPIC_MODEL=qwen36
```

Claude Code は会話タイトル生成などの裏方処理で、メインモデルと別の小さいモデル名
(例: `claude-3-5-haiku-20241022`) を送ってくることがある。指定していないモデル名は既定で 404 になる
ため、これを許可するには起動時に `--model-alias claude-3-5-haiku-20241022` を足す (複数回指定可)。

`CLAUDE_CODE_MAX_CONTEXT_TOKENS` に注意: Claude Code は非 Anthropic モデルの実際の文脈長を検出できない
ため、この環境変数で申告しないと既定の 200k を前提に auto-compaction 等が動く。fastmlx-serve 側の
`--max-context-tokens` (または自動検出値、起動ログに出る) と数値を揃えておくこと — ここがずれると、
Claude Code 側が「まだ十分空きがある」と判断したのに fastmlx-serve 側は 400 (`context_length_exceeded`)
を返す、という不整合が起こる。

### Chatbox

設定画面で API の種類を「OpenAI API 互換」、API ホストを `http://127.0.0.1:8765/v1` に、モデル名を
`qwen36` に設定する。API キー欄は `--api-key` を立てていなければダミーの値で構わない (サーバー側が
見ない)。`--api-key` を立てている場合はそのキーをそのまま入れる。

## 制約

- **リクエストは直列処理。** `fastmlx-serve` は 91GB 級のモデルを 128GB 機に載せている前提で、1 リクエスト
  = 1 生成に直列化する設計 (継続バッチングは未実装)。複数クライアントを同時に繋ぐ場合、後着のリクエスト
  は先着の生成が終わるまで待たされる (`--max-queue` を超えると 503)。
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
