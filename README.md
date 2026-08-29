# mlxturbo

[English](README.en.md) | 日本語

Apple Silicon (MLX) 上のローカル推論エンジンと、OpenAI / Anthropic / Responses 互換の HTTP サーバー。

**速いのは特定の2アーキテクチャだけ**で、それ以外のモデルは素の [mlx-lm](https://github.com/ml-explore/mlx-lm) と同速で動く汎用ランタイムです（後述の対応表を参照）。「汎用高速化ランタイム」ではありません。

Ollama や LM Studio から来た人が最初に踏む差分: **1 プロセスにつき 1 モデルしかロードしない。** モデルの切り替えはプロセスの再起動が要る（複数モデルの同時ホストは非対応）。詳しくは後述の「制約」を参照。

## これは何か

- モデルを 1 プロセスにロードして常駐させ、投機デコード（自己投機: 本体のモデル自身が出す MTP ヘッド + 文脈 suffix-lookup）で decode を高速化するエンジン
- 上記エンジンを OpenAI 互換 (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`) と Anthropic 互換 (`/v1/messages`) の両方で叩ける HTTP サーバー (`mlxturbo-serve`)
- 対応していないモデルでも動く（フォールバック）が、その場合は投機なしの素の mlx-lm と同速

## 対応表（正直に）

| 経路 | 対象 | 投機 | 実測倍率（mlx-lm 比、後述の再現コマンド参照） |
|---|---|---|---|
| `flash_spec` | Qwen3.8-Flash-Next (`qwen4_exp` アーキテクチャ) + MTP サイドカー | あり（MTP 深さ1 + hyper-connections 融合カーネル） | 約 1.25x〜1.39x（課題の種類に依存、後述） |
| `spec` | `qwen3_5` 契約を満たすモデル（例: Qwen3.8-27B 系） | あり（MTP チェイン + suffix-lookup 混成） | 1.3x〜2.2x（プロンプト内容依存、後述） |
| `fallback` | 上記以外の全モデル | なし | 1.0x（素の mlx-lm と同速） |

どの経路が選ばれたかは起動ログと `GET /health` の `runner` / `fallback_reason` で確認できる。「投機が効くはずなのに黙って fallback に落ちている」を防ぎたい場合は `--require-runner flash_spec`（または `spec`）を付けて起動すると、条件を満たさないときに起動自体が失敗する（詳細は [`docs/SERVER.md`](docs/SERVER.md)）。

## 投機が効く条件

- `flash_spec` が動くには、モデルの `model_type` が `qwen4_exp` で、かつ MTP 重みが見つかっている必要がある（`--mtp` で明示指定するか、本体シャードに同梱されているか、サイドカーとして自動発見される）。MTP が見つからないときは投機無しの `flash_spec` にはならず、経路自体が成立しない。実測は [`docs/MTP-FLASH.md`](docs/MTP-FLASH.md) にある
- `spec` が動くには、モデルが `qwen3_5` の contract（`SpecEngine` が要求する層構成や attention の形）に合っている必要がある。合っていなければ `fallback` に落ちる
- 両経路とも、`temp=0`（greedy）は出力分布と厳密に一致する完全一致検証で動き、`temp>0` も棄却サンプリングで分布は厳密に同一のまま。ただし `top_p<1.0` や `repetition_penalty≠1.0` のように分布を変えるサンプリングパラメータを渡すと 400 になる。投機側のブロック検証が「対象分布からの厳密なサンプリング一致」を前提にしているため、サンプリングパラメータで分布そのものを歪められると検証が成り立たなくなる。この制限は `fallback` 経路には無い

## インストール

Apple Silicon Mac（macOS、MLX/Metal が使える環境）が前提。

```
git clone <このリポジトリ>
cd mlxturbo
uv sync
```

## モデルの入手

- `fallback` / `spec` 経路: 通常の mlx-lm 互換チェックポイント（Hugging Face repo ID かローカルパス）をそのまま指定できる
- `flash_spec` 経路（Qwen3.8-Flash-Next）: 元 checkpoint からの変換とMTP抽出が要る。手順は [`docs/MTP-FLASH.md`](docs/MTP-FLASH.md) と `mlxturbo/convert_flash.py --help`（`estimate` / `extract-mtp` / `convert` の各サブコマンド）を参照。`qwen4_exp` アーキテクチャは mlx-lm 本体に無いが、`mlxturbo` を import した時点で自動的に解決される（利用者の mlx-lm パッケージへは何も書き込まない）

## 起動

対話 CLI（`spec` 経路、27B 系向け）:

```
uv run mlxturbo --model <path-or-repo-id> --prompt "こんにちは"
```

HTTP サーバー:

```
uv run mlxturbo-serve --model <path-or-repo-id> --served-model-name mymodel --port 8000
```

接続方法・オプション一覧・API キー認証・opencode / Codex CLI / Claude Code / Chatbox からの接続例は [`docs/SERVER.md`](docs/SERVER.md) に詳しくまとめてある。

## 制約

- **1 プロセス 1 モデル。** モデルは起動時に 1 回だけロードして常駐する設計で、複数モデルの切り替えは別プロセスを立てる必要がある
- **リクエストは直列処理。** 継続バッチング（continuous batching）は未実装で、1 リクエスト = 1 生成に直列化される。複数クライアントを同時に繋ぐと、後着のリクエストは先着の生成が終わるまで待たされる（待ち行列上限 `--max-queue` を超えると 503）
- **spec / flash_spec 経路は非恒等値のサンプリングパラメータを 400 にする。** 理由は上述のとおり、投機のブロック検証が厳密な分布一致を前提にしているため
- **token logprobs は未対応。** レスポンスの `logprobs` は常に `null` または空配列
- 詳しい制約一覧（決定性の範囲、文脈長ガードなど）は [`docs/SERVER.md`](docs/SERVER.md) の「制約」節を参照

## 実測値と再現コマンド

以下はすべて M3 Max 128GB / macOS 26.4 / mlx 0.32.2 での実測。ハードウェア世代が変わると数字は変わる（詳しくは [`docs/research/ROOFLINE-2026-08-26.md`](docs/research/ROOFLINE-2026-08-26.md) の「ハード世代が変わると失効する判定」を参照）。

### `spec` 経路（Qwen3.8-27B-4bit、greedy、512 tok）

| 条件 | decode tok/s | 対 mlx-lm |
|---|---|---|
| mlx-lm 素（フォールバック相当） | 21〜23 | 1.0x |
| 自己投機・難しい内容持続（code） | 31.9 | 1.49x |
| 自己投機・難しい内容持続（prose） | 28.3 | 1.32x |

再現: `uv run mlxturbo --model <model> --prompt "<prompt>"`（生成の後に `decode tok/s` が自動で出力される）。mlx-lm 素との対照は `uv run python bench/baseline.py <model-id>`。

### `flash_spec` 経路（Qwen3.8-Flash-Next、v-l レシピ + MTP 4bit、greedy）

受理率は 160 トークン × 10 課題で測った（反復 200 以上）。

| 課題 | 受理率 | 倍率 |
|---|---|---|
| 散文（英語 0.720 ± 0.046 / 日本語 0.682 ± 0.047） | 約 0.70 | 約 1.39x |
| コード | 0.564 ± 0.068 | 約 1.25x |

投機なしは 31.14 ms/token。**効くかどうかは言語ではなく課題の種類で決まり、コードが最も低い**（信頼区間が重ならない）。どの課題でも投機は効く。

短い試行（48 トークン、反復 30 前後）では言語差があるように見えるが、標準誤差が ±0.09 あり有意ではない。**受理率は反復 300 以上で見ること。**経緯は [`docs/MTP-FLASH.md`](docs/MTP-FLASH.md) を参照。

再現: `tools/spec_flash_accept.py`（受理率）と `tools/spec_flash_bench.py`（速度）。コマンドは [`docs/MTP-FLASH.md`](docs/MTP-FLASH.md) の「道具」節。

## その他のドキュメント

- [`docs/README.md`](docs/README.md) — ドキュメント全体の索引
- [`docs/SERVER.md`](docs/SERVER.md) — サーバーの起動・オプション・接続方法・制約
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — 公開インスタンスの運用（リバースプロキシ・`/health` 監視・ログの読み方・launchd 常駐）
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — この README の実測値がどのコマンド・どの結果 JSON から来ているか
- [`docs/MTP-FLASH.md`](docs/MTP-FLASH.md) — Flash-Next の MTP 投機デコード設計
- [`docs/BACKLOG.md`](docs/BACKLOG.md) — やりたいが未着手のもの
- [`docs/RELEASE.md`](docs/RELEASE.md) — 公開時にやること
