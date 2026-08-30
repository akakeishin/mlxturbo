# INTEGRATIONS — どのツールから繋がるか (2026-08-30)

GUI は作らない。代わりに、既に使われているツールから繋がることに寄せる。
この文書は、いま何が通っていて、何が足りないかの記録。

## 判断

**GUI を作らない理由。**GUI が要る層は LM Studio と MLX Core を既に使っていて、
そこは製品としての完成度の勝負になる。このプロジェクトの持ち物 (分布保証と
品質の実測) と何も関係しない領域で、いちばん人手が要る。

代わりに、**エージェント CLI とエディタ拡張から直接繋がること**に投資する。
プロトコル互換は模倣が容易で単独では堀が浅いが、GUI と違って作った分が確実に
連携先の数になる。

## いまの状態

| 口 | 状態 | 使う側 |
|---|---|---|
| OpenAI `/v1/chat/completions` | あり | 大半のツール |
| Anthropic `/v1/messages` | あり | Claude Code |
| OpenAI Responses `/v1/responses` | あり | Codex CLI (`wire_api = "chat"` は 2026-02-01 に削除済み) |
| **Ollama API** | **無し** | Open WebUI, Continue, その他多数 |

接続手順は `docs/SERVER.md` にある。opencode / Codex CLI / Claude Code / Chatbox
について書いてあるが、**書いてあることと通っていることは別**なので、E2E を
通した記録を残す形に変えていく。

## やること (効く順)

### 1. Ollama API 互換

一番多くのツールが喋る口。1 つ足すと連携先が一気に増える種類のもの。
競合の [mlx-serve](https://github.com/ddalcu/mlx-serve) は OpenAI / Anthropic /
Ollama の 3 つを同一ポートで出している。

必要なのは `/api/tags`、`/api/show`、`/api/chat`、`/api/generate`、`/api/version`
あたり。ストリームは NDJSON で、OpenAI の SSE とは形が違う。

### 2. `mlxturbo launch <agent>`

設定を書いてエージェント CLI を起動するところまで面倒を見る。手順を文書で
読ませるより、1 コマンドで通る方が使われる。mlx-serve は claude / pi / omp /
opencode / codex / hermes / aider に対応している。

こちらが先に押さえるべきなのは Claude Code と Codex CLI。Anthropic Messages と
Responses を両方話せることが、そのまま接続可能性になっている。

### 3. 通したことの記録

「対応」と書くのは簡単で、実際に通したかは別。ツールごとに、どのバージョンで、
何を確かめたか (ツール呼び出し、thinking の分離、長い文脈、中断) を残す。
これは計測の作法と同じ話で、このプロジェクトが得意な形。

## やらないこと

- **GUI** — 上のとおり
- **VS Code 拡張の自作** — Continue や Cline が既にあり、Ollama 互換か
  OpenAI 互換で繋がる。口を用意する方が安い
