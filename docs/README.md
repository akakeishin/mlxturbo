# docs/ 索引

このディレクトリは大きく2つに分かれる。

- **ルート直下** — 利用者向け。サーバーの使い方・機能の設計・やりたいことの一覧。
- **`research/`** — 開発の経緯。実験ログ、カーネル最適化の調査記録、セッション間の申し送り。実装の「なぜ」を追いたいとき用で、利用には不要。

## ルート直下（利用者向け）

| ファイル | 内容 |
|---|---|
| [`SERVER.md`](SERVER.md) | `mlxturbo-serve` の起動・オプション・API キー認証・クライアント接続例（opencode / Codex CLI / Claude Code / Chatbox）・既知の制約 |
| [`MTP-FLASH.md`](MTP-FLASH.md) | Qwen3.8-Flash-Next の MTP 投機デコード設計（`flash_spec` 経路の中身） |
| [`BACKLOG.md`](BACKLOG.md) | やりたいが未着手のもの、根拠つき。決着済みの項目は末尾にまとめてある |

## `research/`（開発ログ・調査記録）

読む必要はないが、実装判断の根拠を遡りたいときに参照する。時系列・トピックが入り混じっているため、まとまった索引は無い。主なもの:

| ファイル | 内容 |
|---|---|
| [`research/ROOFLINE-2026-08-26.md`](research/ROOFLINE-2026-08-26.md) | 旧 README のアーカイブ。ハードウェアのルーフライン分析・m カーブの実測 |
| [`research/STATUS.md`](research/STATUS.md) | 実装状況のスナップショット |
| [`research/PLAN.md`](research/PLAN.md) | 実装計画書（初期フェーズの設計判断） |
| [`research/RESEARCH.md`](research/RESEARCH.md) | 投機デコード研究ダイジェスト |
| [`research/KERNEL-BRIEF.md`](research/KERNEL-BRIEF.md) / [`research/KERNEL-HANDOFF.md`](research/KERNEL-HANDOFF.md) | カーネル専門セッションへの引き継ぎと申し送り |
| [`research/KERNEL-BRIEF-HC.md`](research/KERNEL-BRIEF-HC.md) / [`research/KERNEL-HANDOFF-HC.md`](research/KERNEL-HANDOFF-HC.md) | hyper-connections 融合カーネルの引き継ぎと結果 |
| [`research/KERNEL-BRIEF-MOE-GDN.md`](research/KERNEL-BRIEF-MOE-GDN.md) | MoE ルーティング / GDN カーネルの引き継ぎ |
| [`research/KERNEL-INTEL.md`](research/KERNEL-INTEL.md) | カーネル設計インテリジェンス（一次ソース集） |
| [`research/ISA-NOTES.md`](research/ISA-NOTES.md) / [`research/ISA-DIFF.md`](research/ISA-DIFF.md) / [`research/ISA-QUEUE.md`](research/ISA-QUEUE.md) | Metal ISA 解析基盤・命令レベル差分・実行キュー |
| [`research/GATE-RESULTS-A2.md`](research/GATE-RESULTS-A2.md) / [`research/HYPOTHESES-A2.md`](research/HYPOTHESES-A2.md) | A2 カーネルの gate 実行結果と犯人候補の検討 |
| [`research/ARCH-BETS.md`](research/ARCH-BETS.md) / [`research/BRIDGE-NOTES.md`](research/BRIDGE-NOTES.md) | mlx-lm の外へ出る設計の賭けと bridge 実装ノート |
| [`research/OFFLOAD-RESEARCH.md`](research/OFFLOAD-RESEARCH.md) / [`research/OFFLOAD-DESIGN-SOL.md`](research/OFFLOAD-DESIGN-SOL.md) | SSD offload 調査と設計案 |
| [`research/COMPARE-QUEUE.md`](research/COMPARE-QUEUE.md) | 他エンジンとの比較実行キュー |
| [`research/BAKE-PLAN.md`](research/BAKE-PLAN.md) / [`research/BAKE-RESULTS.md`](research/BAKE-RESULTS.md) | 量子化レシピを焼く計画と実測 |
| [`research/D2-BRIEF.md`](research/D2-BRIEF.md) / [`research/D2-RESULTS.md`](research/D2-RESULTS.md) | D2（MTP ヘッド微調整）セッションのキックオフと結果。学習は中止済み |
| [`research/PREFILL-CHUNKING-DETERMINISM.md`](research/PREFILL-CHUNKING-DETERMINISM.md) | prefill チャンク分割の決定性の範囲 |
| [`research/REVIEW-2026-08-26.md`](research/REVIEW-2026-08-26.md) | 外部レビューの指摘一覧 |

## 参照が壊れていないか

`research/` 配下のファイル同士の相互リンクは `docs/research/` 起点に付け直し済み。ルート直下の3ファイルからの参照はそのままで有効（移動していないため）。ソースコード中のコメントが `docs/PLAN.md` のように移動前のパスで `docs/XXX.md` を指していることがあるが、これは実行時に読まれるパスではなく、記録として書かれた時点のプレフィックスがそのまま残っているだけ — 実体は `docs/research/` 配下にある。
