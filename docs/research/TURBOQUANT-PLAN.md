# TurboQuant (KV cache 3 bit) の実装計画 — **実装は確定** (ユーザー 2026-09-04 11:48)。Codex で実装する可能性が高いので、着手者が読めば足りる形で書く

## 何を作るか

Google の TurboQuant (ICLR 2026、学習不要の KV cache 量子化) を mlxturbo の KV cache に入れる。中身は 3 段:
1. **PolarQuant**: K / V の各ベクトルにランダム直交回転 (ガウス行列の QR、固定 seed、head_dim ごと) を掛けて座標をほぼ正規分布にし、座標ごとに Lloyd-Max のスカラー符号帳 (2〜3 bit) で量子化。ノルムは別に持つ。
2. **QJL (Quantized Johnson-Lindenstrauss)**: 量子化の残差に 1 bit の符号射影を掛けて、内積 (q·k) の偏りを補正する。
3. 保存形: index (3 bit)、QJL の符号 bit、スカラーのノルム。fp16 比 5〜6 倍の圧縮。
参考: 論文 (ICLR 2026)、llama.cpp の議論 (ggml-org/llama.cpp discussions/20969)、ollama PR 15090、解説 (analyticsvidhya 2026/04、towardsdatascience)。

## どこに入るか (mlxturbo)

- 第 1 段 (カーネル不要、先にやる): mlx-lm 組み込みの `QuantizedKVCache` (affine 4 / 8 bit、g64、`mlx_lm/models/cache.py:232`) + `quantized_scaled_dot_product_attention` (`base.py:64`) を Gemma 4 で有効化して測る (速度、KLD、長文脈の正答率、容量)。これが TurboQuant の取り分の上限の目安。
- 第 2 段 (本体): 新しい cache クラス `TurboQuantKVCache` (mlx-lm の `_BaseCache` の契約: `update_and_fetch` / `state` / `trim` / `to_quantized`、mlxturbo の `snapshot_untrimmable_caches` / checkpoint / tail と両立)。書き込み時に回転 + Lloyd-Max + QJL を MLX の op で (prefill 幅は素の op でよい)。
- **速度の取り分は読み側**: packed 3 bit の K/V を直接読んで q との内積 (QJL の補正込み) と softmax・V の和を出す **decode 用 attention カーネル (S ≤ 8、qmv 型、`mx.fast.metal_kernel`)**。`mlxturbo/kernels/qsa_attn_decode.py` (K2b、選択ブロックの attention) と `prefill_attn.py` (uint4 の段階 load) が手本。prefill 幅は dequantize して素の sdpa (取り分は容量だけ)。
- 対象の族: Gemma 4 (full attention 5 層、Hk 8、head_dim 256) が最初。Flash-Next / 27B は GDN 混成で KV が小さいので後 (50k で 0.6 / 1.6 GB)。sliding window 層は 1024 トークンで頭打ちなので対象外でよい。

## ゲート (CLAUDE.md の作法)

- 数値: 素の bf16 KV に対する KLD (`bench/quant_eval.py compare` の型、長文脈は `tools/longctx_quality.py` の正答率)。受け入れ幅は品質規則 (+0.0005) に準じるが、KV 量子化は「代金あり」なので取り分 (容量 / 速度) との釣り合いをユーザーが判断。
- 速度: 冷 micro (KV を 100 MB 超巡回) で decode attention の us と GB/s、in-model は 1 プロセス A/B (`decode_ab_generic` に Gemma 4 用の経路が要る = drafter エンジンの後)。
- 決定性: 回転行列は seed 固定で保存 (checkpoint / tail の復元と両立)。

## 順序

Gemma レーン: drafter エンジン (`gemma4_assistant`) → norm の本数削減 (331 本/step) → KV 量子化 第 1 段 → **TurboQuant 第 2 段 (確定)**。
