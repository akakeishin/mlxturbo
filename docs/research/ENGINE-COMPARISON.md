# mlx-serve / MTPLX / mlxturbo (2026-08-30 実測)

同じ機械 (M3 Max 128GB、macOS 26.4) で、**同じ重み**を 3 つのエンジンに載せて比べた記録。
バージョンは mlx-serve 26.8.11 (MLX Core.app 同梱)、MTPLX 2.10.0 (PyPI)、mlxturbo は手元の HEAD。

## 速度 (同じ重み・同じプロンプト・128 トークン)

重みは `qwen38fn-serve-b` (エキスパート 2bit、末尾 16 層のみ 4bit、他 4bit、lm_head 8bit)。

| エンジン | 投機 | decode |
|---|---|---|
| mlx-serve | **配線されていない** | **52.1 tok/s** |
| mlxturbo | MTP depth 3 | 39.3 tok/s |
| MTPLX | (未測定) | — |

**投機を切っている相手に、投機込みで 1.33 倍負けている。**カーネルの差。

参照の一律 4bit パック (`ddalcu/...-Serve-4bit`) では mlx-serve が 50.4 tok/s。
`--mtp` を付けても 50.6 -> 50.4 で動かない。ログが理由を言っている:

    [qwen4] MTP head loaded (1 hyper-connected QSA+MoE layer; spec wiring pending)

## 投機の対応状況

| | Flash-Next (qwen4_exp) の MTP |
|---|---|
| mlx-serve | **未配線。**ヘッドは読むが投機しない。仕組み自体は `src/mtp.zig` 4522 行で、27B や DeepSeek-V4 では動いている |
| MTPLX | **対応済み** (2026-08-29、youssofal/MTPLX#380)。`--depth`、`--generation-mode {mtp,ar,auto}`、EV による適応 depth |
| mlxturbo | 対応済み。priming (末尾 2048 対で受理率 0.574 -> 0.827)、文脈適応 depth (6144 で 1 に落とす) |

## 機能

| | mlx-serve | MTPLX | mlxturbo |
|---|---|---|---|
| ライセンス | MIT | Apache-2.0 | — |
| OpenAI 互換 | ○ | ○ | ○ |
| Anthropic (`/v1/messages`) | ○ | ○ | ○ |
| Ollama (`/api/chat`) | ○ | — | — |
| embeddings | ○ | — | ○ |
| vision | **○** (別ファイルの bf16 タワー) | 記載なし | ✕ |
| tool calling | ○ | ○ | ○ |
| 継続バッチング | ○ | ○ (scheduler-mode 5 種) | △ (非投機のみ) |
| prefix キャッシュ | ○ (hot cache、SSD 退避) | ○ (SSD session cache) | ○ (セッション再利用) |
| KV 量子化 | ○ (`--decode-attn-quant`、既定 ON) | ○ (`--paged-kv-quantization`) | ✕ |
| ANE prefill | ○ (lossy) | — | ✕ (測って不採用) |
| 分布の厳密性 | lossy を既定 ON | **`qa` に exactness / distribution gates** | Block Verification (要件から外した) |
| GUI | ○ (MLX Core.app) | ○ (Mac アプリ + ダッシュボード) | △ (メニューバー、未整備) |
| モデル取得 | ○ (`pull` / `run` / 短縮名) | ○ (`pull` / 推奨モデル自動選択) | △ (`hub` CLI) |
| 自前で MTP を作る | — | **○ (`forge`)** | ✕ |
| 量子化レシピの道具 | `tests/qwen38_iq_*` (27B 用、imatrix 配分) | — | ○ (`tools/bake.py`、KLD 計測) |
| 対応アーキ | 広い (Qwen/DeepSeek/GLM/Llama/画像/音声/3D まで) | **MTP 持ちに特化して 25 種** | Flash-Next のみ |

## ここから言えること

**「速度と品質の両方を計測で示せる唯一の実装」は成り立たない。**MTPLX は
`qa` に exactness と distribution のゲートを持ち、AIME のベンチランナーも積んでいる。
mlx-serve は逆に lossy を既定 ON にしているが、それは選択であって計測が無いのとは違う。

**エンジンで勝つ線は薄い。**mlx-serve のカーネルは、投機を切った状態でこちらの
投機込みより 1.33 倍速い。MTPLX は Flash-Next の MTP を昨日入れて、いまも毎日動いている。

残るのは**パック**。mlx-serve ネイティブで読める形に焼く道は通っていて
(docs/research/MLX-SERVE-PACK-FORMAT.md)、一律 4bit より 28% 小さくて同速のものが
手元にある。相手の imatrix 配分は 27B 用で、**Flash-Next 向けには公開されていない。**
