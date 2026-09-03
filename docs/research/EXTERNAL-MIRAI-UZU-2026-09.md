# Mirai Labs / uzu (2026-09-04、ユーザー「めっちゃ速そうで脅威」の分析)

出典: https://trymirai.com/local-models/alibaba-qwen3-6-27b-mirai-mirai-m-4 (ベンチ 2026-09-01)、https://trymirai.com/blog/quantization、
https://github.com/trymirai/uzu (MIT、Rust + Metal、独自形式 `lalamo` で変換)、https://trymirai.com/inference-runtime。

## 数字 (Qwen 3.6 27B、Mirai-M 4bit 14.5 GB、**Apple M5 Max 128GB**、uzu 0.5.22)

| | decode | prefill | 備考 |
|---|---|---|---|
| uzu + specdec | **105 tok/s** (HumanEval 114 / MATH-500 117 / MT-Bench **84**) | 761 tok/s | 常駐 21.6 GB。specdec は「<50 MB の小さい draft model」 |
| MTPLX | 55 | **886** | 同じ機体、MTP 頭 |
| MLX | 26 | 541 | 素の mlx-lm |
| llama.cpp | 30 | 671 | |

- **機体が違う**: M5 Max。うちの見込み (IDEAS 9/3 11:45) は M5 Max で帯域 1.5 倍、NAX で行列積 3〜4 倍。M3 Max の数字と直接は比べられない。
- 27B 4bit の重み 14.5 GB を M5 Max の帯域 (550〜600 GB/s 級と推測) で読むと素の decode の壁は 38〜42 tok/s。MLX の 26 はその 65%、uzu の 105 は **specdec で 1 forward あたり 3〜4 トークン**を取っている計算 (MT-Bench で 84 = 一般チャットでは受理が落ちる、とページ自身が書いている)。
- prefill は MTPLX (886) のほうが速い。uzu の強みは decode 側 = draft の質。

## 技術

- **量子化 (Mirai-M)**: 4bit 非対称 (zero point 4bit、bf16 scale、g64) + ブロック対角 Random Hadamard Transform (block 32 = warp 幅、RMSNorm と GEMM/GEMV の prologue/epilogue に融合) + **YAQA の PTQ → teacher の rollout で量子化認識蒸留 (QAD、Muon + STE)**。品質は「Q4_K_M 相当で 20% 小さい」「同品質で 40〜60% 速い」。2B の MMLU-Pro 57.2%、KL 0.041。
  → **同じ重みではない**。蒸留した独自チェックポイントなので、同重量級の比較はできても「同じモデル」の比較ではない。うちの物差し (bf16 teacher との KLD) では別の点。
- **specdec**: 「draft model が先を予測し、本体が 1 pass で検証」「<50 MB」。README には仕組みの詳細無し (n-gram 系の小モデルを族ごとに学習している、という検索スニペットあり。推測)。MTP 頭 (1 層) より長い draft を高い受理率で出しているのが 105 の正体。
- エンジンは Rust + Metal の自前カーネル (MLX ではない)。iOS / iPad / macOS、Android は「Soon」。対応族は Qwen / LFM / Muse-Glimmer。

## うちへの含意

1. **物理では脅威ではない**: 同じ機体・同じ重みなら素の decode の壁は同じ。差は (a) 機体 (M5)、(b) draft の質 (学習した小モデル)、(c) 蒸留した量子化 (品質の点が違う)。
2. **方針で差がつく**: うちは「MTP 頭は学習しない、公式の頭が無い族に投機は付けない、品質を売らない (bf16 比 KLD +0.0005)」。uzu はその全部の逆 (draft を学習、独自量子化を蒸留)。この方針のままなら 27B の decode は MTPLX 級 (MTP 1 層で 2 倍) が上限で、uzu の 4 倍には届かない。
3. **製品としては脅威**: iOS まで同じエンジン、独自量子化で速度と品質の曲線を自分で動かせる、MIT。うちの差別化 (同じ重みでの品質つき計測、多ターン / 長文脈のサーバー) は残るが、「速い」の看板では負ける。
4. **27B レーンでやること**: 同じ重み (mlx 4bit) で mlx-lm / MTPLX (`tools/compare/mtplx-venv/`、2.9.2) / mlx-serve / oMLX / mlxturbo を M3 Max で並べる。uzu は別重みなので「参考点」として、可能なら同じ M3 Max で Qwen3.6-27B Mirai-M を 1 回流して置く (Qwen3.8 の Mirai-M は無い)。
5. **方針を変えるなら**: draft の学習 (<50 MB の小モデル、蒸留) を許すのが一番効く (推定 1.5〜2 倍、MTP 1 層の上に)。次が低ビット + Hadamard の量子化 (品質の点を動かす)。どちらもユーザーの判断。
