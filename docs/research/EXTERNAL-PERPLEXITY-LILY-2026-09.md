# 外部の知見: Perplexity "Lily" (Apple silicon 向け自前エンジン、2026-09-01 公開) の読解と、うちへの対応表

出典: https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon (2026-09-03 に読んだ)。
対象: Qwen3.6-35B-A3B (256 専門家 top-8 + 共有 1、attention 10 層 + Gated DeltaNet 30 層、GQA Q16 / KV2、head 256、Q4 g64 19.4 GB)、
M5 Max 40 コア GPU 128 GB、batch 1。Rust の runtime + 自前 Metal カーネル、MLX も PyTorch も実行経路に無い。OpenAI 互換 chat completions、streaming。デモは GitHub 公開。

## 数字 (対 MLX-LM の direct generation、同じ Q4 バイト、1 リクエストずつ、ラウンド内で両エンジンを交互)

- 256〜128K の 10 点平均: prefill 4,156 対 3,388 tok/s (**1.23x**、幅 1.12〜1.42)、decode 170.0 対 126.4 tok/s (**1.35x**、幅 1.31〜1.37)。
- 4K: prefill 5,749.9 対 4,737.5、decode **186.6** 対 140.9 tok/s。
- 数値: teacher-forced 192 位置で perplexity +0.04%、top-1 一致 96.35%。
- 限界の実測: MoE GEMM 97.9% / GEMV 90.3% (持続の重み読み出し率比)、prefill GEMM 単体 93%、モデル内 80〜86%。GEMV から演算を抜いても 0.2% しか変わらない (帯域律速)。

## 技術とうちの対応

| Lily | 効果 (ablation) | うち | 取り込み |
|---|---|---|---|
| grouped GEMM の中で Q4 を逆量子化 (bf16 を unified memory に作らない) | 512 tok で prefill +77% | MLX の gather_qmm / P3 segmented が既に同じ | 無し |
| MoE ルーティング (histogram / prefix scan / scatter / block map) を 1 command buffer に、層内で CPU 同期しない | 512 tok で +89% | P3 の表作りは同期なし。**P7 の内訳 (行列積以外 800 ms/チャンク) に隠れた同期が無いか見る** | P7 で確認 |
| タイルを専門家の負荷に合わせる (16 行 → 32 行 × 4 simdgroup) | 2K で +13.2% | mix48 (専門家ごと 16/32 行、WM=1) が既定。M5 (NAX) では simdgroup 数の最適が違う | NAX 機で WM を取り直す |
| GDN の再帰状態をレジスタ常駐で逐次 scan (列 = simdgroup、fp32、barrier 無し)。blockwise は 2K で 256 MiB/層 動かす | 2K で +5.6% | うちは **blockwise** (oMLX 移植の Metal) | **候補**: prefill スタブ (GDN scan を 0) の天井 ≥ 3% なら register 常駐 scan の PoL |
| チャンク prefill | — | 同じ (2048、グループ化) | 無し |
| 行並列 GEMV | — | MLX の qmv | 無し |
| トークンの受け渡しを GPU に残す (command buffer 2 本、GPU 上の入力スロット) | — | 投機の受理判定が CPU (段階投入で隠している) | 小 (D4 で余地なしと出ている) |
| 依存を記録した concurrent Metal pass で独立カーネルを重ねる (1 step 795 カーネル / 555 段) | — | MLX の 1 stream では出来ない (P1 two-stream は畳んだ) | **構造的な差**。MLX 内では取れない |
| 4 本の融合 (gate/up + 活性、down + ルータ + 共有専門家、q/k 準備、再帰更新 + norm) | — | MOE_GLU / GDN prework / HC は in-model で負け (冷 DRAM で並列度不足) | 自前カーネルを帯域最適に書けば勝てる証拠。書き直すなら Lily の作りを参照 |
| KV 読み出しの coalescing | 3,840 で decode +2.1% | MLX の sdpa_vector | 無し |
| GQA packing (4 head で KV 行を共有) | 32K で **+23.8%** | Flash-Next は QSA で kv を 2048 に絞る (K2b)。dense の 27B / 35B では MLX の sdpa_vector の gqa 共有に依存 | 27B レーンで MLX の挙動を確認 |
| 32K 以上で固定ブロック attention に切替 | 32K +7.7% / 64K +27.4% / 128K +40.2% | Flash-Next は QSA。dense の族では effective | 27B レーンの候補 |
| **投機 decode は batch 1 で 18% 遅い** (2〜5 行の verify が非効率、行ごとに違う専門家を読む) | — | うちは MTP 頭 (別モデルの drafter ではない) で +25% 程度。**35B-A3B では AR 対 MTP を先に測る** | 35B-A3B の入口で必ず |

## うちのゴール (短文脈 100 tok/s) への含意

Lily の decode 186.6 tok/s (M5 Max、active 3B、Q4) は重み読み出しの壁 (90%) に張り付いている。うちの Flash-Next は S=1 forward 24 ms で、
active 重みの読み出し (~2 GB、400 GB/s で 5 ms) の **約 5 倍** = 壁から遠い。差は演算ではなくカーネル数 (1 step に数百) の起動と直列化。
MLX の op 単位の投入では取れない部分で、取るには (a) 多くの op を 1 カーネルに畳む自前カーネル (帯域最適に書く)、(b) MLX の外で decode ループを組む、のどちらか。
(a) は decode 幅で 4 回負けているが、負けた理由 (並列度不足) は Lily の作りで解ける類。判断材料: **1 step のカーネル数と GPU の空き率を測る** (decode の Metal trace)。

## 機能面

Lily は単一モデル・batch 1・chat completions + streaming のみ。継続バッチ、prefix cache / 温 TTFT、thinking 分離、tool 呼び出し、構造化出力、
マルチモーダルは無い。機能で取り込むものは無い。製品としては cloud + local の Hybrid Compute の部品。
計測の作法で 1 つ: 両エンジンをラウンド内で交互に走らせる (熱の偏りを消す)。Flash-Next は 98 GB × 2 が載らないので出来ないが、27B (15 GB) なら出来る。

## 参照する価値のあるもの

- GitHub のデモの Metal ソース: Metal 4 tensor op (= NAX) を使う grouped GEMM、GQA packing、固定ブロック attention、register 常駐の GDN scan。NAX 機の経路と 27B / 35B レーンの参考実装。
