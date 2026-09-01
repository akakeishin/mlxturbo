# 横展開先のボトルネック予報 (GLM / DeepSeek / Gemma)

2026-09-01。実機ロード無しの事前調査 (scout 2 本: ローカル mlx_lm + HF config、
mlx-serve main ソースの先行知見)。Flash-Next レーンで確立したボトルネック目録
(KERNEL-BRIEF-DECODE-BW.md) に各ファミリーを照射する。数字の出典は各 scout
報告 (config.json とソース path:line で裏取り済み)。

## 早見表

| | GLM-5.3-Flash | DeepSeek V4 Flash | Gemma 4 26B-A4B |
|---|---|---|---|
| mlx_lm 対応 | **無い** (glm5_next) | **無い** (deepseek_v4) | ある (gemma4) |
| 投機の形 | MTP ヘッド内蔵 | MTP ヘッド内蔵 + DSpark | **別配布の assistant drafter** (MTP ヘッドは無い) |
| アーキの近さ | **qwen4_exp とほぼ同族** (KDA 線形注意 4:1 + 疎注意 + MoE 288/top8) | MLA 風 (kv_heads=1, head_dim 512) + DSA indexer + MoE 256/top6 | sliding window 5:1 + MoE 128/top8 |
| prefill MoE の r (2048 幅) | 56.9 → 効率 ~72% 帯、layer-major がそのまま効く | 48.0 → 同上 (moe_int 2048 で per-expert が太く素の効率は良い) | 128 → 効率 ~87%、MoE はほぼ問題にならない |
| mlx-serve の対応 | **無い (空白地帯)** | あり、強い (DSpark 込み 35 tok/s、GGUF 比 1.3x) | あり (drafter 2 倍、SWA prefill カーネル 2.4x) |
| 128GB での現実性 | ~45 層 hidden 4096、収まる見込み | 43 層、GGUF 90.9GiB 実績あり (docs/research/DS4-MTP-SURVEY.md) | 4bit ~15GB、余裕 |

## ファミリー別の予報

### GLM-5.3-Flash — 最有力。うちのスタックがほぼ 1:1 で写る

- 構成が Flash-Next と同型: 線形注意 (KDA) と疎注意 (deepseek_sparse_attention)
  の 4:1 ハイブリッド、full_attn_layers 明示、MoE 288 experts/top8 (+shared)、
  MTP ヘッド (num_nextn_predict_layers 1)。capture/rollback (再帰状態の行別
  take)、layer-major prefill、段階投入、indexer の扱い、バッチ x MTP — 今ある
  資産と作成中の資産が全部運べる。
- 予報されるボトルネック: (1) prefill MoE r=56.9 → Flash-Next と同じ境界
  再計算問題、layer-major で同様に −10% 級が取れるはず。(2) KDA の再帰状態は
  GDN と別実装なので capture の写しを 1 本書く (規律: 写しは本家と対で管理)。
  (3) MLA 併用 (kv_lora_rank 512) — trim は offset 系で安全の見込みだが、
  batch.py の merge/padding は KVCache 形状前提なので要改修。
- 競合状況: mlx-serve に GLM ネイティブ実装は無い (scout 確認、ツールコール
  書式のみ)。**対応すれば空白地帯を先取り**。
- 非 Flash の GLM-5.3 (78 層 hidden 6144) はサイズ的に 128GB に入らない
  公算が大きく、対象は Flash のみ。glm_moe_dsa (GLM-5.3 無印) は mlx_lm に
  実装があるので、コードの参照元としては使える。

### DeepSeek V4 Flash — 事前実測が既にあり、depth 経済が渋い

- repo に実測済みの調査がある: docs/research/DS4-MTP-SURVEY.md (GGUF 実機)。
  幅 2 で受理 68%・収穫 1.68、**verify の位置追加費用が +14-16ms/位置で線形**、
  routed の重なり 0.421。→ depth は 1-2 が上限で、Flash-Next より投機の
  伸びしろが細い。バッチ x MTP の紙モデルでも P を浅く見るべき族。
- mlx_lm に deepseek_v4.py が無く、アーキ移植 (MLA 風 + DSA indexer +
  compress_ratios の層別圧縮) が必要。移植量は qwen4_exp vendor 級。
- mlx-serve は DSpark (ブロック並列ドラフト、Markov head) 込みで強い。
  彼らの rollback 実装には「巻き戻し後の再確保 ~110ms タックス」という注記が
  あり (deepseek_v4.zig:8079-8088)、MLA キャッシュ + 投機 rollback を
  移植する際はこのコストを最初から測ること。
- 彼らが踏んだ罠 (受理率を静かに損なう MTP ヘッドの head-norm 規約ミス、
  MTP ヘッド量子化でのクラッシュ) は、うちの --mtp-bits / 契約判定に
  検査として輸入する価値がある。

### Gemma 4 — 「MTP 版」の正体は別配布ドラフトで、機構が別物

- base config に MTP フィールドは無い。投機は google 配布の
  **gemma4_assistant** (4 層 cross-attention、backbone の KV を直接読む
  kv_shared_only、centroid 機構、arxiv 2607.02770)。DeepSeek/GLM 式の
  「同一 forward 内 nextn ヘッド」とは別方式で、mlxturbo 側は
  FlashSpecEngine ではなく **KV 共有型 drafter エンジンの新設**になる
  (DraftSpecRunner とも違う: drafter が target KV を読むため
  キャッシュの受け渡しが要る)。
- 予報されるボトルネック: (1) sliding window 5:1 — mlx_lm は
  RotatingKVCache で、mlx-serve はここに専用 prefill カーネルを書いて
  26B-A4B の prefill を 2.4 倍にした実績がある。つまり素の MLX の SWA
  prefill には大きい伸びしろが実在する。(2) RotatingKVCache の回転は
  「全行一律前進 + dead slot」というバッチ台帳の前提を壊すので、
  バッチ対応は full 層と sliding 層で台帳を分ける必要がある。
  (3) MoE は r=128 で効率が既に高く、主戦場ではない。
- 4bit ~15GB で 128GB 機に余裕で載る = 想定ユーザー層が一番広い。

## 共通して運べるもの / 運べないもの

- 運べる: 段階投入 (staged)、BPE 境界 checkpoint、warm TTFT 系、layer-major
  prefill (MoE 持ちのみ)、共有タイル gather の設計 (バッチ設計点)、
  n-gram/lookup 投機、計測規律一式。
- 運べない: qwen4_exp 特化カーネル (HC 融合、GDN カーネル)、
  Flash-Next の MTP 契約判定。
- 順序の私見 (確定は着手時に advisor を通す): GLM-5.3-Flash (同族 + 空白地帯)
  → Gemma 26B-A4B (ユーザー層 + SWA カーネルの伸びしろ) → DeepSeek V4 Flash
  (移植量と depth 経済の渋さで最後)。

## 未確認 (着手時に潰す)

- glm5_next の safetensors キー名と MTP ヘッド重みの実構造 (config のみ確認)
- DS4-MTP-SURVEY の重なり 0.421 の計上単位 (routed 12/256 と config の
  top6 の食い違い — GGUF 版との設定差の可能性)
- gemma4_assistant の推論時の hidden state 共有の正確な形 (config からは断定不能)
