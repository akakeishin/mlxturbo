# ds4 (DwarfStar) の MTP を読んだ記録 (2026-08-30)

`antirez/ds4` は mlx-serve が `lib/ds4` に submodule で取り込んでいる C + Metal の
推論エンジン (MIT)。**mlx-serve はエンジンを 2 つ持っている。**

| 経路 | 形式 | 対応 |
|---|---|---|
| MLX | MLX affine safetensors | Qwen 系、DeepSeek-V4 |
| ds4 | **GGUF の K-quant** | DeepSeek V4 Flash / PRO、GLM 5.2、(上流で 5.3 Flash) |

ds4 に Qwen は**一切無い** (`ds4.c` 64,525 行と README で "qwen" が 0 件)。
README いわく "deliberately narrow, not a general GGUF runner"。

## 投機が効いていない、と作者自身が書いている

> The current MTP/speculative decoding path is still experimental: it is
> correctness-gated and currently provides at most a slight speedup, not a
> meaningful generation-speed win.

## 理由は 3 つ揃っている (コードで確認)

**1. ドラフトキャッシュがプロンプトを見ない。**

DeepSeek Flash 側は専用の `mtp_raw_cache` を持つが、行数 `mtp_n_raw` が増えるのは
`metal_graph_eval_mtp_draft_from_hc` の中の 1 箇所だけ (ds4.c:32332)。
つまり**ドラフトを引いたときにしか増えない**。prefill で流し込む経路が無い。
他の 8 箇所は全部 `= 0` のリセット。

GLM 側は別実装 (`glm_graph_mtp_step`) だが同じで、コメントに明記がある:

    It keeps a private compact KV cache (slot = absolute position; only
    positions >= mtp_min_pos are ever selected, so the unwritten prompt
    range is never read).

`glm_mtp_min_pos` は最初の decode 位置で初期化される (ds4.c:56986)。

**2. depth の既定が 1。**`ds4_cli.c:1762` の `.mtp_draft_tokens = 1`。
1 ラウンドで最大 2 トークンなので、受理率が 100% でも上限が 2 倍。
README の例も `--mtp-draft 2`。

**3. margin ゲートで投機を絞っている。**`.mtp_margin = 3.0f` が既定で、
ドラフトの top-1 と top-2 の logit 差がこれ未満なら幅 2 の検証をやめて普通の
1 トークン decode に落ちる (ds4.c:63989)。`--quality` (`strict_mtp`) で外れる。

3 は 1 と 2 の**結果**だと読める。受理率が低いから、無駄な検証を避けるゲートを
置いた。嫌っているのではなく適応している。

## うちが持っているもの

同じ形のヘッド (Flash-Next の MTP) で実測済み:

- プロンプト末尾 2048 対で温めると受理率 **0.574 -> 0.827**、decode +18.1%、
  TTFT +1.0% (32k プロンプト)
- depth 3 は文脈 6k までは効き、8k で反転する。1k で 45.4 -> 51.5 tok/s
- ドラフトキャッシュに投機を入れない不変条件が、ラウンドを跨いで持ち回れる根拠

ds4 側には `raw_window` という窓の構造が既にあるので、priming は設計に沿って
足せる。観測も `DS4_MTP_TIMING` / `DS4_MTP_CONF_LOG` / `--glm-mtp-timing` が
既にあり、**margin の分布をそのまま測れる。**

## ただし混んでいる

`antirez/ds4` の issue:

- #867 Support for QWEN3.8-Flash-Next (OPEN、コメント 12)
- #870 Qwen 3.8 Flash (OPEN)
- #462 Add Qwen3.5/Qwen3.5 MoE (OPEN)

#867 では **Baekpica が fork で qwen4_exp を実装済み、MTP も動いている**
(ドラフト棄却時に PLE の conv 窓と GDN の再帰テンソルをスナップショット、
QSA は append-only なので長さで復元)。francosax が ROCm 向けに並行して移植中。

**着手する前に issue で先行者を確認すること。**Flash-Next は 2026-08 に出た
ばかりで、いま全員が「動かす」仕事をしている。

## 実測 (2026-08-30、M3 Max 128GB、DeepSeek-V4-Flash 90.9GiB GGUF)

`--mtp-draft` を掃引した。`DS4_MTP_MIN_MARGIN=0` で margin ゲートは外してある。
検証時間と収穫は ds4 自身のタイマー (`DS4_MTP_TIMING=1`) から。

| 幅 | 検証 | 1 ラウンドの収穫 | 受理率 | ms/トークン | generation |
|---|---|---|---|---|---|
| 1 (投機なし) | 37.0 ms | 1.00 | — | 37.0 | **27.05 t/s** |
| 2 | 59.3 ms | 1.68 | 68.0% | 35.3 | 26.32 t/s |
| 3 | 75.1 ms | 1.95 | 47.5% | 38.5 | 19.05 t/s |
| 4 | 89.3 ms | 2.10 | 37.4% | 42.5 | 16.62 t/s |
| 6 | 110.4 ms | 2.05 | 24.6% | 53.9 | 14.21 t/s |

**検証は位置あたり +14〜16 ms で線形。**固定費ではなく位置ごとの仕事が支配的で、
「深くすれば償却する」形をしていない。密なモデルなら k 位置の検証は重みを 1 回
読むだけで済むが、**256 中 6 個しか使わない疎な MoE では位置ごとに別のエキスパートが
起きる**ので、読み出しが位置数に比例する。

**収穫は 2.1 で頭打ち。**受理率が 68% -> 47.5% -> 37.4% -> 24.6% と崩れるので、
5 段目 6 段目はほぼ当たらず費用だけ払う。幅 6 の収穫 (2.05) は幅 4 (2.10) より少ない。

### priming の伸びしろは小さい

当初「ドラフトキャッシュがプロンプトを見ていないから受理率が低い」と読んだが、
**受理率は幅 2 で 68% あり、低くない。**問題は検証が償却しないこと。

仮に priming で幅 2 の受理率を 68% -> 90% にできたとして、収穫 1.68 -> 1.90、
ms/トークン 35.3 -> 31.2 で +13%。ドラフト生成の 3.0 ms を引くと **+5% 前後**。

### 未解決

内部タイマー上は幅 2 が 35.3 ms/トークンで投機なしの 37.0 に勝っているのに、
実測の generation は 26.32 対 27.05 で負ける。ドラフト生成 3.0 ms を足すと
37.1 ms でほぼ拮抗するので、そこが説明になりそうだが確認していない。

## これは既知の現象だった (文献で確認、2026-08-30)

疎な MoE で投機の検証が償却しないことは、公表済み。arXiv の実在は abs ページで確認した。

- **arXiv 2506.20675** *Utility-Driven Speculative Decoding for Mixture-of-Experts*
  (MICRO 2025)。batch-1 で 5 つの MoE を測り、**検証コストが 1.5-3 倍**に伸びる。
  draft 長 7 でコード 2.3 倍・数学 3 倍。注意は基準遅延の 8% 程度で安定。
  **35 の (モデル x 課題) のうち 18 で投機が減速させた。**K=0 を動的に許すと
  ほとんどの退行を回避できる
- **arXiv 2607.12696** *Less Experts, Faster Decoding: Cost-Aware Speculative
  Decoding for Mixture-of-Experts*。DeepSeek-V3.1 / Qwen3-235B-A22B / GPT-OSS-120B。
  **batch-1 で素の MTP は平均 1.10 倍、エキスパート費用を考慮した選択で 1.15 倍**

うちの掃引から出した「完全受理でも +14.3% が上限」は、後者の 1.10-1.15 倍と一致する。

### エキスパートの和集合は割に合わない

重なり 0.421 から、削れるルーテッドエキスパートの読み出しは
`82.7 GiB x (12 - 9.474) / 256 = 0.816 GiB`。393 GB/s の上限で **2.2 ms**、
幅 1 のバイト模型が示す実効帯域で **3.0 ms**。

損益分岐に必要な削減は `64.7 - 1.68 x 36.97 = 2.59 ms`。**ぎりぎりの線**で、
最も甘い見積もり (遅延がバイト数に完全比例) でも 36.0 ms/トークン = **+2.7%**。

### 受理率をいくら上げても足りない

常時投機での損益分岐は受理率 **75.0%** (現状 68%)。ドラフト費用 3.0 ms を
完全に消しても `62/1.68 = 36.90 ms/token` でほぼ横ばい。84% で +5%、
85-90% で +6-9%、完全受理で 30.9 t/s = **+14.3%** が幅 2 の硬い天井。

### 訂正

このファイルの前半で「priming が無いから受理率が低い」と読んだが、**外れ**。
受理率は幅 2 で 68% あり低くない。また priming の 0.574 -> 0.827 は
**Flash-Next の別のヘッドの数字**で、DeepSeek-V4 の予測に使ってはいけない。
