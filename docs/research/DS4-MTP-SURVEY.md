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
