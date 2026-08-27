# KERNEL-HANDOFF — カーネルセッションから親セッションへの申し送り

このファイルはカーネルセッションが更新する。新しい項目は上に足す。
詳細な根拠はそれぞれ docs/ISA-DIFF.md §9 と docs/BRIDGE-NOTES.md §5.1 にある。

## 2026-08-27

**公式レップ 4 回目 (再起動後) も無効。原因は他セッションとのマシン共有。**
`bench/results/compare-official2-rep1-invalid-shared.json`。事前条件は完璧
だった (再起動 1.5h、swap 0、AC 満充電、powermode 0、Chrome 全終了、
mlx-lm 素の事前検証 23.063/23.095 = 旧公式 23.249 の 99.2%、2 ラン差 0.14%)。
にもかかわらず、ラン 3-4 以降が全エンジン一律 **約 50%** に落ちた。

ハーネス内蔵の対照で確定する。mlx-lm 素は same-quant 行と recommended 行で
**同一コマンド**が走る。旧 rep1/2/3 では位置 1-3 と 10-12 が揃っていた
(rep3: 23.16/23.15/22.93 vs 23.26/23.30/23.24)。今回は
**23.08/22.05/17.29 vs 10.21/10.56/9.34** で 2.2 倍差。

| 位置 | 旧 rep3 (健全) | 今回 | 比 |
|---|---|---|---|
| 1 mlx-lm sq code | 23.16 | 23.08 | 1.00 |
| 3 mlx-lm sq edit | 22.93 | 17.29 | 0.75 |
| 4 fastmlx sq code | 37.5 | 17.5 | 0.47 |
| 10 mlx-lm rec code | 23.26 | 10.21 | 0.44 |
| 16 mtplx rec code | 44.18 | 22.03 | 0.50 |
| 18 mtplx rec edit | 50.96 | 27.21 | 0.53 |

除外できたもの: swap 増加なし (swapouts はランの前後とも 24776 で据え置き)、
サーマル警告なし、powermode はラン後も 0、低電力モードの再有効化なし。
熱ソーク (旧 invalid-thermal) の単調劣化とも形が違い、ラン 1 が満点のまま
ラン 4 で 50% の台地に入り、そこで平らになる。**電力キャップでも熱でもなく、
GPU を分け合った形。**

外因である裏付け: レップ終了 (17:04:43) 後に同じ計測を打つと、外付け SSD への
92GB rsync が走っている状態でも **21.4 tok/s** まで戻る。恒久的な劣化ではない。

共有元は親セッション (`858042c7-…`)。scratchpad の mtime が
15:52-15:59 に probe_ple.py / verify_weights.py / probe_layers.py、
**16:59:03 に evacuate.sh** (私のレップは 16:47:56-17:04:43)。
17:04 から ~/models/qwen38fn-mlx-v0-95 (92GB) を SSD へ rsync 中。

**この回のギャップ値は読まないこと。** 一律に見えても劣化率はエンジンごとに
0.44-0.54 とばらつき、バースト型 (fastmlx) と連続ストリーミング型 (mlx-lm)
で効き方が違う。これは 3 回目 (invalid-throttle) を無効にした歪みそのもの。

**次レップの条件 (これまでの 4 回で判明した全部):**
1. 再起動直後、swap 0、AC・満充電、`pmset -g` の powermode が **0**
2. Chrome を全終了 (実測 −7%: 21.5 → 23.06)
3. `dasd` が暴れていないこと (旧公式は「dasd のみ常駐」で健全だったので
   常駐自体は可、1 コア食いつぶしは不可)
4. **他セッションが GPU / 大量 I/O を走らせていないこと。今回の唯一の敗因。
   レップ中 (約 17 分) は他セッションを止める必要がある。**
5. 事前検証: 公式同一 argv の mlx-lm 素が 23.2 付近 × 2 ラン一致

計測上の注意: バッテリー駆動では GPU クロックが電力管理で大きく揺れ、
lm_head 級の帯域依存形状は同一条件でもラン間 10 倍振れることを実測した
(サーマル警告なしでも起きる)。バッテリー下の計測は 2 ラン一致フィルタを
通ったもののみ有効。lm_head m=1 の帯域調査 (sol 優先順位 3 位) は AC 待ち。

AC 復帰後の確定 (2026-08-27 昼):
- **lm_head m=1 のカーネル改善余地は無い (A 項は却下)。** AC・静音で素の
  qmm (5120→248320, m=1) は 2.11-2.16ms @ 331-340GB/s = streaming 天井の
  96-99%。「260GB/s」は decompose の slice 込みグラフとバッテリーノイズの
  合成だった (sol の指摘どおり)。entropy+softmax を足しても +0.05ms。
  → draft 側の実レバーは「捨てリンクを作らない」(項目 5 の逐次ゲート化、
  1 本スキップ = 2.15ms+) に一本化される。
- B1 (tools/bridge) は AC 追認済みで正式クローズ (BRIDGE-NOTES §5.1)。
- compare_engines official2 rep1 は**無効** (bench/results/
  compare-official2-rep1-invalid-thermal.json に改名して保存)。2 つの障害:
  (1) **充電中 (83%→) + GPU 持続負荷の熱ソーク**でマシン全体が単調劣化し、
  fastmlx どころか無改変の mlx-lm 素まで 21→4 tok/s に崩壊した。ベンチ前の
  スポット計測は 3 巡安定でも、開始 ~10 分から進行する。pmset にサーマル
  警告は出ない。終了後 GPU は完全回復 (lm_head 2.05ms @348GB/s) を確認済み。
  **教訓: 公式レップは満充電付近・冷えた筐体で開始する。** 中間で mlx-lm の
  同一コマンドの数値が同水準かをラン間検証に使える (今回 same-quant 3 本目
  で既に 17.7 に落ちていた)。
  (2) mtplx が全ラン rc=6 即死 — ~/.mtplx/models が消えていて
  「model is not available locally」。旧 rep1 にも同じ rc=6 があった
  (旧 mtplx 値は rep2/3 由来)。2 モデルを mtplx pull で再取得中。
  冷却・満充電後に rep1 を取り直す。
- **経路表反転は in-model でも正と確認 (公式軌道の直接 A/B)。** 公式ハーネス
  と同一 argv (--no-think、code、512 tok、tok/step 4.44 の高受理軌道) で
  交互 2 ラウンド: 反転後経路表 30.1-30.6 tok/s vs 反転前 27.8-29.0
  (+4-8%)。低受理軌道 (thinking 有効、tok/step 2.34) でも新カーネルが
  +4-7%。D7 の有効/無効は差なし (同期コストは実測上見えない)。
- **公式レップ official2 は 3 回とも無効。** 3 回目は充電なし・満充電でも
  「持続帯域ワークロード選択的スロットリング」が発生: mlx-lm 素 (連続
  ストリーミング) が 17-20 tok/s に絞られる一方、バースト型の fastmlx は
  30.6 と旧公式水準を維持し、比率が歪む。12 日稼働 + swap 17GB + 電力管理
  の迷走が背景。**正式レップは再起動直後のクリーン状態で取り直す**
  (旧公式との比較可能性の最低条件として mlx-lm 素 ~23 tok/s を事前確認)。

### 親側で判断・作業が要るもの

1. **spec.py の同期点 (eval 回数) が最大の残り固定費。**
   bench_chain 初実測で、mx.eval 1 回の往復 (submit 切片) が ~170-190us と
   判明した。per-dispatch のラッパ税は 2.1us しかなく既に償却されている。
   検証 1 ステップ内で eval/同期が複数回あるなら、1 回減らすごとに ~0.2ms
   拾える。受理判定の CPU 往復などが候補 (BRIDGE-NOTES §5.1)。

2. **compare_engines の再計測待ち。** カーネル側で B ステージングのベクトル化
   (+1-3%) と経路表の全面反転 (nocap 奪取 1.06-1.38x、stock 奪取 M=12..15 で
   1.38-1.59x、gate.py baseline_all_identical: True) が入った (8e14bdf)。
   MTPLX recommended とのギャップ (code -10% / prose -13% / edit -31%) の
   更新値は公式プロトコル 1 レップで取り直す。バッテリー駆動で 1 回中断済み、
   AC 接続後に実行予定。

3. **経路較正の判定はチェーン基準に変えた。** 旧較正 (calib-quiet) が M=6/7 を
   nocap に振っていたのは、(a) 単発レイテンシ測定だったこと、(b) test_dispatch
   の "mma" 候補が v5 skinny のままで dispatch 実体 (fast_qmm) とズレていたこと
   の合成だった。(b) はユーザー承認の上で修正済み (bench/test_dispatch.py)。
   以後の較正は依存チェーン 2 ラン (calib-chain-a/b.json の手順) を推奨。

4. **実経路の phase 実測 (バッテリー駆動、相対値は有効)。** spec_bench
   (edit、k=3、mtp-bits 4、fast-qmm) で 32.1 tok/s vs stock 23.5 (1.37x)、
   ステップ 99ms = draft 24.2 + verify 74.4 + maint 0.3、tok/step 3.17。
   m=1 decode は 45.6ms @ 332GB/s でほぼ飽和 (bench/decompose.py)。
   気づき:
   - draft 24.2ms のうち積み上げで説明できるのは 12-14ms (lm_head m=1
     2.75ms x 3-4 本 + MTP 層 + entropy)。残り ~10ms は Python グラフ構築
     または同期の疑い。draft は親側 (spec.py) の構造に依存する部分が大きい。
   - lm_head (5120→248320) m=1 が 2.75ms @ 260GB/s と天井の 75% 止まり。
     ここはカーネル側で調査中 (sol に反証依頼中)。
   - GDN capture 税 (states_all 全書き出し) は層単体実測で T=7 1.69x、
     48 層換算 2.3ms/step、T=16 で 5.6ms。rollback が読むのは 1 スライス
     のみなので、「検証は素の fused カーネル + 受理長確定後に受理分だけ
     再走して状態を得る 2 パス」なら丸ごと消せる。spec.py の構造変更を
     伴うため親側判断。カーネル側は既存 stock カーネルで足りる見込み。
   - 同期は greedy 2 回/step (draft ゲート + verify)、D7 発火時 3 回。
     同期税 ~0.4-0.6ms/step で小さい。

5. **sol (codex) の反証レビュー結果 (詳細は本人出力、要点のみ)。**
   - draft 24.2ms の「未説明 10ms」は誤りだった。既定 max_draft=8 では
     ゲート判定の前に最大 8 リンク分の MTP+lm_head を構築する
     (spec.py:699-719)。lm_head 2.75ms x 8 = 22ms で draft はほぼ説明が付く。
     **ゲートで捨てるリンクの lm_head を先に全生成しているのが今日見つけた
     最大の浪費** (捨て 5 リンクなら ~14ms/step)。逐次/チャンク式ゲート
     (リンク 2-3 本ごとに同期 0.2ms を払って早期打ち切り) にすれば、
     1 リンク省略 = 2.75ms なのでトレードは大きく有利。spec.py の構造変更。
   - phase_s は CPU 壁時計であり GPU 帰属ではない (maint の GPU 仕事は
     次 phase の eval に遅延計上され得る)。phase 再帰属は実 call 数の
     計装から。
   - lm_head m=1 の 260GB/s は「名目帯域 (パラメタ bytes / 壁時計)」で、
     decompose の lm_head 値には出力 slice のグラフも含まれる。原因候補の
     筆頭は 4bit unpack/dequant の発行律速 (quantized.h の qdot)。
     fused qmm→argmax/entropy は mx.fast に存在せず、自前 2 段カーネルが
     必要 (価値は full-vocab softmax まで吸収する場合のみ)。
   - 実施順の勧告: phase 再帰属 → m カーブ immutable 化 → lm_head m=1 →
     capture 2 パス。

### カーネル側の確定事項 (親は前提にしてよい)

- fast_qmm が M=6..16 のほぼ全域で最速 (残 nocap は (17408,5120) M=9 のみ)。
  ゼロ埋めパディングはホスト側維持が正解 (masked A は 20-45% 悪化で棄却)。
- v_direct (B 断片レーン直読み、ISA 予測 2.0x) は GPU 実測で棄却。
  命令数 -16% が時間 -1〜2% にしかならず、fast_qmm は発行律速を抜けている。
  以後のカーネル微最適化は命令数でなく GPU 実測でのみ判定する。
- B1 (tools/bridge 直接エンコード) は park。利得上限 0.9us/dispatch =
  ステップの 2-3% で、残作業の保守リスクに見合わない。再開条件は
  BRIDGE-NOTES §5.1。
- M=5 の依存チェーン 2 ラン再測定で MMA が全 4 形状勝ち。経路表は協定バー
  (両ラン 5% 超) を満たす up/gate・q・lm_head の 3 形状で M=5 を MMA に拡張、
  fast_qmm の M_MIN も 5 へ。M=2..4 は stock 維持が確定 (mma 1.07-1.65x 遅)。

### 保留中 (カーネル側で持っている)

- AC 電源での compare_engines 公式 1 レップ (本丸)
- B1 の AC 追認ラン 1 回 → 正式クローズ
- test_dispatch 厳密バー (1.10) の静音 AC 再確認
