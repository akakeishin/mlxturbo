# KERNEL-HANDOFF — カーネルセッションから親セッションへの申し送り

このファイルはカーネルセッションが更新する。新しい項目は上に足す。
詳細な根拠はそれぞれ docs/ISA-DIFF.md §9 と docs/BRIDGE-NOTES.md §5.1 にある。

## 2026-08-27

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
- compare_engines 公式レップ (official2 rep1) を AC で実行中。

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
