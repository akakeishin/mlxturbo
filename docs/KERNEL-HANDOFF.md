# KERNEL-HANDOFF — カーネルセッションから親セッションへの申し送り

このファイルはカーネルセッションが更新する。新しい項目は上に足す。
詳細な根拠はそれぞれ docs/ISA-DIFF.md §9 と docs/BRIDGE-NOTES.md §5.1 にある。

## 2026-08-27

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

### カーネル側の確定事項 (親は前提にしてよい)

- fast_qmm が M=6..16 のほぼ全域で最速 (残 nocap は (17408,5120) M=9 のみ)。
  ゼロ埋めパディングはホスト側維持が正解 (masked A は 20-45% 悪化で棄却)。
- v_direct (B 断片レーン直読み、ISA 予測 2.0x) は GPU 実測で棄却。
  命令数 -16% が時間 -1〜2% にしかならず、fast_qmm は発行律速を抜けている。
  以後のカーネル微最適化は命令数でなく GPU 実測でのみ判定する。
- B1 (tools/bridge 直接エンコード) は park。利得上限 0.9us/dispatch =
  ステップの 2-3% で、残作業の保守リスクに見合わない。再開条件は
  BRIDGE-NOTES §5.1。

### 保留中 (カーネル側で持っている)

- AC 電源での compare_engines 公式 1 レップ (本丸)
- B1 の AC 追認ラン 1 回 → 正式クローズ
- test_dispatch 厳密バー (1.10) の静音 AC 再確認
