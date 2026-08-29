# D2 セッション キックオフ（MTP ヘッド微調整）

**2026-08-26: 中止。学習は行わない (ユーザー判断)。D0 診断結果 (D2-RESULTS.md) は
汎用パックの設計材料として引き続き有効。**

新セッションはこのファイルから読み始めること。背景の正本は
docs/PLAN.md (Phase D)、docs/RESEARCH.md、docs/KERNEL-INTEL.md、README.md。

**2026-08-26 追記: D0 実測で本ファイルの前提が複数崩れた。vanilla 崩壊カーブに
乗っていない・再スケールは無効果・3.5 tok/step は K=3 で構造的に不可。
docs/D2-RESULTS.md の D0 節を先に読み、目標値はそちらを正とすること。**

## 目的

MTP ヘッド（~0.4B、backbone 凍結）を FastMTP レシピで微調整し、
連鎖受理を平均 ~2.2 → 3.5 トークン/step へ上げる。これが対 MTPLX
（推奨構成で 44 t/s、うち 37 t/s）逆転の主レバー。

## 手順（D0 → D2 の順。D0 を飛ばさない）

1. D0 診断（半日）: 位置別受理率 α_k を実測して vanilla 崩壊カーブ
   （k=1 ~70% / k=2 ~11% / k=3 ~2%）に乗っているか確認。あわせて
   連鎖位置ごとの ĥ の RMS を真の h と比較（Attention Drift 診断）。
   膨張していれば推論時スカラー再スケールを先に試す（訓練ゼロ）
2. データ生成: 実使用分布（コーディング・編集・日本語混在）の
   自己生成サンプル 5〜10 万。一般ドメインは効かない実証があるので
   ユーザーの実ワークロードに寄せる。温度 0.6 / top-p 0.95 / 最大長 4096
3. 学習 (fastmlx/train_mtp.py 新規): 位置減衰 CE（α_k ∝ 0.6^(k-1)、K=3）、
   position-shared で同一ヘッドを深度 1..3 に再帰適用、backbone 凍結。
   cosine LR ピーク 5e-5、batch 64、3 epoch 目安。M3 Max で一晩想定
4. 評価: α_k の before/after、bench/gate.py の greedy 一致（ヘッドは
   draft 専用なので分布同一性は自動で保たれるが gate は必ず回す）、
   spec_bench の実効 t/s
5. 伸び悩んだら: HASS の top-K 蒸留 (K=5-10, w=0.5)、EAGLE のノイズ注入
   U(-0.1, 0.1) を追加（RESEARCH.md 参照）

## 併せて入れると良いもの（同セッション向き）

- D1: 深度制御を KERNEL-INTEL.md の EMA 期待利得式に置換（h=0.18-0.20、
  checkpoint 型の値。0.43 は使わない）
- 位置別受理の計測は spec.py の accept_trace 集計で足りる

## 制約・注意

- MTP 契約実値: base_hidden_variant=post_norm / hidden_variant=post_norm /
  concat_order=embedding_hidden / mtp_position_mode=local
- norm 規約の地雷（KERNEL-INTEL.md「正しさの地雷」）を読むこと。
  ヘッドの保存・再ロードで +1 シフトの二重適用をしない
- 学習は MTP ヘッドのみ。backbone 学習はしない（PLAN.md の境界）
- GPU はカーネル側セッションの計測と排他。長時間学習の開始前に
  相手セッションの計測が走っていないか確認する
- 学習スクリプトと成果物は fastmlx/train_mtp.py と models/mtp-tuned/ に。
  レポートは docs/D2-RESULTS.md へ

## 期待値（実証済み参照値）

FastMTP: k=2 受理 11%→56%、k=3 2%→36%、平均 2.03 倍（7B、H20 1 枚 1 日未満）。
fastmlx の目標: 実効 55〜70 t/s（27B、M3 Max、式は README の速度式参照）。
