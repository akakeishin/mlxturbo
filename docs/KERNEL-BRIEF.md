# KERNEL-BRIEF — カーネル専門セッションへの引き継ぎ (2026-08-27)

このファイルはカーネル専門セッションの起点。親セッション (エンジン/計測/製品担当)
とはファイル境界で分業する。ここに書いていない背景は docs/KERNEL-INTEL.md、
docs/ISA-NOTES.md、docs/STATUS.md を読むこと。

## ミッション

検証 1 ステップの GPU コストを下げ、MTPLX recommended との残ギャップ
(code -10% / prose -13% / edit -31%、docs/STATUS.md の正式ベンチ v1) を
カーネル側から詰める。受理率 (tok/step 4.44/2.65/3.19) は決定的で反復間
完全一致なので、per-step 時間だけが変数。

## 現在の到達点

- 経路表 (fastmlx/kernels/dispatch.py): 形状別 M=0..16 行。実体は
  M=6..8 fast_qmm MMA (一部 nocap)、9..11 nocap、14..16 wide MMA。
  2026-08-27 静音較正 (bench/results/calib-quiet-a/b.json、両ラン 5% 超
  一致行のみ採用) が最新の根拠
- fast_qmm (fastmlx/fast_qmm.py、MIT 確認済み fork): m=8 依存チェーンで
  stock 比 1.57x。8x8 simdgroup MMA + グループ単位 dequant + 8-way split-K
- 自作 3 世代 (kernels/_qmm_skinny_mma_source.py v5 / e120_v4 切替可) は
  1.16-1.18x で頭打ち。差の残り原因は L1 冗長読み吸収 / occupancy と推定
- 物理天井: streaming 345-350GB/s、GEMM ~13.1TF。m=1 decode は 96-99% 飽和
  済みなので稼ぎ場は m=2..17 の検証窓のみ

## 最優先の作業 (Sol の PERF 所見、fast_qmm.py 行番号は 2026-08-27 時点)

1. B ステージング (68-103 行): 16 strided bf16 store + 8 fragment reload +
   64K グループごと barrier 2 回。最大の発行側コストと断定済み
2. M=6/7 のゼロ埋め (271-274 行): 呼び出しごとに 8xK バッファを新規確保+
   コピー。較正で M=6/7 が nocap に負けた直接原因の可能性が高い
3. scale/bias の 4 重アドレス要求 (83-91 行): 4 つの kq レーンが同じ j を共有
4. threadgroup が N 8 列のみ (64, 99-101 行): 活性タイルを ceil(N/8) 回再読
5. split-K の還元コスト (106-116 行): 2KiB partial + barrier + 8 項還元/タイル
6. wide の捨て行 (122-190 行): M=9 は 8 行タイルの 7 行を捨てる。9..13 用の
   中間ジオメトリがあれば nocap から奪える

## 受け入れ基準 (親セッションの gate と同一)

- bench/test_dispatch.py: 全 45 形状で数値一致 (normalized_max 0.0) を維持。
  較正主張をするときは静音窓で 2 ラン、両ラン 5% 超一致行のみ
- 経路表を変えたら bench/gate.py で baseline_all_identical: True を確認
- GPU 計測は同時 1 プロセス (docs/STATUS.md の GPU gate queue 節)
- 性能の最終判定は依存チェーン実測 (bench/ の依存チェーン系スクリプト)。
  単発スループットと勝者が入れ替わった前例あり (M=6/7)

## 使える道具

- ISA 解析: tools/isa/ (metal-tt オフライン AGX コード生成、applegpu 逆アセンブラ
  G13)。手順は docs/ISA-NOTES.md / ISA-DIFF.md
- ブリッジ: tools/bridge/ (DLPack→MTLBuffer、bench_chain.py)。検証ステップの
  直接エンコード (libmlx get_command_encoder 経路) は着手済み未完 (B1 続き)
- 参照実装: tools/reference/e120/ (Layr-Labs アリーナ、MIT。threadgroup/
  barrier/shuffle ゼロの整数 nibble 積和)
- 計測レーンの宿題 (Sol 監査残): canonical trajectory 較正レーンと、実経路
  (capture 込み) の immutable m カーブ。カーネル改善の前にこれを固めると
  以後の A/B が速い

## 境界

- fastmlx/spec.py、bench/gate.py、bench/compare_engines.py は親セッションの
  持ち分。カーネル側から触る必要が出たら理由を書いて止まる
- 訓練系 (D2) は中止済み。訓練を前提にした最適化は提案しない
