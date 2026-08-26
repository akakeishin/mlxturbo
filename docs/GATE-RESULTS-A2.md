# A2 GPU gate 実行結果（2026-08-26、Claude 側）

## 判定: 不合格（コンパイル修正1件は適用済み、性能基準未達）

## 実行経緯

1. A2-1 初回: Metal コンパイル失敗。原因は `_METAL_HEADER` が改行なしの
   `"#include <metal_simdgroup>"` で、ラッパが直後に生成するテンプレート宣言行を
   include 行が飲み込み、カーネル関数が非テンプレート化して `N` 未宣言になっていた。
   `#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n` に修正して解決
   （qmm_skinny_mma.py:21、Claude 側で適用済み・未コミット）。
2. 修正後: 全 M でコンパイル・実行成功。

## 正しさ

- M=13 が normalized_max 1.95e-3 で 1e-3 gate を超過し assert 停止。
- 追試（K=512/N=1024、x スケール 0.1）: 誤差は全行一様に 3.4e-3〜6.8e-3。
  タイル2境界に局在しない = インデックスバグではなく、B 断片を bf16 に丸める
  MMA 方式固有の量子化ノイズ（fp32 アキュムレータでも B 断片精度が支配項）。
  fast_qmm (参照実装) の実測 4-6e-3 と同水準。
- 判断が必要: (a) gate を「MMA 方式の物理限界」に合わせて ~1e-2 へ緩め、
  m>=6 の検証パス限定でノイズを許容する（m=1 の identity gate は stock 経路のまま
  bit-exact を維持）。(b) A/B 断片を float16 にした変種を作る（仮数 10bit で誤差
  ~8 分の 1、1e-3 gate を満たせる見込み。activations は normed なので fp16 範囲は
  安全側）。推奨は (b) を試して (a) をフォールバックにする。

## 性能（依存チェーン 8 段、mlp_up 5120x17408、bf16、参考値）

| m | mlx | mma | 勝率 | fast_qmm 参照 |
|---|---|---|---|---|
| 6 | 0.317ms | 0.499ms | 0.63x | 1.25x |
| 8 | 0.380ms | 0.366ms | 1.04x | 1.57x |
| 12 | 0.553ms | 0.596ms | 0.93x | - |
| 16 | 0.591ms | 0.476ms | 1.24x | (wide 未計測) |

- m16/m8 (mma) = 1.30 で「1.6 以下」基準は満たす。
- しかし本命の m=8 が 1.04x で受け入れ基準 1.5x に未達。m=6 は mlx（2 パス領域）に
  すら負けており、固定費（dequant の重複、scale/bias 再読、split-K reduction）が
  疑わしい。fast_qmm が同条件で 1.57x を出している以上、設計方針でなく実装細部の
  問題。差分候補: B タイルの置き場所（threadgroup vs private）、group 単位 dequant の
  C タイル間再利用、scale/bias の simd_shuffle 共有、reduction の critical path。

## 次のアクション（Sol へ）

1. fast_qmm とのカーネル構造差分を取り（コード転写はしない。構造の比較は可）、
   m=8 の固定費を特定して v2 を作る。
2. 正しさ gate の扱いを上記 (a)/(b) から選び、PLAN.md の受け入れ基準を更新する。
3. 性能計測は Claude 側で再実行するので、GPU gate queue を更新すること。
