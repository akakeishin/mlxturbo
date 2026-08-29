# A2 m=8 固定費の犯人候補（Fable の発散。Sol の発散と独立に作成）

前提の実測: m=8 で mlx 比 1.04x、物理床 0.143ms に対し 0.366ms（2.56 倍）。
m=6 は 0.63x で mlx の 2 パス実装にすら負ける。fast_qmm は同条件 1.57x。
候補は互いに機構が異なるものだけを残した。probe はどれも 1 変数変更。

## H1: dequant の粒度が 8 要素刻み（メモリ階層）

現ソースは group ループ内の kt ループ（8 回転）ごとに dequant しており、
scale/bias と packed word の読み直しが group64 一括方式の最大 8 倍になる。
fast_qmm が「tile 単位→group 単位の変更だけで 0.29→0.15ms」と明記した
現象そのもの。
- probe: dequant を kt ループの外へ出し、group64 ぶんを一度に private/threadgroup
  へ展開する 1 変更。期待: m=8 が 0.36→0.2ms 台
- 反証条件: 変更後も 0.3ms 超なら粒度は主犯でない

## H2: ラッパ・起動の固定費（計測層。観測同値崩し）

同じ「1.04x」という痕跡を説明する別因果: カーネル本体でなく
mx.fast.metal_kernel の呼び出し税（ensure_row_contiguous のコピー、encode、
per-call の Python 層）が 0.1ms 級で乗っている。m=6 が mlx に大負けする
（本体仕事は小さいのに差が開く）事実はこちらの仮説と整合する。
- probe: 同一 grid/threadgroup で即 return する空カーネルを同じ経路で呼び、
  チェーン時間を測る。空で 0.08ms 超なら wrapper 税が主犯級
- 反証条件: 空カーネルが 0.02ms 未満なら棄却

## H3: split-K=8 の縮約経路（並列構造）

8 simdgroup の partial を threadgroup メモリ経由で足す最終縮約が
threadgroup_barrier を挟んで critical path に乗る。K/8=640 の直列短縮は
利得だが、mlp shape では N 並列（2176 threadgroup）が十分深いので
split の必要性自体が薄い可能性。
- probe: SPLIT_K ∈ {2,4,8} のスイープ（テンプレート定数 1 箇所）
- 反証条件: 4 と 8 で差が 5% 未満なら棄却

## H4: スカラー 1 word ロード（命令発行）

packed の読みが uint32 1 個ずつなら発行数が uint4 一括の 4 倍。
scale/bias も 4 lane が同じ値を読む重複があるなら simd_shuffle で 1/4。
- probe: uint4 化のみの変種。期待 5-15%
- 位置づけ: H1/H2 解消後の上積み。単独で 1.5x には届かない

## H5: threadgroup 形状と occupancy（並列構造の別軸）

threadgroup=(32, SPLIT_K, 1)=256 threads で threadgroup メモリ
（B slab + partial）が occupancy を削っている可能性。M3 は Dynamic Caching で
レジスタは動的だが、threadgroup メモリ使用は依然 occupancy に効く。
- probe: B slab を private 化 / threadgroup メモリ半減の変種で A/B
- 反証条件: 差 5% 未満で棄却

## H6（遠距離・記録のみ）: 分解の変更

W を fp16 へ 2-pass dequant してから plain GEMM に落とす案は、mlp 50MB で
読みが 2 倍（+0.14ms）になり床を割れないため、m<=16 では成立しない。
lm_head（N=248320、演算床が高い）だけは例外になり得るが、優先しない。

## H2 probe 実測結果（2026-08-26、Claude 側で実行済み）

同一 grid/経路の空カーネル: 0.064 ms/call。実カーネル 0.471、mlx 0.373（同時計測）。
- 判定: ラッパ税は主犯でないが副犯。本体 ≒ 0.407ms は物理床 0.143ms の 2.8 倍で、
  主犯は本体内（H1/H3/H5）。
- ただし目標水準 (~0.15-0.2ms) では 0.064ms が 3-4 割を占めるため、終盤には
  ラッパ税の削減（複数射影の 1 カーネル束ね、qkv+z+b+a の融合など）が必須になる。
  これは H1 系が解けた後の追加候補として登録。

## 実行順（識別力/コスト比。H2 実測済みを反映）

1. H1 probe（dequant 粒度を group64 一括へ）— 本体 2.8 倍の筆頭容疑
2. H3 スイープ（SPLIT_K 2/4/8）、H5 A/B（B slab 配置）
3. H4（uint4 / simd_shuffle）は仕上げ
4. ラッパ税 0.064ms の削減（射影融合）は本体解決後の終盤課題

Sol は独立に発散すること（このファイルを読む前に自分の候補を出してから
突き合わせるのが望ましい）。
