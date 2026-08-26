# fast_qmm (1.57x) と v4 (1.18x) の命令レベル差分、および v5 の設計差分

対象は m=8、bf16、mlp_up 形状 (K=5120, N=17408, affine-4 / group 64)。
CPU 完結の静的解析のみ。GPU は動かしていない。

## 0. 要旨

**差の原因は命令効率ではなく重みトラフィックにある。**

v4 (E120 QMV、表 ON) は 1 命令あたり 1.55 バイトの重みを流していて、
fast_qmm の 0.70 バイトの 2.2 倍効率が良い。実測 0.336 ms は
重み 100.3 MB に対して 298 GB/s、このマシンのストリーミング天井
345 GB/s の 86% に達している。**v4 は既に帯域律速で、しかもほぼ天井にいる。**

問題は分母のほうで、v4 は m=8 で重み行列を 2 回読む。
`active_input_groups(8) = ceil(8/4) = 2` が grid.x になっていて、
2 つの入力グループが同じ重みを別々に読み直す。これは mlx stock の
`qmv_wide` が nv=5 で `ceil(8/5) = 2` 回読むのと同じ構造で、
**v4 は m カーブ税の本体を継承したまま 1 パスあたりを速くした**ことになる。

したがって v4 の上限は「2 回読みの帯域床」で決まる:

| | 重み読み | 帯域床 | 対 stock 上限 | 実測 |
|---|---|---|---|---|
| mlx stock (qmv_wide nv=5) | 2x | 0.291 ms | — | 0.380 ms |
| v4 (E120 na=4) | 2x | 0.291 ms | **1.31x** | 0.336 ms (1.18x) |
| fast_qmm (8x8 MMA) | 1x | 0.145 ms | **2.61x** | 0.242 ms (1.57x) |

v4 は自分の上限 1.31x の 90% まで来ている。**残りは 0.13x しかなく、
1.5x には設計を変えないと届かない。** 一方 fast_qmm は上限 2.61x の 60% で、
0.9x 以上を命令発行側に落としている。伸ばす余地があるのはこちらだけ。

判定は **(ii) MMA 系の v5**。根拠は §5。
本解析で追加した probe `v_direct` は、fast_qmm 比で hot ループ **-24%**
(412 → 314)、shuffle 0、hot ループ内 barrier 0、スピル 0、G15 コードサイズ
2,066 B (fast_qmm 2,318 B、A2 base 4,046 B)。fast_qmm が実証した発行レートを
そのまま当てはめると **2.0x** 相当になる。

## 1. 何を同じ土俵に載せたか

`tools/isa/variants.py` に 5 変種を追加した (既存変種は無改変)。

| 名前 | 中身 |
|---|---|
| `v_e120_notable` | v4 の表 OFF 経路を m=8 に特殊化して平坦化 |
| `v_e120_table` | v4 の表 ON 経路 (m>=4 の既定) を同上 |
| `v_e120_na8` | E120 のまま inputs-per-group を 8 に = 重み 1 回読み |
| `v_e120_r2na8` | 同上、出力行を 4→2 にしてレジスタ圧を下げた版 |
| `v_direct` | MMA 系。B 断片をレーンが自分の 2 列ぶん直読みして作る |

E120 変種は `fastmlx/kernels/_qmm_skinny_mma_source.py` からの転記
(Layr-Labs、MIT、LICENSE は `tools/reference/e120/` に同梱済み)。
ハーネス側の制約で 2 点だけ形を変えている。どちらも計測対象のループ本体には
触れていない。

- テンプレートを手で展開した。`qmm_metal_file` は `header` を渡さないため。
- `xsums` を `x` に別名で載せた。ハーネスの入力は 4 本固定のため。
  どちらも read-only の device バッファで `air-buffer-no-alias` が付くので、
  表の読みは出荷版と同じロードに落ちる。違うのはベースレジスタだけ。

fast_qmm 側 (`ref_fastqmm_m8`) と A2 の MMA 版 (`base_m8_bf16`、
`tools/isa/snapshots/qmm_skinny_mma_a2.py` に固定) は既存の枠でそのまま載る。
コンパイル条件は mlx 0.32.2 の既定 (`-std=metal3.2 -O2
-fmetal-math-mode=safe`)。逆アセンブルは G13 (`applegpu_g13g`)、
M3 (G15) は `__text` サイズのみ (`docs/ISA-NOTES.md` §2-5 と同じ二段構え)。

## 2. hot ループの命令ヒストグラム (G13、m=8)

「hot」は唯一残る後方分岐の中身。ただし **1 反復が担当する仕事は系統で違う**
ので、この表の絶対値どうしを直接引き算してはいけない (正規化は §3)。

| kernel | hot | MMA | shuffle | load | wait | maxReg | spill | 全体 | G15 `__text` |
|---|---|---|---|---|---|---|---|---|---|
| `ref_fastqmm_m8` | 412 | 8 | 0 | 19 | 9 | 30 | 0 | 487 | 2318 |
| `base_m8_bf16` (v2) | 461 | 8 | 80 | 18 | 17 | 42 | 0 | 543 | 4046 |
| `v_bstage_m8` | 362 | 8 | 0 | 19 | 3 | 36 | 0 | 450 | 2372 |
| `v_direct_m8` | **314** | 8 | 0 | 16 | 4 | 52 | 0 | 401 | **2066** |
| `v_e120_table_m8` (v4 表ON) | 742 | 0 | 0 | 21 | 23 | 127 | 41 | 1029 | 6570 |
| `v_e120_notable_m8` (v4 表OFF) | 1330 | 0 | 0 | 20 | 39 | 127 | 95 | 1613 | 6998 |
| `v_e120_na8_m8` | 521* | 0 | 0 | 22 | 19 | 127 | 46 | 1100 | 6600 |
| `v_e120_r2na8_m8` | 738 | 0 | 0 | 24 | 20 | 127 | 19 | 1032 | 6360 |
| `current_qmm_skinny` (出荷形) | 1650* | 0 | 0 | 22 | 63 | 127 | **482** | 12370 | 55278 |
| `ref_mlx_qmv_wide` (対抗馬) | 147 | 0 | 0 | 13 | 12 | 47 | 0 | 294 | — |

`*` 印は静的な hot 区間に内側ループが残っている。`v_e120_na8` は
i ループ (trip 4) が展開されずに残るので、動的な 1 k-block は
239 + 4x282 = **1367 命令**。出荷形は M 別 8 ケースを 1 本に並べた switch で、
hot 区間が複数ケースをまたぐため参考値。

内訳の要点:

- `v_e120_table_m8` の 742 のうち浮動小数は fmadd32 208 + fmul32 80 +
  fadd32 80 = 368。レーンあたりの MAC は 16 値 x 4 出力行 x 4 入力行 = 256 で、
  **算術下限 256 に対して 368、1.44 倍**。残りは mov 112、bfeil 51、
  convert 64、device_load 21、そして **stack_load 20 + stack_store 15 の
  スピル往復 35**。
- 表 OFF (`v_e120_notable_m8`) は +588 命令 (+79%)。増分は fadd32 +64
  (レーン内 16 値の和を出力行ブロックごとに取り直す)、mov +171、
  スピル +52。**表 ON は正しい判断で、ここを戻す選択肢は無い。**
- `ref_fastqmm_m8` の 412 のうち MMA は 8、dequant 演算 (bfeil 17 +
  convert 16 + fmadd32 16) は 49。残り **約 250 が B スラブの往復**
  (iadd 108、or 64、and 38、threadgroup_store 32、threadgroup_load 8)。
  `bt[(kq*16+t)*8 + j]` がストライド 8 要素の書き込みなので、
  16 個のスカラー store とそのアドレス計算が全部残る。
- `base_m8_bf16` (v2) は shuffle 80 + それに付く mov/icmpsel で 220 命令、
  hot の 48%。`docs/ISA-NOTES.md` §4.1 の結論のまま。
- `v_direct_m8` は shuffle も threadgroup スラブも使わないので、
  上の 250 と 220 の**どちらも払っていない**。残るのは
  iadd 67 + mov 93 + and 34 + or 16 のアドレス計算と nibble マスクで、
  dequant 演算 (bfeil 19 + convert 16 + fmadd32 16) と MMA 8、
  device_load 16 がそれに乗る。

AIR 側で見えるロード幅 (`tools/isa/analyze_air.py`):

| kernel | device ロードの型 |
|---|---|
| `ref_fastqmm_m8` | `bfloat` x2, `i32` x2 |
| `v_direct_m8` | `<4 x i32>` x4, `bfloat` x4 |
| `v_e120_table_m8` | `i16` x1, `<4 x bfloat>` x1, `bfloat` x2, `float` x1 |

E120 の重み読みは **16 bit スカラー** (`ws[i]`、レーンあたり 8 バイト)。
`v_direct` は 128 bit (`uint4`) で、この 4 本が 1 グループ 64 要素ぶんの
2 列を全部持ってくる。

## 3. 正規化 (1 反復の担当が違うので、ここで揃える)

反復数と 1 反復あたりの重みバイトは grid とループから決まる。

- MMA 系: threadgroup = ceil(N/8) = 2176、simdgroup 8 本、
  simdgroup あたり (K/8)/64 = 10 反復 → **174,080 反復**。
  1 反復 = 8 列 x 64 K = packed 256 B + scale/bias 32 B = **288 B**。
- E120 na=4 (v4): threadgroup = 2 (入力グループ) x 2176、simdgroup 2 本、
  K/512 = 10 反復 → **87,040 反復**。
  1 反復 = 4 行 x 512 K = packed 1024 B + scale/bias 128 B = **1152 B**。
- E120 na=8: 入力グループが 1 本になるので **43,520 反復**、1152 B。
- E120 na=8 / 出力行 2: 87,040 反復、576 B。

| kernel | 総発行 (G 命令) | 重み traffic | instr/重みB | 重みB/instr |
|---|---|---|---|---|
| `base_m8_bf16` (v2) | 0.0803 | 50.1 MB (1x) | 1.601 | 0.62 |
| `ref_fastqmm_m8` | 0.0717 | 50.1 MB (1x) | 1.431 | 0.70 |
| `v_bstage_m8` | 0.0630 | 50.1 MB (1x) | 1.257 | 0.80 |
| `v_direct_m8` | **0.0547** | 50.1 MB (1x) | **1.090** | **0.92** |
| `v_e120_table_m8` (v4) | 0.0646 | **100.3 MB (2x)** | **0.644** | **1.55** |
| `v_e120_notable_m8` | 0.1158 | 100.3 MB (2x) | 1.155 | 0.87 |
| `v_e120_na8_m8` | 0.0595 | 50.1 MB (1x) | 1.187 | 0.84 |
| `v_e120_r2na8_m8` | 0.0642 | 50.1 MB (1x) | 1.281 | 0.78 |

重みタイルは 44.56 MB (4bit) + scale/bias 5.57 MB = 50.14 MB。
README の「mlp タイル 50MB」と一致する。

実測との突き合わせ (依存チェーン、m=8、bf16):

| | 時間/op | traffic | 達成帯域 | 天井比 | 発行レート |
|---|---|---|---|---|---|
| mlx stock | 0.380 ms | 100.3 MB | 264 GB/s | 76% | — |
| `base_m8` (v2、1.04x) | 0.366 ms | 50.1 MB | 137 GB/s | 40% | 219 G/s |
| fast_qmm (1.57x) | 0.242 ms | 50.1 MB | 207 GB/s | 60% | **296 G/s** |
| v4 表 ON (1.18x) | 0.336 ms | 100.3 MB | **298 GB/s** | **86%** | 192 G/s |

天井は README の実測ストリーミング上限 345 GB/s。
1.57x / 1.04x は `docs/GATE-RESULTS-A2.md` (stock 0.380 ms)、
1.1817x は `docs/STATUS.md` (stock 1.589666 ms / 4 = 0.397 ms) から。
別 run なので、それぞれ自分の stock で割って ms に戻してある。

この表が全部を言っている。

- **v4 は帯域律速で天井の 86%。**命令発行レート 192 G/s は fast_qmm の
  296 G/s の 65% しか出ていないが、帯域で頭打ちなので効いていない。
  発行レートが低い理由はレジスタ 127 本 + スピル 41 本で、occupancy が
  落ちているためと読める (G13 での観測。M3 の確定は §7)。
- **fast_qmm は帯域律速ではない。**60% で止まっているのは発行側。
  命令を削れば直接時間が減る位置にいる。

## 4. 「fast_qmm が何をしていないか」「v4 が何を余計にしているか」

数えられる形で並べる。

fast_qmm がしていないこと (v2 / mlx 比):

- `simd_shuffle` を 1 本も使わない。v2 は動的レーン番号の shuffle が
  1 MMA あたり 22 命令に展開されて hot の 48% を占める (ISA-NOTES §4.1)。
- 重みを 2 度読まない。mlx の `qmv_wide` は nv=5 で ceil(M/5)=2 回、
  v4 は na=4 で ceil(M/4)=2 回。fast_qmm は threadgroup が出力 8 列を
  丸ごと持ち、K を 8 simdgroup で分けるので 1 回で済む。
- dequant を MMA 粒度でやらない。量子化グループ 64 要素を一度に展開する。
  ISA-NOTES §5 の H1 判定どおり、効いている理由は読み直し削減ではなく
  shuffle 除去。
- 活性化をステージングしない。A 断片は device から `simdgroup_load` で直接。

v4 が余計にしていること (fast_qmm 比):

- **重みを 2 回読む。50.1 MB 増。**これが 0.4x の差の本体。
- スピル往復 35 命令/反復 (stack_load 20 + stack_store 15)。
  レジスタ 127 本に張り付いている。fast_qmm は 30 本、スピル 0。
- 浮動小数演算が算術下限の 1.44 倍 (368 対 256)。整数 nibble 積和の後で
  `acc += scale*partial + sums*bias` を出力行ごとに掛け直すため。
- 表 ON なら **カーネル起動が 2 本**になる (xsums producer + qmv)。
  依存チェーンでは 1 matmul ごとに追加の依存辺が挟まる。
  STATUS の「独立 xsums 起動の固定費」はここ。
- 出荷形は M 別 8 ケースを 1 本に展開していて G15 で 55,278 B。
  M を特殊化すれば 6,570 B (**8.4 分の 1**)。

逆に、v4 が fast_qmm より優れている点も明確にしておく。

- 1 命令あたりの重みバイトが 1.55 対 0.70 で **2.2 倍**。
- MMA の bf16 丸めを通らないので、B 断片の精度劣化が無い
  (v4 は table ON/OFF が bit 一致、stock との normalized max error < 8e-3)。
- threadgroup メモリと barrier をまったく使わない。

## 5. 理論下限と、そこからの浪費

### 5.1 E120 系 (v4 の系譜)

命令の下限は、レーンあたり 16 値 x 出力行 4 x 入力行 NA の MAC 数。
na=4 なら 256 fmadd。実際は 742。差 486 のうち、

- 変換 64 (bf16 → f32、16 値 x 4 入力行) は avoidable でない。
- nibble 展開 bfeil 51 と mask and 16 も同様。
- 余分な浮動小数 112 (368 - 256) は affine 補正の構造由来。
- **mov 112 + スピル 35 + icmpsel 17 = 164 がレジスタ圧の代償**で、
  これが削れる浪費。

つまり na=4 のままなら hot 742 → 概ね 580 まで。**総発行 0.0646 → 0.0505 G。
だが帯域律速なので時間はほぼ動かない。**天井は 1.31x のまま。

重み 1 回読みにするには na=8 が要る。実測:

| | hot (動的) | 総発行 | traffic | v4 の発行レートでの予測 |
|---|---|---|---|---|
| `v_e120_table` (na=4) | 742 | 0.0646 G | 100.3 MB | 1.18x (実測) |
| `v_e120_na8` (na=8, 行4) | 1367 | 0.0595 G | 50.1 MB | 1.28x |
| `v_e120_r2na8` (na=8, 行2) | 738 | 0.0642 G | 50.1 MB | 1.19x |

**トラフィックは半分になるのに総命令が減らない。**
na=8 は蓄積器が出力行 4 x 入力 8 = 32 本になり、KERNEL-INTEL が
CLOSED BRANCH として記録した「8 行/スレッド + 32 蓄積器」と同じ壁に当たる。
実際にコンパイラが i ループの展開を諦め (hot に内側ループが残る)、
スピル 46 本が入る。出力行を 2 に減らして圧を逃がしても
(`v_e120_r2na8`、スピル 19 まで下がる) 反復数が倍になって相殺する。

fast_qmm が出した 296 G/s の発行レートを仮に na=8 が達成できれば 1.89x になる。
だが na=8 はレジスタ 127 本 + スピル 46 本 + 展開失敗で、
v4 (127 本 + スピル 41 本) より occupancy が悪い。v4 自身が 192 G/s
しか出せていない以上、この仮定は取れない。

**E120 系は、na=4 のままなら 1.31x が天井、na=8 にすると蓄積器で潰れる。
どちらの枝も 1.5x に届かない。**

### 5.2 MMA 系

MMA が M 方向をレジスタに畳む密度が、E120 との構造的な差になっている。

- E120 は 1 レーンが M 方向を `vec<float, NA>` で持つ。na=4、出力行 4 で
  acc 16 + partial 16 + a0..a3 16 + sums 4 = 52 本の生きた float。
  na=8 にすると 104 本。
- `simdgroup_matrix<float,8,8>` は M=8 x N=8 を **レーンあたり 2 本**で持つ。
  A 断片 2、B 断片 2 を足しても 6 本。M=16 でも 4+2+2 = 8 本。

**M を広げるコストが E120 は線形、MMA はタイル境界まで定数。**
これが「重み 1 回読み」を安く実装できる唯一の道になっている。

MMA 系の 1 反復下限を数える (8 列 x 64 K、8 MMA):

- MMA 8 本 (必須)
- A 断片 8 本 (`simdgroup_load`、device 直接)
- 重みロード 4 本 (`uint4` x 4 = 2 列 x 32 B) + scale/bias 4 本
- dequant: bfeil 16 + convert 16 + fmadd32 16 = 48
- アドレス計算とループ制御: 20 前後

合計おおよそ **110**。実測の `v_direct` は 314 なので、
残り 200 は主に mov 93 と iadd 67 のアドレス計算 / レジスタ移動。
`wa_words[8]` と `wb_words[8]` を uint4 からスカラー 16 本に展開している
ところで、`kt` の添字が展開後も即値にならず mov が積み上がっている。
ここは配列を使わず uint4 の成分を直接参照する書き方で減らせる見込み。

現状の実測値だけで比較すると:

| kernel | 総発行 | fast_qmm の発行レートでの予測時間 | 対 stock |
|---|---|---|---|
| `ref_fastqmm_m8` | 0.0717 G | 0.242 ms (較正点) | 1.57x |
| `v_bstage_m8` | 0.0630 G | 0.213 ms | 1.79x |
| `v_direct_m8` | 0.0547 G | 0.185 ms | **2.06x** |
| (帯域床) | — | 0.145 ms | 2.61x |

`v_direct` は帯域床 0.145 ms より上にいるので、まだ発行律速。
つまり **命令を削ればそのぶん速くなる領域に居続ける**。

## 6. 判定と v5 の設計差分

### 判定

**(ii) MMA 系の v5 を採る。(i) の v4 改善では 1.5x に届かない。**

- (i) の上限は 1.31x (2 回読みの帯域床 0.291 ms)。実測 1.18x は既にその 90%。
  na=8 で 1 回読みにする枝は蓄積器 32 本で潰れ、実測でも総命令が減らない。
- (ii) は現時点の probe (`v_direct`) で 1 回読みを保ったまま
  fast_qmm 比 -24% の命令数。fast_qmm が実証した発行レートで 2.0x 相当。
  帯域床 2.61x までの余地が残っている。

この判定が反転する条件を 3 つ書いておく。

1. M3 で `v_direct` の `maxTotalThreadsPerThreadgroup` が
   fast_qmm より大きく落ちる場合 (maxReg 52 対 30)。occupancy が落ちれば
   発行レートの仮定が崩れる。
2. `v_direct` の冗長 L1 読み (4 レーンが同じ 32 B を読む、
   simdgroup あたり 8 倍) が L1 帯域で詰まる場合。
3. MMA の B 断片 bf16 丸め (normalized max error 4-6e-3) が
   受理率を実測で下げる場合。GATE-RESULTS-A2 の (a)/(b) 判断が未了。

### v5 の設計差分リスト

fast_qmm から引く構造事実は `docs/KERNEL-INTEL.md` と ISA-NOTES に
記録済みのもののみ。コードは転写しない。

1. **B 断片をレーン直読みで作る。** `simdgroup_matrix<T,8,8>` の 1 レーンは
   `(frag_row, frag_col)` と `(frag_row, frag_col+1)` だけを持ち、
   1 つの k タイルでは 1 列の全 8 行が 1 語の packed word に入っている。
   だからレーンは自分の 2 列ぶんの語を `uint4` で読み、`frag_row` で
   nibble を選べば足りる。shuffle も threadgroup スラブも不要。
   実測 hot 314 (fast_qmm 412 の -24%、v2 461 の -32%)。
2. **threadgroup メモリは split-K の縮約 partials だけにする。**
   hot ループ内の barrier が 0 になる (fast_qmm はグループごとに
   `simdgroup_barrier` 2 本。G13 では命令が生成されないが、
   スケジューラの順序制約としては残る)。
3. **重み読みを 1 回に固定する。** threadgroup が出力 8 列を丸ごと持ち、
   K を 8 simdgroup で分ける。`ceil(M/NA)` の入力グループ分割は持ち込まない。
   これが 0.4x の差の本体なので、他の何と引き換えにしても崩さない。
4. **A 断片は device から `simdgroup_load` で直接読む。** x のステージングを
   しない (fast_qmm と同じ判断。ISA-NOTES §3 の通り x のスカラー 2 本は
   AGX で 32bit 1 本に併合されるので、ステージングの利得が無い)。
5. **m<=8 は C タイル 1 枚、9..16 は 2 枚。** B 断片は 2 枚の C で共有されるので
   重み読みは m=8 と同じ 1 回。実測 `v_direct_m16` は hot 376 / 16 MMA =
   23.5 instr/MMA、maxReg 69、スピル 0 (fast_qmm wide は 458 / 28.6)。
6. **M ごとの特殊化はテンプレート引数で行い、1 カーネルに全幅を switch で
   並べない。** 出荷形 `current_qmm_skinny` は G15 で 55,278 B、
   レジスタ 127 本、スピル 482 本。特殊化版は 6,570 B。
7. **`wa_words[8]` / `wb_words[8]` のスカラー配列展開をやめる。**
   `v_direct` の 314 のうち mov 93 + iadd 67 がここに集まっている。
   uint4 の成分を直接参照する形にすれば §5.2 の下限 110 に寄せられる。
   これが v5 で最初に試す最適化。
8. **精度の扱いは fast_qmm と同条件。** B 断片が bf16 に丸まる点は変わらないので、
   GATE-RESULTS-A2 の (a) gate 緩和 / (b) fp16 断片 の判断は v5 でも必要。
   `base_m8_f16` (hot 514) が (b) の命令コストの目安になる。

### v4 側で今すぐ取れる分 (天井は動かない)

v5 に移るまでの間、v4 に残っている分も数えておく。

- M 特殊化でコードサイズ 55,278 B → 6,570 B。i-cache 圧が下がる。
  設計変更を伴わない。
- xsums producer の融合で起動 1 本削減。依存チェーンでの固定費が減る。
- レジスタ圧の緩和で hot 742 → 概ね 580 (§5.1)。ただし帯域律速なので
  時間には出にくい。

いずれも 1.31x の天井は動かさない。

## 7. GPU が要る確認 (`docs/ISA-QUEUE.md` へ)

1. `v_direct` の M3 での `maxTotalThreadsPerThreadgroup`。
   maxReg 52 が occupancy をどれだけ落とすか。fast_qmm (30) と並べて取る。
2. `v_direct` の正しさと依存チェーン実測。予測 2.0x の検証。
   正しさ gate は fast_qmm と同じ 4-6e-3 帯を想定。
3. 冗長 L1 読みの実コスト。1 語を 4 レーンが読む形が L1 で吸収されるか。
4. `v_e120_na8` の M3 実測。G13 では蓄積器で潰れているが、
   M3 は Dynamic Caching があるので同じにならない可能性がある。
   ここが覆ると §6 の判定 1 が揺れる。

## 8. 再現

```bash
python3 tools/isa/gen_kernels.py
tools/isa/build_air.sh
python3 tools/isa/analyze_air.py
tools/isa/agx_build.sh apple7 applegpu_g13g
python3 tools/isa/analyze_agx.py --json tools/isa/build/agx-g13g-stats.json
tools/isa/agx_build.sh apple9 applegpu_g15s
python3 tools/isa/analyze_agx.py --arch applegpu_g15s \
    --json tools/isa/build/agx-g15s-stats.json
```

`v_e120_na8_m8_bf16` の動的 hot は静的区間に内側ループ (trip 4) が残るので、
`analyze_agx.py` の hot 列 (521) ではなく 239 + 4x282 = 1367 を使う。
逆アセンブルは `tools/isa/build/agx/applegpu_g13g/*.asm`。

## 9. GPU 実測による判定の更新 (2026-08-27)

§7 の宿題のうち 1〜3 を M3 Max の依存チェーン実測 (16 ステップ連鎖、
4 形状 x M=2..16、候補交互計測) で回収した。マシンは静音でないため
すべて同一ラン内の相対比較で判定している。

**`v_direct` は棄却。** 数値は全 64 形状で nmax 4-7e-3 (fast_qmm と同帯) だが、
依存チェーンでは全形状・全 M で fast_qmm に負けた (M=8 で 5-12% 遅、
wide 帯で 20-60% 遅)。M=16 (A は block load) でも負けるので、犯人は
B 断片のレーン直読みそのもの。§6 の反転条件 1 (maxReg 52 の occupancy) と
2 (同じ packed 32B を 8 レーンが読む L1 冗長) が実際に効いた。
静的な命令数 -24% は発行レート仮定が崩れると時間に変換されない。
実装は fastmlx/kernels/qmm_direct.py に負の結果として残してある (経路未接続)。

**masked A 構築も棄却。** fast_qmm の構造のまま M<8 の A 断片を
frag_row < M のマスク付きレーン直読みに変える案 (ホスト側ゼロ埋め廃止) は、
数値こそ通るが M=6/7 で 20-45% 遅くなり nocap にも負ける。
per-lane スカラー A 構築が発行律速のカーネルでは block load +
ホスト側パディング 2 dispatch より高くつく。v_sgload_a (逆方向の交換で悪化)
と対になる結果で、「A は hardware block load、実在しない行はホスト側
ゼロ埋め」が現状の勝ち筋。ゼロ配列は _zpad キャッシュで 1 dispatch に減らした。

**B ステージングのベクトル化は採用。** B スラブを列メジャー bt[j*64+k] にし、
書き込みをストライド 8 のスカラー store 16 本 → vec<bfloat16_t,4> store 4 本、
読み出しを transpose 付き simdgroup_load に変えた。数値同帯のまま
交互計測 2 ラン x 4 形状の 8 比較全てで 1-3% 改善。fast_qmm の
_SRC / _SRC_WIDE 両方に適用済み。

**副産物: M=6/7 の nocap 上書きは再較正対象。** 改善後の fast_qmm は
依存チェーンで 4 形状 x M=6..16 の全 28 行で stock / nocap / v_direct に
勝った (M=6 で nocap 比 0.82-0.89)。現経路表が M=6/7 を nocap に振っている
3 行と M=9..10 は、静音窓での 2 ラン較正 (5% 一致プロトコル) で
MMA に反転する見込み。ゼロ埋めコピー税が M=6/7 敗因という仮説は、
チェーン文脈では支持されなかった (パディング込みでも fast_qmm が勝つ)。

### 9.1 ベクトル化後の ISA 再計測と、命令数→時間の換算限界

B ステージングのベクトル化を fast_qmm に取り込んだ後、同じパイプライン
(G13 / applegpu_g13g) で再集計した。

| kernel | hot (旧) | hot (新) | 差 |
|---|---|---|---|
| `ref_fastqmm_m8` | 412 | **346** | -16% |
| `ref_fastqmm_m16` | 458 | **389** | -15% |

hot ヒストグラムの差分 (m8): threadgroup_store 32→5 (vec4 store 化)、
or -22 / iadd -18 / mov -9 (ストライドアドレス計算の消滅)、
threadgroup_load 8→16 (transpose 付き simdgroup_load の代償)。

**ただし GPU 実測は -1〜2% しか動かなかった** (交互チェーン計測、静穏時)。
命令 -16% → 時間 -2% という換算で、fast_qmm は既に純粋な発行律速ではない。
v_direct の予測 2.0x が外れたのと同じ理由がここでも確認された。
§5.2 の「命令を削ればそのぶん速くなる領域」は現行の 346 命令水準では
成立しておらず、以後のカーネル微最適化は命令数ではなく GPU 実測だけで
判定すべき。split-K 粒度の変更 (4/16)、重みロードの 1 イテレーション先読み
(pipe) も試したが、静穏時の差は ±2% 以内で採用に値しない
(帯域競合下でのみ 5-8% の差が出るが、再現しない)。

### 9.2 経路表較正の保留と、test_dispatch の候補不一致

経路表の M=6/7・9..11 反転は 2 ラン較正を試みたが一致 8/44 行で保留
(クローラ負荷、載せ替えは静音窓待ち)。なお bench/test_dispatch.py の
"mma" 候補は fastmlx/kernels/qmm_skinny_mma.py (v5 skinny) を測っており、
dispatch の MMA 経路が実際に呼ぶ fast_qmm とは別物になっている。
較正表の mma 列と dispatch 実測の意味がズレるので、
次の較正前に候補を fast_qmm に揃えるか列を分けるのが望ましい
(bench/ は親セッション持ち分のため、ここでは指摘に留める)。
