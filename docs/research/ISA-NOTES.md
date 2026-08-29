# AGX ISA 解析基盤と第一次結果（2026-08-26）

対象は `docs/research/GATE-RESULTS-A2.md` の m=8 bf16 = 1.04x を出したカーネル、
すなわち当時の `fastmlx/kernels/_qmm_skinny_mma_source.py` の `build_source(8)`。
命令列まで下ろして H1 / H4 / H5 を判定する。

## 0. 作業中に対象ソースが差し替わった件

解析の途中で `_qmm_skinny_mma_source.py` の working copy が別設計
（Layr-Labs E120 QMV 移植、MMA を使わないレジスタ QMV）に置き換わった。
数値の再現性を保つため、A2 の MMA カーネルは
`tools/isa/snapshots/qmm_skinny_mma_a2.py`（GATE-RESULTS-A2 を出した commit の
中身）に固定した。以降 base_* と書いたらこのスナップショットを指す。
差し替え後のカーネルも同じパイプラインに `current_qmm_skinny` として載せてある
（§6 に所見）。

## 1. 取得したもの（出所とライセンス）

| 置き場所 | 出所 | ライセンス | 用途 |
|---|---|---|---|
| `tools/isa/applegpu/` | https://github.com/dougallj/applegpu （commit 4c5bae6、2026-07-01） | BSD 3-Clause, Copyright (c) 2021 Dougall Johnson | AGX 逆アセンブラ、Mach-O 抽出、compiler-explorer 一式 |

`.gitignore` に `tools/isa/applegpu/` が入っているので clone は commit されない。
`tools/isa/build/` も `tools/isa/build/.gitignore` で除外した。

外部から取ったのはこの 1 件だけ。以下は既にマシン上にあるものを使っている。

- Metal offline toolchain（`xcrun metal` = `metalfe-32023.883`, AIR 2.8 / air64_v28）
- `applegpu-nt` / `metal-tt`（Metal.xctoolchain 同梱の AIR Native Translator）
- `mlx.metallib`（`.venv` にある mlx 0.32.2 のもの。参照カーネル抽出のみ）
- `fastmlx/fast_qmm.py`（リポジトリに vendored 済み。参照カーネルとして再コンパイル）

## 2. 実際のパイプライン（想定手順からの変更点）

想定では「AGX ネイティブは MTLBinaryArchive 経由でしか取れないので GPU 実行キューへ」
だった。**これは不要だった。** Metal.xctoolchain に AIR Native Translator
（`metal-tt` / `applegpu-nt`）が同梱されていて、pipeline script（`.mtlp-json`）を
渡せば **Metal デバイスなしで**ネイティブ AGX コードが出る。
結果、ISA 解析の主要部分は全部 CPU で完結した。

```
build_source(m)                       tools/isa/snapshots/ (固定)
  └─ MLX 生成シグネチャを再現         tools/isa/mlx_signature.py
       └─ standalone .metal           tools/isa/gen_kernels.py
            └─ xcrun metal -S/-c      tools/isa/build_air.sh      → AIR (.ll/.air/.metallib)
                 ├─ AIR 統計          tools/isa/analyze_air.py
                 └─ metal-tt          tools/isa/agx_build.sh      → native AGX (fat Mach-O)
                      └─ Mach-O 抽出  tools/isa/agx_extract.py    → __text バイト列
                           └─ 逆アセ  tools/isa/analyze_agx.py    → 命令ヒストグラム
```

到達までに詰まった点と回避策:

1. `mx.fast.metal_kernel` は関数シグネチャを実行時に生成するので、`build_source()`
   の本体だけでは単体コンパイルできない。生成規則は wheel の
   `libmlx.dylib` の文字列表から復元した（`  const device T* x [[buffer(0)]]` 形式、
   本体に名前が出現した builtin だけを引数に足す 20 個のテーブル、
   `<name>_shape` / `_strides` / `_ndim` も本体に名前が出たときだけ足す、
   関数名は `custom_kernel_<name>`、`template [[host_name(...)]]` で明示実体化）。
   これは復元であって公式仕様ではないので、GPU キューの `verbose=True` ダンプと
   突き合わせて確認すること（`docs/research/ISA-QUEUE.md` Q1）。
2. `applegpu-nt -S` は「plugin interface not implemented: AIRNTEmitAssembly」で
   アセンブリを吐けない。ネイティブ image は出せるので、そちらを Mach-O から
   取り出す方式にした。
3. `applegpu-nt` に直接 `.ll` を渡すと `module AIR version (2.8) is bigger than
   the one of the target (2.5)` で止まる。arch を明示すると 2024 世代の
   legacy plugin にルーティングされるため。`metal-tt -gpu-family apple7|apple9` と
   `-platform_version macos 26.0 26.0` を使えば現行 plugin に載り、AIR 2.8 のまま通る。
   `-arch applegpu_g15p` は macOS 26 では未サポート扱いなので `-gpu-family` を使う。
4. `metal-tt` は入力 metallib を丸ごと出力に pack する。`mlx.metallib`（182MB）を
   食わせると出力が 2.5GB になる。thin → `__compute` 抽出 → 即削除で回した。
5. **dougallj/applegpu は G13（M1）用で、G15（M3）の符号化を復号できない。**
   G15 スライスは翻訳自体は通るが、逆アセンブルはほぼ全て
   `<disassembly failed>`。したがって:
   - 命令レベルの内訳は **G13（`applegpu_g13g`）** で取る。バックエンドは本物の
     AGX コンパイラなので構造（アンロール、スピル、ロード幅、shuffle 展開）は読める。
   - M3 の量は **`__text` バイト数**（同一ターゲット内の相対比較）と、
     GPU キューで取る `maxTotalThreadsPerThreadgroup` で押さえる。
   - G13 の絶対命令数を M3 の性能に直結させないこと。

コンパイル条件は mlx 0.32.2 の既定に合わせた: `-std=metal3.2 -O2
-fmetal-math-mode=safe`。AIR メタデータに `air.compile.fast_math_disable` と
`air.compile.denorms_disable` が出るので、mlx の math_mode="safe" と一致している。

## 3. AIR レベルで分かったこと

`tools/isa/build/air/*.ll`、集計は `tools/isa/analyze_air.py`。

- **バッファは `"air-buffer-no-alias"` が付く。** `y` への store が
  `w` / `scales` / `x` のロードを縛ることはない。エイリアス由来の hoist 阻害は無い。
- **`#pragma unroll` は AIR では展開されない。** kt ループは 8 回転のまま残り、
  `!llvm.loop.unroll.enable` がメタデータとして付くだけ。展開はバックエンド
  （AIR→AGX）で起きる（§4 で 8 回展開を確認）。フロントエンドの .ll を見て
  「アンロールされていない」と判断してはいけない。
- **scale/bias は既に group 粒度に hoist 済み。** group ループ本体で
  `load bfloat` × 2 + `air.simd_shuffle` × 4 が一度だけ。kt ごとではない。
- **`frag_row < 8` のガードは畳まれている。** `frag_row` が [0,7] に収まることを
  コンパイラが証明したので select も分岐も残っていない。
- MMA 1 回あたりの AIR 命令（m=8, bf16, innermost loop = 45 IR 行）:

| | mma | sgmat ld/st | shuffle | load | convert | fmuladd | fptrunc | lshr | and | IR 行 |
|---|---|---|---|---|---|---|---|---|---|---|
| base_m8_bf16 | 1 | 0 | 2 | 3 | 4 | 2 | 2 | 2 | 2 | 45 |
| base_m8_f16 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 35 |
| v_uint4_m8 | 1 | 0 | 2 | 3 | 4 | 2 | 2 | 2 | 2 | 40 |
| v_uint4_sgload_m8 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 30 |
| v_bstage_m8 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 |

- ロード幅（カーネル全体、device address space）: base は `bfloat` × 4 と `i32` × 1。
  つまり **x はスカラー bf16 で 2 本、packed は uint32 で 1 本**。
  `v_uint4` は `<4 x i32>` × 2 に化ける。
- `v_uint4` は AIR では `alloca` が 1 個出て `uint words[8]` がスクラッチに落ちる
  （kt ループが AIR で展開されないので添字が定数化しないため）。ただし AGX
  バックエンドは展開後にレジスタへ昇格させる（§4 でスピル 0）。
  **AIR の alloca を見てスピルと即断してはいけない。**

## 4. AGX レベル（G13 / applegpu_g13g、命令ヒストグラム）

`tools/isa/build/agx/applegpu_g13g/*.asm`、集計は `tools/isa/analyze_agx.py`。
「hot」は唯一残る後方分岐（量子化グループのループ）の中身、
すなわち **64 個の重みを処理する 1 回転**。kt ループはバックエンドで 8 回展開済み。

| kernel | hot instr | MMA | instr/MMA | shuffle | load | wait | maxReg | spill |
|---|---|---|---|---|---|---|---|---|
| base_m8_bf16 | 461 | 8 | 57.6 | 80 | 18 | 17 | 42 | 0 |
| base_m6_bf16 | 493 | 8 | 61.6 | 80 | 18 | 17 | 46 | 0 |
| base_m16_bf16 | 526 | 16 | 32.9 | 80 | 26 | 25 | 46 | 0 |
| v_uint4_m8 | 395 | 8 | 49.4 | 80 | 12 | 9 | 45 | 0 |
| v_sgload_a_m8 | 697 | 8 | 87.1 | 80 | 18 | 17 | 41 | 0 |
| v_bstage_m8 | 362 | 8 | 45.2 | 0 | 19 | 3 | 36 | 0 |
| ref_fastqmm_m8 | 412 | 8 | 51.5 | 0 | 19 | 9 | 30 | 0 |
| ref_fastqmm_m16 | 458 | 16 | 28.6 | 0 | 27 | 3 | 53 | 0 |
| ref_mlx_qmv_wide | 147 | - | - | 0 | 13 | 12 | 47 | 0 |

`base_m8_bf16` の hot ループ内訳（8 MMA、命令 461）:

```
mov                  84   simd_shuffle のレーン番号を毎回 r0h に置き直している
simd_shuffle         80   1 MMA あたり 10 本
icmpsel              60   shuffle 結果の選択
iadd                 52
mov_imm              43
bfeil                24   >>4 と &0xF
device_load          18   packed i32 × 8 + x の i16.xy × 8 + scale/bias × 2
wait                 17   ロードごとに即待ちしている
convert              16   u32→f × 2/MMA
fmadd32              16   scale*v+bias × 2/MMA
fadd32               16   bf16→f32 を fadd(x, -0.0) でやっている
if_icmp / pop_exec    9/9 lane<8 の実行マスク操作
simd_matrix_fmadd32   8   ← 本来の仕事
```

**行列演算は 461 命令中 8 本、1.7%。**
`simd_shuffle` + それに付く `mov` + `icmpsel` で **220 命令、hot ループの 48%**。

### 4.1 simd_shuffle が 1 本で済んでいない

ソース上は kt あたり `simd_shuffle` 2 本（packed0 / packed1）のはずが、
AGX では 1 MMA あたり 8 本 + `mov` 8 本 + `icmpsel` 6 本に展開される。
実際の 1 回転（`tools/isa/build/agx/applegpu_g13g/base_m8_bf16.asm` の 0x3aa〜0x4de）:

```
 3f0: mov          r0h, r34l
 3f6: simd_shuffle r22, r19, r0h
 3fc: mov          r0h, r34h
 402: simd_shuffle r20, r19, r0h
 ...                              （計 8 組）
 424: icmpsel  seq, r22, r35h, r9l, r23, r22
 42e: icmpsel  seq, r23, r36l, r9l, r21, r22
 ...                              （計 6 本）
 4de: simd_matrix_fmadd32 r7_r8, r22l_r22h, r21l_r21h, r7_r8
```

レーン番号が実行時値（`frag_col` / `frag_col+1`）なので、コンパイラが候補レーン
ぶんの shuffle を並べて `icmpsel` で選ぶ形に落としている。**動的レーン番号の
`simd_shuffle` は AGX では 1 命令ではない。**
「lanes 0..7 だけが読んで shuffle で配る」という現ソースの中心設計が、
そのまま最大のコストになっている。

### 4.2 ロードは既に十分まとまっている

x の bf16 スカラー 2 本は AGX で `device_load 1, i16, xy, ...`（32bit 1 本）に
併合済み。1 MMA あたりの device_load は packed 1 + x 1 の計 2 本しかない。
一方 `wait` が 17 本 = ロードごとに即座に待っており、8 回展開してもロードが
バッチされていない。`v_uint4`（group 頭で uint4 × 2 に前倒し）は
load 18→12、wait 17→9、命令 461→395（-14%）で、ここは効く。

### 4.3 m カーブの正体

`base_m16` は MMA を 16 本こなすのに hot 526 命令。`base_m8` は 8 本で 461 命令。
**固定費（shuffle 80、icmpsel 60、アドレス計算）は m を倍にしても増えない。**
instr/MMA が 57.6 → 32.9 になるのはそのため。
GATE-RESULTS の m16/m8 = 1.30 と m=6 が mlx に負ける事実は、
どちらもこの「グループあたり 400 命令超の固定費」で説明が付く。

### 4.4 参照カーネルとの対比

- `ref_fastqmm_m8`（実測 1.57x）: shuffle 0、hot 412、maxReg 30。
  B をグループ単位で dequant して threadgroup に置き、`threadgroup_load` で
  fragment を取る。**同じ MMA 数を、shuffle 一切なしでこなしている。**
- `v_bstage_m8`（fast_qmm と同じ構造を A2 の枠に入れた自作 probe）: hot 362、
  shuffle 0、wait 3。base 比 **-21%、wait は -82%**。
  G15 の `__text` では base 4046B に対し 2372B（**-41%**）で、M3 側では差がさらに開く。
- `ref_mlx_qmv_wide`（mlx の対抗馬、`affine_qmv_wide_bfloat16_t_gs_64_b_4_nv_5_kl_8`）:
  MMA を使わず素の `fmadd32` 48 本、hot 147 命令、shuffle は最終縮約の
  `simd_shuffle_down` 15 本だけ。無駄が非常に少ない。

## 5. H1 / H4 / H5 の判定

**H1（dequant 粒度が 8 要素刻み）— 半分ハズレ、結論は正しい。**
主張されていた「scale/bias と packed word の読み直しが最大 8 倍」は誤り。
scale/bias は既に group 粒度に hoist されている（AIR/AGX 両方で確認）。
packed word は kt あたり uint32 1 本で、データ量としては最小。
しかし **broadcast のコストが group 粒度に落ちていない**のが真の問題で、
kt ごとの `simd_shuffle` 2 本が AGX で 22 命令に展開され hot ループの 48% を占める。
つまり「group64 を一括で展開する」という probe の処方箋は正しく、
効く理由が読み直し削減ではなく shuffle 除去。
証拠: `v_bstage`（group 一括 dequant → threadgroup → `threadgroup_load`）で
shuffle 80→0、hot 461→362、wait 17→3、G15 コードサイズ -41%。

**H4（スカラー 1 word ロード）— 主犯ではないが実在する。**
「packed の読みが uint32 1 個ずつなら発行数が uint4 一括の 4 倍」は
命令数としては正しい（8 本 → 2 本）。ただし device_load は hot ループ 461 命令中
18 本しかなく、削っても -14%。
「scale/bias も 4 lane が同じ値を読む重複があるなら simd_shuffle で 1/4」は
**逆向き**。scale/bias は既に 1 グループ 2 ロードまで削られていて、
その配布に使っている `simd_shuffle` の方が高い。
H4 は「H1/H2 解消後の上積み。単独で 1.5x には届かない」と書かれていて、
その位置づけは実測どおり。

**H5（threadgroup 形状と occupancy）— G13 では棄却側。M3 未確認。**
base_m8 は maxReg 42、スピル 0、threadgroup メモリは `partials` の 2KB のみ
（256 threads × 8 simdgroup で、G13 の 32KB に対して余裕）。
レジスタもスループット制約になっていない。
ただし G13 のレジスタファイルは M3 と別物で、M3 は Dynamic Caching がある。
M3 での確定は `maxTotalThreadsPerThreadgroup`（`docs/research/ISA-QUEUE.md` Q2）待ち。

**H3（split-K=8 の縮約経路）— 命令レベルでは小さい。**
縮約は `threadgroup_barrier` 1 本 + `threadgroup_load` 8 本 + `fadd32` で、
カーネル全体 543 命令のうち hot 外の 82 命令。K/64=80 グループを回るので
1 回あたりに均せば 1 命令ぶん。critical path としての実測は別途必要だが、
命令数としては主犯ではない。

## 6. 差し替え後のカーネル（current_qmm_skinny）への所見

参考情報。依頼範囲外だが、同じパイプラインに載るので測った。

G13 で **命令 12370、maxReg 127、スピル 482 本
（`stack_load` 257 + `stack_store` 225）**。hot ループ 1650 命令のうち
136 本がスピル往復。`__text` は G13 83KB / G15 55KB（base_m8 の 14 倍）。
M（2..9）ごとの特殊化を 8 本 `switch` で並べてインライン展開しているので
サイズが大きいのは設計どおりだが、**レジスタ 127 本 + スピル 482 本は
G13 のレジスタファイル上限に張り付いている**状態。
M3 は Dynamic Caching があるので同じにはならない。
`maxTotalThreadsPerThreadgroup` を M3 で見れば一発で分かる
（`docs/research/ISA-QUEUE.md` Q2 にこのカーネルも入れてある）。

## 7. この基盤の使い方

```bash
python3 tools/isa/gen_kernels.py          # .metal を生成（CPU）
tools/isa/build_air.sh                    # AIR + metallib（CPU）
python3 tools/isa/analyze_air.py          # AIR 統計（CPU）
tools/isa/agx_build.sh apple7 applegpu_g13g   # 逆アセンブル可能なターゲット（CPU）
python3 tools/isa/analyze_agx.py              # 命令ヒストグラム（CPU）
tools/isa/agx_build.sh apple9 applegpu_g15s   # M3 世代、サイズ比較のみ（CPU）
python3 tools/isa/analyze_agx.py --arch applegpu_g15s
```

新しい variant を測るときは `tools/isa/variants.py` に body を 1 個足すだけで
上の全段に乗る。`gen_kernels.py` は working tree の
`fastmlx.kernels._qmm_skinny_mma_source.build_source()` も
`current_qmm_skinny` として自動で拾う。

GPU が要る 3 件は `docs/research/ISA-QUEUE.md` にまとめた。
