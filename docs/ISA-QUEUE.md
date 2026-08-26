# GPU 実行キュー（ISA 解析）

Metal デバイスが要る手順だけをここに置く。実行は親が直列で行う。
CPU で済む部分は全て済ませてあるので、以下は残り 3 件だけ。
背景と CPU 側の結果は `docs/ISA-NOTES.md`。

前提: リポジトリルートで、`.venv` が有効なこと。
どれも 1 分以内に終わる。GPU を占有する重い dispatch は無い。

---

## Q1. MLX が実際に生成するカーネルソースを取る（優先度: 高）

CPU 側の解析は、mlx が `mx.fast.metal_kernel` に付ける関数シグネチャを
`libmlx.dylib` の文字列表から復元して再現している（`tools/isa/mlx_signature.py`）。
本物と突き合わせて、同じテキストをコンパイルしているか確認したい。

```bash
mkdir -p tools/isa/build
.venv/bin/python tools/isa/mlx_dump_source.py > tools/isa/build/mlx-generated.txt 2>&1
```

返してほしいもの: `tools/isa/build/mlx-generated.txt` の中身
（`[[kernel]] void custom_kernel_...` から始まる引数リストの部分だけで十分）。

差分の見方:
- 引数の**順序・個数・アドレス空間**が一致していれば CPU 側の結論はそのまま有効。
- `const constant int* x_shape` 等が余分に/足りなく出ていたら
  `tools/isa/mlx_signature.py` の `build_metal_file` を直して再走する。
- 空白やコメントの差は codegen に影響しないので無視してよい。

失敗しうる点: `verbose=True` は mlx 0.32.2 の `metal_kernel.__call__` の引数。
`TypeError` になったら引数名が変わっているので、その旨だけ返してくれれば直す。

---

## Q2. M3 実機でのレジスタ占有量（優先度: 高。H5 と新カーネルの判定に必要）

`maxTotalThreadsPerThreadgroup` はドライバが「このシェーダのレジスタ使用量なら
1 threadgroup に何スレッド入るか」を返した値で、逆アセンブラ抜きで M3 の
レジスタ圧を測れる唯一の公開値。1024 なら余裕、下がっていればレジスタ律速。
`staticThreadgroupMemoryLength` も同時に取れるので H5 の threadgroup メモリ側も出る。

```bash
tools/isa/gpu_probe.sh
.venv/bin/python tools/isa/gpu_report.py
```

`gpu_probe.sh` は初回に `pipeline_probe.m` を clang でビルドする（ビルド済み）。
`tools/isa/build/metallib/*.metallib` 全 16 本についてパイプラインを作り、
`tools/isa/build/gpu/*.json` と `*.archive.bin`（MTLBinaryArchive）を書く。

返してほしいもの: `gpu_report.py` の表そのまま。特に

| 見たい行 | 何が分かるか |
|---|---|
| `base_m8_bf16` | A2 MMA カーネルが M3 でレジスタ律速か（H5 の可否） |
| `v_bstage_m8_bf16` | group 一括 dequant 版が occupancy を落としていないか |
| `ref_fastqmm_m8` | 1.57x のカーネルの占有量（比較基準） |
| `current_qmm_skinny` | 差し替え後の E120 版。G13 ではスピル 482 本・maxReg 127 だった。M3 で maxTPTG が 1024 未満なら実機でもレジスタ律速 |

`current_qmm_skinny` でパイプライン生成が失敗する場合は、
そのエラーメッセージだけ返してくれれば十分（他のカーネルは続行する）。

---

## Q3. M3 ネイティブコードの逆アセンブル（優先度: 低。やれたら）

`dougallj/applegpu` は G13（M1）用で G15（M3）を復号できない
（`docs/ISA-NOTES.md` §2-5）。Q2 が吐く `*.archive.bin` は M3 ドライバが作った
本物のバイナリアーカイブなので、将来 G15 対応の逆アセンブラが出たとき、
あるいは Xcode の GPU debugger に食わせるときの入力になる。
いまは保存しておくだけでよい。

```bash
.venv/bin/python tools/isa/gpu_report.py --disasm
```

返してほしいもの: 先頭 20 行程度。
`<disassembly failed>` だらけなら想定どおりなので、その旨だけで終わり。

補足: M3 の命令内訳が本当に必要になった場合の代替は Xcode の
Metal Debugger（capture → シェーダのパイプライン統計）で、
これは GUI 操作になるので今回のキューには入れていない。

---

## 実行後にやること

Q1 の結果でシグネチャに差があれば、CPU 側を直して以下を再走する（全部 CPU）。

```bash
.venv/bin/python tools/isa/gen_kernels.py
tools/isa/build_air.sh
.venv/bin/python tools/isa/analyze_air.py
tools/isa/agx_build.sh apple7 applegpu_g13g
.venv/bin/python tools/isa/analyze_agx.py
```
