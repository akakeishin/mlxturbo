# ANE-GATE-RESULTS — 第 1 段 (小さな行列積の判定) と第 2 段 (量子化重み) (2026-08-30)

`docs/research/ANE-PREFILL-BRIEF.md` の調査レーンの結果。mlxturbo/ 本体は
変更していない。実験スクリプトは `tools/ane_matmul_gate.py` /
`tools/ane_op_probe.py` / `tools/ane_quant_probe.py`、モデルと venv は
一時ディレクトリ ($TMPDIR/opencode) に置いた。

## 結論 (先に)

- **第 1 段は条件付きで通過した。** GPU が占有されている現状では、
  seq ≤ ~8192 × hidden 2560 の fp16 matmul 連鎖で ANE (CPU_AND_NE) は GPU
  (CPU_AND_GPU) の **1.29–1.45 倍速い**。宣言基準 (1.2x) を超えたので
  第 2 段へ進んだ
- **ただし GPU が空いている瞬間に取れた 1 回の測定では GPU の方が速い**
  (0.909)。判定は GPU の負荷状態で反転する。詳細は下の「GPU 占有問題」
- **ANE が勝てるのは seq ≤ 8192 まで。** seq = 16384 では GPU に負け
  (paired 0.922、ただし NE/CPU 1.65 でまだ ANE 上で動いている)、
  seq = 32768 では参入をやめて CPU 落ちする (NE/CPU 1.06、GPU 比 0.498)。
  ブレークイーブンは 8192–16384 の間。Flash-Next の chunked prefill が
  4–8k 粒度なら範囲内、まるごと 32k では対象外
- **第 2 段 (量子化重み) は成立する。** 4bit affine (group 64) を numpy で
  dequant → fp16 埋め込みで、推論速度は fp32 直接埋め込みと同一
  (4.679 vs 4.681 ms)。ただし fp16 化のメモリ増大が大きく、
  全層の静态 dequant は現実的でない (下に数字)

## 第 1 段の数字

形: `(seq, 2560) x (2560, 2560)` fp16、6 連鎖 matmul、同一プロセス内交互測定、
中央値 (ROUNDS=40、WARMUP=15)。単位 ms/predict。

### GPU が占有されている状態 (このマシンの通常状態)

| seq | CPU_AND_GPU | CPU_AND_NE | CPU_ONLY | paired GPU/NE | NE/CPU |
|---|---|---|---|---|---|
| 1024 | 11.79 | 9.55 | 15.84 | 1.288 | 1.658 |
| 2048 | 23.17 | 17.55 | 29.84 | 1.344 | 1.70 |
| 4096 | 44.42–50.95 | 32.83–33.08 | 62.5–69.7 | 1.407 (4 ラン集約) | 1.90 |
| 8192 | 91.65 | 63.44 | 137.4 | 1.453 | 2.17 |
| 16384 | 114.7 | 125.2 | 207.0 | 0.922 | 1.653 |
| 32768 | 373.0 | 749.0 | 796.2 | 0.498 | 1.06 |

seq 4096 を 4 ラン集約した pooled paired median は **1.407 (q1=1.30, q3=1.56)**。
min-of-medians でも GPU 44.4 vs NE 32.8 → 1.35。

### GPU が空いていた最初の 1 回

| seq 4096 | CPU_AND_GPU | CPU_AND_NE | CPU_ONLY | GPU/NE |
|---|---|---|---|---|
| median | 29.21 | 32.14 | 50.30 | 0.909 (GPU 勝ち) |

その後、GPU は 29→42→44–51ms と悪化し続けて戻らなかった (CPU_ONLY も
50→62ms)。ANE は全 6 ランで 32.1–33.6ms、ばらつき 4% 以内で動かなかった。

## GPU 占有問題 (重要)

同じコード・同じモデルで判定が 0.909 と 1.407 に反転した。原因は測定ではなく
**このマシンの GPU/CPU を占有している別プロセス**。`ps` で確認した事実:

- `.venv/bin/python` (15% CPU, 20.8% メモリ — 推論サーバー系と思われる)
- Chrome Helper (42.9% CPU)
- load average ≈ 6.1–6.6

GPU と CPU の測定値はこの占有の波で動く。ANE だけは誰とも競合しないので
動かない。これは「測定が間違っていた」のではなく、**比較の母集団が 2 つある**
(GPU 空き時 / 占有時) という話。

どちらの判定を採るかは用途で決まる。デプロイメントでは decode が GPU を常時
占有するので「占有時の比較」が現実的で、その基準では ANE は 1.35–1.41x 速い。
逆に「デバイス単体性能の比較」としては空き GPU の方が ANE より 1 割速い。
このレポートは判定を安易に 1 つに決めず両方を残す。**採用判断をするときは、
prefill の offload 先として GPU と ANE のどちらが空いているかを、実運用の
負荷で測り直すこと。**

### 外乱に強いプロトコル (この結果を取った方法)

1. ラウンドごとの paired ratio (GPU_time/NE_time) を取る。外乱がスイープ全体に
   かかっても相殺されやすい
2. 複数ランを集約して pooled paired median を見る (q1–q3 も出す)
3. 単発の run は信用しない。今回 1 回目 (0.909) と 2 回目 (1.271) が反転した
   事実がその証拠。KERNEL-BRIEF-MOE-GDN と同じ轍を踏みかけたので、
   このメモを残す

## ANE が本当に動いたかの検証方法

powermetrics は root が取れず (sudo にパスワードが要る)、xctrace (Core ML
テンプレート) は attach/launch のどちらも応答しなくてこの環境では使えず、
`log stream` には Core ML のデバイス決定ログが出なかった。代替として
**ソフトウェア指紋**で検証した:

1. `CPU_AND_NE` は許容デバイスが {CPU, NE} だけ。**GPU へのフォールバックは
   仕様上不可能**。それが CPU_ONLY より 1.66–2.17x 速い = NE が実行した
2. コントロール (`tools/ane_op_probe.py`): cumsum / topk を混ぜても CPU_AND_NE
   は CPU_ONLY 水準に落ちず 1.43–1.49x を維持した。Core ML がグラフを分割して
   matmul チェーンだけ NE に載せている = NE への参入が実在する傍証

powermetrics/Instruments での物理確認は残課題 (root 権限が必要)。

## 第 2 段 (量子化重み) の数字

- numpy で 4bit affine (group 64) を dequant → fp16 を mlprogram の const として
  埋め込み、**推論速度は fp32 直接埋め込みと同一** (4.679 vs 4.681 ms)
- dequant 版と fp32 版の出力差は range 比 max 0.079。これは量子化誤差そのもので
  ANE 由来の追加ずれではない (量子化ノイズのオーダー、許容範囲)。
  ブリーフの KLD 指定は logit 分布向けの基準なので、この matmul プローブでは
  max frac-of-range で代用した。KLD での評価は第 3 段で logit が出る段階でやる
- メモリ増大: (2560, 2560) の fp16 化は **13.11 MB/行列**。512 experts/layer で
  **6.71 GB/layer**。全層の静态 dequant は非現実的で、offload する層を絞る
  設計 (第 3 段の話題) が前提になる

## この過程で分かった Core ML / coremltools の制約

1. `coremltools.models.MLModel` にある (`coremltools.MLModel` ではない)
2. MIL Builder の dtype は `coremltools.converters.mil.mil.types.fp16` を使う。
   class repr が `...make_float.<locals>.double` と出るが `__name__` は fp16 で
   実体も fp16。repr に惑わされないこと
3. fp16 の I/O はコンバータが自動で fp32 に変えて cast を挿入する。境界に明示
   cast を書けば警告なしで通る。内部演算は全 compute_unit で同じになる
4. `CPU_AND_NE` は GPU フォールバック不能。**CPU_AND_NE vs CPU_ONLY の速度差は
   NE 参入の指紋として使える** (root なし検証の本命)
5. 単発 matmul 1 本だと predict のオーバーヘッドが支配してデバイス差が埋もれる。
   6 連鎖程度にして演算量を確保した
6. ANE は大きいテンソルでは参入をやめる。seq 16384 では ANE 上で動いているが
   GPU に負け、seq 32768 では NE/CPU 比 1.06 (事実上 CPU 落ち)。**ANE の有効域は
   seq ≤ 8192 × hidden 2560 まで**
7. サーマルドリフトより外部負荷の方が支配的だった。交互測定だけでは足りず、
   paired ratio と複数ラン集約が必要

## 次に確かめるべきこと (第 3 段に向けて)

1. Flash-Next の prefill のどの部分 (MoE expert 行列積 / full attention / GDN)
   を offload すると壁時計に効くか。まず 290–325 tok/s の内訳を分解する
2. offload 層を絞った場合の fp16 メモリ増大の合計 (13.1 MB/行列 × 対象行列数)
3. chunked prefill の実際のチャンク粒度が seq ≤ 8k の ANE 有効域に収まるか
4. 運用時の GPU 占有状態で「GPU が空いている時刻」がどのくらいあるか。
   decode と prefill が常時 GPU を取り合うなら ANE offload の利得が実在する。
   GPU が空く運用なら (この測定では) GPU 単体が 1 割速い
5. powermetrics での物理確認 (root 権限が要る)

## 参考データ (生の数値は `tools/ane_matmul_gate.py` の JSON 出力)

- seq 4096 4 ラン集約: pooled paired median 1.407、min-of-medians 1.353、
  各ランの GPU median 50.95/45.09/44.88/44.42、NE median 32.83/33.08/32.99/32.85
- 32k: GPU 373 / NE 749 / CPU 796
