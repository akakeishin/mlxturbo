# fastmlx bridge — B1 probe の下部工事ノート（2026-08-26）

対象は docs/research/ARCH-BETS.md の B1「検証ステップの直接エンコード」。買いたいのは
ラッパ税 0.064ms/call × 呼び出し数（docs/research/HYPOTHESES-A2.md H2）を
N 呼び出し → 1 submit に潰すこと。その前段として

- MLX の `mx.array` が抱える MTLBuffer を Python から取り出す
- fastmlx 自前の MTLCommandBuffer に N 個の dispatch を直接エンコードして 1 回 submit する

の最小経路を作り、正しさゲートまで通した。時間は測っていない（実行キューは下）。

置き場所は `tools/bridge/` と本ファイルだけ。既存ファイルには触っていない。
`pyproject.toml` も `.venv` も変えていない。

| ファイル | 中身 |
|---|---|
| `tools/bridge/fastmlx_bridge.mm` | C ABI の ObjC++ 実装。Metal / Foundation 以外に依存しない |
| `tools/bridge/build.sh` | `libfastmlx_bridge.dylib` を吐く。Xcode CLT だけで通る |
| `tools/bridge/bridge.py` | ctypes ラッパ。`metal_buffer()` と `Bridge` |
| `tools/bridge/chain_kernels.py` | 2 経路で共有する連鎖カーネルの定義 |
| `tools/bridge/test_bridge.py` | 正しさゲート 12 件。時間は測らない |
| `tools/bridge/bench_chain.py` | 計測ハーネス。静音窓で回す |

環境: MLX 0.32.2（pip wheel）、Apple M3 Max、macOS 25.4、Apple clang 21。
ソースビルドの MLX は要らない。

## 1. MTLBuffer の取り出し方

`mx.array.__dlpack__()` がそのまま答えだった。MLX は DLPack の device を
`kDLMetal(8)` で出し、`DLTensor.data` に **MTLBuffer のポインタ**、
`byte_offset` にバイト単位のオフセットを入れてくる。実測:

```
data = 0xa8e508fc0   class = AGXG15XFamilyBuffer
contents() = 0x103b94000  == memoryview(a) の先頭
length() = 16384
a[100:200] は同じ MTLBuffer + byte_offset=400
```

この経路なので、pybind11 も nanobind も libmlx へのリンクも要らない。
ctypes で capsule から 72 バイトの `DLManagedTensor` を読むだけで済む。
`mlx/extension.py` の CMake + nanobind 経路は使っていない。

MTLDevice は `[buffer device]` から取る。`MTLCreateSystemDefaultDevice()` の
戻りが MLX の使っているオブジェクトと同一である保証を当てにしないため。
テストで MLX の `device_info()["device_name"]` と一致することを確認している。

### capsule の deleter は自分で呼ばない

DLPack の作法どおり `mt.deleter(p)` を呼ぶと、その直後に capsule の
デストラクタが同じ deleter をもう一度走らせ、nanobind が
`Critical unrecoverable error` で abort する（SIGABRT、再現率 100%）。
capsule の名前を `used_dltensor` に書き換える手もあるが、単に
**capsule への参照を落として MLX 側のデストラクタに任せる**のが確実だった。
`bridge.py` はそうしている。リークもしない（同じアドレスが再利用されることを確認）。

## 2. 正しさゲート（実行済み・12/12）

```
$ ./tools/bridge/build.sh && .venv/bin/python tools/bridge/test_bridge.py
ok  device_is_shared         MLX と同じ MTLDevice (Apple M3 Max)
ok  buffer_identity          MTLBuffer.contents + byte_offset == mx.array の先頭
ok  buffer_offset_view       slice は同じ MTLBuffer + byte_offset=400
ok  non_contiguous_rejected  転置 view は拒否される
ok  chain_matches_mlx        N=16 n=4096 完全一致 (MLX / bridge 1submit / 閉形式), cb=1
ok  chain_n32                N=32 n=4096 MLX 32 回呼び出し == bridge 32 dispatch / 1 submit (bit 単位で一致)
ok  split_variant            1 CB / 1 encoder   cb=1 一致
ok  split_variant            1 CB / N encoder   cb=1 一致
ok  split_variant            N CB + MTLEvent    cb=16 一致
ok  split_cb_needs_event     N CB 分割の正答: event なし 3/12, event あり 12/12
ok  split_cb_independent_ok  互いに素な 8 区画は CB 分割でも一致
ok  chain_on_offset_view     N=8 byte_offset=4096 の窓だけが書かれた
ok  mlx_reads_bridge_result  mx.sum(y)=1498000.0 が閉形式と一致
ok  mlx_metallib_loads       17319 関数 (affine_qmv 864 / sdpa 57)、pipeline 生成まで到達
12/12 passed
```

依頼どおり `y = x*2+1` を N=32 チェーンし、`mx.fast.metal_kernel` 32 回と
自前 32 dispatch / 1 submit がビット単位で一致することを確認した。N=16 では
閉形式 `2^N x + (2^N - 1)` とも一致する（N=32 は float32 の厳密範囲を出るので
MLX 経路との直接比較にした。入力を 1/16 刻みの 0..3.75 に取らないと丸めで
閉形式と 1.0 ずれる）。

## 3. 実測で分かった危険: 同一キューでも command buffer は重なって走る

依存のある連鎖を 16 本の command buffer に割って同じキューへ commit すると、
`waitUntilCompleted` を全部に掛けても結果が壊れる。壊れ方は「16 段のはずが
12〜15 段ぶんしか掛かっていない値」で、試行ごとに段数が変わる。
12 回中 2〜3 回しか正解しない。

Metal の自動ハザード追跡は command buffer の内側までで、CB をまたぐ依存は
面倒を見ない。CB を割るなら `MTLEvent` の `encodeWaitForEvent` /
`encodeSignalEvent` で明示的に直列化する必要がある（`ORDER_CB` フラグ）。
event を入れれば 12/12 正解する。

同一 command buffer の中なら、encoder を分けても（`SPLIT_ENCODER`）
順序は保たれた。1 encoder の中の連続 dispatch は
`MTLDispatchTypeSerial` で直列化されるが、MLX の heap 由来（256B 以下）の
buffer は untracked になり得るので、dispatch 間に
`memoryBarrierWithScope:MTLBarrierScopeBuffers` を既定で入れてある
（`NO_BARRIER` で外せる）。

**B1 にとっての含意**: 検証 1 ステップを 1 本の command buffer に収めるのは
性能の話だけでなく、順序保証を無料で手に入れる話でもある。ステップを分割
するなら event のコストが乗る。

## 4. MLX の buffer を触ってよい条件（規約）

破ると静かに壊れる。テストで守っているのは 1〜4 で、5〜7 は設計側の約束。

1. **評価済みの配列だけ触る。** `mx.eval(a)` は a が出来上がるまでブロックする。
   `bridge.py` の `metal_buffer()` は内部で `mx.eval` を呼ぶが、これが見るのは
   その配列だけ。同じ buffer を入力に持つ**未評価のグラフが他に残っている**と、
   こちらの書き込みが MLX の後続 op と競合する。書き込み先を作る側では
   `mx.eval(...)` に加えて `mx.synchronize()` を打ってから submit する。
2. **寿命は Python 参照が握る。** 取り出した MTLBuffer ポインタは、元の
   `mx.array` が生きている間だけ有効。参照が切れると buffer は
   `MetalAllocator` のキャッシュへ戻り、次の確保で別の配列に渡される。
   `Dispatch` は `keep_alive` で元の配列を掴んだままにしている。submit 中に
   配列を捨てないこと。
3. **donation を封じるのも Python 参照。** MLX は eval のとき、以後使われない
   入力の buffer を出力に流用する。Python 側から参照を持ち続けている配列は
   参照数が落ちないので流用対象にならない。逆に言えば、**参照を手放した瞬間に
   その buffer は他の計算の出力になり得る**。捕まえたポインタを配列より長生き
   させてはいけない。
4. **row-contiguous のものだけ。** `mx.fast.metal_kernel` は
   `ensure_row_contiguous=True` で黙ってコピーを挟んでくれるが、こちらには
   その層が無い。`metal_buffer()` は dlpack の strides を見て非連続なら
   例外にする。転置 view を渡したいなら呼び出し側で実体化する。
5. **書き込み先は fastmlx が単独で持つ。** MLX の配列は不変前提で、同じ配列を
   指す別のグラフノードや `mx.compile` のキャッシュがあり得る。bridge が
   in-place で書き換えてよいのは「fastmlx が `mx.zeros` などで作り、MLX の op に
   入力として渡していない」配列に限る。MLX へは読み取り専用の葉としてだけ見せる。
6. **ステップをまたぐ状態（GDN の再帰状態・KV）は所有権を移す。** ここを MLX と
   共有したまま in-place 更新すると 5 の条件を破る。fastmlx 側の slab に置いて、
   MLX には結果だけを新しい配列として返す。
7. **CB を割るなら event。** 3 節のとおり。1 ステップ = 1 command buffer なら
   考えなくてよい。

小さい配列（256 バイト以下）は MLX が `MTLHeap` から切り出す。今回のテスト範囲
では問題は出ていないが、heap 由来 buffer のハザード追跡は untracked 側に倒れる
ので、状態や中間バッファを 256B 未満で持たないほうが安全。

## 5. 実行キュー（静音窓で回す）

計測はこちらでは走らせていない。`bench_chain.py` は 4 経路の壁時計を N について
掃き、最小二乗で「傾き = 1 dispatch あたりの固定費」「切片 = 1 submit あたりの
固定費」を出す。

```sh
# 0. ビルドと正しさゲート（測る前に必ず通す）
./tools/bridge/build.sh
.venv/bin/python tools/bridge/test_bridge.py

# 1. ラッパ税そのもの。空カーネルなので本体時間がほぼ乗らない。
#    H2 probe の 0.064 ms/call と同じ形の比較になる。
.venv/bin/python tools/bridge/bench_chain.py --kernel noop --n 4096 --reps 50

# 2. 長い連鎖。1 検証ステップの呼び出し数（数百オーダー）に近い領域で
#    傾きが崩れないかを見る。
.venv/bin/python tools/bridge/bench_chain.py --kernel noop --n 4096 --reps 20 \
    --steps 32,64,128,256,512

# 3. 本体仕事つき。要素数を上げて「固定費が全体の何割か」を実寸で見る。
.venv/bin/python tools/bridge/bench_chain.py --kernel affine --n 262144 --reps 30

# 4. submit 回数だけの効き。bridge と bridge-cb の差が submit 1 回の値段。
.venv/bin/python tools/bridge/bench_chain.py --kernel noop --n 4096 --reps 30 \
    --steps 1,2,4,8,16,32,64,128
```

出力の 4 経路:

- `mlx` … `mx.fast.metal_kernel` を N 回（現状の fastmlx）
- `bridge` … N dispatch / 1 encoder / 1 command buffer（B1 が買いたい形）
- `bridge-enc` … N dispatch / N encoder / 1 command buffer
- `bridge-cb` … N dispatch / N command buffer（MTLEvent で直列化）

読み方と判定線:

- `mlx` の傾きが 60us/dispatch 前後に出れば、H2 の 0.064ms/call と地続きの
  数字を再現できたことになる。ここが再現しないなら測り方を疑う。
- `bridge` の傾きが **5us/dispatch を切る**なら、N 呼び出し → 1 submit で
  ラッパ税が実質消える。B1 の前提が立つ。
- `bridge` の傾きが **20us/dispatch を超える**なら、encode 自体が重いので
  B1 の取り分は小さい。この場合は「1 カーネルへの融合」（射影の束ね）のほうが
  筋がよく、直接エンコードは棄却寄り。
- `bridge` と `bridge-cb` の差が submit 1 回の値段。B5（software pipelining）で
  submit 回数を減らす価値の見積もりに使える。
- `bridge` と `bridge-enc` の差は encoder 生成の値段。ここが大きいなら、
  検証ステップ全体で encoder を 1 本に保つ制約が効いてくる。

### 5.1 実測結果と B1 の判定 (2026-08-27)

bench_chain.py を初めて実行した (M3 Max、静音 load ~1.8、**バッテリー駆動**。
絶対値は AC で 2-3 割速くなる可能性があるが、経路間の差分構造は不変)。
生ログは bench/results/bridge-chain-noop.txt / bridge-chain-affine.txt。

| 経路 | noop 傾き | affine 傾き | 切片 (noop) |
|---|---|---|---|
| mlx | 2.1 us/dispatch | 12.6 us/dispatch | 192 us/submit |
| bridge | 1.2 | 11.7 | 174 |
| bridge-enc | 1.3 | 12.0 | 172 |
| bridge-cb | 15.0 | 18.7 | 165 |

判定線に対して: bridge 傾きは 5 us を大きく下回り「有効」。bridge-cb は
15-19 us で棄却線どおり脱落。ただし前提が崩れた:

1. **mlx 経路自身の per-dispatch 限界費が 2.1 us しかない。** H2 の
   0.064 ms/call は eval 境界込みの数字で、1 つの eval にまとめて流れる
   dispatch の限界費はその 1/30。MLX の遅延評価が既にラッパ税を償却している。
2. bridge が買える差分は **0.9 us/dispatch で一定** (noop でも affine でも)。
   検証 1 ステップ ~1500-2000 dispatch として 1.4-1.8 ms、ステップ時間の
   **2-3% が完遂時の上限**。切片も ~20 us/submit しか縮まない。
3. 支配項は **切片 ~170-190 us/submit** (mx.eval 1 回の往復) で、これは
   bridge でも消えない。ここを削る手段は「検証ステップあたりの eval 回数の
   削減」であり、spec.py の同期点設計の問題 (親セッション持ち分)。

**B1 判定: 保留 (park)。** §6 の残作業 (MLX 内蔵カーネルの引数レイアウト
移植、バッファ arena、residency 管理) は保守リスクが高い割に上限 2-3% で、
経路表較正 (1 op あたり 1.06-1.59x) と比べて優先度が立たない。再開条件:
(a) ステップあたり dispatch 数が大きく増える設計変更、(b) MLX 側の
per-dispatch 費の退行、(c) eval 回数削減を突き詰めた後に 2-3% が
最後の残りになったとき。AC 追認は下記の通り実施済み。

AC 追認 (同日、AC 電源・静音): noop で mlx 2.1 vs bridge 0.9 us/dispatch、
affine で 12.7 vs 10.5 us/dispatch、切片 ~143-168 us/submit は bridge でも
不変。バッテリー時と同構造で、利得は 1-2 us/dispatch (検証 1 ステップ
~1500-2000 dispatch として 2-4ms、ステップの 2-4%)。**B1 はこれで正式に
クローズ (park)。**再開条件は上記 3 つのまま。

## 6. 検証ステップ全体を直接エンコードするために足りないもの

現状の 1 ステップは、64 層（GDN 48 + full attention 16）＋ lm_head ＋ MTP ヘッド。
GDN 層は RMSNorm → 射影 4 本 → conv1d+silu → q/k norm → `gated_delta_states_step`
→ gated norm → out_proj → MLP、full attention 層は q/k/v 射影 → norm → RoPE →
SDPA → gate → o_proj → MLP。以下は今回の spike で埋まっていない穴。

### A. カーネルの供給

1. **自前カーネルの署名を書き起こす。** `gated_delta_states.py` /
   `qmv_wide_nocap.py` / `qmm_skinny_mma.py` は MSL 文字列を持っているので
   `fmb_library_from_source` にそのまま渡せる。ただし `mx.fast.metal_kernel` が
   自動生成していた部分（入出力ポインタの並び、`*_shape` / `*_strides` /
   `*_ndim` の付与、grid と threadgroup の当て方）は自前で作る必要がある。
   ここは機械的な作業だが、間違えると静かに壊れる。
2. **MLX 内蔵のカーネルをどう呼ぶか。** 自前実装が無いのは
   `mx.fast.scaled_dot_product_attention`（full 16 層）、`mx.fast.rms_norm`、
   RoPE、`nn.Conv1d`、`mx.softmax`、`mx.quantized_matmul` の stock 経路
   （M<6 と M>11）。
   - wheel 同梱の `mlx/lib/mlx.metallib`（182MB / 17319 関数）を
     `fmb_library_from_file` で開き、名前で pipeline を作れることは確認済み。
     `affine_qmv` 864 / `affine_qmm` 972 / `sdpa_vector` 57 / `rope` 18 /
     `rms` 12 / `conv` 180 / `steel` 1482 が入っている。この模型が使う
     `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_{0,1}` も実在する。
   - 足りないのは**引数レイアウト**。MSL 側のヘッダは wheel に同梱されている
     （`include/mlx/backend/metal/kernels/{quantized,sdpa_vector,softmax}.h`）
     ので読めるが、どのバッファを何番に束ねるかは MLX の C++ 側
     （`backend/metal/quantized.cpp` など）にあり、これは wheel に無い。
     GitHub の 0.32.2 タグから写す必要がある。
   - これは MLX の内部 ABI で、版が上がれば黙って壊れる。カーネル名と引数の
     並びを固定する gate（MLX 経由の結果と突き合わせる）が要る。
   - `rms` は `rmsbfloat16` と `rms_loopedbfloat16` の 2 形態しかなく、
     どちらを選ぶかの分岐（行長）も C++ 側にある。

### B. バッファの供給

3. **重みの buffer を層ごとに 1 回だけ取り出してキャッシュする。**
   4bit affine / group64 なので射影 1 本につき w・scales・biases の 3 配列。
   64 層 × 射影 7 箇所 ×3 ≒ 1300 本。`__dlpack__` は capsule を毎回作るので
   step の内側には置けない。モデルロード時に引いて固定するテーブルが要る。
   4 節の規約 2・3 から、そのテーブルは元の `mx.array` への参照も一緒に持つ。
4. **中間バッファの arena。** MLX は op ごとに出力を確保するが、直接エンコード
   ではステップ開始時に確保済みの scratch へオフセットで書き分ける。
   `metal_buffer()` は byte_offset 付き view を正しく扱えることを確認済み
   （`chain_on_offset_view`）なので、大きい `mx.array` を 1 本取って
   スライスで割る形にできる。割り付け表は fastmlx 側で持つ。
5. **residency の確認が済んでいない。** `setBuffer` で束縛したものは
   command buffer 単位で常駐扱いになるが、115GB 級で MLX の
   `MTLResidencySet`（`backend/metal/resident.h`）と自前 encoder の関係を
   見ていない。argument buffer や heap 経由の間接参照を入れるなら
   `useResource` が要る。

### C. 同期と順序

6. **CPU 往復を消す口が要る。** 今の bridge は「`mx.eval` で MLX を止める →
   自前キューへ submit → `waitUntilCompleted`」で、1 ステップに全同期が 2 回
   入る。このままだと B1 で買った分を同期で吐き出す。
   - 逃げ道 A（推奨）: **MLX 自身の command encoder に積む。**
     `mlx::core::metal::get_command_encoder(Stream)` と
     `CommandEncoder::set_buffer(const MTL::Buffer*, int, int64_t)` /
     `dispatch_threadgroups` / `commit` は `libmlx.dylib` から export されて
     いる（`nm -gU` で確認済み）。`set_buffer` は生の `MTL::Buffer*` を取るので、
     dlpack で得たポインタをそのまま渡せる。`mlx::core::array` の C++ オブジェクトを
     Python から取り出す必要が無い＝ nanobind の ABI に踏み込まずに済む。
     キューが 1 本になるので 3 節の順序問題も消える。
     代償は `libmlx.dylib` へのリンクと MLX 内部 API への依存。
     `core.cpython-313-darwin.so` は `@rpath/libmlx.dylib` を動的リンクして
     いるので、同じ dylib を掴めば allocator も device も 1 個のままになる。
   - 逃げ道 B: 自前キューのまま MTLEvent で MLX のキューと繋ぐ。MLX 側に
     event を差し込む口が Python に無いので、結局 libmlx リンクが要る。A より
     手数が多い。
7. **棄却時のロールバックが MLX 側にある。** `spec.py:195-244` は
   `mx.take_along_axis` などを使う。直接エンコードの後でここへ戻ると
   また同期が入る。ロールバックまで自前カーネルに落とすか、
   step 境界の 1 回に同期をまとめるかの設計判断が要る。

### D. 状態

8. **GDN の再帰状態と KV の in-place 更新。** 4 節の規約 5・6 の実装。
   現状は MLX の配列を作り直しているので、直接エンコードに合わせるなら
   リングバッファ化と所有権の移動が要る。ここは B2（自前フォーマット）と
   切り分けずに設計したほうが二度手間にならない。

### E. デバッグ

9. **失敗が見えなくなる。** MLX 経由なら `metal_kernel` の verbose と MLX の
   エラーが出るが、自前 encode では `MTLCommandBuffer.error` しか出ない。
   docs/research/ISA-NOTES.md の基盤と `mx.metal.start_capture` の併用が前提になる。
10. **`ensure_row_contiguous` の安全網が無い。** MLX 経由では黙ってコピーが
    入っていた。自前経路では呼び出し側が contiguous を保証しないと壊れる。
    `metal_buffer()` は例外を投げるが、これは実行時チェックであって
    静的な保証ではない。

## 7. 次の一手

順に:

1. 5 節の実行キューを静音窓で回して `bridge` の傾きを出す。判定線は 5us と 20us。
2. 5us を切ったら、6-C-6 の逃げ道 A（MLX の CommandEncoder に積む）を
   50 行で試す。ここが通れば同期の問題が消えて、B1 の見積もりが実物になる。
3. その上で GDN 層 1 枚（RMSNorm → 射影 4 本 → conv → gated_delta → out_proj）を
   直接エンコードし、MLX 経由と数値一致を取る。層 1 枚で差が出ないなら
   64 層でも出ない。
