# QSA block-sparse attention prefill カーネル: 設計 (2026-09-02、レーン 3)

`docs/research/LANES-2026-09.md` レーン 3 の設計文書。**この文書を書いた
セッションは GPU 実行禁止 (別の計測が走行中)** — 数値は既存の実測 JSON
(`bench/results/gather-union-stats-true.json`、`bench/results/prefill-attn-ab*.json`)
からの再計算と、ソース読解による見積もりに限る。骨組みコード
(`mlxturbo/kernels/qsa_prefill_attn.py`) のうちホスト側 union 添字生成は
CPU で検証済みの実コード、Metal カーネル本体は **未コンパイル・未検証の構造
のみ**。次にこのレーンへ着手するセッションは GPU 検査から始めること。

## 背景 (再掲)

Flash-Next の QSA (`mlxturbo/_vendor/qwen4_exp.py` `QSAIndexer`) は
kv_len が 2048 (`token_budget`) を超えると、各クエリごとに indexer が
top-512 ブロック (`block_topk = token_budget // compress_ratio = 512`、
1 ブロック = compress_ratio(4) トークン = 2048 列) + 自分の端数ブロック
だけを可視にする。head_dim 256、q head 24、kv head 2 (GQA 12)、bf16。
いまの prefill は dense マスク (`Attention.__call__` → `_final_mask` →
`mx.fast.scaled_dot_product_attention`) で kv_len に比例して伸びる。

## (a) 前回のカーネルが dense と同速だった理由

前回のカーネル (`mlxturbo/kernels/prefill_attn.py`) は **1 threadgroup =
1 クエリトークン x 1 kv head**、12 simdgroup が 12 q head を分担する設計
(`prefill_attn.py:426-434` の `grid=(32*gqa, S, B*n_kv)`,
`threadgroup=(32*gqa,1,1)` — threadgroup 数は x=1, y=S, z=B*n_kv なので
文字どおり「クエリ 1 本ごとに 1 threadgroup」)。in-model A/B
(`bench/results/prefill-attn-ab.json`, `prefill-attn-ab2.json`、17k、
複数プロンプト x 3 回、CLAUDE.md の作法どおり ABBA/中央値) の結果は
prefill_s が **-1.1〜-1.25%** (中央値ベース、2 本の JSON で -1.25% と
-1.08%)。ノイズ幅 (rep 間 ±3%、`KERNEL-BRIEF-DECODE-BW.md` の段 1-5 前後比較
と同水準) に埋もれる差であり、「速くなっていない」が正しい要約。

### 仮説 1: クエリごとの再読み出しで K/V の総読み出し量が dense を上回る

kv_len が budget (2048) を超えると、各クエリの可視列はほぼ常に
budget 分 (block_topk x cr = 512 x 4 = 2048 列) に張り付く
(`QSAIndexer._pooled_and_top` の top-k が block_topk 本を必ず選ぶ、
`mlxturbo/_vendor/qwen4_exp.py:439`)。前回のカーネルは 12 head をこの
threadgroup 内で共有する (`prefill_attn.py:12-14` の docstring) が、
**クエリをまたいだ共有は無い** — threadgroup メモリは threadgroup ごとに
独立なので、隣のクエリが同じブロックを選んでいても K/V タイルは
threadgroup ごとに device メモリから読み直される。

具体的な読み出し量 (17k 文脈、PREFILL_STEP_SIZE=2048 のチャンク 1 つ、
1 層):

```
1 クエリ x 1 kv head あたり: ncol(≈2048) x head_dim(256) x itemsize(2, bf16) x 2(K,V)
  = 2048 x 256 x 2 x 2 = 2,097,152 バイト ≈ 2 MiB
S=2048 クエリ x kv_head=2:
  2048 x 2 x 2 MiB = 8192 MiB ≈ 8 GB/層/チャンク
```

これは課題文が挙げる見積もりと一致する。対して dense sdpa
(`mx.fast.scaled_dot_product_attention`) はクエリ方向にもタイルを切って
K/V タイルを複数クエリで使い回す (flash-attention 型)。仮にクエリタイル
64 行なら、1 層 1 チャンクの読み出しは概算で

```
(S/64) クエリタイル x kv_len(17408) x head_dim(256) x itemsize(2) x 2(K,V) x kv_head(2)
  = 32 x 17408 x 256 x 2 x 2 x 2 ≈ 1.14 GB/層/チャンク
```

程度 (dense sdpa 内部のタイル幅は未確認の推定値であり、この数字自体を
主張の根拠にはしない — 傍証として桁だけ見る)。**「疎にした列数は budget
(2048/17408 ≈ 12%) まで削れているのに、クエリ方向の共有を捨てたせいで
総読み出し量は dense より増えている」**というのが仮説の核。刈り込みの
効果 (少ない列だけ読む) を、非共有の代償 (毎クエリ読み直す) が相殺し、
結果として「同速」になった、という説明が数値的に筋が通る。

### 仮説 2: simdgroup 12 本の占有率

1 threadgroup の thread 数は 32*gqa = 384 (12 simdgroup)。
`eligible()` (`prefill_attn.py:358-361`) が明記するとおり Apple GPU の
threadgroup 上限は 32 simdgroup = 1024 thread なので、384 は上限の 37.5%
しか使っていない。grid 全体では threadgroup 数が S x kv_head = 2048 x 2 =
4096 あるので GPU 全体の occupancy 自体は埋まるはずだが、**1 threadgroup
あたりの仕事が小さい** (タイル幅 bk=16 列、`_TARGET_COLS=16` /
`_tile_blocks`) ため、K/V タイルの読み込み (2 回の `threadgroup_barrier`)
と実際の計算 (16 列 x 8 dpl の内積) の比率が悪い。ntiles ≈ 2048/16 = 128
タイルを 1 クエリあたり通るので、バリア待ちのオーバーヘッドが計算時間に
対して無視できない可能性がある。ただし仮説 1 (総バイト数) より根拠が弱い
(occupancy 自体は理論上足りているため) — 優先度は仮説 1 の次に置く。

### 仮説 3: online softmax の縮約の形 (scalar reduction vs matrix fragment)

QK スコアの計算はタイル内の 16 列を 1 列ずつ順に処理し、各列で
head_dim を 8 回 (dpl) に分けて `simd_sum` で 32 レーンにわたる縮約を
かける (`prefill_attn.py:225-238`)。1 タイルあたり `simd_sum` 呼び出しは
16 列 x 8 回 = 128 回。相手 (mlx-serve `msv_attn_qsa256`、
`~/dev/mlx-serve/src/transformer.zig:2709-2870`) は `msv_mma` という
simdgroup 行列演算のフラグメント積 (Apple GPU の行列乗算命令パスと思われる)
を使い、1 命令で 8x8 相当のタイルを処理する。`mx.fast.scaled_dot_product_attention`
自体もこの種のハードウェア行列パスを使っている可能性が高い (未確認)。
**per-column scalar dot product + simd_sum** は Apple GPU の ALU
スループットに対して非効率な可能性が高く、削った列数の得を計算効率の
損が相殺している、という仮説 3。前回セッションの docstring
(`prefill_attn.py:223-224`) 自体が「ncol で回すと動的添字になり thread
ローカルへ落ちる」ことを避けるための定数長ループだと書いており、当時から
この部分の非効率は意識されていた。

### 仮説 4 (副次的): bool → 添字化の 3 パスが毎クエリ x 毎 kv head で重複

`keep_row` (`prefill_attn.py:140`) を昇順の添字列 `tg_sel` に詰める処理
(数える → 排他的前置和 → 詰めて書く、`prefill_attn.py:142-176`) は
**クエリごとに 1 回**、かつ **kv head ごとに 1 回** (grid.z = B*n_kv) 実行
される。しかし `keep_block` (indexer の選択結果) は kv head に依存しない
(`QSAIndexer` は head を跨いで共有、`select_blocks` の戻り値に head 軸が
無い) ので、**kv_head=2 の場合、同じ添字化を毎回 2 回計算している**。
n_blocks ≈ 4352 (17k) を 12 simdgroup で分担しても 1 simdgroup あたり
364→384 (32 の倍数に丸め) 回の反復と 2 回の `threadgroup_barrier` が
S=2048 クエリ x kv_head=2 = 4096 回発生する。これ自体の絶対時間は仮説 1
ほど大きくないと見積もるが、**新設計でホスト側 1 回の演算に潰せる**部分
なので (c) で解消する。

## (b) 新設計: クエリ 4 本 (または 8 本) の union をホストで作る

### 変える点

1. **クエリ T 本 (T=4 または 8) を同じ threadgroup に載せる。** これが
   仮説 1 への直接の対策 — K/V タイルの読み出しを T クエリで共有する。
2. **union はホスト側 (純 MLX op、GPU 上だが 1 回のバッチ演算) で作る。**
   `keep_block` (B,S,n_blocks) を T 行ごとに `mx.any` で潰し、昇順添字列
   に変換してカーネルへ渡す。これで仮説 4 の3パス compaction がカーネル
   内から消える (kernel は device メモリの `union_idx` を読むだけ)。
3. **simdgroup 数は T に依存させない。gqa=12 のまま。** 「12 q head x 4
   クエリ = 48 行」という言い方に引きずられて「1 行 = 1 simdgroup」で
   48 simdgroup を用意する設計は **不可**: threadgroup の thread 数上限は
   32 simdgroup = 1024 thread (`prefill_attn.py:358-361` の根拠を再利用)
   であり、48 simdgroup (1536 thread) は上限超過。正しい対応は
   **1 simdgroup = 1 q head のまま (12 simdgroup)**、各 simdgroup が
   割り当てられた head について **T 行を順番に処理する** (T 個の
   online-softmax 状態 `(m, l, acc[dpl])` をレジスタに保持し、K/V タイルは
   T 行で共有して 1 回だけ読む)。「48 行を simdgroup で分担」は
   *48 個の (head, row) ペアを 12 simdgroup が 4 個ずつ引き受ける* という
   意味に読み替える。T=8 なら 96 ペアを 12 simdgroup が 8 個ずつ。
   T x head_dim の acc レジスタ数 (T=4: 4x8=32 float、T=8: 8x8=64 float)
   は現実的な範囲。
4. **行ごとのマスクが (T=1 では不要だったが) 必要になる。** T=1 のときは
   「union = そのクエリ自身の選択」なので追加マスク無しで正しかった
   (`prefill_attn.py:41-48` の可視性根拠)。T>1 では union は T 行の
   選択の **和集合**であり、各行は union の一部しか実際には選んでいない。
   そのため union 列に対して行ごとの小さい bool マスク (`row_keep`,
   T x U_pad ビット) が要る。ただしこれは **kv_len に比例しない**
   (T*U_pad 止まり、T=4 で最大 T*block_topk=2048 ビット = 256 バイト) ので
   「dense マスクへの逆戻り」ではない — union 幅の小さい表を読むだけ。
   端数ブロック (tail) は従来どおりマスク無しの因果窓の算数で済む
   (行ごとに `q_col` が違うだけ)。

### 読み出し量の見積もり (T の効果)

union 化の効果は `true_union_ratio(T)` (T クエリの選択ブロックの和集合が
n_blocks に占める割合) で決まる。実測 (`bench/results/gather-union-stats-true.json`、
17k 前後、compress_ratio=4):

| T (クエリ数/タイル) | true_union_ratio (全体平均) |
|---|---|
| 32 | 0.665 |
| 64 | 0.737 |

読み出しバイト数は概算で

```
bytes(T) ∝ (S/T) x kv_head x true_union_ratio(T) x n_blocks x cr x head_dim x itemsize x 2
         = S x [true_union_ratio(T) / T] x (定数)
```

なので **`true_union_ratio(T)/T` を最小化する T** が読み出し最小。
T=1 (旧カーネル) は `true_union_ratio(1) = block_topk/n_blocks ≈
512/4352 ≈ 0.1176` (n_blocks は 17k 換算)、`ratio/T = 0.1176`。
T=32 は `0.665/32 = 0.0208` (T=1 比で読み出し 17.7%)。T=64 は
`0.737/64 = 0.01152` (T=1 比で読み出し 9.8%、**T=32 よりさらに少ない**) —
union 比の伸びより T の伸びの方が効くので、この 2 点だけを見ると
「大きい T ほど読み出しは減る」という、直感に反する形になる
(タイルを大きくするほど union が dense に近づくが、その分 T で割った
効果の方が支配的)。

**課題文の「T=4 で kv の 2 割台」という見込みについて**: 上の 2 点
(T=32→0.665, T=64→0.737) だけから T=4/8 を外挿すると、少なくとも 2 通り
のやり方で違う値が出る:

- log2(T) に線形と仮定し、T=32→64 の傾き (0.072 / doubling) をそのまま
  遡って使うと T=8 で ≈0.52、T=4 で ≈0.45。
- T=1 (0.1176) と T=32 (0.665) の 2 点を log2(T) 線形で結ぶと、
  平均の傾きは 0.1095/doubling で、T=4 (log2=2) では ≈0.34、T=8
  (log2=3) では ≈0.45。

どちらも「2 割台」より高く出る。実際の union 曲線は「T が小さいほど
1 クエリ追加あたりの union 増分が大きい」凹型である可能性が高く (最初の
数クエリで局所窓 + グローバルブロックの主要部分をすぐ拾い、以降は
重複が増えて伸びが鈍る)、その場合は T=32→64 の傾き (すでに平らな領域)
を使った外挿は**低く**出すぎる。つまり課題文の「2 割台」も、上の外挿の
「34-52%」も、どちらも憶測の域を出ない。**T=4/8 の実測が無いと
読み出し量の期待値そのものが決められない。**これは (b) の設計の
成否を左右する最優先の未知数であり、実装より先に測る価値がある。

**測定の計画 (追加コード不要)**: `tools/gather_union_stats.py` は
`--tiles` にカンマ区切りの任意の整数を渡せる (`tools/gather_union_stats.py:156-161`
の argparse 定義、コードは tile の値をそのまま `enable_gather_attn(model,
tile=tile)` に渡すだけで T=4/8 も無改造で通る)。次にこのレーンへ着手する
セッションは

```
tools/biglock.sh .venv/bin/python tools/gather_union_stats.py \
    --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram \
    --tiles 4,8,16,32 --out bench/results/gather-union-stats-small-tile.json
```

を GPU 解放後に走らせ、`true_union_ratio_mean` (T=4/8) を得てから
`bytes(T) ∝ true_union_ratio(T)/T` の表を埋め直すこと。**この数字が
「T=1 比で読み出しがどれだけ減るか」を決め、ひいては (e) のゲートが
通る見込みがあるかを左右する。**

## (c) ホスト側: union 添字列の構築

`QSAIndexer.select_blocks` の戻り値 `keep_block` (B, S, n_blocks bool) から、
T 行ごとの昇順パディング済み union 添字列を作る。骨子
(`mlxturbo/kernels/qsa_prefill_attn.py` の `build_union_blocks`、実装済み・
CPU 検証済み):

1. S を T の倍数に切り上げてパディング (端数タイルは全 False 行を足す —
   出力は該当行が空 union になるだけで実害は無く、呼び出し側が本来の S
   行だけを後で切り出す)。
2. `union = mx.any(keep_block_padded.reshape(B, n_tiles, T, n_blocks), axis=2)`
   — T 行ごとの和集合 (bool, (B, n_tiles, n_blocks))。
3. **昇順添字化は `argsort(-union)` ではなく `mx.sort` に置き換える。**
   `Attention._gather_tile_attn` (`mlxturbo/_vendor/qwen4_exp.py`) は
   `order = mx.argsort(-union.astype(mx.int32), axis=-1)` で True を
   先頭に寄せるが、True 集合内部の順序が昇順である保証は無い (安定ソート
   依存)。新設計は K/V タイルを列方向に前進させながら読む都合上、
   union_idx が **真に昇順**である方が読み出しの局所性に効く可能性がある
   ので、`key = where(union, arange(n_blocks), n_blocks)` を作ってから
   `mx.sort(key, axis=-1)` する。True の位置には実ブロック添字 (昇順)
   がそのまま並び、False の位置は sentinel (`n_blocks`) が並んで末尾に
   落ちる — ソート順の保証だけで昇順・パディングが同時に手に入る。
4. **U_pad は完全に実行時の値でよく、カーネルの template 定数にする必要が
   無い。** 旧カーネルの `tg_sel` (`prefill_attn.py:135`) は threadgroup
   メモリに置く配列だったのでコンパイル時サイズが要ったが、新設計の
   `union_idx` / `row_keep` はホストで作った device メモリ上の配列を
   そのまま読むだけで、threadgroup メモリには載せない (K/V タイルの
   ステージング用バッファだけが threadgroup メモリを使い、その大きさは
   BK (列タイル幅、T や U_pad と無関係) で決まる)。よってループ回数は
   ふつうの runtime int パラメータ (`params` 配列経由) で渡せばよく、
   `U_pad` の値ごとにカーネルを作り直す必要は無い — `head_dim / gqa /
   cr / T / scale` が同じなら 1 variant で済む。
   `U_pad = min(n_blocks, T * block_topk)` を静的な安全上限としつつ、
   `U_pad = round_up(max(1, U_true.max()), 32)` を **層ごとに 1 回だけ**
   `.item()` で host に取り、実際にそのチャンクで必要な幅まで絞る
   (T=1 旧カーネルの「クエリごとに 2 バリア」より遥かに軽い、GPU→CPU
   同期は 1 回/層/チャンクだけで済む)。
5. `row_keep` (B, S_pad, U_pad) bool: 各行が union 列のうちどれを実際に
   選んでいるか。`keep_block_padded` の最後にダミー列 (常に False) を 1 本
   足してから `take_along_axis(keep_ext, union_idx_broadcast, axis=-1)`
   で作る (sentinel 添字 `n_blocks` がダミー列を指すようにする安全策)。

これらはすべて `mx` の通常演算 (any / sort / take_along_axis /
reshape / concatenate) で構成でき、**CPU で動く** (Metal kernel 不要)。
正しさは `bench/test_qsa_prefill_attn_host.py` で検証済み。

## (d) 正しさの検査計画 (GPU 復帰後)

`tools/verify_prefill_attn.py` の流儀を踏襲する (新規ファイルは今回書か
ない。プランのみ)。

配列レベルの検査は `_keep_block` 相当の乱数生成 (block_end <= q_col の
制約を満たす形) で `keep_block` を作り、`build_union_blocks` → カーネル
出力を、同じ可視集合の dense sdpa+mask (`_dense_reference` と同じ構成)
と突き合わせる形になる。S ∈ {64, 2048}、kv_len ∈ {4096, 16384}、tail
(kv_len が cr の倍数でない) あり/なしの組を通し、head_dim 256 / n_heads
24 / kv 2 / cr 4 を実モデル形状として使う。許容誤差は前回と同じ扱いで
よい (fp32 は絶対 1e-4、bf16 は相対 2e-2、`verify_prefill_attn.py:49-52`
の理由をそのまま踏襲する — online softmax は加算順が変わるだけで、
ビット一致は要求しない)。

モデルレベルの検査は合成 Flash-Next (`verify_batch_cache.build`) で
kernel on/off の logits を比較する。QSA 不活性域 (kv_len <= token_budget)
はビット一致を要求する (`verify_prefill_attn.py:227-231` と同じ規約)。

`tools/vendor_fingerprint.py` は `mx.set_default_device(mx.cpu)`
(`tools/vendor_fingerprint.py:50`) で動くので、このカーネルの
`eligible()` は GPU 判定で常に False になり、一次検査としては通らない
(既存の 5 カーネルと同じ制約で、同ファイルの docstring に明記されている)。
触った後は必ず `tools/gpu_fingerprint.py` (`tools/biglock.sh` 経由) まで
実行すること。prefill を触る変更なので、最終ゲートは
`tools/verify_prefill_bitident.py` (実モデル、4 分) まで (CLAUDE.md の
記述どおり)。

## (e) ゲート

1. **合成テンソル単体マイクロベンチ**: kv=16384、S=2048、head_dim 256、
   gqa 12、bf16 で、このカーネルが dense sdpa (同じ形状、mask 無し causal)
   の **2 倍以上速い**こと。**届かなければ書き直さず畳む** (課題文の
   反転条件をそのまま採用)。この基準は (b) の読み出し量見積もりが
   T=4/8 で実際にどれだけ効くか (まだ未測) に強く依存するので、
   T を振って (4, 8, 16 あたり) 最良点を探ってからこの基準に当てる。
2. **in-model 判定**: 17k / 50k の prefill_s、KLD (現行比 +0.0005 以内、
   `bench/quant_eval.py compare`)、tok/round。CLAUDE.md の計測作法
   (1 プロセス内 ABBA、複数プロンプト x 512 トークン平均) を厳守。
   17k で -5% 見込み (LANES.md の記載)、50k で -10% 以上が目標
   (LANES.md の記載)。
3. 上記 1 が通っても 2 が反転条件 (17k で改善が見えない、KLD 超過) に
   触れたら、前回同様「実装は残すが既定 off」で畳む
   (`docs/BACKLOG.md` の負けカーネルの扱い方 = 記録して残す)。

## (f) 工数の見積もり

GPU 検査からの見積もり (このセッションはここまで未着手):

| 作業 | 見積もり |
|---|---|
| ホスト側 union 構築の GPU 実測確認 (CPU テストは今回済み、GPU での no-op 確認) | 0.5 日 |
| T=4/8 の union 統計実測 (`gather_union_stats.py --tiles 4,8`、コード変更なし) | 0.5 日 (biglock 待ち込み) |
| Metal カーネル本体の実装 (構造は今回の骨組みから、行ごとマスク・T 本 online softmax の実コード化) | 1-1.5 日 |
| `tools/verify_qsa_prefill_attn.py` (新規、(d) のプラン通り) の作成と合格 | 0.5-1 日 |
| 合成マイクロベンチ (e-1) | 0.5 日 |
| in-model A/B (e-2、17k/50k、biglock 込み) | 0.5-1 日 |
| **合計 (1 回で通った場合)** | **3.5-5 日** |
| 仮説 3 (scalar reduction) が主犯で simdgroup 行列演算への書き直しが要る場合の追加 | +2-3 日 |

最大のリスクは (b) の読み出し量見積もりが T=4/8 の実測で外れること
(「2 割台」ではなく「3-5 割」だった場合、T=1 に対する改善幅が
読み出し軸だけでは 2-3 倍止まりで、仮説 3 のスカラー縮約の非効率が
上乗せされると (e-1) の「dense の 2 倍以上」に届かない可能性がある)。
そのため **実装より先に T=4/8 の union 統計を取ることを最優先の次の一手
とする。**
