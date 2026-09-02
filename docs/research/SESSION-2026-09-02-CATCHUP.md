# 対 mlx-serve 追い上げセッション (2026-09-02、Fable 親)

`docs/NEXT-SESSION-PROMPT.md` を起点に、mlx-serve との差を「実測で埋めるか、
埋まらない理由を実測で示す」までを記録する。判断が反転する条件は測る前に書く。

## 最初に見つけたこと (測る前、ログとソースから)

1. **前セッションの「単独測定」は冷えていない。**`bench/self_snapshot.py` は
   文脈ごとに `long_prompts(tok, c, [q])[0]` を呼んでいて、窓は常に池の先頭
   (`ids[0:win]`)。rep 1 と rep 2 は本文が同一 (末尾の問いだけ違う)、4k / 17k /
   50k の本文は互いに接頭辞。mlx-serve のログ (`~/.mlx-serve/logs/mlx-serve-8140.log`)
   で確認:
   - 17k rep 1: `reused 3802/16828` (4k の接頭辞が当たる)、rep 2: `reused 16797/16834`
   - 50k rep 1: `reused 16803/49828`、rep 2: `reused 40960/49834`
   - 相手の真の冷 prefill は **605-670 tok/s** (17k で約 27s、50k で約 80s)
   - うちの engine 直叩き 17k は 35.3s = 478 tok/s。**差は 2 倍ではなく 1.3-1.4 倍**
2. **thinking が揃っていない。**mlx-serve は qwen4_exp で thinking 既定 off
   (`thinking=false` がログに出る)、うちはテンプレート既定で on。生成する文の種類が
   違うので MTP の受理率も違う。両者とも OpenAI 標準 `reasoning_effort` を読むので
   `"none"` で揃える。
3. **engine 直叩きと HTTP 経由で短文脈 decode が 35% 違う。**`tools/decode_ab.py`
   (ctx 62、512 トークン) は 16.4 ms/tok = 61 tok/s、HTTP 経由の snapshot は
   45.3 tok/s = 22.1 ms/tok。プロンプト差 (テンプレート・thinking) かサーバー経路の
   費用かは未分離。相手の HTTP 経由は 55 tok/s。
4. 相手の構造 (scout 実読): mlx-c 経由で同じ MLX 0.32.2、1 ステップ 1 同期、
   decode 幅の MoE は「take + batched quantized_matmul」+ 自前融合カーネル、
   GDN decode は 1 カーネル、HC 読みは融合 3 カーネル (B*S==1 のみ)、
   prefill は chunk 8192 + `gated_delta_blocked_seq` (oMLX 由来の Metal)。
   MTP は depth 6 の適応制御で 17k-50k の avg_per_round 0.75-0.94、
   round_ms 35-42。

## 反転条件 (測る前に宣言)

ハーネスを直して (窓を重ねない、thinking off で揃える、ログを残す) 両者を測る。

- **prefill**: 相手の 17k 冷 TTFT がうちの 1.25 倍以上速ければ「差は実在」。
  1.1 倍未満なら前セッションの差はハーネスの産物で、prefill レーンは畳む。
- **decode 短文脈**: HTTP 経由でうちが相手に負けていて、かつ engine 内部の
  `decode=` と client 側の tok/s が 5% 以上ずれていれば、サーバー経路が犯人。
  ずれが 5% 未満なら犯人はプロンプト差 (受理率) で、`tools/server_overhead_ab.py`
  (同一 ids で 1 プロセス内 ABBA) で決着させる。
- **decode 長文脈**: 相手の 17k decode がうちの 1.2 倍以上なら、ラウンド費用と
  tok/round に分けて追う (相手のログに round_ms と avg_per_round がある)。

## 直したハーネスで測った現在地 (2026-09-02 11:09-11:25、serve → turbo の順、thinking off)

`bench/self_snapshot.py` を直した (窓を重ねない、`--thinking off` で両者に
`reasoning_effort: none`、サーバーログを `bench/results/logs/` に残す、温め 2 段)。
結果 `bench/results/self-snapshot-{serve,turbo}-0902c.json`。

| 文脈 | 冷 TTFT serve / turbo | 比 | 温 TTFT serve / turbo | decode serve / turbo | 比 |
|---|---|---|---|---|---|
| 0 | 0.17 / 0.49 | 2.8 | 0.72 / **0.45** | 53.7 / 47.5 | 0.88 |
| 4k | 5.77 / 8.28 | 1.43 | 0.87 / **0.46** | 55.3 / 48.8 | 0.88 |
| 17k | 29.2 / 37.6 | 1.29 | 0.92 / **0.51** | 56.0 (46.8, 65.3) / 41.0 | 0.73 |
| 50k | 108 / 163 | 1.51 | 22.8 / **1.33** | 30.8 / **34.4** | **1.12** |

読み方:

- **prefill の差は 1.3-1.5 倍** (2 倍ではない)。前セッションの 2 倍は接頭辞キャッシュの
  産物だった。反転条件 (1.25 倍以上なら実在) は満たすので prefill レーンは続ける。
- **温 TTFT はうちが 2 倍速い。**50k では相手の prefix cache (既定 2GB) が入り切らず
  22.8s、うちは 1.33s。
- **decode は短文脈で 12%、17k で 27% 負け。50k では勝つ** (相手が 30.8 まで落ちる)。
- サーバー内部の `decode=` と client 側の差は 2-7% で、**SSE 経路は犯人ではない。**
  engine 直叩き 61 tok/s との差は主に tok/step (直叩き 2.22、HTTP 1.75-1.93) で、
  thinking off の本文はドラフトが当たりにくい。相手も同じ条件で avg_per_round
  0.6-0.84 (= 1.6-1.84 tok/round)。**差はラウンド費用: うち約 36 ms、相手約 33 ms
  (短文脈)、17k でうち約 40 ms、相手約 37 ms。**
- 相手の 17k decode は rep 間で 46.8 / 65.3 と大きく振れる (テキスト運)。

## n-gram サイドカーの CPU 費用 (2026-09-02、GPU 不使用の実測)

`StreamNGram._gather_pread` (行ごとに future 1 つ):

| 行数 | 冷 | 温 (ページキャッシュ) |
|---|---|---|
| 48 (decode 1 ラウンド) | 0.6 ms | 0.35 ms |
| 32768 (prefill 1 チャンク) | 320 ms | **250 ms** |

温でも 1 行 7.6 us = Python の future 費用。**17k prefill で約 2.5 s、50k で約 7 s が
CPU で消えている** (`tools/prefill_anatomy.py` は `ddalcu-ngram-sep` (RAM 常駐) で
測っていたので、この項は解剖に出ていなかった)。加えて `__call__` 先頭の
`np.array(gid)` がチャンクごとに GPU を止める。手: 行 id はプロンプト全体で
最初から分かるので、バックグラウンドで先読みする (実装中)。

## n-gram layout の decode 差 (2026-09-02、1 プロセス内 ABBA、`tools/decode_ab.py --knob ngram-layout`)

短文脈 3 本 x 512: interleaved (A、本番) 37.9 ms/round、separate (B、RAM 32GB) 37.0 ms/round。
**差 2.4% (0.9 ms/round)。**tok/round は同一 (2.248)、出力一致。`StreamNGram.__call__` の
GPU→CPU 同期と pread の費用はラウンドあたり約 1 ms で、decode の犯人ではない。
結果 `bench/results/ngram-layout-short.json`。

## 幅 S の verify forward 費用 (2026-09-02、`tools/verify_width_cost.py`、interleaved n-gram)

同じ状態から幅 S のトークン列を流して forward 1 回の壁時計 (中央値、n=27、回文順)。
`round` は draft chain + forward + rollback。

| S | ctx 62 forward / round | ctx 16868 forward / round |
|---|---|---|
| 1 | 25.2 / 27.1 | 30.4 / 30.5 |
| 2 | 33.1 / 34.7 | 39.3 / 41.6 |
| 3 | 39.8 / 43.4 | 47.9 / 51.3 |
| 4 | 47.1 / 52.4 | 55.1 / 60.2 |

読み方: 1 行足す費用は短文脈 7.3 ms、17k で 8.5 ms。重みの読み出しで説明できるのは
専門家 10 人分の約 1.2 GB = 3 ms だけで、残り 4-5 ms は行数に比例する別の何か
(GDN の位置ごとの状態、sort 経路の MoE、indexer)。相手 (M4 Max) は S=1 16.0 /
S=2 22.4 / S=4 30.6 で 1 行 4.9 ms。**S=1 の固定費 25.2 ms が相手より大きい**
(この機体での相手の値は `MLX_SERVE_DECODE_FWD_UBENCH` で測定中)。
17k で depth 2 が得にならない理由はここ: S=3 の round 51.3 ms を tok/round 1.97 で
割ると 26.0 ms/tok、S=2 の 41.6 / 1.63 = 25.5 ms/tok と同着。

## RAM 常駐 n-gram は 17k で 41% 遅い (2026-09-02、`--knob ngram-layout --ctx 17000 --prefill-once`)

interleaved 45.9 ms/round に対し separate (RAM 32GB) は 64.9 ms/round。
モデル 91GB + 32GB でメモリ圧がかかる。**前セッションの 17k decode の数字
(`tools/decode_ab.py` の既定例は `--ngram ddalcu-ngram-sep`) はこの構成で
取られていて、本番より 4 割遅い状態を見ていた可能性がある。**
今後の長文脈の計測は interleaved で行うこと。

## 相手の forward 費用をこの機体で測った (2026-09-02、`MLX_SERVE_DECODE_FWD_UBENCH=30`)

mlx-serve 自身の道具。forward 1 回 = build (CPU でグラフ構築) + eval (GPU)。
うちの値は `tools/verify_width_cost.py` (上の表)。

| S | KV | mlx-serve 合計 (build + eval) | ops/forward | mlxturbo forward | 差 |
|---|---|---|---|---|---|
| 1 | 0 | 20.4 (2.2 + 18.2) | 3940 | 25.2 | +4.8 ms (+24%) |
| 2 | 0 | 26.5 (2.6 + 24.0) | 8315 | 33.1 | +6.6 (+25%) |
| 3 | 0 | 32.3 (3.6 + 28.7) | 8315 | 39.8 | +7.5 (+23%) |
| 1 | 17k | 22.8 (1.9 + 20.9) | 4561 | 30.4 | +7.6 (+33%) |
| 2 | 17k | 31.1 (2.9 + 28.1) | 9017 | 39.3 | +8.2 (+26%) |

読み方:

- **decode の差は forward そのものにある。**ラウンドの周辺 (draft、rollback、SSE) ではない。
- 相手の CPU 側は 2-4 ms (Zig)。GPU 側だけでも S=1 で 18.2 ms で、うちの 25.2 ms
  (CPU 込み) より 7 ms 小さい。うちの build / eval の分離は `tools/forward_split.py` で測る。
- 文脈 0 → 17k の伸びは相手 +2.4 ms、うち +5.2 ms。**長文脈の decode 罰がうちは 2 倍**
  (QSA の indexer と attention)。
- 相手の 1 行追加は 6.0 ms、うち 7.3 ms。行あたりの差は小さく、**固定費の差が主**。

## forward を build (CPU) と eval (GPU) に分けた (2026-09-02、`tools/forward_split.py`)

| S | ctx | staged 合計 | plain build / eval / 合計 | 相手 build / eval / 合計 |
|---|---|---|---|---|
| 1 | 62 | 23.3 | 4.1 / 24.0 / 28.0 | 2.2 / 18.2 / 20.4 |
| 2 | 62 | 28.8 | 4.2 / 29.5 / 33.7 | 2.6 / 24.0 / 26.5 |
| 3 | 62 | 34.9 | 4.5 / 35.7 / 40.2 | 3.6 / 28.7 / 32.3 |
| 1 | 16868 | 30.5 | 5.3 / 31.3 / 36.7 | 1.9 / 20.9 / 22.8 |
| 2 | 16868 | 38.3 | 5.4 / 39.2 / 44.6 | 2.9 / 28.1 / 31.1 |

読み方:

- **差は GPU の実行時間そのもの。**うちの eval だけで S=1 短文脈 24.0 ms、相手の
  eval は 18.2 ms (+32%)。Python の構築 4-5 ms は段階投入でほぼ隠れている
  (staged 合計 ≈ plain eval)。
- **文脈 62 → 17k の伸び: うち +7.3 ms、相手 +2.7 ms。**QSA の decode 経路
  (`_gather_tile_attn` の any/argsort/take_along_axis x2/repeat/concat + マスク付き
  sdpa と、indexer の pooled スコア + argpartition) が 12 層で約 20 dispatch ずつ
  積まれ、相手は融合 1-2 カーネル。
- 短文脈の固定差 5.5 ms は文脈に依らない部分 = 融合の有無 (相手: GDN step /
  qk_norm_rope / gdn_prework / normgate / add_rmsnorm / attn_out_gate / router /
  hc_read x3 が全部融合)。うちは HC のみ融合で、他は個別に試して ±0〜-0.9% だった。
  **個別には小さくても合計で 5 ms になりうる**。部品別 (`tools/module_costs.py`) と
  相手の `[decode-prof]` で突き合わせる。

## 相手の prefill トレース (`[prefill-trace]`、chunk 4096)

| tokens | chunks | chunked ms | ms/tok |
|---|---|---|---|
| 4018 | 1 | 5753 | 1.43 |
| 16828 | 4 | 29232 | 1.74 |
| 49827 | 13 | 98933 | 1.99 |

うち (HTTP 冷 TTFT から): 4k 2.17 / 17k 2.23 / 50k 3.27 ms/tok。
**17k までは attention の伸びではなく素の 1 トークン費用が 1.5 倍**、50k で
attention/indexer の伸びが乗る (うち +47%、相手 +14%)。

## 部品別の単体費用 (2026-09-02、`tools/module_costs.py`、短文脈、interleaved n-gram)

| 部品 | T=1 | T=3 | バイトの下限 (393 GB/s) |
|---|---|---|---|
| MoE 48 層 (router+gather+shared) | 8.57 | 13.78 | 3.3 (T=1) |
| GDN 36 層 | 6.36 | 7.36 | 2.6 |
| HC 97 回 | 3.72 | 4.02 | ≈0 (38 us/回 = 起動レイテンシ) |
| attention 12 層 | 1.48 | 1.79 | 0.5 |
| lm_head (8bit) | 1.96 | 2.00 | 1.6 |
| 単体の総和 | 22.09 | 28.93 | |
| 全体 (実測) | 27.86 | 38.08 | |

下限との差 (T=1): MoE 5.3、GDN 3.8、HC 3.7、attn 1.0 = 約 14 ms が
「小さいカーネルの直列レイテンシ」。相手の GPU 18.2 ms も下限 7.9 ms に対して
10 ms の同種の費用を払っている。**差 5.5 ms は、相手が融合で減らした
dispatch 数の分と読める。**

過去の融合 A/B (`bench/results/*-ab2.json`) は全部 `fired={}` で、
発火の確認が無い (gdn-prework は long で +5%、moe-route は +4% と「遅くなる」側に
出ている)。発火カウンタを付けて取り直す価値がある。

## prefill の解剖を本番構成 (interleaved n-gram) で取り直した (2026-09-02 12:10、**熱い状態**)

`tools/prefill_anatomy.py --ctx 17000 --reps 2`。GPU を 1 時間回した後なので絶対値は
冷えた状態 (段 P0: 3617 ms/チャンク) より 1.8 倍遅い。**割合だけ読む。**

| 部品 | chunk 0 (kv=2048) | chunk 7 (kv=16384) |
|---|---|---|
| MoE 48 層 | 2000 (30%) | 2849 (41%) |
| GDN 36 層 | 1178 (18%) | 1487 (21%) |
| Attention 12 層 (indexer 込み) | 1058 (16%) | **1954 (28%)** |
| HC 97 回 | 451 (7%) | 618 (9%) |
| PLE (n-gram 込み) | 41 | 57 |
| 部品和 | 4727 | 6965 |
| 壁時計 (layer-major) | 6506 | 6883 |

- chunk 0 の「部品和 - 壁時計 = -1965 ms (-29%)」は最初のチャンクの JIT と n-gram 冷読みで、
  chunk 7 では +1.2% と合う。
- **attention は kv に比例して伸びる** (chunk 0 → 7 で +900 ms)。dense sdpa にブールマスクを
  渡す経路なので、QSA の疎性が計算量に効いていない。相手は選択ブロックだけ読む
  (`msv_attn_qsa256`、oMLX PR #3244 由来)。17k で平均 1 チャンク約 1.5 s x 8 = 12 s
  (熱い状態) が attention。50k ではこれが支配的になる (うちの 50k は 17k → 50k で
  ms/tok +47%、相手は +14%)。
- MoE の chunk 0 → 7 の差 (+850 ms) は kv に依らない部品なので熱ドリフト。
  **熱い状態での絶対値は使わない。**

## prefill でチャンクの外にある時間

段 P0 のチャンク壁時計 3617 ms (冷) x 8.2 = 29.7 s に対し、engine 直叩きの 17k prefill は
35.3 s、HTTP は 37.6 s。**6-8 s (16-20%) がチャンクの外**にある。候補: グループ境界の
`mx.eval`+`clear_cache`、checkpoint の snapshot、n-gram 行取得、末尾チャンク、MTP priming
(PRIME_WINDOW 2048)。`MLXTURBO_PREFILL_TRACE=1` を足して切り分ける (実装中)。

## 熱の影響を相手側でも確認 (2026-09-02 12:18)

相手の `fwd-ubench` S=1 KV=0 を取り直すと 22.8 ms (build 1.8 + eval 21.0)。11:43 の 20.4 ms
(eval 18.2) から **+12%**。うちの forward_split (eval 24.0) は 12:00 頃の同じ熱い状態で
取ったので、**同じ熱状態での差は 5.5 ms ではなく 3 ms 前後 (+14%) の可能性がある。**
交互に取り直す (chain9)。

## n-gram 先読み・バッチ化の A/B は熱で判定不能 (2026-09-02 12:21-12:32)

`--knob ngram-batch` (A = バッチ pread、B = 行ごと) の 17k、3 本 x ABBA: prefill_s が
32.8 → 55.5 s へ単調に伸び (熱)、しかも行キャッシュが条件をまたいで残るので B が
全部 hit した。**判定不能。**`ngram_sync_ms` は 17k prefill 全体で 77-107 ms
(44 回の GPU→CPU 同期) で小さい。先読み (`--knob ngram-prefetch`) は 2 本目以降で 0%。

判断: バッチ pread はビット一致で CPU 単体 30% 速いので残す (既定)。先読みは
取り分が出ないので **既定 off** (`MLXTURBO_NGRAM_PREFETCH=1` で有効)。
prefill の本丸はここではない。

## GPU の回し方を変える (ユーザー指摘)

連鎖で回し続けて 17k prefill が 33 s → 55 s まで落ちた。以後:
- 8 分冷ましてから、**相手 (ubench) とうちを交互に同じ熱状態で**取る
- 重い計測 (17k x 複数本) は 1 回ずつ、間に休みを入れる
- 絶対値は冷えた最初の 1 本だけを信じ、それ以降は A/B の差だけ

## 冷えた状態で交互に取った forward S=1 (2026-09-02 12:41、8 分冷却後)

| | mlx-serve (fwd-ubench) | mlxturbo (forward_split plain) |
|---|---|---|
| ctx 0 / 62 | 19.9 ms (build 1.8 + eval 18.1) | 28.3 (build 4.3 + eval 24.1)、staged 23.2 |

**熱の影響を除いても GPU 時間の差は約 6 ms (+33%)。**うちの eval は熱くても冷えても
24.0 ms でほぼ動かず、相手は 18.1 (冷) ↔ 21.0 (熱) と動く。

## decode の QSA 経路について訂正 (scout の実読)

相手は decode/verify 幅 (S<16) で自前カーネルを使っていない: ブロック選択は
MLX の matmul/relu/sum/argpartition、attention は **dense bool マスク + MLX 純正 sdpa**
(`splitMaskedSdpa256` で行数 x gqa <= 32 に割る)。自前の `msv_attn_qsa256` は prefill
専用 (qL >= 16、kv > 8192)。つまり **うちの 17k での +7.3 ms は「融合カーネルが無い」
せいではなく、gather 経路 (段 3(b)) が相手の dense マスク経路より重い**か、indexer
の作りの差。過去の A/B (`gather-attn-17k.json`) でも 17k では dense (B) が 5% 速く、
gather が勝つのは 25k 以上。17k の forward を gather off でも測る。

## 冷えた状態の 17k forward S=1 (2026-09-02 12:44-12:48、交互)

| | mlx-serve | mlxturbo |
|---|---|---|
| KV 0 | 19.9 / 19.7 (eval 18.1 / 17.9) | plain eval 24.1、staged 23.2 |
| KV 17k | 23.0 / 22.8 (eval 21.1 / 20.9) | plain eval 31.5 (nocap 31.1)、staged 30.6 |
| 文脈の罰 | **+3.0 ms** | **+7.4 ms** |

短文脈で +6 ms、17k で +10 ms。文脈の罰はうちが 2.5 倍。

## prefill のタイル gather は「効かない設計」だった (2026-09-02、`tools/gather_union_stats.py`)

17k、tile 0/256/64/32 の全部で union_ratio = 1.000。`Attention._gather_tile_attn` は
`U = min(n_blocks, T * block_topk)` を和集合の大きさとして使う (真の和集合を数えない)。
17k は n_blocks ≈ 4218、block_topk = 512 なので T >= 9 で U = n_blocks、**タイルに
割っても全ブロックを集めて dense sdpa に渡している**。前セッションの
「tile=256 で prefill_s が縮まない」「attention はほぼ詰み」は、この実装の帰結。
真の和集合の大きさを測り直す (フックに true_u を足す)。小さければ、タイルごとに
真の union だけ集める経路 (同期 1 回/タイル/層) で prefill attention を kv に依らず
一定に近づけられる。

## oMLX 移植の GDN blocked-seq Metal カーネル (2026-09-02 12:56、`tools/verify_gdn_metal.py`)

逐次カーネルとの差: y の相対誤差 1-2e-5、state 1e-8 (加算順の差の範囲)。
T=2048 (B=1、Hk=16、Hv=48、Dk=Dv=128、bf16) の壁時計、交互 20 回:
逐次 min 4.98 / mean 6.04 ms、Metal 移植 min 3.14 / mean 3.81 ms (**x1.59**)。
36 層 x 8 チャンクで 17k prefill あたり約 0.6 s (1.8%) の見込み。in-model A/B
(`--knob gdn-metal`、prefill_s) で採否を決める。

## 17k decode の文脈罰は gather 経路の有無に依らない (2026-09-02 12:58、同じ熱状態)

forward_split S=1 ctx 17k: gather off (dense マスク) eval 32.3 ms、gather on 31.3 ms。
どちらも相手 (21.0) より 10 ms 大きい。**罰は両経路に共通の部分** (indexer:
pooled スコア + argpartition + keep マスク構築、KV/indexer キャッシュの更新) にある。
xctrace のカーネル区間で相手と突き合わせる。

## prefill のフェーズ別トレース (2026-09-02 13:02、17k = 16869 tok、熱い状態、4 本目)

`MLXTURBO_PREFILL_TRACE=1`。壁時計 49.3 s (熱い。冷えていれば 35 s 前後)。

| 区間 | ms | 1 トークンあたり |
|---|---|---|
| group build+async i=0 g=4 (8192 tok) | 20106 | 2.45 |
| group eval | 1082 | |
| group build+async i=8192 g=3 (6144 tok) | 17165 | 2.79 |
| group eval | 992 | |
| **tail forward 485 tok (端数チャンク)** | **2180** | **4.49** |
| **tail forward 2048 tok (最終チャンク、split + checkpoint + lm_head 込み)** | **7540** | **3.68** |
| prime (MTP、2048) | 229 | |
| first token | 8 | |
| 区間和 - 壁時計 | -7 | |

読み方: チャンクの外に「見えない 6-8 s」があるわけではなく、**末尾 2 チャンクが
高い**。理由は (a) 末尾ほど kv が大きく attention が伸びる (chunk 7 の解剖: attention
1954 ms)、(b) 端数チャンク 485 tok は MoE の行数が少なく (専門家あたり 9.5 行)
効率が落ちる、(c) 末尾は chunk-major で MoE の連結が無い (2048 行 = 専門家あたり
40 行) うえに段階投入も無い (解剖: chunk-major 6883 vs layer-major 6505)、
(d) split / checkpoint / lm_head。

手: 端数チャンクをグループに入れる (~1 s)、最終チャンクも layer-major で流す
(~0.4 s)、attention を真の union で集める (kv 依存を消す。最大)、GDN Metal (0.6 s)。

## GDN Metal カーネルの in-model A/B (2026-09-02 13:37-13:48、17k x 3 本、熱い状態)

`--knob gdn-metal --only long --ctx 17000 --tokens 8`: prefill_s A (Metal) 47.86 s、B (逐次)
48.60 s、**-1.5%** (本ごとに -4.4 / +1.9 / -1.8%)。発火 324 回 (36 層 x 9 チャンク)。
単体 1.59 倍でも prefill の 5% の部品なので取り分はこの程度。bit 一致ではない
(相対 1e-5) ので、既定 on にするなら KLD (`bench/quant_eval.py compare`) を通す。
いまは既定 off のまま。

## コードから読める「未計測の疑い」の棚卸し (2026-09-02 13:55)

1. グループ境界の `mx.clear_cache()` → 既存の A/B (`prefill-pipeline-ab.json`、非同期 + clear 無し)
   で -0.6%。**閉じる。**
2. decode の GDN 層の `mx.contiguous` (36 回) と `mx.concatenate` (114 回、attention 層で 1 層 6 回)
   → xctrace のカーネル区間で相手と突き合わせる (未)。
3. MoE の sort 経路のタイル水増し (1.4-1.9 倍) → 相手も同じ op 列。差の主因なら
   行の並べ方 (グループ内連結) の違いになる (未)。
4. 最終チャンクだけ chunk-major (解剖 6.9 s 対 6.5 s) → 畳むには checkpoint と lm_head の
   作りを変える (未)。

## 読解で見つけた非対称 3 つ (2026-09-02 14:20、scout 3 本)

1. **QSA マスクの表現** (`Attention._final_mask`): うちは bool の `sparse` を
   `where(sparse, 0, finfo.min)` の加算マスクにして sdpa に渡す。MLX 0.32.2 の sdpa vector
   カーネル (`sdpa_vector.h:100-110`) は float マスクなら `use_key = fmask >= finite_min`
   でキーの読み込みを飛ばすが、**うちの値は finite_min そのものなので `>=` が真になり
   飛ばない** → 17k のキー全部を読む。相手は bool のまま渡す (`qsaMaskFromBlockSel`)。
   前セッションの `--knob bool-mask` は 17k で ms/tok -7% と出ていたが採用されていなかった。
   steel (prefill 幅) の attention はマスクで飛ばさない (`steel_attention.h:329-352` は
   フラグメント単位で加算するだけ) ので、**効くのは decode 側だけ**。→ 既定を bool にする (実装中)。
2. **QK-norm + partial RoPE**: 相手は `fusedQkNormRope256` (1 dispatch、rd=64) で
   rms_norm x2 + rope x2 を畳む。うちは rms_norm x2 + transpose x2 + cos/sin 生成 +
   `_rope_partial` x2 (slice x2 + concat x2 ずつ) = 約 14 op/層 x 12 層。文脈に依らない固定費。
   `mx.fast.rope` (partial dims 対応) に置き換えれば数 op に減るが、mrope_interleaved の
   等価性を確認してから (数値は動きうる)。
3. **GDN 層の前処理・後処理**: うちの decode 本番経路は GDN 1 層あたり約 30 dispatch
   (相手 8)。`gdn_prework.py` (conv+silu+split+norm+scale+g/beta を 1 発) と
   `rms_norm_gated.py` は実装済みで capture 経路にも配線されているが**既定 off**。
   過去の A/B は `fired={}` で発火未確認のまま「効かない」と判定されていた。
   発火カウンタ付きで取り直す。MoE router (argpartition+take+softmax) も相手は 1 発
   (`moe_route.py` は過去 +4% 遅い判定、これも発火未確認)。
   HC は両者とも 6 dispatch/層で同数。

prefill の MoE は op 列・MLX の分岐とも同一 (`gather_qmm_rhs`、bm は M/E で決まる)。
差があるならルーティングの偏りとチャンク粒度で、コードからは出ない。

## xctrace は今回は使えなかった (2026-09-02 14:30)

`Metal System Trace` を launch 方式で 32 秒取っても、`metal-gpu-intervals` に python
(MLX) の Compute 区間がほぼ出ない (49k 行のうち python は 87 行、大半は Claude Helper の
描画)。`metal-shader-profiler-intervals` は 0 行 (カーネル名は取れない)。attach 方式は
uv の python に効かず ("Cannot find process")。90 秒版は 19GB で後処理が終わらなかった。
道具 (`tools/gpu_trace_kernels.py`) は実物の形式に合わせて残したが、**カーネル単位の
突き合わせはこのレーンでは出ない**。decode の差は読解で見つけた 3 つの非対称
(bool マスク / QK-norm+RoPE 融合 / GDN 前処理の既定 off) を A/B で潰す方向に切り替える。

## 罠: decode_ab のプロセスが終了時に固まり、91GB を握ったまま残る (2026-09-02 14:50 発見)

`--knob gdn-metal` の decode_ab が 13:48 に結果を書いた後も生き残り (RSS は小さいが
Metal のバッファは解放されない)、以後の GPU ジョブが 2 モデル分のメモリで走って
スワップ 25GB (pageouts 190 万ページ) になっていた。**13:48 以降の計測は無効**:
真の union 統計 (途中で打ち切り)、xctrace の試行、端数チャンク畳み込みの A/B
(27 分経っても終わらず打ち切り)。GDN Metal の A/B (13:37-13:48) も直前のジョブが
残っていた可能性があり、取り直す。連鎖スクリプトは各ジョブ後に
`pkill -f "decode_ab.py --knob"` を入れた。固まる場所は未特定 (sample が取れなかった)。

## 融合 3 つの取り直し (2026-09-02 14:52-、短文脈 3 本 x 512、発火カウンタ付き)

- `gdn-prework`: A +1.4% (遅い) だが **発火カウンタに `gdn_prework` が無い = 一度も dispatch
  されていない**。過去の判定と同じ「動いていない」状態。knob / enable の配線を診断中。
- `rms-norm-gated`: A ms/round +3.3%、**tok/round -6.5%** (出力が動く)、C (機構のみ) は基準と同じ。
  発火表示に `rms_norm_gated` が無いのに出力が動くのは不可解 (カウンタの位置か経路の問題)。
  時間の取り分が無く受理率を落とすので**却下のまま** (過去の判定と同じ)。
  注: この A/B 中に診断用の合成モデル GPU プローブが並走した可能性があり、ラウンド時間の
  絶対値 (60 ms) は信用しない。
- `moe-route`: 発火 12299 回。A ms/round +4.2%、**tok/round -6.1%** (ルーティングが動く)。
  **却下** (過去の判定と同じ。融合しても argpartition + softmax より遅い)。
  注: この 2 本の基準ラウンドが 52-60 ms で、直前の gdn-prework の 39 ms から 30% 悪い。
  同時に走っていた CPU 側の検証 (別エージェント) か熱。絶対値は捨て、A/B の差だけ読む。

## 端数チャンク畳み込みの A/B (2026-09-02 15:07-15:20、17k x 3 本、熱い + スワップ回復直後)

1 本目は 33 → 36 → 51 → 78 s と暴走 (スワップ回復中)。2-3 本目: A (畳む) 57.0 / 55.7 s、
B (畳まない) 56.8 / 57.2 s → **差なし (±0.5%)**。端数チャンク単独の 4.49 ms/tok は
グループに入れても縮まなかった (熱い状態で 17k が 57 s = 冷えた 37.6 s の 1.5 倍)。
既定は on のまま (ビット一致、害なし) だが取り分は無い。

## 熱の現状 (15:20)

4 時間の連続 GPU 使用で 17k prefill が 37.6 s (11:20) → 57 s。以後の A/B は差だけを読む。

## GDN Metal の取り直し (2026-09-02 15:27-、残留プロセス無し、熱い)

1 本目は熱の立ち上がりで 37.7 → 60.7 s と暴走 (使えない)。2 本目: A 56.6 / 55.4、
B 56.7 / 56.8 → **-1.3%**。最初の A/B (-1.5%) と一致。採否は KLD を通してから。

## GDN Metal 取り直し (3 本、15:26-15:44): prefill_s A 53.26 / B 55.78 = **-4.5%**

(1 本目は熱の立ち上がりを含む。2 本目単独では -1.3%。) KLD を通してから既定 on にする。

## 参考: M4 Prefill Engine (mohamedhossammohamed.github.io/m4-prefill-engine)

dense LLaMA 系 1B/8B 向けの C++/Metal 単体実装。MLX 4bit 比で M=128-129 のとき
1.05-1.25 倍、M=2048 で同等。MoE / GDN / QSA は扱わないので Flash-Next には効かない。
**dense LLaMA / Gemma を載せるとき** (ユーザー方針) に dense SwiGLU 射影と
barrier 削減 FlashAttention を読み直す。

## 真の union (2026-09-02 15:44-15:55、ctx 11873、`gather_union_stats.py --tiles 64,32`)

| tile | true_union_ratio 平均 (kv 4k → 12k) | dense 比の FLOP |
|---|---|---|
| 64 | 0.75 → 0.70 (平均 0.74) | 0.73 |
| 32 | 0.73 → 0.59 (平均 0.66) | 0.65 |

**宣言していた反転条件「tile 32 で 6 割超なら畳む」に該当。**隣接 32 クエリの選択ブロックの
和集合が kv の 6 割あり、集めても dense の 3 分の 2 にしかならない。12k → 50k で比は下がる
(kv 12k で 0.59) が、17k 以下では取り分が attention の 3 割 x 全体の 2-3 割 = 全体の 1 割弱で、
タイルごとの同期と gather の費用を払って残るのはその半分。**prefill attention の union gather
レーンは畳む。**50k 専用に再訪するなら tile 16 と kv 50k で測り直す。

## fast-rope の A/B (2026-09-02 15:47-15:58、短文脈 3 本 x 512)

ms/round +0.3% (取り分なし)、tok/round -2.1% (丸めで受理が動く) → ms/tok +2.3%。
**却下 (既定 off のまま)。**attention 12 層の rope まわり十数 op は、ラウンド 38 ms の中では
測れない大きさだった。17k 側は測らずに打ち切り。

## bool マスクの A/B (2026-09-02 15:52-16:08、17k x 3 本 x 512、熱い状態)

`--knob bool-mask` (A = bool、既定 / B = 旧加算マスク): **ms/tok -12.8%** (A 35.5、B 40.7)、
ms/round -12.7%、tok/round 同一 (1.628)、prefill_s 差なし。2-3 本目は A 35.1-36.9 / B 40.1-41.1
で安定。**今日初めての decode の確定した取り分。**既定 on (commit `ea33a66`)。
仕組み: 17k の verify 幅 2 では gather 経路が比の上限 (0.20) で辞退して dense マスク経路に
落ちるので、そこでキーの読み飛ばしが効くようになった。

## bool マスク後の forward S=1 (17k): eval 32.3 ms (変化なし)

S=1 は gather 経路 (union 12% <= 20%) を通るので `_final_mask` を使わない。bool マスクが効いたのは
verify 幅 2 で gather が辞退する dense 経路。**S=1 でも dense+bool の方が gather より安い可能性**が
あるので `--knob gather-attn` (A = gather / B = dense) を 17k で取り直す (待ち行列)。

## 相手のチャンク幅プローブ (2026-09-02 16:20、4k プロンプト、熱い)

`MLX_SERVE_PREFILL_CHUNK`: 2048 → 6.15 s、4096 → 5.82 s (3827 tok)。**チャンク幅は相手の速さの
主因ではない (5.7%)。**うちの G=4 連結 (8192 行) と相手の 4096 行は MoE の行数として同等。

## 4k の 1 トークン費用 2.17 ms の分解 (読み直し)

段 P0 の layer-major チャンク 3617 ms = 1.76 ms/tok に対し、HTTP の 4k は 2.17 ms/tok。差 0.41:
chunk-major (+10%、4k は最終チャンクが半分) 0.18、n-gram 行取得 0.12 (バッチ化後 0.09)、
prime / checkpoint / split 0.1。相手 1.51 との差のうち、素のチャンクの差は 1.76 vs ≈1.45 (20%) で、
GDN のスキャン (Metal 移植で -4.5%) 以外はまだ帰属できていない (MoE のタイル効率が候補)。

## prefill の余地はどこまでか (2026-09-02 16:40、ユーザーの問い「2-3 倍あるのでは」への答え)

**FLOP 天井**: 1 トークンの活性パラメータ ≈ 5.1B (専門家 10 x 48 層 2.36B、GDN/attention 射影
2.5B、共有 0.24B) → 10 GFLOP、17k の attention 込み 12-13 GFLOP。M3 Max 14 TFLOPS で
0.9 ms/tok = 1100 tok/s。うち 447、相手 576。天井までうち 2.5 倍、相手 2 倍。

**実際に取れる見込み** (文献と実装物から):

| 項目 | 手 | 見込み | 費用 |
|---|---|---|---|
| MoE (4-5 割、効率 47%) | fused MoE (vLLM / TRT-LLM の gate+up 連結 N=1280、SiLU と down を共有メモリで) = oMLX `qwen35_moe_gate_up`。うちの `MLXTURBO_WIDE` は decode で負けたが prefill 幅は未測 | 効率 65% で -15% | A/B 1 本 |
| 逆量子化 | Marlin / Machete (CUDA W4A16、2 倍) はレイアウト + テンソルコア。Apple は M5 NAX までテンソル演算が無く、MLX の qmm は既にレジスタ内逆量子化 + simdgroup matmul。事前に bf16 へ戻す案は 512 専門家 x 48 層で毎チャンク 240GB を動かすので不成立 | 2-3 割が上限 | カーネル |
| GDN スキャン | oMLX 移植 (1.59 倍) | -4.5% (実測) | KLD 判定中 |
| QSA attention (50k で 3 割) | ブロック疎の Metal カーネル (真の union 6 割なので gather では取れない) | 50k -20%、17k -8% | 数日 |
| ANE | 相手の dense MLP 4 割 offload (oMLX 由来、私的 API) をうちの dense 射影 (FLOP の 45%) に | 1.2-1.4 倍 | 壊れやすい |
| チャンク外 (4k で 2 割) | 最終チャンクの layer-major 化、checkpoint の軽量化、prime | -5-10% | 半日 |

合計の現実的な上限は **1.5 倍前後**。今日の待ち行列で取れるのは 1 割弱。「2-3 倍」は FLOP 天井の
話で、同じ MLX op を使う相手が 576 tok/s に留まっているのがその証拠。

## GDN Metal の KLD (2026-09-02 16:33-16:50、`quant_eval.py compare --fusions`、相対レーン)

既定 (Metal off): kld_mean 0.01312 / top-1 一致 0.969。Metal on: **0.01326 / 0.966**。
差 +0.00014 で受け入れ幅 (+0.0005) の中。prefill -1.3〜-4.5% と合わせて**既定 on にする**。

## gather 経路 vs dense+bool (2026-09-02 16:31-、17k x 3 本 x 512)

1 本目は熱の立ち上がり (B の prefill 85 s) で無効。2-3 本目: A (gather) 35.2 / 34.6 / 36.4、
B (dense+bool) 34.9 / 35.2 / 36.1 → **差なし**。17k の verify 幅 2 は両者とも dense に落ちる
ので当然で、gather が効くのは S=1 のラウンドだけ。既定は変えない。

## 多日レーンを切った → `docs/research/LANES-2026-09.md`

## GDN 前処理融合の A/B (fp32 写しで発火、2026-09-02 16:52-17:00、短文脈 3 本 x 512)

発火 8316 回。ms/round **-1.1%**、tok/round -0.3%、ms/tok -0.5%。**取り分なし (却下、既定 off)。**
36 層 x 約 20 op を 1 発にしても 0.4 ms しか縮まない = 小さい op は MLX/Metal 側で既に重なっていて、
decode の +6 ms は dispatch 数ではない。部品別の突き合わせ (`decode_prof`) で場所を決める。

## vllm-mlx の調査 (scout 実読、`~/dev/vllm-mlx`、Apache-2.0)

- mlx_lm の `BatchGenerator` をモンキーパッチ。**固定の相方待ち窓は無い**: 毎 tick 待機要求を
  空きがある限り `insert()` して走行中バッチに合流 (mid-run join)、仕事が無いときだけ待つ。
- chunked prefill: prefill を budget ずつに割り、chunk 間に decode を 1 ステップ挟む。
- MTP はバッチ全体に毎ステップ (B=1..N、最小バッチ無し)。
- "paged KV" の実体はプレフィックスキャッシュのブロック格納 (paged attention ではない)。
  GDN 系は mlx_lm の ArraysCache のバッチ対応に委ねる。qwen4_exp 未対応。
- 性能主張: 5 並列で 3.4 倍 (Qwen3-0.6B)。
- **レーン 5 への持ち込み**: 待ち窓を捨てて途中参加に寄せる、prefill を chunk して decode と
  混ぜる、の 2 点。

## 相手の `[decode-prof]` は取れなかった (2026-09-02 17:03)

`MLX_SERVE_DECODE_PROFILE=1 --no-mtp --no-pld --log-level debug` で 200 トークン x 2 を流しても
`[decode-prof]` 行が出ない (decode は 23.8 tok/s まで落ちるので profile 自体は効いている)。
qwen4_exp の forward 経路では report が出ない可能性がある。相手側の部品別は諦め、うちの
`tools/decode_prof.py` と `module_costs.py` で内訳を持ち、相手は合計 (fwd-ubench) だけで比べる。

## うちの部品別 decode プロファイル (2026-09-02 17:10、`tools/decode_prof.py`、S=1、強制 eval)

serial/tok 79.9 ms (embed 0.2 / attn 31.0 / mlp 38.5 / combine 8.2 / lmhead 2.0、
moe: router 11.3 / experts 15.4 / shared ≈1) に対し、通常 forward は 22.9 ms。**強制 eval の
上乗せが +57 ms (境界 ≈290 回 x 0.2 ms)** で、部品の値は同期費用に埋もれる。相手の
`[decode-prof]` も取れなかったので、**この切り方での突き合わせは成立しない**。
decode 短文脈の +6 ms は「部品別 (単体ループ) では MoE 8.6 / GDN 6.4 / HC 3.7 / attn 1.5 /
lm_head 2.0 (計 22.1) に対し全体 27.9」という module_costs の形のまま未帰属。
残る仮説: (a) MLX のビルド差 (相手は mlx-src を自前ビルド、buffer-pool cap 8GB)、
(b) 層間の依存で露出するカーネルのレイテンシ差 (同じ op でも入力の contiguity が違う)、
(c) HC 融合カーネル自身の遅さ (3.7 ms / 97 回 = 38 us)。次は (c) を単体で相手の
hc_read 3 カーネルと比べる (レーン 1 の次の一手)。

## gate+up 連結 (`--knob wide`) の prefill A/B (2026-09-02 17:07-、17k x 3 本、熱い)

A (連結 N=1280) 73.2 / 72.0 / 63.0 s、B (素の 3 gather) 61.6 / 56.8 s → **連結は 15-20% 遅い**。
レーン 2 (fused MoE) のゲート「N=1280 で prefill_s が縮まなければ畳む」に該当、**畳む**。
**訂正 (夜、独立レビュー D-2)**: 既定 `MLXTURBO_SORT_MIN=16` では専門家の連結経路 (`_fused_w`) に
到達せず、`disable_wide_projections` も `_fused_w` を外さない。この A/B が測ったのは GDN / attention /
shared の連結で、専門家の gate+up 連結は未測。畳む判定は保留 (再測の条件は LANES のレーン 2)。
MoE の効率 47% は N でも行数でもなく、ルーティングの偏り (タイル水増し) か MLX の
gather_qmm カーネル自体の話で、うちの層では取れない。

## decode の固定費の読み直し (2026-09-02 17:30)

S=1 → 2 の増分: うち +5.5 ms、相手 +5.9 ms (行あたりの費用は同じ)。固定部分: うち 18.5、相手 12.2。
重みの読み出し (dense 2.5GB + 専門家 10 人 1.2GB ≈ 9.4 ms) を引いた「行数にも読み出しにも
依らない X」は うち 9 ms、相手 3 ms。GDN 前処理融合 (20 op → 1) が -0.4 ms しか効かないので、
X は op 数ではなく**小さいカーネル 1 つの費用** (うちの融合カーネルは 30-38 us/回) の差。
レーン 1 の次: HC 融合 (97 回 x 38 us = 3.7 ms) と前処理カーネルを単体で相手の同等物と並べ、
カーネル自体を速くする。
MTP depth: bool マスクで幅 3 の dense 経路が安くなったので、17k の depth 1/2 を取り直す (待ち行列)。

## フルテスト (2026-09-02 17:25-、10 分冷却 → serve → turbo → serve、文脈 6 点、thinking off)

途中経過 (両者とも前の走行で熱い。絶対値は朝より 5-15% 悪い):

| 文脈 | 冷 TTFT serve / turbo | 温 TTFT serve / turbo | decode serve / turbo | 比 |
|---|---|---|---|---|
| 0 | 0.19 / 0.53 | 0.74 / 0.52 | 51.6 / 46.3 | 0.90 |
| 4k | 6.13 / 8.00 | 0.93 / 0.52 | 49.5 / 47.9 | 0.97 |
| 17k | 29.97 / 36.49 | 1.02 / 0.54 | 47.0 / 40.0 | 0.85 (朝 0.73) |
| 25k | 76.3 / 83.8 | 1.90 / 1.26 | 23.7 / **27.0** | **1.14** |
| 32k | 86.5 / 119.6 | 1.74 / 0.62 | 25.1 / **27.7** | **1.10** |

**両者とも 17k → 25k で decode が 4 割落ちる崖がある** (相手 47 → 24、うち 40 → 27)。相手は prefill も
561 → 325 tok/s に落ちる (kv > 8192 で QSA gather カーネルに切り替わる境界の向こう)。うちの崖は
25k で gather 経路が再び有効になる (幅 2 の上限 4096 <= 0.2 x 24815) ことが候補 — bool マスク後は
dense+bool の方が安い可能性があるので `--knob gather-attn --ctx 25000` を待ち行列に入れた。

## 独立レビュー (2026-09-02 夜)

文脈を渡さない Opus 4 本 (投機中核 / バッチ / サーバー / カーネル) にコードだけで欠陥を探させた。
記録は `docs/research/REVIEW-2026-09-02-INDEPENDENT.md`。本番既定経路で出力を壊すものは無し。
受理率を静かに落とす欠陥が 1 つ (MTP ドラフトキャッシュが受理済みの中間トークンを積まず、
位置が毎ラウンド hit ぶんずれる。`spec_flash.py:857,1537`、親がコードで確認)。契約テスト 2 本が
main で動いていない (`test_server.py` 1 本失敗、`test_qmm_skinny_mma_static.py` ImportError)。
対処の優先順は同ファイル冒頭。

## 独立レビュー (REVIEW-2026-09-02-INDEPENDENT.md) を受けた訂正 (2026-09-02 18:25)

- **D-2 により `--knob wide` の B (off) 側は連結を外せていなかった** → 「gate+up 連結は 15-20% 遅い」
  (レーン 2 を畳んだ根拠) は**無効**。D-2 を直して取り直す (待ち行列 `wide-prefill-17k-b`)。
- A-1 (MTP キャッシュに受理トークンを積まない) の修正を実装中。tok/round が上がれば
  decode 全域に効く。knob `mtp-append` で A/B。
- 誠実版フルテスト (各エンジンの前に 12 分アイドル、turbo → serve) を待ち行列に入れた
  (`self-snapshot-*-0902g.json`)。熱い状態の 0902f は参考値に格下げ。

## フルテスト 0902f (熱い状態、参考値。17:25 冷却 10 分 → serve → turbo → serve2)

| 文脈 | 冷 TTFT serve / turbo / serve2 | 温 TTFT serve / turbo / serve2 | decode serve / turbo / serve2 | 比 |
|---|---|---|---|---|
| 0 | 0.19 / 0.53 / 0.18 | 0.74 / 0.52 / 0.74 | 51.6 / 46.3 / 54.5 | 0.87 |
| 4k | 6.13 / 8.00 / 5.81 | 0.93 / 0.52 / 0.89 | 49.5 / 47.9 / 54.4 | 0.92 |
| 17k | 29.97 / 36.49 / 28.65 | 1.02 / 0.54 / 1.05 | 47.0 / 40.0 / 50.1 | 0.82 |
| 25k | 76.3 / 83.8 / 65.6 | 1.90 / 1.26 / 1.72 | 23.7 / 27.0 / 27.5 | 1.05 |
| 32k | 86.5 / 119.6 / 80.6 | 1.74 / 0.62 / 1.57 | 25.1 / 27.7 / 28.4 | 1.04 |
| 50k | 133 / 204 / 127 | 27.2 / 0.67 / 24.9 | 21.8 / 27.9 / 28.2 | 1.11 |

turbo は serve の直後 (熱い) に走ったので不利。serve2 の方が serve より速い (熱は単調でない)。
**誠実な値は 0902g (各エンジンの前に 12 分アイドル) を待つ。**

## MTP キャッシュに受理トークンを積む (A-1) の A/B、短文脈 (2026-09-02 18:49)

tok/round A 2.184 / B 2.192 (-0.4%)、ms/round +0.8%。本ごとに 2.081 対 2.216 (悪化) /
1.984 対 1.903 (改善) / 2.485 対 2.456 (B は EOS で短い)。**512 トークンでは受理率に効かない。**
レビューの「生成長に比例して落ちる」は 512 の範囲では現れない。17k の結果を見て既定を決める
(効かなければ MLXTURBO_MTP_CACHE_APPEND の既定を 0 に戻す)。

## prefill の「未帰属の 2 割」の棚卸し (2026-09-02 18:55、段 P0 の冷えた実測から)

1 チャンク (2048 tok、kv=2048) の実測と FLOP 下限:

| 部品 | 実測 ms | 下限 ms | 超過 ms | 超過の帰属候補 |
|---|---|---|---|---|
| MoE 48 層 | 2041 | 972 | **1069** | ルーティングの偏りによるタイル水増し (1.4-1.9 倍)。行数 (G) と N の連結では動かず (連結は D-2 で再測中)。`tools/moe_routing_skew.py` で決着させる |
| GDN 36 層 | 1006 | 762 | 244 | スキャン (Metal 移植で -4.5%)、前処理の素 op、fp32 状態の読み書き |
| HC 97 回 | 423 | 234 | 189 | prefill 幅の HC は素 op (融合は -0.9% で却下済み) |
| attention 12 層 | 353 | 274 | 79 | kv=2048 では小さい。kv に比例して伸びる (末尾で 1954 ms) |
| 合計 | 3823 | 2242 | 1581 | |

チャンク外 (4k で 2 割): 最終チャンクの chunk-major (+10%)、n-gram 行取得 (バッチ化で半減)、
prime 0.25 s、checkpoint。

**未帰属 2 割の 3 分の 2 は MoE の超過 1.07 s。**行数分布の偏りが原因なら、重い専門家と軽い
専門家を別の gather_qmm 呼び出し (bm を変える) に分ける Python だけの手が成立する。
偏りでないなら MLX の `gather_qmm_rhs` そのものの効率で、相手も同じ op なので差の説明に
ならず、prefill の差は attention (17k 以上) と GDN/HC の小物に絞られる。

## MTP キャッシュに受理トークンを積む (A-1) の A/B、17k (2026-09-02 19:00)

ms/tok A -1.5%、ms/round -0.3%、tok/round +1.2% (1.669 対 1.649)。短文脈で害なし、17k で
小さく効くので **既定 on (MLXTURBO_MTP_CACHE_APPEND=1) のまま**。レビューの「生成長に比例して
受理率が落ちる」は 512 トークン窓では 1% 程度の効果に留まる。

## wide (gate/up の N 連結) の再 A/B、D-2 修正後、17k prefill (2026-09-02 19:20)

回文順 A B B A × 3 本。prefill_s A 78.6 対 B 48.3 (**+62.6%**)、decode ms/tok +43.2%、
tok/round 同じ。3 本とも A が 25 s 以上遅く、熱 (B が 41〜60 s とばらつく) を差し引いても
逆転の余地はない。**レーン 2 (行数・N の連結でタイル水増しを減らす) は棄却で確定。**
D-2 の並び替えバグは直したが、連結そのものが gather_qmm の効率を落とす。
MoE 超過 1.07 s の帰属は `tools/moe_routing_skew.py` (chain32 で実行待ち) に絞る。

## draft depth 1 対 2、17k、bool マスク後 (2026-09-02 19:40)

ms/tok depth1 27.6 対 depth2 30.3 (**depth1 が -8.9%**)。tok/round は depth2 が 2.001 対 1.669 で
高いが、ms/round が +23.9% で相殺を超える。bool マスクで 17k の decode が速くなっても、
2048 超で depth 1 に落とす現行の方針は変えない。

## gather attention (段 3(b)) の 25k A/B (2026-09-02 19:50)

decode ms/tok A -0.3%、prefill -3.0% (熱の範囲)、tok/round 同じ。結果 JSON の `fired` に gather の
発火が無い (gdn_metal と ngram だけ)。**25k でも union の上限 0.20·kv の条件で毎回辞退していて、
一度も走っていない。**17k と同じ結論で、この経路は 0.20 の閾値のままでは 25k まで出番が無い。
閾値を上げる案は `tools/gather_union_stats.py` の真の union 比 (T=32 で 0.665) からして
読み出し量が減らないので却下のまま。

## QSA タイル union の真の大きさ、T=4/8 (2026-09-02 20:40、`bench/results/gather-union-stats-t4t8.json`)

17k prefill 全体の平均 (kv 4k〜16.8k の記録 44k 本):

| T (束ねるクエリ数) | 真の union 比 (平均) | 同 kv=16.8k | 読み出し ∝ union/T | dense 比の FLOP |
|---|---|---|---|---|
| 32 | 0.665 | — | 0.021 | — |
| 8 | 0.442 | 0.295 | 0.055 | 0.993 |
| 4 | 0.364 | 0.221 | 0.091 | 0.702 |

隣り合うクエリの top-512 ブロックの重なりが小さく、4 本束ねただけで union が kv の 36% (理想は
12%) になる。**FLOP は T=4 でも dense causal の 70%、T=8 で差なし**。dense の steel attention は
クエリタイル 32 本で K/V を共有するので読み出しは 1/32、T=4 のブロック疎はその 3 倍読む。
設計書 `QSA-PREFILL-KERNEL-DESIGN.md` のゲート (合成テンソルで dense sdpa の 2 倍) には
どの T でも届かない。**レーン 3 のブロック疎カーネルは畳む** (設計書と骨組みは記録として残す)。
17k 以上の prefill attention を縮める手は、疎化ではなく dense attention そのものの効率
(相手と同じ steel を使っているので差にならない) か、indexer (QSA の選択側) の費用に絞られる。

## カスタム Metal カーネル 1 回の固定費 (2026-09-02 20:35、モデル無し、直列依存の連鎖 500 本)

**訂正 (20:50): この節の数字は無効。**chain32 (union 統計と MoE 偏りの prefill) と GPU を共有した
状態で測っていた (CLAUDE.md の「別 GPU プロセスを並走させない」に反する)。同じ条件の 2 回目は
カスタム 2.5 us / 組み込み 1.9 us / 1M 要素のカスタム 46.6 us 対 組み込み 10.6 us で、1 回目
(24.8 / 7.4) と 10 倍ずれた。並走の負荷で数字が決まっている。**冷えた GPU で chain33 として
再測するまで、固定費についての結論は出さない。**残る仮説は 2 つ: (1) カスタムカーネルの
固定費が組み込みより大きい、(2) 固定費は同じで、うちの融合カーネル (HC 97 回/forward) の
1 回の実行が並列度不足で 30-60 us かかっている (micro の HC fused 262 us は 1 回 + 同期 200 us)。
どちらでも「S=1 forward 1879 op のうちカスタムカーネルが何回、1 回いくら」を数えれば決まる。

(無効な数字、記録のため) 1 回目: `mx.fast.metal_kernel` 24.8 / 組み込み 7.4 / `mx.fast.rms_norm` 12.6 us。


## MoE ルーティング偏りの実測 (2026-09-02 20:55、`bench/results/moe-routing-skew.json`、17k のチャンク 0 と 7)

| 行数 | M/E | bm | タイル水増し (chosen) | bm 未満の専門家に入る行 | 行単位で分けた場合の下限 |
|---|---|---|---|---|---|
| 2048 (本番のチャンク) | 40 | 32 | **1.40〜1.42** | 12.5〜13.7% | 1.23〜1.25 |
| 8192 | 160 | 64 | 1.22 | 5.8% | 1.10 |

48 層でほぼ同じ値 (層による偏りの差は無い)。チャンク 0 と 7 も同じ。

読み方:
- **MoE の超過 2.1 倍 (2041 対 下限 972 ms/chunk) のうち 1.4 倍がタイルの水増し、残り 1.5 倍が
  `gather_qmm` 自体の効率。**水増しは以前の見積もり (1.4-1.9) の下端で、チャンクを 4096 にしても
  bm が 64 に切り替わるので水増しは 1.4 → 約 1.5 と減らない (`MLXTURBO_PREFILL_CHUNK` が in-model で
  負けた記録と整合)。
- 「bm 未満の専門家を別の呼び出しに分ける」は bm=32 のままだと 1.49 で**悪化**する
  (split_bm32)。効くのは小さい専門家側を行単位 (水増し無し) で処理できる別カーネルがあるとき
  だけで、その上限が 1.40 → 1.23 = MoE の 12%、チャンク時間の 6%、17k prefill で約 2 s。
  過去に自前の MoE gather 系カーネル 2 件 (moe_verify_gather、wide) が gather_qmm に大差で負けて
  いるので、この 12% を取りに行く優先度は低い。
- 残り 1.5 倍は MLX の `gather_qmm_rhs` の効率で、相手も同じ op を使うなら差にならない。
  相手の MLX 写し (`~/dev/mlx-serve/lib/mlx-src`) が量子化 gather を改造しているかを確認する。

## カスタム Metal カーネルの固定費、冷えた GPU で再測 (2026-09-02 21:03、chain33、並走なし、2 回)

| 種類 (直列依存 500 本、us/op) | 1 回目 | 2 回目 |
|---|---|---|
| `mx.fast.metal_kernel` (2560 要素の加算、256 スレッド) | 3.1 | 2.8 |
| 組み込み elementwise (`y + 1.0`) | 4.4 | 3.4 |
| `mx.fast.rms_norm` | 6.7 | 4.6 |
| カスタム 1M 要素 (1 スレッド 1 要素、素朴) | 45.6 | 45.6 |
| 組み込み 1M 要素 | 9.7 | 9.9 |

**カスタムカーネルの起動固定費は組み込みと同じ 3 us 前後。仮説 (1) は棄却。**構築 (Python) は
0.2〜1.0 us/op で無視できる。S=1 forward の 1879 op × 3 us ≈ 5.6 ms がカーネル起動の床。

一方、素朴に書いたカスタムカーネルは同じバイト数で組み込みの **4.7 倍遅い** (45.6 対 9.7 us)。
ベクトル化されない load とスレッド配置の差。仮説 (2) 「うちの融合カーネル 1 回の実行が
データ量に対して遅い」はこれと整合する。`tools/kernel_chain_cost.py` (作成中) で HC / GDN step /
prework / rms_norm_gated の 1 回の費用を連鎖で測り、forward 内の呼び出し回数を掛けて
未帰属 6 ms への寄与を出す。

## prefill attention の経路と head_dim 256 の費用 (2026-09-02 21:30、`tools/sdpa_headdim_micro.py`、GPU 空き)

事実 (MLX 0.32.2 のソース、相手の写しは無改造なので pip と同一):
`ScaledDotProductAttention::use_fallback` は S>8 で head_dim 192/256 なら常に true (NAX 機で
causal かつ配列マスク無しのときだけ融合)。**Flash-Next (head_dim 256) の prefill attention は
融合 (steel) ではなく、スコア [H, S, kv] を実体化する素の経路。**相手 (mlx-serve) は head_dim 256
の flash 型カーネル (`msv_attn_p256`) と QSA 用の block-gather カーネル (`msv_attn_qsa256`、
kv > 8192 で有効) を自前で持ち、この経路を避けている。

費用 (S=2048、Hq=24、Hk=2、bf16、ABAB × 5 の中央値、ms/層):

| kv | d256 bool マスク (現行) | d256 causal | d128×2 head 融合 マスク | 同 causal | 現行 / 融合 |
|---|---|---|---|---|---|
| 2048 | 10.5 | 10.5 | 9.4 | 4.9 | 1.12 |
| 8192 | 40.7 | 40.8 | 36.4 | 31.5 | 1.12 |
| 16896 | 83.8 | 84.0 | 74.9 | 69.3 | 1.12 |

読み方:
- **素の経路の損は 12% しかない。**head_dim 256 では matmul が支配的で、スコアの実体化は小さい。
  MLX が 256 を素の経路に回す判断は正しい。dense のまま融合しても取り分は 1 割。
- 融合 (d128×2) の kv=16.9k は 850 GFLOP を 75 ms = 11 TFLOPS で、GPU の計算上限に張り付いている。
  **kv に比例して伸びる attention は、計算量を減らす (選択ブロックだけ計算する) 以外に縮まない。**
- in-model の末尾チャンク attention は 12 層で 1954 ms = 163 ms/層。マイクロの sdpa 84 ms との差
  約 80 ms/層は indexer (pooled key、スコア、argpartition)、マスク構築、`mask & sparse` の分。
  **indexer 側が sdpa と同じ大きさ**で、こちらは疎化カーネル無しで縮められる可能性がある。

レーン 3 の判定の訂正: T=4/8 の union で畳んだのは早計だった。相手が使っているのは T=1 (クエリ
ごとの直接添字 gather) で、union の問題は無く、kv=16.9k で読むのは 2048 キー × K/V × 2 kv head
= 8 GB/層/チャンク (400 GB/s で 20 ms)、計算は dense causal の 24%。前回の T=1 カーネルが dense と
同速だったのは読み出し量ではなくカーネルの効率 (素朴な load で 85 GB/s 相当) と考えるのが
整合的。**ゲートは変えない (合成で dense sdpa の 2 倍、kv=16.9k S=2048)。手を T=1 + 12 GQA head
で K/V タイル共有 + uint4 の段階 load に変えて再挑戦する。**見込み: 17k prefill で約 2 s (5%)、
50k で約 25 s (15%)。indexer 側の 80 ms/層は別途 `tools/qsa_prefill_split.py` で内訳を出す。

## decode 幅の attention: S ≥ 3 は vector カーネルから外れる (2026-09-02 22:10、`tools/sdpa_headdim_micro.py --S`、`tools/sdpa_split_micro.py`、GPU 空き)

MLX 0.32.2 `scaled_dot_product_attention.cpp:703`: S ≤ 8 の vector カーネルは
`query_sequence_length * gqa_factor > 32` で不適格。Flash-Next は gqa 12 なので **S ≥ 3 は素の経路
(スコア実体化、bool マスクのスキップ無し、全 kv 読み)**。S ≤ 2 だけがキーをスキップできる。

1 層あたり ms (Hq 24、Hk 2、head_dim 256、bool マスクは QSA 風に 2048 キー可視):

| S | kv 4096 | kv 17000 | kv 25000 | kv 50000 | 備考 |
|---|---|---|---|---|---|
| 1 | 0.33 | 0.54 | 0.43 | 0.43 | vector、スキップ効く (kv に依らない) |
| 2 | 0.24 | 0.34 | 0.39 | 0.59 | vector、スキップ効く |
| 3 | 0.97〜1.52 | 2.58〜3.38 | — | 7.08 | 素の経路、kv に比例 |
| 4 | 0.84 | 2.72 | 3.85 | 7.46 | 同上 |
| 8 | 0.88 | 2.85 | — | 7.91 | 同上 |
| 16 | 1.00 | 2.98 | — | 8.37 | 同上 |

幅 2 の呼び出しに分けて `concatenate` した場合 (同じ層、ms):

| S | kv 4096: 一括 → 分割 | kv 17000 | kv 50000 |
|---|---|---|---|
| 3 | 1.52 → 0.72 | 3.38 → 0.60 | 7.09 → 0.82 |
| 4 | 0.83 → 0.40 | 2.70 → 0.58 | 7.43 → 1.01 |
| 6 | 0.81 → 0.47 | 2.70 → 0.76 | 7.43 → 1.33 |
| 8 | 0.87 → 0.58 | 2.85 → 0.93 | 7.90 → 1.72 |

最大絶対誤差 1e-3 (bf16 の丸め、カーネルが違うため)。

読み方:
- 短文脈の本番 (depth 2 → verify 幅 S=3) は毎ラウンド 12 層 × 0.8〜0.9 ms ≈ **10 ms** を attention の
  素の経路に払っている可能性がある。4k の decode ラウンドは約 43 ms なので 2 割。
- 17k 以上で depth 1 (S=2) が勝っていた理由の一部はこれ (S=3 は 12 層で +27 ms/ラウンド)。
  分割すれば depth 2 が長文脈でも成立し得る (tok/round 2.0 対 1.67)。
- 50k の decode 落ち込み (+8.5 ms/tok) のうち attention は S=2 で 12 × (0.59-0.24) = 4 ms/ラウンド
  程度。残りは indexer 等 (`tools/qsa_prefill_split.py --S 1,2,4` で内訳待ち)。
- 実装 (`MLXTURBO_SDPA_SPLIT`、幅 32//gqa=2 で分けて呼ぶ) は Sonnet に出した。判定は in-model A/B
  (`decode_ab --knob sdpa-split --only short`、17k の depth 1/2 再測)、KLD +0.0005 以内。

## 融合カーネル 1 回の費用、直列連鎖 (2026-09-02 22:30、`tools/kernel_chain_cost.py`、N=200、ABBA×3、熱の状態は不明)

| 部品 (S=1 の本番 shape) | 融合 us/回 | 素の op us/回 | 比 |
|---|---|---|---|
| HC gated residual (pre+post の 2 カーネル) | **44.3** | 139.9 | 0.32 |
| GDN recurrent step (`with_states` 対 mlx_lm) | 29.6 | 17.5 | 1.69 |
| GDN prework (融合 1 発 対 素の約 10 op) | 38.6 | 33.4 | 1.15 |
| RMSNormGated | 5.6 | 11.2 | 0.50 |
| 床: `mx.fast.rms_norm` / `y+1` / 自前 2560 要素加算 | 4.2 / 3.2 / 3.0 | | |

読み方:
- **HC は融合しても 1 回 44 us。**触るデータは hyper 20 KB + 低ランク重み約 1.2 MB で、帯域の床は
  3 us。10 倍以上遅い = カーネルの効率 (並列度、load のベクトル化) の問題。1 層 2 回 × 48 層 = 96 回
  なら forward で 4.2 ms、これを 1 ms 以下にできれば decode の 6 ms の半分。
- GDN step の `with_states` (rollback 用の状態書き出し) は mlx_lm の 1.7 倍。36 回で +0.4 ms。
- **ただし `--count-forward` (capture 下の S=1 forward) では HC カーネルの呼び出しが 2 回しか
  数えられなかった** (hc_pre ×1、hc_post ×1)。96 回のはずなので、(a) 本番の decode 経路
  (`_staged_forward`) が HC の融合を素通ししている (GDN 融合と同じ現象)、(b) 適格判定 (D-4 で
  bits/group_size の一致を要求) で大半の層が素の op に落ちている、(c) 数え方の漏れ、のどれかを
  先に決める。(a)(b) なら decode は素の HC (140 us × 96 = 13 ms 相当が重なりつつ走っている) を
  払っていることになり、6 ms の説明そのものになる。

## HC 融合カーネルは 97 層中 1 層しか発火していない (2026-09-02 22:50、scratchpad/hc_fire_diag.py、実モデル S=1)

`GatedResidual` は 97 個 (48 層 × 2 + mixer)。うち 96 個は `block_inject_weight` が**量子化されていない
素の `Linear`** で、`fused._pack_quantized` が None を返し `patched` が素の実装 (`orig`) に落ちる。
発火するのは inject の無い 1 個だけ (plain でも capture でも同じ)。適格判定 (D-4) は原因ではない。
つまり **本番の decode は HC を 96 回、素の op で払っている。**連鎖計測では素 140 us 対 融合 44 us
なので、差 96 us × 96 回 = 最大 9 ms (重なりで一部隠れる)。decode +6 ms の最有力候補。

手: inject を bf16 のまま読むカーネル変種 (inject は小さく、復号不要)。量子化して押し込む手は
KLD を触るので後回し。判定は小さい in-model (S=1 forward の ABBA、scratchpad/sdpa_split_inmodel.py
と同型) → decode_ab の短文脈 3 本。

## 相手の verify 幅の扱い (scout 実読、`~/dev/mlx-serve/src/transformer.zig:3342-3393`)

`splitMaskedSdpa256` (env `MLX_SERVE_SDPA_SPLIT` 既定 on): qL 3〜8 のとき、クエリ行を
`max(1, 32/gqa)` = 2 行ずつに切り、各グループを MLX の array-mask sdpa で個別に呼んで concat。
K/V も head も分けない。qL=2 は素通し (vector に入る)。単体テストに「gqa 12 (24/2) の qL 4 は素の
MLX では非融合フォールバック」と明記されている。うちの `MLXTURBO_SDPA_SPLIT` と同型。
融合カーネル `msv_attn_p256` は qL ≥ 16 (prefill) 専用で、verify 幅には出てこない。

## 温 TTFT の中身 (レーン 8、2026-09-02 23:10、scout 読解 + scratchpad/tokenize_cost.py)

サーバー自身の ttft (`runner.generate` の生成開始からの時計、`turbo-0902f.log`):
reused=4082 new=17 で **0.16 s**、reused=17082 new=16 で **0.19 s**。ハーネスの温 TTFT は 4k 0.46 /
17k 0.51 s なので、**約 0.3 s は生成前 (HTTP、JSON、chat template、tokenize、セッション照合、
executor への受け渡し) か、ハーネス側にある。**

生成前の候補を潰した分:
- chat template + tokenize (モデル無し、CPU、中央値): 4k 2.1 ms、17k 8.9 ms、50k 25.7 ms。
  相手の `tokenize_cache.zig` が問題にしている「1813 トークンで 240 ms」はうちでは起きていない
  (HF の Rust tokenizer が速い)。文脈比例だが小さい。
- セッション照合は Python の LCP ループ (int 比較)。17k で数 ms 級 (読解、未計測)。
- 温ヒットは KV / GDN / indexer / n-gram を参照のまま使い、複製しない (読解)。
- prime は delta 長で頭打ち (PRIME_WINDOW 2048)。delta が十数トークンの温ヒットでは小さい。

残る帰属: 生成内 0.15〜0.19 s (17 トークンの forward 1 回は 40 ms 級なので、capture(light) の準備、
checkpoint、n-gram 行取得、最初のサンプルまでの固定費が 100 ms 前後ある) と、生成前の 0.3 s の
どこか。次は `--log-level debug` 相当の時刻印 (受信 / template 後 / 照合後 / 生成開始 / 最初のトークン)
を 1 リクエストぶん取って部品和 ≈ 壁時計を確認する。50k の 1.33 s は delta の forward が kv に
比例して伸びる分 (attention + indexer) で、こちらは decode の kv 罰と同じ帰属。

## 訂正: sdpa の幅 2 分割は 8/31 から本番に入っていた (2026-09-02 23:30)

`Attention.__call__` の「S × gqa_factor > 32 なら幅 2 に分けて sdpa を呼ぶ」は commit 7222fce
(2026-08-31) で無条件に入っていた。上の「短文脈で 1 ラウンド 10 ms の取り分」は**既に取れている分**で、
新しい取り分ではない (マイクロは分割の無い素の状態を測っていた)。今回足したのは knob
(`MLXTURBO_SDPA_SPLIT`、既定 on)、gather 経路 (`_gather_tile_attn`) への同じ分割、発火カウンタ、CPU 検査。
gather 経路は union ≤ 0.20·kv でしか走らないので、実効はほぼ無い。**decode +6 ms の説明にはならない。**
残る本命は HC 融合カーネルの 96 層不発火 (上の節)。sdpa-split の A/B 連鎖 (chain34) は取り下げた。

## QSA attention 1 層の内訳 (2026-09-02 23:50、`tools/qsa_prefill_split.py`、full attention の最初の層、実モデル)

prefill 幅 S=2048 (部品和 ≈ 壁時計、±3%):

| kv | indexer | マスク | sdpa | その他 (qkv+gate+o_proj) | 層の壁時計 | ×12 層 |
|---|---|---|---|---|---|---|
| 2048 | 0.9 | 0.00 | 10.5 | 19.2 | 29.8 | 358 |
| 8192 | 3.0 | 0.07 | 40.8 | 19.9 | 64.0 | 768 |
| 16896 | 6.7 | 0.00 | 87.2 | 21.2 | 116.7 | 1400 |

- **sdpa が 75% (kv=16.9k)、indexer は 6%。**「indexer が sdpa と同じ大きさ」の疑い (上の節) は
  外れ。以前の tracer の 163 ms/層は熱か別の内訳で、この測り方では 117 ms/層。
- sdpa は合成マイクロ (84 ms) と一致。kv に比例。17k 以上の prefill attention を縮める手は
  **選択ブロックだけ計算する T=1 の gather カーネル**しかない (dense は計算上限に張り付き)。
  見込み: 末尾チャンクで sdpa 87 → 30 ms/層 なら 12 層で -0.7 s/チャンク、17k 全体で -1.5〜2 s (5%)、
  50k で -15% 前後。
- その他 (qkv 14 ms) は S=2048 の射影 75 GFLOP で 5 TFLOPS 相当。素の qmm としては低い。
  中身 (q/k/v 射影、qk norm、rope、gate の split) の分離は未着手。

decode 幅 S=1 (部品ごとの eval 同期 0.2 ms が乗るので部品和は壁時計の 2〜3 倍。**壁時計だけ読む**):

| kv | 経路 | 層の壁時計 ms | ×12 層 ms |
|---|---|---|---|
| 4096 | dense | 0.67 | 8.1 |
| 17000 | gather | 0.76 | 9.1 |
| 25000 | gather | 0.84 | 10.1 |
| 50000 | gather | 0.84 | 10.1 |

- S=1 の attention 層は 12 層で 8〜10 ms と、forward 24 ms の 3 分の 1。kv 4k → 50k で +2 ms しか
  伸びない。**decode の kv 罰 (+8.5 ms/tok) は attention 層以外にある** (indexer の増分キャッシュ、
  verify 幅 S=2 の経路、n-gram、MoE の重み読み)。S=2/4 の結果 (chain35) と合わせて帰属する。
- S=1 では 17k 以上で gather 経路 (段 3(b)) が実際に走っている (union 2048 ≤ 0.2·kv)。

## decode 幅 S=2 の attention 1 層 (2026-09-03 00:05、`tools/qsa_prefill_split.py --S 2`、部品和は同期の床で 1.5〜2.4 倍に膨らむので壁時計だけ読む)

| kv | 経路 | 層の壁時計 ms (同期 1 回込み) | ×12 層 ms | 参考: S=1 の ×12 |
|---|---|---|---|---|
| 4096 | dense | 1.24 | 14.9 | 8.1 |
| 17000 | dense | 1.69 | 20.3 | 9.1 |
| 25000 | gather | 1.97 | 23.6 | 10.1 |
| 50000 | gather | 0.96 | 11.5 | 10.1 |

- S=1 → S=2 で attention 層が 12 層ぶん +7 ms。verify 幅 S=2 の forward が S=1 より +8 ms
  (`verify_width_cost`: 25.2 → 33.1 ms) なのは **MoE の重み読みではなく attention 層の中**。
  部品では sdpa (vector) 0.46、indexer 0.63、射影+gate+o_proj 0.89 ms (各 0.2〜0.3 ms の同期込み)。
  indexer と射影が sdpa より大きい。射影は重み 50 MB 級で帯域の床 0.13 ms、indexer は小さい op の
  列 (pooled の増分、スコア、argpartition)。**どちらも op 数か効率の問題**で、S=2 では sdpa は主役
  でない。
- 50k の壁時計が 25k より小さいのは gather 経路 (union 4096 が kv の 8%) が効いているためだが、
  熱・同期のばらつき (±0.3 ms) の範囲でもある。S=4 (chain36) と合わせて読む。
- 帰属を確定するには同期の床を外す測り方 (部品を N 回回して 1 回 eval) が要る。道具に追加中。
