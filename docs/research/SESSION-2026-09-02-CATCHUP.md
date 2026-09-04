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

## HC 融合を 97 層で発火させた小さい in-model (2026-09-03 00:30、scratchpad/hc_kernel_inmodel.py、ctx 0、ABBA×12)

| S | 融合 on (97 層) | off (素の実装) | 差 |
|---|---|---|---|
| 1 | 31.98 ms | 23.98 ms | **+8.0 ms** |
| 2 | 36.66 | 29.12 | +7.5 |
| 3 | 41.45 | 33.63 | +7.8 |

**融合カーネルの方が 8 ms 遅い。**連鎖マイクロ (融合 44 対 素 140 us/回) と逆で、「micro 勝ち in-model
負け」の前例そのもの。素の op 列は MLX が並べて重ねられるのに対し、カスタムカーネル 1 回 80 us 級
が 97 回直列に並ぶ。**bf16 inject の経路は既定 off (`MLXTURBO_HC_INJECT_BF16=1` で試せる) に戻す。**
decode +6 ms の説明にはならない (1 層しか発火していなかった状態が、今の 24 ms)。

残る問い: 素の HC 97 回が forward 24 ms のうち何 ms か (連鎖の 140 us × 97 = 13.6 ms が上限)。
HC を捨てるスタブ (上限の見積もり) と `MLXTURBO_HC=compiled` (mx.compile 版) を同じ ABBA で測る。

## HC の実装 4 通り、S=1/2 forward、ctx 0 (2026-09-03 00:45、scratchpad/hc_modes_inmodel.py、ABCD-DCBA×10)

| 実装 | S=1 ms | S=2 ms |
|---|---|---|
| 素の op (本番の 96 層の状態) | 23.90 | 29.19 |
| mx.compile 版 (`MLXTURBO_HC=compiled`) | 23.50 | 29.08 |
| 融合カーネル 97 層 (`MLXTURBO_HC_INJECT_BF16=1`) | 31.98 | 36.67 |
| HC を捨てるスタブ (費用の上限) | **19.23** | 22.84 |

- **HC の真の費用は S=1 forward で最大 4.7 ms (24 ms の 2 割)。**decode +6 ms 差の 3 分の 2 に相当。
  mx.compile 版は 0.4 ms しか縮めない (op 数の問題ではなく、qmv 2 本 + 小物の実行そのもの)。
- 今の融合カーネルは 1 回 80 us 級で、素の op 列より 8 ms 遅い。読むデータは 1 回 1.2 MB
  (帯域の床 3 us)、起動の床 3 us × 2 カーネル。**1 回 15〜20 us まで書き直せば forward -3 ms**
  (`hc_pre` の 10240→320 と `hc_post` の 320→10240 の qmv を threadgroup で分担し、uint4 load)。
  ゲート: 連鎖 (`tools/kernel_chain_cost.py`) で ≤ 20 us/回、in-model (hc_modes_inmodel) で
  素より -2 ms 以上、合成検査で最大誤差 1.5e-2 以内。反転条件: 連鎖で 30 us を切れなければ畳む。
- スタブとの差 4.7 ms は「HC を無くした場合」なので上限。相手 (hc_read ×3 の融合) は 18 ms なので、
  この 3〜4 ms を取れば decode 短文脈はほぼ並ぶ。

## 小さい結果 (mlxturbo だけ、冷えた機体、反復なし、256 トークン、thinking off、2026-09-03 01:40、`bench/results/self-snapshot-turbo-small-0903.json`)

| 文脈 | 冷 TTFT | 温 TTFT | decode tok/s | 朝 (11:09) の冷 TTFT / decode | 相手 (朝) の冷 TTFT / decode |
|---|---|---|---|---|---|
| 0 | 0.48 | 0.44 | 50.1 | — | — |
| 4k | **7.18** | 0.46 | **50.6** | 8.28 / 48.8 | 5.77 / 55.3 |
| 17k | **32.5** | 0.50 | **48.7** | 37.6 / 41.0 | 29.2 / 56.0 |
| 50k | **119** | 0.61 | **42.4** | 163 / 34.4 | 108 / 30.8 |

今日入れた既定 (bool マスク、GDN Metal、n-gram バッチ pread、端数チャンクのグループ化、MTP キャッシュ積み)
の合算: 冷 prefill 4k -13% / 17k -14% / 50k -27%、decode 4k +4% / 17k +19% / 50k +23%。
対相手 (相手は朝の値、同じ機体): 冷 prefill 4k 1.24x / 17k 1.11x / 50k 1.10x 負け (朝は 1.43 / 1.29 / 1.51)、
decode 4k -8% / 17k -13% 負け、50k +38% 勝ち。温 TTFT は 2 倍勝ちのまま。
lm_head は 8-bit のまま (相手は 4-bit)。`MLXTURBO_REBIT=head=4` の速度専用の数字は次に取る。

## 小さい結果、lm_head 4-bit (`MLXTURBO_REBIT=head=4`、速度比較専用、2026-09-03 02:05、`bench/results/self-snapshot-turbo-head4-0903.json`)

| 文脈 | 冷 TTFT | 温 TTFT | decode tok/s (8-bit → 4-bit) |
|---|---|---|---|
| 0 | 0.47 | 0.43 | 50.1 → 52.1 |
| 4k | 7.19 | 0.45 | 50.6 → 49.3 |
| 17k | 32.6 | 1.20 (*) | 48.7 → 46.5 |
| 50k | 119.5 | 0.60 | 42.4 → 43.4 |

- lm_head 4-bit の効果 (帯域 0.85 ms/forward ≈ +3〜4% の見込み) は反復 1 回の tok/step の揺れ
  (4k の 1 本目が 1.40 対 1.75) に埋もれる。**公平化の影響は 5% 未満**で、相手との差の主因ではない。
  判定は複数プロンプト × 512 の `decode_ab` でやるべき数字 (今回は「小さい結果」なので取らない)。
- (*) 17k の温ターンで `reused=16827 new=272`: 生成した返信をハーネスが本文で送り返し、再 tokenize した
  ID 列が生成時の ID 列と一致せず、返信部分 (272 トークン) を prefill し直した。8-bit の走行では
  `reused=17082 new=17` で当たっていた。**本文の再 tokenize が生成 ID と一致しない事例**は実クライアント
  でも起きる (温 TTFT 0.5 → 1.2 s)。照合をトークン ID ではなく復号した本文で取る、あるいは返信の
  ID 列をセッションに残して本文一致で採用する手が要る (レーン 8 に追加)。

## depth 適応制御 (DepthController、レーン 10) の 1 回目の A/B (2026-09-03 02:20、`bench/results/depth-adapt-{short,17k}.json`)

| 条件 | ms/tok | ms/round | tok/round | 選んだ深さ (A) |
|---|---|---|---|---|
| 短文脈 3 本 × 512 (B = 静的 depth 2) | A +3.3% | -12.0% | 1.85 対 2.18 | depth 1 が 144〜247、depth 2 が 23〜38、3 は 0 |
| 17k × 512 (B = 静的 depth 1) | **A -6.0%** | +6.5% | 1.88 対 1.65 | depth 1 が 75〜276、depth 2 が 28〜152 |

- 17k では受理率が高い区間で depth 2 を選び、-6.0% (ゲート -5% 通過)。**長文脈は静的 depth 1 より適応が勝つ。**
- 短文脈では線形の費用モデル (T1=25、dT=7) が深さを過大に罰し、depth 1 に張り付いて +3.3% 悪化。実測は
  1 段深くして +4.5 ms 程度 (B の ms/round 37.7 対 A 33.2)。費用を実測の EMA で更新する形に直して再測する。
- 既定は変えない (off)。直した版で短文脈が ±0 以内、17k が -5% 以上なら既定 on。

## depth 適応制御 2 回目 (費用 EMA 版、2026-09-03 03:00、`bench/results/depth-adapt2-{short,17k}.json`)

| 条件 | ms/tok | ms/round | tok/round | 選んだ深さ (A) |
|---|---|---|---|---|
| 短文脈 (B = 静的 depth 2) | A +3.6% | +0.6% | 2.13 対 2.18 | depth 1 が 179〜216、depth 2 が 29〜30 |
| 17k (B = 静的 depth 1) | A -3.2% | +7.8% | 1.84 対 1.65 | depth 1 が 171〜253、depth 2 が 103〜209 |

費用の定数を実測 EMA に替えても短文脈は depth 1 に張り付く。原因は位置 EMA の側: depth 1 に居る間は
a[1] が更新されず、一度低く出ると再探索されない (搾取の罠)。17k は 1 回目 -6.0% → 2 回目 -3.2% で、
選択が保守的になった分だけ取り分が減った。
次: (1) 2048 以下は静的 depth 2 のまま、2048 超だけ適応 (静的規則が depth 1 に落とす領域)、
(2) 周期的な再探索 (32 ラウンドごとに cap を 1 回) と古い位置 EMA の事前値への引き戻し。

## MLX の量子化 matmul の行数 (M) 依存、モデル無し (2026-09-03 04:00、scratchpad/qmm_m_scaling.py、1 回呼んで eval なので同期の床 ~200 us 込み。差だけ読む)

| 射影 (4-bit g64) | M=1 | M=2 | M=3 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|---|
| q_proj 2560→6144 | 244 us | +16 | +36 | +49 | +98 | +167 |
| o_proj 6144→2560 | 277 | +14 | +27 | +67 | +215 | +212 |
| GDN in 2560→10240 | 257 | +10 | +16 | -16 | +39 | +89 |
| MoE gather_qmm (E=512、top-10、2560→640、ランダム経路) | 221 | +17 | +31 | +34 | +84 | — |

- verify 幅 S=1 → 3 で、密な射影は 1 呼び出し +16〜36 us。attention 層 5 本 × 12 層 + GDN 2 本 × 36 層 +
  MoE 3 本 × 48 層で **合計 3〜5 ms**。S=1 → 3 の forward 差 +14.6 ms (25.2 → 39.8) の 3 分の 1 で、
  残りは attention 層の小さい op (indexer、sdpa の分割、concat) と MoE の重み読み増分。
- M=8 で o_proj が +215 us と跳ねる (MLX の qmv → qmm の切替点)。バッチ (B×S ≥ 8) の verify では
  ここが効く可能性。M ≤ 4 の verify 幅では射影の M 依存は小さい。
- MoE の gather_qmm を M=2048 でランダム経路 (ソート無し) で呼ぶと 26 ms/行列 = 2.6 TFLOPS。本番は
  `MLXTURBO_SORT_MIN=16` で行を専門家順に並べて呼んでいるので同じ数字ではない (ソートの有無で
  gather_qmm の経路が変わる)。ソート済みの同じ micro を取ると、MoE の「残り 1.5 倍」の帰属に使える。

## HC カーネル書き直しの判定 (2026-09-03 08:20、Sonnet の検証、biglock 単独)

| 段階 | 連鎖 us/回 | in-model S=1 (97 層発火) |
|---|---|---|
| 書き直し前 | 44.3 | 32.0 ms (素 23.9) |
| barrier 8→2 + inject 畳み | 44.7 | — |
| + down の内積を 32 simdgroup に分担 | **41.1** | **31.4 ms** (素 23.8、compiled 23.5、スタブ 19.2) |

40 tg × 8 行はファイル内の過去の掃引で 41.2 us (今の 16×32 の 39.9 より悪い)、uint4 の packed load は
既に入っていた。**反転条件 (連鎖 30 us) に届かず、in-model でも素より 7.6 ms 遅いので、この設計の
書き直しは畳む。**差分は既定 off の knob の裏に残す。HC の真の費用 4.7 ms は残ったまま。残る手は
(1) HC の低ランク射影 (10240→320→10240) を隣の行列積 (GDN の in_proj / attention の qkv) に畳んで
op と読み出しを減らす、(2) `MLXTURBO_HC=compiled` (mx.compile 版、-0.4 ms) を既定にする小さい取り分。
(2) は A/B が要る (短文脈 3 本 × 512、ms/round -1% 以上で既定 on)。

## 仮説マイクロ (モデル無し、biglock 単独、2026-09-03 08:40、`bench/results/logs/hyp-micros.log`)

**仮説 2: 事前確保した KV バッファへの slice 代入が全長コピーになっている疑い** (相談役)

| N (kv) | `buf[..., i:i+1, :] = x` 1 回 (同期込み) | `mx.slice_update` | バッファ | 全長コピーの床 (400 GB/s) |
|---|---|---|---|---|
| 4096 | 242 us | 244 | 4 MB | 10 us |
| 17000 | 306 | 304 | 17 MB | 44 us |
| 50000 | 481 | 478 | 51 MB | 128 us |

同期の床 (~230 us) を引くと 4k 10 / 17k 75 / 50k 250 us で、**バッファ長に比例している**。この micro では
Python 変数と返り値の 2 参照があるので donation が効かず、コピーになるのは当然でもある。本番の
`KVCache.update_and_fetch` (`self.keys[..., prev:offset, :] = k` の後に `self.keys[..., :offset, :]` の view を
返す) で同じことが起きているかは実モデルで確かめる: 1 ステップの `mx.get_peak_memory` の増分が KV 実体
(17k で 12 層 × K/V 420 MB) と同じ桁なら毎ステップ写している。当たれば 17k で +2 ms、50k で +6 ms の
decode kv 罰の正体で、相手 (Zig の refcount 共有) は払っていない項目。

**番外: MoE decode を take + quantized_matmul に** (相手の形): M=1 で gather_qmm 240 us 対 take+qmm 437 us、
M=2 で 323 対 697、M=3 で 292 対 538。**gather_qmm の方が速い。棄却。**

## n-gram 同期の前倒し (`MLXTURBO_PLE_HOIST`) の前提訂正 (2026-09-03 09:00)

相談役の仮説 3 は「PLE 層が約 5 層あり、forward が 5〜6 区間に分断される」が前提だったが、Flash-Next の
`ple_layer_ids` は `[2]` で **PLE 層は 1 つ**。同期は元々 1 forward に 1 回。knob は実装した (同期を層 2 の
途中から層ループの前に寄せる) が、見込みは小さい (`ngram-prefetch` の 0% の前例に近い)。A/B は短文脈 3 本と
17k × 64 だけ取り、-1% 未満なら畳む。

## depth 適応制御 4 回目 (margin 0.15、2048 超だけ、2026-09-03 09:10、`bench/results/depth-adapt4-17k.json`)

17k: ms/tok **-4.0%**、tok/round +5.7% (1.73 対 1.64)、ms/round +1.3%。depth 2 を 13〜45 ラウンドだけ選ぶ
(保守的)。4 回の A/B (-6.0 / -3.2 / -3.2 / -4.0%) が全部負側で、短文脈は静的 depth 2 のまま (害なし)。
ゲート -5% には 1 回しか届いていないが、方向が安定しているので **`MLXTURBO_DEPTH_ADAPT` を既定 on にする**
(2048 超だけ効く。`=0` で戻る)。margin と explore の調整は伸びしろとして残す (17k で -6% が上限の目安)。

## T=1 gather prefill カーネルの in-model A/B、17k (2026-09-03 09:40、`bench/results/prefill-attn-17k.json`、回文順 3 本)

prefill_s A 31.1 / 31.4 / 32.1 / 33.5 / 33.8 対 B 31.7 / 31.9 / 33.4 / 33.9 / 34.6 / 34.6 → **-1.5%** (発火 48 回 =
12 層 × kv ≥ 12288 の 4 チャンク。見込み -1.5% と一致)。decode は A が +3〜5% 遅い (17.1〜17.8 対
16.6〜16.9 ms/tok): decode 幅 (S=2 < MIN_S 64) でカーネルが辞退した後、古い gather 経路に落ちるため。
decode 幅では既定と同じ経路を通すように直す。50k の A/B (本命、-10% 以上が採用条件) は走行中。

## T=1 gather prefill カーネルの in-model A/B、50k (2026-09-03 00:08、`bench/results/prefill-attn-50k.json`、回文順 3 本)

prefill_s **A 102.3 対 B 130.0 s (-21.3%)**、3 本とも A が 25 s 以上速い (発火 240 回 = 12 層 × 20 チャンク)。
decode ms/tok も -2.3% (decode 幅の経路ずれは修正前の走行だが、50k では gather 経路が既定でも走るので差が
出ない)。**採用条件 (-10%) を大きく超えた。**長文脈の KLD (`tools/kld_prefill_attn.py`、17k / 25k) が
0.001 未満 (不変) か 0.01 未満 (小) なら `MLXTURBO_PREFILL_ATTN` を既定 on にする (kv ≥ 12288 だけ効く)。
見込みの 100k: dense は kv に比例して伸びるので、-30% 級。

## lm_head 4-bit 本焼き (真 bf16 から g64) の KLD (2026-09-03 00:11、`bench/results/quant-eval/compare-head4-baked-0903.json`)

kld_mean **0.01794** / agree 0.962 (現行 8-bit + GDN Metal: 0.01326 / 0.966)。**+0.0047** で、rebit (二重量子化)
の +0.0054 とほぼ同じ。「焼けば +0.0015」の見込みは外れ。受け入れ幅 (+0.0005) の 9 倍。
**本番は lm_head 8-bit のまま。**`~/models/ddalcu-mlxlm-head4` は「相手の一律 4-bit と条件を揃えた速度比較用」
としてだけ使う (公開ベンチでは注記付き)。

## 小さい結果、lm_head 4-bit 本焼き + depth 適応 on (2026-09-03 00:30、反復なし、`bench/results/self-snapshot-turbo-head4baked-0903.json`)

| 文脈 | 冷 TTFT | 温 TTFT | decode tok/s (01:40 の 8-bit 版 →) | tok/step (冷ターン) |
|---|---|---|---|---|
| 0 | 0.48 | 0.44 | 50.1 → 51.8 | 1.75 |
| 4k | 7.19 | 0.45 | 50.6 → 58.7 | 2.41 (テキスト運) |
| 17k | 32.5 | 0.50 | 48.7 → 45.5 | 1.96 (depth 適応で 1.75 → 1.96) |
| 50k | 121.9 | 0.60 | 42.4 → 45.6 | 2.38 |

反復 1 回なので tok/step の揺れ (1.40〜2.41) が decode を ±15% 動かす。17k の冷ターンは tok/step 1.96 で
depth 適応が効いているが、温ターン (42.6) が中央値を下げた。この表は「速度比較用パック + depth 適応」の
存在確認まで。判定は decode_ab の複数プロンプト平均で済ませてある (depth 適応 -4%、head4 は +4% 級の見込み)。

## `MLXTURBO_HC=compiled` (mx.compile 版 HC) の短文脈 A/B (2026-09-03 00:45、`bench/results/hc-compiled-short.json`)

ms/round **+0.6%** (取り分なし)。ms/tok -4.1% は tok/round +4.5% (2.28 対 2.18) によるもので、出力は一致して
いるのに受理数が違う = draft 側の HC の丸めが変わって draft が変わっただけ (テキスト運)。**採用しない。**
HC の 4.7 ms は素の op 列でも compiled でもカーネルでも取れない。残る手は低ランク射影を隣の行列積に畳む案のみ。

## グループ幅 G=8 対 4 の 17k prefill A/B (2026-09-03 01:00、`bench/results/prefill-group8-17k.json`)

prefill_s 8=33.5 対 4=34.0 (**-1.2%**)、decode ±0、出力一致。M/E が 320 になってもタイル水増しの改善
(1.22 → 1.10 の見込み) は prefill 全体の 1% にしか出ない。反転条件 (-2% 未満) に該当。**畳む (既定 4 のまま)。**
MoE の「残り 1.5 倍」は gather_qmm 自体で、行数を太らせても縮まないことがこれで確定。

## PLE hoist の短文脈 A/B (2026-09-03 01:10、`bench/results/ple-hoist-short.json`)

ms/round +0.6%、出力一致。前提 (PLE 層 1 つ) どおり取り分なし。17k の結果が -1% 未満なら畳む (既定 off のまま)。

## PLE hoist の 17k A/B (2026-09-03 01:45、`bench/results/ple-hoist-17k.json`)

prefill_s +0.1%、ms/round -0.1%、出力一致。**畳む** (既定 off のまま)。相談役の仮説 3 は前提 (PLE 5 層) が
外れていた。同期 1 回の位置を動かしても取り分は無い。

## KV キャッシュの全長コピー、実モデルの探針 (2026-09-03 01:50、scratchpad/kv_copy_probe.py、biglock 単独)

| ctx | KV 実体 (12 層 K/V) | S=1 forward 壁 | ステップ 1 回のピークメモリ増分 | `update_and_fetch` 1 層 (同期込み) |
|---|---|---|---|---|
| 4k | 95 MB | 32.2 ms | **130 MB** | 243 us (keys 4.2 MB) |
| 17k | 415 MB | 36.9 ms | **519 MB** | 322 us (17.5 MB) |
| 50k | 1226 MB | 46.1 ms | **1495 MB** | 519 us (51.3 MB) |

ピーク増分が KV 実体の 1.2 倍で伸びる = **ステップごとに KV 全体を写している**。forward 壁の 4k → 50k
+14 ms のうち、1.2 GB の読み書き (≈ 6 ms) がこれ。**ただし**この探針は各 rep の前に `_restore(caches, snap)`
で snapshot に戻しており、snapshot が古いバッファの参照を持つので donation が効かずコピーになるのは
当然でもある。本番の decode (KV は snapshot しない、rollback は GDN 状態だけ) で同じことが起きるかは、
復元なしで連続 step する `--no-restore` 版 (chain53) で決める。そちらでも増分が KV 比例なら、原因は
`update_and_fetch` の view 返し (`self.keys[..., :offset, :]`) か capture/rollback が持つ参照で、
`mx.slice_update` を donation が効く形に組み直す (相手は Zig 側の refcount 共有で払っていない項目)。

## TTFT の内訳、ctx 0 と 4k (2026-09-03 02:05、`--log-level debug` の `[ttft-trace]`、`bench/results/logs/ttft-trace-driver.log`)

| リクエスト | ハーネス TTFT | runner の ttft | `[ttft-trace]` gen→first_token | 生成前 (parse→gen) |
|---|---|---|---|---|
| ctx 0 冷 (new=27) | 0.52 s | 0.19 | 498 ms | 19 ms (template 初回) |
| ctx 0 完全ヒット (reused=27 new=0) | 0.30 s | **0.00** | **305 ms** | 1 ms |
| ctx 0 温 (reused=58 new=23) | 0.45 s | 0.16 | 452 ms | 1 ms |
| 4k 冷 (new=3827) | 7.91 s | 7.61 | 7904 ms | 5 ms |
| 4k 温 (reused=3858 new=18) | 0.46 s | 0.15 | 458 ms | 4 ms |

**完全ヒットで runner の時計が 0.00 s でも、ハンドラは最初のトークンを 300 ms 後に受け取る。**
温ヒットも 450 = 150 (runner) + 300。つまり **リクエストごとに固定 300 ms** が「executor への投入 → runner の
t0」か「runner の最初の yield → ハンドラのキュー受信」にある。tokenize / template / セッション照合は 1〜5 ms
で無罪。これを潰せば温 TTFT 0.45 → 0.15 s (相手 0.87 の 6 倍速)、冷 ctx 0 も 0.5 → 0.2 s。内訳の時刻印
(`[gen-trace]`) を足して次に測る。

## KV 全長コピー、復元なしの探針 (本番の素の forward の形、2026-09-03 02:30)

| ctx | S=1 forward 壁 | ピーク増分 | `update_and_fetch` 1 層 |
|---|---|---|---|
| 4k | 30.7 ms | 22 MB | 190 us |
| 17k | 32.8 | 23 | 191 |
| 50k | 33.7 | 28 | 191 |

snapshot の参照が無ければ **KV の更新は in-place** (増分・時間とも kv に依らない)。素の forward では仮説 2 は
成り立たない。ただし復元ありの探針では 50k で forward +12 ms (1.2 GB の写し) が出ているので、**本番の投機
ラウンド**で `pre` / `capture` / checkpoint が K/V 配列の参照を持っていれば同じことが起きる。ラウンドごとの
ピーク増分を `MLXTURBO_ROUND_TRACE` に足して 17k / 50k で確かめる (実装中)。
素の forward の kv 罰は 4k → 50k で +3 ms しかない (attention 層 +2 ms が主)。実 decode の +8.5 ms/tok との
差が、この投機ラウンド側の候補 (KV コピー、S=2 の verify、draft の MTP) にある。

## T=1 gather prefill カーネルの長文脈 KLD (2026-09-03 02:50、`bench/results/kld-prefill-attn.json`、末尾 64 位置、top-256)

| ctx | 発火 | kld_mean | kld_max | argmax 一致 | top-5 重なり |
|---|---|---|---|---|---|
| 17k | 48 (12 層 × 4) | **0.0398** | 1.19 | 1.00 | 0.88 |
| 25k | 96 (12 層 × 8) | 0.0175 | 0.61 | 0.98 | 0.88 |

受け入れ幅 (+0.0005) の 30〜80 倍。合成の誤差 7e-3 では説明できない大きさで、**可視集合の解釈 (末尾の未確定
ブロックの扱い) か GQA の対応が dense 経路と違う疑い**。速度 (-21% at 50k) は正しさが付くまで採用しない。
`MLXTURBO_PREFILL_ATTN` は既定 off のまま。原因調査中。

## prefill 短文脈の内訳、8k (2026-09-03 03:10、`tools/prefill_anatomy.py --ctx 8000`、完全チャンク 3 本、部品和 ≈ 壁時計 ±3%)

| 部品 (1 チャンク 2048 tok) | チャンク 0 (kv 2k) | チャンク 2 (kv 6k) | 効率 (FLOP 下限比) |
|---|---|---|---|
| MoE 48 層 | 1484 ms | 1649 | 59〜66% (gate/up/down の qmm が各 440 ms、水増し 1.41) |
| GDN 36 層 | 924 | 961 | 80〜83% |
| attention 12 層 (indexer 込み) | 353 | 629 | 78% (indexer は 7〜29 ms) |
| HC 97 回 | 378 | 390 | 60〜62% (素の op) |
| **PLE 1 層 (n-gram 行取得込み)** | **167** | **167** | **7%** |
| 壁時計 | 3416 | 3810 | |

読み方 (短文脈の prefill が遅い理由):
- MoE 43% + GDN 27% が大半で、どちらも FLOP 効率 60〜80% で計算律速。相手も同じ op なので差の主因ではない。
- **PLE の n-gram 行取得 167 ms/チャンク (4.4%) は効率 7%** で、pread の I/O 待ち。次チャンクの行を GPU の
  実行中に先読みすれば丸ごと隠せる (`MLXTURBO_NGRAM_PREFETCH` は decode で却下されたが prefill は未測)。
  A/B: `decode_ab --knob ngram-prefetch --only long --ctx 8000 --tokens 8`。
- HC 97 回 = 380 ms (11%) は素の op で 60%。prefill 幅の融合は -0.9% で却下済み。
- attention は kv=2k で 353 ms (10%)、うち射影が大半 (S=2048 の qkv/o_proj)。
- 4k の TTFT 7.2 s = 2 チャンク弱 × 3.5 s + 固定費 (prime 0.25 s、固定 300 ms の TTFT 経路)。

## n-gram 先読みの prefill A/B、8k (2026-09-03 03:40、`bench/results/ngram-prefetch-8k.json`)

prefill_s A (先読み on) 13.31 対 B 13.42 (**-0.9%**)、decode ±0。PLE の 167 ms/チャンクは pread の待ちでは
なく、行取得後の処理 (ハッシュ、gather、埋め込みの加算) が主で、先読みでは隠れない。反転条件 (-2% 未満) で
畳む (既定 off のまま)。PLE の効率 7% は別の手 (行取得の GPU 側処理の整理) が要る。

## 固定 300 ms の内訳 (2026-09-03 04:00、`[gen-trace]` + `[ttft-trace]`、`bench/results/logs/ttft-trace-driver2.log`)

完全ヒット (reused=27 new=0): `[gen-trace] entry→cache 0.0, cache→t0 0.0, t0→first 0.9, first→queue 0.0 ms`、
executor の submit→run 0.1 ms、なのに `[ttft-trace] gen→first_token 310 ms`。温ヒット (new=23) も
runner 内 154 ms + 外 345 ms。**runner でも executor でもなく、worker がキューに積んだ最初のトークンを
ハンドラが受け取るまでに 300 ms 掛かっている。**ハンドラ側のキュー待ち (poll 間隔や keepalive の timeout) か、
イベントループを塞ぐ処理 (detokenizer の構築など) の疑い。ここを潰せば温 TTFT 0.45 → 0.15 s。

## 固定 300 ms の正体: リクエストごとの detokenizer 構築 (2026-09-03 04:20)

`tokenizer.detokenizer` (mlx_lm の `@property`) は呼ぶたびに `BPEStreamingDetokenizer` を作り直し、248,077 語彙の
`tokenmap` を Python ループで組む。**実トークナイザで 1 回 105 ms** (103 / 111 / 106 ms)。`ThinkingRouter.__init__`
がリクエストごとにこれを 3 回、イベントループのスレッドで呼ぶ → **約 315 ms** が全リクエストの TTFT に固定で
乗る (`[ttft-trace]` gen→first_token 310 ms、runner 内 0.9 ms、executor 0.1 ms と整合)。
相手は Zig 側でトークナイザを 1 回だけ組む。直し方: prototype を 1 回作って複製 + reset。見込み: 温 TTFT
0.45 → 0.15 s (相手 0.87 の 6 倍速)、冷 ctx 0 も 0.5 → 0.2 s、4k 冷は 7.2 → 6.9 s。

## 投機ラウンドごとのピークメモリ (2026-09-03 04:40、`decode_ab --knob null --round-trace`、64 トークン)

| ctx | ラウンド数 | peak_delta 中央値 | 最大 (1 ラウンド目 = ハーネスの復元由来) |
|---|---|---|---|
| 17k | 26 | **33 MB** | 880 MB |
| 50k | 29 | **133 MB** | 1862 MB |

本番の投機ラウンドで KV 全体 (17k 415 MB / 50k 1.2 GB) を毎回写してはいない (中央値が KV の 1 割以下)。
50k の 133 MB は indexer の pooled キャッシュの concat (ブロック確定ごと) 級で、費用は 1 ms/round 未満。
**KV コピー仮説は本番では棄却。**decode の kv 罰 (+8.5 ms/tok) の帰属は attention 層の verify 幅 (S=2) の
中身に戻る。`qsa_prefill_split --S 2 --chain 50` (同期の床を外した部品計測) で決める。

## detokenizer 修正後の TTFT (2026-09-03 04:50、`bench/results/logs/ttft-trace-driver3.log`)

| リクエスト | 修正前 | 修正後 |
|---|---|---|
| ctx 0 冷 (new=27) | 0.52 s | **0.23 s** (runner 0.20) |
| ctx 0 完全ヒット | 0.30 s | **0.003 s** |
| ctx 0 温 (new=23) | 0.45 s | **0.151 s** |
| 4k 冷 | 7.28〜7.91 s | **6.96 s** |
| 4k 温 (new=18) | 0.46 s | **0.154 s** |

`[ttft-trace]` の gen→first_token が完全ヒットで 310 → 1.2 ms。**全リクエストから 300 ms が消えた。**
温 TTFT は相手 (0.87〜0.92 s) の 6 倍速。既定に入った (コミット 3ecf835)。

## decode 幅 S=2 の attention 部品、連鎖版 (2026-09-03 05:10、`qsa_prefill_split --S 2 --chain 50`、部品和 ≈ 壁時計 ±3〜10%、us/層/回)

| kv | 経路 | indexer | マスク/選択 | sdpa | その他 (qkv+gate+o_proj、**復元コピー込み**) | 壁時計 |
|---|---|---|---|---|---|---|
| 4096 | dense | 228 | 0.2 | 71 | 190 | 477 |
| 17000 | dense | 235 | 5.7 | 153 | 324 | 717 |
| 25000 | gather | 246 | 136 | 71 | 423 | 855 |
| 50000 | gather | 307 | 185 | 71 | 712 | 1162 |

- 「その他」の kv 比例は連鎖モードが各回の前にキャッシュを復元するため `update_and_fetch` が全長コピーになる
  分 (17k で +90 us ×2、50k で +260 us ×2) で、本番には無い。差し引くと「その他」は約 190 us で kv に依らない。
- kv に比例する本物: **indexer 228 → 307 us**、**gather 経路のマスク/選択 0 → 185 us** (25k 以上)、dense 経路の
  sdpa 71 → 153 us (17k、bool マスクの vector でもキーを読む分)。合計で 4k → 50k は +270 us/層 =
  **+3.3 ms/round (12 層)**。17k は +240 us/層 = +2.9 ms/round。
- indexer が 4k でも 228 us/層 = 2.7 ms/round と、sdpa (71 us) の 3 倍ある。**decode の attention 層で最大の
  部品は indexer** (pooled スコア、argpartition、keep ブロック構築の小さい op 列)。op 整理の対象。
- decode の kv 罰 (+8.5 ms/tok ≈ +15 ms/round) のうち attention 層で説明できるのは +3 ms。残りの帰属は
  ラウンド全体の built / eval の内訳 (`[round]` trace) で追う。

## decode の kv 罰の帰属、ラウンド間隔で (2026-09-03 05:20、`[round]` trace、`--knob null --tokens 64`)

ラウンドの CPU 側の刻み (eval_done 4.7 / verify_done 5.4 / drafts_submitted 7.1 / rollback 7.2 ms) は 17k と 50k で
ほぼ同じ。**ラウンド間隔は 17k 43 ms、50k 45〜46.5 ms (+2〜3 ms)** で、attention 層の kv 比例 (+3.3 ms/round、
上の節) と一致する。小さい結果で見えた「50k で +8.5 ms/tok」は、120 s の prefill 直後の熱と tok/step の揺れ
(反復 1 回) が大半で、**本物の kv 罰は +3 ms/round (7%) 程度**。帰属は indexer (228 → 307 us/層) と
gather 経路のマスク/選択 (25k 以上で 136〜185 us/層)。相手の kv 罰 +3.0 ms (ubench) と同じ桁で、
「うちは 2 倍」という朝の読みは熱の混入だった。レーン 8 の decode 側はこれで閉じる。
indexer の 228 us/層 (kv に依らない分) は decode 全体で 2.7 ms/round あり、op 整理の対象として残す。

## gather カーネルの分布ずれの原因 (2026-09-03 05:40、Sonnet の調査)

コードのバグではない。可視集合の構成、rope / qk-norm / gate の順、GQA の head 対応は dense 経路と同一 (読解 +
配列レベル検査 4 件合格)。原因は **online softmax の加算順が dense と違う (誤差 7e-3、既知) → 1 層目の出力が
わずかに違う → 次の full attention 層の indexer の top-k (離散) が境界で反転 → 層を重ねて育つ**カスケード。
合成モデル (本番と同じ cr=4 / budget 2048 / head_dim 256) で、最初の発火層は keep_block 反転 0、2 層目以降で
全行が反転し始めることを実測。実モデルでは学習済み重み (境界のタイが多い) と 12 層の積み重ねで KLD 0.04 に
なる。argmax 一致 1.0 / top-5 重なり 0.88 とも整合。

判定の考え方: この「ずれ」は品質の劣化なのか、それとも dense 経路自身が持つ丸めの混沌 (chunk 幅を変えても同じ
規模で動く) なのか。**dense 対 dense (chunk 2048 対 4096、どちらも厳密な意味論、丸めだけ違う) の KLD** を同じ
道具で取り、0.03 級なら「経路の内在的な揺らぎの範囲」として採用 (注記付き)、0.001 級なら不採用。
副次: `kld_prefill_attn` の 2 つ目以降の ctx で gather 経路が残る汚染を直した (25k の値は取り直し)。
`tools/verify_prefill_attn.py` のモデルレベル検査は最初からカーネルが発火していなかった (S=32 < MIN_S、kv 不足)。

## gather カーネルのカスケード、実モデル 17k (2026-09-03 06:00、`tools/prefill_attn_layer_probe.py`)

| full attention 層 | 出力の最大差 | keep_block 反転 (行数 / 最大本数 / 512) |
|---|---|---|
| 3 (最初の発火層) | 7.8e-3 | **0 / 0** |
| 7 | 4.3e-2 | 63 / 20 |
| 11〜19 | 4〜9e-2 | 64 / 32〜58 |
| 23〜31 | 0.17〜0.29 | 64 / 96〜330 |
| 35〜47 | 0.18〜0.64 | 64 / 324〜416 |

最初の発火層は選択が完全一致で丸め誤差 (7.8e-3) だけ。次の層から反転が始まり、後段では 512 ブロック中
300〜400 本が入れ替わる。**選択の反転が層ごとに育つカスケードが実モデルでも確定。**後段の可視集合が
大きく違うのに argmax 一致 1.0 なのは、QSA の選択が冗長 (どのブロックを見ても同じ答えに収束する) ため。
dense 対 dense (chunk 幅違い) で同じ規模の反転が出るかで判定する。

## 小さいベンチ、冷却強化後 + detokenizer 修正 + depth 適応 on (2026-09-03 06:50、mlxturbo だけ、反復なし、8 分冷却、`bench/results/self-snapshot-turbo-small-0903b.json`)

| 文脈 | 冷 TTFT | 温 TTFT | decode tok/s | 01:40 の値 (冷 TTFT / 温 / decode) | 相手 (9/2 朝) |
|---|---|---|---|---|---|
| 0 | **0.17** | **0.13** | 50.3 | 0.48 / 0.44 / 50.1 | — |
| 4k | **6.89** | **0.15** | 51.4 | 7.18 / 0.46 / 50.6 | 5.77 / 0.87 / 55.3 |
| 17k | **31.7** | **0.20** | 46.8 | 32.5 / 0.50 / 48.7 | 29.2 / 0.92 / 56.0 |
| 50k | **117.3** | **0.31** | 44.0 | 119 / 0.61 / 42.4 | 108 / 22.8 / 30.8 |

- 温 TTFT が全点で 3 倍速く (detokenizer 修正)、相手の 0.87〜0.92 s の 4〜6 倍速。ctx 0 の冷 TTFT も 0.48 → 0.17 s。
- 冷 prefill は -1.5〜-4% (detokenizer の 300 ms + 冷却)。対相手 4k 1.19x / 17k 1.09x / 50k 1.09x 負け。
- decode は反復 1 回の tok/step の揺れの範囲 (17k は 1.71、前回 1.75)。対相手 4k -7% / 17k -16% / 50k +43%。
- これが「既知を片付けた」時点の基準 (gather カーネルと indexer の op 整理は未採用)。

## dense 対 dense の対照 (chunk 2048 対 4096)、17k (2026-09-03 07:05、`bench/results/kld-dense-chunk4096.json`)

kld_mean **0.374**、kld_max 5.19、argmax 一致 0.92、top-5 重なり 0.84。カーネル (0.040) の 1 桁上。
**ただし対照として汚れている**: QSA は現在のチャンク内を因果で全部見せ、それ以前の確定ブロックから top-k を
選ぶので、chunk 幅を変えると「全部見える」範囲が倍になり、可視集合の意味論そのものが変わる
(`MLXTURBO_PREFILL_CHUNK=4096` が in-model で負けた記録にも、この意味論の差が混ざっていた可能性)。
丸めだけが違う対照 (GDN Metal の on/off、端数チャンクの畳み込み on/off) で取り直す。

## 丸めだけ違う対照 (GDN Metal on 対 off) の長文脈 KLD、17k (2026-09-03 07:40、`bench/results/kld-gdn-metal-off.json`)

kld_mean **0.111**、kld_max 1.34、argmax 一致 0.95、top-5 重なり 0.85。GDN Metal は短い continuation の KLD で
+0.00014 (受理済み、本番既定) の変更なのに、17k の末尾では gather カーネル (0.040) の **3 倍**動く。
**長文脈の「dense 対 変種」KLD は、QSA の top-k 反転カスケードのせいで、どんな丸めの違いでも 0.04〜0.1 に
なる**。この物差しでは品質の劣化を測れない (経路自身の混沌を測っている)。
判定: gather カーネルの 0.040 は既に受け入れている変更 (GDN Metal) の揺らぎより小さい。品質のゲートは
本来の物差し (`quant_eval.py compare --fusions`、bf16 参照との KLD、longctx-recall / longctx-quote を含む) で
取り、+0.0005 以内なら **`MLXTURBO_PREFILL_ATTN` を既定 on (kv ≥ 12288)** にする。

## indexer-lean の A/B (2026-09-03 07:30)

短文脈 +0.4%、17k +0.3% (出力一致)。op -8% では取り分が出ない。**畳む** (既定 off のまま)。

## T=1 gather prefill カーネルを既定 on にする判断 (2026-09-03 08:00)

- 本来の品質ゲート `quant_eval.py compare --fusions` (bf16 参照) は kld_mean 0.01326 / agree 0.966 で不変。ただし
  continuation の prompt は最長 597 トークンで **カーネルは 1 度も発火していない** (ゲートの盲点)。
- 長文脈の自前 dense 比の KLD は 0.040。同じ物差しで、受理済みの GDN Metal (on 対 off) は 0.111。QSA の top-k
  カスケードのせいで、この物差しはどの丸めの違いでも 0.04〜0.1 になる。カーネルはその範囲の下側。
- 合成の誤差は 7e-3 (GDN Metal と同じ級)。速度は 50k prefill -21.3%、17k -1.5%。
- **既定 on (kv ≥ 12288)。**`MLXTURBO_PREFILL_ATTN=0` で戻る。
- 宿題: 長文脈の品質は bf16 参照が無いので、課題の正答率 (17k / 50k の recall・quote の正解率、dense 対 カーネル) で
  ゲートを作る (レーン 11 に追加)。

## 4-bit の group_size 32 / 64 / 128 (モデル無し、M=1、同期込み、差だけ読む、2026-09-03 08:20)

| 行列 | g32 | g64 (本番) | g128 |
|---|---|---|---|
| q_proj 2560→6144 | 223 us | 224 | 224 |
| GDN in 2560→10240 | 277 | 279 | 275 |
| lm_head 2560→248320 (4-bit) | 1213 | 1109 | **1055 (-5%)** |
| MoE gather M=1 top-10 | 179 | 189 | **176 (-7%)** |

小さい行列は差なし (同期の床)。帯域律速の lm_head と MoE で g128 は 5〜7% 速い。forward 全体では
lm_head -55 us + MoE 48 層 × -13 us ≈ **-0.7 ms (3%)**。パック全体の焼き直し (数時間) と KLD の代償
(g128 は粗い) に対して取り分が小さい。反転条件 (3% 未満) に該当、**畳む**。lm_head だけ g128 という
部分適用も lm_head 4-bit 自体が KLD 幅外なので無し。

## `mx.metal.start_capture` によるカーネル別計測は使えない (2026-09-03 08:55)

S=1 forward 1 回の capture が 35 分経っても終わらず、`.gputrace` が 68 GB に達したので止めて削除した。
MLX の capture はモデルの全バッファ (91 GB) を含めて記録する。相談役の「計測手段 0」は畳む。
カーネル別の帰属は、これまでどおり部品の連鎖計測 (`qsa_prefill_split --chain`、`kernel_chain_cost`) で取る。

## 小さいベンチ、gather カーネル既定 on (2026-09-03 09:10、10 分冷却、反復なし、`bench/results/self-snapshot-turbo-small-0903c.json`)

| 文脈 | 冷 TTFT | 温 TTFT | decode | gather 前 (06:50) | 相手 (9/2 朝) | 対相手 |
|---|---|---|---|---|---|---|
| 0 | 0.17 | 0.13 | 50.3 | 0.17 / 0.13 / 50.3 | — | |
| 4k | 6.89 | 0.15 | 51.5 | 6.89 / 0.15 / 51.4 | 5.77 / 0.87 / 55.3 | prefill 1.19x 負け、decode -7% |
| 17k | **31.0** | 0.20 | 47.5 | 31.7 / 0.20 / 46.8 | 29.2 / 0.92 / 56.0 | prefill 1.06x 負け、decode -15% |
| 50k | **94.2** | 0.31 | 43.4 | 117.3 / 0.31 / 44.0 | 108 / 22.8 / 30.8 | **prefill 0.87x (勝ち)**、decode +41% |

**50k の冷 prefill が相手より速くなった** (94 対 108 s)。17k は 1.06x まで詰まった。朝 (9/2 11:09) からの合算:
冷 prefill 4k -17% / 17k -18% / 50k -42%、温 TTFT 3 倍速、decode 4k +6% / 17k +16% / 50k +26%。
これが仮説 A/B に入る前の基準。

## rerank on/off と draft の hit@2 (2026-09-03 09:30、`bench/results/draft-rerank-short.json`、短文脈 3 本 × 512)

- rerank on (A) 対 off (B): ms/round **-5.2%** (A が速い)、tok/round -0.4% (差なし)。粗い 2-bit lm_head での候補絞りは
  受理率を削っていない。**仮説 7 (rerank が受理率を削る) は棄却。rerank on のまま。**
- draft の 1 段目: hit@1 = 0.59〜0.70、hit@2 = 0.74〜0.87 (rounds 233〜258)。**hit@2 − hit@1 = 14〜17 ポイント**。
  木 (top-2 を両方 verify) の上限は 1 段目の受理 +0.17。tok/round 2.2 → 2.4〜2.5 (+10%) の見込みだが、verify 幅が
  +1 (S=3 → 4) で MoE の重み読みと attention が +5 ms/round (+13%) 増えるので **ms/tok は同着〜負け**。長文脈
  (depth 1、S=2 → 3) も同様。**仮説 5 (木) は畳む。**取り分が出るのは verify 幅の 1 行追加費用が 2 ms を切った場合
  (MoE の重み読みが支配的な今の構造では無理)。

## rerank on/off、17k (2026-09-03 09:50、`bench/results/draft-rerank-17k.json`)

ms/round A (rerank on) **-4.6%**、tok/round -0.5%。hit@1 0.59〜0.60、hit@2 0.75〜0.76 (rounds 263〜266) →
差 16〜17 ポイント (短文脈と同じ)。判定は短文脈と同じ: rerank は維持、木は畳む。

仮説 A/B の残り: temp>0 の厳密棄却サンプリング (上限測定には decode_ab に temp の knob が要る、未)、HC の
split-K カーネル (走行中)、長文脈の品質ゲート (課題の正答率)。

## HC カーネル、split-K の 3 カーネル案 (2026-09-03 10:20、Sonnet、連鎖 N=200)

| 案 (down の threadgroup 数 / 1 行の分割) | 連鎖 us/回 |
|---|---|
| 現行 2 カーネル (20 tg、2 分割) | 42.7 |
| stats 分離 + 80 tg × 4 行 (8 分割) | 44.9 |
| stats 分離 + 160 tg × 2 行 (16 分割) | 44.8 |
| stats 分離 + 320 tg × 1 行 (32 分割) | 49.6 |

threadgroup を増やしても速くならず、カーネルを 3 発に分けた起動と normed の往復が上回る。**HC カーネルの
ジオメトリ書き直しは 2 回目の独立した否定結果。レーン 1 の HC カーネルは畳む** (既定 off のまま)。
HC の 4.7 ms は、素の op 列 (12 op × 97 回) の起動レイテンシそのもので、mx.compile (-0.4 ms) でも取れない。
残る設計変更 (低ランク射影を隣の行列積に畳む) は入力が違う (normed 10240 対 mixed 2560) ので直接は畳めず、
現状の構造では手が無い。decode 短文脈の対相手 -7% はここで止まる。

## temp 0.7 対 greedy の tok/round (2026-09-03 10:50、`bench/results/temp-short.json`、短文脈 3 本 × 512)

tok/round 0.7 = 2.02 対 0.0 = 2.22 (**-8.8%**)、ms/round +1.4%、ms/tok +12.4% (EOS で短い本あり)。
現行の「verify の logits からサンプルし greedy の draft と一致したら採用」でも受理は greedy の 0.91 倍を保っている。
厳密な棄却サンプリング (受理確率 min(1, p/q)) で取り戻せる上限は約 9% で、draft 側の完全な logits (lm_head 1 回
分 ≈ 1 ms/段) の費用と相殺する。反転条件 (0.9 倍以上なら畳む) に該当。**仮説 6 は畳む。**

## mlx-serve の版の確認 (2026-09-03 11:00)

`~/dev/mlx-serve` は origin/main と同じ (遅れ 0)。HEAD 8058076 (2026-09-01 15:41 EDT、YaRN rope 1M ctx) で、
`zig-out/bin/mlx-serve` (9/2 10:18 JST) はその後にビルドされているので **最新版のビルド済み**。フルベンチは
このバイナリで取る (再ビルド不要)。

## 長文脈の品質ゲート、1 回目 (2026-09-03 11:45、`tools/longctx_quality.py --ctxs 17000 --n 6`)

recall: dense 1/6、kernel 0/6。quote: 両方 0/6。dense がほぼ 0 なので**課題が成立していない** (thinking が on の
まま 32〜64 トークンで答えが出ない、または質問の置き方)。直して実モデルの小さい健全性確認 (2k、n=3) を
通してから取り直す。gather カーネルの既定 on はこの結果では動かさない (判定材料になっていない)。

## フルベンチ、mlx-serve 側 (2026-09-03 12:00、origin/main 8058076 のバイナリ、冷却 10 分後、反復 1、256 トークン、thinking off、`bench/results/self-snapshot-full-0903.json`)

| 文脈 | 冷 TTFT | 温 TTFT | decode tok/s | 9/2 朝の値 |
|---|---|---|---|---|
| 0 | 0.18 | 0.72 | 54.7 | — |
| 4k | 5.70 | 0.85 | 55.7 | 5.77 / 0.87 / 55.3 |
| 17k | 27.8 | 0.91 | 49.0 | 29.2 / 0.92 / 56.0 |
| 25k | 41.3 | 0.89 | 60.8 | — |
| 32k | 51.6 | 0.90 | 47.4 | — |
| 50k | **82.1** | 15.9 | **45.9** | 108 / 22.8 / 30.8 |

相手も冷却強化の恩恵を受け、50k の冷 prefill が 108 → 82 s、decode が 30.8 → 45.9 に上がった (朝の「25k で崖」は
熱だった可能性が高い)。**対戦の数字は同じ日・同じ冷却で取り直したこの表を使う。**mlxturbo 側は同条件で取得中。

## フルベンチ (2026-09-03 12:20、両エンジン同日・同冷却、10 分冷却後、反復 1、256 トークン、thinking off)

`bench/results/self-snapshot-full-0903.json` (mlx-serve、origin/main 8058076) と `self-snapshot-full-turbo-0903.json` (mlxturbo)。

| 文脈 | 冷 TTFT serve / turbo | 温 TTFT serve / turbo | decode serve / turbo |
|---|---|---|---|
| 0 | 0.18 / 0.17 | 0.72 / **0.14** | 54.7 / 50.2 |
| 4k | 5.70 / 6.88 (1.21x) | 0.85 / **0.15** | 55.7 / 51.4 (-8%) |
| 17k | 27.8 / 31.6 (1.14x) | 0.91 / **0.20** | 49.0 / 45.3 (-8%) |
| 25k | 41.3 / 46.6 (1.13x) | 0.89 / **0.23** | 60.8 / 48.0 (-21%、serve の tok/step 運) |
| 32k | 51.6 / 58.3 (1.13x) | 0.90 / **0.25** | 47.4 / 44.9 (-5%) |
| 50k | 82.1 / 93.1 (1.13x) | 15.9 / **0.93** | 45.9 / 46.9 (+2%) |

読み方:
- **冷 prefill は 4k で 1.21x、17k 以上で 1.13x 負け。**朝の「50k 1.51x」は相手が熱で落ちていた分で、同条件では 1.13x。
  gather カーネル (-21%) を入れても相手も同じ条件で速くなったので、差は縮まったが逆転はしていない。
- **decode は 4k / 17k で -8%、32k -5%、50k +2%。**朝の「50k +38% 勝ち」も相手の熱だった。25k の -21% は相手の
  tok/step の当たり (60.8) で、反復 1 回の揺れ。
- **温 TTFT は全点で 4〜6 倍速、50k は 17 倍速** (相手は 50k で接頭辞キャッシュが効かず 15.9 s)。
- 反復 1 回なので decode の ±5% は揺れの範囲。prefill の 1.13x は安定した差。
残差の帰属 (今日の分析から): prefill は MoE の gather_qmm (相手と同じ op) 以外の、GDN の 20% 超過、HC の素の op 列、
PLE の行取得 (4%)、attention の射影。decode は HC の 4.7 ms と attention 層の小さい op (indexer 2.7 ms/round)。

## 長文脈の品質ゲート、17k (2026-09-03 13:40、`tools/longctx_quality.py --ctxs 17000 --n 8`、thinking off)

| 課題 | dense 正答率 | kernel 正答率 | 回答一致率 |
|---|---|---|---|
| recall (合言葉) | **1.000** | **1.000** | 0.875 |
| quote (次の文) | 0.625 | 0.625 | 0.875 |

kernel (発火 384 回 = 12 層 × 4 チャンク × 8 問) は dense と同じ正答率。回答が違った 1/8 は quote の別解。
**課題の正答率では gather カーネルの品質劣化は見えない。既定 on を裏付ける。**50k は走行中。

## 長文脈の品質ゲート、50k (2026-09-03 14:35、`tools/longctx_quality.py --ctxs 50000 --n 6`)

| 課題 | dense | kernel | 回答一致率 |
|---|---|---|---|
| recall | 1.000 | 1.000 | 0.833 |
| quote | 1.000 | 1.000 | 0.667 |

kernel (発火 1440 回 = 12 層 × 20 チャンク × 6 問) は 50k でも dense と同じ正答率。回答の文言が違う問 (1/6〜2/6) は
どちらも正解。**17k / 50k とも課題の正答率で劣化なし。T=1 gather カーネルの既定 on を確定。**

## 小物 1: prime 窓 2048 対 512、4k × 256 トークン (2026-09-03 15:00、`bench/results/prime-window-4k.json`)

prefill_s (TTFT) **-2.2%** (6.84 → 6.69 s)、tok/round -0.8%、ms/tok -0.2%。判定基準 (tok/round の低下 2% 未満で
TTFT が減る) を満たす。**`MLXTURBO_PRIME_WINDOW` の既定を 512 に。**取り分は固定 150 ms/prefill (17k では 0.5%)。

## 小物 2: PLE の n-gram 行取得の内訳、チャンク 2 (kv 4k〜6k、2026-09-03 15:00、`tools/ple_split.py`、`bench/results/ple-split.json`)

| 部品 | ms |
|---|---|
| ハッシュ (mx op) | 0.4 |
| 同期 (np.array) | 0.04〜5 |
| **pread の行取得 (冷)** | **252** (温 137) |
| 復号 (4-bit → bf16) | 0.7〜1.6 |
| 残り (射影 + gate 13、short conv 11、他) | 24〜102 |
| 壁時計 (冷) | 361 |

1 チャンク 2048 トークンで 32,768 行 (16 次 × 2048) × 1.3 KB = 42 MB を pread で読み、170〜310 MB/s しか出ない
(小さい読みの束、12 スレッド)。**I/O 待ちそのもの**で、GPU の実行と重ねれば丸ごと隠せる (2048 tok のチャンクの
GPU 時間 3.4 s に対して 250 ms)。`MLXTURBO_NGRAM_PREFETCH=1` が -0.9% しか出なかったのは、先読みが背景スレッドで
本当に重なっていないため (この計測では prefetch_rows=0)。手: チャンク c の forward を投入した直後に、チャンク c+1
の行をスレッドで pread し始め、c+1 の `_prelude` で待ち合わせる。見込み: prefill -4〜7% (全文脈長で)。

## 小物 3: MoE 重み付き和の畳み込み (`MLXTURBO_MOE_COMBINE_FOLD`)、8k (2026-09-03 15:20、`bench/results/moe-combine-8k.json`)

prefill_s **-2.2%** (14.11 → 13.81 s)、decode ms/round +1.3% (fold の経路は decode 幅でも SwitchGLU を迂回するため、
S=1〜3 では (rows, 640) の乗算と個別の gather の分だけ損)。短文脈と 17k の結果を見て、**prefill 幅 (S ≥ 64) だけ
fold にする**形で採用するかを決める。

## 小物 3 の続き: MoE 重み付き和の畳み込み、短文脈と 17k (2026-09-03 15:55)

| 条件 | prefill_s | ms/round | tok/round |
|---|---|---|---|
| 8k | -2.2% | +1.3% | ±0 |
| 短文脈 (decode 512) | — | +1.4% | -4.0% (数値が動いて draft が変わる分) |
| 17k | **-2.5%** | +0.6% | ±0 |

prefill 幅では勝ち、decode 幅では負け。**行数 ≥ 64 (prefill) だけ fold にして既定 on** にする (実装中)。
KLD は `quant_eval compare --fusions` (prompt 約 600 トークンで fold が発火する) で確認してから。

PLE の補足 (`tools/ple_split.py` の再計測): 冷の pread は **257 ms/チャンク (89%)**、`prefill_anatomy` の 167 ms は
ページキャッシュが温まった値だった。本番 (新しい本文) では冷の 257 ms が掛かる = 1 チャンク 3.4〜3.8 s の 7%。
先読みを本当に重ねれば prefill -7% の見込み (実装中)。

## MoE 畳み込み on の KLD (2026-09-03 16:10、`quant_eval compare --fusions`、bf16 参照、48 層で発火)

kld_mean **0.01289** / agree 0.967 (現行 0.01326 / 0.966)。差 -0.0004 で受け入れ幅内 (むしろ僅かに良い。bf16 の
丸め位置が変わる分の揺れ)。**品質は不変。prefill 幅 (行数 ≥ 64) だけ既定 on にする** (実装中)。

## n-gram 先読みの作り直し、8k (2026-09-03 16:40、`bench/results/ngram-prefetch2-8k.json`)

prefill_s A (先読み on) 13.79 対 B 13.84 (**-0.4%**)。作り直し (次の eval 境界の直前に 1 境界ぶんを先読み) でも
取り分が出ない。理由の読み: A/B は同じ本文を 4 回読むので OS のページキャッシュが温まっており、温の 137 ms/チャンク
は**ディスク待ちではなく syscall と Python ループの費用** (32k 行 × `os.pread` 1 回 4 us、12 スレッドでも GIL で
直列)。重ねても消えない。手は `os.preadv` で 1 syscall に数百行を束ねること (実装中)。先読み knob は既定 off のまま。

## n-gram 先読み 17k (2026-09-03 17:00): prefill ±0.0%。先読みは畳む (既定 off のまま)。束ね読み (preadv) へ。

## MoE 畳み込みを既定 on (行数 ≥ 64、2026-09-03 17:05、コミット済み)

## 小さいベンチ、小物をかき集めた既定 (2026-09-03 17:30、10 分冷却、反復なし、`bench/results/self-snapshot-turbo-small-0903d.json`)

既定に入っている変更: QSA bool マスク、GDN Metal、depth 適応 (2048 超)、T=1 gather カーネル (kv ≥ 12k)、detokenizer の
使い回し、`_verify` の `.item()` 撤去、末尾 lm_head 1 行、prime 窓 512、MoE 重み付き和の畳み込み (行数 ≥ 64)。
(preadv は起動時刻の関係で含まれていない可能性あり。効果は 0.4% 級。)

| 文脈 | 冷 TTFT | 温 TTFT | decode | gather on の前回 (09:10) | mlx-serve (12:00) | 対相手 |
|---|---|---|---|---|---|---|
| 0 | 0.17 | 0.14 | 50.2 | 0.17 / 0.13 / 50.3 | 0.18 / 0.72 / 54.7 | |
| 4k | **6.60** | 0.16 | 51.9 | 6.88 / 0.15 / 51.5 | 5.70 / 0.85 / 55.7 | prefill 1.16x 負け、decode -7% |
| 17k | 31.3 | 0.20 | 46.1 | 31.0 / 0.20 / 47.5 | 27.8 / 0.91 / 49.0 | 1.13x、-6% |
| 25k | 45.5 | 0.23 | 44.2 | 46.6 / 0.23 / 48.0 | 41.3 / 0.89 / 60.8 | 1.10x、(相手の tok/step 運) |
| 32k | 56.4 | 0.25 | 42.7 | 58.3 / 0.25 / 44.9 | 51.6 / 0.90 / 47.4 | 1.09x、-10% |
| 50k | **89.9** | 0.90 | 42.7 | 93.1 / 0.93 / 46.9 | 82.1 / 15.9 / 45.9 | 1.09x、-7% |

小物の合算で冷 prefill が 4k -4%、50k -3.4%。対相手の prefill は 1.09〜1.16x 負けまで詰まった。decode は反復 1 回の
tok/step の揺れ (17k の温ターンが 1.40) が ±5% 動かすので、-6〜-10% は前回 (-8%) と同じ水準。温 TTFT は 4〜17 倍速のまま。

分析 (残差の在り処、1 チャンク 2048 tok = 3.4〜3.8 s の内訳から):
- MoE 43% (gather_qmm、相手と同じ op、効率 60〜65%)。G=8 (M/E 320) でも -1.2% しか出なかったので、タイル水増しの
  モデルは費用を説明しておらず、行数を太らせても縮まない。**手なし** (自前カーネルは 2 件負け)。
- GDN 27% (効率 80%、相手も oMLX 移植)。超過 20% = 190 ms/チャンク は前処理の素 op。手はあるが小さい (-2%)。
- HC 11% (380 ms/チャンク、素の op、効率 60%)。prefill 幅の融合は -0.9% で却下済み。**相手は hc_read を融合している**
  のに、うちの融合が効かないのは低ランク GEMM が既に天井のため。手なし。
- attention 10〜16% (射影 19 ms/層 = 5 TFLOPS)。qkv+gate を 1 本の行列積にする案は prefill 幅で未測 (-2% 見込み)。
- PLE 7% (冷 257 / 温 137 ms/チャンク)。**syscall 律速** (32k 行 × pread)。preadv で 1.11 倍。本命は mmap +
  madvise(WILLNEED) で次チャンクの行をカーネル側で非同期に読み、numpy の fancy index で集める (-4〜7% 見込み)。
- チャンク外: prime (512 で 150 ms 減)、checkpoint、n-gram 文脈。小さい。
決め: フルベンチをもう一度回す前に、PLE の mmap 化と qkv wide (prefill 幅) の 2 つを入れて小さいベンチで確かめる。
decode 側は HC 4.7 ms / indexer 2.7 ms に手が無く、-7% は当面残る。

## MoE の gather_qmm を本番と同じソート済み経路で測る (モデル無し、2026-09-03 18:10、scratchpad/moe_gather_micro.py、E=512 top-10 2560→640、行 20480 = 2048 tok × 10)

| 経路 | ms | TFLOPS |
|---|---|---|
| gather_qmm、専門家順ソート済み (**本番**) | **9.01** | 7.45 |
| gather_qmm、未ソート | 45.9 | 1.46 |
| dense 4-bit qmm、同 FLOP | **5.98** | 11.2 |
| dense bf16 matmul、同 FLOP | 5.44 | 12.3 |
| gather_mm bf16 (専門家 bf16 常駐) | 14.7 | 4.6 |

- in-model の gate/up/down (各 440 ms / 48 層 = 9.2 ms) はこの micro と一致。**MoE の行列積は dense の 1.5 倍**の時間で、
  差はそのまま「専門家ごとの区間タイル (M/E=40、bm=32、水増し 1.41)」。
- **G=8 が効かなかった理由は、層主導のグループ処理がチャンクごとに MoE を呼んでいて、行数を太らせていなかった**
  疑い (確認中)。G 個のチャンクの hidden を連結して MoE を 1 回で呼べば、M/E=160 (G=4) で水増し 1.22 → MoE -13%、
  G=8 で 1.10 → -22%。prefill 全体で **-6〜-9%**。これは Python だけの変更で、数値は不変 (行の順序が変わるだけ)。
- bf16 の専門家常駐 (gather_mm) は遅く、量子化の形を変えても dense qmm の 11.2 TFLOPS が上限。

## gather_qmm のトークン数依存 (モデル無し、2026-09-03 18:20、scratchpad/moe_gather_micro2.py)

| MoE 1 回の行数 | M/E | gather_qmm (ソート済み) | 2048 tok 換算 | dense qmm | 比 |
|---|---|---|---|---|---|
| 2048 tok | 40 | 9.00 ms (7.5 TFLOPS) | 9.00 | 5.97 | 1.51 |
| 8192 tok (G=4 の連結) | 160 | 27.5 ms (9.8 TFLOPS) | **6.88** | 23.1 | 1.19 |
| 16384 tok (G=8) | 320 | 52.4 ms (10.3 TFLOPS) | 6.54 | 46.1 | 1.14 |

- `_group_prefill_forward` は既に G=4 チャンクの行を連結して MoE を 1 回で呼んでいる (`ycat = layer.mlp(xcat)`) ので、
  本番の MoE 行列積は 2048 tok 換算 6.9 ms/本 (dense 比 1.19)。`prefill_anatomy` の 9.2 ms/本 (効率 65%) は
  チャンク単体の呼び出しを測っていて、本番より悪く見せていた。G=8 の -1.2% (6.88 → 6.54 = MoE -5% = prefill -2% の
  見込みに対して) と整合。
- 残る MoE の伸びしろは dense 比 1.19 の 16% (= prefill の 6%) で、gather_qmm の区間タイル効率そのもの。取るには
  grouped GEMM のカーネル。**優先度は下げる** (Python 側の手は使い切った)。

## attention qkv の連結、prefill 幅だけ (`--knob wide-attn`)、8k (2026-09-03 18:40)

prefill_s A +0.3%、decode ±0、出力一致。**取り分なし**。q/k/v/gate の 4 本を 1 本にしても M=2048 の qmm は
既に同じ効率で走っている (5 TFLOPS 相当の「射影 19 ms/層」は qmm 以外 (rope、qk-norm、gate の split、
reshape) の小物の合算)。畳む (既定 off のまま)。17k は確認だけ。

## 4k の prefill trace (2026-09-03 19:30、`MLXTURBO_PREFILL_TRACE=1`、`bench/results/logs/trace-4k.log`)

| 区間 | トークン | 時間 | 1 トークンあたり |
|---|---|---|---|
| group build + eval (g=1) | 1825 | 2.95 + 0.16 = 3.11 s | 1.70 ms |
| tail forward (chunk 主導の最終チャンク) | 2048 | 3.54 s | **1.73 ms** |
| prime (窓 512) | — | 0.05 s | |

advisor の「最終チャンクは 1 トークンあたり 2.1 倍」は 4k では成り立たない (1.73 対 1.70 ms/tok、差 2%)。
17k の 2.1 倍は末尾チャンクの attention (kv 17k) が重い分だった。4k を g=2 のグループに畳んでも MoE の行数が
1825 → 3873 (M/E 36 → 75) になる分 (MoE -8% = 4k の -3%) しか無く、**-10% の見込みは外れ**。実装の優先度は下げる
(prefill 全体の 1.13x と同じ、小物の合算)。

## GDN 1 層の部品計測、S=2048 (2026-09-03 19:30、`tools/gdn_split.py`、kv 4k / 16k、層 1 / 18 で同じ)

| 部品 | ms | 割合 | 効率 (FLOP 下限比) |
|---|---|---|---|
| `_project_in` (4-bit qmm × 4) | 15.0〜16.8 | **57%** | 92〜103% (天井) |
| conv1d + silu | 1.0 | 4% | 20% (dispatch 律速) |
| split / gate / beta | 0.5 | 2% | 17% |
| 再帰スキャン (gdn_metal) | 3.2〜3.3 | 12% | (逐次カーネル、FLOP 比は意味なし) |
| norm(out, z) + out_proj | 7.2〜7.3 | 26% | 78〜80% |
| 部品和 / 壁時計 | 27〜29 / 26〜28 | | |

**GDN の 83% (in_proj + out_proj) は既に天井。**前処理 (conv/silu/gate) は dispatch 律速だが 1.5 ms/層 = 1 チャンクの
1.5% で、Metal 側に取り込んでも取り分はその程度。「超過 20% = 190 ms」は前処理ではなく out_proj の 80% と
スキャンの分。**GDN は閉じる** (前処理の取り込みは機会があればの小物)。

### 2026-09-03 07:30 PLE mmap + 背景 madvise の in-model (8k、別プロセス A/B/A/B)

| run | mmap (A) prefill_s | pread (B) prefill_s | 差 |
|---|---|---|---|
| 1 | 12.626 / 12.723 | 13.472 / 13.560 | -6.3% |
| 2 | 12.918 / 13.022 | 13.721 / 13.777 | -5.8% |

各プロセス内の null knob の A/B は ±0.4〜0.8% (揺れの幅)。mmap 側が毎回先に走っているので熱は mmap に有利だが、
60 s の休止を挟んだ 2 巡で差が揃っている。17k の確認 (順序を逆に) を chain82 で取ってから既定を切り替える。
粒度の掃引 (agent): 16 KB が最良。行 id がハッシュで散っているので 64 KB 以上は発行が 10〜20 倍遅く、冷 fetch は改善しない。

案出し Studio の凍結ポートフォリオは `docs/research/IDEAS-2026-09-03.md`。

### 2026-09-03 07:30 段階投入の切り分け (D0、advisor の P0)

`--knob stage-every --variants 0,2 --only short --tokens 512`: 完全直列 (0) は ms/round 43.5 (+16.2%)、既定 2 は 37.5。
判定基準 (測る前に宣言): 悪化 15% 未満なら構築は 5 ms 級で CPU 仮説は死ぬ、35% 以上なら 20 ms 級。
結果は境界のすぐ上で、構築 ≈ 6 ms/round。段階投入がそれを隠しているので、decode は GPU 律速として読む。
層単位の mx.compile レーンは先頭には入れない (取れても構築 6 ms の一部)。融合カーネルの敗北記録の読み替えも不要。

### 2026-09-03 07:36 HC の elementwise を prefill 幅で mx.compile (P2、`--knob hc-prefill-compile`、8k)

prefill_s A 14.013 / B 14.026 (-0.1%)、tok/round 同一。発火は確認済み (env と enable を knob が両方立てる、行数ゲート 64 に対し 2048 行)。
HC の elementwise は compile しても動かない。GEMM 床 234 ms + traffic 床 105 ms = 340 ms に対し実測 380〜423 ms なので、
残りは 1 割強しか無く、その 1 割も op 起動でなく traffic だった。畳む。knob は既定 off のまま残す。

### 2026-09-03 08:10 P3 第 1 段: steel 平坦化の dense qmm クローン (`tools/moe_grouped_gemm_micro.py --stage dense`)

| 形 | M | K | N | stock ms | clone ms | 比 | ビット一致 |
|---|---|---|---|---|---|---|---|
| gate/up | 20480 | 2560 | 640 | 5.998 | 5.997 | 1.000 | yes |
| down | 20480 | 640 | 2560 | 5.936 | 5.972 | 1.006 | yes |

`mx.fast.metal_kernel` の経路で steel と同じ効率が出た (入場料は払えた)。写しは機械抽出 1507 行 (`kernels/_steel_flat.py`、NAX 記号ゼロを assert)。
mmap 17k の 1 巡目: pread 31.8 s → mmap 29.6 s (-6.9%)。

### 2026-09-03 08:20 PLE mmap の 17k 確認 (別プロセス、順序を逆に pread → mmap → pread → mmap)

| run | pread prefill_s | mmap prefill_s | 差 |
|---|---|---|---|
| 1 | 31.90 / 31.77 | 29.59 / 29.66 | -6.9% |
| 2 | 31.10 / 31.19 | 29.01 / 29.04 | -6.8% |

8k の -6% と合わせて 4 対とも揃った。**既定を mmap + 背景 madvise に切り替えた** (`FASTMLX_NGRAM_BACKEND=mmap`、
`MLXTURBO_NGRAM_PREFETCH` は mmap で on / pread で off)。decode 幅 (16 行取得) の確認は chain85 (短 3 本 × 512、A B B A)。

### 2026-09-03 08:40 P3 第 2 段: 専門家セグメント対応 (`tools/moe_grouped_gemm_micro.py --stage segmented`)

| 形 | ケース | stock/dense | seg/dense | seg/stock | ビット一致 |
|---|---|---|---|---|---|
| gate/up | skew r=40 | 1.458 | 1.398 | 0.959 | yes |
| gate/up | skew r=160 | 1.167 | 1.098 | 0.941 | yes |
| down | skew r=40 | 1.414 | 1.415 | 1.000 | yes |
| down | skew r=160 | 1.145 | 1.101 | 0.961 | yes |

判定線 (r=160 で 1.14 未満) は通過。時間はタイル枚数 Σ ceil(rows_e/32)·32 / M × 1.015 で 1% 以内に説明でき、
端数タイルの費用 c ≈ 1.0 (frag の間引きは分岐で 1 割損)。3 本加重で r=160 -5.2% / r=40 -2.7%、in-model 換算 17k -1.9% / 4k -1.0%。
第 3 段 (フック + in-model) で 1:1 の換算とテーブル構築費 (48 層 × 小 op 5 本) を先に確定し、届かなければ BM=16 の経路
(逆算で BM=32 の 0.53 倍、専門家ごとに良い方を選ぶと r=40 1.214 / r=160 1.067 の見込み) を足す。

### 2026-09-03 08:27 D4 command buffer の粒度 (env、別プロセス A B B A、`tools/verify_width_cost.py` 短プロンプト)

A = `MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=1000000`、B = 既定。

| | S=1 forward 中央値 | S=2 forward 中央値 |
|---|---|---|
| A1 / A2 | 22.54 / 23.12 ms | 28.52 / 28.56 ms |
| B1 / B2 | 23.15 / 23.01 ms | 28.23 / 28.08 ms |

S=1 で -1.1%、S=2 で +1.4%。揺れの幅の中で向きも揃わない。判定線 (差 1% 未満なら畳む) どおり **畳む**。
stage 境界の async_eval が commit を強制するので、env で締め切りを緩めても床は変わらない (Challenger の読みどおり)。

### 2026-09-03 08:28 P3 の対比較: 既製 gather_qmm に 16 行揃えのダミー行 (`tools/gather_qmm_pad_micro.py`、Metal 無し)

| 行数 (行/専門家) | そのまま | 16 行揃え | 32 行揃え | dense |
|---|---|---|---|---|
| 81920 (160) | 81.99 ms (1.165) | 79.06 ms (1.123、行 +4.3%) | 82.87 ms (1.178) | 70.38 ms |
| 20480 (40) | 25.51 ms (1.441) | 22.61 ms (1.278、行 +18%) | 27.02 ms (1.527) | 17.70 ms |

有効行あたりの dense 比は 16 行揃えで 1.077 / 1.082 (判定線 1.05〜1.10 の間 = straddle は主因の 6〜8 割で、残りはタイル形)。
**Python だけで r=40 -11.4%、r=160 -3.6%** が取れる。segmented (BM=32) は r=160 -5.9% (gate/up) / -3.9% (down)、
r=40 -4% / 0% なので、小さい専門家は BM=16 (既製 + padding)、大きい専門家は BM=32 (segmented) が良い。
第 3 段の A/B に variant C (pad16 + 既製) を足して 3 者で取る。

### 2026-09-03 08:28 P1 の proof-of-life: 2 本の GPU stream (`tools/two_stream_micro.py --n 8`、モデル無し)

A 単独 (GDN scan 連鎖) 23.3 ms、B 単独 (qmm 連鎖) 74.7 ms、1 stream 交互 97.2 ms、2 stream 93.8 ms。
2 stream / 1 stream = 0.965 (判定線 0.95 以上で直列化)。重なったのは 3.4 ms で、隠せる最大 23 ms の 15%。
Apple GPU + MLX 0.32.2 では compute の command buffer が queue 間でほぼ直列に走る。**P1 は畳む** (in-model に進まない)。

### 2026-09-03 08:28 D5 の proof-of-life: decode 幅の gather_qmm のスケール (`tools/gather_qmm_scale_micro.py`)

1 セット (gate/up/down) あたり: 22 行 / 20 専門家 196 us、22 行 / 11 専門家 139 us、11 / 11 98 us、1 / 1 75 us。
時間は行数でなく**異なる専門家の数**で決まる (22/20 と 22/11 で 34% 差)。つまり重複行の重み読みは既に共有されており、
verify 行間で同じ専門家をまとめ直しても減らない (Challenger の読み: union 21〜22.5/30 の重複は既に安い)。**D5 は畳む**。

### 2026-09-03 08:44 D2 の trace (静的 depth 4、depth 適応 off、短 + 長 × 512、`tools/depth_trace_stats.py`)

- 信号 a (直前 3 ラウンド全採用 → 次も深い): P(hit≥3 | 111) = 0.812 (n=64、判定線 0.5)。ただし 111 は 4800 ラウンド中 64 (1.3%)。
  履歴表で depth を選ぶ in-sample シミュレーション: ms/tok 18.99 → 18.54 (-2.4%)。
- 信号 b (位置 1 の draft マージン): AUC 0.735。マージン下位 10 / 20 / 30% の棄却率 0.85 / 0.83 / 0.84 (判定線 0.70)。
  マージン閾値のカスケード (T(S)=24+7S): 最良 tau=2.875 で ms/tok 17.96 (-5.4%、in-sample)。
- **落とし穴**: 信号 b は今の構造では draft の 1 段目のマージンを host に同期しないと使えず、draft の GPU 実行と次ラウンドの
  構築の重なり (約 1.5〜3 ms/round) を壊す。同期込みの見込みは -1.5% 前後。**D1 (draft の 1 段目を verify のグラフに同梱)**
  が入れば verify の同期でマージンが只で手に入るので、D2b は D1 の上に載せる。順は D1 → D2b。信号 a は同期不要だが取り分 2%。

### 2026-09-03 09:45 D3 (文脈 n-gram の draft) のオフライン集計 (`tools/sam_offline_stats.py`、生成列は `--save-out` の 17k / 50k 各 3 本 × 512)

発火率 (一致長 ≥ 3): 17k 21.5%、50k 25.3% (プロンプト間 19〜32%)。一致長 5+ のバケットでも k=3 までの累積受理は 44%。
費用表 T(S)=24+7S で最良は k=5 (幅 6): 17k -2.1%、50k -2.6%。非線形 (T(6)=72 ms) なら -0.6% / -1.0%。
判定線 -3% に届かず **畳む**。verify 行が安くなる (K2) と損益が変わるので、K2 の後に再計算する余地だけ残す。

### 2026-09-03 09:15 P3 第 3 段: in-model (`--knob moe-grouped-gemm --variants A,B,C`、各 6 本、出力一致)

| ctx | A segmented | B 既製 | C pad16 |
|---|---|---|---|
| 4k | +0.3% | 0 | +1.6% |
| 8k | -1.4% | 0 | +2.7% |
| 17k | -0.6% | 0 | +2.8% |

micro の -5% (r=160) / -4% (r=40) が in-model では 8k -1.4% / 4k ±0 にしか乗らない。pad16 は行数増の分だけ遅い。
**P3 は速度レバーとしては 1% 台で、既定には入れない** (コードは残す。換算率の説明と `--knobs` はエージェントの報告待ち)。

### 2026-09-03 10:15 D1 (draft の 1 段目を verify のグラフに同梱、`--knob draft-presync`)

| | ms/round A | B | tok/round A | B |
|---|---|---|---|---|
| 短 3 本 × 512 | 40.82 (+7.8%、温まった位置で +1.7%) | 37.88 | 2.163 | 2.144 |
| 17k 3 本 × 512 | 45.05 (+5.0%、温まった位置で +1.8%) | 42.91 | 2.051 | 2.096 |

出力一致。round trace: built → eval_done が +2.0 ms、eval_done → drafts_submitted が -0.9 ms。ホストの隙間は半減したが、
素の経路では MTP の draft は「次ラウンドのグラフ構築 (rollback_done → built の約 30 ms) と重なって隠れていた」ので、同期の壁に持ち込むと
隠れていた分がそのまま壁時計に出る。台帳の前提「eval_done → drafts_submitted は GPU に何も無い」が不正確だった。**畳む**。
probe: S 行ブロックと S=1 逐次はビット一致しない (QSA 不発の kv=513 でも 15/15 ずれる。sdpa の経路と bf16 の積み順)。draft トークンは 13〜15/15 一致。
D2b の口: D1 無しでは step 0 のマージンを depth 決定の前に得られない。選択肢は (b) 前ラウンドのマージン (1 ラウンド遅れ、同期ゼロ) か (c) `mx.eval(margin)` を 1 回 (0.3 ms/round)。

D2b の補足 (10:20): 1 ラウンド遅れのマージン (前ラウンドの位置 1 のマージンで次ラウンドの 2 本目の受理を予測) は AUC 0.567 (同ラウンドは 0.735) で
信号にならない。残る (c) 案 (draft の 1 段目の後で `mx.eval(margin)` を 1 回) は、D1 の結果から「隠れていた GPU 仕事を壁に出す」費用が
1.5〜2 ms/round 級と見込まれ、取り分 -1.7 ms/round と相殺する。**D2b は K2 (行が安くなる) の後に損益を取り直すまで畳む**。

### 2026-09-03 10:00 P5 行タイル分割の dense sdpa (`--knob sdpa-rowtile`、R=256、各 6 本)

| ctx | A 行タイル | B 素の fallback | 差 | head 一致 |
|---|---|---|---|---|
| 4k | 6.286 s | 6.356 s | -1.1% | yes |
| 8k | 13.197 s | 13.358 s | -1.2% | yes |

見込み (-1.5%) どおりの小物。上三角の 4.6 ms/層/チャンクを回収した分。発火回数 (trace 付き再走) と 17k、KLD (ビット一致しない設計) を見て既定にするか決める。

### 2026-09-03 10:40 P3 第 3 段の帰属 (Opus の報告)

- 換算率の損失はほぼ無い: 8k は micro からの予測 -171 ms に対し実測 -175 ms。MoE の行列積は prefill の 28〜37% しか無い (残りは router / argsort / 並べ替え / SwiGLU / 共有専門家)。
  17k は予測の 47% (標準誤差 ±192 ms で切り分け不能)。4k は r=35.6 で down の取り分が 0。
- 発火 291 / 435 の内訳: `_PREFILL_TAIL_CHUNKS=1` で末尾 2048 (r=40) が常に chunk 主導に残る + MTP の priming 層 (5120 行) が 1 回。**末尾の r=40 は文脈長によらず必ず来る**
  (MoE 行列積に占める割合 4k 100% / 8k 30% / 17k 15% / 50k 5%)。
- C (pad16) が遅い理由は設計そのもの: 層あたり大きい gather 5 本 (215 GB/s) と `pstart[E].item()` の host 同期 1.17 ms/層。4k / 8k から解いた 2 変数で 17k を +1.7% 以内に予測できる。
  `ngram_sync_ms` が C だけ 6 割減る (同期が GPU を空にする直接の証拠)。改良しても届かない。
- 次: `tail-in-group` (末尾をグループに) が通れば r=40 は短文脈だけに縮む。その後、BM=16 の segmented を micro (`--stage segmented`) で pad16 の 22.61 ms/層に届くか確かめてから in-model。
- 注意: `control_identical` の自動判定は `kind == "short"` の行しか見ないので `--only long` では無検査 (head の突き合わせは手で確認済み)。温めは `variants[0]` だけ。→ decode_ab の直し (小)。

P5 の micro (`tools/sdpa_rowtile_micro.py`、S=2048、Hq 24 / Hk 2、D 256、ms/層): kv 2048: whole 10.52 → R512 7.27 / R256 6.71 / R128 6.81。
kv 4096: 20.57 → 17.40 / 16.74 / 16.65。kv 8192: 40.71 → 38.73 / 39.97 / 40.78。max|diff| は 3 形とも 0.0 (この形ではビット一致)。
kv が大きいほど取り分が縮む (8k では R512 が最良で -2.0 ms)。R を kv で切り替える (kv < 8k は 256、以上は 512) 余地あり。

P5 の 17k (10:13): prefill_s A 28.95 / B 29.25 (-1.0%)、tok/round 同一。4k -1.1% / 8k -1.2% / 17k -1.0% で揃った。micro では max|diff| 0。
発火確認 (trace 付き再走) と runner への配線が済んだら既定 on にし、`quant_eval.py compare --fusions` で KLD を 1 回取る。

### 2026-09-03 10:30 P6 T=1 gather カーネルの段階 load を uint4 化 (`prefill_attn_v2.py`、`qsa_gather_micro.py --variants d`)

kv=16896、S=2048、清浄な 3 プロセス × 9 反復の中央値: load 36.9 → 25.5 ms (-31%、DRAM 床 21.5 の 1.19 倍)、フル 56.4 → 40.8 ms/層 (-27.6%)。
kv 6k〜17k で uint4 版は約 41 ms 平坦、6 点全部で現行とビット一致。dense (-2.86 + 5.51·kv/1000 ms/層) との交差点は 10.8k → 8.1k。
判定線 (load < 30 ms) 通過。本番 `prefill_attn.py` への移植は 3 箇所 (threadgroup の uint4 宣言、列主体の load ループ、整列ガード)。
発火下限 `MLXTURBO_PREFILL_ATTN_MIN_KV` 12288 → 8192 に下げる場合は kv 8k〜12k の課題正答率ゲートを取り直す。
汚れたプロセス (98 GB ジョブの直後) は load が逆に遅く出た。部品の引き算は帰属が壊れるのでフルの数字だけ信じる。

P5 の KLD (10:20、`quant_eval.py compare --fusions`、tag sdpa-rowtile-on-0903): kld_mean 0.012891 = 基準 (moe-combine-on-0903) と差 0.0、top-1 一致 0.9667 同一。
発火は 4k で 25 回/リクエスト (12 層 × 2 チャンク相当)。**既定 on で確定** (`MLXTURBO_SDPA_ROWTILE=256`、CLAUDE.md に記載)。
decode 側 mmap 確認 (chain85 B1、pread): 短 3 本 ms/round 37.44 (null A/B の B)。A1 (mmap) は連鎖の組み替えで落ちたので A2 の 1 本で比べる。

### 2026-09-03 10:40 末尾チャンクをグループに (`--knob tail-in-group`、8 トークン、3 本 × 2)

| ctx | A on | B off | 差 | 出力 |
|---|---|---|---|---|
| 4k | 6.140 s | 6.438 s | **-4.6%** | 一致 |
| 8k | 12.680 s | 13.216 s | -4.1% | 一致 |
| 17k | 28.052 s | 28.424 s | -1.3% | 一致 |

チャンク割り (grid) は保ったまま、chunk 主導で流していた末尾チャンクをグループの最終メンバーに足す形 (予約式は残す)。tok/round 不変。
代償: 末尾で BPE 境界の checkpoint (n-1) を積めない。**作り直しの形**: 最終メンバーを「末尾チャンクから最後の 1 トークンを除いた部分」にし、
最後の 1 トークンだけ chunk 主導で流す → MoE は 3999 行 1 回、checkpoint 復活、mixer/lm_head の追加コードも不要 (上位互換)。追加費用は T=1 の forward 1 回。
ハーネスの罠: 対照検査を (kind, ctx) でまとめると、長さが同じ別プロンプト同士を突き合わせて偽の NG が出る (3 本中 2 本が 3873 トークン)。case 単位に直す。

P6 の品質ゲート (11:19、`tools/longctx_quality.py`、17k、n=8、seed 0): recall は dense 8/8、kernel 8/8、agree 1.0 で、発火下限 12288 (48 発火/問) と 8192 (72 発火/問) の両方で同じ。
下限を 8192 に下げても課題正答率は落ちない (quote 課題の値はエージェントの報告で確認)。

### 2026-09-03 11:24 P8 dense 射影の 4-bit qmm 対 bf16 GEMM (`tools/dequant_gemm_micro.py`、乱数重み、ABAB × 5)

| 形 | M | qmm ms (TFLOPS) | dequant 込み (比) | bf16 常駐 (比) |
|---|---|---|---|---|
| gdn_in_proj 2560→16480 | 2048 | 14.94 (11.6) | 14.00 (0.94) | 13.54 (0.91) |
| gdn_in_proj | 8192 | 59.16 (11.7) | 53.83 (0.91) | 53.11 (0.90) |
| gdn_out_proj 6144→2560 | 2048 / 8192 | 5.72 / 22.11 | 0.95 / 0.91 | 0.92 / 0.90 |
| attn_q_proj 2560→12288 | 2048 / 8192 | 11.16 / 44.67 | 0.94 / 0.91 | 0.91 / 0.89 |
| attn_o_proj 6144→2560 | 2048 / 8192 | 5.72 / 22.51 | 0.95 / 0.91 | 0.92 / 0.90 |
| hc_mix_down / up | 2048 / 8192 | 1.3〜4.9 | 0.93〜1.02 | 0.92〜0.99 |

qmm はこの形で 11.5〜11.7 TFLOPS (見込みの 10.3〜11.3 より上)、bf16 matmul は 12.7〜13.0。max|diff| は全形 0.0 (同じ重みなら累算も同じ)。
判定線 (常駐 0.85 / 込み 0.90) には届かず、取り分は GEMM の 5〜10% = prefill 2〜4% (M=8192 側が有利)。
**保留**: P10 (BM=64 の自前 qmm) が 13 TFLOPS 級に届けば不要。届かなければ P9 (チャンク 8192) と合わせて in-model の候補に戻す。

### 2026-09-03 11:41 decode 側の mmap 確認 (chain85、短 3 本 × 512、別プロセス、null knob の B 行で比較)

pread B1 37.44 / B2 37.44 ms/round、mmap A2 37.03 ms/round (mmap A1 は連鎖の組み替えで消えた)。tok/round は 2.144 で同一。
mmap の 16 行取得は decode を遅くしない (むしろ -1%、揺れの幅)。**mmap 既定は decode 側も問題なし**。
P7 の `tools/moe_split.py` は n-gram サイドカー無しでモデルを読んで shard 重みの欠落で落ちた (--ngram の経路が無い)。直して再投入。

### 2026-09-03 11:50 K2a: 厳密 top-k select カーネル (`mlxturbo/kernels/qsa_select.py`、`tools/verify_qsa_select.py`)

S ∈ {1,2,3,4,6} × kv ∈ {4k, 8.5k, 17k, 25k, 50k} × 意地悪 9 種 (全同点、閾値 0、-0.0、結合順、可視 < k) = 875 通り 2800 行で **argpartition の集合と 100% 一致**。
MLX の `sum(axis=-1)` は 0.0 起点の逐次和 (木状ではない)、argpartition の同点は添字昇順、をどちらも実測で確認。
時間 (us/呼び出し、まとめ eval の GPU 実働 / 1 本ずつの露出レイテンシ): 17k S=2 13.4 / 31.5 (argpartition 73.2 / 127.6)、S=6 14.0 / 30.8、50k S=2 14.7 / 51.6 (常駐上限 6400 ブロック超でキー再読み)。
判定線 (17k S=2 ≤ 30 us) 通過。**K2b (2-pass vector の写し + 常駐ビットマップ + per-query tail) に進む。**
配線の口: `_pooled_and_top` を einsum (466 行) と `mx.maximum` (467 行) の間で `_block_scores` / `_select` に割り、`select_kernel` を別メソッドで足す。`n_vis` は `visible_counts_host` (Python だけ) で作る。

### 2026-09-03 12:00 custom kernel が decode の in-model で負ける理由 (確定、`tools/custom_kernel_overhead_micro.py`、`--knob hc-kernel-stage`)

- CPU 側 (H1) は棄却: 構築費は融合 11.1 us 対 素 16.4 us で**融合の方が安い** (source 8 KB のコピーやハッシュは 1〜2 us)。
- 段階投入との相性 (H3) は棄却: 2×2 で負け幅は段階投入あり +8.17 ms、完全直列 +7.16 ms (直列で縮む = 構築は負けの原因ではない)。
- command buffer 分断 (H2)、非連続入力のコピー (H4、差 1〜2%、本番の入力は連続)、出力割り当て (H5) も余地なし。
- **正体: 連鎖 micro が重み 1 組 (3〜7 MB) を 200 回読む温キャッシュだった。** 重みを 48 組 (157 MB) 巡回させると、素の op 列は 68 → 62.5 us で不変、
  融合カーネルは 81 → 140.8 us (+74%)。差 78 us × 97 層 = +7.6 ms/forward で in-model の +8.2 ms を全量説明。
  融合カーネルは threadgroup と lane の並列度が足りず (hc_pre は grid.y=20、hc_post は 32 lane 中 10 本、その後 1 lane の逐次)、DRAM レイテンシを隠せない。
  MLX の qmv は行ごとに threadgroup を立ててベクトル load するので隠せる。
- **一般化**: decode 幅 (S=1) で qmv / qmm を自前カーネルに置き換える筋は構造的に不利 (moe_glu、moe_route、HC 融合、moe_verify の敗北は全部同じ形)。
  自前カーネルの勝ちは全部 prefill 幅。**判定ゲートは「重みを 100 MB 超巡回させた冷の連鎖」に移す** (温の連鎖で速くても採用しない)。
- HC の現実線: 融合で勝つには冷で 2.25 倍の改善が要る (lane 占有と threadgroup 数)。見込みは 4.7 → 4 ms 級 (-0.5〜-1 ms)。「HC で -2.5 ms」の前提は下方修正。

### 2026-09-03 12:12 P3 BM=16 → 混合タイル (mix48 / WM=1) の micro (`scratchpad/moe-grouped-gemm-bm16.json`、1 プロセス 8 往復、80/80 ビット一致)

MoE 層 1 つ (gate/up ×2 + down) の ms。r=40 は末尾チャンク相当、r=160 は 8k のチャンク相当。

| 変種 | r=40 | dense 比 | r=160 | dense 比 |
|---|---|---|---|---|
| stock (既製 gather_qmm) | 27.02 | 1.408 | 89.39 | 1.163 |
| seg32 (BM=32 / WM=2、現行の `MLXTURBO_MOE_GEMM=on`) | 26.30 | 1.370 | 84.66 | 1.101 |
| seg16 (BM=16 / WM=1) | 23.81 | 1.241 | 85.43 | 1.111 |
| seg32w1 (BM=32 / **WM=1**) | 25.61 | 1.334 | 80.89 | 1.052 |
| **mix48** (行数 < 48 の専門家は 16 行、他は 32 行、WM=1、1 dispatch) | **23.69** | **1.235** | **80.12** | **1.042** |
| pad16 (既製 + ダミー行) | 23.85 | 1.243 | 85.90 | 1.117 |
| dense (床) | 19.19 | 1.000 | 76.87 | 1.000 |

- 予定外の発見: **同じ 32 行タイルを 128 スレッド (WM=2、steel の既定) から 64 スレッド (WM=1) にするだけで速い** (r=40 -2.6%、r=160 -4.5%)。
  r=160 の取り分はほぼ WM=1 で、16 行タイルの寄与は 1% 未満。r=40 は 16 行タイルが主。
- 閾値の掃引 (24〜96) は 40〜96 で平ら、最良 48。16 行タイル 1 枚の費用は 32 行の 0.52〜0.53 倍 (直接測った) で、損益分岐 ~48 行と合う。
- 判定線 22.61 は `gather_qmm_pad_micro.py` の別ルーティングの数字。同条件で測った pad16 (23.85) を mix48 が下回る (r=40 -0.7%、r=160 -6.7%)。
  pad16 は行の水増し (+17%) と層ごとの host 同期が要るが、混合は同期なし。
- flat では seg32w1 が `mx.quantized_matmul` の dense すら下回る (0.951〜0.986)。dense クローンの WM は未計測 (伸びしろ)。
- **判定: mix48 / WM=1 で in-model へ。**見込み prefill -2.3% (対 seg32、MoE は 8k の 43%)。P10 の BM=64 と同じプロセスで `--knobs` 2 本。

### 2026-09-03 12:12 P10 BM=64 自前 4-bit qmm の冷 micro (`scratchpad/qmm_wide.log`、36 層の重みを巡回、ABBA×5、`mlxturbo/kernels/qmm_wide.py`)

| 形 | M | 素 qmm ms (TF) | bf16 常駐 (比) | 最良変種 (比) |
|---|---|---|---|---|
| attn_q 2560→12288 | 2048 | 11.24 (11.5) | 10.18 (0.909) | 10.59 (0.942) |
| attn_q | 8192 | 45.37 (11.4) | 39.72 (0.900) | 42.41 (0.935) |
| gdn_wide 2560→16480 | 2048 | 15.06 (11.5) | 13.55 (0.909) | 14.08 (0.935) |
| proj_out 6144→2560 | 2048 | 5.73 (11.2) | 5.22 (0.913) | 5.43 (0.947) |
| proj_out | 8192 | 23.80 (10.8) | 20.47 (0.899) | 22.25 (0.935) |
| moe_gate_up / moe_down | 2048 | 0.79 / 0.76 | 0.941 / 0.947 | 0.971 / 0.967 |

- 最良は全形 **BM=64 / BN=32 / BK=32 / WM=WN=2 / X の読み 8 要素**。256 スレッド化と BM=128 は全形で負け (占有率)。BN=64 / BK=64 は僅かに劣る。
- 全変種・全形で max diff 0 (ビット一致、BK=64 でも)。冷と温の差はほぼ無い (0.942 / 0.941) = この形の qmm は重みの帯域律速ではない。
- t(BM) = a + b/BM を BM=32 / 64 の対で解くと、BM→∞ の漸近比 0.85〜0.89 = bf16 常駐の実測 TFLOPS に一致。
  **逆量子化は qmm の 8〜12% だけで、残りは bf16 と共有の steel BlockMMA (~13 TFLOPS) が床。BM を上げても bf16 に追いつくだけで追い越せない。**
- **判定: P10 (qmm の置き換え) は畳む** (冷の最良 0.935 > 判定線 0.90、bf16 常駐 0.90 にも負ける)。**P8 も閉じる** (bf16 常駐 0.90 から書き戻しを引くと残らない)。
  dense 射影を 4-bit qmm より速くする方向は両腕とも閉じた。
- 残り 1 つ: BM=64 タイルはビット一致のまま dense 射影 -5〜6.5%。dense 射影はチャンクの約半分なので **-3% 前後の見込み、品質の代金ゼロ**。
  in-model A/B に進める (P3 mix48 と同じプロセス、8k / 17k、判定線 -2%)。配線は `enable_wide_projections` の口ではなく個別の `nn.QuantizedLinear` (q_proj / in_proj_qkv / in_proj_z / out_proj)、行数 ≥ 1024、NAX では off。

### 2026-09-03 12:13 冷の連鎖ツール (`tools/kernel_chain_cost.py` / `tools/micro_kernel_latency.py`、commit 9879cb8、`bench/results/kernel-chain-cost-cold.json`)

HC 97 組 (682 MB)、GDN 36 組を巡回。us/call: hc_gated_residual fused 48.5 / plain 205.5 (0.236)、gdn_recurrent with_states 37.6 / mlx_lm 17.3 (2.17、機能が違う)、
gdn_prework fused 39.2 / plain 33.3 (1.18)、rms_norm_gated 5.7 / 11.3 (0.50)。床 3.2〜4.2 us。
- **注意: この HC 項目の plain (205 us) は、12:00 の帰属で使った `custom_kernel_overhead_micro.py` の本番 plain (冷 62.5 us) と合わない。**
  原因は 12:35 に判明 (下の HC v4 の節): 連鎖ツールの `_quant_linear` が 8bit 既定で、実モデルの HC (4bit/gs64、inject は bf16) と違う。
  8bit で測ると符号が反転して「融合が素より 4.3 倍速い」と出る。4bit に直すと 135.5 / 61.9 で CATCHUP の数字を再現する。ゲートは修正中。
- gdn_prework / rms_norm_gated の重みは 3 MB 以下で冷を再現できない (警告どおり)。gdn_prework は既定 off の knob なので判定は据え置き。
- 待ち手が走っている最中に `tools/biglock.sh` を同じ inode に上書きしたら、待ち手のループ後の続きが化けて `command not found: nue` が出た (コマンド自体は走り終わっていたので実害なし)。
  直すときは別名に書いてから mv で入れ替える。

### 2026-09-03 12:35 HC v4 (elementwise だけ融合、MLX の qmv は素のまま): 速度は棄却、ビット一致と受理率の手がかりが残った (`bench/results/hc-elem.json`)

- 実装: `mlxturbo/kernels/hyper_connection.py` に `fused_gated_residual_elem` (+327 行、既存 3 変種は無傷)、`mlxturbo/fused.py` に `enable_hyper_connection_elem`、`tools/decode_ab.py` に `--knob hc-elem`。
  dispatch は素の 14 → 6 (自前 3 + MLX qmv 3)。
- **ビット一致**: MLX 本体の bf16 sigmoid は `y = 1/(1+exp(|x|)); x<0 ? y : 1-y` (`unary_ops.h:308`)。`hyper_connection.py` 冒頭の総当たりは `exp(-|x|)` 側しか試しておらず、
  それが 1 ulp 差の原因だった。写すと bf16 全 65536 パターンで 65535 一致。elem の mixed / inject は S=1 (4bit / 8bit) で素とビット一致、S=6 4bit で 15360 中 1 個 1 ulp。
  対照の現行 `kernel` 変種は mixed 97.5% 一致、max_abs 7.8e-3。
- 冷連鎖 (97 組、実モデルの 4bit/gs64 + bf16 inject、367.5 MB): 素 61.85 / kernel 135.46 (+119%) / **elem 52.75 (-9.1 us、-14.7%)** / 床 qmv 3 本 36.94、自前 3 本 14.49。
  elem ≈ qmv3 + elem3 で加法的。**判定線 (-25 us) 未達**: 素の elementwise 11 op が 25 us しかなく、3 本に畳んでも 9 us しか取れない。
- in-model (17k、短長 3 本 × 512、A,B,B,A): 短 ms/round +2.4% / 長 +0.1%、tok/round 短 +2.3% / 長 +3.7%、ms/tok 短 +0.1% / 長 -3.3%。**速度の判定線 (ms/round -4%) 未達、棄却。**
- 手がかり: elem (素とビット一致) の tok/round が本番既定 (HC=kernel) より短長とも高い。既定の kernel は 106 呼び出し中 9 だけ発火 (`MLXTURBO_HC_INJECT_BF16` 未設定のため残りは素へ)。
  素直に読むと「既定の融合カーネルが受理率を削っている」。切り分けは A = elem、B = HC=off の 1 プロセス A/B (elem と off はビット一致なので tok/round が完全一致するはず)。**投入済み** (`hc-elem-off.json`)。
  判定線: elem = off がビット一致で、kernel 既定の tok/round が両方の長さで 2% 以上低ければ、HC=kernel を既定から降ろす (品質を売って速度を買わない)。
- ゲートのバグ: `tools/micro_kernel_latency.py` / `kernel_chain_cost.py` の `_quant_linear` は QBITS=8 既定。実モデルの HC は 4bit/gs64、inject は bf16。8bit だと
  素 205 / kernel 48 / elem 198 と符号が反転する (committed の kernel_chain_cost も fused 46 / plain 200)。**HC 項目を 4bit/gs64 + bf16 inject に直す (修正中)。**

### 2026-09-03 12:55 末尾チャンクのグループ化 v2 (末尾 1 トークンを外に出し、n-1 の checkpoint を残す): 4k -4.9%、8k -3.4%、サーバー経路はビット一致 (`bench/results/tail-in-group-v2-{4000,8000}.json`)

- 1 プロセス A,B,B,A、3 プロンプト × 2 レップ、8 トークン。4k A 6.001 / B 6.312 s (**-4.9%**、ケース別 -4.5 / -5.2 / -5.1)、8k 12.337 / 12.771 s (**-3.4%**、-3.5 / -3.3 / -3.4)。
  v1 (末尾を丸ごとグループへ) の -4.6% / -4.1% と同水準。判定線 (4k -3%) 通過。
- 実モデル 17k のビット一致ゲート (`tools/verify_prefill_bitident.py`、sep n-gram): `checkpoints=[]` (サーバーの実構成) で group=0 と group=4 + tail-in-group が**ビット一致**、
  末尾に n-1 と n の checkpoint がある。`checkpoints=None` (generate() / ベンチ / 検証プローブ) だけ FAIL = 末尾を 2047+1 に割るので量子化行列積の丸めが動く
  (4k case0 の A/B 出力が 6 トークン目で分岐したのはこれ。計算は正しい、`docs/research/PREFILL-CHUNKING-DETERMINISM.md`)。
- 合成モデル 4 形で on/off の出力一致、`bench/test_server.py` + `test_depth_controller.py` 435 passed、fingerprint exit 0。
- 残る性質: グループ内部の中間 checkpoint が消える (末尾の n-1 / n は残る)。LCP が末尾 2048 の内側かつ n-1 以外に落ちるターンでは復元点が 1 グループ手前まで下がる (グループ prefill が元から持つ性質を最後の 1 境界ぶん広げる)。
- **判定: 17k を測り直してから既定 on にする** (v1 は 17k -1.3%。17k で遅くならなければ `MLXTURBO_PREFILL_TAIL_IN_GROUP=1` を既定に)。checkpoints=None 経路の丸めの違いは、本番 (サーバー) に影響しないので許容する。17k は投入済み (`tail-in-group-v2-17000.json`)。

### 2026-09-03 13:00 QSA tail を HF のクエリごと規則に直す knob (`MLXTURBO_QSA_TAIL=query|global`、既定 global = 不変、`mlxturbo/qsa_tail.py`)。実装と CPU 検証は完了、GPU は待ち行列

- tail 規則の写しは **5 か所** (台帳の 3 か所 + カーネル + `batch_spec.py` の `_ragged_indexer_call`)。5 つ目を直さないと batch と solo の受理数が食い違い `tools/verify_batch_spec.py` が落ちる。
  - vendor `QSAIndexer.__call__`: 行ごとの列 `[cr*floor((q+1)/cr), q]` を `put_along_axis` で立てる (1 行 cr-1 列、全面の比較を張らない)。`_gather_tile_attn` の端数列はタイルの own 列区間に。
  - `batch.py` の `indexer_call`: 同じ規則 + 左パディング境界 (`col >= left_pad`)。`batch_spec.py`: 論理列版、`row_blocks == 0` の行は行全体 causal のまま。
  - `kernels/prefill_attn.py`: fallback 不要。`tail_base = ((q_col+1)/cr)*cr`、`ntail_vis = q_col - tail_base + 1` (0〜3) だけ。tail が空でない行のブロックは候補に入らないので「読む範囲 = 可視集合」の前提が保たれる。P6 の uint4 化と同居。
  - シーム (`_positions` / `_final_mask` / `_make_masks`) の引数・返り値は不変。`MLXTURBO_QSA_TIEBREAK=1` (同点は添字の若い方、既定 off) は別 knob。
- 可視集合の一致 (`tools/verify_qsa_tail.py`、HF 規則の numpy 参照、S∈{1,2,3,2048} × kv∈{2049..17000}): query は全部 0 セル不一致。global は S=1 だけ 0、S=2048 で 3066〜3072 セル / 1533〜1536 行
  (セル数まで予言と一致)。**prefill 幅では 3/4 の行が自分自身を見ていない、が数字で確定。**
- 副産物: 指紋の `budget=8 chunk=4` と `chunk=19` は global では別のトークン列、**query では同じ列**。可視集合がチャンクの割り方に依存しなくなった (HF 意味論そのもの)。K1 の「チャンク 8192」の前提確認はこれで済む。
- 既定 global で不変: fingerprint 13 行完全一致、`bench/test_server.py` 417 pass、`verify_batch_cache` / `verify_batch_spec` 両モード合格、`verify_gather_attn` query でも 1.2e-07。
- 既知の陳腐化 (私の変更と無関係): `verify_prefill_attn.py` のモデルレベルは、合成モデルの最大 S=32 に対し今日入った `MIN_S=64` ゲート (`_vendor/qwen4_exp.py:1025`) でカーネルが発火せず不合格。合成側の S を 64 以上にするかツール側で MIN_S を下げる。
- GPU (`scratchpad/gpu3.sh`、待ち行列): query の配列レベル検証 → `decode_ab --knob qsa-tail-query --only both --ctx 17000 --tokens 512` (prefill-once は使えない: 可視集合が変わるので) → `longctx_quality --ctxs 17000 --n 6` を query / global。
- 既定を query にする前に残ること: 上の 3 本、teacher の bf16 再生成 (現行 teacher は global で作られているので、そのままでは KLD が「悪化」に見える)、TIEBREAK の単独 A/B、
  未配線 2 か所 (`kernels/qsa_prefill_attn.py`、`kernels/qsa_select.py` の K2) を起こすときに query 規則を入れる。
- 運用: この GPU 列は 2 時間以上、同じ段の新しい待ち手 (poll が短い) に追い越され続けた。biglock の同じ段を先着順 (札の mtime) にした。

### 2026-09-03 13:10 K2b: decode 幅の QSA attention カーネル (`mlxturbo/kernels/qsa_attn_decode.py`、`tools/verify_qsa_attn_decode.py`)。ビット一致、置換対象を 2.0〜2.7 倍

- 訂正: この機械の devc は `'s'` (`applegpu_g15s`)。本家 sdpa の blocks は 4k 128 / 8.5k〜25k **256** / 50k 512 (設計メモの `'d'` 表は誤り、17k は 512×33 回ではなく 256×67 回)。
- ビット一致: 本番の並び (per-query tail の `QSAIndexer.__call__` → S≥3 は 2 行ずつ sdpa → concat) と S∈{1,2,3,4,6} × kv∈{2049..50000} × 選択 4 種 = **120 通り `mx.array_equal`**、tail 位相 16 通りも一致、
  `MLX_SDPA_BLOCKS` 32/64 に釘付けしても (両側同じなら) 一致。`fast::exp` / FMA は問題にならず: 本家 metallib は `-fno-fast-math`、`metal_kernel` の JIT 既定 `math_mode="safe"` で同じ。
- 可視条件は排他ではなく参照どおりの和集合 `bit(i/cr) || (tail_base <= i <= q_col)` (keep_block に不可視ブロックが混ざっても参照と一致する側)。
- 冷連鎖 (12 層の K/V 418 MB を巡回、us/層、blocks は本家の表どおり):

| S | kv | 現行 計 (sdpa) | K2a+K2b | K2a | K2b | 比 |
|---|---|---|---|---|---|---|
| 1 | 17k | 203 (99) | 86 | 13 | 74 | 2.35 |
| 2 | 17k | 281 (173) | 139 | 16 | 123 | 2.03 |
| 2 | 50k | 533 (382) | 194 | 24 | 170 | 2.75 |
| 4 | 17k | 461 (334) | 231 | 14 | 217 | 1.99 |
| 6 | 17k | 627 (498) | 326 | 21 | 305 | 1.92 |
| 6 | 50k | 1278 (1108) | 481 | 37 | 444 | 2.66 |

  判定線 (17k S=2 で K2b ≤ 75 us) は未達 (123) だが、置換対象 (選択 + マスク + sdpa) 全体では 17k S=2 281 → 139 (12 層で **-1.7 ms/forward**)、50k S=6 1278 → 481 (**-9.6 ms**)。
  K/V を threadgroup メモリへ 16 列ずつ載せて 12 simdgroup で共有する改良で 8〜10% (演算順不変)。残りの床は帯域ではなく online softmax の直列鎖と占有率。
- `MLX_SDPA_BLOCKS` を 32/64 に釘付けすると K2b は kv にほぼ依存しなくなる (17k S=2 123 → 92 / 90、50k S=6 444 → 244 / 230): 読む列が budget 2048 + 端数で頭打ちなので、blocks を減らすと 1 threadgroup の仕事が kv でなく budget で決まる。
  本家は全キーのマスクを走査するので kv に比例。partials の往復が blocks に比例するのが表どおりが遅い理由。
  分岐: (a) 表どおり = 現行とビット一致 (既定、`mirror_blocks`)、(b) カーネルだけ 64 = 再結合順が違う「close」、(c) プロセス全体を 64 = ビット一致で速いが QSA 以外の decode sdpa も 64 (4k は速く、50k は 13% 遅い)。
- 配線の前提: `MLXTURBO_QSA_TAIL=query` 必須 (`eligible()` は global で False)、B>1 は本家の `query_transposed` の添字が B=1 前提なので必ず退く。
- **判定: K2c (配線 + in-model) へ進める。**両側 query で速度だけを見る (ビット一致なので KLD 0)。判定線 17k ms/round -3%、tok/round と head 一致 → 既定 on (query の既定化とセット)。(b)/(c) は後で。

### 2026-09-03 13:50 常駐 worker (`tools/ab_daemon.py` / `ab_submit.py` / `ab_bundle.py`、`tools/biglock.sh` はクライアントに)。読み直しゼロを確認、**起動直後の 1 行目が +7〜9% 遅い段差**を発見

- 同じ `--knob null --only short --tokens 64` を 2 回: 読み込み 1 回 (140 s)、ジョブ 18.8 / 18.0 s、head と tok/round は 16 桁一致、prefill_s +0.11%、**ms/round -2.03%**。
  -2% は揺れではない: 1 回目のプロセスでは各ケースの最初の計測行 (A,B,B,A の A) が 40.3〜40.5 ms、残りは ~36.9 (+7.4〜8.9%)。2 回目では消える。
  decode_ab の温めでは吸収されず、回文順でも位置 1 の段差は打ち消せない。**新しいプロセスでは A に約 5% (ms/round で約 2%) の不利が付いていた** (prefill_s は無影響)。
  worker には burn_in (読み込み直後に 32 トークンの捨て A/B、~10 s) を既定で入れた。過去の decode 側 A/B (A = 新変種) の ms/round は 2% ほど A に不利に出ている
  (HC elem の短 +2.4%、D1 の +1.7% はこの範囲。判定は動かさないが、再測するときは burn-in 付きで)。
- ジョブ種: decode_ab (in-process、knob は enable/disable で戻す。戻しは env → patch-point 一覧 → `enable_default_fusions` の 3 段)、tool (`run_with_model(argv, bundle)`)、exec (subprocess、worker はモデルを抱えたまま GPU の番を渡す)。
  乗らないもの: oracle-draft (クラス属性を戻せない)、ngram-layout (32 GB の RamNGram)、--round-trace / --draft-trace (import 時)。self_snapshot / mlx-serve は従来どおり (biglock が 64 で自分でロックを取り、worker に降りてもらう)。
- コードの鮮度: `mlxturbo/**`、`_vendor`、`decode_ab.py`、run_with_model を持つ `tools/*.py` の mtime が変わっていたらジョブ前に自分を作り直す (実地で 1 回発火、ジョブは生き残った)。
- 段 2 (micro) のメモリ待ちは 8 GB。worker が居ると旧 biglock の `pgrep` に「ロック無しの python」と見なされて待ち手が固まった → 両ツールは絶対パスで `os.execv` する。
- 段の待ち規則は biglock と同じ (上の段、同じ段は札の mtime 順)。**先着順と STOP を組み合わせると、止めた古い札に新しい待ち手が譲り続けて固まる** (13:40〜13:52 に gpu3 が 12 分止まった)。止めるなら札を消す (kill) こと。

### 2026-09-03 14:00 K2c: decode QSA カーネルの配線 (`mlxturbo/qsa_decode.py`、knob `MLXTURBO_QSA_DECODE_KERNEL=1`、既定 off)

- vendor: `QSAIndexer._pooled_and_top` を `_block_scores` + `_select_keep` に割り (op 列不変、指紋はバイト一致)、`select_bits` (K2a のビットマップ) と `Attention._decode_qsa_forward` を追加。
  `__call__` の **`_gather_attn` より前の第 3 分岐** (kv ≥ 25k は decode 幅でも ratio guard を通って gather に入るため)。シームの引数・返り値は不変、`_IndexerCache` に新状態なし。
- ゲートは全部 host 側でキャッシュを進める前に判定 (B=1、1≤S≤8、素の KVCache、offset+S > budget、`MLXTURBO_QSA_TAIL=query`、TIEBREAK off、`_positions` / `_final_mask` / `QSAIndexer.__call__` が未差し替え、`n_blocks ≤ MAX_BLOCKS`、mirror_blocks あり)。
  batch / batch_spec は `_positions` 等を差し替えるので同一性判定で退く (`--max-batch>1` はプロセス全体で K2c が消える、`_wide_qkv` の警告と同じ形)。spec_flash の verify forward (S=depth+1) と `_staged_forward` は `Attention.__call__` を通るので K2c に当たる。
- 検証: 指紋バイト一致、`verify_qsa_attn_decode.py` の配線節 (実 Attention、kv=6000、S=1..6) で knob on/off ビット一致・全 S 発火、`bench/test_server.py` 417 pass。
- decode_ab knob `qsa-decode-kernel` (control_identical、DECODE_ONLY)。50k の対照 NG は想定内 (B 側が `_gather_tile_attn` を通り dense と非一致)、17k の不一致は本物。
- **両側 `MLXTURBO_QSA_TAIL=query` 必須。worker は投入側の環境変数を読まないので、この A/B は別プロセス (`BIGLOCK_NO_WORKER=1`) で流す** (chain95)。worker への環境変数の受け渡しは要望済み。

### 2026-09-03 14:05 P3 混合タイル + P10 BM=64 dense 射影の in-model 8k (1 プロセス、`--knobs moe-mix48,qmm-wide`、長文脈 3 本 × 回文、`scratchpad/agent-8k-*.json`)

| knob | 変種 | prefill_s 中央値 | 対 素 |
|---|---|---|---|
| moe-mix48 | A mix48 / WM=1 | 12.328 | **-4.3%** |
| | B seg32 (現行の segmented) | 12.855 | -0.2% |
| | C 素の gather_qmm (基準) | 12.887 | 0 |
| qmm-wide | A BM=64 タイル (5 射影、行数 ≥ 1024) | 12.752 | **-2.6%** |
| | B 素 (基準) | 13.091 | 0 |

- 6 本すべてで A < B < C、位置 1 の段差なし。tok/round は 3 変種とも 2.444、head は変種間で完全一致 (対照 OK)。発火 segmented 291、qmm_wide 528。
- 配線: `fused.enable_moe_grouped_gemm(mode, mix_threshold, bm, wm)` (表とカーネルに同じ設定、キャッシュ鍵に (bm, mix))、`fused.enable_qmm_wide` (`nn.QuantizedLinear.__call__` 1 個の差し替え、対象は q_proj / o_proj / in_proj_qkv / in_proj_z / out_proj の属性付きだけ、2 次元 M / 3 次元 B·S ≥ 1024)。
  3 つの呼び手 (batch / batch_spec / spec_flash) は別口の射影を持たない。ビット一致は `scratchpad/verify_p3p10_wiring.py` で 15 通り + 混合 5 通り確認、指紋 4 つ一致。
- **判定: 両方とも既定に入れる** (判定線 -2%、品質の代金ゼロ)。`MLXTURBO_MOE_GEMM_MIX=48` (0 で off)、`MLXTURBO_QMM_WIDE=auto` (非 NAX で on)。17k は chain95 で確認 (悪化していれば戻す)。
- 素の seg32 だけでは -0.2% で意味が無かった。WM=1 (64 スレッド) と 16 行タイルの組が効いている。

### 2026-09-03 14:39 QSA tail (query) の GPU 3 本 (`scratchpad/ab-qsa-tail.json`、`lcq-{query,global}.json`、gpu3.sh、別プロセス、burn-in 無し)

- decode A/B (A = query、B = global、短長 3 本 × 512、A,B,B,A): 短は位置 1 の段差 (各ケースの最初の A が 41.8〜42.1、2 本目は B と同じ 36.8〜37.3) を除けば同一
  (短は kv < budget で疎化しないので当然)。**17k: ms/round -0.7%、tok/round -0.3%、prefill_s +0.5% = 速度は中立。**2 本目の A (35.1〜35.3) は B (36.1〜37.5) より速いが位置の効果と区別が付かない。
- 正答率 17k (n=6): recall は query / global とも dense 6/6、kernel 6/6。quote は dense 5/6 (両モード同じ)、kernel は query 5/6 / global 6/6 (1 問の差、n=6 では決め手にならない)。
  kernel と dense の一致率は recall で query 1.00 / global 0.83 (query の方が dense と揃う = カーネルの per-query 意味論が dense と一致する側)。
- **判定: 速度は中立、正答率は同等 (n=6 の 1 問差は保留)。HF / mlx-serve / oMLX と同じ意味論なので query を既定候補にする。**小ベンチには K2c とセットで env で入れる (K2c の 17k が通れば)。
  コードの既定に入れる前に、n を増やした正答率 (seed を変えて 12 問) と teacher の再生成 → KLD。

### 2026-09-03 14:45 P6 の移植 (uint4 load を本番カーネルへ) と MIN_KV 8192 (`mlxturbo/kernels/prefill_attn.py`、`tools/verify_prefill_attn.py` 修正)

- 移植後は v2 と 0.2〜0.5% 以内で同速、移植前比 -27〜-28% (kv 8k〜17k、3 者 1 プロセス交互)、3 点とも移植前とビット一致。scalar フォールバックのソースは移植前と文字列一致。
  dense との交差点 8.14k。
- in-model: prefill-attn 17k **-3.7%** (旧 -0.9%)、50k **-23.2%** (旧 -21.3%)。min-kv 17k: 8192 対 12288 で -0.3% (誤差、効く層が 9 チャンク中 2 つ)。品質ゲート 17k n=8: 両方 recall 1.0 / quote 0.625 で同一。
- MIN_KV 8192 の採否は 10k の A/B (chain95) で決める (-1% 以上なら 8192)。
- `verify_prefill_attn.py` のモデルレベルは MIN_S=64 と MIN_KV 12288 の 2 つで発火していなかった。合成側を S=64 に広げ、check_model の中だけ MIN_KV=0 にして戻す形に直し、配列・モデルとも合格。

### 2026-09-03 14:55 HC の切り分け (elem / off / kernel、17k、1 プロセス、`bench/results/hc-elem-off-*.json`) と冷連鎖ゲートの修正

- ゲート修正: HC 項目を実モデルの設定 (4bit/gs64、inject は bf16) に (`hc_weight_set` / `hc_fused_call`)。冷 97 組 367.5 MB で **fused 136.5 / plain 61.6 us (2.22 倍)**、CATCHUP の帯 (135〜141 / 62) を再現。
  修正前は 46.5 / 200.0 で符号が逆だった。GDN 系と床は不変。
- **既定の HC=kernel は受理率を売っていない**: hc-off (A = 素、B = 既定) で tok/round 短 -0.2% / 長 +0.1% (0.2% 以内)、ms/round は既定の方が短 -2.7% / 長 -0.2% 速い。
  9% の呼び出しだけ融合している現状は割に合っている。**HC=kernel の既定は据え置き。**
- elem の tok/round が高く出たのは品質ではなく別の軌道: elem は decode / verify 幅 (M ≤ 6) では全段ビット一致だが、**M ≥ 62 で post 段 (mean の bf16 逐次加算の模倣か、写した sigmoid の 1 パターン) が素と食い違う**
  (mixed の不一致率 M=62 3.2e-5、M=2048 2.5e-4、normed も M=2048 で 7e-6)。prefill がこの経路を通るので軌道が分かれる。ビット一致で使うなら行数の上限が要るが、速度で棄却済みなので実装しない。
- **HC v4 は完全に閉じる。**`hc-elem` / `hc-off` knob は残す (docstring に実測を記録)。

### 2026-09-03 15:35 chain95 の判定 (worker 経由、17k、3 本 × 回文): 末尾 v2 → 既定 on、P3 / P10 は 17k でも退行なし

- 末尾 v2 17k: A 26.43 / B 26.58 s (**-0.6%**、A 25.9〜26.8 / B 26.5〜27.0)。4k -4.9% / 8k -3.4% と合わせて **`MLXTURBO_PREFILL_TAIL_IN_GROUP` を既定 on** (`=0` で off)。
- P3 / P10 17k (`p3mix-p10wide-17k.json`): mix48 **-4.0%** (27.40 s、seg32 -1.0%、素 28.54)、qmm-wide **-2.5%** (28.08 / 28.80)。対照 OK (head 一致)、tok/round 2.667 で同一、decode ±0.6%。
  8k (-4.3% / -2.6%) と同水準で、14:05 の既定化を裏付ける。
- 17k の prefill は素で 28.5 s → 全部入りで 26 s 台 (9/3 朝の 31.6 s から -17%、mlx-serve 27.8 s に対して 0.95x)。

### 2026-09-03 15:45 P6 MIN_KV 10k と K2c 17k (chain95、worker / 別プロセス)

- MIN_KV 8192 対 12288、10k (`prefill-attn-min-kv-u4-10000.json`): prefill_s **-0.8%** (15.00 / 15.12)、decode ms/round +1.5% (8 トークンの窓の揺れ)。判定線 -1% に届かない。**既定は 12288 のまま。**
  交差点 8.1k の直上では取り分が小さく、10k で 0.8% なら効く文脈が 8〜12k に限られる。
- K2c 17k (両側 query、512 トークン × 3 本、`qsa-decode-kernel-17k.json`): **ms/round -0.7%** (34.93 / 35.16)、tok/round +0.5%、head 一致、発火 3550〜3760 / 本 (毎 round 12 層)。
  冷連鎖の見込み (-1.7 ms/forward = -4.8%) に対して実測 -0.25 ms。**判定線 -3% 未達、既定にしない。**小ベンチにも入れない。
  見込みとの差の切り分け: K2 の天井スタブ (stub-indexer-topk + stub-qsa-attn、17k、query) を chain96 で取る。天井が 1 ms 未満なら「QSA の decode 費用は選択 + sdpa ではなく
  スコア計算 (K2 が置き換えない部分) が主」で K2 は畳む。天井が 2 ms 超なら配線の固定費 (host 側のゲート、select_bits の起動) を疑う。

### 2026-09-03 16:01 小ベンチ 0903f (既定 4 つ入り: P3 mix48 / P10 BM=64 / P6 uint4 / 末尾 v2、冷却 10 分、`self-snapshot-turbo-small-0903f.json`) — **退行。push を止めた**

| 文脈 | 冷 TTFT 0903d → 0903f | decode 0903d → 0903f |
|---|---|---|
| 0 | 0.17 → 0.18 | 50.2 → 43.8 |
| 4k | 6.60 → **13.96 (2.1 倍遅い)** | 51.9 → 46.0 |
| 17k | 31.3 → **42.0 (1.34x)** | 46.1 → 42.5 |
| 25k | 45.5 → 52.6 (1.16x) | 44.2 → 44.3 |
| 32k | 56.4 → 60.3 (1.07x) | 42.7 → 42.1 |
| 50k | 89.9 → 87.8 (0.98x) | 42.7 → 43.8 |

- decode_ab の A/B (8k / 17k、1 プロセス) では 3 つとも -2.5〜-4% だったのに、サーバー経路の通しでは短い文脈ほど遅い (4k で +7.4 s、50k では逆に速い)。
  温 TTFT (0.14〜0.32) と冷却の probe (13.15 TFLOPS 一定) は正常。サーバーの自己申告でも 4k の prefill は温め 13.0 s / 計測 13.95 s と両方遅く、初回コンパイルでは説明できない。
  小ベンチ開始時に swap 4.0 GB / 5.1 GB 使用 (メモリ圧の痕跡はあるが、50k が速いのと合わない)。
- **切り分け (走行中、`scratchpad/run_bisect97.sh`)**: サーバー経路で 4k を 5 条件 (全部 on / QMM_WIDE=off / MOE_GEMM_MIX=0 / TAIL_IN_GROUP=0 / 全部 off)。
  decode_ab とサーバーで違うのは checkpoint あり (`checkpoints=[]`) のチャンク割りと、capture / 継続バッチの経路。末尾 v2 は checkpoint ありの経路の実測が 17k のビット一致だけで速度は未計測、が第一容疑。

### 2026-09-03 16:30 退行の原因 = n-gram の mmap 既定 (冷えたページキャッシュ)。pread に戻して確認 (`bisect-*.json`、`bisect-pread.json`)

- 切り分け (サーバー経路、4k 冷 TTFT): 全部 on 14.13 / BM=64 射影 off 13.09 / 混合タイル off 13.16 / 末尾 v2 off 9.86 / 3 つ off 10.28。**3 つ全部 off でも 10.3 s** (前回 6.6) で、新しい既定 3 つは犯人ではない。
  前回の小ベンチ (0903d、06:30) は mmap 既定化 (08:20) の前だった。
- pread に戻した確認 (0 / 4k): **4k 冷 5.91 s (0903d の 6.60 より速い)、decode 51.8 tok/s (0903d 51.9)、ctx 0 decode 45.3**。直った。
- 教訓 (CLAUDE.md に追記): mmap の -6〜-7% は同じ機体で続けて測った見かけ。**I/O を含む経路の A/B は、プロセスを分けてもページキャッシュを共有する。**冷やすには別の 100 GB 超のファイルを読んで追い出すか、purge が要る。
  本番 (モデル 98 GB + サイドカー 32 GB > 128 GB) はキャッシュが冷えているのが普通なので、mmap のページフォルトが直列化して 4k prefill 2.1 倍、decode -15%。pread (12 スレッド) は冷えていても並列に読む。
- 末尾 v2 は切り分けの中で 4k -4.3 s 分 (13.16 → 9.86) を持っていて、サーバー経路でも効いている。

### 2026-09-03 16:35 診断 (chain95、worker): P7 の内訳、prefill の天井スタブ、draft 無しの床。oracle は knob が壊れて失敗

- **P7 MoE の内訳** (`moe-split.json`、層 20、8k プロンプト、5 回、ms/層): M=2048: router 0.82 / topk 0.31 / sort 0.34 / gather 0.49 / **gate_up_qmm 14.98** / swiglu 0.40 / **down_qmm 7.29** / combine 1.95
  (weight_mul_cast 0.87 + unsort_sum 1.07) / shared 2.06 / other 0.23。部品和 28.9 対 壁時計 27.0 (差 7%)。M=8192: 行列積 74.1 / それ以外 21.8、差 2.2%。
  **行列積が 82%、行列積以外は 4.7 ms/層 = 48 層で 225 ms/チャンク** (「800 ms/チャンク」は ablate の積算の過大評価だった)。回収できそうなのは combine 1.95 + router 0.82 + sort/topk/gather 1.14 = 3.9 ms/層 = 187 ms/チャンク = 8k の 6%。
  半分取れて -3%。Lily の「ルーティングを GPU に残す +89%」は CPU 同期がある場合の話で、うちには同期が無い (部品は全部 GPU op)。
- **prefill の天井スタブ** (8k、`stubs-prefill-8k.json`): GDN scan を 0 にして **-5.4%** (11.23 / 11.87 s)、MoE を 1 専門家にして **-3.3%**。GDN の blockwise scan は 8k の 5.4% で、Lily のレジスタ常駐 scan の +5.6% と整合。
  半分回収で -2.5%。
- **draft 無しの床** (`ar-depth0.json`、depth 0、depth 適応 off、512 トークン × 3 本 × 2 長さ): 短 **23.5〜24.4 ms/tok (41〜42.5 tok/s)**、17k 26.4〜27.0 ms/tok (37〜38 tok/s)。
  MTP の倍率は短 52/42 = **1.24x**、17k 46/37.5 = 1.23x。S=1 forward の実測が 24 ms で確定 (帰属の推定と一致)。
- decode の天井スタブ (短、`stub-draft` ほか) は JSON が出ていない (ログ確認中)。oracle-draft は knob の stub_chain が engine の `_draft_chain(first=...)` の引数に追従しておらず TypeError。直して再走。

### 2026-09-03 17:15 天井スタブ (decode 側) と GDN scan の冷 micro、正答率 12 問 (query)

- **draft の費用 (短、`stubs-decode-short-*.json`)**: stub-draft で ms/round 37.6 → 34.5 (**-8.2% = 3.1 ms/round**)。tok/round は 1.0 に落ちる (draft が無いので当然)。
  stub-indexer-topk / stub-qsa-attn は短文脈では ±0 (疎化しないので発火しない、想定どおり)。
- **K2 の天井 (17k、query、`stubs-qsa-17k.json`)**: stub-indexer-topk で ms/round **-4.3%**、stub-qsa-attn で **-8.9%** (合わせて約 13% = 4.7 ms/round)。
  K2c の実測 -0.7% は天井の 1/18 で、配線か K2b の in-model 費用に 1.4 ms/round 以上が消えている。**K2 は畳まず、K2c の差の切り分けへ** (K2c のエージェントに戻した)。
- **GDN レジスタ常駐 scan の冷 micro** (`gdn-scan-micro.json`、2048 行 × 36 層の活性 1.5 GB を巡回、us/チャンク): blocked 2900、最良の reg (l4-db32) **2529 (0.87 倍)**。判定線 0.70 に届かない。
  8k の 5.4% × 13% = prefill -0.7% 相当。他の変種 (l2-db128 7.1 倍、l32-db8 2.4 倍) は大負け。報告待ち。
- **正答率 17k、12 問 (seed 1)**: query は recall dense 12/12 / kernel 12/12、quote dense 7/12 / kernel 8/12。global は走行中。

### 2026-09-03 17:25 QSA tail を query の既定に (ユーザー判断: 正答率で十分、global 側の 12 問 (30 分) は打ち切り)

- 速度は中立 (17k ms/round -0.7%、prefill +0.5%、短は同一)。正答率 17k: query は recall 6/6 → 12/12、quote は dense 5/6 (global と同じ) → 12 問で dense 7/12 / kernel 8/12。
- `mlxturbo/qsa_tail.py` の既定を query に。`MLXTURBO_QSA_TAIL=global` で旧規則。teacher (bf16、query) の作り直しは SSD が読めるようになってから (挿し直しても readdir が EINTR で読めない)。
- K2c (decode QSA カーネル) は query が前提。-0.7% でビット一致なので代金ゼロ方針では既定候補だが、天井 13% との差の切り分けと 50k の確認を待つ。

### 2026-09-03 17:51 D1 (draft 同梱) の burn-in 付き再測 (`draft-presync-burnin.json`、短長 3 本 × 512、worker): ms/tok 短 +1.0% / 長 +0.4% → 畳んだまま

起動直後の段差を除いても A (同梱) が遅い (ms/round 短 +2.0% / 長 +1.5%、tok/round +0.9% / +1.0%、出力一致)。代金ゼロではない (遅い) ので既定に入れない。

### 2026-09-03 18:05 GDN レジスタ常駐 scan の PoL (`mlxturbo/kernels/gdn_scan_reg.py`、`MLXTURBO_GDN_SCAN=reg`、既定 blocked)

- **現行の kernel S は既にレジスタ常駐だった**: 状態の断片をレジスタに 1 回載せて T 全体を回し、行内縮約は simd_shuffle、書き戻しは最後だけ。Lily の +5.6% の比較相手 (2K で 1 層 256 MiB を動かす blockwise) は
  チャンク分解を行列積に作り替える版で、うちの `gated_delta_blocked.py` (逐次の 1.68 倍で棄却済み) に当たる。**その取り分は 9/2 に kernel S を既定にした時点で回収済み。**
- 冷の連鎖 micro (36 層 1.5 GB の活性を巡回、72 歩、ABBA×3): blocked 2882〜2900 us/チャンク、**reg lanes=4 db=32 2512〜2522 (0.871〜0.875)**、lanes=8 1.045、lanes=16 / 32 1.50 / 2.35、lanes=2 2.12。
  取れた 13% は staging ではなく割り当て (1 スレッド 16 → 32 個の d、shuffle 3 段 → 2 段)。独立累算器 (0.90〜0.96)、q 先読み (5.2 倍、溢れ) は全部悪化。
- 数値: mlx_lm 逐次基準の相対誤差 RMS は kernel S 0.7〜1.8e-5、reg 0.4〜2.3e-5 (同級)。lanes=8 は kernel S とビット一致 (差はメモリの持ち方だけ)。
- 契約: `_vendor/qwen4_exp.py:1580` の `_gdn_metal` シーム → `gated_delta_update_blocked_metal` の中で切替 (8 行)。rollback 側 (`capture()` は `gated_delta_update_with_states`) は無関係。
- 判定線 0.70 には未達 (0.875 = prefill -0.7% 相当) だが、**代金ゼロ方針で in-model 4k / 8k / 17k を回して、遅くならなければ既定 reg** (エージェントに戻した)。

### 2026-09-03 18:15 K2c の再測 (burn-in 付き、両側 query、17k、`qsa-decode-kernel-17k-v2.json`): **ms/round -4.1%、ms/tok -4.5%、head 一致 → 既定 on**

- -0.7% は 2 つの汚れが A 側に乗っていた: (1) burn-in 無しの位置 1 の段差 (+2.5 ms/round)、(2) **depth 適応の EMA が variant をまたいで持ち越される** (case 0 の A が 270 round 全部 depth 2、B は 71 → 10 に減衰)。
  clean な 10 行で -1.48 ms/round = 冷連鎖の見込み (12 × 142 us) と一致。
- host 側の固定費を削った: `arch_char()` が `mirror_blocks` ごとに `mx.device_info()` (nanobind が dict を作り直す) → モジュールで 1 回。params / blocks / visible_counts の小さい `mx.array` 24 個/forward → 1 エントリの memo。同期 (`mx.eval` / `.item()`) は無し。
- 発火は 12/round (本体の attention 12 層)。**MTP の draft 層 (`FlashMTPModule` の DecoderLayer) には当たっていない**: `enable_default_fusions` (runner 1885) が MTP 読み込み (1954) より前。gather_attn / prefill_attn も同じ穴。冷連鎖で ~0.12 ms/round。
- 次の手 (品質と速度の取引、KLD が要る): blocks を 64 に釘付け (冷連鎖 123 → 92 us/層、-0.37 ms/forward、partials 1/4)。再結合順が変わるので teacher の後。
- **罠 16**: 1 プロセスの A/B で depth 適応の EMA が variant をまたぐ。decode_ab は variant / row の切り替えで DepthController を作り直すこと (未対応、decode_ab の宿題)。

### 2026-09-03 18:10 decode 1 step の Metal trace (`tools/decode_gpu_trace.py`、観測 dylib で dispatch / command buffer / GPU 時間を取る。`bench/results/decode-gpu-trace-*.json`)

| | 壁時計 ms/round | dispatch/round | CB/round | GPU 合計 ms | busy | カーネル平均 | 隙間平均 |
|---|---|---|---|---|---|---|---|
| 短 (S=3) | 36.9 | 5136 | 327〜358 | 34.6 | **93%** | 6.7 us | 0.5 us |
| 17k | 35.7 / 42.8 (depth 差) | 5737 / 6083 | 315〜329 | 33.2 / 40.2 | 93〜94% | 6.6 us | 0.4 us |
| depth 0 (S=1) | 23.6 | **4499** | 223〜237 | 22.0 | 88〜93% | 5.0 us | 0.7 us |

- decode_ab との整合 ±1%。ioreg (0.1 s) の稼働率 92〜94% と独立に一致。**GPU は空いていない。起動の隙間は 0.4〜0.7 us/dispatch (1 CB に 15〜19 dispatch が詰まる) で、8/31 の「round 間の 7.3 ms の泡」は無い。**
- **dispatch は S=1 で 4499 (Lily の 795 の 5.7 倍、48 層で 94/層)。**`--split-cb` の帰属: 量子化行列積が depth 0 で 10.0 / 22.3 ms (45%、855 dispatch)、短で 18.8 / 34.6 (54%、964)。残り ~3644 dispatch (elementwise、copy、sigmoid、sort、broadcast、softmax) が 55%。
  行列積は active 重み 1.7〜2 GB に対し 170〜200 GB/s = 帯域の半分。
- **壁 (5 ms) からの 5 倍の分解: 1.07 (空き) × 2.2 (糊のカーネル) × 2.0 (行列積が帯域の半分)。融合の的は行列積ではなく ~3600 本の糊。**
- depth 0 の上位 (回数 × us = ms/round): `affine_qmv_fast_b4` 422 × 10.9 = 4.60、`g1_copy` **133 × 13.4 = 1.79**、`affine_qmv_fast_b8` (lm_head 8bit) **1 × 1745 = 1.75 (7%)**、`affine_gather_qmv_fast` 96 × 17.9 = 1.72、
  `vv_Multiply` 257 × 5.3 = 1.36、`vs_Multiply` 264 × 4.5 = 1.20、`v_Sigmoid` 289 × 3.8 = 1.09、`Broadcast strided` 96 × 10 = 0.96、`affine_gather_qmv` 48 × 19.9 = 0.95、`block_softmax` 48 × 19.9 = 0.95。
  短 (S=3): `affine_qmv_wide` 566 × 12.7 = 7.20、`affine_gather_qmv_fast` 100 × **64.2** = 6.42、**`carg_block_sort` 96 × 24.2 = 2.32** (3 行のルーティングに sort)。
- 道具の限界: 既定モードの per-kernel は CB 内で按分 (帰属には `--split-cb`、壁が 2.5 倍)、busy は CB 単位 (中の空きは見えない = 上限)、直列段の長さは見えない、ioreg は走行を 3〜13% 遅くする (opt-in)。
  `xctrace` は MLX の compute を落とし、`mx.metal.start_capture` は常駐 98 GB を全部書くので、どちらも使えなかった。
- **次の的 (decode の本格改修)**: (a) S ≤ 8 の MoE ルーティングを sort 無しに (短で 2.3 ms = 6%)、(b) copy 133 本の出所 (transpose / 非連続) を潰す (1.8 ms)、
  (c) 層ごとの elementwise の糊 (multiply / sigmoid / broadcast / softmax、~4 ms) を `mx.compile` か 1 カーネルに畳む、(d) lm_head 8bit の 1.75 ms (4bit で候補 → 8bit で上位だけ再採点、品質の確認要)、
  (e) `wide` (射影の連結) を decode 幅で burn-in 付きに再測 (前の負けは位置 1 の段差込みの疑い)。

### 2026-09-03 18:20 本番のパックを lm_head 4bit (`~/models/ddalcu-mlxlm-head4`、真 bf16 から g64 で焼いた 9/2 23:03 のもの) に (ユーザー判断)

- 理由: 対戦相手 (mlx-serve / oMLX / rapid-mlx) は一律 4bit で、条件を揃える。decode で lm_head の 1 本 (1.75 ms/forward、7%) が半分になる見込み (+3〜4%)。
- 代金 (了承済み): KLD 0.01326 → 0.01794 (+0.0047)、top-1 一致 0.966 → 0.962。以後の KLD 基準は 0.01794。
- 運用: 小ベンチ / フルベンチはこのパックで。decode_ab の A/B は走行中のものは 8bit のまま (相対比較)、新しいものから head4。公開パック (HF) の差し替えは後日 (BACKLOG)。

### 2026-09-03 18:21 K2c 50k (`qsa-decode-kernel-50k.json`、両側 query、512 トークン): ms/round **-4.3%**、ms/tok -5.0%、tok/round +0.8%。既定 on を裏付け

対照 NG (case 0) は想定内: 50k では B 側が `_gather_tile_attn` を通り dense sdpa とビット一致しない (knob の docstring に記載)。A (K2c) は dense 意味論の並びと一致する側。

### 2026-09-03 18:35 P7 第 2 段の 8k (`moe-down-epi-8k.json`、1 プロセス、3 本 × 回文): **A (combine + x gather 畳み) -4.1%、head 一致 → 既定へ**。oracle の再走は無効

- A 12.40 / B (combine だけ) 12.62 / C (現行: x gather → (act·w) 実体化 → down GEMM → unsort gather + 和) 12.93 s。head は A/B/C で 6 本とも一致。発火 moe_combine 48。
- 仕組み (`kernels/moe_combine.py`、`fused.enable_moe_down_epilogue`): down の後ろの「unsort の gather + ルータ重み + top_k の和」を 1 カーネルに畳み、ルータ重みを down の後で掛けるので `(act·w)` の実体化 (126 MB 往復) も消える。
  x の gather も gate/up GEMM の行の読み方 (`row_src`) に畳む (micro では負けたが in-model では A > B)。丸めが 1 回減るので素より exact に近い側 (ビット一致はしないが head 一致)。
- 既定: `MLXTURBO_MOE_DOWN_EPI=combine` + `MLXTURBO_MOE_GATHER_FOLD=1` (runner への配線と 17k の確認は P7 のエージェントが続行)。
- **oracle-draft の再走は無効**: variant 2 / 6 の tok/round がどちらも 1.067 で draft が受理されていない (ms/round 36.2 / 57.4)。stub の `first=` の扱いか saveout の位置合わせ。差し戻し。

### 2026-09-03 18:31 P9 の見積もり (`prefill-width-8k.json`、チャンク幅 512 / 1024 / 2048、グループ上限を上げて MoE は同じ M=n-1 の 1 回、8k、3 本 × 回文): **畳む**

prefill_s: 2048 幅 12.784 / 1024 幅 12.763 / 512 幅 13.238。非 MoE のメンバー数 4 → 8 で差 -0.2% (誤差)、8 → 16 で +3.6%。
「メンバー数に比例する固定費」B は 2048 幅では実質ゼロで、P9 (4 → 2 か 1) の取り分は 0。細くすると (512) 1 個あたりが重くなる側に振れるだけ。**チャンク幅 8192 は畳む。**
(rapid-mlx の「QSA インデックスのバッチ処理 -30%」は、うちでは既に layer-major の MoE 集約と P6 で回収済みの領域と読む。)

### 2026-09-03 18:44 P7 第 2 段の 17k (`moe-down-epi-17k.json`): A **-2.9%** (28.28 / 現行 29.11)、B -2.1%。head は 6 本中 4 本一致 (1 ケースで A と C が分岐 = 丸め 1 回減の差)。8k -4.1% と合わせて既定へ (runner の配線待ち)

### 2026-09-03 18:51 oracle-draft の再走 (17k、`--save-out` 付き): まだ受理されない (tok/round 1.05 / 1.06)。副産物: 行の追加費用は 17k で **4.9 ms/行** (S=2 34.65 → S=6 54.32 ms/round、K2c on)

受理率の天井はこの knob では取れていない (真の次トークンの位置合わせが崩れている疑い、担当が調査中)。行費用 4.9 ms (以前の見積もり 7 ms) は K2c 後の値。

### 2026-09-03 19:10 oracle の顛末と 2 つの発見 (天井スタブのエージェント)

- `first=` の追従は済 (stub-draft は有効: 短で draft 3.1 ms = 8.2%)。oracle が当たらなかった原因は 2 つ:
  1. **ベンチのプロンプト池はリポジトリ自身** (`tools/_bench_text.py:text_pool()` = docs/**/*.md + README + .py)。セッション中に docs を編集すると、同じトークン位置の窓の中身が変わる。
     saveout と oracle の間に NEXT-SESSION-PROMPT.md を編集したので case 1 / 2 の prompt が別物になった。**罠 17: 走行をまたぐ比較 (別プロセスの A/B、小ベンチどうし) は、docs の編集で prompt が変わる。**池を凍結したファイルに置き換えるべき。
  2. **生成トークンが verify 幅 S に依存する**: 同じ prompt で depth 2 (S=3) の参照と S=2 / S=6 の oracle が出力 index 3 で分岐 (case 1)。K2c (qsa_decode_kernel) が既定になった時点と一致し、それ以前は S=2 / S=6 とも oracle が完全に当たっていた (= S 非依存だった)。
     同じ S での K2c on/off は head 一致 (17k の A/B) なので、K2c は「S ごとに参照 (MLX の sdpa) の並びを写している」= MLX 側の S 依存を持ち込んだのか、K2b の S ごとの経路差か。**確認要 (品質側)**: K2c on / off × S∈{1, 3, 6} で同一 prompt の greedy 出力を比べる。
- 綺麗な 1 点 (K2c 前、case 0): 現行 tok/round 1.896 / 43.87 ms、oracle S=2 2.000 / 34.31、S=6 5.953 / 57.28 (S=6 / S=2 = 1.67 倍)。

### 2026-09-03 19:19 小ベンチ 0903g (今日の既定全部 + lm_head 4bit のパック、pread、排他で冷却 10 分、`self-snapshot-turbo-small-0903g.json`)

| 文脈 | 冷 TTFT (0903d 06:30 →) | 温 TTFT | decode tok/s (0903d →) | 相手 mlx-serve (9/3 12:20) 冷 / decode |
|---|---|---|---|---|
| 0 | 0.17 → 0.18 | 0.14 | 50.2 → **51.4** | 0.18 / 54.7 |
| 4k | 6.60 → **5.95 (-10%)** | 0.16 | 51.9 → 51.7 | 5.70 / 55.7 |
| 17k | 31.3 → **30.3 (-3%)** | 0.20 | 46.1 → **47.5** | 27.8 / 49.0 |
| 25k | 45.5 → **43.5 (-4%)** | 0.23 | 44.2 → **47.5** | 41.3 / 60.8 |
| 32k | 56.4 → **54.3 (-4%)** | 0.26 | 42.7 → **44.8** | 51.6 / 47.4 |
| 50k | 89.9 → **86.7 (-4%)** | 0.80 | 42.7 → **47.0** | 82.1 / 45.9 |

- 入っているもの: P3 混合タイル、P10 BM=64 射影、P6 uint4、末尾 v2、QSA tail query、MIN_KV 8192、K2c、P7 第 2 段、lm_head 4bit (パック)。n-gram は pread。probe は 13.05 TFLOPS で一定 (冷却は効いている)。swap 3.7 GB 使用 (メモリ圧の痕跡は残る)。
- 相手比: 冷 prefill 1.04 / 1.09 / 1.05 / 1.05 / 1.06x 負け (9/3 朝の 1.13〜1.21x から縮小)、decode 4k -7% / 17k -3% / 25k -22% / 32k -5% / 50k **+2%**。温 TTFT は 4〜6 倍速 (50k は 20 倍)。
- **気になる差**: 17k の冷 prefill は decode_ab (末尾 v2 の A) で 26.4 s なのに、サーバー経路では 30.3 s (+15%)。checkpoint あり (BPE 境界) のチャンク割り、n-gram の同期、thinking テンプレートの差のどれかで、サーバー経路にだけ乗っている費用がある。次の prefill の的。
- 変な値は無いので push した。

### 2026-09-03 19:35 decode の糊の融合 (P11、4 つの的): **全部負け。糊の本数を減らしても decode は速くならない**

1 プロセス ABBA、burn-in、短 3 本 × 512、`MLXTURBO_DEPTH_ADAPT=0 --depth 2`。null 対照の雑音 -0.0% (±0.3%)。

| knob | 仕組み | op の減り (S=1 / S=3) | 短 ms/round | tok/round |
|---|---|---|---|---|
| moe-sort-min=128 | verify 幅の MoE gather を argsort 無しに | 0 / -384 (4.7%) | **+0.6%** | ±0 |
| glue-compile | MLP の `*up` と shared 合流を `mx.compile` で 1 本に | -192 / -192 | **+1.2%** | ±0 |
| moe-combine-glue (compile / matmul) | `(y·w).sum(-2)` を 1 本に | -96 | (無効: 関数名の衝突で既存の fold knob を呼んでいた。未計測) | — |
| wide-decode | attention の qkv 連結を decode 幅にも | — | +0.1% | **-4.1%** (qmv の変種が N で変わり出力が割れる) |
| fast-rope (既存) | rope の slice/concat を `mx.fast.rope` に | -96 | -0.0% | +0.3% |

- **読み**: trace の「dispatch 4499 本が壁」は糊の側では成り立たない。GPU busy 93% で、op を 5% 減らしても壁時計は動かない = 残りは行列積 (帯域の半分で走る qmv、S=3 の gather_qmv 64 us/呼び出し = 行ごとに違う専門家を読む代金) の側。
  短文脈 decode で残る手は (a) lm_head 4bit (既定に入れた、-0.9 ms)、(b) 受理率 (draft の top-k 命中率 → rerank、depth 4 は行費用 4.9 ms が壁)、(c) MoE の行間で専門家の読みを共有する自前 gather (D5 は重複率で畳んだ)。
  **短文脈 100 tok/s は M3 Max では現行の構造で届かない (現実線 55〜58)。**長文脈 (K2c) と prefill が勝ち筋。
- copy 133 本の出所 (`tools/decode_copy_probe.py`、グラフの op を出所別に数える): Attention 84 (うち rope の slice/concat 48)、GDN 36 (conv 窓の concat、消すには棄却済みの prework 融合が要る)、indexer 12、PLE 2。
  `Broadcast strided` 96 本 = ルータ重みの実体化 48 + shared gate の sigmoid 48。消しても速くならなかった。
- 訂正: moe-combine-glue の行は knob 関数名の衝突 (`_knob_moe_combine` が既存と同名) で既存の fold knob を呼んでいた。「compile 版でビット一致が崩れた」は fold の既知の負け方の再現で、compile 版は未計測。残る 3 つの判定は有効。
- 4 knob のコードは取り除く (方針: 効かない変種を二重に持たない)。`decode_copy_probe.py` は残す。

### 2026-09-03 19:50 読み直し: decode の壁は「dispatch あたりの床 ≈ 5 us × 4499 本」

- depth 0 の round 22〜25 ms ÷ 4499 dispatch = **4.9〜5.6 us/dispatch**。Lily は 5.4 ms ÷ 795 = 6.8 us/dispatch。**1 本あたりの費用は同じで、本数が 5.7 倍違う。**
- 行列積そのものは遅くない: 1 トークンの dense 射影 ≈ 1.3 GB (GDN 36 層の in_proj / out_proj が 1 GB、attention 12 層 0.28 GB) + MoE 0.25 GB + lm_head 0.3 GB (4bit) ≈ 1.9 GB を 10 ms = 190〜220 GB/s。
  qmv の大物は 300 GB/s 級 (ピークの 7 割)。ここを削る手は重みのビット数だけで、それは品質の取引 (lm_head は 4bit にした)。
- 19:35 の「糊は壁ではない」は言い過ぎ。あの 4 knob は本数を **4〜5%** しか減らしておらず、置き換えたカーネルの副作用 (compile 版が素より遅い、sort 無しの gather が遅い) の中に埋もれた。
  **必要なのは 2〜3 倍の削減** (94 本/層 → 30 本/層)。それには層の塊ごとの大きい融合 (MoE のルーティング〜combine、GDN の前処理〜再帰〜norm、HC の pre/post) を、MLX の qmv と同じ並列度で書くこと。Lily はそれをやった。
- 手の順: (1) MoE decode の融合 (PoL 走行中: ルーティング + gather + gate/up + SwiGLU + down + combine を 3 本程度に、行間で専門家を共有)、(2) GDN の層まるごと (prework は前に負けたが、並列度を直して再挑戦)、(3) HC の pre/post (elem 変種は decode 幅ならビット一致)。
  それぞれ「本数を 1/3 に」を目安に、in-model の ms/round で判定。

### 2026-09-03 19:58 K2c の S 依存の確認: **K2c は無関係** (S=2 / 3 / 6 で on/off とも一致、on≡off 9/9、128 トークン全部一致、round 数も同一)

- `mirror_blocks` は 17k で S によらず 256、pass 1 は行の絶対位置の可視集合で縮約、K2a の select も同じ → 構造的に S 依存を持ち込めない。
- 天井スタブのエージェントが見た「S=3 の参照と S=2 / S=6 の分岐」の容疑は (1) P7 第 2 段 (combine の縮約が M = B·S で並びを変える、丸め級)、(2) 凍結前のプロンプト池の揺れ (観察 19:18、凍結 19:24)。どちらも丸め級で品質の代金ではない。
- 既知の S 依存 (K2c 前から): 17k S=1 は `_gather_forward` の ratio guard を通って gather 経路、S ≥ 2 は dense。

### 2026-09-03 20:00 受理率の余地 (`bench/results/topk-trace.jsonl` 2356 round、depth 4 固定、8bit 頭、`tools/draft_topk_stats.py`): **受理率側は閉じる**

- 真のトークンが draft の top-k に入る率 (短 / 17k): d=1 top-1 **0.655 / 0.654**、top-2 0.776 / 0.761、top-4 0.864 / 0.830、top-8 0.918 / 0.885。d=2 以降は 17k の方が速く落ちる (0.611 対 0.490)。
  rerank は既に効いている (2bit 粗頭の top-1 に対し +0.06〜0.11)。
- 連続受理から出る tok/round (d=1/2/3/4): 短 1.655 / 2.090 / 2.373 / 2.543、17k 1.654 / 1.993 / 2.167 / 2.269 (実測 depth 2 の 2.144 / 2.122 と -2.5% / -6% で整合)。
- **行 1 つの追加費用 = round の 16%** (短 6.1 ms、17k 6.7 ms = verify 4.9 + MTP 1.8)。実測 (短、1 プロセス回文): depth 1 18.76 / depth 2 **17.78** / depth 4 18.86 ms/tok。**depth 4 は短 +6.1%、17k +17% で負け。**
- 木 (根で 2 本、verify 5 行) の上限は短 +9.7% / 17k +8.1% に対し費用 +24〜32% → 負け。top-2 を全部取れたとしても depth 2 で +12% で、行 1 つ (16%) に届かない。
- **残るのは draft 頭の top-1 そのものを上げること (0.655 → 0.776 で top-2 の取り分に相当) = MTP 頭の学習。方針で学習はしないので、受理率側の手は無い。**decode の tok/s は forward の費用 (dispatch 1/3) だけが手。
- 道具: `spec_flash._draft_topk_probe` (属性ゲート、既定 off)、`decode_ab --draft-topk K --topk-trace`、`ab_daemon.reset_engine` でスロットを消す。

### 2026-09-03 20:03 サーバー経路と decode_ab の 17k prefill の差 (+15%): **サーバー固有の費用は無い** (上限 20 ms = 0.07%)

- 1 プロセス内 A/B (`checkpoints=[]` 対 `None`、4 窓 × 2 巡、直交配置): 定常 28.425 対 28.447 s (-0.08%)。段ごとの ms も checkpoint 0.1 / tail split 0 / n-gram 先読み 0 / prime 43 / clear_cache 28 / tail forward 44 で両側同一。
  チャンク割りも同一 (末尾 v2 で最終チャンクが幅 1 なので `split_and_checkpoint_tail` は no-op)。head4 パックでも 27.7〜28.2 s。サーバーのリクエスト処理は parse → template 15〜19 ms が最大。
- **正体 = 新しいプロセスの最初の長い prefill の立ち上がり**: 空焼き後の 1 本目 30.09 → 29.19 → 28.64 → 28.50 → 定常 28.42 (-5.5%)。`self_snapshot` は温めが 4000 トークンなので 17k の冷 TTFT は必ずこの 1 本目。decode_ab は 1 本目を捨てた中央値。
  加えて走行間のばらつき ±5〜7% (今日の 17k は decode_ab 25.5〜29.3、サーバー 27.0〜28.2)。15:12 の 26.43 は動的プール + MIN_KV 8192 / P7 前のコード。
- **手**: サーバー起動時の空焼きに長文 (目標文脈と同じ桁) の prefill を 1 本入れる → ユーザーの最初の長文リクエストが 5〜10% 速くなる。品質の代金ゼロ、代金は起動時間 (17k で ~30 s) とメモリの先触り。ベンチの冷 TTFT もこれで定常値になる (相手と同じ harness で公平)。

### 2026-09-03 20:08 MoE decode の融合カーネル PoL (`kernels/moe_decode_fused.py`、E=512 の重み 2.83 GB を巡回、毎回別の専門家): **重複をまとめる筋は畳む**

- 正しさ: fp32 参照に対し自前の方が素より誤差が 2〜4 倍小さい (累算 fp32、1 回丸め)。
- 冷 micro (MoE 1 層 us): S=1 素 130.7 / まとめ rmax2 144.9 (1.11) / rmax4 165.7 (1.27)、S=3 256.3 / 273.2 (1.07) / 319.5 (1.25)、S=6 420.5 / 490.6 (1.17)。判定線 (S=3 で 0.70) 未達。
- **壁**: MLX の `gather_qmv_fast` は対ごとに読んで **395 GB/s (ピーク付近)**。重複をまとめると読むバイトは 27〜35% 減るのに **220 GB/s** に落ちる = 重複の対が threadgroup を増やして memory-level parallelism を稼いでいた。
  union の比 (S=3 で 0.727) から判定線 0.70 は算術的にも届かず、理想でも 0.78。「S=3 の gather_qmv が S=1 の 3.6 倍」は trace の CB 内按分の歪みで、冷 micro では 1.96 倍。**decode の行列積は帯域の壁に既に張り付いている。**
- 残した芽: `fused:1` (重複もソートもせず gate/up 1 本 + down 1 本、sort / gather / scatter / SwiGLU の op が消える): S=1 **0.929** / S=3 0.979 / S=6 1.056。dispatch を減らす側の手として in-model へ (判定線 短 -1%、head 丸め級)。
- 訂正の連鎖: 18:10 の「行列積が帯域の半分」は CB 内按分の帰属の歪み。20:00 の読み直し (壁 = dispatch の床) が残る。

### 2026-09-03 20:19 HC elem の再測 (burn-in、depth 2 固定、head4、`hc-elem-burnin.json`): 短 ms/round **±0.0%** (ms/tok -0.6%)、17k **-0.7%** (ms/tok -1.1%)。短の head 一致、17k は 1 ケース分岐 (prefill 幅の post 段)

- 前の +2.4% (14:55) は burn-in 無しの位置 1 の段差込みだった。代金ゼロ方針で **既定を elem に** (`MLXTURBO_HC=elem`、kernel / compiled / off で切替)。発火は decode / verify 幅 (行数 <= 8) だけに絞った (`eligible_elem` に行数ゲート) ので、素とビット一致。
  kernel (sigmoid 1 ulp、mixed 97.5% 一致) は既定から降りた = 品質は上がる側。
- dispatch は 14 → 6 / 呼び出し (× 97 = 1 step で -776 本 = 17%) なのに壁時計は 0〜-0.7%。**「dispatch の床 5 us × 本数」の読み (19:50) も過大**: 融合で消えるのは起動の一部だけで、小さいカーネルの中の DRAM レイテンシの連鎖は残る (HC の冷 micro で -9 us/呼び出し = 62 → 53)。
  decode の残りの取り分は「融合した先のカーネルが高い並列度で走ること」に懸かる。本数を減らすだけでは動かない。

### 2026-09-03 20:40 サーバー起動時の長文空焼き (`MLXTURBO_WARMUP_TOKENS`): **効かない → 既定 0 (knob は残す)**

- 16384 トークンの空焼き (痕跡なし、`session=None`、`bench/textpool-frozen.txt` の先頭) を起動時に 1 本。self_snapshot 17k、0 対 16384 を連続で: 冷 TTFT **27.21 対 27.36 s (+0.6%)**、温 TTFT 0.197 / ctx 0 は同一、起動 +27.1 s。
- 効かない理由: self_snapshot は測定前に 4000 トークンの温めを既に 1 本流している (20:03 の「1 本目」はその後の 1 本目)。4 本かけて収束する立ち上がりのうち、空焼き 1 本で買えるのは最初の 1 段だけで、既存の温めが既に取っている。
- 今回の 2 本とも 27.2〜27.4 s で、19:14 の小ベンチ 30.3 s より 10% 速い (作業ツリーの GDN の未 commit 変更か、走行間のばらつき ±5〜7%)。**17k の冷 TTFT はこの幅で揺れる**と読むこと。

### 2026-09-03 20:55 P7 第 3 段 (行列積以外の残り) と低行数 MoE GEMM の掃引: **畳んだ**

**結論を先に: P7 第 3 段は畳んだ (knob のコードも取り除いた)。**判定線 (-1%) に届かない。残り 4 つ (router 0.82 /
topk 0.31 / sort 0.34 / swiglu 0.40 = 1.87 ms/層 = 8k prefill の 2.9%) を部品ごとに
測ったところ、**router は既に床**で、取り分は topk 0.083 + swiglu 0.068〜0.123 の
0.15〜0.21 ms/層 = **8k prefill の -0.24〜-0.33%** しかない。残っている唯一の的は
**sort (0.26〜0.33 ms/層)** で、これは計数ソートのカーネルを書かないと取れない。

#### micro (`tools/moe_route_micro.py`、モデル無し、rows=2048 / E=512 / top_k=10 / K=2560、16 組を巡回、`bench/results/moe-route-micro{,2}.json`)

router (ms/層):

| 変種 | ms |
|---|---|
| fp32_qmm (現行 = `x.astype(fp32)` + 4bit qmm) | 0.843 |
| fp32_qmm (scales/biases は bf16 のまま = 本番の形) | 0.841 |
| fp32_qmm、cast を計測の外に出す | **0.764** |
| bf16_qmm | 0.698 |
| bf16_qmm -> fp32 | 0.777 |
| f16_qmm -> fp32 | 0.760 |
| cast だけ (`x.astype(fp32)`) | 0.258 |

- **cast は単独では 0.258 ms だが、qmm と同じ eval に入れると 0.079 ms しか増えない**
  (0.764 -> 0.843)。fp32 の x を書いてすぐ読むのでキャッシュに乗る。cast を消す
  (bf16 を読んで threadgroup で fp32 に上げる自前 GEMM) の取り分は 0.08 ms/層 =
  8k の 0.12% しかない。**割に合わない。**
- bf16 で回すと -0.145 ms/層 (-0.23%) だが、**top-k スロットの 2.2% (453/20480) が
  入れ替わる** (logits の平均相対誤差 1.7%。logits は打ち消しが効くので、重みを
  bf16 に丸めた誤差が相対では大きく出る)。品質を売って速度を買わないので却下。
- **router は床。**この項目は閉じる。

topk: 素 (符号反転 + argpartition + 切り出し + take_along_axis + softmax) 0.299 ->
`kernels/moe_route.py` のカーネル 0.216 (**-0.083 ms/層**)。選ぶ集合は 20480/20480 一致
(順序は降順で素と違う)。2026-09-01 に decode 幅で負けた (+0.34 ms/token) のと逆で、
prefill 幅は 2048 threadgroup 立つので逐次 top_k 回のパスが隠れる。

sort: `argsort` だけで 0.224、並べ替えまで 0.239、`_inv_perm` 込み 0.247、表
(`counts_from_sorted_ids` + `segment_tables`) 込み 0.293。`counts` 単体が 0.167。
**20480 要素 (80 KB) の並べ替えに 0.24 ms は MLX の汎用ソートの多段起動ぶん**で、
計数ソートなら 0.05〜0.08 に落ちるはず。

swiglu (`tools/moe_grouped_gemm_micro.py --stage glu`、gate/up GEMM 込み、M=20480):

| | ms | 比 | 素との差 |
|---|---|---|---|
| 素 (`nn.silu(g) * u` = 2 op) | 15.130 | 1.000 | — |
| `epi` (up GEMM の store に畳む) | 15.007 | 0.992 | 平均 4.0e-2 (素の平均絶対値 25.44) |
| `compile` (2 op を 1 op に) | 15.062 | 0.995 | **ビット一致** |

`nn.silu` 自体が `mx.compile` 済みなので素は 2 本 (silu + 乗算)。1 本に畳むと 26 MB の
往復が 1 回消えて -0.068 ms/層で**ビット一致**。GEMM の store に畳む (`epi`) 方が
速いが (-0.123)、frag の並び (8 行 x 2 要素) で読むので coalescing が悪く、期待した
-0.3 には届かない。

#### 配線 (測ったときの形。取り除き済み)

- vendor `SparseMoeBlock.__call__` にフック `_MOE_ROUTE` を 1 個追加
  (`_MOE_DOWN_EPILOGUE` と同じ作法。差された関数が `(idx, w)` か `None` を返す)
- `fused.enable_moe_route_kernel` (`MLXTURBO_MOE_ROUTE_KERNEL=1`、行数 >= 1024)
- `fused.enable_moe_swiglu` (`MLXTURBO_MOE_SWIGLU=off|compile|epi`)
- `kernels/moe_grouped_gemm.qmm_segmented` に `glu=` (TGLU) を追加。ついでに
  `bn` / `bk` (列 / K のタイル幅) と `bm=8` を掃引用に開けた。**既定 (bn=32/bk=32)
  では生成される source が 1 文字も変わらない**ことを assert で確認している
- knob `moe-route3` (A = compile + topk カーネル / B = epi + topk カーネル / C = 現行)

合成の `SparseMoeBlock` (本番と同じ形、`enable_moe_grouped_gemm` +
`enable_moe_down_epilogue` を当てた本番経路) で A/B/C を突き合わせ:
A は平均絶対差 4.1e-10 (|C| 平均 0.0115)、B は 3.0e-5。どちらも丸め級。

#### in-model 8k (`bench/results/moe-route3-8k.json`、1 プロセス、長文脈 3 本 x 回文、`--tokens 8`)

| | prefill_s | ms/round | tok/round |
|---|---|---|---|
| A (compile + topk カーネル) | 12.560 (+0.3%) | 37.505 (+1.2%) | 2.222 |
| B (epi + topk カーネル) | 12.492 (-0.3%) | 38.097 (+2.8%) | 2.222 |
| C (現行の既定) | 12.527 | 37.050 | 2.222 |

**判定: 落ちる。**prefill_s は A/B とも ±0.3% で、micro から見積もった -0.24〜-0.33%
は揺れに埋まって見えない (判定線 -1% には遠い)。発火は確認済み (`moe_route=48`、
`moe_combine=48` と同数 = 同じ MoE 層で発火している)。ms/round の +1.2/+2.8% は
decode 側の揺れ (3 ラウンドしか無く、行数ゲートで decode 幅は 1 op も変わらない)。

**4k は測っていない。**per-層の取り分は文脈長に依らないので、8k で揺れに埋まる
ものが 4k で出る理由が無い。

**畳んだ (2026-09-03、親の判定)。**「効かない変種を二重に持たない」方針に沿って、
配線のコードは全部取り除いた: vendor の `_MOE_ROUTE` フック、
`fused.enable_moe_route_kernel` / `enable_moe_swiglu`、`qmm_segmented` の `glu=`
(TGLU)、knob `moe-route3`。残したのは測り直せる道具だけ --
`tools/moe_route_micro.py` (新規) と `tools/moe_grouped_gemm_micro.py` の
`--stage glu` (素 対 compile) / `--bn` / `--bk` / `seg8`、および
`qmm_segmented` の掃引用引数 `bn` / `bk` / `bm=8` (**既定 bn=32/bk=32 では生成
される source が 1 文字も変わらないことを assert で確認**)。

#### 低行数 MoE GEMM (別の的): **BN=64 / BM=8 / BK=64 は全部負け。mix48 が最良のまま**

`tools/moe_grouped_gemm_micro.py --stage segmented`、MoE 層 1 つ (gate/up x2 + down)、
dense 比。`--bn 64` の回と `--bk 64` + `seg8` の回は別プロセス (後者は全体に ~7% 遅い
回) なので、比べるのは各回の中だけ。

| 変種 | r=40 | r=160 |
|---|---|---|
| mix48 (現行の既定) | **1.253 / 1.248** | **1.056 / 1.067** |
| mix48-bn64 | 1.296 | 1.053 |
| seg16-bn64 | 1.350 | 1.162 |
| seg8 (BM=8) | 1.478 | 1.405 |
| mix48-bk64 | 1.373 | 1.154 |
| seg8-bk64 | 1.580 | 1.453 |
| seg32w1-bk64 | 1.462 | 1.146 |

**なぜ 1.10 に寄らないか (実測 2 つから)**:

- 16 行タイル 1 枚 = 32 行タイルの **0.519 倍** (gate/up) / 0.511 (down)、
  端数 (20 行) タイルの費用 c = 0.970 / 0.974 (どちらも今回も再現)
- `seg8` の実測から cost(8)/cost(32) = **0.33** (seg8/seg16 の時間比 1.169 x
  タイル枚数比 1497/2755 x 0.519)。ここから a + 8b = 0.33、a + 16b = 0.519 を解くと
  **タイル 1 枚の固定費 a = 32 行タイルの 14%**、行あたり b = 0.0236
- r=40 の行の水増し Σceil(c_e/BM)·BM / M は BM=8 で 1.076、16 で 1.170、32 で 1.375、
  mix48 で 1.212

固定費 14% と水増しの積で、**「タイル 1 枚 = 専門家 1 つ」の枠内の床は 1.19〜1.25**。
BM を下げると水増しは減るが固定費が増えて相殺し (seg8 が実測 1.478)、BN / BK を
広げるとタイル数が減るぶん占有率が落ちて負ける。**1.10 はこの枠では届かない。**

寄せる唯一の筋は「タイルが専門家境界をまたぐ」形: 専門家セグメントを **8 行**に
揃えて (水増し 1.076)、frag の行グループ (8 行) ごとに B タイルを選ぶ (threadgroup
メモリに W タイルを最大 4 枚)。モデル上は 689 タイル x 1.0 + 444 straddle x 0.14 =
**1.17** で、-6% (4k prefill の -1.5% 相当)。steel の写し (`mlxturbo_mma_rows`) に
per-frag の B 選択を足す大きい変更で、しかも 1.10 には届かない。**優先度は低い。**

### 2026-09-03 21:10 GDN decode の「行列積以外」を 16 本 → 3 本に (PoL、`MLXTURBO_GDN_DECODE_FUSED`)

**結論: 機構は完成した (素とビット一致、dispatch 16→3) が、in-model は ms/round -2.4% で
判定線 (-5%) に届かない。**取れた 0.88 ms/round の内訳から、20:00 の「dispatch 1 本 5 us」は
糊には当てはまらないことが分かった (糊 1 本は 1.9 us)。

**現状の本数** (`tools/gdn_decode_micro.py --mode count`、GPU 不使用。GDN 1 層の前処理+再帰+出力 norm、
view で済む op は除く)。S=1 と S=3 で同じ:

| | 素 | 自前 |
|---|---|---|
| 合計 | **16** | **3** |
| 内訳 | RMSNorm 3 / Multiply 3 / AsType 3 / Sigmoid 2 / Concatenate 1 / Convolution 1 / silu(compiled) 1 / compute_g(compiled) 1 / 再帰カーネル 1 | 前処理カーネル 1 / 再帰カーネル 1 / 出力 norm カーネル 1 |

36 層で 1 forward あたり **468 dispatch** が消える (S=1 の 4499 本の 10.4%)。

**設計 (`mlxturbo/kernels/gdn_prework.py` を書き直した)**: 旧版は 1 threadgroup = 1 トークン位置で、
S=1 では threadgroup が 1 個しか立たず 40 コアの 1 コアで 250 KB を読んでいた (2026-09-03 12:00 の
HC と同じ形の負け方)。現行は **1 threadgroup = (1 トークン位置, 連続 128 チャネル)** で、S=1 でも
81 個立つ。1 スレッド 1 チャネル、`conv_w` の 4 タップは 8 B の連続読み。BLOCK は dk の倍数なので
q/k の 1 head が threadgroup 内に収まり、rms_norm の縮約が閉じる (threadgroup メモリの部分和 1 回、
旧版の tg_q/tg_k 退避は不要)。g/beta は専用の最終ブロックが持つ。

**正しさ: S ∈ {1,2,3,6} で q/k/v/g/beta/conv 状態・ブロック出力・再帰状態 (fp32) が全部ビット一致。**
判定線 (bf16 1e-2、fp32 1e-5) より強い。ここに行くのに丸め位置の総当たり (196608 標本) が要った:

| 変種 | `compute_g` との不一致 |
|---|---|
| log1p の結果を float のまま足す | 2.99% |
| softplus 内の exp を float / precise / fast にする | 15.45% |
| g の外側の exp を素の `metal::exp` にする | 32.68% |
| **全部 T (bf16) で丸め、外側 2 つだけ `metal::precise::exp`** | **0.00%** |

MLX 本体は `Exp` だけ `precise` を明示 (`unary_ops.h:177`)、`LogAddExp` の中は素の `metal::exp`。
`A_log`/`dt_bias` は実モデルどおり bf16 のまま受ける (旧版が作らせていた fp32 の写しは
`a + dt_bias` を fp32 に上げてしまい、素と別の値になる。**写しは作らないよう `enable_gdn_prework_kernel`
から外した**)。silu の sigmoid も本体の写し (`hyper_connection.py` の `mlx_sigmoid`)。

**冷の連鎖 micro** (`tools/gdn_decode_micro.py --mode bench`、36 組巡回、ABBA×4、焼き入れ 1 往復捨て。
前処理+再帰+出力 norm の 3 段だけの us/呼び出し):

| S | block | 自前 | 素 | 比 |
|---|---|---|---|---|
| 1 | 128 | 48.3 | 72.1 | **0.670** |
| 1 | 256 | 51.4 | 71.3 | 0.722 |
| 3 | 128 | 64.1 | 81.8 | 0.784 |
| 3 | 256 | 64.5 | 82.1 | 0.786 |

S=1 の判定線 0.70 は最良配置 (block=128) で通る。ただし**素の側だけ ±20% 揺れる** (S=1 の
plain は 4 セットで 60.2 / 71.3 / 72.1 / 74.0)。S=3 は 4 配置とも 0.78〜0.79 で安定。
GDN の前処理が読む重みは 36 層で 3 MB しか無く、100 MB の冷条件は原理的に作れない
(量子化行列を持たないため。`--with-proj` で in_proj/out_proj を連鎖に入れると 620 MB になる)。

**in-model、短文脈 3 本 × 512** (`--knob gdn-decode-fused`、A=前処理+norm / C=前処理だけ / B=素、
`MLXTURBO_DEPTH_ADAPT=0 --depth 2`、`bench/results/gdn-decode-fused-short.json`):

| | ms/round | tok/round | ms/tok |
|---|---|---|---|
| A (3 本) | 36.06 (**-2.4%**) | 2.090 (-2.2%) | 17.41 (+0.5%) |
| C (前処理だけ、4 本) | 36.15 (**-2.1%**) | 2.114 (-1.1%) | 17.18 (-0.8%) |
| B (素、16 本) | 36.94 | 2.137 | 17.32 |

- **ms/round は 3 本 × 2 反復すべてで -2.4%**(prompt ごとに -2.4 / -2.4 / -2.4%)。ばらつきが無い。
- 出力 norm の寄与は -0.4% だけ (前処理が -2.0%)。
- tok/round は prompt ごとに +1.2 / **-10.0** / +1.7% とばらける。**系統差ではなくテキスト運**
  (micro では S∈{1,2,3,6} でビット一致なので、残る差は `mx.conv1d` の fp32 加算順だけ)。
  3 本平均の -2.2% はこの 1 本に引かれた値で、ms/tok の判定には使えない。
- 発火 `gdn_prework=8640` (= 36 層 × 240 forward = 1 round 1 forward)、`rms_norm_gated=8640`。

**判定: -2.4% は判定線 (-5%) 未達。** 既定 off のまま置く (`MLXTURBO_GDN_DECODE_FUSED=1|pre|norm`)。

**取れた数字の読み替え (20:00 の見立ての訂正)**: 468 dispatch を消して 0.88 ms/round =
**糊 1 本あたり 1.9 us**。20:00 の「round 22〜25 ms ÷ 4499 = 5 us/dispatch」は平均であって、
5 us は 855 本の量子化行列積 (10.9〜17.9 us/本) が持ち上げている。**elementwise の糊は 1.9 us/本。**
全部の糊 (~3600 本) を消しても 6.8 ms (23 ms の 30%) が上限で、「本数を 1/3 にすれば壁時計も
1/3」にはならない。19:35 の「4〜5% 減では動かない」と今回の「10% 減で 2.4%」は同じ直線に乗る。
**GDN の非行列積はこれで閉じる**(残り 3 本を全部消しても、あと -0.5% しか無い)。

**in-model、17k × 512** (`--knob gdn-decode-fused --only long --ctx 17000 --prefill-once`、
`bench/results/gdn-decode-fused-17k.json`):

| | ms/round | tok/round | ms/tok |
|---|---|---|---|
| A (3 本) | 39.64 (**-2.0%**) | 1.957 (-2.1%) | 20.26 (+0.1%) |
| C (前処理だけ) | 39.78 (-1.6%) | 1.977 (-1.1%) | 20.20 (-0.2%) |
| B (素) | 40.43 | 1.999 | 20.24 |

短文脈と同じ形: **絶対の取り分は 0.79〜0.88 ms/round で文脈に依らない** (1 round 1 forward、
36 層 × 13 本)。ラウンドが長い 17k では割合が小さくなるだけ。tok/round のばらつきも短文脈と同じ
(prompt 1 は A -0.4%、prompt 2 は A -4.8% / C +3.7%) で、系統差ではない。

### 2026-09-03 21:10 decode のモジュール別帰属 (`tools/decode_module_attrib.py`、head4、burn-in、`bench/results/decode-anatomy*.json`)

| | 壁 ms/round | dispatch | GPU 和 | busy |
|---|---|---|---|---|
| S=1 短 | 23.00 | 3638 | 20.8 | 91% |
| S=3 短 | 37.18 | 4295 | 34.1 | 92% |
| S=1 17k | 26.51 | 4598 | 23.8 | 90% |
| S=3 17k | 42.71 | 5255 | 39.7 | 93% |

- **行列積は帯域の壁に張り付いている**: 1 トークンで読む重み 3.76 GB (専門家 1327 MB、GDN in_proj 854、HC 367、lm_head 358、GDN out 321、attn 336、shared 133) を 8.7〜8.9 ms = **427〜434 GB/s (ピーク 409.6)**。取り分は無い。
- 残り 58〜63% (S=1 で 12〜15 ms) が糊。モジュール別 (S=1 短、dispatch / ms): MoE 専門家 432 / 5.25、**MoE router〜combine 864 / 3.37**、HC 581 / 3.17、GDN 非行列積 684 / 2.86 (→ 融合で 3 本に、既定 on)、GDN in_proj 144 / 1.97、attention qkv 444 / 1.30、lm_head 0.83、HC 書き戻し 96 / 0.84、shared 240 / 0.44、PLE / n-gram 82 / 0.39。
- **行の費用 (S=3 − S=1 = 2 行、短 14.2 ms)**: MoE 専門家 **+7.4 (55%、和集合が 2.18 倍 = 帯域)**、MTP draft +2.2 (16%)、GDN +1.5 (12%)、HC +1.2 (9%)、prime +0.8。GDN の再帰は行数に対して逐次ではない (36 回 × 18.6 us で不変)。幅分割も主因ではない。
  **行費用の 55% は帯域で、融合では消えない。**
- **次の的 (S=1 短、ms/round)**: (1) MoE gather の `g1_copy` (x を 10 行に複製、96 × 15.6 us = 1.49) + `arange` (144 × 6 us = 0.86) = **2.35 ms** (専門家の行列積 2.22 より大きい) → `fused:1` が当たる (走行中)。
  (2) **MoE router〜combine を 18 → 4 dispatch/層** (block_softmax 48 × 19 us = 0.91、f32 化 copy 0.53、sort 0.31、combine の縮約 4 本 0.97) → **-2.0〜2.3 ms**。間に行列積が挟まらないので 1 本に落ちる。条件は融合先の並列度。
  (3) 最終 mixer の `hc_elem_post` (p1 変種) が 1 呼び出し **462 us** (同形の c0 は 6.9 us、67 倍) → -0.46 ms。`combine=False` 分岐の疑い。micro で確認してから。
  (4) HC 書き戻し `_combine` 96 × 9.6 us = 0.92 → 直後の `hc_elem_pre` に畳んで -0.8 ms。
  合計の見込み -5 ms/round (S=1 23 → 18 ms = 相手の 18 ms 級)。
- 注意: モジュール別の ms は組み立て値 (dispatch 数と region × カーネル名の回数は厳密、時間は位相分け + split-cb の単価 × 回数 + 行列積は重みバイト按分)。

### 2026-09-03 21:30 HC の 2 つの的: post の 462 us は帰属のアーチファクト、`_combine` の畳み込みは in-model +0.6% で棄却

- `hc_elem_post` p1 の 462 us は `--split-cb` の per-kernel が「CB の GPU 区間まるごとを dispatch 数で按分」した値で、1 CB に 1 dispatch のときはそのカーネルの実費ではない。micro では p1 4.3 us / c0 6.6 us (p1 の方が速い)。同じカーネルが CB の詰まり方で 44.8 と 7.0 に振れることを再現。
  **split-cb の per-kernel は帰属の順位付けにだけ使い、絶対値を取り分にしない。**
- `_combine` の畳み込み (次の層の pre に書き戻しを畳む、`is` 判定で 1 枠、ビット一致): 冷 micro -1.6〜-1.8 us/call なのに in-model 短 **+0.6%** (3 本とも同じ符号、tok/round 一致)。
  読み: `_combine` は 10240 要素の平たい elementwise で threadgroup が沢山立ち、**隣の行列積と重なって走れる**。畳むと pre (S=1 で 4 threadgroup) の依存の直列に乗る。連鎖 micro は前後に行列積が無いのでこの重なりを再現しない。
- **「dispatch を減らすだけでは壁時計が動かない」の 3 例目** (HC elem 20:19、GDN 21:05 は勝ったが -2.4%)。MLX は独立な小カーネルを隣の行列積と重ねて走らせているので、糊は既に半分隠れている。
  融合が効くのは「依存の直列に乗っている糊を、並列度の高いカーネルに置き換える」ときだけ。的の見積もり (21:10 の -5 ms) は下方修正: fused:1 と router〜combine の 2 本で **-1〜2 ms** が現実線。
- **decode の判断規則**: この 2 本が既定に入っても入らなくても、短文脈 decode の糊の融合はここで一巡とし、次は 27B レーン (部品の置き換え) へ。残る decode の手は「行列積の帯域そのもの」(重みのビット数 = 品質) だけ。

### 2026-09-03 21:45 prefill の内訳 (今日の既定、head4、`tools/prefill_ctx_anatomy.py`、`bench/results/prefill-anatomy-0903-{4k,8k,17k}.json`、部品和 − 壁時計 = +0.8〜1.3%)

| 部品 | 4k | 8k | 17k |
|---|---|---|---|
| MoE 行列積 | 36.1% | 35.1% | 32.4% |
| MoE それ以外 | 3.2% | 2.5% | 2.5% |
| GDN in_proj / out_proj + norm | 16.6 + 8.3% | 15.8 + 7.8% | 15.2 + 7.5% |
| GDN scan | 3.8% | 3.6% | 3.5% |
| attention q/k/v + norm + rope | 5.0% | 4.9% | 4.6% |
| **attention sdpa / gather** | 4.3% | **8.3%** | **13.0%** |
| **HC 読み** | **11.7%** | 11.4% | 10.5% |
| **PLE / n-gram** | **6.5%** | 6.3% | 6.0% |
| グループ境界 / 末尾チャンク | 0.1 / 0.6% | 0.1 / 0.3% | 0.1 / 0.1% |

- 壁 (11.2 TFLOPS の dense 上限比): MoE GEMM 79〜82%、GDN の射影 80〜83%、attention の射影 95〜97% → 張り付き。**HC 読みだけ 63〜65%** (細長 GEMM 10240→320 / 320→10240 と elementwise)。sdpa は kv=4096 で 7.5 TFLOPS (遅くはない、減らす的)。
- **次の的 3 つ**: (1) attention の kv < 8192 の dense 帯 (8k で 8.3%): budget 2048 の疎性を小 kv でも使える専用カーネルが要る (現行の gather カーネルは kv=8192 で dense の 1.43 倍遅く交差点 11k)。8k で -5%、4k で -2%。
  (2) HC 読み: 下限との差 4.3 / 3.8 / 3.6 点。norm を down の load に、sigmoid と mean を up の store に畳む、細長 GEMM に BM=64 タイル。相手が融合している場所 (9/2 の分析) と同じ。
  (3) **n-gram の行取得 6%**: `_prefetch_ngram_span` は走っている (段の ngram_lookahead 0 ms) のに費用が PLE に残る = **先読みが実際には重なっていない**。 **(訂正 9/4 02:25: 走っていなかった。pread では `prefetch_enabled` の既定が off で、各 PLE 層で即 return していた。0 ms は何もしていなかったから。下の 02:25 の項)**RAM 常駐の separate (32 GB) だと 0.9%。先読みを本当に重ねれば 32 GB を払わずに -5%。
- 死んだ的: グループ境界 0.1%、末尾チャンク 0.1〜0.6% (末尾 v2 が畳んだ)、MoE それ以外 2.5〜3.2%、GDN scan 3.5〜3.8%。
- 相手との差の推測: 4k の +0.25 s は HC 読みの非効率 (0.25 s) と MoE 非行列積 (0.19 s) で桁が合う。17k はばらつき (±5〜7%) の中。

### 2026-09-03 22:00 MoE の router〜combine の decode 融合 (18 → 4 dispatch/層、`kernels/moe_route_decode.py`): 冷 micro 0.569 で通ったが **in-model +0.05% → 畳む**

- 正しさ: top-k の集合 100% 一致、重み 1e-6、shared ゲートはビット一致、region 出力 S=1〜3 ビット一致 (S=6 で 1 ulp)。冷 micro (144 組 106 MB 巡回) S=1 0.569 / S=3 0.710。
- in-model 短 (depth 2 固定、head4): 全部 +0.05% / route だけ +2.6% / combine だけ +1.2%。**決め手は変種 E** (shared ゲートだけ 1 本に、ビット一致 = 生成列も round 数も同一の対照): dispatch **-192 本/round (-5.2%) で ms/round ±0.0%** (6 本全部)。
- trace: 素 3723 本 / GPU 和 32.4 ms / 稼働率 96.1% / 隙間 0.4 us、自前 3078 本 (-17%) / 32.9 ms (+1.4%) / 95.9%。消した隙間 0.26 ms < 融合先が増やした GPU 時間 0.46 ms。
- **決定的な読み替え**: dispatch の本数そのものの値段は、この構成 (稼働率 96%、隙間 0.4 us) では実質ゼロ。GDN 融合の -0.88 ms は「融合先が素より GPU 時間を使わなかった」(冷 micro 48 対 72 us) 取り分で、本数の削減ではない。
  **decode の糊の融合レーンはここで閉じる** (勝ったのは GDN 1 本)。短文脈 decode は MLX の上では 52〜54 tok/s が天井。fused:1 の結果を待って 27B レーンへ。

### 2026-09-04 01:45 小 kv (< 8192) の attention に QSA の疎性を使う案 (union タイル化): 全域で取り分ゼロ → 畳む (`tools/qsa_vis_stats.py`、`bench/results/qsa-vis-stats-{8k,17k}.json`)

- 道具: prefill を 1 本流して `QSAIndexer._select_keep` の keep_block (per-query tail、本番の `MLXTURBO_QSA_TAIL=query`) を持ち帰り、行あたりの可視率とタイル幅 T ごとの union 率を host で出す。GPU は 1 本ぶん。
- 可視率 (8k、S=2048 のチャンク): kv=4096 **50.0%**、5825 35.2%、7872 26.0% (budget 2048 の算数どおり)。17k 側は 25.0 → 12.1%。dense 行タイル (P5、R=256、前方の K/V だけ) は総なめの 78〜89%。
- **QSA の top-512 は隣接クエリ間で揃っていない**: kv=7872 で行あたり 26.0% なのに隣接 8 行の union は 48.7% (+87%)、32 行で 74.3%。レーン 3 (T=4 で union 36%) と同じ形で、4k 帯はもっと悪い。
- 机上のルーフライン (`t = FLOP/EFF + 読み列×2048 B/BW`、EFF は fallback 実効 10.1 TFLOPS / 現行 T=1 カーネル 6.0 TFLOPS、BW は現行の load 実測 207 GB/s と楽観 400)。分母の検算: 8k 全チャンク × 12 層で 958 ms ≈ anatomy の 8.3% (1.08 s) と 11% 以内。

| kv | 最良 T | union/n_blocks | FLOP 比 | 読み比 | t/dense (207) | (400 楽観) |
|---|---|---|---|---|---|---|
| 4096 | 32 | 0.726 | 0.930 | 7.4 | **1.05x** | **0.99x** |
| 5825 | 32 | 0.729 | 0.843 | 6.5 | **0.95x** | 0.90x |
| 7872 | 16 | 0.574 | 0.648 | 10.4 | **0.81x** | 0.73x |

- 挟み撃ち: head_dim 256 / Hq24:Hk2 の演算強度は 12·T FLOP/byte で、機械の 25 を超えるには T ≥ 2〜4 が要るが、その T で union は既に 78% (4096) / 58% (5825)。**0.7 を切る T は無い。kv=4096 は最良でも 0.99x。**
- 8192 より上でも死んでいる: 17k で現行カーネル (load 41.6 / score 17.1 ms) を比で動かすと T=4 で 0.87x が最良だが、P6 の uint4 (load 25 ms) と重ねると 43.7 対 42.1 ms で**上乗せゼロ**。P6 のままでよい。
- 注意: 模型は 8〜12k 帯で現行 MIN_KV=8192 の in-model (10k -0.8%) と符号が合わない (dense 側が模型より高い費用を払っている疑い: `_final_mask` の bool 組み立て、fallback のスコア実体化)。今回の判定 (4k/6k 帯) は dense 側が高い方向の誤差なので動かないが、8〜12k の絶対値は信用しない。
- **残る筋は疎性ではなく「スコアを実体化しない融合」(K1 の arm A、見込み 4k -1.3〜-2.0%)。**21:45 の見立て「8k -5% / 4k -2%」のうち疎性で取れる分はゼロと確定。

### 2026-09-04 02:25 n-gram の先読みは走っていなかった。既定 on + 投入位置を forward の前に → 既定に入れる (17k 冷 -2.9% / 4k 冷 -1.0% / 17k 温 -1.8% / 4k 温 -0.4%、出力一致)

- 診断 (`tools/ngram_prefill_diag.py`、17k、`FASTMLX_NGRAM_NOCACHE=1` = 冷): `prefill-anatomy-0903-17k.json` の `ngram_stats` に `prefetch_rows=0 / prefetch_done=0 / hits=0` が既に記録されていた。`StreamNGram.__init__` の `_pf_default` が pread では "0" で、`_prefetch_ngram_span` は各 PLE 層で即 return。21:45 の「走っている」は誤読。
- 6% の本体は「同期」でも「プール競合」でもない: `sync_ms` は 17k 全体で 18 ms、GIL の取り合いは 1.00x (`tools/ngram_fetch_micro.py`: 1 チャンク 32,768 行の pread 328 ms、F_NOCACHE 340 ms、再読み 151 ms、ヒット経路 7.2 ms)。
- 旧配置 (境界を組み終えて `mx.eval` の直前) では、PLE 層が layer 1 (先頭付近) なので次の境界の行は構築のほぼ最初に要求され、隠せる窓が直前境界の eval 1.4 s しか無い → 1 境界ぶんの pread (約 1 s) を隠しきれず、on にしても最初のチャンクで 23,437 miss (580 ms)。

| mode | 先読み | call の hit/miss | sync_ms | fetch_ms | wall |
|---|---|---|---|---|---|
| off (9/3 の本番) | 無し | 0 / 269,856 | 18 | 2849 | 32.4 s |
| late (旧配置を on に) | 背景 985 ms | 136,691 / 133,165 | 18 | 1834 | 30.3 s |
| **early (新)** | 前景 1.06 s + 背景 1 s | **269,856 / 0** | 14 | **117** | **29.3 s** |

- 直したもの: `_prefetch_ngram_span` を `_group_prefill_forward` の**前**で呼ぶ (`MLXTURBO_NGRAM_PREFETCH_AT`、既定 early)。最初の境界だけ自分のぶんを前景で読み切る (`wait=True`、`start < ctx_len` は本家と同じ EOS 埋めで gid ビット一致、合成テストで固定)。`StreamNGram.prefetch(ids, wait=)`、既定 on (`MLXTURBO_NGRAM_PREFETCH`、`=0` で off)、行キャッシュは prefill の開始で世代を捨てる (上限 1 prefill ぶん、17k で約 70 MB。捨てないと辞書 639 MB + バッファ 400 MB まで育つ)。`FASTMLX_NGRAM_NOCACHE=1` は計測専用 (`F_NOCACHE`)。
- **`--knob ngram-prefetch` の汚染を修正**: 条件切り替えごとに `_cache_gen = None`。これまでは A が積んだ 27 万行が残って次の B が全ヒットしていた (回文順でも相殺できない)。**9/3 までのこの knob の数字 (-0.9% 等) はこの汚染込み。**
- in-model (`--knob ngram-prefetch --only long --tokens 64`、A,B,B,A × 3 ケース、prefill_s、`bench/results/ngram-prefetch-*.json`):

| 条件 | A (on, early) | B (off) | 差 | fetch_ms A/B | 出力 |
|---|---|---|---|---|---|
| 17k 冷 (F_NOCACHE) | 27.523 | 28.343 | **-2.9%** (ケース別 -5.0 / -4.6 / +1.0、最後は A の位置 1 の段差) | 751 / 17,016 | 3 ケース一致 |
| 4k 冷 | 6.034 | 6.095 | **-1.0%** | 229 / 4,030 | 一致 |
| 17k 温 | 26.799 | 27.297 | **-1.8%** (-2.7 / -1.4 / -1.4) | 705 / 10,270 | 一致 |
| 4k 温 | 5.793 | 5.819 | **-0.4%** | 185 / 1,697 | 一致 |

- decode は 17k で ms/round +0.2% (ばらつき)、tok/round 完全同一 (先読みは prefill の呼び手からしか呼ばれない)。B 側は移した呼び出し口が no-op なので 9/3 の本番と 1 op も違わない。`bench/test_ngram_stream.py` 23 本 pass、`tools/vendor_fingerprint.py` 一致。
- **判定: 既定に入れる** (代金ゼロ規則: 4 条件どれでも遅くならない、出力同一、メモリは 1 prefill ぶんの行キャッシュ ≤ 70 MB)。冷 17k の集計 -2.9% は線 (-3%) の上だが、揃ったケースは -5%。
- 残り: 50k は未測定 (`_NGRAM_LOOKAHEAD_WIDTH` 10240 なので境界ごとに追う形になる)。最初の境界の前景取得 (17k 1.05 s / 4k 0.54 s) は prefill の先頭に重ねる相手が無く残る (on-demand 4 回 1387 ms よりは速い)。`MLXTURBO_PLE_HOIST` は group prefill 経路を通らないので本番では効いていない。`prefill-anatomy-*` の「PLE / n-gram 6%」は取り直しが要る。

### 2026-09-04 02:55 HC の読み側 (prefill 幅): 段別の冷 micro → elem の prefill 幅拡張は落とす (非ビット一致で tok/round -4.8%)、qmm_wide を HC の down/up に当てる (ビット一致、17k -0.9%) は既定に

- 道具 `tools/hc_prefill_micro.py` (M=2048、97 組 367.5 MB 巡回、段ごとに正の側から 97 段連鎖。`sum_check` 0.96〜0.99 / `pair_check` 1.00〜1.10 で検算)。`bench/results/hc-prefill-micro.json`。

| 段 | 素 us | 下限 us | 比 | 差 |
|---|---|---|---|---|
| pre (rms + (1+w)) | 285.7 | 210 | 0.76 | 67 |
| down (10240→320, 4bit) | 1500 | 1198 | 0.78 | 346 |
| mid (silu(x/hc)) | 18.5〜61 | 6.6 | 0.11〜0.36 | 55 |
| up (320→10240, 4bit) | 1401 | 1198 | 0.86 | 203 |
| **post+inject** | **908** | **341** | **0.37** | **577** |
| combine (書き戻し) | 162 | 236 | 1.45 | -74 |
| 和 / 壁時計 | 4362 / 4391 | 3190 | 0.73 | 1174 |

- 下限は行列積 11.2 TFLOPS、elementwise 400 GB/s。全体 0.73 は 21:45 の「壁の 63〜65%」と桁が合う。取り分の順は post+inject > down > up > pre ≈ mid。
- 候補の冷 micro: (c) qmm_wide (`m64n32k32w2x2r8`) を down/up に: down 0.82 ↔ 1.01 (孤立測定は走行間で振れる)、up 0.86、**ビット一致**。(d) elem 変種 (`hc_elem_pre/mid/post`) を prefill 幅で: pre 0.82、post+inject **0.425**、合成 full 0.87〜0.96、(c)+(d) 0.840。**ビット一致しない** (normed 5.8e-6、mixed 1.9e-5)。
- 数値の切り分け (docs の誤りを訂正): `hyper_connection.py` の「M≥62 で post 段の縮約 (mean) が素と食い違う」は違う。素の `mean(axis=-2)` は M=2048 でも bf16 逐次加算と厳密一致 (fp32 で溜める模型は 33% 外す)。MLX の bf16 sigmoid も bf16 算術で写しは 65533/65536 一致。残る 1 ulp は写した sigmoid か Metal の bf16 積の丸めで、**prefill 幅の elem をビット一致にする道は無い**。
- in-model (`--knob hc-prefill-fast`、head4、depth 2、prefill_s 中央値、`bench/results/hc-prefill-fast-8k.json`、`hc-prefill-fast-c-{4k,17k}.json`):

| | 4k | 8k | 17k | 数値 | tok/round |
|---|---|---|---|---|---|
| A: elem prefill 幅 + wide | — | **-1.6%** (-0.9 / -1.8 / -2.0) | — | 非ビット一致 (case 2 で head が変わる) | **-4.8%** (-8.0 / -4.3 / -2.6、3 本とも負)、decode ms/round +0.5% |
| C: wide だけ | +0.1% (+1.9 / -0.5 / -1.1) | -0.4% (+0.2 / -0.4 / -1.0) | **-0.9%** (-1.5 / -0.9 / -0.3) | ビット一致 (head 同一) | 同一、decode ±0 |

- **A は落とす**: prefill -1.6% の代金が受理率 -4.8% (品質を売って速度を買わない)。実装も削る (負けた変種は knob で残さない)。
- **C は既定に** (`MLXTURBO_HC_QMM_WIDE`、auto = `MLXTURBO_QMM_WIDE` と同じ NAX 判定、`=0` で off): 代金ゼロ (ビット一致、メモリ増無し) で、4k は揺れの中 (+0.1%)、8k -0.4%、17k -0.9% (3 本とも負)。「測った文脈のどれでも遅くならない」を満たす。細長 GEMM (10240→320 / 320→10240) の BM=64 タイル化。
- 残る的: post+inject (下限の 0.37、577 us/層) は elem でしか詰まらず、それがビット一致しない。norm を down の load に / sigmoid+mean を up の store に畳む案は重みレイアウトの並べ替えか写しの増加が要る (代金あり)。HC 読みの取り分はここで一巡。

### 2026-09-04 03:05 fused:1 (MoE decode 幅、gate/up 融合 + down、rmax=1、ソート無し) を既定 auto に: 短 -1.2〜-1.3% × 2 回 / 17k -1.4%、本番重みの参照テストで自前が素より近い、S=1 の Δ KLD +0.00036

- 配線: `fused.enable_moe_decode_fused` (`MLXTURBO_MOE_DECODE_FUSED=auto|1|0`、auto は非 NAX 機で on = `enable_qmm_wide` と同じ判定)、統合ディスパッチ `dispatched()` の先頭分岐、`MLXTURBO_MOE_DECODE_FUSED_MAX_ROWS`=4 (既定の配線は depth 2 で S ≤ 3。`--max-batch-spec` ≥ 2 は rows ≥ 6 で無言で素に落ちる。PoL は S=1 0.929 / S=3 0.979 で S が増えるほど細るので上限は 4)。`--knob moe-dec-fused`。
- in-model (head4、1 プロセス A,B,B,A、burn-in、depth 2 固定、`bench/results/moe-dec-fused-{short,short2,17k}.json`):

| | ms/round A / B | 差 | prompt 別 | tok/round |
|---|---|---|---|---|
| 短 3 本 × 512 (1 回目) | 38.440 / 38.918 | **-1.23%** | -2.46 / +0.18 / -1.40 | 符号ばらけ (テキスト運) |
| 短 (2 回目) | 35.273 / 35.730 | **-1.28%** | -0.60 / -1.55 / -1.68 | 同上 |
| 17k × 512 | 38.558 / 39.110 | **-1.43%** | -1.40 / -1.28 / -1.56 (回文の 6 組全部 A が速い) | -2.6 / +2.0 / -1.1 |

  prefill_s は ±0 (行数ゲートが効いている)。発火 12.7k〜13.9k/走行。先頭 24 トークンは 17k で 3 本とも素と同一、短は 2/3 (case 1 は 19 トークン目で分岐)。
- 品質ゲート (advisor 9/4、decode 幅だけの非ビット一致は初の事例なので 3 条件を規則にした。CLAUDE.md の品質段落):
  1. **本番重み・本番 routing での逆量子化 fp32 参照テスト** (`tools/moe_decode_fused_ref_model.py`、worker の tool job、実プロンプトの末尾 S トークンで全 48 層の (x, indices) をフック): S=1 / S=3 の 96 呼び出しで **自前/素 の距離比 最小 0.168 / 中央 0.388 / 最大 < 1、反転 0**。対ごとの bf16 丸めを 2 回 (gate, up) 外したぶん真値に近い。S=6 は MAX_ROWS=4 で発火せず (比 1.000、道具の NG 表示はこれ)。
  2. **S=1 の Δ KLD** (`quant_eval compare --fusions --step 1`、継続部分を 1 トークンずつ cache 付きで流す。新設): 基準 (off) **0.01796** / agree 0.9657 → on **0.01832** / agree 0.9607、**Δ +0.00036** (受け入れ幅 +0.0005 の中)。prompt 別は 15 本悪化 / 16 本改善で、平均を押し上げたのは ja-fact (+0.0095) 1 本。札別: experts +0.0006、attn +0.0014、ngram -0.0005、code -0.0001。
  3. 読み: 参照 (fp32) には近づくのに teacher (bf16) からは僅かに離れる = **teacher 自身の bf16 丸めの癖に対する一致度**が落ちている。KLD 対 bf16 teacher は「真値への近さ」の物差しではない点に注意。品質の本命は課題の正答率 (長文脈は `tools/longctx_quality.py`)。
- **判定: 既定 auto に入れる** (速度は 3 走行とも負、遅くなる文脈は無い、参照テスト反転 0、Δ KLD は幅の中)。`MLXTURBO_MOE_DECODE_FUSED=0` が逃げ道。
- 収穫: 短文脈 decode の糊の融合レーンは「勝ちは GDN (-2.4%) と fused:1 (-1.3%) の 2 本」で閉じる。どちらも「融合先が素より GPU 時間を使わない」型。
- 追記 (03:10): 上限の幅 S=4 (`--widths 4`) でも参照テストは反転 0 (比 最小 0.205 / 中央 0.342 / 最大 0.689、`bench/results/moe-dec-fused-ref-model-s4.json`)。道具は上限を超える幅を skip と印すよう直した (既定 widths 1,3,4)。
- 追記 (03:12→03:20 に訂正): `tools/verify_prefill_bitident.py` (group=0 と group=4 の 17k prefill のビット一致ゲート) が FAIL した。fused:1 off でも FAIL。正体は **9/3 15:35 の末尾 v2 で既知の性質** (checkpoints=None の経路は group=0 が末尾 2048 を 1 回、group=4 が 2047+1 に割るので量子化行列積の丸めが動く。サーバーの実構成 checkpoints=[] は両方 2047+1 でビット一致) で、道具が最初の checkpoints=None の比較で exit していて、本命の checkpoints=[] まで届いていなかった。道具を「checkpoints=None は tail-in-group が on なら情報表示、判定は checkpoints=[]」に直して再走。fused:1 を prefill 中に止める案は一度当てたが、サーバー経路では両方の写しが同じ +1 ステップを踏むので要らず、複雑さだけ増えるので戻した。
- 追記 (03:24): 判定を直したゲートで **OK: group=0/4 bit-identical (n=17000, cache arrays=110)、checkpoints=[] 込みでも bit-identical** (末尾 checkpoint n-1 / n あり)。今日の既定 (n-gram 先読み on、HC qmm_wide auto、fused:1 auto) でサーバー構成の写し 2 つは一致。checkpoints=None の不一致は既知の情報表示。

### 2026-09-04 03:41 小ベンチ 0903h (今日の既定 3 つ込み、head4、10 分冷却、256 トークン × 6 文脈 × 1 回、`bench/results/self-snapshot-turbo-small-0903h.json`)

| 文脈 | 冷 prefill 0903g → **0903h** | 相手 (mlx-serve 9/3 12:20) | 比 | decode 0903g → 0903h | 相手 | 温 TTFT |
|---|---|---|---|---|---|---|
| 0 | 0.18 → 0.17 s | — | — | 51.4 → 53.6 | 55.7 (短) | 0.15 |
| 4k | 5.95 → **5.74** (-3.5%) | 5.70 | 1.007x (同着) | 51.7 → 58.5 | 55.7 | 0.19 |
| 17k | 30.34 → **26.53** (-12.6%) | 27.8 | **0.95x 勝ち** | 47.5 → 47.4 | 49.0 | 0.19 |
| 25k | 43.47 → **40.75** (-6.3%) | 41.3 | 0.99x | 47.5 → 52.5 | 60.8 | 0.22 |
| 32k | 54.32 → **50.28** (-7.4%) | 51.6 | 0.97x 勝ち | 44.8 → 49.9 | 47.4 | 0.24 |
| 50k | 86.72 → **80.06** (-7.7%) | 82.1 | 0.975x 勝ち | 47.0 → 44.8 | 45.9 | 0.31 |

- **冷 prefill は 4k 同着、17k 以上は全部勝ち** (前回は 1.04〜1.09x 負け)。取り分の内訳は n-gram 先読み (17k 冷 -2.9%、50k は先読みが境界ごとに追う形で未測定だったが -7.7% に効いている)、HC qmm_wide (-0.9%)、それに 0903g の 17k が「新しいプロセスの最初の長い prefill」の段差 (+5.5%) を踏んでいたぶん。
- decode は 1 回 × 256 トークンなので ±10% の tok/step 運が乗る (4k +13% / 25k +10% / 50k -5% はその範囲)。fused:1 の -1.3% はこの粒度では見えない。判定は decode_ab の複数プロンプト平均 (済み)。
- 温 TTFT 0.15〜0.31 s (50k は 0.80 → 0.31)。
- 相手の数字は 9/3 12:20 の走行 (同じ冷却手順)。同時刻の A/B ではないので、決着はフルベンチ (反復 2、同冷却) で。

### 2026-09-04 04:10 K1 arm A (head_dim 256 の flash attention、kv < 8192 の帯): 判定線 0.75x に 3 倍届かず畳む。小 kv の attention は的として閉じる

- 前提の訂正: 「kv < 8192 は QSA が全部を覆うので causal dense」は誤り。疎化を辞退するのは `kv_len <= token_budget` = **kv ≤ 2048 だけ** (`qwen4_exp.py:407`、budget 2048 はトークン数で block_topk は 512)。2048 < kv < 8192 は疎マスク付きなので、カーネルは `keep_block` + per-query tail を読む形で書いた (数値は fp32 累算で fallback より正確)。
- 冷 micro (`tools/flash_attn_d256_micro.py`、K/V 143〜151 MB 巡回、ABBA、burn-in、S=2048、本番 P5 R=256 との比): kv=2048 **2.45x**、4096 **2.26x**、6144 1.85x、8192 1.39x (kv=8192 の 1.39x は既存 `prefill_attn` の「dense の 1.43 倍」と一致)。実効 2〜4 TFLOPS 対 fallback 9〜10。
- 正体は **head_dim 256 のレジスタ圧**: 1 行あたり q 8 + fp32 累算 8 = 16 レジスタ/レーンで、K/V の読みを 1/BQ にする唯一のレバー (BQ) が使えない (BQ 1 / 2 / 4 / 8 = 39.8 / 39.4 / 46.5 / 92.5 ms)。並べ替え 3 種も同じ壁。
- 天井の側からも届かない (`scratchpad/probe_floor2.py`、kv=4096 S=2048): 本番 P5 16.86 ms に対し、MLX の GEMM 2 本だけ (スコアを書いて読む形) が 13.23 ms (12.18 TFLOPS、bf16 GEMM の天井 12.85)。**fallback の「マスク + softmax + スコア実体化」の代金は 3.56 ms (21%) しか無い。**判定線 0.75x = 12.59 ms は GEMM 2 本より速く、融合カーネルは exp と online softmax 込みで天井の 97% を出す必要がある (QSA のタイル飛ばし T=8 込みでも 84%)。
- 見込みの訂正: 4k の attention は 0.257 s / 6.0 s (4.3%) で、**完全な**融合でも -1.2%、現実的な 10 TFLOPS 級で -0.3%。21:45 の「8k -5% / 4k -2%」は疎性ぶんゼロ (01:45) + 融合ぶんも取れない。
- **判定: 畳む。小 kv の attention は的として閉じる。**カーネル (`kernels/flash_attn_d256.py`、未配線) と micro は負の結果の記録として残す。反転条件: head_dim 256 でも 1 threadgroup に 8〜16 行載るレジスタのある機械 (NAX 機) では別の答えになりうる。kv ≥ 8192 は P6 のまま。

### 2026-09-04 04:51 MoE 行ソートの計数ソート化 (P7 第 3 段の最後の的): micro は 4 倍速いが取り分は prefill の 0.04% → 未配線の記録にする

- `kernels/moe_counting_sort.py` (hist / scatter / tables の 3 カーネル + cumsum、simd_shuffle の突き合わせで原子操作なし・決定的)。micro (`tools/moe_sort_micro.py`、48 層を 1 本のコマンドバッファに積む): 素 99.5 → **25.5 us/層** (rows=2048)、131 → 36 (rows=8192)。`idx_s` / 表は argsort 版と完全一致、合成の `SparseMoeBlock` で `max|diff|=0`。
- **BACKLOG の見込み (-0.25 ms/層 = 8k -0.4%) は 1 桁過大だった**: 元の数字は op 1 本ごとに `mx.eval` する測り方で、投入・同期の往復 (165〜270 us) を測っていた (scatter-add 1 本だけで 0.167 ms が出るのが証拠)。実費は 0.10〜0.13 ms/層。本番はグループ幅で層あたり 1 回なので 8k で 4.6 ms = **0.04%**。
- in-model (head4、ABBA × 3、prefill_s): 4k +0.1 / -0.0%、8k -0.7%、17k -0.7% (±1% の揺れの中)。tok/round と head は完全一致 (ビット一致)。どの文脈でも遅くならない。
- **判定: 既定に入れない。**代金ゼロ規則の「代金」に写し (`segment_tables` の式) と vendor のシーム 1 個 (`_MOE_SORT`) が含まれ、測れない取り分に払わない。配線は外してカーネルと micro を記録として残す。P7 第 3 段はこれで閉じる。
- **罠 19**: op 1 本ごとの `mx.eval` で測った us は投入・同期の往復が床になる (この機体で 165〜270 us)。部品の費用は層ぶん (48 層) を 1 本のコマンドバッファに積んで測る。

### 2026-09-04 08:51 27B レーン: GDN 部品を qwen3_5 に契約で当てた (48 層)。品質の物差しは「素の 4bit (融合なし) を参照にした KLD」

- 移植 (`fused.enable_gdn_port`、`_gdn_call_subclass` / `_gdn_norm_subclass` / `_gdn_patch_update`、`runner.py:1475`): 形は Flash-Next と同一 (n_k 16 / n_v 48 / dk=dv 128、hidden だけ 2560 → 5120)。属性名の違いだけを契約 (`_gdn_spec`) で吸収し、インスタンスの `__class__` を動的サブクラスに差し替える (素の forward の写しは持たない)。qwen4_exp は読み飛ばす。
  合成テスト 12 本: decode/verify 幅 S=1,2,3,6 でビット一致、prefill Metal は相対 5e-4〜1.9e-3 (1 ulp 内)、再帰状態はビット一致。実機 27B: 発火 48 層、貪欲 32 トークン × 3 本が素と一致。qwen4_exp 側は fingerprint 完全一致、458 passed。
- 27B の KLD (`bench/quant_eval.py`、参照 = 素の 4bit の logits (bf16 の 27B は手元に無い)、31 prompt、`bench/results/quant-eval/compare-27b-*.json`):

| 構成 | kld_mean | top-1 一致 |
|---|---|---|
| 今日の既定 (GDN Metal + decode 融合 + qmm_wide + 行タイル) | **0.00027** | 0.995 |
| GDN Metal off (他は on) | 0.00000 | 1.000 |
| 全部 off | 0.00000 | 1.000 |

  → 0.00027 は全部 prefill の GDN Metal 再帰 (積和順の差) で、他の部品はビット一致。Flash-Next での GDN Metal (+0.00014、対 bf16) と同じ性質、受け入れ幅 +0.0005 の中。**27B の以後の KLD の基準はこの 3 本** (参照が素の 4bit なので「bf16 との距離」ではなく「素からのずれ」を測っている点に注意)。
- 煙試験 (冷却なし、64 トークン、MTP 写しあり): decode 4k = mlx-serve 43.2 / oMLX 32.8 / MTPLX 28.4 / mlxturbo 27.3 / mlx-lm 20.9。同じ MTP 頭で mlxturbo が投機組の最下位 = 27B の decode 経路 (SpecEngine + staged、spec_flash の段階投入と融合が無い) が最大の的。

### 2026-09-04 09:15 27B: GDN 部品の速度 A/B (`decode_ab_generic`、1 ケース、`bench/results/gdn-*-27b-*-0904.json`)

| 部品 | 文脈 | 差 | 備考 |
|---|---|---|---|
| GDN Metal (prefill 再帰) | 4k | prefill_s **-1.4%** (18.86 → 18.59) | head 一致 |
| 〃 | 17k | prefill_s +0.1% (86.15 → 86.22) | head / tok/round 一致。取り分なし |
| GDN decode 融合 (前処理 + norm) | 短 3 本 × 2 | ms/round -1.1% (+0.2 / -1.4 / -1.8)、ms/tok -0.1% | head 一致だが tok/round が case 0/1 で微差 (+0.9 / -3.3%) = 1 ulp 級のずれが残る |

- 読み: **27B では GDN 部品の取り分がほぼ無い**。理由の候補 2 つ: (1) `spec.py` の S>1 verify (主経路) が `_linear_capture` の写しを通り、移植した部品が当たっていない (advisor の指摘、第 1 段で是正中)、(2) 27B は dense の MLP と attention が重く、GDN の scan の比重が Flash-Next より小さい。
- GDN Metal は品質の代金がある (KLD 0.00027) のに 17k で取り分が無い → **27B では既定 off が筋** (代金ゼロ規則の逆)。決定は第 1 段の着地後に主経路で測り直してから。

### 2026-09-04 09:37 27B: 行タイルと qmm_wide の速度 A/B (`decode_ab_generic`、1 ケース、`bench/results/sdpa-rowtile-27b-*.json` / `qmm-wide-27b-17k-0904.json`)

| 部品 | 4k prefill | 17k prefill | decode | 数値 |
|---|---|---|---|---|
| sdpa 行タイル (16 層) | **-0.7%** | **-0.5%** | ±0 (17k の +1.4% は揺れ、tok/round 同一) | head 一致、KLD 0.0000 (09:51 の compare で GDN Metal off のとき 0) |
| qmm_wide (368 射影、MLP 込み) | +0.4% (08:55) | +0.2% | ±0 | ビット一致 |

- 行タイル: 代金ゼロで両文脈とも負 → **27B でも on のまま** (取り分は小さい)。
- qmm_wide: 27B では取り分なし (揺れの中)。ビット一致で害も無いので契約どおり当てたままにする (Flash-Next の形で勝っていた BM=64 は 27B の形では MLX の素と同等)。
- GDN Metal (prefill): 09:15 の結果 (4k -1.4% / 17k +0.1%) と品質の代金 (KLD 0.00027) を合わせ、**27B (移植した族) では既定 off にする** (第 1 段の着地後に fused.py を触る。Flash-Next は従来どおり on)。
- 27B の prefill の部品はこれで一巡: 取り分は合計 1% 未満。**27B の的は decode 経路** (round 82〜112 ms 対 下限 35 ms)。

### 2026-09-04 10:09 27B decode 経路の第 1 段: 段階投入を S>1 の verify にも (既定 on、短 -1.9% / 4k -0.4%、生成列は完全一致)。capture の写しをモジュール呼び出しにする案は取り分なし (-0.3%) で既定 off。`staged.py` は `_hidden_forward` に畳んで写しを 1 本減らした

- 発火の確認 (合成 qwen3_5、`_fire`): `_linear_capture` (写し) は `gdn_prework` 0 / `rms_norm_gated` 3、モジュール呼び出しは 3 / 3。**素通しは前処理カーネルだけ**で、出力 norm は写しでも当たっていた。両経路の出力・cache はビット一致。
- 変更: `fused.gdn_capture(sink)` (動的サブクラスに状態の取り出し口)、`spec._capture_via_module` (`MLXTURBO_SPEC_CAPTURE_MODULE`、既定 0)、`_hidden_forward` を 1 本化して `staged.py` を削除、`MLXTURBO_SPEC_STAGED_VERIFY` (**既定 on に変更**)、`MLXTURBO_ROUND_TRACE=1` / `decode_ab_generic --round-trace`。
- ゲート: 合成 (`bench/test_spec_capture_module_qwen3_5.py`、S=1,2,4,8 の出力・cache・sink・巻き戻し・段階投入がビット一致) 27 passed、fingerprint 通過、実機 27B (head 全条件一致):

| knob | 短 3 本 × 512 × 2 (ms/round) | 4k (ms/round) | tok/round |
|---|---|---|---|
| `SPEC_STAGED_VERIFY` | **-1.9%** | **-0.4%** | 完全一致 (1 トークンも変わらない) |
| `SPEC_CAPTURE_MODULE` | -0.3% | -0.1% | 幅 5/7/8/9 の round でずれる |

- ずれの正体 (実機、同一 cache から 2 経路): S=1,2,3,4,6 は 48 層とも完全一致、**S=5,7,8,9 だけ `states_all` が 6e-4〜1.6e-5 ずれる** (`QMM_WIDE=off` でも同じ)。融合前処理の q が実機の形 (n_k 16 / key_dim 2048) で数 ulp ずれる幅がある、が最も筋が通る (合成の形では S=8 まで一致)。**S=1 (本番の decode 幅) は完全一致**なので port の判断は揺るがない。原因未特定 (BACKLOG)。
- 事実 (round trace): **round ≒ 43 ms (幅 1 の固定費) + 約 10 ms × draft 本数** (幅 3 = 73、4 = 84、5 = 93、6 = 97、7 = 102、9 = 115)。固定費 43 は素の下限 35 に対して +8。**mlx-serve 4k: tok/round 2.06、draft 2.00 本/round、round 52 ms = 24.4 ms/tok** (depth 6 を持ちながら幅表で w2)。**同じ 2 本でうちは 73〜75 ms → 20 ms/round 負け。**
- 判定: 段階投入は代金ゼロ (値が変わらず、遅くなる文脈なし) → 既定 on。capture のモジュール化は写しが 1 つ減る利点だけで速度は無く、幅 5+ のずれが未解明なので既定 off。**移植した族の prefill Metal 再帰は既定 off に** (`enable_gdn_port`、明示 `MLXTURBO_GDN_METAL=1` だけ。27B で取り分なし + KLD 0.00027)。
- テストの罠: `bench/test_fusions_other_family.py` の module 直下 `mx.set_default_device(mx.cpu)` が同じ pytest プロセスの他ファイルの GPU 判定を収集時に False にし、一致検査が 1 つも走っていなかった → fixture に直した。
- **第 2 段の的**: 1 リンク 10 ms (lm_head 0.64 GB の読み 1.5 ms + MTP 層 0.5 ms の下限に対して 8 ms の同期と糊) と固定費の +8 ms。

### 2026-09-04 10:20 融合もどき (カーネルを書かずにバイトと同期を減らす) の一覧: **Flash-Next と 27B には的が無い、Gemma 4 型 (稼働率 80%) だけ窓がある** (監査エージェント、`scratchpad/agent-pseudo-fusion-audit.md`、`bench/results/pseudo-fusion-0904/`、道具 `tools/decode_glue_probe.py`)

| | Flash-Next 短 | Flash-Next 17k | 27B S=1 | Gemma4-26B S=1 |
|---|---|---|---|---|
| 壁 ms/step | 34.7 | 30.1 | 43.1 | 12.0 |
| dispatch/step | 3183 | 3436 | 1152 | 1459 |
| 稼働率 | 94.8% | 93.8% | **98.7%** | **79.8%** |
| 隙間 ms/step | 1.8 (5%) | 1.9 (6%) | 0.5〜3.6 | **2.4 (20%)** |
| 重み読みの下限 | (MoE) | | 37.0 ms (dense 15.1 GB) = 壁の 92% | |

- 上位 5 項目 (router の top-k gather / softmax / MoE 出力の f32 昇格 copy / x の top-k 複製 / f32 の contiguous 化、split-cb 配分で 5.9 ms/round) が実際に動かすバイトは **4.1 MB/round = 0.010 ms**。配分値の 590 分の 1。27B も同じ (残差 add 128 本 3.8 MB = 0.009 ms、norm 129 本 2.6 MB)。**dtype 統一・layout・重みへの畳み込みで「バイトを減らす」種類の融合には、どの族にも取り分が無い** (9/3 22:00 の対照 -192 本 ±0.0% と整合)。値段が付いているのは起動と直列の側だけ。
- **27B は既に帯域の壁の 92%、稼働率 98.7%。糊の予算 5 ms。的は round の形 (B1 / B2) のみ。**「27B にも生きる」の見立ては外れ。
- **Gemma 4 だけ**稼働率 79.8%、隙間 2.4 ms (20%)、RMSNorm が 331 本/step (30 層 × 11)。本数がタダでない唯一の帯で、norm を層あたり 3〜4 本に畳めれば 0.5〜1.5 ms (4〜12%) の見込み。ただし物差しが 1 step ごとの `mx.eval` なので本番の生成ループより隙間が広く出ている疑いあり (generate 経路で取り直しが先)。
- 族 × 糊の種類 (コード読み、mlx_lm の全族): MoE 族は `switch_layers.py` の並べ替え一式 (argsort ×2、x 複製、unsort、`arange`) を全族が共有 (Flash-Next だけ自前で畳み済み) → **35B-A3B には 1〜9 がそのまま当てはまる**。DeepSeek / Kimi / GLM-DSA の MLA は `pe_scores` (128 head × kv × f32、4 MB/層) を sdpa の mask に渡していて、**唯一バイトが本当に大きい項目** (机上)。`scores * routed_scaling_factor` (deepseek_v3 系 8 族) と Gemma 4 の 4 種のスケールは重みに畳める (読み込み時)。minimax は自前 norm (3〜4 本)。
- **判定: 融合もどきレーンは Flash-Next / 27B では閉じる。**Gemma 4 は Gemma レーンの中で norm の本数削減として扱う。MLA 族は将来の的。
- 副産物 (別件、要即対応): `fused._moe_fold_block` の combine 分岐で `inv` が未束縛 → `MOE_DOWN_EPI` 既定 on × 行数 ≥ 64 = **Flash-Next の実 prefill 全部が落ちる**回帰。13f0d21 (族の回帰修正) で計数ソートの hunk を除外したときに `if inv is None:` の 5 行だけ紛れ込んだ (キーワードで hunk を選別した副作用)。10:20 に直す。
- 追記 (10:38): 回帰修正 (d8c2c69) 後の Flash-Next 4k prefill 煙試験 OK (`decode_ab --knob null --only long --ctx 4000`、prefill 5.6〜6.0 s、0903h の 5.74 と同じ水準)。

### 2026-09-04 10:41 27B decode 経路の第 2 段: draft を「引いてから捨てる」のをやめて同期ゼロで引く (NOSYNC) = **ms/tok 短 -12.5% / 4k -12.1%** → 既定 on。次 round の先行投入 (PREFETCH) は取り分ゼロ (+0.2 / +1.9%) → 畳む

- 変更 (`mlxturbo/spec.py`): `_draft_chain` (argmax を `.item()` せず配列のまま次段へ、最終段以外は `async_eval`、語彙長の softmax / エントロピーは組まない)、`_plan_depth` (`_gate_depth` の EMA / 事前値だけで**引く前に**本数を決める。観測を積んだ入力では `_gate_depth` と厳密一致、単体テスト 4 件)。
- 実機 27B (`decode_ab_generic`、`bench/results/spec-nosync-27b-{short,4k}-0904.json`):

| knob | 文脈 | ms/tok | ms/round | tok/round | draft 本数/round | 生成列の分岐位置 |
|---|---|---|---|---|---|---|
| NOSYNC 1 / 0 | 短 3 本 × 512 × 2 | **27.67 / 31.63 (-12.5%)** | 80.4 / 94.6 (-14.9%) | 2.94 / 3.05 (-3.6%) | 3.2〜4.1 / 3.4〜4.9 | 27 / 14 / 151 トークン目 |
| NOSYNC 1 / 0 | 4k × 256 | **32.37 / 36.83 (-12.1%)** | 82.1 / 99.3 (-17.3%) | 2.54 / 2.70 (-5.9%) | 3.30 / 3.74 | 52 |
| PREFETCH 1 / 0 | 短 / 4k | +0.2% / +1.9% | 同 | ±0 | 同一 | 完全一致 |

- **帰属の訂正**: 取り分の正体は同期の除去ではなく「引いてから捨てる」をやめたこと。素は毎 round `cap_base = 8` 本を無条件に引き、事後に 3.3 本まで捨てていた → 4.8 本 × 3.2 ms ≒ 15.4 ms/round が消えた (実測差 14.1 / 17.2 を挟む)。同期そのものは温で 0.1 ms/リンク。「1 リンク 10 ms」は draft ではなく **verify の 1 行の値段** (幅を 1 増やす費用 11.6 ms は NOSYNC 後も残る)。
- 固定費の内訳 (`scratchpad/b2_fixed_cost_micro.py`): 幅 1 の round 45.1 ms = trunk forward **42.0** + lm_head **4.6** + 糊 ≈ 0。帯域下限との差は trunk の 64 層ループに +8.6、**lm_head に +3.0 (0.64 GB を 4.6 ms = 139 GB/s、下限の 2.9 倍遅い)**。MTP のリンク 1 本 3.2 ms (append 1.24 + lm_head 1.88)。「固定費 +8 ms」は同期でも Python でもない。
- 数値: 貪欲でも生成列が変わる (検証幅が変わると 4bit `quantized_matmul` の丸めが変わる = prefill チャンクの注記と同じ性質)。ビット一致では受けられないが、どちらも同じモデルの正当な貪欲出力。tok/round は 3〜6% 落ちる (事前の深さ決めが事後より粗い) が、ms/tok は -12%。**判定: 既定 on** (`MLXTURBO_SPEC_DRAFT_NOSYNC=0` が逃げ道)。PREFETCH はコードごと削除。
- **第 3 段の的**: (1) lm_head 4.6 ms (139 GB/s、draft のリンクにも効く)、(2) 幅を 1 増やす 11.6 ms の中身 (GDN の行ごと逐次 48 層が疑い)、(3) trunk の層ループの +8.6 ms、(4) tok/round の -3〜6% を取り返す深さ決め (`DepthController` の E(m)/T(m) 最大化、費用モデルの代金あり)。

### 2026-09-04 10:55 27B: qmm_wide は SpecEngine 経路で一度も発火していなかった (scout)。lm_head 139 GB/s は環境 (直前の重みトラフィック) が疑い

- `SpecEngine.__init__` (`spec.py:314`) の `enable_quantized_dispatch(self.text, active=False)` が全 `nn.QuantizedLinear` の `__class__` を `DispatchedQuantizedLinear` に上書きし、自前の `__call__` が `enable_qmm_wide` の差し替え (`_qmm_wide_dispatch`) をシャドーする。`dispatch_scope()` の外では常に STOCK。**起動ログの「qmm_wide 有効 (368 射影)」は印を付けただけで、27B の A/B が ±0 だったのはこれ** (今朝の移植の取り分の評価は取り直し)。Flash-Next は `enable_quantized_dispatch` を通らないので無関係。修正をエージェントに (`scratchpad/agent-27b-dispatch-fix.md`)。
- lm_head (M=1、0.64 GB、4.6 ms = 139 GB/s): 自前カーネルは通らず stock の qmv。Flash-Next でも常駐時の孤立計測は 109 GB/s (KERNEL-BRIEF-MOE-GDN.md 2026-08-28、原因は直前の重みトラフィック、N の分割やカーネル選択では直らない)。CATCHUP 21:10 の「lm_head 0.83 ms」はバイト按分の推定値で孤立計測ではない。27B での切り分け (非常駐 / 常駐 / decode 直後) は同じエージェントに。
- 追記 (11:02): 段階投入の粒度 (`MLXTURBO_STAGE_EVERY` 2 / 4 / 8 / 16) の掃引 (27B、`bench/results/stage-every-27b-*.json`): 短 ms/tok ±0 / -0.0 / **-1.0** / -0.0%、4k +0.3 / +0.8 / +1.9%。head と tok/round は全て一致。**取り分なし、既定 2 のまま** (CB 168 本/round は費用の正体ではない)。

### 2026-09-04 11:43 Flash-Next の「飛ばす / 積む」: 的なし → 畳む (`bench/results/gdn-state-out-0904.json`)

- 唯一実在した「使わない仕事」= GDN の `state_out` の二度書き (113 MB/forward) を消しても短 ±0.0% / 17k +0.1% (ビット一致、peak メモリ -12.6 MB で配線は効いている)。**バイトを消しても壁時計が動かない直接の例** (隣の行列積と重なって隠れる)。代金 (カーネル変種 4 本) があるので畳んだ (コードは戻した)。
- 「積む」に新しい的は無し (router 18→4 は +0.05%、HC は既に 3 本/層、q/k norm 統合は期待値 0、dtype cast 260 本は 0.010 ms)。IDEAS の D6 (indexer 228 us/層) は長文脈で疎化が働くときの値で短文脈には掛からない。

### 2026-09-04 12:05 天井の監査 (fast_qmm 型の的の一覧、`tools/ceiling_audit_micro.py`、`bench/results/ceiling-{qmm,mid,small}-0904.json`、`scratchpad/agent-ceiling-audit.md`)

達成率 (帯域 410 GB/s / 11.2 TFLOPS 比) × 占有で順位:

| # | 的 | 幅 | 達成率 | 占有 | 天井との差 | 正体 |
|---|---|---|---|---|---|---|
| 1 | **27B sdpa (d256、16 層、kv 17k)** | S=4 | **24%** | 11.5 ms/round | 7.6 ms | KV を行ごとに読み直している |
| 2 | 同 (崖の向こう) | S≥6 | 6.5% | 43 ms/round | 39 ms | S×gqa > 32 の崖。**27B には幅分割のシームが無い** (`runner.py:1491` は無条件に「有効」と print = 誤り) |
| 3 | prefill の MoE GEMM (素の gather_qmm、40 行/専門家) | 20480 行 | 52〜62% | 1568 ms/chunk | 210 ms (P3 後) | BM=16 タイル。P3 の 1.5x で 79〜82% に |
| 4 | HC の束 (down / up / inject / rms) × 97 | S=1 | 4〜72% | 1.96 ms/round | 1.0 ms | 依存の直列に乗った 4 本 × 2.2〜3.5 us の固定費 |
| 5 | `quantized_matmul` の谷 M=4〜24 (全 25 形) | M=8〜12 | **17〜35%** | (本日の経路では S ≤ 8) | | qmv は M=2 までしか行を共有せず、M ≥ 13 は `qmm_t_splitk`。**M=12〜31 では行を水増しした方が速い** (M=12 → 32 で 1.5〜2.2 倍) = バッチ × 投機 (B=4 × S=3 = 12) の帯 |
| 6 | Flash-Next sdpa (12 層、kv 17k) | S=1 | 46% | 2.3 ms/round | 1.2 ms | gqa 12 で threadgroup が埋まらない (K2c が既に置き換え済み) |
| 7 | prefill QSA の `argpartition` (nb=4352) × 12 | S=2048 | 3.7% | 31 ms/chunk | 30 ms | top-512 に全ソート |
| 8 | GDN の in_proj_b / _a (N=48) × 72 | 全幅 | 5〜65% | 0.26 ms/round | 0.25 ms | N=48 では GPU が埋まらない (52 GB/s) |

- **狭い形が M=1 で天井を外すのは起動の固定費 (2.2 us) で、カーネルは悪くない**: 固定費を引くと 25 形中 23 が 394〜472 GB/s。elementwise も同じ 2.2〜3.8 us の床。直す手は融合 (依存の直列に乗ったものだけ) で GEMM の書き直しではない。
- **訂正 2 つ**: (1) CATCHUP 21:45 の「HC 読みは壁の 63〜65%」は細長 GEMM のせいではない (`hc_down` / `hc_up` は M=2048 で 100.9 / 101.4%)。差 100 ms/chunk は周りの elementwise。(2) K2 の stub の「13%」は QSA の einsum + argpartition ではない (17k decode で 0.28 ms = 1.1%)。残りは糊。
- 死んだ的 (確認): prefill 幅の dense 射影 (98〜104%)、27B decode S=1 の qmm (96〜103%)、語彙 248k の softmax / argmax / logsumexp (占有 < 0.1 ms)、router の argpartition / argsort (起動の床だけ)。
- **次の手 (出した)**: 27B の sdpa 幅分割 (S×gqa ≤ 32 に割る、Flash-Next のシームと同じ手、ビット一致のはず、17k で -3% が線)。小 M qmv (別途 PoL 中) は M ≤ 8 を第 1 目標にし、M=12〜31 の谷はバッチ × 投機のレーンで。QSA の argpartition (prefill 1%) と HC の束 (1 ms) は後。
- 追記 (12:08、監査の最終版、mask あり / 冷の gather_qmm 込み): 27B sdpa S=4 は **mask ありで達成率 18.3%、15.3 ms/round、差 11 ms** (S≥6 で 41 ms)。mask は崖を動かさない (1.0〜1.4 倍)。**MoE gather の谷**: 専門家あたり 2.5〜10 行 (1280〜5120 行のチャンク) で 30〜50% (1 行/専門家 64〜96%、40 行 62% の間が最悪) = prefill の端数チャンクと 35B-A3B 級の幅の的。`bench/results/ceiling-{sdpa-mask,gqmm-cold}-0904.json`。

### 2026-09-04 12:15 MLX の更新の試験 (`scratchpad/agent-mlx-upgrade.md`): 上げ先が無い (PyPI 最新 = 本番と同じ 0.32.2 / mlx-lm 0.31.3)。main (0.32.3.dev) を source build して比較 → 非 NAX では**差なし** (短 -0.05%、生成列 4 走行一致、指紋一致、テスト全緑)。**NAX 機で踏む正しさの修正 (#3922) がある**

- 0.32.3 の中身で効くもの: **#3922 sorted `gather_qmm` の NAX カーネルが 32K 行超で行数を short に落として負に折り返し、出力行が未書き込みになる**。本番既定 `MLXTURBO_PREFILL_GROUP=4` × 2048 × top_k 10 = **81,920 行 > 32,768** で、NAX 機では `MLXTURBO_MOE_GEMM=auto` が自前 GEMM を off にするので素の sorted gather_qmm (当のカーネル) が走る → **NAX 機 (本番の対象機) では 0.32.2 のままだと prefill の MoE が壊れる疑い** (M3 Max では再現も反証もできない)。BACKLOG に。
- #4416 (hd256 + array mask の prefill を NAX 融合 attention へ、`MLXTURBO_SDPA_ROWTILE` と的が重なる、NAX 限定)、#4380 (sdpa_vector の GQA 12/16 化は head_dim 128 限定 = うちには当たらない)。
- 速度: Flash-Next 短 512 (プロセスを分けた A,B,B,A) 16.08 対 16.08 ms/tok、ms/round 34.84 対 34.85 → **差なし**。17k は depth 適応の controller が混んだ機体で別の深さを選んで比較にならず打ち切り (`--depth 2` 固定が要る)。ユーザーの判断で速度 A/B はスキップ。
- 判定: いま上げるものは無い。0.32.3 が出たら上げる価値は速度ではなく NAX の正しさ。mlx-lm の次版 0.32.0 は `_mlx_compat` の上限外で import 時に落ちる (設計どおり): そのときの作業は `spec_flash._staged_forward` の写しの PipelineMixin 追随、`layers`→`pipeline_layers`、`BatchKVCache.state` の 3 タプル化、上流の packed gated delta kernel (Dk=128) と自前 GDN の取り直し。

### 2026-09-04 12:16 27B: qmm_wide のシャドーを直して発火 (prefill **4k -7.0% / 17k -5.6%**、生成列一致)。lm_head の 139 GB/s は帰属のアーチファクト (単体は天井の 97〜99%)

- 修正 (`kernels/dispatch.py:290-301`): `DispatchedQuantizedLinear.__call__` の先頭で、非活性 (`dispatch_scope()` の外) なら `super().__call__(x)` に委ねる → 基底の `enable_qmm_wide` の差し替えが効く。活性の枝 (経路表) は不変。修正前の実機: 印 368 本、発火 **0**。修正後: 4k 736 / 17k 2944。
- A/B (`decode_ab_generic --knob MLXTURBO_QMM_WIDE=auto,off`、`bench/results/qmm-wide-27b-fix-{4k,17k}.json`): prefill_s **17.28 対 18.58 (-7.0%)**、**84.09 対 89.03 (-5.6%、4 本とも)**、tok/round ±0、生成列 64 トークン全長一致 (ビット一致の設計どおり)。修正前は +0.4% (差なし)。Flash-Next は指紋一致で無影響。テスト 2 本追加 (修正を外すと落ちる)。
- lm_head の切り分け (`tools/probe_lm_head_bw.py` 一般化、715 MB): 非常駐 335 GB/s / 常駐アイドル 344 / decode 直後 341 = **その状態の天井の 97〜99%**。Flash-Next の「常駐時 109 GB/s」は 27B では再現しない (常駐量 98 対 15〜17 GB の差)。第 2 段の「lm_head 4.6 ms = 139 GB/s」は単体 2.1 ms なので、差 2.5 ms は trace の per-kernel の帰属 (CB 按分) の側。**27B の lm_head は的ではない。**

### 2026-09-04 12:28 層単位 `mx.compile` の PoL (Flash-Next、`tools/compile_layer_poc.py`、`scratchpad/agent-fn-compile-poc.md`): 層まるごとは原理的に不可、**MoE ブロックだけ可で -1.2% (ビット一致) → 既定 on に配線する**

| 単位 | 可否 | 理由 |
|---|---|---|
| HC の read / write、router | 可 | 純関数 (取り分 ±0) |
| **MoE ブロック (`SparseMoeBlock.__call__`)** | **可** | 純関数。dispatch -7.7%、短 ms/round **-1.2%** (3 本とも)、17k **-1.3%**、tok/round 完全一致 |
| GDN / attention ブロック、層まるごと、forward まるごと、`_staged_forward` | **不可** | compile は再トレースしないので cache の副作用 (`cache[0]=` 等) がトレース時の 1 回で固定され 2 回目に古い。PLE の n-gram は host 同期。`inputs=` / `outputs=` で状態を宣言しても vendor の GDN には効かない。`shapeless=True` は全滅 (Slice / Split / reshape) |

- 固定費: 1 グラフ 3〜6 ms のトレース × 48 層 × 形。短文脈 prefill の初回 +0.6 s (98 グラフ)、本番は S=1〜4 で ≈ 240 グラフ ≈ 1.5 s → **起動時の warm-up で払う** (TTFT に乗せない)。
- 判定: 判定線 (-3%) には届かないが代金ゼロ (ビット一致、遅くなる文脈なし) → 既定 on に配線 (エージェント、`scratchpad/agent-fn-moe-compile.md`)。レーン 9 (層単位 compile) はこれで最終回答 = 「層は不可、MoE ブロックだけ」。

### 2026-09-04 12:31 小 M (2〜8 行) の量子化行列積の自前カーネル `kernels/qmv_small_m.py` (`scratchpad/agent-27b-verify-width.md`): **各行が MLX の qmv (M=1) とビット一致** = 「検証幅で丸めが変わらない保証」を達成。速度は in-model で小勝ち (ms/tok -1.6 / -2.3%)、冷 micro では負け

- 超線形の出所 (実機 27B の部品 × S、`tools/verify_width_cost_27b.py`、`bench/results/width-cost-27b-0904.json`): S=1→4 の +21.3 ms のうち **MLP の量子化行列積だけで +18.6 ms**。行列積でない部品 (conv / rms / layernorm) は S に平坦。実効帯域 M=1 372 / M=2 312 / **M=3 218 / M=4 216 GB/s**。重みは M ≤ 5 なら 1 回しか読まないので、落ちているのは **ALU 側** (重み 4 バイトあたり逆量子化 ≈ 32 演算の固定 + M × 8 FMA)。GDN の capture の `states_all` は S=4 で +1.8 ms (第 2 の的)。
- カーネル: mlx の `qmv_fast_impl` (bits 4、values_per_thread 16、block 512、simd_sum) の構造を写し、M 行のアキュムレータを足したもの。**`qmv_small_m(x)[v] == quantized_matmul(x[v:v+1])` が M=1..8 × bf16 / fp16 で全要素ビット一致** (要は `load_vector` の和を T (bf16) のまま足してから float に上げること。float で足すと bias × sum が 1 ulp ずれる)。適格は mlx が qmv_fast を選ぶ条件と同じ (K%512、N%8、gs%16、bits 4、2 次元、M ≤ 8)。配線は `kernels/dispatch.py` の `SMALLM` 経路 + `MLXTURBO_SMALL_M_ROUTE` (**既定 off**、M=2..5・N ≥ 1024・K ≥ 1024)。テスト 37 件 (`bench/test_qmm_smallm.py`)。
- in-model (27B、`bench/results/ab-smallm-kernel-short-0904.json`): 短 case 0 **ms/tok 32.88 対 33.40 (-1.6%)**、case 1 **28.39 対 29.07 (-2.3%)**。ms/round は tok/round が動く (素は幅ごとに丸めが違い、自前は変わらない) ので比較にならず、ms/tok で見る。残り (case 2、4k) は走行中。
- 冷 micro (`tools/qmv_small_m_micro.py`): 1 forward 合計 M=4 で stock 46.5 対 自前 47.9 = **負け** (lm_head だけ 0.78〜0.87x)。**micro と in-model が逆** (孤立では M カーブ 1.14〜1.32x、実機 1.72x)。仮説は forward 全体で多数の qmv が重なったときの ALU / 発行率の飽和。**この的はカーネル単体の micro では判定できない。**
- 判定は残りの A/B (複数プロンプト × 512 の ms/tok 平均) の後。「同じ挙動の保証」は取れているので、遅くならなければ既定 on の候補 (代金: bits 8 と gs≠64 は委譲、Flash-Next には未配線)。次の手: M=2..5 に `mma` (8 行に水増し) / `nocap` の経路も同じ A/B で比べる。

### 2026-09-04 13:05 小 M (M=2..5) の量子化行列積の 3 経路を 27B in-model で比較 → `small_m` を既定 auto に (`MLXTURBO_SMALL_M_ROUTE`)

512 トークン × 短 3 本 × 2 回文 + 4k `--prefill-once`、1 プロセス内で 4 変種を交互に (`bench/results/ab-smallm-3way-{short,4k}-0904.json`)。fp32 参照距離は `bench/results/qmv-small-m-accuracy-0904.json` (K=5120/17408/6144/2560 × M=1..8)。

| 経路 | short ms/tok (3 本平均) | 対素 | 4k ms/tok | 対素 | tok/round (short) | fp32 距離 (RMS 相対) |
|---|---|---|---|---|---|---|
| small_m | 29.53 | 0.981x | 33.04 | 0.954x | 2.988 | 1.65e-3 (素と同じ) |
| nocap | 29.54 | 0.981x | 34.90 | 1.008x | 3.097 | 1.65e-3 (素と同じ) |
| 素 | 30.10 | 1.000x | 34.63 | 1.000x | 2.936 | 1.65e-3 |
| mma | 31.05 | 1.031x | 36.43 | 1.052x | 3.049 | 2.0〜2.9e-3 (素の 1.3〜1.7 倍) |

- mma (`fast_qmm`) は速さでも品質でも負け。split-K の部分和が bf16 を経由するので fp32 から遠い。simdgroup matrix で ALU の壁を越える案はこの帯では否。
- small_m と nocap は fp32 距離が素と区別できない (品質の代金ゼロ)。短文脈は同着 (-1.9%)、4k で small_m だけ -4.6% (理由は未特定)。
- 方針 (ユーザー 12:40) どおり順位はビット一致では付けていない。small_m が各行で qmv と一致するのは計測が楽になる性質として記録。
- 既定 auto = 非 NAX 機で small_m (自前カーネルは NAX 機で auto=off の方針と同じ判定)。`=off` で素。Flash-Next は `dispatch_scope` を通らないので未接続。
- `_load_kernels()` は 2 タプルに戻し、小 M の実体は `_load_small_m()` に分離 (`bench/test_dispatch_static.py` の契約テストはそのまま通る。偽 mx のテストは fixture で経路を off に固定)。テスト: test_qmm_smallm 37 + dispatch 系 = 41 passed、エージェントの広い走行 503 passed。
- 既定の経路が本番の engine で発火することの確認 (`--knob MLXTURBO_SMALL_M_ROUTE=auto,off`、短 3 本 × 256) は GPU の順番待ち → 結果は下に追記。

### 2026-09-04 13:08 Gemma 4 (26B) の 5 者と 27B の「投機なし / 推奨設定」の行 (相手 4 エンジン、`docs/research/COMPARE-QUEUE.md`、`scratchpad/agent-gemma4-smoke.md`)

Gemma 4 26B (MoE、A4B) の decode tok/s (`--ctxs 0,4000 --tokens 64 --reps 1`、冷却無し):

| エンジン | draft | 冷 TTFT (0) | dec (0) | 冷 TTFT (4k) | 温 TTFT (4k) | dec (4k) |
|---|---|---|---|---|---|---|
| mlx-lm | なし | 0.22 | 86.8 | 3.10 | 0.37 | 75.1 |
| oMLX | なし | 0.32 | 122.2 | 0.92 | 0.93 | 107.0 |
| oMLX | あり | 0.34 | 84.4 | 3.51 | 3.48 | 76.3 |
| mlx-serve | なし | 0.06 | 109.2 | 2.56 | 0.18 | 95.0 |
| mlx-serve | あり | 0.14 | 61.6 | 2.68 | 0.27 | 66.7 |
| MTPLX | — | 起動しない (dense 31B しか受けない) | | | | |

- **公式 draft (`gemma4_assistant`) は 26B では 2 者とも遅くなる** (oMLX 122 → 84、mlx-serve 109 → 62)。MoE の検証フォワードの専門家ルーティングの代金が draft の利得を上回る。mlx-serve は起動時に警告し、受理率が落ちるとランタイムが drafter を切る (`avg_per_round 3.00 → 0.60`)。
  → **mlxturbo に assistant drafter のエンジンを作る動機は 26B には無い。**dense 31B なら別 (MTPLX は 31B だけ受ける。bundle `~/models/mtplx-gemma4-31b` は作ってある)。Gemma レーンの順は「norm 331 本の削減 → KV 量子化 第 1 段 → TurboQuant」に詰める (drafter は 31B を測ってから)。
- mlxturbo と mlx-lm は draft なしのみ (`gemma4_assistant` を読めない)。mlxturbo の行 (Gemma 4 draft なし、27B 投機なし / あり) は 1・3 の着地後に `scratchpad/gemma4_smoke_runs3.sh` で取る。
- 27B の相手の推奨設定: **MTPLX の 28.4 はうちの設定のせいではない** (quickstart 既定で 28.2)。`--profile turbo` で 4k 31.5 (+12%)、depth の上限は 3。**oMLX の depth は上げられない** (block_size 3 → 6 で受理は変わらず検証幅だけ倍、33.7 → 16.0)。
  **oMLX の 4k 温 TTFT ≈ 冷 は接頭辞キャッシュのブロックが 4096 トークン**のため (3821 トークンのプロンプトでは完成ブロック 0 個。hot cache / paged-ssd を有効にしても 18.67 s)。
- `--no-mtp` は「投機なし」ではない (mlx-serve は PLD が既定 on、mlxturbo も lookup が残る)。投機ゼロの行は `--no-pld` も付ける。harness に `--extra-body` を足した (mlx-serve の `enable_drafter` のようにリクエスト側にスイッチがあるエンジン用)。
- 追記 (13:11): 既定 auto の発火確認 (`--knob MLXTURBO_SMALL_M_ROUTE=auto,off`、短 3 本 × 256、`ab-smallm-auto-short-0904.json`): ms/tok **-1.6%**、ms/round -2.0%、tok/round +0.2% (3 経路の A/B の -1.9% と同じ帯)。
  生成列は 3 本とも途中で分岐 (位置 37 / 14 / 151)。M=2..5 の行が「幅 1 の丸め」になるぶんの丸め差で、方針 (12:40) どおり不採用の理由にしない。

### 2026-09-04 13:15 27B の sdpa 幅分割 (`MLXTURBO_SDPA_SPLIT_GENERIC`、`scratchpad/agent-27b-sdpa-split.md`) → 既定 auto に

27B (qwen3_5) には幅分割のシームが無かった (runner のログは族に依らず「有効」と刷っていた)。行タイルと同じ名前空間差し替え口に汎用版を足し、1 < S ≤ 16 で S×gqa > 32 のとき K/V をクエリ幅で切って MLX の vector カーネルに戻す。

| 文脈 | ms/tok | ms/round | tok/round | 発火率 | fp32 距離 (分割 / 素) |
|---|---|---|---|---|---|
| 短 3 本 × 256 | -1.6% | +2.0% | +4.0% | 発火あり | 0.53 |
| 4k × 256 | +0.4% (雑音、256 トークン完全一致) | +0.4% | ±0 | 0% | 1.000 (発火せず) |
| 17k × 256 | -1.2% | -0.2% | +1.1% | 7.1% (98 round 中 7) | 0.53 |
| 17k、幅 6 の round だけ | — | -17% (109.5 対 132.0) | — | — | 0.536 |
| 冷 micro kv 17408、S=6 / 8 | — | -61% / -55% (1 層 2704→1041 / 3000→1354 us) | — | — | 0.535 |

- 判定線 (17k -3%) には届かないが、**fp32 参照に素より近く (40 draw で悪化 0)、遅くなる文脈が無い**ので代金ゼロの規則で既定 auto (非 NAX 機で on)。壁の向こうの MLX fallback は中間を bf16 で回し、壁の手前の vector カーネルは走査和を fp32 で持つ。分割は精度の良い方に戻している。
- head の分岐 (位置 60 / 14) はこの丸め由来。方針 (12:40) どおり不採用の理由にしない。
- 残った不明点: 幅 9/10 の round が 1 走行に 2 回しか出ず、micro の -55% は in-model で未確認。50k は未測 (kv が伸びるほど 1 回の取り分は大きい)。速度の A/B では `MLXTURBO_SDPA_SPLIT_GENERIC_TRACE=1` を立てないこと (発火した側だけ 16 回/round の bump が乗る)。
- 副産物: qwen4_exp の既存シームは bool マスク実体化の変種で、K/V を切る変種が冷 micro で 13〜18% 速い (BACKLOG)。
- 検査: `bench/test_sdpa_split_generic.py` 9 本、`vendor_fingerprint` 全一致。commit は MoE compile (同じ fused.py / runner.py を編集中) の着地と一緒に。

### 2026-09-04 13:22 mlxturbo の煙試験 3 行 (fdc06e1 + sdpa 分割 auto、`scratchpad/gemma4_smoke_runs3.sh`、`--ctxs 0,4000 --tokens 64 --reps 1`、冷却無し)

| 行 | 冷 TTFT (0) | dec (0) | 冷 TTFT (4k) | 温 TTFT (4k) | dec (4k) | 08:16 の同じ行 |
|---|---|---|---|---|---|---|
| 27B MTP あり | 0.23 | **33.8** | 16.10 | 0.27 | **32.1** | 28.1 / 17.52 / 0.27 / 27.3 |
| 27B lookup のみ (`--no-mtp`、投機ゼロではない) | 0.22 | 23.4 | 16.11 | 0.26 | 22.3 | — |
| Gemma 4 26B draft なし | 0.18 | 104.3 | 2.69 | **2.76** | 93.9 | — |

- 27B は移植 (段階投入 S>1、NOSYNC、qmm_wide のシャドー修正、小 M、sdpa 分割) で decode +20% (27.3 → 32.1)。相手: mlx-serve 43.2、oMLX 32.8、MTPLX 28.4 (turbo 31.5)、mlx-lm 20.9。**oMLX と同着圏、mlx-serve にはまだ 1.35 倍の差。**冷 TTFT 4k は 16.1 (mlx-serve 15.9、oMLX 19.0)。
- Gemma 4 は draft なしで mlx-lm (86.8 / 75.1) より速く、mlx-serve (109 / 95) に近い。oMLX (122 / 107) が最速。
- **Gemma 4 の温 TTFT が冷と同じ (2.76 対 2.69)。**FallbackRunner の経路で接頭辞キャッシュが効いていない (27B の spec 経路は 0.27)。mlx-lm ですら 0.37。→ BACKLOG (Gemma レーンの最初の直し)。

### 2026-09-04 13:25 MoE ブロックの `mx.compile` を本番に配線 → 既定 auto (`MLXTURBO_MOE_COMPILE`、`scratchpad/agent-fn-moe-compile.md`)

head4、depth 2 固定、回文順、burn-in 済み。A = compile / B = 素 (`bench/results/moe-compile-{short,17k}-0904.json`)。

| | ms/round A | B | 差 | tok/round | head |
|---|---|---|---|---|---|
| 短 3 本 × 512 | 34.485 | 34.880 | -1.1% | 2.186 / 2.186 | 3 本とも出力一致 |
| 17k 3 本 × 512 (`--prefill-once`) | 37.878 | 38.120 | -0.6% | 1.961 / 1.961 | 3 本とも出力一致 |

- PoL からの逸脱 1 つ: 包む幅に上限 (`MLXTURBO_MOE_COMPILE_MAX_ROWS`=16)。prefill 幅は取り分が無く (17k ±0、短 +26%)、端数チャンクの長さが要求ごとに変わるので Compiled が際限なく溜まる。decode 幅の取り分は全部残る。
- 起動時の warm-up は 48 層 × S=1..4 の 192 グラフで 0.22 s。1 要求目の TTFT: warm 0.304 / nowarm 0.320 / off 0.345 s (2 要求目は 0.237 で同じ)。`発火 {'moe_decode_fused': 192}` = 192 個すべてに fused:1 が入っている。
- 17k が PoL の -1.3% に対して -0.6% (符号は 6 本とも負)。`--prefill-once` の有無か熱。バッチ検証 (B>1) と MTP の draft ブロックは warm-up に入れていない (初回に 1 回、≈0.05 s / 5 ms)。
- **A/B を投げた後は `mlxturbo/*.py` を触らない** (worker の code fingerprint が変わって読み直し 290 s。今日 2 回踏んだ)。
- 検査: pytest 4 ファイル 436 passed、`vendor_fingerprint` 全一致。

### 2026-09-04 13:30 27B の投機ゼロの行 (`MLXTURBO_RUNNER=fallback`、99ba892、`smoke-27b-mlxturbo-nospec-forced-0904.json`): 素の効率は相手と同着、差は投機の取り分

| 設定 | dec (0) | dec (4k) | 対 mlx-lm |
|---|---|---|---|
| mlxturbo 投機ゼロ | 23.8 | 22.9 | 1.09x |
| mlxturbo lookup のみ (`--no-mtp`) | 23.4 | 22.3 | 1.07x |
| mlxturbo MTP | 33.8 | 32.1 | 1.54x |
| mlx-serve 投機ゼロ (`--no-mtp --no-pld`) | 24.3 | 25.3 | 1.16x |
| mlx-serve MTP | 28.9 | 43.2 | 1.98x |
| mlx-lm | 21.8 | 20.9 | 1.00x |

- `--no-mtp` (lookup のみ) は投機ゼロと同じ数字 = この文では lookup の取り分ゼロ。`MLXTURBO_RUNNER=fallback` は `_build_base_runner` の先頭で FallbackRunner に落とす計測用の口 (融合は有効のまま)。
- **mlx-serve との差 (4k で 43.2 対 32.1) は素の効率ではなく投機の取り分** (相手は MTP で 1.71 倍、うち 1.40 倍。相手は depth 6 で per_draft 30%、うちは max_draft 8 の gated chain で平均 3.3 本/round)。同じ MTP 頭なので、的は gated chain の閾値 (`GATE_ROLLBACK_COST`、EMA) と chain の入力の質。小 M で verify 幅の代金が下がった今、閾値を掃引し直す (深く引く方が償却しやすくなった)。

### 2026-09-04 14:44 27B gate の上方向 (`h=0.30/0.45/0.60`) は採択線に届かず。`0.19` を維持するが、controller の意味不整合を直すまでレーンは閉じない

`decode_ab_generic.py`、短 3 本 × 512 × 2 回文、1 プロセス内交互測定
(`bench/results/gate-h-up-27b-short-0904.json`)。基準は現行 `h=0.19`。

| h | ms/tok | 対 0.19 | ms/round | tok/round |
|---|---:|---:|---:|---:|
| 0.19 | 28.498 | — | 85.847 | 3.038 |
| 0.30 | 28.056 | **-1.6%** | 73.005 | 2.597 |
| 0.45 | 28.755 | +0.9% | 68.138 | 2.364 |
| 0.60 | 28.208 | -1.0% | 56.818 | 2.013 |

- 事前に置いた採択線は短3本平均 -2%。最良の `0.30` も -1.6%で届かないため、4k / 17k と
  `MAX_DRAFT` 掃引へ進めない。`0.45/0.60` は verify を軽くする一方、tok/round を
  22.2 / 33.7%失う。入力別では `h=0.60` が現行比 -4.1 / -4.1 / **+5.8%** と反転する。
- `0.30` は1ケースで生成列が63トークン目から分岐。幅依存の4bit丸めの範囲だが、速度の
  採択線にも届かない。**当面の既定は 0.19 のまま**。
- blindspot 監査で、掃引対象の `_gate_depth` / `_plan_depth` に参照実装と異なる3点を確認した。
  (1) threshold が現在までの期待受理長ではなく未来 `d+1..cap` の積を使う、(2) 価格判定に
  失敗した位置も `keep` に含める、(3) 最初の miss より深い未観測位置まで0で EMA 更新する。
  その結果、`bench/test_spec_draft_chain_qwen3_5.py` は全位置の受理率が0.95でも深さ1になる
  非単調性を「既存仕様」として固定している。vendored Swift と Flash 側 `DepthController` は、
  first miss で観測を止め、失敗位置を選ばない。**h の上下だけでは正しい深さ価格を測れていない。**
- 次は controller の入力・選択・結果を1回の trace に残し、意味を参照側へ直した短3本だけを
  A/Bする。品質は出力一致ではなく通常の丸め級 / KLD ゲートで見る。掃引用 env はこの再測定の
  口として一時的に残す。

### 2026-09-04 14:48 blindspot 監査で残った次の2本: MTP cache 先頭行の再利用と proposal-only head

- **確定、先に正しさを検査**: 27B は draft の先頭で `_mtp_append(y, _mtp_base(h_last),
  mtp_cache)` を積んだ後、repair でその行まで trim し、同じ token / hidden / cache から先頭行を
  再計算する。単体計測の上限は eligible round あたり約1.24 ms。rejection / partial / full /
  D7 / EOS で cache tensor と次 proposal が一致することを検査し、一致した場合だけ短 A/Bへ進む。
- **未決、trace から**: exact q4 の lm_head matvec 自体は帯域天井の97〜99%で、書き直す的ではない。
  ただし proposal-only の MTP head まで全語彙 q4 を読む必然はない。Flash と vendored Swift にある
  q2 coarse top-32 + exact rerank を、まず exact proposal の top-32 recall を変えずに測る。
  recall が十分な場合だけ速度 A/Bへ進む。追加重み・warm-up・品質の代金があるため既定化は別判定。

### 2026-09-04 14:48 Flash-Next 小Mは N=2560 を外しても短 +0.1% ms/round。レーンを閉じる

全135射影の A は短3本 + 17k 3本の6/6で ms/round +0.26〜+0.70%。最後の切り分けとして、
`out_proj / o_proj / value_proj` など N=2560 を外す C (`N>=6144`、86射影) と素 B を、
短3本 + 17k 3本 × 512、1プロセス回文、長文は prefill-once で取り直した。

| 集計 | C | B | 差 |
|---|---:|---:|---:|
| short ms/tok | 16.403 | 16.207 | **+1.2%** |
| short ms/round | 35.147 | 35.120 | **+0.1%** |
| short tok/round | 2.153 | 2.186 | -1.5% |
| 17k ms/tok | 18.170 | 18.617 | -2.4% |
| 17k ms/round | 30.752 | 30.795 | -0.1% |
| 17k tok/round | 1.694 | 1.657 | +2.2% |

- 短文で遅く、「測った全条件で遅くならない」を満たさない。N=2560 の低並列度だけが負けの
  原因ではなかった。全射影版・部分集合版とも**不採用**。branch `worktree-agent-ae05b9756c852f071`
  の実験 commit `6553991` は main へ入れない。
- 全走行と集計表示の後、worktree に `bench/results/` が無かったため JSON の書き出しだけ
  `FileNotFoundError` で終了した。上の値は harness が書き出し直前に表示した集計。終了コード1を
  成功扱いにはせず、保存事故込みで記録するが、不採用判定は short の非改善だけで決まるため
  再走しない。

### 2026-09-04 16:10 27B controller の参照semanticsは短文 +6.0%。不採用で現行維持

旧実装と参照側には、未来利得 / 失敗位置keep / first miss以深の更新に差がある。最初の試作は
source-order walkとcensoringだけを移し、`h=0.02` で短 -7.4% / 4k -12.0% / 17k -15.1%を
出したが、独立レビューで未到達位置のcold priorがdepth 2を恒久上限にする欠陥を発見した。
またsourceのcost denominatorは最初が1.0で、試作は1+hから始めるoff-by-oneだった。この3結果は
不完全実装の探索値であり、採択根拠から除外する。

参照どおり、full accept時に次位置へ0.95上限のoptimismを移し、最初のdenominatorを1.0にし、
EOS停止をmissにもoptimismにも数えない版を実装した。実事前値からdepth 2→3へ開くことを純関数で
確認。`h=0.005 / 0.01 / 0.02 / 0.05` の短探索では0.05が最良だったため、実験専用
`tools/controller_ab_27b.py` で reference h=0.05 と legacy h=0.19 を比較した。

| 短3本 × 512 × 2 | reference | legacy | 差 |
|---|---:|---:|---:|
| ms/tok | 32.327 | 30.484 | **+6.0%** |
| ms/round | 99.640 | 93.343 | **+6.7%** |
| tok/round | 3.086 | 3.038 | +1.6% |

referenceは受理を1.6%増やすがround単価が6.7%上がり、短の採択線 -2%に届かない。4k / 17k / KLD
へ進めず、**本番はlegacy walk + h=0.19を維持**する。最終結果は
`controller-reference-h005-vs-legacy-h019-27b-short-final-0904.json`。隣のmeta JSONにコマンド、arm、
source SHA256を保存した。本番差分はEMA更新を挙動不変のmethodへ抽出しただけで、legacy更新の
直接テストを追加した。

拡大pytestは 526 passed / 10 failed。3つの失敗ファイルは個別processで 5 / 12 / 13 passed。
`bench/test_ngram_stream.py` がcollection時に `mx.set_default_device(mx.cpu)` を実行し、同じprocessで
後から走るGPUテストをCPU判定にする既存のtest isolation欠陥だった。controllerの故障ではないが、
別論点として直す。

### 2026-09-04 16:25 27B MTP cache repair は rejection round の先頭1行だけ保持。短文 -0.9%で採用

draft の先頭行と repair の先頭行は token / hidden / 直前cacheが同じ。ただし accepted prefix が
2行以上ある場合、全再構築の一括appendと「先頭を保持して残りだけappend」ではGEMM幅が変わり、
合成float32 cacheに最大 `9.536743e-7` の差が出た。strict equivalenceを守るため、先頭draftを
引いていて `consumed == 1` のときだけ保持する。これはfirst-link rejectionと、最初のdraftがEOSの
場合。partial / full / EOSの2位置目以降は従来どおり全再構築する。

合成qwen3_5で rejection / partial / full / D7 rejection / D7 lookup partial / EOS先頭 / EOS後段 /
direct lookupを通し、active K/V cacheと次MTP hidden・proposalをbit一致で確認した (8 passed)。
実モデルは `mtp-repair-reject-retain-27b-short-0904.json`、短3本 × 512 × 2回文、同一processのABBA。

| 短文 | retain ms/tok | legacy | 差 | eligible round |
|---|---:|---:|---:|---:|
| case 0 | 37.491 | 37.850 | -0.9% | 61 / 217 |
| case 1 | 45.571 | 45.927 | -0.8% | 28 / 164 |
| case 2 | 39.148 | 39.570 | -1.1% | 16 / 95 |
| **全体** | **40.737** | **41.115** | **-0.9%** | — |

ms/roundも124.300対125.462で-0.9%。tok/round、受理数、round数は両側同一で、生成列は
512 / 512 / 345 tokenすべて一致した。走行中に34.3→48.2 ms/tokの熱ドリフトがあり、回文ごとの
符号は一部揺れたが、prompt別平均は3/3で非悪化。品質・メモリの代金がなく、実装も既存repairの
局所methodだけなので、1%未満の代金ゼロ改善の規則に従い採用する。A/B専用wrapperとmeta JSONの
source SHA256を残した。

### 2026-09-04 16:27 `test_ngram_stream` のCPU固定を局所化。CPU→GPU同一processで53 passed

module import時の `mx.set_default_device(mx.cpu)` をautouse fixtureへ移し、各テストの前にCPUへ、
終了時に元のdeviceへ戻すようにした。n-gram 23本を先頭に置き、その後ろへcontroller拡大実行で
誤skipしたcapture 5本 / attention+MLP 12本 / GDN 13本を同一pytest processで並べて53 passed。
既存の526 passed / 10 failedは実装故障ではなく、このcollection時のprocess-global設定漏れだった。

### 2026-09-04 16:58 27B proposal headをq2 top-32へ。短 -2.0%、4k -1.6%、17k -3.0%で採用

まずexact q4 proposalを変えないobserverで1,677 linkを測った。継続条件は走行前に「全体R@32
99.9%以上、各prompt 99.5%以上」と置いた。

| 指標 | 結果 |
|---|---:|
| q2 R@1 / R@4 | 85.152% / 98.927% |
| q2 R@8 / R@16 / R@32 | 99.881% / 100% / 100% |
| q4候補行再採点と全語彙q4 proposalの一致 | 99.284% |
| q2常駐 / 構築 | 378.9 MiB / 0.08s |
| 起動時dense temporary | 約2.37 GiB |

再採点A/Bはq2を両armで常駐させ、同一processのABBA、512 token、長文はprefill-once。

| 文脈 | rerank ms/tok | exact ms/tok | 差 | tok/round差 | 生成列 |
|---|---:|---:|---:|---:|---|
| 短3本 × 2回文 | 30.626 | 31.259 | **-2.0%** | 0.0% | 512 / 512 / 345一致 |
| 4k × 2回文 | 37.298 | 37.914 | **-1.6%** | -0.5% | 512一致 |
| 17k × 2回文 | 47.793 | 49.267 | **-3.0%** | **+6.6%** | 262 token目で分岐 |

17kの絶対値は52.4→44.4 ms/tokへ逆向きに動き、筐体温度の変化が大きい。位置順も含むABBA平均を
採用し、絶対tok/sは根拠にしない。17kではrerankのround単価が +3.4%だが、proposalの違いで
受理率が2.427→2.586へ上がり、壁時計は3.0%短くなった。分岐後の両出力は同じ問いへの整合した
文章で、target verifyと通常headはexactのまま。検証幅ごとのq4丸め差に属する。短・4k・17kの
全条件で速く、追加378.9 MiBは27B重みの0.4%未満なので既定採用する。

本体はq4 affine / group64だけでrerankを構築し、それ以外はexactへfail closedする。
`MLXTURBO_DRAFT_RERANK=0`で従来proposalへ戻せる。合成qwen3_5でq2候補行再採点とoff時exactを固定し、
SpecEngine関連は27 passed。本体移植後の実モデル128 token smokeも短3本でms/tok -2.9%、
生成列128 / 128 / 128一致。再現道具は`tools/mtp_head_recall_27b.py`と
`tools/mtp_head_rerank_ab_27b.py`、結果は`mtp-head-q2-*-0904.json`。

### 2026-09-04 17:04 MTPLX #391とVozを現行バックログへ照合

MTPLX #391の16k 80.92 tok/sは、137GB group32 pack、temperature 1、固定depth 3、fans最大 / 40°C
gateの値で、こちらの98GB group64・ABBAとは直接比較できない。24項目のうちn-gram先読み、staged投入、
QSA decode、GDN prefill、session reuseは同等レーンがあり、4096 chunkと広いroute chainは既に棄却済み。
未実装で報告値が大きいfixed-M4 verifier (+21%)とFR-Spec Q8 (+6.27%、coverage 99.64%)だけを、
K/V prefix trimの後へ追加した。2本を同時に変えず、group64 packで個別に判定する。

Vozは467MBの固定ASRグラフをfallback無しでANEへ載せた有力な実例だが、動的KVを持つMLX LLMへの
差し込み口ではない。Appleの公開Core MLにはstateful KVと圧縮weightがある一方、圧縮weightのruntime
展開とPython/MLX境界の代金は実機依存。既存M3 Max実測もANEはGPU占有時の8k以下で勝つが、空きGPUと
16kでは負けている。full model移植は再開せず、FR-Spec 65,536-row head完成後だけ、定常RSS非増加・
I/O込み20%以上・short/17k非悪化の置換実験を行う。

### 2026-09-04 17:23 Flash-Next SDPAのK/V prefix trimは短文+5.2%で不採用

通常attentionの幅分割は各query塊から未来のK/V列を見ないので、K/Vとbool maskを
`offset + t1`まで切る案を試した。合成では非ゼロoffset、S=2/3/4/6/8、fp32/bf16が従来の
full K/V版と許容差内で一致し、関連18 testとvendor fingerprintも通った。

既存の`sdpa-split` knobはoff側でQSA decodeまで外すため、今回の差だけを測る一時
`sdpa-prefix` knobを作り、split自体は両armでonに固定した。短3本 × 512の同一process ABBA:

| 短文 | prefix trim | full K/V | 差 |
|---|---:|---:|---:|
| ms/tok | 25.255 | 24.008 | **+5.2%** |
| ms/round | 54.777 | 52.157 | **+5.0%** |
| tok/round | 2.186 | 2.186 | 0.0% |

生成列は3/3で一致。筐体の熱で絶対値は18ms/tok台から24ms/tok台へ落ちたが、ABBA内で
prefix側が全prompt遅く、round単価にも同じ符号が出た。17kでは既定のQSA decodeが通常attentionを
迂回し、このsliceは発火しない。したがって文脈長gateを足さず、実装と一時knobを戻して不採用とする。
結果は`qwen4-sdpa-prefix-direct-short-0904.json`。途中の`qwen4-sdpa-prefix-17k-0904.json`は
knob交絡を発見して中断した不完全runなので採否には使わない。
