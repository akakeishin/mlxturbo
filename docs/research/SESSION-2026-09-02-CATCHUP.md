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
