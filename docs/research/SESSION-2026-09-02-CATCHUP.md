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
