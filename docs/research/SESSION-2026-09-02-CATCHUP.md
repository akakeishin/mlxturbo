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
