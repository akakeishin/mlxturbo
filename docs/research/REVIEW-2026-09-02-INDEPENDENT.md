# 独立レビュー (2026-09-02 夜、main = 2ff607f)

セッションの文脈を一切渡さない読み取り専用の Opus 4 本に、コードだけを根拠に欠陥を探させた。
領域は (A) 投機デコード中核 (`spec_flash.py` / `spec.py` / vendor シーム)、(B) バッチ経路と
生成器 (`batch.py` / `batch_spec.py` / n-gram / MTP 学習)、(C) サーバーと実行器
(`server.py` / `runner.py` / `hub.py`)、(D) カーネル層と量子化 (`kernels/*` / `fused.py` /
`fast_qmm.py` / `convert*` / 未追跡の `tools/micro_kernel_latency.py`)。
GPU で実モデルを回す検証はさせていない。CPU テストと合成入力の小さな GPU 実行だけ。

親 (Fable) が自分で裏を取ったのは A-1 (コードを読んで確認) のみ。他は各レビュアーの
確信度をそのまま載せる。「確実」はレビュアーが実行して再現したもの、「高」は読解のみ。

## 先に結論

- **出力の正しさを壊す欠陥は、本番の既定経路には見つからなかった。** 投機のロールバック、
  サンプリングの分布、シーム 3 呼び手の引数整合、QSA の未来漏れ、prefill のチャンク境界は
  いずれも問題なし (末尾の「問題なしと確認した項目」)。
- **受理率を静かに落としている欠陥が 1 つ実在する** (A-1)。MTP ドラフトキャッシュが
  受理された中間トークンを積まず、位置が毎ラウンド hit ぶんずれる。既定 on の経路。
  レーン 7 より先に直す価値がある (受理率が上がれば tok/step がそのまま上がる)。
- **契約テストが 2 本、動いていない** (C-7、D-補足)。main で 1 本が失敗、1 本が ImportError。
- **API の既定挙動で直すべきもの** が 3 つ (Anthropic usage の二重計上、`n` の黙殺、
  事前トークン化プロンプトの語彙検査なし)。
- 残りは「既定 off の knob を触ると落ちる」「バッチ投機を再訪するときに踏む」「保守上の穴」。
  バッチ投機は畳んであるので、再訪時の前提としてここに残す。

## 対処の優先順 (親の判定)

1. **A-1** MTP キャッシュに確定トークンを再投入する。`spec.py:1247` と同じ形
   (trim → 確定ぶんを `(embed(t_k), hyper_{k-1})` の対で積む)。**ゲート**: 複数プロンプト x 512
   の平均 tok/round が上がること。上がらなければ戻す。KLD は動かないはず (出力は trunk の
   logits からしか出ない) が、`bench/quant_eval.py compare` は回す。
2. **C-7、D-補足** 壊れている契約テスト 2 本を直す (小さい)。
3. **C-1〜C-3** API の既定挙動 3 件。
4. **D-11** `tools/micro_kernel_latency.py` は commit 前に A/B を交互化し、lm_head を 6bit に
   する。今の形で取った数字は使わない。
5. **B-2** `train_mtp.py` の norm 不一致。次に MTP ヘッドを焼き直すときに直す (bf16 アーカイブの
   再取得待ちで、それまで手を付けない)。
6. **D-5、D-4** 既定 on カーネルの knob / 適格判定の穴。踏む条件が限定的なので、レーン 1 で
   カーネルを触るときに一緒に。
7. **D-2 とレーン 2 の再測**。`--knob wide` の A/B は既定 SORT_MIN で専門家の連結経路に到達せず、
   B 側の `disable` も `_fused_w` を外さない。レーン 2 を畳んだ根拠が専門家の連結の費用では
   なかった疑いが強いので、`disable` を直して `MLXTURBO_SORT_MIN=0` を明示し取り直す
   (LANES のレーン 2 に条件を書いた)。
8. **B-1、B-3** バッチ投機の QSA まわり。再訪の条件が発火するまで保留。ただし
   `batch.py:611-614` の docstring の保証は今は成り立たないので、そこだけ書き換える。
9. 残りは記録のみ。

---

## レーンへの影響 (LANES-2026-09.md に同じ内容を書き込んだ)

- レーン 1 (decode 固定費): D-11 の道具を直してから帰属する。HC カーネルを触るなら D-4 も。
- レーン 2 (fused MoE): D-2 により畳んだ判定を保留に戻す。再測条件はレーン 2 の項。
- レーン 5 (並列デコード): B-1、B-3 を直してからでないと混在長で出力が変わる。
- レーン 6 (tape-replay rollback): A-4 を同時に畳む。
- 既定経路で速くなる見込みがあるのは A-1 (受理率) だけ。D-12 は lookup 経路のみ。

## A. 投機デコード中核

### A-1 MTP ドラフトキャッシュが受理された中間トークンを積まない (確実、親が確認)

`spec_flash.py:857,876` `_draft_chain` は入口で `keep = cache.size() + 1` を取り、チェーンを
引いた後そこまで縮める。つまり 1 ラウンドで MTP キャッシュに残るのは `cur` の 1 列だけ。
一方 `generate_stream` (`spec_flash.py:1537`) の次ラウンドは `cur = toks[-1]`、すなわち
`cur` から 1+hit 個先のトークンで始まる。hit ≥ 1 のラウンドでは、間に挟まる受理済み
トークンが MTP キャッシュに一度も書かれない。

- MTP キャッシュの `offset` (RoPE 位置) が真の位置より毎ラウンド hit ぶん遅れる。depth 2・
  受理率 0.7 なら 512 トークンで数百位置。
- ドラフトヘッドの自己 attention の履歴に穴が空く。
- `_prime_draft_cache` の docstring (`spec_flash.py:948-951`) は「no gap and no duplicate」を
  不変条件として宣言しているが、最初の受理でそれが破れる。
- `spec.py:1247` は検証後に `mtp_cache.trim` → `_mtp_append` で確定ぶんを積み直しており、
  こちらが正しい形。
- 出力は壊れない (採用トークンは trunk の検証 logits からしか出ない)。壊れるのは受理率で、
  生成長に比例して落ちる。
- チェーンが書いた d1..dk の列を残す修正では駄目 (それらは MTP 自身の hyper から作られて
  いて priming の規約と違う)。

### A-2 `MLXTURBO_PREFILL_TAIL_CHUNKS=0` はプロンプト長が 2048 の倍数で落ちる (確実)

`spec_flash.py:471,1192-1270`。TAIL=0 だと group がプロンプトを食い切り、`capture` を張る
最終チャンクが走らず `cap` / `logits` が None のまま `hyper_tail0 = cap.hyper[...]` に到達。
n=4096 / 8192 で再現、n=4196 は chunk に落ちるので通る。既定 TAIL=1 は安全。knob の
docstring が掃引を勧めているので 0 は踏まれうる。

### A-3 `_group_prefill_forward` が `_make_masks` シームを迂回 (事実は確実、危険性は中)

`spec_flash.py:514,540`。layer-major prefill だけ `conv_mask = None` と
`Q.create_attention_mask(hs[ci], [c])` を直に組む。`batch.py` / `batch_spec.py` の
`make_masks` は左パディングの bool マスクと、右パディング列を GDN 状態から外す `conv_mask`
をここで作る。group prefill をバッチに載せた瞬間に両方が消える。現状は単一系列専用で
実害なし。CLAUDE.md のシーム一覧に「4 番目の呼び手」として書かれていない。

### A-4 `capture()` の GDN / PLE 差し替えが `_store_conv_state` 系を通らない (高)

`spec_flash.py:305,319,352`。本家は `_tail_window(cache, ...)` (`cache.lengths` があれば
実長基準) を通すが、capture 側は 3 か所とも生スライス。full capture に到達する経路では
`lengths` が必ず None (`batch_spec.py:1146-1161` が decode 前に `finalize()`) なので現状は
等価。`_tail_window` を変えても capture は追随しない。

### A-5 env のパースが 2 か所だけ空文字に耐えない (確実、軽微)

`spec_flash.py:126,471`。`MLXTURBO_DEPTH_CTX_LIMIT=` / `MLXTURBO_PREFILL_TAIL_CHUNKS=` で
import が `ValueError`。同ファイルの他 knob は `int(env or 0)` で吸収している。

### A-6 `temp<=0` かつ `sampler` ありで argmax を組んで eval してから捨てる (高、現行は到達不能)

`spec_flash.py:1409-1414` は `temp <= 0 and drafts` で `nxt_all` / `dv` を eval するが、
`_verify` は `temp > 0 or sampler is not None` で sampler 側に入り `precomputed` を無視する。
`runner._position_local_sampler` は `temp <= 0` で必ず None を返すので今は踏まない。
`generate_stream` は公開 API なので分岐条件を `_verify` と揃えておく。

### A-7 `MLXTURBO_PIPELINE` 経路は「作って捨てる」遅延グラフそのもの (高、既定 off)

`spec_flash.py:1424-1440`。`lg2 = model(pair2, ...)` は async_eval されず、全採用でなければ
参照ごと捨てる。`drafts2 = _draft_chain(...)` は depth>1 で内部 `async_eval` を投げるので
捨てるラウンドでも GPU が動く。棄却済み knob だが、再測するときの前提として記録。

### A-8 `ids` 幅 0 かつ `resume=None` で落ちる、`resume` の docstring が 2-tuple (高、低頻度)

`spec_flash.py:1268-1270`。`FlashSpecRunner` は `reused -= 1` で必ず 1 トークン残すので本番は
踏まない。`resume` の docstring は `(logits_last, hyper_prev)` だが実装は `mtp_snap` 込みの
3-tuple を unpack する (`:1178,1273`)。

### A-9 `arch.py:248-258` の `take_along_axis` の根拠コメントが誤り (確実)

「範囲外は末尾へクランプ (実測確認済み)」とあるが、mlx 0.32.2 で長さ 5 に添字 7 を渡すと
0 が返る。ガード (`raise ValueError`) 自体は正しい。「クランプなら安全」と読む人が出る前に
コメントを直す。

## B. バッチ経路と生成器

### B-1 バッチ投機で、budget 以下の短い行もブロックマスクを通り自分自身が不可視になる (確実、静的読解)

`batch_spec.py:768,831-836`。`_ragged_indexer_call` の疎化判定は「どれか 1 行でも論理 kv 長が
`token_budget` を超えたら全行が疎化経路」。`in_block` の内側では causal を使わずブロック選択
だけが効き、`visible` は `block_end <= q_col` を要求するので、クエリ自身が属するブロックは
(`q_col ≡ cr-1 mod cr` でない限り) 不可視。例: cr=4、短い行の論理 kv 長 104、クエリ位置 101
は列 100・101 が見えない。単独実行なら kv 長 104 ≤ 2048 で素の causal。docstring
(`:736-740`) の「budget 以下なら素の causal と同じ」は成り立たない。発生条件は「17k と短い
リクエストの同居」で、`spec_batchable` の長さ上限を外した目的そのもの。cr=4 なら各ラウンドの
クエリの 3/4 が影響を受ける。
(本家 `qwen4_exp.py:475-487` の S>1 経路にも同じ形の欠落があるが、そちらは長い行にだけ
効くので単独実行と一致する。)

### B-2 MTP ヘッドに渡す hidden の正規化が学習と推論で食い違う (高)

推論 `spec.py:566-570,1013,1062` は `inner.norm(h)` / `mtp.norm(h_mtp)` を掛けた post-norm
(docstring に「post/post が 2x2 実測で全深さ勝ち」)。学習 `train_mtp.py:86,106` は
`_hidden_forward` の生 (final norm 前) と MTP ブロック出力 (mtp.norm 前) を食わせている。
`inner.norm` は学習重み `w` を要素ごとに掛けるので `pre_fc_norm_hidden` があっても同じに
ならない。`train_mtp.py` で焼いたヘッドは推論で受理率が落ちる。

### B-3 `classify()` の QSA 発火条件がリクエスト単体の長さで、pool 同士でも QSA に入る (高)

`batch.py:647-657` は `prompt_len + max_tokens > budget` で solo/pool を決めるが、QSA が
見るのはバッチで共有される物理列数 (`BatchKVCache._idx` = 最長プロンプト + 経過ステップ)。
A = 2040+8、B = 8+2040 は両方 pool で同居し、物理列数は 4080 > 2048。docstring
(`:611-614`) の「budget を跨ぎうるリクエストは必ず単独」と、それに依拠した KLD ノイズフロアの
主張が崩れる。正しくは「同居しうる行の max(prompt_len) + max(max_tokens)」。

### B-4 単一行 rollback で `keep == 0` のとき GDN 状態を None に落とす (中、到達性は低)

`arch.py:195`。行別版 `rollback_recurrent_rows` (`:242-247`) は同じ状況を ValueError に
するのに、単一行版は黙って履歴を消す。

### B-5 `_truncate` の `max(1, ...)` が max_tokens 超過を許す (中、現状は到達不能)

`batch_spec.py:2341-2347`。到達済みの行は `_after_round` が必ず retire するので今は通らない。
retire が 2 系統 (`_join_lane` 経由の `_settle` と `_decode_round`) に分かれているので、
片方が変わると 1 トークン漏れる。

### B-6 `on_done` が例外を投げると Future が永久に未解決 (中)

`batch.py:733-755`、`batch_spec.py:1915-1932`。`on_done` の後に `future.set_result` なので、
投げると Future が宙に浮く。`batch_spec` は先に `done = True` を立てるので再試行も効かない。
現行の `on_done` は `queue.put` だけなので投げない。

### B-7 n-gram の「RAM 0」表示と既定 4M 行のキャッシュ確保 (中)

`ngram_stream.py:268-272,562-565`。`_NGramCacheGen` は `prefetch_enabled` に関係なく作られ、
上限は既定 419MB (4M 行 x 100B)。`np.empty` なので触るまで commit されないが、`install()` の
`RAM 0` は言い切りすぎ。

### B-8 `batch.py:441` のコメントが実装より狭い (確実、影響は品質のみ)

`block_starts >= left_pad` は「全部パディングのブロック」だけでなくパディング境界を跨ぐ
ブロックも落とす。安全側。

## C. サーバーと実行器

### C-1 Anthropic の usage が prompt を二重計上し、stream と非 stream で値が違う (高)

`server.py:2441-2445,4126,4386-4389`。`prefill_new` が `input_tokens` と
`cache_creation_input_tokens` の両方に入る。本家 API では 3 つが互いに素で合計がプロンプト長。
stream だと `message_start` が `len(prompt_ids)` を流し `message_delta` で訂正しない。

### C-2 OpenAI の `n` が黙って無視される (確実)

`server.py:3316-3560,4436-4500`。`{"n": 3}` で 400 も警告も出ず choices が 1 件。
`response_format` / `tool_choice: required` / `logprobs`+`stream` / embeddings は明示的に
断っているので、方針から漏れているのは `n` と legacy の `best_of` / `echo` / `suffix`。

### C-3 `/v1/completions` の事前トークン化プロンプトに語彙範囲の検査がない (確実)

`server.py:4415-4434`。`[-1]` や `[999999999]` が通り、MLX の gather は範囲外で例外を出さず
黙って値を返す (実測: ゼロ行)。`_parse_logit_bias` (`:1895`) のキーも同様。

### C-4 `max_tokens` が残り文脈長に対して検査されていない (中)

`server.py:2337-2366`。`_check_context_length` はプロンプト長だけ。上限ぎりぎり +
max_tokens 4096 の decode が上限を越えて進み、Metal 一括確保が binding なら 200 を返した後に
ストリームが途中で死ぬ。

### C-5 `_responses_stream` に `cancelled` の分岐がない (コード事実は高、到達性は低)

`server.py:5689`。他 3 経路は専用処理があるが Responses だけ `else: error` に落ち、
`"None"` を message にした `response.failed` を出す。バッチ側は `on_done("cancelled", None)`
を投げる契約 (`batch.py:742`、`batch_spec.py:1924`) なので、4 経路中ここだけ契約外。

### C-6 `_try_trim_session_cache` が `indexer.keys` を切り詰めない (中)

`server.py:427` vs `:513-517`。checkpoint 経路は docstring で「trim 後に `indexer.keys` の
再切り詰めが要る」と明記して実施しているが、`_select_session` が先に試す trim 経路は
`trim_prompt_cache` だけ。現状は GDN の `ArraysCache` が混ざり `can_trim_prompt_cache` が
False なので到達しない。

### C-7 `bench/test_server.py` が main で 1 本落ちる (確実、実測)

379 passed / 1 failed。`test_flash_spec_generate_stream_zero_tokens_prefills_without_yield` が
`spec_flash.py:586` の `for pli in m.ple_layers:` で `SimpleNamespace` に `ple_layers` が無く
AttributeError。`_prefetch_ngram_rows` は `getattr(stream, "prefetch_enabled", False)` で
stream 側は守るが、その手前が無防備。「max_tokens=0 で prefill の cur を漏らさない」契約が
いま守られていない。

### C-8 錠の受け渡し中の切断で、解放が async generator の GC 頼み (中)

`server.py:3627,4152,4625,5498`。`async for keepalive in _acquire_lock_with_keepalive(...)` の
`yield` 中に外側が `aclose()` されると、内側 generator (すでに `lock.acquire()` 済みの task を
抱えうる) は宙に浮き、`except BaseException` 内の release は asyncgen finalizer 待ち。既存
テストは `asyncio.run()` 復帰後に判定するので即時解放を証明していない。`contextlib.aclosing`
で包む。

### C-9 容量系フラグの一部が `_positive_int` を通っていない (確実、影響は低)

`server.py:5993,6100,6135,6143`。`--max-queue 0` で恒久 503、`--max-tokens 0` で cap 0、
`--max-context-tokens -1` で全 400。起動時に気づけない。

### C-10 `stream` / `store` が JSON の型を見ずに truthy (確実、影響は低)

`server.py:3377,3965,4470,5333`。`{"stream": "false"}` でストリーミング有効。

### C-11 `hub.py` の download 先に未検証の `--name` (低、推測含む)

`--name` が絶対パスなら `Path(dir) / "/abs"` は `/abs`。`rfilename` の `..` は
huggingface_hub 側の無害化を未確認。

### C-12 `--api-key` が argv に載る、500 が `str(exc)` を素で返す (中、影響は低)

`server.py:6151`、`:3487,4034,4548,5401`。CLAUDE.md の HF トークンの方針がそのまま当てはまる。

## D. カーネル層と量子化

### D-1 `gated_delta_blocked` の逆行列が Neumann 級数で、ブロック内 k の相関が上がると壊れる (確実、GPU 実測、既定 off)

`gated_delta_blocked.py:193` `_unit_lower_inv_dense`。真の逆行列は良条件 (単位下三角) なのに
級数展開の途中で 1e18 まで伸びる。64x64 一様値の絶対誤差: 0.1→7.9e-8 / 0.4→2.0e-4 /
0.6→8.6e-3 / 0.8→1.5。`tools/verify_gdn_blocked.py:46` は k を正規乱数で作る
(mean|k·k| ≈ 0.09) ので構造的に安全域しか踏まず検出できない。`:97` の「融合カーネルの
正しさの基準」という位置づけは成り立たない。前進代入に替える。

### D-2 `disable_wide_projections` がエキスパート連結を一つも外していない (確実)

`fused.py:698` は `_wide_experts` を消すが、その属性は誰も代入していない。`enable` が
実際に置くのは `sw._fused_w/_fused_s/_fused_b` (`:760`)。`tools/decode_ab.py --knob wide` の
B (off) 側でも連結が有効なまま測る。

### D-3 `_WIDE_FORCE_ON` はプロセス全体に効く (確実)

`fast_qmm.py:153` / `dispatch.py:108`。モジュールグローバルなので `fast_qmm()` の全呼び出しが
読む。`MLXTURBO_FAST_QMM=1` で route が STOCK 以外になった時点で立ち、`MLXLM_FAST_QMM_WIDE`
未設定でも M=9..16 が wide に入る。「Wide の既定は off」が破れる。

### D-4 hyper-connection の `eligible` が down/up/inject の bits・group_size 不一致を通す (確実、実測、既定 on)

`hyper_connection.py:675,715`。判定は各層を個別に見るだけで同値を要求せず、cfg は down の
値だけ拾う。カーネルは up・inject も down の bits/gs で復号。hc=4, d=256, lowrank=64 で
揃い→0.0039 / up だけ 8bit→0.0586 / up だけ gs=32→0.0254。gs 不一致では scales の範囲外
読み。現行レシピは 3 層とも同じ class なので潜在。

### D-5 `MLXTURBO_GDN_BLOCK_T` が fp32 の TB 引き下げを素通しして落ちる (確実、再現、既定 on)

`gdn_blocked_metal.py:204`。env 指定時は dtype 分岐に入らない。fp32 + `=32` で
`Threadgroup memory size (40192) exceeds the maximum (32768)`。`eligible()` は True なので
fallback にも落ちない。

### D-6 `moe_verify_gather` が S > 8 で ValueError (確実、再現、既定 off)

`moe_verify_gather.py:443`、ゲートは `fused.py:909` の `indices.size >= 64` のみ。対象モデルは
top_k=10 で S ≤ 6 なので踏まないが、`SwitchGLU.__call__` は全 MoE モデル共通に差し替わる。

### D-7 要素数 8 未満の入力でカーネルがコンパイルに失敗する (確実、再現、合成のみ)

`qmv_wide_nocap.py:204` / `moe_route.py:138`。MLX 0.32.2 は要素数 8 未満の配列を `constant`
で渡すが、ソースは `const device T*` 直書き。本番形状では踏まないが合成モデルの検証ハーネスは
踏みうる。他 5 カーネルは最小サイズが構造的に 8 以上。

### D-8 MoE 経路の範囲外読み 2 件 (高、Python 側にガードなし、既定 off)

(a) `moe_glu.py:43,78`: 早期 return は `row0 >= H` だけで、その後 4 行を無条件に読む
(書きだけガード)。`eligible()` は H を見ない。
(b) `fused.py:941` `enable_moe_shared_fold`: 513 行バンクを `_fused_w` と `down_proj` にしか
置かないのに、`dispatched()` の優先順で `_fused_w` を見るのは最後の `wide` だけ。既定
`MLXTURBO_SORT_MIN=16` なら `gather_sort` に落ち、添字 512 が 512 エキスパートの `gate_proj`
に入る。`runner.py:1293` が出荷経路から外しているのが救い。

### D-9 `calibration.load(path)` を明示指定すると次の呼び出しでプロファイルが捨てられる (確実、再現)

`calibration.py:52`。明示 path のとき `_loaded` を立てないので、直後の `describe()` が env を
読み直し `_profile = None` で上書き。

### D-10 非 affine 量子化層で `_pack_quantized` が KeyError (確実、実測)

`fused.py:141` (compiled 版 `:109`)。mxfp4/nvfp4 の `QuantizedLinear` は `biases` を持たない。
`_build.qmm` (`:68`) は `mode=` を渡さず affine 固定。`fast_qmm.py:384` が明示的に非 affine を
弾いているのと非対称。

### D-11 `tools/micro_kernel_latency.py` (未追跡) の A/B がブロック測定、lm_head が 8bit (高)

`:281,323,359` は「fused 200 回 → plain 200 回」で CLAUDE.md の「1 プロセス内で交互」に反する。
`:47,377` の `QBITS = 8` に対し `convert_flash.RECIPES["v-fast6"]` の head は 6bit
(675MB vs 516MB) で、`median_us` / `effective_gbps` / 帯域床が 24-30% 過大。`:213` は前周回の
`_combine` の eval を計測窓に含む。カーネル呼び出しのシグネチャ 4 本と plain 参照実装は正しい。

### D-12 `sam.draft()` が「最新」ではなく「最初」の出現の続きを返す (確実、実測)

`sam.py:111,133`。`endpos` を state 生成時に一度書くだけで再出現で更新しない。docstring の
"the most recent earlier occurrence" と食い違う。返る位置は必ず実在の出現なので不正な
ドラフトにはならず、受理率が本来より落ちるだけ。

### D-補足 `bench/test_qmm_skinny_mma_static.py` が ImportError で 1 行も走らない (確実)

`:10` が `ACTIVE_INPUT_GROUPS` ほか e120 の名前を `_qmm_skinny_mma_source` から import
(実体は `_qmm_e120_source.py`)。v5 MMA カーネルの静的ゲートが現在動いていない。
ほか `qwen4_exp.py:1138` の「既定 off」コメントは `_gdn_metal` (既定 on) に付いていて誤り。

---

## 問題なしと確認した項目 (抜粋)

- ロールバックの完全性: KV `trim`、indexer keys、GDN conv 窓 / 状態、PLE 窓、n-gram 文脈の
  オフセットが `conv_input` / `full` の連結の作り方と全部合う。
- サンプリング: 「全位置を先に引き、ドラフトと一致するプレフィックスだけ採用」は決定的
  ドラフトに対する厳密なサンプリング。履歴依存ペナルティは `SUPPORTED_SAMPLING_PARAMS` から
  除外されている (レーン 7 の前提どおり)。
- `_IndexerCache` の pooled 無効化は全経路が `keys` setter を通る。
- `_pipeline_snapshot` / `_restore`: MLX の `arr[a:b] = v` は関数的で先に取った view は変わらない
  (実測)。`snapshot_mtp_cache` の docstring の理由付け (「KVCache は in place」) は実挙動と違う。
- prefill のチャンク境界: 既定 (GROUP=4, TAIL=1, FOLD=1) で n = 2048/2049/3000/4096/8192/
  8292/10240 を模擬し、常に最終チャンクが chunk-major に落ちる。
- `tools/vendor_fingerprint.py`: staged == 本家、group == chunk-major (max|diff| ≤ 1.1e-6)、
  causal 検査 0。
- QSA の未来漏れなし (per-query の `visible` と端数列の因果窓)。
- シーム 3 呼び手の引数・返り値が一致。`_ORIG_ATTN_POSITIONS` による fast-rope ゲートも
  差し替え時に正しく外れる。
- `RaggedLedger.round_mask` の memo は全更新経路が `_invalidate` を通る。`qsa_logical` の
  assert は `_staged_forward` が `pair` を 1 回で流すので常に成立。
- サーバー: マルチバイトのチャンク分断なし (detokenizer は毎回別インスタンス、不完全 UTF-8 は
  flush 保留)、EOS と finish_reason、cancel 時の錠・セッション (`shield` + `_GenerationCancelled`、
  `publish` 前に抜ける)、キュー枠の所有権、1 行の失敗が他行に伝染しない、`_find_stop` の
  無限ループなし、バッチと単独の直列化 (`max_workers=1`)、認証は `compare_digest`。
- カーネル: `prefill_attn` 10 構成で dense sdpa と一致、`qmv_wide_nocap` 本番形状で
  ビット一致、`qmm_skinny_mma` v5 の所有式と整列、`moe_route` の選択集合、`rms_norm_gated`
  誤差 0、`gdn_prework` の `conv_state_out` 誤差 0、GDN の S/T 端数と巻き戻し契約、
  `test_gated_delta_states.py` 全通過、convert 系の norm 収支と shard 索引。
- env knob の既定: `MLXTURBO_GDN_METAL` (既定 on)、`MLXTURBO_HC` (既定 kernel)、
  `_GDN_PREWORK` / `_GDN_BLOCKED` / `_MOE_VERIFY` / `_FAST_ROPE` / `_HC_PREFILL` (既定 off) は
  CLAUDE.md の記載と一致。
- CPU テスト: `test_spec_phase0` / `test_block_verify` / `test_gate_static` / `test_sam` 26 通過、
  `test_ngram_stream` / `test_mlx_compat` / `test_dispatch_static` / `test_spec_dispatch_static`
  16 通過、`test_convert` 3 通過 3 skip (スナップショット不在)。

## 読み切れなかった範囲

- `runner.py` 1000-1400 (`DraftSpecRunner.generate` 本体、`enable_default_fusions`) と
  1600-1821 (`_build_base_runner`)。
- `batch_spec.py` のスケジューラ・台帳・compaction の正しさ (A のレビュアーは未読、B の
  レビュアーはシーム整合と QSA に集中)。
- `spec_flash.py` の `capture()` 本体と `_group_prefill_forward` 内部 (B のレビュアー)。
- `ngram_stream.build_sidecar`。
- 実モデルでの GPU 検証は一切していない。B-1 / B-3 は静的読解のみ。
