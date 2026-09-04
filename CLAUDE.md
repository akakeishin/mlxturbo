# mlxturbo で作業するときの決まり

対 mlx-serve の性能レーンが主戦場。現在地と次の一手は
`docs/research/KERNEL-BRIEF-DECODE-BW.md` の末尾が常に最新。着手前に必ず読む。

## 計測の作法 (破ると数字が嘘になる。全部実測で確立済み)

- A/B は 1 プロセス内で交互に測る。プロセスを分けた比較は熱・キャッシュ状態で数 % ずれる。
  **プロセス起動直後の最初の計測行は +7〜9% 遅い** (温めでは消えず、回文順でも位置 1 の段差は打ち消せない)。
  常駐 worker (`tools/ab_submit.py`、`tools/biglock.sh` 経由なら自動) は読み込み直後に捨て A/B を 1 回入れる。新しいプロセスで測るなら同じ burn-in を入れること。
- tok/step (受理率) の比較は複数プロンプト x 512 トークンの平均でだけ行う。
  単一プロンプトは chunk 境界の丸め 1 つで挙動が変わる (テキスト運)。
- 温キャッシュのマイクロベンチの絶対値を信じない。案の優劣の目安にだけ使い、
  採否は in-model A/B で決める (micro 勝ち in-model 負けの前例が複数ある)。
  **カーネルの連鎖 micro は重みを 100 MB 超 (層数ぶん) 巡回させて冷やすこと。**重み 1 組を 200 回読む温の連鎖では、
  並列度の足りない自前カーネルが DRAM レイテンシを隠せない負けが見えない (HC 融合: 温 +13 us、冷 +78 us/回、2026-09-03 に帰属)。
- 無効化の積み上げで部品時間を見積もらない (ablate の積算は過大評価の前例あり)。
  部品計測をしたら「部品和 ≈ 壁時計 (数 % 以内)」を必ず確認する。
- GPU の仕事は全部 `tools/biglock.sh` で直列にする。親の A/B (run_chainNN.sh の列) が 1 本終わったら、
  エージェントの GPU 待ちを全部先に流してから次の A/B に進む。エージェントどうしでは、モデルを読まない
  micro (`_micro.py` / `verify_*.py` / `smoke_*.py` / 連鎖ツール) を in-model A/B より先に通す。
  biglock がコマンド名と祖先 (run_chain) で 3 段を自動判定する。`BIGLOCK_PRIO=0/1/2` で明示。
- ベンチのプロンプト池は `bench/textpool-frozen.txt` (凍結)。池の元はリポジトリの docs そのものなので、凍結しないと docs を編集した瞬間に
  走行をまたぐ比較の prompt が変わる (2026-09-03 に実際に起きた)。`--freeze` で作り直すのは基準を取り直すときだけ。
- 計測中にダウンロード・別 GPU プロセスを並走させない (335GB ダウンロード並走で
  decode が 21.4 tok/s に落ちた実測がある)。
- 生成長を揃えて比較する (相手 121 トークン vs 自分 512 トークンの窓で
  偽の同着を出した前例)。判定基準は測る前に宣言する。
- サーバーログの phase/round の draft/verify 内訳は帰属が歪んでいる
  (next_drafts の先行投入のため)。合計 ms/round だけ信じる。

## 触ると壊れるもの

- `_vendor/qwen4_exp.py` のシーム (`Attention._positions` / `_final_mask`、
  `GatedDeltaNet._store_conv_state`、`PLELayer._store_short_conv_state`、
  `Qwen4ExpModel._make_masks` / `_store_ngram_ctx` / `_prelude`、
  `DecoderLayer.pre_mlp` / `_combine`) は `mlxturbo/batch.py`、
  `mlxturbo/batch_spec.py`、`mlxturbo/spec_flash.py` の呼び出し口。
  **引数や返り値を変えるときは 3 つの呼び手を全部見ること。**
  差し替え側は 1 クラスあたり 2 個までに保つ。超えたらフックが内部構造を
  漏らし始めた合図で、そのときは写しに戻すほうが正しい (`docs/BACKLOG.md`)。
- `mlxturbo/spec_flash.py` の `_staged_forward` / `_group_prefill_forward` /
  `capture()` には、本家の制御フローの写しが残っている (段階投入のループ骨格、
  レイヤー主導の二重ループ、rollback 用の GDN 転記)。**本家の層まわりを
  変えたらここも見る。**変更後は `tools/vendor_fingerprint.py` (合成モデル、
  CPU、数秒) で一次検査し、prefill を触ったなら
  `tools/verify_prefill_bitident.py` (実モデル、4 分) までやる
  (詳しい対応表は `docs/BACKLOG.md` の「本家フォワードの写し 9 種の整理」)。
- `mlxturbo/spec.py` の `SpecEngine._hidden_forward` は site-packages の
  `mlx_lm.models.qwen3_5.Qwen3_5TextModel.__call__` の層ループの写し (27B/qwen3_5 側。
  2026-09-04 に `staged.py` を畳んで 1 本にした。capture=True の巻き戻し用の状態の控えと
  段階投入 (`MLXTURBO_SPEC_STAGED_VERIFY`、既定 on) を持つ)。**本家 (mlx_lm 更新時) を変えたらここも変える。**
  ゲートは `bench/test_spec_capture_module_qwen3_5.py` (合成、ビット一致) と実機の生成列の一致 (`tools/decode_ab_generic.py`)。
  `_linear_capture` (GDN 層本体の写し) は `MLXTURBO_SPEC_CAPTURE_MODULE=1` でモジュール呼び出しに替わるが、
  27B の実機で幅 5/7/8/9 の verify だけ数 ulp ずれる (原因未特定) ので既定は写しのまま。
- 「作って捨てる」遅延グラフを組まない。捨てる可能性のあるグラフは規模を
  問わず MLX の暗黙 eval に罰される (楽観先組みは 3 回失敗して棄却済み)。
- knob の既定値は 3 種類を区別すること。同じ「既定 off」でも中身が違う。
  `MLXTURBO_PIPELINE` / `MLXTURBO_MOE_GLU` / `MLXTURBO_WIDE` /
  `MLXTURBO_PREFILL_CHUNK` / `MLXTURBO_FAST_QMM` / `MLXTURBO_HC_PREFILL` は
  in-model 実測で負けたから off にしてある。有効化する前に棄却時の記録
  (`docs/research/DECODE-ANATOMY-2026-08-31.md`、
  `docs/research/KERNEL-BRIEF-DECODE-BW.md:283`) を読むこと。一方
  `MLXTURBO_MOE_VERIFY` (`mlxturbo/kernels/moe_verify_gather.py`) も
  2026-09-01 の in-model A/B で決着 (短 decode 3 本とも +46〜52% 遅い)。
  これも「有効化しない」側。逆に
  `MLXTURBO_PREFILL_GROUP=4` / `MLXTURBO_STAGE_EVERY=2` /
  `MLXTURBO_DRAFT_RERANK=1` / `MLXTURBO_HC=elem` (2026-09-03 20:20 に kernel から。decode 幅だけ、ビット一致) / `MLXTURBO_SORT_MIN=16`
  `MLXTURBO_GDN_DECODE_FUSED=1` (2026-09-03 21:05 から既定 on: decode/verify 幅の GDN の非行列積を 16 -> 3 dispatch、短 ms/round -2.4% / 17k -2.0%、micro でビット一致。`=0` で off)
  は本番の既定値そのもので、値を変えると本番の挙動が変わる。
  `MLXTURBO_GDN_METAL` (既定 on、`=0` で off) も同様に本番の既定値で、
  2026-09-02 の in-model A/B (17k prefill -1.3〜-4.5%、KLD +0.00014) で
  on にした。
  `MLXTURBO_DEPTH_ADAPT` (既定 on、`=0` で off) も本番の既定値で、2048 超の文脈で受理率 EMA から
  draft 深さを選ぶ (17k で ms/tok -3〜-6%、2026-09-03)。2048 以下は静的 depth 2 のまま。
  `MLXTURBO_PREFILL_ATTN` (既定 on、`=0` で off) は T=1 gather の prefill attention カーネルで、kv ≥ 8192
  (`MLXTURBO_PREFILL_ATTN_MIN_KV`、2026-09-03 17:20 に 12288 から下げた。交差点 8.1k、10k -0.8%、品質同一) だけ発火する (50k prefill -23%、17k -3.7%、2026-09-03)。長文脈の KLD (自前の dense
  比) 0.040 は、既に受理している GDN Metal の同じ物差しでの 0.111 より小さい。長文脈の品質は
  bf16 参照が無いので、この物差しではなく課題の正答率で見ること (CATCHUP 2026-09-03 07:40)。
  `MLXTURBO_MOE_COMBINE_FOLD` (既定 on、行数 ≥ 64 の prefill 幅だけ、`=0` で off) と `MLXTURBO_PRIME_WINDOW`
  (既定 512) も 2026-09-03 に A/B と KLD で入れた本番の既定値。
  `FASTMLX_NGRAM_BACKEND` は **既定 pread に戻した** (2026-09-03 16:15)。mmap は同じ機体で続けて測ると -6〜-7% に見えたが、それは前の走行が
  32 GB のサイドカーをページキャッシュに残していたため。冷えたキャッシュでは mmap のページフォルトが直列化して 4k prefill が 2.1 倍遅く、
  decode も -15% (小ベンチ 0903f)。本番 (モデル 98 GB + サイドカー 32 GB > 128 GB) ではキャッシュは冷えているのが普通。
  **I/O を含む経路の A/B は、同じ機体で続けて測るとページキャッシュを共有する** (プロセスを分けても同じ)。
  `MLXTURBO_NGRAM_PREFETCH` (既定 1、`=0` で off) と `MLXTURBO_NGRAM_PREFETCH_AT` (既定 early) は 2026-09-04 02:25 に入れた本番の既定値:
  pread では先読みが走っていなかった (既定 off だった) ので on にし、投入位置を group forward の前へ (最初の境界は前景で読み切る)。
  17k 冷 -2.9% / 4k 冷 -1.0% / 17k 温 -1.8%、出力一致、行キャッシュは 1 prefill ぶん (≤ 70 MB)。`FASTMLX_NGRAM_NOCACHE=1` は計測専用 (`F_NOCACHE`) で、
  I/O を含む A/B の「冷」はこれで揃える。
  `MLXTURBO_QSA_TAIL` (既定 query、`=global` で旧規則) は QSA の端数 (tail) の可視範囲をクエリごと `[cr*floor((q+1)/cr), q]` にする本番の既定値
  (HF / mlx-serve / oMLX と同じ。global は prefill 幅の 3/4 の行が自分を見ていなかった。速度は中立、17k の正答率は recall 12/12、2026-09-03 17:25)。
  decode の QSA カーネル `MLXTURBO_QSA_DECODE_KERNEL` (既定 on、`=0` で off) は query が前提。K2a 選択 + K2b attention で本番の並びとビット一致、
  17k で ms/round -4.1% (burn-in 付き、2026-09-03 18:15)。50k は確認中。MTP の draft 層には当たっていない (enable の順序、0.1 ms/round)。
  `MLXTURBO_PREFILL_TAIL_IN_GROUP` (既定 on、`=0` で off) は末尾 2048 チャンクを layer-major のグループに入れる本番の既定値
  (4k -4.9% / 8k -3.4% / 17k -0.6%、サーバー経路はビット一致、2026-09-03 15:35)。checkpoint 無しの経路 (generate() / ベンチ) では末尾を 2047+1 に割るので丸めが動く。
  `MLXTURBO_MOE_GEMM_MIX` (既定 48、`=0` で off) と `MLXTURBO_QMM_WIDE` (既定 auto = 非 NAX で on、`=off`) は 2026-09-03 14:00 に
  8k の in-model (混合タイル -4.3%、BM=64 dense 射影 -2.6%、どちらも素とビット一致) で入れた本番の既定値。NAX 機では
  `MLXTURBO_MOE_GEMM=auto` の判定でどちらも off になる (自前カーネルは NAX 機で auto=off の方針)。
  `MLXTURBO_MOE_DECODE_FUSED` (既定 auto = 非 NAX 機で on、`=0` で off、行数 B*S ≤ `MLXTURBO_MOE_DECODE_FUSED_MAX_ROWS`=4 だけ) は 2026-09-04 03:05 に入れた本番の既定値:
  decode / verify 幅の MoE を gate/up 融合 + down の 2 カーネル (rmax=1、ソート無し) に。短 ms/round -1.2〜-1.3% (2 回) / 17k -1.4%。**非ビット一致**で、本番重みの
  逆量子化 fp32 参照テストでは自前が素より近く (中央 0.39 倍、反転 0/96)、S=1 の Δ KLD は +0.00036 (対 bf16 teacher、幅 +0.0005 の中)。バッチ検証 (rows ≥ 6) は素に落ちる。
  `MLXTURBO_HC_QMM_WIDE` (既定 auto = `MLXTURBO_QMM_WIDE` と同じ NAX 判定、`=0` で off) は 2026-09-04 02:55 に入れた本番の既定値: HC の細長 GEMM
  (10240→320 / 320→10240) を BM=64 タイルに (ビット一致、4k ±0 / 8k -0.4% / 17k -0.9%)。elem 変種の prefill 幅拡張は非ビット一致で tok/round -4.8% だったので落とした。
  `MLXTURBO_SMALL_M_ROUTE` (既定 auto = 非 NAX 機で `small_m`、`=off` で素、`=nocap` / `=mma` で固定) は 2026-09-04 13:05 に入れた本番の既定値:
  qwen3_5 (27B) の verify 幅 M=2..5 の量子化 dense 射影 (N, K ≥ 1024) を `kernels/qmv_small_m.py` に (各行が MLX の qmv とビット一致、fp32 参照への距離は素と同じ)。
  27B in-model (512 × 短 3 本 × 2 回文 + 4k、`ab-smallm-3way-*-0904.json`): small_m 短 -1.9% / 4k -4.6%、nocap 短 -1.9% / 4k +0.8%、mma 短 +3.1% / 4k +5.2% (fp32 から素の 1.3〜1.7 倍遠い)。
  Flash-Next は `dispatch_scope` を通らないので未接続 (当てるなら fused.py に enable が 1 つ要る)。bits=8 / group_size≠64 は素に委譲。
  `MLXTURBO_DRAFT_RERANK` (既定 on、`=0` で exact head) は Flash-Next に加え、qwen3_5 の
  q4 affine / group64 headでもMTP proposalだけをq2全語彙top-32 + q4候補行再採点へ替える。
  target verifyと通常headはexactのまま。27Bはq2 recall@32 100% (1,677 proposal)、短 -2.0% /
  4k -1.6% / 17k -3.0% ms/tok。常駐378.9 MiB、起動時temporary約2.37 GiB、構築0.08s。
  17kは受理率 +6.6%で生成列が262 token目から丸め分岐したため、exact出力が必要なら`=0`を使う。
  `MLXTURBO_SDPA_ROWTILE` (既定 256、`=0` で off) も本番の既定値: head_dim 256 の sdpa は MLX の fallback でタイルを飛ばさないので、
  prefill の dense 経路で q を 256 行ずつに割り前方の K/V だけ渡す (4k -1.1% / 8k -1.2% / 17k -1.0%、KLD 差 0.0、2026-09-03)。
  `MLXTURBO_SDPA_SPLIT_GENERIC` (既定 auto = 非 NAX 機で on、`=0` で off) は 2026-09-04 13:15 に入れた本番の既定値: qwen4_exp 以外の族 (27B など) の decode / verify 幅 (1 < S ≤ 16) で
  S×gqa > 32 のとき sdpa をクエリ幅で割って MLX の vector カーネル (走査和が fp32) に戻す。27B 短 -1.6% / 17k -1.2% (幅 6 の round だけ -17%) / 4k は発火せず同一、fp32 参照への距離は素の 0.53 倍 (壁の向こうの fallback は中間が bf16)。
  qwen4_exp は従来のシーム `enable_sdpa_split` の担当 (bool マスクを実体化する変種。K/V を切る変種の方が冷 micro で 13〜18% 速い → BACKLOG)。
  `MLXTURBO_MOE_COMPILE` (既定 auto = qwen4_exp の `SparseMoeBlock` があるときだけ on、`=0` で off、行数 ≤ `MLXTURBO_MOE_COMPILE_MAX_ROWS`=16 の decode / verify 幅だけ) は 2026-09-04 13:25 に入れた本番の既定値:
  MoE ブロックまるごと (router の f32 化 → argpartition → softmax → take → 共有専門家のゲート → combine) を `mx.compile` で 1 グラフに。行列積は境界のまま、op の並べ替えは無いので**ビット一致**。
  短 -1.1% / 17k -0.6% (dispatch -7.7%)、起動時に 48 層 × S=1..4 の 192 グラフを空焼き (0.22 s、`MLXTURBO_MOE_COMPILE_WARMUP=0` で切る)。prefill 幅は取り分が無く (17k ±0、短 +26%) 端数チャンクの形が要求ごとに変わって Compiled が溜まるので包まない。
  層まるごと / GDN / attention の compile は原理的に不可 (closure の配列を定数に焼き、python の副作用はトレース時の 1 回だけ。`shapeless` はこのモデルでは全滅)。
- 品質を売って速度を買わない。fake を実物より緩くしない。KLD の受け入れ幅は
  現行比 +0.0005 (bench/quant_eval.py compare)。
  **decode 幅 (行数 ≤ 8) にしか発火しない非ビット一致のカーネル**は、prefill 幅で走る通常の compare では数値が 1 度も通らない。既定に入れる条件は 3 つ
  (2026-09-04、fused:1 で確立): (i) 本番の重み・routing での逆量子化 fp32 参照への距離が素以下 (`tools/moe_decode_fused_ref_model.py` の型、反転する層が 0)、
  (ii) `bench/quant_eval.py compare --fusions --step 1` (S=1 で 1 トークンずつ) の on/off 同一条件の Δ KLD ≤ +0.0005。
  **決定性 (検証幅で丸めが変わらないこと) は条件にしない** (ユーザー 2026-09-04 12:40: 量子化モデルに決定性を求める人はいない。品質に影響が無ければよい)。決定性は A/B の計測が楽になる性質としてだけ扱う。
  S=1 の基準は `bench/results/quant-eval/compare-step1-base-0904.json` の 0.01796。
  **本番のパックは lm_head も 4bit (`~/models/ddalcu-mlxlm-head4`、真 bf16 から g64 で焼いたもの) にした** (ユーザー 2026-09-03 18:20。
  相手の一律 4bit と条件を揃える。代金は KLD 0.01326 → 0.01794 (+0.0047)、top-1 一致 0.966 → 0.962 で、これは了承済み)。
  以後の KLD の基準は `compare-head4-baked-0903.json` の 0.01794。8bit 頭のパック (`~/models/ddalcu-mlxlm`) は旧基準の A/B の続きにだけ使う。
- **代金ゼロ (品質・メモリ・複雑さの増分が無い) の改善は、取り分が 1% 未満でも既定に入れる** (ユーザー 2026-09-03 17:20)。
  条件は「測った文脈のどれでも遅くならない」こと。効果が薄いだけを理由に畳まない。畳むのは代金がある (品質、メモリ、遅くなる文脈がある、写しが増える) とき。

## 分業

実装・検証はサブエージェントに出してよいが、計測の判定と commit は親が行う。
HF トークンは huggingface_hub の既定の場所のみ。引数・ログ・環境変数に出さない。
