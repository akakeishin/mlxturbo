# 次のセッション用プロンプト (2026-09-03 深夜版。Fable 親、実装は Sonnet)

以下をそのまま新セッションの最初の入力にする。

---

## 依頼

mlxturbo (`/Users/ht/dev/fastmlx`) が対戦相手 mlx-serve (`~/dev/mlx-serve`) に
**冷 prefill と短〜中文脈の decode で負けている**。多日レーン (`docs/research/LANES-2026-09.md`)
のゲートを上から回し、差を埋める。ANE は M5 前提で本格化する方針、dflash-mlx / vllm-mlx は
調査済み (同ファイル)。並列デコード (continuous batching) も対象。

**読む順**: `docs/research/LANES-2026-09.md` → `docs/research/SESSION-2026-09-02-CATCHUP.md`
(本日の実測の全記録) → `docs/research/REVIEW-2026-09-02-INDEPENDENT.md` (独立レビュー。
受理率を落とす MTP キャッシュの欠陥 A-1 と壊れた契約テスト 2 本はレーンより先に直す) →
`CLAUDE.md` の計測の作法。

## 現在地 (2026-09-03 09:10、既知を片付けた基準。`bench/results/self-snapshot-turbo-small-0903c.json`)

冷 prefill 4k 6.89 / 17k 31.0 / **50k 94.2 s** (相手 5.77 / 29.2 / 108 → 1.19x / 1.06x 負け / **0.87x 勝ち**)、
温 TTFT 0.13〜0.31 s (相手の 4〜6 倍速)、decode 51.5 / 47.5 / 43.4 tok/s (相手比 -7% / -15% / +41%)。
朝 (9/2) からの合算: 冷 prefill -17〜-42%、温 TTFT 3 倍速、decode +6〜+26%。

## 以前の現在地 (2026-09-02 夜。フルテストの最新値は `bench/results/self-snapshot-*-0902f*.json` と
CATCHUP 末尾の表。冷えた状態のフルテストは実装を直してから「ちびちび」取る方針)

前セッションの「prefill 2 倍差」はハーネスの産物だった (接頭辞キャッシュ、thinking 不一致)。
揃えた実差: 冷 prefill 1.3-1.5 倍負け、decode 短文脈 12% / 17k 27% 負け (bool マスクで
17k は -12.8% 改善済み)、温 TTFT 2 倍勝ち、50k は decode も勝ち。

## 夜に決着したこと (全部 CATCHUP に数字あり)

- レーン 2 (MoE の gate/up 連結) は D-2 修正後の再測で 17k prefill +62.6% 悪化。**棄却で確定。**
  タイル水増しは 1.40 倍 (`tools/moe_routing_skew.py`)、残り 1.5 倍は MLX の gather_qmm の効率。
- レーン 3 (QSA ブロック疎 prefill カーネル) は T=4 で真の union が kv の 36%、FLOP は dense の
  70% 止まり。**ゲート未達で畳んだ** (設計書 `QSA-PREFILL-KERNEL-DESIGN.md` は記録)。
- レーン 1 (decode +6 ms): カスタム Metal カーネルの起動固定費は組み込みと同じ 3 us。
  「固定費が高い」仮説は棄却。残るのは「自前の融合カーネル 1 回の実行がデータ量に対して遅い」
  仮説で、`tools/kernel_chain_cost.py` で HC / GDN step / prework の 1 回の費用を出す。
- MTP キャッシュに受理トークンを積む (A-1) は 17k で -1.5%、短文脈 ±0 → 既定 on 維持。
  17k の draft depth は 1 が 2 より -8.9% → 方針維持。gather attention は 25k でも発火せず。
- batch-spec の QSA 欠陥 B-1/B-3 は修正済み (レーン 5 の前提)。
- ベンチ公開の設計は `docs/research/BENCH-DESIGN-2026-09.md` + `bench/suite/`。池 6 種、
  出力長 128/1024、thinking off/on、tier 3 段 (quick 2 h / standard 1.4 h / overnight 4.1 h)。
  実走はまだ。

## 今日入れた既定の変更 (全部 A/B と KLD 済み)

- QSA マスクを bool のまま sdpa へ (17k decode ms/tok -12.8%、出力不変)
- oMLX 移植の GDN blocked-seq Metal (prefill -1.3〜-4.5%、KLD +0.00014)、`MLXTURBO_GDN_METAL=0` で戻る
- n-gram 行取得のバッチ pread (ビット一致)
- 端数チャンクのグループ化 (差なし、害なし)
- MTP キャッシュに受理トークンを積む (`MLXTURBO_MTP_CACHE_APPEND=1`)

却下 (既定 off のまま): n-gram 先読み、fast rope、GDN 前処理融合、RMSNormGated 融合、
MoE router 融合、union gather (真の union が 6 割)、wide 連結。理由と数字は CATCHUP。

## 深夜 (2026-09-03 01:50) に決まったこと

- 小さい結果 (mlxturbo だけ、冷、反復なし): 冷 TTFT 4k 7.18 / 17k 32.5 / 50k 119 s、decode 50.6 / 48.7 /
  42.4 tok/s。対相手 (朝の値) prefill 1.24 / 1.11 / 1.10x 負け、decode 4k -8% / 17k -13%、50k +38%。
- HC 融合カーネルは 97 層中 1 層しか発火していなかったが、97 層で発火させると +8 ms **悪化** (効率が低い)。
  HC の真の費用は S=1 で最大 4.7 ms (スタブ比較)。書き直し (1 回 80 → 20 us) がレーン 1 の手。
  `MLXTURBO_HC_INJECT_BF16=1` で 97 層発火を試せる (既定 off)。
- sdpa の幅 2 分割は 8/31 から本番に入っていた (新しい取り分ではない)。knob 化だけした。
- prefill attention 1 層の内訳 (kv=16.9k): sdpa 75%、indexer 6%。dense は計算上限。T=1 gather カーネルだけが手。
- レーン 9 (forward の構造) とレーン 10 (投機の tok/round) を追加。`DepthController`
  (`MLXTURBO_DEPTH_ADAPT=1`、既定 off) と `MLXTURBO_REBIT=head=4` (速度比較用) を配線済み。A/B は連鎖待ち。
- レーン 5: 非投機の継続バッチングを毎 tick の途中参加に (コミット済み)。ゲート計測 `bench/parallel_join_gate.py` は未実走。
- ユーザー方針: Qwen3.8 Flash だけ見る。overnight tier は指示があるまで走らせない。多日レーン → フルテストの順。

## いまの段取り (2026-09-03 08:00 時点。何をやっているか見失ったらここを読む)


## 現在地 (2026-09-03 19:25、push 済み c09d121)

小ベンチ 0903g (lm_head 4bit): 冷 prefill 4k 5.95 / 17k 30.3 / 25k 43.5 / 32k 54.3 / 50k 86.7 s (相手 mlx-serve 比 1.04〜1.09x 負け)、
decode 51.7 / 47.5 / 47.5 / 44.8 / 47.0 tok/s (相手比 -7 / -3 / -22 / -5 / +2%)、温 TTFT 0.14〜0.80 s (相手の 4〜20 倍速)。
走行中: decode の糊 4 つ (sort 無しルーティング、copy、elementwise の compile、wide 再測)、K2c の S 依存確認 (品質側)、プロンプト池の凍結、サーバー経路と decode_ab の 17k prefill の差 (+15%) の切り分け。

## 方針 (ユーザー 2026-09-03 20:45): 当分ベンチは取らず、改善点探しに徹底的に

小ベンチ / フルベンチは既定が増えたときだけ (基準は 0903g)。いまは診断を先に: decode のモジュール別 × 行数別の GPU 時間 (S=1 / S=3、行 1 つ = round の 16% の正体) と、
prefill の内訳の取り直し (4k / 8k / 17k、今日の既定)。診断の「次の的 3 つ」から実装に入る。

## 数日の集中 (ユーザー 2026-09-03 20:05): decode の tok/s

**結末 (22:00)**: 糊の融合は GDN (-2.4%、ビット一致) だけが勝ち、HC elem ±0、HC 書き戻し +0.6%、MoE router〜combine +0.05% (ビット一致の対照で dispatch -5% → ±0.0%)、MoE の重複まとめは帯域で負け。
**dispatch の本数の値段は稼働率 96% では実質ゼロ**、行列積は 430 GB/s でピーク。短文脈 decode は MLX の上では 52〜54 tok/s が天井 (相手 55.7)。残りは MoE fused:1 の結果だけ。
その後は 27B レーン (部品の置き換え) と prefill の 3 本 (n-gram 先読み、HC 読み、小 kv の attention) に移る。

短文脈の tok/s に数日集中する。壁は dispatch の床 (5 us × 4499 本/step、Lily は 795 本) なので、**層の塊ごとの大きい融合で本数を 1/3 に**:
1. MoE decode (ルーティング + gather + gate/up + SwiGLU + down + combine を 3 本程度、行間で専門家を共有、並列度は MLX の qmv 以上) — PoL 走行中。
2. GDN の層まるごと (前処理 + 再帰 + norm。prework の負けは並列度不足だったので直して再挑戦)。
3. HC の pre / post (decode 幅ならビット一致の elem 変種を土台に)。
4. その後: 受理率 (draft の top-k 命中率 → rerank / depth 4)、MTP の draft 層に K2c 等を当てる (enable の順序)。
判定は 1 プロセス ABBA (burn-in、depth 固定) の短 / 17k の ms/round、head の一致。冷連鎖 micro は重みを 100 MB 超巡回。「本数が減ったのに遅い」ときはカーネルの並列度を疑う。
**訂正 (20:25)**: HC elem で dispatch を -776/step (17%) 減らしても壁時計は 0〜-0.7%。「本数の床 5 us × 本数」の読みは過大で、消えるのは起動の一部だけ。
取り分は「融合した先のカーネルが高い並列度で走ること」に懸かる (HC の冷 micro: 素 62 → 融合 53 us = -9 us/呼び出し止まり)。判定線は本数ではなく冷 micro の us と in-model の ms/round。
prefill も並行 (ユーザー 20:10「prefill もやる」): (a) サーバー経路の +15% (17k 30.3 対 decode_ab 26.4) の切り分けと修正 (最大の的、切り分け中)、
(b) P7 の残り (router 0.82 + sort/topk 0.65 + swiglu 0.40 = 1.9 ms/層 ≈ チャンクの 3%)、(c) 8k〜12k の attention (MIN_KV 8192 で gather に寄せた。dense 側の行タイルとの交差の再確認)、
(d) MoE GEMM の残り (mix48 は dense 比 1.04〜1.24。r=40 の末尾チャンクで 24% 損、専門家あたりの行数が少ない形の効率)。27B / 動的判定はその後。

## push 後の流れ (2026-09-03 17:40 に決めたもの。上から順)

1. **エージェントの計測を判定して既定化** (代金ゼロ方針): GDN レジスタ常駐 scan (in-model 8k / 17k で遅くなければ)、P7 第 2 段 (combine / router / sort の融合、8k -2% 狙い)、
   D1 draft 同梱 (burn-in 付き再測で遅くなければ)、K2c (天井 13% との差 1.4 ms/round を切り分けてから、50k も確認して既定 on)、天井スタブの再走 (draft の費用、oracle の受理率の天井)。
2. **decode の本格改修 (Lily の 4)**: decode 1 step の trace (カーネル数、GPU 空き率) を見て決める。
   空き率が大きければ「カーネル数を減らす融合を帯域最適に書く」(消費の大きい順)、小さければ帯域の壁 (lm_head 8bit、PLE の参照、MoE の専門家読み) を削る。
   その後 depth 4 の既定化 (K2c で行の費用が下がった後、oracle の天井 tok/round を見て rerank と組む)。判定は短 / 17k / 50k の ms/tok。
3. **P9 チャンク 8192** (prefill): P7 の後。query 化で可視集合はチャンク割りに依存しない (前提は済)。判定は 8k / 17k / 50k の prefill_s と温 TTFT (checkpoint の粗さ)。
4. **小ベンチ → フルベンチ (対 mlx-serve)** (ベンチは lm_head 4bit のパック `~/models/ddalcu-mlxlm-head4` で。相手と条件を揃える、ユーザー 2026-09-03 18:55): 2 と 3 で「decode 短文脈が同着以上、prefill 1.03x 以内」が小ベンチで出たらフルベンチ。出なくてもユーザーが呼べばフルベンチ。
5. **27B / 35B-A3B (qwen3_5)**: 素の数字 (mlx-lm / mlx-serve / oMLX / rapid-mlx / うち) → **部品ごとの置き換え** (フォールバックではなく、契約が合う部品だけを差す方式。BACKLOG「決定 (18:55)」) → GDN Metal / sdpa 行タイル / BM=64 qmm (MLP 込み) の移植 → Lily の 5 (GQA packing、固定ブロック attention) → 6 (35B-A3B の AR 対 MTP)。
   teacher (27B の bf16、54 GB) もここで作る。
6. Gemma 4 (assistant drafter の KV 共有エンジン、sliding window の prefill)。
7. 優先度最低: Flash-Next の teacher (bf16、query) の作り直し (SSD)。

各段の commit / push: 既定が増えるごとに commit、小ベンチの記録ごとに push。

**優先度最低 (ユーザー 2026-09-03 17:35)**: KLD の teacher (bf16、query) の作り直し。SSD (`/Volumes/Mobile SSD`) が読めるようになってから `bench/teacher_bf16.py --src <bf16 dir> --continuations bench/results/qe-cont.json --out bench/results/qe-ref-bf16.npz` を `MLXTURBO_QSA_TAIL=query` で (251 GB 読み、約 10 分)。それまで品質の判定は課題の正答率で。

GPU は `tools/biglock.sh` で 1 本ずつ直列。親の連鎖スクリプトは scratchpad
(`/private/tmp/claude-501/-Users-ht-dev-fastmlx/65b31683-391c-444c-b255-622b126131f9/scratchpad/run_chainNN.sh`、
いま 80 番台) にあり、前の連鎖の終了を `pgrep -f run_chainNN.sh` で待って順番を付けている。

案出し Studio (Scout 5 本 + decode advisor) の凍結ポートフォリオと落とした的は `docs/research/IDEAS-2026-09-03.md`。
**最終ゴール (ユーザー 11:35): 短文脈で 100 tok/s。MTP 頭の学習はしない。** 算数は IDEAS の「最終ゴール」の節 (S=1 forward 17 ms 以下、行 2.5 ms 以下、depth 4)。
prefill は 1.5 倍 (相手の 1.2〜1.3 倍速) が現実線。ANE は見送り (INT8 の重みコピーでメモリが増える)。M5 Max / Ultra の机上見込みも IDEAS に。
ユーザーの Commit (07:50): 選抜は親に任せる。**カーネル (P3) はフルベンチの前に書く。**
ユーザー方針 (09:15): **decode より先に prefill をもっとやる** (attention は長文脈有利のままでよいが、まだ手があるはず)。
小さいベンチは **prefill の深掘り (GPU 稼働率トレース、attention、MoE カーネル、融合小物の 1:1) が終わってから** 1 回 (6 文脈)。decode 案はその後 (ユーザー 09:25、10 時の途中ベンチは取り下げ)。
A/B は `--knobs a,b,c` で 1 プロセスにまとめ、プロセス内 ABBA では冷却の休止を入れない (別プロセス比較だけ休止)。
フルベンチは小さいベンチで「decode 同着以上、prefill 1.03x 以内」が出てから。

順番:
1. 測定だけ (今日)。
   - 済: 段階投入の切り分け (完全直列 +16.2% → 構築 6 ms 級、decode は GPU 律速。層単位 compile は先頭に入れない)。
   - 済: HC の elementwise を prefill 幅で compile (-0.1%、畳む。knob は既定 off で残す)。
   - 済: PLE の mmap + 背景 madvise (8k prefill -6%、別プロセス 2 巡)。17k の確認 (chain82) が通ったら
     `FASTMLX_NGRAM_BACKEND` の既定を mmap、`MLXTURBO_NGRAM_PREFETCH` の既定を on にし、CLAUDE.md の knob 段落に足す。
   - 待ち: D2 の trace (depth を受理履歴 / draft マージンで決める案の判定。Sonnet が仕掛けを実装中、親が biglock で 1 回流す)。
   - 待ち: D3 (文脈 n-gram の draft) のオフライン集計。生成トークン列が保存されていなかったので `--save-out` を足してから 17k / 50k を 1 本ずつ。
   - 待ち: D4 (command buffer の粒度、env のみ) を chain83 で A B B A。
   - 未: サーバー経由の 1 リクエストで `[round]` trace を取り、decode_ab の 43 ms との差を見る (detokenize / SSE の GPU 遊び)。
2. 済 (08:28): モデル無しの proof-of-life。P1 (2 stream) は直列化 0.965 で畳む、D5 (専門家共有) は重複行が既に安いので畳む、
   D4 (command buffer 粒度) は ±1% で畳む。既製 gather_qmm に 16 行揃えのダミー行を足すと r=40 -11% / r=160 -3.6% (P3 の variant C)。
3. 済 (10:20): D1 (+1.7%、隠れていた draft を壁に出しただけ)、D2b (1 ラウンド遅れの信号は AUC 0.567)、D3 (17k -2.1% / 50k -2.6%) はいずれも畳んだ。
   **最優先 (品質)**: QSA の tail の意味論が HF と違う (HF はクエリごとに自分の未完成ブロックを可視、うちは global tail)。`MLXTURBO_QSA_TAIL=query` を
   実装中 (Opus)。通ったら既定にし、teacher (bf16) を作り直して KLD を取り直す。verify の受理率にも効く可能性。
   decode 側: 天井スタブ (chain89、`--knobs`) → K2 (radix select K2a を実装中、K2b は 2-pass vector の写し)。
   prefill 側: P5 行タイルは **既定 on で確定** (17k -1.0%、KLD 差 0.0、`MLXTURBO_SDPA_ROWTILE`)。P6 (uint4 load) は micro 通過
   (56.4 → 40.8 ms/層、ビット一致、交差点 8.1k) → 本番カーネルへ移植 + 17k / 50k in-model + 下限 8192 の品質ゲート (Opus 実施中)、
   `tail-in-group` (末尾チャンクをグループに、4k -3〜5% 見込み、A/B 待ち)、BM=16 の segmented (micro 待ち)。K1 の mask arm は後回し。
   **custom kernel が in-model で負ける理由の調査** (ユーザー依頼、Opus 実施中): 仮説 5 つ (per-call CPU 費用 / command buffer 分断 / 段階投入との相性 /
   非連続入力のコピー / 出力割り当て) を micro と in-model の 2×2 (HC 変種 × STAGE_EVERY 2/0) で切り分ける。
4. P3: MoE の grouped GEMM。in-model は 4k ±0 / 8k -1.4% / 17k -0.6% (換算率は 8k で予測どおり。MoE 行列積は prefill の 3 割)。
   **ユーザー方針 (11:20): P3 と P10 (BM=64 の自前 qmm) は M3 Max 向けに入れる** (非 NAX で auto=on、NAX 機では off)。P3 は BM=16 / 混合版の micro →
   pad16 の 22.6 ms/層に届けば in-model → 4k -3% 以上で既定。P10 は micro (qmm の 0.87 倍以下) → `enable_wide_projections` の口で in-model + KLD。
   prefill の追加レーン (台帳 P7〜P9): P7 MoE の行列積以外 800 ms/チャンクの内訳 (`tools/moe_split.py`、chain93)、P8 dense 射影の bf16 GEMM
   (`tools/dequant_gemm_micro.py`、chain92)、P9 チャンク 8192 (P7 / P8 の後)。目標は prefill 1.5 倍、decode 2 倍 (K2 + HC + depth)。
   第 3 段 (fused.py のフック + knob `moe-grouped-gemm` の A=segmented / B=既製 / C=pad16+既製、4k/8k/17k) を Opus が実施中。
   判定線: 17k -2% 未満かつ 4k -4% 未満なら BM=16 経路 (専門家ごとに BM を選ぶ) を足してから再 A/B。KLD はビット一致なので出力一致で代替。
   **NAX 対応機 (M5 系) でも使う** (ユーザー方針 08:30): カーネルは NAX 専用 intrinsic を使わず、`MLXTURBO_MOE_GEMM=auto|on|off` で
   NAX 機では auto=off (MLX の NAX カーネルとの A/B を NAX 機で取り直すまで)。この機の数字は全部「非 NAX」の数字。
5. 小さいベンチ (mlxturbo だけ、冷却 10 分) → 報告 → フルベンチ (mlx-serve は origin/main と同じで再ビルド不要) → 27B / 35B-A3B。
   27B / 35B-A3B では mlx-serve だけでなく **mlx-lm (mlx_lm.server) と oMLX** とも比較する (ユーザー方針 2026-09-03 08:10)。
   Flash-Next は相手が mlx-serve だけでよい (mlx-lm は Flash-Next の MTP を持たない)。

これまでの判定 (全部 CATCHUP に数字あり): 既定に入れたものは CLAUDE.md の knob 段落、畳んだものは IDEAS の「落とした的」と
LANES-2026-09.md。小物をかき集めた小さいベンチ (`self-snapshot-turbo-small-0903d.json`): 冷 prefill 4k 6.60 / 17k 31.3 /
50k 89.9 s (相手 5.70 / 27.8 / 82.1 → 1.16x / 1.13x / 1.09x)、decode 51.9 / 46.1 / 42.7 (相手比 -7〜-10%)。

判定と数字は必ず `docs/research/SESSION-2026-09-02-CATCHUP.md` の末尾に節を足して書く。既定を変えたら
CLAUDE.md の knob の段落も直す。フルテスト (対 mlx-serve) と overnight tier はユーザーの指示があるまで走らせない。

## 計測の節目 (ユーザー方針 2026-09-03 05:30)

1. **小さいベンチ (mlxturbo だけ、mlx-serve は測らない)**: 既知の未着手項目 (上の待ち行列 1〜7) を片付けた
   時点で 1 回。`bench/self_snapshot.py --model ... --ngram ... --ctxs 0,4000,17000,50000 --tokens 256 --reps 1
   --thinking off` (冷えた機体)。前回の小さい結果 (2026-09-03 01:40) と並べて、既定に入れた分の合算を出す。
2. **A/B の仮説検証** (相手がやっていること・やっていないこと両方、レーン 11 の 8 段目) をその後に続ける。
3. **フルベンチ (対 mlx-serve)**: A/B が落ち着いた時点で。**mlx-serve は最新版を取り直してビルドし直す**
   (`git -C ~/dev/mlx-serve pull` → `scripts/build-mlx.sh` / `zig build`、`--mtp` の既定や knob の変更を
   `[spec-stats]` のログで確認)。冷えた機体で 0/4k/17k/25k/32k/50k、必要なら 100k を 1 本。
   **overnight tier はやらない** (ユーザー方針 2026-09-03 08:00)。フルベンチで課題が見つかったら、もう一度
   仮説検証に戻る。
4. **フルベンチの後**: Qwen3.8-27B 4-bit と Qwen3.6-35B-A3B 4-bit を検証に加える (量子化ビットは揃える)
   → 27B / 35B-A3B の入口で **Lily の知見 (`docs/research/EXTERNAL-PERPLEXITY-LILY-2026-09.md`) の 5 と 6** をやる (ユーザー 2026-09-03 16:40):
     (5) GQA packing (4 head で KV 行を共有) と 32K 以上の固定ブロック attention (dense の族で 32K +7.7% / 128K +40%。MLX の sdpa_vector の GQA 共有の実態を先に確認)、
     (6) 35B-A3B は AR 対 MTP を最初に測る (Lily は別モデル drafter で -18%。うちは MTP 頭なので事情が違うが、verify 行が違う専門家を読む増分は同じ)。
   → 27B を 2 族目として載せるときに **アーキテクチャの対応表を切る** (`docs/BACKLOG.md` の「アーキテクチャ追従の投資」、
   ユーザー方針 2026-09-03: 全モデル最高ではなく 8〜9 割の追従が既定で得られる状態を目指す。qwen4_exp の最適化が終わってから)。
   まず Qwen 系を仕上げる。Gemma はその後。

## 残っているレーン (順に)

1. レーン 1: HC カーネルの書き直し (連鎖 ≤ 20 us/回、in-model で -2 ms 以上)。
2. レーン 10: depth-adapt の A/B (`decode_ab --knob depth-adapt`、短文脈 + 17k)。
3. レーン 3: T=1 gather prefill カーネル (多日。ゲート: 合成で dense sdpa の 2 倍、kv=16.9k S=2048)。
4. レーン 9: 層単位 mx.compile、attention 層の射影・indexer の op 整理。lm_head 4-bit は真 bf16 から焼き直す。
5. レーン 5: `bench/parallel_join_gate.py` の実走、batch_spec 側の途中参加の判断。
6. レーン 8: 温 TTFT の生成前 0.3 s (`--log-level debug` の `[ttft-trace]`)、decode の kv 罰 (`qsa_prefill_split --chain`)。
7. フルテスト (対 mlx-serve、冷) は多日レーンの後。overnight tier はユーザーの指示で。

## 守ること (CLAUDE.md に加えて、今日踏んだ罠)

1. **熱**: GPU を 50 分回すと 17k prefill が 37 → 57 s。絶対値は冷えた最初の 1 本だけ。
   A/B は回文順でも 1 本目が暴走するので、**2 本目以降で判定**する。
2. **残留プロセス**: 計測ツールが終了時に固まって Metal のメモリを握る (直したが、
   連鎖では各ジョブ後に `pkill -f "decode_ab.py --knob"` と `sysctl vm.swapusage` を確認)。
3. **発火**: 融合の A/B は `発火` の表示を見てから読む (GDN 前処理は bf16 の A_log で
   落ちていた)。
4. **相手のログを読む**: `--log-level debug` の `[prefill-trace]` と `[spec-stats]`、
   `[hot-cache] reused` で cache hit と thinking を確認する。
5. **xctrace は MLX の GPU 区間を拾わない**。カーネル比較には使えない。
6. **`~/models/ddalcu-ngram-sep` (RAM 常駐) は 17k で 41% 遅い**。長文脈は interleaved で。

## 使える道具 (今日足したもの)

| | |
|---|---|
| `bench/self_snapshot.py` | 窓を重ねない・thinking を揃える・ログを残す版 (`--thinking`, `--server-log`) |
| `tools/verify_width_cost.py` | 幅 S の verify forward 費用 (相手の fwd-ubench と同じ量) |
| `tools/forward_split.py` | build (CPU) / eval (GPU) 分離 + op 数 |
| `tools/decode_prof.py` | 部品別 (強制 eval、サイジング不可) |
| `tools/gather_union_stats.py` | QSA タイル union の真の大きさ |
| `tools/verify_gdn_metal.py` | GDN Metal の一致と単体速度 |
| `bench/quant_eval.py compare --fusions` | 融合 knob 込みの KLD |
| `tools/decode_ab.py --knob {bool-mask,gdn-metal,gdn-prework,fast-rope,fold-tail,ngram-*,gather-attn,wide,mtp-append,depth}` | |
| `tools/moe_routing_skew.py` | 層ごとの行数分布とタイル水増し率 |
| `tools/gather_union_stats.py --tiles 4,8` | QSA タイル union の真の大きさ |
| `tools/kernel_chain_cost.py` | 融合カーネル 1 回の費用 (直列連鎖) と forward 内の呼び出し回数 |
| `bench/suite/run.py --tier {quick,standard,overnight} --dry-run` | 公開ベンチの計画と見積もり |

## 相手の測り方 (再現コマンド)

    MLX_SERVE_DECODE_FWD_UBENCH=30 MLX_SERVE_DECODE_FWD_UBENCH_S=1 MLX_SERVE_DECODE_FWD_UBENCH_KV=0 \
      ~/dev/mlx-serve/zig-out/bin/mlx-serve --serve --model ~/models/ddalcu-flashnext-serve-4bit \
      --host 127.0.0.1 --port 8161 --mtp --log-level info   # [fwd-ubench] 行を読んで落とす

    tools/biglock.sh .venv/bin/python bench/self_snapshot.py --serve-bin ~/dev/mlx-serve/zig-out/bin/mlx-serve \
      --serve-model ~/models/ddalcu-flashnext-serve-4bit --model ~/models/ddalcu-mlxlm \
      --ctxs 0,4000,17000,25000,32000,50000 --tokens 256 --reps 2 --server-log bench/results/logs/serve.log

## 分業

実装は Sonnet のサブエージェント、親 (Fable) は方針・計測の判定・commit。判断に分岐があるときだけ
`opus-advisor`。横断検索は `scout`。

## compact 前の控え (2026-09-03 11:50)

走行中のエージェントと連鎖の一覧 (ID 付き) は scratchpad の `INFLIGHT-2026-09-03.md`。判定待ちの決め事もそこに。
今日の判定と数字は全部 `docs/research/SESSION-2026-09-02-CATCHUP.md` の 2026-09-03 の節、案の台帳は `docs/research/IDEAS-2026-09-03.md`。
既定に入ったもの (今日): mmap (`FASTMLX_NGRAM_BACKEND`)、P5 行タイル (`MLXTURBO_SDPA_ROWTILE`)。knob で待機中: tail-in-group、prefill-attn の uint4 版、qsa-tail、moe-grouped-gemm。
小さいベンチは chain94 (未起動、P6 と tail v2 が入ったら起動)。フルベンチは小さいベンチで「decode 同着以上、prefill 1.03x 以内」が出てから。

追記 (12:15、compact 前の 2 回目): 今日の午前後半の決着 — custom kernel が decode で負ける正体は「温キャッシュの連鎖 micro」(CATCHUP 12:00、CLAUDE.md の作法に追加)。
K2a (radix select) は集合 100% 一致・13 us で通過、K2b を実装中。HC は第 4 変種 (elementwise だけ融合、GEMV は MLX の qmv) を実装中で見込み 4.7 → 2.2 ms。
ANE は見送り (INT8 の重みコピーでメモリが増える)。最終ゴールは短文脈 100 tok/s (MTP 学習なし)。P8 は保留 (P10 次第)。走行中の一覧は scratchpad の INFLIGHT-2026-09-03.md の「12:15 時点」。

## 現在地 (2026-09-04 03:26、commit 8e52d93 まで。小ベンチ 0903h は走行中、数字は下に追記)

深夜に API 500 でエージェント 4 本が消え、出し直して決着 (全部 CATCHUP 2026-09-04 の項に数字):
- **既定に入れた 3 つ**: n-gram 先読み (pread では走っていなかった。on + group forward の前に投入、17k 冷 -2.9% / 温 -1.8%、出力一致)、
  HC の細長 GEMM に qmm_wide (ビット一致、17k -0.9%)、fused:1 (MoE decode 幅、短 -1.2〜-1.3% / 17k -1.4%、Δ KLD +0.00036、参照テスト反転 0)。
- **畳んだ**: 小 kv の疎 attention (union タイル化、kv=4096 で 0.99x)、HC elem の prefill 幅拡張 (非ビット一致で tok/round -4.8%)、n-gram の late 配置、
  fused:1 の prefill 抑止 (要らなかった)。decode の糊の融合レーンは閉じた (勝ち 2 本: GDN、fused:1)。
- **規則にした**: decode 幅だけの非ビット一致カーネルの 3 条件 (CLAUDE.md の品質段落)。S=1 の KLD は `quant_eval compare --fusions --step 1`。
- **道具**: `tools/qsa_vis_stats.py`、`tools/hc_prefill_micro.py`、`tools/ngram_prefill_diag.py` / `ngram_fetch_micro.py`、`tools/moe_decode_fused_ref_model.py` (worker の tool job)。
  ビット一致ゲート `verify_prefill_bitident` は判定を checkpoints=[] に直した (checkpoints=None の不一致は末尾 v2 で既知)。
- **次**: (1) prefill の小 kv attention は疎性ではなく「スコアを実体化しない融合」(K1 arm A、4k -1〜2% 見込み)。(2) 27B レーン (部品の置き換え、Lily 5/6、teacher)。
  (3) BACKLOG の小物: qmm_wide の M<1024 の食い違い (本番のゲート 1024 の外、未調査)、worker が降ろされる原因 2 つ (FASTMLX_NGRAM_DISK の突き合わせ、回収可能メモリ 99 GB 待ち)、計数ソート、HF パックの 4bit 頭化。
- **罠 18**: サブエージェントは `scratchpad/agent-<name>.md` に節目ごとの台帳を書かせる (API 落ちで再開できるように)。セッションの scratchpad は消えるので台帳・ベンチのスクリプトはリポジトリ直下 `scratchpad/`。
- **小ベンチ 0903h (03:41)**: 冷 prefill 4k 5.74 / 17k 26.5 / 25k 40.8 / 32k 50.3 / 50k 80.1 s (相手 5.70 / 27.8 / 41.3 / 51.6 / 82.1 → **4k 同着、17k 以上は 0.95〜0.99x で勝ち**)、
  decode 53.6 / 58.5 / 47.4 / 52.5 / 49.9 / 44.8 tok/s (1 回 × 256 なので ±10% の運)、温 TTFT 0.15〜0.31 s。次の節目はフルベンチ (反復 2、同冷却) で相手と同時刻に取り直すこと。
- **K1 arm A (04:10)**: 畳んだ (冷 micro 2.3x 遅い、天井側でも完全な融合で 4k -1.2% が上限)。**小 kv の attention は閉じた。**prefill の残りの的は MoE 行ソートの計数ソート化 (BACKLOG、-0.4%、ビット一致) だけで、その後は 27B レーン。
- **計数ソート (04:51)**: 未配線の記録に (取り分 0.04%、見込みは単発 eval の往復を測った誤り)。**Flash-Next の prefill / decode の的はこれで全部一巡。**残りはユーザーの指示待ち: フルベンチ (反復 2、同冷却、相手と同時刻) → 27B レーン (最初の直しは MTP サイドカーの読み込み、BACKLOG)。

## 27B レーン開始 (ユーザー 2026-09-04 07:50「とりあえず 27B やろうか。oMLX や mlx-lm とも比較できる」)

Flash-Next の的は一巡 (単独レイテンシは方針の中では 55〜60 tok/s が天井。抜ける道は永続カーネル / 低ビット / 並列のどれかで、判断はユーザー)。
27B (`~/models/qwen38-27b-4bit` + MTP `~/models/qwen38-27b-mtp`) を、部品の置き換え方式で速くし、mlx-lm / mlx-serve / oMLX (可能なら rapid-mlx) と比べる。
段取り: (1) MTP サイドカーの読み込み修正 (mtp.py の quantize の順) → (2) 相手の起動方法と harness の下調べ (scout) → (3) 基準測定 (同じ冷却、同じ池、4 者) → (4) 27B の内訳 (prefill / decode の部品) → (5) 契約が合う部品から当てる (GDN Metal、sdpa 行タイル、BM=64 qmm の MLP 込み)。
Flash-Next のフルベンチは後回し (ユーザーの指示があるまで回さない)。
- **Mirai / uzu (07:55、`docs/research/EXTERNAL-MIRAI-UZU-2026-09.md`)**: Qwen3.6 27B を M5 Max で 105 tok/s (specdec、<50 MB の学習した draft、蒸留した独自 4bit)。同じ機体なら MTPLX 55 / MLX 26。物理の差ではなく機体 + draft の学習 + 独自量子化。27B の比較には MTPLX (`tools/compare/mtplx-venv/`) も入れる。uzu は別重みなので参考点。
- **方針 (ユーザー 08:30)**: フルベンチは回さない。**汎用的に動かせる移植 (動的に部品を当てる) と高速化を先に。**27B の 5 者比較の本番パスは script だけ用意して後回し (煙試験の数字は COMPARE-QUEUE)。
  走行中: GDN 部品の qwen3_5 移植、sdpa 行タイル + MLP qmm_wide の移植、qwen3_5 用の 1 プロセス A/B 道具 (`tools/decode_ab_generic.py`)。次の段は decode 経路 (SpecEngine + staged) の段階投入と糊の融合を 27B に。
- **27B 移植の現在地 (08:58、commit 3788945)**: GDN 部品 (48 層)、sdpa 行タイル (16 層)、qmm_wide (368 射影、MLP 込み) が契約で当たる。KLD は既定 0.00027 (参照 = 素の 4bit、全部 GDN Metal)。
  速度 A/B は走行中 (GDN Metal 4k/17k、GDN decode 融合 短 ×2、行タイル 4k/17k、qmm_wide 17k)。qmm_wide は 4k で取り分なし (+0.4%)。
  **最大の的は decode 経路**: 27B の round は 82〜112 ms (重み読みの下限 35 ms) で、mlx-serve は同じ MTP 頭で 4k 43 tok/s 対 27。round の内訳を測定中 → 設計 (spec_flash の仕組みを族 adapter で汎用化するか) を advisor と決める。
- **35B-A3B (ユーザー 09:07)**: 35B-A3B は **Qwen3.6** (3.8 ではない)。MTP もあるはず。27B 最優先で、35B-A3B は参考までに後で (MoE 部品の契約の試験台。ダウンロードは計測の合間に、パック名は要確認)。
  パック (HF、未ダウンロード): `mlx-community/Qwen3.6-35B-A3B-4bit`、MTP は `mlx-community/Qwen3.6-35B-A3B-MTP-5bit` (4bit の MTP は無し。mtp.py は scales から bits を推定するので 5bit のまま読める見込み)。
  補足 (09:15): 4bit の MTP サイドカーは mlx-community に無い (本体 `Qwen3.6-35B-A3B-4bit` は 2090 テンソルに `mtp.*` 無し、MoE の gate は 8bit の混合)。MLX 形式で MTP を持つのは `…-MTP-5bit` (qwen3_5_mtp、1 層、46 テンソル) と、第三者の OptiQ 系 (`oQ4e-mtp` 等、別レシピ) だけ。5bit の頭をそのまま使う (bits は scales から推定、1 層なので速度差は無視できる)。
- **順序 (ユーザー 09:14)**: 27B の移植 (decode 経路の (B)) が着地したら、**qwen4_exp (Flash-Next) も同じ分岐ルート (契約で部品を当てる汎用経路) に載せる** → その後 **35B-A3B (Qwen3.6、MoE)**。Flash-Next を載せ替えるときは 17k A/B と fingerprint を毎回のゲートに (数字を落とさない)。
- **融合もどき (ユーザー 09:50「先にそれをやる。27B にも生きる考え方」)**: カーネルを書かずに「動かすバイトと同期」を減らす。種類は (1) 重みへの畳み込み (norm の (1+w) や scale を読み込み時に)、(2) 余計なデータ移動 (型変換 copy、contiguous 化、添字・マスクの毎回生成、入力の複製)、(3) MLX 組み込みの融合 op、(4) グラフの形 (同期を減らす)。
  根拠: 勝った融合 2 本の取り分はバイトと同期の削減で、dispatch の本数は稼働率 96% で値段ゼロ。第 1 段 = 両モデルの decode で一覧と順位付け (エージェント走行中、`scratchpad/agent-pseudo-fusion-audit.md`)。27B の decode 経路 (B) と並行。
  範囲の拡張 (ユーザー 10:05「共通か族固有かを分けるなら 35B-A3B や Gemma 系も見ないと共通項が取れない」): 一覧は項目 × 族 (Flash-Next / 27B / Qwen3.6-35B-A3B (コード読みのみ、モデル無し) / Gemma 4 (`~/models/gemma4-26b-4bit`、S=1 の実測)) で、各項目に「共通モジュール由来か族固有か」の列。
- **方針 (ユーザー 10:13)**: 融合もどきの共通項 (mlx_lm の共通モジュール: KVCache / create_attention_mask / QuantizedLinear / switch_layers / rope) は**決め打ちで直接パッチしてよい** (全族が同じコードを通るので動的な当て方は不要)。契約で当てるのは族固有のブロックだけ。共通パッチは mlx_lm のバージョン固定 + fingerprint + 生成列の一致で検査する (qmm_wide の `QuantizedLinear.__call__` 差し替えと同じ作法)。
  範囲 (ユーザー 10:15「色々な族があるので、なるべく幅広く、できれば最新のを」): コード読みの列は mlx_lm/models にある実装を新しい族から順に広く (DeepSeek / Kimi / GLM / Llama / Mistral / MiniMax / Muse …)。実測は Flash-Next / 27B / Gemma 4 の 3 つ。
- **Gemma 4 の 5 者比較 (ユーザー 10:15「Gemma 4 MTP は 5 つの推論エンジンで比較したい」)**: `gemma4-26b-4bit` + 公式 drafter `gemma4-26b-assistant` (KV 共有の cross-attention、MTP 頭ではない) で 27B と同じ煙試験を 5 者で。mlxturbo は assistant のエンジンを持たないので投機なしの数字になる (それを直すのは別レーン、BACKLOG「Gemma 4 対応」)。エージェント走行中 (`scratchpad/agent-gemma4-smoke.md`)。
  補足 (ユーザー 10:18「投機なしも比較に入れる」): Gemma 4 の表は各エンジン × (draft なし / あり)。27B の表にも投機なしの行を足す (mlxturbo `--no-mtp`、oMLX のペア付けなし、mlx-serve `--no-mtp`; mlx-lm と MTPLX AR は既にある) → 「エンジンの素の効率」と「draft の質」を分ける。
- **融合もどきの一覧 (10:20)**: Flash-Next (稼働率 95%) と 27B (98.7%、帯域の 92%) には取り分なし = レーンを閉じる。Gemma 4 だけ稼働率 80% で norm 331 本/step が的 (Gemma レーンで)。MLA 族の `pe_scores` mask は将来の的。**回帰**: `_moe_fold_block` の `inv` 未束縛で Flash-Next の prefill が落ちていた (13f0d21〜、10:20 に修正)。
- **方針 (ユーザー 10:30)**: カーネルを書かずに本数を減らす 5 つは全部やる価値あり: (1) 結果を使わない仕事を飛ばす (短文脈の QSA indexer 等)、(2) 独立な小 op を積んで 1 本に、(3) 層単位の `mx.compile` の PoL、(4) **MLX の更新 (価値あり)**、(5) バッチ (単独ベンチには活かせないので優先度は下)。2 倍の筋は永続カーネルでも厳しい (合意)。
- **位置づけ (ユーザー 11:05)**: 「mlx-lm の LLM 特化版」が最も近い。汎用性寄り (同じ重み・形式・品質、契約で当たる部品で族の追加に追従)。相手 (mlx-serve / MTPLX / uzu / oMLX) は族専用や専用パックで攻める。比較は相手の推奨設定で。
- **目標 (ユーザー 11:10)**: 製品の看板は架け替えてよい。**目標は「皆が (意識してもしなくても) 使う推論エンジン」** = 製品が中に入れるエンジン。条件: (1) 同じ重みで負けない (族専用パック・独自形式に依存しない)、(2) 族の追従 (契約 + 決め打ちの共通パッチ)、(3) 入れ替えやすさ (OpenAI 互換 HTTP + mlx-lm と同じ Python API)、(4) 品質を測って出す。**同時接続 (continuous batching) は採用の場面で効く**ので、27B が落ち着いた後の順番に入れる (単独ベンチには活かせないが目標には要る)。
- **fast_qmm 型の的を全部洗う (ユーザー 11:33)**: MLX 組み込みカーネルがうちの形で天井 (410 GB/s / 11.2 TFLOPS) から大きく外れている箇所の一覧 (quantized_matmul の 8bit・小 K/N、gather_qmm の少行、sdpa の S=2〜8、rms_norm / rope の少行、conv1d、sort 系、softmax / argmax の語彙長、lm_head の M=2〜8)。達成率 × 占有で順位。エージェント走行中 (`scratchpad/agent-ceiling-audit.md`)。小 M qmm と lm_head (環境要因) は別途。
- **製品の方向 (ユーザー 11:39、`docs/research/PRODUCT-DIRECTION-2026-09.md`)**: 中に入るエンジン。P0 = ExecutionPlan / explain / strict plan → Engine / Session API + mlx-lm アダプター → 配布 (3.11+, uvx, doctor) → 常設ベンチ + fast-path hit rate → README。着手は 27B の decode (小 M qmm) の着地後。
- **TurboQuant: 実装確定 (ユーザー 11:48)**。計画は `docs/research/TURBOQUANT-PLAN.md` (Codex で実装する可能性が高いので、着手者が読めば足りる形)。順序は Gemma レーンの中 (drafter → norm → KV 量子化 第 1 段 → TurboQuant)。
- **方針 (ユーザー 12:40)**: ビット一致 / 決定性は品質ゲートの条件にしない。「品質に影響が無ければよい」= 参照 (fp32) への距離が素以下 + Δ KLD ≤ +0.0005。決定性は計測の便利さとしてだけ。小 M の経路 (small_m / mma / nocap) は ms/tok の複数プロンプト平均 + KLD で選ぶ。ExecutionPlan の exactness は「丸め級」でよい。
- **現在地 (2026-09-04 13:15、fdc06e1)**: 小 M (M=2..5) の量子化行列積を既定 auto に (27B 短 -1.9% / 4k -4.6%、fp32 距離は素と同じ)。Gemma 4 26B の 5 者は相手 4 行が揃い、公式 draft は MoE の 26B で 2 者とも遅い → Gemma レーンから drafter を外した。
  **Codex への引き継ぎを書き始めた**: 入口 `AGENTS.md`、本体 `docs/HANDOFF-2026-09-04.md` (走行待ちの節を埋め中)、memory の写し `docs/research/NOTES-FROM-MEMORY-2026-09-04.md`。ユーザーは「どこかで完全に切り替える」意向。
  走行中: mlxturbo の煙試験 3 行 (`scratchpad/gemma4_smoke_runs3.sh`)、sdpa 幅分割の再判定 (fp32 距離 + 短文脈)、MoE compile の in-model、台帳 27 件の要約 (`docs/research/AGENT-LEDGERS-DIGEST-2026-09-04.md` 予定)。
- **13:35**: sdpa 幅分割 (27B) と MoE compile (Flash-Next) を既定 auto に (12f861f)。煙試験 (13:22〜13:30): 27B MTP 33.8 / 32.1、投機ゼロ 23.8 / 22.9 (`MLXTURBO_RUNNER=fallback`、99ba892) → 素の効率は mlx-serve と同着、差は投機の取り分 (1.40 倍対 1.71 倍)。
  走行中 2 本 (判定線と再開コマンドは HANDOFF の「走行中」): 小 M を Flash-Next へ (worktree)、27B の gated chain の閾値の掃引 (`MLXTURBO_SPEC_GATE_H` / `MLXTURBO_SPEC_MAX_DRAFT`)。

## Codex 引き継ぎ監査後の現在地 (2026-09-04 14:18 JST)

ここから下を再開時の正本とする。`CLAUDE.md`、`docs/HANDOFF-2026-09-04.md`、
このファイルの旧節、`docs/research/SESSION-2026-09-02-CATCHUP.md` の末尾と、実際の
作業木・台帳・プロセスを照合した。監査開始時の HEAD は `5e4c381` で `main == origin/main`。
GPU の `decode_ab` / `biglock` プロセスは動いていない。「走行中」ではなく、以下の 2 本が
**途中成果を残して停止中**である。

### まず保全するもの

- 27B の `MLXTURBO_SPEC_GATE_H` / `MLXTURBO_SPEC_MAX_DRAFT` 掃引用のコードと直接テストは
  `edebaea` に分離して commit 済み。関連する SpecEngine / qwen3_5 の 24 tests passed。
  `scratchpad/agent-27b-gate-sweep.md` が計測台帳。
- `.claude/worktrees/agent-ae05b9756c852f071` は Flash-Next の小 M 接続用 worktree。
  実験差分は branch `worktree-agent-ae05b9756c852f071` の `6553991` に commit 済みで、
  main には入れていない。関連 108 tests passed と `tools/vendor_fingerprint.py` 通過。
  `scratchpad/agent-fn-small-m.md` が計測台帳。
- `scratchpad/` と `.claude/` は未追跡だが、上の途中成果を含む。判定を記録して差分を
  回収するまで削除・prune しない。`git worktree prune --dry-run` に出る `base` は消えた
  Claude セッションの古いメタデータで、上の生きた worktree とは別物。

### P0-A: 27B の gated chain の閾値を決着させる

完了済みの第 1 段では、現行 `h=0.19` の 28.087 ms/tok に対し、深く引く側の
`h=0.10 / 0.05 / 0.02` はそれぞれ +5.5% / +6.8% / +8.1% 遅かった。
tok/round の増分より draft 側の ms/round の増分が大きい。したがって下方向は再試行せず、
上方向だけを測る。

再開の 1 コマンド:

```bash
BIGLOCK_PRIO=1 BIGLOCK_NO_WORKER=1 tools/biglock.sh .venv/bin/python tools/decode_ab_generic.py --model ~/models/qwen38-27b-4bit --mtp ~/models/qwen38-27b-mtp --knob MLXTURBO_SPEC_GATE_H=0.19,0.30,0.45,0.60 --baseline 0.19 --ctx 0 --tokens 512 --reps 2 --out bench/results/gate-h-up-27b-short-0904.json
```

判定と後処理:

1. 短 3 本平均の ms/tok を主指標にする。現行比 -2% 以上の候補だけ 4k / 17k へ進め、
   どちらも +0.5% を超えて遅くならないことを確認する。
2. 候補が残ったときだけ `MLXTURBO_SPEC_MAX_DRAFT` を掃引する。tokens/round、round ms、
   draft 幅の分布を mlx-serve の `[spec-stats]` と同じ表に置き、差が受理率か draft 費用かを分ける。
3. 勝つ設定が無ければ現行 `GATE_ROLLBACK_COST=0.19` を維持し、掃引用 env の差分は本体へ
   入れず畳む。どちらの判定でも数字を CATCHUP 末尾、棄却なら BACKLOG に記録する。

### P0-B: Flash-Next の小 M 接続を決着させる

全 135 射影へ当てた第 1 便は、短 3 本 + 17k 3 本の **6 / 6 で ms/round が
+0.26〜+0.70%**。測った全条件で遅くならないという「代金ゼロ」の条件を満たさないので、
この形は既定に入れない。27B と違い、Flash-Next の N=2560 の out/o_proj は
threadgroup が少ないという説明が台帳に残っている。追加で行うのは、N=2560 の 48 射影を
外した部分集合の 1 回だけ。そこで符号が変わらなければレーンを閉じる。

再開の 1 コマンド (まず差分と作業位置を復元する):

```bash
git -C .claude/worktrees/agent-ae05b9756c852f071 status --short
```

完了条件:

1. 同じ worktree で対象集合を明示し、`bench/test_fn_small_m.py`、dispatch / fusion の既存テスト、
   `tools/vendor_fingerprint.py` を通す。
2. `tools/decode_ab.py --knob fn-small-m` を短 3 本 × 512 × 2 と 17k
   `--prefill-once` で 1 プロセス交互測定する。ms/round が測った全条件で遅くならない場合だけ採る。
3. 採るなら root の最新 `main` へ差分を移して再検査する。畳むなら worktree を消す前に、
   全射影と部分集合の数字を CATCHUP / BACKLOG へ残す。worktree の削除は記録と回収の後。

### P1: 27B のレーンを閉じ、qwen4_exp を汎用分岐ルートへ載せる

P0-A の決着で 27B の decode レーンを閉じる。現在の 27B は MTP 33.8 / 32.1 tok/s
(ctx0 / 4k)、投機ゼロ 23.8 / 22.9。mlx-serve との差は素の forward ではなく投機の取り分にある。
閾値で差が縮まらなければ、次に見るのは draft chain の入力品質であり、素のカーネルを再探索しない。

その後 `qwen4_exp` (Flash-Next) を `spec.py` 側と同じ「素の mlx_lm forward に契約の合う部品だけを
当てる」分岐へ載せる。既存の 25 個の融合を一度に剥がさず、部品単位で移す。各段のゲートは
`tools/vendor_fingerprint.py` と 17k の 1 プロセス A/B。現行の Flash-Next の数字を落とした段は戻す。

再開の 1 コマンド:

```bash
rg -n "qwen4_exp|_staged_forward|_hidden_forward|enable_default_fusions" mlxturbo/spec.py mlxturbo/spec_flash.py mlxturbo/fused.py mlxturbo/runner.py
```

### P2: Qwen3.6-35B-A3B を 3 本目の試験台にする

27B → qwen4_exp の汎用ルートが安定した後に着手する。対象は
`mlx-community/Qwen3.6-35B-A3B-4bit` と `...-MTP-5bit`。2026-09-04 18:15時点で本体は
4 shard・約19GBを取得済み。18:17にMTPも取得完了 (`model.safetensors` 580,709,585 bytes)。
最初に AR 対 MTP を測り、MoE 部品の
契約がどこまで自動で当たるかを対応表にする。5bit MTP は 1 層なので、そのまま読む。

再開の 1 コマンド:

```bash
du -sh ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-MTP-5bit 2>/dev/null
```

### P3: Gemma 4 は温 TTFT の回帰から直す

26B の公式 assistant drafter は oMLX と mlx-serve の両方で遅くなったので、mlxturbo 用 drafter の
新設は 26B では行わない。Gemma レーンの最初は FallbackRunner で 4k の温 TTFT が冷と同じ
(2.76 s 対 2.69 s) 問題。接頭辞 cache の型 / snapshot / restore を直し、mlx-lm の 0.37 s を
少なくとも機能上の基準にする。その後、norm 331 本/step の削減 → mlx-lm 組み込み
`QuantizedKVCache` の実測 → `docs/research/TURBOQUANT-PLAN.md` の TurboQuant 実装へ進む。
dense 31B の assistant drafter は別の比較として、26B の結果から推測せず測って判断する。

再開の 1 コマンド:

```bash
rg -n "Gemma 4|温 TTFT|FallbackRunner|snapshot|checkpoint" docs/research/COMPARE-QUEUE.md docs/BACKLOG.md mlxturbo/runner.py
```

### P4: 製品 P0

モデルレーンの上記順序が落ち着いたら `docs/research/PRODUCT-DIRECTION-2026-09.md` の P0 に移る。
順は ExecutionPlan / `mlxturbo explain` / strict plan と HTTP ヘッダ → Engine / Session API と
mlx-lm アダプター → Python 3.11+ / `uvx` / `doctor` → 常設ベンチと fast-path hit rate → README と看板。
速度値だけでなく、MTP の有無、fallback、exactness、cache reuse を公開契約にする。

再開の 1 コマンド:

```bash
sed -n '1,180p' docs/research/PRODUCT-DIRECTION-2026-09.md
```

### 後段・機体待ち

- continuous batching は製品採用には必要だが、単独レイテンシの現在レーンには混ぜない。
- NAX 機では MLX 0.32.2 の sorted `gather_qmm` が 32K 行超で壊れる疑いが最優先。
  0.32.3 が無ければ `MLXTURBO_PREFILL_GROUP=1` または 32K 行の分割ゲートから始める。
- HF 公開パックの lm_head 4bit 化、27B の state_out スキップ、GQA 小幅 attention、
  n-gram 先読みの 50k、sdpa 分割の幅 9/10 と 50k は、上の主線を止めず BACKLOG から拾う。

### 今回は行わないこと

- Flash-Next のフルベンチと overnight tier。フルベンチはユーザーの明示指示があるまで走らせない。
- 畳んだ融合、疎 attention、計数ソート、層まるごとの compile、capture-module を再開しない。
- 途中差分を混ぜて部分 commit しない。採否が出た論点ごとに、実経路のテスト後に扱う。

## Blindspot 監査と追加 A/B 後の更新 (2026-09-04 14:48 JST)

この節を上の P0-A / P0-B より新しい正本とする。

### 完了: gate の h 掃引

- 下方向 `0.10 / 0.05 / 0.02` は現行 `0.19` より +5.5 / +6.8 / +8.1%。
- 上方向 `0.30 / 0.45 / 0.60` の短3本平均は -1.6 / +0.9 / -1.0%。採択線 -2%に
  届かず、4k / 17k と `MAX_DRAFT` 掃引は行わなかった。
- 既定は `h=0.19` のまま。ただし controller 自体の意味不整合が見つかったので、27Bレーンは
  閉じず、次の P0 を先に行う。`MLXTURBO_SPEC_GATE_H` の口はその A/B 用に当面残す。

### 完了・不採用: Flash-Next の小M接続

- 全135射影は6/6で ms/round +0.26〜+0.70%。
- N=2560 を外した C (`N>=6144`) も short ms/round +0.1%、ms/tok +1.2%。17kは
  ms/round -0.1%、ms/tok -2.4%だが、全条件非悪化を満たさない。
- branch `worktree-agent-ae05b9756c852f071` の `6553991` は main へ入れない。worktree は
  記録を確認するまで残してある。削除は別の明示作業として扱う。

### 新 P0-A: 27B controller の semantics を直して短 A/B

現在の `_gate_depth` / `_plan_depth` は、vendored Swift と Flash の `DepthController` に対し、
未来利得の使い方、失敗位置の keep、first miss より深い EMA 更新が異なる。全位置の受理率0.95で
深さ1になる既存テストは正しい仕様ではなく、欠陥を固定している。

作業順:

1. round trace に decision 前の位置別 EMA / obs、選択深さ、accepted、round ms を追加する。
2. acceptance 更新を「accepted prefix の1、それに続く最初の miss の0まで。以深は未観測」へ直す。
3. 深さ walk を参照側の累積順と stop semantics にそろえる。Swift の位置別 price vector を
   scalar `h` にどう写すかは、式を変える前に単体テストの表で明示する。
4. 純関数テストで、高い条件付き受理率ほど選択深さが短くならないこと、miss 以深を更新しないこと、
   failed marginal link を数えないことを固定する。
5. 短3本 × 512 × 2だけ A/B。勝つ場合だけ 4k / 17k と KLDへ進む。

再開の1コマンド:

```bash
rg -n "_expected_future_gain|_gate_depth|_plan_depth|pos_accept_ema|observed =" mlxturbo/spec.py bench/test_spec_draft_chain_qwen3_5.py tools/reference/e120/Qwen36MTPBlockSession.swift
```

### 新 P0-B: MTP cache repair の先頭1行再利用

controller の後に、別論点として扱う。先に rejection / partial / full / D7 / EOS の合成ケースで、
full rebuild と retain-first の cache tensor と次 proposal を比較する。一致した場合だけ一時 knob を
作り、短3本 A/B。理論上限は eligible round あたり約1.24 msなので、広い抽象化は作らない。

**完了・採用 (2026-09-04 16:25)。**partial / fullはappend幅が変わり最大9.54e-7ずれるため畳み、
bit一致する`consumed == 1`だけを保持。合成8ケースがbit一致、短3本はms/tok / ms/roundとも-0.9%、
3/3 prompt非悪化、生成列一致。A/B中は34.3→48.2 ms/tokの熱ドリフトがあったため、絶対値でなく
同一processのABBAとprompt別平均で判定した。

再開の1コマンド:

```bash
rg -n "mtp_off0|mtp_cache.trim|_mtp_append\(window|chain_head" mlxturbo/spec.py bench/test_spec_draft_chain_qwen3_5.py
```

### P1: proposal-only q2 top-32 recall trace

exact q4 lm_head のカーネルは帯域天井なので触らない。Flash / vendored Swift の proposal-only
shortlist を27Bへ移せるか、まず exact proposal を変えず top-32 recall だけ取る。recall が十分な
ときだけ q2 常駐重み・warm-up・速度・品質を審査する。

再開の1コマンド:

```bash
rg -n "_build_rerank|_draft_argmax|DRAFT_RERANK|applyDraftLMHead|_head\(h_mtp" mlxturbo/spec_flash.py mlxturbo/spec.py tools/reference/e120
```

その後の順序は既存どおり、qwen4_exp の汎用分岐 → Qwen3.6-35B-A3B → Gemma 4 温 TTFT →
製品 P0。フルベンチと overnight は引き続きユーザーの明示指示待ち。

## Controller 決着後の更新 (2026-09-04 16:10 JST)

- **完了・不採用**: 完全な参照semantics + 再較正h=0.05は、現行legacy h=0.19比で短
  ms/tok +6.0% / ms/round +6.7% / tok/round +1.6%。本番は現行維持。再現wrapperとmetaを追加。
- **次**: P0-B の MTP cache repair先頭1行再利用。正しさの合成ケースを先に置き、一致時だけ短A/B。
- **別件**: `bench/test_ngram_stream.py` のcollection時CPU固定がGPUテストへ漏れる。fixtureで局所化する。

再開の1コマンド:

```bash
rg -n "mtp_off0|mtp_cache.trim|_mtp_append\(window|chain_head" mlxturbo/spec.py bench/test_spec_draft_chain_qwen3_5.py
```

## MTP cache repair 決着後の更新 (2026-09-04 16:25 JST)

- **完了・採用**: rejection roundだけ先頭MTP cache行を保持。短ms/tok -0.9%、3/3非悪化、bit一致。
- **次**: 27B proposal-only q2 top-32は、exact proposalを変えないrecall traceから始める。
- **並行しない後続**: qwen4_expの汎用分岐とK/V slice SDPA、Qwen3.6-35B、Gemma温TTFT、製品P0。
- **テスト基盤は完了**: `test_ngram_stream` のCPU固定をfixtureへ局所化。CPU→GPU同一processで53 passed。

再開の1コマンド:

```bash
rg -n "_build_rerank|_draft_argmax|DRAFT_RERANK|applyDraftLMHead|_head\(h_mtp" mlxturbo/spec_flash.py mlxturbo/spec.py tools/reference/e120
```

## 27B q2 proposal head 決着後の更新 (2026-09-04 16:58 JST)

- **完了・採用**: q2 coarse top-32 + q4候補行再採点。1,677 proposalのR@32 100%。
- **速度**: 短 -2.0% / 4k -1.6% / 17k -3.0% ms/tok。17kはtok/round +6.6%。
- **代金**: 常駐378.9 MiB、起動時temporary約2.37 GiB。17k生成列は262 token目から丸め分岐。
  target verifyはexactで、`MLXTURBO_DRAFT_RERANK=0`が従来headへの退避口。
- **次**: qwen4_expの汎用分岐とK/V slice SDPA。PR #391のfixed-M4 verifier / FR-Spec等も
  現行実装との差分を棚卸しして、重複しない高価値レーンだけ足す。

再開の1コマンド:

```bash
rg -n "enable_sdpa_split|enable_sdpa_split_generic|_model_layers|_arch_registry" mlxturbo/fused.py mlxturbo/_arch_registry.py mlxturbo/spec_flash.py
```

## MTPLX #391 / Voz 調査の反映 (2026-09-04 17:04 JST)

- qwen4の汎用structure routingは既にある。先に独立したK/V prefix trimを終える。
- その後のFlash-Next高価値順は fixed-M4 verifier/graphbank (+21%報告) → FR-Spec Q8
  65,536-row head (+6.27%報告)。外部のgroup32数値は方向だけ使い、group64で別々にA/Bする。
- ANEへのfull model移植は再開しない。FR-Spec head完成後に限り、MLX headを置換して定常RSS非増加、
  I/O込み20%以上短縮、short/17k非悪化を満たすか小さく測る。

再開の1コマンド:

```bash
rg -n "def _verify|def _draft_chain|_staged_forward|_draft_argmax" mlxturbo/spec_flash.py
```

## Flash-Next K/V prefix trim 決着後の更新 (2026-09-04 17:23 JST)

- **完了・不採用**: SDPA幅分割の各塊でK/Vを`offset+t1`まで切る案は、短3本で
  ms/tok +5.2% / ms/round +5.0%。生成列とtok/roundは同一だった。
- **17kでは対象外**: 既定QSA decodeが通常attentionを迂回するため、このsliceは発火しない。
- **次**: fixed-M4は幅4への固定ではなく、cache offsetとGDN/PLE/KV/indexer状態を明示した
  state-in/state-out境界が作れるかを先に合成検査する。既存compile失敗を無視したgraphbank移植はしない。

再開の1コマンド:

```bash
rg -n "def _verify|def _draft_chain|def _staged_forward|def capture|def rollback" mlxturbo/spec_flash.py
```

## fixed-M4直接portの判定 (2026-09-04 17:28 JST)

- **直接portは不採用**: 幅4は既存`depth=3`で実行できるが、同機体の既存掃引でdepth 2比
  短+11.7% / 4k+8.2% / 17k+3.2%。固定幅だけに取り分はない。
- **graphbankは前提不足**: GDN/PLE/KV/indexer/MTPとcache offsetを明示したstate-in/state-out adapterが
  無く、過去のfull-forward compileはcache副作用のため失敗済み。状態を欠いたprobeは作らない。
- **次**: FR-Spec Q8のrow selectionを出典どおり再現できるか確認し、まず読み取り専用recall traceを作る。
- **後で再開する条件**: qwen4 component replacementでstate-pure adapterが完成し、幅4の全cacheと
  rollback keep=1/3/4が既存経路と一致すること。

再開の1コマンド:

```bash
rg -n "_build_rerank|_draft_argmax|RERANK_BITS|RERANK_TOP|topk" mlxturbo/spec_flash.py tools/decode_ab.py
```

## FR-Spec 64K語彙の実装前判定 (2026-09-04 17:36 JST)

- **完了・不採用**: PR同梱の公式65,536 ID（SHA-256 `950adf…c1fba`）は、現行Flash traceの
  本走行11,780 target tokenでsupport coverage 89.02%。事前線99.9%を通らない。
- **prompt別**: 53.88% / 99.17% / 80.51% / 98.02% / 94.17% / 99.66%。各prompt 99.5%線は1/6。
- **実装しない**: support外は必ずdraft missになるため、Q8 headの構築・A/Bへ進む価値がない。
- **ANEも閉じる**: FR-Spec headの置換を開始条件にしていたため、現時点で対象headが無い。
- **次**: ローカルモデルがあるGemma 26B/31Bの速度レーンへ進む。Qwen3.6-35B本体と
  MTP-5bitも取得済みなので、その次にAR/MTP比較へ進める。

再開の1コマンド:

```bash
rg -n "Gemma|warm TTFT|QuantizedKVCache|TurboQuant|assistant" docs/BACKLOG.md docs/research mlxturbo tools
```

## Gemma 4 温 TTFT 修復後の更新 (2026-09-04 17:56 JST)

- **完了・採用**: `RotatingKVCache` の深いcheckpointと末尾8 token保持をFallbackRunnerへ追加。
- **速度**: 4k温TTFT 2.76→0.382 s (-86.1%、7.2倍)、17k温TTFT 0.415 s。
- **正しさ**: 実Gemmaサーバーでcheckpoint復元と全量再構築の64-token出力が完全一致。
  `bench/test_server.py` + `bench/test_fusions_other_family.py` は437 passed。
- **次**: Gemma 4の同じhiddenに対するpre-feedforward norm 2本とRouter normの共通統計化を、
  生成経路の同一process A/Bで判定する。その後に組み込み`QuantizedKVCache`、TurboQuant。

再開の1コマンド:

```bash
rg -n "pre_feedforward_layernorm|pre_feedforward_layernorm_2|class Router|RMSNorm" .venv/lib/python*/site-packages/mlx_lm/models/gemma4_text.py mlxturbo
```

## 強冷却の運用 (ユーザー 2026-09-04 18時台)

1日最大2回、各数時間の強冷却を用意できる。常時運用は設備上行わない。通常条件の同一process A/Bで
候補が生き残った後、原則Flash-Next研究候補に1枠、その日の別モデル候補に1枠を使う。開始前に
対象モデル・固定条件・所要時間をユーザーへ明示する。候補が弱い日は使わない。

18:07の強冷却直後診断はFlash-Next短文59.5 tok/s、固定GEMM 12.05 TFLOPS。通常条件53.6 tok/s比
約+11%。数分継続後の再測は**61.5 tok/s**、直後GEMM **12.72 TFLOPS**で既往上限13.05の
97.5%まで到達した。現在の設定は普段の常時冷却として合格。追加の強冷却は正式ベンチ時だけ使う。
