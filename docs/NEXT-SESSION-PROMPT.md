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

GPU は `tools/biglock.sh` で 1 本ずつ直列。親の連鎖スクリプトは scratchpad
(`/private/tmp/claude-501/-Users-ht-dev-fastmlx/65b31683-391c-444c-b255-622b126131f9/scratchpad/run_chainNN.sh`、
いま 80 番台) にあり、前の連鎖の終了を `pgrep -f run_chainNN.sh` で待って順番を付けている。

案出し Studio (Scout 5 本 + decode advisor) の凍結ポートフォリオと落とした的は `docs/research/IDEAS-2026-09-03.md`。
ユーザーの Commit (07:50): 選抜は親に任せる。**カーネル (P3) はフルベンチの前に書く。**

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
3. 天井スタブ (chain86、`--knob stub-*`) で上限を取ってから、通った案だけ Sonnet に実装させて ABBA: D1 → D2 → D3。
   D6 は stub-indexer-topk の天井が 1 ms/round 未満なら D7 (top-k select カーネル) ごと畳む。
4. P3: MoE の grouped GEMM。第 1 段 (dense クローン、ビット一致 1.000 倍) と第 2 段 (segmented BM=32、r=160 で 1.098 対 1.167) は済。
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
4. **フルベンチの後**: Qwen3.8-27B 4-bit と Qwen3.6-35B-A3B 4-bit を検証に加える (量子化ビットは揃える)。
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
