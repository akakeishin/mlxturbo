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

## いまの段取り (2026-09-03 05:00 時点。何をやっているか見失ったらここを読む)

GPU は `tools/biglock.sh` で 1 本ずつ直列。親の連鎖スクリプトは scratchpad
(`/private/tmp/claude-501/-Users-ht-dev-fastmlx/65b31683-391c-444c-b255-622b126131f9/scratchpad/run_chain45.sh`)
にあり、旗ファイル (`bench/results/logs/*.ready` / `hc.done`) で順番を付けている。連鎖が消えていたら
下の順番を手で流せばよい (コマンドは各行に書いた)。

GPU の待ち行列 (2026-09-03 03:20 時点の状態):
- 済: HC 検証 (畳む)、仮説マイクロ (KV slice は素の forward では in-place、take+qmm 棄却)、depth margin 版
  (17k -4%、既定 on)、T=1 gather 17k/50k (50k -21.3%) **ただし長文脈 KLD が 0.04 / 0.017 で幅外 → 原因調査中、
  既定 off のまま**、lm_head 4-bit 本焼き (KLD +0.0047、速度比較用に限定)、hc-compiled (取り分なし)、G=8 (-1.2%、
  畳む)、PLE hoist (差なし、畳む)、prefill 8k の内訳 (MoE 43% / GDN 27% / PLE 4.4%)、TTFT 内訳 (固定 300 ms 発見)。
- 済 (05:30): n-gram 先読み (8k -0.9%、畳む)、固定 300 ms = リクエストごとの detokenizer 構築 → 修正済み
  (温 TTFT 0.45 → 0.15 s、完全ヒット 0.003 s)、投機ラウンドの KV コピーは無し (棄却)、decode の kv 罰は
  +3 ms/round で attention 層 (indexer + gather マスク) に帰属、レーン 8 は閉じた。
- 済 (08:00): gather カーネルは **既定 on** (KLD 0.04 は top-k 反転のカスケードで、受理済みの GDN Metal の同じ物差し
  0.111 より小さい)。indexer-lean は畳んだ (+0.3%)。小さいベンチ (冷却強化後、既知を片付けた基準) は
  `bench/results/self-snapshot-turbo-small-0903b.json` (gather 前) と `...-0903c.json` (gather 後、走行中)。
- 済 (10:00): 仮説 A/B の判定 — rerank は受理率を削らず速い (維持)、draft の木は hit@2−hit@1 = 17 pt だが
  verify 幅 +1 の費用に食われて畳む、group_size 128 は forward -3% で焼き直しに見合わず畳む、
  `mx.metal.start_capture` は 68 GB で使えない、indexer-lean は ±0.3% で畳む。
- 済 (10:50): HC の split-K も否定 (レーン 1 の HC カーネルは閉じた)、temp 0.7 の受理は greedy の 0.91 倍で
  厳密棄却サンプリングは畳んだ。仮説 A/B は一通り終わり。
- 走行中 (03:56〜): **フルベンチ** (chain70、`bench/results/self-snapshot-full-0903.json`、両エンジン、0/4k/17k/25k/32k/50k、
  反復 1、冷却 10 分後)。mlx-serve は origin/main と同じで再ビルド不要と確認済み。
- その後 (chain71): 長文脈の品質ゲート `tools/longctx_quality.py` を 17k (n=8) と 50k (n=6) で。1 回目は thinking on の
  ままで課題不成立だった (直して 2k の健全性 recall 3/3、quote 2/3)。dense と kernel の正答率と一致率で gather
  カーネルの採用を裏付ける (kernel が dense より有意に低ければ既定 off に戻す)。
- その後: 長文脈の品質ゲートを 17k / 50k で走らせて gather カーネルの採用を裏付ける → temp 0.7 の tok/round →
  mlx-serve 最新版でフルベンチ → 27B / 35B-A3B。
- その後: 小さいベンチ (mlxturbo だけ、冷、新しい冷却条件の基準) → 仮説 A/B (draft の hit@2、rerank off、
  厳密棄却サンプリング、indexer の op 整理、group_size 128) → mlx-serve 最新版でフルベンチ → 27B / 35B-A3B。

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
