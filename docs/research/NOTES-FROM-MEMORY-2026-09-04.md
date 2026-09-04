# Claude セッションの memory の写し (2026-09-04 12:57 時点、全文)

Claude Code の memory (`~/.claude/projects/-Users-ht-dev-fastmlx/memory/`) は Codex からは見えないので、引き継ぎのために全文をここへ写した。
1 節が memory 1 ファイル。日付の古い節は履歴で、現在地は `docs/HANDOFF-2026-09-04.md` と `docs/NEXT-SESSION-PROMPT.md` の末尾が正。
節の中の `[[name]]` は memory どうしのリンクで、同じ名前の節を指す。

## 目次 (memory の index そのまま)

- [LM Studio 統合の方針](lmstudio-integration-plan.md) — サーバー先行・プラグイン保留・mlx-engine PR 無し。実装前にダンプ観察で潰す論点 2 つ
- [mlx-serve が本命の競合](mlx-serve-competitive-position.md) — 競合の位置づけ (数字は古い。現在地は「対 mlx-serve の揃えた現在地」と「多日レーン」を見る)
- [MTPLX の競合状況](mtplx-competitive-position.md) — 急伸中の GUI 一体型。fastmlx はバックエンド + 品質計測で差別化、人気は追わない
- [Mirai / uzu の競合状況](mirai-uzu-competitive-position.md) — M5 Max で 27B 105 tok/s の正体は機体 + 学習 draft + 蒸留 4bit。同じ重みではない。方針のままなら MTPLX 級が上限
- [draft の学習は保留](draft-training-deferred.md) — アイデアとして残す。特化で汎用性能が落ちるので当面やらない (ユーザー 9/4)。採るなら汎用性能を落とさない条件で
- [申し送りは docs に書く](handoff-in-docs.md) — fastmlx は docs/KERNEL-HANDOFF.md、チャットだけの申し送り不可
- [Flash-Next の速度決着](flash-next-speed-verdict.md) — 旧レーンの記録。bf16 の元は外付け /Volumes/Mobile SSD/models/qwen38fn-bf16 に再取得済み (2026-09-01)
- [残作業](fastmlx-remaining-work.md) — 入口だけ。NEXT-SESSION-PROMPT.md と IDEAS-2026-09-03.md とリポジトリ直下 scratchpad/INFLIGHT-<date>.md が正 (セッション scratchpad は消える)。順は 測定 → PoL → in-model → 小ベンチ → フルベンチ
- [計測の作法](measurement-discipline.md) — 1 プロセス内で交互に測る。ablate の積み上げと温キャッシュの micro を信じない。連鎖 micro は重みを巡回して冷やす。tok/step は複数プロンプト平均で
- [gemma4-lora レーン](gemma4-lora-lane.md) — 12B bf16 高ランク LoRA、CUDA 側と I/F 揃え、mlx-vlm trainer は不採用で自前ループ。実測 171–192 tok/s
- [Opus サブエージェント委譲](opus-subagents-for-work.md) — Fable 親のとき実装・検証は model:opus の general-purpose に出す
- [Qwen3.6 の Mac キャプション](qwen36-mac-caption.md) — 0.88 枚/秒で B200 の 1/28。バッチ化が唯一の手、画像サイズ統一が前提、fastmlx は動作点に届かず
- [互換サーバー](fastmlx-server.md) — mlxturbo として GitHub / PyPI / HF に公開済。汎用は名乗らない。fake を実物より緩くしない、ビット一致を基準にしない
- [目標: 皆が使う推論エンジン](engine-goal-everyone-uses.md) — 製品が中に入れるバックエンド。看板は架け替え可。同じ重みで負けない・族の追従・入れ替えやすさ・品質の根拠。同時接続は後の順番 (ユーザー 9/4)
- [製品の方向 2026-09](product-direction-2026-09.md) — P0 = ExecutionPlan/explain → Engine/Session API → 配布 → 常設ベンチ + hit rate → README。着手は 27B decode の後。TurboPack と統合スケジューラは P1
- [TurboQuant は実装確定](turboquant-decided.md) — KV 3bit (回転 + Lloyd-Max + QJL)。Gemma レーンの中、計画は docs/research/TURBOQUANT-PLAN.md、Codex で着手しうる (ユーザー 9/4)
- [ローカル利用の結論](local-model-verdict.md) — Flash-Next 一律 4bit を mlx-serve + --mtp で。パック改良と MTP 貢献の両レーンは実測で閉じた
- [対 mlx-serve の揃えた現在地](vs-mlx-serve-clean-baseline.md) — 9/4 小ベンチ 0903h: 冷 prefill 4k 同着、17k 以上 0.95〜0.99x で勝ち、decode は ±10% の運、温 TTFT 4-17 倍速。決着はフルベンチ (反復 2)
- [計測の罠 (0902 午後)](fastmlx-measurement-traps-0902.md) — 熱で 25% 落ちる、sep n-gram は 17k で 41% 遅い、capture が GDN 融合を素通し、温キャッシュの連鎖 micro (罠 12)、単発 eval の往復 (罠 19)
- [多日レーン (2026-09)](fastmlx-lanes-2026-09.md) — 現在地・最終ゴール (短文脈 100 tok/s)・既定に入ったもの・進行中 (K2、HC v4、P6〜P10、tail 修正)・畳んだもの
- [多様な入力と勝っている領域](bench-diverse-inputs-and-winning-areas.md) — ベンチは池 6 種、勝っている項目も調べる、overnight はやらない、フルベンチ後に 27B / 35B-A3B (mlx-lm と oMLX も比較)
- [NAX 対応機が本番](nax-target-machine.md) — 自前カーネルは NAX 機で auto=off、NAX 専用命令は使わない、NAX 機で A/B を取り直す
- [QSA の tail は HF ではクエリごと](qsa-tail-semantics.md) — fastmlx は global tail で S>1 の行が自分を見ていない。修正後は teacher を作り直す
- [代金ゼロの改善は小さくても入れる](zero-cost-wins-adopt.md) — 効果が薄いだけを理由に畳まない。畳むのは代金があるとき (ユーザー 2026-09-03)

---

## bench-diverse-inputs-and-winning-areas (feedback)

ユーザーの指示 2 つ (2026-09-02 夜): ベンチの入力は多様な池で取る、勝っている領域にもボトルネックがあるので調べる


1. **ベンチマークの入力は少数パターンではなく多様な池で取る。**同じ池の窓だけだと投機 (MTP / n-gram) の
   受理率が固定され、ばらつきが出ない。出力長 (短 / 長) と thinking on/off も軸にする。時間は掛けてよい
   (一晩の tier がある)。
2. **勝っている項目 (温 TTFT、25k 以上の decode) にもボトルネックはある。**勝ち負けで調査対象を切らず、
   O(文脈長) の仕事がどこに残っているかを同じ作法 (部品和 ≈ 壁時計) で出す。
3. **小さいマイクロで先に当たりを付ける。**ユーザーは「小さいやつを測ったら」と促す。モデル無しの合成マイクロ
   (`tools/sdpa_headdim_micro.py` など) で経路の性質を確かめてから in-model に進むのが速い。

**Why:** 少数パターンのベンチは片方に有利な数字を固定する。勝っている項目を放置すると次の機体や競合で
逆転される。マイクロは 1 分で終わり、in-model の A/B は 20 分掛かる。

**How to apply:** `bench/suite/` の池 6 種を使う。LANES-2026-09.md のレーン 8 (温 TTFT の中身、decode の kv 罰)。
実測は必ず GPU が空いているときに (並走中の数字を一度記録して訂正した)。
関連: [[measurement-discipline]], [[fastmlx-lanes-2026-09]]

4. **overnight tier はやらない** (2026-09-03)。計測の節目は「既知を片付けたら小さいベンチ (mlxturbo だけ) →
   仮説 A/B → mlx-serve 最新版を取り直してフルベンチ → 課題があれば再仮説」。フルベンチの後に
   Qwen3.8-27B 4-bit と Qwen3.6-35B-A3B 4-bit を加える (ビットは揃える)。Qwen 系を先に仕上げ、Gemma は後。

追記 (2026-09-03): 27B / 35B-A3B の比較相手は mlx-serve だけでなく **mlx-lm (mlx_lm.server) と oMLX** も入れる (ユーザー方針)。
Flash-Next は mlx-serve だけでよい (mlx-lm は Flash-Next の MTP を持たない)。

---

## draft-training-deferred (feedback)

draft (小さい draft model) の学習はアイデアとして残すが当面やらない。特化すると汎用性能が落ちるのが理由 (ユーザー 2026-09-04)。MTP 頭の学習禁止とは別物


ユーザー (2026-09-04 08:10、Mirai / uzu の 105 tok/s を見て): 「速度のためにはやるべきところ。いずれやる気はするが、特化すると汎用性能として使いづらいので、あえて手を出していない。アイデアとしてはありなので残しておきたい」。

**Why:** uzu の速さの半分は族ごと・用途ごとに学習した <50 MB の draft で、一般チャットでは落ちる (特化の代金)。ユーザーは汎用性能を優先する。

**How to apply:** draft の学習を提案するときは「汎用性能を落とさない条件」(広い池、受理率が落ちても遅くならない、出力は本体で決まる) を付けて BACKLOG の「後でやる: draft の学習」を指す。着手はユーザーの判断。MTP 頭の学習 (本体側、方針で無し) と混同しない。
関連: [[mirai-uzu-competitive-position]], [[local-model-verdict]]

---

## engine-goal-everyone-uses (user)

最終目標は「皆が (意識してもしなくても) 使う推論エンジン」= 製品が中に入れるバックエンド。看板は架け替え可。条件は同じ重みで負けない・族の追従・入れ替えやすさ・品質の根拠。同時接続は後の順番 (ユーザー 2026-09-04)


ユーザー (2026-09-04 11:10): 「製品の看板は架け替えてもいい。皆が使う推論エンジン (意識してもしなくても) が目標」。oMLX のような製品と看板で張り合わず、その中に入るエンジンを目指す。

**Why:** 同じ土台 (mlx-lm) の製品と GUI や人気で競っても勝ち筋が無い。エンジンとしての差 (同じ重みでの速さ、族の追従、品質の計測) が採用の理由になる。

**How to apply:** 判断の物差しは「製品が入れ替えたくなるか」: (1) 同じ重みで負けない (専用パック・独自形式に依存しない)、(2) 新しい族に 1〜2 日で 8〜9 割、(3) OpenAI 互換 HTTP と mlx-lm と同じ Python API、(4) KLD / ビット一致 / A/B の記録を出す。単独ベンチの数字だけでなく同時接続 (continuous batching) も採用には要る (27B が落ち着いた後の順番)。
関連: [[fastmlx-server]], [[lmstudio-integration-plan]], [[mtplx-competitive-position]], [[mirai-uzu-competitive-position]]

---

## fastmlx-lanes-2026-09 (project)

2026-09-03 昼時点の fastmlx (mlxturbo) の現在地、最終ゴール、レーンの判定一覧、次に見る順


**最終ゴール (ユーザー 2026-09-03)**: Qwen3.8 Flash-Next 4-bit、短文脈で 100 tok/s (MTP 頭の学習はしない)。prefill は 1.5 倍 (相手 mlx-serve の 1.2〜1.3 倍速) が現実線。
フルベンチは小さいベンチで「decode 同着以上、prefill 1.03x 以内」が出てから。その後 27B / 35B-A3B (mlx-lm と oMLX も比較)。

**現在地 (2026-09-03 朝の小ベンチ 0903d + 既定化した分)**: 冷 prefill 4k 6.1 s / 17k 28.9 s / 50k 83 s (相手 5.70 / 27.8 / 82.1)、decode 短 51.9 tok/s (相手 55.7)、温 TTFT 0.15〜0.93 s (相手 0.85〜15.9)。

**既定に入ったもの**: QSA bool マスク、GDN Metal、depth 適応 (2048 超)、T=1 gather prefill カーネル (kv ≥ 12288)、detokenizer 使い回し、prime 窓 512、MoE 畳み込み (行数 ≥ 64)、
n-gram の行取得 mmap + 背景 madvise (8k -6% / 17k -7%、decode も問題なし)、P5 sdpa 行タイル (`MLXTURBO_SDPA_ROWTILE`、-1%、KLD 差 0)。

**進行中 (台帳 `docs/research/IDEAS-2026-09-03.md`、走行中一覧はリポジトリ直下 scratchpad/INFLIGHT-<date>.md)**:
- 品質: QSA の tail を per-query に (HF / mlx-serve / oMLX と一致させる。[[qsa-tail-semantics]])。既定にしたら teacher (bf16) を作り直して KLD。
- prefill: P6 gather カーネルの uint4 load (micro -27.6%、移植と in-model 待ち)、末尾チャンクをグループに (4k -4.6%、checkpoint を保つ v2 待ち)、
  P7 MoE の行列積以外 800 ms/チャンクの内訳、P9 チャンク 8192、P3 (BM=16 混合) / P10 (BM=64 qmm) は M3 Max 向け (非 NAX で auto=on)、P8 (bf16 GEMM) は P10 次第で保留。
- decode: K2a (厳密 top-k select、通過) → K2b (2-pass vector の写し + 常駐ビットマップ) → 配線。HC 第 4 変種 (elementwise だけ融合、GEMV は MLX の qmv)。depth 4 は K2 の後。

**畳んだもの**: D1 (draft 同梱 +1.7%)、D2b、D3 (SAM -2%)、D4 (command buffer)、D5 (専門家共有)、P1 (2 stream 直列化)、K1 の mask arm、HC 融合 3 変種、ANE (INT8 の重みコピーで
メモリ増 → 見送り)、MoE 連結、G=8、PLE hoist、n-gram 先読み、take+qmm、indexer-lean、lm_head 4-bit、draft の木、厳密棄却サンプリング、group_size 128。

**Why:** 数字の出典は `docs/research/SESSION-2026-09-02-CATCHUP.md` の 2026-09-03 の節。custom kernel が decode で負けていた正体は温キャッシュの連鎖 micro
([[fastmlx-measurement-traps-0902]] の罠 12) で、decode 幅で qmv / qmm を自前に置き換える筋は構造的に不利。

**How to apply:** `docs/NEXT-SESSION-PROMPT.md` の「いまの段取り」と「compact 前の控え」から入る。GPU は `tools/biglock.sh` で直列。
関連: [[vs-mlx-serve-clean-baseline]], [[nax-target-machine]], [[bench-diverse-inputs-and-winning-areas]]

追記 (2026-09-03 14:30、ユーザー方針): qwen4_exp の最適化の後は、qwen3_5 (27B / 35B-A3B) → Gemma 4 の順に載せ、2 族目でアーキテクチャの対応表を切る。全モデルで最高は目指さず、新しい族に 8〜9 割が既定で追従する状態が狙い (docs/BACKLOG.md「アーキテクチャ追従の投資」)。融合 25 個が qwen4_exp 直書きなのが現状の障害。

追記 (2026-09-03 18:55、ユーザー決定): 他の族への対応は「フォールバック」ではなく「部品ごとの置き換え」で徐々にネイティブ化する。土台は mlx_lm の素の forward、契約が合う部品だけを差す (一部だけ速い状態を普通にする)。投機の staged forward を層の列を歩く汎用版にできるかが分かれ目。試金石は 27B。詳細は docs/BACKLOG.md「決定 (18:55)」。

追記 (2026-09-03 19:20): 現在地 = 小ベンチ 0903g (冷 prefill 4k 5.95 / 17k 30.3 / 50k 86.7、decode 51.7 / 47.5 / 47.0、lm_head 4bit)。今日既定に入ったもの: P3 mix48、P10 BM=64、P6 uint4、末尾 v2、QSA tail query、MIN_KV 8192、K2c、P7 第 2 段、lm_head 4bit パック。畳んだ: P8、P10 本体、P9、GDN reg scan、HC v4、D1、oracle knob。走行中: decode の糊 4 つ、K2c の S 依存確認、プロンプト池の凍結。

追記 (2026-09-03 20:05、ユーザー): 数日は短文脈 decode の tok/s に集中。壁は dispatch の床 (5 us × 4499 本)、手は層の塊ごとの大きい融合で本数を 1/3 に (MoE → GDN → HC)。価値の位置取り (多ターン・長文脈のバックエンド、新しい族への 8〜9 割の自動追従、品質込みの計測基盤) は合意済みだが、tok/s は当面の主戦場。

追記 (2026-09-04 03:00): 深夜に API 500 で 4 本消えて出し直し。既定に入れた: n-gram 先読み (実は走っていなかった。on + forward の前に投入、17k 冷 -2.9%)、HC の細長 GEMM に qmm_wide (ビット一致、17k -0.9%)。畳んだ: 小 kv の疎 attention (union タイル化は kv=4096 で 0.99x)、HC elem の prefill 幅拡張 (非ビット一致で tok/round -4.8%)、n-gram の late 配置。判定待ち: fused:1 (速度 -1.3〜-1.4%、品質ゲートは本番重みの参照テスト + S=1 の Δ KLD)。decode 幅だけの非ビット一致カーネルの規則 (参照への距離が素以下、丸め回数の削減方向、Δ KLD ≤ +0.0005) を通ったら CLAUDE.md の品質段落に書く。

追記 (2026-09-04 10:22): 27B レーン開始 (ユーザー 07:50)。移植 3 種 (GDN、行タイル、qmm_wide MLP) は契約で当たるが prefill の取り分は合計 1% 未満。的は decode 経路 (round 43 ms + 10 ms × draft、mlx-serve は同じ 2 本で 52 ms 対 73)。第 1 段 (段階投入 S>1、-1.9%) 着地、第 2 段 (draft chain の同期と本数) 走行中。融合もどき (バイトと同期の削減) の一覧: Flash-Next (稼働率 95%) と 27B (98.7%) には的なし、Gemma 4 (80%、norm 331 本/step) だけ窓 → レーンは Flash-Next / 27B で閉じた。順序: 27B 着地 → qwen4_exp も分岐ルートに → 35B-A3B (Qwen3.6)。Gemma 4 は 5 者煙試験 (投機なし / あり) を先に。

---

## fastmlx-measurement-traps-0902 (feedback)

2026-09-02 午後に踏んだ計測の罠 3 つ (熱、RAM 常駐 n-gram のメモリ圧、capture が GDN の融合を素通しする)


1. **熱**: GPU を 50 分回し続けると 17k prefill が 37 s → 45 s に落ちる。絶対値の比較 (相手の値との突き合わせ)
   は冷えた状態で短く取り、A/B の差だけを長い連鎖で取る。
2. **`~/models/ddalcu-ngram-sep` (RAM 32GB 常駐) は 17k で decode が 41% 遅い** (モデル 91GB との合計でメモリ圧)。
   `tools/decode_ab.py` の既定コマンド例はこの sep を使っていたので、前セッションの長文脈 decode の数字は本番より
   遅い状態を見ていた可能性がある。長文脈の計測は `~/models/ddalcu-ngram` (interleaved) で。
3. **`spec_flash.capture()` が `GatedDeltaNet.__call__` を丸ごと差し替える**ので、GDN 内の融合 (gdn_prework /
   rms_norm_gated) は decode で発火しない。過去の A/B (`bench/results/*-ab2.json`) は全部 `fired={}`。
   融合の効果を測る前に発火カウンタを見ること。

4. **`tools/decode_ab.py` のプロセスが終了時に固まって Metal のメモリを握ったまま残ることがある**
   (RSS は小さいので ps では分からない)。次のジョブが 2 モデル分で走ってスワップ 25GB になり、
   計測が全部無効になった。連鎖スクリプトでは各ジョブ後に `pkill -f "decode_ab.py --knob"` と
   `sysctl vm.swapusage` の確認を入れること。
5. **xctrace の Metal System Trace は MLX (python) の GPU 区間を拾わない** (launch でも attach でも)。
   カーネル単位の比較はこの道具では出ない。

**Why:** どれも「効果ゼロ」や「遅い」という誤った結論を生んだ実例。

**How to apply:** 計測結果を読む前に、熱・メモリ・発火の 3 つを確認する。
関連: [[measurement-discipline]], [[vs-mlx-serve-clean-baseline]]
6. **GPU の並走中に取ったマイクロの数字を一度そのまま記録してしまった** (2026-09-02 夜、カスタムカーネル固定費
   24.8 us → 冷えた単独では 3 us)。scratch のマイクロでも `pgrep` で他の GPU プロセスを確認してから走らせる。
7. **サーバーは `--ngram ~/models/ddalcu-ngram` を渡さないと読み込みで落ちる** (n-gram の shard 重みが無い
   エラー)。サイドカーの自動検出はサーバー経路には無い。`bench/self_snapshot.py` にも `--ngram` を渡すこと。
8. **`os._exit(0)` で終わるスクリプトは stdout を flush してから** (バッファが捨てられて出力が消える)。
9. **`pgrep -f` のパターンが自分のシェル行に当たる**。`for ... pgrep -f "foo"` のような待ちループは、
   自分自身に当たって即抜ける。パターンをスクリプトの外 (別ファイル) に置くか、`python` で絞る。
10. **GPU を使う全ジョブ (親の連鎖も agent も) を `tools/biglock.sh` で包む。**pgrep の門番だけでは
    agent の biglock 付き実行と親の生の実行がすれ違い、91GB のモデルが 2 本載って片方が死んだ (2026-09-03)。
    biglock はネストしない。優先順位は flag ファイルで付ける。
11. **`mx.metal.start_capture` は 91 GB のモデルでは使えない** (1 forward の capture が 35 分・68 GB、2026-09-03)。
    カーネル別の帰属は部品の連鎖計測で。長い GPU ジョブは開始後 10 分で進捗 (出力サイズ) を見ること。

追記 (2026-09-03 12:00): **罠 12 = 温キャッシュの連鎖 micro**。重み 1 組 (3〜7 MB) を 200 回読む連鎖では、並列度の低い自前カーネルが冷の DRAM レイテンシを
隠せない負けが見えない (HC 融合: 温 +13 us、冷 (48 組 157 MB 巡回) +78 us/回 × 97 層 = in-model の +8 ms の全量)。カーネルの連鎖 micro は重みを
100 MB 超巡回させて冷やすこと。decode 幅で qmv / qmm を自前に置き換える筋は構造的に不利 (MLX の qmv は行ごとに threadgroup を立てて隠す)。
- 罠 13 (2026-09-03): 待ち手が走っている最中に tools/biglock.sh を同じ inode に上書きしない。zsh はスクリプトを読み進めるので、ループ後の続きが化ける (`command not found: nue`)。直すときは別名に書いてから mv で入れ替える。長文脈 (50k) の A/B は 1 往復で足りる (6 本が ±0.3 s)。3 往復で 28 分かかった。
- 罠 14 (2026-09-03): プロセス起動直後の最初の計測行は +7〜9% 遅い (decode_ab の温めでは消えない、回文順でも位置 1 は打ち消せない)。A,B,B,A の A に ms/round で約 2% の不利が付く。常駐 worker (tools/ab_daemon.py) は burn-in を入れる。過去の decode 側 A/B の小さい負け (+1〜2%) はこの範囲。
- 罠 15 (2026-09-03): biglock の先着順と SIGSTOP を組み合わせない。止めた古い札に新しい待ち手が譲り続けて固まる。順番を変えるなら kill して投げ直す。
- 罠 16 (2026-09-03): 1 プロセスの A/B で depth 適応の EMA が variant をまたいで持ち越される (case 0 の A だけ全 round depth 2 になった)。decode 側の A/B は variant ごとに DepthController を作り直すか、depth を固定して測る。K2c の -0.7% (実は -4.1%) はこれと burn-in 無しの合わせ技。
- 罠 17 (2026-09-03): ベンチのプロンプト池 (`tools/_bench_text.py`) はリポジトリの docs/**/*.md + README + .py そのもの。セッション中に docs を編集すると同じトークン位置の窓の中身が変わり、走行をまたぐ比較 (別プロセス、小ベンチどうし) の prompt が別物になる。池を凍結したファイルに置き換えること。
- 罠 18 (2026-09-04): API の 500 でサブエージェントが 4 本まとめて消える。台帳無しのエージェントは成果の途中経過が残らない (再開もできない)。長い実装・計測のエージェントには `scratchpad/agent-<name>.md` への節目ごとの追記を必須にし、親は再投入の brief に「作業ツリーに残っている途中成果」を書く。
- 罠 19 (2026-09-04): op 1 本ごとに `mx.eval` して測った us は投入・同期の往復 (M3 Max で 165〜270 us) が床になり、小さい op の実費を 1 桁過大に見せる (計数ソートの見込み -0.4% → 実際 0.04%)。部品の費用は層ぶん (48 層) を 1 本のコマンドバッファに積んで us/層 で測る。
- 罠 20 (2026-09-04): 混在した作業ツリーから hunk をキーワードで選別して commit すると、キーワードを含まない断片 (5 行の `if inv is None:`) が紛れて未束縛の変数になり、Flash-Next の実 prefill が落ちた (回帰は 13f0d21〜10:20、その間 27B しか流していなかったので気付かず)。部分 stage の後は必ず「その経路を実際に踏むテスト」(prefill 幅の in-model 1 本) を通す。合成の fingerprint は行数 < 64 で fold を踏まない。

---

## fastmlx-remaining-work (project)

残作業の入口。docs/NEXT-SESSION-PROMPT.md の「いまの段取り」と「compact 前の控え」が正、走行中の一覧はリポジトリ直下 scratchpad/INFLIGHT-<date>.md (セッションの scratchpad は消える)


残作業の一覧は repo 側に置く: `docs/NEXT-SESSION-PROMPT.md` (段取りと判定線)、`docs/research/IDEAS-2026-09-03.md` (案の台帳と判定)、
リポジトリ直下 (untracked) の `scratchpad/INFLIGHT-<date>.md` (走行中のエージェントと GPU 連鎖)。**セッション用の scratchpad (/private/tmp/claude-501/...) はプロセスが再起動すると空になる** (2026-09-04 に台帳と小ベンチのスクリプトを失い、transcript から復元した)。台帳・ベンチのスクリプトはリポジトリ直下の scratchpad/ に置く。この memory には順番だけ持つ:
測定 → proof-of-life → 通った案の in-model → 小さいベンチ (6 文脈、mlxturbo だけ、10 分の静かな窓の後) → フルベンチ (条件付き) → 27B / 35B-A3B。

**Why:** ユーザーの指示「何をやるか分からなくならないように記録を残す」(2026-09-03)。GPU の取り合いで計測が無効になった後、biglock で直列化した。

**How to apply:** 新セッションは NEXT-SESSION-PROMPT.md から入る。結果は CATCHUP の末尾に節を足し、既定を変えたら CLAUDE.md の knob 段落も直す。
GPU を使う全コマンドは `tools/biglock.sh` で包む。フルベンチと overnight はユーザーの指示があるまで走らせない。
関連: [[fastmlx-lanes-2026-09]], [[fastmlx-measurement-traps-0902]]

---

## fastmlx-server (project)

fastmlx の OpenAI/Anthropic/Responses 互換サーバー (fastmlx-serve) の構成と、実装時に踏んだ落とし穴


2026-08-28 実装、コミット済み。`fastmlx-serve` で起動。OpenAI の `/v1/chat/completions` と `/v1/models` と `/v1/completions`、Anthropic の `/v1/messages`、OpenAI Responses の `/v1/responses` を非ストリーム + SSE で話す。tool calling 対応済み。テストは `bench/test_server.py` に 111 件 (`bench/test_spec_phase0.py` の 6 件は別件で元から赤)。

- **生成は `mlxturbo/runner.py` の `build_runner` 経由**。SpecEngine を試し、契約検証に失敗したら `FallbackRunner` (mlx_lm.stream_generate) に落ちる。**Flash-Next / qwen4_exp は必ずフォールバック側**(SpecEngine は qwen3_5 の形を要求し、qwen4_exp には `model.language_model` も `fa_idx` も無い。バグではない)。**Qwen3.6-35B-A3B は契約を満たすので spec 側**に乗る。gemma/kimi/glm はフォールバックなので「載せれば喋る」。
- **融合 HC カーネルは build_runner が有効化する。**効くのは FallbackRunner 経路 = Flash-Next。サーバーがこれを呼び忘れていた時期があり 13-20 tok/s に落ちていた(修正後 29.3-30.3 で CLI 基準と一致)。**サーバー経由の速度がおかしいときは真っ先にここを疑う。**
- リクエストは直列化(asyncio.Lock + 単一スレッド executor)。並列化は `mlxturbo/batch.py` に別途あるが未配線 ([[fastmlx-remaining-work]])。
- テキスト専用なので画像等の非テキストブロックは 400。
- `--served-model-name` で advertise する id を決め、不一致は 404、レスポンスは常に実体の id。

**Codex CLI には `/v1/responses` が必須。**`wire_api = "chat"` は 2026-02-01 に削除され `"responses"` のみが有効な値なので、Chat Completions だけでは設定の工夫で繋ぐ手が無い。構造も別物 (`input` / `instructions` / フラットな `tools` / `output` 配列 / 型付き SSE イベント)。`previous_response_id` と `store` は永続化の設計になるので 400 で断っている。

**踏んだ落とし穴 (再発しやすい順)**

1. **テンプレートが先に `<think>` を開く。**描画済みプロンプトの末尾が `<|im_start|>assistant\n<think>\n` になるモデルでは、モデル自身は `<think>` を生成しない。`ThinkingRouter` を detect から始めると「考えていない」と誤判定して思考が全部本文に混ざる。`_prompt_already_thinking()` で判定して thinking フェーズから始める。
2. **恒等値のサンプリングパラメータ。**`top_p: 1.0` や `frequency_penalty: 0` は分布を変えないので、SpecRunner が非対応でも通さないといけない。**実クライアントはほぼ全てこれらを既定で送るので、弾くと接続すらできない。**さらに、通した後 `params` から取り除かないと `SpecEngine.generate()` が未知の引数で落ちて 400 が 500 に化けるだけ。
3. **Anthropic の履歴に `thinking` / `redacted_thinking` が来る。**サーバー自身が返しているブロックであり、拡張思考 + tool use の規約では次ターンに送り返す必要がある。弾くと Claude Code の 2 ターン目が必ず 400。

**テストの穴に注意。**fake の runner が `**kwargs` で何でも受け取ると、「サーバーが検証を通す」ところは検証できても「通した結果 runner の境界で落ちる」ことを検証できない。実際に単体 111 件が全部緑で実サーバーだけ 500 になった。**fake は実物の厳しさに寄せること**(知らない引数なら TypeError)。

**gpt-5.6-sol のレビューで 12 件出て全部直した。**実使用で踏むもの: フォールバック経路で detokenizer 二重投入(出力が壊れる)、ストリーム切断でロックが先に解放されワーカーが共有状態を触り続ける、`max_tokens: "abc"` で 500、`stream_options.include_usage` 無視(GUI のトークン数表示がこれに依存)、`max_completion_tokens` 無視、`stop`/`stop_sequences` 無視、`model` を検証せず echo。

**opencode で E2E 済み** (2026-08-28)。`opencode run --pure` で Glob → Read → Edit の連鎖が回り、ディスク上のファイルが実際に書き換わるところまで確認。設定は `/Users/ht/.config/opencode/opencode.json` に `@ai-sdk/openai-compatible` で `baseURL` を `/v1` まで書く形 (LM Studio 用の記述が既にあるので複製すればよい)。`--format json` で生イベントが取れるので E2E はスクリプトでアサートできる。`opencode stats` でトークンとコストが出る。GUI なら **Chatbox** (OpenAI 互換と Anthropic の両方を話せる、認証が無いので API キー欄はダミーで可)。LM Studio は登録口が無く `openai-compat-endpoint` プラグイン経由 ([[lmstudio-integration-plan]])。

**配布用の堅牢化済み (2026-08-29)。**`--api-key` (Bearer / x-api-key 両対応、未指定なら認証なし)、`--max-queue` (超過 503)、SSE keepalive 15 秒 (97k プレフィル約 3 分の間クライアントに切られないため)、graceful shutdown (listener を開けたまま 503 で断る — should_exit 直行だと TCP 拒否になる)、`/health` に queue_depth と version。接続手順は docs/SERVER.md。プレフィル分割で文脈長の壁 43k→85k 超、prompt cache は位置チェックポイント (GDN 状態 66MB/個をチャンク境界でスナップショット) で Claude Code の 2 ターン目が reused=96256。

**Flash-Next の MTP がサーバーから効く (2026-08-29, d9d08b0/d17fedd)。**`runner: "flash_spec"` (FlashSpecEngine、深さ 1、4bit)。**MTP は自動発見**: --mtp 明示 (失敗は exit 1) → モデル重み内の mtp.* → `MODEL_DIR/mtp.safetensors` サイドカー → 無ければ fallback + /health に fallback_reason。**v-l にはサイドカーへのリンクを置いてあり、フラグ無し起動で flash_spec が上がるのが既定運用**。`--require-runner flash_spec` を起動スクリプトに書けば降格自体が起こらない。実測 v-l で非投機比 1.26-1.44 倍、サーバー経由 decode 37 tok/s。temperature は「検証 logits からサンプルし draft 一致時のみ 2 トークン目もサンプル」で正確。**落とし穴: MTP ロードの mx.eval を一括にすると Metal watchdog (GPU Timeout) を毎回踏む** — サイドカーが外付け SSD の mmap 遅延読みで、直前の 72GB モデルロードがページキャッシュを追い出すため。テンソルごとに eval を分割して回避済み。同じ形の GPU Timeout を見たら「1 コマンドバッファに遅い I/O 待ちを積んでいないか」を最初に疑う。session は FallbackSession 流用 (純粋追記のみ、部分一致は新規スロット)。

**2026-08-29: 公開準備で 3 つ決まった。**(1) **`fastmlx` から `mlxturbo` へ改名**中 — PyPI と GitHub の `fastmlx` は Prince Canuma (Blaizzy) の「production ready API to host MLX models」で直接競合、`pip install fastmlx` が別物を入れる。`mlxturbo` は PyPI 空き・GitHub 同名 0 件。(2) LICENSE は MIT (著作権者 akakeishin)。(3) **「汎用高速推論ランタイム」とは名乗らない** — 速いのは flash_spec と spec の 2 アーキテクチャだけで、他は素の mlx-lm と同速。汎用を名乗ると Llama 利用者が「速くない・top_p で 400・embeddings 無し」を最初の 10 分で踏む。Kimi K3 のレビュー (23 項目、P0-P3) が判断の土台。

**flash_spec が 2000 トークンで OOM kill されていた (2026-08-29、修正済み)。**`capture()` が `GatedDeltaNet.__call__` を差し替え、巻き戻し用に全位置ぶんの状態を fp32 で確保する (1 層 1 位置 3 MiB x 36 層、T=2000 で 216GB)。プレフィルの最終チャンクは `cap.hyper` の最終位置しか使わないのに丸ごと捕獲していた。`capture(model, light=True)` を追加して `GatedResidual` だけ差し替える形に。**hyper も logits も 48 層のキャッシュもビット一致**で、近似ではない。**トレースバックが出ずプロセスが消え、死後にメモリが空いて見えるのは、カーネルの OOM kill (`memorystatus: killing largest compressed process`) の特徴** — 同じ形を見たら `log show` を最初に見ること。短いプロンプトでは踏まないので、長さを振る試験が要る。

**2026-08-30: 公開準備が一通り終わった (12 コミット、テスト 289 -> 355)。**P1-P2 全項目済み。非恒等サンプリングは 400 でなく**リクエスト単位で非投機に降格**する (実クライアントは top_p を既定で送るので、400 のままだと看板構成が最初の 1 発で弾かれていた)。継続バッチングは `--max-batch` で opt-in、既定は直列、FallbackRunner 限定 (投機との両立は未測定)。投機の汎用化は `--draft-model` と `--lookup-spec` を追加したが**どちらも条件付き** — 前者は適合する小型ドラフトが手元に無く未測定、後者は繰り返し入力で +18% だが自然文では -32% (`async_eval` の二重バッファが無いため)。コメントと docstring は英語化済み、ユーザー向け文字列 (ログ・エラー・argparse help) は日本語のまま。

**「fake が実物より緩い」を 1 日で 3 回踏んだ。**(1) 単体テストの fake runner が `**kwargs` で何でも飲み、単体 111 件緑のまま実サーバーが 500。(2) チェックポイント復元が MLX 演算をするのに、全体一致の経路はテンソルに触らないので単体では見えず、本物の部分一致で初めて 500。(3) バッチ化の合成検証が float32 で、`mx.quantized_matmul` のバッチ長依存の丸めを再現せず「完全一致するはず」という誤った期待値を作った。**fake は実物の厳しさに寄せること。実クライアント・長いプロンプト・実モデルでしか出ない類がある。**

**ビット一致を合否基準にしてはいけない場面がある。**`mx.quantized_matmul` はバッチ長依存の丸めをするので、プレフィル分割もバッチ化も「分割あり/なしで完全一致」は原理的に達成できない。守るべきは**同一構成内の決定性**と、文章の健全性、KLD が量子化ノイズ水準 (v-fast6 で 0.00378) に収まること。vLLM も llama.cpp もクロス構成のビット一致は提供していない。

**2026-08-30: 三箇所すべて公開完了。**GitHub https://github.com/akakeishin/mlxturbo (public, MIT, 145 コミット, テスト 355)、PyPI https://pypi.org/project/mlxturbo/0.1.0/ (`pip install mlxturbo` がクリーン venv で動くことを確認済み)、HF https://huggingface.co/Kandandan/Qwen3.8-Flash-Next-MLX-4bit-MTP (26 ファイル 82.6GB、MTP サイドカー同梱、全シャードのダウンロードを確認)。**`uv publish` は `.pypirc` を読まない** — twine を使うか `UV_PUBLISH_TOKEN` を渡す。HF の public ストレージは無料枠が best-effort で明示上限なし (private は 100GB)、77GB の公開配布に課金は不要だった。

**packaging の落とし穴。**hatchling の `include` は既定の「gitignore されていない全部」に**足す**ので、未追跡だが gitignore もされていない `bench/results/*.json` が入って sdist が 102MB (PyPI 上限 100MB 超) になった。`only-include` を使うこと。また vendor (`qwen4_exp.py`) は `mlxturbo/_vendor/` へ移し、`Path(__file__).parent` 基準で解決する — `tools/` に置いたままだと wheel に入らず、pip で入れた環境で Flash-Next が読めない。**クリーンな venv で `import mlxturbo` → `import mlx_lm.models.qwen4_exp` が wheel 内の `_vendor` を指すことを確認するのが唯一の実効的な検証。**

**`bench/results` は履歴ごと削除した (2026-08-30)。**128 ファイル / 8.3MB、個人パス 450 箇所。.git は 16MB → 1.6MB。`docs/BENCHMARKS.md` の対応表は「主張 → 再現コマンド → 当時の観測値」の形で残し、JSON へのリンクだけ外した。`.gitignore` に `bench/results/` を追加済み。**測定記録を残すか外すかは判断が割れる** — 対応表は「懐疑的な読者が検証できない」という指摘に応えて作ったもので、実際にこの突き合わせで根拠の無い性能主張を 2 件見つけている。今回は「他人の機械の測定値そのものには意味が薄い」を採った。

関連: [[flash-next-speed-verdict]] [[fastmlx-remaining-work]] [[measurement-discipline]]

追記 (2026-09-04 11:05、ユーザー): 位置づけは「mlx-lm の LLM 特化版」が一番近い。汎用性寄り (同じ重み・同じ形式・同じ品質で、契約で当たる部品により族の追加に 8〜9 割ついていく) で攻める。相手: mlx-serve (Zig、族ごとの手調整)、MTPLX (専用パック)、uzu (独自形式 + 蒸留量子化 + 学習 draft)、oMLX (GUI 一体・全モダリティ)。比較は相手の推奨設定で公平に測る (弱く測って勝っても意味が無い)。

---

## flash-next-speed-verdict (project)

旧レーンの記録。v-fast6 は削除済みで常用は mlx-serve に移った。bf16 も削除、逆変換で代用できる


2026-08-28 時点の到達点。朝はロードも通らなかったものが動くようになった。

| | ms/token | tok/s | 容量 | KLD | top1 |
|---|---|---|---|---|---|
| 出発点 (v-stream) | 52.32 | 19.1 | 92GB | 0.00260 | 0.9881 |
| **v-fast6 + 融合** | **32.67** | **30.61** | 91GB | 0.00378 | 0.9850 |
| v-96 + 融合 | 29.59 | 33.79 | 67GiB | 0.02902 | 0.9519 |

MTP 無しで最低ライン 30 tok/s を超えた。内訳は hyper-connections の融合カーネル
(-14.6ms)、ビット配分の見直し (-2.5ms)、n-gram の並列 pread (-2.1ms、未合成)。

手元に残っているのは **v-l (77GB) + ngram-4bit サイドカー (30GB)** と、
比較用の ddalcu 一律 4bit pack (98GB)。v-s / v-m / v-xl / v-fast6 と jundot 版は
2026-08-30 にユーザー判断で削除 (384GB、内蔵の空きは 463GB)。
27B 系 154GB と v-96 67GiB は 2026-08-28 に削除済み。

v-96 の実体は消え、品質の JSON も bench/results の履歴ごと削除で失われた。
残っているのは docs/STATUS.md の速度と、mlxturbo/convert_flash.py のレシピ定義。

**「335GB の再取得が要る」は誤りだった (2026-08-30 に確認)。**bf16 の元は
外付けに残っている:

    /Volumes/Mobile SSD/models/Qwen3.8-Flash-Next   335GB
    131 shard、全部 BF16、vision_config あり

焼き直し、レシピ変更、n-gram サイドカーの作り直し、MTP の抽出は、
**いつでも始められる**。この誤りを前提に「焼き直しは高い」と判断していた
時期があるので、そこは判断し直すこと。

できることは残っている: 課題レーン、カーネル側の MoE/GDN。bf16 参照ダンプも
同じ削除で失われているので、品質評価をやり直すには参照の取り直しが要る。

残作業は [[fastmlx-remaining-work]]。測定の作法は [[measurement-discipline]]。

## 2026-08-30 追記

**bf16 の元 (外付けの 335GB) は削除した。**Flash-Next のパックを焼く理由が
無くなったため ([[local-model-verdict]])。焼き直すなら再取得から。

## 2026-08-31 追記: この文書は履歴になった

`v-fast6` も含めて**焼いたパックは全部削除した。**常用は ddalcu の一律 4bit を
mlx-serve で回す形に決めた ([[local-model-verdict]])。ここの 30.61 tok/s は
うちのエンジンでうちのパックを回した数字で、いまの動作点ではない。

bf16 の元 (335GB) も削除した。ただし **`tools/from_mlx_serve.py` で ddalcu の
パックを mlx_lm 形式に逆変換できる** (名前を戻す、MTP を抜き出す、RMSNorm から
1 を引く、config を実体から起こす、n-gram をインタリーブに戻す)。同じ重みが
手に入るので、エンジン間の比較にはこちらで足りる。再取得が要るのは
**焼き直すとき**だけ。

## 2026-09-03 追記: bf16 の元は外付けに再取得済み

`/Volumes/Mobile SSD/models/qwen38fn-bf16` (362 GB、2026-09-01 10:40 完了)。「削除した」は 8/30 の話で、
その後に取り直されている。lm_head 4-bit の本焼き (rebit ではなく真 bf16 から) はいつでも始められる。

---

## gemma4-lora-lane (project)

Gemma 4 12B 高ランク LoRA 学習レーン (/Users/ht/dev/gemma4-lora) の方針と実測基準


2026-08-28 開始。fastmlx とは別リポジトリ `/Users/ht/dev/gemma4-lora`(uv, mlx-vlm 0.6.17, mlx 0.32.2)。目的: gemma-4-12B(画像タスク)への高ランク LoRA を M3 Max 128GB で最高速度。

決定事項:
- 学習は **bf16 ベース**(ユーザー指定。QLoRA にしない)。重みは mlx-community/gemma-4-12B-it-bf16 (22GB、DL 済み)。
- **CLI/データ形式は CUDA 側 (transformers+peft+TRL) と揃える**(ユーザー指定)。jsonl は messages+images、アダプタは PEFT 互換エクスポートも出す。
- mlx-vlm 付属 trainer は使わない。Gemma 4 では loss mask が既定/completion どちらも壊れている(thinking-channel prefix とトークン列不一致)、vision_embedder が勝手に学習される、LoRA fp32、LR schedule 無し。`load()`/VisionDataset/LoRALinear/save_adapter は再利用し、ループだけ自前。
- gemma-4-12B は **encoder-free VLM**(model_type: gemma4_unified)。ViT 無し、48px パッチ→Linear、画像 1 枚 ≤ 280 soft token。

実測基準(実 12B 形状、rank128、Adam): B=2/S=1024 で 10.7–12.0 s/step、171–192 tok/s(roofline 13.0 TFLOPS の 73%)。grad ckpt は 1.43x 遅いので不使用。CE 8 分割で peak −11.8GB。wired limit 115.4GB、ckpt 無しで ~3,000 token/step が上限。rank256 は +31% 遅く割に合わない。

2026-08-28 プロファイル確定 (/Users/ht/dev/gemma4-lora/tools/profile_step.py、同一プロセスラウンドロビン必須 — サーマルで 5 分 15% 落ちる): 1 ステップの 88% がデコーダ 48 層 GEMM(層時間の 97.5% が GEMM、fwd 11.7 TFLOPS / bwd dX 9.6)、14.6% が lm_head+CE。elementwise・隙間・optimizer・.item() は全てシロ。効く手は (1) head を loss 対象行だけ gather(上限 14.2%、loss 密度依存)、(2) LoRA A/B 横結合(上限 4.7%)、(3) bwd dX の GEMM レイアウト(7.9%、MLX と勝負なので低確度)。M3 Max は行列ユニット無しで GEMM は既にほぼ天井、伸びしろはアルゴリズム側。

実データ: /Volumes/Mobile SSD/hidreamft-data/datasets/g0/ の WebDataset tar 30 本(30 万件、256×256 JPEG + InternVL キャプション平均 88 トークン)。**注意: processor は入力解像度に関係なく画像 1 枚 = 256 soft token (16×16 grid) を出す**ので 1 サンプル中央値 ~358 トークン。バッチは行数でなくトークン数で決める(~3,000 token/step 以下、B=8 で peak 76.5GB。B=16 は 117GB でページングし 1 step 61 秒になった)。

2026-08-28 最適化ラウンド完了: (1) gather-head(loss 対象行だけ lm_head+CE に通す)を本採用 — 実データ A/B(B=10, 7 反復)で −11.1%、実密度では勾配ビット一致、チャンク CE と mx.checkpoint を削除できた。実データ 30 step: 14.39±0.96 s/step、199 tok/s、loss 2.26→1.07、B=8。(2) LoRA 横結合は実測ノイズ以下(符号が反転する差)で既定 off の `--fuse-lora` に格下げ(dropout 0 必須、mlx-vlm 0.6.17 の attention forward を複製しているので version 固定)。(3) 残弾は bwd dX が fwd より 12% 遅い件(上限 7.9%、MLX matmul と勝負、低確度)のみで、宣言基準(1.15x 未満で撤退)により打ち止めが妥当。tar ローダは /Users/ht/dev/gemma4-lora/gemma4_lora/webdataset.py(pread + ヘッダ走査キャッシュ)。A/B は /Users/ht/dev/gemma4-lora/tools/ab_optimizations.py で再現可能。

2026-08-28 ユーザー実測: B200 側は同ジョブで**ちょうど約 80 倍**(見積もり 80〜100 倍が的中、B200 の MFU 換算 ~34%)。分業「Mac=レシピ検証、B200=本番」は数字どおり成立。CUDA 側への申し送りは /Users/ht/dev/gemma4-lora/docs/HANDOFF-HIDREAMFT.md。

2026-08-28 実装完了。`/Users/ht/dev/gemma4-lora/train.py` + `gemma4_lora/`(data/lora/loss/export)。スモーク(32 件合成画像、r=128/α=256、B=8): loss 3.80→0.0005、165 tok/s、peak 62.2GB、生成でターゲット文を再現。completion マスクは実 processor 出力のデコードで検証済み(画像 256 トークン混入ゼロ)。PEFT 互換エクスポート(656 tensors)と mlx 再読込は /Users/ht/dev/gemma4-lora/tools/check_adapters.py で検証済み。落とし穴 2 つ: MLX AdamW は `bias_correction=False` が既定で初期 30 倍更新になり発散する(torch と揃えるため True 必須)。学習は `<|turn>model\n` 直後に答えを置くが、生成は thinking-channel prefix が挟まる(`--append-thought-prefix` で選択可)。B=16 は 190 tok/s に見えたが 3 step の短測でサーマルドリフト(181→152 tok/s)と交絡しており未決着。

関連: [[measurement-discipline]] [[measurement-discipline]]

---

## handoff-in-docs (feedback)

親セッション等への申し送りは毎回ドキュメント (docs/) に書き残す。チャット内だけの申し送りは不可


申し送り事項は毎回リポジトリのドキュメントに書く。fastmlx では docs/KERNEL-HANDOFF.md
(カーネルセッション→親セッション向け) を使う。

**Why:** セッションをまたぐとチャットの申し送りは失われる。親セッションが読むのは
docs だけ。ユーザー指示 (2026-08-27)。

**How to apply:** 発見・保留・依頼事項が出たら、最終報告の前に docs 側へ追記して
コミットする。重い作業は codex の sol に任せてよい ([[measurement-discipline]])。

---

## lmstudio-integration-plan (project)

LM Studio 統合の旧方針。前提だった「GUI が無いと使わない」は MLX Core.app の常用で別の形で解決した


fastmlx を LM Studio から使う件の調査結果と決定 (2026-08-26、実装は保留中)。

調査で確定した事実:
- LM Studio にサードパーティ製エンジンを公式「LM Runtimes」枠 (llama.cpp / mlx-engine の場所) に登録する口は存在しない。Extension Packs は既存 llama.cpp の GPU バックエンド差し替え専用
- 最接近は Generator プラグイン (Node.js、モデル選択ドロップダウンに並ぶ、サンドボックス無しで child_process 可)。公式 lmstudio/openai-compat-endpoint プラグインがテンプレートになる
- ユーザーは「GUI がないと fastmlx を使わない」。人気獲得は目標にしない

決定 (opus-advisor 諮問済み):
1. OpenAI 互換サーバー (/v1/chat/completions SSE + /v1/models) を新規実装し、openai-compat-endpoint プラグインで手動リンクする案を先行。ChatSession の LCP 差分 prefill を HTTP 越しに生かす設計が本丸
2. 専用 Generator プラグインは保留。反転条件: 3 モデル以上を日常的に切り替えるようになった時、または openai-compat 経由で reasoning 表示/ストリーミング/停止が壊れると実機で判明した時
3. mlx-engine への MTP 移植 PR はやらない。反転条件: mlx-engine 側が自発的に MTP 対応を始めた時 (その場合も PR でなく計測データ提供に回る)
4. 常駐は launchd LaunchAgent + 遅延ロード。GPU 計測の「同時 1 プロセス」規律と衝突するので、ベンチ中はサーバーを落とす手順が必要

実装前に潰す設計論点 (机上で決めない):
- LM Studio はタイトル生成などの副次リクエストを同じエンドポイントに投げる可能性が高く、単一セッション前提だと prefix 再利用が壊れる
- 前ターンの think 部分の再送形式で履歴の追記性が壊れるか
- 両方とも、ダンプサーバーで実リクエスト列を 1 会話ぶん観察すれば確定する。stdlib 製の dump_server を一度実装したが worktree ごと破棄済みで、リポジトリには存在しない (会話ログから復元可能。<think> タグ入り応答で再送形式を観察する仕掛け)

関連: [[mtplx-competitive-position]]

## 2026-08-31 追記: 前提が別の形で解決した

この計画の出発点は「GUI がないと fastmlx を使わない」だった。実際には
**mlx-serve の GUI (MLX Core.app) を /Applications に入れて常用する**形に
落ち着いたので、LM Studio 側に口を作る動機は消えた ([[local-model-verdict]])。

決定 1-4 と反転条件は、mlxturbo を再び常用する場面が来たときのために残す。
`mlxturbo hub` と `/api/status`、`app/` のメニューバーアプリは実装済みなので、
必要になれば土台はある。

---

## local-model-verdict (project)

ローカル利用は Qwen3.8-Flash-Next の一律 4bit を mlx-serve で。パック改良と MTP 貢献の両レーンは実測で閉じた (2026-08-30)


**手元で常用するのはこれ。**

    "/Applications/MLX Core.app/Contents/MacOS/mlx-serve" \
      --model ~/models/ddalcu-flashnext-serve-4bit \
      --serve --host 127.0.0.1 --port 11234 --mtp

`--mtp` は必須 (MoE では既定で切れている)。付けて 59.6 tok/s、無しで 50.4。
GUI は `/Applications/MLX Core.app` を開けば同じものが立つ。

## 2026-08-30 に閉じたレーン (すべて実測)

| レーン | 結果 |
|---|---|
| mlx-serve のエンジンで勝つ | **同じ重みで 52.1 対 39.3**。相手は投機を切った状態でこちらの投機込みに勝つ |
| mlx-serve の MTP に貢献 | 穴が無い。受理率 65-77% で設計どおり動いていた |
| **4bit より上のパック** | **top-1 一致が 98.12% -> 98.33%。10GB 積んで 0.2 ポイント** |
| 低ビットパック (2bit) | 28% 小さくても per-token の読み出しは 9% しか減らず、受理率が落ちて速度は帳消し。top-1 93.99%、最悪は code-debug |
| ds4 の MTP | 検証が償却しない。**文献どおり** (arXiv 2506.20675 / 2607.12696) |

詳細はすべて docs にある ([[handoff-in-docs]]):
`docs/research/MLX-SERVE-PACK-FORMAT.md` / `ENGINE-COMPARISON.md` /
`FLASHNEXT-QUALITY-TIERS.md` / `DS4-MTP-SURVEY.md`

## 再開条件 (2026-08-31 追記)

**この論点は 2026-08-30 に一度閉じたのに、翌日「読めば追い越せそう」で再燃した。**
閉じたときに再開条件を書かなかったのが原因なので、ここに書く。

「mlx-serve を decode で追い越す」を再開してよいのは、次のどれかが起きたとき。

1. **decode 中の GPU idle が 30% 以上**と分解測定で出たとき。差の主因が
   カーネルでなく Python 側の隙間ということなので、カーネル再開なしに詰まる。
   **15% 未満なら kernel-bound 確定で閉じたまま。**閾値は測る前に宣言した
2. **分布保証が必須で mlxturbo しか使えない具体的用途**が現れ、そこで decode
   速度が採用の妨げになったとき。用途が無いまま速度だけ追うのは順序が逆
3. 相手のパックか設定が変わって、同一条件の比較で差が 1.2 倍以内に詰まったとき

**潰した容疑者:** 「52.1 は lossy 既定 (`--decode-attn-quant`) 込みの数字では」
→ 実測で否定。ON 60.8 / OFF 59.3 tok/s (短文脈)、長文脈は 47.6 / 47.5。
このフラグが効くのは dense (bf16/f16) の注意を積んだモデルで、Flash-Next の
注意はパックの中で既に 4bit なので余地が無い。**比較は速度対速度で成立していた。**

**未測定:** うちのエンジンの投機 OFF の素の decode。「投機込みで負けた」と
言いながら投機の寄与を測っていない。`tools/from_mlx_serve.py` で逆変換した
`~/models/ddalcu-mlxlm` があれば測れる。

## 再開するときに知っておくこと

- **bf16 の元 (335GB) は削除した。**焼き直すなら再取得から。ただし
  **`tools/from_mlx_serve.py` で ddalcu パックを mlx_lm 形式に逆変換できる**
  ので、エンジン間の比較には再取得が要らない
- **decode は帯域の 61%** しか使っていない。重みを削る方向の効きは上限 5% 前後で、
  残り 39% は同期待ち・n-gram の行取り・indexer・カーネル起動のどれか (未分解)
- MoE の速いカーネルが受けるのは **2/4/8 bit** だけ。5 や 6 を混ぜると黙って落ちる
- **測る前に説明を組み立てない。**この日は 5 つの仮説を立てて 5 つとも外した
  ([[measurement-discipline]])

追記 (2026-09-03 18:55、ユーザー): ベンチ (小ベンチ / フルベンチ) は lm_head も 4bit のパック `~/models/ddalcu-mlxlm-head4` で取る (相手の一律 4bit と条件を揃える。記録に注記)。本番の既定は 8bit のまま。

訂正 (2026-09-03 19:00、ユーザー): **本番も lm_head 4bit** (`~/models/ddalcu-mlxlm-head4`、bf16 から焼いたもの)。KLD の代金 +0.0047 は了承済み。以後の KLD 基準は 0.01794。HF の公開パックは差し替え待ち (BACKLOG)。

---

## measurement-discipline (feedback)

このマシンでの計測は 1 プロセス内で交互に測る。外部実装の数字で自分の実測を上書きしない


2026-08-27〜28 に同じ種類の失敗を繰り返したので規律として残す。

**1. 別々の起動で測った数字を比べない。**98GB のモデルを 2 セッションが同時に
読むとメモリ圧で片方が落ちるか、遅くなる。実際に n-gram の A/B が 90.6ms 対
103.7ms (期待値 50ms) と倍近く外れ、「PLE を切ると速くなる」まで出た。
1 プロセス内で切り替えて交互に測ること。融合カーネルは on/off できるので
交互測定が可能で、そうしたら 41.38 と 42.14 で一致した。

**Why:** 熱・電力状態・他セッションのメモリ圧が、測りたい差より大きく動く。

**How to apply:** `tools/rebit_ab.py` と `tools/ngram_backend_ab.py` がその形。
新しい A/B を書くときは 1 ロード内で完結させる。`tools/biglock.sh` を必ず通す。

**2. マイクロベンチの絶対値を予測に使わない。**lm_head は単体 1.87ms
(256 GB/s) だがモデル内では 4.89ms (104 GB/s)。`micro_moe_gdn.py` が出す
「限界コスト 385 GB/s」も同じ理由で楽観側。**案の優劣を比べる用途に限る。**

**3. 外部実装の数字は仮説として扱い、自分の実測を上書きさせない。**
`ddalcu/mlx-serve` から 3 つ持ち込んで 1 勝 2 敗。当たったのは n-gram の
並列 pread だけ (-2.1ms)。外した 2 つ (gather_qmm が理想の 3 倍 / lm_head は
巨大 N のタイル問題) は、**こちらの実測に基づく解釈をあちらの環境の数字で
置き換えた**もの。lm_head は「差の 3ms は他の部品と帯域を奪い合うぶん」と
最初に正しく書いていたのに、後から読んだコメントで上書きした。

**How to apply:** 持ち込むときは「未検証の仮説」と明示し、着手順を変えろとまでは
書かない。カーネルセッションは両方その場で測って否定した。

**4. 判定基準は数字が出る前に宣言する。**v-96 の採否は「ms/KLD が 434 以上なら
採用圏、100 台なら却下」を測る前に置いた。結果 122 で却下。後から基準を決めると
都合よく読める。

**5. 遅延評価のタイマー位置。**MLX は遅延評価なので、プロンプトの forward を
eval せずにタイマーを開始すると次の eval が巻き込む。これで「S=16 でも S=1 の
1.17 倍」という誤った数字を作り、ディスパッチ律速の根拠と MTP の見込みを
半日引用し続けた。正しくは 2.53 倍。**絶対値が辻褄に合わない時点で疑うこと**
(S=1 が 490ms、逐次デコードは 50ms/token だった)。

## 正式値を更新するときの手順

(2026-08-31 に、畳んだカーネル分業のメモから引き取った。分業自体は
終わったが、この手順は残る)

**静音窓 + 3 反復の中央値 + 同時に GPU プロセスは 1 つ。**サーバーを立てたまま
測らない。今日 90GB のダウンロードと並行して KLD を回して、n-gram の行取りが
ディスクを取り合って数時間かかる見込みになり、止めて測り直した。

## 2026-08-30 の教訓: 測る前に説明を組み立てない

この日、仮説を 5 つ立てて **5 つとも実測で外した。**

| 仮説 | 実測 |
|---|---|
| mlx-serve は Flash-Next で投機していない | していた (サーバー経路で。offline 経路が旧型で使えないだけ) |
| priming が無いから受理率が低い (mlx-serve) | 65-77% で設計どおり |
| 低ビットで小さくすれば速い | per-token の読み出しは 9% しか減らず、受理率が落ちて帳消し |
| priming が無いから受理率が低い (ds4) | 68% あった。問題は検証が償却しないこと |
| ds4 の投機に穴がある | **文献どおりだった** (arXiv 2506.20675 / 2607.12696) |

共通しているのは、**どれも「理屈のうえで空いて見える穴」**だったこと。
実際に空いていたのは 1 つも無く、本当のボトルネックは毎回別の場所にあった。

効いた道具は 3 つ。**相手のコードを読む** (「spec wiring pending」が古いログだと
気づけた)、**相手の計装を使う** (`DS4_MTP_TIMING`、`--expert-profile`、
`[spec-stats]` — 作った人が既に測れるようにしていた)、**掃引する**
(1 点でなく幅 1/2/3/4/6 を並べて初めて「線形」と「頭打ち」が見えた)。

## 2026-08-31 の追補 (mlx-serve 追撃戦で踏んだ罠 4 つ)

6. **並行ダウンロードは計測を汚す。**bf16 再取得と重なった速度測定が 21.4 tok/s
   (実力 42.8) を出した。サーバー内部の decode= とクライアント実測を両方出して
   おくと、この種の汚染は乖離としてすぐ見える (それでログに tok/step と ttft を
   常設した)。

7. **積み上げ無効化 (ablate.py) は部品コストを過大評価する。**部品を外すと下流の
   意味が壊れ、hidden が退化してアクセスパターンまで変わる。HC 7.6ms 説はこれが
   作った偽値で、実入力を捕まえて単体で測ると 3.8ms (tools/module_costs.py)。

8. **温キャッシュのマイクロは「M 比例の読み直し」を誇張する。**fast_qmm が
   マイクロで M=3 に 22-35% 勝つのに実モデルでは verify +2ms。stock qmv の
   2 回目以降の読みは実機では SLC に当たる。in-model A/B だけが判定に使える。

9. **単発 128 トークンの tok/step は ±0.2 の「テキスト運」で動く。**ulp が 1 個
   flip すると greedy の続きが分岐して、受理率が系統的に落ちたように見える
   (2.44→2.23 を 3 回、別々の変更で観測)。複数プロンプト x 512 の平均では消えた。
   数値を動かす最適化を受理率で棄却する前に、必ず平均で判定する。

## 2026-09-02 に足りないと分かったもの

作法は守っていたのに間違えた。**足りなかったのは 2 つ。**

**1. 率を時間と取り違えない。**伸びしろを 5 回過大に見積もり、根拠は毎回「率」
だった。効率の率、水増しの率、部品時間の率。**転嫁率が 0.30 なら 60% の無駄は
18% の時間。**「率が悪い」と「時間がある」は別。

**2. 分母が何を仮定しているかを確かめる。**「効率 47.6%」は下限が qmm の FLOP
しか数えていなかった。「天井の 96%」は一様分布のマイクロで、実分布ではタイルが
1.4 倍余る。「効率 68-76%」は逐次スキャンが到達できない速度を分母にしていた。
**効率が高く見えると、そこを見なくなる。**アルゴリズムの選択ミスは効率の数字に
出ない。

**3. 道具自体を対照で検査する。**A/B が長文脈で A 側に +5.6% の下駄を履かせて
いた (回文順は線形ドリフトしか相殺できず、位置 1 の段差が残る)。
**何もしない null knob を作って測れば分かる。**同じ形で、機構だけの対照 C
(融合せず差し替えだけ) も要る。

**4. 単発値を「謎」と名付けない。**reps 3 で取った +757ms を「未解明の 2.9-6.1s」
として計画に書いたが、雑音の床が 96ms、run 間ドリフトが 131ms だった。再現しない。

追記 (2026-09-03): カーネルの連鎖 micro は重みを層数ぶん (100 MB 超) 巡回させて冷やす。重み 1 組を読み回す温の連鎖では、並列度の足りない自前カーネルが
冷の DRAM レイテンシを隠せない負けが見えない (HC 融合: 温 +13 us → 冷 +78 us/回、in-model の +8 ms の全量)。温で速くても採用しない。

---

## mirai-uzu-competitive-position (reference)

Mirai Labs / uzu (Rust+Metal、独自 4bit 蒸留量子化 Mirai-M、<50 MB の学習 draft で specdec)。Qwen3.6 27B を M5 Max で 105 tok/s。同機体の MTPLX 55 / MLX 26


https://trymirai.com/local-models/alibaba-qwen3-6-27b-mirai-mirai-m-4 (2026-09-01 ベンチ)。分析は `docs/research/EXTERNAL-MIRAI-UZU-2026-09.md` (2026-09-04)。
速さの正体は機体 (M5 Max) + 学習した draft (specdec 3〜4 トークン/forward、チャットでは 84) + 蒸留した独自 4bit (YAQA + QAD + Hadamard)。同じ重みではない。
うちの方針 (draft を学習しない、bf16 比 KLD で品質を守る) のままでは decode は MTPLX 級 (MTP 1 層で 2 倍) が上限。方針を変えるなら draft の学習が一番効く (ユーザー判断)。
関連: [[mtplx-competitive-position]], [[mlx-serve-competitive-position]], [[local-model-verdict]]

---

## mlx-serve-competitive-position (project)

mlx-serve が本命の競合。同一機で decode 1.33-1.48 倍の差。差別化の余地は 2026-08-30 の実測で消えた


[mlx-serve](https://github.com/ddalcu/mlx-serve) (David Dalcu、Zig、star 910、
commit 378) が mlxturbo の本命の競合。2026-08-30 に発見。oMLX や MTPLX
([[mtplx-competitive-position]]) より直接ぶつかる。

**同一機 (M3 Max 128GB)・同一 31 課題で実測した (docs/VS-MLX-SERVE.md)。**

| | mlx-serve | mlxturbo |
|---|---|---|
| decode 中央値 | 64.0 tok/s | 43.3 tok/s |
| decode (48k) | 43.3 | 30.2 |
| cold prefill (48k) | 55.6s | 99.5s |
| 温まった TTFT | 0.2 秒 (文脈長によらず) | 2.6-6.6 秒 |

重みはそれぞれの推奨構成 (ddalcu 68GB + n-gram 30GB mmap で常駐 70GB /
v-l 77GB)。**向こうの方が小さい。**

**相手が持っていてこちらに無いもの**: Ollama API、GUI (公証済み .app)、
GGUF (llama.cpp 埋め込み)、vision、`--ane-prefill` (各 prefill チャンクの
dense MLP 行の 4 割を ANE へ)、`--mtp-depth` 適応的に最大 6-8 (こちらは 1)、
`--mtp-history-window` (こちらの priming と同じもの)、`--kv-quant turbo2/4`、
prefill チャンク 4096 (こちらは 2048)、`mlx-serve launch <agent>`。

**プロトコルは 4 つ同一ポート** (OpenAI / Anthropic / Responses / Ollama)。
「3 プロトコルはほぼ唯一」という前提は誤りだった。

**残る差別化は 2 つだけ。**

1. **分布保証** — Block Verification で厳密同一分布。相手は
   `--decode-attn-quant` を**既定 ON** にして "LOSSY: a real requantization"、
   `--ane-prefill` も "lossy" と自分で書いている
2. **品質の実測** — レシピごとの KLD。相手は速度しか出していない

速度と品質を必ず対で出すこと。速度単独では追いつく頃に相手が先に行っている
(Flash-Next 対応を 2026-08-29 に入れてきた)。

**読み違いの記録**: 競合を 1 つ (oMLX) 見つけた時点で探索をやめ、
POSITIONING.md を書いた。その数時間後に mlx-serve が見つかり、書いた根拠が
3 つとも崩れた。**反転条件を書くだけでは足りず、既に満たしている実装が
無いかを調べる必要がある。**

計測の作法は [[measurement-discipline]]。サーバーの実装は [[fastmlx-server]]。

## 2026-08-30 追記: 差別化の余地は消えた

同じ重み (2/4bit 混在パック) を両エンジンに載せて測ると、**mlx-serve が投機を
切った状態で 52.1 tok/s、mlxturbo が MTP 投機込みで 39.3 tok/s。**カーネルの差が
投機の分を超えている。

「速度と品質の両方を計測で示せる唯一の実装」という言い方も成り立たない。
MTPLX は `qa` に exactness と distribution のゲートを持ち、AIME のベンチランナーも
積んでいる ([[mtplx-competitive-position]])。

結論は [[local-model-verdict]]。

---

## mtplx-competitive-position (project)

MTPLX は 2026-08-29 に Flash-Next + native MTP を入れた。star 1,790 で日 27 のペース。品質ゲートも持っている


MTPLX = youssofal/MTPLX (Apache-2.0、GUI 一体型の MLX 投機デコードアプリ)。2026-08-26 時点で Star 1,682、作成 4 ヶ月、ほぼ毎日リリース、作者ワンマン + コミュニティ PR が入り始めた急伸期。

fastmlx の方針 (ユーザー明言): 人気獲得は目指さない。バックエンドに徹して GUI は LM Studio に任せる。勝負軸は「同一ハードで速度と品質 (KLD) の両方を計測で示せる唯一の実装」。MTPLX の弱点 (長文脈 prefill の崩れ、quant 品質の評判、サーマル持続) は Issue #286/#293 等に証拠あり。詳細な実測比較は docs/KERNEL-INTEL.md にある。

注意: arcee-ai/fastmlx (Star 367、2025-03 で停止) は同名の別物。ユーザーのプロジェクトとは無関係。

関連: [[lmstudio-integration-plan]]

## 2026-08-30 の観測

- star **1,790** (4 日前 1,682、日 27 のペース)。2.10.0 を 2026-08-29 に出し、
  コミットは毎日
- **Flash-Next 対応を 2026-08-29 に入れた** (#380 `feat: support
  Qwen3.8-Flash-Next (qwen4_exp) with native MTP and compiled decode`)。
  #391 で最適化を継続中 (16K/1K で 77.77 tok/s と報告)
- 転んでいる所もある: #393 (262K cold prefill が 119GB まで行って無応答か 507)、
  #400 (M2 Max 96GB で turbo カーネルが crash)
- **`qa` に exactness と distribution のゲートを持ち、AIME のベンチランナーも
  積んでいる。**「速度と品質の両方を計測で示せる唯一の実装」という以前の
  言い方は成り立たない ([[mlx-serve-competitive-position]])

うちのパックを MTPLX に読ませることはできた (`ngram-table.safetensors` を用意し、
config に `ngram_sidecar: true`、量子化エントリに `language_model.` 前置きの
複製を足す)。ただし**最適化ルートが「未検証パック」では切られる**ので、
出た 17.5 tok/s は MTPLX の速度ではない。公平に測るには向こうの
`Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed` (107.2 GiB) が要る。

追記 (2026-09-03): rapid-mlx 0.13.x (raullenchai/Rapid-MLX) も Flash-Next のネイティブ MTP (decode 34.85 tok/s、8K prefill 9.24 s、M3 Ultra の見込み) と 27B の MTP + 継続バッチを出した。同一機体の実測は無いが、M3 Max のうち (52 tok/s、8k 11.2 s) の方が機体差込みで速い。27B レーンの比較表に rapid-mlx を入れる。詳細は docs/research/EXTERNAL-PERPLEXITY-LILY-2026-09.md の「外部の知見 2」。

---

## nax-target-machine (project)

本番は NAX 対応機 (M5 系) でも使う。この機 (M3 Max) は非 NAX で、自前カーネルの既定は NAX で off にし、NAX 機で A/B を取り直す


fastmlx / mlxturbo は NAX 対応機 (M5 系) でも動かす (ユーザー方針 2026-09-03)。開発機の M3 Max は非 NAX。

**Why:** MLX は NAX 機で別のカーネル (`gather_qmm_rhs_nax` 等) を使うので、この機で取った dense 比や
タイル形の数字 (BM=16 の straddle、MoE の dense 比 1.19) は NAX 機では成り立たない。ユーザーは
「NAX がある方は NAX に倒す」方針。

**How to apply:** 自前 Metal カーネルは NAX 専用 intrinsic を使わず simdgroup_matrix の経路だけで書き、
発火は `MLXTURBO_MOE_GEMM=auto|on|off` (auto = 非 NAX で on、NAX 機で off)。機種判定は 1 箇所
(`mlxturbo/kernels/moe_grouped_gemm.py` の `is_nax_device`)。NAX 機に着いたら MoE の dense 比と、
既定 on の自前カーネル (gather prefill attention、GDN Metal) の A/B を knob で取り直す。
関連: [[fastmlx-lanes-2026-09]], [[fastmlx-remaining-work]]

---

## opus-subagents-for-work (feedback)

親が Fable のセッションでは作業(実装・検証)も Opus サブエージェントに積極的に委譲する


2026-08-28、gemma4-lora 作業中にユーザーが明示: 「作業用の Opus サブエージェントは積極的に使ってください」。

**Why:** CLAUDE.md の分業表(親=Opus が実装)は親が Opus である前提。親が Fable のときは、実装・検証のような長い手作業を親で抱えず Opus に出す方がコストと並行性の両面で良い。

**How to apply:** general-purpose を `model: opus` で起動し、実装+スモーク実行まで一括で任せる。読むだけの調査は従来どおり scout(Sonnet)。相談役は opus-advisor(Fable 親のとき fable-advisor は視点が増えないため)。

関連: [[gemma4-lora-lane]]

---

## product-direction-2026-09 (project)

製品の方向 (2026-09-04 決定): 中に入るエンジン (verified fast path)。P0 = ExecutionPlan/explain/strict plan → Engine/Session API + mlx-lm アダプター → 配布 → 常設ベンチ + hit rate → README。着手は 27B decode の後


決定は `docs/research/PRODUCT-DIRECTION-2026-09.md` (GPT-5.6 Pro の意見を踏まえて親が採否、ユーザー 2026-09-04 11:39 承認)。要点: oMLX と製品の座を争わず、実行経路を保証するエンジン。速度の数値ではなく経路と意味論 (MTP の有無、降格、分布の同一性) を公開契約に。TurboPack 契約・Session fork API・統合スケジューラは P1。

**Why:** 同じ土台 (mlx-lm) の製品と看板で競っても勝ち筋が無い。「皆が使う推論エンジン」が目標 ([[engine-goal-everyone-uses]])。

**How to apply:** 27B の decode (小 M qmm + dispatch 修正) が落ち着いたら P0 に移る。Flash-Next の 1〜2% の的はそこで止める。新機能は「製品が入れ替えたくなるか」で判断。
関連: [[engine-goal-everyone-uses]], [[fastmlx-server]], [[lmstudio-integration-plan]]

---

## qsa-tail-semantics (project)

Flash-Next の QSA の tail は HF ではクエリごと (自分自身を含む)。fastmlx の vendor は global tail で S>1 の行が自分を見ていなかった (2026-09-03 に発見、修正中)


HF transformers main (外部リポジトリ、このリポジトリには無い) の qwen4_exp モデリングファイル `Qwen4ExpTextQSAIndexer.forward` の規則:
クエリ q の可視 = 完全ブロック (block_end ≤ q) から top-k ∪ tail = 位置 4·floor((q+1)/4) .. q (クエリごと、自分自身を含む)。
top-k は torch.topk (同点は小さい添字)。dense ゲートは無い。

fastmlx の `mlxturbo/_vendor/qwen4_exp.py` (`_pooled_and_top` 509〜522 行、`_gather_tile_attn`)、`batch.py`、`prefill_attn.py` は
**global tail (kv 末尾の端数列) ∧ causal** で、S=1 の decode だけ一致。prefill の 3/4 の行と verify の非最終行が自分自身と直前 0〜3 トークンを見ていない。

**Why:** 品質の物差し (bf16 teacher) が同じ vendor コードなので KLD に映らない差。mlx-serve は HF と比較して per-query に合わせ済み。
verify 行の logits にも効くので受理率にも関わる。

**How to apply:** `MLXTURBO_QSA_TAIL=query|global` の knob で修正中 (2026-09-03)。既定を query にしたら teacher (bf16) を作り直して KLD を取り直す。
新しいカーネル (K2 等) は per-query tail で書く。関連: [[fastmlx-lanes-2026-09]], [[fastmlx-measurement-traps-0902]]

---

## qwen36-mac-caption (project)

Qwen3.6-35B-A3B を Mac で画像キャプションに使う際の速度決着と、fastmlx が効かない理由


2026-08-28、`/Users/ht/dev/qwen36-capbench` で決着 (リポジトリはその後削除済み、数値は本メモリが一次記録)。B200 の vLLM+fp8 (24.5 枚/秒) を Mac の MLX で置き換えられるかの計測。

**結論: Mac は 0.88 枚/秒(4bit、B=32、guided あり)で B200 の約 28 分の 1。** 高速化の余地はバッチ化で取り切っており、カーネルの残りは無い。

- **バッチ化が唯一の効いた手**(0.257 → 0.880 枚/秒、3.5 倍)。mlx-vlm 0.6.17 に `batch_generate` が既にある。3.5 倍止まりなのは prefill が縮まないため(1 枚 446ms、バッチ時 40%)。
- **画像サイズを揃えないとバッチ化が丸ごと無効**。`group_by_shape` がサイズごとに分裂させる。wild の 16 枚は 15 通りのサイズで、B=1/16 が 103.8/109.6 tok/s と平ら。一律リサイズは最適化でなく前提条件。
- **fastmlx のカーネルは効かない(1.13 倍、宣言バー 1.2 未達)**。理由は動作点で、256 experts に 8 routed だと 1 expert あたり B/32 行しかなく、B=32 で 1.45 行。fastmlx の較正は M=6..16(投機検証幅)なので起動すらしない。加えて Qwen3.6 に hyper-connections は無く、20GB 常駐なので [[flash-next-speed-verdict]] のメモリ圧効果も再現しない。
- `gather_qmm` は全バッチサイズで約 1.5 TFLOPS(ピークの 11%)で平ら。行数が増えないので効率が上がらず、MoE 行列積だけで 750 tok/s の頭打ちを作る。
- OptiQ-4bit は素の 4bit に速度優位なし(0.731 対 0.751 枚/秒)、2.6GB 大きい。
- **B=1 と B=8 は greedy でもビット一致しない**(バッチで行列積の形が変わり丸め差が argmax を反転)。実害は小さい(客観フィールドは 100%、quality 93.3%)が、品質評価は出荷するバッチサイズでやること。

hidreamft 側にも効く発見: `caption_vllm.py` は `enable_prefix_caching=True` を立てているが**実質まったく効いていない**。`to_conversation` が system メッセージを使わず user ターンの先頭に画像を置くので、画像が 1 枚ごとに違う以上、共通接頭辞は先頭 3 トークンで終わる。直すには (1) 指示文を画像より前に出す、(2) さらに `third_lang` を末尾へ動かす(途中にあると共有が 408 中 221 で切れる。末尾なら 422、prefill 2.55 倍)。

なお 0.880 枚/秒 は **guided decoding なし**の実測。guided ありは B=32 で未測定で、B=16 の 11% を外挿して 0.82-0.83 の見込み。

申し送り先の HANDOFF-HIDREAMFT.md はリポジトリごと削除済み([[handoff-in-docs]])。

---

## turboquant-decided (project)

TurboQuant (KV cache 3 bit、回転 + Lloyd-Max + QJL) の実装は確定 (ユーザー 2026-09-04)。Gemma レーンの中で、drafter → norm → KV 量子化 第 1 段 (mlx-lm の QuantizedKVCache) → TurboQuant。計画は docs/research/TURBOQUANT-PLAN.md。Codex で実装する可能性が高い


ユーザー (2026-09-04 11:48): 「TurboQuant 自身は実装するのを確定で。全部記録して。そのうち多分 Codex で実装しそう。ある程度方向が決まった」。

**Why:** Gemma (attention 主体、100k 級の文脈と多セッション) を扱いたい。KV の容量と帯域が効く場面で、学習不要で「同じ重み + KLD で品質を測る」の枠に収まる。

**How to apply:** 計画は `docs/research/TURBOQUANT-PLAN.md` を正とする (着手者が読めば足りる形で保つ)。取り分は読み側の decode attention カーネル (packed 3 bit、S ≤ 8、qmv 型) に懸かる。第 1 段 (mlx-lm の QuantizedKVCache の実測) を先に。Flash-Next / 27B は KV が小さいので後。
関連: [[product-direction-2026-09]], [[engine-goal-everyone-uses]]

---

## vs-mlx-serve-clean-baseline (project)

2026-09-02 に条件を揃えて測り直した mlx-serve との差と、前セッションの数字が歪んでいた理由


前セッション (2026-09-02 午前) の「prefill 2 倍差」はハーネスの産物だった。`bench/self_snapshot.py` が
文脈ごとに池の先頭から窓を切っていて、rep 2 と長文脈が両エンジンの接頭辞キャッシュに当たっていた。
加えて mlx-serve は thinking off、mlxturbo は on で走っていた。直して測り直した現在地 (thinking off、
M3 Max、2026-09-02 11:09):

| 文脈 | 冷 TTFT serve / turbo | 温 TTFT serve / turbo | decode serve / turbo |
|---|---|---|---|
| 4k | 5.77 / 8.28 (1.43x) | 0.87 / 0.46 | 55.3 / 48.8 |
| 17k | 29.2 / 37.6 (1.29x) | 0.92 / 0.51 | 56.0 / 41.0 |
| 50k | 108 / 163 (1.51x) | 22.8 / 1.33 | 30.8 / 34.4 |

decode の差は forward の GPU 時間そのもの (S=1 短文脈: うち eval 24.0 ms、相手 18.2 ms)。
Python の構築 4-5 ms は段階投入で隠れている。17k では QSA の decode 経路がさらに +5 ms。

**Why:** 数字の出所を確かめずに 2 倍差を追うと、存在しない差を埋めようとして時間を失う。

**How to apply:** 対戦の数字は必ず両エンジンのログ (`bench/results/logs/`) で cache reuse と
thinking を確認してから読む。詳細は `docs/research/SESSION-2026-09-02-CATCHUP.md`。
関連: [[measurement-discipline]], [[mlx-serve-competitive-position]]

追記 (2026-09-03 01:40、今日の既定変更の合算、mlxturbo だけ冷えた機体で反復なし): 冷 TTFT 4k 7.18 / 17k 32.5 /
50k 119 s、decode 50.6 / 48.7 / 42.4 tok/s。相手の朝の値と比べて prefill 1.24 / 1.11 / 1.10x 負け、decode 4k -8% /
17k -13%、50k +38%。lm_head 8-bit のままの数字。

追記 (2026-09-03 06:50、冷却強化後、detokenizer 修正 + depth 適応 on、mlxturbo だけ反復なし): 冷 TTFT 4k 6.89 /
17k 31.7 / 50k 117 s、温 TTFT 0.13〜0.31 s (相手 0.87〜0.92 の 4〜6 倍速)、decode 51.4 / 46.8 / 44.0 tok/s。
対相手 prefill 1.19 / 1.09 / 1.09x 負け、decode 4k -7% / 17k -16% / 50k +43%。

追記 (2026-09-03 09:10、gather カーネル既定 on、既知を片付けた基準): 冷 TTFT 4k 6.89 / 17k 31.0 / 50k 94.2 s
(相手比 1.19x / 1.06x 負け / 0.87x 勝ち)、温 TTFT 0.13〜0.31 s、decode 51.5 / 47.5 / 43.4 tok/s (相手比 -7% / -15% / +41%)。

**フルベンチ (2026-09-03 12:20、同日・同冷却、反復 1)**: 冷 prefill serve/turbo 4k 5.70/6.88、17k 27.8/31.6、50k 82.1/93.1
(1.13〜1.21x 負け)。decode 55.7/51.4、49.0/45.3、45.9/46.9 (±8%)。温 TTFT 0.85〜0.91 対 0.15〜0.25 (50k は 15.9 対 0.93)。
朝の「50k で大勝ち」は相手の熱だった。相手も冷却強化で 50k prefill 108 → 82 s、decode 30.8 → 45.9。

追記 (2026-09-03 17:30、小物をかき集めた既定): 冷 prefill 4k 6.60 / 17k 31.3 / 25k 45.5 / 32k 56.4 / 50k 89.9 s
(相手比 1.16 / 1.13 / 1.10 / 1.09 / 1.09x)、decode 51.9 / 46.1 / 44.2 / 42.7 / 42.7 (相手比 -7〜-10%)、温 TTFT 0.14〜0.90 s。
次に入れるのは PLE の mmap 化と attention qkv の連結 (prefill 幅)。

追記 (2026-09-03 19:19、小ベンチ 0903g、今日の既定全部 + lm_head 4bit、排他冷却 10 分): 冷 prefill 4k 5.95 / 17k 30.3 / 25k 43.5 / 32k 54.3 / 50k 86.7 s (相手比 1.04 / 1.09 / 1.05 / 1.05 / 1.06x)、decode 51.7 / 47.5 / 47.5 / 44.8 / 47.0 (相手比 -7 / -3 / -22 / -5 / +2%)、温 TTFT 0.14〜0.80 s。サーバー経路の 17k が decode_ab (26.4) より 15% 遅いのが次の的。

追記 (2026-09-04 03:41、小ベンチ 0903h、今日の既定 3 つ込み): 冷 prefill 4k 5.74 (相手 5.70、同着) / 17k 26.5 (27.8、0.95x) / 25k 40.8 (41.3) / 32k 50.3 (51.6) / 50k 80.1 (82.1) で **17k 以上は勝ちに転じた** (n-gram 先読みと HC qmm_wide、0903g の 17k は最初の長い prefill の段差込みだった)。decode は 1 回 × 256 で ±10% の運 (4k 58.5 / 17k 47.4 / 25k 52.5 / 32k 49.9 / 50k 44.8)。相手の数字は 9/3 12:20 のもので同時刻ではない。決着はフルベンチ (反復 2)。

---

## zero-cost-wins-adopt (feedback)

代金ゼロの改善は取り分が小さくても既定に入れる (ユーザー方針 2026-09-03)。効果が薄いだけを理由に畳まない


代金ゼロ (品質・メモリ・複雑さの増分が無く、測った文脈のどれでも遅くならない) の改善は、取り分が 1% 未満でも既定に入れる。

**Why:** 2026-09-03 に MIN_KV 8192 (-0.8%)、K2c (-0.7%、ビット一致)、GDN のレジスタ常駐 scan (冷 micro 0.87 倍) を「判定線に届かない」で畳みかけたのをユーザーが止めた。小さい代金ゼロの改善を積むのがこのプロジェクトの取り分の出方。

**How to apply:** 判定線は「代金があるものを弾く」ために使う。代金ゼロなら「遅くなる文脈が無い」だけを確かめて入れる。畳むのは、品質・メモリ・遅くなる文脈・写しの増加のどれかがあるとき。関連: [[measurement-discipline]], [[fastmlx-lanes-2026-09]]

