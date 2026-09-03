# mlxturbo で作業するときの決まり

対 mlx-serve の性能レーンが主戦場。現在地と次の一手は
`docs/research/KERNEL-BRIEF-DECODE-BW.md` の末尾が常に最新。着手前に必ず読む。

## 計測の作法 (破ると数字が嘘になる。全部実測で確立済み)

- A/B は 1 プロセス内で交互に測る。プロセスを分けた比較は熱・キャッシュ状態で数 % ずれる。
- tok/step (受理率) の比較は複数プロンプト x 512 トークンの平均でだけ行う。
  単一プロンプトは chunk 境界の丸め 1 つで挙動が変わる (テキスト運)。
- 温キャッシュのマイクロベンチの絶対値を信じない。案の優劣の目安にだけ使い、
  採否は in-model A/B で決める (micro 勝ち in-model 負けの前例が複数ある)。
  **カーネルの連鎖 micro は重みを 100 MB 超 (層数ぶん) 巡回させて冷やすこと。**重み 1 組を 200 回読む温の連鎖では、
  並列度の足りない自前カーネルが DRAM レイテンシを隠せない負けが見えない (HC 融合: 温 +13 us、冷 +78 us/回、2026-09-03 に帰属)。
- 無効化の積み上げで部品時間を見積もらない (ablate の積算は過大評価の前例あり)。
  部品計測をしたら「部品和 ≈ 壁時計 (数 % 以内)」を必ず確認する。
- GPU の仕事は全部 `tools/biglock.sh` で直列にする。モデルを読まない数分の micro は A/B の間に割り込む
  (biglock がコマンド名 `_micro.py` / `verify_*.py` / `smoke_*.py` / 連鎖ツールで自動判定、`BIGLOCK_PRIO=1/0` で明示)。
  20 分の A/B の後ろで micro を 40 分待たせるとエージェントがその間止まる。
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
- `mlxturbo/staged.py` の `staged_forward` は site-packages の
  `mlx_lm.models.qwen3_5.Qwen3_5TextModel.__call__` の写し (27B/qwen3_5 側)。
  **本家 (mlx_lm 更新時) を変えたらここも変える。**qwen4_exp 側の 3 つと違い
  専用のビット一致ゲートは無いので、変更後は出力トークン列の一致で確認する。
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
  `MLXTURBO_DRAFT_RERANK=1` / `MLXTURBO_HC=kernel` / `MLXTURBO_SORT_MIN=16`
  は本番の既定値そのもので、値を変えると本番の挙動が変わる。
  `MLXTURBO_GDN_METAL` (既定 on、`=0` で off) も同様に本番の既定値で、
  2026-09-02 の in-model A/B (17k prefill -1.3〜-4.5%、KLD +0.00014) で
  on にした。
  `MLXTURBO_DEPTH_ADAPT` (既定 on、`=0` で off) も本番の既定値で、2048 超の文脈で受理率 EMA から
  draft 深さを選ぶ (17k で ms/tok -3〜-6%、2026-09-03)。2048 以下は静的 depth 2 のまま。
  `MLXTURBO_PREFILL_ATTN` (既定 on、`=0` で off) は T=1 gather の prefill attention カーネルで、kv ≥ 12288
  (`MLXTURBO_PREFILL_ATTN_MIN_KV`) だけ発火する (50k prefill -21%、2026-09-03)。長文脈の KLD (自前の dense
  比) 0.040 は、既に受理している GDN Metal の同じ物差しでの 0.111 より小さい。長文脈の品質は
  bf16 参照が無いので、この物差しではなく課題の正答率で見ること (CATCHUP 2026-09-03 07:40)。
  `MLXTURBO_MOE_COMBINE_FOLD` (既定 on、行数 ≥ 64 の prefill 幅だけ、`=0` で off) と `MLXTURBO_PRIME_WINDOW`
  (既定 512) も 2026-09-03 に A/B と KLD で入れた本番の既定値。
  `FASTMLX_NGRAM_BACKEND` (既定 mmap、`=pread` で旧経路) と `MLXTURBO_NGRAM_PREFETCH` (mmap では既定 on、`=0` で off) も
  本番の既定値で、2026-09-03 の別プロセス比較 (8k -6%、17k -7%) で切り替えた。n-gram の行取得を mmap + 背景 madvise で行う。
  `MLXTURBO_SDPA_ROWTILE` (既定 256、`=0` で off) も本番の既定値: head_dim 256 の sdpa は MLX の fallback でタイルを飛ばさないので、
  prefill の dense 経路で q を 256 行ずつに割り前方の K/V だけ渡す (4k -1.1% / 8k -1.2% / 17k -1.0%、KLD 差 0.0、2026-09-03)。
- 品質を売って速度を買わない。fake を実物より緩くしない。KLD の受け入れ幅は
  現行比 +0.0005 (bench/quant_eval.py compare)。

## 分業

実装・検証はサブエージェントに出してよいが、計測の判定と commit は親が行う。
HF トークンは huggingface_hub の既定の場所のみ。引数・ログ・環境変数に出さない。
