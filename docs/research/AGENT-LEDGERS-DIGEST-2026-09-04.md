# Agent 台帳ダイジェスト (2026-09-04)

対象 27 台帳 (`scratchpad/agent-*.md`)、価値のある差分があったのは 22 件。

Claude Code → Codex への引き継ぎ (`docs/HANDOFF-2026-09-04.md`) にあたり、
untracked の作業台帳が消える前提で、CATCHUP / HANDOFF / BACKLOG / COMPARE-QUEUE に
まだ写っていない知見だけを抜き出した。数字付きの結論のうち docs に既出のものは
再掲していない。走行中のまま終わった 3 台帳 (`agent-27b-sdpa-split.md`、
`agent-fn-moe-compile.md`、`agent-gemma4-smoke.md`) は時刻を明記した途中経過。

## 目次

1. [agent-27b-ab-tool.md](#agent-27b-ab-toolmd-最終更新-0854)
2. [agent-27b-anatomy.md](#agent-27b-anatomymd-最終更新-0920)
3. [agent-27b-attn-mlp-port.md](#agent-27b-attn-mlp-portmd-最終更新-0856)
4. [agent-27b-bringup.md](#agent-27b-bringupmd-最終更新-0831)
5. [agent-27b-decode-b1.md](#agent-27b-decode-b1md-最終更新-1004)
6. [agent-27b-decode-b2.md](#agent-27b-decode-b2md-最終更新-1044)
7. [agent-27b-dispatch-fix.md](#agent-27b-dispatch-fixmd-最終更新-1216)
8. [agent-27b-gdn-port.md](#agent-27b-gdn-portmd-最終更新-0842)
9. [agent-27b-mtp.md](#agent-27b-mtpmd-最終更新-0801)
10. [agent-27b-sdpa-split.md (走行中)](#agent-27b-sdpa-splitmd-最終更新-1313-走行中)
11. [agent-27b-verify-width.md](#agent-27b-verify-widthmd-最終更新-1259)
12. [agent-ceiling-audit.md](#agent-ceiling-auditmd-最終更新-1207)
13. [agent-flash-attn.md](#agent-flash-attnmd-最終更新-0407)
14. [agent-fn-compile-poc.md](#agent-fn-compile-pocmd-最終更新-1227)
15. [agent-fn-moe-compile.md (走行中)](#agent-fn-moe-compilemd-最終更新-1312-走行中)
16. [agent-fn-skip-stack.md](#agent-fn-skip-stackmd-最終更新-1141)
17. [agent-fusions-family.md](#agent-fusions-familymd-最終更新-0445)
18. [agent-gemma4-smoke.md (走行中)](#agent-gemma4-smokemd-最終更新-1306-走行中)
19. [agent-hc-prefill.md](#agent-hc-prefillmd-最終更新-0311)
20. [agent-mlx-upgrade.md](#agent-mlx-upgrademd-最終更新-1214)
21. [agent-moe-counting-sort.md](#agent-moe-counting-sortmd-最終更新-0449)
22. [agent-moe-fused1.md](#agent-moe-fused1md-最終更新-0252)
23. [agent-ngram-prefetch.md](#agent-ngram-prefetchmd-最終更新-0223)
24. [agent-pseudo-fusion-audit.md](#agent-pseudo-fusion-auditmd-最終更新-1018)
25. [agent-qmm-wide-smallm.md](#agent-qmm-wide-smallmmd-最終更新-0359)
26. [agent-smallkv-attn.md](#agent-smallkv-attnmd-最終更新-0144)
27. [agent-worker-fix.md](#agent-worker-fixmd-最終更新-0354)

---

## agent-27b-ab-tool.md (最終更新 08:54)

27B (qwen3_5) 用の A/B 道具 `tools/decode_ab_generic.py` を新規に作った台帳。

- 「実装のメモ」の節: `fused.disable_gdn_port` を呼び忘れると `_GDN_UPDATE_PATCHED` の番人が 2 回目の `enable_gdn_port` を弾き、prefill Metal 再帰だけが**全 variant で無効**になったまま気付かない。同型の罠は `nn.QuantizedLinear.__call__` の `_QMM_WIDE_STOCK` にもある。A/B 道具を書くときは `disable_*` の抜けが「静かに効かなくなる」形で出ることを前提に、両方を戻す `_OrigGuard` のような仕組みを入れること。
- 「残った不明点」の節: 同じ qmm_wide の A/B を 2 回回すと ms/round が -0.6% と +0.0% にぶれる (雑音 ±0.5%)。1% 未満の判定には `--reps` を増やすか文脈を複数にする必要がある。
- `--prefill-once` はケースごとの空焼きが 32 トークンしかなく、位置 1 の段差 (+2〜3%) が丸ごと A 側の下駄になる。1 ケース 1 回文だと熱ドリフトが 3% 級で乗る (表の 4k decode 側 +3.4% がその実例)。

## agent-27b-anatomy.md (最終更新 09:20)

27B decode 1 round の内訳を測る依頼だったが、計測は 1 本も取れないまま終了した台帳。

- 「モデルの形」「Flash-Next との差分」節の表は、`spec.py` (27B) と `spec_flash.py` (Flash-Next) の設計差を項目別に並べたもの (段階投入・同期回数・presync・prime・巻き戻し・depth 適応・draft の生成順・MTP 頭の回数・糊の融合)。27B は同期が round あたり 2〜3 回 (Flash-Next は 1 回)、draft は `cap_base=8` 本を全部引いてから事後 truncate (捨て仕事が出る)、depth 適応は固定コスト定数で実測フィードバックが無い、という差分の一覧は docs のどこにも表になっていない。
- 唯一取れた実測 (道具がクラッシュする直前の 1 ケース、head4 無し・素の 27B): clean 83.59 ms/round、tok/round 2.208、内訳 draft 21.45 ms (25.7%) / verify 61.90 ms (74.1%) / maint 0.23 ms (0.3%)。非投機 (S=1) は 42.40 ms/token (投機時の 1.12 倍)。lookup 無し (MTP のみ) は 83.11 ms/round。
- 「引き継ぎ」節: 新規道具 `tools/decode_round_anatomy_generic.py` は `from mlxturbo import staged` で `ImportError` になり 1 度もゲートを通っていない。原因はこの台帳より後の作業 (2026-09-04 10:20 前後) で `staged.py` が削除され `_hidden_forward` に統合されたため。**この道具は現状壊れている**。再開するなら import 先を直してから使うこと (`bench/results/round-anatomy-27b-0904.json` は生成されていない)。

## agent-27b-attn-mlp-port.md (最終更新 08:56)

sdpa 行タイル (P5) と qmm_wide の MLP (P10) を 27B (qwen3_5) に当てた台帳。

- 「1 プロセス内の A/B」節: 27B への行タイル移植は**ビット一致ではない設計**である (PV の縮約順が変わる)。実機の 2 プロンプトでたまたま分岐しなかっただけで、恒等の証明にはならない。速度と KLD での正式な判定は別途必要 (この台帳では未実施)。
- qmm_wide の 368 射影の内訳: q/o_proj 16×2 + GDN 48×3 + dense MLP 64×3=192 (移植前は 176)。
- 「当たらなかったもの」節: k_proj/v_proj は元から `_QMM_WIDE_TARGETS` の対象外で micro も未取得。MTP サイドカーの層には `enable_qmm_wide(model, mtp=...)` という口があるが、runner が `mtp` を渡していないため未配線のまま。

## agent-27b-bringup.md (最終更新 08:31)

docs/research/COMPARE-QUEUE.md に全部写っている (5 者の起動コマンド、MTP ペア付け、接頭辞付きサイドカーの sha256、煙試験表まで一致)。差分なし。

## agent-27b-decode-b1.md (最終更新 10:04)

27B decode 第 1 段 (段階投入 / capture のモジュール化) の台帳。CATCHUP 2026-09-04 10:09 の節とほぼ同一の内容だが、1 点だけ docs に抜けがある。

- 「残った不明点」1: 融合前処理の q が実機の形 (n_k=16 / key_dim=2048) で検証幅 5/7/8/9 だけ MLX の素と数 ulp ずれる (幅 6 は一致するのが不可解)。CATCHUP はこの箇所に「原因未特定 (BACKLOG)」と書いているが、**`docs/BACKLOG.md` に該当項目が実在しない**。BACKLOG への転記が漏れている。
- 幅ごとの round 費用 (検証幅 1〜10、round 数、受理平均) の生データ (CATCHUP の要約と同じ数字の元表) は `scratchpad/` にしか残らない。数字自体は CATCHUP 10:09 に要約済みなので再掲は不要。

## agent-27b-decode-b2.md (最終更新 10:44)

27B decode 第 2 段 (draft chain の NOSYNC / PREFETCH) の台帳。CATCHUP 10:41 の「第 3 段の的」4 項目とほぼ同一。

- 「NOSYNC を既定 on にするときに親が決めること」節の 1: NOSYNC は検証幅ごとに 4bit 量子化行列積の丸めが変わるため生成列が非ビット一致になる。**この段では KLD も正答率も測っていない**。既定 on が確定した時点 (10:41) で品質ゲートが素通しになっている可能性がある。CATCHUP 側にもこの品質ゲート欠落の明記が無いため、フォローアップの候補として残る。
- 「PREFETCH を畳むなら消す場所」節: `_prefetch_drafts` (旧 698-734 行)、round ループの `pending` 受け取り (旧 1206-1215 行) など削除箇所の一覧。仕上げの節で実際に削除・元の順序に戻したことまで確認済みなので、再度似た先組みを試すときの後戻り手順として使える。

## agent-27b-dispatch-fix.md (最終更新 12:16)

CATCHUP 2026-09-04 12:16 の節に全部写っている。追加は 1 点のみ: 検討して不成立と分かった代替修正案 (`enable_quantized_dispatch` を `enable_default_fusions` より前に当てる) は、qmm_wide が書き換えるのが基底クラスの属性である以上、呼び出し順を入れ替えてもサブクラス側の `__call__` が常に勝つため効かない、という設計上の理由が判明している。同じ手を再度試す必要はない。

## agent-27b-gdn-port.md (最終更新 08:42)

GDN 部品を qwen3_5 (27B) に「構造の契約」で当てた台帳。CATCHUP 08:51 とほぼ同じ結論だが、移植の作業そのものに関する技術メモは docs に無い。

- 「属性名の違い」表 (qwen4_exp ↔ qwen3_5): `n_v`/`num_v_heads`、`n_k`/`num_k_heads`、`dk`/`head_k_dim`、`dv`/`head_v_dim` など。35B-A3B や他族への同種の移植をするときの雛形になる一覧で、CATCHUP には載っていない。
- 活性化関数 (sigmoid/silu) の判定手法: 起動時に合成入力 (8×dv、鍵を固定した乱数) で素の norm と突き合わせてビット一致する側を採用する。対象クラスに `activation` 属性が無くても機械的に決められる、という汎用の移植テクニック。
- MLX 特有の罠: 契約 (`_gdn_spec`) の戻り値を dict/tuple にすると `nn.Module.__setattr__` が子モジュール扱いしてしまう。`SimpleNamespace` を使う必要がある。

## agent-27b-mtp.md (最終更新 08:01)

`docs/BACKLOG.md` (774-797 行付近) に全部写っている。

## agent-27b-sdpa-split.md (最終更新 13:13、走行中)

27B の sdpa 幅分割 (S≥6 の崖) を汎用シーム化する作業。HANDOFF の未決表 (12:45 時点) より後まで進んでいるが、判定は未確定のまま台帳が終わっている。**13:13 時点の状態**として記録する (親が最終結果を追記する前提)。

- 冷 micro (`tools/sdpa_split_generic_micro.py`、Hq24/Hk4/d256=27B の形、gqa 6、w=5): S≤4 は plain と同値 (分割不発火)。S≥6 で `trim` (K/V を前方に切って mask="causal") が最速 (kv=17408 で plain 比 **-61%**、16 層で 26.6 ms/round)。`mask` 変種 (bool マスク実体化) は全幅で +10〜17% なので trim の方が正解。
- **ビット一致ではない** (kv=4096 で bf16 max|diff| 9.77e-4 = 1 ulp 級、MLX が崖の前後で別カーネル (vector 対 fallback) を選ぶため)。分割が起きない幅 (S≤4) は当然ビット一致。
- 既存の qwen4_exp 側のシームは `mask` 変種を使っており、trim に替えれば 13〜18% 残っている、という指摘がある (この台帳の範囲外、`_vendor/` は担当外のため未着手)。
- 13:13 時点の実機 A/B: 短 3 本×256 は ms/tok **-1.6%** だが tok/round +4.0% (非ビット一致で生成が分岐、テキスト運)。4k (`--prefill-once`) は分割が 1 度も発火せず (head/tok/round 完全一致)、+0.4% は位置 1 の段差そのもの。17k (`--prefill-once`) は ms/tok **-1.2%**、ms/round -0.2% で、**判定線 (17k -3%) には届いていない**。発火は 17k・64 トークンで 4/23 round (17%)、想定より高い発火率だが round の大半が S=3〜4 で分割対象外のまま。
- 台帳自体に「採用/畳む」の最終判断が書かれていない。数値だけ見ると判定線未達で、HANDOFF の未決表がこの時点の状態のまま残っている。

## agent-27b-verify-width.md (最終更新 12:59)

27B verify 幅の超線形 (S=2→4 で +27ms) の出所を切り分け、小 M 量子化行列積カーネルを作った台帳。CATCHUP 12:31〜13:05 とほぼ同一の結論だが、カーネル実装上の技術メモは docs に無い。

- `qmv_small_m` 初版は `rps=1` で x の load が rps 倍に膨らみ大負けした (M=4 で stock の 1.6〜4.2 倍遅い)。MLX の `qvm_fast_impl` が選ぶ `rps=4` (x_thread を 1 回作って 4 行で使い回す) に合わせる必要がある。
- 行のずらしを実行時計算 (`min(out_row+r, N-1)`) のままにすると、M=1 でも本家の 1.9 倍まで遅くなる。**コンパイル時定数に固定する**ことが必須 (アドレス計算がブロックごとに残ってしまうため)。
- 候補カーネル比較 (stock/nocap/mma/qmm_wide) の形状別 micro テーブル: 既存 3 本 (M=2〜4) はどれも stock に勝てず、mma は M≥5 でだけ勝つ。この一次データが「新カーネルが要る」という判断の根拠になっている。

## agent-ceiling-audit.md (最終更新 12:07)

MLX 組み込みカーネルの天井達成率を洗い出した台帳。CATCHUP 2026-09-04 12:05 の表とほぼ同一だが、3 点抜けている。

- [t7]/[t10] の仮説 (未検証のまま): バッチ×投機 (B=4×S=3=M=12、B=8×S=3=M=24) が quantized_matmul の最悪帯 (M=8〜12 で達成率 17〜35%) にちょうど乗る。畳んだバッチ×投機レーンの「1.47x」に留まった一因かもしれない、という仮説は CATCHUP に転記されていない。
- 「上位 10 の的」の表のうち #9・#10 (CATCHUP の表は 8 項目まで): HC 周辺の elementwise (rms_hc 83.7%、364 ms/チャンク中 約 100 ms が elementwise 分)、quantized_embedding の 1 行 gather (25.8 us、床の 10 倍だが占有は小さい)。
- [t17] の計測上の注意: `gather_qmm` のバイト計算は 80〜512 行の範囲で一意行数を `idxs[0]` から取る近似のため過大評価になり、達成率が 110% 超に出ることがある (`tools/ceiling_audit_micro.py` を使うときの注意点)。絶対値ではなく帯の形だけを見ること。

## agent-flash-attn.md (最終更新 04:07)

K1 arm A (head_dim 256 の flash attention) の PoL。CATCHUP 04:10 に結論は全部写っている。Metal カーネルを書く際の技法メモが 1 点だけ docs に無い: `#pragma clang loop unroll(full)` を全ループに付けないと配列がスタックに落ちて 2〜3 倍遅くなる (bq4 で 113 → 46 ms)。head_dim 256 のような大きいレジスタ圧のカーネルを書くときの必須事項として記録に残す。

## agent-fn-compile-poc.md (最終更新 12:27)

層単位 `mx.compile` の PoL。CATCHUP 12:28 に結論 (MoE ブロックだけ可) は全部写っているが、検討過程の 2 点が docs に無い。

- `mx.compile(fun, inputs=, outputs=)` で状態を宣言する抜け道は、玩具コードでは機能するが vendor の GDN には効かない。宣言した重み/状態がむしろ `ValueError: uncaptured inputs is not allowed` を起こす (宣言しなかった側は定数として通る、という逆の挙動)。層を「状態を引数で受け取り返り値で返す純関数」に書き直さない限りこの道は使えない。
- 運用上の注意: `decode_ab` 系の A/B に複数の被験カーネル単位をまとめて投入すると、1 つが例外を出す (GDN/attention/layer は compile 不可) だけで走行全体が落ち、健全な単位の測定結果まで失う。単位ごとの事前確認 (1 本流してから本番 A/B へ) を挟むこと。

## agent-fn-moe-compile.md (最終更新 13:12、走行中)

MoE ブロック `mx.compile` の本番配線。PoL (CATCHUP 12:28) を受けての実装作業で、**13:12 時点の状態**として記録する。

- 13:12 時点で本番配線は完了済み (`mlxturbo/fused.py` の `enable_moe_block_compile`、`runner.py` の起動時 warm-up、`decode_ab.py` の knob)。ただし `CLAUDE.md` にはこの knob (`MLXTURBO_MOE_COMPILE`) の記載がまだ無い。
- PoC からの逸脱: 行数上限 `MLXTURBO_MOE_COMPILE_MAX_ROWS` (既定 16) を追加した。理由は (1) prefill には取り分が無く短文脈はむしろ +26.1% (形ごとの初回トレース費用)、(2) `shapeless=True` が使えないため形ごとに Compiled が要り、prefill の末尾チャンク長は要求ごとに変わる (`ctx % 2048`) ので、サーバーを回し続けると Compiled が際限なく増える。
- 13:12 時点でゲート (a) (pytest 436 passed、fingerprint 一致、27B スタブでも無害) は完了。ゲート (b) (in-model A/B、short/17k) は投入済みだが、biglock の列で他エージェント (SMALL_M_ROUTE の A/B) と競合し、この時点で結果が出ていない (`bench/results/moe-compile-{short,17k}-0904.json` はキューに残ったまま、ログ 0 バイト)。
- 再開コマンド: `MLXTURBO_DEPTH_ADAPT=0 tools/biglock.sh .venv/bin/python tools/decode_ab.py --knob moe-compile --model ~/models/ddalcu-mlxlm-head4 --ngram ~/models/ddalcu-ngram --only short --tokens 512 --depth 2 --out bench/results/moe-compile-short-0904.json` (17k は `--only long --ctx 17000 --prefill-once` に差し替え)。

## agent-fn-skip-stack.md (最終更新 11:41)

Flash-Next の「飛ばす/積む」監査。CATCHUP 11:43 と BACKLOG に主要な結論 (state_out の二度書き) は写っている。

- 「(1) 結果を使わない仕事」の表で監査した 7 項目のうち、state_out 以外の 6 つ (indexer q 側 512 列・`_prime_accepted_gap`・pooled/topk/mask の早期 return・`create_attention_mask` (S=1)・層ごとの `arange` 再構築・未受理行の lm_head) は「対応不要 (MLX の遅延で既に飛んでいる)」か「床の下 (0.05%)」で決着済み。CATCHUP は state_out しか代表例を挙げていないが、他の 6 項目も監査済みで的なしと確定している、という完了の記録。
- 運用メモ: 他エージェントが `BIGLOCK_NO_WORKER=1` の 27B ジョブを回すと常駐 worker に停止合図が飛び、短時間の A/B (この台帳のような) はその都度 98GB の読み直しに巻き込まれる。同種の作業をするときはその段だけ `BIGLOCK_NO_WORKER=1` で回すとよい。

## agent-fusions-family.md (最終更新 04:45)

27B で `enable_default_fusions` が落ちる回帰を直した台帳。BACKLOG 774-802 行に主要な結論は写っているが、直した対象の完全な一覧は docs に無い。

- 「既定で到達して落ちるもの」表: `enable_hc_qmm_wide`→`_hc_gated_residuals`、`enable_moe_combine_fold`、`enable_gdn_decode_fused`、`enable_moe_down_epilogue`、`enable_qmm_wide`、`gather_attn.enable_gather_attn`/`enable_prefill_attn`、`qsa_decode.enable_qsa_decode_kernel_default`。既定 off だが同型の直参照を持つもの (`enable_gdn_prework_kernel`、`enable_wide_projections`、`enable_moe_shared_fold`、`enable_fast_rope` 等) も列挙済み。35B-A3B や Gemma4 への横展開時に同じ穴を踏みうる関数の完全なチェックリストとして使える。

## agent-gemma4-smoke.md (最終更新 13:06、走行中)

Gemma 4 (26B) + 27B の 5 エンジン比較。COMPARE-QUEUE.md に大半 (対応状況・起動コマンド・煙試験表) が書き込み済みだが、**13:06 時点でmlxturbo 自身の行だけ未実施**のまま台帳が終わっている。

- 未実施の 3 行 (「1・3 着地」待ちのまま): Gemma 4 draft なし、27B 投機なし、27B 投機あり。
- 再開コマンド (`scratchpad/gemma4_smoke_runs3.sh` の PRIO を 1 に直せばそのまま使える):
  - Gemma 4: `.venv/bin/python -m mlxturbo.server --model ~/models/gemma4-26b-4bit --host 127.0.0.1 --port 8161` (out `bench/results/smoke-gemma4-26b-mlxturbo-nodraft-0904.json`)
  - 27B 投機なし: 同 `--model ~/models/qwen38-27b-4bit --no-mtp --port 8151` (out `smoke-27b-mlxturbo-nospec-0904.json`)
  - 27B 投機あり: 同 `--mtp ~/models/qwen38-27b-mtp --port 8151` (out `smoke-27b-mlxturbo-b2-0904.json`)

## agent-hc-prefill.md (最終更新 03:11)

CATCHUP 2026-09-04 02:55 の節に全部写っている。

## agent-mlx-upgrade.md (最終更新 12:14)

MLX 0.32.3.dev の試験。CATCHUP 12:15 に結論は全部写っているが、3 つの未消化タスクが docs に無い。

- #4408 (モデル読み込みの I/O 並列度を機体に合わせる上流 PR) は「98GB をこの機体で冷やす手段が無い (purge は sudo 権限が要る)」ため未測定のまま残した。効くとすれば TTFT ではなくサーバー起動時間。
- 17k の版比較は depth-adapt controller の交絡で無効と分かったが、深さ固定 (`--depth 2`) の対照スクリプトが `scratchpad/mlxver_phase3.sh` に用意済みのまま未実行 (ユーザー判断で速度 A/B を打ち切ったため)。GPU が空いたら流せる状態。
- 27B は新 venv (`.venv-mlxnew`、mlx 0.32.3.dev) 側で 1 本も走らせておらず、**新 venv で 27B が動くかどうか自体が未確認**。

## agent-moe-counting-sort.md (最終更新 04:49)

BACKLOG 750-756 行に主要な結論 (未配線の記録) は写っている。追加は 2 点。

- micro の内訳 (us/層): hist 9.5 / cumsum 7.4 / scatter 13.4 / segment_tables 35.2 / counts_from_sorted_ids 16.1 / inv_perm 11.6。`segment_tables` が最重量で、専用の 3 本目カーネルを足して 61 → 25.5 us/層に短縮した経緯。
- 運用の罠: モデルを読まないスクリプトでもファイル名が `_micro.py` でないと `tools/biglock.sh` が「要モデル」の段と誤判定し、常駐 worker の 98GB を不要に降ろす。モデル不要の道具は `BIGLOCK_PRIO=2` を明示すること (CLAUDE.md の biglock の説明は一般論のみで、この具体例は書かれていない)。

## agent-moe-fused1.md (最終更新 02:52)

MoE decode `fused:1` の配線と A/B。CLAUDE.md / CATCHUP に最終結論 (既定 auto、`MAX_ROWS=4`) は既に反映済みで、この台帳で「下書きのみ・未実行」と書かれていた 3 件 (MAX_ROWS を 4 に下げる、NAX 判定の追加、本番重み参照テスト) はすべて実装されている (`_MOE_DEC_FUSED_DEFAULT="auto"`、`MOE_DECODE_FUSED_MAX_ROWS` 既定 4、`tools/moe_decode_fused_ref_model.py` が存在) ことをコードで確認済み。追加で残す価値があるのは運用上の罠 2 点。

- `verify_*.py` という名前のモデルを読む道具は、`ab_submit.py` の `TOOL_JOBS` 分類では正しく扱われるが、`tools/biglock.sh` の正規表現 (`*/verify_*.py*`) には引っかかり「モデル不要の micro」に誤分類され、他エージェントの in-model A/B を追い越してしまう。回避には `verify_` を避けた名前にするか、投入時に `BIGLOCK_PRIO=1` を明示する (実際にこの理由で `tools/moe_decode_fused_ref_model.py` という名前が選ばれている)。
- バッチ×投機 (`--max-batch-spec` ≥2 など) で rows が `MOE_DECODE_FUSED_MAX_ROWS` の門を超える構成は一度も実測されていない。門を超えると `_fire` のログにも出さず無言で素にフォールバックする (出荷の既定構成では rows≤3 なので普段は踏まないが、バッチ機能を有効にしたときの未検証の帯として残る)。

## agent-ngram-prefetch.md (最終更新 02:23)

CATCHUP 2026-09-04 02:25 の節に主要な結論は全部写っている。「残った不明点」節の 3 項目と方法論上の注意 1 点が docs に無い。

- 50k は未測定 (`_NGRAM_LOOKAHEAD_WIDTH`=10240 トークンで、50k は 1 回の先読みが次の 1 境界しか覆えず 17k と形が変わる。取り分は同等以上のはずだが未確認)。
- 最初の境界の前景取得 (17k で 1.05 秒、4k で 0.54 秒) は重ねる相手の GPU 仕事がまだ無いため構造的に消せないまま残っている。消すには prefill の外 (サーバーが要求を受けてからプロンプトを組み立てるまでの間) で先読みを始める設計が要る。
- `MLXTURBO_PLE_HOIST` は group prefill 経路 (`_group_prefill_forward` が `layer.pre_mlp` を直呼びする) を通らないため、本番の prefill では今回のまま無効 (未着手)。`bench/results/prefill-anatomy-*` の「PLE/n-gram 6%」はこの変更後に取り直しが要る。
- 方法論上の注意: `tools/decode_ab.py` の `ngram-prefetch` knob は条件切り替え時に `_cache_gen` をリセットしていなかったため、A (先読み on) が積んだキャッシュが B (off) 側に残って全ヒットしてしまう汚染バグがあった。修正済みだが、**この修正前に取られたこの knob の過去の数字はすべて汚染を含む**。

## agent-pseudo-fusion-audit.md (最終更新 10:18)

融合もどき監査。CATCHUP 10:20 に結論は全部写っているが、3 点補足がある。

- 27B S=1 decode のカーネル種別内訳 (dispatch 1152 本の中身): qmv 497、rms_looped 129、vv_Add 128、swiglu compiled 64、gdn_prework/gated_delta/rms_norm_gated 各 48、rms 32・rope 32・g2_copy 32・gg2_copy 32、sdpa 16・v_Sigmoid 16・vv_Multiply 16。
- 計測上の注意: `decode_gpu_trace.py --split-cb` の per-kernel 配分は 27B のような重み帯域律速のモデルでは意味をなさない (qmv 497 本に 19.4 ms しか配分されないが、重み読みだけで 39 ms 要る = 等分配分の癖)。27B ではこの配分値ではなくバイトと回数で読むこと。
- 道具の落とし穴: 出力先を揃えないと上書きされる。`decode_gpu_trace.py` の plain 実行と `--split-cb` 実行を同じデフォルト出力名で連続して回すと、前者のログが後者に上書きされた (`scratchpad/pf-trace-plain.log` に表だけ残る形で発覚)。走行ごとに out-dir を分けること。

## agent-qmm-wide-smallm.md (最終更新 03:59)

`docs/BACKLOG.md` (758-772 行) に全部写っており、指摘された `tools/hc_prefill_micro.py` の docstring 修正も既にコードに反映されている。

## agent-smallkv-attn.md (最終更新 01:44)

CATCHUP 2026-09-04 01:45 の節に全部写っている。

## agent-worker-fix.md (最終更新 03:54)

常駐 worker の不要な降ろされ方を直した台帳。BACKLOG 733-748 行に主要な結論は写っているが、「残った不明点」節の 3 項目が docs に無い。

- `os.execv` (コード鮮度による作り直し) 経由での環境変数継承は、この修正の実地確認では踏んでいない (煙試験は worker が不在の状態からの起動だったため)。単体テストと明示的な環境変数確認では裏取り済みだが、`mlxturbo/` を実際に編集した直後の worker 再作成ログでの確認はまだ。
- ジョブが `FASTMLX_NGRAM_DISK=0` を明示しても、突き合わせ対象から除外したため worker は無視する。n-gram を RAM 常駐にしたくなったら `bundle.mismatch` 側に判定を持ち込む必要がある。
- メモリ待ちの閾値を 100→95GB に下げた効果は、その夜のログでは最大でも 1 回 (15 秒) の待ちしか観測されておらず、実地の効きは次に前 worker が降りた直後の起動で確認する必要がある (未確認のまま)。
