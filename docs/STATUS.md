# 実装状況

更新日: 2026-08-26

## Phase 0 — 完了

- `convert.py`: 保存時だけ本体 RMSNorm を raw 規約へ戻す修正、dry-run 層数境界、
  affine group/bits と MTP Linear 整除の preflight、実効量子化 metadata を実装。
- `spec.py`: reuse cache の事前 invalidate と成功時一括 publish、MTP cache validity、
  max_tokens/EOS の effective consumed 整合、sharded capture 拒否、mask/length/advance、
  trimmable cache protocol rollback を実装。
- `gated_delta_states.py`: shape/head/state/mask guard、Dk 非対応時の MLX ops fallback、
  mask=false の single-lane write を実装。
- `_mlx_compat.py`: mlx 0.32.2 系 / mlx-lm 0.31.3 系の version・signature contract と
  内部 import を集約。`pyproject.toml` に上限を追加。
- `cli.py`: `fastmlx_mtp` artifact は同梱量子化 MTP を読み、元 snapshot を不要化。
- `spec_bench.py`: baseline 用 `--lookup-len=0`、列長差を含む exact mismatch、先頭
  mismatch の token・位置・前後文脈 JSON 記録を実装。
- 回帰テスト: `bench/test_spec_phase0.py`、`bench/test_mlx_compat.py` を新設し、
  convert/GDN 既存テストを拡張。6件の「壊れる」ごとに回帰を追加済み。

### Codex 側の事前検証結果

- PASS: 全 Python の `py_compile`、`git diff --check`、`uv lock --check`。
- PASS: NumPy stub で実ファイル `spec.py` をロードし、max_tokens=0/exact cap、accepted
  EOS、非投機 single-token path、reuse failure invalidate を実行。
- BLOCKED: この Codex sandbox は mlx import 時に `No Metal device available` となる。
  Computer Use 経由の Terminal も安全制約で拒否された。`test_spec_phase0.py` は全 skip を
  exit 1 にして、gate 成功へ誤認しないようにした。
- 未実行: `bench/test_gated_delta_states.py`、`bench/test_convert.py` の snapshot test、
  `bench/spec_bench.py --n-draft 0 --max-draft 0 --lookup-len 0`。すべて GPU 1 process で
  直列実行が必要。
- mask=false の反復一致テストは race の実測 sanity に留まり、旧32-lane同値書込みを
  単独で反証するものではない。実装修正は Metal source の single-lane 条件で静的確認済み。

### 「怪しい」の判断

- qmv_wide_nocap の production 未接続: PLAN の Phase A3 dispatcher/`enable(model)` が
  接続点なので Phase 0 では設計どおり foundation として維持し、A3 で integration
  test とともに解消する。
- qmv の M 別 compile/cache 増加: 「改善」指摘であり、A3 の対応 shape 表と明示 cache
  policy に含める。根拠のない prewarm は Phase 0 では追加しない。
- qmv の packing 互換性: mlx `<0.33` を固定し、既存の stock 比較テストを version
  contract gate として維持する。
- GDN state_out 削除と lane-striped state I/O は「改善」指摘。正しさ修正と混ぜず、
  性能差を直列 GPU 計測できる後続項目へ見送る。

### GPU gate 実行結果（2026-08-26、Claude 側で実行）

- PASS: `bench/test_gated_delta_states.py` 全件 bit-exact。fused は逐次 8 起動比 1.93x（参考値）
- PASS: `bench/test_spec_phase0.py` 12/12。ただしスタブの `inner` に `norm` 属性が
  無く実 Metal 環境で 6 件落ちたため修正（generate() が _head の引数として
  `self.inner.norm` を評価するので、オーバーライドしていても属性は必要）
- PASS: `bench/test_convert.py` snapshot テスト含む全件（norm 二重シフト回帰込み）
- PASS（参照再定義のうえ）: identity gate。エンジン n_draft=0/lookup=0 は
  「素のモデル手動ループ（model+make_cache+argmax）」と 96 トークン完全 bit 一致。
  一方 mlx-lm `stream_generate` は手動ループとも位置 49 で分岐する＝分岐源は
  stream_generate の実行環境（専用 stream / wired_limit / 非同期パイプライン）であり
  fastmlx 側ではない。**identity gate の参照は手動ループを正とする**（B2 の
  bench/gate.py はこの定義で実装すること）。stream_generate との差は準同点
  記録として残す

## Phase A2 — v3 実装完了、GPU gate 待ち

- v2 は GPU 実測で M=8 が stock 比 `0.82x` へ回帰したため不採用。
- `fastmlx/kernels/_qmm_skinny_mma_source.py`: MIT の E120 を no-table 形で移植。
  rows/simd=4、values/thread=16、block=512、threadgroup `(32,2,1)`、M=2..9 の
  `activeInputGroups` を忠実に実装。K/N は runtime shape から読み、M=8 は4行×2 group。
- `fastmlx/kernels/qmm_skinny_mma.py`: BF16 affine-4/group64 の単一 runtime-shape pipeline、
  E120 grid、row-contiguous input、非対応時の stock fallback を実装。
- `USE_TABLE` は Phase 2 へ分離。v3 Phase 1 は activation chunk-sum をカーネル内で計算する。
- 禁止事項を静的 gate 化: K/N template、8 rows/thread、24超 accumulator、
  split-K/threadgroup partial、`simdgroup_matrix`、barrier はすべて不在。
- `bench/test_qmm_skinny_mma.py`: BF16 normalized error `<8e-3` を K=512 smoke と
  K=5120 live-K の M=2..9 で比較し、M=8 dependent chain の stock 比 `>=1.5x` を
  acceptance とする v3 gate に更新。

### 非GPU検証結果

- PASS: `python3 bench/test_qmm_skinny_mma_static.py`（8件）。
- PASS: A2 の4ファイルに対する `python3 -m py_compile` と `git diff --check`。

## Phase A3 — 完了（GPU gate 通過、`947972c`）

- `fastmlx/kernels/dispatch.py`: 実測対象の `(K,N)` と flatten 後 M をキーに
  stock/nocap/MMA を選ぶ明示表を実装。未知 shape、非 affine、非対応 M は stock 固定。
- `enable(model)`: 全 `QuantizedLinear` を object identity・parameter path・loaded array を
  保ったまま in-place で差し替える。複数回呼び出しは idempotent。
- `bench/test_dispatch_static.py`: shape×M 選択、unknown fallback、3D flatten/restore、
  stock 引数保存、`enable` の identity/idempotency を Metal 非依存で検証。
- `bench/test_dispatch.py`: real `QuantizedLinear` integration、全表 entry の stock 比較、
  stock/nocap/MMA の候補計測と選択 route の妥当性を一括検証する GPU gate。

### 非GPU検証結果

- PASS: `python3 bench/test_dispatch_static.py`（4件）。
- PASS: A3 の Python ファイルに対する `python3 -m py_compile` と `git diff --check`。
- A2 v3 最終監査時点では `bench/test_dispatch_static.py:33` の旧 `MMA` 期待だけが、
  `947972c` で確定した現行 nocap M=6..10 表と不一致。A2 v3 の範囲では routing table と
  A3 test を変更せず、v3 GPU gate 後の route table 再測定・更新時に同時同期する。

## Phase B1 — 完了（GPU gate 通過、`947972c`）

- `fastmlx/spec.py`: target model の `QuantizedLinear` を verification-only mode で差し替え、
  `_hidden_forward(capture=True)` と `_head` だけを dispatcher scope に接続。prefill と draft は
  stock のまま維持する。tied quantized embedding の head も dispatcher を直接通す。
- `bench/test_spec_dispatch_static.py`: `active=False` installation と capture/head scope の配線を
  AST で Metal 非依存検証。
- `bench/test_spec_dispatch.py`: 実モデルの M=8 prefill が stock scope、capture と lm_head が
  custom route へ到達することを記録する GPU integration gate。

### 非GPU検証結果

- PASS: `python3 bench/test_spec_dispatch_static.py`（1件）。
- PASS: `fastmlx/spec.py` と B1 test の `python3 -m py_compile`、`git diff --check`。

## Phase B2 — 完了（GPU gate 通過、`947972c`）

- `bench/gate.py`: raw model の `make_cache → model(...) → argmax` 手動 greedy loop を
  correctness reference とし、`n_draft=0/max_draft=0/lookup_len=0` baseline の完全一致と、
  speculative-on の mismatch 数・位置・token/context を一実行で JSON 化。baseline mismatch は
  non-zero exit、speculative mismatch は診断記録として保持する。
- `bench/test_gate_static.py`: 列長差を含む mismatch record、manual-loop API、baseline 固定値、
  generation pipeline 非依存を Metal 非依存で検証。

### 非GPU検証結果

- PASS: `python3 bench/test_gate_static.py`（2件）。
- PASS: B2 の Python ファイルに対する `python3 -m py_compile` と `git diff --check`。

## GPU gate queue（必ず1プロセスずつ直列実行）

### A2-v4-1 E120 USE_TABLE BF16 correctness / M=8 dependency-chain acceptance

- 2026-08-26 GPU 実行済み: `uv run python bench/test_qmm_skinny_mma.py --dtype bfloat16 --correctness-only`
  は PASS。K=512/N=1024 と K=5120/N=4096 の M=2..9 がすべて normalized max error
  `<8e-3`（最大 `0.005859375`）。M=4..9 の table ON/OFF は両 shape の全セルで
  `mx.array_equal == true`（bit-exact）。M=2/3 は参照どおり no-table のまま。
- 2026-08-26 GPU 実行済み: `uv run python bench/test_qmm_skinny_mma.py --dtype bfloat16`
  は性能 assertion で FAIL。M=8 dependent chain は stock `1.589666 ms`、v4
  `1.345250 ms`、`1.181688x`（単発は `1.121643x`）で、acceptance `>=1.5x` に未達。
  この実行結果から correctness/bit-exact は GPU 確認済みだが、v4 は性能 gate 未通過。
- 次回の正確な再実行コマンド:
  `uv run python bench/test_qmm_skinny_mma.py --dtype bfloat16 --json bench/results/qmm-skinny-mma-a2-v4.json`
  acceptance は table ON/OFF bit-exact、stock normalized max error `<8e-3`、M=8 dependent
  chain `>=1.5x`。現候補は再計測だけではなく、独立 xsums 起動の固定費を避ける producer
  fusion 等の狭い設計変更が必要。ただし K/N runtime、rows/simd=4、accumulator 最大20、
  split-K 不使用の E120/KERNEL-INTEL 契約は維持する。

### A2-v5-1 direct-load MMA production gate（未実行）

- 対象は production 既定の `fastmlx_qmm_skinny_mma_v5_m{M}_bf16`。M=6..8 は
  C タイル 1 枚、M=9..16 は同一 B 断片を共有する C タイル 2 枚。旧 E120 v4 は
  `implementation="e120_v4"` の明示指定でのみ比較可能。M=2..5 の dispatch は現行の
  stock/nocap のまま変更しない。
- BF16 correctness（K=512/N=1024 と K=5120/N=4096、M=6..16、stock 比 normalized
  max error `<8e-3`）を 1 プロセスで実行:
  `uv run python bench/test_qmm_skinny_mma.py --dtype bfloat16 --correctness-only`
- 依存チェーン acceptance（m=8 で stock 比 `>=1.5x`、v5 の m=16 時間が m=8 の
  `<=1.6x`）を静音窓で実行:
  `uv run python bench/test_qmm_skinny_mma.py --dtype bfloat16 --json bench/results/qmm-skinny-mma-a2-v5.json`
- M3 pipeline/occupancy と冗長 L1 読みの実コストを確認するため、上の性能 gate と同じ
  直列窓で次を順に実行:
  `python3 tools/isa/gen_kernels.py`、`tools/isa/build_air.sh`、
  `tools/isa/gpu_probe.sh`、`python3 tools/isa/gpu_report.py`。
  `current_qmm_skinny`（v5 m=8）の `maxTotalThreadsPerThreadgroup` と
  `ref_fastqmm_m8` を比較し、256-thread launch を満たすことを確認する。値は性能 gate に
  `--v5-max-tptg <current>` と `--reference-max-tptg <reference>` で併記する。
  同じ report の `v_e120_na8_m8_bf16` も保存し、M3 Dynamic Caching が ISA-DIFF の
  E120 棄却判断を反転させないか確認する。4 レーンの同一 32 B 冗長読みは公開 counter で
  直接分離できないため、m=8 単発/依存チェーンの達成時間と帯域床との差を実コストの gate
  とする。
- GPU 実行はこの Codex sandbox では行わない。上記 4 コマンドと correctness/performance
  gate は必ず 1 プロセスずつ直列実行し、結果 JSON と `gpu_report.py` の表を保存する。

## コミット状況

- この Codex sandbox では `.git/index.lock: Operation not permitted`（`.git` read-only）のため
  `git add` 自体が拒否され、フェーズ別コミットを作成できない。実装は上記ファイル単位で分離済み。
- hardware gate 待ちの想定メッセージ: `A2: E120 no-table QMV v3（GPU gate待ち）`。

## Phase D 汎用パック (D1 + D3 + D4) — 実装完了、GPU gate 通過

対象は `fastmlx/spec.py`（唯一の編集者として担当）と新規 `fastmlx/sam.py`。
設計根拠は docs/RESEARCH.md、docs/KERNEL-INTEL.md「深度制御」節。3 本とも
WebFetch で一次ソース本文（AdaEDL 2410.18351、Block Verification 2403.10444、
ReSpec 2511.01282）を確認してから式を採用した（HTML 版が数式を落とすケースが
あったため、arXiv PDF を `pdftotext -layout` で落として本文を読んだ）。
kernels/、bench/gate.py、既存の他ファイルは変更していない。新規テスト
(`bench/test_sam.py`、`bench/test_block_verify.py`) のみ追加。

### D1: 確信度ゲート連鎖

現行の深度ラダー（全受理 +2 / 全棄却 1 / 部分受理 a、`max_draft>0` のときだけ
動く between-step 方式）を、MTP 連鎖の各リンクで判定する within-chain の
ゲートに置換した (`SpecEngine._gate_depth`)。

- 確信度信号は AdaEDL (2410.18351) のエントロピー下界
  `1 - sqrt(gamma * H)`（本文で確認: gamma=0.2 が論文既定値、H は draft
  softmax のシャノンエントロピー、自然対数）。
- 位置別受理率 EMA と KERNEL-INTEL.md の式
  `reach *= p_d`, `threshold = h*(1+expected)/(1+d*h)`
  （h=0.18〜0.20 の中央値 0.19 を採用）を等式どおり実装。`expected` は
  d+1..cap の EMA 積の畳み込み（標準的な期待受理長の式）。
- 実装上の注意点（指示どおり）: 確信度の取得を毎リンク同期にしない。
  MTP 連鎖は cap まで丸ごと lazy に組み立て、各リンクの entropy を
  `mx.eval()` を挟まず配列として溜め、リンクループの後で
  **1 回だけ** `mx.eval(*confidences)` してから Python 側でゲート判定する。
  結果として window に含める長さ (`keep`) を事後的に切り詰める設計にした。
  draft 計算そのもの (MTP ブロック ~1.5%) は cap まで毎回行われる — 節約は
  「検証 window を縮める」側で起きる（README にある通り検証側がコストの
  大半を占めるため、こちらのほうが実質的な効き目が大きい）。
- 深度上限は `max_draft if max_draft > 0 else n_draft` を流用（旧ラダーが
  `max_draft==0` のとき無効化されていたのに対し、新ゲートは常時働く）。
- 事前分布は RESEARCH.md 引用の FastMTP vanilla 実測 (k=1 70% / k=2 11% /
  k=3 2%、それ以降は 0.3 倍ずつ減衰) を使う。

**見つけて直した設計バグ（飢餓状態）**: ゲートが浅く打ち切った回は深い位置を
検証しないため EMA を更新できない。固定 50/50 で AdaEDL とブレンドすると、
「一度低く出た事前分布が観測不足のまま効き続けて浅い判定を再生産する」
飢餓ループに陥ることを `code` プロンプトの実機テストで検出した
(tokens/step が旧ラダー比で悪化)。修正: EMA の重みを観測回数で立ち上げる
`w = n/(n+GATE_EMA_WARMUP)` (GATE_EMA_WARMUP=5)。観測 0 回の位置は AdaEDL の
瞬時確信度だけで判定し、EMA は実測が積み上がってから効かせる。

**受け入れ**:
- `bench/gate.py`: `baseline_all_identical: True`（n_draft=0/max_draft=0/
  lookup_len=0 は D1/D3/D4 適用後も raw model 手動ループと bit 一致、必須
  条件）。git worktree で pre-Phase-D 版と並行実行し比較した。
- 決定論的なゲート挙動の直接検証（内容の分岐に左右されない）: `code`
  プロンプトで `_gate_depth` をラップして 34 ステップ分の (entropies, keep)
  を記録したところ、33/34 回が cap=3 の全深度を維持（README で code は
  もともと受理率が高いドメインと記録されており、整合する）。ゲートは
  「受理見込みが薄いときだけ削る」設計どおりに動作し、良好なドメインの
  深度を不当に削っていないことを確認した。
- tokens/step の新旧比較: 自由生成（同じプロンプトを旧実装・新実装で
  それぞれ独立に生成）は近接同点 argmax の入れ替わり（README に既知の
  現象として記載済み、後述）が数トークンごとに起こり得るため、旧実装と
  新実装が **途中から異なる文章** を生成してしまい tokens/step の単純比較が
  ノイズだらけになった（`code`: 旧 3.02 → 新 2.43〜2.46、`prose`: 旧 2.43 →
  新 2.29〜2.31 と悪化して見えたが、後述のとおり生成内容そのものが分岐した
  ケースを比較していた）。そこで `bench/gate.py`（96 トークン、reference
  と bit 一致することを個別確認済みの実行のみを比較対象にする）で
  **内容が一致するケースだけ** を抜き出すと: `edit` は旧 33 ステップ
  (tokens/step 2.909) → 新 34 ステップ (2.824)、accept_hist は
  `{0:11,1:5→6,2:10,3:4}` でほぼ同一分布、`6/7/12` → `6/8/9`
  という lookup 由来の大口ヒットの位置が変わっただけ（ReSpec の長さ計算式が
  旧の倍々ヒューリスティックと異なる値を出すため）。差は -2.9% でノイズ域。
  `code`/`prose` は旧実装も 96 トークン以内で reference から分岐する回が
  あり内容一致サンプルを取れなかった（両実装とも同じ既知の近接同点現象を
  踏んでいるだけで、D1 由来ではないことは worktree 比較で確認済み — 次項）。
  参考値であることを明記する。decode_tps の絶対値は本セッション中に
  並行エージェントがカーネル側 (`fastmlx/kernels/`, `fast_qmm` 経路切替) を
  複数回コミットしており測定窓を跨いで環境が変わったため、そのままでは
  比較に使えない（同一プロンプトの stock decode_tps が試行間で 9→17 t/s
  のように変動した）。
- `prose` の speculative 出力は 96 トークン中 index 81 から reference と
  分岐する（14 箇所不一致、`accept_hist`・`baseline` は bit 一致）。
  git worktree で Phase D 適用前のコミットを別チェックアウトし同一設定で
  実行したところ **全く同じ位置・同じ分岐内容**（"read-your-writes" vs
  "Need be careful" で始まる文脈）で分岐することを確認した。D1/D3/D4 とは
  無関係の既存事象（バッチ検証の縮約順の違いによる近接同点 argmax の
  入れ替わり、README に記載済み）。系統的劣化ではなく単発の準同点。

### D3: lookup の SAM 化 + ReSpec 仲裁

`fastmlx/sam.py`（新規）: token 列に対する動的 suffix automaton
（Blumer et al. 1985 のオンライン構築をアルファベット非依存にした版、
遷移は dict）。`extend(token)` が O(1) 償却で自動を伸ばしつつ、挿入前の
自動機に対して matching statistics カーソルを進める（＝挿入順序により
「今追加した記号自身との自明一致」を排除でき、常に真に過去の再出現だけを
報告する）。`_lookup_draft` の O(n) 後方走査を置換。静的コーパスは
スコープ外のまま。

- 単体テスト `bench/test_sam.py`: naive O(n^3) 全探索との既知列比較
  （ハンドクラフト + 乱数シード20本 + 単一トークン語彙 + 無反復列の
  エッジケース）で最長一致長・終端位置がすべて一致。状態数が O(n)
  であることも確認 (`len(states) <= 2n+2`)。Metal 不要、GPU なしで実行可。
- 長文生成の探索時間検証: 実モデルの自由生成は他のノイズ源（後述）に
  埋もれるため、SuffixAutomaton 単体を 20,000 token（要件の 2000 の 10 倍）
  合成列（モチーフの反復 40% + ランダム 60%、語彙 4000）で駆動し
  500 token ごとの平均処理時間を計測した。先頭 10% 平均 0.855 us/token、
  末尾 10% 平均 0.859 us/token、比 1.004x — 文脈長に対して完全にフラット
  （O(n) 走査だった旧実装なら比例して伸びるはず）。実モデルでの 700 token
  自由生成も統合テストとして完走を確認済み（30.2 tok/s、lookup 由来の
  8/10 accept が複数回発生、SAM 起因のクラッシュ・リークなし）。

仲裁は ReSpec (2511.01282) 流に置換。本文 (`pdftotext` で抽出、Algorithm 1)
を確認した式をそのまま採用:

- entropy-guided trigger: 直近 confirmed トークンの target 分布エントロピー
  について遡り長 k=1..l (論文の実験値 l=3 をそのまま採用) の平均 H_k と
  `C_k = H_k + lambda_e/k` を計算し、C_k 最小の k* の H_k* が
  `theta_entropy` (論文の実験値 1.5) 以下なら retrieval を起動
  (`SpecEngine._respec_trigger`)。エントロピーは confirmed トークンを
  生成した target 分布の byproduct なので追加 forward 不要、既存の
  `mx.eval()` に相乗り（lookup_len=0 なら計算自体スキップ）。
- feedback-driven candidate scoring: 論文は「一致位置ごとの EMA」を
  複数候補の並列検証（tree）に使うが、fastmlx は単一候補（SAM が返す
  最長一致の直近出現ひとつ）しか持たない。ReSpec の粒度を fastmlx の
  設計に対応させ、「一致長バケットごとの EMA」を単一候補版の類推とした
  (`_respec_bucket`, 幅4のバケット、最大8)。EMA 式は論文どおり
  `S <- (1-alpha)*S + alpha*R`, `R = accepted/proposed`。初期値・閾値は
  図中の記載値 `theta_score=0.5`, `S0=0.5` を採用、`alpha` と `lambda_e`
  は本文に具体値の記載がなく fastmlx 側で選んだ較正値
  (alpha=0.3, lambda_e=1.0)。旧クールダウン方式（全棄却で 4 ステップ休止
  + 倍々長さ）を完全に置換。
- 較正の注意: `theta_entropy=1.5` は論文側モデルの語彙サイズで較正された
  値。Qwen3.8 は語彙 ~150K で最大エントロピー ln(V) が大きく、確信度の
  高い予測でも尾質量の広がりでエントロピーが数 nats になり得る
  （実測: MTP head の link エントロピーは 0〜3.5 nats のレンジで
  top1=0.5〜0.99 の範囲に対応、`entropy_probe.py` で実測済み）。
  1.5 という閾値がこの語彙サイズで最適かは未検証で、次の較正候補。

**受け入れ**: (a) 上記 sam.py 単体テスト全通過。(b) 上記 20,000 token
timing 実測でフラット確認 (1.004x)、実モデル 700 token 生成の統合
smoke test 完走。(c) `edit` プロンプトの tokens/step: 96 token・内容一致
条件で旧 2.909 → 新 2.824（-2.9%、ノイズ域、上記 D1 節参照）。lookup の
大口ヒット位置がずれるのは ReSpec の長さ計算式（`lookup_len * quality
score`）が旧の倍々ヒューリスティックと異なる値を出すため。

### D4: Block Verification

`SpecEngine._block_verify_tau` が temp>0 の逐次棄却サンプリング
（`while a < n_avail and u_l[a] < p_l[a]: a += 1`）を置換。arXiv:2403.10444
の Algorithm 2 本文と Eq.(4)(5)（PDF を `pdftotext -layout` で抽出して確認、
HTML/abs ページは数式を欠落させた）をそのまま実装の出発点にした。

**重要な発見（実装前に導出、Monte Carlo で確認）**: fastmlx の draft は
MTP 連鎖・SAM lookup とも常に決定的な提案（`argmax`、delta 分布
`Ms(x)=1` for the drafted token）である。論文の一般式にこの delta 分布を
代入すると、リンクごとの累積結合比 p_i（Algorithm 2 Line 4）が逐次棄却で
既に使っている `p_l`（drafted token の target 確率）の単純な累積積に
閉じ、受理確率 (Eq.5) と残差分布 (Eq.4) は「次にdraftされたトークンを
除いて target logits を再正規化する」という逐次棄却サンプラの最終ステップ
と**同一の式**に帰着する。違いは τ（受理長）の決め方だけ:
逐次棄却は最初に失敗した位置で打ち切る (`τ = min failing position`) の
に対し、block verification は候補の部分ブロック長すべてを評価して
**最長の成功長** を採用する (`τ = max{L : u_l[L-1] <= h_block(L)}`)。

この閉形式から論理的に導かれる帰結（実装前に予想し、実測で確認した）:
決定的な draft は「他にあり得た draft 列」を持たないため、論文の toy
example（Ms が真の分布で複数の draft パターンがあり得る場合）にある
「ブロックをまたいだ結合の再配分」による利得の余地がそもそも存在しない。
つまり **fastmlx の現行アーキテクチャ（決定的 draft）では、
Block Verification は逐次棄却と分布・期待受理長ともに厳密に一致し、
wall-clock 改善は理論上ゼロになる**。これは実装のバグではなく、
決定的提案という設計選択の帰結（README にある通り、MTP は argmax の
ほうが受理率で優る設計判断であり、変える予定はない）。論文の
5〜8% という数字は真に確率的な small-model drafter を前提にしている。
Monte Carlo (`bench/test_block_verify.py`, 語彙5・ブロック長4のランダム
分布、seed 5 本、各 1e4 サンプル) で τ・resample 後の (τ,Y) 同時分布の
TV 距離が 0.013〜0.019（閾値 0.03 未満）であることを確認、別途 2e4
サンプルでの期待受理長も `E[block] >= E[seq] - 0.05` を満たすことを確認
（実測は `block` がわずかに上回ることが多いが、有意差ではなくノイズ域）。

**受け入れ**: (a) `bench/test_block_verify.py` 全通過（分布同一性の
モンテカルロ確認、Metal 不要）。(b) temp=0.7 での受理長は上記の理論的
帰結どおり逐次棄却と同等（改善ゼロが期待値であり、それ自体が受け入れ
「以上」を満たす）。実装は正しさの契約（分布厳密同一、Theorem 1/2 の
「never worse」）を満たし、決定的 draft を使わなくなる将来（tree 化・
確率的サンプリングを持つ draft 源の追加、D5 の射程）で自動的に効いてくる
設計として残す。

### 規律の遵守記録

- `fastmlx/spec.py` はこのセッションで自分だけが編集。kernels/、
  bench/gate.py、bench/spec_bench.py の引数以外は変更していない
  （実際には spec_bench.py への引数追加も不要だった: 比較は既存の
  `--n-draft`/`--lookup-len`/`--max-draft` と git worktree 分離で足りた）。
- 各項目（D1→D4→D3 の順で実施）完了ごとに `bench/gate.py`
  （`baseline_all_identical`）を実行し、途中で D1 の飢餓バグを見つけた
  ときも修正後に再実行して確認している。最終確認も full run で実施
  (`baseline_all_identical: True`)。
- phase_s のキー (`draft`/`verify`/`maint`) と accept 統計のスキーマ
  (`accept_hist`/`accept_trace`/`src_hist`/`steps`/`mean_accepted`/
  `tokens_per_step`) は変更していない。
- 測定中に並行エージェントが `fastmlx/kernels/` と `docs/STATUS.md` を
  同一ワークツリーで並行編集していたことを確認した（`git worktree add`
  でコミット済み過去版を別チェックアウトし比較する形で、working tree の
  `git stash` 等破壊的操作は使わずに旧実装との A/B を行った）。この
  節を追記する前に `docs/STATUS.md` を読み直し、既存の追記内容を保持した
  まま末尾に追加している。

## spec 出力の同点 flip — 調査完了、バグではないと判定 (2026-08-26)

gate の spec 行が prose で baseline と不一致になる件 (B2 時代から存在) の決着。
`bench/tie_flip_probe.py` の決定実験: spec エンジンを一切通さず、同一の逐次
cache 状態 (index 77) から最後の4トークンを (a) 1つずつ (b) m=4 一括で forward
すると、index 81 で `,` と `.` の logit が bf16 完全同点 (21.875) になり、
一括側だけ `.` が 1 目盛り (0.125) 浮いて argmax が flip する。

- 原因はバッチ形状による縮約順の数値差 x bf16 logit 粒度の完全同点。
  4bit/bf16 モデルでは同点が普通に起きる (96 step 中 gap 0.0 が 1 回、
  0.125-0.25 が 6 回)。m=1 と m>1 のビット同一は成立しない契約
- fast_qmm / lookup / D パックは全て無罪 (経路 stock 化・lookup 無効化でも
  同一 idx で同一 flip、かつ上記の通りエンジン抜きで再現)
- 品質主張は「greedy 同点 flip を除き一致 + KLD 等価」で行う。gate の
  合格基準は従来どおり baseline_all_identical のみ

## 正式ベンチ v1 (2026-08-27 静音窓、3 反復中央値、512 tok、no-think)

環境: M3 Max 128GB、load < 3、render/学習ジョブなし。3 エンジン直列 subprocess、
クールダウン 60s。生データは bench/results/compare-official-rep{1,2,3}.json
(+rep1-mtplx)。fastmlx 構成 = D パック + fast_qmm 経路 + post/post +
静音較正済み経路表 (n_draft 3 / max_draft 8 / lookup 16 / mtp 4bit)。

decode tok/s 中央値:

| | code | prose | edit |
|---|---|---|---|
| mlx-lm 素 | 23.2 | 23.1 | 22.9 |
| MTPLX same-quant (AR-only) | 21.3 | 21.4 | 21.2 |
| fastmlx (同一 ckpt、訓練不要) | 37.5-39.9 | 27.3-28.8 | 34.1-34.3 |
| MTPLX recommended (専用 quant + 訓練 MTP) | 44.2 | 33.0 | 49.8 |

判定:
- 同一 checkpoint 勝負: fastmlx が stock 比 1.2-1.7x、MTPLX エンジン
  (AR-only 21 台 = stock 以下) 比 1.3-1.9x で勝ち
- MTPLX recommended (44.2) への訓練不要チャレンジ: 未達。code -10% /
  prose -13% / edit -31%
- ギャップの所在: 彼らの edit 49.8 は訓練済み再帰 MTP の深い受理
  (depth=3 常用) に由来。うちの edit は tok/step 3.19 で頭打ち。
  訓練不要の残レバー: カーネル (fast_qmm PERF 7 件、B ステージングが最大)、
  D6 (FLy、品質トレード opt-in)、D7 (LogitSpec)、検証コスト削減
- fastmlx の反復間ばらつき ±12% (32.3-40.5)。tok/step は完全に決定的
  (4.44/2.65/3.19)。ばらつきは全て per-step 時間 = 熱/ページキャッシュ由来

## D7 実装済み / D6 実装済みだが現形状では無効果 (2026-08-27)

- D7 (LogitSpec 適応、拡張キー lookup): 採用。edit tok/step 3.19->3.57
  (+12%)、tps 中央値 +3%。code -2.5% tok/step だが tps 上端維持、prose 中立
- D6 (FLy 緩和検証、opt-in --fly-theta): 実装・配線済みだが、論文既定
  θ=0.3 では発火ゼロ、緩め (θ=0.15, W=4) でも 512 tok 中 0-1 回で
  tok/step +0.03。原因は (a) 語彙 248k の正規化エントロピーで θ=0.3 が
  H>=3.7 nats 相当と厳しい (b) 遅延窓 W が入るには不一致後に W トークン
  残る必要があり、深さ 3 前後の MTP 連鎖では窓が入らない (lookup 長ドラフト
  限定)。既定 off のまま維持。lookup 支配が強まるか深いドラフトが実現したら
  再評価。RESEARCH.md の「θ->0 で厳密」は逆だったので本文確認の上訂正済み

## Phase Q 開始: Qwen3.8-Flash-Next 最高品質量子化 (2026-08-27)

対象は Qwen/Qwen3.8-Flash-Next (qwen4_exp、実総数 180B / 活性 6B、
bf16 360GB、GDN:QSA=3:1 ハイブリッド + 512-expert MoE + n-gram 51B +
multi-step 訓練済み MTP 2.6B)。既存公開量子化は 113GB (重すぎ) か
n-gram 欠落品のみで、128GB Mac 向け決定版は空席。

- bf16 アーカイブを外付け (Mobile SSD) へ取得中。レシピ試行錯誤は
  再ダウンロード不要になる
- 動作環境: mlx-lm PR #1788 の qwen4_exp.py を tools/vendor/ へ取り込み
  (MIT、fastmlx 変更は sanitize の mtp.* drop のみ)。実 config で
  ModelArgs 構築確認済み。MTP モジュールは fastmlx 側で自作する
  (27B の mtp.py の再演。mtp.layers.0 は QSA+512expert MoE のフル 1 層、
  hyper-connections 4 レーン)
- 変換系: fastmlx/convert_flash.py (install-arch / estimate / extract-mtp /
  convert)。クラス別 quant_predicate によるレシピ:
  v0-95 = experts 4bit + ngram 3bit + 制御系 8bit = 95.7GB (実台帳から算出)
  v0-105 = ngram 4bit = 102.1GB。感度スキャンで配分を更新する
- ライセンス qwen-community-1.0 の再配布条件は公開前に精読すること

## Phase Q の順序確定 + GLM-5.3-Flash 偵察 (2026-08-27)

順序 (ユーザー確定): Flash-Next を完走 (残り DL 139GB → v0-95 焼き → 起動 →
AR 実測 → 等バイト A/B → v-max) してから、Qwen 系アーカイブを全消しして
GLM-5.3-Flash に移る。空きが増えれば GLM は bf16 取得も視野 (当面 DL 禁止)。

GLM-5.3-Flash (zai-org、MIT) の偵察結果:
- glm5_next / 320B 総 / 18B 活性 / 45 層 / 288 experts / linear_attention +
  deepseek_sparse_attention ハイブリッド + mHC。MTP 内蔵
  (num_nextn_predict_layers=1、DeepSeek 流 NextN)
- 配布は FP8 ネイティブ 328GB (F8_E4M3 314GB + bf16 14GB)。vision 同梱
  (テキスト用途では落とす)
- 128GB に入れるには実効 ~2.5bit 動的量子化 (~100-105GB) が唯一の道。
  活性 18B なので載れば AR ~50-60 t/s + 純正 MTP
- mlx-lm に glm5_next 実装なし (glm_moe_dsa は GLM-5.2 系 DSA のみ)。
  モデル実装の自前ポートが必要 = qwen4_exp より一段重い
- ダウンロード時の注意: Mobile SSD は Qwen 完走後の空き ~284GB に対し
  GLM FP8 328GB で ~50GB 不足。Qwen アーカイブ削除後なら bf16 も視野

### 公開方針 (2026-08-27 ユーザー確定、同日改訂)

動いた変種は良し悪しを問わず HF へ公開し、全てに正直な数字 (KLD/greedy
一致 + M3 Max 実測速度 + gate 結果) を付ける。悪い結果も一次データとして
出す (営利目的ではないので誠実さを優先)。推奨/非推奨はカードの数値と明記で
区別する。レシピ名とコミットで再現可能にする。Flash-Next は
qwen-community-1.0 の再配布条項を精読してから (不可ならレシピ公開へ
フォールバック)。GLM は MIT で制約なし。アップロード実行は都度ユーザーの
承認を取る。

## Phase Q: v0-95 焼き込み完了、port の forward が壊れている (2026-08-27)

bf16 アーカイブ (131/131 シャード、335GiB、外付け SSD) から v0-95 を焼き、
`/Users/ht/models/qwen38fn-mlx-v0-95` に 92GiB (98.8GB、見積もり 98.9GB と一致)。
ロードも生成も通るが、**出力は無意味な反復**で使えない。

### 公開 checkpoint に合わせて直した port のバグ 4 件

vendored した PR #1788 は、公開されている bf16 リポジトリではなく作者の変換済み
checkpoint (`Vontra/Qwen3.8-Flash-Next-MLX-4bit`) に対して書かれていた。

1. `model.language_model.*` — 公開 ckpt は `Qwen4ExpForConditionalGeneration`
   (VLM) 形状。sanitize で `model.*` に畳む
2. `mlp.experts.gate_up_proj` `(512, 1280, 2560)` の融合 — SwitchGLU 用に
   前半 gate / 後半 up へ分解。順序は参照 ckpt から byte-range で該当テンソルを
   取って照合済み (ref.gate_proj vs 元の前半 = 0.085、後半 = 1.33)
3. n-gram の `head_dim=160` が `group_size=64` で割れず、mlx が predicate を
   呼ぶ前に足切りして bf16 のまま残る。51B params がそのまま残り 7.942 bpw
   = 178GB になって OOM した。`group_size=32` に落とす。台帳 (estimate) にも
   同じ割り切れ判定を入れた
4. `NGramEmbedding` のハッシュ乗数を config から再計算していたが、実値と
   一致しない (vocab_sizes / offsets は一致)。ckpt の値を使うよう変更

### 変換系の環境依存

外付け SSD 上の mmap を GPU から読むと、カーネル実行中の page-in が USB 越しに
なって Metal のコマンドバッファが監視タイムアウトする
(`kIOGPUCommandBufferCallbackErrorTimeout`、必ずシャード 2 の保存で再現)。
`convert` / `extract-mtp` は既定で CPU デバイスに置く (`--device gpu` で戻せる)。

### レシピのサイズ (n-gram group_size=32 化で一律 +3.2GB)

| | 旧 | 実測ベース |
|---|---|---|
| v0-95 | 95.7 | 98.9 GB |
| v0-105 | 102.1 | 105.3 GB |
| v-exp6 | 102.0 | 105.2 GB |
| v-max-112 | 112.1 | 112.8 GB (6bit 層を 16 -> 12 に詰めた) |

### 変換は正しい / forward が壊れている

- 逆量子化して元の bf16 と比較: 4bit≈0.09、8bit≈0.01、3bit≈0.17 と正常
- 参照 ckpt と構造一致 (差は `language_model.` 接頭辞と vision tower 333 本のみ)
- 層別量子化の復元も欠けなし (per-path 998 / 量子化テンソル 998)
- 参照 ckpt は一律 4bit/gs32 なので、こちらの方が精度は高い
- 層ごとの活性は健全 (mean|x| 0.05 -> 0.37、NaN も発散も無し) なのに
  ロジットは一様ノイズ (max 7.4 / std 1.6)。**大きさを保ったまま情報を壊す
  構造ミス**が forward のどこかにある
- PLE を無効化しても壊れたままなので、幹の側

PR の「実 checkpoint で生成を確認した」という記述は成立しない。上記 4 の乗数は
作者の checkpoint にも元と同じ値で入っており、port はそれを使わず再計算値で
動くため、参照 checkpoint でも同じ経路を通る。

### 次

参照実装 (transformers main の `qwen4_exp`、2707 行) を
`scratchpad/ref-modeling_qwen4_exp.py` に取得済み。モジュール単位で突き合わせる。
`GatedResidual` (hyper-connections) は照合済みで参照と一致。残りは
`GatedDeltaNet` / `QSAIndexer` / `Attention` / `PLELayer` / 幹の配線。

### 決着: RMSNorm の +1 欠落 (2026-08-27)

参照実装 `Qwen4ExpTextRMSNorm` は Gemma 系の **(1.0 + weight)** 規約で、weight は
ゼロ初期化。port は `x * weight` で **+1 が抜けていた**。学習済み weight は 0
近傍の差分なので、掛けると信号が縮んで向きだけ残る。活性の大きさはそれらしい
まま情報だけ壊れる、という観測した症状と一致する。

効く範囲が広い: 全 GatedResidual の `hc_norm` (2/層 × 48 + 最終 mixer)、PLE の
`norm_key` / `norm_query` / `norm_conv`、QSA indexer の q/k norm。
`RMSNormGated` (GDN の出力側) は ones 初期化の `x * weight` 規約で、port は元から
正しい。直したのは非 gated の方だけ。

修正後、実モデルで正常に生成する:

```
日本の首都はどこですか。一文で答えてください。
-> (think) ... 日本の首都は東京です。
21.3 t/s / prompt 154 t/s / peak 99.2GB
```

### 見つけ方: ランダム重みの小モデルで参照と数値比較

180B を PyTorch で走らせるのは無理なので、**構造は同じで小さいモデル**を作って
突き合わせた。GPU も速度測定も要らないので電力が不安定でも回せる。

- `scratchpad/refenv` に torch(CPU) + transformers git main を入れる (本体の venv
  は触らない)
- `gen_ref.py`: hidden 128 / 4 層 (GDN 3 + full 1) / PLE 1 層 / experts 8 /
  GQA 比 3 / hc 4 レーンのランダムモデルを参照実装で作り、**公開 ckpt と同じ
  レイアウト** (`model.language_model.*`、融合 `gate_up_proj`、n-gram は
  shard_i に分割) で保存。層出力とサブモジュールの入出力を npz に落とす
- `cmp_mlx.py`: port に同じ重みを読ませて層ごとに相対誤差を出す
- `cmp_sub.py`: 参照側で記録した入力をそのまま port のモジュールへ流す。
  上流の誤差が混ざらないので、どのモジュールが違うかが一発で出る

これで層0 → `attn_hyper_connection` → `hc_norm` の出力がゼロ、と 3 手で降りられた。

現状の一致度: hyper-connection と MoE は完全一致、GDN が 0.006、logits で 0.014。
GDN の差は chunked delta rule と MLX の逐次実装の数値差と見ているが未確認。
生成は正常なので、Phase Q の計測を止める理由にはしない。

## Phase Q 決着 (1): ビットは n-gram でなく experts に寄せる (2026-08-27)

等バイト A/B の結果。基準は v-max-112 (5.101 bpw)、相対レーン (KLD は
top-K 256 近似)、31 プロンプト・約 3,700 位置。

| 変種 | GB | n-gram | experts | KLD | top1 一致 | Δp RMS |
|---|---|---|---|---|---|---|
| v-max-112 (基準) | 112.8 | 4bit | 4bit + 12層6bit | 0 (床) | 1.0000 | 0 |
| **v-exp6** | **105.2** | **3bit** | **4bit + 10層6bit** | **0.00181** | **0.9898** | **0.01771** |
| v0-105 | 105.3 | 4bit | 4bit | 0.00284 | 0.9856 | 0.02224 |
| v0-95 | 98.9 | 3bit | 4bit | 0.00319 | 0.9871 | 0.02355 |

**ほぼ同一バイト数 (4.754 vs 4.759 bpw) で v-exp6 が 3 指標とも勝つ。**
部品別でも全カテゴリで勝ち、しかも n-gram を叩く課題ですら 36.9% 改善する
(v-exp6 の n-gram は 3bit、v0-105 は 4bit なのに)。

  ngram 36.9% / experts 30.8% / attn 19.8% / struct 41.0% / reason 38.6% / code 47.9%

差し引きでも同じ結論になる。v0-105 と基準の差は 6bit の 12 層だけで KLD
0.00284、v0-95 はそれに n-gram 3bit が加わって 0.00319。**n-gram 3->4bit の
寄与は 0.00035 で全体の 11% しかない。**払った容量はどちらも約 6.4GB。

n-gram は 51.2B params で容量の 1/4 を占めるが、精度への寄与は薄い。

### 指標の読み方で気づいたこと

- 逐語コピー (copy-rare-ja / copy-hex) は KLD が厳密に 0 になる。分布が鋭す
  ぎて両モデルとも確率をほぼ 1.0 置くため、判別には使えない。**壊れたときに
  即分かる床の確認**として持つ
- 判別力があるのは中間エントロピー帯 (summarize 0.0082 / translate 0.0075 /
  ja-fact 0.0070)
- `stress` の札は「その部品だけが効く」意味ではない。n-gram は層 1 の PLE を
  通して全層に効くので、n-gram を上げると experts タグの課題も改善する
- v0-105 は KLD が v0-95 より良いのに top1 一致だけ悪い。3,700 位置で 5 トー
  クン分なのでノイズの範囲だが、Displacement Is Not Direction の実例

### 計測系の床

v-max-112 の自己比較で KLD / top1 / Δp の 3 指標すべて厳密にゼロ。
Δp は当初「参照の top-1 の確率」と比べていて自己比較でも 0 にならなかった。
正準継続は逐次デコード、ダンプは一括 forward で、bf16 同点の縮約順が変わって
argmax がまれに食い違うため。**同じトークンに対する両者の確率差**に定義を
直して解消 (`{key}.tgt_logp` をダンプに持たせた)。

### 次にぶつかる壁

「n-gram を削って experts をもっと厚く」が明らかな次の一手だが、基準より良い
構成は相対レーンでは順位が付かない (基準の KLD は定義上ゼロ)。
**bf16 teacher パスが前提条件に格上がりした。**

teacher パスは 1 パスで済む。teacher forcing は決まったトークン列を 1 回流す
だけで、層ごとに「重みを読む -> 全プロンプトの活性に適用 -> 捨てる」と進めら
れる。実測の読み出し量は 251.5GB (experts が 241.6GB で 96%、dense は 7.4GB、
n-gram 102GB は位置あたり 16 行しか引かないので実質数 MB)。外付け 500MB/s で
8 分。活性は 31 プロンプト x 約 250 位置 x 10240 次元 x 4B = 約 320MB。
メモリ 10GB 程度で走るので、焼きと同居もできる。

## Phase Q 決着 (2): bf16 の絶対基準が取れた (2026-08-27)

### teacher パスは 9 分で通った

`bench/teacher_bf16.py`。360GB の bf16 を載せずに参照 logits を取る。547 秒、
層あたり 10.5 秒 (n-gram を持つ層 1 だけ 33.8 秒)。31 プロンプト 6,865 トークン。

要になった 2 点:

1. **n-gram 表の実体を持たない。** 素直に層を組むと `_ShardedEmbedding` が
   128 枚の埋め込み (計 102GB) を確保して即死する。位置あたり 16 行しか触ら
   ないので、`DiskShardedEmbedding` で行だけ memmap から引く
2. **experts の融合を層単位で解く。** 変換側 sanitize のうち層に効く部分
   (`gate_up_proj` の分解、conv1d の転置) を再現する

積み残し: memmap のページを層ごとに解放していないので RSS が 109GB まで伸びる
(wired は 2.7GB なので回収可能なファイル由来ページ)。層をまたぐたび memmap を
張り直せば済む。

### 絶対値

| 変種 | GB | n-gram | experts | KLD | top1 | Δp RMS |
|---|---|---|---|---|---|---|
| v-exp6 | 105.2 | 3bit | 4bit+10層6bit | 0.00487 | 0.9816 | 0.02963 |
| v-ng2 | 98.8 | 2bit | 4bit+10層6bit | 0.00531 | 0.9802 | 0.03050 |

v-max-112 基準では 0.00181 / 0.00228 だった。**相対レーンは劣化を 2.5 倍ほど
過小評価する。**基準自体が bf16 から離れているため。

bf16 基準の部品別 (v-ng2):

  experts 0.00665 / attn 0.00579 / reason 0.00503 / code 0.00452 /
  ngram 0.00438 / struct 0.00280

**experts が誤差源として突出。**n-gram を 2bit まで潰しているのに ngram
カテゴリは下から 2 番目で、「ビットは experts に寄せ n-gram は削る」が絶対
基準でも成立する。

### n-gram 2bit の可否 (v-ng2)

n-gram 3bit -> 2bit の代償は KLD +25.9% (0.00181 -> 0.00228, v-max-112 基準)
だが **top1 一致は 0.9898 -> 0.9899 で変わらない**。分布はずれるが選ぶ語は
変わらない。同じ 6.4GB で買える 6bit の expert 層 10 層は -0.00138 なので、
収支は 3 倍近いプラス。

この結果、**v0-105 (105.3GB) は存在価値を失った。**v-ng2 は 6.5GB 小さくて
品質で上回る。~99GB 帯の最良も v0-95 から v-ng2 に入れ替わった (KLD 29% 改善)。

### 容量の狙いどころは ~112GB ではなく ~100GB

full_attention は 48 層中 12 層だけ (kv_heads=2, head_dim=256) なので KV は
1 トークン 24.0KB と軽い。それでも推奨上限 115.4GB に対して:

| 構成 | サイズ | 余裕 | KV の余地 |
|---|---|---|---|
| v-exp-max (未実施) | 112.6 | 2.4GB | 約 98k トークン |
| v-exp6 | 105.2 | 9.8GB | 約 399k |
| v-ng2 | 98.8 | 16.2GB | 約 659k |

1M コンテキストのモデルで 98k しか使えないのは機能の 1 割。**v-exp-max は
焼かずに中止した。**

### 次: n-gram を RAM から追い出す

実データで測った n-gram の量子化誤差 (group_size は 160 が 64 で割れないため
32 固定。1bit は MLX 非対応で下限は 2bit):

| bits | 実効 | サイズ | 相対誤差 |
|---|---|---|---|
| 2 | 3.0 | 19.2GB | 0.3612 |
| 3 | 4.0 | 25.6GB | 0.1710 |
| **4** | **5.0** | **32.0GB** | **0.0810** |
| 8 | 9.0 | 57.6GB | 0.0077 |
| bf16 | 16.0 | 102.4GB | 0 |

誤差 0.36 (2bit) でも KLD +26% / top1 変化ゼロで済んだので、4bit (0.081) は
実質無損失に落ち着くはず。ディスク常駐なら RAM はどれでもゼロなので、選ぶ
基準はディスク容量とページキャッシュへの載りやすさだけになる。

狙う構成:

    n-gram   ディスク 4bit  32GB (RAM 0)
    experts  4bit + 40 層前後を 6bit
    default  8bit
    RAM 約 99GB / KV に 16GB の余裕

現最良の v-ng2 に対し、n-gram の誤差 0.36 -> 0.081、6bit の層 10 -> 40 で
両方が改善する。実装は (1) 変換側で n-gram を本体から外す、(2) 読込後に
`_ShardedEmbedding` をディスク版へ差し替え (teacher のクラスを転用)、
(3) 行ギャザーの高速化 (シャードごとに型付き memmap を張って `mm[row_ids]`
の一発にする)。新規は (3) だけ。

## Phase Q 決着 (3): n-gram をディスクへ追い出して KLD 半減 (2026-08-27)

### n-gram が無いとどうなるか

| 構成 | RAM | n-gram | 6bit層 | KLD (bf16基準) | top1 | Δp RMS |
|---|---|---|---|---|---|---|
| v-exp6 | 105.2 | 3bit RAM | 10 | 0.00487 | 0.9816 | 0.02963 |
| v-ng2 | 98.8 | 2bit RAM | 10 | 0.00531 | 0.9802 | 0.03050 |
| **v-stream** | **98.4** | **4bit ディスク** | **40** | **0.00260** | **0.9881** | **0.02114** |
| n-gram 無し | 79.6 | なし | 10 | 0.04795 | 0.9480 | 0.08126 |

**n-gram は必須。**切ると KLD 9.0 倍、top1 が 98.0% -> 94.8%。壊れ方に偏りが
あり、日本語の事実想起 (ja-fact) が 0.0147 -> 0.4450 と 30 倍悪化する。
code-refactor / translate も 13-14 倍。語彙の共起を担うという設計意図が数字に出る。

**しかし 2bit で価値の 99% が残る。**n-gram の総価値は 0.0426 (無し 0.04795 ->
2bit 0.00531) なのに、3bit -> 2bit で失うのは 0.0004 しかない。相対誤差 0.36 と
いう乱暴な潰し方でこれ。ハッシュ表は量子化に対して異様に頑健。

つまり n-gram は **必須だが安く済み、ビットを足しても得しない**部品。だから
「RAM から追い出して容量をゼロにし、浮いた分を experts に回す」が正解になる。

### v-stream

n-gram をサイドカー (4bit, 30GB) に置き、浮いた 19.2GB を experts に回して
6bit の層を 10 -> 40 に増やした。**v-ng2 比で KLD -51.1%、top1 +0.79pt、
しかも RAM は 0.4GB 少ない。**部品別も全カテゴリで 36-56% 改善。

  experts -56.2% / ngram -51.6% / attn -45.3% / code -42.9% /
  reason -42.7% / struct -35.9%

基準に使っていた v-max-112 (112.8GB) の bf16 基準は 0.003-0.004 と逆算できる
ので、**v-stream は 14GB 小さくて品質が上**。

### 速度の切り分けで 2 回外した

生成が 21.3 -> 14.1 tok/s に落ちて見えたので原因を追ったが、最初の 2 つの
見立てはどちらも外れた。

1. **ディスクのページフォルト** と考えて weight/scales/biases を行ごとの
   1 レコードに交互配置した (フォルト 1/3 の狙い)。効かなかった。
   実測すると引きは **1 トークン 2.9ms** しかない
2. **6bit の matmul が遅い** と考えたが、`gather_qmm` は 4bit 比 3% 差で、
   30 層増えても 0.2ms

真因は **測り方が違った**こと。v-ng2 は `mlx_lm.generate`、v-stream は自作の
素朴なループで測っていた。mlx_lm は `async_eval` で待ちを隠すが、自作ループは
毎トークン同期する。同条件で測り直すと:

| 測り方 | v-ng2 | v-stream |
|---|---|---|
| 素朴なループ | 14.16 tok/s | 12.75 tok/s |
| mlx_lm.generate | 21.28 tok/s | (経路が無い) |

**ディスク運用の代償は 10%。**KLD 51% 減と引き換えなら明確に得。

同時に **自作ループが mlx_lm.generate より 33% 遅い**ことも判明した。以前から
の「活性 6B x 4.47bpw = 3.35GB/token、21 tok/s で実効 70GB/s、M3 Max の
400GB/s に対して 18%」という観測の裏付けで、生成ループ自体に伸びしろがある。
MTP を載せる前にここを潰すのが順当、という見立ては変わらない。

### 踏んだ罠

vendored ファイルを編集しても `cmd_convert` が呼ぶ install-arch は
`force=False` で、古いコピーを黙って使い続ける。`FASTMLX_NGRAM_DISK` が効かず
n-gram が本体に残ったまま 169GB まで膨らんだ。vendor が正本なので毎回上書き
するよう変更した。

## 速度: 完全にディスパッチ律速だと確定 (2026-08-27)

llama.cpp が 64GB 機で 30 tok/s 出しているのに対し、こちらは 128GB 機で
全常駐して 21 tok/s しか出ない。原因を切り分けた。

### n-gram は主因ではない

v-ng2 で PLE (n-gram 経路) を切って比較:

| | tok/s | ms/token | 帯域利用率 |
|---|---|---|---|
| そのまま | 16.99 | 58.9 | 14% |
| PLE 無効 | 21.02 | 47.6 | 18% |

n-gram を**完全に無料にしても 21 tok/s**。残り 47.6ms のうち実際のメモリ転送は
6-8ms しかない。**約 40ms が別の何かに消えている。**

### 一括 forward がほぼタダ

S トークンを一度に流したときの forward 時間 (プロンプト込みの計測なので絶対値
は膨らむが、傾きが要点):

| S | ms/forward | S=1 比 |
|---|---|---|
| 1 | 260.78 | 1.00x |
| 2 | 254.51 | 0.98x |
| 4 | 263.40 | 1.01x |
| 8 | 275.73 | 1.06x |
| 16 | 304.27 | 1.17x |

**16 トークンでも 1.17 倍。**15 トークン増えて 43ms なので、一括の中の 1 トークン
は限界コスト **2.9ms**。逐次の 47.6ms に対して **16 倍の差**。47.6ms のほぼ全部が
forward 1 回あたりの固定費 (カーネル起動 ~2000 回/token と見ている) で、トークン
あたりの仕事ではない。

### 効いてくる結論

**MTP の価値が跳ね上がる。**帯域律速のモデルでは MTP は読み出しの償却でしか効かず
受理率に強く依存するが、ディスパッチ律速では一括検証したトークンがほぼタダになる。
4 トークン一括で 3 受理なら 143ms -> 50ms で約 3 倍。27B で 1.2-1.7x だったものが
この形状ではもっと出るはず。Flash-Next の MTP は multi-step 訓練済みで受理が深い
種類のもの。

優先順位は **MTP -> 固定費削減 (mx.compile / Metal 融合 / C++)**。逆順にすると
遅い実装を高速化してから投機を載せることになり二度手間。

なお速度の切り分けでは今日 3 回外している (ページフォルト、6bit matmul、測り方)。
実装より先に測る規律を守ること。

### 固定費の内訳 (tools/ablate.py, v-ng2)

部品を積み上げ式に無効化して差分を取った。

| 段階 | ms/token | tok/s | その部品のコスト |
|---|---|---|---|
| そのまま | 77.70 | 12.9 | |
| - PLE (n-gram) | 47.79 | 20.9 | **29.92ms (38.5%)** |
| - QSA indexer | 47.71 | 21.0 | 0.08ms (0.1%) |
| - MoE ルーティング | 37.65 | 26.6 | 10.07ms (13.0%) |
| - hyper-connections | 17.74 | 56.4 | **19.91ms (25.6%)** |
| - GDN | 9.91 | 100.9 | 7.83ms (10.1%) |
| 残り | 9.91 | | embed/lm_head/norm |

**上位 2 つで 64%。**

- **n-gram 29.9ms**: 前回の測定では 11ms で、同じ手順で 3 倍ばらつく。19.2GB の
  ハッシュ表へのランダムアクセスがページキャッシュの状態に左右されるため。加えて
  毎トークンの GPU->CPU 同期と最大 16 回の Python ループ (`np.unique` /
  `np.nonzero` / シャードごとの `put_along_axis`)。最大かつ最も直しやすい
- **hyper-connections 19.9ms**: 96 回の呼び出しで 0.21ms/回。中身は約 10 op
  (RMSNorm(10240) -> 線形(10240->320) -> silu -> 線形(320->10240) -> sigmoid ->
  reshape -> 乗算 -> mean)。**1 op あたり 20us** でディスパッチ overhead の
  典型値そのもの。融合すれば 5-10 倍縮む
- **QSA indexer 0.08ms**: 短文脈では budget 2048 が全部を覆うので選択が走らない。
  触る必要なし

### 到達の見積もり

| 手 | 削減 | 到達 |
|---|---|---|
| n-gram をシャード連結して 1 回の gather に | -28ms | 20 tok/s |
| hyper-connections を融合カーネルに | -16ms | 30 tok/s |
| MoE と GDN の融合 | -12ms | 45 tok/s |
| lm_head/embed/残りの dispatch | -8ms | **70 tok/s** |

融合カーネルの積み上げで 70 tok/s 圏まで見える。理論上限は帯域律速で 125-160
tok/s (活性 6B x 約 2.5-3.3GB/token を 400GB/s)。C++ への全面移行はそこまで
やって足りない場合の話。

**注意: n-gram の連結は RAM 常駐版でしかできない。**ディスク版 (v-stream) は
ホスト往復が本質的に要る。品質は v-stream が上 (KLD 0.00260 vs 0.00531)、速度は
RAM 常駐版が上、というトレードオフになる。

## 積み残し: ビット配分の詰め方 (2026-08-27 時点のメモ)

速度が片付いた後に試す。優先順は「既存の道具で測れる順 x 効きそうな順」。

### 1. エキスパート別の配分 (いちばん筋が良さそう)

512 のうちよく点くものと滅多に点かないものがあるはず。頻度に応じてビットを
変える。MoE 特有の手で、実装も軽い。

- 較正プロンプト 31 種を流して `mlp.gate` の top-k を数えるだけで頻度が出る
- 上位 N 個を 6bit、裾を 3bit、のような配分。層ごとに分布が違う可能性もある
- 注意: 較正データに引きずられると未知の領域で壊れる。頻度の低い expert ほど
  「その領域でだけ効く」ので、単純な足切りは危険。裾を厚くする方向の配分も試す

### 2. 層別の感度を測って配分する

いまは `_spread(n)` で等間隔に選んでいるだけで、**どの層が効くかを測っていない**。
`_FIRST5_LAST5` などの端寄せも folklore が根拠。

- `teacher_bf16.py` は層ごとに活性を持っているので、そこで
  `||(W - dequant(quant(W))) @ x||^2` を計算すれば活性込みの感度が 1 パスで出る
  (GPTQ / AWQ 系が使う proxy)
- 出た順に 6bit を割り当てれば、同じバイト数で KLD が下がるはず

### 3. テンソル単位の粒度

experts を層でひとまとめにしているが、gate/up/down で感度が違う可能性がある。
down_proj は入力が中間表現なので分布が違う。2 の測定をテンソル単位でやれば
そのまま分かる。

### 4. group_size の使い分け

いまは experts が 64、n-gram が 32 (160 が 64 で割れないため強制)。
32 は scale/bias の overhead が倍になる代わりに精度が上がる。experts を 32 に
した場合の収支は未測定。

### 5. AWQ 系のスケール探索

チャネルごとのスケールを探索して出力誤差を最小化する。実装は重いが、同じ
ビット数のまま品質が上がるので上限を押し上げる。1-4 を試した後の手。

### 測るときの注意

- 逐語コピー系のプロンプト (copy-rare-ja / copy-hex) は分布が鋭すぎて KLD が
  0 になる。判別には使えないので、床の確認用と割り切る
- 判別力があるのは中間エントロピー帯 (summarize / translate / ja-fact)
- `stress` の札は「その部品だけが効く」意味ではない。n-gram は層 1 の PLE を
  通して全層に効く

## 小容量 Mac への対応 (2026-08-27)

n-gram をディスクに置けることが決定的だった。これが無いと床が 92.4GB
(experts 67.9 + n-gram 19.2 + default 5.2) で、**96GB 機は最初から不可能**。

| 機種 | 推奨上限 | レシピ | RAM | 構成 |
|---|---|---|---|---|
| 128GB | 115.4GB | v-stream | 98.4GB | experts 4bit + 40層6bit、default 8bit |
| 96GB | 約72GB | **v-96** | **70.8GB** | experts 4bit 維持、default 4bit |
| 64GB | 約48GB | **v-64** | **48.2GB** | experts 半分3bit・半分2bit |

96GB は素直に成立する。experts を 4bit のまま保てるので、v-stream から失うのは
6bit の 40 層と default の精度だけ。

64GB は experts を 2-3bit まで落とす必要があり、**experts は bf16 基準の部品別で
最悪 (0.00665)** なので劣化は大きい。動くことを優先した構成として、品質の数字を
添えて出す。

### 64GB をまともにする道

llama.cpp が 64GB 機で 79GB のモデルを 30 tok/s で回せるのは experts も mmap して
ページキャッシュに任せているから。同じことをやるなら:

- experts は 512 個中 10 個しか点かない。1 トークンあたり 480 expert x 2.75MB
  = 1.32GB。全部ディスクから読むと 5GB/s でも 264ms/token で論外
- しかし点火頻度に偏りがあるなら、よく使うものを RAM に置いて残りを流す構成が
  成立する。llama.cpp が実際にやっているのはこれ

**この頻度分布の測定は、品質側の「エキスパート別のビット配分」と同じ測定**なので、
1 回測れば両方の設計が決まる。優先度が上がった。

## GLM-5.3-Flash への移植の見通し (2026-08-27)

Flash-Next の次にやる。今日の成果がどれだけ移せるかを先に整理しておく。

### GLM-5.3-Flash (glm5_next) の形

320B / 活性 18B、45 層、288 experts、FP8 配布 328GB。mlx-lm 未対応で自前ポート。
構成要素は `linear_attention` + `deepseek_sparse_attention` + **mHC
(multi-head hyper-connections)**、MTP 内蔵 (`num_nextn_predict_layers=1`、
DeepSeek 流 NextN、事前学習で共同訓練)。

### そのまま移せるもの

| 今日の成果 | 移植性 |
|---|---|
| ディスパッチ律速の診断 (`tools/ablate.py`) | モジュール名を変えるだけ |
| hyper-connection の融合 | **mHC は同じ構造**。直接効く |
| bf16/FP8 teacher (`bench/teacher_bf16.py`) | そのまま。**これが無いと詰む** |
| 評価一式 (31 プロンプト・部品別の札・KLD/Δp/top1) | そのまま |
| クラス別量子化のレシピ機構 | クラス判定だけ書き換え |
| n-gram のディスク運用 | GLM に n-gram は無いので不要 |

**teacher が特に効く。**GLM は FP8 で 328GB あって 128GB には絶対載らない。
従来なら「参照が取れないので絶対値が測れない」で詰むところを、層ストリーミング
なら 1 パスで取れる。Flash-Next 用に作ったものがそのまま前提条件を満たす。

**MTP も内蔵。**ディスパッチ律速という今日の発見が GLM でも成り立つなら、
一括検証したトークンがほぼタダになるので倍率がそのまま効く。

### 新規に要るもの

`glm5_next` のアーキテクチャ実装。mlx-lm 未対応なので自前ポート。ただし今日
Flash-Next の port を 4 箇所直した経験がそのまま活きる。

**移植で真っ先に確認すべき点: RMSNorm の規約。**`x * w` か `x * (1 + w)` か。
Flash-Next はこれで半日溶かした (活性の大きさは保たれるので発散も NaN も出ず、
情報だけが壊れて生成が無意味な反復になる)。参照実装の `_norm` / `forward` を
最初に読むこと。

### 容量の当たり

320B を 128GB に収めるには約 3.2 bits/weight。活性 18B なので 1 トークンの
読み出しは Flash-Next (6B) の 3 倍になり、帯域律速の側に寄る。ディスパッチ
律速から帯域律速へ移るなら、融合の効きは相対的に下がり量子化のビット数が
効いてくる。**Flash-Next とは最適点が違う可能性が高い。**

### 段取り

1. Flash-Next を仕上げる (速度 + 公開)
2. Qwen 系のアーカイブを消して 335GB 空ける
3. GLM の FP8 328GB を落とす
4. `glm5_next` をポート (RMSNorm の規約を最初に確認)
5. teacher で絶対基準 → レシピ探索 → 融合
