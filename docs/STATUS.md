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

## コミット状況

- この Codex sandbox では `.git/index.lock: Operation not permitted`（`.git` read-only）のため
  `git add` 自体が拒否され、フェーズ別コミットを作成できない。実装は上記ファイル単位で分離済み。
- hardware gate 待ちの想定メッセージ: `A2: E120 no-table QMV v3（GPU gate待ち）`。
