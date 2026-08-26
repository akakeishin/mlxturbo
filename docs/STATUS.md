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

## Phase A2 — v2 実装完了、再 GPU gate 待ち

- `fastmlx/kernels/_qmm_skinny_mma_source.py`: PLAN の確定設計だけを入力にした
  clean-room Metal source builder を実装。8x8 `simdgroup_matrix`、group64 単位 dequant、
  8 simdgroup split-K、device `x` 直接 load、M=9..16 の2枚 C tile で B fragment を共有する。
- `fastmlx/kernels/qmm_skinny_mma.py`: M=6..16、4bit/group64、対応 layout/dtype の
  launch と、非対応時の stock `mx.quantized_matmul` fallback を実装。
- `bench/test_qmm_skinny_mma_static.py`: M別 source 構造、partial tile の guarded device
  load、layout eligibility、fallback、launch grid を Metal 非依存で検証。
- `bench/test_qmm_skinny_mma.py`: M=6..16 の stock 比較と、実 MLP shape の依存チェーン
  性能 gate を実装。絶対性能値は参考記録とし、静かなマシンでの最終計測は別途行う。
- 初回 GPU gate は `docs/GATE-RESULTS-A2.md` のとおり不合格。header 改行を修正後、
  M=13 の normalized error `1.95e-3` と M=8 の `1.04x` が基準未達だった。
- v2 は B の threadgroup staging を廃止し、scale/bias と packed word を lane shuffle で
  共有、half A/B fragment と lane-native split-K reduction を採用。fp32 accumulator は維持。
- `docs/HYPOTHESES-A2.md` は Claude/Fable 側の独立仮説として保全。v2 は H4/H5 と
  scale/bias 再読を先に除去しており、再 gate が未達なら PLAN の split-K=8 不変条件を
  保ったまま group64 dequant 配置（H1）を次に切り分ける。

### 非GPU検証結果

- PASS: `python3 bench/test_qmm_skinny_mma_static.py`（6件）。
- PASS: A2 の4ファイルに対する `python3 -m py_compile` と `git diff --check`。

## Phase A3 — 実装完了、GPU gate 待ち

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

## Phase B1 — 実装完了、GPU gate 待ち

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

## Phase B2 — 実装完了、GPU gate 待ち

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

### A2-v2-1 bfloat16 correctness / dependency-chain acceptance

- 正確なコマンド: `uv run python bench/test_qmm_skinny_mma.py --dtype bfloat16 --json bench/results/qmm-skinny-mma-a2-v2.json`
- 期待結果: M=6..16 の normalized max error がすべて `1e-3` 未満、M=8 の依存チェーンが
  stock `mx.quantized_matmul` 比 `1.5x` 以上、M=16/M=8 の MMA chain time 比が `1.6` 以下で、
  JSON が保存される。絶対値は参考記録であり、静かなマシンでの最終計測は別途行う。
- 失敗時に最初に疑う箇所: Metal compile なら half fragment への BF16 device load/cast、
  数値不一致なら lane fragment と packed nibble の対応、性能不足なら split-K reduction。

### A2-v2-2 float16 correctness

- 正確なコマンド: `uv run python bench/test_qmm_skinny_mma.py --dtype float16 --correctness-only --json bench/results/qmm-skinny-mma-a2-v2-fp16.json`
- 期待結果: M=6..16 の normalized max error がすべて `1e-3` 未満で JSON が保存される。
- 失敗時に最初に疑う箇所: full tile の float16 device `simdgroup_load` と partial tile の
  手動 fragment mapping の差。

### A3-1 dispatcher correctness / table calibration

- 正確なコマンド: `uv run python bench/test_dispatch.py --json bench/results/dispatch-a3.json`
- 期待結果: real `QuantizedLinear` の `enable` 前後 normalized max error が `1e-3` 未満、
  全対象 shape×M の dispatch error が MMA は `1e-3`、nocap は `2e-3` 未満で、選択 route が
  stock/nocap/MMA の最速候補から `10%` 以内に入り JSON が保存される。絶対値は参考記録で、
  静かなマシンでの最終計測は別途行う。
- 失敗時に最初に疑う箇所: correctness なら 3D flatten/restore または layer bias 加算、
  性能だけなら該当 `(K,N,M)` の route table entry。

### B1-1 SpecEngine route reachability

- 正確なコマンド: `uv run python bench/test_spec_dispatch.py --json bench/results/spec-dispatch-b1.json`
- 期待結果: M=8 の prefill event はすべて verification inactive、capture には1件以上の
  custom route、head には `(K,N,M)=(5120,248320,8)` の custom route が記録され、JSON が
  保存される。
- 失敗時に最初に疑う箇所: `enable(..., active=False)` で差し替えた class と
  `_hidden_forward` / `_head` の `dispatch_scope` 境界。

### B2-1 one-shot manual-loop identity / speculative mismatch gate

- 正確なコマンド: `uv run python bench/gate.py --max-tokens 96 --n-draft 3 --max-draft 0 --lookup-len 16 --json bench/results/gate-b2.json`
- 期待結果: 3 prompt すべてで raw model manual loop と非投機 baseline が token 列完全一致し
  `baseline_all_identical=true`、speculative-on は一致/不一致のどちらでも `n_mismatch` と先頭
  mismatch の位置・token・context を記録し、JSON が保存される。
- 失敗時に最初に疑う箇所: baseline mismatch なら verification-only dispatch の prefill 漏れ、
  speculative mismatch なら最初の不一致 M に対応する dispatcher route と rollback/consumed。

### 次にやること

1. GPU gate queue を A2-v2 → A3 → B1 → B2 の順に1プロセスずつ実行する。
2. A2-v2 が性能基準未達なら split-K reduction、A3 が性能基準未達なら該当 table entry を修正する。
3. 全 gate 通過後の性能絶対値は別途、静かなマシンで最終計測する。

## コミット状況

- この Codex sandbox では `.git/index.lock: Operation not permitted`（`.git` read-only）のため
  `git add` 自体が拒否され、フェーズ別コミットを作成できない。実装は上記ファイル単位で分離済み。
- hardware gate 待ちの想定メッセージ: `A2: MMA skinny qmm v2（GPU gate待ち）`、
  `A3: shape×M dispatcher を統合（GPU gate待ち）`、`B1: SpecEngine 検証経路を接続（GPU gate待ち）`、
  `B2: manual loop identity gate を追加（GPU gate待ち）`。
