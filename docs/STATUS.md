# 実装状況

更新日: 2026-08-26

## Phase 0 — 一部完了（実装完了、GPU/snapshot gate 未実行）

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

### 検証結果

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

### 次にやること

1. Phase A2 の clean-room MMA skinny qmm に着手する。
2. B2 の bench/gate.py は上記の参照定義（手動ループ）で実装する。
3. 性能絶対値の最終計測は別途、静かなマシンで行う。
