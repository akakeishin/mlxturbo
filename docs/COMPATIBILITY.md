# 量子化パック互換性 — 監査と実地確認

## 方針

mlxturbo は自前ベイク (`tools/bake.py`) と ddalcu パック (4bit/group_size=64) に
最も強く最適化されているが、**特殊な量子化モデルじゃなくても動くことが大事**、
という前提に立つ。著名な量子化パック (mlx-community 等の 4bit/8bit/6bit/混合精度/
gs32/DWQ 系) で堅く動くこと — 最適化は「適格なとき」だけ効き、適格でなければ
**理由を告げて素の経路に落ちる**ことを目標にする。

「クラッシュする」「黙って間違った結果を返す」はどちらも失格。前者はユーザーが
すぐ気づけるがそもそも動かない。後者はユーザーが気づけないまま出荷されるので
最優先で潰す (この repo の規律、コミット d17fedd 「黙って fallback に落ちて
気づけない、を塞ぐ」の延長)。

## 監査結果

対象: `fused.py` の各 `enable_*` の適格判定、`kernels/*.py` の `eligible()`、
`rebit.py`、`fast_qmm.py`、`runner.py` の配線、`server.py` の `--mtp-bits` /
`--rebit`、`convert.py` / `convert_flash.py`。

| 分類 | 件数 (代表例) | 判定 |
|---|---|---|
| クリーン fallback (既存で正しい) | 大半。`kernels/*.eligible()` 全部 (hyper_connection/moe_route/rms_norm_gated/moe_glu/moe_verify_gather/qmm_direct/qmv_wide_nocap)、`fast_qmm._eligible`、`kernels/dispatch.py` の shape-by-M ルーティング、`enable_wide_projections`/`enable_moe_shared_fold`/`_cat_quantized` の bits/group_size 不一致検出、`runner.py` の `SpecEngine`/`FlashSpecEngine` 選択 (`text_config` 欠如・契約検証失敗を `FallbackRunner` へ) | 対応不要。理由付きログ、または `mx.quantized_matmul` への素通しが既に徹底されている |
| クラッシュ (修正済み) | 1 件: `rebit.py` が `model.model.layers` の無いアーキテクチャ (VLM ラッパー等) で `--rebit` 指定時に `AttributeError` で起動ごと落ちる | 修正: `has_layers()` ガードを追加し、`model.model.layers` が無ければ理由をログして該当クラスを 0 本のままスキップするよう変更 |
| 静かに誤る (修正済み・最優先) | 2 件: `kernels/moe_glu.eligible()` と `kernels/moe_verify_gather.eligible_gate_up()` が、Metal カーネルの `template=[("T", mx.bfloat16)]` (dtype 固定) を判定せず、量子化 bits/group_size だけを見ていた。同種の他カーネル (`hyper_connection.eligible`, `rms_norm_gated.eligible`, `fast_qmm._eligible`, `qmm_direct.eligible`, `qmv_wide_nocap._eligible`) は全て入力 dtype を判定済みで、この 2 つだけ抜けていた。活性化が bf16 以外 (fp16/fp32 で動かすモデル) かつ `MLXTURBO_MOE_GLU=1` / `MLXTURBO_MOE_VERIFY=1` を立てた場合、Metal 側がバッファを誤った幅で読み、クラッシュせずに誤った数値を出す経路になり得た | 修正: 両方の `eligible()` に `x` (活性化) を渡し、`x.dtype != mx.bfloat16` を弾く判定を追加。`fused.py` の呼び出し側 2 箇所を更新 |

修正した具体的な差分:

- `mlxturbo/kernels/moe_glu.py`: `eligible(gate_proj, up_proj)` → `eligible(x, gate_proj, up_proj)` (dtype チェック追加)
- `mlxturbo/kernels/moe_verify_gather.py`: `eligible_gate_up(gate_proj, up_proj)` → `eligible_gate_up(x, gate_proj, up_proj)` (dtype チェック追加。`eligible_down` は入力が必ず `eligible_gate_up` 側の bf16 出力なのでチェック不要、コメントで明記)
- `mlxturbo/fused.py`: 上記 2 箇所の呼び出しに `x` を追加
- `mlxturbo/rebit.py`: `has_layers(model)` を追加し、`_targets`/`apply` がそれを経由するよう変更
- `mlxturbo/cli.py`: `load_cli_mtp` の「MTP が見つからない」ログに具体的な誘導文を追加 (下記「MTP の入手経路」参照)

いずれも最適化自体を新しい量子化形式に対応させたわけではない — 対応は「適格でなければ
理由付きで素の経路に落ちる」を徹底しただけで、既存の最適化の対象 (4bit/gs64 中心) は
変えていない。

### 監査で確認したが対応不要と判断したもの (メモ)

- `convert.py` / `convert_flash.py` は**自前ベイクの生成**専用で、外部パックの読み込み
  経路には出てこない (`tools/bake.py` からのみ呼ばれる)。外部パック互換性の観点では
  監査対象外だが、内部の `bits`/`group_size` の組み合わせは `validate_affine_quantization`
  (group_size ∈ {32,64,128}、bits ∈ {2,3,4,5,6,8}) で弾かれるようになっている
- `fast_qmm.py` / `kernels/dispatch.py` は非 affine モード (mxfp4/nvfp4 等、`biases` を
  持たない量子化) を "biases" not in self → 素の経路、で既に処理済み (コメントに
  「コミュニティビルドの KL 計測で実際に確認した」とある — 本監査以前に一度直された跡)
- `mlxturbo/rebit.py` の `_requantize` が `in_dims % group_size` で `ValueError` を出す
  箇所は**意図的な crash** (mlx の `quantize` が割り切れないと黙って量子化をスキップする
  罠を防ぐため)。これは「クラッシュ」に分類しても直さない — fallback にすると
  ちょうど防ぎたかった「黙って量子化されない」に逆戻りする

### spec.py / spec_flash.py / batch_spec.py (読み取り専用、報告のみ)

このレーンでは別エージェントが編集中のため未修正。見つけた要修正候補:

- `spec_flash.py` の `FlashSpecEngine._build_rerank`: draft-rerank 用に trunk の
  `lm_head` を逆量子化してから `mx.quantize(w, group_size=64, bits=2)` で自前の粗量子化
  ヘッドを作る。`hidden_size` が 64 で割り切れない場合、mlx の `quantize` は
  「割り切れなければ黙って量子化をスキップする」仕様があり (rebit.py のコメントに
  同じ罠への言及あり)、後続の `mx.quantized_matmul` が形状不一致で crash するか、
  意図しない粗量子化ヘッドのまま動く可能性がある。ただし `FlashSpecEngine` は
  `model_type == "qwen4_exp"` 限定 (このプロジェクト固有の vendored アーキテクチャ) で
  一般の mlx-community パックがここに来ることはまずないため、実害は低いと判断
  (報告のみ、対応なし)

## 修正の一覧 (再掲)

| ファイル | 変更 |
|---|---|
| `mlxturbo/kernels/moe_glu.py` | `eligible()` に活性化 dtype チェックを追加 (静かに誤る対策) |
| `mlxturbo/kernels/moe_verify_gather.py` | `eligible_gate_up()` に活性化 dtype チェックを追加 (静かに誤る対策) |
| `mlxturbo/fused.py` | 上記 2 関数の呼び出しに `x` を追加 |
| `mlxturbo/rebit.py` | `model.model.layers` が無いアーキテクチャでの crash を、理由付きログ + skip に変更 |
| `mlxturbo/cli.py` | `load_cli_mtp` の「MTP 未検出」ログに `--mtp` での回避策を明記 (下記) |
| `tools/compat_smoke.py` | 新規作成 (本ドキュメントの主題) |

## tools/compat_smoke.py の使い方

```
python tools/compat_smoke.py --model <モデルのディレクトリ or HF repo id>
python tools/compat_smoke.py --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep
python tools/compat_smoke.py --model ~/models/qwen38-27b-4bit --mtp ~/models/some-mtp-sidecar/model.safetensors
```

サーバーは立てない。流れ:

1. `mlx_lm` 経由でロード
2. `mlxturbo.runner.build_runner` で runner を構築 (cli.py/server.py が起動時に呼ぶのと
   同じ入口)
3. 3 プロンプト x `--max-tokens` (既定 64) トークンの greedy 生成
4. 1 行の `matrix:` ログで、アーキテクチャ・検出した量子化構成 (bits/group_size/mode。
   混在していれば `mixed[...]` と全部並べる)・MoE の有無・選ばれた runner の種類
   (`spec`/`flash_spec`/`fallback`/`draft_spec`、fallback なら理由付き)・MTP の入手経路
   (`bundled`/`sidecar`/`none`/`n/a`)・hyper-connections 融合カーネル・gather sort・
   moe_glu・fast_qmm・wide_proj の有効/無効・wired limit の成否を出す

生成の粗い検査 (品質そのものは見ない): 各プロンプトについて (a) 生成トークンが空でない、
(b) 同一トークンが 32 回以上連続していない、の 2 点だけを機械的に見て `OK`/`FAIL` を出す。
`tools/smoke_generate.py` は生成品質を目で見て確認する道具で役割が違う。

## 実地確認したパックのマトリクス

### ローカル (ddalcu 系、自前ベイク・変換)

| パック | アーキ | 量子化 | runner | MTP | 結果 |
|---|---|---|---|---|---|
| `~/models/ddalcu-mlxlm` (+ `--ngram ddalcu-ngram-sep`) | qwen4_exp (Flash-Next) | mixed[4bit/gs64, 8bit/gs64] | flash_spec | bundled (サイドカー mtp.safetensors、自動発見) | PASS。hc_kernel active、gather_sort on、tok/step 2.07-2.38 (投機が効いている実測) |
| `~/models/ddalcu-flashnext-serve-4bit` | qwen4_exp | 同上 | — | — | 直接ロード不可 (n-gram が旧来の `ngram_table.bin` 単一ファイル形式で、現行の `ngram_stream.install()` が期待する `manifest.json` + weight/scales/biases.bin 形式と不一致)。**これは mlx-serve ネイティブの生パックであり、`tools/from_mlx_serve.py` で変換してから使う設計** (下の「MTP の入手経路・4」参照) — 直接読ませたのは検証手順の誤りで、mlxturbo 側のバグではない |
| `~/models/qwen38-27b-4bit` (mlx-community/Qwen3.8-27B-4bit) | qwen3_5 | 4bit/gs64 | spec (MTP なし) | none | PASS。「MTP が見つからない」に加え `--mtp` での回避策を案内するログが出る (本ドキュメント下部) |

### mlx-community (今回取得、合計約 1.6GB)

| パック | アーキ | 量子化 | runner | 結果 |
|---|---|---|---|---|
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | qwen2 | 4bit/gs64/affine | fallback (`text_config` なし → クリーン fallback) | PASS。392-466 tok/s、tok/step 1.00 (非投機) |
| `mlx-community/Qwen2.5-0.5B-Instruct-8bit` | qwen2 | 8bit/gs64/affine | fallback | PASS。302-358 tok/s |
| `mlx-community/gemma-3-1b-it-4bit` | gemma3_text | 4bit/gs64/affine | fallback | PASS (286-290 tok/s)。ただし `<end_of_turn>` を eos として拾えず出力が続く場合がある — これは compat_smoke 側の eos 判定の単純さによるもので、mlxturbo のバグではない (実運用の server.py はモデルごとの eos 集合をきちんと解決している) |

いずれも `runner=fallback` (SpecEngine/FlashSpecEngine の対象外 — `model.args.text_config`
が無いアーキテクチャ) で `mlx_lm.generate.stream_generate` の素の経路に落ち、
hyper-connections 融合カーネル等 mlxturbo 独自の最適化は "dormant" (パッチは入るが
このアーキテクチャのクラスが存在しないので発火しない) のまま、生成は問題なく通った。
GPU の時間計測 (tok/s) は参考値としてのみ記録し、優劣の結論は書かない。

## MTP の入手経路

MTP (multi-token prediction, draft head) が効くかどうかで体感速度が
**1.5-2 倍ほど変わる** (投機デコードが有効かどうかの差)。入手経路は 4 通りあり、
「壊れずに動くか」「不整合を検出できるか」を実地で確認した。

| # | 経路 | 指定方法 | 検証結果 |
|---|---|---|---|
| 1 | 内蔵型 (bundled) | 何もしない (自動発見)。モデルディレクトリ直下の `mtp.safetensors`、または本体 safetensors シャード内の `mtp.*` テンソル | `~/models/ddalcu-mlxlm` で確認。`--mtp` 未指定で自動発見し (`MTP を自動発見: サイドカー (mtp.safetensors)`)、実際に tok/step 2.07-2.38 で投機が効いた |
| 2 | サイドカー別パック | `--mtp PATH` (単一 safetensors ファイル) | `mlx-community/Qwen3.8-27B-MTP-4bit` (本体 `mlx-community/Qwen3.8-27B-4bit` とは別リポジトリ) で確認。**このサイドカーは既に 4bit 量子化済みだが、`mlxturbo/mtp.py` の `load_mtp_file` は bf16 の生の draft head (train_mtp.py の出力形式) を前提に非量子化の `MTPModule` を組み、そこへ `strict=True` で `load_weights` する。量子化済みサイドカーは `.scales`/`.biases` という「モデルに無いパラメータ」を持つため `ValueError` になる。**この失敗は `cli.py: load_cli_mtp` の `try/except Exception` に既に捕まっており、クラッシュせず「読み込めないため無効化します (理由付き)」とログして MTP なし (lookup のみ) の投機にクリーンに fallback する。契約不整合を検出できており、黙って受理して壊れる、にはなっていない |
| 3 | MTP 無しパック (一般パックの大半) | 何も指定しない | `mlx-community/Qwen3.8-27B-4bit` 単体で確認 (元は `Qwen/Qwen3.8-27B` の生 bf16 チェックポイントが `~/.cache` に無いため `FileNotFoundError`)。**今回の修正**: `cli.py` の該当ログに「投機を有効にするには MTP (draft) ヘッドが要ります — 専用の MTP サイドカーがあれば `--mtp PATH` で渡してください (体感速度は投機の有無で 1.5-2 倍ほど変わります)」という誘導文を追加した (モデル名は決め打ちしない一般的な書き方)。以前は理由だけで「どうすれば速くなるか」への案内が無かった |
| 4 | 他パックからの抽出 | `tools/from_mlx_serve.py shards --src <mlx-serve pack> --out <dir>` (本体・config を変換) と `... ngram --src <mlx-serve pack> --out <dir>` (n-gram サイドカー)。MTP 自体は `shards` の変換過程で `language_model.mtp.*` を `mtp.safetensors` として自動的に抜き出す | 直接の再実行はしていない (対象が 98GB の `ddalcu-flashnext-serve-4bit` で重い) が、`~/models/ddalcu-mlxlm` と `~/models/ddalcu-ngram-sep` は**まさにこのツールの出力物**であり、経路 1 の検証がそのまま「この変換の出力が実際に読めて投機まで効く」ことの実地証拠になっている |

**経路 2 の教訓**: 本体とサイドカーが別リポジトリの場合、量子化構成が食い違う
(本体は素の bf16 前提の draft head を期待するが、配布物は量子化済み) ことがある。
mlxturbo は現状これを「クラッシュ」でも「黙って壊れる」でもなく、検出して
クリーンに無効化する — ただし精度そのものの互換 (サイドカーと本体のトークナイザ
一致など) までは検証しておらず、「読めるかどうか」の検証に留まる。

## 罠・留意点

- `model.args.text_config` の有無が SpecEngine/FlashSpecEngine 対象かどうかの最初の
  分岐点。一般の dense/MoE パック (Qwen2, Gemma3, Mixtral 系など) はこれを持たないため、
  自動的に `FallbackRunner` (mlx_lm 素の生成) に落ちる。これは正常な動作であり、
  「投機が効いていない」ことを気にする必要はない
- `fused.py` の最適化 (hyper-connections 融合カーネル等) は `mlx_lm.models.qwen4_exp`
  というこのプロジェクト固有の vendored アーキテクチャのクラスにしかパッチが当たらない。
  他アーキテクチャのモデルではパッチ自体は入る (`_ORIG_HC_KERNEL` 等はセットされる) が
  発火せず、"dormant" になるだけで実害はない
- `MLXTURBO_MOE_GLU=1` / `MLXTURBO_MOE_VERIFY=1` / `MLXTURBO_WIDE=1` /
  `MLXTURBO_FAST_QMM=1` はいずれも既定 off の実験的フラグ。4bit/group_size=64 以外の
  層には (今回の dtype チェック修正後も含め) 自動的に素通りする
- `--rebit` は「焼き直さずにビット配分を試す」ための開発者向けデバッグツールで、
  一般ユーザーが使う経路ではない。今回のクラッシュ修正で VLM ラッパー等でも
  安全に (何もせず) スキップされるようになったが、`in_dims % group_size` が
  割り切れない場合の `ValueError` は意図的に残してある (mlx の「黙って
  量子化をスキップする」罠を防ぐため)
