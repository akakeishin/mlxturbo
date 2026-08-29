# BENCHMARKS — 実測値の再現手順

README.md と docs/ に載っている実測値が、どのコマンドで出たものかの対応表。
「その数字、自分の機械でも出るのか」を確認したい懐疑的な読者向け。ここに載せた再現コマンドは
実在するスクリプトの実際の引数・出力を確認したうえで書いてある。

**元の測定 JSON は配布していない。**他人の機械の測定値そのものには意味が薄く、下のコマンドを
自分で回せば自分の環境の数字が得られる。ここに残してあるのは「どのキーを見れば主張と突き
合わせられるか」で、当時の観測値も併記してある。

計測環境（共通）: M3 Max 128GB / macOS 26.4 / mlx 0.32.2。ハードウェア世代が変わると数字は変わる
（詳しくは [`research/ROOFLINE-2026-08-26.md`](research/ROOFLINE-2026-08-26.md) を参照）。

## `spec` 経路（Qwen3.8-27B-4bit、greedy、512 tok）

README.md の「実測値と再現コマンド」節、`spec` 経路の表に対応。

| 主張 | 生成コマンド | 結果 JSON |
|---|---|---|
| mlx-lm 素（フォールバック相当）decode 21〜23 tok/s、1.0x | `uv run python bench/baseline.py lmstudio-community/Qwen3.8-27B-MLX-4bit` | `qwen38-27b-4bit-baseline.json` — `runs[].decode_tps` が 23.19〜23.28、`best_decode_tps` 23.28（21〜23 のレンジは、このファイルに加えて `spec-adaptive.json`・`spec-adaptive-mtp4.json` 内の `stock_decode_tps`（21.4〜23.0）を合わせた幅） |
| 自己投機・難しい内容持続（code）31.9 tok/s、1.49x | `uv run python bench/spec_bench.py --mtp-bits 4 --prompts code --json spec-adaptive-mtp4.json` | `spec-adaptive-mtp4.json` の `code.sweep."3".spec_decode_tps` = 31.923677726588753、`speedup` = 1.4912509524906445 — 完全一致 |
| 自己投機・難しい内容持続（prose）28.3 tok/s、1.32x | `uv run python bench/spec_bench.py --mtp-bits 4 --prompts prose --json spec-adaptive-mtp4.json` | 同ファイルの `prose.sweep."3".spec_decode_tps` = 28.26517333697319、`speedup` = 1.319699220254132 — 完全一致 |
| 自己投機・易しい内容（序盤32tok）40〜51 tok/s、1.7〜2.2x | `uv run mlxturbo --model <model> --prompt "<短い/易しいプロンプト>"`（README 記載どおり） | **再現できる保存済み JSON が見つからなかった。** `bench/results/*.json` を全件走査したが、この数値の組み合わせを含むファイルは無い。最初にこの数値を書き込んだコミット（`1c6d503`）のメッセージにはこの実測値が書かれているが、同コミットで追加された JSON 群のどれにも入っていない。対話 CLI (`mlxturbo`) の手元実行結果 (`cli.py:175` が出す `[X.X tok/s | ...]` 行) をそのまま書いた可能性が高いが、そのログ自体は repo に無い。**要フォローアップ（下記「見つかった問題」参照）** |

`bench/spec_bench.py` は greedy 同士で出力トークン列が完全一致することも確認したうえで decode tok/s と受理率を測るスクリプト（`--prompts` に `code`/`prose`/`edit` を指定可能、`--mtp-bits 4` が MTP ヘッドの量子化ビット数）。

## `flash_spec` 経路（Qwen3.8-Flash-Next、v-l レシピ + MTP 4bit、greedy、48 tok）

README.md の「実測値と再現コマンド」節、`flash_spec` 経路の表、および [`MTP-FLASH.md`](MTP-FLASH.md) 冒頭の表に対応。

| 主張 | 生成コマンド | 結果 JSON |
|---|---|---|
| 日本語: 受理率 0.741、貪欲 26.94 tok/s、投機 33.92 tok/s、1.26x | `tools/biglock.sh uv run python tools/spec_flash_bench.py --model ~/models/qwen38fn-mlx-v-l --ngram ~/models/qwen38fn-ngram-4bit --mtp "~/models/qwen38fn-mtp.safetensors"` | **JSON ファイルへの保存なし。** `tools/spec_flash_bench.py` は `print()` で標準出力に出すだけで（既定 `--tokens 48 --reps 3`）、`bench/results/` にはこの数値を保持するファイルが無い。記録は [`MTP-FLASH.md`](MTP-FLASH.md) の表のみ |
| 英語: 受理率 0.516、貪欲 26.78 tok/s、投機 30.02 tok/s、1.12x | 同上 | 同上 |

**この表の数値には既知の疑義がある。** 詳しくは次節。

## 見つかった問題（書き換えていない、報告のみ）

### 1. 「40〜51 tok/s (1.7〜2.2x)」の出典が repo 内に無い

README.md の `spec` 経路表にある「自己投機・易しい内容（序盤32tok）」の行だけ、対応する
`bench/results/*.json` が見つからなかった。他の3行（21〜23 / 31.9 / 28.3）はすべて保存済み
JSON と完全一致するのに対し、この行だけ検証できる一次データが repo に残っていない。

再現コマンド自体 (`uv run mlxturbo --model <model> --prompt "<易しいプロンプト>"`) は実在し動くので、
実行すれば同程度の値が出る可能性は高いが、「40〜51」というレンジの根拠になった具体的な実行ログは
見つけられなかった。数値を消したり書き換えたりはしていない — 判断は書いた本人に委ねる。

### 2. flash_spec 経路の headline 表（README + MTP-FLASH.md 冒頭）が、同じ文書の後半で統計的に非有意と自己訂正されている

[`MTP-FLASH.md`](MTP-FLASH.md) 冒頭の「動いている」表（日本語 0.741 / 英語 0.516、48トークン、
反復30前後の実測）は README にもそのまま転記されている。ところが同じ `MTP-FLASH.md` の
「受理率は課題で変わる」節（97〜124行目）に、著者自身による次の注記がある。

> 標本の大きさに注意。最初 48 トークン（反復 30 前後）で測って「英語 0.516 / 日本語 0.741」と読み、
> 言語差があると報告した。標準誤差が ±0.09 あり有意でない差だった。160 トークン x 10 課題で測り直すと
> 英語の方が高い。受理率は反復 300 以上で見ること。

その再測定（`tools/spec_flash_accept.py`、160トークン×10課題、`bench/results/` には非保存）では
英語 0.720±0.046、日本語 0.682±0.047、コード 0.564±0.068 となっており、**冒頭表とは大小関係が
逆転している**。

つまり README.md と MTP-FLASH.md 冒頭に「動いている」実測として載っている数値そのものは、著者の
手元実行の生ログとしては本物だが、著者自身が同じドキュメントの中で「この差は測定誤差の範囲内で
有意ではなかった」と述べている。読者が「日本語の方が受理率が高い」と読み取れる書き方のまま
残っている点は、ドキュメントの正確性として要修正候補（数値は書き換えていない — 判断は書いた
本人に委ねる）。

## 索引

| 主張の載っている場所 | 検証結果 |
|---|---|
| README.md「実測値と再現コマンド」→ `spec` 経路表、行1〜3 | 再現コマンドあり、結果 JSON と完全一致 |
| README.md「実測値と再現コマンド」→ `spec` 経路表、行4 | 再現コマンドはあるが結果 JSON が見つからない（上記「問題1」） |
| README.md「実測値と再現コマンド」→ `flash_spec` 経路表 | 再現コマンドあり（`tools/spec_flash_bench.py`）だが JSON 保存はしない設計。かつ数値の統計的信頼性に自己訂正あり（上記「問題2」） |
| [`MTP-FLASH.md`](MTP-FLASH.md)「受理率は課題で変わる」節 | `tools/spec_flash_accept.py` による、より信頼できる再測定。README には未転記 |
