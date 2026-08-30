# mlx-serve が読むパックの形 (2026-08-30 実測)

参照は `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`、エンジンは MLX Core.app 同梱の
mlx-serve 26.8.11 (MLX 0.32.2)。うちの v-l を変換して読ませながら調べた。

## 判っている形

| | 向こう | うち |
|---|---|---|
| テンソル名 | `language_model.model.*` / `language_model.lm_head.*` | `model.*` / `lm_head.*` |
| vision | `model.visual.*` (prefix 無し) を `model-vision.safetensors` に、index にも載せる。bf16 のまま 898MB | 変換で落としている |
| MTP | `language_model.mtp.*` を trunk のシャードに同居。**量子化済み**で入れる | `mtp.safetensors` に bf16 で分けて、読み込み時に量子化 |
| エキスパート | `switch_mlp.{gate,up,down}_proj` の 3 本 | 同じ (mtp サイドカーだけ融合 `experts.gate_up_proj` のまま) |
| n-gram 表 | プレーナ。`ngram_table.bin` に safetensors ヘッダ + weight 全行 / scales 全行 / biases 全行 | インタリーブ (1 行 100 バイトのレコード) |
| config | `quantization` は 3 キーだけ。ビットは**テンソルの形から割り出す** | mlx_lm 形式の per-tensor エントリ 873 個 |

`config.quantization.bits` を 6 に書き換えても MoE カーネルは `bits=4` のまま
動いた (層 0 が 4bit だから)。config のビット数は表示用で、実際は形を見ている。

## 量子化する範囲がエンジンごとに違う

**ここが罠。**名前は一致するので読み込みは通り、出力だけが壊れる。

| モジュール | 向こう | うち |
|---|---|---|
| `mlp.gate` (router) | 量子化 | 生 bf16 |
| `*_hyper_connection.block_inject_weight` | 生 bf16 | 量子化 |
| `mlp.shared_expert_gate` | 生 bf16 | 量子化 |

48 層ぶんで 192 群ずれる。`tools/to_mlx_serve.py align` が参照パックを見て
揃える。揃えると decode が 27 -> 45 tok/s に上がった (速いカーネルに乗る)。

## 未解決: 混在ビットのパックが壊れる

v-l を変換して読ませると、**読み込みは通り、生成が同じトークンの反復になる。**
`_coordinates_coordinates...` のような形で、崩れ方は決定的 (毎回同じ)。

潰した容疑者:

- n-gram 表 — 向こうの `ngram_table.bin` に差し替えても同じ。プレーナ変換は
  バイト単位で照合済み (5 行を全ブロックで一致確認)
- 量子化範囲のずれ — `align` で揃えた。速くはなったが崩れは同じ
- 融合 MoE カーネル — `MLX_SERVE_MOE_*_FUSED=0` で外しても同じ
- `lm_head` の 6bit — 8bit に焼き直しても同じ
- `embed_tokens` / `hyper_connection_mixer` の 8bit — 4bit に焼き直しても同じ
- config の per-tensor エントリの有無 — どちらでも同じ

残っているのは、v-l が**参照と 1171 群中 606 群でビット数が違う**こと。
6bit が 36 層の GDN 全体・12 層の self_attn 全体・末尾 8 層のエキスパート、
8bit が hyper_connection 系の全層に散っている。どれか (あるいは複数) を
向こうが決め打ちの幅で読んでいる。

Metal ソースの分岐は `BITS == 4` / `5` / `8` が各 213 箇所、`6` が 1 箇所、
2 と 3 は 0 箇所。実行時メッセージは `this MLX runtime supports 2, 3, 4, 5, 6, 8`
と言うので、読めはするが速い経路は 4/5/8 に限られる。

## ここから言えること

**mlx-serve 向けのパックは、ビット幅を揃えて焼く。**混在は、動く保証が無い
うえに速い経路からも外れる。揃えるなら 4 / 5 / 8 のどれか。

パックの 93.7% はルーテッドエキスパート (gate/up/down で各 31.2%)。
lm_head は 0.9%、それ以外は全部合わせて 4%。**大きさと速さはエキスパートの
ビット数だけで決まる。**一律 4bit / gs64 の実効は 4.5 bit。

- 一律 5bit → 実効 5.5 bit、エキスパート 77 GiB。128GB 機なら載る
- 一律 3bit → 実効 3.5 bit、エキスパート 49 GiB。速い経路から外れる
- 4bit のまま gs128 → 実効 4.25 bit、-5.6%。速い経路に残る

## 変換の手順

    uv run python tools/to_mlx_serve.py shards --src <ours> --out <native>
    uv run python tools/to_mlx_serve.py mtp --src <ours>/mtp.safetensors \
        --out <native>/model-00017.safetensors --reference <ddalcu pack>
    uv run python tools/to_mlx_serve.py align --pack <native> --reference <ddalcu pack>
    uv run python tools/to_mlx_serve.py ngram --src <ngram sidecar> --out <native>

シャードはヘッダだけ書き直して本体をバイト列でコピーするので 30 秒程度。
n-gram の並べ替えが 25 秒、align が数分。
