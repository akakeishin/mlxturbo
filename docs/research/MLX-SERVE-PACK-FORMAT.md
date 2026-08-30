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

## 犯人は RMSNorm の +1 だった (決着)

このアーキの `Qwen4ExpTextRMSNorm` は `x * (1 + w)` を計算する。向こうのパックは
**+1 を畳み込んだ重み**を持ち、エンジンは素直に `w` を掛ける。うちは HF の生の
重み (ほぼ 0) を入れていたので、向こうで読むとほぼ 0 倍になり、生成が同じ
トークンの反復になっていた。

形も dtype も 2762 群すべて一致し、値も bf16 の元と突き合わせて向こうと同じ
量子化誤差だったので、突き止めるのに時間がかかった。潰した容疑者は全部無実:
混在ビット (一律 5bit でも壊れた)、MTP、vision、n-gram 表、量子化する範囲、
lm_head と embed のビット、config の形、融合カーネル。

対象は向こうの `tests/convert_qwen38_flash_next.py` の `NORM_FOLD_SUFFIXES`:

    hc_norm / q_norm / k_norm / q_layernorm / k_layernorm
    ple.norm_key / ple.norm_query / ple.norm_conv
    pre_fc_norm_embedding / pre_fc_norm_hidden

`linear_attn.norm` はゲート付きで素の重みなので触らない。`tools/to_mlx_serve.py
fold` がこれをやる。**2 回かけてはいけない**ので、済んだ印を `.norm_folded` に置く。

うちの `tools/smoke_generate.py` の説明にも「RMSNorm の +1 欠落は生成が無意味な
反復になったが活性の大きさはそれらしいままだった」と書いてある。同じ罠を
逆向きに踏んだ。

## 混在ビットは無実

+1 を直したら一律 5bit が通った (43.1 tok/s、一律 4bit は 51.5)。混在が読めるか
どうかはこれとは別の問題で、まだ試していない。

## 向こうの変換スクリプトから判ったその他の約束

- bf16 のまま置くのは「1 次元」「2 次元だが行数 32 未満」「最終軸が 64 の倍数
  でない」もの。行数 32 未満に当たるのは `shared_expert_gate`・
  `block_inject_weight`・GDN の `in_proj_a/b`。router の `mlp.gate` は 512 行
  あるので量子化される
- `embed_tokens` は既定で 4bit gs64 固定
- n-gram 表の 3/5/6bit は mlx-serve 26.9.1 以降でないと雑音として読まれる (#305)。
  手元は 26.8.11 なので 4bit に留める
- 深さ方向の conv1d は HF の `[C, 1, K]` のまま置く

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
