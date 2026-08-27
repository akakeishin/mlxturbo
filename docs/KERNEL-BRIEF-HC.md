# KERNEL-BRIEF-HC — hyper-connections の融合カーネル (2026-08-27)

カーネル専門セッション向けの 2 本目のミッション。1 本目 (docs/KERNEL-BRIEF.md、
Qwen3.8-27B の検証ステップ短縮) とは対象モデルが違う。こちらは
**Qwen3.8-Flash-Next (qwen4_exp)**。

背景は docs/STATUS.md の「速度: 完全にディスパッチ律速だと確定」以降。

## ミッション

`GatedResidual` (hyper-connections) をひとつの Metal カーネルに畳む。
**19.9ms/token を 2ms 前後へ。**

## なぜここか

Flash-Next のデコードは**完全にディスパッチ律速**。一括 forward は S=16 でも
S=1 の 1.17 倍しかかからない (15 トークン増えて 43ms、限界コスト 2.9ms/token に
対し逐次は 47.6ms)。コストのほぼ全部が「カーネルを何回起動したか」で決まる。

部品別の内訳 (`tools/ablate.py`、v-ng2、積み上げ式の無効化):

| 部品 | ms/token | 割合 |
|---|---|---|
| n-gram | 2.9 | 解決済み (連結サイドカー) |
| **hyper-connections** | **19.9** | **残りの 41%** |
| MoE ルーティング | 10.1 | |
| GDN | 7.8 | |
| QSA indexer | 0.08 | 短文脈では走らない。触らなくてよい |
| 残り (embed/lm_head/norm) | 9.9 | |

hyper-connections は **1 層に 2 回 x 48 層 = 96 回**呼ばれ、1 回 0.21ms。
中身は約 15 op なので **1 op あたり 20us** で、これは MLX のディスパッチ
overhead の典型値そのもの。演算量ではなく起動回数の問題。

## 対象の中身

`tools/vendor/qwen4_exp.py` の `GatedResidual.__call__`。48 層 x 2 個
(`attn_hyper_connection` / `mlp_hyper_connection`) と、最終段の
`hyper_connection_mixer` 1 個 (`use_combine=False`)。

```python
normed = self.hc_norm(hyper)                      # RMSNorm、レーンごとに統計
w = nn.silu(self.input_mix_weight_down(normed) / self.hc)   # 10240 -> 320
w = mx.sigmoid(self.input_mix_weight_up(w))                 # 320 -> 10240
w = w.reshape(*w.shape[:-1], self.hc, self.d)
mixed = (w * normed.reshape(*normed.shape[:-1], self.hc, self.d)).mean(axis=-2)
if self.block_inject_weight is None:
    return mixed
inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc)  # 10240 -> 4
return mixed, hyper, inject
```

形 (デコード時、S=1):

- `hyper`: (1, 1, 10240)  hc_count=4 レーン x hidden 2560
- `hc_norm`: RMSNorm(10240, group_size=2560)。**`(1 + weight)` 規約**
  (Gemma 系。ここを `x * w` にすると生成が壊れる。STATUS の「RMSNorm の +1
  欠落」参照)
- `input_mix_weight_down`: QuantizedLinear 10240 -> 320 (hc_lowrank)
- `input_mix_weight_up`: QuantizedLinear 320 -> 10240
- `block_inject_weight`: QuantizedLinear 10240 -> 4
- 量子化は default クラスなので **8bit / group_size 64** (v-stream の場合)。
  レシピで変わるので `lin.bits` / `lin.group_size` を見ること

## 融合しやすい理由

**中間が 320 次元しかない。**スレッドグループ内に収まるので、
「rms_norm -> 量子化行列積 (10240->320) -> silu -> 量子化行列積 (320->10240)
-> sigmoid -> レーン重み付き平均」を 1 カーネルに入れられる。

出力は 2560 次元 (と inject 4 次元) だけなので、書き戻しも軽い。

## 済ませた試行 (やり直さなくてよい)

`mx.compile` を試した。**1.8ms しか減らない** (51.09 -> 49.29 ms/token)。
行列積が間に挟まって elementwise の連なりが 1-3 op ずつに分断されるため、
融合できる区間が短い。実装は `fastmlx/fused.py` に残してある
(`enable_hyper_connection()`)。

**加えて数値が 5% ずれる** (`logits 相対誤差 = 0.0496`、top1 は一致)。
小モデルの差分テストでモデル全体が 1.4% だったことを考えると大きすぎるので、
原因未特定のまま採用していない。カーネルを書くときは下の受け入れ基準で
数値一致を先に押さえること。

## 受け入れ基準

1. ~~**数値**: 同じ入力・同じ重みで、融合前後の logits 相対誤差が **1e-3 未満**。~~
   **2026-08-27 に取り下げ。**この基準はどんな実装でも届かないことが対照実験で
   判明した (素と op 単位で同じで sigmoid だけ fp32 にした、素より*正確*な版が
   7.3% ずれる)。`mx.compile` 版の 5% も実装の誤りではなく bf16 の丸めが 96 段で
   増幅した量だった。品質は 2 の KLD に一本化する。docs/KERNEL-HANDOFF-HC.md 参照
2. **品質**: `bench/quant_eval.py compare` を融合ありで回し、bf16 基準の
   KLD が融合なしと **1e-5 以内**で一致すること
   (v-stream の現在値は KLD 0.00260 / top1 0.9881)
3. **速度**: `tools/ablate.py` で hyper-connections の寄与が **5ms 未満**に
   落ちること
4. 既定は off。`fastmlx/fused.py` の enable/disable と同じ形で明示的に有効化する

## 道具

```bash
# 内訳を測る (部品を積み上げ式に無効化して差分)
uv run python tools/ablate.py --model ~/models/qwen38fn-mlx-v-stream \
    --ngram ~/models/qwen38fn-ngram-4bit

# デコードの内訳と一括 forward のスケーリング
uv run python tools/decode_profile.py --model <model> \
    [--ngram <sidecar> --ngram-mode disk|ram]

# 数値一致と速度を同時に見る雛形
#   scratchpad の hc_test.py が原型。融合前後で logits を比べてから時間を測る

# 品質 (bf16 基準)
uv run python bench/quant_eval.py compare --model <model> \
    --ngram ~/models/qwen38fn-ngram-4bit \
    --continuations bench/results/qe-cont.json \
    --ref-dump bench/results/qe-ref-bf16.npz --tag <tag>
```

測定に使うモデル:

- `~/models/qwen38fn-mlx-v-stream` + `~/models/qwen38fn-ngram-4bit`
  (現在の最良。98.4GB、KLD 0.00260、19.4 tok/s)

`~/models/qwen38fn-mlx-v-ng2` は **2026-08-27 に削除した** (内蔵の空きが
64GB しかなく、焼き 1 本ぶんも置けなかった)。速度でも RAM でも v-stream に
負けていたので、測定は上の 1 本に寄せること。`--ngram` を付け忘れると
n-gram の重みが無くて読み込みが落ちる。

## 境界

- 触ってよい: `fastmlx/fused.py`、`fastmlx/kernels/` 配下、新規のカーネル
- 触らない: `fastmlx/convert_flash.py`、`fastmlx/ngram_stream.py`、
  `bench/` 配下、`tools/vendor/qwen4_exp.py` (親が port の修正を持っている)
- vendored arch を差し替える必要が出たら、親に言うこと。
  **`install-arch --force` を打たないと site-packages に届かない**
  (これで 169GB まで膨らませた前科がある)

## 次の標的 — 融合で取れるのはここだけ (2026-08-27 追記)

**当初この節は「次は MoE 10.1ms と GDN 7.8ms、どちらも同じ起動回数の型」と
書いていた。測ったら違った。** ablate の数字を 1 トークンあたりの読み出し量
(`tools/byte_budget.py`) と突き合わせると、3 つは別種の問題だとわかる。

| 部品 | ablate | 帯域下限 | 差 = 固定費 | 融合の実測 |
|---|---|---|---|---|
| **hyper-connections** | 19.9ms | 1.8ms | **18.1ms** | (このミッション) |
| MoE ルーティング | 10.1ms | 5.1ms | 5.0ms | 0.31ms |
| GDN | 7.8ms | 5.8ms | 2.0ms | 0.59ms |

hyper-connections だけが**ほぼ全部固定費**で、読み出しは 1.8ms しかない。
MoE と GDN は読み出しが大半なので、**融合で取れる上限は固定費のぶんだけ**。

### 天井はいくらか (2026-08-27 再訂正)

上の「融合の実測」列は `tools/micro_moe_gdn.py` で `mx.compile`・否定の省略・
共有エキスパートを switch の第 512 番に畳む、を試した値。**これは弱い手での
実測であって、手書きカーネルの天井ではない。**混同しないよう分けて書く。

MoE の固定費 5.9ms の内訳 (部品を丸ごと外した上限から、消える読み出しを引いた値):

| 畳む対象 | 取れる上限 |
|---|---|
| ルータ頭 (argpartition/take_along_axis/softmax) | ~1.2ms |
| 共有エキスパート (別 MLP を 3 本の行列積で回している) | ~1.3ms |
| 合成 (`* w` -> `sum` -> `astype`) | ~0.85ms |
| gather_qmm 3 本の起動そのもの | 取れない (~2.5ms) |

**MoE の現実的な天井は 3.4ms 前後**、GDN は固定費 2.2ms が丸ごと天井。
hyper-connections の 18ms とは桁が違うので、投入する労力は加減すること。
hc_pre/hc_post が 21ms -> 4.8ms で止まったように、起動費用はゼロにならない。

共有エキスパートについては、**switch の第 512 番として畳む式が厳密に一致する**
ことを確認済み (相対誤差 0.00、`micro_moe_gdn.check_merged_shared_algebra`)。
合成が `sum_e y_e * w_e` なので、w の末尾に `sigmoid(shared_gate(x))` を継げば
同じ式になる。ただし共有だけ 8bit で routed が 4/6bit なので、同じテンソルに
入れるならビットを揃える必要があり、そこはレシピ側 = 親の判断。

### それより大きい未調査ブロックがある

v-fast6 + 融合の 37.84ms の内訳:

| | ms | |
|---|---|---|
| 帯域の下限 | 14.3 | ビットの仕事 |
| MoE の固定費 | 5.9 | 天井 3.4ms |
| hyper-connections (融合後) | 4.8 | ほぼ底 |
| GDN の固定費 | 2.2 | 天井 2.2ms |
| **残り (embed/lm_head/norm/dispatch)** | **~10** | **未分解** |

**「残り」の約 10ms が、いまや最大の固定費で最大の未知。**lm_head の帯域は
1.8ms しかないので、8ms 前後が正体不明の起動コストとして残っている。
親が `ablate.py` を分解して中身を出す。MoE/GDN より大きい鉱脈がここにある
可能性があるので、着手前に親の分解結果を待つ判断もありうる。

## 一般化

GLM-5.3-Flash も **mHC (multi-head hyper-connections)** を持つ。ここで書く
カーネルは構造がほぼそのまま移る見込み (docs/STATUS.md の GLM 節)。
形状を引数で受ける作りにしておくと後が楽。
