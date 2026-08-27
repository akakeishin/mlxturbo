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

1. **数値**: 同じ入力・同じ重みで、融合前後の logits 相対誤差が **1e-3 未満**。
   `mx.compile` 版が 5% ずれた前例があるので、速度より先にここを通すこと
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
融合が効くのはここだけで、MoE と GDN は削っても 0.3-0.6ms しか出ない
(`tools/micro_moe_gdn.py` で実装して測った。ルータ頭の `mx.compile`、
否定の省略、共有エキスパートを switch の第 512 番に畳む — どれも 0.1-0.3ms)。

**つまりこのミッションを終えたら、融合レーンは一度そこで止まる。**
MoE と GDN の残りはビット配分の仕事で、親が持つ。GDN 投影を 8bit から 4bit に
落とすだけで -3.27 ms/token 出ており、GDN の融合案全部を足したより 5 倍大きい。

積み上げの見積もり: hyper-connections で 33ms 圏 (30 tok/s)、GDN 4bit で
-3.3ms、hc/head/attn もビットが落とせればさらに -2.4ms。合わせて 27ms 圏
(37 tok/s)。現在のビット構成での帯域下限は 16.8ms (60 tok/s) で、それより
下は experts のビットを削る話になる。

## 一般化

GLM-5.3-Flash も **mHC (multi-head hyper-connections)** を持つ。ここで書く
カーネルは構造がほぼそのまま移る見込み (docs/STATUS.md の GLM 節)。
形状を引数で受ける作りにしておくと後が楽。
