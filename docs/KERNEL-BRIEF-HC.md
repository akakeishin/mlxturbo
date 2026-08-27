# KERNEL-BRIEF-HC — hyper-connections の融合カーネル (2026-08-27)

カーネル専門セッション向けの 2 本目のミッション。1 本目 (docs/KERNEL-BRIEF.md、
Qwen3.8-27B の検証ステップ短縮) とは対象モデルが違う。こちらは
**Qwen3.8-Flash-Next (qwen4_exp)**。

背景は docs/STATUS.md の「速度: 完全にディスパッチ律速だと確定」以降。

## ミッション

`GatedResidual` (hyper-connections) をひとつの Metal カーネルに畳む。
**19.9ms/token を 2ms 前後へ。**

## なぜここか

hyper-connections は**読み出しがほとんど無いのに時間だけ食う**。1 トークンあたり
0.682GB (= 385GB/s なら 1.8ms) しか読まないのに、実測 19.9ms かかる
(`tools/byte_budget.py` と `tools/ablate.py`)。差の 18ms は起動回数そのもの。

> **2026-08-28 訂正**: この節は当初「一括 forward は S=16 でも S=1 の 1.17 倍
> しかかからない」を根拠にしていたが、その計測は誤りだった (プロンプトの
> forward を eval せずにタイマーを開始していた)。正しくは 2.53 倍で限界
> トークンは 5.77ms。**このミッションの判断自体は変わらない** — 上の
> 「読み出し 1.8ms に対し実測 19.9ms」という独立した根拠で成立しており、
> 実際に融合カーネルが 20.9 -> 4.51ms にしたことで裏付けられた。
> 詳細は docs/STATUS.md の同日の訂正節。

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

- `~/models/qwen38fn-mlx-v-fast6` + `~/models/qwen38fn-ngram-4bit`
  (現在の最良。91GB / 6.206 bpw、KLD 0.00378、融合ありで 29.5 tok/s)

**`~/models/qwen38fn-mlx-v-ng2` と `~/models/qwen38fn-mlx-v-stream` は削除した**
(2026-08-27 と 08-28。内蔵の空きが足りず、焼き 1 本ぶんも置けなかった)。
v-fast6 が両者の上位互換 (v-stream 比で -2.47ms、KLD は 0.00260 -> 0.00378)。
過去の数字と比べるときは、v-stream 基準の値がそのままでは使えない点に注意。

`--ngram` を付け忘れると n-gram の重みが無くて読み込みが落ちる。

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
| gather_qmm 3 本の起動そのもの | ~2.5ms (下の訂正を読むこと) |

**MoE の現実的な天井は 3.4ms 前後**、GDN は固定費 2.2ms が丸ごと天井。

> ### 訂正 (2026-08-28): gather_qmm 自体に手を入れる余地がある
>
> 上の表は **gather_qmm 3 本の呼び出しが最適に近い**という前提で「取れない」と
> 書いた。その前提が怪しい。
>
> 別実装 (ddalcu/mlx-serve、MIT、Zig) の `moeDecodeGatherQmv` のコメントに、
> デコード時 (B*S=1) の専用 gather-qmv カーネルについてこう書いてある:
>
> > 「GPU 常駐の添字でエキスパートバンクを直接読むので、take 経路の 3 倍では
> > なく理想の 9.8 MB/投影 を動かす。201MB バンクでのマイクロベンチ:
> > **37us (専用) 対 72us (batched take) 対 349us (stock gather_qmm)**」
>
> 数字はあちらの環境・あちらのモデルのもので、こちらでは未検証。ただし
> **こちらの独立した計測とも整合する**: `tools/micro_moe_gdn.py` の
> ビット掃引を直線に当てると、バイト数ゼロに外挿した切片が 48 層で 6.7ms
> (= gather_qmm 1 回あたり 46us) 残る。この切片はバイト数で説明できない量で、
> 起動費用 (15-20us) だけでも説明しきれない。
>
> **つまり「gather_qmm は触れない」は根拠が弱い。**着手前に、こちらの環境で
> stock の `mx.gather_qmm` が実際に何バイト動かしているかを測ること
> (デコード形状 x=[1,1,2560]、rhs_indices=10 個、bank=[512,640,2560])。
> 理想の 3 倍動いているなら、ルータ頭や合成を畳むより先にそこを直す方が大きい。
>
> あちらは MIT だが、**コードを写さないこと。**上のコメントは「どこに余地が
> あるか」の手がかりであって、実装はこちらで書く。
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

**「残り」の中身は分解済み** (`tools/rest_profile.py`、2026-08-28):
lm_head 4.89ms + full attention 2.53ms + 残差 0.30ms + 端数。未知の起動コストでは
なかった。

### lm_head に専用カーネルが要る (2026-08-28)

**lm_head は単体で測ると 6bit で 1.87ms なのに、モデルの中では 4.89ms かかる。**
実効帯域にすると 104 GB/s で、単体の 256 GB/s の半分以下。

同じ場所を別実装も踏んでいて、そちらは**巨大 N 専用の行列積カーネル**を書いて
いる。理由が書いてある:

> 「巨大な N (lm_head の類) では、細かいタイルの split-K グリッドが
> スケジューラを取り合って潰し合う」

こちらの語彙は 248320。あちらのコードにも `lm_head_n == 248320` の分岐があり、
**同じモデルを見て同じ問題に当たっている**。

対策の方向は「NSG 本の独立したアキュムレータを持つ幅広の multi-simdgroup
タイル」。MLX 純正のタイル分割が巨大 N に向いていないという話なので、
MoE の後の標的として GDN 2.2ms より優先度が高い (伸びしろ 3ms 前後)。

## 一般化

GLM-5.3-Flash も **mHC (multi-head hyper-connections)** を持つ。ここで書く
カーネルは構造がほぼそのまま移る見込み (docs/STATUS.md の GLM 節)。
形状を引数で受ける作りにしておくと後が楽。
