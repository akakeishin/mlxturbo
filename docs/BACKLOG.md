# BACKLOG — やりたいが未着手のもの (2026-08-28)

速度の決着は [[docs/KERNEL-BRIEF-MOE-GDN.md]] と直近のコミットで一区切り。ここには「やる価値はあるが手を付けていない」ものを、着手前に分かっている根拠つきで置く。

## 1. マルチモーダル対応 (画像 → 音声・動画)

元 checkpoint は VLM (`Qwen4ExpForConditionalGeneration`)。1,658 キー中 **333 が vision 系** (`model.visual.blocks.*`) だが、変換で意図的に捨てている。変換後の v-fast6 に vision 系は **0 キー**。

落としている箇所:
- `fastmlx/convert_flash.py:331` — `mtp.` / `vision_tower.` / `model.visual.` を skip
- `tools/vendor/qwen4_exp.py:849` — mlx-lm 側の `sanitize` でも同じものを skip (`# text-only pour l'instant`)

必要な作業は 3 段。

1. 変換で `model.visual.*` を残す (`convert_flash.py:331`)
2. **mlx-lm 側に vision タワーのクラスを書く。** 現状 `qwen4_exp.py` に vision/visual の言及は skip の 2 行だけで、クラス自体が存在しない。**ここが一番重い**
3. `SpecEngine` が埋め込み入力を受け取れるようにする。`fastmlx/spec.py:164` が `self.inner.embed_tokens(tokens[None])` と自前で埋め込んでいるのを、外から渡せるようにする。モデルの `__call__` は既に `input_embeddings` を受ける口を持っている (`qwen4_exp.py:746`) ので、そこは流用できる

投機デコード固有の注意点として、**n-gram lookup はトークン ID 上で動くので、画像プレースホルダ (同じ ID が数百個並ぶ) の区間で誤マッチを量産する。** lookup の対象から画像区間を外す処理が要る。MTP 側はドラフトが常にテキストなので影響しない。

音声・動画は扱う場所自体が無い。ViT のぶん重みも増えるので、91GB がさらに膨らむ点も込みで判断すること。

## 2. MTP の価値を決着させる

現状の 30.61 tok/s は **MTP 無し**。そして今の構成では、有効にしても遅くなる公算が高い。

`bench/results/mtp-cost.json` の実測から損益分岐を引くと、verify は m1 が 51.85ms、1 トークン増えるごとに約 +15.2ms、draft 1 トークンが 4.72ms。

| | 旧 base (51.85ms) | 現 base (約 32.7ms) |
|---|---|---|
| n_draft=3 の分岐点 | 平均受理 1.17 | 平均受理 **約 2.8** |
| n_draft=1 の分岐点 | 0.42 | 約 0.61 |

実測受理は `bench/results/mtp-diag-d0.json` で n_draft=3 のとき平均 1.38 (深度別の連鎖受理 0.69 / 0.43 / 0.26)。旧 base では得だったが、**現 base では 2.8 に届かず概算 1.2 倍遅くなる**。n_draft=1 なら深度1の 0.69 が分岐点 0.61 を上回るので 5% 前後の得は残る。

自分で base を速くしたぶん、投機が回収すべき固定費が減って損益分岐が上がった、という構図。

着手するなら順番は:

1. **Flash-Next の MTP 重みを再取得** (28/131 シャードに `mtp.*` が散っている。現在スナップショットはメタデータのみで実体ゼロ)
2. Flash-Next での深度別受理率を測る (`spec.py` の accept_trace)。上の数字は全て **Qwen3.8-27B** のもので、Flash-Next では未測定
3. verify コストを現構成で測り直す (上の値は最適化前)
4. n_draft=1 に絞って損益を判定

見込みの取り分が 5% 前後なので、数十 GB の再取得に見合うかは微妙。

## 3. サーバーの並列化 (継続バッチング)

直列サーバーは実装済み。並列化の取り分は MoE の算数から 2〜3 倍ある (512 experts に 10 routed なので、B=8 で 1 トークンあたりの読み出しが 4.7GB → 約 2.0GB)。

ただし着手前に確かめること:

- **prefill が許容範囲か。** README の実測で 219 tok/s。1 万トークンの文脈なら最初のトークンまで 45 秒で、バッチ化しても縮まない。ここが辛さの本体なら並列化は的外れ
- **並列リクエストが実際に発生しているか。** 1 人が 1 問ずつ待つ使い方ならバッチ化はレイテンシを悪化させるだけ。効くのはエージェントが並列にサブエージェントを投げるとき
- メモリ。91GB 常駐に KV を 1 本あたり数百 MB (8K 文脈で約 450MB) 積むので、同時 4〜8 本が現実的な上限

継続バッチングを投機デコードと GDN の再帰状態の上に載せるのは片手間の規模ではない。

## 4. 他モデル対応 (gemma / kimi / glm)

サーバーには「`validate_spec_model_contract` が通らなければ通常生成にフォールバック」を入れてあるので、**載せれば喋る**状態にはなる。ただし投機デコードは効かない。

`SpecEngine` は `fastmlx/_mlx_compat.py:111` で GDN ハイブリッド固有の構造 (`fa_idx` / `ssm_idx` / 層ごとの `is_linear` / linear cache の `advance`) を要求するため、他アーキテクチャで投機を効かせるには contract の一般化が要る。lookup (SAM) 側はモデル非依存なので、そちらだけ先に切り出す手はある。

この Mac では VRAM が先に効くので、載せられるサイズが実質的な制約になる。
