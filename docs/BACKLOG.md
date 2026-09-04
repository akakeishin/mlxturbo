# BACKLOG — やりたいが未着手のもの (2026-08-29 更新)

ここには「やる価値はあるが手を付けていない」ものを、着手前に分かっている根拠つきで置く。決着したものは末尾の「済み」へ移した。

## 1. マルチモーダル対応 (画像 → 音声・動画)

元 checkpoint は VLM (`Qwen4ExpForConditionalGeneration`)。1,658 キー中 **333 が vision 系** (`model.visual.blocks.*`) だが、変換で意図的に捨てている。変換後の v 系成果物に vision 系は **0 キー**。元 checkpoint (360GB / 131 シャード) は外付け SSD 配下の `models/Qwen3.8-Flash-Next` に再取得済みなので、素材はいつでも読める。

落としている箇所:
- `mlxturbo/convert_flash.py:331` — `mtp.` / `vision_tower.` / `model.visual.` を skip
- `mlxturbo/_vendor/qwen4_exp.py:849` — mlx-lm 側の `sanitize` でも同じものを skip (`# text-only pour l'instant`)

必要な作業は 3 段。

1. 変換で `model.visual.*` を残す (`convert_flash.py:331`)
2. **mlx-lm 側に vision タワーのクラスを書く。** 現状 `qwen4_exp.py` に vision/visual の言及は skip の 2 行だけで、クラス自体が存在しない。**ここが一番重い**
3. エンジンが埋め込み入力を受け取れるようにする。`mlxturbo/spec.py:164` が `self.inner.embed_tokens(tokens[None])` と自前で埋め込んでいるのを、外から渡せるようにする。モデルの `__call__` は既に `input_embeddings` を受ける口を持っている (`qwen4_exp.py:746`)。`spec_flash.py` 側も同様の口が要る

投機デコード固有の注意点として、**n-gram lookup はトークン ID 上で動くので、画像プレースホルダ (同じ ID が数百個並ぶ) の区間で誤マッチを量産する。** lookup の対象から画像区間を外す処理が要る。MTP 側はドラフトが常にテキストなので影響しない。

音声・動画は扱う場所自体が無い。ViT のぶん重みも増えるので、91GB がさらに膨らむ点も込みで判断すること。

## 2. サーバーの並列化 (継続バッチング) — 実装・配線・実モデル検証済み

**測定・実装・配線が済んでいる。`--max-batch N` で有効化できる (既定 1 = 直列、README.md 参照)。**

- 取り分の実測: gemma4-26B で B=4 実測 2.10x。Flash-Next はエキスパート和集合からの予測で 2.05〜2.19x (`tools/observe_flashnext_batch.py` の和集合モードで再現できる)。「512 エキスパートなら飽和しにくい」は誤りで、飽和は (B × top_k) / num_experts で決まる
- prefill には一切効かない (B に線形)。cold 32k は TTFT 112 秒で壁時計の 84% を prefill が占める (同スクリプトの prefill/decode モードで再現できる)。**体感を狙うなら prefill 側が先**
- 実装: `mlxturbo/batch.py` に `enable_batch_cache()`。`mlxturbo/runner.py` の `maybe_build_batch_coordinator()` が `--max-batch` から配線する (投機経路 spec/flash_spec は対象外、`FallbackRunner` に載るリクエストだけがまとめられる)。合成モデルの CPU 検証で QSA off と長さ揃いは B=1/2/4 完全一致。**残る制限**: QSA 有効 (2048 超) かつ長さ不揃いでバッチ出力が単体と一致しない (破損ではなく QSA のブロックグリッドが絶対列で切られるため。パディングを揃えれば 3.7e-8)。実モデル検証は `tools/verify_batch_real.py --mode kld` で実施済み: bit-exact は保証しない設計 (`mlxturbo/batch.py:705-712` 参照。プレフィルチャンク幅と同様、`mx.quantized_matmul` はバッチ長総数で丸めが変わる MLX の性質で、vLLM/llama.cpp も同じ非保証) だが、next-token KLD はこのプロジェクトの量子化ノイズ床 (v-fast6: 0.00378) と同オーダーに収まる。唯一 QSA 選択差が効く `long-uneq, B=4` だけノイズ床の最大 ~4.4x まで上がるため、QSA が発火しうるリクエスト (`classify()` の "solo" tier) は常に単独実行に倒し、バッチを共有させない設計になっている
- やる根拠は速度でなく**可用性**: 直列サーバーだとサブエージェントを流しっぱなしにしたまま別の作業ができない。B=4 でレイテンシは 0.53x に悪化するので、対話 1 本の用途では逆効果

## 3. 他モデル対応で投機を効かせる (一部済み)

契約に合わないモデルでも「載せれば喋る」状態は元からある (fallback、理由は `/health` の
`fallback_reason` に出る)。2026-08-30 に、投機を効かせる汎用経路を 2 つ足した。どちらも
既定 off。

`--draft-model` (`DraftSpecRunner`) は mlx_lm 自身の `speculative_generate_step` を包む
だけなので、アーキテクチャに依存しない。ただし**適合する小型ドラフトが要る**。手元に
無く、self-draft では 367 → 248 tok/s と当然ながら悪化した。真の高速化ケースは未測定で、
ここが残っている最大の穴。

`--lookup-spec` (`LookupSpecRunner`) は n-gram lookup だけを使う。trim 可能なキャッシュ
かつ貪欲限定。繰り返しの多い入力で 361 → 427 tok/s (+18%) だが、**繰り返しの無い自然文
では 367 → 245 tok/s (-32%) と遅くなる**。自前ループが mlx_lm の `generate_step` が持つ
`async_eval` 二重バッファを持たないため。ここを埋めれば常時 on にできる可能性がある。

残るのは、`spec` / `flash_spec` 級の速度 (1.25-1.39x) を他アーキテクチャで出すこと。
それには MTP 相当のドラフトヘッドか、アーキテクチャ固有の状態捕獲が要る。

## 4. `n` (複数候補)

直列のままだと生成時間が候補数倍になるので §2 のバッチ化が前提。同一プロンプトの n 候補は
§2 のバッチにそのまま載る。

`logprobs` は 2026-08-30 に fallback / 降格経路限定で実装した。投機経路で要求されたら
自動で非投機に降格する。ストリーミングとの併用は 400 で断っている — ThinkingRouter や
tool_calls が絡むとトークン列と content の対応が崩れ、正しく対応付ける実装が要るため。
そこを埋めるのは残件。

## 5. prefill の高速化 (体感の本丸)

prefill は 290〜325 tok/s しか出ない (デコードの約 11 倍でしかない)。Claude Code は毎ターン 54k 超を送るので、初回 TTFT を支配する。prompt cache の再利用 (fadadf7) で 2 ターン目以降は 9 割超を飛ばせるようになったが、**初回とキャッシュミス時はまるごと食らう**。バッチ化では縮まない (B に線形)。カーネル側の課題として未着手。

## 6. LoRA アダプタ対応 (`adapter_path`)

**mlx_lm 側の受け口は既に揃っている。fastmlx/mlxturbo 側は未配線**。以下は調査のみで、実装はしていない (`runner.py` / `server.py` が別レーンで編集中のため)。

mlx_lm の受け口:

`mlx_lm/utils.py` の `load(path_or_hf_repo, tokenizer_config=None, model_config=None, adapter_path=None, lazy=False, return_config=False, revision=None)` (453-502行) は `adapter_path` をキーワード引数として直接受け取り、非 None なら 492-494行で `model = load_adapters(model, adapter_path); model.eval()` を呼ぶ。`load_adapters` (utils.py 423-426行) は `mlx_lm/tuner/utils.py` の実体 (113-138行) へのラッパーで、`adapter_path/adapter_config.json` を読んで `fine_tune_type` (既定 `"lora"`) を見つつ `linear_to_lora_layers()` (同ファイル 38-110行) でモデルの `Linear`/`Embedding`/`SwitchLinear` を `LoRALinear` 系へ差し替え、その後 `model.load_weights(adapter_path/adapters.safetensors, strict=False)` で重みを流し込む。ディレクトリ構成は `adapter_config.json` + `adapters.safetensors` の組で固定。`mlx_lm/generate.py` の CLI は `--adapter-path` フラグ (81-83行) からそのまま `load(model_path, adapter_path=args.adapter_path, ...)` (2008-2012行) へ渡すだけ。

fastmlx/mlxturbo 側の現状:

`mlxturbo/_mlx_compat.py:90-93` の契約チェック (`_require_signature`) が `mlx_lm_load` のシグネチャに `adapter_path` が含まれることを検査しているだけで、実際に値を渡している呼び出しはリポジトリ内のどこにも無い。`mlxturbo/cli.py:122`、`mlxturbo/server.py:101,4792` の `mlx_lm_load(args.model, return_config=True)` はいずれも `adapter_path` 未指定。`grep -rn -i "adapter\|lora" mlxturbo/*.py` でヒットするのはこの契約チェックの1箇所と、無関係な英語コメント2箇所 (`fast_qmm.py:377` の「adapters」は量子化レイヤー再利用の話、`server.py:1962` の「adapter」は一般名詞) のみ。`runner.py` (`build_runner` 以下) にも adapter_path/LoRA を受け取る引数・分岐は無い。

配線先と見積もり:

`--adapter-path` フラグ自体は `cli.py`/`server.py` どちらの argparse にも追加できる (前者は編集可能、後者は今回のレーン境界で触れない)。値を実際に効かせるには、`mlx_lm_load(args.model, adapter_path=..., return_config=True)` という形で呼び出しに1引数足すだけで済む — この呼び出しは `build_runner` より前、まだモデルをロードしている段階 (`cli.py:122` / `server.py:101,4792`) にあるので、`build_runner` 自体のシグネチャを変える必要は無い。ただし `server.py` 側は今回のレーン境界外なので着手できない。`cli.py` 側だけなら技術的には可能だが、対話 CLI だけ先に対応してもサーバー経由の運用 (本来の主用途) には効かないため、今回は見送って調査止まりとした。

分岐点として、**実行時に `adapter_path` を渡すだけ (LoRALinear のまま使う) か、事前に `mlx_lm` 付属の `fuse.py` でオフライン fuse 済みのモデルディレクトリを用意してそれを読ませるか**で影響範囲が変わる。前者は `SpecEngine`/`FlashSpecEngine` が構築される前にモデルの `named_modules` 構造が `<parent>.linear.weight` 型に変わる (`LoRALinear.from_base()` が元の `Linear` を `self.linear` として内包する) ため、`_mlx_compat.py` の契約チェックや `spec_flash.py` の MTP サイドカーのテンソル名突合に影響しないか未検証。後者 (`mlx_lm.tuner.fuse`) は `LoRALinear.fuse()` でマージ後の平のテンソルに戻してから保存するため (`fuse.py:60-75`)、テンソル名・形状は元のモデルと同じに戻り、既存の契約チェック・MTP 探索への影響はほぼ無いはず。着手するなら後者 (オフライン fuse → 通常のモデルとして読み込む) の方が安全側で、見積もりも軽い (`cli.py`/`server.py` へのフラグ追加 + 呼び出し1箇所の引数追加のみ)。前者 (実行時 LoRA、fuse 無し) は `_mlx_compat.py` の契約チェックと `spec_flash.py` 側の名前突合を実地で確認する一手間が追加で要る。

---

## 2026-09-01 に作りながら分かった制限 (直す前提で残す)

カーネル 4 本とサーバー配線を一気に入れた日の記録。**どれも「知らずに踏む」
種類のもの**なので、直す・直さないの判断より先に、存在を残す。

### A. バッチ x 投機のサーバー配線 (`--max-batch-spec`、既定 off、`df90d2c`)

1. **2048 トークンまでしかまとめられない。**`indexer_budget` が硬い上限で、
   `prompt + max_tokens` がそれを超える要求は従来どおり直列。
   **実運用のエージェント文脈 (17k 級) はこの機構の恩恵を一切受けない。**
   直すには QSA のブロック境界を dead slot 込みで行ごとに組み直す必要がある
   (`mlxturbo/batch_spec.py` の既知の未対応)。**ここが本丸。**
2. **バッチ経路ではセッションのプレフィックス再利用が効かない。**短い会話の
   2 ターン目以降の TTFT が悪化しうる。**decode には出ないので
   `bench/batch_b1_gate.py` には映らない。**有効化の判断はこれ込みで。
3. **走行中のバッチに後から入れない (closed batch)。**次の編成は現バッチが
   空になった時点。直列キューよりは悪くないが、連続バッチングではない。
   途中参加には新入りの KV を走行中の物理列数へ左詰めで揃え、`ArraysCache`
   の内部オフセットと MTP priming 窓まで整合させる必要がある (実機検証が要る)。
4. **バッチ経路の temp>0 の分布が未確認。**`_sample_rows` は
   `FlashSpecEngine._verify` の (1,T+1) を (B,T+1) に広げただけで議論は
   通るが、KLD で確かめていない。実クライアントは temp 0.7 が既定なので、
   有効化するならここは測る。
5. **「同時要求が直列に待たされないこと」を測る道具が無い。**
   `bench/batch_spec_throughput.py` は `BatchSpecGenerator` を直接叩く
   ライブラリ計測で、サーバー越しではない。判定基準そのものを測れていない。

### B. 一次検査の穴

6. **`tools/vendor_fingerprint.py` は CPU で走るので、2026-09-01 に足した
   `eligible()` が常に False になる。**Metal を要求する分岐を、唯一の一次検査が
   **一度も実行していない。**緑が出ても新しい分岐については何も保証しない。
7. **GDN の分岐が 2 箇所にある** (`_vendor/qwen4_exp.py` の
   `GatedDeltaNet.__call__` と `spec_flash.py` の `capture()` 内 `gdn`)。
   条件は今日時点で一字一句同じだが、**共有の述語も回帰ゲートも無い。**
   ずれたときの症状は「draft と verify で数値系統が食い違い受理率が動く」で、
   バグとして表面化しない。`gdn_prework.wants(module, mask, cache)` のような
   述語を kernels 側に置いて両方から呼ぶのが最小の直し。
8. **`eligible()` が無言。**約 20 の条件が黙って False を返す。発火カウンタ
   (`mlxturbo/kernels/_fire.py`、`75bdd9d`) で空振りは見えるようになったが、
   **なぜ落ちたかは出ない。**`prefill_attn` の `_warn_once` 方式を横展開する。

### C. カーネル個別

9. **GDN 前処理が `_store_conv_state` シームを迂回して `cache[0]` に直接書く。**
   今日は等価 (オーバーライドが存在せず、`lengths is None` なら `_tail_window`
   は末尾切りと同一)。**反転条件: lengths を持たないキャッシュ型に
   `_store_conv_state` のオーバーライドが生えたとき。**上の 7 の `wants()` に
   この前提を書き残すこと。
10. **`prefill_attn` に S の下限が無い。**decode 幅でも比のゲートを通れば
    入るので、`--knob prefill-attn` は prefill だけの knob ではない。
    A/B は `prefill_s` と `ms_per_tok` を別々に読むこと。
11. **`prefill_attn` がキャッシュの内部表現に依存する** (確保済みバッファと
    確保幅を直接受け取る)。連続性 (strides) を見ていないので、**keys が
    ビューや遅延 concat になるキャッシュ型が入ると、`ensure_row_contiguous`
    が CAP 幅を毎回コピーする性能罠に静かに落ちる。**
12. **`_segments_gpu` が 1 MoE 層あたり 2 回走る。**`fused.py` が
    `gather_gate_up` と `gather_down` に同じ添字を渡すのに、それぞれが内部で
    呼び直す。中身は 15 本前後の直列小カーネルで、それを倍払っている。
    セグメント 3 本組を一度作って両方へ渡すだけで消える。
13. **`_max_seg_bound` が S だけで弾くようになった** (旧版は実データの最長ラン
    を見ていた)。現行は `indices.size < 64` のゲートが先に効いて到達しないが、
    **top_k か verify 幅を変えると効いてくる。**
14. **`enable_hc_write` だけ env ゲートが関数内に無い。**兄弟 2 関数と流儀が
    不揃いで、直接呼ぶと無条件でパッチが入る。

### D. 計測の道具

15. **`tools/` の GDN 前処理テストの閾値が絶対値 (0.001)。**q も k も bf16 の
    1 ulp ずれるが、k の値が約 11 倍大きいので **k だけ引っかかって見える。**
    「q は一致、k だけおかしい」という誤読を 1 回した。相対誤差で見ること。
16. **A/B で `--prefill-once` を使い忘れると、17k の prefill を 12 回やり直す**
    (1 本あたり 7.6 分、実行時間の 37%)。decode の knob では必ず付ける。

## n-gram サイドカーのレイアウト選択が 50k を殺していた (2026-09-02)

**症状**: 50k のプロンプトが `[METAL] Command buffer execution failed:
Insufficient Memory` で落ちる。サーバーは 200 を返してストリームを開き、
31 秒後にエラーを流す (黙って固まりはしない)。17k までは通る。

**原因**: サイドカーの `manifest.json` の `layout` で経路が決まる
(`mlxturbo/ngram_stream.py:273` の `install`)。

| layout | 経路 | RAM | 50k |
|---|---|---|---|
| `separate` (`~/models/ddalcu-ngram-sep`) | RAM 常駐 (`RamNGram`) | **32GB** | **落ちる** |
| `interleaved` (`~/models/ddalcu-ngram`) | ディスク参照 (`StreamNGram`, pread) | **0** | **通る (実測)** |

モデル 91GB + n-gram 32GB = 123GB / 128GB。残り 5GB に 50k の KV と活性が
入らない。**性能ではなく構成の問題。**

**記録の訂正**: 以前「`layout=separate` の RAM 常駐は設計どおりで設定ミスでは
ない」と結論した。設計どおりなのは正しいが、**「だから問題ない」は誤り。**
128GB の機体では長文脈が通らなくなる。

**未決**: どちらを既定にするか。`separate` は gather 1 回で速いが 50k を殺す。
`interleaved` は RAM 0 だが行ごとに pread する。**decode の差は未測。**
モデルの宣言は 262144 トークンなので、50k で落ちる構成を推奨とは呼びにくい。

**やること**:
1. decode の差を測る (17k で `separate` と `interleaved`)。プロセスをまたぐ
   比較になるので、差が数 % なら判定しない
2. 差が小さいなら `interleaved` を既定にする
3. 差が大きいなら、**要求の長さで選ぶか、起動時に「この構成では約 N トークン
   まで」と警告する。**31 秒使ってから落ちるより先に言う方が親切

## 訂正: バッチは割に合う。噛まなかった原因は n-gram の layout だった (2026-09-02)

**下の節「バッチは実運用の形では割に合わない」は誤り。**噛めば勝つ。

### 噛まなかった原因

**`ddalcu-ngram-sep` (layout=separate、RAM 32GB 常駐) だと `rows_fit` が通らず、
`_admit_next` が落ちて全要求が単独に倒れる。**`ddalcu-ngram` (interleaved、
RAM 0) なら毎回噛む。**`--max-batch-spec` の実効性が n-gram サイドカーの
layout に依存する。**これは記録に無かった論点。

### 噛めば勝つ

1880 トークン x 2 本、512 生成、1 プロセス内:

| | 壁時計 |
|---|---|
| バッチ | **25.05s** |
| 直列 | 31.74s |
| | **-21%** |

閉じたバッチ (一括 prefill、1 ラウンド目から全行そろう) でも
1880x2 で -3.3%、3962x2 で -3.1% (256 生成) / -4.1% (512 生成)。

### 下の節で「壁時計が変わらない」と書いた測定の誤り

`wait_ms=1000` で噛ませたが、**その 1 秒の待ちが取り分を相殺していた**
(14.0 対 13.8s)。担当がコーディネータを直接叩いた測定には待ちが乗らないので
-21% が出る。**機構は効いていて、HTTP 経路の相方待ちが食っている。**

### したがって残る課題

1. **n-gram を interleaved にする** (既に推奨済み。`separate` だと噛まない)
2. **相方待ちの設計。**既定 15ms では会えず、1000ms では取り分を食う。
   到着を待つのではなく、**走行中のバッチに後から入れる** (chunked prefill で
   実装済み) 方に寄せるべき
3. 文脈長連動 depth は移したが、**実運用の窓では効果が雑音に埋もれた**
   (1880x2 で +0.7%)。閉じたバッチでは -3.1〜-4.1% で機構は効いている

## A-1 は外れた。**しかしバッチは実運用の形では割に合わない** (2026-09-02)

A-1 (QSA の ragged 対応) で長さの上限は消えた (`5ba1c84`)。**そのうえで測ったら、
バッチは実運用の形では取り分がゼロだった。**

### 発火の確認 (1815/1821 トークンのプロンプトを 2 本同時)

| `wait_ms` | `/health` の発火 | 壁時計 | 要求ごとの decode |
|---|---|---|---|
| **15 (既定)** | `joins: 0` | 14.0s | 50.9 / 54.6 |
| 200 | `joins: 0` | 13.8s | 54.0 / 57.3 |
| **1000** | **`joins: 1`** | **14.0s** | **39.1 / 18.5** |

**既定の 15ms では相方に会えない。**200ms でも会えない。**1 秒待って初めて噛む。**
スレッド起動 + HTTP + 1800 トークンのトークナイズで、2 本目が届くまでに
15ms は軽く超える。

**そして噛んでも壁時計が変わらない** (14.0 対 13.8s)。要求ごとの decode は
大幅に悪化 (39.1/18.5 対 54.0/57.3)。

### なぜ割に合わないか

**バッチ経路には楽観パイプラインも次ラウンド draft の先行投入も適応 depth も
乗っていない** (単独経路にはある)。加えて A-1 で入れた論理順 gather が
17k で ~17MB/層/ラウンド。**長文脈ではラウンドあたりの費用が効き、2 行では
取り返せない。**

短いプロンプト x B=4 では **+19%** (43.0 → 51.4 tok/s) が出ている。
**取り分があるのは「短いプロンプト x 多い同時数」だけ。**

### したがって

- **`--max-batch-spec` は既定 off のまま。**上げる根拠が無い
- **既定 `wait_ms=15` は実質「発火しない」設定。**上げれば発火するが、
  発火しても得しないので上げる意味も無い
- **バッチ経路に単独経路の最適化 (パイプライン・先行 draft・適応 depth) を
  移すのが先。**それをやらずに同時実行の比較をしても、直列より遅い側を測る
- 対戦比較の N=4 が両エンジンとも崩壊する領域なのは前回と同じ

**「相手のカーネルは `B*S==1` でしか効かないので同時実行はうちの土俵」という
仮説は、まだ検証できていない。**土俵に立てていない。

## 同時実行の比較は A-1 が外れるまで測れない (2026-09-02、実測で判明)

`bench/vs_mlx_serve_load.py --turbo-max-batch-spec 8 --ns 1,2,4 --tokens 128`
(`bench/results/vs-mlx-serve-load.json`):

| N | スループット serve / turbo | TTFT 中央 serve / turbo | decode 中央 serve / turbo |
|---|---|---|---|
| 1 | 22.71 / **17.70** | 3.22 / **4.50** | 52.16 / 49.71 |
| 2 | 22.97 / **17.47** | 6.06 / **8.28** | 53.25 / 50.71 |
| 4 | **3.42 / 2.74** | **79.23 / 102.63** | 47.57 / 43.67 |

**N=4 は両エンジンとも崩壊している** (相手も 22.7 → 3.4)。同時実行の性能を
測っている領域ではなく、詰まっている領域。

**そしてうちのバッチは一度も発火していない。**ハーネスのプロンプトプールは
ctx 2000/8000/24000 で、`spec_batchable` の条件は
`prompt + max_tokens + depth+1 <= indexer_budget (2048)`。生成 128 トークンだと
**一番短い 2000 トークンのプロンプトでも 2000+128+2 = 2130 > 2048。**
**プール内のどれも条件を通らない。**うちは全部直列で走っている。

**「同時実行はうちの非対称」という仮説は、A-1 が外れるまで検証できない。**

B=1 ゲートの文脈で測った **同時 4 本 +19%** (43.0 → 51.4 tok/s) は、
ゲートの**短いプロンプト**でのもの。**そこでは発火する。実運用の長さでは
発火しない。**

**したがってバッチのレーンの最優先は A-1 (QSA のブロック境界を dead slot 込みで
行ごとに組み直す) になった。**chunked prefill は動いたが、**それが効く範囲に
実際の要求が入ってこない。**

## バッチングの設計で参考になるものが見つかった (2026-09-02)

`Layr-Labs/mlxfast-gemma4-26b-a4b-engine` (MIT、ベンダリング部も MIT) の
`ContinuousBatchingV2` は vLLM-V1 方式で、**うちの batch_spec に無いものを
全部持っている。**

| | うち (`mlxturbo/batch_spec.py`) | 向こう |
|---|---|---|
| バッチの形 | **閉じたバッチ** (現バッチが空くまで待たせる) | **走行中に後から join できる** |
| prefill と decode | **別フェーズ** | **フェーズ分割なし。1 ステップ 1 トークン予算** |
| 長さのばらつき | 比 1.5 倍以内でまとめる | **chunked prefill で吸収** (512 トークンずつ) |
| KV 不足 | — | **preemption** (生成済みトークンを保持して退避・キュー先頭へ再投入) |
| 上限 | **prompt + max_tokens <= 2048** | `maxBatchedTokensPerStep` 2048、`prefillChunkSize` 512 |

**設計の肝: chunked prefill が prefill と decode を 1 つの予算に統一している。**
だから走行中に join できる。新しい要求の prefill を刻んで、走っている decode と
同じステップに混ぜる。

**うちは prefill を「まとめて 1 回」でやっているので、閉じたバッチにするしか
なかった。**これが A-3 (途中参加できない) の原因で、**chunked prefill で解ける。**

**訂正 (2026-09-02)**: 一度「A-1 (2048 上限) と A-3 は共通の原因」と書いたが
**誤り。**A-1 は QSA 固有の別問題で、`prompt + max_tokens` が
`indexer_budget` (2048) を超えると **QSA が発火して `ragged_attention` が
`NotImplementedError` で止まる**ため。prefill を刻んでも KV は伸びるので
解けない。**A-1 は QSA のブロック境界を dead slot 込みで行ごとに組み直す
別の仕事。**混同すると「chunked prefill を入れたのに 17k が通らない」で
つまずく。

投機との同居も参考になる: MTP は decode 準備ができた行 (remaining==1) にだけ
適用して `n=1+k`、予算が足りなければ素の decode に落ちる (小さい予算を食わない)。
矩形の上限は `B*(1+k) <= 8`。

### カーネル側 (取り込む価値は低いが、傍証として重要)

本家 MLX v0.32.0 に対し 12 ファイルに手が入っている (素の MLX ではない。
Layr Labs 自社エンジン "DARKBLOOM" からの移植)。目を引くのは 2 つ:

- `qdot_affine4_registered`: 重みをレジスタに常駐させ、**1 回のフェッチで
  複数行を処理**する 4bit dot
- NAX GEMM の行タイル分割を 2x2 から 4 本の行ストリップに変更。理由が
  **「A オペランドの二重フェッチを削減」**

**後者は fable の見立て (「勘定されていないバイト移動」= 活性のトラフィック) と
同じ方向。**独立したチームが同種の仕事を最適化して活性の再フェッチを潰しに
行っている、という傍証になる。

ただし**うちの律速はカーネルではない** (`gather_qmm` が実形状で天井の 96%)
ので、移植しても数字は動かない見込み。**取るなら設計、カーネルではない。**

権利: 参加規約に提出コードの権利帰属も勝った差分の公開も**記載なし**。

## Gemma 4 対応の下調べ (2026-09-02、読解のみ・未実行)

### mlxturbo は無改造でどこまで動くか

| | 状態 | 根拠 |
|---|---|---|
| `mlx_lm` の gemma4 | **完備** | `gemma4.py` (VLM ラッパ) + `gemma4_text.py`。MoE は `enable_moe_block`、`Router` (117-143) + `Experts`/`SwitchGLU` (153-173)。GQA + sliding window (4:1)、KV は標準の `KVCache`/`RotatingKVCache` |
| ランナー選択 | **`FallbackRunner` に落ちる** | `runner.py:1649` の `model_type == "qwen4_exp"` に入らず、`SpecEngine` の契約検証 (`_mlx_compat.py:120-176`) が qwen3_5 の GDN 専用属性を要求して失敗 → `runner.py:1785-1791` の except で捕捉。**コード読解による推論で、実行では未確認** |
| 投機 | **無い** | `FallbackRunner` は `stream_generate` を直接呼ぶだけ (`runner.py:540-604`)。`LookupSpecRunner` / `DraftSpecRunner` への配線を書く必要がある |
| `--max-batch` | **効く** | `can_batch` (`runner.py:761-764`) が `FallbackRunner` 限定。`batch.py` のパッチは qwen4_exp のキャッシュ専用なので Gemma4 は対象外 (パッチが要らない側) |
| `server.py` のアーキ分岐 | **実行時はゼロ** | `qwen4_exp`/`qwen3_5` のヒット 6 箇所は全部コメント・ヘルプ・ログ |
| `bench/quant_eval.py` | **移る** | docstring に「モデル非依存」、Flash-Next 固有の前提は無い |
| `tools/bake.py` / `convert_flash.py` / `convert.py` | **移らない** | 前者 2 つは qwen4_exp のテンソルパス、後者は 27B (qwen3_5) 専用 |

**未確認**: Gemma4 の `previous_kvs` (KV 共有層) が `BatchGenerator` の
merge/filter/extract と整合するか。26B-A4B-it は `num_kv_shared_layers: 0`
なので当面は影響しない見込みだが未検証。

### モデルの実態 (`google/gemma-4-26B-A4B-it` の config)

総 25.2B / アクティブ 3.8B、30 層、**128 専門家 top-8 + 共有 dense 1**、
hidden 2816、**moe_intermediate 704**、head_dim 256 (full attention 層は
global_head_dim 512)、sliding_window 1024、文脈 256K、vocab 262144。
MLX 変換済みが `mlx-community/gemma-4-26b-a4b-it-{4bit..bf16, mxfp4, nvfp4}`
と QAT 4bit まで既にある。

**`moe_intermediate = 704` も 512 の倍数ではない。**うちの 640 と同じ形だが、
2026-09-02 の実測で「K による差は無い」と決着済みなので、ここは問題にしない。

### リーダーボード (yukon.org/mlxfast) は別物だった

**mlxturbo で参加するものではない。**対象は Swift/MLX の別エンジン
(`Layr-Labs/mlxfast-gemma4-26b-a4b-engine`、`mlx-swift` をベンダリング)。
提出は `yukon` CLI でのソース差分で、編集できるのは 93 エントリ
(Swift のランタイムとベンダリング済み Metal カーネル)。

**量子化は固定** (`mlx-community/gemma-4-26B-A4B-it-qat-4bit` の指定
リビジョン、affine group_size=64 の 4bit)。再量子化は明示的に禁止で、
例外は投機ヘッドのみ。**うちのベイクのレーンも使えない。**

採点は `prefill_gain^0.25 * decode_gain^0.75`、**バッチ 8 固定**、
draft depth は上限 3。締切の記載は見つからなかった。

**持ち込めるのは知識だけで、コードも量子化も持ち込めない。**参加するなら
Swift と Metal を書く別レーンとして立てること。

## 済み (BACKLOG から出たもの)

- **tool calling** → 73e061a で実装。opencode / Codex / Claude Code の 3 クライアントで実機検証済み。懸念だった「モデルが構文を安定して出すか」は Qwen3.6 / Flash-Next とも問題なし
- **MTP の価値の決着** → 決着した。27B の数字から引いた「現 base では 1.2 倍遅くなる」は Flash-Next には当てはまらず、専用エンジン (`spec_flash.py`、深さ 1) で **1.26〜1.44 倍**。MTP 重みは元 checkpoint から抽出したサイドカー (5.2GB、4 段の量子化成果物で共用)。経緯と実測は `docs/MTP-FLASH.md`
- **公開の前提** → `fastmlx` から `mlxturbo` へ改名 (PyPI/GitHub の `fastmlx` は競合の既存プロジェクト)、MIT ライセンス、README の書き直し (「汎用高速推論ランタイム」とは名乗らない)、docs の整理、CI、運用手引き、個人パスの除去、コメントの英語化。手順は `docs/RELEASE.md`
- **プロトコル層の穴** → 非恒等サンプリングの 400 をリクエスト単位の非投機降格に変更 (実クライアントは `top_p` を既定で送るので、看板構成が最初の 1 発で 400 を返していた)、`response_format` の 400 明示、stop の早期打ち切り (ストリームで 11 倍)、Responses の `store`/`previous_response_id`、Anthropic の usage キャッシュ量、`/v1/embeddings` の 501 明示
- **サーバーの配布準備** → 認証・キュー上限・SSE keepalive・graceful shutdown・文脈長ガード・prompt cache 再利用・MTP 自動発見・`--require-runner`。接続手順は `docs/SERVER.md`

### 2026-09-01 追記: 横展開の方向 (ユーザー方針)

**順序 (2026-09-01 ユーザー決定): Qwen3.8-27B の復帰が先、GLM はその後。**
27B は spec.py の出発点で、BPE checkpoint 修正と段階投入の移植が済み次第、
モデル取得 (4bit ~15GB) → compat_smoke → 投機検証 → 計測で復帰させる。
GLM-5.3-Flash は 320B 級 (config 算術) で 4bit 165GB は 128GB に入らず、
**2bit 主体の混合レシピ (~85-95GB) が前提**。一律 2bit は受理率ごと崩れる
実測 (Flash-Next) があるので、層感度ベースのベイクとセットでのみ成立する。

対応を広げるなら **MTP ヘッドを積んだモデル** (DeepSeek V4 Flash、
GLM 5.3 Flash、**Gemma の MTP 版 — 出ているとユーザー指摘 2026-09-01**) が
先で、dense は Qwen3.8-27B (spec.py の出発点) の復帰だけ。MTP 無しモデルへの
持ち札は、(1) n-gram / lookup 投機 (ヘッド不要、ngram_stream + lookup_spec)、
(2) 段階投入 (_staged_forward のグラフ泡刈り、アーキテクチャ非依存の手法)、
(3) 継続バッチング (B=6-8 で合計 2-2.5x)、の 3 つで、qwen4 特化カーネルは
運べない。素の単発 decode で llama.cpp に大差を付ける材料は無いことは
正直に書いておく。**バッチ x 投機の機構 (同期ラウンド・dead-slot マスク・
admission) はアーキテクチャ非依存に書くこと**。**GDN (Gated DeltaNet) を
Flash-Next 独自として扱わないこと** — 線形注意/再帰系は Qwen3.8-27B の
ハイブリッド (linear:full 3:1) にも GLM-5.3-Flash の KDA にも入っており、
今後の主流側。「再帰状態を持つ層の行別 take」は**族として汎用の部品**として
書き、qwen4_exp に閉じるのは状態テンソルの形とフックの当て先だけにする。

## アーキ能力レイヤの設計 (2026-09-01 決定、advisor 判断)

**採用: 薄い能力関数レジストリ (能力関数は汎用・解決は族ごと)。範囲は 2 能力のみ。**
根拠は予報ではなく実測: qwen4 だけで既に同じ知識が 3 箇所に重複している
(spec_flash.rollback / batch_spec.batched_rollback / snapshot_pre 系が
`layer_type == "full_attention"`、`layer.linear_attn`、スロットの意味
(conv/状態/PLE conv/n-gram)、`layer.ple` を独立に書いている)。

切る 2 能力:
1. **層トポロジ + キャッシュスロットの名前付け** — 再帰状態を持つ層とその
   スロットを **名前付きで** 返す (`conv` / `state` / `ple_conv` / `ngram`、
   無い族は None)。**添字で返さないこと** — 添字のまま汎用 API にすると
   「汎用の名前で qwen4 を再凍結」し、結合が今より見えにくくなる。
   再帰層 0 個 (Gemma の sliding window 等) を正常系として返し、
   rollback が no-op になる契約を最初から入れる。
2. **rollback ループ** — 上を使った per-row / 単一行の巻き戻し。
   **indexer/疎注意の trim は別能力として分離** (族ごとに有無が違うため、
   rollback 本体に埋めると if が増える)。

やらないこと (明示的な決定):
- **`_arch()` の一本化はしない。** seam を `model -> モジュール` にすると
  他族を渡したとき「即 ImportError」から「実行途中の AttributeError」に
  劣化する。切るのは `model -> 能力` だけ。`_arch()` は qwen4 固有ヘルパー
  のまま残す。
- **フォワードの写し 3 つ (_staged_forward / _group_prefill_forward /
  capture の GDN 転記) は抽象化しない。**「本家と一字一句同じ」が正しさの
  根拠で、tools/verify_prefill_bitident.py がそれを守っている。抽象の下に
  隠すとゲートの検査対象との対応が切れる。
- **今回 27B を載せ替えない。** qwen4 の 3 コピーを 1 つに畳むところまでで
  止め、qwen3_5 側は「この形で表せるか」を紙で確認するだけ (2 族同時に
  動かすと回帰の切り分けができない)。
- 命名は概念で書く (`recurrent state` / `conv window`)。クラス名
  (GatedDeltaNet) を汎用 API の名前に使わない。

反転条件: GLM の KDA が「位置ごとの再帰状態の一括取り出し」で表せないと
判明したら能力 2 は捨てて族ごとに持つ / spec.py (27B) 経路が今後実行され
ないと決まったら 2 族対応は不要で qwen4 内の重複解消に格下げ /
バッチ x 投機の配線を捨てる判断をしたら今回はやらない。

## 本家フォワードの写し 9 種の整理 (2026-09-01、fable-advisor がコード実読)

### 対応表 (番号 → file:line → 何の写しか)

以前は番号だけが本文中に散らばっていて、一覧できる表が無かった (D4、Opus
正しさ/設計レビュー指摘)。今日 `staged.py` (qwen3_5/dense 側への段階投入の
移植) が増えたので 9 番目として追加する。

| # | 写し (file:line) | 複製元 (file:line) | 内容 | 扱い |
|---|---|---|---|---|
| 1 | `mlxturbo/spec_flash.py:238` `_staged_forward` | `mlxturbo/_vendor/qwen4_exp.py` `Qwen4ExpModel.__call__` + lm_head | 段階投入。既定 2 層ごとに `mx.async_eval(h)` を挟み、グラフ構築中の GPU 泡 (7.3ms) を刈る | **段 4 済み**。前段 (mask 生成・PLE prev_ctx 更新) は本家の `_prelude` を呼ぶ形になり、写しはループ骨格だけになった。骨格は本家に無い制御フローなので残す |
| 2 | `mlxturbo/spec_flash.py:296` `_group_prefill_forward` | 同上 | layer-major prefill。層主導 x G チャンクの二重ループで、MoE 行を concat して 1 回の GEMM にまとめる | **段 5 済み**。層の中身は本家の `pre_mlp` / `_combine` を呼ぶ形になり、残るのは二重ループと MoE の呼び出し粒度 (= 最適化の本体) だけ |
| 3 | ~~`mlxturbo/batch.py:584` `model_call`~~ | 同上 `Qwen4ExpModel.__call__` | mask 生成・conv_mask 構築・n-gram prev_ctx 更新の 3 箇所がバッチ (左パディング) 対応版 | **解消済み (段 3)**。vendor に `_make_masks` / `_store_ngram_ctx` を切って解消 |
| 4 | `mlxturbo/spec_flash.py:158` `capture()` 内 `gdn`/`ple_conv` | `mlxturbo/_vendor/qwen4_exp.py` `GatedDeltaNet.__call__` / `PLELayer._short_conv` | rollback 用に `states_all` 等を保持するための転記 (カーネル差し替えのみ、ロジック不変) | 対象外。「本家と一字一句同じ」が `tools/verify_prefill_bitident.py` のビット一致ゲートの根拠なので抽象化しない (本文の番号ラベルが唯一無い項目 -- 除外 3 項目リストの 3 番目 `capture の GDN 転記` が写し 1/2 と同じ並びで対応する、という消去法での再構成) |
| 5 | ~~`mlxturbo/batch.py` `gdn_call` + `ple_short_conv`~~ | `GatedDeltaNet.__call__` / `PLELayer._short_conv` | conv 状態の取り出しを `cache.lengths` (右パディング下の実長) 基準にする差分のみ | **解消済み (段 1)**。vendor に `_store_conv_state` / `_store_short_conv_state` を切り、batch はその 2 つだけ差し替える |
| 6 | `mlxturbo/spec.py:432` `_linear_capture` | site-packages `mlx_lm/models/qwen3_5.py` の `GatedDeltaNet` (27B/qwen3_5 側) | qwen3_5 用の GDN 捕捉版 | 対象外。上流が site-packages で vendor していない (関数 1 つのために qwen3_5 を vendor するのは割に合わない) |
| 7 | ~~`mlxturbo/batch.py:456` `attention_call`~~ | `mlxturbo/_vendor/qwen4_exp.py` `Attention.__call__` | rope 位置導出のみが左パディング対応の差分 | **解消済み (段 2)**。vendor に `_positions` / `_final_mask` を切り、batch はその 2 つだけ差し替える |
| 8 | ~~`mlxturbo/batch_spec.py:248` `ragged_attention()` 内 `call`~~ | 同上 `Attention.__call__` | rope 位置と最終 mask 構成 (`cache.ledger.next_round_mask`) の 2 点が dead-slot 台帳対応の差分 | **解消済み (段 2)**。写し 7 と同じシームで消えた |
| 9 | `mlxturbo/staged.py:35` `staged_forward` | site-packages `mlx_lm/models/qwen3_5.py` の `Qwen3_5TextModel.__call__` | 写し 1 と同じ段階投入手法を qwen3_5 (27B/dense) 側へアーキ非依存に一般化移植したもの | 対象外。写し 6 と同じ理由 (上流が site-packages)。今日追加 |

**前提の変更 (ユーザー方針)**: 上流 mlx-lm PR #1788 への追随は気にしない。
`mlxturbo/_vendor/qwen4_exp.py` は**うちが所有して自由に改変してよいファイル**
として扱い、必要なら気合いで時々取り込む。この前提により「本家に手を入れる
コスト」がほぼ消え、写しを外側に作る理由も消えた。

**判断の訂正**: 「写しは目的が違うので括れず、上流に手を入れる形でのみ解消
可能」という当初の主張は**過大だった**。実読の結果、B/C 群の差分はシーム
1-2 個に局在している:
- 5 (batch.py の GDN/PLE 写し) の差分は conv 状態の格納だけ →
  vendor に `_store_conv_state` を切れば本体 37 行が消える
- 7 (batch の Attention) と 8 (batch_spec の ragged Attention) の差分は
  「rope 位置の出し方」と「最終 mask の組み方」の 2 点だけ →
  `_positions` / `_final_mask` を切れば両方ともオーバーライド 2 個になる
- 3 (batch の model_call) は mask 生成・conv_mask・n-gram 更新の 3 箇所

**前例がある**: `GatedDeltaNet._project_in` (qwen4_exp.py:417-428) は
fused と capture のために既に切られたシームで、コメントにその旨がある。
「本家にフック点を作る」はこの repo で一度成功しているやり方。

**ドリフトが実際に起きている (綻びの実物)**: batch.py の写しは vendor に
後から入った `_wide_qkv` 融合と sdpa 幅の壁分割を**持っていない**。数値は
不変だが性能改善が伝播しておらず、「本家を変えたら写しも変える」規律は
既に性能面で破れている。ビット一致ゲートが守るのは A 群 1↔2 だけで、
B/C 群には構造的な守りが無い。

**方針**: アーキ能力レイヤの完了後、B/C 群 (5 → 7/8 → 3) からシーム化する。
純粋なメソッド抽出 (計算順・数値不変) で、検査は verify_batch_cache /
verify_batch_real と probe の出力一致。残すもの: 6 (qwen3_5 版、上流が
site-packages なので対象外)、1 と 2 のループ骨格 (async_eval 差し込みと
layer-major の二重ループは写しでなく別物の制御フロー)。2 の pre_mlp/post_mlp
分解は効果対リスク比が最も悪く、やらない選択も可。

**反転条件**: シームのオーバーライドが 1 呼び手あたり 3 個を超えたら
(= フック仕様が内部構造化した兆候) そこで止めて写しに戻す / probe で
decode が退行したら縮小 / PR #1788 が上流に入って vendor を捨てる方針に
戻したら写し方式が再び正しくなる。

**ついでの発見**: batch.py:437-453 の QSA tail 因果性の修正は「本家のバグ
修正が写しにだけ住んでいる」例。vendor を所有するなら本体へ移すのが筋。

### 写しの整理: 実行順と各段の受け入れ検査 (2026-09-01 確定、ユーザー承認)

前節の方針を作業単位に落とす。**着手はアーキ能力レイヤ (arch.py) の完了後。**
各段は独立していて、途中で止めても壊れない。段ごとにコミットする。

**段 1: 写し 5 の解消 (batch.py の GDN/PLE)** — **完了 (2026-09-01)**
vendor に `GatedDeltaNet._store_conv_state(cache, conv_input)` と
`PLELayer._store_short_conv_state` 相当のシームを切り、batch.py はそのメソッド
だけ差し替える。写し本体 (~37 行) が消える。副次効果として batch 経路が
vendor の `_wide_qkv` 融合と sdpa 壁分割を自動で獲得する (現在は写しに
伝播しておらず黙って遅い)。検査: tools/verify_batch_cache.py、
tools/verify_batch_real.py、compat_smoke。

結果: 写し 76 行が override 2 個 (+ 共通ヘルパ 1) になった。
`verify_batch_cache` の出力は変更前後でバイト一致、`verify_batch_spec` も
全ケース一致。オーバーライドは 1 呼び手 (クラス) あたり 1 個で、反転条件
(3 個超) には遠い。batch 経路は `_project_in` (wide 融合) を自動で獲得した。

**段 2: 写し 7 と 8 の解消 (Attention 2 種)** — **完了 (2026-09-01)**
vendor の `Attention.__call__` に `_positions(cache, S)` と
`_final_mask(mask, sparse, dtype)` を切る。batch.py (左パディング) と
batch_spec.py (dead-slot 台帳) はこの 2 つのオーバーライドだけになり、
qkv/rope/sdpa/gate の本体は vendor に 1 つだけ残る。検査: 段 1 と同じ +
tools/verify_batch_spec.py。

結果: 写し 2 つ (batch.py 88 行 + batch_spec.py 36 行) が override 2 個ずつに
なった。QSA には「列位置と真の位置」の区別 (`positions` 引数) が入り、
batch の左パディング対応が本家の規約として表に出た。副次効果として両経路が
`_wide_qkv` 融合と sdpa 壁分割を獲得する。検査は verify_batch_cache がバイト
一致、verify_batch_spec 全ケース一致、加えて**単一系列の指紋**
(`tools/vendor_fingerprint.py`、QSA 活性/不活性 x chunk 割り 4 通りの logits と
全キャッシュ配列の md5) が変更前後で一致。オーバーライドはクラスあたり 2 個。

**段 3: 写し 3 の解消 (batch.py の model_call)** — **完了 (2026-09-01)**
vendor `__call__` を `_make_masks` / `_update_ngram_ctx` + 層ループに分解。
batch はこの 2 つを差し替える。検査: 段 1 と同じ。

結果: `_update_ngram_ctx` は当初案より狭い `_store_ngram_ctx` (末尾文脈の
書き込みだけ) にした。prev_ctx の組み立ては本家と同一なので、共有できる分は
共有したほうが写しが減る。写し 57 行が override 2 個になった。検査は 3 つ
(verify_batch_cache バイト一致 / verify_batch_spec 全ケース一致 /
vendor_fingerprint 一致) とも通過。

**段 1-3 の集計**: 写し 5 つ (257 行) が override 8 個になった。1 クラス
あたり最大 2 個で、反転条件 (3 個超) には触れていない。

**その後の追記 (2026-09-01 夜)**: バッチ x 投機の配線で「不揃いなプロンプト長」を
扱う段になり、再帰状態の窓取り 3 つは **override すら要らない**と分かった。
`cache.lengths` を見るのは上流 (qwen3_next) の規約そのものなので、本家の
`_tail_window` に畳んだ。`lengths` を持たないキャッシュでは末尾を取る従来の
動きで、単一系列は 1 ビットも変わらない。batch.py の override は 8 個から
**5 個** (QSAIndexer 丸ごと + Attention 2 + `_make_masks`) に減り、
batch_spec.py は右パディング prefill をタダで獲得した。
シームは「消せるなら消す」ほうがよい、という実例。オーバーライドが
必要な差分は「位置の意味 (列 vs 真の位置)」「マスクの出どころ」「状態を
どの列から取るか」の 3 種類に収まっており、フック仕様が内部構造に依存する
兆候は出ていない。

**段 4: 写し 1 の前段共有 (低リスク、単独でも価値がある)** — **完了 (2026-09-01)**
`_staged_forward` の前段 ~25 行 (mask 生成・PLE prev_ctx 更新) は本家と
完全に同一。`_prelude` として括り出せば消え、残るのは「層ループ +
async_eval 差し込み」という本物の差分だけになる。**ループ骨格自体は残す** —
2 層ごとの async_eval は本家に無い制御フローで、これが +15% の実体
(グラフ構築中の GPU 泡 7.3ms の刈り取り)。検査: ビット一致ゲート
tools/verify_prefill_bitident.py + probe の出力一致。

結果: 本家に `_prelude` (mask 2 種 + prev_ctx) を切り、`__call__` と
`_staged_forward` の両方がそれを呼ぶ。写しから 22 行が消えた。
`tools/vendor_fingerprint.py` に写し 2 つの検査を足してある
(`_staged_forward` == `Model.__call__` はビット一致、
`_group_prefill_forward` == chunk-major は許容差つき。後者は MoE の行を
concat する以上 CPU 非量子化では累積順が動くため。実測 8.3e-7、実モデルでの
ビット一致は verify_prefill_bitident が見る)。

**段 5: 写し 2 の分解** — **実施 (2026-09-01)**
`_group_prefill_forward` の二重ループ (レイヤー主導 x G チャンク) は
layer-major prefill の本体そのもので、消すことは最適化を捨てることと同義。
ただし `DecoderLayer` を `pre_mlp` / `post_mlp` に分解すれば
hyper-connection 合成式の重複が消えて写しが半減する。advisor 評価では
効果対リスク比が最も悪い。**判断材料**: 段 1-3 で「シームのオーバーライドが
1 呼び手あたり 3 個を超えない」が実地で確かめられたら、同じ基準を段 5 に
当てて可否を決める。超えたなら段 5 はやらない。
さらに踏み込む案として「vendor の `__call__` にレイヤー主導モードを持たせ、
写し 2 を丸ごと消す」もあるが、本家が 2 つの走行モードを抱える複雑さと
引き換えなので、段 1-3 の後に改めて判断する。

**判断と結果**: 段 1-3 の実地でオーバーライドは 1 クラスあたり最大 2 個に
収まり、基準を満たしたので実施した。`DecoderLayer` に `pre_mlp` (PLE +
attention まで進めて MoE の入力を返す) と `_combine` (hyper-connection の
合成) を切り、`__call__` は `pre_mlp` → `mlp` → `_combine` の 3 行になった。
`_group_prefill_forward` の内側ループも同じ部品を呼ぶだけになり、層の中身の
重複が消えた。**残した二重ループが layer-major prefill の本体**で、これを
消すことは最適化を捨てることと同義 (17k TTFT 34.5→32.4s の実体)。
「本家に走行モードを 2 つ持たせる」案は採らない。本家が誰の都合で分岐して
いるか読めなくなるほうが高くつく。

副産物として、prefill の mask 生成が `x` (attention 分岐の入力) ではなく
`hs[ci]` (層の入力) から作る形になった。値は同じ (mask はチャンク幅と
キャッシュのオフセットだけで決まる) が、`pre_mlp` を呼ぶ前に決まっている
必要があるため。

**残すと決めたもの**: 写し 6 (spec.py の qwen3_5 版 `_linear_capture`)。
上流が site-packages の mlx_lm 本体で vendor していないため対象外。
関数 1 つのために qwen3_5 を vendor するのは割に合わない。

**ついでに片付ける**: batch.py の QSA tail 因果性の修正は
「本家のバグ修正が写しにだけ住んでいる」状態。vendor を所有する立場なので
本体へ移す (段 1 か 2 のついでに)。

**完了 (2026-09-01) と、そのとき分かったこと。**端数ブロック (kv 長が
compress_ratio の倍数でないときの末尾 1-3 列) を `ones` にすると、
`sparse` が非 None のとき Attention は causal を捨てる規約なので、その列が
**呼び出し内の全クエリから見える** = 未来が漏れる。合成モデルで本番と同じ形
(budget ぶんを QSA 不活性で流してから、続く 1 回を活性で流す) を組んで測ると、
手前の位置の logits が最大 9e-2 動いた。修正後は完全に 0。
`tools/vendor_fingerprint.py` の causal 検査がこれを常設で見る。

**効く場所**: 端数が立つのは decode/verify のラウンド (kv 長は任意) と、
2048 の倍数でない prefill チャンク。**投機の verify ラウンドでは、draft
トークンが自分より後ろの draft を見ていた**ことになる。受理判定を漏れた
文脈の上で下していたので、分布の意味では直すべきもの。

**未測**: この修正で tok/step (受理率) と 17k の数字が動く。複数プロンプト
x 512 の平均で測り直すまで、スコアボードの 17k 側は「修正前の値」として
扱うこと。revert はこのコミット 1 本を戻すだけでよい。

**残る穴 (別件)**: 可視ブロックが 1 つも無い行 (q_col < compress_ratio-1) は
mask が全面 -inf になり、softmax が全列一様になる = やはり未来を見る。
本番構成 (budget 2048、prefill チャンク 2048) では QSA が効き始める時点で
q_col >= 2048 なので発生しないが、budget をチャンク幅より小さくすると踏む。
恒久解は「causal を捨てず sparse と連言を取る」で、batch 経路は左パディングの
bool mask があるときだけ既にそうしている。効果と速度の両方を測る話なので
別項目に切る。

## アーキテクチャ追従の投資 (ユーザー方針 2026-09-03 14:30: qwen4_exp の最適化が終わってから着手)

**目標**: 全モデルで最高の最適化は目指さない。新しい族が出たとき **8〜9 割の追従が既定で得られる** 状態にする。
毎日のように新しいアーキテクチャが出るので、全部を族ごとに手で書くのは無理という前提。

**「8〜9 割」の定義 (新しい族を載せたときに、対応表 1 枚で当たるもの)**
- 汎用の計測と運用: worker (`tools/ab_daemon.py`)、biglock、decode_ab、小ベンチ、KLD の道具 (teacher は族ごとに作る)。
- 投機 decode の制御: depth 適応、rerank、prime 窓、detokenizer (MTP 頭がある族はそのまま。無い族は draft 経路が別途要る)。
- 汎用カーネル: GDN の Metal 再帰 (GDN を持つ族全部)、head_dim 256 の sdpa 行タイル / d=256 flash attention、BM=64 の qmm (dense 射影 + MLP)、MoE の segmented GEMM (混合タイル)。

**族ごとに残るもの (1〜2 割)**: 本家 forward の写し (staged / group prefill / capture) とビット一致ゲート、族固有のサイドカー (PLE n-gram、QSA)、
KLD の bf16 teacher、MTP 無しの族の draft 経路 (n-gram か小さい draft モデル)。

**現状の障害** (2026-09-03 に確認): `mlxturbo/fused.py` の enable_* 25 個が全部 `mlx_lm.models.qwen4_exp` を import し、`runner.py` は
`model_type == "qwen4_exp"` で分岐するので、27B (qwen3_5_text) では融合が一つも当たらない。BM=64 qmm の対象も q/o/in_proj だけで MLP が無い
(Flash-Next の MLP は MoE だったため)。

**順序と、対応表を切る場所**
1. qwen4_exp の最適化を終える (prefill 1.5 倍、decode の改修)。
2. **qwen3_5 (27B / 35B-A3B) を 2 族目として載せる。ここで対応表を切る** (1 族目では共通部分が見えない、2 族目で初めて分かる)。
   対応表の中身: 族ごとの attention / GDN / MLP / MoE のモジュール名と形 (head_dim、Hq/Hk、GDN の頭数と次元、layer_types)、
   融合の適格判定、forward の写しの所在。enable_* は対応表越しに呼ぶ。上の「アーキ能力レイヤの設計」(能力は名前付き、写しは抽象化しない、
   `_arch()` は一本化しない) の決定はそのまま生かす。
3. Gemma 4 (手元に 26B / 31B の 4-bit) を 3 族目として載せ、対応表で追従がどれだけ速くなったかを測る。Gemma 4 は GDN も MoE も無く
   sliding window + global の混成、同一 forward の MTP 頭は無く別配布の assistant drafter (KV 共有型) を使うので、効くのは attention のカーネルと qmm、それに KV 共有 drafter のエンジン (「Gemma 4 対応の下調べ」と下の追記を参照)。

**反転条件**: 2 族目で共通化できたカーネルが 2 つ未満なら、対応表は作らず族ごとの配線に留める (抽象の維持費だけ残るため)。
mlx_lm の更新で写しが壊れる頻度が月 2 回を超えるなら、写しの数を減らす方 (層のフックで staged を組む) を先にやる。

### 追記 (2026-09-03 14:40、ユーザー): MTP 頭の無い族の投機と、vision / 音声

- **投機の倍率はこちら側で用意する (族ごとに draft の方式が違う)。**訂正 (14:45、ユーザー指摘): Gemma 4 は同一 forward 内の MTP 頭こそ無いが、
  google が **別配布の drafter (`gemma4_assistant`、4 層の cross-attention が backbone の KV を直接読む kv_shared_only、arxiv 2607.02770)** を出していて、
  手元にもある (`~/models/gemma4-31b-assistant` / `gemma4-26b-assistant`、4 層 hidden 1024 head_dim 256)。これを使う。
  ただし機構が Qwen の MTP (同一 forward の nextn 頭) とも `DraftSpecRunner` (独立の小モデル) とも違い、**drafter が target の KV を読むので、
  KV 共有型の drafter エンジンを新設**する (`docs/research/EXPANSION-BOTTLENECKS.md` の Gemma 4 の節)。
  **方針 (ユーザー 2026-09-03 14:50): 公式の MTP 頭 (drafter) が配られていない族は、基本的に投機に対応しない** (素の decode + 汎用カーネルまで)。
  lookup / 小モデル draft / 自前の頭の学習を代用にはしない。対応表の「draft の種類」は MTP 頭 / KV 共有 drafter / なし、の 3 値。
- **vision / 音声も追従の対象に含める** (BACKLOG 1 節「マルチモーダル対応」と同じ話)。Gemma 4 と Qwen の VL 系は mlx_lm 側が VLM ラッパ (`gemma4.py` + `gemma4_text.py` の形) なので、
  8〜9 割の追従はまず text 側の経路で取り、vision / 音声の encoder は別の adapter として足す。投機 decode 側で要るのは「画像 / 音声トークンを含む prefix の prefill と、
  その KV を持ったままの draft / verify」で、text だけの前提を置いている箇所 (prime 窓、n-gram の文脈、checkpoint の位置) を洗うのが先。

## 公開パックの lm_head を 4bit に (2026-09-03、ユーザー判断で本番を 4bit 頭に)

ローカルの本番は `~/models/ddalcu-mlxlm-head4` (真 bf16 から焼いた 4bit 頭)。HF に公開しているパックは 8bit 頭のままなので、差し替えて README に KLD の代金 (+0.0047) を書く。

### 追記 (2026-09-03 18:50、ユーザー): 族ごとの対応表ではなく、**モデルを読んで動的に判定する**方向で

全アーキテクチャへの個別対応は無理なので、読み込み時にモデルの構造を歩いて「当てられる最適化」を自動で選ぶ形にしたい。設計の芯:
1. **構造の探索 (duck typing)**: `model_type` ではなくモジュールの形で判定する。GDN (in_proj_qkv / a / b、A_log、conv1d を持つ再帰層)、attention (q/k/v/o、head_dim、GQA 比、sliding window の cache)、
   MoE (SwitchGLU / gather_qmm の gate/up/down と router)、dense MLP と射影 (`nn.QuantizedLinear` の形)、lm_head、draft 頭 (族が配るもの) を列挙する。
2. **契約検査つきの適用**: 汎用カーネル (GDN Metal、head_dim 256 の行タイル / flash attention、BM=64 qmm、MoE の混合タイル GEMM、K2 型の疎 attention) は、それぞれ `eligible()` で形の契約を確かめて当て、
   起動時に合成入力で素の実装と突き合わせる (ビット一致か許容内)。合わなければその最適化だけ外して素で動く (落ちない)。
3. **起動時の較正**: タイル幅や WM、行数の閾値は機体とモデルで最適が動く (M3 と NAX、mix48 の閾値 48 など) ので、初回読み込み時に短い micro で選んで、モデル × 機体の鍵でキャッシュする。
4. **動的にできないもの**: 投機 decode の staged / group prefill の forward は今は族ごとの写し。mlx_lm のモデルが `model.layers` + cache の慣習に沿っている範囲で、写しではなく「層の列を歩く汎用の staged forward」に置き換えられれば、投機も族を問わなくなる
   (写しが要るのは HC や PLE のような族固有のフックがある場合だけ)。draft 頭は族が配るものだけ (方針)。サイドカー (PLE、QSA) は族固有のまま。
5. 2 族目 (27B) を載せるときに、対応表を「手で書く表」ではなく上の探索の出力にする。3 族目 (Gemma 4) で探索が当たるかを試す。

### 決定 (2026-09-03 18:55、ユーザー): フォールバックではなく「部品ごとの置き換え」で徐々にネイティブ化する

- 土台は mlx_lm の素の forward のまま。そこにフックで部品 (GDN Metal、BM=64 qmm、行タイル、MoE の混合 GEMM、疎 attention) を差す。契約が合う部品だけが当たり、合わない部分は素で動く。
  「一部だけ速い」状態を普通にする (いまの「知っている族は全部速い / 知らない族は全部素」の崖を坂にする)。
- フォールバックは残すが、範囲を「契約検査に落ちた部品だけ」に縮める。モデル丸ごと素に落とすのは読み込み自体が失敗したときだけ。
- 投機の staged forward は今は族ごとの写し。これを「`model.layers` + cache の慣習に沿って層の列を歩く汎用版」にできるかが、この方式が効くかの分かれ目。
  写しが要るのは HC や PLE のような族固有のフックがある族だけにする。
- 「ネイティブ」は MLX の上で部品を自前にする意味。MLX を捨てる (Lily の形) 判断は、decode の糊の融合をやり切っても帯域の壁に届かないと分かったときに改めて。
- 反転条件: 汎用の staged forward が mlx_lm の主要な族で本家とビット一致にできないなら、投機だけ族ごとの写しのままにし、部品の置き換えだけを汎用にする。
- 試金石は 27B: Flash-Next の取り分のうち、部品の置き換えだけでどれだけ移るかをそこで測る。

## 常駐 worker の降ろされ方 (2026-09-03 20:15、観察)

エージェントが `BIGLOCK_NO_WORKER=1` や worker に載らない道具 (verify_* 等で段 0/1 のもの) を投げるたびに、worker は「降りろ」を受けてモデルを捨て、次の decode_ab で読み直す (温で 43 s、冷で 3 分)。
5 本のエージェントが並ぶと数回/時の読み直しになる。直し方の候補: (1) 段 2 (micro) と読み込み不要の道具は降ろさない (既にそう)、(2) worker を通らない decode_ab (特殊 knob) を減らす = knob を worker に載せる、
(3) 降ろすのを「本当にメモリが足りない (回収可能 < 必要量) とき」だけにする。

追記 (2026-09-04 02:20、観察): 降ろされ方がもう 2 つ。
- **`FASTMLX_NGRAM_DISK` の突き合わせループ。**`ab_daemon.env_delta` はジョブの env と worker の launch_env を `MLXTURBO_` / `FASTMLX_` 接頭辞で総当たり比較するが、
  `FASTMLX_NGRAM_DISK=1` は `--ngram` 付きの読み込みで worker 自身がプロセス内で立てる (`mlxturbo/cli.py:138`、`server.py:6845`)。素のシェルから投げたジョブは
  このキーを持たないので永久に一致せず、「env を当てて作り直したのに一致しない」で 2 回作り直して諦める (98 GB × 2)。直し方: worker が自分で立てるキーは比較から外す
  (launch_env を「読み込み前」に取るか、除外リストを持つ)。当面の回避は投入時に `FASTMLX_NGRAM_DISK=1` を明示すること。
- **「回収可能メモリ 99 GB < 100 GB」で 10 分待ち。**前の worker が降りた直後はページキャッシュの回収が間に合わず、新しい worker が閾値の手前で待つ。閾値を 95 GB
  (biglock の MEM_NEED と同じ) に揃えるか、待ちの上限を短くする。
- コードの鮮度検査は他エージェントの編集 (`hyper_connection.py` / `ngram_stream.py` / `spec_flash.py`) でも作り直しを起こす。4 本が並ぶ夜は読み直しが数回/時になった。

直した (2026-09-04 03:53): 上の 2 つ。`ab_daemon.SELF_SET_ENV` (= `FASTMLX_NGRAM_DISK`) を `ab_env` の両側から落として突き合わせループを閉じ (launch_env は元から読み込み前に取っていたが、`os.execv` と `ab_submit.start_daemon` の Popen が読み込み後の env を継承するので足りなかった)、`--ngram` 無しの起動では worker が自分でこのキーを消すようにした。メモリ待ちの閾値は 100 → `MEM_NEED_GB = 95` (biglock の段 0/1 と同じ)。単体テストは `bench/test_ab_daemon_env.py`。

## MoE の行のソートを計数ソートにする (2026-09-03、着手は親の判断)

**決着 (2026-09-04 04:51): 作って測ったが未配線の記録にする** (`kernels/moe_counting_sort.py`、CATCHUP 04:51)。下の見込み -0.25 ms/層 は単発 `mx.eval` の往復を測った誤りで、実費は 0.10〜0.13 ms/層、取り分は 8k で 0.04%。

`_moe_combine_fold` の `mx.argsort(idx_flat)` は 20480 要素 (80 KB、値は 0..511) の並べ替えに **0.224 ms/層**、`row_src` / `_inv_perm` / 表まで入れて 0.26〜0.33 ms/層 かかる。MLX の汎用ソートの多段起動ぶんで、GPU の実働ではない。
ブロック局所ヒストグラム (threadgroup 内で数える) → `cumsum` → ブロック内順位 (原子操作なしで決定的) の 2 カーネル + 小 op 数本にすれば 0.05〜0.08 ms/層 に落ちる見込み = **-0.25 ms/層 = 8k prefill の -0.4%**。
専門家内の行順は結果に影響しない (行ごとの GEMM は位置に依らず、`combine` は k の固定順で足す) ので**出力はビット一致**。P7 第 3 段で残った唯一の的 (router は床、swiglu と topk は合わせて -0.15 ms/層で揺れの中、`docs/research/SESSION-2026-09-02-CATCHUP.md` の 2026-09-03 20:55)。

## `qmm_wide` は M < 1024 で HC の down (10240→320) が素と食い違う → 欠陥ではない (2026-09-04 04:00 に決着)

HC エージェントの副産物 (03:10): `tools/hc_prefill_micro.py` で本番の行数ゲート `fused._QMM_WIDE_MIN_ROWS` (=1024) より小さい幅を測ると、down (K=10240, N=320) が
M=256 で相対 0.53、M=512 で 0.373、M ≥ 2048 で 0.0。調査 (`tools/qmm_wide_shape_micro.py`、`bench/results/qmm-wide-shapes.json`) の結論:

- **素の側が実装を切り替えている。**MLX 0.32.2 の `quantized_matmul(transpose=True)` は M < 13 で qmv 系、K ≥ 128 かつ ceil(M/32)·ceil(N/32) < 256 で
  `qmm_t_splitk` (部分和が bf16 を経由)、それ以外で `qmm_t`。N=320 の down は M ≤ 800 が splitk 帯で、そこでは素と同形の写し (`m32n32k32w2x2`) も
  BM=64 も 1 ビット違わず同じだけ素から外れ、**fp32 参照には写しのほうが近い** (M=256 で素 0.062 / 写し 0.031)。品質の問題ではない。
  判定は `qmm_wide.stock_bit_matches(M, K, N)` (境界 13 / 256 / 128 は M3 Max + MLX 0.32.2 の実測。MLX の契約ではないので更新で動きうる)。
  `eligible()` には入れない (splitk 帯でも写しは `qmm_t` の答えそのもの)。
- **副産物で本物の欠陥を 1 つ直した**: `_WIDE_HEADER` の `loader_w.load_safe(short2(BK, num_outs))` は本家 (BN == BK == 32) の写しで、BN=64 のタイルでは
  N の端の有効な行 (32..63) まで 0 埋めしていた → `short2(num_outs, BK)`。本番のタイル `m64n32k32w2x2r8` は直す前後でビット一致 (変わるのは BN=64 の 4 タイル ×
  `N % 64 > 32` の形だけ、本番の N は全部 64 の倍数)。`bench/test_qmm_wide_shapes.py` 4 本。
- `MLXTURBO_QMM_WIDE_MIN_ROWS` を下げるときの注意は数値の質ではなく、**splitk 帯では素とのビット一致が成り立たない**こと (in-model のビット一致検査は必ず落ちる。
  品質は写しが良い側)。速度はその帯では未測定。

## 27B レーンの下調べ (scout、2026-09-04 04:20、読むだけ) と、今夜見つけた回帰

- **回帰 (実行確認済み 04:20)**: `~/models/qwen38-27b-4bit` (`mlx_lm.models.qwen3_5.Model`、`model.language_model.model.layers`、`model.model` 無し) を読むと
  `enable_default_fusions` が `enable_hc_qmm_wide` → `_hc_gated_residuals` の `model.model` 直参照で AttributeError。同型の直参照は `enable_gdn_decode_fused`
  (9/3 21:05 から既定 on) / `enable_qmm_wide` / `gdn_prework` / `moe_combine_fold` 系 / `ple_hoist` にもある (scout の読解)。**qwen4_exp 以外の族はサーバーが起動しない**
  状態だった可能性が高い (9/3 21:05 以降)。層の列挙ヘルパに寄せて「契約が合わなければ何もしない」に直す (エージェント走行中)。
- 27B の形 (`config.json`): hidden 5120、head_dim 256、Hq 24 / Hk 4 (GQA 6)、64 層 (4 層に 1 回 full attention → GDN 48 / attn 16)、sliding window 無し、
  MoE / QSA / PLE / HC 無し。MTP: `~/models/qwen38-27b-mtp` (`qwen3_5_mtp`、1 層、block_size 3)。
- Flash-Next の既定部品が 27B に当たるか: GDN Metal / sdpa 行タイル / BM=64 qmm は**形は当たる** (head_dim 256、GDN あり、`q_proj/o_proj/in_proj_qkv/in_proj_z/out_proj` の属性名も一致) が、
  実装が `Q.<Class>` (qwen4_exp) へのクラスパッチか `model.model.layers` 直参照なので 27B には届かない。MoE 系 / HC / n-gram / QSA 系 / fused:1 は無関係。
  `_QMM_WIDE_TARGETS` に MLP (`gate_proj/up_proj/down_proj`) は無い。
- 27B の投機: `SpecEngine` 経路 (`runner.py:2069-2097`)。`load_cli_mtp` に `args.mtp` が渡っていない (`runner.py:2079`、`--mtp` は qwen4_exp 分岐にしか届かない)。
  27B の MTP 頭は `FASTMLX_MTP_PATH` か `--original` の snapshot 経由でしか見つからない (設計か見落としかは未確認)。
- 最初に当てる候補 3 つ (机上、`BACKLOG` 713 行付近の方針どおり): GDN Metal、sdpa 行タイル (`qwen3_next.scaled_dot_product_attention` を差す)、BM=64 qmm (dense 射影 + MLP)。
- 基準測定の道具: `bench/self_snapshot.py` はモデル非依存 (サーバーに文字列を渡すだけ)。`tools/decode_ab.py` は族固有コードが qwen4_exp 向け。mlx-serve は `qwen3_5` / Qwen3.8-27B (draft head baked in) を公式表に載せている (`~/dev/mlx-serve/docs/models.md:8`)。
- **順序 (ユーザー)**: フルベンチ → 27B。GPU の基準測定はフルベンチの指示の後。
- 追記 (04:45): 回帰は直した (`fused._model_layers` に寄せ、契約が合わなければ何もしない。`bench/test_fusions_other_family.py`)。27B の煙試験 (32 トークン) は起動・生成 OK。
  落ちなくなった結果、契約の合う部品は 27B にも当たる: `qmm_wide` が 176 射影 (q/o_proj、prefill 幅 ≥ 1024)。速度は未測定 (27B レーンの基準測定で on/off を取る。`MLXTURBO_QMM_WIDE=off` で切れる)。
  27B の射影の形 (K=5120→N=6144、K=6144→N=5120、M=1024 / 2048) では `qmm_wide` は素とビット一致 (`tools/qmm_wide_shape_micro.py`、`bench/results/qmm-wide-shapes-27b.json`、04:50)。品質の代金は無い。
  **27B の MTP サイドカー (`~/models/qwen38-27b-mtp`、量子化済み) は読めない**: `mtp.py:125-138` が重みを読んでから `nn.quantize` する順なので、`fc.scales` 等 16 個が
  「model に無い」と弾かれて None に落ち、lookup だけの投機になる (起動は続く)。`--mtp` 自体は `FASTMLX_MTP_PATH` 経由で 27B 経路に届いている。27B レーンの最初の直しはこれ。
  **直した (2026-09-04 08:10)**: `load_mtp_file` が量子化済みサイドカーなら「quantize → load_weights」の順に切り替え、ディレクトリ指定も受ける (`mlxturbo/mtp.py`、`bench/test_mtp_quantized_sidecar.py`)。
  `投機デコード有効 (MTP: あり / lookup: 有効)` が出て、4k の煙試験で tok/step 1.00 → 2.33〜3.97、decode 22.2 → 29.0 tok/s (`bench/results/smoke-27b-mtp-0904.json`)。
  貪欲の出力は 2 プロンプト x 96 トークンで 1 箇所だけ食い違い、そこは top-2 が bf16 で完全同値 (どちらも +16.125) の同点。MTP あり / 無しとも自己再現する。
- 追記 (08:05、scout): 相手の 27B の動かし方。**mlx-lm** は `qwen3_5_mtp` を読まず (`sanitize` が `mtp.` を捨てる)、投機は `--draft-model` の別モデルだけ → 素の基準。**oMLX 0.6.4** (`/opt/homebrew/bin/omlx`、ソース `~/dev/omlx`) は
  `omlx serve --model-dir <親>` で発見、`qwen3_5_mtp` を drafter として識別 (ペア付けは model_settings の `vlm_mtp_draft_model`、CLI 無し)。**MTPLX 2.9.2** は `tools/compare/mtplx-venv/`。
  **mlx-serve** は `ddalcu/Qwen3.8-27B-MLX-Serve-4bit` (18.2 GB、draft head baked in) が推奨でローカルに無い。sidecar 探索は同一ディレクトリ直下の `mtp.safetensors` 等。rapid-mlx は無し。
  HTTP harness は `bench/vs_mlx_serve.py` の `stream_once` (エンジン非依存)、`self_snapshot.py` は mlxturbo / mlx-serve の argv 直書き。池は 27B でも同じ (vocab 完全一致)。

## 後でやる: draft の学習 (小さい draft model の蒸留、uzu 型) — ユーザー 2026-09-04 08:10「速度のためにはやるべき。いずれやる気はするが、特化すると汎用性能として使いづらいので、あえて手を出していない。アイデアとして残す」

Mirai / uzu (`docs/research/EXTERNAL-MIRAI-UZU-2026-09.md`) の 105 tok/s の半分は「<50 MB の学習した draft」で、MTP 1 層 (2 倍) の上に 1.5〜2 倍。
ただし uzu は族ごと・用途ごとに学習していて、一般チャット (MT-Bench) では 84 に落ちる = 特化の代金。うちで採るなら条件は「汎用性能を落とさない」:
(1) 学習データは特定領域に寄せない (teacher の rollout を広い池から)、(2) 受理率が落ちる入力でも素の decode より遅くならない設計 (depth 適応は既にある)、
(3) 品質は draft に依らない (検証は本体なので出力は同じ。速度だけの話)。MTP 頭の学習 (方針で無し) とは別物で、本体を触らない小モデルの学習。
着手時期はユーザーの判断。先に 27B レーンの基準測定。

## そのうち: qwen4_exp のベタ書き (spec_flash の写しとシーム) も剥がす (ユーザー 2026-09-04 09:12)

27B の decode 経路 (spec.py) から同期と写しを剥がす (B) を先に。それが着地したら、`spec_flash.py` の qwen4_exp 依存 (`capture()` の `__call__` 差し替え、`_staged_forward` の層呼び出し規約、
HC 型の MTP 頭、QSA cache、`_arch()` の決め打ち) も同じ形 (モジュール呼び出し + 状態の取り出し口 + `arch.py` の duck typing) に寄せていく。
順序はユーザー: 27B → そのうち Flash-Next。Flash-Next 側は 17k の A/B と fingerprint を毎回のゲートに。

## そのうち: 看板の架け替え (ユーザー 2026-09-04 11:17「多分普通に掛け替える」)

位置づけを「皆が使う推論エンジン (製品が中に入れるバックエンド)」に寄せるのに合わせて、名前と看板を掛け替える。対象: パッケージ名 (PyPI `mlxturbo`)、モジュール名、HF のパック名 (`ddalcu/…`)、README の位置づけ、docs の自称。決めるときは一括で。時期はユーザーの判断。

## 小 M (2〜8) の量子化行列積の自前カーネル (ユーザー 2026-09-04 11:32「実装する。まず micro で徹底検証」)

MLX の `quantized_matmul` は M=1 (qmv) で 400 GB/s 級なのに M=2〜8 (fast_qmm) で 209 GB/s。投機デコードの verify 幅が全部ここを踏む (27B の S=4 で +32 ms/round)。
自前の multi-row qmv (同じ重みタイルに M 行を同時に掛けて重みを 1 回だけ読む)。**数値の目標は各行が qmv (M=1) とビット一致** = verify 幅で丸めが変わらない「同じ挙動の保証」。
いろんな族で使い回せるので、**MLX 本体に issue / PR を出す候補** (qmv の作法に寄せて書く)。PoL は `scratchpad/agent-27b-verify-width.md`、micro は `tools/qmm_smallm_micro.py`、テストは `bench/test_qmm_smallm.py`。
