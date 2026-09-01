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
