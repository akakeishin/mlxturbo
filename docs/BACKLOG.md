# BACKLOG — やりたいが未着手のもの (2026-08-29 更新)

ここには「やる価値はあるが手を付けていない」ものを、着手前に分かっている根拠つきで置く。決着したものは末尾の「済み」へ移した。

## 1. マルチモーダル対応 (画像 → 音声・動画)

元 checkpoint は VLM (`Qwen4ExpForConditionalGeneration`)。1,658 キー中 **333 が vision 系** (`model.visual.blocks.*`) だが、変換で意図的に捨てている。変換後の v 系成果物に vision 系は **0 キー**。元 checkpoint (360GB / 131 シャード) は外付け `/Volumes/Mobile SSD/models/Qwen3.8-Flash-Next` に再取得済みなので、素材はいつでも読める。

落としている箇所:
- `fastmlx/convert_flash.py:331` — `mtp.` / `vision_tower.` / `model.visual.` を skip
- `tools/vendor/qwen4_exp.py:849` — mlx-lm 側の `sanitize` でも同じものを skip (`# text-only pour l'instant`)

必要な作業は 3 段。

1. 変換で `model.visual.*` を残す (`convert_flash.py:331`)
2. **mlx-lm 側に vision タワーのクラスを書く。** 現状 `qwen4_exp.py` に vision/visual の言及は skip の 2 行だけで、クラス自体が存在しない。**ここが一番重い**
3. エンジンが埋め込み入力を受け取れるようにする。`fastmlx/spec.py:164` が `self.inner.embed_tokens(tokens[None])` と自前で埋め込んでいるのを、外から渡せるようにする。モデルの `__call__` は既に `input_embeddings` を受ける口を持っている (`qwen4_exp.py:746`)。`spec_flash.py` 側も同様の口が要る

投機デコード固有の注意点として、**n-gram lookup はトークン ID 上で動くので、画像プレースホルダ (同じ ID が数百個並ぶ) の区間で誤マッチを量産する。** lookup の対象から画像区間を外す処理が要る。MTP 側はドラフトが常にテキストなので影響しない。

音声・動画は扱う場所自体が無い。ViT のぶん重みも増えるので、91GB がさらに膨らむ点も込みで判断すること。

## 2. サーバーの並列化 (継続バッチング)

**測定は済んでいて、実装も opt-in の形で存在する。配線だけしていない。**

- 取り分の実測: gemma4-26B で B=4 実測 2.10x。Flash-Next はエキスパート和集合からの予測で 2.05〜2.19x (`bench/results/batch-flashnext-expert-union.json`)。「512 エキスパートなら飽和しにくい」は誤りで、飽和は (B × top_k) / num_experts で決まる
- prefill には一切効かない (B に線形)。cold 32k は TTFT 112 秒で壁時計の 84% を prefill が占める (`bench/results/batch-flashnext-prefill-decode.json`)。**体感を狙うなら prefill 側が先**
- 実装: `fastmlx/batch.py` に `enable_batch_cache()` (既定 off、未配線)。合成モデルの CPU 検証で QSA off と長さ揃いは B=1/2/4 完全一致。**残る制限**: QSA 有効 (2048 超) かつ長さ不揃いでバッチ出力が単体と一致しない (破損ではなく QSA のブロックグリッドが絶対列で切られるため。パディングを揃えれば 3.7e-8)。実モデル検証 `tools/verify_batch_real.py` は未実行
- やる根拠は速度でなく**可用性**: 直列サーバーだとサブエージェントを流しっぱなしにしたまま別の作業ができない。B=4 でレイテンシは 0.53x に悪化するので、対話 1 本の用途では逆効果

継続バッチングを投機デコードと GDN の再帰状態の上に載せるのは片手間の規模ではない。着手するなら `verify_batch_real.py` の実行が最初の一歩。

## 3. 他モデル対応 (gemma / kimi / glm)

サーバーは「spec/flash_spec の契約に合わなければ通常生成にフォールバック」なので、**載せれば喋る**。fallback の理由は `/health` の `fallback_reason` に出る。ただし投機デコードは効かない。

- 27B 系の `SpecEngine` は `fastmlx/_mlx_compat.py:111` の契約 (GDN ハイブリッド固有の形) を要求する
- Flash-Next 系は `spec_flash.py` の `FlashSpecEngine` (qwen4_exp 固有)
- 他アーキテクチャで投機を効かせるには契約の一般化が要る。lookup (SAM) 側はモデル非依存なので、そちらだけ先に切り出す手はある

この Mac では VRAM が先に効くので、載せられるサイズが実質的な制約になる。

## 4. n / logprobs (公開後の需要待ち)

一度「この Mac では不要」と見送ったが、**公開して他のモデル・他の機械で使われるなら判断が変わり得る**ので、却下ではなく保留に格上げしておく。

`n` (複数候補) は、直列のままだと生成時間が候補数倍になるので §2 のバッチ化が前提。同一プロンプトの n 候補は §2 のバッチにそのまま載る。

`logprobs` は投機経路では正しい値を返しにくい。ドラフトのまま受理されたトークンの logprob は verify 時点のロジットにあり、棄却後に引き直したトークンのそれは残差分布にある。区別せず最終ロジットの softmax を返すと、投機の痕跡が消えた不正確な値になる。fallback 経路に限れば素直に実装できる。

## 5. prefill の高速化 (体感の本丸)

prefill は 290〜325 tok/s しか出ない (デコードの約 11 倍でしかない)。Claude Code は毎ターン 54k 超を送るので、初回 TTFT を支配する。prompt cache の再利用 (fadadf7) で 2 ターン目以降は 9 割超を飛ばせるようになったが、**初回とキャッシュミス時はまるごと食らう**。バッチ化では縮まない (B に線形)。カーネル側の課題として未着手。

---

## 済み (BACKLOG から出たもの)

- **tool calling** → 73e061a で実装。opencode / Codex / Claude Code の 3 クライアントで実機検証済み。懸念だった「モデルが構文を安定して出すか」は Qwen3.6 / Flash-Next とも問題なし
- **MTP の価値の決着** → 決着した。27B の数字から引いた「現 base では 1.2 倍遅くなる」は Flash-Next には当てはまらず、専用エンジン (`spec_flash.py`、深さ 1) で **1.26〜1.44 倍**。MTP 重みは元 checkpoint から抽出したサイドカー (5.2GB、4 段の量子化成果物で共用)。経緯と実測は `docs/MTP-FLASH.md`
- **サーバーの配布準備** → 認証・キュー上限・SSE keepalive・graceful shutdown・文脈長ガード・prompt cache 再利用・MTP 自動発見・`--require-runner`。接続手順は `docs/SERVER.md`
