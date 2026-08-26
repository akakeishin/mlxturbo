# mlx-lm の外へ出る賭け（2026-08-26 発散。SSD の生き残り案込み）

前提: m=1 decode と prefill は物理天井近傍（README）。mlx-lm を捨てて買える
ものは「検証パスの実行効率」「データレイアウト」「ステップ間の隙間」に限られる。
以下は互いに主要機構が異なる 5 本。定番（全面書き直し、llama.cpp 乗換え、
FreeToken 移植）は買うものが特定できないため候補にしない。

## B1: 検証ステップの直接エンコード（実行の所有権）

- 核: 検証 1 ステップ（射影→GDN→attention→MLP→lm_head）だけ MLX の op 実行を
  迂回し、fastmlx が 1 本の command buffer に直接エンコードする。MLX は
  テンソルの置き場としてだけ使う
- 機構: ラッパ税 0.064ms/call × 呼び出し数と、op 境界のディスパッチ隙間を消す。
  目標水準 (~0.15ms/カーネル) ではラッパ税だけで 3-4 割を占める実測があり、
  カーネル改善の終盤で必ず当たる壁を先に壊す
- 代償: mlx-lm 互換の細道が増える。デバッグは ISA 基盤頼み
- probe: ISA 基盤の変種機構で「線形層 1 層ぶんを 1 command buffer に直接
  エンコード」した試作を作り、同一計算の MLX 経由と依存チェーンで差を測る

## B2: fastmlx 専用モデルフォーマット + Metal 3 I/O（データの所有権）

- 核: safetensors/stacked-expert を捨て、(layer, unit) 単位の slab 整列 +
  MTLIOCommandQueue でファイル→MTLBuffer 直接ロードの自前形式
- 機構: 起動が変わる (115GB 級を並列 direct I/O で数十秒→MTLIO でさらに短縮)。
  expert/層単位の部分常駐がフォーマットレベルで可能になり、B3 と将来の
  SSD 階層の前提になる。sol 設計書 (OFFLOAD-DESIGN-SOL.md) の expert store の一般化
- 代償: 変換ツールと二重管理。MLX 側から MTLBuffer を差し込む拡張が必要
  (= 小さな ObjC++ 拡張。これが B1 とも共通の下部工事)
- probe: MTLIO で 20GB を MTLBuffer へ直接ロードする 50 行スパイク。
  スループットと mx.array 化の可否だけ確認する

## B3: 際どい常駐の安全弁 = tail expert 部分 offload（SSD の生き残り①）

- 核: V4 Flash 115GB 常駐の恐怖 (OS 残 10GB) に対し、最低頻度 expert
  ~10-15GB だけ SSD に置き、RAM に OS の呼吸域を返す
- 機構: 全量 offload (棄却済み) と違い、必要 hit 率 ~97% は頻度スキューで
  現実圏。miss は 16-21MB 読み ≈ 3-4ms が稀に挟まるだけ
- 代償: B2 のフォーマットが前提。routing trace の採取が要る
- probe: V4 Flash 入手後、M1 の道具で expert 頻度分布を採り、
  「上位何 GB で累積 97% を超えるか」を見る。超えなければ棄却

## B4: セッション KV の SSD 永続（SSD の生き残り②。最安・確実）

- 核: ChatSession の KV + GDN 状態 + MTP キャッシュを SSD へ書き、
  再起動・セッション切替をまたいで差分 prefill を効かせる
- 機構: 読みは prefix 一括の連続読みで償却が要らない (512 tok ≈ 33MB、
  5.5GB/s なら 6ms)。offload 研究が否定したのはランダム expert 読みで、
  連続一括読みは SSD の得意分野。TTFT ゼロ化の適用範囲が「同一プロセス内」
  から「いつでも」に広がる
- 代償: 状態の版管理 (モデル/quant/mlx-lm が変わったら無効化)。容量は
  1 セッション数十 MB で無風
- probe: caches の serialize/deserialize + 差分 prefill 接続の 1 日スパイク。
  gate は再ロード後の greedy 一致

## B5: ステップの software pipelining（遠距離移植: CPU パイプライン）

- 核: 現ステップの verify を GPU が実行中に、次ステップの lookup draft 探索と
  検証窓の組み立てを CPU で先行させ、受理判定後に即 submit する
- 機構: step 境界の直列化 (eval 同期→python→次 submit) を隠す。MTP draft は
  h_last 依存で先行不能だが、lookup draft は文脈だけに依存するので先行可能
- 代償: 判定が外れた分の組み立ては捨てる (CPU 仕事なので安い)
- probe: phase_s に「step 境界のアイドル」を追加計測してから着手判定。
  アイドルが 2ms/step 未満なら棄却

## 順序の提案

確実性とコストで B4 → (Sol の v4 完了後に) B1 の probe → B2 スパイク →
B3/B5 は probe の数字次第。B2/B3 は V4 Flash (M5) と束ねると二度手間がない。
