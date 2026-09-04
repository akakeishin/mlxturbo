# 製品の方向 (2026-09-04、ユーザー決定。GPT-5.6 Pro の意見を踏まえて親が採否を付けた)

## 位置づけ

- oMLX (製品: DMG / brew / メニューバー / 管理画面 / 全モダリティ / SSD KV cache / 継続バッチング、2.1 万 star) と「高機能な Mac 向けローカル AI サーバー」の座を争わない。
- **目標は「皆が (意識してもしなくても) 使う推論エンジン」**: oMLX を含む各種サーバーが中の実行エンジンとして採用したくなるもの。oMLX が中で使うようになったら目標達成。
- 一文: **モデルとリクエストごとに、最速の検証済み実行経路を選ぶ。高速化できないときも、その事実と理由を隠さない。** (The verified fast path for MLX inference. Embeddable, agent-native, never silently slow.)
- 近い位置づけは「mlx-lm の LLM 特化版」(同じ重み・形式・品質のまま、契約で当たる部品で族に追従)。看板と名前は後で一括で架け替える (BACKLOG)。

## 採る (P0、27B の decode レーン (小 M qmm + dispatch 修正) が落ち着いてから着手)

1. **ExecutionPlan / `mlxturbo explain` / リクエスト単位の strict plan + HTTP ヘッダ** (`x-mlxturbo-plan` / `-exactness` / `-cache` / `-downgrade`)。runner / fallback_reason / downgrade_reason を公開契約に。速度の数値ではなく経路と意味論 (MTP の有無、非投機への降格、分布の同一性) を保証する。**最初にやる。**
2. **Engine / Session の公開 API** + mlx-lm 互換アダプター (`stream_generate` 形)。ChatSession / checkpoint / fork を公開に引き上げる。公開型に `mx.array` を漏らさない (将来の backend 境界)。
3. **配布**: Python 3.11〜3.13、`uvx mlxturbo serve …`、`mlxturbo doctor` (Fast path / MTP / kernels / memory / exactness / unsupported options)。brew は後。
4. **同一重み・同一 payload の相手比較の常設** (今日の 5 者 harness) + **fast-path hit rate** を主指標に足す (tok/s と両方)。北極星: 実効的な高速化価値 ≒ 対象リクエスト率 × 1 件あたりの削減時間。
5. README の主語を「実行エンジン」に (看板の架け替えと一括)。
6. core / serve の分割: 公開 API を先に、パッケージ分割はリリース時。

## 採るが後 (P1)

- **TurboPack**: 「契約で当てる部品」の方向そのもの。契約 (probe → SupportReport / build_plan / load / create_session / conformance_tests) は定義するが、別パッケージへの分割は族が 2〜3 の今は早い。qwen4_exp と qwen3_5 を tree 内の最初の Pack として形を揃える。
- **Session の fork / rollback / checkpoint の公開**と、その指標 (32K〜128K からの分岐 TTFT、append の再計算トークン数、rollback 時間、4 分岐の総時間、長時間の p95 TTFT)。うちの一番強い場所。
- **統合スケジューラ** (通常バッチ + 投機を 1 つの token-budget scheduler に、policy = latency / throughput / memory / interactive / exactness)。「同時接続」の項目、27B の後の順番。
- OpenAI / Anthropic / Responses の conformance suite、oMLX 向けアダプター。
- 1 worker = 1 model を設計原則にし、上に軽い router (プロセス分離のマルチモデル)。

## 採らない / 保留

- 「特定モデルの decode の数 % 最適化を追わない」は半分だけ。**小 M qmm は全族の verify 幅に効く共通項で「同じ重みで負けない」に要る → 仕上げる。**それが終わったら Flash-Next の 1〜2% の的は止めて P0 へ。
- Input IR / VLM Pack / router のマルチモデル / 非 MLX backend / 豪華なダッシュボード / モデル検索 UI / メニューバー / 同一プロセス内マルチモデル / OCR・Embedding・Reranker の網羅: 後、または追わない (oMLX の後追いになる)。

## 順番

27B の decode (今週) → ExecutionPlan + explain (2〜3 日) → Engine / Session API + mlx-lm アダプター (3〜5 日) → 配布 (1〜2 日) → 常設ベンチ + hit rate (1 日) → README と看板 → TurboPack 契約 → Session fork API → スケジューラ統合。
