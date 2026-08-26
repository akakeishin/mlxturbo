# fastmlx 実装計画書

実装主体: Codex Sol（マルチエージェント可、サブエージェント最大12並列）。
統括と受け入れ判定: Claude 側セッション。人間の意思決定者: ht。

## 1. 目的と成功条件

一次目標は「自分が毎日使う最速のローカル推論エンジン」。全 M 系チップで動く設計を保ち、
特定のメモリ容量を前提にしない。対 CUDA の主張はしない。比較対象は Mac 上の既存
スタック（mlx-lm 素、llama.cpp、Rapid-MLX、MTPLX）。

成功条件（M3 Max、静かな環境での実測）:
- S1: 難しい内容の持続 decode が mlx-lm 素の 2.0 倍以上（現状 1.3〜1.5 倍）
- S2: greedy 出力が非投機と分布厳密同一のまま（identity gate 通過）
- S3: チャット折り返し TTFT 0.3 秒未満を維持
- S4: 既存 OSS（MTPLX / Rapid-MLX / mlx-lm PR#990）との A/B で総合最速
- S5: 比較マトリクス 4 モデル（下記 Phase M）で S1/S2 が成立し、
  Qwen3.8 専用ハックでないことを示す

## 2. 現状

エンジン: fastmlx/spec.py（SpecEngine）。MTP 連鎖ドラフト + 文脈 lookup 混成、
m 可変一括検証、受理適応深度、厳密棄却サンプリング（temp>=0）、ChatSession 差分 prefill。
線形アテンション 48 層の巻き戻しは gated_delta_update_with_states カーネル
（fastmlx/kernels/gated_delta_states.py、mlx_lm カーネル比 bit-exact）で全位置状態を保持。

実測（コミット済み JSON が bench/results/ にある）:
- mlx-lm 素 decode 21〜23 tok/s。fastmlx は易しい内容 40〜51、難持続 28〜35 tok/s
- 受理: 易しい文 ~3/3、難しい文平均 1.0〜1.5（連鎖 2 リンク目以降が弱い）

## 3. 物理制約（確定事実。再測定不要）

- ストリーミング読み天井 345〜350GB/s。decode m=1 は 344.5GB/s ≒ 天井（磨く余地なし）
- GEMM 天井 ~13.1 TFLOPS。prefill は 10.7TF = 82%（余地 1.22 倍、犯人は qmm_t タイル）
- 検証税: M=6〜12 は qmv_wide の 5 ベクトル上限で重み ceil(M/5) 回読み。
  M=13〜32 は qmm_t 32^3 タイル（算術強度 12.8 MAC/B、必要 38）で帯域の 2〜3 割
- 物理目標: m∈{2..10} は帯域 roof（mlp タイルで ~0.145ms）、m∈{11..16} は演算 roof。
  m=16 を帯域天井に置くことは不可能
- scales/biases が量子化バイトの 11.1%（実効 4.50bit/weight）。group128 化で ~1.06 倍
- GatedDeltaNet の scan は prefill でも decode でも無実（全視点一致）

## 4. フェーズと作業項目

依存関係: A → B は直列。C と D は A/B と並行可。E は最後。
GPU を使う計測は同時に 1 プロセスだけ（結果が壊れる）。実装・単体正しさ確認は並列可。

### Phase 0: レビュー指摘の修正（最初にやる。A より先）

docs/REVIEW-2026-08-26.md（Codex Sol による ca9dc7f のレビュー）の指摘を処理する。
「壊れる」は全件修正、「怪しい」は修正または根拠つきで見送り判断を残す。特に:

- convert.py の norm 二重シフト（最重篤）: MTP 入り成果物の再ロード時に本体
  RMSNorm へ +1 が二重適用される。保存時に shift 対象 norm を raw 規約へ戻し、
  source-loaded と output-reloaded の logit 一致テストを必ず追加する
- spec.py の session 例外安全性: reuse 中の失敗で cache と processed が
  食い違う。成功時のみ一括 publish に変える
- max_tokens 超過・受理列中 EOS の consumed/fed_gen/巻き戻しの整合
- gated_delta_states の形状ガード（Dk%32、Hv%Hk、state/mask 形状）と
  非対応形状の ops fallback、mask=false 分岐の 32 lane 同一アドレス書き込み
- 契約の明文化: n_draft=0 かつ lookup_len=0 が非投機 baseline（bench/gate.py は
  これを固定）。sharded model は capture 入口で明示拒否。rollback は
  isinstance でなく is_trimmable()/trim() プロトコル経由
- _mlx_compat.py の新設: mlx-lm 内部依存（レビュー末尾に列挙あり）を集約し、
  mlx / mlx-lm の上限を pyproject に固定、起動時 contract test を置く

### Phase A: カーネル（検証税の解消）

- A1: qmv_wide 上限解除カーネルの完成と統合
  fastmlx/kernels/qmv_wide_nocap.py（作業中の成果物を引き継ぐ）。
  vecs_per_tg を M（6〜12）に。受け入れ: 相対誤差 < 2e-3、依存チェーンで
  M=6..12 が mlx 比 1.5 倍以上、M=2..5 はフォールバックで劣化ゼロ。
- A2: MMA skinny qmm のクリーン実装（M=6〜16）
  設計は確定済み: 8x8 simdgroup_matrix、量子化グループ（64 要素）単位の dequant、
  K の 8-simdgroup split-K、x は threadgroup にステージせず device から直接
  simdgroup_load、M=9..16 は同じ B タイルを共有する C タイル 2 枚。
  fastmlx/fast_qmm.py（vendored、Apple 著作権表示あり）はライセンス未確認のため
  **参照のみに使い、コードはコピーしない**。クリーンに書き直すこと。
  受け入れ: 誤差は fp32 アキュムレータで 1e-3 未満、依存チェーンで m=8 が
  mlx 比 1.5 倍以上、m=16 が m=8 の 1.6 倍以内。
- A3: ディスパッチャ
  fastmlx/kernels/dispatch.py。shape(K,N) と M で stock / nocap / MMA を選ぶ表。
  QuantizedLinear の差し替えは enable(model) 方式（fast_qmm.py の方式を参考に自前で）。
  lm_head（N=248320）と mlp（N=17408）で最適が違う前提で表を実測から作る。
- A4（任意、A1-A3 後）: prefill 用大 M qmm タイル拡大の実験
  qmm_t の BM/BN/BK を 64 に上げた版。賞金 1.22 倍。難しければ捨ててよい。

### Phase B: エンジン統合と正しさゲート

- B1: A のカーネルを SpecEngine の検証パスへ接続（_hidden_forward の capture 経路と
  _head）。m ごとの経路選択は A3 のディスパッチャ経由。
- B2: 自動 identity gate の整備
  bench/spec_bench.py に (a) n_draft=0 の spec vs stock greedy 完全一致テスト、
  (b) 投機 on で n_mismatch と不一致位置の記録（済み）、を CI 的に一発で回す
  スクリプト bench/gate.py としてまとめる。カーネル変更のたびに必ず通す。
- B3: phase_s（draft/verify/maint 分解）の JSON 保存と、async_eval による
  draft 時間の過小計上の修正。

### Phase C: 再量子化と配布物

- C1: fastmlx/convert.py（作業中の成果物を引き継ぐ）で group128 版を実変換し、
  品質ゲート + decode 速度 A/B。1.06 倍と品質劣化のトレードオフを表にする。
  品質ゲートは KLD を主指標にする: bf16 参照に対する出力分布の KL divergence を
  固定評価セットで測り、閾値超えは速度がどうであれ不合格。greedy 一致率と
  logit 差は補助指標。背景: MTPLX は KLD の悪い野良 quant を土台にして
  「品質が悪い」という評判が固定された（作者自身の総括）。同じ失敗をしない。
  現在使っている lmstudio 配布 4bit も同じゲートで一度検証する。
- C2: mlx-community/Qwen3.8-27B-MTP-4bit サイドカーの検証。
  自前抽出 MTP と同一かを重みレベルで確認。同等なら配布依存を削減できる。
- C3（任意）: 混合精度量子化の実験。低ビット FP のハード支援が無い世代では
  層ごとの INT4/INT8/BF16 の使い分けが品質と速度の主戦場になる。
  層別感度（KLD への寄与）を測り、感度上位層（embed/lm_head/最初と最後の
  ブロックが定番候補）だけ 8bit にした構成の品質/速度/サイズを C1 の表に足す。

### Phase D: アルゴリズム（受理率の分子側）

設計根拠は docs/RESEARCH.md（必読）。効き順:

- D0: 診断計測（半日、最初にやる）
  (a) 位置別受理率 α_k の分解出力（accept_trace から集計可能）。
  FastMTP の vanilla 値（70%/11%/2%）と比較し、同じ崩壊カーブか確認。
  (b) 連鎖位置ごとの ĥ の RMS と真の h の RMS の比較（Attention Drift 診断）。
  膨張していれば推論時スカラー再スケールを試す（訓練ゼロの応急処置）。
- D1: 停止則の刷新（訓練不要、即日）
  AdaEDL のエントロピー打ち切りを draft 連鎖に入れる。深度・m の大枠は
  Sequoia の式に実測 t(m) を代入して解く（Leviathan 閉形式は使用禁止。
  検証コスト線形仮定が m カーブと矛盾）。式の校正が難しければ BanditSpec
  （深度=腕、報酬=実測 tok/s の UCB）に切り替え。
- D2: FastMTP レシピで MTP ヘッド微調整（本命。受理 1.0-1.5 → 2.5-3.0 期待）
  位置減衰 CE（α_k ∝ 0.6^(k-1)、K=3）、position-shared 再帰適用、backbone 凍結。
  データは実使用分布の自己生成（一般ドメインは無効果、RESEARCH.md 参照）。
  MTP ヘッドは ~0.4B なので M3 Max で射程内。学習スクリプトは
  fastmlx/train_mtp.py としてエンジン付帯に置く（backbone 学習はやらない）。
  伸び悩んだら HASS の top-K 蒸留と EAGLE のノイズ注入 U(-0.1,0.1) を追加。
- D3: lookup の SAM 化と ReSpec 仲裁
  SAM-Decoding の二重 suffix automaton（静的+動的）で O(n) 走査を置換。
  仲裁は ReSpec 方式: エントロピー閾値で retrieval 起動、一致位置ごとの
  受理率 EMA、source-aware verification。CPU 側は 20-30µs/token を予算に。
- D4: Block Verification（分布厳密のまま +5-8%。棄却サンプラ差し替えのみ）
- D5: 木投機（着手条件: A2 完了 + D2 完了後も受理が頭打ちの場合のみ）
  GOOSE の異方的 spine から。GDN 48 層は chain 据え置き、full attention 16 層
  のみ木マスク。4bit×木検証の相殺警告（RESEARCH.md）を先に実測で確認。

### Phase M: マルチモデル対応（B 完了後。C/D と並行可）

比較マトリクス（2 家系 × Dense/MoE。モデル ID は実行時に HF で正確な名称と
MLX 量子化の有無を確認してから固定する）:

| モデル | 型 | 意味 |
|---|---|---|
| Qwen3.8-27B | Dense (hybrid GDN) | 現行の基準 |
| Qwen3.6-35B-A3B | MoE (hybrid GDN + MTP) | 同家系の MoE |
| Gemma 4 31B | Dense | 別家系の Dense。MTP 実装が Qwen 系と別物 |
| Gemma 4 26B-A4B | MoE | 別家系の MoE |

- M1: MoE の検証税カーブの実測（最初にやる。設計判断が変わる）
  MoE では m トークン一括検証が「m 個が踏むエキスパートの和集合」を読むため、
  Dense の「重みは m に依らず 1 回」という償却が弱まる。実データの routing で
  和集合サイズの m 依存を測り、MoE 向けの最適 m とゲート方針を決める。
- M2: MTP ローダの抽象化。fastmlx/mtp.py の Qwen 依存（重み名、+1 norm 規約、
  ブロック構造）をモデル別アダプタに分離。Gemma 4 の MTP 構造を調査して実装。
- M3: キャッシュ巻き戻しの抽象化。Gemma 4 の sliding window
  （RotatingKVCache は任意 prefix へ trim 不可）に対する投機巻き戻しの正しさを
  設計・検証。窓内 m トークンの trim で足りるかをまず確認する。
- M4: マトリクス 4 モデルで identity gate + ベンチを通す。

### Phase E: 計測・比較・公開準備

- E1: 正式ベンチプロトコル: 再起動直後・Spotlight 静止確認（mdutil -s）・
  電源接続・他プロセス最小、を bench/PROTOCOL.md に明文化して全数値を取り直す。
- E2: 競合 A/B: MTPLX、Rapid-MLX、mlx-lm PR#990、llama.cpp --spec-type draft-mtp を
  同一モデル・同一プロンプト・同一マシンで。各エンジンのバージョンを記録。
  Phase M のマトリクス 4 モデルで実施（対応していないエンジンは「非対応」と記録。
  それ自体が比較結果になる）。
- E3: fastmlx/fast_qmm.py の扱い決定（A2 完成後に削除、または上流ライセンス確認）。
- E4: 速度と品質（KLD）を必ず併記して公開する。MLX 界隈は品質ベンチ不在が
  課題と認識されており（生態系の当事者発言）、再現手順つきの計測公開自体が
  fastmlx の信用と差別化になる。上流へのマージは前提にしない
  （mlx-lm の外部カーネル・MTP PR が長期放置されている実績があるため、
  独立エンジンとして完結させる）。

## 5. 実装の契約（不変条件）

1. greedy 出力は非投機と分布厳密同一。カーネル変更で準同点 argmax の入れ替わりは
   許容するが、n_mismatch の位置と文脈を必ず記録し、系統的な劣化は不合格。
2. 検証幅 m∈{1..32} は可変。m をタイル幅にゼロパディングして KV に偽トークンを
   積むことは禁止（因果が壊れる）。
3. ArraysCache（線形状態）は前進のみ。巻き戻しは states_all の差し替えでのみ行う。
4. メモリ予算は実行時パラメータ。states_all は 48 層 × m × 3.1MB 消費する
   （m=32 で ~4.8GB）ことを API ドキュメントに明記。
5. 計測は依存チェーンと単発の両方を残す。独立呼び出しの積み上げ計測は禁止。
   GPU 計測は同時 1 プロセス。数値には計測条件を必ず併記。
6. mlx-lm 内部（fa_idx/ssm_idx、gated_delta_update シグネチャ、KVCache.offset、
   sanitize の norm +1 規約）への依存は fastmlx/_mlx_compat.py に集約し、
   mlx-lm のバージョン上限を pyproject に固定する。

## 6. Sol のオーケストレーション指針

- 並列化できる単位: A1、A2、C1、C2、D3 は互いに独立（ファイルも独立）。
  B は A の後。D1/D2 は B の後。
- サブエージェントへの分割は「新規ファイル単位」で行い、既存ファイルの同時編集を
  避ける。spec.py への統合は 1 エージェントに直列で担当させる。
- GPU 計測を伴う受け入れテストは直列キューで 1 つずつ実行する。
- 各項目の完了条件は本計画書の「受け入れ」基準。数値が出ない場合は
  計測条件（バックグラウンド負荷）を先に疑う。
- 迷ったら docs/RESEARCH.md と README.md の実測を正とする。仮説を追加測定なしで
  実装判断に使わない。

## 7. リスクと反転条件

- A2 で m=8 が 1.5 倍に届かない → fast_qmm 上流のライセンス確認に切り替え、
  vendored 利用の可否を先に確定する。
- D1 で受理率が改善しない（確信度が受理を予測しない）→ D2 優先に反転。
- C1 で group128 の品質劣化が体感可能 → 4.5bit 据え置き、C は MTP 同梱のみに縮小。
- 競合 A/B で MTPLX が総合で速い → 差分を計測で特定し、勝っている部品を
  設計として取り込む（コードはライセンス確認後）。
