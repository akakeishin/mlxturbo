# KERNEL BRIEF: decode の持続帯域 160 -> 206GB/s (mlx-serve 超えの残り全部)

2026-08-31。docs/research/DECODE-ANATOMY-2026-08-31.md の続き。ここは
「何を作れば超えるか」を、実測予算つきで着手可能な形にする。

## 予算 (全部このセッションの実測)

- 目標: 同一 4bit 重みで mlx-serve --mtp の **62.1 / 51.7 tok/s** (短/長) を超える
- 現在: 53.6 / 39.8。ラウンド = draft 5.3 + verify 39.0 + 糊 1.3 = 45.6ms で 2.44 トークン
- 必要: 短はラウンド **39.3ms 以下** (-6.3ms)、長は **39.1ms 以下** (-11ms)
- 物理: T=3 検証の読みは ~6.7GB。393GB/s (逐次読みの実測ピーク) なら 17ms、
  mlx-serve は実効 206GB/s で回し、うちは **~160GB/s** (GDN 165 / lm_head 164 /
  全体 4.08GB/28ms=146)。**差の正体は依存チェーン下の持続読み出し帯域**

## なぜ Python 層では届かないか (再訪防止)

- カーネル数削減の単価は 2-8us/本 (wide 連結・HC 融合・moe_route の実績)。
  6ms には千本単位が要る
- 個々の op は「前の op の出力待ち」でレイテンシが露出する。読みの深さ
  (in-flight リクエスト数) はカーネル内部の構造で決まり、op の並べ替えでは
  変わらない
- 温キャッシュのマイクロは常に楽観 (fast_qmm の罠)。判定は in-model A/B のみ

## 作るもの (優先順)

### 1. MoE ブロックを 2 ディスパッチに (取り分 ~3ms/T=3)

現状 1 層 ~10 ディスパッチ (router qmm、topk 7op、gather x3、shared x4、和)。

- K1 `moe_route+`: router qmm (512x2560 4bit) + top-10 選択 + softmax +
  shared gate。既存 kernels/moe_route.py (選択部のみで純損だった) に
  **router の行列積ごと**入れて 1 本にする。出力: idx (T,10)、w (T,10)、sg (T,1)
- K2 `moe_glu_down`: gate+up を 1 タイルで読み silu*mul、続けて down を
  threadgroup 内で消化して w 付きで fp32 atomically に加算。中間 (T,K,640) を
  デバイスメモリに書き戻さない。エキスパートあたり中間 640 要素 = TG メモリに収まる
- shared expert は K2 に 513 番目として同乗 (Python 版は gather が太って純損
  だったが、カーネル内なら行の追加はタダ)

### 2. GDN ブロックを 3 ディスパッチに (取り分 ~2ms)

in_proj 4 本 + conv + silu + rms x2 + delta + norm-gated + out_proj (~12 本) を
[wide-proj+conv+silu+rms] / [既存 gated_delta] / [norm-gated+out_proj] に。
wide-proj は Python 連結だと qmv 変種が変わるだけだったが、専用カーネルなら
8 行 MMA タイルで読みを 1 回にできる (fast_qmm の M=6..8 実績 3.4 倍が根拠)。

### 3. 検証幅の lm_head (取り分 ~1ms)

318MB を M=3 で 1.94ms (164GB/s)。fast_qmm は K=2560%512=0・N 巨大で適格なのに
in-model で負けた — zpad の concat 起動と split-K の読み順が原因候補。
zpad をカーネル内へ、読み順を行連続に。

## 着手済み: moe_glu の足場と初期実測 (2026-08-31)

`mlxturbo/kernels/moe_glu.py` に gate+up+silu*mul の 1 ディスパッチ版を置いた
(未配線、どこからも呼ばれない)。検算は通る (相対 1e-2、bf16 の丸め相当)。
速度は温キャッシュの 48 層直列 (T=3、30 対) で:

| 版 | ms |
|---|---|
| 素 (gather_qmm x2 + swiglu) | 4.7-4.9 |
| 融合 v1 (スカラー nibble 展開) | 7.4 |
| 融合 v2 (uint4 + unroll) | 6.7 |

v4 まで進めた。v3 = qmv_fast_impl (mlx quantized.h) の構造 (simdgroup 4 行、
スレッド 16 値レジスタ常駐、バイアスは sum(x) に畳む)。v4 = さらに x を
vec<T,4>、重みを uint2 でロードし simdgroup 2 本に (qmv_fast と同一構成)。
単体 (温キャッシュ): T=1 は素の gather+swiglu に 26% 勝つ (2.07 vs 2.80ms/48層)
が **T=3 で 1.4 倍負けたまま** (7.06 vs 4.98)。スケーリングが線形 (対数) で、
素の gather は T=3 で 1.78 倍にしか伸びない。ランダム添字の温キャッシュでも
この差が出るので、dedup ではなく占有率かタイル形の差。**次の一手は推測でなく
Xcode の Metal capture で素の gather の実行形 (行/対の割り付け、simdgroup 数、
read 幅) を見ること。**盲目の反復は禁止 (measurement-discipline)。温キャッシュの単体では
T=1 で素に勝つ (2.21 vs 2.78ms) が T=3 で負け、**in-model では短 +3% 遅 /
長 -5%** (verify 43.9 vs 42.6ms)。負けの構造は明確: このカーネルは対ごとに
エキスパート行を独立に読むが、MLX の gather (+ソート) は同一エキスパートを
引く対をまとめて重みタイルを使い回す。つまり融合で勝つには **ソート済みの
対グループ単位で 1 エキスパートの gate/up タイルを 1 回読み、複数の x 行に
適用する** (= gather の機構ごと再実装) が要る。これが次のカーネルセッションの
関門。v1-v3 は mlxturbo/kernels/moe_glu.py に残してある (MLXTURBO_MOE_GLU=1
で配線されるが既定 off)。

## ラウンド間の泡 7.3ms: 実在するが、楽観先組みでは取れない (2026-08-31)

xctrace (Metal System Trace は CLI で取れる。`xcrun xctrace record --template
'Metal System Trace' --attach <pid>`、export は schema=metal-gpu-intervals) で
decode 中の GPU タイムラインを取ると、**5ms 超のアイドルがフォワード回数と
同数 (223 回/8 秒)、平均 7.3ms** — ラウンド間で CPU が次のグラフを組む間、
GPU が止まっている。ラウンドの ~16%。

これを「全採用を仮定した次ラウンドのグラフを verify の GPU 実行中に先組み」
(cur を argmax の遅延配列のまま使う) で取ろうとしたが、**逆に -28〜40%**。
切り分け (組んで毎回捨てるモードが最遅) により、毒は使い方でなく**組むこと
自体**: 捨てたはずの遅延サブグラフが評価される (MLX はグラフが深くなると
暗黙の全評価を走らせる)。トークン列は全モードで一致しており正しさは無傷。
コードは MLXTURBO_PIPELINE=1/2 で残してある (既定 0)。

泡を取る残りの道: (a) グラフ構築自体を軽くする (mx.compile の shapeless 化、
offset を配列入力に変える改修が前提)、(b) MLX 側の暗黙 eval 閾値の制御。
どちらも MLX 内部仕様の調査が先。

indexer のプールブロックをラウンド間でキャッシュする案も試して棄却 (2026-09-01
未明): 長文 50.2 -> 48.4 と逆行し、tok/step も 1.98 -> 1.90 と動いた (どこかで
値がずれており「差分プール = ビット同一」の主張が破れている)。差分 concat の
連鎖はどのみち速くない。revert 済み。

## 受け入れ基準 (BRIEF の規律)

- in-model A/B (サーバー経由、同一プロンプト、複数プロンプト x 512 の平均 tok/step)
- KLD: bench/quant_eval.py compare で kld_mean が現行比 +0.0005 以内
- 単体マイクロは案の優劣にのみ使う。絶対値は信じない

## 見込みの正直な評価

1+2+3 が全部予算どおりでも短 ~60-62 / 長 ~44-46。**短は際どく、長は届かない
可能性が高い。**長を締めるには T=2 経路 (depth1) の同じ融合と、indexer の
ブロックプーリングをラウンド間でキャッシュする改修が追加で要る。
