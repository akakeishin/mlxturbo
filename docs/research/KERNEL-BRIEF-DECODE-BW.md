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

## prefill MoE の予算 (2026-09-01、カーネルレーンの第 2 の的)

17k prefill 37.5s の最大部品は MoE 12.9s。mx.gather_qmm (sorted、実体は
affine_gather_qmm_rhs = steel BlockMMA + セグメント切替) は**セグメントの
太さに単調**: 行/expert 20/40/80/160/320 で 5.7/7.5/8.9/9.8/10.3 TFLOPS
(密 qmm の上限 11.2)。チャンク幅を上げれば太くなるが、in-model では
attention/indexer の一時増と相殺して -2% 止まり、8192 は wired 張り付き構成で
Metal OOM。gate+up 融合バンク (N=1280) は効かない (7.63 = 分離と同値)。

**書くべきカーネル**: gather_qmm_rhs のセグメント処理を 2 段タイル化し、
細いセグメント (40 行/expert) でも BlockMMA の稼働率を保つ。到達目標は
S=2048 で 10 TFLOPS (現 7.5)。当たれば prefill -3.5s (17k 454 -> 500+)。
decode 側の共有タイル gather (関門は上記) と同じファイルの兄弟カーネル。

**保留 (2026-09-01、advisor 判断)**: 上のカーネルは書かずに保留。同じ的を
Python だけで撃てる layer-major prefill (レイヤー主導で G チャンクの MoE を
concat して1回で流し、attention/indexer は 2048 のまま) を先にやる。
r=40->40G で標準カーネルのまま効率曲線を登る。根拠: (1) 律速はタイル内
expert 境界のフル GEMM やり直し (quantized.h の affine_gather_qmm_rhs、
BM=32) で、eff ≈ 1/(1+32/r) が実測曲線と整合、(2) gather_qmm は行独立で、
BM=32/64 をまたいだ分割でもビット一致を micro で確認済み → A/B が
テキスト運と無縁、(3) mx.fast.metal_kernel は steel ヘッダを include
できない (ヘッダ平坦化の下ごしらえが必要) と判明し、カーネル案の初期費用が
想定より高い。**反転条件**: chunk 4096 の in-model MoE 部分時間が
12.9s から予測 (~10.9s) の半分も落ちない / G=2 で wired 構成が OOM /
17k 実プロンプトのビット一致が破れて検証戦略が KLD に落ちる — いずれかで
カーネル案に戻る。layer-major が入って r=160+ に達したら、この
カーネルの上積みは ≤8% に縮むので、その時点で本節のカーネルは棚上げ確定。

**決着 (2026-09-01)**: layer-major は入った (spec_flash._group_prefill_forward、
MLXTURBO_PREFILL_GROUP 既定 4)。反転条件はどれも発火せず: in-model の
MoE 部分時間 18.4→14.9s (chunk 4096 の観測)、17k 実プロンプトで
logits/hyper/全キャッシュ 110 配列ビット一致、wired+ngramRAM のサーバーで
OOM なし。17k TTFT はサーバー実測 34.49→32.40s (493→524 tok/s)、
in-process では 41.5→37.0s。4k 以下と decode は不変 (グループ条件が
効かない長さ)。**prefill 面の segment-aware GEMM カーネルは棚上げ確定**
(r=160 で eff 87%、カーネル天井までの残りは ≤8%)。decode 面の共有タイル
gather (verify の MoE 13.8ms) は別問題として残る。

## 最終スコアボード (2026-09-01、同一セッション、両者最新)

mlx-serve は main ソースビルド 3afb77d (26.9.1-dev)。順序バイアスは往復で
検証してほぼゼロ (彼らは後攻でも eq 18.84 = 冷スロットと同値)。

| セル | mlx-serve | mlxturbo | 差 |
|---|---|---|---|
| short decode | 15.8-15.9 ms/tok | **14.6** (mtp-bits4) | **+8% 勝ち** |
| 4k 等窓 121tok | 18.85 ms/tok | 19.74 (mtp-bits4) / 20.65 (rebit併用) | -4.5% 負け |
| 17k decode | 43.6-49.3 (要求ごとに劣化) | 42.9 (mtp-bits4) / 46.3 (rebit併用) | ほぼ同着 |
| 17k prefill | 27.0-28.0s (608-633 tok/s) | 32.4-32.9s (517-524) | **-17% 負け** |

読み方の注意 3 つ:
- 構成またぎの単一プロンプト比較はテキスト運入り (rebit は verify logits を
  変え tok/step が動く)。rebit 併用が 17k decode で +3.4 tok/s は方向として
  実在しそうだが、採否は複数プロンプト x 512 の平均を取ってから。
- 彼らの 17k decode はサーバー寿命内で要求ごとに 49.3→45.5→43.6 と落ちる
  (再現 2 回)。うちは 42.9-46.3 で安定。持久戦ではうちが並ぶ。
- MLXTURBO_PHASE_TIMERS の draft/verify 内訳は next_drafts の前ラウンド末尾
  async 投入以降、帰属が歪む (draft バケット冒頭の eval(drafts) が同じ
  コマンドバッファの verify 層完了まで待つ)。合計 ms/round だけ信じること。

prefill の残差 -17% の解剖: layer-major 後の MoE は ~9.9s で、密天井まで
搾っても残り -1.3s。彼らとの差 ~5s の主体は attention/GDN/その他の prefill
カーネル側 (decode で実証済みの 1.4x 速い基礎カーネル群と同根) にある。
つまり prefill 並走には彼ら級のカーネルを面で書く必要があり、MoE 一点では
届かない。目標を維持するか降ろすかはユーザー判断待ち。

## 受け入れ基準 (BRIEF の規律)

- in-model A/B (サーバー経由、同一プロンプト、複数プロンプト x 512 の平均 tok/step)
- KLD: bench/quant_eval.py compare で kld_mean が現行比 +0.0005 以内
- 単体マイクロは案の優劣にのみ使う。絶対値は信じない

## 見込みの正直な評価

1+2+3 が全部予算どおりでも短 ~60-62 / 長 ~44-46。**短は際どく、長は届かない
可能性が高い。**長を締めるには T=2 経路 (depth1) の同じ融合と、indexer の
ブロックプーリングをラウンド間でキャッシュする改修が追加で要る。

## 2026-09-01 夕: 残弾の棚卸しと解剖の決着 (advisor 監査込み)

**prefill「残り 8.6s」は解剖済み。**部品和 41.2s vs 壁 42.7s (3.6% 以内、
帰属ゴーストなし)。eval 障壁つき部品時間 (17k, layer-major 有効):
moe=15.0 / gdn=9.4 / attn=9.0 / **hc=5.5** / ple=2.3。**MTP priming は 0.12s
で容疑消滅。**HC は kernel/eager とも M=2048 で 3.8ms/call と同格で、帯域
理論値 (~0.3-0.6ms) の約 10 倍遅い — ここが**カーネル面④** (取り分 ~3s、
elementwise + 低ランク matmul の融合で、attention/GDN より書きやすい)。

**H (相手の lossy フラグ) は測る前に決着。**ddalcu パックは attention も
U32 (4bit 済み) で、--decode-attn-quant は dense attention 専用ゆえ no-op。
--ane-prefill は opt-in 未使用。**彼らのカーネルは同じ量子化重みで本当に
速い** — カーネルレーンの正当性は本物。品質を売って追わない方針は不変。

**decode 共有タイルの取り分上限を実測。**verify 3 トークンの top-10 union は
21.1-22.5/30 (重み読み -25〜30%)。素の経路は gather_qmv(_fast) で行ごとに
重みをフルストリームしており共有ゼロ (ディスパッチ特定は
mlxturbo/kernels/moe_verify_gather.py の docstring)。v1 (gate 単体) は micro
互角 — 極小サイズでは帯域でなく占有率律速の疑い。勝敗は in-model でのみ判定。

**warm TTFT (2 ターン目追記) は 4〜8 倍負けで実在。**1k: 2.66s vs 0.66s、
16k: 6.14s vs 0.73s。完全再送 (diff-0) は 0.36s vs 0.18s で小差。修正レーン
進行中 (照合破れ vs checkpoint 粒度の診断から)。

### 進行中 / 待ちのキュー

1. Sonnet ①: warm TTFT の診断と最小修正 (目標: 追記ターン 1s 未満)
2. Sonnet ②: 共有タイル v2 (gate+up 融合 + down K=640 対応 +
   MLXTURBO_MOE_VERIFY 実験配線、既定 off)
3. bf16 (Qwen/Qwen3.8-Flash-Next, 360GB) を外付けへ再取得中 — 完了後に
   tools/bake.py で lm_head 4bit を真 bf16 から焼く (rebit head=4 の
   二重量子化 +0.0054 KLD を解消し、17k decode +3.4 tok/s 相当を無償化)
4. ダウンロード完了後のクリーン計測: v2 の in-model A/B、warm TTFT 修正の
   確定値、depth 再掃引 (verify が安くなったら depth3 の損益が反転しうる。
   tok/s と KLD を同じ表に出す)
5. ~~ユーザー判断待ち 2 件~~ **決定済み (2026-09-01 ユーザー宣言)**:
   (a) 勝利条件は**素の速度** — 品質正規化は看板から降ろす。KLD 3 倍勝ちは
   商品性として維持するが、勝敗判定には使わない。速度を買える lossy 手
   (lm_head 4bit 等) は KLD 実測つきで採用検討してよい。
   (b) バッチ x 投機の共存は**本気でやる** — FallbackRunner 限定バッチの
   妥協はしない。投機エンジン (FlashSpecEngine) と継続バッチングの共存が
   次の大レーン。

### 計測中のダウンロード並走は禁止 (再掲)

いま 360GB 再取得が走っている。完走まで時間計測の結論を出さない。
