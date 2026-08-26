# カーネル設計インテリジェンス（2026-08-26 採掘、A2 v3 の正本）

出典はすべて一次ソース確認済み。MIT 参照コードは tools/reference/e120/ に
LICENSE ごと同梱（Layr-Labs/qwen-3.8-mtp-challenge、フロンティア 3.72x）。

## E120 カーネルの設計（v3 はこれに寄せる）

- rows_per_simd=4（出力行）、values_per_thread=16、block_size=512、
  threadgroup (32,2,1)、grid (activeInputGroups(M)*32, (N/8)*2, 1)
- threadgroup メモリゼロ・barrier ゼロ・simdgroup_matrix 不使用。
  縮約は simd_sum のみ、状態は全部レジスタ
- 蓄積器は vec<float,NA>（M 方向がベクトル幅）
- dequant を内ループから排除: 生 nibble のまま整数積和し、512-K ブロック毎に
  1 回だけ acc += scale*partial + sums*bias の affine 補正
- sums（活性化のレーン内 16 要素和）は出力行に依存しないため、活性化毎に
  1 回だけ別カーネルで表を作り全 matvec が読む（USE_TABLE、M>=4 で有効）。
  表の ON/OFF は加算順同一でビット一致
- M=8 は「4 行 x 2 グループ」に分割（8 行/スレッドはやらない）。重み 2 回読みでも勝つ
- フロンティア PR は表生成を RMSNorm カーネルへ融合（ラッパ税 H2 の解答例）

## 実証済みの禁止事項

- 8 行/スレッド + 32 蓄積器: MTPLX が CLOSED BRANCH (0.51-0.87x)。
  蓄積器は 24/スレッドが実証上限
- K/N をテンプレート定数にするな: E120 で K 展開時にコンパイラが誤答を
  出した記録あり（174,080 出力中 174,072 不一致）。K/N はランタイム値で渡す
- split-K + threadgroup partials: E120 は不使用。MTPLX も split-K を
  単純 simd_sum へ書き換えて +5〜27%
- 幅広ロードもグループ整列レーンも回帰した記録あり。pack-interleave の
  32bit ロードが coalescing の要（MTPLX verify_kernels.py 冒頭）

## 深度制御（Phase D1 で使う）

位置別受理率の EMA + 期待利得閾値（MIT、Qwen36MTPBlockSession.swift:1117-1143）:
reach *= p_d、threshold = h*(1+expected)/(1+d*h)、reach<=threshold で停止。
h は巻き戻しコスト依存: repair-forward 型は 0.43、checkpoint 型
（fastmlx の全位置状態保持はこちら）は 0.18-0.20。
m カーブの段差には「深度価格を位置ごとのベクトルにする」対処が同ファイルにある。

## GDN 状態の将来最適化

innovation tape（delta [B,T,Hv,Dv] fp32 のみ記録、replay は
state = state*g + k_norm*delta の axpy）で全状態保持より Dk 倍小さい。
m を 16 以上へ広げる時に採用検討（現行 m<=8 では不要）。

## Phase C の初期レシピ（MTPLX Optimized-Speed 実値、KLD 0.022）

- 8bit/g64 へ昇格: lm_head、embed_tokens、全 48 層の linear_attn.out_proj、
  末尾 8 層 (56-63) の mlp.gate/up/down
- 本体 4bit/g32、GDN conv・recurrent state・全 norm は bf16 据え置き
- 実効 5.807 bit/weight (20.4GB)。帯域レバー（4.5bit 化）とは逆方向の
  トレードオフなので両方測る
- full attention 層は i mod 4 == 3（3,7,...,63）で確定

## 正しさの地雷（検証済み）

- mlx-lm sanitize は mtp.* キーが存在するだけで本体 norm を +1 汚染する
  （受理 0-2% への崩壊実例）。現行ロード経路は 1 テンソル比較で汚染なしを
  確認済み (diff 0.0)。convert 成果物は Phase 0 の raw 規約保存で防御済み
- MTP 契約実値: base_hidden_variant=post_norm、hidden_variant=post_norm、
  concat_order=embedding_hidden、mtp_position_mode=local
- MTP ヘッドの INT4/g64 量子化は「非量子化比で受理フラット以上」の検証記録あり

## 同一ハード比較（M3 Max、MTPLX Issue #286/#293）

- MTPLX depth3: decode 41.4 (1K) / 31.6 (8K) / 19.8 (16K)、prefill 198 (1K) /
  115 (16K、長文脈で崩れる)。fastmlx は短文脈 decode 互角、prefill は既に上 (219)
- 同筐体で 27B を回し続けると約 30% サーマルスロットルしたという報告が
  Issue #293 にあり、fastmlx が README に書いた計測規律と同じ結論になっている

## ライセンス

- E120 (Layr-Labs + vendored mlx-swift-lm): MIT。表示は LICENSE 同梱で足りる
- MTPLX: Apache-2.0 + NOTICE（製品内表示の要求あり）。コードは借りない。
  設計事実の参照のみ
- fastmlx/fast_qmm.py: 出所ライセンス未解決のまま。E120 移植が完成したら削除
