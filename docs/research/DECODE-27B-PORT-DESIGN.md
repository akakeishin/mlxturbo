# 27B (qwen3_5) の decode 経路の移植: 事実と設計 (2026-09-04 09:05 起草)

## 実測 (M3 Max、MTP あり、貪欲、`tools/decode_ab_generic.py`)

| 文脈 | ms/round | tok/round | ms/tok |
|---|---|---|---|
| 短 case 0 / 1 / 2 | 82 / 95 / 111 | 2.17 / 3.46 / 4.74 | 38 / 27 / 23 |
| 4k | 99 | 2.84 | 35 |

素の 1 トークン forward の帯域下限 = 14.5 GB / 410 GB/s ≒ 35 ms。mlx-lm の素 48 ms/token。相手 (同じ MTP 頭、4k): mlx-serve 23 ms/tok (43 tok/s)、oMLX 30 (33)、MTPLX 35 (28)、mlxturbo 37 (27)。

## 2 つのエンジンの round の差 (scout 2026-09-04、path:line は台帳 `scratchpad/` の scout 出力と `spec.py` / `spec_flash.py`)

| 項目 | `spec.py` (SpecEngine、27B が通る) | `spec_flash.py` (Flash-Next) |
|---|---|---|
| draft | MTP を逐次 depth 回、**段ごとに confidence を eval + .item() で同期** (AdaEDL の打ち切り) | 逐次だが**同期ゼロ** (`async_eval` で次段を隠す) |
| verify | S>1 は `_hidden_forward(capture=True)` で**段階投入なし** (S==1 だけ `staged.py`) | 常に `_staged_forward` (層ごとの async_eval) |
| 受理判定 | eval 1 回 | 検証グラフに argmax を同梱して同期 1 回 |
| 巻き戻し | GDN の状態を毎 round **全層ぶん `states_all` に積んで**手動再構成 | `arch.py` の `rollback_recurrent` (duck typing、ただし `_arch()` が qwen4_exp 固定) |
| 次 round の draft | 先行投入なし | round 末尾で `async_eval` 先行投入 |
| prime | prompt 全長を MTP に通す (窓なし) | 窓 512 |
| depth | AdaEDL の confidence gate + 位置別 EMA | 静的 2 / 長文脈は DepthController |
| rerank / lookup | rerank なし / SAM lookup あり | rerank あり / lookup なし |
| 族依存 | `staged.py` は qwen3_5 型の呼び出し規約に決め打ち | `capture()` / `_staged_forward` / MTP 頭 (HC 型) / QSA cache が qwen4_exp 依存。骨格・presync・prime・DepthController は非依存 |

## 候補
- (A) `spec_flash` の骨格を族 adapter で汎用化して 27B を載せる。
- (B) `spec.py` に個別に持ち込む: 段階投入の S>1 適用、同期の 1 回化 (confidence の同期をやめる / 検証に同梱)、次 round の draft 先行投入、巻き戻しの capture の軽量化 (GDN の動的サブクラスに状態の取り出し口)、prime 窓。

判断は advisor の返答と round の内訳 (別途測定) を見て親が決める。判定は `decode_ab_generic` の ms/round・tok/round・head 一致、KLD (参照 = 素 4bit)、Flash-Next 側の fingerprint と A/B の不変。

## 判断 (advisor 2026-09-04 09:09、親が採用): **(B)。「spec_flash の最適化を持ち込む」ではなく「spec.py の round から同期と写しを剥がす」。(A) は今やらない**

- 汎用性が要るのは round のループではなくモデル側のシーム (capture / 状態スナップショット / 層呼び出し / MTP 頭)。`arch.py` の duck typing を族で揃えればエンジンが 2 本でも「8〜9 割ついていく」は満たせる。統合は 3 族目が同じ工数窓に入ったときに、動いている 2 実装から抽出する。
- 数字はエンジンの糊を指している: 4k で 99 ms/round = 素 48 + 51。相手 (23 ms/tok) の round は ≈ 48 + 17。**差の本体 30〜35 ms/round は同期点と写し**で、骨格の汎用化では減らない。
- **`spec.py:446` の `_linear_capture` は GDN 本体の写しで `la(...)` を呼ばない → 今朝の GDN 移植は S>1 の verify (主経路) に当たっていない疑い** (0902 の「capture が GDN 融合を素通し」の再発)。
- (A) は Flash-Next 側にだけ費用を払わせる (1 週間の A/B で調整済みの骨格を触る)。共有できるのは presync / prime / DepthController の族非依存部分だけで、それは写すのが安い。

順 (取り分の根拠つき): (0) 82/95/111 が draft 本数 (depth) の差なら 1 リンク ≈ 10 ms (帯域下限 1.5 ms) で draft chain が第 1 の的、(1) capture 経路の自前部品の素通しの是正 (ビット一致)、(2) 段階投入を S>1 に (`_hidden_forward` に層コールバックを足して `staged.py` の写しを消す、ビット一致)、(3) `mx.eval(*confidences)` の廃止 (挙動が変わる: 複数プロンプト平均の tok/round + KLD で判定)、(4) 次 round の draft の先行投入 (verify の eval → MTP 追いつき → 次 draft を async_eval → rollback + 後始末の順)、(5) capture の軽量化、(6) prime 窓は最後 (TTFT 側)。
反転条件: 内訳が「同期 + 糊 < 10 ms/round、draft chain < 8 ms」なら B の項目ごと捨てて帯域レーンへ。mlx-serve の 4k の tok/round が 4 以上なら差は draft の質 (rerank / depth 適応を先に)。rerank / depth-adapt / presync が単体で移せないと分かったら A。3 族目が同じ工数窓に入るなら A。
守り: `spec_flash.py` に入れる変更は `_arch()` の族解決の 1 点だけ (それも Flash-Next の 17k A/B の再走を掛ける)。共有ヘルパーは spec.py 側だけが使う状態で着地。ゲートは (1)(2) = 生成列の完全一致、(3)(4) = 複数プロンプト平均の tok/round + KLD、混ぜない。

## コード読みで確定した事実 (anatomy エージェント 2026-09-04 09:25、計測は列待ち)

- 段階投入は本番の decode に効いていない: `spec.py:1078-1096` は S>1 なら必ず `capture=True` の分岐で、`staged` は S=1 だけ。MTP がある限り S ≥ 2。
- 同期は 1 round に 2〜3 回: `mx.eval(*confidences)` (1067)、verify の `mx.eval(preds, window, ent_row)` (1108/1111)、D7 発火時の `mx.eval(d1)` (1019)。spec_flash は貪欲で 1 回。
- **draft を引いてから捨てる**: `cap_base = max_draft = 8` (本番既定) なので最大 8 本引いてから `_gate_depth` が事後に切る (1053-1071)。各本に lm_head (語彙 248,320 × 5120、4bit ≒ 0.64 GB) の射影が付く → 8 本で読みだけで ≈ 12 ms + MTP 層。「1 リンク ≈ 10 ms」の正体の候補。
- maint (rollback と MTP の積み直し) に同期が無いので、その GPU 費用は次 round の draft の同期に乗る (相の帰属が歪む)。
- `_gate_depth` の rollback コストは固定定数 0.19 (spec.py:63)。prime / presync / DepthController は spec_flash 側だけ。
- 道具: `tools/decode_round_anatomy_generic.py` (spec.mx の proxy で eval の回数と待ち、5 メソッドを包んで相ごとに sync / build / glue、GPU は `decode_gpu_trace.Probe`)。結果は `bench/results/round-anatomy-27b-0904.json` / `scratchpad/anatomy-27b-0904.log`。

## 計測 (anatomy 2026-09-04 09:36、短文脈 case 0 のみ。以降は `staged` の import で落ちた = 第 1 段が `staged.py` を消している最中)

| | 値 |
|---|---|
| round (clean) | **83.6 ms**、tok/round 2.21、72 round |
| draft (MTP 連鎖 + lm_head 射影 + 同期) | **21.5 ms (25.7%)** |
| verify (S ≈ 4 の trunk forward + 受理) | **61.9 ms (74.1%)** |
| maint (rollback + MTP 積み直し) | 0.2 ms (同期が無いので次 round の draft に乗る) |
| 非投機 (S=1、MTP / lookup 無し) | **42.4 ms/token** (投機は 37.9 ms/tok → **投機の利得は 1.12 倍しか無い**) |
| lookup 無し (MTP だけ) | 83.1 ms/round (lookup の費用 +0.5 ms) |
| 発火 | `rms_norm_gated` 3456 = 48 層 × 72 round (S=1 の経路だけ。S>1 の verify には当たっていない) |

読み: 素の 1 トークン forward 42 ms (帯域下限 35 + 糊 7)。verify は S≈4 で 62 ms = 42 + **行の費用 20 ms** (GDN の再帰が行ごとに逐次 48 層 × 4、`_linear_capture` の写しで自前部品なし、状態の控え)。draft は 4 リンクで 21 ms = 1 リンク 5 ms (lm_head 0.64 GB の読み 1.5 ms + MTP 層 + 同期)。**的の順: verify の行の費用 (20 ms、第 1 段) → draft の同期と本数 (21 ms、第 2 段) → S=1 の糊 (7 ms)。**mlx-serve の 23 ms/tok は tok/round 2.2 なら round 50 ms 相当 = うちの verify 62 ms より軽い。

## 第 1 段の結果 (2026-09-04 10:15、CATCHUP 同時刻)

round ≒ **43 ms + 10 ms × draft 本数**。mlx-serve は draft 2 本で 52 ms (24.4 ms/tok)、うちは同じ 2 本で 73〜75 ms。差 20 ms/round のうち固定費 +8 (43 対 35)、リンク 1 本 +8 (10 対 下限 2)。段階投入 (S>1) は -1.9% で既定 on。capture の写しの素通しは前処理だけで、当てても -0.3% (幅 5+ で数 ulp ずれる穴あり)。
第 2 段: draft chain の同期 (`mx.eval(*confidences)` と各リンクの `.item()`) の廃止、引いてから捨てる本数 (max_draft 8 → 幅表 / EMA で先に決める)、次 round の draft の先行投入、固定費 +8 の内訳。判定は複数プロンプト平均の tok/round と ms/tok、KLD (参照 = 素 4bit)。
