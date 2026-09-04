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

## 判断 (advisor 2026-09-04 09:20、親が採用): **(B)。「spec_flash の最適化を持ち込む」ではなく「spec.py の round から同期と写しを剥がす」。(A) は今やらない**

- 汎用性が要るのは round のループではなくモデル側のシーム (capture / 状態スナップショット / 層呼び出し / MTP 頭)。`arch.py` の duck typing を族で揃えればエンジンが 2 本でも「8〜9 割ついていく」は満たせる。統合は 3 族目が同じ工数窓に入ったときに、動いている 2 実装から抽出する。
- 数字はエンジンの糊を指している: 4k で 99 ms/round = 素 48 + 51。相手 (23 ms/tok) の round は ≈ 48 + 17。**差の本体 30〜35 ms/round は同期点と写し**で、骨格の汎用化では減らない。
- **`spec.py:446` の `_linear_capture` は GDN 本体の写しで `la(...)` を呼ばない → 今朝の GDN 移植は S>1 の verify (主経路) に当たっていない疑い** (0902 の「capture が GDN 融合を素通し」の再発)。
- (A) は Flash-Next 側にだけ費用を払わせる (1 週間の A/B で調整済みの骨格を触る)。共有できるのは presync / prime / DepthController の族非依存部分だけで、それは写すのが安い。

順 (取り分の根拠つき): (0) 82/95/111 が draft 本数 (depth) の差なら 1 リンク ≈ 10 ms (帯域下限 1.5 ms) で draft chain が第 1 の的、(1) capture 経路の自前部品の素通しの是正 (ビット一致)、(2) 段階投入を S>1 に (`_hidden_forward` に層コールバックを足して `staged.py` の写しを消す、ビット一致)、(3) `mx.eval(*confidences)` の廃止 (挙動が変わる: 複数プロンプト平均の tok/round + KLD で判定)、(4) 次 round の draft の先行投入 (verify の eval → MTP 追いつき → 次 draft を async_eval → rollback + 後始末の順)、(5) capture の軽量化、(6) prime 窓は最後 (TTFT 側)。
反転条件: 内訳が「同期 + 糊 < 10 ms/round、draft chain < 8 ms」なら B の項目ごと捨てて帯域レーンへ。mlx-serve の 4k の tok/round が 4 以上なら差は draft の質 (rerank / depth 適応を先に)。rerank / depth-adapt / presync が単体で移せないと分かったら A。3 族目が同じ工数窓に入るなら A。
守り: `spec_flash.py` に入れる変更は `_arch()` の族解決の 1 点だけ (それも Flash-Next の 17k A/B の再走を掛ける)。共有ヘルパーは spec.py 側だけが使う状態で着地。ゲートは (1)(2) = 生成列の完全一致、(3)(4) = 複数プロンプト平均の tok/round + KLD、混ぜない。
