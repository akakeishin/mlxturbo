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
