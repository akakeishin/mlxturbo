# Fable 5.1 xhigh 性能監査の統合

記録時刻: 2026-09-05 08:33:21 JST

`claude -p` の Fable 5.1 xhigh を読み取り専用で 3 本走らせた。担当は、全体の費用分解、Gemma 4 26B の B=2、全モデル 100 tok/s の現実性。以下は回答の転載ではなく、リポジトリの実測と照合した採否である。

## 結論

「全般的に遅い」を一つの共通バグで説明することはできない。

| 系統 | 現在の主因 | 次に値段が大きい仕事 |
|---|---|---|
| Flash-Next | 1 round 約 3,600 dispatch と stateful graph 境界。現方式は帯域・構築泡の天井に近い | state-pure adapter から fixed-M4 graphbank へ進む |
| Qwen3.8 27B | AR は競合と同着。差は draft の取り分と小幅 verify | 同じ物差しで投機統計を比較し、長文 cap と GQA 小幅 attention を測る |
| Qwen3.6 | 短文/4k は 100 tok/s を超える。長文低下の約 90% は verify の KV/SDPA | GQA 小幅 attention。専用直読みが成立する場合だけ量子化 KV を再検討 |
| Gemma 4 26B | AR の dispatch 税、MoE の行間共有限界、同時要求の初回合流 | 2 ms cohort window を既定化し、B=2 aggregate を回収する |
| Gemma 4 31B | dense target 用 BF16 shared-KV assistant が未接続 | B=1 exact verify を先に通し、block 2/4/6/8 を測る |

100 tok/s は全モデル共通の合格線にはしない。Qwen3.6 短文と Gemma 26B 短文では達成可能だが、27B と Gemma 31B の単一ストリーム、現方式の Flash-Next では重み帯域から非現実的。これらは「競合より速い」「以前より速い」「aggregate throughput」を別の軸として示す。

## Gemma B=2 監査への訂正

Fable 2 本は、事前に渡した B=1 88.3 tok/s、B=2 aggregate 88.5 tok/s を本物の continuous batch 測定として扱い、fp32 昇格や batch mask を第一容疑にした。この入力が誤っていた。

旧クライアントは `ignore_eos=true` を付けていた。`mlxturbo/server.py` はこの指定を batch coordinator へ流さず、直列経路へ落とす。従って 88.3→88.5 は batch の速度ではない。

通常 EOS、128 completion tokens、同じ常用冷却で取り直した結果:

| 条件 | B=1 | B=2 aggregate | B=2 / B=1 |
|---|---:|---:|---:|
| cohort wait 0 ms | 88.88 tok/s | 63.30 tok/s | 0.712x |
| cohort wait 2 ms | 89.52 tok/s | 136.02 tok/s | 1.519x |

2 ms では 2 件とも最初の MLX tick 前に入り、各要求の server-side decode は約 74 tok/s だった。0 ms では 2 件目が途中参加し、各要求は約 34.5 tok/s まで落ちた。単発の client wall は 1.44 s と 1.43 s で、2 ms の代金はこの短い測定の揺れ未満。

従って、ここで採る修正は dtype の推測修正ではなく、pool cohort の開始時だけ 2 ms 待つこと。fp32 昇格は長い/padded prefill の別問題として残るが、今回の B=2 aggregate 停滞の説明には使わない。

## 優先順位

1. Gemma continuous batch の 2 ms cohort window を回帰テスト付きで採る。
2. Gemma 31B shared-KV assistant の exact B=1 を完成させ、短文/4k/17k と block 2/4/6/8 を測る。
3. 27B と Qwen3.6 に共通する GQA 小幅 decode attention の達成率を上げる。
4. 27B は競合と投機統計を同一条件で並べ、既に二度負けた controller 移植を繰り返さない。
5. Flash-Next は QSA 1 層 state boundary gate の次へ進み、fixed-M4 graphbank を小さなゲートに分ける。

小さな代金ゼロ候補として、verify 幅 6..11 の M-padding と Flash-Next prefill の `_segments_gpu` 重複解消は A/B の価値がある。Gemma S=1 compile は形状が固定なので過去の可変幅 MoE compile 棄却をそのまま外挿せず、短い A/B 一回で閉じる。

## 停止線

- 2 ms cohort window は B=1を悪化させず、B=2 aggregate が 1.3xを下回らないこと。
- 新しい attention kernel は micro だけで採らず、27B 17k と Qwen3.6 50k の in-model で非退行を確認する。
- Gemma 31B assistant は greedy token 一致、rollback、block 掃引を通すまで既定化しない。
- graphbank は state/logits/cache の一致ゲートを一段ずつ通し、巨大な一括移植にしない。

## 16:04 再監査: 現存artifactの上限と棄却案の再審

最新`c3efa83`を対象にFable 5.1 xhighを読み取り専用で3本走らせた。担当はFlash単体、
全モデル共通、棄却案の再審。ユーザー判断により自作MTPやモデル固有drafterの学習は当面優先しない。

- Flash単体: 強冷却の現行は35.26 ms/round、2.186 tok/round、61.4 tok/s。既存artifactだけで
  100 tok/sへ届く大口は無い。理想proposal時の検証幅別round費用を測り、物理上限を先に確定する。
- 全モデル共通: Qwen 27B/35Bの`_POS_ACCEPT_PRIOR`はFastMTP資料由来で、このpackの実測ではない。
  現行NOSYNC経路の位置別採択率を保存し、priorとの差が±0.05以内なら閉じる。差が大きい場合だけ
  prior較正を製品A/Bする。期待上限は3〜8%だが未実測。
- 棄却再審: 追加で復活させる案は0件。木はGDN/PLEの逐次stateが兄弟行を表現できず、MoE compile
  汎用化は0.5%級に対してgraph保持の代金があり、prefill chunk 4096の取り分はlayer-majorで回収済み。

Fableが提案した旧`oracle-draft`の直接再利用は採らない。この道具は検証幅でtarget生成列が変わり、
保存済みoracle位置がずれて受理されないことを既に2回確認している。代わりに通常のdepth掃引から
`(depth+1) / ms_per_round(depth)`を計算し、完全受理時の上限だけを求める。

速度目標は100 tok/s一律から、現行artifactの実測上限と同条件競合比へ変更する。

| 対象 | 現在 | 第一目標 | 根拠 |
|---|---:|---:|---|
| Flash短文・強冷却 | 61.4 tok/s | 65 tok/s持続 | 残存runtime機構の実測上限がおおむね+5% |
| Flash 17k・強冷却 | 53.1 tok/s | 55 tok/s | QSA採用後から約+3.6% |
| Qwen 27B 4k | 32.1 tok/s | 同条件43.2 tok/sへ接近 | 同じMTPを使う比較先との差が投機制御側 |
| Qwen 3.6 / Gemma 26B | 既に100前後以上の条件あり | 長文・aggregateの現行比+5%以上 | 絶対100より文脈別退行を減らす |

100 tok/sは、将来の公式assistantや新architectureで受理率・round費用の上限が変わった場合の
stretchとして残す。公開値は同一pack、prompt、出力長、sampling、冷却で競合を横並びにして決める。
