# mlxturbo で作業するときの決まり

対 mlx-serve の性能レーンが主戦場。現在地と次の一手は
`docs/research/KERNEL-BRIEF-DECODE-BW.md` の末尾が常に最新。着手前に必ず読む。

## 計測の作法 (破ると数字が嘘になる。全部実測で確立済み)

- A/B は 1 プロセス内で交互に測る。プロセスを分けた比較は熱・キャッシュ状態で数 % ずれる。
- tok/step (受理率) の比較は複数プロンプト x 512 トークンの平均でだけ行う。
  単一プロンプトは chunk 境界の丸め 1 つで挙動が変わる (テキスト運)。
- 温キャッシュのマイクロベンチの絶対値を信じない。案の優劣の目安にだけ使い、
  採否は in-model A/B で決める (micro 勝ち in-model 負けの前例が複数ある)。
- 無効化の積み上げで部品時間を見積もらない (ablate の積算は過大評価の前例あり)。
  部品計測をしたら「部品和 ≈ 壁時計 (数 % 以内)」を必ず確認する。
- 計測中にダウンロード・別 GPU プロセスを並走させない (335GB ダウンロード並走で
  decode が 21.4 tok/s に落ちた実測がある)。
- 生成長を揃えて比較する (相手 121 トークン vs 自分 512 トークンの窓で
  偽の同着を出した前例)。判定基準は測る前に宣言する。
- サーバーログの phase/round の draft/verify 内訳は帰属が歪んでいる
  (next_drafts の先行投入のため)。合計 ms/round だけ信じる。

## 触ると壊れるもの

- `mlxturbo/spec_flash.py` の `_staged_forward` / `_group_prefill_forward` /
  `capture()` は `_vendor/qwen4_exp.py` の Model.__call__ の写し。
  **本家を変えたら写しも全部変える。**変更後は
  `tools/verify_prefill_bitident.py` で旧経路とのビット一致を確認する。
- 「作って捨てる」遅延グラフを組まない。捨てる可能性のあるグラフは規模を
  問わず MLX の暗黙 eval に罰される (楽観先組みは 3 回失敗して棄却済み)。
- 既定 off の knob (MLXTURBO_PIPELINE / MOE_GLU / WIDE / PREFILL_CHUNK など) は
  実測で負けて off にしたもの。有効化するなら棄却時の記録
  (docs/research/DECODE-ANATOMY-2026-08-31.md) を先に読む。
- 品質を売って速度を買わない。fake を実物より緩くしない。KLD の受け入れ幅は
  現行比 +0.0005 (bench/quant_eval.py compare)。

## 分業

実装・検証はサブエージェントに出してよいが、計測の判定と commit は親が行う。
HF トークンは huggingface_hub の既定の場所のみ。引数・ログ・環境変数に出さない。
