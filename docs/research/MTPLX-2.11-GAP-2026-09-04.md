# MTPLX 2.11.1 と mlxturbo の差分

2026-09-04 21:21 JST 時点。比較対象は
[MTPLX 2.11.1 のリリースノート](https://github.com/youssofal/MTPLX/releases/tag/v2.11.1)と
[PR #391](https://github.com/youssofal/MTPLX/pull/391)。手元の独立環境も
`mtplx 2.9.2` から `2.11.1` へ更新し、wheel の実装を読んだ。

この文書の数字は相手の M5 Max 128GB、専用 pack、temperature 1、copy lane on の同時刻比較で、
mlxturbo の M3 Max、group64 pack、greedy ABBAとは直接比較しない。使うのは「何が本当に出荷され、
どの部品に取り分があったか」という方向だけである。

## 新版で確定したこと

| 対象 | 2.10.2 | 2.11.1 | 読み方 |
|---|---:|---:|---|
| Flash-Next decode 16k | 53.2 tok/s | 68.4 tok/s | +29%。うち compiled fixed-M4 単体は52.6→60.3 |
| Flash-Next decode 100k | 47.5 tok/s | 60.9 tok/s | +28%。compiled lane、後半のexact小物を合成 |
| Flash-Next decode 206k | 32.2 tok/s | 44.2 tok/s | +37%。後半stackは再計測未完なので主張は前半値まで |
| Flash-Next cold TTFT 16k | 15.1秒 | 14.3秒 | -5.3%。decodeほどは縮んでいない |
| Flash-Next cold TTFT 100k | 117.0秒 | 113.2秒 | -3.2%。cold本体は引き続き別の主戦場 |
| 27B decode 16k | 38.6 tok/s | 41.4 tok/s | +7%。NAX flash route |
| 27B decode 88k | 20.5 tok/s | 30.9 tok/s | +51%。route + long-context設定、短文脈では設定が負ける |
| tool turn後のdead time | 5〜7.5秒 | 0.01秒 | 不要な初回gather、SSD再hydrate、postcommit待ち等を修復 |
| 数秒休止後のserver TTFT | 1.06秒 | 0.08秒 | 1秒周期keepalive。電力・熱との交換条件あり |

## 旧判断の訂正

`docs/BACKLOG.md` 17:28節で棄却したのは、mlxturbo の既存 `depth=3` をそのまま固定しても
遅かったという**素朴なfixed幅4**である。2.11.1が出荷したものは別物で、次を持つ。

- GDN、PLE、QSA/KV、hyper connectionを明示的にcaptureし、物理4行のstate planを作る。
- prefill後にgraphを一度installし、accept済み長をhost側ledgerとして渡す。
- capacity generationごとにbankを成長させ、短い窓やmemory超過はeagerへ戻す。
- PLE sidecarはhost historyから補助入力を組み、QSAは専用bankへpromotionする。
- 実機geometry `(48 layers, GDN 36, QSA 12, HC 4, indexer ratio 4)` をload時に検査する。

したがって「direct port不採用」は維持するが、**state-pure adapterを先に作った上での
fixed-M4 graphbankは再開**する。相手のコードを丸ごと移すのではなく、mlxturboの汎用
component replacement契約へ状態を明示する順で進める。

## 部品ごとの差分と判定

| 2.11.1の部品 | 相手の取り分 | mlxturboの現状 | 判定 |
|---|---:|---|---|
| PLE chunk lookahead | 16k prefill 14.7秒中 -2.0秒 | 同型を実装済み。50k coldでwall -7.5%、出力3/3一致 | 完了。重複移植しない |
| first-chunk gather at arrival | gather 0.62→0.013秒 | 最初のspanをrunner内で同期wait。request到着時には始めない | **小さく実装可**。warm bank hit時は必ず起動しない |
| n-gram pre-read | cold decode 56→68.8 tok/s | 常駐32GB sidecarでpage cache余力が小さい | 既定採用しない。予算付きopt-inを将来測る |
| compiled fixed-M4 graphbank | 16k round -12%、100k -18.5% | state-in/out adapter未完成 | **再開**。qwen4汎用化と同じ主線 |
| two-kernel MoE route | cycle -2.4% | 融合routeはあるが同じ10 dispatch集合か未照合 | graphbank前に重複監査、未実装部分だけA/B |
| exact op diet | 合成cycle -10.7% | bank/RoPE/residual/K20を個別照合していない | graphbank内の小物として1件ずつ測る |
| QSA verify glue | cycle -1.2% | QSA decode kernelはあるがcompiled body内のdispatch削減は未実装 | graphbank後 |
| M4 kernel trio | 合計約+2% | stage-3 pack固有契約を持たない | pack契約を確認できる場合だけ |
| block verification | tokens/window +1.6〜2.0% | 現行accept lawとの差分未監査 | 分布同一性testを先に作る |
| FR-Spec Q8 64k | PR corpusでcoverage 99.64% | 手元11,780 tokenで89.02%、各prompt線も失敗 | 棄却維持 |
| 27B NAX flash route | 16k +7%、88k route単体+16% | M3 MaxにNAX実行面なし | M5/NAX依存のまま |
| agent session修復群 | tool turn 85秒→0.58秒等 | hot prefill P0後半が未完 | **最優先監査**。cache identityとpostcommitを含める |
| GPU keepalive | pause後1.06→0.08秒 | 未実装 | 常用既定にしない。電力・温度・sustained decodeを同時計測 |

## 実行順

1. hot prefill P0に、bank hit時の不要gather、cache identity、postcommit完了待ちを追加して監査する。
2. request到着時のfirst-chunk gatherを、warm restoreが見込まれる場合は起動しない条件付きでA/Bする。
3. qwen4 component replacementでGDN/PLE/QSA/HC/cache offset/rollbackをstate-in/out化する。
4. 幅4のlogits、全cache、rollback keep=1/3/4を既存経路と一致させてからgraphbankを試作する。
5. route kernel、op diet、verify glue、block verificationを一度に合成せず、1件ずつ寄与を測る。
6. keepaliveは速度だけで採らず、休止後TTFT、10分間の消費電力、筐体温度、直後512 tokenの
   sustained decodeを同じ表にする。

Flash-Next専用packはdry-runで約115GB（本体約80GB、n-gram 32GB、MTP 1.7GB）。取得後に
MTPLX 2.11.1を相手の推奨設定で走らせる。ダウンロード中はGPUベンチを並走させない。
