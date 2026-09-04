# BACKLOG / NEXT 全件監査 (2026-09-04 21:07 JST)

`docs/BACKLOG.md` と `docs/NEXT-SESSION-PROMPT.md` は実験の時系列も残す正本なので、
採用後や棄却後の古い記述が意図的に残っている。この文書は、その全項目を現行HEAD
`7713f23`と突き合わせた**実行台帳**。速度・品質の数字そのものはCATCHUPを正本とし、
ここでは次に作業が要るかだけを判定する。

状態は次の4つに限る。

- **完了**: 現行コード、対象テスト、必要な実モデル検証まで済んだ。
- **棄却**: 宣言済みの速度・品質線を通らず、再開条件も記録済み。
- **実行可**: 現在の機体・モデル・テストで次へ進める。
- **依存あり**: モデル、artifact、NAX機、公開権限など、手元に無いものが要る。

## 今すぐ残っている主線

| 優先 | 項目 | 状態 | 完了条件 / 次の一手 |
|---:|---|---|---|
| 0 | hot prefill cache方針の実測 | 実行可 | P0完了。固定suffix 300行は50k p95最大0.964秒、byte差最大1.232%。次は開始前容量予約を持つbyte-budget LRUを固定traceで比較する |
| 1 | Qwen3.6-35B-A3B MTP最終表 | 完了 | 3,830 token以上を条件付き幅4へ配線。temp=0.7の4k/17k/50k 9/9勝ち、ΔKLD +0.000043、server全452 test。絶対50k 69.06 tok/sは後GEMM -6.0%のため参考値 |
| 2 | qwen4 state-pure adapter + fixed-M4 graphbank | 実行可 | MTPLX 2.11.1でexact経路の16k round -12%を確認。まず汎用component replacementで全状態とrollbackを明示し、その後だけ幅4 graphを試す |
| 3 | 継続batchの残制限 A2/A4/A5 | 実行可 | prefix reuse、temperature 0.7分布、HTTP同時要求の3件。solo非退行、品質ゲート、実server throughput |
| 4 | streaming logprobs / tool token対応 | 実行可 | 現在400で拒否するstream logprobsを実装し、ThinkingRouter/tool_callsのtoken対応を含め4 API経路を検証 |
| 5 | Gemma KV量子化 | 実行可 | full/rotating cacheの契約を先に固定し、cold/warm/qualityとメモリを比較。既存Gemma warm修復とは別件 |
| 6 | LookupSpecRunner async評価 | 実行可 | decodeの不要な同期を除き、生成列/品質を守って短・4k・17kでA/B |
| 7 | multimodal / LoRA | 実行可 | 元VLMは外付けSSDへ再取得済み。LoRAは小adapter fixtureを先に作り、どちらもtext非退行を最初のゲートにする |

## 製品機能とAPI

| BACKLOG節 | 判定 | 根拠 / 残り |
|---|---|---|
| §1 multimodal | 実行可 | 元Qwen4 VLMは外付けSSDへ再取得済み。vision重み保持、vision tower、`input_embeddings`のSpec/Flash接続、画像placeholderのn-gram除外、audio/videoが未実装 |
| §2 continuous batching基礎 | 完了 | `batch.py`、`runner.py`、`batch_spec.py`に接続済み。残る制限は下のA2/A4/A5へ分離 |
| §3 DraftSpecRunner | 依存あり | 適合draft modelが手元に無い。対象artifact取得後に着手 |
| §3 LookupSpecRunner async | 実行可 | `async_eval`未接続。既存モデルで検証可能 |
| §3 他モデル族MTP/state capture | 実行可 | qwen4の状態契約を汎用adapterへ移す主線と統合 |
| §4 non-stream logprobs | 完了 | 現行serverで対応済み |
| §4 streaming logprobs | 実行可 | chat/completions streamは現在400。token単位のlogprobとusage/finishを接続する |
| §4 tool/ThinkingRouter token mapping | 実行可 | streaming logprobsと同じtoken境界で検証する |
| §5 prefill一般論 | 完了 | 具体項目へ分解済み。50k n-gramは`62e64b9`で完了。モデル本体は下のcold節 |
| §6 LoRA | 実行可 | server/CLI adapter引数とfixtureが未検証。既存モデルから小fixtureを作って契約を先に閉じる |
| GuideLLM 0.7.3 | 完了 | 隔離venv、JSON/CSV/HTML、`ignore_eos`全4経路、実server 2/2を確認 (`7012540`) |

## 継続batch A1〜A5

| ID | 判定 | 根拠 / 残り |
|---|---|---|
| A1 2048上限 | 完了 | `5ba1c84`で撤去 |
| A2 session/prefix reuse | 実行可 | batch側は`prefill_reused=0`。B>1安全な再利用、B=1非退行、cache一致が必要 |
| A3 in-flight join | 完了 | `964a47a`で接続 |
| A4 temperature > 0 | 実行可 | temperature 0.7のsolo/batch分布・KLD・課題品質を比較 |
| A5 HTTP同時要求 | 実行可 | library直叩きだけでなく実serverのp50/p95 TTFT、aggregate tok/s、join数を測る |

## 一次検査とカーネル契約 B6〜D16

| ID | 判定 | 根拠 / 残り |
|---|---|---|
| B6 GPU fingerprint | 完了 | `42e601d`。Metal 8系統が全て発火し総合合格 |
| B7 GDN外側述語 | 完了 | `505718e`。5条件共有、契約8 test、実8kのcache 110配列一致 |
| B8 eligible理由 | 完了 | GDN/prefill-attn等にwarn-onceが入り、GPU fingerprintで空振りを検出 |
| C9 GDN conv-state契約 | 完了 | `wants()`がlengths付きcacheを拒否。将来cache override追加時だけ再監査 |
| C10 prefill-attn S下限 | 完了 | `MIN_S=64`。decode/verifyを通さずGPU指紋でも発火確認 |
| C11 KV直接buffer契約 | 完了 | `0c92440`。標準KVCache継承だけを許可、4契約と実GPU検証合格 |
| C12 `_segments_gpu`二重計算 | 棄却 | 対象の`MLXTURBO_MOE_VERIFY`カーネル自体がin-model +46〜52%で既定off。再採用時だけ再開 |
| C13 `_max_seg_bound` | 棄却 | C12と同じ既定off経路だけに存在。top_k/verify幅を変えて再採用するときが反転条件 |
| C14 HC write env gate | 完了 | `0bca20f`。明示offの直接呼出しをno-op化、10 test |
| D15 GDN誤差尺度 | 完了 | `verify_gdn_metal.py`は相対誤差で判定済み |
| D16 `--prefill-once` | 完了 | decode knobの運用規則としてCLAUDE.mdへ固定。実装課題ではない |

## Flash-Next cold prefill

現行常用冷却の17kは27.215秒、約620 input tok/s。内訳はMoE 34.8%、GDN 29.3%、
attention 21.3%、HC read 10.5%。50k n-gram prefetchはoff 93.252秒からon 86.255秒へ
wall -7.5%、約535→578 tok/s (+8.1%)、出力3/3一致で既定onを確定した (`62e64b9`)。

| 案 | 判定 | 根拠 / 再開条件 |
|---|---|---|
| n-gram interleaved layout | 完了 | RAM 0、50k OOM回避。separate 32GBは使わない |
| n-gram early prefetch | 完了 | 4k/17k非悪化、50k -7.5%。行cacheは1 prefill分≤70MB |
| chunk 4096/8192 | 棄却 | 一時メモリと非MoE費用が相殺、OOMリスク |
| prefill pipeline | 棄却 | 既存A/B -0.6%。2 group同居のメモリ代に見合わない |
| causal-mask融合 | 棄却 | 実測で採用線未達 |
| small-kv QSA / flash | 棄却 | 最大でも約1%、適用範囲も狭い |
| K/V prefix trim | 棄却 | 短+5.2%悪化、17kはQSA経路で発火0 |
| HC elem prefill拡張 | 棄却 | 非bit-exactかつtok/round -4.8% |
| GDN state_out削除 | 棄却 | in-model ±0.0%、peak -12.6MBだけ |
| qkv wide | 棄却 | 8k prefill +0.3%、取り分なし |
| grouped MoE / qmm_wide / GDN blocked / prefill-attn | 完了 | 既定採用済み。新候補は現行後の差分として測る |
| persistent streaming MoE | 実行可・研究 | 1層の非GEMM余白17.1/19.6msが事前線16msを通過。全48層array_equalを先に通し、17k wall≤25.854秒でのみ採用 |
| 新しいGDN/attention候補 | 実行可 | 全体5%を説明できるmicro/traceを先に出し、出力一致またはΔKLD≤+0.0005でA/B |

## 汎用architecture / forward copy

| 項目 | 判定 | 根拠 / 残り |
|---|---|---|
| qwen4 capability / component表 | 完了 | `arch.py`に能力判定と契約がある |
| qwen3_5を含む汎用component replacement | 実行可 | qwen4固有名とcapture/state処理をadapterへ寄せる |
| forward copy #1〜5/#7/#8 | 完了 | 本家呼出しまたは共有helperへ移行済み |
| capture-module ULP | 実行可・低 | 27Bで許容差内だが残存。cache/rollbackを守って共有化できる場合だけ実施 |
| generic staged #9 | 依存あり | 対象族と実モデルを揃えてから。形だけの抽象化はしない |
| QSA全面`-inf`穴 | 実行可・低 | 境界testを先に追加し、該当入力が実モデルで発生するか確認 |
| 起動較正 / dynamic discovery | 実行可 | 複数族の実測が揃った段階で汎用化主線へ統合 |

## モデル別レーン

| モデル / 案 | 判定 | 根拠 / 残り |
|---|---|---|
| Flash-Next full / GuideLLM | 完了 | 常用冷却fullと外部形式の結果あり |
| Flash fixed幅4の直接port | 棄却 | 幅4自体は遅い。MTPLX 2.11.1のstateful graphbankとは別案 |
| Flash fixed-M4 graphbank | 実行可 | 2.11.1で出荷・exact検査・16k round -12%。state-pure adapterとrollback一致が前提 |
| Flash first-chunk gather at request arrival | 実行可 | 現行はrunner内で同期wait。warm bank hit時に起動しない条件を先に固定してA/B |
| Flash FR-Spec Q8 head | 棄却 | 手元trace coverage 89.02%、宣言線99.9%未達 |
| ANE coarse head | 依存あり | FR-Spec成立が開始条件だったため停止。新しい小headができた場合だけ再評価 |
| Qwen3.6 MTP読込 / warm checkpoint | 完了 | 短/4k decode +51〜57%、4k温TTFT -73.5% |
| Qwen3.6最終full/quality | 完了 | temp=0.7の4k/17k/50k 9/9勝ち、KLD +0.000043。短文1/3の+8.9%を避け、実測構造だけ3,830 token以上を3/3/0へ配線。452 testと実起動合格。50k参考69.06 tok/sは後GEMM -6.0%を併記 |
| Qwen3.6 MoE compile汎用化 | 棄却 | 短ms/round -0.6%だが50k +0.9%、未空焼き幅のtrace税あり。偽40層ログだけ修正 |
| 27B controller / MTP cache / q2 rerank | 完了 | 採否と品質をCATCHUPへ記録済み |
| 27B state_out skip | 実行可・低 | Flash側は実測±0.0%だが別族なので未決。27B固有のtraceで占有が見えた場合だけA/B |
| 27B NOSYNC recall追加確認 | 実行可・任意 | 速度採用済み。公開品質表を厚くするときだけ追加 |
| Gemma warm TTFT | 完了 | tail/checkpoint/rotating snapshotで4k -86.1%、17k 0.415秒 |
| Gemma RMS commonstat | 棄却 | 実測で取り分なし |
| Gemma KV量子化 | 実行可 | cache能力と品質を別レーンで検証 |
| NAX sorted gather | 依存あり | M5/NAX実機とMLX修正版が必要。非NAXでは再現不能 |

## hot prefill / cache policy

| 段 | 判定 | 根拠 / 残り |
|---|---|---|
| session LCP/checkpoint/reused/new | 完了 | debug telemetryとpool/eviction byteを実装済み |
| runner prefill/first-token | 完了 | `bbdc7ad`。追加同期なし、443 test、実モデル煙試験 |
| tokenize/LCP/restore | 完了 | debug requestだけ各段を計時し、通常requestの時計・同期は増やさない。server全448 test合格（`7d967a3`） |
| batch-forfeited reuse | 完了 | read-only LCP probeをrequest/累積へ接続。互換しないspec→fallback降格は偽LCPを0化（`8ec7c03`） |
| preemption recomputed tokens | 完了 | 復帰prefill完了時だけrequest/累積へ加算。cancel/未完了を含めない（`7d967a3`） |
| suffix 0/16/64/256反復 | 完了 | 6文脈×2 mode×4 suffix×5回=300行。LCP/reused/new全一致、byte差最大1.232%。50k hot p95最大0.964秒 |
| Flash末尾checkpoint幅 | 完了 | denseと同じ8へ統一。4k末尾書換えの全再計算6.46秒を3,992 token再利用・0.08秒へ修復 (`3904b9b`) |
| Flash以外のTTFT非分割区間 | 完了 | debug telemetry/logへ`runner_unsplit (prefill+first_token)`を明示。通常requestは不変、server全455 test (`d083a42`) |
| count-8→byte-budget LRU | 実行可・後続 | prospective admission、固定multi-session trace、実tokenizer retemplateを先に実装。scratch 10%、OOM/swap 0、予測差5%以内、p95/保存prefill秒で判定 |
| value score | 実行可・保留 | score/減衰/tie-breakが未定義。byte-LRUを先に通し、hold-out traceで明確に勝つ場合だけ既定候補へ進める |
| 無制限cache-all | 棄却 | 50k実測2.079GiB/session、8本約16.63GiB。262kでは成立しない |
| shared-prefix / COW / paging | 実行可・条件付 | exact-token重複が実測で多い場合だけ。先にimmutable anchor、pagingは最後 |

## 配布・公開・比較

| 項目 | 判定 | 根拠 / 残り |
|---|---|---|
| server protocol / distribution docs | 完了 | OpenAI互換、運用、GuideLLM手順あり |
| HF lm_head公開 | 依存あり | 外部書込みなので、公開対象・repo・revisionの最終指定が必要 |
| README/比較表の最終更新 | 実行可・後続 | Qwen3.6最終表と残る機能ゲート完了後に一度だけ更新 |
| teacher再構築 | 依存あり | SSD/model artifactが必要。QSA品質の既存判定は完了済み |

## 直近の再開順

1. prospective admissionと固定multi-session traceを作り、count-8とbyte-budget LRUを比較する。value scoreは定義と勝ち筋が固まるまで後段。
2. qwen4汎用component replacementをstate-pure adapterへ進め、MTPLX 2.11.1のfixed-M4 graphbankを再検証する。
3. batch A2/A4/A5、streaming logprobs、Gemma KVの順に、各1論点1commitで閉じる。
4. モデル/artifact/NAX/公開権限が要る項目だけ、必要物と代替検証を具体化して確認する。
