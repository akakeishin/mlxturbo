# Hot prefill: 速さの代金と次の設計

## 結論

2026-09-04 の常時冷却フルベンチで、Flash-Next の warm TTFT は 0.14〜0.75 秒、
mlx-serve 比 1.23〜28.10 倍だった。特に 50k は 0.60 秒対 16.87 秒で、会話継続時の
体感を変える大きさである。

ただし、これは prefill の計算を消した数字ではない。KV、GDN、indexer、MTP の状態を
統合メモリへ前払いで残し、同じ token 列の接頭辞が来たときに計算を省いた数字である。
したがって **hot path は維持するが、全会話・全状態の無制限保持は採らない**。先に実バイト、
命中で節約した秒数、追放、batch によって失った再利用を観測し、その後に個数上限 LRU を
byte 予算へ置き換える。

## 今回の 0.60 秒が意味する範囲

50k の mlxturbo は 49,826 または 50,088 token を再利用し、新しく計算したのは 274 または
16 token だった。warm TTFT はそれぞれ約 0.71 / 0.36 秒である。中央値 0.60 秒はこの2本から
得たもので、次のことはまだ示していない。

- process 起動直後や session 追放後も 0.60 秒になること
- 複数会話が競合したときの p95
- suffix 長が同じときの安定した分布
- batch 待ちがあるときも prefix reuse を維持できること

同じ 50k の cold TTFT は 84.32 秒だった。つまり hit すれば約84秒を省ける一方、再起動、LRU
追放、tokenizer/chat template の変化、token ID の LCP 不一致があれば cold へ戻る。decoded text
が同じに見えても token ID 列が変われば同じ cache とは扱えない。

## 容量の下限

実モデル設定は 48層（full attention 12、linear attention 36）、KV head 2、head dim 256、
indexer head dim 128、linear value head 48、linear key/value dim 128、GDN state は float32、
checkpoint retention は8である。50k token の1 sessionについて、KV/indexerの256-token単位の
確保を50,176 tokenとして、配列形状から求めた下限は次のとおり。MTP、tail、graph cache、
生成時 scratch は含まない。

| 保持物 | 50kあたり |
|---|---:|
| attention KV | 約1.148 GiB |
| raw indexer | 約0.144 GiB |
| pooled indexer | 約0.036 GiB |
| GDN/PLE checkpoint 1個 | 約0.108 GiB |
| GDN/PLE checkpoint 8個 | 約0.862 GiB |
| **下限合計 / session** | **約2.189 GiB** |
| **8 sessionすべてが50k** | **約17.52 GiB** |

現行の `--max-sessions 8` は個数だけを見る。4k と 50k が同じ1枠なので、実際のメモリ圧を
表していない。また KV を論理的に trim しても、基礎 buffer の割当が縮まるとは限らない。
追放判定には token数や論理長ではなく、保持中の配列の allocated bytes を使う必要がある。

モデル自体が約98 GBあり、生成時の一時領域も必要になる。50kを8本保持する設計は、128 GBの
統合メモリでは swap、OOM、decode低下のいずれかへ近づく。`CHECKPOINT_RETENTION=8` と
`max_sessions=8` が同じ数字でも、掛け合わせた容量が安全という意味にはならない。

## 速さと引き換えに起きること

### 1. hit と miss の落差が大きい

50kでは warm 約0.60秒と cold約84.32秒の差が大きい。平均だけを公開すると速く見えるが、
session数、会話長、再起動、追放方針で利用者の待ち時間は二峰性になる。公開値には hit率と
p50/p95を併記する。

### 2. 状態を一部だけ戻すと生成が壊れる

Flash-Next は attention KV だけのモデルではない。GDN recurrent/conv state、indexer、RoPEと
論理位置、dead mask、MTP chainを同じ位置へ戻す必要がある。現在は checkpoint restore 後に
MTP stateを捨てて作り直す安全側の実装である。圧縮、共有、copy-on-writeを入れる場合も、
この契約を崩してはならない。

### 3. cache と batch の最適解が衝突する

現行の speculative batch は sessionを持たない。別requestが待っていると、単発ならhotな会話を
batch側へ送り、prefixを再計算する可能性がある。一方、常にsessionを優先すればthroughputを
落とすことがある。TTFTだけでなく、hit率を変えた B=1/2/4/8 のgoodputで決める。

### 4. 長い cache は他の処理の余白を奪う

統合メモリには安価な「CPU側へ逃がす」境界がない。保持量を増やすほど、モデル、Metalの
一時buffer、graph cache、同時requestが同じ容量を争う。disk保存は再起動をまたぐ安定した長大
prefixで、復元がcold prefillより明確に速い場合だけ候補にする。

## 採る設計

最初からpaged KVを全面導入せず、次の順で測る。

1. active conversation用の小さなmutable leafを残す。
2. RAM側はimmutableなprefix anchorをbyte予算で管理し、完全一致prefixだけrefcount/leaseで共有する。
3. valueは単なるrecencyではなく、おおむね `最近さ × 節約できるprefill秒 / allocated byte` で評価する。
4. copy量、重複、batch再計算、断片化が支配的だと測れた場合だけpaged/COW KVへ進む。
5. quantized cacheやdisk tierは最後の実験にする。

cache keyには token ID 列だけでなく、model revision、tokenizer/chat template、adapter、量子化、
位置設定を含める。異なる条件の状態は共有しない。

## 実装前に必要な計測

### P0: telemetry

各requestについて次を記録する。通常運用の速度に影響させないため、まずdebug計測として入れる。

- match kind（miss / exact / append / trim / checkpoint）
- LCP、選択checkpoint位置、reused/new token
- tokenize/LCP探索/restore/prefill/first-tokenの時間
- session、checkpoint、MTP、indexerごとのallocated bytes
- 追放したbytes/tokensと理由
- MLX active/cache memory、RSS
- batchを選んだため失ったreuseと再計算token
- preemption後に再計算したtoken

計測自体の時間増は1%未満を目標とし、bytesの内訳と実測差は5%以内に合わせる。

2026-09-04 19:30の最初の実装では、session選択の `miss/exact/append/trim/checkpoint`、
LCP、checkpoint位置、reused/new、追放token/allocated bytesを1 request 1行の生成ログへ追加した。
`/api/status`にはMLX active/cache memoryと、pool全体のallocated bytes（同一配列を重複排除）、
未知session数、processed token、累積選択/追放統計を追加した。byte走査はstatus poll時と追放時だけで、
通常requestでは行わない。436件のserver testがMetal実機で通った。

2026-09-04 20:24には、Flash-Nextのdebug requestだけでrunner内部を
`runner_prefill_s` / `runner_first_token_s`へ分けた。既存の同期点だけを使い、追加`mx.eval`は無い。
通常requestは計測フラグも結果キーも作らない。実Flash-Nextの60-token煙試験では、serverの
`ttft=0.97s`に対してprefill 971.2ms、first-token 0.4msと対応した。server全回帰は443件通過。

残るP0は、tokenize/LCP探索/restoreを個別の時間へ分けること、Flash以外のrunnerでは
分割不能な区間を明示すること、batchを選んだため失ったLCP、preemption後の再計算tokenを
記録すること、固定suffixの反復でallocated bytesとMLX active memoryの差を5%以内へ
合わせることである。

### P1: byte-budget比較

固定suffix 0/16/64/256、pure append / retokenized、文脈 0/4k/17k/25k/32k/50kを各5回以上。
会話を8/16/64本に増やし、現在のcount-8 LRU、byte-LRU、value scoreを比較する。worst-caseの
生成scratchに10%余白を残し、OOM/swap 0、cold退行は測定ノイズ内を合格線とする。

### P2: cache-aware concurrency

B=1/2/4/8、hit率 0/25/50/90%を組み合わせ、現行batchとprobe-first hot routingを比較する。
p50/p95 TTFT、throughput、saved-prefill-second/byteを同時に見る。

### P3以降

- suffix 64/256でmodel forwardがwarm TTFTの70%以上なら、suffix経路を最適化する。64/256で
  15%以上、16で2%以内の退行を合格線にする。
- exact prefixの重複が実測された場合だけimmutable anchorを実装する。保持bytes 25%以上削減、
  warm p95退行5%以内を合格線にする。
- paging/COWは共有時のcopyまたはbatch再計算が支配的な場合だけ。品質は既存の生成一致に加え、
  ΔKLD <= +0.0005を守る。

## 公開時の表現

今回の値は「same-process prefix hit時のwarm TTFT」と明記する。cold、warm hit率、p50/p95、
同時数、cache予算、suffix token数を一緒に載せる。0.60秒だけを一般TTFTとして扱わない。
GuideLLMの固定scenarioで反復を増やし、`ignore_eos`対応後に同一output token数の正式比較へ進む。
