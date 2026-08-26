## 結論

「cold expert の SSD 読みは受理数だけ償却される」という一般形は成立しません。成立条件は、検証窓内の expert 和集合がほとんど増えず、かつ RAM キャッシュ後の missing bytes が小さいことです。

判断は次です。

- 316 GB の GLM-5.2 を前提とした production 戦線：現時点では **NO-GO**
- 316 GB を取得せずに行う、SSD・routing locality・I/O replay の短期測定：**GO**
- 測定で後述の閾値を越えた場合のみ、DeepSeek-V4 を境界実験台として実行系を試作する

GLM-5.2 REAP25 の実 config は 78 層中最初の3層が dense、残り75層が MoE、192 experts、top-8、hidden 6144、expert intermediate 2048 です。[GLM-5.2 REAP25 config](https://huggingface.co/pipenetwork/GLM-5.2-REAP25-MLX-4bit/blob/main/config.json)

4bit/group64 の scales/biases 込み実効 4.5 bit/weight とすると、

\[
S_e=3\cdot6144\cdot2048\cdot4.5/8
=21{,}233{,}664\ {\rm bytes}
=20.25\ {\rm MiB/expert}
\]

したがって routed experts 全体は約305.8 GB、1 token が踏む expert は

\[
75\cdot8\cdot21.23{\rm MB}=12.74{\rm GB/token}
\]

です。90 GB の expert cache は各層平均56.5 experts、全体の29.4%しか保持できません。

## 1. アーキテクチャ案

### コンポーネント

1. **Resident core**

   embeddings、lm_head、attention/DSA、router、norm、最初の dense 層、shared expert、MTP、KV、実行 scratch は常駐させます。GLM の316 GBから routed experts 305.8 GBを引くと、常駐重みは概算10 GBです。

2. **Expert store**

   safetensors shard をそのまま demand load してはいけません。現在の mlx-lm はロード時に experts を `SwitchGLU` の巨大な stacked tensor にまとめるため、expert 単位の退避ができません。[deepseek_v32.py](/Users/ht/dev/fastmlx/.venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v32.py:534)

   `(layer, expert)` ごとに gate/up/down の packed weight、scales、biases を連続した1 slabにし、4 KiB以上で整列させます。75ファイル×固定 expert offset 程度にし、1 expert 1ファイルの14,400ファイル構成は避けます。

3. **Router-split MoE executor**

   各層を現在の monolithic `layer(h)` から次へ分割します。

   ```text
   attention
     → post-attention norm
     → exact router
     → m positions の expert union
     → cache acquire / I/O submit
     → shared expert + resident expert を実行
     → cold I/O 完了待ち
     → cold expert を実行
     → score で集約
   ```

   exact router は現層では精度100%ですが、次層の router は現層 expert 出力に依存するため先行評価できません。75層分をまとめて prefetch することは不可能です。

4. **Hot expert cache**

   初期値は expert 用80–90 GB程度です。120 GB wired limitを使用量の目標にしてはいけません。

   \[
   C_{\rm expert}
   =M_{\rm usable}-W_{\rm core}-KV-A(m)-M_{\rm staging}-M_{\rm OS}
   \]

   PLAN上、fastmlx の検証状態だけでも \(m=32\) では約4.8 GBです。[docs/PLAN.md](/Users/ht/dev/fastmlx/docs/PLAN.md:197)

   政策は次を推奨します。

   - 層ごとの最低 quotaを持つ。global LRUだけにしない
   - 60–70%：calibration traceから決める静的 hot
   - 25–35%：SLRU/TinyLFU の protected/probation
   - 約5%：prefetch disposable
   - 初回 miss は読み込むが原則非 admission。2回目以降に victim と頻度比較
   - GPU利用中の slot は refcount と Metal fenceで保護し、I/O完了だけで再利用可能にしない
   - freed bufferをMLX allocator cacheに残さない。preallocated poolを使い、`mx.set_cache_limit()` は小さくする

### macOS I/O

production の第一候補は Metal 3 I/O です。

- `MTLDevice.makeIOFileHandle(url:)`
- `MTLDevice.makeIOCommandQueue(descriptor:)`
- `MTLIOCommandQueueDescriptor.type = .concurrent`
- `MTLIOCommandBuffer.loadBuffer:offset:size:sourceHandle:sourceHandleOffset:`
- `MTLSharedEvent` と I/O側 `signalEvent`、GPU側 `encodeWaitForEvent`

Metal I/O はファイルから `MTLBuffer` へ直接ロードできます。[Apple Metal resource loading](https://developer.apple.com/documentation/metal/resource-loading)

初期 queue 設定は以下です。

- exact miss用：normal/high priority、`maxCommandsInFlight=8`
- predictive prefetch用：low priority、QD 2–4
- `maxCommandBufferCount=2`
- 実測 sweep：QD 1/2/4/8/16

Appleは concurrent queueと `maxCommandsInFlight` を正式に提供しています。[MTLIOCommandQueueDescriptor](https://developer.apple.com/documentation/metal/mtliocommandqueuedescriptor)

ただしMLXの公開Python APIから任意の `MTLBuffer` を weight arrayとして差し込む経路はありません。productionには ObjC++/C++ extensionと cache-aware expert kernelが必要です。これは単なる loader オプションではなく、新しい実行 backendです。

fallbackは次です。

- `dispatch_io_create(DISPATCH_IO_RANDOM, ...)`
- `dispatch_io_read`
- または bounded thread pool上の `pread`
- `fcntl(F_NOCACHE)` をA/Bしてpage cache二重保持を抑える
- `F_RDADVISE` / `F_RDAHEAD` は advisoryで、完了・resident保証がないため予測prefetchの本体にはしない

[Dispatch I/O](https://developer.apple.com/documentation/dispatch/dispatch-i-o)、[Apple fcntl](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)

### Router prefetch

- **現層 exact route**：可能、精度100%。ただし attention 後でしか分からない
- **次層 predictive route**：MTP hidden、直前層route、token IDなどからの推定になる
- 現行fastmlxのdraftは MTP moduleとlm_headだけで、target modelの層別expert IDを生成していません。[spec.py](/Users/ht/dev/fastmlx/fastmlx/spec.py:407)

予測集合 \(P\)、実 target cold集合 \(T\) とすると、

\[
{\rm recall}=|P\cap T|/|T|,\qquad
{\rm precision}=|P\cap T|/|P|
\]

総読み量は \( |T|+|P\setminus T| \)、router後の残stallは \( |T\setminus P| \) です。単一streamではprefetchは読み量を減らさず、時間を前へ移すだけなので、precision 90%以上、recall 80–85%以上、prefetch幅1.25倍以内を満たさなければ切るべきです。

### 熱別ビット配分

bit幅は「現在RAMにいるか」ではなく、expert IDごとに固定します。cache状態で2bit/4bitを切り替えると、同じpromptでもcache履歴により出力が変わります。

group64のoverheadを0.5 bit/weightとすると、

| 保存bit | 実効bit | 1 expert |
|---:|---:|---:|
| 2 | 2.5 | 11.25 MiB |
| 3 | 3.5 | 15.75 MiB |
| 4 | 4.5 | 20.25 MiB |
| 6 | 6.5 | 29.25 MiB |
| 8 | 8.5 | 38.25 MiB |

推奨は、router/shared/dense/MTP/lm_headを4–8bit、頻出かつ感度の高いexpertsを4bit、低頻度かつ低感度expertを3bitです。2bitはKLD/perplexity gate通過時だけに限定します。訪問頻度だけではなく、

\[
p_{\rm route}(e)\cdot
\|y_e^{q}-y_e^{ref}\|
\]

で感度を測るべきです。既存PLANのKLD gateと統合できます。[docs/PLAN.md](/Users/ht/dev/fastmlx/docs/PLAN.md:98)

## 2. 投機との相乗の定量見積もり

層 \(l\)、検証位置 \(i\) のexpert集合を \(A_{l,i}\)、resident集合を \(C_l\)、expert byteを \(s_l\)、draft受理数を \(a\) とします。

fastmlxでは1回の検証で出力するtoken数は、通常

\[
c=1+\mathbb E[a]
\]

です。したがって償却分母はdraft受理数 \(a\) ではなく \(1+a\) です。現行実測は mean accepted 1.38、tokens/step 2.38です。[D2-RESULTS.md](/Users/ht/dev/fastmlx/docs/D2-RESULTS.md:23)

\[
B_{\rm AR}
=\sum_l s_l|A_{l,1}\setminus C_l|
\]

\[
B_{\rm spec/out}
=
\frac{
 B_{\rm draft}
 +B_{\rm prefetch\ waste}
 +\sum_l s_l
 \left|
 \left(\bigcup_{i=1}^{m}A_{l,i}\right)\setminus C_l
 \right|
}{c}
\]

従って「\(c\) 倍償却」になるのは、検証窓内でcold expert集合が全く増えず、draft自身のI/Oとprefetch wasteもゼロの場合だけです。

独立一様routingなら、

\[
U(m)=N\left[1-\left(1-\frac{k}{N}\right)^m\right]
\]

GLMの \(N=192,k=8\) では、

| m | expert union/層 |
|---:|---:|
| 1 | 8.00 |
| 2 | 15.67 |
| 4 | 30.05 |
| 8 | 55.41 |

現行 \(m=4,c=2.38\) ならAR比は、

\[
\frac{U(4)}{k\,c}
=\frac{30.05}{8\cdot2.38}
=1.58
\]

つまり局所性がなければ、MTPはSSD bytes/outputを58%悪化させます。全draft受理で \(c=4\) でも比は0.94で、4倍ではなく6%改善にすぎません。

SSD上のbreak-even条件は、

\[
U_{\rm cold}(4)<k\,c=19.04
\]

です。25%以上のI/O改善を着手条件にするなら、

\[
U_{\rm cold}(4)\le0.75\cdot8\cdot2.38=14.3
\]

が必要です。

木投機はさらに厳しく、木の全検証node数を \(B\) とすると、独立routingの和集合は \(U(B)\) です。15 nodeを評価してaccepted pathが最大4 tokenなら、

\[
U(15)=90.6,\qquad
\frac{90.6}{8\cdot4}=2.83
\]

となります。SSD offloadでは、木化は実測unionがchainより有利と証明されるまで無効にすべきです。

## 3. 失敗モード

### 局所性なし

90 GB cache、uniform popularityならhit率は約29.4%です。I/Oだけから得られる楽観的上限は次になります。

| 実行 | cold GB/output | 5–7 GB/sでのI/O上限 |
|---|---:|---:|
| AR、m=1 | 8.99 | 0.56–0.78 t/s |
| MTP m=4、独立routing、c=2.38 | 14.19 | 0.35–0.49 t/s |
| MTP m=4、窓内route完全一致 | 3.78 | 1.32–1.85 t/s |
| MTP、完全一致、cache hit 70% | 1.61 | 3.11–4.36 t/s |

最後の行まで届いて初めて実用候補です。すべてcompute、router同期、tail latencyを無視した上限なので、実速は低下します。

### swap death

120 GBをwireし切ると、OS、file I/O buffer、KV、Metal scratchの余地がほぼ消えます。file readとswap writeが同じSSDを奪い合い、帯域低下→処理時間増加→さらにmemory pressureという正帰還になります。

監視対象は最低でも以下です。

- `vm.swapusage`
- `memory_pressure`
- `vm_stat`
- `mx.get_active_memory()`
- `mx.get_cache_memory()`
- SSD p95 latency

swapが増えたrunは性能サンプルとして不合格です。wired limitは「許可されたresident上限」であって安全なcache容量ではありません。[MLX set_wired_limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)

### Prefetch外れ

GLMの1 expertは20.25 MiBなので、false positive 1個につき理想値でも3.0–4.25 msとcache slotを失います。各層で4個外すと、75層で約0.9–1.3秒分の余計なI/Oになり得ます。

false negativeは現層router後の完全stallです。平均精度だけでなく、step単位p95のmissing expert数を使う必要があります。

### MLX lazy load / wired limit

小さいが重要な補正があります。pinnedされたMLX 0.32.2の実ソースでは、safetensors loaderはliteralなOS `mmap` ではなく、lazy `Load` と `ParallelFileReader` の `pread` 系実装です。[load.h](/Users/ht/dev/fastmlx/.venv/lib/python3.13/site-packages/mlx/include/mlx/io/load.h:59)

さらにmlx-lmの既定 `lazy=False` はロード後に全parameterを `mx.eval()` します。[utils.py](/Users/ht/dev/fastmlx/.venv/lib/python3.13/site-packages/mlx_lm/utils.py:282)

したがって316 GBでは、

- `lazy=True`
- expert stackingを回避する独自loader
- 明示的slot pool
- 明示的eviction

が必須です。lazy loadだけではexpert cacheにはなりません。literal mmap版を作った場合も、file-backed pageとMetal bufferの二重resident化を避ける必要があります。

## 4. go/no-go の判定実験

### 測定1：SSDとI/O経路

316 GBモデルは不要です。RAMより大きい既存ファイル集合または試験ファイルを使い、実 expert と同じ20.25 MiB random slabを読みます。

- API：MTLIO、比較として `dispatch_io_read`/`pread`
- QD：1/2/4/8/16
- 15分以上
- cold/F_NOCACHE条件とwarm page-cache条件を分離
- GPU側で同時にRAM帯域kernelを走らせる
- sustained p10 GB/s、p50/p95 read latency、最初/最後5分、GPU低下率を記録

一次gate：

- sustained p10 ≥4.5 GB/s
- 最後5分が最初5分の85%以上
- 同時GPU compute低下 ≤10%
- swap 0
- MTLIO-loaded bufferを全量copyなしでexpert kernelが消費できる

### 測定2：小MoE routing trace

実使用分布の code/prose/edit/tool-agent から最低8k、できれば20k decode tokenを採ります。

ARと実際のMTP verify window \(m=2,4,6\) について、各層・各位置のexpert IDを保存し、以下を求めます。

- \(U_l(m)\)：window union
- consecutive Jaccard
- reuse-distance CDF
- expert popularity curve
- cache size別のbyte hit率
- accepted token数 \(c\)
- draft-route predictorのprecision/recall
- rejected draftが追加したcold bytes

単なる「前tokenと何個一致したか」では不十分です。実cache policyをtrace replayし、missing bytes/outputを直接求めます。

### 測定3：GLMへのスケーリング

proxy MoEの \(N',k'\) に対し、独立routing比からの相関係数を

\[
\delta_m=
\frac{U_{\rm observed}'(m)}
{N'[1-(1-k'/N')^m]}
\]

とし、

\[
\hat U_G(m)=
{\rm clip}\left(
\delta_m\cdot192
[1-(1-8/192)^m],
8,8m
\right)
\]

でGLMへ写します。cache容量率を合わせてtrace replayしたbyte hit率を \(H_G(C)\) とすると、

\[
B_{\rm cold/step}
=
75\cdot20.25{\rm MiB}\cdot
\hat U_G(m)\,[1-H_G(C)]
+B_{\rm draft}+B_{\rm FP}
\]

RAM側の最低読量は概算、

\[
B_{\rm RAM/step}\simeq
10.2{\rm GB}
+1.5925{\rm GB}\cdot \hat U_G(m)
\]

です。

SSDで \(n\) slabsを読む実測時間を \(R_{\rm QD}(n)\) とすれば、最終予測は、

\[
T_{\rm step}=
T_{\rm draft}
+\sum_{l=1}^{75}
\left[
T_{\rm attn/router,l}
+\max(T_{\rm hot/shared,l},R_{\rm QD}(n_l))
+T_{\rm coldcompute,l}
\right]
+T_{\rm maint}
\]

\[
{\rm TPS}=\frac{c}{T_{\rm step}}
\]

とします。帯域だけの簡略式より、層ごとのI/O barrierを残すことが重要です。

### 測定4：小MoEでのend-to-end再現

小MoEのexpert cacheを意図的に制限し、traceどおりに20.25 MiB相当のslabをMTLIOで読み、予測式と実TPSの誤差を確認します。

- 予測誤差 ≤20%
- p95 step latencyを含める
- cache 80/90/96 GB相当の3点をreplay
- MTP onは実測で `bytes/output < AR` の場合だけ

proxyからGLMへの外挿が最大の不確実性なので、中央予測4 t/s以上かつ悲観側2 t/s以上をGLM取得前のgateにします。

速度目標ごとのcold-byte条件は、

\[
B_{\rm cold/out}
\le
\beta_{\rm SSD}
\left(\frac1v-T_{\rm nonIO/out}\right)
\]

です。non-I/Oを80–100 ms/outputとすると、

- 2 t/s：4.5 GB/s SSDで約1.8–1.9 GB/output以下
- 5 t/s：約0.45–0.54 GB/output以下

になります。

## 5. 総合判断

現ロードマップにproduction規模の「第3戦線」として加える価値は、今はありません。

理由は、これは既存fastmlxへの小さな機能追加ではなく、

- expert専用file format
- model loader差し替え
- router分割
- cache controller
- Metal I/O
- buffer lifetime/fence
- cache-aware quantized MoE kernel
- 新しいcorrectness/KLD matrix

を伴う別backendだからです。一方、進行中のA2 kernelとMTP改善は既存モデルすべてへ効き、現在の成功条件に直結しています。[README.md](/Users/ht/dev/fastmlx/README.md:35)

したがって順序は以下です。

1. A2 GPU gateとMTPの現行測定を完了
2. PLANのM1としてrouting traceとSSD測定だけ行う
3. gate通過ならDeepSeek-V4を人工的に64–96 GB cacheへ制限し、境界offload試験
4. DeepSeekでAR 5 t/s以上、MTPでさらに10%以上改善した場合だけGLM backendへ進む

結論をGOへ反転する条件はすべて次を満たす場合です。

- actual storageのsustained p10 ≥4.5 GB/s
- zero-copyまたはcopy込み実効帯域が同水準
- \(U_{\rm cold}(4)<19.0\)、望ましくは14.3以下
- GLM換算cold bytes ≤1.8–2.4 GB/output
- proxy中央予測 ≥4 t/s、悲観予測 ≥2 t/s
- swap 0、長時間低下15%以内
- 3bit cold expertsがKLD gateを通過
- model sourceとpacked storeを20%以上の空き付きで保存可能

逆に、以下のどれか1つでproduction案はNO-GOです。

- MTLIO bufferをMLX側で使うために広範なMLX forkまたは全量copyが必要
- MTPのmissing union/outputがAR以上
- 最良cache policyでもcold bytes >4–5 GB/output
- external SSDしか使えず4.5 GB/sを下回る
- swap発生またはp95 stallが1秒を超える
- cache状態依存のbit切替が必要になる

なお現PLAN記載の内蔵SSD空き267 GBでは316 GB artifact自体が入りません。[docs/PLAN.md](/Users/ht/dev/fastmlx/docs/PLAN.md:174) 外部SSDを使うなら、内蔵SSDの5–7 GB/sではなく外部SSDの実測値が判断基準です。

独立した Sol xhigh レビューでも、「短期のpre-download測定のみconditional GO、production戦線は現時点NO-GO」という同じ判定でした。ファイル変更は行っていません。

