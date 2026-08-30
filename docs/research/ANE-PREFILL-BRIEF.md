# ANE-PREFILL-BRIEF — Neural Engine で prefill を速くできるか (2026-08-30)

このレーンの問い: **Apple Neural Engine (ANE) を使って Flash-Next の prefill を
速くできるか。** 判定できたら止める。実装まで行くかは、その数字で決める。

このレーンは `mlxturbo` 本体と独立している。失敗しても本体に影響しない
(効かなければ採用しないだけ)。

## なぜやるか

prefill が体感を支配している。

- prefill は **290-325 tok/s** しか出ない。デコードの約 11 倍でしかない
- cold 32k のリクエストは **TTFT 112 秒**、壁時計の **84%** を prefill が占める
- Claude Code は空のディレクトリでも**毎ターン約 54,000 トークン**送ってくる
  (42 個の tool 定義だけで 134,000 文字)
- prompt cache の再利用で 2 ターン目以降は 9 割超を飛ばせるが、**初回とキャッシュミス時は
  まるごと食らう**
- バッチ化は B に線形なので、ここには一切効かない

競合の [oMLX](https://github.com/jundot/omlx) は ANE prefill offload で
**GPU-only 比 26.3% 速い** (M3 Ultra、4K 文脈) と公表している。取り分は小さくない。

## 分かっていること (調べ直さなくてよい)

**環境**

- Apple M3 Max / 128GB / macOS 26.4
- `coremltools` **9.0** が利用可能。`ComputeUnit` に **`CPU_AND_NE` が存在する**
- `/System/Library/PrivateFrameworks/` に `ANECompiler.framework`、`ANEServices.framework`、
  `AppleNeuralEngine.framework` が実在する
- **M3 Max の ANE は 1 基**。oMLX の "fused MLP/down" 経路は dual ANE を要求するので
  対象外。ただし oMLX 自身が "Single, dual, and fused dispatch paths" と書いており、
  **single 経路は存在する**

**oMLX が「private ANE compiler」と呼んでいるもの**

`ANECompiler.framework` を直接叩く低レベル経路を指していると読める。ただし
**Core ML (`coremltools`) 経由という公開ルートが別にある**。「私的コンパイラが無いと
不可能」ではない。どちらが速いか、そもそも Core ML 経由で使い物になるかは未検証。

**モデルの形 (Qwen3.8-Flash-Next, v-l レシピ)**

| | |
|---|---|
| 層数 | 48 (うち full attention 12、GDN 36。`full_attention_interval: 4`) |
| hidden_size | 2560 |
| attention heads / KV heads | 24 / 2、head_dim 256 |
| MoE | 512 experts / top_k 10 |
| GDN | value heads 48、key head dim 128 |
| 量子化 | 4bit affine、group_size 64 (expert の 83% が 4bit、残り 6bit) |

**この形が ANE に向くかは自明でない。**ANE は fp16 の密行列を得意とするが、
このモデルの重みは 4/6bit 量子化されており、逆量子化のコストが乗る。

## 段取り

**第 1 段が門。ここで芽が無ければ深追いしないこと。**

### 第 1 段: ANE が本当に速いのか (小さく確かめる)

Core ML で単純な行列積を作り、`CPU_AND_NE` と `CPU_AND_GPU` で速度を比べる。
形は Flash-Next の prefill に近いもの、例えば `(1, 4096, 2560) x (2560, 2560)` の fp16。

**注意**: `coremltools` の MIL Builder は dtype の指定に癖がある。
`mb.TensorSpec(shape=..., dtype=...)` に numpy の型をそのまま渡すと
`AttributeError: type object 'numpy.float16' has no attribute '__type_info__'` になる。
`coremltools.converters.mil.mil.types` の型を使うこと。ここで詰まった経緯があるので、
API の正しい形を最初に確かめてから進めること。

判定:

- ANE が GPU より**明確に速い** (1.2x 以上) → 第 2 段へ
- **同等または遅い** → そこで止めて報告。ANE は見送り
- ANE に落ちていない可能性を必ず確認すること。`compute_units` を指定しても、
  対応しない演算があると黙って GPU/CPU にフォールバックする。
  `powermetrics` か Instruments で ANE が実際に動いているかを見る

### 第 2 段: 量子化された重みで成立するか

第 1 段が通ったら、**4bit 量子化された重みを ANE で扱えるか**を確かめる。
逆量子化を CPU/GPU でやって ANE には fp16 を渡す形になるなら、その転送コストが
利得を食い潰さないかを測る。

### 第 3 段: prefill の一部を切り出せるか

Flash-Next の prefill で最も重い部分を特定し (MoE の expert 行列積か、
full attention か、GDN か)、そこだけ ANE に載せられるかを見る。
**全部を載せる必要はない。26.3% は一部の offload で達成されている。**

## 守ること

- **出力が変わってはいけない。**prefill の結果が変われば生成が変わる。
  ANE の fp16 と GPU の bf16 で数値が違うのは避けられないが、
  **どの程度違うかを KLD で測って報告すること**。量子化ノイズ (v-fast6 で KLD 0.00378)
  と同オーダーなら許容範囲、桁が違えば不可
- **測る前に判定基準を宣言すること。**このリポジトリの作法
- **同一プロセス内で交互に測ること。**サーマルドリフトで 5 分に 15% 落ちる
- 効かなかったら「効かなかった」と報告すること。**このリポジトリには、遅延評価の
  測定ミスで実在しない改善余地を見てカーネルを 1 本書いた記録がある**
  (`docs/research/KERNEL-BRIEF-MOE-GDN.md`)。同じ轍を踏まないこと

## やらないこと

- **`mlxturbo/` 本体を変更しない。**このレーンは調査。採用の判断は数字が出てから
- dual ANE 前提の経路 (fused MLP/down)
- ANE の低レベル API (`ANECompiler.framework` の直接利用)。
  まず公開ルート (Core ML) で芽があるかを見る

## 報告してほしいこと

1. 第 1 段の数字 (ANE 対 GPU、形と条件つき)。**ANE に本当に落ちていたかの確認方法も**
2. 止めた場合はその理由
3. 進んだ場合は、どこまで行けたか、次に何を確かめるべきか
4. Core ML / ANE についてこの過程で分かった制約 (対応演算、dtype、形状の制限など)

## 参考

- `docs/BACKLOG.md` §5 (prefill の高速化)
- `docs/research/KERNEL-BRIEF-MOE-GDN.md` (カーネル最適化の記録。**失敗も書いてある**)
- `docs/POSITIONING.md` (このプロジェクトが何を看板にするかの判断)
- oMLX のリリースノート https://github.com/jundot/omlx/releases
  (v0.6.3 の "Made Qwen GDN prefill offload recurrent-safe"、
  "Retained ANE prefill performance" あたり)
