# VS-MLX-SERVE — 同一機での実測 (2026-08-30)

[mlx-serve](https://github.com/ddalcu/mlx-serve) (Zig, star 910) と、同じ M3 Max
128GB・同じ 31 課題・同じ HTTP 経路で比べた記録。

## 揃えたもの / 揃わないもの

揃えた: プロンプト (`bench/eval_prompts.py` の 31 件)、生成 128 トークン、
temperature 0、ストリームの壁時計から TTFT とデコードを分離、暖機 1 回を捨てて
2 回の中央値、1 プロセスずつ順に起動。

揃わない: **重みが違う**。mlxturbo は v-l (77GB、n-gram をサイドカーに分離)、
mlx-serve は ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit (68GB + n-gram 表 30GB を
mmap、lm_head 8bit、vision tower 同梱)。どちらも `qwen4_exp` 4bit group 64 で、
それぞれの推奨構成どうしの比較。同一重みの比較ではない。

両者とも MTP 投機を有効 (mlx-serve は MoE では `--mtp` が要る)。mlx-serve は
PLD (prompt lookup) も既定 ON。

## 31 課題 (短いプロンプト)

| | decode 中央値 | 範囲 |
|---|---|---|
| mlx-serve | **64.0 tok/s** | 50.4 - 79.4 |
| mlxturbo | 43.3 tok/s | 40.0 - 49.8 |

## 文脈長を振る (128 トークン生成)

| 文脈 | decode mlx-serve | decode mlxturbo | cold TTFT mlx-serve | cold TTFT mlxturbo | warm TTFT mlx-serve | warm TTFT mlxturbo |
|---|---|---|---|---|---|---|
| 1k | 55.0 | 45.1 | 2.75s | 3.72s | 0.18s | **3.66s** |
| 4k | 50.4 | 37.3 | 3.73s | 4.80s | 0.20s | **4.76s** |
| 16k | 48.9 | 34.9 | 15.8s | 37.4s | 0.26s | **2.65s** |
| 48k | 43.3 | 30.2 | 55.6s | 99.5s | 0.22s | **6.56s** |

## 読み取れること

**1. 温まった TTFT が 10-30 倍違う。ここが最大の差。**

mlx-serve は文脈長に関係なく 0.2 秒前後で、プロンプトの状態を丸ごと持っていて
復元に計算がいらないことを示している。こちらは 2048 刻みのチェックポイント
までしか戻れず、必ず 1.6k-2k トークンを計算し直す。**1k のプロンプトには復元点が
一つも無く、毎回まるごと prefill している** (だから 1k の warm TTFT が 16k より
遅いという倒錯が起きる)。

なお「新しいプロンプトが処理済み列への純粋な追記」の場合 (エージェントの
2 ターン目以降) は 1 番目の pass が拾うので計算し直しは起きない。効いていないのは
共通の前置きを持つ独立したリクエストと、同じプロンプトの投げ直し。

**2. decode は 1.2-1.4 倍の差。文脈が伸びるほど開く。**

相手の `--mtp-depth` は適応的に 1 ラウンド最大 6-8 トークン引く。こちらは
depth 1。受理率 0.76 で depth 1 なら上限は 1.7 倍前後にしかならない。

**3. cold prefill は長文で 1.8-2.4 倍の差。**

相手の prefill チャンクはこのモデルで 4096 (8192 から自動縮小)、こちらは 2048。
加えて `--ane-prefill` で各チャンクの dense MLP 行の 4 割を Neural Engine に
逃がす経路を持つ。

## 相手が公言している代償

`--decode-attn-quant` (既定 ON): "LOSSY: a real requantization"。
`--ane-prefill`: "int8/fp16, lossy"。

速さのために品質を削っていることを自分で書いている。こちらの Block Verification
は厳密同一分布で、焼き直しレシピごとの KLD も測って出している。**速度で並ぶ前に、
この差を数字で示せる形にしておく価値がある。**

## 再現

```bash
# mlx-serve
mlx-serve --model ~/models/ddalcu-flashnext-serve-4bit --serve \
  --host 127.0.0.1 --port 11234 --mtp
# mlxturbo
python -m mlxturbo.server --model ~/models/qwen38fn-mlx-v-l \
  --ngram ~/models/qwen38fn-ngram-4bit \
  --mtp ~/models/qwen38fn-mlx-v-l/mtp.safetensors \
  --host 127.0.0.1 --port 11235
```

計測スクリプトは `tools/vs_engine.py` / `tools/vs_longctx.py`。
