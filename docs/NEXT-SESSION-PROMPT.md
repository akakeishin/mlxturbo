# 次のセッション用プロンプト (Fable 親、実装は Sonnet)

以下をそのまま新セッションの最初の入力にする。

---

## 依頼

mlxturbo (`/Users/ht/dev/fastmlx`) が対戦相手 mlx-serve (`~/dev/mlx-serve`) に
**速度で負けている**。その差を埋めて追い越すところまで、自律的に進めてほしい。

**やり方の指定はしない。**相手のソースを読むのも、うちを解剖するのも、
相手が採っていない手を考えるのも、全部あなたの裁量。**ただし下の「守ること」は
破らないこと。**

## いまの現在地 (2026-09-02 実測、条件をそろえた単独測定)

`bench/self_snapshot.py` で両者を同じ手順で 1 つずつ測った値。
mlx-serve は最新 (`8058076`、YaRN 1M) をビルド済み。

| 文脈 | 冷 TTFT serve / turbo | 温 TTFT serve / turbo | decode serve / turbo | prefill tok/s serve / turbo |
|---|---|---|---|---|
| 0 | 0.17 / **0.69** | 0.72 / **1.16** | 55.0 / **45.3** | 109 / **28** |
| 4k | 2.99 / **7.30** | 0.85 / **1.21** | 54.1 / **42.5** | 1275 / **523** |
| **17k** | **10.91 / 21.48** | 0.88 / **1.35** | 49.1 / **37.6** | 1541 / **783** |
| **50k** | **36.77 / 68.51** | 1.55 / **1.96** | 45.0 / **35.7** | 1355 / **727** |

**prefill が約半分、decode が約 4 分の 3。**品質は逆に**うちが 3 倍良い**
(KLD 0.00445 対 0.01308、top-1 98.12%)。

再現コマンド:

    tools/biglock.sh .venv/bin/python bench/self_snapshot.py \
      --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram \
      --ctxs 0,4000,17000,50000 --tokens 256 --reps 2

    tools/biglock.sh .venv/bin/python bench/self_snapshot.py \
      --serve-bin ~/dev/mlx-serve/zig-out/bin/mlx-serve \
      --serve-model ~/models/ddalcu-flashnext-serve-4bit \
      --model ~/models/ddalcu-mlxlm --ctxs 0,4000,17000,50000 --tokens 256 --reps 2

## 前のセッションが確かめたこと (再調査は不要。ただし疑ってよい)

**`docs/research/KERNEL-PROGRAM.md` の冒頭「読む順」から入ること。**1650 行あるが、
最初の 40 行と「このプログラムの答え」を読めば全体が分かる。

要点だけ:

- **prefill 35s の内訳と実質の伸びしろ**: MoE 1.4-1.7s / GDN 射影 0.6s (効率 88%) /
  GDN スキャン 約 1s / attention 2.1s / HC 0s (取る手が無い)。**合計 5s 前後**
- **decode に未帰属は無い。**段階投入 (`STAGE_EVERY=2`) が既に 18% 取っている
- **相手は別の下限で走っている。**`gated_delta_blocked_seq` (seq>=64 の prefill 幅
  専用、ブロック化スキャン) は逐次の FLOP を仮定したうちの下限を割れる。
  相手の 536ms/チャンクに対しうちの計算上の下限が 762ms
- **カーネルを 5 本書いて採用 1 本** (HC 書き戻しの -0.8%)。融合で dispatch を
  減らす手はこの系でほとんど効かない
- **「負けたから off」の knob 6 件を測り直して、判定の変更ゼロ**
- **バッチは配線済みだが実運用の形では割に合わない** (短いプロンプト x B=4 で
  +19%、1815 トークン x B=2 では壁時計が変わらない)

**5s の積み上げでは 2 倍差は埋まらない。**別の下限に移るか、うちが見落として
いる何かがある。**そこを見つけるのがこのセッションの仕事。**

## 守ること (破ると数字が嘘になる。全部この 2 日で実際に踏んだ)

`/Users/ht/dev/fastmlx/CLAUDE.md` の「計測の作法」を必ず読むこと。加えて:

1. **率は時間ではない。**転嫁率 0.30 なら 60% の無駄は 18% の時間。
   「効率が低い = 伸びしろがある」ではない (2026-09-02 に 3 例)
2. **効率の分母が何を仮定しているか確かめる。**機械の天井で割ると
   アルゴリズムの選択ミスが効率の数字に出ない
3. **単体の測定は「その op の中での改善率」であって「ラウンドへの寄与」ではない**
   (`gather_qmm` 96% / HC 52% / GDN スキャン 20% / `fast_qmm` の qkv -22%、
   いずれもラウンドでは 1% 以下)
4. **発火を確かめてから数字を読む。**`mlxturbo/kernels/_fire.py` の発火カウンタと
   `/health` の `spec_batch`。**「効果ゼロ」の 3 件が「動いていなかった」だった**
5. **数値を変えるカーネルは `ms/round` ではなく `tok/round` で判定する。**
   投機デコードでは受理率が品質の指標そのもの
6. **道具自体を対照で検査する。**`--knob null` (何もしない) と、機構だけの
   対照 C の型がある。**A/B が長文脈で A 側に +5.6% の下駄を履かせていた**
7. **同じものを 2 回測ると 2 回目は違う条件で走る** (接頭辞キャッシュ)。
   温めは測定と別のプロンプト、繰り返しごとに別の窓
8. **`tools/biglock.sh` を入れ子にしない** (自分のロックを 127 分待った)
9. **`--prefill-once` は `DECODE_ONLY_KNOBS` の knob だけ**

## 分業

- **実装は Sonnet のサブエージェント**に出す。親 (Fable) は方針と計測の判定と
  commit を持つ
- 横断検索・ファイル特定は `scout`
- 判断に分岐があるときだけ `opus-advisor` (親が Fable なので Opus 側を呼ぶ)

## 使える道具 (この 2 日で整備した)

| | |
|---|---|
| `bench/self_snapshot.py` | 両エンジンを同じ手順で単独測定 |
| `tools/decode_ab.py` | knob 式 A/B。回文順、グループごとの温め捨て、発火表示 |
| `tools/prefill_anatomy.py` | prefill を部品に割る。MoE は中身まで |
| `tools/decode_anatomy.py` | decode を部品に割る |
| `tools/gpu_fingerprint.py` | GPU 分岐の一次検査。**発火 0 は不合格** |
| `tools/vendor_fingerprint.py` | CPU の一次検査。**GPU 分岐は保証しない** |
| `tools/calibrate.py` | 原始量 6 つを測って閾値を式から出す |
| `tools/probe_tile_padding.py` | タイル水増しの因果 |
| `tools/probe_moe_pressure.py` | メモリ圧の仮説検証 |
| `tools/biglock.sh` | GPU の直列化とメモリ待ち |
| `bench/quant_eval.py` | KLD。**品質を売って速度を買わない** |

## 期待する成果

**mlx-serve を超える。**超えられないなら、**なぜ超えられないかを実測で示す。**

どちらでも、**判断が反転する条件を先に宣言してから測ること。**
