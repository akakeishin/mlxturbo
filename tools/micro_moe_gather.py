"""decode の MoE 48 層が帯域下限の 50% しか出ない理由を切り分ける
(docs/research/KERNEL-PROGRAM.md 段 5)。

## 背景

decode 幅の実形状で `gather_qmm` (実体は `gather_qmv_fast` / `gather_qmv`)
を測ると、達成帯域 393GB/s に対して 50% しか出ていない。moe_glu v1-v3・
共有タイル v2・moe_route と、カーネルは 3 回書いて 3 回負けている
(`tools/probe_gather_qmm.py` の前例)。今度は**カーネルから入らず**、
「なぜ素の実装が下限に届かないのか」を先に切り分ける。

## 何で切り分けるか

1. decode 幅の実形状 (M=2 位置、E=512 バンク、union ~16、K=2560/640) での
   `gather_qmm` の単体帯域
2. **同じバイト数を密の `mx.quantized_matmul` で読んだときの帯域**
   (gather を経由しない、あらかじめ union 個ぶんだけを連結した密な重み)
3. 差の帰属:

   - 密が高効率 (~90%) で gather が低い (~50%) -> **添字が犯人**。
     専門家の重みを連続領域へ集めてから密 qmm に渡す案が成立しうる
   - 密も低い (~50%) -> **MLX の 4bit qmv 自体の天井**。「彼らのカーネルが
     1.4 倍速い」という既知の事実と符合し、専用カーネルを書く多日レーンが
     確定する

`tools/probe_gather_qmm.py` (M=1、top_k を振ってバンク依存を見る) を土台に
しているが、あちらは top_k=固定・M=1 の抽象形状だった。ここは decode の
**実際の呼び出し**に合わせ、M=2 位置それぞれが top_k=10 を引いた union を
測る。同じことの二度書きを避けるため、バンク依存の掃引はあちらに任せ、
ここでは M=2・union 制御・gate/up と down の両方向に絞る。

## union の作り方

`mlx_lm.models.switch_layers.SwitchGLU` は 2 位置なら `indices.size = 20 < 64`
で `do_sort=False` (ソート無し経路)。ここも `sorted_indices=False` を既定にする
(`--sorted-indices` で両方測れる)。2 位置それぞれの top_k 添字は独立ではなく
重なるので、実際に読まれる専門家数は `top_k <= union <= 2*top_k` の範囲に
収まる (`docs/research/KERNEL-PROGRAM.md` の実測: 3 トークンで union 21-22/30、
2 トークンへの内挿で ~16/20)。union の実数は `tools/expert_stats.py` の出力
(`bench/results/expert-stats.json` の `distinct_by_S["2"]`) があればそれを使い、
無ければ `--union` で明示するか、上の内挿値 16 に落ちる。

union を指定値ちょうどにするため、2 位置の添字は「共有プール」から作る:
重なり数 `overlap = 2*top_k - union` を固定し、両位置で `overlap` 個を共有、
残りを別々に引く。union=top_k なら 2 位置が完全一致、union=2*top_k なら
完全に独立 (現実の分布はこの間のどこか)。

## 制約 (このリポジトリの計測の作法)

- **温めてから測る。最初の数回は捨てる** (`bench_chain` の warmup)。
- `mx.eval` は 1 回あたり ~160us の固定費が乗るので、CHAIN 本を直列に積んで
  1 回だけ eval し、割って求める (`tools/probe_gather_qmm.py` と同じ手口)。
- 呼び出しごとに添字/入力を変えて CSE を避ける。
- **この道具の結果だけで採否を決めないこと。** ここでは micro で勝って
  in-model で負けた前例が複数ある (moe_glu 系)。目的は「どこが遅いかの
  切り分け」であって、カーネルを書くかどうかの判定ではない。採否は
  実機の in-model A/B (複数プロンプト x 512 平均) で決める。

    uv run python tools/micro_moe_gather.py --model ~/models/qwen38fn-mlx-v-fast6
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# docs/research/KERNEL-PROGRAM.md の「達成帯域」。理論値 400 ではなく実測の到達値
SUSTAINED_GBPS = 393.0

# `mx.eval` 1 回には約 160us の固定費が乗る (probe_gather_qmm.py の実測)。
# 1 呼び出しごとに eval すると測りたい 30-80us がその中に埋もれるので、
# CHAIN 回を直列に積んでから 1 回だけ eval して割る。
CHAIN = 48


def bench_chain(make, n: int = 8, reps: int = 5) -> float:
    """make(i) が i 番目の呼び出しを作る。CHAIN 本積んで 1 回 eval、中央値を us で返す。"""
    import mlx.core as mx

    def go():
        return [make(i) for i in range(CHAIN)]

    for _ in range(2):
        mx.eval(go())
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(n):
            mx.eval(go())
        out.append((time.perf_counter() - t) / n * 1e6 / CHAIN)
    return statistics.median(out)


def load_shape(model_path: Path) -> dict:
    """config.json から MoE の形状と量子化設定を取る (ハードコードしない)。"""
    cfg = json.loads((model_path / "config.json").read_text())
    q = cfg.get("quantization") or {}
    # 一律量子化 (bake.py の "default" レシピ) を想定。エキスパート専用の
    # 上書きキーがあれば拾い、無ければ全体設定にフォールバック
    # (tools/probe_lm_head_bw.py の `.get("lm_head", ...)` と同じ形)
    qe = q.get("experts", q.get("switch_mlp", q))
    return {
        "hidden": cfg["hidden_size"],
        "n_experts": cfg["num_experts"],
        "top_k": cfg["num_experts_per_tok"],
        "moe_inter": cfg["moe_intermediate_size"],
        "bits": qe.get("bits", q.get("bits", 4)),
        "group_size": qe.get("group_size", q.get("group_size", 64)),
    }


def load_union(args, top_k: int) -> tuple[int, str]:
    """decode 2 位置ぶんの union 実数を決める。優先順位: --union > 実測ファイル > 内挿既定値。"""
    lo, hi = top_k, 2 * top_k
    if args.union is not None:
        return max(lo, min(hi, args.union)), "--union"

    stats_path = Path(args.stats_file).expanduser()
    if stats_path.exists():
        try:
            data = json.loads(stats_path.read_text())
            row = data["distinct_by_S"][str(args.union_s)]
            return max(lo, min(hi, round(row["measured"]))), str(stats_path)
        except (KeyError, ValueError, TypeError):
            pass

    # docs/research/KERNEL-PROGRAM.md の内挿値: 3 トークンで union 21-22/30 の実測を
    # 2 トークンへ内挿した ~16/20。未実測であることを明示する
    return max(lo, min(hi, 16)), "既定値 (未実測、3トークン実測 21-22/30 からの内挿 ~16)"


def make_union_indices(rng: np.random.Generator, n_experts: int, top_k: int, union: int):
    """decode 2 位置ぶんの top_k 添字を、union がちょうど指定値になるように作る。

    重なり数 `overlap = 2*top_k - union` を共有プールから引き、残りを
    位置ごとに別々のプールから引く。union=top_k で完全一致、union=2*top_k
    で完全独立になる (現実の点火はこの間のどこか)。
    """
    union = max(top_k, min(2 * top_k, union))
    overlap = 2 * top_k - union
    pool = rng.choice(n_experts, union, replace=False)
    shared, rest = pool[:overlap], pool[overlap:]
    uniq = top_k - overlap
    row0 = np.concatenate([shared, rest[:uniq]])
    row1 = np.concatenate([shared, rest[uniq : 2 * uniq]])
    rng.shuffle(row0)
    rng.shuffle(row1)
    return row0.astype(np.uint32), row1.astype(np.uint32)


def run_direction(name: str, k_in: int, n_out: int, shape: dict, union: int,
                   sorted_indices: bool, n: int, reps: int) -> None:
    """1 方向 (gate/up あるいは down) ぶんの gather vs 密 を測る。"""
    import mlx.core as mx

    n_experts, top_k = shape["n_experts"], shape["top_k"]
    bits, gs = shape["bits"], shape["group_size"]

    mx.random.seed(0)
    dense = (mx.random.normal((n_experts, n_out, k_in)) * 0.02).astype(mx.bfloat16)
    w, s, b = mx.quantize(dense, group_size=gs, bits=bits)
    del dense
    mx.eval(w, s, b)

    rng = np.random.default_rng(0)
    idx_pairs = []
    for _ in range(CHAIN):
        row0, row1 = make_union_indices(rng, n_experts, top_k, union)
        idx_pairs.append(mx.array(np.stack([row0, row1])[None]))  # (1, 2, top_k)
    mx.eval(idx_pairs)

    xs = [(mx.random.normal((1, 2, k_in))).astype(mx.bfloat16) for _ in range(CHAIN)]
    mx.eval(xs)

    def g(i, xs=xs, idx_pairs=idx_pairs, sorted_indices=sorted_indices):
        return mx.gather_qmm(
            xs[i], w, s, b, rhs_indices=idx_pairs[i], transpose=True,
            group_size=gs, bits=bits, sorted_indices=sorted_indices,
        )

    us_gather = bench_chain(g, n=n, reps=reps)

    # 密の同バイト量: union 個ぶんだけを連結した重みを 1 回だけ作り、
    # gather を経由せずに測る (「もう集め終わっている」場合の天井)
    dw = (mx.random.normal((union * n_out, k_in)) * 0.02).astype(mx.bfloat16)
    qw, qs, qb = mx.quantize(dw, group_size=gs, bits=bits)
    del dw
    mx.eval(qw, qs, qb)

    def d(i, xs=xs, qw=qw, qs=qs, qb=qb):
        return mx.quantized_matmul(
            xs[i], qw, scales=qs, biases=qb, transpose=True,
            group_size=gs, bits=bits,
        )

    us_dense = bench_chain(d, n=n, reps=reps)

    ideal_bytes = union * n_out * k_in * bits / 8
    gbps_gather = ideal_bytes / us_gather / 1000
    gbps_dense = ideal_bytes / us_dense / 1000

    print(f"\n=== {name}  K={k_in} -> N={n_out}  union={union}  "
          f"理想 {ideal_bytes / 1e6:.2f}MB ===")
    print(f"  {'':>10s} {'us':>10s} {'GB/s':>8s} {'下限比%':>8s}")
    print(f"  {'gather':>10s} {us_gather:10.1f} {gbps_gather:8.1f} "
          f"{100 * gbps_gather / SUSTAINED_GBPS:7.1f}%")
    print(f"  {'密':>10s} {us_dense:10.1f} {gbps_dense:8.1f} "
          f"{100 * gbps_dense / SUSTAINED_GBPS:7.1f}%")
    print(f"  gather/密 倍率: {us_gather / us_dense:.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="config.json を含むモデルディレクトリ")
    ap.add_argument("--bits", type=int, default=None, help="既定は config.json から取る")
    ap.add_argument("--group-size", type=int, default=None, help="既定は config.json から取る")
    ap.add_argument("--union", type=int, default=None,
                    help="2 位置ぶんの union 実数。未指定なら --stats-file か内挿既定値")
    ap.add_argument("--union-s", type=int, default=2,
                    help="expert-stats.json の distinct_by_S のどの S を union として使うか")
    ap.add_argument("--stats-file", default=str(REPO_ROOT / "bench" / "results" / "expert-stats.json"))
    ap.add_argument("--sorted-indices", action="store_true",
                    help="sorted_indices=True 経路も測る (既定は decode M=2 の実経路に合わせ False のみ)")
    ap.add_argument("--n", type=int, default=8, help="bench_chain の内側ループ回数")
    ap.add_argument("--reps", type=int, default=5, help="bench_chain の外側繰り返し回数 (中央値を取る)")
    args = ap.parse_args()

    model_path = Path(args.model).expanduser()
    shape = load_shape(model_path)
    if args.bits is not None:
        shape["bits"] = args.bits
    if args.group_size is not None:
        shape["group_size"] = args.group_size

    union, union_src = load_union(args, shape["top_k"])

    print(f"モデル: {model_path}")
    print(f"hidden={shape['hidden']}  experts={shape['n_experts']}  "
          f"top_k={shape['top_k']}  moe_inter={shape['moe_inter']}  "
          f"{shape['bits']}bit gs={shape['group_size']}")
    print(f"union (2 位置ぶん) = {union}  [{union_src}]")
    print("\n下限は 2 の下限との比較専用の目安であって、この道具の結果だけで"
          "採否を決めないこと (micro 勝ち in-model 負けの前例が複数ある)。")

    directions = [
        ("gate/up", shape["hidden"], shape["moe_inter"]),
        ("down", shape["moe_inter"], shape["hidden"]),
    ]
    sorted_options = [False] + ([True] if args.sorted_indices else [])

    for name, k_in, n_out in directions:
        for sorted_indices in sorted_options:
            label = name if not sorted_indices else f"{name} (sorted_indices=True)"
            run_direction(label, k_in, n_out, shape, union, sorted_indices,
                          args.n, args.reps)

    print("\n判定の読み方 (docs/research/KERNEL-PROGRAM.md 段5):")
    print("  密が高効率で gather が低い -> 添字が犯人 (連続領域へ集めてから密 qmm の案が成立しうる)")
    print("  密も低い                 -> MLX の 4bit qmv 自体の天井 (専用カーネルを書く多日レーン)")


if __name__ == "__main__":
    main()
