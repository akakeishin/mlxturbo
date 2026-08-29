"""エキスパートの点火分布を測る。1 回の測定で 3 つの設計が決まる。

1. **MTP の深さ** — 一括 forward が S にどれだけ比例して高くなるかは、
   隣り合うトークンが同じエキスパートを引くかで決まる。独立なら S=16 で
   512 x (1 - (1 - 10/512)^16) = 約 138 個だが、相関があればもっと少なく、
   一括検証はそのぶん安くなる。docs/STATUS.md の訂正節で独立を仮定した
   見積もりを置いたので、実測で置き換える
2. **ビット配分** — よく引かれるエキスパートに厚く盛る。全部一律に配るのは、
   点火頻度に偏りがある場合には損
3. **小容量 Mac** — 熱いものを RAM に置いて残りをディスクに流す構成が
   成立するかどうか。llama.cpp のオフロードが実際にやっていること

    uv run python tools/expert_stats.py --model <m> --ngram <s>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "bench" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--continuations", default=str(RESULTS / "qe-cont.json"))
    ap.add_argument("--out", default=str(RESULTS / "expert-stats.json"))
    args = ap.parse_args()

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q
    import numpy as np
    from mlx_lm import load

    model, _ = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)

    # 層ごとに (位置, top_k) の添字を溜める。層の同一性は呼ばれた順で決まるので、
    # forward 1 回のうち何番目に呼ばれたかを数える
    captured: list[list[np.ndarray]] = []
    call_no = {"i": 0}
    orig = Q.SparseMoeBlock.__call__

    def recording(self, x):
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, self.top_k - 1, axis=-1)[..., : self.top_k]
        w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
        mx.eval(idx)
        i = call_no["i"]
        call_no["i"] += 1
        while len(captured) <= i:
            captured.append([])
        captured[i].append(np.array(idx).reshape(-1, self.top_k))
        out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        return out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    Q.SparseMoeBlock.__call__ = recording

    cont = json.loads(Path(args.continuations).read_text())
    n_prompts = 0
    for key, entry in cont["prompts"].items():
        ids = entry["prompt_ids"] + entry["continuation_ids"]
        call_no["i"] = 0
        cache = model.make_cache()
        mx.eval(model(mx.array(ids)[None], cache=cache))
        n_prompts += 1
        print(f"  {key}: {len(ids)} 位置", flush=True)

    Q.SparseMoeBlock.__call__ = orig

    n_layers = len(captured)
    n_experts = model.model.args.num_experts
    top_k = model.model.layers[0].mlp.top_k
    print(f"\n{n_layers} 層 x {n_prompts} プロンプト、top_k={top_k}、"
          f"experts={n_experts}\n")

    # --- 1. 隣り合うトークンの重なり (MTP の深さを決める) ---
    # 位置 t と t+1 が同じエキスパートを何個共有するか。独立なら
    # top_k^2 / n_experts = 100/512 = 0.195 個
    shared_tot, pairs = 0, 0
    # S 個の窓に現れる相異なるエキスパート数 (一括検証の読み出しに直結)
    distinct_by_S: dict[int, list[int]] = {s: [] for s in (1, 2, 4, 8, 16)}
    for layer_rows in captured:
        for arr in layer_rows:
            for t in range(len(arr) - 1):
                shared_tot += len(set(arr[t]) & set(arr[t + 1]))
                pairs += 1
            for s in distinct_by_S:
                for t in range(0, len(arr) - s + 1, max(1, s)):
                    distinct_by_S[s].append(len(set(arr[t : t + s].ravel())))

    print("隣接トークンの重なり")
    print(f"  実測 {shared_tot / max(pairs, 1):.2f} 個 / {top_k} "
          f"(独立なら {top_k * top_k / n_experts:.2f} 個)")

    print("\nS トークンの窓に現れる相異なるエキスパート数")
    print(f"  {'S':>3s} {'実測':>8s} {'独立仮定':>10s} {'比':>7s}")
    rows = {}
    for s in sorted(distinct_by_S):
        vals = distinct_by_S[s]
        got = float(np.mean(vals)) if vals else 0.0
        indep = n_experts * (1 - (1 - top_k / n_experts) ** s)
        rows[s] = {"measured": got, "independent": indep}
        print(f"  {s:3d} {got:8.1f} {indep:10.1f} {got / indep:7.2f}x")

    # --- 2. 点火頻度の偏り (ビット配分と RAM 常駐の設計) ---
    print("\n層ごとの点火の偏り (上位が全点火の何 % を占めるか)")
    print(f"  {'層':>4s} {'上位32':>8s} {'上位64':>8s} {'上位128':>8s} {'未点火':>8s}")
    per_layer = []
    for i, layer_rows in enumerate(captured):
        c = Counter()
        for arr in layer_rows:
            c.update(arr.ravel().tolist())
        total = sum(c.values())
        ranked = [n for _, n in c.most_common()]
        cum = np.cumsum(ranked)
        def pct(k):
            return 100 * (cum[min(k, len(cum)) - 1] / total) if total else 0.0
        rec = {
            "layer": i,
            "top32": pct(32), "top64": pct(64), "top128": pct(128),
            "never": n_experts - len(c),
        }
        per_layer.append(rec)
        if i % 8 == 0 or i == len(captured) - 1:
            print(f"  {i:4d} {rec['top32']:7.1f}% {rec['top64']:7.1f}% "
                  f"{rec['top128']:7.1f}% {rec['never']:8d}")

    print(f"\n  全層平均: 上位32 {np.mean([r['top32'] for r in per_layer]):.1f}%, "
          f"上位64 {np.mean([r['top64'] for r in per_layer]):.1f}%, "
          f"上位128 {np.mean([r['top128'] for r in per_layer]):.1f}%")
    print(f"  一度も引かれなかった数の平均: "
          f"{np.mean([r['never'] for r in per_layer]):.0f} / {n_experts}")
    print(f"\n  注: 標本はプロンプト {n_prompts} 本ぶんで、点火の裾は測りきれて"
          "いない。「未点火」は\n      この標本で引かれなかっただけで、"
          "死んでいる証拠ではない。")

    # --- 3. 熱いものを RAM、残りをディスクに流す構成が成立するか ---
    # 常駐率 h のとき、1 トークンあたりディスクから読む量:
    #   48 層 x top_k x (1 - カバー率) x エキスパート 1 個のバイト数
    per_expert_mb = 3 * 640 * 2560 * 0.5 / 1e6  # 4bit
    print("\n熱いエキスパートだけ RAM に置いた場合のディスク読み出し")
    print(f"  {'常駐数':>8s} {'カバー率':>9s} {'MB/token':>10s} {'@5GB/s':>9s}")
    for k, cov in (
        (32, np.mean([r["top32"] for r in per_layer])),
        (64, np.mean([r["top64"] for r in per_layer])),
        (128, np.mean([r["top128"] for r in per_layer])),
    ):
        cold = n_layers * top_k * (1 - cov / 100) * per_expert_mb
        print(f"  {k:8d} {cov:8.1f}% {cold:10.0f} {cold / 5000 * 1000:8.0f}ms")
    print("  (逐次デコードは 34-50ms/token。ここに乗せられる余地はほぼ無い)")

    Path(args.out).write_text(json.dumps({
        "model": args.model,
        "n_layers": n_layers, "n_experts": n_experts, "top_k": top_k,
        "n_prompts": n_prompts,
        "adjacent_overlap": shared_tot / max(pairs, 1),
        "adjacent_overlap_independent": top_k * top_k / n_experts,
        "distinct_by_S": rows,
        "per_layer": per_layer,
    }, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
