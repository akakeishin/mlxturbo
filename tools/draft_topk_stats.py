"""`decode_ab --draft-topk K --topk-trace <path>` が吐いた JSON Lines を集計する。

1 行 = 1 ラウンドで、形は::

    {"round": 3, "depth": 4, "hit": 2, "margins": [...], "pos": 812,
     "prompt_id": "short:0",
     "topk": [[[粗上位K], [再採点後上位K]], ...],   # 段ごと (長さ depth)
     "true": [t0, t1, t2, t3, t4]}                  # 長さ depth+1

## 段と真のトークンの対応

段 i (0-indexed) の draft は「先行する i 個の draft を前置きした条件」で
引いたもの。検証フォワードは同じ列 [cur, d1..dk] を流すので、位置 i の
argmax ``true[i]`` は**まさにその条件下でのトランクの答え**になる。つまり
受理が途中で切れたラウンドでも、段 i の命中判定に ``true[i]`` をそのまま
使える (切れた後の段も「その prefix なら何が正解だったか」は定義できる)。

**木 (複数候補) の上限を読むときの注意**: 段 i>=1 の候補集合は「段 i-1 の
top-1 を前置きした」条件のもの。木が段 1 で別の枝を選べば段 2 の prefix は
変わるので、下の top-k の連鎖は**その分だけ楽観**。段 1 の数字だけは条件が
無いので厳密。

## 出力

    (a) 段ごとの top-1/2/4/8 命中率 (再採点後と、rerank 前の粗ヘッド)
    (b) 連続受理の分布 P(chain >= m) と、そこから出る tok/round
    (c) top-k を取れたと仮定したときの tok/round の上限
    (d) 行費用モデル (ms/round = base + row_ms * (S - 2)) を当てた ms/tok

使い方::

    .venv/bin/python tools/draft_topk_stats.py bench/results/topk-trace.jsonl \\
        --kind short --base-ms 24.0 --row-ms 3.0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

KS = (1, 2, 4, 8)


def load(path, kind=None, depth=None):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("prompt_id") or ""
            if pid.startswith("warmup:"):
                continue
            if kind and not pid.startswith(kind + ":"):
                continue
            if not r.get("topk") or not r.get("true"):
                continue
            if depth is not None and r["depth"] != depth:
                continue
            recs.append(r)
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--kind", default=None,
                    help="prompt_id の接頭辞で絞る (short / long)")
    ap.add_argument("--depth", type=int, default=None,
                    help="この深さで引いたラウンドだけ見る")
    ap.add_argument("--base-ms", type=float, default=None,
                    help="S=2 (depth 1) のときの ms/round")
    ap.add_argument("--row-ms", type=float, default=None,
                    help="検証幅 1 行あたりの追加費用 (ms)")
    args = ap.parse_args(argv)

    recs = load(args.path, args.kind, args.depth)
    if not recs:
        print("該当するラウンドが無い")
        return 1
    D = max(r["depth"] for r in recs)
    recs = [r for r in recs if r["depth"] == D and len(r["topk"]) == D
            and len(r["true"]) >= D and all(t is not None for t in r["topk"])]
    n = len(recs)
    pids = sorted({r["prompt_id"] for r in recs})
    print(f"{args.path}  kind={args.kind or '全部'}  depth={D}  "
          f"{n} ラウンド  prompt {len(pids)} 本 ({', '.join(pids)})")

    # ---- (a) 段ごとの命中率 -----------------------------------------
    hits = defaultdict(int)     # (d, k, which) -> count
    for r in recs:
        for i in range(D):
            coarse, rer = r["topk"][i]
            t = r["true"][i]
            for k in KS:
                if rer[:k].count(t):
                    hits[(i, k, "rerank")] += 1
                if coarse is not None and coarse[:k].count(t):
                    hits[(i, k, "coarse")] += 1
            # 粗 top-32 の中に真のトークンが居るか (rerank の天井)
            if coarse is not None and t in coarse:
                hits[(i, 0, "pool")] += 1

    print("\n=== (a) 段ごとの命中率 (真のトークンが draft の上位 k に入る率) ===")
    print(f"  {'段':>3s}  " + "  ".join(f"top-{k}".rjust(7) for k in KS)
          + "     " + "  ".join(f"粗top-{k}".rjust(8) for k in KS))
    for i in range(D):
        a = "  ".join(f"{hits[(i, k, 'rerank')] / n:7.3f}" for k in KS)
        b = "  ".join(f"{hits[(i, k, 'coarse')] / n:8.3f}" for k in KS)
        print(f"  d={i + 1:d}  {a}     {b}")
    print("  (粗 top-K は rerank 前 = 2bit 頭のスコア順。"
          "top-8 は粗 top-32 を trunk 頭で再採点した後の順)")
    gain = "  ".join(
        f"d={i + 1}:{(hits[(i, 1, 'rerank')] - hits[(i, 1, 'coarse')]) / n:+.3f}"
        for i in range(D))
    print(f"  rerank が top-1 に足している分 (再採点後 - 粗): {gain}")

    # ---- (b) 連続受理 -------------------------------------------------
    # 連鎖 m = 段 1..m が全部 top-1 で当たる。engine の hit と一致するはず
    chain = [0] * (D + 1)
    mismatch = 0
    for r in recs:
        m = 0
        while m < D and r["topk"][m][1][0] == r["true"][m]:
            m += 1
        for j in range(m + 1):
            chain[j] += 1
        if m != r["hit"]:
            mismatch += 1
    print("\n=== (b) 連続受理 (top-1 だけで何段続くか) ===")
    print("  m       P(chain>=m)")
    for m in range(D + 1):
        print(f"  {m}    {chain[m] / n:11.4f}")
    print(f"  engine の hit と食い違ったラウンド: {mismatch}/{n}"
          f"{'  ** 対応付けが疑わしい **' if mismatch else '  (対応 OK)'}")
    tpr = [1.0 + sum(chain[m] / n for m in range(1, d + 1)) for d in range(D + 1)]
    print("  depth ごとの tok/round (この trace から):  "
          + "  ".join(f"d={d}:{tpr[d]:.3f}" for d in range(1, D + 1)))

    # ---- (c) top-k を取れたときの上限 ---------------------------------
    print("\n=== (c) top-k を全部取れたと仮定したときの上限 ===")
    print("  k    " + "  ".join(f"P(>= {m})".rjust(9) for m in range(1, D + 1))
          + "   tok/round(d=2)  tok/round(d=4)")
    for k in KS:
        c = [0] * (D + 1)
        for r in recs:
            m = 0
            while m < D and r["true"][m] in r["topk"][m][1][:k]:
                m += 1
            for j in range(m + 1):
                c[j] += 1
        row = "  ".join(f"{c[m] / n:9.4f}" for m in range(1, D + 1))
        t2 = 1.0 + sum(c[m] / n for m in range(1, 3))
        t4 = 1.0 + sum(c[m] / n for m in range(1, D + 1))
        print(f"  {k:<3d}  {row}   {t2:13.3f}  {t4:14.3f}")
    print("  (段 2 以降は「段 1 の top-1 を前置きした」条件の候補集合なので、"
          "木の上限としては楽観側)")

    # ---- (d) 行費用を当てる -------------------------------------------
    if args.base_ms is not None and args.row_ms is not None:
        print("\n=== (d) 行費用 (ms/round = base + row_ms * (S-2)、S = depth+1) ===")
        print(f"  base(S=2)={args.base_ms} ms  row={args.row_ms} ms/行")
        print("  depth   ms/round   tok/round   ms/tok")
        for d in range(1, D + 1):
            ms = args.base_ms + args.row_ms * (d + 1 - 2)
            print(f"  {d:<6d}{ms:9.2f}{tpr[d]:12.3f}{ms / tpr[d]:9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
