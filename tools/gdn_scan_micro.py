"""GDN prefill scan の冷の連鎖 micro (kernel S 対 レジスタ常駐)。

CLAUDE.md の作法どおり:

- **冷**: 36 層ぶんの活性 (q/k/v/g/beta、1 組あたり約 46 MB、計 1.6 GB) を
  用意して巡回する。1 組を使い回す温のマイクロベンチだと、並列度の足りない
  カーネルが DRAM 潜伏を隠せない負けが見えない (HC 融合の前例)。
- **連鎖**: 各ステップの `state_out` を次のステップの `state_in` に渡し、
  Python 側でグラフを組みきってから 1 回だけ `mx.eval` する。eval 時間を
  ステップ数で割ったものが 1 チャンク (T=2048) あたりの費用。
- **ABBA**: 1 プロセス内で回文順に交互に測り、中央値を取る。

絶対値は信じないこと。ここで見るのは実装どうしの比だけで、採否は
`tools/decode_ab.py --knob gdn-scan-reg` の in-model A/B で決める。

    tools/biglock.sh .venv/bin/python tools/gdn_scan_micro.py
    tools/biglock.sh .venv/bin/python tools/gdn_scan_micro.py --sweep    # 割り当て
    tools/biglock.sh .venv/bin/python tools/gdn_scan_micro.py --sweep2   # 追い込み
    tools/biglock.sh .venv/bin/python tools/gdn_scan_micro.py --sweep3   # acc/preq

2026-09-03 の結果 (結論とその読みは `mlxturbo/kernels/gdn_scan_reg.py` の
docstring): 底は lanes=4 db=32 の **x0.875** で、判定線 0.7 に届かない。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import mlx.nn as nn

import mlxturbo  # noqa: F401
from mlx_lm.models.gated_delta import compute_g
from mlxturbo.kernels import gdn_blocked_metal as gbm
from mlxturbo.kernels import gdn_scan_reg as gsr

# Qwen3.8-Flash-Next の GDN の形と層数 (config.json の layer_types:
# 48 層のうち full_attention が 12、linear_attention が 36)
N_K, N_V, DK, DV = 16, 48, 128, 128
N_LAYERS = 36

# (lanes, db, keep_k)。keep_k=0 は k をレジスタに抱えず delta の後にもう一度
# 読む (レジスタが半分になるかわりに L1 読みが 2 倍)。
SWEEP_LAYOUTS = [
    (32, 8, 1),
    (16, 16, 1),
    (8, 32, 1),
    (4, 64, 1),
    (4, 32, 1),
    (2, 128, 1),
    (2, 64, 1),
]

# 第 3 巡: 命令レベルの並列度 (独立累算器 acc、q の先読み preq) を足す。
# (lanes, db, keep_k, acc, preq)
SWEEP3_LAYOUTS = [
    (4, 32, 1, 1, 0),
    (4, 32, 1, 2, 0),
    (4, 32, 1, 4, 0),
    (4, 32, 1, 8, 0),
    (4, 32, 0, 4, 1),
    (4, 32, 1, 4, 1),
    (4, 16, 1, 4, 1),
    (8, 32, 1, 4, 0),
    (8, 32, 1, 4, 1),
    (2, 32, 0, 4, 1),
]

# 第 2 巡: 第 1 巡の傾向 (lanes=4 が底、threadgroup は小さいほうが速い) を
# 追う。keep_k=0 でレジスタを半分にした版も見る。
SWEEP2_LAYOUTS = [
    (4, 32, 1),
    (4, 16, 1),
    (4, 8, 1),
    (8, 16, 1),
    (8, 8, 1),
    (4, 32, 0),
    (4, 16, 0),
    (2, 64, 0),
    (2, 32, 0),
    (2, 16, 0),
]


def build_acts(n_sets: int, T: int, dtype, seed: int = 11):
    """層ごとに別の活性を作る (冷やすため)。1 組 = 1 GDN 層 1 チャンクぶん。"""
    mx.random.seed(seed)
    inv = DK**-0.5
    acts = []
    for _ in range(n_sets):
        q = (inv**2) * mx.fast.rms_norm(
            mx.random.normal((1, T, N_K, DK)).astype(dtype), None, 1e-6
        )
        k = inv * mx.fast.rms_norm(
            mx.random.normal((1, T, N_K, DK)).astype(dtype), None, 1e-6
        )
        v = nn.silu(mx.random.normal((1, T, N_V, DV)).astype(dtype))
        a = mx.random.normal((1, T, N_V)).astype(dtype)
        b = mx.random.normal((1, T, N_V)).astype(dtype)
        A_log = mx.random.normal((N_V,)) * 0.5
        dt_bias = mx.ones((N_V,))
        g = compute_g(A_log, a, dt_bias).astype(mx.float32)
        beta = mx.sigmoid(b).astype(mx.float32)
        mx.eval(q, k, v, g, beta)
        acts.append((q, k, v, g, beta))
    state0 = (mx.random.normal((1, N_V, DV, DK)) * 0.1).astype(mx.float32)
    mx.eval(state0)
    nbytes = sum(x.nbytes for t in acts for x in t) / 1e6
    return acts, state0, nbytes


def make_call(kind, lanes, db, keep_k=True, acc=1, preq=False):
    if kind == "blocked":
        return lambda q, k, v, g, beta, st: gbm.gated_delta_blocked_seq(
            q, k, v, g, beta, st
        )
    return lambda q, k, v, g, beta, st: gsr.gated_delta_scan_reg(
        q, k, v, g, beta, st, lanes=lanes, db=db, keep_k=keep_k, acc=acc, preq=preq
    )


def time_chain(call, acts, state0, steps: int):
    """連鎖を組んで 1 回だけ eval する。(構築 ms, eval ms/step) を返す。"""
    t0 = time.perf_counter()
    st = state0
    ys = []
    for i in range(steps):
        q, k, v, g, beta = acts[i % len(acts)]
        y, st = call(q, k, v, g, beta, st)
        ys.append(y)
    build_ms = (time.perf_counter() - t0) * 1e3
    t1 = time.perf_counter()
    mx.eval(st, *ys)
    eval_ms = (time.perf_counter() - t1) * 1e3
    del ys
    return build_ms, eval_ms / steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="割り当ての全組を掃く")
    ap.add_argument("--sweep2", action="store_true", help="第 2 巡の組を掃く")
    ap.add_argument("--sweep3", action="store_true", help="第 3 巡 (acc/preq) を掃く")
    ap.add_argument("--layers", type=int, default=N_LAYERS, help="巡回させる活性の組数")
    ap.add_argument("--steps", type=int, default=72, help="連鎖のステップ数")
    ap.add_argument("--pairs", type=int, default=3, help="回文順を何周するか")
    ap.add_argument("--tokens", type=int, default=2048, help="1 チャンクの T")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    acts, state0, nbytes = build_acts(args.layers, args.tokens, mx.bfloat16)
    print(
        f"活性 {args.layers} 組 x {nbytes / args.layers:.1f} MB = {nbytes:.0f} MB "
        f"(T={args.tokens}, 連鎖 {args.steps} 歩、ABBA x {args.pairs})"
    )

    variants: list[tuple[str, object]] = [("blocked", make_call("blocked", 0, 0))]
    if args.sweep3:
        layouts = SWEEP3_LAYOUTS
    elif args.sweep2:
        layouts = SWEEP2_LAYOUTS
    elif args.sweep:
        layouts = SWEEP_LAYOUTS
    else:
        layouts = [(gsr.DEFAULT_LANES, gsr.DEFAULT_DB, int(gsr.DEFAULT_KEEP_K))]
    for spec in layouts:
        lanes, db, keep = spec[0], spec[1], spec[2]
        acc = spec[3] if len(spec) > 3 else 1
        preq = spec[4] if len(spec) > 4 else 0
        ok, why = gsr.layout_ok(DK, DV, lanes, db)
        if not ok:
            print(f"  reg l{lanes} db{db}: 対象外 ({why})")
            continue
        tag = (f"reg-l{lanes}-db{db}" + ("" if keep else "-nok")
               + (f"-a{acc}" if acc > 1 else "") + ("-pq" if preq else ""))
        variants.append(
            (tag, make_call("reg", lanes, db, bool(keep), acc, bool(preq)))
        )

    # 暖機 (カーネルのコンパイルを計測から外す)
    for _name, call in variants:
        time_chain(call, acts, state0, 2)

    samples: dict[str, list[float]] = {name: [] for name, _ in variants}
    order = variants + variants[::-1]  # 回文順 (ABBA)
    for _ in range(args.pairs):
        for name, call in order:
            _build_ms, per = time_chain(call, acts, state0, args.steps)
            samples[name].append(per)

    base = statistics.median(samples["blocked"])
    print(f"\n1 チャンク (T={args.tokens}, 36 層ぶんを巡回) あたりの us:")
    rows = []
    for name, _ in variants:
        med = statistics.median(samples[name])
        lo, hi = min(samples[name]), max(samples[name])
        ratio = med / base
        mark = "  <- 判定線 0.7 を通る" if name != "blocked" and ratio <= 0.7 else ""
        print(
            f"  {name:14s} {med * 1e3:8.1f} us  "
            f"(min {lo * 1e3:7.1f} / max {hi * 1e3:7.1f})  x{ratio:.3f}{mark}"
        )
        rows.append({"name": name, "us": med * 1e3, "ratio": ratio,
                     "samples_us": [s * 1e3 for s in samples[name]]})

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {"tokens": args.tokens, "layers": args.layers, "steps": args.steps,
                 "pairs": args.pairs, "act_mb": nbytes, "rows": rows},
                indent=2, ensure_ascii=False,
            )
        )
        print(f"\n-> {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
