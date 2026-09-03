"""HC の書き戻し畳み込み (旧 `MLXTURBO_HC_WRITE=fold`) の冷の連鎖マイクロ。

**この変種は 2026-09-03 に棄却され、実装は取り除いてある**
(`fused_gated_residual_elem_fold` はもう無いので、このファイルはそのままでは
走らない)。残してあるのは測った数字と、micro と in-model が逆に出た記録のため:

  冷の連鎖 (重み 97 組 367.5 MB、ABBA): m=1 で A 58.6 -> B 57.0 us/call (-1.6)、
  m=3 で 79.1 -> 77.3 (-1.8)。96 呼び出しなら -0.15 ms/round の見込み。
  in-model (短 3 本 x 512、`bench/results/hc-write-fold-short.log`):
  ms/round **+0.6%** で 3 ケースとも畳んだ側が遅い。tok/round は完全一致。

  逆になる見立て: `_combine` は 10240 要素の平たい elementwise で threadgroup が
  沢山立ち、隣の行列積と重なって走れる。畳むと、その仕事が「1 threadgroup =
  1 レーン」(S=1 では 4 threadgroup しか立たない) の pre の中に入って依存の
  直列に乗る。**前後に行列積の無い連鎖 micro はこの重なりを再現しない。**

以下は当時のまま。

比べるのは実モデルの 1 層ぶんの並び:

  A (既定) : h = _combine(hyper, x, inject)          … mx.compile 済み 1 本
             (mixed, inj) = elem(h)                   … 自前 3 本 + qmv 3 本
  B (fold) : (mixed, h, inj) = elem_fold(hyper, x, inject)
                                                      … 自前 3 本 + qmv 3 本
             (pre が h を作りながら二乗和まで進める)

連鎖は実モデルの再帰そのもの: 次の step の (hyper_prev, branch, inject_w) は
前の step の (h, mixed, inj)。CLAUDE.md の作法どおり **重みを層数ぶん
(既定 97 組 = 367 MB) 巡回させて冷やす** (`tools/micro_kernel_latency.py` の
`hc_weight_set` をそのまま使う。温の 1 組使い回しでは並列度の足りない自前
カーネルの負けが見えない)。呼び出し順は ABBA。

モデルは読まない。

    tools/biglock.sh uv run python tools/hc_write_fold_micro.py \
        --out bench/results/hc-write-fold-micro.json
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
sys.path.insert(0, str(REPO_ROOT / "tools"))

WARMUP = 4
N_PAIRS = 12          # ABBA を N_PAIRS 回 (1 標本 = 97 step の連鎖) -> 各側 2*N_PAIRS 標本


def _summ(samples, warmup):
    kept = samples[warmup:]
    return {
        "n": len(samples),
        "kept": len(kept),
        "median_us": round(statistics.median(kept), 3),
        "min_us": round(min(kept), 3),
        "max_us": round(max(kept), 3),
    }


def run(n_sets: int, m: int):
    import mlx.core as mx

    from micro_kernel_latency import (HC, HC_LOWRANK, HIDDEN, RMS_EPS,
                                      _weight_bytes, hc_weight_set)
    from mlxturbo import fused
    from mlxturbo.kernels.hyper_connection import fused_gated_residual_elem

    try:
        from mlxturbo.kernels.hyper_connection import \
            fused_gated_residual_elem_fold
    except ImportError:
        raise SystemExit(
            "fused_gated_residual_elem_fold が無い。この変種は 2026-09-03 に "
            "in-model (+0.6%) で棄却して実装ごと取り除いた (冒頭の注記)。"
            "測り直すなら畳んだ pre カーネルから書き直すこと。")

    dtype = mx.bfloat16
    weight_sets = hc_weight_set(dtype, n_sets)
    per_set_bytes = sum(_weight_bytes(t) for t in weight_sets[0])
    combine_fn = fused._build_combine(HC, HIDDEN)

    def init():
        hyper = (mx.random.normal((m, HC * HIDDEN)) * 0.5).astype(dtype)
        branch = (mx.random.normal((m, HIDDEN)) * 0.5).astype(dtype)
        inj = (mx.random.normal((m, HC)) * 0.5).astype(dtype)
        mx.eval(hyper, branch, inj)
        return hyper, branch, inj

    def step_a(state, k):
        """既定: _combine (1 本) のあとに elem の pre。"""
        hyper, branch, inj = state
        norm_weight, down, up, inject_w = weight_sets[k]
        h = combine_fn(hyper, branch, inj)
        mixed, inj2 = fused_gated_residual_elem(
            h, norm_weight, RMS_EPS, HC, HIDDEN, down, up, inject_w)
        return (h, mixed, inj2)

    def step_b(state, k):
        """fold: pre が書き戻しも兼ねる。"""
        hyper, branch, inj = state
        norm_weight, down, up, inject_w = weight_sets[k]
        mixed, h, inj2 = fused_gated_residual_elem_fold(
            hyper, branch, inj, norm_weight, RMS_EPS, HC, HIDDEN,
            down, up, inject_w)
        return (h, mixed, inj2)

    def chain(step, state, n_steps):
        """n_steps ぶんの連鎖を遅延グラフで組んでから 1 回だけ eval する。

        1 呼び出しごとに eval すると往復の同期 (150〜240 us) が支配して
        カーネルの差が埋もれる。実モデルの decode も 1 フォワード分の
        グラフを 1 回 eval する形なので、こちらの方が形も近い。
        """
        t0 = time.perf_counter()
        for i in range(n_steps):
            state = step(state, i % n_sets)
        mx.eval(state)
        return state, (time.perf_counter() - t0) / n_steps * 1e6

    # 焼き入れ (カーネルのコンパイルと最初の巡回)
    st_a, _ = chain(step_a, init(), n_sets)
    st_b, _ = chain(step_b, init(), n_sets)

    sa, sb = [], []
    for _ in range(N_PAIRS):
        st_a, us = chain(step_a, st_a, n_sets)
        sa.append(us)
        st_b, us = chain(step_b, st_b, n_sets)
        sb.append(us)
        st_b, us = chain(step_b, st_b, n_sets)
        sb.append(us)
        st_a, us = chain(step_a, st_a, n_sets)
        sa.append(us)

    a = _summ(sa, WARMUP)
    b = _summ(sb, WARMUP)
    return {
        "m": m,
        "weight_sets": n_sets,
        "weight_mb": round(per_set_bytes * n_sets / 1e6, 1),
        "plain_combine_plus_elem": a,
        "fold": b,
        "ratio_fold_over_plain": round(b["median_us"] / a["median_us"], 4),
        "delta_us_per_call": round(b["median_us"] - a["median_us"], 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/hc-write-fold-micro.json")
    ap.add_argument("--weight-sets", type=int, default=97)
    ap.add_argument("--rows", default="1,3")
    args = ap.parse_args()

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    res = {"note": "A = _combine (mx.compile) + elem、B = elem_fold。"
                   "重み 97 組を巡回させた冷の連鎖、ABBA",
           "cases": []}
    for m in [int(v) for v in args.rows.split(",") if v.strip()]:
        r = run(args.weight_sets, m)
        res["cases"].append(r)
        print(f"[m={r['m']} 重み {r['weight_mb']} MB] "
              f"A {r['plain_combine_plus_elem']['median_us']:.1f} / "
              f"B {r['fold']['median_us']:.1f} us  "
              f"(fold/plain {r['ratio_fold_over_plain']:.3f}, "
              f"{r['delta_us_per_call']:+.1f} us/呼び出し)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"-> {out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
