"""`hc_elem_post` の c0 (combine=True) と p1 (combine=False) を同じ形で直接比べる。

背景: 2026-09-03 21:10 の decode 帰属 (`bench/results/decode-anatomy-splitcb-*.json`)
で、最終 mixer の `hc_elem_post_..._p1` が **1 呼び出し 462 us**、同じ形・同じ
grid の `_c0` が 6.9 us と 67 倍の差に見えた。p1 のソースは c0 から inject の
分岐 (`combine=False` なので出ない) を抜いただけの真部分集合なので、実カーネル
がここまで遅い理由が無い。

疑い 2 つ:
  (1) 実在する (combine=False 分岐の何かが遅い) -> ソースを直す
  (2) 帰属のアーチファクト。`tools/bridge/metal_probe.mm` の `record_cb` は
      **command buffer の GPU 時間をその CB 内のディスパッチ数で等分**する
      (`share = ms / names->size()`)。呼び出しが 1 本しか入っていない CB に
      乗ると、その CB 全体の時間を 1 本で背負う。

判定: この micro で c0 と p1 の us/call が同水準なら (2)。
モデルは読まない。数十秒で終わる。

    tools/biglock.sh uv run python tools/hc_elem_post_micro.py \
        --out bench/results/hc-elem-post-micro.json
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

# 実寸 (mlxturbo/_vendor/qwen4_exp.py の TextArgs の既定値)
HIDDEN = 2560
HC = 4
HC_LOWRANK = 320
RMS_EPS = 1e-6
HC_QBITS = 4
HC_QGROUP = 64

N_SETS = 8          # 入力を巡回させる組数 (同じ配列の使い回しを避ける)
N_CALLS = 400       # スループット計測の呼び出し数
N_LAT = 120         # レイテンシ (1 呼び出しごと eval) の試行数
WARMUP = 20


def _quant_linear(out_features, in_features, dtype, group_size, bits):
    import mlx.core as mx

    w = (mx.random.normal((out_features, in_features)) * 0.02).astype(dtype)
    wq, sc, bi = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(wq, sc, bi)
    return (wq, sc, bi, group_size, bits)


def _post_inputs(m, dtype, n_sets):
    """post カーネルの入力 (normed, up_raw, inject_raw) を n_sets 組。"""
    import mlx.core as mx

    sets = []
    for _ in range(n_sets):
        normed = (mx.random.normal((m, HC * HIDDEN)) * 0.5).astype(dtype)
        up_raw = (mx.random.normal((m, HC * HIDDEN)) * 0.5).astype(dtype)
        inject_raw = (mx.random.normal((m, HC)) * 0.5).astype(dtype)
        mx.eval(normed, up_raw, inject_raw)
        sets.append((normed, up_raw, inject_raw))
    return sets


def _call_post(post, cfg, inputs, m, dtype):
    import mlx.core as mx
    from mlxturbo.kernels.hyper_connection import _ELEM_THREADS

    normed, up_raw, inject_raw = inputs
    post_x = ((HIDDEN + _ELEM_THREADS - 1) // _ELEM_THREADS) * _ELEM_THREADS
    ins = [normed, up_raw]
    shapes = [(m, HIDDEN)]
    dts = [dtype]
    if cfg["combine"]:
        ins.append(inject_raw)
        shapes.append((m, HC))
        dts.append(dtype)
    return post(
        inputs=ins,
        template=[("T", dtype)],
        grid=(post_x, 1, m),
        threadgroup=(_ELEM_THREADS, 1, 1),
        output_shapes=shapes,
        output_dtypes=dts,
    )


def _throughput(post, cfg, sets, m, dtype, n_calls):
    """独立な n 本をまとめて 1 回 eval し、1 本あたりの us。"""
    import mlx.core as mx

    outs = []
    t0 = time.perf_counter()
    for i in range(n_calls):
        outs.append(_call_post(post, cfg, sets[i % len(sets)], m, dtype))
    mx.eval(outs)
    return (time.perf_counter() - t0) / n_calls * 1e6


def _latency(post, cfg, sets, m, dtype, n):
    """1 本ずつ eval した us (起動 + 同期のレイテンシ)。"""
    import mlx.core as mx

    samples = []
    for i in range(n):
        t0 = time.perf_counter()
        mx.eval(_call_post(post, cfg, sets[i % len(sets)], m, dtype))
        samples.append((time.perf_counter() - t0) * 1e6)
    return samples


def _summ(samples, warmup):
    kept = samples[warmup:]
    return {
        "n": len(samples),
        "kept": len(kept),
        "median_us": round(statistics.median(kept), 3),
        "min_us": round(min(kept), 3),
        "max_us": round(max(kept), 3),
    }


def run_post(m: int):
    """post カーネルだけを c0 / p1 で交互 (ABBA) に測る。"""
    import mlx.core as mx

    from mlxturbo.kernels import hyper_connection as hck

    dtype = mx.bfloat16
    cfg_c = {"hc": HC, "d": HIDDEN, "lowrank": HC_LOWRANK,
             "eps": RMS_EPS, "combine": True}
    cfg_p = dict(cfg_c, combine=False)
    _, _, post_c = hck._get_kernels_elem(cfg_c)
    _, _, post_p = hck._get_kernels_elem(cfg_p)
    sets = _post_inputs(m, dtype, N_SETS)

    # 焼き入れ (コンパイル)
    for cfg, post in ((cfg_c, post_c), (cfg_p, post_p)):
        mx.eval(_call_post(post, cfg, sets[0], m, dtype))

    tp_c, tp_p = [], []
    for _ in range(3):
        tp_c.append(_throughput(post_c, cfg_c, sets, m, dtype, N_CALLS))
        tp_p.append(_throughput(post_p, cfg_p, sets, m, dtype, N_CALLS))
        tp_p.append(_throughput(post_p, cfg_p, sets, m, dtype, N_CALLS))
        tp_c.append(_throughput(post_c, cfg_c, sets, m, dtype, N_CALLS))

    lat_c = _latency(post_c, cfg_c, sets, m, dtype, N_LAT)
    lat_p = _latency(post_p, cfg_p, sets, m, dtype, N_LAT)
    lat_p += _latency(post_p, cfg_p, sets, m, dtype, N_LAT)
    lat_c += _latency(post_c, cfg_c, sets, m, dtype, N_LAT)

    med_c = statistics.median(tp_c)
    med_p = statistics.median(tp_p)
    return {
        "m": m,
        "throughput_us_per_call": {
            "c0_combine": round(med_c, 3),
            "p1_plain": round(med_p, 3),
            "ratio_p1_over_c0": round(med_p / med_c, 4),
            "c0_samples": [round(v, 3) for v in tp_c],
            "p1_samples": [round(v, 3) for v in tp_p],
        },
        "latency_us_per_call": {
            "c0_combine": _summ(lat_c, WARMUP),
            "p1_plain": _summ(lat_p, WARMUP),
        },
    }


def run_full(m: int):
    """`fused_gated_residual_elem` を丸ごと (combine あり/なし) 連鎖で測る。

    実モデルと同じ呼び方。combine なしは出力が `mixed` (m, d) だけなので、
    次の hyper は `mx.tile` で作る (両側とも同じ後処理は付かないので、
    combine あり側にも同じ tile を通して条件を揃える)。
    """
    import mlx.core as mx

    from mlxturbo.kernels.hyper_connection import fused_gated_residual_elem

    dtype = mx.bfloat16
    norm_weight = (mx.random.normal((HC * HIDDEN,)) * 0.02).astype(dtype)
    down = _quant_linear(HC_LOWRANK, HC * HIDDEN, dtype, HC_QGROUP, HC_QBITS)
    up = _quant_linear(HC * HIDDEN, HC_LOWRANK, dtype, HC_QGROUP, HC_QBITS)
    inject = ((mx.random.normal((HC, HC * HIDDEN)) * 0.02).astype(dtype),)
    hyper0 = (mx.random.normal((m, HC * HIDDEN)) * 0.5).astype(dtype)
    mx.eval(norm_weight, hyper0)

    def step(hyper, combine):
        out = fused_gated_residual_elem(
            hyper, norm_weight, RMS_EPS, HC, HIDDEN, down, up,
            inject if combine else None,
        )
        mixed = out[0] if combine else out
        return mixed

    def bench(combine, n):
        hyper = hyper0
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            mixed = step(hyper, combine)
            mx.eval(mixed)
            samples.append((time.perf_counter() - t0) * 1e6)
            hyper = mx.tile(mixed, (1, HC))
            mx.eval(hyper)   # 次入力は計測窓の外で確定させる
        return samples

    bench(True, 10)
    bench(False, 10)
    a, b = [], []
    for _ in range(2):
        a += bench(True, 60)
        b += bench(False, 60)
        b += bench(False, 60)
        a += bench(True, 60)
    return {
        "m": m,
        "combine_true": _summ(a, WARMUP),
        "combine_false": _summ(b, WARMUP),
        "ratio_false_over_true": round(
            statistics.median(b[WARMUP:]) / statistics.median(a[WARMUP:]), 4),
    }


def run_probe(m: int):
    """帰属のアーチファクトそのものを再現する。

    `tools/decode_module_attrib.py --split-cb` は `MLX_MAX_OPS_PER_BUFFER=1` で
    走らせ、`tools/bridge/metal_probe.mm` の `record_cb` が **command buffer の
    GPUStartTime→GPUEndTime をその CB のディスパッチ数で等分**する
    (`share = ms / names->size()`)。1 op = 1 CB なので、報告される「カーネルの
    us/call」は **その CB の GPU 区間まるごと**になる。

    ここでは post カーネル (4 us) の直前に lm_head 相当の大きい量子化行列積
    (248320 x 2560、4bit = 358 MB) を挟んで同じ形を作り、post に何 us が
    付くかを見る。実モデルの最終 mixer の post も lm_head の直前に居る。
    """
    import mlx.core as mx

    from decode_gpu_trace import Probe

    from mlxturbo.kernels import hyper_connection as hck

    dtype = mx.bfloat16
    cfg_p = {"hc": HC, "d": HIDDEN, "lowrank": HC_LOWRANK,
             "eps": RMS_EPS, "combine": False}
    _, _, post_p = hck._get_kernels_elem(cfg_p)
    sets = _post_inputs(m, dtype, 2)

    # lm_head 相当 (行数を絞って 4bit にしたもの)。post のあとに置く
    head = _quant_linear(24832, HIDDEN, dtype, HC_QGROUP, HC_QBITS)
    mx.eval(_call_post(post_p, cfg_p, sets[0], m, dtype))

    probe = Probe()
    if not probe.available and not probe.install():
        return {"m": m, "error": probe.error or "probe を入れられない"}
    def _sweep(burst: bool, n: int = 40):
        probe.enable(True)
        probe.reset()
        outs = []
        for i in range(n):
            (mixed,) = _call_post(post_p, cfg_p, sets[i % len(sets)], m, dtype)
            logits = mx.quantized_matmul(
                mixed, head[0], scales=head[1], biases=head[2],
                transpose=True, group_size=head[3], bits=head[4])
            if burst:
                outs.append(logits)
            else:
                mx.eval(logits)
        if burst:
            mx.eval(outs)
        probe.quiesce(2000)
        st = probe.stats()
        probe.enable(False)
        rows = []
        for k in st["kernels"]:
            if k["count"] == 0:
                continue
            rows.append({"name": k["name"][:70], "count": k["count"],
                         "us_per_call": round(k["gpu_ms"] * 1000 / k["count"], 1)})
        rows.sort(key=lambda r: -r["us_per_call"])
        return {"calls": n, "dispatches": st["dispatches"],
                "command_buffers": st["command_buffers"],
                "disp_per_cb": round(st["dispatches"]
                                     / max(st["command_buffers"], 1), 2),
                "kernels": rows[:8]}

    # per_eval = 1 呼び出しごとに eval (GPU が host を待って CB の区間に
    # 泡が入る) / burst = 40 本ぶんを遅延グラフで組んで 1 回 eval (実モデルの
    # フォワードと同じ形)。**同じカーネルの us/call がどれだけ動くか**が
    # 帰属の歪みの大きさ
    return {"m": m, "per_eval": _sweep(False), "burst": _sweep(True)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/hc-elem-post-micro.json")
    ap.add_argument("--rows", default="1,3")
    ap.add_argument("--probe", action="store_true",
                    help="metal_probe を入れて帰属のアーチファクトを再現する "
                         "(MLX_MAX_OPS_PER_BUFFER=1 と併せて使う)")
    args = ap.parse_args()

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    res = {"note": "c0 (combine=True) vs p1 (combine=False) の hc_elem_post。"
                   "帰属 (decode-anatomy-splitcb) では p1 が 462 us/call、c0 が 6.9 us/call",
           "post": [], "full": []}
    if args.probe:
        res["probe"] = [run_probe(m) for m in
                        [int(v) for v in args.rows.split(",") if v.strip()]]
        for r in res["probe"]:
            for mode in ("per_eval", "burst"):
                s = r.get(mode)
                if not s:
                    continue
                print(f"[probe m={r.get('m')} {mode}] disp/cb {s['disp_per_cb']}"
                      f" ({s['dispatches']} disp / {s['command_buffers']} CB)")
                for k in s["kernels"]:
                    nm = k["name"].replace("\n", " ")[:60]
                    print(f"    {nm:<62} cnt {k['count']:5d} "
                          f"{k['us_per_call']:9.1f} us/call")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"-> {args.out}")
        sys.stdout.flush()
        os._exit(0)
    for m in [int(v) for v in args.rows.split(",") if v.strip()]:
        res["post"].append(run_post(m))
        res["full"].append(run_full(m))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for r in res["post"]:
        t = r["throughput_us_per_call"]
        l = r["latency_us_per_call"]
        print(f"[post m={r['m']}] throughput c0 {t['c0_combine']:.2f} / "
              f"p1 {t['p1_plain']:.2f} us  (p1/c0 {t['ratio_p1_over_c0']:.3f})")
        print(f"[post m={r['m']}] latency    c0 {l['c0_combine']['median_us']:.2f} / "
              f"p1 {l['p1_plain']['median_us']:.2f} us")
    for r in res["full"]:
        print(f"[full m={r['m']}] combine=True {r['combine_true']['median_us']:.2f} / "
              f"False {r['combine_false']['median_us']:.2f} us "
              f"(false/true {r['ratio_false_over_true']:.3f})")
    print(f"-> {out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
