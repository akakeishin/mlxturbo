"""27B (qwen3_next) の decode/verify 幅 sdpa を S 軸で分割する案の冷 micro。

的 (`scratchpad/agent-ceiling-audit.md` の [t13]/[t14]): head_dim 256 の
`mx.fast.scaled_dot_product_attention` は q 行数 `S * gqa_factor` が 32 を
越えると速い vector カーネルから落ち、全 KV にスコアを実体化する経路に
なる。27B は gqa 6 なので崖は S >= 6 (Flash-Next は gqa 12 で S >= 3、
そちらは `_vendor/qwen4_exp.py` のシームが既に分割している)。

3 つの呼び方を同じ形・同じ冷え方で比べる:

  plain : 1 回の sdpa (mask="causal")                     ... 今の 27B
  trim  : w 行ずつに割り、K/V を前方 `[0, offset+t1)` に切って mask="causal"
  mask  : w 行ずつに割り、K/V は丸ごと渡して bool causal マスクの行を切る
          (`_vendor/qwen4_exp.py` の既存シームと同じ組み方)

`w = max(1, 32 // gqa)`。S * gqa <= 32 の幅では分割は起きない (plain と
同じ本数になる) ので、崖の手前 (S=2..4) は「余計なことをしていない」ことの
確認になる。

作法 (CLAUDE.md): K/V を 140 MB ぶん巡回させて冷やす、burn-in、独立本を
まとめて 1 回 eval、build 時間は分離 (`tools/ceiling_audit_micro.py` と同じ)。

実行:

    tools/biglock.sh .venv/bin/python tools/sdpa_split_generic_micro.py \
        --json bench/results/sdpa-split-generic-0904.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import mlx.core as mx

DT = mx.bfloat16
PEAK_BW = 409.6e9  # M3 Max
PEAK_TF_BF16 = 12.85e12

# (タグ, Hq, Hk, head_dim)
CASES = [
    ("q27", 24, 4, 256),  # Qwen3.8-27B: 24 heads / 4 kv / head_dim 256 (gqa 6)
    ("fn", 24, 2, 256),   # Flash-Next 参考 1 点 (gqa 12、既存シームが担当)
]
S_LIST = [2, 3, 4, 6, 8]
KV_LIST = [2048, 4096, 17408]


def _stats(samples):
    return {
        "us": round(statistics.median(samples), 3),
        "us_min": round(min(samples), 3),
        "spread": round(
            (max(samples) - min(samples)) / max(statistics.median(samples), 1e-9), 3
        ),
    }


def bench(make, n_sets: int, iters: int, reps: int = 5, warm: int = 2):
    for _ in range(warm):
        mx.eval(make(0))
    ev = []
    it = 0
    for _ in range(reps):
        outs = []
        for _ in range(iters):
            outs.append(make(it % n_sets))
            it += 1
        t1 = time.perf_counter()
        mx.eval(*outs)
        t2 = time.perf_counter()
        ev.append((t2 - t1) / iters * 1e6)
        del outs
    return _stats(ev)


def measure(make, n_sets: int, out_bytes: int, reps: int = 5, hi: int = 128):
    p = bench(make, n_sets, 3, reps=2, warm=1)
    n = int(max(2, min(hi, round(25.0 * 1000.0 / max(p["us"], 0.5)))))
    n = max(2, min(n, 1_500_000_000 // max(out_bytes, 1)))
    st = bench(make, n_sets, n, reps=reps)
    st["iters"] = n
    return st


def _sets_for(mb_target: float, one_mb: float, cap: int = 64) -> int:
    return int(max(2, min(cap, math.ceil(mb_target / max(one_mb, 1e-6)))))


def _causal_mask(S: int, kv: int):
    """`(1, 1, S, kv)` の bool causal マスク (q は絶対位置 kv-S .. kv-1)。"""
    offset = kv - S
    return (mx.arange(kv)[None, :] <= (offset + mx.arange(S))[:, None])[None, None]


def call_plain(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")


def call_trim(q, k, v, scale, w):
    S = q.shape[2]
    offset = k.shape[2] - S
    outs = []
    t0 = 0
    while t0 < S:
        t1 = min(t0 + w, S)
        kv_end = offset + t1
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, t0:t1], k[:, :, :kv_end], v[:, :, :kv_end],
                scale=scale, mask="causal",
            )
        )
        t0 = t1
    return mx.concatenate(outs, axis=2) if len(outs) > 1 else outs[0]


def call_mask(q, k, v, scale, w, m):
    S = q.shape[2]
    outs = []
    t0 = 0
    while t0 < S:
        t1 = min(t0 + w, S)
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, t0:t1], k, v, scale=scale, mask=m[..., t0:t1, :],
            )
        )
        t0 = t1
    return mx.concatenate(outs, axis=2) if len(outs) > 1 else outs[0]


def ref_fp32(q, k, v, scale, Hq, Hk):
    """fp32 の参照 attention (MLX の sdpa を使わない素の並び)。

    `mx.fast.scaled_dot_product_attention` を fp32 で呼ぶと、壁の手前と
    向こうで**参照自身が別のカーネルになる**ので比較の基準にならない。ここは
    matmul -> causal マスク -> softmax -> matmul を float32 で自分で組む。
    """
    S = q.shape[2]
    kv = k.shape[2]
    offset = kv - S
    r = Hq // Hk
    q32 = q.astype(mx.float32)
    k32 = mx.repeat(k.astype(mx.float32), r, axis=1)
    v32 = mx.repeat(v.astype(mx.float32), r, axis=1)
    scores = (q32 @ k32.swapaxes(-1, -2)) * scale
    keep = (mx.arange(kv)[None, :] <= (offset + mx.arange(S))[:, None])[None, None]
    scores = mx.where(keep, scores, mx.array(-3.4e38, dtype=mx.float32))
    return mx.softmax(scores, axis=-1) @ v32


def accuracy(Hq, Hk, D, S, kv, w, draws: int = 3):
    """fp32 参照からの距離: 素の bf16 sdpa 対 分割した bf16 sdpa。

    採否の条件 (2026-09-04 の方針): **分割の距離が素の距離以下**であること。
    ビット一致は求めない (壁の手前と向こうで MLX が別のカーネルを選ぶので、
    そもそも一致しえない)。
    """
    scale = D ** -0.5
    out = {"plain_max": 0.0, "split_max": 0.0,
           "plain_rms": 0.0, "split_rms": 0.0, "worse": 0}
    for d in range(draws):
        mx.random.seed(1000 + d)
        q = mx.random.normal((1, Hq, S, D)).astype(DT)
        k = mx.random.normal((1, Hk, kv, D)).astype(DT)
        v = mx.random.normal((1, Hk, kv, D)).astype(DT)
        ref = ref_fp32(q, k, v, scale, Hq, Hk)
        a = call_plain(q, k, v, scale).astype(mx.float32)
        b = call_trim(q, k, v, scale, w).astype(mx.float32)
        mx.eval(ref, a, b)
        pm = float(mx.max(mx.abs(a - ref)))
        sm = float(mx.max(mx.abs(b - ref)))
        pr = float(mx.sqrt(mx.mean((a - ref) ** 2)))
        sr = float(mx.sqrt(mx.mean((b - ref) ** 2)))
        out["plain_max"] = max(out["plain_max"], pm)
        out["split_max"] = max(out["split_max"], sm)
        out["plain_rms"] += pr / draws
        out["split_rms"] += sr / draws
        if sr > pr:
            out["worse"] += 1
        del q, k, v, ref, a, b
    out["rms_ratio"] = out["split_rms"] / max(out["plain_rms"], 1e-30)
    return out


def bitident(Hq, Hk, D, S, kv, w):
    """plain / trim / mask の 3 つを 1 組の入力で突き合わせる。"""
    q = mx.random.normal((1, Hq, S, D)).astype(DT)
    k = mx.random.normal((1, Hk, kv, D)).astype(DT)
    v = mx.random.normal((1, Hk, kv, D)).astype(DT)
    m = _causal_mask(S, kv)
    scale = D ** -0.5
    a = call_plain(q, k, v, scale)
    b = call_trim(q, k, v, scale, w)
    c = call_mask(q, k, v, scale, w, m)
    mx.eval(a, b, c)
    out = {}
    for name, x in (("trim", b), ("mask", c)):
        d = mx.abs(a.astype(mx.float32) - x.astype(mx.float32))
        out[name] = {
            "equal": bool(mx.array_equal(a, x)),
            "max_abs": float(mx.max(d)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=140.0, help="K/V の巡回総量 (MB)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json", default="")
    ap.add_argument("--cases", default="q27,fn")
    ap.add_argument("--accuracy", action="store_true",
                    help="fp32 参照からの距離だけを測る (時間の計測はしない)")
    ap.add_argument("--acc-s", default="4,6,7,8,9",
                    help="--accuracy で見る S (4 は発火しない対照)")
    ap.add_argument("--acc-kv", default="4096,17408")
    ap.add_argument("--draws", type=int, default=3)
    args = ap.parse_args()

    want = {c.strip() for c in args.cases.split(",") if c.strip()}
    rows = []

    # burn-in: プロセス起動直後の最初の計測は +7〜9% 遅い (CLAUDE.md)。
    _q = mx.random.normal((1, 24, 4, 256)).astype(DT)
    _k = mx.random.normal((1, 4, 4096, 256)).astype(DT)
    mx.eval(_q, _k)
    for _ in range(3):
        mx.eval(call_plain(_q, _k, _k, 256 ** -0.5))
    del _q, _k

    if args.accuracy:
        acc_s = [int(s) for s in args.acc_s.split(",") if s.strip()]
        acc_kv = [int(s) for s in args.acc_kv.split(",") if s.strip()]
        for tag, Hq, Hk, D in CASES:
            if tag not in want:
                continue
            gqa = Hq // Hk
            w = max(1, 32 // gqa)
            print(f"[{tag}] Hq={Hq} Hk={Hk} gqa={gqa} d={D} w={w} "
                  f"(draws={args.draws})", flush=True)
            print("    kv     S  発火  素 max / RMS         分割 max / RMS       "
                  "RMS 比  分割が悪い draw", flush=True)
            for kv in acc_kv:
                for S in acc_s:
                    a = accuracy(Hq, Hk, D, S, kv, w, draws=args.draws)
                    fires = S * gqa > 32
                    print(f"  {kv:6d} {S:3d}  {'yes' if fires else ' no':4}  "
                          f"{a['plain_max']:.3e} / {a['plain_rms']:.3e}  "
                          f"{a['split_max']:.3e} / {a['split_rms']:.3e}  "
                          f"{a['rms_ratio']:6.3f}  {a['worse']}/{args.draws}",
                          flush=True)
                    rows.append({"op": "accuracy", "shape": tag, "kv": kv, "S": S,
                                 "w": w, "fires": fires, **a})
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1)
            print(f"wrote {args.json} ({len(rows)} rows)")
        return

    for tag, Hq, Hk, D in CASES:
        if tag not in want:
            continue
        gqa = Hq // Hk
        w = max(1, 32 // gqa)
        print(f"[{tag}] Hq={Hq} Hk={Hk} gqa={gqa} head_dim={D} -> w={w}", flush=True)

        print("  ビット一致 (kv=4096):", flush=True)
        for S in S_LIST:
            bi = bitident(Hq, Hk, D, S, 4096, w)
            n_chunk = math.ceil(S / w)
            print(f"    S={S} (chunks={n_chunk})  "
                  f"trim equal={bi['trim']['equal']} max|d|={bi['trim']['max_abs']:.3e}  "
                  f"mask equal={bi['mask']['equal']} max|d|={bi['mask']['max_abs']:.3e}",
                  flush=True)
            rows.append({"op": "bitident", "shape": tag, "S": S, "kv": 4096,
                         "w": w, "chunks": n_chunk, **{f"{k}_{kk}": vv
                                                       for k, d in bi.items()
                                                       for kk, vv in d.items()}})

        for kv in KV_LIST:
            one_mb = 2 * Hk * kv * D * 2 / 1e6
            n_sets = _sets_for(args.mb, one_mb)
            ks = [mx.random.normal((1, Hk, kv, D)).astype(DT) for _ in range(n_sets)]
            vs = [mx.random.normal((1, Hk, kv, D)).astype(DT) for _ in range(n_sets)]
            mx.eval(ks, vs)
            print(f"  kv={kv}  K/V {one_mb:.1f} MB x {n_sets} = "
                  f"{one_mb * n_sets:.0f} MB", flush=True)
            for S in S_LIST:
                q = mx.random.normal((1, Hq, S, D)).astype(DT)
                m = _causal_mask(S, kv)
                mx.eval(q, m)
                scale = D ** -0.5
                wr = int(Hq * S * D * 2)
                base = None
                for name in ("plain", "trim", "mask"):
                    if name == "plain":
                        def make(i, q=q):
                            return call_plain(q, ks[i], vs[i], scale)
                    elif name == "trim":
                        def make(i, q=q):
                            return call_trim(q, ks[i], vs[i], scale, w)
                    else:
                        def make(i, q=q, m=m):
                            return call_mask(q, ks[i], vs[i], scale, w, m)
                    st = measure(make, n_sets, wr, reps=args.reps)
                    rd = int(2 * Hk * kv * D * 2 + Hq * S * D * 2)
                    gbs = (rd + wr) / (st["us"] * 1e-6) / 1e9
                    flops = 2.0 * 2 * Hq * S * kv * D
                    tf = flops / (st["us"] * 1e-6) / 1e12
                    ach = max(gbs * 1e9 / PEAK_BW, tf * 1e12 / PEAK_TF_BF16) * 100
                    if base is None:
                        base = st["us"]
                    rows.append({"op": "sdpa_split", "shape": tag, "kv": kv, "S": S,
                                 "variant": name, "w": w,
                                 "chunks": math.ceil(S / w) if name != "plain" else 1,
                                 "GBs": round(gbs, 1), "TFLOPS": round(tf, 2),
                                 "achieved_pct": round(ach, 1),
                                 "rel_to_plain": round(st["us"] / base, 3), **st})
                    print(f"    S={S:<2} {name:6} {st['us']:9.2f} us  "
                          f"達成 {ach:5.1f}%  対 plain {st['us'] / base:5.3f}",
                          flush=True)
                del q, m
            del ks, vs

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"wrote {args.json} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
