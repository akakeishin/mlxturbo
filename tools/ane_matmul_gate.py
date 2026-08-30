"""Stage-1 ANE gate: fp16 matmul chain on CPU_AND_NE vs CPU_AND_GPU vs CPU_ONLY.

Follows docs/research/ANE-PREFILL-BRIEF.md. Declares criteria before measuring:
proceed to stage 2 only if CPU_AND_NE is >=1.2x faster than CPU_AND_GPU.
Measurement is interleaved in the same process (thermal drift guard).
Does not touch mlxturbo/ (research lane).

Shapes mimic Qwen3.8-Flash-Next prefill: (4096, 2560) x (2560, 2560), fp16,
CHAIN matmuls to raise the per-call compute so predict-call overhead doesn't
hide device differences. Random fp32-safe values to avoid denormal edge cases.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

import coremltools as ct
from coremltools.models import MLModel
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

SHAPE = (4096, 2560)          # (seq, hidden) — --seq overrides seq dim
WEIGHT = (2560, 2560)         # (hidden, hidden)
CHAIN = 6                     # stacked matmuls per predict
UNITS = ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE")
ROUNDS = 40                   # interleaved rounds
WARMUP = 15                   # per-unit warmup iterations


def build_model(path: str, seq: int = SHAPE[0]) -> None:
    """Build and save the chained fp16 matmul program.

    I/O is fp32 with explicit fp16 casts at the boundary (Core ML's own
    fallback for fp16 I/O); the internal chain runs fp16 matmul on every
    compute unit, so routing comparisons stay apples-to-apples.
    """
    shape = (seq, WEIGHT[1])
    specs = [mb.TensorSpec(shape=shape, dtype=types.fp32)]
    rng = np.random.default_rng(42)

    @mb.program(input_specs=specs)
    def prog(x):
        y = mb.cast(x=x, dtype="fp16")
        for i in range(CHAIN):
            w = (rng.standard_normal(WEIGHT) * 0.05).astype(np.float16)
            y = mb.matmul(x=y, y=mb.const(val=w, name=f"w{i:d}"))
        return mb.cast(x=y, dtype="fp32")

    model = ct.convert(prog, convert_to="mlprogram",
                       minimum_deployment_target=ct.target.macOS15)
    model.save(path)
    print(f"built {path}")


def load_all(path: str):
    return {
        name: MLModel(path, compute_units=getattr(ct.ComputeUnit, name))
        for name in UNITS
    }


def bench(model, x: np.ndarray, iters: int) -> list[float]:
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        model.predict({"x": x})
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return samples


def run_gate(path: str, seq: int = SHAPE[0]) -> dict:
    shape = (seq, WEIGHT[1])
    models = load_all(path)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(shape) * 0.1).astype(np.float32)

    # warmup
    for name in UNITS:
        bench(models[name], x, WARMUP)

    # interleaved measurement
    lat = {name: [] for name in UNITS}
    for _ in range(ROUNDS):
        for name in UNITS:
            t0 = time.perf_counter_ns()
            models[name].predict({"x": x})
            lat[name].append((time.perf_counter_ns() - t0) / 1e6)

    # numeric equivalence check (device numeric drift)
    key = list(models["CPU_ONLY"].predict({"x": x}).keys())[0]
    outs = {name: models[name].predict({"x": x})[key] for name in UNITS}
    eq = {}
    for a, b in (("CPU_AND_NE", "CPU_AND_GPU"), ("CPU_ONLY", "CPU_AND_GPU")):
        oa, ob = outs[a].astype(np.float64), outs[b].astype(np.float64)
        rng_span = float(np.max(ob) - np.min(ob)) + 1e-12
        eq[f"{a} vs {b}"] = {"max_abs": float(np.max(np.abs(oa - ob))),
                             "max_frac_of_range": float(np.max(np.abs(oa - ob)) / rng_span)}

    result = {"shape": shape, "chain": CHAIN, "rounds": ROUNDS, "eq": eq}
    med = {}
    for name in UNITS:
        m = statistics.median(lat[name])
        med[name] = m
        result[name] = {
            "median_ms": round(m, 4),
            "mean_ms": round(statistics.mean(lat[name]), 4),
            "p10_ms": round(sorted(lat[name])[len(lat[name]) // 10], 4),
            "min_ms": round(min(lat[name]), 4),
            "max_ms": round(max(lat[name]), 4),
        }
    gpu = med["CPU_AND_GPU"]
    ne = med["CPU_AND_NE"]
    cpu = med["CPU_ONLY"]
    result["ratio_NE_over_GPU"] = round(gpu / ne, 3)  # >1.2 = ANE wins
    result["ratio_NE_over_CPU"] = round(cpu / ne, 3)

    # paired per-round ratios: immune to slow drift between sweeps, exposes
    # whether GPU and NE systematically differ or just get hit by contention
    pairs = sorted(g / n for g, n in zip(lat["CPU_AND_GPU"], lat["CPU_AND_NE"]))
    q = pairs[len(pairs) // 4]
    result["paired_NE_over_GPU_median"] = round(statistics.median(pairs), 3)
    result["paired_NE_over_GPU_q1"] = round(q, 3)
    result["paired_NE_over_GPU_q3"] = round(pairs[3 * len(pairs) // 4], 3)
    result["verdict"] = ("PROCEED" if gpu / ne >= 1.2 else "STOP")
    return {**result, "_raw": lat}


def summarize_runs(runs: list[dict]) -> dict:
    """Aggregate repeat runs. Min-of-medians per unit cleans external load;
    pooled paired ratios show the device-vs-device signal with contention
    averaged out (paired rounds face the same background second-by-second)."""
    out = {"n_runs": len(runs)}
    for name in UNITS:
        meds = [r[name]["median_ms"] for r in runs]
        out[name] = {"min_of_medians_ms": round(min(meds), 4),
                     "meds_ms": [round(m, 3) for m in meds]}
    out["ratio_min_NE_over_GPU"] = round(
        out["CPU_AND_GPU"]["min_of_medians_ms"]
        / out["CPU_AND_NE"]["min_of_medians_ms"], 3)
    pooled = [g / n_ for r in runs for g, n_ in zip(
        r["_raw"]["CPU_AND_GPU"], r["_raw"]["CPU_AND_NE"])]
    pooled.sort()
    out["pooled_paired_median"] = round(statistics.median(pooled), 3)
    out["pooled_paired_q1"] = round(pooled[len(pooled) // 4], 3)
    out["pooled_paired_q3"] = round(pooled[3 * len(pooled) // 4], 3)
    p = out["pooled_paired_median"]
    out["final_verdict"] = "PROCEED" if p >= 1.2 else "STOP"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/tmp/ane_gate_matmul.mlpackage")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--seq", type=int, default=SHAPE[0],
                    help="sequence dim (default 4096; 32768 = cold-prefill shape)")
    ap.add_argument("--phase", help="run one unit only (for instrumentation)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat gate N times and aggregate robustly")
    args = ap.parse_args()

    if args.build:
        build_model(args.path, args.seq)
        return

    if args.phase is not None:
        model = MLModel(args.path, compute_units=getattr(ct.ComputeUnit, args.phase))
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((args.seq, WEIGHT[1])) * 0.1).astype(np.float32)
        bench(model, x, WARMUP)
        n, t_end = 0, time.time() + args.seconds
        while time.time() < t_end:
            model.predict({"x": x})
            n += 1
        print(f"phase={args.phase} iters={n}")
        return

    n = args.repeat
    runs = []
    for _ in range(n):
        r = run_gate(args.path, args.seq)
        clean = {k: v for k, v in r.items() if k != "_raw"}
        runs.append(r)
        print(json.dumps(clean, indent=2, sort_keys=True))
    if n > 1:
        print(json.dumps(summarize_runs(runs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
