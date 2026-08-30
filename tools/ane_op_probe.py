"""Control probe: mix in an op the ANE (probably) cannot run and watch CPU_AND_NE
collapse back to CPU_ONLY speed. Validates the routing fingerprint used in the
gate: CPU_AND_NE may use {CPU, NE} only, so CPU_AND_NE > CPU_ONLY implies NE.
If some op variant makes CPU_AND_NE ≈ CPU_ONLY, the gap really comes from NE
running the matmul chain (and the op forces the whole graph off it).

Not a distribution-quality benchmark; a routing control. Interleaved medians,
same-process, like the gate.
"""

from __future__ import annotations

import json
import statistics
import time

import numpy as np

import coremltools as ct
from coremltools.models import MLModel
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

SHAPE = (4096, 2560)
WEIGHT = (2560, 2560)
CHAIN = 6
VARIANTS = {
    "matmul_only": None,
    "plus_cumsum": "cumsum",
    "plus_topk": "topk",
    "plus_gather": "gather",
}
ROUNDS = 20
WARMUP = 8


def build(name: str, op: str | None, path: str) -> None:
    specs = [mb.TensorSpec(shape=SHAPE, dtype=types.fp32)]
    rng = np.random.default_rng(42)

    @mb.program(input_specs=specs)
    def prog(x):
        y = mb.cast(x=x, dtype="fp16")
        for i in range(CHAIN):
            w = (rng.standard_normal(WEIGHT) * 0.05).astype(np.float16)
            y = mb.matmul(x=y, y=mb.const(val=w, name=f"w{i:d}"))
        if op == "cumsum":
            y = mb.cumsum(x=y, axis=1)
        elif op == "topk":
            vals, _idx = mb.topk(x=y, k=4, axis=1)  # -> (4096, 4)
            y = vals
        elif op == "gather":
            idx = np.arange(0, 2560, 4, dtype=np.int32)
            y = mb.gather(x=y, indices=mb.const(val=idx), axis=1)  # -> (4096,640)
        return mb.cast(x=y, dtype="fp32")

    model = ct.convert(prog, convert_to="mlprogram",
                       minimum_deployment_target=ct.target.macOS15)
    model.save(path)


def probe() -> dict:
    results = {}
    for name, op in VARIANTS.items():
        path = f"/tmp/ane_probe_{name}.mlpackage"
        build(name, op, path)
        models = {
            "CPU_ONLY": MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY),
            "CPU_AND_NE": MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE),
            "CPU_AND_GPU": MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_GPU),
        }
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(SHAPE) * 0.1).astype(np.float32)
        lat = {k: [] for k in models}
        for k, m in models.items():
            for _ in range(WARMUP):
                m.predict({"x": x})
        for _ in range(ROUNDS):
            for k, m in models.items():
                t0 = time.perf_counter_ns()
                m.predict({"x": x})
                lat[k].append((time.perf_counter_ns() - t0) / 1e6)
        results[name] = {
            k: round(statistics.median(v), 3) for k, v in lat.items()
        }
        results[name]["NE_over_CPU"] = round(
            results[name]["CPU_ONLY"] / results[name]["CPU_AND_NE"], 3)
        print(f"{name}: {results[name]}", flush=True)
    return results


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2, sort_keys=True))
