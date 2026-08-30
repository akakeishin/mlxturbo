"""Stage-2 probe: can 4-bit affine weights (group_size=64, like Flash-Next) be
used in the ANE after dequantization to fp16?

Answered with the same chained gate harness: quantize a random fp32 weight,
dequantize with numpy the same way MLX affine does, embed as fp16 const, and
compare ANE output-error and timing against the fp32-direct reference.

Also reports per-tensor bytes of the (2560,2560) linear and extrapolates the
fp16 memory inflation of embedding MoE expert weights statically.
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

SEQ = (1024, 2560)
WEIGHT = (2560, 2560)
CHAIN = 3
GROUP = 64
BITS = 4
ROUNDS = 25
WARMUP = 10


def quantize_affine(w_f32: np.ndarray, group: int, bits: int):
    """Same packing MLX uses for affine: per-group scale + bias, uint32 pack."""
    out, in_dim = w_f32.shape
    gsz = w_f32.reshape(out, -1, group)          # (out, in/group, group)
    mn, mx = gsz.min(-1), gsz.max(-1)
    scale = (mx - mn) / ((1 << bits) - 1) + 1e-12
    q = np.round((gsz - mn[..., None]) / scale[..., None]).astype(np.uint32)
    n_vals = 32 // bits
    q_flat = q.reshape(out, -1)                  # (out, in_dim)
    q_grp = q_flat.reshape(out, -1, n_vals)      # (out, in_dim/8, 8)
    acc = np.zeros((out, q_grp.shape[1]), dtype=np.uint32)
    for i in range(n_vals):
        acc |= (q_grp[:, :, i] << (i * bits))
    return acc, mn.astype(np.float32), scale.astype(np.float32)


def dequantize_affine(packed, mn, scale, group, bits, shape):
    out, in_dim = shape
    n_vals = 32 // bits
    flat = packed.astype(np.uint32)
    idx = np.arange(n_vals, dtype=np.uint32)
    q = ((flat[:, :, None] >> (idx * bits)) & 0xF).astype(np.float32)
    q = q.reshape(out, -1, group)                # (out, in/group, group)
    w = q * scale[:, :, None].astype(np.float32) + mn[:, :, None].astype(np.float32)
    return w.reshape(out, -1)[:, :in_dim]


def build(w_np: np.ndarray, path: str) -> None:
    specs = [mb.TensorSpec(shape=SEQ, dtype=types.fp32)]

    @mb.program(input_specs=specs)
    def prog(x):
        y = mb.cast(x=x, dtype="fp16")
        for _ in range(CHAIN):
            y = mb.matmul(x=y, y=mb.const(val=w_np))
        return mb.cast(x=y, dtype="fp32")

    model = ct.convert(prog, convert_to="mlprogram",
                       minimum_deployment_target=ct.target.macOS15)
    model.save(path)


def main() -> None:
    rng = np.random.default_rng(42)
    w_f32 = (rng.standard_normal(WEIGHT) * 0.05).astype(np.float32)
    packed, mn, scale = quantize_affine(w_f32.copy(), GROUP, BITS)
    w_deq = dequantize_affine(packed, mn, scale, GROUP, BITS, WEIGHT)

    # check numpy dequant roundtrip error before embedding
    print("dequant max_abs:", float(np.max(np.abs(w_f32 - w_deq))))

    w_fp16 = w_deq.astype(np.float16)
    w_ref = w_f32.astype(np.float16)
    p_deq, p_ref = "/tmp/ane_stage2_deq.mlpackage", "/tmp/ane_stage2_ref.mlpackage"
    build(w_fp16, p_deq)
    build(w_ref, p_ref)

    m_deq = MLModel(p_deq, compute_units=ct.ComputeUnit.CPU_AND_NE)
    m_ref = MLModel(p_ref, compute_units=ct.ComputeUnit.CPU_AND_NE)
    x = (rng.standard_normal(SEQ) * 0.1).astype(np.float32)
    key = list(m_ref.predict({"x": x}).keys())[0]
    o_deq = m_deq.predict({"x": x})[key].astype(np.float64)
    o_ref = m_ref.predict({"x": x})[key].astype(np.float64)
    span = float(np.max(o_ref) - np.min(o_ref))
    eq = {"max_abs": float(np.max(np.abs(o_deq - o_ref))),
          "max_frac_of_range": float(np.max(np.abs(o_deq - o_ref)) / span)}

    lat = {"deq": [], "ref": []}
    for m, k in ((m_deq, "deq"), (m_ref, "ref")):
        for _ in range(WARMUP):
            m.predict({"x": x})
    for _ in range(ROUNDS):
        for m, k in ((m_deq, "deq"), (m_ref, "ref")):
            t0 = time.perf_counter_ns()
            m.predict({"x": x})
            lat[k].append((time.perf_counter_ns() - t0) / 1e6)

    w_bytes = WEIGHT[0] * WEIGHT[1] * 2  # fp16
    print(json.dumps({
        "eq_deq_vs_ref": eq,
        "deq_ms": round(statistics.median(lat["deq"]), 3),
        "ref_ms": round(statistics.median(lat["ref"]), 3),
        "linear_fp16_bytes": w_bytes,
        "fp16_MB_per_expert_linear_2560x2560": round(w_bytes / 1e6, 2),
        "note_512_experts_per_layer_MB": round(512 * w_bytes / 1e6, 1),
        "dequant_ok": bool(eq["max_frac_of_range"] < 1e-1),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
