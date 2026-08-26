#!/usr/bin/env python3
"""Emit standalone ``.metal`` files for every kernel under ISA study.

CPU only -- no Metal device is touched.  Writes to ``tools/isa/build/metal``.

    python3 tools/isa/gen_kernels.py [--k 5120] [--n 17408]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import variants  # noqa: E402
from mlx_signature import build_metal_file, qmm_metal_file  # noqa: E402
from snapshots.qmm_skinny_mma_a2 import build_source  # noqa: E402

# mlp_up of the benchmarked model (docs/GATE-RESULTS-A2.md).
DEFAULT_K = 5120
DEFAULT_N = 17408


def _cases(k: int, n: int) -> dict[str, str]:
    out: dict[str, str] = {}

    # Shipped kernel, exactly as fastmlx builds it.  bf16 is the benchmarked
    # dtype, so base_m8_bf16 is the kernel behind the 1.04x number.
    for m in (6, 8, 12, 16):
        out[f"base_m{m}_bf16"] = qmm_metal_file(
            name=f"base_m{m}_bf16",
            body=build_source(m, fp16_input=False),
            dtype="bfloat16_t",
            k=k,
            n=n,
        )
    out["base_m8_f16"] = qmm_metal_file(
        name="base_m8_f16",
        body=build_source(8, fp16_input=True),
        dtype="half",
        k=k,
        n=n,
    )

    # Single-mechanism probes.
    for vname, builder in variants.VARIANTS.items():
        for m in (8, 16):
            out[f"{vname}_m{m}_bf16"] = qmm_metal_file(
                name=f"{vname}_m{m}_bf16",
                body=builder(m),
                dtype="bfloat16_t",
                k=k,
                n=n,
            )

    out.update(_fastqmm_reference(k, n))
    out.update(_current_kernel())
    return out


def _current_kernel() -> dict[str, str]:
    """Whatever fastmlx.kernels currently builds, on the same pipeline.

    The A2 kernels above come from a snapshot so their numbers stay comparable;
    this entry tracks the working tree so a new design can be measured against
    them without editing this script.
    """

    try:
        from fastmlx.kernels import _qmm_skinny_mma_source as cur
    except Exception as exc:  # noqa: BLE001 - a broken working tree is not fatal
        print(f"current kernel skipped: {exc}")
        return {}
    try:
        body = cur.build_source()
    except TypeError:
        return {}
    header = getattr(cur, "METAL_HEADER", "")
    return {
        "current_qmm_skinny": build_metal_file(
            name="current_qmm_skinny",
            body=body,
            input_names=["w", "scales", "biases", "x"],
            input_types=["uint32_t", "bfloat16_t", "bfloat16_t", "bfloat16_t"],
            output_names=["y"],
            output_types=["bfloat16_t"],
            template_params=[],
            template_args=[],
            header=header,
        )
    }


_SRC_RE = re.compile(r"^(_SRC(?:_WIDE)?) = r\"\"\"(.*?)^\"\"\"", re.S | re.M)


def _fastqmm_reference(k: int, n: int) -> dict[str, str]:
    """The vendored fast_qmm bodies, wrapped in their own MLX signature.

    fast_qmm is the 1.57x reference in docs/GATE-RESULTS-A2.md, so it belongs
    on the same disassembly pipeline.  Its source is lifted textually rather
    than imported so this script never has to load mlx.
    """

    path = ROOT / "fastmlx" / "fast_qmm.py"
    if not path.exists():
        return {}
    bodies = dict(_SRC_RE.findall(path.read_text()))
    out: dict[str, str] = {}
    for key, name, m in (("_SRC", "ref_fastqmm_m8", 8), ("_SRC_WIDE", "ref_fastqmm_m16", 16)):
        body = bodies.get(key)
        if body is None:
            continue
        out[name] = build_metal_file(
            name=name,
            body=body,
            input_names=["x", "w", "sc", "bi"],
            input_types=["bfloat16_t", "uint32_t", "bfloat16_t", "bfloat16_t"],
            output_names=["out"],
            output_types=["bfloat16_t"],
            template_params=[("int", "KD"), ("int", "ND"), ("int", "MD")],
            template_args=[str(k), str(n), str(m)],
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--out", default=str(HERE / "build" / "metal"))
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = _cases(args.k, args.n)
    for name, text in sorted(cases.items()):
        path = outdir / f"{name}.metal"
        path.write_text(text)
        print(f"wrote {path} ({len(text)} bytes)")
    print(f"{len(cases)} kernels, K={args.k} N={args.n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
