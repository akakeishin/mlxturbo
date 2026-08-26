#!/usr/bin/env python3
"""Emit standalone ``.metal`` files for every kernel under ISA study.

CPU only -- no Metal device is touched.  Writes to ``tools/isa/build/metal``.

    python3 tools/isa/gen_kernels.py [--k 5120] [--n 17408]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from fastmlx.kernels._qmm_skinny_mma_source import build_source  # noqa: E402

import variants  # noqa: E402
from mlx_signature import qmm_metal_file  # noqa: E402

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
