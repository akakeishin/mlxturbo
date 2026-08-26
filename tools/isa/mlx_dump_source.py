#!/usr/bin/env python3
"""GPU-QUEUE STEP.  Print the Metal source MLX actually generates.

``mx.fast.metal_kernel`` builds the kernel signature itself and only does so at
dispatch time, so this needs a Metal device.  ``tools/isa/mlx_signature.py``
reproduces that signature offline; run this once and diff the two to confirm
the offline pipeline is compiling the same text the driver sees.

    python3 tools/isa/mlx_dump_source.py > tools/isa/build/mlx-generated.txt

MLX prints the generated source to stdout when a kernel is called with
``verbose=True``, then runs it, so the arrays here are small on purpose.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import mlx.core as mx  # noqa: E402

from snapshots.qmm_skinny_mma_a2 import (  # noqa: E402
    BITS,
    GROUP_SIZE,
    SPLIT_K,
    build_source,
)

M = 8
K = 512
N = 64


def dump_a2_mma() -> None:
    kernel = mx.fast.metal_kernel(
        name="fastmlx_qmm_skinny_mma_m8_bf16",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        source=build_source(M, fp16_input=False),
        header="#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n",
    )
    x = mx.zeros((M, K), dtype=mx.bfloat16)
    w = mx.zeros((N, K * BITS // 32), dtype=mx.uint32)
    scales = mx.zeros((N, K // GROUP_SIZE), dtype=mx.bfloat16)
    biases = mx.zeros((N, K // GROUP_SIZE), dtype=mx.bfloat16)
    (y,) = kernel(
        inputs=[x, w, scales, biases],
        template=[("T", mx.bfloat16), ("K", K), ("N", N)],
        grid=(32, SPLIT_K, N // 8),
        threadgroup=(32, SPLIT_K, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
        verbose=True,
    )
    mx.eval(y)


def dump_current() -> None:
    """The kernel fastmlx currently ships, if its host module can build one."""

    try:
        from fastmlx.kernels import _qmm_skinny_mma_source as cur
    except Exception as exc:  # noqa: BLE001
        print(f"// current kernel unavailable: {exc}")
        return
    try:
        body = cur.build_source()
    except TypeError:
        print("// current build_source() needs arguments; skipped")
        return
    kernel = mx.fast.metal_kernel(
        name="fastmlx_current_qmv",
        input_names=["w", "scales", "biases", "x"],
        output_names=["y"],
        source=body,
        header=getattr(cur, "METAL_HEADER", ""),
    )
    m, k, n = 8, cur.BLOCK_SIZE, 64
    x = mx.zeros((m, k), dtype=mx.bfloat16)
    w = mx.zeros((n, k * cur.BITS // 32), dtype=mx.uint32)
    scales = mx.zeros((n, k // cur.GROUP_SIZE), dtype=mx.bfloat16)
    biases = mx.zeros((n, k // cur.GROUP_SIZE), dtype=mx.bfloat16)
    (y,) = kernel(
        inputs=[w, scales, biases, x],
        grid=(cur.active_input_groups(m) * 32, (n // 8) * 2, 1),
        threadgroup=cur.THREADGROUP,
        output_shapes=[(m, n)],
        output_dtypes=[mx.bfloat16],
        verbose=True,
    )
    mx.eval(y)


if __name__ == "__main__":
    print("//////// A2 skinny MMA (snapshot) ////////")
    dump_a2_mma()
    print("//////// current fastmlx kernel ////////")
    dump_current()
