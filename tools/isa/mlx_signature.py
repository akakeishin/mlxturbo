"""Wrap an ``mx.fast.metal_kernel`` body into a standalone ``.metal`` file.

MLX generates the kernel signature itself, so a body extracted from
``build_source()`` will not compile on its own.  This module reproduces that
generated signature offline so ``xcrun metal`` can compile the exact same text
without a GPU.

The reproduction was derived from the format strings embedded in the shipped
``libmlx.dylib`` (mlx 0.32.2); see ``docs/ISA-NOTES.md`` for the extraction.
Whitespace may differ from MLX byte-for-byte, which does not affect codegen.
``tools/isa/mlx_dump_source.py`` (GPU queue) dumps MLX's real text so the two
can be diffed.
"""

from __future__ import annotations

# (attribute name, metal type).  MLX emits an attribute parameter only when its
# name occurs in the kernel body, so the generated signature is body-dependent.
METAL_ATTRIBUTES: list[tuple[str, str]] = [
    ("dispatch_quadgroups_per_threadgroup", "uint"),
    ("dispatch_simdgroups_per_threadgroup", "uint"),
    ("dispatch_threads_per_threadgroup", "uint3"),
    ("grid_origin", "uint3"),
    ("grid_size", "uint3"),
    ("quadgroup_index_in_threadgroup", "uint"),
    ("quadgroups_per_threadgroup", "uint"),
    ("simdgroup_index_in_threadgroup", "uint"),
    ("simdgroups_per_threadgroup", "uint"),
    ("thread_execution_width", "uint"),
    ("thread_index_in_quadgroup", "uint"),
    ("thread_index_in_simdgroup", "uint"),
    ("thread_index_in_threadgroup", "uint"),
    ("thread_position_in_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"),
    ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"),
    ("threads_per_grid", "uint3"),
    ("threads_per_simdgroup", "uint"),
    ("threads_per_threadgroup", "uint3"),
]

# The part of mlx/backend/metal/kernels/utils.h that a custom kernel actually
# depends on.  Pulling the real header would drag in the whole MLX kernel tree
# for no codegen difference: what the body needs is the namespace and the
# ``bfloat16_t`` typedef, which mlx 0.32.2 defines as native ``bfloat``
# (verified against the string table of libmlx.dylib).
MLX_PREAMBLE = """#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

typedef bfloat bfloat16_t;
"""


def _used_attributes(body: str) -> list[tuple[str, str]]:
    return [(n, t) for n, t in METAL_ATTRIBUTES if n in body]


def build_metal_file(
    *,
    name: str,
    body: str,
    input_names: list[str],
    input_types: list[str],
    output_names: list[str],
    output_types: list[str],
    template_params: list[tuple[str, str]],
    template_args: list[str],
    header: str = "",
) -> str:
    """Return a compilable ``.metal`` translation unit.

    ``template_params`` is ``[("typename", "T"), ("int", "K"), ...]`` and
    ``template_args`` the concrete instantiation, e.g. ``["bfloat16_t", "5120"]``.
    """

    func = f"custom_kernel_{name}"
    params: list[str] = []
    buf = 0
    for n, t in zip(input_names, input_types):
        params.append(f"  const device {t}* {n} [[buffer({buf})]]")
        buf += 1
        # MLX appends a shape / strides / ndim buffer for an input only when the
        # body mentions it by name.
        if f"{n}_shape" in body:
            params.append(f"  const constant int* {n}_shape [[buffer({buf})]]")
            buf += 1
        if f"{n}_strides" in body:
            params.append(f"  const constant int64_t* {n}_strides [[buffer({buf})]]")
            buf += 1
        if f"{n}_ndim" in body:
            params.append(f"  const constant int& {n}_ndim [[buffer({buf})]]")
            buf += 1
    for n, t in zip(output_names, output_types):
        params.append(f"  device {t}* {n} [[buffer({buf})]]")
        buf += 1
    for attr, ty in _used_attributes(body):
        params.append(f"  {ty} {attr} [[{attr}]]")

    signature = f"""[[kernel]] void {func}(
{",\n".join(params)}) {{
{body}
}}
"""
    if not template_params:
        # MLX emits a plain kernel when the call site passes no template args.
        return f"{MLX_PREAMBLE}\n{header}\n{signature}"

    tdef = ", ".join(f"{kind} {pname}" for kind, pname in template_params)
    targs = ", ".join(template_args)
    inst = f"{func}<{targs}>"
    host = f"{func}_" + "_".join(a.replace(" ", "") for a in template_args)

    return f"""{MLX_PREAMBLE}
{header}
template <{tdef}>
{signature}
template [[host_name("{host}")]] [[kernel]] decltype({inst}) {inst};
"""


def qmm_metal_file(*, name: str, body: str, dtype: str, k: int, n: int) -> str:
    """Signature used by ``mlxturbo.kernels.qmm_skinny_mma``."""

    return build_metal_file(
        name=name,
        body=body,
        input_names=["x", "w", "scales", "biases"],
        input_types=[dtype, "uint32_t", dtype, dtype],
        output_names=["y"],
        output_types=[dtype],
        template_params=[("typename", "T"), ("int", "K"), ("int", "N")],
        template_args=[dtype, str(k), str(n)],
    )
