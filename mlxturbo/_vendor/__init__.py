"""Vendored third-party model code shipped inside the mlxturbo package.

Not a regular import target: `mlxturbo._arch_registry` loads
`qwen4_exp.py` directly via `importlib.util.spec_from_file_location`
so that it lands under the `mlx_lm.models` namespace instead of this
one. See `mlxturbo/_arch_registry.py` for why.
"""
