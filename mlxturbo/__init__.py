from . import _arch_registry
from .kernels.dispatch import enable

# Make qwen4_exp (Flash-Next) resolvable without writing into the user's mlx_lm.
# Importing any mlxturbo submodule always goes through here (a package __init__
# runs before its submodules). See the module docstring of _arch_registry.py for
# the details.
_arch_registry.install()

__all__ = ["enable"]
