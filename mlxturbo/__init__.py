from . import _arch_registry
from .kernels.dispatch import enable

# qwen4_exp (Flash-Next) を利用者の mlx_lm へ書き込まずに解決できるようにする。
# mlxturbo のどのサブモジュールを import しても (パッケージ __init__ は
# サブモジュールより先に実行されるので) 必ずここを通る。詳細は
# _arch_registry.py のモジュール docstring を参照
_arch_registry.install()

__all__ = ["enable"]
