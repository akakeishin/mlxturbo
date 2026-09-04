"""GDN 前処理融合の外側ゲートを CPU 合成条件で固定する。

配列の形・dtype を見る ``eligible()`` や Metal カーネルは呼ばず、
``GatedDeltaNet.__call__`` と ``spec_flash.capture()`` が共有する
``gdn_prework.wants()`` の契約だけを検査する。
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_predicate():
    """MLX/Metal を初期化せず、対象モジュールの wants() を読み込む。"""
    root = Path(__file__).resolve().parent.parent
    package = "_gdn_prework_contract"
    package_module = ModuleType(package)
    package_module.__path__ = [str(root / "mlxturbo" / "kernels")]
    fire_module = ModuleType(f"{package}._fire")
    mlx_module = ModuleType("mlx")
    core_module = ModuleType("mlx.core")
    mlx_module.core = core_module
    names = {
        package: package_module,
        f"{package}._fire": fire_module,
        "mlx": mlx_module,
        "mlx.core": core_module,
    }
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names}
    try:
        sys.modules.update(names)
        name = f"{package}.gdn_prework"
        spec = importlib.util.spec_from_file_location(
            name, root / "mlxturbo" / "kernels" / "gdn_prework.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module.wants
    finally:
        for name, old in previous.items():
            if old is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


gp_wants = _load_predicate()


_MISSING = object()


def _module(*, enabled: bool = True, training: bool = False):
    return SimpleNamespace(_gdn_prework=enabled, training=training)


def _cache(*, lengths=_MISSING):
    cache = SimpleNamespace()
    if lengths is not _MISSING:
        cache.lengths = lengths
    return cache


@pytest.mark.parametrize(
    ("module", "mask", "cache", "expected"),
    [
        (_module(enabled=True), None, _cache(), True),
        (_module(enabled=False), None, _cache(), False),
        (_module(), object(), _cache(), False),
        (_module(), None, None, False),
        (_module(training=True), None, _cache(), False),
        (_module(), None, _cache(lengths=[1]), False),
        (_module(), None, _cache(lengths=None), True),
    ],
    ids=[
        "enabled",
        "disabled",
        "mask",
        "cache-none",
        "training",
        "cache-lengths-present",
        "cache-lengths-none",
    ],
)
def test_wants_outer_contract(module, mask, cache, expected):
    """有効化以降の 4 条件を一つでも外すと素の経路を選ぶ。"""
    assert gp_wants(module, mask, cache) is expected


def test_callers_use_shared_outer_predicate():
    """本家写しと capture 写しが同じ述語を呼ぶ。"""
    root = Path(__file__).resolve().parent.parent
    expected_args = [
        "Name(id='self', ctx=Load())",
        "Name(id='mask', ctx=Load())",
        "Name(id='cache', ctx=Load())",
    ]
    for relative in (
        "mlxturbo/_vendor/qwen4_exp.py",
        "mlxturbo/spec_flash.py",
    ):
        tree = ast.parse((root / relative).read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "wants"
            and [ast.dump(arg) for arg in node.args]
            == expected_args
        ]
        assert len(calls) == 1, f"{relative}: shared wants() call count={len(calls)}"
