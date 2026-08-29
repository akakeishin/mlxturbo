"""Resolve qwen4_exp (Qwen3.8-Flash-Next) without writing into the user's mlx_lm.

## Background

The Flash-Next model class is not part of mlx-lm proper (mlx-lm PR #1788 is
still unmerged). This package carries a vendored copy as
`mlxturbo/_vendor/qwen4_exp.py`, so that it ships inside the wheel and
resolves from an installed environment, not just a repo checkout.

mlx-lm has exactly one path for resolving a model class,

    importlib.import_module(f"mlx_lm.models.{model_type}")

in `mlx_lm/utils.py:_get_classes()`, and mlx-lm itself has no plugin-style
registration mechanism. Previously we resolved this by having
`convert_flash.py install-arch` physically copy the vendored file into the
user's site-packages (`<mlx_lm install dir>/models/qwen4_exp.py`), but that
rewrites the user's mlx_lm package, with the side effects that it disappears
on `uv sync` or on an mlx-lm update, and that any other project sharing the
same mlx_lm gets dragged along with it (review comment: merely trying out
mlxturbo breaks your environment).

## What this does instead

It inserts into `sys.meta_path` a minimal finder that redirects only the fully
qualified name `mlx_lm.models.qwen4_exp` to the vendored file. It takes no part
in any other import (find_spec returns None and defers to the default finders
as-is), so it affects nothing else, mlx_lm included.

Nothing is written into site-packages. The `mlx_lm.models` package itself stays
as it is (the normal installation), and relative imports inside the vendored
file such as `from .base import ...` ride naturally on the usual
`mlx_lm.models.base` resolution, because the finder gives the spec the correct
`__name__` / `__package__` (= "mlx_lm.models").

## Why a meta_path finder rather than registering directly in `sys.modules`

Simply placing `sys.modules["mlx_lm.models.qwen4_exp"] = <already-executed
module>` ahead of time when mlxturbo is imported would also work for every
resolution path — `import mlx_lm.models.qwen4_exp as Q` /
`from mlx_lm.models import qwen4_exp` / `importlib.import_module(...)` — because
CPython's import-as / from-import fall back to `sys.modules[fullname]` when the
attribute lookup fails. But
`NGRAM_ON_DISK = os.environ.get("FASTMLX_NGRAM_DISK") == "1"` at the top of the
vendored file is evaluated exactly once when the module executes, and the
convention on the calling side (cli.py / server.py / convert_flash.py / things
under tools) is to set that environment variable immediately before qwen4_exp
is actually loaded (it is not yet set at the time mlxturbo is imported).
Registering into `sys.modules` up front would execute the module body before
that convention gets its chance, freezing it in a state that ignored
`FASTMLX_NGRAM_DISK`. A meta_path finder lets us defer execution until the
moment the import of `mlx_lm.models.qwen4_exp` is actually requested (= after
the caller has set the environment variable), so the convention stays intact.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path

MODULE_NAME = "mlx_lm.models.qwen4_exp"
# Resolved relative to this module's own installed location (not the repo
# root), so this works the same from a `pip install`-ed wheel as it does from
# a repo checkout — nothing here depends on `tools/` existing on disk.
VENDOR_PATH = Path(__file__).resolve().parent / "_vendor" / "qwen4_exp.py"


class _FlashNextArchFinder(importlib.abc.MetaPathFinder):
    """Redirect only `mlx_lm.models.qwen4_exp` to the vendored file."""

    def find_spec(self, fullname, path, target=None):
        if fullname != MODULE_NAME:
            return None
        if not VENDOR_PATH.exists():
            # Should not happen for a normal install (the vendor file ships
            # inside the mlxturbo package itself), but silently defer to the
            # default finders rather than raise — that would also break
            # imports of models that never use qwen4_exp.
            return None
        return importlib.util.spec_from_file_location(fullname, VENDOR_PATH)


def install() -> None:
    """Make qwen4_exp resolvable within the `mlx_lm.models` namespace.

    Idempotent (calling it twice still installs only one finder). It is called
    from `mlxturbo`'s `__init__.py`, so it is always in effect as soon as
    `mlxturbo` is imported (regardless of whether qwen4_exp actually gets used).
    """

    if any(isinstance(f, _FlashNextArchFinder) for f in sys.meta_path):
        return
    # Even if an old install-arch copy is still sitting in site-packages, this
    # one takes precedence (the vendored file is the source of truth). Since it
    # takes no part in any other name, inserting it at the front has no effect
    # on other imports.
    sys.meta_path.insert(0, _FlashNextArchFinder())
