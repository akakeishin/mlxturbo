"""qwen4_exp (Qwen3.8-Flash-Next) を利用者の mlx_lm へ書き込まずに解決する。

## 背景

Flash-Next のモデルクラスは mlx-lm 本体に無い (mlx-lm PR #1788 が未マージ)。
このリポジトリは vendored 版を `tools/vendor/qwen4_exp.py` として持っている。

mlx-lm がモデルクラスを解決する経路はただ一つ、
`mlx_lm/utils.py:_get_classes()` の

    importlib.import_module(f"mlx_lm.models.{model_type}")

だけで、mlx-lm 自身にプラグイン的な登録機構は無い。以前はこの vendored
ファイルを `convert_flash.py install-arch` で利用者の site-packages
(`<mlx_lm install先>/models/qwen4_exp.py`) へ物理コピーして解決させていたが、
これは利用者の mlx_lm パッケージを書き換える — `uv sync` や mlx-lm の
アップデートで消える、他のプロジェクトが同じ mlx_lm を使っていれば道連れに
なる、という副作用があった (レビュー指摘: mlxturbo を試すだけで環境が壊れる)。

## やっていること

`sys.meta_path` に、`mlx_lm.models.qwen4_exp` という完全修飾名だけを
vendored ファイルへ差し替える最小の finder を差し込む。この名前以外の
import には一切関与しない (find_spec は None を返し、既定の finder に
そのまま譲る) ので、mlx_lm を含めた他のあらゆる import に影響しない。

site-packages には何も書き込まない。`mlx_lm.models` パッケージの実体は
そのまま (通常のインストール) を使い、`from .base import ...` のような
vendored ファイル内の相対 import も、finder が spec に正しい `__name__` /
`__package__` (= "mlx_lm.models") を持たせることで、通常の
`mlx_lm.models.base` 解決に自然に乗る。

## なぜ `sys.modules` への直接登録ではなく meta_path finder なのか

`sys.modules["mlx_lm.models.qwen4_exp"] = <exec 済みモジュール>` を
mlxturbo の import 時に前もって置いておくだけでも、
`import mlx_lm.models.qwen4_exp as Q` / `from mlx_lm.models import qwen4_exp` /
`importlib.import_module(...)` のいずれの解決経路も (CPython の
import-as / from-import は attribute 参照に失敗すると `sys.modules[fullname]`
へフォールバックするため) 動く。だが vendored ファイル冒頭の
`NGRAM_ON_DISK = os.environ.get("FASTMLX_NGRAM_DISK") == "1"` は
モジュール実行時に 1 度だけ評価される値で、呼び出し側 (cli.py / server.py /
convert_flash.py / tools 配下) は「実際に qwen4_exp を読み込む直前」に
この環境変数を立てる規約になっている (mlxturbo の import 時点ではまだ
立っていない)。`sys.modules` への前倒し登録だと、この規約より前に
モジュール本体が実行されてしまい `FASTMLX_NGRAM_DISK` を無視した状態で
固まる。meta_path finder なら、実際に `mlx_lm.models.qwen4_exp` の import
が要求された瞬間 (= 呼び出し側が環境変数を立てた後) まで実行を遅らせられる
ので、この規約を崩さない。
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path

MODULE_NAME = "mlx_lm.models.qwen4_exp"
VENDOR_PATH = Path(__file__).resolve().parent.parent / "tools" / "vendor" / "qwen4_exp.py"


class _FlashNextArchFinder(importlib.abc.MetaPathFinder):
    """`mlx_lm.models.qwen4_exp` だけを vendored ファイルへ差し替える。"""

    def find_spec(self, fullname, path, target=None):
        if fullname != MODULE_NAME:
            return None
        if not VENDOR_PATH.exists():
            # vendor が無い環境 (配布物から tools/ を外した等) では黙って
            # 既定の finder に譲る。ここで例外を投げると qwen4_exp を使わない
            # モデルの import まで巻き込んで壊れる
            return None
        return importlib.util.spec_from_file_location(fullname, VENDOR_PATH)


def install() -> None:
    """qwen4_exp を `mlx_lm.models` 名前空間へ解決できるようにする。

    べき等 (2 回呼んでも finder は 1 つしか積まない)。`mlxturbo` の
    `__init__.py` から呼ばれるので、`mlxturbo` を import した時点で
    (実際に qwen4_exp を使うかとは無関係に) 常に有効になる。
    """

    if any(isinstance(f, _FlashNextArchFinder) for f in sys.meta_path):
        return
    # 既に site-packages に古い install-arch のコピーが残っていても、
    # こちらを優先する (vendor が正本)。他の名前には関与しないため
    # 先頭に挿しても他の import への影響は無い
    sys.meta_path.insert(0, _FlashNextArchFinder())
