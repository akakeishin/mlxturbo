"""27B MTP cache repair の先頭行保持を再現可能に比較する薄い wrapper。

本番 ``mlxturbo/spec.py`` に旧分岐を残さず、計測 process の中だけ
``MLXTURBO_MTP_REPAIR_AB_ARM=retain,legacy`` を解釈する。``retain`` は
先頭 draft が不採用だった round の同一 cache 行を保持し、``legacy`` は
その行も trim して同じ 1-token MTP append をやり直す。

    tools/biglock.sh .venv/bin/python tools/mtp_repair_ab_27b.py \
      --model ~/models/qwen38-27b-4bit --mtp ~/models/qwen38-27b-mtp \
      --knob MLXTURBO_MTP_REPAIR_AB_ARM=retain,legacy \
      --baseline legacy --ctx 0 --tokens 512 --reps 2 --out <path>
"""

from __future__ import annotations

import os

from mlxturbo.spec import SpecEngine

from tools import decode_ab_generic

_ARM_ENV = "MLXTURBO_MTP_REPAIR_AB_ARM"
_repair = SpecEngine._repair_mtp_cache


def _ab_repair(self, *args, **kwargs):
    kwargs["reuse_first"] = os.environ.get(_ARM_ENV) != "legacy"
    return _repair(self, *args, **kwargs)


SpecEngine._repair_mtp_cache = _ab_repair


if __name__ == "__main__":
    raise SystemExit(decode_ab_generic.main())
