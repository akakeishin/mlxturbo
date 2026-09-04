"""27B MTP proposal の q2 top-32 + q4 rerank を短文 A/B する wrapper。

本番のrerank / exact両methodをそのまま使い、計測process内だけ
``MLXTURBO_MTP_HEAD_AB_ARM=rerank,legacy`` を解釈する。q2 head はengine構築時に
exact q4 headから一度だけ作られ、両armで同じ約379 MiBを常駐させる。
target verifyと通常のtoken headは常にexactのまま。

    BIGLOCK_NO_WORKER=1 tools/biglock.sh .venv/bin/python \
      tools/mtp_head_rerank_ab_27b.py \
      --model ~/models/qwen38-27b-4bit --mtp ~/models/qwen38-27b-mtp \
      --knob MLXTURBO_MTP_HEAD_AB_ARM=rerank,legacy --baseline legacy \
      --ctx 0 --tokens 512 --reps 2 --out <path>
"""

from __future__ import annotations

import os

from mlxturbo.spec import SpecEngine

from tools import decode_ab_generic

_ARM_ENV = "MLXTURBO_MTP_HEAD_AB_ARM"
_exact_argmax = SpecEngine._draft_argmax_exact
_rerank_argmax = SpecEngine._draft_argmax


def _ab_draft_argmax(engine, h_mtp):
    if os.environ.get(_ARM_ENV) != "rerank":
        return _exact_argmax(engine, h_mtp)
    return _rerank_argmax(engine, h_mtp)


SpecEngine._draft_argmax = _ab_draft_argmax


if __name__ == "__main__":
    raise SystemExit(decode_ab_generic.main())
