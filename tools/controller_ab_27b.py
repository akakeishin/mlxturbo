"""27B controller の参照実装と旧実装を再現可能に比較する薄いwrapper。

本番 ``mlxturbo/spec.py`` に棄却済みの分岐を残さず、計測processの中だけ
``MLXTURBO_CONTROLLER_AB_ARM=reference,legacy`` を解釈する。reference arm は
本番実装と ``MLXTURBO_SPEC_GATE_H``、legacy arm は2026-09-04修正前の未来利得
walk、失敗位置keep、全tail miss更新、固定 ``h=0.19`` を使う。

    MLXTURBO_SPEC_GATE_H=0.05 tools/biglock.sh .venv/bin/python \
      tools/controller_ab_27b.py --model ~/models/qwen38-27b-4bit \
      --mtp ~/models/qwen38-27b-mtp \
      --knob MLXTURBO_CONTROLLER_AB_ARM=reference,legacy \
      --baseline legacy --ctx 0 --tokens 512 --reps 2 --out <path>
"""

from __future__ import annotations

import os

from mlxturbo.spec import GATE_EMA_ALPHA, SpecEngine, _pos_accept_prior

from tools import decode_ab_generic

_ARM_ENV = "MLXTURBO_CONTROLLER_AB_ARM"
_LEGACY_H = 0.19
_legacy_update = SpecEngine._record_pos_accept


def _legacy_plan(cls, pos_accept_ema, pos_obs_count=None, cap=1, h=_LEGACY_H):
    del cls, pos_obs_count, h
    if cap <= 0:
        return 0
    reach = 1.0
    keep = 0
    for d in range(1, cap + 1):
        reach *= max(0.0, min(1.0, pos_accept_ema.get(d, _pos_accept_prior(d))))
        future = 0.0
        running = 1.0
        for k in range(d + 1, cap + 1):
            running *= pos_accept_ema.get(k, _pos_accept_prior(k))
            future += running
        keep = d
        threshold = _LEGACY_H * (1.0 + future) / (1.0 + d * _LEGACY_H)
        if reach <= threshold:
            break
    return keep


def _reference_plan(cls, pos_accept_ema, pos_obs_count=None, cap=1, h=_LEGACY_H):
    del cls, pos_obs_count
    if cap <= 0:
        return 0
    reach = 1.0
    expected = 0.0
    keep = 0
    for d in range(1, cap + 1):
        reach *= max(0.0, min(1.0, pos_accept_ema.get(d, _pos_accept_prior(d))))
        threshold = h * (1.0 + expected) / (1.0 + (d - 1) * h)
        if reach <= threshold:
            break
        expected += reach
        keep = d
    return keep


def _ab_plan(cls, pos_accept_ema, pos_obs_count=None, cap=1, h=_LEGACY_H):
    if os.environ.get(_ARM_ENV) == "legacy":
        return _legacy_plan(cls, pos_accept_ema, pos_obs_count, cap, h)
    return _reference_plan(cls, pos_accept_ema, pos_obs_count, cap, h)


def _reference_update(
    pos_accept_ema, pos_obs_count, accepted, drafted, stopped_early=False,
):
    for d in range(1, min(accepted, drafted) + 1):
        pos_accept_ema[d] = (
            (1.0 - GATE_EMA_ALPHA)
            * pos_accept_ema.get(d, _pos_accept_prior(d))
            + GATE_EMA_ALPHA
        )
        pos_obs_count[d] = pos_obs_count.get(d, 0) + 1
    if accepted < drafted and not stopped_early:
        d = accepted + 1
        pos_accept_ema[d] = (
            (1.0 - GATE_EMA_ALPHA)
            * pos_accept_ema.get(d, _pos_accept_prior(d))
        )
        pos_obs_count[d] = pos_obs_count.get(d, 0) + 1
    elif accepted == drafted and drafted > 0 and not stopped_early:
        d = drafted + 1
        old = pos_accept_ema.get(d, _pos_accept_prior(d))
        if old < 0.95:
            pos_accept_ema[d] = old + GATE_EMA_ALPHA * (0.95 - old)


def _ab_update(
    pos_accept_ema, pos_obs_count, accepted, drafted, stopped_early=False,
):
    if os.environ.get(_ARM_ENV) == "legacy":
        return _legacy_update(
            pos_accept_ema, pos_obs_count, accepted, drafted, stopped_early,
        )
    return _reference_update(
        pos_accept_ema, pos_obs_count, accepted, drafted, stopped_early,
    )


SpecEngine._plan_depth = classmethod(_ab_plan)
SpecEngine._record_pos_accept = staticmethod(_ab_update)

if __name__ == "__main__":
    raise SystemExit(decode_ab_generic.main())
