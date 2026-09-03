"""GDN scan の 3 者 (逐次カーネル / kernel S / レジスタ常駐) を突き合わせる。

**GPU 必須。**`tools/verify_gdn_metal.py` と同じ形で、基準は `mlx_lm` の逐次
カーネル (`gated_delta_update`)。ビット一致は要求しない (加算順が違う) が、
レジスタ常駐版の逐次版との差が、既に本番で受けている kernel S の差と同じ級で
あることを見る。合格の目安は kernel S と同じ相対誤差 1e-3。

    tools/biglock.sh .venv/bin/python tools/verify_gdn_scan_reg.py
    tools/biglock.sh .venv/bin/python tools/verify_gdn_scan_reg.py --sweep
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import mlx.nn as nn

import mlxturbo  # noqa: F401  (qwen4_exp の解決と mlx 契約の検査)
from mlx_lm.models.gated_delta import gated_delta_update
from mlxturbo.kernels import gdn_blocked_metal as gbm
from mlxturbo.kernels import gdn_scan_reg as gsr

# Qwen3.8-Flash-Next の GDN の形 (verify_gdn_metal.py と同じ)
N_K, N_V, DK, DV = 16, 48, 128, 128

# 掃く割り当て (lanes, db)。lanes*db = threadgroup のスレッド数。
LAYOUTS = [
    (32, 8),   # Lily の説明どおり「1 行 = 1 simdgroup」
    (16, 16),
    (8, 32),   # kernel S と同じ割り当て
    (4, 64),
    (4, 32),
    (2, 128),
    (2, 64),
]


def _make_inputs(B, T, dtype, seed, with_state):
    """本番の `GatedDeltaNet.__call__` が渡すのと同じ性質の入力を作る。"""
    mx.random.seed(seed)
    q = mx.random.normal((B, T, N_K, DK)).astype(dtype)
    k = mx.random.normal((B, T, N_K, DK)).astype(dtype)
    v = mx.random.normal((B, T, N_V, DV)).astype(dtype)
    inv = DK**-0.5
    q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv * mx.fast.rms_norm(k, None, 1e-6)
    v = nn.silu(v)
    a = mx.random.normal((B, T, N_V)).astype(dtype)
    b = mx.random.normal((B, T, N_V)).astype(dtype)
    A_log = mx.random.normal((N_V,)) * 0.5
    dt_bias = mx.ones((N_V,))
    state = None
    if with_state:
        state = (mx.random.normal((B, N_V, DV, DK)) * 0.1).astype(mx.float32)
    return q, k, v, a, b, A_log, dt_bias, state


def _rel(x, ref):
    x = x.astype(mx.float32)
    ref = ref.astype(mx.float32)
    d = x - ref
    scale = math.sqrt(float(mx.mean(ref * ref)))
    if scale == 0:
        scale = 1.0
    return (
        float(mx.max(mx.abs(d))) / scale,
        math.sqrt(float(mx.mean(d * d))) / scale,
    )


def run(B, T, dtype, seed, with_state, layouts, tol=1e-3):
    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(B, T, dtype, seed, with_state)

    y_seq, s_seq = gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=True
    )
    mx.eval(y_seq, s_seq)

    y_met, s_met = gbm.gated_delta_update_blocked_metal(
        q, k, v, a, b, A_log, dt_bias, state
    )
    mx.eval(y_met, s_met)

    tag = "state あり" if with_state else "state=None"
    ym, yr = _rel(y_met, y_seq)
    sm, sr = _rel(s_met, s_seq)
    flag = "  ** 1e-3 超え **" if max(yr, sr) > tol else ""
    print(
        f"B={B} T={T} {dtype} {tag}  blocked (kernel S): "
        f"y max={ym:.3e} rel={yr:.3e} | state max={sm:.3e} rel={sr:.3e}{flag}"
    )

    worst = max(yr, sr)
    for lanes, db in layouts:
        ok, why = gsr.layout_ok(DK, DV, lanes, db)
        if not ok:
            print(f"    reg lanes={lanes} db={db}: 対象外 ({why})")
            continue
        y_reg, s_reg = gsr.gated_delta_update_scan_reg(
            q, k, v, a, b, A_log, dt_bias, state, lanes=lanes, db=db
        )
        mx.eval(y_reg, s_reg)
        ym, yr = _rel(y_reg, y_seq)
        sm, sr = _rel(s_reg, s_seq)
        # kernel S との直接差も見る (本番で入れ替える相手はこちら)
        _, yrm = _rel(y_reg, y_met)
        _, srm = _rel(s_reg, s_met)
        flag = "  ** 1e-3 超え **" if max(yr, sr) > tol else ""
        print(
            f"    reg lanes={lanes:2d} db={db:3d}: "
            f"y rel={yr:.3e} | state rel={sr:.3e} "
            f"| vs kernel S: y {yrm:.3e} state {srm:.3e}{flag}"
        )
        worst = max(worst, yr, sr)
    return worst


def check_dispatch(tol=1e-3) -> float:
    """`MLXTURBO_GDN_SCAN` の差し替え口 (gdn_blocked_metal 側) を通す。

    呼び手 (`_vendor/qwen4_exp.py`) が実際に呼ぶのは
    `gated_delta_update_blocked_metal` なので、そこが switch を見て
    レジスタ常駐版へ渡っていること (発火カウンタ) と、返る値が kernel S と
    同じ級であることを見る。
    """
    from mlxturbo.kernels import _fire

    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(1, 512, mx.bfloat16, 5, True)
    _fire.reset()
    gsr.set_active(True)
    y_reg, s_reg = gbm.gated_delta_update_blocked_metal(
        q, k, v, a, b, A_log, dt_bias, state
    )
    mx.eval(y_reg, s_reg)
    fired_reg = _fire.snapshot()
    gsr.set_active(False)
    y_blk, s_blk = gbm.gated_delta_update_blocked_metal(
        q, k, v, a, b, A_log, dt_bias, state
    )
    mx.eval(y_blk, s_blk)
    fired_all = _fire.snapshot()

    ok_fire = (
        fired_reg.get("gdn_scan_reg") == 1
        and fired_reg.get("gdn_metal") is None
        and fired_all.get("gdn_metal") == 1
    )
    _, yr = _rel(y_reg, y_blk)
    _, sr = _rel(s_reg, s_blk)
    print(
        f"\n差し替え口 (gated_delta_update_blocked_metal): 発火 {fired_all} "
        f"{'OK' if ok_fire else '** 期待と違う **'}\n"
        f"    reg vs kernel S: y rel={yr:.3e} state rel={sr:.3e}"
        + ("" if max(yr, sr) <= tol else "  ** 1e-3 超え **")
    )
    return max(yr, sr) if ok_fire else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="割り当ての全組を見る")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    layouts = LAYOUTS if args.sweep else [(gsr.DEFAULT_LANES, gsr.DEFAULT_DB)]
    worst = 0.0
    for T in (64, 512, 2048):
        for with_state in (False, True):
            worst = max(
                worst, run(1, T, mx.bfloat16, 7 + T, with_state, layouts, args.tol)
            )
    worst = max(worst, check_dispatch(args.tol))
    print(f"\n最大相対誤差 (RMS) = {worst:.3e}  (合格の目安 {args.tol})")
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
