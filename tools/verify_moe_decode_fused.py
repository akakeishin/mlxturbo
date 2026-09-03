"""`mlxturbo/kernels/moe_decode_fused.py` の正しさゲート (合成重み、数秒)。

判定は「素の並び (`mx.gather_qmm` x2 + SwiGLU + `mx.gather_qmm`) との差が、
素の並び自身が fp32 参照 (dequantize してから密に掛ける) に対して持つ誤差と
同じ帯に収まるか」。ビット一致は求めない (積和の順序が違う)。

bf16 で K=2560 の内積を取ると真値が 0 近傍の要素で相対誤差が発散するので、
相対誤差は |ref| が最大の 10% 以上の要素だけで測る (`moe_verify_gather.py`
の atol/rtol の議論と同じ理由)。

多重度が RMAX (=4) を超える敵対ケース (S=6 の全行が同じ専門家) も入れる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from mlxturbo.kernels import moe_decode_fused as mdf  # noqa: E402

GS, BITS = 64, 4
QK = dict(transpose=True, group_size=GS, bits=BITS, mode="affine")


def make_set(E, out_dim, in_dim):
    hi = mx.random.randint(0, 1 << 16, shape=(E, out_dim, in_dim // 8), dtype=mx.uint32)
    lo = mx.random.randint(0, 1 << 16, shape=(E, out_dim, in_dim // 8), dtype=mx.uint32)
    w = hi * 65536 + lo
    s = (mx.random.uniform(shape=(E, out_dim, in_dim // GS)) * 0.015 + 0.005).astype(
        mx.bfloat16)
    b = (-7.5 * s).astype(mx.bfloat16)
    return w, s, b


def deq(m):
    return mx.dequantize(m[0], m[1], m[2], group_size=GS, bits=BITS,
                         mode="affine").astype(mx.float32)


def ref_mlx(x, idx, gate, up, dn, topk, K):
    S = x.shape[0]
    P = S * topk
    xp = mx.repeat(x[:, None, :], topk, axis=1).reshape(P, 1, K)
    ii = idx.reshape(P)
    g = mx.gather_qmm(xp, *gate, rhs_indices=ii, **QK).squeeze(-2)
    u = mx.gather_qmm(xp, *up, rhs_indices=ii, **QK).squeeze(-2)
    h = mx.sigmoid(g) * g * u
    y = mx.gather_qmm(h[:, None, :], *dn, rhs_indices=ii, **QK).squeeze(-2)
    return h, y


def ref_f32(x, idx, dg, du, dd, topk):
    S, K = x.shape
    ii = np.asarray(idx).reshape(-1)
    xf = np.asarray(x.astype(mx.float32))
    G = np.asarray(dg); U = np.asarray(du); D = np.asarray(dd)
    hs, ys = [], []
    for p, e in enumerate(ii):
        v = xf[p // topk]
        g = G[e] @ v
        u = U[e] @ v
        h = (g / (1.0 + np.exp(-g))) * u
        hs.append(h)
        ys.append(D[e] @ h)
    return np.stack(hs), np.stack(ys)


def relerr(a, ref, frac=0.10):
    a = np.asarray(a, dtype=np.float64); ref = np.asarray(ref, dtype=np.float64)
    m = np.abs(ref) > frac * np.abs(ref).max()
    if not m.any():
        return 0.0
    return float((np.abs(a - ref)[m] / np.abs(ref)[m]).max())


def gen_idx(S, topk, E, share, rng, adversarial=False):
    union: list[int] = []
    rows = []
    for i in range(S):
        if adversarial and i > 0:
            row = rows[0][:]
        else:
            borrowed = list(rng.choice(union, size=min(share, len(union), topk),
                                       replace=False)) if (i and union) else []
            pool = [e for e in range(E) if e not in set(borrowed)]
            fresh = list(rng.choice(pool, size=topk - len(borrowed), replace=False))
            row = borrowed + fresh
            rng.shuffle(row)
        rows.append([int(v) for v in row])
        for e in row:
            if int(e) not in union:
                union.append(int(e))
    return mx.array(np.asarray(rows, dtype=np.uint32)), len(union)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--inter", type=int, default=640)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--share", type=int, default=4)
    ap.add_argument("--tol", type=float, default=2.0,
                    help="fused の誤差 / 素の誤差 の許容比")
    a = ap.parse_args()

    mx.random.seed(7)
    E, K, H, topk = a.experts, a.hidden, a.inter, a.topk
    gate = make_set(E, H, K); up = make_set(E, H, K); dn = make_set(E, K, H)
    mx.eval([t for m in (gate, up, dn) for t in m])
    dg, du, dd = deq(gate), deq(up), deq(dn)
    mx.eval(dg, du, dd)

    rng = np.random.default_rng(3)
    ok = True
    cases = [(s, False) for s in (1, 2, 3, 6)] + [(6, True)]
    for S, adv in cases:
        idx, u = gen_idx(S, topk, E, a.share, rng, adv)
        x = (mx.random.normal((S, K)) * 0.05).astype(mx.bfloat16)
        mx.eval(x, idx)

        h_m, y_m = ref_mlx(x, idx, gate, up, dn, topk, K)
        h_f = mdf.gate_up(x, idx.reshape(-1), gate, up, topk)
        y_f = mdf.down(h_f, idx.reshape(-1), dn, topk, K)
        mx.eval(h_m, y_m, h_f, y_f)
        h_r, y_r = ref_f32(x, idx, dg, du, dd, topk)

        # h の比較は同じ入力 x なので直接。y は入力 h が違うので
        # 「自分の h から出した y」を各々の fp32 参照と比べる。
        eh_m = relerr(np.asarray(h_m.astype(mx.float32)), h_r)
        eh_f = relerr(np.asarray(h_f.astype(mx.float32)), h_r)
        ey_m = relerr(np.asarray(y_m.astype(mx.float32)), y_r)
        ey_f = relerr(np.asarray(y_f.astype(mx.float32)), y_r)
        # 融合 vs 素の直接比較 (どちらも bf16 出力)
        d_h = relerr(np.asarray(h_f.astype(mx.float32)),
                     np.asarray(h_m.astype(mx.float32)))
        # 判定: fp32 参照に対して素より悪くない (1.5 倍以内) こと。
        # 「素との直接差」は素自身の誤差 (2e-2〜1.4e-1) に支配されるので
        # 判定に使わず、両者の誤差の和の帯 (3 倍) に収まるかだけ見る。
        good = (eh_f <= max(1.5 * eh_m, 1e-2) and ey_f <= max(1.5 * ey_m, 1e-2)
                and d_h <= 3.0 * (eh_m + eh_f))
        ok &= good
        print(f"{'OK ' if good else 'NG '}S={S}{' adv' if adv else '    '} "
              f"union {u:2d}/{S*topk:2d}  "
              f"h: 素 {eh_m:.2e} / 融合 {eh_f:.2e}   "
              f"y: 素 {ey_m:.2e} / 融合 {ey_f:.2e}   "
              f"融合-素 {d_h:.2e}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
