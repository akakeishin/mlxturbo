"""`mlxturbo/kernels/moe_route_decode.py` の 2 本を素の並びと突き合わせる。

合成 (E=512 / H=2560 / top_k=10 / 4bit gs64 = 本番の形) で S ∈ {1,2,3,6} を
比べる。見るのは 3 つ:

1. **top-k の集合が 100% 一致するか** (順序は問わない。こちらは値の降順)。
2. ルータ重み (専門家番号で揃えて並べ直したうえで) の相対差。
3. region の出力 (`(y*w).sum(-2).astype(bf16) + sigmoid(sg)*shared`) の相対差。
   判定線は 1e-2 (丸め級)。

`switch_mlp` の中身 (gather_qmm) はここでは測らない。専門家の出力は
「専門家番号 -> ベクトル」の乱数表 (E, H) から引く = 実物と同じ依存関係
(素と自前で添字が違えば引く値も違う) になる。

    tools/biglock.sh .venv/bin/python tools/verify_moe_route_decode.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlxturbo.kernels import moe_route_decode as mrd  # noqa: E402

H, E, K = 2560, 512, 10
BITS, GS = 4, 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    if mx.default_device() != mx.gpu:
        print("GPU が要る")
        return 2

    mx.random.seed(args.seed)
    gw, gs_, gb = mx.quantize(
        mx.random.normal((E, H)).astype(mx.bfloat16), group_size=GS, bits=BITS)
    sgw = mx.random.normal((1, H)).astype(mx.bfloat16) * 0.05
    expert_out = mx.random.normal((E, H)).astype(mx.bfloat16)
    mx.eval(gw, gs_, gb, sgw, expert_out)

    ok = True
    for S in (1, 2, 3, 6):
        set_bad = 0.0
        n_slots = 0
        w_max = out_max = sg_max = 0.0
        for _ in range(args.reps):
            x = (mx.random.normal((1, S, H)) * 0.5).astype(mx.bfloat16)
            shared = mx.random.normal((1, S, H)).astype(mx.bfloat16)
            mx.eval(x, shared)

            # ---- 素 ----
            logits = mx.quantized_matmul(
                x.astype(mx.float32), gw, gs_, gb, transpose=True,
                group_size=GS, bits=BITS)
            p_idx = mx.argpartition(-logits, K - 1, axis=-1)[..., :K]
            p_w = mx.softmax(mx.take_along_axis(logits, p_idx, axis=-1),
                             axis=-1, precise=True)
            p_sg = mx.sigmoid(x @ sgw.T)
            p_y = expert_out[p_idx]                       # (1,S,K,H)
            p_out = ((p_y * p_w[..., None]).sum(axis=-2).astype(x.dtype)
                     + p_sg * shared)

            # ---- 自前 ----
            assert mrd.route_eligible(logits, K), "route_eligible が False"
            f_idx, f_w, f_sg = mrd.route(logits, K, x=x, sgw=sgw)
            f_y = expert_out[f_idx.astype(mx.int32)]
            assert mrd.combine_eligible(f_y, f_w, f_sg, shared, K), "combine 不可"
            f_out = mrd.combine(f_y, f_w, f_sg, shared, K)

            # ---- 集合と重み: 専門家番号で並べ直して比べる ----
            fo = mx.argsort(f_idx.astype(mx.int32), axis=-1)
            po = mx.argsort(p_idx.astype(mx.int32), axis=-1)
            fi = mx.take_along_axis(f_idx.astype(mx.int32), fo, axis=-1)
            pi = mx.take_along_axis(p_idx.astype(mx.int32), po, axis=-1)
            fw = mx.take_along_axis(f_w, fo, axis=-1)
            pw = mx.take_along_axis(p_w, po, axis=-1)
            mx.eval(fi, pi, fw, pw, f_out, p_out, f_sg, p_sg)

            set_bad += float(mx.sum(fi != pi))
            n_slots += int(fi.size)
            w_max = max(w_max, float(mx.max(
                mx.abs(fw - pw) / mx.maximum(mx.abs(pw), 1e-9))))
            sg1 = mrd.shared_gate(x, sgw)          # 単独カーネル (mode=sgate)
            mx.eval(sg1)
            sg_solo = float(mx.max(mx.abs(
                sg1.astype(mx.float32) - p_sg.astype(mx.float32))))
            sg_max = max(sg_max, sg_solo, float(mx.max(
                mx.abs(f_sg.reshape(p_sg.shape).astype(mx.float32)
                       - p_sg.astype(mx.float32))
                / mx.maximum(mx.abs(p_sg.astype(mx.float32)), 1e-6))))
            den = mx.maximum(mx.abs(p_out.astype(mx.float32)), 1e-3)
            out_max = max(out_max, float(mx.max(
                mx.abs(f_out.astype(mx.float32) - p_out.astype(mx.float32))
                / den)))

        bad = set_bad > 0 or w_max > 1e-5 or out_max > 1e-2 or sg_max > 1e-2
        ok = ok and not bad
        print(f"S={S}: 集合不一致 {int(set_bad)}/{n_slots}"
              f"  重み相対 max {w_max:.3e}"
              f"  sgate 相対 max {sg_max:.3e} (単独カーネル込み)"
              f"  出力相対 max {out_max:.3e}  {'NG' if bad else 'ok'}")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
