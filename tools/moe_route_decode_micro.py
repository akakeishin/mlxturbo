"""MoE の「router〜combine」region だけを冷やして連鎖で測る (モデル不要)。

## 測る範囲

`SparseMoeBlock.__call__` のうち **専門家の GEMV (`switch_mlp`) を除いた全部**。
router の行列積は両方に同じだけ入れる (触らないので)。

    x.astype(f32) -> router qmm -> [top-k, softmax, shared gate] -> [combine]

`switch_mlp` の出力に相当する `y` (rows, k, H) は乱数の葉として与える。
専門家 GEMV は 2026-09-03 20:08 の PoL で「既に帯域のピーク」と決着済みなので、
ここに入れると測りたい糊の差が薄まるだけになる。

## 冷やし方 (CLAUDE.md「連鎖 micro は重みを 100 MB 超 巡回させて冷やす」)

router の重み 1 組が 4bit で 640 KB + scales/biases 80 KB = 約 720 KB。
本番の層数 48 では 35 MB しか無く、SLC に載ってしまう。そこで
`--sets` を 2 通り回す (既定 `48,144`): 48 = 本番の層数、144 = 104 MB で
確実に DRAM から読む条件。**判定は冷えている 144 側で行う。**

`x` は 1 組の出力を次の組の入力に渡して直列にする (MLX の遅延グラフが
勝手に並べ替えないよう、1 巡ごとに `mx.eval`)。

    tools/biglock.sh .venv/bin/python tools/moe_route_decode_micro.py
    tools/biglock.sh .venv/bin/python tools/moe_route_decode_micro.py --sweep
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlxturbo.kernels import moe_route_decode as mrd  # noqa: E402

H, E, K = 2560, 512, 10
BITS, GS = 4, 64


class Bank:
    """`sets` 組ぶんの router 重み / shared ゲート / 専門家出力。"""

    def __init__(self, sets: int, S: int, seed: int = 0):
        mx.random.seed(seed)
        self.sets = sets
        self.S = S
        self.gw, self.gs, self.gb, self.sgw, self.y, self.shared = [], [], [], [], [], []
        for _ in range(sets):
            w, s, b = mx.quantize(
                mx.random.normal((E, H)).astype(mx.bfloat16),
                group_size=GS, bits=BITS)
            self.gw.append(w)
            self.gs.append(s)
            self.gb.append(b)
            self.sgw.append((mx.random.normal((1, H)) * 0.05).astype(mx.bfloat16))
            self.y.append(mx.random.normal((1, S, K, H)).astype(mx.bfloat16))
            self.shared.append(
                (mx.random.normal((1, S, H)) * 0.1).astype(mx.bfloat16))
        self.x0 = (mx.random.normal((1, S, H)) * 0.5).astype(mx.bfloat16)
        mx.eval(self.gw, self.gs, self.gb, self.sgw, self.y, self.shared, self.x0)

    def bytes(self) -> int:
        one = self.gw[0].nbytes + self.gs[0].nbytes + self.gb[0].nbytes
        return one * self.sets


def _logits(bank: Bank, i: int, x):
    return mx.quantized_matmul(
        x.astype(mx.float32), bank.gw[i], bank.gs[i], bank.gb[i],
        transpose=True, group_size=GS, bits=BITS)


def _step_plain(bank: Bank, i: int, x):
    logits = _logits(bank, i, x)
    idx = mx.argpartition(-logits, K - 1, axis=-1)[..., :K]
    w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1,
                   precise=True)
    sg = mx.sigmoid(x @ bank.sgw[i].T)
    # idx を生かすため専門家出力を添字でずらす (両側に同じ 1 op。gather は無し)
    return ((bank.y[i] * w[..., None]).sum(axis=-2).astype(x.dtype)
            + sg * bank.shared[i])


def _step_fused(bank: Bank, i: int, x, vec: int, tg: int):
    logits = _logits(bank, i, x)
    _idx, w, sg = mrd.route(logits, K, x=x, sgw=bank.sgw[i])
    return mrd.combine(bank.y[i], w, sg, bank.shared[i], K, vec=vec, tg=tg)


def chain(bank: Bank, step) -> tuple[float, float]:
    """依存連鎖を 1 度も eval せずに組んでから 1 回だけ eval する。

    段ごとに `mx.eval` すると 1 段あたり 200 us の command buffer 境界が乗り、
    測りたい糊の差 (数十 us) が埋もれる (2026-09-03 に実測: 段ごと eval だと
    素が 260 us/段。本番の帰属は 70 us/層)。
    """
    x = bank.x0
    t0 = time.perf_counter()
    for i in range(bank.sets):
        x = step(i, x)
    t1 = time.perf_counter()
    mx.eval(x)
    t2 = time.perf_counter()
    n = bank.sets
    return (t1 - t0) * 1e6 / n, (t2 - t1) * 1e6 / n     # 組み立て / GPU (us/組)


def timed(bank: Bank, step) -> float:
    return chain(bank, step)[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="48,144")
    ap.add_argument("--widths", default="1,2,3,6")
    ap.add_argument("--reps", type=int, default=4, help="ABBA を何往復")
    ap.add_argument("--sweep", action="store_true",
                    help="combine の (VEC, TG) を掃く")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if mx.default_device() != mx.gpu:
        print("GPU が要る")
        return 2

    configs = ([(1, 32), (1, 64), (2, 32), (2, 64), (4, 32), (4, 64), (4, 128)]
               if args.sweep else [(mrd.COMBINE_VEC, mrd.COMBINE_TG)])
    rows = []
    for n_sets in [int(v) for v in args.sets.split(",")]:
        for S in [int(v) for v in args.widths.split(",")]:
            bank = Bank(n_sets, S)
            mb = bank.bytes() / 1e6
            for vec, tg in configs:
                sp = lambda i, x: _step_plain(bank, i, x)              # noqa: E731
                sf = lambda i, x: _step_fused(bank, i, x, vec, tg)     # noqa: E731
                # 焼き入れ 1 往復は捨てる
                timed(bank, sp)
                timed(bank, sf)
                ps, fs = [], []
                for _ in range(args.reps):        # ABBA
                    ps.append(timed(bank, sp))
                    fs.append(timed(bank, sf))
                    fs.append(timed(bank, sf))
                    ps.append(timed(bank, sp))
                p = statistics.median(ps)
                f = statistics.median(fs)
                rows.append({"sets": n_sets, "weight_mb": round(mb, 1), "S": S,
                             "vec": vec, "tg": tg, "plain_us": round(p, 2),
                             "fused_us": round(f, 2), "ratio": round(f / p, 3),
                             "plain_all": [round(v, 2) for v in ps],
                             "fused_all": [round(v, 2) for v in fs]})
                print(f"sets={n_sets:3d} ({mb:5.0f} MB) S={S} VEC={vec} TG={tg:3d}"
                      f"  素 {p:7.2f} us  自前 {f:7.2f} us  比 {f / p:.3f}")
            del bank
            mx.clear_cache()

    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
