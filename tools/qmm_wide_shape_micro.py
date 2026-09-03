"""`kernels/qmm_wide.py` の広タイルが素とビット一致する形を (M, K, N) で掃引する。

`docs/BACKLOG.md` 末尾の「qmm_wide は M < 1024 で HC の down が素と食い違う」の
最小再現。モデルは読まない (合成の重み、数秒)。

## 何を比べるか

行ごとに 4 者を出す。

  ``stock``   : ``mx.quantized_matmul(..., transpose=True)`` (比較の基準)
  ``clone``   : ``qmm_wide`` を **素と同形のタイル** ``m32n32k32w2x2`` で呼んだもの。
                写しが素と同じ計算をしているかの対照。
  ``wide``    : 本番のタイル (既定 ``m64n32k32w2x2r8``)。
  ``f32``     : ``mx.dequantize`` して float32 で回した参照 (真値の代わり)。

``clone`` が ``stock`` と食い違う M があれば、犯人は**タイルではなく MLX 側の
dispatch** (M で実装が入れ替わっている)。``clone`` は一致するのに ``wide`` だけ
食い違うなら、犯人は BM=64 のタイルの張り方。どちらが真値に近いかは
``err_vs_f32`` (最大絶対誤差) で分かる。

## 判定 (最後の 1 行)

素との差そのものは判定に使わない。素は M と出力タイル数で `qmv` /
`qmm_t_splitk` / `qmm_t` に振れる (`qmm_wide.stock_bit_matches`)。見るのは 2 つ:

1. **変種どうしが相互にビット一致するか** (`tiles!=each`)。K の縮約順はタイル形に
   依らないので、割れたらタイルの張り方の欠陥。
2. 素が `qmm_t` を選ぶ帯で、素とビット一致するか。

``--tiles all`` で `TILES` を全部回す。N の端は 2 通り仕込んである
(``odd_n`` = N が BN=32 でも割り切れない、``odd_n64`` = BN=32 では割り切れるが
BN=64 では割り切れない)。BN=64 のタイルは ``N % BN`` が BK より大きいときだけ
壊れていたので、``odd_n`` (300 % 64 = 44 > 32) が要る。

    tools/biglock.sh .venv/bin/python tools/qmm_wide_shape_micro.py
    tools/biglock.sh .venv/bin/python tools/qmm_wide_shape_micro.py \
        --shapes hc_down --rows 8,64,256,2048 --out /tmp/sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 本番で当たる形。名前 -> (K, N)
SHAPES: dict[str, tuple[int, int]] = {
    # HC の読み側 (hyper_connection)。K = HC * HIDDEN = 4 * 2560
    "hc_down": (10240, 320),
    "hc_up": (320, 10240),
    # dense 射影の代表 (attention の q_proj)
    "q_proj": (2560, 12288),
    # 端の確認用: N が BN=32 で割り切れない
    "odd_n": (2560, 300),
    # 端の確認用: N が BN=32 では割り切れるが BN=64 では割り切れない
    "odd_n64": (2560, 352),
}

DEFAULT_ROWS = "8,32,64,100,128,192,250,256,320,512,768,1000,1024,1536,2048"


def _quantize(mx, N: int, K: int, dtype, group_size: int, bits: int):
    """(N, K) の重みを量子化して (packed, scales, biases) を返す。"""
    w = (mx.random.normal((N, K)) * 0.05).astype(dtype)
    wq, sc, bi = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(wq, sc, bi)
    return wq, sc, bi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--rows", default=DEFAULT_ROWS)
    ap.add_argument("--tiles", default="m32n32k32w2x2,m64n32k32w2x2r8")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import mlx.core as mx

    from mlxturbo.kernels import qmm_wide as qw

    mx.random.seed(args.seed)
    dtype = mx.bfloat16
    gs, bits = args.group_size, args.bits

    tiles = {}
    for name in (list(qw.TILES) if args.tiles == "all" else args.tiles.split(",")):
        name = name.strip()
        if not name:
            continue
        if name not in qw.TILES:
            raise SystemExit(f"tile={name!r} は qmm_wide.TILES に無い: "
                             f"{sorted(qw.TILES)}")
        tiles[name] = qw.TILES[name]

    rows = [int(r) for r in args.rows.split(",") if r.strip()]
    shapes = []
    for s in args.shapes.split(","):
        s = s.strip()
        if not s:
            continue
        if s not in SHAPES:  # "K x N" の直書きも受ける (境界の追い込み用)
            k, _, n = s.partition("x")
            SHAPES[s] = (int(k), int(n))
        shapes.append(s)

    def cmp(a, b):
        a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
        return {
            "mismatch": float(f"{(a32 != b32).astype(mx.float32).mean().item():.3g}"),
            "max_abs": float(f"{mx.abs(a32 - b32).max().item():.3g}"),
        }

    results: dict[str, dict] = {}
    for sname in shapes:
        K, N = SHAPES[sname]
        wq, sc, bi = _quantize(mx, N, K, dtype, gs, bits)
        wf = mx.dequantize(wq, sc, bi, group_size=gs, bits=bits).astype(mx.float32)
        mx.eval(wf)

        elig = {n: qw.eligible(mx.zeros((64, K), dtype=dtype), wq, sc, bi, t,
                               group_size=gs, bits=bits)
                for n, t in tiles.items()}
        rows_out = []
        names = list(tiles)
        first = names[0]
        print(f"\n=== {sname}: K={K} N={N} (eligible={elig})")
        head = (f"{'M':>6} {'stock':>8} "
                + " ".join(f"{n + '!=stock':>22}" for n in names)
                + f" {'tiles!=each':>11} {'|stock-f32|':>12} "
                + " ".join(f"{'|' + n[:6] + '-f32|':>12}" for n in names))
        print(head)
        for M in rows:
            x = (mx.random.normal((M, K)) * 0.5).astype(dtype)
            mx.eval(x)
            stock = mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                        group_size=gs, bits=bits)
            f32 = (x.astype(mx.float32) @ wf.T)
            outs = {}
            for n, t in tiles.items():
                if elig[n]:
                    outs[n] = qw.qmm_wide(x, wq, sc, bi, tile=t,
                                          group_size=gs, bits=bits)
            mx.eval(stock, f32, *outs.values())

            row = {"M": M}
            for n, o in outs.items():
                row[n] = cmp(o, stock)
                row[n]["err_vs_f32"] = float(
                    f"{mx.abs(o.astype(mx.float32) - f32).max().item():.4g}")
            row["stock_err_vs_f32"] = float(
                f"{mx.abs(stock.astype(mx.float32) - f32).max().item():.4g}")
            # タイルどうしの食い違い (これが 0 なのが自前カーネルの不変条件)
            row["tiles_disagree"] = max(
                (cmp(outs[n], outs[first])["mismatch"] for n in outs), default=0.0)
            row["stock_bit_matches"] = qw.stock_bit_matches(M, K, N)
            rows_out.append(row)

            def g(n, k):
                return row.get(n, {}).get(k, float("nan"))

            if row["stock_bit_matches"]:
                lane = "qmm_t"
            elif M < qw._STOCK_QMV_MAX_ROWS:
                lane = "qmv"
            else:
                lane = "splitk"
            print(f"{M:>6} {lane:>8} "
                  + " ".join(f"{g(n, 'mismatch'):>13.3g}/{g(n, 'max_abs'):<8.3g}"
                             for n in names)
                  + f" {row['tiles_disagree']:>11.3g}"
                  + f" {row['stock_err_vs_f32']:>12.4g} "
                  + " ".join(f"{g(n, 'err_vs_f32'):>12.4g}" for n in names))
        results[sname] = {"K": K, "N": N, "eligible": elig, "rows": rows_out}

    # 欠陥の判定は 2 つだけ。素との差そのものは判定に使わない (素が M で
    # 実装を替えるから -- qmm_wide.py の「数値」の節)
    bad = []
    for sname, r in results.items():
        for row in r["rows"]:
            if row["tiles_disagree"]:
                bad.append(f"{sname} M={row['M']}: タイルどうしが割れた "
                           f"({row['tiles_disagree']})")
            if row["stock_bit_matches"]:
                for n in row:
                    if isinstance(row[n], dict) and row[n]["mismatch"]:
                        bad.append(f"{sname} M={row['M']} {n}: 素が qmm_t を"
                                   f"選ぶ帯なのにビット一致しない "
                                   f"({row[n]['mismatch']})")
    print("\n" + ("FAIL:\n  " + "\n  ".join(bad) if bad
                  else "OK: 全変種が相互にビット一致、qmm_t 帯では素とも一致"))
    results["verdict"] = {"ok": not bad, "problems": bad}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
