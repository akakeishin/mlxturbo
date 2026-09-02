"""MoE grouped GEMM レーン (P3) の proof-of-life。

## 第 1 段 (`--stage dense`)

`mlxturbo/kernels/_steel_flat.py` に平坦化した steel が、素の
`mx.quantized_matmul(transpose=True)` と

  - **ビット一致する** (`mx.array_equal`。近似では不可)
  - **1.10 倍以内の時間で回る** (同一プロセス内 ABBA、ウォームアップ後の中央値)

かを見る。どちらかが落ちたら `mx.fast.metal_kernel` 経路の固定費で勝ち目が
無いということなので、セグメント対応 (第 2 段) には進まない
(反転条件は `docs/research/IDEAS-2026-09-03.md` の P3)。

形は Flash-Next の MoE そのもの (512 専門家 x 2560 -> 640):

  - gate/up: M=20480 (2048 tok x top-k 10), K=2560, N=640
  - down   : M=20480,                        K=640,  N=2560

## 使い方 (GPU。必ず lock 経由で)

    tools/biglock.sh .venv/bin/python tools/moe_grouped_gemm_micro.py \
        --stage dense --json bench/results/moe-grouped-gemm-dense.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable

import mlx.core as mx

from mlxturbo.kernels.moe_grouped_gemm import (
    BITS,
    ENV_KNOB,
    GROUP_SIZE,
    dense_eligible,
    enabled,
    is_nax_device,
    qmm_dense_clone,
)

# (名前, M, K, N)。M は 2048 tok x top-k 10 の行数
SHAPES = [
    ("gate/up", 20480, 2560, 640),
    ("down", 20480, 640, 2560),
]


def _timed(fn: Callable[[], mx.array]) -> float:
    """1 回の呼びの壁時計 (ms)。"""
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    return (time.perf_counter() - t0) * 1e3


def _abba(
    stock: Callable[[], mx.array],
    clone: Callable[[], mx.array],
    rounds: int,
    warmup: int,
) -> tuple[list[float], list[float]]:
    """1 プロセス内で A B B A の順に交互に測る (熱とキャッシュを揃える)。"""
    for _ in range(warmup):
        mx.eval(stock())
        mx.eval(clone())

    a: list[float] = []
    b: list[float] = []
    for _ in range(rounds):
        a.append(_timed(stock))
        b.append(_timed(clone))
        b.append(_timed(clone))
        a.append(_timed(stock))
    return a, b


def _first_diff(ref: mx.array, got: mx.array) -> dict:
    """最初に食い違う要素の位置と値 (ビット一致しなかったときの手掛かり)。"""
    ne = ref != got
    flat = mx.flatten(ne)
    idx = int(mx.argmax(flat.astype(mx.int32)).item())
    row, col = divmod(idx, ref.shape[1])
    return {
        "flat_index": idx,
        "row": row,
        "col": col,
        "ref": float(ref[row, col].item()),
        "got": float(got[row, col].item()),
        "n_diff": int(mx.sum(ne.astype(mx.int32)).item()),
    }


def run_dense(rounds: int, warmup: int, seed: int) -> list[dict]:
    results = []
    for name, M, K, N in SHAPES:
        mx.random.seed(seed)
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        w_fp = mx.random.normal((N, K)).astype(mx.bfloat16)
        wq, scales, biases = mx.quantize(w_fp, group_size=GROUP_SIZE, bits=BITS)
        mx.eval(x, wq, scales, biases)
        del w_fp

        assert dense_eligible(x, wq, scales, biases), f"{name} の形が適格でない"

        def stock() -> mx.array:
            return mx.quantized_matmul(
                x,
                wq,
                scales,
                biases,
                transpose=True,
                group_size=GROUP_SIZE,
                bits=BITS,
            )

        def clone() -> mx.array:
            return qmm_dense_clone(x, wq, scales, biases)

        y_ref = stock()
        y_got = clone()
        mx.eval(y_ref, y_got)
        exact = bool(mx.array_equal(y_ref, y_got).item())
        max_diff = float(
            mx.max(mx.abs(y_ref.astype(mx.float32) - y_got.astype(mx.float32))).item()
        )
        entry: dict = {
            "shape": name,
            "M": M,
            "K": K,
            "N": N,
            "bit_exact": exact,
            "max_abs_diff": max_diff,
        }
        if not exact:
            entry["first_diff"] = _first_diff(y_ref, y_got)
        del y_ref, y_got

        a, b = _abba(stock, clone, rounds, warmup)
        stock_ms = statistics.median(a)
        clone_ms = statistics.median(b)
        entry.update(
            {
                "stock_ms": stock_ms,
                "clone_ms": clone_ms,
                "ratio": clone_ms / stock_ms,
                "stock_samples": len(a),
                "clone_samples": len(b),
            }
        )
        results.append(entry)

        print(
            f"{name:8s} M={M} K={K} N={N}  "
            f"stock {stock_ms:7.3f} ms  clone {clone_ms:7.3f} ms  "
            f"比 {clone_ms / stock_ms:5.3f}  "
            f"ビット一致 {'yes' if exact else 'NO'}  max|diff| {max_diff:g}"
        )
        if not exact:
            print(f"         最初の差: {entry['first_diff']}")

        del x, wq, scales, biases
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dense"], default="dense")
    ap.add_argument("--rounds", type=int, default=10, help="ABBA の回数 (A/B 各 20 本)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    info = (getattr(mx, "device_info", None) or mx.metal.device_info)()
    print(
        f"device: {info.get('device_name')} "
        f"arch={info.get('architecture')} "
        f"nax={is_nax_device()} "
        f"{ENV_KNOB}={'on' if enabled() else 'off'}"
    )

    results = run_dense(args.rounds, args.warmup, args.seed)

    if args.json:
        payload = {
            "stage": args.stage,
            "device": {k: str(v) for k, v in info.items()},
            "nax": is_nax_device(),
            "knob_on": enabled(),
            "rounds": args.rounds,
            "results": results,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
