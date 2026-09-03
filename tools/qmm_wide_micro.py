"""P10 (広タイル qmm) のマイクロベンチ。

`mlxturbo/kernels/qmm_wide.py` の変種を `mx.quantized_matmul(transpose=True)`
と **1 プロセス内で ABBA 交互**に比べる。形は本番の dense 射影の実寸
(`~/models/ddalcu-trunk-mlxturbo/config.json` から):

  hidden 2560 / head_dim 256 / n_heads 24 / n_kv 2
  linear: key 128x16=2048、value 128x48=6144、conv_dim 10240

判定線 (2026-09-03 に較正): 形 `attn_q` と `proj_out` の M=2048 で素の
**0.90 倍以下**。当初の 0.87 は `tools/dequant_gemm_micro.py` の実測
(素の qmm がこの形で 11.5〜11.7 TFLOPS、bf16 常駐 `mx.matmul` が 12.7〜13.0
TFLOPS = 0.89〜0.92 倍) と突き合わせると 13.3 TFLOPS 超 = **bf16 の steel
gemm より速い**ことを要求してしまうので、bf16 常駐と同等以上 (0.90) に
下げた。ここに届けば P8 (層ごと dequant + bf16 matmul、5 GB/チャンクの
書き戻しと一時バッファ付き) を置き換える価値がある。届かなければ畳む。

比較の 3 本目として **bf16 常駐 `mx.matmul`** (`mx.dequantize` を計測外で
1 回だけ済ませた重み、`dequant_gemm_micro.py` の `bf16_resident` と同じ)
を形ごとに 1 行出す。P10 の変種はこれを超えて初めて意味がある。

温キャッシュのマイクロの絶対値は信じないこと (CLAUDE.md)。ここで見るのは
変種どうしの順位と、素に対する比だけ。採否は in-model A/B で決める。

使い方 (GPU を使うので必ず biglock 経由):

    tools/biglock.sh uv run python tools/qmm_wide_micro.py \
        --shapes attn_q,proj_out --m 2048 --json bench/results/qmm-wide-sweep.json

    tools/biglock.sh uv run python tools/qmm_wide_micro.py \
        --tiles m64n32k32w2x2,m64n64k32w2x4 --m 2048,8192
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable

import mlx.core as mx

from mlxturbo.kernels.qmm_wide import (
    BITS,
    GROUP_SIZE,
    STOCK,
    TILES,
    Tile,
    qmm_wide,
    tile_ok,
)

# (名前, K, N)。N はゲート込みの実寸
SHAPES: dict[str, tuple[int, int]] = {
    # (a) GDN / attention の入力射影
    "attn_q": (2560, 12288),        # q_proj (n_heads*head_dim*2 = 出力ゲート込み)
    "attn_qkv": (2560, 13312),      # 連結 qkv (12288 + 512 + 512)
    "gdn_qkv": (2560, 10240),       # in_proj_qkv (conv_dim)
    "gdn_wide": (2560, 16480),      # 連結 in_proj (10240 + 6144 + 48 + 48)
    # (b) 出力射影 (attention の o_proj と GDN の out_proj が同じ形)
    "proj_out": (6144, 2560),
    # (c) MoE の形 (参考。本番は gather_qmm なので直接は当たらない)
    "moe_gate_up": (2560, 640),
    "moe_down": (640, 2560),
}

DEFAULT_SHAPES = "attn_q,gdn_wide,proj_out,moe_gate_up,moe_down"


def _timed(fn: Callable[[], mx.array]) -> float:
    """1 回の呼びの壁時計 (ms)。"""
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    return (time.perf_counter() - t0) * 1e3


def _abba(
    stock: Callable[[], mx.array],
    variant: Callable[[], mx.array],
    rounds: int,
    warmup: int,
) -> tuple[list[float], list[float]]:
    """A B B A の順に交互に測る (熱とキャッシュを揃える)。"""
    for _ in range(warmup):
        mx.eval(stock())
        mx.eval(variant())

    a: list[float] = []
    b: list[float] = []
    for _ in range(rounds):
        a.append(_timed(stock))
        b.append(_timed(variant))
        b.append(_timed(variant))
        a.append(_timed(stock))
    return a, b


def _numerics(ref: mx.array, got: mx.array) -> dict:
    """ビット一致か、違うなら max|diff| と相対 RMS。"""
    n_diff = int(mx.sum((ref != got).astype(mx.int32)).item())
    if n_diff == 0:
        return {"bit_identical": True, "n_diff": 0, "max_abs": 0.0, "rel_rms": 0.0}
    r = ref.astype(mx.float32)
    g = got.astype(mx.float32)
    d = g - r
    return {
        "bit_identical": False,
        "n_diff": n_diff,
        "max_abs": float(mx.max(mx.abs(d)).item()),
        "rel_rms": float(
            (mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(r * r))).item()
        ),
    }


def _tflops(m: int, k: int, n: int, ms: float) -> float:
    return 2.0 * m * k * n / (ms * 1e-3) / 1e12


def run(
    shapes: list[str],
    tiles: list[Tile],
    ms_list: list[int],
    rounds: int,
    warmup: int,
    seed: int,
    with_bf16: bool = True,
) -> list[dict]:
    results: list[dict] = []
    for shape_name in shapes:
        K, N = SHAPES[shape_name]
        # 重みは形ごとに 1 回だけ作る (変種のあいだで同じものを使う)
        mx.random.seed(seed)
        w_fp = mx.random.normal((N, K)).astype(mx.bfloat16)
        wq, scales, biases = mx.quantize(w_fp, group_size=GROUP_SIZE, bits=BITS)
        mx.eval(wq, scales, biases)
        del w_fp

        for M in ms_list:
            mx.random.seed(seed + 1)
            x = mx.random.normal((M, K)).astype(mx.bfloat16)
            mx.eval(x)

            def stock() -> mx.array:
                return mx.quantized_matmul(
                    x, wq, scales, biases, transpose=True,
                    group_size=GROUP_SIZE, bits=BITS,
                )

            y_ref = stock()
            mx.eval(y_ref)

            # 3 本目の基準: bf16 常駐 matmul (P8 の上限)。逆量子化は計測外で
            # 1 回だけ。タイル掃引の前に済ませて重みを捨てる (84 MB の常駐が
            # 変種の計測のキャッシュ当たりを変えないように)
            if with_bf16:
                w_bf = mx.dequantize(
                    wq, scales, biases, group_size=GROUP_SIZE, bits=BITS)
                mx.eval(w_bf)

                def bf16() -> mx.array:
                    return mx.matmul(x, w_bf.T)

                y_bf = bf16()
                mx.eval(y_bf)
                num_bf = _numerics(y_ref, y_bf)
                del y_bf
                a, b = _abba(stock, bf16, rounds, warmup)
                a_ms, b_ms = statistics.median(a), statistics.median(b)
                results.append({
                    "shape": shape_name, "M": M, "K": K, "N": N,
                    "tile": "bf16_resident",
                    "stock_ms": a_ms, "variant_ms": b_ms,
                    "ratio": b_ms / a_ms,
                    "stock_tflops": _tflops(M, K, N, a_ms),
                    "variant_tflops": _tflops(M, K, N, b_ms),
                    "stock_all": a, "variant_all": b,
                    **num_bf,
                })
                print(
                    f"  {shape_name:11s} M={M:5d} {'bf16_resident':20s} "
                    f"素 {a_ms:8.3f} ms ({_tflops(M, K, N, a_ms):5.2f} TF)  "
                    f"bf16 {b_ms:8.3f} ms ({_tflops(M, K, N, b_ms):5.2f} TF)  "
                    f"比 {b_ms / a_ms:.3f}  "
                    f"max|d| {num_bf['max_abs']:.3e} relRMS {num_bf['rel_rms']:.2e}"
                )
                del w_bf
                clear = getattr(mx, "clear_cache", None) or mx.metal.clear_cache
                clear()

            for tile in tiles:
                why = tile_ok(tile, K, N, GROUP_SIZE, BITS)
                if why is not None:
                    print(f"  {shape_name} M={M} {tile.name}: 不適格 ({why})")
                    results.append({
                        "shape": shape_name, "M": M, "K": K, "N": N,
                        "tile": tile.name, "skipped": why,
                    })
                    continue

                def variant(_t: Tile = tile) -> mx.array:
                    return qmm_wide(x, wq, scales, biases, tile=_t)

                try:
                    y_got = variant()
                    mx.eval(y_got)
                except Exception as exc:  # コンパイル失敗 (レジスタ/tgp メモリ)
                    print(f"  {shape_name} M={M} {tile.name}: 失敗 {exc}")
                    results.append({
                        "shape": shape_name, "M": M, "K": K, "N": N,
                        "tile": tile.name, "error": str(exc)[:400],
                    })
                    continue

                num = _numerics(y_ref, y_got)
                del y_got

                a, b = _abba(stock, variant, rounds, warmup)
                a_ms = statistics.median(a)
                b_ms = statistics.median(b)
                row = {
                    "shape": shape_name, "M": M, "K": K, "N": N,
                    "tile": tile.name,
                    "threads": tile.threads,
                    "tgp_bytes": tile.tgp_bytes,
                    "accum_regs": tile.accum_regs,
                    "stock_ms": a_ms,
                    "variant_ms": b_ms,
                    "ratio": b_ms / a_ms,
                    "stock_tflops": _tflops(M, K, N, a_ms),
                    "variant_tflops": _tflops(M, K, N, b_ms),
                    "stock_all": a,
                    "variant_all": b,
                    **num,
                }
                results.append(row)
                print(
                    f"  {shape_name:11s} M={M:5d} {tile.name:20s} "
                    f"素 {a_ms:8.3f} ms ({row['stock_tflops']:5.2f} TF)  "
                    f"変種 {b_ms:8.3f} ms ({row['variant_tflops']:5.2f} TF)  "
                    f"比 {row['ratio']:.3f}  "
                    + ("ビット一致" if num["bit_identical"]
                       else f"max|d| {num['max_abs']:.3e} relRMS {num['rel_rms']:.2e}")
                )
            del x, y_ref
        del wq, scales, biases
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=DEFAULT_SHAPES,
                    help=f"カンマ区切り。既定 {DEFAULT_SHAPES}。all で全部")
    ap.add_argument("--tiles", default="all",
                    help="カンマ区切りのタイル名。既定 all")
    ap.add_argument("--m", default="2048", help="カンマ区切りの M。既定 2048")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-bf16", action="store_true",
                    help="bf16 常駐 matmul の基準行を出さない")
    args = ap.parse_args()

    shapes = list(SHAPES) if args.shapes == "all" else args.shapes.split(",")
    for s in shapes:
        if s not in SHAPES:
            raise SystemExit(f"知らない形: {s} (候補 {list(SHAPES)})")

    if args.tiles == "all":
        tiles = list(TILES.values())
    else:
        tiles = []
        for name in args.tiles.split(","):
            if name not in TILES:
                raise SystemExit(f"知らないタイル: {name} (候補 {list(TILES)})")
            tiles.append(TILES[name])
        if STOCK not in tiles:
            tiles.insert(0, STOCK)  # 写しが素と一致する基準線は毎回入れる

    ms_list = [int(v) for v in args.m.split(",")]

    info = getattr(mx, "device_info", None) or mx.metal.device_info
    print(f"device: {info().get('architecture')}  "
          f"タイル {len(tiles)} 種 x 形 {len(shapes)} 種 x M {ms_list}")
    print("ABBA x %d (warmup %d)、中央値で比較" % (args.rounds, args.warmup))

    results = run(shapes, tiles, ms_list, args.rounds, args.warmup, args.seed,
                  with_bf16=not args.no_bf16)

    # 形 x M ごとの最良。判定線は 0.90 (bf16 常駐と同等以上、2026-09-03 較正)
    print("\n== 形 x M ごとの最良 (素比。判定線 0.90) ==")
    keys = sorted({(r["shape"], r["M"]) for r in results if "ratio" in r})
    for shape_name, M in keys:
        rows = [r for r in results
                if r.get("shape") == shape_name and r.get("M") == M
                and "ratio" in r]
        bf16 = next((r for r in rows if r["tile"] == "bf16_resident"), None)
        rows = [r for r in rows if r["tile"] != "bf16_resident"]
        if not rows:
            continue
        best = min(rows, key=lambda r: r["ratio"])
        verdict = "通過" if best["ratio"] <= 0.90 else "未達"
        if bf16 is not None:
            verdict += (
                f"、bf16 常駐 {bf16['ratio']:.3f} ({bf16['variant_tflops']:5.2f} TF) "
                + ("に勝ち" if best["ratio"] < bf16["ratio"] else "に負け")
            )
        print(f"  {shape_name:11s} M={M:5d}  最良 {best['tile']:20s} "
              f"比 {best['ratio']:.3f} ({best['variant_tflops']:5.2f} TF) "
              f"[{verdict}]")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "device": str(info().get("architecture")),
                "rounds": args.rounds,
                "warmup": args.warmup,
                "pass_ratio": 0.90,
                "results": results,
            }, fh, indent=2, ensure_ascii=False)
        print(f"\n書いた: {args.json}")


if __name__ == "__main__":
    main()
