"""`kernels/qmm_wide.py` の形の契約 (M x K x N の端まで) を GPU で確かめる。

合成の重みだけで数秒。`tools/qmm_wide_shape_micro.py` の掃引を、走行のたびに
効く不変条件 3 つに絞ったもの。

素 (`mx.quantized_matmul`) との**ビット一致は、素が `qmm_t` を選ぶ帯でだけ**
成り立つ。MLX 0.32.2 は同じ呼び出しを M と出力タイル数で `qmv` /
`qmm_t_splitk` / `qmm_t` に振り分ける (`qmm_wide.stock_bit_matches`)。
splitk 帯では素の側が bf16 の部分和を経由して真値から離れるので、そこでは
「素と一致するか」ではなく「**変種どうしが一致し、float32 参照に対して素より
悪くない**」を見る。

    uv run pytest bench/test_qmm_wide_shapes.py -q
    tools/biglock.sh .venv/bin/python -m pytest bench/test_qmm_wide_shapes.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from mlxturbo.kernels import qmm_wide as qw

GROUP_SIZE = 64
BITS = 4

# (K, N)。HC の細長い 2 本と、N が BN=32 で割り切れない形
SHAPES = ((10240, 320), (320, 10240), (2560, 300))
# BM=32 / 64 / 128 のどれでも端数タイルが出る行数を混ぜる
ROWS = (13, 32, 100, 250, 1000)


def _weights(K, N, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    w = (mx.random.normal((N, K)) * 0.05).astype(dtype)
    wq, sc, bi = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    ref = mx.dequantize(wq, sc, bi, group_size=GROUP_SIZE,
                        bits=BITS).astype(mx.float32)
    mx.eval(wq, sc, bi, ref)
    return wq, sc, bi, ref


def _x(M, K, dtype=mx.bfloat16):
    x = (mx.random.normal((M, K)) * 0.5).astype(dtype)
    mx.eval(x)
    return x


def _call(x, wq, sc, bi, tile):
    return qw.qmm_wide(x, wq, sc, bi, tile=tile,
                       group_size=GROUP_SIZE, bits=BITS)


def test_stock_lane_predicate_boundaries():
    """`stock_bit_matches` が実測した境界 (M3 Max / MLX 0.32.2) を写している。"""
    # M < 13 は qmv 系 (N が大きくタイルが足りていても外れる)
    assert not qw.stock_bit_matches(12, 320, 10240)
    assert qw.stock_bit_matches(13, 320, 10240)
    # 出力タイル数 ceil(M/32) * ceil(N/32) が 256 未満なら splitk
    assert not qw.stock_bit_matches(800, 10240, 320)   # 25 * 10 = 250
    assert qw.stock_bit_matches(832, 10240, 320)       # 26 * 10 = 260
    assert not qw.stock_bit_matches(384, 10240, 640)   # 12 * 20 = 240
    assert qw.stock_bit_matches(400, 10240, 640)       # 13 * 20 = 260
    assert not qw.stock_bit_matches(192, 10240, 1280)  # 6 * 40 = 240
    assert qw.stock_bit_matches(193, 10240, 1280)      # 7 * 40 = 280
    # K が小さいと分割しない
    assert qw.stock_bit_matches(256, 64, 320)
    assert not qw.stock_bit_matches(256, 128, 320)


def test_all_tiles_agree_with_each_other():
    """K の縮約順はタイル形に依らない -- 全変種が相互にビット一致する。

    これが自前カーネル側の不変条件。割れたらタイルの張り方の欠陥。
    """
    for K, N in SHAPES:
        wq, sc, bi, _ = _weights(K, N)
        tiles = [t for t in qw.TILES.values()
                 if qw.eligible(mx.zeros((64, K), dtype=mx.bfloat16),
                                wq, sc, bi, t, GROUP_SIZE, BITS)]
        assert len(tiles) >= 2, (K, N)
        for M in ROWS:
            x = _x(M, K)
            outs = [_call(x, wq, sc, bi, t) for t in tiles]
            mx.eval(outs)
            for t, o in zip(tiles[1:], outs[1:]):
                assert mx.array_equal(o, outs[0]).item(), (
                    f"K={K} N={N} M={M}: {t.name} が {tiles[0].name} と割れた")


def test_bit_identical_to_stock_in_qmm_t_band():
    """素が `qmm_t` を選ぶ帯では `mx.quantized_matmul` とビット一致する。"""
    seen = 0
    for K, N in SHAPES:
        wq, sc, bi, _ = _weights(K, N)
        for M in ROWS:
            if not qw.stock_bit_matches(M, K, N):
                continue
            seen += 1
            x = _x(M, K)
            stock = mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                        group_size=GROUP_SIZE, bits=BITS)
            for name in ("m32n32k32w2x2", "m64n32k32w2x2r8"):
                out = _call(x, wq, sc, bi, qw.TILES[name])
                mx.eval(stock, out)
                assert mx.array_equal(out, stock).item(), (
                    f"K={K} N={N} M={M} {name}: 素と一致しない")
    assert seen >= 5


def test_splitk_band_is_no_worse_than_stock():
    """素が splitk / qmv に振れる帯では、写しのほうが真値に近い (少なくとも同等)。

    ここが「M < 1024 で down が素と食い違う」の正体。素の側が bf16 の部分和を
    経由するので離れる。写しは `qmm_t` の答えそのままなので離れない。
    """
    seen = 0
    for K, N in SHAPES:
        wq, sc, bi, ref = _weights(K, N)
        for M in ROWS:
            if qw.stock_bit_matches(M, K, N):
                continue
            seen += 1
            x = _x(M, K)
            exact = x.astype(mx.float32) @ ref.T
            stock = mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                        group_size=GROUP_SIZE, bits=BITS)
            out = _call(x, wq, sc, bi, qw.TILES["m64n32k32w2x2r8"])
            mx.eval(exact, stock, out)
            e_stock = mx.abs(stock.astype(mx.float32) - exact).max().item()
            e_wide = mx.abs(out.astype(mx.float32) - exact).max().item()
            assert e_wide <= e_stock * 1.01 + 1e-6, (
                f"K={K} N={N} M={M}: wide {e_wide} > stock {e_stock}")
    assert seen >= 5


if __name__ == "__main__":
    test_stock_lane_predicate_boundaries()
    test_all_tiles_agree_with_each_other()
    test_bit_identical_to_stock_in_qmm_t_band()
    test_splitk_band_is_no_worse_than_stock()
    print("ok")
