"""`mlxturbo/kernels/qmv_small_m.py` の正しさを固定する (合成、GPU、数秒)。

守りたい性質は 1 つだけ:

    qmv_small_m(x[0:M])[v] == mx.quantized_matmul(x[v:v+1])   (v = 0..M-1)

**行ごとに、幅 1 で素を呼んだ結果とビット一致**すること。これが立つと、投機
デコードの検証幅 S が round ごとに変わっても 1 行の答えが動かない = 幅で丸めが
変わって生成列が分岐することがない。

一致の要は `load_vector` の写しで、mlx は

    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];

を **T (bf16/fp16) のまま**足してから float の `sum` に足している。ここを
float に上げてから足すと `bias * sum` の項が 1 ulp ずれ、出力の 3〜6% の要素が
割れる (2026-09-04 実測)。下の `test_rows_bit_match_stock_m1` がその見張り。

対応範囲の外 (K が 512 の倍数でない / group_size != 64 / bits != 4 / 3 次元 /
M > 8) では `mx.quantized_matmul` に委譲する。委譲は「素そのもの」なので
こちらもビット一致で検査する。
"""

import pytest

mx = pytest.importorskip("mlx.core")

from mlxturbo.kernels.qmv_small_m import (  # noqa: E402
    M_MAX,
    eligible,
    qmv_small_m,
)

GROUP_SIZE = 64
BITS = 4

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal GPU が要る"
)


def _weights(k: int, n: int, dtype, group_size: int = GROUP_SIZE):
    """packed な 4bit 重み (dense を作らずに済ませる)。"""
    w = mx.random.randint(0, 2**31, shape=(n, k // 8), dtype=mx.uint32)
    scales = (mx.random.uniform(shape=(n, k // group_size)) * 0.02).astype(dtype)
    biases = (
        mx.random.uniform(shape=(n, k // group_size)) * 0.01 - 0.005
    ).astype(dtype)
    mx.eval(w, scales, biases)
    return w, scales, biases


def _stock(x, w, s, b, group_size: int = GROUP_SIZE, bits: int = BITS):
    return mx.quantized_matmul(
        x, w, s, b, transpose=True, group_size=group_size, bits=bits
    )


def _rowwise_ref(x, w, s, b):
    """各行を幅 1 で素に流したもの。"""
    return mx.concatenate(
        [_stock(x[v : v + 1], w, s, b) for v in range(x.shape[0])], axis=0
    )


# ---------------------------------------------------------------- 一致

@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("M", list(range(1, M_MAX + 1)))
def test_rows_bit_match_stock_m1(M, dtype):
    """M=1..8 の各行が、幅 1 で素を呼んだ結果とビット一致する。"""
    k, n = 1024, 1024
    w, s, b = _weights(k, n, dtype)
    x = (mx.random.normal((M, k)) * 0.1).astype(dtype)
    mx.eval(x)
    assert eligible(x, w, s, b, GROUP_SIZE, BITS)
    got = qmv_small_m(x, w, s, b, group_size=GROUP_SIZE, bits=BITS)
    ref = _rowwise_ref(x, w, s, b)
    mx.eval(got, ref)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert mx.array_equal(got, ref), (
        f"M={M} dtype={dtype}: 行ごとの幅 1 とビット一致しない "
        f"({int(mx.sum(got != ref).item())}/{got.size} 要素)"
    )


@pytest.mark.parametrize("nsg", [1, 2, 4, 8])
def test_nsg_does_not_change_values(nsg):
    """threadgroup あたりの simdgroup 数は答えを変えない (行の縮約順に無関係)。"""
    k, n = 1024, 2048
    dtype = mx.bfloat16
    w, s, b = _weights(k, n, dtype)
    x = (mx.random.normal((4, k)) * 0.1).astype(dtype)
    mx.eval(x)
    got = qmv_small_m(x, w, s, b, group_size=GROUP_SIZE, bits=BITS, nsg=nsg)
    ref = _rowwise_ref(x, w, s, b)
    mx.eval(got, ref)
    assert mx.array_equal(got, ref), f"nsg={nsg} で割れた"


@pytest.mark.parametrize("n", [1032, 2056, 4104])
def test_n_edge_rows(n):
    """N が threadgroup の出力行数 (nsg x rps) の倍数でなくても端の行まで正しい。

    ここは nsg=2 / rps=4 の既定で 8 行ずつ配るが、n はどれも 8 の倍数で
    16 / 32 の倍数ではない形を選んである。
    """
    k, dtype = 1024, mx.bfloat16
    w, s, b = _weights(k, n, dtype)
    x = (mx.random.normal((3, k)) * 0.1).astype(dtype)
    mx.eval(x)
    assert eligible(x, w, s, b, GROUP_SIZE, BITS)
    got = qmv_small_m(x, w, s, b, group_size=GROUP_SIZE, bits=BITS)
    ref = _rowwise_ref(x, w, s, b)
    mx.eval(got, ref)
    assert got.shape == (3, n)
    assert mx.array_equal(got, ref)


@pytest.mark.parametrize("n", [1023, 1025, 1030])
def test_n_not_multiple_of_eight_delegates(n):
    """N が 8 の倍数でないと mlx が端の行を別扱いにする -> 当てない。

    N=1025 では素の側の端 1 行だけが 1 ulp ずれる (3075 要素中 1 つ、
    max|d| 7.6e-6、2026-09-04)。本番の N は全部 8 の倍数。
    """
    k, dtype = 1024, mx.bfloat16
    w, s, b = _weights(k, n, dtype)
    x = (mx.random.normal((3, k)) * 0.1).astype(dtype)
    mx.eval(x)
    assert not eligible(x, w, s, b, GROUP_SIZE, BITS)
    got = qmv_small_m(x, w, s, b, group_size=GROUP_SIZE, bits=BITS)
    ref = _stock(x, w, s, b)
    mx.eval(got, ref)
    assert mx.array_equal(got, ref)


def test_production_shape_bit_match():
    """本番の形 (27B の GDN in_proj_z) でも一致する。"""
    k, n, dtype = 5120, 6144, mx.bfloat16
    w, s, b = _weights(k, n, dtype)
    for M in (1, 3, 4, 5, 8):
        x = (mx.random.normal((M, k)) * 0.1).astype(dtype)
        mx.eval(x)
        got = qmv_small_m(x, w, s, b, group_size=GROUP_SIZE, bits=BITS)
        ref = _rowwise_ref(x, w, s, b)
        mx.eval(got, ref)
        assert mx.array_equal(got, ref), f"M={M} で割れた"


# ---------------------------------------------------------------- 委譲

def test_k_not_block_multiple_delegates():
    """K が 512 の倍数でないと mlx 側が qmv_fast を選ばない -> 委譲。"""
    k, n, dtype = 1024 + 64, 512, mx.bfloat16  # K が 512 の倍数でない
    w, s, b = _weights(k, n, dtype)
    x = (mx.random.normal((4, k)) * 0.1).astype(dtype)
    mx.eval(x)
    assert not eligible(x, w, s, b, GROUP_SIZE, BITS)
    got = qmv_small_m(x, w, s, b, group_size=GROUP_SIZE, bits=BITS)
    ref = _stock(x, w, s, b)
    mx.eval(got, ref)
    assert mx.array_equal(got, ref)


def test_group_size_128_delegates():
    k, n, dtype = 1024, 512, mx.bfloat16
    w, s, b = _weights(k, n, dtype, group_size=128)
    x = (mx.random.normal((4, k)) * 0.1).astype(dtype)
    mx.eval(x)
    assert not eligible(x, w, s, b, 128, BITS)
    got = qmv_small_m(x, w, s, b, group_size=128, bits=BITS)
    ref = _stock(x, w, s, b, group_size=128)
    mx.eval(got, ref)
    assert mx.array_equal(got, ref)


def test_bits8_delegates():
    """bits=8 は mlx 側の qdot の形が違うので当てない。"""
    k, n, dtype = 1024, 512, mx.bfloat16
    dense = (mx.random.normal((n, k)) * 0.05).astype(dtype)
    w, s, b = mx.quantize(dense, group_size=GROUP_SIZE, bits=8)
    mx.eval(w, s, b)
    x = (mx.random.normal((4, k)) * 0.1).astype(dtype)
    mx.eval(x)
    assert not eligible(x, w, s, b, GROUP_SIZE, 8)
    got = qmv_small_m(x, w, s, b, group_size=GROUP_SIZE, bits=8)
    ref = _stock(x, w, s, b, bits=8)
    mx.eval(got, ref)
    assert mx.array_equal(got, ref)


@pytest.mark.parametrize("M", [0, M_MAX + 1, 16])
def test_m_outside_window_delegates(M):
    k, n, dtype = 1024, 512, mx.bfloat16
    w, s, b = _weights(k, n, dtype)
    x = (mx.random.normal((max(M, 1), k)) * 0.1).astype(dtype)
    if M == 0:
        x = x[:0]
    mx.eval(x)
    assert not eligible(x, w, s, b, GROUP_SIZE, BITS)
    got = qmv_small_m(x, w, s, b, group_size=GROUP_SIZE, bits=BITS)
    ref = _stock(x, w, s, b)
    mx.eval(got, ref)
    assert mx.array_equal(got, ref)


def test_3d_input_delegates():
    """3 次元 (B, S, K) は委譲する (平坦化は呼び手 = dispatch の役目)。"""
    k, n, dtype = 1024, 512, mx.bfloat16
    w, s, b = _weights(k, n, dtype)
    x = (mx.random.normal((2, 2, k)) * 0.1).astype(dtype)
    mx.eval(x)
    assert not eligible(x, w, s, b, GROUP_SIZE, BITS)
    got = qmv_small_m(x, w, s, b, group_size=GROUP_SIZE, bits=BITS)
    ref = _stock(x, w, s, b)
    mx.eval(got, ref)
    assert mx.array_equal(got, ref)


def test_float32_activation_delegates():
    k, n = 1024, 512
    w, s, b = _weights(k, n, mx.bfloat16)
    x = (mx.random.normal((4, k)) * 0.1).astype(mx.float32)
    mx.eval(x)
    assert not eligible(x, w, s, b, GROUP_SIZE, BITS)


# ---------------------------------------------------------------- 配線

def test_dispatch_route_small_m_bit_matches():
    """`kernels/dispatch.py` の小 M 経路に載せても行ごとの一致が保たれる。

    3 次元 (B, S, K) の入力 (= 本番の検証フォワードの形) を平坦化して通す。
    """
    import os

    from mlxturbo.kernels import dispatch as D

    k, n, dtype = 5120, 6144, mx.bfloat16
    w, s, b = _weights(k, n, dtype)
    x = (mx.random.normal((1, 4, k)) * 0.1).astype(dtype)
    mx.eval(x)
    ref = _rowwise_ref(x.reshape(4, k), w, s, b).reshape(1, 4, n)
    prev = os.environ.get("MLXTURBO_SMALL_M_ROUTE")
    os.environ["MLXTURBO_SMALL_M_ROUTE"] = "small_m"
    try:
        with D.dispatch_scope():
            got = D.quantized_matmul(x, w, s, b, group_size=GROUP_SIZE, bits=BITS)
        mx.eval(got, ref)
        assert got.shape == (1, 4, n)
        assert mx.array_equal(got, ref)
    finally:
        if prev is None:
            os.environ.pop("MLXTURBO_SMALL_M_ROUTE", None)
        else:
            os.environ["MLXTURBO_SMALL_M_ROUTE"] = prev
        D.refresh_small_m_route()


def test_dispatch_default_is_auto():
    """既定 (env 無し) は auto: 非 NAX 機なら small_m、NAX 機なら素のまま。
    明示の off なら機種に関わらず素のまま (経路表が動かない)。"""
    import os

    from mlxturbo.kernels import dispatch as D
    from mlxturbo.kernels.moe_grouped_gemm import is_nax_device

    prev = os.environ.pop("MLXTURBO_SMALL_M_ROUTE", None)
    try:
        assert D.refresh_small_m_route() == (None if is_nax_device() else D.SMALLM)
        os.environ["MLXTURBO_SMALL_M_ROUTE"] = "auto"
        assert D.refresh_small_m_route() == (None if is_nax_device() else D.SMALLM)

        os.environ["MLXTURBO_SMALL_M_ROUTE"] = "off"
        assert D.refresh_small_m_route() is None
        k, n, dtype = 5120, 6144, mx.bfloat16
        w, s, b = _weights(k, n, dtype)
        x = (mx.random.normal((4, k)) * 0.1).astype(dtype)
        mx.eval(x)
        with D.dispatch_scope():
            got = D.quantized_matmul(x, w, s, b, group_size=GROUP_SIZE, bits=BITS)
        ref = _stock(x, w, s, b)
        mx.eval(got, ref)
        assert mx.array_equal(got, ref)
    finally:
        if prev is None:
            os.environ.pop("MLXTURBO_SMALL_M_ROUTE", None)
        else:
            os.environ["MLXTURBO_SMALL_M_ROUTE"] = prev
        D.refresh_small_m_route()


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("applegpu_g13s", False),  # M1
        ("applegpu_g14s", False),  # M2
        ("applegpu_g15s", False),  # M3
        ("applegpu_g16s", False),  # M4
        ("applegpu_g17s", True),   # M5
        ("applegpu_g18s", True),  # M6: M5 の保守設定を継承
        ("applegpu_g19s", True),  # 後続世代も未知扱いで有効化しない
    ],
)
def test_nax_generation_is_forward_compatible(monkeypatch, architecture, expected):
    """M6以降を未知世代として非NAX側へ誤配線しない。"""
    from mlxturbo.kernels import moe_grouped_gemm as mgg

    monkeypatch.setattr(mgg.mx, "device_info", lambda: {"architecture": architecture})
    assert mgg.apple_gpu_family() == (int(architecture[-3:-1]), False)
    assert mgg.is_nax_device() is expected
