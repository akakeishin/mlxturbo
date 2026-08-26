"""tools/bridge の正しさゲート。時間は測らない (計測は bench_chain.py)。

実行:
    ./tools/bridge/build.sh
    .venv/bin/python tools/bridge/test_bridge.py
または
    .venv/bin/python -m pytest tools/bridge/test_bridge.py -q
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chain_kernels as ck  # noqa: E402
from bridge import (  # noqa: E402
    ORDER_CB,
    SPLIT_CB,
    SPLIT_ENCODER,
    WAIT,
    Bridge,
    buffer_info,
    metal_buffer,
)

TG = 256


def _read(a: mx.array) -> np.ndarray:
    """mx.array の buffer を MLX の演算を通さず直接読む。"""
    mx.eval(a)
    return np.array(memoryview(a))


def _make_x(n: int) -> mx.array:
    # 1/16 刻みで 0..3.75。N=16 まで 2^N*x + (2^N - 1) < 2^24 に収まるので
    # float32 で厳密比較できる (これを外すと丸めで閉形式と 1.0 ずれる)。
    x = (mx.arange(n, dtype=mx.float32) % 61.0) * 0.0625
    mx.eval(x)
    return x


# ---------------------------------------------------------------------------
# 1. MTLBuffer が本当に MLX のものか
# ---------------------------------------------------------------------------
def test_buffer_identity():
    for shape in [(4096,), (128, 64), (1, 1024)]:
        a = mx.zeros(shape, dtype=mx.float32)
        mx.eval(a)
        buf, off = metal_buffer(a)
        length, contents = buffer_info(buf)
        cpu = np.asarray(memoryview(a)).ctypes.data
        assert buf != 0, "null MTLBuffer"
        assert contents + off == cpu, (
            f"{shape}: contents({hex(contents)})+off({off}) != mv({hex(cpu)})"
        )
        assert length >= off + a.nbytes, f"{shape}: buffer too short"
    print("ok  buffer_identity          MTLBuffer.contents + byte_offset == mx.array の先頭")


def test_buffer_offset_view():
    a = mx.arange(1024, dtype=mx.float32)
    mx.eval(a)
    v = a[100:200]
    mx.eval(v)
    buf_a, off_a = metal_buffer(a)
    buf_v, off_v = metal_buffer(v)
    assert buf_a == buf_v, "slice が別 buffer になっている"
    assert off_v - off_a == 100 * 4, f"byte_offset が合わない: {off_v - off_a}"
    print("ok  buffer_offset_view       slice は同じ MTLBuffer + byte_offset=400")


def test_non_contiguous_rejected():
    a = mx.zeros((32, 32), dtype=mx.float32)
    mx.eval(a)
    t = mx.transpose(a)
    mx.eval(t)
    try:
        metal_buffer(t)
    except Exception as e:
        assert "row-contiguous" in str(e), f"想定外の例外: {e}"
        print("ok  non_contiguous_rejected  転置 view は拒否される")
        return
    # MLX が eval 時に実体化していれば contiguous になっているので、それも可
    print("ok  non_contiguous_rejected  (MLX が eval で実体化: 拒否不要)")


# ---------------------------------------------------------------------------
# 2. 本題: N dispatch 1 submit が MLX の N 回チェーンと一致するか
# ---------------------------------------------------------------------------
def _bridge_chain(br, pipe, x, y, steps, n, flags=WAIT):
    """dispatch 0: y = f(x)、以降 dispatch k: y = f(y)。1 submit。"""
    consts = ck.bridge_constants(n)
    ds = [br.dispatch(pipe, [x, y], (n, 1, 1), (TG, 1, 1), constants=consts)]
    for _ in range(steps - 1):
        ds.append(br.dispatch(pipe, [y, y], (n, 1, 1), (TG, 1, 1), constants=consts))
    br.submit(ds, flags=flags)
    return ds


def _check_chain(steps: int, n: int):
    x = _make_x(n)
    x_host = _read(x).copy()

    # (a) MLX 経路
    y_mlx = ck.mlx_chain(x, steps, threadgroup=TG)
    mx.eval(y_mlx)
    a_host = _read(y_mlx)

    # (b) bridge 経路
    y = mx.zeros((n,), dtype=mx.float32)
    mx.eval(y)
    br = Bridge.for_array(x)
    lib = br.add_library(ck.BRIDGE_MSL)
    pipe = br.pipeline(lib, "chain_affine")
    assert br.max_threads(pipe) >= TG
    _bridge_chain(br, pipe, x, y, steps, n)
    b_host = _read(y)

    # (c) 閉形式
    c_host = ck.closed_form(x_host.astype(np.float64), steps)

    assert np.array_equal(a_host, b_host), (
        f"MLX と bridge が不一致: max|diff|={np.abs(a_host - b_host).max()}"
    )
    assert np.array_equal(b_host.astype(np.float64), c_host), (
        f"閉形式と不一致: max|diff|={np.abs(b_host - c_host).max()}"
    )
    print(
        f"ok  chain_matches_mlx        N={steps} n={n} 完全一致 "
        f"(MLX / bridge 1submit / 閉形式), cb={br.last_command_buffers}"
    )
    assert br.last_command_buffers == 1, "1 submit のはずが command buffer が複数"
    br.close()


def test_chain_matches_mlx():
    _check_chain(16, 4096)


def test_chain_n32():
    """依頼どおりの N=32。閉形式は float32 の範囲を出るので MLX 経路と直接比較する。"""
    n, steps = 4096, 32
    x = _make_x(n)

    y_mlx = ck.mlx_chain(x, steps, threadgroup=TG)
    mx.eval(y_mlx)
    a_host = _read(y_mlx)

    y = mx.zeros((n,), dtype=mx.float32)
    mx.eval(y)
    br = Bridge.for_array(x)
    lib = br.add_library(ck.BRIDGE_MSL)
    pipe = br.pipeline(lib, "chain_affine")
    _bridge_chain(br, pipe, x, y, steps, n)
    b_host = _read(y)

    assert np.array_equal(a_host, b_host), (
        f"N=32 で不一致: max|diff|={np.abs(a_host - b_host).max()}"
    )
    assert br.last_command_buffers == 1
    print(
        f"ok  chain_n32                N=32 n={n} MLX 32 回呼び出し == bridge 32 dispatch "
        f"/ 1 submit (bit 単位で一致)"
    )
    br.close()


def _check_split_variants(steps: int, n: int):
    """encoder 分割 / command buffer 分割でも同じ結果になること。"""
    x = _make_x(n)
    ref = ck.closed_form(_read(x).astype(np.float64), steps)
    br = Bridge.for_array(x)
    lib = br.add_library(ck.BRIDGE_MSL)
    pipe = br.pipeline(lib, "chain_affine")
    for label, flags, want_cb in [
        ("1 CB / 1 encoder", WAIT, 1),
        ("1 CB / N encoder", WAIT | SPLIT_ENCODER, 1),
        ("N CB + MTLEvent", WAIT | SPLIT_CB | ORDER_CB, steps),
    ]:
        y = mx.zeros((n,), dtype=mx.float32)
        mx.eval(y)
        _bridge_chain(br, pipe, x, y, steps, n, flags=flags)
        got = _read(y).astype(np.float64)
        assert np.array_equal(got, ref), f"{label}: 不一致"
        assert br.last_command_buffers == want_cb, (
            f"{label}: cb={br.last_command_buffers} (期待 {want_cb})"
        )
        print(f"ok  split_variant            {label:<18} cb={br.last_command_buffers} 一致")
    br.close()


def test_chain_split_variants():
    _check_split_variants(16, 4096)


def test_split_cb_needs_event():
    """同一キューでも command buffer 同士は重なって走ることの記録。

    依存のある連鎖を CB 分割で流すと、MTLEvent を挟まない限り dispatch が
    取りこぼされる (「step 13 相当の値」などが返る)。ORDER_CB を付けた側は
    毎回正しいことを assert し、付けない側は成功回数を出すだけに留める
    (レースなので回数は環境と負荷で変わる)。
    """
    steps, n, trials = 16, 4096, 12
    x = _make_x(n)
    ref = ck.closed_form(_read(x).astype(np.float64), steps)
    br = Bridge.for_array(x)
    lib = br.add_library(ck.BRIDGE_MSL)
    pipe = br.pipeline(lib, "chain_affine")

    counts = {}
    for label, flags in [("event なし", WAIT | SPLIT_CB), ("event あり", WAIT | SPLIT_CB | ORDER_CB)]:
        ok = 0
        for _ in range(trials):
            y = mx.zeros((n,), dtype=mx.float32)
            mx.eval(y)
            _bridge_chain(br, pipe, x, y, steps, n, flags=flags)
            if np.array_equal(_read(y).astype(np.float64), ref):
                ok += 1
        counts[label] = ok

    assert counts["event あり"] == trials, (
        f"ORDER_CB を付けても {trials - counts['event あり']}/{trials} 回ずれた"
    )
    print(
        f"ok  split_cb_needs_event      N CB 分割の正答: "
        f"event なし {counts['event なし']}/{trials}, event あり {counts['event あり']}/{trials}"
    )
    br.close()


def test_split_cb_independent_ok():
    """依存がなければ CB 分割でも event は要らない (書き込み先が互いに素)。"""
    n, parts = 1024, 8
    x = _make_x(n)
    y = mx.zeros((n * parts,), dtype=mx.float32)
    mx.eval(y)
    br = Bridge.for_array(x)
    lib = br.add_library(ck.BRIDGE_MSL)
    pipe = br.pipeline(lib, "chain_affine")
    consts = ck.bridge_constants(n)

    buf_y, off_y = metal_buffer(y)
    buf_x, off_x = metal_buffer(x)
    from bridge import Dispatch  # noqa: PLC0415

    ds = [
        Dispatch(
            pipe,
            [(buf_x, off_x), (buf_y, off_y + k * n * 4)],
            (n, 1, 1),
            (TG, 1, 1),
            constants=consts,
            keep_alive=(x, y),
        )
        for k in range(parts)
    ]
    br.submit(ds, flags=WAIT | SPLIT_CB)
    got = _read(y).astype(np.float64).reshape(parts, n)
    ref = ck.closed_form(_read(x).astype(np.float64), 1)
    for k in range(parts):
        assert np.array_equal(got[k], ref), f"part {k} が不一致"
    print(f"ok  split_cb_independent_ok  互いに素な {parts} 区画は CB 分割でも一致")
    br.close()


def test_chain_on_offset_view():
    """byte_offset != 0 の view に対して正しい範囲だけ書くこと。"""
    steps, n = 8, 1024
    base = mx.zeros((4 * n,), dtype=mx.float32)
    mx.eval(base)
    x = _make_x(n)
    view = base[n : 2 * n]
    mx.eval(view)

    br = Bridge.for_array(base)
    lib = br.add_library(ck.BRIDGE_MSL)
    pipe = br.pipeline(lib, "chain_affine")
    _bridge_chain(br, pipe, x, view, steps, n)

    host = _read(base)
    ref = ck.closed_form(_read(x).astype(np.float64), steps)
    assert np.array_equal(host[n : 2 * n].astype(np.float64), ref), "view 内が不一致"
    assert np.all(host[:n] == 0) and np.all(host[2 * n :] == 0), "view の外を壊した"
    print(f"ok  chain_on_offset_view     N={steps} byte_offset={n * 4} の窓だけが書かれた")
    br.close()


def test_mlx_reads_bridge_result():
    """bridge が書いた buffer を MLX の op がそのまま読めること。"""
    steps, n = 8, 2048
    x = _make_x(n)
    y = mx.zeros((n,), dtype=mx.float32)
    mx.eval(y)
    br = Bridge.for_array(x)
    lib = br.add_library(ck.BRIDGE_MSL)
    pipe = br.pipeline(lib, "chain_affine")
    _bridge_chain(br, pipe, x, y, steps, n)

    # MLX 側の演算 (別 command buffer / 別キュー) から読む
    s = mx.sum(y)
    mx.eval(s)
    ref = float(ck.closed_form(_read(x).astype(np.float64), steps).sum())
    got = float(s)
    assert abs(got - ref) <= abs(ref) * 1e-6, f"mx.sum={got} ref={ref}"
    print(f"ok  mlx_reads_bridge_result  mx.sum(y)={got:.1f} が閉形式と一致")
    br.close()


def test_mlx_metallib_loads():
    """MLX 同梱の mlx.metallib を自前 device で開き、pipeline まで作れること。

    B1 の続きで MLX 本体のカーネル (affine_qmv / sdpa_vector / rms) を
    自前 encoder から名指しできるかの下見。
    """
    import mlx  # noqa: PLC0415

    # mlx.__file__ は None (namespace package) なので __path__ を使う
    path = os.path.join(mlx.__path__[0], "lib", "mlx.metallib")
    if not os.path.exists(path):
        print("skip mlx_metallib_loads     mlx.metallib が見つからない")
        return
    a = mx.zeros((1024,), dtype=mx.float32)
    mx.eval(a)
    br = Bridge.for_array(a)
    lib = br.add_library_file(path)
    names = br.function_names(lib)
    assert len(names) > 1000, f"関数が少なすぎる: {len(names)}"
    wanted = "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0"
    assert wanted in names, f"{wanted} が無い"
    pipe = br.pipeline(lib, wanted)
    assert br.max_threads(pipe) > 0
    n_qmv = sum(1 for n in names if "affine_qmv" in n)
    n_sdpa = sum(1 for n in names if "sdpa" in n)
    print(
        f"ok  mlx_metallib_loads       {len(names)} 関数 (affine_qmv {n_qmv} / sdpa {n_sdpa})、"
        f"pipeline 生成まで到達"
    )
    br.close()


def test_device_is_shared():
    a = mx.zeros((1024,), dtype=mx.float32)
    mx.eval(a)
    br = Bridge.for_array(a)
    name = br.device_name
    assert name, "device name が空"
    info = mx.device_info()
    assert name == info["device_name"], f"bridge={name} mlx={info['device_name']}"
    print(f"ok  device_is_shared         MLX と同じ MTLDevice ({name})")
    br.close()


TESTS = [
    test_device_is_shared,
    test_buffer_identity,
    test_buffer_offset_view,
    test_non_contiguous_rejected,
    test_chain_matches_mlx,
    test_chain_n32,
    test_chain_split_variants,
    test_split_cb_needs_event,
    test_split_cb_independent_ok,
    test_chain_on_offset_view,
    test_mlx_reads_bridge_result,
    test_mlx_metallib_loads,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
