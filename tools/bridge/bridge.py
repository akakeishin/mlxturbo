"""mx.array の MTLBuffer を取り出し、自前の MTLCommandBuffer へ直接エンコードする。

B1 (docs/ARCH-BETS.md) の下部工事。使い方の要約:

    import mlx.core as mx
    from bridge import Bridge, metal_buffer

    x = mx.zeros((4096,), dtype=mx.float32)
    mx.eval(x)                       # ← 必須。規約は docs/BRIDGE-NOTES.md
    br = Bridge.for_array(x)
    lib = br.add_library(SOURCE)
    pipe = br.pipeline(lib, "chain_affine")
    br.submit([br.dispatch(pipe, [x, y], grid=(n, 1, 1), threadgroup=(256, 1, 1))])

MTLBuffer の取得は mx.array.__dlpack__() 経由。MLX 0.32.2 の DLPack は
device_type=kDLMetal(8) で、DLTensor.data に MTLBuffer のポインタ、
byte_offset にバイト単位のオフセットを入れてくる。nanobind / pybind11 も
libmlx へのリンクも要らない。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import Iterable, Sequence

import mlx.core as mx

__all__ = [
    "Bridge",
    "Dispatch",
    "BridgeError",
    "metal_buffer",
    "buffer_info",
    "WAIT",
    "THREADGROUPS",
    "NO_BARRIER",
    "SPLIT_CB",
    "SPLIT_ENCODER",
    "UNRETAINED",
    "ORDER_CB",
]

# fmb_submit のフラグ (fastmlx_bridge.mm と一致させること)
WAIT = 1 << 0
THREADGROUPS = 1 << 1
NO_BARRIER = 1 << 2
SPLIT_CB = 1 << 3
SPLIT_ENCODER = 1 << 4
UNRETAINED = 1 << 5
# SPLIT_CB で分けた command buffer 同士を MTLEvent で直列化する。
# 同一キューでも CB は重なって走るので、依存のある連鎖には必須 (実測)。
ORDER_CB = 1 << 6

_HERE = os.path.dirname(os.path.abspath(__file__))
_DYLIB = os.path.join(_HERE, "libfastmlx_bridge.dylib")

_ERRLEN = 1024

# DLPack kDLMetal
_DL_METAL = 8


class BridgeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# DLPack から MTLBuffer を取り出す
# ---------------------------------------------------------------------------


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
        ("ndim", ctypes.c_int32),
        ("dtype_code", ctypes.c_uint8),
        ("dtype_bits", ctypes.c_uint8),
        ("dtype_lanes", ctypes.c_uint16),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_pyapi = ctypes.pythonapi
_pyapi.PyCapsule_GetName.restype = ctypes.c_char_p
_pyapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
_pyapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
_pyapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def metal_buffer(a: mx.array) -> tuple[int, int]:
    """(MTLBuffer ポインタ, バイトオフセット) を返す。

    a は評価済み (mx.eval 済み) かつ row-contiguous であること。
    返るポインタは a が生きている間だけ有効 (docs/BRIDGE-NOTES.md の「寿命」)。
    """
    if not isinstance(a, mx.array):
        raise TypeError(f"expected mx.array, got {type(a).__name__}")
    mx.eval(a)

    capsule = a.__dlpack__()
    name = _pyapi.PyCapsule_GetName(capsule)
    ptr = _pyapi.PyCapsule_GetPointer(capsule, name)
    if not ptr:
        raise BridgeError("PyCapsule_GetPointer returned NULL")
    mt = _DLManagedTensor.from_address(ptr)
    t = mt.dl_tensor

    if t.device_type != _DL_METAL:
        raise BridgeError(
            f"array is not on the Metal device (dlpack device_type={t.device_type}); "
            "the bridge only handles GPU-resident arrays"
        )
    if not t.data:
        raise BridgeError("dlpack data pointer is NULL")

    # row-contiguous でなければ拒否する。オフセット付き view は許すが、
    # transpose / broadcast 済みの view は勝手に潰さず呼び出し側に返す。
    if t.strides:
        expected = 1
        for i in range(t.ndim - 1, -1, -1):
            if t.shape[i] != 1 and t.strides[i] != expected:
                raise BridgeError(
                    "array is not row-contiguous; call mx.contiguous()/reshape "
                    "and mx.eval() before handing it to the bridge"
                )
            expected *= t.shape[i]

    buf = int(t.data)
    off = int(t.byte_offset)
    # capsule はここで参照が切れ、MLX 側のカプセル destructor が deleter を
    # 呼んで DLManagedTensor を解放する。こちらから deleter を呼ぶと
    # destructor と二重に走って nanobind が abort する (実測)。
    del mt, t, capsule
    return buf, off


# ---------------------------------------------------------------------------
# dylib
# ---------------------------------------------------------------------------


class _FMBDispatch(ctypes.Structure):
    _fields_ = [
        ("pipeline", ctypes.c_int32),
        ("n_buffers", ctypes.c_int32),
        ("buffers", ctypes.POINTER(ctypes.c_void_p)),
        ("offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("bytes", ctypes.c_void_p),
        ("bytes_len", ctypes.c_uint32),
        ("bytes_index", ctypes.c_int32),
        ("grid_x", ctypes.c_uint32),
        ("grid_y", ctypes.c_uint32),
        ("grid_z", ctypes.c_uint32),
        ("tg_x", ctypes.c_uint32),
        ("tg_y", ctypes.c_uint32),
        ("tg_z", ctypes.c_uint32),
        ("threadgroup_mem_len", ctypes.c_uint32),
        ("threadgroup_mem_index", ctypes.c_int32),
    ]


_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_DYLIB):
        raise BridgeError(
            f"{_DYLIB} not found — run tools/bridge/build.sh first"
        )
    lib = ctypes.CDLL(_DYLIB)
    c_err = ctypes.c_char_p

    lib.fmb_context_create.restype = ctypes.c_void_p
    lib.fmb_context_create.argtypes = [ctypes.c_void_p, c_err, ctypes.c_int]
    lib.fmb_context_destroy.restype = None
    lib.fmb_context_destroy.argtypes = [ctypes.c_void_p]
    lib.fmb_device_name.restype = ctypes.c_char_p
    lib.fmb_device_name.argtypes = [ctypes.c_void_p]
    lib.fmb_dispatch_struct_size.restype = ctypes.c_int
    lib.fmb_dispatch_struct_size.argtypes = []
    lib.fmb_buffer_info.restype = ctypes.c_int
    lib.fmb_buffer_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
        c_err,
        ctypes.c_int,
    ]
    lib.fmb_library_from_source.restype = ctypes.c_int
    lib.fmb_library_from_source.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        c_err,
        ctypes.c_int,
    ]
    lib.fmb_library_from_file.restype = ctypes.c_int
    lib.fmb_library_from_file.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        c_err,
        ctypes.c_int,
    ]
    lib.fmb_library_function_count.restype = ctypes.c_int
    lib.fmb_library_function_count.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.fmb_library_function_name.restype = ctypes.c_int
    lib.fmb_library_function_name.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.fmb_pipeline.restype = ctypes.c_int
    lib.fmb_pipeline.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_char_p,
        c_err,
        ctypes.c_int,
    ]
    lib.fmb_pipeline_max_threads.restype = ctypes.c_int
    lib.fmb_pipeline_max_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.fmb_submit.restype = ctypes.c_int
    lib.fmb_submit.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_FMBDispatch),
        ctypes.c_int,
        ctypes.c_int,
        c_err,
        ctypes.c_int,
    ]
    for name in (
        "fmb_last_encode_ms",
        "fmb_last_wall_ms",
        "fmb_last_gpu_ms",
    ):
        getattr(lib, name).restype = ctypes.c_double
        getattr(lib, name).argtypes = [ctypes.c_void_p]
    lib.fmb_last_command_buffers.restype = ctypes.c_int
    lib.fmb_last_command_buffers.argtypes = [ctypes.c_void_p]

    got = lib.fmb_dispatch_struct_size()
    if got != ctypes.sizeof(_FMBDispatch):
        raise BridgeError(
            f"FMBDispatch layout mismatch: C={got} python={ctypes.sizeof(_FMBDispatch)}"
        )
    _lib = lib
    return _lib


def buffer_info(buffer_ptr: int) -> tuple[int, int]:
    """MTLBuffer の (length, contents ポインタ) を返す。検算用。"""
    lib = _load()
    length = ctypes.c_uint64(0)
    contents = ctypes.c_void_p(0)
    err = ctypes.create_string_buffer(_ERRLEN)
    rc = lib.fmb_buffer_info(
        ctypes.c_void_p(buffer_ptr),
        ctypes.byref(length),
        ctypes.byref(contents),
        err,
        _ERRLEN,
    )
    if rc != 0:
        raise BridgeError(err.value.decode())
    return int(length.value), int(contents.value or 0)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class Dispatch:
    """1 回の compute dispatch。参照を握って ctypes バッファを生かしておく。"""

    __slots__ = ("_c", "_keep")

    def __init__(
        self,
        pipeline: int,
        buffers: Sequence[tuple[int, int]],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        constants: bytes | None = None,
        constants_index: int | None = None,
        threadgroup_mem: int = 0,
        threadgroup_mem_index: int = -1,
        keep_alive: Iterable[object] = (),
    ):
        n = len(buffers)
        cbufs = (ctypes.c_void_p * max(n, 1))(*[ctypes.c_void_p(b) for b, _ in buffers])
        coffs = (ctypes.c_uint64 * max(n, 1))(*[o for _, o in buffers])

        cbytes = None
        blen = 0
        bidx = -1
        if constants:
            cbytes = ctypes.create_string_buffer(constants, len(constants))
            blen = len(constants)
            bidx = n if constants_index is None else constants_index

        self._c = _FMBDispatch(
            pipeline=pipeline,
            n_buffers=n,
            buffers=cbufs,
            offsets=coffs,
            bytes=ctypes.cast(cbytes, ctypes.c_void_p) if cbytes else None,
            bytes_len=blen,
            bytes_index=bidx,
            grid_x=grid[0],
            grid_y=grid[1],
            grid_z=grid[2],
            tg_x=threadgroup[0],
            tg_y=threadgroup[1],
            tg_z=threadgroup[2],
            threadgroup_mem_len=threadgroup_mem,
            threadgroup_mem_index=threadgroup_mem_index,
        )
        # MTLBuffer を握っている mx.array を含め、submit まで生かす
        self._keep = (cbufs, coffs, cbytes, tuple(keep_alive))


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class Bridge:
    def __init__(self, buffer_hint: int = 0):
        lib = _load()
        err = ctypes.create_string_buffer(_ERRLEN)
        ctx = lib.fmb_context_create(ctypes.c_void_p(buffer_hint), err, _ERRLEN)
        if not ctx:
            raise BridgeError(err.value.decode())
        self._lib = lib
        self._ctx = ctypes.c_void_p(ctx)

    @classmethod
    def for_array(cls, a: mx.array) -> "Bridge":
        """MLX の buffer を所有する MTLDevice からキューを作る (device 一致を保証)。"""
        buf, _ = metal_buffer(a)
        return cls(buffer_hint=buf)

    def close(self):
        if getattr(self, "_ctx", None):
            self._lib.fmb_context_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def device_name(self) -> str:
        return self._lib.fmb_device_name(self._ctx).decode()

    def add_library(self, source: str, fast_math: bool = False) -> int:
        err = ctypes.create_string_buffer(_ERRLEN)
        lib_id = self._lib.fmb_library_from_source(
            self._ctx, source.encode(), 1 if fast_math else 0, err, _ERRLEN
        )
        if lib_id < 0:
            raise BridgeError(err.value.decode())
        return lib_id

    def add_library_file(self, path: str) -> int:
        """既製の .metallib を読む。MLX 同梱の mlx.metallib もこれで開ける。"""
        err = ctypes.create_string_buffer(_ERRLEN)
        lib_id = self._lib.fmb_library_from_file(
            self._ctx, os.fspath(path).encode(), err, _ERRLEN
        )
        if lib_id < 0:
            raise BridgeError(err.value.decode())
        return lib_id

    def function_names(self, lib_id: int, contains: str | None = None) -> list[str]:
        n = self._lib.fmb_library_function_count(self._ctx, lib_id)
        if n < 0:
            raise BridgeError(f"bad library id {lib_id}")
        buf = ctypes.create_string_buffer(512)
        out = []
        for i in range(n):
            if self._lib.fmb_library_function_name(self._ctx, lib_id, i, buf, 512) < 0:
                continue
            name = buf.value.decode()
            if contains is None or contains in name:
                out.append(name)
        return out

    def pipeline(self, lib_id: int, fn_name: str) -> int:
        err = ctypes.create_string_buffer(_ERRLEN)
        pid = self._lib.fmb_pipeline(self._ctx, lib_id, fn_name.encode(), err, _ERRLEN)
        if pid < 0:
            raise BridgeError(err.value.decode())
        return pid

    def max_threads(self, pipe_id: int) -> int:
        return self._lib.fmb_pipeline_max_threads(self._ctx, pipe_id)

    def dispatch(
        self,
        pipeline: int,
        arrays: Sequence[mx.array],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        constants: bytes | None = None,
        constants_index: int | None = None,
    ) -> Dispatch:
        """mx.array 列から Dispatch を組む (buffer 取り出しつき)。"""
        bufs = [metal_buffer(a) for a in arrays]
        return Dispatch(
            pipeline,
            bufs,
            grid,
            threadgroup,
            constants=constants,
            constants_index=constants_index,
            keep_alive=tuple(arrays),
        )

    def submit(self, dispatches: Sequence[Dispatch], flags: int = WAIT) -> None:
        n = len(dispatches)
        if n == 0:
            raise BridgeError("empty dispatch list")
        arr = (_FMBDispatch * n)()
        for i, d in enumerate(dispatches):
            arr[i] = d._c
        err = ctypes.create_string_buffer(_ERRLEN)
        rc = self._lib.fmb_submit(self._ctx, arr, n, flags, err, _ERRLEN)
        if rc != 0:
            raise BridgeError(err.value.decode())

    # 直前の submit の計測値
    @property
    def last_encode_ms(self) -> float:
        return self._lib.fmb_last_encode_ms(self._ctx)

    @property
    def last_wall_ms(self) -> float:
        return self._lib.fmb_last_wall_ms(self._ctx)

    @property
    def last_gpu_ms(self) -> float:
        return self._lib.fmb_last_gpu_ms(self._ctx)

    @property
    def last_command_buffers(self) -> int:
        return self._lib.fmb_last_command_buffers(self._ctx)
