"""MoE の combine (unsort + ルータ重み + top_k 軸の和) を 1 カーネルに畳む
(P7 第 2 段)。

## 何を置き換えるのか

prefill の MoE は `_moe_combine_fold` (`mlxturbo/_vendor/qwen4_exp.py`) を
通る。down GEMM の**後**に残っているのは 3 つで、`bench/results/moe-split.json`
(層 20、M=2048、8k プロンプト) の内訳ではこうなっている:

    weight_mul_cast  0.87 ms   (act * w) の実体化 + bf16 への cast
                               ※ これは down の「前」だが同じ話 (下記)
    unsort_sum       1.07 ms   out[inv_order] の gather + sum(axis=-2)

down_proj は bias 無しの線形写像なので、ルータ重みは down の前 (SwiGLU 出力、
640 幅) でも後 (down 出力、2560 幅) でも数式上は同じ。**後で掛けるなら、
unsort の gather と top_k の和と同じ読みの中で掛けられるので追加の
メモリ往復がゼロになる。**そこでこのカーネルは

    y[t, n] = Σ_k w[t*top_k + k] * out_sorted[inv[t*top_k + k], n]

を 1 パスで計算する。読むのは out_sorted (16384x2560 bf16 = 84MB) 1 回、
書くのは (2048x2560 = 10.5MB) 1 回だけ。置き換わる 3 つ (weight_mul_cast の
126MB 往復 + unsort の 168MB 往復 + sum の 94MB) が全部消える。

累算は fp32、丸めは書くときの 1 回だけなので、現行 (`(act*w)` を bf16 に
丸めてから down GEMM、top_k の和も bf16 の出力から取る) より**丸めが 2 回
少ない**。ビット一致はしない。

**精度は上がる側にずれる** (2026-09-03、`scratchpad/accuracy_check.py`、
本番の形 M=16384/E=512、参照は fp32 の逆量子化 + fp32 の重み + fp32 の和):

    現行          平均絶対誤差 1.281e-3  最大 3.09e-2
    epi (別変種)  平均絶対誤差 1.174e-3  最大 3.09e-2
    combine       平均絶対誤差 7.14e-4  最大 1.37e-2   (参照の平均 2.78e-1)

17k の A/B で 3 本中 1 本の生成トークンが現行と分かれたが、分かれた先は
**厳密解に近い側**。

## 読み方 (coalescing)

1 スレッドが 1 トークンの VEC 列を持つ。同じ threadgroup の隣のスレッドは
隣の列なので、`out_sorted[s]` の行の中を連続に読む (完全に coalesce する)。
行 s はトークンごとに散らばるが、行の中は 2560 要素連続なので、散らばりは
「64 バイト単位の連続読みが top_k 本ある」形にしかならない。

## 配線

`mlxturbo.fused.enable_moe_down_epilogue(mode="combine")` が
`_moe_combine_fold` のフック経由で使う。A/B は
`tools/decode_ab.py --knob moe-down-epi`。
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import _fire

# 1 スレッドが持つ列数。N=2560 は 4 で割り切れる。8 にすると 1 スレッドの
# 累算器が 8 本になり、top_k=8 のループで register が増える
VEC = 4

_KERNELS: dict[tuple, Any] = {}

_SOURCE = """
  const int N = dims[0];

  // grid は (N/VEC, 1, rows) ちょうどで張るので、範囲外のスレッドは来ない
  // (端数の分岐を置くと、下の threadgroup バリアに全スレッドが到達しなく
  // なる)。形の保証は python 側の `eligible` が持つ。
  const uint t = threadgroup_position_in_grid.z;
  const uint lid = thread_position_in_threadgroup.x;
  const size_t base = (size_t)t * (size_t)TOPK;

  // 同じ threadgroup のスレッドは全員同じトークン t を見るので、
  // 行番号とルータ重み (top_k 本) は 1 回だけ読んで共有する。素朴に
  // 各スレッドが読むと、この 2 本だけで y と同じだけの読み要求が出る
  // (640 スレッド/トークン x 8 本)
  threadgroup uint tg_s[TOPK];
  threadgroup float tg_w[TOPK];
  for (uint i = lid; i < (uint)TOPK; i += TG) {
    tg_s[i] = inv[base + i];
    tg_w[i] = w[base + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint n = thread_position_in_grid.x * VEC;
  float acc[VEC];
  STEEL_UNROLL
  for (short v = 0; v < VEC; v++) {
    acc[v] = 0.0f;
  }

  STEEL_UNROLL
  for (short k = 0; k < TOPK; k++) {
    const float wk = tg_w[k];
    const device T* p = y + (size_t)tg_s[k] * (size_t)N + (size_t)n;
    STEEL_UNROLL
    for (short v = 0; v < VEC; v++) {
      acc[v] += wk * (float)p[v];
    }
  }

  device T* o = out + (size_t)t * (size_t)N + (size_t)n;
  STEEL_UNROLL
  for (short v = 0; v < VEC; v++) {
    o[v] = (T)acc[v];
  }
"""

_HEADER = """
#define STEEL_UNROLL _Pragma("clang loop unroll(full)")
"""


def _get_kernel():
    kernel = _KERNELS.get("combine")
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="mlxturbo_moe_combine",
            input_names=["y", "inv", "w", "dims"],
            output_names=["out"],
            source=_SOURCE,
            header=_HEADER,
            ensure_row_contiguous=True,
        )
        _KERNELS["combine"] = kernel
    return kernel


def eligible(y: mx.array, inv: mx.array, w: mx.array, rows: int,
             top_k: int) -> bool:
    """このカーネルが使える形か。**host 同期はしない** (形と dtype だけ)。"""

    if mx.default_device() != mx.gpu:
        return False
    if y.ndim != 2 or inv.ndim != 1 or w.ndim != 1:
        return False
    M, N = y.shape
    if M != rows * top_k or inv.shape[0] != M or w.shape[0] != M:
        return False
    if N % VEC != 0:
        return False
    if y.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    return 1 <= top_k <= 32


def _threads(n_x: int) -> int:
    for tg in (256, 128, 64, 32, 16, 8, 4, 2):
        if n_x % tg == 0:
            return tg
    return 1


def combine(y: mx.array, inv: mx.array, w: mx.array, rows: int,
            top_k: int) -> mx.array:
    """``y`` (M, N、専門家順) から ``(rows, N)`` の重み付き和を作る。

    ``inv`` はトークン t の k 番目が入っている**ソート後の行**
    (`argsort(order)` = 現行の `inv_order` そのもの、uint32 の (M,))、
    ``w`` はそのルータ重み (fp32 の (M,))。どちらも t*top_k + k の並び。
    """

    _fire.bump("moe_combine")
    M, N = y.shape
    if inv.dtype != mx.uint32:
        inv = inv.astype(mx.uint32)
    if w.dtype != mx.float32:
        w = w.astype(mx.float32)
    dims = mx.array([N, rows], dtype=mx.int32)
    n_x = N // VEC
    tg = _threads(n_x)
    kernel = _get_kernel()
    (out,) = kernel(
        inputs=[y, inv, w, dims],
        template=[("T", y.dtype), ("TOPK", int(top_k)), ("VEC", VEC),
                  ("TG", int(tg))],
        grid=(n_x, 1, rows),
        threadgroup=(tg, 1, 1),
        output_shapes=[(rows, N)],
        output_dtypes=[y.dtype],
    )
    return out


__all__ = ["VEC", "combine", "eligible"]
