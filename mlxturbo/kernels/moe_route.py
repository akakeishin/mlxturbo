"""MoE のルーティング (top-k 選択と softmax) を 1 本の Metal カーネルに畳む。

**結論: 有効にする価値は無い。既定 off のまま置いてある。**
7 op を 3 op に減らしたのに、v-fast6 の実測で **+0.34ms/token 遅くなる**
(静かな条件、振れ 0.17。汚染された別の回では +2.44ms で、方向は一致)。

理由は「素の 5 op が既に安い」こと。`argpartition` も `softmax` も 512 要素
しか触らないので GPU 実働はほぼ無料で、削れるのはディスパッチ費用だけ。
一方こちらは 1 スレッドグループ 32 スレッドで「最大を取って外す」を 10 回
**逐次に**回すので、並列度が使えずその逐次実行が食い潰す。

速くするなら 10 回の逐次パスをやめて、各レーンがローカル top-k を持ってから
マージする形にする必要がある。ただし上限はルーティング全体の 2.75ms
(tools/ablate_moe.py, v-fast6) で、そこまで投資する価値があるかは別途判断。

素の実装 (`SparseMoeBlock.__call__`) はここが 7 op:

    logits = self.gate(x.astype(mx.float32))                       # astype, 行列積
    idx = mx.argpartition(-logits, top_k - 1, axis=-1)[..., :top_k]  # 符号反転, 分割, 切り出し
    w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)

48 層すべてで呼ばれ、ルーティングだけで実測 **2.69ms/token**
(tools/ablate_moe.py、expert 行列積 8.16ms とは別勘定)。1 層 56us / 5 op で
MLX のディスパッチ費用そのもの。

## gate の行列積はカーネルに入れない

`gate` の全出力 (512) が揃わないと top-k を始められない。行列積を含めると
グリッド全体のバリアが要り、hyper-connections と同じく 2 本に割れる。
一方 top-k と softmax だけなら 512 要素なので **1 スレッドグループに収まり、
1 本で済む**。行列積は MLX に任せて 7 op -> 3 op にするのが得。

## 順序について

`mx.argpartition` が返す top-k の**順序は未規定**で、再現する意味がない。
このカーネルは**降順**という決定的な順序で返す。

呼び出し側は `(switch_mlp(x, idx) * w[..., None]).sum(axis=-2)` と使うので、
**選ばれた集合と (index, 重み) の対応が合っていれば結果は順序に依らない**
— ただし総和の丸め順は変わる。bf16 で 10 項を足す順が違うぶんだけ、
素とビット一致はしない (docs/KERNEL-HANDOFF-HC.md の「対照実験」と同じ性質)。

## 精度

logits も softmax も fp32 なので、bf16 の丸めは入らない。`mx.softmax(...,
precise=True)` と同じく最大値を引いてから exp する。
"""

from __future__ import annotations

from math import prod
from typing import Any

import mlx.core as mx

_KERNELS: dict[tuple, Any] = {}


def _source(n_experts: int, top_k: int) -> str:
    return f"""
    uint lane = thread_index_in_simdgroup;
    int  row  = (int)threadgroup_position_in_grid.y;

    threadgroup float tg[{n_experts}];
    threadgroup float tg_sel[{top_k}];
    threadgroup uint  tg_idx[{top_k}];

    const device float* lr = logits + (size_t)row * {n_experts};
    for (int i = (int)lane; i < {n_experts}; i += 32) {{
        tg[i] = lr[i];
    }}
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // top-k を「最大を取って外す」を k 回で求める。k=10、n=512 なので
    // 部分ソートを書くより素直で速い
    for (int p = 0; p < {top_k}; p++) {{
        float best = -INFINITY;
        int   bi = -1;
        for (int i = (int)lane; i < {n_experts}; i += 32) {{
            float v = tg[i];
            if (v > best) {{ best = v; bi = i; }}
        }}
        // simdgroup 内のラダー縮約。lane 0 に最大とその index が集まる
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {{
            float ov = simd_shuffle_down(best, off);
            int   oi = simd_shuffle_down(bi, off);
            // 同値は index の小さい方を残して決定的にする
            if (ov > best || (ov == best && oi >= 0 && oi < bi)) {{
                best = ov; bi = oi;
            }}
        }}
        if (lane == 0) {{
            tg_sel[p] = best;
            tg_idx[p] = (uint)bi;
            tg[bi] = -INFINITY;   // 取ったものを次の探索から外す
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // softmax。降順に取っているので tg_sel[0] が最大値
    if (lane == 0) {{
        float m = tg_sel[0];
        float s = 0.0f;
        for (int p = 0; p < {top_k}; p++) {{
            float e = metal::exp(tg_sel[p] - m);
            tg_sel[p] = e;
            s += e;
        }}
        for (int p = 0; p < {top_k}; p++) {{
            idx[(size_t)row * {top_k} + p] = tg_idx[p];
            weights[(size_t)row * {top_k} + p] = tg_sel[p] / s;
        }}
    }}
"""


def _get_kernel(n_experts: int, top_k: int):
    key = (n_experts, top_k)
    k = _KERNELS.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"moe_route_{n_experts}_{top_k}_{len(_KERNELS)}",
            input_names=["logits"],
            output_names=["idx", "weights"],
            source=_source(n_experts, top_k),
        )
        _KERNELS[key] = k
    return k


# threadgroup メモリに logits を丸ごと置く。Apple GPU の上限 32KB に余裕を持たせる
MAX_EXPERTS = 4096


def eligible(logits: mx.array, top_k: int) -> bool:
    """このカーネルで扱える形かを判定する。外れたら呼び出し側は素の実装へ。"""

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    if logits.dtype != mx.float32:
        return False
    n = logits.shape[-1]
    if n > MAX_EXPERTS or top_k > n or top_k < 1:
        return False
    return True


def route(logits: mx.array, top_k: int):
    """top-k の index (uint32) と softmax 重み (float32) を返す。

    `idx` は**降順**。`mx.argpartition` の未規定な順序は再現しない
    (モジュール docstring の「順序について」を参照)。
    """

    n = logits.shape[-1]
    lead = logits.shape[:-1]
    rows = prod(lead) if lead else 1
    kernel = _get_kernel(n, top_k)
    flat = mx.contiguous(logits.reshape((rows, n)))

    idx, w = kernel(
        inputs=[flat],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return idx.reshape((*lead, top_k)), w.reshape((*lead, top_k))


__all__ = ["eligible", "route"]
