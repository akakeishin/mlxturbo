"""QSA のブロック top-k 選択を 1 dispatch の Metal カーネルにする (段 K2a)。

置き換える相手は `QSAIndexer._pooled_and_top`
(`mlxturbo/_vendor/qwen4_exp.py`) の**選択部分**、つまり einsum の出力
``scores`` (B, S, n_blocks, 4) fp32 を受け取ってから keep_block を作るまでの

    scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(self.head_dim)
    visible = block_end[None, None, :] <= q_col[None, :, None]
    scores = mx.where(visible, scores, -mx.inf)
    k = min(self.block_topk, n_blocks)
    top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
    keep_block = mx.zeros((B, S, n_blocks + 1), dtype=mx.bool_)
    top = mx.where(mx.take_along_axis(visible, top, axis=-1), top, n_blocks)
    keep_block = mx.put_along_axis(keep_block, top, mx.array(True), axis=-1)[..., :n_blocks]

の 10 本前後 (argpartition は GPU では全ソートなので、n_blocks=4250 では
それだけで multi_block_sort の 5 本になる)。**内積 (einsum) は畳まない。**
fp32 128 項の和は steel gemm の MMA 順で丸まるので自前の順では再現できず、
境界の同点付近で選択が反転しうる (設計メモ §2a)。

## 選ぶ集合は argpartition と同じでなければならない

`mx.argpartition` は Metal では安定な全ソート (`sort.cpp` の
`ArgPartition::eval_gpu -> gpu_merge_sort`、thread sort は odd-even、merge の
比較は strict `<`) なので、**同点は添字の小さい方から**選ばれる。
このカーネルはその規則をそのまま写す:

1. fp32 のスコアを順序保存の uint32 キーに直す (正なら最上位ビットを立て、
   負なら全ビット反転)。`relu` が返しうる ``-0.0`` は ``+0.0`` に潰す
   (IEEE の比較では等しいのに、ビット列は違って radix の順が狂うため)。
2. radix select (上位 8 bit から 4 pass、各 pass 256 bin のヒストグラム) で
   **k 番目に大きいキー** t を出す。
3. ``key > t`` を全部取り、``key == t`` は**添字の昇順で足りない分だけ**取る。

これが `argpartition(-scores, k-1)[..., :k]` の選ぶ集合と一致する。

## 可視判定

``visible`` は ``block_end = n*cr + cr - 1 <= q_col``、すなわち
``n < (q_col + 1) // cr`` なので**常に先頭からの連続範囲**になる。よって
可視ブロック数 ``n_vis = min(n_blocks, (q_col + 1) // cr)`` を行ごとに渡せば、
``mx.where(visible, scores, -inf)` を作らずに走査範囲を切るだけで足りる
(`visible_counts` がこの計算)。``n_vis < k`` の行 (offset < cr*k-1、本番では
踏まないが呼び手の都合で起きうる) は可視ブロックを全部返し、``cnt`` が
``k`` より小さくなる。

## 出力

- ``sel`` (rows, k) int32: **昇順**のブロック添字。足りない分は本家と同じ
  番兵 ``n_blocks`` で埋める (`_pooled_and_top` が
  ``mx.where(..., top, n_blocks)`` で使っている値と同じ。そのまま
  (n_blocks+1) 幅のバッファへ書ける)。
- ``bits`` (rows, ceil(n_blocks/32)) uint32: 選んだブロックのビットマップ。
  段 K2b の mirror モード (threadgroup 常駐の keep 判定) 用。**端数 (tail) は
  入っていない。**本家 (HF transformers の qwen4_exp) の可視集合は
  「完全ブロックの top-k」∪「クエリごとの端数 ``4*floor((q+1)/4) .. q``
  (自分自身を含む)」で、後者は完全ブロックの選択とは無関係に常に見える。
  端数は K2b 側で ``q`` から直に判定すること (ビットマップに混ぜない)。
- ``cnt`` (rows,) int32: 実際に選んだ本数 (= ``min(k, n_vis)``)。

``mode`` で ``sel`` と ``bits`` のどちらを作るかを切る。両方要るのは検証
(`tools/verify_qsa_select.py`) だけで、**使わない側は作らない**
(設計メモ §5-8「作って捨てる禁止」)。

## 形

1 threadgroup (1024 thread) = 1 行。decode/verify 幅は S <= 8 なので
threadgroup は最大 8 個しか出ない。n_blocks <= 16384 (kv 64k) まで。
n_blocks が threadgroup メモリに載る間 (<= ``_CACHE_CAP``) はキーを 1 回
だけ作って常駐させ、4 pass はそこから読む。超える長さは pass ごとに
device から読み直す (その分だけ遅い。kv 25.6k 相当までは常駐側)。

**まだ配線していない。**`_pooled_and_top` を割って呼ぶのは段 K2b と一緒に
やる (設計メモ「配線」)。
"""

from __future__ import annotations

import math
from math import prod
from typing import Any

import mlx.core as mx

from . import _fire

# 1 threadgroup のスレッド数。Apple GPU の上限。
_NT = 1024

# キーを threadgroup メモリに常駐させる上限 (uint32 個)。
# 25600 B + ヒストグラム 1 KB + ビットマップ 800 B + 端数 < 32 KB。
# 6400 ブロックは compress_ratio=4 で kv 25.6k までを覆う。
_CACHE_CAP = 6400

# 常駐しない側で扱う n_blocks の上限 (kv 64k)。ビットマップの
# threadgroup 配列を 512 word 固定で取るための値。
MAX_BLOCKS = 16384

_KERNELS: dict[tuple, Any] = {}
_DIVISORS: dict[float, mx.array] = {}


_HEADER = r"""
// スコア 1 ブロック分 (relu -> h の逐次和 -> /sqrt(head_dim)) を、
// 順序保存の uint32 キーに直す。
//
// relu は本家の `mx.maximum(x, 0)` (MLX の Maximum は `x > y ? x : y` なので
// -0.0 も +0.0 も +0.0 を返す) と同じ形で書く。和は 0.0f から始める逐次和で、
// MLX の `row_reduce_small` (最終軸が短い reduce) と同じ順序。
inline uint mlxturbo_qsa_key(const device float* p, float d) {
    float s = 0.0f;
    s = s + ((p[0] > 0.0f) ? p[0] : 0.0f);
    s = s + ((p[1] > 0.0f) ? p[1] : 0.0f);
    s = s + ((p[2] > 0.0f) ? p[2] : 0.0f);
    s = s + ((p[3] > 0.0f) ? p[3] : 0.0f);
    uint u = as_type<uint>(s / d);
    if (u == 0x80000000u) { u = 0u; }          // -0.0 は +0.0 と同じキーに
    return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
}
"""


def _source(k: int, cap: int, nwmax: int, mode: str) -> str:
    want_sel = mode in ("idx", "both")
    want_bits = mode in ("bits", "both")
    cached = cap > 0

    key_of = "kc[i]" if cached else "mlxturbo_qsa_key(rrow + (size_t)i * 4, DIV)"
    cache_decl = f"    threadgroup uint kc[{cap}];\n" if cached else ""
    cache_store = "            kc[i] = key;\n" if cached else ""

    bits_decl = f"    threadgroup atomic_uint bt[{nwmax}];\n" if want_bits else ""
    bits_clear = (
        "    for (int w = (int)tid; w < NW; w += NT) {\n"
        "        atomic_store_explicit(&bt[w], 0u, memory_order_relaxed);\n"
        "    }\n"
        if want_bits
        else ""
    )
    bits_zero_out = (
        "        for (int w = (int)tid; w < NW; w += NT) {\n"
        "            bits[(size_t)row * NW + w] = 0u;\n"
        "        }\n"
        if want_bits
        else ""
    )
    bits_set = (
        "                atomic_fetch_or_explicit(\n"
        "                    &bt[i >> 5], 1u << ((uint)i & 31u), memory_order_relaxed);\n"
        if want_bits
        else ""
    )
    bits_flush = (
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (int w = (int)tid; w < NW; w += NT) {\n"
        "        bits[(size_t)row * NW + w] =\n"
        "            atomic_load_explicit(&bt[w], memory_order_relaxed);\n"
        "    }\n"
        if want_bits
        else ""
    )

    sel_fill = (
        "    for (int i = (int)tid; i < K; i += NT) { sel[(size_t)row * K + i] = NB; }\n"
        if want_sel
        else ""
    )
    sel_write = (
        "                if (pos < (uint)K) { srow[pos] = i; }\n                pos++;\n"
        if want_sel
        else ""
    )
    sel_row = (
        "    device int* srow = sel + (size_t)row * K;\n"
        "    uint pos = gt_before + ((eq_before < need_eq) ? eq_before : need_eq);\n"
        if want_sel
        else ""
    )

    return f"""
    constexpr int NT = {_NT};
    constexpr int K  = {k};

    const uint tid  = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sgid = simdgroup_index_in_threadgroup;
    const uint row  = threadgroup_position_in_grid.y;

    const int   NB  = raw_shape[1];          // n_blocks
    const int   NW  = (NB + 31) / 32;        // ビットマップの word 数
    const float DIV = divisor[0];            // sqrt(head_dim)

    threadgroup atomic_uint hist[256];
    threadgroup uint tg_gt[NT / 32];
    threadgroup uint tg_eq[NT / 32];
    threadgroup uint tg_prefix;
    threadgroup uint tg_krem;
{cache_decl}{bits_decl}
    const device float* rrow = raw + (size_t)row * NB * 4;

    int nv = nvis[row];
    nv = (nv < 0) ? 0 : ((nv > NB) ? NB : nv);
    const int keff = (nv < K) ? nv : K;

    if (keff == 0) {{
{sel_fill}{bits_zero_out}        if (tid == 0) {{ cnt[row] = 0; }}
        return;
    }}

    // ---- 0. 出力の下地とヒストグラムの初期化 ----------------------------
{sel_fill}{bits_clear}    if (tid == 0) {{ cnt[row] = keff; }}
    for (int b = (int)tid; b < 256; b += NT) {{
        atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // ---- 1. キーを作りながら pass 0 のヒストグラムを取る ----------------
    for (int i = (int)tid; i < nv; i += NT) {{
        uint key = mlxturbo_qsa_key(rrow + (size_t)i * 4, DIV);
{cache_store}        atomic_fetch_add_explicit(&hist[key >> 24], 1u, memory_order_relaxed);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 2. radix select: 上位 8 bit から 4 pass で k 番目のキーを出す ---
    uint prefix = 0u;
    uint krem   = (uint)keff;
    uint himask = 0u;
    for (int p = 0; p < 4; ++p) {{
        const uint shift = (uint)(24 - 8 * p);
        if (p > 0) {{
            for (int b = (int)tid; b < 256; b += NT) {{
                atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = (int)tid; i < nv; i += NT) {{
                uint key = {key_of};
                if ((key & himask) == prefix) {{
                    atomic_fetch_add_explicit(
                        &hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // 256 bin を降順に走って「k 番目が入る bin」を決める。
        // simdgroup 0 の lane l が bin (248-8l)..(255-8l) を持つ
        if (sgid == 0) {{
            const uint base = 248u - 8u * lane;
            uint s = 0u;
            for (int t = 0; t < 8; ++t) {{
                s += atomic_load_explicit(&hist[base + (uint)t], memory_order_relaxed);
            }}
            const uint excl = simd_prefix_exclusive_sum(s);
            uint chosen = 0u;
            uint higher = 0u;
            const bool hit = (excl < krem) && ((excl + s) >= krem);
            if (hit) {{
                uint run = excl;
                for (int t = 7; t >= 0; --t) {{
                    uint c = atomic_load_explicit(
                        &hist[base + (uint)t], memory_order_relaxed);
                    if (run + c >= krem) {{ chosen = base + (uint)t; higher = run; break; }}
                    run += c;
                }}
            }}
            uint hl = simd_min(hit ? lane : 32u);
            if (hl > 31u) {{ hl = 0u; }}
            chosen = simd_shuffle(chosen, (ushort)hl);
            higher = simd_shuffle(higher, (ushort)hl);
            if (lane == 0u) {{
                tg_prefix = prefix | (chosen << shift);
                tg_krem   = krem - higher;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix  = tg_prefix;
        krem    = tg_krem;
        himask |= (255u << shift);
    }}

    // ---- 3. `> t` を全部 + `== t` を添字の昇順で不足分だけ ---------------
    //
    // 添字の昇順で書けるように、ここだけスレッドに**連続**した区間を割る。
    // 選ばれる本数の走査前の累積 (gt_before / eq_before) が分かれば、
    // 各スレッドは自分の書き出し位置を独立に決められる。
    const uint thr     = prefix;
    const uint need_eq = krem;

    const int C  = (nv + NT - 1) / NT;
    int c0 = (int)tid * C;  if (c0 > nv) {{ c0 = nv; }}
    int c1 = c0 + C;        if (c1 > nv) {{ c1 = nv; }}

    uint ngt = 0u;
    uint neq = 0u;
    for (int i = c0; i < c1; ++i) {{
        uint key = {key_of};
        if (key > thr) {{ ngt++; }}
        else if (key == thr) {{ neq++; }}
    }}
    const uint egt = simd_prefix_exclusive_sum(ngt);
    const uint eeq = simd_prefix_exclusive_sum(neq);
    if (lane == 31u) {{ tg_gt[sgid] = egt + ngt; tg_eq[sgid] = eeq + neq; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint bgt = 0u;
    uint beq = 0u;
    for (uint g = 0u; g < sgid; ++g) {{ bgt += tg_gt[g]; beq += tg_eq[g]; }}
    const uint gt_before = bgt + egt;
    const uint eq_before = beq + eeq;

    uint erank = eq_before;
{sel_row}    for (int i = c0; i < c1; ++i) {{
        uint key = {key_of};
        bool take = false;
        if (key > thr) {{ take = true; }}
        else if (key == thr) {{ take = (erank < need_eq); erank++; }}
        if (take) {{
{sel_write}{bits_set}        }}
    }}
{bits_flush}"""


def _get_kernel(k: int, cap: int, nwmax: int, mode: str):
    key = (k, cap, nwmax, mode)
    entry = _KERNELS.get(key)
    if entry is None:
        names = []
        dtypes = []
        if mode in ("idx", "both"):
            names.append("sel")
            dtypes.append(mx.int32)
        if mode in ("bits", "both"):
            names.append("bits")
            dtypes.append(mx.uint32)
        names.append("cnt")
        dtypes.append(mx.int32)
        kern = mx.fast.metal_kernel(
            name=f"mlxturbo_qsa_select_{k}_{cap}_{nwmax}_{mode}",
            input_names=["raw", "nvis", "divisor"],
            output_names=names,
            source=_source(k, cap, nwmax, mode),
            header=_HEADER,
        )
        entry = (kern, tuple(dtypes))
        _KERNELS[key] = entry
    return entry


def _divisor(head_dim: int) -> mx.array:
    d = math.sqrt(head_dim)
    arr = _DIVISORS.get(d)
    if arr is None:
        arr = mx.array([d], dtype=mx.float32)
        _DIVISORS[d] = arr
    return arr


def visible_counts(q_col: mx.array, compress_ratio: int, n_blocks: int) -> mx.array:
    """行ごとの可視ブロック数を返す (int32 の (S,))。

    本家の ``visible = block_end <= q_col`` は
    ``n * cr + cr - 1 <= q_col`` すなわち ``n < (q_col + 1) // cr`` なので、
    可視ブロックは常に先頭からの連続範囲になる。

    **配線では使わないこと。**GPU の op が 3 本増えて、それだけで K2a の
    予算の半分を食う。本番の ``q_col`` は ``mx.arange(offset, offset + S)``
    で値が Python 側で分かっているので `visible_counts_host` を使う。
    こちらは検証と、``q_col`` が配列でしか手に入らない場合の口。
    """
    n = (q_col.astype(mx.int32) + 1) // compress_ratio
    return mx.minimum(n, n_blocks).astype(mx.int32)


def visible_counts_host(
    offset: int, s_len: int, compress_ratio: int, n_blocks: int
) -> mx.array:
    """`visible_counts` と同じものを、GPU の op を使わずに作る。

    ``q_col = mx.arange(offset, offset + S)`` の値は Python 側で分かるので、
    ホストで数えて配列にするだけで済む (`_pooled_and_top` の配線はこちら)。
    """
    return mx.array(
        [min(n_blocks, (offset + s + 1) // compress_ratio) for s in range(s_len)],
        dtype=mx.int32,
    )


def eligible(raw: mx.array, n_vis: mx.array, k: int) -> bool:
    """このカーネルで扱える形か。外れたら呼び出し側は素の argpartition へ。"""

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    if raw.dtype != mx.float32 or raw.ndim < 3:
        return False
    if raw.shape[-1] != 4:
        return False
    n_blocks = raw.shape[-2]
    if n_blocks < 1 or n_blocks > MAX_BLOCKS:
        return False
    if k < 1 or k > n_blocks:
        return False
    if n_vis.ndim != 1 or n_vis.shape[0] != prod(raw.shape[:-2]):
        return False
    return True


def select(
    raw: mx.array,
    n_vis: mx.array,
    k: int,
    head_dim: int = 128,
    mode: str = "idx",
):
    """ブロックスコアの生値から top-k ブロックを選ぶ。

    ``raw``: (..., n_blocks, 4) fp32 = einsum ``"bshd,bnd->bsnh"`` の出力。
    ``n_vis``: (rows,) int32 = 行ごとの可視ブロック数 (`visible_counts`)。
    ``k``: 取る本数 (本家の ``min(block_topk, n_blocks)``)。
    ``mode``: ``"idx"`` / ``"bits"`` / ``"both"``。

    戻り値は ``mode`` に応じて ``(sel, cnt)`` / ``(bits, cnt)`` /
    ``(sel, bits, cnt)``。先頭の 2 軸は ``raw`` のものを復元する。
    """

    if mode not in ("idx", "bits", "both"):
        raise ValueError(f"mode は idx/bits/both のどれか: {mode!r}")
    if raw.ndim < 3 or raw.shape[-1] != 4 or raw.dtype != mx.float32:
        # カーネルは indexer_n_heads=4 の fp32 を前提に和を展開してある
        raise ValueError(f"raw は (..., n_blocks, 4) fp32: {raw.shape} {raw.dtype}")
    if raw.shape[-2] > MAX_BLOCKS:
        raise ValueError(f"n_blocks は {MAX_BLOCKS} まで: {raw.shape[-2]}")

    _fire.bump("qsa_select")
    lead = raw.shape[:-2]
    n_blocks = raw.shape[-2]
    rows = prod(lead) if lead else 1
    nw = (n_blocks + 31) // 32
    cap = _CACHE_CAP if n_blocks <= _CACHE_CAP else 0
    nwmax = (cap + 31) // 32 if cap else (MAX_BLOCKS + 31) // 32

    kern, dtypes = _get_kernel(k, cap, nwmax, mode)

    shapes = []
    if mode in ("idx", "both"):
        shapes.append((rows, k))
    if mode in ("bits", "both"):
        shapes.append((rows, nw))
    shapes.append((rows,))

    flat = raw.reshape((rows, n_blocks, 4))
    outs = kern(
        inputs=[flat, n_vis, _divisor(head_dim)],
        grid=(_NT, rows, 1),
        threadgroup=(_NT, 1, 1),
        output_shapes=shapes,
        output_dtypes=list(dtypes),
    )
    return tuple(o.reshape((*lead, *o.shape[1:])) for o in outs)


__all__ = [
    "MAX_BLOCKS",
    "eligible",
    "select",
    "visible_counts",
    "visible_counts_host",
]
