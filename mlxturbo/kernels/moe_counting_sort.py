"""MoE の「行を専門家順に並べる」を決定的な計数ソートにする (P7 第 3 段)。

## 判定 (2026-09-04 04:51): **未配線、記録用。既定には入れない**

micro は通った (99.5 → 25.5 us/層、rows=8192 で 131 → 36、ビット一致、走行間で
決定的) が、素の argsort 一式の実費は 0.10〜0.13 ms/層で、BACKLOG の見込み
(-0.25 ms/層 = 8k -0.4%) は単発 `mx.eval` の投入・同期の往復 (165〜270 us) を
測っていた誤り。本番の prefill はグループ幅で層あたり 1 回しか通らないので
取り分は 8k で 48 × 95 us = 4.6 ms = **prefill の 0.04%** (in-model: 4k ±0 /
8k -0.7% / 17k -0.7% は ±1% の揺れの中、tok/round と head は完全一致)。
代金は `tables` カーネルが `moe_grouped_gemm.segment_tables` の式の写しを持つこと、
`_moe_combine_fold` にフック (`_MOE_SORT`) が 1 つ増えること (vendor のシームは
1 クラス 2 個までの決まり)。**測れない取り分に写しとシームを払わない**ので、配線
(`qwen4_exp.py` のフック、`fused._moe_fold_block` の extra、runner、decode_ab の
knob) は外し、カーネルと `tools/moe_sort_micro.py` だけ負の結果の記録として残す。
反転条件: prefill のチャンク幅が小さくなって層あたりのソート回数が増える構成
(短いチャンクの連続、バッチの prefill) では層あたり 1 回の前提が崩れる。


## 置き換える相手

`_moe_combine_fold` (`mlxturbo/_vendor/qwen4_exp.py`) の並べ替え一式:

    order   = mx.argsort(idx_flat)      # (M,) 20480 要素、値は 0..511
    idx_s   = idx_flat[order]
    row_src = order // top_k
    inv     = zeros(M).at[order].add(arange(M))     # fused._inv_perm

さらに `_moe_fold_block` (`mlxturbo/fused.py`) 側で
`counts_from_sorted_ids` + `segment_tables` の表作りが続く。48 層を 1 本の
コマンドバッファに積んで測ると (`tools/moe_sort_micro.py`、rows=2048 /
E=512 / top_k=10) 合わせて **99.5 us/層**で、内訳は argsort 一式 39.7 /
segment_tables 35.2 / counts 16.1 / inv_perm 11.6。値域が 0..511 と狭いので
計数ソートで足りる。ここは 25.5 us/層。

**単発 eval の絶対値 (`argsort` だけで 0.27 ms) は投入と同期の往復が床に
なっていて実費ではない** (`counts_from_sorted_ids` の scatter-add 1 本だけで
0.167 ms が出る)。本番は 48 層が 1 本のコマンドバッファに並ぶので、層ぶんを
積んで測った us/層 のほうを見ること。

## 中身 (3 カーネル + cumsum)

1. `hist`   : ブロック局所ヒストグラム。ブロック b (= BLOCK 要素) の専門家 e
              の本数を `hist[e][b]` に置く。(E, B) int32。
2. `cumsum` : `hist` を行優先で並べたまま **exclusive** 累積和を取る。
              行優先なので `offs[e*B + b]` は「専門家 e のブロック b の
              書き出し先頭」そのものになる (前の専門家を全部足した位置 +
              同じ専門家の前のブロックぶん)。mx の op 1 本。
3. `scatter`: ブロック内順位を出して 4 本 (`order` / `idx_s` / `row_src` /
              `inv`) を 1 パスで書く。
4. `tables` : `offs[e*B]` が専門家 e の行の先頭そのものなので、そこから
              `segment_tables` と同じ ``(row_start, tile_prefix)`` を 1
              threadgroup で作る (mx の小 op 7 本ぶん = 35 us/層 が消える)。

## 決定性 (原子操作なし)

ブロック内順位は **simdgroup の突き合わせ + simdgroup を順番に回す**で出す。

- 同じ simdgroup の 32 レーンのうち自分と同じ専門家のレーンは、キーを 32 回
  ブロードキャスト (`simd_shuffle`) して数える。自分より小さいレーンの本数が
  simdgroup 内の順位、先頭のレーンが「代表」。
- simdgroup どうしは `for (s = 0; s < NSG; ++s) { if (sgid == s) {...} barrier; }`
  で **番号順に**カーソル (threadgroup の `cur[E]`) を進める。原子操作を使うと
  実行順で並びが変わる (結果は変わらないが走行ごとに違う配列になる) ので使わない。

同じ入力なら必ず同じ `order` が出る。

## 出力がビット一致する理由

専門家の中での行の順は結果に影響しない。行ごとの GEMM は行の位置に依らず
(`qmm_segmented` は行単位の内積)、`moe_combine` は k の固定順で足すため。
`argsort` 版と一致させるのは「専門家ごとの行の集合」と「専門家の境界」だけで、
専門家内の並びは一致しなくてよい (`argsort` も MLX では安定ソートではない)。
合成の `SparseMoeBlock` で `_moe_fold_block` を通した出力がビット一致することは
確認済み (2026-09-04)。

## 配線

`mlxturbo.fused._moe_sort` (`MLXTURBO_MOE_COUNTING_SORT`)。既定 off。
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import _fire

# 1 threadgroup のスレッド数と 1 スレッドが持つ要素数。
# ブロックの長さは TG * CPT。M=20480 なら B=40 ブロック = 40 threadgroup で、
# M3 Max の 40 コアにちょうど 1 つずつ乗る。
TG = 128
CPT = 4
BLOCK = TG * CPT

# `cur[E]` を threadgroup メモリに置く上限 (uint32 個 = 4 * E バイト)。
MAX_EXPERTS = 2048

_KERNELS: dict[tuple, Any] = {}


_HIST_SOURCE = """
  const int M = dims[0];
  const int B = dims[1];

  const uint tid = thread_position_in_threadgroup.x;
  const uint b   = threadgroup_position_in_grid.x;

  threadgroup atomic_uint th[NE];
  for (uint e = tid; e < (uint)NE; e += TGSZ) {
    atomic_store_explicit(&th[e], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int i0 = (int)b * BLK;
  for (int j = 0; j < CPT_; ++j) {
    const int i = i0 + j * TGSZ + (int)tid;
    if (i < M) {
      atomic_fetch_add_explicit(&th[ids[i]], 1u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint e = tid; e < (uint)NE; e += TGSZ) {
    hist[(size_t)e * (size_t)B + (size_t)b] =
        (int)atomic_load_explicit(&th[e], memory_order_relaxed);
  }
"""


_SCATTER_SOURCE = """
  const int M = dims[0];
  const int B = dims[1];

  const uint tid  = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sgid = simdgroup_index_in_threadgroup;
  const uint b    = threadgroup_position_in_grid.x;

  // ブロック内のカーソル (専門家 e をここまでに何本置いたか)。
  threadgroup uint cur[NE];
  for (uint e = tid; e < (uint)NE; e += TGSZ) { cur[e] = 0u; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int i0 = (int)b * BLK;
  for (int j = 0; j < CPT_; ++j) {
    const int i = i0 + j * TGSZ + (int)tid;
    const bool ok = (i < M);
    // 範囲外は誰とも一致しない番兵にする (書き出しは ok で止める)
    const uint key = ok ? (uint)ids[i] : 0xFFFFFFFFu;

    // ---- simdgroup 内の順位と代表レーン (原子操作なし) ----------------
    uint rank = 0u;
    uint cnt  = 0u;
    uint lead = 32u;
    for (uint l = 0; l < 32u; ++l) {
      const uint k2 = simd_shuffle(key, (ushort)l);
      if (k2 == key) {
        if (l < lane) { rank++; }
        if (lead == 32u) { lead = l; }
        cnt++;
      }
    }

    // ---- simdgroup を番号順に回してカーソルを進める --------------------
    uint base = 0u;
    for (uint s = 0; s < (uint)NSG; ++s) {
      if (sgid == s) {
        base = ok ? cur[key] : 0u;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (ok && lane == lead) { cur[key] = base + cnt; }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (ok) {
      const uint pos =
          (uint)offs[(size_t)key * (size_t)B + (size_t)b] + base + rank;
      order[pos]   = (uint)i;
      idx_s[pos]   = key;
      row_src[pos] = (uint)(i / TOPK);
      inv[i]       = pos;
    }
  }
"""


_TABLES_SOURCE = """
  const int M = dims[0];
  const int B = dims[1];

  const uint tid = thread_position_in_threadgroup.x;

  threadgroup int rs_[NE + 1];
  threadgroup int tl_[NE + 1];

  for (uint e = tid; e <= (uint)NE; e += TGSZ) {
    const int rs = (e < (uint)NE) ? offs[(size_t)e * (size_t)B] : M;
    row_start[e] = rs;
    rs_[e] = rs;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // tiles[e] = ceil(counts[e] / bm_e)。bm_e は `segment_tables` と同じ比較
  // (`rows_e < MIX` なら 16 行タイル)。MIX == 0 が「混合なし」
  // (本数は必ず 0 以上なので `c < 0` は起きない)。
  for (uint e = tid; e < (uint)NE; e += TGSZ) {
    const int c = rs_[e + 1] - rs_[e];
    const int bm = (c < MIX) ? BM16 : BM32;
    tl_[e] = (c + bm - 1) / bm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // E は 2048 まで。1 スレッドの直列走査 (~2 us) で足りる
  if (tid == 0) {
    int run = 0;
    for (int e = 0; e < NE; ++e) {
      const int t = tl_[e];
      tl_[e] = run;
      run += t;
    }
    tl_[NE] = run;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e <= (uint)NE; e += TGSZ) { tile_prefix[e] = tl_[e]; }
"""


def _hist_kernel(n_experts: int):
    key = ("hist", n_experts)
    kern = _KERNELS.get(key)
    if kern is None:
        kern = mx.fast.metal_kernel(
            name=f"mlxturbo_moe_csort_hist_{n_experts}",
            input_names=["ids", "dims"],
            output_names=["hist"],
            source=_HIST_SOURCE,
            ensure_row_contiguous=True,
        )
        _KERNELS[key] = kern
    return kern


def _scatter_kernel(n_experts: int, top_k: int):
    key = ("scatter", n_experts, top_k)
    kern = _KERNELS.get(key)
    if kern is None:
        kern = mx.fast.metal_kernel(
            name=f"mlxturbo_moe_csort_scat_{n_experts}_{top_k}",
            input_names=["ids", "offs", "dims"],
            output_names=["order", "idx_s", "row_src", "inv"],
            source=_SCATTER_SOURCE,
            ensure_row_contiguous=True,
        )
        _KERNELS[key] = kern
    return kern


def _tables_kernel(n_experts: int):
    key = ("tables", n_experts)
    kern = _KERNELS.get(key)
    if kern is None:
        kern = mx.fast.metal_kernel(
            name=f"mlxturbo_moe_csort_tab_{n_experts}",
            input_names=["offs", "dims"],
            output_names=["row_start", "tile_prefix"],
            source=_TABLES_SOURCE,
            ensure_row_contiguous=True,
        )
        _KERNELS[key] = kern
    return kern


def eligible(ids: mx.array, n_experts: int, top_k: int) -> bool:
    """このカーネルで扱える形か。**host 同期はしない** (形と dtype だけ)。"""

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    if ids.ndim != 1 or ids.size == 0:
        return False
    if ids.dtype not in (mx.uint32, mx.int32):
        return False
    if not (1 <= n_experts <= MAX_EXPERTS):
        return False
    if top_k < 1 or ids.shape[0] % top_k != 0:
        return False
    return True


def sort_rows(ids: mx.array, n_experts: int, top_k: int,
              bm: int = 32, mix_threshold: int | None = None):
    """専門家添字 ``ids`` (M,) を専門家順に並べる材料を返す。

    戻り値 ``(order, idx_s, row_src, inv, tables)``:

      ``order``   (M,) uint32 : 並べ替え後の行 s が元の何番目か
      ``idx_s``   (M,) uint32 : 並べ替え後の専門家添字 (= ids[order])
      ``row_src`` (M,) uint32 : ``order // top_k``
      ``inv``     (M,) uint32 : ``order`` の逆置換 (= argsort(order))
      ``tables``  : ``(row_start, tile_prefix)`` = `moe_grouped_gemm`
                    の `segment_tables(counts, bm, mix_threshold)` と
                    **同じ 2 本** (どちらも (E+1,) int32)

    `mx.argsort` 版と一致するのは「専門家ごとの行の集合」と ``tables``。
    専門家内の並びは違ってよい (モジュールの docstring 参照)。
    """

    _fire.bump("moe_counting_sort")
    if ids.dtype != mx.uint32:
        ids = ids.astype(mx.uint32)
    M = ids.shape[0]
    B = (M + BLOCK - 1) // BLOCK
    E = int(n_experts)
    dims = mx.array([M, B], dtype=mx.int32)
    tpl = [("NE", E), ("TGSZ", TG), ("CPT_", CPT), ("BLK", BLOCK),
           ("NSG", TG // 32)]

    (hist,) = _hist_kernel(E)(
        inputs=[ids, dims],
        template=tpl,
        grid=(TG * B, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(E, B)],
        output_dtypes=[mx.int32],
    )
    offs = mx.cumsum(hist.reshape(-1), axis=0, inclusive=False)
    order, idx_s, row_src, inv = _scatter_kernel(E, int(top_k))(
        inputs=[ids, offs, dims],
        template=tpl + [("TOPK", int(top_k))],
        grid=(TG * B, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(M,), (M,), (M,), (M,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32],
    )
    # 表 (`segment_tables` と同じ 2 本) は offs から作る。専門家 e の先頭は
    # offs[e*B] (前の専門家を全部足した位置) なので、E+1 本の走査で済む
    from . import moe_grouped_gemm as mgg

    # `segment_tables` は混合モードのとき `bm` ではなくモジュール定数 BM を
    # 使う (16 行タイルと 32 行タイルの 2 択)。そこも含めて写す
    big = int(bm) if mix_threshold is None else mgg.BM
    tables = _tables_kernel(E)(
        inputs=[offs, dims],
        template=[("NE", E), ("TGSZ", 256), ("BM32", big),
                  ("BM16", mgg.BM16),
                  ("MIX", 0 if mix_threshold is None else int(mix_threshold))],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(E + 1,), (E + 1,)],
        output_dtypes=[mx.int32, mx.int32],
    )
    return order, idx_s, row_src, inv, tuple(tables)


__all__ = ["BLOCK", "CPT", "MAX_EXPERTS", "TG", "eligible", "sort_rows"]
