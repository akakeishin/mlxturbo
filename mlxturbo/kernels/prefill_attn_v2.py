"""レーン 3 (多日) の再挑戦: prefill QSA attention の候補カーネル。

`mlxturbo/kernels/prefill_attn.py` (T=1、クエリ 1 本 x 1 kv head、dense と
同速だった初版) の再設計。`docs/research/SESSION-2026-09-02-CATCHUP.md` の
「レーン 3 の判定の訂正」により、T=4/8 の union 化 (畳んだ)
ではなく **T=1 のまま実装効率を上げる**方針に戻っている。この判定の根拠は
`docs/research/QSA-PREFILL-KERNEL-DESIGN.md` の仮説 1-4:

- 仮説 2: タイル幅 (既定 16 列) が狭く、バリア比率が悪い可能性
- 仮説 3: per-column scalar dot + simd_sum が Apple GPU の行列演算パスより遅い
- 仮説 4: bool→添字圧縮 (3 パス) が同じクエリで kv head ごとに重複している

**このファイルは実験用の並行モジュール。**本番の `Attention._gather_forward`
からはまだ呼ばれない (`prefill_attn.py` の `_prefill_attn` フラグ・呼び出し口
は変更していない)。`tools/qsa_gather_micro.py` から呼ばれる。

## タイル幅の物理制約 (仮説 2 の検討で判明)

課題文が挙げた「タイルを 64〜128 列に」は head_dim=256/bf16 では threadgroup
メモリの壁 (Apple GPU 32KB、``prefill_attn.MAX_TG_BYTES`` = 30KB の安全域) に
収まらない。K+V タイルは列あたり ``2 * head_dim * itemsize`` = 1024 バイト
必要で、64 列で 65536 バイト (壁の 2 倍超)。``tg_sel``/``tg_cnt`` の予約
(4*block_topk + 4*gqa ≈ 2096 バイト) を差し引くと、実際に載る最大列数は
**24 列 (bb=6)** --- 現行 16 列の 1.5 倍が上限で、「64〜128 列」は不可能。
``_tile_blocks_target`` は 1 刻みで最大 bb を探す (現行 `_tile_blocks` は
2 の冪で半減するだけなので、64 を渡しても 16 に潰れて何も変わらない)。

## stage 別カーネル (内訳計測用)

``prefill_attn_stage(stage, ...)``:

- ``"compact"``: bool → 添字圧縮 (3 パス) だけ。出力は ``total`` (選択数) を
  broadcast しただけの値 (コンパイラに処理を消されないための最小限の書き込み)。
- ``"compact_load"``: 上記 + K/V を threadgroup メモリへタイルごとに load
  する部分だけ (score 計算は無く、load した値の単純な和を書く)。
- ``"full"``: 完全な attention (現行カーネルと同じ算法、online softmax)。
  ``prefill_attn_v2`` (候補カーネル本体) はこの stage を
  ``TARGET_COLS`` (既定 24、環境変数 ``MLXTURBO_QSA_V2_TARGET_COLS`` で
  上書き可) で呼ぶだけの薄いラッパー。
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

from . import prefill_attn as PA

_KERNELS: dict[tuple, Any] = {}

# 候補 (i): タイル幅。既定は物理上限の 24 列 (bb=6)。
# 環境変数で振って `tools/qsa_gather_micro.py` から A/B するための口。
TARGET_COLS = int(os.environ.get("MLXTURBO_QSA_V2_TARGET_COLS", "24"))

# "tile"  = 候補 (i) だけ (現行と同じ 3 パス圧縮 + 広いタイル)
# "direct" = 候補 (i) + (iv) (host 側 1 回の mx.sort、3 パス圧縮をカーネルから除去)
MODE = os.environ.get("MLXTURBO_QSA_V2_MODE", "tile")

# "direct" は tg_sel/tg_cnt が要らない分、"tile" より広いタイルが物理的に載る
# (24 列 → 28 列)。既定の要求値を分けておく (`_tile_blocks_target` が実際の
# 上限まで自動で切り詰めるので、大きめに要求して壁に当てさせる)。
TARGET_COLS_DIRECT = int(os.environ.get("MLXTURBO_QSA_V2_TARGET_COLS_DIRECT", "32"))


def _tile_blocks_target(
    cr: int, head_dim: int, itemsize: int, gqa: int, kmax: int, target_cols: int
) -> int:
    """1 刻みで、threadgroup メモリに収まる最大の ``bb`` を探す。

    現行 `prefill_attn._tile_blocks` は 2 の冪で半減するだけなので、
    target_cols=64 を渡しても 16 (bb=4) まで一気に落ちて何も変わらない。
    ここでは K/V タイル + tg_sel/tg_cnt の予約を合わせて壁を超えない
    最大値を 1 列刻みで探す。
    """
    reserve = 4 * kmax + 4 * gqa  # tg_sel + tg_cnt
    bb = max(1, target_cols // cr)
    while bb > 1 and 2 * bb * cr * head_dim * itemsize + reserve > PA.MAX_TG_BYTES:
        bb -= 1
    return bb


def _source(head_dim, gqa, cr, bb, kmax, scale, stage: str, sw: int | None = None) -> str:
    d = head_dim
    dpl = (d + 31) // 32
    bk = bb * cr
    nsg = gqa
    nth = nsg * 32
    # 候補: load タイル幅 (bk) と score/softmax の register チャンク幅 (sw) を
    # 分離する。breakdown 実測 (2026-09-03) で bk を広げると load は速くなるが
    # score+softmax は `sc[bk]` の register 圧で遅くなった (bb=4: 17.12ms,
    # bb=6: 18.93ms、kv=16896)。sw < bk なら online softmax のまま (どの粒度
    # でも数学的に等価) tile 内をさらに sw 幅で細分し、score 側だけ register を
    # 絞る --- barrier は tile 境界 (bk 幅) のままなので load 側の得は保つ。
    if sw is None or sw >= bk:
        sw = bk

    header = f"""
    const int S      = params[0];
    const int NB     = params[1];
    const int KVLEN  = params[2];
    const int CAP    = params[3];
    const int OFFSET = params[4];
    const int NKV    = params[5];

    const uint tid   = thread_position_in_threadgroup.x;
    const uint sg    = simdgroup_index_in_threadgroup;
    const uint lane  = thread_index_in_simdgroup;
    const int  s     = (int)threadgroup_position_in_grid.y;
    const int  bh    = (int)threadgroup_position_in_grid.z;
    const int  b     = bh / NKV;
    const int  kvh   = bh - b * NKV;
    const int  H     = NKV * {nsg};
    const int  h     = kvh * {nsg} + (int)sg;
    const int  q_col = OFFSET + s;

    threadgroup int tg_sel[{kmax}];
    threadgroup int tg_cnt[{nsg}];

    const device bool* keep_row = keep + ((size_t)b * S + s) * (size_t)NB;

    // --- 1) keep_block を昇順のまま詰める (3 パス) --------------------
    int chunk = (NB + {nsg} - 1) / {nsg};
    chunk = ((chunk + 31) / 32) * 32;
    const int lo = (int)sg * chunk;
    const int hi = min(lo + chunk, NB);

    int cnt = 0;
    for (int base = lo; base < hi; base += 32) {{
        const int idx = base + (int)lane;
        const uint f = (idx < hi && keep_row[idx]) ? 1u : 0u;
        cnt += (int)simd_sum(f);
    }}
    if (lane == 0) {{ tg_cnt[sg] = cnt; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int base_off = 0;
    for (uint g = 0; g < sg; ++g) {{ base_off += tg_cnt[g]; }}
    int total = 0;
    for (int g = 0; g < {nsg}; ++g) {{ total += tg_cnt[g]; }}
    total = min(total, {kmax});

    int w = base_off;
    for (int base = lo; base < hi; base += 32) {{
        const int idx = base + (int)lane;
        const uint f = (idx < hi && keep_row[idx]) ? 1u : 0u;
        const uint pre = simd_prefix_exclusive_sum(f);
        if (f != 0u) {{
            const int slot = w + (int)pre;
            if (slot < {kmax}) {{ tg_sel[slot] = idx; }}
        }}
        w += (int)simd_sum(f);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

    tail_calc = f"""
    const int tail_base = NB * {cr};
    const int ntail = KVLEN - tail_base;
    int ntail_vis = q_col - tail_base + 1;
    ntail_vis = max(0, min(ntail_vis, ntail));
"""

    if stage == "compact":
        body = header + f"""
    // --- stage=compact: ここで終わり。total を書いて DCE を防ぐ -------
    device T* orow = out + (((size_t)b * S + s) * H + h) * {d};
    const float val = (float)total;
    for (int t = 0; t < {dpl}; ++t) {{
        const int dd = (int)lane + 32 * t;
        if (dd < {d}) {{ orow[dd] = (T)val; }}
    }}
"""
        return body

    if stage == "compact_load":
        body = header + tail_calc + f"""
    threadgroup T tg_k[{bk} * {d}];
    threadgroup T tg_v[{bk} * {d}];

    const device T* kbase = k + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};
    const device T* vbase = v + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};

    float acc[{dpl}];
    for (int t = 0; t < {dpl}; ++t) {{ acc[t] = 0.0f; }}

    const int ntiles = (total + {bb} - 1) / {bb};
    const int ttiles = (ntail_vis + {bk} - 1) / {bk};

    for (int ti = 0; ti < ntiles + ttiles; ++ti) {{
        const bool is_tail = ti >= ntiles;
        const int t0 = is_tail ? 0 : ti * {bb};
        const int tb = is_tail ? (ti - ntiles) * {bk} : 0;
        const int ncol = is_tail ? min({bk}, ntail_vis - tb)
                                 : min({bb}, total - t0) * {cr};

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = (int)tid; e < ncol * {d}; e += {nth}) {{
            const int cc = e / {d};
            const int dd = e - cc * {d};
            const int col = is_tail
                ? (tail_base + tb + cc)
                : (tg_sel[t0 + cc / {cr}] * {cr} + (cc % {cr}));
            tg_k[e] = kbase[(size_t)col * {d} + dd];
            tg_v[e] = vbase[(size_t)col * {d} + dd];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // score は計算しない --- load した値の単純和だけ (DCE 防止、
        // かつ score/softmax の費用を含めないための代用)
        for (int j = 0; j < {bk}; ++j) {{
            if (j >= ncol) {{ continue; }}
            for (int t = 0; t < {dpl}; ++t) {{
                const int dd = (int)lane + 32 * t;
                if (dd < {d}) {{
                    acc[t] += (float)tg_k[j * {d} + dd] + (float)tg_v[j * {d} + dd];
                }}
            }}
        }}
    }}

    device T* orow = out + (((size_t)b * S + s) * H + h) * {d};
    for (int t = 0; t < {dpl}; ++t) {{
        const int dd = (int)lane + 32 * t;
        if (dd < {d}) {{ orow[dd] = (T)acc[t]; }}
    }}
"""
        return body

    if stage == "full":
        body = header + tail_calc + f"""
    const device T* qrow = q + (((size_t)b * S + s) * H + h) * {d};
    float qv[{dpl}];
    float acc[{dpl}];
    for (int t = 0; t < {dpl}; ++t) {{
        const int dd = (int)lane + 32 * t;
        qv[t] = (dd < {d}) ? (float)qrow[dd] : 0.0f;
        acc[t] = 0.0f;
    }}
    float m = -INFINITY;
    float l = 0.0f;

    threadgroup T tg_k[{bk} * {d}];
    threadgroup T tg_v[{bk} * {d}];

    const device T* kbase = k + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};
    const device T* vbase = v + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};

    const int ntiles = (total + {bb} - 1) / {bb};
    const int ttiles = (ntail_vis + {bk} - 1) / {bk};

    for (int ti = 0; ti < ntiles + ttiles; ++ti) {{
        const bool is_tail = ti >= ntiles;
        const int t0 = is_tail ? 0 : ti * {bb};
        const int tb = is_tail ? (ti - ntiles) * {bk} : 0;
        const int ncol = is_tail ? min({bk}, ntail_vis - tb)
                                 : min({bb}, total - t0) * {cr};

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = (int)tid; e < ncol * {d}; e += {nth}) {{
            const int cc = e / {d};
            const int dd = e - cc * {d};
            const int col = is_tail
                ? (tail_base + tb + cc)
                : (tg_sel[t0 + cc / {cr}] * {cr} + (cc % {cr}));
            tg_k[e] = kbase[(size_t)col * {d} + dd];
            tg_v[e] = vbase[(size_t)col * {d} + dd];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load タイル (幅 {bk}) の中を score チャンク幅 {sw} でさらに分割する
        // (候補: load/score の register 幅を分離、barrier は増やさない --- 直上の
        // docstring 参照)。online softmax はどの粒度でチャンクしても数学的に
        // 等価 (m, l, acc は tile 境界と同じ役割をチャンク境界でも果たすだけ)。
        for (int jb = 0; jb < {bk}; jb += {sw}) {{
            float sc[{sw}];
            float mt = -INFINITY;
            for (int jj = 0; jj < {sw}; ++jj) {{
                const int j = jb + jj;
                float p = 0.0f;
                if (j < ncol) {{
                    for (int t = 0; t < {dpl}; ++t) {{
                        const int dd = (int)lane + 32 * t;
                        if (dd < {d}) {{ p += qv[t] * (float)tg_k[j * {d} + dd]; }}
                    }}
                }}
                const float sj = simd_sum(p) * {scale!r}f;
                sc[jj] = (j < ncol) ? sj : -INFINITY;
                mt = metal::max(mt, sc[jj]);
            }}

            const float mnew = metal::max(m, mt);
            const float corr = metal::exp(m - mnew);
            for (int t = 0; t < {dpl}; ++t) {{ acc[t] *= corr; }}
            l *= corr;
            for (int jj = 0; jj < {sw}; ++jj) {{
                const int j = jb + jj;
                if (j >= ncol) {{ continue; }}
                const float p = metal::exp(sc[jj] - mnew);
                l += p;
                for (int t = 0; t < {dpl}; ++t) {{
                    const int dd = (int)lane + 32 * t;
                    if (dd < {d}) {{ acc[t] += p * (float)tg_v[j * {d} + dd]; }}
                }}
            }}
            m = mnew;
        }}
    }}

    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    device T* orow = out + (((size_t)b * S + s) * H + h) * {d};
    for (int t = 0; t < {dpl}; ++t) {{
        const int dd = (int)lane + 32 * t;
        if (dd < {d}) {{ orow[dd] = (T)(acc[t] * inv); }}
    }}
"""
        return body

    raise ValueError(f"未知の stage: {stage!r}")


def _source_direct(head_dim, gqa, cr, bb, kmax, scale) -> str:
    """候補 (iv): 3 パスの bool→添字圧縮をカーネルから外し、``row_idx``
    (host 側で ``mx.sort`` 1 回だけ作った (B,S,block_topk) の昇順添字) を
    直接読む版。kv head ごとの重複計算 (仮説 4) が消え、``tg_sel``/``tg_cnt``
    の threadgroup メモリも要らなくなる分、K/V タイルを少し広げられる。

    **簡略化**: 呼び出し側 (`build_row_idx`) が「全行が block_topk 本ちょうど
    選んでいる」ことを保証する前提で、sentinel (無効添字 = n_blocks) の
    列ごとマスクは実装していない。この前提は `prefill_attn_v2.py` の
    合成マイクロベンチの kv/S の組 (offset が十分大きい) では常に成り立つが、
    本番配線 (kv_len が budget 付近で可視ブロック数が block_topk 未満の行が
    混じる場合) にはそのまま使えない --- 配線するなら先に対応すること。
    """
    d = head_dim
    dpl = (d + 31) // 32
    bk = bb * cr
    nsg = gqa
    nth = nsg * 32

    return f"""
    const int S      = params[0];
    const int NB     = params[1];
    const int KVLEN  = params[2];
    const int CAP    = params[3];
    const int OFFSET = params[4];
    const int NKV    = params[5];
    const int TOPK   = params[6];

    const uint tid   = thread_position_in_threadgroup.x;
    const uint sg    = simdgroup_index_in_threadgroup;
    const uint lane  = thread_index_in_simdgroup;
    const int  s     = (int)threadgroup_position_in_grid.y;
    const int  bh    = (int)threadgroup_position_in_grid.z;
    const int  b     = bh / NKV;
    const int  kvh   = bh - b * NKV;
    const int  H     = NKV * {nsg};
    const int  h     = kvh * {nsg} + (int)sg;
    const int  q_col = OFFSET + s;

    const device int* row = row_idx + ((size_t)b * S + s) * (size_t)TOPK;

    const int tail_base = NB * {cr};
    const int ntail = KVLEN - tail_base;
    int ntail_vis = q_col - tail_base + 1;
    ntail_vis = max(0, min(ntail_vis, ntail));

    const device T* qrow = q + (((size_t)b * S + s) * H + h) * {d};
    float qv[{dpl}];
    float acc[{dpl}];
    for (int t = 0; t < {dpl}; ++t) {{
        const int dd = (int)lane + 32 * t;
        qv[t] = (dd < {d}) ? (float)qrow[dd] : 0.0f;
        acc[t] = 0.0f;
    }}
    float m = -INFINITY;
    float l = 0.0f;

    threadgroup T tg_k[{bk} * {d}];
    threadgroup T tg_v[{bk} * {d}];

    const device T* kbase = k + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};
    const device T* vbase = v + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};

    const int ntiles = (TOPK + {bb} - 1) / {bb};
    const int ttiles = (ntail_vis + {bk} - 1) / {bk};

    for (int ti = 0; ti < ntiles + ttiles; ++ti) {{
        const bool is_tail = ti >= ntiles;
        const int t0 = is_tail ? 0 : ti * {bb};
        const int tb = is_tail ? (ti - ntiles) * {bk} : 0;
        const int ncol = is_tail ? min({bk}, ntail_vis - tb)
                                 : min({bb}, TOPK - t0) * {cr};

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = (int)tid; e < ncol * {d}; e += {nth}) {{
            const int cc = e / {d};
            const int dd = e - cc * {d};
            const int col = is_tail
                ? (tail_base + tb + cc)
                : (row[t0 + cc / {cr}] * {cr} + (cc % {cr}));
            tg_k[e] = kbase[(size_t)col * {d} + dd];
            tg_v[e] = vbase[(size_t)col * {d} + dd];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sc[{bk}];
        float mt = -INFINITY;
        for (int j = 0; j < {bk}; ++j) {{
            float p = 0.0f;
            if (j < ncol) {{
                for (int t = 0; t < {dpl}; ++t) {{
                    const int dd = (int)lane + 32 * t;
                    if (dd < {d}) {{ p += qv[t] * (float)tg_k[j * {d} + dd]; }}
                }}
            }}
            const float sj = simd_sum(p) * {scale!r}f;
            sc[j] = (j < ncol) ? sj : -INFINITY;
            mt = metal::max(mt, sc[j]);
        }}

        const float mnew = metal::max(m, mt);
        const float corr = metal::exp(m - mnew);
        for (int t = 0; t < {dpl}; ++t) {{ acc[t] *= corr; }}
        l *= corr;
        for (int j = 0; j < {bk}; ++j) {{
            if (j >= ncol) {{ continue; }}
            const float p = metal::exp(sc[j] - mnew);
            l += p;
            for (int t = 0; t < {dpl}; ++t) {{
                const int dd = (int)lane + 32 * t;
                if (dd < {d}) {{ acc[t] += p * (float)tg_v[j * {d} + dd]; }}
            }}
        }}
        m = mnew;
    }}

    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    device T* orow = out + (((size_t)b * S + s) * H + h) * {d};
    for (int t = 0; t < {dpl}; ++t) {{
        const int dd = (int)lane + 32 * t;
        if (dd < {d}) {{ orow[dd] = (T)(acc[t] * inv); }}
    }}
"""


def build_row_idx(keep_block: mx.array, n_blocks: int, block_topk: int) -> mx.array:
    """候補 (iv) の host 側添字構築。``mx.sort`` 1 回だけ (kv head で共有)。

    `keep_block` (B,S,n_blocks) の True 位置のブロック添字を昇順に詰め、
    先頭 ``block_topk`` 本を返す (padding は sentinel ``n_blocks``)。
    呼び出し側は `assert_no_padding` で全行 block_topk ちょうどであることを
    確かめてからカーネルへ渡すこと (直下参照)。
    """
    key = mx.where(
        keep_block, mx.arange(n_blocks, dtype=mx.int32)[None, None, :], n_blocks
    )
    return mx.sort(key, axis=-1)[..., :block_topk].astype(mx.int32)


def assert_no_padding(row_idx: mx.array, n_blocks: int) -> None:
    """全行が ``block_topk`` 本ちょうど選んでいるか (sentinel が無いか) を確認する。

    host 同期 (`.item()`) が 1 回入るので、呼び出しは検査用途 (呼び出し前の
    1 回だけ) に限る --- ホットパスの毎回では呼ばない。
    """
    ok = bool(mx.all(row_idx[..., -1] < n_blocks).item())
    if not ok:
        raise ValueError(
            "row_idx に sentinel (padding) が混じっている --- "
            "_source_direct はこの前提を検査していない (未対応)"
        )


_KERNELS_DIRECT: dict[tuple, Any] = {}


def _get_kernel_direct(head_dim, gqa, cr, bb, kmax, scale):
    key = (head_dim, gqa, cr, bb, kmax, scale)
    kern = _KERNELS_DIRECT.get(key)
    if kern is None:
        kern = mx.fast.metal_kernel(
            name=f"prefill_attn_direct_{head_dim}_{gqa}_{cr}_{bb}_{kmax}",
            input_names=["q", "k", "v", "row_idx", "params"],
            output_names=["out"],
            source=_source_direct(head_dim, gqa, cr, bb, kmax, scale),
        )
        _KERNELS_DIRECT[key] = kern
    return kern


def prefill_attn_direct(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    keep_block: mx.array,
    cache,
    *,
    cr: int,
    kv_len: int,
    n_blocks: int,
    block_topk: int,
    offset: int,
    scale: float,
    target_cols: int | None = None,
) -> mx.array:
    """候補 (iv): host 側 1 回の ``mx.sort`` で作った添字を直接読むカーネル。"""
    B, n_heads, S, head_dim = q.shape
    n_kv = k.shape[1]
    gqa = n_heads // n_kv
    itemsize = q.dtype.size
    tc = TARGET_COLS_DIRECT if target_cols is None else target_cols
    # tg_sel/tg_cnt が要らない分、reserve をほぼ 0 にして bb を探す
    bb = _tile_blocks_target(cr, head_dim, itemsize, 0, 0, tc)

    keys, values, cap = PA._kv_buffers(cache, k, v, kv_len)

    row_idx = build_row_idx(keep_block, n_blocks, block_topk)
    assert_no_padding(row_idx, n_blocks)

    q_bshd = q.transpose(0, 2, 1, 3)
    params = mx.array(
        [S, n_blocks, kv_len, cap, offset, n_kv, block_topk], dtype=mx.int32
    )

    kernel = _get_kernel_direct(head_dim, gqa, cr, bb, block_topk, float(scale))
    (out,) = kernel(
        inputs=[q_bshd, keys, values, row_idx, params],
        template=[("T", q.dtype)],
        grid=(32 * gqa, S, B * n_kv),
        threadgroup=(32 * gqa, 1, 1),
        output_shapes=[(B, S, n_heads, head_dim)],
        output_dtypes=[q.dtype],
    )
    return out


def _get_kernel(head_dim, gqa, cr, bb, kmax, scale, stage: str, sw: int | None = None):
    key = (head_dim, gqa, cr, bb, kmax, scale, stage, sw)
    kern = _KERNELS.get(key)
    if kern is None:
        kern = mx.fast.metal_kernel(
            name=f"prefill_attn_v2_{stage}_{head_dim}_{gqa}_{cr}_{bb}_{kmax}_{sw or bb*cr}",
            input_names=["q", "k", "v", "keep", "params"],
            output_names=["out"],
            source=_source(head_dim, gqa, cr, bb, kmax, scale, stage, sw),
        )
        _KERNELS[key] = kern
    return kern


# 候補: score/softmax の register チャンク幅 (bk とは独立)。
# **実測 (2026-09-03, kv=16896): sw=16 (bk=24 を分割) は sw=bk (分割無し) より
# 遅い** (64.18ms 対 56.15ms --- チャンクごとの online softmax 補正
# (m/l の再スケールと acc 更新) が二重になる分がレジスタ圧の得を上回る)。
# 既定は「分割しない」(sw>=bk になる大きな値) --- `_source` の
# `if sw is None or sw >= bk: sw = bk` がそのケースを畳む。
SCORE_CHUNK = int(os.environ.get("MLXTURBO_QSA_V2_SCORE_CHUNK", "1024"))


def prefill_attn_stage(
    stage: str,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    keep_block: mx.array,
    cache,
    *,
    cr: int,
    kv_len: int,
    n_blocks: int,
    block_topk: int,
    offset: int,
    scale: float,
    target_cols: int | None = None,
    score_chunk: int | None = None,
) -> mx.array:
    """内訳計測用の入口。``stage`` は "compact" / "compact_load" / "full"。"""
    B, n_heads, S, head_dim = q.shape
    n_kv = k.shape[1]
    gqa = n_heads // n_kv
    itemsize = q.dtype.size
    tc = TARGET_COLS if target_cols is None else target_cols
    bb = _tile_blocks_target(cr, head_dim, itemsize, gqa, block_topk, tc)
    sw = SCORE_CHUNK if score_chunk is None else score_chunk

    keys, values, cap = PA._kv_buffers(cache, k, v, kv_len)

    q_bshd = q.transpose(0, 2, 1, 3)
    params = mx.array([S, n_blocks, kv_len, cap, offset, n_kv], dtype=mx.int32)

    kernel = _get_kernel(
        head_dim, gqa, cr, bb, block_topk, float(scale), stage,
        sw if stage == "full" else None,
    )
    (out,) = kernel(
        inputs=[q_bshd, keys, values, keep_block, params],
        template=[("T", q.dtype)],
        grid=(32 * gqa, S, B * n_kv),
        threadgroup=(32 * gqa, 1, 1),
        output_shapes=[(B, S, n_heads, head_dim)],
        output_dtypes=[q.dtype],
    )
    return out


def prefill_attn_v2(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    keep_block: mx.array,
    cache,
    *,
    cr: int,
    kv_len: int,
    n_blocks: int,
    block_topk: int,
    offset: int,
    scale: float,
) -> mx.array:
    """候補カーネル本体。``MODE`` (環境変数 ``MLXTURBO_QSA_V2_MODE``) で切り替え:

    - "tile" (既定): 候補 (i) だけ (3 パス圧縮は現行のまま、タイル幅だけ拡張)
    - "direct": 候補 (i) + (iv) (host 側 1 回の mx.sort、3 パス圧縮を除去)
    """
    if MODE == "direct":
        return prefill_attn_direct(
            q, k, v, keep_block, cache,
            cr=cr, kv_len=kv_len, n_blocks=n_blocks, block_topk=block_topk,
            offset=offset, scale=scale,
        )
    return prefill_attn_stage(
        "full", q, k, v, keep_block, cache,
        cr=cr, kv_len=kv_len, n_blocks=n_blocks, block_topk=block_topk,
        offset=offset, scale=scale,
    )


__all__ = [
    "prefill_attn_stage", "prefill_attn_v2", "prefill_attn_direct",
    "build_row_idx", "TARGET_COLS", "MODE",
]
