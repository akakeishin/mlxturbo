"""prefill の小 kv 帯 (kv < 8192) の QSA 注意を、スコアを実体化せずに 1 本の
Metal カーネルで済ませる (段 K1 arm A)。

`docs/research/IDEAS-2026-09-03.md` の K1 arm A。いまの本番経路 (段 P5、
`mlxturbo/fused.py` の `enable_sdpa_rowtile`) は q を 256 行ずつに割って
MLX の sdpa を呼ぶが、head_dim=256 では sdpa は steel ではなく fallback
(matmul -> where -> softmax -> matmul) に落ちるので、**(S, kv, n_heads) の
スコア行列を毎回書いて読み直している** (kv=4096 で 305 MB)。ここはその往復を
消し、online softmax でスコアを register / threadgroup に閉じ込める。

## 可視集合 (`prefill_attn.py` と同じ規約、こちらは gather しない)

- 完全ブロック: `QSAIndexer._select_keep` の `keep_block` (B, S, n_blocks) を
  そのまま読む。選ばれたブロックは `block_end <= q_col` を満たすので追加の
  因果チェックは要らない。
- 端数 (tail): `MLXTURBO_QSA_TAIL` の規約に従う。既定 (`query`) は
  クエリごとに列 `[cr*floor((q+1)/cr), q]`、`global` はブロック格子の外を
  因果性の範囲で。どちらも添字の算数で閉じる。

`prefill_attn` (段 P1) との違いは **集めるか流すか**。あちらは選んだ列だけを
threadgroup に集める (1 threadgroup = 1 クエリ) ので、小 kv では 1 列あたりの
仕事が足りず読み律速になる (kv=8192 で dense の 1.43 倍遅い)。こちらは
**列を順に流して行タイル (BQ 行) で共有し**、行タイルの union が空の列タイル
だけを飛ばす。演算強度は `gqa * BQ * 12` FLOP/byte なので BQ=4 でも 48 で、
機械の比 (~30) を超える。

## 形

- 1 threadgroup = (BQ クエリ行) x (1 kv head)、simdgroup 数 = GQA (12)。
  simdgroup `sg` が q head `kvh*gqa+sg` の BQ 行ぶん全部を持つ。
- 列方向に BK 列ずつのタイル。K と V を threadgroup メモリへ (uint4 段階 load)。
- スコアは `simd_sum` の内積。累算 (m / l / acc) は fp32、出力で入力 dtype へ。
- 出力は **(B, S, n_heads, head_dim)** (`prefill_attn` と同じ) なので呼び出し側は
  転置なしで `reshape(B, S, -1)` できる。

## 精度

参照 (fallback sdpa) とビット一致はしない (加算順が変わる)。可視集合は同じ。
累算が全部 fp32 なので、fp32 の素の attention に対する相対誤差は
**fallback より小さい** (2.2e-3 対 3.8e-3〜1.0e-2、`tools/flash_attn_d256_micro.py`)。

## 2026-09-04: 棄却 (未配線、記録用)

`tools/flash_attn_d256_micro.py` の冷 micro で、本番 P5 (行タイル R=256 +
fallback sdpa) に対し kv=2048/4096/6144/8192 で **2.45 / 2.26 / 1.85 / 1.39 倍
遅い**。判定線は kv=4096 で 0.75 倍だったので 3 倍届かない。

- **BQ (1 threadgroup が持つクエリ行数) を増やすほど遅い** (kv=4096, BK=16:
  bq1 39.8 / bq2 39.4 / bq4 46.5 / bq8 92.5 ms)。BQ は K/V の読み量を 1/BQ に
  する唯一のレバーなのに、head_dim 256 では 1 行あたり q 8 + fp32 累算 8 =
  16 レジスタ/レーン要るのでレジスタが溢れる。**帯域を減らす手が塞がっている。**
- 天井の側からも届かない: 0.75x = 12.59 ms は、MLX 自身の GEMM 2 本 (同じ
  行タイル、スコアを書いて読み直す形) の 13.23 ms より速い。スコアの往復を
  全部消しても、完全 causal の仕事 156.3 GFLOP を機械の bf16 GEMM 天井
  12.85 TFLOPS で回して 12.2 ms。融合カーネルは exp と online softmax 込みで
  **機械の GEMM 天井の 97%** を出す必要がある (自前は 2.7 TFLOPS)。
  QSA のタイル飛ばしを T=8 で効かせても (union 0.658) 84% が要る。
- kv=8192 の 1.39 倍は既存 `prefill_attn` (T=1 gather) の「kv=8192 で dense の
  1.43 倍遅い」と一致する。**流す形にしても集める形と同じ壁に当たる。**

反転条件: head_dim 256 でも 1 threadgroup に 8〜16 行を載せられるだけの
レジスタがある機械 (NAX 機で取り直す価値はある)、または `simdgroup_matrix`
で steel 並みの効率が出せることを別途示せたとき。
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import _fire

# prefill と decode/verify を分ける S の下限 (`prefill_attn.MIN_S` と同じ理由:
# online softmax はビット一致しないので verify 幅に入れると受理率が動く)。
MIN_S = 64

MAX_TG_BYTES = 30 * 1024

_KERNELS: dict[tuple, Any] = {}
_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    print(f"[mlxturbo] flash attention (d256) カーネル: {msg}")


def _source(
    head_dim: int,
    gqa: int,
    cr: int,
    bq: int,
    bk: int,
    scale_log2: float,
    tail_mode: str,
    vec: bool,
    itemsize: int,
) -> str:
    d = head_dim
    dpl = (d + 31) // 32
    nsg = gqa
    nth = nsg * 32
    ve = 16 // itemsize
    if vec and (d % ve != 0):
        vec = False
    # head_dim が 32 の倍数なら dd = lane + 32*e は必ず範囲内 (実行時の
    # 境界チェックを消すと内側ループが完全に展開できる)
    guard = "" if d % 32 == 0 else f"if (dd < {d}) "
    unroll = "#pragma clang loop unroll(full)"

    if vec:
        tg_kv_src = f"""    threadgroup uint4 tg_k4[{bk} * {d} / {ve}];
    threadgroup uint4 tg_v4[{bk} * {d} / {ve}];
    threadgroup T* tg_k = (threadgroup T*)tg_k4;
    threadgroup T* tg_v = (threadgroup T*)tg_v4;"""
        vpc = d // ve
        load_src = f"""        for (int cc = (int)sg; cc < ncol; cc += {nsg}) {{
            const int col = c0 + cc;
            const device uint4* ksrc =
                (const device uint4*)(kbase + (size_t)col * {d});
            const device uint4* vsrc =
                (const device uint4*)(vbase + (size_t)col * {d});
            for (int dv = (int)lane; dv < {vpc}; dv += 32) {{
                tg_k4[cc * {vpc} + dv] = ksrc[dv];
                tg_v4[cc * {vpc} + dv] = vsrc[dv];
            }}
        }}"""
    else:
        tg_kv_src = f"""    threadgroup T tg_k[{bk} * {d}];
    threadgroup T tg_v[{bk} * {d}];"""
        load_src = f"""        for (int e = (int)tid; e < ncol * {d}; e += {nth}) {{
            const int cc = e / {d};
            const int dd = e - cc * {d};
            const int col = c0 + cc;
            tg_k[e] = kbase[(size_t)col * {d} + dd];
            tg_v[e] = vbase[(size_t)col * {d} + dd];
        }}"""

    if tail_mode == "query":
        # クエリごとの tail: 列 [cr*floor((q+1)/cr), q]。
        tail_src = "const int tb = ((qc + 1) / %d) * %d; ok = (col >= tb);" % (cr, cr)
    else:
        # global tail: ブロック格子の外 (col >= NB*cr) を因果性の範囲で。
        tail_src = "ok = (col >= NB * %d);" % cr

    return f"""
    const int S      = params[0];   // このチャンクのクエリ行数
    const int NB     = params[1];   // 完全ブロック数 (kv_len / cr)
    const int KVLEN  = params[2];   // 可視な列の総数 (offset + S)
    const int CAP    = params[3];   // KV キャッシュの列方向の確保幅
    const int OFFSET = params[4];   // このチャンク先頭のキャッシュ列位置
    const int NKV    = params[5];   // kv head 数

    const uint tid  = thread_position_in_threadgroup.x;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int  rblk = (int)threadgroup_position_in_grid.y;
    const int  bh   = (int)threadgroup_position_in_grid.z;
    const int  b    = bh / NKV;
    const int  kvh  = bh - b * NKV;
    const int  H    = NKV * {nsg};
    const int  h    = kvh * {nsg} + (int)sg;
    const int  s0   = rblk * {bq};
    const int  nrow = min({bq}, S - s0);

    threadgroup uint tg_vis[{bq}];
{tg_kv_src}

    const device bool* keepbase = keep + (size_t)b * S * NB;

    // --- q をレジスタへ (scale と log2(e) を畳んでおく) --------------------
    float qv[{bq}][{dpl}];
    float acc[{bq}][{dpl}];
    float mm[{bq}];
    float ll[{bq}];
    {unroll}
    for (int t = 0; t < {bq}; ++t) {{
        const device T* qrow =
            q + (((size_t)b * S + s0 + min(t, nrow - 1)) * H + h) * {d};
        {unroll}
        for (int e = 0; e < {dpl}; ++e) {{
            const int dd = (int)lane + 32 * e;
            qv[t][e] = (t < nrow)
                ? (float)qrow[dd] * {scale_log2!r}f : 0.0f;
            acc[t][e] = 0.0f;
        }}
        mm[t] = -INFINITY;
        ll[t] = 0.0f;
    }}

    const device T* kbase = k + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};
    const device T* vbase = v + ((size_t)b * NKV + kvh) * (size_t)CAP * {d};

    // この行タイルのどの行からも見えない列は触らない
    const int cmax = min(KVLEN, OFFSET + s0 + nrow);

    for (int c0 = 0; c0 < cmax; c0 += {bk}) {{
        // 前タイルの tg_vis / tg_k / tg_v を読み終わるまで上書きしない
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- 1) 行ごとの可視ビット (BK 列ぶん) ---------------------------
        // simdgroup t (< BQ) が行 t を担当。レーン j が列 c0+j の 1 ビット。
        if ((int)sg < {bq}) {{
            const int t = (int)sg;
            uint bit = 0u;
            if (t < nrow && (int)lane < {bk}) {{
                const int col = c0 + (int)lane;
                const int qc = OFFSET + s0 + t;
                if (col < KVLEN && col <= qc) {{
                    const int blk = col / {cr};
                    bool ok = (blk < NB)
                        ? keepbase[(size_t)(s0 + t) * NB + blk] : false;
                    if (!ok) {{ {tail_src} }}
                    if (ok) {{ bit = 1u << lane; }}
                }}
            }}
            // ビットは互いに素なので和 = or
            const uint m = simd_sum(bit);
            if (lane == 0) {{ tg_vis[t] = m; }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint uni = 0u;
        for (int t = 0; t < {bq}; ++t) {{ uni |= tg_vis[t]; }}
        if (uni == 0u) {{ continue; }}   // 全行から見えない列タイル

        const int ncol = min({bk}, KVLEN - c0);

        // --- 2) K/V タイルを threadgroup へ ------------------------------
{load_src}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- 3) online softmax ------------------------------------------
        // **列 j が外、行 t が内**。K/V の threadgroup 読みは 1 列につき
        // {dpl} 本で、それを BQ 行で使い回す (行を外にすると同じ列を BQ 回
        // 読み直すことになり、読みが FMA と同数になって半分に落ちる)。
        // `vm` は threadgroup 一様 (tg_vis 経由) なので、その中で simd_sum を
        // 呼んでもレーンは分岐しない。
        // 行 t が外、列 j が内。`vm` は threadgroup 一様 (tg_vis 経由) なので
        // その中で simd_sum を呼んでもレーンは分岐しない。
        // (j を外にして K/V の threadgroup 読みを BQ 行で使い回す形も、
        //  スコア配列 sc[BQ][BK] の増分でレジスタが溢れて遅くなった。
        //  1 パス融合 (sc 無し、列ごとに再スケール) も同様に遅い。)
        {unroll}
        for (int t = 0; t < {bq}; ++t) {{
            const uint vm = tg_vis[t];
            float sc[{bk}];
            float mt = -INFINITY;
            {unroll}
            for (int j = 0; j < {bk}; ++j) {{
                float sj = -INFINITY;
                if ((vm & (1u << j)) != 0u) {{
                    float p = 0.0f;
                    {unroll}
                    for (int e = 0; e < {dpl}; ++e) {{
                        p += qv[t][e] * (float)tg_k[j * {d} + (int)lane + 32 * e];
                    }}
                    sj = simd_sum(p);
                }}
                sc[j] = sj;
                mt = metal::max(mt, sj);
            }}
            if (mt == -INFINITY) {{ continue; }}
            const float mnew = metal::max(mm[t], mt);
            const float corr = metal::exp2(mm[t] - mnew);   // 初回 exp2(-inf)=0
            {unroll}
            for (int e = 0; e < {dpl}; ++e) {{ acc[t][e] *= corr; }}
            ll[t] *= corr;
            {unroll}
            for (int j = 0; j < {bk}; ++j) {{
                if (sc[j] == -INFINITY) {{ continue; }}
                const float p = metal::exp2(sc[j] - mnew);
                ll[t] += p;
                {unroll}
                for (int e = 0; e < {dpl}; ++e) {{
                    acc[t][e] += p * (float)tg_v[j * {d} + (int)lane + 32 * e];
                }}
            }}
            mm[t] = mnew;
        }}
    }}

    {unroll}
    for (int t = 0; t < {bq}; ++t) {{
        if (t >= nrow) {{ break; }}
        const float inv = (ll[t] > 0.0f) ? (1.0f / ll[t]) : 0.0f;
        device T* orow = out + (((size_t)b * S + s0 + t) * H + h) * {d};
        {unroll}
        for (int e = 0; e < {dpl}; ++e) {{
            const int dd = (int)lane + 32 * e;
            {guard}orow[dd] = (T)(acc[t][e] * inv);
        }}
    }}
"""


def _get_kernel(head_dim, gqa, cr, bq, bk, scale_log2, tail_mode, vec, itemsize):
    key = (head_dim, gqa, cr, bq, bk, scale_log2, tail_mode, vec, itemsize)
    kern = _KERNELS.get(key)
    if kern is None:
        suffix = f"_u4{itemsize}" if vec else ""
        kern = mx.fast.metal_kernel(
            name=(
                f"flash_attn_d{head_dim}_{gqa}_{cr}_{bq}_{bk}"
                f"_{tail_mode}{suffix}"
            ),
            input_names=["q", "k", "v", "keep", "params"],
            output_names=["out"],
            source=_source(
                head_dim, gqa, cr, bq, bk, scale_log2, tail_mode, vec, itemsize
            ),
        )
        _KERNELS[key] = kern
    return kern


def tg_bytes(head_dim: int, bk: int, bq: int, itemsize: int) -> int:
    return 2 * bk * head_dim * itemsize + 4 * bq


def run(
    q_bshd: mx.array,
    keys: mx.array,
    values: mx.array,
    keep_block: mx.array,
    *,
    cap: int,
    cr: int,
    kv_len: int,
    n_blocks: int,
    offset: int,
    scale: float,
    bq: int,
    bk: int,
    tail_mode: str,
    n_kv: int,
    n_heads: int,
) -> mx.array:
    """低水準の入口 (micro からも呼ぶ)。``q_bshd`` は (B, S, n_heads, D)。"""
    import math

    B, S = q_bshd.shape[0], q_bshd.shape[1]
    head_dim = q_bshd.shape[3]
    if head_dim % 32 != 0:
        raise ValueError("head_dim は 32 の倍数であること (内側ループの前提)")
    if bk > 32 or bk % cr != 0:
        raise ValueError("BK は 32 以下かつ cr の倍数であること")
    gqa = n_heads // n_kv
    itemsize = q_bshd.dtype.size
    vec = (head_dim * itemsize) % 16 == 0 and (cap * head_dim * itemsize) % 16 == 0
    params = mx.array([S, n_blocks, kv_len, cap, offset, n_kv], dtype=mx.int32)
    kernel = _get_kernel(
        head_dim, gqa, cr, bq, bk,
        float(scale * math.log2(math.e)), tail_mode, vec, itemsize,
    )
    nblk = (S + bq - 1) // bq
    (out,) = kernel(
        inputs=[q_bshd, keys, values, keep_block, params],
        template=[("T", q_bshd.dtype)],
        grid=(32 * gqa, nblk, B * n_kv),
        threadgroup=(32 * gqa, 1, 1),
        output_shapes=[(B, S, n_heads, head_dim)],
        output_dtypes=[q_bshd.dtype],
    )
    return out


def _kv_buffers(cache, k: mx.array, v: mx.array, kv_len: int):
    """`prefill_attn._kv_buffers` と同じ (確保済みバッファ + 確保幅)。"""
    from . import prefill_attn as _pa

    return _pa._kv_buffers(cache, k, v, kv_len)


def eligible(q, k, v, keep_block, cache, cr, kv_len, n_blocks, bq, bk) -> bool:
    if q.shape[2] < MIN_S:
        _warn_once("min_s", f"S={q.shape[2]} は decode/verify 幅なので使わない")
        return False
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        _warn_once("gpu", "GPU が既定デバイスでないので使わない")
        return False
    if q.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        _warn_once("dtype", f"dtype {q.dtype} は非対応")
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        _warn_once("dtype_kv", "q/k/v の dtype が揃っていない")
        return False
    if keep_block.dtype != mx.bool_ or keep_block.ndim != 3:
        _warn_once("keep", "keep_block が (B,S,n_blocks) の bool でない")
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        _warn_once("ndim", "q/k/v の ndim が 4 でない")
        return False
    head_dim = q.shape[3]
    if k.shape[3] != head_dim or v.shape[3] != head_dim:
        _warn_once("headdim", "k/v の head_dim が q と違う")
        return False
    if head_dim % 32 != 0:
        # 内側ループは dd = lane + 32*e を境界チェック無しで引く
        _warn_once("headdim32", f"head_dim {head_dim} が 32 の倍数でない")
        return False
    n_kv = k.shape[1]
    if n_kv <= 0 or q.shape[1] % n_kv != 0:
        _warn_once("n_kv", f"n_kv={n_kv} が q の head 数を割り切れない")
        return False
    gqa = q.shape[1] // n_kv
    if gqa < bq or gqa > 32:
        # 可視ビットの計算を simdgroup 0..BQ-1 に割るので gqa >= bq が要る
        _warn_once("gqa", f"GQA {gqa} が BQ={bq} 未満、または 32 超")
        return False
    if bk > 32:
        _warn_once("bk", f"BK={bk} は 32 (uint ビットマスク) を超える")
        return False
    if cr < 1 or bk % cr != 0:
        _warn_once("cr", f"BK={bk} が cr={cr} の倍数でない")
        return False
    if n_blocks * cr > kv_len or kv_len - n_blocks * cr >= cr:
        _warn_once("blocks", "n_blocks と kv_len の関係が想定と違う")
        return False
    if tg_bytes(head_dim, bk, bq, q.dtype.size) > MAX_TG_BYTES:
        _warn_once("tg", "threadgroup メモリが上限を超える")
        return False
    if _kv_buffers(cache, k, v, kv_len) is None:
        _warn_once("cache", "KV キャッシュの確保済みバッファを取れない")
        return False
    return True


def flash_attn(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    keep_block: mx.array,
    cache,
    *,
    cr: int,
    kv_len: int,
    n_blocks: int,
    offset: int,
    scale: float,
    bq: int = 4,
    bk: int = 16,
) -> mx.array:
    """``q`` は (B, n_heads, S, D)。戻りは (B, S, n_heads, D)。"""
    _fire.bump("flash_attn_d256")
    B, n_heads, S, head_dim = q.shape
    n_kv = k.shape[1]
    keys, values, cap = _kv_buffers(cache, k, v, kv_len)
    from mlxturbo import qsa_tail as _qsa_tail

    return run(
        q.transpose(0, 2, 1, 3), keys, values, keep_block,
        cap=cap, cr=cr, kv_len=kv_len, n_blocks=n_blocks, offset=offset,
        scale=scale, bq=bq, bk=bk, tail_mode=_qsa_tail.MODE,
        n_kv=n_kv, n_heads=n_heads,
    )


__all__ = ["eligible", "flash_attn", "run", "MIN_S"]
