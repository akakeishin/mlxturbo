"""decode / verify 幅 (S<=8) の MoE を「専門家ごとに 1 回だけ読む」自前カーネル。

## 判定 (2026-09-03、PoL): **union 読みは負け。既定に入れない**

冷の連鎖 micro (`tools/moe_decode_micro.py`、E=512 の重み 2 組 2.83 GB、
48 層、乱数添字で毎回別の専門家 = 常に冷、ABBA x2)。MoE 1 層の us:

    S=1 (union 10/10)   素 130.7  rmax1 132.6  rmax2 144.9 (1.109)  rmax4 165.7 (1.267)
    S=3 (union 21.8/30) 素 256.3  rmax1 255.1  rmax2 273.2 (1.066)  rmax4 319.5 (1.247)
    S=6 (union 39.1/60) 素 420.5  rmax1 438.1  rmax2 490.6 (1.167)  rmax4 591.5 (1.407)

**S=1 では重複が 1 つも無い** (union = 対の数) のに rmax2 が +9%、rmax4 が
+27%。つまりこの遅さは「重複をまとめる仕組みそのもの」の代金 (アキュムレータ
rg/ru が 4 x rmax x 2 本に増えてレジスタが膨らみ、常駐 simdgroup が減る) で、
重複が実在する S=3 / S=6 でも、減ったバイト (27% / 35%) はその代金を返せない。

帯域で見ると: rmax=1 は対ごとの読みで 379 GB/s (素の 395 GB/s とほぼ同じ =
ピーク付近) だが、rmax=2 にすると読むバイトは 35% 減るのに 220 GB/s しか
出ない。**decode 幅の MoE は「重複を読む」ことでは損をしていない** --
重複の対は threadgroup を増やして memory-level parallelism を稼いでおり、
それを畳むと帯域そのものが落ちる (`IDEAS-2026-09-03.md` の D5 「畳む」を
別角度から裏付けた)。

そもそも判定線 (S=3 で素の 0.70 倍) は算術的に届かない: union 21.8/30 =
0.727 なので、バイトを完全に union に絞っても重み読みが 0.727 倍になるだけで、
素の 256 us のうち重み読み以外 (~50 us の固定費 + 対ごとの糊) は残る。
理想でも 0.78 倍が上限。

**残った芽 (未検証)**: rmax=1 (= 重複をまとめない) + ソート無し + gate/up 融合
の `fused:1` は S=1 で 0.929、S=3 で 0.979。`_gather_sort` / `_scatter_unsort` /
argsort と SwiGLU の素の op が消えるぶん。ただし温/冷どちらの micro でも
`moe_glu` (同じ融合をソート付きで) は in-model で負けており、この -2〜-7% は
in-model A/B を通すまで採らない。

以下は PoL のときの設計メモ。


## 何を変えるか

素の経路は `_gather_sort` で (行 x top-k) の対を専門家順に並べ、
`gather_qmv` が **対ごとに** 専門家の重みを読む。S=3 なら 30 対 = 同じ専門家を
引く行が居ても 30 回ぶん読む。実測の union は 21〜22.5/30 なので、
**重複ぶん (25〜30%) が丸ごう無駄な DRAM 読み**になる。

ここでは 1 threadgroup = (専門家の出現位置, 出力 8 行) にして、同じ専門家を
引く行 (最大 RMAX 行) を 1 回の重み読みで一緒に処理する。

## 並列度 (過去の負けの正体を踏まえた設計)

`moe_glu` / `moe_verify_gather` / HC 融合が decode で負けたのは
「threadgroup と lane が少なく DRAM レイテンシを隠せない」ため
(CATCHUP 2026-09-03 12:00)。ここは MLX の `gather_qmv_fast` と**同じ形**を保つ:

  - 1 threadgroup = simdgroup 2 本 (64 スレッド)、出力 4 行 x 2 = 8 行
  - grid = (64, H/8, P)。P = S*top_k。重複対の threadgroup は即 return する
    ので、実働は (union 数 x H/8) = S=3 で 22 x 80 = 1760 (MLX は 2400)
  - K 方向は lane ごとに 16 値のベクトル load (uint2 = 16 nibble)

つまり「重みを読む threadgroup の数」は union のぶんだけ減るが、
**1 threadgroup あたりの読み幅と lane 数は MLX と同一**。減るのは重複読みだけ。

## 重複のまとめ方 (ホスト同期なし)

添字はデバイス上にあるので、重複の数え上げは**カーネル内**でやる。
対 p の threadgroup は idx[0..p-1] を走査して同じ専門家の出現回数 c を数え、
`c % RMAX != 0` なら return (別の threadgroup が担当する)。
c % RMAX == 0 なら p から最大 RMAX 個の同専門家の対を集めて一緒に処理する。
P<=80 なのでこの走査は 80 回の uint load で済み、ソート (`carg_block_sort`、
短 decode で 2.3 ms/round) そのものが要らなくなる。

RMAX で刻むので多重度が RMAX を超えても正しい (4 個ずつの塊に割れる)。

## 数値

積和は fp32、出力 bf16。`w = s*q + b` の affine 展開は MLX と同じだが積和の
順序が違うのでビット一致はしない (`moe_glu.py` と同じ性質)。
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

_KERNELS: dict[tuple, Any] = {}

GROUP_SIZE = 64
BITS = 4
NSIMD = 2                    # gather_qmv_fast と同じ simdgroup 2 本
ROWS_PER_TG = NSIMD * 4      # 出力 4 行 x simdgroup 2 本
RMAX_CAP = 4                 # 1 threadgroup がまとめる同専門家の行数の上限
# 実効値は min(RMAX_CAP, S)。RMAX を上げるとレジスタ (アキュムレータ rg/ru が
# 4 x RMAX x 2 本) が増え、residency が落ちて DRAM レイテンシを隠せなくなる
# ので、S から決まる自明な上限 (多重度 <= S) までしか広げない。


def _dedup_prelude(P: int, rmax: int) -> str:
    """対 p が「担当 (leader)」かを決め、担当する対を pr0..pr{RMAX-1} に置く。

    idx は行 x top-k を平坦にしたもの (ソート不要)。同じ専門家の出現を
    RMAX 個ずつの塊に割り、塊の先頭だけが働く。"""
    slots = "\n        ".join(
        f"else if (nrows == {r}) {{ pr{r} = q; }}" for r in range(1, rmax)
    )
    decl = " ".join(f"uint pr{r} = 0;" for r in range(1, rmax))
    return f"""
    const uint p = threadgroup_position_in_grid.z;
    const uint e = idx[p];
    uint before = 0;
    for (uint q = 0; q < p; q++) {{ before += (idx[q] == e) ? 1 : 0; }}
    if (before % {rmax} != 0) return;          // 別の threadgroup が担当する
    uint pr0 = p; {decl}
    uint nrows = 1;
    for (uint q = p + 1; q < {P} && nrows < {rmax}; q++) {{
        if (idx[q] != e) continue;
        if (false) {{}}
        {slots}
        nrows++;
    }}
"""


def _row_ptrs(topk: int, K: int, rmax: int) -> str:
    """対 pr* -> トークン行 (pr/topk) -> x の先頭。"""
    lines = []
    for r in range(rmax):
        lines.append(
            f"    const device vec<T, 4>* xv{r} = "
            f"(const device vec<T, 4>*)(x_in + (size_t)(pr{r} / {topk}) * {K}) "
            f"+ lane * 4;"
        )
    return "\n".join(lines)


def _source_gate_up(K: int, H: int, topk: int, P: int, rmax: int) -> str:
    assert K % 512 == 0, "gate/up の K (hidden) は 512 の倍数が前提"
    n_iters = K // 512
    acc_decl = "\n    ".join(
        f"float rg{r}[4] = {{0,0,0,0}}; float ru{r}[4] = {{0,0,0,0}};"
        for r in range(rmax)
    )
    body_rows = []
    for r in range(rmax):
        body_rows.append(f"""
        if ({r} < nrows) {{
            float xt[16]; float xsum = 0.0f;
            #pragma unroll
            for (int v = 0; v < 4; v++) {{
                const vec<T, 4> xx = xv{r}[v];
                #pragma unroll
                for (int i = 0; i < 4; i++) {{
                    xt[v * 4 + i] = (float)xx[i]; xsum += xt[v * 4 + i];
                }}
            }}
            #pragma unroll
            for (int j = 0; j < 4; j++) {{
                float ag = 0.0f, au = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {{
                    ag += xt[i] * (float)((wg[j].x >> (4 * i)) & 0xF)
                        + xt[8 + i] * (float)((wg[j].y >> (4 * i)) & 0xF);
                    au += xt[i] * (float)((wu[j].x >> (4 * i)) & 0xF)
                        + xt[8 + i] * (float)((wu[j].y >> (4 * i)) & 0xF);
                }}
                rg{r}[j] += gs[j] * ag + gb[j] * xsum;
                ru{r}[j] += us[j] * au + ub[j] * xsum;
            }}
        }}""")
    adv = "\n        ".join(f"xv{r} += 128;" for r in range(rmax))
    # simd_sum は simdgroup 全体で呼ぶ必要があるので lane 分岐の外で取る
    reduce_block = "\n".join(
        f"""
        {{
            const float g = simd_sum(rg{r}[j]);
            const float u = simd_sum(ru{r}[j]);
            if ({r} < nrows && lane == 0 && row0 + j < {H}) {{
                out[(size_t)pr{r} * {H} + row0 + j] =
                    (T)(g * (1.0f / (1.0f + metal::exp(-g))) * u);
            }}
        }}""" for r in range(rmax)
    )
    return f"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint row0 = threadgroup_position_in_grid.y * ({NSIMD} * 4) + sg * 4;
    if (row0 >= {H}) return;
{_dedup_prelude(P, rmax)}
    const size_t wrow2 = (size_t)({K} / 16);      // uint2 / 行
    const size_t grow  = (size_t)({K} / {GROUP_SIZE});
    const size_t ebase = (size_t)e * {H};

    const device uint2* gw2 = (const device uint2*)gate_w + (ebase + row0) * wrow2 + lane;
    const device uint2* uw2 = (const device uint2*)up_w   + (ebase + row0) * wrow2 + lane;
    const device T* gsl = gate_s + (ebase + row0) * grow;
    const device T* gbl = gate_b + (ebase + row0) * grow;
    const device T* usl = up_s   + (ebase + row0) * grow;
    const device T* ubl = up_b   + (ebase + row0) * grow;
{_row_ptrs(topk, K, rmax)}
    const uint gofs = lane / 4;
    {acc_decl}

    for (int it = 0; it < {n_iters}; it++) {{
        uint2 wg[4]; uint2 wu[4];
        float gs[4], gb[4], us[4], ub[4];
        const uint gbase = it * 8 + gofs;
        #pragma unroll
        for (int j = 0; j < 4; j++) {{
            wg[j] = gw2[(size_t)j * wrow2];
            wu[j] = uw2[(size_t)j * wrow2];
            gs[j] = (float)gsl[j * grow + gbase];
            gb[j] = (float)gbl[j * grow + gbase];
            us[j] = (float)usl[j * grow + gbase];
            ub[j] = (float)ubl[j * grow + gbase];
        }}
{"".join(body_rows)}
        gw2 += 32; uw2 += 32;
        {adv}
    }}
    #pragma unroll
    for (int j = 0; j < 4; j++) {{
{reduce_block}
    }}
"""


def _source_down(K: int, H: int, topk: int, P: int, rmax: int) -> str:
    """down: K = moe_intermediate (640) は 512 の倍数ではないので、最後の
    イテレーションだけ有効レーンを絞る (`moe_verify_gather` と同じ作法)。"""
    assert K % GROUP_SIZE == 0
    full_iters = K // 512
    tail_vals = K - full_iters * 512
    tail_lanes = tail_vals // 16          # 1 lane = 16 値
    n_iters = full_iters + (1 if tail_vals else 0)
    acc_decl = "\n    ".join(
        f"float ry{r}[4] = {{0,0,0,0}};" for r in range(rmax)
    )
    body_rows = []
    for r in range(rmax):
        body_rows.append(f"""
        if ({r} < nrows && active) {{
            float xt[16]; float xsum = 0.0f;
            #pragma unroll
            for (int v = 0; v < 4; v++) {{
                const vec<T, 4> xx = xv{r}[v];
                #pragma unroll
                for (int i = 0; i < 4; i++) {{
                    xt[v * 4 + i] = (float)xx[i]; xsum += xt[v * 4 + i];
                }}
            }}
            #pragma unroll
            for (int j = 0; j < 4; j++) {{
                float ay = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {{
                    ay += xt[i] * (float)((wy[j].x >> (4 * i)) & 0xF)
                        + xt[8 + i] * (float)((wy[j].y >> (4 * i)) & 0xF);
                }}
                ry{r}[j] += ys[j] * ay + yb[j] * xsum;
            }}
        }}""")
    adv = "\n        ".join(f"xv{r} += 128;" for r in range(rmax))
    reduce_block = "\n".join(
        f"""
        {{
            const float y = simd_sum(ry{r}[j]);
            if ({r} < nrows && lane == 0 && row0 + j < {H}) {{
                out[(size_t)pr{r} * {H} + row0 + j] = (T)y;
            }}
        }}""" for r in range(rmax)
    )
    return f"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint row0 = threadgroup_position_in_grid.y * ({NSIMD} * 4) + sg * 4;
    if (row0 >= {H}) return;
{_dedup_prelude(P, rmax)}
    const size_t wrow2 = (size_t)({K} / 16);
    const size_t grow  = (size_t)({K} / {GROUP_SIZE});
    const size_t ebase = (size_t)e * {H};

    const device uint2* yw2 = (const device uint2*)down_w + (ebase + row0) * wrow2 + lane;
    const device T* ysl = down_s + (ebase + row0) * grow;
    const device T* ybl = down_b + (ebase + row0) * grow;
{_row_ptrs(topk, K, rmax)}
    const uint gofs = lane / 4;
    {acc_decl}

    for (int it = 0; it < {n_iters}; it++) {{
        const bool active = (it < {full_iters}) || (lane < {tail_lanes});
        uint2 wy[4]; float ys[4], yb[4];
        const uint gbase = it * 8 + gofs;
        if (active) {{
            #pragma unroll
            for (int j = 0; j < 4; j++) {{
                wy[j] = yw2[(size_t)j * wrow2];
                ys[j] = (float)ysl[j * grow + gbase];
                yb[j] = (float)ybl[j * grow + gbase];
            }}
        }}
{"".join(body_rows)}
        yw2 += 32;
        {adv}
    }}
    #pragma unroll
    for (int j = 0; j < 4; j++) {{
{reduce_block}
    }}
"""


def _get(name: str, src: str, inputs: list[str]):
    k = _KERNELS.get(name)
    if k is None:
        k = mx.fast.metal_kernel(
            name=name, input_names=inputs, output_names=["out"], source=src
        )
        _KERNELS[name] = k
    return k


def _wsb(m):
    """(w, s, b) タプル or SwitchLinear から重み 3 本を取り出して 2 次元に。"""
    if isinstance(m, (tuple, list)):
        w, s, b = m
    else:
        w, s, b = m.weight, m.scales, m.biases
    return (w.reshape(-1, w.shape[-1]), s.reshape(-1, s.shape[-1]),
            b.reshape(-1, b.shape[-1]))


def gate_up(x, idx_flat, gate, up, topk: int, rmax: int = 0):
    """x (S, K) bf16、idx_flat (P,) uint32 -> silu(gate)*up の (P, H) bf16。"""
    S, K = x.shape
    P = int(idx_flat.shape[0])
    rmax = _rmax(rmax, P // topk)
    gw, gs, gb = _wsb(gate)
    uw, us, ub = _wsb(up)
    H = _out_dim(gate)
    kern = _get(f"moe_dec_gu_{K}x{H}_t{topk}_p{P}_r{rmax}",
                _source_gate_up(K, H, topk, P, rmax),
                ["x_in", "idx", "gate_w", "gate_s", "gate_b",
                 "up_w", "up_s", "up_b"])
    (out,) = kern(
        inputs=[x, idx_flat.astype(mx.uint32), gw, gs, gb, uw, us, ub],
        template=[("T", mx.bfloat16)],
        output_shapes=[(P, H)],
        output_dtypes=[mx.bfloat16],
        grid=(32 * NSIMD, (H + ROWS_PER_TG - 1) // ROWS_PER_TG, P),
        threadgroup=(32 * NSIMD, 1, 1),
    )
    return out


def single(xin, idx_flat, m, topk: int, out_dim: int, rmax: int = 0):
    """1 本の量子化行列 (専門家ごと) を union 読みで掛ける汎用カーネル。

    `gate`/`up` を別々のカーネルで出す分割版のために `down` から切り出した。
    xin は (S, K) (topk>1: 対 -> 行は pr/topk) か (P, K) (topk=1)。
    """
    P = int(idx_flat.shape[0])
    K = xin.shape[-1]
    rmax = _rmax(rmax, P // topk if topk > 1 else P)
    w, sc, bi = _wsb(m)
    kern = _get(f"moe_dec_1_{K}x{out_dim}_t{topk}_p{P}_r{rmax}",
                _source_down(K, out_dim, topk, P, rmax),
                ["x_in", "idx", "down_w", "down_s", "down_b"])
    (out,) = kern(
        inputs=[xin, idx_flat.astype(mx.uint32), w, sc, bi],
        template=[("T", mx.bfloat16)],
        output_shapes=[(P, out_dim)],
        output_dtypes=[mx.bfloat16],
        grid=(32 * NSIMD, (out_dim + ROWS_PER_TG - 1) // ROWS_PER_TG, P),
        threadgroup=(32 * NSIMD, 1, 1),
    )
    return out


def down(h, idx_flat, down_m, topk: int, hidden: int, rmax: int = 0):
    """h (P, Kint) bf16 -> (P, hidden) bf16。h は対ごとなので topk では割らない
    (行の割り出しは `_row_ptrs` が pr/topk でやるが、down の x は対ごとに
    1 行なので topk=1 として渡す)。"""
    P, Kint = h.shape
    rmax = _rmax(rmax, P // topk)
    dw, ds, db = _wsb(down_m)
    H = hidden
    kern = _get(f"moe_dec_dn_{Kint}x{H}_t{topk}_p{P}_r{rmax}",
                _source_down(Kint, H, 1, P, rmax),
                ["x_in", "idx", "down_w", "down_s", "down_b"])
    (out,) = kern(
        inputs=[h, idx_flat.astype(mx.uint32), dw, ds, db],
        template=[("T", mx.bfloat16)],
        output_shapes=[(P, H)],
        output_dtypes=[mx.bfloat16],
        grid=(32 * NSIMD, (H + ROWS_PER_TG - 1) // ROWS_PER_TG, P),
        threadgroup=(32 * NSIMD, 1, 1),
    )
    return out


def _rmax(req: int, S: int) -> int:
    """実効 RMAX。同専門家の多重度は行数 S を超えないので min(cap, S)。
    `MLXTURBO_MOE_DEC_FUSED_RMAX` で実験用に上書きできる。"""
    import os

    if req:
        return max(1, req)
    env = os.environ.get("MLXTURBO_MOE_DEC_FUSED_RMAX")
    cap = int(env) if env else RMAX_CAP
    return max(1, min(cap, S))


def _out_dim(m) -> int:
    if isinstance(m, (tuple, list)):
        return int(m[1].shape[-2])
    return int(m.scales.shape[-2])


def eligible(x, *mods) -> bool:
    if x.dtype != mx.bfloat16:
        return False
    for m in mods:
        if isinstance(m, (tuple, list)):
            continue
        if not hasattr(m, "scales"):
            return False
        if m.bits != BITS or m.group_size != GROUP_SIZE:
            return False
        if getattr(m, "mode", "affine") != "affine":
            return False
    return mx.default_device() == mx.gpu and mx.metal.is_available()
