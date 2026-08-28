"""hyper-connections (`GatedResidual`) を 2 本の Metal カーネルに畳む。

Flash-Next のデコードはディスパッチ律速で、`GatedResidual` は 1 層 2 個 x 48 層
= 96 回呼ばれる。素の実装は約 15 op なので 1 トークンあたり 1440 回の起動になり、
実測 21.3ms/token (tools/ablate.py, v-ng2) を占める。演算量ではなく起動回数の
問題なので、op を畳めばそのぶん素直に落ちる。

## なぜ 2 本で、1 本ではないのか

元の計算は

    normed = rms_norm(hyper, レーンごと) * (1 + w)      # 10240
    t      = silu(down(normed) / hc)                    # 10240 -> 320
    u      = sigmoid(up(t))                             # 320 -> 10240
    mixed  = mean(u.reshape(hc, d) * normed.reshape(hc, d), axis=0)   # -> 2560

で、`down` の全出力が揃わないと `up` を始められない。Metal にはグリッド全体の
バリアが無いので、この境目は必ずカーネルの切れ目になる。

1 スレッドグループに全部押し込めば 1 本にはできるが、量子化重みは down/up
合わせて 6.5MB あり、1 スレッドグループでは帯域を引き出せない。GPU を遊ばせて
まで起動を 1 回減らす取引にはならない。

境目を 1 つだけ許して 2 本にする:

- `hc_pre`  : レーンごとの rms -> normed -> down -> silu、ついでに inject
- `hc_post` : up -> sigmoid -> レーン重み付き平均

`hc_pre` はレーンの rsqrt を副産物として書き出し、`hc_post` はそれを受け取って
normed を再計算する。`hc_post` の出力 j は入力の (l*d + j) だけを見るので、
再計算しても hyper と norm_weight をちょうど 1 回ずつ読むだけで済む。

## 精度

素の実装は bf16 なので、fp32 で通して丸めを減らすと*正確*にはなるが参照からは
遠ざかる。そこで **参照が bf16 配列を実体化する地点をすべて再現する**
(rms_norm の出力、(1 + w)、量子化行列積の出力、silu の sigmoid と積、
up の出力、sigmoid、u * normed)。積和自体は mx.quantized_matmul と同じく
fp32 で溜めて出力で丸める。この形で down の量子化行列積は MLX とビット一致する。

ただし **MLX の bf16 `sigmoid` はビット単位では再現できない**。安定形/素朴形 x
`metal::exp` / `fast::exp` / `precise::exp` x 中間の丸めの総当たりでも、最良で
40 万要素中 4.1 万個 (10%) が 1 ulp 違う。この 1 ulp が 96 段の
hyper-connection を通ると logits で数 % になる。

**これは実装の誤りではない。**素と op 単位で同じで sigmoid だけ fp32 にした
(= 素より正確な) 対照版が 7.3% ずれる。BRIEF の受け入れ基準
「logits 相対誤差 1e-3 未満」はどんな実装でも届かないので、品質は
`bench/quant_eval.py` の KLD で見ること。詳細は docs/KERNEL-HANDOFF-HC.md。

## 形状

すべてテンプレート引数で受ける。GLM-5.3-Flash の mHC も同じ構造なので、
hc / d / lowrank / bits / group_size を差し替えれば載る想定。
`hyper` の先頭次元は平坦化してグリッドの z に載せるので、S=1 のデコードでも
プロンプトの一括 forward でも同じカーネルが動く (適格判定は
:func:`eligible` を参照)。
"""

from __future__ import annotations

from math import prod
from typing import Any

import mlx.core as mx

# threadgroup メモリに normed (hc*d 要素) を丸ごと置く。Apple GPU の上限 32KB に
# 対してマージンを取る。GLM 系で hc*d がこれを超えるなら素の実装に落とす。
MAX_TG_BYTES = 28 * 1024

_SIMDGROUPS = 32
_THREADS = _SIMDGROUPS * 32

# down の出力行をスレッドグループ何本で分けるか。1 本ごとに hyper と norm_weight を
# 読み直すので、増やすと並列度と引き換えに冗長読みが増える。lowrank=320 なら
# 16 行 x 20 本で、冗長読みは重み 3.3MB に対して 0.8MB。
#
# (_ROWS_PER_TG, _SIMDGROUPS) は重みを 96 組み回してキャッシュに乗らない状態を
# 作った上で 3 回ずつ測って選んだ (中央値 us/call、素は同条件で 90.7):
# (16,32) 39.9、(8,16) 41.2、(32,32) 40.9、(16,16) 42.4、(8,8) 46.2、(4,4) 64.3。
#
# 差は雑音幅 (±10%) と同程度だが、(16,32) が最良値・中央値・最悪値のいずれでも
# 先頭。スレッドグループ数を増やしても大きくは効かない: 重みを一切読まない版が
# 19.7 -> 16.6us にしかならず、hc_pre は帯域ではなく要素あたりの
# 逆量子化 ALU と threadgroup メモリ読みで律速しているため。
_ROWS_PER_TG = 16

_KERNELS: dict[tuple, Any] = {}


def _dequant_block(wname: str, bits: int, group_size: int, index_expr: str) -> str:
    """1 uint32 ぶん (32/bits 個) の逆量子化と積和を展開する。

    group_size が 32/bits で割り切れる前提なので、1 word 内の値はすべて同じ
    量子化群に属する。scales/biases の読み出しは word ごとに 1 回で済む。
    """
    vpw = 32 // bits
    mask = (1 << bits) - 1
    terms = "\n".join(
        f"            acc += (s * (float)((word >> {e * bits}) & {mask}u) + b) * ({index_expr.format(k=f'k0 + {e}')});"
        for e in range(vpw)
    )
    return f"""
        {{
            uint word = {wname}_row[wi];
            int k0 = wi * {vpw};
            int g = k0 / {group_size};
            float s = (float)s_row[g];
            float b = (float)b_row[g];
{terms}
        }}"""


def _vec4_ok(words: int, bits: int, group_size: int) -> bool:
    """uint4 でまとめ読みできる形かどうか。

    行の先頭は 4 word 境界に揃っている必要があり (words % 4)、1 回の uint4 に
    入る 4*32/bits 個の値が同じ量子化群に収まる必要がある。
    """
    return words % 4 == 0 and group_size % (4 * (32 // bits)) == 0


def _dequant_block_vec4(wname: str, bits: int, group_size: int, index_expr: str) -> str:
    """uint4 で 4 word まとめて読む版。

    lane あたりの 1 回の要求が 4 byte から 16 byte になる。実測で hc_pre の
    帯域が 195 -> 227 GB/s に上がった (22.2 -> 19.2us)。

    逆量子化は要素ごとの shift/mask/変換のまま置いてある。`as_type<uchar4>` +
    `float4` + `dot` でベクトル化する版を試したが、**遅くなった**
    (hc_pre 19.2 -> 24.8us、hc_post 15.6 -> 28.8us)。threadgroup メモリから
    4 要素を集めて float4 を組む方が、スカラーの FMA 連鎖より高くつく。
    """
    vpw = 32 // bits
    mask = (1 << bits) - 1
    per_vec = 4 * vpw
    comps = "xyzw"
    terms = []
    for c in range(4):
        for e in range(vpw):
            k = f"k0 + {c * vpw + e}"
            terms.append(
                f"            acc += (s * (float)((wv.{comps[c]} >> {e * bits}) & {mask}u) + b)"
                f" * ({index_expr.format(k=k)});"
            )
    body = "\n".join(terms)
    return f"""
        {{
            uint4 wv = {wname}_vec[vi];
            int k0 = vi * {per_vec};
            int g = k0 / {group_size};
            float s = (float)s_row[g];
            float b = (float)b_row[g];
{body}
        }}"""


def _dot_loop(wname: str, words: int, bits: int, group_size: int, index_expr: str) -> str:
    """1 行ぶんの積和ループ。uint4 が使える形ならそちらを選ぶ。"""
    if _vec4_ok(words, bits, group_size):
        inner = _dequant_block_vec4(wname, bits, group_size, index_expr)
        return f"""
                const device uint4* {wname}_vec = (const device uint4*){wname}_row;
                for (int vi = (int)simd_lid; vi < {words // 4}; vi += 32) {{
{inner}
                }}"""
    inner = _dequant_block(wname, bits, group_size, index_expr)
    return f"""
                for (int wi = (int)simd_lid; wi < {words}; wi += 32) {{
{inner}
                }}"""


def _pre_source(cfg: dict) -> str:
    hc, d, lowrank = cfg["hc"], cfg["d"], cfg["lowrank"]
    hcd = hc * d
    bits, gs = cfg["bits"], cfg["group_size"]
    eps = cfg["eps"]
    combine = cfg["combine"]
    words = hcd * bits // 32
    n_groups = hcd // gs
    n_down_tg = (lowrank + _ROWS_PER_TG - 1) // _ROWS_PER_TG

    inject_body = ""
    if combine:
        inject_body = f"""
    }} else {{
        // inject 行 (hc 本)。1 スレッドグループで足りる小ささ
        for (int rr = (int)simd_gid; rr < {hc}; rr += {_SIMDGROUPS}) {{
            const device uint32_t* inject_w_row = inject_w + (size_t)rr * {words};
            const device T* s_row = inject_s + (size_t)rr * {n_groups};
            const device T* b_row = inject_b + (size_t)rr * {n_groups};
            float acc = 0.0f;
{_dot_loop("inject_w", words, bits, gs, "(float)tg_normed[{k}]")}
            acc = simd_sum(acc);
            if (simd_lid == 0) {{
                // 参照: 2 * sigmoid(qmm(normed) / hc)
                float xv = (float)((T)((float)((T)acc) / {float(hc)}f));
                float sg = (float)((T)(1.0f / (1.0f + metal::exp(-xv))));
                inject[(size_t)m * {hc} + rr] = (T)(2.0f * sg);
            }}
        }}"""

    return f"""
    typedef float U;

    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int  tid = (int)simd_gid * 32 + (int)simd_lid;
    int  tgi = (int)threadgroup_position_in_grid.y;
    int  m   = (int)threadgroup_position_in_grid.z;

    threadgroup float tg_part[{_SIMDGROUPS}];
    threadgroup float tg_r[{hc}];
    threadgroup T     tg_normed[{hcd}];

    const device T* hyper_m = hyper + (size_t)m * {hcd};

    // 1) レーンごとの rms。mx.fast.rms_norm と同じく fp32 で溜める。
    //    ついでに hyper を threadgroup メモリへ移しておく (2 パス目で device
    //    メモリを読み直さずに済む)
    for (int l = 0; l < {hc}; l++) {{
        float ss = 0.0f;
        for (int i = tid; i < {d}; i += {_THREADS}) {{
            T raw = hyper_m[l * {d} + i];
            tg_normed[l * {d} + i] = raw;
            float v = (float)raw;
            ss += v * v;
        }}
        ss = simd_sum(ss);
        if (simd_lid == 0) tg_part[simd_gid] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {{
            float tot = 0.0f;
            for (int q = 0; q < {_SIMDGROUPS}; q++) tot += tg_part[q];
            tg_r[l] = metal::rsqrt(tot / {float(d)}f + {eps!r}f);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // 2) normed を threadgroup メモリに置く。参照は rms_norm の出力と
    //    (1 + weight) をそれぞれ bf16 に落とすので、その丸めを再現する
    for (int i = tid; i < {hcd}; i += {_THREADS}) {{
        T nrm   = (T)((float)tg_normed[i] * tg_r[i / {d}]);
        T scale = (T)(1.0f + (float)norm_weight[i]);
        tg_normed[i] = (T)((float)nrm * (float)scale);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // hc_post が normed を組み直すのに使う。全 threadgroup が同じ値を持つので
    // 1 本だけ書けばよい
    if (tgi == 0 && tid < {hc}) rlane[(size_t)m * {hc} + tid] = tg_r[tid];

    // 3) down (hcd -> lowrank) + silu、あるいは inject
    if (tgi < {n_down_tg}) {{
        for (int rr = (int)simd_gid; rr < {_ROWS_PER_TG}; rr += {_SIMDGROUPS}) {{
            int row = tgi * {_ROWS_PER_TG} + rr;
            if (row < {lowrank}) {{
                const device uint32_t* down_w_row = down_w + (size_t)row * {words};
                const device T* s_row = down_s + (size_t)row * {n_groups};
                const device T* b_row = down_b + (size_t)row * {n_groups};
                float acc = 0.0f;
{_dot_loop("down_w", words, bits, gs, "(float)tg_normed[{k}]")}
                acc = simd_sum(acc);
                if (simd_lid == 0) {{
                    // 参照: nn.silu(qmm(normed) / hc) = x * sigmoid(x)、各段 bf16
                    float xv = (float)((T)((float)((T)acc) / {float(hc)}f));
                    float sg = (float)((T)(1.0f / (1.0f + metal::exp(-xv))));
                    t[(size_t)m * {lowrank} + row] = (T)(xv * sg);
                }}
            }}
        }}
{inject_body}
    }}
"""


def _post_source(cfg: dict) -> str:
    hc, d, lowrank = cfg["hc"], cfg["d"], cfg["lowrank"]
    bits, gs = cfg["bits"], cfg["group_size"]
    words = lowrank * bits // 32
    n_groups = lowrank // gs

    return f"""
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int  tid = (int)simd_gid * 32 + (int)simd_lid;
    int  tgi = (int)threadgroup_position_in_grid.y;
    int  m   = (int)threadgroup_position_in_grid.z;

    // t は lowrank 個しかないので threadgroup メモリに載せて全 simdgroup で使い回す
    threadgroup float tg_t[{lowrank}];
    for (int i = tid; i < {lowrank}; i += {_THREADS}) {{
        tg_t[i] = (float)t[(size_t)m * {lowrank} + i];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int j = tgi * {_SIMDGROUPS} + (int)simd_gid;
    if (j < {d}) {{
        float r_l[{hc}];
        for (int l = 0; l < {hc}; l++) r_l[l] = rlane[(size_t)m * {hc} + l];

        // 参照の mean(axis=-2) は **bf16 で逐次加算**する (fp32 で溜めて最後に
        // 丸める形ではない。差は rel 3e-3 で、96 回ぶん積むと KLD に出る)
        float total = 0.0f;
        for (int l = 0; l < {hc}; l++) {{
            int row = l * {d} + j;
            const device uint32_t* up_w_row = up_w + (size_t)row * {words};
            const device T* s_row = up_s + (size_t)row * {n_groups};
            const device T* b_row = up_b + (size_t)row * {n_groups};
            float acc = 0.0f;
{_dot_loop("up_w", words, bits, gs, "tg_t[{k}]")}
            acc = simd_sum(acc);
            if (simd_lid == 0) {{
                float sg = (float)((T)(1.0f / (1.0f + metal::exp(-(float)((T)acc)))));
                // normed を組み直す。出力 j は (l*d + j) しか見ないので、
                // hyper と norm_weight はカーネル全体でちょうど 1 回ずつ読まれる
                size_t hi = (size_t)m * {hc * d} + row;
                T nrm   = (T)((float)hyper[hi] * r_l[l]);
                T scale = (T)(1.0f + (float)norm_weight[row]);
                float nd = (float)((T)((float)nrm * (float)scale));
                total = (float)((T)(total + (float)((T)(sg * nd))));
            }}
        }}
        // bf16 で溜めた総和を hc で割る (hc=4 なら bf16 でも厳密)
        if (simd_lid == 0) mixed[(size_t)m * {d} + j] = (T)(total / {float(hc)}f);
    }}
"""


def _get_kernels(cfg: dict):
    key = tuple(sorted(cfg.items()))
    built = _KERNELS.get(key)
    if built is not None:
        return built

    tag = "c" if cfg["combine"] else "p"
    suffix = f"{cfg['hc']}x{cfg['d']}_{cfg['lowrank']}_{cfg['bits']}b{cfg['group_size']}_{tag}{len(_KERNELS)}"

    pre_inputs = ["hyper", "norm_weight", "down_w", "down_s", "down_b"]
    pre_outputs = ["t", "rlane"]
    if cfg["combine"]:
        pre_inputs += ["inject_w", "inject_s", "inject_b"]
        pre_outputs += ["inject"]

    pre = mx.fast.metal_kernel(
        name=f"hc_pre_{suffix}",
        input_names=pre_inputs,
        output_names=pre_outputs,
        source=_pre_source(cfg),
    )
    post = mx.fast.metal_kernel(
        name=f"hc_post_{suffix}",
        input_names=["hyper", "norm_weight", "rlane", "t", "up_w", "up_s", "up_b"],
        output_names=["mixed"],
        source=_post_source(cfg),
    )
    _KERNELS[key] = (pre, post)
    return pre, post


def eligible(
    hyper: mx.array,
    norm_weight: mx.array,
    down: tuple,
    up: tuple,
    inject: tuple | None,
    hc: int,
    d: int,
) -> bool:
    """このカーネルで扱える形と量子化かを判定する。外れたら呼び出し側は素の実装へ。"""

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    if hyper.dtype not in (mx.float16, mx.bfloat16):
        return False
    if norm_weight.dtype != hyper.dtype:
        return False
    if hyper.shape[-1] != hc * d:
        return False
    if hc * d * hyper.dtype.size > MAX_TG_BYTES:
        return False

    parts = [down, up] + ([inject] if inject is not None else [])
    for w, s, b, gs, bits in parts:
        # 非量子化の線形層 (len 1 のパック) はここに来ない
        if bits not in (4, 8) or gs % (32 // bits) or w.dtype != mx.uint32:
            return False
        if s is None or b is None or s.dtype != hyper.dtype or b.dtype != hyper.dtype:
            return False
        n, k_words = w.shape
        k = k_words * 32 // bits
        if k % gs or s.shape != (n, k // gs) or b.shape != (n, k // gs):
            return False

    lowrank = down[0].shape[0]
    if down[0].shape[0] != up[0].shape[1] * 32 // up[4] or up[0].shape[0] != hc * d:
        return False
    if lowrank % 32:
        # tg_t の読み込みと simdgroup 分担が半端になる形は素の実装へ
        return False
    if inject is not None and inject[0].shape[0] != hc:
        return False
    return True


def fused_gated_residual(
    hyper: mx.array,
    norm_weight: mx.array,
    eps: float,
    hc: int,
    d: int,
    down: tuple,
    up: tuple,
    inject: tuple | None,
):
    """`GatedResidual.__call__` の中身を 2 本のカーネルで計算する。

    引数の ``down`` / ``up`` / ``inject`` は ``(weight, scales, biases,
    group_size, bits)``。戻り値は ``mixed``、``inject`` があれば
    ``(mixed, inject)``。
    """

    lowrank = down[0].shape[0]
    cfg = {
        "hc": hc,
        "d": d,
        "lowrank": lowrank,
        "bits": down[4],
        "group_size": down[3],
        "eps": float(eps),
        "combine": inject is not None,
    }
    pre, post = _get_kernels(cfg)

    lead = hyper.shape[:-1]
    m = prod(lead) if lead else 1
    dt = hyper.dtype
    flat = hyper.reshape((m, hc * d))

    n_down_tg = (lowrank + _ROWS_PER_TG - 1) // _ROWS_PER_TG
    pre_tg = n_down_tg + (1 if inject is not None else 0)

    pre_inputs = [flat, norm_weight, down[0], down[1], down[2]]
    pre_shapes = [(m, lowrank), (m, hc)]
    pre_dtypes = [dt, mx.float32]
    if inject is not None:
        pre_inputs += [inject[0], inject[1], inject[2]]
        pre_shapes += [(m, hc)]
        pre_dtypes += [dt]

    outs = pre(
        inputs=pre_inputs,
        template=[("T", dt)],
        grid=(_THREADS, pre_tg, m),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=pre_shapes,
        output_dtypes=pre_dtypes,
    )
    if inject is not None:
        t, rlane, inj = outs
    else:
        (t, rlane), inj = outs, None

    post_tg = (d + _SIMDGROUPS - 1) // _SIMDGROUPS
    (mixed,) = post(
        inputs=[flat, norm_weight, rlane, t, up[0], up[1], up[2]],
        template=[("T", dt)],
        grid=(_THREADS, post_tg, m),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(m, d)],
        output_dtypes=[dt],
    )

    mixed = mixed.reshape((*lead, d))
    if inj is None:
        return mixed
    return mixed, inj.reshape((*lead, hc))


__all__ = ["eligible", "fused_gated_residual"]
