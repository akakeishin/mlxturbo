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

from . import _fire

# threadgroup メモリに normed (hc*d 要素) を丸ごと置く。Apple GPU の上限 32KB に
# 対してマージンを取る。GLM 系で hc*d がこれを超えるなら素の実装に落とす。
MAX_TG_BYTES = 28 * 1024

_SIMDGROUPS = 32
_THREADS = _SIMDGROUPS * 32

# down の出力行をスレッドグループ何本で分けるか。1 本ごとに hyper と norm_weight を
# 読み直すので、増やすと並列度と引き換えに冗長読みが増える。lowrank=320 なら
# 16 行 x 20 本で、冗長読みは重み 3.3MB に対して 0.8MB。
#
# down 側の 1 threadgroup は _SIMDGROUPS(32) 本の simdgroup のうち
# _ROWS_PER_TG(16) 本しか使わない (rr の for ループが simd_gid>=16 で回らない
# ので、残り 16 本は down の仕事では常に遊んでいる)。inject (hc=4 本) は
# その遊んでいる simdgroup を tgi==0 の中で使い回す (_fold_inject_ok 参照)。
# 専用の 21 番目の threadgroup を割り当てて normed をもう一度計算させるより、
# 既に tg_normed を持っている threadgroup に相乗りさせる方が起動 1 回ぶん安い。
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


def _fold_inject_ok(hc: int) -> bool:
    """inject (hc 行) を down の空き simdgroup (_ROWS_PER_TG.._SIMDGROUPS-1)
    に相乗りさせられるか。hc がその余白に収まらない構成なら専用
    threadgroup (旧経路) に戻す。現行の hc=4 では 16 本の余白に対して
    4 本なので常に True。
    """
    return hc <= (_SIMDGROUPS - _ROWS_PER_TG)


_KERNELS: dict[tuple, Any] = {}

# prefill (1 行 = 1 threadgroup) カーネルのキャッシュと M しきい値。しきい値
# 未満は decode とみなし hc_pre/hc_post のままにする (小さい M では
# threadgroup 数が稼げず、1 行 1 threadgroup 化は並列度を落とすだけで得しない)。
# 有効化そのものは fused.py 側の MLXTURBO_HC_PREFILL=1 ゲートが握るので、
# この定数だけでは既定の decode 挙動に影響しない。
PREFILL_M_THRESHOLD = 128

_KERNELS_PREFILL: dict[tuple, Any] = {}


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


def _dot_loop_half(wname: str, words: int, bits: int, group_size: int, index_expr: str) -> str:
    """1 行ぶんの積和ループを 2 simdgroup (`half_idx` が 0/1) で分担する版。

    呼び出し側が `_vec4_ok(words, bits, group_size)` と `words // 4 >= 2` を
    保証すること (uint4 の範囲を単純に前半/後半で割るだけなので、奇数なら
    後半が 1 個多く持つ)。1 行の総和を 2 つの simdgroup が別々の部分和として
    計算し、呼び出し元が `simd_sum` の結果 2 つを device/threadgroup メモリで
    合算する想定 (このループ自体は合算しない)。

    生成コードは呼び出し元スコープの `int half_idx` (0 か 1) を参照する。
    Metal では `half` は 16bit float の予約型名なので、変数名としては使えない
    (`int half = ...` はコンパイルエラーになる -- 実測済み)。
    """
    inner = _dequant_block_vec4(wname, bits, group_size, index_expr)
    return f"""
                const device uint4* {wname}_vec = (const device uint4*){wname}_row;
                int {wname}_total_vec = {words // 4};
                int {wname}_half_vec = {wname}_total_vec / 2;
                int {wname}_vstart = half_idx * {wname}_half_vec;
                int {wname}_vend = (half_idx == 0) ? {wname}_half_vec : {wname}_total_vec;
                for (int vi = {wname}_vstart + (int)simd_lid; vi < {wname}_vend; vi += 32) {{
{inner}
                }}"""


def _pre_source(cfg: dict) -> str:
    hc, d, lowrank = cfg["hc"], cfg["d"], cfg["lowrank"]
    hcd = hc * d
    bits, gs = cfg["bits"], cfg["group_size"]
    eps = cfg["eps"]
    inject_kind = cfg["inject_kind"]
    words = hcd * bits // 32
    n_groups = hcd // gs
    n_down_tg = (lowrank + _ROWS_PER_TG - 1) // _ROWS_PER_TG
    # 候補 (b, 2026-09-03 in-model 検証): down の 1 行ぶんの内積を 2 simdgroup
    # (half=0/1) で分担し、down では今まで遊んでいた上位 16 simdgroup も動員
    # する。全 32 simdgroup が down で埋まるので、この間は inject を fold
    # する空きが無く、専用 threadgroup (旧経路) に戻る。
    split_down = _vec4_ok(words, bits, gs) and (words // 4) >= 2
    fold_inject = (not split_down) and inject_kind is not None and _fold_inject_ok(hc)

    inject_body = ""
    if inject_kind == "quant":
        if fold_inject:
            inject_body = f"""
    // inject 行 (hc 本、量子化)。down で使わない上位 simdgroup
    // ({_ROWS_PER_TG}..{_SIMDGROUPS - 1}) を tgi==0 の中で相乗りさせる
    // (この threadgroup は既に tg_normed を持っているので、専用
    // threadgroup を割り当てて正規化をもう一度計算させるより安い)
    if (tgi == 0 && (int)simd_gid >= {_ROWS_PER_TG}) {{
        for (int rr = (int)simd_gid - {_ROWS_PER_TG}; rr < {hc}; rr += {_SIMDGROUPS - _ROWS_PER_TG}) {{
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
        }}
    }}"""
        else:
            inject_body = f"""
    }} else {{
        // inject 行 (hc 本、量子化)。hc が down の空き simdgroup に収まらない
        // ので専用 threadgroup に戻す (_fold_inject_ok 参照。現行の hc=4 では
        // 通らない経路)
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
    elif inject_kind == "bf16":
        if fold_inject:
            inject_body = f"""
    // inject 行 (hc 本、非量子化)。折り込み版 (量子化 inject と同じ理由)。
    // 重みは (hc, hcd) の T をそのまま読むだけで逆量子化は不要 (80KB 前後
    // なので帯域上も無視できる)
    if (tgi == 0 && (int)simd_gid >= {_ROWS_PER_TG}) {{
        for (int rr = (int)simd_gid - {_ROWS_PER_TG}; rr < {hc}; rr += {_SIMDGROUPS - _ROWS_PER_TG}) {{
            const device T* inject_w_row = inject_w + (size_t)rr * {hcd};
            float acc = 0.0f;
            for (int k = (int)simd_lid; k < {hcd}; k += 32) {{
                acc += (float)inject_w_row[k] * (float)tg_normed[k];
            }}
            acc = simd_sum(acc);
            if (simd_lid == 0) {{
                // 参照: 2 * sigmoid(qmm(normed) / hc)。qmm は非量子化なら
                // 単なる行列積 (mlxturbo/fused.py の core() 参照)
                float xv = (float)((T)((float)((T)acc) / {float(hc)}f));
                float sg = (float)((T)(1.0f / (1.0f + metal::exp(-xv))));
                inject[(size_t)m * {hc} + rr] = (T)(2.0f * sg);
            }}
        }}
    }}"""
        else:
            inject_body = f"""
    }} else {{
        // inject 行 (hc 本、非量子化)。block_inject_weight が
        // QuantizedLinear に変換されず bf16/fp16 の nn.Linear のまま残った
        // 層向け (診断: 97 層中 96 層がこれに該当、hc_fire_diag.py 参照)。
        // hc が down の空き simdgroup に収まらないので専用 threadgroup に
        // 戻す (_fold_inject_ok 参照。現行の hc=4 では通らない経路)
        for (int rr = (int)simd_gid; rr < {hc}; rr += {_SIMDGROUPS}) {{
            const device T* inject_w_row = inject_w + (size_t)rr * {hcd};
            float acc = 0.0f;
            for (int k = (int)simd_lid; k < {hcd}; k += 32) {{
                acc += (float)inject_w_row[k] * (float)tg_normed[k];
            }}
            acc = simd_sum(acc);
            if (simd_lid == 0) {{
                // 参照: 2 * sigmoid(qmm(normed) / hc)。qmm は非量子化なら
                // 単なる行列積 (mlxturbo/fused.py の core() 参照)
                float xv = (float)((T)((float)((T)acc) / {float(hc)}f));
                float sg = (float)((T)(1.0f / (1.0f + metal::exp(-xv))));
                inject[(size_t)m * {hc} + rr] = (T)(2.0f * sg);
            }}
        }}"""

    if split_down:
        down_body = f"""
        // 候補 (b): 1 行を 2 simdgroup (half_idx=0/1) で分担する。down では
        // 単独 simdgroup だと {_ROWS_PER_TG} 本しか埋まらなかった (残り
        // {_SIMDGROUPS - _ROWS_PER_TG} 本は常に遊んでいた) ので、1 行の内積を
        // 前半/後半に割って両方使い切る。部分和は tg_down_partial 経由で
        // 合算する (追加バリア 1 回)。合算順序は「同じ 2 項を足すだけ」
        // なので (T)acc への丸め (bf16, 8bit 仮数) 前で fp32 の LSB が
        // 入れ替わり得るだけ -- 参照との一致は bf16 丸め後で見ているので
        // 実質的な影響は無い (bench/test_hc_kernel_inject.py で確認)。
        // 変数名は half_idx (Metal では `half` が 16bit float の予約型名で
        // 変数名に使えない -- 実測でコンパイルエラーになった)。
        int rr = (int)simd_gid % {_ROWS_PER_TG};
        int half_idx = (int)simd_gid / {_ROWS_PER_TG};
        int row = tgi * {_ROWS_PER_TG} + rr;
        if (row < {lowrank}) {{
            const device uint32_t* down_w_row = down_w + (size_t)row * {words};
            const device T* s_row = down_s + (size_t)row * {n_groups};
            const device T* b_row = down_b + (size_t)row * {n_groups};
            float acc = 0.0f;
{_dot_loop_half("down_w", words, bits, gs, "(float)tg_normed[{k}]")}
            acc = simd_sum(acc);
            if (simd_lid == 0) tg_down_partial[rr * 2 + half_idx] = acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (half_idx == 0 && simd_lid == 0 && row < {lowrank}) {{
            float acc = tg_down_partial[rr * 2 + 0] + tg_down_partial[rr * 2 + 1];
            // 参照: nn.silu(qmm(normed) / hc) = x * sigmoid(x)、各段 bf16
            float xv = (float)((T)((float)((T)acc) / {float(hc)}f));
            float sg = (float)((T)(1.0f / (1.0f + metal::exp(-xv))));
            t[(size_t)m * {lowrank} + row] = (T)(xv * sg);
        }}"""
    else:
        down_body = f"""
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
        }}"""

    down_partial_decl = (
        f"threadgroup float tg_down_partial[{_ROWS_PER_TG * 2}];" if split_down else ""
    )

    return f"""
    typedef float U;

    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int  tid = (int)simd_gid * 32 + (int)simd_lid;
    int  tgi = (int)threadgroup_position_in_grid.y;
    int  m   = (int)threadgroup_position_in_grid.z;

    threadgroup float tg_part[{hc * _SIMDGROUPS}];
    threadgroup float tg_r[{hc}];
    threadgroup T     tg_normed[{hcd}];
    {down_partial_decl}

    const device T* hyper_m = hyper + (size_t)m * {hcd};

    // 1) レーンごとの rms。mx.fast.rms_norm と同じく fp32 で溜める。
    //    ついでに hyper を threadgroup メモリへ移しておく (2 パス目で device
    //    メモリを読み直さずに済む)。tg_part をレーン別の領域に広げてあるので
    //    4 レーン分の書き込みは互いに依存が無く、バリア無しで済む
    //    (元は同じ tg_part を使い回していたため「書く->barrier->読む->barrier」
    //    を 4 レーン分 = 8 回払っていた。ここでは最後に 1 回だけ同期する。
    //    各レーンの総和の相手・順序は変えていないのでビット単位で従来と同じ)
    for (int l = 0; l < {hc}; l++) {{
        float ss = 0.0f;
        for (int i = tid; i < {d}; i += {_THREADS}) {{
            T raw = hyper_m[l * {d} + i];
            tg_normed[l * {d} + i] = raw;
            float v = (float)raw;
            ss += v * v;
        }}
        ss = simd_sum(ss);
        if (simd_lid == 0) tg_part[l * {_SIMDGROUPS} + simd_gid] = ss;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {hc}) {{
        float tot = 0.0f;
        for (int q = 0; q < {_SIMDGROUPS}; q++) tot += tg_part[tid * {_SIMDGROUPS} + q];
        tg_r[tid] = metal::rsqrt(tot / {float(d)}f + {eps!r}f);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
{down_body}
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

    inject_kind = cfg["inject_kind"]
    tag = {"quant": "c", "bf16": "cb", None: "p"}[inject_kind]
    suffix = f"{cfg['hc']}x{cfg['d']}_{cfg['lowrank']}_{cfg['bits']}b{cfg['group_size']}_{tag}{len(_KERNELS)}"

    pre_inputs = ["hyper", "norm_weight", "down_w", "down_s", "down_b"]
    pre_outputs = ["t", "rlane"]
    if inject_kind == "quant":
        pre_inputs += ["inject_w", "inject_s", "inject_b"]
        pre_outputs += ["inject"]
    elif inject_kind == "bf16":
        pre_inputs += ["inject_w"]
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


def _prefill_source(cfg: dict) -> str:
    """M 大 (prefill) 向け 1 ディスパッチ版。1 行 = 1 threadgroup に畳み、
    down->silu->up->sigmoid->mix->inject を同じ threadgroup 内で
    barrier 区切りにする (threadgroup をまたぐバリアが無いので、hc_pre/hc_post
    のようにカーネルを分ける必要があるのは「複数 threadgroup が同じ行の
    down 出力を待つ」場合だけ。1 行を 1 threadgroup に閉じ込めれば
    threadgroup 内バリアで足りる)。

    hc_pre/hc_post の 2 段構成は「1 行あたり複数 threadgroup」を許して
    デコード (M 小) 向けに threadgroup 数を稼ぐ設計 (pre_tg=21、post_tg=80)。
    M が大きい prefill では threadgroup 数は M だけで GPU を埋められるので、
    その分割はむしろ次の 2 つを M 回 (= 行数ぶん) 繰り返す無駄になる:
    (a) down の出力 t と rlane を device に書いて hc_post が読み直す往復、
    (b) hyper を threadgroup メモリへ移す作業をその行の threadgroup 数だけ
        繰り返す (post 側は現に hyper と norm_weight を読み直している)。
    1 行 1 threadgroup にすれば t は threadgroup メモリに置いたままで済み、
    up 側の normed も tg_normed の再読み出しで足りる (hyper の 2 回目の
    device 読みそのものが不要になる)。

    **これで解決しない罠**: down/up の量子化重み (合わせて ~6.5MB) は
    「1 行 1 threadgroup」である限り、行 (= M) の数だけ丸ごと読み直され、
    逆量子化 (shift/mask/変換) の ALU コストも行ごとに再計算される。
    KERNEL-HANDOFF-HC.md が「S=725 で 1 forward あたり約 4.7GB x 96 回」
    「プレフィルを速くしたいなら M ブロック化が要る」と書いている通り、
    これは threadgroup メモリの持ち方やディスパッチ回数を変えても消えない
    (重みが SLC に乗って帯域では律速しなくなっても、要素あたりの逆量子化
    ALU は行の数だけ律儀に繰り返される)。本当に M に対して縮まるのは
    「重み1回の読み+逆量子化を複数行ぶんの FMA に使い回す」真の M ブロック化
    (バッチ GEMM) であり、1 行 1 threadgroup の枠内では原理的に届かない。
    ここでの改善は (a)(b) の往復排除どまりで、down/up 自体の再読み・
    再逆量子化コストはこのカーネルでも eager と同じだけ残る。
    """
    hc, d, lowrank = cfg["hc"], cfg["d"], cfg["lowrank"]
    hcd = hc * d
    bits, gs = cfg["bits"], cfg["group_size"]
    eps = cfg["eps"]
    combine = cfg["combine"]
    words = hcd * bits // 32
    n_groups = hcd // gs
    up_words = lowrank * bits // 32
    up_groups = lowrank // gs

    inject_body = ""
    if combine:
        inject_body = f"""
    // inject (hc 本)。down と同じ入力 tg_normed を読むだけの別ループ。
    // hc=4 なので simd_gid 0..3 だけが仕事をする (barrier には全スレッドが
    // 到達するので、ループ回数が 0 のスレッドグループがいても問題ない)
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
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int  tid = (int)simd_gid * 32 + (int)simd_lid;
    int  m   = (int)threadgroup_position_in_grid.z;

    threadgroup float tg_part[{_SIMDGROUPS}];
    threadgroup float tg_r[{hc}];
    threadgroup T     tg_normed[{hcd}];
    threadgroup T     tg_t[{lowrank}];

    const device T* hyper_m = hyper + (size_t)m * {hcd};

    // 1) レーンごとの rms (hc_pre と同じ式)。hyper はここで一度だけ
    //    threadgroup メモリへ移す (この行の中で以降 device 読みは発生しない)
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

    // 2) normed を threadgroup メモリに書き戻す (hc_pre と同じ丸め順)
    for (int i = tid; i < {hcd}; i += {_THREADS}) {{
        T nrm   = (T)((float)tg_normed[i] * tg_r[i / {d}]);
        T scale = (T)(1.0f + (float)norm_weight[i]);
        tg_normed[i] = (T)((float)nrm * (float)scale);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) down (hcd -> lowrank) + silu。hc_pre の tgi*ROWS_PER_TG は
    //    「複数 threadgroup で lowrank 行を分担する」割付けだったが、ここは
    //    1 threadgroup が lowrank 行を全部持つので単純なストライドでよい
    for (int row = (int)simd_gid; row < {lowrank}; row += {_SIMDGROUPS}) {{
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
            tg_t[row] = (T)(xv * sg);
        }}
    }}
{inject_body}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4) up (lowrank -> hcd) + sigmoid + レーン平均。hc_post と同じ式だが、
    //    normed の再構成に device の hyper/norm_weight を読み直す代わりに
    //    tg_normed (phase 2 で書いた値そのもの) を読む。ビット単位で同じ値
    //    なので数値は不変、hyper の 2 回目の device 読みだけが減る
    for (int j = (int)simd_gid; j < {d}; j += {_SIMDGROUPS}) {{
        float total = 0.0f;
        for (int l = 0; l < {hc}; l++) {{
            int row = l * {d} + j;
            const device uint32_t* up_w_row = up_w + (size_t)row * {up_words};
            const device T* s_row = up_s + (size_t)row * {up_groups};
            const device T* b_row = up_b + (size_t)row * {up_groups};
            float acc = 0.0f;
{_dot_loop("up_w", up_words, bits, gs, "(float)tg_t[{k}]")}
            acc = simd_sum(acc);
            if (simd_lid == 0) {{
                float sg = (float)((T)(1.0f / (1.0f + metal::exp(-(float)((T)acc)))));
                float nd = (float)tg_normed[row];
                total = (float)((T)(total + (float)((T)(sg * nd))));
            }}
        }}
        if (simd_lid == 0) mixed[(size_t)m * {d} + j] = (T)(total / {float(hc)}f);
    }}
"""


def _get_kernels_prefill(cfg: dict):
    key = tuple(sorted(cfg.items()))
    built = _KERNELS_PREFILL.get(key)
    if built is not None:
        return built

    tag = "c" if cfg["combine"] else "p"
    suffix = (
        f"{cfg['hc']}x{cfg['d']}_{cfg['lowrank']}_{cfg['bits']}b{cfg['group_size']}"
        f"_{tag}{len(_KERNELS_PREFILL)}"
    )

    inputs = ["hyper", "norm_weight", "down_w", "down_s", "down_b", "up_w", "up_s", "up_b"]
    outputs = ["mixed"]
    if cfg["combine"]:
        inputs += ["inject_w", "inject_s", "inject_b"]
        outputs += ["inject"]

    kern = mx.fast.metal_kernel(
        name=f"hc_prefill_{suffix}",
        input_names=inputs,
        output_names=outputs,
        source=_prefill_source(cfg),
    )
    _KERNELS_PREFILL[key] = kern
    return kern


def _prefill_tg_bytes(cfg: dict) -> int:
    """このカーネルが使う threadgroup メモリの総量 (バイト)。

    tg_normed (hcd 個、T) + tg_t (lowrank 個、T) + tg_r (hc 個、float) +
    tg_part (_SIMDGROUPS 個、float)。eligible() の hc*d*dtype.size 判定
    (MAX_TG_BYTES に対して margin 8KB 前後) に、lowrank ぶんの追加分が
    収まるかをここで確かめる。
    """
    hc, d, lowrank = cfg["hc"], cfg["d"], cfg["lowrank"]
    elem = cfg["dtype_size"]
    return (hc * d + lowrank) * elem + hc * 4 + _SIMDGROUPS * 4


def eligible_prefill(
    hyper: mx.array,
    norm_weight: mx.array,
    down: tuple,
    up: tuple,
    inject: tuple | None,
    hc: int,
    d: int,
    m: int,
) -> bool:
    """prefill カーネルの適格判定。形/量子化は :func:`eligible` と同じ条件に、
    threadgroup メモリの追加分 (tg_t) が収まるかと、M がしきい値以上かを足す。
    """
    if m < PREFILL_M_THRESHOLD:
        return False
    if not eligible(hyper, norm_weight, down, up, inject, hc, d):
        return False
    lowrank = down[0].shape[0]
    cfg = {
        "hc": hc, "d": d, "lowrank": lowrank,
        "dtype_size": hyper.dtype.size,
    }
    return _prefill_tg_bytes(cfg) <= MAX_TG_BYTES


def fused_gated_residual_prefill(
    hyper: mx.array,
    norm_weight: mx.array,
    eps: float,
    hc: int,
    d: int,
    down: tuple,
    up: tuple,
    inject: tuple | None,
):
    """`GatedResidual.__call__` を 1 ディスパッチ (1 行 = 1 threadgroup) で
    計算する、prefill 幅 (M 大) 向けの経路。引数・戻り値は
    :func:`fused_gated_residual` と同じ。

    既定 off、``MLXTURBO_HC_PREFILL=1`` かつ M がしきい値以上のときだけ
    ``fused.enable_hyper_connection_kernel`` から呼ばれる (ゲートは
    ``fused.py`` 側)。decode 経路 (`fused_gated_residual` / hc_pre+hc_post)
    はここでは一切変更していない。
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
    kern = _get_kernels_prefill(cfg)

    lead = hyper.shape[:-1]
    m = prod(lead) if lead else 1
    dt = hyper.dtype
    flat = hyper.reshape((m, hc * d))

    inputs = [flat, norm_weight, down[0], down[1], down[2], up[0], up[1], up[2]]
    output_shapes = [(m, d)]
    output_dtypes = [dt]
    if inject is not None:
        inputs += [inject[0], inject[1], inject[2]]
        output_shapes += [(m, hc)]
        output_dtypes += [dt]

    outs = kern(
        inputs=inputs,
        template=[("T", dt)],
        grid=(_THREADS, 1, m),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
    if inject is not None:
        mixed, inj = outs
    else:
        (mixed,) = outs
        inj = None

    mixed = mixed.reshape((*lead, d))
    if inj is None:
        return mixed
    return mixed, inj.reshape((*lead, hc))


def eligible(
    hyper: mx.array,
    norm_weight: mx.array,
    down: tuple,
    up: tuple,
    inject: tuple | None,
    hc: int,
    d: int,
) -> bool:
    """このカーネルで扱える形と量子化かを判定する。外れたら呼び出し側は素の実装へ。

    ``inject`` は None (combine なし)、量子化 5-tuple ``(w, s, b, gs, bits)``、
    または非量子化 bf16/fp16 の 2-tuple ``("bf16", weight)`` (fused.py の
    ``_pack_inject_bf16`` が作る印) のいずれか。後者は prefill 幅のカーネル
    (:func:`eligible_prefill` 経由) には渡らない -- ``fused.py`` 側が prefill
    分岐には量子化 inject しか渡さないので、ここでの `inject_bf16` 判定は
    decode 幅 (:func:`fused_gated_residual`) の呼び出しでしか True にならない。
    """

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

    inject_bf16 = inject is not None and len(inject) == 2 and inject[0] == "bf16"

    parts = [down, up]
    if inject is not None and not inject_bf16:
        parts.append(inject)
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

    # fused_gated_residual は down の bits/group_size (down[3]/[4]) だけを
    # cfg に載せてカーネルを組み、up/inject もそれで復号する。3 層のうち
    # どれか 1 つでも bits/group_size が違うと、そのバンクだけ誤った
    # 幅/グループ境界で読まれる (bits 違いは値が化ける、group_size 違いは
    # scales/biases の範囲外読み)。up 8bit だけ混ぜて 0.0586、gs=32 だけ
    # 混ぜて 0.0254 の誤差が実測されている (D-4)。
    if up[3] != down[3] or up[4] != down[4]:
        return False
    if inject is not None and not inject_bf16 and (inject[3] != down[3] or inject[4] != down[4]):
        return False

    lowrank = down[0].shape[0]
    if down[0].shape[0] != up[0].shape[1] * 32 // up[4] or up[0].shape[0] != hc * d:
        return False
    if lowrank % 32:
        # tg_t の読み込みと simdgroup 分担が半端になる形は素の実装へ
        return False

    if inject_bf16:
        w = inject[1]
        if w.dtype != hyper.dtype:
            return False
        if w.shape != (hc, hc * d):
            return False
    elif inject is not None and inject[0].shape[0] != hc:
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

    引数の ``down`` / ``up`` は ``(weight, scales, biases, group_size, bits)``。
    ``inject`` はそれに加えて非量子化 bf16/fp16 の 2-tuple
    ``("bf16", weight)`` も受け付ける (:func:`eligible` 参照)。戻り値は
    ``mixed``、``inject`` があれば ``(mixed, inject)``。
    """

    _fire.bump("hc_kernel")
    lowrank = down[0].shape[0]
    inject_bf16 = inject is not None and len(inject) == 2 and inject[0] == "bf16"
    if inject is None:
        inject_kind = None
    elif inject_bf16:
        inject_kind = "bf16"
    else:
        inject_kind = "quant"
    cfg = {
        "hc": hc,
        "d": d,
        "lowrank": lowrank,
        "bits": down[4],
        "group_size": down[3],
        "eps": float(eps),
        "inject_kind": inject_kind,
    }
    pre, post = _get_kernels(cfg)

    lead = hyper.shape[:-1]
    m = prod(lead) if lead else 1
    dt = hyper.dtype
    flat = hyper.reshape((m, hc * d))

    n_down_tg = (lowrank + _ROWS_PER_TG - 1) // _ROWS_PER_TG
    # _pre_source の split_down/fold_inject と同じ条件をここでも評価する
    # (grid.y = pre_tg の決め方がカーネル本体の tgi 割り付けと一致していないと
    # down/inject の行が抜け落ちる)。
    words = hc * d * down[4] // 32
    split_down = _vec4_ok(words, down[4], down[3]) and (words // 4) >= 2
    fold_inject = (not split_down) and inject is not None and _fold_inject_ok(hc)
    # inject が down の空き simdgroup に相乗りできるなら、専用の 21 本目の
    # threadgroup は要らない (_pre_source の fold_inject 分岐を参照)。
    pre_tg = n_down_tg + (0 if (inject is None or fold_inject) else 1)

    pre_inputs = [flat, norm_weight, down[0], down[1], down[2]]
    pre_shapes = [(m, lowrank), (m, hc)]
    pre_dtypes = [dt, mx.float32]
    if inject_kind == "quant":
        pre_inputs += [inject[0], inject[1], inject[2]]
        pre_shapes += [(m, hc)]
        pre_dtypes += [dt]
    elif inject_kind == "bf16":
        pre_inputs += [inject[1]]
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


# --------------------------------------------- 第 4 変種: elementwise だけを畳む
#
# 上の 3 変種 (hc_pre/hc_post、prefill 1 ディスパッチ) は GEMV 2 本も融合の中に
# 取り込んでいる。重みを 100MB 超巡回させた冷の連鎖で測ると、この取り込みが
# そのまま負けの正体になる (CATCHUP 2026-09-03 12:00): down/up の逆量子化内積は
# 20 threadgroup / 32 simdgroup 中 10 本 / 最後は 1 lane の逐次なので、DRAM の
# レイテンシを隠す並列度が無い。MLX の qmv は出力行ごとに threadgroup を立てて
# ベクトル load するので隠せる。冷で +78us/回 の差はここから出ている。
#
# そこで GEMV は `mx.quantized_matmul` (量子化でなければ素の行列積) にそのまま
# 残し、**その前後の elementwise だけ**を自前カーネルに畳む。畳む対象は重みを
# 読まない (hyper 20KB + norm_weight 20KB + 中間 20KB 程度) ので、冷の DRAM
# レイテンシの問題そのものが起きない。
#
#   hc_elem_pre  : レーンごとの rms -> normed              (素の 3 op)
#   (qmv down)   : mx.quantized_matmul                     -- MLX のまま
#   (qmv inject) : mx.quantized_matmul / 素の行列積        -- MLX のまま
#   hc_elem_mid  : silu(down / hc)                         (素の 2 op)
#   (qmv up)     : mx.quantized_matmul                     -- MLX のまま
#   hc_elem_post : sigmoid(up) * normed の hc 平均 + inject (素の 6 op)
#
# 素の 14 ディスパッチ (combine あり。rms_norm / 1+w / 乗算 / qmv down / 除算 /
# silu (mx.compile 済みなので 1 本) / qmv up / sigmoid / 乗算 / mean / qmv inject /
# 除算 / sigmoid / 2 倍) が 6 (自前 3 + qmv 3) になる。1 要素 1 thread で、
# threadgroup ごとの重複読みは作らない (pre は 1 threadgroup = 1 レーン、
# post は 1 thread = 出力 1 要素)。
#
# 数値は素の実装と同じ順・同じ丸め位置を踏む。sigmoid も MLX 本体
# (mlx/backend/metal/kernels/unary_ops.h の `struct Sigmoid`) の式を
# **そのままの形で**書き写す (下の `_MLX_SIGMOID_HEADER`)。冒頭の「精度」の節が
# 言う 1 ulp 差は、その総当たりが `exp(-abs(x))` 側の変形だけを試していたため:
# 本体は `exp(+abs(x))` で組み立てていて、丸め位置が違う。bf16 の全ビットパターン
# (65536 個、非有限を 0 に置換) で mx.sigmoid と突き合わせると、写しは 65535/65536
# 一致、`exp(-abs(x))` 変形は 63747/65536 (97.3%) しか一致しない (2026-09-03 実測)。
# その結果、この変種の mixed / inject は **decode 幅では**素の実装とビット一致する。
#
# ただし **行数 M が増えると一致は崩れる** (2026-09-03 実測、4bit/gs64・bf16 inject、
# 素との不一致要素の割合):
#
#   M      normed      silu(down/hc)   mixed          inject
#   1/2/6  0           0               0              0
#   62     0           0               3.2e-5         0
#   512    0           0               1.3e-5         0
#   2048   7.0e-6      4.0e-4          2.5e-4         0
#
# - M<=6 (decode/verify 幅) は全段ビット一致。
# - M>=62 で post 段だけ崩れる。normed と up の入力が完全一致しているので、
#   原因は post カーネルの中 — `mean(axis=-2)` を bf16 逐次加算で模した式
#   (`_post_source` から引き継いだ、S=1 で確かめられたモデル) が M の大きい
#   col-reduce では合っていないか、写した sigmoid が全 bf16 パターン中 1 個だけ
#   外す (65535/65536) のどちらか。切り分けは未了。
# - M=2048 では `mx.fast.rms_norm` 側も縮約の形が変わるらしく normed も崩れる。
#
# **この差は in-model に出る。**`--knob hc-elem` / `hc-off` の 3 者比較
# (2026-09-03、`bench/results/hc-elem-off-*.json`) で、出力トークン列の先頭 24 は
# elem / 素 / 既定カーネルの 3 者で一致するのに tok/round は食い違った。prefill
# (M=62〜17k) がこの経路を通るため。**素とのビット一致が要るなら
# `eligible_elem` に行数の上限 (decode/verify 幅だけ通す) を足すこと。**

_KERNELS_ELEM: dict[tuple, Any] = {}

_ELEM_THREADS = 256

# MLX 本体の bf16 sigmoid をそのままの式で写す。`metal::exp(metal::abs(x))`
# (符号を反転しない側) と `(x < 0) ? y : 1 - y` の組み合わせがちょうど本体の
# 丸め位置になる。式を数学的に等価な形へ書き換えると bf16 では 1 ulp ずれる
# ので、**この 2 行は本体の写しとして触らないこと** (本体を追うときは
# mlx/backend/metal/kernels/unary_ops.h の `struct Sigmoid` を見る)。
_MLX_SIGMOID_HEADER = """
template <typename T>
inline T mlx_sigmoid(T x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
"""


def _elem_pre_source(cfg: dict) -> str:
    """レーンごとの rms_norm と (1 + weight) の適用。1 threadgroup = 1 レーン。

    grid は (threads, hc, m) なので threadgroup は hc*m 本。レーンの要素
    (d 個) は 1 threadgroup が丸ごと持つので、レーンをまたぐ重複読みは無い。
    """
    hc, d, eps = cfg["hc"], cfg["d"], cfg["eps"]
    hcd = hc * d
    tpg = _ELEM_THREADS
    nsimd = tpg // 32
    per = (d + tpg - 1) // tpg
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint sgi = simdgroup_index_in_threadgroup;
    uint sli = thread_index_in_simdgroup;
    int  l   = (int)threadgroup_position_in_grid.y;
    int  m   = (int)threadgroup_position_in_grid.z;

    threadgroup float tg_part[{nsimd}];

    const device T* xrow = hyper       + (size_t)m * {hcd} + (size_t)l * {d};
    const device T* wrow = norm_weight + (size_t)l * {d};
    device T*       orow = normed      + (size_t)m * {hcd} + (size_t)l * {d};

    // 1) レーンの二乗和。mx.fast.rms_norm と同じく fp32 で溜める。読んだ値は
    //    レジスタに残して 2 パス目で device メモリを読み直さない
    float xv[{per}];
    float ss = 0.0f;
    for (int p = 0; p < {per}; p++) {{
        int i = (int)tid + p * {tpg};
        float v = (i < {d}) ? (float)xrow[i] : 0.0f;
        xv[p] = v;
        ss += v * v;
    }}
    ss = simd_sum(ss);
    if (sli == 0) tg_part[sgi] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int q = 0; q < {nsimd}; q++) tot += tg_part[q];
    float r = metal::rsqrt(tot / {float(d)}f + {eps!r}f);

    // 2) 参照は rms_norm の出力と (1 + weight) をそれぞれ bf16 に落とすので、
    //    その丸めを再現する
    for (int p = 0; p < {per}; p++) {{
        int i = (int)tid + p * {tpg};
        if (i < {d}) {{
            T nrm   = (T)(xv[p] * r);
            T scale = (T)(1.0f + (float)wrow[i]);
            orow[i] = (T)((float)nrm * (float)scale);
        }}
    }}
"""


def _elem_mid_source(cfg: dict) -> str:
    """down の出力に silu(x / hc) をかける。lowrank 個しかないので 1 次元。"""
    hc, lowrank = cfg["hc"], cfg["lowrank"]
    return f"""
    int i = (int)thread_position_in_grid.x;
    int m = (int)thread_position_in_grid.z;
    if (i < {lowrank}) {{
        size_t off = (size_t)m * {lowrank} + (size_t)i;
        // 参照: nn.silu(qmm(normed) / hc) = x * sigmoid(x)、各段 bf16
        T x = (T)((float)down_raw[off] / {float(hc)}f);
        T s = mlx_sigmoid<T>(x);
        t[off] = (T)((float)x * (float)s);
    }}
"""


def _elem_post_source(cfg: dict) -> str:
    """sigmoid(up) * normed のレーン平均、ついでに inject の elementwise。

    出力 1 要素 = 1 thread。thread j は入力の (l*d + j) だけを hc 回読むので、
    up_raw と normed はカーネル全体でちょうど 1 回ずつ読まれる。
    """
    hc, d, combine = cfg["hc"], cfg["d"], cfg["combine"]
    hcd = hc * d
    inject_body = ""
    if combine:
        inject_body = f"""
    // inject は hc 個しかないので、先頭の thread に相乗りさせる
    // (専用ディスパッチを 1 本足すより安い)。参照: 2 * sigmoid(qmm(normed) / hc)
    if (j < {hc}) {{
        size_t io = (size_t)m * {hc} + (size_t)j;
        T x = (T)((float)inject_raw[io] / {float(hc)}f);
        T s = mlx_sigmoid<T>(x);
        inject[io] = (T)(2.0f * (float)s);
    }}"""
    return f"""
    int j = (int)thread_position_in_grid.x;
    int m = (int)thread_position_in_grid.z;

    if (j < {d}) {{
        // 参照の mean(axis=-2) は **bf16 で逐次加算**する (fp32 で溜めて最後に
        // 丸める形ではない。_post_source の同じ注記を参照)
        float total = 0.0f;
        for (int l = 0; l < {hc}; l++) {{
            size_t idx = (size_t)m * {hcd} + (size_t)l * {d} + (size_t)j;
            T sg = mlx_sigmoid<T>(up_raw[idx]);
            float prod = (float)((T)((float)sg * (float)normed[idx]));
            total = (float)((T)(total + prod));
        }}
        // bf16 で溜めた総和を hc で割る (hc=4 なら bf16 でも厳密)
        mixed[(size_t)m * {d} + (size_t)j] = (T)(total / {float(hc)}f);
    }}
{inject_body}
"""


def _get_kernels_elem(cfg: dict):
    key = tuple(sorted(cfg.items()))
    built = _KERNELS_ELEM.get(key)
    if built is not None:
        return built

    tag = "c" if cfg["combine"] else "p"
    suffix = f"{cfg['hc']}x{cfg['d']}_{cfg['lowrank']}_{tag}{len(_KERNELS_ELEM)}"

    pre = mx.fast.metal_kernel(
        name=f"hc_elem_pre_{suffix}",
        input_names=["hyper", "norm_weight"],
        output_names=["normed"],
        source=_elem_pre_source(cfg),
    )
    mid = mx.fast.metal_kernel(
        name=f"hc_elem_mid_{suffix}",
        input_names=["down_raw"],
        output_names=["t"],
        source=_elem_mid_source(cfg),
        header=_MLX_SIGMOID_HEADER,
    )
    post_inputs = ["normed", "up_raw"]
    post_outputs = ["mixed"]
    if cfg["combine"]:
        post_inputs.append("inject_raw")
        post_outputs.append("inject")
    post = mx.fast.metal_kernel(
        name=f"hc_elem_post_{suffix}",
        input_names=post_inputs,
        output_names=post_outputs,
        source=_elem_post_source(cfg),
        header=_MLX_SIGMOID_HEADER,
    )
    _KERNELS_ELEM[key] = (pre, mid, post)
    return pre, mid, post


def _elem_qmm(x: mx.array, w: tuple):
    """量子化線形 (5-tuple) と素の線形 (1-tuple) のどちらも受ける。

    ここは MLX の qmv/gemv をそのまま呼ぶ -- 第 4 変種の要点は「GEMV を
    自前カーネルに取り込まない」ことなので、この関数の中身を融合カーネルに
    差し替えると変種の意味が消える。
    """
    if len(w) == 1:
        return x @ w[0].T
    wt, sc, bi, gs, bits = w
    return mx.quantized_matmul(
        x, wt, scales=sc, biases=bi, transpose=True, group_size=gs, bits=bits
    )


def eligible_elem(
    hyper: mx.array,
    norm_weight: mx.array,
    hc: int,
    d: int,
) -> bool:
    """第 4 変種で扱える形か。

    GEMV を MLX に任せるので、:func:`eligible` と違って重みの量子化 (bits /
    group_size / 3 層の一致) を一切見ない。見るのは hyper と norm_weight の
    dtype と形だけ。
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    if hyper.dtype not in (mx.float16, mx.bfloat16):
        return False
    if norm_weight.dtype != hyper.dtype:
        return False
    if hyper.shape[-1] != hc * d:
        return False
    if norm_weight.shape != (hc * d,):
        return False
    return True


def fused_gated_residual_elem(
    hyper: mx.array,
    norm_weight: mx.array,
    eps: float,
    hc: int,
    d: int,
    down: tuple,
    up: tuple,
    inject: tuple | None,
):
    """`GatedResidual.__call__` の elementwise だけを 3 本のカーネルに畳む。

    ``down`` / ``up`` / ``inject`` は量子化 5-tuple
    ``(weight, scales, biases, group_size, bits)`` か、素の 1-tuple
    ``(weight,)``。戻り値は ``mixed``、``inject`` があれば
    ``(mixed, inject)`` (:func:`fused_gated_residual` と同じ)。
    """

    _fire.bump("hc_elem")
    lowrank = down[0].shape[0]
    combine = inject is not None
    cfg = {
        "hc": hc,
        "d": d,
        "lowrank": lowrank,
        "eps": float(eps),
        "combine": combine,
    }
    pre, mid, post = _get_kernels_elem(cfg)

    lead = hyper.shape[:-1]
    m = prod(lead) if lead else 1
    dt = hyper.dtype
    flat = hyper.reshape((m, hc * d))

    (normed,) = pre(
        inputs=[flat, norm_weight],
        template=[("T", dt)],
        grid=(_ELEM_THREADS, hc, m),
        threadgroup=(_ELEM_THREADS, 1, 1),
        output_shapes=[(m, hc * d)],
        output_dtypes=[dt],
    )

    down_raw = _elem_qmm(normed, down)
    inject_raw = _elem_qmm(normed, inject) if combine else None

    mid_x = ((lowrank + _ELEM_THREADS - 1) // _ELEM_THREADS) * _ELEM_THREADS
    (t,) = mid(
        inputs=[down_raw],
        template=[("T", dt)],
        grid=(mid_x, 1, m),
        threadgroup=(_ELEM_THREADS, 1, 1),
        output_shapes=[(m, lowrank)],
        output_dtypes=[dt],
    )

    up_raw = _elem_qmm(t, up)

    post_x = ((d + _ELEM_THREADS - 1) // _ELEM_THREADS) * _ELEM_THREADS
    post_inputs = [normed, up_raw]
    post_shapes = [(m, d)]
    post_dtypes = [dt]
    if combine:
        post_inputs.append(inject_raw)
        post_shapes.append((m, hc))
        post_dtypes.append(dt)
    outs = post(
        inputs=post_inputs,
        template=[("T", dt)],
        grid=(post_x, 1, m),
        threadgroup=(_ELEM_THREADS, 1, 1),
        output_shapes=post_shapes,
        output_dtypes=post_dtypes,
    )

    if not combine:
        return outs[0].reshape((*lead, d))
    mixed, inj = outs
    return mixed.reshape((*lead, d)), inj.reshape((*lead, hc))


__all__ = [
    "eligible",
    "fused_gated_residual",
    "eligible_prefill",
    "fused_gated_residual_prefill",
    "eligible_elem",
    "fused_gated_residual_elem",
    "PREFILL_M_THRESHOLD",
]
