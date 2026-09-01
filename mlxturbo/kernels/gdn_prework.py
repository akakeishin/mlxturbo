"""GDN の前処理 (conv1d 以降・再帰カーネル手前) を 1 本の Metal カーネルに畳む。

`GatedDeltaNet.__call__` は再帰カーネル本体 (`gated_delta_kernel`) の手前に
別々の dispatch が並ぶ:

    conv1d -> silu -> q の rms_norm+スケール -> k の rms_norm+スケール
    -> 次段 conv 状態の書き出し -> g (compute_g) -> beta (sigmoid)

decode/verify 幅 (S が小さい) では 1 op あたりの要素数がごく少なく、
`hyper_connection.py` / `rms_norm_gated.py` と同じ「起動回数そのものが
コストの大半」というディスパッチ律速の構図になる。このモジュールはこの並びを
1 回の起動に畳む。

## 適用範囲

decode/verify 幅だけ (:data:`MAX_S` / :data:`MAX_M`)。prefill 幅は対象外で、
`GatedDeltaNet.__call__` 側の既存経路 (conv1d -> silu -> rms_norm -> ...) が
そのまま使われる。単一系列 (`mask is None`、右パディング無し) のみを扱う。
バッチの右パディング (`cache.lengths` あり) は `_tail_window` の分岐が要る
ため対象外にしてある -- ここは decode のホットパスを削る話で、右パディング
バッチ decode は別の頻度の低い経路。

## スレッド配置

1 threadgroup = 1 トークン位置 (`m = b*S + s`)。

1) 手すきの先頭 {n_v} スレッドが g・beta を計算する (conv とは独立な小さい
   計算なので、他スレッドが conv1d を計算している間に終わる)。
2) 全スレッドがチャネル方向に分担して conv1d (受容野 `conv_kernel_size` タップ、
   深さ方向) + silu を計算し、q/k 分は threadgroup メモリへ、v 分は直接
   出力へ書く。ついでに次段の conv 状態 (K-1 列) も -- こちらは全 s で
   同じ値になるので `s == S-1` のスレッドグループだけが書く。
3) バリアの後、32 simdgroup を q の {n_k} head + k の {n_k} head に割り当て、
   1 head (dk 個) を 1 simdgroup が `simd_sum` で縮約して rms_norm + スケールを
   書く (`rms_norm_gated.py` と同じ「1 行 = 1 simdgroup」の形)。

## 精度

conv1d は fp32 で溜めてから T (bf16/fp16) に丸める (`mx.conv1d` と同じ丸め
回数であることを実測で確認: 手書きの 4 タップ総和は mx.conv1d と fp32 で
2.98e-8 以内)。q/k の rms_norm は「参照が bf16 実体化する箇所」を 2 回とも
再現する (`bf16(bf16(x*rsqrt) * scale)`、`rms_norm_gated.py` と同じ形)。
g (compute_g) は fp32 のみで完結するので参照とほぼ完全一致 (実測
max diff 6e-8)。

**silu と beta の sigmoid は参照とビット一致しない。** `mx.sigmoid` の bf16 出力は
`metal::exp` ベースの手書き sigmoid と 1 ulp 単位でずれる (実測: beta で
最大 diff 0.0039、bf16 の 1 ulp 相当)。`hyper_connection.py` で先に踏んだのと
同じ制約で、実装の誤りではない。品質判定は `bench/quant_eval.py` の KLD で
行うこと (ビット一致を基準にしない)。
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import _fire

_KERNELS: dict[tuple, Any] = {}

# 1 threadgroup = 32 simdgroup x 32 lane。q の 16 head + k の 16 head をちょうど
# 32 simdgroup に 1:1 で割り当てる (eligible() が 2*n_k <= 32 を確かめる)。
_SIMDGROUPS = 32
_THREADS = _SIMDGROUPS * 32

# threadgroup メモリに置く tg_q/tg_k (それぞれ key_dim 個、T) の予算。
# Apple GPU の上限 32KB に対してマージンを取る (hyper_connection.py と同じ値)。
MAX_TG_BYTES = 28 * 1024

# decode/verify 幅の適格しきい値。競合 (mlx-serve) の同種融合が使っている
# 「S<=9 かつ batch*seq<=16」に合わせてある。外れたら素の経路 (conv1d ->
# silu -> rms_norm -> ...) にそのまま落ちる。
MAX_S = 9
MAX_M = 16


def _source(cfg: dict) -> str:
    n_k, n_v = cfg["n_k"], cfg["n_v"]
    dk = cfg["dk"]
    key_dim, value_dim, conv_dim = cfg["key_dim"], cfg["value_dim"], cfg["conv_dim"]
    K = cfg["K"]
    km1 = K - 1
    eps = cfg["eps"]
    q_scale, k_scale = cfg["q_scale"], cfg["k_scale"]
    two_key_dim = 2 * key_dim
    two_n_k = 2 * n_k

    return f"""
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int  tid  = (int)simd_gid * 32 + (int)simd_lid;
    int  m    = (int)threadgroup_position_in_grid.z;
    int  s    = m % S;
    int  base = m - s;   // この b の s=0 に対応する行番号 (= b * S)
    int  b    = base / S;

    threadgroup T tg_q[{key_dim}];
    threadgroup T tg_k[{key_dim}];

    // 0) g・beta。conv1d とは独立な計算なので、conv のループに入る前に
    //    手すきの先頭 {n_v} スレッドへ割り振る
    if (tid < {n_v}) {{
        int hv = tid;
        float av   = (float)a[(size_t)m * {n_v} + hv];
        float bv   = (float)b_in[(size_t)m * {n_v} + hv];
        float Alog = (float)A_log[hv];
        float dtb  = (float)dt_bias[hv];
        float xv = av + dtb;
        // softplus(x) = logaddexp(x, 0) の安定形 (nn.softplus と同じ式)
        float sp = metal::max(xv, 0.0f)
                 + metal::log(1.0f + metal::exp(-metal::fabs(xv)));
        g_out[(size_t)m * {n_v} + hv] = metal::exp(-metal::exp(Alog) * sp);
        float sg = 1.0f / (1.0f + metal::exp(-bv));
        beta_out[(size_t)m * {n_v} + hv] = (T)sg;
    }}

    // 1) conv1d (受容野 {K} タップ、深さ方向) + silu。チャネルごとに独立。
    //    conv_state_in / mixed_qkv の該当タップは、この b の行 (base 起点)
    //    からしか読まない -- 単一系列 (mask なし) の前提そのもの
    for (int c = tid; c < {conv_dim}; c += {_THREADS}) {{
        const device T* w_c = conv_w + (size_t)c * {K};
        float acc = 0.0f;
        for (int j = 0; j < {K}; j++) {{
            int idx = s + j;
            float v;
            if (idx < {km1}) {{
                v = (float)conv_state_in[((size_t)b * {km1} + idx) * {conv_dim} + c];
            }} else {{
                int sq = idx - {km1};
                v = (float)mixed_qkv[((size_t)(base + sq)) * {conv_dim} + c];
            }}
            acc += v * (float)w_c[j];
        }}
        // mx.conv1d の出力は T に丸まってから silu に渡る (実測で確認済み)
        T conv_bf = (T)acc;
        float cv = (float)conv_bf;
        float sg = 1.0f / (1.0f + metal::exp(-cv));
        T act = (T)(cv * sg);

        if (c < {key_dim}) {{
            tg_q[c] = act;
        }} else if (c < {two_key_dim}) {{
            tg_k[c - {key_dim}] = act;
        }} else {{
            v_out[(size_t)m * {value_dim} + (c - {two_key_dim})] = act;
        }}

        // 2) 次段 conv 状態 (K-1 列)。全 s で同じ値になるので、この b の
        //    最後の s だけが書く (冗長な device メモリ書き込みを避ける)
        if (s == S - 1) {{
            for (int i = 0; i < {km1}; i++) {{
                int idx2 = S + i;
                float v2;
                if (idx2 < {km1}) {{
                    v2 = (float)conv_state_in[((size_t)b * {km1} + idx2) * {conv_dim} + c];
                }} else {{
                    int sq2 = idx2 - {km1};
                    v2 = (float)mixed_qkv[((size_t)(base + sq2)) * {conv_dim} + c];
                }}
                conv_state_out[((size_t)b * {km1} + i) * {conv_dim} + c] = (T)v2;
            }}
        }}
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) q/k の rms_norm + スケール。1 head (dk 個) を 1 simdgroup が担当
    //    (rms_norm_gated.py と同じ「1 行 = 1 simdgroup」の形)
    if (simd_gid < {n_k}) {{
        int head = (int)simd_gid;
        const threadgroup T* row = tg_q + head * {dk};
        float ss = 0.0f;
        for (int i = (int)simd_lid; i < {dk}; i += 32) {{
            float v = (float)row[i];
            ss += v * v;
        }}
        ss = simd_sum(ss);
        float r = metal::rsqrt(ss / {float(dk)}f + {eps!r}f);
        device T* orow = q_out + (size_t)m * {key_dim} + head * {dk};
        for (int i = (int)simd_lid; i < {dk}; i += 32) {{
            // 参照 (inv_scale**2) * rms_norm(q, None, eps) は
            // bf16(bf16(x*rsqrt) * scale) の 2 段丸め (実測で確認済み)
            float nv = (float)((T)((float)row[i] * r));
            orow[i] = (T)(nv * {q_scale!r}f);
        }}
    }} else if (simd_gid < {two_n_k}) {{
        int head = (int)simd_gid - {n_k};
        const threadgroup T* row = tg_k + head * {dk};
        float ss = 0.0f;
        for (int i = (int)simd_lid; i < {dk}; i += 32) {{
            float v = (float)row[i];
            ss += v * v;
        }}
        ss = simd_sum(ss);
        float r = metal::rsqrt(ss / {float(dk)}f + {eps!r}f);
        device T* orow = k_out + (size_t)m * {key_dim} + head * {dk};
        for (int i = (int)simd_lid; i < {dk}; i += 32) {{
            float nv = (float)((T)((float)row[i] * r));
            orow[i] = (T)(nv * {k_scale!r}f);
        }}
    }}
"""



def _as_dtype_scalar(value: float, dtype) -> float:
    """スカラーを `dtype` に丸めてから float に戻す。

    カーネルに埋め込む定数を、参照実装が実際に掛けている値 (弱い昇格で
    テンソルの dtype に落ちたもの) に揃えるために使う。
    """
    return float(mx.array(value, dtype))


def _get_kernel(cfg: dict):
    key = tuple(sorted((k, v) for k, v in cfg.items() if k != "dtype")) + (
        ("dtype", cfg["dtype"]),
    )
    kern = _KERNELS.get(key)
    if kern is not None:
        return kern
    suffix = f"{cfg['conv_dim']}_{cfg['n_k']}x{cfg['dk']}_{cfg['n_v']}_{len(_KERNELS)}"
    kern = mx.fast.metal_kernel(
        name=f"gdn_prework_{suffix}",
        input_names=[
            "mixed_qkv", "conv_state_in", "conv_w", "a", "b_in",
            "A_log", "dt_bias", "S",
        ],
        output_names=[
            "q_out", "k_out", "v_out", "g_out", "beta_out", "conv_state_out",
        ],
        source=_source(cfg),
    )
    _KERNELS[key] = kern
    return kern


def eligible(
    mixed_qkv: mx.array,
    conv_state: mx.array,
    conv_w: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    n_k: int,
    n_v: int,
    dk: int,
    key_dim: int,
    value_dim: int,
) -> bool:
    """このカーネルで扱える形・幅・量子化かを判定する。外れたら呼び出し側は素の経路へ。"""

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    if mixed_qkv.dtype not in (mx.float16, mx.bfloat16):
        return False
    if conv_w.dtype != mixed_qkv.dtype or conv_state.dtype != mixed_qkv.dtype:
        return False
    if a.dtype != mixed_qkv.dtype or b.dtype != mixed_qkv.dtype:
        return False
    if A_log.dtype != mx.float32 or dt_bias.dtype != mx.float32:
        return False
    if mixed_qkv.ndim != 3:
        return False
    B, S, conv_dim = mixed_qkv.shape
    if S > MAX_S or B * S > MAX_M:
        return False
    if conv_dim != 2 * key_dim + value_dim or key_dim != n_k * dk:
        return False
    if conv_w.ndim != 3 or conv_w.shape[0] != conv_dim or conv_w.shape[2] != 1:
        return False
    K = conv_w.shape[1]
    if conv_state.shape != (B, K - 1, conv_dim):
        return False
    if a.shape != (B, S, n_v) or b.shape != (B, S, n_v):
        return False
    if A_log.shape != (n_v,) or dt_bias.shape != (n_v,):
        return False
    if n_k <= 0 or 2 * n_k > _SIMDGROUPS:
        return False
    if 2 * key_dim * mixed_qkv.dtype.size > MAX_TG_BYTES:
        return False
    return True


def fused_gdn_prework(
    mixed_qkv: mx.array,
    conv_state: mx.array,
    conv_w: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    n_k: int,
    n_v: int,
    dk: int,
    dv: int,
    key_dim: int,
    value_dim: int,
    eps: float = 1e-6,
):
    """`GatedDeltaNet.__call__` の conv1d 以降・再帰カーネル手前を 1 本のカーネルで計算する。

    戻り値は `(q, k, v, g, beta, conv_state_out)`。q/k/v は `(B, S, n_k|n_v, dk|dv)`、
    g は `(B, S, n_v)` (fp32、`compute_g` と同じ dtype)、beta は `(B, S, n_v)`
    (`mixed_qkv.dtype`、`mx.sigmoid(b)` と同じ dtype)、conv_state_out は
    `(B, K-1, conv_dim)` (次段の `cache[0]` にそのまま入る形)。
    """

    _fire.bump("gdn_prework")
    B, S, conv_dim = mixed_qkv.shape
    K = conv_w.shape[1]
    inv_scale = dk**-0.5
    cfg = {
        "n_k": n_k, "n_v": n_v, "dk": dk, "dv": dv,
        "key_dim": key_dim, "value_dim": value_dim, "conv_dim": conv_dim,
        "K": K, "eps": float(eps),
        # 参照は `inv_scale * mx.fast.rms_norm(...)` と書いていて、MLX の
        # 弱い昇格で Python スカラーが**相手配列の dtype に落ちてから**掛かる。
        # つまり参照が実際に掛けているのは T に丸めた値。カーネルに fp32 の
        # 全精度を埋めると、そこだけ精度が高くなって最終丸めが割れる。
        # dk=128 では q 側 (2^-7) はちょうど表せるので差が出ないが、
        # k 側 (2^-3.5) は bf16 で仮数が丸まり、1 ulp の食い違いになる。
        "q_scale": _as_dtype_scalar(inv_scale**2, mixed_qkv.dtype),
        "k_scale": _as_dtype_scalar(inv_scale, mixed_qkv.dtype),
        "dtype": mixed_qkv.dtype,
    }
    kernel = _get_kernel(cfg)

    M = B * S
    flat_qkv = mx.contiguous(mixed_qkv.reshape((M, conv_dim)))
    flat_a = mx.contiguous(a.reshape((M, n_v)))
    flat_b = mx.contiguous(b.reshape((M, n_v)))

    outs = kernel(
        inputs=[
            flat_qkv, conv_state, conv_w, flat_a, flat_b, A_log, dt_bias, S,
        ],
        template=[("T", mixed_qkv.dtype)],
        grid=(_THREADS, 1, M),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[
            (M, key_dim), (M, key_dim), (M, value_dim),
            (M, n_v), (M, n_v),
            (B, K - 1, conv_dim),
        ],
        output_dtypes=[
            mixed_qkv.dtype, mixed_qkv.dtype, mixed_qkv.dtype,
            mx.float32, mixed_qkv.dtype,
            mixed_qkv.dtype,
        ],
    )
    q_out, k_out, v_out, g_out, beta_out, conv_state_out = outs
    q_out = q_out.reshape((B, S, n_k, dk))
    k_out = k_out.reshape((B, S, n_k, dk))
    v_out = v_out.reshape((B, S, n_v, dv))
    g_out = g_out.reshape((B, S, n_v))
    beta_out = beta_out.reshape((B, S, n_v))
    return q_out, k_out, v_out, g_out, beta_out, conv_state_out


__all__ = ["eligible", "fused_gdn_prework", "MAX_S", "MAX_M", "MAX_TG_BYTES"]
