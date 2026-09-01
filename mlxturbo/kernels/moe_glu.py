"""MoE エキスパートの gate+up+silu*mul を 1 本の Metal カーネルに畳む。

素の SwitchGLU は 1 層あたり gather_qmm x2 (gate, up) + swiglu の 3 ディスパッチ。
検証フォワード (T=2..4、対 22..44 個) では、対ごとの行列が 0.8MB と小さく、
カーネルごとの読みのランプアップが償却できない (DECODE-ANATOMY-2026-08-31.md:
小さい読みは ~160GB/s、大きい連続読みは 290-350GB/s)。ここでは 1 つの
スレッドグループが同じエキスパートの gate 行と up 行を続けて読み、
activation まで済ませて 1 ディスパッチにする。

数値: 積和は fp32 で溜め、silu(gate) * up を fp32 で計算して bf16 で丸める。
gather_qmm + 素の swiglu ともビット一致はしない (積和の順序と、bf16 sigmoid の
1 ulp 問題 = kernels/hyper_connection.py の「精度」の節と同じ性質)。品質は
KLD、速度と受理率は in-model の複数プロンプト平均で判定すること。
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

_KERNELS: dict[tuple, Any] = {}

GROUP_SIZE = 64
BITS = 4
NSIMD = 2          # qmv_fast と同じ simdgroup 2 本 (64 スレッド)
ROWS_PER_TG = NSIMD * 4   # qmv_fast と同じく simdgroup あたり 4 行


def _source(K: int, H: int) -> str:
    """qmv_fast_impl (mlx quantized.h) 準拠の v4:
    simdgroup 2 本 x 4 行、スレッドあたり 16 値。x は vec4、重みは uint2 で
    ロードし、バイアスは sum(x) に畳む。gate/up の 2 行列を同じ x レジスタで
    続けて処理する。"""
    assert K % 512 == 0, "block=512 の等分が前提"
    n_iters = K // 512
    return f"""
    constexpr int VPT = 16;
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint pair = threadgroup_position_in_grid.z;
    const uint row0 = threadgroup_position_in_grid.y * ({NSIMD} * 4) + sg * 4;
    if (row0 >= {H}) return;

    const uint e = idx[pair];
    const size_t wrow2 = (size_t)({K} / 16);       // uint2 / 行
    const size_t grow = (size_t)({K} / 64);
    const size_t ebase = (size_t)e * {H};

    const device uint2* gw2 = (const device uint2*)gate_w + (ebase + row0) * wrow2 + lane;
    const device uint2* uw2 = (const device uint2*)up_w + (ebase + row0) * wrow2 + lane;
    const device T* gsl = gate_s + (ebase + row0) * grow;
    const device T* gbl = gate_b + (ebase + row0) * grow;
    const device T* usl = up_s + (ebase + row0) * grow;
    const device T* ubl = up_b + (ebase + row0) * grow;
    const device vec<T, 4>* xv =
        (const device vec<T, 4>*)(x + (size_t)pair * {K}) + lane * 4;

    const uint gofs = lane / 4;
    float rg[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    float ru[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    float xt[VPT];

    for (int it = 0; it < {n_iters}; it++) {{
        float xsum = 0.0f;
        #pragma unroll
        for (int v = 0; v < 4; v++) {{
            const vec<T, 4> xx = xv[v];
            #pragma unroll
            for (int i = 0; i < 4; i++) {{
                xt[v * 4 + i] = (float)xx[i];
                xsum += xt[v * 4 + i];
            }}
        }}
        const uint gbase = it * 8 + gofs;
        #pragma unroll
        for (int r = 0; r < 4; r++) {{
            const uint2 wg = gw2[(size_t)r * wrow2];
            const uint2 wu = uw2[(size_t)r * wrow2];
            float ag = 0.0f;
            float au = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {{
                ag += xt[i] * (float)((wg.x >> (4 * i)) & 0xF)
                    + xt[8 + i] * (float)((wg.y >> (4 * i)) & 0xF);
                au += xt[i] * (float)((wu.x >> (4 * i)) & 0xF)
                    + xt[8 + i] * (float)((wu.y >> (4 * i)) & 0xF);
            }}
            rg[r] += (float)gsl[r * grow + gbase] * ag + (float)gbl[r * grow + gbase] * xsum;
            ru[r] += (float)usl[r * grow + gbase] * au + (float)ubl[r * grow + gbase] * xsum;
        }}
        gw2 += 32;
        uw2 += 32;
        xv += 128;
    }}
    #pragma unroll
    for (int r = 0; r < 4; r++) {{
        float g = simd_sum(rg[r]);
        float u = simd_sum(ru[r]);
        if (lane == 0 && row0 + r < {H}) {{
            const float sig = 1.0f / (1.0f + metal::exp(-g));
            out[(size_t)pair * {H} + row0 + r] = (T)(g * sig * u);
        }}
    }}
"""


def _get_kernel(K: int, H: int):
    key = (K, H)
    k = _KERNELS.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"moe_glu_{K}x{H}",
            input_names=["x", "idx", "gate_w", "gate_s", "gate_b",
                         "up_w", "up_s", "up_b"],
            output_names=["out"],
            source=_source(K, H),
        )
        _KERNELS[key] = k
    return k


def eligible(x, gate_proj, up_proj) -> bool:
    """量子化・形状がこのカーネルの前提に合うか。外れたら素の経路へ。

    カーネルは `template=[("T", mx.bfloat16)]` で T を bf16 に固定している
    (fused_glu 参照)。x が bf16 以外 (fp16/fp32 のモデル) だとバッファを
    誤った幅で読む静かな誤りになるため、ここで弾く。
    """
    if x.dtype != mx.bfloat16:
        return False
    for l in (gate_proj, up_proj):
        if not hasattr(l, "scales"):
            return False
        if l.bits != BITS or l.group_size != GROUP_SIZE:
            return False
        if getattr(l, "mode", "affine") != "affine":
            return False
    K = gate_proj.weight.shape[-1] * 8
    if K % 512:
        return False
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def fused_glu(x_pairs: mx.array, idx_flat: mx.array, gate_proj, up_proj) -> mx.array:
    """x_pairs (P, K) bf16 と idx (P,) から silu(gate)*up (P, H) bf16 を返す。

    注意: MLX の量子化は q が符号なし 4bit で scales/biases が bf16。
    ここでは w = s*q + b の affine 展開をそのまま積和に入れている。
    """
    P, K = x_pairs.shape
    H = gate_proj.scales.shape[-2]
    kern = _get_kernel(K, H)
    (out,) = kern(
        inputs=[x_pairs, idx_flat.astype(mx.uint32),
                gate_proj.weight.reshape(-1, gate_proj.weight.shape[-1]),
                gate_proj.scales.reshape(-1, gate_proj.scales.shape[-1]),
                gate_proj.biases.reshape(-1, gate_proj.biases.shape[-1]),
                up_proj.weight.reshape(-1, up_proj.weight.shape[-1]),
                up_proj.scales.reshape(-1, up_proj.scales.shape[-1]),
                up_proj.biases.reshape(-1, up_proj.biases.shape[-1])],
        template=[("T", mx.bfloat16)],
        output_shapes=[(P, H)],
        output_dtypes=[mx.bfloat16],
        grid=(32 * NSIMD, (H + ROWS_PER_TG - 1) // ROWS_PER_TG, P),
        threadgroup=(32 * NSIMD, 1, 1),
    )
    return out
