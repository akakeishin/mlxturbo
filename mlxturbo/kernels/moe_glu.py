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
NSIMD = 4          # 1 スレッドグループのsimdgroup 数 (128 スレッド)
ROWS_PER_TG = 32   # 1 スレッドグループが受け持つ出力行数


def _source(K: int, H: int) -> str:
    words = K // 8          # 4bit: u32 1 語に 8 値
    groups = K // GROUP_SIZE
    return f"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint pair = threadgroup_position_in_grid.z;
    const uint row0 = threadgroup_position_in_grid.y * {ROWS_PER_TG};

    const uint e = idx[pair];

    // 対の入力 x を fp32 で TG メモリへ (全 simdgroup 共有)
    threadgroup float tx[{K}];
    threadgroup float gxsum[{groups}];
    const size_t xoff = (size_t)pair * {K};
    for (uint i = sg * 32 + lane; i < {K}; i += {NSIMD} * 32) {{
        tx[i] = (float)x[xoff + i];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // グループごとの Sigma x (バイアス項用)
    for (uint g = sg * 32 + lane; g < {groups}; g += {NSIMD} * 32) {{
        float s = 0.0f;
        for (uint i = 0; i < {GROUP_SIZE}; i++) s += tx[g * {GROUP_SIZE} + i];
        gxsum[g] = s;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const size_t ebase = (size_t)e * {H};
    for (uint r = row0 + sg; r < row0 + {ROWS_PER_TG} && r < {H}; r += {NSIMD}) {{
        const size_t woff = (ebase + r) * {words};
        const size_t goff = (ebase + r) * {groups};

        float accg = 0.0f;
        float accu = 0.0f;
        // uint4 で 4 語 = 32 値をまとめて読む。{words} 語 = {words} / 4 本の uint4。
        // 1 本の uint4 (32 値) は 64 値グループの半分なので g = j4 >> 1
        const device uint4* gw4 = (const device uint4*)(gate_w + woff);
        const device uint4* uw4 = (const device uint4*)(up_w + woff);
        for (uint j4 = lane; j4 < {words} / 4; j4 += 32) {{
            const uint g = j4 >> 1;
            const float sgc = (float)gate_s[goff + g];
            const float suc = (float)up_s[goff + g];
            const uint4 wg = gw4[j4];
            const uint4 wu = uw4[j4];
            float pg = 0.0f;
            float pu = 0.0f;
            const uint base = j4 * 32;
            #pragma unroll
            for (uint w = 0; w < 4; w++) {{
                const uint bg = wg[w];
                const uint bu = wu[w];
                const uint b2 = base + w * 8;
                #pragma unroll
                for (uint i = 0; i < 8; i++) {{
                    const float xv = tx[b2 + i];
                    pg += (float)((bg >> (4 * i)) & 0xF) * xv;
                    pu += (float)((bu >> (4 * i)) & 0xF) * xv;
                }}
            }}
            accg += sgc * pg;
            accu += suc * pu;
        }}
        // バイアス項: Sigma_g b_g * (Sigma_{{i in g}} x_i)
        for (uint g = lane; g < {groups}; g += 32) {{
            accg += (float)gate_b[goff + g] * gxsum[g];
            accu += (float)up_b[goff + g] * gxsum[g];
        }}
        accg = simd_sum(accg);
        accu = simd_sum(accu);
        if (lane == 0) {{
            const float sig = 1.0f / (1.0f + metal::exp(-accg));
            out[(size_t)pair * {H} + r] = (T)(accg * sig * accu);
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


def eligible(gate_proj, up_proj) -> bool:
    """量子化・形状がこのカーネルの前提に合うか。外れたら素の経路へ。"""
    for l in (gate_proj, up_proj):
        if not hasattr(l, "scales"):
            return False
        if l.bits != BITS or l.group_size != GROUP_SIZE:
            return False
        if getattr(l, "mode", "affine") != "affine":
            return False
    K = gate_proj.weight.shape[-1] * 8
    if K % GROUP_SIZE:
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
