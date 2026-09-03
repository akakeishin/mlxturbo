"""GDN の `RMSNormGated` を 1 本の Metal カーネルに畳む。

**結論: 有効にする価値は無い。既定 off のまま置いてある。**
実測で **-0.01 〜 -0.11ms/token** しか縮まない。着手の根拠だった
「RMSNormGated 2.37ms」は測定の誤りだった (下の「なぜ空振りだったか」)。
加えて実モデルでは参照とビット一致しない (72 呼び出し中 14 件で、
356352 要素のうち 1-4 個が 1 ulp ずれる)。**利得がほぼ無いのに数値の
risk だけ増えるので、有効にしないこと。**

残してあるのは、同じ形の標的 (elementwise の連なり) を次に狙うときの
下敷きと、踏んだ罠の記録のため。

## なぜ空振りだったか

`tools/ablate_gdn.py` の変種で `self.in_proj_z(x)` の戻り値を捨てていた。
MLX は遅延評価なので**使われないテンソルは計算そのものが起きない**。
つまり「RMSNormGated を外した」差分に in_proj_z の行列積 (36 層で 0.57GB、
帯域で 1.4-1.9ms) が丸ごと混ざっていた。norm 本体は 0.4ms 程度しかない。

**ablation の変種を書くときは、外した op の入力が下流で消費され続けている
ことを確かめること。**消費されないと、その入力を作る計算まで一緒に消える。

## 元の狙い (以下は記録)

素の実装は 6 op:

    out = mx.fast.rms_norm(x, weight, eps)          # 1
    g   = mx.sigmoid(gate.astype(mx.float32))       # 2 (astype, sigmoid)
    return (g * out.astype(mx.float32)).astype(...) # 3 (astype, mul, astype)

GDN のある 36 層で 1 回ずつ呼ばれ、実測 **2.37ms/token** (tools/ablate_gdn.py、
振れ 0.08ms)。1 層 66us / 6 op = 11us/op で、MLX のディスパッチ費用そのもの。
データは 1 トークンあたり 48 ヘッド x 128 要素しかないので、演算量ではなく
起動回数の問題。

## hyper_connection.py との違い

量子化重みを読まないので、あちらのような「threadgroup メモリに入力を置いて
simdgroup で行を分担する」構造が要らない。1 行 (= 1 ヘッド、dv 要素) を
1 simdgroup が担当し、二乗和を `simd_sum` で縮約して書き戻すだけ。

## 精度

**ここはほぼ完全に一致させられる。**参照は gate を `astype(mx.float32)` して
から sigmoid を掛けており、**gate 経路に bf16 の丸めが無い**。bf16 に落ちるのは
`rms_norm` の出力と最終結果の 2 箇所だけで、どちらも再現できる。

hyper-connections で問題になった「MLX の bf16 sigmoid がビット一致しない」は
ここでは起きない (fp32 の sigmoid は式の違いが 1e-7 程度で、最終の bf16
丸めに埋もれる)。詳細は docs/KERNEL-HANDOFF-HC.md の「対照実験」。

## 踏んだ罠 2 つ

1. `mx.fast.rms_norm` の**出力 dtype は weight で決まる**。weight が fp32 だと
   出力も fp32 のままで bf16 の丸めが入らない。このカーネルは weight が x と
   同じ幅のときだけ引き受ける (`eligible` 参照)
2. weight の掛け方は **`bf16(bf16(x * rsqrt) * weight)`**。1 度で丸める
   `bf16(x * rsqrt * weight)` だと 26% の要素が 1 ulp ずれる
"""

from __future__ import annotations

from math import prod
from typing import Any

import mlx.core as mx

from . import _fire

_KERNELS: dict[tuple, Any] = {}

# 1 行 = 1 simdgroup。threadgroup にまとめる行数
_ROWS_PER_TG = 8


def _source(dv: int, eps: float, activation: str) -> str:
    if activation == "sigmoid":
        act = "float g = 1.0f / (1.0f + metal::exp(-gv));"
    elif activation == "silu":
        act = "float g = gv / (1.0f + metal::exp(-gv));"
    else:
        raise ValueError(f"未対応の activation: {activation!r}")

    return f"""
    uint simd_lid = thread_index_in_simdgroup;
    int  row = (int)thread_position_in_grid.y;

    const device T* xr = x + (size_t)row * {dv};
    const device T* gr = gate + (size_t)row * {dv};
    device T* orow = out + (size_t)row * {dv};

    // 二乗和。mx.fast.rms_norm と同じく fp32 で溜める
    float ss = 0.0f;
    for (int i = (int)simd_lid; i < {dv}; i += 32) {{
        float v = (float)xr[i];
        ss += v * v;
    }}
    ss = simd_sum(ss);
    float r = metal::rsqrt(ss / {float(dv)}f + {eps!r}f);

    for (int i = (int)simd_lid; i < {dv}; i += 32) {{
        // 参照 mx.fast.rms_norm(x, weight, eps) は bf16(bf16(x * rsqrt) * weight)。
        // 正規化値を先に丸めてから weight を掛け、もう一度丸める。
        // 1 度で丸める (v * r * weight) だと 26% の要素が 1 ulp ずれる
        float nv = (float)((T)((float)xr[i] * r));
        float o = (float)((T)(nv * (float)weight[i]));
        // gate は fp32 に上げてから活性化する (参照も fp32 のまま計算しており、
        // ここに bf16 の丸めは入らない)
        float gv = (float)gr[i];
        {act}
        orow[i] = (T)(g * o);
    }}
"""


def _get_kernel(dv: int, eps: float, activation: str):
    key = (dv, eps, activation)
    k = _KERNELS.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"rms_norm_gated_{dv}_{activation}_{len(_KERNELS)}",
            input_names=["x", "weight", "gate"],
            output_names=["out"],
            source=_source(dv, eps, activation),
        )
        _KERNELS[key] = k
    return k


def eligible(x: mx.array, weight: mx.array, gate: mx.array | None) -> bool:
    """このカーネルで扱える形かを判定する。外れたら呼び出し側は素の実装へ。"""

    if gate is None:
        return False
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return False
    if x.dtype not in (mx.float16, mx.bfloat16) or gate.dtype != x.dtype:
        return False
    # mx.fast.rms_norm の出力 dtype は weight で決まる。weight が fp32 だと
    # 出力も fp32 のままで、bf16 への丸めが入らない。このカーネルは
    # 「weight も x と同じ幅」= 出力が bf16 に落ちる側だけを扱う
    # (実モデルの linear_attn.norm.weight は bf16)
    if weight.dtype != x.dtype:
        return False
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        return False
    if gate.shape != x.shape:
        return False
    # 1 行を 1 simdgroup (32 lane) が担当するので、極端に短い行は素に任せる
    return x.shape[-1] >= 32


def rms_norm_gated(
    x: mx.array,
    weight: mx.array,
    gate: mx.array,
    eps: float,
    activation: str = "sigmoid",
    rows_per_tg: int | None = None,
) -> mx.array:
    """`RMSNormGated.__call__` の中身を 1 本のカーネルで計算する。

    `rows_per_tg` は 1 threadgroup にまとめる行数 (既定 :data:`_ROWS_PER_TG`)。
    decode 幅では行数が n_v=48 しかないので、既定の 8 だと threadgroup が
    6 個しか立たず 40 コアに散らない。**decode 経路からは 1 を渡すこと**
    (1 行 = 1 simdgroup = 1 threadgroup で 48 個立つ)。prefill 幅は行数が
    数万あるのでどちらでも変わらない。
    """

    _fire.bump("rms_norm_gated")
    dv = x.shape[-1]
    rows = prod(x.shape[:-1])
    kernel = _get_kernel(dv, float(eps), activation)
    # カーネルは行優先の連続配置を仮定して x + row*dv で読む。metal_kernel は
    # ストライドを見てくれないので、非連続なビューが来たらここで実体化する。
    # (GDN の out は gated_delta_update の出力で、非連続なことがある)
    flat_x = mx.contiguous(x.reshape((rows, dv)))
    flat_g = mx.contiguous(gate.reshape((rows, dv)))

    (out,) = kernel(
        inputs=[flat_x, weight, flat_g],
        template=[("T", x.dtype)],
        grid=(32, rows, 1),
        threadgroup=(32, min(rows_per_tg or _ROWS_PER_TG, max(rows, 1)), 1),
        output_shapes=[(rows, dv)],
        output_dtypes=[x.dtype],
    )
    return out.reshape(x.shape)


__all__ = ["eligible", "rms_norm_gated"]
