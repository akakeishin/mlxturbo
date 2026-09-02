"""mlxturbo/kernels/hyper_connection.py の inject 分岐 (量子化 / 非量子化
bf16) が `GatedResidual.__call__` (mlxturbo/_vendor/qwen4_exp.py) 相当の
素朴な計算と一致するかを、モデル無し (乱数の hyper と重み) で確認する。

診断 (scratchpad/hc_fire_diag.py, 実モデル ~/models/ddalcu-mlxlm, S=1):
97 層の `GatedResidual` のうち 96 層は `block_inject_weight` が
QuantizedLinear に変換されず bf16 の `nn.Linear` のまま残っていた。
`fused_gated_residual` の inject は量子化 5-tuple しか受け付けなかったため、
`fused.py` の `_pack_quantized` が None を返し、それらの層は down/up が
量子化で問題なくても `GatedResidual.__call__` 全体が素の実装に落ちていた
(97 層中 1 層 (inject が無い mixer 層) だけ発火)。

kernels/hyper_connection.py に inject を非量子化 bf16 のまま読む分岐を足し
(`_pre_source` の "bf16" ケース)、`eligible`/`fused_gated_residual` がそれを
`("bf16", weight)` という 2-tuple の印で受け取れるようにした
(`fused.py` の `_pack_inject_bf16` が作る)。この検証はその新しい分岐と、
既存の量子化 inject 分岐 (回帰確認)、inject 無し (combine=False) の 3 通りを
実寸に近い形 (hc=4, d=2560, lowrank=320, 4bit group_size=64) で検査する。

実行: uv run python bench/test_hc_kernel_inject.py
"""

import mlx.core as mx
import mlx.nn as nn

from mlxturbo.kernels import hyper_connection as hck

mx.random.seed(0)

HC = 4
D = 2560
LOWRANK = 320
BITS = 4
GROUP_SIZE = 64
EPS = 1e-6
DTYPE = mx.bfloat16
# bf16 の丸め幅 (docs/KERNEL-HANDOFF-HC.md、hyper_connection.py 冒頭の説明と
# 同じ基準)。ここでの mixed の値域は概ね [-1, 1] で、1 bf16 ulp が約 0.0078
# (2^-7)。sigmoid はビット単位では再現できないため、閾値をまたぐ要素が
# 1 ulp を超えてずれることがある (6 seed の実測で最大 0.0117、事前調査済み)。
# 1e-2 ちょうどだと雑音で偶発的に落ちるので、2 ulp 弱の余裕を取る。
TOL = 1.5e-2


def _quantize(w):
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    return (wq, scales, biases, GROUP_SIZE, BITS)


def _qmm(x, packed):
    w, s, b, gs, bits = packed
    return mx.quantized_matmul(
        x, w, scales=s, biases=b, transpose=True, group_size=gs, bits=bits
    )


def _mixed_ref(hyper, norm_weight, down, up):
    """GatedResidual.__call__ の mixed 部分 (inject より前) の素朴な計算。"""
    x = hyper.reshape(*hyper.shape[:-1], HC, D)
    x = mx.fast.rms_norm(x, None, EPS).reshape(hyper.shape)
    normed = x * (1.0 + norm_weight)

    w = nn.silu(_qmm(normed, down) / HC)
    w = mx.sigmoid(_qmm(w, up))
    w = w.reshape(*w.shape[:-1], HC, D)
    mixed = (w * normed.reshape(*normed.shape[:-1], HC, D)).mean(axis=-2)
    return normed, mixed


def make_common(m: int):
    hyper = mx.random.normal((m, HC * D)).astype(DTYPE)
    norm_weight = (mx.random.normal((HC * D,)) * 0.1).astype(DTYPE)
    down_plain = (mx.random.normal((LOWRANK, HC * D)) * 0.02).astype(DTYPE)
    up_plain = (mx.random.normal((HC * D, LOWRANK)) * 0.02).astype(DTYPE)
    down = _quantize(down_plain)
    up = _quantize(up_plain)
    mx.eval(hyper, norm_weight, down, up)
    return hyper, norm_weight, down, up


def check(name: str, mixed, inject, ref_mixed, ref_inject) -> None:
    mx.eval(mixed, ref_mixed)
    dm = float(mx.abs(mixed.astype(mx.float32) - ref_mixed.astype(mx.float32)).max())
    if inject is None:
        print(f"[{name}] max|Δmixed|={dm:.4e} tol={TOL:.1e} -> {'OK' if dm <= TOL else 'FAIL'}")
        assert dm <= TOL, f"{name}: mixed mismatch beyond bf16 rounding tolerance"
        return
    mx.eval(inject, ref_inject)
    di = float(mx.abs(inject.astype(mx.float32) - ref_inject.astype(mx.float32)).max())
    ok = dm <= TOL and di <= TOL
    print(
        f"[{name}] max|Δmixed|={dm:.4e} max|Δinject|={di:.4e} tol={TOL:.1e} "
        f"-> {'OK' if ok else 'FAIL'}"
    )
    assert ok, f"{name}: mismatch beyond bf16 rounding tolerance"


def check_bf16_inject(m: int = 8) -> None:
    """新設の分岐: block_inject_weight が非量子化 bf16 (診断の 96/97 層に相当)。"""
    hyper, norm_weight, down, up = make_common(m)
    inject_w = (mx.random.normal((HC, HC * D)) * 0.05).astype(DTYPE)
    mx.eval(inject_w)

    assert hck.eligible(hyper, norm_weight, down, up, ("bf16", inject_w), HC, D), (
        "bf16 inject should be eligible for the fused kernel"
    )
    mixed, inject = hck.fused_gated_residual(
        hyper, norm_weight, EPS, HC, D, down, up, ("bf16", inject_w)
    )

    normed, ref_mixed = _mixed_ref(hyper, norm_weight, down, up)
    # nn.Linear(bias=False) の forward と同じ式 (self.block_inject_weight(normed))
    ref_inject = 2 * mx.sigmoid((normed @ inject_w.T) / HC)
    check("bf16 inject", mixed, inject, ref_mixed, ref_inject)


def check_quant_inject_regression(m: int = 8) -> None:
    """既存の分岐 (回帰確認): block_inject_weight が量子化されている場合。"""
    hyper, norm_weight, down, up = make_common(m)
    inject_plain = (mx.random.normal((HC, HC * D)) * 0.05).astype(DTYPE)
    inject = _quantize(inject_plain)
    mx.eval(inject)

    assert hck.eligible(hyper, norm_weight, down, up, inject, HC, D)
    mixed, inj = hck.fused_gated_residual(
        hyper, norm_weight, EPS, HC, D, down, up, inject
    )

    normed, ref_mixed = _mixed_ref(hyper, norm_weight, down, up)
    ref_inject = 2 * mx.sigmoid(_qmm(normed, inject) / HC)
    check("quant inject (regression)", mixed, inj, ref_mixed, ref_inject)


def check_no_inject(m: int = 8) -> None:
    """combine=False (block_inject_weight が無い層、診断の mixer 層に相当)。"""
    hyper, norm_weight, down, up = make_common(m)
    assert hck.eligible(hyper, norm_weight, down, up, None, HC, D)
    mixed = hck.fused_gated_residual(hyper, norm_weight, EPS, HC, D, down, up, None)

    _, ref_mixed = _mixed_ref(hyper, norm_weight, down, up)
    check("no inject", mixed, None, ref_mixed, None)


def main() -> None:
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        print("Metal not available; skipping (this kernel is GPU-only).")
        return

    print("=== 正しさ: fused vs 素朴な計算 (hc=4, d=2560, lowrank=320, 4bit g64) ===")
    check_bf16_inject()
    check_quant_inject_regression()
    check_no_inject()
    print()
    print("すべての検証を通過しました。")


if __name__ == "__main__":
    main()
