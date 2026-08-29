"""デコードの固定費を削るための融合。

Flash-Next のデコードは完全にディスパッチ律速で、一括 forward は S=16 でも
S=1 の 1.17 倍しかかからない (docs/STATUS.md)。つまりコストのほぼ全部が
「カーネルを何回起動したか」で決まる。

内訳の実測 (tools/ablate.py):

    n-gram              2.9ms   連結サイドカーで解決済み
    hyper-connections  19.9ms   96 回の呼び出し x 約 15 op、1 op あたり 20us
    MoE ルーティング     10.1ms
    GDN                 7.8ms
    残り                9.9ms

hyper-connections が最大の残り。中身は行列積 3 本と elementwise の連鎖で、
`mx.compile` に通せば elementwise がまとまる。全 96 個が同じ形なので、
重みを引数で渡せばコンパイルは 1 回で済む。

    from fastmlx import fused
    fused.enable_hyper_connection()

`mx.compile` は 1.8ms しか減らなかった (51.09 -> 49.29 ms/token)。行列積が
間に挟まって elementwise の連なりが 1-3 op ずつに分断されるためで、こちらは
参考実装として残してある。本命は Metal カーネルへの融合:

    from fastmlx import fused
    fused.enable_hyper_connection_kernel()

GDN の `RMSNormGated` も同じ形で畳める:

    from fastmlx import fused
    fused.enable_rms_norm_gated()

いずれも既定は off。中身は fastmlx/kernels/ 配下、測り方と数値の扱いは
docs/KERNEL-HANDOFF-HC.md と docs/KERNEL-BRIEF-MOE-GDN.md。
"""

from __future__ import annotations

_ORIG_HC = None
_ORIG_HC_KERNEL = None
_ORIG_RNG = None
_ORIG_MOE = None
_COMPILED = {}


def _build(hc: int, d: int, eps: float, use_combine: bool):
    """(hc, d, eps, use_combine) ごとに 1 つだけコンパイルする。"""

    import mlx.core as mx
    import mlx.nn as nn

    key = (hc, d, eps, use_combine)
    if key in _COMPILED:
        return _COMPILED[key]

    def qmm(x, w):
        """量子化済み線形。w は (weight, scales, biases, group_size, bits)。"""
        if len(w) == 1:
            return x @ w[0].T
        wt, sc, bi, gs, bits = w
        return mx.quantized_matmul(
            x, wt, scales=sc, biases=bi, transpose=True, group_size=gs, bits=bits
        )

    def core(hyper, norm_w, down_w, up_w, inject_w):
        # RMSNorm はレーンごとに統計を取る。参照実装と同じ (1 + weight) 規約
        shape = hyper.shape
        x = hyper.reshape(*shape[:-1], -1, d)
        x = mx.fast.rms_norm(x, None, eps).reshape(shape)
        normed = x * (1.0 + norm_w)

        w = nn.silu(qmm(normed, down_w) / hc)
        w = mx.sigmoid(qmm(w, up_w))
        mixed = (
            w.reshape(*w.shape[:-1], hc, d) * normed.reshape(*shape[:-1], hc, d)
        ).mean(axis=-2)
        if not use_combine:
            return mixed
        inject = 2 * mx.sigmoid(qmm(normed, inject_w) / hc)
        return mixed, inject

    fn = mx.compile(core)
    _COMPILED[key] = fn
    return fn


def enable_hyper_connection() -> None:
    """`GatedResidual.__call__` をコンパイル済みの実装に差し替える。

    重みは引数で渡す。閉じ込めるとインスタンスごとに別グラフになり、96 個
    ぶんコンパイルすることになる。
    """

    global _ORIG_HC
    import mlx_lm.models.qwen4_exp as Q

    if _ORIG_HC is not None:
        return
    _ORIG_HC = Q.GatedResidual.__call__

    def _pack(lin):
        """線形層から重みを取り出す。量子化されていれば scales/biases も。"""
        if "scales" in lin:
            return (lin.weight, lin.scales, lin.biases, lin.group_size, lin.bits)
        return (lin.weight,)

    def patched(self, hyper):
        use_combine = self.block_inject_weight is not None
        fn = _build(self.hc, self.d, self.hc_norm.eps, use_combine)
        inject_w = _pack(self.block_inject_weight) if use_combine else None
        out = fn(
            hyper,
            self.hc_norm.weight,
            _pack(self.input_mix_weight_down),
            _pack(self.input_mix_weight_up),
            inject_w,
        )
        if not use_combine:
            return out
        mixed, inject = out
        return mixed, hyper, inject

    Q.GatedResidual.__call__ = patched


def _pack_quantized(lin):
    """線形層から (weight, scales, biases, group_size, bits) を取り出す。

    量子化されていない層は融合カーネルの対象外なので None を返す。
    """

    if "scales" not in lin:
        return None
    return (lin["weight"], lin["scales"], lin["biases"], lin.group_size, lin.bits)


def enable_hyper_connection_kernel() -> None:
    """`GatedResidual.__call__` を Metal 融合カーネルに差し替える。

    形は :func:`enable_hyper_connection` と同じで、既定は off。カーネルが
    扱えない形・量子化 (:func:`fastmlx.kernels.hyper_connection.eligible`
    参照) に当たった呼び出しだけ、その場で素の実装へ落ちる。
    """

    global _ORIG_HC_KERNEL
    import mlx_lm.models.qwen4_exp as Q

    from .kernels import hyper_connection as hck

    if _ORIG_HC_KERNEL is not None:
        return
    _ORIG_HC_KERNEL = Q.GatedResidual.__call__
    orig = _ORIG_HC_KERNEL

    def patched(self, hyper):
        down = _pack_quantized(self.input_mix_weight_down)
        up = _pack_quantized(self.input_mix_weight_up)
        combine = self.block_inject_weight is not None
        inject = _pack_quantized(self.block_inject_weight) if combine else None
        if (
            down is None
            or up is None
            or (combine and inject is None)
            or not hck.eligible(hyper, self.hc_norm.weight, down, up, inject,
                                self.hc, self.d)
        ):
            return orig(self, hyper)

        out = hck.fused_gated_residual(
            hyper, self.hc_norm.weight, self.hc_norm.eps, self.hc, self.d,
            down, up, inject,
        )
        if not combine:
            return out
        mixed, inj = out
        # 素の実装と同じく hyper をそのまま返す (残差の合流で使われる)
        return mixed, hyper, inj

    Q.GatedResidual.__call__ = patched


def disable_hyper_connection_kernel() -> None:
    global _ORIG_HC_KERNEL
    if _ORIG_HC_KERNEL is None:
        return
    import mlx_lm.models.qwen4_exp as Q

    Q.GatedResidual.__call__ = _ORIG_HC_KERNEL
    _ORIG_HC_KERNEL = None


def enable_rms_norm_gated() -> None:
    """GDN の `RMSNormGated.__call__` を Metal 融合カーネルに差し替える。

    素は 6 op (rms_norm / astype / sigmoid / astype / 乗算 / astype) で、
    GDN のある 36 層で 1 回ずつ呼ばれる。実測 2.37ms/token
    (tools/ablate_gdn.py)。

    このカーネルは参照とビット完全一致する (量子化重みを読まず、gate 経路が
    fp32 のままなので、hyper-connections で問題になった bf16 sigmoid の
    1 ulp 問題が起きない)。
    """

    global _ORIG_RNG
    import mlx_lm.models.qwen4_exp as Q

    from .kernels import rms_norm_gated as rng

    if _ORIG_RNG is not None:
        return
    _ORIG_RNG = Q.RMSNormGated.__call__
    orig = _ORIG_RNG

    def patched(self, x, gate=None):
        w = self.weight
        if not rng.eligible(x, w, gate):
            return orig(self, x, gate)
        return rng.rms_norm_gated(x, w, gate, self.eps, self.activation)

    Q.RMSNormGated.__call__ = patched


def enable_moe_route() -> None:
    """`SparseMoeBlock.__call__` の top-k 選択と softmax をカーネルに差し替える。

    素はここが 7 op (astype / 行列積 / 符号反転 / argpartition / 切り出し /
    take_along_axis / softmax) で、48 層すべてで呼ばれる。ルーティングだけで
    実測 2.69ms/token (tools/ablate_moe.py、expert 行列積 8.16ms とは別勘定)。

    `gate` の行列積は MLX に残す (全出力が揃わないと top-k を始められず、
    含めるとカーネルが 2 本に割れるため)。7 op -> 3 op になる。

    top-k の順序は `mx.argpartition` と違い降順で決定的。重み付き和は順序に
    依らないが、bf16 の加算順が変わるぶん素とビット一致はしない。
    """

    global _ORIG_MOE
    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q

    from .kernels import moe_route as mr

    if _ORIG_MOE is not None:
        return
    _ORIG_MOE = Q.SparseMoeBlock.__call__
    orig = _ORIG_MOE

    def patched(self, x):
        logits = self.gate(x.astype(mx.float32))
        if not mr.eligible(logits, self.top_k):
            # MLX は遅延評価なので、ここで捨てた logits は評価されない
            return orig(self, x)
        idx, w = mr.route(logits, self.top_k)
        out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        return out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    Q.SparseMoeBlock.__call__ = patched


def disable_moe_route() -> None:
    global _ORIG_MOE
    if _ORIG_MOE is None:
        return
    import mlx_lm.models.qwen4_exp as Q

    Q.SparseMoeBlock.__call__ = _ORIG_MOE
    _ORIG_MOE = None


def disable_rms_norm_gated() -> None:
    global _ORIG_RNG
    if _ORIG_RNG is None:
        return
    import mlx_lm.models.qwen4_exp as Q

    Q.RMSNormGated.__call__ = _ORIG_RNG
    _ORIG_RNG = None


def disable_hyper_connection() -> None:
    global _ORIG_HC
    if _ORIG_HC is None:
        return
    import mlx_lm.models.qwen4_exp as Q

    Q.GatedResidual.__call__ = _ORIG_HC
    _ORIG_HC = None


__all__ = [
    "disable_hyper_connection",
    "disable_hyper_connection_kernel",
    "disable_moe_route",
    "disable_rms_norm_gated",
    "enable_hyper_connection",
    "enable_hyper_connection_kernel",
    "enable_moe_route",
    "enable_rms_norm_gated",
]
