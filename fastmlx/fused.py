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
"""

from __future__ import annotations

_ORIG_HC = None
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


def disable_hyper_connection() -> None:
    global _ORIG_HC
    if _ORIG_HC is None:
        return
    import mlx_lm.models.qwen4_exp as Q

    Q.GatedResidual.__call__ = _ORIG_HC
    _ORIG_HC = None


__all__ = ["disable_hyper_connection", "enable_hyper_connection"]
