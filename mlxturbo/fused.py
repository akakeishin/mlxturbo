"""Fusion for cutting down decoding's fixed costs.

Flash-Next's decoding is entirely dispatch-bound: a batched forward costs only
1.17x at S=16 what it costs at S=1 (docs/STATUS.md). In other words, almost all of
the cost is determined by "how many times a kernel was launched".

Measured breakdown (tools/ablate.py):

    n-gram              2.9ms   already solved by the concatenated sidecar
    hyper-connections  19.9ms   96 calls x about 15 ops, 20us per op
    MoE routing        10.1ms
    GDN                 7.8ms
    the rest            9.9ms

hyper-connections is the largest remainder. Its body is a chain of 3 matmuls and
elementwise ops, and putting it through `mx.compile` groups the elementwise ops
together. All 96 of them have the same shape, so if the weights are passed as
arguments, compilation happens only once.

    from mlxturbo import fused
    fused.enable_hyper_connection()

`mx.compile` only cut 1.8ms (51.09 -> 49.29 ms/token). That is because the matmuls
sit in between and break the elementwise runs into stretches of 1-3 ops each, so
this one is kept as a reference implementation. The real target is fusion into a
Metal kernel:

    from mlxturbo import fused
    fused.enable_hyper_connection_kernel()

GDN's `RMSNormGated` folds down in the same shape:

    from mlxturbo import fused
    fused.enable_rms_norm_gated()

All of them are off by default. The bodies live under mlxturbo/kernels/; how they
are measured and how the numbers are handled is in docs/KERNEL-HANDOFF-HC.md and
docs/KERNEL-BRIEF-MOE-GDN.md.
"""

from __future__ import annotations

_ORIG_HC = None
_ORIG_HC_KERNEL = None
_ORIG_HC_COMBINE = None
_ORIG_RNG = None
_ORIG_MOE = None
_COMPILED = {}
_COMBINE_COMPILED = {}
_COMBINE_PLAIN: dict = {}


def _build(hc: int, d: int, eps: float, use_combine: bool):
    """Compile exactly one instance per (hc, d, eps, use_combine)."""

    import mlx.core as mx
    import mlx.nn as nn

    key = (hc, d, eps, use_combine)
    if key in _COMPILED:
        return _COMPILED[key]

    def qmm(x, w):
        """Quantized linear. w is (weight, scales, biases, group_size, bits)."""
        if len(w) == 1:
            return x @ w[0].T
        wt, sc, bi, gs, bits = w
        return mx.quantized_matmul(
            x, wt, scales=sc, biases=bi, transpose=True, group_size=gs, bits=bits
        )

    def core(hyper, norm_w, down_w, up_w, inject_w):
        # RMSNorm takes its statistics per lane. Same (1 + weight) convention as
        # the reference implementation
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
    """Replace `GatedResidual.__call__` with the compiled implementation.

    The weights are passed as arguments. Capturing them would give a separate graph
    per instance, which would mean compiling 96 times over.
    """

    global _ORIG_HC
    import mlx_lm.models.qwen4_exp as Q

    if _ORIG_HC is not None:
        return
    _ORIG_HC = Q.GatedResidual.__call__

    def _pack(lin):
        """Take the weights out of a linear layer; scales/biases too if it is quantized."""
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
    """Take (weight, scales, biases, group_size, bits) out of a linear layer.

    A layer that is not quantized is out of scope for the fused kernel, so return
    None.
    """

    if "scales" not in lin:
        return None
    return (lin["weight"], lin["scales"], lin["biases"], lin.group_size, lin.bits)


def enable_hyper_connection_kernel() -> None:
    """Replace `GatedResidual.__call__` with the fused Metal kernel.

    The shape is the same as :func:`enable_hyper_connection`, and it is off by
    default. Only those calls that hit a shape or a quantization the kernel cannot
    handle (see :func:`mlxturbo.kernels.hyper_connection.eligible`) drop to the
    plain implementation, on the spot.

    prefill 幅 (M 大) 向けの 1 ディスパッチ経路 (kernels/hyper_connection.py の
    `fused_gated_residual_prefill`) は既定 off。環境変数
    `MLXTURBO_HC_PREFILL=1` を立てたときだけ、この関数を呼んだ時点でその分岐を
    組み込む (enable_moe_verify_gather と同じゲート方式 — 呼ぶだけでは何も
    起きない)。env var が立っていなければ decode 用の hc_pre/hc_post 経路
    (下の分岐) だけが有効になり、挙動は今まで通り。
    """

    global _ORIG_HC_KERNEL
    import os

    import mlx_lm.models.qwen4_exp as Q

    from .kernels import hyper_connection as hck

    if _ORIG_HC_KERNEL is not None:
        return
    _ORIG_HC_KERNEL = Q.GatedResidual.__call__
    orig = _ORIG_HC_KERNEL

    # 起動時に 1 回だけ読む (呼び出しごとの getenv を避ける)。既定 off なので
    # このフラグが False なら以下の prefill 分岐は一度も実行されない。
    prefill_on = os.environ.get("MLXTURBO_HC_PREFILL") == "1"

    def patched(self, hyper):
        down = _pack_quantized(self.input_mix_weight_down)
        up = _pack_quantized(self.input_mix_weight_up)
        combine = self.block_inject_weight is not None
        inject = _pack_quantized(self.block_inject_weight) if combine else None
        if down is None or up is None or (combine and inject is None):
            return orig(self, hyper)

        if prefill_on:
            m = 1
            for s in hyper.shape[:-1]:
                m *= s
            if hck.eligible_prefill(hyper, self.hc_norm.weight, down, up,
                                     inject, self.hc, self.d, m):
                out = hck.fused_gated_residual_prefill(
                    hyper, self.hc_norm.weight, self.hc_norm.eps, self.hc,
                    self.d, down, up, inject,
                )
                if not combine:
                    return out
                mixed, inj = out
                return mixed, hyper, inj

        if not hck.eligible(hyper, self.hc_norm.weight, down, up, inject,
                            self.hc, self.d):
            return orig(self, hyper)

        out = hck.fused_gated_residual(
            hyper, self.hc_norm.weight, self.hc_norm.eps, self.hc, self.d,
            down, up, inject,
        )
        if not combine:
            return out
        mixed, inj = out
        # Return hyper as-is, just like the plain implementation (it is used where
        # the residual joins back)
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
    """Replace GDN's `RMSNormGated.__call__` with the fused Metal kernel.

    Plain, this is 6 ops (rms_norm / astype / sigmoid / astype / multiply / astype),
    called once in each of the 36 layers that have GDN. Measured at 2.37ms/token
    (tools/ablate_gdn.py).

    **ビット一致という上の主張は成り立っていない (2026-09-02 実測)。**
    in-model A/B (--knob rms-norm-gated、下駄を取った後) で ms/round は
    短 -0.6% / 長 -0.4% と下がるが、**tok/round が短 -2.0% / 長 -1.7% 落ちる。**
    受理率が動くということは数値が変わっているということで、ビット一致なら
    起きない。差し引きで負けるので既定 off のまま。

    A/B の対照チェックが通っていたのは、**先頭 24 トークンしか比べていない**
    ため (`head=out[:24]`)。それ以降で分岐している。
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
    """Replace the top-k selection and softmax in `SparseMoeBlock.__call__` with a kernel.

    Plain, this part is 7 ops (astype / matmul / sign flip / argpartition / slice /
    take_along_axis / softmax), and it is called in all 48 layers. Routing alone
    measures 2.69ms/token (tools/ablate_moe.py, counted separately from the 8.16ms
    of expert matmuls).

    The `gate` matmul is left in MLX (top-k cannot start until all of the outputs
    are in, and including it would split the kernel in two). 7 ops -> 3 ops.

    Unlike `mx.argpartition`, the top-k ordering is descending and deterministic. A
    weighted sum does not depend on the ordering, but since the bf16 addition order
    changes, it is not bit-identical to plain.
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
            # MLX is lazily evaluated, so the logits discarded here are never evaluated
            return orig(self, x)
        idx, w = mr.route(logits, self.top_k)
        out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        return out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    Q.SparseMoeBlock.__call__ = patched



def enable_rms_norm_gated_nofuse() -> None:
    """`enable_rms_norm_gated` と**同じ機構だけ**を入れる対照。カーネルは呼ばない。

    差し替えたメソッドの中で `eligible()` まで評価し、そのうえで必ず素の実装へ
    落とす。A (融合) と C (これ) の差がカーネルの取り分、C と B (素) の差が
    差し替えと適格判定の費用。

    この対照が要る理由: 2026-09-01 に `gdn_prework` が**一度も発火していないのに
    長文脈で +5.3% 遅い**ことが分かった。原因は `eligible()` の評価そのもので、
    層ごと・フォワードごとに走る。この関数の「空振り」という記録
    (`runner.py`) も、同じ費用に取り分が埋もれた結果かもしれない。
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
        rng.eligible(x, w, gate)  # 評価だけして結果は使わない (対照)
        return orig(self, x, gate)

    Q.RMSNormGated.__call__ = patched


def enable_moe_route_nofuse() -> None:
    """`enable_moe_route` と**同じ機構だけ**を入れる対照。ルーティングのカーネルは
    呼ばない。

    `gate` の matmul と `eligible()` まで同じように走らせ、そのうえで必ず素の
    実装へ落とす。**gate の出力は MLX が遅延評価なので捨てても実行されない**
    (本体の `patched` が同じ理屈で捨てているのと同じ)。

    対照が要る理由は `enable_rms_norm_gated_nofuse` の docstring を参照。
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
        mr.eligible(logits, self.top_k)  # 評価だけして結果は使わない (対照)
        return orig(self, x)

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


def enable_gdn_prework_kernel() -> None:
    """GatedDeltaNet の前処理 (conv1d -> silu -> q/k の rms_norm+スケール ->
    次段 conv 状態の書き出し -> g -> beta) を 1 dispatch のカーネルに畳む
    (mlxturbo/kernels/gdn_prework.py)。

    `GatedResidual`/`RMSNormGated` の enable_* とは違い、この経路は
    `GatedDeltaNet.__call__` 側にすでにあるシーム (`getattr(self,
    "_gdn_prework", False)`、``_project_in`` の ``_wide_in`` と同じ形) を
    使う。ここではそのフラグをクラス属性として立てるだけで、
    実際に使えるかどうか (形・dtype・decode/verify 幅かどうか) は毎呼び出し
    `gdn_prework.eligible` が判定する。外れれば素の経路 (conv1d -> silu ->
    rms_norm -> ...) にそのまま落ちる。

    既定 off。環境変数 `MLXTURBO_GDN_PREWORK=1` が立っているときだけ
    有効化する (enable_moe_verify_gather と同じゲート方式 -- 呼ぶだけでは
    何も起きない)。採否は in-model A/B (tools/decode_ab.py --knob
    gdn-prework) で決める。
    """
    import os

    import mlx_lm.models.qwen4_exp as Q

    if os.environ.get("MLXTURBO_GDN_PREWORK") != "1":
        return
    Q.GatedDeltaNet._gdn_prework = True


def disable_gdn_prework_kernel() -> None:
    import mlx_lm.models.qwen4_exp as Q

    Q.GatedDeltaNet._gdn_prework = False


def enable_gdn_blocked_kernel() -> None:
    """GDN の再帰を prefill 幅だけブロック化スキャンで解く
    (mlxturbo/kernels/gated_delta_blocked.py)。

    `enable_gdn_prework_kernel` と同じ形で、`GatedDeltaNet.__call__` 側の
    シーム (`getattr(self, "_gdn_blocked", False)`) を立てるだけ。実際に
    使えるかどうか (幅・形・マスクの有無) は毎呼び出し
    `gated_delta_blocked.eligible` が判定し、外れれば逐次カーネルに落ちる。

    既定 off。環境変数 `MLXTURBO_GDN_BLOCKED=1` が立っているときだけ
    有効化する (呼ぶだけでは何も起きない)。
    """
    import os

    import mlx_lm.models.qwen4_exp as Q

    if os.environ.get("MLXTURBO_GDN_BLOCKED") != "1":
        return
    Q.GatedDeltaNet._gdn_blocked = True


def disable_gdn_blocked_kernel() -> None:
    import mlx_lm.models.qwen4_exp as Q

    Q.GatedDeltaNet._gdn_blocked = False


def enable_gdn_metal_kernel() -> None:
    """GDN の再帰を oMLX (jundot/oMLX) 移植の blocked-sequential Metal
    カーネルで解く (mlxturbo/kernels/gdn_blocked_metal.py)。

    `enable_gdn_blocked_kernel` と同じ形で、`GatedDeltaNet.__call__` 側の
    シーム (`getattr(self, "_gdn_metal", False)`) を立てるだけ。実際に
    使えるかどうか (幅・形・マスクの有無) は毎呼び出し
    `gdn_blocked_metal.eligible` が判定し、外れれば `_gdn_blocked` か
    逐次カーネルに落ちる。

    既定 off。環境変数 `MLXTURBO_GDN_METAL=1` のときだけ有効化する
    (呼ぶだけでは何も起きない)。
    """
    import os

    import mlx_lm.models.qwen4_exp as Q

    if os.environ.get("MLXTURBO_GDN_METAL") != "1":
        return
    Q.GatedDeltaNet._gdn_metal = True


def disable_gdn_metal_kernel() -> None:
    import mlx_lm.models.qwen4_exp as Q

    Q.GatedDeltaNet._gdn_metal = False


def disable_hyper_connection() -> None:
    global _ORIG_HC
    if _ORIG_HC is None:
        return
    import mlx_lm.models.qwen4_exp as Q

    Q.GatedResidual.__call__ = _ORIG_HC
    _ORIG_HC = None


# --------------------------------------------------- hyper-connection の書き戻し
#
# `enable_hyper_connection_kernel` が畳んでいるのは「読み」側 (GatedResidual.
# __call__、hyper -> mixed/inject)。「書き戻し」側 (DecoderLayer._combine、
# mixed/inject を hyper へ書き戻す) は手つかずのまま素の mx 演算で、層あたり
# 2 回 (attn 用 / mlp 用)、48 層で計 96 回呼ばれる
# (`_vendor/qwen4_exp.py:DecoderLayer._combine`)。
#
#     hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)
#
# reshape と None インデックス (ExpandDims) はどちらも contiguous な末尾軸の
# 分割/結合なので view のまま (copy_shared_buffer、カーネル起動を持たない —
# mlx/backend/common/common.cpp の ExpandDims::eval、
# mlx/backend/metal/copy.cpp の reshape_gpu の copy_necessary 分岐)。実際に
# 起動するのは multiply と add の 2 個で、どちらも mx.compile の
# is_fusable() (unary/binary/ternary/broadcast) に入るため 1 個の Compiled
# カーネルに畳める (mlx/backend/metal/compiled.cpp の Compiled::eval_gpu は
# get_kernel + dispatch がそれぞれ 1 回だけ)。matmul を挟まないぶん
# GatedResidual より単純で、専用 Metal カーネルを書くまでもなく mx.compile
# だけで 1 ディスパッチに畳める。


def _build_combine(hc: int, d: int):
    """`DecoderLayer._combine` を 1 kernel に畳む。(hc, d) ごとに 1 回だけ compile する。"""

    import mlx.core as mx

    key = (hc, d)
    fn = _COMBINE_COMPILED.get(key)
    if fn is not None:
        return fn

    def core(hyper, x, inject):
        lead = x.shape[:-1]
        hyper_r = hyper.reshape(*lead, hc, d)
        mixed = hyper_r + x[..., None, :] * inject[..., None]
        return mixed.reshape(*lead, hc * d)

    compiled = mx.compile(core)
    _COMBINE_COMPILED[key] = compiled
    return compiled


def enable_hc_write() -> None:
    """`DecoderLayer._combine` (hyper-connection の書き戻し) を mx.compile で畳む。

    素の実装と op 単位で完全に同じ計算 (multiply → add) を、呼び出す順序を
    変えずにまとめて 1 kernel にするだけなので、丸めの入る余地が無くビット
    同一になる。既定 off、``MLXTURBO_HC_WRITE=1`` で
    `runner.enable_default_fusions` から呼ばれる。decode 幅 (S=1..4) でも
    prefill 幅 (S=2048) でも同じ compiled 関数を使う (mx.compile 側が
    shape ごとに自分でキャッシュする)。
    """

    global _ORIG_HC_COMBINE
    import mlx_lm.models.qwen4_exp as Q

    if _ORIG_HC_COMBINE is not None:
        return
    _ORIG_HC_COMBINE = Q.DecoderLayer._combine

    def patched(hyper, x, inject):
        hc = inject.shape[-1]
        d = x.shape[-1]
        fn = _build_combine(hc, d)
        return fn(hyper, x, inject)

    Q.DecoderLayer._combine = staticmethod(patched)


def enable_hc_write_nofuse() -> None:
    """`enable_hc_write` と**同じ差し替えの機構だけ**を入れる対照。融合はしない。

    2026-09-01 の A/B で、hc-write が長文脈で +5.2%、gdn-prework が +5.3%
    遅くなった。gdn-prework は**一度も発火していない**のに遅く、原因は
    `eligible()` の評価そのものだった (層ごと・フォワードごとに 20 個の条件)。
    つまり **knob を有効にする機構自体に費用がある。**

    そうなると「融合が効かなかった」と畳んだ過去の判定が、実は
    「融合の取り分 < 機構の費用」だった可能性が出る (`moe_route` の +0.34ms、
    `rms_norm_gated` の空振りはどれもその桁)。

    この対照は `_combine` を同じ形で差し替え、同じ per-call の Python の
    仕事 (形の読み出し、辞書引き、関数呼び出し 1 段) をしたうえで、
    **compile を通さない素の式**を呼ぶ。A (融合) と C (この対照) の差が
    融合の取り分、C と B (素) の差が差し替えの費用。
    """

    global _ORIG_HC_COMBINE
    import mlx_lm.models.qwen4_exp as Q

    if _ORIG_HC_COMBINE is not None:
        return
    _ORIG_HC_COMBINE = Q.DecoderLayer._combine

    def patched(hyper, x, inject):
        hc = inject.shape[-1]
        d = x.shape[-1]
        fn = _build_combine_plain(hc, d)
        return fn(hyper, x, inject)

    Q.DecoderLayer._combine = staticmethod(patched)


def _build_combine_plain(hc: int, d: int):
    """`_build_combine` と同じ引き当てをして、compile していない式を返す。"""

    key = (hc, d)
    fn = _COMBINE_PLAIN.get(key)
    if fn is not None:
        return fn

    def core(hyper, x, inject):
        lead = x.shape[:-1]
        hyper_r = hyper.reshape(*lead, hc, d)
        mixed = hyper_r + x[..., None, :] * inject[..., None]
        return mixed.reshape(*lead, hc * d)

    _COMBINE_PLAIN[key] = core
    return core


def disable_hc_write() -> None:
    global _ORIG_HC_COMBINE
    if _ORIG_HC_COMBINE is None:
        return
    import mlx_lm.models.qwen4_exp as Q

    Q.DecoderLayer._combine = staticmethod(_ORIG_HC_COMBINE)
    _ORIG_HC_COMBINE = None


# ------------------------------------------------------------ wide projections
#
# 同じ入力に掛かる量子化行列を出力次元で連結し、qmm のカーネル起動回数を減らす。
# 各出力行の量子化パラメータは行単位なので、連結しても数値はビット単位で不変。
# 特に効くのは行数が極端に少ない qmm (GDN の in_proj_a/b は 48 行、
# shared_expert_gate は 1 行) で、単独ではレイテンシしか買えない。
#
#   GDN:       in_proj_qkv / z / b / a   4 本 -> 1 本 (36 層)
#   Attention: q_proj / k_proj / v_proj  3 本 -> 1 本 (12 層)
#   MoE 共有:  shared gate / up / shared_expert_gate  3 本 -> 1 本 (48 層)
#   エキスパート: switch_mlp の gate / up  gather 2 回 -> 1 回 (48 層)
#
# ビット幅か group_size が揃っていないモジュールはその場で素通し (連結しない)。
# 元のモジュールは残す (rebit や保存経路が触るため)。増えるメモリは連結分。


def _cat_quantized(lins):
    """QuantizedLinear の列を出力次元で連結。(w, scales, biases, gs, bits) か、
    揃っていなければ None。"""
    import mlx.core as mx

    if any(not hasattr(l, "scales") for l in lins):
        return None
    gs, bits = lins[0].group_size, lins[0].bits
    if any(l.group_size != gs or l.bits != bits for l in lins[1:]):
        return None
    w = mx.concatenate([l.weight for l in lins], axis=0)
    sc = mx.concatenate([l.scales for l in lins], axis=0)
    bi = mx.concatenate([l.biases for l in lins], axis=0)
    mx.eval(w, sc, bi)
    return w, sc, bi, gs, bits


def disable_wide_projections(model, mtp=None) -> int:
    """`enable_wide_projections` を打ち消す。戻り値は外した数。

    連結した重みは ``_wide_in`` / ``_wide_qkv`` という属性に置いてあり、
    `_project_in` と `Attention.__call__` は「無ければ素の 4 本 / 3 本」に
    落ちる (本家 `_vendor/qwen4_exp.py` 参照)。属性を消せば素に戻る。
    A/B で交互に測るために要る。
    """
    n = 0

    def each_layer():
        for layer in model.model.layers:
            yield layer
        if mtp is not None:
            for layer in mtp.layers:
                yield layer

    for layer in each_layer():
        for mod, attr in (
            (getattr(layer, "linear_attn", None), "_wide_in"),
            (getattr(layer, "self_attn", None), "_wide_qkv"),
            (getattr(layer, "mlp", None), "_wide_shared"),
            (getattr(layer, "mlp", None), "_wide_experts"),
        ):
            if mod is not None and getattr(mod, attr, None) is not None:
                setattr(mod, attr, None)
                n += 1
    return n


def enable_wide_projections(model, mtp=None) -> dict:
    """読み込み済みモデルに連結射影を仕込む。戻り値は種類別の適用層数。"""
    import mlx.core as mx

    counts = {"gdn": 0, "attn": 0, "shared": 0, "experts": 0}

    def each_layer():
        for layer in model.model.layers:
            yield layer
        if mtp is not None:
            for layer in mtp.layers:
                yield layer

    for layer in each_layer():
        la = getattr(layer, "linear_attn", None)
        if la is not None:
            cat = _cat_quantized(
                [la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a])
            if cat is not None:
                w, sc, bi, gs, bits = cat
                c1 = la.conv_dim
                c2 = c1 + la.value_dim
                c3 = c2 + la.n_v
                la._wide_in = (w, sc, bi, gs, bits, (c1, c2, c3))
                counts["gdn"] += 1
        sa = getattr(layer, "self_attn", None)
        if sa is not None and hasattr(sa, "q_proj"):
            cat = _cat_quantized([sa.q_proj, sa.k_proj, sa.v_proj])
            if cat is not None:
                w, sc, bi, gs, bits = cat
                c1 = sa.n_heads * sa.head_dim * 2
                c2 = c1 + sa.n_kv_heads * sa.head_dim
                sa._wide_qkv = (w, sc, bi, gs, bits, (c1, c2))
                counts["attn"] += 1
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and getattr(mlp, "_router513", None) is not None:
            continue          # shared 畳み込み済み: shared/experts はそちらが持つ
        se = getattr(mlp, "shared_expert", None) if mlp is not None else None
        if se is not None:
            cat = _cat_quantized(
                [se.gate_proj, se.up_proj, mlp.shared_expert_gate])
            if cat is not None:
                w, sc, bi, gs, bits = cat
                h = _rows(se.gate_proj)
                mlp._wide_shared = (w, sc, bi, gs, bits, h)
                counts["shared"] += 1
        sw = getattr(mlp, "switch_mlp", None) if mlp is not None else None
        if sw is not None and hasattr(sw.gate_proj, "scales"):
            g, u = sw.gate_proj, sw.up_proj
            if g.group_size == u.group_size and g.bits == u.bits:
                w = mx.concatenate([g.weight, u.weight], axis=1)
                sc = mx.concatenate([g.scales, u.scales], axis=1)
                bi = mx.concatenate([g.biases, u.biases], axis=1)
                mx.eval(w, sc, bi)
                sw._fused_w, sw._fused_s, sw._fused_b = w, sc, bi
                sw._fused_h = _rows(g)
                counts["experts"] += 1
    if counts["experts"]:
        _ensure_moe_dispatch_installed()
    return counts


def _rows(lin) -> int:
    """量子化済み Linear/SwitchLinear の出力行数 (パック前)。"""
    return lin.scales.shape[-2]


# --- SwitchGLU の統合ディスパッチ ---------------------------------------
#
# 以前は enable_wide_projections (経由の _patch_switch_glu) /
# enable_gather_sort / enable_moe_glu / enable_moe_verify_gather の 4 つが
# それぞれ独立に SL.SwitchGLU.__call__ を差し替え、別々の _ORIG_* に退避
# していた。掛け順によって disable_* の復元先が壊れる (例: verify の後に
# sort を掛けてから disable_moe_verify_gather すると、sort のパッチごと
# verify 以前の実装で上書きされる) ため、1 つのディスパッチ関数 + 分岐に
# 畳む (C1、Opus 設計レビュー指摘)。
#
# 分岐の優先順位は、これまで runner.py が実際に呼んでいた順序 (wide ->
# gather_sort -> moe_glu -> moe_verify、後から掛けたパッチが前を覆う) を
# 「後から掛けた方が優先」という固定順位として書き下しただけで、
# どの条件でどの実装が走るかという既存の挙動は変えていない。
# enable_gather_sort の実装自体には素通し条件が無い (常にそこで確定する)
# ため、wide の _fused_w 経路より必ず優先される -- これは統合前から
# あった挙動 (MLXTURBO_WIDE=1 でも既定の SORT_MIN>0 では wide 経路は
# 事実上到達しない) で、ここで新たに変えたわけではない。
_MOE_DISPATCH_STOCK = None          # 素の SL.SwitchGLU.__call__ (フォールバック先)
_MOE_DISPATCH_SORT_MIN: "int | None" = None   # enable_gather_sort が設定
_MOE_DISPATCH_GLU_ON = False        # enable_moe_glu が設定
_MOE_DISPATCH_VERIFY_ON = False     # enable_moe_verify_gather が設定
_MOE_DISPATCH_WIDE_SORT_MIN: "int | None" = None  # インストール時に一度だけ読む


def _ensure_moe_dispatch_installed() -> None:
    """4 経路共通のディスパッチ関数を SL.SwitchGLU.__call__ に一度だけ入れる。

    実際にどの経路が使われるかは呼び出し時に _MOE_DISPATCH_* を見て決まる
    (このインストール自体は決定に関与しない、副作用は初回だけ)。
    """
    global _MOE_DISPATCH_STOCK, _MOE_DISPATCH_WIDE_SORT_MIN
    import os

    import mlx.core as mx
    import mlx_lm.models.switch_layers as SL

    if _MOE_DISPATCH_STOCK is not None:
        return
    _MOE_DISPATCH_STOCK = SL.SwitchGLU.__call__
    stock = _MOE_DISPATCH_STOCK

    # 検証フォワード (T=2..4 -> 添字 22..44) は既定の 64 に届かずソートされない。
    # ソートすると同じエキスパートを引く行が隣接し、重みタイルの再利用が効く。
    # 値は並べ替えて戻すだけなので不変。閾値は MLXTURBO_WIDE_SORT_MIN で変え
    # られる (enable_gather_sort 側の MLXTURBO_SORT_MIN とは別の変数 --
    # 同名だと既定 64 と既定 16 が逆向きに衝突するため改名した。この連結射影
    # 経路は既定 off の実験経路なので、改名の影響はこちら側に閉じている)。
    _MOE_DISPATCH_WIDE_SORT_MIN = int(os.environ.get("MLXTURBO_WIDE_SORT_MIN", "64"))

    def wide(self, x, indices):
        """enable_wide_projections/enable_moe_shared_fold が仕込んだ
        `_fused_w` があれば gate+up を 1 回の gather_qmm にまとめる。
        無ければ None を返して呼び手 (dispatched) に素通しさせる。"""
        if not hasattr(self, "_fused_w"):
            return None
        xx = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= _MOE_DISPATCH_WIDE_SORT_MIN
        idx = indices
        inv_order = None
        if do_sort:
            xx, idx, inv_order = SL._gather_sort(xx, indices)
        gp = self.gate_proj
        both = mx.gather_qmm(
            xx, self._fused_w, self._fused_s, self._fused_b,
            rhs_indices=idx, transpose=True,
            group_size=gp.group_size, bits=gp.bits, mode=gp.mode,
            sorted_indices=do_sort,
        )
        h = self._fused_h
        x_gate, x_up = both[..., :h], both[..., h:]
        out = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        if do_sort:
            out = SL._scatter_unsort(out, inv_order, indices.shape)
        return out.squeeze(-2)

    def gather_sort(self, x, indices, min_size):
        """enable_gather_sort: ソート閾値だけを下げる (構造は素の 3 gather
        のまま)。素通し条件を持たない -- 有効なら常にここで確定する。"""
        xx = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= min_size
        idx = indices
        inv_order = None
        if do_sort:
            xx, idx, inv_order = SL._gather_sort(xx, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(xx, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(xx, idx, sorted_indices=do_sort)
        out = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        if do_sort:
            out = SL._scatter_unsort(out, inv_order, indices.shape)
        return out.squeeze(-2)

    def moe_glu(self, x, indices):
        """enable_moe_glu: gate+up+silu*mul を自作 1 ディスパッチカーネルへ。
        適格でなければ None (素通し)。"""
        from .kernels import moe_glu as _moe_glu_kernel

        if indices.size >= 64 or not _moe_glu_kernel.eligible(
            x, self.gate_proj, self.up_proj
        ):
            return None
        topk = indices.shape[-1]
        K = x.shape[-1]
        H = self.gate_proj.scales.shape[-2]
        # down 側の gather のために enable_gather_sort と同じソートを保つ
        # (同一エキスパートを引く行を隣接させる。自作カーネル自身は順序不問)
        do_sort = indices.size >= 16
        xx = mx.expand_dims(x, (-2, -3))
        idx = indices
        inv_order = None
        if do_sort:
            xx, idx, inv_order = SL._gather_sort(xx, indices)
            x_pairs = xx.reshape(-1, K)              # ソート済み対ごとの x
            idx_flat = idx
        else:
            x_tok = x.reshape(-1, K)
            x_pairs = mx.repeat(x_tok[:, None, :], topk, axis=1).reshape(-1, K)
            idx_flat = indices.reshape(-1)
        act = _moe_glu_kernel.fused_glu(x_pairs, idx_flat, self.gate_proj, self.up_proj)
        if do_sort:
            act = act[:, None, :]                    # (P, 1, H)
            out = self.down_proj(act, idx, sorted_indices=True)
            out = SL._scatter_unsort(out, inv_order, indices.shape)
            return out.squeeze(-2)
        act = act.reshape(*indices.shape, 1, H)
        out = self.down_proj(act, indices, sorted_indices=False)
        return out.squeeze(-2)

    def moe_verify(self, x, indices):
        """enable_moe_verify_gather: verify 幅だけ v2 (gate+up 融合 + down)
        へ。適格でなければ None (素通し)。indices.size >= 64 (mlx_lm 自身の
        ソート閾値、moe_glu と同じ基準) は prefill 幅とみなして素通し。"""
        from .kernels import moe_verify_gather as mvg

        if indices.size >= 64:
            return None
        gp, up, dp = self.gate_proj, self.up_proj, self.down_proj
        if not (mvg.eligible_gate_up(x, gp, up) and mvg.eligible_down(dp)):
            return None
        K = x.shape[-1]
        H = gp.scales.shape[-2]
        # S (1 verify ラウンドのトークン数) は形状だけから決まる静的な値。
        # gather_gate_up/gather_down 側の同期なしセグメント計算 (`_max_seg_bound`)
        # がこれを使ってセグメント長の静的上限を出す (実データは読まない)。
        #
        # **`indices.shape[-2]` にしないこと。**バッチ検証では indices が
        # (B, T, top_k) になり、それは T であってトークン数ではない。
        # `_max_seg_bound` の安全証明は「セグメント長 <= トークン数」なので、
        # B=2, T=3 のとき実トークン数 6 に対し上限 3 のカーネルが選ばれ、
        # あふれた行が書かれないまま返る (metal_kernel の出力は零初期化
        # されないので、例外も出ずにゴミが混ざる)。
        S = indices.size // indices.shape[-1]
        xx = mx.expand_dims(x, (-2, -3))
        xx, idx, inv_order = SL._gather_sort(xx, indices)
        x_sorted = xx.reshape(-1, K)
        act = mvg.gather_gate_up(
            x_sorted, idx, gp.weight, gp.scales, gp.biases,
            up.weight, up.scales, up.biases, K, H, S,
        )
        # down カーネルは (P, H) の2次元で返る。gather_qmm 経由の素の実装は
        # M=1 の中間次元を挟むが、自作カーネルは最初から持たないので
        # [:, None, :] や squeeze(-2) は不要 (挟むと形が壊れる)
        out = mvg.gather_down(act, idx, dp.weight, dp.scales, dp.biases, H, K, S)
        out = SL._scatter_unsort(out, inv_order, indices.shape)
        return out

    def dispatched(self, x, indices):
        if _MOE_DISPATCH_VERIFY_ON:
            out = moe_verify(self, x, indices)
            if out is not None:
                return out
        if _MOE_DISPATCH_GLU_ON:
            out = moe_glu(self, x, indices)
            if out is not None:
                return out
        if _MOE_DISPATCH_SORT_MIN is not None:
            return gather_sort(self, x, indices, _MOE_DISPATCH_SORT_MIN)
        out = wide(self, x, indices)
        if out is not None:
            return out
        return stock(self, x, indices)

    SL.SwitchGLU.__call__ = dispatched


def enable_moe_shared_fold(model) -> int:
    """shared expert を「常に選ばれる 513 番目のエキスパート」に畳み込む。

    shared_expert_intermediate_size (640) はエキスパートと同一なので、
    gate/up/down の各バンクに shared の行を 1 枚足し、毎トークンの添字に
    512 を固定で追加すれば、shared の qmm 3 本 + gate 1 本が直列経路から消える
    (gather はエキスパート方向に並列なので、1 本増えても直列時間は伸びない)。

    sigmoid(shared_expert_gate(x)) は router に行を足して同じ行列積から取る。
    router は逆量子化して fp32 の密行列にする (513 行 x 2560、5MB。読み出しは
    誤差の内で、量子化 router の qmm と密 fp32 の行列積は同じ値を出す)。

    重み和の足し込み順が変わるぶんだけ bf16 の丸めが動く (moe_route カーネルと
    同じ種類の差)。ビット/gs が揃っていない層は素通し。
    """
    import mlx.core as mx

    def deq(lin):
        if hasattr(lin, "scales"):
            return mx.dequantize(
                lin.weight, lin.scales, lin.biases,
                group_size=lin.group_size, bits=lin.bits)
        return lin.weight

    n = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        se = getattr(mlp, "shared_expert", None) if mlp is not None else None
        sw = getattr(mlp, "switch_mlp", None) if mlp is not None else None
        if se is None or sw is None:
            continue
        parts = [sw.gate_proj, sw.up_proj, sw.down_proj,
                 se.gate_proj, se.up_proj, se.down_proj]
        if any(not hasattr(x, "scales") for x in parts):
            continue
        gs, bits = sw.gate_proj.group_size, sw.gate_proj.bits
        if any(x.group_size != gs or x.bits != bits for x in parts):
            continue

        def bank(swl, lin, attr):
            return mx.concatenate(
                [getattr(swl, attr), getattr(lin, attr)[None]], axis=0)

        g, u, d = sw.gate_proj, sw.up_proj, sw.down_proj
        fw = mx.concatenate(
            [bank(g, se.gate_proj, "weight"), bank(u, se.up_proj, "weight")], axis=1)
        fs = mx.concatenate(
            [bank(g, se.gate_proj, "scales"), bank(u, se.up_proj, "scales")], axis=1)
        fb = mx.concatenate(
            [bank(g, se.gate_proj, "biases"), bank(u, se.up_proj, "biases")], axis=1)
        dw = bank(d, se.down_proj, "weight")
        ds = bank(d, se.down_proj, "scales")
        db = bank(d, se.down_proj, "biases")
        router = mx.concatenate(
            [deq(mlp.gate), deq(mlp.shared_expert_gate)], axis=0
        ).astype(mx.float32)
        mx.eval(fw, fs, fb, dw, ds, db, router)

        sw._fused_w, sw._fused_s, sw._fused_b = fw, fs, fb
        sw._fused_h = _rows(g)
        d.weight, d.scales, d.biases = dw, ds, db
        mlp._router513 = router
        n += 1
    if n:
        _ensure_moe_dispatch_installed()
    return n


def enable_gather_sort(min_size: int = 16) -> None:
    """SwitchGLU のソート閾値だけを下げる (構造は素の 3 gather のまま)。

    既定の閾値 64 は検証フォワード (T=2..4 -> 添字 22..44) に届かず、同じ
    エキスパートを引く行が散らばったまま読まれる。ソートすれば重みタイルの
    再利用が効く。並べ替えて戻すだけなので出力の値は不変。

    実体は _ensure_moe_dispatch_installed が入れる統合ディスパッチ関数
    (C1) の一分岐。この関数自体は「gather_sort 経路を有効にして min_size
    を確定する」だけで、以後の呼び出しは冪等 (最初の min_size が残る --
    以前の実装もそうだった)。
    """
    global _MOE_DISPATCH_SORT_MIN
    if _MOE_DISPATCH_SORT_MIN is not None:
        return
    _MOE_DISPATCH_SORT_MIN = min_size
    _ensure_moe_dispatch_installed()


def enable_moe_glu() -> None:
    """SwitchGLU の gate+up+silu*mul を自作 1 ディスパッチカーネルに差し替える。

    kernels/moe_glu.py (qmv_fast 構造)。down_proj はそのまま gather_qmm。
    適格でない層・経路 (量子化が 4bit/gs64 でない、prefill のソート済み大バッチ
    など) は素の実装へ落ちる。数値は gather_qmm x2 + swiglu とビット一致しない
    (積和順と bf16 sigmoid)。判定は in-model の複数プロンプト平均で。

    実体は _ensure_moe_dispatch_installed が入れる統合ディスパッチ関数
    (C1) の一分岐。
    """
    global _MOE_DISPATCH_GLU_ON
    if _MOE_DISPATCH_GLU_ON:
        return
    _MOE_DISPATCH_GLU_ON = True
    _ensure_moe_dispatch_installed()


def enable_moe_verify_gather() -> None:
    """SwitchGLU を kernels/moe_verify_gather.py の v2 (gate+up 融合 + down) に
    差し替える。verify (decode) 幅だけが対象で、既定は off。

    実験的なカーネルなので、この関数を呼ぶだけでは何も起きない —
    環境変数 `MLXTURBO_MOE_VERIFY=1` が立っているときだけパッチが入る
    (呼び出し側が env var を忘れても既定 off が保たれるように、ゲートを
    関数自身の中に持たせている)。indices.size >= 64 (mlx_lm 自身のソート
    閾値、enable_moe_glu と同じ基準) は prefill 幅とみなして素の経路のまま。

    実体は _ensure_moe_dispatch_installed が入れる統合ディスパッチ関数
    (C1) の一分岐。
    """
    global _MOE_DISPATCH_VERIFY_ON
    import os

    if os.environ.get("MLXTURBO_MOE_VERIFY") != "1":
        return
    if _MOE_DISPATCH_VERIFY_ON:
        return
    _MOE_DISPATCH_VERIFY_ON = True
    _ensure_moe_dispatch_installed()


def disable_moe_verify_gather() -> None:
    """enable_moe_verify_gather を打ち消す。統合ディスパッチ (C1) では
    フラグを下ろすだけでよく、他の 3 経路 (wide/gather_sort/moe_glu) が
    その時点で有効かどうかに関係なく、それらの分岐だけを正しく戻す
    (以前のクラスメソッド丸ごと差し替え方式では、これが掛け順によって
    壊れていた)。"""
    global _MOE_DISPATCH_VERIFY_ON
    _MOE_DISPATCH_VERIFY_ON = False


def enable_fast_rope(model) -> int:
    """decode の attention 層で QK-norm 後の rope を `mx.fast.rope` 1 dispatch
    に畳む (``Attention._qkv`` の ``_fast_rope`` 分岐、
    `mlxturbo/_vendor/qwen4_exp.py`)。素の経路は cos/sin の生成
    (`RotaryEmbedding.__call__`、mx.cos/mx.sin) + `_rope_partial` x2
    (各回 slice x2 + concatenate x2) を op のまま積む。層あたり
    cos 1 / sin 1 / concatenate 6 / (rms_norm は変えない) が減る計算。

    CPU 上の合成入力で `mx.fast.rope(dims=64, traditional=False, base=1e7)`
    が `RotaryEmbedding` + `_rope_partial` と (テキストのみ、offset+arange(S)
    の位置に対して) 浮動小数の丸み差だけで一致することを確認済み
    (`docs/research/KERNEL-BRIEF-DECODE-BW.md`)。出力はビット不一致
    (積和の順が変わる) だが、既定の丸め誤差の範囲内 --- 採否は
    `tools/decode_ab.py --knob fast-rope` の in-model 計測で決める。

    実験的な分岐なので、この関数を呼ぶだけでは何も起きない --- 環境変数
    `MLXTURBO_FAST_ROPE=1` が立っているときだけ `_fast_rope` を立てる
    (呼び出し側が env var を忘れても既定 off が保たれるように、ゲートを
    関数自身の中に持たせている、`enable_moe_verify_gather` と同じ作法)。

    `Attention._qkv` 側にも実行時ガードがある: バッチ経路
    (`mlxturbo/batch.py` / `batch_spec.py`) が `Attention._positions` を
    差し替えている間は、この属性が立っていても素の経路に落ちる (パディング・
    dead slot で positions が offset+arange(S) の形を保証できないため)。

    戻り値は適用した層数 (full_attention 層のみ。linear_attention 層は
    `self_attn` を持たない)。
    """
    import os

    if os.environ.get("MLXTURBO_FAST_ROPE") != "1":
        return 0
    n = 0
    for layer in model.model.layers:
        sa = getattr(layer, "self_attn", None)
        if sa is not None and hasattr(sa, "q_norm"):
            sa._fast_rope = True
            n += 1
    return n


def disable_fast_rope(model) -> int:
    """`enable_fast_rope` を打ち消す。戻り値は外した数。A/B で交互に測るために要る。"""
    n = 0
    for layer in model.model.layers:
        sa = getattr(layer, "self_attn", None)
        if sa is not None and getattr(sa, "_fast_rope", False):
            sa._fast_rope = False
            n += 1
    return n


__all__ = [
    "enable_gather_sort",
    "enable_gdn_blocked_kernel",
    "enable_gdn_metal_kernel",
    "enable_gdn_prework_kernel",
    "enable_moe_glu",
    "enable_moe_shared_fold",
    "enable_moe_verify_gather",
    "enable_wide_projections",
    "disable_gdn_blocked_kernel",
    "disable_gdn_metal_kernel",
    "disable_gdn_prework_kernel",
    "disable_hc_write",
    "disable_hyper_connection",
    "disable_hyper_connection_kernel",
    "disable_moe_route",
    "disable_moe_verify_gather",
    "disable_rms_norm_gated",
    "enable_hc_write",
    "enable_hc_write_nofuse",
    "enable_hyper_connection",
    "enable_hyper_connection_kernel",
    "enable_moe_route",
    "enable_moe_route_nofuse",
    "enable_rms_norm_gated",
    "enable_rms_norm_gated_nofuse",
    "enable_fast_rope",
    "disable_fast_rope",
]
