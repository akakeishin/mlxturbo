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
_ORIG_RNG = None
_ORIG_MOE = None
_COMPILED = {}


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

    This kernel is bit-exact with the reference (it does not read quantized weights,
    and the gate path stays in fp32, so the 1 ulp problem with bf16 sigmoid that was
    an issue in hyper-connections does not arise).
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
        _patch_switch_glu()
    return counts


def _rows(lin) -> int:
    """量子化済み Linear/SwitchLinear の出力行数 (パック前)。"""
    return lin.scales.shape[-2]


_ORIG_SWITCH_GLU = None


def _patch_switch_glu() -> None:
    """SwitchGLU の gate/up gather を、_fused_* があれば 1 回にまとめる。"""
    global _ORIG_SWITCH_GLU
    import mlx.core as mx
    import mlx_lm.models.switch_layers as SL

    if _ORIG_SWITCH_GLU is not None:
        return
    _ORIG_SWITCH_GLU = SL.SwitchGLU.__call__
    orig = _ORIG_SWITCH_GLU

    import os

    # 検証フォワード (T=2..4 -> 添字 22..44) は既定の 64 に届かずソートされない。
    # ソートすると同じエキスパートを引く行が隣接し、重みタイルの再利用が効く。
    # 値は並べ替えて戻すだけなので不変。閾値は MLXTURBO_SORT_MIN で変えられる。
    sort_min = int(os.environ.get("MLXTURBO_SORT_MIN", "64"))

    def patched(self, x, indices):
        if not hasattr(self, "_fused_w"):
            return orig(self, x, indices)
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= sort_min
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = SL._gather_sort(x, indices)
        gp = self.gate_proj
        both = mx.gather_qmm(
            x, self._fused_w, self._fused_s, self._fused_b,
            rhs_indices=idx, transpose=True,
            group_size=gp.group_size, bits=gp.bits, mode=gp.mode,
            sorted_indices=do_sort,
        )
        h = self._fused_h
        x_gate, x_up = both[..., :h], both[..., h:]
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        if do_sort:
            x = SL._scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    SL.SwitchGLU.__call__ = patched


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
        _patch_switch_glu()
    return n


_ORIG_SWITCH_SORT = None


def enable_gather_sort(min_size: int = 16) -> None:
    """SwitchGLU のソート閾値だけを下げる (構造は素の 3 gather のまま)。

    既定の閾値 64 は検証フォワード (T=2..4 -> 添字 22..44) に届かず、同じ
    エキスパートを引く行が散らばったまま読まれる。ソートすれば重みタイルの
    再利用が効く。並べ替えて戻すだけなので出力の値は不変。
    """
    global _ORIG_SWITCH_SORT
    import mlx_lm.models.switch_layers as SL

    if _ORIG_SWITCH_SORT is not None:
        return
    _ORIG_SWITCH_SORT = SL.SwitchGLU.__call__

    import mlx.core as mx

    def patched(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= min_size
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = SL._gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        if do_sort:
            x = SL._scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    SL.SwitchGLU.__call__ = patched

__all__ = [
    "enable_gather_sort",
    "enable_moe_shared_fold",
    "enable_wide_projections",
    "disable_hyper_connection",
    "disable_hyper_connection_kernel",
    "disable_moe_route",
    "disable_rms_norm_gated",
    "enable_hyper_connection",
    "enable_hyper_connection_kernel",
    "enable_moe_route",
    "enable_rms_norm_gated",
]
