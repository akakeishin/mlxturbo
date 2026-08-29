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
