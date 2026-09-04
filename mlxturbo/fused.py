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

from contextlib import contextmanager

_ORIG_HC = None
_ORIG_HC_KERNEL = None
_ORIG_HC_COMBINE = None
_ORIG_HC_PREFILL_COMPILE = None
_ORIG_RNG = None
_ORIG_MOE = None
_COMPILED = {}
_COMBINE_COMPILED = {}
_COMBINE_PLAIN: dict = {}
_HC_PREFILL_COMPILE_PRE: dict = {}
_HC_PREFILL_COMPILE_POST: dict = {}


# --- 層の列挙 (族ごとにラッパの形が違う) --------------------------------
#
# `enable_default_fusions` は model_type で分岐せずに全部の enable_* を呼ぶ。
# クラスを差し替えるだけのもの (`Q.<Class>.<attr> = ...`) は他の族では
# 単に呼ばれないので素通りするが、**モデルの構造を直接舐める**もの
# (`model.model.layers`) は族が違うと AttributeError で落ちる。
#
#   qwen4_exp (Flash-Next)       model.model.layers
#   qwen3_5   (Qwen3.8-27B)      model.language_model.model.layers
#                                (`Model.layers` プロパティも同じものを返す)
#
# 契約が合わない族では「何もしない」(エラーにしない) のが方針
# (`docs/BACKLOG.md` の「動的な構造の探索 (duck typing)」)。


def _model_body(model):
    """層を持つ本体 (`model.model` / `model.language_model.model` / `model`
    自身) を返す。どれも `layers` を持たなければ None。"""
    if model is None:
        return None
    for path in (("model",), ("language_model", "model"), ()):
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None and getattr(obj, "layers", None) is not None:
            return obj
    return None


def _model_layers(model) -> list:
    """`model` のデコーダ層を並び順のまま返す。見つからなければ空リスト。

    qwen4_exp では `list(model.model.layers)` と 1 ビットも変わらない
    (同じ順の同じオブジェクト)。他の族や層を持たないスタブでは空になり、
    呼び手のループが 1 周も回らないので enable_* は 0 / None を返す。
    """
    body = _model_body(model)
    if body is None:
        return []
    return list(body.layers)


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
    None. Same for a non-affine mode (mxfp4/nvfp4) -- those have no `biases`
    (fast_qmm.py:382 rejects it the same way), and the kernel assumes affine.
    """

    if "scales" not in lin or "biases" not in lin:
        return None
    if getattr(lin, "mode", "affine") != "affine":
        return None
    return (lin["weight"], lin["scales"], lin["biases"], lin.group_size, lin.bits)


def _pack_inject_bf16(lin):
    """`block_inject_weight` が量子化されていない bf16/fp16 の `nn.Linear` の
    ときに重みだけ取り出す。`_pack_quantized` が None を返したとき (量子化
    経路が使えないとき) だけ decode 幅のカーネルから呼ばれる第二の経路で、
    量子化 inject の経路そのものは変えない。

    診断 (`scratchpad/hc_fire_diag.py`): 97 層の `GatedResidual` のうち 96 層は
    `block_inject_weight` が `nn.Linear` のまま (形は (hc, hc*d)、hc=4 なら
    80KB) で、量子化されているのは 1 層 (mixer) だけ。逆量子化が要らないので
    そのまま Metal カーネルへ渡せる (kernels/hyper_connection.py の
    `_pre_source` 側の "bf16" 分岐)。

    prefill 幅のカーネル (kernels/hyper_connection.py の `_prefill_source`)
    は量子化 inject しか読めないので、この関数の戻り値は decode 幅
    (`fused_gated_residual`) の呼び出しにしか使わない。
    """

    import mlx.core as mx
    import mlx.nn as nn

    if not isinstance(lin, nn.Linear):
        return None
    w = lin.weight
    if w.dtype not in (mx.bfloat16, mx.float16):
        return None
    return ("bf16", w)


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
    inject_bf16_on = os.environ.get("MLXTURBO_HC_INJECT_BF16") == "1"

    def patched(self, hyper):
        down = _pack_quantized(self.input_mix_weight_down)
        up = _pack_quantized(self.input_mix_weight_up)
        combine = self.block_inject_weight is not None
        inject_q = _pack_quantized(self.block_inject_weight) if combine else None
        if down is None or up is None:
            return orig(self, hyper)

        # prefill 幅のカーネル (`_prefill_source`) は量子化 inject しか読めない
        # ので、この分岐には量子化パック (`inject_q`) だけを渡す。combine な
        # のに量子化パックが無い (= bf16 のまま) ときはここを素通りして下の
        # decode 幅の判定に進む -- 元の実装が「combine かつ inject 無し」で
        # 即 orig に落ちていたのと同じ理由で、prefill 側は今まで通り一度も
        # このケースに触れない。
        if prefill_on and (not combine or inject_q is not None):
            m = 1
            for s in hyper.shape[:-1]:
                m *= s
            if hck.eligible_prefill(hyper, self.hc_norm.weight, down, up,
                                     inject_q, self.hc, self.d, m):
                out = hck.fused_gated_residual_prefill(
                    hyper, self.hc_norm.weight, self.hc_norm.eps, self.hc,
                    self.d, down, up, inject_q,
                )
                if not combine:
                    return out
                mixed, inj = out
                return mixed, hyper, inj

        # decode 幅。量子化 inject が使えなければ非量子化 bf16 を試す
        # (診断: 97 層中 96 層が block_inject_weight 未量子化のため、これが
        # 無いとカーネル全体が素の実装に落ちていた)
        inject = inject_q
        if combine and inject is None:
            # 2026-09-03: 97 層で発火させると S=1 forward が 24 → 32 ms と遅くなった
            # (CATCHUP「HC 融合を 97 層で発火させた小さい in-model」)。既定は
            # 素の実装のまま。MLXTURBO_HC_INJECT_BF16=1 のときだけ bf16 inject を試す。
            if not inject_bf16_on:
                return orig(self, hyper)
            inject = _pack_inject_bf16(self.block_inject_weight)
            if inject is None:
                return orig(self, hyper)

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


_ORIG_HC_ELEM = None


def enable_hyper_connection_elem() -> None:
    """`GatedResidual.__call__` の **elementwise だけ** を Metal カーネルに畳む
    (第 4 変種、`kernels/hyper_connection.py` の `fused_gated_residual_elem`)。

    `enable_hyper_connection_kernel` (GEMV 2 本も融合の中に取り込む版) との
    違いは、down/up/inject の行列積を `mx.quantized_matmul` のまま残すこと。
    冷の連鎖では自前の逆量子化内積が DRAM レイテンシを隠せず負ける
    (CATCHUP 2026-09-03 12:00) ので、重みを読む部分は MLX の qmv に任せて
    起動回数だけを減らす。素の 14 ディスパッチが 6 (自前 3 + qmv 3) になる
    (combine あり)。

    量子化されていない `nn.Linear` もそのまま扱える (`_hc_prefill_compile_pack`
    が `(weight,)` を返し、カーネル側の `_elem_qmm` が素の行列積に落とす)。
    97 層中 96 層の `block_inject_weight` が bf16 のまま残っている構成でも
    全層で発火する -- `enable_hyper_connection_kernel` が
    `MLXTURBO_HC_INJECT_BF16` を要求するのは、あちらが inject の逆量子化も
    自前で持っているから。

    `enable_hyper_connection_kernel` / `enable_hyper_connection` /
    `enable_hyper_connection_prefill_compiled` と同じ
    `Q.GatedResidual.__call__` を取り合うので同時に使わない。切り替えるときは
    先に相手の disable を呼ぶこと。

    **素とビット一致するのは decode 幅 (M<=6) だけ。**M>=62 では post 段が
    1e-5〜2.5e-4 の割合で 1 ulp ずれる (`kernels/hyper_connection.py` の
    第 4 変種の節に実測表がある)。`eligible_elem` は行数を見ないので prefill
    幅の呼び出しもこの経路を通り、in-model の軌道は素と分かれる。ビット一致が
    要る用途なら先に行数の上限を足すこと。

    2026-09-03 の in-model A/B (`--knob hc-elem`) では ms/round が短 +2.4% /
    長 +0.7% で **速度目的では棄却**。残してあるのは起動回数を減らす筋の
    実測記録として。

    **prefill 幅まで広げる案は 2026-09-04 に棄却済み**
    (`eligible_elem` の行数上限 8 を外す変種)。冷 micro は HC 読み 0.871 まで
    落ちるが、in-model は 8k prefill -1.6% と引き換えに tok/round が 3 本とも
    落ちた (平均 -4.8%)。prefill 幅の down/up を速くするのは
    `enable_hc_qmm_wide` (ビット一致) の担当。

    発火の確認は `mlxturbo.kernels._fire.snapshot()` の `hc_elem`。
    """

    global _ORIG_HC_ELEM
    import mlx_lm.models.qwen4_exp as Q

    from .kernels import hyper_connection as hck

    if _ORIG_HC_ELEM is not None:
        return
    _ORIG_HC_ELEM = Q.GatedResidual.__call__
    orig = _ORIG_HC_ELEM

    def patched(self, hyper):
        if not hck.eligible_elem(hyper, self.hc_norm.weight, self.hc, self.d):
            return orig(self, hyper)
        combine = self.block_inject_weight is not None
        out = hck.fused_gated_residual_elem(
            hyper,
            self.hc_norm.weight,
            self.hc_norm.eps,
            self.hc,
            self.d,
            _hc_prefill_compile_pack(self.input_mix_weight_down),
            _hc_prefill_compile_pack(self.input_mix_weight_up),
            _hc_prefill_compile_pack(self.block_inject_weight) if combine else None,
        )
        if not combine:
            return out
        mixed, inj = out
        # 素の実装と同じく hyper はそのまま返す (残差が合流する側で使われる)
        return mixed, hyper, inj

    Q.GatedResidual.__call__ = patched


def disable_hyper_connection_elem() -> None:
    global _ORIG_HC_ELEM
    if _ORIG_HC_ELEM is None:
        return
    import mlx_lm.models.qwen4_exp as Q

    Q.GatedResidual.__call__ = _ORIG_HC_ELEM
    _ORIG_HC_ELEM = None


def _hc_prefill_compile_pack(lin):
    """`input_mix_weight_down`/`up`/`block_inject_weight` から重みを取り出す。

    `enable_hyper_connection` の内側にある `_pack` と同じ変換 (量子化なら
    (weight, scales, biases, group_size, bits)、そうでなければ (weight,))。
    `_pack_quantized` (非量子化を None で弾く、kernel 側専用の契約) とは別に
    持つ -- こちらは `_build`/`enable_hyper_connection` と同じく非量子化
    (bf16 の `nn.Linear`) もそのまま扱う。
    """
    if "scales" in lin:
        return (lin.weight, lin.scales, lin.biases, lin.group_size, lin.bits)
    return (lin.weight,)


def _hc_prefill_compile_qmm(x, w):
    """量子化線形。w は `_hc_prefill_compile_pack` の戻り値。"""
    import mlx.core as mx

    if len(w) == 1:
        return x @ w[0].T
    wt, sc, bi, gs, bits = w
    return mx.quantized_matmul(
        x, wt, scales=sc, biases=bi, transpose=True, group_size=gs, bits=bits
    )


def _hc_prefill_compile_pre(hc: int, d: int, eps: float):
    """`RMSNorm(hyper)` + `(1+w)` 乗算 (down GEMM の直前まで) を 1 グラフに
    畳む。``shapeless=True`` なので prefill チャンク幅 (2048 / 1779 / 端数)
    が変わっても再コンパイルされない -- 変わるのは行数だけで、次元数
    (`hyper` の rank) と dtype は固定なので shapeless の制約に収まる。

    **`x.reshape(*x.shape[:-1], -1, d)` (Python 側で読んだ `.shape` を展開
    して作る target shape) は使わない。**`shapeless=True` は「トレースした
    グラフをそのまま別の形に使い回す」もので、Python の `.shape` 展開は
    トレース時の具体的な整数をグラフへ焼き込む (最初の呼び出しが T=4 なら
    reshape の目標形状に literal 4 が刻まれる)。次に T=3 (端数チャンク) で
    呼ぶと「size 384 を shape (1,4,...) へ reshape できない」で落ちる (実装
    中に実測で踏んだ -- 最小再現は 1 行の elementwise + reshape + reshape の
    合成関数を T=4 で 1 回呼んでから T=3 で呼ぶだけで再現する)。安全な形は
    「先頭の可変長をまとめて 1 つの `-1` に潰し、固定なのは `hc`/`d` だけ」
    にすること -- `-1` はグラフ側で
    毎回の実サイズから解決されるので、行数が変わっても壊れない
    (`mlx_lm.models.qwen3_next._precise_swiglu` や `gated_delta.compute_g`
    も reshape を伴わない/伴っても target が全部固定という形でこの罠を
    踏んでいない)。
    """
    import mlx.core as mx

    key = (hc, d, eps)
    fn = _HC_PREFILL_COMPILE_PRE.get(key)
    if fn is not None:
        return fn

    def pre(hyper, norm_w):
        # 先頭 (B, T...) をまとめて 1 つの可変長行に潰す。hc/d は固定なので
        # ここで焼き込んでも別の行数で使い回して安全 (上の docstring 参照)。
        x = hyper.reshape(-1, hc, d)
        x = mx.fast.rms_norm(x, None, eps)
        x = x.reshape(-1, hc * d)
        return x * (1.0 + norm_w)

    fn = mx.compile(pre, shapeless=True)
    _HC_PREFILL_COMPILE_PRE[key] = fn
    return fn


def _hc_prefill_compile_post(hc: int, d: int, use_combine: bool):
    """up GEMM の直後 (`sigmoid` -> reshape -> `w * normed` -> `mean`) と、
    combine のときは inject の `sigmoid` も同じグラフに畳む。inject は
    `normed` にしか依存しない (up の結果とは無関係) ので、同じ
    `mx.compile` 呼び出しの中に同居させても up 側の計算とは独立に評価される。

    `_hc_prefill_compile_pre` と同じ理由で、reshape の target shape は
    `-1` (可変長の行) と `hc`/`d` (固定) だけで組む -- `w.shape[:-1]` を
    展開しない。入出力とも先頭の行軸は 1 本に潰れたまま
    (呼び出し側の `patched` が `hyper.shape` を使って素の Python で
    元の形に戻す -- そちらは毎回フレッシュに実行されるので焼き込みの心配が無い)。
    """
    import mlx.core as mx

    key = (hc, d, use_combine)
    fn = _HC_PREFILL_COMPILE_POST.get(key)
    if fn is not None:
        return fn

    if use_combine:

        def post(normed, up_raw, inject_raw):
            w = mx.sigmoid(up_raw).reshape(-1, hc, d)
            mixed = (w * normed.reshape(-1, hc, d)).mean(axis=-2)
            inject = 2.0 * mx.sigmoid(inject_raw / hc)
            return mixed, inject

    else:

        def post(normed, up_raw):
            w = mx.sigmoid(up_raw).reshape(-1, hc, d)
            return (w * normed.reshape(-1, hc, d)).mean(axis=-2)

    fn = mx.compile(post, shapeless=True)
    _HC_PREFILL_COMPILE_POST[key] = fn
    return fn


def enable_hyper_connection_prefill_compiled(model=None) -> None:
    """`GatedResidual.__call__` の GEMM 2 本 (`input_mix_weight_down`/`up`、
    combine のときは inject の `block_inject_weight` も合わせて 3 本) 以外の
    elementwise 部分だけを、prefill 幅 (行数 >= `MLXTURBO_HC_COMPILE_MIN_ROWS`、
    既定 64) のときに限って `mx.compile(shapeless=True)` に畳む。

    2 つのグラフに分ける (`_hc_prefill_compile_pre` / `_hc_prefill_compile_post`)。
    GEMM 自体は 4bit 量子化のままそのつど `mx.quantized_matmul` を直接呼ぶ
    (`_hc_prefill_compile_qmm`) -- `enable_hyper_connection` (全体を 1 グラフに
    する版) と違い、量子化 GEMM をコンパイル境界の外に出すことで down/up の
    間の `silu(.../hc)` 1 op だけが素のまま挟まる (2 GEMM の間にどうしても
    residency する 1 op で、まとめても短い chain 1 本にしかならない)。

    prefill 1 チャンク (2048 tok) で HC が 97 回 x 380ms/チャンク (効率 60%)
    かかっている内訳のうち、GEMM を除いた elementwise が 90-120ms/チャンクと
    見積もられている (docs/research/KERNEL-BRIEF-DECODE-BW.md)。ここを
    2 ディスパッチ (pre/post) に畳めれば -2〜3% が仮説 (在庫は in-model A/B
    で検証、`tools/decode_ab.py --knob hc-prefill-compile`)。

    行数 < 64 (decode/verify 幅) は enable した時点の `Q.GatedResidual.__call__`
    (`_ORIG_HC_PREFILL_COMPILE` に控えてある) にそのまま落ちる -- decode 幅は
    素のままにする。

    **出力はビット同一ではない。**GEMM の呼び出し順・量子化経路自体は
    素の実装から変えていないが、`_hc_prefill_compile_pre`/`_post` の内部で
    先頭 (B, T...) を 1 本の可変長行に潰してから `hc_norm` のグループ計算を
    しており (`shapeless=True` の reshape 焼き込み回避、両関数の docstring
    参照)、これが vendor の `RMSNorm.__call__` が使う reshape 経路
    (先頭を展開したまま保持) と違う。CPU 上の合成モデルで実測すると
    (`tools/vendor_fingerprint.py`) md5 は on/off で変わり、logits の
    max|diff|=1.04e-7 (float32 の丸みの水準、`group_prefill_forward` が
    記録している 8e-7 と同じ桁)。正しさの問題ではないが、
    `tools/decode_ab.py --knob hc-prefill-compile` の対照は
    `control_identical=False` で扱うこと。

    既存の融合 Metal カーネル (`enable_hyper_connection_kernel`) とは同時に
    使わない。どちらも `Q.GatedResidual.__call__` を取り合うため、両方
    有効化すると disable の順序次第でどちらの状態に戻るか不定になる
    (`enable_hyper_connection`/`enable_hyper_connection_kernel` 同士も同じ
    問題を抱えているが、`runner.py` の `hc_mode` if/elif/else が互いを
    排他にしているので今まで表面化していない)。`_ORIG_HC_KERNEL is not None`
    (= kernel 側が有効) ならここで例外を出す -- 呼び出し側が先に
    `disable_hyper_connection_kernel()` を呼ぶこと。

    既定 off。環境変数 `MLXTURBO_HC_PREFILL_COMPILE=1` が立っているときだけ
    有効化する (`enable_moe_verify_gather`/`enable_gdn_prework_kernel` と同じ
    ゲート方式 -- 呼ぶだけでは何も起きない)。`model` は他の `enable_*(model)`
    とシグネチャを揃えてあるだけで今のところ未使用 -- パッキング
    (`_hc_prefill_compile_pack`) は呼び出しごとに `self` から行うので、
    事前に model の層を歩いて写しを作る必要が無い (`enable_gdn_prework_kernel`
    の `A_log`/`dt_bias` の fp32 写しとは事情が違う)。
    """
    global _ORIG_HC_PREFILL_COMPILE
    import os

    import mlx_lm.models.qwen4_exp as Q

    if os.environ.get("MLXTURBO_HC_PREFILL_COMPILE") != "1":
        return
    if _ORIG_HC_KERNEL is not None:
        raise RuntimeError(
            "enable_hyper_connection_prefill_compiled: "
            "enable_hyper_connection_kernel と同時には使えない (同じ "
            "GatedResidual.__call__ を取り合う)。先に "
            "disable_hyper_connection_kernel() を呼ぶこと。"
        )
    if _ORIG_HC_PREFILL_COMPILE is not None:
        return
    _ORIG_HC_PREFILL_COMPILE = Q.GatedResidual.__call__
    orig = _ORIG_HC_PREFILL_COMPILE

    # 起動時に 1 回だけ読む (呼び出しごとの getenv を避ける)。
    min_rows = int(os.environ.get("MLXTURBO_HC_COMPILE_MIN_ROWS", "64"))

    import mlx.nn as nn

    def patched(self, hyper):
        m = 1
        for s in hyper.shape[:-1]:
            m *= s
        if m < min_rows:
            return orig(self, hyper)

        use_combine = self.block_inject_weight is not None
        lead = hyper.shape[:-1]  # 素の Python で毎回フレッシュに読む (焼き込み無し)
        pre = _hc_prefill_compile_pre(self.hc, self.d, self.hc_norm.eps)
        normed = pre(hyper, self.hc_norm.weight)  # (rows, hc*d) に潰れている

        down_raw = _hc_prefill_compile_qmm(
            normed, _hc_prefill_compile_pack(self.input_mix_weight_down)
        )
        w_mid = nn.silu(down_raw / self.hc)
        up_raw = _hc_prefill_compile_qmm(
            w_mid, _hc_prefill_compile_pack(self.input_mix_weight_up)
        )

        post = _hc_prefill_compile_post(self.hc, self.d, use_combine)
        if not use_combine:
            mixed = post(normed, up_raw)
            return mixed.reshape(*lead, self.d)
        inject_raw = _hc_prefill_compile_qmm(
            normed, _hc_prefill_compile_pack(self.block_inject_weight)
        )
        mixed, inject = post(normed, up_raw, inject_raw)
        mixed = mixed.reshape(*lead, self.d)
        inject = inject.reshape(*lead, self.hc)
        return mixed, hyper, inject

    Q.GatedResidual.__call__ = patched


def disable_hyper_connection_prefill_compiled() -> None:
    global _ORIG_HC_PREFILL_COMPILE
    if _ORIG_HC_PREFILL_COMPILE is None:
        return
    import mlx_lm.models.qwen4_exp as Q

    Q.GatedResidual.__call__ = _ORIG_HC_PREFILL_COMPILE
    _ORIG_HC_PREFILL_COMPILE = None


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


def enable_gdn_prework_kernel(model=None) -> None:
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

    `model` を渡すと、その全 GatedDeltaNet 層 (``model.model.layers[*].linear_attn``)
    の `A_log`/`dt_bias` を fp32 に変換した写し (``_A_log_f32``/``_dt_bias_f32``)
    を層に持たせる。実モデル (mlx_lm の 4bit 変換など) では `A_log`/`dt_bias`
    が bf16 で読み込まれていることがあり (2026-09-02、実機ログで確認)、
    `eligible()` は dtype を fp32 限定で見るのでそのままだと毎回弾かれて
    カーネルが一度も発火しない。素の経路 (`mlx_lm` の `compute_g`) は
    `A_log.astype(float32)` して計算するので、fp32 に揃えるのは意味的に
    同じ。写しは enable 時に 1 回だけ作る (毎ステップ astype を dispatch
    しないため)。`model` を渡さない場合はこの写しを作らず、
    `A_log`/`dt_bias` が元から fp32 でない層は従来どおり `eligible()` の
    dtype 判定で弾かれる。

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
    # 注 (2026-09-03、カーネル書き直し): **fp32 の写しはもう作らない。**
    # 素の `compute_g` は `a + dt_bias` を bf16 で足して bf16 で softplus する
    # ので、fp32 の写しを渡すとカーネルだけが別の丸め位置になる (前は
    # eligible() が fp32 限定だったのでこの写しが要った)。現行のカーネルは
    # bf16 のまま素と同じ順で丸めるので、素の重みをそのまま渡すのが正しい。
    # 古い写しが残っていると eligible() が dtype 不一致で弾くので消す。
    if model is None:
        return
    for layer in _model_layers(model):
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None:
            continue
        for attr in ("_A_log_f32", "_dt_bias_f32"):
            if hasattr(gdn, attr):
                delattr(gdn, attr)


def disable_gdn_prework_kernel() -> None:
    import mlx_lm.models.qwen4_exp as Q

    Q.GatedDeltaNet._gdn_prework = False


# `enable_gdn_decode_fused` が差し替えた `RMSNormGated.__call__` の元
_ORIG_RNG_DECODE = None

# 出力 norm を融合カーネルに回す行数の上限。行数は B*S*n_v なので、
# n_v=48・decode/verify 幅 (gdn_prework.MAX_M=16) で 768。prefill 幅
# (1 チャンク 2048 トークン = 98304 行) は必ず素の経路に落ちる。
_GDN_NORM_MAX_ROWS = 768


def enable_gdn_decode_fused(model=None) -> None:
    """decode/verify 幅の GDN 層の「行列積以外」を 3 本にする (2026-09-03)。

    前処理 (`gdn_prework`、素 9 本) + 再帰 (元から 1 本) + 出力 norm
    (`rms_norm_gated`、素 6 本) で、GDN 1 層の非行列積が **16 本 -> 3 本**。
    36 層で 1 step あたり約 470 dispatch (S=1 の 4499 本の 1 割) が消える。

    `enable_gdn_prework_kernel` (前処理だけ、`MLXTURBO_GDN_PREWORK=1`) との
    違いは出力 norm も畳むことと、**その norm を decode/verify 幅に限る**こと。
    `enable_rms_norm_gated` は幅を見ずに全部差し替えるので prefill にも効き、
    2026-09-02 の A/B で受理率が動いた (`enable_rms_norm_gated` の docstring)。
    ここでは行数 <= :data:`_GDN_NORM_MAX_ROWS` のときだけカーネルに回す。

    **既定 on** (2026-09-03 21:05: 短 ms/round -2.4% / 17k -2.0%、micro でビット一致、代金ゼロ方針)。`MLXTURBO_GDN_DECODE_FUSED=0` で off:

    - `1` / `all`: 前処理 + 出力 norm (3 本)
    - `pre`: 前処理だけ (4 本。norm の寄与を切り分けるとき)
    - `norm`: 出力 norm だけ

    採否は in-model A/B (`tools/decode_ab.py --knob gdn-decode-fused`) で決める。
    """

    global _ORIG_RNG_DECODE
    import os

    import mlx_lm.models.qwen4_exp as Q

    mode = (os.environ.get("MLXTURBO_GDN_DECODE_FUSED") or "1").lower()  # 既定 on (2026-09-03 21:05)
    if mode in ("", "0", "off"):
        return
    if mode not in ("1", "all", "pre", "norm"):
        raise ValueError(
            f"MLXTURBO_GDN_DECODE_FUSED={mode!r} は 1/all/pre/norm のいずれか")

    if mode in ("1", "all", "pre"):
        Q.GatedDeltaNet._gdn_prework = True
        if model is not None:
            for layer in _model_layers(model):
                gdn = getattr(layer, "linear_attn", None)
                if gdn is None:
                    continue
                for attr in ("_A_log_f32", "_dt_bias_f32"):
                    if hasattr(gdn, attr):
                        delattr(gdn, attr)

    if mode in ("1", "all", "norm") and _ORIG_RNG_DECODE is None:
        from math import prod

        from .kernels import rms_norm_gated as rng

        _ORIG_RNG_DECODE = Q.RMSNormGated.__call__
        orig = _ORIG_RNG_DECODE

        def patched(self, x, gate=None):
            if gate is None or prod(x.shape[:-1]) > _GDN_NORM_MAX_ROWS:
                return orig(self, x, gate)
            w = self.weight
            if not rng.eligible(x, w, gate):
                return orig(self, x, gate)
            # 行数が 48〜768 しかないので 1 行 = 1 threadgroup (既定の 8 行
            # まとめだと threadgroup が 6 個しか立たず 40 コアに散らない)
            return rng.rms_norm_gated(x, w, gate, self.eps, self.activation,
                                       rows_per_tg=1)

        Q.RMSNormGated.__call__ = patched


def disable_gdn_decode_fused() -> None:
    global _ORIG_RNG_DECODE
    import mlx_lm.models.qwen4_exp as Q

    Q.GatedDeltaNet._gdn_prework = False
    if _ORIG_RNG_DECODE is not None:
        Q.RMSNormGated.__call__ = _ORIG_RNG_DECODE
        _ORIG_RNG_DECODE = None


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

    2026-09-02 に既定 on にした (in-model A/B: 17k prefill_s -1.3〜-4.5%、
    KLD +0.00014、受け入れ幅 +0.0005 の中)。環境変数 `MLXTURBO_GDN_METAL=0`
    で無効化できる。
    """
    import os

    import mlx_lm.models.qwen4_exp as Q

    if os.environ.get("MLXTURBO_GDN_METAL") == "0":
        return
    Q.GatedDeltaNet._gdn_metal = True


def disable_gdn_metal_kernel() -> None:
    import mlx_lm.models.qwen4_exp as Q

    Q.GatedDeltaNet._gdn_metal = False


# --------------------------------------------------------------------------
# GDN の自前部品を「構造の契約」で他の族に当てる (2026-09-04、27B レーン 第 1 段)
# --------------------------------------------------------------------------
#
# 上の 4 つの enable_gdn_* は `Q.GatedDeltaNet.<flag> = True` を立てるだけで、
# 実際にカーネルを呼ぶのは `_vendor/qwen4_exp.py` の `GatedDeltaNet.__call__`
# にあるシームのほう。qwen3_5 (Qwen3.8-27B) の `GatedDeltaNet` にはシームが
# 無いので、そのままでは 1 つも届かない。
#
# ここでは `docs/BACKLOG.md:713` の方針 (構造の探索 = duck typing、契約検査
# つきの適用、素の forward はそのままでフックだけ差す) で当てる:
#
#   1. 形の契約 (`_gdn_spec`) が合う GDN インスタンスだけを選ぶ。
#   2. decode/verify 幅の前処理は、そのインスタンスの `__class__` を動的
#      サブクラスに差し替えて `__call__` の**速い側だけ**を自前にする
#      (契約・幅が外れたら素の `__call__` をそのまま呼ぶ = 写しを作らない)。
#      `__class__` の差し替えは `kernels/dispatch.py:227` と同じ手で、
#      モジュールの identity・パラメータ名・凍結状態・読み込み済みの配列を
#      そのまま保つ。
#   3. 出力 norm も同じく norm インスタンスの `__class__` 差し替え。活性化
#      (sigmoid / silu) は族ごとに違い、qwen3_5 側には `activation` 属性が
#      無い (`_precise_swiglu` = silu 固定) ので、**起動時に合成入力で素と
#      突き合わせて決める** (ビット一致した活性化だけ採る)。
#   4. prefill 幅の再帰 (GDN Metal) は、GDN のクラスを定義しているモジュール
#      の `gated_delta_update` を包む (形の契約に合えば Metal、外れれば素を
#      そのまま呼ぶ)。クラスパッチではなく関数の差し替えで、契約に合う GDN が
#      1 層でも見つかったときだけ入れる。
#
# qwen4_exp はシームを持っているので**この経路の対象外** (二重に当てない)。
# 環境変数は上と共用: `MLXTURBO_GDN_METAL` (既定 on) が 4、
# `MLXTURBO_GDN_DECODE_FUSED` (既定 1) が 2 と 3 を切る。

# ベースクラスごとに 1 つだけ作る動的サブクラス (48 層で 48 個作らない)
_GDN_CALL_CLASSES: dict = {}
_GDN_NORM_CLASSES: dict = {}
# 包んだ `gated_delta_update` を持つモジュール (disable で戻すため)
_GDN_UPDATE_PATCHED: list = []


def _gdn_spec(gdn):
    """GDN 層の形の契約。合えば必要な数を集めた `SimpleNamespace`、
    合わなければ None。

    戻り値を dict / tuple にしないこと: `nn.Module.__setattr__` は
    array/dict/list/tuple をモジュール辞書 (パラメータ・子モジュール) に
    入れてしまい、`children()` / `named_modules()` の走査に混ざる。

    族ごとに属性名が違う (qwen4_exp の `n_v`/`n_k`/`dk`/`dv` に対して
    qwen3_5 は `num_v_heads`/`num_k_heads`/`head_k_dim`/`head_v_dim`) ので、
    意味ごとに候補名を並べて拾う。名前が 1 つでも欠けるか、形の関係
    (`key_dim == n_k*dk` など) が崩れていれば None を返して当てない。
    """

    import mlx.core as mx

    for name in ("conv1d", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                 "norm", "out_proj"):
        if getattr(gdn, name, None) is None:
            return None
    A_log = getattr(gdn, "A_log", None)
    dt_bias = getattr(gdn, "dt_bias", None)
    if not isinstance(A_log, mx.array) or not isinstance(dt_bias, mx.array):
        return None

    def pick(*names):
        for n in names:
            v = getattr(gdn, n, None)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        return None

    n_v = pick("n_v", "num_v_heads")
    n_k = pick("n_k", "num_k_heads")
    dk = pick("dk", "head_k_dim")
    dv = pick("dv", "head_v_dim")
    key_dim = pick("key_dim")
    value_dim = pick("value_dim")
    conv_dim = pick("conv_dim")
    K = pick("conv_kernel_size")
    if None in (n_v, n_k, dk, dv, key_dim, value_dim, conv_dim, K):
        return None
    if key_dim != n_k * dk or value_dim != n_v * dv:
        return None
    if conv_dim != 2 * key_dim + value_dim:
        return None
    w = getattr(gdn.conv1d, "weight", None)
    if not isinstance(w, mx.array) or w.shape != (conv_dim, K, 1):
        return None
    if A_log.shape != (n_v,) or dt_bias.shape != (n_v,):
        return None
    norm_w = getattr(gdn.norm, "weight", None)
    if not isinstance(norm_w, mx.array) or norm_w.shape != (dv,):
        return None
    if not isinstance(getattr(gdn.norm, "eps", None), float):
        return None
    from types import SimpleNamespace

    return SimpleNamespace(
        n_k=n_k, n_v=n_v, dk=dk, dv=dv,
        key_dim=key_dim, value_dim=value_dim, conv_dim=conv_dim, K=K,
    )


def _gdn_norm_activation(norm, base) -> str | None:
    """出力 norm の活性化 (`sigmoid` / `silu`) を決める。

    `activation` 属性を持つ族 (qwen4_exp) はそれをそのまま使う。持たない族
    (qwen3_5 は `Qwen3NextRMSNormGated` で `_precise_swiglu` = silu 固定) は、
    **合成入力 1 回で素の実装と突き合わせて**決める (`docs/BACKLOG.md:716`
    「起動時に合成入力で素の実装と突き合わせる」)。ビット一致する活性化が
    無ければ None を返し、その層の norm は素のままにする。
    """

    import mlx.core as mx

    from .kernels import rms_norm_gated as rng

    w = norm.weight
    act = getattr(norm, "activation", None)
    rows, dv = 8, w.shape[0]
    # 大域の PRNG 状態を進めない (生成の乱数列を動かさないため、鍵を明示する)
    key = mx.random.key(0)
    x = mx.random.normal((rows, dv), key=key).astype(w.dtype)
    gate = mx.random.normal((rows, dv), key=mx.random.key(1)).astype(w.dtype)
    if not rng.eligible(x, w, gate):
        return None
    try:
        ref = base.__call__(norm, x, gate)
        mx.eval(ref)
    except Exception:  # noqa: BLE001 -- 素が呼べない形なら当てない
        return None
    for cand in ((act,) if act else ("silu", "sigmoid")):
        if cand not in ("silu", "sigmoid"):
            continue
        got = rng.rms_norm_gated(x, w, gate, norm.eps, cand, rows_per_tg=1)
        mx.eval(got)
        if got.dtype == ref.dtype and mx.array_equal(got, ref):
            return cand
    return None


def _gdn_norm_subclass(base, activation: str):
    """`base` (族の RMSNormGated) の動的サブクラス。decode/verify 幅の
    行数 (<= :data:`_GDN_NORM_MAX_ROWS`) だけ融合カーネルへ回す。"""

    key = (base, activation)
    cls = _GDN_NORM_CLASSES.get(key)
    if cls is not None:
        return cls

    from math import prod

    from .kernels import rms_norm_gated as rng

    def patched(self, x, gate=None):
        if gate is None or prod(x.shape[:-1]) > _GDN_NORM_MAX_ROWS:
            return base.__call__(self, x, gate)
        w = self.weight
        if not rng.eligible(x, w, gate):
            return base.__call__(self, x, gate)
        # 行数が 48〜768 しかないので 1 行 = 1 threadgroup
        # (`enable_gdn_decode_fused` と同じ理由)
        return rng.rms_norm_gated(x, w, gate, self.eps, activation, rows_per_tg=1)

    cls = type(
        f"{base.__name__}_mlxturbo_gdn",
        (base,),
        {"__call__": patched, "_mlxturbo_gdn_base": base},
    )
    cls.__module__ = __name__
    _GDN_NORM_CLASSES[key] = cls
    return cls


# --------------------------------------------------------------------------
# 巻き戻し用の状態を取り出す口 (2026-09-04、27B レーン 第 1 段)
# --------------------------------------------------------------------------
#
# `mlxturbo/spec.py` の `_linear_capture` は GDN の本体を手で書き下した写しで、
# `la(...)` を一度も呼んでいなかった。そのため上の動的サブクラスが S>1 の
# 検証フォワードに **1 度も当たっていなかった** (`_fire` で確認。数字は
# `scratchpad/agent-27b-decode-b1.md`)。
#
# 写しが必要だった理由は 1 つだけ ---「巻き戻しのために位置ごとの再帰状態
# (`states_all`) と conv の入力列 (`conv_input`) を層ごとに控える」。そこで
# サブクラスの側に取り出し口を足して、呼び手が `la(...)` を呼べるようにする。
#
# `gdn_capture(sink)` で武装している間、契約の合う GDN の `__call__` は
# `(cache, states_all, conv_input, K, old_lengths, old_left_padding)` を
# `sink` に積む (`spec.py` の `_rollback` がそのまま受ける形)。契約・幅が
# 外れたときは **cache を 1 つも書き換える前に** `GdnCaptureUnsupported` を
# 投げる -- 黙って捕捉し損ねると巻き戻しが静かに壊れるため。呼び手は
# `gdn_capture_ready` で先に全層を検査してから武装すること。

_GDN_CAPTURE: list | None = None


class GdnCaptureUnsupported(RuntimeError):
    """`gdn_capture` の下で契約・幅が外れた (cache はまだ書き換えていない)。"""


@contextmanager
def gdn_capture(sink: list):
    """この文の中で呼ばれた GDN に、巻き戻し用の材料を `sink` へ積ませる。"""

    global _GDN_CAPTURE
    prev = _GDN_CAPTURE
    _GDN_CAPTURE = sink
    try:
        yield
    finally:
        _GDN_CAPTURE = prev


def gdn_capture_ready(gdn, inputs, mask=None, cache=None) -> bool:
    """`gdn_capture` の下で `gdn(...)` が捕捉できるかの前検査。

    層ループを回し始めてから捕捉に失敗すると、手前の層の cache は既に進んで
    いて戻せない。そこで **1 層も走らせる前に**全層を検査できるようにする。

    射影の結果 (`mixed_qkv` / `a` / `b`) はまだ無いので、同じ形・同じ dtype の
    プレースホルダを渡して `gdn_prework.eligible` をそのまま使う (dtype と形しか
    見ないので評価されない)。射影の出力 dtype は入力 dtype と同じ。
    """

    import mlx.core as mx

    from .kernels import gdn_prework as gp

    if getattr(type(gdn), "_mlxturbo_gdn_base", None) is None:
        return False
    spec = getattr(gdn, "_mlxturbo_gdn_spec", None)
    if spec is None or inputs.ndim != 3:
        return False
    B, S, _ = inputs.shape
    if (
        S > gp.MAX_S
        or B * S > gp.MAX_M
        or mask is not None
        or cache is None
        or gdn.training
        or getattr(cache, "lengths", None) is not None
        or getattr(gdn, "sharding_group", None) is not None
    ):
        return False
    conv_state = (
        cache[0]
        if cache[0] is not None
        else mx.zeros((B, spec.K - 1, spec.conv_dim), dtype=inputs.dtype)
    )
    probe_qkv = mx.zeros((B, S, spec.conv_dim), dtype=inputs.dtype)
    probe_ab = mx.zeros((B, S, spec.n_v), dtype=inputs.dtype)
    return gp.eligible(
        probe_qkv, conv_state, gdn.conv1d.weight, probe_ab, probe_ab,
        gdn.A_log, gdn.dt_bias,
        spec.n_k, spec.n_v, spec.dk, spec.key_dim, spec.value_dim,
    )


def _gdn_call_subclass(base):
    """`base` (族の GatedDeltaNet) の動的サブクラス。

    decode/verify 幅で契約が合うときだけ `gdn_prework` + 逐次カーネル +
    出力 norm を自前で通す。**それ以外は素の `__call__` をそのまま呼ぶ**
    (素の forward の写しは持たない)。

    `gdn_capture` で武装されている間は、逐次カーネルを位置ごとの状態も返す
    版 (`gated_delta_update_with_states_gb`) に替えて `sink` に材料を積む。
    """

    cls = _GDN_CALL_CLASSES.get(base)
    if cls is not None:
        return cls

    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_kernel

    from .kernels import gdn_prework as gp
    from .kernels.gated_delta_states import gated_delta_update_with_states_gb

    def _decline(self, inputs, mask, cache, sink, why):
        if sink is not None:
            raise GdnCaptureUnsupported(
                f"gdn_capture 中に自前経路の契約が外れた ({why})。"
                " 呼び手は gdn_capture_ready で先に検査すること"
            )
        return base.__call__(self, inputs, mask, cache)

    def patched(self, inputs, mask=None, cache=None):
        sink = _GDN_CAPTURE
        spec = getattr(self, "_mlxturbo_gdn_spec", None)
        if spec is None or inputs.ndim != 3:
            return _decline(self, inputs, mask, cache, sink, "契約なし/ndim")
        B, S, _ = inputs.shape
        # 幅の判定を射影より先に置く (prefill では 1 op も余分に組まない)
        if (
            S > gp.MAX_S
            or B * S > gp.MAX_M
            or mask is not None
            or cache is None
            or self.training
            or getattr(cache, "lengths", None) is not None
            or getattr(self, "sharding_group", None) is not None
        ):
            return _decline(self, inputs, mask, cache, sink, "幅/mask/cache")

        n_k, n_v = spec.n_k, spec.n_v
        dk, dv = spec.dk, spec.dv
        key_dim, value_dim = spec.key_dim, spec.value_dim
        conv_w = self.conv1d.weight
        A_log, dt_bias = self.A_log, self.dt_bias

        mixed_qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, n_v, dv)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)
        conv_state = (
            cache[0]
            if cache[0] is not None
            else mx.zeros((B, spec.K - 1, spec.conv_dim), dtype=inputs.dtype)
        )
        if not gp.eligible(mixed_qkv, conv_state, conv_w, a, b, A_log, dt_bias,
                           n_k, n_v, dk, key_dim, value_dim):
            # MLX は遅延評価なので、ここで捨てた射影は評価されない
            # (`enable_moe_route` の patched と同じ理屈)
            return _decline(self, inputs, mask, cache, sink, "eligible=False")

        q, k, v, g, beta, new_conv_state = gp.fused_gdn_prework(
            mixed_qkv, conv_state, conv_w, a, b, A_log, dt_bias,
            n_k, n_v, dk, dv, key_dim, value_dim,
        )
        if sink is None:
            cache[0] = new_conv_state
            state = cache[1]
            if state is None:
                state = mx.zeros((B, n_v, dv, dk), dtype=mx.float32)
            out, state = gated_delta_kernel(q, k, v, g, beta, state, None)
            cache[1] = state
            cache.advance(S)
        else:
            # 巻き戻し用: 位置ごとの状態と conv の入力列を控える。`mask` は
            # 上で None と確認済みなので、素の `mx.where(mask, qkv, 0)` は
            # 恒等 -- conv_input は素の経路と同じ列そのもの。
            old_lengths = getattr(cache, "lengths", None)
            old_left_padding = getattr(cache, "left_padding", None)
            conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
            out, states_all = gated_delta_update_with_states_gb(
                q, k, v, g, beta, cache[1], None
            )
            cache[0] = new_conv_state
            cache[1] = states_all[:, -1]
            cache.advance(S)
            sink.append(
                (cache, states_all, conv_input, spec.K,
                 old_lengths, old_left_padding)
            )
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))

    cls = type(
        f"{base.__name__}_mlxturbo_gdn",
        (base,),
        {"__call__": patched, "_mlxturbo_gdn_base": base},
    )
    cls.__module__ = __name__
    _GDN_CALL_CLASSES[base] = cls
    return cls


def _gdn_patch_update(mod) -> bool:
    """`mod.gated_delta_update` を GDN Metal 付きの版で包む。

    prefill 幅の再帰だけを差し替える (`gdn_blocked_metal.eligible` が
    Dk==128 / Dv%32==0 / mask 無し / T>=64 を見る)。外れたら元の関数を
    そのまま呼ぶので、decode/verify 幅と mask 付きは素のまま。
    """

    orig = getattr(mod, "gated_delta_update", None)
    if orig is None or getattr(orig, "_mlxturbo_gdn_metal", False):
        return False

    from .kernels import gdn_blocked_metal as gbm

    def patched(q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
                use_kernel=True):
        if use_kernel and gbm.eligible(q, k, v, state, mask):
            return gbm.gated_delta_update_blocked_metal(
                q, k, v, a, b, A_log, dt_bias, state)
        return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)

    patched._mlxturbo_gdn_metal = True
    patched._mlxturbo_gdn_orig = orig
    mod.gated_delta_update = patched
    _GDN_UPDATE_PATCHED.append(mod)
    return True


def enable_gdn_port(model=None) -> dict:
    """qwen4_exp 以外の族の GDN 層に、GDN の自前部品を当てる (2026-09-04)。

    戻り値は当たった数の内訳 ``{"metal": 0/1, "prework": 層数,
    "norm": 層数, "layers": 契約の合った層数}``。契約が合う層が 1 つも
    無ければ全部 0 で、モデルには何も起きない。

    qwen4_exp は `_vendor/qwen4_exp.py` の `GatedDeltaNet.__call__` に
    シームを持っていて上の 4 つの enable_gdn_* が当たるので、ここでは
    **対象外にする** (二重に当てない)。

    切り方は Flash-Next と共用:

    - ``MLXTURBO_GDN_METAL=0`` -> prefill 幅の再帰 (Metal) を入れない
    - ``MLXTURBO_GDN_DECODE_FUSED=0`` -> decode/verify 幅の前処理と
      出力 norm を入れない (``pre`` / ``norm`` で片方だけ)
    """

    import os
    import sys

    counts = {"metal": 0, "prework": 0, "norm": 0, "layers": 0}
    if model is None:
        return counts

    mode = (os.environ.get("MLXTURBO_GDN_DECODE_FUSED") or "1").lower()
    if mode not in ("1", "all", "pre", "norm", "0", "off", ""):
        raise ValueError(
            f"MLXTURBO_GDN_DECODE_FUSED={mode!r} は 1/all/pre/norm/0 のいずれか")
    want_pre = mode in ("1", "all", "pre")
    want_norm = mode in ("1", "all", "norm")
    # 移植した族では prefill の Metal 再帰は **明示 =1 のときだけ** (既定 off)。
    # 27B で 4k -1.4% / 17k +0.1% と取り分が無く、KLD 0.00027 (参照 = 素 4bit) の
    # 代金だけ残る (CATCHUP 2026-09-04 09:15 / 09:36)。qwen4_exp は従来どおり
    # enable_gdn_metal_kernel 側で既定 on。
    want_metal = os.environ.get("MLXTURBO_GDN_METAL") == "1"

    norm_act: dict = {}
    modules: set = set()
    for layer in _model_layers(model):
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None:
            continue
        base = type(gdn)
        if getattr(base, "_mlxturbo_gdn_base", None) is not None:
            base = base._mlxturbo_gdn_base   # 既に当たっている (idempotent)
        if base.__module__ == "mlx_lm.models.qwen4_exp":
            continue                          # シーム持ち。ここでは触らない
        spec = _gdn_spec(gdn)
        if spec is None:
            continue
        counts["layers"] += 1
        modules.add(base.__module__)

        if want_pre:
            gdn._mlxturbo_gdn_spec = spec
            if type(gdn) is base:
                gdn.__class__ = _gdn_call_subclass(base)
            counts["prework"] += 1

        if want_norm:
            norm = gdn.norm
            nbase = type(norm)
            if getattr(nbase, "_mlxturbo_gdn_base", None) is not None:
                nbase = nbase._mlxturbo_gdn_base
            act = norm_act.get(nbase)
            if act is None:
                act = _gdn_norm_activation(norm, nbase)
                norm_act[nbase] = act or False
            if act:
                if type(norm) is nbase:
                    norm.__class__ = _gdn_norm_subclass(nbase, act)
                counts["norm"] += 1

    if want_metal and counts["layers"]:
        for name in modules:
            mod = sys.modules.get(name)
            if mod is not None and _gdn_patch_update(mod):
                counts["metal"] += 1
    return counts


def disable_gdn_port(model=None) -> dict:
    """`enable_gdn_port` を打ち消す (A/B で交互に測るために要る)。"""

    counts = {"metal": 0, "prework": 0, "norm": 0}
    while _GDN_UPDATE_PATCHED:
        mod = _GDN_UPDATE_PATCHED.pop()
        fn = getattr(mod, "gated_delta_update", None)
        orig = getattr(fn, "_mlxturbo_gdn_orig", None)
        if orig is not None:
            mod.gated_delta_update = orig
            counts["metal"] += 1
    if model is None:
        return counts
    for layer in _model_layers(model):
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None:
            continue
        base = getattr(type(gdn), "_mlxturbo_gdn_base", None)
        if base is not None:
            gdn.__class__ = base
            counts["prework"] += 1
        if hasattr(gdn, "_mlxturbo_gdn_spec"):
            del gdn._mlxturbo_gdn_spec
        norm = getattr(gdn, "norm", None)
        nbase = getattr(type(norm), "_mlxturbo_gdn_base", None) if norm else None
        if nbase is not None:
            norm.__class__ = nbase
            counts["norm"] += 1
    return counts


def enable_sdpa_split() -> None:
    """decode/verify 幅の sdpa を、vector カーネルの適格幅 (S * gqa_factor <= 32)
    に収まる幅で S 軸に分割して呼ぶ (docs/research/SDPA-WIDTH-WALL.md,
    docs/research/KERNEL-BRIEF-DECODE-BW.md)。

    `Attention.__call__` / `Attention._gather_tile_attn` 側に既にある
    シーム (`getattr(self, "_sdpa_split_width", True)`) を立てるだけ。実際に
    分割するかは毎呼び出し「1 < S <= 8 かつマスクが bool 配列 (または
    causal 文字列で kv が十分長い) かつ S * gqa_factor > 32」で判定し、
    外れれば元の単発 sdpa 呼び出しにそのまま落ちる。

    Flash-Next は Hq=24 / Hk=2 (gqa_factor=12) なので、検証フォワード
    (S=3 など) はこの壁を越えて全 KV を読む素の経路に落ちていた。合成
    マイクロ (`bench/results/sdpa-headdim-micro-decode*.json`) では幅 2 に
    割ると 1 層あたり 4k で 1.52->0.72ms、17k で 3.38->0.60ms、50k で
    7.09->0.82ms (3〜9 倍)。

    分割ロジック自体は 2026-08-31 の変更 (`ebbe4a78`/`7222fce4`) で
    `getattr` ガード無しに既に本番で有効だった。この関数が足すのは
    on/off の knob だけ -- **既定 on**、環境変数 `MLXTURBO_SDPA_SPLIT=0`
    で無効化できる (`enable_gdn_metal_kernel` と同じ書き方)。採否確認は
    `tools/decode_ab.py --knob sdpa-split`。発火の確認は
    `mlxturbo.kernels._fire.snapshot()` の `sdpa_split`。
    """
    import os

    import mlx_lm.models.qwen4_exp as Q

    if os.environ.get("MLXTURBO_SDPA_SPLIT") == "0":
        Q.Attention._sdpa_split_width = False
        return
    Q.Attention._sdpa_split_width = True


def disable_sdpa_split() -> None:
    import mlx_lm.models.qwen4_exp as Q

    Q.Attention._sdpa_split_width = False


_SDPA_ROWTILE_ROWS = 0
_SDPA_ROWTILE_TRACE = False
_SDPA_ROWTILE_MIN_S = 64
# 差し替え済みの名前空間。モジュール名 -> 素の関数。族ごとに attention の
# `__call__` が定義されているモジュールが違う (qwen4_exp / qwen3_next / ...)
# ので、「どこを差し替えたか」を覚えておいて二重差し替えを避ける。
_SDPA_ROWTILE_ORIGS: dict = {}
# 行タイルが得をする条件。MLX 0.32.2 の fast sdpa は head_dim <= 128 なら
# steel のカーネルでマスク済みタイルを飛ばすので、行タイルに割る取り分が無い。
# 256 は fallback (matmul->where->softmax->matmul) に落ちる = 上三角を毎回
# 計算しているので、そこだけを対象にする。
_SDPA_ROWTILE_MIN_HEAD_DIM = 129
_SDPA_ROWTILE_FN = "scaled_dot_product_attention"


def _sdpa_rowtile_call(orig, q, k, v, cache, scale, mask, sinks, rows):
    """行タイルの本体 (段 P5)。q を `rows` 行ずつに割り、タイル t は K/V を
    前方 `[0, offset+(t+1)*rows)` だけに絞って `orig` (素の
    `scaled_dot_product_attention`) を呼び、`concatenate(axis=2)` で戻す。

    QSA の可視集合は `block_end <= q_col` / tail `col <= q_col` なので、
    各行タイルの可視 key は元から `[0, offset+(t+1)*rows)` の中に全部入る
    (近似無し --- 切り捨てるのは、そのタイルのどの行からも見えないと
    証明済みの列だけ)。`mask=="causal"` はタイルごとにそのまま "causal" を
    渡す -- fallback (`mlx/fast.cpp`) は `offset = kL - qL` で対角を出すので、
    K/V を `[0, kv_end)` に切ってあれば各タイルの対角は自動的に
    `offset+t0` (= このタイルの先頭クエリの絶対位置) になり、正しい。
    """
    import mlx.core as mx

    S = q.shape[2]
    kv_len = k.shape[2]
    offset = kv_len - S
    outs = []
    t0 = 0
    while t0 < S:
        t1 = min(t0 + rows, S)
        kv_end = offset + t1
        k_t = k[:, :, :kv_end, :]
        v_t = v[:, :, :kv_end, :]
        q_t = q[:, :, t0:t1, :]
        m_t = mask if isinstance(mask, str) else mask[..., t0:t1, :kv_end]
        outs.append(orig(q_t, k_t, v_t, cache=cache, scale=scale, mask=m_t, sinks=sinks))
        t0 = t1
    return mx.concatenate(outs, axis=2)


def _make_sdpa_rowtile_dispatch(modname: str, orig):
    """`modname` の `scaled_dot_product_attention` の身代わりを作る (段 P5)。

    族のモデルファイルはモジュール直下に `scaled_dot_product_attention` と
    いう名前を import している (`from .base import ...`)。qwen4_exp は
    `_vendor/qwen4_exp.py` (`_arch_registry.py` の meta_path フックで
    `mlx_lm.models.qwen4_exp` として読み込まれる)、Qwen3.8-27B は
    `mlx_lm.models.qwen3_next` (qwen3_5 が `Qwen3NextAttention` を import
    しているので、sdpa を呼ぶのは qwen3_next 側の名前)。その名前をこの
    関数に差し替えることで、モデルファイルを 1 行も触らずに
    `Attention.__call__` の dense 分岐 (S>=64、量子化 cache でない) だけを
    行タイルに割る (`docs/research/IDEAS-2026-09-03.md` の P5)。`Attention`
    クラス自体には触らないので、既存の 2 枠 (`_positions`/`_final_mask`、
    batch.py/batch_spec.py) は増えない。

    **qwen4_exp の `_gather_tile_attn` (段 3(b)、`MLXTURBO_GATHER_ATTN`/
    `MLXTURBO_PREFILL_ATTN` -- 既定 on -- が立てる) の else 節も同じ名前を
    呼ぶ。**そちらの k/v は union で `take_along_axis` して集めた列で、
    位置順ではない。行タイルの「前方 `[0, kv_end)` だけ見れば残りは因果的に
    不可視」という前提は位置順の K/V でしか成り立たないので、誤って
    そちらを行タイル化すると shape は揃ったまま**サイレントに違う値を
    返しうる**。呼び出し元フレームの関数名とモジュール名で区別する
    (`Attention.__call__` だけ通す、`_gather_tile_attn` は素通しさせる ---
    qwen4_exp でこの名前を呼ぶのはこの 2 か所だけ、qwen3_next では
    `Qwen3NextAttention.__call__` の 1 か所だけ、2026-09 時点で grep 済み)。
    """

    def _dispatch(queries, keys, values, cache=None, scale=1.0, mask=None, sinks=None):
        rows = _SDPA_ROWTILE_ROWS
        if (
            rows > 0
            and sinks is None
            and not (cache is not None and hasattr(cache, "bits"))
        ):
            mask_ok = (mask == "causal") if isinstance(mask, str) else (mask is not None)
            if (
                mask_ok
                and queries.shape[2] >= _SDPA_ROWTILE_MIN_S
                and queries.shape[2] > rows
            ):
                import sys

                frame = sys._getframe(1)
                if (
                    frame.f_code.co_name == "__call__"
                    and frame.f_globals.get("__name__") == modname
                ):
                    if _SDPA_ROWTILE_TRACE:
                        from mlxturbo.kernels import _fire

                        _fire.bump("sdpa_rowtile")
                    return _sdpa_rowtile_call(
                        orig, queries, keys, values, cache, scale, mask, sinks, rows
                    )
        return orig(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)

    _dispatch.__name__ = "_sdpa_rowtile_dispatch"
    _dispatch.__qualname__ = f"_sdpa_rowtile_dispatch[{modname}]"
    _dispatch._mlxturbo_rowtile_module = modname
    return _dispatch


def _sdpa_rowtile_patch(ns: dict, modname: str) -> bool:
    """名前空間 `ns` (モジュールの `__dict__`) の
    `scaled_dot_product_attention` を身代わりに差し替える。

    既に差し替えてあるか、その名前が呼べる関数でなければ何もしない
    (戻り値は「今回新しく差し替えたか」)。
    """
    if modname in _SDPA_ROWTILE_ORIGS:
        return False
    orig = ns.get(_SDPA_ROWTILE_FN)
    if not callable(orig) or getattr(orig, "_mlxturbo_rowtile_module", None):
        return False
    _SDPA_ROWTILE_ORIGS[modname] = orig
    ns[_SDPA_ROWTILE_FN] = _make_sdpa_rowtile_dispatch(modname, orig)
    return True


def _sdpa_rowtile_attn_namespaces(model):
    """行タイルを差せる attention の `__call__` の名前空間を列挙する。

    戻り値は `[(モジュール名, その `__call__` の globals, 層数), ...]`。
    契約 (`docs/BACKLOG.md` の「動的な構造の探索 (duck typing)」):

      1. 層が `self_attn` を持ち、そこに `q_proj` がある (attention 層)。
      2. `head_dim` が :data:`_SDPA_ROWTILE_MIN_HEAD_DIM` 以上 --- MLX の
         fast sdpa がタイルを飛ばさない (fallback に落ちる) 幅だけが的。
      3. その `__call__` が Python 関数で、定義元の名前空間に
         `scaled_dot_product_attention` という呼べる名前がある。

    1 つでも欠ければその層は数えない。全部欠ければ空リストで、
    `enable_sdpa_rowtile` はその族に対して何もしない。
    """
    found: dict = {}
    for layer in _model_layers(model):
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(attn, "q_proj", None) is None:
            continue
        head_dim = getattr(attn, "head_dim", None)
        if not isinstance(head_dim, int) or isinstance(head_dim, bool):
            continue
        if head_dim < _SDPA_ROWTILE_MIN_HEAD_DIM:
            continue
        call = getattr(type(attn), "__call__", None)
        ns = getattr(call, "__globals__", None)
        if not isinstance(ns, dict):
            continue
        modname = ns.get("__name__")
        if not isinstance(modname, str):
            continue
        fn = ns.get(_SDPA_ROWTILE_FN)
        if not callable(fn):
            continue
        if modname in found:
            found[modname][1] += 1
        else:
            found[modname] = [ns, 1]
    return [(m, ns, n) for m, (ns, n) in found.items()]


def enable_sdpa_rowtile(model=None, rows: int = 256) -> int:
    """P5: `Attention.__call__` の dense sdpa (S>=64、量子化 cache でない) を
    q 行 `rows` 行ずつのタイルに割り、各タイルは前方 K/V だけを見て呼ぶ
    (`docs/research/IDEAS-2026-09-03.md` の P5)。MLX 0.32.2 の sdpa は
    head_dim=256・S>8 では常に fallback (matmul->where->softmax->matmul) で
    タイルを飛ばさないので、現チャンクの上三角ぶんを毎回無駄に計算している
    --- 可視集合はタイル分割の前後で不変 (近似無し) だが、PV の縮約が
    タイルごとに切れて和の順序が変わるので出力はビット一致しない
    (`tools/decode_ab.py --knob sdpa-rowtile` は `control_identical=False`、
    KLD で確認する)。

    モデルファイルは触らない --- モジュール直下の
    `scaled_dot_product_attention` という名前 (`from .base import ...` で
    import された参照) を身代わりに差し替えるだけ。**どのモジュールを
    差し替えるかは族で決め打ちしない**: `model` の層を歩いて attention 層
    (`self_attn` に `q_proj`、`head_dim` が fallback 幅) を見つけ、その
    `__call__` が定義されている名前空間を差す
    (`_sdpa_rowtile_attn_namespaces`)。qwen4_exp は
    `mlx_lm.models.qwen4_exp`、Qwen3.8-27B (qwen3_5) は
    `mlx_lm.models.qwen3_next` になる。qwen4_exp は `model=None` でも
    呼べる従来の呼び方を保つため無条件でも差す (層が 1 つも無ければ
    ただ差し替えるだけで発火しない)。

    戻り値は行タイルが当たる attention 層の数 (契約が合わなければ 0)。

    既定 off。環境変数 `MLXTURBO_SDPA_ROWTILE=<R>` でも上書きできる (未設定/0
    なら off)。`MLXTURBO_SDPA_ROWTILE_TRACE=1` で発火するたびに
    `mlxturbo.kernels._fire.bump("sdpa_rowtile")` が乗る
    (`mlxturbo.kernels._fire.snapshot()` で回数を見る)。

    proof-of-life は `tools/sdpa_rowtile_micro.py` (モデル無し)。判定は
    `tools/decode_ab.py --knob sdpa-rowtile` の prefill_s と tok/round
    (悪化しないこと)。
    """
    global _SDPA_ROWTILE_ROWS, _SDPA_ROWTILE_TRACE
    import os

    env = os.environ.get("MLXTURBO_SDPA_ROWTILE")
    r = int(env) if env else int(rows)
    _SDPA_ROWTILE_TRACE = os.environ.get("MLXTURBO_SDPA_ROWTILE_TRACE") == "1"
    if r <= 0:
        _SDPA_ROWTILE_ROWS = 0
        return 0

    # qwen4_exp は従来どおり無条件 (`model=None` の呼び方を保つ)
    try:
        import mlx_lm.models.qwen4_exp as Q
    except Exception:  # noqa: BLE001 -- 族のモジュールが無い環境でも進む
        pass
    else:
        _sdpa_rowtile_patch(Q.__dict__, Q.__name__)

    n = 0
    for modname, ns, count in _sdpa_rowtile_attn_namespaces(model):
        _sdpa_rowtile_patch(ns, modname)
        if modname in _SDPA_ROWTILE_ORIGS:
            n += count
    _SDPA_ROWTILE_ROWS = r
    return n


def disable_sdpa_rowtile() -> None:
    global _SDPA_ROWTILE_ROWS

    _SDPA_ROWTILE_ROWS = 0


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
#
# scope (2026-09-03): 4 種類まとめての A/B (17k prefill +62%) は experts の
# gather 連結込みだった。attention だけを切り出して測るための絞り込み。
# `enable_wide_projections(model, scope={"attn"})` で attn 以外を素通しにする。
# 既定 (scope=None かつ env 未設定) は今まで通り全部で、呼び出し互換は崩さない。

_WIDE_SCOPES = frozenset({"gdn", "attn", "shared", "experts"})


def _resolve_wide_scope(scope):
    """`scope` 引数と `MLXTURBO_WIDE_SCOPE` (カンマ区切り) から有効な集合を決める。

    優先順位: 明示引数 > 環境変数 > 既定 (全部)。呼び出し側が `scope` を
    渡さず env も立っていなければ、これまでの「常に全部」という挙動のまま。
    """
    import os

    if scope is not None:
        resolved = {s for s in scope}
    else:
        env = os.environ.get("MLXTURBO_WIDE_SCOPE")
        if env:
            resolved = {s.strip() for s in env.split(",") if s.strip()}
        else:
            resolved = set(_WIDE_SCOPES)
    unknown = resolved - _WIDE_SCOPES
    if unknown:
        raise ValueError(
            f"unknown wide-projection scope entries: {sorted(unknown)}"
            f" (valid: {sorted(_WIDE_SCOPES)})"
        )
    return resolved


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
        for layer in _model_layers(model):
            yield layer
        if mtp is not None:
            for layer in mtp.layers:
                yield layer

    for layer in each_layer():
        for mod, attr in (
            (getattr(layer, "linear_attn", None), "_wide_in"),
            (getattr(layer, "self_attn", None), "_wide_qkv"),
            (getattr(layer, "mlp", None), "_wide_shared"),
        ):
            if mod is not None and getattr(mod, attr, None) is not None:
                setattr(mod, attr, None)
                n += 1
        sa = getattr(layer, "self_attn", None)
        if sa is not None and getattr(sa, "_wide_min_rows", None) is not None:
            sa._wide_min_rows = None
        # experts の連結は mlp.switch_mlp に _fused_w/_fused_s/_fused_b として
        # 置かれる (enable_wide_projections:760)。dispatched() の wide() は
        # `hasattr(self, "_fused_w")` で見るので、setattr(..., None) では
        # 属性自体が残って hasattr が True のままになる -- delattr で消す。
        mlp = getattr(layer, "mlp", None)
        sw = getattr(mlp, "switch_mlp", None) if mlp is not None else None
        if sw is not None and getattr(sw, "_fused_w", None) is not None:
            for attr in ("_fused_w", "_fused_s", "_fused_b"):
                delattr(sw, attr)
                n += 1
    return n


def enable_wide_projections(model, mtp=None, scope=None) -> dict:
    """読み込み済みモデルに連結射影を仕込む。戻り値は種類別の適用層数。

    ``scope``: 仕込む種類を絞る集合 (``{"gdn", "attn", "shared", "experts"}``
    の部分集合)。``None`` なら `MLXTURBO_WIDE_SCOPE` (カンマ区切り) を見て、
    それも無ければ全部 (これまでの既定動作のまま)。scope に入っていない
    種類は counts が 0 のまま、対応する属性も仕込まれない。

    attention (``scope`` に "attn" を含む場合) は仕込んだ ``self_attn`` に
    ``_wide_min_rows`` も置く (`MLXTURBO_WIDE_MIN_ROWS`、既定 64)。
    `_vendor/qwen4_exp.py` の `Attention._qkv` がこれを見て、行数
    (B*S) がこの値未満なら `_wide_qkv` を無視し、decode 幅は従来の
    個別 3 射影に落ちる (連結射影は M が大きい prefill だけを狙う)。
    """
    import os

    import mlx.core as mx

    scope_set = _resolve_wide_scope(scope)
    min_rows = int(os.environ.get("MLXTURBO_WIDE_MIN_ROWS", "64"))

    counts = {"gdn": 0, "attn": 0, "shared": 0, "experts": 0}

    def each_layer():
        for layer in _model_layers(model):
            yield layer
        if mtp is not None:
            for layer in mtp.layers:
                yield layer

    for layer in each_layer():
        if "gdn" in scope_set:
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
        if "attn" in scope_set:
            sa = getattr(layer, "self_attn", None)
            if sa is not None and hasattr(sa, "q_proj"):
                cat = _cat_quantized([sa.q_proj, sa.k_proj, sa.v_proj])
                if cat is not None:
                    w, sc, bi, gs, bits = cat
                    c1 = sa.n_heads * sa.head_dim * 2
                    c2 = c1 + sa.n_kv_heads * sa.head_dim
                    sa._wide_qkv = (w, sc, bi, gs, bits, (c1, c2))
                    sa._wide_min_rows = min_rows
                    counts["attn"] += 1
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and getattr(mlp, "_router513", None) is not None:
            continue          # shared 畳み込み済み: shared/experts はそちらが持つ
        if "shared" in scope_set:
            se = getattr(mlp, "shared_expert", None) if mlp is not None else None
            if se is not None:
                cat = _cat_quantized(
                    [se.gate_proj, se.up_proj, mlp.shared_expert_gate])
                if cat is not None:
                    w, sc, bi, gs, bits = cat
                    h = _rows(se.gate_proj)
                    mlp._wide_shared = (w, sc, bi, gs, bits, h)
                    counts["shared"] += 1
        if "experts" in scope_set:
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
_MOE_DISPATCH_DEC_FUSED_ON = False  # enable_moe_decode_fused が設定
_MOE_DISPATCH_DEC_FUSED_MAX_ROWS = 4  # decode / verify 幅の上限 (行数 B*S)。既定の配線は S <= 3 (depth 2)
# `MLXTURBO_MOE_DECODE_FUSED` の既定。auto = 非 NAX 機で on / NAX 機で off
# (自前カーネルは NAX 機に当てない方針)。"0" で off。
_MOE_DEC_FUSED_DEFAULT = "auto"  # 2026-09-04 03:05 から auto (下の enable_moe_decode_fused の docstring)


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

    def moe_dec_fused(self, x, indices):
        """enable_moe_decode_fused: decode / verify 幅 (行数 <= 8) を
        `kernels/moe_decode_fused.py` の `fused:1` 変種に差し替える。

        素の経路は (行 x top_k) の対を `_gather_sort` で並べ替えてから
        `gather_qmm` を 3 回呼ぶ。`gather_qmm` は decode 幅で x を top_k 行に
        複製する `g1_copy` (96 x 15.6 us = 1.49 ms/round) と添字の `arange`
        (144 x 6 us = 0.86 ms/round) を出す (CATCHUP 2026-09-03 21:10)。
        `fused:1` は重複をまとめず (rmax=1)、ソートもせず、x を対ごとに
        カーネル内で `pr / top_k` から引くので、複製も添字生成も要らない。
        gate/up が 1 カーネル、down が 1 カーネル。

        **rmax は 1 に固定する。**重複をまとめる rmax>=2 は PoL で負けた
        (重複の対が threadgroup を増やして帯域を稼いでいた。冷 micro で
        S=1 1.109 / S=3 1.066、`kernels/moe_decode_fused.py` の冒頭)。

        適格でなければ None (素通し)。"""
        from .kernels import _fire
        from .kernels import moe_decode_fused as mdf

        topk = indices.shape[-1]
        rows = indices.size // topk
        if rows > _MOE_DISPATCH_DEC_FUSED_MAX_ROWS:
            return None
        gp, up, dp = self.gate_proj, self.up_proj, self.down_proj
        if not mdf.eligible(x, gp, up, dp):
            return None
        K = x.shape[-1]
        if K % 512 != 0:                     # gate/up カーネルの前提
            return None
        idx_flat = indices.reshape(-1)
        # h: (rows*topk, moe_intermediate) = silu(gate) * up
        h = mdf.gate_up(x.reshape(-1, K), idx_flat, gp, up, topk, rmax=1)
        y = mdf.down(h, idx_flat, dp, topk, K, rmax=1)
        _fire.bump("moe_decode_fused")
        # 素の経路の out.squeeze(-2) と同じ (…, top_k, hidden)
        return y.reshape(*indices.shape, K)

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
        if _MOE_DISPATCH_DEC_FUSED_ON:
            out = moe_dec_fused(self, x, indices)
            if out is not None:
                return out
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
    for layer in _model_layers(model):
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


def enable_moe_decode_fused(model=None, max_rows: int | None = None,
                            mode: str | None = None) -> None:
    """decode / verify 幅の MoE を `kernels/moe_decode_fused.py` の `fused:1`
    に差し替える (gate/up 1 カーネル + down 1 カーネル、重複もソートも無し)。

    ``mode``: ``"auto"`` (非 NAX 機だけ on) / ``"1"`` / ``"0"``。``None`` なら
    環境変数 `MLXTURBO_MOE_DECODE_FUSED` を読む (既定は `_MOE_DEC_FUSED_DEFAULT` = auto。
    2026-09-04 03:05 に既定 on: 短 ms/round -1.2〜-1.3% × 2 回 / 17k -1.4%、本番重みの
    逆量子化 fp32 参照テストで自前が素より近い (中央 0.39 倍、反転 0/96)、S=1 の
    Δ KLD +0.00036 (受け入れ幅 +0.0005 の中、対 bf16 teacher)。`=0` で off)。
    NAX 機の判定は `moe_grouped_gemm.is_nax_device` で `enable_qmm_wide` と同じ
    もの -- あちらは MLX 側が別カーネルを持っていて A/B 未実施なので、自前
    カーネルは当てない。ゲートを関数自身の中に持たせるのは
    `enable_moe_verify_gather` と同じ作法 (呼び出し側が env を忘れても既定が保たれる)。

    行数 (B*S) が `MLXTURBO_MOE_DECODE_FUSED_MAX_ROWS` (既定 4) を超える幅は
    必ず素の経路に落ちる (無言で切り替わる。既定の配線は depth 2 で S <= 3、
    `--max-batch-spec` >= 2 のバッチ検証は rows >= 6 で素に落ちる。PoL は
    S=1 0.929 / S=3 0.979 で S が増えるほど取り分が細るので上限は 4 に留める)。prefill 幅は `enable_moe_grouped_gemm` (段 P3) と
    `enable_moe_down_epilogue` (段 P7 第 2 段) のまま -- どちらも
    `QuantizedSwitchLinear.__call__` / `_moe_combine_fold` 側の差し替えで、
    ここ (`SwitchGLU.__call__` の分岐) とは掛かる場所が違うので共存する。
    `enable_moe_combine_fold` の行数ゲート (既定 64) より小さい幅しか通らない
    ので、fold が発火する側と重なることも無い。

    数値: 積和は fp32 で 1 回丸め (素の `gather_qmm` は対ごとに丸める) なので
    ビット一致はしない。判定線 (親): 短文脈 3 本 x 512 の ms/round -1% 以上、
    head の不一致が丸め級。

    実体は `_ensure_moe_dispatch_installed` が入れる統合ディスパッチ関数
    (C1) の一分岐。`model` は受け取るだけで使わない (他の enable_* と
    呼び口を揃えるため)。

    発火の確認: `mlxturbo.kernels._fire.snapshot()` の `moe_decode_fused`。
    """
    global _MOE_DISPATCH_DEC_FUSED_ON, _MOE_DISPATCH_DEC_FUSED_MAX_ROWS
    import os

    from .kernels import moe_grouped_gemm as mgg

    if mode is None:
        mode = (os.environ.get("MLXTURBO_MOE_DECODE_FUSED",
                               _MOE_DEC_FUSED_DEFAULT).strip().lower()
                or _MOE_DEC_FUSED_DEFAULT)
    if mode in ("1", "on", "true"):
        mode = "on"
    elif mode in ("0", "off", "false"):
        mode = "off"
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"mode={mode!r} は auto / 1 / 0 のどれかにすること")
    if mode == "off" or (mode == "auto" and mgg.is_nax_device()):
        _MOE_DISPATCH_DEC_FUSED_ON = False
        return
    if max_rows is None:
        max_rows = int(os.environ.get("MLXTURBO_MOE_DECODE_FUSED_MAX_ROWS", "4"))
    _MOE_DISPATCH_DEC_FUSED_MAX_ROWS = max_rows
    if _MOE_DISPATCH_DEC_FUSED_ON:
        return
    _MOE_DISPATCH_DEC_FUSED_ON = True
    _ensure_moe_dispatch_installed()


def disable_moe_decode_fused(model=None) -> None:
    """`enable_moe_decode_fused` を打ち消す (フラグを下ろすだけ)。A/B で
    交互に測るために要る。"""
    global _MOE_DISPATCH_DEC_FUSED_ON
    _MOE_DISPATCH_DEC_FUSED_ON = False


def enable_moe_combine_fold(model) -> int:
    """SparseMoeBlock の重み付き和を down_proj の前に畳む (行数ゲート付き、
    ``SparseMoeBlock._combine_fold_min_s`` 分岐、
    `mlxturbo/_vendor/qwen4_exp.py` の `_moe_combine_fold`)。素の経路は
    switch_mlp の出力 (rows, top_k, hidden_size)=(rows, top_k, 2560) を
    実体化してから router 重み w を掛けて sum するが、down_proj は bias
    無しの線形写像なので、w を down_proj の「入力」(SwiGLU 出力、
    (rows, top_k, moe_intermediate_size)=(rows, top_k, 640)) に先掛けしても
    数式上は同じ結果になる。乗算が触る実体は 4 分の 1 で済む (top_k 軸の
    和自体は down_proj の出力側で `sum(axis=-2)` のまま取るので、その
    実体化は残る)。

    実測 (prefill 8k、`tools/prefill_anatomy.py --ctx 8000`、
    `bench/results/logs/prefill-anatomy-8k-0903.log`、
    `docs/research/SESSION-2026-09-02-CATCHUP.md` の「prefill 短文脈の内訳、
    8k」): MoE 48 層の内訳で「ルータ重み + top-K 縮約」が 142ms/チャンク
    (効率 9.9%) と最大。

    数式上は等価だが bf16 の丸め順が変わるため fold 発火時の出力はビット
    不一致 (積和の結合順が変わるだけ、bench/test_moe_combine_fold.py で
    許容誤差 1e-2 を確認)。fold が発火する側は switch_mlp.__call__ を
    経由しない (gate_proj/up_proj/down_proj を直接呼ぶ) ため、その間は
    同じ SwitchGLU.__call__ に載っている他の 3 経路
    (enable_wide_projections の連結射影 / enable_gather_sort のソート閾値
    変更 / enable_moe_glu / enable_moe_verify_gather) を素通りする --
    ソート判定・並べ替え自体は `_moe_combine_fold` が `MLXTURBO_SORT_MIN`
    を読んで自前で再現しているので正しさは保たれるが、それらのカーネル
    差し替えの効果は乗らない。

    **行数ゲート**: 初回の in-model A/B (2026-09-03) は prefill 8k -2.2%/
    17k -2.5% と勝った一方、decode は 8k +1.3%/17k +0.6%/短文脈 ms/round
    +1.4% (tok/round -4.0%) と負けた。行数 (B×S、`x.shape[0]*x.shape[1]`)
    が少ないと、乗算を 4 分の 1 に減らす分より gate_proj/up_proj/down_proj
    を個別に呼ぶディスパッチ増分のほうが効く。そこで
    `MLXTURBO_MOE_COMBINE_FOLD_MIN_S` (既定 64) 未満の行数では必ず素の経路
    に落とす (decode/verify 幅 S<=8 は必ずここに入る)。この閾値はここで
    起動時に 1 回だけ読んで `_combine_fold_min_s` に積む -- `__call__` の
    ホットパスでは getenv しない。

    2026-09-03 に既定 on にした (行数ゲート込みで再測定予定。prefill は
    勝ち筋が確認済みで、decode 幅は行数ゲートで必ず素の経路に固定される
    ため理論上ここでは負けない)。`MLXTURBO_MOE_COMBINE_FOLD=0` で無効化
    できる (呼び出し側が env var を忘れても既定 on が保たれるように、
    ゲートを関数自身の中に持たせている。`enable_gdn_metal_kernel` と同じ
    「既定 on、`=0` で無効化」の作法)。prefill に効く変更なので decode_ab
    の DECODE_ONLY には入れない。

    戻り値は適用した層数 (MoE 層は全 48 層)。
    """
    import os

    if os.environ.get("MLXTURBO_MOE_COMBINE_FOLD") == "0":
        return 0
    min_s = int(os.environ.get("MLXTURBO_MOE_COMBINE_FOLD_MIN_S", "64"))
    n = 0
    for layer in _model_layers(model):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp"):
            mlp._combine_fold_min_s = min_s
            n += 1
    return n


def disable_moe_combine_fold(model) -> int:
    """`enable_moe_combine_fold` を打ち消す。戻り値は外した数。A/B で交互に測るために要る。"""
    n = 0
    for layer in _model_layers(model):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and getattr(mlp, "_combine_fold_min_s", None) is not None:
            mlp._combine_fold_min_s = None
            n += 1
    return n


# --- MoE grouped GEMM (P3、kernels/moe_grouped_gemm.py の第 3 段) -------------
#
# 差し替え先を `SparseMoeBlock.__call__` ではなく
# `mlx_lm.models.switch_layers.QuantizedSwitchLinear.__call__` にしてある。
# 理由は 2 つ:
#
#  - 置き換えたいのは `mx.gather_qmm(..., sorted_indices=True)` 1 本だけで、
#    それはこのクラスの `__call__` にしか無い。SparseMoeBlock 側で受けると
#    `_moe_combine_fold` (vendor) のソート・並べ替え・SwiGLU まで写す羽目に
#    なり、「写しを増やさない」規則 (CLAUDE.md) に反する。
#  - 本番の prefill は既定 on の `MLXTURBO_MOE_COMBINE_FOLD` 経路を通るので
#    `SwitchGLU.__call__` (統合ディスパッチ C1) を経由しない。gate/up/down は
#    そこから `QuantizedSwitchLinear` を直接呼ぶので、ここに置けば fold 経路と
#    素の SwitchGLU 経路の両方を 1 か所で拾える。
#
# 差し替えは `QuantizedSwitchLinear` に対してこれ 1 個だけ (SwitchGLU 側の
# 統合ディスパッチとは別クラス)。C1 と同じく「インストールは 1 回、経路の
# 選択は呼び出し時にモジュール変数で」の作法に合わせてあるので、A/B で
# 交互に切り替えても掛け順が壊れない。
_MOE_GEMM_STOCK = None        # 素の SL.QuantizedSwitchLinear.__call__
# "off" (素通し) / "seg" (自前カーネル) / "pad16" (16 行揃え + 既製 gather_qmm)
_MOE_GEMM_MODE = "off"
_MOE_GEMM_MIN_ROWS = 1024     # インストール時に 1 回だけ読む (ホットパスで getenv しない)
_MOE_GEMM_TRACE = False
# seg のタイル設定 (P3 混合タイル)。`enable_moe_grouped_gemm` の引数か
# `MLXTURBO_MOE_GEMM_MIX` で決める。mix が None なら現行の seg32 (BM=32/WM=2)。
# **`segment_tables` と `qmm_segmented` に同じ値を渡すこと。**表とカーネルで
# 数え方が食い違うと行の割り当てが壊れる (どちらも `rows_e < mix` で分ける)。
_MOE_GEMM_BM = 32
_MOE_GEMM_WM = None           # None = bm 既定 (bm=32 なら WM=2、16 なら 1)
_MOE_GEMM_MIX = None          # 48 なら行数 < 48 の専門家だけ 16 行タイル (WM=1)
# (indices, num_experts, tables) の 1 段キャッシュ。`_moe_combine_fold` も
# `gather_sort` も gate/up/down の 3 本に**同じ添字オブジェクト**を渡すので、
# 同一性 (`is`) で引けばテーブル構築は 3 回に 1 回で済む。参照を持っている
# 間は id が再利用されないので、同一性判定は安全。
_MOE_GEMM_TABLES = None
_MOE_GEMM_PAD_TABLES = None   # pad16 用 (indices, num_experts, (src, idx_pad, keep))
_MOE_GEMM_PAD_X = None        # pad16 の x_pad を gate/up で使い回す 1 段キャッシュ
_MOE_GEMM_TRACE_SEEN: set = set()

# pad16 の揃え幅。素の `affine_gather_qmm_rhs` のタイル (quantized.cpp:1652 の
# bm=16) に合わせてある。ここを 32 にすると水増しが倍になる
_MOE_GEMM_PAD = 16


def _moe_gemm_tables(indices, num_experts):
    """`indices` (専門家順にソート済み、(M,)) から `(row_start, tile_prefix)`。

    host 同期は入らない (`counts_from_sorted_ids` も `segment_tables` も
    mx op だけ)。同じ添字での 2 回目以降はキャッシュを返す。

    タイル設定 (`_MOE_GEMM_BM` / `_MOE_GEMM_MIX`) も鍵に入れる。A/B で設定を
    切り替えたときに、前の設定で作った表を使い回さないため。
    """
    global _MOE_GEMM_TABLES

    from .kernels import moe_grouped_gemm as mgg

    key = (_MOE_GEMM_BM, _MOE_GEMM_MIX)
    cached = _MOE_GEMM_TABLES
    if (cached is not None and cached[0] is indices
            and cached[1] == num_experts and cached[2] == key):
        return cached[3]
    tables = mgg.segment_tables(
        mgg.counts_from_sorted_ids(indices, num_experts),
        bm=_MOE_GEMM_BM,
        mix_threshold=_MOE_GEMM_MIX,
    )
    _MOE_GEMM_TABLES = (indices, num_experts, key, tables)
    return tables


def _moe_gemm_pad_tables(indices, num_experts):
    """pad16 (variant C) の 3 本を作る。戻り値は `(src, idx_pad, keep)`。

      - ``src`` (M_pad,)     : 水増し後の行 -> 元の x の行。端数のダミー行は
                               その専門家の**最後の実在行**を指す (0 行でも
                               よいが、行が全部同じ専門家に属していたほうが
                               読みが局所的になる)
      - ``idx_pad`` (M_pad,) : 水増し後の行の専門家 id (`rhs_indices` 用)
      - ``keep`` (M,)        : 元の行 -> 水増し後の行 (出力の取り戻し)

    **ここだけ host 同期が 1 回入る** (`M_pad` の確定)。MLX の配列は形が
    静的なので、`Σ_e ceil(c_e / 16) * 16` を Python の int にしないと
    水増し後の配列を作れない。静的上限 (`M + 16*E`) で張る手もあるが、
    r=40 で行数 +40% (実際の +18% に対して) になって micro の勝ちが消える
    ので採らない。同期は添字 1 つにつき 1 回 (= MoE 層 1 つにつき 1 回。
    gate/up/down の 3 本はキャッシュを共有する)。
    """
    global _MOE_GEMM_PAD_TABLES

    import mlx.core as mx

    from .kernels import moe_grouped_gemm as mgg

    cached = _MOE_GEMM_PAD_TABLES
    if cached is not None and cached[0] is indices and cached[1] == num_experts:
        return cached[2]

    E = num_experts
    pad = _MOE_GEMM_PAD
    counts = mgg.counts_from_sorted_ids(indices, E)          # (E,)
    padded = ((counts + (pad - 1)) // pad) * pad             # (E,)
    zero = mx.zeros((1,), dtype=mx.int32)
    start = mx.concatenate([zero, mx.cumsum(counts)])        # (E+1,)
    pstart = mx.concatenate([zero, mx.cumsum(padded)])       # (E+1,)
    m_pad = int(pstart[E].item())                            # ← 唯一の host 同期

    # 水増し後の行 -> 専門家。境界に 1 を撒いて累積和を取るだけ (探索不要)。
    # 行を持たない専門家は境界が重なるので、そこで id が 2 以上飛ぶ。
    # 長さを M_pad+1 にしてあるのは pstart[e] == M_pad (末尾の空専門家) が
    # 範囲外にならないようにするため
    bumps = mx.zeros((m_pad + 1,), dtype=mx.int32)
    bumps = bumps.at[pstart[1:E]].add(mx.ones((E - 1,), dtype=mx.int32))
    seg = mx.cumsum(bumps)[:m_pad]                           # (M_pad,)
    off = mx.arange(m_pad, dtype=mx.int32) - pstart[seg]
    src = start[seg] + mx.minimum(off, counts[seg] - 1)      # (M_pad,)
    idx_pad = seg.astype(indices.dtype)

    idx_i = indices.astype(mx.int32)
    m = indices.shape[0]
    keep = pstart[idx_i] + (mx.arange(m, dtype=mx.int32) - start[idx_i])

    tables = (src, idx_pad, keep)
    _MOE_GEMM_PAD_TABLES = (indices, num_experts, tables)
    return tables


def _moe_gemm_trace(rows: int, n_out: int) -> None:
    """`MLXTURBO_MOE_GEMM_TRACE=1` のときだけ、初出の形ごとに 1 行 stderr へ。

    終了時に累計 (`_fire` の `moe_grouped_gemm_segmented` /
    `moe_grouped_gemm_pad16`) も 1 行出す。
    """
    import atexit
    import sys

    from .kernels import _fire

    key = (rows, n_out, _MOE_GEMM_MODE)
    if key in _MOE_GEMM_TRACE_SEEN:
        return
    if not _MOE_GEMM_TRACE_SEEN:
        atexit.register(
            lambda: print(
                "[mlxturbo] moe_grouped_gemm 発火合計:"
                f" seg {_fire.snapshot().get('moe_grouped_gemm_segmented', 0)} 回"
                f" / pad16 {_fire.snapshot().get('moe_grouped_gemm_pad16', 0)} 回",
                file=sys.stderr,
            )
        )
    _MOE_GEMM_TRACE_SEEN.add(key)
    print(
        f"[mlxturbo] moe_grouped_gemm 発火 ({_MOE_GEMM_MODE}):"
        f" rows={rows} N={n_out}",
        file=sys.stderr,
    )


def enable_moe_grouped_gemm(
    model=None,
    mode: str = "seg",
    mix_threshold: int | None = None,
    bm: int | None = None,
    wm: int | None = None,
) -> None:
    """prefill 幅の `mx.gather_qmm(sorted_indices=True)` を差し替える。

    ``mode`` は 2 通り:

      - ``"seg"`` (既定): 自前の grouped GEMM
        (`kernels/moe_grouped_gemm.qmm_segmented`) に置き換える
      - ``"pad16"``: 既製の `mx.gather_qmm` のまま、**専門家セグメントを
        16 行の倍数にダミー行で揃えて** straddle を物理的に無くす
        (`tools/gather_qmm_pad_micro.py` の (b) を in-model に持ち込んだ形)。
        micro では r=40 で -11%、r=160 で -3.6%。ただし in-model では
        水増しの gather (x_pad) と出力の取り戻しが**別途**乗る --
        micro の数字は gather_qmm の呼び 1 本だけを測ったもの

    素は BM=16 のタイルを行の並びだけで切るので、専門家境界をまたいだタイルは
    セグメントごとにフル BM x BN x K をやり直す (`quantized.h:2517`)。
    こちらはタイル 1 枚が必ず 1 専門家に収まり、形も dense と同じ
    BM=32 / 128 スレッドになる。**素とビット一致する** (micro の 10 ケースで
    確認済み、`bench/results/moe-grouped-gemm-segmented.json`)。

    発火条件 (どれか外れたら素の `mx.gather_qmm` に落ちる):

      - `sorted_indices=True` で呼ばれている (prefill 幅の gather_sort 経路)
      - 行数が `MLXTURBO_MOE_GEMM_MIN_ROWS` (既定 1024) 以上。decode/verify 幅
        (数十行) は素のまま -- あちらは `gather_qmv` で経路自体が別
      - `MLXTURBO_MOE_GEMM` (auto|on|off) が on を解く。`auto` は非 NAX 機だけ
        on (NAX 機は MLX 側が別カーネルを持っていて A/B 未実施。
        `moe_grouped_gemm.is_nax_device`)
      - 形が `segmented_eligible` を満たす (4bit/gs64、N が 32 の倍数、
        K が 32 と 64 の倍数、bias 無し、affine)。pad16 も同じ判定を使う
        (既製カーネルに投げるだけなので条件はもっと緩いが、A と C を同じ
        母集団で比べるために揃えてある)
      - GPU が既定デバイス (`mx.fast.metal_kernel` なので CPU では使えない)

    **`runner.enable_default_fusions` には配線していない。**in-model A/B
    (`tools/decode_ab.py --knob moe-grouped-gemm`) で prefill が勝ってから
    配線する。micro (行列積だけ) の取り分は r=160 で -5.2%、r=40 で -4.1%。

    ``mode="seg"`` のタイル設定 (P3 混合タイル、2026-09-03):

      - ``mix_threshold``: 専門家ごとに行数がこの値未満なら 16 行タイル、
        それ以外は 32 行タイル (1 dispatch、threadgroup は 64 スレッドなので
        WM は必ず 1)。``None`` なら現行の seg32。micro
        (`tools/moe_grouped_gemm_micro.py --stage segmented`) の取り分は
        mix48 が MoE 層 1 つで r=40 -9.9% / r=160 -5.4%
      - ``bm`` / ``wm``: 混合しないときのタイル行数と simdgroup の本数
        (既定 32 / None = WM 2)。``wm=1`` だけを試す対照用

    引数が ``None`` のときは環境変数 `MLXTURBO_MOE_GEMM_MIX` (既定 `48`、`0` =
    混合なし、`48` で mix48/WM=1) を読む。**設定は `segment_tables` と
    `qmm_segmented` の両方に同じ値が渡る** (`_moe_gemm_tables` と
    `dispatched` がどちらもモジュール変数を見る)。差し替え本体は 1 回しか
    入れ替えないので、A/B で交互に呼んでも設定だけが切り替わる。
    """
    global _MOE_GEMM_STOCK, _MOE_GEMM_MODE, _MOE_GEMM_MIN_ROWS, _MOE_GEMM_TRACE
    global _MOE_GEMM_BM, _MOE_GEMM_WM, _MOE_GEMM_MIX
    import os

    import mlx.core as mx
    import mlx_lm.models.switch_layers as SL

    from .kernels import _fire
    from .kernels import moe_grouped_gemm as mgg

    if mode not in ("seg", "pad16"):
        raise ValueError(f"mode={mode!r} は seg / pad16 のどちらかにすること")
    _MOE_GEMM_MODE = mode if (mode == "pad16" or mgg.enabled()) else "off"
    if mix_threshold is None:
        mix_threshold = int(os.environ.get("MLXTURBO_MOE_GEMM_MIX", "48") or "0")
    # 0 (と env の未設定) は「混合しない」。`qmm_segmented` は
    # `mix_threshold is not None` で混合モードに入るので、ここで None に潰す
    _MOE_GEMM_MIX = int(mix_threshold) if mix_threshold else None
    _MOE_GEMM_BM = 32 if bm is None else int(bm)
    _MOE_GEMM_WM = wm
    if _MOE_GEMM_STOCK is not None:
        return
    _MOE_GEMM_MIN_ROWS = int(os.environ.get("MLXTURBO_MOE_GEMM_MIN_ROWS", "1024"))
    _MOE_GEMM_TRACE = os.environ.get("MLXTURBO_MOE_GEMM_TRACE") == "1"
    _MOE_GEMM_STOCK = SL.QuantizedSwitchLinear.__call__
    stock = _MOE_GEMM_STOCK

    def pad16(self, x_key, x2, indices, w, scales, biases, K):
        """16 行揃え + 既製 `mx.gather_qmm`。x_pad は gate/up で使い回す。

        キャッシュの鍵は**呼び出し側が渡した `x` そのもの** (`x_key`)。
        `x2` は `reshape` の戻りで毎回別オブジェクトになるので鍵にならない。
        gate と up は同じ `x` を受け取る (`_moe_combine_fold` / `gather_sort`
        のどちらも同じ配列を 2 回渡す) ので、水増しの gather は 1 回で済む。
        down は別の配列 (SwiGLU 出力) なのでもう 1 回走る。
        """
        global _MOE_GEMM_PAD_X

        src, idx_pad, keep = _moe_gemm_pad_tables(indices, w.shape[0])
        cached = _MOE_GEMM_PAD_X
        if cached is not None and cached[0] is x_key:
            x_pad = cached[1]
        else:
            x_pad = x2[src]
            _MOE_GEMM_PAD_X = (x_key, x_pad)
        y = mx.gather_qmm(
            x_pad.reshape(-1, 1, K),
            w,
            scales,
            biases,
            rhs_indices=idx_pad,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=True,
        )
        _fire.bump("moe_grouped_gemm_pad16")
        return y[keep]

    def dispatched(self, x, indices, sorted_indices=False):
        # 素通しの判定を先に済ませる (decode 幅はここで全部落ちる)
        if (
            _MOE_GEMM_MODE != "off"
            and sorted_indices
            and x.ndim == 3
            and x.shape[1] == 1
            and x.shape[0] >= _MOE_GEMM_MIN_ROWS
            and indices.ndim == 1
            and indices.shape[0] == x.shape[0]
            and "bias" not in self
            and getattr(self, "mode", "affine") == "affine"
            and mx.default_device() == mx.gpu
        ):
            w = self["weight"]
            scales = self["scales"]
            biases = self.get("biases")
            rows, _, K = x.shape
            x2 = x.reshape(rows, K)
            if biases is not None and mgg.segmented_eligible(
                x2, w, scales, biases, self.group_size, self.bits
            ):
                if _MOE_GEMM_TRACE:
                    _moe_gemm_trace(rows, w.shape[1])
                if _MOE_GEMM_MODE == "pad16":
                    # 出力は (M, 1, N) のまま返る (keep が行を選ぶだけ)
                    return pad16(self, x, x2, indices, w, scales, biases, K)
                out = mgg.qmm_segmented(
                    x2,
                    w,
                    scales,
                    biases,
                    None,
                    tables=_moe_gemm_tables(indices, w.shape[0]),
                    # 表と同じ設定を渡す (食い違うと行の割り当てが壊れる)
                    bm=_MOE_GEMM_BM,
                    wm=_MOE_GEMM_WM,
                    mix_threshold=_MOE_GEMM_MIX,
                    group_size=self.group_size,
                    bits=self.bits,
                )
                # 素の gather_qmm は (M, 1, N) を返す。呼び手 (SwitchGLU /
                # _moe_combine_fold) はこの中間次元を前提にしているので戻す
                return out.reshape(rows, 1, w.shape[1])
        return stock(self, x, indices, sorted_indices)

    SL.QuantizedSwitchLinear.__call__ = dispatched


def disable_moe_grouped_gemm() -> None:
    """`enable_moe_grouped_gemm` を打ち消す (フラグを下ろすだけ)。

    差し替え自体は残るが `dispatched` が素通しに固定される。A/B で交互に
    測るためにこの形にしてある (統合ディスパッチ C1 と同じ理屈)。
    """
    global _MOE_GEMM_MODE, _MOE_GEMM_TABLES, _MOE_GEMM_PAD_TABLES, _MOE_GEMM_PAD_X
    _MOE_GEMM_MODE = "off"
    _MOE_GEMM_TABLES = None
    _MOE_GEMM_PAD_TABLES = None
    _MOE_GEMM_PAD_X = None


# --- P7 第 2 段: combine を down GEMM の store に畳む -------------------------
#
# prefill 幅の MoE で行列積以外に残っていた費用 (8k、層 20、M=2048、
# `bench/results/moe-split.json`) のうち、
#
#   combine の weight_mul_cast  0.87 ms  ((act * w) の実体化 + bf16 への cast)
#   combine の unsort_sum       1.07 ms  (`out[inv_order]` の gather + sum)
#   sort の argsort 2 本目       ~0.17 ms  (`inv_order = argsort(order)`)
#
# の 3 つは、down GEMM (自前の segmented カーネル) の store でまとめて畳める:
#
#   - ルータ重みは累算器 (fp32) に掛けてから 1 回だけ丸めて書く
#     -> `act` の実体化 (16384x640 の read + fp32 write + cast) が丸ごと消える
#   - 行 s の結果を `order[s]` の位置へ直接書く (scatter)
#     -> `out[inv_order]` の gather (16384x2560 の read+write) が消える。
#        `result[order[s]] = out[s]` なので使うのは `order` そのもので、
#        `inv_order` (2 本目の argsort) も要らなくなる
#
# 残るのは top_k 軸の和 (`sum(axis=-2)`) だけ。専門家をまたぐ行の和なので
# タイル内では取れず、atomic を使うと非決定になるので畳まない。
#
# 2 つ目 (`MLXTURBO_MOE_GATHER_FOLD` 相当のフラグ): x の gather 0.49 ms も
# gate/up GEMM の**行の読み方**に畳める (`row_src`)。素は
# `xx[order // top_k]` で 16384x2560 (84MB) を実体化してから GEMM に渡すが、
# GEMM が行ごとに `row_src[r]` を見て読めば実体化が要らない (読む先の x は
# 2048 行 = 10MB で、キャッシュにも乗りやすくなる)。
#
# 差し替え口は vendor の `_moe_combine_fold` に置いた 1 個のフック
# (`_MOE_DOWN_EPILOGUE`)。`QuantizedSwitchLinear.__call__` (P3 の差し替え)
# 側では受けられない -- あちらは `order` も `w` も `row_src` も見えない。
# **フックが受け持つのは gate/up -> SwiGLU -> down の 3 本まとめて**で、
# 素の経路 (フック未設置・不適格) は 1 行も変わらない。
_MOE_DOWN_EPI_ON = False
_MOE_GATHER_FOLD_ON = False
# "combine" (down の後ろを 1 カーネルに畳む) / "epi" (down の store に畳む)
_MOE_FOLD_MODE = "combine"


def _moe_fold_block(switch_mlp, x_rows, row_src, idx_s, order, w_flat):
    """`_moe_combine_fold` から呼ばれる gate/up -> SwiGLU -> down。

    ``switch_mlp`` 以外は `_moe_combine_fold` が持っている並べ替えの材料:

      ``switch_mlp`` : SwitchGLU (3 射影は QuantizedSwitchLinear)
      ``x_rows`` : gather **前**の x (rows, 1, K)
      ``row_src``: 行 s が読む x の行 (M,) = `order // top_k`
      ``idx_s``  : ソート済みの専門家添字 (M,)
      ``order``  : 行 s の出力位置 (M,)。`argsort(idx.flatten())` そのもの
      ``w_flat`` : ルータ重み (M,) fp32、**トークン順** (t*top_k + k の並び。
                   ソート順ではない -- combine カーネルはこの並びで読む)

    戻り値は **(rows, hidden)** (top_k の和まで取った最終形)。
    条件が揃わなければ ``None`` を返し、呼び手は素の経路に落ちる。
    判定は `dispatched` (P3) と同じ内容 -- あちらが素の `mx.gather_qmm` に
    落ちる形でここだけ自前カーネルに入る、という食い違いを避けるため。
    """
    import mlx.core as mx

    from .kernels import moe_grouped_gemm as mgg

    if not _MOE_DOWN_EPI_ON or _MOE_GEMM_MODE != "seg":
        return None
    if mx.default_device() != mx.gpu:
        return None
    if x_rows.ndim != 3 or x_rows.shape[1] != 1:
        return None
    M = order.shape[0]
    if M < _MOE_GEMM_MIN_ROWS:
        return None
    if idx_s.ndim != 1 or idx_s.shape[0] != M:
        return None
    if row_src.shape != (M,) or w_flat.shape != (M,):
        return None

    rows, _, K = x_rows.shape
    up, gate, down = (switch_mlp.up_proj, switch_mlp.gate_proj,
                      switch_mlp.down_proj)
    parts = []
    for proj in (up, gate, down):
        if "bias" in proj or getattr(proj, "mode", "affine") != "affine":
            return None
        b = proj.get("biases")
        if b is None:
            return None
        parts.append((proj["weight"], proj["scales"], b))
    (w_up, s_up, b_up), (w_gate, s_gate, b_gate), (w_dn, s_dn, b_dn) = parts

    x2 = x_rows.reshape(rows, K)
    if not mgg.segmented_eligible(x2, w_up, s_up, b_up,
                                  up.group_size, up.bits):
        return None
    H = w_up.shape[1]
    # down の適格判定は形と dtype しか見ないので、まだ作っていない SwiGLU
    # 出力 (M, H) の代わりにダミーの 1 行を渡す (MLX は遅延なので、評価
    # されないこの配列に GPU の仕事は発生しない)
    probe = mx.zeros((1, H), dtype=x_rows.dtype)
    if not mgg.segmented_eligible(probe, w_dn, s_dn, b_dn,
                                  down.group_size, down.bits):
        return None

    top_k = M // rows
    if _MOE_FOLD_MODE == "combine":
        from .kernels import moe_combine as mc

        # 形の判定だけ先に済ませる (遅延グラフを組んでから捨てない)
        if rows * top_k != M or w_dn.shape[1] % mc.VEC != 0:
            return None

    tables = _moe_gemm_tables(idx_s, w_up.shape[0])
    kw = dict(tables=tables, bm=_MOE_GEMM_BM, wm=_MOE_GEMM_WM,
              mix_threshold=_MOE_GEMM_MIX, group_size=up.group_size,
              bits=up.bits)
    if _MOE_GATHER_FOLD_ON:
        src = row_src
    else:
        src = None
        x2 = x_rows.reshape(rows, K)[row_src]
    x_up = mgg.qmm_segmented(x2, w_up, s_up, b_up, None, row_src=src, **kw)
    x_gate = mgg.qmm_segmented(x2, w_gate, s_gate, b_gate, None,
                               row_src=src, **kw)
    act = switch_mlp.activation(x_up, x_gate)
    kw_dn = dict(kw, group_size=down.group_size, bits=down.bits)
    if _MOE_FOLD_MODE == "epi":
        # down の store でルータ重みを掛けて出力位置へ直接書く。残るのは
        # top_k 軸の和だけ。store は行 s を order[s] へ書くので、重みも
        # ソート順 (行 s の重み) で渡す
        scattered = mgg.qmm_segmented(
            act, w_dn, s_dn, b_dn, None, row_dst=order,
            row_scale=w_flat[order], **kw_dn)
        return mx.unflatten(scattered, 0, (rows, top_k)).sum(axis=-2)
    # combine: down は現行のまま (ソート順に store)、その後ろの
    # 「unsort + 重み掛け + 和」を 1 カーネルに畳む。カーネルは
    # (t, k) の並びで読むので、重みは**トークン順**の w_flat をそのまま渡す
    out_sorted = mgg.qmm_segmented(act, w_dn, s_dn, b_dn, None, **kw_dn)
    return mc.combine(out_sorted, _inv_perm(order), w_flat, rows, top_k)


def _inv_perm(order):
    """置換 `order` の逆置換。`mx.argsort(order)` と同じ値を scatter で作る
    (16384 要素で argsort の ~0.17 ms に対して 64KB の書き込み 1 回)。"""
    import mlx.core as mx

    n = order.shape[0]
    idx = mx.arange(n, dtype=mx.uint32)
    return mx.zeros((n,), dtype=mx.uint32).at[order].add(idx)


def enable_moe_down_epilogue(model=None, mode: str | None = None,
                             gather_fold: bool | None = None) -> int:
    """P7 第 2 段。MoE の combine (unsort + ルータ重み掛け + 和) を畳む。

    ``mode`` は 2 通り (既定は ``"combine"``):

      - ``"combine"``: down は現行のまま (ソート順に store) で、その後ろの
        「unsort の gather + ルータ重み + top_k の和」を 1 カーネルに畳む
        (`kernels/moe_combine.py`)。ルータ重みを down の**後**で掛けるので、
        `(act*w)` の実体化 (126MB 往復) も消える
      - ``"epi"``: down の segmented GEMM の store でルータ重みを掛けて
        出力位置へ直接書く (scatter)。top_k の和だけ後に残る

    どちらも `enable_moe_grouped_gemm` (mode="seg") が有効なときだけ発火する
    (フックが gate/up/down を自前カーネルで通すため)。行数ゲートも P3 と
    同じ (`MLXTURBO_MOE_GEMM_MIN_ROWS`、既定 1024) なので decode/verify 幅は
    1 op も変わらない。

    **ビット一致しない。**丸めが 1 回減る (現行は `(act * w)` を bf16 に
    丸めてから down GEMM、こちらは fp32 の累算器に w を掛けて 1 回だけ
    丸める) ので、素より exact に近い側にずれる。

    差し替えは vendor の `_moe_combine_fold` が読むフック 1 個だけ
    (受け持つのは gate/up -> SwiGLU -> down の 3 本)。A/B で交互に測れるよう、
    `disable_moe_down_epilogue` はフラグだけ下ろす (C1 / P3 と同じ作法)。

    ``gather_fold`` は x の gather も gate/up GEMM の行の読み方に畳むか
    (`row_src`)。**既定 on** (`MLXTURBO_MOE_GATHER_FOLD=0` で off)。in-model
    8k では combine だけの -1.7% に対して、これを足すと -3.8%。

    **既定 on** (2026-09-03、8k prefill -3.8%)。`MLXTURBO_MOE_DOWN_EPI` で
    `combine` (既定) / `epi` / `off` を選ぶ。引数で渡した値のほうが強い
    (A/B の knob は必ず明示で渡すので、env の設定に影響されない)。

    戻り値は「フックに届きうる MoE 層の数」(``model`` を渡したときだけ。
    渡さなければ 0)。off のときは必ず 0。
    """
    global _MOE_DOWN_EPI_ON, _MOE_GATHER_FOLD_ON, _MOE_FOLD_MODE
    import os

    if mode is None:
        mode = (os.environ.get("MLXTURBO_MOE_DOWN_EPI", "combine").strip()
                .lower() or "combine")
    if mode in ("off", "0", "none", "false", "no"):
        disable_moe_down_epilogue()
        return 0
    if mode not in ("combine", "epi"):
        raise ValueError(f"mode={mode!r} は combine / epi / off のどれか")
    _MOE_FOLD_MODE = mode
    if gather_fold is None:
        gather_fold = os.environ.get(
            "MLXTURBO_MOE_GATHER_FOLD", "1") not in ("0", "")

    # 実際に読み込まれているのは `mlx_lm.models.qwen4_exp` (中身は
    # `_vendor/qwen4_exp.py`、`_arch_registry` の finder が差している)。
    # `mlxturbo._vendor.qwen4_exp` として import すると同じファイルの
    # **別のモジュール実体**になり、フックが本番の側に立たない
    import mlx_lm.models.qwen4_exp as Q

    Q._MOE_DOWN_EPILOGUE = _moe_fold_block
    _MOE_DOWN_EPI_ON = True
    _MOE_GATHER_FOLD_ON = bool(gather_fold)
    if model is None:
        return 0
    # フックは `_moe_combine_fold` の中にあるので、届くのは combine-fold が
    # 効いている MoE 層だけ (行数ゲートは呼び出し時に見る)
    return sum(
        1 for layer in _model_layers(model)
        if getattr(getattr(layer, "mlp", None), "_combine_fold_min_s", None)
        is not None
    )


def disable_moe_down_epilogue() -> None:
    """`enable_moe_down_epilogue` を打ち消す (フラグを下ろすだけ)。"""
    global _MOE_DOWN_EPI_ON
    _MOE_DOWN_EPI_ON = False


# --- P10: prefill 幅の dense 射影を BM=64 の自前 qmm へ -------------------
#
# MLX の dense qmm_t は BM=32 (quantized.cpp:1058-1065) なので、W タイル 1 枚の
# 逆量子化を 32 行ごとに払い直す。BM=64 のタイル
# (`kernels/qmm_wide.py` の `m64n32k32w2x2r8`) は同じ W タイルで 64 行を養う
# ので逆量子化が半分になる。micro では素の 0.935〜0.947 (M=2048 / 8192)、
# **ビット一致** (K の縮約順はタイル形を変えても動かない -- qmm_wide.py の
# 「数値」の節)。
#
# 差し替えは `nn.QuantizedLinear.__call__` に 1 個だけ。どの射影を通すかは
# `enable_qmm_wide` が層を歩いてモジュール属性 `_qmm_wide` (タイル) を
# 置くかどうかで決める。3 つの呼び手 (`batch.py` / `batch_spec.py` /
# `spec_flash.py`) はどれもモジュール呼び出し (`self.q_proj(x)` など) で
# 射影に入る -- `spec_flash._staged_forward` / `_group_prefill_forward` の
# 写しも `self.out_proj(...)` を呼ぶだけで、別口の qmm は持たない
# (`MLXTURBO_WIDE` の連結射影は別の口で、既定 off のまま触らない)。
_QMM_WIDE_STOCK = None        # 素の nn.QuantizedLinear.__call__
_QMM_WIDE_ON = False          # A/B で交互に切り替えるのはこれだけ
_QMM_WIDE_MIN_ROWS = 1024     # decode/verify 幅は BM=64 が行を無駄にする
# 本番で当たる dense 射影 (K -> N): attention の q_proj 2560->12288 と
# o_proj 6144->2560、GDN の in_proj_qkv 2560->10240 / in_proj_z 2560->6144 /
# out_proj 6144->2560。k_proj / v_proj (N=512) は micro で測っていないので
# 入れない。
#
# `mlp` の 3 本は Qwen3.8-27B (qwen3_5) の dense MLP
# (gate/up 5120->17408、down 17408->5120、64 層で 192 射影)。qwen4_exp の
# `layer.mlp` は `SparseMoeBlock` で、専門家は `mlp.switch_mlp` (SwitchGLU の
# `QuantizedSwitchLinear`)、共有専門家は `mlp.shared_expert` の下にあり、
# **`mlp` 直下には gate_proj/up_proj/down_proj が無い**ので当たらない
# (`bench/test_fusions_other_family.py` で確認)。形の適格判定はどの族でも
# `qmm_wide.eligible()` に任せる。
_QMM_WIDE_TARGETS = (
    ("self_attn", ("q_proj", "o_proj")),
    ("linear_attn", ("in_proj_qkv", "in_proj_z", "out_proj")),
    ("mlp", ("gate_proj", "up_proj", "down_proj")),
)


def _qmm_wide_dispatch(self, x):
    """`nn.QuantizedLinear.__call__` の身代わり。素通しの判定を先に済ませる。"""
    tile = getattr(self, "_qmm_wide", None)
    if _QMM_WIDE_ON and tile is not None:
        from .kernels import qmm_wide as qw

        if x.ndim == 2:
            if x.shape[0] >= _QMM_WIDE_MIN_ROWS:
                return qw.qmm_wide(
                    x, self["weight"], self["scales"], self["biases"],
                    tile=tile, group_size=self.group_size, bits=self.bits)
        elif x.ndim == 3 and x.shape[0] * x.shape[1] >= _QMM_WIDE_MIN_ROWS:
            B, S, K = x.shape
            out = qw.qmm_wide(
                x.reshape(B * S, K), self["weight"], self["scales"],
                self["biases"], tile=tile, group_size=self.group_size,
                bits=self.bits)
            return out.reshape(B, S, -1)
    return _QMM_WIDE_STOCK(self, x)


def enable_qmm_wide(model, mtp=None, mode: str | None = None,
                    tile_name: str = "m64n32k32w2x2r8",
                    min_rows: int | None = None) -> int:
    """prefill 幅の dense 射影を `kernels/qmm_wide.qmm_wide` に通す (P10)。

    戻り値は印を付けた射影の数 (層数 x 種類)。0 なら 1 つも発火しない。

    ``mode``: ``"auto"`` (非 NAX 機だけ on) / ``"on"`` / ``"off"``。``None``
    なら環境変数 `MLXTURBO_QMM_WIDE` を読む (**既定 auto** = 非 NAX で on)。NAX 機の判定は
    `moe_grouped_gemm.is_nax_device` と同じもの -- あちらは MLX 側が別
    カーネルを持っていて A/B 未実施なので、自前カーネルは当てない。

    行数 (2 次元なら M、3 次元なら B*S) が ``min_rows``
    (`MLXTURBO_QMM_WIDE_MIN_ROWS`、既定 1024) 未満なら素の
    `mx.quantized_matmul` に落ちる。prefill のチャンク幅 (既定 2048) は超え、
    decode/verify 幅 (数行) は必ず落ちる。

    形の適格判定 (`qmm_wide.eligible` と同じ内容) はここで 1 回だけ済ませ、
    通った射影にだけタイルを属性で置く。ホットパスは属性 1 つと行数の比較
    だけ。**素とビット一致する**ので `--knob qmm-wide` は
    `control_identical=True` で回せる。
    """
    global _QMM_WIDE_STOCK, _QMM_WIDE_ON, _QMM_WIDE_MIN_ROWS
    import os

    import mlx.core as mx
    import mlx.nn as nn

    from .kernels import moe_grouped_gemm as mgg
    from .kernels import qmm_wide as qw

    if mode is None:
        mode = os.environ.get("MLXTURBO_QMM_WIDE", "auto").strip().lower() or "auto"
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"mode={mode!r} は auto / on / off のどれかにすること")
    if mode == "off" or (mode == "auto" and mgg.is_nax_device()):
        _QMM_WIDE_ON = False
        return 0
    if mx.default_device() != mx.gpu:
        _QMM_WIDE_ON = False
        return 0

    tile = qw.TILES.get(tile_name)
    if tile is None:
        raise ValueError(f"tile={tile_name!r} は qmm_wide.TILES に無い"
                         f" (候補: {', '.join(sorted(qw.TILES))})")
    if min_rows is None:
        min_rows = int(os.environ.get("MLXTURBO_QMM_WIDE_MIN_ROWS", "1024"))
    _QMM_WIDE_MIN_ROWS = int(min_rows)

    if _QMM_WIDE_STOCK is None:
        _QMM_WIDE_STOCK = nn.QuantizedLinear.__call__
        nn.QuantizedLinear.__call__ = _qmm_wide_dispatch

    def each_layer():
        for layer in _model_layers(model):
            yield layer
        if mtp is not None:
            for layer in mtp.layers:
                yield layer

    n = 0
    for layer in each_layer():
        for holder, names in _QMM_WIDE_TARGETS:
            mod = getattr(layer, holder, None)
            if mod is None:
                continue
            for name in names:
                lin = getattr(mod, name, None)
                if lin is None or not isinstance(lin, nn.QuantizedLinear):
                    continue
                if "bias" in lin or getattr(lin, "mode", "affine") != "affine":
                    continue
                w, scales = lin["weight"], lin["scales"]
                biases = lin.get("biases")
                if biases is None:
                    continue
                # `eligible` は x の dtype / K だけを見るので、形を合わせた
                # 空の probe で判定できる (実際の x は (行, K) の bf16)
                x_probe = mx.zeros(
                    (1, w.shape[1] * 32 // lin.bits), dtype=scales.dtype)
                if not qw.eligible(x_probe, w, scales, biases, tile,
                                   lin.group_size, lin.bits):
                    continue
                lin._qmm_wide = tile
                n += 1
    _QMM_WIDE_ON = n > 0
    return n


# HC (hyper-connection) の読み側の細長い 2 本。`_QMM_WIDE_TARGETS` と別に
# 持つのは、あちらが「本番の既定に入っている dense 射影」の集合で、こちらは
# 2026-09-04 に prefill の内訳から出てきた別の的だから (`enable_hc_qmm_wide`)。
_HC_QMM_WIDE_HOLDERS = ("attn_hyper_connection", "mlp_hyper_connection")
_HC_QMM_WIDE_NAMES = ("input_mix_weight_down", "input_mix_weight_up")


def _hc_gated_residuals(model, mtp=None):
    """`GatedResidual` を 97 個 (48 層 x 2 + mixer) 全部たどる。"""
    holders = list(_HC_QMM_WIDE_HOLDERS)
    layers = _model_layers(model)
    if mtp is not None:
        layers += list(mtp.layers)
    for layer in layers:
        for holder in holders:
            mod = getattr(layer, holder, None)
            if mod is not None:
                yield mod
    body = _model_body(model)
    mixer = getattr(body, "hyper_connection_mixer", None) if body is not None else None
    if mixer is not None:
        yield mixer


def enable_hc_qmm_wide(model, mtp=None, mode: str | None = None,
                       tile_name: str = "m64n32k32w2x2r8") -> int:
    """HC の `input_mix_weight_down` / `_up` を prefill 幅で qmm_wide に通す。

    **2026-09-04 から既定 on。**`enable_qmm_wide` (段 P10) と同じ差し替え
    (`nn.QuantizedLinear.__call__`) と同じ行数ゲート (`_QMM_WIDE_MIN_ROWS`、
    既定 1024) に相乗りし、印を付ける射影を HC の細長い 2 本
    (10240->320 と 320->10240、4bit/gs64) に広げるだけ。

    ``mode``: ``"auto"`` (既定、`enable_qmm_wide` の判定に従う = `_QMM_WIDE_ON`
    が立っているときだけ効く。NAX 機や `MLXTURBO_QMM_WIDE=off` では何も起きない)
    / ``"1"`` (印を付ける) / ``"0"`` (何もしない)。``None`` なら環境変数
    `MLXTURBO_HC_QMM_WIDE` を読む。**`=0` が逃げ道。**

    数値は素とビット一致する (`qmm_wide` の K の縮約順はタイル形に依らない)。
    冷の連鎖 micro (M=2048、97 組 367.5 MB 巡回、
    `bench/results/hc-prefill-micro.json`) で down 0.820 / up 0.865、
    in-model (`--knob hc-qmm-wide`、`bench/results/hc-prefill-fast-c-*.json`)
    は 17k prefill -0.9% (3 本とも負) / 8k -0.4% / 4k +0.1% (揺れの中)、
    tok/round 同一、decode ±0 -- 代金ゼロなので取り分が小さくても既定に入れた
    (CLAUDE.md の「代金ゼロの改善は 1% 未満でも既定に入れる」)。

    戻り値は印を付けた射影の数 (本番のパックなら 97 x 2 = 194)。
    """
    import os

    import mlx.core as mx
    import mlx.nn as nn

    from .kernels import qmm_wide as qw

    global _QMM_WIDE_STOCK

    if mode is None:
        mode = os.environ.get("MLXTURBO_HC_QMM_WIDE", "auto").strip().lower()
    mode = mode or "auto"
    if mode in ("0", "off", "false"):
        return 0
    if mode not in ("auto", "1", "on", "true"):
        raise ValueError(f"mode={mode!r} は auto / 1 / 0 のどれかにすること")
    if mx.default_device() != mx.gpu:
        return 0
    if mode == "auto":
        # `enable_qmm_wide` とまったく同じ門を自前で通す。あちらは runner の
        # 後ろで呼ばれるので `_QMM_WIDE_ON` をここでは当てにできない
        from .kernels import moe_grouped_gemm as mgg

        wide_mode = (os.environ.get("MLXTURBO_QMM_WIDE", "auto").strip().lower()
                     or "auto")
        if wide_mode == "off" or (wide_mode == "auto" and mgg.is_nax_device()):
            return 0
    tile = qw.TILES.get(tile_name)
    if tile is None:
        raise ValueError(f"tile={tile_name!r} は qmm_wide.TILES に無い")
    if _QMM_WIDE_STOCK is None:
        _QMM_WIDE_STOCK = nn.QuantizedLinear.__call__
        nn.QuantizedLinear.__call__ = _qmm_wide_dispatch

    n = 0
    for gr in _hc_gated_residuals(model, mtp):
        for name in _HC_QMM_WIDE_NAMES:
            lin = getattr(gr, name, None)
            if lin is None or not isinstance(lin, nn.QuantizedLinear):
                continue
            if "bias" in lin or getattr(lin, "mode", "affine") != "affine":
                continue
            w, scales = lin["weight"], lin["scales"]
            biases = lin.get("biases")
            if biases is None:
                continue
            x_probe = mx.zeros(
                (1, w.shape[1] * 32 // lin.bits), dtype=scales.dtype)
            if not qw.eligible(x_probe, w, scales, biases, tile,
                               lin.group_size, lin.bits):
                continue
            lin._qmm_wide = tile
            n += 1
    return n


def disable_hc_qmm_wide(model, mtp=None) -> None:
    """`enable_hc_qmm_wide` の印だけを外す (dense 射影の印は触らない)。"""
    for gr in _hc_gated_residuals(model, mtp):
        for name in _HC_QMM_WIDE_NAMES:
            lin = getattr(gr, name, None)
            if lin is not None:
                lin._qmm_wide = None


def disable_qmm_wide() -> None:
    """`enable_qmm_wide` を打ち消す (フラグを下ろすだけ)。

    差し替えと属性は残るが `_qmm_wide_dispatch` が素通しに固定される。A/B で
    交互に測るためにこの形にしてある (`disable_moe_grouped_gemm` と同じ理屈)。
    """
    global _QMM_WIDE_ON
    _QMM_WIDE_ON = False


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
    for layer in _model_layers(model):
        sa = getattr(layer, "self_attn", None)
        if sa is not None and hasattr(sa, "q_norm"):
            sa._fast_rope = True
            n += 1
    return n


def disable_fast_rope(model) -> int:
    """`enable_fast_rope` を打ち消す。戻り値は外した数。A/B で交互に測るために要る。"""
    n = 0
    for layer in _model_layers(model):
        sa = getattr(layer, "self_attn", None)
        if sa is not None and getattr(sa, "_fast_rope", False):
            sa._fast_rope = False
            n += 1
    return n


def enable_ple_hoist(model) -> int:
    """全 PLE 層の n-gram 埋め込みを層ループの前にまとめて計算する
    (``Qwen4ExpModel._prelude`` / ``_hoist_ple`` の ``_ple_hoist`` 分岐、
    `mlxturbo/_vendor/qwen4_exp.py`)。

    PLE の入力は隠れ状態 h に依らず ids と直前文脈だけで決まるので、48 層の
    ループへ入る前に全部計算できる。素の経路は PLE 層 (Flash-Next で 5 層)
    それぞれの ``ple_embedding(ids, prev_ctx)`` 呼び出しが n-gram テーブル側の
    GPU->CPU 同期 (``mlxturbo/ngram_stream.py`` の ``StreamNGram.__call__`` の
    ``np.array(gid.reshape(-1))``) を挟み、それが `_staged_forward` の 2 層
    ごとの async_eval 投入をその境界で断ち切っていた。まとめて計算すれば、
    サイドカーが全 PLE 層で共有されている構成 (``ngram_stream.install()`` 後の
    ``StreamNGram``/``RamNGram``) では同期は 1 forward で 1 回になる。

    実験的な分岐なので、この関数を呼ぶだけでは何も起きない --- 環境変数
    `MLXTURBO_PLE_HOIST=1` が立っているときだけ ``_ple_hoist`` を立てる
    (`enable_fast_rope` と同じ作法)。PLE 層を持たないモデルでは何もしない。

    戻り値は対象になった PLE 層数 (0 なら対象外)。
    """
    import os

    if os.environ.get("MLXTURBO_PLE_HOIST") != "1":
        return 0
    m = getattr(model, "model", model)
    n = len(getattr(m, "ple_layers", None) or [])
    if n:
        m._ple_hoist = True
    return n


def disable_ple_hoist(model) -> int:
    """`enable_ple_hoist` を打ち消す。戻り値は 1 (外した) / 0 (元々 off)。
    A/B で交互に測るために要る。"""
    m = getattr(model, "model", model)
    if getattr(m, "_ple_hoist", False):
        m._ple_hoist = False
        return 1
    return 0


__all__ = [
    "enable_gather_sort",
    "enable_gdn_blocked_kernel",
    "enable_gdn_metal_kernel",
    "enable_gdn_decode_fused",
    "enable_gdn_prework_kernel",
    "enable_gdn_port",
    "disable_gdn_port",
    "enable_moe_glu",
    "enable_moe_shared_fold",
    "enable_moe_verify_gather",
    "enable_wide_projections",
    "disable_gdn_blocked_kernel",
    "disable_gdn_metal_kernel",
    "disable_gdn_decode_fused",
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
    "enable_hyper_connection_prefill_compiled",
    "disable_hyper_connection_prefill_compiled",
    "enable_moe_route",
    "enable_moe_route_nofuse",
    "enable_rms_norm_gated",
    "enable_rms_norm_gated_nofuse",
    "enable_fast_rope",
    "disable_fast_rope",
    "enable_sdpa_split",
    "disable_sdpa_split",
    "enable_ple_hoist",
    "disable_ple_hoist",
]
