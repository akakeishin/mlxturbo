"""GPU が要る `eligible()` 分岐だけを狙う一次検査 (B-6 の穴埋め)。

`tools/vendor_fingerprint.py` は CPU 専用で価値がある (合成モデル・数秒)
一方、2026-09-02 に足した融合カーネル群の `eligible()` はどれも
``mx.default_device() == mx.gpu`` を要求するので、CPU の検査は**それらの
GPU 分岐を一度も通らない**。ここは同じ「合成モデル」という性質を保った
まま GPU に置き、次の knob それぞれについて

  1. 発火したこと (`kernels._fire.snapshot()` が増える、または同等の
     呼び出しカウンタが増える)
  2. 出力が knob off (素の経路) と一致すること

を確かめる。**発火 0 のまま「一致」を出さないこと**が主目的 --- 2026-09-02
に `gdn_prework` が投機の検証フォワードに届かず、発火 0 のまま「効果なし」
と誤判定して数時間を失った前例があるので、ここでは発火 0 を無条件で不合格
として扱う。

対象 (どれも既定 off、on にして検査する):

    hc-write          mlxturbo.fused.enable_hc_write (DecoderLayer._combine)
    gdn-prework        mlxturbo/kernels/gdn_prework.py (model() 直叩き)
    gdn-prework(capture) 同上だが spec_flash.capture()+_staged_forward 経由
                       (投機の検証フォワードが実際に通る経路。実機 GDN 形
                       n_k=16 の境界も使う)
    prefill-attn       mlxturbo/kernels/prefill_attn.py
    gdn-blocked        mlxturbo/kernels/gated_delta_blocked.py
    moe-verify         mlxturbo/kernels/moe_verify_gather.py
    rms-norm-gated     mlxturbo.fused.enable_rms_norm_gated (env ゲート無し)
    moe-route          mlxturbo.fused.enable_moe_route (env ゲート無し)

## dtype の選び方

`gdn-prework` と `moe-verify` の `eligible()` は bf16/fp16 を要求する
(fp32 を弾く) のでそれぞれ bf16 で組む。`hc-write` / `prefill-attn` /
`gdn-blocked` は fp32 でも `eligible()` を通るので、bf16 の 1 ulp 差が
8 層の合成モデルで増幅されるのを避けるため fp32 で組む
(`tools/verify_prefill_attn.py` のモデルレベル検査も同じ理由で fp32)。
bf16 を要求する 2 つは、bf16 特有の丸め (docstring に実測込みで記載済み)
がそのまま乗るので、しきい値をその分だけ緩めてある。

## 正しさ判定について

ここで見るのは「壊れていないこと」の一次検査であって、
`tools/verify_prefill_attn.py` / `tools/verify_gdn_blocked.py` /
`moe_verify_gather._verify_v2` が持つ厳密な精度検証の代わりではない。
しきい値はそれぞれのカーネルの docstring に書かれている精度の性質
(ビット一致するもの/しないもの) に合わせて緩め・厳しめを使い分けてある。

## 使い方

GPU を使うので必ず biglock 経由で。合成モデルで 100GB もメモリも要らない
ので `MLXTURBO_MIN_FREE_GB` を小さく上書きしてよい:

    MLXTURBO_MIN_FREE_GB=4 tools/biglock.sh .venv/bin/python tools/gpu_fingerprint.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# `_gather_max_ratio` (mlxturbo/_vendor/qwen4_exp.py) はこの env を **import 時**
# に一度だけ読む。合成モデルは offset=0 直後の 1 チャンクで比を測ると
# rows(=S) が kv_len と同じ桁になり、既定の比 (実測 0.1-0.2) では
# 「集める価値が無い」と常に判定されて gather 自体に入らない
# (offset を先に進めてから大きい 1 チャンクを流す本検査の作り方だとこの
# 比が支配的になる)。ここは正しさではなく「割に合うか」のヒューリスティクス
# なので、検査用に緩めて確実に候補に入れる。
os.environ.setdefault("MLXTURBO_GATHER_MAX_RATIO", "100")
os.environ.setdefault("MLXTURBO_GDN_PREWORK", "1")
os.environ.setdefault("MLXTURBO_GDN_BLOCKED", "1")
os.environ.setdefault("MLXTURBO_MOE_VERIFY", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mlx.core as mx  # noqa: E402

import mlxturbo  # noqa: E402,F401
from mlxturbo import fused, gather_attn  # noqa: E402
from mlxturbo.kernels import _fire  # noqa: E402
from verify_batch_cache import TINY, build  # noqa: E402


def _cast_bf16_except(model, keep_fp32_suffixes):
    """モデルの fp32 パラメータを bf16 に落とす。ただし ``keep_fp32_suffixes``
    に一致するもの (末尾がそれで終わるパス) はそのまま fp32 に残す。

    `gdn_prework.eligible` は A_log/dt_bias が fp32 であることを要求する
    (`mlxturbo/kernels/gdn_prework.py` の ``dtype_alog`` 分岐)。実モデルの
    チェックポイントも A_log/dt_bias は fp32 のまま保存されている --- 丸ごと
    bf16 に落とすと検査そのものが `eligible=False` で空振りする。
    """
    from mlx.utils import tree_map_with_path

    def cast(path, a):
        if a.dtype != mx.float32:
            return a
        if any(path.endswith(suf) for suf in keep_fp32_suffixes):
            return a
        return a.astype(mx.bfloat16)

    model.update(tree_map_with_path(cast, model.parameters()))
    mx.eval(model.parameters())
    return model


def _diff(a: mx.array, b: mx.array) -> tuple[float, float]:
    a32 = a.astype(mx.float32)
    b32 = b.astype(mx.float32)
    d = float(mx.max(mx.abs(a32 - b32)))
    scale = float(mx.max(mx.abs(a32)))
    rel = d / scale if scale > 0 else d
    return d, rel


def check_hc_write() -> dict:
    """`DecoderLayer._combine` の mx.compile 融合。fp32、素の式とビット一致
    するはず (`mlxturbo/fused.py` の `enable_hc_write` docstring)。

    `_build_combine` は kernels/ の外 (fused.py) にあり、そちらは今回の
    作業範囲外 (触ってよいのは kernels/*.py と本ファイルだけ) なので、
    発火の数え上げはこのスクリプト内だけのモンキーパッチで行う
    (`tools/verify_prefill_attn.py` が `PA.prefill_attn` を数えるのと同じ
    やり方 --- ファイルは書き換えず、実行時にオブジェクトを差し替えるだけ)。
    """
    model = build(8)  # fp32、無改造
    ids = [(i * 7 + 3) % TINY["vocab_size"] for i in range(40)]
    cache = model.make_cache()
    base = model(mx.array(ids)[None], cache=cache)
    mx.eval(base)

    orig_build_combine = fused._build_combine
    fired = [0]

    def counting_build_combine(hc, d):
        fn = orig_build_combine(hc, d)

        def counted(*a, **kw):
            fired[0] += 1
            return fn(*a, **kw)

        return counted

    fused._build_combine = counting_build_combine
    fused.enable_hc_write()
    try:
        cache2 = model.make_cache()
        on = model(mx.array(ids)[None], cache=cache2)
        mx.eval(on)
    finally:
        fused.disable_hc_write()
        fused._build_combine = orig_build_combine

    diff, rel = _diff(base, on)
    # docstring 通りならビット一致 (diff==0.0) のはず
    ok = fired[0] > 0 and diff == 0.0
    return {"name": "hc-write", "fired": fired[0], "diff": diff, "rel": rel, "ok": ok,
            "note": "fp32、ビット一致を要求"}


def check_gdn_prework() -> dict:
    """`gdn_prework.fused_gdn_prework`。bf16 必須 (`eligible` の dtype 判定)。

    decode 幅の呼び出し (S=1) を複数回重ね、cache (conv 状態・GDN 再帰状態)
    経由で誤差が伝播することも込みで見る。bf16 の sigmoid が参照とビット
    一致しない (docstring 実測: beta で最大 diff 0.0039) ので、その分と
    cache 経由の伝播を見込んで緩めのしきい値にしてある。
    """
    # 現行カーネルは A_log/dt_bias も入力と同じ bf16 で受け、素の
    # compute_g と同じ丸め位置を保つ。TINY の n_k=2 は列ブロック境界を
    # 満たさないので、直接呼出し側も実機 GDN 形を使う。
    model = _build_real_gdn_shape(8)
    _cast_bf16_except(model, ())

    ids = [(i * 7 + 3) % TINY["vocab_size"] for i in range(20)]

    cache = model.make_cache()
    model(mx.array(ids)[None], cache=cache)
    base_logits = []
    for tok in range(5):
        logits = model(mx.array([[tok]]), cache=cache)
        mx.eval(logits)
        base_logits.append(logits)

    fused.enable_gdn_prework_kernel(model)
    _fire.reset()
    try:
        cache2 = model.make_cache()
        model(mx.array(ids)[None], cache=cache2)
        on_logits = []
        for tok in range(5):
            logits = model(mx.array([[tok]]), cache=cache2)
            mx.eval(logits)
            on_logits.append(logits)
        fired = _fire.snapshot().get("gdn_prework", 0)
    finally:
        fused.disable_gdn_prework_kernel()

    diff = max(_diff(a, b)[0] for a, b in zip(base_logits, on_logits))
    rel = max(_diff(a, b)[1] for a, b in zip(base_logits, on_logits))
    # bf16 の 1 ulp が decode ステップを重ねるたびに cache 経由で積み増す
    # (実測で 2 桁動くことがある --- 5 ステップの実測最大は 0.16)。
    # ここでは「大崩れしていないこと」だけを見る緩めの絶対値ゲート。
    ok = fired > 0 and diff < 0.5
    return {"name": "gdn-prework", "fired": fired, "diff": diff, "rel": rel, "ok": ok,
            "note": "bf16 decode x5、絶対値ゲート (cache 経由で伝播するため)"}


def _build_real_gdn_shape(budget: int):
    """`build()` と同じ作り方だが、linear_* を実機 Flash-Next の形
    (n_k=16, n_v=48, dk=dv=128, conv_kernel=4, conv_dim=10240) に差し替える。

    `check_gdn_prework` の TINY 形 (n_k=2) は `eligible()` の
    `2*n_k<=32 simdgroup` の境界を一度も踏まない。実機はちょうど
    `2*n_k=32` の境界そのものなので、境界条件を別で確かめる。
    """
    import mlx_lm.models.qwen4_exp as Q
    from mlx.utils import tree_map

    cfg = dict(TINY, indexer_budget=budget)
    cfg.update(
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    mx.random.seed(0)
    model = Q.Model(Q.ModelArgs(model_type="qwen4_exp", text_config=cfg))
    model.update(
        tree_map(
            lambda a: mx.random.normal(a.shape) * 0.05 if a.dtype == mx.float32 else a,
            model.parameters(),
        )
    )
    mx.eval(model.parameters())
    model.eval()
    return model


def check_gdn_prework_capture() -> dict:
    """`gdn_prework` を **`spec_flash.capture()` + `_staged_forward` 経由**
    (投機の検証フォワードが実際に通る経路) で確かめる。

    `check_gdn_prework` は `model(...)` を直接呼ぶだけで、`capture()` が
    `GatedDeltaNet.__call__` ごと差し替える経路を一度も通らない。
    2026-09-01 に実際に起きた事故 (`mlxturbo/kernels/_fire.py` の docstring
    参照: capture() の差し替えで投機の検証フォワードに融合が一度も届かず、
    発火 0 のまま「効果なし」と誤判定した) はこちらの経路でしか再現しない。

    実機の GDN 形 (n_k=16 で `2*n_k<=32` の境界ちょうど) を使い、S=1..3 の
    複数呼び出し (検証フォワードの幅) を通す。

    A_log/dt_bias も実モデルと同じ bf16 にする。現行カーネルは素の
    compute_g と丸め位置を揃えるためfp32の写しを使わない。modelを渡す
    enableは、旧版の写しが残っていれば削除してから本番と同じ経路を立てる。
    **発火 0 のまま「一致」を出さない** (モジュールdocstringの方針どおり)。
    """
    from mlxturbo import spec_flash

    model = _build_real_gdn_shape(8)
    _cast_bf16_except(model, ())

    gdn0 = model.model.layers[2].linear_attn
    shape_ok = (gdn0.n_k, gdn0.n_v, gdn0.dk, gdn0.conv_dim) == (16, 48, 128, 10240)
    want_alog_dtype = mx.bfloat16
    alog_ok = gdn0.A_log.dtype == want_alog_dtype and gdn0.dt_bias.dtype == want_alog_dtype

    ids = [(i * 7 + 3) % TINY["vocab_size"] for i in range(20)]

    def run(use_capture: bool):
        cache = model.make_cache()
        model(mx.array(ids)[None], cache=cache)
        logits = []
        widths = (1, 2, 3, 1, 2)
        if use_capture:
            with spec_flash.capture(model):
                for i, s in enumerate(widths):
                    toks = [[(i * 3 + j) % TINY["vocab_size"] for j in range(s)]]
                    lg = spec_flash._staged_forward(model, mx.array(toks), cache)
                    mx.eval(lg)
                    logits.append(lg[:, -1])
        else:
            for i, s in enumerate(widths):
                toks = [[(i * 3 + j) % TINY["vocab_size"] for j in range(s)]]
                lg = model(mx.array(toks), cache=cache)
                mx.eval(lg)
                logits.append(lg[:, -1])
        return logits

    base_logits = run(False)

    fused.enable_gdn_prework_kernel(model)
    _fire.reset()
    try:
        on_logits = run(True)
        fired = _fire.snapshot().get("gdn_prework", 0)
    finally:
        fused.disable_gdn_prework_kernel()

    diff = max(_diff(a, b)[0] for a, b in zip(base_logits, on_logits))
    rel = max(_diff(a, b)[1] for a, b in zip(base_logits, on_logits))
    # check_gdn_prework と同じ緩めの絶対値ゲート (bf16 sigmoid が参照とビット
    # 一致しない分 + cache 経由の伝播)。
    ok = shape_ok and alog_ok and fired > 0 and diff < 0.5
    name = "gdn-prework(capture)"
    note = ("実機 GDN 形 (n_k=16 境界) x capture()+_staged_forward、S=1..3、"
            "A_log/dt_bias も bf16 (実機と同じ状態)")
    if not shape_ok:
        note += "  ★形が実機と不一致★"
    if not alog_ok:
        note += f"  ★A_log/dt_bias dtype が想定 ({want_alog_dtype}) と不一致★"
    return {"name": name, "fired": fired, "diff": diff, "rel": rel, "ok": ok, "note": note}


def check_gdn_blocked() -> dict:
    """`gated_delta_blocked.gated_delta_update_blocked`。fp32 で通る
    (`eligible` は q/k/v の dtype を制約しない)。prefill 幅 (T>=64) が必須。
    """
    model = build(8)  # fp32
    ids = [(i * 7 + 3) % TINY["vocab_size"] for i in range(80)]

    cache = model.make_cache()
    base = model(mx.array(ids)[None], cache=cache)
    mx.eval(base)

    fused.enable_gdn_blocked_kernel()
    _fire.reset()
    try:
        cache2 = model.make_cache()
        on = model(mx.array(ids)[None], cache=cache2)
        mx.eval(on)
        fired = _fire.snapshot().get("gdn_blocked", 0)
    finally:
        fused.disable_gdn_blocked_kernel()

    diff, rel = _diff(base, on)
    # fp32 の丸め水準 (加算順が変わるだけ)。tools/verify_prefill_attn.py の
    # TOL_MODEL と同じ 1e-4。
    ok = fired > 0 and diff < 1e-4
    return {"name": "gdn-blocked", "fired": fired, "diff": diff, "rel": rel, "ok": ok,
            "note": "fp32 prefill T=80、fp32 丸め水準を期待"}


def check_prefill_attn() -> dict:
    """`prefill_attn.prefill_attn`。fp32 で通る (`eligible` は fp32 を許す)。

    `Attention._gather_forward` の早期救済 (``offset < compress_ratio - 1``)
    に引っかからないよう、先に数トークン流して offset を進めてから
    S>=MIN_S(=64) の 1 チャンクを流す。
    """
    model = build(8)  # fp32
    head = [1, 2]
    body = [(i * 7 + 3) % TINY["vocab_size"] for i in range(70)]

    cache = model.make_cache()
    model(mx.array(head)[None], cache=cache)
    base = model(mx.array(body)[None], cache=cache)
    mx.eval(base)

    # 本番の8192下限は速度ゲートで、合成モデルの正しさ検査とは無関係。
    # verify_prefill_attn.check_model と同じく、この呼出しの間だけ0にする。
    min_kv_key = "MLXTURBO_PREFILL_ATTN_MIN_KV"
    saved_min_kv = os.environ.get(min_kv_key)
    os.environ[min_kv_key] = "0"
    gather_attn.enable_prefill_attn(model)
    _fire.reset()
    try:
        cache2 = model.make_cache()
        model(mx.array(head)[None], cache=cache2)
        on = model(mx.array(body)[None], cache=cache2)
        mx.eval(on)
        fired = _fire.snapshot().get("prefill_attn", 0)
    finally:
        gather_attn.disable_prefill_attn(model)
        gather_attn.disable_gather_attn(model)
        if saved_min_kv is None:
            os.environ.pop(min_kv_key, None)
        else:
            os.environ[min_kv_key] = saved_min_kv

    diff, rel = _diff(base, on)
    ok = fired > 0 and diff < 1e-4
    return {"name": "prefill-attn", "fired": fired, "diff": diff, "rel": rel, "ok": ok,
            "note": "fp32 prefill S=70 (offset=2 から)、fp32 丸め水準を期待"}


def check_moe_verify() -> dict:
    """`moe_verify_gather` (gate+up 融合 + down)。bf16 + 4bit/gs64 量子化必須
    (`eligible_gate_up` / `eligible_down`)。

    TINY モデルの hidden_size (64) は gate/up の `K % 512 == 0` 前提を
    満たさないので、`moe_verify_gather._verify_v2` と同じやり方で
    `SparseMoeBlock` を実形状 (`Q.TextArgs()` の既定値、K=2560/H=640/E=512/
    top_k=10) の合成量子化重みで組み立てる (実チェックポイントは読まない)。
    """
    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo.kernels import moe_verify_gather as mvg

    mx.random.seed(7)
    args = Q.TextArgs()
    block = Q.SparseMoeBlock(args)
    sw = block.switch_mlp
    sw.gate_proj.weight = sw.gate_proj.weight.astype(mx.bfloat16)
    sw.up_proj.weight = sw.up_proj.weight.astype(mx.bfloat16)
    sw.down_proj.weight = sw.down_proj.weight.astype(mx.bfloat16)
    sw.gate_proj = sw.gate_proj.to_quantized(group_size=mvg.GROUP_SIZE, bits=mvg.BITS)
    sw.up_proj = sw.up_proj.to_quantized(group_size=mvg.GROUP_SIZE, bits=mvg.BITS)
    sw.down_proj = sw.down_proj.to_quantized(group_size=mvg.GROUP_SIZE, bits=mvg.BITS)
    mx.eval(block.parameters())

    x_in = mx.random.normal(shape=(1, 3, args.hidden_size)).astype(mx.bfloat16)
    mx.eval(x_in)

    base = block(x_in)
    mx.eval(base)

    fused.enable_moe_verify_gather()
    _fire.reset()
    try:
        on = block(x_in)
        mx.eval(on)
        snap = _fire.snapshot()
        fired = min(snap.get("moe_verify_gate_up", 0), snap.get("moe_verify_down", 0))
    finally:
        fused.disable_moe_verify_gather()

    diff = mvg._max_rel_err(on, base.astype(mx.float32))
    # `_max_rel_err` は atol=0.05/rtol=0.02 の allclose 型 (moe_verify_gather.py
    # の docstring 通り、bf16 の K 本蓄積誤差の床)。1.0 未満なら通過。
    ok = fired > 0 and diff < 1.0
    return {"name": "moe-verify", "fired": fired, "diff": diff, "rel": diff, "ok": ok,
            "note": "bf16、SparseMoeBlock 実形状 (K=2560/H=640/E=512)、allclose 型誤差"}


def check_rms_norm_gated() -> dict:
    """`fused.enable_rms_norm_gated`。`RMSNormGated.__call__` を無条件で
    差し替えるだけ (env ゲート無し) で、`spec_flash.capture()` が触るのは
    `GatedDeltaNet.__call__`/`PLELayer._short_conv`/`GatedResidual.__call__`
    の 3 つだけなので、この置き換えは capture() の影響を受けない
    (`GatedDeltaNet.__call__` 内から `self.norm(out, z)` として素通しで
    呼ばれるだけ)。capture()+_staged_forward 経由でも発火することを確認する。

    `eligible()` は x/weight が fp16/bf16 であることを要求する
    (`mlxturbo/kernels/rms_norm_gated.py`)。TINY は既定 fp32 で組まれる
    ので bf16 に落としてから使う。
    """
    from mlxturbo import spec_flash

    model = build(8)
    _cast_bf16_except(model, ())
    ids = [(i * 7 + 3) % TINY["vocab_size"] for i in range(20)]

    cache = model.make_cache()
    model(mx.array(ids)[None], cache=cache)
    base = model(mx.array([[1]]), cache=cache)
    mx.eval(base)

    fused.enable_rms_norm_gated()
    _fire.reset()
    try:
        cache2 = model.make_cache()
        model(mx.array(ids)[None], cache=cache2)
        with spec_flash.capture(model):
            on = spec_flash._staged_forward(model, mx.array([[1]]), cache2)
            mx.eval(on)
        fired = _fire.snapshot().get("rms_norm_gated", 0)
    finally:
        fused.disable_rms_norm_gated()

    diff, rel = _diff(base, on)
    ok = fired > 0
    return {"name": "rms-norm-gated", "fired": fired, "diff": diff, "rel": rel, "ok": ok,
            "note": "capture()+_staged_forward 経由、env ゲート無しの無条件差し替え"}


def check_moe_route() -> dict:
    """`fused.enable_moe_route`。`SparseMoeBlock.__call__` を無条件で差し替える
    だけ (env ゲート無し) で、capture() が触る 3 メソッドに含まれないので
    capture() の影響を受けない。capture()+_staged_forward 経由でも発火する
    ことを確認する。
    """
    from mlxturbo import spec_flash

    model = build(8)
    ids = [(i * 7 + 3) % TINY["vocab_size"] for i in range(20)]

    cache = model.make_cache()
    model(mx.array(ids)[None], cache=cache)
    base = model(mx.array([[1]]), cache=cache)
    mx.eval(base)

    fused.enable_moe_route()
    _fire.reset()
    try:
        cache2 = model.make_cache()
        model(mx.array(ids)[None], cache=cache2)
        with spec_flash.capture(model):
            on = spec_flash._staged_forward(model, mx.array([[1]]), cache2)
            mx.eval(on)
        fired = _fire.snapshot().get("moe_route", 0)
    finally:
        fused.disable_moe_route()

    diff, rel = _diff(base, on)
    ok = fired > 0
    return {"name": "moe-route", "fired": fired, "diff": diff, "rel": rel, "ok": ok,
            "note": "capture()+_staged_forward 経由、env ゲート無しの無条件差し替え"}


def main() -> int:
    if not mx.metal.is_available():
        print("Metal が使えないのでこの検査は走らせられない")
        return 1
    mx.set_default_device(mx.gpu)

    checks = [
        check_hc_write,
        check_gdn_prework,
        check_gdn_prework_capture,
        check_prefill_attn,
        check_gdn_blocked,
        check_moe_verify,
        check_rms_norm_gated,
        check_moe_route,
    ]

    results = []
    for fn in checks:
        r = fn()
        results.append(r)
        status = "合格" if r["ok"] else "★不合格★"
        print(
            f"{r['name']:14s} 発火={r['fired']:4d}  "
            f"diff={r['diff']:.4e}  rel={r['rel']:.4e}  {status}"
        )
        print(f"               {r['note']}")
        if r["fired"] == 0:
            print(f"               ★発火 0 --- eligible() のどこかで落ちている★")

    ok = all(r["ok"] for r in results)
    print(f"\n=== 総合判定: {'合格' if ok else '不合格'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
