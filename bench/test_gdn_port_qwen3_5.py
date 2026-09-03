"""GDN の自前部品を qwen3_5 (Qwen3.8-27B) に「構造の契約」で当てる検査。

`mlxturbo/fused.py` の `enable_gdn_port` / `disable_gdn_port` が対象。上の
4 つの `enable_gdn_*` は `_vendor/qwen4_exp.py` の `GatedDeltaNet.__call__`
にあるシームを立てるだけなので、シームを持たない族には届かない。
`enable_gdn_port` は同じ 3 部品 (prefill の GDN Metal 再帰、decode 幅の
前処理、出力 norm) をモジュールの形の契約で当てる。

2 段構成:

- **契約の検査 (CPU、数秒)**: 形の探索、qwen4_exp を対象外にすること、
  当てても素の構造 (パラメータ名・子モジュール) が動かないこと、
  disable で元に戻ること。GPU が無くても走る。
- **一致の検査 (GPU 必須)**: 本物の `qwen3_5.GatedDeltaNet` を小さく組んで、
  素と自前の出力・cache を突き合わせる。decode/verify 幅 (前処理 + 出力
  norm) は**ビット一致**、prefill 幅の Metal 再帰は加算順が違うので
  ビット一致しない (Flash-Next での実績と同じ物差し。相対誤差で見る)。

実行 (GPU を使うので biglock 経由):

    tools/biglock.sh .venv/bin/python -m pytest bench/test_gdn_port_qwen3_5.py -q
    .venv/bin/python bench/test_gdn_port_qwen3_5.py     # 直接 (段 2 の micro 扱い)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlxturbo  # noqa: E402,F401 -- sys.meta_path フック (mlx_lm.models.qwen4_exp)
from mlx_lm.models import qwen3_5 as Q35  # noqa: E402
from mlx_lm.models.cache import ArraysCache  # noqa: E402
from mlxturbo import fused  # noqa: E402
from mlxturbo.kernels import _fire  # noqa: E402

GPU = mx.metal.is_available() and mx.default_device() == mx.gpu

# 合成の形。dk は GDN Metal カーネルの定数 (128) に合わせる。key_dim は
# gdn_prework の BLOCK (128) の境界に乗る必要がある。
N_K, N_V, DK, DV, K = 2, 6, 128, 128, 4
HIDDEN = 256
KEY_DIM = N_K * DK          # 256
VALUE_DIM = N_V * DV        # 768
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 1280


def _args(n_layers: int = 2) -> Q35.TextModelArgs:
    return Q35.TextModelArgs(
        model_type="qwen3_5",
        hidden_size=HIDDEN,
        intermediate_size=512,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        linear_num_value_heads=N_V,
        linear_num_key_heads=N_K,
        linear_key_head_dim=DK,
        linear_value_head_dim=DV,
        linear_conv_kernel_dim=K,
        head_dim=64,
    )


def _make_gdn(dtype=mx.bfloat16) -> Q35.GatedDeltaNet:
    gdn = Q35.GatedDeltaNet(_args())
    gdn.set_dtype(dtype)
    gdn.eval()
    return gdn


class _StubLayer(nn.Module):
    def __init__(self, gdn):
        super().__init__()
        self.linear_attn = gdn


class _StubBody(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers


class _StubLanguageModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = _StubBody(layers)


class _StubModel(nn.Module):
    """`mlx_lm.models.qwen3_5.Model` と同じ形 (`.language_model.model.layers`)。"""

    def __init__(self, gdns):
        super().__init__()
        self.language_model = _StubLanguageModel([_StubLayer(g) for g in gdns])


class _StubNoGdnLayer(nn.Module):
    """`linear_attn` を持たない層 (契約が合わない側)。"""


class _StubNoGdn(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.language_model = _StubLanguageModel([_StubNoGdnLayer() for _ in range(n)])


def _env(**kw):
    """環境変数を一時的に差し替えるコンテキスト。"""

    class _Ctx:
        def __enter__(self):
            self.old = {k: os.environ.get(k) for k in kw}
            for k, v in kw.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        def __exit__(self, *a):
            for k, v in self.old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            return False

    return _Ctx()


# --------------------------------------------------------------- 契約の検査


def test_spec_matches_qwen3_5():
    """qwen3_5 の GatedDeltaNet が契約に合い、数が正しく読める。"""
    spec = fused._gdn_spec(_make_gdn())
    assert spec is not None
    assert (spec.n_k, spec.n_v, spec.dk, spec.dv) == (N_K, N_V, DK, DV)
    assert (spec.key_dim, spec.value_dim, spec.conv_dim) == (KEY_DIM, VALUE_DIM, CONV_DIM)
    assert spec.K == K


def test_spec_matches_qwen4_exp_too():
    """同じ契約が qwen4_exp の GatedDeltaNet にも合う (族に依らない形の判定)。

    当てるかどうかは別の話 (qwen4_exp はシームを持っているので対象外)。
    """
    from verify_batch_cache import build  # tools/ (sys.path に入れてある)

    model = build(8)
    gdns = [l.linear_attn for l in model.model.layers
            if getattr(l, "linear_attn", None) is not None]
    assert gdns, "qwen4_exp の合成モデルに GDN 層が無い"
    spec = fused._gdn_spec(gdns[0])
    assert spec is not None
    assert spec.key_dim == spec.n_k * spec.dk
    assert spec.conv_dim == 2 * spec.key_dim + spec.value_dim


def test_enable_skips_qwen4_exp():
    """qwen4_exp には当てない (シーム経由で既に当たっているため二重にしない)。"""
    from verify_batch_cache import build

    model = build(8)
    before = [type(l.linear_attn) for l in model.model.layers
              if getattr(l, "linear_attn", None) is not None]
    got = fused.enable_gdn_port(model)
    after = [type(l.linear_attn) for l in model.model.layers
             if getattr(l, "linear_attn", None) is not None]
    assert got == {"metal": 0, "prework": 0, "norm": 0, "layers": 0}
    assert before == after
    assert fused.disable_gdn_port(model)["prework"] == 0


def test_enable_no_op_without_contract():
    """契約が合わない (GDN を持たない・層が無い) モデルでは何もしない。"""
    for model in (_StubNoGdn(), None):
        got = fused.enable_gdn_port(model)
        assert got["layers"] == 0 and got["prework"] == 0 and got["metal"] == 0


def test_enable_and_disable_roundtrip():
    """当てると `__class__` が動的サブクラスになり、外すと元に戻る。

    モジュール大域の `gated_delta_update` も元の関数に戻ること。
    """
    gdns = [_make_gdn() for _ in range(3)]
    model = _StubModel(gdns)
    base = type(gdns[0])
    nbase = type(gdns[0].norm)
    orig_update = Q35.gated_delta_update
    with _env(MLXTURBO_GDN_DECODE_FUSED="1", MLXTURBO_GDN_METAL=None):
        got = fused.enable_gdn_port(model)
    try:
        assert got["layers"] == 3
        assert got["prework"] == 3
        assert got["metal"] == 1
        assert Q35.gated_delta_update is not orig_update
        for g in gdns:
            assert type(g) is not base and isinstance(g, base)
            assert type(g)._mlxturbo_gdn_base is base
        # 出力 norm は GPU でしか判定できない (合成入力の突き合わせが要る)
        assert got["norm"] == (3 if GPU else 0)
        if GPU:
            assert type(gdns[0].norm) is not nbase
    finally:
        back = fused.disable_gdn_port(model)
    assert Q35.gated_delta_update is orig_update
    assert back["prework"] == 3 and back["metal"] == 1
    for g in gdns:
        assert type(g) is base
        assert type(g.norm) is nbase
        assert not hasattr(g, "_mlxturbo_gdn_spec")


def test_enable_is_idempotent():
    gdns = [_make_gdn()]
    model = _StubModel(gdns)
    base = type(gdns[0])
    with _env(MLXTURBO_GDN_DECODE_FUSED="1", MLXTURBO_GDN_METAL=None):
        fused.enable_gdn_port(model)
        cls1 = type(gdns[0])
        got2 = fused.enable_gdn_port(model)
    try:
        assert type(gdns[0]) is cls1
        assert got2["layers"] == 1 and got2["prework"] == 1
    finally:
        fused.disable_gdn_port(model)
    assert type(gdns[0]) is base


def test_structure_untouched():
    """当てても素の構造 (パラメータ名・子モジュール) が 1 つも動かない。

    契約 (`_gdn_spec`) を dict / tuple で持たせると `nn.Module.__setattr__` が
    モジュール辞書に入れてしまい、`children()` の走査に混ざる。
    """
    from mlx.utils import tree_flatten

    gdns = [_make_gdn()]
    model = _StubModel(gdns)
    before_p = sorted(k for k, _ in tree_flatten(gdns[0].parameters()))
    before_m = sorted(k for k, _ in gdns[0].named_modules())
    with _env(MLXTURBO_GDN_DECODE_FUSED="1", MLXTURBO_GDN_METAL=None):
        fused.enable_gdn_port(model)
    try:
        assert sorted(k for k, _ in tree_flatten(gdns[0].parameters())) == before_p
        assert sorted(k for k, _ in gdns[0].named_modules()) == before_m
    finally:
        fused.disable_gdn_port(model)


def test_knobs_select_parts():
    """`MLXTURBO_GDN_DECODE_FUSED` / `MLXTURBO_GDN_METAL` で部品を切れる。"""
    gdns = [_make_gdn()]
    model = _StubModel(gdns)
    with _env(MLXTURBO_GDN_DECODE_FUSED="0", MLXTURBO_GDN_METAL="0"):
        got = fused.enable_gdn_port(model)
    assert got == {"metal": 0, "prework": 0, "norm": 0, "layers": 1}
    with _env(MLXTURBO_GDN_DECODE_FUSED="pre", MLXTURBO_GDN_METAL="0"):
        got = fused.enable_gdn_port(model)
    try:
        assert got["prework"] == 1 and got["norm"] == 0 and got["metal"] == 0
    finally:
        fused.disable_gdn_port(model)


# --------------------------------------------------------------- 一致の検査


def _decode_run(gdn, steps, S, B=1, seed=7):
    """`steps` 回の decode/verify 幅フォワード。出力と cache を返す。"""
    cache = ArraysCache(size=2)
    outs = []
    for i in range(steps):
        x = mx.random.normal((B, S, HIDDEN), key=mx.random.key(seed + i))
        x = x.astype(mx.bfloat16)
        outs.append(gdn(x, None, cache))
    mx.eval(outs, cache[0], cache[1])
    return outs, cache


def _prefill_run(gdn, T, B=1, seed=11):
    cache = ArraysCache(size=2)
    x = mx.random.normal((B, T, HIDDEN), key=mx.random.key(seed)).astype(mx.bfloat16)
    out = gdn(x, None, cache)
    mx.eval(out, cache[0], cache[1])
    return [out], cache


def _rel_err(a, b) -> float:
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    denom = float(mx.maximum(mx.abs(b32).max(), mx.array(1e-6)))
    return float(mx.abs(a32 - b32).max()) / denom


def _compare(ref, got, *, exact: bool, label: str, tol: float = 0.0):
    (ref_outs, ref_cache), (got_outs, got_cache) = ref, got
    pairs = list(zip(ref_outs, got_outs)) + [
        (ref_cache[0], got_cache[0]), (ref_cache[1], got_cache[1])
    ]
    for i, (r, g) in enumerate(pairs):
        assert r.shape == g.shape and r.dtype == g.dtype, f"{label}[{i}] 形/dtype 不一致"
        if exact:
            assert mx.array_equal(r, g), (
                f"{label}[{i}] がビット一致しない (相対 {_rel_err(g, r):.3e})")
        else:
            err = _rel_err(g, r)
            assert err <= tol, f"{label}[{i}] 相対誤差 {err:.3e} > {tol}"


def test_decode_bit_identical():
    """decode/verify 幅: 前処理 + 出力 norm を当てても素とビット一致。"""
    if not GPU:
        return
    for S in (1, 2, 3, 6):
        gdn = _make_gdn()
        ref = _decode_run(gdn, steps=3, S=S)
        model = _StubModel([gdn])
        with _env(MLXTURBO_GDN_DECODE_FUSED="1", MLXTURBO_GDN_METAL="0"):
            got_counts = fused.enable_gdn_port(model)
        try:
            assert got_counts["prework"] == 1 and got_counts["norm"] == 1
            before = _fire.snapshot()
            got = _decode_run(gdn, steps=3, S=S)
            after = _fire.snapshot()
        finally:
            fused.disable_gdn_port(model)
        assert after.get("gdn_prework", 0) - before.get("gdn_prework", 0) == 3, \
            f"S={S}: gdn_prework が 3 回発火していない"
        assert after.get("rms_norm_gated", 0) - before.get("rms_norm_gated", 0) == 3, \
            f"S={S}: 出力 norm が 3 回発火していない"
        _compare(ref, got, exact=True, label=f"decode S={S}")


def test_prefill_metal_close():
    """prefill 幅: Metal 再帰は加算順が違うのでビット一致しないが誤差は小さい。

    出力 norm は行数 (B*T*n_v) が `_GDN_NORM_MAX_ROWS` を超えるので発火しない
    (本番の 27B でも n_v=48 x decode 幅 <=16 で 768 が上限)。
    """
    if not GPU:
        return
    T = 256
    gdn = _make_gdn()
    ref = _prefill_run(gdn, T)
    model = _StubModel([gdn])
    with _env(MLXTURBO_GDN_DECODE_FUSED="1", MLXTURBO_GDN_METAL=None):
        counts = fused.enable_gdn_port(model)
    try:
        assert counts["metal"] == 1
        before = _fire.snapshot()
        got = _prefill_run(gdn, T)
        after = _fire.snapshot()
    finally:
        fused.disable_gdn_port(model)
    assert after.get("gdn_metal", 0) - before.get("gdn_metal", 0) == 1, \
        "prefill の GDN Metal が発火していない"
    assert after.get("gdn_prework", 0) == before.get("gdn_prework", 0), \
        "prefill 幅で decode 用の前処理が発火した"
    assert after.get("rms_norm_gated", 0) == before.get("rms_norm_gated", 0), \
        f"prefill 幅 (B*T*n_v={T * N_V} 行) で出力 norm が発火した"
    # 実測 (2026-09-04): 出力の相対誤差は T=128/256/1024 で 5.0e-4 / 1.3e-3 /
    # 1.9e-3 (bf16 の 1 ulp = 3.9e-3 の内側)。再帰状態 (fp32) と次段 conv 状態は
    # T=256/1024 でビット一致。許容は 2 ulp 相当に取る。
    _compare(ref, got, exact=False, label=f"prefill T={T}", tol=8e-3)


def test_prefill_metal_off_is_bit_identical():
    """`MLXTURBO_GDN_METAL=0` の prefill は素そのもの (経路が変わらない)。"""
    if not GPU:
        return
    T = 256
    gdn = _make_gdn()
    ref = _prefill_run(gdn, T)
    model = _StubModel([gdn])
    with _env(MLXTURBO_GDN_DECODE_FUSED="1", MLXTURBO_GDN_METAL="0"):
        fused.enable_gdn_port(model)
    try:
        got = _prefill_run(gdn, T)
    finally:
        fused.disable_gdn_port(model)
    _compare(ref, got, exact=True, label=f"prefill(metal off) T={T}")


def test_norm_activation_detected():
    """出力 norm の活性化を合成入力で決める (qwen3_5 は silu 固定)。"""
    if not GPU:
        return
    gdn = _make_gdn()
    nbase = type(gdn.norm)
    assert nbase.__name__ == "Qwen3NextRMSNormGated"
    assert fused._gdn_norm_activation(gdn.norm, nbase) == "silu"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = True
    for fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"{fn.__name__}: FAILED {type(e).__name__}: {e}")
        else:
            print(f"{fn.__name__}: OK{'' if GPU else ' (GPU 無しの範囲)'}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
