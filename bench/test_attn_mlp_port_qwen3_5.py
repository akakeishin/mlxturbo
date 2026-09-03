"""sdpa 行タイル (段 P5) と qmm_wide の MLP (段 P10) が qwen3_5 に当たること。

27B レーンの第 2 段。どちらも「族で決め打ちしない」形にしてある:

- `fused.enable_sdpa_rowtile` は層を歩いて attention 層 (`self_attn` に
  `q_proj`、`head_dim` が MLX の fast sdpa に乗らない幅) を探し、その
  `__call__` が定義されている名前空間の `scaled_dot_product_attention` を
  差し替える。qwen4_exp は `mlx_lm.models.qwen4_exp`、Qwen3.8-27B (qwen3_5)
  は `mlx_lm.models.qwen3_next` (qwen3_5 が `Qwen3NextAttention` を import
  しているので、sdpa を呼ぶのはあちら側の名前)。
- `fused._QMM_WIDE_TARGETS` に `mlp:(gate_proj, up_proj, down_proj)` を足した。
  27B の dense MLP に当たり、qwen4_exp の `SparseMoeBlock` (直下に 3 本を
  持たない) には当たらない。

2 段構成:

- **契約の検査 (CPU、数秒)**: 名前空間の探索、head_dim が足りない族を
  外すこと、差し替えが 1 回だけであること、`mlp` の印が付く数。
- **一致の検査 (GPU 必須)**: 本物の `Qwen3NextAttention` を小さく組んで、
  行タイルの出力を素と突き合わせる (加算順が変わるのでビット一致しない ---
  相対誤差で見る) と、発火数 (`_fire`) が 0 でないこと。MLP の qmm_wide は
  **ビット一致**。

実行:

    .venv/bin/python -m pytest bench/test_attn_mlp_port_qwen3_5.py -q
    .venv/bin/python bench/test_attn_mlp_port_qwen3_5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlxturbo  # noqa: E402,F401 -- sys.meta_path フック (mlx_lm.models.qwen4_exp)
from mlx_lm.models import qwen3_5 as Q35  # noqa: E402
from mlx_lm.models.cache import KVCache  # noqa: E402
from mlxturbo import fused  # noqa: E402
from mlxturbo.kernels import _fire  # noqa: E402

GPU = mx.metal.is_available() and mx.default_device() == mx.gpu

QWEN3_NEXT = "mlx_lm.models.qwen3_next"
HEAD_DIM = 256          # 27B と同じ (MLX の fast sdpa に乗らない幅)
HIDDEN = 256


def _args(head_dim: int = HEAD_DIM, n_layers: int = 4) -> Q35.TextModelArgs:
    return Q35.TextModelArgs(
        model_type="qwen3_5",
        hidden_size=HIDDEN,
        intermediate_size=512,
        num_hidden_layers=n_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=128,
        head_dim=head_dim,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25},
    )


def _make_attn(head_dim: int = HEAD_DIM, dtype=mx.float32):
    attn = Q35.Attention(_args(head_dim))
    attn.set_dtype(dtype)
    attn.eval()
    return attn


class _StubLayer(nn.Module):
    def __init__(self, attn=None, mlp=None):
        super().__init__()
        if attn is not None:
            self.self_attn = attn
        if mlp is not None:
            self.mlp = mlp


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

    def __init__(self, layers):
        super().__init__()
        self.language_model = _StubLanguageModel(layers)


def _attn_model(n_attn: int = 3, n_other: int = 5, head_dim: int = HEAD_DIM):
    """27B と同じ並び (attention 層と、それを持たない層が混ざる)。"""
    layers = [_StubLayer(attn=_make_attn(head_dim)) for _ in range(n_attn)]
    layers += [_StubLayer() for _ in range(n_other)]
    return _StubModel(layers)


# --- 契約の検査 (CPU) ----------------------------------------------------


def test_namespaces_finds_qwen3_next():
    """qwen3_5 の attention 層から qwen3_next の名前空間が 1 つ見つかる。"""
    found = fused._sdpa_rowtile_attn_namespaces(_attn_model(n_attn=3, n_other=5))
    assert len(found) == 1
    modname, ns, count = found[0]
    assert modname == QWEN3_NEXT
    assert count == 3
    assert ns is Q35.Attention.__call__.__globals__


def test_namespaces_rejects_small_head_dim():
    """head_dim が MLX の fast sdpa に乗る幅 (<=128) なら対象にしない。"""
    assert fused._sdpa_rowtile_attn_namespaces(_attn_model(head_dim=128)) == []
    assert fused._sdpa_rowtile_attn_namespaces(_attn_model(head_dim=64)) == []


def test_namespaces_empty_without_attention():
    """`self_attn` を持たない族 / 層の無いモデルでは空。"""
    assert fused._sdpa_rowtile_attn_namespaces(_StubModel([_StubLayer()])) == []
    assert fused._sdpa_rowtile_attn_namespaces(None) == []


def test_enable_patches_qwen3_next_once():
    """差し替えは名前空間ごとに 1 回だけ (素の関数を上書きしない)。"""
    import mlx_lm.models.qwen4_exp as Q4

    n = fused.enable_sdpa_rowtile(_attn_model(n_attn=3), rows=64)
    try:
        assert n == 3
        patched = Q35.Attention.__call__.__globals__["scaled_dot_product_attention"]
        assert getattr(patched, "_mlxturbo_rowtile_module", None) == QWEN3_NEXT
        assert QWEN3_NEXT in fused._SDPA_ROWTILE_ORIGS
        # qwen4_exp 側も従来どおり差さっている
        assert getattr(Q4.scaled_dot_product_attention,
                       "_mlxturbo_rowtile_module", None) == Q4.__name__
        orig = fused._SDPA_ROWTILE_ORIGS[QWEN3_NEXT]
        # 2 回目で身代わりを「素」として覚え直さない
        assert fused.enable_sdpa_rowtile(_attn_model(n_attn=3), rows=64) == 3
        assert fused._SDPA_ROWTILE_ORIGS[QWEN3_NEXT] is orig
        assert Q35.Attention.__call__.__globals__[
            "scaled_dot_product_attention"] is patched
    finally:
        fused.disable_sdpa_rowtile()


def test_enable_returns_zero_when_off():
    """R<=0 なら 0 を返して何も差さない。"""
    assert fused.enable_sdpa_rowtile(_attn_model(), rows=0) == 0
    assert fused._SDPA_ROWTILE_ROWS == 0


def test_qmm_wide_targets_include_dense_mlp():
    holders = dict(fused._QMM_WIDE_TARGETS)
    assert holders["mlp"] == ("gate_proj", "up_proj", "down_proj")


# --- 一致の検査 (GPU) ----------------------------------------------------


def _prefill(attn, x, mask="causal"):
    cache = KVCache()
    out = attn(x, mask, cache)
    mx.eval(out)
    return out


def test_rowtile_matches_stock_on_qwen3_next_attention():
    """行タイル on/off で `Qwen3NextAttention` の出力が一致する (数値の範囲)。

    可視集合はタイル分割で変わらないが、PV の縮約が切れて和の順序が動くので
    ビット一致はしない (Flash-Next での実績と同じ物差し)。
    """
    if not GPU:
        return
    mx.random.seed(0)
    attn = _make_attn(dtype=mx.float32)
    x = mx.random.normal((1, 512, HIDDEN)).astype(mx.float32)
    mx.eval(x)

    fused.disable_sdpa_rowtile()
    ref = _prefill(attn, x)

    _fire.reset()
    try:
        n = fused.enable_sdpa_rowtile(_StubModel([_StubLayer(attn=attn)]), rows=64)
        assert n == 1
        fused._SDPA_ROWTILE_TRACE = True   # enable が env から書くので後で立てる
        got = _prefill(attn, x)
    finally:
        fused.disable_sdpa_rowtile()
        fused._SDPA_ROWTILE_TRACE = False

    assert _fire.snapshot().get("sdpa_rowtile", 0) == 1, "行タイルが発火していない"
    assert got.shape == ref.shape
    rel = (mx.abs(got - ref).max() / mx.abs(ref).max()).item()
    assert rel < 1e-5, f"素との相対誤差が大きい: {rel}"


def test_rowtile_passes_through_decode_width():
    """decode 幅 (S=1、mask=None) は素の sdpa に落ちる (発火しない)。"""
    if not GPU:
        return
    mx.random.seed(0)
    attn = _make_attn(dtype=mx.float32)
    x = mx.random.normal((1, 1, HIDDEN)).astype(mx.float32)
    mx.eval(x)

    _fire.reset()
    try:
        fused.enable_sdpa_rowtile(_StubModel([_StubLayer(attn=attn)]), rows=64)
        fused._SDPA_ROWTILE_TRACE = True   # enable が env から書くので後で立てる
        cache = KVCache()
        out = attn(x, None, cache)
        mx.eval(out)
    finally:
        fused.disable_sdpa_rowtile()
        fused._SDPA_ROWTILE_TRACE = False
    assert _fire.snapshot().get("sdpa_rowtile", 0) == 0


def _quantized_mlp(dim=1024, hidden=2048, dtype=mx.bfloat16):
    mlp = Q35.MLP(dim, hidden)
    mlp.set_dtype(dtype)
    nn.quantize(mlp, group_size=64, bits=4)
    mlp.eval()
    return mlp


def test_qmm_wide_marks_dense_mlp_and_is_bit_identical():
    """27B の dense MLP に印が付き、prefill 幅の出力が素とビット一致する。"""
    if not GPU:
        return
    mx.random.seed(0)
    mlps = [_quantized_mlp() for _ in range(2)]
    model = _StubModel([_StubLayer(mlp=m) for m in mlps])

    x = (mx.random.normal((1024, 1024)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    from mlxturbo import fused as F

    F._QMM_WIDE_ON = False
    ref = [m(x) for m in mlps]
    mx.eval(ref)

    n = fused.enable_qmm_wide(model, mode="on")
    try:
        assert n == 6, f"印を付けた射影が 6 本でない: {n}"
        assert all(getattr(m.gate_proj, "_qmm_wide", None) is not None for m in mlps)
        assert all(getattr(m.down_proj, "_qmm_wide", None) is not None for m in mlps)
        got = [m(x) for m in mlps]
        mx.eval(got)
    finally:
        F._QMM_WIDE_ON = False

    for a, b in zip(got, ref):
        assert a.dtype == b.dtype
        assert mx.array_equal(a, b), "MLP の qmm_wide が素とビット一致しない"


def test_qmm_wide_decode_width_falls_back():
    """行数 < `_QMM_WIDE_MIN_ROWS` は素の quantized_matmul のまま。"""
    if not GPU:
        return
    mx.random.seed(0)
    mlp = _quantized_mlp()
    model = _StubModel([_StubLayer(mlp=mlp)])
    x = (mx.random.normal((8, 1024)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    from mlxturbo import fused as F

    F._QMM_WIDE_ON = False
    ref = mlp(x)
    mx.eval(ref)
    fused.enable_qmm_wide(model, mode="on")
    try:
        got = mlp(x)
        mx.eval(got)
    finally:
        F._QMM_WIDE_ON = False
    assert mx.array_equal(got, ref)


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
            print(f"{fn.__name__}: OK")
    print(f"GPU={GPU}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
