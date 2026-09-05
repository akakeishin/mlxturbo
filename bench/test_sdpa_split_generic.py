"""sdpa の幅分割 (汎用版) が qwen3_5 / 27B に当たること。

`fused.enable_sdpa_split` のシームは `mlxturbo/_vendor/qwen4_exp.py` にしか
無く、27B (qwen3_5 -> `mlx_lm.models.qwen3_next.Qwen3NextAttention`) は
S * gqa_factor > 32 の壁をそのまま踏んでいた
(`scratchpad/agent-ceiling-audit.md` の [t14])。
`fused.enable_sdpa_split_generic` は行タイル (段 P5) と同じ差し替え口
(attention の `__call__` の名前空間の `scaled_dot_product_attention`) に
分割の分岐を足す。

2 段構成:

- **契約の検査 (CPU、数秒)**: 既定 off、knob の読み方、当たる層数、
  qwen4_exp を除くこと、head_dim で切らないこと (行タイルとの違い)、
  `enable_sdpa_split` が数える層数。
- **一致の検査 (GPU 必須)**: 本物の `Qwen3NextAttention` を小さく組んで、
  壁の向こう (発火する幅) の出力が素と数値的に一致すること、壁の手前は
  **ビット一致** (= 何もしていない) であること、発火数 (`_fire`)。

実行:

    .venv/bin/python -m pytest bench/test_sdpa_split_generic.py -q
"""

from __future__ import annotations

import os
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
HEAD_DIM = 256
HIDDEN = 128
# 27B は Hq 24 / Hk 4 = gqa 6 (壁は S>=6)。試験は小さく組むので gqa 12
# (Hq 12 / Hk 1、壁は S>=3、w=2) にして、同じ分岐を短い S で踏む。
N_HEADS = 12
N_KV = 1
GQA = N_HEADS // N_KV


def _args(head_dim: int = HEAD_DIM, n_layers: int = 4) -> Q35.TextModelArgs:
    return Q35.TextModelArgs(
        model_type="qwen3_5",
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=n_layers,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
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
    def __init__(self, attn=None):
        super().__init__()
        if attn is not None:
            self.self_attn = attn


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


def _attn_model(n_attn: int = 3, n_other: int = 5, head_dim: int = HEAD_DIM,
                attn=None):
    layers = [_StubLayer(attn=attn or _make_attn(head_dim)) for _ in range(n_attn)]
    layers += [_StubLayer() for _ in range(n_other)]
    return _StubModel(layers)


# --- 契約の検査 (CPU) ----------------------------------------------------


def test_default_is_auto():
    """環境変数が未設定なら auto (非 NAX 機で on、NAX 機では 0)。明示の 0 なら
    機種に関わらず 0 を返してフラグも立たない。"""
    from mlxturbo.kernels.moe_grouped_gemm import is_nax_device

    old = os.environ.pop("MLXTURBO_SDPA_SPLIT_GENERIC", None)
    try:
        n = fused.enable_sdpa_split_generic(_attn_model())
        if is_nax_device():
            assert n == 0 and fused._SDPA_SPLIT_GENERIC is False
        else:
            assert n > 0 and fused._SDPA_SPLIT_GENERIC is True
        fused.disable_sdpa_split_generic()
        os.environ["MLXTURBO_SDPA_SPLIT_GENERIC"] = "0"
        assert fused.enable_sdpa_split_generic(_attn_model()) == 0
        assert fused._SDPA_SPLIT_GENERIC is False
    finally:
        if old is not None:
            os.environ["MLXTURBO_SDPA_SPLIT_GENERIC"] = old
        else:
            os.environ.pop("MLXTURBO_SDPA_SPLIT_GENERIC", None)


def test_enable_counts_attention_layers():
    """`mode="1"` で attention 層の数を返し、qwen3_next の名前を差し替える。"""
    try:
        n = fused.enable_sdpa_split_generic(_attn_model(n_attn=3, n_other=5),
                                            mode="1")
        assert n == 3
        assert fused._SDPA_SPLIT_GENERIC is True
        patched = Q35.Attention.__call__.__globals__["scaled_dot_product_attention"]
        assert getattr(patched, "_mlxturbo_rowtile_module", None) == QWEN3_NEXT
    finally:
        fused.disable_sdpa_split_generic()
    assert fused._SDPA_SPLIT_GENERIC is False


def test_enable_ignores_head_dim():
    """行タイルと違い head_dim では切らない (壁は S*gqa であって head_dim ではない)。"""
    try:
        assert fused.enable_sdpa_split_generic(
            _attn_model(n_attn=2, head_dim=64), mode="1") == 2
    finally:
        fused.disable_sdpa_split_generic()
    # 行タイル側の契約は変わっていない (head_dim 64 は対象外)
    assert fused._sdpa_rowtile_attn_namespaces(_attn_model(head_dim=64)) == []


def test_enable_skips_qwen4_exp():
    """qwen4_exp は本体シームを明示するので汎用版は数えない。"""
    import mlx_lm.models.qwen4_exp as Q4

    assert Q4.Attention.__call__.__globals__["_MLXTURBO_NATIVE_SDPA_SPLIT_SEAM"] is True
    found = fused._sdpa_rowtile_attn_namespaces(_attn_model(n_attn=2),
                                                min_head_dim=0)
    assert [m for m, _, _ in found] == [QWEN3_NEXT]
    assert Q4.__name__ not in [m for m, _, _ in found]


def test_enable_skips_any_native_sdpa_seam_capability():
    """定義元が能力を宣言すれば、モジュール名によらず二重patchしない。"""
    ns = Q35.Attention.__call__.__globals__
    marker = "_MLXTURBO_NATIVE_SDPA_SPLIT_SEAM"
    missing = object()
    old = ns.get(marker, missing)
    ns[marker] = True
    try:
        assert fused.enable_sdpa_split_generic(_attn_model(n_attn=2), mode="1") == 0
        assert fused._SDPA_SPLIT_GENERIC is False
    finally:
        fused.disable_sdpa_split_generic()
        if old is missing:
            ns.pop(marker, None)
        else:
            ns[marker] = old


def test_enable_empty_model():
    """attention 層が無ければ 0 のまま (フラグも立たない)。"""
    try:
        assert fused.enable_sdpa_split_generic(_StubModel([_StubLayer()]),
                                               mode="1") == 0
        assert fused._SDPA_SPLIT_GENERIC is False
        assert fused.enable_sdpa_split_generic(None, mode="1") == 0
    finally:
        fused.disable_sdpa_split_generic()


def test_enable_sdpa_split_counts_seam_layers():
    """従来のシーム側は「qwen4_exp の Attention を持つ層」を数える。"""
    import mlx_lm.models.qwen4_exp as Q4

    assert fused.enable_sdpa_split(_attn_model(n_attn=3)) == 0  # qwen3_5 は対象外
    assert Q4.Attention._sdpa_split_width is True
    assert fused.enable_sdpa_split(None) == 0
    old = os.environ.get("MLXTURBO_SDPA_SPLIT")
    os.environ["MLXTURBO_SDPA_SPLIT"] = "0"
    try:
        assert fused.enable_sdpa_split(_attn_model(n_attn=3)) == 0
        assert Q4.Attention._sdpa_split_width is False
    finally:
        if old is None:
            os.environ.pop("MLXTURBO_SDPA_SPLIT", None)
        else:
            os.environ["MLXTURBO_SDPA_SPLIT"] = old
        fused.enable_sdpa_split()


# --- 一致の検査 (GPU) ----------------------------------------------------


def _decode_step(attn, x, cache):
    mask = cache.make_mask(x.shape[1], return_array=False, window_size=None)
    out = attn(x, mask, cache)
    mx.eval(out)
    return out


def _run(attn, kv: int, S: int):
    """kv トークンぶん prefill してから幅 S を 1 回。"""
    mx.random.seed(0)
    cache = KVCache()
    pre = mx.random.normal((1, kv, HIDDEN)).astype(mx.float32)
    mx.eval(pre)
    mx.eval(attn(pre, "causal", cache))
    step = mx.random.normal((1, S, HIDDEN)).astype(mx.float32)
    mx.eval(step)
    return _decode_step(attn, step, cache)


def test_split_matches_stock_above_the_wall():
    """壁の向こう (S*gqa>32) で分割 on/off の出力が数値的に一致し、発火する。

    分割の前後で「どの key が見えるか」は変わらないが、MLX が別のカーネル
    (壁の向こうの fallback 対 vector) を選ぶのでビット一致はしない
    (`tools/sdpa_split_generic_micro.py` の bitident が bf16 で 1 ulp 級)。
    """
    if not GPU:
        return
    attn = _make_attn(dtype=mx.float32)
    S = 4  # gqa 12 -> S*gqa = 48 > 32
    assert S * GQA > fused._SDPA_SPLIT_QROWS

    fused.disable_sdpa_split_generic()
    ref = _run(attn, 1024, S)

    os.environ["MLXTURBO_SDPA_SPLIT_GENERIC_TRACE"] = "1"
    _fire.reset()
    try:
        assert fused.enable_sdpa_split_generic(
            _attn_model(n_attn=1, n_other=0, attn=attn), mode="1") == 1
        got = _run(attn, 1024, S)
    finally:
        fused.disable_sdpa_split_generic()
        os.environ.pop("MLXTURBO_SDPA_SPLIT_GENERIC_TRACE", None)

    assert _fire.snapshot().get("sdpa_split_generic", 0) >= 1
    rel = float(mx.max(mx.abs(got - ref)) / mx.max(mx.abs(ref)))
    assert rel < 1e-5, rel


def test_split_is_bitident_below_the_wall():
    """壁の手前 (S*gqa<=32) は素通り = ビット一致、発火もしない。"""
    if not GPU:
        return
    attn = _make_attn(dtype=mx.float32)
    S = 2  # gqa 12 -> S*gqa = 24 <= 32
    assert S * GQA <= fused._SDPA_SPLIT_QROWS

    fused.disable_sdpa_split_generic()
    ref = _run(attn, 1024, S)

    os.environ["MLXTURBO_SDPA_SPLIT_GENERIC_TRACE"] = "1"
    _fire.reset()
    try:
        assert fused.enable_sdpa_split_generic(
            _attn_model(n_attn=1, n_other=0, attn=attn), mode="1") == 1
        got = _run(attn, 1024, S)
    finally:
        fused.disable_sdpa_split_generic()
        os.environ.pop("MLXTURBO_SDPA_SPLIT_GENERIC_TRACE", None)

    assert _fire.snapshot().get("sdpa_split_generic", 0) == 0
    assert bool(mx.array_equal(got, ref))


def test_split_does_not_fire_at_prefill_width():
    """prefill 幅 (S > _SDPA_SPLIT_MAX_S) は分割の対象外 (行タイルの担当)。"""
    if not GPU:
        return
    attn = _make_attn(dtype=mx.float32)
    os.environ["MLXTURBO_SDPA_SPLIT_GENERIC_TRACE"] = "1"
    _fire.reset()
    try:
        assert fused.enable_sdpa_split_generic(
            _attn_model(n_attn=1, n_other=0, attn=attn), mode="1") == 1
        cache = KVCache()
        x = mx.random.normal((1, 64, HIDDEN)).astype(mx.float32)
        mx.eval(x)
        mx.eval(attn(x, "causal", cache))
    finally:
        fused.disable_sdpa_split_generic()
        os.environ.pop("MLXTURBO_SDPA_SPLIT_GENERIC_TRACE", None)
    assert _fire.snapshot().get("sdpa_split_generic", 0) == 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
