"""qwen4_exp 以外の族を読んでも `enable_default_fusions` が落ちないこと。

回帰の内容 (2026-09-04): `mlxturbo/runner.py` の `enable_default_fusions` は
model_type で分岐せず、読み込んだモデルに全部の enable_* を当てる。クラスを
差し替えるだけのもの (``Q.<Class>.<attr> = ...``) は他の族には当たらないので
素通りするが、**モデルの構造を直接舐める**もの (``model.model.layers``) は
族が違うと AttributeError で落ちていた。Qwen3.8-27B
(`mlx_lm.models.qwen3_5.Model`) のラッパは ``model.language_model.model.layers``
で ``model.model`` を持たないので、既定 on の融合を仕込む段
(`enable_hc_qmm_wide` / `enable_gdn_decode_fused` / `enable_moe_combine_fold` /
`enable_moe_down_epilogue` / `enable_qmm_wide` / `gather_attn` / `qsa_decode`)
がそこで止まる。

直したあとの契約は 2 つ:

1. 層が見つからない (または層に部品が無い) 族では、構造を舐める enable_* は
   **何もせず 0 / None を返す** (エラーにしない。`docs/BACKLOG.md` の
   「動的な構造の探索 (duck typing)」)。
2. qwen4_exp 側の挙動は 1 ビットも変わらない。層の列挙
   (`fused._model_layers`) が ``list(model.model.layers)`` と同じ順の同じ
   オブジェクトを返し、各 enable_* が数える層数も変わらない。

CPU で走る合成モデルの一次検査 (数秒)。GPU 分岐は通らないので、実機の煙
試験 (`bench/self_snapshot.py --model ~/models/qwen38-27b-4bit`) の代わりには
ならない。

実行:

    .venv/bin/python bench/test_fusions_other_family.py
    .venv/bin/python -m pytest bench/test_fusions_other_family.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mlx.core as mx  # noqa: E402

import mlx.nn as nn  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_device():
    """このファイルのテストだけ CPU で。module 直下で set_default_device すると
    同じ pytest プロセスの他ファイルの GPU 判定 (mx.default_device() == gpu) が
    収集時に False になり、一致検査が 1 つも走らなくなる (2026-09-04 に発覚)。"""
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)

import mlxturbo  # noqa: E402,F401 -- sys.meta_path フック (mlx_lm.models.qwen4_exp)
from mlxturbo import fused, gather_attn, indexer_lean, qsa_decode  # noqa: E402
from mlxturbo.runner import enable_default_fusions  # noqa: E402


# --- 他の族のスタブ (27B と同じ形だけを真似る) ---------------------------


class _StubLayer(nn.Module):
    """部品 (`self_attn` / `linear_attn` / `mlp`) を 1 つも持たない層。

    27B の層は当然これらとは別の中身を持つが、fastmlx の融合が探す名前
    (qwen4_exp の名前) では引っかからない。「名前が合わない = 何もしない」
    が確認したい契約なので、空の層で足りる。
    """


class _StubTextModel(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = [_StubLayer() for _ in range(n_layers)]


class _StubLanguageModel(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.model = _StubTextModel(n_layers)


class _StubOtherFamily(nn.Module):
    """`mlx_lm.models.qwen3_5.Model` と同じ形 (``.language_model.model.layers``
    を持ち、``.model`` は持たない)。"""

    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.args = SimpleNamespace(model_type="qwen3_5")
        self.model_type = "qwen3_5"
        self.language_model = _StubLanguageModel(n_layers)


class _StubNoLayers(nn.Module):
    """層がどこからも見つからないモデル (契約が合わない極端な側)。"""

    def __init__(self):
        super().__init__()
        self.args = SimpleNamespace(model_type="whatever")


# 構造を舐める enable_* / disable_*。`(呼ぶもの, 期待する戻り値)` で、
# 期待値 0 は「層に部品が無いので 1 つも当たらない」、None は戻り値を
# 持たない関数。
def _structure_walkers(model):
    return [
        ("fused.enable_gdn_prework_kernel", lambda: fused.enable_gdn_prework_kernel(model), None),
        ("fused.enable_gdn_decode_fused", lambda: fused.enable_gdn_decode_fused(model), None),
        ("fused.enable_moe_combine_fold", lambda: fused.enable_moe_combine_fold(model), 0),
        ("fused.disable_moe_combine_fold", lambda: fused.disable_moe_combine_fold(model), 0),
        ("fused.enable_moe_shared_fold", lambda: fused.enable_moe_shared_fold(model), 0),
        ("fused.enable_wide_projections",
         lambda: sum(fused.enable_wide_projections(model).values()), 0),
        ("fused.disable_wide_projections", lambda: fused.disable_wide_projections(model), 0),
        ("fused.enable_qmm_wide", lambda: fused.enable_qmm_wide(model), 0),
        ("fused.enable_sdpa_rowtile", lambda: fused.enable_sdpa_rowtile(model), 0),
        ("fused.enable_hc_qmm_wide", lambda: fused.enable_hc_qmm_wide(model), 0),
        ("fused.disable_hc_qmm_wide", lambda: fused.disable_hc_qmm_wide(model), None),
        ("fused._hc_gated_residuals", lambda: len(list(fused._hc_gated_residuals(model))), 0),
        ("fused.enable_moe_down_epilogue", lambda: fused.enable_moe_down_epilogue(model), 0),
        ("fused.enable_fast_rope", lambda: fused.enable_fast_rope(model), 0),
        ("fused.disable_fast_rope", lambda: fused.disable_fast_rope(model), 0),
        ("fused.enable_ple_hoist", lambda: fused.enable_ple_hoist(model), 0),
        ("gather_attn.enable_gather_attn", lambda: gather_attn.enable_gather_attn(model), 0),
        ("gather_attn.enable_prefill_attn", lambda: gather_attn.enable_prefill_attn(model), 0),
        ("indexer_lean.enable_indexer_lean", lambda: indexer_lean.enable_indexer_lean(model), 0),
        ("qsa_decode.enable_qsa_decode_kernel",
         lambda: qsa_decode.enable_qsa_decode_kernel(model), 0),
    ]


def test_model_layers_finds_language_model_wrapper():
    """`.language_model.model.layers` の族でも層が見つかる。"""
    stub = _StubOtherFamily(n_layers=4)
    layers = fused._model_layers(stub)
    assert len(layers) == 4
    assert all(a is b for a, b in zip(layers, stub.language_model.model.layers))


def test_model_layers_empty_when_no_layers():
    """層が無ければ空リスト (例外にしない)。"""
    assert fused._model_layers(_StubNoLayers()) == []
    assert fused._model_layers(None) == []
    assert fused._model_body(_StubNoLayers()) is None


def test_qmm_wide_extra_shapes_are_role_and_shape_gated():
    """他族の追加routeは、同名でも未測定shapeへ外挿しない。"""
    cases = {
        ("mlp", "gate_up_proj"): (4096, 27392),
        ("mlp", "w1"): (2048, 5504),
        ("mlp", "w2"): (2048, 5504),
        ("mlp", "c_proj"): (5504, 2048),
        ("self_attn", "q_b_proj"): (1536, 6144),
        ("self_attn", "kv_a_proj_with_mqa"): (4096, 576),
    }
    assert set(fused._QMM_WIDE_KNOWN_EXTRA) == set(cases)
    for role, shape in cases.items():
        assert shape in fused._QMM_WIDE_KNOWN_EXTRA[role]
        assert (shape[0], shape[1] + 32) not in fused._QMM_WIDE_KNOWN_EXTRA[role]

    glm = object()
    deepseek = object()
    layer = SimpleNamespace(
        mlp=SimpleNamespace(gate_up_proj=glm),
        self_attn=SimpleNamespace(q_b_proj=deepseek),
    )
    got = list(fused._qmm_wide_candidates(layer))
    assert (glm, fused._QMM_WIDE_KNOWN_EXTRA[("mlp", "gate_up_proj")]) in got
    assert (deepseek, fused._QMM_WIDE_KNOWN_EXTRA[("self_attn", "q_b_proj")]) in got


def test_structure_walkers_no_op_on_other_family():
    """他の族では構造を舐める enable_* が全部 0 / None を返す (例外を出さない)。"""
    for model in (_StubOtherFamily(n_layers=4), _StubNoLayers()):
        for name, call, expect in _structure_walkers(model):
            got = call()
            assert expect is None or got == expect, f"{name}: {got!r} != {expect!r}"


def test_moe_combine_fold_does_not_report_unwired_switch_mlp():
    """同名部品だけでは有効扱いにしない。別族にはconsumerが無い。"""
    model = _StubOtherFamily(n_layers=1)
    mlp = SimpleNamespace(switch_mlp=object())
    model.language_model.model.layers[0].mlp = mlp

    assert fused.enable_moe_combine_fold(model) == 0
    assert not hasattr(mlp, "_combine_fold_min_s")


def test_enable_default_fusions_other_family_does_not_raise():
    """回帰そのもの: 他の族に `enable_default_fusions` を当てても落ちない。"""
    for model in (_StubOtherFamily(n_layers=4), _StubNoLayers()):
        enable_default_fusions(model, log_prefix="[test]")


# --- qwen4_exp 側 (従来どおり当たること) ---------------------------------


def _build_qwen4_exp():
    from verify_batch_cache import build  # tools/ (sys.path に入れてある)

    return build(8)


def test_qwen4_exp_layer_enumeration_unchanged():
    """qwen4_exp では層の列挙が ``list(model.model.layers)`` と同一。"""
    model = _build_qwen4_exp()
    ref = list(model.model.layers)
    got = fused._model_layers(model)
    assert len(got) == len(ref) and len(ref) > 0
    assert all(a is b for a, b in zip(got, ref))
    assert fused._model_body(model) is model.model


def test_qwen4_exp_fusions_still_apply():
    """qwen4_exp では従来どおり層に当たる (数は構造から数え直した基準と一致)。"""
    model = _build_qwen4_exp()
    layers = list(model.model.layers)

    n_moe = sum(1 for l in layers
                if getattr(getattr(l, "mlp", None), "switch_mlp", None) is not None)
    n_attn = sum(1 for l in layers
                 if getattr(getattr(l, "self_attn", None), "indexer", None) is not None)
    n_hc = sum(1 for l in layers
               for h in ("attn_hyper_connection", "mlp_hyper_connection")
               if getattr(l, h, None) is not None)
    n_hc += 1 if getattr(model.model, "hyper_connection_mixer", None) is not None else 0

    assert n_moe > 0 and n_attn > 0 and n_hc > 0
    assert fused.enable_moe_combine_fold(model) == n_moe
    assert fused.disable_moe_combine_fold(model) == n_moe
    assert len(list(fused._hc_gated_residuals(model))) == n_hc
    assert gather_attn.enable_gather_attn(model) == n_attn
    assert indexer_lean.enable_indexer_lean(model) == n_attn
    assert indexer_lean.disable_indexer_lean(model) == n_attn
    assert qsa_decode.enable_qsa_decode_kernel(model) == n_attn
    assert qsa_decode.disable_qsa_decode_kernel(model) == n_attn


def test_qwen4_exp_moe_has_no_dense_mlp_projections():
    """`_QMM_WIDE_TARGETS` に足した `mlp` の 3 本は qwen4_exp に当たらない。

    27B (qwen3_5) の dense MLP のために
    `("mlp", ("gate_proj","up_proj","down_proj"))` を足したが、qwen4_exp の
    `layer.mlp` は `SparseMoeBlock` で、専門家は `mlp.switch_mlp` (SwitchGLU)、
    共有専門家は `mlp.shared_expert` の下にある。**`mlp` 直下にはこの 3 つの
    名前が無い**ので、印を付ける層が 1 つも増えない。
    """
    model = _build_qwen4_exp()
    names = dict(fused._QMM_WIDE_TARGETS)["mlp"]
    assert names == ("gate_proj", "up_proj", "down_proj")
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        assert mlp is not None
        for name in names:
            assert getattr(mlp, name, None) is None, f"{type(mlp).__name__}.{name}"


def test_sdpa_rowtile_patches_qwen4_exp_for_any_model():
    """行タイルの差し替え先は族で決め打ちしないが、qwen4_exp は常に差さる。

    `enable_sdpa_rowtile(model=None)` という従来の呼び方 (`tools/decode_ab.py`
    と `runner.py`) を保つため。他の族のモデルを渡しても qwen4_exp 側の
    差し替えは変わらず、その族の attention 層が見つからなければ 0 を返す。
    """
    import mlx_lm.models.qwen4_exp as Q4

    try:
        assert fused.enable_sdpa_rowtile(None, rows=256) == 0
        assert getattr(Q4.scaled_dot_product_attention,
                       "_mlxturbo_rowtile_module", None) == Q4.__name__
        assert fused.enable_sdpa_rowtile(_StubOtherFamily(n_layers=4)) == 0
    finally:
        fused.disable_sdpa_rowtile()


def test_enable_default_fusions_qwen4_exp_still_runs():
    model = _build_qwen4_exp()
    enable_default_fusions(model, log_prefix="[test]")
    # 既定 on の gather 経路が実際に立っていること (素通りしていない印)
    hit = sum(1 for l in model.model.layers
              if getattr(getattr(l, "self_attn", None), "_gather_attn", False))
    assert hit > 0


def test_enable_hc_write_honors_direct_env_disable(monkeypatch):
    """runnerを経由しない直接呼出しでも明示offなら差し替えない。"""
    import mlx_lm.models.qwen4_exp as Q4

    fused.disable_hc_write()
    original = Q4.DecoderLayer._combine
    monkeypatch.setenv("MLXTURBO_HC_WRITE", "0")
    fused.enable_hc_write()
    assert Q4.DecoderLayer._combine is original
    assert fused._ORIG_HC_COMBINE is None


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
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
