"""投機の検証フォワード (`SpecEngine._hidden_forward(capture=True)`) を、
GDN の写し (`_linear_capture`) からモジュール呼び出しに戻す検査。

対象は 2 つの knob (どちらも既定 off、値は 1 ビットも変わらないのが条件):

- ``MLXTURBO_SPEC_CAPTURE_MODULE=1``: 検証フォワードの GDN 層を
  ``layer(...)`` で呼び、巻き戻し用の材料 (位置ごとの再帰状態と conv の
  入力列) は ``fused.gdn_capture`` の取り出し口から受ける。これで
  ``fused.enable_gdn_port`` の decode 幅の前処理カーネルが S>1 の検証にも
  当たる (写しの経路には届いていなかった)。
- ``MLXTURBO_SPEC_STAGED_VERIFY=1``: 検証フォワードにも段階投入
  (``_STAGE_EVERY`` 層ごとの ``mx.async_eval``) を掛ける。

合成の qwen3_5 (4 層、うち 3 層が GDN) で、S=1,2,4,8 について

- 隠れ状態の出力
- 全キャッシュ (GDN の conv 状態・再帰状態、full attention の K/V と offset)
- 巻き戻し (``_rollback``) の後のキャッシュ

がビット一致することを見る。GPU が無ければ何もしない (自前カーネルが
発火しないので比較する意味が無い)。

    tools/biglock.sh .venv/bin/python -m pytest \\
        bench/test_spec_capture_module_qwen3_5.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import mlxturbo  # noqa: E402,F401 -- sys.meta_path フック
from mlx_lm.models import qwen3_5 as Q35  # noqa: E402
from mlxturbo import fused  # noqa: E402
from mlxturbo.kernels import _fire  # noqa: E402
from mlxturbo.spec import SpecEngine  # noqa: E402

GPU = mx.metal.is_available() and mx.default_device() == mx.gpu

# dk は GDN Metal カーネルの定数 (128)、key_dim は gdn_prework の BLOCK (128)
# の境界に乗せる (bench/test_gdn_port_qwen3_5.py と同じ形)。
N_K, N_V, DK, DV, K = 2, 6, 128, 128, 4
HIDDEN, VOCAB, N_LAYERS = 256, 64, 4


def _env(**kw):
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


def _build_model(dtype=mx.bfloat16):
    text = dict(
        model_type="qwen3_5",
        hidden_size=HIDDEN,
        intermediate_size=512,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        linear_num_value_heads=N_V,
        linear_num_key_heads=N_K,
        linear_key_head_dim=DK,
        linear_value_head_dim=DV,
        linear_conv_kernel_dim=K,
        head_dim=64,
        full_attention_interval=4,
    )
    model = Q35.Model(Q35.ModelArgs(model_type="qwen3_5", text_config=text))
    model.set_dtype(dtype)
    model.eval()
    return model


def _n_linear(model) -> int:
    return sum(1 for la in model.language_model.model.layers if la.is_linear)


def _cache_arrays(caches) -> list:
    """比較用に、全キャッシュの中身を 1 本の配列の列に平す。"""
    out = []
    for c in caches:
        if hasattr(c, "keys"):
            out.append(("kv_offset", mx.array(c.offset)))
            if c.keys is not None:
                out.append(("kv_keys", c.keys[..., : c.offset, :]))
                out.append(("kv_values", c.values[..., : c.offset, :]))
        else:
            # ArraysCache は offset を持たない (advance は lengths /
            # left_padding だけを動かす)。単一系列ではどちらも None。
            out.append(("gdn_conv", c[0]))
            out.append(("gdn_state", c[1]))
            for name in ("lengths", "left_padding"):
                v = getattr(c, name, None)
                if v is not None:
                    out.append((f"gdn_{name}", v))
    return out


def _same(a, b, label: str) -> None:
    assert len(a) == len(b), f"{label}: 本数が違う ({len(a)} != {len(b)})"
    for i, ((na, x), (nb, y)) in enumerate(zip(a, b)):
        assert na == nb, f"{label}[{i}]: 種類が違う ({na} != {nb})"
        assert x.shape == y.shape and x.dtype == y.dtype, (
            f"{label}[{i}] ({na}): 形/dtype 不一致 {x.shape}{x.dtype} != {y.shape}{y.dtype}")
        assert bool(mx.array_equal(x, y)), f"{label}[{i}] ({na}) がビット一致しない"


def _run(model, S: int, *, capture_module: str, staged_verify: str,
         prefill: int = 24, consumed: int | None = None):
    """prefill -> 検証フォワード (capture) -> 必要なら巻き戻し。

    戻り値は (隠れ状態, キャッシュの中身, sink の中身, 発火数の差分)。
    """
    engine = SpecEngine(model, mtp=None)
    caches = model.language_model.make_cache()
    ids = mx.arange(prefill).astype(mx.int32) % VOCAB
    win = (mx.arange(S).astype(mx.int32) + 7) % VOCAB

    with _env(MLXTURBO_SPEC_CAPTURE_MODULE=capture_module,
              MLXTURBO_SPEC_STAGED_VERIFY=staged_verify):
        h_pre, _ = engine._hidden_forward(ids, caches, capture=False)
        mx.eval(h_pre)
        before = _fire.snapshot()
        h, sink = engine._hidden_forward(win, caches, capture=True)
        mx.eval(h, [s[1] for s in sink], [s[2] for s in sink])
        after = _fire.snapshot()
        if consumed is not None:
            engine._rollback(caches, sink, S, consumed)
    arrays = _cache_arrays(caches)
    mx.eval([x for _, x in arrays])
    sink_arrays = []
    for cache, states_all, conv_input, kernel, lengths, left_padding in sink:
        sink_arrays.append(("states_all", states_all))
        sink_arrays.append(("conv_input", conv_input))
        assert kernel == K and lengths is None and left_padding is None
    fired = {
        name: after.get(name, 0) - before.get(name, 0)
        for name in ("gdn_prework", "rms_norm_gated", "gdn_metal")
    }
    return h, arrays, sink_arrays, fired


def _with_port(fn):
    model = _build_model()
    counts = fused.enable_gdn_port(model)
    try:
        assert counts["prework"] == _n_linear(model), counts
        assert counts["norm"] == _n_linear(model), counts
        return fn(model)
    finally:
        fused.disable_gdn_port(model)


def test_capture_module_is_bit_identical():
    """写し vs モジュール呼び出しで、出力・キャッシュ・sink がビット一致。"""
    if not GPU:
        return

    def body(model):
        n_lin = _n_linear(model)
        for S in (1, 2, 4, 8):
            h0, c0, s0, f0 = _run(model, S, capture_module="0", staged_verify="0")
            h1, c1, s1, f1 = _run(model, S, capture_module="1", staged_verify="0")
            assert bool(mx.array_equal(h0, h1)), f"S={S}: 隠れ状態がビット一致しない"
            _same(c0, c1, f"S={S} cache")
            _same(s0, s1, f"S={S} sink")
            # 写しの経路は前処理カーネルを素通りする (出力 norm だけ当たる)
            assert f0["gdn_prework"] == 0, f"S={S}: 写しで gdn_prework が発火した"
            assert f0["rms_norm_gated"] == n_lin, f"S={S}: 写しの出力 norm の発火数"
            assert f1["gdn_prework"] == n_lin, f"S={S}: モジュール経路の前処理の発火数"
            assert f1["rms_norm_gated"] == n_lin, f"S={S}: モジュール経路の出力 norm"

    _with_port(body)


def test_rollback_is_bit_identical():
    """巻き戻し後のキャッシュもビット一致 (sink の意味が同じことの検査)。"""
    if not GPU:
        return

    def body(model):
        for S, consumed in ((2, 1), (4, 1), (4, 3), (8, 5)):
            _, c0, _, _ = _run(model, S, capture_module="0", staged_verify="0",
                               consumed=consumed)
            _, c1, _, _ = _run(model, S, capture_module="1", staged_verify="0",
                               consumed=consumed)
            _same(c0, c1, f"S={S} consumed={consumed} rollback")

    _with_port(body)


def test_staged_verify_is_bit_identical():
    """検証フォワードの段階投入は値を変えない (両方の capture 経路で)。"""
    if not GPU:
        return

    def body(model):
        for module in ("0", "1"):
            for S in (2, 4):
                h0, c0, s0, _ = _run(model, S, capture_module=module,
                                     staged_verify="0")
                h1, c1, s1, _ = _run(model, S, capture_module=module,
                                     staged_verify="1")
                assert bool(mx.array_equal(h0, h1)), \
                    f"module={module} S={S}: 段階投入で隠れ状態が変わった"
                _same(c0, c1, f"module={module} S={S} staged cache")
                _same(s0, s1, f"module={module} S={S} staged sink")

    _with_port(body)


def test_falls_back_when_port_is_absent():
    """`enable_gdn_port` を当てていなければ knob=1 でも写しに落ちる。"""
    if not GPU:
        return
    model = _build_model()
    n_lin = _n_linear(model)
    _, _, _, fired = _run(model, 4, capture_module="1", staged_verify="0")
    assert fired["gdn_prework"] == 0
    assert fired["rms_norm_gated"] == 0
    assert n_lin == 3


def test_capture_ready_rejects_prefill_width():
    """prefill 幅では取り出し口を使わない (幅の契約が外れる)。"""
    if not GPU:
        return

    def body(model):
        engine = SpecEngine(model, mtp=None)
        caches = model.language_model.make_cache()
        x = mx.zeros((1, 64, HIDDEN), dtype=mx.bfloat16)
        assert engine._capture_via_module(x, None, caches) is False
        with _env(MLXTURBO_SPEC_CAPTURE_MODULE="1"):
            engine._capture_module_ok.clear()
            assert engine._capture_via_module(x, None, caches) is False
            engine._capture_module_ok.clear()
            x4 = mx.zeros((1, 4, HIDDEN), dtype=mx.bfloat16)
            assert engine._capture_via_module(x4, None, caches) is True

    _with_port(body)


if __name__ == "__main__":
    test_capture_module_is_bit_identical()
    test_rollback_is_bit_identical()
    test_staged_verify_is_bit_identical()
    test_falls_back_when_port_is_absent()
    test_capture_ready_rejects_prefill_width()
    print("ok")
