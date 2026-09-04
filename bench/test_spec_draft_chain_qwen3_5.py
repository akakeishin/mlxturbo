"""draft chain の同期を外す knob の検査 (``MLXTURBO_SPEC_DRAFT_NOSYNC``、
2026-09-04 から既定 on、``=0`` で従来)。

深さを **引く前に** 位置別受理率の EMA から決め (``_plan_depth``)、各リンクの
argmax を ``.item()`` せず配列のまま次段へ渡す。confidence の ``mx.eval`` と
語彙長の softmax/エントロピーごと無くなる。

検査は 2 段:

1. ``_plan_depth`` は純関数なので、**同じ受理率入力なら ``_gate_depth`` と
   同じ本数**を返す (位置がよく観測されている = AdaEDL の項に重みが乗らない
   領域で厳密に一致する)。
2. 合成 qwen3_5 + 合成 MTP の貪欲生成で、**draft の作り方を変えても出力
   トークン列が変わらない** (投機は速度と受理率だけを変える)。深さも本数も
   変わるので、これが「変わってよいのは速度と受理率だけ」の固定になる。

    tools/biglock.sh .venv/bin/python -m pytest \\
        bench/test_spec_draft_chain_qwen3_5.py -q
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
import mlxturbo  # noqa: E402,F401 -- sys.meta_path フック
from mlx_lm.models import qwen3_5 as Q35  # noqa: E402
from mlxturbo._mlx_compat import KVCache, TextModelArgs  # noqa: E402
from mlxturbo.mtp import MTPModule  # noqa: E402
from mlxturbo.spec import (  # noqa: E402
    ChatSession,
    GATE_EMA_WARMUP,
    GATE_ROLLBACK_COST,
    SpecEngine,
)

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


def _text_config() -> dict:
    return dict(
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


def _build(dtype=mx.float32):
    """合成の qwen3_5 と、その args で組んだ合成 MTP。

    重みは乱数のままで良い (見たいのは「draft の作り方を変えても本体の
    貪欲出力が変わらない」ことなので、draft の質は問わない)。dtype は
    float32 -- bf16 だと検証幅ごとの丸めで本体側の argmax が割れうる。
    """
    mx.random.seed(0)
    model = Q35.Model(Q35.ModelArgs(model_type="qwen3_5", text_config=_text_config()))
    args = TextModelArgs.from_dict(_text_config())
    mtp = MTPModule(args)
    model.set_dtype(dtype)
    mtp.set_dtype(dtype)
    model.eval()
    mtp.eval()
    return model, mtp


# ---------------------------------------------------------------- 1. 深さの決定

def test_plan_depth_matches_gate_on_the_same_acceptance_rates():
    """よく観測された位置 (w_ema -> 1) では、引く前の決定 (`_plan_depth`) と
    引いた後の切り方 (`_gate_depth`) が同じ本数を返す。

    `_gate_depth` の p_d は EMA と AdaEDL の下界の観測数重み付き平均なので、
    観測数を十分に積むと下界の項が消えて EMA だけになる = `_plan_depth` の
    定義そのものになる。エントロピーを 3 通り (完全に自信あり / 中間 /
    自信なし) 振っても本数が動かないことで「同じ受理率入力」を確かめる。
    """
    n_obs = 100_000 * GATE_EMA_WARMUP  # w_ema = n/(n+warmup) ~ 1
    cases = [
        {},                                             # 事前値だけ
        {1: 0.95, 2: 0.90, 3: 0.85, 4: 0.80, 5: 0.7},   # よく通る
        {1: 0.60, 2: 0.10, 3: 0.01},                    # すぐ切れる
        {1: 0.99, 2: 0.99, 3: 0.99, 4: 0.99, 5: 0.99, 6: 0.99, 7: 0.99, 8: 0.99},
    ]
    for ema in cases:
        obs = {d: n_obs for d in range(1, 9)}
        for cap in (1, 2, 3, 5, 8):
            plan = SpecEngine._plan_depth(ema, obs, cap)
            for h in (0.0, 1.0, 5.0):  # AdaEDL の下界を 1.0 / 0.55 / 0.0 にする
                gate = SpecEngine._gate_depth([h] * cap, ema, obs)
                assert plan == gate, (
                    f"ema={ema} cap={cap} entropy={h}: plan={plan} gate={gate}")


def test_record_pos_accept_preserves_legacy_tail_zero_updates():
    """A/B用のmethod抽出後も、出荷policyのEMA更新は従来どおり。"""
    ema = {d: 0.5 for d in range(1, 5)}
    obs = {d: 7 for d in range(1, 5)}
    SpecEngine._record_pos_accept(ema, obs, accepted=1, drafted=4)
    assert abs(ema[1] - 0.6) < 1e-12
    assert abs(ema[2] - 0.4) < 1e-12
    assert abs(ema[3] - 0.4) < 1e-12
    assert abs(ema[4] - 0.4) < 1e-12
    assert obs == {1: 8, 2: 8, 3: 8, 4: 8}


def test_plan_depth_cuts_the_chain_that_the_gate_would_have_thrown_away():
    """事前値 (FastMTP の k=1 70% / k=2 11% / k=3 2%) のままなら、
    cap=8 でも 2 本しか引かない (= 引いてから捨てる 6 本が消える)。"""
    assert SpecEngine._plan_depth({}, {}, 8) == 2
    assert SpecEngine._plan_depth({}, {}, 1) == 1
    assert SpecEngine._plan_depth({}, {}, 0) == 0
    # EMA が実際に答えを動かすこと (対照が死んでいない検査)。減衰の形が
    # 事前値より緩ければ 1 本深くなる。
    assert SpecEngine._plan_depth({1: 0.70, 2: 0.35, 3: 0.15, 4: 0.05}, {}, 8) == 3
    # 注意: 深さは受理率に対して単調ではない -- 閾値
    # h*(1+expected)/(1+d*h) が expected と一緒に上がるので、全位置の受理率が
    # 一様に高いと 1 本で止まる。これは `_gate_depth` から受け継いだ性質
    # (両者は同じ walk) で、この段で触る対象ではない。
    assert SpecEngine._plan_depth({d: 0.95 for d in range(1, 9)}, {}, 8) == 1
    assert SpecEngine._gate_depth(
        [0.0] * 8, {d: 0.95 for d in range(1, 9)},
        {d: 10 ** 6 for d in range(1, 9)}) == 1


# ---------------------------------------------------- 2. 出力列は draft に依らない

def _gen(model, mtp, prompt, *, nosync: str, max_tokens: int = 24):
    engine = SpecEngine(model, mtp=mtp)
    with _env(MLXTURBO_SPEC_DRAFT_NOSYNC=nosync):
        return engine.generate(
            prompt, max_tokens=max_tokens, n_draft=3, max_draft=8,
            lookup_len=16, lookup_ngram=4, temp=0.0,
        )


def test_greedy_output_does_not_depend_on_the_draft():
    """貪欲なら本体の出力は draft に依らない -- 変わるのは受理率と速度だけ。"""
    model, mtp = _build()
    prompt = [(i * 7 + 3) % VOCAB for i in range(32)]
    base = _gen(model, mtp, prompt, nosync="0")
    got = _gen(model, mtp, prompt, nosync="1")
    assert got["tokens"] == base["tokens"], (
        "NOSYNC=1 で出力列が変わった\n"
        f"  base={base['tokens']}\n  got ={got['tokens']}")
    # 「変わってよいところ」が実際に動いていること (対照が死んでいない検査)
    assert got["steps"] > 0 and base["steps"] > 0


def test_mtp_cache_length_does_not_depend_on_the_draft():
    """draft が MTP キャッシュへ積んだ行は、そのラウンドの maint で必ず落ちる。

    落ちなければ MTP の文脈が投機的なトークンで汚れて受理率が崩れる。
    生成の最後のキャッシュ長が knob の両側で同じことで見る。
    """
    model, mtp = _build()
    prompt = [(i * 5 + 1) % VOCAB for i in range(24)]
    lens = []
    for nosync in ("0", "1"):
        engine = SpecEngine(model, mtp=mtp)
        with _env(MLXTURBO_SPEC_DRAFT_NOSYNC=nosync):
            session = ChatSession()
            engine.generate(prompt, max_tokens=24, n_draft=3, max_draft=8,
                            lookup_len=16, lookup_ngram=4, temp=0.0,
                            session=session)
            lens.append(session.mtp_cache.offset)
    assert lens[0] == lens[1], f"MTP キャッシュの長さが knob で変わった: {lens}"


def test_draft_argmax_off_is_the_exact_mtp_head_argmax():
    """rerankを切れば、従来のexact q4 argmaxそのものへ戻る。"""
    model, mtp = _build()
    with _env(MLXTURBO_DRAFT_RERANK="0"):
        engine = SpecEngine(model, mtp=mtp)
    h_mtp = mx.random.normal((1, 1, HIDDEN))
    got = engine._draft_argmax(h_mtp)
    want = mx.argmax(
        engine._head(h_mtp[:, -1:], engine.mtp.norm)[0, -1], axis=-1
    ).reshape(1)
    mx.eval(got, want)
    assert got.tolist() == want.tolist()


def test_draft_argmax_q2_top32_reranks_q4_rows():
    """q2粗選択とq4候補行の再採点を、合成の量子化headで固定する。"""
    model, mtp = _build()
    model.language_model.lm_head = nn.QuantizedLinear.from_linear(
        model.language_model.lm_head, group_size=64, bits=4
    )
    with _env(MLXTURBO_DRAFT_RERANK="1"):
        engine = SpecEngine(model, mtp=mtp)
    assert engine._rerank is not None

    h_mtp = mx.random.normal((1, 1, HIDDEN))
    row = engine.mtp.norm(h_mtp)[:, -1]
    lm = engine.text.lm_head
    cw, cs, cb = engine._rerank
    coarse = mx.quantized_matmul(
        row, cw, scales=cs, biases=cb, transpose=True,
        group_size=64, bits=2, mode="affine",
    )
    top = mx.argpartition(-coarse, 31, axis=-1)[..., :32]
    rows = mx.dequantize(
        lm.weight[top[0]], lm.scales[top[0]], lm.biases[top[0]],
        group_size=64, bits=4, mode="affine",
    )
    scores = row.astype(rows.dtype) @ rows.T
    best = mx.argmax(scores, axis=-1, keepdims=True)
    want = mx.take_along_axis(top, best, axis=-1).reshape(1)
    got = engine._draft_argmax(h_mtp)
    mx.eval(got, want)
    assert got.tolist() == want.tolist()


def _active_cache(cache):
    n = cache.offset
    return cache.keys[..., :n, :], cache.values[..., :n, :]


def _assert_same_cache(a, b, name):
    assert a.offset == b.offset
    for got, want in zip(_active_cache(a), _active_cache(b)):
        diff = mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)))
        mx.eval(diff)
        assert float(diff.item()) == 0.0, f"{name}: cache max_abs={diff.item()}"


def test_retain_first_mtp_repair_matches_full_rebuild_for_all_boundaries():
    """先頭行保持と全再構築で K/V と次 proposal が一致する。

    repair が区別する状態は「この round に MTP 行があるか」と accepted
    prefix の長さだけ。拒否・部分受理・全受理、D7 から lookup へ移った場合、
    accepted EOS の代表値をすべて通す。
    """
    model, mtp = _build()
    engine = SpecEngine(model, mtp=mtp)
    cases = {
        "rejection": (3, 1),
        "partial": (3, 2),
        "full": (3, 4),
        "d7_rejection": (1, 1),
        "d7_lookup_partial": (1, 3),
        "accepted_eos_first": (3, 1),
        "accepted_eos_later": (3, 2),
        "direct_lookup": (0, 3),
    }
    for case_i, (name, (draft_rows, consumed)) in enumerate(cases.items()):
        mx.random.seed(100 + case_i)
        prefix_tokens = mx.array([3, 5, 7, 9])
        prefix_h = mx.random.normal((1, 4, HIDDEN))
        window = mx.array([(11 + case_i + i) % VOCAB for i in range(4)])
        h_last = mx.random.normal((1, 1, HIDDEN))
        hs = mx.random.normal((1, 4, HIDDEN))

        legacy = KVCache()
        retained = KVCache()
        for cache in (legacy, retained):
            engine._mtp_append(prefix_tokens, engine._mtp_base(prefix_h), cache)
        mtp_off0 = legacy.offset
        assert retained.offset == mtp_off0

        if draft_rows:
            draft_h = mx.concatenate(
                [h_last, mx.random.normal((1, draft_rows - 1, HIDDEN))], axis=1
            )
            for cache in (legacy, retained):
                # Production drafting appends one link per MTP call.  Keeping
                # that call shape here is essential to the bit-exact check.
                for i in range(draft_rows):
                    engine._mtp_append(
                        window[i : i + 1],
                        engine._mtp_base(draft_h[:, i : i + 1]),
                        cache,
                    )

        engine._repair_mtp_cache(
            legacy, mtp_off0, window, consumed, h_last, hs,
            reuse_first=False,
        )
        engine._repair_mtp_cache(
            retained, mtp_off0, window, consumed, h_last, hs,
            reuse_first=True,
        )
        _assert_same_cache(retained, legacy, name)

        next_tok = mx.array([(31 + case_i) % VOCAB])
        next_h = hs[:, consumed - 1 : consumed]
        legacy_out = engine._mtp_append(
            next_tok, engine._mtp_base(next_h), legacy
        )
        retained_out = engine._mtp_append(
            next_tok, engine._mtp_base(next_h), retained
        )
        mx.eval(legacy_out, retained_out)
        diff = mx.max(mx.abs(
            legacy_out.astype(mx.float32) - retained_out.astype(mx.float32)
        ))
        assert float(diff.item()) == 0.0, f"{name}: next hidden max_abs={diff.item()}"
        legacy_tok = mx.argmax(engine._head(legacy_out, engine.mtp.norm), axis=-1)
        retained_tok = mx.argmax(engine._head(retained_out, engine.mtp.norm), axis=-1)
        mx.eval(legacy_tok, retained_tok)
        assert legacy_tok.tolist() == retained_tok.tolist(), name
        _assert_same_cache(retained, legacy, name)


# ------------------------------------------- 3. 掃引用の env の口 (h / max_draft)

def _record_depth_calls(name: str):
    """`SpecEngine` の深さ決定 (`_plan_depth` / `_gate_depth`) を包んで、
    呼ばれた引数を記録する。返り値は (記録リスト, 戻す関数)。

    記録は `(位置引数, キーワード引数)` の組。`generate()` は cap を位置で、
    h をキーワードで渡すので、どちらの形でも拾えるようにしておく。
    """
    orig_desc = SpecEngine.__dict__[name]      # classmethod の記述子そのもの
    orig = getattr(SpecEngine, name)           # 呼べる形 (cls 束縛済み)
    calls: list[tuple] = []

    def spy(*a, **kw):
        calls.append((a, kw))
        return orig(*a, **kw)

    setattr(SpecEngine, name, staticmethod(spy))
    return calls, lambda: setattr(SpecEngine, name, orig_desc)


def test_generate_reads_gate_h_and_max_draft_from_env():
    """`MLXTURBO_SPEC_GATE_H` と `MLXTURBO_SPEC_MAX_DRAFT` が generate() の
    入口で読まれ、深さの決定にそのまま渡る。"""
    model, mtp = _build()
    prompt = [(i * 7 + 3) % VOCAB for i in range(32)]
    calls, restore = _record_depth_calls("_plan_depth")
    try:
        engine = SpecEngine(model, mtp=mtp)
        with _env(MLXTURBO_SPEC_DRAFT_NOSYNC="1",
                  MLXTURBO_SPEC_GATE_H="0.05",
                  MLXTURBO_SPEC_MAX_DRAFT="5"):
            engine.generate(prompt, max_tokens=16, n_draft=3, max_draft=8,
                            lookup_len=16, lookup_ngram=4, temp=0.0)
    finally:
        restore()
    assert calls, "_plan_depth が 1 回も呼ばれていない (対照が死んでいる)"
    assert all(kw["h"] == 0.05 for _a, kw in calls), [kw for _a, kw in calls]
    # cap = min(max_draft, 残りトークン数) なので、序盤は上書きした 5 が出る
    caps = [a[2] for a, _kw in calls]
    assert max(caps) == 5, caps


def test_generate_falls_back_to_the_built_in_gate_h_and_the_argument():
    """env が無ければ h は GATE_ROLLBACK_COST、cap は引数の max_draft。
    NOSYNC=0 の側 (`_gate_depth`) にも同じ h が渡る。"""
    model, mtp = _build()
    prompt = [(i * 7 + 3) % VOCAB for i in range(32)]
    for nosync, name in (("1", "_plan_depth"), ("0", "_gate_depth")):
        calls, restore = _record_depth_calls(name)
        try:
            engine = SpecEngine(model, mtp=mtp)
            with _env(MLXTURBO_SPEC_DRAFT_NOSYNC=nosync,
                      MLXTURBO_SPEC_GATE_H=None,
                      MLXTURBO_SPEC_MAX_DRAFT=None):
                engine.generate(prompt, max_tokens=16, n_draft=3, max_draft=6,
                                lookup_len=16, lookup_ngram=4, temp=0.0)
        finally:
            restore()
        assert calls, f"{name} が 1 回も呼ばれていない (対照が死んでいる)"
        assert all(kw["h"] == GATE_ROLLBACK_COST for _a, kw in calls), [
            kw for _a, kw in calls]
        if name == "_plan_depth":
            caps = [a[2] for a, _kw in calls]
            assert max(caps) == 6, caps
