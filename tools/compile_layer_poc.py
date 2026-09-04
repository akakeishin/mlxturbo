"""層単位 `mx.compile` の PoL (Flash-Next decode、レーン 9)。

**この道具はリポジトリのモジュール (`mlxturbo/`) を編集しない。**包む / 剥がす
を全部この道具の中でやる (別エージェントが同じ decode 経路を編集中のため)。

## 何を調べるか

decode 1 forward は 3,600 本の小さいカーネルで、糊が 12〜15 ms。カーネルを
書かずに dispatch の本数を減らす 3 つ目の手として、`mx.compile` を

  (a) 層の中の elementwise の塊 (HC の読み/書き、MoE の router 頭、GDN 前処理)
  (b) 層まるごと (`DecoderLayer.__call__`)
  (c) 1 forward まるごと (`Qwen4ExpModel.__call__` の層ループ)

の 3 段で当て、短 decode の ms/round が -3% 以上動くかを見る。

## 段

    # 段 1: 可否 (GPU 不要、合成モデル・CPU・数秒)
    .venv/bin/python tools/compile_layer_poc.py feasibility

    # 段 2: dispatch 数 (GPU、decode_gpu_trace の probe を流用)
    BIGLOCK_NO_WORKER=1 BIGLOCK_PRIO=1 tools/biglock.sh \
        .venv/bin/python tools/compile_layer_poc.py dispatch \
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram

    # 段 3: in-model A/B (GPU、decode_ab に knob を動的登録して回文順)
    BIGLOCK_NO_WORKER=1 BIGLOCK_PRIO=1 tools/biglock.sh \
        .venv/bin/python tools/compile_layer_poc.py ab \
        --units hc-read,layer --model ~/models/ddalcu-mlxlm --only short

## 可否の理屈 (段 1 が確かめること)

`mx.compile` は「同じ形の入力で 2 回目以降は python 本体を実行せず、記録した
グラフを再生する」。したがって

- 関数の中の **python レベルの副作用** (`cache[0] = ...`、`cache.advance(S)`、
  `cache.keys = ...`) は**トレース時の 1 回しか起きない**。
- 関数が **closure から読む配列** (キャッシュの中身) はトレース時の値のまま
  焼き込まれる。

decode の層は KV / conv 状態 / indexer / n-gram 文脈のどれも毎ステップ書き換える
ので、(b) と (c) は「純関数に書き直して状態を引数と返り値に出す」ことなしには
成立しないはず。段 1 はそれを合成モデルで**実際に**確かめる (推測で不可と
書かない)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))


# ---------------------------------------------------------------------------
# 包む単位
# ---------------------------------------------------------------------------
#
# 各単位は「patch() -> unpatch()」を返す関数。patch は必ず**その場で**元の
# 属性を控え、unpatch で完全に戻す (A/B を 1 プロセスで交互に回すため)。
#
# `mx.compile` の使い方は 2 通りある:
#   - 重みを引数で渡す自由関数を 1 個だけ compile する (層をまたいで共有、
#     `mlxturbo/fused.py` の HC / _combine がこの形)
#   - インスタンスごとに closure を compile する (層ごとに別の Compiled)
# 前者は Compiled が 1 個で済むが、重みが引数になるぶん入力の木が大きい。
# 後者は素の python に近いが 48 層ぶんの Compiled ができる。両方試す。

_STATS: dict = {}


def _reset_stats() -> None:
    _STATS.clear()
    _STATS["compiles"] = 0   # mx.compile を作った回数 (= Compiled の個数)
    _STATS["traces"] = 0     # 包んだ python 本体が実際に走った回数
    _STATS["calls"] = 0      # 差し替えたメソッドが呼ばれた回数
    _STATS["shapes"] = set()


def _note_compile(shape_key) -> None:
    _STATS["compiles"] = _STATS.get("compiles", 0) + 1
    _STATS.setdefault("shapes", set()).add(shape_key)


def _traced(fn):
    """包む python 本体に「実際に走った回数」の counter を付ける。

    `mx.compile` は同じ形の 2 回目以降は本体を走らせず記録したグラフを
    再生する。したがって **traces < calls** なら、その単位の中の python
    レベルの副作用 (キャッシュへの書き込み) はもう起きていない。"""
    def inner(*a, **kw):
        _STATS["traces"] = _STATS.get("traces", 0) + 1
        return fn(*a, **kw)
    return inner


# --------------------------------------------------------------- (a) 部品


def unit_hc_read(model, shapeless: bool):
    """`GatedResidual.__call__` (HC の読み側) を mx.compile で包む。

    純関数 (hyper だけを読み、状態を書かない)。本番の既定は
    `MLXTURBO_HC=elem` の自前カーネル (decode / verify 幅のみ)。この単位は
    その上から差し替えるので、A/B の意味は「自前カーネル vs mx.compile」。
    """
    import mlx.core as mx
    from mlx_lm.models import qwen4_exp as Q

    orig = Q.GatedResidual.__call__
    cache: dict = {}

    def patched(self, hyper):
        key = (id(self), hyper.shape[:-1] if shapeless else hyper.shape, hyper.dtype)
        fn = cache.get(key)
        if fn is None:
            fn = mx.compile(_traced(lambda h: orig(self, h)), shapeless=shapeless)
            cache[key] = fn
            _note_compile(("hc-read", key[1]))
        _STATS["calls"] = _STATS.get("calls", 0) + 1
        return fn(hyper)

    Q.GatedResidual.__call__ = patched
    return lambda: setattr(Q.GatedResidual, "__call__", orig)


def unit_hc_write(model, shapeless: bool):
    """`DecoderLayer._combine` (HC の書き戻し) を mx.compile で包む。

    本番の既定 (`MLXTURBO_HC_WRITE=1`) が既に mx.compile 版なので、これは
    「同じ手を PoL の枠組みで再現できるか」の対照 (勝ち幅が既知の目印)。
    """
    import mlx.core as mx
    from mlx_lm.models import qwen4_exp as Q

    orig = Q.DecoderLayer._combine
    fn = mx.compile(_traced(lambda h, x, i: orig(h, x, i)), shapeless=shapeless)

    def patched(hyper, x, inject):
        _STATS["calls"] = _STATS.get("calls", 0) + 1
        return fn(hyper, x, inject)

    _note_compile(("hc-write", "shared"))
    Q.DecoderLayer._combine = staticmethod(patched)
    return lambda: setattr(Q.DecoderLayer, "_combine", staticmethod(orig))


def unit_moe_router(model, shapeless: bool):
    """MoE の router 頭 (gate 行列積 -> argpartition -> softmax) を包む。

    `SparseMoeBlock.__call__` の前半だけを差し替える。専門家の行列積
    (`switch_mlp` / decode 幅は `moe_dec_fused`) は境界のままにする。
    """
    import mlx.core as mx
    from mlx_lm.models import qwen4_exp as Q

    orig = Q.SparseMoeBlock.__call__
    cache: dict = {}

    def patched(self, x):
        # r513 / wide / fold の分岐は素のまま (差し替えるのは素の router 頭
        # だけ。本番の decode 幅はこの経路を通る)
        if getattr(self, "_router513", None) is not None:
            return orig(self, x)
        key = (id(self), x.shape[:-1] if shapeless else x.shape, x.dtype)
        fn = cache.get(key)
        if fn is None:
            top_k = self.top_k
            gate = self.gate   # 本番は QuantizedLinear。重みは closure の定数
            # (推論中は変わらないので焼き込んで安全)

            def route(xx):
                logits = gate(xx.astype(mx.float32))
                i = mx.argpartition(-logits, top_k - 1, axis=-1)[..., :top_k]
                ww = mx.softmax(mx.take_along_axis(logits, i, axis=-1),
                                axis=-1, precise=True)
                return i, ww

            fn = mx.compile(_traced(route), shapeless=shapeless)
            cache[key] = fn
            _note_compile(("moe-router", key[1]))
        _STATS["calls"] = _STATS.get("calls", 0) + 1
        idx, w = fn(x)
        min_s = getattr(self, "_combine_fold_min_s", None)
        use_fold = min_s is not None and x.shape[0] * x.shape[1] >= min_s
        if use_fold:
            out = Q._moe_combine_fold(self.switch_mlp, x, idx, w).astype(x.dtype)
        else:
            out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        wide = getattr(self, "_wide_shared", None)
        if wide is None:
            return out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        wq, sc, bi, gs, bits, h = wide
        gus = mx.quantized_matmul(
            x, wq, sc, bi, transpose=True, group_size=gs, bits=bits)
        g, u, sg = gus[..., :h], gus[..., h: 2 * h], gus[..., 2 * h:]
        shared = self.shared_expert.down_proj(mx.nn.silu(g) * u)
        return out + mx.sigmoid(sg) * shared

    Q.SparseMoeBlock.__call__ = patched
    return lambda: setattr(Q.SparseMoeBlock, "__call__", orig)


# ---------------------------------------------- (a/b) ブロックまるごと
#
# 層の中を 3 つのブロックに割ったときの単位。**MoE だけが純関数**
# (`SparseMoeBlock.__call__` はキャッシュを一切触らない) で、GDN と
# Attention はキャッシュを書く。行列積を境に MLX が自動で切るので、
# 「行列積の間の elementwise がまとまるか」を見るならこの粒度が一番大きい
# 純関数の単位になる。


def _wrap_block(cls, attr, n_lead: int, label: str, shapeless: bool):
    """`cls.attr` を「先頭 n_lead 個の引数だけを compile の入力にし、
    残りは closure に閉じ込める」形で包む。"""
    import mlx.core as mx

    orig = getattr(cls, attr)
    fns: dict = {}

    def patched(self, *args):
        lead, rest = args[:n_lead], args[n_lead:]
        key = (id(self),
               tuple((a.shape[:-1] if shapeless else a.shape, a.dtype)
                     for a in lead),
               tuple(a is None for a in rest))
        fn = fns.get(key)
        if fn is None:
            def core(*ll, _rest=rest):
                return orig(self, *ll, *_rest)
            fn = mx.compile(_traced(core), shapeless=shapeless)
            fns[key] = fn
            _note_compile((label, key[1]))
        _STATS["calls"] = _STATS.get("calls", 0) + 1
        return fn(*lead)

    setattr(cls, attr, patched)
    return lambda: setattr(cls, attr, orig)


def unit_moe_block(model, shapeless: bool):
    """`SparseMoeBlock.__call__` をまるごと包む (**純関数**)。"""
    from mlx_lm.models import qwen4_exp as Q
    return _wrap_block(Q.SparseMoeBlock, "__call__", 1, "moe-block", shapeless)


def unit_gdn_block(model, shapeless: bool):
    """`GatedDeltaNet.__call__` をまるごと包む (conv 状態と再帰状態を書く)。"""
    from mlx_lm.models import qwen4_exp as Q
    return _wrap_block(Q.GatedDeltaNet, "__call__", 1, "gdn-block", shapeless)


def unit_attn_block(model, shapeless: bool):
    """`Attention.__call__` をまるごと包む (KV と indexer キャッシュを書く)。"""
    from mlx_lm.models import qwen4_exp as Q
    return _wrap_block(Q.Attention, "__call__", 1, "attn-block", shapeless)


# ------------------------------------------------------- (b) 層まるごと


def unit_layer(model, shapeless: bool):
    """`DecoderLayer.__call__` を層ごとに mx.compile で包む。

    配列でない引数 (rope / cache / idx_cache) は closure に閉じ込め、配列
    (h / mask / ids / prev_ctx) だけを compile の入力にする。**キャッシュは
    closure なので、2 回目以降は python が走らず状態が更新されない**はず ---
    それを段 1 が実測で確かめる。
    """
    import mlx.core as mx
    from mlx_lm.models import qwen4_exp as Q

    return _wrap_layer(model, shapeless, skip_ple=False)


def unit_layer_nople(model, shapeless: bool):
    """`DecoderLayer.__call__` を **PLE 層以外だけ** 包む。

    PLE 層は n-gram テーブルの参照で GPU->CPU 同期 (`np.array` /
    `np.unique`) を踏むので compile のトレース中に例外になる。それを
    除いた残り (Flash-Next なら 48 層中 43 層) が包めるかを見る単位。
    """
    return _wrap_layer(model, shapeless, skip_ple=True)


def _wrap_layer(model, shapeless: bool, skip_ple: bool):
    import mlx.core as mx
    from mlx_lm.models import qwen4_exp as Q

    orig = Q.DecoderLayer.__call__
    cache_fns: dict = {}
    lm = getattr(model, "model", model)
    ple_ids = {id(lm.layers[i]) for i in getattr(lm, "ple_layers", [])}

    def patched(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx):
        if skip_ple and id(self) in ple_ids:
            return orig(self, h, rope, mask, conv_mask, cache, idx_cache,
                        ids, prev_ctx)
        key = (id(self), h.shape[:-1] if shapeless else h.shape, h.dtype,
               mask is None, conv_mask is None, prev_ctx is None)
        fn = cache_fns.get(key)
        if fn is None:
            def core(hh, mm, ii, pp):
                return orig(self, hh, rope, mm, conv_mask, cache, idx_cache, ii, pp)
            fn = mx.compile(_traced(core), shapeless=shapeless)
            cache_fns[key] = fn
            _note_compile(("layer", key[1]))
        _STATS["calls"] = _STATS.get("calls", 0) + 1
        return fn(h, mask, ids, prev_ctx)

    Q.DecoderLayer.__call__ = patched
    return lambda: setattr(Q.DecoderLayer, "__call__", orig)


# ------------------------------------------------ (c) 1 forward まるごと


def unit_model(model, shapeless: bool):
    """`Qwen4ExpModel.__call__` (層ループ全体) を mx.compile で包む。"""
    import mlx.core as mx
    from mlx_lm.models import qwen4_exp as Q

    orig = Q.Qwen4ExpModel.__call__
    cache_fns: dict = {}

    def patched(self, ids, cache=None, input_embeddings=None):
        key = (id(self), ids.shape[:-1] if shapeless else ids.shape,
               input_embeddings is None)
        fn = cache_fns.get(key)
        if fn is None:
            def core(ii):
                return orig(self, ii, cache=cache, input_embeddings=input_embeddings)
            fn = mx.compile(_traced(core), shapeless=shapeless)
            cache_fns[key] = fn
            _note_compile(("model", key[1]))
        _STATS["calls"] = _STATS.get("calls", 0) + 1
        return fn(ids)

    Q.Qwen4ExpModel.__call__ = patched
    return lambda: setattr(Q.Qwen4ExpModel, "__call__", orig)


def unit_staged(model, shapeless: bool):
    """decode の本番経路 `spec_flash._staged_forward` の層ループを包む。

    `Qwen4ExpModel.__call__` は prefill でしか通らない (decode は
    `_staged_forward`)。層ループ全体を 1 つの compile にすると 2 層ごとの
    `async_eval` が消えるので、そこも含めた差になる。
    """
    import mlx.core as mx

    from mlxturbo import spec_flash as SF

    orig = SF._staged_forward
    cache_fns: dict = {}

    def patched(model_, ids, caches):
        key = (id(model_), ids.shape[:-1] if shapeless else ids.shape)
        fn = cache_fns.get(key)
        if fn is None:
            def core(ii):
                return orig(model_, ii, caches)
            fn = mx.compile(_traced(core), shapeless=shapeless)
            cache_fns[key] = fn
            _note_compile(("staged", key[1]))
        _STATS["calls"] = _STATS.get("calls", 0) + 1
        return fn(ids)

    SF._staged_forward = patched

    def undo():
        SF._staged_forward = orig
    return undo


UNITS = {
    # (a) elementwise の塊
    "hc-read": unit_hc_read,
    "hc-write": unit_hc_write,
    "moe-router": unit_moe_router,
    # (a/b) ブロックまるごと
    "moe-block": unit_moe_block,
    "gdn-block": unit_gdn_block,
    "attn-block": unit_attn_block,
    # (b) 層まるごと
    "layer": unit_layer,
    "layer-nople": unit_layer_nople,
    # (c) 1 forward まるごと
    "model": unit_model,
    "staged": unit_staged,
}

TIER = {
    "hc-read": "a", "hc-write": "a", "moe-router": "a",
    "moe-block": "ab", "gdn-block": "ab", "attn-block": "ab",
    "layer": "b", "layer-nople": "b", "model": "c", "staged": "c",
}


# ---------------------------------------------------------------------------
# 段 1: 可否 (合成モデル・CPU)
# ---------------------------------------------------------------------------


def stage_feasibility(args) -> int:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    import mlxturbo  # noqa: F401  (vendor を mlx_lm へ差し込む)
    from verify_batch_cache import build

    print("段 1: 可否 (合成モデル、CPU)。判定は「decode を 6 歩流して"
          "素と同じトークン列が出るか」。\n")

    model = build(4096)
    prompt = [7, 11, 23, 40, 5, 19, 31, 2, 88, 41, 3, 60]

    def run(n=8):
        cache = model.make_cache()
        model(mx.array(prompt[:-1])[None], cache=cache)
        logits = model(mx.array(prompt[-1:])[None], cache=cache)
        out = []
        cur = int(mx.argmax(logits[0, -1]))
        for _ in range(n):
            out.append(cur)
            logits = model(mx.array([[cur]]), cache=cache)
            cur = int(mx.argmax(logits[0, -1]))
        return out

    ref = run()
    print(f"素の出力: {ref}\n")

    rows = []
    names = args.units.split(",") if args.units else list(UNITS)
    for name in names:
        for shapeless in (False, True):
            _reset_stats()
            undo = None
            try:
                undo = UNITS[name](model, shapeless)
                got = run()
                ok = got == ref
                verdict = "一致" if ok else "不一致"
                err = None
            except Exception as e:  # noqa: BLE001
                got, ok, err = None, False, f"{type(e).__name__}: {e}"
                verdict = "例外"
            finally:
                if undo is not None:
                    undo()
            calls = _STATS.get("calls", 0)
            traces = _STATS.get("traces", 0)
            if err is None and calls == 0:
                verdict, ok = "未発火", False   # 包んだが 1 回も呼ばれていない
            rows.append(dict(unit=name, tier=TIER[name], shapeless=shapeless,
                             ok=ok, verdict=verdict, out=got, error=err,
                             compiles=_STATS.get("compiles", 0),
                             calls=calls, traces=traces))
            tag = "shapeless" if shapeless else "形ごと  "
            extra = (f"  compile {_STATS.get('compiles', 0)} 個 / 呼び {calls}"
                     f" / 本体 {traces} 回")
            if err:
                extra = f"  {err[:100]}"
            print(f"{name:<12} ({TIER[name]:>2}) {tag}  {verdict}{extra}")
            if got is not None and not ok:
                print(f"               出力 {got}")

    print()
    ok_units = [r["unit"] for r in rows if r["ok"]]
    ng_units = sorted({r["unit"] for r in rows} - set(ok_units))
    print(f"可: {sorted(set(ok_units))}")
    print(f"不可 / 未検査: {ng_units}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"-> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# 段 2 / 3 の共通: 実モデルを読む
# ---------------------------------------------------------------------------


def _load(args):
    from ab_bundle import load_bundle
    return load_bundle(args.model, ngram_path=args.ngram, mtp_path=args.mtp,
                       mtp_bits=4, log_prefix="[compile-poc]")


def stage_dispatch(args) -> int:
    """decode 1 ラウンドの dispatch 数を単位ごとに数える。"""
    import mlx.core as mx

    from decode_gpu_trace import Probe
    from decode_ab import SHORT_PROMPTS, run_once

    bundle = _load(args)
    eng, tok, eos_ids = bundle.engine, bundle.tokenizer, bundle.eos_ids
    # 深さを固定する (decode_ab --depth 2 + MLXTURBO_DEPTH_ADAPT=0 と同じ)
    eng.depth = 2
    eng.depth_ctx_limit = 1 << 30
    eng._depth_adapt = False

    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": SHORT_PROMPTS[0]}],
        add_generation_prompt=True))[None]

    # burn-in (プロセス最初の 1 本は +7〜9% 遅い / probe の install には
    # MLX が GPU を 1 回使っていることが要る)
    run_once(eng, ids, 32, eos_ids)

    probe = Probe()
    if not probe.install():
        print(f"probe を入れられない: {probe.error}")
        return 1

    names = args.units.split(",") if args.units else ["hc-read", "layer"]
    rows = []
    # 回文順 (baseline を前後に置いて熱ドリフトを見る)
    for name in ["baseline"] + names + ["baseline"]:
        undo = None
        err = None
        if name != "baseline":
            try:
                undo = UNITS[name](eng.model, args.shapeless)
            except Exception as e:  # noqa: BLE001
                print(f"{name:<12} patch 失敗 {type(e).__name__}: {e}")
                continue
        try:
            _reset_stats()
            # 温め兼 compile の固定費 (この 1 本目に全部乗る)
            t_warm = time.perf_counter()
            run_once(eng, ids, 16, eos_ids)
            warm_s = time.perf_counter() - t_warm
            n_compiles_warm = _STATS.get("compiles", 0)
            probe.reset()
            probe.enable(True)
            out, _pf, dec_s, accepted, rounds = run_once(
                eng, ids, args.tokens, eos_ids)
            probe.enable(False)
            probe.quiesce()
            st = probe.stats()
        except Exception as e:  # noqa: BLE001
            probe.enable(False)
            err = f"{type(e).__name__}: {e}"
            print(f"{name:<12} 実行で落ちた -> 不可  {err[:120]}")
            rows.append(dict(unit=name, error=err))
            continue
        finally:
            if undo is not None:
                undo()
        rows.append(dict(
            unit=name, rounds=rounds, accepted=accepted,
            dispatch_per_round=st["dispatches"] / max(rounds, 1),
            cb_per_round=st["command_buffers"] / max(rounds, 1),
            ms_per_round=dec_s * 1000 / max(rounds, 1),
            gpu_union_ms=st["gpu_union_ms"], warm_s=warm_s,
            compiles_warm=n_compiles_warm,
            compiles=_STATS.get("compiles", 0),
            calls=_STATS.get("calls", 0), traces=_STATS.get("traces", 0),
            head=out[:16]))
        r = rows[-1]
        print(f"{name:<12} dispatch/round {r['dispatch_per_round']:8.1f}"
              f"  cb/round {r['cb_per_round']:6.1f}"
              f"  ms/round {r['ms_per_round']:7.2f}"
              f"  compile {r['compiles']} 個 (呼び {r['calls']} / 本体"
              f" {r['traces']})  温め {warm_s:5.2f}s  head {out[:6]}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"-> {args.out}")

    if args.ab_after:
        print("\n" + "=" * 72)
        print("同じプロセスで続けて in-model A/B (decode_ab のハーネスをそのまま使う)")
        print("=" * 72 + "\n")
        return _run_ab_with_bundle(args, args.ab_after.split(","), bundle)
    return 0


def _run_ab_with_bundle(args, names, bundle) -> int:
    """`decode_ab.run_with_model` に knob を動的登録して回す。

    98GB の読み込みを 1 回で済ませるため、段 2 と同じプロセスから呼ぶ。
    ハーネス (回文順 A,B,B,A / 温め / 熱の扱い) は decode_ab のものをそのまま
    使う --- **decode_ab.py 自体は 1 行も編集しない**。
    """
    import mlx.core as mx

    import decode_ab
    from decode_ab import SHORT_PROMPTS, run_once

    eng, tok, eos_ids = bundle.engine, bundle.tokenizer, bundle.eos_ids
    probe_ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": SHORT_PROMPTS[0]}],
        add_generation_prompt=True))[None]

    knob_names = []
    for n in names:
        if n not in UNITS:
            print(f"unit {n!r} は無い (候補: {', '.join(UNITS)})")
            return 2
        # **A/B に入れる前に 1 本流して確かめる。**壊れる単位が 1 つでも
        # 混ざると decode_ab の走行ごと落ちて、健全な単位の測定まで失われる
        # (2026-09-04 12:20 に moe-router のバグで実際に失われた)。
        undo = None
        try:
            undo = UNITS[n](eng.model, args.shapeless)
            run_once(eng, probe_ids, 8, eos_ids)
            ok = True
        except Exception as e:  # noqa: BLE001
            print(f"compile-{n}: 事前確認で落ちた -> A/B から外す "
                  f"({type(e).__name__}: {str(e)[:90]})")
            ok = False
        finally:
            if undo is not None:
                undo()
        if not ok:
            continue
        kn = f"compile-{n}"
        decode_ab.KNOBS[kn] = (_make_knob(n, args.shapeless), ["A", "B"],
                               False, "B")
        knob_names.append(kn)

    if not knob_names:
        print("A/B に回せる単位が無い")
        return 1

    argv = ["--knobs", ",".join(knob_names), "--model", args.model,
            "--only", args.ab_only, "--ctx", str(args.ab_ctx),
            "--tokens", str(args.ab_tokens),
            "--depth", "2", "--no-burn-in"]
    if args.ngram:
        argv += ["--ngram", args.ngram]
    if args.out:
        argv += ["--out", str(Path(args.out).with_name(
            Path(args.out).stem + "-ab.json"))]
    return decode_ab.run_with_model(argv, bundle)


# ---------------------------------------------------------------------------
# 段 3: in-model A/B (decode_ab に knob を動的登録)
# ---------------------------------------------------------------------------


def _make_knob(unit_name: str, shapeless: bool):
    def setup(ctx):
        eng = ctx["eng"]
        state: dict = {}

        def apply(variant):
            undo = state.pop("undo", None)
            if undo is not None:
                undo()
            if variant == "B":
                return
            state["undo"] = UNITS[unit_name](eng.model, shapeless)

        return apply
    return setup


def stage_ab(args, rest) -> int:
    """`tools/decode_ab.py` の KNOBS に `compile-<unit>` を差し込んで走らせる。

    decode_ab.py 本体は 1 行も編集しない (KNOBS は普通の dict なので、
    import した側から足せる)。モデルはこちらで読んで `run_with_model` に渡す。
    """
    if not args.model:
        print("--model が要る")
        return 2
    bundle = _load(args)
    names = args.units.split(",") if args.units else ["moe-block"]
    args.ab_tokens = args.tokens
    # 読み込み直後の 1 本目は +7〜9% 遅い (decode_ab.burn_in の docstring)。
    # `_run_ab_with_bundle` の事前確認 (単位ごとに 1 本) がその役目も兼ねるが、
    # 素の状態で 1 本焼いてから入る。
    from decode_ab import SHORT_PROMPTS, run_once

    import mlx.core as mx
    ids = mx.array(bundle.tokenizer.apply_chat_template(
        [{"role": "user", "content": SHORT_PROMPTS[0]}],
        add_generation_prompt=True))[None]
    run_once(bundle.engine, ids, 32, bundle.eos_ids)
    return _run_ab_with_bundle(args, names, bundle)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("feasibility", "dispatch", "ab"))
    ap.add_argument("--units", default=None,
                    help=f"カンマ区切り ({', '.join(UNITS)})")
    ap.add_argument("--shapeless", action="store_true",
                    help="mx.compile(shapeless=True) で包む")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--ab-after", default=None,
                    help="段 2 の後、同じプロセスで続けて in-model A/B を回す "
                         "(カンマ区切りの unit 名)。98GB の読み込みが 1 回で"
                         "済む。ハーネスは decode_ab の run_with_model")
    ap.add_argument("--ab-tokens", type=int, default=512,
                    help="--ab-after の生成長 (既定 512)")
    ap.add_argument("--ab-only", default="short", choices=("short", "long", "both"),
                    help="A/B の長さ (既定 short)。**`--prefill-once` は使えない**"
                         " -- moe-block / layer 系は prefill 幅でも compile が"
                         "掛かるので、prefill を 1 回に畳むと比較が成立しない")
    ap.add_argument("--ab-ctx", type=int, default=17000,
                    help="--ab-only long のときの文脈長 (既定 17000)")
    ap.add_argument("--out", default=None)
    args, rest = ap.parse_known_args()

    if args.stage == "feasibility":
        return stage_feasibility(args)
    if args.stage == "dispatch":
        if not args.model:
            ap.error("--model が要る")
        return stage_dispatch(args)
    return stage_ab(args, rest)


if __name__ == "__main__":
    raise SystemExit(main())
