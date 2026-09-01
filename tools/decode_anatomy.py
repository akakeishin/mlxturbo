"""長い文脈での decode 1 ラウンドの内訳を、実キャッシュのまま測る。

## なぜ既存のツールで足りないか

`tools/module_costs.py` は attention を **cache 無し・mask 無し**で叩く
(q/k/v 射影 + 現在の S トークンだけの sdpa)。短文脈ならそれで足りるが、
17k で増えた ~20ms/token の容疑者は「長い KV の読み出し」と「QSA の
ブロック選択」で、どちらも cache 無しでは測れない。

ここでは**本物のキャッシュを持ったまま**部品を呼び、呼ぶ前後で
キャッシュを退避・復元して繰り返す。MLX の配列は不変なので、参照と
offset を控えておけば付け替えで完全に戻せる (`spec_flash._pipeline_snapshot`
と同じ理屈)。

## 内訳の切り方

    indexer   QSA のブロック選択 (pooled の作り直し・rope・einsum・top-k)
    attn      Attention 全体 (indexer 込み)。差し引きで sdpa + 射影が出る
    gdn       線形注意 36 層
    moe       MoE 48 層
    hc        hyper-connection 97 回
    lm_head   出力射影

**部品和 ≈ 壁時計 (数 % 以内) を必ず確認すること** (CLAUDE.md の作法)。
合わなければ、見えていない項目があるということで、その差自体が結論になる。

    tools/biglock.sh .venv/bin/python tools/decode_anatomy.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep --ctx 17000
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def med_ms(fn, reps=7):
    import mlx.core as mx

    fn()
    mx.synchronize() if hasattr(mx, "synchronize") else None
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def snapshot(caches):
    """キャッシュの参照と offset を控える (配列は不変なので付け替えで戻る)。"""
    st = []
    for c in caches:
        if hasattr(c, "keys"):
            st.append(("a", c.keys, c.values, c.offset,
                       c.indexer.keys, c.indexer.offset))
        else:
            st.append(("l", [c[i] for i in range(4)]))
    return st


def restore(caches, st):
    for c, rec in zip(caches, st):
        if rec[0] == "a":
            _, c.keys, c.values, c.offset, ik, io = rec
            c.indexer.keys = ik
            c.indexer.offset = io
        else:
            for i, v in enumerate(rec[1]):
                c[i] = v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--width", type=int, default=2, help="検証フォワードの幅")
    args = ap.parse_args()

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401
    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo.runner import enable_default_fusions

    model, tok = load(os.path.expanduser(args.model))
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[anatomy]")

    files = sorted(REPO_ROOT.glob("docs/**/*.md")) + [REPO_ROOT / "README.md"]
    pool = tok.encode("\n\n".join(f.read_text() for f in files if f.exists()))
    body = tok.decode(pool[: max(args.ctx - 200, 16)])
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True))[None]
    n = ids.shape[1]
    print(f"ctx={n} width={args.width}")

    cache = model.make_cache()
    step = 2048
    for i in range(0, n, step):
        mx.eval(model(ids[:, i : i + step], cache=cache))
        mx.clear_cache()
    print("prefill 済み", flush=True)

    W = args.width
    chunk = mx.array([[int(ids[0, -1].item())] * W])

    grabbed = {"moe": [], "gdn": [], "hc": [], "attn": [], "idx": []}
    o = {
        "moe": Q.SparseMoeBlock.__call__,
        "gdn": Q.GatedDeltaNet.__call__,
        "hc": Q.GatedResidual.__call__,
        "attn": Q.Attention.__call__,
        "idx": Q.QSAIndexer.__call__,
    }

    def wrap(key, fn, nargs):
        def g(self, *a):
            grabbed[key].append((self, *a))
            return fn(self, *a)
        return g

    Q.SparseMoeBlock.__call__ = wrap("moe", o["moe"], 1)
    Q.GatedDeltaNet.__call__ = wrap("gdn", o["gdn"], 3)
    Q.GatedResidual.__call__ = wrap("hc", o["hc"], 1)
    Q.Attention.__call__ = wrap("attn", o["attn"], 5)
    Q.QSAIndexer.__call__ = wrap("idx", o["idx"], 5)
    pre = snapshot(cache)
    mx.eval(model(chunk, cache=cache))
    Q.SparseMoeBlock.__call__ = o["moe"]
    Q.GatedDeltaNet.__call__ = o["gdn"]
    Q.GatedResidual.__call__ = o["hc"]
    Q.Attention.__call__ = o["attn"]
    Q.QSAIndexer.__call__ = o["idx"]
    restore(cache, pre)
    print("捕まえた: " + " ".join(f"{k}={len(v)}" for k, v in grabbed.items()),
          flush=True)

    def bench(key, call):
        def run():
            st = snapshot(cache)
            outs = [call(*t) for t in grabbed[key]]
            mx.eval(outs)
            restore(cache, st)
            return outs
        return med_ms(run)

    res = {
        "moe": bench("moe", lambda s, x: o["moe"](s, x)),
        "hc": bench("hc", lambda s, h: o["hc"](s, h)[0]),
        "gdn": bench("gdn", lambda s, x, m, c: o["gdn"](s, x, m, c)),
        "attn": bench("attn", lambda s, *a: o["attn"](s, *a)),
        "idx": bench("idx", lambda s, *a: o["idx"](s, *a)),
    }
    res["lm_head"] = med_ms(
        lambda: model.lm_head(mx.zeros((1, W, model.args.text.hidden_size),
                                       dtype=mx.bfloat16)))

    def whole():
        st = snapshot(cache)
        out = model(chunk, cache=cache)
        mx.eval(out)
        restore(cache, st)
        return out

    total = med_ms(whole)

    print()
    order = ["moe", "gdn", "attn", "idx", "hc", "lm_head"]
    label = {"moe": "MoE 48 層", "gdn": "GDN 36 層", "attn": "Attention 12 層 (indexer 込み)",
             "idx": "  うち indexer", "hc": "HC 97 回", "lm_head": "lm_head"}
    parts = sum(res[k] for k in order if k != "idx")
    for k in order:
        print(f"  {label[k]:32s} {res[k]:7.2f} ms  ({res[k] / total * 100:5.1f}%)")
    print(f"  {'部品和 (indexer は attn に含む)':32s} {parts:7.2f} ms")
    print(f"  {'壁時計 (フォワード 1 回)':32s} {total:7.2f} ms")
    gap = (parts - total) / total * 100
    print(f"\n  部品和 - 壁時計 = {parts - total:+.2f} ms ({gap:+.1f}%)")
    if abs(gap) > 10:
        print("  ** 数 % を超えてずれている。見えていない項目があるか、"
              "単体計測が重なりを再現できていない **")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
