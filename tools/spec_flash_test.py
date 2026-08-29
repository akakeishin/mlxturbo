"""捕獲と巻き戻しが正しいかを、素の経路との一致で確かめる。

投機は「混ぜて検証し、外れたら捨てる」なので、**捨てた後の状態が
「最初から入れなかった場合」と一致していなければならない**。ここが狂うと
受理率ではなく出力そのものが壊れる (しかも静かに壊れる)。

  A: forward([a]) -> forward([c])
  B: forward([a, b]) を捕獲つきで -> keep=1 に巻き戻し -> forward([c])

A と B の logits が一致すれば、巻き戻しは正しい。

    tools/biglock.sh uv run python tools/spec_flash_test.py \\
        --model ~/models/qwen38fn-mlx-v-l --ngram ~/models/qwen38fn-ngram-4bit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    args = ap.parse_args()

    import os

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    from mlxturbo import spec_flash

    model, tok = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)

    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて説明してください。"}],
        add_generation_prompt=True))[None]
    S = ids.shape[1]
    prompt, a, b, c = ids[:, : S - 3], ids[:, S - 3: S - 2], ids[:, S - 2: S - 1], ids[:, S - 1:]

    def rel(x, y):
        x, y = np.array(x).ravel(), np.array(y).ravel()
        return float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-12))

    # --- 経路 A: b を通さない
    ca = model.make_cache()
    model(prompt, cache=ca)
    model(a, cache=ca)
    la = model(c, cache=ca).astype(mx.float32)
    mx.eval(la)

    # --- 捕獲版が素と一致するか (b まで通した状態で比べる)
    cs = model.make_cache()
    model(prompt, cache=cs)
    ref_ab = model(mx.concatenate([a, b], axis=1), cache=cs).astype(mx.float32)
    mx.eval(ref_ab)

    cb = model.make_cache()
    model(prompt, cache=cb)
    pre = spec_flash.snapshot_pre(model, cb)
    with spec_flash.capture(model) as cap:
        got_ab = model(mx.concatenate([a, b], axis=1), cache=cb).astype(mx.float32)
        mx.eval(got_ab)
    print(f"  捕獲版 vs 素 (a,b の logits)     rel = {rel(ref_ab, got_ab):.3e}")
    print(f"  hyper を捕まえたか: {cap.hyper is not None}  "
          f"GDN {len(cap.gdn)} 層 / PLE {len(cap.ple)} 層")

    # --- 経路 B: b を捨てて c を通す
    spec_flash.rollback(model, cb, cap, pre, keep=1, total=2, ids_kept=a)
    lb = model(c, cache=cb).astype(mx.float32)
    mx.eval(lb)
    r = rel(la, lb)
    print(f"  巻き戻し後 vs 素 (c の logits)   rel = {r:.3e}"
          f"   {'一致' if r < 1e-5 else '**不一致**'}")
    if r < 1e-5:
        return

    # **対照**: 巻き戻しを一切使わず、バッチ幅だけを変えて比べる。
    # mx.quantized_matmul はバッチ長依存の丸めをする (spec.py の注記) ので、
    # 幅 1 と幅 2 では position a の値がそもそも違う。ここで同じ桁の差が
    # 出るなら、上の不一致は巻き戻しの誤りではなく丸めの差。
    cp = model.make_cache()
    model(prompt, cache=cp)
    model(a, cache=cp)
    lp = model(b, cache=cp).astype(mx.float32)
    cq = model.make_cache()
    model(prompt, cache=cq)
    lq = model(mx.concatenate([a, b], axis=1), cache=cq)[:, -1:].astype(mx.float32)
    mx.eval(lp, lq)
    ctrl = rel(lp, lq)
    print(f"\n  対照: 幅1x2 vs 幅2 (巻き戻し無し) rel = {ctrl:.3e}")
    print(f"  -> 巻き戻しの差 {r:.3e} が対照と同じ桁なら、原因は丸めであって"
          f"巻き戻しではない")

    # **決定的な試験**: 捨てるトークンの中身を変えても結果が変わらないこと。
    # 両経路とも「幅2 で forward -> keep=1 に巻き戻し -> 幅1 で c」と揃うので
    # バッチ幅の丸めが相殺され、巻き戻しの正しさだけが残る。
    # 巻き戻しが正しければ、捨てた中身は結果に一切影響しない = ビット一致。
    def drop_then_c(dropped):
        cx = model.make_cache()
        model(prompt, cache=cx)
        prex = spec_flash.snapshot_pre(model, cx)
        with spec_flash.capture(model) as capx:
            model(mx.concatenate([a, dropped], axis=1), cache=cx)
        spec_flash.rollback(model, cx, capx, prex, keep=1, total=2, ids_kept=a)
        out = model(c, cache=cx).astype(mx.float32)
        mx.eval(out)
        return out

    alt = mx.array([[int(np.array(b)[0][0]) + 137]])   # 別のトークンを捨てる
    d1, d2 = drop_then_c(b), drop_then_c(alt)
    rr = rel(d1, d2)
    print(f"\n  捨てる中身を変えた比較        rel = {rr:.3e}"
          f"   {'巻き戻しは正しい' if rr < 1e-6 else '**巻き戻しに誤りがある**'}")

    # どのキャッシュが食い違っているかを層ごとに突き合わせる。
    # 経路 A のキャッシュ (b を通していない) が正解
    print("\n  --- 食い違っているキャッシュ ---")
    ref = model.make_cache()
    model(prompt, cache=ref)
    model(a, cache=ref)
    cb2 = model.make_cache()
    model(prompt, cache=cb2)
    pre2 = spec_flash.snapshot_pre(model, cb2)
    with spec_flash.capture(model) as cap2:
        model(mx.concatenate([a, b], axis=1), cache=cb2)
    spec_flash.rollback(model, cb2, cap2, pre2, keep=1, total=2, ids_kept=a)

    bad = {}
    for i, (layer, rc, gc) in enumerate(zip(model.model.layers, ref, cb2)):
        if layer.layer_type == "full_attention":
            items = [("kv.offset", rc.offset, gc.offset)]
            for nm in ("keys", "values"):
                rv, gv = getattr(rc, nm), getattr(gc, nm)
                if rv is not None and gv is not None:
                    n = rc.offset
                    items.append((f"kv.{nm}", rv[..., :n, :], gv[..., :n, :]))
            rk, gk = rc.indexer.keys, gc.indexer.keys
            if rk is not None and gk is not None:
                items.append(("indexer", rk, gk))
        else:
            items = [(f"arr[{j}]", rc[j], gc[j]) for j in range(4)]
        for nm, rv, gv in items:
            if rv is None and gv is None:
                continue
            if rv is None or gv is None:
                bad.setdefault(nm, []).append((i, "片方 None"))
                continue
            if hasattr(rv, "shape"):
                if rv.shape != gv.shape:
                    bad.setdefault(nm, []).append((i, f"形 {rv.shape} vs {gv.shape}"))
                    continue
                d = rel(rv.astype(mx.float32), gv.astype(mx.float32))
                if d > 1e-6:
                    bad.setdefault(nm, []).append((i, f"rel={d:.2e}"))
            elif rv != gv:
                bad.setdefault(nm, []).append((i, f"{rv} vs {gv}"))
    if not bad:
        print("    キャッシュは全部一致している (原因は別の場所)")
    for nm, lst in bad.items():
        print(f"    {nm:12s} {len(lst)} 層で食い違い  例: 層{lst[0][0]} {lst[0][1]}")


if __name__ == "__main__":
    main()
