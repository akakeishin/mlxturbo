"""投機 1 反復の内訳を測る。深さを上げる価値はここで決まる。

一括 forward は幅が増えてもほぼ同値 (S=16 で S=1 の 1.17 倍、docs/STATUS.md)
なので、深さを増やす費用は **draft 1 回ぶんの MTP forward** に集約される。
draft が本体 forward に対して高いなら、深さを増やしても割に合わない。

    tools/biglock.sh uv run python tools/spec_flash_cost.py \\
        --model ~/models/qwen38fn-mlx-v-l --ngram ~/models/qwen38fn-ngram-4bit \\
        --mtp "~/models/qwen38fn-mtp.safetensors"
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    import os

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    from mlxturbo import fused, mtp_flash, spec_flash

    model, tok = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)
    fused.enable_hyper_connection_kernel()
    q = {"group_size": 64, "bits": args.mtp_bits} if args.mtp_bits else None
    mtp = mtp_flash.load_flash_mtp(args.mtp, model.args.text, quantize=q)
    eng = spec_flash.FlashSpecEngine(model, mtp)

    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて説明してください。"}],
        add_generation_prompt=True))[None]
    caches = model.make_cache()
    with spec_flash.capture(model) as cap:
        lg = model(ids, cache=caches)
        mx.eval(lg)
    hyper = cap.hyper[:, -1:]
    cur = mx.argmax(lg[:, -1], axis=-1).reshape(1, 1)

    def timeit(fn, n):
        for _ in range(3):
            mx.eval(fn())
        out = []
        for _ in range(n):
            t = time.perf_counter()
            mx.eval(fn())
            out.append((time.perf_counter() - t) * 1000)
        return statistics.median(out)

    # 本体 forward: 幅 1 / 2 / 3 / 4 (キャッシュを汚さないよう使い捨て)
    print(f"{'':28s} ms")
    base = None
    for w in (1, 2, 3, 4):
        toks = mx.tile(cur, (1, w))

        def f(toks=toks):
            c = [x for x in caches]  # 同じ参照で良い: 測るのは時間だけ
            return model(toks, cache=None if False else c)

        ms = timeit(lambda toks=toks: model(toks, cache=model.make_cache()), 5)
        if w == 1:
            base = ms
        print(f"  本体 forward 幅{w} (プレフィル込み)  {ms:7.2f}")

    # 実際の反復に近い形: キャッシュ付きの幅 1 と幅 2
    def step(w):
        c = model.make_cache()
        model(ids, cache=c)
        toks = mx.tile(cur, (1, w))
        return timeit(lambda: model(toks, cache=c), args.reps)

    widths = {w: step(w) for w in (1, 2, 3, 4, 5)}
    s1 = widths[1]
    print()
    for w, ms in widths.items():
        extra = "" if w == 1 else f"   (幅1 比 {ms / s1:.2f}x、1 本増の限界費用 {ms - widths[w - 1]:.2f}ms)"
        print(f"  デコード 1 歩 幅{w}              {ms:7.2f}{extra}")

    d = timeit(lambda: eng._draft(cur, hyper), args.reps)
    print(f"\n  **MTP draft 1 回**             {d:7.2f}   (本体 1 歩の {d / s1:.2f} 倍)")

    # lm_head だけの費用 (draft の中身の内訳)
    emb = model.model.embed_tokens(cur)
    Q = spec_flash._arch()
    cc = Q._AttnCache()
    mask = Q.create_attention_mask(emb, None)
    body = timeit(lambda: mtp(emb, hyper, eng.rope, mask, Q._AttnCache(),
                              Q._AttnCache().indexer), args.reps)
    h = mtp(emb, hyper, eng.rope, mask, cc, cc.indexer)
    mx.eval(h)
    head = timeit(lambda: model.lm_head(h), args.reps)
    print(f"    うち MTP 本体                {body:7.2f}")
    print(f"    うち lm_head                 {head:7.2f}")

    # **深さ d は幅 d+1 の検証が要る。**幅 2 で全部を見積もると深さの費用を
    # 過小評価する (最初これを間違えた)
    print(f"\n=== 深さの見積もり (検証は幅 depth+1、draft は depth 回) ===")
    print(f"{'':8s}" + "".join(f"{f'p={p}':>12s}" for p in (0.5, 0.6, 0.74)))
    for depth in (1, 2, 3, 4):
        if depth + 1 not in widths:
            continue
        cost = widths[depth + 1] + depth * d
        row = ""
        for p in (0.5, 0.6, 0.74):
            toks = sum(p ** i for i in range(depth + 1))
            row += f"{cost / toks:12.2f}"
        print(f"  深さ{depth}  {row}   (1 反復 {cost:.2f}ms)")
    print(f"  投機なし{s1:12.2f}{s1:12.2f}{s1:12.2f}")


if __name__ == "__main__":
    main()
