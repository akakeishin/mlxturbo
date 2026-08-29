"""Flash-Next の MTP ヘッドが本当に t+2 を当てるかを確かめ、合成の仕方を決める。

`mlxturbo/mtp_flash.py` の docstring 参照: `pre_fc_norm_hidden` は 10240 なのに
`fc_hidden` の入力は 2560 で、重みの形だけでは合成が決まらない。**当てられる
方が正解**なので、変種を並べて実データで判定する。

判定の目安:

- 正しい合成なら、t+2 の的中率が「本体が t+1 を当てる率」に近い桁になる
- 間違っていれば的中率はほぼ 0 になる (draft は本体の状態を読み違える)

    tools/biglock.sh uv run python tools/mtp_flash_probe.py \\
        --model ~/models/qwen38fn-mlx-v-l --ngram ~/models/qwen38fn-ngram-4bit \\
        --mtp "~/models/qwen38fn-mtp.safetensors"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROMPTS = [
    "分散システムにおける結果整合性について、具体例を挙げて説明してください。",
    "Write a short paragraph about why quantization matters for local LLM inference.",
    "次の関数の問題点を指摘し、修正版を書いてください:\n\ndef mean(xs):\n    return sum(xs) / len(xs)",
    "機械学習における過学習とは何か、対策とあわせて説明してください。",
    "Explain the difference between latency and throughput in one paragraph.",
    "日本の四季について、それぞれの特徴を簡潔にまとめてください。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--variants", default=",".join(("lane", "mean")),
                    help="カンマ区切り。既に lane が正解と判明しているので絞れる")
    ap.add_argument("--bits", default="0",
                    help="カンマ区切り。0 は bf16 (例 0,8,6,4)")
    args = ap.parse_args()

    import os

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    import mlx_lm.models.qwen4_exp as Q

    from mlxturbo import mtp_flash

    model, tok = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)
    targs = model.args.text

    # 本体の最終 mixer に入る直前の hyper 状態を捕まえる
    captured = []
    orig_call = Q.GatedResidual.__call__
    mixer = model.model.hyper_connection_mixer

    def spy(self, hyper):
        if self is mixer:
            captured.append(hyper)
        return orig_call(self, hyper)

    Q.GatedResidual.__call__ = spy

    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    combos = [(v, int(b)) for b in args.bits.split(",") for v in want]
    results = {}
    print(f"{'変種':6s} {'bits':>5s} {'t+2 的中率':>10s} {'平均 logprob':>12s}")
    for variant, bits in combos:
        quant = {"group_size": 64, "bits": bits} if bits else None
        mtp = mtp_flash.load_flash_mtp(args.mtp, targs, variant=variant,
                                       quantize=quant)
        hit = tot = 0
        base_hit = 0
        lp_sum = 0.0
        for text in PROMPTS:
            ids = mx.array(tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True))[None]
            captured.clear()
            logits = model(ids, cache=model.make_cache())
            mx.eval(logits)
            hyper = captured[-1]
            S = ids.shape[1]
            if S < 4:
                continue
            # 位置 j の入力は (embed(t_{j+1}), hyper_j) で、出力は t_{j+2}
            emb = model.model.embed_tokens(ids[:, 1:])
            hin = hyper[:, :-1]
            cache = Q._AttnCache()
            mask = Q.create_attention_mask(emb, None)
            out = mtp(emb, hin, model.model.rope, mask, cache, cache.indexer)
            lg = model.lm_head(out).astype(mx.float32)
            logp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            pred = mx.argmax(lg, axis=-1)
            tgt = ids[:, 2:S]
            # 正解トークンの対数尤度。argmax より鋭く変種を分ける
            tl = mx.take_along_axis(logp[:, : S - 2], tgt[..., None], axis=-1)[..., 0]
            mx.eval(pred, tl)
            p = np.array(pred)[0][: S - 2]
            t = np.array(ids)[0][2:S]
            hit += int((p == t).sum())
            lp_sum += float(np.array(tl).sum())
            tot += len(t)
            # 参考: 本体が t+1 を当てる率
            bp = np.array(mx.argmax(logits, axis=-1))[0][: S - 1]
            base_hit += int((bp == np.array(ids)[0][1:S]).sum())
        acc, lp = hit / max(tot, 1), lp_sum / max(tot, 1)
        results[(variant, bits)] = (acc, lp, base_hit / max(tot, 1))
        print(f"  {variant:6s} {bits if bits else 'bf16':>5} {acc:10.3f} {lp:12.4f}",
              flush=True)
        del mtp
        mx.clear_cache()

    Q.GatedResidual.__call__ = orig_call
    any_base = next(iter(results.values()))[2]
    print(f"\n参考: 本体の t+1 的中率 {any_base:.3f}")
    best = max(results, key=lambda k: results[k][1])
    print(f"最良 (logprob 基準): 変種={best[0]} bits={best[1] or 'bf16'}  "
          f"的中率 {results[best][0]:.3f} logprob {results[best][1]:.4f}")
    if results[best][0] < 0.15:
        print("**どちらも当たっていない。**合成の仮説が両方とも外れているか、"
              "hyper の取り方 (最終 mixer 直前) が違う。")


if __name__ == "__main__":
    main()
