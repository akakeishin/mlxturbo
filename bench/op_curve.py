"""L0: m カーブの犯人捜し。

検証幅 m を増やしたときのコスト増(m=8 で 2.33 倍)が、どの op から
来ているかを層単位と素の quantized_matmul 単位で切り分ける。
"""

import argparse
import json
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def median_time(fn, reps=20):
    fn()
    mx.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lmstudio-community/Qwen3.8-27B-MLX-4bit")
    ap.add_argument("--widths", default="1,2,4,8,16")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]

    model, tokenizer = load(args.model)
    inner = model.language_model.model
    lin = next(l for l in inner.layers if l.is_linear)
    fa = next(l for l in inner.layers if not l.is_linear)
    lm_head = model.language_model.lm_head

    # 文脈 512 のキャッシュを作る
    caches = make_prompt_cache(model)
    prompt = mx.array(tokenizer.encode("システム設計。" * 400)[:512])
    mx.eval(model(prompt[None], cache=caches))
    fa_cache = caches[inner.fa_idx]

    D = 5120
    results = {"layer": {}, "qmm": {}, "sdpa": {}}

    # --- 層単位: linear 層(キャッシュなし・状態は使い回さない)と full-attn 層 ---
    for m in widths:
        x = mx.random.normal((1, m, D)).astype(mx.float16)

        t_lin = median_time(lambda: mx.eval(lin(x, mask=None, cache=None)))

        off0 = fa_cache.offset

        def fa_step():
            fa_cache.offset = off0
            mx.eval(fa(x, mask=None if m == 1 else "causal", cache=fa_cache))

        t_fa = median_time(fa_step)
        fa_cache.offset = off0

        t_head = median_time(lambda: mx.eval(lm_head(x)))

        results["layer"][m] = {
            "linear_ms": t_lin * 1e3,
            "full_attn_ms": t_fa * 1e3,
            "lm_head_ms": t_head * 1e3,
            "model48lin16fa_ms": (48 * t_lin + 16 * t_fa + t_head) * 1e3,
        }

    # --- 素の quantized_matmul: 代表 shape ---
    shapes = {
        "mlp_up": (D, 17408),
        "mlp_down": (17408, D),
        "attn_q": (D, 12288),
        "lm_head": (D, 248320),
    }
    for name, (k, n) in shapes.items():
        w = mx.random.normal((n, k)).astype(mx.float16)
        wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
        curve = {}
        for m in widths:
            x = mx.random.normal((m, k)).astype(mx.float16)
            t = median_time(
                lambda: mx.eval(
                    mx.quantized_matmul(
                        x, wq, scales, biases, transpose=True, group_size=64, bits=4
                    )
                )
            )
            curve[m] = t * 1e3
        base = curve[widths[0]]
        results["qmm"][name] = {
            "ms": curve,
            "ratio": {m: curve[m] / base for m in widths},
        }

    print(json.dumps(results, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
