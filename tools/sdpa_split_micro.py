"""decode 幅 S=3..8 の sdpa (head_dim 256、bool マスク) を、幅 2 / 幅 1 の呼び出しに分割した場合と比べる。

MLX 0.32.2 の vector カーネルは `S * gqa_factor > 32` で不適格 (scaled_dot_product_attention.cpp:703)。
Flash-Next (gqa 12) では S >= 3 が素の経路 (スコア実体化、全 kv 読み) に落ちる。
GPU が空いているときに `.venv/bin/python tools/sdpa_split_micro.py` で走らせる。"""
import sys, time, statistics; sys.path.insert(0, __import__("os").path.dirname(__file__))
import mlx.core as mx
from sdpa_headdim_micro import qsa_like_mask
Hq, Hk, D = 24, 2, 256
def run(S, kv, reps=7):
    mask = qsa_like_mask(S, kv); q = mx.random.normal((1, Hq, S, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, Hk, kv, D)).astype(mx.bfloat16); v = mx.random.normal((1, Hk, kv, D)).astype(mx.bfloat16)
    mx.eval(mask, q, k, v); sc = D ** -0.5
    def whole(): return mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=mask)
    def split(w):
        outs = [mx.fast.scaled_dot_product_attention(q[:, :, i:i+w], k, v, scale=sc, mask=mask[:, :, i:i+w]) for i in range(0, S, w)]
        return mx.concatenate(outs, axis=2)
    cases = {"whole": whole, "split2": lambda: split(2), "split1": lambda: split(1)}
    for f in cases.values():
        for _ in range(2): mx.eval(f())
    ref = whole(); mx.eval(ref)
    err = max(float(mx.max(mx.abs(cases[n]().astype(mx.float32) - ref.astype(mx.float32)))) for n in ("split2", "split1"))
    samples = {n: [] for n in cases}
    for r in range(reps):
        for n in (list(cases) if r % 2 == 0 else list(cases)[::-1]):
            t0 = time.perf_counter(); mx.eval(cases[n]()); samples[n].append((time.perf_counter() - t0) * 1e3)
    m = {n: statistics.median(v) for n, v in samples.items()}
    print(f"S={S} kv={kv:6d}  whole {m['whole']:6.2f} ms  split2 {m['split2']:6.2f} ms  split1 {m['split1']:6.2f} ms  (max abs err {err:.2e})", flush=True)
for S in (3, 4, 6, 8):
    for kv in (4096, 17000, 50000):
        run(S, kv)
