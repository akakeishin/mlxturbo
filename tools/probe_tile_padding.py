"""タイルの水増しが本当に時間に効いているかを測る (MoE の札 1 番)。

**有効な行数を 20480 に固定し、水増し率だけを変える。**同じ仕事量で並びだけが違う。

水増しが時間に効くなら、水増し率の順に時間が並ぶ。効かないなら 3 つとも同じ。
**後者なら「タイルの水増し 4.6s」は札ではなくなる。**

温キャッシュのマイクロなので絶対値は見ない。**3 条件の比だけ。**回文順で交互。
"""
import time
import mlx.core as mx

E, N, K = 512, 2560, 640
GROUP, BITS, T = 64, 4, 32
TOTAL = 20480

def pad_factor(counts, t):
    return sum(-(-c // t) * t for c in counts) / sum(counts)

# 水増しが本当に違う 3 つを作る (合計はどれも 20480)
CASES = {
    # 全部 T の倍数 -> 水増しゼロ
    "倍数ぴったり": [32] * 384 + [64] * 128,
    # 均等 40 行 (40 は 32 の倍数でないので 64 に切り上がる)
    "均等 40": [40] * 512,
    # 裾が重い: 511 人が 1 行、1 人が残り全部
    "裾が重い": [1] * 511 + [TOTAL - 511],
}

def bench(counts, reps=15):
    idx = mx.array([e for e, c in enumerate(counts) for _ in range(c)],
                   dtype=mx.uint32)
    M = int(idx.size)
    x = mx.random.normal((M, 1, K)).astype(mx.bfloat16)
    w = mx.random.normal((E, N, K)).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=BITS)
    f = lambda: mx.gather_qmm(x, wq, sc, bi, rhs_indices=idx, transpose=True,
                              group_size=GROUP, bits=BITS, sorted_indices=True)
    mx.eval(f())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(f())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    del x, w, wq, sc, bi, idx
    mx.clear_cache()
    return ts[len(ts) // 2] * 1e3

print(f"合計行数 {TOTAL}  E={E}  N={N}  K={K}  T={T} と仮定")
for k, c in CASES.items():
    assert sum(c) == TOTAL, (k, sum(c))
    print(f"  {k:12s} 水増し率 T=32 {pad_factor(c,32):.3f}  T=64 {pad_factor(c,64):.3f}"
          f"  行数の幅 {min(c)}-{max(c)}")

kinds = list(CASES)
acc = {k: [] for k in kinds}
for k in kinds + kinds[::-1]:
    acc[k].append(bench(CASES[k]))

print("\n条件ごとの平均 (回文順で 2 回ずつ):")
base = sum(acc["倍数ぴったり"]) / 2
for k in kinds:
    ms = sum(acc[k]) / len(acc[k])
    print(f"  {k:12s} {ms:7.2f} ms  倍数ぴったり比 {ms/base:.3f}"
          f"  (水増し率 {pad_factor(CASES[k],32):.3f})")
