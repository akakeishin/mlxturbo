"""細い m 向け quantized matmul カーネルの試作と検証。

out[M,N] = x[M,K] @ dequant(W[N,K])^T を、1 simdgroup = 1 出力列で計算する。
重み(4bit group-64 affine)は m に依らず 1 回だけ読み、m 本のアキュムレータへ
同時に流す。mx.quantized_matmul の m>=4 のタイル税(1.7-2.3 倍)を消すのが狙い。
"""

import argparse
import json
import time

import mlx.core as mx

_SOURCE = """
    uint lane = thread_position_in_grid.x;
    uint n0 = thread_position_in_grid.y * NB;

    float acc[M][NB];
    for (int m = 0; m < M; m++)
        for (int nb = 0; nb < NB; nb++) acc[m][nb] = 0.0f;

    for (uint wi = lane; wi < K / 8; wi += 32) {
        // NB 列ぶんの重み 8 個を先に展開し、m ループで使い回す
        float wv[NB][8];
        for (int nb = 0; nb < NB; nb++) {
            uint n = n0 + nb;
            uint32_t packed = w[n * (K / 8) + wi];
            float scale = (float)scales[n * (K / 64) + wi / 8];
            float bias = (float)biases[n * (K / 64) + wi / 8];
            for (int j = 0; j < 8; j++) {
                wv[nb][j] = scale * (float)((packed >> (4 * j)) & 0xF) + bias;
            }
        }
        uint k0 = wi * 8;
        for (int m = 0; m < M; m++) {
            const device half4* xv = (const device half4*)(x + m * K + k0);
            half4 xa = xv[0];
            half4 xb = xv[1];
            float xf[8] = {(float)xa.x, (float)xa.y, (float)xa.z, (float)xa.w,
                           (float)xb.x, (float)xb.y, (float)xb.z, (float)xb.w};
            for (int nb = 0; nb < NB; nb++) {
                float s = 0.0f;
                for (int j = 0; j < 8; j++) s += wv[nb][j] * xf[j];
                acc[m][nb] += s;
            }
        }
    }
    for (int m = 0; m < M; m++) {
        for (int nb = 0; nb < NB; nb++) {
            float total = simd_sum(acc[m][nb]);
            if (lane == 0) {
                out[m * N + n0 + nb] = (T)total;
            }
        }
    }
"""

_kernel = mx.fast.metal_kernel(
    name="skinny_qmm",
    input_names=["x", "w", "scales", "biases"],
    output_names=["out"],
    source=_SOURCE,
)


def skinny_qmm(x, wq, scales, biases, n, k, nb=2):
    m = x.shape[0]
    return _kernel(
        inputs=[x, wq, scales, biases],
        template=[("T", x.dtype), ("M", m), ("K", k), ("N", n), ("NB", nb)],
        grid=(32, n // nb, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]


def median_time(fn, reps=30):
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
    ap.add_argument("--widths", default="1,2,4,8,16")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]

    shapes = {"mlp_up": (5120, 17408), "attn_q": (5120, 12288), "lm_head": (5120, 248320)}
    results = {}
    for name, (k, n) in shapes.items():
        w = mx.random.normal((n, k)).astype(mx.float16)
        wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
        entry = {}
        for m in widths:
            x = (mx.random.normal((m, k)) * 0.1).astype(mx.float16)
            ref = mx.quantized_matmul(
                x, wq, scales, biases, transpose=True, group_size=64, bits=4
            )
            ours = skinny_qmm(x, wq, scales, biases, n, k)
            mx.eval(ref, ours)
            denom = mx.abs(ref).max().item() + 1e-6
            err = (mx.abs(ours - ref).max().item()) / denom
            t_ref = median_time(
                lambda: mx.eval(
                    mx.quantized_matmul(
                        x, wq, scales, biases, transpose=True, group_size=64, bits=4
                    )
                )
            )
            t_ours = median_time(lambda: mx.eval(skinny_qmm(x, wq, scales, biases, n, k)))
            entry[m] = {
                "rel_err": err,
                "mlx_ms": t_ref * 1e3,
                "ours_ms": t_ours * 1e3,
                "win": t_ref / t_ours,
            }
        results[name] = entry
        print(name, json.dumps(entry, indent=2))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
