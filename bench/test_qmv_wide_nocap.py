"""mlxturbo/kernels/qmv_wide_nocap.py の正しさと性能を確認するスクリプト。

対象は mlx v0.32.2 の qmv_wide カーネル (`quantized.h` の `qmv_wide_impl`) を
タイル5本上限なしで移植した `qmv_wide_nocap`。M(検証幅)=6〜12 で量子化重みの
読み直しを ceil(M/5) 回 -> 1 回に減らすのが狙い (README.md の「m カーブ税の
真因」参照)。

検証1 (正しさ): K=5120, N in {17408, 12288} (mlp_up 相当・attn_q 相当)、
  M in {6, 8, 10, 12}、dtype in {float16, bfloat16} で
  mx.quantized_matmul(transpose=True) との相対誤差 < 2e-3 を確認する
  (実測は 0 = bit-exact。乗算・縮約の順序が元カーネルと同じなため)。
  併せてフォールバック境界 (M=5, M=13) が mx.quantized_matmul と完全一致する
  ことも確認する。

検証2 (性能・簡易): 独立呼び出しの med time は GPU が複数ディスパッチを
  重ねて隠すため、投機デコードのような依存チェーンの実態を表さない
  (README.md 「計測の規律」)。そのため bench/decompose.py の chain_time に
  倣い、出力を次段の入力へ混ぜ込む8段の連鎖を1回の eval にまとめて計測する。
  他プロセスも GPU を使っている可能性があるため各点は15回中央値のみ取り、
  結果は「参考値」として扱う。最終的な絶対値は静かな環境で取り直すこと。

実行: uv run python bench/test_qmv_wide_nocap.py [--json out.json]
"""

import argparse
import json
import time

import mlx.core as mx

from mlxturbo.kernels.qmv_wide_nocap import qmv_wide_nocap

K = 5120
SHAPES = {"mlp_up": 17408, "attn_q": 12288}  # K -> N, bench/skinny_qmm.py と同じ命名
M_VALUES = (6, 8, 10, 12)
DTYPES = (mx.float16, mx.bfloat16)
GROUP_SIZE = 64
BITS = 4

mx.random.seed(0)


def _quantized_matmul_ref(x, wq, scales, biases):
    return mx.quantized_matmul(
        x, wq, scales, biases, transpose=True, group_size=GROUP_SIZE, bits=BITS
    )


def max_rel_err(ours, ref) -> float:
    ours32 = ours.astype(mx.float32)
    ref32 = ref.astype(mx.float32)
    denom = mx.abs(ref32).max().item() + 1e-6
    return mx.abs(ours32 - ref32).max().item() / denom


def check_correctness() -> list:
    """K=5120, N in {17408, 12288}, M in {6,8,10,12}, dtype in {fp16, bf16} で
    qmv_wide_nocap と mx.quantized_matmul の相対誤差を比較する。
    """
    rows = []
    for name, n in SHAPES.items():
        w_by_dtype = {}
        for dtype in DTYPES:
            w = mx.random.normal((n, K)).astype(dtype)
            wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
            w_by_dtype[dtype] = (wq, scales, biases)

        for dtype in DTYPES:
            wq, scales, biases = w_by_dtype[dtype]
            for m in M_VALUES:
                x = (mx.random.normal((m, K)) * 0.1).astype(dtype)
                ref = _quantized_matmul_ref(x, wq, scales, biases)
                ours = qmv_wide_nocap(x, wq, scales, biases, group_size=GROUP_SIZE, bits=BITS)
                mx.eval(ref, ours)
                err = max_rel_err(ours, ref)
                exact = bool(mx.array_equal(ours, ref))
                rows.append(
                    {
                        "shape": name,
                        "K": K,
                        "N": n,
                        "M": m,
                        "dtype": str(dtype),
                        "rel_err": err,
                        "bit_exact": exact,
                    }
                )
                print(
                    f"  [{name} K={K} N={n} dtype={dtype}] M={m}: "
                    f"rel_err={err:.3e} bit_exact={exact}"
                )
                assert err < 2e-3, f"rel_err too large: {name} M={m} dtype={dtype} err={err}"
    return rows


def check_fallback_boundary() -> None:
    """M<6 (M=5) と M>12 (M=13) は mx.quantized_matmul にフォールバックし、
    その結果と完全一致することを確認する (対応範囲外のときの安全網の検証)。
    """
    dtype = mx.bfloat16
    n = SHAPES["mlp_up"]
    w = mx.random.normal((n, K)).astype(dtype)
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)

    for m in (5, 13):
        x = (mx.random.normal((m, K)) * 0.1).astype(dtype)
        ref = _quantized_matmul_ref(x, wq, scales, biases)
        ours = qmv_wide_nocap(x, wq, scales, biases, group_size=GROUP_SIZE, bits=BITS)
        mx.eval(ref, ours)
        exact = bool(mx.array_equal(ours, ref))
        print(f"  [fallback] M={m}: bit_exact_vs_mlx={exact}")
        assert exact, f"fallback path should match mx.quantized_matmul exactly at M={m}"


def chain_time(fn, x0: mx.array, k: int, n: int = 8, reps: int = 15, warmup: int = 5) -> float:
    """出力を次段の入力へ混ぜ込む n 段の連鎖を1回の eval にまとめて計測する
    (bench/decompose.py の chain_time と同じ考え方)。

    独立呼び出しを並べる計測 (bench/skinny_qmm.py の median_time) は GPU が
    複数ディスパッチを重ねて隠すため、投機デコードのような真の依存チェーンの
    レイテンシを表さない。ここでは各段の出力 (M, N) の先頭 K 列を次段の入力
    (M, K) へ混ぜ込むことで、段どうしを本物のデータ依存にする。
    """

    def run():
        h = x0
        for _ in range(n):
            out = fn(h)
            h = out[:, :k] * 0.05 + x0 * 0.01
        mx.eval(h)

    for _ in range(warmup):
        run()
    mx.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] / n


def run_perf_reference(dtype=mx.bfloat16) -> list:
    rows = []
    for name, n in SHAPES.items():
        w = mx.random.normal((n, K)).astype(dtype)
        wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)

        for m in M_VALUES:
            x0 = (mx.random.normal((m, K)) * 0.1).astype(dtype)

            t_mlx = chain_time(lambda h: _quantized_matmul_ref(h, wq, scales, biases), x0, K)
            t_ours = chain_time(
                lambda h: qmv_wide_nocap(h, wq, scales, biases, group_size=GROUP_SIZE, bits=BITS),
                x0,
                K,
            )
            ratio = t_mlx / t_ours if t_ours > 0 else float("nan")
            rows.append(
                {
                    "shape": name,
                    "N": n,
                    "M": m,
                    "dtype": str(dtype),
                    "mlx_ms_per_stage": t_mlx * 1e3,
                    "ours_ms_per_stage": t_ours * 1e3,
                    "ratio_mlx_over_ours": ratio,
                }
            )
            print(
                f"  [{name} N={n}] M={m}: mlx={t_mlx * 1e3:.4f}ms "
                f"ours={t_ours * 1e3:.4f}ms ratio(mlx/ours)={ratio:.2f}x  (参考値)"
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    if not mx.metal.is_available():
        print("Metal not available; skipping (this kernel is GPU-only).")
        return

    print("=== 検証1: 正しさ (K=5120, N in {17408, 12288}, M in {6,8,10,12}) ===")
    correctness = check_correctness()
    check_fallback_boundary()

    print()
    print("=== 検証2: 性能 (参考値、依存チェーン8段・各点15回中央値) ===")
    print(
        "  注意: 他プロセスも GPU を使っている可能性がある。ここでの数値は参考値。"
        "最終計測は静かな環境で取り直すこと。"
    )
    perf = run_perf_reference()

    print()
    print("すべての検証を通過しました。")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"correctness": correctness, "perf_reference": perf}, f, indent=2)


if __name__ == "__main__":
    main()
