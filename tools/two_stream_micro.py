"""P1 (低占有率カーネルを別 GPU stream に逃がして重ねる案) の proof-of-life。

## 目的

Apple GPU (M3 Max, MLX 0.32.2) が `mx.new_stream(mx.gpu)` で作った 2 本の
stream の command buffer を実際に同時実行するか (P1 の前提が成立するか) を、
モデルを読まずに測る。

- stream A: 占有率の低い直列カーネルの連鎖。GDN の blocked-seq Metal カーネル
  (`mlxturbo/kernels/gdn_blocked_metal.py` の `gated_delta_blocked_seq`) を
  Flash-Next (qwen4_exp) の実形状 (B=1, T=2048, Hk=16, Hv=48, Dk=Dv=128) で
  `--n` 回、前の呼び出しの最終状態 (state_out) を次の state 入力に渡して
  直列に連ねる。
- stream B: `mx.quantized_matmul` (M=2048, K=2560, N=10240, 4-bit, group_size
  64) を `--n` 回、前の出力の先頭 K 列を次の入力に混ぜて直列に連ねる。

stream A と stream B は互いにデータ依存を持たない (state 連鎖と x 連鎖は
別物)。依存があると MLX がその依存を自動で張って直列化してしまい、
2 stream にした効果を測れなくなるため。

## 計測 4 通り

  1. A 単独 (既定の GPU stream)
  2. B 単独 (既定の GPU stream)
  3. 1 stream で A と B を交互に積む (既定の GPU stream 1 本、命令の発行順は
     A[0], B[0], A[1], B[1], ... だが stream は 1 本)
  4. 2 stream で A を stream 1、B を stream 2 に積む (`with mx.stream(s):`)

ウォームアップ 2 回のあと、4 通りを 1 ラウンドとして `--reps` ラウンド
測る。ラウンドの並び順は奇数ラウンドを正順、偶数ラウンドを逆順にする
(CLAUDE.md 「A/B は 1 プロセス内で交互に測る」を 4 条件へ一般化した ABBA
式のカウンターバランス)。各条件の中央値を取る。

## 判定基準

  - 2 stream / 1 stream >= 0.95 (ほぼ同じ)
    -> MLX がどのみち直列化している。2 stream 化の効果は無い。畳む。
  - 2 stream <= 1 stream - 0.6 * min(A 単独, B 単独)
    -> 明確に重なっている。P1 (低占有率カーネルの stream 退避) に進む価値あり。
  - どちらでもない -> 判定保留 (`--n` を増やすか、再計測)。

## 使い方 (GPU 必須。このファイルを書いた時点では実行していない)

    tools/biglock.sh .venv/bin/python tools/two_stream_micro.py --n 8 \
        --json bench/results/two-stream-micro.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import mlx.core as mx
from mlx_lm.models.gated_delta import compute_g

from mlxturbo.kernels.gdn_blocked_metal import gated_delta_blocked_seq

# Flash-Next (qwen4_exp) の GDN 実形状。gdn_blocked_metal.py の docstring
# ("Dk=128, Dv=128, Hk=16, Hv=48" ) と揃える。
B, T, HK, HV, DK, DV = 1, 2048, 16, 48, 128, 128

# stream B (量子化行列積) の実形状。M=2048 は decode 幅ではなく、GDN 側の
# T=2048 と同程度の「重い」仕事量にして、両 stream の仕事の大きさを揃える。
M, K, N = 2048, 2560, 10240
GROUP_SIZE, BITS = 64, 4


def _make_gdn_inputs(n: int, seed: int):
    """stream A の n 反復ぶんの合成入力を先組みする (q/k/v/a/b は反復ごとに
    新しく作り、state だけを直列に渡す)。q/k は本番と同じく rms_norm 済み。
    """
    mx.random.seed(seed)
    A_log = mx.random.normal((HV,)) * 0.5
    dt_bias = mx.ones((HV,))
    inv = DK**-0.5
    qs, ks, vs, gs, betas = [], [], [], [], []
    for _ in range(n):
        q = mx.random.normal((B, T, HK, DK)).astype(mx.bfloat16)
        k = mx.random.normal((B, T, HK, DK)).astype(mx.bfloat16)
        v = mx.random.normal((B, T, HV, DV)).astype(mx.bfloat16)
        a = mx.random.normal((B, T, HV)).astype(mx.bfloat16)
        b = mx.random.normal((B, T, HV)).astype(mx.bfloat16)
        q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv * mx.fast.rms_norm(k, None, 1e-6)
        qs.append(q)
        ks.append(k)
        vs.append(v)
        gs.append(compute_g(A_log, a, dt_bias))
        betas.append(mx.sigmoid(b))
    return qs, ks, vs, gs, betas


def _make_moe_inputs(n: int, seed: int):
    """stream B の n 反復ぶんの合成入力。重みは 1 回だけ作って使い回す
    (decode で重みが定常・活性が流れるのと同じ形)。
    """
    mx.random.seed(seed + 1000)
    w_full = mx.random.uniform(low=-0.02, high=0.02, shape=(N, K)).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(w_full, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    x0 = mx.random.normal((M, K)).astype(mx.bfloat16)
    mixes = [mx.random.normal((M, K)).astype(mx.bfloat16) for _ in range(n)]
    return wq, sc, bi, x0, mixes


def _gdn_step(q, k, v, g, beta, state):
    _, state = gated_delta_blocked_seq(q, k, v, g, beta, state, block_t=None)
    return state


def _moe_step(x, wq, sc, bi, mix):
    y = mx.quantized_matmul(
        x, wq, sc, bi, transpose=True, group_size=GROUP_SIZE, bits=BITS, mode="affine"
    )
    # 前の出力の一部 (先頭 K 列) を次の入力に混ぜる。最適化で消えないように
    # 毎回新しい乱数 (mix) も混ぜて数値を有界に保つ。
    return (0.1 * y[:, :K] + 0.9 * mix).astype(mx.bfloat16)


def build_a_alone(n: int, seed: int):
    qs, ks, vs, gs, betas = _make_gdn_inputs(n, seed)
    mx.eval(qs, ks, vs, gs, betas)

    def go():
        state = None
        for i in range(n):
            state = _gdn_step(qs[i], ks[i], vs[i], gs[i], betas[i], state)
        return state

    return go


def build_b_alone(n: int, seed: int):
    wq, sc, bi, x0, mixes = _make_moe_inputs(n, seed)
    mx.eval(wq, sc, bi, x0, mixes)

    def go():
        x = x0
        for i in range(n):
            x = _moe_step(x, wq, sc, bi, mixes[i])
        return x

    return go


def build_one_stream(n: int, seed: int):
    """A と B を 1 本の (既定の) GPU stream に交互に積む。"""
    qs, ks, vs, gs, betas = _make_gdn_inputs(n, seed)
    wq, sc, bi, x0, mixes = _make_moe_inputs(n, seed)
    mx.eval(qs, ks, vs, gs, betas, wq, sc, bi, x0, mixes)

    def go():
        state = None
        x = x0
        for i in range(n):
            state = _gdn_step(qs[i], ks[i], vs[i], gs[i], betas[i], state)
            x = _moe_step(x, wq, sc, bi, mixes[i])
        return state, x

    return go


def build_two_stream(n: int, seed: int, stream_a, stream_b):
    """A を stream_a、B を stream_b に積む。A と B の間にデータ依存を
    作らない (state 連鎖と x 連鎖は独立) ので、MLX が依存で直列化する余地は
    無い -- 実際に重なるかどうかは GPU が 2 本の command buffer を同時に
    捌けるかにだけ懸かる。
    """
    qs, ks, vs, gs, betas = _make_gdn_inputs(n, seed)
    wq, sc, bi, x0, mixes = _make_moe_inputs(n, seed)
    mx.eval(qs, ks, vs, gs, betas, wq, sc, bi, x0, mixes)

    def go():
        state = None
        x = x0
        for i in range(n):
            with mx.stream(stream_a):
                state = _gdn_step(qs[i], ks[i], vs[i], gs[i], betas[i], state)
            with mx.stream(stream_b):
                x = _moe_step(x, wq, sc, bi, mixes[i])
        return state, x

    return go


def bench_once(go) -> float:
    t0 = time.perf_counter()
    mx.eval(go())
    return (time.perf_counter() - t0) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="A/B それぞれの連鎖の長さ")
    ap.add_argument("--reps", type=int, default=3, help="ABBA 式カウンターバランスのラウンド数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="結果の書き出し先")
    args = ap.parse_args()

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise SystemExit(
            "GPU が既定デバイスでない。このマイクロは GPU stream の重なりを "
            "測るものなので GPU 専用 (CPU では意味のある結果にならない)"
        )

    stream_a = mx.new_stream(mx.gpu)
    stream_b = mx.new_stream(mx.gpu)

    builders = {
        "A単独": build_a_alone(args.n, args.seed),
        "B単独": build_b_alone(args.n, args.seed),
        "1stream交互": build_one_stream(args.n, args.seed),
        "2stream": build_two_stream(args.n, args.seed, stream_a, stream_b),
    }
    names = list(builders)

    # ウォームアップ (捨てる)
    for name in names:
        for _ in range(2):
            bench_once(builders[name])

    samples: dict[str, list[float]] = {name: [] for name in names}
    for r in range(args.reps):
        order = names if r % 2 == 0 else names[::-1]
        for name in order:
            samples[name].append(bench_once(builders[name]))

    medians = {name: statistics.median(v) for name, v in samples.items()}
    a_ms, b_ms = medians["A単独"], medians["B単独"]
    one_ms, two_ms = medians["1stream交互"], medians["2stream"]

    ratio = two_ms / one_ms
    overlap_bar = one_ms - 0.6 * min(a_ms, b_ms)
    if ratio >= 0.95:
        verdict = "直列化: 2 stream が 1 stream とほぼ同じ。P1 は畳む。"
    elif two_ms <= overlap_bar:
        verdict = "重なっている: 2 stream が 1 stream より明確に速い。P1 に進む価値あり。"
    else:
        verdict = "判定保留: どちらの閾値も満たさない (--n を増やすか再計測)。"

    print(f"n={args.n} reps={args.reps}")
    for name in names:
        print(f"  {name:12s} {medians[name]:9.3f} ms  (samples={samples[name]})")
    print(
        f"判定: 2 stream / 1 stream = {ratio:.2f}"
        f" (0.95 以上なら直列化 = 畳む、{overlap_bar:.3f} ms 以下なら重なっている)"
        f" -> {verdict}"
    )

    if args.json:
        out_dir = os.path.dirname(args.json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(
                {
                    "note": __doc__,
                    "n": args.n,
                    "reps": args.reps,
                    "samples_ms": samples,
                    "median_ms": medians,
                    "ratio_two_over_one": ratio,
                    "overlap_threshold_ms": overlap_bar,
                    "verdict": verdict,
                },
                f,
                ensure_ascii=False,
                indent=1,
            )
        print("書き出し:", args.json)

    os._exit(0)


if __name__ == "__main__":
    main()
