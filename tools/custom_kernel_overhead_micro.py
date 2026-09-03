"""`mx.fast.metal_kernel` の「呼び出し 1 回の費用」をモデル無しで分解する。

## 何を測るのか

融合カーネルは連鎖マイクロで勝つのに in-model の decode で負ける
(HC: 連鎖 41-44 us/回 対 素 140 us/回、なのに in-model は素 23.9 ms /
融合 32.0 ms、`docs/research/SESSION-2026-09-02-CATCHUP.md`)。連鎖は
「組む → eval」を 1 回ずつ直列にするので **CPU 側の構築費と GPU 実行を
足した値**しか出ない。本番の decode は段階投入 (`MLXTURBO_STAGE_EVERY=2`)
で構築と実行を重ねているので、両者のどちらが伸びたかで結論が変わる。

そこでこのスクリプトは eval を挟まずに **構築だけ**の CPU 時間を測る。
遅延グラフなので `k(inputs=..., ...)` の戻り値を捨てれば GPU は 1 回も
走らない。builtin op の構築費と並べれば、custom kernel の per-call CPU 費が
builtin の何倍かが直接出る。

## 節

- `build`   : 構築のみ (eval しない) の CPU us/回。builtin op、custom kernel
              (小さい source / 本番 HC の source)、本番の HC 経路 3 種
              (素の実装 / 融合カーネル / eligible と pack だけ)。
- `source`  : source 文字列の長さを変えたときの構築費。`write_signature` が
              毎回 source を組み直しているなら長さに比例するはず。
- `chain`   : 直列連鎖 (組む → eval を毎回)。既存の連鎖マイクロの再現。
- `contig`  : 非連続入力 (転置 / 最終軸のスライス) を custom kernel に渡した
              ときの費用。`ensure_row_contiguous=True` (既定) は eval 時に
              copy を挿す。
- `staged`  : 本番の段階投入の模倣。1 「層」= HC 1 回 + GPU の埋め草、
              K 層ごとに `mx.async_eval`。構築が隠れる条件と、隠れなくなる
              条件を出す。

## 使い方

    tools/biglock.sh .venv/bin/python tools/custom_kernel_overhead_micro.py
    tools/biglock.sh .venv/bin/python tools/custom_kernel_overhead_micro.py --only build,source

モデルを読まないので数十秒で終わる。GPU は使うので biglock 経由で。
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from mlxturbo.kernels import hyper_connection as hck

HC = 4
D = 2560
HDIM = HC * D
LOWRANK = 320
BITS = 4
GROUP = 64
EPS = 1e-6


# ----------------------------------------------------------------- 補助


def _median_us(fn, n: int, warmup: int = 20) -> float:
    """`fn` を n 回まわしたときの 1 回あたり us。3 本取って中央値。"""
    for _ in range(warmup):
        fn()
    reps = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        reps.append((time.perf_counter() - t0) / n * 1e6)
    return statistics.median(reps)


def _row(name: str, us: float, base: float | None = None) -> None:
    rel = f"  ({us / base:5.1f}x)" if base else ""
    print(f"  {name:<44} {us:8.2f} us/回{rel}")


def _make_gated_residual():
    """本番と同じ形の `GatedResidual` を合成で組む。

    down/up は 4bit 量子化、`block_inject_weight` は bf16 のまま
    (実モデルの 97 層中 96 層がこの状態、CATCHUP 2026-09-02 22:50)。
    """
    import mlx_lm.models.qwen4_exp as Q

    gr = Q.GatedResidual.__new__(Q.GatedResidual)
    nn.Module.__init__(gr)
    gr.hc = HC
    gr.d = D
    gr.hc_norm = Q.RMSNorm(HDIM, group_size=D, eps=EPS)
    gr.hc_norm.weight = mx.random.normal((HDIM,)).astype(mx.bfloat16) * 0.02
    down = nn.Linear(HDIM, LOWRANK, bias=False)
    up = nn.Linear(LOWRANK, HDIM, bias=False)
    down.weight = down.weight.astype(mx.bfloat16)
    up.weight = up.weight.astype(mx.bfloat16)
    gr.input_mix_weight_down = nn.QuantizedLinear.from_linear(
        down, group_size=GROUP, bits=BITS
    )
    gr.input_mix_weight_up = nn.QuantizedLinear.from_linear(
        up, group_size=GROUP, bits=BITS
    )
    inj = nn.Linear(HDIM, HC, bias=False)
    inj.weight = inj.weight.astype(mx.bfloat16)
    gr.block_inject_weight = inj
    mx.eval(gr.parameters())
    return gr


def _hc_packs(gr):
    from mlxturbo import fused as F

    down = F._pack_quantized(gr.input_mix_weight_down)
    up = F._pack_quantized(gr.input_mix_weight_up)
    inject = F._pack_inject_bf16(gr.block_inject_weight)
    return down, up, inject


def _tiny_kernel(source_pad: int = 0):
    """入力 1 / 出力 1 の最小 custom kernel。`source_pad` バイトのコメントで
    source 文字列だけを伸ばせる (中身の仕事は変えない)。"""
    pad = ("// " + "x" * 60 + "\n") * (source_pad // 64) if source_pad else ""
    src = pad + """
        uint elem = thread_position_in_grid.x;
        out[elem] = inp[elem] + T(1);
    """
    return mx.fast.metal_kernel(
        name=f"tiny_{source_pad}",
        input_names=["inp"],
        output_names=["out"],
        source=src,
    )


# ----------------------------------------------------------------- 節: build


def sec_build(n: int) -> dict:
    print("\n== build: 構築だけ (eval しない) の CPU 費用 ==")
    x = mx.random.normal((1, HDIM)).astype(mx.bfloat16)
    y = mx.random.normal((1, HDIM)).astype(mx.bfloat16)
    w1 = mx.random.normal((HDIM,)).astype(mx.bfloat16)
    mx.eval(x, y, w1)

    out = {}

    def b_add():
        _ = x + y

    def b_three():
        _ = mx.sigmoid(x * y + x)

    def b_rms():
        _ = mx.fast.rms_norm(x, w1, EPS)

    def b_qmm():
        _ = mx.quantized_matmul(x, *qw, transpose=True, group_size=GROUP, bits=BITS)

    gr = _make_gated_residual()
    down, up, inject = _hc_packs(gr)
    qw = (down[0], down[1], down[2])

    tiny = _tiny_kernel()
    small = mx.random.normal((1024,)).astype(mx.float32)
    mx.eval(small)

    def b_tiny():
        _ = tiny(
            inputs=[small],
            template=[("T", mx.float32)],
            grid=(1024, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1024,)],
            output_dtypes=[mx.float32],
        )

    # 本番の HC カーネル 2 本を、fused_gated_residual と同じ引数で叩く
    cfg = {
        "hc": HC,
        "d": D,
        "lowrank": LOWRANK,
        "bits": BITS,
        "group_size": GROUP,
        "eps": EPS,
        "inject_kind": "bf16",
    }
    pre, post = hck._get_kernels(cfg)
    n_down_tg = (LOWRANK + hck._ROWS_PER_TG - 1) // hck._ROWS_PER_TG
    words = HDIM * BITS // 32
    split_down = hck._vec4_ok(words, BITS, GROUP) and (words // 4) >= 2
    fold = (not split_down) and hck._fold_inject_ok(HC)
    pre_tg = n_down_tg + (0 if fold else 1)
    post_tg = (D + hck._SIMDGROUPS - 1) // hck._SIMDGROUPS
    t_in = mx.zeros((1, LOWRANK), mx.bfloat16)
    r_in = mx.zeros((1, HC), mx.float32)
    mx.eval(t_in, r_in)

    def b_hc_pre():
        _ = pre(
            inputs=[x, gr.hc_norm.weight, down[0], down[1], down[2], inject[1]],
            template=[("T", mx.bfloat16)],
            grid=(hck._THREADS, pre_tg, 1),
            threadgroup=(hck._THREADS, 1, 1),
            output_shapes=[(1, LOWRANK), (1, HC), (1, HC)],
            output_dtypes=[mx.bfloat16, mx.float32, mx.bfloat16],
        )

    def b_hc_post():
        _ = post(
            inputs=[x, gr.hc_norm.weight, r_in, t_in, up[0], up[1], up[2]],
            template=[("T", mx.bfloat16)],
            grid=(hck._THREADS, post_tg, 1),
            threadgroup=(hck._THREADS, 1, 1),
            output_shapes=[(1, D)],
            output_dtypes=[mx.bfloat16],
        )

    def b_eligible():
        _ = hck.eligible(x, gr.hc_norm.weight, down, up, inject, HC, D)

    from mlxturbo import fused as F

    def b_pack():
        F._pack_quantized(gr.input_mix_weight_down)
        F._pack_quantized(gr.input_mix_weight_up)
        F._pack_inject_bf16(gr.block_inject_weight)

    def b_fused_call():
        _ = hck.fused_gated_residual(
            x, gr.hc_norm.weight, EPS, HC, D, down, up, inject
        )

    plain_call = type(gr).__call__

    def b_plain_call():
        _ = plain_call(gr, x)

    base = _median_us(b_add, n)
    out["builtin_add"] = base
    _row("builtin: x + y (1 op)", base, base)
    for name, fn in [
        ("builtin: sigmoid(x*y+x) (3 op)", b_three),
        ("builtin: mx.fast.rms_norm (1 op)", b_rms),
        ("builtin: quantized_matmul (1 op)", b_qmm),
        ("custom : 最小カーネル (source 80B)", b_tiny),
        ("custom : HC hc_pre (本番、source 8KB)", b_hc_pre),
        ("custom : HC hc_post (本番、source 4.6KB)", b_hc_post),
        ("HC 部品: eligible() だけ", b_eligible),
        ("HC 部品: _pack_quantized x2 + bf16 x1", b_pack),
        ("HC 経路: fused_gated_residual (2 dispatch)", b_fused_call),
        ("HC 経路: 素の GatedResidual.__call__", b_plain_call),
    ]:
        us = _median_us(fn, n)
        out[name] = us
        _row(name, us, base)

    print(
        f"\n  参考: 素の HC 1 回 = {out['HC 経路: 素の GatedResidual.__call__']:.1f} us、"
        f"融合 1 回 = {out['HC 経路: fused_gated_residual (2 dispatch)']:.1f} us、"
        f"差 x97 層 = "
        f"{(out['HC 経路: fused_gated_residual (2 dispatch)'] - out['HC 経路: 素の GatedResidual.__call__']) * 97 / 1000:.2f} ms/forward"
    )
    return out


# ---------------------------------------------------------------- 節: source


def sec_source(n: int) -> dict:
    print("\n== source: source 文字列の長さと構築費 ==")
    small = mx.random.normal((1024,)).astype(mx.float32)
    mx.eval(small)
    out = {}
    for pad in (0, 2048, 8192, 32768, 131072):
        k = _tiny_kernel(pad)

        def fn(k=k):
            _ = k(
                inputs=[small],
                template=[("T", mx.float32)],
                grid=(1024, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(1024,)],
                output_dtypes=[mx.float32],
            )

        # 初回だけ JIT ビルドが走るので warmup で追い出してから測る
        fn()
        mx.eval(mx.zeros(1))
        us = _median_us(fn, n)
        out[pad] = us
        _row(f"source ~{pad // 1024:>4} KB", us)
    return out


# ----------------------------------------------------------------- 節: chain


def sec_chain(n: int, sets: int = 48) -> dict:
    """直列連鎖。`sets` 個の重みを順に回すことでキャッシュの温度を変える。

    既存の `tools/kernel_chain_cost.py` は 1 組の重みを N 回続けて読むので、
    down/up の 3.3MB が SLC に居座ったまま測っている (温キャッシュ)。本番の
    decode は HC 呼び出しの間に MoE と attention が数百 MB を流すので、
    HC が読む重みは毎回 DRAM から来る。`sets` を増やして総量をキャッシュより
    大きくすれば、その状態を連鎖のまま作れる。**CLAUDE.md の「温キャッシュの
    マイクロの絶対値を信じない」がここで効く。**
    """
    print("\n== chain: 直列連鎖 (キャッシュ温度を変えて) ==")
    x = mx.random.normal((1, HDIM)).astype(mx.bfloat16)
    mx.eval(x)
    out = {}

    for tag, n_sets in (("温 (重み 1 組を使い回す)", 1), (f"冷 (重み {sets} 組を巡回)", sets)):
        grs = [_make_gated_residual() for _ in range(n_sets)]
        packs = [_hc_packs(g) for g in grs]
        plain_call = type(grs[0]).__call__
        mb = n_sets * (
            grs[0].input_mix_weight_down["weight"].nbytes
            + grs[0].input_mix_weight_up["weight"].nbytes
        ) / 1e6

        def run_plain():
            h = x
            for i in range(n):
                mixed, _hy, inj = plain_call(grs[i % n_sets], h)
                h = mx.concatenate([mixed] * HC, axis=-1)
            mx.eval(h)

        def run_fused():
            h = x
            for i in range(n):
                g = grs[i % n_sets]
                down, up, inject = packs[i % n_sets]
                mixed, inj = hck.fused_gated_residual(
                    h, g.hc_norm.weight, EPS, HC, D, down, up, inject
                )
                h = mx.concatenate([mixed] * HC, axis=-1)
            mx.eval(h)

        print(f"  -- {tag} (重み {mb:.0f} MB)")
        for name, fn in [("素の op 列", run_plain), ("融合カーネル 2 本", run_fused)]:
            fn()
            reps = []
            for _ in range(3):
                t0 = time.perf_counter()
                fn()
                reps.append((time.perf_counter() - t0) / n * 1e6)
            us = statistics.median(reps)
            out[f"{tag} / {name}"] = us
            _row(f"連鎖 {name}", us)
        del grs, packs
        mx.clear_cache()
    return out


# ---------------------------------------------------------------- 節: contig


def sec_contig(n: int) -> dict:
    print("\n== contig: 非連続入力 (ensure_row_contiguous の copy) ==")
    gr = _make_gated_residual()
    down, up, inject = _hc_packs(gr)
    rows = 64
    cont = mx.random.normal((rows, HDIM)).astype(mx.bfloat16)
    big = mx.random.normal((rows, HDIM * 2)).astype(mx.bfloat16)
    tr = mx.random.normal((HDIM, rows)).astype(mx.bfloat16)
    mx.eval(cont, big, tr)
    sliced = big[:, :HDIM]          # 最終軸のスライス -> 行が連続でない
    transposed = tr.T               # 転置 -> row_contiguous でない
    out = {}
    for name, arr in [
        ("連続 (row_contiguous)", cont),
        ("最終軸スライス big[:, :10240]", sliced),
        ("転置 (HDIM, rows).T", transposed),
    ]:
        flags = arr.flags() if hasattr(arr, "flags") else None
        rc = getattr(flags, "row_contiguous", "?") if flags else "?"

        def fn(a=arr):
            mixed, inj = hck.fused_gated_residual(
                a, gr.hc_norm.weight, EPS, HC, D, down, up, inject
            )
            mx.eval(mixed, inj)

        fn()
        reps = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            reps.append((time.perf_counter() - t0) / n * 1e6)
        us = statistics.median(reps)
        out[name] = us
        _row(f"{name} [row_contiguous={rc}]", us)
    return out


# ---------------------------------------------------------------- 節: staged


def sec_staged(layers: int) -> dict:
    """段階投入の模倣。1 「層」= HC 1 回 + GPU の埋め草 (行列積)、
    K 層ごとに `mx.async_eval`。埋め草の大きさを変えると
    「構築が隠れる / 隠れない」の境目が動く。"""
    print("\n== staged: 段階投入 (async_eval every K) の模倣 ==")
    gr = _make_gated_residual()
    down, up, inject = _hc_packs(gr)
    plain_call = type(gr).__call__
    x = mx.random.normal((1, HDIM)).astype(mx.bfloat16)
    out = {}

    # 埋め草: 本番 1 層あたりの「HC 以外の GPU 仕事」を粗く模す行列積。
    # 大きさで GPU 側の余裕を変える。
    fillers = {}
    for tag, k in (("軽 (128x2048x2048)", 2048), ("重 (128x4096x4096)", 4096)):
        a = mx.random.normal((128, k)).astype(mx.bfloat16)
        b = mx.random.normal((k, k)).astype(mx.bfloat16)
        mx.eval(a, b)
        fillers[tag] = (a, b)
    mx.eval(x)

    for ftag, (fa, fb) in fillers.items():
        for K in (0, 2):
            for mode in ("plain", "fused"):

                def run():
                    h = x
                    acc = fa
                    for i in range(layers):
                        if mode == "plain":
                            mixed, _hy, inj = plain_call(gr, h)
                        else:
                            mixed, inj = hck.fused_gated_residual(
                                h, gr.hc_norm.weight, EPS, HC, D, down, up, inject
                            )
                        acc = mx.matmul(acc, fb)
                        # 埋め草を HC の連鎖に繋いで、両者が同じ 1 本の依存鎖に
                        # 乗るようにする ((1,1) をブロードキャストするだけ)
                        h = mx.concatenate([mixed] * HC, axis=-1) + acc[0:1, 0:1]
                        if K and (i + 1) % K == 0:
                            mx.async_eval(h, acc)
                    mx.eval(h, acc)

                run()
                reps = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    run()
                    reps.append((time.perf_counter() - t0) * 1e3)
                ms = statistics.median(reps)
                key = f"{ftag} / async_eval every {K or '-'} / {mode}"
                out[key] = ms
                _row(key, ms * 1000 / layers)
    print("  (上の数字は 1 層あたり us。K=0 は完全直列 = eval を最後に 1 回だけ)")
    return out


# ------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        default="build,source,chain,contig,staged",
        help="走らせる節をカンマ区切りで",
    )
    ap.add_argument("--n", type=int, default=2000, help="build/source の反復")
    ap.add_argument("--chain", type=int, default=200, help="chain の連鎖長")
    ap.add_argument("--chain-sets", type=int, default=48,
                    help="chain の冷キャッシュ側で巡回する重みの組数")
    ap.add_argument("--contig-n", type=int, default=100)
    ap.add_argument("--layers", type=int, default=97, help="staged の層数")
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="既定デバイスを CPU にする。`build`/`source` は eval しないので "
        "GPU を 1 度も踏まず、biglock 無しで走らせられる (GPU を使う "
        "`chain`/`contig`/`staged` はこの指定では選べない)",
    )
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if args.cpu:
        gpu_secs = only & {"chain", "contig", "staged"}
        if gpu_secs:
            ap.error(f"--cpu では {sorted(gpu_secs)} は測れない (eval が要る)")
        mx.set_default_device(mx.cpu)

    print(f"mlx {mx.__version__}  device={mx.default_device()}")
    if "build" in only:
        sec_build(args.n)
    if "source" in only:
        sec_source(args.n)
    if "chain" in only:
        sec_chain(args.chain, args.chain_sets)
    if "contig" in only:
        sec_contig(args.contig_n)
    if "staged" in only:
        sec_staged(args.layers)


if __name__ == "__main__":
    main()
