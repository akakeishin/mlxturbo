"""HC の「読み側」(`GatedResidual.__call__`) を **prefill 幅** で段ごとに測る冷の連鎖。

`tools/micro_kernel_latency.py` の HC 項目は decode 幅 (M=1) 専用で、
prefill (M=2048) の段ごとの内訳が無かった。prefill の内訳
(docs/research/SESSION-2026-09-02-CATCHUP.md の 2026-09-03 21:45) では HC 読みが
4k 11.7% / 8k 11.4% / 17k 10.5% を占め、行列積の下限に対して 63〜65% しか
出ていない。取り分がどの段にあるのかを決めるのがこのファイル。

## 測る段 (実モデルの `GatedResidual.__call__` の並び)

    pre    : hyper -> rms_norm(レーンごと) -> * (1 + norm_weight) = normed
    down   : normed @ down^T            (K=10240 -> N=320、4bit/gs64)
    mid    : silu(down_raw / hc)
    up     : mid @ up^T                 (K=320 -> N=10240、4bit/gs64)
    post   : sigmoid(up_raw) * normed のレーン平均 = mixed
    inject : normed @ inject^T (bf16) -> 2 * sigmoid(x / hc)

各段は **それ自身の 97 段連鎖** として正の側から測る (無効化の積み上げはしない、
CLAUDE.md)。段和と全体の壁時計の差は結果の `sum_check` に出す。

## 連鎖の張り方

- `pre` / `mid` は出力の形が入力と同じなので自然に直列に張れる (x_{i+1} = f(x_i))。
- `down` / `up` / `post` / `inject` は形が変わるので、`--pool` 枚の入力を
  巡回させる。直列でない (MLX が隣の step と重ねられる) 可能性があるので、
  自然に直列な `gemm_pair` (down -> mid -> up) を併せて測り、
  `down + mid + up ≈ gemm_pair` を確認する (結果の `pair_check`)。
- 全体 (`full_*`) は実モデルどおり `_combine` を挟んで hyper を持ち回る。

**重みは 97 組 (367.5 MB) を巡回させて冷やす** (CLAUDE.md の作法)。重みの作り方は
`tools/micro_kernel_latency.py` の `hc_weight_set` (4bit/gs64、inject は bf16) を
そのまま使う。

## 比べる相手

- 段ごと: 素の MLX op 列 vs `kernels/hyper_connection.py` の第 4 変種
  (`hc_elem_pre` / `hc_elem_mid` / `hc_elem_post`)。この 3 本は本番では行数 <= 8
  でしか発火しない (`eligible_elem` の行数ゲート) が、カーネル自体は行数を
  見ないのでここでは prefill 幅でも呼べる。**その prefill 幅への拡大は
  2026-09-04 の in-model で棄却済み** (tok/round が平均 -4.8%)。`elem*` の行は
  「elementwise を畳んだらどこまで縮むか」の上限の記録として残してある。
- 行列積: `mx.quantized_matmul` vs `kernels/qmm_wide.py` の広タイル (P10)。
  こちらは `--knob hc-qmm-wide` として **2026-09-04 に既定 on**。
- 全体 (`full`): `plain` = wide 前の素 / `plain_wide` = 現在の既定 /
  `elem` / `elem_wide` = 棄却した変種。

**孤立して測った段 (`down` / `up` / `post_inject` / `inject`) は pool を巡回させる
ぶん連鎖の依存が切れる**ので、`full` や `gemm_pair` の合成比とは走行間で 10〜20%
食い違う。段ごとの比 (素 対 elem / wide) は走行をまたいでも安定しているが、
合成比は機体の熱で振れる。**採否は in-model で決める** (CLAUDE.md)。

## 数値の検査 (`numerics`)

- `normed` / `mid` / `mixed` / `inject`: elem の各段 対 素の不一致率
- `mixed_model_vs_plain`: 素の `mean(axis=-2)` を bf16 逐次加算で組み直した模型
  対 素。0 なら模型が正しい
- `mixed_model_vs_elem`: その模型 対 カーネル。0 でなければ食い違いはカーネルの
  中 (写した sigmoid か Metal の bf16 積の丸め)
- `qmm_wide`: 広タイルが素とビット一致するか (本番の既定、`fused.enable_hc_qmm_wide`)。
  **`--rows` は本番の行数ゲート (`fused._QMM_WIDE_MIN_ROWS`、既定 1024) 以上で測ること。**
  下回ると down 側が素と食い違う (M=256 で 0.53 / M=512 で 0.373、M>=2048 で 0.0)。
  本番はゲートで届かないので実害は無いが、小さい `--rows` の数字を欠陥と読み違えない
  ように `below_prod_min_rows` を結果に出している

モデルは読まない。

    tools/biglock.sh uv run python tools/hc_prefill_micro.py \
        --out bench/results/hc-prefill-micro.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

WARMUP_SAMPLES = 2      # ABBA の最初の 1 往復は捨てる
N_PAIRS = 4             # ABBA を N_PAIRS 回 -> 各側 2*N_PAIRS 標本


def _summ(samples):
    kept = samples[WARMUP_SAMPLES:] or samples
    gpu = [s[0] for s in kept]
    build = [s[1] for s in kept]
    return {
        "n": len(samples),
        "kept": len(kept),
        "median_us": round(statistics.median(gpu), 2),
        "min_us": round(min(gpu), 2),
        "max_us": round(max(gpu), 2),
        "build_us": round(statistics.median(build), 2),
    }


def _chain(step, state, n_steps, n_sets):
    """n_steps ぶんの連鎖を遅延グラフで組んでから 1 回だけ eval する。

    MLX は eval まで 1 つも submit しないので、**構築 (Python) と GPU を
    はっきり分けられる**。返す `median_us` は eval 区間 = GPU 時間で、
    構築時間は `build_us` に分けて残す (段が小さいと構築が支配するので、
    どちらも見ないと読み違える)。
    """
    import mlx.core as mx

    t0 = time.perf_counter()
    for i in range(n_steps):
        state = step(state, i % n_sets)
    t1 = time.perf_counter()
    mx.eval(state)
    t2 = time.perf_counter()
    return state, ((t2 - t1) / n_steps * 1e6, (t1 - t0) / n_steps * 1e6)


def _abba(cases, state_init, n_steps, n_sets):
    """`cases` = {名前: step 関数}。ABBA (2 者) / 回文 (3 者以上) で交互に回す。

    1 プロセス内で交互に測る (CLAUDE.md)。焼き入れを 1 巡してから本番。
    """
    names = list(cases)
    order = names + names[::-1]
    states = {n: state_init(n) for n in names}
    for n in names:                       # 焼き入れ (コンパイルと最初の巡回)
        states[n], _ = _chain(cases[n], states[n], n_steps, n_sets)
    samples = {n: [] for n in names}
    for _ in range(N_PAIRS):
        for n in order:
            states[n], us = _chain(cases[n], states[n], n_steps, n_sets)
            samples[n].append(us)
    return {n: _summ(v) for n, v in samples.items()}


# --------------------------------------------------------------------- 段


def build(args):
    import mlx.core as mx
    import mlx.nn as nn

    from micro_kernel_latency import (HC, HC_LOWRANK, HIDDEN, RMS_EPS,
                                      _qmm_linear, _weight_bytes, hc_weight_set)
    from mlxturbo.kernels import hyper_connection as hck
    from mlxturbo.kernels import qmm_wide as qw

    dtype = mx.bfloat16
    m = args.rows
    hcd = HC * HIDDEN
    sets = hc_weight_set(dtype, args.weight_sets)
    per_set = sum(_weight_bytes(t) for t in sets[0])

    elem_cfg = {"hc": HC, "d": HIDDEN, "lowrank": HC_LOWRANK,
                "eps": float(RMS_EPS), "combine": True}
    k_pre, k_mid, k_post = hck._get_kernels_elem(elem_cfg)
    TH = hck._ELEM_THREADS

    def rnd(shape, scale=0.5):
        return (mx.random.normal(shape) * scale).astype(dtype)

    # 巡回させる入力プール (同じ 1 枚を使い回すと L2 に居座る)
    pool_wide = [rnd((m, hcd)) for _ in range(args.pool)]
    pool_low = [rnd((m, HC_LOWRANK)) for _ in range(args.pool)]
    mx.eval(pool_wide, pool_low)

    # ---- 素の段 ---------------------------------------------------------
    def plain_pre(x, ws):
        norm_weight = ws[0]
        y = x.reshape(m, HC, HIDDEN)
        y = mx.fast.rms_norm(y, None, RMS_EPS).reshape(m, hcd)
        return y * (1.0 + norm_weight)

    def plain_mid(x):
        return nn.silu(x / HC)

    def plain_post(normed, up_raw):
        w = mx.sigmoid(up_raw).reshape(m, HC, HIDDEN)
        return (w * normed.reshape(m, HC, HIDDEN)).mean(axis=-2)

    def plain_inject(normed, ws):
        return 2 * mx.sigmoid(_qmm_linear(normed, ws[3]) / HC)

    # ---- elem カーネルの段 ----------------------------------------------
    def elem_pre(x, ws):
        (normed,) = k_pre(
            inputs=[x, ws[0]], template=[("T", dtype)],
            grid=(TH, HC, m), threadgroup=(TH, 1, 1),
            output_shapes=[(m, hcd)], output_dtypes=[dtype])
        return normed

    def elem_mid(x):
        gx = ((HC_LOWRANK + TH - 1) // TH) * TH
        (t,) = k_mid(
            inputs=[x], template=[("T", dtype)],
            grid=(gx, 1, m), threadgroup=(TH, 1, 1),
            output_shapes=[(m, HC_LOWRANK)], output_dtypes=[dtype])
        return t

    def elem_post(normed, up_raw, inject_raw):
        gx = ((HIDDEN + TH - 1) // TH) * TH
        return k_post(
            inputs=[normed, up_raw, inject_raw], template=[("T", dtype)],
            grid=(gx, 1, m), threadgroup=(TH, 1, 1),
            output_shapes=[(m, HIDDEN), (m, HC)],
            output_dtypes=[dtype, dtype])

    # ---- 行列積 ---------------------------------------------------------
    tiles = {n: qw.TILES[n] for n in args.tiles.split(",") if n.strip()}

    def wide(x, w, tile):
        wt, sc, bi, gs, bits = w
        return qw.qmm_wide(x, wt, sc, bi, tile=tile, group_size=gs, bits=bits)

    ctx = dict(
        mx=mx, m=m, hcd=hcd, dtype=dtype, sets=sets, per_set=per_set,
        pool_wide=pool_wide, pool_low=pool_low, tiles=tiles, wide=wide,
        plain_pre=plain_pre, plain_mid=plain_mid, plain_post=plain_post,
        plain_inject=plain_inject, elem_pre=elem_pre, elem_mid=elem_mid,
        elem_post=elem_post, qmm=_qmm_linear, rnd=rnd,
        HC=HC, HIDDEN=HIDDEN, HC_LOWRANK=HC_LOWRANK, eps=float(RMS_EPS),
    )
    return ctx


def run_stages(ctx, args):
    """段ごとの ms と、素 / elem・素 qmm / qmm_wide の比。"""
    mx = ctx["mx"]
    m, hcd = ctx["m"], ctx["hcd"]
    sets, n_sets = ctx["sets"], len(ctx["sets"])
    pool_wide, pool_low = ctx["pool_wide"], ctx["pool_low"]
    HC, HIDDEN, LOW = ctx["HC"], ctx["HIDDEN"], ctx["HC_LOWRANK"]
    out = {}

    def sink(acc, *arrs):
        for a in arrs:
            acc = acc + a[:1, :1].astype(mx.float32)
        return acc

    zero = mx.zeros((1, 1), dtype=mx.float32)

    # ---- pre (自然に直列) ------------------------------------------------
    def _pre_step(fn):
        def step(state, k):
            return fn(state, sets[k])
        return step

    out["pre"] = _abba(
        {"plain": _pre_step(ctx["plain_pre"]), "elem": _pre_step(ctx["elem_pre"])},
        lambda _n: ctx["rnd"]((m, hcd)), args.steps, n_sets)

    # ---- mid (自然に直列) ------------------------------------------------
    out["mid"] = _abba(
        {"plain": lambda s, k: ctx["plain_mid"](s),
         "elem": lambda s, k: ctx["elem_mid"](s)},
        lambda _n: ctx["rnd"]((m, LOW)), args.steps, n_sets)

    # ---- down / up / gemm_pair -----------------------------------------
    def down_step(fn):
        def step(state, k):
            acc = state
            y = fn(pool_wide[k % len(pool_wide)], sets[k][1])
            return sink(acc, y)
        return step

    def up_step(fn):
        def step(state, k):
            acc = state
            y = fn(pool_low[k % len(pool_low)], sets[k][2])
            return sink(acc, y)
        return step

    cases = {"stock": down_step(lambda x, w: ctx["qmm"](x, w))}
    for name, tile in ctx["tiles"].items():
        cases[f"wide_{name}"] = down_step(
            lambda x, w, t=tile: ctx["wide"](x, w, t))
    out["down"] = _abba(cases, lambda _n: zero, args.steps, n_sets)

    cases = {"stock": up_step(lambda x, w: ctx["qmm"](x, w))}
    for name, tile in ctx["tiles"].items():
        cases[f"wide_{name}"] = up_step(lambda x, w, t=tile: ctx["wide"](x, w, t))
    out["up"] = _abba(cases, lambda _n: zero, args.steps, n_sets)

    def pair_step(state, k):
        x = ctx["qmm"](state, sets[k][1])
        x = ctx["plain_mid"](x)
        return ctx["qmm"](x, sets[k][2])

    out["gemm_pair"] = _abba({"stock": pair_step},
                             lambda _n: ctx["rnd"]((m, hcd)), args.steps, n_sets)

    # ---- post + inject ---------------------------------------------------
    def post_plain(state, k):
        j = k % len(pool_wide)
        normed = pool_wide[j]
        up_raw = pool_wide[(j + 1) % len(pool_wide)]
        mixed = ctx["plain_post"](normed, up_raw)
        inj = ctx["plain_inject"](normed, sets[k])
        return sink(state, mixed, inj)

    def post_elem(state, k):
        j = k % len(pool_wide)
        normed = pool_wide[j]
        up_raw = pool_wide[(j + 1) % len(pool_wide)]
        inject_raw = ctx["qmm"](normed, sets[k][3])
        mixed, inj = ctx["elem_post"](normed, up_raw, inject_raw)
        return sink(state, mixed, inj)

    out["post_inject"] = _abba({"plain": post_plain, "elem": post_elem},
                               lambda _n: zero, args.steps, n_sets)

    # inject 単体 (post から切り離した bf16 GEMM + elementwise)
    def inject_only(state, k):
        normed = pool_wide[k % len(pool_wide)]
        return sink(state, ctx["plain_inject"](normed, sets[k]))

    out["inject"] = _abba({"plain": inject_only}, lambda _n: zero,
                          args.steps, n_sets)

    # ---- 書き戻し (_combine)。全体の壁時計に入るので段和にも足す ---------
    from mlxturbo import fused

    combine = fused._build_combine(HC, HIDDEN)
    branch = ctx["rnd"]((m, HIDDEN))
    inj0 = ctx["rnd"]((m, HC))
    mx.eval(branch, inj0)

    out["combine"] = _abba(
        {"plain": lambda s, k: combine(s, branch, inj0)},
        lambda _n: ctx["rnd"]((m, hcd)), args.steps, n_sets)

    return out


def run_full(ctx, args):
    """実モデルどおり `_combine` を挟んだ全体の連鎖。

    `plain` = wide を入れる前の素、`plain_wide` = **現在の本番の既定**
    (`--knob hc-qmm-wide` の A)、`elem` / `elem_wide` = 棄却した elem 幅拡大の
    記録。
    """
    mx = ctx["mx"]
    m, hcd = ctx["m"], ctx["hcd"]
    sets, n_sets = ctx["sets"], len(ctx["sets"])
    HC, HIDDEN = ctx["HC"], ctx["HIDDEN"]
    tile = next(iter(ctx["tiles"].values()), None)

    from mlxturbo import fused

    combine = fused._build_combine(HC, HIDDEN)

    def qmm_w(x, w):
        return ctx["wide"](x, w, tile)

    def plain_full(state, k, mm):
        hyper, branch, inj = state
        h = combine(hyper, branch, inj)
        ws = sets[k]
        normed = ctx["plain_pre"](h, ws)
        t = ctx["plain_mid"](mm(normed, ws[1]))
        up_raw = mm(t, ws[2])
        mixed = ctx["plain_post"](normed, up_raw)
        inj2 = ctx["plain_inject"](normed, ws)
        return (h, mixed, inj2)

    def elem_full(state, k, mm):
        hyper, branch, inj = state
        h = combine(hyper, branch, inj)
        ws = sets[k]
        normed = ctx["elem_pre"](h, ws)
        t = ctx["elem_mid"](mm(normed, ws[1]))
        up_raw = mm(t, ws[2])
        inject_raw = ctx["qmm"](normed, ws[3])   # inject は bf16 なので素のまま
        mixed, inj2 = ctx["elem_post"](normed, up_raw, inject_raw)
        return (h, mixed, inj2)

    def init(_n):
        return (ctx["rnd"]((m, hcd)), ctx["rnd"]((m, HIDDEN)),
                ctx["rnd"]((m, HC)))

    cases = {
        "plain": lambda s, k: plain_full(s, k, ctx["qmm"]),
        "elem": lambda s, k: elem_full(s, k, ctx["qmm"]),
    }
    if tile is not None:
        cases["plain_wide"] = lambda s, k: plain_full(s, k, qmm_w)
        cases["elem_wide"] = lambda s, k: elem_full(s, k, qmm_w)
    return _abba(cases, init, args.steps, n_sets)


def check_numerics(ctx):
    """elem の 3 段が素とどれだけ食い違うか (prefill 幅、1 組の重みで)。"""
    mx = ctx["mx"]
    m, hcd = ctx["m"], ctx["hcd"]
    HC, HIDDEN, LOW = ctx["HC"], ctx["HIDDEN"], ctx["HC_LOWRANK"]
    ws = ctx["sets"][0]
    hyper = ctx["rnd"]((m, hcd))
    mx.eval(hyper)

    def rate(a, b):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        diff = (a != b).astype(mx.float32).mean().item()
        mad = mx.abs(a - b).max().item()
        return {"mismatch_rate": float(f"{diff:.3g}"), "max_abs": float(f"{mad:.3g}")}

    n_p = ctx["plain_pre"](hyper, ws)
    n_e = ctx["elem_pre"](hyper, ws)
    mx.eval(n_p, n_e)
    res = {"normed": rate(n_p, n_e)}

    d_p = ctx["qmm"](n_p, ws[1])
    m_p = ctx["plain_mid"](d_p)
    m_e = ctx["elem_mid"](d_p)
    mx.eval(m_p, m_e)
    res["mid"] = rate(m_p, m_e)

    up_raw = ctx["qmm"](m_p, ws[2])
    inject_raw = ctx["qmm"](n_p, ws[3])
    mixed_p = ctx["plain_post"](n_p, up_raw)
    inj_p = 2 * mx.sigmoid(inject_raw / HC)
    mixed_e, inj_e = ctx["elem_post"](n_p, up_raw, inject_raw)
    mx.eval(mixed_p, inj_p, mixed_e, inj_e)
    res["mixed"] = rate(mixed_p, mixed_e)
    res["inject"] = rate(inj_p, inj_e)

    # post カーネルの「模型」(bf16 逐次加算) を素の op で組み、素の mean と
    # 突き合わせる。模型 == 素 なら食い違いはカーネルの中 (写した sigmoid か
    # Metal の bf16 算術)、模型 != 素 なら縮約の模型そのものが違う
    w_p = mx.sigmoid(up_raw).reshape(m, HC, HIDDEN)
    v = (w_p * n_p.reshape(m, HC, HIDDEN)).astype(mx.bfloat16)
    acc = v[:, 0, :]
    for l in range(1, HC):
        acc = (acc + v[:, l, :]).astype(mx.bfloat16)
    model = (acc.astype(mx.float32) / HC).astype(mx.bfloat16)
    mx.eval(model)
    res["mixed_model_vs_plain"] = rate(mixed_p, model)
    res["mixed_model_vs_elem"] = rate(model, mixed_e)

    # qmm_wide の一致 (ビット一致が期待値)
    # 本番の行数ゲート (`fused._QMM_WIDE_MIN_ROWS`、既定 1024)。これを下回る幅で
    # 呼ぶと down 側が素と食い違う (M=256 で 0.53 / M=512 で 0.373、M>=2048 は 0.0)
    # ことが 2026-09-04 に出た。**本番はこのゲートで届かない**が、`--rows` を
    # 小さくして測ると混乱するので、下回っていることを結果に残す。
    from mlxturbo import fused as _fused
    res["qmm_wide"] = {"below_prod_min_rows": m < _fused._QMM_WIDE_MIN_ROWS,
                       "prod_min_rows": _fused._QMM_WIDE_MIN_ROWS}
    for name, tile in ctx["tiles"].items():
        dw = ctx["wide"](n_p, ws[1], tile)
        uw = ctx["wide"](m_p, ws[2], tile)
        mx.eval(dw, uw)
        res["qmm_wide"][name] = {"down": rate(d_p, dw), "up": rate(up_raw, uw)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/hc-prefill-micro.json")
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--weight-sets", type=int, default=97)
    ap.add_argument("--steps", type=int, default=97)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--tiles", default="m64n32k32w2x2r8")
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    ctx = build(args)

    res = {
        "rows": args.rows,
        "weight_sets": args.weight_sets,
        "weight_mb": round(ctx["per_set"] * args.weight_sets / 1e6, 1),
        "steps_per_chain": args.steps,
        "pool": args.pool,
        "note": "段ごとの us/call (M 行、97 組を巡回させた冷の連鎖、ABBA)",
    }
    res["numerics"] = check_numerics(ctx)
    res["stages"] = run_stages(ctx, args)
    if not args.skip_full:
        res["full"] = run_full(ctx, args)

    st = res["stages"]
    plain_sum = (st["pre"]["plain"]["median_us"] + st["mid"]["plain"]["median_us"]
                 + st["down"]["stock"]["median_us"] + st["up"]["stock"]["median_us"]
                 + st["post_inject"]["plain"]["median_us"]
                 + st["combine"]["plain"]["median_us"])
    res["sum_check"] = {"stage_sum_us": round(plain_sum, 2)}
    if "full" in res:
        full = res["full"]["plain"]["median_us"]
        res["sum_check"]["full_us"] = full
        res["sum_check"]["ratio"] = round(plain_sum / full, 4)
    pair = st["gemm_pair"]["stock"]["median_us"]
    split = (st["down"]["stock"]["median_us"] + st["mid"]["plain"]["median_us"]
             + st["up"]["stock"]["median_us"])
    res["pair_check"] = {"pair_us": pair, "down+mid+up_us": round(split, 2),
                         "ratio": round(split / pair, 4)}

    # ---- 表示 -----------------------------------------------------------
    print(f"[M={args.rows} 重み {res['weight_mb']} MB / {args.weight_sets} 組]")
    for stage, cases in st.items():
        base = cases.get("plain") or cases.get("stock")
        line = f"  {stage:12s}"
        for name, s in cases.items():
            r = s["median_us"] / base["median_us"]
            line += f"  {name} {s['median_us']:8.1f} ({r:.3f})"
        print(line)
    if "full" in res:
        f = res["full"]
        base = f["plain"]["median_us"]
        line = f"  {'full':12s}"
        for name, s in f.items():
            line += f"  {name} {s['median_us']:8.1f} ({s['median_us'] / base:.3f})"
        print(line)
    print(f"  段和 {res['sum_check']['stage_sum_us']:.1f} us"
          + (f" / 壁時計 {res['sum_check'].get('full_us')} us"
             f" (比 {res['sum_check'].get('ratio')})" if "full" in res else ""))
    print(f"  pair 検算 {res['pair_check']['down+mid+up_us']:.1f}"
          f" / {res['pair_check']['pair_us']:.1f}"
          f" (比 {res['pair_check']['ratio']})")
    print(f"  数値 (elem 対 素): " + ", ".join(
        f"{k} {v['mismatch_rate']:.2g}" for k, v in res["numerics"].items()
        if isinstance(v, dict) and "mismatch_rate" in v))
    print(f"  qmm_wide の一致: {res['numerics']['qmm_wide']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"-> {out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
