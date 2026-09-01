"""MoE の「48 層を 1 eval」の項が kv とともに増える件を、常駐メモリで再現する。

`docs/research/KERNEL-PROGRAM.md` の「MoE の中身割り (2026-09-02)」で残った謎:
同じ 48 層の MoE を**層ごと**に測ると kv が伸びるほど**減る** (1720 -> 1572ms)
のに、**48 層を 1 eval** にまとめると**増える** (2085 -> 2329ms)。MoE の仕事は
kv に依存しないので、同じ graph の中で kv に依存するのは KV キャッシュの
**占有量**だけ。

**それならダミーの常駐配列でも再現するはず**というのがここで試す仮説。
再現すればアロケータかキャッシュの挙動、しなければ別の筋。

## 測り方

- **1 プロセス内で交互に測る** (CLAUDE.md の作法)。ダミー量の条件を
  ラウンドロビンで回し、各ラウンドで 1 条件 1 レップを取って中央値を出す。
  条件ごとにまとめて測ると熱の傾きが条件差に化ける。
- 見たいのは**絶対値ではなく「ダミー量で時間が動くか」という関係**。
- 下見のパス 1 で kv の占有量を実測してから、その量をダミーの水準に足す。
  卓上の見積もりではなく、`get_active_memory` の差で取る。
- `mx.clear_cache()` を各レップの前に打つ条件も混ぜる。アロケータの
  バッファキャッシュが効いているかどうかがここに出る。

    tools/biglock.sh .venv/bin/python tools/probe_moe_pressure.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep --ctx 17000
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GIB = 1 << 30


def sys_free_gb():
    """OS 側の回収可能メモリと圧縮領域 (GB)。スワップに落ちたら圧縮が伸びる。"""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    except Exception:
        return (0.0, 0.0)
    pg = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip().rstrip(".")
            if v.isdigit():
                pg[k.strip()] = int(v)
    unit = 16384 / 1e9
    free = (pg.get("Pages free", 0) + pg.get("Pages inactive", 0)
            + pg.get("Pages speculative", 0)) * unit
    comp = pg.get("Pages occupied by compressor", 0) * unit
    return (free, comp)


def mem_gb():
    import mlx.core as mx

    return (mx.get_active_memory() / 1e9, mx.get_cache_memory() / 1e9,
            mx.get_peak_memory() / 1e9)


def alloc_dummy(gb: float):
    """常駐のダミーを確保する。1GiB 刻みの実体で、ゼロ書き込みまで済ませる。"""
    import mlx.core as mx

    held = []
    n = int(round(gb))
    for _ in range(n):
        a = mx.zeros((GIB // 4,), dtype=mx.float32)
        mx.eval(a)
        held.append(a)
    rest = gb - n
    if rest > 0.01:
        a = mx.zeros((int(rest * GIB) // 4,), dtype=mx.float32)
        mx.eval(a)
        held.append(a)
    return held


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=3, help="ラウンドロビンの周回数")
    ap.add_argument("--reps", type=int, default=2, help="1 ラウンドで取るレップ数")
    ap.add_argument("--gb", default="0,2,4,8,16", help="ダミー量 (GB) の水準")
    ap.add_argument("--clear-variant", action="store_true",
                    help="clear_cache を毎レップ打つ条件も混ぜる")
    ap.add_argument("--skip-last", action="store_true",
                    help="末尾チャンクの測定を省く (chunk 0 の対照だけ回すとき)")
    args = ap.parse_args()

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401
    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo.runner import enable_default_fusions

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    # キャッシュの退避・復元は prefill_anatomy と同じものを使う (作法を揃える)
    from prefill_anatomy import pending, restore, snapshot

    lim = mx.set_memory_limit(1 << 40)
    mx.set_memory_limit(lim)
    clim = mx.set_cache_limit(1 << 40)
    mx.set_cache_limit(clim)
    print(f"MLX memory_limit={lim / 1e9:.1f}GB cache_limit={clim / 1e9:.1f}GB",
          flush=True)

    model, tok = load(os.path.expanduser(args.model))
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[moe-pressure]")

    from _bench_text import long_prompts

    body = long_prompts(tok, args.ctx, ["上の文書の要点を 5 つに整理してください。"])[0]
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True))[None]
    n = ids.shape[1]
    step = args.chunk
    n_full = n // step
    last = n_full - 1
    a0, c0, p0 = mem_gb()
    print(f"ctx={n} chunk={step} 完全チャンク={n_full} 計測点=[0, {last}]"
          f" rounds={args.rounds} reps={args.reps}", flush=True)
    print(f"モデル読み込み直後: active={a0:.1f}GB cache={c0:.1f}GB peak={p0:.1f}GB"
          f" / OS 回収可能={sys_free_gb()[0]:.1f}GB", flush=True)

    orig_call = Q.SparseMoeBlock.__call__

    def advance(cache, lo, hi, trace=False):
        """チャンク lo..hi-1 を本番どおり流す。trace なら各チャンク後の常駐を出す。"""
        rec = []
        for ci in range(lo, hi):
            mx.eval([model.model(ids[:, ci * step:(ci + 1) * step], cache=cache)]
                    + pending(cache))
            mx.clear_cache()
            if trace:
                a, c, p = mem_gb()
                rec.append((ci, a))
                print(f"  chunk {ci} 後 (kv={(ci + 1) * step:5d}):"
                      f" active={a:8.3f}GB cache={c:7.3f}GB peak={p:8.3f}GB",
                      flush=True)
        return rec

    def capture(cache, ci):
        """チャンク ci の MoE 48 層の入力を、本番の 1 フォワードから捕まえる。"""
        grabbed = []

        def g(self, *a):
            grabbed.append((self, *a))
            return orig_call(self, *a)

        Q.SparseMoeBlock.__call__ = g
        pre = snapshot(cache)
        mx.eval([model.model(ids[:, ci * step:(ci + 1) * step], cache=cache)]
                + pending(cache))
        Q.SparseMoeBlock.__call__ = orig_call
        restore(cache, pre)
        mx.clear_cache()
        return grabbed

    def one_eval(grabbed):
        """48 層を 1 回の eval にまとめる (謎の項が出る側)。"""
        outs = [orig_call(mod, x) for mod, x in grabbed]
        mx.eval(outs)
        return outs

    def layer_sum(grabbed, reps=3):
        """層ごとに測って合計する (kv とともに減る側)。"""
        tot = 0.0
        for mod, x in grabbed:
            mx.eval(orig_call(mod, x))
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                mx.eval(orig_call(mod, x))
                ts.append((time.perf_counter() - t0) * 1000)
            tot += statistics.median(ts)
            mx.clear_cache()
        return tot

    def sweep(grabbed, conds, tag):
        """条件をラウンドロビンで回す。1 プロセス内・交互。"""
        samples = {k: [] for k, _, _, _ in conds}
        mem = {}
        for r in range(args.rounds):
            for key, gb, clear_each, release in conds:
                held = alloc_dummy(gb)
                if release:
                    # 確保と解放だけ済ませて、測るときは常駐ゼロに戻す。
                    # 常駐量が効いているのか、確保・解放の churn が効いて
                    # いるのかを分ける唯一の対照。
                    del held
                    held = []
                    mx.clear_cache()
                mx.reset_peak_memory()
                mx.eval(one_eval(grabbed))          # 温め (捨てる)
                for _ in range(args.reps):
                    if clear_each:
                        mx.clear_cache()
                    t0 = time.perf_counter()
                    one_eval(grabbed)
                    samples[key].append((time.perf_counter() - t0) * 1000)
                mem[key] = mem_gb() + sys_free_gb()
                del held
                mx.clear_cache()
            print(f"  [{tag}] ラウンド {r + 1}/{args.rounds} 済", flush=True)
        print(f"\n  {tag}: 48 層を 1 eval")
        print(f"  {'条件':22s}{'中央値 ms':>10s}{'最小':>8s}{'最大':>8s}"
              f"{'active':>9s}{'cache':>8s}{'peak':>9s}{'OS空き':>9s}{'圧縮':>8s}")
        base = None
        out = {}
        for key, _, _, _ in conds:
            v = samples[key]
            med = statistics.median(v)
            out[key] = med
            if base is None:
                base = med
            a, c, p, f, cm = mem[key]
            print(f"  {key:22s}{med:10.1f}{min(v):8.1f}{max(v):8.1f}"
                  f"{a:9.1f}{c:8.1f}{p:9.1f}{f:9.1f}{cm:8.1f}"
                  f"   [{med - base:+.1f} ms]")
        return out

    # ---- パス 1: kv の占有量を実測する (測定はしない下見) -------------------
    print(f"\n[パス 1] kv の占有量を測る", flush=True)
    cache = model.make_cache()
    rec = advance(cache, 0, n_full, trace=True)
    kv_gb = rec[last][1] - rec[0][1]
    print(f"  chunk 0 -> {last} の active 増分 = {kv_gb:.3f} GB"
          f" (kv {step} -> {n_full * step})", flush=True)
    del cache, rec
    mx.clear_cache()

    levels = [float(x) for x in args.gb.split(",")]
    conds0 = [(f"ダミー {g:g}GB", g, False, False) for g in levels]
    # kv 相当のダミーを、実測した増分と同じ量で 1 点足す
    conds0.insert(1, (f"kv 相当 {kv_gb:.2f}GB", kv_gb, False, False))
    # 対照。確保と解放の churn だけ同じで、測る時点の常駐はゼロ
    for g in levels:
        if g > 0:
            conds0.append((f"ダミー {g:g}GB 確保して解放", g, False, True))
    if args.clear_variant:
        conds0.append(("ダミー 0GB + clear", 0.0, True, False))
        conds0.append((f"ダミー {levels[-1]:g}GB + clear", levels[-1], True, False))

    # ---- パス 2: chunk 0 でダミーを振る -----------------------------------
    print(f"\n[パス 2] chunk 0 (kv={step}) でダミーを振る", flush=True)
    cache = model.make_cache()
    g0 = capture(cache, 0)
    print(f"  捕まえた MoE 層 = {len(g0)}", flush=True)
    r0 = sweep(g0, conds0, f"chunk 0 (kv={step})")
    ls0 = layer_sum(g0)
    print(f"  chunk 0 の層ごと合計 = {ls0:.1f} ms", flush=True)
    del g0
    mx.clear_cache()

    # ---- パス 2b: chunk last で同じことをする ------------------------------
    if args.skip_last:
        b0 = r0[conds0[0][0]]
        print(f"\n[まとめ] chunk 0 のみ。層ごと合計 = {ls0:.1f} ms /"
              f" 48 層を 1 eval (ダミー 0GB) = {b0:.1f} ms")
        for key, _, _, _ in conds0[1:]:
            print(f"    {key:26s}{r0[key] - b0:+8.1f} ms")
        return 0

    print(f"\n[パス 2b] chunk {last} (kv={n_full * step}) へ進める", flush=True)
    advance(cache, 0, last)
    gl = capture(cache, last)
    print(f"  捕まえた MoE 層 = {len(gl)}", flush=True)
    condsL = [(f"ダミー {g:g}GB", g, False, False) for g in levels if g in (0.0, 2.0, 8.0)]
    rl = sweep(gl, condsL, f"chunk {last} (kv={n_full * step})")
    lsl = layer_sum(gl)
    print(f"  chunk {last} の層ごと合計 = {lsl:.1f} ms", flush=True)

    # ---- まとめ ------------------------------------------------------------
    b0 = r0[conds0[0][0]]
    bl = rl[condsL[0][0]]
    print(f"\n[まとめ]")
    print(f"  層ごと合計       chunk 0 = {ls0:8.1f} ms / chunk {last} = {lsl:8.1f} ms"
          f"  [{lsl - ls0:+.1f}]")
    print(f"  48 層を 1 eval   chunk 0 = {b0:8.1f} ms / chunk {last} = {bl:8.1f} ms"
          f"  [{bl - b0:+.1f}]")
    print(f"  差 (1 eval - 層ごと) chunk 0 = {b0 - ls0:+.1f} ms"
          f" / chunk {last} = {bl - lsl:+.1f} ms")
    print(f"  chunk 0 のダミー応答 (0GB 比):")
    for key, _, _, _ in conds0[1:]:
        print(f"    {key:24s}{r0[key] - b0:+8.1f} ms")
    print(f"  chunk {last} - chunk 0 の 1 eval 差 = {bl - b0:+.1f} ms。"
          f"これをダミーで説明できたかが判定", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
