"""prefill 中の n-gram 行取得と先読みの時刻関係を、チャンクごとに並べて出す。

本番のチャンク割り (group prefill) をそのまま踏み、1 プロセス内で
先読みの 3 条件を順に回して比べる:

    off   … 先読み無効 (2026-09-03 までの本番既定。pread は既定 off)
    late  … 旧配置。境界を組み終えて `mx.eval` を投入する直前に次を先読み
    early … 新配置。境界を**組み始める前**に次を先読み (最初の境界は前景取得)

出す表 (1 境界 = group build 1 回):
    build   … `_group_prefill_forward` の開始/終了 (壁時計の相対秒)
    fetch   … その境界の `StreamNGram.__call__` の sync_ms / fetch_ms と hit/miss
    pf      … 先読みの投入時刻、背景スレッドの開始〜完了、読んだ行数

使い方 (biglock 経由):
    tools/biglock.sh .venv/bin/python tools/ngram_prefill_diag.py \\
        --model ~/models/ddalcu-mlxlm-head4 --ngram ~/models/ddalcu-ngram \\
        --ctx 17000 --out bench/results/ngram-prefill-diag-17k.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# `_vendor/qwen4_exp.py` が import 時に読む (NGRAM_ON_DISK)。モデルを読む前に
# 立てないと n-gram の表を 102GB ぶん確保しようとして load が落ちる。
os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", required=True)
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--modes", default="off,late,early")
    ap.add_argument("--warmup", type=int, default=1, help="捨てる prefill の本数")
    ap.add_argument("--reps", type=int, default=1, help="条件ごとの採用本数")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401
    import mlxturbo.spec_flash as SF
    from mlxturbo import mtp_flash
    from mlxturbo.ngram_stream import StreamNGram, install
    from mlxturbo.runner import enable_default_fusions, set_wired_limit_default

    mpath = os.path.expanduser(args.model)
    model, tok = load(mpath)
    install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[ngram-diag]")
    set_wired_limit_default(log_prefix="[ngram-diag]")
    mtp = mtp_flash.load_flash_mtp(os.path.join(mpath, "mtp.safetensors"),
                                   model.args.text)
    mx.eval(mtp.parameters())
    eng = SF.FlashSpecEngine(model, mtp)

    stream = None
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is not None:
            stream = ple.ple_embedding.ngram_embedding
            break
    assert isinstance(stream, StreamNGram), type(stream)
    print(f"[diag] backend={stream.backend} threads={stream.n_threads} "
          f"nocache={os.environ.get('FASTMLX_NGRAM_NOCACHE')} "
          f"ple_layers={model.model.ple_layers}", flush=True)

    from _bench_text import long_prompts
    body = long_prompts(tok, args.ctx,
                        ["上の文書の要点を 5 つに整理してください。"])[0]
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True))[None]
    n = ids.shape[1]
    print(f"[diag] ctx={n} step={SF.PREFILL_STEP_SIZE} G={SF._PREFILL_GROUP}",
          flush=True)

    # ---------------------------------------------------------------- 計測の糸
    ev: list[dict] = []
    T0 = [time.perf_counter()]

    def _now():
        return time.perf_counter() - T0[0]

    raw_call = StreamNGram.__call__
    raw_prefetch = StreamNGram.prefetch
    raw_worker = StreamNGram._prefetch_worker
    raw_group = SF._group_prefill_forward

    def call_wrap(self, gid):
        s0 = dict(self.stats)
        t0 = _now()
        out = raw_call(self, gid)
        t1 = _now()
        s1 = self.stats
        ev.append({
            "k": "call", "t0": t0, "t1": t1,
            "rows": s1["rows"] - s0["rows"],
            "hits": s1["hits"] - s0["hits"],
            "misses": s1["misses"] - s0["misses"],
            "sync_ms": s1["sync_ms"] - s0["sync_ms"],
            "fetch_ms": s1["fetch_ms"] - s0["fetch_ms"],
        })
        return out

    def prefetch_wrap(self, flat_ids, wait=False):
        t0 = _now()
        raw_prefetch(self, flat_ids, wait=wait)
        ev.append({"k": "pf_submit", "t0": t0, "t1": _now(),
                   "rows": int(len(flat_ids)), "wait": bool(wait)})

    bg_threads = [0]  # 0 = 既定のプール (n_threads) をそのまま使う

    def worker_wrap(self, ids64):
        t0 = _now()
        # 背景先読みだけスレッド数を絞る条件を作れるようにする。主スレッドは
        # この間キャッシュヒットしか踏まない (プールを使わない) ので、
        # 差し替えは安全。**狙いは GIL の取り合いを減らすこと**: 背景の
        # `_gather_pread` は 1 行 1 回の `os.preadv` を Python ループで回すので、
        # 12 スレッドで回すと主スレッド (MLX のグラフ構築) が GIL の
        # 順番待ちに巻き込まれうる。先読みには十分な締切 (直前境界の実行
        # まるごと) があるので、並列度は最小で足りるはず。
        n_bg = bg_threads[0]
        if n_bg:
            pool, nth = self._pool, self.n_threads
            self._pool = ThreadPoolExecutor(max_workers=n_bg)
            self.n_threads = n_bg
            try:
                raw_worker(self, ids64)
            finally:
                self._pool.shutdown(wait=False)
                self._pool, self.n_threads = pool, nth
        else:
            raw_worker(self, ids64)
        ev.append({"k": "pf_worker", "t0": t0, "t1": _now(),
                   "rows": int(ids64.size), "bg_threads": n_bg})

    def group_wrap(model_, chunks, caches):
        t0 = _now()
        out = raw_group(model_, chunks, caches)
        ev.append({"k": "build", "t0": t0, "t1": _now(),
                   "widths": [c.shape[1] for c in chunks]})
        return out

    StreamNGram.__call__ = call_wrap
    StreamNGram.prefetch = prefetch_wrap
    StreamNGram._prefetch_worker = worker_wrap
    SF._group_prefill_forward = group_wrap

    def run_once(mode):
        # mode: off / late / early / early_bgN (N = 背景先読みのスレッド数)
        base, _, nbg = mode.partition("_bg")
        bg_threads[0] = int(nbg) if nbg else 0
        stream.prefetch_enabled = base != "off"
        stream._cache_gen = None  # 走行をまたいでキャッシュを持ち越さない
        SF._NGRAM_PREFETCH_AT = "early" if base == "early" else "late"
        stream.reset_stats()
        ev.clear()
        caches = model.make_cache()
        mx.clear_cache()
        T0[0] = time.perf_counter()
        gen = eng.generate_stream(ids, 0, caches=caches, checkpoints=[])
        try:
            while True:
                next(gen)
        except StopIteration:
            pass
        wall = (time.perf_counter() - T0[0]) * 1e3
        rec = {"mode": mode, "wall_ms": wall, "stats": dict(stream.stats),
               "events": sorted(ev, key=lambda e: e["t0"])}
        del caches
        mx.clear_cache()
        return rec

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = []
    # burn-in (プロセス起動直後の最初の計測行は +7〜9% 遅い)
    for _ in range(args.warmup):
        r = run_once(modes[0])
        print(f"[捨] {modes[0]}: wall={r['wall_ms']:.0f} ms", flush=True)
    for mode in modes:
        for _ in range(args.reps):
            r = run_once(mode)
            results.append(r)
            print(f"\n=== mode={mode}  wall={r['wall_ms']:.0f} ms", flush=True)
            print("    " + stream.stats_line(), flush=True)
            print(f"    {'ev':<10}{'t0(s)':>9}{'t1(s)':>9}{'dur(ms)':>9}"
                  f"{'rows':>9}{'hit':>9}{'miss':>9}{'sync':>8}{'fetch':>9}",
                  flush=True)
            for e in r["events"]:
                extra = ""
                if e["k"] == "build":
                    extra = f"  widths={e['widths']}"
                if e["k"] == "pf_submit" and e.get("wait"):
                    extra = "  (前景)"
                print(f"    {e['k']:<10}{e['t0']:>9.2f}{e['t1']:>9.2f}"
                      f"{(e['t1'] - e['t0']) * 1e3:>9.0f}"
                      f"{e.get('rows', ''):>9}{e.get('hits', ''):>9}"
                      f"{e.get('misses', ''):>9}"
                      f"{round(e['sync_ms']) if 'sync_ms' in e else '':>8}"
                      f"{round(e['fetch_ms']) if 'fetch_ms' in e else '':>9}"
                      f"{extra}", flush=True)

    StreamNGram.__call__ = raw_call
    StreamNGram.prefetch = raw_prefetch
    StreamNGram._prefetch_worker = raw_worker
    SF._group_prefill_forward = raw_group

    print("\n=== まとめ", flush=True)
    for r in results:
        s = r["stats"]
        builds = [round((e["t1"] - e["t0"]) * 1e3)
                  for e in r["events"] if e["k"] == "build"]
        print(f"  {r['mode']:<11} builds={builds}", flush=True)
        print(f"  {r['mode']:<6} wall={r['wall_ms']:>8.0f} ms  "
              f"fetch_ms={s['fetch_ms']:>7.0f}  sync_ms={s['sync_ms']:>6.0f}  "
              f"pf_bg_ms={s['prefetch_bg_ms']:>7.0f}  "
              f"pf_wait_ms={s['prefetch_wait_ms']:>7.0f}  "
              f"hit={s['hits']:>7}/{s['hits'] + s['misses']}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"tool": "ngram_prefill_diag", "model": args.model,
             "ngram": args.ngram, "ctx_tokens": n,
             "nocache": os.environ.get("FASTMLX_NGRAM_NOCACHE"),
             "runs": results}, indent=1, ensure_ascii=False))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
