"""27B (qwen3_5) の decode 1 ラウンドの内訳を測る。

`tools/decode_gpu_trace.py` と `tools/decode_module_attrib.py` は Flash-Next
(`FlashSpecEngine.generate_stream`) 前提 --- 1 発返しの `SpecEngine.generate`
には `generate_stream` が無く、round の切れ目を外から掴む口が違う。ここは
**27B の `mlxturbo/spec.py` の round** をそのまま計装する。

`mlxturbo/` は 1 行も変えない。計装は全部この道具の側から:

- `spec.mx` / `staged.mx` を薄い proxy に差し替えて `mx.eval` /
  `mx.async_eval` の**回数と待ち時間**を拾う (それ以外の属性は素通し)
- `SpecEngine._hidden_forward` / `_head` / `_mtp_append` / `_mtp_base` /
  `_rollback` をクラス側で包んで**呼び出し回数と CPU 時間**を拾う
- `generate(on_tokens=...)` が round の終わりで呼ばれるのを round の
  切れ目に使う (`mlxturbo/spec.py` の maint 相の最後)

## 相の判定 (spec.py の round の形をそのまま追う)

    draft  : round の頭から、最初の _hidden_forward 呼び出しまで
             (MTP 頭を cap 回 + confidences の同期 1 回 + window の async_eval)
    verify : _hidden_forward から _rollback まで
             (trunk の S 行 forward の**グラフ組み立て** + lm_head + 同期 1 回)
    maint  : _rollback から on_tokens まで
             (KV の trim、GDN 状態の巻き戻し、mtp_cache の trim と積み直し)

各相をさらに 3 つに割る:

    sync   : mx.eval / mx.async_eval の中で待った時間 (= GPU 待ち)
    build  : 包んだメソッドの中の時間 (= 遅延グラフを組む CPU 時間)
    glue   : 残り (= Python の糊。.item()/.tolist()/list 操作/SAM/EMA)

`build` と `sync` は重ならない (包んだメソッドはどれも遅延で、中で同期
しない)。唯一 staged 経路の `_hidden_forward` だけ中で `async_eval` するので、
そのぶんは `build` から引いてある。

## GPU 側

`decode_gpu_trace.Probe` (tools/bridge/libmetal_probe.dylib) をそのまま使う。
2 通りの走行を分ける:

    probe      : 計装を切って probe だけ。dispatch/round, GPU 時間, 稼働率
    probe-phase: 計装 + on_tokens でキャッシュを同期させ、相ごとの dispatch 数

後者は同期を 1 つ足すので**壁時計が変わる** (dispatch 数は変わらない)。
相ごとの数は後者から、壁時計と稼働率は前者から読むこと。

## S の費用

`--s-list` の各 S について、prefill 済みのキャッシュを毎回戻しながら
`_hidden_forward(window, caches, capture=True)` + `_head` + `mx.eval` を測る。
S=1 だけは本番が capture=False + staged を通るので両方測る。

## 走らせ方

    BIGLOCK_NO_WORKER=1 BIGLOCK_PRIO=1 tools/biglock.sh \\
        .venv/bin/python tools/decode_round_anatomy_generic.py \\
        --model ~/models/qwen38-27b-4bit --mtp ~/models/qwen38-27b-mtp \\
        --ctx-list 0,4000 --tokens 160 \\
        --out bench/results/round-anatomy-27b-0904.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


# --------------------------------------------------------------- mx の proxy

class _MxProxy(types.ModuleType):
    """`mx.eval` / `mx.async_eval` だけ数える薄い包み。

    `types.ModuleType` を継ぐのは、`mlxturbo/spec.py` の中で `mx` が
    モジュールとして扱われても壊れないようにするため。定義していない属性は
    `__getattr__` で本物に素通しする (1 回あたり数百 ns、1 round に数十回
    なので round の 0.02% 未満。clean 走行との差で確かめる)。
    """

    def __init__(self, real, tracer):
        super().__init__("mlx.core.__anatomy_proxy__")
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_tracer", tracer)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def eval(self, *a, **k):
        real = object.__getattribute__(self, "_real")
        tr = object.__getattribute__(self, "_tracer")
        t0 = time.perf_counter()
        r = real.eval(*a, **k)
        tr.on_sync("eval", time.perf_counter() - t0)
        return r

    def async_eval(self, *a, **k):
        real = object.__getattribute__(self, "_real")
        tr = object.__getattribute__(self, "_tracer")
        t0 = time.perf_counter()
        r = real.async_eval(*a, **k)
        tr.on_sync("async_eval", time.perf_counter() - t0)
        return r


# --------------------------------------------------------------- 相の追跡

PHASES = ("draft", "verify", "maint")


def _blank_round() -> dict:
    r = {"total_ms": 0.0, "S": 0, "tokens": 0}
    for p in PHASES:
        r[p + "_ms"] = 0.0
        r[p + "_sync_ms"] = 0.0
        r[p + "_build_ms"] = 0.0
        r[p + "_n_eval"] = 0
        r[p + "_n_async"] = 0
        r[p + "_dispatch"] = 0
    r["draft_head_calls"] = 0
    r["draft_mtp_append_calls"] = 0
    r["draft_mtp_append_ms"] = 0.0
    r["draft_head_ms"] = 0.0
    r["verify_hf_ms"] = 0.0
    r["verify_head_ms"] = 0.0
    r["maint_rollback_ms"] = 0.0
    r["maint_mtp_append_ms"] = 0.0
    return r


class Tracer:
    """round の中の相と、包んだ呼び出しの時間を積む。"""

    def __init__(self):
        self.active = False
        self.rounds: list[dict] = []
        self.cur: dict | None = None
        self.phase = "draft"
        self._t_phase = 0.0
        self._t_round = 0.0
        self._sync_total = 0.0  # build から引くための累計
        self.probe_count = None  # 相ごと dispatch を取るときだけ callable
        self._last_disp = 0

    # -- round --------------------------------------------------------
    def start_round(self):
        self.cur = _blank_round()
        self.phase = "draft"
        now = time.perf_counter()
        self._t_phase = now
        self._t_round = now
        if self.probe_count is not None:
            self._last_disp = self.probe_count()

    def _close_phase(self, now):
        if self.cur is None:
            return
        self.cur[self.phase + "_ms"] += (now - self._t_phase) * 1000.0
        if self.probe_count is not None:
            d = self.probe_count()
            self.cur[self.phase + "_dispatch"] += d - self._last_disp
            self._last_disp = d
        self._t_phase = now

    def switch(self, phase):
        if self.cur is None:
            return
        self._close_phase(time.perf_counter())
        self.phase = phase

    def end_round(self, n_tokens):
        if self.cur is None:
            return
        now = time.perf_counter()
        self._close_phase(now)
        self.cur["total_ms"] = (now - self._t_round) * 1000.0
        self.cur["tokens"] = n_tokens
        for p in PHASES:
            self.cur[p + "_glue_ms"] = max(
                self.cur[p + "_ms"]
                - self.cur[p + "_sync_ms"]
                - self.cur[p + "_build_ms"],
                0.0,
            )
        self.rounds.append(self.cur)
        self.cur = None

    # -- 計装からの通知 ------------------------------------------------
    def on_sync(self, kind, dt):
        self._sync_total += dt
        if self.cur is None:
            return
        self.cur[self.phase + "_sync_ms"] += dt * 1000.0
        self.cur[self.phase + ("_n_eval" if kind == "eval" else "_n_async")] += 1

    def on_build(self, dt, key=None):
        if self.cur is None:
            return
        self.cur[self.phase + "_build_ms"] += dt * 1000.0
        if key is not None:
            self.cur[key] += dt * 1000.0


def install(tracer):
    """`mlxturbo` を触らずに計装を当てる。戻り値は剥がす関数。"""
    import mlx.core as mx

    from mlxturbo import spec as spec_mod
    try:  # staged.py は 2026-09-04 の第 1 段で spec.py に畳まれて無くなる
        from mlxturbo import staged as staged_mod
    except ImportError:
        staged_mod = None

    proxy = _MxProxy(mx, tracer)
    orig_spec_mx = spec_mod.mx
    orig_staged_mx = staged_mod.mx if staged_mod is not None else None
    spec_mod.mx = proxy
    if staged_mod is not None:
        staged_mod.mx = proxy

    Eng = spec_mod.SpecEngine
    orig = {
        n: getattr(Eng, n)
        for n in ("_hidden_forward", "_head", "_mtp_append", "_mtp_base", "_rollback")
    }

    def timed(fn, key_by_phase, phase_switch=None):
        def wrapper(self, *a, **k):
            if not tracer.active:
                return fn(self, *a, **k)
            if phase_switch is not None:
                phase_switch(a, k)
            s0 = tracer._sync_total
            t0 = time.perf_counter()
            out = fn(self, *a, **k)
            dt = time.perf_counter() - t0 - (tracer._sync_total - s0)
            key = key_by_phase.get(tracer.phase)
            tracer.on_build(dt, key)
            return out

        return wrapper

    def hf_switch(a, k):
        if tracer.phase == "draft":
            tracer.switch("verify")
        if tracer.cur is not None and a:
            try:
                tracer.cur["S"] = int(a[0].shape[0])
            except Exception:  # noqa: BLE001
                pass

    def rb_switch(a, k):
        if tracer.phase == "verify":
            tracer.switch("maint")

    def counting_head(fn):
        inner = timed(fn, {"draft": "draft_head_ms", "verify": "verify_head_ms"})

        def wrapper(self, *a, **k):
            if tracer.active and tracer.cur is not None and tracer.phase == "draft":
                tracer.cur["draft_head_calls"] += 1
            return inner(self, *a, **k)

        return wrapper

    def counting_append(fn):
        inner = timed(
            fn,
            {"draft": "draft_mtp_append_ms", "maint": "maint_mtp_append_ms"},
        )

        def wrapper(self, *a, **k):
            if tracer.active and tracer.cur is not None and tracer.phase == "draft":
                tracer.cur["draft_mtp_append_calls"] += 1
            return inner(self, *a, **k)

        return wrapper

    Eng._hidden_forward = timed(
        orig["_hidden_forward"], {"verify": "verify_hf_ms"}, hf_switch
    )
    Eng._rollback = timed(orig["_rollback"], {"maint": "maint_rollback_ms"}, rb_switch)
    Eng._head = counting_head(orig["_head"])
    Eng._mtp_append = counting_append(orig["_mtp_append"])
    Eng._mtp_base = timed(orig["_mtp_base"], {})

    def uninstall():
        spec_mod.mx = orig_spec_mx
        if staged_mod is not None:
            staged_mod.mx = orig_staged_mx
        for n, f in orig.items():
            setattr(Eng, n, f)

    return uninstall


# --------------------------------------------------------------- 集計

def _stats(vals):
    if not vals:
        return {"mean": 0.0, "median": 0.0, "n": 0}
    return {
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "n": len(vals),
    }


def summarize_rounds(rounds: list[dict], skip: int) -> dict:
    use = rounds[skip:]
    if not use:
        use = rounds
    keys = [k for k in use[0] if k not in ("S", "tokens")]
    out = {"n_rounds": len(use)}
    for k in keys:
        out[k] = sum(r[k] for r in use) / len(use)
    out["S_mean"] = sum(r["S"] for r in use) / len(use)
    out["tok_per_round"] = sum(r["tokens"] for r in use) / len(use)
    out["S_hist"] = {}
    for r in use:
        out["S_hist"][str(r["S"])] = out["S_hist"].get(str(r["S"]), 0) + 1
    out["round_ms_median"] = statistics.median(r["total_ms"] for r in use)
    return out


def fmt_anatomy(name: str, s: dict) -> str:
    tot = s["total_ms"] or 1.0
    L = [f"--- {name} ---"]
    L.append(
        f"  ラウンド {s['n_rounds']}  壁時計 {s['total_ms']:.2f} ms/round "
        f"(中央値 {s['round_ms_median']:.2f})  tok/round {s['tok_per_round']:.2f}  "
        f"ms/tok {s['total_ms'] / max(s['tok_per_round'], 1e-9):.2f}  "
        f"S 平均 {s['S_mean']:.2f}  S 分布 {s['S_hist']}"
    )
    L.append(f"    {'相':<8} {'ms':>8} {'%':>6} | {'sync':>8} {'build':>8} {'glue':>8}"
             f" | {'eval':>5} {'async':>5}")
    for p, lab in (("draft", "draft"), ("verify", "verify"), ("maint", "maint")):
        L.append(
            f"    {lab:<8} {s[p + '_ms']:8.2f} {100 * s[p + '_ms'] / tot:6.1f} |"
            f" {s[p + '_sync_ms']:8.2f} {s[p + '_build_ms']:8.2f}"
            f" {s[p + '_glue_ms']:8.2f} |"
            f" {s[p + '_n_eval']:5.2f} {s[p + '_n_async']:5.2f}"
        )
    sy = sum(s[p + "_sync_ms"] for p in PHASES)
    bu = sum(s[p + "_build_ms"] for p in PHASES)
    gl = sum(s[p + "_glue_ms"] for p in PHASES)
    L.append(
        f"    {'合計':<8} {sy + bu + gl:8.2f} {100 * (sy + bu + gl) / tot:6.1f} |"
        f" {sy:8.2f} {bu:8.2f} {gl:8.2f}"
    )
    L.append(
        f"  内訳: draft MTP 頭 {s['draft_mtp_append_calls']:.2f} 回"
        f" (append {s['draft_mtp_append_ms']:.2f} ms + head {s['draft_head_ms']:.2f} ms)"
        f" / verify trunk 組み立て {s['verify_hf_ms']:.2f} ms"
        f" + lm_head {s['verify_head_ms']:.2f} ms"
        f" / maint rollback {s['maint_rollback_ms']:.2f} ms"
        f" + mtp 積み直し {s['maint_mtp_append_ms']:.2f} ms"
    )
    if any(s.get(p + "_dispatch") for p in PHASES):
        L.append(
            "  dispatch/round: "
            + "  ".join(f"{p} {s[p + '_dispatch']:.1f}" for p in PHASES)
            + f"   合計 {sum(s[p + '_dispatch'] for p in PHASES):.1f}"
        )
    return "\n".join(L)


# --------------------------------------------------------------- 走行

def run_gen(eng, ids, n_tokens, eos_ids, nd, md, session=None, on_tokens=None,
            lookup_len=16):
    import mlx.core as mx

    mx.clear_cache()
    t0 = time.perf_counter()
    res = eng.generate(
        ids,
        max_tokens=n_tokens,
        n_draft=nd,
        max_draft=md,
        lookup_len=lookup_len,
        temp=0.0,
        eos_ids=eos_ids,
        session=session,
        on_tokens=on_tokens,
    )
    res["_wall_s"] = time.perf_counter() - t0
    return res


def make_on_tokens(tracer, sync_caches=None):
    """round の切れ目。`sync_caches` を渡すと maint の GPU 仕事をそこで
    同期させる (相ごとの dispatch を取る走行だけ。壁時計は変わる)。"""
    import mlx.core as mx

    state = {"first": True}

    def cb(toks):
        if state["first"]:
            # ループ前の 1 回目 (prefill 直後の 1 トークン)
            state["first"] = False
            tracer.start_round()
            return
        if sync_caches is not None:
            arrs = []
            for c in sync_caches:
                st = getattr(c, "state", None)
                if st is None:
                    continue
                arrs.extend(x for x in (st if isinstance(st, list) else [st])
                            if x is not None)
            if arrs:
                mx.eval(arrs)
        tracer.end_round(len(toks))
        tracer.start_round()

    return cb


# --------------------------------------------------------------- S の費用

def measure_s_cost(eng, caches, snap, s_list, reps, restore_cache):
    """S 行の trunk forward + lm_head の費用。毎回キャッシュを戻す。"""
    import mlx.core as mx

    out = []
    for S in s_list:
        window = mx.array([100 + i for i in range(S)], dtype=mx.int32)
        mx.eval(window)
        for mode in (("capture", True), ("staged", False)):
            label, cap = mode
            if S > 1 and label == "staged":
                # 本番は S>1 で必ず capture=True を通る。参考として測る
                pass
            ts = []
            for r in range(reps + 1):
                for c, rec in zip(caches, snap):
                    restore_cache(c, rec)
                mx.eval([c.state for c in caches if getattr(c, "state", None) is not None])
                t0 = time.perf_counter()
                if cap:
                    hs, sink = eng._hidden_forward(window, caches, capture=True)
                else:
                    hs, sink = eng._hidden_forward(
                        window, caches, capture=False, staged=True
                    )
                lg = eng._head(hs, eng.inner.norm)
                p = mx.argmax(lg, axis=-1)
                mx.eval(p)
                dt = (time.perf_counter() - t0) * 1000.0
                if r > 0:
                    ts.append(dt)
            out.append({"S": S, "mode": label, "ms_median": statistics.median(ts),
                        "ms_min": min(ts), "reps": len(ts)})
    for c, rec in zip(caches, snap):
        restore_cache(c, rec)
    return out


# --------------------------------------------------------------- main

def build_parser():
    ap = argparse.ArgumentParser(
        description="27B (qwen3_5) の decode 1 ラウンドの内訳",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--ctx-list", default="0,4000",
                    help="0 = 短文脈 3 本、N = 池から切った N トークンの窓 1 本")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--skip-rounds", type=int, default=4,
                    help="集計から外す先頭のラウンド数")
    ap.add_argument("--n-draft", type=int, default=3)
    ap.add_argument("--max-draft", type=int, default=8)
    ap.add_argument("--s-list", default="1,2,4,8")
    ap.add_argument("--s-reps", type=int, default=7)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--no-ablate", action="store_true",
                    help="lookup 無しの対照を測らない")
    ap.add_argument("--no-nospec", action="store_true",
                    help="非投機 (n_draft=0, lookup 無し) の対照を測らない")
    ap.add_argument("--no-s-cost", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap


def main() -> int:
    args = build_parser().parse_args()

    import mlx.core as mx

    # probe は Metal のドライバのクラスが出来てから入れる (モデル読み込み前)
    probe = None
    if not args.no_probe:
        from decode_gpu_trace import Probe

        mx.eval(mx.ones((8, 8)) @ mx.ones((8, 8)))
        probe = Probe()
        if probe.install():
            print(f"metal_probe: 差し替え {probe.install_hits}")
            probe.enable(False)
        else:
            print(f"metal_probe: 使えない ({probe.error})。壁時計だけになる。")
            probe = None

    from decode_ab_generic import (_restore_cache, _restore_session,
                                   _snap_cache, _snapshot_session, build_cases,
                                   load_model, prefill_once)

    model, tok, eng, eos_ids, _guard = load_model(args)
    mx.random.seed(args.seed)
    nd, md = args.n_draft, args.max_draft

    from mlxturbo.kernels import _fire

    def disp_count():
        # 軽い読み出し (mp_stats の scalar だけ。カーネル名の走査を避ける)
        import ctypes
        d = ctypes.c_uint64(); cb = ctypes.c_uint64(); cbd = ctypes.c_uint64()
        gs = ctypes.c_double(); gu = ctypes.c_double(); nn = ctypes.c_int()
        probe._lib.mp_stats(ctypes.byref(d), ctypes.byref(cb), ctypes.byref(cbd),
                            ctypes.byref(gs), ctypes.byref(gu), ctypes.byref(nn))
        return int(d.value)

    tracer = Tracer()

    # ---- 空焼き (読み込み直後の 1 本は +7〜9% 遅い) ----
    t0 = time.perf_counter()
    for _kind, ids in build_cases(tok, 0):
        run_gen(eng, ids, 32, eos_ids, nd, md)
    print(f"[anatomy] 空焼き 1 本を捨てた ({time.perf_counter() - t0:.1f}s)", flush=True)

    ctx_list = [int(c) for c in args.ctx_list.split(",") if c.strip()]
    s_list = [int(s) for s in args.s_list.split(",") if s.strip()]
    results = {"cases": [], "s_cost": [], "meta": {
        "model": args.model, "mtp": args.mtp, "tokens": args.tokens,
        "n_draft": nd, "max_draft": md, "skip_rounds": args.skip_rounds,
    }}

    for ctx in ctx_list:
        cases = build_cases(tok, ctx)
        for ci, (kind, ids) in enumerate(cases):
            name = f"{kind}-ctx{len(ids)}-#{ci}"
            print(f"\n=== {name} ===", flush=True)
            # 全ケースで prefill を 1 回に畳む。decode の窓を prefill から
            # 切り離せるだけでなく、条件ごとに**まったく同じ状態**から
            # 始められる (session.tail からの再開)
            sess, snap, took = prefill_once(eng, ids, nd, md)
            print(f"  prefill 1 回 ({took:.1f}s)。以降 decode のみ", flush=True)

            def one(n_tokens, lookup_len=16, nd_=None, md_=None):
                _restore_session(sess, snap)
                return run_gen(eng, ids, n_tokens, eos_ids,
                               nd if nd_ is None else nd_,
                               md if md_ is None else md_,
                               session=sess, lookup_len=lookup_len)

            def one_cb(n_tokens, cb):
                _restore_session(sess, snap)
                return run_gen(eng, ids, n_tokens, eos_ids, nd, md,
                               session=sess, on_tokens=cb)

            # ケースごとの空焼き
            one(32)

            # --- 1) clean (計装なし・probe なし) ---
            _fire.reset()
            res = one(args.tokens)
            clean = {
                "wall_s": res["_wall_s"],
                "ttft_s": res["ttft_s"],
                "decode_s": res["_wall_s"] - res["ttft_s"],
                "steps": res["steps"],
                "tokens": len(res["tokens"]),
                "ms_per_round": (res["_wall_s"] - res["ttft_s"]) / max(res["steps"], 1) * 1000,
                "tok_per_round": res["tokens_per_step"],
                "phase_s": res["phase_s"],
                "accept_hist": res["accept_hist"],
                "src_hist": res["src_hist"],
                "fired": _fire.snapshot(),
            }
            print(f"  clean: {clean['ms_per_round']:.2f} ms/round  "
                  f"tok/round {clean['tok_per_round']:.3f}  "
                  f"steps {clean['steps']}", flush=True)
            ph = clean["phase_s"]
            tot_ph = sum(ph.values()) or 1.0
            nst = max(clean["steps"], 1)
            print("    generate() 内蔵 phase_s: "
                  + "  ".join(f"{k} {v / nst * 1000:.2f} ms/round "
                              f"({100 * v / tot_ph:.1f}%)" for k, v in ph.items()),
                  flush=True)
            if clean["fired"]:
                print("    発火: " + " ".join(f"{k}={v}" for k, v in
                                            sorted(clean["fired"].items())), flush=True)

            # --- 1b) 非投機 (n_draft=0, lookup 無し) = 素の 1 トークン decode ---
            if not args.no_nospec:
                res_ns = one(max(args.tokens // 3, 24), lookup_len=0, nd_=0, md_=0)
                dec_ns = res_ns["_wall_s"] - res_ns["ttft_s"]
                nospec = {
                    "rounds": res_ns["steps"],
                    "ms_per_round": dec_ns / max(res_ns["steps"], 1) * 1000,
                    "tokens": len(res_ns["tokens"]),
                }
                nospec["speedup_vs_spec"] = nospec["ms_per_round"] / max(
                    clean["ms_per_round"] / max(clean["tok_per_round"], 1e-9), 1e-9)
                case_nospec = nospec
                print(f"  非投機 (S=1, MTP/lookup 無し): "
                      f"{nospec['ms_per_round']:.2f} ms/token"
                      f"  (投機 {clean['ms_per_round'] / max(clean['tok_per_round'], 1e-9):.2f}"
                      f" ms/tok に対して {nospec['speedup_vs_spec']:.2f}x)", flush=True)
            else:
                case_nospec = None

            # --- 1c) lookup (SAM + verify のエントロピー行 + D7) を切った対照 ---
            #  track_lookup が False になると verify の
            #  softmax(logits.astype(f32)) と entropy 行 (S x 248320 の fp32)
            #  を組まなくなり、SAM の extend と D7 の追加同期も消える。
            #  MTP の draft はそのまま。差が「lookup 側の糊 + エントロピー税」
            if not args.no_ablate:
                res_nl = one(args.tokens, lookup_len=0)
                dec_nl = res_nl["_wall_s"] - res_nl["ttft_s"]
                nolookup = {
                    "rounds": res_nl["steps"],
                    "ms_per_round": dec_nl / max(res_nl["steps"], 1) * 1000,
                    "tok_per_round": res_nl["tokens_per_step"],
                    "tokens": len(res_nl["tokens"]),
                }
                nolookup["delta_ms_per_round"] = (
                    clean["ms_per_round"] - nolookup["ms_per_round"])
                print(f"  lookup 無し (MTP のみ): {nolookup['ms_per_round']:.2f} ms/round"
                      f"  tok/round {nolookup['tok_per_round']:.3f}"
                      f"  (既定との差 {nolookup['delta_ms_per_round']:+.2f} ms/round)",
                      flush=True)
            else:
                nolookup = None

            # --- 2) 計装あり (probe なし) ---
            un = install(tracer)
            try:
                tracer.rounds = []
                tracer.probe_count = None
                tracer.active = True
                res2 = one_cb(args.tokens, make_on_tokens(tracer))
                tracer.active = False
            finally:
                un()
            anat = summarize_rounds(tracer.rounds, args.skip_rounds)
            anat["instr_overhead_pct"] = (
                ((res2["_wall_s"] - res2["ttft_s"]) / max(res2["steps"], 1) * 1000)
                / clean["ms_per_round"] - 1.0) * 100.0
            print(fmt_anatomy(name + " [計装]", anat), flush=True)
            print(f"  計装の上乗せ {anat['instr_overhead_pct']:+.1f}% "
                  f"(clean {clean['ms_per_round']:.2f} ms/round に対して)", flush=True)

            case = {"name": name, "kind": kind, "ctx": len(ids),
                    "clean": clean, "nospec": case_nospec,
                    "nolookup": nolookup, "anatomy": anat}

            # --- 3) probe (計装なし) ---
            if probe is not None:
                probe.quiesce(); probe.reset(); probe.enable(True)
                t0 = time.perf_counter()
                res3 = one(args.tokens)
                wall = time.perf_counter() - t0
                probe.enable(False)
                pending = probe.quiesce()
                st = probe.stats()
                n = max(res3["steps"], 1)
                dec = res3["_wall_s"] - res3["ttft_s"]
                gpu_case = {
                    "rounds": res3["steps"],
                    "decode_s": dec,
                    "ms_per_round": dec / n * 1000,
                    "dispatch_per_round": st["dispatches"] / n,
                    "cb_per_round": st["command_buffers"] / n,
                    "gpu_sum_ms_per_round": st["gpu_sum_ms"] / n,
                    "gpu_union_ms_per_round": st["gpu_union_ms"] / n,
                    # 稼働率は decode 区間の壁時計に対して (prefill は除く)
                    "gpu_busy_frac": st["gpu_union_ms"] / (dec * 1000.0),
                    "mean_kernel_gpu_us": st["gpu_sum_ms"] * 1000.0 / max(st["dispatches"], 1),
                    "mean_gap_us": (dec * 1000.0 - st["gpu_union_ms"]) * 1000.0
                                   / max(st["dispatches"], 1),
                    "probe_overhead_pct": (dec / n * 1000) / clean["ms_per_round"] * 100 - 100,
                    "pending": pending,
                    "kernels": [
                        {"name": k["name"], "per_round": k["count"] / n,
                         "ms_per_round": k["gpu_ms"] / n,
                         "us_per_call": k["gpu_ms"] * 1000.0 / max(k["count"], 1)}
                        for k in st["kernels"][:35]
                    ],
                }
                case["gpu"] = gpu_case
                print(f"  [probe] dispatch {gpu_case['dispatch_per_round']:.1f}/round  "
                      f"CB {gpu_case['cb_per_round']:.1f}/round  "
                      f"GPU 和 {gpu_case['gpu_sum_ms_per_round']:.2f} / 和集合 "
                      f"{gpu_case['gpu_union_ms_per_round']:.2f} ms/round  "
                      f"稼働率 {100 * gpu_case['gpu_busy_frac']:.1f}%  "
                      f"probe 上乗せ {gpu_case['probe_overhead_pct']:+.1f}%", flush=True)
                print(f"    カーネル平均 {gpu_case['mean_kernel_gpu_us']:.1f} us  "
                      f"隙間平均 {gpu_case['mean_gap_us']:.1f} us/dispatch", flush=True)
                print(f"    {'カーネル':<46} {'回/round':>9} {'ms/round':>9} {'us/回':>8}")
                for k in gpu_case["kernels"][:14]:
                    nm = k["name"]
                    if len(nm) > 46:
                        nm = nm[:22] + "…" + nm[-23:]
                    print(f"    {nm:<46} {k['per_round']:9.1f} "
                          f"{k['ms_per_round']:9.3f} {k['us_per_call']:8.1f}")

                # --- 4) 相ごとの dispatch (計装 + maint の同期) ---
                un = install(tracer)
                try:
                    tracer.rounds = []
                    tracer.probe_count = disp_count
                    tracer.active = True
                    probe.quiesce(); probe.reset(); probe.enable(True)
                    cb = make_on_tokens(tracer, sync_caches=sess.caches)
                    one_cb(args.tokens, cb)
                    probe.enable(False)
                    probe.quiesce()
                    tracer.active = False
                    tracer.probe_count = None
                finally:
                    un()
                pa = summarize_rounds(tracer.rounds, args.skip_rounds)
                case["anatomy_phase_dispatch"] = pa
                print(fmt_anatomy(name + " [計装+probe: 相ごと dispatch]", pa),
                      flush=True)

            results["cases"].append(case)

            # --- 5) S の費用 (このケースのキャッシュ状態で) ---
            if not args.no_s_cost:
                _restore_session(sess, snap)
                caches = sess.caches
                csnap = [_snap_cache(c) for c in caches]
                rows = measure_s_cost(eng, caches, csnap, s_list, args.s_reps,
                                      _restore_cache)
                for r in rows:
                    r["case"] = name
                    r["ctx"] = len(ids)
                results["s_cost"].extend(rows)
                print(f"  [S の費用] ctx={len(ids)}  "
                      f"(trunk forward + lm_head + eval、キャッシュは毎回戻す)")
                print(f"    {'S':>3} {'mode':<9} {'ms(中央値)':>11} {'ms(最小)':>9}"
                      f" {'S=1 比':>8}")
                base = next((r["ms_median"] for r in rows
                             if r["S"] == 1 and r["mode"] == "capture"), None)
                for r in rows:
                    rel = f"{r['ms_median'] / base:7.2f}x" if base else "     -"
                    print(f"    {r['S']:3d} {r['mode']:<9} {r['ms_median']:11.2f}"
                          f" {r['ms_min']:9.2f} {rel:>8}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"\n書き出し: {p}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
