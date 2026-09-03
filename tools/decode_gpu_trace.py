"""decode 1 ラウンド (投機 decode の 1 round) の GPU の使われ方を測る。

## 何を出すか

短文脈 / 17k / draft 無し (depth 0) の 3 本について、安定した区間の
**1 ラウンドあたり**:

- dispatch 数 (= GPU に投げた compute カーネルの回数) と、カーネル名ごとの
  内訳 (回数と GPU 時間)
- command buffer の数
- GPU 実行時間の和 (`GPUEndTime - GPUStartTime` の総和) と和集合
- 稼働率 = GPU 時間の和集合 / ラウンドの壁時計
- カーネル間の隙間の目安 = (壁時計 - GPU 時間) / dispatch 数

狙いは「S=1 forward の 24ms が、active 重み 2GB の読み出し (400GB/s で 5ms)
の 5 倍ある」差が **カーネル数と直列化** なのか **GPU が空いている時間** なのかを
数字にすること (docs/research/EXTERNAL-PERPLEXITY-LILY-2026-09.md の
「795 カーネル / 555 直列段」に対応する自分側の数)。

## どうやって測るか (先に潰した経路)

`xctrace` (Metal System Trace) は python (MLX) の Compute 区間をほぼ拾えず
(`docs/research/SESSION-2026-09-02-CATCHUP.md`)、`mx.metal.start_capture` の
`.gputrace` は 98GB の常駐リソースを丸ごと書き出して終わらない
(`tools/gpu_capture_forward.py` 冒頭の実測)。そこでこの道具は
**Metal の具象クラスのメソッドを実行時に差し替える観測用 dylib**
(`tools/bridge/metal_probe.mm`) を使う。MLX にも libmlx にも手を入れない。

差し替えるのは `dispatchThreadgroups:` / `dispatchThreads:` /
`dispatchThreadgroupsWithIndirectBuffer:` / `setComputePipelineState:` /
`commit` / `newComputePipelineStateWith...` の 6 種。
`commit` で completion handler を足し、command buffer ごとの
`GPUStartTime` / `GPUEndTime` を拾う。

## 近似している点 (報告に必ず書くこと)

1. **カーネル名ごとの GPU 時間は、command buffer の GPU 時間をその CB に
   積まれた dispatch 数で等分した配分値**。Metal は dispatch ごとの
   タイムスタンプを (counter sample buffer 無しには) 返さないため。
   MLX は 1 CB に 15〜19 dispatch を積むので、**この配分は事実上 dispatch
   数の按分に近く、カーネル別の時間としては信用できない** (2026-09-03 実測:
   量子化行列積の取り分が、等分だと 21〜26%、`--split-cb` だと 41〜50%)。
   `--split-cb` を付けると `MLX_MAX_OPS_PER_BUFFER=1` で 1 CB = 1 op に
   割るので配分が厳密になる。ただし commit の回数が増えて壁時計が 2.5 倍に
   なり (depth0 で 25.5 → 64.0 ms/round)、1 カーネルあたり 0.8〜1.0 us の
   CB 固定費が上乗せされる (カーネル数の多い「のり」側に効く)。
   **絶対値の比較には使わない**。配分を見るときだけ使い、固定費
   (= split-cb の合計 - 既定の合計、を dispatch 数で割る) を引いて読む。
   この指定はプロセス全体に効く env なので、走行を分けること。
   **`--split-cb` の「1 カーネルあたり us」は、そのカーネルが入った CB の
   GPU 区間 (GPUStartTime→GPUEndTime) まるごとであって実費ではない。**
   1 CB 1 dispatch では、その CB に落ちた泡や待ちを 1 本で背負う (前例:
   最終 mixer の `hc_elem_post` が 462 us/call と出たが、実費は 4.3 us。
   回数が 1 でも 3 でも合計 ~0.5 ms/round で固定 = 回数に比例しない、が
   見分け方。`tools/hc_elem_post_micro.py --probe` が同じ形を再現する:
   同じカーネルが投入の仕方だけで 44.8 と 7.0 us/call に振れる)。
   **回数に比例しない項目を見たら、必ず単体 micro で裏を取ること。**
2. dispatch 数は compute の dispatch だけ (直接・間接の両方)。blit (コピー)
   は command buffer としては数えるが dispatch には数えない。
3. install より前に作られた pipeline は名前の台帳に無く、`label` が空なら
   `(pre-install pipeline)` になる。install は MLX が GPU を 1 回使った
   直後に行うので、対象はごく少数のはず (出力で確認できる)。
4. 「隙間」は依存の連鎖の長さそのものではない。壁時計から GPU 実行時間を
   引いた残りを dispatch 数で割った**平均**であって、直列段の数は MLX の
   外からは見えない。

## 使い方

    tools/biglock.sh .venv/bin/python tools/decode_gpu_trace.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram

    # 道具の検算だけ (GPU を使わない。合成の小さいモデルを CPU で流す)
    .venv/bin/python tools/decode_gpu_trace.py --synthetic

    # dylib と集計部だけの smoke (GPU を数秒使う)
    BIGLOCK_PRIO=2 tools/biglock.sh .venv/bin/python tools/decode_gpu_trace.py --smoke
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

DYLIB = REPO_ROOT / "tools" / "bridge" / "libmetal_probe.dylib"
OUT_DIR = REPO_ROOT / "bench" / "results"


# ---------------------------------------------------------------------------
# metal_probe.dylib の ctypes 包み
# ---------------------------------------------------------------------------
class Probe:
    """`tools/bridge/metal_probe.mm` の C ABI をそのまま呼ぶ。

    `available` が False のときは全メソッドが無害な no-op になり、
    呼び手 (壁時計の計測) はそのまま動く。
    """

    def __init__(self, path: Path = DYLIB):
        self.available = False
        self.install_hits: list[int] = []
        self.error: str | None = None
        self._lib = None
        if not path.exists():
            self.error = f"{path} が無い (tools/bridge/build_metal_probe.sh でビルドする)"
            return
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as e:  # pragma: no cover - 環境依存
            self.error = f"dylib を読めない: {e}"
            return
        lib.mp_install.argtypes = [ctypes.POINTER(ctypes.c_int * 8)]
        lib.mp_install.restype = ctypes.c_int
        lib.mp_debug.argtypes = [ctypes.c_int]
        lib.mp_debug.restype = None
        lib.mp_enable.argtypes = [ctypes.c_int]
        lib.mp_enable.restype = None
        lib.mp_reset.argtypes = []
        lib.mp_reset.restype = None
        lib.mp_quiesce.argtypes = [ctypes.c_int]
        lib.mp_quiesce.restype = ctypes.c_int
        lib.mp_stats.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
        ]
        lib.mp_stats.restype = None
        lib.mp_name.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double),
        ]
        lib.mp_name.restype = ctypes.c_int
        lib.mp_interval_count.argtypes = []
        lib.mp_interval_count.restype = ctypes.c_uint64
        lib.mp_swizzled_class.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                          ctypes.c_int]
        lib.mp_swizzled_class.restype = ctypes.c_int
        self._lib = lib

    KINDS = ("dispatchThreadgroups", "dispatchThreads", "setComputePipelineState",
             "commit", "newComputePipelineState* (前置き一致)", "-", "-",
             "dispatchThreadgroupsIndirect")

    def swizzled(self) -> dict[str, list[str]]:
        """種類ごとに、実際に差し替えたクラス名。差し替えが Metal の具象クラス
        だけに当たっているかを目で見るため。"""
        out: dict[str, list[str]] = {}
        if self._lib is None:
            return out
        buf = ctypes.create_string_buffer(256)
        for kind, label in enumerate(self.KINDS):
            names = []
            for idx in range(48):
                if self._lib.mp_swizzled_class(kind, idx, buf, 256) < 0:
                    break
                names.append(buf.value.decode("utf-8", "replace"))
            out[label] = names
        return out

    def install(self) -> bool:
        """**MLX が GPU を 1 回でも使ったあとに呼ぶこと。**Metal ドライバの
        具象クラスはそれまでロードされない。"""
        if self._lib is None:
            return False
        hits = (ctypes.c_int * 8)()
        total = self._lib.mp_install(ctypes.byref(hits))
        self.install_hits = list(hits)
        if total <= 0:
            self.error = "差し替えるクラスが見つからない (Metal 未初期化?)"
            return False
        self.available = True
        return True

    def debug(self, n: int) -> None:
        """pipeline の名前がどこに入っているかを stderr に n 件だけ吐かせる。"""
        if self._lib is not None:
            self._lib.mp_debug(n)

    def enable(self, on: bool) -> None:
        if self._lib is not None:
            self._lib.mp_enable(1 if on else 0)

    def reset(self) -> None:
        if self._lib is not None:
            self._lib.mp_reset()

    def quiesce(self, timeout_ms: int = 5000) -> int:
        if self._lib is None:
            return 0
        return int(self._lib.mp_quiesce(timeout_ms))

    def stats(self) -> dict:
        if self._lib is None:
            return {"dispatches": 0, "command_buffers": 0, "cb_with_dispatch": 0,
                    "gpu_sum_ms": 0.0, "gpu_union_ms": 0.0, "kernels": []}
        d = ctypes.c_uint64()
        cb = ctypes.c_uint64()
        cbd = ctypes.c_uint64()
        gs = ctypes.c_double()
        gu = ctypes.c_double()
        nn = ctypes.c_int()
        self._lib.mp_stats(ctypes.byref(d), ctypes.byref(cb), ctypes.byref(cbd),
                           ctypes.byref(gs), ctypes.byref(gu), ctypes.byref(nn))
        kernels = []
        buf = ctypes.create_string_buffer(512)
        cnt = ctypes.c_uint64()
        ms = ctypes.c_double()
        for i in range(nn.value):
            rc = self._lib.mp_name(i, buf, 512, ctypes.byref(cnt), ctypes.byref(ms))
            if rc < 0:
                continue
            if cnt.value == 0:
                continue
            kernels.append({"name": buf.value.decode("utf-8", "replace"),
                            "count": int(cnt.value), "gpu_ms": float(ms.value)})
        kernels.sort(key=lambda k: (-k["gpu_ms"], -k["count"]))
        return {
            "dispatches": int(d.value),
            "command_buffers": int(cb.value),
            "cb_with_dispatch": int(cbd.value),
            "gpu_sum_ms": float(gs.value),
            "gpu_union_ms": float(gu.value),
            "intervals": int(self._lib.mp_interval_count()),
            "kernels": kernels,
        }


# ---------------------------------------------------------------------------
# ioreg の Device Utilization (probe の裏取り。粒度 0.1s、ラウンドより粗い)
# ---------------------------------------------------------------------------
class UtilSampler(threading.Thread):
    """`tools/gpu_util_sampler.py` と同じ ioreg の読み方を、同じプロセスの
    背景スレッドで回す (別プロセスを立てずに窓の平均だけ取るため)。"""

    def __init__(self, interval: float = 0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()

    def run(self) -> None:
        from gpu_util_sampler import sample

        while not self._stop.is_set():
            try:
                u = sample()
            except Exception:
                u = None
            if u is not None:
                self.samples.append(u)
            self._stop.wait(self.interval)

    def stop(self) -> list[int]:
        self._stop.set()
        self.join(timeout=3.0)
        return self.samples


def ioreg_once() -> int | None:
    try:
        from gpu_util_sampler import sample

        return sample()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ラウンドの計測
# ---------------------------------------------------------------------------
def run_rounds(eng, caches, snap, resume, base_pos, warmup: int, rounds: int,
               probe: Probe | None, util: bool):
    """control 済みの状態から decode を流し、warmup を過ぎた区間だけ測る。

    戻り値は dict。`probe` が None なら壁時計と ioreg だけ。

    eos は渡さない (`eos_ids=()`)。ラウンド数を宣言どおりに揃えるためで、
    トークン列自体は eos を渡した場合と同じ (停止判定に使うだけ)。
    """
    import mlx.core as mx

    from decode_ab import _restore

    _restore(caches, snap)
    empty = mx.zeros((1, 0), dtype=mx.int32)
    # 1 ラウンドが返すトークンは高々 depth+1 なので、余裕を持って上限を置く。
    n_tokens = (warmup + rounds) * 8 + 16
    gen = eng.generate_stream(empty, n_tokens, caches=caches, eos_ids=(),
                              resume=resume, base_pos=base_pos, temp=0.0)

    round_ms: list[float] = []
    toks_per_round: list[int] = []
    sampler = None
    win_t0 = None
    win_t1 = None
    n_tok_window = 0
    try:
        for i in range(warmup + rounds):
            if i == warmup:
                if probe is not None:
                    probe.quiesce()
                    probe.reset()
                    probe.enable(True)
                if util:
                    sampler = UtilSampler()
                    sampler.start()
                win_t0 = time.perf_counter()
            t0 = time.perf_counter()
            try:
                toks = next(gen)
            except StopIteration:
                break
            dt = time.perf_counter() - t0
            if i >= warmup:
                round_ms.append(dt * 1000.0)
                toks_per_round.append(len(toks))
                n_tok_window += len(toks)
        win_t1 = time.perf_counter()
    finally:
        if win_t1 is None:
            win_t1 = time.perf_counter()
        gen.close()
        if probe is not None:
            probe.enable(False)

    pending = probe.quiesce() if probe is not None else 0
    samples = sampler.stop() if sampler is not None else []

    n = len(round_ms)
    wall_ms = (win_t1 - win_t0) * 1000.0 if win_t0 is not None else 0.0
    out = {
        "rounds": n,
        "window_wall_ms": wall_ms,
        "wall_ms_per_round": wall_ms / max(n, 1),
        "round_ms_median": statistics.median(round_ms) if round_ms else 0.0,
        "round_ms_min": min(round_ms) if round_ms else 0.0,
        "tokens": n_tok_window,
        "tok_per_round": n_tok_window / max(n, 1),
        "ms_per_token": wall_ms / max(n_tok_window, 1),
        "ioreg_util_samples": len(samples),
        "ioreg_util_mean": (sum(samples) / len(samples)) if samples else None,
        "ioreg_util_min": min(samples) if samples else None,
        # ラウンドごとの生値 (区間が安定しているかを後から見るため)
        "round_ms": [round(v, 4) for v in round_ms],
        "tokens_per_round": toks_per_round,
    }
    if probe is not None:
        st = probe.stats()
        st["completion_pending_after"] = pending
        out["probe"] = st
        d = max(st["dispatches"], 1)
        out["dispatch_per_round"] = st["dispatches"] / max(n, 1)
        out["cb_per_round"] = st["command_buffers"] / max(n, 1)
        out["gpu_sum_ms_per_round"] = st["gpu_sum_ms"] / max(n, 1)
        out["gpu_union_ms_per_round"] = st["gpu_union_ms"] / max(n, 1)
        out["gpu_busy_frac"] = st["gpu_union_ms"] / wall_ms if wall_ms > 0 else None
        if st["dispatches"] > 0:
            out["mean_kernel_gpu_us"] = st["gpu_sum_ms"] * 1000.0 / d
            out["mean_gap_us"] = (wall_ms - st["gpu_union_ms"]) * 1000.0 / d
        else:
            out["mean_kernel_gpu_us"] = None
            out["mean_gap_us"] = None
    return out


def fmt_case(name: str, res: dict, res_clean: dict | None) -> str:
    lines = [f"--- {name} ---"]
    lines.append(
        f"  ラウンド {res['rounds']}  壁時計 {res['wall_ms_per_round']:.2f} ms/round "
        f"(中央値 {res['round_ms_median']:.2f}, 最小 {res['round_ms_min']:.2f})  "
        f"tok/round {res['tok_per_round']:.2f}  ms/tok {res['ms_per_token']:.2f}"
    )
    if res_clean is not None:
        base = res_clean["wall_ms_per_round"]
        over = (res["wall_ms_per_round"] / base - 1.0) * 100.0 if base else 0.0
        note = ""
        if res_clean.get("ioreg_util_mean") is not None:
            note = "  ** ioreg のサンプリングでこの走行自体が遅くなっている **"
        lines.append(
            f"  probe 無しの壁時計 {base:.2f} ms/round  → probe の上乗せ {over:+.1f}%{note}"
        )
        if res_clean.get("ioreg_util_mean") is not None:
            lines.append(
                f"  ioreg Device Utilization: 平均 {res_clean['ioreg_util_mean']:.1f}% "
                f"(最小 {res_clean['ioreg_util_min']}%, n={res_clean['ioreg_util_samples']})"
            )
    p = res.get("probe")
    if not p:
        lines.append("  (probe 無し: dispatch 数と GPU 時間は取れていない)")
        return "\n".join(lines)
    lines.append(
        f"  dispatch {res['dispatch_per_round']:.1f} /round   "
        f"command buffer {res['cb_per_round']:.1f} /round   "
        f"カーネル名 {len(p['kernels'])} 種"
    )
    lines.append(
        f"  GPU 時間 和 {res['gpu_sum_ms_per_round']:.2f} ms/round   "
        f"和集合 {res['gpu_union_ms_per_round']:.2f} ms/round   "
        f"稼働率 {100.0 * (res['gpu_busy_frac'] or 0):.1f}%"
    )
    if res.get("mean_kernel_gpu_us") is None:
        lines.append("  dispatch が 0 件 (GPU を通っていない、または差し替えが効いていない)")
    else:
        lines.append(
            f"  カーネル平均 {res['mean_kernel_gpu_us']:.1f} us   "
            f"隙間平均 {res['mean_gap_us']:.1f} us/dispatch"
        )
    if p.get("completion_pending_after"):
        lines.append(f"  ** completion handler の未消化 {p['completion_pending_after']} 件 **")
    lines.append(f"  上位カーネル (GPU 時間、1 round あたりに直した値):")
    n = max(res["rounds"], 1)
    lines.append(f"    {'カーネル':<52} {'回/round':>9} {'ms/round':>9} {'us/回':>8}")
    for k in p["kernels"][:30]:
        per_round_n = k["count"] / n
        per_round_ms = k["gpu_ms"] / n
        us = k["gpu_ms"] * 1000.0 / max(k["count"], 1)
        nm = k["name"]
        if len(nm) > 52:
            nm = nm[:24] + "…" + nm[-27:]
        lines.append(f"    {nm:<52} {per_round_n:9.1f} {per_round_ms:9.3f} {us:8.1f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 実モデルの走行
# ---------------------------------------------------------------------------
def build_cases(args) -> list[dict]:
    """--cases を (名前, ctx, depth 上書き) に展開する。"""
    spec = {
        "short": {"ctx": 0, "depth": None},
        "17k": {"ctx": args.long_ctx, "depth": None},
        "depth0": {"ctx": 0, "depth": 0},
    }
    out = []
    for nm in [c.strip() for c in args.cases.split(",") if c.strip()]:
        if nm not in spec:
            raise SystemExit(f"未知の case: {nm} (使えるのは {','.join(spec)})")
        out.append({"name": nm, **spec[nm]})
    return out


def main_real(args) -> int:
    import mlx.core as mx

    # probe は Metal のドライバのクラスが出来てから入れる。ここで軽い GPU op を
    # 1 回だけ流してデバイスを起こす (モデルを読む前なので数 ms)。
    mx.eval(mx.ones((8, 8)) @ mx.ones((8, 8)))
    probe = Probe()
    if probe.install():
        print(f"metal_probe: 差し替え {probe.install_hits}")
        for k, v in probe.swizzled().items():
            print(f"    {k:<28} {v}")
        if args.debug_names:
            probe.debug(args.debug_names)
    else:
        print(f"metal_probe: 使えない ({probe.error})。壁時計と ioreg だけになる。")
    probe_arg = probe if probe.available else None

    from verify_width_cost import build_prompt_ids, build_runner
    from decode_ab import prefill_once

    t0 = time.perf_counter()
    eng, model, tok, eos_ids = build_runner(args)
    print(f"モデル読み込み {time.perf_counter() - t0:.1f}s  device={mx.device_info()['device_name']}")

    from mlxturbo import spec_flash

    idle = ioreg_once()
    print(f"ioreg Device Utilization (計測前の待機時): {idle}%")

    cases = build_cases(args)
    results = {}
    default_depth = eng.depth
    default_limit = eng.depth_ctx_limit
    default_adapt = eng._depth_adapt

    # プロセス起動直後の 1 本目は +7〜9% 遅い (CLAUDE.md)。捨てる 1 本を先に流す。
    print("burn-in (捨てる 1 本)...", flush=True)
    ids0 = build_prompt_ids(tok, 0)
    caches0, snap0, resume0, _ = prefill_once(eng, ids0, eos_ids)
    run_rounds(eng, caches0, snap0, resume0, ids0.shape[1], 4, 16, None, False)
    del caches0, snap0, resume0
    mx.clear_cache()

    for case in cases:
        name = case["name"]
        print(f"\n=== case {name} (ctx={case['ctx']}, depth={case['depth']}) ===", flush=True)
        if case["depth"] is None:
            eng.depth = default_depth
            eng.depth_ctx_limit = default_limit
            eng._depth_adapt = default_adapt
        else:
            # decode_ab の --depth と同じ入れ方 (ctx_limit も上げて、engine が
            # 文脈長で depth を落とさないようにする)。適応 depth も切る。
            eng.depth = case["depth"]
            eng.depth_ctx_limit = 1 << 30
            eng._depth_adapt = False

        ids = build_prompt_ids(tok, case["ctx"])
        n = ids.shape[1]
        t0 = time.perf_counter()
        caches, snap, resume, _first = prefill_once(eng, ids, eos_ids)
        print(f"  prefill n={n} ({time.perf_counter() - t0:.1f}s)", flush=True)

        # clean → probe → clean の回文順で流す (probe の上乗せを、時間ドリフト
        # と同じ向きに乗せないため。decode_ab の A→B→B→A と同じ理屈)。
        clean1 = run_rounds(eng, caches, snap, resume, n, args.warmup, args.rounds,
                            None, args.ioreg)
        res = run_rounds(eng, caches, snap, resume, n, args.warmup, args.rounds, probe_arg, False)
        clean2 = run_rounds(eng, caches, snap, resume, n, args.warmup, args.rounds,
                            None, args.ioreg)
        res_clean = dict(clean1)
        res_clean["wall_ms_per_round"] = (clean1["wall_ms_per_round"]
                                          + clean2["wall_ms_per_round"]) / 2.0
        res_clean["reps"] = [clean1["wall_ms_per_round"], clean2["wall_ms_per_round"]]
        if clean2.get("ioreg_util_mean") is not None and clean1.get("ioreg_util_mean") is not None:
            res_clean["ioreg_util_mean"] = (clean1["ioreg_util_mean"]
                                            + clean2["ioreg_util_mean"]) / 2.0
            res_clean["ioreg_util_samples"] = (clean1["ioreg_util_samples"]
                                               + clean2["ioreg_util_samples"])
            res_clean["ioreg_util_min"] = min(clean1["ioreg_util_min"], clean2["ioreg_util_min"])

        res["ctx"] = n
        res["ctx_arg"] = case["ctx"]
        res["depth_override"] = case["depth"]
        res["engine_depth"] = eng.depth
        res["depth_adapt"] = bool(eng._depth_adapt)
        res["mtp_depth_default"] = spec_flash.MTP_DEPTH
        res["clean"] = res_clean
        results[name] = res
        print(fmt_case(name, res, res_clean), flush=True)

        out_path = Path(args.out_dir) / f"decode-gpu-trace-{name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"  書き出し: {out_path}", flush=True)

        del caches, snap, resume
        mx.clear_cache()

    combined = Path(args.out_dir) / "decode-gpu-trace.json"
    combined.write_text(json.dumps({
        "device": mx.device_info().get("device_name"),
        "model": args.model,
        "ngram": args.ngram,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "probe_available": probe.available,
        "probe_install_hits": probe.install_hits,
        "ioreg_idle_before": idle,
        "cases": results,
    }, ensure_ascii=False, indent=1))
    print(f"\nまとめ: {combined}")

    sys.stdout.flush()
    sys.stderr.flush()
    # verify_width_cost.py と同じ理由: interpreter shutdown 待ちで 98GB を
    # 握ったままプロセスが残った実測があるので、書き終えたら即落とす。
    os._exit(0)


# ---------------------------------------------------------------------------
# 検算 (GPU 無し / 数秒)
# ---------------------------------------------------------------------------
def main_synthetic(args) -> int:
    """合成の小さい Flash-Next を CPU で流し、引数の解釈・ラウンド計測ループ・
    集計・整形・JSON 書き出しを通す。Metal が無いので probe は必ず不発
    (`available=False`) になり、その経路も一緒に確かめる。"""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    import mlxturbo  # noqa: F401
    from mlxturbo.mtp_flash import FlashMTPModule
    from mlxturbo.spec_flash import FlashSpecEngine
    from verify_batch_spec import build

    from decode_ab import prefill_once

    probe = Probe()
    installed = probe.install()
    # Metal のクラス自体は (mlx が Metal.framework にリンクしているので) 見つかり
    # うるが、CPU デバイスでは 1 回も呼ばれないので dispatch は 0 のまま。
    # 「差し替えが入っても 0 件で壊れない」経路をここで通す。
    print(f"probe.install() -> {installed} err={probe.error} hits={probe.install_hits}")

    model = build()
    mtp = FlashMTPModule(model.args.text, variant="lane")
    mx.eval(mtp.parameters())
    eng = FlashSpecEngine(model, mtp)

    vocab = model.args.text.vocab_size
    ids = mx.array([[(i * 7 + 3) % vocab for i in range(24)]], dtype=mx.int32)
    caches, snap, resume, _ = prefill_once(eng, ids, ())
    n = ids.shape[1]

    for case in build_cases(args):
        if case["depth"] is not None:
            eng.depth = case["depth"]
            eng.depth_ctx_limit = 1 << 30
            eng._depth_adapt = False
        res = run_rounds(eng, caches, snap, resume, n, 2, 4,
                         probe if probe.available else None, False)
        res_clean = run_rounds(eng, caches, snap, resume, n, 2, 4, None, False)
        res["clean"] = res_clean
        print(fmt_case(f"synthetic:{case['name']}", res, res_clean))
        assert res["rounds"] == 4, res["rounds"]
        assert res["wall_ms_per_round"] > 0.0
        # probe が居ない経路では dispatch 系のキーが出ないこと
        assert ("probe" in res) == probe.available

    # 集計の検算: 作り話の probe 出力で fmt_case が壊れないこと
    fake = {
        "rounds": 10, "window_wall_ms": 350.0, "wall_ms_per_round": 35.0,
        "round_ms_median": 35.0, "round_ms_min": 34.0, "tokens": 20,
        "tok_per_round": 2.0, "ms_per_token": 17.5,
        "ioreg_util_samples": 0, "ioreg_util_mean": None, "ioreg_util_min": None,
        "dispatch_per_round": 800.0, "cb_per_round": 80.0,
        "gpu_sum_ms_per_round": 30.0, "gpu_union_ms_per_round": 29.0,
        "gpu_busy_frac": 29.0 / 35.0, "mean_kernel_gpu_us": 37.5, "mean_gap_us": 7.5,
        "probe": {"dispatches": 8000, "command_buffers": 800, "cb_with_dispatch": 800,
                  "gpu_sum_ms": 300.0, "gpu_union_ms": 290.0, "intervals": 800,
                  "completion_pending_after": 0,
                  "kernels": [{"name": "steel_gemm_" + "x" * 60, "count": 970,
                               "gpu_ms": 120.0}]},
    }
    print(fmt_case("synthetic:fake-aggregation", fake, None))
    out = Path(args.out_dir) / "decode-gpu-trace-synthetic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fake, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out}\n合成の検算は通った。")
    return 0


def main_smoke(args) -> int:
    """dylib だけの smoke (GPU を数秒使う)。実モデルは読まない。"""
    import mlx.core as mx

    a = mx.random.normal((512, 512))
    mx.eval(a)
    probe = Probe()
    if not probe.install():
        print(f"install 失敗: {probe.error}")
        return 1
    print(f"差し替え {probe.install_hits}")
    for k, v in probe.swizzled().items():
        print(f"  {k:<26} {v}")
    probe.reset()
    if args.debug_names:
        probe.debug(args.debug_names)
    probe.enable(True)
    t0 = time.perf_counter()
    for _ in range(20):
        b = a @ a
        b = mx.exp(b * 1e-4)
        mx.eval(b)
    wall = (time.perf_counter() - t0) * 1000.0
    probe.enable(False)
    print(f"quiesce -> {probe.quiesce()}")
    st = probe.stats()
    print(f"壁時計 {wall:.2f} ms  dispatch {st['dispatches']}  CB {st['command_buffers']}  "
          f"GPU 和 {st['gpu_sum_ms']:.2f} ms  和集合 {st['gpu_union_ms']:.2f} ms")
    for k in st["kernels"][:10]:
        print(f"  {k['name']:<60} {k['count']:5d} {k['gpu_ms']:8.3f} ms")
    ok = st["dispatches"] > 0 and st["gpu_sum_ms"] > 0 and st["gpu_union_ms"] <= wall * 1.05
    print("OK" if ok else "** 整合しない (GPU 時間 > 壁時計、または 0) **")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm")
    ap.add_argument("--ngram", default="~/models/ddalcu-ngram")
    ap.add_argument("--mtp", default=None, help="既定は --model の中の mtp.safetensors")
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--cases", default="short,17k,depth0",
                    help="short / 17k / depth0 をカンマ区切りで")
    ap.add_argument("--long-ctx", type=int, default=17000, help="17k case の文脈長")
    ap.add_argument("--rounds", type=int, default=64, help="測る区間のラウンド数")
    ap.add_argument("--warmup", type=int, default=8, help="測る前に捨てるラウンド数")
    ap.add_argument("--ioreg", action="store_true",
                    help="probe 無しの走行で ioreg の Device Utilization も採る "
                         "(probe の裏取り)。**ioreg は 0.1s ごとに別プロセスを起こす"
                         "ので、その走行の壁時計が 3〜13%% 遅くなる** (2026-09-03 実測: "
                         "17k で probe 有り 35.66 ms/round に対し ioreg 付きの clean が "
                         "40.22 ms/round)。壁時計の裏取りには使わないこと")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--synthetic", action="store_true",
                    help="合成モデル (CPU、数秒) で道具側だけ検算する")
    ap.add_argument("--split-cb", action="store_true",
                    help="MLX_MAX_OPS_PER_BUFFER=1 で 1 command buffer = 1 op に割り、"
                         "カーネル別の GPU 時間の配分を厳密にする。commit の回数が"
                         "増えて壁時計が変わるので、絶対値の比較には使わないこと")
    ap.add_argument("--debug-names", type=int, default=0,
                    help="pipeline の名前の在り処を先頭 N 件だけ stderr に吐く (診断用)")
    ap.add_argument("--smoke", action="store_true",
                    help="dylib だけの smoke (GPU を数秒。実モデルは読まない)")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.rounds < 1 or args.warmup < 0:
        raise SystemExit("--rounds は 1 以上、--warmup は 0 以上")
    if args.split_cb:
        # MLX はデバイス初期化時にこの env を読む。mlx を import する前に立てる
        # (この関数はまだ mlx を触っていない)。
        os.environ["MLX_MAX_OPS_PER_BUFFER"] = "1"
    if args.synthetic:
        return main_synthetic(args)
    if args.smoke:
        return main_smoke(args)
    return main_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
