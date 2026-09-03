"""decode 1 ラウンドの GPU 時間と dispatch を **モジュール別**に帰属する。

## なぜ既存の道具で足りないか

`tools/decode_gpu_trace.py` はカーネル名ごとの回数と GPU 時間までしか出せない
(dylib は MLX の eval スレッドから呼ばれるので Python のスタックが無い)。
ところが `affine_qmv_fast_b4` は attention の q/k/v/o、GDN の in_proj/out_proj、
HC の低ランク 2 本、shared expert、lm_head の**全部**が使う。名前だけでは
「どのモジュールが何 ms 使ったか」が割れない。

`tools/decode_copy_probe.py` はモジュール別の op 数を出すが、時間は出さない。

ここは**両方を 1 回の走行で直接測る**。カーネル名からの按分をしない。

## やりかた

モジュールの `__call__` (と、いくつかの内側のメソッド) を包み、

    入口: 引数から辿れる配列を mx.eval (上流を流し切る) → probe の計数を控える
    出口: 戻り値とキャッシュを mx.eval             → probe の計数との差を取る

差から**子の region の合計を引いた残り**がその region の self 時間になる
(普通のプロファイラの self/total と同じ勘定)。probe はカーネル名ごとの
累計も持っているので、**region x カーネル名**の表まで取れる。

## 何が歪むか (報告に必ず書くこと。2026-09-03 実測)

1. **境界ごとに mx.eval + quiesce を強制するので GPU 時間そのものが膨らむ。**
   region が自分の command buffer を持つと層をまたぐ重なりが消えるためで、
   短文脈 S=1 で GPU 和は L1 x1.01 / L2 x2.26 / L3 x2.84 になった。
   膨らみは region の呼び出し回数に比例せず、一律スケールでも CB 数の
   引き算でも直せない (引き算は安い region が負になる)。
   **だから GPU 時間は L1 (位相だけ) しか信用しない。**
   モジュール別の時間は `--split-cb` のカーネル単価 x ここで採った回数、
   行列積は重みバイト数の按分で出すこと (`decode-anatomy-splitcb-*.json`)。
2. **dispatch 数と region x カーネル名の回数は歪まない** (self 合計 3662 対
   素 3638、+0.7%)。この道具の主産物はそちら。
3. GPU 時間は command buffer の completion handler が動いて初めて台帳に載る。
   `mx.eval` はハンドラまでは待たないので、境界の読み取り前に `quiesce` を
   挟む (入れないと「直前の region の時間が次の region に付く」)。
4. `capture()` (spec_flash) が検証フォワードの間だけ `GatedDeltaNet.__call__` と
   `GatedResidual.__call__` と `PLELayer._short_conv` を差し替えるので、
   こちらは `capture` そのものを包んで、**差し替わった後の関数**を包み直す。
   これをしないと GDN が丸ごと `pre_mlp` の残余に落ちる。
5. region ごとのカーネル一覧は上位 12 種で切ってある (`Regions.table`)。
   溢れた dispatch は `self_dispatch_per_round` との差で分かるので、
   読み手が平均単価で埋めること。

## 使いかた

    tools/biglock.sh .venv/bin/python tools/decode_module_attrib.py \\
        --model ~/models/ddalcu-mlxlm-head4 --ngram ~/models/ddalcu-ngram

    # 道具側だけの検算 (合成モデル、CPU、数秒。probe は不発になる)
    .venv/bin/python tools/decode_module_attrib.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

OUT_DIR = REPO_ROOT / "bench" / "results"


# ---------------------------------------------------------------------------
# 引数・戻り値から mx.array を掘り出す (decode_copy_probe と同じ理屈)
# ---------------------------------------------------------------------------
def _arrays(obj, acc, depth=0):
    import mlx.core as mx

    if depth > 3:
        return
    if isinstance(obj, mx.array):
        acc.append(obj)
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            _arrays(o, acc, depth + 1)
    elif isinstance(obj, dict):
        for o in obj.values():
            _arrays(o, acc, depth + 1)


def _cache_arrays(objs, acc):
    """引数に混ざったキャッシュ (mlx_lm の *Cache) の中身を拾う。

    GDN の conv 状態や KV は戻り値ではなくキャッシュに書かれるので、ここを
    見ないとその dispatch が「あとで誰かの region で評価される」ことになる。
    """
    for v in objs:
        if isinstance(v, (list, tuple)):
            _cache_arrays(v, acc)
            continue
        st = getattr(v, "state", None)
        if st is not None and not callable(st):
            _arrays(st, acc)
        idx = getattr(v, "indexer", None)
        if idx is not None:
            st = getattr(idx, "state", None)
            if st is not None and not callable(st):
                _arrays(st, acc)


def _flush(*objs):
    import mlx.core as mx

    acc: list = []
    _arrays(list(objs), acc)
    _cache_arrays(list(objs), acc)
    if acc:
        try:
            mx.eval(*acc)
        except Exception:      # noqa: BLE001  評価できない中間 (非 mx) は捨てる
            pass


# ---------------------------------------------------------------------------
# region の勘定
# ---------------------------------------------------------------------------
class Regions:
    """self/total の勘定。probe の累計との差で region ごとの dispatch と GPU 時間。"""

    def __init__(self, probe):
        self.probe = probe
        self.enabled = False
        self.stack: list[dict] = []
        # 名前 -> [self dispatch, self gpu_ms, 呼び出し回数, total dispatch,
        #          total gpu_ms, self command buffer 数]
        self.acc: dict[str, list] = defaultdict(lambda: [0, 0.0, 0, 0, 0.0, 0])
        self.kern: dict[str, Counter] = defaultdict(Counter)   # 名前 -> {カーネル: 回数}
        self.kern_ms: dict[str, dict] = defaultdict(lambda: defaultdict(float))
        self.rounds = 0
        self._round0 = None
        # 包みの深さ。level 1 = 位相だけ、2 = 層の大きい塊、3 = 内側まで。
        # 深くするほど region ごとに command buffer が割れて GPU 時間が膨らむので、
        # **浅い level を錨にして深い level の内訳を読む**。
        self.level = 3
        # いま居る位相 (draft / verify)。MTP の draft も同じ vendor のクラスを
        # 使うので、位相で名前を分けないと verify の attention と混ざる。
        self.phase = None

    # -- probe の読み取り --------------------------------------------------
    def _snap(self):
        if self.probe is None:
            return 0, 0.0, {}, 0
        # **GPU 時間は command buffer の completion handler が動いて初めて
        # 台帳に載る。**mx.eval は GPU の完了までは待つがハンドラの実行までは
        # 待たないので、quiesce を挟まないと「直前の region の時間が次の
        # region に付く」ずれ方をする (実測: `_qkv` が 444 dispatch で 0.000 ms、
        # その時間が親の `Attention` に乗った)。dispatch 数は投入時に数える
        # ので、こちらは quiesce 無しでも正しい。
        self.probe.quiesce(50)
        st = self.probe.stats()
        km = {k["name"]: (k["count"], k["gpu_ms"]) for k in st["kernels"]}
        return st["dispatches"], st["gpu_sum_ms"], km, st["command_buffers"]

    @staticmethod
    def _kdiff(now, before):
        out = {}
        for nm, (c, ms) in now.items():
            c0, ms0 = before.get(nm, (0, 0.0))
            if c > c0:
                out[nm] = (c - c0, ms - ms0)
        return out

    # -- 1 ラウンドの枠 ----------------------------------------------------
    def round_begin(self):
        if not self.enabled:
            return
        self._round0 = self._snap()

    def round_end(self):
        if not self.enabled or self._round0 is None:
            return
        d0, ms0, k0, cb0 = self._round0
        d1, ms1, k1, cb1 = self._snap()
        self.acc["(round total)"][0] += d1 - d0
        self.acc["(round total)"][1] += ms1 - ms0
        self.acc["(round total)"][2] += 1
        self.acc["(round total)"][5] += cb1 - cb0
        for nm, (c, ms) in self._kdiff(k1, k0).items():
            self.kern["(round total)"][nm] += c
            self.kern_ms["(round total)"][nm] += ms
        self.rounds += 1
        self._round0 = None

    # -- region --------------------------------------------------------------
    @contextmanager
    def region(self, name, enter_objs, exit_objs_fn, phase=None):
        if not self.enabled:
            yield
            return
        if phase is None and self.phase:
            name = f"{self.phase}/{name}"
        prev_phase = self.phase
        if phase is not None:
            self.phase = phase
        _flush(enter_objs)
        d0, ms0, k0, cb0 = self._snap()
        frame = {"cd": 0, "cms": 0.0, "ccb": 0,
                 "ck": Counter(), "ckms": defaultdict(float)}
        self.stack.append(frame)
        try:
            yield
        finally:
            self.stack.pop()
            self.phase = prev_phase
            _flush(exit_objs_fn())
            d1, ms1, k1, cb1 = self._snap()
            dd, dms, dcb = d1 - d0, ms1 - ms0, cb1 - cb0
            kd = self._kdiff(k1, k0)
            rec = self.acc[name]
            rec[0] += dd - frame["cd"]
            rec[1] += dms - frame["cms"]
            rec[2] += 1
            rec[3] += dd
            rec[4] += dms
            rec[5] += dcb - frame["ccb"]
            for nm, (c, ms) in kd.items():
                sc = c - frame["ck"].get(nm, 0)
                sms = ms - frame["ckms"].get(nm, 0.0)
                if sc:
                    self.kern[name][nm] += sc
                    self.kern_ms[name][nm] += sms
            if self.stack:
                p = self.stack[-1]
                p["cd"] += dd
                p["cms"] += dms
                p["ccb"] += dcb
                for nm, (c, ms) in kd.items():
                    p["ck"][nm] += c
                    p["ckms"][nm] += ms

    # -- 包み --------------------------------------------------------------
    def wrap_method(self, cls, meth, name, phase=None, level=3):
        orig = getattr(cls, meth)

        def wrapped(self_, *a, **kw):
            if not self.enabled or level > self.level:
                return orig(self_, *a, **kw)
            args = list(a) + list(kw.values())
            box = {}
            with self.region(name, args, lambda: (box.get("o"), args), phase=phase):
                box["o"] = orig(self_, *a, **kw)
            return box["o"]

        setattr(cls, meth, wrapped)
        return orig

    def wrap_static(self, cls, meth, name, level=3):
        orig = getattr(cls, meth)

        def wrapped(*a, **kw):
            if not self.enabled or level > self.level:
                return orig(*a, **kw)
            args = list(a) + list(kw.values())
            box = {}
            with self.region(name, args, lambda: box.get("o")):
                box["o"] = orig(*a, **kw)
            return box["o"]

        setattr(cls, meth, staticmethod(wrapped))
        return orig

    def wrap_func(self, mod, fname, name, phase=None, level=3):
        orig = getattr(mod, fname)

        def wrapped(*a, **kw):
            if not self.enabled or level > self.level:
                return orig(*a, **kw)
            args = list(a) + list(kw.values())
            box = {}
            with self.region(name, args, lambda: (box.get("o"), args), phase=phase):
                box["o"] = orig(*a, **kw)
            return box["o"]

        setattr(mod, fname, wrapped)
        return orig

    # -- 集計 --------------------------------------------------------------
    def table(self):
        n = max(self.rounds, 1)
        rows = []
        for name, (sd, sms, calls, td, tms, scb) in self.acc.items():
            rows.append({
                "region": name,
                "calls_per_round": calls / n,
                "self_cb_per_round": scb / n,
                "self_dispatch_per_round": sd / n,
                "self_gpu_ms_per_round": sms / n,
                "total_dispatch_per_round": td / n,
                "total_gpu_ms_per_round": tms / n,
                "kernels": sorted(
                    ({"name": k, "count_per_round": c / n,
                      "gpu_ms_per_round": self.kern_ms[name][k] / n}
                     for k, c in self.kern[name].items()),
                    key=lambda r: -r["gpu_ms_per_round"])[:12],
            })
        rows.sort(key=lambda r: -r["self_gpu_ms_per_round"])
        return rows


# ---------------------------------------------------------------------------
# 包む対象
# ---------------------------------------------------------------------------
def install_hooks(reg, eng):
    """モデル側とエンジン側の region を仕掛ける。

    **`enable_default_fusions` の後**に呼ぶこと (融合で差し替わった後の関数を
    包むため)。`capture()` の中で差し替わるものは、capture を包んで中で
    包み直す。
    """
    import mlxturbo  # noqa: F401
    from mlx_lm.models import qwen4_exp as Q
    from mlx_lm.models import switch_layers as SL
    from mlxturbo import spec_flash

    # --- モデルの層 -------------------------------------------------------
    reg.wrap_method(Q.DecoderLayer, "pre_mlp", "layer.pre_mlp", level=3)
    reg.wrap_static(Q.DecoderLayer, "_combine", "layer._combine(HC write)", level=3)
    reg.wrap_method(Q.SparseMoeBlock, "__call__", "MoE(SparseMoeBlock)", level=2)
    reg.wrap_method(SL.SwitchGLU, "__call__", "MoE.switch_mlp(experts)", level=3)
    reg.wrap_method(Q.MLP, "__call__", "MoE.shared_expert", level=3)
    reg.wrap_method(Q.Attention, "__call__", "Attention", level=2)
    reg.wrap_method(Q.Attention, "_qkv", "Attention._qkv(norm+rope+qkv)", level=3)
    if hasattr(Q, "QSAIndexer"):
        reg.wrap_method(Q.QSAIndexer, "__call__", "QSAIndexer", level=3)
    reg.wrap_method(Q.GatedDeltaNet, "_project_in", "GDN._project_in", level=3)
    if hasattr(Q, "PLELayer"):
        reg.wrap_method(Q.PLELayer, "__call__", "PLE", level=2)
    # capture の外 (prefill 等) 用。capture 内では下で包み直す
    reg.wrap_method(Q.GatedDeltaNet, "__call__", "GDN(GatedDeltaNet)", level=2)
    reg.wrap_method(Q.GatedResidual, "__call__", "HC(GatedResidual)", level=3)

    # --- capture の中での差し替えに追随 -----------------------------------
    orig_capture = spec_flash.capture

    @contextmanager
    def capture_wrapped(model, light=False):
        with orig_capture(model, light=light) as cap:
            saved = []
            for cls, meth, nm, lv in ((Q.GatedDeltaNet, "__call__", "GDN(GatedDeltaNet)", 2),
                                      (Q.GatedResidual, "__call__", "HC(GatedResidual)", 3),
                                      (Q.PLELayer, "_short_conv", "PLE._short_conv", 3)):
                saved.append((cls, meth, getattr(cls, meth)))
                reg.wrap_method(cls, meth, nm, level=lv)
            try:
                yield cap
            finally:
                for cls, meth, fn in saved:
                    setattr(cls, meth, fn)

    spec_flash.capture = capture_wrapped

    # --- エンジンの位相 ---------------------------------------------------
    reg.wrap_func(spec_flash, "_staged_forward", "_staged_forward", phase="V", level=1)
    E = type(eng)
    reg.wrap_method(E, "_draft_chain", "_draft_chain", phase="D", level=1)
    reg.wrap_method(E, "_verify", "_verify(sample+compare)", phase="X", level=1)
    if hasattr(E, "_presync_step0"):
        reg.wrap_method(E, "_presync_step0", "_presync_step0", phase="D", level=1)
    if hasattr(E, "_prime_accepted_gap"):
        reg.wrap_method(E, "_prime_accepted_gap", "_prime_accepted_gap", phase="P", level=1)


# ---------------------------------------------------------------------------
# 重みのバイト数の棚卸し (帯域律速の分母。GPU 不要)
# ---------------------------------------------------------------------------
def weight_census(model) -> dict:
    """モジュール種別ごとの「1 トークンの forward で読む重みのバイト数」。

    量子化された `QuantizedLinear` / `QuantizedSwitchLinear` は
    `weight` (uint32 に詰めた) + `scales` + `biases` の実バイトを足す。
    MoE の専門家は **選ばれた top_k ぶんだけ**読むので、層あたりの
    全専門家のバイト数を (top_k / num_experts) で割って計上する。
    """
    import mlx.core as mx

    def nbytes(a):
        return int(a.size) * a.dtype.size if isinstance(a, mx.array) else 0

    def mod_bytes(m):
        tot = 0
        for k in ("weight", "scales", "biases"):
            v = getattr(m, k, None)
            if isinstance(v, mx.array):
                tot += nbytes(v)
        return tot

    text = model.args.text
    top_k = text.num_experts_per_tok
    n_exp = text.num_experts
    out: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for li, layer in enumerate(model.model.layers):
        if getattr(layer, "self_attn", None) is not None:
            at = layer.self_attn
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                m = getattr(at, nm, None)
                if m is not None:
                    out["Attention.qkvo"] += mod_bytes(m)
                    counts["Attention.qkvo"] += 1
            idx = getattr(at, "indexer", None)
            if idx is not None:
                for nm, m in idx.__dict__.items():
                    if hasattr(m, "weight"):
                        out["QSAIndexer"] += mod_bytes(m)
        gdn = getattr(layer, "linear_attn", None)
        if gdn is not None:
            for nm in ("in_proj_qkvz", "in_proj_ba", "out_proj", "conv1d"):
                m = getattr(gdn, nm, None)
                if m is not None:
                    key = "GDN.conv1d" if nm == "conv1d" else "GDN.proj"
                    out[key] += mod_bytes(m)
            # 名前が版で違うので、残りの Linear も拾う
            for nm, m in vars(gdn).items():
                if nm in ("in_proj_qkvz", "in_proj_ba", "out_proj", "conv1d"):
                    continue
                if hasattr(m, "weight") and not nm.endswith("norm"):
                    out["GDN.proj_other"] += mod_bytes(m)
        moe = getattr(layer, "mlp", None)
        if moe is not None:
            sm = getattr(moe, "switch_mlp", None)
            if sm is not None:
                per_layer = 0
                for nm in ("gate_proj", "up_proj", "down_proj"):
                    m = getattr(sm, nm, None)
                    if m is not None:
                        per_layer += mod_bytes(m)
                out["MoE.experts(top_k 分)"] += per_layer * top_k / n_exp
                out["MoE.experts(全専門家)"] += per_layer
            se = getattr(moe, "shared_expert", None)
            if se is not None:
                for nm in ("gate_proj", "up_proj", "down_proj"):
                    m = getattr(se, nm, None)
                    if m is not None:
                        out["MoE.shared_expert"] += mod_bytes(m)
            for nm in ("gate", "shared_expert_gate"):
                m = getattr(moe, nm, None)
                if m is not None:
                    out["MoE.router"] += mod_bytes(m)
        for nm in ("attn_hyper_connection", "mlp_hyper_connection"):
            hc = getattr(layer, nm, None)
            if hc is not None:
                for k2 in ("input_mix_weight_down", "input_mix_weight_up",
                           "block_inject_weight"):
                    m = getattr(hc, k2, None)
                    if m is not None:
                        out["HC.lowrank"] += mod_bytes(m)
        ple = getattr(layer, "ple", None)
        if ple is not None:
            for nm, m in vars(ple).items():
                if hasattr(m, "weight"):
                    out["PLE"] += mod_bytes(m)
    mixer = getattr(model.model, "hyper_connection_mixer", None)
    if mixer is not None:
        for k2 in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
            m = getattr(mixer, k2, None)
            if m is not None:
                out["HC.lowrank"] += mod_bytes(m)
    lm = getattr(model, "lm_head", None)
    if lm is not None:
        out["lm_head"] += mod_bytes(lm)
    return {k: float(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# ラウンドの走行
# ---------------------------------------------------------------------------
def run_rounds(eng, caches, snap, resume, base_pos, warmup, rounds, probe, reg):
    import mlx.core as mx

    from decode_ab import _restore

    _restore(caches, snap)
    empty = mx.zeros((1, 0), dtype=mx.int32)
    n_tokens = (warmup + rounds) * 8 + 16
    gen = eng.generate_stream(empty, n_tokens, caches=caches, eos_ids=(),
                              resume=resume, base_pos=base_pos, temp=0.0)
    round_ms, toks = [], []
    t_win0 = None
    try:
        for i in range(warmup + rounds):
            if i == warmup:
                if probe is not None:
                    probe.quiesce()
                    probe.reset()
                    probe.enable(True)
                if reg is not None:
                    reg.enabled = True
                t_win0 = time.perf_counter()
            if reg is not None and i >= warmup:
                reg.round_begin()
            t0 = time.perf_counter()
            try:
                out = next(gen)
            except StopIteration:
                break
            dt = time.perf_counter() - t0
            if reg is not None and i >= warmup:
                reg.round_end()
            if i >= warmup:
                round_ms.append(dt * 1000.0)
                toks.append(len(out))
        t_win1 = time.perf_counter()
    finally:
        if t_win0 is None:
            t_win0 = t_win1 = time.perf_counter()
        gen.close()
        if probe is not None:
            probe.enable(False)
        if reg is not None:
            reg.enabled = False
    if probe is not None:
        probe.quiesce()
    n = max(len(round_ms), 1)
    wall = (t_win1 - t_win0) * 1000.0
    res = {
        "rounds": len(round_ms),
        "wall_ms_per_round": wall / n,
        "round_ms_median": statistics.median(round_ms) if round_ms else 0.0,
        "tok_per_round": sum(toks) / n,
        "ms_per_token": wall / max(sum(toks), 1),
        "tokens_per_round": toks,
    }
    if probe is not None:
        st = probe.stats()
        res["dispatch_per_round"] = st["dispatches"] / n
        res["cb_per_round"] = st["command_buffers"] / n
        res["gpu_sum_ms_per_round"] = st["gpu_sum_ms"] / n
        res["gpu_union_ms_per_round"] = st["gpu_union_ms"] / n
        res["gpu_busy_frac"] = st["gpu_union_ms"] / wall if wall > 0 else None
        res["kernels"] = [{"name": k["name"], "count_per_round": k["count"] / n,
                           "gpu_ms_per_round": k["gpu_ms"] / n}
                          for k in st["kernels"]]
    return res


CASES = {
    "s1-short": {"ctx": 0, "depth": 0},
    "s3-short": {"ctx": 0, "depth": 2},
    "s1-17k": {"ctx": 17000, "depth": 0},
    "s3-17k": {"ctx": 17000, "depth": 2},
}


def fmt(name, plain, hooked, reg_rows):
    L = [f"\n=== {name} ==="]
    L.append(f"  素:     壁 {plain['wall_ms_per_round']:.2f} ms/round  "
             f"tok/round {plain['tok_per_round']:.2f}  "
             f"dispatch {plain.get('dispatch_per_round', 0):.0f}  "
             f"GPU 和 {plain.get('gpu_sum_ms_per_round', 0):.2f} ms")
    L.append(f"  hooked: 壁 {hooked['wall_ms_per_round']:.2f} ms/round  "
             f"tok/round {hooked['tok_per_round']:.2f}  "
             f"dispatch {hooked.get('dispatch_per_round', 0):.0f}  "
             f"GPU 和 {hooked.get('gpu_sum_ms_per_round', 0):.2f} ms")
    tot = next((r for r in reg_rows if r["region"] == "(round total)"), None)
    named = [r for r in reg_rows if r["region"] != "(round total)"]
    sd = sum(r["self_dispatch_per_round"] for r in named)
    sms = sum(r["self_gpu_ms_per_round"] for r in named)
    if tot:
        L.append(f"  検算: region の self 合計 dispatch {sd:.0f} / round total "
                 f"{tot['self_dispatch_per_round']:.0f}  "
                 f"(素の走行 {plain.get('dispatch_per_round', 0):.0f})")
        L.append(f"        self 合計 GPU {sms:.2f} ms / round total "
                 f"{tot['self_gpu_ms_per_round']:.2f} ms")
    scale = (plain.get("gpu_sum_ms_per_round", 0.0) / sms) if sms > 0 else 1.0
    L.append(f"  素の GPU 合計に合わせる係数 {scale:.3f}")
    L.append(f"    {'region':<34} {'呼/round':>8} {'self disp':>10} "
             f"{'self ms':>9} {'素換算ms':>9} {'%':>6}")
    for r in named:
        if r["self_dispatch_per_round"] < 0.5 and r["self_gpu_ms_per_round"] < 0.01:
            continue
        pct = 100.0 * r["self_gpu_ms_per_round"] / sms if sms else 0.0
        L.append(f"    {r['region']:<34} {r['calls_per_round']:8.1f} "
                 f"{r['self_dispatch_per_round']:10.1f} "
                 f"{r['self_gpu_ms_per_round']:9.3f} "
                 f"{r['self_gpu_ms_per_round'] * scale:9.3f} {pct:6.1f}")
    return "\n".join(L)


def main_real(args) -> int:
    import mlx.core as mx

    from decode_gpu_trace import Probe

    mx.eval(mx.ones((8, 8)) @ mx.ones((8, 8)))
    probe = Probe()
    if probe.install():
        print(f"metal_probe: 差し替え {probe.install_hits}")
    else:
        print(f"metal_probe: 使えない ({probe.error})")
        return 1

    from verify_width_cost import build_runner, build_prompt_ids
    from decode_ab import prefill_once

    t0 = time.perf_counter()
    eng, model, tok, eos_ids = build_runner(args)
    print(f"モデル読み込み {time.perf_counter() - t0:.1f}s  "
          f"device={mx.device_info()['device_name']}")

    census = weight_census(model)
    print("\n=== 重みのバイト数 (1 トークンぶんの読み出し、MB) ===")
    for k, v in sorted(census.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<28} {v / 1e6:9.1f} MB")

    reg = Regions(probe)
    install_hooks(reg, eng)

    default_depth = eng.depth
    default_limit = eng.depth_ctx_limit
    default_adapt = eng._depth_adapt

    print("\nburn-in (捨てる 1 本)...", flush=True)
    ids0 = build_prompt_ids(tok, 0)
    c0, s0, r0, _ = prefill_once(eng, ids0, eos_ids)
    run_rounds(eng, c0, s0, r0, ids0.shape[1], 4, 12, None, None)
    del c0, s0, r0
    mx.clear_cache()

    names = [c.strip() for c in args.cases.split(",") if c.strip()]
    prefilled: dict[int, tuple] = {}
    out_all = {}
    for nm in names:
        case = CASES[nm]
        print(f"\n=== case {nm} (ctx={case['ctx']}, depth={case['depth']}) ===",
              flush=True)
        eng.depth = case["depth"]
        eng.depth_ctx_limit = 1 << 30
        eng._depth_adapt = False

        if case["ctx"] not in prefilled:
            # 17k の KV を短文脈のものと同時に抱えない (98GB の常駐に足すと
            # メモリ待ちに入る)。使わない文脈のキャッシュは先に落とす。
            for k in list(prefilled):
                del prefilled[k]
            mx.clear_cache()
            ids = build_prompt_ids(tok, case["ctx"])
            t0 = time.perf_counter()
            prefilled[case["ctx"]] = (ids,) + prefill_once(eng, ids, eos_ids)
            print(f"  prefill n={ids.shape[1]} ({time.perf_counter() - t0:.1f}s)",
                  flush=True)
        ids, caches, snap, resume, _first = prefilled[case["ctx"]]
        n = ids.shape[1]

        plain = run_rounds(eng, caches, snap, resume, n, args.warmup, args.rounds,
                           probe, None)
        if args.split_cb:
            rec = {"case": nm, "ctx": n, "depth": case["depth"],
                   "split_cb": True, "plain": plain,
                   "note": ("MLX_MAX_OPS_PER_BUFFER=1。カーネル 1 回あたりの GPU 時間が"
                            " 厳密になる代わりに 1 回あたり CB の固定費が乗り、壁時計は"
                            " 2.5 倍。単価を取る用。")}
            out_all[nm] = rec
            pth = Path(args.out_dir) / f"decode-anatomy-splitcb-{nm}.json"
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            print(f"  書き出し: {pth}", flush=True)
            continue
        levels = {}
        for lv in (1, 2, 3):
            reg.level = lv
            reg.acc.clear(); reg.kern.clear(); reg.kern_ms.clear(); reg.rounds = 0
            hk = run_rounds(eng, caches, snap, resume, n, 2, args.hook_rounds,
                            probe, reg)
            rows_lv = reg.table()
            levels[f"L{lv}"] = {"hooked": hk, "regions": rows_lv}
            print(fmt(f"{nm} L{lv}", plain, hk, rows_lv), flush=True)
        hooked = levels["L3"]["hooked"]
        rows = levels["L3"]["regions"]

        rec = {
            "case": nm, "ctx": n, "ctx_arg": case["ctx"], "depth": case["depth"],
            "model": args.model, "device": mx.device_info().get("device_name"),
            "plain": plain, "hooked": hooked, "regions": rows, "levels": levels,
            "weight_bytes": census,
            "note": ("hooked は region 境界で mx.eval を強制した走行。dispatch の帰属は"
                     " 厳密、GPU 時間は region ごとに CB の立ち上がりを 1 回ずつ"
                     " 余分に払うので合計が素より大きい。比率で読むこと。"),
        }
        out_all[nm] = rec
        p = Path(args.out_dir) / f"decode-anatomy-{nm}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        print(f"  書き出し: {p}", flush=True)

    eng.depth = default_depth
    eng.depth_ctx_limit = default_limit
    eng._depth_adapt = default_adapt

    comb = Path(args.out_dir) / ("decode-anatomy-splitcb.json" if args.split_cb
                                 else "decode-anatomy.json")
    comb.write_text(json.dumps({"cases": out_all, "weight_bytes": census},
                               ensure_ascii=False, indent=1))
    print(f"\nまとめ: {comb}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main_synthetic(args) -> int:
    """合成の小さい Flash-Next を CPU で流し、包みと勘定だけ確かめる。"""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    import mlxturbo  # noqa: F401
    from mlxturbo.mtp_flash import FlashMTPModule
    from mlxturbo.spec_flash import FlashSpecEngine
    from verify_batch_spec import build

    from decode_ab import prefill_once

    from decode_gpu_trace import Probe

    probe = Probe()
    probe.install()
    model = build()
    mtp = FlashMTPModule(model.args.text, variant="lane")
    mx.eval(mtp.parameters())
    eng = FlashSpecEngine(model, mtp)
    print("重みの棚卸し:", {k: round(v / 1e6, 3) for k, v in weight_census(model).items()})

    reg = Regions(probe if probe.available else None)
    install_hooks(reg, eng)

    vocab = model.args.text.vocab_size
    ids = mx.array([[(i * 7 + 3) % vocab for i in range(24)]], dtype=mx.int32)
    caches, snap, resume, _ = prefill_once(eng, ids, ())
    for depth in (0, 2):
        eng.depth = depth
        eng.depth_ctx_limit = 1 << 30
        eng._depth_adapt = False
        plain = run_rounds(eng, caches, snap, resume, ids.shape[1], 1, 3, None, None)
        for lv in (1, 2):
            reg.level = lv
            reg.acc.clear(); reg.kern.clear(); reg.kern_ms.clear(); reg.rounds = 0
            hk = run_rounds(eng, caches, snap, resume, ids.shape[1], 1, 3, None, reg)
            got_lv = {r["region"] for r in reg.table()}
            assert "_staged_forward" in got_lv, lv
            if lv == 1:
                assert not any(r.startswith(("V/", "D/", "P/")) for r in got_lv), \
                    f"level 1 で層の region が立っている: {sorted(got_lv)}"
            else:
                assert "V/MoE(SparseMoeBlock)" in got_lv, sorted(got_lv)
                assert "V/HC(GatedResidual)" not in got_lv, "level 2 に HC が居る"
        reg.level = 3
        reg.acc.clear(); reg.kern.clear(); reg.kern_ms.clear(); reg.rounds = 0
        hooked = run_rounds(eng, caches, snap, resume, ids.shape[1], 1, 3, None, reg)
        rows = reg.table()
        got = {r["region"] for r in rows}
        print(f"\ndepth={depth} rounds={reg.rounds} region {len(got)} 種")
        for r in rows:
            print(f"    {r['region']:<34} 呼 {r['calls_per_round']:6.1f}")
        assert plain["rounds"] == 3 and hooked["rounds"] == 3
        need = {"_staged_forward", "V/layer.pre_mlp", "V/MoE(SparseMoeBlock)",
                "V/HC(GatedResidual)", "V/layer._combine(HC write)"}
        missing = need - got
        assert not missing, f"包めていない region: {missing}"
        if depth > 0:
            assert "_draft_chain" in got, "draft の region が立っていない"
            assert any(r.startswith("D/") for r in got), \
                f"draft 位相の接頭辞が付いていない: {sorted(got)}"
        # 出力トークン列が包みで変わらないこと
        assert plain["tokens_per_round"] == hooked["tokens_per_round"], (
            plain["tokens_per_round"], hooked["tokens_per_round"])
    print("\n合成の検算は通った (包み・勘定・トークン列の不変)。")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm-head4")
    ap.add_argument("--ngram", default="~/models/ddalcu-ngram")
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--cases", default="s1-short,s3-short,s1-17k,s3-17k")
    ap.add_argument("--rounds", type=int, default=48, help="素の走行のラウンド数")
    ap.add_argument("--hook-rounds", type=int, default=6,
                    help="包んだ走行のラウンド数 (境界ごとに同期するので短くてよい)")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--split-cb", action="store_true",
                    help="MLX_MAX_OPS_PER_BUFFER=1 で 1 command buffer = 1 op に割り、"
                         "**カーネル 1 回あたりの本当の GPU 時間**を採る。包んだ走行は"
                         "しない (歪みが二重になる)。壁時計が 2.5 倍になるので絶対値の"
                         "比較には使わず、カーネルの単価を得るためだけに使う。"
                         "出力は decode-anatomy-splitcb-<case>.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    for nm in [c.strip() for c in args.cases.split(",") if c.strip()]:
        if nm not in CASES:
            raise SystemExit(f"未知の case: {nm} (使えるのは {','.join(CASES)})")
    if args.split_cb:
        # MLX はデバイス初期化時にこの env を読む。mlx を import する前に立てる。
        os.environ["MLX_MAX_OPS_PER_BUFFER"] = "1"
    if args.synthetic:
        return main_synthetic(args)
    return main_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
