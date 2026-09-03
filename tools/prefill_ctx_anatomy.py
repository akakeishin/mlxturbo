"""prefill 1 本 (文脈まるごと) の壁時計を、部品と段に割る。

`tools/prefill_anatomy.py` は「1 チャンク (幅 2048) の内訳」を出す道具で、
チャンク 3 点の標本しか取らない。こちらは **文脈 1 本ぶんの prefill
(4k / 8k / 17k) の壁時計そのもの**を割る。本番のチャンク割り
(`spec_flash.generate_stream` のグループ / 端数 / 末尾 v2) をそのまま辿り、

  1. **段 (stage)**: 本物の `generate_stream(ids, 0)` を
     `MLXTURBO_PREFILL_TRACE=1` で流し、group build / ngram lookahead /
     group eval / clear_cache / checkpoint / tail split / tail forward /
     prime の ms を実測する。ここが**壁時計の側**。
  2. **部品 (part)**: 同じチャンク割りを自分で辿り直し、各ステップで
     MoE / GDN / Attention (indexer 込み) / HC (pre/post) / PLE / embed を
     `prefill_anatomy` と同じ退避・復元で個別に測る。ここが**部品の側**。

**部品和 ≈ forward の壁時計 (group build + group eval + tail forward)** を
必ず突き合わせる (CLAUDE.md の作法)。無効化の積み上げは使わない。

## 既知の帰属の歪み (読むときの注意)

- **n-gram の行読み**は本番だと `_prefetch_ngram_span` が 1 グループ先を
  背景で読んでいる。部品の側は同じチャンクを繰り返し測るので 2 レップ目
  以降は必ず温かい = PLE が本番より軽く出る。冷の値は段の側 (trace) と
  `StreamNGram.stats` の `sync_ms` / `fetch_ms` で別に出す。
- MoE はグループの行を concat して 1 回で呼ぶ (G=4 なら M=8192)。部品の
  側もグループ経路 (`_group_prefill_forward`) で捕まえるので本番と同じ幅。

    tools/biglock.sh .venv/bin/python tools/prefill_ctx_anatomy.py \\
        --model ~/models/ddalcu-mlxlm-head4 --ngram ~/models/ddalcu-ngram-sep \\
        --ctx 8000 --out bench/results/prefill-anatomy-0903-8k.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# spec_flash が import 時に読む (_PREFILL_TRACE)。モデルを読む前に立てる。
os.environ["MLXTURBO_PREFILL_TRACE"] = "1"
os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

import prefill_anatomy as PA  # noqa: E402

# trace の 1 行: "[prefill] group build i=0 g=4 t=12.34 dur=5.67"
_LINE = re.compile(r"^\[prefill\] (?P<label>.+?) t=(?P<t>[\d.]+) dur=(?P<dur>[\d.]+)$")
_TOTAL = re.compile(
    r"^\[prefill\] total dur_sum=(?P<sum>[\d.]+) wall=(?P<wall>[\d.]+) "
    r"gap=(?P<gap>[-\d.]+)$")

# 段の名寄せ (label の先頭で判定する)。
_STAGE_KEYS = [
    ("group build", "group_build"),
    ("group eval", "group_eval"),
    ("ngram lookahead", "ngram_lookahead"),
    ("clear_cache", "clear_cache"),
    ("checkpoint", "checkpoint"),
    ("tail split", "tail_split"),
    ("tail forward", "tail_forward"),
    ("prime chunk", "prime"),
]


def stage_of(label: str) -> str:
    for pre, key in _STAGE_KEYS:
        if label.startswith(pre):
            return key
    return "other"


def parse_trace(text: str) -> dict:
    """1 回の prefill の trace 出力を段ごとの ms に畳む。"""
    stages: dict[str, float] = {}
    events = []
    total = None
    for line in text.splitlines():
        m = _TOTAL.match(line)
        if m:
            total = {"dur_sum_ms": float(m["sum"]), "wall_ms": float(m["wall"]),
                     "gap_ms": float(m["gap"])}
            continue
        m = _LINE.match(line)
        if not m:
            continue
        label, dur = m["label"], float(m["dur"])
        stages[stage_of(label)] = stages.get(stage_of(label), 0.0) + dur
        events.append((label, dur))
    return {"stages": stages, "events": events, "total": total}


# --------------------------------------------------------------- チャンク割り


def build_schedule(n: int, step: int, G: int, tail_chunks: int,
                   fold_tail: bool, tail_in_group: bool) -> list:
    """`spec_flash.generate_stream` の prefill ループと同じ割り方を再現する。

    戻り値は先頭からのステップ列。``kind`` は ``group`` (layer-major、
    ``widths`` がメンバー幅) か ``chunk`` (chunk 主導)。**この写しは trace の
    ラベル (group build i=.. g=..) と突き合わせて検算する** (`check_schedule`)。
    """
    steps = []
    i = 0
    while i < n:
        remaining = n - i
        g = min(G, (remaining - tail_chunks * step) // step)
        group = None
        frac_len = 0
        group_tail = False
        g_min = 1 if tail_in_group else 2
        if G > 1 and g >= g_min:
            group = [step] * g
            if fold_tail and g < G:
                after = remaining - g * step
                if step < after < 2 * step:
                    frac_len = after - step
                    group.append(frac_len)
        elif fold_tail and G > 1 and remaining > step and remaining - step < step:
            frac_len = remaining - step
            group = [frac_len]
        if tail_in_group and group is not None:
            after = remaining - sum(group)
            if 2 <= after <= step:
                group.append(after - 1)
                group_tail = True
            elif frac_len == 0 and len(group) == 1:
                group = None
        if group is not None:
            steps.append({"kind": "group", "start": i, "widths": group,
                          "tail_in_group": group_tail})
            i += sum(group)
            continue
        if remaining > step:
            j = i + min(step, remaining - step)
        else:
            j = n
        steps.append({"kind": "chunk", "start": i, "width": j - i,
                      "last": j == n})
        i = j
    return steps


def check_schedule(steps: list, events: list) -> list:
    """trace のラベルから読める (i, g) と、写しの割り方を突き合わせる。"""
    seen = [(int(m["i"]), int(m["g"]))
            for lb, _ in events
            if (m := re.match(r"group build i=(?P<i>\d+) g=(?P<g>\d+)", lb))]
    mine = [(s["start"], len(s["widths"])) for s in steps if s["kind"] == "group"]
    return [f"trace={seen}", f"replica={mine}",
            "一致" if seen == mine else "**不一致: 写しを直すこと**"]


# --------------------------------------------------------------- 部品の計測

PART_ORDER = ["moe", "gdn", "attn", "hc_pre", "hc_post", "ple", "embed"]
PART_LABEL = {
    "moe": "MoE 48 層",
    "gdn": "GDN 36 層",
    "attn": "Attention 12 層 (indexer 込み)",
    "indexer": "  うち QSA indexer",
    "attn_qkv": "  うち q/k/v 射影+norm+rope+KV",
    "attn_core": "  うち sdpa/gather + o_proj",
    "hc_pre": "HC 読み (GatedResidual)",
    "hc_post": "HC 書き (_combine)",
    "ple": "PLE 1 層 (n-gram 込み)",
    "embed": "embed + tile",
}


# 捕まえる部品を何回に分けて捕まえるか。**1 回のフォワードで全部の引数を
# 掴むと落ちる** (実測: 8k のグループで Metal の OOM)。掴んだ配列は解放
# されないので、MoE の x (7866 行で 40MB x 48 層) や HC の hyper (63MB x 96 回)
# が同時に生きると 7GB を超え、71GB のモデル + 30GB の n-gram を載せた
# この機体では収まらない。部品ごとにフォワードを分けて、その組の引数だけを
# 掴む (フォワードは決定的なので、分けても捕まる引数は同じもの)。
CAPTURE_PASSES = [
    ("moe",),
    ("gdn", "ple"),
    ("attn",),
    ("attn_qkv", "idx", "sel"),
    ("hc_pre", "hc_post"),
]
# それでも大きい HC は間引いて捕まえ、測った値を間引いた比で戻す
# (同じ形の呼び出しなので費用は呼び出しごとに同じ)。**間引き幅は呼び出しの
# 周期と互いに素にすること。**HC 読みは 1 層あたり (2 種類 x メンバー数) 回
# 呼ばれるので、周期 8 (G=4) や 10 (G=5) を割り切る幅で間引くと、
# 毎回同じ種類・同じメンバーだけを拾って偏る。7 は 8 とも 10 とも 4 とも
# 2 とも互いに素。
# MoE の x (グループ幅で 40MB x 48 層) と GDN の x も、この機体では
# 掴みっぱなしにできない (モデル 71GB + n-gram の RAM 常駐 32GB で余地が
# 25GB しかなく、8k / 17k は attention の捕獲パスで Metal の OOM に倒れた)。
# ここも間引く。周期はメンバー数 (4 か 5) なので 7 で逃げる。
CAPTURE_STRIDE = {"hc_pre": 7, "hc_post": 7, "moe": 4, "gdn": 7}
# 再生 (計測) を 1 つの eval にまとめる本数。MoE は 1 層の中間が
# グループ幅 (M=78660 行) で 0.7GB あり、48 層を 1 eval にすると落ちる。
# 束を刻むと束の境目に同期が入るが、束は数個なので ms 単位では効かない。
REPLAY_BATCH = {"moe": 6, "gdn": 24, "attn": 12, "attn_qkv": 24,
                "idx": 24, "sel": 24, "hc_pre": 24, "hc_post": 24, "ple": 4}


def measure_step(model, Q, mx, caches, chunks, is_group, reps, SF,
                 on_moe=None, on_gdn=None):
    """1 ステップ (group か chunk) の部品を測る。

    捕まえ方も測り方も `prefill_anatomy.measure` と同じ (実フォワードで
    引数を捕まえ、キャッシュを退避・復元しながら部品だけを繰り返す)。
    違うのは**本番のグループ経路で捕まえる**ことと、上の理由で捕獲を
    数回のフォワードに分けること。``on_moe`` / ``on_gdn`` はその組の
    引数が生きている間に呼ばれるコールバック (中の内訳用)。
    """
    TARGET = {
        "moe": (Q.SparseMoeBlock, "__call__"),
        "gdn": (Q.GatedDeltaNet, "__call__"),
        "hc_pre": (Q.GatedResidual, "__call__"),
        "ple": (Q.PLELayer, "__call__"),
        "attn": (Q.Attention, "__call__"),
        "attn_qkv": (Q.Attention, "_qkv"),
        "idx": (Q.QSAIndexer, "__call__"),
        "sel": (Q.QSAIndexer, "select_blocks"),
    }
    orig = {k: getattr(cls, name) for k, (cls, name) in TARGET.items()}
    orig_combine = Q.DecoderLayer._combine

    def forward():
        if is_group:
            hs = SF._group_prefill_forward(model, chunks, caches)
            return list(hs)
        return [model.model(chunks[0], cache=caches)]

    def as_array(r):
        if r is None:
            return mx.zeros(1)
        if isinstance(r, tuple):
            return r[0]
        if isinstance(r, Q.QSABlockSelection):
            return r.keep_block
        return r

    res, counts = {}, {}
    for keys in CAPTURE_PASSES:
        grabbed = {k: [] for k in keys}
        seen = {k: 0 for k in keys}

        def keep(key, seen=seen) -> bool:
            i = seen[key]
            seen[key] = i + 1
            return i % CAPTURE_STRIDE.get(key, 1) == 0

        def wrap(key, fn, grabbed=grabbed):
            def g(self, *a):
                if keep(key):
                    grabbed[key].append((self, *a))
                return fn(self, *a)
            return g

        def wrap_combine(fn, grabbed=grabbed):
            def g(hyper, x, inject):
                if keep("hc_post"):
                    grabbed["hc_post"].append((hyper, x, inject))
                return fn(hyper, x, inject)
            return g

        import gc
        gc.collect()
        mx.clear_cache()
        print(f"    [mem] pass={'+'.join(keys)} active="
              f"{mx.get_active_memory() / 2**30:.1f}GiB cache="
              f"{mx.get_cache_memory() / 2**30:.1f}GiB peak="
              f"{mx.get_peak_memory() / 2**30:.1f}GiB", flush=True)
        mx.reset_peak_memory()
        installed = [k for k in keys if k in TARGET]
        for k in installed:
            cls, name = TARGET[k]
            setattr(cls, name, wrap(k, orig[k]))
        if "hc_post" in keys:
            Q.DecoderLayer._combine = staticmethod(wrap_combine(orig_combine))
        pre = PA.snapshot(caches)
        try:
            mx.eval(forward() + PA.pending(caches))
        finally:
            for k in installed:
                cls, name = TARGET[k]
                setattr(cls, name, orig[k])
            if "hc_post" in keys:
                Q.DecoderLayer._combine = staticmethod(orig_combine)
        PA.restore(caches, pre)
        mx.clear_cache()

        def bench(key, fn=None, grabbed=grabbed, seen=seen):
            got = grabbed[key]
            if not got:
                return 0.0
            f = fn if fn is not None else orig[key]
            bs = REPLAY_BATCH.get(key, len(got))

            def run():
                st = PA.snapshot(caches)
                tails = []
                for lo in range(0, len(got), bs):
                    part = [as_array(f(*t)) for t in got[lo:lo + bs]]
                    mx.eval(part)
                    tails.append(part[-1])
                mx.eval(PA.pending(caches))
                PA.restore(caches, st)
                return tails
            ms = PA.med_ms(run, reps)
            mx.clear_cache()
            return ms * (seen[key] / len(got))

        for k in keys:
            res[k] = bench(k, fn=orig_combine if k == "hc_post" else None)
            counts[k] = (seen[k], len(grabbed[k]))
        if "moe" in keys and on_moe is not None:
            on_moe(grabbed["moe"])
        if "gdn" in keys and on_gdn is not None:
            on_gdn(grabbed["gdn"])
        grabbed.clear()
        del grabbed
        mx.clear_cache()

    res["indexer"] = res.pop("idx") + res.pop("sel")
    res["attn_core"] = res["attn"] - res["attn_qkv"] - res["indexer"]
    emb = model.model.embed_tokens
    res["embed"] = PA.med_ms(
        lambda: [mx.tile(emb(c), (1, 1, model.model.hc)) for c in chunks], reps)
    res["_counts"] = counts

    def whole():
        st = PA.snapshot(caches)
        out = forward()
        mx.eval(out + PA.pending(caches))
        PA.restore(caches, st)
        return out

    res["_wall"] = PA.med_ms(whole, reps)
    mx.clear_cache()
    return res


# ----------------------------------------------------- MoE の本番経路の内訳

MOE_ORDER = ["router", "topk", "sort", "tables", "gemm_up", "gemm_gate",
             "swiglu", "gemm_down", "combine", "shared"]
MOE_LABEL = {
    "router": "  router (fp32 qmm 2560->512)",
    "topk": "  top-k 選択 + softmax",
    "sort": "  並べ替え (argsort + 添字)",
    "tables": "  segmented GEMM の表",
    "gemm_up": "  up GEMM (gather 畳み込み)",
    "gemm_gate": "  gate GEMM (gather 畳み込み)",
    "swiglu": "  SwiGLU",
    "gemm_down": "  down GEMM",
    "combine": "  combine カーネル (unsort+重み+和)",
    "shared": "  共有専門家 (ゲート込み)",
}


def moe_parts_prod(mod, x, reps, mx, Q, F):
    """P3 混合タイル + P7 第 2 段 (既定) の MoE を、呼び出し順そのまま割る。

    `_vendor/qwen4_exp.py` の `SparseMoeBlock.__call__` -> `_moe_combine_fold`
    -> `fused._moe_fold_block` の並びを実引数でなぞる。素の SwitchGLU 経路
    (行数 < 64) は対象外。
    """
    from mlxturbo.kernels import moe_grouped_gemm as mgg
    from mlxturbo.kernels import moe_combine as mc

    B, S, _ = x.shape
    rows = B * S
    top_k = mod.top_k
    assert getattr(mod, "_combine_fold_min_s", None) is not None
    assert rows >= mod._combine_fold_min_s, f"行数 {rows} が fold 閾値未満"
    assert getattr(mod, "_router513", None) is None
    assert getattr(mod, "_wide_shared", None) is None
    assert F._MOE_DOWN_EPI_ON and F._MOE_GEMM_MODE == "seg", "P3/P7 が既定でない"

    sw, gate, se, seg = (mod.switch_mlp, mod.gate, mod.shared_expert,
                         mod.shared_expert_gate)
    up, gp, dn = sw.up_proj, sw.gate_proj, sw.down_proj
    (w_up, s_up, b_up) = up["weight"], up["scales"], up["biases"]
    (w_gate, s_gate, b_gate) = gp["weight"], gp["scales"], gp["biases"]
    (w_dn, s_dn, b_dn) = dn["weight"], dn["scales"], dn["biases"]

    # -- 前提を非計測で 1 回だけ確定させる (moe_split / gdn_split と同じ流儀)
    logits = gate(x.astype(mx.float32))
    mx.eval(logits)
    idx = mx.argpartition(-logits, top_k - 1, axis=-1)[..., :top_k]
    w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
    mx.eval(idx, w)
    idx_flat = idx.flatten()
    order = mx.argsort(idx_flat)
    idx_s = idx_flat[order]
    row_src = order // top_k
    w_flat = w.flatten()
    mx.eval(order, idx_s, row_src, w_flat)
    x2 = x.reshape(rows, x.shape[-1])
    tables = F._moe_gemm_tables(idx_s, w_up.shape[0])
    mx.eval([t for t in tables if isinstance(t, mx.array)])
    kw = dict(tables=tables, bm=F._MOE_GEMM_BM, wm=F._MOE_GEMM_WM,
              mix_threshold=F._MOE_GEMM_MIX, group_size=up.group_size,
              bits=up.bits)
    src = row_src if F._MOE_GATHER_FOLD_ON else None
    x_up = mgg.qmm_segmented(x2, w_up, s_up, b_up, None, row_src=src, **kw)
    x_gate = mgg.qmm_segmented(x2, w_gate, s_gate, b_gate, None, row_src=src, **kw)
    mx.eval(x_up, x_gate)
    act = sw.activation(x_up, x_gate)
    mx.eval(act)
    kw_dn = dict(kw, group_size=dn.group_size, bits=dn.bits)
    out_sorted = mgg.qmm_segmented(act, w_dn, s_dn, b_dn, None, **kw_dn)
    mx.eval(out_sorted)
    inv = F._inv_perm(order)
    mx.eval(inv)

    parts = {
        "router": lambda: gate(x.astype(mx.float32)),
        "topk": lambda: (
            i2 := mx.argpartition(-logits, top_k - 1, axis=-1)[..., :top_k],
            mx.softmax(mx.take_along_axis(logits, i2, axis=-1), axis=-1,
                       precise=True))[1],
        "sort": lambda: (o := mx.argsort(idx_flat), idx_flat[o], o // top_k)[1],
        # `F._moe_gemm_tables` は同じ添字なら表を使い回すので、繰り返し測ると
        # 2 回目以降が 0 になる。本番の初回と同じ仕事を測るため、キャッシュを
        # 通さず素の 2 op (counts + segment_tables) を直接呼ぶ
        "tables": lambda: mgg.segment_tables(
            mgg.counts_from_sorted_ids(idx_s, w_up.shape[0]),
            bm=F._MOE_GEMM_BM, mix_threshold=F._MOE_GEMM_MIX)[1],
        "gemm_up": lambda: mgg.qmm_segmented(x2, w_up, s_up, b_up, None,
                                             row_src=src, **kw),
        "gemm_gate": lambda: mgg.qmm_segmented(x2, w_gate, s_gate, b_gate, None,
                                               row_src=src, **kw),
        "swiglu": lambda: sw.activation(x_up, x_gate),
        "gemm_down": lambda: mgg.qmm_segmented(act, w_dn, s_dn, b_dn, None,
                                               **kw_dn),
        "combine": lambda: mc.combine(out_sorted, inv, w_flat, rows, top_k),
        "shared": lambda: mx.sigmoid(seg(x)) * se(x),
    }
    ms = {}
    for k in MOE_ORDER:
        ms[k] = PA.med_ms(parts[k], reps)
        mx.clear_cache()
    layer_ms = PA.med_ms(lambda: mod(x), reps)
    mx.clear_cache()
    ssum = sum(ms.values())
    return {"rows": rows, "ms": ms, "parts_sum_ms": ssum,
            "layer_wallclock_ms": layer_ms,
            "gap_pct": (ssum - layer_ms) / layer_ms * 100 if layer_ms else 0.0}


# ------------------------------------------------------------------- 本体


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--ctx", type=int, default=8000)
    ap.add_argument("--reps", type=int, default=3, help="部品の中央値レップ数")
    ap.add_argument("--stage-reps", type=int, default=3,
                     help="段 (本物の prefill) の計測本数 (捨てる分は別)")
    ap.add_argument("--stage-warmup", type=int, default=2,
                     help="立ち上がりを外すために捨てる prefill の本数")
    ap.add_argument("--sub", action="store_true", default=True,
                     help="MoE / GDN の中の内訳も出す")
    ap.add_argument("--no-sub", dest="sub", action="store_false")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.gated_delta import compute_g

    import mlxturbo  # noqa: F401
    import mlx_lm.models.qwen4_exp as Q
    import mlxturbo.spec_flash as SF
    import mlxturbo.fused as F
    from mlxturbo import mtp_flash
    from mlxturbo.runner import enable_default_fusions, set_wired_limit_default
    from mlxturbo.kernels.gdn_blocked_metal import gated_delta_blocked_seq
    import gdn_split as GS

    assert SF._PREFILL_TRACE, "MLXTURBO_PREFILL_TRACE が効いていない"

    mpath = os.path.expanduser(args.model)
    model, tok = load(mpath)
    if args.ngram:
        from mlxturbo.ngram_stream import install
        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[prefill-ctx]")
    set_wired_limit_default(log_prefix="[prefill-ctx]")
    mtp = mtp_flash.load_flash_mtp(os.path.join(mpath, "mtp.safetensors"),
                                   model.args.text)
    mx.eval(mtp.parameters())
    eng = SF.FlashSpecEngine(model, mtp)

    from _bench_text import long_prompts
    body = long_prompts(tok, args.ctx,
                        ["上の文書の要点を 5 つに整理してください。"])[0]
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True))[None]
    n = ids.shape[1]
    step = SF.PREFILL_STEP_SIZE
    print(f"ctx={n} step={step} G={SF._PREFILL_GROUP} "
          f"tail_in_group={SF._PREFILL_TAIL_IN_GROUP} "
          f"fold_tail={SF._PREFILL_FOLD_TAIL} reps={args.reps}", flush=True)

    # ---------------------------------------------------------------- 段
    ng = getattr(model, "_ngram_stream", None)
    if ng is None:
        for layer in model.model.layers:
            ple = getattr(layer, "ple", None)
            emb = getattr(getattr(ple, "ple_embedding", None),
                          "ngram_embedding", None)
            if emb is not None and hasattr(emb, "stats"):
                ng = emb
                break

    runs = []
    for r in range(args.stage_warmup + args.stage_reps):
        caches = model.make_cache()
        mx.clear_cache()
        if ng is not None:
            ng.reset_stats()
        buf = io.StringIO()
        t0 = time.perf_counter()
        with redirect_stdout(buf):
            gen = eng.generate_stream(ids, 0, caches=caches, checkpoints=[])
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass
        wall = (time.perf_counter() - t0) * 1e3
        rec = parse_trace(buf.getvalue())
        rec["outer_wall_ms"] = wall
        if ng is not None:
            rec["ngram_stats"] = dict(ng.stats)
        runs.append(rec)
        del caches
        mx.clear_cache()
        tag = "捨" if r < args.stage_warmup else "採"
        print(f"  [{tag}] prefill {r}: wall={wall:.0f} ms "
              f"stages={ {k: round(v) for k, v in rec['stages'].items()} }",
              flush=True)

    kept = runs[args.stage_warmup:]
    stage_keys = sorted({k for r in kept for k in r["stages"]})
    stages_med = {k: statistics.median([r["stages"].get(k, 0.0) for r in kept])
                  for k in stage_keys}
    wall_med = statistics.median([r["outer_wall_ms"] for r in kept])
    trace_wall_med = statistics.median(
        [r["total"]["wall_ms"] for r in kept if r["total"]])
    stage_sum = sum(stages_med.values())

    print(f"\n段 (中央値 {len(kept)} 本): 壁時計 {wall_med:.0f} ms "
          f"(trace 内 {trace_wall_med:.0f})", flush=True)
    for k in sorted(stages_med, key=lambda z: -stages_med[z]):
        print(f"  {k:22s}{stages_med[k]:9.1f} ms{stages_med[k] / wall_med * 100:7.1f}%")
    print(f"  {'段の和':22s}{stage_sum:9.1f} ms"
          f"{stage_sum / wall_med * 100:7.1f}%")
    print(f"  {'計測外 (隙間)':22s}{wall_med - stage_sum:9.1f} ms"
          f"{(wall_med - stage_sum) / wall_med * 100:7.1f}%", flush=True)

    # --------------------------------------------------------------- 部品
    steps = build_schedule(n, step, SF._PREFILL_GROUP, SF._PREFILL_TAIL_CHUNKS,
                           SF._PREFILL_FOLD_TAIL, SF._PREFILL_TAIL_IN_GROUP)
    chk = check_schedule(steps, kept[-1]["events"])
    print("\nチャンク割りの検算: " + " / ".join(chk), flush=True)

    caches = model.make_cache()
    mx.clear_cache()
    per_step = []


    for si, st in enumerate(steps):
        i = st["start"]
        if st["kind"] == "group":
            offs, acc = [], i
            chunks = []
            for wdt in st["widths"]:
                chunks.append(ids[:, acc:acc + wdt])
                acc += wdt
            is_group = True
            desc = f"group start={i} members={st['widths']}"
        else:
            chunks = [ids[:, i:i + st["width"]]]
            is_group = False
            desc = f"chunk start={i} width={st['width']}"
        print(f"\n[step {si}] {desc}", flush=True)

        rec = {"index": si, "desc": desc, "kind": st["kind"], "start": i,
               "widths": st.get("widths", [st.get("width")])}

        def on_moe(got, rec=rec, si=si, is_group=is_group):
            """MoE の引数が生きている間に、本番経路の内訳を取る。"""
            if not (args.sub and is_group and got):
                return
            target = model.model.layers[20].mlp
            hit = [t for t in got if t[0] is target] or got[len(got) // 2:]
            try:
                mod, xm = hit[0]
                rec["moe_sub"] = moe_parts_prod(mod, xm, args.reps, mx, Q, F)
                d = rec["moe_sub"]
                print("  MoE 1 層 (層 20) の内訳: " + " ".join(
                    f"{k}={d['ms'][k]:.2f}" for k in MOE_ORDER)
                    + f" | 和 {d['parts_sum_ms']:.2f} 対 層壁"
                      f" {d['layer_wallclock_ms']:.2f} ({d['gap_pct']:+.1f}%)",
                    flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  MoE 内訳を取れなかった: {e!r}", flush=True)

        def on_gdn(got, rec=rec, chunks=chunks, is_group=is_group):
            if not (args.sub and is_group and got):
                return
            # GDN の呼び出しはレイヤー主導 (層ごとに G メンバー) の順で
            # 並ぶ。18 本目の GDN 層の**先頭メンバー**を代表に取る
            wide = max(t[1].shape[1] for t in got)
            hit = [t for t in got if t[1].shape[1] == wide]
            try:
                gdn, xg, maskg, cg = hit[len(hit) // 2]
                rec["gdn_sub"] = GS.measure_parts(
                    gdn, cg, xg, maskg, args.reps, mx, nn, compute_g,
                    gated_delta_blocked_seq)
                d = rec["gdn_sub"]
                print("  GDN 1 層 (18 本目) の内訳: " + " ".join(
                    f"{k}={d['ms'][k]:.2f}" for k in GS.PART_ORDER)
                    + f" | 和 {d['parts_sum_ms']:.2f} 対 層壁"
                      f" {d['layer_wallclock_ms']:.2f} ({d['gap_pct']:+.1f}%)",
                    flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  GDN 内訳を取れなかった: {e!r}", flush=True)

        res = measure_step(model, Q, mx, caches, chunks, is_group,
                           args.reps, SF, on_moe=on_moe, on_gdn=on_gdn)
        parts_sum = sum(res[k] for k in PART_ORDER)
        gap = (parts_sum - res["_wall"]) / res["_wall"] * 100
        print("  捕まえた: " + " ".join(f"{k}={v}" for k, v in
                                        res["_counts"].items() if v))
        for k in PART_ORDER:
            print(f"  {PART_LABEL[k]:36s}{res[k]:9.1f} ms")
            if k == "attn":
                for sk in ("attn_qkv", "indexer", "attn_core"):
                    print(f"  {PART_LABEL[sk]:36s}{res[sk]:9.1f} ms")
        print(f"  {'部品和':36s}{parts_sum:9.1f} ms")
        print(f"  {'このステップの壁時計':36s}{res['_wall']:9.1f} ms"
              f"   差 {parts_sum - res['_wall']:+.1f} ({gap:+.1f}%)", flush=True)
        rec.update({"parts_ms": {k: res[k] for k in PART_ORDER},
                    "attn_split_ms": {k: res[k] for k in
                                      ("attn_qkv", "indexer", "attn_core")},
                    "counts": res["_counts"],
                    "parts_sum_ms": parts_sum, "wall_ms": res["_wall"],
                    "gap_pct": gap})
        per_step.append(rec)

        # 本番と同じ前進 (キャッシュを確定させる)
        if is_group:
            hs = SF._group_prefill_forward(model, chunks, caches)
            mx.eval(list(hs) + PA.pending(caches))
        else:
            out = model.model(chunks[0], cache=caches)
            mx.eval([out] + PA.pending(caches))
        mx.clear_cache()

    # ------------------------------------------------------------ まとめ
    tot = {k: sum(r["parts_ms"][k] for r in per_step) for k in PART_ORDER}
    tot_attn = {k: sum(r["attn_split_ms"][k] for r in per_step)
                for k in ("attn_qkv", "indexer", "attn_core")}
    parts_sum = sum(tot.values())
    fwd_wall = sum(r["wall_ms"] for r in per_step)
    fwd_stage = (stages_med.get("group_build", 0.0)
                 + stages_med.get("group_eval", 0.0)
                 + stages_med.get("tail_forward", 0.0))

    print(f"\n=== 文脈 {n} トークンの prefill ===", flush=True)
    print(f"  {'部品':36s}{'ms':>10s}{'% of 壁':>9s}")
    for k in PART_ORDER:
        print(f"  {PART_LABEL[k]:36s}{tot[k]:10.1f}{tot[k] / wall_med * 100:8.1f}%")
        if k == "attn":
            for sk in ("attn_qkv", "indexer", "attn_core"):
                print(f"  {PART_LABEL[sk]:36s}{tot_attn[sk]:10.1f}"
                      f"{tot_attn[sk] / wall_med * 100:8.1f}%")
    print(f"  {'部品和':36s}{parts_sum:10.1f}{parts_sum / wall_med * 100:8.1f}%")
    print(f"  {'forward の壁時計 (私の再現)':36s}{fwd_wall:10.1f}"
          f"{fwd_wall / wall_med * 100:8.1f}%")
    print(f"  {'forward の壁時計 (本番 trace)':36s}{fwd_stage:10.1f}"
          f"{fwd_stage / wall_med * 100:8.1f}%")
    print(f"  部品和 - forward(再現) = {parts_sum - fwd_wall:+.1f} ms "
          f"({(parts_sum - fwd_wall) / fwd_wall * 100:+.1f}%)")
    print(f"  forward(再現) - forward(本番) = {fwd_wall - fwd_stage:+.1f} ms "
          f"({(fwd_wall - fwd_stage) / fwd_stage * 100:+.1f}%)")
    for k in ("ngram_lookahead", "clear_cache", "checkpoint", "tail_split",
              "prime"):
        v = stages_med.get(k, 0.0)
        print(f"  {('段: ' + k):36s}{v:10.1f}{v / wall_med * 100:8.1f}%")
    print(f"  {'段: 計測外の隙間':36s}{wall_med - stage_sum:10.1f}"
          f"{(wall_med - stage_sum) / wall_med * 100:8.1f}%")
    print(f"  {'prefill 壁時計':36s}{wall_med:10.1f}{100.0:8.1f}%", flush=True)

    out = {
        "tool": "prefill_ctx_anatomy",
        "model": args.model, "ngram": args.ngram,
        "ctx_tokens": n, "step": step, "group": SF._PREFILL_GROUP,
        "reps": args.reps, "stage_reps": args.stage_reps,
        "stage_warmup": args.stage_warmup,
        "schedule": steps, "schedule_check": chk,
        "wall_ms_median": wall_med, "trace_wall_ms_median": trace_wall_med,
        "stages_ms_median": stages_med, "stage_sum_ms": stage_sum,
        "stage_runs": [{"outer_wall_ms": r["outer_wall_ms"],
                        "stages": r["stages"], "total": r["total"],
                        "ngram_stats": r.get("ngram_stats")} for r in runs],
        "per_step": per_step,
        "totals_ms": tot, "attn_totals_ms": tot_attn,
        "parts_sum_ms": parts_sum,
        "forward_wall_ms_replica": fwd_wall,
        "forward_wall_ms_production": fwd_stage,
    }
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\n書いた: {p}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
