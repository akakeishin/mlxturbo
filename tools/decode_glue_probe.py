"""どの族でも使える「decode 1 step の糊」の物差し (S=1、投機なし)。

`tools/decode_gpu_trace.py` は Flash-Next の `FlashSpecEngine`、
`tools/decode_round_anatomy_generic.py` は `mlxturbo.spec.SpecEngine` に
それぞれ結び付いていて、**投機エンジンを持たない族** (Gemma 4 など) には
当たらない。ここは投機を一切通さず、`mlx_lm.load` で読めるモデルなら
何でも「S=1 の decode 1 step」を測る。

出すもの (1 step あたり):

- probe (`tools/bridge/libmetal_probe.dylib`) から: dispatch 数、command
  buffer 数、GPU 時間の和/和集合、稼働率、**カーネル名ごとの回数と GPU 時間**
- 遅延グラフから: **op (プリミティブ) の census**。`--attrib` を付けると
  `tools/decode_copy_probe.py` の `Attributor` でモジュール別に分ける
  (族のモジュール定義が入っている python モジュールのクラスを自動で包む)
- 重みバイト (量子化を含む実バイト) と、それを 409.6 GB/s で読んだときの
  下限 ms。**「糊の予算 = 測った GPU 時間 - 重み読みの下限」** をここから読む

読むときの注意 (`decode_gpu_trace.py` と同じ):

- カーネル名ごとの GPU 時間は **CB の GPU 区間を dispatch 数で等分した配分値**。
  MLX は 1 CB に 15〜19 本積むので、実質「回数 x 平均」に近い。行列積のように
  1 本が重いカーネルは**過小**に、小さい elementwise は**過大**に出る。
  順位付けの目安にだけ使い、絶対値を取り分にしない。
- `MLX_MAX_OPS_PER_BUFFER=1` (`--split-cb`) を付けると配分は厳密になるが、
  CB 固定費 0.8〜1.0 us/本が乗って壁時計が数倍になる。差の読み方は
  `decode_gpu_trace.py` の冒頭を見ること。

使い方:

    BIGLOCK_NO_WORKER=1 BIGLOCK_PRIO=1 tools/biglock.sh \\
        .venv/bin/python tools/decode_glue_probe.py \\
        --model ~/models/gemma4-26b-4bit --ctx 64 --rounds 64 --attrib \\
        --out bench/results/glue-probe-gemma4.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

PEAK_BW = 409.6e9      # M3 Max のピーク帯域 (B/s)


def weight_bytes(model) -> int:
    """モデルの全パラメータの実バイト数 (量子化後の実体)。"""
    from mlx.utils import tree_flatten

    total = 0
    for _, v in tree_flatten(model.parameters()):
        try:
            total += v.nbytes
        except AttributeError:
            pass
    return total


def op_census(model, cache, token_id: int):
    """1 forward ぶんの遅延グラフから、op ごとの「作った配列の本数」を数える。"""
    import mlx.core as mx

    from decode_copy_probe import graph_outputs

    ids = mx.array([[token_id]])
    out = model(ids, cache=cache)
    counts: Counter = Counter()
    for _nid, (op, n) in graph_outputs([out]).items():
        counts[op] += n
    mx.eval(out)
    return counts


def install_attrib(model):
    """モデルの構造に出てくるクラスを自動で包む (族に依存しない)。

    `decode_copy_probe.Attributor` をそのまま使い、対象クラスは
    「モデルの木に実際に現れる `nn.Module` の型のうち、mlx_lm.models.* /
    mlxturbo._vendor.* に定義されているもの」。`nn.Linear` 系は行列積そのもの
    なので別枠にする。
    """
    import mlx.nn as nn

    from decode_copy_probe import Attributor

    attr = Attributor()
    seen: dict[type, str] = {}

    def walk(mod, depth=0):
        if depth > 8:
            return
        t = type(mod)
        m = t.__module__ or ""
        if (m.startswith("mlx_lm.models") or m.startswith("mlxturbo")) and t not in seen:
            seen[t] = t.__name__
        for child in mod.children().values():
            if isinstance(child, nn.Module):
                walk(child, depth + 1)
            elif isinstance(child, (list, tuple)):
                for c in child:
                    if isinstance(c, nn.Module):
                        walk(c, depth + 1)
            elif isinstance(child, dict):
                for c in child.values():
                    if isinstance(c, nn.Module):
                        walk(c, depth + 1)

    walk(model)
    for cls, name in seen.items():
        attr.wrap(cls, name)
    attr.wrap(nn.Linear, "Linear")
    if hasattr(nn, "QuantizedLinear"):
        attr.wrap(nn.QuantizedLinear, "QuantizedLinear")
    return attr, sorted(seen.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=64, help="prefill のトークン数")
    ap.add_argument("--rounds", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--attrib", action="store_true",
                    help="op census をモジュール別に分ける (グラフを 1 本余分に組む)")
    ap.add_argument("--no-fusions", action="store_true",
                    help="mlxturbo の既定の融合を掛けない (素の mlx_lm)")
    ap.add_argument("--split-cb", action="store_true",
                    help="MLX_MAX_OPS_PER_BUFFER=1 (配分は厳密、壁時計は壊れる)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.split_cb:
        os.environ["MLX_MAX_OPS_PER_BUFFER"] = "1"

    import mlx.core as mx

    import mlxturbo  # noqa: F401  arch の登録
    from mlx_lm.utils import load

    from decode_gpu_trace import Probe

    # **pipeline を 1 つも作る前に差し替える。**Metal の具象クラスはドライバが
    # GPU を 1 回使うまでロードされないので、まず極小の計算を 1 本流してから
    # install する。モデルの読み込みより後だと、読み込み中に作られた pipeline
    # が名前の台帳に載らず「(pre-install pipeline)」に潰れる。
    mx.eval(mx.zeros((4, 4)) @ mx.ones((4, 4)))
    probe = Probe()
    probe.install()

    t0 = time.perf_counter()
    model, tok = load(os.path.expanduser(args.model))
    load_s = time.perf_counter() - t0

    fusions = None
    if not args.no_fusions:
        try:
            from mlxturbo.runner import enable_default_fusions

            fusions = enable_default_fusions(model, log_prefix="[glue_probe]")
        except Exception as e:      # noqa: BLE001  族が合わなければ素のまま
            fusions = f"(掛からず: {type(e).__name__}: {e})"

    wb = weight_bytes(model)
    floor_ms = wb / PEAK_BW * 1000.0

    vocab = getattr(getattr(model, "args", None), "vocab_size", None)
    if vocab is None:
        vocab = 32000
    ids = mx.array([[(i * 7 + 11) % 20000 for i in range(args.ctx)]])

    cache = model.make_cache() if hasattr(model, "make_cache") else None
    if cache is None:
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(model)

    # ---- prefill + burn-in (プロセス起動直後の 1 本目は +7〜9% 遅い) ----
    logits = model(ids, cache=cache)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
    for _ in range(4):
        logits = model(y[:, None], cache=cache)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)

    probe.enable(False)
    probe.reset()

    # ---- 計測 ----
    step_ms: list[float] = []
    for i in range(args.warmup + args.rounds):
        if i == args.warmup and probe.available:
            probe.quiesce()
            probe.reset()
            probe.enable(True)
        t = time.perf_counter()
        logits = model(y[:, None], cache=cache)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        dt = (time.perf_counter() - t) * 1000.0
        if i >= args.warmup:
            step_ms.append(dt)
    st = {"dispatches": 0, "kernels": []}
    if probe.available:
        probe.quiesce()
        probe.enable(False)
        st = probe.stats()

    n = len(step_ms)
    wall = sum(step_ms) / n
    res = {
        "model": args.model,
        "ctx": args.ctx,
        "rounds": n,
        "split_cb": bool(args.split_cb),
        "fusions": str(fusions),
        "load_s": round(load_s, 2),
        "weight_bytes": wb,
        "weight_read_floor_ms": round(floor_ms, 3),
        "wall_ms_per_step": round(wall, 3),
        "wall_ms_median": round(statistics.median(step_ms), 3),
        "dispatch_per_step": round(st["dispatches"] / n, 1),
        "cb_per_step": round(st.get("command_buffers", 0) / n, 1),
        "gpu_sum_ms_per_step": round(st.get("gpu_sum_ms", 0.0) / n, 3),
        "gpu_union_ms_per_step": round(st.get("gpu_union_ms", 0.0) / n, 3),
        "kernels": [
            {"name": k["name"],
             "count_per_step": round(k["count"] / n, 2),
             "gpu_ms_per_step": round(k["gpu_ms"] / n, 4)}
            for k in st.get("kernels", [])
        ],
    }
    union = res["gpu_union_ms_per_step"]
    res["gpu_busy_frac"] = round(union / wall, 4) if wall else 0.0
    res["glue_budget_ms"] = round(union - floor_ms, 3)

    print(f"\n=== {args.model}  ctx={args.ctx}  rounds={n} ===")
    print(f"  読み込み {load_s:.1f}s  融合 {fusions}")
    print(f"  重み {wb/1e9:.2f} GB  → 409.6 GB/s の下限 {floor_ms:.2f} ms/step"
          f"  ※ dense のときだけ意味がある (MoE は 1 トークンで全専門家を読まない)")
    print(f"  壁時計 {wall:.3f} ms/step (中央値 {res['wall_ms_median']:.3f})")
    print(f"  dispatch {res['dispatch_per_step']}/step  CB {res['cb_per_step']}/step")
    print(f"  GPU 和 {res['gpu_sum_ms_per_step']:.3f} / 和集合 {union:.3f} ms"
          f"  稼働率 {res['gpu_busy_frac']*100:.1f}%")
    print(f"  糊の予算 (GPU 和集合 - 重み読みの下限) {res['glue_budget_ms']:.2f} ms/step")
    print(f"\n  上位カーネル (等分配分。絶対値は信用しない)")
    print(f"    {'カーネル':<52s} {'回/step':>8s} {'ms/step':>9s} {'us/回':>8s}")
    for k in res["kernels"][:30]:
        c = k["count_per_step"]
        print(f"    {k['name'][:52]:<52s} {c:8.1f} {k['gpu_ms_per_step']:9.3f}"
              f" {1000*k['gpu_ms_per_step']/max(c,1e-9):8.1f}")

    # ---- op census ----
    if args.attrib:
        attr, classes = install_attrib(model)
        attr.enabled = True
        out = model(y[:, None], cache=cache)
        attr.claim("(top level)", [out])
        attr.enabled = False
        mx.eval(out)
        attr.report("op census (モジュール別)")
        res["attrib_classes"] = classes
        res["op_census"] = {f"{o}|{p}": n for (o, p), n in attr.counts.items()}
    else:
        cen = op_census(model, cache, int(y.item()) if hasattr(y, "item") else 5)
        print("\n  op census (全体)")
        for op, c in cen.most_common(40):
            print(f"    {op:36s} {c:6d}")
        res["op_census"] = dict(cen)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"\n  書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
