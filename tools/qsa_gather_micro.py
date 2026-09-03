"""合成テンソルで QSA prefill attention のカーネル候補を測る (レーン 3、多日)。

`docs/research/LANES-2026-09.md` の「## レーン 3」と
`docs/research/SESSION-2026-09-02-CATCHUP.md` の「レーン 3 の判定の訂正」を
踏まえた再挑戦の計測台。kv in {8192, 16896}, S=2048, Hq=24, Hk=2, D=256, bf16
で、クエリごとに causal 内のランダム 512 ブロック (block_topk=512, cr=4,
選ぶ列は kv の 12% 相当) を選び、

  (a) dense sdpa + bool マスク (基準)
  (b) 現行 prefill_attn カーネル (mlxturbo/kernels/prefill_attn.py)
  (c) 新カーネル (mlxturbo/kernels/prefill_attn_v2.py)

を ABAB (交互) で測る。(b)(c) の出力は (a) と比べた最大絶対誤差・相対誤差を
出す (bf16 の許容は相対 1.5e-2)。

段階の内訳 (`--stage-breakdown`) は現行カーネルの 3 パス (bool→添字圧縮 /
K・V の load / スコア+softmax) を個別に切って測る (`prefill_attn_v2.py` の
stage="compact"/"load"/"full")。

使い方 (GPU 空きを確認してから):

    tools/biglock.sh .venv/bin/python tools/qsa_gather_micro.py \\
        --kvs 8192,16896 --variants a,b,c --reps 5

GPU の占有チェック (このツール自体は確認しない --- 呼ぶ前に手動で):

    pgrep -f "self_snapshot|mlxturbo.server|mlx-serve --serve|decode_ab|quant_eval|kernel_chain_cost|test_hc_kernel|hc_fire_diag|hc_modes_inmodel"
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

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import mlxturbo  # noqa: E402,F401
from mlxturbo.kernels import prefill_attn as PA  # noqa: E402

CR = 4
BLOCK_TOPK = 512
HQ = 24
HK = 2
HEAD_DIM = 256


def make_keep_block(S: int, kv_len: int, cr: int, block_topk: int, seed: int = 0):
    """QSA の規約に沿った (1, S, n_blocks) の ``keep_block`` を作る (causal 内)。

    ``offset = kv_len - S`` が大きい (S=2048 チャンクが長文脈の末尾) 前提で、
    全行の可視ブロック数が ``block_topk`` を上回ることを assert する
    (下回るとサンプリングの前提が崩れる --- kv/S の組が本番のこの帯に
    収まっていれば起きない)。
    """
    offset = kv_len - S
    n_blocks = kv_len // cr
    assert kv_len - n_blocks * cr == 0, "このマイクロは cr で割り切れる kv だけ扱う (tail=0)"
    rng = np.random.default_rng(seed)
    q_col = offset + np.arange(S)
    n_vis = np.minimum(n_blocks, (q_col + 1) // cr)
    assert int(n_vis.min()) >= block_topk, (
        f"offset={offset} が小さすぎる "
        f"(最小可視ブロック {int(n_vis.min())} < block_topk {block_topk})"
    )
    scores = rng.random((S, n_blocks)).astype(np.float32)
    block_idx = np.arange(n_blocks)[None, :]
    scores = np.where(block_idx < n_vis[:, None], scores, -1.0)
    order = np.argpartition(-scores, kth=block_topk - 1, axis=-1)[:, :block_topk]
    keep = np.zeros((S, n_blocks), dtype=bool)
    rows = np.arange(S)[:, None]
    keep[rows, order] = True
    return keep[None], n_blocks, offset


def dense_ref(q, k, v, keep_block_mx, cr, scale):
    keep = mx.repeat(keep_block_mx, cr, axis=-1)
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=keep[:, None]
    )
    return out.transpose(0, 2, 1, 3)  # (B,S,H,D) にそろえる (カーネル出力と同じ並び)


def bench_ms(fn, reps: int, warm: int = 2) -> float:
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def max_err(got, ref):
    a = got.astype(mx.float32)
    b = ref.astype(mx.float32)
    diff = float(mx.max(mx.abs(a - b)))
    scale_ref = float(mx.max(mx.abs(b)))
    rel = diff / scale_ref if scale_ref > 0 else diff
    return diff, rel


def build_case(kv_len: int, S: int, seed: int = 0):
    from mlx_lm.models.cache import KVCache

    keep_np, n_blocks, offset = make_keep_block(S, kv_len, CR, BLOCK_TOPK, seed)
    mx.random.seed(seed)
    q = (mx.random.normal((1, HQ, S, HEAD_DIM)) * 0.3).astype(mx.bfloat16)
    k = (mx.random.normal((1, HK, kv_len, HEAD_DIM)) * 0.3).astype(mx.bfloat16)
    v = (mx.random.normal((1, HK, kv_len, HEAD_DIM)) * 0.3).astype(mx.bfloat16)
    mx.eval(q, k, v)
    keep_block = mx.array(keep_np)
    mx.eval(keep_block)

    cache = KVCache()
    k_view, v_view = cache.update_and_fetch(k, v)
    scale = HEAD_DIM ** -0.5
    return dict(
        q=q, k=k_view, v=v_view, keep_block=keep_block, cache=cache,
        cr=CR, kv_len=kv_len, n_blocks=n_blocks, block_topk=BLOCK_TOPK,
        offset=offset, scale=scale,
    )


def load_pre(path: str):
    """移植前の `prefill_attn.py` の写しを、パッケージの一員として読み込む。

    モジュール名を ``mlxturbo.kernels.prefill_attn_pre`` にしてやると、写しの
    中の ``from . import _fire`` が本物の `mlxturbo.kernels._fire` に解決する
    (ファイルの置き場所はリポジトリの外でよい)。移植前後のビット一致を
    **同じプロセスで** 見るための口。
    """
    import importlib.util

    name = "mlxturbo.kernels.prefill_attn_pre"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(variants, kvs, S, reps, out_path, pre_path=None):
    results = []
    PA2 = None
    PAP = None
    if "c" in variants or "d" in variants:
        from mlxturbo.kernels import prefill_attn_v2 as PA2  # noqa: N806
    if "e" in variants:
        if not pre_path:
            raise SystemExit("variant e には --pre <移植前の prefill_attn.py> が要る")
        PAP = load_pre(pre_path)  # noqa: N806

    for kv_len in kvs:
        case = build_case(kv_len, S)
        fns = {}
        if "a" in variants:
            fns["a_dense"] = lambda c=case: dense_ref(
                c["q"], c["k"], c["v"], c["keep_block"], c["cr"], c["scale"]
            )
        if "b" in variants:
            fns["b_current"] = lambda c=case: PA.prefill_attn(
                c["q"], c["k"], c["v"], c["keep_block"], c["cache"],
                cr=c["cr"], kv_len=c["kv_len"], n_blocks=c["n_blocks"],
                block_topk=c["block_topk"], offset=c["offset"], scale=c["scale"],
            )
        if "c" in variants:
            fns["c_new"] = lambda c=case: PA2.prefill_attn_v2(
                c["q"], c["k"], c["v"], c["keep_block"], c["cache"],
                cr=c["cr"], kv_len=c["kv_len"], n_blocks=c["n_blocks"],
                block_topk=c["block_topk"], offset=c["offset"], scale=c["scale"],
            )
        if "d" in variants:
            fns["d_u4"] = lambda c=case: PA2.prefill_attn_v2_u4(
                c["q"], c["k"], c["v"], c["keep_block"], c["cache"],
                cr=c["cr"], kv_len=c["kv_len"], n_blocks=c["n_blocks"],
                block_topk=c["block_topk"], offset=c["offset"], scale=c["scale"],
            )
        if "e" in variants:
            fns["e_pre"] = lambda c=case: PAP.prefill_attn(
                c["q"], c["k"], c["v"], c["keep_block"], c["cache"],
                cr=c["cr"], kv_len=c["kv_len"], n_blocks=c["n_blocks"],
                block_topk=c["block_topk"], offset=c["offset"], scale=c["scale"],
            )

        names = list(fns)
        outs = {}
        for n in names:
            o = fns[n]()
            mx.eval(o)
            outs[n] = o

        samples = {n: [] for n in names}
        for r in range(reps):
            order = names if r % 2 == 0 else names[::-1]
            for n in order:
                t0 = time.perf_counter()
                mx.eval(fns[n]())
                samples[n].append((time.perf_counter() - t0) * 1e3)

        row = {"kv": kv_len, "S": S}
        for n in names:
            row[n] = statistics.median(samples[n])
        if "a_dense" in outs:
            ref = outs["a_dense"]
            for n in names:
                if n == "a_dense":
                    continue
                diff, rel = max_err(outs[n], ref)
                row[f"{n}_abs_err"] = diff
                row[f"{n}_rel_err"] = rel
        if "a_dense" in row and "b_current" in row:
            row["b_over_a"] = row["b_current"] / row["a_dense"]
        if "a_dense" in row and "c_new" in row:
            row["c_over_a"] = row["c_new"] / row["a_dense"]
        if "a_dense" in row and "d_u4" in row:
            row["d_over_a"] = row["d_u4"] / row["a_dense"]
        if "b_current" in row and "d_u4" in row:
            row["d_over_b"] = row["d_u4"] / row["b_current"]
            # uint4 版は演算順が現行と同一なのでビット一致するはず
            row["d_bitident_b"] = bool(
                mx.array_equal(outs["d_u4"], outs["b_current"]).item()
            )
        if "b_current" in row and "e_pre" in row:
            row["b_over_e"] = row["b_current"] / row["e_pre"]
            row["b_bitident_e"] = bool(
                mx.array_equal(outs["b_current"], outs["e_pre"]).item()
            )

        err_str = "  ".join(
            f"{k}={v:.2e}" for k, v in row.items() if k.endswith("err")
        )
        print(
            f"kv={kv_len:6d} S={S}  "
            + "  ".join(f"{n}={row[n]:.2f}ms" for n in names)
            + ("  " + err_str if err_str else "")
            + (
                f"  bitident(d==b)={row['d_bitident_b']}"
                if "d_bitident_b" in row else ""
            )
            + (
                f"  bitident(b==e)={row['b_bitident_e']}"
                if "b_bitident_e" in row else ""
            ),
            flush=True,
        )
        results.append(row)
        del case, fns, outs

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(
            {"note": __doc__, "rows": results}, open(out_path, "w"),
            ensure_ascii=False, indent=1,
        )
        print("書き出し:", out_path)
    return results


def run_stage_breakdown(kvs, S, reps, out_path, stages=None):
    """カーネルの 3 パス (compact / load / score+softmax) の内訳を測る。

    ``stages`` に ``"compact_load_u4"`` / ``"full_u4"`` を混ぜると、P6 の
    uint4 化した段階 load の版も同じプロセス内で交互 (ABAB) に測る。
    load 単体は ``compact_load - compact`` (scalar) と
    ``compact_load_u4 - compact`` (uint4) の差で出す。
    """
    from mlxturbo.kernels import prefill_attn_v2 as PA2

    if stages is None:
        stages = ["compact", "compact_load", "full"]

    def call(name, c):
        base = name[:-3] if name.endswith("_u4") else name
        return PA2.prefill_attn_stage(
            base, c["q"], c["k"], c["v"], c["keep_block"], c["cache"],
            cr=c["cr"], kv_len=c["kv_len"], n_blocks=c["n_blocks"],
            block_topk=c["block_topk"], offset=c["offset"], scale=c["scale"],
            vec=name.endswith("_u4"),
        )

    results = []
    for kv_len in kvs:
        case = build_case(kv_len, S)
        fns = {st: (lambda c=case, s=st: call(s, c)) for st in stages}
        outs = {}
        for st in stages:
            o = fns[st]()
            mx.eval(o)
            outs[st] = o
        samples = {st: [] for st in stages}
        for r in range(reps):
            order = stages if r % 2 == 0 else stages[::-1]
            for st in order:
                t0 = time.perf_counter()
                mx.eval(fns[st]())
                samples[st].append((time.perf_counter() - t0) * 1e3)
        row = {"kv": kv_len, "S": S}
        for st in stages:
            row[st] = statistics.median(samples[st])
        parts = []
        if "compact" in row:
            row["compact_ms"] = row["compact"]
            parts.append(f"compact={row['compact_ms']:.2f}ms")
            for st in ("compact_load", "compact_load_u4"):
                if st in row:
                    row[f"load_ms{st[len('compact_load'):]}"] = row[st] - row["compact"]
            if "load_ms" in row:
                parts.append(f"load={row['load_ms']:.2f}ms")
            if "load_ms_u4" in row:
                parts.append(f"load_u4={row['load_ms_u4']:.2f}ms")
        for full, cl in (("full", "compact_load"), ("full_u4", "compact_load_u4")):
            if full in row and cl in row:
                key = "score_softmax_ms" + full[len("full"):]
                row[key] = row[full] - row[cl]
                parts.append(f"{key.replace('_ms','')}={row[key]:.2f}ms")
        for full in ("full", "full_u4"):
            if full in row:
                parts.append(f"{full}={row[full]:.2f}ms")
        # 段階の出力そのものも一致を見る (compact_load 系は load した値の和)
        for a_, b_ in (("compact_load", "compact_load_u4"), ("full", "full_u4")):
            if a_ in outs and b_ in outs:
                row[f"bitident_{b_}"] = bool(
                    mx.array_equal(outs[a_], outs[b_]).item()
                )
                parts.append(f"bitident({b_})={row[f'bitident_{b_}']}")
        print(f"kv={kv_len:6d}  " + "  ".join(parts), flush=True)
        results.append(row)
        del case, fns, outs

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(
            {"note": "prefill_attn 内訳 (compact/load/score+softmax)", "rows": results},
            open(out_path, "w"), ensure_ascii=False, indent=1,
        )
        print("書き出し:", out_path)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvs", default="8192,16896")
    ap.add_argument("--S", type=int, default=2048)
    ap.add_argument(
        "--variants", default="a,b,c",
        help=(
            "a=dense基準 b=現行カーネル c=新カーネル d=uint4 load 版 (P6) "
            "e=移植前の本番カーネル (--pre で写しを渡す)"
        ),
    )
    ap.add_argument(
        "--pre", default=None,
        help="variant e が読む、移植前の prefill_attn.py の写しのパス",
    )
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--stage-breakdown", action="store_true", help="現行カーネルの3パス内訳を測る")
    ap.add_argument(
        "--stages", default="compact,compact_load,full",
        help="--stage-breakdown で測る段階 (compact_load_u4 / full_u4 を含められる)",
    )
    ap.add_argument("--out", default="bench/results/qsa-gather-micro.json")
    a = ap.parse_args()
    if not mx.metal.is_available():
        print("Metal が使えない")
        raise SystemExit(1)
    mx.set_default_device(mx.gpu)
    kvs = [int(x) for x in a.kvs.split(",")]
    if a.stage_breakdown:
        out = a.out if a.out != "bench/results/qsa-gather-micro.json" else "bench/results/qsa-gather-micro-breakdown.json"
        run_stage_breakdown(
            kvs, a.S, a.reps, out, [s for s in a.stages.split(",") if s]
        )
    else:
        variants = set(a.variants.split(","))
        run(variants, kvs, a.S, a.reps, a.out, a.pre)
    os._exit(0)


if __name__ == "__main__":
    main()
