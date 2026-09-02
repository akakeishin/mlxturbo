"""prefill MoE のルーティング偏りが、gather_qmm のタイル水増しにどれだけ
効いているかを決着させる道具 (`docs/research/KERNEL-BRIEF-DECODE-BW.md` の
prefill 効率 47% の主因仮説の一つ)。

bm の式の出典: `~/dev/mlx-serve/lib/mlx-src/mlx/backend/metal/quantized.cpp:1497`
(`gather_qmm_rhs_nax`) -- ``bm = (M/E < 64) ? 32 : 64`` (M=総行数、E=専門家数)。
**平均 M/E だけでタイル幅を決める**ので、行数の少ない専門家はタイルを
埋められず空回りする。その空回り率 (タイル水増し率 = Σ ceil(rows_e/bm)*bm
/ Σ rows_e) を層ごと・行数モードごとに出す。

17k プロンプトのチャンク 0 と 7 (offset 0 / 14336、2048 tok ずつ) を、
rows=2048 (1 チャンク単独) と rows=8192 (前後を含む 4 チャンク連結 =
`MLXTURBO_PREFILL_GROUP=4` の group prefill と同じ行数) の両方で見る。
prefill は本物の経路 (実キャッシュ) を通して流し、`SparseMoeBlock.__call__`
を一時的にラップして top-k の専門家添字だけを層ごとに捕まえる (実行はせず
router の分岐だけ再現するので、素の `self.gate` 経路と
`enable_moe_shared_fold` の `_router513` 経路の両方に対応する)。

    tools/biglock.sh .venv/bin/python tools/moe_routing_skew.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def route_idx(mod, x, top_k: int, num_experts: int):
    """`SparseMoeBlock.__call__` の routing 分岐をそのまま再現し、top-k の
    専門家添字 (値は 0..num_experts-1) を返す。

    `enable_moe_shared_fold` (既定 off) が `_router513` を仕込んでいれば
    その分岐、無ければ素の `self.gate` 分岐 -- どちらも実際の forward が
    通る条件 (`mod` の属性だけで決まり、入力 x には依らない) と同じなので、
    ここで独立に計算しても実物の routing と一致する。
    """
    import mlx.core as mx

    r513 = getattr(mod, "_router513", None)
    if r513 is not None:
        logits = x.astype(mx.float32) @ r513.T
        lr = logits[..., :num_experts]
        return mx.argpartition(-lr, top_k - 1, axis=-1)[..., :top_k]
    logits = mod.gate(x.astype(mx.float32))
    return mx.argpartition(-logits, top_k - 1, axis=-1)[..., :top_k]


def pending(caches):
    """キャッシュに残っている遅延ノードを eval 対象として集める。

    `tools/prefill_anatomy.py` の同名関数と同じ理由 -- キャッシュへの
    書き込みは誰かが読むまで走らない MLX の遅延評価で、チャンクの区切りで
    ここを eval しておかないと書き込みグラフが次のチャンクへこぼれ、
    チャンクを跨いで「作って捨てる」遅延グラフが積み上がる
    (CLAUDE.md の計測の作法)。
    """
    out = []
    for c in caches:
        st = getattr(c, "state", None)
        if isinstance(st, (list, tuple)):
            out.extend(v for v in st if v is not None)
        elif st is not None:
            out.append(st)
        ic = getattr(c, "indexer", None)
        if ic is not None and ic._buf is not None:
            out.append(ic._buf)
    return out


def tile_inflation(counts, bm: int) -> float:
    """タイル水増し率 = Σ_e ceil(rows_e / bm) * bm / Σ_e rows_e。"""
    import numpy as np

    total = float(counts.sum())
    if total <= 0:
        return 1.0
    inflated = float(np.ceil(counts / bm).sum() * bm)
    return inflated / total


def split_inflation(counts, threshold_bm: int, rowwise: bool) -> float:
    """行数が `threshold_bm` 未満の専門家だけを別呼び出しに分けた場合の
    水増し率の見積もり。

    - under 側: `rowwise=False` なら bm=32 のタイルのまま、`rowwise=True`
      なら行単位 (`gather_qmv` 的、水増しゼロ) で処理したと仮定する。
    - over 側 (行数の多い専門家): 分離後も平均 M/E は 64 を超え続けるはず
      なので bm=64 のタイルのまま。
    """
    import numpy as np

    total = float(counts.sum())
    if total <= 0:
        return 1.0
    under = counts < threshold_bm
    if rowwise:
        lo = float(counts[under].sum())
    else:
        lo = float(np.ceil(counts[under] / 32).sum() * 32)
    hi = float(np.ceil(counts[~under] / 64).sum() * 64)
    return (lo + hi) / total


def layer_stats(counts, num_experts: int) -> dict:
    """1 層ぶんの専門家別行数ヒストグラムから、水増し率一式を出す。"""
    import numpy as np

    total = float(counts.sum())
    mean_me = total / num_experts if num_experts else 0.0
    # gather_qmm_rhs_nax と同じ決め方 (平均 M/E で 32/64)
    chosen_bm = 32 if mean_me < 64 else 64
    under_share = (
        float(counts[counts < chosen_bm].sum()) / total if total > 0 else 0.0
    )
    stats = {
        "rows": total,
        "mean_per_expert": mean_me,
        "min": float(counts.min()),
        "p10": float(np.percentile(counts, 10)),
        "median": float(np.percentile(counts, 50)),
        "p90": float(np.percentile(counts, 90)),
        "max": float(counts.max()),
        "zero_experts": int((counts == 0).sum()),
        "chosen_bm": chosen_bm,
        "tile_inflation_bm32": tile_inflation(counts, 32),
        "tile_inflation_bm64": tile_inflation(counts, 64),
        "under_bm_row_share": under_share,
        "split_bm32_inflation": split_inflation(counts, chosen_bm, rowwise=False),
        "split_rowwise_inflation": split_inflation(counts, chosen_bm, rowwise=True),
    }
    stats["tile_inflation_chosen"] = stats[f"tile_inflation_bm{chosen_bm}"]
    return stats


def avg_stats(layer_dicts: list) -> dict:
    """48 層ぶんの層別統計を、単純平均で 1 行にまとめる。"""
    skip = {"layer", "chosen_bm"}
    keys = [k for k in layer_dicts[0] if k not in skip]
    out = {k: float(statistics.mean(d[k] for d in layer_dicts)) for k in keys}
    out["layers_bm32"] = sum(1 for d in layer_dicts if d["chosen_bm"] == 32)
    out["layers_bm64"] = sum(1 for d in layer_dicts if d["chosen_bm"] == 64)
    return out


def block_for(point: int, n_full: int, width: int = 4) -> list:
    """`point` を含む、幅 `width` の連続チャンク区間を選ぶ。

    末尾に寄っている点 (例: 最終チャンク) だと前向きの区間が完全チャンク数を
    はみ出すので、区間全体を `[0, n_full)` に収まるようクランプする
    (17k デフォルトのチャンク 7 のような「最後の完全チャンク」でも
    4 チャンク連結が組めるようにするため)。
    """
    if n_full < width:
        raise ValueError(f"完全チャンクが {n_full} 本しか無く、幅 {width} を組めない")
    start = min(max(0, point - (width - 1)), n_full - width)
    return list(range(start, start + width))


def print_table(case: dict) -> None:
    print(f"  {'層':>4s}{'meanM/E':>9s}{'bm':>4s}{'tile32':>9s}{'tile64':>9s}"
          f"{'under%':>8s}{'split32':>9s}{'splitRow':>9s}")
    for st in case["layers"]:
        print(f"  {st['layer']:4d}{st['mean_per_expert']:9.1f}{st['chosen_bm']:4d}"
              f"{st['tile_inflation_bm32']:9.3f}{st['tile_inflation_bm64']:9.3f}"
              f"{st['under_bm_row_share'] * 100:7.1f}%"
              f"{st['split_bm32_inflation']:9.3f}"
              f"{st['split_rowwise_inflation']:9.3f}")
    avg = case["avg"]
    print(f"\n  [point={case['point']} rows={case['rows']} chunks={case['chunks']}]"
          f" 全層平均: mean_M/E={avg['mean_per_expert']:.1f}"
          f" tile_bm32={avg['tile_inflation_bm32']:.3f}"
          f" tile_bm64={avg['tile_inflation_bm64']:.3f}"
          f" chosen={avg['tile_inflation_chosen']:.3f}"
          f" under_bm_share={avg['under_bm_row_share']:.3f}"
          f" split_bm32={avg['split_bm32_inflation']:.3f}"
          f" split_rowwise={avg['split_rowwise_inflation']:.3f}"
          f" (bm32層={int(avg['layers_bm32'])} bm64層={int(avg['layers_bm64'])})",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--chunk", type=int, default=2048, help="prefill チャンク幅")
    ap.add_argument("--points", default="0,7",
                     help="チャンク番号 (0始まり)。カンマ区切り (既定: 0,7)")
    ap.add_argument("--rows", default="2048,8192",
                     help="--chunk 幅そのもの、または --chunk*4。カンマ区切り"
                          " (既定: 両方)")
    ap.add_argument("--out", default="bench/results/moe-routing-skew.json")
    args = ap.parse_args()

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    import mlxturbo  # noqa: F401
    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo.runner import enable_default_fusions

    model, tok = load(os.path.expanduser(args.model))
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[moe-routing-skew]")

    ta = model.args.text
    num_experts = ta.num_experts
    top_k = ta.num_experts_per_tok
    n_layers = ta.num_hidden_layers

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from _bench_text import long_prompts

    body = long_prompts(tok, args.ctx, ["上の文書の要点を 5 つに整理してください。"])[0]
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True))[None]
    n = ids.shape[1]
    step = args.chunk
    n_full = n // step
    if n_full < 1:
        print("完全チャンクが 0 本。--ctx を増やすこと")
        return 1

    points = [int(v) for v in args.points.split(",") if v.strip() != ""]
    rows_modes = [int(v) for v in args.rows.split(",") if v.strip() != ""]
    for p in points:
        if not (0 <= p < n_full):
            print(f"--points の {p} が範囲外 (完全チャンク数 {n_full})")
            return 1
    width4 = step * 4
    for r in rows_modes:
        if r not in (step, width4):
            print(f"--rows は {step} か {width4} のみ受け付ける (指定: {r})")
            return 1

    blocks: dict = {}
    if width4 in rows_modes:
        if n_full < 4:
            print("rows=chunk*4 には完全チャンクが最低 4 本要る (--ctx を増やすこと)")
            return 1
        for p in points:
            blocks[p] = block_for(p, n_full, 4)

    needed = set(points)
    for b in blocks.values():
        needed.update(b)
    max_chunk = max(needed)
    print(f"ctx={n} chunk={step} 完全チャンク={n_full} points={points}"
          f" rows={rows_modes} needed_chunks={sorted(needed)}"
          f" num_experts={num_experts} top_k={top_k} layers={n_layers}",
          flush=True)

    cache = model.make_cache()
    orig_call = Q.SparseMoeBlock.__call__
    per_chunk_counts: dict = {}

    for ci in range(max_chunk + 1):
        chunk_ids = ids[:, ci * step : (ci + 1) * step]
        if ci in needed:
            captured: list = []

            def wrap_call(self, x, _cap=captured):
                _cap.append((self, x))
                return orig_call(self, x)

            Q.SparseMoeBlock.__call__ = wrap_call
            out = model.model(chunk_ids, cache=cache)
            mx.eval([out] + pending(cache))
            Q.SparseMoeBlock.__call__ = orig_call

            idxs = [route_idx(mod, x, top_k, num_experts) for mod, x in captured]
            mx.eval(idxs)
            per_chunk_counts[ci] = [
                np.bincount(np.array(a).ravel(), minlength=num_experts)[:num_experts]
                for a in idxs
            ]
            print(f"  chunk {ci}: 層 {len(idxs)} 本 捕まえた", flush=True)
            captured.clear()
            del idxs
        else:
            out = model.model(chunk_ids, cache=cache)
            mx.eval([out] + pending(cache))
        mx.clear_cache()

    result = {
        "config": {
            "model": args.model,
            "ngram": args.ngram,
            "ctx": n,
            "chunk": step,
            "n_full": n_full,
            "points": points,
            "rows_modes": rows_modes,
            "num_experts": num_experts,
            "top_k": top_k,
            "num_layers": n_layers,
            "bm_formula_source":
                "mlx-src/mlx/backend/metal/quantized.cpp:1497 (gather_qmm_rhs_nax):"
                " bm = (M/E < 64) ? 32 : 64  (M=総行数, E=専門家数)",
        },
        "cases": [],
    }

    for p in points:
        for r in rows_modes:
            chunks = blocks[p] if r == width4 else [p]
            layer_list = []
            for layer_idx in range(n_layers):
                counts = sum(per_chunk_counts[c][layer_idx] for c in chunks)
                layer_list.append(
                    {"layer": layer_idx, **layer_stats(counts, num_experts)})
            case = {
                "point": p,
                "rows": r,
                "chunks": chunks,
                "layers": layer_list,
                "avg": avg_stats(layer_list),
            }
            result["cases"].append(case)
            print(flush=True)
            print_table(case)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n書いた: {out_path}", flush=True)

    # 計測ツールなので destructor 待ちでプロセスが Metal のメモリを握ったまま
    # 残る前例がある (prefill_anatomy.py と同じ理由) -- 即 _exit で落とす
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
