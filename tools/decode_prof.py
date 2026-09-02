"""相手 (`~/dev/mlx-serve/src/transformer.zig` の `[decode-prof]`,
12872-13172 / 23882-23899) と同じ切り方で decode S=1 の forward を部品ごとに
割る。**部品の境界で `mx.eval` を強制してから時計を読む** -- 融合や
async_eval との重なりを壊すやり方なので、**サイジング (絶対値) には使わない
こと**。両者を同じ切り方で割ってはじめて差が部品に帰属できる、というのが
この道具の唯一の存在理由。

## 割り方

    embed     embed_tokens + tile + _prelude
    attn      層ごとの pre_mlp (PLE + attention/GDN の mixer + HC read)、全層合計
    mlp       層ごとの MoE (SparseMoeBlock)、全層合計
    combine   層ごとの HC write (`DecoderLayer._combine`)、全層合計
    lmhead    hyper_connection_mixer + 出力射影

`--moe-detail` で `mlp` の中を router/experts/shared にさらに割る
(`tools/prefill_anatomy.py` の `moe_parts` と同じ手法だが、値は変えない
-- router の融合 (`_router513`) と shared の融合 (`_wide_shared`) の分岐も
本家のまま transcribe する)。層種別 (linear_attention / full_attention) 別
の内訳も出す。

engine の組み方・prefill 1 回化・キャッシュ退避復元・回文掃引・中央値の
出し方は `tools/verify_width_cost.py` と `tools/decode_ab.py` をそのまま
import して使い回す (写しを作ると挙動がずれる)。「通常 forward」の基準値は
production と同じ `spec_flash._staged_forward` (段階投入・async_eval あり)
を 1 回 eval しただけの壁時計で、強制 eval を挟んだ合計との差が「強制 eval
の上乗せ」。

    tools/biglock.sh .venv/bin/python tools/decode_prof.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep \\
        --ctx 17000 --moe-detail
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# decode_ab.py / verify_width_cost.py と同じキャッシュ退避復元・engine
# 構築・prefill 1 回化・draft 列作り・回文掃引・中央値のヘルパーをそのまま
# 使う (写しを作ると挙動がずれる)。
from decode_ab import _restore, prefill_once  # noqa: E402
from verify_width_cost import (  # noqa: E402
    build_pair,
    build_prompt_ids,
    build_runner,
    summarize,
    sweep_order,
)


def _moe_call_with_eval(mod, x):
    """`SparseMoeBlock.__call__` (`_vendor/qwen4_exp.py`) の写し。
    router / experts / shared の境界で `mx.eval` を挟むだけで、計算内容 --
    融合ルータ (`_router513`) 分岐と融合 shared (`_wide_shared`) 分岐を含む
    -- は本家と完全に同一にする (値は変えない)。戻り値は
    ``(result, {"router": s, "experts": s, "shared": s})`` (秒)。

    `_router513` 経路では shared expert がバンク 513 として router+experts
    に畳み込み済みのため、shared 単独の区間は存在しない (0.0 を返す)。
    """
    import mlx.core as mx
    import mlx.nn as nn

    r513 = getattr(mod, "_router513", None)
    if r513 is not None:
        t0 = time.perf_counter()
        logits = x.astype(mx.float32) @ r513.T
        lr = logits[..., :512]
        sg = mx.sigmoid(logits[..., 512:])
        idx = mx.argpartition(-lr, mod.top_k - 1, axis=-1)[..., : mod.top_k]
        w = mx.softmax(mx.take_along_axis(lr, idx, axis=-1), axis=-1, precise=True)
        idx = mx.concatenate(
            [idx, mx.full((*idx.shape[:-1], 1), 512, dtype=idx.dtype)], axis=-1
        )
        w = mx.concatenate([w, sg], axis=-1)
        mx.eval(idx, w)
        t_router = time.perf_counter() - t0

        t0 = time.perf_counter()
        out = (mod.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        mx.eval(out)
        t_experts = time.perf_counter() - t0
        return out, {"router": t_router, "experts": t_experts, "shared": 0.0}

    t0 = time.perf_counter()
    logits = mod.gate(x.astype(mx.float32))
    idx = mx.argpartition(-logits, mod.top_k - 1, axis=-1)[..., : mod.top_k]
    w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
    mx.eval(idx, w)
    t_router = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = (mod.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
    mx.eval(out)
    t_experts = time.perf_counter() - t0

    t0 = time.perf_counter()
    wide = getattr(mod, "_wide_shared", None)
    if wide is None:
        result = out + mx.sigmoid(mod.shared_expert_gate(x)) * mod.shared_expert(x)
    else:
        wq, sc, bi, gs, bits, h = wide
        gus = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=gs, bits=bits)
        g, u, sg = gus[..., :h], gus[..., h : 2 * h], gus[..., 2 * h :]
        shared = mod.shared_expert.down_proj(nn.silu(g) * u)
        result = out + mx.sigmoid(sg) * shared
    mx.eval(result)
    t_shared = time.perf_counter() - t0

    return result, {"router": t_router, "experts": t_experts, "shared": t_shared}


def _segmented_forward(model, ids, caches, moe_detail: bool = False):
    """`Qwen4ExpModel.__call__` (`_vendor/qwen4_exp.py:1612-1623`) の写し。
    `mlxturbo/spec_flash.py:_staged_forward` と同じ前段 (`_prelude`) ・
    層ループの骨格を使うが、段階投入 (`async_eval`) の代わりに**部品の
    境界で `mx.eval` を強制する** (相手の切り方に合わせるための、本物の
    差分はここだけ)。呼び出し側で `capture(model)` の中に置くこと
    (production の検証 forward と同じ状態で GDN/PLE/HC を通すため)。

    戻り値は ``(logits, seg, moe_seg, by_type)``。``seg`` は
    ``{"embed", "attn", "mlp", "combine", "lmhead", "serial_total"}`` -> 秒。
    ``moe_seg`` は ``moe_detail`` のときだけ ``{"router", "experts",
    "shared"}`` -> 秒 (全層合計)、それ以外は ``None``。``by_type`` は
    層種別 (``linear_attention`` / ``full_attention``) ごとの
    ``{"attn", "mlp", "combine"}`` -> 秒 (全層合計) と ``"n"`` (層数)。
    """
    import mlx.core as mx

    m = model.model
    seg = {"embed": 0.0, "attn": 0.0, "mlp": 0.0, "combine": 0.0, "lmhead": 0.0}
    moe_seg = {"router": 0.0, "experts": 0.0, "shared": 0.0} if moe_detail else None
    by_type: dict[str, dict] = {}

    t0 = time.perf_counter()
    h = m.embed_tokens(ids)
    mask, conv_mask, prev_ctx = m._prelude(ids, h, caches)
    h = mx.tile(h, (1, 1, m.hc))
    mx.eval(h)
    seg["embed"] = time.perf_counter() - t0

    for layer, c in zip(m.layers, caches):
        idx_c = c.indexer if (c is not None and hasattr(c, "indexer")) else None
        lt = layer.layer_type
        d = by_type.setdefault(lt, {"attn": 0.0, "mlp": 0.0, "combine": 0.0, "n": 0})
        d["n"] += 1

        t0 = time.perf_counter()
        x, hyper, inject = layer.pre_mlp(h, m.rope, mask, conv_mask, c, idx_c, ids, prev_ctx)
        mx.eval(x, hyper, inject)
        dt_attn = time.perf_counter() - t0

        if moe_detail:
            mlp_out, mp = _moe_call_with_eval(layer.mlp, x)
            dt_mlp = mp["router"] + mp["experts"] + mp["shared"]
            for k in moe_seg:
                moe_seg[k] += mp[k]
        else:
            t0 = time.perf_counter()
            mlp_out = layer.mlp(x)
            mx.eval(mlp_out)
            dt_mlp = time.perf_counter() - t0

        t0 = time.perf_counter()
        h = layer._combine(hyper, mlp_out, inject)
        mx.eval(h)
        dt_combine = time.perf_counter() - t0

        seg["attn"] += dt_attn
        seg["mlp"] += dt_mlp
        seg["combine"] += dt_combine
        d["attn"] += dt_attn
        d["mlp"] += dt_mlp
        d["combine"] += dt_combine

    # hyper_connection_mixer (最終合成) は「相手と並べやすい 1 行」の 5 キー
    # (embed/attn/mlp/combine/lmhead) には無い区分なので、直後に読む出力射影
    # と合わせて lmhead に含める。
    t0 = time.perf_counter()
    out = m.hyper_connection_mixer(h)
    if model.args.text.tie_word_embeddings:
        logits = m.embed_tokens.as_linear(out)
    else:
        logits = model.lm_head(out)
    mx.eval(logits)
    seg["lmhead"] = time.perf_counter() - t0

    seg["serial_total"] = sum(
        seg[k] for k in ("embed", "attn", "mlp", "combine", "lmhead")
    )
    return logits, seg, moe_seg, by_type


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None, help="既定は --model の中の mtp.safetensors")
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=0, help="既定 0 = 短プロンプト")
    ap.add_argument("--widths", default="1", help="decode の幅 S (既定 1、2 も可)")
    ap.add_argument("--reps", type=int, default=30, help="S ごとの反復回数 (最初の 3 回を含む)")
    ap.add_argument(
        "--moe-detail",
        action="store_true",
        help="mlp を router/experts/shared にさらに割る (追加の mx.eval が入るぶん mlp 自体の上乗せも増える)",
    )
    ap.add_argument("--out", default=str(REPO_ROOT / "bench" / "results" / "decode-prof.json"))
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    if not widths:
        print("--widths が空")
        return 1
    if args.reps < 3:
        print("--reps は 3 より大きい必要がある (最初の 3 回を捨てる作法のため)")
        return 1

    eng, model, tok, eos_ids = build_runner(args)
    ids = build_prompt_ids(tok, args.ctx)

    import mlx.core as mx
    from mlxturbo.spec_flash import _staged_forward  # noqa: SLF001 (依頼どおり直接使う)
    from mlxturbo.spec_flash import capture

    print(
        f"ctx={ids.shape[1]} (--ctx {args.ctx})  widths={widths}  reps={args.reps}"
        f"  moe_detail={args.moe_detail}"
    )
    print(
        "  相手 ([decode-prof], ~/dev/mlx-serve/src/transformer.zig) と同じ切り方: "
        "部品境界で mx.eval を強制してから時計を読む。サイジング用ではない。"
    )

    caches, snap, resume, _first = prefill_once(eng, ids, eos_ids)
    print(f"  prefill 1 回だけ流した (n={ids.shape[1]})。以降は同じ状態から退避・復元する。")

    pair_full, _cur = build_pair(eng, resume, max(widths))

    rounds = 2 if args.reps >= 2 else 1
    base = args.reps // rounds
    rem = args.reps % rounds

    seg_keys = ("embed", "attn", "mlp", "combine", "lmhead", "serial_total")
    raw: dict[int, dict] = {
        s: {
            **{k: [] for k in seg_keys},
            "baseline": [],
            "moe_router": [],
            "moe_experts": [],
            "moe_shared": [],
            "by_type": {},
        }
        for s in widths
    }

    for sweep_idx, order in enumerate(sweep_order(widths, rounds)):
        n_this = base + (1 if sweep_idx < rem else 0)
        for s in order:
            pair = pair_full[:, :s]
            for _ in range(n_this):
                # ---- 強制 eval の区切りで測る本命 ----
                _restore(caches, snap)
                with capture(model):
                    logits, seg, moe_seg, by_type = _segmented_forward(
                        model, pair, caches, moe_detail=args.moe_detail
                    )
                del logits
                for k in seg_keys:
                    raw[s][k].append(seg[k])
                if args.moe_detail:
                    raw[s]["moe_router"].append(moe_seg["router"])
                    raw[s]["moe_experts"].append(moe_seg["experts"])
                    raw[s]["moe_shared"].append(moe_seg["shared"])
                for lt, d in by_type.items():
                    bucket = raw[s]["by_type"].setdefault(
                        lt, {"attn": [], "mlp": [], "combine": [], "n": d["n"]}
                    )
                    bucket["attn"].append(d["attn"])
                    bucket["mlp"].append(d["mlp"])
                    bucket["combine"].append(d["combine"])

                # ---- 基準値: production と同じ通常 forward (強制 eval 無し) ----
                _restore(caches, snap)
                t0 = time.perf_counter()
                with capture(model):
                    lg = _staged_forward(model, pair, caches)
                mx.eval(lg)
                raw[s]["baseline"].append(time.perf_counter() - t0)

    print("\n=== decode-prof (相手の [decode-prof] と並べやすい 1 行) ===")
    result: dict = {
        "ctx": ids.shape[1],
        "ctx_arg": args.ctx,
        "widths": widths,
        "reps": args.reps,
        "moe_detail": args.moe_detail,
        "note": "部品境界で mx.eval を強制している (相手と同じ切り方)。サイジングには使わないこと。",
        "by_width": {},
    }
    for s in widths:
        r = raw[s]
        summ = {k: summarize(r[k]) for k in seg_keys}
        baseline = summarize(r["baseline"])
        n = summ["serial_total"]["n"]
        total_med = summ["serial_total"]["median_ms"]
        part_sum = sum(summ[k]["median_ms"] for k in ("embed", "attn", "mlp", "combine", "lmhead"))
        gap = (part_sum - total_med) / total_med * 100 if total_med else 0.0
        overhead = total_med - baseline["median_ms"]
        overhead_pct = overhead / baseline["median_ms"] * 100 if baseline["median_ms"] else 0.0

        moe_summ = None
        moe_suffix = ""
        if args.moe_detail:
            moe_summ = {
                "router": summarize(r["moe_router"]),
                "experts": summarize(r["moe_experts"]),
                "shared": summarize(r["moe_shared"]),
            }
            moe_suffix = (
                " | moe: router={:.3f} experts={:.3f} shared={:.3f}".format(
                    moe_summ["router"]["median_ms"],
                    moe_summ["experts"]["median_ms"],
                    moe_summ["shared"]["median_ms"],
                )
            )

        print(f"\n-- S={s} (n={n}) --")
        print(
            f"  [decode-prof] n={n} serial/tok={total_med:.3f}ms"
            f" embed={summ['embed']['median_ms']:.3f}"
            f" attn={summ['attn']['median_ms']:.3f}"
            f" mlp={summ['mlp']['median_ms']:.3f}"
            f" combine={summ['combine']['median_ms']:.3f}"
            f" lmhead={summ['lmhead']['median_ms']:.3f} ms" + moe_suffix
        )
        print(f"  部品和 = {part_sum:.3f} ms  (壁時計比 {gap:+.1f}%)")
        print(
            f"  通常 forward (_staged_forward、強制 eval 無し) 中央値 = {baseline['median_ms']:.3f} ms"
            f"  [強制 eval の上乗せ {overhead:+.3f} ms ({overhead_pct:+.1f}%)]"
        )

        by_type_summ = {}
        if r["by_type"]:
            print("  層種別ごと (中央値 ms, 全層合計):")
            for lt, bucket in r["by_type"].items():
                bt = {
                    "n_layers": bucket["n"],
                    "attn": summarize(bucket["attn"]),
                    "mlp": summarize(bucket["mlp"]),
                    "combine": summarize(bucket["combine"]),
                }
                by_type_summ[lt] = bt
                print(
                    f"    {lt:18s}(n={bucket['n']:2d})"
                    f"  attn={bt['attn']['median_ms']:8.3f}"
                    f"  mlp={bt['mlp']['median_ms']:8.3f}"
                    f"  combine={bt['combine']['median_ms']:8.3f}"
                )

        result["by_width"][str(s)] = {
            "segments": summ,
            "baseline": baseline,
            "forced_eval_overhead_ms": overhead,
            "forced_eval_overhead_pct": overhead_pct,
            "part_sum_vs_wallclock_pct": gap,
            "moe": moe_summ,
            "by_layer_type": by_type_summ,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out_path}")
    # 計測ツールなので destructor (スレッドプール等の後始末) に用は無い。
    # interpreter shutdown 待ちでプロセスが Metal のメモリを握ったまま
    # 1 時間以上残った実測があるので、結果を書き終えたら即 _exit で落とす
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
