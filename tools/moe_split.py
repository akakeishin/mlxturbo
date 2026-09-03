"""MoE 1 層 (行数 M) の「行列積以外」の時間がどこに行くかを割る。

背景 (`docs/research/KERNEL-PROGRAM.md:1253` 付近の分解、ctx=16867、
chunk=2048、layer-major、出荷経路と同じ融合構成): MoE 48 層の prefill 時間
2041 ms のうち gather_qmm の行列積 (up/gate/down の 3 本) は 1238 ms で、
**残り 800 ms が行列積以外。内訳が未測。**候補: router (`self.gate(x)` の
fp32 qmm 2560->512 + softmax/正規化)、top-k (512 専門家から 10、
`mx.argpartition`)、行のソート (`argsort` で専門家順、`inv_order`)、x の
gather (`xx[order // top_k]`、8192 行なら 419MB)、gate/up の出力の SwiGLU
(silu x mul)、down 後の scatter/合成 (`out[inv_order]`、router 重みの
掛け算、`sum`)、共有専門家 (dense の GEMM)、その他 (最終加算・型変換)。

## 測るのは combine_fold 経路 (production の既定)

`mlxturbo/_vendor/qwen4_exp.py` の `SparseMoeBlock.__call__` は、行数
(B×S) が `_combine_fold_min_s` (`mlxturbo/fused.py` の
`enable_moe_combine_fold`、既定 64) 以上なら `_moe_combine_fold` を通る。
これは「router 重み w を down_proj の出力側で掛けて sum する」素の経路
(`(switch_mlp(x, idx) * w[..., None]).sum(-2)`) とは違い、**w を down_proj
の「入力」(SwiGLU 出力) に先掛けしてから down_proj を呼ぶ**別の graph
(実体化する量が 1/4 で済む代わりに、`switch_mlp.__call__` 自体は経由しない
-- gate_proj/up_proj/down_proj を自前で呼ぶ)。prefill (M=2048/8192 は
どちらも閾値 64 を大きく超える) は既定でこちらを通るので、**このツールは
combine_fold 経路だけを割る。**素の SwitchGLU 経路 (M<64 の decode/verify
幅、または `MLXTURBO_MOE_COMBINE_FOLD=0`) は対象外。

`tools/prefill_anatomy.py` の `moe_parts`/`moe_bounds` (部品ラベルは
router/topk/sort/up/gate/swiglu/down/unsort/combine/shared で似ている) は
**素の SwitchGLU 経路を `mlx_lm.models.switch_layers._gather_sort` 等を
直接呼んで再現したもの**で、combine_fold が既定 on の実機では通らない
graph を測っている。ここではその代わりに `_moe_combine_fold` の呼び出し順
そのままを割る。router 重みの掛け算 (fold の要) は down_proj の**前**で
起きるので、`combine` の内訳は「weight 乗算+cast (down_proj 前)」と
「unsort+sum (down_proj 後)」の 2 段に分かれる -- 表では両方を足した
1 行として出しつつ、内訳は別途表示・JSON に残す。

## GDN/PLE 用の道具との違い

`tools/gdn_split.py`/`tools/ple_split.py` はキャッシュ (conv_state/再帰状態)
を持つ層を測るので、実フォワードを繰り返すたびにキャッシュの
snapshot/restore が要る。`SparseMoeBlock` はキャッシュを持たない純関数
(`x -> out`) なので、その機構は要らない -- 捕まえた実引数 `x` に対して
部品を何度でもそのまま呼べる。

## 部品の切り方

    router       self.gate(x.astype(fp32))                    (fp32 qmm)
    topk         argpartition + take_along_axis + softmax
    sort         argsort x2 (order/inv_order) + idx の並べ替え (添字のみ)
    gather       x 行の gather (order // top_k) + w の並べ替え
    gate_up_qmm  up_proj + gate_proj (どちらも gather_qmm)
    swiglu       activation(x_up, x_gate) 単体 (silu x mul)
    down_qmm     down_proj (gather_qmm)
    combine      weight 乗算+cast (down_proj 前) + unsort+sum (down_proj 後)
    shared       共有専門家 (sigmoid ゲート込み、MLP は silu/mul が別 op)
    other        最終の型変換 + out+shared の加算

各部品は捕まえた実引数から作った中間値 (非計測で 1 回だけ確定) を使い、
部品そのものを N 回回して中央値を取る (`prefill_anatomy.med_ms` と同じ、
温めの 1 本は捨てる)。最後に部品和と、層の実 `__call__` を直接叩いた
壁時計を突き合わせて `部品和 ≈ 壁時計 (数%以内)` を確認する
(CLAUDE.md の計測の作法)。

## 入力の作り方

実モデル (既定 `~/models/ddalcu-mlxlm`) を `mlxturbo.runner.load_model` 相当
(`mlx_lm.load` + `enable_default_fusions`、`tools/gdn_split.py` と同じ経路)
で読み込み、**フックは本番と同じ既定を当てる** (r513/wide_shared の融合は
既定で入らないので未対応、assert で弾く)。実文書プロンプトを 1 回 prefill
し、対象層 (既定 20、DecoderLayer.mlp は全 48 層にある) の
`SparseMoeBlock.__call__` に渡る実引数 `x` を、先頭 M トークンぶんの
チャンクで捕まえる (M=2048 と M=8192 で別々に、使い捨てキャッシュで 1 回ずつ)。
乱数ではなく実ルーティング (実 router の出力にそのまま従う) を使う。

使い方 (GPU を使うので必ず biglock 経由で):

    tools/biglock.sh .venv/bin/python tools/moe_split.py \\
        --model ~/models/ddalcu-mlxlm --layer 20 --rows 2048,8192 \\
        --json bench/results/moe-split.json

## 常駐 worker (`tool` ジョブ)

98GB を読み直さずに済むよう `run_with_model(argv, bundle)` を持つ (規約は
`tools/ab_bundle.py` の docstring)。CLI (`main`) と worker は同じ
`parse_args` → `run` を通るので、**出力も終了コードも変わらない。**
捕まえた実引数 `x` は `SparseMoeBlock.__call__` を一時的に包んで取るが、
それは `capture_x` が自分の finally で戻す (worker から呼ばれたときは
`run_with_model` が例外経路の網としてもう一枚控える)。

**`--ngram` は worker の構成と一致させること。**PLE の埋め込みが変われば
層 20 に届く `x` が変わる。食い違えば 64 を返し、`tools/biglock.sh` が
従来どおり別プロセスで流し直す。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import prefill_anatomy as PA  # noqa: E402  (med_ms を共有)

PART_ORDER = ["router", "topk", "sort", "gather", "gate_up_qmm", "swiglu",
              "down_qmm", "combine", "shared", "other"]
PART_LABEL = {
    "router": "router (fp32 qmm 2560->512)",
    "topk": "topk (argpartition+softmax)",
    "sort": "sort (argsort x2、添字のみ)",
    "gather": "gather (x 行 + w の並べ替え)",
    "gate_up_qmm": "gate_up qmm (up_proj+gate_proj)",
    "swiglu": "swiglu (silu x mul)",
    "down_qmm": "down qmm",
    "combine": "combine (weight乗算+cast + unsort+sum)",
    "shared": "shared expert (ゲート込み)",
    "other": "other (型変換 + 最終加算)",
}


def capture_x(model, ids, layer_idx: int, M: int, mx, Q):
    """先頭 M トークンのチャンクを実フォワードで 1 回流し、層 layer_idx の
    SparseMoeBlock.__call__ に渡る実引数 x (評価済み) を捕まえる。

    `gdn_split.capture_layers` と違い、この後の計測点を続けない (MoE は
    kv に依存しない純関数なので、捕まえたらキャッシュは使い捨てでよい)。
    捕まえた直後に `x` を eval するのは、レイヤー 0..layer_idx-1 の費用を
    ここで確定させ、直後の isolated replay 計測に上流の遅延評価が混ざらない
    ようにするため (`ple_split.py` の `hidden` 先行 eval と同じ理由)。
    """
    target_mod = model.model.layers[layer_idx].mlp
    target_id = id(target_mod)
    captured: dict = {}
    orig_call = Q.SparseMoeBlock.__call__

    def wrapped(self, x):
        if id(self) == target_id:
            mx.eval(x)
            captured["x"] = x
        return orig_call(self, x)

    chunk = ids[:, :M]
    cache = model.make_cache()
    Q.SparseMoeBlock.__call__ = wrapped
    try:
        out = model.model(chunk, cache=cache)
        mx.eval(out)
    finally:
        Q.SparseMoeBlock.__call__ = orig_call
    mx.clear_cache()
    if "x" not in captured:
        raise RuntimeError(
            f"層 {layer_idx} の SparseMoeBlock.__call__ が呼ばれなかった")
    return captured["x"], target_mod


def measure_parts(mod, x, reps: int, mx, Q) -> dict:
    """捕まえた実引数 x で、combine_fold 経路の部品を測る。"""
    assert getattr(mod, "_combine_fold_min_s", None) is not None, (
        "enable_default_fusions が当たっていない (moe_combine_fold 未設定)。"
        " ロード順を確認すること")
    assert getattr(mod, "_router513", None) is None, (
        "想定外: enable_moe_shared_fold が有効 (このツールは既定構成専用)")
    assert getattr(mod, "_wide_shared", None) is None, (
        "想定外: MLXTURBO_WIDE=1 相当が有効 (このツールは既定構成専用)")

    B, S, _ = x.shape
    top_k = mod.top_k
    use_fold = B * S >= mod._combine_fold_min_s
    assert use_fold, (
        f"行数 {B * S} が _combine_fold_min_s ({mod._combine_fold_min_s}) 未満で"
        " fold 経路に入らない (素の SwitchGLU 経路になる)。--rows を増やすこと")

    sw = mod.switch_mlp
    gate = mod.gate
    se = mod.shared_expert
    seg = mod.shared_expert_gate
    sort_min = getattr(Q, "_MOE_COMBINE_SORT_MIN", 64)

    # -- 前提: 実物の中間値を非計測で 1 回だけ確定させる
    #    (gdn_split.measure_parts/ple_split.py と同じ流儀)
    logits = gate(x.astype(mx.float32))
    mx.eval(logits)
    idx = mx.argpartition(-logits, top_k - 1, axis=-1)[..., :top_k]
    w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
    mx.eval(idx, w)
    assert idx.size >= sort_min, (
        "do_sort=False の分岐は未対応 (このツールが想定する M では起きないはず)")

    xx0 = mx.expand_dims(x, (-2, -3))
    idx_flat = idx.flatten()
    order = mx.argsort(idx_flat)
    inv_order = mx.argsort(order)
    idx_s = idx_flat[order]
    mx.eval(order, inv_order, idx_s)

    xx = xx0.flatten(0, -3)[order // top_k]
    w_s = w.flatten()[order][:, None, None]
    mx.eval(xx, w_s)

    x_up = sw.up_proj(xx, idx_s, sorted_indices=True)
    x_gate = sw.gate_proj(xx, idx_s, sorted_indices=True)
    mx.eval(x_up, x_gate)

    act_raw = sw.activation(x_up, x_gate)
    mx.eval(act_raw)

    act = (act_raw * w_s).astype(x.dtype)
    mx.eval(act)

    down_out = sw.down_proj(act, idx_s, sorted_indices=True)
    mx.eval(down_out)

    summed = mx.unflatten(down_out[inv_order], 0, idx.shape).squeeze(-2).sum(axis=-2)
    mx.eval(summed)

    combined_cast = summed.astype(x.dtype)
    mx.eval(combined_cast)

    shared_gate_val = mx.sigmoid(seg(x))
    shared_val = se(x)
    shared_out = shared_gate_val * shared_val
    mx.eval(shared_gate_val, shared_val, shared_out)

    final = combined_cast + shared_out
    mx.eval(final)

    # -- 部品 (どれもキャッシュを持たないベタ関数、snapshot/restore は不要)
    def part_router():
        return gate(x.astype(mx.float32))

    def part_topk():
        ii = mx.argpartition(-logits, top_k - 1, axis=-1)[..., :top_k]
        return ii, mx.softmax(mx.take_along_axis(logits, ii, axis=-1),
                               axis=-1, precise=True)

    def part_sort():
        idxf = idx.flatten()
        o = mx.argsort(idxf)
        return o, mx.argsort(o), idxf[o]

    def part_gather():
        return (xx0.flatten(0, -3)[order // top_k],
                w.flatten()[order][:, None, None])

    def part_qmm_up_gate():
        return (sw.up_proj(xx, idx_s, sorted_indices=True),
                sw.gate_proj(xx, idx_s, sorted_indices=True))

    def part_swiglu():
        return sw.activation(x_up, x_gate)

    def part_down_qmm():
        return sw.down_proj(act, idx_s, sorted_indices=True)

    def part_combine_weight():
        return (act_raw * w_s).astype(x.dtype)

    def part_combine_unsort_sum():
        return mx.unflatten(down_out[inv_order], 0, idx.shape).squeeze(-2).sum(axis=-2)

    def part_shared():
        return mx.sigmoid(seg(x)) * se(x)

    def part_other():
        return summed.astype(x.dtype) + shared_out

    def whole():
        """層の実 __call__ を直接叩く (壁時計)。キャッシュ無しの純関数なので
        snapshot/restore は不要 (gdn_split.whole と違うところ)。"""
        return mod(x)

    ms = {}
    for key, fn in (
        ("router", part_router), ("topk", part_topk), ("sort", part_sort),
        ("gather", part_gather), ("gate_up_qmm", part_qmm_up_gate),
        ("swiglu", part_swiglu), ("down_qmm", part_down_qmm),
    ):
        ms[key] = PA.med_ms(fn, reps)
        mx.clear_cache()

    combine_weight_ms = PA.med_ms(part_combine_weight, reps)
    mx.clear_cache()
    combine_unsort_ms = PA.med_ms(part_combine_unsort_sum, reps)
    mx.clear_cache()
    ms["combine"] = combine_weight_ms + combine_unsort_ms

    ms["shared"] = PA.med_ms(part_shared, reps)
    mx.clear_cache()
    ms["other"] = PA.med_ms(part_other, reps)
    mx.clear_cache()

    layer_ms = PA.med_ms(whole, reps)
    mx.clear_cache()

    parts_sum = sum(ms[k] for k in PART_ORDER)
    gap_pct = (parts_sum - layer_ms) / layer_ms * 100 if layer_ms else 0.0
    return {
        "M": B * S,
        "ms": ms,
        "combine_breakdown_ms": {
            "weight_mul_cast": combine_weight_ms,
            "unsort_sum": combine_unsort_ms,
        },
        "parts_sum_ms": parts_sum,
        "layer_wallclock_ms": layer_ms,
        "gap_pct": gap_pct,
    }


def print_row(row: dict) -> None:
    ms = row["ms"]
    layer_ms = row["layer_wallclock_ms"]
    print(f"\n  [M={row['M']}]")
    print(f"    {'部品':38s}{'ms':>9s}{'割合':>8s}")
    for k in PART_ORDER:
        pct = ms[k] / layer_ms * 100 if layer_ms else 0.0
        print(f"    {PART_LABEL[k]:38s}{ms[k]:9.2f}{pct:7.1f}%")
    parts_pct = row["parts_sum_ms"] / layer_ms * 100 if layer_ms else 0.0
    print(f"    {'部品和':38s}{row['parts_sum_ms']:9.2f}{parts_pct:7.1f}%")
    print(f"    {'壁時計 (層 __call__ 直測)':38s}{layer_ms:9.2f}")
    print(f"    部品和 - 壁時計 = "
          f"{row['parts_sum_ms'] - layer_ms:+.2f} ms ({row['gap_pct']:+.1f}%)")
    if abs(row["gap_pct"]) > 10:
        print("    ** 10% を超えてずれている。見えていない項目があるか、"
              "単体計測が重なりを再現できていない **")
    cb = row["combine_breakdown_ms"]
    print("\n    combine の内訳 (参考、上の combine 行の中身):")
    print(f"      weight 乗算 + cast (down_proj 前)   {cb['weight_mul_cast']:9.2f} ms")
    print(f"      unsort + sum (down_proj 後)         {cb['unsort_sum']:9.2f} ms")
    print(flush=True)


# 「worker には載せられない」を表す終了コード。`tools/ab_submit.py` の
# NOT_ROUTABLE と同じ値で、`tools/biglock.sh` はこれを受けると worker に
# 98GB を返させてから、従来どおり別プロセスで流し直す。
NOT_ROUTABLE = 64


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm")
    ap.add_argument("--ngram", default=None,
                     help="省略時はモデル同梱の埋め込み表を使う (PLE 自体は動く)")
    ap.add_argument("--layer", type=int, default=20,
                     help="対象 DecoderLayer の index (0 始まり。全層に MoE がある)")
    ap.add_argument("--rows", default="2048,8192", help="M (=B*S) の候補、カンマ区切り")
    ap.add_argument("--ctx", type=int, default=None,
                     help="プロンプトの目安トークン数 (既定: max(rows)+300)")
    ap.add_argument("--reps", type=int, default=5, help="中央値を取るレップ数")
    ap.add_argument("--json", default=str(REPO_ROOT / "bench/results/moe-split.json"))
    return ap


def parse_args(argv=None):
    """引数を解釈して ``(args, rows, ctx)`` を返す。

    CLI (`main`) と常駐 worker (`run_with_model`) の両方がこれを通る。
    """
    args = build_parser().parse_args(argv)
    rows = [int(r) for r in args.rows.split(",")]
    ctx = args.ctx or (max(rows) + 300)
    return args, rows, ctx


def run(model, tok, args, rows, ctx) -> int:
    """読み込み済みのモデルで内訳を測る本体 (CLI と worker が共有する)。"""
    import mlx.core as mx

    import mlxturbo  # noqa: F401  -- qwen4_exp を _vendor 版に差し替える
    import mlx_lm.models.qwen4_exp as Q

    n_layers = len(model.model.layers)
    if not (0 <= args.layer < n_layers):
        print(f"--layer は 0..{n_layers - 1} の範囲で指定すること (指定={args.layer})")
        return 1
    target_mod = model.model.layers[args.layer].mlp
    if not hasattr(target_mod, "switch_mlp"):
        print(f"層 {args.layer} に MoE (switch_mlp) が無い")
        return 1

    from _bench_text import long_prompts

    body = long_prompts(tok, ctx, ["上の文書の要点を 5 つに整理してください。"])[0]
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True))[None]
    n = ids.shape[1]
    if n < max(rows):
        print(f"プロンプトが {n} トークンしか無い (--rows の最大 {max(rows)} に届かない)。"
              f"--ctx を増やすこと")
        return 1
    print(f"model={args.model} layer={args.layer} rows={rows} "
          f"prompt_tokens={n} reps={args.reps}", flush=True)

    results = {
        "model": args.model, "layer_idx": args.layer, "prompt_tokens": n,
        "reps": args.reps, "rows": [],
    }

    for M in rows:
        x, mod = capture_x(model, ids, args.layer, M, mx, Q)
        row = measure_parts(mod, x, args.reps, mx, Q)
        results["rows"].append(row)
        print_row(row)
        del x
        mx.clear_cache()

    out_path = Path(os.path.expanduser(args.json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n書いた: {out_path}", flush=True)
    return 0


def run_with_model(argv, bundle) -> int:
    """読み込み済みの一式で本体を走らせる (`tool` ジョブの入口)。

    規約は `tools/ab_bundle.py` の docstring。この道具がモデルに触るのは
    `capture_x` の `SparseMoeBlock.__call__` の包みだけ (あちらの finally で
    戻る) なので、ここではその 1 点を例外経路の網として控えるだけでよい ---
    部品の計測はどれも純関数で、キャッシュも層の属性も残さない。
    """
    try:
        args, rows, ctx = parse_args(argv)
    except SystemExit as e:  # argparse の --help / 引数エラー
        return int(e.code or 0)

    bad = bundle.mismatch(model_path=args.model, ngram_path=args.ngram)
    if bad:
        print(f"常駐 worker には載せられない: {bad}")
        return NOT_ROUTABLE

    import mlxturbo  # noqa: F401  -- qwen4_exp を _vendor 版に差し替える
    import mlx_lm.models.qwen4_exp as Q

    saved_call = Q.SparseMoeBlock.__call__
    try:
        return run(bundle.model, bundle.tokenizer, args, rows, ctx)
    finally:
        Q.SparseMoeBlock.__call__ = saved_call


def main() -> int:
    args, rows, ctx = parse_args()

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    from mlx_lm import load

    import mlxturbo  # noqa: F401  -- qwen4_exp を _vendor 版に差し替える
    from mlxturbo.runner import enable_default_fusions, set_wired_limit_default

    model, tok = load(os.path.expanduser(args.model))
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[moe-split]")
    # engine を直叩きなので、常駐条件を本番と揃えるためモデル読み込み直後に呼ぶ
    # (tools/gdn_split.py と同じ理由。mlxturbo/runner.py 参照)。
    set_wired_limit_default(log_prefix="[moe-split]")

    rc = run(model, tok, args, rows, ctx)
    if rc:
        return rc

    sys.stdout.flush()
    sys.stderr.flush()
    # 計測ツールなので destructor 待ちに用は無い (gdn_split.py/ple_split.py と同じ理由)。
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
