"""fused:1 の正しさを **本番の重みと本番の routing** で確かめる (`tool` ジョブ)。

合成 (E=32、`scratchpad/moe_dec_fused_correct.py`) では自前が素より
逆量子化参照に近かった (自前 max 2.1〜2.3% 対 素 6.6〜6.9%)。それが本番の
E=512・実際の x・実際の top-k でも成り立つかを、48 層それぞれで測る。

**反転条件: どこかの層で「自前 対 参照」が「素 対 参照」より大きければ、
その層で自前は素より真値から遠い = 採らない。**

## 置き場所

`tools/moe_decode_fused_ref_model.py` (名前を verify_* にしないのは、biglock が
verify_* をモデルを読まない micro 段に振るため)。`tools/*.py` に `run_with_model` を持つファイルを
**増やすと** `ab_daemon.code_fingerprint()` の見張りの集合が変わり、
常駐 worker が 98GB を読み直す (`fingerprint_diff` は新しいキーも差分に
数える)。列が空いてから移すこと。

## 走らせ方

    FASTMLX_NGRAM_DISK=1 tools/biglock.sh .venv/bin/python \
        tools/moe_decode_fused_ref_model.py \
        --model ~/models/ddalcu-mlxlm-head4 --ngram ~/models/ddalcu-ngram

`tools/ab_submit.py` の `TOOL_JOBS` に載っていれば worker の中で
in-process に走る (98GB の読み直し無し)。載っていなければ `--` 経由で
`tool` ジョブとして投げる。

## 何を測るか

1. 実プロンプト 1 本を prefill し、続く S トークンの forward を S∈{1,3,6} で流す。
2. その forward の間だけ `SwitchGLU.__call__` を包み、層ごとに
   **(x, indices)** を控える (値は計算せず素通し。捕まえるのは
   `SparseMoeBlock` が素の経路 `(switch_mlp(x, idx) * w).sum(-2)` で呼ぶ
   ときの引数そのもの。decode 幅は `_combine_fold_min_s` (既定 64) に
   届かないので必ずこちらを通る)。
3. 控えた (x, idx) ごとに 3 つを出す:
   - **素**: `disable_moe_decode_fused()` した状態の `SwitchGLU.__call__`
     (= 本番の統合ディスパッチ = `_gather_sort` + `gather_qmm` x3)
   - **自前**: `MLXTURBO_MOE_DECODE_FUSED=1` + `enable_moe_decode_fused()`
   - **参照**: 選ばれた専門家の行だけ逆量子化して fp32 で対ごとに計算
4. 物差しは `scratchpad/moe_dec_fused_correct.py` と同じ
   `max|a-b| / max|b|` (と平均版)。数字が直接くらべられる。

## 費用の見積もり

参照は対ごとに 3 本の行列を逆量子化する。1 専門家 1 本 = 640x2560 fp32 =
6.5 MB、対あたり 3 本 = 20 MB。S=6 なら 60 対 x 48 層 = 2880 対 = 56 GB の
帯域 = 400 GB/s で 0.14 秒。**全 48 層でも十分に安い**ので代表層に絞らない
(`--layers` で絞れるようにはしてある)。一時領域は対ごとに eval して捨てる。
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

import mlx.core as mx

NOT_ROUTABLE = 64

# S を変えて流す幅。1 = draft、3 = 既定 depth 2 の検証幅、6 = 行数ゲート 8 の
# 手前 (バッチ B=2 x depth 2 に相当する幅でもある)。
DEFAULT_WIDTHS = (1, 3, 4)  # 上限は fused._MOE_DISPATCH_DEC_FUSED_MAX_ROWS (既定 4)。超える幅は発火せず skip 扱い

# 実プロンプト。decode_ab の SHORT_PROMPTS[1] と同じ文
# (池を跨いだ比較をしないので 1 本で足りる)。
DEFAULT_PROMPT = (
    "Explain why speculative decoding helps when decoding is dispatch bound."
)


# ---------------------------------------------------------------- 参照

def _deq(lin, e: int):
    """専門家 `e` の重みだけ逆量子化して fp32 に。

    `SwitchLinear` の weight は (E, out, in/pack)。`mx.dequantize` に
    渡すのは 1 専門家ぶんの 2 次元 (out, in/pack) で足りるので、
    E=512 を丸ごと展開しない。
    """
    w = lin.weight[e]
    s = lin.scales[e]
    b = lin.biases[e]
    return mx.dequantize(w, s, b, group_size=lin.group_size, bits=lin.bits,
                         mode=getattr(lin, "mode", "affine")).astype(mx.float32)


def reference(mod, x2d, idx2d):
    """逆量子化 fp32 の参照。x2d (rows, K)、idx2d (rows, topk) -> (rows, topk, K)。

    積和は fp32、h の丸めもしない (自前も素もここから離れる側)。
    """
    gp, up, dp = mod.gate_proj, mod.up_proj, mod.down_proj
    rows, topk = idx2d.shape
    xf = x2d.astype(mx.float32)
    out_rows = []
    for r in range(rows):
        per = []
        for t in range(topk):
            e = int(idx2d[r, t])
            wg, wu, wd = _deq(gp, e), _deq(up, e), _deq(dp, e)
            g = xf[r] @ wg.T
            u = xf[r] @ wu.T
            h = (g * mx.sigmoid(g)) * u
            y = h @ wd.T
            mx.eval(y)                  # 対ごとに畳んで一時領域を返す
            per.append(y)
            del wg, wu, wd, g, u, h
        out_rows.append(mx.stack(per))
    return mx.stack(out_rows)


def rel(a, b):
    """(max 相対, 平均 相対)。`scratchpad/moe_dec_fused_correct.py` と同じ式。"""
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    d = mx.abs(a - b)
    return (float(mx.max(d) / mx.maximum(mx.max(mx.abs(b)), 1e-6)),
            float(mx.mean(d) / mx.maximum(mx.mean(mx.abs(b)), 1e-6)))


# ---------------------------------------------------------------- 捕獲

def capture(model, ids, width: int):
    """`ids` を prefill し、末尾 `width` トークンの forward で (x, idx) を控える。

    返り値は [(層番号, module, x, indices), ...]。値は素通しなので、この
    forward 自体は本番とビット一致で走る。
    """
    import mlx_lm.models.switch_layers as SL

    recs = []
    stock = SL.SwitchGLU.__call__
    order = {"n": 0}

    def spy(self, x, indices):
        recs.append((order["n"], self, x, indices))
        order["n"] += 1
        return stock(self, x, indices)

    caches = model.make_cache()
    head, tail = ids[:, :-width], ids[:, -width:]
    if head.shape[1] > 0:
        mx.eval(model(head, cache=caches))
    SL.SwitchGLU.__call__ = spy
    try:
        mx.eval(model(tail, cache=caches))
    finally:
        SL.SwitchGLU.__call__ = stock
    # x/indices は借り物のグラフを掴んでいるので、ここで実体化して切り離す
    out = []
    for n, mod, x, idx in recs:
        x = mx.array(x)
        idx = mx.array(idx)
        mx.eval(x, idx)
        out.append((n, mod, x, idx))
    return out


def _call_variant(mod, x, idx, fused_on: bool):
    """本番の統合ディスパッチをそのまま呼ぶ (自前 on/off だけ切り替える)。"""
    import mlx_lm.models.switch_layers as SL

    from mlxturbo import fused

    fused.disable_moe_decode_fused()
    if fused_on:
        os.environ["MLXTURBO_MOE_DECODE_FUSED"] = "1"
        fused.enable_moe_decode_fused()
    try:
        out = SL.SwitchGLU.__call__(mod, x, idx)
        mx.eval(out)
        return out
    finally:
        fused.disable_moe_decode_fused()
        if fused_on:
            os.environ.pop("MLXTURBO_MOE_DECODE_FUSED", None)


# ---------------------------------------------------------------- 本体

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--widths", default=",".join(map(str, DEFAULT_WIDTHS)),
                    help="forward の幅 S をカンマ区切りで (既定 1,3,6)")
    ap.add_argument("--layers", default=None,
                    help="調べる層をカンマ区切りで絞る (既定は捕まえた全部)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="自前 対 参照 の max 相対誤差の上限 (既定 1e-2)")
    ap.add_argument("--out", default=None, help="結果 JSON の書き出し先")
    return ap


def run_with_model(argv, bundle) -> int:
    """読み込み済みの一式で走らせる (`tool` ジョブの入口、規約は tools/ab_bundle.py)。

    借り物に触るのは `SwitchGLU.__call__` の差し替えと
    `fused.enable/disable_moe_decode_fused` だけで、どちらも finally で戻す。
    """
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)

    bad = bundle.mismatch(model_path=args.model, ngram_path=args.ngram)
    if bad:
        print(f"常駐 worker には載せられない: {bad}")
        return NOT_ROUTABLE

    from mlxturbo import fused
    from mlxturbo.kernels import _fire

    model, tok = bundle.model, bundle.tokenizer
    ids = mx.array(tok.encode(args.prompt))[None]
    widths = [int(v) for v in args.widths.split(",") if v.strip()]
    want = ({int(v) for v in args.layers.split(",")} if args.layers else None)

    rows_out = []
    worst = None          # (比, 説明) 自前/素 の比が一番悪かったところ
    n_bad = 0
    ok = True

    for S in widths:
        if ids.shape[1] <= S:
            print(f"S={S}: プロンプトが短すぎる (トークン {ids.shape[1]})")
            ok = False
            continue
        recs = capture(model, ids, S)
        if not recs:
            print(f"S={S}: SwitchGLU の呼び出しを 1 つも捕まえられなかった")
            ok = False
            continue
        print(f"\n--- S={S} (捕まえた MoE 呼び出し {len(recs)} 個) ---")
        print(f"{'#':>4} {'rows':>5} {'topk':>5} "
              f"{'自前対参照':>11} {'素対参照':>10} {'比':>7} {'発火':>5}")
        for n, mod, x, idx in recs:
            if want is not None and n not in want:
                continue
            topk = idx.shape[-1]
            rows = idx.size // topk
            x2d = x.reshape(-1, x.shape[-1])
            idx2d = idx.reshape(-1, topk)

            plain = _call_variant(mod, x, idx, fused_on=False)
            _fire.reset()
            got = _call_variant(mod, x, idx, fused_on=True)
            fired = _fire.snapshot().get("moe_decode_fused", 0)
            ref = reference(mod, x2d, idx2d).reshape(plain.shape)

            fm, fa = rel(got, ref)
            pm, pa = rel(plain, ref)
            gm, _ = rel(got, plain)
            ratio = fm / pm if pm > 0 else float("inf")
            # **反転条件**: 自前の方が参照から遠い
            reversed_ = fm > pm
            # 行数が上限 (fused._MOE_DISPATCH_DEC_FUSED_MAX_ROWS) を超える幅は
            # 設計どおり発火しない (素に落ちる)。判定からは外して skip と印す
            from mlxturbo import fused as _fused
            skipped = fired == 0 and rows > _fused._MOE_DISPATCH_DEC_FUSED_MAX_ROWS
            good = skipped or ((fired == 1) and (fm < args.tol) and not reversed_)
            n_bad += (not good)
            ok &= good
            if not skipped and (worst is None or ratio > worst[0]):
                worst = (ratio, f"S={S} 呼び出し #{n}")
            rows_out.append(dict(S=S, call=n, rows=rows, topk=topk,
                                 fused_vs_ref_max=fm, fused_vs_ref_mean=fa,
                                 plain_vs_ref_max=pm, plain_vs_ref_mean=pa,
                                 fused_vs_plain_max=gm, ratio=ratio,
                                 fired=fired, reversed=reversed_,
                                 skipped=skipped))
            flag = ("  (skip: rows > MAX_ROWS)" if skipped else
                    "  <== 反転" if reversed_ else ("" if good else "  <== NG"))
            print(f"{n:>4} {rows:>5} {topk:>5} {fm:>11.3e} {pm:>10.3e} "
                  f"{ratio:>7.3f} {fired:>5}{flag}")
            del plain, got, ref

    if rows_out:
        rs = [r["ratio"] for r in rows_out if not r["skipped"]]
        n_skip = sum(1 for r in rows_out if r["skipped"])
        print(f"\n=== まとめ ===")
        print(f"  調べた呼び出し {len(rows_out)} 個 (skip {n_skip}: rows > MAX_ROWS)、NG {n_bad} 個")
        if not rs:
            rs = [float("nan")]; worst = worst or (float("nan"), "-")
        print(f"  自前/素 の比: 最小 {min(rs):.3f}  中央 "
              f"{sorted(rs)[len(rs)//2]:.3f}  最大 {max(rs):.3f}"
              f"  (最大は {worst[1]})")
        print(f"  比が 1 を超えた呼び出し (自前の方が参照から遠い): "
              f"{sum(1 for r in rows_out if r['reversed'])} 個")
        print("PASS" if ok else "FAIL")
    if args.out:
        Path(args.out).write_text(json.dumps(rows_out, ensure_ascii=False, indent=1))
        print(f"書き出し: {args.out}")
    return 0 if ok else 1


def main(argv=None) -> int:
    """worker を使わない単独実行 (98GB を自分で読む)。列が空いているときだけ。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import ab_bundle

    bundle = ab_bundle.load_bundle(model=args.model, ngram=args.ngram,
                                   load_mtp=False)
    return run_with_model(argv, bundle)


if __name__ == "__main__":
    raise SystemExit(main())
