"""prefill 1 チャンク (幅 2048) の GDN 1 層ぶんを、さらに部品で割る。

`tools/prefill_anatomy.py` は GDN を「36 層まとめて 924〜961 ms (1 チャンク
3.4〜3.8 s の 27%、FLOP 効率 80〜83%)」までしか割っていない
(`docs/research/SESSION-2026-09-02-CATCHUP.md` の「prefill 短文脈の内訳、
8k」)。ここでは**1 層**を、`mlxturbo/_vendor/qwen4_exp.py` の
`GatedDeltaNet.__call__` (1207〜1327 行) の呼び出し順そのままに 5 部品へ割る。

    a  _project_in       in_proj の 4-bit qmm、qkv/z/b/a への分割
    b  conv1d + silu     conv_state との concatenate 込み
    c  split/gate/beta   q/k/v の split・reshape・rms_norm、
                         compute_g (A_log/dt_bias) と sigmoid(b)
    d  再帰スキャン       _gdn_metal (既定 on) の blocked-seq Metal
                         (`mlxturbo/kernels/gdn_blocked_metal.py`)
    e  norm + out_proj   RMSNormGated と out_proj (4-bit qmm)

作法は `prefill_anatomy.py` と同じ (CLAUDE.md): 本物のキャッシュを持ったまま
1 チャンクを実フォワードし、捕まえた入力とキャッシュの退避・復元で部品を
繰り返し測る。**部品和 ≈ 層の壁時計 (層 `__call__` を直接呼んだ値) を確認する。**
med_ms は N=5 回 (温め 1 回捨て) の中央値。

`_project_in` 以降で捕まえるのは 1 層ぶんの実データ (`x`) だけで、他は
その場で非計測のまま 1 回だけ計算して確定させる (moe_parts と同じ流儀)。

対象は最初の GDN 層 (層順位 1) と中央付近の 18 番目 (36 層中)。kv (offset)
は 4k と 16k の 2 点を、`--ctx 17000 --chunk 2048` の既定チャンク境界に
合わせて選ぶ (チャンク 1 の終端 kv=4096、チャンク 7 の終端 kv=16384 --
`prefill_anatomy.py` が「chunk 0 (kv=2048)」「chunk 7 (kv=16384)」と
呼んでいたのと同じ、末尾 kv での命名)。

    tools/biglock.sh .venv/bin/python tools/gdn_split.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep \\
        --ctx 17000 --out bench/results/gdn-split.json
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
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import prefill_anatomy as PA  # noqa: E402  (med_ms/snapshot/restore/pending/定数を共有)

PART_ORDER = ["a", "b", "c", "d", "e"]
PART_LABEL = {
    "a": "(a) _project_in (4本qmm)",
    "b": "(b) conv1d + silu",
    "c": "(c) split/reshape + gate/beta",
    "d": "(d) 再帰スキャン (gdn_metal)",
    "e": "(e) norm(out,z) + out_proj",
}


def gdn_part_bounds(ta, S: int) -> dict:
    """GDN 1 層・1 チャンク (S トークン) ぶんの部品ごとの FLOP/バイト下限。

    数え方は `prefill_anatomy.bounds` の gdn 行 (射影のみを数え、再帰
    スキャンは行列積の天井を当てない) と揃えてある。QBYTE/PEAK_FLOPS/BW は
    `prefill_anatomy` のものをそのまま使う。
    """
    d = ta.hidden_size
    n_k, n_v = ta.linear_num_key_heads, ta.linear_num_value_heads
    dk, dv = ta.linear_key_head_dim, ta.linear_value_head_dim
    K = ta.linear_conv_kernel_dim
    key_dim, value_dim = dk * n_k, dv * n_v
    conv_dim = key_dim * 2 + value_dim
    out = {}

    def put(key, flop, byt):
        f_ms = flop / PA.PEAK_FLOPS * 1000
        b_ms = byt / PA.BW * 1000
        out[key] = (flop, byt, max(f_ms, b_ms), "計算" if f_ms >= b_ms else "帯域")

    # (a) 4 本の qmm (連結射影が有効でも出力幅の合計は変わらない)
    out_total = conv_dim + value_dim + 2 * n_v
    put("a", 2 * S * d * out_total,
        d * out_total * PA.QBYTE + S * d * 2 + S * out_total * 2)

    # (b) depthwise conv (groups=conv_dim) + silu。conv_state (K-1 列) も読む
    put("b", S * conv_dim * K * 2 + S * conv_dim * 4,
        conv_dim * K * 2 + (K - 1) * conv_dim * 2 + S * conv_dim * 2 * 2)

    # (c) split/reshape は無料。rms_norm(q)/rms_norm(k) (~4 flop/elem) と
    #     compute_g (exp x2 + softplus、~6 flop/elem)、sigmoid(b) (~1 flop/elem)
    put("c", 2 * S * key_dim * 4 + S * n_v * 6 + S * n_v * 1,
        2 * S * key_dim * 2 * 2 + S * n_v * 2 * 2 + S * n_v * 2 * 2)

    # (d) 再帰スキャン。prefill_anatomy.bounds の scan 式 (1 層ぶん)。
    #     状態は fp32 [Hv,Dv,Dk] を読んで書く。g/beta は fp32 化される
    flop_d = S * n_v * dk * dv * 4
    byt_d = (S * key_dim * 2 * 2 + S * value_dim * 2
             + S * n_v * 4 * 2 + n_v * dv * dk * 4 * 2 + S * value_dim * 2)
    put("d", flop_d, byt_d)

    # (e) RMSNormGated (rms_norm + sigmoid/silu ゲート + 乗算、~4 flop/elem)
    #     + out_proj (量子化 qmm)
    put("e", S * value_dim * 4 + 2 * S * value_dim * d,
        value_dim * d * PA.QBYTE + S * value_dim * 2 * 2 + S * d * 2)

    return out


def capture_layers(model, ids, cache, ci: int, step: int, layer_set, mx, Q):
    """チャンク ci を実フォワードし、対象 GDN 層の (x, mask) を捕まえて
    キャッシュを元に戻す (`prefill_anatomy.measure` と同じ退避・復元)。
    """
    captured = {}
    # nn.Module は dict のサブクラスで __hash__ を持たない (フック不可)。
    # id() をキーにして識別する。
    target_mods = {id(model.model.layers[i].linear_attn): i for i in layer_set}
    orig_call = Q.GatedDeltaNet.__call__

    def wrapped(self, x, mask, c):
        li = target_mods.get(id(self))
        if li is not None:
            captured[li] = (x, mask)
        return orig_call(self, x, mask, c)

    chunk = ids[:, ci * step: (ci + 1) * step]
    pre = PA.snapshot(cache)
    Q.GatedDeltaNet.__call__ = wrapped
    try:
        out = model.model(chunk, cache=cache)
        mx.eval([out] + PA.pending(cache))
    finally:
        Q.GatedDeltaNet.__call__ = orig_call
    PA.restore(cache, pre)
    mx.clear_cache()
    return captured


def measure_parts(gdn, cache_obj, x, mask, reps: int, mx, nn, compute_g,
                   gated_delta_blocked_seq) -> dict:
    """1 層ぶんの部品を、捕まえた実データで測る。"""
    assert mask is None, (
        "conv_mask は単一系列の prefill では常に None のはず "
        "(Qwen4ExpModel._make_masks 参照)。バッチ計測には未対応")
    B, S, _ = x.shape

    conv_state = cache_obj[0]
    if conv_state is None:
        conv_state = mx.zeros((B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=x.dtype)
    state_in = cache_obj[1]
    inv_scale = gdn.dk ** -0.5

    # -- 前提: 実物の中間値を非計測で 1 回だけ確定させる (moe_parts と同じ流儀)
    mixed_qkv, z, b_, a_ = gdn._project_in(x)
    mx.eval(mixed_qkv, z, b_, a_)
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    conv_out = nn.silu(gdn.conv1d(conv_input))
    mx.eval(conv_out)
    q0, k0, v0 = mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], axis=-1)
    q0 = q0.reshape(B, S, gdn.n_k, gdn.dk)
    k0 = k0.reshape(B, S, gdn.n_k, gdn.dk)
    v0 = v0.reshape(B, S, gdn.n_v, gdn.dv)
    q_n = (inv_scale ** 2) * mx.fast.rms_norm(q0, None, 1e-6)
    k_n = inv_scale * mx.fast.rms_norm(k0, None, 1e-6)
    zz = z.reshape(B, S, gdn.n_v, gdn.dv)
    g = compute_g(gdn.A_log, a_, gdn.dt_bias)
    beta = mx.sigmoid(b_)
    mx.eval(q_n, k_n, v0, zz, g, beta)
    out_scan, state_out = gated_delta_blocked_seq(q_n, k_n, v0, g, beta, state_in)
    mx.eval(out_scan, state_out)

    # -- 部品 (a)〜(e)。どれもキャッシュを書かないベタ関数なので
    #    snapshot/restore は要らない (壁時計だけ層の実 __call__ を使う)
    def part_a():
        return gdn._project_in(x)

    def part_b():
        return nn.silu(gdn.conv1d(mx.concatenate([conv_state, mixed_qkv], axis=1)))

    def part_c():
        qq, kk, vv = mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], axis=-1)
        qq = qq.reshape(B, S, gdn.n_k, gdn.dk)
        kk = kk.reshape(B, S, gdn.n_k, gdn.dk)
        vv = vv.reshape(B, S, gdn.n_v, gdn.dv)
        qq = (inv_scale ** 2) * mx.fast.rms_norm(qq, None, 1e-6)
        kk = inv_scale * mx.fast.rms_norm(kk, None, 1e-6)
        zzz = z.reshape(B, S, gdn.n_v, gdn.dv)
        gg = compute_g(gdn.A_log, a_, gdn.dt_bias)
        bb = mx.sigmoid(b_)
        return qq, kk, vv, zzz, gg, bb

    def part_d():
        return gated_delta_blocked_seq(q_n, k_n, v0, g, beta, state_in)

    def part_e():
        return gdn.out_proj(gdn.norm(out_scan, zz).reshape(B, S, -1))

    def whole():
        """層の実 __call__ を直接叩く (壁時計)。cache を書くので退避・復元する。"""
        st = PA.snapshot([cache_obj])
        r = gdn(x, mask, cache_obj)
        mx.eval(r)
        PA.restore([cache_obj], st)
        return r

    ms = {}
    for key, fn in (("a", part_a), ("b", part_b), ("c", part_c),
                     ("d", part_d), ("e", part_e)):
        ms[key] = PA.med_ms(fn, reps)
        mx.clear_cache()
    layer_ms = PA.med_ms(whole, reps)
    mx.clear_cache()

    parts_sum = sum(ms[k] for k in PART_ORDER)
    gap_pct = (parts_sum - layer_ms) / layer_ms * 100 if layer_ms else 0.0
    return {"S": S, "ms": ms, "parts_sum_ms": parts_sum,
            "layer_wallclock_ms": layer_ms, "gap_pct": gap_pct}


def print_row(row: dict, lb: dict) -> None:
    ms = row["ms"]
    print(f"    {'部品':32s}{'実測 ms':>9s}{'FLOP G':>9s}{'下限 ms':>9s}{'効率':>7s}  律速")
    for k in PART_ORDER:
        flop, byt, low, kind = lb[k]
        eff = low / ms[k] * 100 if ms[k] else 0.0
        print(f"    {PART_LABEL[k]:32s}{ms[k]:9.2f}{flop / 1e9:9.1f}"
              f"{low:9.2f}{eff:6.1f}%  {kind}")
    print(f"    {'部品和':32s}{row['parts_sum_ms']:9.2f}")
    print(f"    {'壁時計 (層 __call__ 直測)':32s}{row['layer_wallclock_ms']:9.2f}")
    print(f"    部品和 - 壁時計 = "
          f"{row['parts_sum_ms'] - row['layer_wallclock_ms']:+.2f} ms ({row['gap_pct']:+.1f}%)")
    if abs(row["gap_pct"]) > 10:
        print("    ** 10% を超えてずれている。見えていない項目があるか、"
              "単体計測が重なりを再現できていない **")
    print(flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--chunk", type=int, default=2048, help="prefill チャンク幅")
    ap.add_argument("--reps", type=int, default=5, help="中央値を取るレップ数")
    ap.add_argument("--kv-points", default="4096,16384",
                     help="狙う kv (チャンク終端) の目安。カンマ区切り")
    ap.add_argument("--layer-rank", default="1,18",
                     help="対象 GDN 層の順位 (1 始まり、線形注意層だけを数える)")
    ap.add_argument("--out", default=str(REPO_ROOT / "bench/results/gdn-split.json"))
    args = ap.parse_args()

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.gated_delta import compute_g

    import mlxturbo  # noqa: F401
    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo.runner import enable_default_fusions, set_wired_limit_default
    from mlxturbo.kernels.gdn_blocked_metal import gated_delta_blocked_seq

    model, tok = load(os.path.expanduser(args.model))
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[gdn-split]")
    # engine を直叩きなので、常駐条件を本番と揃えるためモデル読み込み直後に呼ぶ
    # (tools/prefill_anatomy.py と同じ理由。mlxturbo/runner.py 参照)。
    set_wired_limit_default(log_prefix="[gdn-split]")

    ta = model.args.text
    layer_types = ta.layer_types
    gdn_idxs = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    ranks = [int(r) for r in args.layer_rank.split(",")]
    target_layers = []
    rank_of = {}
    for r in ranks:
        if r < 1 or r > len(gdn_idxs):
            print(f"層順位 {r} は範囲外 (GDN 層は {len(gdn_idxs)} 層)")
            return 1
        li = gdn_idxs[r - 1]
        target_layers.append(li)
        rank_of[li] = r
    print(f"GDN 層総数={len(gdn_idxs)} 対象順位={ranks} -> 層 idx={target_layers}")

    from _bench_text import long_prompts

    body = long_prompts(tok, args.ctx, ["上の文書の要点を 5 つに整理してください。"])[0]
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True))[None]
    n = ids.shape[1]
    step = args.chunk
    n_full = n // step
    if n_full < 2:
        print(f"完全チャンクが {n_full} 本しか無い。--ctx を増やすこと")
        return 1

    kv_targets = [int(x) for x in args.kv_points.split(",")]
    points = sorted({min(range(n_full), key=lambda i: abs((i + 1) * step - t))
                      for t in kv_targets})
    print(f"ctx={n} chunk={step} 完全チャンク={n_full} 計測点(チャンク)={points}"
          f" (kv終端={[(p + 1) * step for p in points]}) reps={args.reps}", flush=True)

    cache = model.make_cache()
    layer_bounds = gdn_part_bounds(ta, step)

    results = {
        "model": args.model, "ctx": n, "chunk": step, "reps": args.reps,
        "gdn_layers_total": len(gdn_idxs), "target_ranks": ranks,
        "target_layer_idx": target_layers,
        "bounds_per_layer": {
            k: {"flop": v[0], "bytes": v[1], "low_ms": v[2], "kind": v[3]}
            for k, v in layer_bounds.items()
        },
        "points": [],
    }

    for ci in range(n_full):
        if ci in points:
            kv_end = (ci + 1) * step
            print(f"\n[chunk {ci}] kv終端={kv_end}", flush=True)
            captured = capture_layers(model, ids, cache, ci, step, target_layers, mx, Q)
            for li in target_layers:
                x, mask = captured[li]
                print(f"  GDN 層 #{rank_of[li]} (idx={li})", flush=True)
                row = measure_parts(
                    model.model.layers[li].linear_attn, cache[li], x, mask,
                    args.reps, mx, nn, compute_g, gated_delta_blocked_seq)
                row.update({"chunk": ci, "kv_end": kv_end,
                            "layer_idx": li, "layer_rank": rank_of[li]})
                results["points"].append(row)
                print_row(row, layer_bounds)
        mx.eval([model.model(ids[:, ci * step: (ci + 1) * step], cache=cache)]
                + PA.pending(cache))
        mx.clear_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n書いた: {out_path}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    # 計測ツールなので destructor 待ちに用は無い (prefill_anatomy.py と同じ理由)。
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
