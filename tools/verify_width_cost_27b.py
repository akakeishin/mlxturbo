"""27B (qwen3_5) の検証幅 S の費用を部品ごとに割る micro。

的: `bench/results/round-anatomy-27b-b2-0904.json` の
「trunk forward + lm_head + eval が S=1 43〜45 / S=2 48 / S=4 74〜76 ms」---
S=1→2 が +4.7 ms なのに S=2→4 が +27 ms という**超線形**の出所を 1 つに絞る。

合成モデルではなく**実機の 27B を読んで**、本番と同じ層・同じ重み・同じ
キャッシュ状態で測る。部品ごとの時間は「1 フォワードぶん」に揃える
(GDN 部品は 48 層ぶん、attention 部品は 16 層ぶん、MLP は 64 層ぶんを
まとめて 1 回 `mx.eval` する) ので、**部品の和 ≈ 全体**が確かめられる
(`CLAUDE.md` の「部品和 ≈ 壁時計」)。

## 冷やし方

1 層の重みだけを 200 回読む温の micro は当てにならない (`CLAUDE.md`)。
ここは部品ごとに**層を全部巡回**するので、1 回の計測で GDN 射影 2.8 GB /
MLP 9.6 GB を読む。温キャッシュに載らない量なので本番の DRAM 条件に近い。

## 走らせ方

    BIGLOCK_NO_WORKER=1 BIGLOCK_PRIO=1 tools/biglock.sh \\
        .venv/bin/python tools/verify_width_cost_27b.py \\
        --model ~/models/qwen38-27b-4bit \\
        --s-list 1,2,3,4,6,8 --reps 5 \\
        --out bench/results/width-cost-27b-0904.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


# ------------------------------------------------------------------ 計測の器

def _median_ms(fn, reps: int, setup=None) -> dict:
    """`fn` を reps+1 回まわして中央値・最小値 (ms) を返す (1 本目は捨てる)。

    `fn` は最後に `mx.eval` まで済ませて返すこと。`setup` はキャッシュを
    戻すなどの前処理で、計測の外で走る。
    """
    import mlx.core as mx

    ts = []
    for r in range(reps + 1):
        if setup is not None:
            setup()
        mx.synchronize()
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1000.0
        if r > 0:
            ts.append(dt)
    return {"ms": statistics.median(ts), "ms_min": min(ts), "n": len(ts)}


# ------------------------------------------------------------------ 部品

class WidthProbe:
    """1 つの S について、フォワードの部品ごとの費用を測る。"""

    def __init__(self, eng, caches, snap, restore, reps: int):
        import mlx.core as mx

        self.mx = mx
        self.eng = eng
        self.inner = eng.inner
        self.caches = caches
        self.snap = snap
        self.restore = restore
        self.reps = reps
        self.layers = list(self.inner.layers)
        self.lin = [(i, la) for i, la in enumerate(self.layers) if la.is_linear]
        self.att = [(i, la) for i, la in enumerate(self.layers) if not la.is_linear]

    # -- 補助 ------------------------------------------------------------

    def _reset(self):
        for c, rec in zip(self.caches, self.snap):
            self.restore(c, rec)

    def _masks(self, x):
        from mlxturbo.spec import create_attention_mask, create_ssm_mask

        fa = create_attention_mask(x, self.caches[self.inner.fa_idx])
        ssm = create_ssm_mask(x, self.caches[self.inner.ssm_idx])
        return fa, ssm

    def _hidden(self, S):
        """S 行の埋め込み (評価済み)。"""
        mx = self.mx
        window = mx.array([1000 + i for i in range(S)], dtype=mx.int32)
        x = self.inner.embed_tokens(window[None])
        mx.eval(x)
        return window, x

    # -- 測定 ------------------------------------------------------------

    def run(self, S: int) -> dict:
        mx = self.mx
        eng = self.eng
        window, x = self._hidden(S)
        self._reset()
        fa_mask, ssm_mask = self._masks(x)
        mx.eval([m for m in (fa_mask, ssm_mask) if isinstance(m, mx.array)])
        rows: dict = {}

        # --- 全体 (本番の verify と同じ形) ---------------------------------
        def whole_capture():
            hs, _sink = eng._hidden_forward(window, self.caches, capture=True)
            lg = eng._head(hs, self.inner.norm)
            mx.eval(mx.argmax(lg, axis=-1))

        rows["whole_capture"] = _median_ms(whole_capture, self.reps, self._reset)

        def whole_plain():
            hs, _ = eng._hidden_forward(window, self.caches, capture=False)
            lg = eng._head(hs, self.inner.norm)
            mx.eval(mx.argmax(lg, axis=-1))

        rows["whole_plain"] = _median_ms(whole_plain, self.reps, self._reset)

        # --- 層の塊 --------------------------------------------------------
        # 同じ h を全層に入れる (層をまたぐ依存を切って部品の費用だけ見る)。
        # cache は毎回戻すので状態は汚れない。
        def gdn_layers_capture():
            h = x
            sink: list = []
            for i, la in self.lin:
                h = eng._linear_capture(la, h, self.caches[i], sink, ssm_mask)
            mx.eval(h)

        rows["gdn_layers_capture"] = _median_ms(
            gdn_layers_capture, self.reps, self._reset)

        def gdn_layers_plain():
            h = x
            for i, la in self.lin:
                h = la(h, mask=ssm_mask, cache=self.caches[i])
            mx.eval(h)

        rows["gdn_layers_plain"] = _median_ms(
            gdn_layers_plain, self.reps, self._reset)

        def attn_layers():
            h = x
            for i, la in self.att:
                h = la(h, mask=fa_mask, cache=self.caches[i])
            mx.eval(h)

        rows["attn_layers"] = _median_ms(attn_layers, self.reps, self._reset)

        # --- MLP 64 層ぶん -------------------------------------------------
        def mlp_all():
            outs = []
            for la in self.layers:
                outs.append(la.mlp(la.post_attention_layernorm(x)))
            mx.eval(outs[-1], *outs[:-1])

        rows["mlp_all"] = _median_ms(mlp_all, self.reps)

        # --- lm_head -------------------------------------------------------
        def head_only():
            lg = self.eng._head(x, self.inner.norm)
            mx.eval(mx.argmax(lg, axis=-1))

        rows["lm_head"] = _median_ms(head_only, self.reps)

        # --- GDN の内訳 (48 層ぶん) ---------------------------------------
        rows.update(self._gdn_parts(x, ssm_mask))
        # --- attention の内訳 (16 層ぶん) ---------------------------------
        rows.update(self._attn_parts(x, fa_mask))
        return rows

    # -- GDN の内訳 -------------------------------------------------------

    def _gdn_parts(self, x, ssm_mask) -> dict:
        """`_linear_capture` の中身を 4 つに割って 48 層ぶんまとめて測る。

        `mlxturbo/spec.py:_linear_capture` の写しだが、**時間の帰属だけが
        目的**で本番はこちらを呼ばない。段の切れ目は
        (1) 射影 (2) conv + rms (3) 再帰 (4) 出力 norm + out_proj。
        """
        mx = self.mx
        import mlx.nn as nn
        from mlx_lm.models.gated_delta import gated_delta_kernel

        from mlxturbo.kernels.gated_delta_states import (
            gated_delta_update_with_states,
        )
        from mlxturbo._mlx_compat import compute_g

        out: dict = {}
        gs = [la.linear_attn for _i, la in self.lin]

        # (0) 入力 norm
        def part_ln():
            ys = [la.input_layernorm(x) for _i, la in self.lin]
            mx.eval(*ys)

        out["gdn_input_layernorm"] = _median_ms(part_ln, self.reps)

        xin = [la.input_layernorm(x) for _i, la in self.lin]
        mx.eval(*xin)

        # (1) 射影
        def part_proj():
            ys = []
            for g, xi in zip(gs, xin):
                ys.append(g.in_proj_qkv(xi))
                ys.append(g.in_proj_z(xi))
                ys.append(g.in_proj_b(xi))
                ys.append(g.in_proj_a(xi))
            mx.eval(*ys)

        out["gdn_proj"] = _median_ms(part_proj, self.reps)

        # 射影の結果を控える (以降の段の入力)
        B, S, _ = x.shape
        qkvs, zs, bs, as_ = [], [], [], []
        for g, xi in zip(gs, xin):
            qkvs.append(g.in_proj_qkv(xi))
            zs.append(g.in_proj_z(xi).reshape(B, S, g.num_v_heads, g.head_v_dim))
            bs.append(g.in_proj_b(xi))
            as_.append(g.in_proj_a(xi))
        mx.eval(*qkvs, *zs, *bs, *as_)

        conv_states = []
        for (i, _la), g in zip(self.lin, gs):
            cs = self.caches[i][0]
            if cs is None:
                cs = mx.zeros((B, g.conv_kernel_size - 1, g.conv_dim), dtype=x.dtype)
            conv_states.append(cs)
        mx.eval(*conv_states)

        # (2) conv + split + rms
        def _conv_stage(g, qkv, cs):
            conv_input = mx.concatenate([cs, qkv], axis=1)
            conv_out = nn.silu(g.conv1d(conv_input))
            q, k, v = [
                t.reshape(B, S, h_, d)
                for t, h_, d in zip(
                    mx.split(conv_out, [g.key_dim, 2 * g.key_dim], -1),
                    [g.num_k_heads, g.num_k_heads, g.num_v_heads],
                    [g.head_k_dim, g.head_k_dim, g.head_v_dim],
                )
            ]
            inv = k.shape[-1] ** -0.5
            q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv * mx.fast.rms_norm(k, None, 1e-6)
            return q, k, v, conv_input

        def part_conv():
            ys = []
            for g, qkv, cs in zip(gs, qkvs, conv_states):
                q, k, v, ci = _conv_stage(g, qkv, cs)
                ys += [q, k, v, ci]
            mx.eval(*ys)

        out["gdn_conv_rms"] = _median_ms(part_conv, self.reps)

        qs, ks, vs = [], [], []
        for g, qkv, cs in zip(gs, qkvs, conv_states):
            q, k, v, _ci = _conv_stage(g, qkv, cs)
            qs.append(q)
            ks.append(k)
            vs.append(v)
        mx.eval(*qs, *ks, *vs)

        # (3-a) 再帰 (capture 版 = states_all を全位置ぶん出す)
        states_in = []
        for (i, _la), g in zip(self.lin, gs):
            st = self.caches[i][1]
            if st is None:
                st = mx.zeros((B, g.num_v_heads, g.head_v_dim, g.head_k_dim),
                              dtype=mx.float32)
            states_in.append(st)
        mx.eval(*states_in)

        def part_rec_states():
            ys = []
            for g, q, k, v, a, b, st in zip(gs, qs, ks, vs, as_, bs, states_in):
                o, sa = gated_delta_update_with_states(
                    q, k, v, a, b, g.A_log, g.dt_bias, st, None)
                ys += [o, sa]
            mx.eval(*ys)

        out["gdn_recur_states"] = _median_ms(part_rec_states, self.reps)

        # (3-b) 再帰 (素 = 最終状態だけ。capture の代金の対照)
        def part_rec_plain():
            ys = []
            for g, q, k, v, a, b, st in zip(gs, qs, ks, vs, as_, bs, states_in):
                beta = mx.sigmoid(b)
                gg = compute_g(g.A_log, a, g.dt_bias)
                o, so = gated_delta_kernel(q, k, v, gg, beta, st, None)
                ys += [o, so]
            mx.eval(*ys)

        out["gdn_recur_plain"] = _median_ms(part_rec_plain, self.reps)

        # (4) 出力 norm + out_proj
        outs = []
        for g, q, k, v, a, b, st in zip(gs, qs, ks, vs, as_, bs, states_in):
            o, _sa = gated_delta_update_with_states(
                q, k, v, a, b, g.A_log, g.dt_bias, st, None)
            outs.append(o)
        mx.eval(*outs)

        def part_outproj():
            ys = []
            for g, o, z in zip(gs, outs, zs):
                ys.append(g.out_proj(g.norm(o, z).reshape(B, S, -1)))
            mx.eval(*ys)

        out["gdn_norm_outproj"] = _median_ms(part_outproj, self.reps)

        # (5) states_all の割り付け + 書き出しだけ (帯域の下限)
        n_v, d_v, d_k = gs[0].num_v_heads, gs[0].head_v_dim, gs[0].head_k_dim
        out["_states_bytes_per_fwd"] = len(gs) * B * S * n_v * d_v * d_k * 4
        return out

    # -- attention の内訳 -------------------------------------------------

    def _attn_parts(self, x, fa_mask) -> dict:
        mx = self.mx
        out: dict = {}
        atts = [la.self_attn for _i, la in self.att]

        def part_attn_only():
            ys = []
            for (i, la), at in zip(self.att, atts):
                ys.append(at(la.input_layernorm(x), fa_mask, self.caches[i]))
            mx.eval(*ys)

        out["attn_self_attn"] = _median_ms(part_attn_only, self.reps, self._reset)
        return out


# ------------------------------------------------------------------ main

def build_parser():
    ap = argparse.ArgumentParser(
        description="27B (qwen3_5) の検証幅 S の費用を部品ごとに割る")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=24,
                    help="キャッシュに積む文脈長 (短文脈の round に合わせる)")
    ap.add_argument("--s-list", default="1,2,3,4,6,8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("MLXTURBO_QUIET", "1")

    import mlx.core as mx
    from decode_ab_generic import _restore_cache, _snap_cache, load_model

    la = types.SimpleNamespace(
        model=args.model, mtp=None, mtp_bits=4, no_mtp=True)
    print("[width] モデル読み込み...", flush=True)
    t0 = time.perf_counter()
    model, tok, eng, _eos, _guard = load_model(la)
    print(f"[width] 読み込み {time.perf_counter() - t0:.1f}s", flush=True)

    caches = eng.text.make_cache()
    ids = [10 + (i * 7) % 1000 for i in range(max(args.ctx, 2))]
    h = eng._hidden_forward(mx.array(ids, dtype=mx.int32), caches, capture=False)[0]
    mx.eval(h)
    snap = [_snap_cache(c) for c in caches]

    s_list = [int(s) for s in args.s_list.split(",") if s.strip()]
    probe = WidthProbe(eng, caches, snap, _restore_cache, args.reps)

    # 読み込み直後の 1 本目は +7〜9% 遅い (CLAUDE.md)。捨て走行を 1 回。
    print("[width] burn-in...", flush=True)
    probe.run(2)

    result = {"model": args.model, "ctx": args.ctx, "reps": args.reps, "S": {}}
    for S in s_list:
        print(f"[width] S={S} ...", flush=True)
        result["S"][str(S)] = probe.run(S)

    # ---- 表 -------------------------------------------------------------
    keys = [k for k in result["S"][str(s_list[0])] if not k.startswith("_")]
    base = str(s_list[0])
    print()
    hdr = f"{'部品':<22}" + "".join(f"{('S=' + str(s)):>12}" for s in s_list)
    print(hdr)
    print("-" * len(hdr))
    for k in keys:
        row = f"{k:<22}"
        for s in s_list:
            row += f"{result['S'][str(s)][k]['ms']:>12.2f}"
        print(row)
    print()
    print(f"{'部品 (S=' + base + ' 比)':<22}" +
          "".join(f"{('S=' + str(s)):>12}" for s in s_list))
    for k in keys:
        b = result["S"][base][k]["ms"]
        row = f"{k:<22}"
        for s in s_list:
            row += f"{result['S'][str(s)][k]['ms'] / max(b, 1e-9):>12.2f}"
        print(row)

    # 部品和 ≈ 全体 の検査
    print()
    for s in s_list:
        r = result["S"][str(s)]
        # `attn_layers` は DecoderLayer 丸ごと (= MLP 込み) なので和には
        # 入れない。attention 側は `attn_self_attn` (input_layernorm 込み)、
        # MLP は 64 層ぶんの `mlp_all` で 1 度だけ数える。
        parts = (r["gdn_input_layernorm"]["ms"] + r["gdn_proj"]["ms"]
                 + r["gdn_conv_rms"]["ms"] + r["gdn_recur_states"]["ms"]
                 + r["gdn_norm_outproj"]["ms"] + r["attn_self_attn"]["ms"]
                 + r["mlp_all"]["ms"] + r["lm_head"]["ms"])
        whole = r["whole_capture"]["ms"]
        print(f"  S={s}: 部品和 {parts:.1f} ms / 全体 {whole:.1f} ms "
              f"({parts / max(whole, 1e-9):.2f}x)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"\n書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
