"""MLX 組み込みカーネルの「天井達成率」を、うちの使う形だけで洗う冷 micro。

## 何のための道具か

`fast_qmm` の的 (MLX の `quantized_matmul` が M=1 で 400 GB/s なのに M=2〜8 で
209 GB/s に落ちる) と**同じ型**の穴が他にもあるはず、という問いに答える。
形 (K / N / 行数) ごとに us を測り、

  * 帯域換算 GB/s = (重み + 入力 + 出力) バイト / 時間
  * 計算換算 TFLOPS = 2·M·K·N / 時間

を出して、M3 Max の天井 (帯域 409.6 GB/s、量子化 dense GEMM の実測上限
11.2 TFLOPS、bf16 GEMM 12.85 TFLOPS) と比べた**達成率**を並べる。
達成率が低い形 × 1 round / 1 prefill での占有が大きい形、が的になる。

**判断はしない。**一覧を出すだけ。採否は in-model A/B (親)。

## 計測の作法 (docs/CLAUDE.md)

- **冷やす**: 重み / KV を `--mb` MB ぶん (既定 140) 用意して巡回する。
  温キャッシュ (1 組を使い回す) の絶対値は信用しない。
- **burn-in**: プロセス起動直後の最初の計測は +7〜9% 遅い。捨てる走行を入れる。
- **本数の取り方**: 依存の無い同形カーネルを `iters` 本、1 度も eval せずに
  組んでから 1 回だけ `mx.eval` し、eval 時間を本数で割る
  (`tools/kernel_chain_cost.py` と同じ規約)。Python 側の組み立て時間は
  別に出す (`build_us`)。1 本ずつ eval すると host の往復 (数十 us) が
  小さいカーネルを飲み込むため。
  **含意**: 独立な本は GPU 上で重なりうるので、これは「詰めて投げたときの
  実効スループット」であって単発レイテンシではない。帯域律速の形では
  重なっても帯域以上は出ないので GB/s の解釈は変わらない。小さい
  elementwise では「重なった状態の実効値」= in-model に近い方の値になる。

## 使い方

    tools/biglock.sh .venv/bin/python tools/ceiling_audit_micro.py \
        --sections qmm --out bench/results/ceiling-qmm.json

    tools/biglock.sh .venv/bin/python tools/ceiling_audit_micro.py \
        --sections all --out bench/results/ceiling-all.json

節 (`--sections`):
  qmm   : `mx.quantized_matmul` を本番の (K, N) × 行数で掃引 (両モデル)
  gqmm  : `mx.gather_qmm` (MoE) を専門家あたりの行数で掃引
  sdpa  : `mx.fast.scaled_dot_product_attention` (head_dim 256) の S × kv
  norm  : `mx.fast.rms_norm` / `layer_norm` / `rope` の小行数の固定費
  conv  : GDN の depthwise conv1d
  idx   : argsort / argpartition / take_along_axis / scatter_add (MoE routing、QSA)
  red   : softmax / logsumexp / argmax の語彙長 (248320) と 512
  embed : 量子化 embedding の 1 行 gather
  dense : bf16 の細い matmul (HC inject N=4) と QSA のブロックスコア einsum

モデルは読まない (合成テンソルのみ)。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------- 天井の定数
PEAK_BW = 409.6e9        # M3 Max のピーク帯域 (B/s)。tools/decode_glue_probe.py と同じ
PEAK_TF_QMM = 11.2e12    # 量子化 dense GEMM の実測上限 (CATCHUP 2026-09-03 21:45)
PEAK_TF_BF16 = 12.85e12  # bf16 GEMM の天井 (CATCHUP 2026-09-03 22:50)

DT = mx.bfloat16
GS = 64

# ------------------------------------------------------------------ 計測の芯


def _stats(samples):
    return {
        "us": round(statistics.median(samples), 3),
        "us_min": round(min(samples), 3),
        "spread": round((max(samples) - min(samples)) / max(statistics.median(samples), 1e-9), 3),
    }


def bench(make, n_sets: int, iters: int, reps: int = 5, warm: int = 2):
    """`make(i)` が返す遅延配列を `iters` 本組んで 1 回だけ eval する。

    戻り値 (中央値の us/call, build の us/call)。`i` は 0..n_sets-1 を巡回する
    (重みや KV の組を切り替えて冷キャッシュを作る)。
    """
    for _ in range(warm):
        mx.eval(make(0))
    ev, bu = [], []
    it = 0
    for _ in range(reps):
        outs = []
        t0 = time.perf_counter()
        for _ in range(iters):
            outs.append(make(it % n_sets))
            it += 1
        t1 = time.perf_counter()
        mx.eval(*outs)
        t2 = time.perf_counter()
        ev.append((t2 - t1) / iters * 1e6)
        bu.append((t1 - t0) / iters * 1e6)
        del outs
    return _stats(ev), round(statistics.median(bu), 3)


def _auto_iters(pilot_us: float, out_bytes: int, target_ms: float = 25.0,
                mem_cap: int = 1_500_000_000, hi: int = 256) -> int:
    """総 eval が `target_ms` 前後、かつ出力の総バイトが `mem_cap` 以内になる本数。"""
    n = int(max(2, min(hi, round(target_ms * 1000.0 / max(pilot_us, 0.5)))))
    if out_bytes > 0:
        n = max(2, min(n, mem_cap // max(out_bytes, 1)))
    return int(n)


def measure(make, n_sets: int, out_bytes: int, reps: int = 5, hi: int = 256):
    """パイロット 1 発で本数を決めてから本計測する。"""
    p, _ = bench(make, n_sets, 3, reps=2, warm=1)
    iters = _auto_iters(p["us"], out_bytes, hi=hi)
    st, build = bench(make, n_sets, iters, reps=reps)
    st["iters"] = iters
    st["build_us"] = build
    return st


def rates(us: float, rd: int, wr: int, flops: float, peak_tf: float = PEAK_TF_QMM):
    """バイトと FLOP から達成率を出す。`rd`/`wr` は 1 呼び出しあたりのバイト。"""
    t = us * 1e-6
    gbs = (rd + wr) / t / 1e9
    tf = flops / t / 1e12 if flops else 0.0
    out = {
        "GBs": round(gbs, 1),
        "bw_pct": round(gbs * 1e9 / PEAK_BW * 100, 1),
        "bytes": int(rd + wr),
    }
    if flops:
        out["TFLOPS"] = round(tf, 2)
        out["tf_pct"] = round(tf * 1e12 / peak_tf * 100, 1)
        out["bound"] = "compute" if out["tf_pct"] >= out["bw_pct"] else "bw"
        out["achieved_pct"] = max(out["tf_pct"], out["bw_pct"])
    else:
        out["bound"] = "bw"
        out["achieved_pct"] = out["bw_pct"]
    return out


def _sets_for(mb_target: float, one_mb: float, cap: int = 192) -> int:
    return int(max(2, min(cap, math.ceil(mb_target / max(one_mb, 1e-6)))))


# ------------------------------------------------------------------- 節: qmm

# 本番で当たる (K, N)。name -> (K, N, bits, 層数, メモ)
# 層数は「1 round / 1 chunk あたり何回呼ばれるか」(占有の重み付け用)。
QMM_SHAPES = [
    # --- Flash-Next (qwen4_exp、hidden 2560、48 層 = GDN 36 + attn 12) ---
    ("fn_gdn_in_qkv",   2560, 10240, 4, 36, "GDN in_proj_qkv"),
    ("fn_gdn_in_z",     2560,  6144, 4, 36, "GDN in_proj_z"),
    ("fn_gdn_in_ba",    2560,    48, 4, 72, "GDN in_proj_b / _a (N=48)"),
    ("fn_gdn_out",      6144,  2560, 4, 36, "GDN out_proj"),
    ("fn_attn_qg",      2560, 12288, 4, 12, "attention q_proj (+gate)"),
    ("fn_attn_kv",      2560,   512, 4, 24, "attention k_proj / v_proj"),
    ("fn_attn_o",       6144,  2560, 4, 12, "attention o_proj"),
    ("fn_idx_qk",       2560,   640, 4, 12, "QSA index_qk_proj"),
    ("fn_hc_down",     10240,   320, 4, 97, "HC input_mix_weight_down"),
    ("fn_hc_up",         320, 10240, 4, 97, "HC input_mix_weight_up"),
    ("fn_moe_router",   2560,   512, 4, 48, "MoE gate (router)"),
    ("fn_shared_gu",    2560,   640, 4, 96, "shared_expert gate/up"),
    ("fn_shared_down",   640,  2560, 4, 48, "shared_expert down"),
    ("fn_ple_key",      2560, 10240, 4,  1, "PLE key_proj"),
    ("fn_lm_head",      2560, 248320, 4, 1, "lm_head (4bit パック)"),
    # --- 27B (qwen3_5、hidden 5120、64 層 = GDN 48 + attn 16、dense MLP) ---
    ("q27_gdn_in_qkv",  5120, 10240, 4, 48, "27B GDN in_proj_qkv"),
    ("q27_gdn_in_z",    5120,  6144, 4, 48, "27B GDN in_proj_z"),
    ("q27_gdn_out",     6144,  5120, 4, 48, "27B GDN out_proj"),
    ("q27_attn_qg",     5120, 12288, 4, 16, "27B attention q_proj (+gate)"),
    ("q27_attn_kv",     5120,  1024, 4, 32, "27B attention k/v_proj"),
    ("q27_attn_o",      6144,  5120, 4, 16, "27B attention o_proj"),
    ("q27_mlp_gu",      5120, 17408, 4, 128, "27B MLP gate/up"),
    ("q27_mlp_down",   17408,  5120, 4, 64, "27B MLP down"),
    ("q27_lm_head",     5120, 248320, 4, 1, "27B lm_head"),
    ("q27_mtp_fc",     10240,  5120, 4,  1, "27B MTP fc (embed|hidden -> hidden)"),
]

QMM_ROWS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 2048]

# 本番でその幅が出ない形は上限を切る (lm_head と MTP fc は draft/verify 幅までしか
# 使わない。2048 行の logits は 1 GB を超えるので測っても意味が無い)。
ROW_CAP = {"fn_lm_head": 64, "q27_lm_head": 64, "q27_mtp_fc": 64}


def _quant(N, K, bits, gs=GS):
    w = (mx.random.normal((N, K)) * 0.05).astype(DT)
    wq, sc, bi = mx.quantize(w, group_size=gs, bits=bits)
    mx.eval(wq, sc, bi)
    del w
    return wq, sc, bi


def _qmm_bytes(M, K, N, bits, gs=GS):
    wb = N * K * bits / 8 + 2 * N * math.ceil(K / gs) * 2
    return int(wb + M * K * 2), int(M * N * 2)


def section_qmm(args):
    rows = [int(r) for r in args.rows.split(",")] if args.rows else QMM_ROWS
    want = args.shapes.split(",") if args.shapes else None
    out = []
    for name, K, N, bits, nlayer, memo in QMM_SHAPES:
        if want and name not in want:
            continue
        for bit in ([bits] + ([8] if args.bits8 else [])):
            one_mb = (N * K * bit / 8 + 2 * N * math.ceil(K / GS) * 2) / 1e6
            n_sets = _sets_for(args.mb, one_mb)
            ws = [_quant(N, K, bit) for _ in range(n_sets)]
            print(f"  [{name}] K={K} N={N} {bit}bit  重み {one_mb:.2f} MB x {n_sets} "
                  f"= {one_mb*n_sets:.0f} MB", flush=True)
            for M in rows:
                if M > ROW_CAP.get(name, 1 << 30):
                    continue
                x = mx.random.normal((M, K)).astype(DT)
                mx.eval(x)

                def make(i, x=x, bit=bit):
                    wq, sc, bi = ws[i]
                    return mx.quantized_matmul(x, wq, scales=sc, biases=bi,
                                               transpose=True, group_size=GS, bits=bit)

                rd, wr = _qmm_bytes(M, K, N, bit)
                st = measure(make, n_sets, wr, reps=args.reps)
                r = rates(st["us"], rd, wr, 2.0 * M * K * N)
                out.append({"op": "quantized_matmul", "shape": name, "K": K, "N": N,
                            "bits": bit, "M": M, "calls_per_pass": nlayer,
                            "memo": memo, "mb_cycled": round(one_mb * n_sets, 1),
                            **st, **r})
                print(f"    M={M:<5} {st['us']:9.2f} us  {r['GBs']:7.1f} GB/s "
                      f"({r['bw_pct']:5.1f}%)  {r.get('TFLOPS', 0):6.2f} TF "
                      f"({r.get('tf_pct', 0):5.1f}%)  達成 {r['achieved_pct']:5.1f}% "
                      f"[{r['bound']}]", flush=True)
                del x
            del ws
            mx.clear_cache()
    return out


# ------------------------------------------------------------------ 節: gqmm

# MoE (Flash-Next): 512 専門家、top-10、moe_intermediate 640。
# name -> (E, K, N)。行数は「(トークン数 x top_k) の合計行」で掃く。
GQMM_SHAPES = [
    ("fn_moe_gate_up", 512, 2560, 640, "switch_mlp gate/up (K=2560 N=640)"),
    ("fn_moe_down",    512,  640, 2560, "switch_mlp down (K=640 N=2560)"),
]
GQMM_ROWS = [10, 20, 40, 80, 160, 512, 1280, 5120, 20480]


def section_gqmm(args):
    out = []
    for name, E, K, N, memo in GQMM_SHAPES:
        one_mb = E * (N * K * 4 / 8 + 2 * N * math.ceil(K / GS) * 2) / 1e6
        n_sets = _sets_for(args.mb, one_mb, cap=8)
        ws = [_quant_experts(E, N, K) for _ in range(n_sets)]
        print(f"  [{name}] E={E} K={K} N={N}  重み {one_mb:.1f} MB x {n_sets}", flush=True)
        for R in GQMM_ROWS:
            x = mx.random.normal((R, 1, K)).astype(DT)
            # 本番と同じ: ソート済みの専門家 index (sorted_indices=True)。
            # **index も巡回させる**。1 本の index を使い回すと、行数が少ないとき
            # 触る専門家が毎回同じ 10〜20 人になり、その重み (10〜20 MB) が
            # キャッシュに残って温の値が出る (2026-09-04 に実際に出た)。
            n_idx = max(2, min(48, math.ceil(220.0 / max(R * one_mb / E, 1e-6))))
            idxs = [mx.sort(mx.random.randint(0, E, (R,))) for _ in range(n_idx)]
            mx.eval(x, idxs)
            uniq = int(mx.unique(idxs[0]).size) if hasattr(mx, "unique") else min(R, E)

            def make(i, x=x):
                # 重みの組と index を別々の周期で回す (i は max(n_sets, n_idx) まで来る)
                wq, sc, bi = ws[i % n_sets]
                return mx.gather_qmm(x, wq, sc, bi, rhs_indices=idxs[i % n_idx],
                                     transpose=True,
                                     group_size=GS, bits=4, sorted_indices=True)

            # 読むバイト: 触れた専門家ぶんの重み (重複は 1 回と数える) + x
            wb_e = N * K * 4 / 8 + 2 * N * math.ceil(K / GS) * 2
            rd = int(uniq * wb_e + R * K * 2)
            wr = int(R * N * 2)
            st = measure(make, max(n_sets, n_idx), wr, reps=args.reps)
            r = rates(st["us"], rd, wr, 2.0 * R * K * N)
            out.append({"op": "gather_qmm", "shape": name, "E": E, "K": K, "N": N,
                        "rows": R, "experts_touched": uniq, "n_idx": n_idx, "memo": memo,
                        "rows_per_expert": round(R / max(uniq, 1), 2), **st, **r})
            print(f"    rows={R:<6} exp={uniq:<4} {st['us']:9.2f} us "
                  f"{r['GBs']:7.1f} GB/s ({r['bw_pct']:5.1f}%) "
                  f"{r.get('TFLOPS', 0):6.2f} TF ({r.get('tf_pct', 0):5.1f}%) "
                  f"達成 {r['achieved_pct']:5.1f}% [{r['bound']}]", flush=True)
            del x, idxs
        del ws
    return out


def _quant_experts(E, N, K):
    w = (mx.random.normal((E, N, K)) * 0.05).astype(DT)
    wq, sc, bi = mx.quantize(w, group_size=GS, bits=4)
    mx.eval(wq, sc, bi)
    del w
    return wq, sc, bi


# ------------------------------------------------------------------ 節: sdpa

SDPA_CASES = [
    ("fn", 24, 2, 256),   # Flash-Next: 24 heads / 2 kv / head_dim 256
    ("q27", 24, 4, 256),  # 27B: 24 heads / 4 kv / head_dim 256
]
SDPA_S = [1, 2, 3, 4, 6, 8]
SDPA_KV = [512, 2048, 4096, 8192, 17408, 51200]


def section_sdpa(args):
    out = []
    for tag, Hq, Hk, D in SDPA_CASES:
        for kv in SDPA_KV:
            one_mb = 2 * Hk * kv * D * 2 / 1e6
            n_sets = _sets_for(args.mb, one_mb, cap=64)
            ks = [mx.random.normal((1, Hk, kv, D)).astype(DT) for _ in range(n_sets)]
            vs = [mx.random.normal((1, Hk, kv, D)).astype(DT) for _ in range(n_sets)]
            mx.eval(ks, vs)
            print(f"  [{tag}] kv={kv}  K/V {one_mb:.1f} MB x {n_sets} "
                  f"= {one_mb*n_sets:.0f} MB", flush=True)
            for S in SDPA_S:
                q = mx.random.normal((1, Hq, S, D)).astype(DT)
                # 本番の decode/verify には必ずマスクがある。マスクの有無で
                # MLX のカーネル選択が変わりうるので両方測る (masks の有無は
                # 可視集合を変えないよう全 True の bool にする)。
                msk = mx.ones((1, 1, S, kv), dtype=mx.bool_)
                mx.eval(q, msk)
                scale = D ** -0.5

                for mtag, mm in (("", None), ("_masked", msk)):
                    def make(i, q=q, mm=mm):
                        return mx.fast.scaled_dot_product_attention(
                            q, ks[i], vs[i], scale=scale, mask=mm)

                    rd = int(2 * Hk * kv * D * 2 + Hq * S * D * 2
                             + (S * kv if mm is not None else 0))
                    wr = int(Hq * S * D * 2)
                    flops = 2.0 * 2 * Hq * S * kv * D  # QK^T と PV
                    st = measure(make, n_sets, wr, reps=args.reps)
                    r = rates(st["us"], rd, wr, flops, peak_tf=PEAK_TF_BF16)
                    out.append({"op": "sdpa_d256", "shape": tag + mtag, "heads": Hq,
                                "kv_heads": Hk, "kv": kv, "S": S,
                                "masked": mm is not None, **st, **r})
                    print(f"    S={S:<3}{mtag:8} {st['us']:9.2f} us  "
                          f"{r['GBs']:7.1f} GB/s ({r['bw_pct']:5.1f}%)  "
                          f"{r.get('TFLOPS', 0):6.2f} TF "
                          f"達成 {r['achieved_pct']:5.1f}% [{r['bound']}]", flush=True)
                del q, msk
            del ks, vs
    return out


# ------------------------------------------------------------------ 節: norm

NORM_ROWS = [1, 2, 3, 4, 6, 8, 64, 256, 2048]


def section_norm(args):
    out = []
    cases = []
    # (name, 行の形を作る関数, 1 呼び出しのバイト, memo)
    for tag, H in (("fn", 2560), ("q27", 5120)):
        cases.append((f"{tag}_rms_hidden", "rms", (H,), H, f"hidden {H} の RMSNorm"))
    cases.append(("fn_rms_hc", "rms_hc", (4, 2560), 4 * 2560, "HC レーンごと (4, 2560)"))
    cases.append(("gdn_rms_head", "rms", (48 * 128,), 48 * 128, "GDN gated RMSNorm 幅 6144"))
    cases.append(("fn_rope", "rope", (24, 256), 24 * 256, "rope 24 head x 256 (dims=64)"))
    cases.append(("q27_rope", "rope", (24, 256), 24 * 256, "27B rope (同形)"))
    cases.append(("fn_ln_hidden", "ln", (2560,), 2560, "layer_norm hidden 2560"))

    for name, kind, shape, nelem, memo in cases:
        one_mb = nelem * 2 * 2048 / 1e6
        n_sets = _sets_for(args.mb, one_mb, cap=64)
        for S in NORM_ROWS:
            xs = [mx.random.normal((1, S) + shape).astype(DT) for _ in range(n_sets)]
            w = mx.ones((shape[-1],)).astype(DT)
            b = mx.zeros((shape[-1],)).astype(DT)
            mx.eval(xs, w, b)

            if kind in ("rms", "rms_hc"):
                def make(i):
                    return mx.fast.rms_norm(xs[i], w, 1e-6)
            elif kind == "ln":
                def make(i):
                    return mx.fast.layer_norm(xs[i], w, b, 1e-6)
            else:  # rope: (B, n_heads, S, D) が要る
                xs = [mx.random.normal((1, shape[0], S, shape[1])).astype(DT)
                      for _ in range(n_sets)]
                mx.eval(xs)

                def make(i):
                    return mx.fast.rope(xs[i], 64, traditional=False, base=1e7,
                                        scale=1.0, offset=0)

            nb = int(S * nelem * 2)
            st = measure(make, n_sets, nb, reps=args.reps)
            r = rates(st["us"], nb, nb, 0)
            out.append({"op": kind, "shape": name, "S": S, "elem_per_row": nelem,
                        "memo": memo, **st, **r})
            print(f"  [{name}] S={S:<5} {st['us']:8.2f} us  {r['GBs']:7.1f} GB/s "
                  f"({r['bw_pct']:5.1f}%)", flush=True)
            del xs
    return out


# ------------------------------------------------------------------ 節: conv

def section_conv(args):
    """GDN の depthwise conv1d (conv_dim 10240、kernel 4)。"""
    out = []
    C, KS = 10240, 4
    one_mb = C * KS * 2 / 1e6
    n_sets = _sets_for(args.mb, one_mb, cap=192)
    ws = [mx.random.normal((C, KS, 1)).astype(DT) for _ in range(n_sets)]
    mx.eval(ws)
    for S in [1, 2, 3, 4, 6, 8, 64, 256, 2048]:
        x = mx.random.normal((1, S + KS - 1, C)).astype(DT)
        mx.eval(x)

        def make(i, x=x):
            return mx.conv1d(x, ws[i], padding=0, groups=C)

        rd = int((S + KS - 1) * C * 2 + C * KS * 2)
        wr = int(S * C * 2)
        st = measure(make, n_sets, wr, reps=args.reps)
        r = rates(st["us"], rd, wr, 2.0 * S * C * KS)
        out.append({"op": "conv1d_depthwise", "shape": "gdn_conv", "S": S,
                    "C": C, "k": KS, **st, **r})
        print(f"  [gdn_conv] S={S:<5} {st['us']:8.2f} us  {r['GBs']:7.1f} GB/s "
              f"({r['bw_pct']:5.1f}%)", flush=True)
        del x
    return out


# ------------------------------------------------------------------- 節: idx

def section_idx(args):
    """MoE routing と QSA の選択で使う索引系。"""
    out = []
    n_sets = 48

    def add(name, make, rd, wr, memo, **extra):
        st = measure(make, n_sets, wr, reps=args.reps)
        r = rates(st["us"], rd, wr, 0)
        out.append({"op": "index", "shape": name, "memo": memo, **extra, **st, **r})
        print(f"  [{name}] {st['us']:8.2f} us  {r['GBs']:7.1f} GB/s "
              f"({r['bw_pct']:5.1f}%)  {memo}", flush=True)

    # --- MoE router: (M, 512) の top-10 ---
    for M in (1, 2, 4, 8, 64, 2048):
        logits = [mx.random.normal((M, 512)).astype(mx.float32) for _ in range(n_sets)]
        mx.eval(logits)
        add(f"argpartition_512_top10_M{M}",
            lambda i: mx.argpartition(logits[i], 512 - 10, axis=-1)[..., -10:],
            M * 512 * 4, M * 10 * 4, "router top-10 (argpartition)", M=M)
        add(f"argsort_512_M{M}", lambda i: mx.argsort(logits[i], axis=-1),
            M * 512 * 4, M * 512 * 4, "router 全ソート (対照)", M=M)
        idxs = [mx.argpartition(l, 502, axis=-1)[..., -10:] for l in logits]
        mx.eval(idxs)
        add(f"take_along_axis_512_M{M}",
            lambda i: mx.take_along_axis(logits[i], idxs[i], axis=-1),
            M * 512 * 4, M * 10 * 4, "選ばれた 10 個の重みを取る", M=M)
        del logits, idxs

    # --- MoE の並べ替え (sorted_indices 用の argsort、行 = M*10) ---
    for R in (10, 20, 80, 20480):
        keys = [mx.random.randint(0, 512, (R,)) for _ in range(n_sets)]
        mx.eval(keys)
        add(f"argsort_expert_rows_{R}", lambda i: mx.argsort(keys[i]),
            R * 4, R * 4, "専門家 index の並べ替え", rows=R)
        src = [mx.random.normal((R, 2560)).astype(DT) for _ in range(min(n_sets, 8))]
        ordr = [mx.argsort(k) for k in keys[:len(src)]]
        mx.eval(src, ordr)
        add(f"take_rows_{R}x2560", lambda i: mx.take(src[i % len(src)],
                                                     ordr[i % len(src)], axis=0),
            R * 2560 * 2, R * 2560 * 2, "行の並べ替え (gather)", rows=R)
        del keys, src, ordr

    # --- QSA の選択: (1, S, n_blocks) から top-512 ---
    for S, nb in ((1, 4352), (4, 4352), (2048, 4352), (1, 12800)):
        sc = [mx.random.normal((1, S, nb)).astype(mx.float32) for _ in range(min(n_sets, 8))]
        mx.eval(sc)
        ns = len(sc)
        add(f"argpartition_blocks_{nb}_S{S}",
            lambda i: mx.argpartition(sc[i % ns], nb - 512, axis=-1)[..., -512:],
            S * nb * 4, S * 512 * 4, "QSA ブロック top-512", S=S, n_blocks=nb)
        del sc

    # --- scatter_add (MoE combine 側で使う形) ---
    for R in (10, 80, 20480):
        vals = [mx.random.normal((R, 2560)).astype(mx.float32) for _ in range(min(n_sets, 8))]
        dst = mx.zeros((max(R // 10, 1), 2560), mx.float32)
        ids = [mx.random.randint(0, max(R // 10, 1), (R,)) for _ in range(len(vals))]
        mx.eval(vals, dst, ids)
        nv = len(vals)
        add(f"scatter_add_{R}x2560",
            lambda i: dst.at[ids[i % nv]].add(vals[i % nv]),
            R * 2560 * 4, R * 2560 * 4, "行の散布加算", rows=R)
        del vals, ids
    return out


# ------------------------------------------------------------------- 節: red

def section_red(args):
    out = []
    n_sets = 8
    V = 248320
    for name, W, dt in (("vocab_f32", V, mx.float32), ("vocab_bf16", V, DT),
                        ("router_512", 512, mx.float32)):
        for M in (1, 2, 4, 8):
            xs = [mx.random.normal((M, W)).astype(dt) for _ in range(n_sets)]
            mx.eval(xs)
            nb = M * W * (4 if dt == mx.float32 else 2)
            for op, fn, wb in (
                ("softmax", lambda i: mx.softmax(xs[i], axis=-1), nb),
                ("logsumexp", lambda i: mx.logsumexp(xs[i], axis=-1), M * 4),
                ("argmax", lambda i: mx.argmax(xs[i], axis=-1), M * 4),
                ("max", lambda i: mx.max(xs[i], axis=-1), M * 4),
            ):
                st = measure(fn, n_sets, wb, reps=args.reps)
                r = rates(st["us"], nb, wb, 0)
                out.append({"op": op, "shape": f"{name}_M{M}", "M": M, "W": W,
                            "dtype": str(dt).split(".")[-1], **st, **r})
                print(f"  [{op} {name} M={M}] {st['us']:8.2f} us  "
                      f"{r['GBs']:7.1f} GB/s ({r['bw_pct']:5.1f}%)", flush=True)
            del xs
    return out


# ----------------------------------------------------------------- 節: embed

def section_embed(args):
    out = []
    V, H = 248320, 2560
    emb = nn.QuantizedEmbedding(V, H, group_size=GS, bits=4)
    mx.eval(emb.parameters())
    for M in (1, 2, 4, 8, 64, 2048):
        ids = [mx.random.randint(0, V, (1, M)) for _ in range(16)]
        mx.eval(ids)

        def make(i):
            return emb(ids[i % 16])

        rd = int(M * (H * 4 / 8 + 2 * (H // GS) * 2))
        wr = int(M * H * 2)
        st = measure(make, 16, wr, reps=args.reps)
        r = rates(st["us"], rd, wr, 0)
        out.append({"op": "quantized_embedding", "shape": "fn_embed", "M": M, **st, **r})
        print(f"  [embed M={M}] {st['us']:8.2f} us  {r['GBs']:7.1f} GB/s "
              f"({r['bw_pct']:5.1f}%)", flush=True)
        del ids
    return out


def section_dense(args):
    """量子化されていない細い matmul と QSA のスコア einsum。

    - HC の `block_inject_weight` は本番のパックでも量子化対象に入らず bf16 の
      nn.Linear のまま残る (`tools/micro_kernel_latency.py` の HC_QBITS の注記)。
      形は (10240 -> 4) で、N=4 は MLX の GEMM タイル (BN=32/64) の 1/8 以下。
    - QSA のブロックスコアは `einsum("bshd,bnd->bsnh")` (indexer 4 head x 128)。
      CATCHUP 17:15 の天井スタブで 17k の ms/round の 4.3% (top-k) + 8.9% (attn)。
    """
    out = []
    # --- HC inject: (M, 10240) @ (10240, 4) bf16 ---
    one_mb = 4 * 10240 * 2 / 1e6
    n_sets = _sets_for(args.mb, one_mb)
    ws = [mx.random.normal((4, 10240)).astype(DT) for _ in range(n_sets)]
    mx.eval(ws)
    for M in (1, 2, 4, 8, 64, 256, 2048):
        x = mx.random.normal((M, 10240)).astype(DT)
        mx.eval(x)

        def make(i, x=x):
            return x @ ws[i].T

        rd, wr = int(4 * 10240 * 2 + M * 10240 * 2), int(M * 4 * 2)
        st = measure(make, n_sets, wr, reps=args.reps)
        r = rates(st["us"], rd, wr, 2.0 * M * 10240 * 4, peak_tf=PEAK_TF_BF16)
        out.append({"op": "matmul_bf16", "shape": "fn_hc_inject", "K": 10240,
                    "N": 4, "M": M, "calls_per_pass": 97,
                    "memo": "HC block_inject_weight (bf16、N=4)", **st, **r})
        print(f"  [hc_inject] M={M:<5} {st['us']:8.2f} us  {r['GBs']:7.1f} GB/s "
              f"({r['bw_pct']:5.1f}%)  {r.get('TFLOPS', 0):5.2f} TF", flush=True)
        del x
    del ws

    # --- QSA のブロックスコア einsum ---
    Hi, Di = 4, 128
    for nb in (512, 4352, 12800):
        one_mb = nb * Di * 4 / 1e6
        n_sets = _sets_for(args.mb, one_mb, cap=64)
        ks = [mx.random.normal((1, nb, Di)).astype(mx.float32) for _ in range(n_sets)]
        mx.eval(ks)
        for S in (1, 2, 4, 8, 2048):
            q = mx.random.normal((1, S, Hi, Di)).astype(mx.float32)
            mx.eval(q)

            def make(i, q=q):
                return mx.einsum("bshd,bnd->bsnh", q, ks[i])

            rd = int(nb * Di * 4 + S * Hi * Di * 4)
            wr = int(S * nb * Hi * 4)
            st = measure(make, n_sets, wr, reps=args.reps)
            r = rates(st["us"], rd, wr, 2.0 * S * Hi * nb * Di, peak_tf=PEAK_TF_BF16)
            out.append({"op": "qsa_block_score", "shape": f"qsa_nb{nb}",
                        "n_blocks": nb, "S": S, **st, **r})
            print(f"  [qsa_score nb={nb}] S={S:<5} {st['us']:8.2f} us "
                  f"{r['GBs']:7.1f} GB/s ({r['bw_pct']:5.1f}%) "
                  f"{r.get('TFLOPS', 0):5.2f} TF", flush=True)
            del q
        del ks
    return out


SECTIONS = {
    "qmm": section_qmm, "gqmm": section_gqmm, "sdpa": section_sdpa,
    "norm": section_norm, "conv": section_conv, "idx": section_idx,
    "red": section_red, "embed": section_embed, "dense": section_dense,
}


# --------------------------------------------------------------- 報告 (集計)
#
# 「達成率が低い × 占有が大きい」で並べるための重み。1 パス (decode 1 round /
# prefill 2048 チャンク 1 本) あたりの**呼び出し回数**をモデル構造から数えた。
# qmm / dense の行は `calls_per_pass` を測定時に持っているのでそちらを使う。
# ここに書くのはそれ以外の節の分。

CALLS_PER_PASS = {
    # (op, shape) -> 1 パスの回数
    ("sdpa_d256", "fn"): 12,          # full_attention 12 層
    ("sdpa_d256", "q27"): 16,         # 27B は 16 層
    ("gather_qmm", "fn_moe_gate_up"): 96,   # 48 層 x (gate, up)
    ("gather_qmm", "fn_moe_down"): 48,
    ("conv1d_depthwise", "gdn_conv"): 36,
    ("quantized_embedding", "fn_embed"): 1,
    ("qsa_block_score", "qsa_nb4352"): 12,   # QSA は full_attention 層だけ
    ("rms_hc", "fn_rms_hc"): 97,             # HC の レーンごと RMSNorm
    ("rms", "fn_rms_hidden"): 48,            # 層あたり input_layernorm
    ("rms", "gdn_rms_head"): 36,             # GDN の gated RMSNorm
    ("rms", "q27_rms_hidden"): 64,
    ("rope", "fn_rope"): 12,
    ("rope", "q27_rope"): 16,
}

# sdpa は kv ごとに行があるので、パスごとに 1 本だけ選ぶ。
PASS_KV = {"fn_decode_s1": 2048, "fn_decode_s3": 2048,
           "fn_prefill_2048": 4096, "q27_decode_s1": 17408}

# 1 パスの壁時計 (ms)。decode は CATCHUP 21:10、prefill は
# bench/results/prefill-anatomy-0903-*.json の wall を 2048 トークンに正規化した値。
PASS_MS = {
    "fn_decode_s1": 23.00,    # Flash-Next decode 1 round (S=1、短文脈)
    "fn_decode_s3": 37.18,    # 同 S=3 (verify 幅)
    "fn_prefill_2048": 3138.0,  # 4k の 1 チャンク相当 (5924 ms / 3867 tok x 2048)
    "q27_decode_s1": 45.10,   # 27B の幅 1 round (scratchpad/b2_fixed_cost_micro.py)
}


def _calls(row):
    if "calls_per_pass" in row:
        return row["calls_per_pass"]
    return CALLS_PER_PASS.get((row.get("op"), row.get("shape")))


def _width(row):
    for k in ("M", "S", "rows"):
        if k in row:
            return row[k]
    return None


def report(paths):
    """測った JSON をまとめて「達成率 x 占有」で並べる。"""
    rows = []
    for p in paths:
        rows += json.load(open(p))["rows"]

    def emit(title, pass_key, width_pick, model_prefix):
        base = PASS_MS[pass_key]
        kv_pick = PASS_KV[pass_key]
        recs = []
        for r in rows:
            n = _calls(r)
            if not n:
                continue
            if "kv" in r and r["kv"] != kv_pick:
                continue
            sh = str(r.get("shape", ""))
            if model_prefix == "fn" and sh.startswith("q27"):
                continue
            if model_prefix == "q27" and not sh.startswith("q27"):
                continue
            w = _width(r)
            if w != width_pick(r):
                continue
            occ = n * r["us"] / 1000.0
            recs.append((occ * (1 - min(r["achieved_pct"], 100) / 100.0), occ, r))
        recs.sort(reverse=True)
        print(f"\n### {title} (1 パス {base:.1f} ms)")
        print(f"{'項目':28} {'形':22} {'幅':>6} {'us':>9} {'GB/s':>8} "
              f"{'TF':>7} {'達成%':>7} {'占有ms':>8} {'占有%':>7} {'天井差ms':>9}")
        for loss, occ, r in recs[:25]:
            print(f"{r.get('shape',''):28} "
                  f"K{r.get('K', r.get('kv', r.get('C', '-')))}xN{r.get('N', '-'):<8} "
                  f"{_width(r):>6} {r['us']:9.2f} {r['GBs']:8.1f} "
                  f"{r.get('TFLOPS', 0):7.2f} {r['achieved_pct']:7.1f} "
                  f"{occ:8.2f} {occ/base*100:7.1f} {loss:9.2f}")

    emit("Flash-Next decode S=1", "fn_decode_s1", lambda r: 1, "fn")
    emit("Flash-Next verify 幅 (S=4 相当)", "fn_decode_s3",
         lambda r: 4 if r.get("op") != "gather_qmm" else 40, "fn")
    emit("Flash-Next prefill (2048 幅)", "fn_prefill_2048",
         lambda r: 2048 if r.get("op") != "gather_qmm" else 20480, "fn")
    emit("27B decode S=1", "q27_decode_s1", lambda r: 1, "q27")

    print("\n### 達成率の低い順 (占有を掛けない生の一覧、上位 30)")
    flat = sorted(rows, key=lambda r: r["achieved_pct"])
    print(f"{'op':22} {'形':24} {'幅':>6} {'us':>9} {'GB/s':>8} {'達成%':>7}")
    for r in flat[:30]:
        print(f"{r['op'][:22]:22} {str(r.get('shape',''))[:24]:24} "
              f"{str(_width(r)):>6} {r['us']:9.2f} {r['GBs']:8.1f} "
              f"{r['achieved_pct']:7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="qmm")
    ap.add_argument("--rows", default="")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--mb", type=float, default=140.0, help="巡回させる重み/KV の総 MB")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--bits8", action="store_true", help="8bit (g64) も測る")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="", help="測った JSON をカンマ区切りで渡すと集計だけ出す")
    a = ap.parse_args()

    if a.report:
        report([p for p in a.report.split(",") if p])
        return

    mx.random.seed(a.seed)
    names = list(SECTIONS) if a.sections == "all" else a.sections.split(",")

    # burn-in: プロセス起動直後の段差 (+7〜9%) を捨てる
    w = _quant(4096, 4096, 4)
    xb = mx.random.normal((64, 4096)).astype(DT)
    for _ in range(6):
        mx.eval(mx.quantized_matmul(xb, w[0], scales=w[1], biases=w[2],
                                    transpose=True, group_size=GS, bits=4))
    del w, xb

    rows = []
    for n in names:
        n = n.strip()
        if not n:
            continue
        print(f"\n=== 節 {n} ===", flush=True)
        t0 = time.perf_counter()
        rows += SECTIONS[n](a)
        print(f"=== 節 {n}: {time.perf_counter()-t0:.1f} s ===", flush=True)

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"peak_bw_gbs": PEAK_BW / 1e9,
                       "peak_tf_qmm": PEAK_TF_QMM / 1e12,
                       "peak_tf_bf16": PEAK_TF_BF16 / 1e12,
                       "mb_cycled_target": a.mb, "rows": rows}, f, indent=1)
        print(f"\n書き出し: {a.out} ({len(rows)} 行)")


if __name__ == "__main__":
    main()
