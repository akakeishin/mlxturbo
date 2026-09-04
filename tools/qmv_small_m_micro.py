"""検証幅 M=1..8 の量子化 dense 射影を、候補カーネルごとに比べる冷 micro。

`tools/verify_width_cost_27b.py` の実機の内訳で、27B の verify 幅 S の超線形は
**量子化 dense 行列積 (MLX の `qmv_wide`) の M カーブ**に絞れた
(mlp_all が S=1 25.8 ms → S=4 44.4 ms で、全体の増分 +21.3 ms のうち +18.6 ms)。
実効帯域は M=1 372 / M=2 312 / M=3 218 / M=4 216 GB/s で、重みの読みは
M<=5 ならどれも 1 回なのに落ちる。ここはその帯を埋める候補を並べて測る。

候補:

- `stock`     : `mx.quantized_matmul` (M=1 は qmv_fast、M>=2 は qmv_wide)
- `sm/<nsg>x<rps>` : `mlxturbo/kernels/qmv_small_m.py` (この的の本命。mlx の
                qmv_fast の構造のまま M 行を同時に持つ。nsg = threadgroup
                あたりの simdgroup 数、rps = 1 スレッドが持つ出力行数
                (`results_per_simdgroup`、本家 4))
- `nocap`     : `kernels/qmv_wide_nocap.py` (mlx の `qmv_wide` の写し)
- `mma`       : `mlxturbo/fast_qmm.py` (M<5 は 5 行に 0 詰めして渡す)

## 冷やし方 (温の連鎖を信じない、`CLAUDE.md`)

同じ重み 1 組を回すと DRAM が温まって「自前カーネルが並列度不足を隠せない
負け」が見えない。形ごとに **重みの写しを `--copies` 本 (100 MB 超) 作って
巡回**し、1 リンク 1 組で連鎖を組む。連鎖はリンク間に本物のデータ依存を入れる
(独立呼び出しを並べると GPU が重ねてしまい、投機デコードの直列な待ちを表さない)。
形ごとの初回はページの初回触りが乗るので、M ループの前に空焼きを 3 回入れる。

## 判定

- `bit`: **各行が `mx.quantized_matmul` を M=1 で呼んだ結果とビット一致**するか。
  これが取れると検証幅で丸めが動かない (生成列が幅で分岐しない)。
- `GB/s`: 重みバイト / 時間。M=1 の素 (qmv_fast) の 372 GB/s 級が上限の目安。

    BIGLOCK_PRIO=2 tools/biglock.sh .venv/bin/python tools/qmv_small_m_micro.py \\
        --out bench/results/qmv-small-m-0904.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

GROUP_SIZE = 64
BITS = 4

# (K, N, 1 forward での本数)。27B (qwen3_5) と Flash-Next (qwen4_exp) の
# 本番の dense 射影。K は mlx が qmv_fast を選ぶ条件 (512 の倍数) を満たす。
SHAPES = {
    # --- 27B (hidden 5120 / MLP 17408 / 64 層、GDN 48 + attention 16) ---
    "27b_mlp_gate_up": (5120, 17408, 128),
    "27b_mlp_down": (17408, 5120, 64),
    "27b_gdn_in_proj_qkv": (5120, 10240, 48),
    "27b_gdn_in_proj_z": (5120, 6144, 48),
    "27b_gdn_out_proj": (6144, 5120, 48),
    "27b_attn_q": (5120, 12288, 16),
    "27b_lm_head": (5120, 248320, 1),
    # --- Flash-Next (hidden 2560、fused._QMM_WIDE_TARGETS の射影) ---
    "fn_attn_q": (2560, 12288, 1),
    "fn_attn_o": (6144, 2560, 1),
    "fn_gdn_in_proj_qkv": (2560, 10240, 1),
    "fn_gdn_in_proj_z": (2560, 6144, 1),
    "fn_lm_head": (2560, 248320, 1),
}


def _quantized(k: int, n: int, dt):
    """packed な 4bit 重み (dense を作らずに済ませる。test_dispatch と同じ手)。"""
    w = mx.random.randint(0, 2**31, shape=(n, k // 8), dtype=mx.uint32)
    scales = (mx.random.uniform(shape=(n, k // GROUP_SIZE)) * 0.02).astype(dt)
    biases = (mx.random.uniform(shape=(n, k // GROUP_SIZE)) * 0.01 - 0.005).astype(dt)
    mx.eval(w, scales, biases)
    return w, scales, biases


def _stock(x, q):
    w, s, b = q
    return mx.quantized_matmul(x, w, s, b, transpose=True,
                               group_size=GROUP_SIZE, bits=BITS)


def _small_m(nsg: int, rps: int):
    from mlxturbo.kernels.qmv_small_m import qmv_small_m

    def fn(x, q):
        w, s, b = q
        return qmv_small_m(x, w, s, b, group_size=GROUP_SIZE, bits=BITS,
                           nsg=nsg, rps=rps)

    return fn


def _nocap(x, q):
    from mlxturbo.kernels.qmv_wide_nocap import qmv_wide_nocap

    w, s, b = q
    return qmv_wide_nocap(x, w, s, b, group_size=GROUP_SIZE, bits=BITS, m_min=1)


def _mma(x, q):
    """fast_qmm。M<5 は窓の外なので 5 行に 0 詰めしてから渡す (費用は M=5 と同じ)。"""
    from mlxturbo.fast_qmm import M_MIN, fast_qmm

    w, s, b = q
    M, K = x.shape
    xin = x
    if M < M_MIN:
        xin = mx.concatenate([x, mx.zeros((M_MIN - M, K), dtype=x.dtype)], axis=0)
    out = fast_qmm(xin, w, s, b, group_size=GROUP_SIZE, bits=BITS, force_wide=True)
    return out[:M] if M < M_MIN else out


def build_cands(cfgs) -> dict:
    """cfgs: (nsg, rps) の列。rps は mlx の `results_per_simdgroup` (本家 4)。"""
    c = {"stock": _stock}
    for g, r in cfgs:
        c[f"sm/{g}x{r}"] = _small_m(g, r)
    c["nocap"] = _nocap
    c["mma"] = _mma
    return c


def chain_ms(fn, x0, qs, reps: int) -> float:
    """`qs` を 1 リンク 1 組で回す依存の連鎖の中央値 (ms)。

    次段の入力は `x0 * (1 + 1e-6 * out[:, :1])`。1 列しか使わないので余計な
    帯域を足さず、リンク間に本物のデータ依存が入る。
    """
    def run():
        x = x0
        for q in qs:
            out = fn(x, q)
            x = x0 * (1.0 + 1e-6 * out[:, :1].astype(x0.dtype))
        mx.eval(x)

    run()
    mx.synchronize()
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        run()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


# ---------------------------------------------------------------- 参照距離

# 精度を見る形。K が縮約の長さ = 誤差を決めるので K だけ本番に合わせ、N は
# fp32 の参照を作れる大きさに切る (出力 1 要素あたりの誤差は N に依らない)。
ACC_SHAPES = {
    "K=5120 (27B hidden)": (5120, 4096),
    "K=17408 (27B mlp down)": (17408, 4096),
    "K=6144 (27B gdn out)": (6144, 4096),
    "K=2560 (Flash-Next hidden)": (2560, 4096),
}


def _fp32_ref(x, q, group_size: int = GROUP_SIZE):
    """float32 で逆量子化してから float32 で掛けた「真値」。"""
    w, s, b = q
    wf = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32),
                       group_size=group_size, bits=BITS)
    return x.astype(mx.float32) @ wf.T


def run_accuracy(m_list, cfgs, dt, reps_unused=None) -> dict:
    """各候補の fp32 参照からの距離を、素と並べて出す。

    ユーザー 2026-09-04 12:40 の方針: ビット一致は条件にしない。**品質に
    影響が無ければよい**ので、判定は「fp32 参照への距離が素以下か」。
    """
    cands = build_cands(cfgs)
    out: dict = {}
    print("\n== fp32 参照からの距離 (max|d| / max|ref|、括弧は RMS 相対)")
    for name, (k, n) in ACC_SHAPES.items():
        q = _quantized(k, n, dt)
        print(f"\n-- {name}  K={k} N={n}")
        rows = {}
        for M in m_list:
            x = (mx.random.normal((M, k)) * 0.1).astype(dt)
            mx.eval(x)
            ref = _fp32_ref(x, q)
            mx.eval(ref)
            scale = float(mx.abs(ref).max())
            rms_ref = float(mx.sqrt(mx.mean(ref * ref)))
            per = {}
            line = f"  M={M:>2} "
            for cname, fn in cands.items():
                got = fn(x, q).astype(mx.float32)
                mx.eval(got)
                d = got - ref
                mrel = float(mx.abs(d).max()) / max(scale, 1e-30)
                rrel = float(mx.sqrt(mx.mean(d * d))) / max(rms_ref, 1e-30)
                per[cname] = {"max_rel": mrel, "rms_rel": rrel}
                line += f" | {cname} {mrel:.2e} ({rrel:.2e})"
            rows[str(M)] = per
            print(line, flush=True)
        out[name] = {"K": k, "N": n, "M": rows}
        del q
        mx.clear_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="M=1..8 の量子化射影の候補比べ")
    ap.add_argument("--m-list", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--copies", type=int, default=8,
                    help="1 形につき作る重みの写し (= 連鎖のリンク数)")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--cfg", default="2x4,2x2,1x4,4x4",
                    help="small_m の (simdgroup 数)x(results_per_simdgroup) の掃引")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--shapes", default=None, help="測る形をカンマ区切りで絞る")
    ap.add_argument("--accuracy", action="store_true",
                    help="速さではなく fp32 参照からの距離だけを出す")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dt = mx.bfloat16 if args.dtype == "bfloat16" else mx.float16
    m_list = [int(m) for m in args.m_list.split(",") if m.strip()]
    cfgs = [tuple(int(t) for t in c.split("x")) for c in args.cfg.split(",")
            if c.strip()]
    cands = build_cands(cfgs)
    names = list(SHAPES) if args.shapes is None else args.shapes.split(",")
    result: dict = {"copies": args.copies, "reps": args.reps,
                    "dtype": args.dtype, "cfg": args.cfg, "shapes": {}}

    if args.accuracy:
        result = {"dtype": args.dtype, "cfg": args.cfg,
                  "accuracy": run_accuracy(m_list, cfgs, dt)}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps(result, indent=1, ensure_ascii=False))
            print(f"\n書き出し: {args.out}")
        return 0

    for name in names:
        k, n, count = SHAPES[name]
        copies = max(1, min(args.copies, int(2.0e9 / (n * k * 0.5 + 1))))
        qs = [_quantized(k, n, dt) for _ in range(copies)]
        # 重み + scales/biases のバイト数 (1 リンク)
        wbytes = n * k // 2 + n * (k // GROUP_SIZE) * 4
        print(f"\n== {name}  K={k} N={n} (本番 {count} 本/forward)"
              f"  写し {copies} 本 = {copies * wbytes / 1e6:.0f} MB", flush=True)

        # 形ごとの空焼き (初回のページ触りを M=1 の行に乗せない)
        x_warm = (mx.random.normal((4, k)) * 0.1).astype(dt)
        mx.eval(x_warm)
        for _ in range(3):
            chain_ms(_stock, x_warm, qs, 1)

        rows: dict = {}
        for M in m_list:
            x0 = (mx.random.normal((M, k)) * 0.1).astype(dt)
            mx.eval(x0)
            # 各行を M=1 で素に流したもの = 一致の基準
            ref1 = mx.concatenate(
                [_stock(x0[v:v + 1], qs[0]) for v in range(M)], axis=0)
            mx.eval(ref1)
            per = {}
            for cname, fn in cands.items():
                try:
                    got = fn(x0, qs[0])
                    mx.eval(got)
                    bit = bool(got.shape == ref1.shape
                               and mx.array_equal(got, ref1))
                    ms = chain_ms(fn, x0, qs, args.reps)
                except Exception as exc:  # noqa: BLE001
                    per[cname] = {"error": str(exc)[:160]}
                    continue
                per[cname] = {"ms": ms, "ms_per_link": ms / copies, "bit": bit,
                              "gbs": wbytes / (ms / copies) / 1e6}
            rows[str(M)] = per
            base = per["stock"]["ms_per_link"]
            line = (f"  M={M:>2}  stock {base * 1000:7.1f} us"
                    f" {per['stock']['gbs']:6.0f} GB/s"
                    f"{' bit' if per['stock']['bit'] else '    '}")
            for cname in cands:
                if cname == "stock":
                    continue
                c = per.get(cname, {})
                if "ms" in c:
                    line += (f" | {cname} {c['ms_per_link'] * 1000:6.1f}us"
                             f" {c['gbs']:5.0f} ({c['ms_per_link'] / base:.2f}x"
                             f"{' bit' if c['bit'] else ''})")
                else:
                    line += f" | {cname} x"
            print(line, flush=True)
        result["shapes"][name] = {"K": k, "N": n, "count": count,
                                  "copies": copies, "wbytes": wbytes, "M": rows}
        del qs
        mx.clear_cache()

    # ---- 27B の 1 forward ぶんに直した合計 -------------------------------
    fw = [n for n in names if n.startswith("27b_")]
    if fw:
        print("\n== 27B の 1 forward ぶん (本数 x ms/link、ms)")
        hdr = f"{'候補':<14}" + "".join(f"{('M=' + str(m)):>9}" for m in m_list)
        print(hdr)
        totals: dict = {}
        for cname in cands:
            row = f"{cname:<14}"
            totals[cname] = {}
            for M in m_list:
                tot, ok = 0.0, True
                for name in fw:
                    sh = result["shapes"][name]
                    c = sh["M"][str(M)].get(cname, {})
                    if "ms_per_link" not in c:
                        ok = False
                        break
                    tot += c["ms_per_link"] * sh["count"]
                totals[cname][str(M)] = tot if ok else None
                row += f"{tot:>9.2f}" if ok else f"{'x':>9}"
            print(row)
        result["forward_totals_27b"] = totals

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"\n書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
