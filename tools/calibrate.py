"""段 C: 閾値の原始量をこのマシンで 1 回だけ測って JSON に落とす。

`docs/research/KERNEL-PROGRAM.md` の「閾値をマシン非依存にする (段 C)」の
実行部。いま既定に埋まっている閾値はほぼ全部 **M3 Max 1 台で測った値**で、
別の機種では動く。定数を消すことはできないが、**式と、そのマシンで 1 回
測れば済む少数の原始量**に分けることはできる。式は `mlxturbo/calibration.py`。

    tools/biglock.sh uv run python tools/calibrate.py --out bench/results/calibration-m3max.json
    uv run python tools/calibrate.py --show bench/results/calibration-m3max.json

## ここで測っているのは「そのマシンの上限」であって、モデル内の実効値ではない

温キャッシュのマイクロベンチの絶対値を信じない、は `CLAUDE.md` の作法。
ここで出す 6 つは**採否の判断材料ではなく、式に入れる係数**である。だから
中央値ではなく**最良値 (最短時間)** を採る -- 熱や他プロセスで下振れした
測定を混ぜると「上限」が上限でなくなるため。逆に言うと、ここの数字を
「モデルの中でこれだけ出ている」と読んではいけない。

各原始量が具体的に何を測っているか:

- `B`  大きい bf16 配列を別の配列へ写す。読み + 書きの合計バイトを時間で
       割る。**達成帯域**であって理論値ではない。参考に読み専 (総和) も
       併記する。docs が下限計算に使っている 393GB/s よりは低く出る
       (M3 Max で 345 / 読み専 363)。
- `G`  同じ KV レイアウト `(1, n_kv_heads, kv, head_dim)` の kv 軸から
       一定割合を `mx.take` で集める。連続長は `head_dim * 2` バイトなので、
       **head_dim ごとに別の値**になる。`B` に対する比で持つ。
- `L`  依存鎖にした極小 op の本数を変えて、傾きから 1 起動あたりの費用を出す。
       `mx.eval` 自身の固定費 (約 160us) は差分で消える。
- `F`  大きい密 4bit `quantized_matmul` の到達 FLOPS。計算律速側の上限。
- `r`  `gather_qmm` を 1 エキスパートあたりの行数を振って回したときの効率
       曲線と、その膝。`tools/probe_gather_qmm.py` と同じ形 (bank 512 x
       640 x 2560、4bit gs=64) を使う。
- `D`  decode 幅の層 1 つぶんの遅延グラフを**構築するだけ**の CPU 時間
       (eval しない)。`async_eval` の費用 `L` との比が `_STAGE_EVERY`。

## 出た値をどう使うか

**このツールは既定値を書き換えない。**`--show` は「式から出た値」と「いまの
既定」を並べて出すだけで、閾値を動かすかは人が決める。文書の反転条件は
「較正で出した閾値が掃引で決めた値と 2 割以上ずれるなら、式が現実を
捉えていない。式を捨てて掃引の値を既定にする」。

実行時に効かせたいときだけ `MLXTURBO_CALIBRATION=<json>` を渡す。
指定が無ければ M3 Max の実測値のままで、使う側が一度だけログに出す。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlxturbo import calibration as C  # noqa: E402

# Flash-Next (qwen4_exp) の形。r 曲線と `M*` の当てはめに使う
D_MODEL = 2560
MOE_INTER = 640
N_EXPERTS = 512
TOP_K = 10
N_KV_HEADS = 2
PREFILL_CHUNK = 2048

# 掃引で出したゼロ交差 (M3 Max、17k/25k/50k、幅 2 で集める割合
# 24%/16%/8% -> ms/round +1.1%/-6.7%/-15.4%)。既定 0.20 はここから
# 安全側に倒した値なので、**式が当てるべき相手はこちら**。
# 反転条件の分母に使うだけで、式には入れない。
SWEPT_GATHER_CROSSING = 0.23


def best(fn, warmup: int = 2, reps: int = 5) -> float:
    """`fn()` を eval まで含めて回し、**最短時間**を秒で返す。

    上限を測っているので中央値ではなく最短を採る (docstring 冒頭の注記)。
    """
    for _ in range(warmup):
        mx.eval(fn())
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        out.append(time.perf_counter() - t)
    return min(out)


# ---- B: 連続読みの達成帯域 -------------------------------------------------


def measure_B(mb: int = 1024, chain: int = 1) -> tuple[float, float]:
    """大きい配列の写しで達成帯域を測る。返り値は (コピー, 読み専) の バイト/秒。

    コピーは読み + 書きの両方が実際に走るので、動いたバイトは `2 * nbytes`。

    **鎖にしない (`chain=1`)。**1 回の eval に複数の写しを積むと最小値だけが
    伸びる (M3 Max 実測: chain 1/4/8 で min 345/399/435 GB/s、中央値は
    340/354/355 でほぼ動かない)。理論値 400 を超える min が出る時点で
    「鎖の中で重なった」ぶんを帯域として数えてしまっており、上限の測り方に
    ならない。配列は 1GiB -- 256MB では eval の固定費 (約 160us) が
    数 % 乗る。"""
    n = mb * 1024 * 1024 // 2  # bf16
    x = (mx.random.normal((n,)) * 0.01).astype(mx.bfloat16)
    mx.eval(x)
    nbytes = n * 2

    def copies():
        return [x + float(i + 1) for i in range(chain)]

    t = best(copies)
    b_copy = chain * 2 * nbytes / t

    # 読み専は「読んだバイト / 時間」。総和は書き戻しがほぼ無い。配列を
    # chain 等分して別々に総和を取るので、全体をちょうど 1 回読む
    t2 = best(lambda: [mx.sum(x[i * (n // chain):(i + 1) * (n // chain)])
                       for i in range(chain)])
    b_read = nbytes / t2
    del x
    return b_copy, b_read


# ---- G: 飛び飛び読みの効率 -------------------------------------------------


G_FRACS = (0.125, 0.25, 0.5)
G_FRAC_REF = 0.25  # 式に入れる G を測る密度。理由は measure_G の docstring


def measure_G(B: float, head_dims=(64, 128, 256), mb: int = 2048,
              fracs=G_FRACS) -> dict:
    """KV レイアウトの kv 軸を `mx.take` で集めたときの達成帯域を `B` で割る。

    形は `(1, n_kv_heads, kv, head_dim)`。kv 軸で集めるので、1 要素あたり
    連続で読めるのは `head_dim * 2` バイト。**head_dim が短いほど G は落ちる**
    (文書の「head_dim=128 の QSA モデルが来れば連続長は半分」)。

    総バイトを head_dim によらず一定にしてあるので、head_dim 間の差は
    連続長の差だけ。`B` と同じ数え方 (読み + 書き) で割るので、写しも take も
    読みと書きが 1:1 である以上、比を取れば書きぶんは相殺し、残るのは
    **読みの連続性の差**だけになる。

    **G は集める密度にも依る (純粋なマシン定数ではない)。**添字がソート済み
    なので、密度が上がるほど隣り合う行が実際に隣り合い、連続読みに近づく。
    M3 Max の実測でも密度 0.25 と 0.5 で 1 割ほど動く。式が解いている `u*` は
    まさにその密度なので、**閾値の近傍 (0.25) で測った G を式に入れる**。
    密度依存そのものは `G_by_frac` に残す -- 消えない依存を平均で隠さない。
    """
    total_elems = mb * 1024 * 1024 // 2
    by_frac = {}
    for frac in fracs:
        row = {}
        for hd in head_dims:
            kv = total_elems // (N_KV_HEADS * hd)
            x = (mx.random.normal((1, N_KV_HEADS, kv, hd)) * 0.01).astype(mx.bfloat16)
            mx.eval(x)
            take = max(1, int(kv * frac))
            rng = np.random.default_rng(0)
            idx = mx.array(np.sort(rng.choice(kv, take, replace=False))
                           .astype(np.uint32))
            mx.eval(idx)

            t = best(lambda x=x, idx=idx: mx.take(x, idx, axis=2))
            moved = 2 * take * N_KV_HEADS * hd * 2  # 集める読み + 書き
            row[str(hd)] = moved / t / B
            del x, idx
        by_frac[str(frac)] = row
    return by_frac


# ---- L: カーネル起動 1 回の費用 --------------------------------------------


def measure_L(short: int = 128, long: int = 1024) -> float:
    """依存鎖の長さを変え、傾きから 1 起動あたりの費用を秒で返す。

    極小の op を直列に積むと時間はほぼ起動回数に比例する。2 点の差分を
    取るので `mx.eval` 自身の固定費 (約 160us) は消える。
    """
    x = mx.zeros((1,), dtype=mx.float32)
    one = mx.ones((1,), dtype=mx.float32)
    mx.eval(x, one)

    def chain(n):
        def go():
            y = x
            for _ in range(n):
                y = y + one
            return y
        return go

    t_s = best(chain(short), reps=7)
    t_l = best(chain(long), reps=7)
    return (t_l - t_s) / (long - short)


# ---- F: 密 4bit qmm の上限 -------------------------------------------------


def measure_F(m: int = 4096, k: int = 4096, n: int = 4096, gs: int = 64,
              bits: int = 4) -> float:
    """大きい密 `quantized_matmul` の到達 FLOPS。

    M を大きく取るので重みの読みは償却され、計算律速側の上限が出る。
    """
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    qw, qs, qb = mx.quantize(w, group_size=gs, bits=bits)
    del w
    x = (mx.random.normal((m, k)) * 0.02).astype(mx.bfloat16)
    mx.eval(qw, qs, qb, x)

    def go():
        return mx.quantized_matmul(x, qw, scales=qs, biases=qb, transpose=True,
                                   group_size=gs, bits=bits)

    t = best(go)
    del qw, qs, qb, x
    return 2.0 * m * k * n / t


# ---- r(rows): gather_qmm の効率曲線 ---------------------------------------


def measure_r(F: float, experts: int = 128,
              rows=(5, 10, 20, 40, 80, 160, 320), gs: int = 64,
              bits: int = 4) -> tuple[list, int]:
    """1 エキスパートあたりの行数を振って `gather_qmm` の効率曲線を出す。

    形は `tools/probe_gather_qmm.py` と同じ (bank 512 x 640 x 2560、4bit)。
    ソート済みの添字で「各エキスパートにちょうど `rows` 行」を与える --
    これは group prefill が作る形そのもの。`r` は密 4bit qmm の上限 `F` に
    対する比で持つ。

    返り値は (曲線, 膝の行数, 膝を挟む 2 点)。膝は「最大効率の 95% に最初に
    届く行数」だが、**この判定は測定ゆらぎで 1 段ずれる。**M3 Max では 80 行が
    ピーク比 0.94、160 行が 0.98 で、走らせるたびに 80 と 160 を行き来した。
    だから膝は 1 点ではなく**挟む 2 点**でも返し、閾値の突き合わせは幅で読む。
    """
    bank = (mx.random.normal((N_EXPERTS, MOE_INTER, D_MODEL)) * 0.02).astype(mx.bfloat16)
    w, s, b = mx.quantize(bank, group_size=gs, bits=bits)
    del bank
    mx.eval(w, s, b)

    curve = []
    for r in rows:
        tokens = experts * r
        x = (mx.random.normal((tokens, 1, 1, D_MODEL)) * 0.02).astype(mx.bfloat16)
        idx = mx.array(np.repeat(np.arange(experts, dtype=np.uint32), r)
                       .reshape(tokens, 1))
        mx.eval(x, idx)

        def go(x=x, idx=idx):
            return mx.gather_qmm(x, w, s, b, rhs_indices=idx, transpose=True,
                                 group_size=gs, bits=bits, sorted_indices=True)

        t = best(go, reps=9)
        flops = 2.0 * tokens * MOE_INTER * D_MODEL / t
        curve.append({"rows": r, "flops": flops, "r": flops / F})
        del x, idx
    del w, s, b

    peak = max(c["flops"] for c in curve)
    knee = curve[-1]["rows"]
    below = curve[0]["rows"]
    for c in curve:
        if c["flops"] >= 0.95 * peak:
            knee = c["rows"]
            break
        below = c["rows"]
    return curve, knee, (below, knee)


# ---- D: グラフ構築 1 層あたりの CPU 時間 -----------------------------------


def _synthetic_layer(h, qkv, o, rw, up, down, gs=64, bits=4):
    """decode 幅の層 1 つぶんの遅延グラフを組む (eval しない)。

    正規化 -> qkv -> sdpa -> o -> 残差 -> 正規化 -> ルータ -> MoE の
    up/活性/down -> 残差。**構築の CPU 時間だけ**が目的なので、数値の
    中身は問わない (重みは実体を持たせてあるが結果は捨てる)。op 数は
    `_OPS_PER_LAYER`。
    """
    (qw, qsc, qb) = qkv
    (ow, osc, ob) = o
    (uw, usc, ub) = up
    (dw, dsc, db) = down

    def norm(t):
        return t * mx.rsqrt(mx.mean(t * t, axis=-1, keepdims=True) + 1e-6)

    y = norm(h)
    q = mx.quantized_matmul(y, qw, scales=qsc, biases=qb, transpose=True,
                            group_size=gs, bits=bits)
    q = q.reshape(1, q.shape[0], -1, 64).transpose(0, 2, 1, 3)
    a = mx.fast.scaled_dot_product_attention(q, q, q, scale=0.125)
    a = a.transpose(0, 2, 1, 3).reshape(h.shape[0], -1)
    y = mx.quantized_matmul(a, ow, scales=osc, biases=ob, transpose=True,
                            group_size=gs, bits=bits)
    h = h + y
    y = norm(h)
    logits = y @ rw
    idx = mx.argpartition(-logits, TOP_K, axis=-1)[..., :TOP_K].astype(mx.uint32)
    g = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1)
    xx = mx.expand_dims(y, (-2, -3))
    hu = mx.gather_qmm(xx, uw, usc, ub, rhs_indices=idx, transpose=True,
                       group_size=gs, bits=bits)
    act = mx.sigmoid(hu) * hu
    hd = mx.gather_qmm(act, dw, dsc, db, rhs_indices=idx, transpose=True,
                       group_size=gs, bits=bits)
    out = (hd.squeeze(-2) * g[..., None]).sum(axis=-2)
    return h + out


_OPS_PER_LAYER = 30  # `_synthetic_layer` が積む op 数 (概算。D の読み方の分母)


def measure_D(layers: int = 48, width: int = 2, gs: int = 64,
              bits: int = 4) -> float:
    """層ループを `layers` 回**構築するだけ**の時間を層数で割る (秒/層)。

    `mx.eval` を timer の外に置くので、ここに入るのは MLX の遅延グラフ構築
    (Python + C++) の CPU 時間だけ。`_STAGE_EVERY` はこの `D` と
    `async_eval` の費用 `L` の比で決まるので、**絶対値より L との比**が本題。
    """
    h = (mx.random.normal((width, D_MODEL)) * 0.02).astype(mx.bfloat16)

    def q(shape):
        w = (mx.random.normal(shape) * 0.02).astype(mx.bfloat16)
        out = mx.quantize(w, group_size=gs, bits=bits)
        del w
        return out

    qkv = q((D_MODEL, D_MODEL))
    o = q((D_MODEL, D_MODEL))
    rw = (mx.random.normal((D_MODEL, N_EXPERTS)) * 0.02).astype(mx.bfloat16)
    up = q((N_EXPERTS, MOE_INTER, D_MODEL))
    down = q((N_EXPERTS, D_MODEL, MOE_INTER))
    mx.eval(h, qkv, o, rw, up, down)

    def build():
        y = h
        for _ in range(layers):
            y = _synthetic_layer(y, qkv, o, rw, up, down, gs=gs, bits=bits)
        return y

    # 1 回目は Python の型キャッシュなどが暖まっていないので捨てる
    for _ in range(2):
        mx.eval(build())

    out = []
    for _ in range(5):
        t = time.perf_counter()
        y = build()
        out.append((time.perf_counter() - t) / layers)
        mx.eval(y)
    del qkv, o, rw, up, down, h
    return min(out)


# ---- 突き合わせ ------------------------------------------------------------


def current_defaults() -> dict:
    """いま本番で効いている既定値を、書いてある場所から引く。

    ここでハードコードすると「既定を変えたのに表が古い」が起きるので、
    実装から読む。読めない環境ではハードコードした値に落ちる。
    """
    d = {"gather_ratio_256": 0.20, "prefill_group": 4, "stage_every": 2,
         "qmm_m_min": 3, "_source": "fallback"}
    try:
        import mlxturbo  # noqa: F401
        import mlx_lm.models.qwen4_exp as Q

        from mlxturbo import spec_flash

        d["gather_ratio_256"] = Q._GATHER_RATIO_MEASURED[256]
        d["prefill_group"] = spec_flash._PREFILL_GROUP
        d["stage_every"] = spec_flash._STAGE_EVERY
        d["_source"] = "implementation"
    except Exception as e:  # インポートできない環境でも表は出す
        d["_source"] = f"fallback ({type(e).__name__})"
    return d


def show(prof: dict) -> None:
    """式から出た閾値と、いまの既定を並べる。**合わせに行かない。**"""
    p = prof["primitives"]
    B = p["B_bytes_per_s"]
    L = p["L_seconds"]
    F = p["F_flops"]
    D = p["D_seconds_per_layer"]
    G = p["G_by_head_dim"]
    cur = current_defaults()

    print("\n=== 原始量 (このマシンの上限) ===")
    print(f"  B  連続読み (コピー)      {B / 1e9:8.1f} GB/s")
    print(f"     連続読み (読み専)      {p['B_read_only_bytes_per_s'] / 1e9:8.1f} GB/s")
    for hd, g in sorted(G.items(), key=lambda kv: int(kv[0])):
        print(f"  G  飛び飛び read head_dim={hd:>3s}  {g:8.3f}"
              f"  (連続 {int(hd) * 2} B、{g * B / 1e9:.1f} GB/s、"
              f"密度 {p['G_frac_ref']})")
    for frac, row in sorted(p["G_by_frac"].items(), key=lambda kv: float(kv[0])):
        cells = "  ".join(f"hd{k}:{v:.3f}"
                          for k, v in sorted(row.items(), key=lambda kv: int(kv[0])))
        print(f"     密度 {float(frac):.3f} での G   {cells}")
    print(f"  L  カーネル起動 1 回      {L * 1e6:8.2f} us")
    print(f"  F  密 4bit qmm の上限     {F / 1e12:8.2f} TFLOPS")
    print(f"  D  グラフ構築 1 層        {D * 1e6:8.2f} us"
          f"  (op {_OPS_PER_LAYER} 個ぶん)")
    br = p.get("r_knee_bracket", [p["r_knee_rows"], p["r_knee_rows"]])
    print(f"  r  gather_qmm の膝        {p['r_knee_rows']:8d} 行/エキスパート"
          f"  (挟む 2 点 {br[0]}-{br[1]})")
    print("     " + "  ".join(f"{c['rows']}行:{c['r']:.2f}" for c in p["r_curve"]))

    print("\n=== 式から出た閾値 vs いまの既定 ===")
    print(f"  (既定の出所: {cur['_source']})")

    # gather の比 u*。kv 長で動くので、実測を取った 4 点で出す
    print("\n  gather の比 u* = (1 - n*L*B/(kv*bytes)) / (2 + 1/G)"
          "   [head_dim=256, n=3]")
    bytes_per_token = N_KV_HEADS * 256 * 2 * 2
    g256 = G.get("256")
    for kv in (17000, 25000, 32000, 50000):
        u = C.gather_max_ratio(B, g256, L, kv, bytes_per_token)
        print(f"    kv={kv:6d}  u* = {u:.3f}")
    u_far = C.gather_max_ratio(B, g256, L, 10 ** 9, bytes_per_token)
    print(f"    kv->無限大 (起動項を無視) u* = {u_far:.3f}")
    d = cur["gather_ratio_256"]
    print(f"    いまの既定 (_GATHER_RATIO_MEASURED[256]) = {d:.3f}"
          f"   ずれ +{(u_far - d) / d * 100:.0f}%")
    print(f"    掃引のゼロ交差 (M3 Max 実測) = {SWEPT_GATHER_CROSSING:.3f}"
          f"   ずれ {(u_far - SWEPT_GATHER_CROSSING) / SWEPT_GATHER_CROSSING * 100:+.0f}%")
    print("    ** 反転条件は「掃引で決めた値と 2 割以上ずれたら式を捨てる」。"
          "既定 0.20 は交差点から安全側に 1 割倒した値なので、"
          "式が当てるべき相手は交差点のほう **")

    # 他の head_dim。**外挿ではなく測った G から出す**
    for hd in sorted((int(k) for k in G), reverse=True):
        if hd == 256:
            continue
        bt = N_KV_HEADS * hd * 2 * 2
        u = C.gather_max_ratio(B, G[str(hd)], L, 25000, bt)
        print(f"    head_dim={hd:3d} の同型モデル (kv=25000) u* = {u:.3f}")

    # PREFILL_GROUP
    pg = C.prefill_group(p["r_knee_rows"], PREFILL_CHUNK, TOP_K, N_EXPERTS)
    pg_lo = C.prefill_group(br[0], PREFILL_CHUNK, TOP_K, N_EXPERTS)
    pg_hi = C.prefill_group(br[1], PREFILL_CHUNK, TOP_K, N_EXPERTS)
    print(f"\n  MLXTURBO_PREFILL_GROUP = 膝({p['r_knee_rows']}行)"
          f" / (2048*{TOP_K}/{N_EXPERTS}) = {pg:.1f}"
          f"   (膝の幅で {pg_lo:.1f}-{pg_hi:.1f})"
          f"   いまの既定 {cur['prefill_group']}")

    # STAGE_EVERY
    se = C.stage_every(L, D)
    print(f"  MLXTURBO_STAGE_EVERY   = L/D = {L * 1e6:.2f}/{D * 1e6:.2f}"
          f" = {se:.2f}   いまの既定 {cur['stage_every']}")

    # fast_qmm.M_MIN。Flash-Next の qkv と lm_head の形で当てる
    print("  fast_qmm.M_MIN         = 1/(1 - FLOP*B/(F*kv_bytes))"
          f"   いまの既定 {cur['qmm_m_min']}")
    for name, (k, n) in {"qkv (2560x2560)": (2560, 2560),
                         "lm_head (2560x248320)": (2560, 248320)}.items():
        wbytes = k * n * 4 / 8 + (k * n / 64) * 4  # 4bit 本体 + scale/bias
        m = C.fast_qmm_m_min(B, F, 2.0 * k * n, wbytes)
        print(f"    {name:24s} M* = {m:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="JSON の書き出し先")
    ap.add_argument("--show", default=None,
                    help="既にある JSON を読んで突き合わせだけ出す")
    args = ap.parse_args()

    if args.show:
        show(json.loads(Path(args.show).read_text()))
        return

    chip = platform.processor()
    try:
        sp = subprocess.run(["system_profiler", "SPHardwareDataType"],
                            capture_output=True, text=True, timeout=20).stdout
        for line in sp.splitlines():
            if "Chip:" in line:
                chip = line.split(":", 1)[1].strip()
    except Exception:
        pass

    print(f"[calibrate] {chip} / mlx {mx.__version__} で原始量を測る")
    t0 = time.perf_counter()

    b_copy, b_read = measure_B()
    print(f"  B  = {b_copy / 1e9:.1f} GB/s (コピー) /"
          f" {b_read / 1e9:.1f} GB/s (読み専)")
    g_by_frac = measure_G(b_copy)
    g = g_by_frac[str(G_FRAC_REF)]
    print(f"  G  = (密度 {G_FRAC_REF}) "
          + ", ".join(f"head_dim {k}: {v:.3f}" for k, v in g.items()))
    L = measure_L()
    print(f"  L  = {L * 1e6:.2f} us/起動")
    F = measure_F()
    print(f"  F  = {F / 1e12:.2f} TFLOPS")
    curve, knee, bracket = measure_r(F)
    shape = ", ".join("{}:{:.2f}".format(c["rows"], c["r"]) for c in curve)
    print(f"  r  = 膝 {knee} 行/エキスパート (挟む 2 点 {bracket}) ({shape})")
    D = measure_D()
    print(f"  D  = {D * 1e6:.2f} us/層 (構築のみ)")

    prof = {
        "machine": {
            "chip": chip,
            "mlx": mx.__version__,
            "measured_at": date.today().isoformat(),
        },
        "primitives": {
            "B_bytes_per_s": b_copy,
            "B_read_only_bytes_per_s": b_read,
            "G_by_head_dim": g,
            "G_frac_ref": G_FRAC_REF,
            "G_by_frac": g_by_frac,
            "L_seconds": L,
            "F_flops": F,
            "r_curve": curve,
            "r_knee_rows": knee,
            "r_knee_bracket": list(bracket),
            "D_seconds_per_layer": D,
            "D_ops_per_layer": _OPS_PER_LAYER,
        },
    }
    print(f"[calibrate] {time.perf_counter() - t0:.1f}s")

    if args.out:
        Path(args.out).write_text(json.dumps(prof, ensure_ascii=False, indent=2))
        print(f"[calibrate] 書き出した: {args.out}")
        print(f"[calibrate] 実行時に効かせるなら MLXTURBO_CALIBRATION={args.out}"
              " (指定が無ければ既定のまま)")
    show(prof)


if __name__ == "__main__":
    main()
