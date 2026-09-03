"""decode/verify 幅 (S=1〜8) の GDN 層の「行列積以外」を、モデルを読まずに測る。

対象は 1 層の並びのうち **前処理 → 再帰 → 出力 norm** の 3 段:

    concat(conv 状態) -> conv1d -> silu -> split -> q/k の rms_norm+スケール
    -> sigmoid(b) -> compute_g          … 前処理 (素だと 9 dispatch)
    -> gated_delta_kernel_with_states   … 再帰 (1 dispatch)
    -> RMSNormGated (rms_norm/astype/sigmoid/mul/astype)  … 出力 norm (素だと 6)

`mlxturbo/kernels/gdn_prework.py` (列ブロック版) と
`mlxturbo/kernels/rms_norm_gated.py` を入れると 3 dispatch になる。
その 3 段の壁時計を素 / 自前で 1 プロセス内 ABBA 交互に測る。

## 3 つのモード

- `--mode count`: **GPU を一切使わない。**遅延グラフを組んで
  `mx.export_to_dot` でプリミティブが作った配列を数え、素 / 自前の
  dispatch 相当の本数を段ごとに出す (`tools/decode_copy_probe.py` と同じ数え方
  で、GDN 1 層だけを見る版)。
- `--mode check`: 素と自前の出力を S ∈ {1,2,3,6} で突き合わせる。
  bf16 の出力は相対 1e-2、fp32 の再帰状態は 1e-5 を判定線にする。
- `--mode bench`: 依存連鎖 N 段を 1 回だけ eval する ABBA。
  `--sets` 組 (既定 36 = GDN の層数) の重みを巡回する。

## 冷キャッシュについて (CLAUDE.md の作法)

GDN 前処理が読む重みは 1 層あたり conv1d 82 KB + A_log/dt_bias/norm 数百 B しか
無く、36 組を巡回しても 3 MB で **100 MB の冷条件には届かない** (これは道具の
不備ではなく「GDN の前処理には量子化行列のような大きい重みが無い」という
事実。`tools/kernel_chain_cost.py` の同じ項目にも同じ警告がある)。

そこで `--with-proj` を用意した。本番の並びどおり **in_proj 4 本と out_proj を
連鎖に含める**と、1 層あたり 4bit/gs64 の量子化重みが約 17 MB になり、36 組で
620 MB を巡回する = 実モデルと同じ冷条件になる。判定は 2 本立てで読む:

- `block` (既定、3 段だけ): 判定線 **S=1 で素の 0.7 倍以下**。
- `layer` (`--with-proj`): 冷条件での確認。**符号が block と一致すること**
  (行列積が入るぶん比は 1 に寄るので、比の絶対値では判定しない)。

    tools/biglock.sh .venv/bin/python tools/gdn_decode_micro.py \\
        --mode all --out bench/results/gdn-decode-micro.json

最後に os._exit(0) で落ちる (MLX の Metal 終了処理で固まる前例があるため。
`tools/micro_kernel_latency.py` と同じ)。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 実寸 (~/models/ddalcu-mlxlm/config.json の text_config)
HIDDEN = 2560
N_K = 16
N_V = 48
DK = 128
DV = 128
KEY_DIM = N_K * DK          # 2048
VALUE_DIM = N_V * DV        # 6144
CONV_DIM = 2 * KEY_DIM + VALUE_DIM   # 10240
CONV_KERNEL = 4
RMS_EPS = 1e-6
QBITS = 4                   # linear_attn.* は全部 4bit / group_size 64
QGROUP = 64

N_SETS = 36                 # GDN の層数
N_CHAIN = 120
N_PAIRS = 3
WARMUP = 12


# --------------------------------------------------------------- 重みと素の並び


def _quant_linear(n_out, n_in, dtype):
    import mlx.core as mx

    w = mx.random.normal((n_out, n_in)).astype(dtype)
    wq, sc, bi = mx.quantize(w, group_size=QGROUP, bits=QBITS)
    return (wq, sc, bi)


def _qmm(x, packed):
    import mlx.core as mx

    wq, sc, bi = packed
    return mx.quantized_matmul(
        x, wq, sc, bi, transpose=True, group_size=QGROUP, bits=QBITS
    )


class LayerWeights:
    """GDN 1 層ぶんの重み。`with_proj=False` なら射影は作らない (メモリ節約)。"""

    def __init__(self, dtype, with_proj: bool):
        import mlx.core as mx
        import mlx.nn as nn

        self.conv1d = nn.Conv1d(
            CONV_DIM, CONV_DIM, kernel_size=CONV_KERNEL, groups=CONV_DIM, bias=False
        )
        self.conv1d.weight = mx.random.normal((CONV_DIM, CONV_KERNEL, 1)).astype(dtype) * 0.5
        self.conv_w = self.conv1d.weight
        # 実モデルの A_log/dt_bias/norm.weight は bf16 (safetensors で確認)
        self.A_log = mx.random.normal((N_V,)).astype(dtype)
        self.dt_bias = mx.random.normal((N_V,)).astype(dtype)
        self.norm_weight = (1.0 + 0.02 * mx.random.normal((DV,))).astype(dtype)
        leaves = [self.conv_w, self.A_log, self.dt_bias, self.norm_weight]
        self.proj = None
        if with_proj:
            self.proj = {
                "qkv": _quant_linear(CONV_DIM, HIDDEN, dtype),
                "z": _quant_linear(VALUE_DIM, HIDDEN, dtype),
                "b": _quant_linear(N_V, HIDDEN, dtype),
                "a": _quant_linear(N_V, HIDDEN, dtype),
                "out": _quant_linear(HIDDEN, VALUE_DIM, dtype),
            }
            for p in self.proj.values():
                leaves.extend(p)
        mx.eval(leaves)
        self.nbytes = sum(t.nbytes for t in leaves)


def project_in(w: LayerWeights, x):
    """`GatedDeltaNet._project_in` の写し (4 本の量子化射影)。"""
    return (_qmm(x, w.proj["qkv"]), _qmm(x, w.proj["z"]),
            _qmm(x, w.proj["b"]), _qmm(x, w.proj["a"]))


def plain_block(w: LayerWeights, mixed_qkv, z, b, a, conv_state, rec_state):
    """素の並び (前処理 → 再帰 → 出力 norm)。

    `mlxturbo/_vendor/qwen4_exp.py` の `GatedDeltaNet.__call__` と
    `mlxturbo/spec_flash.py` の capture 版 `gdn()` の mask なし分岐そのまま。
    再帰は本番の検証フォワードと同じ `gated_delta_update_with_states`
    (位置ごとの状態も返す版) を使う。

    戻り値は `(out, conv_state_out, states_all)`。`out` は out_proj 手前の
    `self.norm(out, z)` の結果 (B, S, value_dim)。
    """
    import mlx.core as mx
    import mlx.nn as nn

    from mlxturbo.kernels.gated_delta_states import gated_delta_update_with_states

    B, S, _ = mixed_qkv.shape
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    conv_state_out = mx.contiguous(conv_input[:, -(CONV_KERNEL - 1):, :])
    conv_out = nn.silu(w.conv1d(conv_input))

    q, k, v = mx.split(conv_out, [KEY_DIM, 2 * KEY_DIM], axis=-1)
    q = q.reshape(B, S, N_K, DK)
    k = k.reshape(B, S, N_K, DK)
    v = v.reshape(B, S, N_V, DV)

    inv_scale = DK**-0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, RMS_EPS)
    k = inv_scale * mx.fast.rms_norm(k, None, RMS_EPS)

    out, states_all = gated_delta_update_with_states(
        q, k, v, a, b, w.A_log, w.dt_bias, rec_state, None
    )
    # RMSNormGated.__call__ (activation="sigmoid") の写し
    normed = mx.fast.rms_norm(out, w.norm_weight, RMS_EPS)
    g = mx.sigmoid(z.astype(mx.float32))
    y = (g * normed.astype(mx.float32)).astype(out.dtype)
    return y.reshape(B, S, -1), conv_state_out, states_all


def fused_block(w: LayerWeights, mixed_qkv, z, b, a, conv_state, rec_state,
                block=None, rows_per_tg=1):
    """自前の 3 本 (前処理カーネル → 再帰カーネル → 出力 norm カーネル)。"""
    from mlxturbo.kernels.gated_delta_states import gated_delta_update_with_states_gb
    from mlxturbo.kernels.gdn_prework import fused_gdn_prework
    from mlxturbo.kernels.rms_norm_gated import rms_norm_gated

    B, S, _ = mixed_qkv.shape
    q, k, v, g, beta, conv_state_out = fused_gdn_prework(
        mixed_qkv, conv_state, w.conv_w, a, b, w.A_log, w.dt_bias,
        N_K, N_V, DK, DV, KEY_DIM, VALUE_DIM, RMS_EPS, block=block,
    )
    out, states_all = gated_delta_update_with_states_gb(
        q, k, v, g, beta, rec_state, None
    )
    y = rms_norm_gated(out, w.norm_weight, z, RMS_EPS, "sigmoid",
                       rows_per_tg=rows_per_tg)
    return y.reshape(B, S, -1), conv_state_out, states_all


# --------------------------------------------------------------------- 入力


def make_inputs(S, dtype, with_proj):
    import mlx.core as mx

    B = 1
    xs = {
        "conv_state": mx.random.normal((B, CONV_KERNEL - 1, CONV_DIM)).astype(dtype),
        "rec_state": (0.1 * mx.random.normal((B, N_V, DV, DK))).astype(mx.float32),
    }
    if with_proj:
        xs["x"] = mx.random.normal((B, S, HIDDEN)).astype(dtype)
    else:
        xs["mixed_qkv"] = mx.random.normal((B, S, CONV_DIM)).astype(dtype)
        xs["z"] = mx.random.normal((B, S, N_V, DV)).astype(dtype)
        xs["b"] = mx.random.normal((B, S, N_V)).astype(dtype)
        xs["a"] = mx.random.normal((B, S, N_V)).astype(dtype)
    mx.eval(list(xs.values()))
    return xs


def _unpack(w, xs, S):
    """`--with-proj` のとき射影を通して (mixed_qkv, z, b, a) を作る。"""
    if "x" in xs:
        mixed_qkv, z, b, a = project_in(w, xs["x"])
        return mixed_qkv, z.reshape(1, S, N_V, DV), b, a
    return xs["mixed_qkv"], xs["z"], xs["b"], xs["a"]


# ----------------------------------------------------------------- 1) count


# 形だけを変える op (view で済み、Metal の起動を出さない)。
# `Contiguous` は入力が既に連続なら起動を出さない (B=1 の decode はその側。
# `tools/decode_copy_probe.py` の実測で確認済み)。
_FREE_OPS = {
    "Reshape", "Broadcast", "Full", "Squeeze", "ExpandDims",
    "Flatten", "Unflatten", "Contiguous", "Split", "Slice",
}

_NODE_RE = re.compile(r'^\{ (\d+) \[label ="([^"]*)"')
_EDGE_RE = re.compile(r'^(\d+) -> "')


def _graph_outputs(arrays):
    """`arrays` から辿れるプリミティブを数える。

    返すのは ``(プリミティブ本数の Counter, 出力配列本数の Counter)``。
    dispatch に近いのは**プリミティブ本数**のほう (1 プリミティブ = 1 起動。
    `mx.split` や自前カーネルのように 1 起動で複数の配列を返すものがある)。
    `tools/decode_copy_probe.py` は出力配列で数えているが、あちらは
    `mx.compile` の `Compiled` が層をまたいで同じインスタンスになる事情が
    あるため (ここは 1 層だけなので素直にプリミティブを数えられる)。
    """
    import mlx.core as mx

    buf = io.StringIO()
    mx.export_to_dot(buf, *arrays)
    names, edges = {}, Counter()
    for line in buf.getvalue().splitlines():
        line = line.strip()
        m = _NODE_RE.match(line)
        if m:
            names[m.group(1)] = m.group(2)
            continue
        m = _EDGE_RE.match(line)
        if m:
            edges[m.group(1)] += 1
    prims, outs = Counter(), Counter()
    for nid, name in names.items():
        prims[name] += 1
        outs[name] += edges.get(nid, 0)
    return prims, outs


def run_count(widths, dtype, with_proj):
    """GPU を使わずに、素 / 自前のグラフに現れる op を数える。

    `mx.zeros` の遅延配列で組むので eval しない = GPU は 1 度も触らない
    (走行中の A/B と並走しても影響しない)。
    """
    import mlx.core as mx
    import mlx.nn as nn

    rows = []
    for S in widths:
        B = 1
        conv1d = nn.Conv1d(CONV_DIM, CONV_DIM, kernel_size=CONV_KERNEL,
                            groups=CONV_DIM, bias=False)
        conv1d.weight = mx.zeros((CONV_DIM, CONV_KERNEL, 1), dtype=dtype)

        class _W:
            pass

        w = _W()
        w.conv1d = conv1d
        w.conv_w = conv1d.weight
        w.A_log = mx.zeros((N_V,), dtype=dtype)
        w.dt_bias = mx.zeros((N_V,), dtype=dtype)
        w.norm_weight = mx.zeros((DV,), dtype=dtype)

        mixed_qkv = mx.zeros((B, S, CONV_DIM), dtype=dtype)
        z = mx.zeros((B, S, N_V, DV), dtype=dtype)
        bb = mx.zeros((B, S, N_V), dtype=dtype)
        aa = mx.zeros((B, S, N_V), dtype=dtype)
        conv_state = mx.zeros((B, CONV_KERNEL - 1, CONV_DIM), dtype=dtype)
        rec_state = mx.zeros((B, N_V, DV, DK), dtype=mx.float32)

        # 入力そのもの (mx.zeros は Full プリミティブを 1 本ずつ作る) は
        # 素・自前の両方に等しく乗るので、基準として引いてから読む
        inputs = [mixed_qkv, z, bb, aa, conv_state, rec_state,
                  w.conv_w, w.A_log, w.dt_bias, w.norm_weight]
        base, _ = _graph_outputs(inputs)
        pl = plain_block(w, mixed_qkv, z, bb, aa, conv_state, rec_state)
        fu = fused_block(w, mixed_qkv, z, bb, aa, conv_state, rec_state)
        cp, _ = _graph_outputs(list(pl))
        cf, _ = _graph_outputs(list(fu))
        dp, df = cp - base, cf - base

        def split(c):
            hot = Counter({k: v for k, v in c.items() if k not in _FREE_OPS})
            free = Counter({k: v for k, v in c.items() if k in _FREE_OPS})
            return hot, free

        hp, fp_ = split(dp)
        hf, ff = split(df)
        rows.append({
            "S": S,
            "plain_dispatch": sum(hp.values()),
            "fused_dispatch": sum(hf.values()),
            "plain_ops": dict(hp.most_common()),
            "fused_ops": dict(hf.most_common()),
            "plain_free": dict(fp_.most_common()),
            "fused_free": dict(ff.most_common()),
        })
    return rows


# ----------------------------------------------------------------- 2) check


def _prework_plain(w, mixed_qkv, b, a):
    """前処理だけの素の op 列 (`fused_gdn_prework` と同じ 6 本を返す)。"""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import compute_g

    B, S, _ = mixed_qkv.shape
    conv_input = mx.concatenate([w.conv_state_dbg, mixed_qkv], axis=1)
    cs = mx.contiguous(conv_input[:, -(CONV_KERNEL - 1):, :])
    conv_out = nn.silu(w.conv1d(conv_input))
    q, k, v = mx.split(conv_out, [KEY_DIM, 2 * KEY_DIM], axis=-1)
    inv = DK**-0.5
    q = (inv**2) * mx.fast.rms_norm(q.reshape(B, S, N_K, DK), None, RMS_EPS)
    k = inv * mx.fast.rms_norm(k.reshape(B, S, N_K, DK), None, RMS_EPS)
    return (q, k, v.reshape(B, S, N_V, DV),
            compute_g(w.A_log, a, w.dt_bias), mx.sigmoid(b), cs)


def run_check(widths, dtype, block, rows_per_tg):
    """素との一致。ブロック全体だけでなく**前処理カーネルの 6 出力を 1 本ずつ**
    見る。ブロック全体だけだと、どの出力が食い違ったのかが分からない
    (実際 g だけが 48 個中 13 個ずれていた事例を、この内訳で見つけた)。"""
    import mlx.core as mx

    from mlxturbo.kernels.gdn_prework import fused_gdn_prework

    out = []
    w = LayerWeights(dtype, with_proj=False)
    for S in widths:
        xs = make_inputs(S, dtype, with_proj=False)
        args = (w, xs["mixed_qkv"], xs["z"], xs["b"], xs["a"],
                xs["conv_state"], xs["rec_state"])
        yp, csp, stp = plain_block(*args)
        yf, csf, stf = fused_block(*args, block=block, rows_per_tg=rows_per_tg)

        w.conv_state_dbg = xs["conv_state"]
        pre_p = _prework_plain(w, xs["mixed_qkv"], xs["b"], xs["a"])
        pre_f = fused_gdn_prework(
            xs["mixed_qkv"], xs["conv_state"], w.conv_w, xs["a"], xs["b"],
            w.A_log, w.dt_bias, N_K, N_V, DK, DV, KEY_DIM, VALUE_DIM,
            RMS_EPS, block=block,
        )
        mx.eval(yp, csp, stp, yf, csf, stf, *pre_p, *pre_f)

        def rel(x, y):
            x32, y32 = x.astype(mx.float32), y.astype(mx.float32)
            den = mx.maximum(mx.abs(x32), mx.abs(y32)) + 1e-6
            return float(mx.max(mx.abs(x32 - y32) / den))

        pre = {}
        for name, xp, xf in zip(("q", "k", "v", "g", "beta", "conv_state"),
                                 pre_p, pre_f):
            pre[name] = {
                "rel": rel(xp, xf),
                "n_diff": int(mx.sum(xp.astype(mx.float32)
                                     != xf.astype(mx.float32)).item()),
                "n": int(xp.size),
            }
        row = {
            "S": S,
            "rel_out": rel(yp, yf),
            "rel_conv_state": rel(csp, csf),
            "rel_states_fp32": rel(stp, stf),
            "exact_conv_state": bool(mx.all(csp == csf).item()),
            "exact_out": bool(mx.all(yp == yf).item()),
            "prework": pre,
        }
        row["pass"] = (row["rel_out"] <= 1e-2
                       and row["rel_conv_state"] <= 1e-2
                       and row["rel_states_fp32"] <= 1e-5)
        out.append(row)
    return out


# ----------------------------------------------------------------- 3) bench


def _chain(step_fn, carry, n):
    """依存連鎖 n 段を 1 度も eval せずに組んでから 1 回だけ eval する。"""
    import mlx.core as mx

    leaves = []
    t0 = time.perf_counter()
    for _ in range(n):
        carry, extra = step_fn(carry)
        leaves.extend(extra)
    t1 = time.perf_counter()
    mx.eval(*(list(carry) + leaves))
    t2 = time.perf_counter()
    return (t1 - t0), (t2 - t1)


def _make_step(sets, xs, S, dtype, fused, with_proj, block, rows_per_tg):
    """連鎖 1 段。carry = (link, conv_state, rec_state)。

    `link` は次段の入力 (with_proj なら hidden の x、そうでなければ mixed_qkv)。
    出力 norm の結果からしか作らないので、**3 段すべてが次段の本当の祖先**に
    なる (どれか 1 つでも刈られると素の側だけ速く見える)。
    """
    import mlx.core as mx

    idx = {"i": 0}

    def step(carry):
        link, conv_state, rec_state = carry
        w = sets[idx["i"] % len(sets)]
        idx["i"] += 1
        if with_proj:
            mixed_qkv, z, b, a = project_in(w, link)
            z = z.reshape(1, S, N_V, DV)
        else:
            mixed_qkv, z, b, a = link, xs["z"], xs["b"], xs["a"]
        fn = fused_block if fused else plain_block
        kw = {"block": block, "rows_per_tg": rows_per_tg} if fused else {}
        y, conv_state_out, states_all = fn(
            w, mixed_qkv, z, b, a, conv_state, rec_state, **kw
        )
        next_rec = states_all[:, -1]
        if with_proj:
            nxt = _qmm(y, w.proj["out"])
        else:
            # y は (B,S,VALUE_DIM)。conv_dim に詰め直して次段の mixed_qkv にする
            # (KEY_DIM*2 + VALUE_DIM == CONV_DIM。A/B で同じ 2 op なので比に効かない)
            nxt = mx.concatenate([y, y[..., : 2 * KEY_DIM]], axis=-1)
        return (nxt, conv_state_out, next_rec), []

    return step


def _init_carry(xs, with_proj):
    return (xs["x"] if with_proj else xs["mixed_qkv"],
            xs["conv_state"], xs["rec_state"])


def _warm(step, carry, n):
    import mlx.core as mx

    for _ in range(n):
        carry, _extra = step(carry)
        mx.eval(*carry)
    return carry


def run_bench(widths, dtype, n_sets, n_chain, n_pairs, with_proj,
              blocks, rows_list):
    import mlx.core as mx

    sets = [LayerWeights(dtype, with_proj) for _ in range(n_sets)]
    total_mb = sum(s.nbytes for s in sets) / 1e6
    print(f"[gdn_decode_micro] 重み {n_sets} 組 = {total_mb:.1f} MB を巡回"
          + ("" if total_mb >= 100 else "  ** 100 MB 未満: 冷条件は再現できていない **"))

    results = []
    for S in widths:
        xs = make_inputs(S, dtype, with_proj)
        for block, rows_per_tg in [(b, r) for b in blocks for r in rows_list]:
            fused_step = _make_step(sets, xs, S, dtype, True, with_proj,
                                     block, rows_per_tg)
            plain_step = _make_step(sets, xs, S, dtype, False, with_proj,
                                     None, rows_per_tg)
            _warm(fused_step, _init_carry(xs, with_proj), WARMUP)
            _warm(plain_step, _init_carry(xs, with_proj), WARMUP)

            f_us, p_us, f_build, p_build = [], [], [], []
            # 焼き入れの 1 往復 (毎回いちばん高く出る) は捨てる
            for _round in range(n_pairs + 1):
                _burn = _round == 0
                # ABBA (1 プロセス内で交互、CLAUDE.md の作法)
                for who in ("A", "B", "B", "A"):
                    step = fused_step if who == "A" else plain_step
                    bs, es = _chain(step, _init_carry(xs, with_proj), n_chain)
                    if _burn:
                        continue
                    (f_us if who == "A" else p_us).append(es / n_chain * 1e6)
                    (f_build if who == "A" else p_build).append(bs / n_chain * 1e6)
            fm, pm = statistics.median(f_us), statistics.median(p_us)
            row = {
                "S": S, "block": block, "rows_per_tg": rows_per_tg,
                "fused_us": round(fm, 2), "plain_us": round(pm, 2),
                "ratio": round(fm / pm, 3),
                "fused_build_us": round(statistics.median(f_build), 2),
                "plain_build_us": round(statistics.median(p_build), 2),
                "fused_all": [round(x, 2) for x in f_us],
                "plain_all": [round(x, 2) for x in p_us],
            }
            results.append(row)
            print(f"  S={S} block={block} rows/tg={rows_per_tg}: "
                  f"自前 {fm:8.2f} / 素 {pm:8.2f} us  比 {fm / pm:.3f}")
    return results, total_mb


# ----------------------------------------------------------------------- CLI


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "count", "check", "bench"])
    ap.add_argument("--s", default="1,3", help="幅 S (カンマ区切り)")
    ap.add_argument("--check-s", default="1,2,3,6")
    ap.add_argument("--sets", type=int, default=N_SETS)
    ap.add_argument("--n-chain", type=int, default=N_CHAIN)
    ap.add_argument("--pairs", type=int, default=N_PAIRS)
    ap.add_argument("--blocks", default="256",
                    help="前処理カーネルの列ブロック幅 (カンマ区切りで掃引)")
    ap.add_argument("--rows-per-tg", default="1",
                    help="出力 norm カーネルの 1 threadgroup あたりの行数"
                         " (カンマ区切りで掃引)")
    ap.add_argument("--with-proj", action="store_true",
                    help="in_proj/out_proj も連鎖に含める (冷条件、620 MB 巡回)")
    ap.add_argument("--out", default=None)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    import mlx.core as mx

    dtype = mx.bfloat16
    widths = [int(x) for x in args.s.split(",")]
    blocks = [int(x) for x in args.blocks.split(",")]
    rows_list = [int(x) for x in str(args.rows_per_tg).split(",")]
    res: dict = {"config": vars(args)}

    if args.mode in ("all", "count"):
        res["count"] = run_count(widths, dtype, args.with_proj)
        print("\n=== op 数 (GDN 1 層の前処理+再帰+出力 norm、GPU 不使用) ===")
        for r in res["count"]:
            print(f"  S={r['S']}: 素 {r['plain_dispatch']} 本 / 自前 "
                  f"{r['fused_dispatch']} 本  (view で済む op は除く)")
            print(f"    素  : {r['plain_ops']}")
            print(f"    自前: {r['fused_ops']}")
            print(f"    (view) 素 {r['plain_free']} / 自前 {r['fused_free']}")

    if args.mode in ("all", "check"):
        res["check"] = run_check([int(x) for x in args.check_s.split(",")],
                                  dtype, blocks[0], rows_list[0])
        print("\n=== 素との一致 ===")
        for r in res["check"]:
            print(f"  S={r['S']}: out {r['rel_out']:.3e} / conv_state "
                  f"{r['rel_conv_state']:.3e} / states(fp32) "
                  f"{r['rel_states_fp32']:.3e}  -> {'OK' if r['pass'] else 'NG'}")
            pre = "  ".join(
                f"{k}={v['rel']:.1e}({v['n_diff']}/{v['n']})"
                for k, v in r["prework"].items())
            print(f"    前処理の内訳: {pre}")

    if args.mode in ("all", "bench"):
        print("\n=== 冷の連鎖 ABBA ===")
        rows, mb = run_bench(widths, dtype, args.sets, args.n_chain,
                              args.pairs, args.with_proj, blocks, rows_list)
        res["bench"] = rows
        res["weights_mb"] = round(mb, 1)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n書いた: {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
