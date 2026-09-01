"""GatedDeltaNet の再帰を prefill 幅だけブロック化スキャンで解く (段 P3)。

`docs/research/KERNEL-PROGRAM.md` の「段 P3 の見送りを撤回する」の決着そのもの。
mlx_lm の `_make_gated_delta_kernel` は `for (int t = 0; t < T; ++t)` の逐次
スキャンで、prefill (T=2048) では threadgroup ごとに 2048 回の直列反復を回す。
1 反復が使えるのはスカラー FMA と simd_sum だけなので、行列積のユニットに
載せ替えれば FLOP を増やしても勝てる、というのが動機。

**2026-09-02 の実測では勝てていない。下の「測った結果」を先に読むこと。**

## 何をしているか

再帰は 1 位置あたり

    S_t = g_t S_{t-1} + delta_t k_t^T,   delta_t = beta_t (v_t - (g_t S_{t-1}) k_t),
    y_t = S_t q_t

で、S は [Dv, Dk]。**dv 行どうしは独立**なので、時間方向だけが直列になっている。
長さ C のブロックに切り、ブロック先頭の状態を S_0、ブロック内の累積ゲートを
A_t = prod_{s<=t} g_s と書くと

    S_t = A_t S_0 + sum_{j<=t} (A_t/A_j) delta_j k_j^T

が成り立つ。これを delta について解くと、C x C の**単位下三角の連立**になる:

    (I + W) D = beta * (V - Khat S_0^T),
        W[t,j] = beta_t (A_t/A_j) (k_j . k_t)   (j < t、それ以外 0)
        Khat_t = A_t k_t

出力と次ブロックの状態はそのまま行列積で書ける:

    Y   = Qhat S_0^T + P D,   P[t,j] = (A_t/A_j)(q_t . k_j)  (j <= t)
    S_C = A_C S_0 + D^T Kbar, Kbar_j = (A_C/A_j) k_j

**直列に残るのはブロック間 (T/C 回) だけ**で、ブロック内は全部 matmul。
T=2048 / C=64 なら直列は 32 回になる (逐次版は 2048 回)。

## 数値

A_t/A_j は j <= t で必ず 1 以下だが、A_t や 1/A_j を単体で作ると桁が飛ぶ。
ここでは g を対数で持ち (`log_g = -exp(A_log) * softplus(a + dt_bias)`、
`compute_g` の指数そのものなので log を取り直す誤差が無い)、
`exp(cum_t - cum_j)` の形でしか指数に戻さない。**外に出る係数は全部 1 以下**
(A_t = exp(cum_t) <= 1、A_C/A_j <= 1) なので、下に潰れることはあっても
飛ぶことはない。累算は fp32。

逐次版とはビット一致しない (加算順が変わる)。合成テンソルでの突き合わせは
`tools/verify_gdn_blocked.py`。

## 単位下三角の逆行列

W は狭義下三角なので (I+W)^{-1} = sum_i (-W)^i が有限和で、
prod_j (I + (-W)^{2^j}) に畳める。C=64 をそのまま畳むと 64^3 の matmul が
10 本になって本体と同じ桁の費用になるので、**sub x sub (既定 32) の対角
ブロックだけ畳んで、ブロック行の前進代入で組み上げる。**前進代入は 1 段ごとに
C x C の連結を実体化するので、段数 (C/sub - 1) を増やすと逆に高くつく
(C=64 の実測: sub=16 で 1.90ms、sub=32 で 0.52ms)。

## prefill 幅限定

`MIN_T` (既定 64) 未満、マスク付き (バッチの右パディング)、ベクトルゲート
(`g.ndim == 4`) は対象外で、呼び手は既存の逐次経路にそのまま落ちる。
decode/verify 幅ではブロック化の下ごしらえ (C x C 行列と逆行列) が本体より
重くなるので、そこを取りに行かないのは相手 (mlx-serve の
`gated_delta_blocked_seq`) と同じ切り方。

## 測った結果 (2026-09-02、M3 Max 40 コア、GDN 1 層 S=2048、1 プロセス内で交互)

    GDN __call__ 全体 (逐次)        33.0 ms
      うち再帰スキャン               6.7 ms   <- ブロック化が触れるのはここだけ
      射影 4 本 (_project_in)       17.5 ms
      out_proj + RMSNormGated        8.2 ms
    GDN __call__ 全体 (ブロック化)   37.6 ms  (+14%)
      うちブロック化スキャン C=64    11.2 ms  (逐次の 1.68 倍)

**負けている。**理由は 2 つで、どちらも MLX の op 単位という実装形態から来る:

- 逐次カーネルは状態 (Dv x Dk = 64KB) をレジスタに置いたまま 2048 歩進むので、
  状態のグローバル往復がゼロ。ブロック化を MLX の op で書くと、C x C の行列と
  D をブロックごとにグローバルへ書いて読み直す
- ブロック間の直列 (T/C = 32 回) が残るので、その 32 回はバッチ B*Hv=48 の
  小さな行列積になる。実測で 48 x 128x128x128 は fp32 3.7 / bf16 6.9 TFLOPS
  止まりで、密の 11.5 / 12.9 TFLOPS に届かない

**勝つには融合 Metal カーネル (simdgroup_matrix) が要る。**相手 (mlx-serve の
`gated_delta_blocked_seq`) が 2 倍を出しているのはそちら。op 単位の書き換えでは
この形は取れない。

なお、仮にスキャンをゼロにしても GDN 部品の 20% しか消えない (上の内訳)。
「GDN 1034ms の半分が取れる」という見立ては、`prefill_anatomy` の `gdn` が
**射影込みの `__call__` 全体**を包んでいることの読み違いだった。

## 既定 off

`MLXTURBO_GDN_BLOCKED=1` (`mlxturbo.fused.enable_gdn_blocked_kernel`) の
ときだけ動く。上のとおり既定で有効化する理由は無い。
在庫として残してあるのは、融合カーネルを書くときの**参照実装と正しさの基準**
としてで、`tools/verify_gdn_blocked.py` が逐次版との突き合わせを持っている。
in-model で測り直すなら `tools/decode_ab.py --knob gdn-blocked` (prefill に
効く knob なので `--prefill-once` は使えない)。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from . import _fire

# prefill と decode/verify を分ける T の下限。これ未満はブロック 1 個に
# 収まってしまい、下ごしらえ (C x C の生成と逆行列) だけが残る。
MIN_T = int(os.environ.get("MLXTURBO_GDN_BLOCKED_MIN_T", "64"))

# ブロック長 C。直列回数 T/C と、ブロック内の C^2 項の費用の折り合い。
# 1 位置あたりの積和は 2C(Dk+Dv) + 3 Dk Dv で、C=64 / Dk=Dv=128 なら
# 逐次版 (4 Dk Dv) の約 1.25 倍。**FLOP は増えるが行き先が変わる。**
BLOCK = int(os.environ.get("MLXTURBO_GDN_BLOCK", "64"))

# 逆行列の対角ブロック長。BLOCK の約数であること。前進代入の段ごとに
# C x C の連結 (実体化) が要るので、段数 (BLOCK/SUB_BLOCK - 1) を増やすと
# すぐ高くつく。C=64 の実測は sub=16 で 1.90ms、sub=32 で 0.52ms。
SUB_BLOCK = int(os.environ.get("MLXTURBO_GDN_SUBBLOCK", "32"))

_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    """同じ理由の見送りを 1 度だけ知らせる (黙って落ちないため)。"""
    if key in _warned:
        return
    _warned.add(key)
    print(f"[mlxturbo] GDN ブロック化スキャン: {msg}")


def eligible(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    beta_src: mx.array,
    state: Optional[mx.array],
    mask: Optional[mx.array],
    block: Optional[int] = None,
) -> bool:
    """この呼び出しでブロック化スキャンを使えるか。

    外れたら呼び手は既存の逐次経路 (`gated_delta_update`) にそのまま落ちる。
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        _warn_once("gpu", "GPU が既定デバイスでないので使わない")
        return False
    if mask is not None:
        # バッチの右パディング。ブロック内で「状態を進めない位置」を作る形に
        # なり、下三角の連立が崩れる。ここは取りに行かない
        _warn_once("mask", "mask 付き (バッチの右パディング) は対象外")
        return False
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        _warn_once("shape_qkv", "q/k/v の形が (B,T,H,D) でないか q と k の形が揃っていない")
        return False
    B, T, Hk, Dk = q.shape
    if v.shape[:2] != (B, T):
        _warn_once("shape_v", "v の先頭 2 軸が (B, T) と揃っていない")
        return False
    Hv, Dv = v.shape[2:]
    if Hv < Hk or Hv % Hk != 0:
        _warn_once("gqa", f"Hv={Hv} が Hk={Hk} の倍数でない (GQA 前提が崩れる)")
        return False
    if beta_src.shape != (B, T, Hv):
        _warn_once("shape_beta", "beta_src の形が (B, T, Hv) でない")
        return False
    if T < MIN_T:
        _warn_once(
            "min_t",
            f"T={T} は decode/verify 幅 (MIN_T={MIN_T} 未満) なので使わない "
            "(下ごしらえの費用がブロック化の得を上回る)",
        )
        return False
    if state is not None and state.dtype != mx.float32:
        _warn_once("state_dtype", "state が fp32 でない")
        return False
    if state is not None and state.shape != (B, Hv, Dv, Dk):
        _warn_once("state_shape", "state の形が (B, Hv, Dv, Dk) でない")
        return False
    c = block or BLOCK
    if c < 8 or c & (c - 1):
        _warn_once("block", f"ブロック長 {c} は 8 以上の 2 冪でないので使わない")
        return False
    return True


def _unit_lower_inv_dense(w: mx.array) -> mx.array:
    """狭義下三角 ``w`` に対する ``(I + w)^{-1}`` を倍々の積で作る。

    ``(I+w)^{-1} = sum_i (-w)^i`` は ``w^n = 0`` で打ち切れる有限和で、
    ``prod_{j} (I + x^{2^j})`` (x = -w) に畳める。n=16 なら squaring 3 回と
    積 3 回の計 6 本。
    """
    n = w.shape[-1]
    x = -w
    r = x + mx.eye(n, dtype=w.dtype)
    p = 1
    while 2 * p < n:
        x = x @ x
        r = r + r @ x
        p *= 2
    return r


def _unit_lower_inv(w: mx.array, sub: int) -> mx.array:
    """``(I + w)^{-1}`` (w は狭義下三角、[..., C, C])。

    C をそのまま倍々にすると C^3 の matmul が log2(C)*2 本並ぶので、
    ``sub`` 角の対角ブロックだけ倍々で潰し、ブロック行の前進代入
    ``R_i,:i = -A_i (W_i,:i R_:i,:i)`` で組み上げる。直列は nb-1 段。
    """
    c = w.shape[-1]
    if c <= sub or c % sub:
        return _unit_lower_inv_dense(w)
    nb = c // sub
    diag = mx.stack(
        [w[..., i * sub : (i + 1) * sub, i * sub : (i + 1) * sub] for i in range(nb)],
        axis=0,
    )
    ai = _unit_lower_inv_dense(diag)  # [nb, ..., sub, sub]
    r = ai[0]
    for i in range(1, nb):
        wrow = w[..., i * sub : (i + 1) * sub, : i * sub]
        row = -(ai[i] @ (wrow @ r))
        zero = mx.zeros(r.shape[:-1] + (sub,), dtype=r.dtype)
        r = mx.concatenate(
            [
                mx.concatenate([r, zero], axis=-1),
                mx.concatenate([row, ai[i]], axis=-1),
            ],
            axis=-2,
        )
    return r


def gated_delta_blocked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    log_g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    block: Optional[int] = None,
) -> Tuple[mx.array, mx.array]:
    """ブロック化スキャン本体。

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v:    [B, T, Hv, Dv]
      - log_g, beta: [B, T, Hv] (log_g は log g、必ず 0 以下)
      - state: [B, Hv, Dv, Dk] (fp32) or None
    Returns:
      - y: [B, T, Hv, Dv] (q と同じ dtype)
      - state_out: [B, Hv, Dv, Dk] (fp32)
    """
    out_dtype = q.dtype
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    c = block or BLOCK
    bh = B * Hv

    if (rep := Hv // Hk) > 1:
        q = mx.repeat(q, rep, -2)
        k = mx.repeat(k, rep, -2)
    log_g = log_g.astype(mx.float32)
    beta = beta.astype(mx.float32)

    # 末尾を C の倍数まで詰める。log_g=0 (g=1)、beta=0 なので delta=0 で
    # 状態は動かず、q=0 なので出力も 0 -- 実位置の結果は変わらない
    pad = (-T) % c
    if pad:
        q = mx.pad(q, [(0, 0), (0, pad), (0, 0), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, pad), (0, 0), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, pad), (0, 0), (0, 0)])
        log_g = mx.pad(log_g, [(0, 0), (0, pad), (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, pad), (0, 0)])
    tp = T + pad
    nc = tp // c

    def _blocks(x, last):
        # [B, T, Hv, X] -> [nc, B*Hv, C, X]。**並べ替えは入力の dtype のまま
        # やって、fp32 化はそのあと。**先に fp32 にすると倍の幅を並べ替える
        x = x.reshape(B, nc, c, Hv, last).transpose(1, 0, 3, 2, 4)
        return x.reshape(nc, bh, c, last).astype(mx.float32)

    qb = _blocks(q, Dk)
    kb = _blocks(k, Dk)
    vb = _blocks(v, Dv)
    lg = log_g.reshape(B, nc, c, Hv).transpose(1, 0, 3, 2).reshape(nc, bh, c)
    bt = beta.reshape(B, nc, c, Hv).transpose(1, 0, 3, 2).reshape(nc, bh, c)

    # ブロック内の累積 log ゲート (t を含む)。単調に減る
    cum = mx.cumsum(lg, axis=-1)                       # [nc, bh, c]
    diff = cum[..., :, None] - cum[..., None, :]       # [t, j] = cum_t - cum_j
    tri = mx.tril(mx.ones((c, c), dtype=mx.float32))   # j <= t
    stri = mx.tril(mx.ones((c, c), dtype=mx.float32), k=-1)
    # j <= t では diff <= 0。上三角は exp を通さずに 0 で潰す
    dec = tri * mx.exp(mx.minimum(diff, 0.0))
    diff = None

    kk = mx.matmul(kb, kb.swapaxes(-1, -2))            # [t, j] = k_t . k_j
    w = (stri * dec * kk) * bt[..., :, None]
    kk = None
    tinv = _unit_lower_inv(w, SUB_BLOCK)
    w = None
    p = dec * mx.matmul(qb, kb.swapaxes(-1, -2))       # tri は dec に入っている

    sfac = mx.exp(cum)                                  # A_t <= 1
    gamma = mx.exp(cum[..., -1])                        # A_C

    # ループの外に出せる係数は全部畳んでおく。ループ本体の要素ごとの op が
    # 1 本 10us 前後あり、32 回まわると行列積と同じ桁になる
    #   - beta_j は tinv の**列**に畳める (d = tinv diag(beta) rhs)
    #   - A_C/A_j は kb の**行**に畳める (状態更新の縮約添字が j)
    #   - A_t は Khat/Qhat 共通なので 2C 本まとめて 1 回で掛ける
    tinv = tinv * bt[..., None, :]
    kbk = kb * mx.exp(cum[..., -1:] - cum)[..., None]
    sf2 = mx.concatenate([sfac, sfac], axis=-1)[..., None]
    dec = None
    cum = None

    # Khat S^T と Qhat S^T は同じ S^T を読むので 1 本の matmul にまとめる
    qk_cat = mx.concatenate([kb, qb], axis=-2)          # [nc, bh, 2C, Dk]
    qb = None
    kb = None

    if state is None:
        st = mx.zeros((bh, Dv, Dk), dtype=mx.float32)
    else:
        st = state.reshape(bh, Dv, Dk).astype(mx.float32)

    ys = []
    for i in range(nc):
        stt = st.swapaxes(-1, -2)                       # [bh, Dk, Dv]
        # [bh, 2C, Dv]: 前半が A_t (S k_t)、後半が A_t (S q_t)
        both = mx.matmul(qk_cat[i], stt) * sf2[i]
        rhs = vb[i] - both[:, :c]
        d = mx.matmul(tinv[i], rhs)                     # [bh, C, Dv]
        ys.append(both[:, c:] + mx.matmul(p[i], d))
        st = gamma[i][:, None, None] * st + mx.matmul(d.swapaxes(-1, -2), kbk[i])

    y = mx.stack(ys, axis=0)                            # [nc, bh, C, Dv]
    y = y.reshape(nc, B, Hv, c, Dv).transpose(1, 0, 3, 2, 4).reshape(B, tp, Hv, Dv)
    if pad:
        y = y[:, :T]
    return y.astype(out_dtype), st.reshape(B, Hv, Dv, Dk)


def gated_delta_update_blocked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    block: Optional[int] = None,
) -> Tuple[mx.array, mx.array]:
    """`mlx_lm.models.gated_delta.gated_delta_update` と同じ入出力 (mask 無し)。

    ゲートは `compute_g` の指数をそのまま log として取る
    (`log g = -exp(A_log) * softplus(a + dt_bias)`)。exp -> log の往復が
    無いので、g が下に潰れた位置でも -inf にならない。
    """
    beta = mx.sigmoid(b)
    log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    _fire.bump("gdn_blocked")
    return gated_delta_blocked(q, k, v, log_g, beta, state, block)
