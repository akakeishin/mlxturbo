# SPDX-License-Identifier: Apache-2.0
"""GatedDeltaNet の prefill scan を「レジスタ常駐・threadgroup メモリ無し」で書く。

`gdn_blocked_metal.py` (oMLX 移植の kernel S) と**同じ再帰をそのまま**計算する
別実装。契約 (入出力・状態の形・dtype) は kernel S と 1 対 1 で、
`MLXTURBO_GDN_SCAN=reg` のときだけ差し替わる (既定は blocked = kernel S)。

## kernel S との違い

kernel S は k/q を長さ TB のトークンブロックで threadgroup メモリへ協調ロード
してから回す。そのため

  - TB=32・Dk=128・bf16 で threadgroup メモリを 1 threadgroup あたり
    約 20 KB 使う (`_threadgroup_bytes`)。コアあたりの threadgroup メモリは
    有限なので、同時に載る threadgroup 数 = 潜伏を隠す相手の数がここで決まる
  - トークンブロックごとに `threadgroup_barrier` が 2 回入る

こちらは **threadgroup メモリを一切使わず、barrier も無い**。k/q/v/g/beta は
device から直接読み (同じ行を読む隣接スレッドは L1 が受ける)、状態はスレッドの
レジスタに 1 回載せたまま T 全体を運び、行内の縮約は simd_shuffle だけで行い、
書き戻しは scan の終わりだけ。Perplexity の Lily が M5 Max で報告した作り
(`docs/research/EXTERNAL-PERPLEXITY-LILY-2026-09.md`) と同じ形。

## スレッドの割り当て (可変)

状態 [Dv, Dk] の 1 行 (dv 固定) を `lanes` 本のレーンで分け合う。1 スレッドが
持つのは `Dk/lanes` 個の d で、それが float4 のレジスタ `st[NF]` になる。

  - `lanes` を小さくすると 1 スレッドの持ち分 (=レジスタ) が増え、
    行内縮約の shuffle 段数 (log2 lanes) が減り、命令のうち実仕事の割合が上がる
  - `lanes` を大きくするとスレッド数が増えて並列度は上がるが、
    1 トークンあたりの shuffle 段数が増える

kernel S は lanes=8 (1 スレッド 16 個の d、float4 x 4) に相当する。どこが底かは
機械で決まるので `tools/gdn_scan_micro.py` で掃く。

## 測った結果 (2026-09-03、M3 Max 40 コア、冷の連鎖 micro、既定 off のまま)

`tools/gdn_scan_micro.py` (36 層ぶん 1.5 GB の活性を巡回、連鎖 72 歩、ABBA x 3)。
1 チャンク (T=2048) あたりの us、kernel S = 1.000:

    blocked (kernel S)          2882 us   x1.000
    reg lanes=4  db=32          2522 us   **x0.875**   <- この族の底
    reg lanes=4  db=16/8        2547-2573 x0.883-0.892
    reg lanes=8  db=32          3029 us   x1.045   (kernel S と同じ割り当て)
    reg lanes=16 db=16          4340 us   x1.497
    reg lanes=32 db=8           6823 us   x2.353   (Lily の説明どおりの「1 行 = 1 simdgroup」)
    reg lanes=2  db=32          6108 us   x2.118

**判定線 (冷で kernel S の 0.7 倍以下) に届かない。**しかも 8k prefill 全体に
対する GDN scan の取り分は天井スタブで 5.4% なので、0.875 倍を全部取れても
prefill は -0.7% にしかならず、親の採用線 (-2%) の下。既定は blocked のまま。

内訳の読み:

  - **kernel S も既に「レジスタ常駐」**である (状態 [Dv,Dk] の断片をスレッドの
    レジスタに 1 回載せて T 全体を回し、行内の縮約は simd_shuffle、書き戻しは
    終わりだけ)。違いは k/q を threadgroup メモリに置くかどうかだけで、
    同じ割り当て (lanes=8) で staging を外すと **1.045 倍と逆に遅くなる**
    (staging の取り分は +4.5%)。
  - 取れた 13% は staging の有無ではなく **割り当て** (1 スレッドが 16 個 ->
    32 個の d を持ち、行内縮約の shuffle が 3 段 -> 2 段になる) から来ている。
  - そこから先はレジスタで頭打ちになる。独立累算器 (acc=2/4/8) は
    x0.898/0.928/0.961 と**足すほど悪化**し、q の先読み (preq) を keep_k と
    併せると x5.23 (溢れ)。逆にスレッドあたりの持ち分を増やす lanes=2 は
    並列度不足で x2.1。**両側が壁**なので、この族の底は 0.87 前後。
  - Lily の +5.6% は「2K で 1 層 256 MiB を動かす blockwise」= チャンク分解を
    行列積に作り替える版 (うちの `gated_delta_blocked.py`、逐次の 1.68 倍で
    棄却済み) が比較対象。うちは 2026-09-02 に kernel S を既定 on にした時点で
    その取り分を既に取っている。

## 数値

kernel S と同じく fp32 の状態で同じ順に累算する (加算順は行内縮約の木が
同じ形なら一致するが、`lanes` を変えると木の形が変わるのでビット一致は
要求しない)。突き合わせは `tools/verify_gdn_scan_reg.py`
(逐次カーネル / kernel S / この実装の 3 者)。

## g/beta の規約

kernel S と同じ。`g` は 1 ステップぶんの乗算係数そのもの (対数ではない) で、
`mlx_lm.models.gated_delta.compute_g` の出力をそのまま渡す。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.gated_delta import compute_g

from . import _fire

_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    """同じ理由の見送りを 1 度だけ知らせる (黙って落ちないため)。"""
    if key in _warned:
        return
    _warned.add(key)
    print(f"[mlxturbo] GDN register-resident scan: {msg}")


_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# 1 スレッドが持つ d の本数 (Dk / lanes) と、1 threadgroup が持つ dv 行数。
# 既定は micro の掃きで決めた組。env で上書きできる (掃き用)。
DEFAULT_LANES = int(os.environ.get("MLXTURBO_GDN_SCAN_LANES", "4"))
DEFAULT_DB = int(os.environ.get("MLXTURBO_GDN_SCAN_DB", "32"))

# k を delta の後まで抱えるか (抱えるとレジスタ 2 倍、抱えないと k を 2 回読む)。
DEFAULT_KEEP_K = os.environ.get("MLXTURBO_GDN_SCAN_KEEPK", "1") != "0"

# 独立な累算器の本数 (1 = 直列)。**増やすと遅くなる** (上の測定、x0.898/0.928/
# 0.961)。レジスタが足りていないので、掃き用に残してあるだけ。
DEFAULT_ACC = int(os.environ.get("MLXTURBO_GDN_SCAN_ACC", "1"))

# q を先に読んでおくか。**keep_k と併せると溢れて x5.2** (上の測定)。掃き用。
DEFAULT_PREQ = os.environ.get("MLXTURBO_GDN_SCAN_PREQ", "0") != "0"

# 差し替えの switch。既定は blocked (= kernel S、本番の既定)。
# `gdn_blocked_metal.gated_delta_update_blocked_metal` がここを見る。
_ACTIVE = os.environ.get("MLXTURBO_GDN_SCAN", "blocked").lower() == "reg"


def active() -> bool:
    """レジスタ常駐 scan に差し替える設定になっているか。"""
    return _ACTIVE


def set_active(on: bool) -> None:
    """A/B ハーネス (`tools/decode_ab.py --knob gdn-scan-reg`) 用の切り替え。

    env (`MLXTURBO_GDN_SCAN`) は import 時に 1 度だけ読む。1 プロセス内で
    交互に測るときはこちらを使う。
    """
    global _ACTIVE
    _ACTIVE = bool(on)


_SRC = """
    constexpr int L    = {LANES};        // 1 つの dv 行を分け合うレーン数
    constexpr int DB   = {DB};           // 1 threadgroup が持つ dv 行数
    constexpr int SEGD = Dk / L;         // 1 スレッドが持つ d の本数
    constexpr int NF   = SEGD / 4;       // float4 何本ぶんか

    const int tid = thread_position_in_threadgroup.x;   // 0 .. DB*L-1
    const int blk = threadgroup_position_in_grid.x;     // Dv/DB ブロック
    const int hv  = threadgroup_position_in_grid.y;
    const int b   = threadgroup_position_in_grid.z;
    const int hk  = hv / (Hv / Hk);

    const int row = tid / L;             // threadgroup 内の dv 行
    const int seg = tid % L;             // 行内のレーン番号
    const int dv  = blk * DB + row;
    const int d0  = seg * SEGD;

    // 行内縮約の相手は同じ simdgroup の中に居る (L は 32 の約数、
    // 行の先頭は L の倍数のレーンに来る)。
    const uint lane     = (uint)(tid & 31);
    const uint row_lane = lane - (lane % (uint)L);   // 行の先頭レーン

    const size_t krow = (size_t)Hk * Dk;
    const size_t vrow = (size_t)Hv * Dv;
    const device InT* k_ptr = k + ((size_t)b * T * Hk + hk) * Dk + d0;
    const device InT* q_ptr = q + ((size_t)b * T * Hk + hk) * Dk + d0;
    const device InT* v_ptr = v + ((size_t)b * T * Hv + hv) * Dv + dv;
    const device float* g_ptr    = g    + (size_t)b * T * Hv + hv;
    const device float* beta_ptr = beta + (size_t)b * T * Hv + hv;
    device InT* y_ptr = y + ((size_t)b * T * Hv + hv) * Dv + dv;

    // 状態はここで 1 回だけ読み、scan の終わりまでレジスタに置く
    float4 st[NF];
    {{
        const device float4* S_in = (const device float4*)(
            state_in + (((size_t)b * Hv + hv) * Dv + dv) * Dk + d0);
        for (int i = 0; i < NF; ++i) st[i] = S_in[i];
    }}

    // 独立な累算器の本数。1 だと `p4 += ...` が NF 段の直列依存になるので、
    // ACC 本に分けて最後に足す (依存の木を浅くする)。
    constexpr int ACC = (NF < {ACC}) ? NF : {ACC};

    for (int t = 0; t < T; ++t) {{
        const float gt = g_ptr[(size_t)t * Hv];
        const float bt = beta_ptr[(size_t)t * Hv];
        const device vec<InT,4>* k4 =
            (const device vec<InT,4>*)(k_ptr + (size_t)t * krow);
        const device vec<InT,4>* q4 =
            (const device vec<InT,4>*)(q_ptr + (size_t)t * krow);
{KLOAD}{QLOAD}
        // kv_mem = (g*state) . k ; 減衰は状態に先に掛ける (kernel S と同じ順)
        float4 p4[ACC];
        for (int a = 0; a < ACC; ++a) p4[a] = 0.0f;
        for (int i = 0; i < NF; ++i) {{
            st[i] *= gt;
            p4[i % ACC] += st[i] * {KF};
        }}
        for (int a = 1; a < ACC; ++a) p4[0] += p4[a];
        float part = p4[0].x + p4[0].y + p4[0].z + p4[0].w;
        for (int off = L / 2; off >= 1; off >>= 1)
            part += simd_shuffle_down(part, off);
        const float kv_mem = simd_shuffle(part, row_lane);
        const float delta = ((float)v_ptr[(size_t)t * vrow] - kv_mem) * bt;

        float4 o4[ACC];
        for (int a = 0; a < ACC; ++a) o4[a] = 0.0f;
        for (int i = 0; i < NF; ++i) {{
            st[i] += {KF} * delta;
            o4[i % ACC] += st[i] * {QF};
        }}
        for (int a = 1; a < ACC; ++a) o4[0] += o4[a];
        float out = o4[0].x + o4[0].y + o4[0].z + o4[0].w;
        for (int off = L / 2; off >= 1; off >>= 1)
            out += simd_shuffle_down(out, off);
        if (seg == 0) y_ptr[(size_t)t * vrow] = (InT)out;
    }}

    {{
        device float4* S_out = (device float4*)(
            state_out + (((size_t)b * Hv + hv) * Dv + dv) * Dk + d0);
        for (int i = 0; i < NF; ++i) S_out[i] = st[i];
    }}
"""

# k を抱える版 / 都度読む版。抱えると float4 x NF のレジスタが増えるかわりに
# device (L1) からの読みが 1 トークンあたり 1 回で済む。
_KLOAD_KEEP = """        float4 kf[NF];
        for (int i = 0; i < NF; ++i) kf[i] = float4(k4[i]);
"""
_KLOAD_RELOAD = ""

# q を p の縮約より前に読んでおく版 (L1 の潜伏を shuffle の裏に隠す狙い)。
_QLOAD_PRE = """        float4 qf[NF];
        for (int i = 0; i < NF; ++i) qf[i] = float4(q4[i]);
"""
_QLOAD_LATE = ""

_kernels: dict[tuple, object] = {}


def _get_kernel(lanes: int, db: int, keep_k: bool, acc: int = 1, preq: bool = False):
    key = (lanes, db, keep_k, acc, preq)
    kernel = _kernels.get(key)
    if kernel is None:
        source = _SRC.format(
            LANES=lanes,
            DB=db,
            ACC=acc,
            KLOAD=_KLOAD_KEEP if keep_k else _KLOAD_RELOAD,
            QLOAD=_QLOAD_PRE if preq else _QLOAD_LATE,
            KF="kf[i]" if keep_k else "float4(k4[i])",
            QF="qf[i]" if preq else "float4(q4[i])",
        )
        kernel = mx.fast.metal_kernel(
            name=(f"mlxturbo_gdn_scan_reg_l{lanes}_db{db}"
                  f"_{'k' if keep_k else 'r'}a{acc}{'p' if preq else ''}"),
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            source=source,
            header=_HEADER,
        )
        _kernels[key] = kernel
    return kernel


def gated_delta_scan_reg(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    lanes: int | None = None,
    db: int | None = None,
    keep_k: bool | None = None,
    acc: int | None = None,
    preq: bool | None = None,
) -> Tuple[mx.array, mx.array]:
    """レジスタ常駐 scan 本体。`gated_delta_blocked_seq` と同じ入出力。

    q,k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; g,beta: [B,T,Hv] (g は乗算係数、
    対数ではない); state: [B,Hv,Dv,Dk] fp32。y は q.dtype、state_out は fp32。
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    in_dtype = q.dtype
    lanes = DEFAULT_LANES if lanes is None else lanes
    db = DEFAULT_DB if db is None else db
    keep_k = DEFAULT_KEEP_K if keep_k is None else keep_k
    acc = DEFAULT_ACC if acc is None else acc
    preq = DEFAULT_PREQ if preq is None else preq
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    kern = _get_kernel(lanes, db, keep_k, acc, preq)
    tgt = db * lanes
    y, state_out = kern(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", in_dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(tgt * (Dv // db), Hv, B),
        threadgroup=(tgt, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[in_dtype, mx.float32],
    )
    return y, state_out


def layout_ok(Dk: int, Dv: int, lanes: int, db: int) -> tuple[bool, str]:
    """`lanes`/`db` の組がカーネルの前提を満たすか (満たさない理由も返す)。"""
    if lanes not in (1, 2, 4, 8, 16, 32):
        return False, f"lanes={lanes} は 32 の約数の 2 冪でない"
    if Dk % (lanes * 4):
        return False, f"Dk={Dk} が lanes*4={lanes * 4} で割り切れない (float4 単位)"
    if Dv % db:
        return False, f"Dv={Dv} が db={db} で割り切れない"
    tgt = db * lanes
    if tgt % 32 or tgt > 1024:
        return False, f"threadgroup のスレッド数 {tgt} が 32 の倍数でないか 1024 超"
    return True, ""


def eligible(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    state: Optional[mx.array],
    mask: Optional[mx.array],
    min_t: int = 64,
    lanes: int | None = None,
    db: int | None = None,
) -> bool:
    """この呼び出しでレジスタ常駐 scan を使えるか。

    条件は kernel S (`gdn_blocked_metal.eligible`) と揃えてある。外れたら
    呼び手は kernel S か逐次カーネルへそのまま落ちる。
    """
    lanes = DEFAULT_LANES if lanes is None else lanes
    db = DEFAULT_DB if db is None else db
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        _warn_once("gpu", "GPU が既定デバイスでないので使わない")
        return False
    if mask is not None:
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
    if Hv % Hk != 0:
        _warn_once("gqa", f"Hv={Hv} が Hk={Hk} の倍数でない (GQA 前提が崩れる)")
        return False
    ok, why = layout_ok(Dk, Dv, lanes, db)
    if not ok:
        _warn_once("layout", why)
        return False
    if T < min_t:
        _warn_once("min_t", f"T={T} は decode/verify 幅 (min_t={min_t} 未満) なので使わない")
        return False
    if state is not None and state.dtype != mx.float32:
        _warn_once("state_dtype", "state が fp32 でない")
        return False
    if state is not None and state.shape != (B, Hv, Dv, Dk):
        _warn_once("state_shape", "state の形が (B, Hv, Dv, Dk) でない")
        return False
    return True


def gated_delta_update_scan_reg(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    lanes: int | None = None,
    db: int | None = None,
) -> Tuple[mx.array, mx.array]:
    """`gdn_blocked_metal.gated_delta_update_blocked_metal` と同じ入出力。"""
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    _fire.bump("gdn_scan_reg")
    return gated_delta_scan_reg(q, k, v, g, beta, state, lanes, db)
