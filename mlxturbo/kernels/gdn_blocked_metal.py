# SPDX-License-Identifier: Apache-2.0
"""oMLX (jundot/oMLX, Apache-2.0) の GatedDeltaNet prefill 用 Metal カーネルの移植。

出典: jundot/oMLX commit ebf4f1f
      omlx/custom_kernels/qwen35_prefill/gdn.py の `gated_delta_blocked_seq`
      本体 (`_HEADER` / `_KERNEL_S_SRC` / `_get_kernel_s` / `_normalize_block_t`)。
      https://github.com/jundot/oMLX

`_KERNEL_S_SRC` はカーネル文字列そのもの (変更点は環境変数名
``OMLX_GDN_BLOCK_T`` -> ``MLXTURBO_GDN_BLOCK_T`` の 1 箇所のみ)。それ以外
(`eligible`、`gated_delta_update_blocked_metal`) はうちの呼び出し口
(`mlx_lm.models.gated_delta.gated_delta_update` と同じ g/beta の式) に
合わせて足した薄い包み。

## なぜ

`mlx_lm` 標準の逐次 Metal カーネル (`gated_delta_kernel`,
`_make_gated_delta_kernel`) は 1 threadgroup が Dv/4 個の d 列だけを担当し、
k/q を threadgroup ごとに device から読み直す。1 v-head を Dv/4 個の
threadgroup に割ると、同じ k/q 行が **32 回** 重複して読まれる計算になる
(mlx-serve の実測: 27B/16K で層あたり k/q の再読トラフィックが約 13GB/層)。

このカーネル (`gated_delta_blocked_seq`, 通称 kernel S) は **同じ逐次再帰を
そのまま計算する** (チャンク分解も WY 表現も無い。`gated_delta_blocked.py`
の行列積版とは別の道具)。違いは Apple GPU 向けの並べ替えだけ:

  - k/q/v を長さ TB (既定 32、fp32 入力は 16) のトークンブロックで
    threadgroup メモリへ協調ロードし、device からの再読を 1 回に減らす。
    v-head を Dv/32 個の threadgroup に分割 (stock の Dv/4 より粗い) ので、
    重複読みは 8 倍で済む。
  - 状態 [Dv, Dk] はスレッドのレジスタに常駐させたまま T 全体を回す
    (thread は (dv, 16 幅の d セグメント) を担当、8 スレッド/dv 行を
    simd_shuffle_down で縮約)。

mlx-serve はこれを Zig に移植して既定 on にしており、27B/16K で
14.9ms (blocked) 対 29.7ms (stock 逐次) と記録している。Flash-Next
(qwen4_exp) は Dk=128, Dv=128, Hk=16, Hv=48 でこのカーネルの制約
(Dk==128, Dv%32==0) を満たす。

## g/beta の規約 (重要: gated_delta_blocked.py とは違う)

このカーネルの `g` 入力は **1 ステップぶんの乗算係数をそのまま** (対数では
ない) 使う。カーネル本体で `st[i] *= gt;` と直接掛けている
(oMLX の chunked kernel A 側は内部で `log(g)` を取り直して使うが、kernel S
はそれをしない)。これは `mlx_lm.models.gated_delta.compute_g` /
`gated_delta_kernel` と同じ規約で、`gated_delta_blocked.py` が使う
`log_g = -exp(A_log) * softplus(a + dt_bias)` (対数) とは別物。
`gated_delta_update_blocked_metal` は `compute_g` をそのまま呼ぶので、
log 経由の往復誤差は無い。

## 精度

逐次版とは加算順が違うのでビット一致しない。state は fp32 で累算される
(`_KERNEL_S_SRC` の `float4 st[4]`) ので、逐次版 (同じく fp32 累算) との
差は小さいはず。突き合わせは `tools/verify_gdn_metal.py` (GPU 必須、
このリポジトリでは計測用に温存中の GPU を使うので後回しにする)。

## 適用範囲

`eligible()` が判定する。prefill 幅 (既定 T>=64)、mask 無し (バッチの
右パディング非対応)、Dk==128 (カーネル定数)、Dv%32==0 (DB=32 の前提)、
Hv%Hk==0 (GQA) のときだけ真。外れたら呼び手は既存のブロック化スキャン
(`gated_delta_blocked.py`) か逐次カーネルへそのまま落ちる。

2026-09-02 に既定 on にした。`mlxturbo.fused.enable_gdn_metal_kernel` が
`MLXTURBO_GDN_METAL=0` でなければ `GatedDeltaNet._gdn_metal` を立てる。
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
    print(f"[mlxturbo] GDN blocked-seq Metal カーネル: {msg}")


_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# oMLX gdn.py の `_KERNEL_S_SRC` そのまま (kernel S: blocked-sequential
# Gated DeltaNet prefill)。変更点は末尾の `_normalize_block_t` が読む
# 環境変数名だけ (OMLX_GDN_BLOCK_T -> MLXTURBO_GDN_BLOCK_T、この文字列の
# 中には出てこない)。
_KERNEL_S_SRC = """
    constexpr int TB = 32;                             // time block
    constexpr int DB = 32;                             // dv rows per threadgroup
    const int tid = thread_position_in_threadgroup.x;  // 0..255
    const int blk = threadgroup_position_in_grid.x;    // Dv/DB block
    const int hv  = threadgroup_position_in_grid.y;
    const int b   = threadgroup_position_in_grid.z;
    const int hk  = hv / (Hv / Hk);
    const int dv0 = blk * DB;

    // thread -> (dv row, 16-wide d segment); 8 threads per dv row, all in
    // the same simdgroup (lane = (dv%4)*8 + seg).
    const int dv  = tid / 8;            // 0..31
    const int seg = tid % 8;            // 0..7
    const int d0  = seg * 16;

    threadgroup InT k_s[TB][Dk + 8];
    threadgroup InT q_s[TB][Dk + 8];
    threadgroup InT v_s[TB][DB + 8];
    threadgroup float g_s[TB];
    threadgroup float b_s[TB];

    const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
    const size_t krow = (size_t)Hk * Dk;

    // state fragment in registers: [dv0+dv][d0..d0+16]
    float4 st[4];
    {
        const device float4* S_in = (const device float4*)(
            state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
        for (int i = 0; i < 4; ++i) st[i] = S_in[i];
    }

    device InT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;

    for (int t0 = 0; t0 < T; t0 += TB) {
        const int tt = min(TB, T - t0);
        // cooperative staging (coalesced): k/q rows, v slice, g/beta
        for (int p = tid; p < tt * Dk; p += 256) {
            const int r = p / Dk, d = p % Dk;
            k_s[r][d] = k_base[(size_t)(t0 + r) * krow + d];
            q_s[r][d] = q_base[(size_t)(t0 + r) * krow + d];
        }
        for (int p = tid; p < tt * DB; p += 256) {
            const int r = p / DB, d = p % DB;
            v_s[r][d] = v_base[(size_t)(t0 + r) * Hv * Dv + d];
        }
        for (int p = tid; p < tt; p += 256) {
            g_s[p] = g[((size_t)b * T + t0 + p) * Hv + hv];
            b_s[p] = beta[((size_t)b * T + t0 + p) * Hv + hv];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < tt; ++t) {
            const float gt = g_s[t];
            const float bt = b_s[t];
            const threadgroup vec<InT,4>* k4 =
                (const threadgroup vec<InT,4>*)&k_s[t][d0];
            const threadgroup vec<InT,4>* q4 =
                (const threadgroup vec<InT,4>*)&q_s[t][d0];
            float4 kf[4];
            for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);
            // kv_mem = (g*state) . k ; decay applied to state first
            float4 p4 = 0.0f;
            for (int i = 0; i < 4; ++i) {
                st[i] *= gt;
                p4 += st[i] * kf[i];
            }
            float part = p4.x + p4.y + p4.z + p4.w;
            // reduce across the 8 segment-threads of this dv row
            part += simd_shuffle_down(part, 4);
            part += simd_shuffle_down(part, 2);
            part += simd_shuffle_down(part, 1);
            const float kv_mem = simd_shuffle(part, (tid % 32) / 8 * 8);
            const float delta = ((float)v_s[t][dv] - kv_mem) * bt;

            float4 o4 = 0.0f;
            for (int i = 0; i < 4; ++i) {
                st[i] += kf[i] * delta;
                o4 += st[i] * float4(q4[i]);
            }
            float out = o4.x + o4.y + o4.z + o4.w;
            out += simd_shuffle_down(out, 4);
            out += simd_shuffle_down(out, 2);
            out += simd_shuffle_down(out, 1);
            if (seg == 0) {
                y_base[(size_t)(t0 + t) * Hv * Dv + dv] = (InT)out;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        device float4* S_out = (device float4*)(
            state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
        for (int i = 0; i < 4; ++i) S_out[i] = st[i];
    }
"""

_SUPPORTED_BLOCK_T = (16, 32, 48)
_kernel_s_by_tb: dict[int, object] = {}

# threadgroup メモリ使用量 (バイト) = TB * (k_s + q_s + v_s の行あたりバイト数
# + g_s/b_s の行あたり 8 バイト)。k_s/q_s は [Dk+8] 個、v_s は [DB+8] 個の
# InT。Dk/DB は eligible() が Dk==128 / Dv%32==0 (DB=32 固定) を要求するので
# ここでは固定値として扱う (`_KERNEL_S_SRC` の `constexpr int DB = 32;` と同じ)。
_KERNEL_DK = 128
_KERNEL_DB = 32
_METAL_THREADGROUP_LIMIT = 32768  # Metal の threadgroup メモリ上限 (32KiB)


def _threadgroup_bytes(block_t: int, input_dtype) -> int:
    itemsize = 4 if input_dtype == mx.float32 else 2
    per_row = (2 * (_KERNEL_DK + 8) + (_KERNEL_DB + 8)) * itemsize + 8
    return block_t * per_row


def _normalize_block_t(block_t: int | str | None, input_dtype=None) -> int:
    if block_t is None:
        configured_block_t = os.environ.get("MLXTURBO_GDN_BLOCK_T")
        if configured_block_t is not None:
            block_t = configured_block_t
        else:
            # fp32 入力 (mamba_ssm_dtype 相当) は TB=32 だと threadgroup
            # メモリが Metal の 32KiB 上限を超える (40,192 バイト、128/128
            # 16/32 のレイアウトで)。TB=16 なら 20,096 バイトに収まる。
            block_t = 16 if input_dtype == mx.float32 else 32
    block_t = int(block_t)
    if block_t not in _SUPPORTED_BLOCK_T:
        raise ValueError(
            f"MLXTURBO_GDN_BLOCK_T must be one of {_SUPPORTED_BLOCK_T}, got {block_t}"
        )
    # env 指定 (MLXTURBO_GDN_BLOCK_T) のときも上の dtype 分岐を素通りしない。
    # 以前は block_t is None の枝でしか fp32 の引き下げをしていなかったので、
    # env で明示的に 32/48 を指定すると fp32 入力でそのまま threadgroup
    # メモリ超過 (40,192 > 32,768) で落ちていた (D-5)。収まる最大の TB まで
    # 丸め下げる。
    while _threadgroup_bytes(block_t, input_dtype) > _METAL_THREADGROUP_LIMIT:
        smaller = [t for t in _SUPPORTED_BLOCK_T if t < block_t]
        if not smaller:
            # Dk==128/DB=32 が固定である限り TB=16 は常に収まるので、実際には
            # 到達しない安全弁。
            raise ValueError(
                f"block_t={block_t} ({input_dtype}) needs "
                f"{_threadgroup_bytes(block_t, input_dtype)} bytes of threadgroup "
                f"memory, over the {_METAL_THREADGROUP_LIMIT} limit, even at the "
                "smallest supported block size"
            )
        block_t = max(smaller)
    return block_t


def _get_kernel_s(block_t: int | str | None = None, input_dtype=None):
    block_t = _normalize_block_t(block_t, input_dtype)
    kernel = _kernel_s_by_tb.get(block_t)
    if kernel is None:
        source = _KERNEL_S_SRC.replace(
            "constexpr int TB = 32;", f"constexpr int TB = {block_t};"
        )
        kernel = mx.fast.metal_kernel(
            name=f"mlxturbo_gdn_blocked_seq_tb{block_t}",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            source=source,
            header=_HEADER,
        )
        _kernel_s_by_tb[block_t] = kernel
    return kernel


def gated_delta_blocked_seq(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    block_t: int | None = None,
) -> Tuple[mx.array, mx.array]:
    """Blocked-sequential Gated DeltaNet prefill (exact recurrence)。

    q,k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; g,beta: [B,T,Hv] (g は乗算係数、
    対数ではない); state: [B,Hv,Dv,Dk] fp32。y は q.dtype、state_out は fp32。
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    in_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    ks = _get_kernel_s(block_t, in_dtype)
    y, state_out = ks(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", in_dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(256 * (Dv // 32), Hv, B),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[in_dtype, mx.float32],
    )
    return y, state_out


def eligible(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    state: Optional[mx.array],
    mask: Optional[mx.array],
    min_t: int = 64,
) -> bool:
    """この呼び出しで Metal 版ブロック化スキャンを使えるか。

    外れたら呼び手は既存のブロック化スキャン (`gated_delta_blocked.eligible`)
    か逐次カーネル (`gated_delta_update`) へそのまま落ちる。
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        _warn_once("gpu", "GPU が既定デバイスでないので使わない")
        return False
    if mask is not None:
        # バッチの右パディング。kernel S は「状態を進めない位置」を扱えない
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
    if Dk != 128:
        _warn_once("dk", f"Dk={Dk} はカーネル定数の 128 と一致しない")
        return False
    if Dv % 32 != 0:
        _warn_once("dv", f"Dv={Dv} が 32 の倍数でない (DB=32 の前提が崩れる)")
        return False
    if T < min_t:
        _warn_once(
            "min_t",
            f"T={T} は decode/verify 幅 (min_t={min_t} 未満) なので使わない "
            "(下ごしらえの費用が得を上回る)",
        )
        return False
    if state is not None and state.dtype != mx.float32:
        _warn_once("state_dtype", "state が fp32 でない")
        return False
    if state is not None and state.shape != (B, Hv, Dv, Dk):
        _warn_once("state_shape", "state の形が (B, Hv, Dv, Dk) でない")
        return False
    # gated_delta_update_blocked_metal は block_t=None で呼ばれる (呼び手の
    # qwen4_exp.py は指定しない) ので、_normalize_block_t が実際に選ぶ TB を
    # ここでも再現し、threadgroup メモリ予算に収まることを確認する。env の
    # MLXTURBO_GDN_BLOCK_T が不正な値、または (Dk/DB が将来変わるなどで)
    # 最小の TB でも収まらない場合は ValueError になるので、ここで捕まえて
    # 素通し (呼び手を逐次カーネルへ落とす) にする (D-5)。
    try:
        resolved_tb = _normalize_block_t(None, q.dtype)
    except ValueError as exc:
        _warn_once("block_t", f"TB を決められない: {exc}")
        return False
    if _threadgroup_bytes(resolved_tb, q.dtype) > _METAL_THREADGROUP_LIMIT:
        _warn_once(
            "tg_bytes",
            f"TB={resolved_tb} ({q.dtype}) が threadgroup メモリ上限を超える",
        )
        return False
    return True


def gated_delta_update_blocked_metal(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    block_t: int | None = None,
) -> Tuple[mx.array, mx.array]:
    """`mlx_lm.models.gated_delta.gated_delta_update` と同じ入出力 (mask 無し)。

    g/beta は同じ式 (`compute_g`、`sigmoid`) で作る。kernel S は g を
    1 ステップぶんの乗算係数としてそのまま使う (対数ではない) ので、
    `gated_delta_blocked.py` と違って log 経由の往復が無い。
    """
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    _fire.bump("gdn_metal")
    return gated_delta_blocked_seq(q, k, v, g, beta, state, block_t)
