"""段 K1 arm A (`mlxturbo/kernels/flash_attn_d256.py`) の proof-of-life。モデル無し。

本番の小 kv 帯 (kv < `MLXTURBO_PREFILL_ATTN_MIN_KV` = 8192) の prefill attention を、
**スコアを実体化しない自前 flash** に置き換えたときの費用を測る。

比較相手は **本番の P5 経路そのもの** (`mlxturbo/fused.py` の
`enable_sdpa_rowtile`、既定 R=256): q を 256 行ずつに割り、各タイルに前方の
K/V とマスクのスライスだけを渡して `mx.fast.scaled_dot_product_attention`
(head_dim 256 では fallback = matmul -> where -> softmax -> matmul) を呼ぶ形。

マスクは両者で **同じ keep_block から** 作る (可視集合が一致していないと
速度の比較に意味が無い): 行ごとに可視ブロックから top-512 をランダムに選び、
`MLXTURBO_QSA_TAIL=query` の規約 (クエリごとに列 `[cr*floor((q+1)/cr), q]`)
で端数を足す。参照にはトークン幅へ展開した bool を渡し、カーネルには
`keep_block` をそのまま渡す。

**冷やし方**: K/V の組を `--mb` MB ぶん用意して巡回する
(`docs/CLAUDE.md` の「カーネルの連鎖 micro は重みを 100 MB 超巡回させて冷やす」)。
温キャッシュの絶対値は信用しない。

判定線 (親): kv=4096 で **本番 P5 経路の 0.75 倍以下**。

    tools/biglock.sh .venv/bin/python tools/flash_attn_d256_micro.py \
        --json bench/results/flash-attn-d256-micro.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time

import mlx.core as mx


def make_keep_block(S: int, kv: int, cr: int, budget: int, seed: int = 0):
    """(1, S, n_blocks) bool。行ごとに可視ブロックから top-k をランダムに選ぶ。

    `QSAIndexer._select_keep` と同じ形 (`block_end <= q_col` の中から
    `k = min(budget//cr, n_blocks)` 個)。スコアの中身は速度に効かないので乱数。
    """
    offset = kv - S
    n_blocks = kv // cr
    q_col = mx.arange(offset, offset + S)
    block_end = mx.arange(n_blocks) * cr + (cr - 1)
    visible = block_end[None, :] <= q_col[:, None]          # (S, n_blocks)
    k = min(budget // cr, n_blocks)
    scores = mx.random.uniform(shape=(S, n_blocks), key=mx.random.key(seed))
    scores = mx.where(visible, scores, -mx.inf)
    top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
    keep = mx.zeros((S, n_blocks + 1), dtype=mx.bool_)
    top = mx.where(mx.take_along_axis(visible, top, axis=-1), top, n_blocks)
    keep = mx.put_along_axis(keep, top, mx.array(True), axis=-1)[..., :n_blocks]
    return keep[None], n_blocks, q_col


def expand_mask(keep_block, n_blocks: int, kv: int, cr: int, q_col, tail_mode: str):
    """`QSAIndexer.__call__` のトークン幅展開 (B, 1, S, kv)。"""
    B, S = keep_block.shape[0], keep_block.shape[1]
    keep = mx.repeat(keep_block, cr, axis=-1)
    tail = kv - n_blocks * cr
    if tail_mode == "query":
        keep = mx.concatenate(
            [keep, mx.zeros((B, S, tail + 1), dtype=mx.bool_)], axis=-1
        )
        own = ((q_col + 1) // cr) * cr
        cols = own[:, None] + mx.arange(cr - 1)[None, :]
        cols = mx.where(cols <= q_col[:, None], cols, kv)
        keep = mx.put_along_axis(
            keep, mx.broadcast_to(cols[None], (B, S, cr - 1)),
            mx.array(True), axis=-1,
        )[..., :kv]
    elif tail:
        tail_col = n_blocks * cr + mx.arange(tail)
        keep = mx.concatenate(
            [keep, mx.broadcast_to(
                tail_col[None, None, :] <= q_col[None, :, None], (B, S, tail))],
            axis=-1,
        )
    return keep[:, None]


def rowtile_sdpa(q, k, v, mask, scale: float, rows: int):
    """本番 P5 (`fused._sdpa_rowtile_call`) と同じ算法。"""
    S = q.shape[2]
    kv_len = k.shape[2]
    offset = kv_len - S
    outs = []
    t0 = 0
    while t0 < S:
        t1 = min(t0 + rows, S)
        kv_end = offset + t1
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, t0:t1, :], k[:, :, :kv_end, :], v[:, :, :kv_end, :],
                scale=scale, mask=mask[..., t0:t1, :kv_end],
            )
        )
        t0 = t1
    return mx.concatenate(outs, axis=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=2048)
    ap.add_argument("--kvs", default="2048,4096,6144,8192")
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--rowtile", type=int, default=256)
    ap.add_argument("--bq", default="4,8")
    ap.add_argument("--bk", default="16")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mb", type=int, default=140, help="巡回させる K/V の総 MB")
    ap.add_argument("--tail", default="query", choices=("query", "global"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    from mlxturbo.kernels import flash_attn_d256 as FA

    S, Hq, Hk, D = a.S, a.heads, a.kv_heads, a.dim
    scale = D ** -0.5
    bqs = [int(x) for x in a.bq.split(",")]
    bks = [int(x) for x in a.bk.split(",")]
    rows = []

    for kv in (int(x) for x in a.kvs.split(",")):
        set_mb = 2 * Hk * kv * D * 2 / 1e6          # K と V で 1 組
        nset = max(2, math.ceil(a.mb / set_mb))
        ks = [mx.random.normal((1, Hk, kv, D)).astype(mx.bfloat16) for _ in range(nset)]
        vs = [mx.random.normal((1, Hk, kv, D)).astype(mx.bfloat16) for _ in range(nset)]
        q = mx.random.normal((1, Hq, S, D)).astype(mx.bfloat16)
        q_bshd = mx.contiguous(q.transpose(0, 2, 1, 3))
        keep_block, n_blocks, q_col = make_keep_block(S, kv, a.cr, a.budget)
        mask = expand_mask(keep_block, n_blocks, kv, a.cr, q_col, a.tail)
        mx.eval(ks, vs, q, q_bshd, keep_block, mask)
        offset = kv - S

        def ref(i):
            return rowtile_sdpa(q, ks[i], vs[i], mask, scale, a.rowtile)

        def flash(i, bq, bk):
            return FA.run(
                q_bshd, ks[i], vs[i], keep_block, cap=kv, cr=a.cr, kv_len=kv,
                n_blocks=n_blocks, offset=offset, scale=scale, bq=bq, bk=bk,
                tail_mode=a.tail, n_kv=Hk, n_heads=Hq,
            )

        cases = {f"P5_R{a.rowtile}": ref}
        for bq in bqs:
            for bk in bks:
                cases[f"flash_bq{bq}_bk{bk}"] = (
                    lambda i, bq=bq, bk=bk: flash(i, bq, bk))
        names = list(cases)

        # 正しさ: fp32 の素の attention を真値にして、参照 (P5) と各変種の
        # 相対誤差を**同じ物差しで**出す (bf16 どうしの差だけ見ると、
        # どちらがどれだけ真値から離れているかが分からない)。
        def truth():
            qq = mx.repeat(q, Hq // Hk, axis=1) if False else q
            kk = mx.repeat(ks[0], Hq // Hk, axis=1).astype(mx.float32)
            vv = mx.repeat(vs[0], Hq // Hk, axis=1).astype(mx.float32)
            s = (qq.astype(mx.float32) * scale) @ kk.swapaxes(-1, -2)
            s = mx.where(mask, s, mx.array(-mx.inf, mx.float32))
            return (mx.softmax(s, axis=-1) @ vv).transpose(0, 2, 1, 3)

        t32 = truth()
        mx.eval(t32)
        denom = mx.maximum(mx.abs(t32).max(), 1e-6)
        errs = {}
        for n in names:
            o = cases[n](0)
            o = o.transpose(0, 2, 1, 3) if n.startswith("P5") else o
            mx.eval(o)
            errs[n] = float(mx.abs(o.astype(mx.float32) - t32).max() / denom)
            del o
        del t32

        # burn-in (プロセス起動直後の段差を捨てる)
        for _ in range(2):
            for n in names:
                mx.eval(cases[n](0))

        samp = {n: [] for n in names}
        it = 0
        for r in range(a.reps):
            order = names if r % 2 == 0 else names[::-1]
            for n in order:
                i = it % nset
                it += 1
                t0 = time.perf_counter()
                mx.eval(cases[n](i))
                samp[n].append((time.perf_counter() - t0) * 1e3)

        base = statistics.median(samp[names[0]])
        row = {"kv": kv, "S": S, "n_sets": nset, "mb": round(nset * set_mb, 1)}
        parts = []
        for n in names:
            ms = statistics.median(samp[n])
            row[n] = ms
            row[n + "_ratio"] = ms / base
            # 実仕事量ベースの TFLOPS は変種で違うので dense 基準で出す
            row[n + "_tflops_dense"] = (2 * 2 * S * kv * D * Hq / 1e9) / ms
            row[n + "_relerr_vs_fp32"] = errs[n]
            parts.append(
                f"{n} {ms:7.2f}ms ({ms/base:5.3f}x, err {errs[n]:.1e})")
        rows.append(row)
        print(f"kv={kv:6d} [{nset}組 {nset*set_mb:.0f}MB] " + " | ".join(parts),
              flush=True)
        del ks, vs, q, q_bshd, keep_block, mask, cases

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump({"note": __doc__, "args": vars(a), "rows": rows},
                      f, ensure_ascii=False, indent=1)
        print("書き出し:", a.json)
    os._exit(0)


if __name__ == "__main__":
    main()
