"""段 K2a の選択カーネル (`mlxturbo/kernels/qsa_select.py`) を検める。

見るのは 2 つ。

1. **集合の一致**: `QSAIndexer._pooled_and_top` の選択部分 (relu -> h の和 ->
   /sqrt(head_dim) -> 可視ブロックだけ -> `mx.argpartition` で top-k ->
   keep_block) を切り出した参照と、カーネルの返す集合が行ごとに完全一致するか。
   乱数だけでなく意地悪 (同点を大量に、全ゼロ行、-0.0 混入、可視 < k、
   和の結合順で 1 ulp 動く並び) を踏む。**同点規則 (安定ソート = 添字の昇順)
   の読み違いはここでしか出ない。**
2. **時間**: 同じ形で「カーネル」と「置き換える相手の op 列」を 1 プロセス内
   ABAB で測る。判定線は 17k (n_blocks=4250) S=2 でカーネル <= 30 us。

参照は `mlxturbo/_vendor/qwen4_exp.py` 461-481 行の写し。**本家を変えたら
ここも変える** (写しであることが判定の前提)。端数 (tail) の可視規約
(`MLXTURBO_QSA_TAIL`) はブロック選択の外 (`__call__` 側) の話なので、
ここの判定には入らない。

使い方 (GPU を使うので biglock 経由で):

    tools/biglock.sh .venv/bin/python tools/verify_qsa_select.py
    tools/biglock.sh .venv/bin/python tools/verify_qsa_select.py --quick
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from mlxturbo.kernels import qsa_select as QS  # noqa: E402

HEAD_DIM = 128        # TextArgs.indexer_head_dim
N_IHEADS = 4          # TextArgs.indexer_n_heads (スコアの最終軸)
CR = 4                # TextArgs.indexer_compress_ratio
BUDGET = 2048         # TextArgs.indexer_budget
BLOCK_TOPK = BUDGET // CR   # = 512

S_LIST = [1, 2, 3, 4, 6]
KV_LIST = [4096, 8500, 17000, 25000, 50000]


# --------------------------------------------------------------------------
# 参照 (本家 `_pooled_and_top` の選択部分の写し)
# --------------------------------------------------------------------------
def ref_keep_block(raw: mx.array, q_col: mx.array, n_blocks: int, k: int) -> mx.array:
    """(B, S, n_blocks, 4) の生スコアから keep_block (B, S, n_blocks) を作る。

    `mlxturbo/_vendor/qwen4_exp.py` の 461-481 行そのまま。
    """
    B, S = raw.shape[0], raw.shape[1]
    scores = mx.maximum(raw, 0).sum(axis=-1) / math.sqrt(HEAD_DIM)

    block_starts = mx.arange(n_blocks) * CR
    block_end = block_starts + CR - 1
    visible = block_end[None, None, :] <= q_col[None, :, None]
    scores = mx.where(visible, scores, -mx.inf)

    top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
    keep_block = mx.zeros((B, S, n_blocks + 1), dtype=mx.bool_)
    top = mx.where(mx.take_along_axis(visible, top, axis=-1), top, n_blocks)
    keep_block = mx.put_along_axis(keep_block, top, mx.array(True), axis=-1)[
        ..., :n_blocks
    ]
    return keep_block


# --------------------------------------------------------------------------
# 意地悪な入力
# --------------------------------------------------------------------------
def _cases(rng: np.random.Generator, S: int, n_blocks: int):
    """(名前, raw (1,S,n_blocks,4) fp32) を順に返す。"""
    shape = (1, S, n_blocks, N_IHEADS)

    yield "rand", rng.standard_normal(shape, dtype=np.float32)

    # relu で 30-60% が 0 に潰れる形 (本番のスコア分布に近い)
    x = rng.standard_normal(shape, dtype=np.float32)
    x[rng.random(shape) < 0.45] = -1.0
    yield "relu_zero", x

    # 値を粗い格子に載せて同点を大量に作る。k 番目の値に数十〜数百の重複が出る
    g = rng.integers(0, 5, size=shape).astype(np.float32) * np.float32(0.5)
    g -= np.float32(0.5)
    yield "ties_coarse", g

    # さらに粗く: 和が 0/0.25/0.5/... の数種類しか取らない
    g2 = rng.integers(0, 3, size=shape).astype(np.float32) * np.float32(0.25)
    g2 -= np.float32(0.25)
    yield "ties_tiny", g2

    # 全部 0 (すべて同点。top-k は「添字の小さい方から k 個」になるはず)
    yield "all_zero", np.full(shape, -1.0, dtype=np.float32)

    # -0.0 を混ぜる。relu は +0.0 を返すはずで、ブロック全体が -0.0 の行も作る
    z = rng.standard_normal(shape, dtype=np.float32)
    z[rng.random(shape) < 0.3] = np.float32(-0.0)
    z[:, :, ::7, :] = np.float32(-0.0)
    yield "neg_zero", z

    # 閾値そのものが 0 になる形。95% のブロックが relu で 0 に潰れるので、
    # 「同点の 0 を添字の昇順で 300 本ほど拾う」ところで規則が試される。
    # 0 の作り方を -0.0 と -1.0 で混ぜ、-0.0 の正規化も同時に踏む
    zt = rng.standard_normal(shape, dtype=np.float32)
    dead = rng.random(shape[:-1]) < 0.95
    zt[dead] = np.where(
        rng.random((int(dead.sum()), N_IHEADS)) < 0.5,
        np.float32(-0.0),
        np.float32(-1.0),
    ).astype(np.float32)
    yield "zero_threshold", zt

    # 和の結合順で 1 ulp 動く並び。大小混在の 4 つ組を作り、値を密に集める
    big = np.float32(1.0)
    tiny = (rng.random(shape) * np.float32(2.0) ** -22).astype(np.float32)
    a = np.empty(shape, dtype=np.float32)
    a[..., 0] = big
    a[..., 1] = tiny[..., 1]
    a[..., 2] = big
    a[..., 3] = tiny[..., 3]
    yield "assoc", a

    # スコアが等しい塊を「ちょうど k 番目」に置く。閾値の同点処理を直撃する
    v = np.zeros(shape, dtype=np.float32)
    lvl = np.zeros((n_blocks,), dtype=np.float32)
    hi = min(BLOCK_TOPK - 5, n_blocks - 1)
    lvl[:hi] = 2.0                      # 確実に入る
    lvl[hi : hi + 40] = 1.0             # ここに閾値が来る (k-hi 個だけ取る)
    lvl[hi + 40 :] = 0.0                # 落ちる
    v[..., 0] = lvl[None, None, :]
    yield "boundary", v


def _q_col_variants(S: int, kv_len: int, n_blocks: int):
    """(名前, q_col (S,) int32) を返す。"""
    # 本番の並び: 直近 S トークンが query
    yield "tail", mx.arange(kv_len - S, kv_len, dtype=mx.int32)
    # 可視 < k を踏ませる (本番では起きないが、カーネルが count<k を返せること)
    for nv in (0, 1, 100, BLOCK_TOPK - 1, BLOCK_TOPK, BLOCK_TOPK + 1):
        if nv * CR > kv_len:
            continue
        base = nv * CR - 1        # (q_col+1)//cr == nv になる最小の q_col
        if base < 0:
            base = 0
        cols = np.clip(np.arange(base, base + S), 0, kv_len - 1).astype(np.int32)
        yield f"nvis={nv}", mx.array(cols)


# --------------------------------------------------------------------------
# 比較
# --------------------------------------------------------------------------
def _kernel_sets(raw: mx.array, q_col: mx.array, n_blocks: int, k: int):
    n_vis = QS.visible_counts(q_col, CR, n_blocks)
    sel, bits, cnt = QS.select(
        raw, n_vis, k, head_dim=HEAD_DIM, mode="both"
    )
    mx.eval(sel, bits, cnt)
    return np.array(sel), np.array(bits), np.array(cnt)


def _compare(raw_np, q_col, n_blocks, k, label, problems):
    B, S = raw_np.shape[0], raw_np.shape[1]
    raw = mx.array(raw_np)
    keep = np.array(ref_keep_block(raw, q_col, n_blocks, k))
    sel, bits, cnt = _kernel_sets(raw, q_col, n_blocks, k)
    sel = sel.reshape(B * S, k)
    bits = bits.reshape(B * S, -1)
    cnt = cnt.reshape(B * S)
    keep = keep.reshape(B * S, n_blocks)

    ok = True
    for r in range(B * S):
        ref = set(np.flatnonzero(keep[r]).tolist())
        got = set(int(x) for x in sel[r, : cnt[r]])
        # 番兵の埋め方も見る (足りない分は n_blocks)
        pad = sel[r, cnt[r] :]
        # uint32 の word ごとに bit0 が最小添字
        got_bits = set()
        for wi, word in enumerate(bits[r]):
            word = int(word)
            while word:
                b = word & -word
                got_bits.add(wi * 32 + b.bit_length() - 1)
                word ^= b

        if ref != got:
            ok = False
            problems.append(
                f"{label} row={r}: 集合が違う "
                f"(ref {len(ref)} / kernel {len(got)}, "
                f"ref-only {sorted(ref - got)[:8]}, kernel-only {sorted(got - ref)[:8]})"
            )
        if ref != got_bits:
            ok = False
            problems.append(f"{label} row={r}: bits が sel と食い違う")
        if pad.size and not np.all(pad == n_blocks):
            ok = False
            problems.append(f"{label} row={r}: 番兵が {n_blocks} でない")
        if len(ref) != int(cnt[r]):
            ok = False
            problems.append(
                f"{label} row={r}: cnt={int(cnt[r])} だが参照は {len(ref)} 本"
            )
    return ok


# --------------------------------------------------------------------------
# 前提の確認: MLX の最終軸 reduce が 0.0 からの逐次和か
# --------------------------------------------------------------------------
def check_reduce_order() -> bool:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 777, N_IHEADS), dtype=np.float32) * 1e3
    x[rng.random(x.shape) < 0.4] = -1.0
    # 大小混在も混ぜて結合順の差が出やすくする
    x[:, ::5, 0] = 1.0
    x[:, ::5, 1] = 2.0**-24
    a = mx.array(x)
    r = mx.maximum(a, 0)
    mlx_sum = r.sum(axis=-1)
    seq = mx.zeros(r.shape[:-1], dtype=mx.float32)
    for h in range(N_IHEADS):
        seq = seq + r[..., h]
    tree = (r[..., 0] + r[..., 1]) + (r[..., 2] + r[..., 3])
    mx.eval(mlx_sum, seq, tree)
    same_seq = bool(mx.array_equal(mlx_sum.view(mx.uint32), seq.view(mx.uint32)))
    same_tree = bool(mx.array_equal(mlx_sum.view(mx.uint32), tree.view(mx.uint32)))
    print(
        f"  MLX の sum(axis=-1) は 0.0 起点の逐次和とビット一致: {same_seq} "
        f"(木状の和とは {same_tree})"
    )
    return same_seq


# --------------------------------------------------------------------------
# 時間
# --------------------------------------------------------------------------
def _median(ts):
    return statistics.median(ts)


def bench(S: int, kv_len: int, reps: int = 32, rounds: int = 9):
    n_blocks = kv_len // CR
    k = min(BLOCK_TOPK, n_blocks)
    rng = np.random.default_rng(7)
    raws = [
        mx.array(rng.standard_normal((1, S, n_blocks, N_IHEADS), dtype=np.float32))
        for _ in range(reps)
    ]
    q_col = mx.arange(kv_len - S, kv_len, dtype=mx.int32)
    n_vis = QS.visible_counts(q_col, CR, n_blocks)
    block_end = mx.arange(n_blocks) * CR + (CR - 1)
    mx.eval(raws, n_vis, block_end, q_col)

    def call_kernel(x):
        sel, cnt = QS.select(x, n_vis, k, head_dim=HEAD_DIM, mode="idx")
        return sel

    def call_ref(x):
        scores = mx.maximum(x, 0).sum(axis=-1) / math.sqrt(HEAD_DIM)
        visible = block_end[None, None, :] <= q_col[None, :, None]
        scores = mx.where(visible, scores, -mx.inf)
        top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
        keep = mx.zeros((1, S, n_blocks + 1), dtype=mx.bool_)
        top = mx.where(mx.take_along_axis(visible, top, axis=-1), top, n_blocks)
        return mx.put_along_axis(keep, top, mx.array(True), axis=-1)[..., :n_blocks]

    def amortized(fn):
        outs = [fn(x) for x in raws]
        mx.eval(outs)

    def single(fn):
        mx.eval(fn(raws[0]))

    for _ in range(3):
        amortized(call_kernel)
        amortized(call_ref)

    # 同期そのものの費用 (これを引かないと 1 本ずつの数字は全部同期の値になる)
    tiny = mx.array([1.0], dtype=mx.float32)
    def call_null(_x):
        return tiny + 1.0

    ka, ra, ks, rs, ns = [], [], [], [], []
    for _ in range(rounds):          # ABAB (1 プロセス内で交互に)
        t = time.perf_counter(); amortized(call_kernel); ka.append((time.perf_counter() - t) / reps)
        t = time.perf_counter(); amortized(call_ref);    ra.append((time.perf_counter() - t) / reps)
    for _ in range(60):
        t = time.perf_counter(); single(call_kernel); ks.append(time.perf_counter() - t)
        t = time.perf_counter(); single(call_ref);    rs.append(time.perf_counter() - t)
        t = time.perf_counter(); single(call_null);   ns.append(time.perf_counter() - t)

    null = _median(ns) * 1e6
    return {
        "S": S,
        "kv": kv_len,
        "n_blocks": n_blocks,
        "kernel_us": _median(ka) * 1e6,
        "ref_us": _median(ra) * 1e6,
        "kernel_lat_us": _median(ks) * 1e6 - null,
        "ref_lat_us": _median(rs) * 1e6 - null,
        "null_us": null,
    }


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="形を絞って早く回す")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        print("GPU が要る (metal_kernel は CPU では動かない)")
        return 2

    s_list = [1, 2] if args.quick else S_LIST
    kv_list = [4096, 17000] if args.quick else KV_LIST

    print("== 前提の確認 ==")
    check_reduce_order()

    print("\n== 集合の一致 ==")
    problems: list[str] = []
    n_checked = 0
    n_rows = 0
    for S in s_list:
        for kv_len in kv_list:
            n_blocks = kv_len // CR
            k = min(BLOCK_TOPK, n_blocks)
            rng = np.random.default_rng(1000 * S + kv_len)
            case_fail = 0
            for name, raw_np in _cases(rng, S, n_blocks):
                for qname, q_col in _q_col_variants(S, kv_len, n_blocks):
                    # 可視の変種は形が多いので、乱数以外は tail と極端な 2 つに絞る
                    if name not in ("rand", "ties_coarse") and qname not in (
                        "tail",
                        "nvis=0",
                        f"nvis={BLOCK_TOPK - 1}",
                    ):
                        continue
                    label = f"S={S} kv={kv_len} {name}/{qname}"
                    before = len(problems)
                    _compare(raw_np, q_col, n_blocks, k, label, problems)
                    case_fail += len(problems) - before
                    n_checked += 1
                    n_rows += S
            mark = "ok" if case_fail == 0 else f"NG({case_fail})"
            print(f"  S={S} kv={kv_len:>5} n_blocks={n_blocks:>5} k={k:>3}: {mark}")

    print(f"\n  形 x ケース {n_checked} 通り / 行 {n_rows} 本")
    if problems:
        print(f"  不一致 {len(problems)} 件:")
        for p in problems[:40]:
            print("   -", p)
        return 1
    print("  すべて一致 (100%)")

    if args.no_bench:
        return 0

    print("\n== 時間 (1 プロセス内 ABAB、中央値) ==")
    print("  us/呼び出し。左 2 列は 32 本まとめて 1 回 eval (GPU 実働の目安)、")
    print("  右 2 列は 1 本ずつ eval して同期費用 (null) を引いた露出レイテンシ。")
    print(
        f"  {'S':>2} {'kv':>6} {'blocks':>7} "
        f"{'kern':>8} {'argpart':>8} {'比':>5} | "
        f"{'kern lat':>9} {'argp lat':>9} {'null':>7}"
    )
    rows = []
    for S in s_list:
        for kv_len in kv_list:
            r = bench(S, kv_len)
            rows.append(r)
            print(
                f"  {r['S']:>2} {r['kv']:>6} {r['n_blocks']:>7} "
                f"{r['kernel_us']:>8.1f} {r['ref_us']:>8.1f} "
                f"{r['ref_us'] / r['kernel_us']:>5.2f} | "
                f"{r['kernel_lat_us']:>9.1f} {r['ref_lat_us']:>9.1f} "
                f"{r['null_us']:>7.1f}"
            )

    target = [r for r in rows if r["S"] == 2 and r["kv"] == 17000]
    if target:
        us = target[0]["kernel_us"]
        print(f"\n  判定線 (17k S=2 で <= 30 us): {us:.1f} us -> "
              f"{'通る' if us <= 30 else '通らない'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
