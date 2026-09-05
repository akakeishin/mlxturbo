"""`mlxturbo/kernels/qsa_prefill_attn.py` のホスト側 (union 添字列生成) を
CPU だけで検査するスクリプト。

対象は :func:`build_union_blocks` --- `QSAIndexer.select_blocks` が返す
`keep_block` (B, S, n_blocks bool) から、T クエリごとの union を昇順・
パディング済みの添字列にする純 MLX 演算 (Metal カーネル不要、GPU 不要)。
設計の根拠は `docs/research/QSA-PREFILL-KERNEL-DESIGN.md` の (b)/(c)。

Metal カーネル本体 (`_source`) はこのスクリプトの対象外 --- GPU が要る
ので、次に GPU が使えるセッションで `tools/verify_prefill_attn.py` の
流儀 (設計文書 (d)) に沿った別スクリプトを書くこと。

実行: .venv/bin/python bench/test_qsa_prefill_attn_host.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from mlxturbo.kernels.qsa_prefill_attn import build_union_blocks  # noqa: E402


def _keep_block(rng, B, S, n_blocks, cr, offset, block_topk):
    """QSA の規約どおりの ``keep_block`` を作る。

    `tools/verify_prefill_attn.py` の ``_keep_block`` と同じ生成規則:
    `QSAIndexer._pooled_and_top` は ``block_end <= q_col`` を満たすブロック
    だけを候補にするので、可視ブロックの本数は ``(q_col+1)//cr``。そこから
    最大 ``block_topk`` 本をランダムに選ぶ (スコアは検査対象ではないので
    乱数でよい --- ここで見たいのは union の集合演算が正しいかだけ)。
    """
    keep = np.zeros((B, S, n_blocks), dtype=bool)
    for b in range(B):
        for s in range(S):
            q_col = offset + s
            n_vis = min(n_blocks, (q_col + 1) // cr)
            if n_vis <= 0:
                continue
            k = min(block_topk, n_vis)
            sel = rng.choice(n_vis, size=k, replace=False)
            keep[b, s, sel] = True
    return keep


def _brute_force_unions(keep_np, tile):
    """タイルごとの真の union (ブロック添字の集合) をブルートフォースで作る。"""
    B, S, n_blocks = keep_np.shape
    n_tiles = (S + tile - 1) // tile
    unions = []
    for b in range(B):
        row = []
        for g in range(n_tiles):
            lo, hi = g * tile, min((g + 1) * tile, S)
            u: set[int] = set()
            for s in range(lo, hi):
                u |= set(np.nonzero(keep_np[b, s])[0].tolist())
            row.append(u)
        unions.append(row)
    return unions


def check_union_and_row_keep(
    B, S, n_blocks, cr, offset, block_topk, tile, tag
) -> bool:
    """union_idx (昇順・sentinel パディング・取りこぼし無し) と row_keep の
    両方を、ブルートフォース参照と突き合わせる。"""
    rng = np.random.default_rng(0)
    keep_np = _keep_block(rng, B, S, n_blocks, cr, offset, block_topk)
    keep = mx.array(keep_np)

    res = build_union_blocks(keep, tile, block_topk)
    mx.eval(res.union_idx, res.row_keep)
    union_idx = np.array(res.union_idx)
    row_keep = np.array(res.row_keep)

    unions_ref = _brute_force_unions(keep_np, tile)

    ok = True
    for b in range(B):
        for g in range(res.n_tiles):
            row = union_idx[b, g]
            is_sentinel = row == n_blocks
            if is_sentinel.any():
                first = int(np.argmax(is_sentinel))
                if not is_sentinel[first:].all():
                    ok = False
                    print(f"  {tag} b={b} g={g}: sentinel が末尾に固まっていない")
            real = row[~is_sentinel]
            if len(real) > 1 and not np.all(real[:-1] <= real[1:]):
                ok = False
                print(f"  {tag} b={b} g={g}: union_idx が昇順でない: {real.tolist()}")
            if set(real.tolist()) != unions_ref[b][g]:
                ok = False
                print(
                    f"  {tag} b={b} g={g}: union 不一致 "
                    f"got={sorted(real.tolist())} want={sorted(unions_ref[b][g])}"
                )

    for b in range(B):
        for s in range(S):
            g = s // tile
            row = union_idx[b, g]
            for j in range(res.u_pad):
                blk = int(row[j])
                want = bool(keep_np[b, s, blk]) if blk < n_blocks else False
                got = bool(row_keep[b, s, j])
                if got != want:
                    ok = False
                    print(
                        f"  {tag} b={b} s={s} j={j} blk={blk}: row_keep 不一致 "
                        f"got={got} want={want}"
                    )

    print(
        f"  {tag}: B={B} S={S} n_blocks={n_blocks} tile={tile} "
        f"u_pad={res.u_pad} n_tiles={res.n_tiles} s_pad={res.s_pad} "
        f"-> {'合格' if ok else '不合格'}"
    )
    return ok


def check_tile1_matches_own_selection() -> bool:
    """tile=1 の退化ケース: union は各クエリ自身の選択とちょうど一致するはず
    (T=1 では union がそのクエリ自身の選択そのものになる --- 設計文書 (b)
    の「T=1 だったから行ごとマスクが要らなかった」の裏返し)。"""
    rng = np.random.default_rng(1)
    B, S, n_blocks, cr, offset, block_topk = 1, 12, 40, 4, 16, 6
    keep_np = _keep_block(rng, B, S, n_blocks, cr, offset, block_topk)
    keep = mx.array(keep_np)
    res = build_union_blocks(keep, tile=1, block_topk=block_topk)
    mx.eval(res.union_idx, res.row_keep)
    union_idx = np.array(res.union_idx)

    ok = True
    for s in range(S):
        own = sorted(np.nonzero(keep_np[0, s])[0].tolist())
        row = union_idx[0, s]
        real = sorted(row[row < n_blocks].tolist())
        if real != own:
            ok = False
            print(f"  tile=1 s={s}: got={real} want={own}")
    print(f"  tile=1 退化ケース: {'合格' if ok else '不合格'}")
    return ok


def check_ragged_and_saturation() -> bool:
    """S がタイル幅で割り切れない場合 (端数タイルのパディング行) と、
    n_blocks が tile*block_topk 未満で静的上限がすぐ張り付く場合。"""
    ok = True
    ok &= check_union_and_row_keep(1, 10, 37, 4, 4, 5, 4, "ragged (S=10, tile=4)")
    ok &= check_union_and_row_keep(1, 9, 37, 4, 100, 5, 8, "ragged (S=9, tile=8)")
    ok &= check_union_and_row_keep(
        1, 16, 20, 4, 4, 5, 8, "saturation (n_blocks < tile*block_topk)"
    )
    return ok


def check_multi_batch() -> bool:
    return check_union_and_row_keep(3, 24, 64, 4, 8, 6, 4, "multi-batch (B=3)")


def check_zero_visible_tile() -> bool:
    """先頭に近いクエリ (block_end<=q_col を満たすブロックがまだ無い) は
    可視ブロック 0 本になる。タイルの一部の行が全 False でも
    build_union_blocks が壊れないこと。"""
    keep_np = np.zeros((1, 4, 10), dtype=bool)
    keep_np[0, 3, 0] = True  # cr=4 相当: s=0..2 は不可視、s=3 で block0 が可視
    keep = mx.array(keep_np)
    res = build_union_blocks(keep, tile=4, block_topk=4)
    mx.eval(res.union_idx, res.row_keep)
    union_idx = np.array(res.union_idx)
    row_keep = np.array(res.row_keep)

    ok = True
    real = sorted(union_idx[0, 0][union_idx[0, 0] < 10].tolist())
    if real != [0]:
        ok = False
        print(f"  zero-visible: union got={real} want=[0]")
    else:
        j0 = int(np.nonzero(union_idx[0, 0] == 0)[0][0])
        if not bool(row_keep[0, 3, j0]):
            ok = False
            print("  zero-visible: row_keep[s=3] が block0 を見ていない")
        if bool(row_keep[0, 0, j0]) or bool(row_keep[0, 1, j0]) or bool(row_keep[0, 2, j0]):
            ok = False
            print("  zero-visible: 不可視のはずの行が block0 を見ている")
    print(f"  zero-visible タイル: {'合格' if ok else '不合格'}")
    return ok


def main() -> int:
    print("=== build_union_blocks (ホスト側 union 添字列) の CPU 検査 ===")
    print("設計文書: docs/research/QSA-PREFILL-KERNEL-DESIGN.md (c)\n")
    ok = True

    ok &= check_union_and_row_keep(1, 32, 128, 4, 32, 16, 4, "T=4 基本形")
    ok &= check_union_and_row_keep(1, 32, 128, 4, 32, 16, 8, "T=8 基本形")
    ok &= check_union_and_row_keep(2, 20, 96, 4, 12, 12, 4, "B=2")
    ok &= check_tile1_matches_own_selection()
    ok &= check_ragged_and_saturation()
    ok &= check_multi_batch()
    ok &= check_zero_visible_tile()

    print(f"\n=== 総合判定: {'合格' if ok else '不合格'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)
    raise SystemExit(main())
