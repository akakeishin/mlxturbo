"""`docs/research/KERNEL-PROGRAM.md` 段 P1a の判定材料: prefill の QSA が選ぶ
ブロックを「タイル (連続するクエリ行のかたまり) ごとに和集合を取ったら、
和集合はどれだけ縮むか」を実装ゼロで見る道具。

## 背景

prefill は 1 チャンク 2048 クエリを 1 回の sdpa で流している。その 2048
クエリ全体で選択ブロックの和集合を取ると、当初の見立てどおりほぼ全ブロックに
なる (段 3(b) の当初却下の根拠)。しかし fable 指摘のとおりこれは「2048 クエリ
全体」で見た場合の話にすぎず、**gather の成否はタイル幅の関数**である。
隣り合うクエリの選択は強く相関する (局所窓 + 少数のグローバルブロック) ので、
タイルを 128-256 程度に切れば和集合は縮む可能性がある。

## 判定基準 (実装前に宣言、変更しないこと)

    タイル 128 で union/kv_len が概ね 0.4 を超えたら、タイル gather は捨てる
    (スコア削減が gather の帯域コストとタイル毎の K/V 複製で相殺される)。
    0.25 を切るなら期待値は壁 -3s 級。

## やっていること

1. 実文プロンプト (`tools/_bench_text.py` の `long_prompts`) をチャンク幅
   (既定 2048) で prefill しながら流す。
2. 各チャンクの各層 (self_attn を持つ層) で `QSAIndexer.select_blocks` を
   呼び直し、`keep_block` ((S, n_blocks) の bool) を捕まえる。
   本番フォワードが呼ぶ `Attention.__call__` を素通りラップして入力
   (x・rope・offset・idx_cache) だけを控え、キャッシュは
   `tools/decode_anatomy.py` の `snapshot`/`restore` で退避・復元してから
   呼び直す (本番の 1 回の呼び出しだけが cache を進める。select_blocks の
   やり直しはその副作用を残さない)。
3. タイル幅 {128, 256, 512, 2048} ごとに、そのタイル内のクエリ行の
   ブロック和集合 (`keep_block[ts:te].any(axis=0).sum()`) を数え、
   `union_blocks / n_blocks` と `union_blocks * compress_ratio / kv_len`
   (= 実際に読む列の割合) を、チャンク内のタイルと層をまたいで平均する。

出力は「タイル幅 x チャンク位置」の表。tile=2048 の列は元の却下判断
(2048 クエリ全体の和集合) の再現であり、他の列との差がタイル分割の効果を表す。

## 制約

**GPU では実行しないこと。**`--synthetic` は `tools/verify_batch_cache.py` の
`build` が組む合成 Flash-Next を CPU で流す自己確認専用で、指標そのものの
判定には使わない。実モデルでの判定は `--model`/`--ngram` を渡して別途 GPU で
走らせる (このツールはその実行ゆえの検証はしていない)。

使い方:

    # 自己確認 (CPU、合成モデル)
    .venv/bin/python tools/union_by_tile.py --synthetic

    # 判定用 (GPU、実モデル、このプロセス内ではキューに入れるだけで実行しない)
    .venv/bin/python tools/union_by_tile.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep --ctx 17000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def _attn_layer_index(model) -> dict:
    """``id(self_attn インスタンス) -> 層番号`` の対応表 (GDN 層は含まない)。"""
    out = {}
    for i, layer in enumerate(model.layers):
        sa = getattr(layer, "self_attn", None)
        if sa is not None:
            out[id(sa)] = i
    return out


def capture_select_blocks(model, ids, chunk: int) -> list:
    """`chunk` 幅ずつ prefill しながら、各層の `QSAIndexer.select_blocks` の
    結果を実フォワードの副作用を残さず捕まえる。

    戻り値はチャンクごとのリストで、各要素は
    ``[(layer_idx, offset, S, keep_block, n_blocks, kv_len, compress_ratio), ...]``。
    `select_blocks` が ``None`` (疎化が要らない = kv_len<=token_budget) を
    返した呼び出しは含めない。
    """
    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q

    from decode_anatomy import restore, snapshot

    layer_of = _attn_layer_index(model)
    orig_call = Q.Attention.__call__

    calls: list = []  # このチャンク分だけの一時バッファ

    def wrap(self, x, rope, mask, cache, idx_cache):
        offset = cache.offset if cache is not None else 0
        calls.append((self, x, rope, idx_cache, offset))
        return orig_call(self, x, rope, mask, cache, idx_cache)

    Q.Attention.__call__ = wrap
    try:
        cache = model.make_cache()
        n = ids.shape[1]
        results: list = []
        for lo in range(0, n, chunk):
            calls.clear()
            pre = snapshot(cache)
            mx.eval(model(ids[:, lo : lo + chunk], cache=cache))
            post = snapshot(cache)
            # select_blocks をやり直す前に、この呼び出しが見ていた
            # 「チャンク処理前」の cache 状態へ一旦戻す (idx_cache は
            # 本番の 1 回目の呼び出しで既に offset が進んでいるため、
            # そのまま呼び直すと raw key が二重に積まれる)。
            restore(cache, pre)
            chunk_res = []
            for sa, x, rope, idx_c, offset in calls:
                S = x.shape[1]
                blocks = sa.indexer.select_blocks(x, rope, idx_c, offset)
                if blocks is not None:
                    mx.eval(blocks.keep_block)
                    chunk_res.append(
                        (
                            layer_of.get(id(sa), -1),
                            offset,
                            S,
                            blocks.keep_block[0],  # (S, n_blocks) B=1 前提
                            blocks.n_blocks,
                            blocks.kv_len,
                            sa.indexer.compress_ratio,
                        )
                    )
            # select_blocks のやり直しで idx_cache が (本番と同じだけ) 再び
            # 進んだ分は、本番の実測状態 (post) で上書きして揃える。次の
            # チャンクは常に「本番が実際に進んだ状態」から続く。
            restore(cache, post)
            results.append(chunk_res)
        return results
    finally:
        Q.Attention.__call__ = orig_call


def tile_union_stats(keep_block, compress_ratio: int, kv_len: int, tile: int):
    """`keep_block` ((S, n_blocks) bool) をタイル幅 `tile` で割り、タイルごとの
    ``union_blocks/n_blocks`` と ``union_blocks*compress_ratio/kv_len``
    (実際に読む列の割合) を返す (どちらも長さ=タイル数のリスト)。"""
    S, n_blocks = keep_block.shape
    ratios = []
    read_fracs = []
    for ts in range(0, S, tile):
        te = min(ts + tile, S)
        union = keep_block[ts:te].any(axis=0)
        u = int(union.sum())
        ratios.append(u / n_blocks)
        read_fracs.append(u * compress_ratio / kv_len)
    return ratios, read_fracs


def build_table(per_chunk: list, tiles: list) -> list:
    """`capture_select_blocks` の戻り値から、チャンクごと x タイル幅ごとの
    ``(union/n_blocks 平均, read_frac 平均)`` を計算する (層をまたいだ平均、
    かつ 1 チャンク内の複数タイルをまたいだ平均)。"""
    rows = []
    for chunk_idx, chunk_res in enumerate(per_chunk):
        if not chunk_res:
            rows.append((chunk_idx, None, None, {}))
            continue
        offset0 = chunk_res[0][1]
        kv_len0 = chunk_res[0][5]
        cell = {}
        for tile in tiles:
            all_ratios: list = []
            all_reads: list = []
            for _layer_idx, _offset, _S, keep_block, n_blocks, kv_len, cr in chunk_res:
                ratios, reads = tile_union_stats(keep_block, cr, kv_len, tile)
                all_ratios.extend(ratios)
                all_reads.extend(reads)
            cell[tile] = (
                sum(all_ratios) / len(all_ratios),
                sum(all_reads) / len(all_reads),
            )
        rows.append((chunk_idx, offset0, kv_len0, cell))
    return rows


def print_table(rows: list, tiles: list) -> None:
    header = f"{'chunk':>5} {'offset':>7} {'kv_len':>7}"
    for t in tiles:
        header += f"  tile={t:<5}(uni/nblk, read%)"
    print(header)
    for chunk_idx, offset0, kv_len0, cell in rows:
        if not cell:
            print(f"{chunk_idx:5d}  (この チャンクは疎化が起きていない -- kv_len<=token_budget)")
            continue
        line = f"{chunk_idx:5d} {offset0:7d} {kv_len0:7d}"
        for t in tiles:
            ratio, read = cell[t]
            line += f"   {ratio:6.3f}, {read * 100:5.1f}%      "
        print(line)

    print()
    tile128_vals = [cell[128][0] for _, _, _, cell in rows if cell and 128 in cell]
    if tile128_vals:
        mean128 = sum(tile128_vals) / len(tile128_vals)
        print(f"tile=128 の union/n_blocks (全チャンク平均) = {mean128:.3f}")
        if mean128 > 0.4:
            print("  -> 判定基準: 0.4 超え。タイル gather は捨てる候補 "
                  "(スコア削減が帯域コストと K/V 複製で相殺される)")
        elif mean128 < 0.25:
            print("  -> 判定基準: 0.25 未満。期待値は壁 -3s 級")
        else:
            print("  -> 判定基準の 0.25-0.4 の間。二択が付かない (要追加検討)")
    else:
        print("tile=128 のデータが無い (プロンプトが短すぎて QSA が一度も"
              "活性化していない可能性がある -- --ctx か --synthetic-len を上げること)")


def _default_tiles(synthetic: bool) -> str:
    return "4,8,16" if synthetic else "128,256,512,2048"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="実モデルのパス (--synthetic 無指定時は必須)")
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--chunk", type=int, default=None, help="既定: 実モデル2048 / synthetic 16")
    ap.add_argument("--tiles", default=None, help="カンマ区切り。既定は実モデル/synthetic で別")
    ap.add_argument("--synthetic", action="store_true",
                     help="GPU を使わない自己確認用。verify_batch_cache.build の合成モデルを CPU で流す")
    ap.add_argument("--budget", type=int, default=8,
                     help="--synthetic 専用: indexer_budget (小さいほど早く QSA が活性化する)")
    ap.add_argument("--synthetic-len", type=int, default=96,
                     help="--synthetic 専用: 合成プロンプトのトークン数")
    args = ap.parse_args()

    if not args.synthetic and not args.model:
        ap.error("--model か --synthetic のどちらかが要る")

    import mlx.core as mx

    if args.synthetic:
        mx.set_default_device(mx.cpu)

    import mlxturbo  # noqa: F401  (mlx_lm.models.qwen4_exp をこの vendor 実装へ差し替える)

    tiles = [int(t) for t in (args.tiles or _default_tiles(args.synthetic)).split(",")]

    if args.synthetic:
        from verify_batch_cache import build

        chunk = args.chunk or 16
        model = build(args.budget)
        vocab = model.args.text.vocab_size
        n = args.synthetic_len
        ids = mx.array([[(i * 7 + 3) % vocab for i in range(n)]])
        print(f"=== 自己確認 (synthetic, CPU, budget={args.budget}, "
              f"len={n}, chunk={chunk}, tiles={tiles}) ===\n")
    else:
        chunk = args.chunk or 2048
        if args.ngram:
            os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")
        from mlx_lm import load

        from mlxturbo.runner import enable_default_fusions

        model, tok = load(os.path.expanduser(args.model))
        if args.ngram:
            from mlxturbo.ngram_stream import install

            install(model, os.path.expanduser(args.ngram))
        enable_default_fusions(model, log_prefix="[union-by-tile]")

        from _bench_text import long_prompts

        body = long_prompts(tok, args.ctx, ["上の文書の要点を5つに整理してください。"])[0]
        ids = mx.array(
            tok.apply_chat_template(
                [{"role": "user", "content": body}], add_generation_prompt=True
            )
        )[None]
        print(f"=== union_by_tile (ctx={ids.shape[1]}, chunk={chunk}, tiles={tiles}) ===\n")

    per_chunk = capture_select_blocks(model, ids, chunk)
    rows = build_table(per_chunk, tiles)
    print_table(rows, tiles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
