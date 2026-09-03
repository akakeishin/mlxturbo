"""QSA の可視率を、行タイル幅ごとに実測する (kv < 8192 の帯の取り分を決める道具)。

`tools/gather_union_stats.py` は `_gather_tile_attn` を実際に走らせて union を
数えるので、タイル幅ごとに prefill を 1 本ずつ流し直す (幅の数だけ GPU を使う)。
こちらは **prefill を 1 本だけ流し、`QSAIndexer._select_keep` の戻り値
(`keep_block`, (B,S,n_blocks) bool) をそのまま持ち帰って**、タイル幅の掃引は
host 側の numpy でやる。幅を後から足せるし、GPU の占有は 1 本ぶんで済む。

見たい量 (`docs/research/IDEAS-2026-09-03.md` の「kv < 8192 の dense 帯」):

- **行あたりの可視率** = (min(block_topk, 完全ブロック数) * cr + tail) / kv_len。
  budget 2048 なので kv=4096 で 50%、6144 で 33%、8192 で 25% になるはず (確認)。
- **タイルあたりの union 率** = |タイル内の行の選択ブロックの和集合| / n_blocks。
  タイル単位で gather する案 (union を 1 度読んで T 行で共有する) の取り分は
  ここで決まる。
- **dense 行タイル (P5、既定 R=256) に対する比**。P5 は既に本番の既定なので、
  比較の分母は「S*kv_len の総なめ」ではなく **タイル t が [0, offset+(t+1)R) の
  K/V だけを読む形**でなければ取り分を過大評価する。

出力は JSON と、標準出力の表。

使い方:

    tools/biglock.sh .venv/bin/python tools/qsa_vis_stats.py \\
        --model ~/models/ddalcu-mlxlm-head4 --ngram ~/models/ddalcu-ngram-interleaved \\
        --ctx 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

QUESTION = "上の文書の要点を、初めて読む人向けに 5 つに整理してください。"

TILES = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def analyse(rec: dict, cr: int, block_topk: int, rowtile: int) -> dict:
    """1 回の呼び出し (層 x チャンク) の keep_block からタイル幅ごとの比を出す。

    ``rec['keep']`` は (S, n_blocks) の bool (B=1 前提)。
    """
    keep: np.ndarray = rec["keep"]
    S, n_blocks = keep.shape
    kv_len = rec["kv_len"]
    offset = rec["offset"]
    n_bcols = n_blocks * cr          # 完全ブロックが覆う列数
    tail = kv_len - n_bcols          # 格子から溢れた末尾の列数

    q_col = np.arange(offset, offset + S)
    # 行ごとの tail (`MLXTURBO_QSA_TAIL=query`、本番の既定): 行 q は
    # 自分の未完成ブロックの列 [cr*floor((q+1)/cr), q] を必ず見る (0〜cr-1 列)。
    own = ((q_col + 1) // cr) * cr
    per_row_tail = q_col - own + 1               # = (q+1) % cr
    per_row_blocks = keep.sum(axis=1)
    per_row_cols = per_row_blocks * cr + per_row_tail

    out = {
        "offset": offset,
        "kv_len": kv_len,
        "n_blocks": n_blocks,
        "S": S,
        "tail": tail,
        "row_vis_cols_mean": float(per_row_cols.mean()),
        "row_vis_frac_mean": float((per_row_cols / kv_len).mean()),
        "row_blocks_mean": float(per_row_blocks.mean()),
        # dense の 2 つの分母 (総なめ / 行タイル R)
        "dense_full_cols": float(S * kv_len),
        "tiles": {},
    }

    # dense 行タイル (P5): タイル t は [0, offset + t1) を読む。
    # FLOP の分母は行数で重み付け、K/V の読みバイトの分母は重み無し
    # (1 タイルの K/V スライスはタイル内の全行で 1 度だけ読まれる)。
    dense_rt_flop = 0.0
    dense_rt_bytes = 0.0
    t0 = 0
    while t0 < S:
        t1 = min(t0 + rowtile, S)
        dense_rt_flop += (t1 - t0) * (offset + t1)
        dense_rt_bytes += (offset + t1)
        t0 = t1
    out["dense_rowtile_cols"] = float(dense_rt_flop)
    out["dense_rowtile_bytes_cols"] = float(dense_rt_bytes)

    for T in TILES:
        if T > S:
            continue
        flop = 0.0
        bytes_cols = 0.0
        u_ratios = []
        t0 = 0
        while t0 < S:
            t1 = min(t0 + T, S)
            if T == 1:
                u_mask = keep[t0]
            else:
                u_mask = keep[t0:t1].any(axis=0)
            u = int(u_mask.sum())
            # union が覆う列 (ブロック単位) と、それに覆われない tail の列。
            # tail の列はタイル内のどれかの行から見える範囲
            # [own(先頭行), q(末尾行)] のうち union のブロックに入らないもの。
            lo, hi = int(own[t0]), int(q_col[t1 - 1])
            in_tail = np.zeros(kv_len, dtype=bool)
            in_tail[lo : hi + 1] = True
            covered = np.repeat(u_mask, cr)
            extra = int((in_tail[:n_bcols] & ~covered).sum())
            extra += int(in_tail[n_bcols:].sum())
            cols = u * cr + extra
            flop += (t1 - t0) * cols
            bytes_cols += cols
            u_ratios.append(u / n_blocks if n_blocks else 0.0)
            t0 = t1
        out["tiles"][str(T)] = {
            "union_ratio_mean": float(np.mean(u_ratios)),
            "union_ratio_max": float(np.max(u_ratios)),
            "cols_read": float(flop),
            "vs_dense_full": float(flop / (S * kv_len)),
            "vs_dense_rowtile": float(flop / dense_rt_flop),
            "bytes_vs_dense_rowtile": float(bytes_cols / dense_rt_bytes),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=8000)
    ap.add_argument("--rowtile", type=int, default=256,
                    help="dense 側の行タイル幅 (P5 の既定 256)")
    ap.add_argument("--layers", default="",
                    help="カンマ区切りで層を絞る (既定は全部)")
    ap.add_argument("--out", default="bench/results/qsa-vis-stats.json")
    args = ap.parse_args()

    only_layers = set(int(x) for x in args.layers.split(",") if x.strip())

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401  (アーキ登録)
    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo import mtp_flash, spec_flash
    from mlxturbo.runner import enable_default_fusions, set_wired_limit_default

    model_path = os.path.expanduser(args.model)
    model, tok = load(model_path)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[qsa-vis-stats]")
    set_wired_limit_default(log_prefix="[qsa-vis-stats]")

    # indexer -> 層番号
    idx_of: dict[int, int] = {}
    for i, layer in enumerate(model.model.layers):
        sa = getattr(layer, "self_attn", None)
        if sa is not None and getattr(sa, "indexer", None) is not None:
            idx_of[id(sa.indexer)] = i

    records: list[dict] = []
    seen: set = set()
    orig_select = Q.QSAIndexer._select_keep

    def hooked(self, raw, block_end, q_col, n_blocks):
        keep = orig_select(self, raw, block_end, q_col, n_blocks)
        S = keep.shape[1]
        li = idx_of.get(id(self), -1)
        if S >= 64 and (li in only_layers or not only_layers):
            offset = int(q_col[0].item())
            key = (li, offset, S, n_blocks)
            if key not in seen:
                seen.add(key)
                kb = np.asarray(keep[0], dtype=bool)  # (S, n_blocks)
                records.append(dict(layer=li, offset=offset, S=S,
                                    n_blocks=n_blocks,
                                    kv_len=offset + S, keep=kb))
        return keep

    Q.QSAIndexer._select_keep = hooked

    mtp_path = args.mtp or os.path.join(model_path, "mtp.safetensors")
    q = {"group_size": 64, "bits": args.mtp_bits} if args.mtp_bits else None
    mtp = mtp_flash.load_flash_mtp(os.path.expanduser(mtp_path),
                                   model.args.text, quantize=q)
    mx.eval(mtp.parameters())
    eng = spec_flash.FlashSpecEngine(model, mtp)

    eos = tok.eos_token_ids if hasattr(tok, "eos_token_ids") else ()
    eos_ids = tuple(eos) if eos else ()

    from _bench_text import long_prompts

    prompt = long_prompts(tok, args.ctx, [QUESTION])[0]
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True))[None]
    n_ctx = ids.shape[1]
    cr = model.args.text.indexer_compress_ratio
    budget = model.args.text.indexer_budget
    block_topk = budget // cr

    caches = model.make_cache()
    # prefill だけで足りる (max_tokens=0)。decode 幅の記録は S>=64 の網で
    # どのみち落ちるが、無駄に GPU を回さない。
    gen = eng.generate_stream(ids, 0, caches=caches, eos_ids=eos_ids,
                              checkpoints=[])
    try:
        while True:
            next(gen)
    except StopIteration:
        pass

    Q.QSAIndexer._select_keep = orig_select

    print(f"ctx={n_ctx} cr={cr} budget={budget} block_topk={block_topk} "
          f"rowtile={args.rowtile}  records={len(records)}")

    per = [analyse(r, cr, block_topk, args.rowtile) for r in records]
    # チャンク (kv_len) ごとに層をまたいで束ねる
    by_kv: dict[int, list[dict]] = {}
    for a in per:
        by_kv.setdefault(a["kv_len"], []).append(a)

    out = {"meta": dict(model=model_path, ctx=n_ctx, cr=cr, budget=budget,
                        block_topk=block_topk, rowtile=args.rowtile,
                        tiles=list(TILES)),
           "by_kv": {}}

    for kv in sorted(by_kv):
        aa = by_kv[kv]
        row_frac = float(np.mean([a["row_vis_frac_mean"] for a in aa]))
        row_cols = float(np.mean([a["row_vis_cols_mean"] for a in aa]))
        dense_rt_frac = float(np.mean([a["dense_rowtile_cols"] / a["dense_full_cols"]
                                       for a in aa]))
        ent = dict(n_layers=len(aa), n_blocks=aa[0]["n_blocks"], S=aa[0]["S"],
                   row_vis_cols_mean=row_cols, row_vis_frac_mean=row_frac,
                   dense_rowtile_frac_of_full=dense_rt_frac, tiles={})
        for T in TILES:
            k = str(T)
            if k not in aa[0]["tiles"]:
                continue
            ent["tiles"][k] = dict(
                union_ratio_mean=float(np.mean([a["tiles"][k]["union_ratio_mean"]
                                                for a in aa])),
                union_ratio_max=float(np.max([a["tiles"][k]["union_ratio_max"]
                                              for a in aa])),
                vs_dense_full=float(np.mean([a["tiles"][k]["vs_dense_full"]
                                             for a in aa])),
                vs_dense_rowtile=float(np.mean([a["tiles"][k]["vs_dense_rowtile"]
                                                for a in aa])),
                bytes_vs_dense_rowtile=float(np.mean(
                    [a["tiles"][k]["bytes_vs_dense_rowtile"] for a in aa])),
            )
        out["by_kv"][str(kv)] = ent

        print(f"\nkv={kv}  n_blocks={ent['n_blocks']}  S={ent['S']}  "
              f"層 {ent['n_layers']}  行あたり可視 {row_cols:.0f} 列 "
              f"({row_frac*100:.1f}% of kv)  "
              f"dense 行タイル R={args.rowtile} は総なめの "
              f"{dense_rt_frac*100:.1f}%")
        print("   T |  union/n_blocks (mean/max) | FLOP/総なめ | FLOP/行タイル | K/V 読み/行タイル")
        for T in TILES:
            k = str(T)
            if k not in ent["tiles"]:
                continue
            e = ent["tiles"][k]
            print(f" {T:4d} |   {e['union_ratio_mean']:.3f} / {e['union_ratio_max']:.3f}"
                  f"          |    {e['vs_dense_full']:.3f}    |     "
                  f"{e['vs_dense_rowtile']:.3f}     |    "
                  f"{e['bytes_vs_dense_rowtile']:8.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out_path}")

    # 生の keep_block も (層 0 と中間層だけ) 残す: 後から別の粒度を試すため
    npz = out_path.with_suffix(".npz")
    save = {}
    for r in records:
        save[f"L{r['layer']}_kv{r['kv_len']}"] = np.packbits(r["keep"], axis=-1)
        save[f"L{r['layer']}_kv{r['kv_len']}_shape"] = np.array(r["keep"].shape)
    np.savez_compressed(npz, **save)
    print(f"書き出し: {npz}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
