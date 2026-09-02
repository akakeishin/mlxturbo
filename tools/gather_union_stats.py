"""段 P1a: gather attention のタイル幅ごとに union 比を実測する道具。

prefill (QSA attention) でクエリ行をタイル幅 T に割ったとき、タイルごとの
選択ブロックの和集合 (union) が kv 全体の何割になるかを、実モデル・実プロンプト
(既定 17k) で測る。union が小さいほど「union だけ gather して小さい dense
sdpa に渡す」経路 (`Attention._gather_tile_attn`、`mlxturbo/gather_attn.py`)
の計算量削減が大きい。`bench/results/gather-tile-prefill.json` で tile=256 が
prefill_s を縮めなかった件の、原因切り分け用。

## 辞退の無効化

`Attention._gather_forward` は prefill 幅では比の上限
(`MLXTURBO_GATHER_MAX_RATIO`、既定 0.20) にほぼ必ず抵触して辞退し、dense 経路
に落ちる (この道具が見たい union 統計そのものが取れなくなる)。env で
`MLXTURBO_GATHER_MAX_RATIO=1.0` を立てるだけでは足りない場合がある --
`bound = rows * (token_budget // cr)` はタイルの行数と block_topk の積で
決まる**上限の見積もり**で、n_blocks/kv_len で頭打ちにしていないため、
小さいチャンクでは `bound * cr` が kv_len を 100 倍以上超えることがある
(比 1.0 でも辞退したままになりうる)。

そこでこの道具は `mlx_lm.models.qwen4_exp._gather_max_ratio` そのものを
`inf` を返す関数に差し替える (属性注入。モデルのコード = `_vendor/qwen4_exp.py`
は変えない)。`_gather_forward` はこの関数の戻り値としきい値比較するだけなので、
`inf` なら比の判定では絶対に辞退しない。

## 使い方

    tools/biglock.sh .venv/bin/python tools/gather_union_stats.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-interleaved

## 集計の注記

- レコードは層ごとに別リストで拾う (`enable_gather_attn` の共有 stats
  リストのままだと層が分からない) ので、層 x チャンク x タイルの実体は
  そのまま保持できるが、出力の JSON は「チャンク (kv_len 近似) x タイル幅」
  に潰した平均・最大だけを書く (層をまたいだ束ねが KERNEL-PROGRAM.md 段
  P1a が見たい量そのもの)。
- チャンクの識別に使う kv_len は `n_blocks * compress_ratio` の近似
  (端数 tail 分、最大 compress_ratio-1 トークンを無視する。17k 規模では
  誤差 0.1% 未満)。
- prefill 壁時計は CLAUDE.md の計測の作法どおり熱ドリフトの影響を受ける
  参考値。案の優劣は union_ratio / kv_frac / flop_ratio_vs_dense で見ること。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 17k 窓 1 本を切るための質問文 (tools/_bench_text.py の long_prompts と同じ作法)。
QUESTION = "上の文書の要点を、初めて読む人向けに 5 つに整理してください。"


def _aggregate(layer_lists: list[tuple[int, list]], cr: int) -> dict:
    """層 x サブタイル呼び出しの生レコードを、チャンク (kv_len 近似) で集計する。

    ``layer_lists`` は ``[(layer_idx, [(T, n_blocks, U, n_sel, union_ratio,
    kv_frac, true_u), ...]), ...]``。チャンクの識別は ``n_blocks * cr``
    (= kv_len の近似、tail を無視) をキーにする -- 同じチャンクを処理した
    呼び出しは層が違っても kv_len が一致するはず。

    ``U`` は実装が集めるブロック数の上限 (``min(n_blocks, T*block_topk)``)
    であって、和集合の実測ではない (17k のように T>=9 になる幅では毎回
    n_blocks に張り付き、union_ratio が恒常的に 1.000 になる)。``true_u``
    は `Attention._gather_tile_attn` (`mlxturbo/_vendor/qwen4_exp.py`) が
    ``mx.sum(union)`` で直接数えた真の和集合の大きさ。``true_union_ratio``
    と ``flop_ratio_vs_dense_true`` はこちらを使う。従来の U ベースの値
    (``union_ratio`` / ``flop_ratio_vs_dense``) は「実装が実際に集めている
    量 (詰め物込み)」として併記する。
    """
    rows = []
    for layer_idx, lst in layer_lists:
        for (T, n_blocks, U, n_sel, union_ratio, kv_frac, true_u) in lst:
            # tail (端数列。ブロック格子の外、全タイル共通で読み直す分) は
            # レコードに直接残っていないが、n_sel = U*cr + tail の構造
            # (`_gather_tile_attn` 参照) から逆算できる。
            tail = n_sel - U * cr
            true_union_ratio = (true_u / n_blocks) if n_blocks else 0.0
            rows.append(dict(layer=layer_idx, T=T, n_blocks=n_blocks, U=U,
                             n_sel=n_sel, union_ratio=union_ratio,
                             kv_frac=kv_frac, kv_len=n_blocks * cr,
                             true_u=true_u, tail=tail,
                             true_union_ratio=true_union_ratio,
                             true_n_sel=true_u * cr + tail))

    by_chunk: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_chunk[r["kv_len"]].append(r)

    by_chunk_out = []
    for kv_len in sorted(by_chunk):
        rs = by_chunk[kv_len]
        ur = [r["union_ratio"] for r in rs]
        kf = [r["kv_frac"] for r in rs]
        tur = [r["true_union_ratio"] for r in rs]
        by_chunk_out.append(dict(
            kv_len_approx=kv_len,
            n_records=len(rs),
            union_ratio_mean=sum(ur) / len(ur),
            union_ratio_max=max(ur),
            kv_frac_mean=sum(kf) / len(kf),
            kv_frac_max=max(kf),
            true_union_ratio_mean=sum(tur) / len(tur),
            true_union_ratio_max=max(tur),
        ))

    if rows:
        ur_all = [r["union_ratio"] for r in rows]
        kf_all = [r["kv_frac"] for r in rows]
        tur_all = [r["true_union_ratio"] for r in rows]
        overall = dict(
            union_ratio_mean=sum(ur_all) / len(ur_all),
            kv_frac_mean=sum(kf_all) / len(kf_all),
            true_union_ratio_mean=sum(tur_all) / len(tur_all),
        )
        # dense (T 行 x kv_len 列を毎回総なめ) に対する、gather が実際に読む
        # 列数の比。1.0 未満なら理屈のうえで計算量が縮んでいるはず
        # (U ベース。実装が実際に集めている量 -- U が上限に張り付いていれば
        # 1.0 に近いまま動かない)。
        num = sum(r["T"] * r["n_sel"] for r in rows)
        den = sum(r["T"] * r["kv_len"] for r in rows)
        flop_ratio = num / den if den else 0.0

        # 同じ比を真の和集合 (true_u) で計算し直したもの。U の頭打ちを
        # 取り除いた「タイル分割が理屈のうえでどこまで縮むか」の値。
        num_true = sum(r["T"] * r["true_n_sel"] for r in rows)
        flop_ratio_true = num_true / den if den else 0.0
    else:
        overall = dict(union_ratio_mean=0.0, kv_frac_mean=0.0,
                        true_union_ratio_mean=0.0)
        flop_ratio = 0.0
        flop_ratio_true = 0.0

    return dict(n_records=len(rows), by_chunk=by_chunk_out, overall=overall,
                flop_ratio_vs_dense=flop_ratio,
                flop_ratio_vs_dense_true=flop_ratio_true)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None,
                    help="既定は --model の中の mtp.safetensors")
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--tiles", default="0,256,128,64,32",
                    help="カンマ区切りのタイル幅。0 はタイル無し (S 全体を 1 回)")
    ap.add_argument("--out", default="bench/results/gather-union-stats.json")
    args = ap.parse_args()

    tiles = [int(t.strip()) for t in args.tiles.split(",")]

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401  (アーキ登録の meta_path フックを立てる)
    from mlxturbo import mtp_flash, spec_flash
    from mlxturbo.gather_attn import disable_gather_attn, enable_gather_attn
    from mlxturbo.runner import enable_default_fusions

    # 辞退 (比の上限、既定 0.20) をこの道具の中だけで無効化する。理由は
    # モジュール docstring のとおり (env の "十分大きい値" は推測が要る)。
    import mlx_lm.models.qwen4_exp as Q

    Q._gather_max_ratio = lambda *a, **kw: float("inf")

    model_path = os.path.expanduser(args.model)
    model, tok = load(model_path)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    # 出荷経路と同じ融合を当てる (decode_ab.py と同じ理由: gather のソート
    # など、これを当てないと本番と違う構成を測ることになる)
    enable_default_fusions(model, log_prefix="[gather-union-stats]")

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

    def run_prefill(caches) -> float:
        """prefill + decode 1 トークンだけ流し、最初の yield までの壁時計を返す。"""
        mx.clear_cache()
        t0 = time.perf_counter()
        gen = eng.generate_stream(ids, 1, caches=caches, eos_ids=eos_ids)
        t_prefill = None
        try:
            while True:
                next(gen)
                if t_prefill is None:
                    t_prefill = time.perf_counter() - t0
        except StopIteration:
            pass
        return t_prefill if t_prefill is not None else time.perf_counter() - t0

    # 温め: 最初の 1 本は重みのページインで遅い (decode_ab.py と同じ作法)。
    # union 比には影響しないので統計は捨てる
    run_prefill(model.make_cache())

    print(f"ctx={n_ctx}  compress_ratio={cr}  tiles={tiles}")
    print("辞退無効化: mlx_lm.models.qwen4_exp._gather_max_ratio を inf 返しに"
          " 差し替え (属性注入、モデルのコードは変えない)\n")

    out = {
        "meta": dict(
            model=model_path, ngram=args.ngram, ctx=n_ctx,
            compress_ratio=cr, tiles=tiles,
            note="prefill_s は熱ドリフトの影響を受ける参考値。案の優劣は"
                 " true_union_ratio / kv_frac / flop_ratio_vs_dense_true で"
                 " 見ること (union_ratio / flop_ratio_vs_dense は U の上限"
                 " 張り付きで 1.000 に固定されがちな、実装が実際に集めている"
                 " 量の参考値) (CLAUDE.md の計測の作法)。",
        ),
        "tiles": {},
    }

    for tile in tiles:
        disable_gather_attn(model)
        enable_gather_attn(model, tile=tile)

        # 層ごとに別の stats リストを持たせる (共有 1 本だと層が分からない)。
        layer_lists: list[tuple[int, list]] = []
        for i, layer in enumerate(model.model.layers):
            sa = getattr(layer, "self_attn", None)
            if sa is not None and hasattr(sa, "indexer"):
                lst: list = []
                sa._gather_stats = lst
                layer_lists.append((i, lst))

        caches = model.make_cache()  # タイル幅ごとに作り直す
        prefill_s = run_prefill(caches)

        agg = _aggregate(layer_lists, cr)
        out["tiles"][str(tile)] = dict(prefill_s=prefill_s, **agg)

        print(f"tile={tile:4d}  prefill {prefill_s:6.2f}s  "
              f"records={agg['n_records']:5d}  "
              f"union_ratio(mean) {agg['overall']['union_ratio_mean']:.3f}  "
              f"true_union_ratio(mean) {agg['overall']['true_union_ratio_mean']:.3f}  "
              f"kv_frac(mean) {agg['overall']['kv_frac_mean']:.3f}  "
              f"flop_ratio_vs_dense {agg['flop_ratio_vs_dense']:.3f}  "
              f"flop_ratio_vs_dense_true {agg['flop_ratio_vs_dense_true']:.3f}")
        for c in agg["by_chunk"]:
            print(f"    kv~{c['kv_len_approx']:6d}  n={c['n_records']:4d}  "
                  f"union_ratio mean={c['union_ratio_mean']:.3f} "
                  f"max={c['union_ratio_max']:.3f}  "
                  f"true_union_ratio mean={c['true_union_ratio_mean']:.3f} "
                  f"max={c['true_union_ratio_max']:.3f}  "
                  f"kv_frac mean={c['kv_frac_mean']:.3f} "
                  f"max={c['kv_frac_max']:.3f}")

    disable_gather_attn(model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
