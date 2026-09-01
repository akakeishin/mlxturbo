"""prefill 1 チャンク (幅 2048) の内訳を、実キャッシュのまま測る。

`tools/decode_anatomy.py` の prefill 版 (`docs/research/KERNEL-PROGRAM.md`
段 P0)。作法はあちらと同じ -- 本物のキャッシュを持ったまま部品を呼び、呼ぶ
前後でキャッシュを退避・復元して繰り返す。MLX の配列は不変なので、参照と
offset を控えておけば付け替えで完全に戻せる。

## decode 版と違うところ

- **単位が 1 チャンク (2048 トークン) 1 回のフォワード。**decode の width=2
  と違って kv が伸びるほど費用が変わるので、**先頭・中間・末尾の 3 点**で取る。
- **indexer を attention から分けて出す。**decode では長文ペナルティの 43%
  が indexer だった。prefill 幅では `Attention.__call__` が gather 経路
  (既定 on) に入って `QSAIndexer.select_blocks` を呼ぶので、`__call__` だけを
  包むと 1 回も捕まらない。両方を包む。
- **中間チャンクには lm_head が無い** (最終チャンクだけが logits を作る) ので
  部品にも入れない。
- 退避が decode 版より 1 段細かい。`_IndexerCache.keys` の setter は pooled
  キャッシュ (段 X1) を無条件に捨てるので、そこを通して戻すと 2 レップ目
  以降だけ全ブロック作り直しになり、indexer が本番より重く出る。ここでは
  `_buf` / `_pooled` を直に控える。

## 内訳の切り方

    moe       MoE 48 層
    gdn       線形注意 36 層
    attn      Attention 12 層 (indexer 込み)。差し引きで sdpa + 射影が出る
    indexer   QSA のブロック選択 (pooled・rope・einsum・top-k)
    hc        hyper-connection 97 回
    ple       PLE 1 層 (n-gram の行読みを含む)
    embed     embed_tokens + hc 本への tile

## 下限と効率

各部品に**下限 (帯域または FLOP から出した卓上値) と効率 (下限/実測)**を
併記する。FLOP の天井は密の 4bit 行列積の実測上限 11.2 TFLOPS
(`docs/research/KERNEL-PROGRAM.md`)、帯域の天井は逐次読みの実到達ピーク
393GB/s (`docs/research/KERNEL-BRIEF-DECODE-BW.md`)。**2 つのうち大きい方が
下限**で、どちらが縛っているかも出す。バイトは「最低これだけは動く」量だけを
数える (重みは 1 回、活性は入出力を 1 回ずつ)。中間の実体化は数えないので、
実装が中間を書き出していれば実測はここから離れる -- その差が読みどころ。

数え方は KERNEL-PROGRAM.md の prefill 節の表と揃えてある。GDN は射影だけを
数える (再帰スキャンは行列積ではなく別の天井を持つので下限には入れず、規模
だけ参考に出す)。attention のスコアは**密の因果**で数える -- sdpa は加算
マスクを渡されても密の全スコアを計算するので、いま回している FLOP は疎では
ない。**合計を 1 つの効率に潰さないこと** (11.2 TFLOPS は 4bit qmm の天井で
あって、sdpa にも GDN の再帰にも別の天井がある)。

**部品和 ≈ 壁時計 (数 % 以内) を必ず確認すること** (CLAUDE.md の作法)。
合わなければ見えていない項目があるということで、その差自体が結論になる。

壁時計は 2 つ出す。**チャンク主導** (`model.model(chunk, cache)`) が部品を
捕まえた経路そのもので、部品和と突き合わせるのはこちら。**layer-major**
(`spec_flash._group_prefill_forward`、既定 G=4) が本番の中間チャンクの経路で、
1 チャンクあたりに割って併記する。計測点まで進めるのはチャンク主導で行う
(layer-major とはキャッシュの中身がビット一致する規約なので、そこで測る
状態は本番と同じ)。

    tools/biglock.sh .venv/bin/python tools/prefill_anatomy.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep --ctx 17000
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 密の 4bit 行列積の実測上限 (docs/research/KERNEL-PROGRAM.md の prefill 節)。
PEAK_FLOPS = 11.2e12
# 逐次読みの実到達ピーク (docs/research/KERNEL-BRIEF-DECODE-BW.md)。
BW = 393e9
# 4bit + group64 の 1 重みあたりバイト数 (scales/biases 込み)。
QBYTE = 0.5 + 2 * 2 / 64


def med_ms(fn, reps):
    import mlx.core as mx

    # 温めの 1 本は捨てる。レップの中では `clear_cache` を打たない
    # (バッファの取り直しが計測に乗る)。解放は呼び出し側が部品の境目で行う。
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def snapshot(caches):
    """キャッシュの参照と offset を控える (配列は不変なので付け替えで戻る)。

    indexer は `keys` の setter を通さずに `_buf` を直に控える。setter が
    pooled キャッシュを捨てるのは正しい規約 (縮み・並べ替えがありうる経路
    だから) だが、ここで通すとレップごとに pooled を作り直すことになって
    本番より重く出る。
    """
    st = []
    for c in caches:
        ic = getattr(c, "indexer", None)
        if ic is not None:
            st.append(("a", c.keys, c.values, c.offset,
                       ic._buf, ic.offset, ic._pooled, ic._pooled_n))
        else:
            st.append(("l", [c[i] for i in range(4)]))
    return st


def pending(caches):
    """キャッシュに残っている遅延ノードを、評価対象として集める。

    MLX は遅延なので、キャッシュへの書き込みは誰かが読むまで走らない。
    出力だけを eval すると、そのチャンクぶんの書き込みが次のチャンクへ
    こぼれて帰属が狂う (実測: QSA が不活性な先頭チャンクでは indexer の
    qk 射影が丸ごと後ろへ流れ、0.3ms に見えていた)。本番の
    `generate_stream` もグループ境界で `c.state` を eval しているので、
    ここでもチャンクの区切りで揃える。indexer のバッファは `state` に
    含まれないので明示的に足す。
    """
    out = []
    for c in caches:
        st = getattr(c, "state", None)
        if isinstance(st, (list, tuple)):
            out.extend(x for x in st if x is not None)
        elif st is not None:
            out.append(st)
        ic = getattr(c, "indexer", None)
        if ic is not None and ic._buf is not None:
            out.append(ic._buf)
    return out


def restore(caches, st):
    for c, rec in zip(caches, st):
        if rec[0] == "a":
            _, c.keys, c.values, c.offset, buf, io, pooled, pn = rec
            ic = c.indexer
            ic._buf, ic.offset = buf, io
            ic._pooled, ic._pooled_n = pooled, pn
        else:
            for i, v in enumerate(rec[1]):
                c[i] = v


def bounds(a, S: int, offset: int, ngram_rec: float):
    """1 チャンクぶんの FLOP とバイトを数え、部品ごとの下限を返す。

    戻り値は ``({部品: (flop, bytes, 下限 ms, 律速)}, 再帰スキャンの flop,
    ブロック数)``。
    """
    d = a.hidden_size
    L = a.num_hidden_layers
    n_attn = sum(1 for t in a.layer_types if t == "full_attention")
    n_gdn = L - n_attn
    hc = a.hc_count
    hcd = hc * d
    kv = offset + S
    n_blocks = kv // a.indexer_compress_ratio
    out = {}

    def put(key, flop, byt):
        f_ms = flop / PEAK_FLOPS * 1000
        b_ms = byt / BW * 1000
        out[key] = (flop, byt, max(f_ms, b_ms), "計算" if f_ms >= b_ms else "帯域")

    # -- embed: 表引き 1 行 + hc 本への tile (どちらも行列積ではない)
    put("embed", 0.0, S * d * 2 * 2 + S * hcd * 2)

    # -- hc: down (hcd->lowrank) と up (lowrank->hcd) が本体。inject は
    #    hcd->hc で桁が 2 つ小さい。97 回 = 48 層 x 2 + 最後の mixer 1
    n_hc = 2 * L + 1
    lr = a.hc_lowrank
    put("hc",
        n_hc * S * 2 * 2 * hcd * lr + (n_hc - 1) * S * 2 * hcd * hc,
        n_hc * (2 * hcd * lr * QBYTE + S * (hcd * 2 + d * 2)))

    # -- ple: key/value 射影と depthwise conv。n-gram は行読みなので帯域側
    pd = a.ple_embed_dim
    ngram_heads = (a.ngram_size - 1) * a.heads_per_ngram
    put("ple",
        S * (2 * pd * hcd + 2 * pd * d + 2 * hcd * a.ple_conv_kernel_size),
        (pd * hcd + pd * d) * QBYTE + S * (hcd * 2 * 3 + d * 2)
        + S * ngram_heads * ngram_rec)

    # -- gdn: 射影 4 本 + out_proj。**再帰スキャンは数えない**
    conv_dim = (a.linear_key_head_dim * a.linear_num_key_heads * 2
                + a.linear_value_head_dim * a.linear_num_value_heads)
    vdim = a.linear_value_head_dim * a.linear_num_value_heads
    put("gdn",
        n_gdn * S * (2 * d * (conv_dim + vdim + 2 * a.linear_num_value_heads)
                     + 2 * vdim * d),
        n_gdn * (d * (conv_dim + vdim) * QBYTE + vdim * d * QBYTE
                 + S * (d * 2 + conv_dim * 2 + vdim * 2 + d * 2)))

    # -- attn: 射影 (q は出力ゲート込みで 2 倍幅) + 密因果のスコア
    nh, nkv, hd = a.num_attention_heads, a.num_key_value_heads, a.head_dim
    proj = 2 * d * (nh * hd * 2 + 2 * nkv * hd) + 2 * (nh * hd) * d
    pairs = S * offset + S * (S + 1) / 2
    put("attn",
        n_attn * (S * proj + 4 * nh * hd * pairs),
        n_attn * (d * (nh * hd * 2 + 2 * nkv * hd + nh * hd) * QBYTE
                  + 2 * nkv * hd * kv * 2  # K/V を 1 回読む
                  + S * (d * 2 + nh * hd * 2)))

    # -- indexer: qk 射影 + pooled との einsum (fp32)。einsum の出力
    #    (B,S,n_blocks,heads) は実体化されるので、そのバイトも数える。
    #    kv が budget 以下だと選択そのものが要らず (`_pooled_and_top` が
    #    None を返す)、射影とキャッシュ更新しか走らない
    ih, ikv, ihd = a.indexer_n_heads, a.indexer_kv_heads, a.indexer_head_dim
    active = kv > a.indexer_budget
    put("indexer",
        n_attn * (S * 2 * d * (ih + ikv) * ihd
                  + (2 * S * n_blocks * ih * ihd if active else 0)),
        n_attn * (d * (ih + ikv) * ihd * QBYTE + S * ihd * 2
                  + (S * n_blocks * ih * 4 * 2 + n_blocks * ihd * 2
                     if active else 0)))

    # -- moe: ルータ + top_k 個 + 共有エキスパート。2048 行 x top_k なら
    #    512 個のエキスパートはまず全部触られるので、重みは全部読む前提
    ne, tk, mi = a.num_experts, a.num_experts_per_tok, a.moe_intermediate_size
    put("moe",
        L * S * (2 * d * ne + (tk + 1) * 3 * 2 * d * mi),
        L * ((ne + 1) * 3 * d * mi * QBYTE + S * (d * 2 + d * 2)))

    # GDN の再帰スキャン (状態更新 + 読み出し)。行列積の天井が当たらないので
    # 下限には入れないが、規模は見えるようにしておく
    scan = (n_gdn * S * a.linear_num_value_heads
            * a.linear_key_head_dim * a.linear_value_head_dim * 4)
    return out, scan, n_blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--chunk", type=int, default=2048, help="prefill チャンク幅")
    ap.add_argument("--reps", type=int, default=3, help="中央値を取るレップ数")
    args = ap.parse_args()

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401
    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo.runner import enable_default_fusions
    from mlxturbo.spec_flash import _PREFILL_GROUP, _group_prefill_forward

    model, tok = load(os.path.expanduser(args.model))
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[prefill-anatomy]")

    ta = model.args.text
    ple = model.model.layers[model.model.ple_layers[0]].ple
    ng = ple.ple_embedding.ngram_embedding
    # n-gram 1 行のバイト数 (サイドカーは 4bit + scales/biases、素の表は bf16)
    if hasattr(ng, "bits"):
        ngram_rec = ng.dim * ng.bits / 8 + (ng.dim // ng.group_size) * 2 * 2
    else:
        ngram_rec = ng.dim * 2

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from _bench_text import long_prompts

    body = long_prompts(tok, args.ctx, ["上の文書の要点を 5 つに整理してください。"])[0]
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True))[None]
    n = ids.shape[1]
    step = args.chunk
    n_full = n // step
    if n_full < 3:
        print(f"完全チャンクが {n_full} 本しか無い。--ctx を増やすこと")
        return 1
    points = sorted({0, n_full // 2, n_full - 1})
    print(f"ctx={n} chunk={step} 完全チャンク={n_full} 計測点={points}"
          f" reps={args.reps} G={_PREFILL_GROUP}", flush=True)

    cache = model.make_cache()

    # 包む先: 部品名 -> (クラス, メソッド名)。select_blocks は gather 経路
    # (既定 on) の indexer の入口で、prefill 幅ではこちらだけが呼ばれる
    TARGET = {
        "moe": (Q.SparseMoeBlock, "__call__"),
        "gdn": (Q.GatedDeltaNet, "__call__"),
        "hc": (Q.GatedResidual, "__call__"),
        "ple": (Q.PLELayer, "__call__"),
        "attn": (Q.Attention, "__call__"),
        "idx": (Q.QSAIndexer, "__call__"),
        "sel": (Q.QSAIndexer, "select_blocks"),
    }
    orig = {k: getattr(cls, name) for k, (cls, name) in TARGET.items()}

    def as_array(r):
        """部品の戻り値から eval できる配列を取り出す。"""
        if r is None:
            return mx.zeros(1)
        if isinstance(r, tuple):
            return r[0]
        if isinstance(r, Q.QSABlockSelection):
            return r.keep_block
        return r

    def measure(ci: int):
        chunk = ids[:, ci * step : (ci + 1) * step]
        offset = ci * step
        grabbed = {k: [] for k in TARGET}

        def wrap(key, fn):
            def g(self, *a):
                grabbed[key].append((self, *a))
                return fn(self, *a)
            return g

        for k, (cls, name) in TARGET.items():
            setattr(cls, name, wrap(k, orig[k]))
        # gather 経路が実際に何ブロック集めているか (既存の計測用フック)
        stats = []
        for layer in model.model.layers:
            sa = getattr(layer, "self_attn", None)
            if sa is not None:
                sa._gather_stats = stats

        pre = snapshot(cache)
        mx.eval([model.model(chunk, cache=cache)] + pending(cache))
        for k, (cls, name) in TARGET.items():
            setattr(cls, name, orig[k])
        for layer in model.model.layers:
            sa = getattr(layer, "self_attn", None)
            if sa is not None:
                sa._gather_stats = None
        restore(cache, pre)
        mx.clear_cache()
        print("  捕まえた: " + " ".join(f"{k}={len(v)}"
                                     for k, v in grabbed.items() if v), flush=True)
        if stats:
            T, nb, U, n_sel, ur, kvf = stats[0]
            print(f"  gather 経路が活性: T={T} n_blocks={nb} union={U} ({ur:.0%})"
                  f" 集める列/kv={kvf:.0%}", flush=True)

        def bench(key):
            if not grabbed[key]:
                return 0.0
            fn = orig[key]

            def run():
                st = snapshot(cache)
                outs = [as_array(fn(*t)) for t in grabbed[key]]
                mx.eval(outs + pending(cache))
                restore(cache, st)
                return outs
            ms = med_ms(run, args.reps)
            mx.clear_cache()
            return ms

        res = {k: bench(k) for k in ("moe", "gdn", "hc", "ple", "attn")}
        res["indexer"] = bench("idx") + bench("sel")
        emb = model.model.embed_tokens
        res["embed"] = med_ms(
            lambda: mx.tile(emb(chunk), (1, 1, model.model.hc)), args.reps)
        grabbed.clear()
        mx.clear_cache()

        def whole():
            st = snapshot(cache)
            out = model.model(chunk, cache=cache)
            mx.eval([out] + pending(cache))
            restore(cache, st)
            return out

        total = med_ms(whole, args.reps)
        mx.clear_cache()

        group = None
        if _PREFILL_GROUP > 1 and ci + _PREFILL_GROUP <= n_full:
            chunks = [ids[:, (ci + k) * step : (ci + k + 1) * step]
                      for k in range(_PREFILL_GROUP)]

            def whole_group():
                st = snapshot(cache)
                hs = _group_prefill_forward(model, chunks, cache)
                mx.eval(hs + pending(cache))
                restore(cache, st)
                return hs

            group = med_ms(whole_group, args.reps) / _PREFILL_GROUP
            mx.clear_cache()

        lb, scan, n_blocks = bounds(ta, step, offset, ngram_rec)
        order = ["moe", "gdn", "attn", "indexer", "hc", "ple", "embed"]
        label = {"moe": "MoE 48 層", "gdn": "GDN 36 層",
                 "attn": "Attention 12 層 (indexer 込み)",
                 "indexer": "  うち indexer", "hc": "HC 97 回",
                 "ple": "PLE 1 層 (n-gram 込み)", "embed": "embed + tile"}
        parts = sum(res[k] for k in order if k != "indexer")
        print(f"  {'部品':34s}{'実測 ms':>9s}{'FLOP T':>9s}"
              f"{'下限 ms':>9s}{'効率':>7s}  律速")
        for k in order:
            flop, byt, low, kind = lb[k]
            eff = low / res[k] * 100 if res[k] else 0.0
            print(f"  {label[k]:34s}{res[k]:9.1f}{flop / 1e12:9.2f}"
                  f"{low:9.1f}{eff:6.1f}%  {kind}")
        print(f"  {'部品和 (indexer は attn に含む)':34s}{parts:9.1f}")
        print(f"  {'壁時計 (チャンク主導)':34s}{total:9.1f}")
        if group is not None:
            print(f"  {'壁時計 (layer-major /チャンク)':34s}{group:9.1f}"
                  f"  [{group - total:+.1f} ms]")
        gap = (parts - total) / total * 100 if total else 0.0
        print(f"  部品和 - 壁時計 = {parts - total:+.1f} ms ({gap:+.1f}%)")
        if abs(gap) > 10:
            print("  ** 数 % を超えてずれている。見えていない項目があるか、"
                  "単体計測が重なりを再現できていない **")
        print(f"  (参考) GDN の再帰スキャン {scan / 1e12:.2f} T / n_blocks={n_blocks}",
              flush=True)

    for ci in range(n_full):
        if ci in points:
            print(f"\n[chunk {ci}] offset={ci * step} kv={(ci + 1) * step}",
                  flush=True)
            measure(ci)
        mx.eval([model.model(ids[:, ci * step : (ci + 1) * step], cache=cache)]
                + pending(cache))
        mx.clear_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
