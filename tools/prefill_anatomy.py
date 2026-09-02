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

MoE は prefill の半分を占めるので、さらに中を割って別表で出す。切り方は
`SparseMoeBlock.__call__` と `SwitchGLU` の gather_sort 経路
(`fused.enable_gather_sort`、既定 SORT_MIN=16) の呼び出し順そのまま。

    router    gate の行列積 (x を fp32 にしてから)
    topk      argpartition + take_along_axis + softmax
    sort      expand_dims + _gather_sort (argsort 2 回 + 行の take)
    up/gate   gather_qmm 2 本 (2560 -> 640)
    swiglu    silu(gate) * up
    down      gather_qmm 1 本 (640 -> 2560)
    unsort    _scatter_unsort + squeeze
    combine   ルータ重みの適用 + top-K の縮約
    shared    shared_expert_gate + shared_expert + 加算

内訳は**層ごとに**測る。前提 (logits、添字、並べ替え済みの活性) は計測区間の
外で作って確定させ、部品にはその実物を渡す。並べ替え済みの活性だけで 1 層
105MB あり、48 層ぶん抱えられないため。層ごとに同期が入るぶん、48 層まとめて
測った MoE 全体とは重なりの有無が違う -- その差も出す。

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


# MoE の中の部品。SparseMoeBlock.__call__ の呼び出し順そのまま。
MOE_ORDER = ["router", "topk", "sort", "up", "gate", "swiglu", "down",
             "unsort", "combine", "shared"]
MOE_LABEL = {
    "router": "  router の行列積 (fp32)",
    "topk": "  top-k 選択 + softmax",
    "sort": "  並べ替え (argsort x2 + take)",
    "up": "  up 行列積 (2560 -> 640)",
    "gate": "  gate 行列積 (2560 -> 640)",
    "swiglu": "  SwiGLU",
    "down": "  down 行列積 (640 -> 2560)",
    "unsort": "  書き戻し (scatter unsort)",
    "combine": "  ルータ重み + top-K 縮約",
    "shared": "  共有専門家 (ゲート込み)",
}


def moe_bounds(a, mod, S: int):
    """MoE 48 層ぶんの、部品ごとの FLOP とバイトを数える。

    バイトは**2 通り**数える。

    - **最小バイト**: `bounds` と同じ規約 (重みは 1 回、活性は入出力を
      1 回ずつ)。中間の実体化は数えない。
    - **動いたバイト**: コードが実際に作る配列を、op ごとに読み書きで数える。
      `x.astype(mx.float32)` の写し、`argpartition` の全幅の出力、
      `us * w[..., None]` が bf16 x fp32 で fp32 に昇格して 210MB を書くこと
      など、**FLOP がゼロのまま DRAM を往復する量**はここにしか出ない。

    どちらも下限は `max(FLOP/11.2T, バイト/393GB/s)` で出す (ルーフライン)。
    重なりゼロの側の括弧として `FLOP + バイト` も呼び出し側で出す。

    エキスパートの重みは 512 個を全部読む前提 -- S=2048 x top_k=10 の
    20480 行なら 1 個あたり平均 40 行で、まず全部触られる。部品の FLOP の
    合計は `bounds` の moe 行と一致する。
    """
    d, L = a.hidden_size, a.num_hidden_layers
    ne, tk = a.num_experts, a.num_experts_per_tok
    mi, sh = a.moe_intermediate_size, a.shared_expert_intermediate_size
    P = S * tk
    sw, se = mod.switch_mlp, mod.shared_expert
    out = {}

    def put(key, flop, byt, act):
        flop, byt, act = L * flop, L * byt, L * act
        f = flop / PEAK_FLOPS * 1000
        out[key] = (flop, byt, max(f, byt / BW * 1000), act,
                    max(f, act / BW * 1000), f + act / BW * 1000,
                    "計算" if f >= act / BW * 1000 else "帯域")

    def wb(lin, n):
        """重み n 個を 1 回読むバイト数 (量子化なら 4bit + scales/biases)。"""
        return n * (QBYTE if hasattr(lin, "scales") else 2)

    # router: astype が fp32 の写しを 1 枚作り、qmm はそれを読む
    put("router", 2 * S * d * ne,
        wb(mod.gate, ne * d) + S * d * 2 + S * ne * 4,
        S * d * 2 + S * d * 4 + S * d * 4 + wb(mod.gate, ne * d) + S * ne * 4)
    # topk: -logits / argpartition の出力はどちらも ne 幅 (tk 幅ではない)
    put("topk", 0.0, S * ne * 4 + 2 * S * tk * 4, 5 * S * ne * 4 + 6 * S * tk * 4)
    # sort: argsort 2 回 + order//M + indices[order] で添字を 4 枚作り、
    #       行の take は P 行を集めて P 行書く
    put("sort", 0.0, S * d * 2 + P * d * 2 + 3 * S * tk * 4,
        8 * P * 4 + 2 * P * d * 2)
    # up/gate/down: gather_qmm は中間を作らないので最小 = 動いたバイト
    for key, lin in (("up", sw.up_proj), ("gate", sw.gate_proj)):
        b = wb(lin, ne * mi * d) + P * d * 2 + P * mi * 2
        put(key, 2 * P * d * mi, b, b)
    # swiglu: mx.compile 済みの融合 op 1 つ (silu と mul が分かれない)
    put("swiglu", 0.0, 3 * P * mi * 2, 3 * P * mi * 2)
    b = wb(sw.down_proj, ne * mi * d) + P * mi * 2 + P * d * 2
    put("down", 2 * P * mi * d, b, b)
    put("unsort", 0.0, 2 * P * d * 2, 2 * P * d * 2)
    # combine: us(bf16) * w(fp32) は fp32 に昇格する。P x d の fp32 が
    #          1 枚 (2048 x 10 x 2560 x 4 = 210MB) 実体化して sum が読み直す
    put("combine", 0.0, P * d * 2 + S * d * 2,
        P * d * 2 + S * tk * 4 + 2 * P * d * 4 + 2 * S * d * 4 + S * d * 2)
    # shared: MLP は silu と mul が別 op (SwitchGLU 側と違い融合されない)
    swb = (wb(se.gate_proj, d * sh) + wb(se.up_proj, d * sh)
           + wb(se.down_proj, d * sh) + wb(mod.shared_expert_gate, d))
    put("shared", 2 * S * d * (3 * sh + 1),
        swb + S * (3 * d * 2 + 3 * sh * 2),
        swb + S * (2 * d * 2 + 2 * sh * 2)          # gate/up 射影
        + S * (2 * sh * 2 + 3 * sh * 2)             # silu と mul (別 op)
        + S * (sh * 2 + d * 2)                      # down 射影
        + S * (2 * d * 2) + S * (3 * d * 2))        # sigmoid の掛けと加算
    return out


def moe_parts(grabbed, reps):
    """捕まえた MoE 48 層を、層ごとに部品へ割って ms を積む。

    **無効化の差分ではなく、部品そのものを実物の入力で走らせた実測。**
    前提は計測区間の外で eval して確定させる。1 層ぶん作っては測って捨てる
    (並べ替え済みの活性が 1 層 105MB なので 48 層は抱えられない)。

    同じ層で `SparseMoeBlock.__call__` そのものも測って `_layer` に積む。
    部品和の突き合わせ先はこちら -- 48 層を 1 回の eval にまとめて測った
    「MoE 全体」とは、同時に生きる中間の量が 48 倍違う。2 つの差自体が
    読みどころなので両方出す。

    `_pad32` / `_pad64` はエキスパートあたりの行数から出したタイルの水増し率
    (sum(ceil(c/T)*T) / sum(c))。gather_qmm はソート済みの添字を
    エキスパートごとの区間に切って走るので、区間が短いほどタイルが余る。

    `#` で始まる鍵は**積み上げの段**。qmm 3 本だけの graph から始めて、
    SwiGLU・並べ替え・書き戻し・router 一式を 1 段ずつ足す。段の増分と、
    個別に測った部品を突き合わせると、「部品の時間」なのか「投入の間の
    空白」なのかが分かれる。無効化の引き算ではなく、毎段が実物の graph。
    """
    import mlx.core as mx
    import mlx_lm.models.switch_layers as SL
    import numpy as np

    def topk(logits, tk):
        i = mx.argpartition(-logits, tk - 1, axis=-1)[..., :tk]
        return i, mx.softmax(mx.take_along_axis(logits, i, axis=-1),
                             axis=-1, precise=True)

    acc = {k: 0.0 for k in MOE_ORDER}
    acc.update({k: 0.0 for k in ("_layer", "#qmm", "#silu", "#sort", "#glu")})
    pad32, pad64 = [], []
    for mod, x in grabbed:
        sw, tk = mod.switch_mlp, mod.top_k
        acc["_layer"] += med_ms(lambda: mod(x), reps)
        logits = mod.gate(x.astype(mx.float32))
        mx.eval(logits)
        idx, w = topk(logits, tk)
        mx.eval(idx, w)
        c = np.bincount(np.array(idx).ravel(), minlength=logits.shape[-1])
        for T, dst in ((32, pad32), (64, pad64)):
            dst.append(float(-(-c // T).sum() * T / c.sum()))
        xs, idxs, inv = SL._gather_sort(mx.expand_dims(x, (-2, -3)), idx)
        mx.eval(xs, idxs, inv)
        x_up = sw.up_proj(xs, idxs, sorted_indices=True)
        x_gate = sw.gate_proj(xs, idxs, sorted_indices=True)
        mx.eval(x_up, x_gate)
        act = sw.activation(x_up, x_gate)
        mx.eval(act)
        dn = sw.down_proj(act, idxs, sorted_indices=True)
        mx.eval(dn)
        us = SL._scatter_unsort(dn, inv, idx.shape).squeeze(-2)
        mx.eval(us)
        comb = (us * w[..., None]).sum(axis=-2).astype(x.dtype)
        mx.eval(comb)

        acc["router"] += med_ms(lambda: mod.gate(x.astype(mx.float32)), reps)
        acc["topk"] += med_ms(lambda: topk(logits, tk), reps)
        acc["sort"] += med_ms(
            lambda: SL._gather_sort(mx.expand_dims(x, (-2, -3)), idx), reps)
        acc["up"] += med_ms(
            lambda: sw.up_proj(xs, idxs, sorted_indices=True), reps)
        acc["gate"] += med_ms(
            lambda: sw.gate_proj(xs, idxs, sorted_indices=True), reps)
        acc["swiglu"] += med_ms(lambda: sw.activation(x_up, x_gate), reps)
        acc["down"] += med_ms(
            lambda: sw.down_proj(act, idxs, sorted_indices=True), reps)
        acc["unsort"] += med_ms(
            lambda: SL._scatter_unsort(dn, inv, idx.shape).squeeze(-2), reps)
        acc["combine"] += med_ms(
            lambda: (us * w[..., None]).sum(axis=-2).astype(x.dtype), reps)
        acc["shared"] += med_ms(
            lambda: comb + mx.sigmoid(mod.shared_expert_gate(x))
            * mod.shared_expert(x), reps)

        # -- 積み上げ。段ごとに 1 つの graph で、間に eval を入れない
        def qmm3():
            """qmm 3 本だけ。down は確定済みの act を読む (鎖にしない)。"""
            return (sw.up_proj(xs, idxs, sorted_indices=True),
                    sw.gate_proj(xs, idxs, sorted_indices=True),
                    sw.down_proj(act, idxs, sorted_indices=True))

        def with_silu():
            u = sw.up_proj(xs, idxs, sorted_indices=True)
            g = sw.gate_proj(xs, idxs, sorted_indices=True)
            return sw.down_proj(sw.activation(u, g), idxs, sorted_indices=True)

        def with_sort():
            xx, ii, _ = SL._gather_sort(mx.expand_dims(x, (-2, -3)), idx)
            u = sw.up_proj(xx, ii, sorted_indices=True)
            g = sw.gate_proj(xx, ii, sorted_indices=True)
            return sw.down_proj(sw.activation(u, g), ii, sorted_indices=True)

        acc["#qmm"] += med_ms(qmm3, reps)
        acc["#silu"] += med_ms(with_silu, reps)
        acc["#sort"] += med_ms(with_sort, reps)
        acc["#glu"] += med_ms(lambda: sw(x, idx), reps)  # + 書き戻し

        del logits, idx, w, xs, idxs, inv, x_up, x_gate, act, dn, us, comb
        mx.clear_cache()
    acc["_pad32"] = statistics.median(pad32)
    acc["_pad64"] = statistics.median(pad64)
    return acc


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
    from mlxturbo.runner import enable_default_fusions, set_wired_limit_default
    from mlxturbo.spec_flash import _PREFILL_GROUP, _group_prefill_forward

    model, tok = load(os.path.expanduser(args.model))
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[prefill-anatomy]")
    # engine を直叩きなので server.py の _load() を経由しない -- 常駐条件を
    # 本番と揃えるため、ここで自前で wire する
    # (mlxturbo/runner.py の set_wired_limit_default 参照)。
    set_wired_limit_default(log_prefix="[prefill-anatomy]")

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
            T, nb, U, n_sel, ur, kvf, true_u = stats[0]
            print(f"  gather 経路が活性: T={T} n_blocks={nb} union={U} ({ur:.0%})"
                  f" 集める列/kv={kvf:.0%} true_union={true_u} ({true_u / nb if nb else 0.0:.0%})",
                  flush=True)

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
        mp = moe_parts(grabbed["moe"], args.reps) if grabbed["moe"] else None
        mlb = moe_bounds(ta, grabbed["moe"][0][0], step) if grabbed["moe"] else None
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

        if mp is None:
            return
        print(f"\n  {'MoE 48 層の内訳':30s}{'実測 ms':>9s}{'FLOP T':>8s}"
              f"{'最小GB':>8s}{'下限':>8s}{'効率':>7s}"
              f"{'動GB':>8s}{'下限2':>8s}{'効率2':>7s}  律速")
        for k in MOE_ORDER:
            flop, byt, low, act, low2, nolap, kind = mlb[k]
            ms = mp[k]
            print(f"  {MOE_LABEL[k]:30s}{ms:9.1f}{flop / 1e12:8.2f}"
                  f"{byt / 1e9:8.1f}{low:8.1f}{low / ms * 100 if ms else 0:6.1f}%"
                  f"{act / 1e9:8.1f}{low2:8.1f}"
                  f"{low2 / ms * 100 if ms else 0:6.1f}%  {kind}")
        msum = sum(mp[k] for k in MOE_ORDER)
        mlow = sum(v[2] for v in mlb.values())
        mlow2 = sum(v[4] for v in mlb.values())
        print(f"  {'  MoE 部品和':30s}{msum:9.1f}"
              f"{sum(v[0] for v in mlb.values()) / 1e12:8.2f}"
              f"{sum(v[1] for v in mlb.values()) / 1e9:8.1f}{mlow:8.1f}"
              f"{mlow / msum * 100:6.1f}%"
              f"{sum(v[3] for v in mlb.values()) / 1e9:8.1f}{mlow2:8.1f}"
              f"{mlow2 / msum * 100:6.1f}%")
        print(f"  下限2 は max(FLOP, 動バイト)。重なりゼロ側の括弧 (FLOP+動バイト)"
              f" は {sum(v[5] for v in mlb.values()):.0f} ms"
              f" (部品和の {sum(v[5] for v in mlb.values()) / msum * 100:.0f}%)")
        lay = mp["_layer"]
        print(f"  {'  MoE 全体 (層ごとに測って合計)':30s}{lay:9.1f}")
        print(f"  {'  MoE 全体 (48 層を 1 eval で)':30s}{res['moe']:9.1f}")
        print(f"  MoE 部品和 - 層ごと合計 = {msum - lay:+.1f} ms"
              f" ({(msum - lay) / lay * 100:+.1f}%)")
        print(f"  層ごと合計 - 48 層まとめ = {lay - res['moe']:+.1f} ms"
              f" ({(lay - res['moe']) / res['moe'] * 100:+.1f}%)")
        print(f"  (参考) タイルの水増し率 中央値: T=32 {mp['_pad32']:.2f} 倍"
              f" / T=64 {mp['_pad64']:.2f} 倍")

        print(f"\n  {'積み上げ (48 層合計)':30s}{'壁時計 ms':>10s}{'増分':>9s}"
              f"{'個別の部品':>11s}{'差':>9s}")
        prev, ladder = 0.0, [
            ("qmm 3 本のみ", "#qmm", ("up", "gate", "down")),
            ("+ SwiGLU", "#silu", ("swiglu",)),
            ("+ 並べ替え", "#sort", ("sort",)),
            ("+ 書き戻し (= SwitchGLU)", "#glu", ("unsort",)),
            ("+ router/top-k/縮約/共有", "_layer",
             ("router", "topk", "combine", "shared")),
        ]
        for name, key, parts in ladder:
            inc = mp[key] - prev
            ind = sum(mp[p] for p in parts)
            print(f"  {name:30s}{mp[key]:10.1f}{inc:9.1f}{ind:11.1f}"
                  f"{inc - ind:9.1f}")
            prev = mp[key]
        print(flush=True)

    for ci in range(n_full):
        if ci in points:
            print(f"\n[chunk {ci}] offset={ci * step} kv={(ci + 1) * step}",
                  flush=True)
            measure(ci)
        mx.eval([model.model(ids[:, ci * step : (ci + 1) * step], cache=cache)]
                + pending(cache))
        mx.clear_cache()
    # 計測ツールなので destructor (スレッドプール等の後始末) に用は無い。
    # interpreter shutdown 待ちでプロセスが Metal のメモリを握ったまま
    # 1 時間以上残った実測があるので、結果を書き終えたら即 _exit で落とす
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
