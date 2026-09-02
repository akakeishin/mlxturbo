"""PLE 1 層 (n-gram 込み) の 167ms/チャンク (S=2048) がどこに行くかを割る。

事実 (`tools/prefill_anatomy.py --ctx 8000`、
`bench/results/logs/prefill-anatomy-8k-0903.log`): 「PLE 1 層 (n-gram 込み)」
が 166.8/167.0/166.7 ms/チャンク、FLOP 効率 7%。`MLXTURBO_NGRAM_PREFETCH=1`
(次チャンクの行の先読み) の A/B は prefill -0.9% で、pread の I/O 待ちが
主ではないと分かっている。ここでは同じ実チャンクを使って、167ms の中身
(ハッシュ計算・GPU 同期・pread・復号・射影/short_conv) を割り出す。

## prefill_anatomy.py と違うところ

- **層 1 個の内部を割る** (prefill_anatomy は層をまたいだ部品の内訳)。
- **I/O を含む部品 (pread) は N 回回して中央値、を使わない。**同じ行を
  何度も読むとページキャッシュに乗って不当に速くなる (`tools/ngram_pread_bench.py`
  の docstring と同じ罠)。pread はコールド (実フォワード中の 1 回だけ) の
  値を使い、ウォーム再測は参考値として別に出す。
- **ハッシュ計算 (a) と GPU 同期 (b) は現状コードでは 1 点に融合している。**
  `NGramEmbedding.ngram_ids` が返す `gid` は遅延グラフのままで、最初に
  評価されるのは `StreamNGram.__call__` の `np.array(gid.reshape(-1))` --
  そこでハッシュの計算 (a) と同期 (b) が同時に走る。これを割るため、
  ここでは実フォワードの外で `gid` を先に `mx.eval` してから (a) を計測し、
  評価済みの `gid` に対する `np.array` 変換だけを (b) として測る (計算が
  終わった後の変換だけを切り出す)。実フォワード側は素のまま (未評価の
  `gid` を渡す) にして `StreamNGram.stats["sync_ms"]` (a+b 融合) と
  `["fetch_ms"]` (c) をそのまま読み、突き合わせに使う。

## 内訳の切り方 (指示どおり a〜e)

    a  ngram_ids のハッシュ計算 (shift/xor/mod の mx op 列、GPU)
    b  np.array の同期 (評価済み gid に対する変換だけ)
    c  pread のバッチ読み (I/O、スレッド 12、コールド)
    d  読んだ行の復号 (4bit -> bf16 の dequantize、embed 変数上に成立)
       + reshape。dequantize 自体が「packed*scale + bias」の加算を含む
    e  `_short_conv` と残り (key/value 射影・RMSNorm x3・gate・sigmoid・
       乗算・reshape・norm_conv・short_conv・最終加算)

各部品は 3 通りの数字を出す:

- **isolated replay** (a/b/d/e): 実チャンクで捕まえた実引数を使い、
  部品そのものを N 回回して中央値を取る (I/O を含まないので安全に繰り返せる)。
- **実フォワード内の生値**: `StreamNGram.stats` の sync_ms/fetch_ms と、
  `NGramEmbedding.__call__` / `PLELayer.__call__` を包んで強制 eval した
  区間時間 (embed 計・layer 計)。d と e はここから引き算でも出せるので、
  isolated replay の値と付き合わせて「部品和 ≈ 壁時計」を確認する。
- **壁時計**: 対象チャンクを実フォワードで 1 回だけ流したときの、
  `PLELayer.__call__` の区間時間 (強制 eval、コールド)。

## 触らないもの

`mlxturbo/_vendor/qwen4_exp.py` と `mlxturbo/ngram_stream.py` は読むだけで
変えない。ここでのクラスメソッド差し替えは対象チャンクを流す一瞬だけで、
try/finally で必ず元に戻す (prefill_anatomy.py の `wrap`/restore と同じ作法)。

使い方 (StreamNGram = interleaved レイアウトのサイドカーが要る。
`ddalcu-ngram-sep` は RamNGram に振り分けられるので対象外):

    tools/biglock.sh .venv/bin/python tools/ple_split.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram --ctx 8000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def med_ms(fn, reps):
    """温めの 1 本は捨て、reps 回の中央値を取る (prefill_anatomy.py と同じ)。

    I/O を含まない (GPU 計算だけの) 部品にのみ使うこと -- 同じ入力を
    繰り返すのでページキャッシュの影響は無いが、I/O のあるものに使うと
    2 本目以降が不当に速く出る。
    """
    import mlx.core as mx

    mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def pending(caches):
    """キャッシュに残っている遅延ノードを eval 対象として集める
    (prefill_anatomy.py と同一。対象チャンクより手前を進める段でも
    indexer のバッファが後ろのチャンクへ帰属を漏らさないために要る)。
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm")
    ap.add_argument("--ngram", default="~/models/ddalcu-ngram",
                     help="interleaved レイアウトのサイドカー (StreamNGram/pread 経路)")
    ap.add_argument("--ctx", type=int, default=8000)
    ap.add_argument("--chunk", type=int, default=2048, help="prefill チャンク幅")
    ap.add_argument("--chunk-index", type=int, default=2,
                     help="実引数を捕まえるチャンク番号 (0 始まり)")
    ap.add_argument("--reps", type=int, default=5,
                     help="isolated replay (I/O を含まない部品) の中央値を取るレップ数")
    ap.add_argument("--out", default=str(REPO_ROOT / "bench/results/ple-split.json"))
    args = ap.parse_args()

    # NGRAM_ON_DISK はモジュール読み込み時に固定される定数
    # (mlxturbo/_vendor/qwen4_exp.py の `NGRAM_ON_DISK = os.environ.get(...)`)
    # ので、mlx / mlxturbo を import する前に立てること。
    os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    import mlxturbo  # noqa: F401  -- qwen4_exp を _vendor 版に差し替える
    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo.ngram_stream import StreamNGram, install
    from mlxturbo.runner import enable_default_fusions, set_wired_limit_default

    model, tok = load(os.path.expanduser(args.model))
    install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[ple-split]")
    set_wired_limit_default(log_prefix="[ple-split]")

    ple_layer_idx = model.model.ple_layers[0]
    ple_mod = model.model.layers[ple_layer_idx].ple
    ng = ple_mod.ple_embedding.ngram_embedding
    if not isinstance(ng, StreamNGram):
        print(f"このツールは StreamNGram (pread 経路) 専用。渡されたサイドカーは "
              f"{type(ng).__name__} に振り分けられた (--ngram が layout=separate かも)")
        return 1

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from _bench_text import long_prompts

    body = long_prompts(tok, args.ctx, ["上の文書の要点を 5 つに整理してください。"])[0]
    ids_full = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True))[None]
    n_total = ids_full.shape[1]
    step = args.chunk
    n_full = n_total // step
    target = args.chunk_index
    if target >= n_full:
        print(f"完全チャンクが {n_full} 本しか無い。--chunk-index を減らすか --ctx を増やすこと")
        return 1
    print(f"ctx={n_total} chunk={step} 完全チャンク={n_full} 対象チャンク={target} "
          f"reps={args.reps} ple_layer_idx={ple_layer_idx} "
          f"ngram_heads={ple_mod.ple_embedding.ngram_heads} dim={ng.dim}", flush=True)

    cache = model.make_cache()

    # -- 対象チャンクの手前まで実際の prefill と同じ進め方で進める (計測しない)
    for ci in range(target):
        chunk = ids_full[:, ci * step:(ci + 1) * step]
        mx.eval([model.model(chunk, cache=cache)] + pending(cache))
        mx.clear_cache()

    chunk = ids_full[:, target * step:(target + 1) * step]
    pc = cache[ple_layer_idx]
    pre_conv_state = pc[2]  # 対象チャンクへ入る直前の short-conv 状態 (replay の基点)

    # ---- 対象チャンクを実フォワードで 1 回だけ流し、実引数と壁時計を捕まえる ----
    #
    # NGramEmbedding.__call__ と PLELayer.__call__ を一瞬だけ包んで、
    # 強制 eval した区間時間を測る。StreamNGram.__call__ 自体は素のまま
    # (gid を未評価で渡す) にするので、そちらの sync_ms/fetch_ms は
    # 本番と同じ意味 (sync_ms は a+b 融合、fetch_ms は c 純粋) のまま読める。
    captured: dict = {}
    layer_ms = [0.0]
    embed_ms = [0.0]
    orig_embed_call = Q.NGramEmbedding.__call__
    orig_layer_call = Q.PLELayer.__call__

    def embed_wrap(self, ids_, prev_context):
        t0 = time.perf_counter()
        out = orig_embed_call(self, ids_, prev_context)
        mx.eval(out)
        embed_ms[0] += (time.perf_counter() - t0) * 1000
        return out

    def layer_wrap(self, hidden, ids_, prev_ctx, cache_):
        captured["self"] = self
        captured["hidden"] = hidden
        captured["ids"] = ids_
        captured["prev_ctx"] = prev_ctx
        captured["cache"] = cache_
        # `hidden` はここまで (embed_tokens + tile + レイヤー 0 の全部) が
        # 遅延グラフのまま。PLE (レイヤー 1) は `query = norm_query(hidden)`
        # で hidden を読むので、ここで先に確定させておかないと、
        # 直後の `mx.eval(out)` が「レイヤー 0 の壁時計」まで PLE の時間として
        # 数えてしまう (実測でこれを踏んだ: 最初の版は 360.9ms を「PLE 1 層」
        # として報告したが、e (射影+short_conv) の isolated replay 23.85ms と
        # 突き合わせると 78ms 分の説明が付かず、原因が hidden の遅延評価だと
        # 分かった)。ここでの eval はレイヤー 0 の費用を確定させるだけで、
        # PLE 自身の時間には数えない。
        mx.eval(hidden)
        t0 = time.perf_counter()
        out = orig_layer_call(self, hidden, ids_, prev_ctx, cache_)
        mx.eval(out)
        layer_ms[0] += (time.perf_counter() - t0) * 1000
        return out

    ng.reset_stats()
    Q.NGramEmbedding.__call__ = embed_wrap
    Q.PLELayer.__call__ = layer_wrap
    try:
        out = model.model(chunk, cache=cache)
        mx.eval([out] + pending(cache))
    finally:
        Q.NGramEmbedding.__call__ = orig_embed_call
        Q.PLELayer.__call__ = orig_layer_call
    mx.clear_cache()

    if not captured:
        print("PLELayer.__call__ が 1 度も呼ばれなかった (ple_layers が空？)")
        return 1
    assert captured["self"] is ple_mod

    sync_ms_real = ng.stats["sync_ms"]     # a+b 融合 (現状コードの sync_ms)
    fetch_ms_real = ng.stats["fetch_ms"]   # c 純粋
    layer_total_ms = layer_ms[0]           # PLE 1 層の壁時計 (コールド)
    embed_total_ms = embed_ms[0]           # a+b+c+d
    d_decode_derived_ms = embed_total_ms - sync_ms_real - fetch_ms_real
    e_rest_derived_ms = layer_total_ms - embed_total_ms

    hidden = captured["hidden"]
    ids_ = captured["ids"]
    prev_ctx = captured["prev_ctx"]
    cache_entry = captured["cache"]
    n_new = ids_.shape[1]

    # ---- (a) ngram_ids のハッシュ計算だけを切り出す (isolated replay) ----
    def hash_once():
        history = mx.concatenate([prev_ctx, ids_], axis=1)
        return ple_mod.ple_embedding.ngram_ids(history)[:, -n_new:]

    a_replay_ms = med_ms(hash_once, args.reps)
    gid = hash_once()
    mx.eval(gid)  # 以降 (b) はこの評価済み gid を使う -- 計算とは切り離す
    mx.clear_cache()

    # ---- (b) 同期後の np.array 変換だけ (gid は評価済み、GPU 待ちは無い) ----
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        flat = np.array(gid.reshape(-1), copy=False).astype(np.int64)
        ts.append((time.perf_counter() - t0) * 1000)
    b_replay_ms = statistics.median(ts)
    flat = np.array(gid.reshape(-1), copy=False).astype(np.int64)

    # ---- (c) pread のバッチ読み。コールドは実フォワードの fetch_ms_real を使う。
    #      ここでの再測は同じ行なのでページキャッシュに乗る -- 参考値として
    #      別枠で出し、部品和には混ぜない。
    t0 = time.perf_counter()
    rec = ng._gather_cached(flat)
    c_warm_first_ms = (time.perf_counter() - t0) * 1000
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        ng._gather_cached(flat)
        ts.append((time.perf_counter() - t0) * 1000)
    c_warm_median_ms = statistics.median(ts)

    # ---- (d) 復号 4bit -> bf16 (dequantize) + reshape (isolated replay) ----
    def decode_once():
        rows_n = rec.shape[0]
        w = mx.array(rec[:, :ng.wb].copy().view(np.uint32).reshape(rows_n, ng.npack))
        s = mx.array(
            rec[:, ng.wb:ng.wb + ng.sb].copy().view(np.uint16).reshape(rows_n, ng.ngrp)
        ).view(mx.bfloat16)
        b = mx.array(
            rec[:, ng.wb + ng.sb:].copy().view(np.uint16).reshape(rows_n, ng.ngrp)
        ).view(mx.bfloat16)
        out_ = mx.dequantize(w, s, b, group_size=ng.group_size, bits=ng.bits)
        return out_.reshape(*gid.shape, ng.dim)

    d_replay_ms = med_ms(decode_once, args.reps)
    emb_raw = decode_once()
    mx.eval(emb_raw)
    emb_full = emb_raw.reshape(*gid.shape[:2], -1)  # NGramEmbedding.__call__ の最終形
    mx.clear_cache()

    # ---- (e) 射影+ゲート / norm_conv / short_conv / 最終加算 (isolated replay) ----
    def proj_gate():
        emb = emb_full.astype(hidden.dtype)
        key = ple_mod.norm_key(ple_mod.key_proj(emb))
        key = key.reshape(*key.shape[:-1], ple_mod.hc, ple_mod.d)
        value = ple_mod.value_proj(emb)
        query = ple_mod.norm_query(hidden)
        query = query.reshape(*query.shape[:-1], ple_mod.hc, ple_mod.d)
        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(ple_mod.d)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        return gated.reshape(*gated.shape[:-2], -1)

    proj_gate_ms = med_ms(proj_gate, args.reps)
    gated_fixed = proj_gate()
    mx.eval(gated_fixed)

    def short_conv_and_add():
        cache_entry[2] = pre_conv_state
        return gated_fixed + ple_mod._short_conv(ple_mod.norm_conv(gated_fixed), cache_entry)

    mx.eval(short_conv_and_add())  # 温め
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        r = short_conv_and_add()
        mx.eval(r)
        ts.append((time.perf_counter() - t0) * 1000)
    short_conv_rest_ms = statistics.median(ts)
    cache_entry[2] = pre_conv_state  # 使い終わったら実キャッシュを元に戻す
    mx.clear_cache()

    e_replay_ms = proj_gate_ms + short_conv_rest_ms

    # ---- 参考: 同じ実引数で PLELayer.__call__ を何度か回した「ウォーム」壁時計。
    #      これがコールドの layer_total_ms よりだいぶ速ければ、pread のページ
    #      キャッシュが壁時計に効いているということ。
    def layer_call_once():
        cache_entry[2] = pre_conv_state
        return ple_mod(hidden, ids_, prev_ctx, cache_entry)

    mx.eval(layer_call_once())
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        r = layer_call_once()
        mx.eval(r)
        ts.append((time.perf_counter() - t0) * 1000)
    layer_warm_median_ms = statistics.median(ts)
    cache_entry[2] = pre_conv_state
    mx.clear_cache()

    # ---- 表 ----
    rows = [
        ("a  ngram_ids のハッシュ計算 (GPU)", a_replay_ms),
        ("b  np.array の同期 (評価済み配列の変換のみ)", b_replay_ms),
        ("c  pread バッチ読み (コールド、実フォワード内の実測)", fetch_ms_real),
        ("d  復号 4bit->bf16 (dequantize) + reshape", d_replay_ms),
        ("e  射影+ゲート/norm_conv/short_conv/最終加算", e_replay_ms),
    ]
    parts = sum(v for _, v in rows)
    print(f"\n[chunk {target}] offset={target * step} kv={(target + 1) * step} S={n_new}")
    print(f"  {'部品':46s}{'ms':>9s}{'割合':>8s}")
    for label, ms in rows:
        print(f"  {label:46s}{ms:9.2f}{ms / layer_total_ms * 100:7.1f}%")
    print(f"  {'部品和':46s}{parts:9.2f}{parts / layer_total_ms * 100:7.1f}%")
    print(f"  {'壁時計 (実フォワード、コールド、PLE 1 層)':46s}{layer_total_ms:9.2f}")
    gap = (parts - layer_total_ms) / layer_total_ms * 100
    print(f"  部品和 - 壁時計 = {parts - layer_total_ms:+.2f} ms ({gap:+.1f}%)")
    if abs(gap) > 10:
        print("  ** 10% を超えてずれている。見えていない項目があるか、"
              "isolated replay が実フォワードの重なりを再現できていない **")

    print("\n  実フォワード内の生値 (突き合わせ用):")
    print(f"    embed 計 (a+b+c+d、強制 eval)              {embed_total_ms:9.2f} ms")
    print(f"    ng.stats sync_ms (a+b 融合、未分離)          {sync_ms_real:9.2f} ms")
    print(f"    ng.stats fetch_ms (c)                     {fetch_ms_real:9.2f} ms")
    print(f"    embed 計から引き算した d (embed-sync-fetch)   {d_decode_derived_ms:9.2f} ms"
          f"  [isolated replay の d との差 {d_replay_ms - d_decode_derived_ms:+.2f} ms]")
    print(f"    layer 計 - embed 計 (= e の実測)              {e_rest_derived_ms:9.2f} ms"
          f"  [isolated replay の e との差 {e_replay_ms - e_rest_derived_ms:+.2f} ms]")
    print(f"    {ng.stats_line()}")
    print(f"    sync_ms(実測,a+b) - a(replay) = {sync_ms_real - a_replay_ms:+.2f} ms"
          f"  (理論上の純 b。isolated replay の b={b_replay_ms:.2f} ms と近ければ辻褄が合う)")

    print("\n  e の内訳 (参考、(e) 表示行の中身):")
    print(f"    射影+ゲート (key/value/gate/sigmoid)      {proj_gate_ms:9.2f} ms")
    print(f"    norm_conv+short_conv+最終加算            {short_conv_rest_ms:9.2f} ms")

    print("\n  ページキャッシュの影響 (参考、部品和には混ぜない):")
    print(f"    pread ウォーム再測 1 回目                 {c_warm_first_ms:9.2f} ms")
    print(f"    pread ウォーム再測 中央値 ({args.reps} 回)      {c_warm_median_ms:9.2f} ms"
          f"  (コールド {fetch_ms_real:.2f} ms との差 {fetch_ms_real - c_warm_median_ms:+.2f} ms)")
    print(f"    PLE 層ウォーム壁時計 中央値 ({args.reps} 回)    {layer_warm_median_ms:9.2f} ms"
          f"  (コールド {layer_total_ms:.2f} ms との差 {layer_total_ms - layer_warm_median_ms:+.2f} ms)",
          flush=True)

    result = {
        "model": args.model, "ngram": args.ngram, "ctx": n_total, "chunk": step,
        "chunk_index": target, "n_new": n_new, "reps": args.reps,
        "wall_ms_cold": layer_total_ms,
        "parts_ms": {
            "a_hash": a_replay_ms, "b_sync": b_replay_ms,
            "c_pread_cold": fetch_ms_real, "d_decode": d_replay_ms,
            "e_rest": e_replay_ms,
        },
        "parts_sum_ms": parts,
        "gap_pct": gap,
        "real_forward": {
            "embed_total_ms": embed_total_ms,
            "sync_ms_real": sync_ms_real,
            "fetch_ms_real": fetch_ms_real,
            "d_decode_derived_ms": d_decode_derived_ms,
            "e_rest_derived_ms": e_rest_derived_ms,
        },
        "e_breakdown": {
            "proj_gate_ms": proj_gate_ms,
            "short_conv_rest_ms": short_conv_rest_ms,
        },
        "pread_warm_ms": {
            "first": c_warm_first_ms, "median": c_warm_median_ms,
        },
        "layer_warm_median_ms": layer_warm_median_ms,
        "ngram_stats": dict(ng.stats),
    }
    out_path = Path(os.path.expanduser(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n書いた: {out_path}", flush=True)

    # 計測ツールなので interpreter shutdown 待ち (StreamNGram のスレッドプール
    # join) でプロセスが残らないよう、結果を書き終えたら即 _exit で落とす
    # (prefill_anatomy.py と同じ理由)。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
