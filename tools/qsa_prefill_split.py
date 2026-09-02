"""full-attention 層 1 層 (indexer 込み) の時間を 4 部品に割る。

    (a) indexer   QSAIndexer._pooled_and_top -- pooled key の作成・q・k
                  スコア・可視判定・argpartition (:387-447)
    (b) mask/選択  dense 経路: QSAIndexer.__call__ のブロック→トークン展開
                  (:449-505) + Attention._final_mask の mask & sparse
                  (:565-586)。gather 経路 (段 3(b)、既定 on): select_blocks
                  の展開ぶん (ほぼ (a) と同額) + _gather_tile_attn のうち
                  sdpa 呼び出しを除いた分 (union・argsort・take_along_axis
                  による K/V の gather、keep_sel の構築)
    (c) sdpa      mx.fast.scaled_dot_product_attention 本体 (base.py 経由)。
                  head_dim 256 は MLX 0.32.2 で fallback (スコア実体化) --
                  `tools/sdpa_headdim_micro.py` に同じ形の合成マイクロがある。
                  decode 幅 (S<=8) は vector カーネル (bool マスクでキー読みを
                  スキップする経路) に入るので、prefill 幅とは別物
    (d) その他     Attention._qkv (q/k/v 射影・QK-norm・rope・KV キャッシュ
                  更新) + 末尾の o_proj(out * sigmoid(gate))

kv (= 過去の文脈 + このステップの新規トークン数 S) は `--kvs` で指定した
点を **昇順に 1 回の prefill で通過しながら** 測る -- 50k のような長い kv を
毎回ゼロから prefill すると分単位で伸びるので、共有 1 本のキャッシュを
ランプアップ (`--chunk` 幅、既定 2048 = 本番の PREFILL_STEP_SIZE) で進め、
目標 kv に達したところで `--S` 幅 (既定 2048 = prefill チャンク。decode を見る
なら 1/2/4) の 1 ステップだけ計測してから、その同じステップを本物の
(ラップしない) forward でもう一度実行してキャッシュを確定させ、次の kv へ
進む。indexer の pooled 増分キャッシュ (`_IndexerCache`) はこの一連の流れの
中で本番と同じように「まだ確定していないブロックだけ新規計算」される。

対象は 12 層 (既定構成) のうち **最初の** full-attention 層だけ。同じ層クラス
の他 11 層は素通り (計測に混ざらない) -- `Attention.__call__` / `_qkv` /
`_final_mask` / `_gather_tile_attn` / `QSAIndexer.__call__` / `.select_blocks`
/ `._pooled_and_top` を一時的にラップし、`self is <対象インスタンス>` の
ときだけ引数を控える。**QSAIndexer の中身は変えない (ラップして呼ぶだけ)。**
`Attention._positions` などのシームも同様、差し替えではなく呼び出しの前後で
時間を取るだけ。

部品ごとに「その部品だけを走らせて mx.eval」する形で `--reps` 回 (既定 5) の
中央値 (ms) を取る。(b) は 2 つの部品 (指標を構成する外側のメソッド全体 と
(a) or (c)) を独立に測ってから引き算で出す -- この引き算そのものは
`tools/prefill_anatomy.py` の MoE 内訳 (層ごと合計 vs 積み上げ) と同じ流儀で、
隠さず表に出す。(d) は `_qkv` を直接測り、末尾の gate/o_proj は sdpa の実出力
(捕まえた実配列、計測区間の外で eval 確定済み) と qkv の実 gate を使って
`o_proj(out.transpose(0,2,1,3).reshape(B,S,-1) * sigmoid(gate))` を直接測る
(qwen4_exp.py:964-965 / :909-910 と同じ式。QSAIndexer/Attention の実装は
変えていない -- 計測用に実配列で同じ式をもう一度評価するだけ)。

**部品和 ≈ 壁時計 (層 1 つを実引数で 1 回呼んだ時間) を必ず確認する**
(CLAUDE.md の計測の作法)。壁時計は `Attention.__call__` を捕まえた実引数
(x, rope, mask, cache, idx_cache) でそのまま呼び直したもので、部品側が拾って
いない host 側分岐 (`_positions`、gather 可否のホスト算数、decode の
split_mask 用 causal 配列組み --- いずれも小さいはず) はここでは部品に
割り当てず、差として出す。既知の限界として明記する。

12 層換算 (`x12`) と、12 層ぶんを 1 トークンあたりに割った `ms_per_tok_x12`
(= 部品 ms * n_attn / S) も出す -- decode の ms/tok が kv とともに伸びる分の
うち、attention のどの部品が効いているかを直接比べるための列。

    tools/biglock.sh .venv/bin/python tools/qsa_prefill_split.py \\
        --model ~/models/ddalcu-mlxlm --S 2048 --kvs 2048,8192,16896

    tools/biglock.sh .venv/bin/python tools/qsa_prefill_split.py \\
        --model ~/models/ddalcu-mlxlm --S 2 --kvs 4096,17000,25000,50000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def wrap_instance(cls, name, is_target, calls, bucket):
    """`cls.name` を一時的にラップし、`self` が対象インスタンスのときだけ
    `(self, args, 戻り値)` を `calls[bucket]` に積む。**呼ばれるたびに
    必ず元の実装を呼ぶ** (対象以外のインスタンスの挙動は変えない)。
    戻り値は素の `getattr` (= 元の未ラップ関数、再ラップ時の巻き込み防止)。
    """
    orig_fn = getattr(cls, name)

    def g(self, *a):
        r = orig_fn(self, *a)
        if is_target(self):
            calls[bucket].append((self, a, r))
        return r

    setattr(cls, name, g)
    return orig_fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--S", type=int, default=2048,
                     help="計測する 1 ステップの新規トークン数。"
                          "2048=prefill チャンク、1/2/4=decode 幅")
    ap.add_argument("--kvs", default="2048,8192,16896",
                     help="kv (= 過去の文脈 + S) の一覧。昇順でなくてよい"
                          "(内部でソートする)。カンマ区切り")
    ap.add_argument("--chunk", type=int, default=2048,
                     help="kv の間を埋めるランプアップ prefill のチャンク幅"
                          " (本番の PREFILL_STEP_SIZE と同じ既定)")
    ap.add_argument("--reps", type=int, default=5, help="中央値を取るレップ数")
    ap.add_argument("--out", default=None,
                     help="既定: bench/results/qsa-attn-split-S<S>.json")
    args = ap.parse_args()

    kvs = sorted({int(v) for v in args.kvs.split(",") if v.strip() != ""})
    if not kvs:
        print("--kvs が空")
        return 1
    if kvs[0] < args.S:
        print(f"最小の kv ({kvs[0]}) が S ({args.S}) 未満 -- kv = offset + S "
              f"なので kv >= S が必要")
        return 1
    for i in range(1, len(kvs)):
        if kvs[i] - kvs[i - 1] < args.S:
            print(f"--kvs の隣接差が S={args.S} 未満 ({kvs[i - 1]} -> {kvs[i]})"
                  f": 前の点を消化しきる前に次の offset に届かない")
            return 1

    out_path = Path(args.out) if args.out else Path(
        f"bench/results/qsa-attn-split-S{args.S}.json")
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401  (arch_registry の meta_path フックを張る)
    import mlx_lm.models.qwen4_exp as Q
    from mlxturbo.runner import enable_default_fusions

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from _bench_text import long_prompts
    import prefill_anatomy as PA  # snapshot/pending/restore/med_ms を借りる

    model, tok = load(os.path.expanduser(args.model))
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[qsa-attn-split]")

    ta = model.args.text
    layer_types = ta.layer_types
    if "full_attention" not in layer_types:
        print("full_attention 層が無いアーキテクチャ")
        return 1
    target_layer_idx = layer_types.index("full_attention")
    n_attn = sum(1 for t in layer_types if t == "full_attention")
    target_attn = model.model.layers[target_layer_idx].self_attn
    target_idx = target_attn.indexer

    max_kv = kvs[-1]
    body = long_prompts(
        tok, max_kv + 256, ["上の文書の要点を 5 つに整理してください。"]
    )[0]
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True))[None]
    if ids.shape[1] < max_kv:
        print(f"プロンプトが {ids.shape[1]} tok しかなく、必要な {max_kv}"
              f" に届かない (--kvs を減らすこと)")
        return 1

    cache = model.make_cache()
    layer_cache = cache[target_layer_idx]

    mode = "decode 幅 (vector カーネル想定)" if args.S <= 8 else \
        "prefill 幅 (head_dim 256 fallback 想定)"
    print(f"S={args.S} [{mode}] kvs={kvs} chunk={args.chunk} reps={args.reps}"
          f" target_layer={target_layer_idx} n_attn={n_attn}"
          f" indexer_budget={target_idx.token_budget}"
          f" compress_ratio={target_idx.compress_ratio}", flush=True)

    def measure_point(offset: int) -> dict:
        """kv = offset + args.S の 1 点を測る。キャッシュは非破壊 (呼び出し
        側が snapshot/restore する必要は無い -- 中で完結する)。
        """
        S = args.S
        chunk = ids[:, offset: offset + S]

        calls = {k: [] for k in (
            "pooled", "idx_call", "select_blocks", "final_mask", "qkv",
            "gather_tile",
        )}
        sdpa_calls = []  # (args, kwargs, 戻り値) の list。self を持たない
        capture_on = [False]
        whole_box = {}

        orig = {
            "pooled": wrap_instance(
                Q.QSAIndexer, "_pooled_and_top",
                lambda s: s is target_idx, calls, "pooled"),
            "idx_call": wrap_instance(
                Q.QSAIndexer, "__call__",
                lambda s: s is target_idx, calls, "idx_call"),
            "select_blocks": wrap_instance(
                Q.QSAIndexer, "select_blocks",
                lambda s: s is target_idx, calls, "select_blocks"),
            "final_mask": wrap_instance(
                Q.Attention, "_final_mask",
                lambda s: s is target_attn, calls, "final_mask"),
            "qkv": wrap_instance(
                Q.Attention, "_qkv",
                lambda s: s is target_attn, calls, "qkv"),
            "gather_tile": wrap_instance(
                Q.Attention, "_gather_tile_attn",
                lambda s: s is target_attn, calls, "gather_tile"),
        }

        orig_call = Q.Attention.__call__

        def wrapped_call(self, *a):
            if self is not target_attn:
                return orig_call(self, *a)
            capture_on[0] = True
            try:
                r = orig_call(self, *a)
            finally:
                capture_on[0] = False
            whole_box["args"] = a
            return r

        Q.Attention.__call__ = wrapped_call

        orig_sdpa = Q.scaled_dot_product_attention

        def wrapped_sdpa(*a, **kw):
            r = orig_sdpa(*a, **kw)
            if capture_on[0]:
                sdpa_calls.append((a, kw, r))
            return r

        Q.scaled_dot_product_attention = wrapped_sdpa

        pre = PA.snapshot(cache)
        out = model.model(chunk, cache=cache)
        mx.eval([out] + PA.pending(cache))

        Q.QSAIndexer._pooled_and_top = orig["pooled"]
        Q.QSAIndexer.__call__ = orig["idx_call"]
        Q.QSAIndexer.select_blocks = orig["select_blocks"]
        Q.Attention._final_mask = orig["final_mask"]
        Q.Attention._qkv = orig["qkv"]
        Q.Attention._gather_tile_attn = orig["gather_tile"]
        Q.Attention.__call__ = orig_call
        Q.scaled_dot_product_attention = orig_sdpa

        PA.restore(cache, pre)
        mx.clear_cache()

        dense = len(calls["final_mask"]) > 0
        gather = len(calls["gather_tile"]) > 0
        if not dense and not gather:
            print("  ** 警告: dense (_final_mask) でも gather"
                  " (_gather_tile_attn) でも捕まらなかった。既定外の融合"
                  " カーネル (MLXTURBO_PREFILL_ATTN など) が有効になっていないか"
                  " 確認すること **")
            branch = "unknown"
        else:
            branch = "dense" if dense else "gather"

        def bench_layer(pairs):
            """[(元関数, self, args), ...] を層キャッシュだけ退避して測る。"""
            if not pairs:
                return 0.0

            def run():
                st = PA.snapshot([layer_cache])
                outs = [fn(self_obj, *a) for fn, self_obj, a in pairs]
                mx.eval(outs + PA.pending([layer_cache]))
                PA.restore([layer_cache], st)
                return outs

            ms = PA.med_ms(run, args.reps)
            mx.clear_cache()
            return ms

        def bench_bucket(key):
            items = calls[key]
            fn = orig[key]
            return bench_layer([(fn, s, a) for s, a, _ in items])

        def bench_sdpa():
            if not sdpa_calls:
                return 0.0

            def run():
                st = PA.snapshot([layer_cache])
                outs = [orig_sdpa(*a, **kw) for a, kw, _ in sdpa_calls]
                mx.eval(outs + PA.pending([layer_cache]))
                PA.restore([layer_cache], st)
                return outs

            ms = PA.med_ms(run, args.reps)
            mx.clear_cache()
            return ms

        a_ms = bench_bucket("pooled")
        c_ms = bench_sdpa()
        qkv_ms = bench_bucket("qkv")

        b_detail = {}
        if dense:
            idx_total = bench_bucket("idx_call")
            fm_ms = bench_bucket("final_mask")
            b_ms = max(idx_total - a_ms, 0.0) + fm_ms
            b_detail = {"idx_call_total_ms": idx_total, "final_mask_ms": fm_ms}
        elif gather:
            sel_total = bench_bucket("select_blocks")
            gta_total = bench_bucket("gather_tile")
            b_ms = max(sel_total - a_ms, 0.0) + max(gta_total - c_ms, 0.0)
            b_detail = {"select_blocks_total_ms": sel_total,
                        "gather_tile_total_ms": gta_total}
        else:
            b_ms = 0.0

        # -- (d) の後半: gate + o_proj。sdpa の実出力と qkv の実 gate を使って
        #    qwen4_exp.py:964-965 (dense) / :909-910 (gather、同じ式) と同じ
        #    計算をもう一度評価するだけ (実装を書き換えたり差し替えたりはしない)。
        sdpa_outs = [r for (_, _, r) in sdpa_calls]
        if len(sdpa_outs) == 1:
            out_val = sdpa_outs[0]
        elif len(sdpa_outs) > 1:
            out_val = mx.concatenate(sdpa_outs, axis=2)  # decode の split_mask
        else:
            out_val = None
        gate_val = calls["qkv"][0][2][3] if calls["qkv"] else None

        tail_ms = 0.0
        if out_val is not None and gate_val is not None:
            mx.eval(out_val, gate_val)
            B = chunk.shape[0]

            def tail():
                o = out_val.transpose(0, 2, 1, 3).reshape(B, S, -1)
                return [target_attn.o_proj(o * mx.sigmoid(gate_val))]

            tail_ms = PA.med_ms(tail, args.reps)
            mx.clear_cache()

        d_ms = qkv_ms + tail_ms

        whole_ms = 0.0
        if "args" in whole_box:
            wargs = whole_box["args"]

            def run_whole():
                st = PA.snapshot([layer_cache])
                r = orig_call(target_attn, *wargs)
                mx.eval([r] + PA.pending([layer_cache]))
                PA.restore([layer_cache], st)
                return [r]

            whole_ms = PA.med_ms(run_whole, args.reps)
            mx.clear_cache()

        parts = a_ms + b_ms + c_ms + d_ms
        gap_pct = (parts - whole_ms) / whole_ms * 100 if whole_ms else 0.0

        comps = {"indexer": a_ms, "mask": b_ms, "sdpa": c_ms, "other": d_ms}
        return {
            "S": S, "kv": offset + S, "offset": offset, "branch": branch,
            "n_sdpa_calls": len(sdpa_calls),
            "indexer_ms": a_ms, "mask_ms": b_ms, "sdpa_ms": c_ms,
            "other_ms": d_ms, "qkv_ms": qkv_ms, "tail_ms": tail_ms,
            "b_detail": b_detail,
            "parts_sum_ms": parts, "whole_layer_ms": whole_ms,
            "gap_pct": gap_pct,
            "x12": {k: v * n_attn for k, v in comps.items()},
            "ms_per_tok_x12": {k: v * n_attn / S for k, v in comps.items()},
        }

    def advance(lo: int, hi: int) -> None:
        """[lo, hi) を `--chunk` 幅で本物の forward で進める (ラップ無し)。"""
        i = lo
        while i < hi:
            j = min(i + args.chunk, hi)
            mx.eval([model.model(ids[:, i:j], cache=cache)] + PA.pending(cache))
            mx.clear_cache()
            i = j

    def print_point(p: dict) -> None:
        print(f"\n[kv={p['kv']} offset={p['offset']} S={p['S']}"
              f" 経路={p['branch']} sdpa呼び出し={p['n_sdpa_calls']}]",
              flush=True)
        print(f"  {'部品':28s}{'実測 ms':>10s}{'x12 ms':>10s}"
              f"{'x12 ms/tok':>12s}")
        label = {"indexer": "(a) indexer (pooled+top)",
                 "mask": "(b) mask/選択構築",
                 "sdpa": "(c) sdpa 本体",
                 "other": "(d) その他 (qkv+gate+o_proj)"}
        for k in ("indexer", "mask", "sdpa", "other"):
            ms = {"indexer": p["indexer_ms"], "mask": p["mask_ms"],
                  "sdpa": p["sdpa_ms"], "other": p["other_ms"]}[k]
            print(f"  {label[k]:28s}{ms:10.3f}{p['x12'][k]:10.2f}"
                  f"{p['ms_per_tok_x12'][k]:12.4f}")
        print(f"  {'  うち qkv':28s}{p['qkv_ms']:10.3f}")
        print(f"  {'  うち gate+o_proj':28s}{p['tail_ms']:10.3f}")
        if p["branch"] == "dense":
            print(f"  (b) 内訳: QSAIndexer.__call__ 全体="
                  f"{p['b_detail'].get('idx_call_total_ms', 0.0):.3f}ms"
                  f" - indexer={p['indexer_ms']:.3f}ms"
                  f" + final_mask={p['b_detail'].get('final_mask_ms', 0.0):.3f}ms")
        elif p["branch"] == "gather":
            print(f"  (b) 内訳: select_blocks 全体="
                  f"{p['b_detail'].get('select_blocks_total_ms', 0.0):.3f}ms"
                  f" - indexer={p['indexer_ms']:.3f}ms"
                  f" + (_gather_tile_attn 全体="
                  f"{p['b_detail'].get('gather_tile_total_ms', 0.0):.3f}ms"
                  f" - sdpa={p['sdpa_ms']:.3f}ms)")
        print(f"  {'部品和 (a+b+c+d)':28s}{p['parts_sum_ms']:10.3f}")
        print(f"  {'壁時計 (層 1 つ、実引数で再実行)':28s}{p['whole_layer_ms']:10.3f}")
        print(f"  部品和 - 壁時計 = {p['parts_sum_ms'] - p['whole_layer_ms']:+.3f} ms"
              f" ({p['gap_pct']:+.1f}%)")
        if abs(p["gap_pct"]) > 10:
            print("  ** 数 % を超えてずれている。host 側の分岐 (_positions、"
                  "gather 可否判定、decode の causal split_mask 組み立てなど)"
                  " が部品に入っていない可能性がある **")
        elif abs(p["gap_pct"]) > 3:
            print("  (数 % のずれ。host 側の未計上分岐ぶんとして許容範囲)")
        print(f"  12層換算 合計 ms/tok (indexer+mask+sdpa+other) = "
              f"{sum(p['ms_per_tok_x12'].values()):.4f}", flush=True)

    current = 0
    results = []
    for kv in kvs:
        target_offset = kv - args.S
        advance(current, target_offset)
        current = target_offset
        point = measure_point(current)
        results.append(point)
        print_point(point)
        # 実本体を進めて確定させる (ラップ無しの本物の forward)
        advance(current, current + args.S)
        current += args.S

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {
            "model": args.model, "ngram": args.ngram, "S": args.S,
            "kvs": kvs, "chunk": args.chunk, "reps": args.reps,
            "target_layer_idx": target_layer_idx, "n_attn": n_attn,
            "indexer_budget": target_idx.token_budget,
            "compress_ratio": target_idx.compress_ratio,
        },
        "points": results,
    }, ensure_ascii=False, indent=2))
    print(f"\n書いた: {out_path}", flush=True)

    # 計測ツールなので destructor 待ちでプロセスが Metal のメモリを握ったまま
    # 残る前例がある (他の tools/*.py と同じ理由) -- 即 _exit で落とす
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
