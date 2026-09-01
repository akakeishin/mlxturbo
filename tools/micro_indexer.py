"""QSAIndexer.__call__ の内訳を、実キャッシュ・実入力のまま段ごとに計時する。

## 背景

`docs/research/KERNEL-PROGRAM.md` の段 2。`tools/decode_anatomy.py` の解剖で、
17k 文脈の decode 1 ラウンド (幅 2) のうち `QSAIndexer.__call__`
(`mlxturbo/_vendor/qwen4_exp.py`) が 3.80ms/フォワードかかっていることが
分かっている。帯域の下限は 0.13ms なので実測はその 3% しか出ていない。
このツールは「中身のどこが主体か」を段ごとの計時で割るためのもので、
採否や次の一手の判断はしない (数字を出すだけ)。

## なぜ decode_anatomy.py だけでは足りないか

decode_anatomy.py は `QSAIndexer.__call__` を丸ごと 1 個のブロックとして
計時する (`idx` の行)。ここではその内側をさらに割る。作法は
decode_anatomy.py と揃えてある --- 実キャッシュを退避・復元しながら実入力で
呼び直す、`med_ms`、`_bench_text.long_prompts` でプロンプトを作る、
`enable_default_fusions` を通す。

## 段の切り方

`QSAIndexer.__call__` の中身を上から読んだ順に、以下の 7 段 + 全体基準に割る
(コード上の行と対応)。境界に迷う「糊」の計算 (block_starts / cos・sin の
算出など) がいくつかあり、それらは直後の段に含めてある。どこに何を含めたかは
各 `bench_*` 関数の docstring に書く --- 曖昧なまま「純粋な段」を装わない。

    (a) index_qk_proj と reshape (q, raw_k を作る)
    (b) cache.update(raw_k)                       状態変化あり (退避・復元が要る)
    (c) pooled の作り直し (reshape + mean + k_layernorm)
    (d) pooled への rope (_rope_partial)
    (e) q 側の layernorm+rope と einsum + relu-sum
    (f) visible マスクの構成 + scores へのマスク適用 + argpartition top-k
    (g) keep マスクの構成 (put_along_axis / repeat / 末尾 concat)
    (h) __call__ 全体 (基準、状態変化あり)

## 計時の作法

- (a) は `x` (実キャッシュの手前で捕まえた実入力、フック元の `mx.eval` で
  実体化済み) だけを使う純関数なので、そのまま繰り返し計時できる。
- (b) は `cache.update` が `_IndexerCache` を書き換えるので、計時の 1 回ごとに
  `keys` を退避・復元する (decode_anatomy.py と同じ理屈 --- MLX 配列は不変
  なので、参照を差し戻せば完全に戻る)。
- (c)-(g) はどれも純関数だが、後段は前段の出力を要る。前段の計算時間を
  計時に混入させないために、`compute_reference()` で (b)-(g) を 1 回だけ通しで
  実行して `mx.eval` し、その具体値 (もう計算グラフを遡らない) を各段の
  固定入力として使う。
- (h) は退避・復元込みでオリジナルの `QSAIndexer.__call__` をそのまま呼ぶ。
  他の段の計時と順序依存はない (毎回退避・復元するため)。

## 部品和と全体の比較

(a)-(g) の和と (h) を並べて出す。ずれたらそう書く
(decode_anatomy.py と同じ「部品和 ≈ 壁時計」の作法。ここでの残差は
「見えていない計算がある」と「単体計測が本番の重なりを再現できていない」の
どちらかで、この道具では区別できない)。

## 制約

- 実行は 1 GPU プロセスを専有する。他の計測と同時に走らせない
  (`CLAUDE.md` の計測作法)。
- 判定はしない。`docs/research/KERNEL-PROGRAM.md` 段 2 の反転条件
  (「pooled が主体なら B4」「argpartition が主体で帯域律速なら段 3 へ」) は
  出力を読んだ人間が決める。

    tools/biglock.sh .venv/bin/python tools/micro_indexer.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep --ctx 17000
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def med_ms(fn, reps=7):
    import mlx.core as mx

    fn()
    mx.synchronize() if hasattr(mx, "synchronize") else None
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def snapshot(caches):
    """モデル全体のキャッシュの参照と offset を控える (decode_anatomy.py と同じ)。

    実フォワードを 1 回流してフックで実入力を捕まえるときにだけ使う。
    段別の計時 (b) は `_IndexerCache` だけを触るので `snap_idx`/`restore_idx`
    の方を使い、こちらは呼ばない。
    """
    st = []
    for c in caches:
        if hasattr(c, "keys"):
            st.append(("a", c.keys, c.values, c.offset,
                       c.indexer.keys, c.indexer.offset))
        else:
            st.append(("l", [c[i] for i in range(4)]))
    return st


def restore(caches, st):
    for c, rec in zip(caches, st):
        if rec[0] == "a":
            _, c.keys, c.values, c.offset, ik, io = rec
            c.indexer.keys = ik
            c.indexer.offset = io
        else:
            for i, v in enumerate(rec[1]):
                c[i] = v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--ctx", type=int, default=17000)
    ap.add_argument("--width", type=int, default=2, help="検証フォワードの幅")
    args = ap.parse_args()

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401
    import mlx_lm.models.qwen4_exp as Q  # mlxturbo が _vendor 版へ差し替え済み
    from mlxturbo.runner import enable_default_fusions

    model, tok = load(os.path.expanduser(args.model))
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))
    enable_default_fusions(model, log_prefix="[micro_indexer]")

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from _bench_text import long_prompts

    body = long_prompts(tok, args.ctx, ["上の文書の要点を 5 つに整理してください。"])[0]
    ids = mx.array(tok.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True))[None]
    n = ids.shape[1]
    print(f"ctx={n} width={args.width}")

    cache = model.make_cache()
    step = 2048
    for i in range(0, n, step):
        mx.eval(model(ids[:, i : i + step], cache=cache))
        mx.clear_cache()
    print("prefill 済み", flush=True)

    W = args.width
    chunk = mx.array([[int(ids[0, -1].item())] * W])

    # ---- 実入力をフックで捕まえる (decode_anatomy.py の idx フックと同じ形) ----
    grabbed: list[tuple] = []
    orig_call = Q.QSAIndexer.__call__

    def hook(self, x, rope, cache_, offset, positions=None):
        grabbed.append((self, x, rope, cache_, offset, positions))
        return orig_call(self, x, rope, cache_, offset, positions)

    Q.QSAIndexer.__call__ = hook
    pre = snapshot(cache)
    mx.eval(model(chunk, cache=cache))
    Q.QSAIndexer.__call__ = orig_call
    restore(cache, pre)
    print(f"捕まえた: idx={len(grabbed)} 層", flush=True)

    if not grabbed:
        print("QSAIndexer が 1 回も発火しなかった (kv_len <= token_budget?)。"
              " --ctx を上げるか indexer_budget を確認すること")
        return 1

    token_budget = grabbed[0][0].token_budget
    kv_len0 = grabbed[0][3].keys.shape[1] if grabbed[0][3].keys is not None else 0
    print(f"token_budget={token_budget} 捕まえた時点の kv_len(層0)={kv_len0}",
          flush=True)

    # ---- _IndexerCache だけの退避・復元 ----
    # (b) の計時と compute_reference の cache.update だけがここを使う。
    # `keys` の setter が offset も入力の shape[1] へ戻すので、offset を別に
    # 控える必要は無い (decode_anatomy.py の全体 snapshot と違い、ここは
    # indexer キャッシュ以外に触らない分だけ軽い)。
    def snap_idx():
        return [t[3].keys for t in grabbed]

    def restore_idx(snap):
        for t, k in zip(grabbed, snap):
            t[3].keys = k

    # ---- (h) 全体 (基準)。他の段と状態の奪い合いが無いよう独立に計時する ----
    def run_h():
        s = snap_idx()
        outs = [orig_call(*t) for t in grabbed]
        mx.eval(outs)
        restore_idx(s)
        return outs

    h_ms = med_ms(run_h)

    # ---- (a) index_qk_proj と reshape (純関数、x だけに依存) ----
    def run_a():
        outs = []
        for self_, x, rope, cache_, offset, positions in grabbed:
            B, S, _ = x.shape
            qk = self_.index_qk_proj(x)
            split = self_.n_heads * self_.head_dim
            q = qk[..., :split].reshape(B, S, self_.n_heads, self_.head_dim)
            raw_k = qk[..., split:].reshape(B, S, self_.head_dim)
            outs.append((q, raw_k))
        mx.eval(outs)
        return outs

    a_ms = med_ms(run_a)
    a_out = run_a()  # (b) 以降の固定入力として使う実体化済みの q, raw_k

    # ---- (b) cache.update(raw_k)。唯一状態を変える段 ----
    def run_b():
        s = snap_idx()
        outs = [cache_.update(raw_k)
                for (self_, x, rope, cache_, offset, positions), (q, raw_k)
                in zip(grabbed, a_out)]
        mx.eval(outs)
        restore_idx(s)
        return outs

    b_ms = med_ms(run_b)

    # ---- (b)-(g) を 1 回だけ通しで実行し、各段の固定入力を確定させる ----
    # cache.update はここでも状態を変えるので、退避してから復元する。
    # 以降 (c)-(g) の計時はここで作った mx.eval 済みの値だけを使うので、
    # cache には二度と触れない。
    def compute_reference():
        s = snap_idx()
        refs = []
        for (self_, x, rope, cache_, offset, positions), (q, raw_k) in zip(
            grabbed, a_out
        ):
            B, S, _ = x.shape
            raw_k_full = cache_.update(raw_k)
            kv_len = raw_k_full.shape[1]
            n_blocks = kv_len // self_.compress_ratio

            pooled_mean = (
                raw_k_full[:, : n_blocks * self_.compress_ratio]
                .reshape(B, n_blocks, self_.compress_ratio, self_.head_dim)
                .astype(mx.float32)
                .mean(axis=2)
                .astype(raw_k_full.dtype)
            )
            pooled = self_.k_layernorm(pooled_mean)

            block_starts = mx.arange(n_blocks) * self_.compress_ratio
            cos_k, sin_k = rope(block_starts[None, :])
            pooled_roped = Q._rope_partial(pooled, cos_k, sin_k)

            q_col = mx.arange(offset, offset + S)
            cos_q, sin_q = rope(q_col[None, :] if positions is None else positions)
            q_normed = self_.q_layernorm(q)
            q_roped = Q._rope_partial(
                q_normed, cos_q[:, :, None, :], sin_q[:, :, None, :]
            )

            scores_raw = mx.einsum(
                "bshd,bnd->bsnh",
                q_roped.astype(mx.float32),
                pooled_roped.astype(mx.float32),
            )
            scores_relu = mx.maximum(scores_raw, 0).sum(axis=-1) / math.sqrt(
                self_.head_dim
            )

            block_end = block_starts + self_.compress_ratio - 1
            visible = block_end[None, None, :] <= q_col[None, :, None]
            scores_masked = mx.where(visible, scores_relu, -mx.inf)

            k_top = min(self_.block_topk, n_blocks)
            top = mx.argpartition(-scores_masked, k_top - 1, axis=-1)[..., :k_top]

            refs.append(dict(
                self_=self_, x=x, rope=rope, offset=offset, positions=positions,
                q=q, raw_k_full=raw_k_full, kv_len=kv_len, n_blocks=n_blocks,
                pooled=pooled, block_starts=block_starts,
                pooled_roped=pooled_roped, q_col=q_col,
                scores_relu=scores_relu, visible=visible, k_top=k_top, top=top,
            ))
        flat = [v for r in refs for v in r.values() if isinstance(v, mx.array)]
        mx.eval(flat)
        restore_idx(s)
        return refs

    refs = compute_reference()

    # ---- (c) pooled の作り直し (reshape + mean + k_layernorm) ----
    def run_c():
        outs = []
        for r in refs:
            self_, raw_k_full, n_blocks = r["self_"], r["raw_k_full"], r["n_blocks"]
            B = r["x"].shape[0]
            pooled_mean = (
                raw_k_full[:, : n_blocks * self_.compress_ratio]
                .reshape(B, n_blocks, self_.compress_ratio, self_.head_dim)
                .astype(mx.float32)
                .mean(axis=2)
                .astype(raw_k_full.dtype)
            )
            outs.append(self_.k_layernorm(pooled_mean))
        mx.eval(outs)
        return outs

    c_ms = med_ms(run_c)

    # ---- (d) pooled への rope。block_starts から cos/sin を引く糊も含む ----
    def run_d():
        outs = []
        for r in refs:
            cos_k, sin_k = r["rope"](r["block_starts"][None, :])
            outs.append(Q._rope_partial(r["pooled"], cos_k, sin_k))
        mx.eval(outs)
        return outs

    d_ms = med_ms(run_d)

    # ---- (e) q 側の layernorm+rope と einsum + relu-sum ----
    # pooled 側の (c)(d) に対応する q 側の下ごしらえを、直後の einsum と
    # まとめて 1 段にしてある (task の 7 項目にこの下ごしらえの専用枠が
    # 無いため。単独では計らない糊)。
    def run_e():
        outs = []
        for r in refs:
            self_ = r["self_"]
            cos_q, sin_q = r["rope"](
                r["q_col"][None, :] if r["positions"] is None else r["positions"]
            )
            q_normed = self_.q_layernorm(r["q"])
            q_roped = Q._rope_partial(
                q_normed, cos_q[:, :, None, :], sin_q[:, :, None, :]
            )
            scores = mx.einsum(
                "bshd,bnd->bsnh",
                q_roped.astype(mx.float32),
                r["pooled_roped"].astype(mx.float32),
            )
            scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(self_.head_dim)
            outs.append(scores)
        mx.eval(outs)
        return outs

    e_ms = med_ms(run_e)

    # ---- (f) visible マスク構成 + scores マスク適用 + argpartition top-k ----
    # visible (block_end <= q_col) は (g) の top クランプでも再利用されるが、
    # 構成そのものはここでしか計らない (argpartition の直前の依存として)。
    def run_f():
        outs = []
        for r in refs:
            self_ = r["self_"]
            block_end = r["block_starts"] + self_.compress_ratio - 1
            visible = block_end[None, None, :] <= r["q_col"][None, :, None]
            scores_masked = mx.where(visible, r["scores_relu"], -mx.inf)
            k_top = min(self_.block_topk, r["n_blocks"])
            top = mx.argpartition(-scores_masked, k_top - 1, axis=-1)[..., :k_top]
            outs.append(top)
        mx.eval(outs)
        return outs

    f_ms = med_ms(run_f)

    # ---- (g) keep マスクの構成 (put_along_axis / repeat / 末尾 concat) ----
    # offset < compress_ratio-1 の救済分岐は、本番構成 (budget 2048、prefill
    # チャンク 2048) では 17k decode 中は踏まないので実装だけ揃えて計らない
    # (decode_anatomy.py 元コードのコメントと同じ理由)。
    def run_g():
        outs = []
        for r in refs:
            self_ = r["self_"]
            B, S = r["x"].shape[0], r["x"].shape[1]
            n_blocks, kv_len = r["n_blocks"], r["kv_len"]
            keep_block = mx.zeros((B, S, n_blocks + 1), dtype=mx.bool_)
            top = mx.where(
                mx.take_along_axis(r["visible"], r["top"], axis=-1),
                r["top"], n_blocks,
            )
            keep_block = mx.put_along_axis(
                keep_block, top, mx.array(True), axis=-1
            )[..., :n_blocks]
            keep = mx.repeat(keep_block, self_.compress_ratio, axis=-1)
            tail = kv_len - n_blocks * self_.compress_ratio
            if tail:
                tail_col = n_blocks * self_.compress_ratio + mx.arange(tail)
                keep = mx.concatenate(
                    [
                        keep,
                        mx.broadcast_to(
                            tail_col[None, None, :] <= r["q_col"][None, :, None],
                            (B, S, tail),
                        ),
                    ],
                    axis=-1,
                )
            outs.append(keep[:, None])
        mx.eval(outs)
        return outs

    g_ms = med_ms(run_g)

    # ---- まとめ ----
    order = [
        ("a", "index_qk_proj + reshape", a_ms),
        ("b", "cache.update(raw_k)", b_ms),
        ("c", "pooled 再構築 (mean + k_layernorm)", c_ms),
        ("d", "pooled への rope", d_ms),
        ("e", "q layernorm+rope と einsum/relu-sum", e_ms),
        ("f", "visible + scores マスク + argpartition", f_ms),
        ("g", "keep マスク構成 (put_along_axis/repeat/concat)", g_ms),
    ]
    parts = sum(v for _, _, v in order)

    print()
    print(f"  層数 = {len(grabbed)} (幅 W={W} の 1 検証ラウンドぶん)")
    print()
    for tag, label, val in order:
        print(f"  ({tag}) {label:42s} {val:7.3f} ms  ({val / h_ms * 100:5.1f}%)")
    print(f"  {'部品和 (a)-(g)':46s} {parts:7.3f} ms  ({parts / h_ms * 100:5.1f}%)")
    print(f"  {'(h) __call__ 全体 (基準)':46s} {h_ms:7.3f} ms  (100.0%)")
    gap = (parts - h_ms) / h_ms * 100
    print(f"\n  部品和 - 全体 = {parts - h_ms:+.3f} ms ({gap:+.1f}%)")
    if abs(gap) > 10:
        print("  ** 数 % を超えてずれている。見えていない計算があるか、"
              "単体計測が本番の重なりを再現できていない可能性がある **")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
