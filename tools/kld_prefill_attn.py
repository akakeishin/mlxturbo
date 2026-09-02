"""prefill attention 融合カーネル (段 P1、`MLXTURBO_PREFILL_ATTN=1`) が、
長文脈 prefill の出力分布をどれだけ動かすかを見る。

`mlxturbo/gather_attn.py` の `enable_prefill_attn(model)` / `disable_prefill_attn(model)`
がこの knob の本体 (`tools/decode_ab.py --knob prefill-attn` の A/B と同じ関数)。
kv >= 12288 あたりで `mlxturbo/kernels/prefill_attn.py` の融合カーネルが発火する
(短い文脈では `eligible()` が MIN_S 未満などで弾いて既存の gather 経路へ落ちる)。

既存の `bench/quant_eval.py compare` は継続長 128 程度の短い continuation しか
通さないので、kv >= 12k のカーネル発火域を一度も踏まない。ここではその域を
直接踏んで、カーネル on/off で出力分布がどれだけ動くかを見る。

## 手順

1. `disable_prefill_attn(model)` の状態 (= 本番の既定、カーネル off) で
   prefill をチャンク幅 `--chunk` (既定 2048) で回し、最終チャンクの末尾
   `--tail` 位置の logits を fp32 で確保する -> 分布 p
2. キャッシュを作り直し、`enable_prefill_attn(model)` (= `MLXTURBO_PREFILL_ATTN=1`
   と同じ状態) で同じ prefill をもう一度回し、同じ位置の logits を取る -> 分布 q
3. 位置ごとに KL(p‖q) を出し、平均・最大・argmax 一致率・top-5 の重なりを出す

KLD は `bench/quant_eval.py` の `evaluate()` にある `kld_mean` と同じ式
(参照側 top-K 近似、既定 K=256 で揃えている): 分布 p の対数確率トップK の位置
だけを取り、``sum(p_k * (logp_k - logq_k))`` を位置ごとに計算する。式と K を
揃えているので、ここで出る数字は他所の KLD 計測 (受け入れ幅 現行比 +0.0005) と
同じ物差しで見比べられる。ここでの独自の目安は絶対値ベースで、0.001 未満なら
「不変」、0.01 未満なら「小」と表示する (それ以上は「要確認」)。

prefill 幅 (S=2048) はカーネルの `eligible()` の MIN_S 判定を通るはずだが、
kv がまだ閾値未満だったり cache の型が合わなかったりすると `_gather_forward`
が既存経路へ黙って落ちる (`mlxturbo/kernels/prefill_attn.py` が理由を 1 度だけ
表示する)。**カーネルが 1 度も発火していないのに「分布は不変」と出すのは
無意味な結果なので**、`prefill_attn()` の実行回数を数えて `kernel_fired` に
出し、0 なら警告する (`tools/verify_prefill_attn.py` の counted 手法と同じ)。

モデルは 1 回だけ読む。GPU を使うので実行は biglock 経由:

    tools/biglock.sh .venv/bin/python tools/kld_prefill_attn.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram \\
        --ctxs 17000,25000
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

# bench/quant_eval.py の cmd_dump/--topk の既定と揃える (kld_mean を同じ式・
# 同じ K で比べられるようにするため)。
DEFAULT_TOPK = 256


def _run_prefill(model, cache, ids, chunk: int, tail: int, pending):
    """``ids`` ((1, n) の mx.array) をチャンク幅 ``chunk`` で ``cache`` に流し、
    最終チャンクの末尾 ``tail`` 位置の logits (fp32) を返す。

    中間チャンクは logits を捨てて次へ進む。キャッシュへの書き込みは MLX が
    遅延評価するので、チャンクごとに ``pending(cache)`` (indexer のバッファを
    含めて eval を強制する、`tools/prefill_anatomy.py` の作法) を一緒に eval
    してから ``mx.clear_cache()`` で解放する。
    """

    import mlx.core as mx

    n = ids.shape[1]
    tail_logits = None
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        part = ids[:, start:end]
        logits = model(part, cache=cache)
        if end == n:
            t = min(tail, part.shape[1])
            tail_logits = logits[:, -t:, :].astype(mx.float32)
            mx.eval(tail_logits, *pending(cache))
        else:
            mx.eval(logits, *pending(cache))
        mx.clear_cache()
    if tail_logits is None:
        raise RuntimeError("ids が空 (n=0)")
    return tail_logits


def _flat(a):
    """mx.array の入れ子リストを平らな int の列にする (top-k の添字が (1, k) 形で来ても壊れない)。"""
    import mlx.core as mx
    return [int(v) for v in mx.array(a).reshape(-1).tolist()]


def _kld_stats(logits_p, logits_q, topk: int) -> dict:
    """位置ごとに ``bench/quant_eval.py`` の kld_mean と同じ式で KL(p‖q) を出す。

    ``logits_p`` / ``logits_q`` は同じ位置集合の (positions, vocab) fp32。
    p (``logits_p``) の対数確率トップ ``topk`` の位置だけを使う近似で、
    ``sum(p_k * (logp_k - logq_k))`` を位置ごとに計算する
    (``bench/quant_eval.py`` の ``evaluate()`` の ``kld_pos`` と同一式)。
    """
    import mlx.core as mx
    # (B, tail, vocab) でも (tail, vocab) でも受ける: 末尾の vocab 軸だけ残して平らにする
    logits_p = mx.array(logits_p).reshape(-1, mx.array(logits_p).shape[-1])
    logits_q = mx.array(logits_q).reshape(-1, mx.array(logits_q).shape[-1])

    import mlx.core as mx
    import numpy as np

    logp_full = logits_p - mx.logsumexp(logits_p, axis=-1, keepdims=True)
    logq_full = logits_q - mx.logsumexp(logits_q, axis=-1, keepdims=True)

    k = min(topk, logp_full.shape[-1])
    idx = mx.argpartition(-logp_full, k - 1, axis=-1)[..., :k]
    top_logp = mx.take_along_axis(logp_full, idx, axis=-1)
    # argpartition は順序を保証しない。降順に並べ直す
    # (bench/quant_eval.py の cmd_dump と同じ手順)。
    order = mx.argsort(-top_logp, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    top_logp = mx.take_along_axis(top_logp, order, axis=-1)
    top_logq = mx.take_along_axis(logq_full, idx, axis=-1)

    argmax_p = mx.argmax(logits_p, axis=-1)
    argmax_q = mx.argmax(logits_q, axis=-1)
    top5_p = mx.argpartition(-logits_p, 4, axis=-1)[..., :5]
    top5_q = mx.argpartition(-logits_q, 4, axis=-1)[..., :5]
    mx.eval(top_logp, top_logq, argmax_p, argmax_q, top5_p, top5_q)

    logp_np = np.array(top_logp, dtype=np.float64)
    logq_np = np.array(top_logq, dtype=np.float64)
    p_np = np.exp(logp_np)
    kld_pos = (p_np * (logp_np - logq_np)).sum(axis=-1)

    argmax_agree = np.array(argmax_p) == np.array(argmax_q)
    t5p = np.array(top5_p)
    t5q = np.array(top5_q)
    overlap = np.array(
        [len(set(_flat(t5p[i])) & set(_flat(t5q[i]))) for i in range(t5p.shape[0])],
        dtype=np.float64,
    )

    return {
        "positions": int(kld_pos.shape[0]),
        "topk": k,
        "kld_mean": float(kld_pos.mean()),
        "kld_max": float(kld_pos.max()),
        "kld_per_position": [float(x) for x in kld_pos],
        "argmax_agree_rate": float(argmax_agree.mean()),
        "top5_overlap_mean": float((overlap / 5.0).mean()),
    }


def _verdict(kld_mean: float) -> str:
    if kld_mean < 0.001:
        return "不変"
    if kld_mean < 0.01:
        return "小"
    return "要確認"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="prefill attention 融合カーネル (MLXTURBO_PREFILL_ATTN) の"
        " on/off で、長文脈 prefill の出力分布がどれだけ動くかを測る"
    )
    ap.add_argument("--model", default="~/models/ddalcu-mlxlm")
    ap.add_argument("--ngram", default="~/models/ddalcu-ngram")
    ap.add_argument(
        "--ctxs",
        default="17000,25000",
        help="kv 長のカンマ区切り一覧 (カーネルは kv>=12288 あたりで発火)",
    )
    ap.add_argument(
        "--tail", type=int, default=64, help="最終チャンクの末尾何位置で分布を比べるか"
    )
    ap.add_argument("--chunk", type=int, default=2048, help="prefill チャンク幅")
    ap.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOPK,
        help="KLD 近似に使う p 側 top-K (bench/quant_eval.py の既定と揃えてある)",
    )
    ap.add_argument(
        "--question",
        default="上の文書の要点を5つに整理してください。",
        help="長文脈プロンプトの末尾に付ける質問 (tools/_bench_text.py の作法)",
    )
    ap.add_argument("--out", default="bench/results/kld-prefill-attn.json")
    args = ap.parse_args()

    ctxs = sorted({int(v) for v in args.ctxs.split(",") if v.strip() != ""})
    if not ctxs:
        print("--ctxs が空")
        return 1
    if args.tail <= 0:
        print("--tail は正の整数にすること")
        return 1

    if args.ngram:
        # n-gram をディスクに置いた構成。vendored arch は import 時に旗を読む。
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    import mlx.core as mx

    import mlxturbo  # noqa: F401  (arch_registry の meta_path フックを張る)
    from mlx_lm import load

    from mlxturbo.gather_attn import disable_prefill_attn, enable_prefill_attn
    from mlxturbo.kernels import prefill_attn as prefill_attn_kernel
    from mlxturbo import runner as mlxturbo_runner

    model_path = os.path.expanduser(args.model)
    ngram_path = os.path.expanduser(args.ngram) if args.ngram else None

    model, tok = load(model_path)
    # 読み込み直後に呼ぶ (常駐条件を本番と揃える。engine を直叩きなので
    # server.py の _load() を経由しないぶん、ここで自前で wire する)。
    if hasattr(mlxturbo_runner, "set_wired_limit_default"):
        mlxturbo_runner.set_wired_limit_default(log_prefix="[kld-prefill-attn]")
    if ngram_path:
        from mlxturbo.ngram_stream import install

        install(model, ngram_path)
    # 出荷経路 (build_runner) が起動時に通す融合・置き換えと同じものを当てる。
    # これで prefill_attn 以外の knob (MLXTURBO_GDN_METAL 等) が本番の既定値
    # のまま揃い、on/off の差分が prefill_attn だけに帰属する。
    mlxturbo_runner.enable_default_fusions(model, log_prefix="[kld-prefill-attn]")

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from _bench_text import long_prompts  # noqa: E402
    import prefill_anatomy as PA  # noqa: E402  (pending() を借りる)

    print(
        f"model={args.model} ngram={args.ngram} ctxs={ctxs} chunk={args.chunk}"
        f" tail={args.tail} topk={args.topk}",
        flush=True,
    )

    results: dict = {
        "kind": "kld-prefill-attn",
        "model": args.model,
        "ngram": args.ngram,
        "chunk": args.chunk,
        "tail": args.tail,
        "topk": args.topk,
        "ctxs": {},
    }

    ok_overall = True
    for ctx in ctxs:
        print(f"=== ctx={ctx} ===", flush=True)
        body = long_prompts(tok, ctx, [args.question])[0]
        ids = mx.array(
            tok.apply_chat_template(
                [{"role": "user", "content": body}], add_generation_prompt=True
            )
        )[None]
        print(f"  実 kv={ids.shape[1]}", flush=True)

        # (1) 分布 p: disable_prefill_attn (= 本番の既定、カーネル off)
        disable_prefill_attn(model)
        cache_p = model.make_cache()
        logits_p = _run_prefill(model, cache_p, ids, args.chunk, args.tail, PA.pending)
        del cache_p
        mx.clear_cache()

        # (2) 分布 q: enable_prefill_attn (= MLXTURBO_PREFILL_ATTN=1 と同じ状態)。
        # カーネルが実際に発火した回数を数える (0 なら比較そのものが無意味)。
        fired = [0]
        orig_kernel_fn = prefill_attn_kernel.prefill_attn

        def _counted(*a, **kw):
            fired[0] += 1
            return orig_kernel_fn(*a, **kw)

        prefill_attn_kernel.prefill_attn = _counted
        n_layers = enable_prefill_attn(model)
        try:
            cache_q = model.make_cache()
            logits_q = _run_prefill(model, cache_q, ids, args.chunk, args.tail, PA.pending)
        finally:
            prefill_attn_kernel.prefill_attn = orig_kernel_fn
            disable_prefill_attn(model)
        del cache_q
        mx.clear_cache()

        stats = _kld_stats(logits_p, logits_q, args.topk)
        stats["kv"] = int(ids.shape[1])
        stats["prefill_attn_layers"] = n_layers
        stats["kernel_fired"] = fired[0]
        stats["verdict"] = _verdict(stats["kld_mean"])
        if fired[0] == 0:
            print(
                "  ★カーネルが1度も発火していない"
                " (kv が閾値未満、または eligible() が別の理由で弾いている可能性。"
                " この ctx の比較は無意味なので verdict を信用しないこと) ★",
                flush=True,
            )
            ok_overall = False
        print(
            f"  positions={stats['positions']} kernel_fired={fired[0]}/"
            f"{n_layers} layers"
            f" kld_mean={stats['kld_mean']:.6f} kld_max={stats['kld_max']:.6f}"
            f" argmax_agree={stats['argmax_agree_rate']:.4f}"
            f" top5_overlap={stats['top5_overlap_mean']:.4f}"
            f" -> {stats['verdict']}",
            flush=True,
        )
        results["ctxs"][str(ctx)] = stats
        del logits_p, logits_q
        mx.clear_cache()

    results["ok"] = ok_overall
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"wrote {out_path}", flush=True)

    # 計測ツールなので destructor (スレッドプール等の後始末) に用は無い。
    # interpreter shutdown 待ちでプロセスが Metal のメモリを握ったまま残る
    # 前例があるので、結果を書き終えたら即 _exit で落とす (他の tools/*.py と同じ)。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
