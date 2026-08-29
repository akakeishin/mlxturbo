"""Phase Q: 量子化変種の品質・速度スイート (モデル非依存)。

多数の変種を繰り返し測る前提の設計:
  参照 logits を一度だけダンプして保存し、各変種はダンプとの比較だけを行う。
  kld_probe.py (27B 用) は参照モデルを毎回ロードするが、Flash-Next の参照
  (bf16 360GB) はメモリに載らないため、この分離が必須になる。

レーンは 2 本:
  相対レーン (今すぐ使える): 参照 = 走る中で最大の変種 (例 v-max-112)。
    変種間の優劣 (等バイト A/B など) はこれで決着する
  絶対レーン (後日): 参照 = bf16/FP8 をシャードストリーミングで 1 パスだけ
    流す teacher ダンプ。実装は初回起動後 (subcommand `teacher` は未実装)

KLD は参照 top-K (既定 256) + 裾の質量補正による近似。裾質量は通常 <0.1% で、
変種間比較には十分。モデルカードには近似である旨を書く。

使い方 (すべて GPU 実行):
  # 1. 正準継続を作る (参照モデルで 1 回)
  uv run python bench/quant_eval.py continuations --model <ref> \
      --out bench/results/qe-cont.json
  # 2. 参照 logits をダンプ (1 回)
  uv run python bench/quant_eval.py dump --model <ref> \
      --continuations bench/results/qe-cont.json --out bench/results/qe-ref.npz
  # 3. 各変種を比較 + 速度 (変種ごと)
  uv run python bench/quant_eval.py compare --model <variant> \
      --continuations bench/results/qe-cont.json \
      --ref-dump bench/results/qe-ref.npz --tag v0-95
  uv run python bench/quant_eval.py speed --model <variant> --tag v0-95
  # 4. 一覧表
  uv run python bench/quant_eval.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RESULTS_DIR = REPO_ROOT / "bench" / "results" / "quant-eval"

# プロンプトの設計と部品別の札は bench/eval_prompts.py 側にある
from bench.eval_prompts import PROMPTS, STRESS_KINDS  # noqa: E402

CALIB_PROMPTS: dict[str, str] = {k: v.text for k, v in PROMPTS.items()}


def _load(model_ref: str, ngram: str | None = None, rebit_spec: str | None = None):
    if ngram:
        # n-gram をディスクに置いた構成。vendored arch は import 時に旗を読む
        import os

        os.environ["FASTMLX_NGRAM_DISK"] = "1"
    from mlx_lm import load

    model, tok = load(model_ref)
    if ngram:
        from mlxturbo.ngram_stream import install

        install(model, ngram)
    if rebit_spec:
        from mlxturbo import rebit

        rebit.apply(model, rebit_spec)
    return model, tok


def _prompt_ids(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True
        )


def _teacher_logits(model, full_ids):
    import mlx.core as mx

    cache = model.make_cache()
    logits = model(mx.array(full_ids)[None], cache=cache)
    mx.eval(logits)
    return logits[0]


def cmd_continuations(args):
    import mlx.core as mx

    model, tokenizer = _load(args.model)
    eos = {tokenizer.eos_token_id}
    out = {"model": args.model, "gen_tokens": args.gen_tokens, "prompts": {}}
    for key, text in CALIB_PROMPTS.items():
        ids = _prompt_ids(tokenizer, text)
        cache = model.make_cache()
        logits = model(mx.array(ids)[None], cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
        cont = []
        while len(cont) < args.gen_tokens and cur not in eos:
            cont.append(cur)
            logits = model(mx.array([[cur]]), cache=cache)
            cur = int(mx.argmax(logits[0, -1], axis=-1))
        out["prompts"][key] = {"prompt_ids": ids, "continuation_ids": cont}
        print(f"[cont] {key}: {len(cont)} tokens")
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}")


def cmd_dump(args):
    import mlx.core as mx

    model, _ = _load(args.model)
    cont = json.loads(Path(args.continuations).read_text())
    arrays = {}
    meta = {"model": args.model, "topk": args.topk, "prompts": {}}
    for key, entry in cont["prompts"].items():
        full = entry["prompt_ids"] + entry["continuation_ids"]
        start = len(entry["prompt_ids"]) - 1
        logits = _teacher_logits(model, full)[start:-1].astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        idx = mx.argpartition(-logp, args.topk - 1, axis=-1)[:, : args.topk]
        top = mx.take_along_axis(logp, idx, axis=-1)
        # argpartition は順序を保証しないので降順に並べ直す。idx[:, 0] を
        # 参照の top-1 として扱う箇所があるため、ここを曖昧にしない
        order = mx.argsort(-top, axis=-1)
        idx = mx.take_along_axis(idx, order, axis=-1)
        top = mx.take_along_axis(top, order, axis=-1)
        # 正準トークンに参照が置く対数確率。Δp を「同じトークンに対する
        # 両者の確率差」として測るために要る。参照の top-1 で代用すると、
        # 逐次デコードと一括 forward の argmax がまれに食い違う分だけ
        # 自己比較でも 0 にならない (bf16 同点の縮約順の違い、STATUS 参照)
        tgt_ids = mx.array(full[start + 1 :])[:, None]
        tgt_logp = mx.take_along_axis(logp, tgt_ids, axis=-1)[:, 0]
        mx.eval(idx, top, tgt_logp)
        arrays[f"{key}.tgt_logp"] = np.array(tgt_logp, dtype=np.float32)
        idx_np = np.array(idx, dtype=np.int32)
        top_np = np.array(top, dtype=np.float32)
        tail = np.log1p(-np.minimum(np.exp(top_np).sum(axis=-1), 1 - 1e-9))
        arrays[f"{key}.idx"] = idx_np
        arrays[f"{key}.logp"] = top_np
        arrays[f"{key}.tail"] = tail.astype(np.float32)
        meta["prompts"][key] = {"positions": int(idx_np.shape[0])}
        print(f"[dump] {key}: {idx_np.shape[0]} positions")
        del logits, logp, idx, top
        mx.clear_cache()
    np.savez_compressed(args.out, **arrays)
    Path(str(args.out) + ".meta.json").write_text(json.dumps(meta))
    print(f"wrote {args.out}")


def evaluate(model, cont: dict, ref, quiet: bool = False) -> dict:
    """読み込み済みモデルを参照ダンプと突き合わせ、プロンプト別の指標を返す。

    cmd_compare と cmd_sweep で共有する。sweep はモデルを 1 回しか読まずに
    ビット構成を積み上げていくので、ここがモデルを読まないことが要件。
    """

    import mlx.core as mx

    per_prompt = {}
    for key, entry in cont["prompts"].items():
        full = entry["prompt_ids"] + entry["continuation_ids"]
        start = len(entry["prompt_ids"]) - 1
        logits = _teacher_logits(model, full)[start:-1].astype(mx.float32)
        logq_all = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        idx = mx.array(ref[f"{key}.idx"])
        logq = mx.take_along_axis(logq_all, idx, axis=-1)
        argmax_q = mx.argmax(logits, axis=-1)
        mx.eval(logq, argmax_q)
        logq_np = np.array(logq, dtype=np.float64)
        logp_np = ref[f"{key}.logp"].astype(np.float64)
        p = np.exp(logp_np)
        # top-K 近似 KLD: 裾は「参照裾質量が変種でも同じ集合に落ちる」仮定の
        # 下限側近似。裾質量 <0.1% なら変種間比較への影響は無視できる
        kld_pos = (p * (logp_np - logq_np)).sum(axis=-1)
        ref_top1 = ref[f"{key}.idx"][:, 0]
        agree = (np.array(argmax_q) == ref_top1).mean()
        # 変種自身の継続 PPL (正準トークン列に対する)
        tgt = np.array(full[start + 1 :], dtype=np.int64)
        tgt_logq = np.array(
            mx.take_along_axis(logq_all, mx.array(tgt)[:, None], axis=-1)[:, 0],
            dtype=np.float64,
        )
        # Δp: 正準トークン (= 参照の top-1) に参照と変種が置く確率の差。
        # llama.cpp の perplexity ツールが出す指標に合わせている。KLD が
        # 分布全体の距離なのに対し、こちらは「実際に選ばれる語がどれだけ
        # 押し下げられたか」を見る
        dp = np.exp(tgt_logq) - np.exp(ref[f"{key}.tgt_logp"].astype(np.float64))
        per_prompt[key] = {
            "kld_mean": float(kld_pos.mean()),
            "kld_median": float(np.median(kld_pos)),
            "kld_p95": float(np.quantile(kld_pos, 0.95)),
            "kld_p99": float(np.quantile(kld_pos, 0.99)),
            "kld_max": float(kld_pos.max()),
            "top1_agree": float(agree),
            "delta_p_mean": float(dp.mean()),
            "delta_p_rms": float(np.sqrt((dp**2).mean())),
            "ppl": float(np.exp(-tgt_logq.mean())),
            "positions": int(kld_pos.shape[0]),
            "stress": list(PROMPTS[key].stress) if key in PROMPTS else [],
        }
        if not quiet:
            print(f"[cmp] {key}: kld={per_prompt[key]['kld_mean']:.5f} "
                  f"agree={agree:.3f}", flush=True)
        del logits, logq_all
        mx.clear_cache()
    return per_prompt


def summarize(per_prompt: dict) -> dict:
    """プロンプト別の指標 → 全体と札別の集計。"""

    klds = [v["kld_mean"] for v in per_prompt.values()]
    agrees = [v["top1_agree"] for v in per_prompt.values()]
    # 部品別の集計。平均を 1 つに潰すと「n-gram と experts のどちらに
    # ビットを盛るか」が読めなくなるので、札ごとに分けて持つ
    by_stress = {}
    for kind in STRESS_KINDS:
        ks = [k for k, v in per_prompt.items() if kind in v["stress"]]
        if not ks:
            continue
        by_stress[kind] = {
            "prompts": len(ks),
            "kld_mean": float(np.mean([per_prompt[k]["kld_mean"] for k in ks])),
            "top1_agree": float(np.mean([per_prompt[k]["top1_agree"] for k in ks])),
            "delta_p_rms": float(np.mean([per_prompt[k]["delta_p_rms"] for k in ks])),
        }
    return {
        "kld_mean": float(np.mean(klds)),
        "kld_worst_prompt": max(per_prompt, key=lambda k: per_prompt[k]["kld_mean"]),
        "top1_agree_mean": float(np.mean(agrees)),
        "delta_p_rms_mean": float(
            np.mean([v["delta_p_rms"] for v in per_prompt.values()])
        ),
        "by_stress": by_stress,
    }


def cmd_sweep(args):
    """1 回のロードでビット構成を積み上げながら、品質と速度を同時に測る。

    独立に測るならモデルを構成の数だけ読み直すことになり、92GB x N の読み込みで
    半日が溶ける。積み上げなら 1 回で済む。段を 1 つずつ足すので、増分はその段の
    寄与として読める (効果がおおむね加法的である限り)。

    最終行が、その全部を適用したレシピの見込み値になる。

        uv run python bench/quant_eval.py sweep --model <m> --ngram <s> \
            --continuations ... --ref-dump ... --steps gdn=4 hc=4 head=4
    """

    import time

    import mlx.core as mx

    model, tok = _load(args.model, getattr(args, "ngram", None))
    cont = json.loads(Path(args.continuations).read_text())
    if args.prompts:
        keep = set(args.prompts.split(","))
        cont = {**cont, "prompts": {k: v for k, v in cont["prompts"].items()
                                    if k in keep}}
        print(f"プロンプトを {len(cont['prompts'])} 本に絞った")
    ref = np.load(args.ref_dump)
    ids = _prompt_ids(tok, "分散システムについて詳しく説明してください。")

    def speed() -> float:
        cache = model.make_cache()
        logits = model(mx.array(ids)[None], cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
        for _ in range(5):
            logits = model(mx.array([[cur]]), cache=cache)
            cur = int(mx.argmax(logits[0, -1], axis=-1))
        best = None
        for _ in range(2):
            t0 = time.perf_counter()
            for _ in range(args.speed_tokens):
                logits = model(mx.array([[cur]]), cache=cache)
                cur = int(mx.argmax(logits[0, -1], axis=-1))
            dt = (time.perf_counter() - t0) / args.speed_tokens * 1000
            best = dt if best is None else min(best, dt)
        return best

    from mlxturbo import rebit

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"sweep-{args.tag}.json"

    rows = []
    applied: list[str] = []

    def save():
        # 段ごとに書く。98GB のモデルを 2 セッションが同時に読むとメモリ圧で
        # 落ちるので、最後にまとめて書くと 30 分ぶんが丸ごと消える (3 回やった)
        out.write_text(json.dumps({
            "kind": "rebit-sweep",
            "tag": args.tag,
            "model": args.model,
            "ref_dump": str(args.ref_dump),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "note": "rebit は二重量子化なので、実際に焼いた場合より悪く出る",
            "complete": len(rows) == len(args.steps) + 1,
            "rows": rows,
        }, indent=1))

    for label in ["(そのまま)"] + list(args.steps):
        if label != "(そのまま)":
            rebit.apply(model, label)
            applied.append(label)
        ms = speed()
        per_prompt = evaluate(model, cont, ref, quiet=True)
        s = summarize(per_prompt)
        rows.append({
            "step": label,
            "applied": list(applied),
            "ms_per_token": ms,
            **{k: v for k, v in s.items() if k != "by_stress"},
            "by_stress": s["by_stress"],
        })
        print(f"  {label:14s} {ms:6.2f} ms/token ({1000 / ms:5.2f} tok/s)  "
              f"KLD {s['kld_mean']:.5f}  top1 {s['top1_agree_mean']:.4f}  "
              f"最悪 {s['kld_worst_prompt']}", flush=True)
        save()

    base = rows[0]
    print("\n  段ごとの差分 (直前の段からの増分)")
    print(f"  {'段':14s} {'ms':>8s} {'KLD 増':>10s} {'ms/KLD':>10s}")
    for prev, cur in zip(rows, rows[1:]):
        d_ms = prev["ms_per_token"] - cur["ms_per_token"]
        d_kld = cur["kld_mean"] - prev["kld_mean"]
        rate = d_ms / d_kld if d_kld > 1e-9 else float("inf")
        print(f"  {cur['step']:14s} {d_ms:+8.2f} {d_kld:+10.5f} {rate:10.0f}")
    print(f"\n  合計 {base['ms_per_token'] - rows[-1]['ms_per_token']:+.2f} ms/token, "
          f"KLD {base['kld_mean']:.5f} -> {rows[-1]['kld_mean']:.5f}")

    save()
    print(f"wrote {out}")


def cmd_compare(args):
    model, _ = _load(args.model, getattr(args, "ngram", None),
                     getattr(args, "rebit", None))
    if getattr(args, "disable_ple", False):
        # n-gram/PLE を丸ごと切る。埋め込みがゼロなら PLE の出力もゼロになる
        # ので、層を外すのと等価。「n-gram が無いことの代償」を測るための経路
        n = 0
        for layer in model.model.layers:
            if getattr(layer, "ple", None) is not None:
                layer.ple = None
                n += 1
        print(f"PLE を無効化した ({n} 層)")
    cont = json.loads(Path(args.continuations).read_text())
    ref = np.load(args.ref_dump)
    per_prompt = evaluate(model, cont, ref)
    result = {
        "kind": "compare",
        "tag": args.tag,
        "model": args.model,
        "rebit": getattr(args, "rebit", None),
        "ref_dump": str(args.ref_dump),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **summarize(per_prompt),
        "per_prompt": per_prompt,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"compare-{args.tag}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}: kld_mean={result['kld_mean']:.5f} agree={result['top1_agree_mean']:.3f}")


def cmd_speed(args):
    import mlx.core as mx

    t0 = time.perf_counter()
    model, tokenizer = _load(args.model)
    load_s = time.perf_counter() - t0
    rows = {}
    for key in args.prompts.split(","):
        text = CALIB_PROMPTS[key]
        ids = _prompt_ids(tokenizer, text)
        cache = model.make_cache()
        t0 = time.perf_counter()
        logits = model(mx.array(ids)[None], cache=cache)
        cur = mx.argmax(logits[0, -1], axis=-1).reshape(1)
        mx.eval(cur)
        prefill_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        n = 0
        while n < args.gen_tokens:
            logits = model(cur[None], cache=cache)
            cur = mx.argmax(logits[0, -1], axis=-1).reshape(1)
            mx.eval(cur)
            n += 1
        decode_s = time.perf_counter() - t0
        rows[key] = {
            "prefill_tps": len(ids) / prefill_s,
            "decode_tps": n / decode_s,
        }
        print(f"[speed] {key}: {rows[key]['decode_tps']:.1f} tok/s")
    result = {
        "kind": "speed",
        "tag": args.tag,
        "model": args.model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "load_s": load_s,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "decode_tps_mean": float(np.mean([r["decode_tps"] for r in rows.values()])),
        "per_prompt": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"speed-{args.tag}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}: decode_mean={result['decode_tps_mean']:.1f} tok/s peak={result['peak_memory_gb']:.1f}GB")


def cmd_report(args):
    del args
    rows = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        tag = d["tag"]
        rows.setdefault(tag, {})
        if d["kind"] == "compare":
            rows[tag].update(
                kld=d["kld_mean"],
                agree=d["top1_agree_mean"],
                worst=d["kld_worst_prompt"],
            )
        else:
            rows[tag].update(
                tps=d["decode_tps_mean"], mem=d["peak_memory_gb"], load=d["load_s"]
            )
    if not rows:
        print("結果なし")
        return
    print(f"{'tag':12s} {'KLD':>9s} {'top1一致':>8s} {'decode':>8s} {'peak GB':>8s}  worst prompt")
    for tag, r in sorted(rows.items()):
        print(
            f"{tag:12s} {r.get('kld', float('nan')):9.5f} {r.get('agree', float('nan')):8.3f} "
            f"{r.get('tps', float('nan')):8.1f} {r.get('mem', float('nan')):8.1f}  {r.get('worst', '-')}"
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("continuations")
    p.add_argument("--model", required=True)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_continuations)

    p = sub.add_parser("dump")
    p.add_argument("--model", required=True)
    p.add_argument("--continuations", required=True)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_dump)

    p = sub.add_parser("compare")
    p.add_argument("--model", required=True)
    p.add_argument("--continuations", required=True)
    p.add_argument("--ref-dump", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--ngram", help="n-gram サイドカーのディレクトリ")
    p.add_argument("--disable-ple", action="store_true",
                   help="n-gram/PLE を切って測る (無しの代償を見る)")
    p.add_argument("--rebit", help="読み込み後にビットを打ち直す "
                   "(例 gdn=4,hc=4)。焼かずにビット配分を試すため")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("sweep", help="1 回のロードでビット構成を積み上げて測る")
    p.add_argument("--model", required=True)
    p.add_argument("--continuations", required=True)
    p.add_argument("--ref-dump", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--ngram", help="n-gram サイドカーのディレクトリ")
    p.add_argument("--steps", nargs="+", required=True,
                   help="積み上げる rebit 指定 (例 gdn=4 hc=4 head=4)")
    p.add_argument("--prompts", default=None,
                   help="評価に使うプロンプトを絞る (カンマ区切り)。段あたりの "
                        "時間が縮むので、落とされる前に終わる")
    p.add_argument("--speed-tokens", type=int, default=40)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("speed")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--prompts", default="ja-explain,code-py,repeat-edit")
    p.set_defaults(fn=cmd_speed)

    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
