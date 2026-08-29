"""Flash-Next の実配置で、バッチ化の判断材料を 2 つ観測する。

コードは一切変えない。読み取りだけ。

観測 1: 活性エキスパートの和集合
    デコード 1 ステップ・1 層あたり、独立した B 本のシーケンスが選ぶ
    エキスパートの和集合の大きさ。バッチ生成の機構は要らない。B 本を
    別々に走らせ、同じステップ位置の router 選択を記録して後から和集合を
    取る。プロンプトは互いに無関係な話題にする (似た内容だとルータの選択が
    揃い、和集合が不当に小さく出る)。

観測 2: プレフィルとデコードの比
    mlxturbo/runner.py の build_runner を通して、コーディングエージェント
    らしい形状のリクエストを流す。初回 (prompt cache 無し) と 2 ターン目
    以降 (FallbackSession による差分 prefill) の両方を測る。

    tools/biglock.sh .venv/bin/python tools/observe_flashnext_batch.py \
        --model ~/models/qwen38fn-mlx-v-fast6 --ngram ~/models/qwen38fn-ngram-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "tools"))
from bench_batch_moe import SEED_TEXTS  # noqa: E402


def iqr(xs):
    xs = sorted(xs)
    if len(xs) < 4:
        return max(xs) - min(xs)
    q = statistics.quantiles(xs, n=4)
    return q[2] - q[0]


# ---------------------------------------------------------------- 観測 1


def build_union_prompts_natural(n):
    """bench/results/qe-cont.json の実プロンプトをそのまま n 本使う。

    種文を 40 回繰り返して長さを揃えるやり方は、同じ文が並ぶ退化した文脈を
    作る。ルータがそれで狭いところに寄っていないかを見るための対照。長さは
    揃わないが、和集合はステップ位置ごとに取るので揃っている必要はない。
    """
    cont = json.loads((REPO_ROOT / "bench/results/qe-cont.json").read_text())
    keys = sorted(cont["prompts"])
    if len(keys) < n:
        raise SystemExit(f"qe-cont のプロンプトが足りない: {len(keys)} < {n}")
    # 隣り合うキーは似た系統 (copy-*, mtp-* 等) なので間を空けて採る
    step = len(keys) // n
    picked = [keys[i * step] for i in range(n)]
    return picked, [cont["prompts"][k]["prompt_ids"] for k in picked]


def build_union_prompts(tokenizer, n, prompt_tokens):
    """話題の違う n 本のプロンプトを、長さを揃えて作る。"""
    out = []
    for i in range(n):
        if i >= len(SEED_TEXTS):
            raise SystemExit(f"B={n} は種文 {len(SEED_TEXTS)} 本を超える")
        body = (SEED_TEXTS[i] + "\n\n") * 40
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}], add_generation_prompt=True
        )
        if len(ids) < prompt_tokens:
            raise SystemExit(f"seed {i} が短い: {len(ids)} < {prompt_tokens}")
        ids = ids[:1] + ids[len(ids) - prompt_tokens + 1 :]
        out.append(ids)
    return out


def record_router_choices(model, prompt_ids, n_steps):
    """1 本を逐次デコードし、各ステップ・各層の top_k エキスパート id を拾う。

    返り値: (n_steps, n_moe_layers, top_k) の入れ子リスト。
    """
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models import qwen4_exp

    rec = {"on": False, "rows": []}
    orig = qwen4_exp.SparseMoeBlock.__call__

    def patched(self, x):
        out = orig(self, x)
        if rec["on"] and x.shape[1] == 1:
            # 直前の argpartition をもう一度引く形にはしない。gate は
            # forward 内で既に評価されているので、同じ式を辿り直しても
            # 追加の重み読み出しは起きない (logits は hidden x 512 の小行列)
            logits = self.gate(x.astype(mx.float32))
            idx = mx.argpartition(-logits, self.top_k - 1, axis=-1)[..., : self.top_k]
            rec["rows"].append(np.asarray(idx).reshape(-1).tolist())
        return out

    qwen4_exp.SparseMoeBlock.__call__ = patched
    try:
        cache = model.make_cache()
        logits = model(mx.array(prompt_ids)[None], cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
        rec["on"] = True
        for _ in range(n_steps):
            logits = model(mx.array([[cur]]), cache=cache)
            cur = int(mx.argmax(logits[0, -1], axis=-1))
        rec["on"] = False
    finally:
        qwen4_exp.SparseMoeBlock.__call__ = orig

    rows = rec["rows"]
    if len(rows) % n_steps:
        raise SystemExit(f"router 呼び出し数 {len(rows)} が {n_steps} で割れない")
    n_layers = len(rows) // n_steps
    return [rows[t * n_layers : (t + 1) * n_layers] for t in range(n_steps)]


def union_stats(per_seq, batches, stagger=0):
    """B 本の記録から、ステップ x 層ごとの和集合サイズを集計する。

    ``stagger`` が 0 でなければ、シーケンス s のステップを s*stagger だけ
    ずらして重ねる。実運用の同時リクエストは互いに違う位置にいるので、
    「全員が同じステップ位置にいる」揃え方より現実に近い。生成の冒頭は
    どのプロンプトでも似た定型句になりがちで、揃えると和集合が小さめに
    出る (= 利得が良く見える) 側に偏る。
    """
    max_off = stagger * (max(batches) - 1)
    n_steps = min(len(r) for r in per_seq) - max_off
    if n_steps <= 0:
        raise SystemExit("stagger が大きすぎてステップが残らない")
    n_layers = len(per_seq[0][0])
    out = {}
    for b in batches:
        sizes = []
        per_step = []
        for t in range(n_steps):
            step_sizes = []
            for l in range(n_layers):
                u = set()
                for s in range(b):
                    u.update(per_seq[s][t + s * stagger][l])
                step_sizes.append(len(u))
            sizes.extend(step_sizes)
            per_step.append(sum(step_sizes) / n_layers)
        out[str(b)] = {
            "batch": b,
            "samples": len(sizes),
            "steps": n_steps,
            "moe_layers": n_layers,
            "unique_experts_per_layer_step_mean": sum(sizes) / len(sizes),
            "unique_experts_per_layer_step_median": statistics.median(sizes),
            "unique_experts_per_layer_step_min": min(sizes),
            "unique_experts_per_layer_step_max": max(sizes),
            # 生成が進んで内容が離れるほど和集合が広がるなら、この列が
            # 右肩上がりになる。頭打ちならバッチ利得は保たれる
            "mean_per_step": per_step,
            # 冒頭の定型句を外した後半だけの平均
            "unique_mean_second_half": sum(sizes[len(sizes) // 2 :])
            / max(len(sizes) - len(sizes) // 2, 1),
        }
    return out


def weight_bytes(model):
    """エキスパート重みとそれ以外を、確保済みバイト数で分ける。"""
    from mlx.utils import tree_flatten

    expert = 0
    other = 0
    for k, v in tree_flatten(model.parameters()):
        n = getattr(v, "nbytes", 0)
        if ".switch_mlp." in k:
            expert += n
        else:
            other += n
    return {"expert_bytes": expert, "other_bytes": other}


def bandwidth_model(bytes_split, union, num_experts, batches):
    """帯域律速なら出るはずの利得。

    1 ステップの読み出し量 = 非エキスパート重み
        + エキスパート重み * (和集合サイズ / 総エキスパート数)
    B 本を 1 ステップで捌けるので、合計スループット比は B * bytes(1)/bytes(B)。
    """

    def step_bytes(b):
        u = union[str(b)]["unique_experts_per_layer_step_mean"]
        return bytes_split["other_bytes"] + bytes_split["expert_bytes"] * (
            u / num_experts
        )

    base = step_bytes(batches[0])
    return {
        str(b): {
            "step_read_bytes": step_bytes(b),
            "predicted_speedup_vs_b1": (b / batches[0]) * base / step_bytes(b),
        }
        for b in batches
    }


# ---------------------------------------------------------------- 観測 2


def code_context(tokenizer, n_tokens):
    """コーディングエージェントらしい中身のプロンプトを、指定長ちょうどで作る。

    このリポジトリ自身のソースを文脈として詰める。散文より、実際に
    opencode が投げてくるものに形が近い。
    """
    chunks = []
    for p in sorted((REPO_ROOT / "mlxturbo").glob("*.py")) + sorted(
        (REPO_ROOT / "tools").glob("*.py")
    ):
        try:
            chunks.append(f"=== {p.relative_to(REPO_ROOT)} ===\n" + p.read_text())
        except Exception:
            continue
    body = "\n\n".join(chunks)
    question = (
        "\n\n上のコードについて質問します。FallbackRunner の prompt cache "
        "再利用条件を説明し、部分巻き戻しをしない設計判断の根拠を、"
        "具体的なコード箇所を引きながら詳しく述べてください。"
    )
    # 二分探索せず、まず余分に取ってから末尾を残して先頭を削る
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": body + question}], add_generation_prompt=True
    )
    if len(ids) < n_tokens:
        raise SystemExit(f"文脈が短い: {len(ids)} < {n_tokens}")
    return ids[:1] + ids[len(ids) - n_tokens + 1 :]


def followup_ids(tokenizer, n_tokens):
    """2 ターン目の追記分 (ユーザーの新しい発言 / ツール結果) を作る。"""
    if n_tokens <= 200:
        body = "続けて、その設計だと困る場面を 3 つ挙げてください。"
    else:
        body = ("=== tool result: cat mlxturbo/fused.py ===\n"
                + (REPO_ROOT / "mlxturbo" / "fused.py").read_text())
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True
    )
    # 先頭のシステム/BOS 部分は履歴の途中には出ないので落とす
    ids = ids[max(0, len(ids) - n_tokens) :]
    if len(ids) < n_tokens:
        ids = ids * (n_tokens // len(ids) + 1)
    return ids[len(ids) - n_tokens :]


def timed_generate(runner, prompt_ids, session, max_tokens):
    import mlx.core as mx

    mx.clear_cache()
    t0 = time.perf_counter()
    res = runner.generate(
        prompt_ids,
        max_tokens=max_tokens,
        temp=0.0,
        eos_ids=set(),
        on_tokens=None,
        session=session,
    )
    wall = time.perf_counter() - t0
    n = len(res["tokens"])
    return {
        "prompt_tokens": len(prompt_ids),
        "prefill_reused": res["prefill_reused"],
        "prefill_new": res["prefill_new"],
        "gen_tokens": n,
        "ttft_s": res["ttft_s"],
        "decode_tps": res["decode_tps"],
        "wall_s": wall,
    }


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    # Defaults point at the author's local layout; override for other machines.
    ap.add_argument(
        "--model", default=str(Path.home() / "models" / "qwen38fn-mlx-v-fast6")
    )
    ap.add_argument(
        "--ngram", default=str(Path.home() / "models" / "qwen38fn-ngram-4bit")
    )
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--union-steps", type=int, default=96)
    ap.add_argument(
        "--stagger",
        type=int,
        default=7,
        help="シーケンス s のステップを s*stagger ずらして重ねる版も出す",
    )
    ap.add_argument("--union-prompt-tokens", type=int, default=512)
    ap.add_argument(
        "--union-prompt-source", choices=("seeds", "natural"), default="seeds"
    )
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gen-tokens", type=int, default=500)
    ap.add_argument("--out-prefix", default="bench/results/flashnext")
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--skip-union", action="store_true")
    args = ap.parse_args()

    batches = [int(x) for x in args.batches.split(",")]

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx

    from mlxturbo._mlx_compat import mlx_lm_load
    from mlxturbo.runner import FallbackSession, build_runner

    t0 = time.perf_counter()
    model, tokenizer, config = mlx_lm_load(args.model, return_config=True)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    class A:
        pass

    a = A()
    a.model = args.model
    a.original = None
    a.mtp_bits = 4
    a.no_mtp = True  # fallback 経路では MTP を使わない。取得も走らせない
    a.no_fused = False
    runner = build_runner(model, tokenizer, config, a)
    kind = getattr(runner, "KIND", None)
    print(f"runner kind = {kind}", flush=True)

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = None

    common = {
        "model": args.model,
        "ngram_sidecar": args.ngram,
        "model_note": "Qwen3.8-Flash-Next (qwen4_exp) v-fast6 / 512 experts, "
        "top_k 10, 48 層 (linear 36 + full 12), moe_intermediate 640",
        "runner_kind": kind,
        "mlx_version": getattr(mx, "__version__", None),
        "platform": platform.platform(),
        "hardware": "Apple M3 Max / 128GB unified",
        "git_commit": commit,
        "git_dirty_note": "コミットしていない作業ツリーで測定",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    # ------------------------------------------------ 観測 2 (先に測る)
    if not args.skip_timing:
        print("\n=== 観測 2: プレフィル / デコード ===", flush=True)
        base_lengths = [2048, 8192, 32768]
        contexts = {n: code_context(tokenizer, n) for n in base_lengths}
        follow = {128: followup_ids(tokenizer, 128), 2048: followup_ids(tokenizer, 2048)}

        # ウォームアップ (捨てる)
        timed_generate(runner, contexts[2048], FallbackSession(), 16)
        mx.clear_cache()

        trials = {}

        def add(key, r):
            trials.setdefault(key, []).append(r)
            print(
                f"  {key:28s} prompt={r['prompt_tokens']:6d} "
                f"(new {r['prefill_new']:6d} / reuse {r['prefill_reused']:6d})  "
                f"ttft={r['ttft_s']:7.3f}s  decode={r['decode_tps']:6.2f} tok/s  "
                f"gen={r['gen_tokens']}",
                flush=True,
            )

        for rep in range(args.reps):
            for n in base_lengths:
                sess = FallbackSession()
                add(f"cold/{n}", timed_generate(runner, contexts[n], sess, args.gen_tokens))
                mx.clear_cache()
                for d in (128, 2048):
                    ids = list(sess.processed) + follow[d]
                    add(
                        f"warm/{n}/+{d}",
                        timed_generate(runner, ids, sess, args.gen_tokens),
                    )
                    mx.clear_cache()
            print(f"  --- rep {rep + 1}/{args.reps} 完了", flush=True)

        def summarize(rs):
            ttft = [r["ttft_s"] for r in rs]
            tps = [r["decode_tps"] for r in rs]
            s = {
                "reps": len(rs),
                "prompt_tokens": rs[0]["prompt_tokens"],
                "prefill_new_median": statistics.median(r["prefill_new"] for r in rs),
                "prefill_reused_median": statistics.median(
                    r["prefill_reused"] for r in rs
                ),
                "ttft_s_median": statistics.median(ttft),
                "ttft_s_iqr": iqr(ttft),
                "decode_tps_median": statistics.median(tps),
                "decode_tps_iqr": iqr(tps),
                "gen_tokens_median": statistics.median(r["gen_tokens"] for r in rs),
            }
            # 出力長ごとのプレフィル比率。デコード時間は実測 tok/s から出す
            for out_n in (200, 500):
                dec = out_n / s["decode_tps_median"]
                s[f"prefill_share_out{out_n}"] = s["ttft_s_median"] / (
                    s["ttft_s_median"] + dec
                )
                s[f"decode_s_out{out_n}"] = dec
            return s

        timing = {k: summarize(v) for k, v in trials.items()}
        payload = {
            "what": "Flash-Next のコーディングエージェント形状での "
            "プレフィル / デコード比",
            "conditions": dict(
                common,
                entry_point="mlxturbo.runner.build_runner (HTTP 非経由)",
                prompt_content="このリポジトリ自身の Python ソースを文脈に詰めた"
                "コード読解リクエスト",
                cold="FallbackSession を新規に作る (プロンプト全量 prefill)。"
                "サブエージェントの初回リクエストに相当",
                warm="同じ session を引き継ぎ、処理済み列の後ろに追記だけを足す "
                "(+128 = 短いユーザー発言、+2048 = ツール結果の貼り付け)。"
                "通常の対話の 2 ターン目以降に相当",
                gen_tokens=args.gen_tokens,
                reps=args.reps,
                sampler="temp=0 / eos 停止は stream_generate 側の判定に任せる",
                interleave="1 プロセス内で rep ごとに全条件を巡回",
                derivation="出力 200/500 のプレフィル比率は、実測 ttft と実測 "
                "decode_tps から出した (decode_s = N / decode_tps)",
            ),
            "summary": timing,
            "trials": trials,
        }
        out2 = REPO_ROOT / f"{args.out_prefix}-prefill-decode.json"
        out2.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"wrote {out2}", flush=True)

    # ------------------------------------------------ 観測 1
    if not args.skip_union:
        print("\n=== 観測 1: 活性エキスパートの和集合 ===", flush=True)
        if args.union_prompt_source == "natural":
            names, prompts = build_union_prompts_natural(max(batches))
            print(f"  プロンプト源: qe-cont {names}", flush=True)
        else:
            names = [f"seed{i}" for i in range(max(batches))]
            prompts = build_union_prompts(
                tokenizer, max(batches), args.union_prompt_tokens
            )
        per_seq = []
        for i, p in enumerate(prompts):
            mx.clear_cache()
            r = record_router_choices(model, p, args.union_steps)
            per_seq.append(r)
            print(f"  seq {i}: {len(r)} steps x {len(r[0])} MoE 層 記録", flush=True)

        num_experts = model.args.text_config["num_experts"]
        top_k = model.args.text_config["num_experts_per_tok"]
        bytes_split = weight_bytes(model)
        union = union_stats(per_seq, batches)
        bw = bandwidth_model(bytes_split, union, num_experts, batches)
        union_stag = union_stats(per_seq, batches, stagger=args.stagger)
        bw_stag = bandwidth_model(bytes_split, union_stag, num_experts, batches)

        payload = {
            "what": "Flash-Next の、デコード 1 ステップ・1 層あたりの "
            "ユニーク活性エキスパート数 (独立 B 本の和集合)",
            "conditions": dict(
                common,
                num_experts=num_experts,
                top_k=top_k,
                method="バッチ生成はしない。B 本を別々に逐次デコードし、"
                "同じステップ位置の router 選択を後から和集合にする",
                prompt_source=args.union_prompt_source,
                prompts="話題の異なる 8 本 (1 本 = 1 話題、複製なし)"
                if args.union_prompt_source == "seeds"
                else "bench/results/qe-cont.json の実プロンプトから 8 本",
                prompt_names=names,
                prompt_tokens=args.union_prompt_tokens
                if args.union_prompt_source == "seeds"
                else [len(p) for p in prompts],
                steps=args.union_steps,
                sampler="argmax",
                stagger_note="expert_union は全シーケンスを同じステップ位置で "
                "重ねた版、expert_union_staggered は s 本目を s*stagger "
                "ステップずらして重ねた版。実運用の同時リクエストは互いに "
                "違う位置にいるので後者の方が現実に近い",
            ),
            "expert_union": union,
            "expert_union_staggered": union_stag,
            "weight_bytes": bytes_split,
            "bandwidth_model": bw,
            "bandwidth_model_staggered": bw_stag,
            "stagger_steps": args.stagger,
            "per_seq_first_step_layer0": [r[0][0] for r in per_seq],
        }
        out1 = REPO_ROOT / f"{args.out_prefix}-expert-union.json"
        out1.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"wrote {out1}", flush=True)
        for b in batches:
            u = union[str(b)]
            us = union_stag[str(b)]
            print(
                f"  B={b:2d}  和集合 mean={u['unique_experts_per_layer_step_mean']:6.1f}"
                f" / {num_experts} (後半 {u['unique_mean_second_half']:5.1f})"
                f"  予測利得 x{bw[str(b)]['predicted_speedup_vs_b1']:.2f}"
                f"   | ずらし: mean={us['unique_experts_per_layer_step_mean']:6.1f}"
                f"  予測 x{bw_stag[str(b)]['predicted_speedup_vs_b1']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
