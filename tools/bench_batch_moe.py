"""MoE モデルをバッチ化したときの合計スループットを測る。

同時リクエスト対応 (continuous batching) に投資する価値があるかを決めるための
測定。B を振って「全シーケンスの生成トークン数 ÷ 壁時計」を見る。

- 1 プロセス内で B を交互に振る。別プロセス比較は時間帯で 20-30% 動くので使わない
- プレフィルは計測窓の外に出す。逐次デコードそのものを測る
- 実行は tools/biglock.sh 経由

使い方:
  tools/biglock.sh .venv/bin/python tools/bench_batch_moe.py \
      --model /path/to/model --out bench/results/batch-moe.json
"""

import argparse
import importlib
import json
import platform
import statistics
import subprocess
import time
from collections import Counter

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

_gen = importlib.import_module("mlx_lm.generate")
BatchGenerator = _gen.BatchGenerator


# 内容の違うプロンプトを使う。同じ文を並べるとルータの選ぶエキスパートが
# 揃ってしまい、「B が増えると活性エキスパートの和集合が広がる」という
# 懸念そのものを測れなくなる。
SEED_TEXTS = [
    "Explain how a modern operating system schedules threads across "
    "heterogeneous CPU cores, covering priority inversion and work stealing.",
    "日本の鉄道における信号保安装置の歴史を、腕木式信号機から ATC まで順に "
    "説明してください。事故を契機に導入された仕組みを重点的に。",
    "Describe the chemistry of lithium iron phosphate cells: charge transport, "
    "degradation mechanisms, and why they tolerate deep cycling.",
    "Write a detailed analysis of the double-entry bookkeeping system, its "
    "origins in Venetian trade, and how it constrains accounting errors.",
    "分散データベースにおける合意アルゴリズムを Paxos と Raft の違いを軸に "
    "解説してください。リーダー選出とログ複製の扱いに触れること。",
    "Trace the development of the sonata form from Scarlatti through Haydn to "
    "late Beethoven, focusing on how the recapitulation changed.",
    "Explain the fluid dynamics of a river meander: secondary flow, point bar "
    "deposition, and the conditions under which an oxbow lake forms.",
    "半導体の露光工程について、EUV 光源の生成からレジスト現像までを "
    "工程順に説明してください。歩留まりを落とす要因も挙げること。",
    "Analyse the failure modes of suspension bridge deck design, from Tacoma "
    "Narrows onward, and how aeroelastic testing changed the discipline.",
    "Describe how a compiler performs register allocation via graph colouring, "
    "including spilling heuristics and live range splitting.",
    "中世ヨーロッパの荘園制がどのように解体していったかを、貨幣経済の浸透と "
    "疫病による人口減少の両面から説明してください。",
    "Explain the immunology of mRNA vaccines: lipid nanoparticle delivery, "
    "antigen presentation, and why booster intervals matter.",
    "Give a thorough account of how tectonic plate boundaries produce three "
    "distinct classes of earthquake, with depth profiles for each.",
    "画像圧縮における離散コサイン変換の役割を、量子化テーブルの設計と "
    "ブロックノイズの発生機序まで含めて説明してください。",
    "Describe the operation of a turbofan engine at cruise: bypass ratio, "
    "compressor stall margins, and thrust specific fuel consumption.",
    "Explain how modern search engines build and serve an inverted index, "
    "covering posting list compression and query-time skipping.",
    "Describe the population dynamics of a predator-prey system and where the "
    "Lotka-Volterra model breaks down against field data.",
    "能楽の様式がどのように成立したかを、世阿弥の伝書に見える理念と "
    "上演形態の変遷から説明してください。",
    "Explain how a modern gearbox synchroniser works, the metallurgy of the "
    "blocker ring, and why double-clutching became unnecessary.",
    "Give an account of the Bretton Woods system, why it collapsed in 1971, "
    "and what replaced the dollar-gold link.",
    "Describe the neuroscience of memory consolidation during sleep, "
    "distinguishing hippocampal replay from synaptic homeostasis.",
    "土壌における窒素循環を、固定・硝化・脱窒の各段階と、それを担う微生物 "
    "群集の観点から説明してください。",
    "Explain how sourdough fermentation works: the yeast-lactobacillus "
    "symbiosis, gluten development, and crumb structure.",
    "Trace the evolution of naval gunnery fire control from optical rangefinders "
    "to radar-directed systems, and its effect on engagement ranges.",
    "Explain the design of a modern hydraulic excavator: load-sensing pumps, "
    "boom regeneration circuits, and operator feedback.",
    "写真レンズの収差補正について、球面収差・コマ・像面湾曲をどのように "
    "設計で打ち消すかを、非球面素子の役割まで含めて説明してください。",
    "Describe the epidemiology of antibiotic resistance in hospital settings, "
    "including horizontal gene transfer and stewardship programmes.",
    "Explain the legal doctrine of adverse possession, its justifications, and "
    "how requirements differ across common law jurisdictions.",
    "Describe how a pipe organ produces sound: flue versus reed pipes, wind "
    "chest regulation, and the effect of room acoustics on voicing.",
    "気象予報における数値モデルのアンサンブル手法を、初期値摂動の作り方と "
    "予測可能性の限界という観点から説明してください。",
    "Explain the metallurgy of welding stainless steel: sensitisation, delta "
    "ferrite content, and why post-weld treatment matters.",
    "Give a detailed account of how coffee roasting changes bean chemistry, "
    "from the Maillard reaction through first crack to development time.",
]


def build_prompts(tokenizer, n, prompt_tokens):
    """長さの揃った n 本のプロンプトを作る。"""
    prompts = []
    for i in range(n):
        # 1 本のプロンプトには 1 話題だけを入れる。話題を混ぜると全シーケンスが
        # 似た内容になり、ルータの選ぶエキスパートまで揃ってしまう。
        # 実際の同時リクエストは互いに無関係なので、そちらに寄せる。
        if i >= len(SEED_TEXTS):
            raise SystemExit(f"B={n} は種文 {len(SEED_TEXTS)} 本を超える")
        body = (SEED_TEXTS[i] + "\n\n") * 40
        msg = [{"role": "user", "content": body}]
        ids = tokenizer.apply_chat_template(msg, add_generation_prompt=True)
        if len(ids) < prompt_tokens:
            raise SystemExit(f"seed {i} が短い: {len(ids)} < {prompt_tokens}")
        # 末尾 (生成プロンプト側) を残して先頭を削る
        ids = ids[:1] + ids[len(ids) - prompt_tokens + 1 :]
        assert len(ids) == prompt_tokens
        prompts.append(ids)
    return prompts


def run_trial(model, prompts, gen_tokens):
    """1 回分の計測。プレフィルと最初の 1 トークンは計測窓の外。"""
    b = len(prompts)
    mx.clear_cache()
    mx.reset_peak_memory()

    gen = BatchGenerator(
        model,
        max_tokens=gen_tokens,
        sampler=lambda x: mx.argmax(x, axis=-1),
        completion_batch_size=b,
        prefill_batch_size=b,
    )
    try:
        gen.insert(prompts, [gen_tokens] * b)

        # 1 回目の next_generated がプレフィル全部と最初の 1 トークンを含む
        t_pre = time.perf_counter()
        first = gen.next_generated()
        mx.synchronize()
        prefill_s = time.perf_counter() - t_pre
        if len(first) != b:
            raise SystemExit(f"プレフィルが分割された: {len(first)} != {b}")

        # ここから逐次デコードだけを測る
        t0 = time.perf_counter()
        tokens = 0
        steps = 0
        while responses := gen.next_generated():
            tokens += len(responses)
            steps += 1
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        peak = mx.get_peak_memory() / 1e9
    finally:
        gen.close()

    expected = b * (gen_tokens - 1)
    if tokens != expected:
        raise SystemExit(f"生成数が合わない: {tokens} != {expected}")

    return {
        "batch": b,
        "decode_tokens": tokens,
        "decode_steps": steps,
        "decode_s": elapsed,
        "total_tps": tokens / elapsed,
        "per_seq_tps": steps / elapsed,
        "prefill_s": prefill_s,
        "peak_gb": peak,
    }


def summarize(trials):
    vals = sorted(t["total_tps"] for t in trials)
    per = sorted(t["per_seq_tps"] for t in trials)

    def iqr(xs):
        if len(xs) < 4:
            return max(xs) - min(xs)
        q1 = statistics.quantiles(xs, n=4)[0]
        q3 = statistics.quantiles(xs, n=4)[2]
        return q3 - q1

    return {
        "reps": len(trials),
        "total_tps_median": statistics.median(vals),
        "total_tps_iqr": iqr(vals),
        "total_tps_min": vals[0],
        "total_tps_max": vals[-1],
        "per_seq_tps_median": statistics.median(per),
        "per_seq_tps_iqr": iqr(per),
        "peak_gb_max": max(t["peak_gb"] for t in trials),
        "prefill_s_median": statistics.median(t["prefill_s"] for t in trials),
    }


def measure_expert_union(model, prompts, gen_tokens):
    """B を上げたとき 1 ステップあたりに触るエキスパートが何個に増えるかを見る。

    Router が返す top_k_indices を拾って、デコード 1 ステップ内の
    ユニークエキスパート数を層ごとに数える。計測とは別に走らせる。
    """
    from mlx_lm.models import gemma4_text

    b = len(prompts)
    record = {"active": False, "counts": []}
    orig = gemma4_text.Router.__call__

    def patched(self, x):
        idx, w = orig(self, x)
        if record["active"] and x.shape[1] == 1:
            u = len(set(idx.reshape(-1).tolist()))
            record["counts"].append(u)
        return idx, w

    gemma4_text.Router.__call__ = patched
    try:
        gen = BatchGenerator(
            model,
            max_tokens=gen_tokens,
            sampler=lambda x: mx.argmax(x, axis=-1),
            completion_batch_size=b,
            prefill_batch_size=b,
        )
        gen.insert(prompts, [gen_tokens] * b)
        gen.next_generated()
        mx.synchronize()
        record["active"] = True
        while gen.next_generated():
            pass
        mx.synchronize()
        record["active"] = False
        gen.close()
    finally:
        gemma4_text.Router.__call__ = orig

    counts = record["counts"]
    if not counts:
        return None
    return {
        "batch": b,
        "samples": len(counts),
        "unique_experts_per_layer_step_mean": sum(counts) / len(counts),
        "unique_experts_per_layer_step_median": statistics.median(counts),
        "unique_experts_per_layer_step_max": max(counts),
    }


def weight_bytes(model):
    """エキスパート重みとそれ以外を、実際に確保されているバイト数で分ける。

    量子化済みなので packed weight + scales + biases をそのまま数える。
    デコード 1 ステップの読み出し量を見積もる材料。
    """
    from mlx.utils import tree_flatten

    expert = 0
    other = 0
    for k, v in tree_flatten(model.parameters()):
        n = v.nbytes if hasattr(v, "nbytes") else 0
        if ".experts." in k:
            expert += n
        else:
            other += n
    return {"expert_bytes": expert, "other_bytes": other}


def predict_from_expert_union(
    bytes_split, expert_union, num_experts, batches, summary=None
):
    """活性エキスパートの観測から、帯域律速なら出るはずの利得を出す。

    1 ステップの読み出し量 = 非エキスパート重み + エキスパート重み * (活性数/総数)
    B 本を 1 ステップで捌けるので、合計スループット比は
      B * bytes(1) / bytes(B)
    """
    out = {}
    if str(batches[0]) not in expert_union:
        return out

    def step_bytes(b):
        u = expert_union[str(b)]["unique_experts_per_layer_step_mean"]
        return bytes_split["other_bytes"] + bytes_split["expert_bytes"] * (
            u / num_experts
        )

    base = step_bytes(batches[0])
    for b in batches:
        if str(b) not in expert_union:
            continue
        entry = {
            "step_read_bytes": step_bytes(b),
            "predicted_speedup_vs_b1": (b / batches[0]) * base / step_bytes(b),
        }
        if summary is not None and str(b) in summary:
            # 1 ステップの読み出し量 x ステップ毎秒 = 実効帯域。
            # 予測が実測を上回る分がどこへ消えたかの手がかりになる。
            entry["achieved_gb_per_s"] = (
                step_bytes(b) * summary[str(b)]["per_seq_tps_median"] / 1e9
            )
        out[str(b)] = entry
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="/Users/ht/.lmstudio/models/lmstudio-community/"
        "gemma-4-26B-A4B-it-QAT-MLX-4bit",
    )
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--out", default="bench/results/batch-moe.json")
    p.add_argument("--expert-stats", action="store_true", default=True)
    p.add_argument("--no-expert-stats", dest="expert_stats", action="store_false")
    args = p.parse_args()

    batches = [int(x) for x in args.batches.split(",")]

    model, tokenizer = load(args.model)
    caches = make_prompt_cache(model)
    cache_kinds = dict(Counter(type(c).__name__ for c in caches))
    batchable = all(hasattr(c, "merge") for c in caches)
    print(f"cache: {cache_kinds}  batchable={batchable}")
    if not batchable:
        raise SystemExit("merge を持たないキャッシュがある。測定は成立しない")
    del caches

    bytes_split = weight_bytes(model)
    prompts = build_prompts(tokenizer, max(batches), args.prompt_tokens)

    # ウォームアップ (捨てる)。最大 B と最小 B を 1 回ずつ踏んでおく
    for b in (batches[0], batches[-1]):
        run_trial(model, prompts[:b], 16)
    mx.clear_cache()

    results = {b: [] for b in batches}
    for rep in range(args.reps):
        for b in batches:
            r = run_trial(model, prompts[:b], args.gen_tokens)
            results[b].append(r)
            print(
                f"rep{rep} B={b:2d}  total={r['total_tps']:7.2f} tok/s  "
                f"per-seq={r['per_seq_tps']:6.2f}  peak={r['peak_gb']:.2f} GB  "
                f"prefill={r['prefill_s']:.2f}s"
            )
            mx.clear_cache()

    summary = {str(b): summarize(results[b]) for b in batches}
    base = summary[str(batches[0])]["total_tps_median"]
    for b in batches:
        summary[str(b)]["speedup_vs_b1"] = (
            summary[str(b)]["total_tps_median"] / base
        )

    expert = {}
    if args.expert_stats:
        for b in batches:
            e = measure_expert_union(model, prompts[:b], 32)
            if e:
                expert[str(b)] = e
                print(
                    f"B={b:2d} 層あたり活性エキスパート (1 ステップ): "
                    f"mean={e['unique_experts_per_layer_step_mean']:.1f} / 128"
                )
            mx.clear_cache()

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = None

    payload = {
        "what": "MoE モデルのバッチ化による合計スループット利得の実測",
        "conditions": {
            "model": args.model,
            "model_note": "gemma4 26B-A4B QAT 4bit / 128 experts, top_k 8, "
            "30 層 (sliding 25 + full 5)",
            "cache_kinds": cache_kinds,
            "batchable_by_mlx_lm_rule": batchable,
            "entry_point": "mlx_lm.generate.BatchGenerator (サーバー非経由)",
            "prompt_tokens": args.prompt_tokens,
            "gen_tokens_per_seq": args.gen_tokens,
            "prompts": "話題の異なる 32 本から B 本を採る (1 本 = 1 話題、複製なし)",
            "sampler": "argmax, stop token 無し (必ず gen_tokens 分生成する)",
            "timing": "プレフィルと最初の 1 トークンは計測窓の外。"
            "decode_tokens = B*(gen_tokens-1)",
            "interleave": "1 プロセス内で rep ごとに B を巡回",
            "reps": args.reps,
            "mlx_version": mx.__version__ if hasattr(mx, "__version__") else None,
            "platform": platform.platform(),
            "git_commit": commit,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "criteria": {
            "invest": "B=4 の合計 tok/s が B=1 の 2.5 倍超",
            "reject": "1.5 倍未満",
            "hold": "1.5〜2.5 倍",
        },
        "summary": summary,
        "expert_union": expert,
        "weight_bytes": bytes_split,
        "bandwidth_model": predict_from_expert_union(
            bytes_split, expert, 128, batches, summary
        ),
        "trials": {str(b): results[b] for b in batches},
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")
    for b in batches:
        s = summary[str(b)]
        print(
            f"B={b:2d}  total {s['total_tps_median']:7.2f} tok/s "
            f"(IQR {s['total_tps_iqr']:.2f})  x{s['speedup_vs_b1']:.2f}  "
            f"per-seq {s['per_seq_tps_median']:6.2f}  peak {s['peak_gb_max']:.2f} GB"
        )
    bw = payload["bandwidth_model"]
    if bw:
        print("\n帯域モデル (活性エキスパート数から出した予測):")
        for b in batches:
            if str(b) in bw:
                print(
                    f"B={b:2d}  予測 x{bw[str(b)]['predicted_speedup_vs_b1']:.2f}  "
                    f"実測 x{summary[str(b)]['speedup_vs_b1']:.2f}  "
                    f"実効帯域 {bw[str(b)].get('achieved_gb_per_s', 0):.0f} GB/s"
                )


if __name__ == "__main__":
    main()
