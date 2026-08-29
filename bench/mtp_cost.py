"""D0 追補: 投機 1 step のコスト分解マイクロベンチ。

同一プロセス・同一文脈で以下を直接計測し、README や別セッションの数値の
寄せ集めで推定していた draft 単価を 1 本の数値にする。

- draft 1 トークンの単価と内訳: MTP ブロック (_mtp_append) / head
  (mtp.norm + lm_head + argmax)
- 検証 forward の m カーブ: m=1,2,4,6,8 の壁時計と m=1 比
- rollback の単価

ctx は code プロンプトを greedy で伸ばして作る (既定 512)。反復中に
文脈が数百トークン漂うが、比較目的には効かない。バックグラウンドの
CPU 負荷がある環境では絶対値が数割ぶれるので、判定は倍率と桁で行う。

使い方:
  uv run python bench/mtp_cost.py --json bench/results/mtp-cost.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate import PROMPTS  # noqa: E402


def timed(fn, reps, warmup=5):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "reps": reps,
    }


def main():
    import mlx.core as mx

    from mlxturbo._mlx_compat import KVCache, TextModelArgs, mlx_lm_load
    from mlxturbo.mtp import find_snapshot, load_mtp
    from mlxturbo.spec import SpecEngine

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="lmstudio-community/Qwen3.8-27B-MLX-4bit"
    )
    parser.add_argument("--original", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--mtp-bits", type=int, default=0)
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()

    model, tokenizer = mlx_lm_load(args.model)
    text_args = TextModelArgs.from_dict(model.args.text_config)
    quant = {"bits": args.mtp_bits, "group_size": 64} if args.mtp_bits else None
    mtp = load_mtp(find_snapshot(args.original), text_args, quantize=quant)
    mx.eval(mtp.parameters())
    engine = SpecEngine(model, mtp)

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPTS["code"]}],
        add_generation_prompt=True,
    )

    # ctx 長まで greedy で文脈を作る (m=1 の非投機ループ)。
    caches = engine.text.make_cache()
    mtp_cache = KVCache()
    prompt = mx.array(prompt_ids)
    h_all, _ = engine._hidden_forward(prompt, caches, capture=False)
    engine._mtp_append(prompt[1:], h_all[:, :-1], mtp_cache)
    h_last = h_all[:, -1:]
    y = mx.argmax(engine._head(h_last, engine.inner.norm), axis=-1).reshape(1)
    mx.eval(y)
    while mtp_cache.offset < args.ctx:
        hs, _ = engine._hidden_forward(y, caches, capture=False)
        engine._mtp_append(y, h_last, mtp_cache)
        h_last = hs[:, -1:]
        y = mx.argmax(
            engine._head(h_last, engine.inner.norm), axis=-1
        ).reshape(1)
        mx.eval(y)
    ctx_len = mtp_cache.offset
    results = {"ctx_len": ctx_len, "mtp_bits": args.mtp_bits}

    # --- draft 単価 ---
    def draft_block_only():
        h = engine._mtp_append(y, h_last, mtp_cache)
        mx.eval(h)
        mtp_cache.trim(1)

    results["draft_block_ms"] = timed(draft_block_only, args.reps)

    def head_only():
        d = mx.argmax(engine._head(h_last, engine.mtp.norm), axis=-1)
        mx.eval(d)

    results["draft_head_ms"] = timed(head_only, args.reps)

    def draft_token_full():
        h = engine._mtp_append(y, h_last, mtp_cache)
        d = mx.argmax(engine._head(h[:, -1:], engine.mtp.norm), axis=-1)
        mx.eval(d)
        mtp_cache.trim(1)

    results["draft_token_ms"] = timed(draft_token_full, args.reps)

    # --- 検証 m カーブ (capture 込み、直後に consumed=1 で rollback) ---
    verify = {}
    rollback = {}
    for m in (1, 2, 4, 6, 8):
        window = mx.concatenate([y] * m) if m > 1 else y
        rb_samples = []

        def verify_m():
            hs, sink = engine._hidden_forward(
                window, caches, capture=m > 1
            )
            preds = mx.argmax(engine._head(hs, engine.inner.norm), axis=-1)
            mx.eval(preds)
            if m > 1:
                t0 = time.perf_counter()
                engine._rollback(caches, sink, m, 1)
                mx.eval(caches[engine.inner.ssm_idx][1])
                rb_samples.append((time.perf_counter() - t0) * 1e3)

        verify[f"m{m}"] = timed(verify_m, args.reps)
        if rb_samples:
            rollback[f"m{m}"] = {
                "mean_ms": statistics.mean(rb_samples[5:]),
                "reps": len(rb_samples) - 5,
            }

    base = verify["m1"]["median_ms"]
    for m, v in verify.items():
        v["vs_m1"] = v["median_ms"] / base
    results["verify_ms"] = verify
    results["rollback_ms"] = rollback

    # 導出値: 検証は rollback 込みの数値も出す
    results["derived"] = {
        "draft_token_ms_median": results["draft_token_ms"]["median_ms"],
        "draft_block_share": (
            results["draft_block_ms"]["median_ms"]
            / results["draft_token_ms"]["median_ms"]
        ),
        "m1_step_ms": base,
        "note": "背景CPU負荷下の参考値。判定は倍率と桁で行うこと",
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
