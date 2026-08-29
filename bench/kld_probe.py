"""bf16 参照モデルと量子化モデルの teacher-forced KLD を測る最小実装。

docs/PLAN.md Phase C1 の品質ゲート下ごしらえ:
    「品質ゲートは KLD を主指標にする: bf16 参照に対する出力分布の KL
    divergence を固定評価セットで測り、閾値超えは速度がどうであれ不合格。
    greedy 一致率と logit 差は補助指標にする」

  重要: このスクリプトは実行しない（GPU 実行の準備タスクの一部として作成）。
  bf16 の参照モデル（既定 Qwen/Qwen3.8-27B、~54GB）はローカルキャッシュ済み
  の前提。ネットワークダウンロードは行わず、--require-local-cache（既定 on）
  が事前に huggingface_hub のローカルキャッシュを確認し、無ければ
  `hf download <repo>` を促して即座に失敗する。

同一性レーンについて: 依頼元は docs/KERNEL-INTEL.md に「KLD 計測レーン」節
（token_ids を同一性レーンに使う、という注意）があるとしていたが、探索時点
（2026-08-26）の同ファイルにはその見出しは存在しなかった。最も近い記述は
docs/PLAN.md Phase C1（上記引用）。ここでは teacher forcing の一般原則から
以下を実装する:

    bf16 参照モデル自身の greedy 継続を「正解の token_ids」として固定し、
    bf16・量子化モデルの両方に *同じ* token_ids を teacher force する。
    各モデルが自分の argmax で異なる続きへ分岐すると、後半の KLD が
    「分布の違い」ではなく「文脈（それまでのトークン列）の違い」を測って
    しまう。同一 token_ids を両モデルの「同一性レーン」として使うことで、
    KLD が純粋に (同じ文脈が与えられたときの) 出力分布の差だけを表すように
    する。

生成した継続トークン列は --continuation-cache に保存し、複数の量子化候補
を同じ bf16 継続で評価できるようにする（bf16 モデルは KLD 計算そのものに
毎回必要だが、貪欲デコードのシーケンシャルループは 1 回で済む）。

使い方（例。実行は GPU 作業なので静音プロトコル後に）:

    uv run --project <repo> python bench/kld_probe.py \\
        --ref-model Qwen/Qwen3.8-27B \\
        --quant-model lmstudio-community/Qwen3.8-27B-MLX-4bit \\
        --prompts all --gen-tokens 128 \\
        --output bench/results/kld-lmstudio-4bit-$(date -u +%Y%m%dT%H%M%SZ).json

まずはトークナイザだけ読んでプレビュー（GPU 不使用、モデル重みは読まない）:

    uv run --project <repo> python bench/kld_probe.py \\
        --quant-model lmstudio-community/Qwen3.8-27B-MLX-4bit --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# bench/spec_bench.py は import 時点で mlx.core / mlx_lm / mlxturbo.mtp /
# mlxturbo.spec を import する（クラス定義のみ。GPU 計算は発生しない）。
from bench.spec_bench import PROMPTS  # noqa: E402

DEFAULT_REF_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_GEN_TOKENS = 128
DEFAULT_CACHE_PATH = REPO_ROOT / "bench" / "results" / "kld-continuations.json"


# ---------------------------------------------------------------------------
# ローカルキャッシュ確認（ネットワークダウンロードを絶対に踏まないためのガード）
# ---------------------------------------------------------------------------


def _looks_like_local_path(model_ref: str) -> bool:
    return Path(model_ref).expanduser().exists()


def check_local_snapshot(model_ref: str) -> None:
    """model_ref がローカル HF キャッシュに完全に存在することを確認する。

    ネットワークへは一切出ない（local_files_only=True）。無ければ
    SystemExit で即座に落とし、`hf download` の実行を促す。
    """

    if _looks_like_local_path(model_ref):
        return
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        snapshot_download(model_ref, local_files_only=True)
    except LocalEntryNotFoundError as exc:
        raise SystemExit(
            f"{model_ref} がローカル HF キャッシュに見つからない。"
            f"先に `hf download {model_ref}` を実行すること"
            f"（このスクリプトはネットワークダウンロードを行わない設計）。"
            f"detail: {exc}"
        ) from exc
    except Exception as exc:  # pragma: no cover - ベストエフォート
        raise SystemExit(
            f"{model_ref} のローカルキャッシュ確認に失敗した: {exc}. "
            "--no-require-local-cache で確認をスキップできるが、その場合 "
            "load() が黙ってネットワークへ出る可能性がある点に注意"
        ) from exc


# ---------------------------------------------------------------------------
# トークン列の用意（同一性レーン）
# ---------------------------------------------------------------------------


def build_prompt_ids(tokenizer: Any, prompt_text: str, enable_thinking: bool) -> list[int]:
    messages = [{"role": "user", "content": prompt_text}]
    kwargs: dict[str, Any] = {"add_generation_prompt": True}
    if not enable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return list(tokenizer.apply_chat_template(messages, **kwargs))
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return list(tokenizer.apply_chat_template(messages, **kwargs))


def eos_ids_of(tokenizer: Any) -> set[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return set(ids)
    single = getattr(tokenizer, "eos_token_id", None)
    return {single} if single is not None else set()


def generate_canonical_continuation(
    model: Any, tokenizer: Any, prompt_ids: list[int], max_new_tokens: int
) -> list[int]:
    """bf16 参照モデル自身の greedy 継続。両モデルへの teacher-forcing の
    ターゲットになる「同一性レーン」の token_ids。"""

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    eos = eos_ids_of(tokenizer)
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt_ids)[None], cache=cache)
    next_id = int(mx.argmax(logits[0, -1], axis=-1))
    generated: list[int] = []
    for _ in range(max_new_tokens):
        if next_id in eos:
            break
        generated.append(next_id)
        logits = model(mx.array([[next_id]]), cache=cache)
        next_id = int(mx.argmax(logits[0, -1], axis=-1))
    return generated


def load_or_build_continuation(
    args: argparse.Namespace,
    ref_model: Any,
    tokenizer: Any,
    prompt_key: str,
    prompt_text: str,
) -> dict[str, Any]:
    cache_path = args.continuation_cache
    cache: dict[str, Any] = {}
    if cache_path.exists() and not args.no_cache:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    cache_key = f"{prompt_key}|gen{args.gen_tokens}|think{int(args.enable_thinking)}|{args.ref_model}"
    if cache_key in cache and not args.no_cache:
        return cache[cache_key]

    prompt_ids = build_prompt_ids(tokenizer, prompt_text, args.enable_thinking)
    continuation_ids = generate_canonical_continuation(
        ref_model, tokenizer, prompt_ids, args.gen_tokens
    )
    entry = {
        "prompt_ids": prompt_ids,
        "continuation_ids": continuation_ids,
        "ref_model": args.ref_model,
        "gen_tokens_requested": args.gen_tokens,
        "enable_thinking": args.enable_thinking,
    }
    cache[cache_key] = entry
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


# ---------------------------------------------------------------------------
# teacher-forced logits と KLD
# ---------------------------------------------------------------------------


def teacher_forced_logits(model: Any, full_ids: list[int]):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array(full_ids)[None], cache=cache)
    mx.eval(logits)
    return logits[0]  # (seq_len, vocab)


def compute_kld_for_prompt(
    args: argparse.Namespace,
    ref_model: Any,
    quant_model: Any,
    tokenizer: Any,
    prompt_key: str,
    prompt_text: str,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn

    entry = load_or_build_continuation(args, ref_model, tokenizer, prompt_key, prompt_text)
    prompt_ids = entry["prompt_ids"]
    continuation_ids = entry["continuation_ids"]
    if not continuation_ids:
        return {
            "prompt_key": prompt_key,
            "error": "bf16 継続が 0 トークン（即 EOS）。gen-tokens やプロンプトを見直すこと",
        }

    full_ids = prompt_ids + continuation_ids
    prompt_len = len(prompt_ids)
    gen_len = len(continuation_ids)

    logits_ref = teacher_forced_logits(ref_model, full_ids)
    logits_quant = teacher_forced_logits(quant_model, full_ids)

    # 継続トークンを予測する区間だけを見る（プロンプト部分は品質ゲート対象外）
    span_ref = logits_ref[prompt_len - 1 : prompt_len - 1 + gen_len].astype(mx.float32)
    span_quant = logits_quant[prompt_len - 1 : prompt_len - 1 + gen_len].astype(mx.float32)

    log_p_ref = nn.log_softmax(span_ref, axis=-1)
    log_q_quant = nn.log_softmax(span_quant, axis=-1)

    # KL(P_ref || Q_quant) = sum_v exp(target)*(target - input)
    kld_per_pos = nn.losses.kl_div_loss(
        inputs=log_q_quant, targets=log_p_ref, axis=-1, reduction="none"
    )
    mx.eval(kld_per_pos)

    argmax_ref = mx.argmax(span_ref, axis=-1)
    argmax_quant = mx.argmax(span_quant, axis=-1)
    target_ids = mx.array(continuation_ids)

    # 自己整合性チェック: bf16 の逐次貪欲デコードと、同じ token_ids を一括
    # teacher-force した時の argmax は本来一致するはず（数値的な非決定性が
    # あれば不一致が出る。同一性レーンが壊れていないことの確認になる）
    self_consistency_mismatches = int(mx.sum((argmax_ref != target_ids).astype(mx.int32)))

    greedy_agreement = float(mx.mean((argmax_ref == argmax_quant).astype(mx.float32)))

    idx = mx.arange(gen_len)
    ref_logit_at_target = span_ref[idx, target_ids]
    quant_logit_at_target = span_quant[idx, target_ids]
    logit_diff = ref_logit_at_target - quant_logit_at_target
    quant_prob_at_ref_argmax = mx.exp(log_q_quant[idx, target_ids])

    mx.eval(
        greedy_agreement,
        logit_diff,
        quant_prob_at_ref_argmax,
        self_consistency_mismatches,
    )

    kld_list = [float(x) for x in kld_per_pos.tolist()]

    return {
        "prompt_key": prompt_key,
        "prompt_tokens": prompt_len,
        "continuation_tokens": gen_len,
        "kld_mean": sum(kld_list) / len(kld_list),
        "kld_max": max(kld_list),
        "kld_first10": kld_list[:10],
        "kld_last10": kld_list[-10:],
        "greedy_agreement": greedy_agreement,
        "self_consistency_mismatches": self_consistency_mismatches,
        "mean_abs_logit_diff_at_realized_token": float(mx.mean(mx.abs(logit_diff))),
        "mean_quant_prob_at_ref_argmax": float(mx.mean(quant_prob_at_ref_argmax)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "bf16 参照モデルと量子化モデルの teacher-forced KLD を測る。"
            "実行すると GPU 推論が走る（bf16 27B のロード含む）。"
            "--dry-run はトークナイザのみ読み込みプレビューする"
        )
    )
    ap.add_argument("--ref-model", default=DEFAULT_REF_MODEL)
    ap.add_argument("--quant-model", required=True)
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=["all"],
        help=f"bench/spec_bench.py の PROMPTS キー、または 'all'（現状: {sorted(PROMPTS)}）",
    )
    ap.add_argument("--gen-tokens", type=int, default=DEFAULT_GEN_TOKENS)
    ap.add_argument("--enable-thinking", action="store_true", default=False)
    ap.add_argument(
        "--require-local-cache",
        action="store_true",
        default=True,
        help="ref/quant モデルがローカル HF キャッシュに無ければ即失敗する（既定 on）",
    )
    ap.add_argument(
        "--no-require-local-cache", dest="require_local_cache", action="store_false"
    )
    ap.add_argument("--continuation-cache", type=Path, default=DEFAULT_CACHE_PATH)
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="継続トークンのキャッシュを使わず毎回 bf16 で再生成する",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="トークナイザだけ読んでプロンプト長を表示する。モデル重みは読まない・GPU 不使用",
    )
    return ap.parse_args(argv)


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    from mlx_lm.utils import load_tokenizer

    check_kwargs_ok = True
    if args.require_local_cache:
        check_local_snapshot(args.ref_model)
        check_local_snapshot(args.quant_model)
    tokenizer = load_tokenizer(args.ref_model)
    prompt_keys = list(PROMPTS.keys()) if args.prompts == ["all"] else args.prompts
    preview = []
    for key in prompt_keys:
        if key not in PROMPTS:
            raise SystemExit(f"unknown prompt key: {key} (choices: {sorted(PROMPTS)})")
        ids = build_prompt_ids(tokenizer, PROMPTS[key], args.enable_thinking)
        preview.append({"prompt_key": key, "prompt_tokens": len(ids)})
    return {
        "dry_run": True,
        "ref_model": args.ref_model,
        "quant_model": args.quant_model,
        "local_cache_checked": args.require_local_cache and check_kwargs_ok,
        "prompt_preview": preview,
    }


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    from mlx_lm import load

    if args.require_local_cache:
        check_local_snapshot(args.ref_model)
        check_local_snapshot(args.quant_model)

    prompt_keys = list(PROMPTS.keys()) if args.prompts == ["all"] else args.prompts
    for key in prompt_keys:
        if key not in PROMPTS:
            raise SystemExit(f"unknown prompt key: {key} (choices: {sorted(PROMPTS)})")

    t0 = time.perf_counter()
    ref_model, tokenizer = load(args.ref_model)
    ref_load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    quant_model, _quant_tokenizer = load(args.quant_model)
    quant_load_s = time.perf_counter() - t0

    per_prompt = []
    for key in prompt_keys:
        per_prompt.append(
            compute_kld_for_prompt(args, ref_model, quant_model, tokenizer, key, PROMPTS[key])
        )

    valid = [p for p in per_prompt if "kld_mean" in p]
    aggregate = {}
    if valid:
        total_tokens = sum(p["continuation_tokens"] for p in valid)
        aggregate = {
            "kld_mean_unweighted": sum(p["kld_mean"] for p in valid) / len(valid),
            "kld_mean_token_weighted": (
                sum(p["kld_mean"] * p["continuation_tokens"] for p in valid) / total_tokens
            ),
            "greedy_agreement_mean": sum(p["greedy_agreement"] for p in valid) / len(valid),
            "total_self_consistency_mismatches": sum(
                p["self_consistency_mismatches"] for p in valid
            ),
            "total_continuation_tokens": total_tokens,
        }

    return {
        "ref_model": args.ref_model,
        "quant_model": args.quant_model,
        "ref_load_s": ref_load_s,
        "quant_load_s": quant_load_s,
        "gen_tokens_requested": args.gen_tokens,
        "enable_thinking": args.enable_thinking,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "per_prompt": per_prompt,
        "aggregate": aggregate,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quant_slug = args.quant_model.split("/")[-1]
        args.output = REPO_ROOT / "bench" / "results" / f"kld-{quant_slug}-{stamp}.json"

    result = run_dry_run(args) if args.dry_run else run_full(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[kld_probe] wrote {args.output}")
    if not args.dry_run and result.get("aggregate"):
        agg = result["aggregate"]
        print(
            f"[kld_probe] KLD mean (token-weighted) = "
            f"{agg['kld_mean_token_weighted']:.4f}  "
            f"greedy_agreement = {agg['greedy_agreement_mean']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
