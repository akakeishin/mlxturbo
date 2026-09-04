"""mlxturbo interactive CLI.

uv run mlxturbo                       # chat REPL
uv run mlxturbo --prompt "..."        # one-shot
"""

import argparse
import os
import sys
import time

from ._mlx_compat import mlx_lm_load, resolve_local_model_path
from .convert import load_quantized_mtp
from .mtp import find_snapshot, load_mtp, load_mtp_file
from .runner import FallbackSession, build_runner
from .spec import ChatSession

# Environment variable used to carry --mtp PATH (server.py) through build_runner
# down to load_cli_mtp. build_runner (mlxturbo/runner.py, owned by another lane
# and therefore off limits for edits) calls load_cli_mtp with a fixed set of
# positional arguments only, so there is no way to pass an extra path without
# editing the caller. This is the same trick as setting FASTMLX_NGRAM_DISK up
# front for --ngram and having from_env pick it up (see the comment at the top
# of cli.py).
MTP_PATH_ENV = "FASTMLX_MTP_PATH"


def load_cli_mtp(
    model_path, config, text_args, original, mtp_bits, no_mtp=False, mtp_path=None
):
    """Load bundled MTP when present, otherwise use the raw source checkpoint.

    MTP is optional: when weights aren't available (or ``--no-mtp`` is
    passed), this returns ``None`` instead of raising. ``SpecEngine`` accepts
    ``mtp=None`` and falls back to lookup-only (SAM) speculation. We warn
    once at startup rather than silently losing the MTP speedup.

    ``mtp_path`` (or, if omitted, the ``FASTMLX_MTP_PATH`` env var — see
    ``MTP_PATH_ENV`` above) points at a single-file MTP sidecar
    (the thing ``load_mtp_file`` operates on). When given, it takes precedence
    over the existing search (the bundled artifact / the raw ``--original``
    checkpoint). If the sidecar's format does not match what ``load_mtp_file``
    expects (tensor names and so on) and loading fails, we do *not* fall back to
    the existing search — we keep the existing stance of logging that it could
    not be read and starting up without MTP.
    """

    if no_mtp:
        print("[mlxturbo] --no-mtp: MTP を無効化し、lookup のみで投機します")
        return None
    mtp_path = mtp_path or os.environ.get(MTP_PATH_ENV)
    if mtp_path:
        try:
            quant = {"bits": mtp_bits, "group_size": 64} if mtp_bits else None
            return load_mtp_file(mtp_path, text_args, quantize=quant)
        except Exception as exc:  # incl. sidecar format mismatch; don't force a fit
            print(
                f"[mlxturbo] --mtp {mtp_path} を読み込めないため無効化します "
                f"({type(exc).__name__}: {exc}); lookup のみで投機します"
            )
            return None
    try:
        if config.get("fastmlx_mtp"):
            return load_quantized_mtp(resolve_local_model_path(model_path), text_args)
        quant = {"bits": mtp_bits, "group_size": 64} if mtp_bits else None
        return load_mtp(find_snapshot(original), text_args, quantize=quant)
    except (FileNotFoundError, RuntimeError) as exc:
        print(
            f"[mlxturbo] MTP 重みが見つからないため無効化します "
            f"({type(exc).__name__}: {exc}); lookup のみで投機します。"
            " 投機を有効にするには MTP (draft) ヘッドが要ります —"
            " 専用の MTP サイドカー (train_mtp.py が吐く単一 safetensors) が"
            " あれば --mtp PATH で渡してください (体感速度は投機の有無で"
            " 1.5-2 倍ほど変わります)"
        )
        return None


def main() -> None:
    # `mlxturbo hub ...` は app/ のメニューバーアプリ向けの JSON 専用の口で、
    # 下の chat REPL の一モードではない。REPL の argparse より前で振り分けて、
    # サブコマンドが --model や --temp と同居しないようにしている。
    if len(sys.argv) > 1 and sys.argv[1] == "hub":
        from .hub import main as hub_main

        hub_main(sys.argv[2:])
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lmstudio-community/Qwen3.8-27B-MLX-4bit")
    ap.add_argument("--original", default="Qwen/Qwen3.8-27B")
    ap.add_argument(
        "--assistant-model",
        default=None,
        metavar="PATH",
        help="Gemma 4 dense 31B 用の gemma4_assistant サイドカーを明示して"
        " B=1 MTP を有効化する。対象外または破損した組み合わせは起動を止める",
    )
    ap.add_argument(
        "--draft-block-size",
        type=int,
        choices=(2, 4, 6, 8),
        default=None,
        help="Gemma 4 assistant MTP の 1 ラウンド総幅 (2/4/6/8、既定4)",
    )
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument(
        "--n-draft", type=int, default=None,
        help="MTPの基本draft数。未指定は族既定（通常3）",
    )
    ap.add_argument(
        "--max-draft", type=int, default=None,
        help="MTPの適応draft上限。未指定は族既定（通常8）",
    )
    ap.add_argument(
        "--lookup-len", type=int, default=None,
        help="n-gram lookupの最大長。未指定は族既定（通常16）",
    )
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument(
        "--no-mtp",
        action="store_true",
        help="MTP を読み込まず lookup (SAM) のみで投機する",
    )
    ap.add_argument(
        "--no-fused",
        action="store_true",
        help="hyper-connections 融合カーネルを無効化する (既定は有効)。"
        "qwen4_exp (Flash-Next 系) の通常生成経路にのみ効く",
    )
    ap.add_argument(
        "--ngram",
        default=None,
        help="n-gram (PLE) 表をチェックポイント本体に持たず外部サイドカーへ"
        "分離してある変換 (mlxturbo/ngram_stream.py) の場合、そのディレクトリ"
        "を指定する。指定すると FASTMLX_NGRAM_DISK=1 を立ててから読み込み、"
        "読み込み後に mlxturbo.ngram_stream.install() で差し替える",
    )
    ap.add_argument(
        "--fly-theta",
        type=float,
        default=0.0,
        help="D6 (FLy) 緩和検証の正規化エントロピー閾値。0 で無効 (既定、"
        "厳密検証)。論文既定は 0.3。有効にすると greedy 出力は厳密で"
        "なくなる (品質 99%%以上という主張は論文値、うちの計測レーンでは未検証)",
    )
    ap.add_argument("--fly-window", type=int, default=6)
    ap.add_argument("--prompt", default=None)
    ap.add_argument(
        "--no-think",
        action="store_true",
        help="思考モードを切る。履歴が追記のみになり、2ターン目以降の prefill が差分だけになる",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.ngram:
        # NGRAM_ON_DISK in qwen4_exp.py is evaluated when the module is imported,
        # so it has to be set before the load call (this holds even when the
        # import happens lazily by way of mlx_lm.utils.load).
        os.environ["FASTMLX_NGRAM_DISK"] = "1"
    model, tokenizer, config = mlx_lm_load(args.model, return_config=True)
    if args.ngram:
        from .ngram_stream import install

        install(model, args.ngram)
    print(f"[mlxturbo] loaded in {time.perf_counter() - t0:.1f}s: {args.model}")
    runner = build_runner(
        model,
        tokenizer,
        config,
        args,
        n_draft=args.n_draft,
        max_draft=args.max_draft,
        lookup_len=args.lookup_len,
    )

    eos_ids = set(tokenizer.eos_token_ids)
    # Pick the session type according to the kind of runner (same reason as in
    # main() of server.py): SpecRunner requires a ChatSession (spec.py's own
    # caches/mtp_cache/h_last), while FallbackRunner requires a FallbackSession
    # (mlx_lm prompt_cache). Passing a ChatSession to both raises an
    # AttributeError on the FallbackRunner side because the `.cache` attribute is
    # missing (see FallbackRunner.generate).
    if getattr(runner, "KIND", None) == "spec":
        session = ChatSession()
    elif getattr(runner, "KIND", None) == "gemma4_assistant_spec":
        from .gemma4_mtp import Gemma4AssistantSession

        session = Gemma4AssistantSession()
    else:
        session = FallbackSession()

    def run_turn(messages):
        kwargs = {"add_generation_prompt": True}
        if args.no_think:
            kwargs["enable_thinking"] = False
        try:
            prompt_ids = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            prompt_ids = tokenizer.apply_chat_template(messages, **kwargs)
        detok = tokenizer.detokenizer
        detok.reset()

        def on_tokens(toks, text=None):
            del text
            for t in toks:
                if t in eos_ids:
                    continue
                detok.add_token(t)
            print(detok.last_segment, end="", flush=True)

        res = runner.generate(
            prompt_ids,
            max_tokens=args.max_tokens,
            temp=args.temp,
            eos_ids=eos_ids,
            on_tokens=on_tokens,
            session=session,
            fly_theta=args.fly_theta,
            fly_window=args.fly_window,
        )
        detok.finalize()
        print(detok.last_segment, flush=True)
        fly_note = (
            f" | fly {res.get('fly_defer_accepts', 0)}" if args.fly_theta > 0 else ""
        )
        print(
            f"\n[{res['decode_tps']:.1f} tok/s | {res['tokens_per_step']:.2f} tok/step"
            f" | ttft {res['ttft_s']:.2f}s"
            f" | prefill 再利用 {res['prefill_reused']} / 新規 {res['prefill_new']}"
            f"{fly_note}]"
        )
        return tokenizer.decode(
            [t for t in res["tokens"] if t not in eos_ids]
        )

    if args.prompt is not None:
        run_turn([{"role": "user", "content": args.prompt}])
        return

    messages = []
    print("[mlxturbo] チャット開始。exit で終了。")
    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": user})
        reply = run_turn(messages)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
