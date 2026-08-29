"""mlxturbo 対話 CLI。

uv run mlxturbo                       # チャット REPL
uv run mlxturbo --prompt "..."        # ワンショット
"""

import argparse
import os
import time

from ._mlx_compat import mlx_lm_load, resolve_local_model_path
from .convert import load_quantized_mtp
from .mtp import find_snapshot, load_mtp, load_mtp_file
from .runner import FallbackSession, build_runner
from .spec import ChatSession

# --mtp PATH (server.py) を build_runner 経由の load_cli_mtp まで運ぶための
# 環境変数。build_runner (mlxturbo/runner.py, 他レーンの担当領域につき編集不可)
# は load_cli_mtp を固定の位置引数だけで呼ぶので、呼び出し元を編集せずに
# 追加のパスを通す口が無い。--ngram を FASTMLX_NGRAM_DISK で先に立てて
# from_env で拾わせているのと同じ手筋 (cli.py 冒頭のコメント参照)。
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
    (``load_mtp_file``の対象)。指定されていれば既存の探索 (バンドル済み
    アーティファクト / ``--original`` の生チェックポイント) より優先する。
    サイドカーの形式が ``load_mtp_file`` の想定 (テンソル名など) と合わず
    読み込みに失敗しても、既存探索へフォールバックはしない — 「読めなかった
    旨をログに出して MTP 無しで起動する」既存の姿勢をそのまま踏襲する。
    """

    if no_mtp:
        print("[mlxturbo] --no-mtp: MTP を無効化し、lookup のみで投機します")
        return None
    mtp_path = mtp_path or os.environ.get(MTP_PATH_ENV)
    if mtp_path:
        try:
            quant = {"bits": mtp_bits, "group_size": 64} if mtp_bits else None
            return load_mtp_file(mtp_path, text_args, quantize=quant)
        except Exception as exc:  # サイドカー形式の不一致まで含め、無理に合わせない
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
            f"({type(exc).__name__}: {exc}); lookup のみで投機します"
        )
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lmstudio-community/Qwen3.8-27B-MLX-4bit")
    ap.add_argument("--original", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--n-draft", type=int, default=3)
    ap.add_argument("--max-draft", type=int, default=8)
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
        "なくなる (品質 >=99% 主張は論文値、うちの計測レーンでは未検証)",
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
        # qwen4_exp.py の NGRAM_ON_DISK はモジュール import 時に評価されるので、
        # (寄り遅延 import される mlx_lm.utils.load 経由でも) 読み込み呼び出し
        # より前に立てておく必要がある。
        os.environ["FASTMLX_NGRAM_DISK"] = "1"
    model, tokenizer, config = mlx_lm_load(args.model, return_config=True)
    if args.ngram:
        from .ngram_stream import install

        install(model, args.ngram)
    print(f"[mlxturbo] loaded in {time.perf_counter() - t0:.1f}s: {args.model}")
    runner = build_runner(
        model, tokenizer, config, args, n_draft=args.n_draft, max_draft=args.max_draft
    )

    eos_ids = set(tokenizer.eos_token_ids)
    # runner の種類に応じて session の型を選ぶ (server.py の main() と同じ理由):
    # SpecRunner は ChatSession (spec.py 独自の caches/mtp_cache/h_last) を、
    # FallbackRunner は FallbackSession (mlx_lm prompt_cache) を要求する。
    # 両方に ChatSession を渡すと FallbackRunner 側で `.cache` 属性が無く
    # AttributeError になる (FallbackRunner.generate 参照)。
    session = ChatSession() if getattr(runner, "KIND", None) == "spec" else FallbackSession()

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

        def on_tokens(toks):
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
