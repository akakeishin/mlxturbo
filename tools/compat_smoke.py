#!/usr/bin/env python
"""互換性スモークツール: 著名な量子化パック (mlx-community 等) が mlxturbo の上で
「壊れずに動くか」「どの最適化が有効/フォールバックしたか」を 1 コマンドで確認する。

    python tools/compat_smoke.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
    python tools/compat_smoke.py --model ~/models/some-gs32-pack

サーバーは立てない。流れは (1) ロード (mlx_lm 経由)、(2) runner 構築
(mlxturbo.runner.build_runner、cli.py/server.py が起動時に呼ぶのと同じ入口)、
(3) 3 プロンプト x 64 トークンの greedy 生成、(4) どの最適化が効いたかの
1 行マトリクス出力、の順。

tools/smoke_generate.py は生成品質を目視で確認するための道具で、これとは
役割が違う。こちらは「経路の可視化」と「壊れていないかの粗い機械判定」に徹する
(出力が空でないか、同一トークンが異常に連続していないかだけを見る。品質その
ものの良し悪しは判定しない)。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Once upon a time, in a small village,",
]

DEFAULT_MAX_TOKENS = 64
REPEAT_LIMIT = 32  # 同一トークンがこれ以上連続したら生成が壊れている疑い


def _detect_quantization(model) -> str:
    """モデル内の量子化済み Linear から (bits, group_size, mode) の組を拾う。

    複数の組み合わせが混在していれば "mixed" として全部並べる (層ごとに bits/gs
    が違う混合精度パックの検出。DWQ 等、層単位で bits が違う構成もここに出る)。
    """
    combos = set()
    for _, mod in model.named_modules():
        if hasattr(mod, "scales"):
            combos.add((mod.bits, mod.group_size, getattr(mod, "mode", "affine")))
    if not combos:
        return "quantize なし (bf16/fp16)"
    if len(combos) == 1:
        bits, gs, mode = next(iter(combos))
        return f"{bits}bit/gs{gs}/{mode}"
    parts = ", ".join(f"{b}bit/gs{g}/{m}" for b, g, m in sorted(combos))
    return f"mixed[{parts}]"


def _has_moe(model) -> bool:
    for _, mod in model.named_modules():
        if type(mod).__name__ == "SwitchGLU":
            return True
    return False


def _hc_kernel_status(model) -> str:
    """hyper-connections 融合カーネル (fused.py) が実際にこのモデルへ効くか。

    パッチ自体は qwen4_exp (Flash-Next) のクラスへ無条件にあたるので、
    「パッチが入っているか」と「このモデルで実際に使われるか」は別。後者は
    モデルの model_type で判定する。
    """
    from mlxturbo import fused

    if fused._ORIG_HC_KERNEL is None:
        return "off (--no-fused)"
    arch = getattr(getattr(model, "args", None), "model_type", None)
    if arch == "qwen4_exp":
        return "active (qwen4_exp)"
    return f"dormant (このモデルは対象外: model_type={arch})"


def _gather_sort_status() -> str:
    from mlxturbo import fused

    if fused._ORIG_SWITCH_SORT is None:
        return "off"
    sort_min = int(os.environ.get("MLXTURBO_SORT_MIN", "16"))
    return f"on (閾値 {sort_min})" if sort_min else "off (MLXTURBO_SORT_MIN=0)"


def _opt_flag_status(name: str, env: str) -> str:
    return "on" if os.environ.get(env) == "1" else "off (既定)"


def _set_wired_limit() -> str:
    import mlx.core as mx

    try:
        rec = mx.metal.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(rec)
        return f"ok ({rec / 2**30:.0f}GiB)"
    except Exception as exc:  # noqa: BLE001  wire できない環境でも続行する
        return f"failed ({type(exc).__name__}: {exc})"


def _repeat_run_length(tokens: list[int]) -> int:
    """同一トークンの最長連続数。"""
    best = run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="モデルのディレクトリ、または Hugging Face の repo id")
    ap.add_argument("--original", default="Qwen/Qwen3.8-27B",
                    help="qwen3_5 (27B) 系の SpecEngine が生チェックポイントを"
                    " 探すときの参照名。対象外のモデルでは使われない")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument(
        "--ngram",
        default=None,
        help="n-gram (PLE) 表を外部サイドカーへ分離してある変換の場合、そのディレクトリ"
        " (cli.py/server.py の --ngram と同じ)",
    )
    ap.add_argument(
        "--mtp",
        default=None,
        metavar="PATH",
        help="MTP ヘッドを単一 safetensors サイドカーから読み込む場合のパス"
        " (cli.py/server.py の --mtp と同じ)。qwen4_exp (Flash-Next) では"
        " 自動発見より優先、qwen3_5 (27B 系) では単一ファイルの draft head"
        " を期待する (mlxturbo/mtp.py の load_mtp_file 参照。既に量子化済みの"
        " MTP 専用パックはこの形を満たさず読み込みに失敗することがある —"
        " その場合もクラッシュせず理由を出して MTP 無しに倒れる)",
    )
    args_cli = ap.parse_args(argv)

    from mlxturbo._mlx_compat import mlx_lm_load
    from mlxturbo.runner import FallbackSession, build_runner
    from mlxturbo.spec import ChatSession

    if args_cli.ngram:
        # NGRAM_ON_DISK はモジュール import 時に読まれるので、load より前に
        # 立てる必要がある (cli.py/server.py と同じ理由)
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    if args_cli.mtp:
        # build_runner は qwen4_exp 以外 (qwen3_5/27B 系) では --mtp を
        # load_cli_mtp へ直接橋渡ししない (mtp_path 引数は env 経由のみ)。
        # server.py/cli.py と同じ環境変数トリックで渡す
        # (mlxturbo/cli.py の MTP_PATH_ENV / MTP_PATH_ENV のコメント参照)
        from mlxturbo.cli import MTP_PATH_ENV

        os.environ[MTP_PATH_ENV] = args_cli.mtp

    print(f"[compat_smoke] loading: {args_cli.model}")
    t0 = time.perf_counter()
    try:
        model, tokenizer, config = mlx_lm_load(args_cli.model, return_config=True)
    except Exception as exc:  # noqa: BLE001  ロード失敗はここで打ち切る
        print(f"[compat_smoke] FAIL ロードで例外: {type(exc).__name__}: {exc}")
        return 1
    print(f"[compat_smoke] loaded in {time.perf_counter() - t0:.1f}s")

    if args_cli.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args_cli.ngram)
        print(f"[compat_smoke] n-gram サイドカー導入: {args_cli.ngram}")

    wired = _set_wired_limit()
    quant = _detect_quantization(model)
    has_moe = _has_moe(model)
    arch = getattr(getattr(model, "args", None), "model_type", None)

    # server.py/cli.py が build_runner に渡す argparse.Namespace と同じ形の
    # 最小限のスタンドイン。サーバーは起動しない (--host/--port 等は不要)
    build_args = SimpleNamespace(
        model=args_cli.model,
        original=args_cli.original,
        mtp_bits=4,
        mtp_depth=None,
        no_mtp=False,
        mtp=args_cli.mtp,
        no_fused=False,
        draft_model=None,
        num_draft_tokens=4,
        lookup_spec=False,
        lookup_max_draft=8,
        lookup_min_match=2,
    )

    try:
        runner = build_runner(
            model, tokenizer, config, build_args, log_prefix="[compat_smoke]"
        )
    except SystemExit as exc:
        print(f"[compat_smoke] FAIL runner 構築が exit({exc.code}) しました"
              " (--mtp 等、明示指定の読み込み失敗)")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[compat_smoke] FAIL runner 構築で例外: {type(exc).__name__}: {exc}")
        return 1

    hc_status = _hc_kernel_status(model)
    gather_sort = _gather_sort_status()
    moe_glu = _opt_flag_status("moe_glu", "MLXTURBO_MOE_GLU")
    fast_qmm = _opt_flag_status("fast_qmm", "MLXTURBO_FAST_QMM")
    wide = _opt_flag_status("wide", "MLXTURBO_WIDE")
    kind = getattr(runner, "KIND", "?")
    reason = getattr(runner, "fallback_reason", None)
    kind_note = kind if reason is None else f"{kind} ({reason})"

    # MTP 経路: spec/flash_spec のときだけ意味がある。SpecEngine/FlashSpecEngine
    # は生成した runner.engine.mtp に読み込んだ draft head をそのまま持つので、
    # 出典 (bundled/sidecar/none) はそこから直接読める (ログの文面をパースしない)。
    engine = getattr(runner, "engine", None)
    mtp_obj = getattr(engine, "mtp", None) if engine is not None else None
    if kind not in ("spec", "flash_spec"):
        mtp_path = "n/a (このモデルは spec/flash_spec 対象外)"
    elif mtp_obj is None:
        mtp_path = "none (投機は lookup のみ。上のログ参照)"
    elif args_cli.mtp:
        mtp_path = f"sidecar (--mtp {args_cli.mtp})"
    else:
        mtp_path = "bundled (自動発見。上のログ参照)"

    print(
        "[compat_smoke] matrix: "
        f"arch={arch} quant={quant} moe={'あり' if has_moe else 'なし'} "
        f"runner={kind_note} mtp={mtp_path} hc_kernel={hc_status} "
        f"gather_sort={gather_sort} moe_glu={moe_glu} fast_qmm={fast_qmm} "
        f"wide_proj={wide} wired={wired}"
    )

    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    session_factory = ChatSession if kind == "spec" else FallbackSession

    all_ok = True
    for prompt in PROMPTS:
        session = session_factory()
        try:
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001  chat template が無いトークナイザ向け
            prompt_ids = tokenizer.encode(prompt)

        try:
            res = runner.generate(
                prompt_ids,
                max_tokens=args_cli.max_tokens,
                temp=0.0,
                eos_ids=eos_ids,
                # runner の種類によって on_tokens(toks) / on_tokens(toks, text)
                # の両方が来る (server.py の on_tokens と同じ受け方に揃える)
                on_tokens=lambda _toks, _text=None: None,
                session=session,
            )
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            print(f"[compat_smoke] FAIL {prompt[:30]!r}: 生成で例外 "
                  f"{type(exc).__name__}: {exc}")
            continue

        toks = [t for t in res.get("tokens", []) if t not in eos_ids]
        if not toks:
            all_ok = False
            print(f"[compat_smoke] FAIL {prompt[:30]!r}: 生成トークンが空")
            continue

        run_len = _repeat_run_length(toks)
        if run_len >= REPEAT_LIMIT:
            all_ok = False
            print(f"[compat_smoke] FAIL {prompt[:30]!r}: 同一トークンが "
                  f"{run_len} 連続 (閾値 {REPEAT_LIMIT}) — 生成が壊れている疑い")
            continue

        text = tokenizer.decode(toks)
        tps = res.get("decode_tps", 0.0)
        tok_step = res.get("tokens_per_step")
        step_note = f", {tok_step:.2f} tok/step" if tok_step is not None else ""
        print(f"[compat_smoke] OK   {prompt[:30]!r}: {len(toks)} tokens, "
              f"{tps:.1f} tok/s{step_note}, repeat_max={run_len} -> {text[:60]!r}")

    print(f"[compat_smoke] {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
