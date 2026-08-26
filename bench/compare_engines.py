"""mlx-lm 素 vs fastmlx vs MTPLX の同一マシン・同一プロンプト速度比較ハーネス。

docs/PLAN.md Phase E2 の準備物。各エンジンを直列 subprocess で起動し、
decode tok/s・TTFT（prefill 時間の代理指標）・生成トークン数を JSON へ書き出す。

  重要: このスクリプトを実行すると実際に 3 エンジンぶんの推論が走る
  （= GPU ベンチ実行）。他プロセスが GPU を使っている間は実行しないこと。
  用意だけして実行しないタスクの一部として作成された。

前提:
- mlx-lm 素・fastmlx は fastmlx 既存の .venv をそのまま使う
  （`uv run --project <repo>` 経由。.venv は汚さない）。
- MTPLX は tools/compare/mtplx-venv/ にインストール済みの独立 venv を使う
  （PyPI 版 mtplx、fastmlx 側の依存とは完全分離）。
- 同一プロンプト集合は bench/spec_bench.py の PROMPTS をそのまま import する
  （import すると fastmlx/mlx-lm も import されるため、このスクリプト自体は
  fastmlx の .venv 内で実行すること。GPU 計算は発生しない、モジュール読み込みのみ）。

2 つの比較モード:
- same-quant: 3 エンジンとも lmstudio-community/Qwen3.8-27B-MLX-4bit を使う。
  MTPLX 側はこの checkpoint に MTP ヘッドが同梱されていないため --no-mtp
  （AR-only）で走らせる。つまりこの行はカーネル/実装の生の速度比較であって
  MTPLX の投機デコード機能そのものの比較ではない。
- recommended: mlx-lm 素とfastmlx は同一 checkpoint のまま
  （fastmlx は自前抽出+量子化した MTP を足すのが「推奨構成」そのものなので
  base checkpoint は変わらない）。MTPLX だけ公式推奨の
  Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed（4bit dynamic quant、
  約20.4GB、実効5.807bit/weight。HF で実在確認済み）+ --mtp に切り替える。

使い方（例。実行は GPU 作業。静音プロトコル後に本番実行すること）:

    uv run --project /Users/ht/dev/fastmlx python bench/compare_engines.py \\
        --modes same-quant recommended \\
        --engines mlx-lm fastmlx mtplx \\
        --prompts all \\
        --max-tokens 512 \\
        --cooldown-sec 60 \\
        --gpu-note "直前 5 分は GPU アイドル、他プロセスなし" \\
        --output bench/results/compare-engines-$(date -u +%Y%m%dT%H%M%SZ).json

まずは --dry-run でコマンド列だけ確認すること（GPU を使わない）:

    uv run --project /Users/ht/dev/fastmlx python bench/compare_engines.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# bench/spec_bench.py は import 時点で mlx.core / mlx_lm / fastmlx.mtp /
# fastmlx.spec を import する（クラス定義のみで推論は走らない）。そのため
# このハーネス自体は fastmlx の .venv 内で実行される前提。
from bench.spec_bench import PROMPTS  # noqa: E402

# ---------------------------------------------------------------------------
# デフォルト値
# ---------------------------------------------------------------------------

# README / fastmlx/cli.py のデフォルトと同一（比較対象の backbone を揃える）
LMSTUDIO_4BIT = "lmstudio-community/Qwen3.8-27B-MLX-4bit"
QWEN38_ORIGINAL = "Qwen/Qwen3.8-27B"
# MTPLX 公式カタログの推奨構成。HF API で実在確認済み
# (2026-08-26: 4-bit dynamic quant, base_model=Qwen/Qwen3.8-27B,
#  KERNEL-INTEL.md の Phase C 初期レシピが参照している実測 KLD 0.022 の対象)。
MTPLX_OPTIMIZED_SPEED = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"

DEFAULT_MTPLX_VENV = REPO_ROOT / "tools" / "compare" / "mtplx-venv"
DEFAULT_MAX_TOKENS = 512  # README の実測表と同じ長さ
DEFAULT_COOLDOWN_SEC = 60
DEFAULT_TIMEOUT_SEC = 1800  # モデルロード + 512 tok 生成の上限目安

ENGINE_CHOICES = ("mlx-lm", "fastmlx", "mtplx")
MODE_CHOICES = ("same-quant", "recommended")


# ---------------------------------------------------------------------------
# 環境記録
# ---------------------------------------------------------------------------


def _run_text(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - ベストエフォート
        return f"<failed: {exc}>"


def collect_environment(gpu_note: str) -> dict[str, Any]:
    """chip・macOS・電源・実行時刻・直前の GPU 使用状況の注記を記録する。"""

    pmset_out = _run_text(["pmset", "-g", "batt"])
    power_source = pmset_out.splitlines()[0] if pmset_out else "unknown"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "chip": _run_text(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "macos_version": _run_text(["sw_vers", "-productVersion"]),
        "arch": platform.machine(),
        "hostname": platform.node(),
        "power_source": power_source,
        "gpu_note": gpu_note,
    }


# ---------------------------------------------------------------------------
# コマンド組み立て
# ---------------------------------------------------------------------------


def build_mlx_lm_command(args: argparse.Namespace, model: str, prompt: str) -> list[str]:
    cmd = [
        "uv", "run", "--project", str(REPO_ROOT),
        "mlx_lm.generate",
        "--model", model,
        "--prompt", prompt,
        "--max-tokens", str(args.max_tokens),
        "--temp", str(args.temp),
        "--seed", str(args.seed),
        "--verbose", "True",
    ]
    if args.no_think:
        cmd += ["--chat-template-config", '{"enable_thinking": false}']
    return cmd


def build_fastmlx_command(args: argparse.Namespace, prompt: str) -> list[str]:
    cmd = [
        "uv", "run", "--project", str(REPO_ROOT),
        "fastmlx",
        "--model", args.fastmlx_model,
        "--original", args.fastmlx_original,
        "--temp", str(args.temp),
        "--max-tokens", str(args.max_tokens),
        "--n-draft", str(args.n_draft),
        "--max-draft", str(args.max_draft),
        "--mtp-bits", str(args.mtp_bits),
        "--prompt", prompt,
    ]
    # fastmlx/cli.py に --seed は無い（temp=0 なら決定的なので実害なし）
    if args.no_think:
        cmd.append("--no-think")
    return cmd


def build_mtplx_command(args: argparse.Namespace, prompt: str, mode: str) -> list[str]:
    mtplx_bin = str(Path(args.mtplx_venv) / "bin" / "mtplx")
    model = args.mtplx_model_same_quant if mode == "same-quant" else args.mtplx_model_recommended
    cmd = [
        mtplx_bin, "ask", prompt,
        "--model", model,
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temp),
        "--seed", str(args.seed),
        "--stats", "--json",
        "--reasoning", "off" if args.no_think else "auto",
        "--yes",  # 非対話実行なので確認プロンプトを自動 yes
    ]
    if mode == "same-quant":
        # lmstudio 4bit には MTPLX が認識する MTP ヘッドが同梱されていない
        # (README/KERNEL-INTEL.md: mlx-lm sanitize が mtp.* を捨てる件と同根)。
        # --no-mtp で AR-only にする = カーネル実装だけの比較になる。
        cmd.append("--no-mtp")
    else:
        cmd += ["--depth", str(args.depth), "--mtp"]
    cmd += list(args.mtplx_extra_args)
    return cmd


def build_mtplx_inspect_command(args: argparse.Namespace, model: str) -> list[str]:
    mtplx_bin = str(Path(args.mtplx_venv) / "bin" / "mtplx")
    return [mtplx_bin, "inspect", model, "--json", "--no-strict-exit-code"]


# ---------------------------------------------------------------------------
# 出力パーサ
# ---------------------------------------------------------------------------

_MLX_LM_PROMPT_RE = re.compile(r"Prompt:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec")
_MLX_LM_GEN_RE = re.compile(r"Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec")
_MLX_LM_MEM_RE = re.compile(r"Peak memory:\s*([\d.]+)\s*GB")


def parse_mlx_lm(stdout: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    m = _MLX_LM_PROMPT_RE.search(stdout)
    if m:
        prompt_tokens, prompt_tps = int(m.group(1)), float(m.group(2))
        result["prompt_tokens"] = prompt_tokens
        result["prompt_tps"] = prompt_tps
        # TTFT の代理指標: prefill (prompt 処理) にかかった時間
        result["ttft_s"] = (prompt_tokens / prompt_tps) if prompt_tps else None
    m = _MLX_LM_GEN_RE.search(stdout)
    if m:
        result["generated_tokens"] = int(m.group(1))
        result["decode_tok_s"] = float(m.group(2))
    m = _MLX_LM_MEM_RE.search(stdout)
    if m:
        result["peak_memory_gb"] = float(m.group(1))
    if not result:
        result["parse_error"] = "mlx_lm.generate の統計行が見つからない（--verbose False?）"
    return result


_FASTMLX_LOAD_RE = re.compile(r"\[fastmlx\] loaded in ([\d.]+)s")
_FASTMLX_STATS_RE = re.compile(
    r"\[([\d.]+) tok/s \| ([\d.]+) tok/step \| ttft ([\d.]+)s \| "
    r"prefill 再利用 (\d+) / 新規 (\d+)\]"
)


def parse_fastmlx(stdout: str, wall_time_s: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    m = _FASTMLX_LOAD_RE.search(stdout)
    load_s = float(m.group(1)) if m else None
    if load_s is not None:
        result["load_s"] = load_s
    m = _FASTMLX_STATS_RE.search(stdout)
    if m:
        decode_tps = float(m.group(1))
        ttft_s = float(m.group(3))
        result["decode_tok_s"] = decode_tps
        result["tokens_per_step"] = float(m.group(2))
        result["ttft_s"] = ttft_s
        result["prefill_reused"] = int(m.group(4))
        result["prefill_new"] = int(m.group(5))
        # fastmlx/cli.py は生成トークン総数を出力しないため、壁時計から概算する。
        # load 時間・ttft を差し引いた残りを decode_tok_s で割り戻す粗い推定値。
        # 既存ファイル (fastmlx/cli.py) は変更不可のためこの近似で代用する。
        if load_s is not None:
            decode_elapsed = max(wall_time_s - load_s - ttft_s, 0.0)
            result["generated_tokens_estimate"] = round(decode_tps * decode_elapsed)
            result["generated_tokens_estimate_basis"] = (
                "(wall_time_s - load_s - ttft_s) * decode_tok_s による概算。"
                "fastmlx CLI が生成トークン数を直接出力しないための代替"
            )
    else:
        result["parse_error"] = "fastmlx CLI の統計行が見つからない"
    return result


def parse_mtplx(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"parse_error": "JSON ブロックが見つからない", "raw_stdout_tail": stdout[-2000:]}
    try:
        payload = json.loads(stdout[start : end + 1])
    except json.JSONDecodeError as exc:
        return {"parse_error": f"JSON decode failed: {exc}", "raw_stdout_tail": stdout[-2000:]}
    if "error" in payload:
        return {"engine_error": payload, "raw_payload": payload}
    stats = payload.get("stats", {})
    return {
        "generated_tokens": stats.get("generated_tokens"),
        "decode_tok_s": stats.get("decode_tok_s"),
        # MTPLX の JSON に明示的な TTFT フィールドは無い。prompt_eval_time_s
        # (prefill 時間) を代理指標として扱う。他 2 エンジンと同じ定義。
        "ttft_s": stats.get("prompt_eval_time_s"),
        "generation_mode": stats.get("generation_mode"),
        "mtp_depth": stats.get("mtp_depth"),
        "verify_calls": stats.get("verify_calls"),
        "verify_time_s": stats.get("verify_time_s"),
        "draft_time_s": stats.get("draft_time_s"),
        "accepted_by_depth": stats.get("accepted_by_depth"),
        "drafted_by_depth": stats.get("drafted_by_depth"),
        "raw_payload": payload,
    }


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    mode: str
    engine: str
    prompt_key: str
    command: list[str]
    dry_run: bool
    returncode: int | None = None
    wall_time_s: float | None = None
    parsed: dict[str, Any] = field(default_factory=dict)
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "engine": self.engine,
            "prompt_key": self.prompt_key,
            "command": self.command,
            "dry_run": self.dry_run,
            "returncode": self.returncode,
            "wall_time_s": self.wall_time_s,
            "parsed": self.parsed,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "error": self.error,
        }


def run_subprocess(cmd: list[str], timeout: float) -> tuple[int, str, str, float]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout
        )
        wall = time.perf_counter() - t0
        return proc.returncode, proc.stdout, proc.stderr, wall
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(
            "utf-8", "replace"
        )
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(
            "utf-8", "replace"
        )
        return -1, stdout, stderr, wall


def build_command(args: argparse.Namespace, engine: str, mode: str, prompt: str) -> list[str]:
    if engine == "mlx-lm":
        return build_mlx_lm_command(args, args.mlx_lm_model, prompt)
    if engine == "fastmlx":
        return build_fastmlx_command(args, prompt)
    if engine == "mtplx":
        return build_mtplx_command(args, prompt, mode)
    raise ValueError(f"unknown engine: {engine}")


def parse_output(engine: str, stdout: str, wall_time_s: float) -> dict[str, Any]:
    if engine == "mlx-lm":
        return parse_mlx_lm(stdout)
    if engine == "fastmlx":
        return parse_fastmlx(stdout, wall_time_s)
    if engine == "mtplx":
        return parse_mtplx(stdout)
    raise ValueError(f"unknown engine: {engine}")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    environment = collect_environment(args.gpu_note)
    prompt_keys = list(PROMPTS.keys()) if args.prompts == ["all"] else args.prompts
    for key in prompt_keys:
        if key not in PROMPTS:
            raise SystemExit(f"unknown prompt key: {key} (choices: {sorted(PROMPTS)})")

    inspect_records: list[dict[str, Any]] = []
    if "mtplx" in args.engines and args.mtplx_inspect:
        models_to_inspect = set()
        if "same-quant" in args.modes:
            models_to_inspect.add(args.mtplx_model_same_quant)
        if "recommended" in args.modes:
            models_to_inspect.add(args.mtplx_model_recommended)
        for model in sorted(models_to_inspect):
            cmd = build_mtplx_inspect_command(args, model)
            record: dict[str, Any] = {"model": model, "command": cmd}
            if args.dry_run:
                record["dry_run"] = True
            else:
                rc, out, err, wall = run_subprocess(cmd, timeout=120)
                record.update(
                    returncode=rc,
                    wall_time_s=wall,
                    stdout_tail=out[-4000:],
                    stderr_tail=err[-2000:],
                )
            inspect_records.append(record)

    batches = [(mode, engine) for mode in args.modes for engine in args.engines]
    records: list[RunRecord] = []
    for batch_index, (mode, engine) in enumerate(batches):
        for prompt_key in prompt_keys:
            prompt = PROMPTS[prompt_key]
            cmd = build_command(args, engine, mode, prompt)
            rec = RunRecord(
                mode=mode, engine=engine, prompt_key=prompt_key, command=cmd, dry_run=args.dry_run
            )
            if args.dry_run:
                print(f"[dry-run] {mode}/{engine}/{prompt_key}: {' '.join(cmd)}")
            else:
                print(f"[run] {mode}/{engine}/{prompt_key} ...", flush=True)
                rc, out, err, wall = run_subprocess(cmd, timeout=args.timeout_sec)
                rec.returncode = rc
                rec.wall_time_s = wall
                rec.stdout_tail = out[-8000:]
                rec.stderr_tail = err[-4000:]
                if rc != 0:
                    rec.error = f"exit code {rc}"
                else:
                    try:
                        rec.parsed = parse_output(engine, out, wall)
                    except Exception as exc:  # pragma: no cover - パーサの想定外入力
                        rec.error = f"parse failed: {exc}"
                decode = rec.parsed.get("decode_tok_s")
                ttft = rec.parsed.get("ttft_s")
                print(
                    f"[done] {mode}/{engine}/{prompt_key} "
                    f"decode_tok_s={decode} ttft_s={ttft} wall_time_s={wall:.1f}",
                    flush=True,
                )
            records.append(rec)

        is_last_batch = batch_index == len(batches) - 1
        if not args.dry_run and not is_last_batch and args.cooldown_sec > 0:
            print(f"[cooldown] {args.cooldown_sec}s ...", flush=True)
            time.sleep(args.cooldown_sec)

    return {
        "environment": environment,
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k != "func"
        },
        "mtplx_inspect": inspect_records,
        "runs": [r.to_dict() for r in records],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "mlx-lm 素 / fastmlx / MTPLX の同一マシン・同一プロンプト速度比較。"
            "実行すると GPU 推論が走る。--dry-run でまずコマンド列だけ確認すること。"
        )
    )
    ap.add_argument("--modes", nargs="+", choices=MODE_CHOICES, default=list(MODE_CHOICES))
    ap.add_argument("--engines", nargs="+", choices=ENGINE_CHOICES, default=list(ENGINE_CHOICES))
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=["all"],
        help=f"bench/spec_bench.py の PROMPTS キー、または 'all'（現状: {sorted(PROMPTS)}）",
    )
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--temp", type=float, default=0.0, help="0 で greedy（3 エンジン共通・決定的）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-think",
        action="store_true",
        default=True,
        help="Qwen3.8 の thinking モードを 3 エンジンとも切る（README 推奨。decode 比較のノイズを減らす）",
    )
    ap.add_argument("--think", dest="no_think", action="store_false", help="--no-think を打ち消す")
    ap.add_argument("--cooldown-sec", type=int, default=DEFAULT_COOLDOWN_SEC)
    ap.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    ap.add_argument("--dry-run", action="store_true", help="コマンドを組み立てて表示するだけ。実行しない")

    ap.add_argument("--mlx-lm-model", default=LMSTUDIO_4BIT)
    ap.add_argument("--fastmlx-model", default=LMSTUDIO_4BIT)
    ap.add_argument("--fastmlx-original", default=QWEN38_ORIGINAL)
    ap.add_argument("--n-draft", type=int, default=3)
    ap.add_argument("--max-draft", type=int, default=8)
    ap.add_argument("--mtp-bits", type=int, default=4)

    ap.add_argument("--mtplx-venv", type=Path, default=DEFAULT_MTPLX_VENV)
    ap.add_argument("--mtplx-model-same-quant", default=LMSTUDIO_4BIT)
    ap.add_argument("--mtplx-model-recommended", default=MTPLX_OPTIMIZED_SPEED)
    ap.add_argument("--depth", type=int, default=3, help="MTPLX recommended 行の MTP depth")
    ap.add_argument(
        "--mtplx-inspect",
        action="store_true",
        default=True,
        help="各 MTPLX モデルに対し先に `mtplx inspect` を走らせ互換性情報を記録する（GPU 不使用）",
    )
    ap.add_argument("--no-mtplx-inspect", dest="mtplx_inspect", action="store_false")
    ap.add_argument(
        "--mtplx-extra-args",
        nargs="*",
        default=[],
        help="mtplx ask にそのまま追加する引数（例: --unsafe-force-unverified）",
    )

    ap.add_argument("--gpu-note", required=True, help="直前の GPU 使用状況の注記（例: '直前5分アイドル'）")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力 JSON パス。省略時は bench/results/compare-engines-<UTC時刻>.json",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = REPO_ROOT / "bench" / "results" / f"compare-engines-{stamp}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    result = run_all(args)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[compare_engines] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
