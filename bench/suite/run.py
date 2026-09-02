#!/usr/bin/env python3
"""公開ベンチマークの実行ドライバ。

やること (`docs/research/BENCH-DESIGN-2026-09.md` の (d) 節が仕様):

  1. シナリオ x 文脈 x エンジンの全組を「ブロック」に展開する
     (1 ブロック = 1 回のサーバー起動。文脈ごとに交互に起動する、というのは
     「両サーバーを同時にメモリへ載せない」制約から来る)
  2. ブロックの実行順をシャッフルする (乱数種はログに残す。再現できる
     ランダム化であって、無作為でごまかしているわけではない)
  3. ブロックごとに: 起動 → 2 段ウォームアップ → `reps` 回の反復測定 →
     停止 → 残留プロセス確認 → 冷却 (既定 180 秒)
  4. 生ログを JSON で `bench/results/suite/<run-id>/` に書き出す

**GPU を使う工程 (サーバー起動・推論) は `--dry-run` では一切呼ばない。**
`--dry-run` は「どの順で何を起動するか」と「所要時間の見積もり」だけを
表示する (モデルもトークナイザも読まない)。

    uv run python bench/suite/run.py --dry-run
        # 既定計画: Flash-Next, mlxturbo vs mlx-serve, point x 文脈 6 点,
        # 各 3 反復、ランダム順、冷却 180 秒

実行 (GPU を使う。このスクリプト自身は対戦の判定をしない — 判定は
CLAUDE.md の「計測の判定と commit は親が行う」を読む人間が行う):

    tools/biglock.sh uv run python bench/suite/run.py
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "bench", REPO_ROOT / "bench" / "suite",
           REPO_ROOT / "tools"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from engines import (  # noqa: E402
    ENGINE_REGISTRY, EngineAdapter, LlamaCppAdapter, MlxLmAdapter,
    MlxServeAdapter, MlxturboAdapter, OMlxAdapter, Server,
    install_term_handler, stream_with_usage,
)
from scenarios import (  # noqa: E402
    DEFAULT_CTXS, SCENARIO_NAMES, Scenario, build_scenario,
)

# ── 既定プリセット: Flash-Next, mlxturbo vs mlx-serve ────────────────────
# 値の根拠は docs/NEXT-SESSION-PROMPT.md の再現コマンドと
# bench/results/logs/turbo-0902c.log (n-gram/MTP はサイドカー自動発見な
# ので --ngram / --mtp は不要、というログ上の事実)。
DEFAULT_MLXTURBO = dict(model="~/models/ddalcu-mlxlm", ngram=None, mtp=None)
DEFAULT_MLXSERVE = dict(
    binary="~/dev/mlx-serve/zig-out/bin/mlx-serve",
    model="~/models/ddalcu-flashnext-serve-4bit", mtp=True)

DEFAULT_HOST = "127.0.0.1"
ENGINE_PORTS = {"mlxturbo": 8151, "mlx-serve": 8161,
                "omlx": 8171, "llama.cpp": 8181, "mlx-lm": 8191}


def default_engines() -> dict[str, EngineAdapter]:
    """既定の設定を持つ全アダプタ。stub (omlx/llama.cpp/mlx-lm) も含める —
    「未登録エンジン」(CLI のタイプミス) と「まだ実装していないエンジン」
    (`is_available()` が False で理由を名乗る) を区別するため。
    """
    return {
        "mlxturbo": MlxturboAdapter(**DEFAULT_MLXTURBO),
        "mlx-serve": MlxServeAdapter(**DEFAULT_MLXSERVE),
        "omlx": OMlxAdapter(),
        "llama.cpp": LlamaCppAdapter(),
        "mlx-lm": MlxLmAdapter(),
    }


# ── 所要時間の見積もり (dry-run 表示専用。実測ではない) ──────────────────
# 2026-09-02 の「直したハーネスで測った現在地」(SESSION-2026-09-02-CATCHUP.md)
# を較正点にした線形補間。文脈 25000/32000 は 17000-50000 間の補間で、
# 実測ではなく見積もりであることを常に明記する。
_CALIB = {
    "mlxturbo": {
        "cold_ttft": {0: 0.49, 4000: 8.28, 17000: 37.6, 50000: 163.0},
        "decode_tps": {0: 47.5, 4000: 48.8, 17000: 41.0, 50000: 34.4},
        "warm_ttft": {0: 0.45, 4000: 0.46, 17000: 0.51, 50000: 1.33},
        "boot_s": 180.0,
    },
    "mlx-serve": {
        "cold_ttft": {0: 0.17, 4000: 5.77, 17000: 29.2, 50000: 108.0},
        "decode_tps": {0: 53.7, 4000: 55.3, 17000: 56.0, 50000: 30.8},
        "warm_ttft": {0: 0.72, 4000: 0.87, 17000: 0.92, 50000: 22.8},
        "boot_s": 180.0,
    },
}


def _interp(table: dict[int, float], x: int) -> float:
    xs = sorted(table)
    if x <= xs[0]:
        return table[xs[0]]
    if x >= xs[-1]:
        lo, hi = xs[-2], xs[-1]
    else:
        lo = max(v for v in xs if v <= x)
        hi = min(v for v in xs if v >= x)
        if lo == hi:
            return table[lo]
    t = (x - lo) / (hi - lo)
    return table[lo] + t * (table[hi] - table[lo])


def estimate_block_seconds(engine_kind: str, ctx: int | None, reps: int,
                           tokens: int, cooldown: float) -> dict:
    """1 ブロック (起動→測定→冷却) の推定秒数。較正表が無いエンジン
    (stub) は `None` を返す — 「わからない」を 0 で埋めない。
    """
    calib = _CALIB.get(engine_kind)
    if calib is None or ctx is None:
        # point 以外のシナリオ (agent/code-edit/rag) はターン構成が
        # 較正点と違うので、いまは大まかな下限として ctx=4000 相当を使う
        # (較正のやり直しは実測が要る。過大な精度を主張しない)。
        c = 4000
    else:
        c = ctx
    if calib is None:
        return {"boot_s": None, "measure_s": None, "cooldown_s": cooldown,
                "total_s": None, "note": f"{engine_kind} の較正データが無い"}
    boot_s = calib["boot_s"]
    cold = _interp(calib["cold_ttft"], c)
    tps = _interp(calib["decode_tps"], c)
    warm = _interp(calib["warm_ttft"], c)
    decode_s = tokens / tps if tps > 0 else 0.0
    per_rep = cold + decode_s + warm
    warmup_s = 20.0  # 2 段ウォームアップ (短文 + 長文プロンプト 1 回ずつ)
    measure_s = per_rep * reps
    total = boot_s + warmup_s + measure_s + cooldown
    return {"boot_s": boot_s, "warmup_s": warmup_s, "measure_s": measure_s,
            "cooldown_s": cooldown, "total_s": total,
            "note": f"ctx={c} 較正点からの補間 (実測ではない)"}


# ── ブロック定義とスケジューリング ────────────────────────────────────

@dataclass
class Block:
    """1 回のサーバー起動で完結する測定単位。"""

    index: int
    scenario_name: str
    scenario_kwargs: dict
    engine_kind: str
    ctx: int | None
    reps: int
    tokens: int
    block_seed: int

    def label(self) -> str:
        c = f"ctx={self.ctx}" if self.ctx is not None else "ctx=-"
        return f"{self.scenario_name:12s} {c:10s} {self.engine_kind}"


def build_schedule(scenario_specs: list[tuple[str, int | None, dict]],
                   engine_kinds: list[str], reps: int, tokens: int,
                   seed: int) -> list[Block]:
    """シナリオ x 文脈の各点について、エンジンの起動順をランダム化して
    ブロック列を作る。

    「文脈ブロックごとに交互に起動する」の実装: シナリオ x 文脈の順序
    そのものをシャッフルし、かつ各点でどちらのエンジンを先に起動するかも
    毎回コインを振る。これにより「エンジン A を全部先に片付けてから B」
    という、系統的な熱ドリフトの影響を最も受けやすい順序を避ける。
    """
    rng = random.Random(seed)
    order = list(scenario_specs)
    rng.shuffle(order)
    blocks: list[Block] = []
    idx = 0
    for name, ctx, kwargs in order:
        pair = list(engine_kinds)
        rng.shuffle(pair)
        for eng in pair:
            blocks.append(Block(
                index=idx, scenario_name=name, scenario_kwargs=kwargs,
                engine_kind=eng, ctx=ctx, reps=reps, tokens=tokens,
                block_seed=rng.randrange(1 << 30)))
            idx += 1
    return blocks


def resolve_scenario_specs(scenario_names: list[str], ctxs: list[int],
                           tokens: int) -> list[tuple[str, int | None, dict]]:
    """CLI の `--scenarios` を `(名前, ctx, build_scenario への kwargs)` の
    列に展開する。`point` だけ文脈ごとに複数点へ広がる。
    """
    specs: list[tuple[str, int | None, dict]] = []
    for name in scenario_names:
        if name == "point":
            for c in ctxs:
                specs.append(("point", c, {"ctx": c, "tokens": tokens}))
        elif name == "rag":
            specs.append(("rag", None, {"mode": "fresh", "tokens": tokens}))
            specs.append(("rag", None, {"mode": "shared", "tokens": tokens}))
        elif name == "parallel":
            # 既定計画には含めない (scenarios.py の enabled_by_default 参照)。
            # 明示指定されたときだけ、定義の確認用に 1 点だけ載せる。
            specs.append(("parallel", None, {}))
        else:
            specs.append((name, None, {"tokens": tokens}))
    return specs


# ── 残留プロセス検査 (実行時のみ。dry-run では絶対に呼ばない) ────────────

def check_residual_processes(patterns: list[str]) -> list[str]:
    """`pkill` 対象になりうるプロセスと `vm.swapusage` を確認する。

    NEXT-SESSION-PROMPT.md「守ること」2. の実装: サーバー停止後に
    Metal のメモリを握ったまま残るプロセスが無いか、スワップが
    異常に膨らんでいないかを見る。警告文字列のリストを返すだけで、
    実際に kill するかは呼び出し側 (run.py の実行ループ) が判断する。
    """
    warnings: list[str] = []
    for pat in patterns:
        try:
            out = subprocess.run(["pgrep", "-fl", pat], capture_output=True,
                                 text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                warnings.append(f"残留プロセスの疑い ({pat}):\n{out.stdout.strip()}")
        except (OSError, subprocess.TimeoutExpired) as e:
            warnings.append(f"pgrep 失敗 ({pat}): {e}")
    try:
        out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                             text=True, timeout=10)
        if out.returncode == 0:
            warnings.append(f"[参考] {out.stdout.strip()}")
    except (OSError, subprocess.TimeoutExpired) as e:
        warnings.append(f"vm.swapusage 取得失敗: {e}")
    return warnings


# ── dry-run 表示 ─────────────────────────────────────────────────────

def print_plan(blocks: list[Block], seed: int, cooldown: float,
              out_dir: Path) -> dict:
    """コマンド列・順序・推定所要時間を表示する。**副作用は無い**
    (サーバー起動・HTTP・ファイル書き込みはこの関数の外)。
    """
    engines = default_engines()
    print(f"=== ベンチ計画 (dry-run, seed={seed}) ===")
    print(f"ブロック数: {len(blocks)}  冷却: {cooldown:.0f}s  出力先: {out_dir}\n")

    total = 0.0
    unknown = 0
    rows = []
    for b in blocks:
        adapter = engines.get(b.engine_kind)
        if adapter is None:
            avail, reason = False, f"未登録エンジン (候補: {list(ENGINE_REGISTRY)})"
        else:
            avail = adapter.is_available()
            reason = adapter.unavailable_reason()
        est = estimate_block_seconds(b.engine_kind, b.ctx, b.reps, b.tokens,
                                     cooldown)
        if est["total_s"] is None:
            unknown += 1
        else:
            total += est["total_s"]
        if adapter is None:
            argv = ["<未登録エンジン>"]
        elif avail:
            argv = adapter.build_argv(DEFAULT_HOST,
                                      ENGINE_PORTS.get(b.engine_kind, 8200))
        else:
            # stub アダプタ (未実装) は build_argv 自体が NotImplementedError を
            # 投げる設計 (engines.py の _StubAdapter 参照) — dry-run はそれを
            # クラッシュではなく SKIP として見せる。
            argv = [f"<未実装: {reason}>"]
        avail_mark = "OK" if avail else f"SKIP ({reason})"
        scenario = build_scenario(b.scenario_name, **b.scenario_kwargs)
        turn_labels = [t.label for t in scenario.turns]
        print(f"[{b.index:03d}] {b.label():34s} reps={b.reps} "
              f"tokens={b.tokens} 見積={_fmt_s(est['total_s'])}  [{avail_mark}]")
        print(f"       argv: {' '.join(argv)}")
        print(f"       turns ({len(turn_labels)}): {turn_labels}")
        rows.append(dict(index=b.index, scenario=b.scenario_name, ctx=b.ctx,
                         engine=b.engine_kind, reps=b.reps, tokens=b.tokens,
                         argv=argv, turns=turn_labels, available=avail,
                         unavailable_reason=reason, estimate=est))

    print(f"\n合計推定 (較正できたブロックのみ): {_fmt_s(total)}"
         f"  (較正データ無しのブロック: {unknown} 件、時間見積もりから除外)")
    print("見積もりは 2026-09-02 の実測 (SESSION-2026-09-02-CATCHUP.md) からの"
         " 線形補間であり、実測ではない。熱・残留プロセス・機体差で外れる。")
    return dict(seed=seed, cooldown=cooldown, out_dir=str(out_dir),
               total_estimate_s=total, unknown_blocks=unknown, blocks=rows)


def _fmt_s(s: float | None) -> str:
    if s is None:
        return "?"
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


# ── 実行 (GPU を使う。dry-run では呼ばれない) ────────────────────────

def run_block(block: Block, adapter: EngineAdapter, out_dir: Path,
             thinking: str, server_log_dir: Path) -> dict:
    """1 ブロックを実際に走らせる。起動 → ウォームアップ → `reps` 回の
    測定 → 停止 → 残留プロセス確認、まで面倒を見る。

    冷 TTFT の信頼できる値は **各ブロックの rep=0** (フレッシュ起動直後の
    最初のリクエスト) だけである。rep>=1 はサーバーは起動済みだが GPU が
    温まっていくので、絶対値としては rep=0 より当てにならない
    (`docs/research/BENCH-DESIGN-2026-09.md` (i) 節)。rep>=1 は分散の確認と
    ばらつきの目安に使う。
    """
    from transformers import AutoTokenizer  # 遅延 import (dry-run を汚さない)

    thinking_extra = {
        "off": {"reasoning_effort": "none"},
        "on": {"reasoning_effort": "medium"},
        "default": None,
    }[thinking]

    scenario = build_scenario(block.scenario_name, **block.scenario_kwargs)
    port = ENGINE_PORTS.get(block.engine_kind, 8200)
    log_path = server_log_dir / f"{block.index:03d}-{block.engine_kind}.log"

    reps_out = []
    rng = random.Random(block.block_seed)
    tok = AutoTokenizer.from_pretrained(
        _tokenizer_source(adapter))

    with Server(adapter.name, adapter.build_argv(DEFAULT_HOST, port), port,
               log_path=str(log_path)):
        mid = adapter.model_id(port)
        # ウォームアップ 1 段目 (短文、カーネルの初回コンパイル)
        stream_with_usage(port, [{"role": "user", "content": "こんにちは。"}],
                          8, mid, extra_body=thinking_extra)
        for r in range(block.reps):
            from scenarios import MaterializeCtx, PoolCursor
            pool = PoolCursor(tok)
            mctx = MaterializeCtx(tok=tok, pool=pool, rng=rng,
                                  target_ctx=block.ctx or 0)
            history: list[dict] = []
            turn_results = []
            for turn in scenario.turns:
                content = turn.content_fn(mctx)
                if turn.reset_history:
                    history = []
                messages = history + [{"role": "user", "content": content}]
                ttft, dec_s, n, reply, usage = stream_with_usage(
                    port, messages, turn.max_tokens, mid,
                    extra_body=thinking_extra)
                turn_results.append(dict(
                    label=turn.label, ttft_s=ttft, decode_s=dec_s,
                    n_tokens=n, cached_tokens=(usage or {})
                    .get("prompt_tokens_details", {}).get("cached_tokens"),
                    reply_head=reply[:160]))
                history = messages + [{"role": "assistant", "content": reply}]
            reps_out.append(dict(rep=r, is_fresh_boot=(r == 0),
                                 turns=turn_results))
        warnings = check_residual_processes(
            [adapter.name, "mlxturbo.server", "mlx-serve --serve"])

    return dict(block=asdict(block), scenario=scenario.name,
               reps=reps_out, residual_warnings=warnings,
               server_log=str(log_path))


def _tokenizer_source(adapter: EngineAdapter) -> str:
    import os
    if isinstance(adapter, MlxturboAdapter):
        return os.path.expanduser(adapter.model)
    if isinstance(adapter, MlxServeAdapter):
        return os.path.expanduser(adapter.model)
    raise NotImplementedError(
        "このアダプタ種別のトークナイザ取得元が未定義 (engines.py を見て追加すること)")


def main() -> int:
    install_term_handler()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="point",
                    help=f"カンマ区切り。候補: {','.join(SCENARIO_NAMES)}"
                         " (既定 point のみ)")
    ap.add_argument("--engines", default="mlxturbo,mlx-serve",
                    help=f"カンマ区切り。候補: {','.join(ENGINE_REGISTRY)}")
    ap.add_argument("--ctxs", default=",".join(str(c) for c in DEFAULT_CTXS),
                    help="point シナリオの文脈トークン数リスト")
    ap.add_argument("--tokens", type=int, default=512,
                    help="生成トークン数の既定値 (CLAUDE.md の"
                         " 「tok/step は複数プロンプト x 512 の平均」に合わせた既定)")
    ap.add_argument("--reps", type=int, default=3,
                    help="ブロックあたりの反復 (3 回以上、CLAUDE.md 系の作法)")
    ap.add_argument("--cooldown", type=float, default=180.0,
                    help="ブロック間の冷却秒数 (既定 3 分)")
    ap.add_argument("--seed", type=int, default=None,
                    help="ブロック順のシャッフルに使う乱数種。省略時は生成して表示する"
                         " (dry-run を再現したいときはここに渡し直す)")
    ap.add_argument("--out-dir", default=None,
                    help="既定: bench/results/suite/<timestamp>/")
    ap.add_argument("--thinking", choices=("off", "on", "default"), default="off")
    ap.add_argument("--dry-run", action="store_true",
                    help="コマンド列・順序・所要時間見積もりを表示するだけ。"
                         " サーバー起動・HTTP・モデル読み込みを一切行わない")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 30)
    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for s in scenario_names:
        if s not in SCENARIO_NAMES:
            print(f"未知のシナリオ: {s} (候補: {SCENARIO_NAMES})", file=sys.stderr)
            return 2
    engine_kinds = [e.strip() for e in args.engines.split(",") if e.strip()]
    for e in engine_kinds:
        if e not in ENGINE_REGISTRY:
            print(f"未知のエンジン: {e} (候補: {list(ENGINE_REGISTRY)})", file=sys.stderr)
            return 2
    ctxs = [int(c) for c in args.ctxs.split(",")]

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "bench" / "results" / "suite" / time.strftime("%Y%m%d-%H%M%S"))

    specs = resolve_scenario_specs(scenario_names, ctxs, args.tokens)
    blocks = build_schedule(specs, engine_kinds, args.reps, args.tokens, seed)

    if args.dry_run:
        plan = print_plan(blocks, seed, args.cooldown, out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2))
        print(f"\n計画を書き出した (dry-run。何も起動していない): {out_dir / 'plan.json'}")
        return 0

    # ── ここから先は GPU を使う実行 (このエージェントは呼ばない) ──────
    engines = default_engines()
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log_dir = out_dir / "logs"
    server_log_dir.mkdir(exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(
        print_plan(blocks, seed, args.cooldown, out_dir),
        ensure_ascii=False, indent=2))

    raw_path = out_dir / "raw.json"
    results = []
    for i, block in enumerate(blocks):
        adapter = engines.get(block.engine_kind)
        if adapter is None or not adapter.is_available():
            reason = adapter.unavailable_reason() if adapter else "未登録エンジン"
            print(f"[{block.index:03d}] SKIP {block.label()}: {reason}")
            continue
        print(f"[{block.index:03d}] 実行: {block.label()}", flush=True)
        result = run_block(block, adapter, out_dir, args.thinking, server_log_dir)
        results.append(result)
        raw_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        if result["residual_warnings"]:
            for w in result["residual_warnings"]:
                print(f"  [警告] {w}")
        if i < len(blocks) - 1:
            print(f"  冷却 {args.cooldown:.0f}s...", flush=True)
            time.sleep(args.cooldown)

    print(f"\n書き出し: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
