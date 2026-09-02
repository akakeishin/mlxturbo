"""verify forward の幅 S ごとの費用を測る (mlx-serve の
``MLX_SERVE_DECODE_FWD_UBENCH`` と同じ量・同じ定義)。

mlx-serve は「幅 S の verify forward 1 回の ms」を出す (M4 Max: S=1 16.0 /
S=2 22.4 / S=4 30.6 ms)。こちらは同じ量 (``spec_flash._staged_forward`` を
``capture()`` の中で呼び ``mx.eval(lg)`` までの壁時計) を同じ土俵で測る。

## 作法 (CLAUDE.md の計測の作法に従う)

- 1 プロセス内で全 S を測る (プロセスを分けた比較は熱・キャッシュ状態で
  数 % ずれる)。
- 各 S で最初の 3 回は捨てる (JIT・キャッシュの温まりの影響を除く。
  ``decode_ab.py`` の「プロセスの最初の 1 本は捨てる」と同じ理屈)。
- S の訪問順は昇順・降順を交互にした回文掃引 (``--rounds`` 既定 2:
  1,2,3,4,4,3,2,1) にする。線形の熱ドリフトを特定の S に偏らせないため
  (``decode_ab.py`` の A→B→B→A と同じ理屈を S 個の値に一般化したもの)。
- 反復ごとにキャッシュを退避・復元し、毎回まったく同じ状態から forward を
  始める (``decode_ab.py`` の ``_snapshot``/``_restore`` を再利用)。

## rollback の keep=0 について

`spec_flash.rollback(..., keep=0, total=S)` で verify forward を丸ごと
巻き戻すことを検討したが、**keep=0 は安全ではないので使っていない**。
``arch.rollback_recurrent`` の GDN 再帰状態の巻き戻しが
``c[rl.state] = states_all[:, keep - 1] if keep > 0 else None`` になっており、
keep=0 のときは forward 前の実際の状態を復元せず ``None`` (= 次回ゼロ状態
扱い) に落としてしまう。本番でも ``rollback`` は keep=0 で呼ばれることが
ない (``_verify`` は必ず 1 個以上のトークンを返す)。そのため、幅だけを測る
forward-only 計測は反復のたびに ``decode_ab._snapshot``/``_restore`` で
キャッシュそのものを丸ごと退避・復元して戻す。``--with-draft`` のラウンド
計測は本番と同じ ``eng._verify`` で実際の keep (常に 1 以上) を出してから
``rollback`` を呼ぶので、この keep=0 の問題には触れない。

## 使用例

    tools/biglock.sh .venv/bin/python tools/verify_width_cost.py \\
        --model ~/models/ddalcu-mlxlm --ngram ~/models/ddalcu-ngram-sep \\
        --ctx 17000 --widths 1,2,3,4 --with-draft
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# decode_ab.py と同じキャッシュ退避・復元・prefill 1 回化のヘルパーを
# そのまま使う (写しを作ると挙動がずれる)。
from decode_ab import SHORT_PROMPTS, _restore, prefill_once  # noqa: E402


def build_runner(args):
    """decode_ab.py の main() 前半と同じ手順で model/mtp/eng を組む
    (``mlxturbo.runner.build_runner`` は server 用の ``Runner`` 抽象を返す
    別物なのでここでは使わない -- decode_ab.py が直接組んでいるのと同じ
    ``FlashSpecEngine`` を素で得る)。"""
    import mlx.core as mx
    from mlx_lm import load

    import mlxturbo  # noqa: F401
    from mlxturbo import mtp_flash, spec_flash

    if args.ngram:
        os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")

    model_path = os.path.expanduser(args.model)
    model, tok = load(model_path)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, os.path.expanduser(args.ngram))

    # 出荷経路と同じ融合を当てる (decode_ab.py と同じ理由: gather のソートなど
    # 既定 knob が入らないまま測ると閾値がずれる)。
    from mlxturbo.runner import enable_default_fusions

    enable_default_fusions(model, log_prefix="[verify_width_cost]")

    mtp_path = args.mtp or os.path.join(model_path, "mtp.safetensors")
    q = {"group_size": 64, "bits": args.mtp_bits} if args.mtp_bits else None
    mtp = mtp_flash.load_flash_mtp(os.path.expanduser(mtp_path), model.args.text, quantize=q)
    mx.eval(mtp.parameters())

    eng = spec_flash.FlashSpecEngine(model, mtp)
    eos = tok.eos_token_ids if hasattr(tok, "eos_token_ids") else ()
    eos_ids = tuple(eos) if eos else ()
    return eng, model, tok, eos_ids


def build_prompt_ids(tok, ctx: int):
    """--ctx 0 は短プロンプト (decode_ab.SHORT_PROMPTS[0])、それ以外は
    tools/_bench_text.py の長文脈窓 (decode_ab.py と同じ素材)。"""
    import mlx.core as mx
    from _bench_text import long_prompts

    if ctx <= 0:
        text = SHORT_PROMPTS[0]
    else:
        text = long_prompts(tok, ctx, ["この文書について 1 段落で要約してください。"])[0]
    ids = mx.array(
        tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
    )[None]
    return ids


def build_pair(eng, resume, max_width: int):
    """prefill 境界の resume から、実際の (貪欲な) MTP ドラフト列を最大
    max_width 個だけ 1 度だけ引く。以降は全ての S・全ての反復で、この同じ
    列の先頭 S 個を使い回す (テキスト運で forward の中身が反復ごとに
    変わるのを避けるため)。ターゲットモデルの caches には触れない
    (`_draft_chain` が触るのは MTP 自身の使い捨てキャッシュだけ)。
    """
    import mlx.core as mx
    from mlxturbo.spec_flash import restore_mtp_cache

    logits_tail, hyper_tail0, mtp_snap = resume
    cur = eng._sample(logits_tail, 0.0)
    mtp_cache = restore_mtp_cache(mtp_snap)
    depth = max(0, max_width - 1)
    drafts = eng._draft_chain(cur, hyper_tail0, mtp_cache, depth) if depth else []
    pair = mx.concatenate([cur] + drafts, axis=1)
    mx.eval(pair)
    return pair, cur


def summarize(times_s: list[float]) -> dict:
    """最初の 3 回を捨てて中央値・最小値を出す (ms)。"""
    warm = times_s[3:] if len(times_s) > 3 else times_s
    ms = [t * 1000.0 for t in warm]
    return {"n": len(ms), "median_ms": statistics.median(ms), "min_ms": min(ms)}


def sweep_order(widths: list[int], rounds: int) -> list[list[int]]:
    """回文掃引の各周を返す (奇数番目の周は降順)。例: widths=[1,2,3,4],
    rounds=2 なら [[1,2,3,4], [4,3,2,1]] -- 通して並べると 1,2,3,4,4,3,2,1。"""
    out = []
    for i in range(rounds):
        out.append(list(widths) if i % 2 == 0 else list(reversed(widths)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--mtp", default=None, help="既定は --model の中の mtp.safetensors")
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=0, help="既定 0 = 短プロンプト")
    ap.add_argument("--widths", default="1,2,3,4")
    ap.add_argument("--reps", type=int, default=30, help="S ごとの反復回数 (最初の 3 回を含む)")
    ap.add_argument("--rounds", type=int, default=2, help="回文掃引 (昇順/降順) の周回数")
    ap.add_argument(
        "--with-draft",
        action="store_true",
        help="draft_chain + verify forward + rollback の 1 ラウンド費用も測る",
    )
    ap.add_argument("--out", default=str(REPO_ROOT / "bench" / "results" / "verify-width-cost.json"))
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    if not widths:
        print("--widths が空")
        return 1
    if args.rounds < 1:
        print("--rounds は 1 以上")
        return 1
    if args.reps < args.rounds:
        print("--reps は --rounds 以上 (周ごとに最低 1 反復必要)")
        return 1

    eng, model, tok, eos_ids = build_runner(args)
    ids = build_prompt_ids(tok, args.ctx)

    import mlx.core as mx
    from mlxturbo import spec_flash
    from mlxturbo.spec_flash import capture, rollback, snapshot_pre
    from mlxturbo.spec_flash import _staged_forward  # noqa: SLF001 (依頼どおり直接使う)
    from mlxturbo.spec_flash import restore_mtp_cache

    knob_defaults = {
        "MLXTURBO_STAGE_EVERY": os.environ.get("MLXTURBO_STAGE_EVERY", "2"),
        "MLXTURBO_PREFILL_GROUP": os.environ.get("MLXTURBO_PREFILL_GROUP", "4"),
        "MLXTURBO_DRAFT_RERANK": os.environ.get("MLXTURBO_DRAFT_RERANK", "1"),
        "MLXTURBO_HC": os.environ.get("MLXTURBO_HC", "kernel"),
        "MLXTURBO_SORT_MIN": os.environ.get("MLXTURBO_SORT_MIN", "16"),
    }
    print(
        f"ctx={ids.shape[1]} (--ctx {args.ctx})  MTP_DEPTH(既定)={spec_flash.MTP_DEPTH}  "
        f"widths={widths}  reps={args.reps}  rounds={args.rounds}  with_draft={args.with_draft}"
    )
    print("  knob 既定値: " + "  ".join(f"{k}={v}" for k, v in knob_defaults.items()))

    caches, snap, resume, _first = prefill_once(eng, ids, eos_ids)
    print(f"  prefill 1 回だけ流した (n={ids.shape[1]})。以降は同じ状態から退避・復元する。")

    _logits_tail, hyper_tail0, mtp_snap = resume
    pair_full, cur = build_pair(eng, resume, max(widths))

    fwd_times: dict[int, list[float]] = {s: [] for s in widths}
    round_times: dict[int, list[float]] = {s: [] for s in widths} if args.with_draft else {}

    base = args.reps // args.rounds
    rem = args.reps % args.rounds
    for sweep_idx, order in enumerate(sweep_order(widths, args.rounds)):
        n_this = base + (1 if sweep_idx < rem else 0)
        for s in order:
            # ---- forward-only: 幅 S の verify forward 1 回だけの ms ----
            pair = pair_full[:, :s]
            for _ in range(n_this):
                _restore(caches, snap)
                t0 = time.perf_counter()
                with capture(model) as _cap:
                    lg = _staged_forward(model, pair, caches)
                mx.eval(lg)
                fwd_times[s].append(time.perf_counter() - t0)

            # ---- --with-draft: draft_chain + verify forward + rollback ----
            if args.with_draft:
                depth = max(0, s - 1)
                for _ in range(n_this):
                    _restore(caches, snap)
                    mtp_cache = restore_mtp_cache(mtp_snap)
                    t0 = time.perf_counter()
                    drafts = eng._draft_chain(cur, hyper_tail0, mtp_cache, depth) if depth else []
                    rpair = mx.concatenate([cur] + drafts, axis=1)
                    pre = snapshot_pre(model, caches)
                    with capture(model) as cap:
                        lg = _staged_forward(model, rpair, caches)
                    mx.eval(lg)
                    toks, _hypers, _hit = eng._verify(cap, lg, drafts, 0.0)
                    rollback(
                        model, caches, cap, pre,
                        keep=len(toks), total=rpair.shape[1],
                        ids_kept=rpair[:, : len(toks)],
                    )
                    round_times[s].append(time.perf_counter() - t0)

    print("\n=== S ごとの verify forward 費用 ===")
    result = {
        "ctx": ids.shape[1],
        "ctx_arg": args.ctx,
        "mtp_depth_default": spec_flash.MTP_DEPTH,
        "widths": widths,
        "reps": args.reps,
        "rounds": args.rounds,
        "with_draft": args.with_draft,
        "knobs": knob_defaults,
        "forward": {},
        "round": {} if args.with_draft else None,
    }
    for s in widths:
        fwd = summarize(fwd_times[s])
        result["forward"][str(s)] = fwd
        line = (
            f"  S={s}  forward median {fwd['median_ms']:7.3f} ms"
            f"  min {fwd['min_ms']:7.3f} ms  (n={fwd['n']})"
        )
        if args.with_draft:
            rd = summarize(round_times[s])
            result["round"][str(s)] = {"n": rd["n"], "median_ms": rd["median_ms"]}
            line += f"   | round median {rd['median_ms']:7.3f} ms  (n={rd['n']})"
        print(line)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\n書き出し: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
