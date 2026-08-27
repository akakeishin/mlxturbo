"""同じプロセス内でビット打ち直しの前後を測る。

別ランどうしを比べると、熱や電力状態の違いがそのまま差に化ける (official2 rep1
を熱ソークで捨てた件)。1 回のロードで前後を測れば、その分は消える。

    uv run python tools/rebit_ab.py --model <path> --ngram <sidecar> --rebit gdn=4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def bench(model, ids, n: int) -> float:
    import mlx.core as mx

    cache = model.make_cache()
    logits = model(mx.array(ids)[None], cache=cache)
    cur = int(mx.argmax(logits[0, -1], axis=-1))
    for _ in range(5):
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
    t0 = time.perf_counter()
    for _ in range(n):
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--rebit", default=None)
    ap.add_argument("--fused-hc", action="store_true",
                    help="hyper-connections の融合カーネルを入れて前後を測る")
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if not args.rebit and not args.fused_hc:
        ap.error("--rebit か --fused-hc のどちらかが要る")

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"
    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(args.model)
    if args.ngram:
        from fastmlx.ngram_stream import install

        install(model, args.ngram)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて詳しく説明してください。"}],
        add_generation_prompt=True,
    )

    # 92GB のページが順に載る過程が段の差に化けるので、測る前に一度通す。
    # sweep の速度列はこれを怠って基準行が 69.75ms になり使えなくなった
    print("温め中 (ページを載せる)", flush=True)
    bench(model, ids, args.tokens)

    before = [bench(model, ids, args.tokens) for _ in range(args.repeats)]
    mem_before = mx.get_peak_memory() / 1e9
    print(f"変更前 {min(before):6.2f} ms/token ({1000 / min(before):5.2f} tok/s) "
          f"  全試行 {[round(v, 2) for v in before]}", flush=True)

    what = []
    if args.rebit:
        from fastmlx import rebit

        rebit.apply(model, args.rebit)
        what.append(args.rebit)
    if args.fused_hc:
        from fastmlx import fused

        fused.enable_hyper_connection_kernel()
        what.append("融合カーネル")
    mx.reset_peak_memory()

    after = [bench(model, ids, args.tokens) for _ in range(args.repeats)]
    print(f"変更後 {min(after):6.2f} ms/token ({1000 / min(after):5.2f} tok/s) "
          f"  全試行 {[round(v, 2) for v in after]}", flush=True)

    # 融合は切り戻せるので、交互に測り直して裏を取る。前半と後半に分けて測ると
    # 他セッションが途中で 98GB を読み始めただけで差が化ける (実際に化けた)。
    # 交互なら遅くなった区間が両側に等しく乗る
    if args.fused_hc and not args.rebit:
        from fastmlx import fused

        off, on = [], []
        for _ in range(args.repeats):
            fused.disable_hyper_connection_kernel()
            off.append(bench(model, ids, args.tokens))
            fused.enable_hyper_connection_kernel()
            on.append(bench(model, ids, args.tokens))
        print(f"\n  交互測定 (ドリフトを相殺)")
        print(f"    融合なし {min(off):6.2f} ms/token  {[round(v, 2) for v in off]}")
        print(f"    融合あり {min(on):6.2f} ms/token  {[round(v, 2) for v in on]}")
        print(f"    差 {min(off) - min(on):+.2f} ms/token "
              f"({1000 / min(on) - 1000 / min(off):+.2f} tok/s)")
    d = min(before) - min(after)
    print(f"\n  {' + '.join(what)}: {d:+.2f} ms/token "
          f"({1000 / min(after) - 1000 / min(before):+.2f} tok/s)")
    print(f"  ピーク RAM {mem_before:.1f} -> {mx.get_peak_memory() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
