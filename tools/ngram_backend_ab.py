"""StreamNGram の mmap 経路と pread 経路を、同じロード済みモデル内で A/B する。

tools/decode_profile.py を mmap 版・pread 版で 2 回別プロセス起動して比べると、
片方が別セッションの熱ソークや電力モードに当たっただけで結果が入れ替わる
(直近のコミット "経路表反転を公式軌道の in-model A/B で正当化" と同じ理由)。
実際、この検証中も pread 実行時 20.7 tok/s / mmap 実行時 16.1 tok/s と
そのまま比べられない差が出た。

なので 1 回のモデルロードの中で PLE 層の `ngram_embedding` を pread<->mmap
で何度も差し替えながら短いブロックずつ計測し、ブロックの順序を反転して
交互に取る。ドリフト(サーマル・電力モード)は両条件に均等に乗るので、
差し替えの後先だけで結果が決まらないことを確認できる。

使い方:
  tools/biglock.sh uv run python tools/ngram_backend_ab.py \
      --model ~/models/qwen38fn-mlx-v-fast6 \
      --ngram ~/models/qwen38fn-ngram-4bit --blocks 6 --tokens 12
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def swap_backend(model, stream) -> None:
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        ple.ple_embedding.ngram_embedding = stream


def timed_tokens(model, cache, cur: int, n: int, mx) -> tuple[int, float]:
    t0 = time.perf_counter()
    for _ in range(n):
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
    dt = time.perf_counter() - t0
    return cur, dt / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", required=True)
    ap.add_argument("--blocks", type=int, default=6, help="backend ごとの計測ブロック数")
    ap.add_argument("--tokens", type=int, default=12, help="1 ブロックのトークン数")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument(
        "--fused",
        action="store_true",
        help="hyper-connections 融合カーネルを有効にしてから測る。既定 off の"
        "ままだと 1 トークン 49ms 前後の (出荷経路より遅い) 土俵で比べることに"
        "なり、n-gram の読み出しが占める割合が実際より小さく見える",
    )
    args = ap.parse_args()

    os.environ["FASTMLX_NGRAM_DISK"] = "1"
    import mlx.core as mx
    from mlx_lm import load

    from fastmlx.ngram_stream import StreamNGram

    model, tok = load(args.model)
    if args.fused:
        from fastmlx import fused

        fused.enable_hyper_connection_kernel()
        print("hyper-connections 融合カーネル有効")
    sidecar = Path(args.ngram)
    pread_stream = StreamNGram(sidecar, backend="pread", n_threads=args.threads)
    mmap_stream = StreamNGram(sidecar, backend="mmap")
    streams = {"pread": pread_stream, "mmap": mmap_stream}

    ids = tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて詳しく説明してください。"}],
        add_generation_prompt=True,
    )
    swap_backend(model, pread_stream)
    cache = model.make_cache()
    logits = model(mx.array(ids)[None], cache=cache)
    cur = int(mx.argmax(logits[0, -1], axis=-1))
    # 温める。両 backend を一度ずつ通しておく (初回だけ遅いモジュールが
    # あった場合に片方だけが割を食うのを防ぐ)
    for name in ("pread", "mmap"):
        swap_backend(model, streams[name])
        cur, _ = timed_tokens(model, cache, cur, 3, mx)

    # ブロックごとに backend を反転させながら交互に計測する。奇数ブロックは
    # pread から、偶数ブロックは mmap から始めて、順序効果を両方に均等に配る
    results: dict[str, list[float]] = {"pread": [], "mmap": []}
    order = ["pread", "mmap"]
    for b in range(args.blocks):
        seq = order if b % 2 == 0 else list(reversed(order))
        for name in seq:
            swap_backend(model, streams[name])
            cur, ms = timed_tokens(model, cache, cur, args.tokens, mx)
            results[name].append(ms)
            print(f"  block {b:2d}  {name:6s} {ms:6.2f} ms/token")

    print()
    for name in ("pread", "mmap"):
        vals = results[name]
        print(
            f"{name:6s} 中央値 {statistics.median(vals):6.2f} ms/token  "
            f"平均 {statistics.mean(vals):6.2f}  "
            f"最小 {min(vals):6.2f}  最大 {max(vals):6.2f}  (n={len(vals)})"
        )
    med_p, med_m = statistics.median(results["pread"]), statistics.median(results["mmap"])
    print(f"\npread は mmap 比 {med_m / med_p:.2f}x 速い (中央値ベース)")


if __name__ == "__main__":
    main()
