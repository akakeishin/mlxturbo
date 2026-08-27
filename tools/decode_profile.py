"""デコード 1 トークンの内訳を測る。

Flash-Next は活性 6B x 4.47bpw = 3.35GB/token なので、M3 Max の 400GB/s なら
100 tok/s 前後が上限のはず。実測 21 tok/s は帯域の 18% しか使っておらず、
帯域律速ではなくオーバーヘッド律速。どこで詰まっているかを切り分ける。

見立て: n-gram の引き (`_ShardedEmbedding`) が毎トークン GPU -> CPU の同期を
強制している。np.unique / np.nonzero でホストに降りてから、触れたシャードごとに
numpy と MLX を往復する。層 1 でこれが起きるとパイプラインが毎トークン切れる。

    uv run python tools/decode_profile.py --model <path> [--ngram <sidecar>]
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


def bench(model, tok, ids, n_tokens: int, label: str) -> float:
    import mlx.core as mx

    cache = model.make_cache()
    logits = model(mx.array(ids)[None], cache=cache)
    cur = int(mx.argmax(logits[0, -1], axis=-1))
    # 数トークン捨てて温める
    for _ in range(3):
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
    dt = time.perf_counter() - t0
    tps = n_tokens / dt
    print(f"  {label:28s} {tps:6.2f} tok/s  ({dt / n_tokens * 1000:5.1f} ms/token)")
    return tps


def bench_batch(model, ids, sizes=(1, 2, 4, 8, 16)):
    """S トークンを一度に流したときの 1 forward の時間。

    ディスパッチ律速なら S を増やしても時間はほぼ変わらない。そうであれば
    MTP のような一括検証が受理数ぶんそのまま効く。帯域律速なら S に比例する。
    """
    import mlx.core as mx

    print("\n=== 一括 forward のスケーリング ===")
    print(f"  {'S':>3s} {'ms/forward':>11s} {'ms/token':>9s} {'S=1 比':>8s}")
    base = None
    for s in sizes:
        chunk = mx.array([[ids[-1]] * s])
        # 温め
        for _ in range(2):
            c = model.make_cache()
            mx.eval(model(mx.array(ids)[None], cache=c))
            mx.eval(model(chunk, cache=c))
        ts = []
        for _ in range(8):
            c = model.make_cache()
            # プロンプトの forward は**必ずここで評価しきる**。MLX は遅延評価
            # なので、eval せずにタイマーを開始すると、次の eval がプロンプト
            # ぶんまで巻き込んで計測区間に入る。それをやったせいで S=1 が
            # 490ms (逐次デコードは 50ms/token) と出て、S=16/S=1 が 1.2 倍に
            # 見えていた
            mx.eval(model(mx.array(ids)[None], cache=c))
            t = time.perf_counter()
            mx.eval(model(chunk, cache=c))
            ts.append((time.perf_counter() - t) * 1000)
        ms = sorted(ts)[len(ts) // 2]
        base = base or ms
        print(f"  {s:3d} {ms:11.2f} {ms / s:9.2f} {ms / base:8.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram-mode", default="disk", choices=("disk","ram"),
                    help="disk=memmap から引く / ram=連結テーブルを常駐")
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--rebit", default=None,
                    help="読み込み後にビットを打ち直す (例 gdn=4)。"
                         "焼かずに帯域を削ったときの速度を見る")
    args = ap.parse_args()

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"
    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(args.model)
    if args.ngram:
        from fastmlx.ngram_stream import install, install_ram

        (install_ram if args.ngram_mode == "ram" else install)(model, args.ngram)
    if args.rebit:
        from fastmlx import rebit

        rebit.apply(model, args.rebit)

    ids = tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて詳しく説明してください。"}],
        add_generation_prompt=True,
    )

    print(f"モデル: {args.model}")
    print(f"peak={mx.get_peak_memory() / 1e9:.1f}GB\n")

    base = bench(model, tok, ids, args.tokens, "そのまま")

    # PLE (n-gram) を切る。埋め込みがゼロなら PLE 出力もゼロなので層を外すのと同じ
    saved = []
    for layer in model.model.layers:
        if getattr(layer, "ple", None) is not None:
            saved.append((layer, layer.ple))
            layer.ple = None
    nople = bench(model, tok, ids, args.tokens, "PLE 無効 (n-gram 経路なし)")
    for layer, ple in saved:
        layer.ple = ple

    print()
    print(f"n-gram 経路のコスト: {(1 / base - 1 / nople) * 1000:.1f} ms/token "
          f"({100 * (nople - base) / base:+.1f}% の速度差)")
    print(f"活性 6B と仮定した帯域利用率: そのまま {base * 3.35 / 400 * 100:.0f}%, "
          f"PLE 無効 {nople * 3.35 / 400 * 100:.0f}%")

    bench_batch(model, ids)





if __name__ == "__main__":
    main()
