"""StreamNGram の mmap 経路と pread 経路が完全に同じ値を返すか確認する。

pread は量子化済みの行をそのまま読んで逆量子化するだけで、mmap 経路と
アルゴリズムは同一(レコードの切り出し方も同じ)。ビット単位で一致しなければ
どちらかの実装が壊れている。

モデルは読まない。サイドカー単体で StreamNGram を 2 通りの backend で立てて、
同じ gid 配列を投げて出力を比較する。biglock は不要。

使い方:
  uv run python tools/verify_ngram_pread.py --sidecar ~/models/qwen38fn-ngram-4bit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--n-rows", type=int, nargs="+", default=[1, 16, 128])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 12, 32])
    args = ap.parse_args()

    import mlx.core as mx
    import numpy as np

    from mlxturbo.ngram_stream import StreamNGram

    sidecar = Path(args.sidecar).expanduser()
    mmap_stream = StreamNGram(sidecar, backend="mmap")

    rng = np.random.default_rng(0)
    ok = 0
    total = 0
    for n in args.n_rows:
        for threads in args.threads:
            pread_stream = StreamNGram(sidecar, backend="pread", n_threads=threads)
            for _ in range(args.trials):
                gid = mx.array(rng.integers(0, mmap_stream.rows, size=n).astype(np.uint32))
                out_mmap = mmap_stream(gid)
                out_pread = pread_stream(gid)
                total += 1
                # 値だけでなく生バイトも見る。bfloat16 は NaN 表現が複数あり
                # 得るので、array_equal より確実に「ビット単位で同じ」を言える
                same_value = bool(mx.array_equal(out_mmap, out_pread))
                # bfloat16 は numpy のバッファプロトコルに乗らないので、
                # uint16 に view し直してから生バイトを比べる
                same_bytes = np.array(out_mmap.view(mx.uint16), copy=False).tobytes() == np.array(
                    out_pread.view(mx.uint16), copy=False
                ).tobytes()
                if same_value and same_bytes:
                    ok += 1
                else:
                    print(
                        f"不一致: n_rows={n} threads={threads} "
                        f"value_eq={same_value} bytes_eq={same_bytes}"
                    )
    print(f"{ok}/{total} 試行が mmap と pread でビット単位一致")
    if ok != total:
        raise SystemExit(1)
    print("OK: mmap 経路と pread 経路は完全に一致する")


if __name__ == "__main__":
    main()
