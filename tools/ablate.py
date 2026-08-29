"""デコードの固定費 47.6ms/token がどこに消えているかを部品別に測る。

一括 forward が S=16 でも S=1 の 1.17 倍しかかからないので、コストのほぼ全部は
「forward 1 回あたりの固定費」= カーネル起動の回数。どのモジュールが起動回数を
稼いでいるかを、部品を 1 つずつ無効化して差分で見る。

無効化するとモデルは壊れるが、ここで見たいのは速度だけなので構わない。
出力の正しさは quant_eval で別に見ている。

    uv run python tools/ablate.py --model <path>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def measure(model, ids, n=25) -> float:
    import mlx.core as mx

    cache = model.make_cache()
    logits = model(mx.array(ids)[None], cache=cache)
    cur = int(mx.argmax(logits[0, -1], axis=-1))
    for _ in range(3):
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
    ap.add_argument("--ngram", default=None,
                    help="n-gram サイドカー。ngram_disk で焼いたモデルには必須")
    ap.add_argument("--rebit", default=None,
                    help="読み込み後にビットを打ち直す (例 gdn=4)")
    ap.add_argument("--fused-hc", action="store_true",
                    help="hyper-connections を融合カーネルで動かしてから測る "
                         "(mlxturbo.fused.enable_hyper_connection_kernel)")
    args = ap.parse_args()

    import os

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    import mlx_lm.models.qwen4_exp as Q

    model, tok = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)
    if args.rebit:
        from mlxturbo import rebit

        rebit.apply(model, args.rebit)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて説明してください。"}],
        add_generation_prompt=True,
    )
    if args.fused_hc:
        from mlxturbo import fused

        fused.enable_hyper_connection_kernel()
        print("hyper-connections を融合カーネルに差し替えた")
    layers = model.model.layers

    rows: list[tuple[str, float]] = []

    def record(name):
        ms = measure(model, ids)
        rows.append((name, ms))
        print(f"  {name:34s} {ms:6.2f} ms/token  ({1000 / ms:5.2f} tok/s)", flush=True)
        return ms

    print(f"peak={mx.get_peak_memory() / 1e9:.1f}GB\n=== 積み上げ式の無効化 ===")
    base = record("そのまま")

    # 1. PLE (n-gram)
    for layer in layers:
        if getattr(layer, "ple", None) is not None:
            layer.ple = None
    record("- PLE (n-gram)")

    # 2. QSA indexer: 疎選択をやめて全体を見る (None を返せば密になる)
    orig_idx = Q.QSAIndexer.__call__
    Q.QSAIndexer.__call__ = lambda self, x, rope, cache, offset: None
    record("- QSA indexer")

    # 3. MoE のルーティング: 共有エキスパートだけにする
    orig_moe = Q.SparseMoeBlock.__call__

    def moe_shared_only(self, x):
        return self.shared_expert(x)

    Q.SparseMoeBlock.__call__ = moe_shared_only
    record("- MoE ルーティング")

    # 4. hyper-connections: レーン混合をやめて素通し
    orig_hc = Q.GatedResidual.__call__

    def hc_passthrough(self, hyper):
        mixed = hyper.reshape(*hyper.shape[:-1], self.hc, self.d).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        return mixed, hyper, mx.ones((*hyper.shape[:-1], self.hc), dtype=hyper.dtype)

    Q.GatedResidual.__call__ = hc_passthrough
    record("- hyper-connections")

    # 5. 線形アテンション (GDN)
    orig_gdn = Q.GatedDeltaNet.__call__
    Q.GatedDeltaNet.__call__ = lambda self, x, mask, cache: self.out_proj(
        mx.zeros((*x.shape[:-1], self.value_dim), dtype=x.dtype)
    )
    record("- GDN")

    Q.QSAIndexer.__call__ = orig_idx
    Q.SparseMoeBlock.__call__ = orig_moe
    Q.GatedResidual.__call__ = orig_hc
    Q.GatedDeltaNet.__call__ = orig_gdn

    print("\n=== 各段の削減幅 ===")
    for (pname, pms), (name, ms) in zip(rows, rows[1:]):
        print(f"  {name[2:]:32s} {pms - ms:6.2f} ms  ({(pms - ms) / base * 100:4.1f}% of 全体)")
    print(f"  {'残り (embed/lm_head/norm/dispatch)':32s} {rows[-1][1]:6.2f} ms")


if __name__ == "__main__":
    main()
