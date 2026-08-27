"""MoE の 11ms の内訳を、ルーティング / shared gate / expert 行列積 に割る。

`tools/ablate.py` の MoE 段は `shared_expert(x)` だけ残して全部消すので、
**expert の重み読み (1.77GB/token) まで一緒に消えている**。融合カーネルで
削れるのは起動費用のぶんだけなので、そこを分けないと取り分を見誤る。

段を 1 つずつ外して差分を取る:

    そのまま          gate -> argpartition -> softmax -> switch_mlp -> shared
    - ルーティング     idx と重みを固定にする (switch_mlp は残す)
    - shared gate     shared_expert_gate と sigmoid を外す
    - expert 行列積    switch_mlp ごと外す (= ablate.py の MoE 段と同じ)

マシンが他のジョブを抱えていると差分が壊れる (docs/KERNEL-HANDOFF-HC.md の
「ablate は他のジョブと同時に走らせると壊れる」)。**必ず tools/biglock.sh
経由で、静かなときに回すこと。**各段を繰り返して中央値を取る。

    tools/biglock.sh uv run python tools/ablate_moe.py \
        --model ~/models/qwen38fn-mlx-v-stream --ngram ~/models/qwen38fn-ngram-4bit
"""

from __future__ import annotations

import argparse
import statistics
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


def _variants(Q, mx):
    """(名前, __call__) の並び。上から順に段を外していく。"""

    orig = Q.SparseMoeBlock.__call__

    def _fixed(self, x):
        # idx と重みは 1 度作って使い回す (毎回作ると測りたくない op が乗る)
        key = x.shape[:-1]
        cached = getattr(self, "_fastmlx_fixed", None)
        if cached is None or cached[0] != key:
            idx = mx.broadcast_to(mx.arange(self.top_k), (*key, self.top_k))
            w = mx.full((*key, self.top_k), 1.0 / self.top_k, dtype=mx.float32)
            mx.eval(idx, w)
            self._fastmlx_fixed = (key, idx, w)
            cached = self._fastmlx_fixed
        return cached[1], cached[2]

    def no_route(self, x):
        idx, w = _fixed(self, x)
        out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        return out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    def no_shared_gate(self, x):
        idx, w = _fixed(self, x)
        out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        return out + self.shared_expert(x)

    def no_expert(self, x):
        return self.shared_expert(x)

    return orig, [
        ("そのまま", orig),
        ("- ルーティング (gate/topk/softmax)", no_route),
        ("- shared gate (sigmoid)", no_shared_gate),
        ("- expert 行列積 (switch_mlp)", no_expert),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    import os

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    import mlx_lm.models.qwen4_exp as Q

    model, tok = load(args.model)
    if args.ngram:
        from fastmlx.ngram_stream import install

        install(model, args.ngram)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて説明してください。"}],
        add_generation_prompt=True,
    )

    orig, variants = _variants(Q, mx)
    samples: dict[str, list[float]] = {name: [] for name, _ in variants}

    print(f"peak={mx.get_peak_memory() / 1e9:.1f}GB  reps={args.reps}")
    # 段ごとにまとめてではなく、ラウンドを繰り返して交互に測る。
    # マシンの状態がゆっくり動いても、段の間の比較が壊れにくい
    for r in range(args.reps):
        for name, fn in variants:
            Q.SparseMoeBlock.__call__ = fn
            samples[name].append(measure(model, ids))
        print(f"  ラウンド {r + 1} 完了", flush=True)
    Q.SparseMoeBlock.__call__ = orig

    print("\n=== 段ごと (中央値) ===")
    med = {}
    for name, _ in variants:
        med[name] = statistics.median(samples[name])
        spread = max(samples[name]) - min(samples[name])
        print(f"  {name:36s} {med[name]:6.2f} ms/token  (振れ {spread:4.2f})")

    print("\n=== 差分 ===")
    names = [n for n, _ in variants]
    for a, b in zip(names, names[1:]):
        print(f"  {b[2:]:34s} {med[a] - med[b]:6.2f} ms")
    print(f"\n  MoE 全体 (そのまま - expert なし) {med[names[0]] - med[names[-1]]:6.2f} ms")
    print("  ※ 融合で削れるのは上 2 段。expert 行列積は重み読みなので削れない")


if __name__ == "__main__":
    main()
