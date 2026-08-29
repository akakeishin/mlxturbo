"""hyper-connection の融合が (1) 数値一致するか (2) 速いか を確かめる。

融合カーネルを書くときの受け入れ確認に使う。速度より先に数値を見ること
(mx.compile 版は top1 が一致したまま logits が 5% ずれた)。
基準は docs/KERNEL-BRIEF-HC.md。

「fp32 sigmoid」だけは融合ではなく**対照**。素の実装と MLX の op 単位で完全に
同じで、sigmoid を fp32 で計算するぶん素より正確なだけの版。ここに出る誤差は
実装の間違いではなく bf16 の丸めが 96 段で膨らんだ量そのものなので、
「相対誤差 1e-3 未満」が原理的に届く基準かどうかがこれで分かる。

    uv run python tools/hc_equiv_test.py \
        --model ~/models/qwen38fn-mlx-v-stream --ngram ~/models/qwen38fn-ngram-4bit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROMPT = "分散システムについて説明してください。"


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


def _install_sigmoid32(Q):
    """素の実装と同じ op 構成で、sigmoid だけ fp32 で計算する対照版。"""

    import mlx.core as mx

    def sig32(x):
        return (1.0 / (1.0 + mx.exp(-x.astype(mx.float32)))).astype(x.dtype)

    def patched(self, hyper):
        normed = self.hc_norm(hyper)
        a = self.input_mix_weight_down(normed) / self.hc
        w = a * sig32(a)  # nn.silu と同じ x * sigmoid(x)
        w = sig32(self.input_mix_weight_up(w))
        w = w.reshape(*w.shape[:-1], self.hc, self.d)
        mixed = (w * normed.reshape(*normed.shape[:-1], self.hc, self.d)).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * sig32(self.block_inject_weight(normed) / self.hc)
        return mixed, hyper, inject

    Q.GatedResidual.__call__ = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/ht/models/qwen38fn-mlx-v-stream")
    ap.add_argument("--ngram", default="/Users/ht/models/qwen38fn-ngram-4bit")
    ap.add_argument("--no-speed", action="store_true")
    args = ap.parse_args()

    import os

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    import mlx_lm.models.qwen4_exp as Q

    from mlxturbo import fused

    model, tok = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )
    full = mx.array(ids)[None]

    def logits():
        out = model(full, cache=model.make_cache())[0, -1].astype(mx.float32)
        mx.eval(out)
        return np.array(out)

    base = logits()

    def report(name, arr):
        rel = np.linalg.norm(arr - base) / max(np.linalg.norm(base), 1e-9)
        top1 = "一致" if arr.argmax() == base.argmax() else "不一致"
        print(f"  {name:26s} 相対誤差 {rel:.6f}  top1 {top1}", flush=True)

    print("=== 数値 (プロンプト一括 forward、最終位置の logits) ===")
    report("素をもう一度 (再現性)", logits())

    # importlib.reload では既存インスタンスの __class__ が古いままなので戻らない。
    # 差し替え前の関数を持っておいて明示的に書き戻す
    orig_call = Q.GatedResidual.__call__
    _install_sigmoid32(Q)
    sig32_logits = logits()
    Q.GatedResidual.__call__ = orig_call
    report("対照: fp32 sigmoid", sig32_logits)

    fused.enable_hyper_connection()
    report("mx.compile", logits())
    fused.disable_hyper_connection()

    fused.enable_hyper_connection_kernel()
    kernel_logits = logits()
    fused.disable_hyper_connection_kernel()
    report("融合カーネル", kernel_logits)

    if args.no_speed:
        return

    print("\n=== 速度 (逐次デコード) ===")
    before = measure(model, ids)
    fused.enable_hyper_connection_kernel()
    after = measure(model, ids)
    fused.disable_hyper_connection_kernel()
    print(f"  融合なし {before:6.2f} ms/token ({1000 / before:5.2f} tok/s)")
    print(
        f"  融合あり {after:6.2f} ms/token ({1000 / after:5.2f} tok/s)"
        f"   {after - before:+.2f} ms"
    )


if __name__ == "__main__":
    main()
