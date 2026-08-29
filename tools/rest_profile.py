"""ablate.py が「残り」としてまとめている約 10ms を分解する。

ablate.py は PLE / QSA / MoE / hyper-connections / GDN まで外して止まる。
hyper-connections を融合カーネルで畳んだいま、その「残り」が最大の固定費に
なった。中身が分からないままだと次にどこへ手を入れるか決まらない。

ablate.py と同じ順で外した上で、さらに続きを外す:

    full attention (12 層)  ... ablate.py は GDN しか外していない
    残差の合成              ... 1 層 2 回 x 48 層
    lm_head                 ... 248320 語彙への射影
    最終段の hyper_connection_mixer

ablate.py を複製しているのは、あちらをカーネルセッションが触っているため。
用が済んだら向こうへ畳んでよい。

    uv run python tools/rest_profile.py --model <path> --ngram <sidecar>
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


def measure(model, ids, n=30) -> float:
    import mlx.core as mx

    cache = model.make_cache()
    logits = model(mx.array(ids)[None], cache=cache)
    cur = int(mx.argmax(logits[0, -1], axis=-1))
    for _ in range(5):
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
    best = None
    for _ in range(2):
        t0 = time.perf_counter()
        for _ in range(n):
            logits = model(mx.array([[cur]]), cache=cache)
            cur = int(mx.argmax(logits[0, -1], axis=-1))
        ms = (time.perf_counter() - t0) / n * 1000
        best = ms if best is None else min(best, ms)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    ap.add_argument("--fused-hc", action="store_true")
    args = ap.parse_args()

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx
    import mlx_lm.models.qwen4_exp as Q
    from mlx_lm import load

    model, tok = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)
    if args.fused_hc:
        from mlxturbo import fused

        fused.enable_hyper_connection_kernel()
        print("hyper-connections は融合カーネル")

    ids = tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて詳しく説明してください。"}],
        add_generation_prompt=True,
    )

    rows: list[tuple[str, float]] = []

    def record(name):
        # 温めてから測る。92GB のページが順に載る過程が段の差に化ける
        ms = measure(model, ids)
        rows.append((name, ms))
        print(f"  {name:34s} {ms:6.2f} ms/token  ({1000 / ms:5.2f} tok/s)", flush=True)

    print("=== 積み上げ式の無効化 ===")
    record("そのまま")

    for layer in model.model.layers:
        if getattr(layer, "ple", None) is not None:
            layer.ple = None
    record("- PLE (n-gram)")

    Q.QSAIndexer.__call__ = lambda self, x, rope, cache, offset: None
    record("- QSA indexer")

    Q.SparseMoeBlock.__call__ = lambda self, x: self.shared_expert(x)
    record("- MoE ルーティング")

    def hc_passthrough(self, hyper):
        mixed = hyper.reshape(*hyper.shape[:-1], self.hc, self.d).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        return mixed, hyper, mx.ones((*hyper.shape[:-1], self.hc), dtype=hyper.dtype)

    Q.GatedResidual.__call__ = hc_passthrough
    record("- hyper-connections")

    Q.GatedDeltaNet.__call__ = lambda self, x, mask, cache: self.out_proj(
        mx.zeros((*x.shape[:-1], self.value_dim), dtype=x.dtype)
    )
    record("- GDN (44 層)")

    # --- ここから先が ablate.py の「残り」の中身 ---

    # full attention (12 層)。ablate.py は GDN しか外していないので、
    # 12 層ぶんの q/k/v/o と SDPA が「残り」に入ったままだった
    Q.Attention.__call__ = lambda self, x, rope, mask, cache, idx_cache: self.o_proj(
        mx.zeros((*x.shape[:-1], self.n_heads * self.head_dim), dtype=x.dtype)
    )
    record("- full attention (12 層)")

    # 残差の合成: hyper + (x[...,None,:] * inject[...,None]).reshape(...)
    # 1 層 2 回 x 48 層。レーン混合をやめて素の加算にする
    orig_layer = Q.DecoderLayer.__call__

    def layer_plain_residual(self, h, rope, mask, conv_mask, cache, idx_c, ids, prev):
        if self.ple is not None:
            h = h + self.ple(h, ids, prev, cache)
        x, hyper, _ = self.attn_hyper_connection(h)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x, conv_mask, cache)
        else:
            x = self.self_attn(x, rope, mask, cache, idx_c)
        h = hyper + mx.tile(x, (1, 1, self.attn_hyper_connection.hc))
        x, hyper, _ = self.mlp_hyper_connection(h)
        x = self.mlp(x)
        return hyper + mx.tile(x, (1, 1, self.mlp_hyper_connection.hc))

    Q.DecoderLayer.__call__ = layer_plain_residual
    record("- 残差の合成 (96 回)")

    # lm_head: 2560 -> 248320 の射影。読み出しは 0.675GB
    orig_head = model.lm_head

    class _StubHead:
        def __call__(self, x):
            return mx.zeros((*x.shape[:-1], 256), dtype=x.dtype)

    model.lm_head = _StubHead()
    record("- lm_head")

    model.lm_head = orig_head
    Q.DecoderLayer.__call__ = orig_layer

    print("\n=== 各段の削減幅 ===")
    base = rows[0][1]
    for (_, pms), (name, ms) in zip(rows, rows[1:]):
        print(f"  {name[2:]:32s} {pms - ms:6.2f} ms  "
              f"({(pms - ms) / base * 100:4.1f}% of 全体)")
    print(f"  {'ここまで外して残る量':32s} {rows[-1][1]:6.2f} ms")
    print(f"\n  ({len(rows) - 1} 段で {base - rows[-1][1]:.2f} ms を説明した "
          f"/ 全体 {base:.2f} ms)")


if __name__ == "__main__":
    main()
