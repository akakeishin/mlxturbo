"""GDN の 8ms と、ablate.py の「残り」10ms の中身を割る。

`tools/ablate.py` は GDN を `out_proj(zeros)` まで落とすので、消えた時間に
射影の重み読み (1.51GB/token) が混ざっている。融合で削れるのは起動費用の
ぶんだけなので分ける。

「残り」も測る。ここには embed / lm_head / 最終 norm だけでなく、
**12 層ぶんの full attention 本体**と各層の hyper-connection の合流演算が
入っている (ablate.py が外しているのは QSA indexer だけ)。MoE ルーティングが
2.69ms しか無かったいま、こちらの方が大きい可能性がある。

**必ず tools/biglock.sh 経由で、静かなときに回すこと。**
docs/KERNEL-BRIEF-MOE-GDN.md の「引き継ぐ規律」を読むこと。

    tools/biglock.sh uv run python tools/ablate_gdn.py \
        --model ~/models/qwen38fn-mlx-v-fast6 --ngram ~/models/qwen38fn-ngram-4bit
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


def _gdn_variants(Q, mx, nn):
    orig = Q.GatedDeltaNet.__call__

    def _conv(self, x, mixed_qkv, cache):
        """conv1d は状態の連結が要る (S=1 では padding=0 で長さが足りない)。"""
        B = x.shape[0]
        conv_state = (
            cache[0]
            if (cache is not None and cache[0] is not None)
            else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        )
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :, :])
        return nn.silu(self.conv1d(conv_input))

    def no_delta(self, x, mask, cache):
        """状態更新だけ外す。射影・conv1d・RMSNormGated は残す。"""
        B, S, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.n_v, self.dv)
        self.in_proj_b(x)
        self.in_proj_a(x)
        conv_out = _conv(self, x, mixed_qkv, cache)
        v = conv_out[..., 2 * self.key_dim :].reshape(B, S, self.n_v, self.dv)
        if cache is not None:
            cache.advance(S)
        return self.out_proj(self.norm(v, z).reshape(B, S, -1))

    def no_norm(self, x, mask, cache):
        """さらに RMSNormGated を外す。

        **z を捨ててはいけない。**MLX は遅延評価なので、戻り値を使わない
        `self.in_proj_z(x)` は評価そのものが起きず、in_proj_z の行列積
        (36 層で 0.57GB) まで一緒に消えてしまう。最初この書き方をして
        「RMSNormGated 2.37ms」という 4 倍過大な値を出した。
        norm の代わりに素の乗算を置いて z を消費させる。
        """
        B, S, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.n_v, self.dv)
        self.in_proj_b(x)
        self.in_proj_a(x)
        conv_out = _conv(self, x, mixed_qkv, cache)
        v = conv_out[..., 2 * self.key_dim :].reshape(B, S, self.n_v, self.dv)
        if cache is not None:
            cache.advance(S)
        return self.out_proj((v * z).reshape(B, S, -1))

    def no_conv(self, x, mask, cache):
        """さらに conv1d + silu も外す。射影だけ残る (z は消費したまま)。"""
        B, S, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.n_v, self.dv)
        self.in_proj_b(x)
        self.in_proj_a(x)
        v = mixed_qkv[..., 2 * self.key_dim :].reshape(B, S, self.n_v, self.dv)
        if cache is not None:
            cache.advance(S)
        return self.out_proj((v * z).reshape(B, S, -1))

    def qkv_only(self, x, mask, cache):
        """in_proj_z/b/a を外す。ここで初めて z が消える。"""
        if cache is not None:
            cache.advance(x.shape[1])
        return self.out_proj(self.in_proj_qkv(x)[..., 2 * self.key_dim :])

    def out_only(self, x, mask, cache):
        """ablate.py の GDN 段と同じ。"""
        if cache is not None:
            cache.advance(x.shape[1])
        return self.out_proj(
            mx.zeros((*x.shape[:-1], self.value_dim), dtype=x.dtype)
        )

    return orig, [
        ("そのまま", orig),
        ("- 状態更新 (gated_delta_update)", no_delta),
        ("- RMSNormGated", no_norm),
        ("- conv1d + silu", no_conv),
        ("- in_proj_z/b/a", qkv_only),
        ("- in_proj_qkv", out_only),
    ]


def _rest_variants(Q, mx):
    """「残り」10ms のうち full attention 本体 (12 層) がどれだけかを見る。"""
    orig_attn = Q.Attention.__call__

    def attn_zero(self, x, rope, mask, cache, idx_cache):
        return mx.zeros_like(x)

    return orig_attn, [
        ("そのまま", orig_attn),
        ("- full attention 本体 (12 層)", attn_zero),
    ]


def _head_variants(Q, mx):
    """lm_head (2560 -> 248320、8bit で 636MB/token) の取り分を見る。"""
    orig = Q.Model.__call__

    def no_head(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        # 語彙全体を作らずに 1 列だけ返す。out を消費するので遅延評価で
        # 本体まで消えることはない
        return out[..., :1]

    return orig, [
        ("そのまま", orig),
        ("- lm_head (2560 -> 248320)", no_head),
    ]


def _run(label, variants, setter, model, ids, reps):
    samples = {name: [] for name, _ in variants}
    for r in range(reps):
        for name, fn in variants:
            setter(fn)
            samples[name].append(measure(model, ids))
        print(f"  {label} ラウンド {r + 1} 完了", flush=True)
    setter(variants[0][1])

    print(f"\n=== {label}: 段ごと (中央値) ===")
    med = {}
    for name, _ in variants:
        med[name] = statistics.median(samples[name])
        print(f"  {name:34s} {med[name]:6.2f} ms/token  (振れ {max(samples[name]) - min(samples[name]):4.2f})")
    print(f"\n=== {label}: 差分 ===")
    names = [n for n, _ in variants]
    for a, b in zip(names, names[1:]):
        print(f"  {b[2:]:32s} {med[a] - med[b]:6.2f} ms")
    return med


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

    from fastmlx import fused

    model, tok = load(args.model)
    if args.ngram:
        from fastmlx.ngram_stream import install

        install(model, args.ngram)
    # hyper-connections は融合済みを前提に測る (これが現状の出発点)
    fused.enable_hyper_connection_kernel()
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて説明してください。"}],
        add_generation_prompt=True,
    )
    print(f"peak={mx.get_peak_memory() / 1e9:.1f}GB  reps={args.reps}  (HC は融合済み)")

    import mlx.nn as nn

    orig_gdn, gdn = _gdn_variants(Q, mx, nn)

    def set_gdn(fn):
        Q.GatedDeltaNet.__call__ = fn

    _run("GDN", gdn, set_gdn, model, ids, args.reps)
    set_gdn(orig_gdn)

    orig_attn, rest = _rest_variants(Q, mx)

    def set_attn(fn):
        Q.Attention.__call__ = fn

    _run("full attention", rest, set_attn, model, ids, args.reps)
    set_attn(orig_attn)

    orig_head, head = _head_variants(Q, mx)

    def set_head(fn):
        Q.Model.__call__ = fn

    _run("lm_head", head, set_head, model, ids, args.reps)
    set_head(orig_head)


if __name__ == "__main__":
    main()
