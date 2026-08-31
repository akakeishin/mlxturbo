"""デコード 1 forward の固定費を、グラフ構築 (CPU/Python) と GPU 実行に分けて測る。

ablate.py は「どの部品か」を出したが、その時間が Python のグラフ構築なのか
GPU のカーネル起動なのかは分けられない。MLX は遅延評価なので、
model() の呼び出し自体 = グラフ構築、mx.eval = GPU 実行 + 待ち。
両者を別々に計れば、mx.compile で削れる側 (構築) の大きさが分かる。

構築と実行はパイプラインで重なるので、和は 1 トークンの実時間より長くて良い。
見たいのは「構築だけで何 ms あるか」。

    uv run python tools/probe_overhead.py --model <path> --ngram <sidecar>
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


def split_build_eval(model, ids, width: int, n: int = 20):
    """幅 width の forward を、構築時間と eval 時間に分けて中央値を返す。"""
    import mlx.core as mx

    cache = model.make_cache()
    mx.eval(model(mx.array(ids)[None], cache=cache))
    tok = mx.array([[ids[-1]] * width])
    for _ in range(3):
        mx.eval(model(tok, cache=cache))
    builds, evals, walls = [], [], []
    for _ in range(n):
        t0 = time.perf_counter()
        out = model(tok, cache=cache)
        t1 = time.perf_counter()
        mx.eval(out)
        t2 = time.perf_counter()
        builds.append((t1 - t0) * 1000)
        evals.append((t2 - t1) * 1000)
        walls.append((t2 - t0) * 1000)
    med = lambda xs: sorted(xs)[len(xs) // 2]
    return med(builds), med(evals), med(walls)


def bench(model, ids, label: str, width: int = 1):
    b, e, w = split_build_eval(model, ids, width)
    print(f"  {label:42s} T={width}  build={b:5.1f}ms  eval={e:5.1f}ms  計={w:5.1f}ms",
          flush=True)
    return w


def fuse_gate_up(model) -> int:
    """switch_mlp の gate/up を 1 本の gather に連結する (出力次元で concat)。

    量子化行列の連結は uint32 パックのまま行える (パックは入力次元方向、
    連結は出力次元方向なので独立)。数値は変わらない。
    """
    import mlx.core as mx

    n = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        sw = getattr(mlp, "switch_mlp", None) if mlp is not None else None
        if sw is None:
            continue
        g, u = sw.gate_proj, sw.up_proj
        if not hasattr(g, "scales"):
            continue
        w = mx.concatenate([g.weight, u.weight], axis=1)
        s = mx.concatenate([g.scales, u.scales], axis=1)
        b = mx.concatenate([g.biases, u.biases], axis=1)
        mx.eval(w, s, b)
        sw._fused_w, sw._fused_s, sw._fused_b = w, s, b
        sw._fused_h = g.weight.shape[1]
        n += 1
    if n == 0:
        return 0

    import mlx_lm.models.switch_layers as SL

    orig = SL.SwitchGLU.__call__

    def patched(self, x, indices):
        if not hasattr(self, "_fused_w"):
            return orig(self, x, indices)
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = SL._gather_sort(x, indices)
        gp = self.gate_proj
        both = mx.gather_qmm(
            x, self._fused_w, self._fused_s, self._fused_b,
            rhs_indices=idx, transpose=True,
            group_size=gp.group_size, bits=gp.bits, mode=gp.mode,
            sorted_indices=do_sort,
        )
        h = self._fused_h
        x_gate, x_up = both[..., :h], both[..., h:]
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        if do_sort:
            x = SL._scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    SL.SwitchGLU.__call__ = patched
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngram", default=None)
    args = ap.parse_args()

    if args.ngram:
        os.environ["FASTMLX_NGRAM_DISK"] = "1"

    import mlx.core as mx

    import mlxturbo  # noqa: F401
    from mlx_lm import load

    model, tok = load(args.model)
    if args.ngram:
        from mlxturbo.ngram_stream import install

        install(model, args.ngram)

    ids = tok.apply_chat_template(
        [{"role": "user", "content": "分散システムについて説明してください。"}],
        add_generation_prompt=True,
    )

    if os.environ.get("PROBE_T3_PARTS") == "1":
        import mlx_lm.models.qwen4_exp as Q

        def parts_at(width):
            print(f"=== T={width} の部品別 (積み上げ無効化) ===")
            base = bench(model, ids, "そのまま", width)
            saved = []
            for layer in model.model.layers:
                if getattr(layer, "ple", None) is not None:
                    saved.append((layer, layer.ple)); layer.ple = None
            m1 = bench(model, ids, "- PLE", width)
            omoe = Q.SparseMoeBlock.__call__
            Q.SparseMoeBlock.__call__ = lambda self, x: self.shared_expert(x)
            m2 = bench(model, ids, "- MoE experts+router", width)
            ohc = Q.GatedResidual.__call__
            def hc_pass(self, hyper):
                mixed = hyper.reshape(*hyper.shape[:-1], self.hc, self.d).mean(axis=-2)
                if self.block_inject_weight is None:
                    return mixed
                return mixed, hyper, mx.ones((*hyper.shape[:-1], self.hc), dtype=hyper.dtype)
            Q.GatedResidual.__call__ = hc_pass
            m3 = bench(model, ids, "- HC", width)
            ogdn = Q.GatedDeltaNet.__call__
            Q.GatedDeltaNet.__call__ = lambda self, x, mask, cache: self.out_proj(
                mx.zeros((*x.shape[:-1], self.value_dim), dtype=x.dtype))
            m4 = bench(model, ids, "- GDN", width)
            oattn = Q.Attention.__call__
            Q.Attention.__call__ = lambda self, x, rope, mask, cache, idx_cache: x
            m5 = bench(model, ids, "- full attention", width)
            Q.Attention.__call__ = oattn
            Q.GatedDeltaNet.__call__ = ogdn
            Q.GatedResidual.__call__ = ohc
            Q.SparseMoeBlock.__call__ = omoe
            for layer, ple in saved:
                layer.ple = ple
            print(f"  内訳: PLE={base-m1:.1f} MoE={m1-m2:.1f} HC={m2-m3:.1f}"
                  f" GDN={m3-m4:.1f} attn={m4-m5:.1f} 残り={m5:.1f}")

        parts_at(1)
        parts_at(3)
        return

    print("=== 1. 素 (現状の既定) ===")
    bench(model, ids, "そのまま", 1)
    bench(model, ids, "そのまま", 2)
    bench(model, ids, "そのまま", 4)

    print("=== 2. + moe_route カーネル ===")
    from mlxturbo import fused as F

    F.enable_moe_route()
    bench(model, ids, "moe_route", 1)
    bench(model, ids, "moe_route", 2)

    print("=== 3. + rebit hc=4 + HC 融合カーネル ===")
    from mlxturbo import rebit

    rebit.apply(model, "hc=4", verbose=False)
    F.enable_hyper_connection_kernel()
    bench(model, ids, "moe_route + HC カーネル", 1)
    bench(model, ids, "moe_route + HC カーネル", 2)

    print("=== 4. + gate/up 連結 ===")
    n = fuse_gate_up(model)
    print(f"  ({n} 層を連結)")
    bench(model, ids, "moe_route + HC + gate/up 連結", 1)
    bench(model, ids, "moe_route + HC + gate/up 連結", 2)
    bench(model, ids, "moe_route + HC + gate/up 連結", 4)

    # 出力が壊れていないか一言だけ吐かせる
    print("=== 検算 (greedy 20 トークン) ===")
    cache = model.make_cache()
    logits = model(mx.array(ids)[None], cache=cache)
    cur = int(mx.argmax(logits[0, -1], axis=-1))
    outs = [cur]
    for _ in range(20):
        logits = model(mx.array([[cur]]), cache=cache)
        cur = int(mx.argmax(logits[0, -1], axis=-1))
        outs.append(cur)
    print(" ", tok.decode(outs))


if __name__ == "__main__":
    main()
