"""m=1 decode と prefill の部品分解。

同一グラフ内に N 回連鎖させて 1 eval で測ることで、単発 eval の
起動オーバーヘッドを均し、実行中に近い条件で部品時間を出す。
部品の総和と実測ステップ時間の差が「隙間」(ディスパッチ・グルー) になる。
注意: 同じ層を連鎖させるので SLC の重み再利用で実機より甘く出うる。
"""

import argparse
import json
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def module_bytes(mod) -> int:
    return sum(v.nbytes for _, v in tree_flatten(mod.parameters()))


def chain_time(fn, x, n=8, reps=10):
    def run():
        h = x
        for _ in range(n):
            h = fn(h)
        mx.eval(h)

    run()
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lmstudio-community/Qwen3.8-27B-MLX-4bit")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    inner = model.language_model.model
    lm_head = model.language_model.lm_head
    lin = next(l for l in inner.layers if l.is_linear)
    fa = next(l for l in inner.layers if not l.is_linear)
    la = lin.linear_attn
    D = 5120

    b_lin = module_bytes(lin)
    b_fa = module_bytes(fa)
    b_head = module_bytes(lm_head)
    total_bytes = module_bytes(model)

    out = {
        "bytes_gb": {
            "linear_layer": b_lin / 1e9,
            "fa_layer": b_fa / 1e9,
            "lm_head": b_head / 1e9,
            "model_total": total_bytes / 1e9,
        }
    }

    # ---- 実測の decode 1 step (文脈 512) ----
    caches = make_prompt_cache(model)
    prompt = mx.array(tokenizer.encode("設計。" * 400)[:512])
    mx.eval(model(prompt[None], cache=caches))
    tok = mx.array([[1000]])

    def step():
        mx.eval(model(tok, cache=caches))

    step()
    mx.synchronize()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        step()
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    t_step = sorted(ts)[len(ts) // 2]

    # ---- 部品 (m=1, 連鎖) ----
    x1 = mx.random.normal((1, 1, D)).astype(mx.float16)
    t_lin1 = chain_time(lambda h: lin(h, mask=None, cache=None), x1)
    t_fa1 = chain_time(lambda h: fa(h, mask=None, cache=None), x1)
    t_head1 = chain_time(lambda h: lm_head(h)[..., :D], x1, n=4)

    sum_parts = 48 * t_lin1 + 16 * t_fa1 + t_head1
    out["decode_m1"] = {
        "step_ms": t_step * 1e3,
        "achieved_gbs": total_bytes / t_step / 1e9,
        "linear_ms": t_lin1 * 1e3,
        "linear_gbs": b_lin / t_lin1 / 1e9,
        "fa_ms": t_fa1 * 1e3,
        "fa_gbs": b_fa / t_fa1 / 1e9,
        "lm_head_ms": t_head1 * 1e3,
        "lm_head_gbs": b_head / t_head1 / 1e9,
        "sum_parts_ms": sum_parts * 1e3,
        "gap_ms": (t_step - sum_parts) * 1e3,
    }

    # ---- 線形層の内訳 (m=1) ----
    t_qkv = chain_time(lambda h: la.in_proj_qkv(h)[..., :D], x1, n=16)
    t_z = chain_time(lambda h: la.in_proj_z(h)[..., :D], x1, n=16)
    t_outp = chain_time(
        lambda h: la.out_proj(mx.broadcast_to(h[..., :1], (1, 1, 6144))), x1, n=16
    )
    xin = mx.random.normal((1, 4, la.conv_dim)).astype(mx.float16)
    t_conv = chain_time(
        lambda h: mx.broadcast_to(la.conv1d(h), (1, 4, la.conv_dim)), xin, n=16
    )
    out["linear_inner_m1_ms"] = {
        "in_proj_qkv": t_qkv * 1e3,
        "in_proj_z": t_z * 1e3,
        "out_proj": t_outp * 1e3,
        "conv1d": t_conv * 1e3,
    }

    # ---- prefill (S=512, 連鎖) ----
    x512 = mx.random.normal((1, 512, D)).astype(mx.float16)
    t_lin512 = chain_time(lambda h: lin(h, mask=None, cache=None), x512, n=4, reps=5)
    t_fa512 = chain_time(lambda h: fa(h, mask="causal", cache=None), x512, n=4, reps=5)
    sum512 = 48 * t_lin512 + 16 * t_fa512

    # 実測 prefill
    caches2 = make_prompt_cache(model)
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval(model(prompt[None], cache=caches2))
    mx.synchronize()
    t_prefill = time.perf_counter() - t0

    # 線形層 prefill の内訳: 射影 vs scan
    t_proj512 = chain_time(lambda h: la.in_proj_qkv(h)[..., :D], x512, n=4, reps=5)
    q = mx.random.normal((1, 512, 16, 128)).astype(mx.float16)
    k = mx.random.normal((1, 512, 16, 128)).astype(mx.float16)
    v = mx.random.normal((1, 512, 48, 128)).astype(mx.float16)
    a_ = mx.random.normal((1, 512, 48)).astype(mx.float16)
    b_ = mx.random.normal((1, 512, 48)).astype(mx.float16)
    from mlx_lm.models.gated_delta import gated_delta_update

    def scan_once(_):
        o, s = gated_delta_update(q, k, v, a_, b_, la.A_log, la.dt_bias, None, None)
        return o[..., :1, :1, :1].reshape(1, 1, 1)

    def scan_run():
        outs = [scan_once(None) for _ in range(4)]
        mx.eval(*outs)

    scan_run()
    mx.synchronize()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        scan_run()
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    t_scan512 = sorted(ts)[len(ts) // 2] / 4

    out["prefill_512"] = {
        "actual_ms": t_prefill * 1e3,
        "linear_layer_ms": t_lin512 * 1e3,
        "fa_layer_ms": t_fa512 * 1e3,
        "sum_parts_ms": sum512 * 1e3,
        "gap_ms": (t_prefill - sum512) * 1e3,
        "linear_qkv_proj_ms": t_proj512 * 1e3,
        "delta_scan_only_ms": t_scan512 * 1e3,
    }

    print(json.dumps(out, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
