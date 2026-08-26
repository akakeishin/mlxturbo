"""fastmlx/kernels/gated_delta_states.py の正しさと性能を確認するスクリプト。

検証1 (正しさ): 実寸 (B=1, T in {1,4,8}, Hk=16, Hv=48, Dk=Dv=128) で
  (a) out が mlx_lm.gated_delta_update と一致 (fp16 丸め水準)
  (b) states_all[:, t] が「t+1 トークンで区切って逐次呼びした状態」と一致 (fp32 でほぼ厳密)
  (c) 最終状態 states_all[:, -1] が一括呼びの返り状態と一致 (fp32 でほぼ厳密)
を確認する。

検証2 (性能・簡易): T=8 で「一括 + states 出力」1 回 vs 「逐次 8 回呼び」の
時間を単発比較する (依存チェーンは不要)。他プロセスが GPU を使っている
可能性があるため参考値として扱うこと。

実行: uv run python bench/test_gated_delta_states.py
"""

import time

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_update

from fastmlx.kernels.gated_delta_states import gated_delta_update_with_states

B = 1
HK = 16
HV = 48
DK = 128
DV = 128
DTYPE = mx.float16

mx.random.seed(0)


def make_inputs(T: int):
    q = mx.random.normal((B, T, HK, DK)).astype(DTYPE)
    k = mx.random.normal((B, T, HK, DK)).astype(DTYPE)
    v = mx.random.normal((B, T, HV, DV)).astype(DTYPE)
    a = mx.random.normal((B, T, HV)).astype(DTYPE)
    b = mx.random.normal((B, T, HV)).astype(DTYPE)
    A_log = mx.log(mx.random.uniform(low=0.0, high=16.0, shape=(HV,)))
    dt_bias = mx.ones((HV,))

    # 実運用 (qwen3_5.py / fastmlx/spec.py) と同じく q, k を rms_norm してから渡す。
    # これをしないと再帰的な状態更新が T が伸びるにつれ発散し (fp16 で inf/nan)、
    # 「両実装が同じ値を返すか」という検証の意味がなくなる。
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    q = q.astype(DTYPE)
    k = k.astype(DTYPE)

    mx.eval(q, k, v, a, b, A_log, dt_bias)
    return q, k, v, a, b, A_log, dt_bias


def max_abs_diff(x, y) -> float:
    return mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max().item()


def max_rel_diff(x, y) -> float:
    x32 = x.astype(mx.float32)
    y32 = y.astype(mx.float32)
    denom = mx.maximum(mx.abs(x32), mx.abs(y32))
    denom = mx.maximum(denom, mx.array(1e-6))
    return (mx.abs(x32 - y32) / denom).max().item()


def check_out_matches_batch(T: int) -> None:
    """(a) out が mlx_lm.gated_delta_update と一致するか。"""
    q, k, v, a, b, A_log, dt_bias = make_inputs(T)

    out_ref, state_ref = gated_delta_update(q, k, v, a, b, A_log, dt_bias, None, None)
    out_new, states_all = gated_delta_update_with_states(
        q, k, v, a, b, A_log, dt_bias, None, None
    )
    mx.eval(out_ref, state_ref, out_new, states_all)

    diff = max_abs_diff(out_ref, out_new)
    rel = max_rel_diff(out_ref, out_new)
    exact = bool(mx.array_equal(out_ref, out_new))
    print(
        f"  [T={T}] (a) out: max_abs_diff={diff:.3e} max_rel_diff={rel:.3e} "
        f"bit_exact={exact}"
    )
    assert diff < 1e-2, f"out mismatch too large at T={T}: {diff}"


def check_states_against_sequential(T: int) -> None:
    """(b) states_all[:, t] が逐次呼び (t+1 トークンごとに区切って呼ぶ) と一致するか。
    (c) 最終状態が一括呼びの返り状態と一致するか。
    """
    q, k, v, a, b, A_log, dt_bias = make_inputs(T)

    _, states_all = gated_delta_update_with_states(
        q, k, v, a, b, A_log, dt_bias, None, None
    )
    mx.eval(states_all)

    # 逐次呼び: 位置ごとに gated_delta_update(mlx_lm 版) を呼び、状態を引き継ぐ
    state = None
    seq_states = []
    for t in range(T):
        _, state = gated_delta_update(
            q[:, t : t + 1],
            k[:, t : t + 1],
            v[:, t : t + 1],
            a[:, t : t + 1],
            b[:, t : t + 1],
            A_log,
            dt_bias,
            state,
            None,
        )
        seq_states.append(state)
    mx.eval(*seq_states)

    max_diff_over_t = 0.0
    for t in range(T):
        d = max_abs_diff(states_all[:, t], seq_states[t])
        max_diff_over_t = max(max_diff_over_t, d)
    print(f"  [T={T}] (b) states_all vs sequential: max_abs_diff over all t={max_diff_over_t:.3e}")
    assert max_diff_over_t < 1e-3, f"state mismatch too large at T={T}: {max_diff_over_t}"

    # (c) 一括呼び (mlx_lm) の最終状態と states_all[:, -1] を比較
    _, state_batch = gated_delta_update(q, k, v, a, b, A_log, dt_bias, None, None)
    mx.eval(state_batch)
    d_final = max_abs_diff(states_all[:, -1], state_batch)
    print(f"  [T={T}] (c) states_all[:, -1] vs batch final state: max_abs_diff={d_final:.3e}")
    assert d_final < 1e-3, f"final state mismatch too large at T={T}: {d_final}"

    # 追加の健全性確認: states_all[:, -1] は逐次呼びの最終状態とも一致するはず
    d_seq_final = max_abs_diff(states_all[:, -1], seq_states[-1])
    print(f"  [T={T}]     (sanity) states_all[:, -1] vs sequential final: max_abs_diff={d_seq_final:.3e}")


def check_mask_branch(T: int) -> None:
    """おまけ: mask 付きの分岐も一致するかの簡易確認 (仕様の必須項目ではない)。"""
    q, k, v, a, b, A_log, dt_bias = make_inputs(T)
    mask = mx.array([[True] * (T - 1) + [False]]) if T > 1 else mx.array([[True]])

    out_ref, state_ref = gated_delta_update(q, k, v, a, b, A_log, dt_bias, None, mask)
    out_new, states_all = gated_delta_update_with_states(
        q, k, v, a, b, A_log, dt_bias, None, mask
    )
    mx.eval(out_ref, state_ref, out_new, states_all)

    diff_out = max_abs_diff(out_ref, out_new)
    diff_state = max_abs_diff(state_ref, states_all[:, -1])
    print(
        f"  [T={T}] (mask sanity) out max_abs_diff={diff_out:.3e} "
        f"final_state max_abs_diff={diff_state:.3e}"
    )
    assert diff_out < 1e-2
    assert diff_state < 1e-3


def check_shape_guards_and_ops_fallback() -> None:
    """Unsupported tile widths fall back; inconsistent shapes fail before launch."""

    B_, T_, Hk_, Hv_, Dk_, Dv_ = 1, 3, 2, 4, 80, 8
    q = mx.random.normal((B_, T_, Hk_, Dk_)).astype(DTYPE)
    k = mx.random.normal((B_, T_, Hk_, Dk_)).astype(DTYPE)
    v = mx.random.normal((B_, T_, Hv_, Dv_)).astype(DTYPE)
    a = mx.random.normal((B_, T_, Hv_)).astype(DTYPE)
    b = mx.random.normal((B_, T_, Hv_)).astype(DTYPE)
    A_log = mx.log(mx.random.uniform(low=0.1, high=2.0, shape=(Hv_,)))
    dt_bias = mx.ones((Hv_,))
    mask = mx.array([[True, False, True]])

    out, states = gated_delta_update_with_states(
        q, k, v, a, b, A_log, dt_bias, None, mask
    )
    state_ref = None
    out_ref = []
    states_ref = []
    for t in range(T_):
        yt, state_ref = gated_delta_update(
            q[:, t : t + 1],
            k[:, t : t + 1],
            v[:, t : t + 1],
            a[:, t : t + 1],
            b[:, t : t + 1],
            A_log,
            dt_bias,
            state_ref,
            mask[:, t : t + 1],
            use_kernel=False,
        )
        # mlx-lm 0.31.3's ops path restores masked state but does not zero y;
        # the Metal contract does, so normalize the reference output here.
        yt = mx.where(mask[:, t : t + 1, None, None], yt, 0)
        out_ref.append(yt)
        states_ref.append(state_ref)
    out_ref = mx.concatenate(out_ref, axis=1)
    states_ref = mx.stack(states_ref, axis=1)
    mx.eval(out, states, out_ref, states_ref)
    assert max_abs_diff(out, out_ref) < 1e-2
    assert max_abs_diff(states, states_ref) < 1e-3

    bad_state = mx.zeros((B_, Hv_, Dv_, Dk_ - 1), dtype=mx.float32)
    bad_mask = mx.ones((B_, T_ + 1), dtype=mx.bool_)
    bad_state_dtype = mx.zeros((B_, Hv_, Dv_, Dk_), dtype=mx.float16)
    bad_mask_dtype = mx.ones((B_, T_), dtype=mx.int32)
    for kwargs in (
        {"state": bad_state, "mask": mask},
        {"state": None, "mask": bad_mask},
        {"state": bad_state_dtype, "mask": mask},
        {"state": None, "mask": bad_mask_dtype},
    ):
        try:
            gated_delta_update_with_states(
                q, k, v, a, b, A_log, dt_bias, kwargs["state"], kwargs["mask"]
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid recurrent shape should fail: {kwargs}")

    bad_v = v[:, :, :1]
    bad_a = a[:, :, :1]
    bad_b = b[:, :, :1]
    try:
        gated_delta_update_with_states(
            q,
            k,
            bad_v,
            bad_a,
            bad_b,
            A_log[:1],
            dt_bias[:1],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Hv < Hk should fail before Metal launch")

    try:
        gated_delta_update_with_states(
            q,
            k,
            v[:, :, :3],
            a[:, :, :3],
            b[:, :, :3],
            A_log[:3],
            dt_bias[:3],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Hv % Hk != 0 should fail before Metal launch")


def check_mask_false_is_deterministic(T: int = 4) -> None:
    q, k, v, a, b, A_log, dt_bias = make_inputs(T)
    mask = mx.zeros((B, T), dtype=mx.bool_)
    first_out, first_states = gated_delta_update_with_states(
        q, k, v, a, b, A_log, dt_bias, None, mask
    )
    mx.eval(first_out, first_states)
    assert not bool(mx.any(first_out).item())
    for _ in range(8):
        out, states = gated_delta_update_with_states(
            q, k, v, a, b, A_log, dt_bias, None, mask
        )
        mx.eval(out, states)
        assert bool(mx.array_equal(out, first_out))
        assert bool(mx.array_equal(states, first_states))


def bench_once(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def run_perf_comparison(T: int = 8, iters: int = 10) -> None:
    q, k, v, a, b, A_log, dt_bias = make_inputs(T)

    def fused():
        return gated_delta_update_with_states(q, k, v, a, b, A_log, dt_bias, None, None)

    def sequential():
        state = None
        out = None
        for t in range(T):
            out, state = gated_delta_update(
                q[:, t : t + 1],
                k[:, t : t + 1],
                v[:, t : t + 1],
                a[:, t : t + 1],
                b[:, t : t + 1],
                A_log,
                dt_bias,
                state,
                None,
            )
        return out, state

    t_fused = bench_once(fused, iters=iters)
    t_seq = bench_once(sequential, iters=iters)
    print(f"  T={T}: fused (1 launch, states出力込み) = {t_fused * 1e3:.3f} ms/call")
    print(f"  T={T}: sequential ({T} launches)         = {t_seq * 1e3:.3f} ms/call")
    if t_fused > 0:
        print(f"  speedup (sequential / fused) = {t_seq / t_fused:.2f}x  (参考値)")


def main() -> None:
    if not mx.metal.is_available():
        print("Metal not available; skipping (this kernel is GPU-only).")
        return

    print("=== 検証1: 正しさ ===")
    for T in (1, 4, 8):
        print(f"-- T={T} --")
        check_out_matches_batch(T)
        check_states_against_sequential(T)
        check_mask_branch(T)
    print("-- shape guards / ops fallback --")
    check_shape_guards_and_ops_fallback()
    print("-- mask=false deterministic writes --")
    check_mask_false_is_deterministic()

    print()
    print("=== 検証2: 性能 (参考値、単発比較) ===")
    run_perf_comparison(T=8, iters=10)

    print()
    print("すべての検証を通過しました。")


if __name__ == "__main__":
    main()
