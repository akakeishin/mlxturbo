"""GatedDeltaNet カーネルの派生版: 全位置の再帰状態も一括で出力する。

投機デコードの検証パス（`fastmlx/spec.py` の `_linear_capture`）は、巻き戻しの
ために「各位置を処理し終えた直後の状態」を必要としている。現状は
`mlx_lm.models.gated_delta.gated_delta_update` を位置ごとに T 回呼んでおり、
T 回のカーネル起動と、呼び出しごとの state (fp32, [B, Hv, Dv, Dk]) の
読み書きが発生する。

このモジュールは mlx_lm のカーネル文字列 (`gated_delta_kernel` /
`_make_gated_delta_kernel`) を土台に、時間ループの中で state を
`states_all[:, t]` へも書き足すだけの改造を加えた自前カーネルを提供する。
1 simdgroup が状態の 1 行を担当し、状態はレジスタに常駐したまま T ステップを
カーネル内でループする構造は変えていない。

mlx_lm 側のソースからの変更点はこれだけ:
  - 出力に `states_all` ([B, T, Hv, Dv, Dk], fp32) を追加
  - 時間ループの各ステップの末尾（ポインタを次の t へ進める直前）で、
    そのステップの state をレジスタから `states_all` へ書き出す
  - 上記に伴うポインタ (`sall_`) のセットアップと前進を追加
それ以外のロジック（decay・delta 更新・出力射影・マスク処理・state_in/state_out
の扱い）は mlx_lm の実装をそのまま踏襲している。
"""

from typing import Optional, Tuple

import mlx.core as mx

from .._mlx_compat import compute_g


def _make_gated_delta_states_kernel(has_mask: bool = False, vectorized: bool = False):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        // states_all: [B, T, Hv, Dv, Dk] (fp32). states_all[:, t] is the state
        // immediately after position t has been processed.
        auto sall_ = states_all + ((b_idx * T * Hv + hv_idx) * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * {g_access};
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(0);
            }}
          }}
          // Record the state right after this position was processed, for
          // every position (this is the only substantive addition vs.
          // mlx_lm's gated_delta_step kernel).
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            sall_[s_idx] = state[i];
          }}
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          sall_ += Hv * Dv * Dk;
          {g_advance}
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_states_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out", "states_all"],
        source=source,
    )


_gated_delta_states_kernel = _make_gated_delta_states_kernel(
    has_mask=False, vectorized=False
)
_gated_delta_states_kernel_masked = _make_gated_delta_states_kernel(
    has_mask=True, vectorized=False
)
_gated_delta_states_kernel_vec = _make_gated_delta_states_kernel(
    has_mask=False, vectorized=True
)
_gated_delta_states_kernel_vec_masked = _make_gated_delta_states_kernel(
    has_mask=True, vectorized=True
)


def gated_delta_kernel_with_states(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    _validate_kernel_shapes(q, k, v, g, beta, state, mask)
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk % 32 != 0 or mx.default_device() != mx.gpu or not mx.metal.is_available():
        out, states_all = _gated_delta_ops_with_states(
            q, k, v, g, beta, state, mask
        )
        return out, states_all[:, -1], states_all
    input_type = q.dtype
    state_type = state.dtype
    if g.ndim == 4:
        kernel = _gated_delta_states_kernel_vec
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_states_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _gated_delta_states_kernel
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_states_kernel_masked
            inputs.append(mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape, (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type, mx.float32],
    )


def _validate_kernel_shapes(q, k, v, g, beta, state, mask) -> None:
    if q.ndim != 4 or k.ndim != 4 or q.shape != k.shape:
        raise ValueError(
            f"q and k must have the same [B,T,Hk,Dk] shape; "
            f"got {q.shape}, {k.shape}"
        )
    B, T, Hk, Dk = q.shape
    if min(B, T, Hk, Dk) <= 0:
        raise ValueError(f"q/k dimensions must be positive; got {q.shape}")
    if v.ndim != 4 or v.shape[:2] != (B, T):
        raise ValueError(f"v must have shape [B,T,Hv,Dv]; got {v.shape}")
    Hv, Dv = v.shape[2:]
    if Hv < Hk or Hv % Hk != 0:
        raise ValueError(
            f"Hv must be a positive multiple of Hk; got Hk={Hk}, Hv={Hv}"
        )
    if beta.shape != (B, T, Hv):
        raise ValueError(f"beta must have shape {(B, T, Hv)}; got {beta.shape}")
    valid_g_shapes = ((B, T, Hv), (B, T, Hv, Dk))
    if g.shape not in valid_g_shapes:
        raise ValueError(
            f"g must have shape {valid_g_shapes[0]} or {valid_g_shapes[1]}; "
            f"got {g.shape}"
        )
    expected_state = (B, Hv, Dv, Dk)
    if state.shape != expected_state:
        raise ValueError(f"state must have shape {expected_state}; got {state.shape}")
    if state.dtype != mx.float32:
        raise ValueError(f"state must use float32 accumulation; got {state.dtype}")
    if mask is not None and mask.shape != (B, T):
        raise ValueError(f"mask must have shape {(B, T)}; got {mask.shape}")
    if mask is not None and mask.dtype != mx.bool_:
        raise ValueError(f"mask must use bool dtype; got {mask.dtype}")


def _gated_delta_ops_with_states(q, k, v, g, beta, state, mask=None):
    """Shape-general MLX-ops fallback that also records every recurrent state."""

    Hk = q.shape[2]
    Hv = v.shape[2]
    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    outputs = []
    states = []
    for t in range(q.shape[1]):
        old_state = state
        gt = g[:, t]
        decay = gt[..., None, None] if gt.ndim == 2 else gt[..., None, :]
        state = state * decay
        kv_mem = (state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        state = state + k[:, t, :, None, :] * delta[..., None]
        out = (state * q[:, t, :, None, :]).sum(axis=-1)
        if mask is not None:
            mt = mask[:, t]
            state = mx.where(mt[:, None, None, None], state, old_state)
            out = mx.where(mt[:, None, None], out, 0)
        outputs.append(out.astype(q.dtype))
        states.append(state.astype(mx.float32))
    return mx.stack(outputs, axis=1), mx.stack(states, axis=1)


def gated_delta_update_with_states(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """`mlx_lm.models.gated_delta.gated_delta_update` と同じ入出力の意味を持つが、

    位置ごとの状態も一括で返す。

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - a, b: [B, T, Hv]
      - state: [B, Hv, Dv, Dk] (fp32) or None
    Returns:
      - out: [B, T, Hv, Dv] -- gated_delta_update と同一
      - states_all: [B, T, Hv, Dv, Dk] (fp32).
        states_all[:, t] は位置 t を処理し終えた直後の状態。
        states_all[:, -1] は一括呼び出しの最終状態 (state_out) と一致する。
    """
    if q.ndim != 4 or k.ndim != 4 or q.shape != k.shape:
        raise ValueError(
            f"q and k must have the same [B,T,Hk,Dk] shape; got {q.shape}, {k.shape}"
        )
    B, T, _Hk, Dk = q.shape
    if v.ndim != 4 or v.shape[:2] != (B, T):
        raise ValueError(f"v must have shape [B,T,Hv,Dv]; got {v.shape}")
    Hv, Dv = v.shape[-2:]
    if a.shape != (B, T, Hv) or b.shape != (B, T, Hv):
        raise ValueError(
            f"a and b must have shape {(B, T, Hv)}; got {a.shape}, {b.shape}"
        )
    if A_log.shape != (Hv,) or dt_bias.shape != (Hv,):
        raise ValueError(
            f"A_log and dt_bias must have shape {(Hv,)}; "
            f"got {A_log.shape}, {dt_bias.shape}"
        )
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    out, _state_out, states_all = gated_delta_kernel_with_states(
        q, k, v, g, beta, state, mask
    )
    return out, states_all
