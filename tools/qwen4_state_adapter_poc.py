"""Qwen4 Flash-Next の fixed-M4 state adapter を合成形状で検査する。

現行 ``_AttnCache`` の可変な Python オブジェクトを直接 ``mx.compile`` に
閉じ込めるのではなく、次の5葉へ写す最小形だけを扱う:

    [K, V, array offset, raw index, pooled index]

K/V は ``KVCache`` の配列、array offset はその論理長、raw/pooled index は
``_IndexerCache`` の生キーと確定済みブロックキーに対応する。実際の
Attention は呼ばない。全葉を固定容量にし、更新は state-in/state-out の純関数
としてだけ表す。

検査するもの:

* Python int の offset を closure に置いた実装は、同じ形の2回目の compile
  replay で最初の位置へ書き戻すこと (stale 対照)。
* offset を tensor の state leaf として渡す実装は、2回目も次の位置へ書くこと。
* tensor state の rollback が keep=1/3/4 の各々で5葉すべてを期待値と一致させる
  こと。rollback は未使用域をゼロまたは sentinel に戻し、論理 state として
  比較できる形にする。

GPU/MLX を使う実行例:

    tools/biglock.sh .venv/bin/python tools/qwen4_state_adapter_poc.py

stdout は機械可読な JSON 1行、成功は exit 0、検査失敗は exit 1、MLX 不在など
実行不能は exit 2。
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np

try:
    import mlx.core as mx
except Exception as exc:  # pragma: no cover - 実行環境の診断用
    mx = None
    _MLX_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _MLX_IMPORT_ERROR = None


TASK_ID = "qwen4_state_adapter_poc_0905"

# 合成の固定形状。M=4 は fixed-M4 の更新幅、RATIO=2 は indexer の pooled
# block を小さく表すためのもの。実モデルの寸法を再現する目的ではない。
BATCH = 1
KV_HEADS = 2
HEAD_DIM = 4
RAW_DIM = 3
CAPACITY = 16
POOL_RATIO = 2
POOL_CAPACITY = CAPACITY // POOL_RATIO
STEP = 4
POOL_STEP = STEP // POOL_RATIO


State = tuple[Any, Any, Any, Any, Any]
Payload = tuple[Any, Any, Any, Any]


def _initial_numpy_state() -> State:
    """固定容量stateのホスト側期待値を作る。"""

    return (
        np.zeros((BATCH, KV_HEADS, CAPACITY, HEAD_DIM), dtype=np.float32),
        np.zeros((BATCH, KV_HEADS, CAPACITY, HEAD_DIM), dtype=np.float32),
        np.zeros((1,), dtype=np.int32),
        np.full((BATCH, CAPACITY, RAW_DIM), -1, dtype=np.int32),
        np.full((BATCH, POOL_CAPACITY, RAW_DIM), -1, dtype=np.int32),
    )


def _payload_numpy(seed: int) -> Payload:
    """同じshapeで書き込み位置を区別できる入力を作る。"""

    k = (seed + np.arange(BATCH * KV_HEADS * STEP * HEAD_DIM, dtype=np.float32))
    k = (k.reshape(BATCH, KV_HEADS, STEP, HEAD_DIM) / 100.0).astype(np.float32)
    v = (seed + 1000.0 + np.arange(BATCH * KV_HEADS * STEP * HEAD_DIM, dtype=np.float32))
    v = (v.reshape(BATCH, KV_HEADS, STEP, HEAD_DIM) / 100.0).astype(np.float32)
    raw = (
        seed * 100
        + np.arange(BATCH * STEP * RAW_DIM, dtype=np.int32).reshape(BATCH, STEP, RAW_DIM)
    )
    pooled = (
        seed * 1000
        + np.arange(BATCH * POOL_STEP * RAW_DIM, dtype=np.int32).reshape(
            BATCH, POOL_STEP, RAW_DIM
        )
    )
    return k, v, raw, pooled


def _host_append(state: State, payload: Payload) -> State:
    """純関数版の比較用。Python側の期待値だけを組み立てる。"""

    k, v, offset, raw, pooled = (np.array(x, copy=True) for x in state)
    new_k, new_v, new_raw, new_pooled = payload
    start = int(offset[0])
    k[:, :, start : start + STEP, :] = new_k
    v[:, :, start : start + STEP, :] = new_v
    raw[:, start : start + STEP, :] = new_raw
    pool_start = start // POOL_RATIO
    pooled[:, pool_start : pool_start + POOL_STEP, :] = new_pooled
    offset[0] = start + STEP
    return k, v, offset, raw, pooled


def _host_rollback(state: State, keep: int) -> State:
    """論理 rollback の比較用。未使用域も明示的に消去する。"""

    k, v, offset, raw, pooled = (np.array(x, copy=True) for x in state)
    k[:, :, keep:, :] = 0.0
    v[:, :, keep:, :] = 0.0
    raw[:, keep:, :] = -1
    pooled[:, keep // POOL_RATIO :, :] = -1
    offset[0] = keep
    return k, v, offset, raw, pooled


def _mx_state(state: State) -> State:
    return tuple(mx.array(x) for x in state)  # type: ignore[return-value]


def _mx_payload(payload: Payload) -> Payload:
    return tuple(mx.array(x) for x in payload)  # type: ignore[return-value]


def _write_kv(old: Any, incoming: Any, start: Any) -> Any:
    """KV の sequence 軸へ、tensor start から固定幅を一度だけ書く。"""

    capacity = old.shape[2]
    width = incoming.shape[2]
    positions = mx.arange(capacity, dtype=mx.int32)
    slots = start + mx.arange(width, dtype=mx.int32)
    mask = (positions[:, None] == slots[None, :]).astype(incoming.dtype)
    updates = mx.sum(
        mask[None, None, :, :, None] * incoming[:, :, None, :, :], axis=3
    )
    live = (positions >= start) & (positions < start + width)
    return mx.where(live[None, None, :, None], updates, old)


def _write_index(old: Any, incoming: Any, start: Any) -> Any:
    """raw/pooled index の sequence 軸へ固定幅を書き込む。"""

    capacity = old.shape[1]
    width = incoming.shape[1]
    positions = mx.arange(capacity, dtype=mx.int32)
    slots = start + mx.arange(width, dtype=mx.int32)
    mask = (positions[:, None] == slots[None, :]).astype(incoming.dtype)
    updates = mx.sum(mask[None, :, :, None] * incoming[:, None, :, :], axis=2)
    live = (positions >= start) & (positions < start + width)
    return mx.where(live[None, :, None], updates, old)


def _tensor_update(
    k_state: Any,
    v_state: Any,
    offset_state: Any,
    raw_state: Any,
    pooled_state: Any,
    new_k: Any,
    new_v: Any,
    new_raw: Any,
    new_pooled: Any,
) -> State:
    """5葉を引数に取り、5葉を返す純関数 state-in/state-out 更新。"""

    start = offset_state[0]
    return (
        _write_kv(k_state, new_k, start),
        _write_kv(v_state, new_v, start),
        offset_state + mx.array([STEP], dtype=offset_state.dtype),
        _write_index(raw_state, new_raw, start),
        _write_index(pooled_state, new_pooled, start // POOL_RATIO),
    )


def _closure_update_factory() -> Any:
    """Python int offset を閉じ込めた stale 対照を返す。"""

    offset = 0

    def closure_update(
        k_state: Any,
        v_state: Any,
        offset_state: Any,
        raw_state: Any,
        pooled_state: Any,
        new_k: Any,
        new_v: Any,
        new_raw: Any,
        new_pooled: Any,
    ) -> State:
        nonlocal offset
        # offset_state は意図的に読まず、Python int の offset をグラフへ焼く。
        start = mx.array(offset, dtype=mx.int32)
        out = (
            _write_kv(k_state, new_k, start),
            _write_kv(v_state, new_v, start),
            mx.array([offset + STEP], dtype=offset_state.dtype),
            _write_index(raw_state, new_raw, start),
            _write_index(pooled_state, new_pooled, start // POOL_RATIO),
        )
        offset += STEP
        return out

    return closure_update


def _rollback_state(
    k_state: Any,
    v_state: Any,
    offset_state: Any,
    raw_state: Any,
    pooled_state: Any,
    keep_state: Any,
) -> State:
    """tensor keep を受け、固定容量の5葉を論理長までに巻き戻す。"""

    token_positions = mx.arange(CAPACITY, dtype=mx.int32)
    pool_positions = mx.arange(POOL_CAPACITY, dtype=mx.int32)
    token_live = token_positions < keep_state[0]
    pool_live = pool_positions < (keep_state[0] // POOL_RATIO)
    return (
        mx.where(
            token_live[None, None, :, None],
            k_state,
            mx.zeros(k_state.shape, dtype=k_state.dtype),
        ),
        mx.where(
            token_live[None, None, :, None],
            v_state,
            mx.zeros(v_state.shape, dtype=v_state.dtype),
        ),
        keep_state.astype(offset_state.dtype),
        mx.where(
            token_live[None, :, None],
            raw_state,
            mx.full(raw_state.shape, -1, dtype=raw_state.dtype),
        ),
        mx.where(
            pool_live[None, :, None],
            pooled_state,
            mx.full(pooled_state.shape, -1, dtype=pooled_state.dtype),
        ),
    )


def _all_equal(got: State, expected: State) -> bool:
    """5葉を遅延評価込みで完全一致比較する。"""

    mx.eval(*(got + expected))
    checks = tuple(mx.all(a == b) for a, b in zip(got, expected))
    mx.eval(*checks)
    return all(bool(x) for x in checks)


def _same_shapes(left: State, right: State) -> bool:
    return all(a.shape == b.shape for a, b in zip(left, right))


def _run_checks() -> dict[str, Any]:
    initial_np = _initial_numpy_state()
    payload1_np = _payload_numpy(1)
    payload2_np = _payload_numpy(2)
    initial = _mx_state(initial_np)
    payload1 = _mx_payload(payload1_np)
    payload2 = _mx_payload(payload2_np)

    expected1_np = _host_append(initial_np, payload1_np)
    expected2_np = _host_append(expected1_np, payload2_np)
    expected1 = _mx_state(expected1_np)
    expected2 = _mx_state(expected2_np)

    # 対照: first trace で offset=0 が焼かれ、replay は offset=4 を読まない。
    closure_fn = mx.compile(_closure_update_factory())
    closure_first = closure_fn(*initial, *payload1)
    closure_second = closure_fn(*closure_first, *payload2)
    closure_first_ok = _all_equal(closure_first, expected1)
    closure_second_ok = _all_equal(closure_second, expected2)
    closure_shape_stable = _same_shapes(closure_first, initial) and _same_shapes(
        closure_second, initial
    )

    # 本命: array offset を5葉のstate-in/state-outに含める。
    tensor_fn = mx.compile(_tensor_update)
    tensor_first = tensor_fn(*initial, *payload1)
    tensor_second = tensor_fn(*tensor_first, *payload2)
    tensor_first_ok = _all_equal(tensor_first, expected1)
    tensor_second_ok = _all_equal(tensor_second, expected2)
    tensor_shape_stable = _same_shapes(tensor_first, initial) and _same_shapes(
        tensor_second, initial
    )

    # 同じrollbackグラフを keep=1/3/4 でreplayする。各入力はfull stateから
    # 独立に作るので、keepの順番による副作用を混ぜない。
    rollback_fn = mx.compile(_rollback_state)
    rollback_results: dict[str, Any] = {}
    rollback_ok = True
    for keep in (1, 3, 4):
        got = rollback_fn(*tensor_first, mx.array([keep], dtype=mx.int32))
        expected = _mx_state(_host_rollback(expected1_np, keep))
        ok = _all_equal(got, expected)
        rollback_results[str(keep)] = {
            "all_five_leaves_equal": ok,
            "shape_stable": _same_shapes(got, tensor_first),
        }
        rollback_ok &= ok and rollback_results[str(keep)]["shape_stable"]

    return {
        "task_id": TASK_ID,
        "status": "PASS"
        if (
            closure_first_ok
            and not closure_second_ok
            and closure_shape_stable
            and tensor_first_ok
            and tensor_second_ok
            and tensor_shape_stable
            and rollback_ok
        )
        else "FAIL",
        "state_leaves": ["K", "V", "array_offset", "raw_index", "pooled_index"],
        "fixed_shape": {
            "K": [BATCH, KV_HEADS, CAPACITY, HEAD_DIM],
            "V": [BATCH, KV_HEADS, CAPACITY, HEAD_DIM],
            "array_offset": [1],
            "raw_index": [BATCH, CAPACITY, RAW_DIM],
            "pooled_index": [BATCH, POOL_CAPACITY, RAW_DIM],
        },
        "checks": {
            "python_int_offset_closure": {
                "first_replay_correct": closure_first_ok,
                "second_replay_stale": not closure_second_ok,
                "second_replay_matches_sequential_expected": closure_second_ok,
                "same_shape_replay": closure_shape_stable,
            },
            "tensor_offset_state_in_out": {
                "first_replay_correct": tensor_first_ok,
                "second_replay_correct": tensor_second_ok,
                "same_shape_replay": tensor_shape_stable,
            },
            "rollback": rollback_results,
        },
    }


def main() -> int:
    if mx is None:
        print(
            json.dumps(
                {
                    "task_id": TASK_ID,
                    "status": "BLOCKED",
                    "reason": "MLX import failed",
                    "error": _MLX_IMPORT_ERROR,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2

    try:
        result = _run_checks()
    except Exception as exc:  # pragma: no cover - 実行時診断をJSONにする
        result = {
            "task_id": TASK_ID,
            "status": "FAIL",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
