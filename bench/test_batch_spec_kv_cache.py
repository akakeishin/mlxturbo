"""ragged batch投機のKV予約領域を検査する。"""

import mlx.core as mx
import pytest

from mlxturbo.batch_spec import (
    RaggedAttnCache,
    RaggedDraftCache,
    RaggedLedger,
    _KV_CACHE_STEP,
)


@pytest.mark.parametrize(
    "factory",
    [lambda: RaggedAttnCache(RaggedLedger(2)), lambda: RaggedDraftCache(2)],
)
def test_ragged_kv_cache_reuses_capacity_between_boundaries(factory):
    cache = factory()
    expected = []
    for value in (1, 2, 3):
        x = mx.full((2, 1, 1, 4), value, dtype=mx.float32)
        keys, values = cache.update_and_fetch(x, x + 10)
        expected.append(value)
        mx.eval(keys, values)

        assert cache.size() == len(expected)
        assert keys.shape[2] == len(expected)
        assert cache.keys.shape[2] == _KV_CACHE_STEP
        assert keys[0, 0, :, 0].tolist() == expected
        assert values[0, 0, :, 0].tolist() == [v + 10 for v in expected]


def test_ragged_draft_trim_then_append_preserves_prefix():
    cache = RaggedDraftCache(1)
    first = mx.arange(5, dtype=mx.float32).reshape(1, 1, 5, 1)
    cache.update_and_fetch(first, first + 10)
    assert cache.trim(2) == 2
    assert cache.size() == 3

    keys, values = cache.update_and_fetch(
        mx.array([[[[9.0]]]]), mx.array([[[[19.0]]]])
    )
    mx.eval(keys, values)

    assert cache.size() == 4
    assert cache.keys.shape[2] == _KV_CACHE_STEP
    assert keys.reshape(-1).tolist() == [0.0, 1.0, 2.0, 9.0]
    assert values.reshape(-1).tolist() == [10.0, 11.0, 12.0, 19.0]
