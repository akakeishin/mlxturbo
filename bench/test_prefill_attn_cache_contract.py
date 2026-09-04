"""`prefill_attn` が既知の連続KVバッファだけを直接読む契約。"""

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from mlxturbo.kernels import prefill_attn


def _filled(cache):
    k = mx.zeros((1, 2, 5, 8), dtype=mx.bfloat16)
    v = mx.zeros((1, 2, 5, 8), dtype=mx.bfloat16)
    k_view, v_view = cache.update_and_fetch(k, v)
    return k_view, v_view


def test_base_kv_cache_backing_buffer_is_accepted():
    cache = KVCache()
    k, v = _filled(cache)

    got = prefill_attn._kv_buffers(cache, k, v, kv_len=5)

    assert got is not None
    keys, values, capacity = got
    assert keys is cache.keys
    assert values is cache.values
    assert capacity == cache.keys.shape[2]
    assert capacity > 5


def test_inherited_kv_cache_contract_is_accepted():
    class CacheWithIndexer(KVCache):
        pass

    cache = CacheWithIndexer()
    k, v = _filled(cache)

    assert prefill_attn._kv_buffers(cache, k, v, kv_len=5) is not None


def test_overridden_update_contract_is_rejected():
    class ViewBackedCache(KVCache):
        def update_and_fetch(self, keys, values):
            return super().update_and_fetch(keys, values)

    cache = ViewBackedCache()
    k, v = _filled(cache)

    assert prefill_attn._kv_buffers(cache, k, v, kv_len=5) is None


def test_duck_typed_cache_is_rejected_even_with_matching_shapes():
    class DuckCache:
        pass

    cache = DuckCache()
    cache.keys = mx.zeros((1, 2, 256, 8), dtype=mx.bfloat16)
    cache.values = mx.zeros((1, 2, 256, 8), dtype=mx.bfloat16)
    cache.offset = 5
    k = cache.keys[..., :5, :]
    v = cache.values[..., :5, :]

    assert prefill_attn._kv_buffers(cache, k, v, kv_len=5) is None
