from types import SimpleNamespace

import mlx.core as mx

import mlxturbo  # noqa: F401 - installs the vendored qwen4_exp module
import mlx_lm.models.qwen4_exp as Q

import mlxturbo.spec_flash as spec_flash


def test_partial_reject_restores_preverify_pooled_prefix(monkeypatch):
    cache = Q._AttnCache()
    cache.keys = mx.zeros((1, 1, 15, 2))
    cache.values = mx.zeros((1, 1, 15, 2))
    cache.offset = 12
    cache.indexer.update(mx.arange(24).reshape(1, 12, 2))
    cached_pooled = mx.arange(6).reshape(1, 3, 2)
    cached_fp32 = cached_pooled.astype(mx.float32)
    cache.indexer._pooled = cached_pooled
    cache.indexer._pooled_n = 3
    cache.indexer._pooled_f32 = cached_fp32
    cache.indexer._pooled_f32_n = 3

    layer = SimpleNamespace(
        layer_type="full_attention",
        self_attn=SimpleNamespace(indexer=SimpleNamespace(compress_ratio=4)),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    pre = spec_flash.snapshot_pre(model, [cache])

    cache.offset = 15
    cache.indexer.update(mx.full((1, 3, 2), 99))
    cache.indexer._pooled = mx.arange(8).reshape(1, 4, 2)
    cache.indexer._pooled_n = 4
    cache.indexer._pooled_f32 = cache.indexer._pooled.astype(mx.float32)
    cache.indexer._pooled_f32_n = 4
    monkeypatch.setattr(spec_flash._archmod, "rollback_recurrent", lambda *_args, **_kwargs: None)

    spec_flash.rollback(
        model,
        [cache],
        SimpleNamespace(),
        pre,
        keep=1,
        total=3,
    )

    assert cache.offset == 13
    assert cache.indexer.keys.shape[1] == 13
    assert cache.indexer._pooled is cached_pooled
    assert cache.indexer._pooled_n == 3
    assert cache.indexer._pooled_f32 is cached_fp32
    assert cache.indexer._pooled_f32_n == 3
