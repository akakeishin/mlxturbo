import mlx.core as mx

from mlxturbo.spec_flash import _retain_hyper_tail


def test_hyper_tail_uses_current_chunk_when_it_already_covers_window():
    previous = mx.arange(8).reshape(1, 8, 1)
    current = mx.arange(8, 20).reshape(1, 12, 1)

    retained = _retain_hyper_tail(previous, current, 6)
    mx.eval(retained)

    assert retained.shape == (1, 6, 1)
    assert retained.reshape(-1).tolist() == [14, 15, 16, 17, 18, 19]


def test_hyper_tail_concatenates_only_missing_rows_from_previous_tail():
    previous = mx.arange(8).reshape(1, 8, 1)
    current = mx.arange(8, 10).reshape(1, 2, 1)

    retained = _retain_hyper_tail(previous, current, 6)
    mx.eval(retained)

    assert retained.shape == (1, 6, 1)
    assert retained.reshape(-1).tolist() == [4, 5, 6, 7, 8, 9]
