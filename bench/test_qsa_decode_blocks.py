from mlxturbo.kernels.qsa_attn_decode import decode_blocks


def test_decode_blocks_uses_64_only_in_17k_band(monkeypatch):
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    monkeypatch.delenv("MLXTURBO_QSA_BLOCKS64", raising=False)

    assert decode_blocks(2048, 12, 3, "s") == 128
    assert decode_blocks(4096, 12, 3, "s") == 128
    assert decode_blocks(15999, 12, 3, "s") == 256
    assert decode_blocks(16000, 12, 3, "s") == 64
    assert decode_blocks(18000, 12, 3, "s") == 64
    assert decode_blocks(18001, 12, 3, "s") == 256
    assert decode_blocks(50000, 12, 3, "s") == 512


def test_decode_blocks_can_restore_mlx_table(monkeypatch):
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    monkeypatch.setenv("MLXTURBO_QSA_BLOCKS64", "0")

    assert decode_blocks(17000, 12, 3, "s") == 256


def test_explicit_mlx_blocks_pin_wins(monkeypatch):
    monkeypatch.setenv("MLX_SDPA_BLOCKS", "128")
    monkeypatch.delenv("MLXTURBO_QSA_BLOCKS64", raising=False)

    assert decode_blocks(17000, 12, 3, "s") == 128
