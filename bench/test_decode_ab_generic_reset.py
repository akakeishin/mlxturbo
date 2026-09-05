import pytest

from tools import decode_ab_generic


def test_result_row_preserves_acceptance_trace():
    row = decode_ab_generic._result_row(
        {
            "tokens": [1, 2, 3, 4],
            "steps": 2,
            "accept_hist": {0: 1, 2: 1},
            "accept_trace": [0, 2],
            "src_hist": {"mtp": {0: 1, 2: 1}, "lookup": {}},
            "ttft_s": 0.25,
        },
        wall=1.25,
        resumed=False,
    )

    assert row["accepted"] == 2
    assert row["accept_hist"] == {0: 1, 2: 1}
    assert row["accept_trace"] == [0, 2]
    assert row["src_hist"] == {"mtp": {0: 1, 2: 1}, "lookup": {}}


def test_parse_args_accepts_three_long_cases():
    args, name, variants, baseline = decode_ab_generic.parse_args([
        "--model", "dummy", "--knob", "FLAG=1,0", "--ctx", "17000",
        "--long-count", "3",
    ])
    assert args.long_count == 3
    assert (name, variants, baseline) == ("FLAG", ["1", "0"], "0")


def test_parse_args_rejects_zero_long_cases():
    with pytest.raises(SystemExit):
        decode_ab_generic.parse_args([
            "--model", "dummy", "--knob", "FLAG=1,0", "--long-count", "0",
        ])


def test_reset_fusions_disables_generic_sdpa_split(monkeypatch):
    from mlxturbo import fused, gather_attn, indexer_lean, qsa_decode

    calls = []
    fused_noops = [
        "disable_hyper_connection_kernel",
        "disable_hyper_connection_elem",
        "disable_hyper_connection",
        "disable_hyper_connection_prefill_compiled",
        "disable_hc_write",
        "disable_hc_qmm_wide",
        "disable_qmm_wide",
        "disable_moe_verify_gather",
        "disable_moe_decode_fused",
        "disable_moe_combine_fold",
        "disable_moe_grouped_gemm",
        "disable_moe_down_epilogue",
        "disable_moe_route",
        "disable_moe_block_compile",
        "disable_rms_norm_gated",
        "disable_gdn_prework_kernel",
        "disable_gdn_decode_fused",
        "disable_gdn_blocked_kernel",
        "disable_gdn_metal_kernel",
        "disable_gdn_port",
        "disable_sdpa_split",
        "disable_sdpa_rowtile",
        "disable_fast_rope",
        "disable_ple_hoist",
    ]
    for name in fused_noops:
        monkeypatch.setattr(fused, name, lambda *args: None)
    monkeypatch.setattr(gather_attn, "disable_gather_attn", lambda *args: None)
    monkeypatch.setattr(gather_attn, "disable_prefill_attn", lambda *args: None)
    monkeypatch.setattr(qsa_decode, "disable_qsa_decode_kernel", lambda *args: None)
    monkeypatch.setattr(indexer_lean, "disable_indexer_lean", lambda *args: None)
    monkeypatch.setattr(
        fused, "disable_sdpa_split_generic", lambda: calls.append("generic")
    )

    class Guard:
        def restore(self):
            pass

    decode_ab_generic._reset_fusions(object(), Guard(), set())
    assert calls == ["generic"]
