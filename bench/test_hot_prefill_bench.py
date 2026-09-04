from __future__ import annotations

import pytest

from bench.hot_prefill_bench import (
    _byte_acceptance,
    _expected_selection,
    _expected_pool_lcp,
    _flash_expected_core_components,
    _input_tps,
    _lcp,
    _next_prompt,
    _replacement_tail,
)


_TINY_FLASH_CONFIG = {
    "layer_types": ["linear_attention", "full_attention"],
    "hidden_size": 8,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 5,
    "linear_key_head_dim": 3,
    "linear_value_head_dim": 6,
    "linear_conv_kernel_dim": 4,
    "hc_count": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 2,
    "indexer_compress_ratio": 2,
    "indexer_budget": 8,
    "ngram_size": 3,
    "ple_layer_ids": [1],
    "ple_conv_kernel_size": 4,
}


def test_pure_append_uses_exact_requested_delta():
    current = [1, 2, 3]
    prompt, expected_lcp = _next_prompt(current, 4, [8, 9], "pure_append", 1)
    assert prompt == [1, 2, 3, 8, 9, 8, 9]
    assert expected_lcp == 3
    assert _lcp(current, prompt) == 3
    assert len(prompt) - expected_lcp == 4


def test_suffix_branches_are_independent_from_the_same_base():
    base = [1, 2, 3]
    branches = []
    branch_id = 0
    for suffix in (0, 16, 64, 256):
        branches.append(
            _next_prompt(base, suffix, [8, 9, 10], "pure_append", 1, branch_id)
        )
        if suffix > 0:
            branch_id += 1

    assert [len(prompt) for prompt, _ in branches] == [3, 19, 67, 259]
    assert [expected_lcp for _, expected_lcp in branches] == [3, 3, 3, 3]
    assert [prompt[:4] for prompt, _ in branches] == [
        [1, 2, 3],
        [1, 2, 3, 8],
        [1, 2, 3, 9],
        [1, 2, 3, 10],
    ]
    assert all(_lcp(base, prompt) == 3 for prompt, _ in branches)


def test_reset_lcp_uses_the_previous_measured_prompt():
    base = [1, 2, 3, 4, 5]

    pure_prompt, _ = _next_prompt(base, 16, [8, 9], "pure_append", 2, 0)
    assert _lcp(pure_prompt, base) == len(base)
    assert _expected_pool_lcp(pure_prompt, base) == len(base) - 1
    assert _expected_pool_lcp(base, base) == len(base)

    synthetic_prompt, _ = _next_prompt(
        base, 16, [8, 9], "synthetic_tail_rewrite", 2, 0
    )
    assert _lcp(synthetic_prompt, base) == len(base) - 2
    assert _expected_pool_lcp(synthetic_prompt, base) == len(base) - 2


def test_synthetic_tail_rewrite_rewrites_fixed_tail_then_appends():
    current = [1, 2, 3, 4, 5]
    prompt, expected_lcp = _next_prompt(
        current, 3, [8, 9], "synthetic_tail_rewrite", 2
    )
    assert prompt == [1, 2, 3, 8, 9, 8, 9, 8]
    assert expected_lcp == 3
    assert _lcp(current, prompt) == 3
    assert len(prompt) - expected_lcp == 5


def test_expected_selection_reports_exact_lcp_reuse_and_new_counts():
    assert _expected_selection([1, 2, 3, 8, 9], 3, 0) == {
        "lcp": 3,
        "reused_tokens": 0,
        "new_tokens": 5,
    }
    assert _expected_selection([1, 2, 3], 3, 3) == {
        "lcp": 3,
        "reused_tokens": 3,
        "new_tokens": 0,
    }


def test_flash_expected_bytes_include_capacity_live_state_and_checkpoints():
    assert _flash_expected_core_components(_TINY_FLASH_CONFIG, 10, 2) == {
        "session_cache": 5004,
        "indexer": 1044,
        "checkpoints": 908,
    }


def test_byte_acceptance_compares_only_the_modeled_core():
    expected = _flash_expected_core_components(_TINY_FLASH_CONFIG, 10, 2)
    status = {
        "session_telemetry": {
            "pool_unknown_sessions": 0,
            "pool_processed_tokens": 10,
            "pool_checkpoint_count": 2,
            "pool_allocated_bytes_by_component": {
                **expected,
                "mtp_cache": 99,
                "h_last": 77,
                "tail": 55,
            },
        }
    }
    result = _byte_acceptance(status, _TINY_FLASH_CONFIG, 5.0)
    assert result["passes"] is True
    assert result["actual_bytes"] == 6956
    assert result["error_pct"] == 0


def test_byte_acceptance_rejects_unknown_or_out_of_range_measurements():
    with pytest.raises(RuntimeError, match="確定していない"):
        _byte_acceptance(
            {"session_telemetry": {"pool_unknown_sessions": 1}},
            _TINY_FLASH_CONFIG,
            5.0,
        )

    expected = _flash_expected_core_components(_TINY_FLASH_CONFIG, 10, 2)
    status = {
        "session_telemetry": {
            "pool_unknown_sessions": 0,
            "pool_processed_tokens": 10,
            "pool_checkpoint_count": 2,
            "pool_allocated_bytes_by_component": {
                **expected,
                "session_cache": expected["session_cache"] + 1000,
            },
        }
    }
    with pytest.raises(RuntimeError, match="許容外"):
        _byte_acceptance(status, _TINY_FLASH_CONFIG, 5.0)


def test_replacement_tail_rejects_invalid_width():
    with pytest.raises(ValueError):
        _replacement_tail([1, 2], 0, [3, 4])
    with pytest.raises(ValueError):
        _replacement_tail([1, 2], 2, [3, 4])


def test_replacement_tail_skips_duplicate_old_tokens_in_palette():
    assert _replacement_tail([1, 2, 3], 1, [3, 3, 8]) == [8]


def test_input_tps_handles_fixed_cost_and_zero_time():
    assert _input_tps(50_000, 80.0) == 625.0
    assert _input_tps(50_000, 0.0) == 0.0
