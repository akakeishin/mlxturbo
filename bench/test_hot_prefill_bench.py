from __future__ import annotations

import pytest

from bench.hot_prefill_bench import _input_tps, _lcp, _next_prompt, _replacement_tail


def test_pure_append_uses_exact_requested_delta():
    current = [1, 2, 3]
    prompt, expected_lcp = _next_prompt(current, 4, [8, 9], "pure_append", 1)
    assert prompt == [1, 2, 3, 8, 9, 8, 9]
    assert expected_lcp == 3
    assert _lcp(current, prompt) == 3
    assert len(prompt) - expected_lcp == 4


def test_retokenized_rewrites_fixed_tail_then_appends():
    current = [1, 2, 3, 4, 5]
    prompt, expected_lcp = _next_prompt(current, 3, [8, 9], "retokenized", 2)
    assert prompt == [1, 2, 3, 8, 9, 8, 9, 8]
    assert expected_lcp == 3
    assert _lcp(current, prompt) == 3
    assert len(prompt) - expected_lcp == 5


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
