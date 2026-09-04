import pytest

from tools.spec_ar_ab_generic import _first_divergence, _summarize


def test_first_divergence():
    assert _first_divergence([1, 2, 3], [1, 4, 3]) == 1
    assert _first_divergence([1, 2], [1, 2]) is None
    assert _first_divergence([1, 2], [1, 2, 3]) == 2


def test_summary_keeps_crossover_evidence():
    rows = [
        {"variant": "mtp", "ms_per_tok": 15.0, "tok_per_round": 2.5,
         "tokens": [1, 2]},
        {"variant": "ar", "ms_per_tok": 14.0, "tok_per_round": 1.0,
         "tokens": [1, 3]},
    ]
    got = _summarize(rows, 50_000, 49_832, 52.3, case_idx=2)
    assert got["ctx"] == 49_832
    assert got["case_idx"] == 2
    assert got["prefill_s"] == 52.3
    assert got["mtp_vs_ar_pct"] == pytest.approx(7.142857)
    assert got["first_divergence"] == 1
