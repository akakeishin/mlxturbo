import pytest

from tools import spec_ar_ab_generic as ab
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


def test_run_forwards_temperature(monkeypatch):
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(ab.G, "run_resumed", fake_run)
    got = ab._run(None, [], None, None, 8, (), "cap3", temp=0.7)
    assert got["variant"] == "cap3"
    assert seen["lookup_len"] == 0
    assert seen["temp"] == 0.7


def test_parser_accepts_fixed_length_temperature_run():
    args = ab.build_parser().parse_args([
        "--model", "m", "--mtp", "h", "--temp", "0.7",
        "--ignore-eos", "--out", "o.json",
    ])
    assert args.temp == pytest.approx(0.7)
    assert args.ignore_eos is True
