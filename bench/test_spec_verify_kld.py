"""CPU/stub-safe tests for the Qwen3.6 verify-width quality harness."""

from __future__ import annotations

import pytest

from tools import spec_verify_kld as harness


def test_partition_tokens_keeps_final_remainder():
    assert harness.partition_tokens([0, 1, 2, 3, 4, 5, 6], 4) == [
        [0, 1, 2, 3],
        [4, 5, 6],
    ]
    assert harness.partition_tokens([], 9) == []


def test_partition_tokens_rejects_non_positive_width():
    with pytest.raises(ValueError):
        harness.partition_tokens([1], 0)


def test_teacher_force_uses_capture_contract_for_each_width():
    # Headless runners do not have an MLX device, so this intentionally uses
    # plain Python logits and still checks the width-specific engine calls.
    class Inner:
        norm = object()

    class StubEngine:
        inner = Inner()

        def __init__(self):
            self.calls = []

        def _hidden_forward(self, tokens, caches, **kwargs):
            token_list = list(tokens)
            self.calls.append((token_list, kwargs))
            return token_list, []

        def _head(self, hidden, norm):
            return [[[float(token), 0.0] for token in hidden]]

    engine = StubEngine()
    logits = harness.teacher_force(engine, object(), [1, 2, 3, 4, 5, 6], 4)
    assert [call[0] for call in engine.calls] == [[1, 2, 3, 4], [5]]
    assert engine.calls[0][1] == {"capture": True}
    assert engine.calls[1][1] == {"capture": False, "staged": True}
    assert len(logits) == 2

    engine = StubEngine()
    harness.teacher_force(engine, object(), [1, 2, 3], 1)
    assert all(call[1] == {"capture": False, "staged": True} for call in engine.calls)


def test_kld_summary_uses_reference_topk_and_temperature():
    ref = [[3.0, 1.0, -1.0, -2.0], [0.0, 2.0, 1.0, -3.0]]
    same = harness.summarize_kld(ref, ref, topk=2, temperature=0.7)
    assert same["positions"] == 2
    assert same["kld_mean"] == pytest.approx(0.0, abs=1e-12)
    assert same["top1_agreement"] == 1.0
    assert same["tail_mass"] > 0.0

    changed = harness.summarize_kld(
        ref, [[3.0, 1.0, -1.0, -2.0], [0.0, 1.0, 2.0, -3.0]], topk=2, temperature=0.7
    )
    assert changed["kld_mean"] > 0.0
    assert changed["top1_agreement"] == pytest.approx(0.5)


def test_aggregate_and_primary_verdict():
    cases = [
        {"widths": {"1": {"positions": 2, "kld_mean": 0.0, "top1_agreement": 1.0,
                            "reference_tail_mass": 0.1, "candidate_tail_mass": 0.1},
                    "4": {"positions": 2, "kld_mean": 0.001, "top1_agreement": 0.9,
                          "reference_tail_mass": 0.1, "candidate_tail_mass": 0.11},
                    "9": {"positions": 2, "kld_mean": 0.0002, "top1_agreement": 0.8,
                          "reference_tail_mass": 0.1, "candidate_tail_mass": 0.12}}},
        {"widths": {"1": {"positions": 1, "kld_mean": 0.0, "top1_agreement": 1.0,
                            "reference_tail_mass": 0.2, "candidate_tail_mass": 0.2},
                    "4": {"positions": 1, "kld_mean": 0.003, "top1_agreement": 1.0,
                          "reference_tail_mass": 0.2, "candidate_tail_mass": 0.19},
                    "9": {"positions": 1, "kld_mean": 0.002, "top1_agreement": 1.0,
                          "reference_tail_mass": 0.2, "candidate_tail_mass": 0.21}}},
    ]
    aggregate = harness.aggregate_width_summaries(cases, [1, 4, 9])
    assert aggregate["4"]["kld_mean"] == pytest.approx(0.002)
    assert aggregate["9"]["kld_mean"] == pytest.approx(0.0011)
    verdict = harness.quality_verdict(
        aggregate["4"]["kld_mean"], aggregate["9"]["kld_mean"]
    )
    assert verdict["cap3_le_current"] is False
    assert verdict["pass"] is False
    assert verdict["delta"] == pytest.approx(0.0009)

    tail_fail = harness.quality_verdict(0.0, 0.0, reference_tail_mass_max=0.002)
    assert tail_fail["kld_pass"] is True
    assert tail_fail["tail_pass"] is False
    assert tail_fail["pass"] is False


def test_parser_defaults_and_validation():
    args = harness.parse_args(["--model", "model", "--out", "result.json"])
    assert args.temp == pytest.approx(0.7)
    assert args.topk == 256
    assert args.ctx_values == [0, 4000, 17000]
    assert args.width_values == [1, 4, 9]

    with pytest.raises(SystemExit):
        harness.parse_args(["--model", "m", "--out", "o", "--widths", "4,9"])
    with pytest.raises(SystemExit):
        harness.parse_args(["--model", "m", "--out", "o", "--widths", "1,4"])
    with pytest.raises(SystemExit):
        harness.parse_args(["--model", "m", "--out", "o", "--tokens", "1"])
    with pytest.raises(SystemExit):
        harness.parse_args(["--model", "m", "--out", "o", "--temp", "-0.1"])
