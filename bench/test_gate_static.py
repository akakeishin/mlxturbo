"""Metal-free contract tests for bench/gate.py."""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import gate


class _Tokenizer:
    def decode(self, tokens):
        return ",".join(str(token) for token in tokens)


def test_length_and_value_mismatches_are_recorded():
    positions = gate.mismatch_positions([1, 2, 3], [1, 9, 3, 4])
    assert positions == [1, 3]
    records = gate.mismatch_records(
        [1, 2, 3], [1, 9, 3, 4], positions, _Tokenizer(), radius=1
    )
    assert records[0]["index"] == 1
    assert records[0]["reference_token"] == 2
    assert records[0]["actual_token"] == 9
    assert records[1]["reference_token"] is None
    assert records[1]["actual_token"] == 4


def test_gate_uses_manual_loop_and_fixed_baseline_contract():
    source = Path(gate.__file__).read_text()
    assert "stream_generate" not in source
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    manual_names = {
        node.id
        for node in ast.walk(functions["manual_greedy"])
        if isinstance(node, ast.Name)
    }
    manual_attrs = {
        node.attr
        for node in ast.walk(functions["manual_greedy"])
        if isinstance(node, ast.Attribute)
    }
    assert "model" in manual_names
    assert "make_cache" in manual_attrs
    assert "argmax" in manual_attrs

    generate_calls = [
        node
        for node in ast.walk(functions["main"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
    ]
    assert len(generate_calls) == 2
    baseline = generate_calls[0]
    constants = {
        kw.arg: kw.value.value
        for kw in baseline.keywords
        if isinstance(kw.value, ast.Constant)
    }
    assert constants["n_draft"] == 0
    assert constants["max_draft"] == 0
    assert constants["lookup_len"] == 0


def main():
    tests = [
        test_length_and_value_mismatches_are_recorded,
        test_gate_uses_manual_loop_and_fixed_baseline_contract,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


if __name__ == "__main__":
    main()
