"""Metal-free wiring checks for the Phase B1 SpecEngine integration."""

import ast
from pathlib import Path


def test_spec_engine_has_verification_only_dispatch_wiring():
    path = Path(__file__).resolve().parent.parent / "fastmlx" / "spec.py"
    tree = ast.parse(path.read_text())
    spec_engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SpecEngine"
    )
    methods = {
        node.name: node
        for node in spec_engine.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    init_calls = [
        node for node in ast.walk(methods["__init__"]) if isinstance(node, ast.Call)
    ]
    enable_call = next(
        call
        for call in init_calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "enable_quantized_dispatch"
    )
    active = next(kw.value for kw in enable_call.keywords if kw.arg == "active")
    assert isinstance(active, ast.Constant) and active.value is False

    hidden_names = {
        node.id
        for node in ast.walk(methods["_hidden_forward"])
        if isinstance(node, ast.Name)
    }
    assert "dispatch_scope" in hidden_names
    assert "capture" in hidden_names

    head_names = {
        node.id
        for node in ast.walk(methods["_head"])
        if isinstance(node, ast.Name)
    }
    assert "dispatch_scope" in head_names
    assert "dispatched_quantized_matmul" in head_names


def main():
    test_spec_engine_has_verification_only_dispatch_wiring()
    print("[PASS] test_spec_engine_has_verification_only_dispatch_wiring")


if __name__ == "__main__":
    main()
