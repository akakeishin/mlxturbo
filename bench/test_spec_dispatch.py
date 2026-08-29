"""GPU route-reachability gate for Phase B1."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from mlx_lm import load

from mlxturbo.kernels import dispatch
from mlxturbo.spec import SpecEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="lmstudio-community/Qwen3.8-27B-MLX-4bit"
    )
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()

    model, _ = load(args.model)
    engine = SpecEngine(model, SimpleNamespace())
    original = dispatch.quantized_matmul
    phase = ""
    events = []

    def traced(x, w, scales, biases, **kwargs):
        table = kwargs.get("table")
        m = 1
        for dim in x.shape[:-1]:
            m *= dim
        k, n = x.shape[-1], w.shape[0]
        route = dispatch.select_route(k, n, m, table)
        events.append(
            {
                "phase": phase,
                "K": k,
                "N": n,
                "M": m,
                "route": route,
                "verification_active": table is None,
            }
        )
        return original(x, w, scales, biases, **kwargs)

    dispatch.quantized_matmul = traced
    try:
        tokens = mx.array([1] * 8)
        phase = "prefill"
        prefill_h, _ = engine._hidden_forward(
            tokens, engine.text.make_cache(), capture=False
        )
        mx.eval(prefill_h)

        phase = "capture"
        capture_h, _ = engine._hidden_forward(
            tokens, engine.text.make_cache(), capture=True
        )
        mx.eval(capture_h)

        phase = "head"
        logits = engine._head(capture_h, engine.inner.norm)
        mx.eval(logits)
    finally:
        dispatch.quantized_matmul = original

    prefill = [event for event in events if event["phase"] == "prefill"]
    capture = [event for event in events if event["phase"] == "capture"]
    head = [event for event in events if event["phase"] == "head"]
    assert prefill and all(not event["verification_active"] for event in prefill)
    assert any(
        event["verification_active"] and event["route"] != dispatch.STOCK
        for event in capture
    ), capture
    assert any(
        event["verification_active"]
        and event["K"] == 5120
        and event["N"] == 248320
        and event["route"] != dispatch.STOCK
        for event in head
    ), head

    report = {
        "prefill_events": prefill,
        "capture_custom_events": [
            event for event in capture if event["route"] != dispatch.STOCK
        ],
        "head_events": head,
        "capture_shape": list(capture_h.shape),
        "logits_shape": list(logits.shape),
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as file:
            json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
