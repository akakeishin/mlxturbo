"""Qwen3.8 Flash-Next fixed-state inventory and rollback plan gate.

This is a diagnostic only.  It makes the persistent state explicit as exactly
134 tensor leaves (12 QSA attention layers x 5, 36 GDN layers x 2, one PLE
window, and one shared n-gram context), then compares capture-derived commits
with the existing ``spec_flash.rollback`` implementation.  It does not compile
or replace the model forward.

stdout is one JSON line.  PASS requires an exact fixed-shape round trip,
capture-derived rollback agreement for keep=1/3/4, and exact continuation
state/logits from each rollback.  Metal/model/sidecar absence is unavailable.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
for path in (REPO_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import qwen4_qsa_state_gate as B0

TASK_ID = "qwen4_full_state_plan_impl_0905"
WIDTH = 4
EXPECTED_LEAVES = 134
EXPECTED_FULL = 12
EXPECTED_GDN = 36
EXPECTED_PLE = 1
EXPECTED_NGRAM = 1
DEFAULT_MODEL = B0.DEFAULT_MODEL
DEFAULT_NGRAM = B0.DEFAULT_NGRAM

mx: Any = None


@dataclass(frozen=True)
class LeafMeta:
    """Fixed physical shape plus host-side logical metadata."""

    name: str
    family: str
    layer: int | None
    slot: int | str | None
    shape: tuple[int, ...]
    dtype: str
    logical_axis: int | None
    logical_length: int


@dataclass(frozen=True)
class PlannedState:
    leaves: tuple[Any, ...]
    metadata: tuple[LeafMeta, ...]


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in value.shape)


def _dtype(value: Any) -> str:
    return str(value.dtype)


def _logical_view(value: Any, meta: LeafMeta) -> Any:
    if meta.logical_axis is None:
        return value
    sl = [slice(None)] * value.ndim
    sl[meta.logical_axis] = slice(0, meta.logical_length)
    return value[tuple(sl)]


def _max_abs(left: Any, right: Any) -> float:
    if _shape(left) != _shape(right):
        return float("inf")
    diff = mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
    mx.eval(diff)
    return float(mx.max(diff).item()) if diff.size else 0.0


def _array_equal(left: Any, right: Any) -> bool:
    if _shape(left) != _shape(right) or _dtype(left) != _dtype(right):
        return False
    value = mx.all(left == right)
    mx.eval(value)
    return bool(value.item())


def _compare_states(left: PlannedState, right: PlannedState) -> dict[str, Any]:
    """Compare every leaf in its declared logical region."""

    same_metadata = len(left.metadata) == len(right.metadata)
    mismatches: list[dict[str, Any]] = []
    max_abs = 0.0
    if same_metadata:
        for i, (lm, rm) in enumerate(zip(left.metadata, right.metadata)):
            if lm != rm:
                same_metadata = False
                mismatches.append(
                    {"index": i, "left": lm.__dict__, "right": rm.__dict__}
                )
                continue
            la = _logical_view(left.leaves[i], lm)
            ra = _logical_view(right.leaves[i], rm)
            diff = _max_abs(la, ra)
            max_abs = max(max_abs, diff)
            if not _array_equal(la, ra):
                mismatches.append(
                    {
                        "index": i,
                        "name": lm.name,
                        "max_abs": diff,
                        "shape": list(_shape(la)),
                    }
                )
    else:
        mismatches.append(
            {"reason": "metadata_count", "left": len(left.metadata), "right": len(right.metadata)}
        )
    return {
        "leaf_count": len(left.leaves),
        "right_leaf_count": len(right.leaves),
        "metadata_equal": same_metadata,
        "max_abs": max_abs,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:8],
        "pass": same_metadata and not mismatches,
    }


class FullStatePlan:
    """Qwen-specific fixed-state pack/install mapping for the diagnostic."""

    def __init__(self, model: Any, qwen: Any, capacity: int, ratio: int):
        self.model = model
        self.qwen = qwen
        self.capacity = int(capacity)
        self.ratio = int(ratio)
        layers = list(model.model.layers)
        self.full_layers = [
            i for i, layer in enumerate(layers)
            if getattr(layer, "layer_type", None) == "full_attention"
        ]
        self.gdn_layers = [
            i for i, layer in enumerate(layers)
            if getattr(layer, "layer_type", None) == "linear_attention"
        ]
        self.ple_layers = [
            i for i, layer in enumerate(layers)
            if getattr(layer, "ple", None) is not None
        ]
        if (
            len(self.full_layers) != EXPECTED_FULL
            or len(self.gdn_layers) != EXPECTED_GDN
            or len(self.ple_layers) != EXPECTED_PLE
            or len(layers) != EXPECTED_FULL + EXPECTED_GDN
        ):
            raise ValueError(
                "topology mismatch: "
                f"layers={len(layers)} full={len(self.full_layers)} "
                f"gdn={len(self.gdn_layers)} ple={len(self.ple_layers)}"
            )
        self.ple_layer = self.ple_layers[0]
        self.ngram_layer = self.ple_layer
        if ratio != WIDTH:
            raise ValueError(f"fixed-M4 requires compress ratio {WIDTH}, got {ratio}")
        text_args = model.args.text
        if int(getattr(text_args, "num_hidden_layers", len(layers))) != len(layers):
            raise ValueError("text config layer count disagrees with model topology")
        configured_ple = [int(v) - 1 for v in getattr(text_args, "ple_layer_ids", [])]
        if configured_ple != self.ple_layers or configured_ple != [self.ple_layer]:
            raise ValueError(f"PLE config mismatch: configured={configured_ple} actual={self.ple_layers}")
        if any(
            int(layers[i].self_attn.indexer.compress_ratio) != WIDTH
            or int(layers[i].self_attn.indexer.n_heads) != 4
            for i in self.full_layers
        ):
            raise ValueError("attention QSA config is not fixed-M4/indexer4")
        self.adapters = {
            i: B0.FixedStateAdapter(
                qwen, self.capacity, int(layers[i].self_attn.indexer.compress_ratio)
            )
            for i in self.full_layers
        }
        pool_caps = {adapter.pool_capacity for adapter in self.adapters.values()}
        if len(pool_caps) != 1:
            raise ValueError(f"attention pool capacities differ: {sorted(pool_caps)}")
        self.pool_capacity = pool_caps.pop()
        self.groups: dict[str, tuple[int, ...]] = {}
        if self.ple_layer not in self.gdn_layers:
            raise ValueError("the single PLE layer must be a GDN layer")
        layout: list[tuple[str, str, int | None, int | str | None]] = []
        for layer_idx in self.full_layers:
            layout.extend(
                (name, "attention", layer_idx, name)
                for name in ("K", "V", "array_offset", "raw_index", "pooled_index")
            )
        for layer_idx in self.gdn_layers:
            layout.extend(
                (("gdn_conv", "gdn", layer_idx, 0), ("gdn_state", "gdn", layer_idx, 1))
            )
        layout.extend((("ple_conv", "ple", self.ple_layer, 2), ("ngram_context", "ngram", self.ngram_layer, 3)))
        self.expected_layout = tuple(layout)

    def _meta(
        self,
        name: str,
        family: str,
        layer: int | None,
        slot: int | str | None,
        value: Any,
        logical_axis: int | None = None,
        logical_length: int | None = None,
    ) -> LeafMeta:
        if logical_axis is not None and logical_length is None:
            logical_length = int(value.shape[logical_axis])
        return LeafMeta(
            name,
            family,
            layer,
            slot,
            _shape(value),
            _dtype(value),
            logical_axis,
            int(logical_length or 0),
        )

    def _check_cache_topology(self, caches: list[Any]) -> int:
        if len(caches) != len(self.model.model.layers):
            raise ValueError(
                f"cache count mismatch: {len(caches)} != {len(self.model.model.layers)}"
            )
        ngram_slots = []
        for i in self.gdn_layers:
            cache = caches[i]
            if not isinstance(getattr(cache, "state", None), (list, tuple)):
                raise ValueError(f"GDN cache {i} is not a four-slot ArraysCache")
            if cache[0] is None or cache[1] is None:
                raise ValueError(f"GDN cache {i} has empty conv/state slot")
            if cache[2] is not None:
                # The one PLE window is named and counted separately below.
                if i != self.ple_layer:
                    raise ValueError(f"unexpected PLE state on GDN cache {i}")
            if cache[3] is not None:
                ngram_slots.append(i)
        if len(ngram_slots) != EXPECTED_NGRAM or ngram_slots[0] != self.ngram_layer:
            raise ValueError(f"expected one n-gram context at {self.ngram_layer}, got {ngram_slots}")
        for i in self.full_layers:
            cache = caches[i]
            if not hasattr(cache, "indexer"):
                raise ValueError(f"full-attention cache {i} lacks indexer")
            if cache.keys is None or cache.values is None or cache.indexer.keys is None:
                raise ValueError(f"full-attention cache {i} is not populated")
        ple_cache = caches[self.ple_layer]
        if ple_cache[2] is None:
            raise ValueError("PLE cache window is empty")
        return ngram_slots[0]

    def pack(self, caches: list[Any]) -> PlannedState:
        ngram_layer = self._check_cache_topology(caches)
        leaves: list[Any] = []
        metadata: list[LeafMeta] = []
        self.groups = {}

        # Stable state-plan order: all attention leaves, all GDN leaves, PLE,
        # then the single shared n-gram context.
        for layer_idx in self.full_layers:
            start = len(leaves)
            fixed = self.adapters[layer_idx].pack(caches[layer_idx])
            names = ("K", "V", "array_offset", "raw_index", "pooled_index")
            for value, item, name in zip(fixed.leaves, fixed.metadata, names):
                leaves.append(value)
                metadata.append(
                    self._meta(
                        name,
                        "attention",
                        layer_idx,
                        name,
                        value,
                        None if name == "array_offset" else item.axis,
                        item.logical_length,
                    )
                )
            self.groups[f"attention:{layer_idx}"] = tuple(range(start, len(leaves)))

        for layer_idx in self.gdn_layers:
            cache = caches[layer_idx]
            start = len(leaves)
            leaves.extend((cache[0], cache[1]))
            metadata.extend(
                (
                    self._meta("gdn_conv", "gdn", layer_idx, 0, cache[0]),
                    self._meta("gdn_state", "gdn", layer_idx, 1, cache[1]),
                )
            )
            self.groups[f"gdn:{layer_idx}"] = tuple(range(start, len(leaves)))

        ple_cache = caches[self.ple_layer]
        start = len(leaves)
        leaves.append(ple_cache[2])
        metadata.append(self._meta("ple_conv", "ple", self.ple_layer, 2, ple_cache[2]))
        self.groups["ple"] = (start,)

        ctx = caches[ngram_layer][3]
        start = len(leaves)
        leaves.append(ctx)
        metadata.append(self._meta("ngram_context", "ngram", ngram_layer, 3, ctx))
        self.groups["ngram"] = (start,)

        if len(leaves) != EXPECTED_LEAVES:
            raise ValueError(f"state leaf count mismatch: {len(leaves)} != {EXPECTED_LEAVES}")
        return PlannedState(tuple(leaves), tuple(metadata))

    def _validate_state(self, state: PlannedState) -> None:
        if len(state.leaves) != EXPECTED_LEAVES or len(state.metadata) != EXPECTED_LEAVES:
            raise ValueError("planned state does not contain exactly 134 leaves")
        for i, (value, item) in enumerate(zip(state.leaves, state.metadata)):
            if i >= len(self.expected_layout) or (item.name, item.family, item.layer, item.slot) != self.expected_layout[i]:
                raise ValueError(f"leaf {i} has unexpected state-plan identity")
            if _shape(value) != item.shape or _dtype(value) != item.dtype:
                raise ValueError(f"leaf {i} metadata shape/dtype mismatch")
            if item.logical_axis is not None and not (0 <= item.logical_axis < len(item.shape)):
                raise ValueError(f"leaf {i} has invalid logical axis")
            if item.logical_length < 0 or (
                item.logical_axis is not None
                and item.logical_length > item.shape[item.logical_axis]
            ):
                raise ValueError(f"leaf {i} has invalid logical length")

    def unpack(self, state: PlannedState, caches: list[Any]) -> None:
        self._validate_state(state)
        self._check_cache_topology(caches)
        cursor = 0
        for layer_idx in self.full_layers:
            fixed_leaves = state.leaves[cursor : cursor + 5]
            fixed_meta = state.metadata[cursor : cursor + 5]
            cursor += 5
            adapter_meta = tuple(
                B0.TensorLeafMeta(
                    item.name,
                    item.logical_axis if item.logical_axis is not None else 0,
                    item.shape[item.logical_axis] if item.logical_axis is not None else item.shape[0],
                    item.logical_length,
                    item.dtype,
                )
                for item in fixed_meta
            )
            fixed = B0.FixedTensorState(
                tuple(fixed_leaves), adapter_meta, fixed_meta[4].logical_length
            )
            self.adapters[layer_idx].unpack(fixed, caches[layer_idx])
        for layer_idx in self.gdn_layers:
            caches[layer_idx][0] = state.leaves[cursor]
            caches[layer_idx][1] = state.leaves[cursor + 1]
            cursor += 2
        caches[self.ple_layer][2] = state.leaves[cursor]
        cursor += 1
        caches[self.ngram_layer][3] = state.leaves[cursor]
        cursor += 1
        if cursor != EXPECTED_LEAVES:
            raise ValueError(f"unpack cursor mismatch: {cursor}")

    def committed_from_capture(
        self,
        post: PlannedState,
        cap: Any,
        pre: dict[str, Any],
        ids: Any,
        keep: int,
    ) -> PlannedState:
        if keep < 1 or keep > WIDTH:
            raise ValueError(f"keep must be in 1..{WIDTH}: {keep}")
        self._validate_state(post)
        leaves = list(post.leaves)
        metadata = list(post.metadata)
        cursor = 0
        for layer_idx in self.full_layers:
            base = pre["kv"][layer_idx][0]
            fixed_meta = tuple(
                B0.TensorLeafMeta(
                    item.name,
                    item.logical_axis if item.logical_axis is not None else 0,
                    item.shape[item.logical_axis] if item.logical_axis is not None else item.shape[0],
                    item.logical_length,
                    item.dtype,
                )
                for item in metadata[cursor : cursor + 5]
            )
            fixed = B0.FixedTensorState(
                tuple(leaves[cursor : cursor + 5]),
                fixed_meta,
                fixed_meta[4].logical_length,
            )
            rolled = self.adapters[layer_idx].rollback(fixed, keep, WIDTH, base)
            leaves[cursor : cursor + 5] = list(rolled.leaves)
            for j, item in enumerate(rolled.metadata):
                metadata[cursor + j] = self._meta(
                    item.name,
                    "attention",
                    layer_idx,
                    item.name,
                    rolled.leaves[j],
                    None if item.name == "array_offset" else item.axis,
                    item.logical_length,
                )
            cursor += 5

        for layer_idx in self.gdn_layers:
            linear = self.model.model.layers[layer_idx].linear_attn
            record = cap.gdn.get(id(linear))
            if record is None:
                raise ValueError(f"capture has no GDN record for layer {layer_idx}")
            conv_input, states_all = record
            conv_len = int(linear.conv_kernel_size) - 1
            conv = mx.contiguous(conv_input[:, keep : keep + conv_len, :])
            state = states_all[:, keep - 1]
            leaves[cursor] = conv
            leaves[cursor + 1] = state
            metadata[cursor] = self._meta("gdn_conv", "gdn", layer_idx, 0, conv)
            metadata[cursor + 1] = self._meta("gdn_state", "gdn", layer_idx, 1, state)
            cursor += 2

        ple = self.model.model.layers[self.ple_layer].ple
        full = cap.ple.get(id(ple))
        if full is None:
            raise ValueError("capture has no PLE record")
        n = int(ple.short_conv_state_len)
        ple_state = mx.contiguous(full[:, keep : keep + n, :])
        leaves[cursor] = ple_state
        metadata[cursor] = self._meta("ple_conv", "ple", self.ple_layer, 2, ple_state)
        cursor += 1

        ctx = pre["ctx"][self.ngram_layer]
        if ctx is None:
            raise ValueError("snapshot_pre has no n-gram context")
        ctx_len = int(self.model.args.text.ngram_size) - 1
        ctx_state = mx.concatenate([ctx, ids[:, :keep]], axis=1)[:, -ctx_len:]
        leaves[cursor] = ctx_state
        metadata[cursor] = self._meta(
            "ngram_context", "ngram", self.ngram_layer, 3, ctx_state
        )
        cursor += 1
        if cursor != EXPECTED_LEAVES:
            raise ValueError(f"capture commit cursor mismatch: {cursor}")
        return PlannedState(tuple(leaves), tuple(metadata))


def _run_model(model: Any, ids: Any, caches: list[Any]) -> Any:
    logits = model(ids, cache=caches)
    B0._eval_cache(caches, logits)
    return logits


def _run_gate(args: argparse.Namespace) -> dict[str, Any]:
    global mx
    if not mx.metal.is_available():
        return {"task_id": TASK_ID, "status": "UNAVAILABLE", "reason": "Metal unavailable"}
    model_path = Path(os.path.expanduser(args.model))
    ngram_path = Path(os.path.expanduser(args.ngram)) if args.ngram else None
    if not model_path.exists():
        return {"task_id": TASK_ID, "status": "UNAVAILABLE", "reason": f"model path does not exist: {model_path}"}
    if ngram_path is None or not ngram_path.exists():
        return {"task_id": TASK_ID, "status": "UNAVAILABLE", "reason": f"n-gram sidecar path does not exist: {ngram_path}"}

    os.environ.setdefault("FASTMLX_NGRAM_DISK", "1")
    mx.set_default_device(mx.gpu)
    import mlxturbo  # noqa: F401
    from mlx_lm import load
    import mlx_lm.models.qwen4_exp as qwen
    from mlxturbo import spec_flash as sf
    from mlxturbo.ngram_stream import install

    with contextlib.redirect_stdout(io.StringIO()):
        model, tokenizer = load(str(model_path))
        install(model, str(ngram_path))

    layer_types = list(model.args.text.layer_types)
    if layer_types.count("full_attention") != EXPECTED_FULL or layer_types.count("linear_attention") != EXPECTED_GDN:
        return {
            "task_id": TASK_ID,
            "status": "UNAVAILABLE",
            "reason": f"unexpected layer topology: full={layer_types.count('full_attention')} linear={layer_types.count('linear_attention')}",
        }
    target_layer = layer_types.index("full_attention")
    target_attn = model.model.layers[target_layer].self_attn
    ratio = int(target_attn.indexer.compress_ratio)
    if ratio != WIDTH:
        return {"task_id": TASK_ID, "status": "UNAVAILABLE", "reason": f"compress ratio {ratio} != {WIDTH}"}
    prefix = max(int(args.prefix), int(target_attn.indexer.token_budget))
    requested = prefix + WIDTH * 2
    ids = B0._make_ids(tokenizer, requested, args.question)
    if ids.shape[1] < requested:
        return {"task_id": TASK_ID, "status": "UNAVAILABLE", "reason": f"prompt has {ids.shape[1]} tokens, needs {requested}"}

    with contextlib.redirect_stdout(io.StringIO()):
        prefix_cache = model.make_cache()
        chunk = max(1, int(args.chunk))
        for start in range(0, prefix, chunk):
            _run_model(model, ids[:, start : min(prefix, start + chunk)], prefix_cache)
    capacity = B0._round_capacity(prefix + WIDTH)
    plan = FullStatePlan(model, qwen, capacity, ratio)

    pre_cache = B0._clone_caches(prefix_cache, qwen)
    pre_snapshot = sf.snapshot_pre(model, pre_cache)
    post_cache = B0._clone_caches(pre_cache, qwen)
    verify_ids = ids[:, prefix : prefix + WIDTH]
    with contextlib.redirect_stdout(io.StringIO()):
        with sf.capture(model) as cap:
            verify_logits = _run_model(model, verify_ids, post_cache)
            capture_values = [v for pair in cap.gdn.values() for v in pair]
            capture_values.extend(cap.ple.values())
            B0._eval_cache(post_cache, verify_logits, *capture_values)

    pre_state = plan.pack(pre_cache)
    post_state = plan.pack(post_cache)
    roundtrip_cache = B0._clone_caches(pre_cache, qwen)
    plan.unpack(post_state, roundtrip_cache)
    roundtrip_state = plan.pack(roundtrip_cache)
    roundtrip_check = _compare_states(post_state, roundtrip_state)

    rollback_results: dict[str, Any] = {}
    all_rollback = True
    for keep in (1, 3, 4):
        planned = plan.committed_from_capture(
            post_state, cap, pre_snapshot, verify_ids, keep
        )
        production_cache = B0._clone_caches(post_cache, qwen)
        sf.rollback(
            model,
            production_cache,
            cap,
            pre_snapshot,
            keep=keep,
            total=WIDTH,
            ids_kept=verify_ids[:, :keep],
        )
        B0._eval_cache(production_cache)
        production_state = plan.pack(production_cache)
        rollback_check = _compare_states(planned, production_state)

        installed_cache = B0._clone_caches(pre_cache, qwen)
        plan.unpack(planned, installed_cache)
        installed_state = plan.pack(installed_cache)
        install_check = _compare_states(planned, installed_state)

        # Continue immediately after the accepted prefix.  Using tokens after
        # the full verification window would compare equal states too, but it
        # would skip the rejected suffix and weaken the rollback contract.
        continuation_ids = ids[:, prefix + keep : prefix + keep + WIDTH]
        continuation_check: dict[str, Any]
        if continuation_ids.shape[1] != WIDTH:
            continuation_check = {"pass": False, "reason": "continuation width unavailable"}
        else:
            planned_cont_cache = B0._clone_caches(pre_cache, qwen)
            plan.unpack(planned, planned_cont_cache)
            with contextlib.redirect_stdout(io.StringIO()):
                production_cont_logits = _run_model(
                    model, continuation_ids, production_cache
                )
                planned_cont_logits = _run_model(
                    model, continuation_ids, planned_cont_cache
                )
            production_cont_state = plan.pack(production_cache)
            planned_cont_state = plan.pack(planned_cont_cache)
            state_check = _compare_states(planned_cont_state, production_cont_state)
            logits_equal = _array_equal(planned_cont_logits, production_cont_logits)
            continuation_check = {
                "logits_exact": logits_equal,
                "logits_max_abs": _max_abs(planned_cont_logits, production_cont_logits),
                "state": state_check,
                "pass": bool(logits_equal and state_check["pass"]),
            }

        item_pass = bool(
            rollback_check["pass"]
            and install_check["pass"]
            and continuation_check["pass"]
        )
        all_rollback = all_rollback and item_pass
        rollback_results[str(keep)] = {
            "keep": keep,
            "planned_vs_production": rollback_check,
            "planned_install_roundtrip": install_check,
            "continuation": continuation_check,
            "pass": item_pass,
        }

    topology = {
        "layer_count": len(layer_types),
        "full_attention": len(plan.full_layers),
        "gdn": len(plan.gdn_layers),
        "ple": len(plan.ple_layers),
        "ngram": EXPECTED_NGRAM,
        "expected_leaves": EXPECTED_LEAVES,
        "actual_leaves": len(post_state.leaves),
        "pass": len(post_state.leaves) == EXPECTED_LEAVES,
    }
    all_pass = bool(topology["pass"] and roundtrip_check["pass"] and all_rollback)
    return {
        "task_id": TASK_ID,
        "status": "PASS" if all_pass else "FAIL",
        "model": str(model_path),
        "prefix_tokens": prefix,
        "width": WIDTH,
        "target_layer": target_layer,
        "fixed_attention_capacity": capacity,
        "fixed_pool_capacity": plan.pool_capacity,
        "state_plan": {
            "leaf_order": "12 attention groups, 36 GDN pairs, PLE, ngram",
            "leaf_count": len(post_state.leaves),
            "metadata_fields": ["name", "family", "layer", "slot", "shape", "dtype", "logical_axis", "logical_length"],
            "metadata": [item.__dict__ for item in post_state.metadata],
        },
        "checks": {
            "topology": topology,
            "post_state_pack_install_roundtrip": roundtrip_check,
            "rollback_keep": {"cases": rollback_results, "pass": all_rollback},
        },
        "limitations": [
            "Diagnostic only: no whole-model mx.compile or graphbank is implemented.",
            "The continuation check compares two eager forwards after independently installed states; it does not transcribe the model forward.",
            "A PASS establishes the planned 134-leaf state mapping against current spec_flash rollback, not a production state-plan seam.",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ngram", default=DEFAULT_NGRAM)
    parser.add_argument("--prefix", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--question", default="State-plan gate. Return a short answer.")
    return parser.parse_args()


def main() -> int:
    global mx
    args = _parse_args()
    try:
        import mlx.core as mx_module
    except Exception as exc:
        print(json.dumps({"task_id": TASK_ID, "status": "UNAVAILABLE", "reason": f"MLX import failed: {type(exc).__name__}"}, separators=(",", ":")))
        return 2
    mx = mx_module
    B0.mx = mx_module
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = _run_gate(args)
    except Exception as exc:
        result = {
            "task_id": TASK_ID,
            "status": "FAIL",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "limitations": ["The real state-plan gate did not complete; no PASS claim is made."],
        }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return {"PASS": 0, "FAIL": 1, "UNAVAILABLE": 2}.get(result.get("status"), 1)


if __name__ == "__main__":
    raise SystemExit(main())
