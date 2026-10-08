# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: materialization cases."""

from __future__ import annotations

from tests.graph_sdk.reference._effects_v1.builders import (
    _items,
    _materialization_decl,
    _materialization_event,
    _materialization_spec,
    _materialization_trace,
)
from tests.graph_sdk.reference._effects_v1.case import (
    _case,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Object,
)


def _materialization_specs() -> list[Object]:
    cases: list[Object] = []
    initial_single = _materialization_spec("initial", "single", declaration="D0")
    initial_collection = _materialization_spec("initial", "collection", declaration="D0")
    adaptive_single = _materialization_spec("adaptive", "single")
    adaptive_collection = _materialization_spec("adaptive", "collection")
    for name, spec in (
        ("admit_initial_single", initial_single),
        ("admit_initial_collection", initial_collection),
        ("admit_adaptive_single", adaptive_single),
        ("admit_adaptive_collection", adaptive_collection),
    ):
        cases.append(_case("materialization", name, _materialization_decl(spec, admission=True), [], "admission"))
    admission_mutants: tuple[tuple[str, Object, Object], ...] = (
        ("single_output_mismatch", initial_single, {"output_type": "text_collection"}),
        ("single_max_items", initial_single, {"max_items": 2}),
        ("collection_scalar_output", initial_collection, {"output_type": "text"}),
        ("collection_zero_max", initial_collection, {"max_items": 0}),
        ("adaptive_binding_identity", adaptive_collection, {"declaration": "D0"}),
    )
    for name, base, changes in admission_mutants:
        mutant = dict(base)
        mutant.update(changes)
        cases.append(_case("materialization", name, _materialization_decl(mutant, admission=True), [], "admission"))
    collection_ceiling = dict(initial_collection, max_items=4)
    cases.append(
        _case(
            "materialization",
            "collection_ceiling",
            _materialization_decl(collection_ceiling),
            _materialization_trace(collection_ceiling, _items("one")),
            "execution_preflight",
        )
    )
    text_collection = _materialization_spec("initial", "collection", declaration="D0")
    nested = _materialization_spec(
        "adaptive", "collection", association="A1", item_type="text_collection", output_type="nested_collection"
    )
    cases.extend(
        [
            _case(
                "materialization",
                "conflicting_schema",
                _materialization_decl(
                    text_collection,
                    dict(text_collection, association="D1", declaration="D1", item_type="bytes"),
                    admission=True,
                ),
                [],
                "admission",
            ),
            _case(
                "materialization",
                "nested_collection_schema",
                _materialization_decl(text_collection, nested, admission=True),
                [],
                "admission",
            ),
            _case(
                "materialization",
                "collection_root_input",
                dict(_materialization_decl(text_collection, admission=True), root_input_types=["text_collection"]),
                [],
                "admission",
            ),
        ]
    )
    for label, spec in (("initial", initial_single), ("adaptive", adaptive_single)):
        cases.append(
            _case(
                "materialization",
                f"{label}_single_exact",
                _materialization_decl(spec),
                _materialization_trace(spec, _items("one")),
            )
        )
        cases.append(
            _case(
                "materialization",
                f"{label}_single_multiple",
                _materialization_decl(spec),
                _materialization_trace(spec, _items("one", "two")),
            )
        )
    for label, spec in (("initial", initial_collection), ("adaptive", adaptive_collection)):
        for size in (1, 2, 3, 4, 0):
            values = tuple(f"v{index}" for index in range(size))
            cases.append(
                _case(
                    "materialization",
                    f"{label}_collection_{'one_over' if size == 4 else size}",
                    _materialization_decl(spec, max_requests=1 if size == 0 else 2),
                    _materialization_trace(spec, _items(*values)),
                )
            )
        reordered = list(reversed(_items("zero", "one")))
        cases.append(
            _case(
                "materialization",
                f"{label}_canonical_reorder",
                _materialization_decl(spec),
                _materialization_trace(spec, reordered),
            )
        )
        malformed: tuple[tuple[str, list[Object]], ...] = (
            ("duplicate", [{"key": 0, "value": "a", "version": 1}, {"key": 0, "value": "b", "version": 1}]),
            ("wrong_item", [{"key": 0, "value": 7, "version": 1}]),
            ("nested_value", [{"key": 0, "value": {"items": []}, "version": 1}]),
        )
        for suffix, values in malformed:
            trace = _materialization_trace(spec, values)
            if suffix in ("wrong_item", "nested_value"):
                trace[-1]["kind"] = "source_item_constructor"
            cases.append(
                _case(
                    "materialization",
                    f"{label}_{suffix}",
                    _materialization_decl(spec, max_requests=1 if suffix == "duplicate" else 2),
                    trace,
                )
            )
        one_over_malformed = _items("a", "b", "c", "d")
        cases.append(
            _case(
                "materialization",
                f"{label}_outer_count_precedence",
                _materialization_decl(spec),
                _materialization_trace(spec, one_over_malformed),
            )
        )
    for suffix, limits in (
        ("artifact_count_one_over", {"max_artifacts": 2}),
        ("logical_bytes_one_over", {"max_artifact_bytes": 7}),
        ("provenance_one_over", {"max_provenance_edges": 1}),
    ):
        cases.append(
            _case(
                "materialization",
                f"initial_{suffix}",
                _materialization_decl(initial_collection, limit_changes=limits),
                _materialization_trace(initial_collection, _items("aa", "bb")),
                "execution_preflight",
            )
        )
    for suffix, maximum in (("declared_bytes_exact", 4), ("declared_bytes_one_over", 3)):
        byte_spec = dict(initial_collection, max_bytes=maximum)
        cases.append(
            _case(
                "materialization",
                f"initial_{suffix}",
                _materialization_decl(byte_spec),
                _materialization_trace(byte_spec, _items("aa", "bb")),
            )
        )
    cases.append(
        _case(
            "materialization",
            "adaptive_foreign_association",
            _materialization_decl(adaptive_collection, max_requests=1),
            [
                *_materialization_trace(adaptive_collection, _items("a"))[:-1],
                dict(_materialization_event(adaptive_collection, _items("a")), association="A1"),
            ],
        )
    )
    cases.extend(
        [
            _case(
                "materialization",
                "initial_unmaterialized_source_result",
                _materialization_decl(initial_collection),
                _materialization_trace(initial_collection, _items("a")),
            ),
            _case(
                "materialization",
                "adaptive_unmaterialized_result",
                _materialization_decl(adaptive_collection),
                _materialization_trace(adaptive_collection, _items("a")),
            ),
            _case(
                "materialization",
                "binding_finish_before_materialization",
                _materialization_decl(initial_collection),
                _materialization_trace(initial_collection, _items("a")),
            ),
        ]
    )
    first = _materialization_spec("initial", "collection", declaration="D0", node="N0")
    second = _materialization_spec("initial", "collection", declaration="D1", node="N1")
    cases.append(
        _case(
            "materialization",
            "same_port_distinct_nodes",
            _materialization_decl(first, second),
            [
                *_materialization_trace(first, _items("a")),
                *_materialization_trace(second, _items("b"), request="R1"),
            ],
        )
    )
    for path, spec in (("initial", initial_single), ("adaptive", adaptive_single)):
        cases.append(
            _case(
                "materialization",
                f"{path}_single_zero_collection_limit",
                _materialization_decl(spec, limit_changes={"max_collection_items": 0}),
                _materialization_trace(spec, _items("one")),
            )
        )
        for terminal in ("lost", "cancelled"):
            prefix = _materialization_trace(spec, _items("one"))[:-1]
            terminals: list[Object] = (
                [{"kind": "lost", "request": "R0"}]
                if terminal == "lost"
                else [{"kind": "cancel", "request": "R0"}, {"kind": "stop", "request": "R0", "usage": "unknown"}]
            )
            if terminal == "lost":
                terminals.insert(0, {"kind": "cancel", "request": "R0"})
            cases.append(
                _case(
                    "materialization",
                    f"{path}_late_{terminal}",
                    _materialization_decl(spec),
                    [*prefix, *terminals, _materialization_event(spec, _items("late"))],
                )
            )
    # Historical inaccessible provenance mutations are explicit positive invariant aliases.
    for suffix in ("missing_parent", "foreign_parent", "invented_parent", "missing_source_fact"):
        cases.append(
            _case(
                "materialization",
                f"adaptive_{suffix}",
                _materialization_decl(adaptive_collection),
                _materialization_trace(adaptive_collection, _items("one")),
            )
        )
    bridge_decl = _materialization_decl(adaptive_collection)
    bridge_decl["runtime_mappings"] = [
        {"condition": "result", "failure": None, "reported_outcome": "ok", "outcome": "ok", "category": "success"}
    ]
    bridge_decl["outcome_categories"] = {"ok": "success"}
    cases.append(
        _case(
            "materialization",
            "adaptive_result_bridge",
            bridge_decl,
            [
                {"kind": "bridge_start", "task": "A0"},
                *_materialization_trace(adaptive_collection, _items("one")),
                {"kind": "bridge_condition", "task": "A0", "condition": "result", "reported_outcome": "ok"},
                {"kind": "bridge_emit", "task": "A0", "outcome": "ok", "category": "success"},
            ],
        )
    )
    return cases
