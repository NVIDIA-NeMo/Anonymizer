# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: map cases."""

from __future__ import annotations

from tests.graph_sdk.reference._effects_v1.builders import (
    _map_decl,
    _map_event,
)
from tests.graph_sdk.reference._effects_v1.case import (
    _case,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Object,
    _array,
    _object,
)


def _map_specs() -> list[Object]:
    cases: list[Object] = []
    for name, declaration in (
        ("admit_operation", _map_decl()),
        ("admit_subgraph", _map_decl(member_kind="subgraph")),
        ("admit_control_only", _map_decl(item_input=None)),
        ("default_override_dependency", _map_decl(default_dependencies=["unavailable_default"])),
        ("max_zero_scalar_join", _map_decl(max_children=0, outward_scalar="join")),
        ("max_one_scalar_ordinary", _map_decl(max_children=1, outward_scalar="ordinary")),
        ("max_one_scalar_workflow", _map_decl(max_children=1, outward_scalar="workflow_output")),
    ):
        cases.append(_case("map", name, declaration, [], "admission"))
    for destination in ("join", "ordinary", "workflow_output"):
        cases.append(
            _case(
                "map",
                f"max_two_scalar_{destination}",
                _map_decl(max_children=2, outward_scalar=destination),
                [],
                "admission",
            )
        )
    admission_mutants: list[tuple[str, Object]] = []
    missing_expansion = _map_decl()
    missing_expansion["expansions"] = []
    admission_mutants.append(("missing_expansion", missing_expansion))
    duplicate_map = _map_decl()
    _array(duplicate_map["maps"]).append(dict(_object(_array(duplicate_map["maps"])[0]), member="M1"))
    admission_mutants.append(("duplicate_map_source", duplicate_map))
    map_loop = _map_decl()
    map_loop["loops"] = [{"join": "LJ", "member": "LM", "owner": "W0", "scope": "S0", "starter": "E"}]
    admission_mutants.append(("map_loop_duplicate_source", map_loop))
    duplicate_loop = _map_decl()
    duplicate_loop["loops"] = [
        {"join": "LJ0", "member": "LM0", "owner": "W0", "scope": "S0", "starter": "L"},
        {"join": "LJ1", "member": "LM1", "owner": "W0", "scope": "S0", "starter": "L"},
    ]
    admission_mutants.append(("duplicate_loop_source", duplicate_loop))
    minimum = _map_decl()
    _object(_array(minimum["schemas"])[1])["minimum"] = 1
    admission_mutants.append(("context_minimum_conflict", minimum))
    schema_conflict = _map_decl()
    _array(schema_conflict["schemas"]).append(
        {"item_type": "bytes", "kind": "collection", "minimum": 0, "type": "members_t"}
    )
    admission_mutants.append(("collection_item_schema_conflict", schema_conflict))
    item_mismatch = _map_decl()
    _object(_array(item_mismatch["expansions"])[0])["item_type"] = "bytes"
    admission_mutants.append(("item_type_mismatch", item_mismatch))
    admission_mutants.append(("context_override_conflict", _map_decl(context_override=True)))
    admission_mutants.append(("dependency_summary_mismatch", _map_decl(retained_dependencies=["default_root"])))
    admission_mutants.append(("false_identity_summary", _map_decl(retained_identity=True)))
    foreign = _map_decl()
    _object(_array(foreign["maps"])[0])["owner"] = "W1"
    _array(foreign["maps"]).append(dict(_object(_array(foreign["maps"])[0]), member="M1"))
    admission_mutants.append(("foreign_before_duplicate", foreign))
    for name, declaration in admission_mutants:
        cases.append(_case("map", name, declaration, [], "admission"))
    for extra in (0, 1, 2):
        cases.append(
            _case(
                "map",
                f"membership_with_{extra}_other_outputs",
                _map_decl(),
                [_map_event(("a", "b"), extra_outputs=extra)],
            )
        )
    for size in (0, 1, 2, 3):
        cases.append(
            _case(
                "map",
                f"membership_{'one_over' if size == 3 else size}",
                _map_decl(),
                [_map_event(tuple(chr(97 + index) for index in range(size)))],
            )
        )
    malformed_events: list[tuple[str, Object]] = [
        ("missing_membership_port", _map_event(("a",), port="other")),
        ("wrong_membership_type", _map_event(("a",), artifact_type="text")),
    ]
    wrong_association = _map_event(("a",))
    wrong_association.pop("parent")
    wrong_association.update({"kind": "local_result", "supplied_association": "E0", "returned_association": "E1"})
    cases.append(_case("map", "wrong_parent", _map_decl(), [wrong_association], "local_callback"))
    duplicate_port = _map_event(("a",))
    _array(duplicate_port["outputs"]).append(dict(_object(_array(duplicate_port["outputs"])[0])))
    malformed_events.append(("duplicate_membership_port", duplicate_port))
    duplicate_item = _map_event(("a", "b"))
    _object(_array(_object(_array(duplicate_item["outputs"])[0])["items"])[1])["key"] = 0
    duplicate_item["kind"] = "collection_constructor"
    malformed_events.append(("duplicate_item", duplicate_item))
    noncanonical = _map_event(("a", "b"))
    _array(_object(_array(noncanonical["outputs"])[0])["items"]).reverse()
    noncanonical["kind"] = "collection_constructor"
    malformed_events.append(("noncanonical_items", noncanonical))
    for name, event in malformed_events:
        cases.append(_case("map", name, _map_decl(), [event]))
    cases.extend(
        [
            _case("map", "subgraph_item_binding", _map_decl(member_kind="subgraph"), [_map_event(("left", "right"))]),
            _case("map", "control_only_members", _map_decl(item_input=None), [_map_event(("a", "b"))]),
            _case(
                "map",
                "default_override_no_fallback",
                _map_decl(default_dependencies=["unavailable_default"]),
                [_map_event(("actual",))],
            ),
            _case(
                "map",
                "prospective_transition_rejected",
                _map_decl(),
                [{"kind": "close_parent", "parent": "E0", "category": "cancelled"}, _map_event(("a",))],
            ),
        ]
    )
    for name, limit, value in (
        ("artifact_count_one_over", "max_artifacts", 2),
        ("artifact_bytes_one_over", "max_artifact_bytes", 7),
        ("provenance_one_over", "max_provenance_edges", 1),
    ):
        declaration = _map_decl()
        _object(declaration["limits"])[limit] = value
        if name == "provenance_one_over":
            cases.append(_case("map", name, declaration, [], "map_execution_preflight"))
        else:
            cases.append(_case("map", name, declaration, [_map_event(("aa", "bb"))]))
    exact = _map_decl()
    _object(exact["limits"]).update({"max_artifacts": 3, "max_artifact_bytes": 8, "max_provenance_edges": 2})
    cases.append(_case("map", "bounds_exact", exact, [_map_event(("aa", "bb"))]))
    for maximum in (0, 1, 2):
        for destination in ("join", "ordinary", "workflow_output"):
            declaration = _map_decl(max_children=maximum, outward_scalar=destination)
            count = 0 if maximum == 0 else 1
            cases.append(
                _case(
                    "map",
                    f"resolve_{destination}_max_{maximum}",
                    declaration,
                    [
                        {
                            "destination": destination,
                            "kind": "resolve_scalar",
                            "maximum": maximum,
                            "member_count": count,
                            "source_kind": "map",
                        }
                    ],
                )
            )
    for destination in ("join", "ordinary", "workflow_output"):
        cases.append(
            _case(
                "map",
                f"resolve_{destination}_max_1_empty",
                _map_decl(max_children=1, outward_scalar=destination),
                [
                    {
                        "destination": destination,
                        "kind": "resolve_scalar",
                        "maximum": 1,
                        "member_count": 0,
                        "source_kind": "map",
                    }
                ],
            )
        )
    for name, outcomes in (
        ("loop_exit", ["again", "exit"]),
        ("loop_bypass", []),
        ("loop_prior_continue", ["again"]),
        ("loop_failure", ["again", "failure"]),
        ("loop_overflow", ["again", "again"]),
    ):
        cases.append(
            _case(
                "map",
                name,
                _map_decl(),
                [{"kind": "resolve_scalar", "outcomes": outcomes, "source_kind": "loop"}],
            )
        )
    for name, maximum, values in (
        ("collection_items_exact", 2, ("a", "b")),
        ("collection_items_one_over", 1, ("a", "b")),
        ("overflow_collection_storage_one_over", 2, ("a", "b", "c")),
        ("overflow_collection_storage_exact", 3, ("a", "b", "c")),
    ):
        declaration = _map_decl()
        _object(declaration["limits"])["max_collection_items"] = maximum
        cases.append(_case("map", name, declaration, [_map_event(values)]))
    invalid_limit = _map_decl()
    _object(invalid_limit["limits"])["max_collection_items"] = True
    cases.append(_case("map", "collection_items_invalid_limit", invalid_limit, [], "admission"))
    # Historical parser-only ID is an explicit redundant alias of the real rejection.
    prospective = next(case for case in cases if case["case_id"] == "map/prospective_transition_rejected")
    cases.append(
        _case(
            "map",
            "caller_transition_verdict_rejected",
            _object(prospective["declaration"]),
            [_object(event) for event in _array(prospective["events"])],
        )
    )
    return cases
