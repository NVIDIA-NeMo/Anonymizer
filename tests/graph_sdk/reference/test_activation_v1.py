# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused self-tests for the finite activation reference and frozen corpus."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import TypeAlias, cast

import pytest

from tests.graph_sdk.reference import activation_v1 as reference
from tests.graph_sdk.reference.corpora import corpus_bytes
from tests.graph_sdk.reference.source_layout import assert_isolated_generation, assert_reference_imports, source_digest

Json: TypeAlias = reference.Json
Object: TypeAlias = reference.Object
HERE = Path(__file__).parent
GENERATOR = HERE / "activation_v1.py"
CORPUS = HERE / "activation_v1_cases.json"
MANIFEST_PATH = HERE / "activation_v1_manifest.json"
SUPPORT = HERE / "activation_v1_support.md"
FROZEN_BYTES = corpus_bytes("activation")
CASES = reference.load_cases(json.loads(FROZEN_BYTES))
MANIFEST = cast(Object, json.loads(MANIFEST_PATH.read_bytes()))


@pytest.fixture(scope="module")
def generations() -> tuple[tuple[Object, ...], tuple[Object, ...]]:
    return reference.generate_cases(), reference.generate_cases()


def _object(value: Json) -> Object:
    assert isinstance(value, dict)
    return cast(Object, value)


def _array(value: Json) -> list[Json]:
    assert isinstance(value, list)
    return value


def _cases(family: str) -> tuple[Object, ...]:
    return tuple(case for case in CASES if case["family"] == family)


def _case(suffix: str) -> Object:
    return next(case for case in CASES if case["case_id"] == suffix)


def _reduce(case: Object, *, events: list[Object] | None = None) -> Object:
    declaration = cast(Object, case["declaration"])
    if case["boundary"] == "dynamic_admission":
        return reference.admit_dynamic_workflow(declaration)
    if case["boundary"] == "static_admission":
        return reference.admit_static_support(declaration)
    supplied = events if events is not None else cast(list[Object], case["events"])
    return reference.reduce_trace(declaration, supplied)


def _state(result: Object) -> Object:
    return _object(result["state"])


def test_generation_twice_is_byte_identical_and_hashes_frozen_bytes(
    generations: tuple[tuple[Object, ...], tuple[Object, ...]],
) -> None:
    first, second = generations
    assert reference.canonical_bytes(first) == reference.canonical_bytes(second) == FROZEN_BYTES
    assert hashlib.sha256(FROZEN_BYTES).hexdigest() == MANIFEST["corpus_sha256"]


def test_every_result_is_reduced_from_events_and_traces_preserve_full_state() -> None:
    for case in CASES:
        assert _reduce(case) == case["expected"], case["case_id"]
        base = _object(case["expected"])
        for trace_value in _array(case["traces"]):
            trace = _object(trace_value)
            actual = (
                reference.admit_dynamic_workflow(_object(trace["declaration"]))
                if trace["boundary"] == "dynamic_admission"
                else reference.admit_static_support(_object(trace["declaration"]))
                if trace["boundary"] == "static_admission"
                else reference.reduce_trace(_object(trace["declaration"]), cast(list[Object], trace["events"]))
            )
            assert actual == trace["expected"], case["case_id"]
            if actual["status"] != "accepted" or base["status"] != "accepted":
                continue
            normalized = cast(Json, _state(actual))
            if trace["name"] == "rename":
                normalized = reference.alpha_normalize_state(_object(normalized), inverse=True)
            assert normalized == base["state"], case["case_id"]
            assert _object(cast(Json, normalized))["semantic_hash"] == _state(base)["semantic_hash"]


def test_family_enumeration_coordinates_and_nested_witness_are_exact() -> None:
    assert tuple(dict.fromkeys(cast(str, case["family"]) for case in CASES)) == reference.FAMILY_IDS
    assert Counter(cast(str, case["family"]) for case in CASES) == {
        "sequence_single": 6,
        "sequence_linked_pair": 36,
        "sequence_independent_siblings": 36,
        "sequence_mutations": 11,
        "choice": 8,
        "subgraph": 10,
        "map": 25,
        "join": 26,
        "loop": 17,
        "nested_map_loop": 11,
        "precedence": 7,
        "terminal_coverage": 15,
    }
    table: dict[tuple[int, int], int] = {}
    for case in _cases("nested_map_loop")[:9]:
        name = cast(str, case["case_id"]).rsplit("/", 1)[1]
        _, maps, _, loops = name.split("_")
        entries = _array(_state(_object(case["expected"]))["entries"])
        table[int(maps), int(loops)] = len(entries)
        assert len({cast(str, _object(entry)["activation"]) for entry in entries}) == len(entries)
        declaration = _object(case["declaration"])
        aggregates = [_object(value) for value in _array(declaration["aggregates"])]
        assert aggregates[0]["kind"] == "map"
        assert len([item for item in aggregates if item["kind"] == "loop"]) == int(maps)
        assert len(_array(declaration["subgraphs"])) == int(maps)
    assert table == {(m, i): 2 + m * (3 + i) for m in (0, 1, 2) for i in (0, 1, 2)}
    assert table[2, 2] == 12


def test_sibling_cases_have_explicit_common_sink_and_preserve_commuted_suffix() -> None:
    cases = _cases("sequence_independent_siblings")
    assert len(cases) == 36
    for case in cases:
        declaration = _object(case["declaration"])
        assert [cast(str, _object(seed)["template"]) for seed in _array(declaration["seeds"])] == ["N0", "N1", "N2"]
        assert declaration["edges"] == [["A0", "A2"], ["A1", "A2"]]
        assert declaration["input_dependencies"] == []
        events = [_object(event) for event in _array(case["events"])]
        assert [(event["kind"], event.get("key")) for event in events[-2:]] == [
            ("start", "A2"),
            ("terminal_success", "A2"),
        ]
        commute = next(
            _object(trace)
            for trace in _array(case["traces"])
            if _object(trace)["name"] == "commute_independent_siblings"
        )
        assert _array(commute["events"])[-2:] == _array(case["events"])[-2:]


def test_two_node_subgraphs_use_distinct_scoped_templates() -> None:
    for case_id in ("subgraph/002/body_2_success", "subgraph/003/body_2_failure"):
        seeds = [_object(seed) for seed in _array(_object(_case(case_id)["declaration"])["seeds"])]
        assert [(seed["template"], seed["scope"]) for seed in seeds] == [
            ("N0", []),
            ("N1", ["N0"]),
            ("N2", ["N0"]),
        ]
        subgraph = _object(_array(_object(_case(case_id)["declaration"])["subgraphs"])[0])
        assert subgraph == {"parent": "A0", "roots": ["A1"], "sink": "A2"}
        first_outcomes = [
            _object(outcome)
            for outcome in _array(_object(_case(case_id)["declaration"])["outcomes"])
            if (_object(outcome)["scope"], _object(outcome)["template"]) == (["N0"], "N1")
        ]
        assert [(outcome["name"], outcome["produced_ports"]) for outcome in first_outcomes] == [("ok", ["result"])]


def test_value_dependencies_require_the_named_source_port() -> None:
    linked = _cases("sequence_linked_pair")
    assert len(linked) == 36
    for case in linked:
        declaration = _object(case["declaration"])
        producer_outcomes = [
            _object(outcome)
            for outcome in _array(declaration["outcomes"])
            if (_object(outcome)["scope"], _object(outcome)["template"]) == ([], "N0")
        ]
        assert [(outcome["name"], outcome["produced_ports"]) for outcome in producer_outcomes] == [("ok", ["result"])]
        dependency = _object(_array(declaration["input_dependencies"])[0])
        assert dependency == {
            "destination": "A1",
            "destination_port": "input",
            "source": "A0",
            "source_port": "result",
        }
        left_category = cast(str, case["case_id"]).split("/")[1].split("-")[0]
        left_terminal = _object(_array(case["events"])[3])
        if left_category != "000":
            assert left_terminal["outcome"] is None

    negative = _case("sequence_mutations/010/named_missing_result")
    assert negative["boundary"] == "static_admission"
    assert _object(negative["expected"])["code"] == "missing"
    carry_only = reference._decl(
        (reference._seed("A0", "N0"), reference._seed("A1", "N1")),
        required=("A0", "A1"),
        edges=(("A0", "A1"),),
    )
    result = reference.reduce_trace(
        carry_only,
        [
            reference._event("initialize"),
            reference._select("A0", "A1"),
            reference._event("start", key="A0"),
            reference._terminal("A0", "success", "again"),
        ],
    )
    entries = {_object(entry)["activation"]: _object(entry) for entry in _array(_state(result)["entries"])}
    assert entries["A1"]["status"] == "blocked"


def test_named_transition_perturbations_change_the_verdict_or_state() -> None:
    probes = (
        "sequence_mutations/000/start_before_ready",
        "choice/000/ok_forward",
        "subgraph/000/body_1_success",
        "map/014/failed_partial",
        "join/006/children_2_success-failure",
        "loop/015/abnormal_member_loss",
        "terminal_coverage/011/closed_2_2",
    )
    for case_id in probes:
        case = _case(case_id)
        events = cast(list[Object], case["events"])
        without_last = _reduce(case, events=events[:-1])
        assert without_last != case["expected"], case_id


def test_actual_limits_exact_capacity_and_optional_event_reserve() -> None:
    for case_id in (
        "sequence_single/000/base",
        "choice/000/ok_forward",
        "map/000/bound_0_size_0_empty",
        "subgraph/000/body_1_success",
    ):
        case = _case(case_id)
        declaration = _object(case["declaration"])
        seeds = _array(declaration["seeds"])
        limits = _object(declaration["limits"])
        map_count = sum(_object(seed)["role"] == "map_expander" for seed in seeds)
        assert limits["max_entries"] == len(seeds)
        assert limits["max_events"] == 3 * len(seeds) + map_count
        assert cast(int, limits["max_parent_depth"]) >= 1
    ordinary = _case("sequence_single/000/base")
    declaration = _object(ordinary["declaration"])
    for field in ("max_events", "max_entries", "max_parent_depth"):
        changed = dict(declaration)
        limits = dict(_object(declaration["limits"]))
        limits[field] = cast(int, limits[field]) - 1
        changed["limits"] = cast(Json, limits)
        assert reference.reduce_trace(changed, cast(list[Object], ordinary["events"]))["code"] == "limit_exceeded"
    settled = cast(list[Object], ordinary["events"])
    reselection = settled[:-1] + [reference._select("A0")] + settled[-1:]
    assert reference.reduce_trace(declaration, reselection)["code"] == "limit_exceeded"
    spare = reference._spare(declaration)
    assert reference.reduce_trace(spare, reselection)["status"] == "accepted"


def test_membership_join_loop_subgraph_and_completion_are_derived() -> None:
    zero = _case("loop/003/enter_bound_zero")
    expansion = _object(_array(_state(_object(zero["expected"]))["expansions"])[0])
    assert expansion == {"members": [], "parent": "A0", "status": "overflow"}
    failed_empty = _case("map/013/failed_empty")
    failed_partial = _case("map/014/failed_partial")
    assert _object(_array(_state(_object(failed_empty["expected"]))["expansions"])[0])["members"] == []
    assert _object(_array(_state(_object(failed_partial["expected"]))["expansions"])[0])["members"] == ["A1"]
    mixed = _case("join/006/children_2_success-failure")
    entries = {
        _object(entry)["activation"]: _object(entry) for entry in _array(_state(_object(mixed["expected"]))["entries"])
    }
    assert entries["A11"]["status"] == "blocked"
    ready = _case("join/000/children_0_empty")
    assert _state(_object(ready["expected"]))["complete"] is False
    ready_entries = {
        _object(entry)["activation"]: _object(entry) for entry in _array(_state(_object(ready["expected"]))["entries"])
    }
    assert ready_entries["A11"]["status"] == "ready"
    subgraph = _case("subgraph/000/body_1_success")
    sub_entries = {
        _object(entry)["activation"]: _object(entry)
        for entry in _array(_state(_object(subgraph["expected"]))["entries"])
    }
    assert sub_entries["A0"]["status"] == "success"


def test_subgraph_waits_for_every_body_root_and_nested_dynamic_obligation() -> None:
    multi_root = reference._decl(
        (
            reference._seed("A0", role="subgraph"),
            reference._seed("A1", "N1", parent="A0"),
            reference._seed("A2", "N2", parent="A0"),
        ),
        required=("A0",),
        subgraphs=({"parent": "A0", "roots": ["A1", "A2"], "sink": "A2"},),
    )
    sink_first = [
        reference._event("initialize"),
        reference._select("A0"),
        reference._event("start", key="A0"),
        reference._event("start", key="A2"),
        reference._terminal("A2", "success", "ok"),
    ]
    partial = reference.reduce_trace(multi_root, sink_first)
    partial_entries = {_object(value)["activation"]: _object(value) for value in _array(_state(partial)["entries"])}
    assert partial_entries["A0"]["status"] == "running"
    assert partial_entries["A1"]["status"] == "ready"
    closed = reference.reduce_trace(
        multi_root,
        sink_first + [reference._event("start", key="A1"), reference._terminal("A1", "success", "ok")],
    )
    assert {_object(value)["activation"]: _object(value) for value in _array(_state(closed)["entries"])}["A0"][
        "status"
    ] == "success"

    nested_dynamic = reference._decl(
        (
            reference._seed("A0", role="subgraph"),
            reference._seed("A1", parent="A0", role="map_expander"),
            reference._seed("A2", "N1", parent="A1", role="map_member"),
            reference._seed("A3", "N2", parent="A0", role="join"),
            reference._seed("A4", "N2", parent="A0"),
        ),
        required=("A0",),
        subgraphs=({"parent": "A0", "roots": ["A1", "A3", "A4"], "sink": "A4"},),
        aggregates=(
            {
                "accepted_categories": ["success"],
                "bound": 1,
                "expansion_outcomes": ["ok"],
                "join": "A3",
                "kind": "map",
                "members": ["A2"],
                "parent": "A1",
            },
        ),
    )
    nested_events = [
        reference._event("initialize"),
        reference._select("A0"),
        reference._event("start", key="A0"),
        reference._event("start", key="A1"),
        reference._terminal("A1", "success", "ok"),
        reference._event("membership_open", parent="A1", members=["A2"]),
        reference._event("start", key="A2"),
        reference._event("start", key="A4"),
        reference._terminal("A4", "success", "ok"),
    ]
    nested = reference.reduce_trace(reference._spare(nested_dynamic, 3), nested_events)
    nested_entries = {_object(value)["activation"]: _object(value) for value in _array(_state(nested)["entries"])}
    assert nested_entries["A0"]["status"] == "running"
    assert nested_entries["A2"]["status"] == "running"
    nested_closed = reference.reduce_trace(
        reference._spare(nested_dynamic, 3),
        nested_events
        + [
            reference._terminal("A2", "success", "ok"),
            reference._event("membership_close", parent="A1", members=["A2"]),
            reference._event("start", key="A3"),
            reference._terminal("A3", "success", "ok"),
        ],
    )
    assert {_object(value)["activation"]: _object(value) for value in _array(_state(nested_closed)["entries"])}["A0"][
        "status"
    ] == "success"


def test_loop_admission_uses_starter_initial_carry_and_iteration_order() -> None:
    positive = _case("loop/006/bound_2_executed_2_stop")
    events = [_object(value) for value in _array(positive["events"])]
    assert [event["key"] for event in events if event["kind"] == "start"] == ["A0", "A1", "A2"]
    assert _object(positive["expected"])["status"] == "accepted"
    wrong_iteration = _case("loop/010/wrong_iteration")
    wrong_events = cast(list[Object], wrong_iteration["events"])
    assert reference.reduce_trace(_object(wrong_iteration["declaration"]), wrong_events[:-1])["status"] == "accepted"
    positive_next = wrong_events[:-1] + [reference._event("start", key="A1")]
    next_result = reference.reduce_trace(_object(wrong_iteration["declaration"]), positive_next)
    assert next_result["status"] == "accepted"
    assert {_object(value)["activation"]: _object(value) for value in _array(_state(next_result)["entries"])}["A1"][
        "status"
    ] == "running"
    for case_id, code in (
        ("loop/010/wrong_iteration", "missing"),
        ("loop/011/duplicate_iteration", "duplicate"),
        ("loop/012/foreign_iteration", "foreign_owner"),
        ("loop/014/terminal_gap", "missing"),
    ):
        assert _object(_case(case_id)["expected"])["code"] == code
    duplicate = _case("loop/011/duplicate_iteration")
    duplicate_seeds = [_object(value) for value in _array(_object(duplicate["declaration"])["seeds"])]
    assert [(seed["key"], seed["template"]) for seed in duplicate_seeds if seed["key"] == "A1"] == [
        ("A1", "N1"),
        ("A1", "N2"),
    ]
    assert duplicate["boundary"] == "initialization"
    assert duplicate["events"] == [{"kind": "initialize"}]
    for case_id in ("loop/008/missing_initial", "loop/009/missing_carried"):
        case = _case(case_id)
        assert case["boundary"] == "dynamic_admission"
        assert case["events"] == []
        assert _object(case["expected"])["code"] == "missing"
    for case_id in ("loop/015/abnormal_member_loss", "loop/016/missing_carried_output"):
        state = _state(_object(_case(case_id)["expected"]))
        expansion = _object(_array(state["expansions"])[0])
        entries = {_object(value)["activation"]: _object(value) for value in _array(state["entries"])}
        assert expansion["status"] == "failed"
        assert entries["A11"]["status"] == "blocked"
    carried_output = _case("loop/016/missing_carried_output")
    carried_aggregate = _object(_array(_object(carried_output["declaration"])["aggregates"])[0])
    assert carried_aggregate["carried_binding"] == {
        "destination_port": "input",
        "source_kind": "member_output",
        "source_port": "carry",
    }
    assert "A2" not in {
        cast(str, _object(entry)["activation"])
        for entry in _array(_state(_object(carried_output["expected"]))["entries"])
    }


def test_loop_outcome_partitions_are_exact_and_scoped() -> None:
    for case in (*_cases("loop"), *_cases("nested_map_loop")):
        declaration = _object(case["declaration"])
        seeds = {cast(str, _object(seed)["key"]): _object(seed) for seed in _array(declaration["seeds"])}
        outcomes: dict[tuple[tuple[str, ...], str], set[str]] = {}
        for value in _array(declaration["outcomes"]):
            outcome = _object(value)
            identity = (tuple(cast(list[str], outcome["scope"])), cast(str, outcome["template"]))
            outcomes.setdefault(identity, set()).add(cast(str, outcome["name"]))
        for value in _array(declaration["aggregates"]):
            aggregate = _object(value)
            if aggregate["kind"] != "loop":
                continue
            starter = seeds[cast(str, aggregate["starter"])]
            starter_identity = (tuple(cast(list[str], starter["scope"])), cast(str, starter["template"]))
            member_identity = (
                tuple(cast(list[str], aggregate["member_scope"])),
                cast(str, aggregate["member_template"]),
            )
            assert outcomes[starter_identity] == set(
                (*cast(list[str], aggregate["enter_outcomes"]), *cast(list[str], aggregate["bypass_outcomes"]))
            )
            assert outcomes[member_identity] == set(
                (*cast(list[str], aggregate["continue_outcomes"]), *cast(list[str], aggregate["exit_outcomes"]))
            )
            if case["boundary"] not in ("dynamic_admission", "initialization"):
                assert reference.admit_dynamic_workflow(declaration)["status"] == "accepted"

    nested = _object(_case("nested_map_loop/008/map_2_loop_2")["declaration"])
    names_by_identity: dict[tuple[tuple[str, ...], str], set[str]] = {}
    for value in _array(nested["outcomes"]):
        outcome = _object(value)
        identity = (tuple(cast(list[str], outcome["scope"])), cast(str, outcome["template"]))
        names_by_identity.setdefault(identity, set()).add(cast(str, outcome["name"]))
    assert names_by_identity[((), "N0")] == {"ok", "fail", "again", "stop"}
    assert names_by_identity[((), "N1")] == {"ok", "fail", "again", "stop"}
    assert names_by_identity[(("N1",), "N0")] == {"again", "stop"}
    assert names_by_identity[(("N1",), "N1")] == {"again", "stop"}

    malformed = dict(nested)
    malformed["outcomes"] = [
        *cast(list[Json], nested["outcomes"]),
        {
            "category": "success",
            "name": "extra",
            "produced_ports": [],
            "scope": ["N1"],
            "template": "N0",
        },
    ]
    assert reference.admit_dynamic_workflow(malformed)["code"] == "missing"


def test_membership_is_owned_bounded_monotone_duplicate_sensitive_and_conjunctive() -> None:
    declaration = reference._aggregate_decl(1)
    seed_keys = [cast(str, _object(value)["key"]) for value in _array(declaration["seeds"])]
    assert seed_keys == ["A0", "A11", "A1"]
    unselected = [reference._event("initialize"), reference._event("membership_close", parent="A0", members=[])]
    assert reference.reduce_trace(declaration, unselected)["code"] == "missing"
    over_bound = [
        reference._event("initialize"),
        reference._select("A0", "A11"),
        reference._event("membership_close", parent="A0", members=["A1", "A2"]),
    ]
    assert reference.reduce_trace(declaration, over_bound)["code"] == "limit_exceeded"
    duplicate = [
        reference._event("initialize"),
        reference._select("A0", "A11"),
        reference._event("membership_open", parent="A0", members=["A1", "A1"]),
    ]
    assert reference.reduce_trace(declaration, duplicate)["code"] == "duplicate"
    closed = reference._map_events(("A1",), ("success",))
    repeated = closed + [reference._event("membership_close", parent="A0", members=["A1"])]
    accepted = reference.reduce_trace(reference._spare(declaration), repeated)
    assert accepted["status"] == "accepted"
    open_case = _case("join/025/open_complete_survivors")
    entries = {
        _object(value)["activation"]: _object(value)
        for value in _array(_state(_object(open_case["expected"]))["entries"])
    }
    assert entries["A11"]["status"] == "unstarted"
    overflow_with_ready_child = [
        reference._event("initialize"),
        reference._select("A0", "A11"),
        reference._event("membership_open", parent="A0", members=["A1"]),
        reference._event("start", key="A0"),
        reference._terminal("A0", "success", "ok"),
        reference._event("membership_overflow", parent="A0", observed_count=2),
    ]
    ready_overflow = reference.reduce_trace(reference._spare(declaration, 3), overflow_with_ready_child)
    ready_entries = {
        _object(value)["activation"]: _object(value) for value in _array(_state(ready_overflow)["entries"])
    }
    assert ready_entries["A1"]["status"] == "ready"
    assert _state(ready_overflow)["complete"] is False
    overflow_with_running_child = (
        overflow_with_ready_child[:3] + [reference._event("start", key="A1")] + overflow_with_ready_child[3:]
    )
    overflow = reference.reduce_trace(reference._spare(declaration, 3), overflow_with_running_child)
    overflow_state = _state(overflow)
    assert _object(_array(overflow_state["expansions"])[0])["members"] == ["A1"]
    assert overflow_state["complete"] is False
    overflow_entries = {_object(value)["activation"]: _object(value) for value in _array(overflow_state["entries"])}
    assert overflow_entries["A1"]["status"] == "running"
    completed = reference.reduce_trace(
        reference._spare(declaration, 3),
        overflow_with_running_child + [reference._terminal("A1", "success", "ok")],
    )
    assert _state(completed)["complete"] is True


def test_closed_empty_membership_waits_for_admitted_expander_outcome() -> None:
    declaration = reference._aggregate_decl(0)
    closed_empty = [
        reference._event("initialize"),
        reference._select("A0", "A11"),
        reference._event("membership_close", parent="A0", members=[]),
    ]
    before_terminal = reference.reduce_trace(declaration, closed_empty)
    before_entries = {
        _object(value)["activation"]: _object(value) for value in _array(_state(before_terminal)["entries"])
    }
    assert before_entries["A11"]["status"] == "unstarted"

    after_terminal = reference.reduce_trace(
        declaration,
        closed_empty + [reference._event("start", key="A0"), reference._terminal("A0", "success", "ok")],
    )
    after_entries = {
        _object(value)["activation"]: _object(value) for value in _array(_state(after_terminal)["entries"])
    }
    assert after_entries["A11"]["status"] == "ready"


def test_duplicate_sensitive_declaration_inputs_reject_before_canonicalization() -> None:
    repeated_required = reference._decl((reference._seed("A0"),), required=("A0", "A0"))
    assert reference.reduce_trace(repeated_required, [reference._event("initialize")])["code"] == "duplicate"
    repeated_edge = reference._decl(
        (reference._seed("A0"), reference._seed("A1")),
        edges=(("A0", "A1"), ("A0", "A1")),
    )
    assert reference.reduce_trace(repeated_edge, [reference._event("initialize")])["code"] == "duplicate"


def test_depth_limit_precedes_duplicate_seed_for_every_seed_permutation() -> None:
    root = reference._seed("A0")
    unparented = reference._seed("A1")
    parented = reference._seed("A1", parent="A0")
    for seeds in ((root, unparented, parented), (root, parented, unparented)):
        declaration = reference._decl(seeds, limits=(9, 3, 1))
        assert reference.reduce_trace(declaration, [reference._event("initialize")])["code"] == "limit_exceeded"


def test_parent_cycles_have_no_finite_depth_but_do_not_hide_finite_overflow() -> None:
    cyclic_seeds = [
        reference._seed("A0", parent="A1"),
        reference._seed("A1", parent="A0"),
    ]
    for maximum_depth in (0, 1, 2, 12):
        declaration = reference._decl((reference._seed("A0"), reference._seed("A1")), limits=(6, 2, maximum_depth))
        declaration["seeds"] = cyclic_seeds
        assert reference.reduce_trace(declaration, [reference._event("initialize")])["code"] == "cycle"

    mixed = reference._decl(
        (
            reference._seed("A0"),
            reference._seed("A1"),
            reference._seed("A2"),
            reference._seed("A3", parent="A2"),
        ),
        limits=(12, 4, 1),
    )
    mixed["seeds"] = cyclic_seeds + [reference._seed("A2"), reference._seed("A3", parent="A2")]
    assert reference.reduce_trace(mixed, [reference._event("initialize")])["code"] == "limit_exceeded"


def test_named_outcomes_categories_ports_and_invocation_ownership_are_exact() -> None:
    case = _case("sequence_single/000/base")
    state = _state(_object(case["expected"]))
    entry = _object(_array(state["entries"])[0])
    assert entry["outcome"] == "ok"
    assert entry["produced_ports"] == ["result"]
    assert entry["scope"] == []
    wrong_category = [
        reference._event("initialize"),
        reference._select("A0"),
        reference._event("start", key="A0"),
        reference._terminal("A0", "failure", "ok"),
    ]
    assert reference.reduce_trace(_object(case["declaration"]), wrong_category)["code"] == "contradictory"
    unknown = [
        reference._event("initialize"),
        reference._select("A0"),
        reference._event("start", key="A0"),
        reference._terminal("A0", "success", "unknown"),
    ]
    assert reference.reduce_trace(_object(case["declaration"]), unknown)["code"] == "invalid_value"
    foreign = [reference._event("initialize"), reference._event("select", keys=["A0"], invocation="I1")]
    assert reference.reduce_trace(_object(case["declaration"]), foreign)["code"] == "foreign_owner"


def test_initialization_precedence_and_event_parsing_are_boundary_local() -> None:
    declaration = reference._decl((reference._seed("A0", invocation="I1"),), required=("A0",), limits=(0, 0, 0))
    assert reference.reduce_trace(declaration, [reference._event("initialize")])["code"] == "limit_exceeded"
    ordinary = _case("sequence_single/000/base")
    events = [
        reference._event("initialize"),
        reference._event("start", key="A0"),
        {"kind": "start", "key": 7, "invocation": "I0"},
    ]
    assert reference.reduce_trace(_object(ordinary["declaration"]), events)["code"] == "missing"
    exhausted = cast(list[Object], ordinary["events"])
    malformed = exhausted + [{"kind": "start", "key": 7, "invocation": "I0"}]
    assert reference.reduce_trace(_object(ordinary["declaration"]), malformed)["code"] == "invalid_type"
    foreign = exhausted + [reference._event("start", key="A11", invocation="I1")]
    assert reference.reduce_trace(_object(ordinary["declaration"]), foreign)["code"] == "limit_exceeded"


def test_exact_depth_required_occurrences_and_prospective_capacity() -> None:
    chain = reference._decl((reference._seed("A0"), reference._seed("A1", parent="A0")), limits=(6, 2, 2))
    assert reference.reduce_trace(chain, [reference._event("initialize")])["status"] == "accepted"
    shallow = dict(chain)
    shallow["limits"] = {"max_entries": 2, "max_events": 6, "max_parent_depth": 1}
    assert reference.reduce_trace(shallow, [reference._event("initialize")])["code"] == "limit_exceeded"
    for case_id in ("choice/000/ok_forward", "map/002/bound_1_size_1_s", "subgraph/000/body_1_success"):
        case = _case(case_id)
        declaration = dict(_object(case["declaration"]))
        limits = dict(_object(declaration["limits"]))
        limits["max_entries"] = cast(int, limits["max_entries"]) - 1
        declaration["limits"] = cast(Json, limits)
        assert reference.reduce_trace(declaration, cast(list[Object], case["events"]))["code"] == "limit_exceeded"
    running = _case("map/016/closed_missing_terminal")
    assert _state(_object(running["expected"]))["complete"] is False
    declaration = reference._aggregate_decl(1)
    membership_first = [
        reference._event("initialize"),
        reference._select("A0", "A11"),
        reference._event("membership_close", parent="A0", members=["A1"]),
        reference._event("start", key="A1"),
        reference._terminal("A1", "success", "ok"),
        reference._event("start", key="A0"),
        reference._terminal("A0", "success", "ok"),
    ]
    assert reference.reduce_trace(declaration, membership_first)["status"] == "accepted"
    running_failed = [
        reference._event("initialize"),
        reference._select("A0", "A11"),
        reference._event("membership_open", parent="A0", members=["A1"]),
        reference._event("start", key="A1"),
        reference._event("start", key="A0"),
        reference._terminal("A0", "failure", "fail"),
    ]
    result = reference.reduce_trace(reference._spare(declaration), running_failed)
    assert result["status"] == "accepted"
    assert _state(result)["complete"] is False


def test_all_five_abnormal_outcomes_are_none_and_produce_no_output() -> None:
    cases = [case for case in _cases("sequence_mutations") if "/abnormal_" in cast(str, case["case_id"])]
    assert {cast(str, case["case_id"]).rsplit("_", 1)[1] for case in cases} == set(reference.CATEGORIES[1:])
    for case in cases:
        state = _state(_object(case["expected"]))
        entry = _object(_array(state["entries"])[0])
        assert entry["outcome"] is None
        assert state["outputs"] == []
    assert _object(_case("sequence_mutations/009/success_without_outcome")["expected"])["code"] == "invalid_value"


def test_precedence_comes_from_actual_malformed_boundaries() -> None:
    assert [_object(case["expected"])["code"] for case in _cases("precedence")] == [
        "invalid_type",
        "limit_exceeded",
        "foreign_owner",
        "duplicate",
        "missing",
        "overlap",
        "cycle",
    ]
    forbidden = {
        "defects",
        "defect_classes",
        "mutation",
        "categories",
        "covered",
        "expansion_status",
        "join_status",
        "failure_cause",
    }
    for case in CASES:
        assert forbidden.isdisjoint(_object(case["declaration"]))

    case = _case("precedence/004/missing_before_contradictory")
    declaration = _object(case["declaration"])
    assert declaration["required"] == ["A0", "A11"]
    assert declaration["limits"] == {"max_entries": 3, "max_events": 10, "max_parent_depth": 2}
    seeds = [_object(seed) for seed in _array(declaration["seeds"])]
    assert [(seed["key"], seed["template"], seed["parent"]) for seed in seeds] == [
        ("A11", "N2", None),
        ("A1", "N1", "A11"),
    ]
    assert (
        _object(reference.reduce_trace(declaration, [_object(event) for event in _array(case["events"])]))["code"]
        == "missing"
    )

    contradictory_only = dict(declaration)
    contradictory_only["seeds"] = [reference._seed("A0", "N0", role="map_expander"), *seeds]
    assert reference.reduce_trace(contradictory_only, [reference._event("initialize")])["code"] == "contradictory"


def test_semantic_sets_are_order_independent_and_siblings_commute() -> None:
    case = _case("sequence_independent_siblings/000-001/base")
    traces = [_object(value) for value in _array(case["traces"])]
    commute = next(trace for trace in traces if trace["name"] == "commute_independent_siblings")
    assert commute["expected"] == case["expected"]
    declaration = _object(case["declaration"])
    reversed_seeds = list(reversed(_array(declaration["seeds"])))
    permuted = dict(declaration)
    permuted["seeds"] = reversed_seeds
    assert reference.reduce_trace(permuted, cast(list[Object], case["events"])) == case["expected"]


def test_case_boundaries_and_scoped_identity_are_explicit() -> None:
    assert {cast(str, case["boundary"]) for case in CASES} <= {
        "event_construction",
        "static_admission",
        "dynamic_admission",
        "initialization",
        "transition",
    }
    assert [case["boundary"] for case in _cases("precedence")] == [
        "event_construction",
        "transition",
        "initialization",
        "initialization",
        "initialization",
        "static_admission",
        "static_admission",
    ]
    nested = _case("nested_map_loop/008/map_2_loop_2")
    seeds = [_object(seed) for seed in _array(_object(nested["declaration"])["seeds"])]
    identities = [(tuple(cast(list[str], seed["scope"])), cast(str, seed["template"])) for seed in seeds]
    assert identities.count((("N1",), "N0")) == 2
    assert ((), "N0") in identities
    renamed = _object(reference._rename(nested["declaration"]))
    renamed_scopes = {tuple(cast(list[str], _object(seed)["scope"])) for seed in _array(renamed["seeds"])}
    assert renamed_scopes == {(), ("N0",)}
    renamed_result = reference.reduce_trace(
        renamed,
        cast(list[Object], reference._rename(nested["events"])),
    )
    assert reference.alpha_normalize_state(_state(renamed_result), inverse=True) == _state(_object(nested["expected"]))
    expected_scopes = {
        tuple(cast(list[str], _object(entry)["scope"]))
        for entry in _array(_state(_object(nested["expected"]))["entries"])
    }
    assert expected_scopes == {(), ("N1",)}
    assert _case("choice/004/foreign_selector")["boundary"] == "initialization"
    duplicate_missing = _case("precedence/003/duplicate_before_missing")
    assert duplicate_missing["events"] == [{"kind": "initialize"}]
    duplicate_missing_seeds = [_object(seed) for seed in _array(_object(duplicate_missing["declaration"])["seeds"])]
    assert [(seed["key"], seed["template"]) for seed in duplicate_missing_seeds] == [
        ("A0", "N0"),
        ("A0", "N1"),
    ]
    assert _object(duplicate_missing["declaration"])["required"] == ["A0", "A1"]
    for case in CASES:
        if case["boundary"] == "transition" and case["mode"] == "rejected":
            initialized = reference.reduce_trace(_object(case["declaration"]), [reference._event("initialize")])
            assert initialized["status"] == "accepted", case["case_id"]


def test_manifest_counts_sources_and_provenance_are_actual() -> None:
    assert MANIFEST["consumption_addendum_sha256"] == reference.CONSUMPTION_ADDENDUM_SHA256
    assert MANIFEST["counts"] == reference.counts(CASES)
    assert MANIFEST["corpus_sha256"] == hashlib.sha256(FROZEN_BYTES).hexdigest()
    assert MANIFEST["generator_sha256"] == source_digest("activation_v1")
    assert MANIFEST["self_test_sha256"] == source_digest("activation_v1", self_tests=True)
    assert MANIFEST["support_sha256"] == hashlib.sha256(SUPPORT.read_bytes()).hexdigest()
    counts = _object(MANIFEST["counts"])
    assert counts == {
        "case_count": 208,
        "event_count": 2154,
        "max_activations": 12,
        "max_dynamic_depth": 2,
        "max_loop_iterations": 3,
        "max_map_children": 3,
        "max_templates": 3,
        "trace_count": 322,
    }
    assert _object(MANIFEST["generation_provenance"])["generations"] == 2


def test_clean_subprocess_denies_product_and_unrelated_imports() -> None:
    assert_reference_imports("activation_v1")
    assert_isolated_generation("activation_v1")


def test_adapter_exceptions_remain_harness_failures() -> None:
    spec = importlib.util.spec_from_file_location("missing-reference-adapter", HERE / "absent.py")
    assert spec is not None
    with pytest.raises(FileNotFoundError):
        assert spec.loader is not None
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
