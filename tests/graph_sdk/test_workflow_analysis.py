# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiler work envelopes, complete tiny products and hand-derived mutants."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import ParamSpec, cast

import pytest

from anonymizer.engine.graph_sdk import _workflow_composition as composition
from anonymizer.engine.graph_sdk import _workflow_values as values
from anonymizer.engine.graph_sdk._workflow_analysis import _ANALYSIS, _AnalysisMeter
from anonymizer.engine.graph_sdk._workflow_composition import validate_dynamic_input_summaries
from anonymizer.graph import ContractViolation, NodeId, SubgraphNode, ValidationCode, WorkflowId
from tests.graph_sdk.reference.workflow_fixture_builders import (
    _context,
    _evidence,
    _input,
    _limits,
    _node,
    _outcome_binding,
    _output,
    _q,
    _ref,
    _workflow,
    _wrap,
)
from tests.graph_sdk.test_workflow_static import Json, Object, _Adapter, _execute, _object, _without_state_effects
from tests.graph_sdk.workflow_construction import AdmittedWorkflow, admit_static_workflow

_P = ParamSpec("_P")


def _chain(count: int) -> Object:
    operation = _q()
    interface = _q(factor=count)
    for specification in (operation, interface):
        for outcome in cast(list[Object], specification["outcomes"]):
            outcome["category"] = "success"
    declaration = _workflow(
        interface,
        [_node(f"N{index}", operation) for index in range(count)],
        limits=_limits(nodes=count, bindings=count + 3, edges=count - 1, states=1),
    )
    declaration["input_bindings"] = [_input("i0", "N0", "i0")] + [
        _input(f"N{index - 1}", f"N{index}", "i0") for index in range(1, count)
    ]
    declaration["output_bindings"] = [_output(f"N{count - 1}", "o0", "o0")]
    declaration["sequence"] = [
        {"before": _ref(f"N{index - 1}"), "after": _ref(f"N{index}")} for index in range(1, count)
    ]
    declaration["outcome_bindings"] = [_outcome_binding(f"N{count - 1}", name, name) for name in ("ok", "fail")]
    return declaration


def _readmit(workflow: AdmittedWorkflow, limit: int = 1_000_000) -> AdmittedWorkflow:
    return admit_static_workflow(
        workflow=workflow.workflow,
        interface=workflow.interface,
        nodes=tuple(workflow.operations),
        input_bindings=tuple(workflow.input_bindings),
        output_bindings=tuple(workflow.output_bindings),
        outcome_bindings=tuple(workflow.outcome_bindings),
        sequence=tuple(workflow.sequence),
        choices=tuple(workflow.choices),
        protection=tuple(workflow.protection_requirements),
        limits=replace(workflow.limits, max_analysis_work=limit),
    )


def _meters(monkeypatch: pytest.MonkeyPatch) -> list[_AnalysisMeter]:
    observed: list[_AnalysisMeter] = []
    tick = _AnalysisMeter.tick

    def record(meter: _AnalysisMeter) -> None:
        if not observed or observed[-1] is not meter:
            observed.append(meter)
        tick(meter)

    monkeypatch.setattr(_AnalysisMeter, "tick", record)
    return observed


@pytest.mark.parametrize("count", (1, 2, 3, 4))
def test_tiny_product_normalization_matches_frozen_independent_answers(count: int) -> None:
    declaration = _chain(count)
    frozen = (Path(__file__).parent / "reference" / "frozen" / "workflow_analysis_known_answers.json").read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == "a4b8fc5a5b1bce0fcd6d9c470cb54d737ab5fd41b280ac342ee7d7dc70184fdd"
    fixture = next(item for item in json.loads(frozen) if item["count"] == count)
    assert fixture["declaration"] == declaration
    expected = fixture["expected"]
    actual = _execute(declaration, None)
    assert expected["status"] == "accepted"
    assert actual[:3] == (expected["status"], expected["code"], _without_state_effects(expected["normalized"]))


@pytest.mark.parametrize("field", ("ports", "cost", "context", "evidence", "model"))
def test_near_neighbor_effect_is_not_erased(field: str) -> None:
    declaration = _chain(3)
    first = _object(cast(list[Object], declaration["nodes"])[0]["operation"])
    alternate = cast(list[Object], first["outcomes"])[1]
    if field == "ports":
        alternate["produced_ports"] = []
    elif field == "cost":
        _object(alternate["ceiling"])["max_activations"] = 2
    elif field == "context":
        alternate["context"] = [_context("i0", "private-context")]
    elif field == "evidence":
        alternate["evidence"] = [_evidence("assessment", "label", "o0", ("i0",), ("field",))]
    else:
        alternate["model_requirements"] = [{"capability": "model", "revision": 1}]
    # Every node is mandatory. A missing port blocks its successor; any added
    # summary fact or cost is absent from the declared interface. These expected
    # codes are hand-derived; the historical static reference does not derive
    # aggregate outcome summaries for arbitrary newly constructed products.
    expected_code = "missing" if field == "ports" else "contradictory"
    actual = _execute(declaration, None)
    assert actual[:2] == ("rejected", expected_code)


def test_exact_measured_budget_and_one_less_have_no_identity_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = _Adapter().admit(_chain(3))
    observed = _meters(monkeypatch)
    effects: list[str] = []

    def forbidden_identity() -> None:
        effects.append("identity")
        pytest.fail("admission allocated an identity")

    monkeypatch.setattr(WorkflowId, "new", forbidden_identity)
    monkeypatch.setattr(NodeId, "new", forbidden_identity)
    _readmit(workflow)
    count = observed[-1].work
    assert count > 0
    assert _readmit(workflow, count).interface == workflow.interface
    assert observed[-1].work == count
    with pytest.raises(ContractViolation) as failure:
        _readmit(workflow, count - 1)
    assert failure.value.code is ValidationCode.ANALYSIS_LIMIT_EXCEEDED
    assert observed[-1].work == count - 1
    assert _ANALYSIS.get() is None
    assert effects == []
    # Rejection cannot poison a later independent compilation.
    _readmit(workflow, count)
    assert observed[-1].work == count


def test_work_is_invariant_under_fresh_opaque_ids_and_set_order(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _meters(monkeypatch)
    counts = []
    for _ in range(12):
        workflow = _Adapter().admit(copy.deepcopy(_chain(4)))
        _readmit(workflow)
        counts.append(observed[-1].work)
    assert len(set(counts)) == 1


def test_nested_and_replacement_analyses_share_the_outer_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    declaration = _wrap(_chain(2))
    declaration["limits"] = _limits(nodes=4, depth=3, bindings=8, states=1)
    # This graph declares influence without identity, so replacing identity on
    # both the root and nested input leaves its retained summary valid.
    body = _object(cast(list[Object], declaration["nodes"])[0]["body"])
    for scope in (declaration, body):
        operations = [_object(scope["interface"])] + [
            _object(node["operation"]) for node in cast(list[Object], scope["nodes"])
        ]
        for operation in operations:
            for dependency in cast(list[Object], operation["output_dependencies"]):
                dependency["identity_input"] = None
    workflow = _Adapter().admit(declaration)
    child = next(node.body for node in workflow.operations if isinstance(node, SubgraphNode))
    replacements = (*workflow.input_bindings, *child.input_bindings)
    observed = _meters(monkeypatch)
    # An existing top-level compilation owns the budget even across rechecks.
    meter = _AnalysisMeter(1_000_000)
    token = _ANALYSIS.set(meter)
    try:
        _readmit(workflow)
        first = meter.work
        validate_dynamic_input_summaries(workflow, replacements)
        assert meter.work > first
        assert all(item is meter for item in observed)
    finally:
        _ANALYSIS.reset(token)
    assert _ANALYSIS.get() is None


@pytest.mark.parametrize("value", (True, 0, -1, 1.0))
def test_analysis_limit_requires_a_positive_exact_integer(value: object) -> None:
    workflow = _Adapter().admit(_chain(1))
    with pytest.raises(ContractViolation):
        replace(workflow.limits, max_analysis_work=cast(int, value))


def test_equivalent_outcomes_fit_without_redefining_choice_states(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2**12 complete outcomes, but only two externally distinguishable paths.
    observed = _meters(monkeypatch)
    workflow = _Adapter().admit(_chain(12))
    assert workflow.limits.max_choice_states == 1
    assert observed[-1].work < workflow.limits.max_analysis_work


@pytest.mark.parametrize("count", (1, 2, 3, 4))
def test_quotient_and_complete_product_have_the_same_hand_derived_interface(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration = _chain(count)
    optimized = _Adapter().admit(declaration)
    visited = []

    def observe(function: Callable[_P, None]) -> Callable[_P, None]:
        def record(*args: _P.args, **kwargs: _P.kwargs) -> None:
            visited.append(1)
            function(*args, **kwargs)

        return record

    monkeypatch.setattr(composition, "_validate_composition_path", observe(composition._validate_composition_path))
    monkeypatch.setattr(
        composition,
        "_outcome_options",
        lambda nodes, mappings, choices, interface: [node.operation.outcomes for node in nodes],
    )
    complete = _Adapter().admit(declaration)
    assert len(visited) == 2**count
    assert optimized.interface == complete.interface
    for outcome in complete.interface.outcomes:
        assert outcome.produced_ports == {"o0"}
        assert outcome.ceiling.max_activations == count
        assert outcome.ceiling.max_input_bytes == 8 * count
        assert outcome.context == outcome.evidence == outcome.model_requirements == frozenset()
    dependency = complete.interface.output_dependencies[0]
    assert dependency.inputs == {"i0"}
    assert dependency.identity_input == "i0"


def test_rejected_evidence_projection_charges_scalar_scans(monkeypatch: pytest.MonkeyPatch) -> None:
    declaration = _chain(3)
    operation = _object(cast(list[Object], declaration["nodes"])[0]["operation"])
    alternate = cast(list[Object], operation["outcomes"])[1]
    alternate["evidence"] = [_evidence("assessment", "label", "o0", ("i0",), ("field",))]
    scans: list[tuple[int, int, int, bool]] = []
    validate = values._validate_scalars

    def observe(
        *,
        strings: tuple[object, ...] = (),
        nonnegative_integers: tuple[object, ...] = (),
        positive_integers: tuple[object, ...] = (),
    ) -> None:
        meter = _ANALYSIS.get()
        before = 0 if meter is None else meter.work
        completed = False
        try:
            validate(strings=strings, nonnegative_integers=nonnegative_integers, positive_integers=positive_integers)
            completed = True
        finally:
            if meter is not None:
                scans.append((len(strings), before, meter.work - before, completed))

    monkeypatch.setattr(values, "_validate_scalars", observe)
    adapter = _Adapter()
    with pytest.raises(ContractViolation) as original:
        adapter.admit(declaration)
    assert original.value.code is ValidationCode.CONTRADICTORY
    evidence_scans = [scan for scan in scans if scan[0] == 4]
    assert len(evidence_scans) == 1
    _, before, charged, completed = evidence_scans[0]
    assert charged == 8 and completed  # Four type checks, then four nonempty checks.

    original_limits = adapter.limits

    def bounded_limits(value: Json) -> values.WorkflowLimits:
        return replace(original_limits(value), max_analysis_work=before + charged - 1)

    monkeypatch.setattr(adapter, "limits", bounded_limits)
    scans.clear()
    with pytest.raises(ContractViolation) as exhausted:
        adapter.admit(declaration)
    assert exhausted.value.code is ValidationCode.ANALYSIS_LIMIT_EXCEEDED
    assert [scan for scan in scans if scan[0] == 4] == [(4, before, 7, False)]
    assert _ANALYSIS.get() is None


@pytest.mark.parametrize("limit", (25, 26))
def test_scalar_collection_construction_and_repeated_scans_share_the_budget(limit: int) -> None:
    # Three strings are inspected twice; five integers are copied, type-checked,
    # filtered into typed collections and range-checked: 3*2 + 5*4 = 26 visits.
    meter = _AnalysisMeter(limit)
    token = _ANALYSIS.set(meter)
    try:
        if limit == 26:
            values._validate_scalars(strings=("a", "b", "c"), nonnegative_integers=(0, 1), positive_integers=(1, 2, 3))
        else:
            with pytest.raises(ContractViolation) as exhausted:
                values._validate_scalars(
                    strings=("a", "b", "c"), nonnegative_integers=(0, 1), positive_integers=(1, 2, 3)
                )
            assert exhausted.value.code is ValidationCode.ANALYSIS_LIMIT_EXCEEDED
        assert meter.work == limit
    finally:
        _ANALYSIS.reset(token)


def test_retained_workflow_comparison_uses_active_worklist_budget() -> None:
    left = _Adapter().admit(_chain(2))
    right = _readmit(left)
    assert left == right
    meter = _AnalysisMeter(1)
    token = _ANALYSIS.set(meter)
    try:
        with pytest.raises(ContractViolation) as exhausted:
            _ = left == right
        assert exhausted.value.code is ValidationCode.ANALYSIS_LIMIT_EXCEEDED
        assert meter.work == 1
    finally:
        _ANALYSIS.reset(token)


def test_evidence_bearing_path_exact_work_and_one_less(monkeypatch: pytest.MonkeyPatch) -> None:
    declaration = _chain(3)
    operation = _object(cast(list[Object], declaration["nodes"])[0]["operation"])
    for specification in (operation, _object(declaration["interface"])):
        for outcome in cast(list[Object], specification["outcomes"]):
            outcome["evidence"] = [_evidence("assessment", "label", "o0", ("i0",), ("field",))]
    workflow = _Adapter().admit(declaration)
    observed = _meters(monkeypatch)
    _readmit(workflow)
    required = observed[-1].work
    assert _readmit(workflow, required).interface == workflow.interface
    with pytest.raises(ContractViolation) as exhausted:
        _readmit(workflow, required - 1)
    assert exhausted.value.code is ValidationCode.ANALYSIS_LIMIT_EXCEEDED
    assert observed[-1].work == required - 1
    assert _ANALYSIS.get() is None


def test_dynamic_endpoint_lifting_charges_variable_path_copy() -> None:
    workflow = WorkflowId.new()
    container = NodeId.new(workflow=workflow)
    endpoint = values.MapItemPort(
        path=(container,) * 3,
        expander=NodeId.new(workflow=workflow),
        member=NodeId.new(workflow=workflow),
        item_input="item",
        membership_port="members",
        expansion_outcome="expanded",
    )
    # Three copied path entries, four validated path entries and two scans of
    # the three endpoint strings all use the same compiler meter.
    meter = _AnalysisMeter(13)
    token = _ANALYSIS.set(meter)
    try:
        assert endpoint.lifted(container).path == (container,) * 4
        assert meter.work == 13
    finally:
        _ANALYSIS.reset(token)


@pytest.mark.parametrize(
    ("start", "backward", "kind", "expected", "work"),
    (
        ("a", False, "wo", {"one", "two"}, 21),
        ("a", False, "wi", {"a"}, 20),
        ("two", True, "wi", {"a", "b"}, 25),
        ("two", True, "ci", {"context"}, 24),
        ("one", False, "wo", {"one", "two"}, 13),
        ("a", True, "wi", {"a"}, 10),
        ("other", True, "wi", {"isolated"}, 12),
        ("isolated", False, "wo", set(), 11),
        ("absent", False, "wo", set(), 9),
        ("absent", True, "wi", {"absent"}, 10),
    ),
)
def test_endpoint_walk_literal_direction_cycles_and_exact_work(
    start: str, backward: bool, kind: str, expected: set[str], work: int
) -> None:
    owner = WorkflowId.new()
    first, second = NodeId.new(workflow=owner), NodeId.new(workflow=owner)
    vertices: dict[str, composition.Vertex] = {
        "a": ("wi", None, "a"),
        "b": ("wi", None, "b"),
        "context": ("ci", None, "context"),
        "input": ("ni", first, "same"),
        "output": ("no", first, "same"),
        "other": ("no", second, "same"),
        "one": ("wo", None, "one"),
        "two": ("wo", None, "two"),
        "isolated": ("wi", None, "isolated"),
        "absent": ("wi", None, "absent"),
    }
    edges = {
        (vertices[source], vertices[destination])
        for source, destination in (
            ("a", "input"),
            ("b", "input"),
            ("context", "input"),
            ("input", "output"),
            ("output", "input"),
            ("output", "one"),
            ("one", "two"),
            ("isolated", "other"),
        )
    }
    # Eight index entries, each reachable edge, one pop per enqueued vertex
    # (including the cycle revisit), then each distinct returned endpoint.
    for limit in (work, work - 1):
        meter = _AnalysisMeter(limit)
        token = _ANALYSIS.set(meter)
        try:
            if limit == work:
                assert composition._walk_endpoints(
                    vertices[start], edges, backward=backward, endpoint_kind=kind
                ) == frozenset(expected)
            else:
                with pytest.raises(ContractViolation) as exhausted:
                    composition._walk_endpoints(vertices[start], edges, backward=backward, endpoint_kind=kind)
                assert exhausted.value.code is ValidationCode.ANALYSIS_LIMIT_EXCEEDED
            assert meter.work == limit
        finally:
            _ANALYSIS.reset(token)


@pytest.mark.parametrize(
    ("port", "code"),
    (
        ([], ValidationCode.INVALID_TYPE),
        ({}, ValidationCode.INVALID_TYPE),
        (False, ValidationCode.INVALID_TYPE),
        ("", ValidationCode.INVALID_VALUE),
        ("missing", ValidationCode.MISSING),
    ),
)
def test_input_destination_port_reentry_has_typed_rejection(port: object, code: ValidationCode) -> None:
    workflow = _Adapter().admit(_chain(1))
    destination = next(iter(workflow.input_bindings)).destination
    if code is not ValidationCode.MISSING:
        with pytest.raises(ContractViolation) as constructor:
            values.NodeInputRef(node=destination.node, port=cast(str, port))
        assert constructor.value.code is code
    original = destination.port
    object.__setattr__(destination, "port", port)
    try:
        with pytest.raises(ContractViolation) as rejected:
            _readmit(workflow)
        assert rejected.value.code is code
    finally:
        object.__setattr__(destination, "port", original)
    assert _readmit(workflow).interface == workflow.interface


@pytest.mark.parametrize("port", ([], {}))
def test_input_destination_reentry_preserves_analysis_budget_precedence(
    port: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = _Adapter().admit(_chain(1))
    destination = next(iter(workflow.input_bindings)).destination
    original = destination.port
    observed = _meters(monkeypatch)
    object.__setattr__(destination, "port", port)
    try:
        with pytest.raises(ContractViolation) as exhausted:
            _readmit(workflow, limit=1)
        assert exhausted.value.code is ValidationCode.ANALYSIS_LIMIT_EXCEEDED
        assert observed[-1].work == 1
        assert _ANALYSIS.get() is None
        with pytest.raises(ContractViolation) as malformed:
            _readmit(workflow)
        assert malformed.value.code is ValidationCode.INVALID_TYPE
    finally:
        object.__setattr__(destination, "port", original)
