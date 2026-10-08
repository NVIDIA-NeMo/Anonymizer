# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Product contract tests for immutable graph identities and canonical records."""

from __future__ import annotations

import copy
import json
import pickle
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk.records import (
    AbsenceRef,
    CandidateRef,
    CanonicalRecord,
    Completion,
    DecisionRef,
    EvidenceRef,
    ExpectedMembership,
    Qualification,
    ReasonCode,
    TargetStatus,
    TerminalCategory,
    TerminalFact,
)
from anonymizer.graph._values import (
    ActivationKey,
    ArtifactRef,
    ContractViolation,
    DatumId,
    GraphId,
    InvocationId,
    PlanId,
    TaskAttemptId,
    ValidationCode,
)
from tests.graph_sdk.reference import data_v1


class _RecordAdapter:
    """Map neutral reference labels to fresh nominal product identities."""

    def __init__(self, facts: dict[str, data_v1.JsonValue]) -> None:
        self._facts = facts
        self._plans: dict[str, PlanId] = {}
        self._invocations: dict[str, InvocationId] = {}
        self._graph = GraphId.new()
        self._foreign_graph = GraphId.new()
        self._datums: dict[tuple[str, int], DatumId] = {}

    def construct(self, boundary: str) -> object:
        dispatch = {
            "activation": lambda: self._activation(self._facts),
            "artifact": lambda: self._artifact(self._facts),
            "absence": lambda: self._absence(self._facts),
            "evidence": lambda: self._evidence(self._facts),
            "membership": lambda: self._membership(self._facts),
            "terminal": lambda: self._terminal(self._facts),
            "status": lambda: self._status(self._facts),
            "canonical": self._canonical,
        }
        return dispatch[boundary]()

    def _plan(self, label: str) -> PlanId:
        return self._plans.setdefault(label, PlanId.new())

    def _invocation(self, label: data_v1.JsonValue, plan_label: str = "P") -> InvocationId:
        if not isinstance(label, str):
            return cast(InvocationId, label)
        if label not in self._invocations:
            self._invocations[label] = InvocationId.new(plan=self._plan(plan_label))
        return self._invocations[label]

    def _datum(self, value: data_v1.JsonValue) -> DatumId:
        if not isinstance(value, list):
            return cast(DatumId, value)
        label = (cast(str, value[0]), cast(int, value[1]))
        if label not in self._datums:
            graph = self._graph if label[0] == "local" else self._foreign_graph
            self._datums[label] = DatumId.new(graph=graph)
        return self._datums[label]

    def _activation(self, value: dict[str, data_v1.JsonValue]) -> ActivationKey:
        parent_value = value["parent"]
        return ActivationKey(
            invocation=self._invocation(value["invocation"]),
            occurrence=cast(int, value["occurrence"]),
            parent=None if parent_value is None else self._activation(_object(parent_value)),
            iteration=None if value["iteration"] is None else cast(int, value["iteration"]),
        )

    def _artifact(self, value: dict[str, data_v1.JsonValue]) -> ArtifactRef:
        return ArtifactRef(
            invocation=self._invocation(value["invocation"]),
            key=cast(int, value["key"]),
            version=cast(int, value["version"]),
        )

    def _absence(self, value: dict[str, data_v1.JsonValue]) -> AbsenceRef:
        return AbsenceRef(
            invocation=self._invocation(value["invocation"]),
            query=cast(int, value["query"]),
            scope_revision=cast(int, value["scope_revision"]),
        )

    def _consumed(self, value: data_v1.JsonValue) -> ArtifactRef | CandidateRef | DecisionRef | AbsenceRef:
        raw = _object(value)
        kind = cast(str, raw["kind"])
        reference = _object(raw["ref"])
        if kind == "absence":
            return self._absence(reference)
        artifact = self._artifact(reference)
        if kind == "artifact":
            return artifact
        if kind == "candidate":
            return CandidateRef(artifact=artifact, target=self._datum(raw["target"]))
        if kind == "decision":
            return DecisionRef(artifact=artifact)
        raise AssertionError("reference corpus admitted an unknown consumed reference")

    def _evidence(self, value: dict[str, data_v1.JsonValue]) -> EvidenceRef:
        return EvidenceRef(
            artifact=self._artifact(_object(value["artifact"])),
            consumed=frozenset(self._consumed(item) for item in _list(value["consumed"])),
        )

    def _membership(self, value: dict[str, data_v1.JsonValue]) -> ExpectedMembership:
        parent_value = value["parent"]
        return ExpectedMembership(
            invocation=self._invocation(value["invocation"]),
            parent=None if parent_value is None else self._activation(_object(parent_value)),
            members=frozenset(self._activation(_object(item)) for item in _list(value["members"])),
            closed=cast(bool, value["closed"]),
        )

    def _terminal(self, value: dict[str, data_v1.JsonValue]) -> TerminalFact:
        activation = self._activation(_object(value["activation"]))
        attempt_value = value["attempt"]
        attempt = None
        if attempt_value is not None:
            attempt_activation = self._activation(_object(_object(attempt_value)["activation"]))
            attempt = TaskAttemptId.new(activation=attempt_activation)
        reasons_value = value["reasons"]
        reasons = (
            frozenset(cast(ReasonCode, item) for item in reasons_value)
            if isinstance(reasons_value, list)
            else cast(frozenset[ReasonCode], reasons_value)
        )
        return TerminalFact(
            activation=activation,
            attempt=attempt,
            category=cast(TerminalCategory, value["category"]),
            reasons=reasons,
        )

    def _status(self, value: dict[str, data_v1.JsonValue]) -> TargetStatus:
        return TargetStatus(
            target=self._datum(value["target"]),
            completion=cast(Completion, value["completion"]),
            qualification=cast(Qualification, value["qualification"]),
            artifact_available=cast(bool, value["artifact_available"]),
            protection_available=cast(bool, value["protection_available"]),
        )

    def _canonical(self) -> CanonicalRecord:
        plan_label = cast(str, self._facts["plan"])
        invocation_label = cast(str, self._facts["invocation"])
        invocation_plan = cast(str, self._facts["invocation_plan"])
        plan = self._plan(plan_label)
        invocation = self._invocation(invocation_label, invocation_plan)
        return CanonicalRecord(
            plan=plan,
            invocation=invocation,
            graph=self._graph,
            targets=frozenset(self._datum(item) for item in _list(self._facts["targets"])),
            memberships=tuple(self._membership(_object(item)) for item in _list(self._facts["memberships"])),
            terminals=tuple(self._terminal(_object(item)) for item in _list(self._facts["terminals"])),
            artifacts=frozenset(self._artifact(_object(item)) for item in _list(self._facts["artifacts"])),
            evidence=tuple(self._evidence(_object(item)) for item in _list(self._facts["evidence"])),
            statuses=tuple(self._status(_object(item)) for item in _list(self._facts["statuses"])),
        )


def _object(value: data_v1.JsonValue) -> dict[str, data_v1.JsonValue]:
    return cast(dict[str, data_v1.JsonValue], value)


def _list(value: data_v1.JsonValue) -> list[data_v1.JsonValue]:
    return cast(list[data_v1.JsonValue], value)


def _adapt_record_case(case: data_v1.FixtureCase) -> tuple[str, str | None]:
    declaration = case["declaration"]
    assert declaration["kind"] == "record"
    adapter = _RecordAdapter(declaration["facts"])
    try:
        adapter.construct(declaration["boundary"])
    except ContractViolation as error:
        return "reject", error.code.value
    return "accept", None


def test_all_reference_record_cases_match_product_validation() -> None:
    corpus_path = Path(__file__).parent / "reference/data_v1_cases.json"
    cases = data_v1._parse_cases(json.loads(corpus_path.read_bytes()))
    record_cases = [case for case in cases if case["declaration"]["kind"] == "record"]
    assert len(record_cases) == 74
    for case in record_cases:
        actual_verdict, actual_code = _adapt_record_case(case)
        expected = case["expected"]
        assert actual_verdict == expected["verdict"], case["case_id"]
        if expected["verdict"] == "reject":
            assert actual_code == expected["code"], case["case_id"]


def test_opaque_identity_is_fresh_immutable_copy_stable_and_nonserializable() -> None:
    graph = GraphId.new()
    plan = PlanId.new()
    invocation = InvocationId.new(plan=plan)
    activation = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    identities = (
        graph,
        DatumId.new(graph=graph),
        plan,
        invocation,
        TaskAttemptId.new(activation=activation),
    )
    assert GraphId.new() != graph
    assert DatumId.new(graph=graph) != DatumId.new(graph=graph)
    for identity in identities:
        assert copy.copy(identity) is identity
        assert copy.deepcopy(identity) is identity
        assert type(identity).__name__ in repr(identity)
        with pytest.raises(TypeError, match="^graph identity serialization is not supported$"):
            pickle.dumps(identity)


def test_structural_values_are_hashable_and_deep_activation_chains_are_iterative() -> None:
    invocation = InvocationId.new(plan=PlanId.new())
    left = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    right = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    for occurrence in range(1, 2500):
        left = ActivationKey(invocation=invocation, occurrence=occurrence, parent=left, iteration=occurrence)
        right = ActivationKey(invocation=invocation, occurrence=occurrence, parent=right, iteration=occurrence)
    assert left == right
    assert hash(left) == hash(right)
    assert copy.copy(left) is left
    assert copy.deepcopy(left) is left
    assert "ActivationKey" in repr(left)


def test_nested_fields_are_immutable_and_reject_mutable_collections() -> None:
    invocation = InvocationId.new(plan=PlanId.new())
    activation = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    membership = ExpectedMembership(invocation=invocation, parent=None, members=frozenset({activation}), closed=False)
    with pytest.raises(FrozenInstanceError):
        setattr(membership, "closed", True)
    with pytest.raises(ContractViolation) as rejected:
        ExpectedMembership(invocation=invocation, parent=None, members=cast(Any, [activation]), closed=False)
    assert rejected.value.code is ValidationCode.INVALID_TYPE


def test_replace_revalidates_values_and_cross_owner_integrity() -> None:
    plan = PlanId.new()
    invocation = InvocationId.new(plan=plan)
    foreign_invocation = InvocationId.new(plan=plan)
    activation = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    artifact = ArtifactRef(invocation=invocation, key=0, version=1)
    status = TargetStatus(
        target=DatumId.new(graph=GraphId.new()),
        completion="closed",
        qualification="met",
        artifact_available=True,
        protection_available=True,
    )
    with pytest.raises(ContractViolation) as invalid_artifact:
        replace(artifact, version=0)
    assert invalid_artifact.value.code is ValidationCode.INVALID_VALUE
    with pytest.raises(ContractViolation) as foreign_parent:
        replace(
            activation, parent=ActivationKey(invocation=foreign_invocation, occurrence=1, parent=None, iteration=None)
        )
    assert foreign_parent.value.code is ValidationCode.FOREIGN_OWNER
    with pytest.raises(ContractViolation) as contradictory_status:
        replace(status, completion="pending")
    assert contradictory_status.value.code is ValidationCode.CONTRADICTORY


def test_canonical_record_replace_revalidates_snapshot_integrity() -> None:
    plan = PlanId.new()
    invocation = InvocationId.new(plan=plan)
    graph = GraphId.new()
    target = DatumId.new(graph=graph)
    activation = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    record = CanonicalRecord(
        plan=plan,
        invocation=invocation,
        graph=graph,
        targets=frozenset({target}),
        memberships=(
            ExpectedMembership(
                invocation=invocation,
                parent=None,
                members=frozenset({activation}),
                closed=False,
            ),
        ),
        terminals=(),
        artifacts=frozenset(),
        evidence=(),
        statuses=(
            TargetStatus(
                target=target,
                completion="pending",
                qualification="unknown",
                artifact_available=False,
                protection_available=False,
            ),
        ),
    )
    assert replace(record) == record
    with pytest.raises(ContractViolation) as missing_status:
        replace(record, statuses=())
    assert missing_status.value.code is ValidationCode.MISSING
    with pytest.raises(ContractViolation) as foreign_plan:
        replace(record, invocation=InvocationId.new(plan=PlanId.new()))
    assert foreign_plan.value.code is ValidationCode.FOREIGN_OWNER


def test_errors_and_representations_do_not_expose_identity_or_payload_values() -> None:
    invocation = InvocationId.new(plan=PlanId.new())
    artifact = ArtifactRef(invocation=invocation, key=987654321, version=3)
    assert "ArtifactRef" in repr(artifact)
    assert "987654321" not in repr(artifact)
    with pytest.raises(ContractViolation) as captured:
        replace(artifact, key=-987654321)
    rendered = f"{captured.value!s} {captured.value!r}"
    assert "987654321" not in rendered
    assert isinstance(captured.value, ContractViolation)
    assert captured.value.code is ValidationCode.INVALID_VALUE


@pytest.mark.parametrize(
    ("completion", "qualification", "artifact_available", "protection_available"),
    [
        ("closed", "not_assessed", True, False),
        ("closed", "unmet", True, False),
        ("pending", "unknown", False, False),
        ("closed", "met", True, False),
        ("closed", "met", True, True),
    ],
)
def test_status_axes_remain_independent(
    completion: Completion,
    qualification: Qualification,
    artifact_available: bool,
    protection_available: bool,
) -> None:
    status = TargetStatus(
        target=DatumId.new(graph=GraphId.new()),
        completion=completion,
        qualification=qualification,
        artifact_available=artifact_available,
        protection_available=protection_available,
    )
    assert (status.completion, status.qualification) == (completion, qualification)
    assert status.artifact_available is artifact_available
    assert status.protection_available is protection_available


def test_terminal_category_and_target_status_are_separate_axes() -> None:
    invocation = InvocationId.new(plan=PlanId.new())
    activation = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    terminal = TerminalFact(
        activation=activation,
        attempt=TaskAttemptId.new(activation=activation),
        category="success",
        reasons=frozenset(),
    )
    status = TargetStatus(
        target=DatumId.new(graph=GraphId.new()),
        completion="closed",
        qualification="not_assessed",
        artifact_available=True,
        protection_available=False,
    )
    assert terminal.category == "success"
    assert status.qualification == "not_assessed"
    assert not status.protection_available


@pytest.mark.parametrize(
    ("category", "reasons"),
    [
        ("success", frozenset()),
        ("failure", frozenset({"execution_failed"})),
        ("cancelled", frozenset({"cancel_requested"})),
        ("lost", frozenset({"transport_lost"})),
    ],
)
def test_structural_terminals_do_not_invent_operation_attempts(
    category: TerminalCategory, reasons: frozenset[ReasonCode]
) -> None:
    activation = ActivationKey(
        invocation=InvocationId.new(plan=PlanId.new()), occurrence=0, parent=None, iteration=None
    )
    terminal = TerminalFact(activation=activation, attempt=None, category=category, reasons=reasons, structural=True)
    assert terminal.structural and terminal.attempt is None
    with pytest.raises(ContractViolation) as error:
        replace(terminal, structural=False)
    assert error.value.code is ValidationCode.CONTRADICTORY
    with pytest.raises(ContractViolation) as error:
        replace(terminal, attempt=TaskAttemptId.new(activation=activation))
    assert error.value.code is ValidationCode.CONTRADICTORY
    with pytest.raises(ContractViolation) as error:
        replace(terminal, structural=cast(bool, 1))
    assert error.value.code is ValidationCode.INVALID_TYPE
