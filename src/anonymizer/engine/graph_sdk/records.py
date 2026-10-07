# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable canonical-record values and their integrity validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

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

TerminalCategory: TypeAlias = Literal["success", "failure", "cancelled", "lost", "blocked", "inconsistent"]
ReasonCode: TypeAlias = Literal[
    "execution_failed",
    "cancel_requested",
    "transport_lost",
    "missing",
    "duplicate",
    "foreign",
    "stale",
    "contradictory",
    "prerequisite",
]
Completion: TypeAlias = Literal["pending", "closed"]
Qualification: TypeAlias = Literal["not_assessed", "met", "unmet", "unknown"]

_TERMINAL_CATEGORIES = frozenset({"success", "failure", "cancelled", "lost", "blocked", "inconsistent"})
_REASON_CODES = frozenset(
    {
        "execution_failed",
        "cancel_requested",
        "transport_lost",
        "missing",
        "duplicate",
        "foreign",
        "stale",
        "contradictory",
        "prerequisite",
    }
)
_COMPLETIONS = frozenset({"pending", "closed"})
_QUALIFICATIONS = frozenset({"not_assessed", "met", "unmet", "unknown"})


class _PrivateRepr:
    __slots__ = ()

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


def _require_frozenset(value: object, element_type: type[object] | tuple[type[object], ...]) -> None:
    if not isinstance(value, frozenset) or any(not isinstance(item, element_type) for item in value):
        raise ContractViolation(ValidationCode.INVALID_TYPE)


def _require_tuple(value: object, element_type: type[object]) -> None:
    if not isinstance(value, tuple) or any(not isinstance(item, element_type) for item in value):
        raise ContractViolation(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CandidateRef(_PrivateRepr):
    artifact: ArtifactRef
    target: DatumId

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactRef) or not isinstance(self.target, DatumId):
            raise ContractViolation(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DecisionRef(_PrivateRepr):
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactRef):
            raise ContractViolation(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AbsenceRef(_PrivateRepr):
    invocation: InvocationId
    query: int
    scope_revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, InvocationId):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if isinstance(self.query, bool) or not isinstance(self.query, int):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if isinstance(self.scope_revision, bool) or not isinstance(self.scope_revision, int):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if self.query < 0 or self.scope_revision <= 0:
            raise ContractViolation(ValidationCode.INVALID_VALUE)


ConsumedRef: TypeAlias = ArtifactRef | CandidateRef | DecisionRef | AbsenceRef


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class EvidenceRef(_PrivateRepr):
    artifact: ArtifactRef
    consumed: frozenset[ConsumedRef]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactRef):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        _require_frozenset(self.consumed, (ArtifactRef, CandidateRef, DecisionRef, AbsenceRef))
        if any(_consumed_invocation(item) != self.artifact.invocation for item in self.consumed):
            raise ContractViolation(ValidationCode.FOREIGN_OWNER)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExpectedMembership(_PrivateRepr):
    invocation: InvocationId
    parent: ActivationKey | None
    members: frozenset[ActivationKey]
    closed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, InvocationId):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if self.parent is not None and not isinstance(self.parent, ActivationKey):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        _require_frozenset(self.members, ActivationKey)
        if not isinstance(self.closed, bool):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if self.parent is not None and self.parent.invocation != self.invocation:
            raise ContractViolation(ValidationCode.FOREIGN_OWNER)
        if any(member.invocation != self.invocation for member in self.members):
            raise ContractViolation(ValidationCode.FOREIGN_OWNER)
        if any(member == self.parent or member.parent != self.parent for member in self.members):
            raise ContractViolation(ValidationCode.CONTRADICTORY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TerminalFact(_PrivateRepr):
    activation: ActivationKey
    attempt: TaskAttemptId | None
    category: TerminalCategory
    reasons: frozenset[ReasonCode]

    def __post_init__(self) -> None:
        if not isinstance(self.activation, ActivationKey):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if self.attempt is not None and not isinstance(self.attempt, TaskAttemptId):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if not isinstance(self.category, str):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if not isinstance(self.reasons, frozenset) or any(not isinstance(reason, str) for reason in self.reasons):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if self.category not in _TERMINAL_CATEGORIES or any(reason not in _REASON_CODES for reason in self.reasons):
            raise ContractViolation(ValidationCode.INVALID_VALUE)
        if self.attempt is not None and self.attempt.activation != self.activation:
            raise ContractViolation(ValidationCode.FOREIGN_OWNER)
        if (self.category == "success" and self.reasons) or (self.category != "success" and not self.reasons):
            raise ContractViolation(ValidationCode.CONTRADICTORY)
        if self.attempt is None and self.category not in ("blocked", "inconsistent"):
            raise ContractViolation(ValidationCode.CONTRADICTORY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TargetStatus(_PrivateRepr):
    target: DatumId
    completion: Completion
    qualification: Qualification
    artifact_available: bool
    protection_available: bool

    def __post_init__(self) -> None:
        if not isinstance(self.target, DatumId):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if not isinstance(self.completion, str) or not isinstance(self.qualification, str):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if not isinstance(self.artifact_available, bool) or not isinstance(self.protection_available, bool):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if self.completion not in _COMPLETIONS or self.qualification not in _QUALIFICATIONS:
            raise ContractViolation(ValidationCode.INVALID_VALUE)
        if self.protection_available and not (self.completion == "closed" and self.qualification == "met"):
            raise ContractViolation(ValidationCode.CONTRADICTORY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CanonicalRecord(_PrivateRepr):
    plan: PlanId
    invocation: InvocationId
    graph: GraphId
    targets: frozenset[DatumId]
    memberships: tuple[ExpectedMembership, ...]
    terminals: tuple[TerminalFact, ...]
    artifacts: frozenset[ArtifactRef]
    evidence: tuple[EvidenceRef, ...]
    statuses: tuple[TargetStatus, ...]

    def __post_init__(self) -> None:
        self._validate_types()
        self._validate_values()
        self._validate_owners()
        self._validate_duplicates()
        self._validate_presence()

    def _validate_types(self) -> None:
        if not isinstance(self.plan, PlanId) or not isinstance(self.invocation, InvocationId):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if not isinstance(self.graph, GraphId):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        _require_frozenset(self.targets, DatumId)
        _require_tuple(self.memberships, ExpectedMembership)
        _require_tuple(self.terminals, TerminalFact)
        _require_frozenset(self.artifacts, ArtifactRef)
        _require_tuple(self.evidence, EvidenceRef)
        _require_tuple(self.statuses, TargetStatus)

    def _validate_values(self) -> None:
        if any(status.target.graph == self.graph and status.target not in self.targets for status in self.statuses):
            raise ContractViolation(ValidationCode.INVALID_VALUE)
        for evidence in self.evidence:
            for consumed in evidence.consumed:
                if isinstance(consumed, CandidateRef):
                    if consumed.target.graph == self.graph and consumed.target not in self.targets:
                        raise ContractViolation(ValidationCode.INVALID_VALUE)

    def _validate_owners(self) -> None:
        if self.invocation.plan != self.plan:
            raise ContractViolation(ValidationCode.FOREIGN_OWNER)
        if any(target.graph != self.graph for target in self.targets):
            raise ContractViolation(ValidationCode.FOREIGN_OWNER)
        if any(status.target.graph != self.graph for status in self.statuses):
            raise ContractViolation(ValidationCode.FOREIGN_OWNER)
        nested_invocations = [artifact.invocation for artifact in self.artifacts]
        nested_invocations.extend(membership.invocation for membership in self.memberships)
        nested_invocations.extend(terminal.activation.invocation for terminal in self.terminals)
        nested_invocations.extend(evidence.artifact.invocation for evidence in self.evidence)
        if any(invocation != self.invocation for invocation in nested_invocations):
            raise ContractViolation(ValidationCode.FOREIGN_OWNER)
        for evidence in self.evidence:
            for consumed in evidence.consumed:
                if _consumed_invocation(consumed) != self.invocation:
                    raise ContractViolation(ValidationCode.FOREIGN_OWNER)
                if isinstance(consumed, CandidateRef) and consumed.target.graph != self.graph:
                    raise ContractViolation(ValidationCode.FOREIGN_OWNER)

    def _validate_duplicates(self) -> None:
        parents = [membership.parent for membership in self.memberships]
        members = [member for membership in self.memberships for member in membership.members]
        terminal_activations = [terminal.activation for terminal in self.terminals]
        evidence_artifacts = [evidence.artifact for evidence in self.evidence]
        status_targets = [status.target for status in self.statuses]
        if len(parents) != len(set(parents)) or len(members) != len(set(members)):
            raise ContractViolation(ValidationCode.DUPLICATE)
        if len(terminal_activations) != len(set(terminal_activations)):
            raise ContractViolation(ValidationCode.DUPLICATE)
        if len(evidence_artifacts) != len(set(evidence_artifacts)) or len(status_targets) != len(set(status_targets)):
            raise ContractViolation(ValidationCode.DUPLICATE)

    def _validate_presence(self) -> None:
        members = {member for membership in self.memberships for member in membership.members}
        if any(membership.parent is not None and membership.parent not in members for membership in self.memberships):
            raise ContractViolation(ValidationCode.MISSING)
        if any(terminal.activation not in members for terminal in self.terminals):
            raise ContractViolation(ValidationCode.MISSING)
        for evidence in self.evidence:
            if evidence.artifact not in self.artifacts:
                raise ContractViolation(ValidationCode.MISSING)
            if any(
                _consumed_artifact(item) not in self.artifacts
                for item in evidence.consumed
                if not isinstance(item, AbsenceRef)
            ):
                raise ContractViolation(ValidationCode.MISSING)
        if any(target not in {status.target for status in self.statuses} for target in self.targets):
            raise ContractViolation(ValidationCode.MISSING)


def _consumed_invocation(reference: ConsumedRef) -> InvocationId:
    if isinstance(reference, AbsenceRef):
        return reference.invocation
    return _consumed_artifact(reference).invocation


def _consumed_artifact(reference: ArtifactRef | CandidateRef | DecisionRef) -> ArtifactRef:
    if isinstance(reference, ArtifactRef):
        return reference
    return reference.artifact
