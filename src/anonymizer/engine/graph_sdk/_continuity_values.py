# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed stable values and private receipts for continuity."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    OpaqueIdentity,
    PrivateValue,
    reject,
    require_count,
    require_instance,
    require_text,
)
from anonymizer.engine.graph_sdk.requests import ContinuityArtifactValue, TextArtifactValue, TextCollectionValue
from anonymizer.graph._values import DatumId
from anonymizer.graph.workflow import CoverageAtom, NodeId

if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk.continuity import ContinuityState
    from anonymizer.engine.graph_sdk.data import ValidatedDataGraph
    from anonymizer.engine.graph_sdk.executor import ExecutionResult
    from anonymizer.engine.graph_sdk.preparation import PreparedPlan
    from anonymizer.engine.graph_sdk.qualification import QualificationResult

ContinuityLifetime = Literal["per_call", "across_calls"]
DecisionChange = Literal["create", "reuse", "supersede"]
CommitStatus = Literal["no_change", "committed", "partial", "conflicted"]
SNAPSHOT_FORMAT = "nemo-anonymizer-continuity"
SNAPSHOT_VERSION = 2
_CONTINUITY_KEY = object()


def _initialize(value: object, key: object, values: dict[str, object]) -> None:
    if key is not _CONTINUITY_KEY:
        raise TypeError("continuity values require their factory")
    for name, item in values.items():
        object.__setattr__(value, name, item)


class StateIdentity(OpaqueIdentity):
    __slots__ = ()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StateNamespace(PrivateValue):
    name: str

    def __post_init__(self) -> None:
        require_text(self.name)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StateGroupRef(PrivateValue):
    name: str

    def __post_init__(self) -> None:
        require_text(self.name)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StateSubjectRef(PrivateValue):
    kind: str
    name: str
    version: int

    def __post_init__(self) -> None:
        require_text(self.kind)
        require_text(self.name)
        require_count(self.version, positive=True)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StateSupportRef(PrivateValue):
    kind: str
    name: str
    version: int

    def __post_init__(self) -> None:
        require_text(self.kind)
        require_text(self.name)
        require_count(self.version, positive=True)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StateDecisionKey(PrivateValue):
    group: StateGroupRef
    subject: StateSubjectRef
    slot: str

    def __post_init__(self) -> None:
        require_instance(self.group, StateGroupRef)
        require_instance(self.subject, StateSubjectRef)
        require_text(self.slot)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StateDecisionRef(PrivateValue):
    key: StateDecisionKey
    revision: int

    def __post_init__(self) -> None:
        require_instance(self.key, StateDecisionKey)
        require_count(self.revision, positive=True)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RetainedDecision(PrivateValue):
    reference: StateDecisionRef
    value: ContinuityArtifactValue
    supports: frozenset[StateSupportRef]

    def __post_init__(self) -> None:
        require_instance(self.reference, StateDecisionRef)
        require_instance(self.value, (TextArtifactValue, TextCollectionValue))
        require_instance(self.supports, frozenset)
        for support in self.supports:
            require_instance(support, StateSupportRef)
        if not self.supports:
            reject(EffectCode.MISSING)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TargetStateGroup(PrivateValue):
    target: DatumId
    group: StateGroupRef

    def __post_init__(self) -> None:
        require_instance(self.target, DatumId)
        require_instance(self.group, StateGroupRef)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContinuityScope(PrivateValue):
    lifetime: ContinuityLifetime
    state: ContinuityState | None
    groups: tuple[TargetStateGroup, ...]

    def __post_init__(self) -> None:
        from anonymizer.engine.graph_sdk.continuity import ContinuityState

        require_text(self.lifetime)
        require_instance(self.groups, tuple)
        for group in self.groups:
            require_instance(group, TargetStateGroup)
        if self.state is not None:
            require_instance(self.state, ContinuityState)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContinuitySupportDecl(PrivateValue):
    reference: StateSupportRef
    node: NodeId
    outcome: str
    promise: str
    required_coverage: frozenset[CoverageAtom]

    def __post_init__(self) -> None:
        require_instance(self.reference, StateSupportRef)
        require_instance(self.node, NodeId)
        require_text(self.outcome)
        require_text(self.promise)
        require_instance(self.required_coverage, frozenset)
        for atom in self.required_coverage:
            require_instance(atom, CoverageAtom)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContinuityDecisionDecl(PrivateValue):
    target: DatumId
    node: NodeId
    outcome: str
    port: str
    key: StateDecisionKey
    change: DecisionChange
    prior: StateDecisionRef | None
    supports: tuple[ContinuitySupportDecl, ...]

    def __post_init__(self) -> None:
        require_instance(self.target, DatumId)
        require_instance(self.node, NodeId)
        require_text(self.outcome)
        require_text(self.port)
        require_instance(self.key, StateDecisionKey)
        require_text(self.change)
        if self.prior is not None:
            require_instance(self.prior, StateDecisionRef)
        require_instance(self.supports, tuple)
        for support in self.supports:
            require_instance(support, ContinuitySupportDecl)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContinuityLimits(PrivateValue):
    max_groups: int
    max_targets: int
    max_decisions: int
    max_decisions_per_group: int
    max_supports_per_decision: int
    max_total_supports: int
    max_name_bytes: int
    max_slot_bytes: int
    max_value_depth: int
    max_value_atoms: int
    max_value_bytes: int
    max_commit_steps: int
    max_snapshot_bytes: int
    max_snapshot_depth: int

    def __post_init__(self) -> None:
        positive = {
            "max_supports_per_decision",
            "max_name_bytes",
            "max_slot_bytes",
            "max_value_depth",
            "max_commit_steps",
            "max_snapshot_depth",
        }
        for item in fields(self):
            require_count(getattr(self, item.name), positive=item.name in positive)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class ContinuityView(PrivateValue):
    identity: StateIdentity
    namespace: StateNamespace
    revision: int
    decisions: tuple[RetainedDecision, ...]

    def __init__(self, *, _key: object, **values: object) -> None:
        _initialize(self, _key, values)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class ContinuitySession(PrivateValue):
    _state: ContinuityState
    data: ValidatedDataGraph
    lifetime: ContinuityLifetime
    base: ContinuityView
    groups: tuple[TargetStateGroup, ...]
    limits: ContinuityLimits

    def __init__(self, *, _key: object, **values: object) -> None:
        _initialize(self, _key, values)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class AdmittedContinuity(PrivateValue):
    prepared: PreparedPlan
    session: ContinuitySession
    declarations: tuple[ContinuityDecisionDecl, ...]
    limits: ContinuityLimits

    def __init__(self, *, _key: object, **values: object) -> None:
        _initialize(self, _key, values)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DecisionConflict(PrivateValue):
    key: StateDecisionKey
    expected: StateDecisionRef | None
    current: StateDecisionRef | None
    proposed_change: DecisionChange

    def __post_init__(self) -> None:
        require_instance(self.key, StateDecisionKey)
        require_text(self.proposed_change)
        for reference in (self.expected, self.current):
            if reference is not None:
                require_instance(reference, StateDecisionRef)
                if reference.key != self.key:
                    reject(EffectCode.CONTRADICTORY)
        if self.proposed_change not in {"create", "reuse", "supersede"}:
            reject(EffectCode.UNSUPPORTED)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class ContinuityCommit(PrivateValue):
    status: CommitStatus
    execution: ExecutionResult
    qualification: QualificationResult
    identity: StateIdentity
    namespace: StateNamespace
    base_revision: int
    resulting_revision: int
    retained: tuple[StateDecisionRef, ...]
    conflicts: tuple[DecisionConflict, ...]
    releasable: tuple[DatumId, ...]
    withheld: tuple[DatumId, ...]

    def __init__(self, *, _key: object, **values: object) -> None:
        _initialize(self, _key, values)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class ContinuitySnapshot(PrivateValue):
    format: Literal["nemo-anonymizer-continuity"]
    version: Literal[2]
    namespace: StateNamespace
    revision: int
    decisions: tuple[RetainedDecision, ...]

    def __init__(self, *, _key: object, **values: object) -> None:
        _initialize(self, _key, values)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class ContinuityHandoff(PrivateValue):
    source: StateIdentity
    namespace: StateNamespace
    base_revision: int
    decisions: tuple[RetainedDecision, ...]

    def __init__(self, *, _key: object, **values: object) -> None:
        _initialize(self, _key, values)


def _key_order(key: StateDecisionKey) -> tuple[str, str, str, int, str]:
    return key.group.name, key.subject.kind, key.subject.name, key.subject.version, key.slot


def _support_order(support: StateSupportRef) -> tuple[str, str, int]:
    return support.kind, support.name, support.version
