# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mutable state owned by a running invocation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TypeAlias

from anonymizer.engine.graph_sdk._execution_values import (
    ArtifactProvenanceFact,
    DecisionResponse,
    DecisionWait,
    DecisionWaitId,
    ExecutionAssessmentFact,
    ExecutionImplementation,
    ExecutionPortFact,
    LocalAssessmentResult,
    OperationExecutionPolicy,
    ProvenanceKey,
    RuntimeOutcome,
)
from anonymizer.engine.graph_sdk.requests import (
    ArtifactValue,
    AssociationInput,
    AssociationResult,
    ExternalSettlement,
    PhysicalRequestId,
    RequestEvent,
    RequestPolicyBinding,
    RequestState,
    SemanticAssociation,
    advance_requests,
    bind_request_policies,
)
from anonymizer.graph._values import (
    ActivationKey,
    ArtifactRef,
    DatumId,
)
from anonymizer.graph.workflow import (
    NodeId,
)


@dataclass(slots=True)
class _ExecutionControl:
    cancelled: bool = False
    scheduler_changed: asyncio.Event = field(default_factory=asyncio.Event)
    pending: dict[DecisionWaitId, DecisionWait] = field(default_factory=dict)
    responses: dict[DecisionWaitId, DecisionResponse] = field(default_factory=dict)
    closed: set[DecisionWaitId] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _FactCheckpoint:
    values: dict[ArtifactRef, ArtifactValue]
    produced: dict[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef]
    provenance_count: int
    passthrough_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey]
    ports: tuple[ExecutionPortFact, ...]
    assessment_count: int
    next_artifact: int


@dataclass(slots=True)
class _ExecutionFacts:
    values: dict[ArtifactRef, ArtifactValue] = field(default_factory=dict)
    produced: dict[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef] = field(default_factory=dict)
    provenance: list[ArtifactProvenanceFact] = field(default_factory=list)
    ports: list[ExecutionPortFact] = field(default_factory=list)
    assessments: list[ExecutionAssessmentFact] = field(default_factory=list)
    input_parents: list[tuple[DatumId, ActivationKey, str, ProvenanceKey]] = field(default_factory=list)
    passthrough_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey] = field(default_factory=dict)
    next_artifact: int = 0

    def checkpoint(self) -> _FactCheckpoint:
        return _FactCheckpoint(
            values=self.values.copy(),
            produced=self.produced.copy(),
            provenance_count=len(self.provenance),
            passthrough_parents=self.passthrough_parents.copy(),
            ports=tuple(self.ports),
            assessment_count=len(self.assessments),
            next_artifact=self.next_artifact,
        )

    def restore(self, checkpoint: _FactCheckpoint) -> None:
        self.values.clear()
        self.values.update(checkpoint.values)
        self.produced.clear()
        self.produced.update(checkpoint.produced)
        del self.provenance[checkpoint.provenance_count :]
        self.passthrough_parents.clear()
        self.passthrough_parents.update(checkpoint.passthrough_parents)
        self.ports.clear()
        self.ports.extend(checkpoint.ports)
        del self.assessments[checkpoint.assessment_count :]
        self.next_artifact = checkpoint.next_artifact


@dataclass(slots=True)
class _RequestAuthority:
    state: RequestState

    def apply(self, event: RequestEvent) -> None:
        self.state = advance_requests(state=self.state, event=event)

    def bind(self, binding: RequestPolicyBinding) -> None:
        self.state = bind_request_policies(state=self.state, binding=binding)


@dataclass(frozen=True, slots=True)
class _ExecutionJob:
    state_index: int
    target: DatumId
    activation: ActivationKey
    node: NodeId
    policy: OperationExecutionPolicy
    implementation: ExecutionImplementation
    inputs: tuple[AssociationInput, ...]
    input_parents: dict[str, ProvenanceKey]
    association: SemanticAssociation


def _group_external_jobs(jobs: list[_ExecutionJob]) -> list[list[_ExecutionJob]]:
    groups: list[list[_ExecutionJob]] = []
    for job in jobs:
        shared = all(item.capability.attribution == "keyed_shared_request" for item in job.policy.implementations)
        group = next(
            (group for group in groups if shared and group[0].node == job.node and group[0].policy == job.policy),
            None,
        )
        if group is None:
            groups.append([job])
        else:
            group.append(job)
    return groups


@dataclass(frozen=True, slots=True)
class _DeferredAcceptance:
    request: PhysicalRequestId
    results: tuple[AssociationResult, ...]
    settlement: ExternalSettlement | None


_ExecutionTask: TypeAlias = asyncio.Task[
    tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
]
