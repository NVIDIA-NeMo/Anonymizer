# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mutable state owned by a running invocation."""

from __future__ import annotations

import asyncio
from collections import ChainMap
from dataclasses import dataclass, field
from typing import TypeAlias

from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_values import (
    _FACT_KEY,
    ArtifactProvenanceFact,
    ArtifactRole,
    BoundInputKey,
    DecisionResponse,
    DecisionWait,
    DecisionWaitId,
    ExecutionAssessmentFact,
    ExecutionImplementation,
    ExecutionPortFact,
    ImplementationHandle,
    InitialCollectionKey,
    LocalAssessmentResult,
    LocalCompleted,
    MapItemKey,
    OperationExecutionPolicy,
    OperationOutputKey,
    ProvenanceKey,
    RuntimeOutcome,
    SubmissionFailure,
)
from anonymizer.engine.graph_sdk.requests import (
    ArtifactValue,
    AssociationInput,
    AssociationResult,
    SemanticAssociation,
)
from anonymizer.graph._values import (
    ActivationKey,
    ArtifactRef,
    DatumId,
)
from anonymizer.graph.workflow import (
    ArtifactType,
    NodeId,
)


@dataclass(slots=True)
class _ExecutionControl:
    cancelled: bool = False
    scheduler_changed: asyncio.Event = field(default_factory=asyncio.Event)
    pending: dict[DecisionWaitId, DecisionWait] = field(default_factory=dict)
    responses: dict[DecisionWaitId, DecisionResponse] = field(default_factory=dict)
    closed: set[DecisionWaitId] = field(default_factory=set)
    measured_handles: dict[SemanticAssociation, ImplementationHandle] = field(default_factory=dict)


@dataclass(slots=True)
class _ExecutionFacts:
    """Canonical invocation facts, with isolated synchronous publication deltas."""

    values: ChainMap[ArtifactRef, ArtifactValue] = field(default_factory=ChainMap)
    provenance: ChainMap[ProvenanceKey, ArtifactProvenanceFact] = field(default_factory=ChainMap)
    ports: ChainMap[tuple[ActivationKey, str], ExecutionPortFact] = field(default_factory=ChainMap)
    assessments: ChainMap[tuple[ActivationKey, str], ExecutionAssessmentFact] = field(default_factory=ChainMap)
    input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey] = field(default_factory=dict)
    passthrough_parents: ChainMap[tuple[DatumId, ActivationKey, str], ProvenanceKey] = field(default_factory=ChainMap)
    next_artifact: int = 0
    _sources: ChainMap[tuple[DatumId, ActivationKey | NodeId, str], ProvenanceKey] = field(default_factory=ChainMap)
    _parent: _ExecutionFacts | None = field(default=None, repr=False)

    def delta(self) -> _ExecutionFacts:
        return _ExecutionFacts(
            values=self.values.new_child(),
            provenance=self.provenance.new_child(),
            ports=self.ports.new_child(),
            assessments=self.assessments.new_child(),
            input_parents=self.input_parents,
            passthrough_parents=self.passthrough_parents.new_child(),
            next_artifact=self.next_artifact,
            _sources=self._sources.new_child(),
            _parent=self,
        )

    def publish(self, delta: _ExecutionFacts) -> None:
        if delta._parent is not self:
            raise RuntimeError("publication delta does not belong to this invocation")
        self.values.update(delta.values.maps[0])
        self.provenance.update(delta.provenance.maps[0])
        self.ports.update(delta.ports.maps[0])
        self.assessments.update(delta.assessments.maps[0])
        self.passthrough_parents.update(delta.passthrough_parents.maps[0])
        self._sources.update(delta._sources.maps[0])
        self.next_artifact = delta.next_artifact
        delta._parent = None

    def artifact(self, target: DatumId, occurrence: ActivationKey | NodeId, port: str) -> ArtifactRef | None:
        source = self.source(target, occurrence, port)
        return None if source is None else source.artifact

    def source(self, target: DatumId, occurrence: ActivationKey | NodeId, port: str) -> ArtifactProvenanceFact | None:
        key = self._sources.get((target, occurrence, port))
        return self.provenance.get(key) if key is not None else None

    def add_output(
        self,
        key: OperationOutputKey,
        node: NodeId,
        artifact: ArtifactRef,
        artifact_type: ArtifactType,
        parents: frozenset[ProvenanceKey],
        role: ArtifactRole,
    ) -> None:
        self.add_provenance(
            ArtifactProvenanceFact(
                _key=_FACT_KEY,
                key=key,
                artifact=artifact,
                parents=parents,
                decision=role == "decision",
            )
        )
        self.ports[key.activation, key.port] = ExecutionPortFact(
            _key=_FACT_KEY,
            activation=key.activation,
            node=node,
            target=key.target,
            port=key.port,
            artifact=artifact,
            artifact_type=artifact_type,
            role=role,
        )

    def add_provenance(self, fact: ArtifactProvenanceFact) -> None:
        key = fact.key
        if key in self.provenance:
            reject(EffectCode.DUPLICATE)
        self.provenance[key] = fact
        if isinstance(key, (BoundInputKey, InitialCollectionKey)):
            prior = self.source(key.target, key.node, key.port)
            if (
                isinstance(key, BoundInputKey)
                and prior is not None
                and (isinstance(prior.key, InitialCollectionKey) or prior.artifact.version >= fact.artifact.version)
            ):
                return
            occurrence = key.node
        elif isinstance(key, MapItemKey):
            occurrence = key.member
        elif isinstance(key, OperationOutputKey):
            occurrence = key.activation
        else:
            return
        self._sources[key.target, occurrence, key.port] = key


@dataclass(frozen=True, slots=True)
class _OperationJob:
    activation: ActivationKey
    node: NodeId
    policy: OperationExecutionPolicy
    implementation: ExecutionImplementation
    inputs: tuple[AssociationInput, ...]
    input_parents: dict[str, ProvenanceKey]
    association: SemanticAssociation


@dataclass(frozen=True, slots=True)
class _ExecutionJob(_OperationJob):
    target: DatumId


@dataclass(frozen=True, slots=True)
class _InvocationJob(_OperationJob):
    """An operation activation owned by the invocation, with no target attribution."""


def _group_external_jobs(jobs: list[_ExecutionJob]) -> list[list[_ExecutionJob]]:
    groups: list[list[_ExecutionJob]] = []
    for job in jobs:
        shared = job.policy.kind == "delegated" or all(
            item.capability.attribution == "keyed_shared_request" for item in job.policy.implementations
        )
        group = next(
            (group for group in groups if shared and group[0].node == job.node and group[0].policy == job.policy),
            None,
        )
        if group is None:
            groups.append([job])
        else:
            group.append(job)
    return groups


_OperationResult: TypeAlias = tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
_SubmissionResults: TypeAlias = dict[SemanticAssociation, _OperationResult]
_ExecutionTask: TypeAlias = (
    asyncio.Task[_OperationResult | LocalCompleted | SubmissionFailure] | asyncio.Task[_SubmissionResults]
)
