# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mutable state owned by a running invocation."""

from __future__ import annotations

import asyncio
from collections import ChainMap
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

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
    InvocationId,
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


_Scope = TypeVar("_Scope", bound=DatumId | InvocationId, covariant=True)


@dataclass(frozen=True, slots=True)
class _OperationJob(Generic[_Scope]):
    """One bound operation; its typed scope distinguishes target and invocation work."""

    scope: _Scope
    policy: OperationExecutionPolicy
    inputs: tuple[AssociationInput, ...]
    input_parents: dict[str, ProvenanceKey]
    association: SemanticAssociation

    @property
    def activation(self) -> ActivationKey:
        return self.association.task.activation

    @property
    def node(self) -> NodeId:
        return self.policy.node

    @property
    def implementation(self) -> ExecutionImplementation:
        return self.policy.implementations[0]


_OperationResult: TypeAlias = tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
_SubmissionResults: TypeAlias = dict[SemanticAssociation, _OperationResult]
_ExecutionTask: TypeAlias = (
    asyncio.Task[_OperationResult | LocalCompleted | SubmissionFailure] | asyncio.Task[_SubmissionResults]
)
