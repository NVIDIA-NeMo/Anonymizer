# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mutable state owned by a running invocation."""

from __future__ import annotations

import asyncio
from collections import ChainMap
from collections.abc import MutableMapping
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
    ImplementationHandle,
    LocalAssessmentResult,
    OperationExecutionPolicy,
    ProvenanceKey,
    RuntimeOutcome,
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


@dataclass(frozen=True, slots=True)
class _FactCheckpoint:
    values: MutableMapping[ArtifactRef, ArtifactValue]
    produced: MutableMapping[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef]
    passthrough_parents: MutableMapping[tuple[DatumId, ActivationKey, str], ProvenanceKey]
    provenance: int
    ports: int
    assessments: int
    next_artifact: int
    replaced_ports: dict[int, ExecutionPortFact] = field(default_factory=dict)


@dataclass(slots=True)
class _ExecutionFacts:
    values: MutableMapping[ArtifactRef, ArtifactValue] = field(default_factory=dict)
    produced: MutableMapping[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef] = field(default_factory=dict)
    provenance: list[ArtifactProvenanceFact] = field(default_factory=list)
    ports: list[ExecutionPortFact] = field(default_factory=list)
    assessments: list[ExecutionAssessmentFact] = field(default_factory=list)
    input_parents: list[tuple[DatumId, ActivationKey, str, ProvenanceKey]] = field(default_factory=list)
    passthrough_parents: MutableMapping[tuple[DatumId, ActivationKey, str], ProvenanceKey] = field(default_factory=dict)
    next_artifact: int = 0

    _checkpoint: _FactCheckpoint | None = field(default=None, init=False, repr=False)

    def checkpoint(self) -> _FactCheckpoint:
        """Open a synchronous publication delta, without copying accepted facts."""
        if self._checkpoint is not None:
            raise RuntimeError("publication delta is already open")
        checkpoint = _FactCheckpoint(
            values=self.values,
            produced=self.produced,
            passthrough_parents=self.passthrough_parents,
            provenance=len(self.provenance),
            ports=len(self.ports),
            assessments=len(self.assessments),
            next_artifact=self.next_artifact,
        )
        self.values = ChainMap({}, self.values)
        self.produced = ChainMap({}, self.produced)
        self.passthrough_parents = ChainMap({}, self.passthrough_parents)
        self._checkpoint = checkpoint
        return checkpoint

    def replace_port(self, index: int, fact: ExecutionPortFact) -> None:
        checkpoint = self._checkpoint
        if checkpoint is not None and index < checkpoint.ports:
            checkpoint.replaced_ports.setdefault(index, self.ports[index])
        self.ports[index] = fact

    def commit(self, checkpoint: _FactCheckpoint) -> None:
        if self._checkpoint is None:
            return  # already discarded
        if self._checkpoint is not checkpoint:
            raise RuntimeError("publication checkpoint does not own the open delta")
        if not (
            isinstance(self.values, ChainMap)
            and isinstance(self.produced, ChainMap)
            and isinstance(self.passthrough_parents, ChainMap)
        ):
            raise RuntimeError("publication mapping has no open delta")
        checkpoint.values.update(self.values.maps[0])
        checkpoint.produced.update(self.produced.maps[0])
        checkpoint.passthrough_parents.update(self.passthrough_parents.maps[0])
        self._close(checkpoint)

    def restore(self, checkpoint: _FactCheckpoint) -> None:
        if self._checkpoint is None:
            return
        if self._checkpoint is not checkpoint:
            raise RuntimeError("publication checkpoint does not own the open delta")
        del self.provenance[checkpoint.provenance :]
        del self.ports[checkpoint.ports :]
        del self.assessments[checkpoint.assessments :]
        for index, fact in checkpoint.replaced_ports.items():
            self.ports[index] = fact
        self.next_artifact = checkpoint.next_artifact
        self._close(checkpoint)

    def _close(self, checkpoint: _FactCheckpoint) -> None:
        self.values = checkpoint.values
        self.produced = checkpoint.produced
        self.passthrough_parents = checkpoint.passthrough_parents
        self._checkpoint = None


@dataclass(frozen=True, slots=True)
class _ExecutionJob:
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


_OperationResult: TypeAlias = tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
_SubmissionResults: TypeAlias = dict[SemanticAssociation, _OperationResult]
_ExecutionTask: TypeAlias = asyncio.Task[_OperationResult] | asyncio.Task[_SubmissionResults]
