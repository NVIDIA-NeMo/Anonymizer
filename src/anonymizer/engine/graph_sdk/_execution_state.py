# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mutable state owned by a running invocation."""

from __future__ import annotations

import asyncio
from collections import ChainMap
from collections.abc import MutableMapping, MutableSequence
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar, cast, overload

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


_T = TypeVar("_T")


class _FactList(MutableSequence[_T], Generic[_T]):
    """Stage append and replacement operations without copying prior facts."""

    def __init__(self, base: MutableSequence[_T]) -> None:
        self.base = base
        self.added: list[_T] = []
        self.replaced: dict[int, _T] = {}

    def __len__(self) -> int:
        return len(self.base) + len(self.added)

    @overload
    def __getitem__(self, index: int) -> _T: ...

    @overload
    def __getitem__(self, index: slice) -> list[_T]: ...

    def __getitem__(self, index: int | slice) -> _T | list[_T]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index >= len(self.base):
            return self.added[index - len(self.base)]
        return self.replaced[index] if index in self.replaced else self.base[index]

    def __setitem__(self, index: int | slice, value: _T | object) -> None:
        if not isinstance(index, int) or not 0 <= index < len(self):
            raise TypeError("fact staging supports indexed replacement only")
        item = cast(_T, value)
        if index >= len(self.base):
            self.added[index - len(self.base)] = item
        else:
            self.replaced[index] = item

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError("fact staging cannot delete facts")

    def insert(self, index: int, value: _T) -> None:
        if index != len(self):
            raise TypeError("fact staging can only append")
        self.added.append(value)

    def commit(self) -> None:
        for index, value in self.replaced.items():
            self.base[index] = value
        self.base.extend(self.added)


@dataclass(frozen=True, slots=True)
class _FactCheckpoint:
    values: MutableMapping[ArtifactRef, ArtifactValue]
    produced: MutableMapping[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef]
    provenance: MutableSequence[ArtifactProvenanceFact]
    passthrough_parents: MutableMapping[tuple[DatumId, ActivationKey, str], ProvenanceKey]
    ports: MutableSequence[ExecutionPortFact]
    assessments: MutableSequence[ExecutionAssessmentFact]
    next_artifact: int


@dataclass(slots=True)
class _ExecutionFacts:
    values: MutableMapping[ArtifactRef, ArtifactValue] = field(default_factory=dict)
    produced: MutableMapping[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef] = field(default_factory=dict)
    provenance: MutableSequence[ArtifactProvenanceFact] = field(default_factory=list)
    ports: MutableSequence[ExecutionPortFact] = field(default_factory=list)
    assessments: MutableSequence[ExecutionAssessmentFact] = field(default_factory=list)
    input_parents: list[tuple[DatumId, ActivationKey, str, ProvenanceKey]] = field(default_factory=list)
    passthrough_parents: MutableMapping[tuple[DatumId, ActivationKey, str], ProvenanceKey] = field(default_factory=dict)
    next_artifact: int = 0

    def checkpoint(self) -> _FactCheckpoint:
        """Open one association delta; authoritative containers remain untouched."""
        assert not isinstance(self.values, ChainMap)
        checkpoint = _FactCheckpoint(
            values=self.values,
            produced=self.produced,
            provenance=self.provenance,
            passthrough_parents=self.passthrough_parents,
            ports=self.ports,
            assessments=self.assessments,
            next_artifact=self.next_artifact,
        )
        self.values = ChainMap({}, self.values)
        self.produced = ChainMap({}, self.produced)
        self.passthrough_parents = ChainMap({}, self.passthrough_parents)
        self.provenance = _FactList(self.provenance)
        self.ports = _FactList(self.ports)
        self.assessments = _FactList(self.assessments)
        return checkpoint

    def commit(self, checkpoint: _FactCheckpoint) -> None:
        if self.values is checkpoint.values:
            return  # already discarded
        assert isinstance(self.values, ChainMap) and isinstance(self.produced, ChainMap)
        assert isinstance(self.passthrough_parents, ChainMap)
        assert isinstance(self.provenance, _FactList) and isinstance(self.ports, _FactList)
        assert isinstance(self.assessments, _FactList)
        checkpoint.values.update(self.values.maps[0])
        checkpoint.produced.update(self.produced.maps[0])
        checkpoint.passthrough_parents.update(self.passthrough_parents.maps[0])
        self.provenance.commit()
        self.ports.commit()
        self.assessments.commit()
        allocated = self.next_artifact
        self.restore(checkpoint)
        self.next_artifact = allocated

    def restore(self, checkpoint: _FactCheckpoint) -> None:
        self.values = checkpoint.values
        self.produced = checkpoint.produced
        self.provenance = checkpoint.provenance
        self.passthrough_parents = checkpoint.passthrough_parents
        self.ports = checkpoint.ports
        self.assessments = checkpoint.assessments
        self.next_artifact = checkpoint.next_artifact


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


_ExecutionTask: TypeAlias = asyncio.Task[
    tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
]
