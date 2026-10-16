# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nominal context providers and pure views of retained execution results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass
from typing import Literal

from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject, require_instance, require_owner
from anonymizer.engine.graph_sdk.context import ContextProvider, ContextSelector, RetrievalBounds, SourceResult
from anonymizer.engine.graph_sdk.executor import ExecutionResult
from anonymizer.engine.graph_sdk.qualification import QualificationResult
from anonymizer.engine.graph_sdk.records import CandidateRef, CanonicalRecord, EvidenceRef, TargetStatus
from anonymizer.engine.graph_sdk.requests import (
    ArtifactValue,
    PhysicalRequestId,
    RequestAssociation,
    StopResult,
    TextArtifactValue,
)
from anonymizer.graph._values import ArtifactRef, DatumId

ProjectionKind = Literal["protected", "annotation", "research", "execution_only", "custom"]
ContentKind = Literal["annotation", "research", "execution_only", "custom"]
_PROJECTION_KEY = object()


class RemoteContext(ContextProvider, ABC):
    """Subclass the existing context protocol without changing resource ownership."""

    @abstractmethod
    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: RequestAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResult:
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, request: PhysicalRequestId) -> StopResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProjectionSource(PrivateValue):
    _key: InitVar[object]
    execution: ExecutionResult
    qualification: QualificationResult
    record: CanonicalRecord

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _PROJECTION_KEY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProtectedProjectionItem(PrivateValue):
    _key: InitVar[object]
    target: DatumId
    candidate: CandidateRef
    content: str
    evidence: frozenset[EvidenceRef]

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _PROJECTION_KEY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ArtifactProjectionItem(PrivateValue):
    _key: InitVar[object]
    target: DatumId
    artifact: ArtifactRef
    content: ArtifactValue
    status: TargetStatus

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _PROJECTION_KEY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProtectedProjection(PrivateValue):
    _key: InitVar[object]
    record: CanonicalRecord
    items: tuple[ProtectedProjectionItem, ...]

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _PROJECTION_KEY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContentProjection(PrivateValue):
    _key: InitVar[object]
    kind: ContentKind
    record: CanonicalRecord
    items: tuple[ArtifactProjectionItem, ...]

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _PROJECTION_KEY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ArtifactSelection(PrivateValue):
    target: DatumId
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        require_instance(self.target, DatumId)
        require_instance(self.artifact, ArtifactRef)


def bind_projection_source(*, execution: ExecutionResult, qualification: QualificationResult) -> ProjectionSource:
    require_instance(execution, ExecutionResult)
    require_instance(qualification, QualificationResult)
    if (
        qualification._result is not execution
        or qualification._admitted.execution is not execution._execution
        or qualification.record.invocation != execution.record.invocation
    ):
        reject(EffectCode.FOREIGN_OWNER)
    return ProjectionSource(
        _key=_PROJECTION_KEY,
        execution=execution,
        qualification=qualification,
        record=qualification.record,
    )


def _validate_source(source: ProjectionSource) -> None:
    require_instance(source, ProjectionSource)
    if source.record is not source.qualification.record:
        reject(EffectCode.FOREIGN_OWNER)
    bind_projection_source(execution=source.execution, qualification=source.qualification)


def project_protected(*, source: ProjectionSource) -> ProtectedProjection:
    _validate_source(source)
    artifacts = dict(source.execution.artifacts)
    items = []
    for output in source.qualification.qualified:
        value = artifacts.get(output.candidate.artifact)
        if value is None:
            reject(EffectCode.MISSING)
        if type(value) is not TextArtifactValue:
            reject(EffectCode.UNSUPPORTED)
        items.append(
            ProtectedProjectionItem(
                _key=_PROJECTION_KEY,
                target=output.target,
                candidate=output.candidate,
                content=value.text,
                evidence=output.evidence,
            )
        )
    return ProtectedProjection(_key=_PROJECTION_KEY, record=source.record, items=tuple(items))


def project_content(
    *, source: ProjectionSource, kind: ContentKind, artifacts: tuple[ArtifactSelection, ...]
) -> ContentProjection:
    require_instance(source, ProjectionSource)
    require_instance(artifacts, tuple)
    if type(kind) is not str:
        reject(EffectCode.INVALID_TYPE)
    if len(artifacts) > len(source.execution.ports):
        reject(EffectCode.LIMIT_EXCEEDED)
    for item in artifacts:
        require_instance(item, ArtifactSelection)
    _validate_source(source)
    values = dict(source.execution.artifacts)
    statuses = {status.target: status for status in source.record.statuses}
    available = {(port.target, port.artifact) for port in source.execution.ports}
    if any(item.target not in statuses or item.artifact.invocation != source.record.invocation for item in artifacts):
        reject(EffectCode.FOREIGN_OWNER)
    if len(set(artifacts)) != len(artifacts):
        reject(EffectCode.DUPLICATE)
    if any((item.target, item.artifact) not in available or item.artifact not in values for item in artifacts):
        reject(EffectCode.MISSING)
    if kind not in {"annotation", "research", "execution_only", "custom"}:
        reject(EffectCode.UNSUPPORTED)
    items = tuple(
        ArtifactProjectionItem(
            _key=_PROJECTION_KEY,
            target=item.target,
            artifact=item.artifact,
            content=values[item.artifact],
            status=statuses[item.target],
        )
        for item in artifacts
    )
    return ContentProjection(_key=_PROJECTION_KEY, kind=kind, record=source.record, items=items)
