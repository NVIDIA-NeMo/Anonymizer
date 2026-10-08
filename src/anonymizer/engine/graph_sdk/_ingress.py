# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect only closed SDK callback values, with one aggregate response budget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from anonymizer.engine.graph_sdk._execution_values import AssessmentFinding, LocalAssessmentResult
from anonymizer.engine.graph_sdk._ingress_values import IngressLimits
from anonymizer.engine.graph_sdk._request_values import (
    _FAILURES,
    AssociationResult,
    BindingAssociation,
    BindingDeclarationId,
    BindingId,
    ExactUsage,
    ExternalSettlement,
    FailureClass,
    ObserveSettlement,
    PhysicalRequestId,
    PortArtifact,
    RejectSettlement,
    RequestAssociation,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
    UnknownUsage,
)
from anonymizer.engine.graph_sdk.context import ContextSourceRef, SourceItem, SourceResponse
from anonymizer.graph._values import ArtifactRef, TaskAttemptId
from anonymizer.graph.workflow import ArtifactType


class _ExhaustedIngress(Exception):
    pass


def inspect_settlement(value: object, request: PhysicalRequestId) -> ObserveSettlement | RejectSettlement | None:
    """Keep usage and stop certainty independent of payload acceptance."""
    if value is None:
        return None
    if (
        type(value) is not ExternalSettlement
        or type(value.request) is not PhysicalRequestId
        or value.request != request
    ):
        return RejectSettlement(request=request)
    if type(value.disposition) is not str or value.disposition not in {"completed", "rejected", "stopped", "unknown"}:
        return RejectSettlement(request=request)
    if value.remote_stopped is not None and type(value.remote_stopped) is not bool:
        return RejectSettlement(request=request)
    if (value.disposition == "unknown") != (value.remote_stopped is not True):
        return RejectSettlement(request=request)
    if not inspect_usage(value.usage):
        return RejectSettlement(request=request)
    return ObserveSettlement(settlement=value)


def inspect_usage(value: object) -> bool:
    """Accept only closed, nonnegative usage observations."""
    return (
        type(value) is UnknownUsage
        or type(value) is ExactUsage
        and (
            type(value.input_units) is int
            and value.input_units >= 0
            and type(value.output_units) is int
            and value.output_units >= 0
        )
    )


def inspect_failure(value: object) -> FailureClass:
    """Normalize malformed foreign failure tags without executing value hooks."""
    return cast(FailureClass, value) if type(value) is str and value in _FAILURES else "malformed_response"


@dataclass(slots=True)
class _IngressBudget:
    limits: IngressLimits
    collection_items: int = 0
    assessments: int = 0
    metadata_items: int = 0
    text_bytes: int = 0
    metadata_bytes: int = 0

    def debit(self, name: str, count: int) -> None:
        used = getattr(self, name)
        if count > getattr(self.limits, "max_" + name) - used:
            raise _ExhaustedIngress
        setattr(self, name, used + count)

    def depth(self, depth: int) -> None:
        if depth > self.limits.max_depth:
            raise _ExhaustedIngress

    def text(self, value: object, depth: int, *, payload: bool = False) -> bool:
        self.depth(depth)
        if type(value) is not str:
            return False
        name = "text_bytes" if payload else "metadata_bytes"
        if not payload:
            self.debit("metadata_items", 1)
        if len(value) > getattr(self.limits, "max_" + name) - getattr(self, name):
            raise _ExhaustedIngress
        valid = True
        for offset in range(0, len(value), 1024):
            chunk = value[offset : offset + 1024]
            try:
                encoded = chunk.encode("utf-8", "strict")
            except UnicodeEncodeError:
                # Charge rejected code points too; keep scanning known siblings.
                encoded = chunk.encode("utf-8", "surrogatepass")
                valid = False
            self.debit(name, len(encoded))
        return valid

    def artifact(self, value: object, depth: int) -> bool:
        self.depth(depth)
        if type(value) is TextArtifactValue:
            return self.text(value.text, depth + 1, payload=True)
        if type(value) is not TextCollectionValue:
            return False
        self.depth(depth + 1)
        if type(value.items) is not tuple:
            return False
        self.debit("collection_items", len(value.items))
        prior: tuple[int, int] | None = None
        valid = True
        for item in value.items:
            self.depth(depth + 2)
            if type(item) is not TextCollectionItem:
                valid = False
                continue
            self.depth(depth + 3)  # key, version and scalar value
            if type(item.key) is int and item.key >= 0 and type(item.version) is int and item.version > 0:
                identity = (item.key, item.version)
                valid = (prior is None or identity > prior) and valid
                prior = identity
            else:
                valid = False
            self.debit("metadata_items", 2)
            scalar = type(item.value) is TextArtifactValue
            # Even a malformed known nested value must not hide aggregate bytes.
            valid = self.artifact(item.value, depth + 3) and scalar and valid
        return valid

    def row(self, row: AssociationResult, depth: int) -> bool:
        self.depth(depth)
        self.debit("metadata_items", 1)
        valid = self.text(row.outcome, depth + 1)
        self.depth(depth + 1)
        if type(row.consumed_context_ports) is not frozenset:
            valid = False
        else:
            if len(row.consumed_context_ports) > self.limits.max_metadata_items - self.metadata_items:
                raise _ExhaustedIngress
            for port in row.consumed_context_ports:
                valid = self.text(port, depth + 2) and valid
        if type(row.outputs) is not tuple or len(row.outputs) > self.limits.max_outputs_per_association:
            return False
        for output in row.outputs:
            self.depth(depth + 2)
            if type(output) is not PortArtifact:
                valid = False
                continue
            valid = self.text(output.port, depth + 3) and valid
            self.depth(depth + 3)
            if type(output.artifact_type) is ArtifactType:
                valid = self.text(output.artifact_type.name, depth + 4) and valid
                valid = (type(output.artifact_type.revision) is int and output.artifact_type.revision > 0) and valid
            else:
                valid = False
            if output.artifact is not None:
                valid = type(output.artifact) is ArtifactRef and valid
                self.debit("metadata_items", 1)
            valid = self.artifact(output.value, depth + 3) and valid
        return valid


def _trusted_association(value: object) -> bool:
    if type(value) is SemanticAssociation:
        return type(value.task) is TaskAttemptId
    if type(value) is BindingAssociation:
        return (
            type(value.declaration) is BindingDeclarationId
            and type(value.declaration.binding) is BindingId
            and type(value.declaration.ordinal) is int
            and value.declaration.ordinal >= 0
        )
    return False


def inspect_associations(
    rows: object,
    expected: frozenset[RequestAssociation],
    limits: IngressLimits,
    *,
    assessments: object = (),
) -> frozenset[RequestAssociation] | None:
    """Return invalid correlated members; None means an uncorrelatable envelope.

    Aggregate exhaustion rejects the envelope before any publication. A malformed
    member after trustworthy correlation remains local to its association.
    """
    if type(rows) is not tuple or len(rows) > limits.max_associations:
        return None
    if any(type(row) is not AssociationResult or not _trusted_association(row.association) for row in rows):
        return None
    typed_rows = cast(tuple[AssociationResult, ...], rows)
    keys = tuple(row.association for row in typed_rows)
    if len(keys) != len(expected) or frozenset(keys) != expected:
        return None
    budget = _IngressBudget(limits)
    invalid: set[RequestAssociation] = set()
    try:
        budget.depth(2)  # response root, then its results tuple
        for row in typed_rows:
            if not budget.row(row, 3):
                invalid.add(row.association)
        if type(assessments) is not tuple:
            return expected
        budget.debit("assessments", len(assessments))
        for assessment in assessments:
            budget.depth(3)
            if type(assessment) is not LocalAssessmentResult or not _trusted_association(assessment.association):
                return expected
            if assessment.association not in expected:
                return expected
            budget.debit("metadata_items", 1)
            valid = budget.text(assessment.promise, 4)
            valid = budget.text(assessment.evidence_port, 4) and valid
            budget.depth(4)
            if type(assessment.finding) is AssessmentFinding:
                valid = budget.text(assessment.finding.status, 5) and valid
                valid = budget.text(assessment.finding.code, 5) and valid
            else:
                valid = False
            if not valid:
                invalid.add(assessment.association)
    except _ExhaustedIngress:
        return expected
    return frozenset(invalid)


def inspect_source(response: SourceResponse, association: RequestAssociation, limits: IngressLimits) -> bool:
    """Validate a bounded closed retrieval payload before materialization."""
    if (
        type(response) is not SourceResponse
        or type(response.source) is not ContextSourceRef
        or type(response.items) is not tuple
    ):
        return False
    budget = _IngressBudget(limits)
    try:
        budget.depth(2)
        valid = budget.text(response.source.name, 3)
        if type(response.source.revision) is not int or response.source.revision <= 0:
            return False
        budget.debit("collection_items", len(response.items))
        for item in response.items:
            budget.depth(3)
            if type(item) is not SourceItem or not _trusted_association(item.association):
                return False
            if item.association != association:
                return False
            if type(item.key) is not int or item.key < 0 or type(item.version) is not int or item.version <= 0:
                return False
            budget.debit("metadata_items", 3)
            valid = budget.text(item.text, 4, payload=True) and valid
    except _ExhaustedIngress:
        return False
    return valid


def inspect_source_identity(value: object, expected: ContextSourceRef) -> bool:
    """Compare source identity only after excluding foreign equality hooks."""
    return (
        type(value) is ContextSourceRef
        and type(value.name) is str
        and type(value.revision) is int
        and value.revision > 0
        and value == expected
    )
