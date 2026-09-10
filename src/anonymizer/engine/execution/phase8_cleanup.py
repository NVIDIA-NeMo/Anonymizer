# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity-bound cleanup evidence for the private Phase 8 runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anonymizer.engine.execution.phase10_inspection import (
        _Phase10InspectionRejected,
        _Phase10OwnerCapture,
    )


class _Phase8CleanupPhase(str, Enum):
    PRE_REDUCTION = "pre_reduction"
    POST_REDUCTION = "post_reduction"


class _Phase8CleanupComponent(str, Enum):
    OPERATION = "operation"
    BACKEND = "backend"
    RUNTIME = "runtime"


class _Phase8CleanupStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8CleanupProof:
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


_CLEANUP_SEAL = object()


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8CleanupReceipt:
    """Closed, content-free cleanup receipt issued by one runtime owner."""

    phase: _Phase8CleanupPhase
    component: _Phase8CleanupComponent
    status: _Phase8CleanupStatus
    identity: object = field(compare=False)
    active_operation_count: int = 0
    active_workframe_reference_count: int = 0
    token_reference_count: int = 0
    source_projection_reference_count: int = 0
    baseline_reference_count: int = 0
    obligation_reference_count: int = 0
    provisional_revision_reference_count: int = 0
    evaluation_evidence_reference_count: int = 0
    retained_candidate_cell_count: int = 0
    withheld_candidate_reference_count: int = 0
    _proof: _Phase8CleanupProof | None = field(default=None, compare=False)

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 8 cleanup receipts are not serializable")

    def _phase10_snapshot(self) -> _Phase10OwnerCapture | _Phase10InspectionRejected:
        """Issue a detached bounded view of one sealed cleanup terminal."""
        from anonymizer.engine.execution.phase10_inspection import (
            _phase10_count_bucket,
            _Phase10CaptureBoundary,
            _Phase10CleanupState,
            _Phase10CountBucket,
            _Phase10Diagnostic,
            _Phase10InspectionRejected,
            _Phase10LifecycleState,
            _Phase10OwnerCapture,
            _Phase10ReasonCategory,
            _Phase10ReconciliationState,
            _Phase10RejectionCode,
            _Phase10ReleaseState,
            _Phase10SemanticProfile,
            _Phase10Snapshot,
            _Phase10Stage,
            _Phase10StageSummary,
            _Phase10SubjectKind,
            _Phase10TerminalState,
            _Phase10TerminalSummary,
        )

        if not _is_phase8_cleanup_receipt(
            self,
            identity=self.identity,
            phase=self.phase,
            component=self.component,
        ):
            return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)
        count = sum(
            (
                self.active_operation_count,
                self.active_workframe_reference_count,
                self.token_reference_count,
                self.source_projection_reference_count,
                self.baseline_reference_count,
                self.obligation_reference_count,
                self.provisional_revision_reference_count,
                self.evaluation_evidence_reference_count,
                self.retained_candidate_cell_count,
                self.withheld_candidate_reference_count,
            )
        )
        bucket = _phase10_count_bucket(count)
        if bucket is None:
            return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)
        boundary = (
            _Phase10CaptureBoundary.PRE_REDUCTION_CLEANUP_TERMINAL
            if self.phase is _Phase8CleanupPhase.PRE_REDUCTION
            else _Phase10CaptureBoundary.POST_REDUCTION_CLEANUP_TERMINAL
        )
        cleanup_state = {
            _Phase8CleanupStatus.VERIFIED: _Phase10CleanupState.VERIFIED,
            _Phase8CleanupStatus.FAILED: _Phase10CleanupState.FAILED,
            _Phase8CleanupStatus.UNCONFIRMED: _Phase10CleanupState.UNCONFIRMED,
        }[self.status]
        terminal_state = {
            _Phase8CleanupStatus.VERIFIED: _Phase10TerminalState.SUCCEEDED,
            _Phase8CleanupStatus.FAILED: _Phase10TerminalState.FAILED,
            _Phase8CleanupStatus.UNCONFIRMED: _Phase10TerminalState.INCONSISTENT,
        }[self.status]
        snapshot = _Phase10Snapshot(
            (
                _Phase10StageSummary(
                    _Phase10Stage.CLEANUP,
                    _Phase10LifecycleState.CLEANUP_TERMINAL,
                    _Phase10CountBucket.ONE,
                ),
            ),
            (_Phase10TerminalSummary(_Phase10Stage.CLEANUP, terminal_state, bucket),),
            _Phase10ReconciliationState.RECONCILED,
            cleanup_state,
            _Phase10ReleaseState.NOT_ENTERED,
        )
        category = {
            _Phase8CleanupStatus.VERIFIED: None,
            _Phase8CleanupStatus.FAILED: _Phase10ReasonCategory.CLEANUP_FAILED,
            _Phase8CleanupStatus.UNCONFIRMED: _Phase10ReasonCategory.CLEANUP_UNCONFIRMED,
        }[self.status]
        diagnostics = (
            ()
            if category is None
            else (
                _Phase10Diagnostic(
                    boundary,
                    _Phase10Stage.CLEANUP,
                    terminal_state,
                    category,
                    bucket,
                    _Phase10ReconciliationState.RECONCILED,
                    cleanup_state,
                ),
            )
        )
        return _Phase10OwnerCapture(
            _Phase10SubjectKind.CLEANUP_RECEIPT,
            _Phase10SemanticProfile.GROUPED_REWRITE_V1,
            boundary,
            _Phase10LifecycleState.CLEANUP_TERMINAL,
            snapshot,
            diagnostics,
        )


def _issue_phase8_cleanup_receipt(
    phase: _Phase8CleanupPhase,
    component: _Phase8CleanupComponent,
    status: _Phase8CleanupStatus,
    identity: object,
    *,
    active_operation_count: int = 0,
    active_workframe_reference_count: int = 0,
    token_reference_count: int = 0,
    source_projection_reference_count: int = 0,
    baseline_reference_count: int = 0,
    obligation_reference_count: int = 0,
    provisional_revision_reference_count: int = 0,
    evaluation_evidence_reference_count: int = 0,
    retained_candidate_cell_count: int = 0,
    withheld_candidate_reference_count: int = 0,
) -> _Phase8CleanupReceipt:
    """Seal one receipt after the owning component has measured its state."""
    values = (
        phase,
        component,
        status,
        identity,
        active_operation_count,
        active_workframe_reference_count,
        token_reference_count,
        source_projection_reference_count,
        baseline_reference_count,
        obligation_reference_count,
        provisional_revision_reference_count,
        evaluation_evidence_reference_count,
        retained_candidate_cell_count,
        withheld_candidate_reference_count,
    )
    candidate = _Phase8CleanupReceipt(*values)
    snapshot = _cleanup_snapshot(candidate)
    if snapshot is None:
        raise TypeError("private Phase 8 cleanup receipt is malformed")
    return _Phase8CleanupReceipt(*values, _Phase8CleanupProof(_CLEANUP_SEAL, snapshot))


def _is_phase8_cleanup_receipt(
    value: object,
    *,
    identity: object,
    phase: _Phase8CleanupPhase,
    component: _Phase8CleanupComponent,
) -> bool:
    if not isinstance(value, _Phase8CleanupReceipt) or value._proof is None:
        return False
    return (
        value.identity is identity
        and value.phase is phase
        and value.component is component
        and value._proof.seal is _CLEANUP_SEAL
        and value._proof.snapshot == _cleanup_snapshot(value)
    )


def _cleanup_snapshot(value: _Phase8CleanupReceipt) -> tuple[object, ...] | None:
    counts = (
        value.active_operation_count,
        value.active_workframe_reference_count,
        value.token_reference_count,
        value.source_projection_reference_count,
        value.baseline_reference_count,
        value.obligation_reference_count,
        value.provisional_revision_reference_count,
        value.evaluation_evidence_reference_count,
        value.retained_candidate_cell_count,
        value.withheld_candidate_reference_count,
    )
    if (
        not isinstance(value.phase, _Phase8CleanupPhase)
        or not isinstance(value.component, _Phase8CleanupComponent)
        or not isinstance(value.status, _Phase8CleanupStatus)
        or value.identity is None
        or any(type(count) is not int or count < 0 for count in counts)
    ):
        return None
    return (value.phase, value.component, value.status, id(value.identity), *counts)
