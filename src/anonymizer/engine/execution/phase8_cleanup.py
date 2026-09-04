# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity-bound cleanup evidence for the private Phase 8 runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
