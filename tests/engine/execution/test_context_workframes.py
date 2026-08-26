# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from hashlib import sha256
from typing import cast

import pytest

import anonymizer.engine.constants as execution_constants
from anonymizer.engine.constants import (
    COL_CONTEXT_BINDING_ID,
    COL_CONTEXT_ORDINAL,
    COL_CONTEXT_OWNER_WORK_ID,
    COL_CONTEXT_TEXT,
    COL_TARGET_WORK_ID,
    COL_TEXT,
)
from anonymizer.engine.execution.accounting_evidence import _AttemptId, _Dispatch, _InvocationId, _RowToken
from anonymizer.engine.execution.accounting_plan import _AccountingLimits
from anonymizer.engine.execution.context_admission import _compile_context_plan, _ContextPlan
from anonymizer.engine.execution.context_contract import (
    _BackendArtifactClass,
    _ContextBackendCapability,
    _ContextExecutionContract,
    _ContextLimits,
    _ContextOrdering,
    _ContextProfile,
    _ContextSchemaVersion,
    _RetentionPosture,
)
from anonymizer.engine.execution.context_workframes import (
    _BackendArtifactId,
    _BackendClosureAttestation,
    _ContextBindingEvidence,
    _ContextBindingId,
    _ContextCleanupStatus,
    _ContextPayloadToken,
    _ContextReconciliationStatus,
    _lower_context_workframes,
    _make_context_binding_evidence,
    _TargetWorkId,
    _WorkframeClosedError,
    _WorkframeStateError,
)
from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _CoherenceScope,
    _ContextScope,
    _DatumId,
    _DatumPurpose,
    _ProtectionGraph,
    _TextDatum,
)


def test_lowering_keeps_target_and_ordered_context_frames_separate() -> None:
    plan = _plan()
    frames = _lower_context_workframes(
        plan,
        tuple(projection.owner_task for projection in plan.projections),
        target_work_ids=("target-work-a", "target-work-b"),
        identity_factory=_identities("binding-a0", "binding-a1", "binding-b0", "artifact-a"),
    )

    task_column = getattr(execution_constants, "COL_TASK_ID")
    attempt_column = getattr(execution_constants, "COL_ATTEMPT_ID")
    assert list(frames.target_frame.columns) == [COL_TARGET_WORK_ID, task_column, attempt_column, COL_TEXT]
    assert list(frames.context_frame.columns) == [
        COL_CONTEXT_BINDING_ID,
        COL_CONTEXT_OWNER_WORK_ID,
        COL_CONTEXT_ORDINAL,
        COL_CONTEXT_TEXT,
    ]
    assert frames.target_frame.to_dict("records") == [
        {
            COL_TARGET_WORK_ID: "target-work-a",
            task_column: plan.projections[0].owner_task,
            attempt_column: None,
            COL_TEXT: "alpha",
        },
        {
            COL_TARGET_WORK_ID: "target-work-b",
            task_column: plan.projections[1].owner_task,
            attempt_column: None,
            COL_TEXT: "beta",
        },
    ]
    assert frames.context_frame[COL_CONTEXT_TEXT].tolist() == ["gamma", "beta", "alpha"]
    assert frames.context_frame[COL_CONTEXT_ORDINAL].tolist() == [0, 1, 0]
    serialized = f"{frames.target_frame.columns.tolist()}{frames.context_frame.columns.tolist()}"
    assert "target-a" not in serialized
    assert "context-c" not in serialized


def test_exact_binding_and_closure_evidence_verifies_then_closes_all_state() -> None:
    frames = _frames()
    _bind_dispatches(frames)
    evidence = _exact_evidence(frames)

    reconciliation = frames.reconcile(evidence)
    cleanup = frames.close(
        (_BackendClosureAttestation(frames.artifact_id, _BackendArtifactClass.CONTEXT_REQUEST, True),)
    )

    assert reconciliation.status is _ContextReconciliationStatus.VERIFIED
    assert cleanup.status is _ContextCleanupStatus.VERIFIED
    assert frames.target_frame.empty
    assert frames.context_frame.empty
    rendered = repr(vars(frames))
    for canary in ("alpha", "beta", "gamma", "binding-a0", "target-work-a", "artifact-a"):
        assert canary not in rendered
        assert sha256(canary.encode()).hexdigest() not in rendered
    with pytest.raises(_WorkframeClosedError):
        frames.expected_bindings()
    with pytest.raises(_WorkframeClosedError):
        _ = frames.artifact_id


def test_dispatch_binding_is_an_absorbing_one_shot_transition() -> None:
    frames = _frames()
    dispatches = _dispatches_for(frames)

    frames.bind_dispatches(dispatches)
    original_attempts = tuple(frames.target_frame[getattr(execution_constants, "COL_ATTEMPT_ID")])

    with pytest.raises(_WorkframeStateError):
        frames.bind_dispatches(dispatches)
    with pytest.raises(_WorkframeStateError):
        frames.bind_dispatches(tuple(reversed(dispatches)))

    assert tuple(frames.target_frame[getattr(execution_constants, "COL_ATTEMPT_ID")]) == original_attempts


def test_missing_binding_is_local_to_its_compiled_owner() -> None:
    frames = _frames()
    _bind_dispatches(frames)

    result = frames.reconcile(_exact_evidence(frames)[1:])

    assert result.status is _ContextReconciliationStatus.LOCAL_INVALID
    assert result.affected_tasks == (frames.tasks[0],)


def test_cross_target_binding_is_a_global_attribution_failure() -> None:
    frames = _frames()
    _bind_dispatches(frames)
    expected = frames.expected_bindings()
    _binding_id, _owner_work_id, _ordinal = expected[0]
    exact = _exact_evidence(frames)

    result = frames.reconcile(
        (
            replace(exact[0], owner_target_work_id=_TargetWorkId("target-work-b")),
            *exact[1:],
        )
    )

    assert result.status is _ContextReconciliationStatus.GLOBAL_INVALID
    assert result.affected_tasks == ()


def test_reordered_evidence_is_transport_only_and_known_duplicate_is_local() -> None:
    reordered = _frames()
    _bind_dispatches(reordered)
    exact = tuple(reversed(_exact_evidence(reordered)))
    assert reordered.reconcile(exact).status is _ContextReconciliationStatus.VERIFIED

    duplicated = _frames()
    _bind_dispatches(duplicated)
    exact_duplicate = _exact_evidence(duplicated)
    duplicate_evidence = (*exact_duplicate, exact_duplicate[0])
    result = duplicated.reconcile(duplicate_evidence)

    assert result.status is _ContextReconciliationStatus.LOCAL_INVALID
    assert result.affected_tasks == (duplicated.tasks[0],)


@pytest.mark.parametrize(
    ("attestations", "status"),
    [
        ((), _ContextCleanupStatus.UNCONFIRMED),
        ((False,), _ContextCleanupStatus.FAILED),
        ((True, True), _ContextCleanupStatus.UNCONFIRMED),
    ],
)
def test_cleanup_requires_one_exact_trusted_artifact_attestation(
    attestations: tuple[bool, ...],
    status: _ContextCleanupStatus,
) -> None:
    frames = _frames()
    _bind_dispatches(frames)
    assert frames.reconcile(_exact_evidence(frames)).status is _ContextReconciliationStatus.VERIFIED

    result = frames.close(
        tuple(
            _BackendClosureAttestation(frames.artifact_id, _BackendArtifactClass.CONTEXT_REQUEST, closed)
            for closed in attestations
        )
    )

    assert result.status is status
    assert frames.target_frame.empty
    assert frames.context_frame.empty


def test_pre_bind_evidence_and_attestation_fail_closed() -> None:
    frames = _frames()

    reconciliation = frames.reconcile(_exact_evidence(frames))
    cleanup = frames.close(
        (_BackendClosureAttestation(frames.artifact_id, _BackendArtifactClass.CONTEXT_REQUEST, True),)
    )

    assert reconciliation.status is _ContextReconciliationStatus.GLOBAL_INVALID
    assert cleanup.status is _ContextCleanupStatus.UNCONFIRMED


@pytest.mark.parametrize(
    "attestation",
    [
        _BackendClosureAttestation(
            _BackendArtifactId("foreign-artifact"),
            _BackendArtifactClass.CONTEXT_REQUEST,
            True,
        ),
        _BackendClosureAttestation(
            _BackendArtifactId("artifact-a"),
            _BackendArtifactClass.CONTEXT_REQUEST,
            True,
            schema_version=cast(_ContextSchemaVersion, "future-schema"),
        ),
    ],
)
def test_foreign_or_incompatible_cleanup_evidence_is_unconfirmed(
    attestation: _BackendClosureAttestation,
) -> None:
    frames = _frames()
    _bind_dispatches(frames)
    evidence = _exact_evidence(frames)
    assert frames.reconcile(evidence).status is _ContextReconciliationStatus.VERIFIED

    assert frames.close((attestation,)).status is _ContextCleanupStatus.UNCONFIRMED


def test_mutated_context_payload_cannot_satisfy_a_compiled_binding() -> None:
    frames = _frames()
    _bind_dispatches(frames)
    evidence = list(_exact_evidence(frames))
    evidence[0] = replace(evidence[0], payload_token=_ContextPayloadToken("foreign-payload"))

    result = frames.reconcile(tuple(evidence))

    assert result.status is _ContextReconciliationStatus.LOCAL_INVALID
    assert result.affected_tasks == (frames.tasks[0],)

    with pytest.raises(_WorkframeStateError):
        _make_context_binding_evidence("binding", "owner", 0, "mutated-context")


def test_malformed_unhashable_binding_evidence_fails_closed_and_can_cleanup() -> None:
    frames = _frames()
    _bind_dispatches(frames)
    evidence = list(_exact_evidence(frames))
    object.__setattr__(evidence[0], "binding_id", cast(_ContextBindingId, []))

    result = frames.reconcile(tuple(evidence))
    cleanup = frames.close(
        (_BackendClosureAttestation(frames.artifact_id, _BackendArtifactClass.CONTEXT_REQUEST, True),)
    )

    assert result.status is _ContextReconciliationStatus.GLOBAL_INVALID
    assert cleanup.status is _ContextCleanupStatus.VERIFIED


def _frames():
    plan = _plan()
    return _lower_context_workframes(
        plan,
        tuple(projection.owner_task for projection in plan.projections),
        target_work_ids=("target-work-a", "target-work-b"),
        identity_factory=_identities("binding-a0", "binding-a1", "binding-b0", "artifact-a"),
    )


def _exact_evidence(frames) -> tuple[_ContextBindingEvidence, ...]:
    return tuple(
        _make_context_binding_evidence(
            row[COL_CONTEXT_BINDING_ID],
            row[COL_CONTEXT_OWNER_WORK_ID],
            row[COL_CONTEXT_ORDINAL],
            row[COL_CONTEXT_TEXT],
        )
        for _index, row in frames.context_frame.iterrows()
    )


def _dispatches_for(frames) -> tuple[_Dispatch, ...]:
    return tuple(
        _Dispatch(
            _InvocationId("invocation"),
            task,
            _AttemptId(f"attempt-{index}"),
            _RowToken(work_id.value),
        )
        for index, (task, work_id) in enumerate(zip(frames.tasks, frames.target_work_ids(), strict=True))
    )


def _bind_dispatches(frames) -> None:
    frames.bind_dispatches(_dispatches_for(frames))


def _identities(*values: str):
    iterator: Iterator[str] = iter(values)
    return lambda: next(iterator)


def _plan() -> _ContextPlan:
    target_a = _TextDatum(_DatumId("target-a"), "alpha", _DatumPurpose.TARGET)
    target_b = _TextDatum(_DatumId("target-b"), "beta", _DatumPurpose.TARGET)
    context = _TextDatum(_DatumId("context-c"), "gamma", _DatumPurpose.CONTEXT_ONLY)
    graph = _ProtectionGraph(
        datums=(target_a, target_b, context),
        links=(),
        context_scopes=(
            _ContextScope(target_a.id, (context.id, target_b.id)),
            _ContextScope(target_b.id, (target_a.id,)),
        ),
        coherence_scopes=(_CoherenceScope((target_a.id,)), _CoherenceScope((target_b.id,))),
        atomic_groups=(_AtomicGroup((target_a.id,)), _AtomicGroup((target_b.id,))),
    )
    limits = _ContextLimits(2, 32, 4, 128)
    contract = _ContextExecutionContract(
        _ContextProfile.TARGET_CONTEXT_V1,
        _ContextSchemaVersion.V1,
        limits,
        True,
        _ContextOrdering.DECLARED,
        (_BackendArtifactClass.CONTEXT_REQUEST,),
    )
    capability = _ContextBackendCapability(
        contract.profile,
        contract.schema_version,
        limits,
        True,
        contract.ordering,
        contract.required_artifacts,
        _RetentionPosture.DISABLED,
    )
    result = _compile_context_plan(
        graph,
        accounting_limits=_AccountingLimits(8, 64, 256),
        contract=contract,
        capability=capability,
    )
    assert isinstance(result, _ContextPlan)
    return result
