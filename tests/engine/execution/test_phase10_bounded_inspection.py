# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import pickle
from dataclasses import FrozenInstanceError
from typing import Any

import pytest


def _module() -> Any:
    return importlib.import_module("anonymizer.engine.execution.phase10_inspection")


def _provenance(module: Any, kind: Any) -> Any:
    return module._Phase10Provenance(
        module._Phase10InspectionSchemaVersion.V1,
        module._Phase10ContractVersion.V1,
        kind,
        module._Phase10SubjectKind.INVOCATION_SNAPSHOT,
        module._Phase10SemanticProfile.GROUPED_REWRITE_V1,
        module._Phase10ImplementationProfile.PANDAS_RUNTIME_V1,
        module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED,
        module._Phase10LifecycleState.TERMINAL,
    )


def _diagnostic(module: Any) -> Any:
    return module._Phase10Diagnostic(
        module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED,
        module._Phase10Stage.REWRITE,
        module._Phase10TerminalState.FAILED,
        module._Phase10ReasonCategory.BACKEND_FAILED,
        module._Phase10CountBucket.ONE,
        module._Phase10ReconciliationState.RECONCILED,
        module._Phase10CleanupState.NOT_ENTERED,
    )


def test_phase10_private_values_reject_unknown_closed_values() -> None:
    module = _module()

    for enum_type in (
        module._Phase10Operation,
        module._Phase10ViewKind,
        module._Phase10SubjectKind,
        module._Phase10CaptureBoundary,
        module._Phase10LifecycleState,
        module._Phase10ReasonCategory,
        module._Phase10CountBucket,
    ):
        with pytest.raises(ValueError):
            enum_type("unknown")
    with pytest.raises(TypeError):
        module._Phase10Diagnostic(
            "terminal_evidence_accepted",
            module._Phase10Stage.REWRITE,
            module._Phase10TerminalState.FAILED,
            module._Phase10ReasonCategory.BACKEND_FAILED,
            module._Phase10CountBucket.ONE,
            module._Phase10ReconciliationState.RECONCILED,
            module._Phase10CleanupState.NOT_ENTERED,
        )
    for value in (True, 1.0, -1, 65_537):
        with pytest.raises(TypeError):
            module._Phase10DeclaredLimit(module._Phase10LimitName.SUBJECTS_PER_CALL, value)


def test_phase10_grant_is_identity_bound_single_use_and_cause_free() -> None:
    module = _module()
    owner = object()
    subject = object()
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.INSPECT)

    assert type(grant) is module._Phase10InspectionGrant
    assert not module._consume_phase10_inspection_grant(
        grant,
        owner,
        subject,
        module._Phase10Operation.EXPLAIN,
    )
    assert module._consume_phase10_inspection_grant(
        grant,
        owner,
        subject,
        module._Phase10Operation.INSPECT,
    )
    assert not module._consume_phase10_inspection_grant(
        grant,
        owner,
        subject,
        module._Phase10Operation.INSPECT,
    )
    assert repr(grant).startswith("<private ")
    assert repr(owner) not in repr(grant)
    assert repr(subject) not in repr(grant)
    assert not hasattr(grant, "_state")
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(grant)
    with pytest.raises(FrozenInstanceError):
        grant.operation = module._Phase10Operation.EXPLAIN


def test_phase10_grant_rejects_forgery_wrong_identity_and_expiry() -> None:
    module = _module()
    owner = object()
    subject = object()
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.DIAGNOSE)

    assert not module._consume_phase10_inspection_grant(
        grant,
        object(),
        subject,
        module._Phase10Operation.DIAGNOSE,
    )
    module._revoke_phase10_inspection_grant(grant)
    assert not module._consume_phase10_inspection_grant(
        grant,
        owner,
        subject,
        module._Phase10Operation.DIAGNOSE,
    )
    forged = module._Phase10InspectionGrant(module._Phase10Operation.DIAGNOSE, object(), object())
    assert not module._consume_phase10_inspection_grant(
        forged,
        owner,
        subject,
        module._Phase10Operation.DIAGNOSE,
    )


def test_phase10_snapshot_view_provenance_and_diagnostic_are_immutable_and_masked() -> None:
    module = _module()
    stage = module._Phase10StageSummary(
        module._Phase10Stage.REWRITE,
        module._Phase10LifecycleState.TERMINAL,
        module._Phase10CountBucket.ONE,
    )
    terminal = module._Phase10TerminalSummary(
        module._Phase10Stage.REWRITE,
        module._Phase10TerminalState.FAILED,
        module._Phase10CountBucket.ONE,
    )
    snapshot = module._Phase10Snapshot(
        (stage,),
        (terminal,),
        module._Phase10ReconciliationState.RECONCILED,
        module._Phase10CleanupState.NOT_ENTERED,
        module._Phase10ReleaseState.WITHHELD,
    )
    view = module._Phase10InspectView(_provenance(module, module._Phase10ViewKind.INSPECT), snapshot)

    for value in (stage, terminal, snapshot, view.provenance, _diagnostic(module), view):
        assert repr(value).startswith("<private ")
        with pytest.raises(TypeError, match="not serializable"):
            pickle.dumps(value)
    with pytest.raises(FrozenInstanceError):
        snapshot.cleanup_state = module._Phase10CleanupState.VERIFIED


def test_phase10_encoder_emits_exact_canonical_json_and_preserves_semantic_array_order() -> None:
    module = _module()
    provenance = module._Phase10Provenance(
        module._Phase10InspectionSchemaVersion.V1,
        module._Phase10ContractVersion.V1,
        module._Phase10ViewKind.EXPLAIN,
        module._Phase10SubjectKind.ADMITTED_PLAN,
        module._Phase10SemanticProfile.GROUPED_REWRITE_V1,
        module._Phase10ImplementationProfile.PANDAS_RUNTIME_V1,
        module._Phase10CaptureBoundary.ADMISSION_TERMINAL,
        module._Phase10LifecycleState.TERMINAL,
    )
    view = module._Phase10ExplainView(
        provenance,
        module._Phase10Route.MIXED,
        (module._Phase10Capability.GROUPED_REWRITE, module._Phase10Capability.TERMINAL_ACCOUNTING),
        (
            module._Phase10DeclaredLimit(module._Phase10LimitName.MAX_STAGE_SUMMARIES, 8),
            module._Phase10DeclaredLimit(module._Phase10LimitName.SUBJECTS_PER_CALL, 1),
        ),
        (
            module._Phase10Aggregate(
                module._Phase10AggregateDimension.RELATIONSHIPS, module._Phase10CountBucket.TWO_TO_FOUR
            ),
        ),
        None,
    )

    result = module._encode_phase10_view(view)

    assert type(result) is module._Phase10CanonicalEncoding
    assert not result.value.endswith(b"\n")
    assert result.value == json.dumps(
        json.loads(result.value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = json.loads(result.value)
    assert [item["name"] for item in payload["declared_limits"]] == [
        "max_stage_summaries",
        "subjects_per_call",
    ]
    assert payload["provenance"] == {
        "capture_boundary": "admission_terminal",
        "capture_lifecycle_state": "terminal",
        "implementation_profile_version": "pandas-runtime-v1",
        "inspection_contract_version": "anonymizer-phase10-bounded-inspection/v1",
        "inspection_schema_version": "phase10-bounded-inspection-view/v1",
        "semantic_profile_version": "anonymizer-phase8-grouped-rewrite/v1",
        "subject_kind": "admitted_plan",
        "view_kind": "explain",
    }
    assert repr(result).startswith("<private ")
    assert result.value.decode("utf-8") not in repr(result)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(result)


def test_phase10_encoder_rejects_unknown_values_and_encoding_overflow_without_partial_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    unknown = module._encode_phase10_view(object())
    assert type(unknown) is module._Phase10InspectionRejected
    assert unknown.code is module._Phase10RejectionCode.REDACTION_FAILED

    view = module._Phase10DiagnoseView(
        _provenance(module, module._Phase10ViewKind.DIAGNOSE),
        (_diagnostic(module),),
    )
    monkeypatch.setattr(module, "_MAX_CANONICAL_JSON_BYTES", 1)
    oversized = module._encode_phase10_view(view)

    assert type(oversized) is module._Phase10InspectionRejected
    assert oversized.code is module._Phase10RejectionCode.LIMIT_EXCEEDED
    assert "inspection_limit_exceeded" not in repr(oversized)


def test_phase10_types_are_not_exposed_through_public_surfaces() -> None:
    import anonymizer
    import anonymizer.interface
    from anonymizer import Anonymizer

    public_names = set(dir(anonymizer)) | set(dir(anonymizer.interface)) | set(dir(Anonymizer))
    assert not any(
        name in public_names
        for name in (
            "explain",
            "inspect",
            "diagnose",
            "InspectionGrant",
            "InspectionView",
            "Phase10Snapshot",
        )
    )
