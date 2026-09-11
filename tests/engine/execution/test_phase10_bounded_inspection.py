# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import importlib
import json
import pickle
import weakref
from dataclasses import FrozenInstanceError, fields, replace
from typing import Any, cast

import pytest

from anonymizer.engine.execution.accounting_admission import (
    _AccountingAdmissionCode,
    _AccountingRejected,
    _compile_accounting_plan,
)
from anonymizer.engine.execution.accounting_ledger import _AccountingLedger
from anonymizer.engine.execution.accounting_plan import _AccountingLimits, _AccountingPlan
from anonymizer.engine.execution.graph import _DatumId, _RewriteGroup, _TextDatum, _trivial_graph
from anonymizer.engine.execution.phase8_cleanup import (
    _issue_phase8_cleanup_receipt,
    _Phase8CleanupComponent,
    _Phase8CleanupPhase,
    _Phase8CleanupStatus,
)


def _module() -> Any:
    return importlib.import_module("anonymizer.engine.execution.phase10_inspection")


class _InspectionIdentity:
    pass


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


def _accounting_plan(*texts: str, stages: tuple[str, ...] = ("protect",)) -> _AccountingPlan:
    graph = _trivial_graph(
        tuple(_TextDatum(_DatumId(f"private-datum-{index}"), text) for index, text in enumerate(texts))
    )
    result = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(
            max_datums=len(texts),
            max_datum_bytes=256,
            max_graph_bytes=512,
            max_stages=len(stages),
        ),
        stages=stages,
    )
    assert isinstance(result, _AccountingPlan)
    return result


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
    owner = _InspectionIdentity()
    subject = _InspectionIdentity()
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
    owner = _InspectionIdentity()
    subject = _InspectionIdentity()
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


def test_phase10_explain_projects_admitted_plan_and_rejection_without_content() -> None:
    module = _module()
    assert callable(getattr(module, "_explain_phase10", None)), "Tier 2 explain builder is missing"
    owner = _InspectionIdentity()
    plan = _accounting_plan("EXPLAIN-CONTENT-CANARY")
    grant = module._issue_phase10_inspection_grant(owner, plan, module._Phase10Operation.EXPLAIN)

    admitted = module._explain_phase10(owner, plan, grant)

    assert type(admitted) is module._Phase10ExplainView
    assert admitted.route is module._Phase10Route.NDD
    assert admitted.provenance.subject_kind is module._Phase10SubjectKind.ADMITTED_PLAN
    assert admitted.required_capabilities == (module._Phase10Capability.TERMINAL_ACCOUNTING,)
    encoded = module._encode_phase10_view(admitted)
    assert type(encoded) is module._Phase10CanonicalEncoding
    assert b"EXPLAIN-CONTENT-CANARY" not in encoded.value
    assert b"private-datum" not in encoded.value

    rejection = _AccountingRejected(_AccountingAdmissionCode.TOO_MANY_DATUMS)
    rejection_grant = module._issue_phase10_inspection_grant(
        owner,
        rejection,
        module._Phase10Operation.EXPLAIN,
    )
    rejected = module._explain_phase10(owner, rejection, rejection_grant)

    assert type(rejected) is module._Phase10ExplainView
    assert rejected.route is module._Phase10Route.REJECTED
    assert rejected.rejection_category is module._Phase10ReasonCategory.LIMIT_EXCEEDED
    assert rejected.provenance.subject_kind is module._Phase10SubjectKind.ADMISSION_REJECTION


def test_phase10_explain_projects_grouped_rewrite_plan_in_compiled_order() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_admission")
    graph = _trivial_graph((_TextDatum(_DatumId("phase8-private"), "PHASE8-PLAN-CANARY"),))
    graph = replace(graph, rewrite_groups=(_RewriteGroup((graph.datums[0].id,)),))
    plan = phase8._compile_phase8_plan(graph, max_repairs=2)
    assert phase8._is_admitted_phase8_plan(plan)
    owner = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, plan, module._Phase10Operation.EXPLAIN)

    explained = module._explain_phase10(owner, plan, grant)

    assert type(explained) is module._Phase10ExplainView
    assert explained.required_capabilities[-1] is module._Phase10Capability.GROUPED_REWRITE
    assert tuple(item.dimension for item in explained.relationship_buckets) == (
        module._Phase10AggregateDimension.DATUMS,
        module._Phase10AggregateDimension.GROUPS,
        module._Phase10AggregateDimension.OPERATIONS,
        module._Phase10AggregateDimension.REPAIRS,
    )
    encoded = module._encode_phase10_view(explained)
    assert type(encoded) is module._Phase10CanonicalEncoding
    assert b"PHASE8-PLAN-CANARY" not in encoded.value


def test_phase10_inspect_and_diagnose_use_one_owner_linearized_accounting_snapshot() -> None:
    module = _module()
    owner = _InspectionIdentity()
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("LEDGER-CONTENT-CANARY"))
    ledger.open()
    (task,) = ledger.ready_tasks()
    dispatch = ledger.dispatch(task)
    ledger.accept_failure(dispatch)
    capture: Any = ledger._phase10_snapshot()
    assert type(capture) is module._Phase10OwnerCapture
    assert repr(capture).startswith("<private ")
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(capture)

    inspect_grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.INSPECT)
    inspected = module._inspect_phase10(owner, capture, inspect_grant)

    assert type(inspected) is module._Phase10InspectView
    assert inspected.provenance.capture_boundary is module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED
    assert inspected.snapshot.stage_summaries == (
        module._Phase10StageSummary(
            module._Phase10Stage.TRANSFORM,
            module._Phase10LifecycleState.TERMINAL,
            module._Phase10CountBucket.ONE,
        ),
    )
    assert inspected.snapshot.terminal_summaries == (
        module._Phase10TerminalSummary(
            module._Phase10Stage.TRANSFORM,
            module._Phase10TerminalState.FAILED,
            module._Phase10CountBucket.ONE,
        ),
    )

    diagnose_grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.DIAGNOSE)
    diagnosed = module._diagnose_phase10(owner, capture, diagnose_grant)

    assert type(diagnosed) is module._Phase10DiagnoseView
    assert diagnosed.diagnostics == (
        module._Phase10Diagnostic(
            module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED,
            module._Phase10Stage.TRANSFORM,
            module._Phase10TerminalState.FAILED,
            module._Phase10ReasonCategory.BACKEND_FAILED,
            module._Phase10CountBucket.ONE,
            module._Phase10ReconciliationState.RECONCILED,
            module._Phase10CleanupState.NOT_ENTERED,
        ),
    )
    for view in (inspected, diagnosed):
        encoded = module._encode_phase10_view(view)
        assert type(encoded) is module._Phase10CanonicalEncoding
        assert b"LEDGER-CONTENT-CANARY" not in encoded.value
        assert b"private-datum" not in encoded.value


def test_phase10_denies_before_subject_snapshot_access() -> None:
    module = _module()
    owner = _InspectionIdentity()

    class DenialProbe:
        def __getattribute__(self, _name: str) -> object:
            raise AssertionError("subject accessed before grant validation")

    subject = DenialProbe()
    forged = module._Phase10InspectionGrant(module._Phase10Operation.INSPECT, object(), object())

    denied = module._inspect_phase10(owner, subject, forged)

    assert type(denied) is module._Phase10InspectionRejected
    assert denied.code is module._Phase10RejectionCode.DENIED


def test_phase10_cleanup_receipt_snapshot_is_bucketed_and_identity_free() -> None:
    module = _module()
    owner = _InspectionIdentity()
    identity = object()
    receipt = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.POST_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.UNCONFIRMED,
        identity,
        retained_candidate_cell_count=17,
    )
    capture: Any = receipt._phase10_snapshot()
    assert type(capture) is module._Phase10OwnerCapture
    grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.INSPECT)

    inspected = module._inspect_phase10(owner, capture, grant)

    assert type(inspected) is module._Phase10InspectView
    assert inspected.provenance.subject_kind is module._Phase10SubjectKind.CLEANUP_RECEIPT
    assert inspected.provenance.capture_boundary is module._Phase10CaptureBoundary.POST_REDUCTION_CLEANUP_TERMINAL
    assert inspected.snapshot.cleanup_state is module._Phase10CleanupState.UNCONFIRMED
    assert inspected.snapshot.stage_summaries == (
        module._Phase10StageSummary(
            module._Phase10Stage.CLEANUP,
            module._Phase10LifecycleState.CLEANUP_TERMINAL,
            module._Phase10CountBucket.ONE,
        ),
    )
    assert inspected.snapshot.terminal_summaries == (
        module._Phase10TerminalSummary(
            module._Phase10Stage.CLEANUP,
            module._Phase10TerminalState.INCONSISTENT,
            module._Phase10CountBucket.SEVENTEEN_TO_SIXTY_FOUR,
        ),
    )
    encoded = module._encode_phase10_view(inspected)
    assert type(encoded) is module._Phase10CanonicalEncoding
    assert str(id(identity)).encode() not in encoded.value


def test_phase10_phase8_operation_snapshot_collapses_rounds_and_preserves_reason_order() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(2, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    ledger.succeed(phase8._Phase8Stage.validate_baselines())
    ledger.fail(phase8._Phase8Stage.analyze(), phase8._Phase8Reason.BACKEND_FAILURE)
    owner = _InspectionIdentity()
    capture: Any = ledger._phase10_snapshot()
    assert type(capture) is module._Phase10OwnerCapture
    inspect_grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.INSPECT)

    inspected = module._inspect_phase10(owner, capture, inspect_grant)

    assert type(inspected) is module._Phase10InspectView
    assert tuple(item.stage for item in inspected.snapshot.stage_summaries) == (
        module._Phase10Stage.VALIDATE,
        module._Phase10Stage.ANALYZE,
        module._Phase10Stage.REWRITE,
        module._Phase10Stage.EVALUATE,
        module._Phase10Stage.REPAIR,
    )
    diagnose_grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.DIAGNOSE)
    diagnosed = module._diagnose_phase10(owner, capture, diagnose_grant)
    assert type(diagnosed) is module._Phase10DiagnoseView
    assert tuple(item.reason_category for item in diagnosed.diagnostics) == (
        module._Phase10ReasonCategory.BACKEND_FAILED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
    )


def test_phase10_reason_table_is_total_for_every_current_phase4_through_phase8_owner_reason() -> None:
    module = _module()

    assert module._unmapped_phase10_reason_values() == ()
    assert {reason_type.__name__ for reason_type in module._PHASE10_REASON_TABLE} == {
        "_AccountingAdmissionCode",
        "_ApplicationRejectionCode",
        "_BundleRejectionCode",
        "_CauseCode",
        "_ContextAdmissionCode",
        "_ContextBindingFault",
        "_MentionRejectionCode",
        "_PatchRejectionCode",
        "_Phase6PlanRejectionCode",
        "_Phase7AdmissionCode",
        "_Phase7ContractRejectionCode",
        "_Phase7NddReason",
        "_Phase8AdmissionCode",
        "_Phase8Reason",
        "_ResolutionRejectionCode",
        "_RolePolicyRejectionCode",
        "_UnsupportedRoleReason",
    }
    assert module._map_phase10_reason(object()) is module._Phase10ReasonCategory.UNEXPECTED_FAILURE


def test_phase10_phase8_cancellation_diagnostics_distinguish_pre_dispatch_from_trusted_stop() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    before = phase8._Phase8OperationLedger(plan)
    before.cancel(phase8._Phase8Stage.validate_baselines(), trusted_stop=True, dispatched=False)
    before_capture: Any = before._phase10_snapshot()
    assert tuple(item.reason_category for item in before_capture.diagnostics) == (
        module._Phase10ReasonCategory.CANCELLATION_BEFORE_DISPATCH,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
    )

    after = phase8._Phase8OperationLedger(plan)
    after.succeed(phase8._Phase8Stage.validate_baselines())
    after.cancel(phase8._Phase8Stage.analyze(), trusted_stop=True, dispatched=True)
    after_capture: Any = after._phase10_snapshot()
    assert tuple(item.reason_category for item in after_capture.diagnostics) == (
        module._Phase10ReasonCategory.STOP_ACKNOWLEDGED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
    )


def test_phase10_phase8_successful_no_repair_route_has_no_failure_diagnostic() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(1, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    for stage in (
        phase8._Phase8Stage.validate_baselines(),
        phase8._Phase8Stage.analyze(),
        phase8._Phase8Stage.rewrite(),
        phase8._Phase8Stage.evaluate(0),
    ):
        ledger.succeed(stage)
    ledger.close_pass(phase8._Phase8Stage.evaluate(0))
    capture: Any = ledger._phase10_snapshot()
    assert capture.diagnostics == ()
    owner = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.DIAGNOSE)

    rejected = module._diagnose_phase10(owner, capture, grant)

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.STATE_UNAVAILABLE


def test_phase10_owner_capture_rejects_incoherent_boundary_and_diagnostic_state() -> None:
    module = _module()
    snapshot = module._Phase10Snapshot(
        (
            module._Phase10StageSummary(
                module._Phase10Stage.REWRITE,
                module._Phase10LifecycleState.TERMINAL,
                module._Phase10CountBucket.ONE,
            ),
        ),
        (
            module._Phase10TerminalSummary(
                module._Phase10Stage.REWRITE,
                module._Phase10TerminalState.FAILED,
                module._Phase10CountBucket.ONE,
            ),
        ),
        module._Phase10ReconciliationState.RECONCILED,
        module._Phase10CleanupState.NOT_ENTERED,
        module._Phase10ReleaseState.WITHHELD,
    )
    diagnostic = module._Phase10Diagnostic(
        module._Phase10CaptureBoundary.POST_DISPATCH,
        module._Phase10Stage.REWRITE,
        module._Phase10TerminalState.FAILED,
        module._Phase10ReasonCategory.BACKEND_FAILED,
        module._Phase10CountBucket.ONE,
        module._Phase10ReconciliationState.RECONCILED,
        module._Phase10CleanupState.NOT_ENTERED,
    )

    with pytest.raises(TypeError, match="owner capture"):
        module._Phase10OwnerCapture(
            module._Phase10SubjectKind.INVOCATION_SNAPSHOT,
            module._Phase10SemanticProfile.GROUPED_REWRITE_V1,
            module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED,
            module._Phase10LifecycleState.TERMINAL,
            snapshot,
            (diagnostic,),
        )


def test_phase10_closed_graph_execution_snapshot_copies_no_candidate_or_plan_content() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan("GRAPH-PLAN-CONTENT-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    (task,) = ledger.ready_tasks()
    ledger.mark_task_succeeded(task, "GRAPH-CANDIDATE-CONTENT-CANARY")
    result = ledger.finish()
    execution = graph_runtime._AccountingGraphExecution(plan, result, ())

    capture: Any = execution._phase10_snapshot()

    assert type(capture) is module._Phase10OwnerCapture
    assert capture.subject_kind is module._Phase10SubjectKind.TERMINAL_RECEIPT
    assert capture.snapshot.release_state is module._Phase10ReleaseState.RELEASED
    owner = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.INSPECT)
    inspected = module._inspect_phase10(owner, capture, grant)
    encoded = module._encode_phase10_view(inspected)
    assert type(encoded) is module._Phase10CanonicalEncoding
    assert b"GRAPH-PLAN-CONTENT-CANARY" not in encoded.value
    assert b"GRAPH-CANDIDATE-CONTENT-CANARY" not in encoded.value


def test_phase10_phase8_lifecycle_snapshot_uses_only_terminal_and_cleanup_receipts() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    identity = object()
    pre = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        identity,
    )
    post = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.POST_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        identity,
        retained_candidate_cell_count=1,
    )
    lifecycle = phase8._Phase8LifecycleExecution(
        ((object(), "LIFECYCLE-CANDIDATE-CANARY"),),
        ("succeeded",),
        False,
        pre,
        post,
    )

    capture: Any = lifecycle._phase10_snapshot()

    assert type(capture) is module._Phase10OwnerCapture
    assert capture.snapshot.cleanup_state is module._Phase10CleanupState.VERIFIED
    assert capture.snapshot.release_state is module._Phase10ReleaseState.RELEASED
    owner = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.INSPECT)
    inspected = module._inspect_phase10(owner, capture, grant)
    encoded = module._encode_phase10_view(inspected)
    assert type(encoded) is module._Phase10CanonicalEncoding
    assert b"LIFECYCLE-CANDIDATE-CANARY" not in encoded.value
    assert str(id(identity)).encode() not in encoded.value


def test_phase10_owner_snapshot_rejects_more_than_eight_distinct_stages_without_partial_view() -> None:
    module = _module()
    stages = ("admission", "detect", "augment", "validate", "finalize", "resolve", "classify", "transform", "verify")
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("LIMIT-CANARY", stages=stages))
    ledger.open()

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


@pytest.mark.parametrize("closed", ["owner", "subject"])
def test_phase10_grant_expires_without_retaining_closed_identity(closed: str) -> None:
    module = _module()

    class Identity:
        pass

    owner = Identity()
    subject = Identity()
    owner_ref = weakref.ref(owner)
    subject_ref = weakref.ref(subject)
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.INSPECT)
    assert type(grant) is module._Phase10InspectionGrant
    nonce = grant._nonce
    try:
        if closed == "owner":
            del owner
        else:
            del subject
        gc.collect()

        assert (owner_ref() if closed == "owner" else subject_ref()) is None
        assert nonce not in module._GRANT_STATES
    finally:
        module._revoke_phase10_inspection_grant(grant)


def test_phase10_lifecycle_snapshot_rejects_throwing_cleanup_state_without_leaking() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")

    class ThrowingCleanup:
        @property
        def status(self) -> object:
            raise RuntimeError("THROWING-CLEANUP-PRIVACY-CANARY")

    execution = phase8._Phase8LifecycleExecution((), (), True, cast(Any, ThrowingCleanup()), None)
    try:
        rejected: Any = execution._phase10_snapshot()
    except Exception as error:  # pragma: no cover - red-phase privacy witness
        pytest.fail(f"owner snapshot leaked a malformed cleanup exception: {type(error).__name__}")

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert "THROWING-CLEANUP-PRIVACY-CANARY" not in repr(rejected)


def test_phase10_graph_snapshot_contains_throwing_nested_state_without_leaking() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")

    class ThrowingAccounting:
        @property
        def tasks(self) -> object:
            raise RuntimeError("THROWING-GRAPH-PRIVACY-CANARY")

    execution = graph_runtime._AccountingGraphExecution(
        _accounting_plan("GRAPH-OWNER-CANARY"),
        cast(Any, ThrowingAccounting()),
        (),
    )
    try:
        rejected: Any = execution._phase10_snapshot()
    except Exception as error:  # pragma: no cover - red-phase privacy witness
        pytest.fail(f"graph snapshot leaked malformed nested state: {type(error).__name__}")

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert "THROWING-GRAPH-PRIVACY-CANARY" not in repr(rejected)


def test_phase10_partial_phase8_terminal_snapshot_has_a_coherent_capture_pair() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(1, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    ledger.succeed(phase8._Phase8Stage.validate_baselines())

    capture: Any = ledger._phase10_snapshot()

    assert type(capture) is module._Phase10OwnerCapture
    assert capture.capture_boundary is module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED
    assert capture.capture_lifecycle_state is module._Phase10LifecycleState.TERMINAL


def test_phase10_closed_phase8_lifecycle_preserves_known_terminal_reason() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    service = importlib.import_module("anonymizer.engine.execution.phase8_service")
    member = object()
    operation_plan = phase8._compile_group_operation_plan(0, 3)
    assert operation_plan is not None

    def fail_operation(_members: tuple[object, ...], _baselines: dict[object, str]) -> Any:
        return phase8._Phase8GroupOutcome(
            phase8._GroupFailed(phase8._Phase8Reason.BACKEND_FAILURE),
            None,
            0,
            phase8._Phase8OperationLedger(operation_plan),
        )

    execution = service._Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=((member,),),
        atomic_groups=((member,),),
        dependencies=(),
        phase7_released=((member, "KNOWN-REASON-CONTENT-CANARY"),),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=(fail_operation,),
    )

    capture: Any = execution._phase10_snapshot()

    assert type(capture) is module._Phase10OwnerCapture
    assert tuple(item.reason_category for item in capture.diagnostics) == (
        module._Phase10ReasonCategory.BACKEND_FAILED,
    )


def test_phase10_diagnostic_bytes_follow_closed_reason_order_not_task_order() -> None:
    module = _module()
    ledger_module = importlib.import_module("anonymizer.engine.execution.accounting_ledger")
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")

    def encoded(codes: tuple[Any, Any]) -> bytes:
        ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("FIRST-CANARY", "SECOND-CANARY"))
        ledger.open()
        tasks = ledger.ready_tasks()
        assert len(tasks) == 2
        for task, code in zip(tasks, codes, strict=True):
            ledger._states[task] = outcomes._TaskFailed(task, ledger_module._causes(code))
        capture = ledger._phase10_snapshot()
        assert type(capture) is module._Phase10OwnerCapture
        owner = _InspectionIdentity()
        grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.DIAGNOSE)
        view = module._diagnose_phase10(owner, capture, grant)
        encoding = module._encode_phase10_view(view)
        assert type(encoding) is module._Phase10CanonicalEncoding
        return encoding.value

    verification = outcomes._CauseCode.VERIFICATION_FAILED
    backend = outcomes._CauseCode.KNOWN_FAILURE

    assert encoded((verification, backend)) == encoded((backend, verification))


def test_phase10_builder_budget_accepts_exact_limit_and_rejects_one_over() -> None:
    module = _module()
    budget = module._phase10_new_builder_budget()
    assert type(budget) is module._Phase10BuilderBudget

    for _ in range(module._MAX_STAGE_SUMMARIES + module._MAX_TERMINAL_SUMMARIES + module._MAX_DIAGNOSTICS):
        assert module._phase10_reserve_builder_row(budget)
    assert budget.used_bytes == module._MAX_BUILDER_WORKING_BYTES
    assert not module._phase10_reserve_builder_row(budget)
    assert budget.used_bytes == module._MAX_BUILDER_WORKING_BYTES


def test_phase10_owner_snapshot_reserves_builder_bytes_before_first_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "_BUILDER_BASE_BYTES", module._MAX_BUILDER_WORKING_BYTES)
    monkeypatch.setattr(module, "_BUILDER_ROW_BYTES", 1)
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("BUILDER-BUDGET-CANARY"))
    ledger.open()

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


def test_phase10_grant_rejects_nonweakrefable_bindings_without_id_fallback() -> None:
    module = _module()
    owner = _InspectionIdentity()
    subject = _InspectionIdentity()

    bad_owner = module._issue_phase10_inspection_grant(object(), subject, module._Phase10Operation.INSPECT)
    bad_subject = module._issue_phase10_inspection_grant(owner, object(), module._Phase10Operation.INSPECT)

    for rejected in (bad_owner, bad_subject):
        assert type(rejected) is module._Phase10InspectionRejected
        assert rejected.code is module._Phase10RejectionCode.SUBJECT_INVALID


def test_phase10_supported_phase8_rejection_has_weak_identity_binding() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_admission")
    owner = _InspectionIdentity()
    subject = phase8._Phase8Rejected(phase8._Phase8AdmissionCode.INVALID_INPUT)
    subject_ref = weakref.ref(subject)

    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.EXPLAIN)

    assert subject_ref() is subject
    assert type(grant) is module._Phase10InspectionGrant
    assert repr(subject).startswith("<private ")
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(subject)
    assert type(module._explain_phase10(owner, subject, grant)) is module._Phase10ExplainView


def test_phase10_phase8_private_sources_mask_repr_and_disable_pickle() -> None:
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    identity = object()
    receipt = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        identity,
    )
    values = (
        _AccountingLedger(_accounting_plan("PRIVATE-LEDGER-SERIALIZATION-CANARY")),
        phase8._Phase8OperationLedger(plan),
        phase8._Phase8LifecycleExecution(
            ((object(), "PRIVATE-SOURCE-SERIALIZATION-CANARY"),),
            ("succeeded",),
            False,
        ),
        receipt,
    )

    for value in values:
        representation = repr(value)
        assert representation.startswith("<private ")
        assert "PRIVATE-SOURCE-SERIALIZATION-CANARY" not in representation
        assert "0x" not in representation
        with pytest.raises(TypeError, match="not serializable"):
            pickle.dumps(value)


def test_phase10_snapshot_contains_throwing_lock_canaries() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")

    class ThrowingLock:
        def __enter__(self) -> object:
            raise RuntimeError("LOCK-PRIVACY-CANARY")

        def __exit__(self, *_args: object) -> None:
            return None

    accounting: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("LOCK-CANARY"))
    accounting._lock = cast(Any, ThrowingLock())
    operation_plan = phase8._compile_group_operation_plan(0, 3)
    assert operation_plan is not None
    operation = phase8._Phase8OperationLedger(operation_plan)
    operation._lock = cast(Any, ThrowingLock())

    for source in (accounting, operation):
        try:
            rejected: Any = source._phase10_snapshot()
        except Exception as error:  # pragma: no cover - red-phase privacy witness
            pytest.fail(f"snapshot leaked lock canary: {type(error).__name__}")
        assert type(rejected) is module._Phase10InspectionRejected
        assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED
        assert "LOCK-PRIVACY-CANARY" not in repr(rejected)


def test_phase10_phase8_operation_rejects_undispatched_terminal_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(1, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    evaluate_zero = phase8._Phase8Stage.evaluate(0)
    evaluate_one = phase8._Phase8Stage.evaluate(1)
    ledger._terminals[evaluate_zero] = phase8._StageFailed(evaluate_zero, phase8._Phase8Reason.ANALYSIS_STATE_MISSING)
    ledger._terminals[evaluate_one] = phase8._StageFailed(evaluate_one, phase8._Phase8Reason.BACKEND_FAILURE)

    capture: Any = ledger._phase10_snapshot()

    assert type(capture) is module._Phase10InspectionRejected
    assert capture.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_view_budget_precedes_explain_subject_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    owner = _InspectionIdentity()
    subject = _accounting_plan("EXPLAIN-BUDGET-ORDER-CANARY")
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.EXPLAIN)
    monkeypatch.setattr(module, "_BUILDER_BASE_BYTES", module._MAX_BUILDER_WORKING_BYTES)
    monkeypatch.setattr(module, "_BUILDER_ROW_BYTES", 1)

    def traverse_subject(_subject: object) -> object:
        raise RuntimeError("EXPLAIN-TRAVERSAL-CANARY")

    monkeypatch.setattr(module, "_phase10_explain_details", traverse_subject)

    result = module._explain_phase10(owner, subject, grant)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


def test_phase10_view_budget_precedes_diagnose_subject_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    owner = _InspectionIdentity()
    subject = _AccountingRejected(_AccountingAdmissionCode.TOO_MANY_DATUMS)
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.DIAGNOSE)
    monkeypatch.setattr(module, "_BUILDER_BASE_BYTES", module._MAX_BUILDER_WORKING_BYTES)
    monkeypatch.setattr(module, "_BUILDER_ROW_BYTES", 1)

    def traverse_subject(_subject: object) -> object:
        raise RuntimeError("DIAGNOSE-TRAVERSAL-CANARY")

    monkeypatch.setattr(module, "_phase10_admission_diagnostic", traverse_subject)

    result = module._diagnose_phase10(owner, subject, grant)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


def test_phase10_graph_execution_private_input_rejects_pickle_canaries() -> None:
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    execution = graph_runtime._AccountingGraphExecution(
        cast(Any, "GRAPH-PRIVATE-INPUT-CANARY"),
        cast(Any, "ACCOUNTING-PRIVATE-INPUT-CANARY"),
        (),
    )

    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(execution)


def test_phase10_phase8_operation_snapshot_rejects_unknown_terminal_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    ledger._terminals[phase8._Phase8Stage.validate_baselines()] = cast(Any, object())

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_phase8_lifecycle_snapshot_rejects_success_reason_mismatch_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    lifecycle = phase8._Phase8LifecycleExecution(
        (),
        ("succeeded",),
        False,
        terminal_group_reasons=(phase8._Phase8Reason.BACKEND_FAILURE,),
    )

    rejected: Any = lifecycle._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_phase8_lifecycle_snapshot_rejects_non_boolean_embargo_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    lifecycle = phase8._Phase8LifecycleExecution((), ("blocked",), cast(Any, "embargoed"))

    rejected: Any = lifecycle._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_phase8_lifecycle_snapshot_rejects_release_without_verified_cleanup_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    lifecycle = phase8._Phase8LifecycleExecution((), ("succeeded",), False)

    rejected: Any = lifecycle._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_malformed_exact_grant_is_cause_free_denial() -> None:
    module = _module()
    malformed = object.__new__(module._Phase10InspectionGrant)

    assert not module._consume_phase10_inspection_grant(
        malformed,
        _InspectionIdentity(),
        _InspectionIdentity(),
        module._Phase10Operation.INSPECT,
    )


def test_phase10_malformed_authorized_subject_is_cause_free_redaction() -> None:
    module = _module()
    owner = _InspectionIdentity()
    malformed = object.__new__(_AccountingRejected)
    grant = module._issue_phase10_inspection_grant(owner, malformed, module._Phase10Operation.EXPLAIN)

    result = module._explain_phase10(owner, malformed, grant)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_malformed_view_encoding_is_cause_free_redaction() -> None:
    module = _module()
    malformed = object.__new__(module._Phase10InspectView)

    result = module._encode_phase10_view(malformed)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_cleanup_snapshot_rejects_unknown_receipt_subclass_mutation() -> None:
    module = _module()
    cleanup = importlib.import_module("anonymizer.engine.execution.phase8_cleanup")

    class PrivacyCanaryReceipt(cleanup._Phase8CleanupReceipt):
        pass

    original = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        object(),
    )
    subclass = PrivacyCanaryReceipt(*(getattr(original, item.name) for item in fields(original)))

    rejected: Any = subclass._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_lifecycle_snapshot_rejects_mismatched_cleanup_pair_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    pre = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        object(),
    )
    post = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.POST_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        object(),
    )
    lifecycle = phase8._Phase8LifecycleExecution((), ("failed",), True, pre, post)

    rejected: Any = lifecycle._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_lifecycle_snapshot_rejects_wrong_cleanup_slot_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    post = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.POST_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.UNCONFIRMED,
        object(),
    )
    lifecycle = phase8._Phase8LifecycleExecution((), ("failed",), True, post, None)

    rejected: Any = lifecycle._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def _phase8_lifecycle_with_reasons(reasons: tuple[Any, ...]) -> Any:
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    service = importlib.import_module("anonymizer.engine.execution.phase8_service")
    members = tuple(object() for _ in reasons)
    operation_plan = phase8._compile_group_operation_plan(0, 3)
    assert operation_plan is not None
    operations = tuple(
        (
            lambda reason: (
                lambda _members, _baselines: phase8._Phase8GroupOutcome(
                    phase8._GroupFailed(reason),
                    None,
                    0,
                    phase8._Phase8OperationLedger(operation_plan),
                )
            )
        )(reason)
        for reason in reasons
    )
    return service._Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=tuple((member,) for member in members),
        atomic_groups=tuple((member,) for member in members),
        dependencies=(),
        phase7_released=tuple((member, "REASON-LIMIT-CANARY") for member in members),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=operations,
    )


def test_phase10_lifecycle_diagnostic_accepts_exact_four_owning_reasons() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    execution = _phase8_lifecycle_with_reasons(
        (
            phase8._Phase8Reason.ANALYSIS_RECONCILIATION,
            phase8._Phase8Reason.CANDIDATE_RECONCILIATION,
            phase8._Phase8Reason.EVALUATION_RECONCILIATION,
            phase8._Phase8Reason.REPAIR_RECONCILIATION,
        )
    )

    capture: Any = execution._phase10_snapshot()

    assert type(capture) is module._Phase10OwnerCapture


def test_phase10_lifecycle_diagnostic_rejects_fifth_owning_reason() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    execution = _phase8_lifecycle_with_reasons(
        (
            phase8._Phase8Reason.ANALYSIS_RECONCILIATION,
            phase8._Phase8Reason.CANDIDATE_RECONCILIATION,
            phase8._Phase8Reason.EVALUATION_RECONCILIATION,
            phase8._Phase8Reason.REPAIR_RECONCILIATION,
            phase8._Phase8Reason.REWRITE_RECONCILIATION,
        )
    )

    rejected: Any = execution._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


def test_phase10_post_dispatch_cancellation_reports_only_acknowledged_stop() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan("POST-DISPATCH-CANCEL-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    (task,) = ledger.ready_tasks()
    dispatch = ledger.dispatch(task)
    ledger.request_cancellation()
    ledger.acknowledge_stop(dispatch)

    live_capture: Any = ledger._phase10_snapshot()
    result = ledger.finish()
    closed_capture: Any = graph_runtime._AccountingGraphExecution(plan, result, ())._phase10_snapshot()

    for capture in (live_capture, closed_capture):
        assert type(capture) is module._Phase10OwnerCapture
        assert tuple(item.reason_category for item in capture.diagnostics) == (
            module._Phase10ReasonCategory.STOP_ACKNOWLEDGED,
        )


def test_phase10_accounting_snapshot_rejects_unknown_task_state_mutation() -> None:
    module = _module()
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("UNKNOWN-STATE-CANARY"))
    ledger.open()
    (task,) = ledger._plan.tasks
    ledger._states[task] = cast(Any, object())

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_graph_snapshot_rejects_mismatched_task_outcome_mutation() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan("UNKNOWN-OUTCOME-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    result = ledger.finish()
    malformed = replace(result, tasks=(cast(Any, object()),))

    rejected: Any = graph_runtime._AccountingGraphExecution(plan, malformed, ())._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_lifecycle_snapshot_rejects_cancelled_backend_reason_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    service = importlib.import_module("anonymizer.engine.execution.phase8_service")
    member = object()

    def cancel_with_backend_reason(_members: tuple[object, ...], _baselines: dict[object, str]) -> Any:
        plan = phase8._compile_group_operation_plan(0, 3)
        assert plan is not None
        return phase8._Phase8GroupOutcome(
            phase8._GroupCancelled(phase8._Phase8Reason.BACKEND_FAILURE),
            None,
            0,
            phase8._Phase8OperationLedger(plan),
        )

    execution = service._Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=((member,),),
        atomic_groups=((member,),),
        dependencies=(),
        phase7_released=((member, "CANCEL-REASON-CANARY"),),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=(cancel_with_backend_reason,),
    )

    rejected: Any = execution._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_operation_snapshot_rejects_missing_attempt_correlation_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    stage = phase8._Phase8Stage.validate_baselines()
    ledger._dispatched.add(stage)
    ledger._terminals[stage] = phase8._StageSucceeded(stage)

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_encoder_rejects_mutated_nested_stage_summary_without_reading_canary() -> None:
    module = _module()
    stage = module._Phase10StageSummary(
        module._Phase10Stage.REWRITE,
        module._Phase10LifecycleState.TERMINAL,
        module._Phase10CountBucket.ONE,
    )

    class ForeignStage:
        value = "NESTED-STAGE-CANARY"

    object.__setattr__(stage, "stage", ForeignStage())
    snapshot = module._Phase10Snapshot.__new__(module._Phase10Snapshot)
    object.__setattr__(snapshot, "stage_summaries", (stage,))
    object.__setattr__(snapshot, "terminal_summaries", ())
    object.__setattr__(snapshot, "reconciliation_state", module._Phase10ReconciliationState.RECONCILED)
    object.__setattr__(snapshot, "cleanup_state", module._Phase10CleanupState.NOT_ENTERED)
    object.__setattr__(snapshot, "release_state", module._Phase10ReleaseState.WITHHELD)
    view = module._Phase10InspectView.__new__(module._Phase10InspectView)
    object.__setattr__(view, "provenance", _provenance(module, module._Phase10ViewKind.INSPECT))
    object.__setattr__(view, "snapshot", snapshot)

    rejected = module._encode_phase10_view(view)

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert "NESTED-STAGE-CANARY" not in repr(rejected)


def test_phase10_grant_issue_and_weak_expiry_contain_throwing_lock_canary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    owner = _InspectionIdentity()
    subject = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.INSPECT)

    class ThrowingLock:
        def __enter__(self) -> object:
            raise RuntimeError("GRANT-LOCK-CANARY")

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(module, "_GRANT_LOCK", ThrowingLock())
    rejected = module._issue_phase10_inspection_grant(subject, owner, module._Phase10Operation.INSPECT)
    del owner
    gc.collect()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert "GRANT-LOCK-CANARY" not in capsys.readouterr().err
    del grant


def test_phase10_encoder_contains_throwing_json_encoder_canary(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    view = module._Phase10DiagnoseView(
        _provenance(module, module._Phase10ViewKind.DIAGNOSE),
        (_diagnostic(module),),
    )

    def throw_encoder(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("ENCODER-CANARY")

    monkeypatch.setattr(module.json, "dumps", throw_encoder)
    rejected = module._encode_phase10_view(view)

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.ENCODING_FAILED
    assert "ENCODER-CANARY" not in repr(rejected)


def test_phase10_cleanup_snapshot_rejects_foreign_proof_mutation() -> None:
    module = _module()
    cleanup = importlib.import_module("anonymizer.engine.execution.phase8_cleanup")
    identity = object()
    receipt = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        identity,
    )
    object.__setattr__(receipt, "_proof", cast(Any, object()))

    rejected: Any = receipt._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert not cleanup._is_phase8_cleanup_receipt(
        receipt,
        identity=identity,
        phase=_Phase8CleanupPhase.PRE_REDUCTION,
        component=_Phase8CleanupComponent.RUNTIME,
    )


def test_phase10_phase8_attempt_receipt_masks_repr_and_rejects_pickle() -> None:
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    receipt = phase8._Phase8AttemptReceipt(
        object(),
        phase8._Phase8Stage.validate_baselines(),
        phase8._Phase8TerminalKind.FAILED,
    )

    assert repr(receipt) == "<private phase 8 attempt receipt>"
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(receipt)


def test_phase10_phase8_operation_rejects_failed_cancellation_reason_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    stage = phase8._Phase8Stage.validate_baselines()
    assert ledger.dispatch(stage)
    ledger._terminals[stage] = phase8._StageFailed(stage, phase8._Phase8Reason.CANCELLATION)
    ledger._attempts[stage] = phase8._Phase8AttemptReceipt(
        plan.group_id,
        stage,
        phase8._Phase8TerminalKind.FAILED,
    )

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_live_accounting_rejects_released_state_before_close_mutation() -> None:
    module = _module()
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("OPEN-RELEASE-CANARY"))
    ledger.open()
    ledger._phase10_release_state = "released"

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_graph_snapshot_rejects_empty_groups_and_mismatched_invocation_mutations() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")
    plan = _accounting_plan("GROUP-CORRELATION-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    result = ledger.finish()
    empty_groups = replace(result, groups=())
    mismatched_invocation = replace(result, invocation=outcomes._InvocationCompleted(()))

    for malformed in (empty_groups, mismatched_invocation):
        rejected: Any = graph_runtime._AccountingGraphExecution(plan, malformed, ())._phase10_snapshot()
        assert type(rejected) is module._Phase10InspectionRejected
        assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_group_release_failure_emits_publication_diagnostic() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan("PUBLICATION-FAILURE-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    (task,) = ledger.ready_tasks()
    ledger.accept_success(ledger.dispatch(task), "protected")
    result = ledger.finish(group_release_predicate=lambda _outputs: False)

    for capture in (
        ledger._phase10_snapshot(),
        graph_runtime._AccountingGraphExecution(plan, result, ())._phase10_snapshot(),
    ):
        assert type(capture) is module._Phase10OwnerCapture
        capture = cast(Any, capture)
        assert module._Phase10ReasonCategory.PUBLICATION_FAILED in {
            item.reason_category for item in capture.diagnostics
        }


@pytest.mark.parametrize(
    ("mark", "category"),
    (
        ("mark_cleanup_failed", "CLEANUP_FAILED"),
        ("mark_cleanup_unconfirmed", "CLEANUP_UNCONFIRMED"),
    ),
)
def test_phase10_accounting_cleanup_failure_emits_diagnostic(mark: str, category: str) -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan(f"{category}-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    getattr(ledger, mark)()
    live_capture: Any = ledger._phase10_snapshot()
    result = ledger.finish()
    closed_capture: Any = graph_runtime._AccountingGraphExecution(plan, result, ())._phase10_snapshot()

    for capture in (live_capture, closed_capture):
        assert type(capture) is module._Phase10OwnerCapture
        assert getattr(module._Phase10ReasonCategory, category) in {
            item.reason_category for item in capture.diagnostics
        }
    assert live_capture.capture_boundary is module._Phase10CaptureBoundary.PRE_REDUCTION_CLEANUP_TERMINAL
    assert live_capture.capture_lifecycle_state is module._Phase10LifecycleState.CLEANUP_TERMINAL
    assert all(
        item.boundary is module._Phase10CaptureBoundary.PRE_REDUCTION_CLEANUP_TERMINAL
        for item in live_capture.diagnostics
    )


def test_phase10_untrusted_post_dispatch_cancellation_reports_only_execution_lost() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan("UNTRUSTED-CANCEL-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    (task,) = ledger.ready_tasks()
    ledger.dispatch(task)
    ledger.request_cancellation()
    result = ledger.finish()

    for capture in (
        ledger._phase10_snapshot(),
        graph_runtime._AccountingGraphExecution(plan, result, ())._phase10_snapshot(),
    ):
        assert type(capture) is module._Phase10OwnerCapture
        capture = cast(Any, capture)
        assert tuple(item.reason_category for item in capture.diagnostics) == (
            module._Phase10ReasonCategory.EXECUTION_LOST,
        )


def test_phase10_graph_snapshot_rejects_failed_task_with_cancellation_mutation() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")
    ledger_module = importlib.import_module("anonymizer.engine.execution.accounting_ledger")
    plan = _accounting_plan("TASK-SEMANTIC-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    result = ledger.finish()
    malformed = replace(
        result,
        tasks=(outcomes._TaskFailed(plan.tasks[0], ledger_module._causes(outcomes._CauseCode.CANCELLATION)),),
    )

    rejected: Any = graph_runtime._AccountingGraphExecution(plan, malformed, ())._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_phase8_preflight_failure_preserves_closed_reason() -> None:
    module = _module()
    service = importlib.import_module("anonymizer.engine.execution.phase8_service")
    execution = service._Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=(),
        atomic_groups=(),
        dependencies=(),
        phase7_released=(),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=(),
    )

    capture: Any = execution._phase10_snapshot()

    assert type(capture) is module._Phase10OwnerCapture
    assert tuple(item.reason_category for item in capture.diagnostics) == (
        module._Phase10ReasonCategory.ADMISSION_REJECTED,
    )


@pytest.mark.parametrize("reason_count", (4, 5))
def test_phase10_accounting_diagnostic_enforces_distinct_owning_reason_limit(reason_count: int) -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")
    ledger_module = importlib.import_module("anonymizer.engine.execution.accounting_ledger")
    plan = _accounting_plan(*(f"REASON-{index}-CANARY" for index in range(reason_count)))
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    codes = (
        outcomes._CauseCode.MISSING,
        outcomes._CauseCode.DUPLICATE,
        outcomes._CauseCode.UNKNOWN,
        outcomes._CauseCode.FOREIGN,
        outcomes._CauseCode.STALE,
    )
    for task, code in zip(plan.tasks, codes, strict=False):
        ledger._states[task] = outcomes._TaskInconsistent(task, ledger_module._causes(code))
    live_capture: Any = ledger._phase10_snapshot()
    result = ledger.finish()
    closed_capture: Any = graph_runtime._AccountingGraphExecution(plan, result, ())._phase10_snapshot()

    expected = module._Phase10OwnerCapture if reason_count == 4 else module._Phase10InspectionRejected
    assert type(live_capture) is expected
    assert type(closed_capture) is expected
    if reason_count == 5:
        assert live_capture.code is module._Phase10RejectionCode.LIMIT_EXCEEDED
        assert closed_capture.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


def test_phase10_encoder_rejects_mutated_nested_explain_rows() -> None:
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
    limit = module._Phase10DeclaredLimit(module._Phase10LimitName.SUBJECTS_PER_CALL, 1)
    aggregate = module._Phase10Aggregate(
        module._Phase10AggregateDimension.DATUMS,
        module._Phase10CountBucket.ONE,
    )
    view = module._Phase10ExplainView(
        provenance,
        module._Phase10Route.NDD,
        (),
        (limit,),
        (aggregate,),
        None,
    )
    object.__setattr__(limit, "value", cast(Any, "DECLARED-LIMIT-CANARY"))

    rejected = module._encode_phase10_view(view)

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_graph_correlation_never_invokes_candidate_equality_canary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")

    class CandidateCanary:
        calls = 0

        def __eq__(self, _other: object) -> bool:
            type(self).calls += 1
            print("CANDIDATE-EQUALITY-CANARY")
            raise RuntimeError("CANDIDATE-EQUALITY-CANARY")

    plan = _accounting_plan("CANDIDATE-CORRELATION-CANARY")
    ledger: _AccountingLedger[CandidateCanary] = _AccountingLedger(plan)
    ledger.open()
    (task,) = ledger.ready_tasks()
    ledger.accept_success(ledger.dispatch(task), CandidateCanary())
    result = ledger.finish()
    (group,) = result.groups
    assert type(group) is outcomes._GroupReleased
    cloned_group = outcomes._GroupReleased(
        group.group,
        tuple((datum_id, CandidateCanary()) for datum_id, _candidate in group.outputs),
    )
    malformed = replace(result, invocation=outcomes._InvocationCompleted((cloned_group,)))

    rejected: Any = graph_runtime._AccountingGraphExecution(plan, malformed, ())._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert CandidateCanary.calls == 0
    assert "CANDIDATE-EQUALITY-CANARY" not in capsys.readouterr().out


class _ComparisonCanary:
    calls = 0

    def __eq__(self, _other: object) -> bool:
        type(self).calls += 1
        print("MALFORMED-COMPARISON-CANARY")
        raise RuntimeError("MALFORMED-COMPARISON-CANARY")

    def __ne__(self, _other: object) -> bool:
        type(self).calls += 1
        print("MALFORMED-COMPARISON-CANARY")
        raise RuntimeError("MALFORMED-COMPARISON-CANARY")

    def __hash__(self) -> int:
        type(self).calls += 1
        print("MALFORMED-COMPARISON-CANARY")
        return 8675309

    def __bool__(self) -> bool:
        type(self).calls += 1
        print("MALFORMED-COMPARISON-CANARY")
        raise RuntimeError("MALFORMED-COMPARISON-CANARY")


def _assert_no_comparison_canary(capsys: pytest.CaptureFixture[str]) -> None:
    captured = capsys.readouterr()
    assert _ComparisonCanary.calls == 0
    assert "MALFORMED-COMPARISON-CANARY" not in captured.out
    assert "MALFORMED-COMPARISON-CANARY" not in captured.err


def test_phase10_accounting_task_correlation_never_invokes_malformed_equality_canary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")
    ledger_module = importlib.import_module("anonymizer.engine.execution.accounting_ledger")
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan("TASK-COMPARISON-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    result = ledger.finish()
    malformed = replace(
        result,
        tasks=(
            outcomes._TaskFailed(
                cast(Any, _ComparisonCanary()),
                ledger_module._causes(outcomes._CauseCode.KNOWN_FAILURE),
            ),
        ),
    )
    _ComparisonCanary.calls = 0
    capsys.readouterr()

    rejected: Any = graph_runtime._AccountingGraphExecution(plan, malformed, ())._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    _assert_no_comparison_canary(capsys)


def test_phase10_accounting_state_keys_never_invoke_malformed_hash_or_equality_canary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("STATE-KEY-CANARY"))
    ledger.open()
    malformed_key = _ComparisonCanary()
    ledger._states = cast(Any, {malformed_key: object()})
    _ComparisonCanary.calls = 0
    capsys.readouterr()

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    _assert_no_comparison_canary(capsys)


@pytest.mark.parametrize("field", ("terminal", "attempt"))
def test_phase10_phase8_correlation_never_invokes_malformed_equality_canary(
    field: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    stage = plan.stages[0]
    assert ledger.dispatch(stage)
    assert ledger.fail(stage, phase8._Phase8Reason.BACKEND_FAILURE)
    target = ledger._terminals[stage] if field == "terminal" else ledger._attempts[stage]
    object.__setattr__(target, "stage", cast(Any, _ComparisonCanary()))
    _ComparisonCanary.calls = 0
    capsys.readouterr()

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    _assert_no_comparison_canary(capsys)


def test_phase10_cleanup_proof_never_invokes_malformed_equality_canary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    identity = object()
    receipt = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        identity,
    )
    proof: Any = receipt._proof
    object.__setattr__(proof, "snapshot", (*proof.snapshot[:-1], _ComparisonCanary()))
    _ComparisonCanary.calls = 0
    capsys.readouterr()

    rejected: Any = receipt._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    _assert_no_comparison_canary(capsys)


@pytest.mark.parametrize("field", ("_phase10_release_state", "_opened", "_cleanup_failed"))
def test_phase10_accounting_scalars_never_invoke_malformed_comparison_canary(
    field: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("SCALAR-CANARY"))
    ledger.open()
    setattr(ledger, field, _ComparisonCanary())
    _ComparisonCanary.calls = 0
    capsys.readouterr()

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    _assert_no_comparison_canary(capsys)


def test_phase10_phase8_retired_flag_never_invokes_malformed_truthiness_canary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    ledger._retired = cast(Any, _ComparisonCanary())
    _ComparisonCanary.calls = 0
    capsys.readouterr()

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    _assert_no_comparison_canary(capsys)
