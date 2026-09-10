# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import pickle
from dataclasses import FrozenInstanceError, replace
from typing import Any

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


def test_phase10_explain_projects_admitted_plan_and_rejection_without_content() -> None:
    module = _module()
    assert callable(getattr(module, "_explain_phase10", None)), "Tier 2 explain builder is missing"
    owner = object()
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
    owner = object()
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
    owner = object()
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
    owner = object()

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
    owner = object()
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
    owner = object()
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
    owner = object()
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
    owner = object()
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
    owner = object()
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
