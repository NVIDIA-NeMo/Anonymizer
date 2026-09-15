# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.execution.accounting_admission import _compile_accounting_plan
from anonymizer.engine.execution.accounting_ledger import _AccountingLedger
from anonymizer.engine.execution.accounting_plan import _AccountingLimits, _AccountingPlan
from anonymizer.engine.execution.graph import _DatumId, _TextDatum, _trivial_graph
from tests.engine.execution.phase10_reference_model import (
    CAPTURE_BOUNDARIES,
    COUNT_BUCKETS,
    LIFECYCLE_STATES,
    LIMITS,
    MUTATION_CLASSES,
    OPERATIONS,
    REASONS,
    SUBJECTS,
    TERMINALS,
    ReferenceCase,
    ReferenceDiagnostic,
    canonical_corpus_bytes,
    case_by_name,
    finite_reference_cases,
    reduce_reference,
    reference_manifest,
)


class _Identity:
    pass


def _module() -> Any:
    return importlib.import_module("anonymizer.engine.execution.phase10_inspection")


def _contract() -> Any:
    module = importlib.import_module("anonymizer.engine.execution.phase10_contract")
    contract = module._load_phase10_contract()
    assert module._is_admitted_phase10_contract(contract)
    return contract


def _accounting_plan(*texts: str) -> _AccountingPlan:
    graph = _trivial_graph(tuple(_TextDatum(_DatumId(f"datum-{index}"), text) for index, text in enumerate(texts)))
    result = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=len(texts), max_datum_bytes=256, max_graph_bytes=1_024, max_stages=1),
    )
    assert isinstance(result, _AccountingPlan)
    return result


def _production_encoding(case: ReferenceCase) -> str:
    module = _module()
    reference = reduce_reference(case)
    assert reference.decision == "view"
    provenance = module._Phase10Provenance(
        module._Phase10InspectionSchemaVersion.V1,
        module._Phase10ContractVersion.V1,
        module._Phase10ViewKind(case.operation),
        module._Phase10SubjectKind(case.subject_kind),
        module._Phase10SemanticProfile(case.semantic_profile_version),
        module._Phase10ImplementationProfile.PANDAS_RUNTIME_V1,
        module._Phase10CaptureBoundary(case.capture_boundary),
        module._Phase10LifecycleState(case.lifecycle_state),
    )
    if case.operation == "explain":
        aggregates = tuple(
            module._Phase10Aggregate(
                module._Phase10AggregateDimension(dimension),
                module._Phase10CountBucket(reference.count_bucket if index == 0 and reference.count_bucket else bucket),
            )
            for index, (dimension, bucket) in enumerate(case.aggregates)
        )
        view = module._Phase10ExplainView(
            provenance,
            module._Phase10Route(case.route),
            tuple(module._Phase10Capability(value) for value in case.capabilities),
            tuple(module._Phase10DeclaredLimit(module._Phase10LimitName(name), value) for name, value in LIMITS),
            aggregates,
            module._Phase10ReasonCategory(case.rejection_category) if case.rejection_category else None,
        )
    elif case.operation == "inspect":
        stage_positions = {stage: index for index, stage in enumerate(case.stages)}
        terminals = sorted(case.terminals, key=lambda item: (stage_positions[item[0]], TERMINALS.index(item[1])))
        snapshot = module._Phase10Snapshot(
            tuple(
                module._Phase10StageSummary(
                    module._Phase10Stage(stage),
                    module._Phase10LifecycleState(
                        case.stage_lifecycle_states[index] if case.stage_lifecycle_states else case.lifecycle_state
                    ),
                    module._Phase10CountBucket(case.stage_task_buckets[index] if case.stage_task_buckets else "1"),
                )
                for index, stage in enumerate(case.stages)
            ),
            tuple(
                module._Phase10TerminalSummary(
                    module._Phase10Stage(stage),
                    module._Phase10TerminalState(terminal),
                    module._Phase10CountBucket(bucket),
                )
                for stage, terminal, bucket in terminals
            ),
            module._Phase10ReconciliationState(case.reconciliation_state),
            module._Phase10CleanupState(case.cleanup_state),
            module._Phase10ReleaseState(case.release_state),
        )
        view = module._Phase10InspectView(provenance, snapshot)
    else:
        diagnostics = case.diagnostics
        if case.subject_kind == "admission_rejection_receipt" and not diagnostics:
            diagnostics = (
                ReferenceDiagnostic(
                    "admission_terminal",
                    "admission",
                    "rejected",
                    case.rejection_category or "admission_rejected",
                    reconciliation_state="not_entered",
                ),
            )
        stage_positions = {stage: index for index, stage in enumerate(case.stages)}
        diagnostics = tuple(
            sorted(
                diagnostics,
                key=lambda item: (
                    CAPTURE_BOUNDARIES.index(item.boundary),
                    stage_positions.get(item.stage, len(stage_positions)),
                    TERMINALS.index(item.terminal_state),
                    REASONS.index(item.reason_category),
                    COUNT_BUCKETS.index(item.impact_count_bucket),
                ),
            )
        )
        view = module._Phase10DiagnoseView(
            provenance,
            tuple(_production_diagnostic(module, item) for item in diagnostics),
        )
    encoding = module._encode_phase10_view(view)
    assert type(encoding) is module._Phase10CanonicalEncoding
    return encoding.value.decode("utf-8")


def _production_diagnostic(module: Any, item: ReferenceDiagnostic) -> Any:
    return module._Phase10Diagnostic(
        module._Phase10CaptureBoundary(item.boundary),
        module._Phase10Stage(item.stage),
        module._Phase10TerminalState(item.terminal_state),
        module._Phase10ReasonCategory(item.reason_category),
        module._Phase10CountBucket(item.impact_count_bucket),
        module._Phase10ReconciliationState(item.reconciliation_state),
        module._Phase10CleanupState(item.cleanup_state),
    )


@pytest.mark.parametrize(
    "case",
    [case for case in finite_reference_cases() if reduce_reference(case).decision == "view"],
    ids=lambda case: case.name,
)
def test_every_admitted_reference_payload_matches_production_canonical_serializer(case: ReferenceCase) -> None:
    expected = reduce_reference(case)

    assert _production_encoding(case) == expected.canonical_json


def _real_plan_subject_registry() -> tuple[tuple[str, object, tuple[str, ...]], ...]:
    accounting_admission = importlib.import_module("anonymizer.engine.execution.accounting_admission")
    context_admission = importlib.import_module("anonymizer.engine.execution.context_admission")
    phase6 = importlib.import_module("anonymizer.engine.execution.phase6_plan")
    phase7 = importlib.import_module("anonymizer.engine.execution.phase7_admission")
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_admission")

    context_tests = importlib.import_module("tests.engine.execution.test_context_admission")
    context_contract, context_capability = context_tests._contract_and_capability()
    context_plan = context_tests._compile_context_plan(
        context_tests._context_graph(),
        accounting_limits=context_tests._ACCOUNTING_LIMITS,
        contract=context_contract,
        capability=context_capability,
    )

    phase6_tests = importlib.import_module("tests.engine.execution.test_phase6_runtime")
    phase6_plan = phase6_tests._plan(phase6_tests._context_graph())

    phase7_tests = importlib.import_module("tests.engine.execution.test_phase7_admission")
    phase7_source, _backend, phase7_execution = phase7_tests._qualified_phase6(
        ("Alice", "555-0100"),
        (("target-0",), ("target-1",)),
        {},
    )
    phase7_plan = phase7_tests._compile_phase7(
        phase7_source,
        phase7_execution,
        phase7_source.coherence_scopes,
    )

    graph = _trivial_graph((_TextDatum(_DatumId("phase8-owner"), "phase8-private"),))
    graph_module = importlib.import_module("anonymizer.engine.execution.graph")
    graph = replace(graph, rewrite_groups=(graph_module._RewriteGroup((graph.datums[0].id,)),))
    phase8_plan = phase8._compile_phase8_plan(graph, max_repairs=0)

    accepted = (
        ("accounting-plan", _accounting_plan("accounting-private"), ("explain",)),
        ("context-plan", context_plan, ("explain",)),
        ("phase6-plan", phase6_plan, ("explain",)),
        ("phase7-plan", phase7_plan, ("explain",)),
        ("phase8-plan", phase8_plan, ("explain",)),
    )
    rejected = (
        (
            "accounting-rejection",
            accounting_admission._AccountingRejected(accounting_admission._AccountingAdmissionCode.TOO_MANY_DATUMS),
            ("explain", "diagnose"),
        ),
        (
            "context-rejection",
            context_admission._ContextRejected(context_admission._ContextAdmissionCode.MALFORMED_GRAPH),
            ("explain", "diagnose"),
        ),
        (
            "phase6-rejection",
            phase6._Phase6Rejected(phase6._Phase6PlanRejectionCode.INVALID_PROFILE),
            ("explain", "diagnose"),
        ),
        (
            "phase7-rejection",
            phase7._Phase7Rejected(phase7._Phase7AdmissionCode.INVALID_INPUT),
            ("explain", "diagnose"),
        ),
        (
            "phase8-rejection",
            phase8._Phase8Rejected(phase8._Phase8AdmissionCode.INVALID_INPUT),
            ("explain", "diagnose"),
        ),
    )
    return (*accepted, *rejected)


def _real_capture_registry() -> tuple[tuple[str, object], ...]:
    module = _module()
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("ledger-private"))
    ledger.open()
    (task,) = ledger.ready_tasks()
    dispatch = ledger.dispatch(task)
    ledger.accept_failure(dispatch)
    accounting_capture = ledger._phase10_snapshot()

    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    graph_plan = _accounting_plan("graph-private")
    graph_ledger: _AccountingLedger[str] = _AccountingLedger(graph_plan)
    graph_ledger.open()
    (graph_task,) = graph_ledger.ready_tasks()
    graph_ledger.mark_task_succeeded(graph_task, "graph-candidate-private")
    graph_capture = graph_runtime._AccountingGraphExecution(graph_plan, graph_ledger.finish(), ())._phase10_snapshot()

    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    operation_plan = phase8._compile_group_operation_plan(0, 3)
    assert operation_plan is not None
    operation_ledger = phase8._Phase8OperationLedger(operation_plan)
    operation_ledger.fail(phase8._Phase8Stage.analyze(), phase8._Phase8Reason.BACKEND_FAILURE)
    operation_capture = operation_ledger._phase10_snapshot()

    cleanup = importlib.import_module("anonymizer.engine.execution.phase8_cleanup")
    cleanup_identity = object()
    pre_cleanup = cleanup._issue_phase8_cleanup_receipt(
        cleanup._Phase8CleanupPhase.PRE_REDUCTION,
        cleanup._Phase8CleanupComponent.RUNTIME,
        cleanup._Phase8CleanupStatus.VERIFIED,
        cleanup_identity,
    )
    post_cleanup = cleanup._issue_phase8_cleanup_receipt(
        cleanup._Phase8CleanupPhase.POST_REDUCTION,
        cleanup._Phase8CleanupComponent.RUNTIME,
        cleanup._Phase8CleanupStatus.VERIFIED,
        cleanup_identity,
        retained_candidate_cell_count=1,
    )
    lifecycle_capture = phase8._Phase8LifecycleExecution(
        ((object(), "lifecycle-private"),),
        ("succeeded",),
        False,
        pre_cleanup,
        post_cleanup,
    )._phase10_snapshot()
    cleanup_capture = post_cleanup._phase10_snapshot()

    captures = (
        ("accounting-ledger", accounting_capture),
        ("accounting-graph-execution", graph_capture),
        ("phase8-operation-ledger", operation_capture),
        ("phase8-lifecycle", lifecycle_capture),
        ("phase8-cleanup", cleanup_capture),
    )
    assert all(type(capture) is module._Phase10OwnerCapture for _name, capture in captures)
    return captures


def _owner_case(
    name: str,
    *,
    operation: str,
    subject_kind: str,
    capture_boundary: str,
    lifecycle_state: str,
    stages: tuple[str, ...],
    stage_lifecycle_states: tuple[str, ...],
    stage_task_buckets: tuple[str, ...],
    terminals: tuple[tuple[str, str, str], ...],
    diagnostics: tuple[ReferenceDiagnostic, ...],
    reconciliation_state: str,
    cleanup_state: str,
    release_state: str,
    semantic_profile_version: str,
    route: str,
    capabilities: tuple[str, ...],
    aggregates: tuple[tuple[str, str], ...],
    rejection_category: str | None,
) -> ReferenceCase:
    return ReferenceCase(
        name=name,
        operation=operation,
        grant_state="valid",
        subject_kind=subject_kind,
        capture_boundary=capture_boundary,
        lifecycle_state=lifecycle_state,
        stages=stages,
        stage_lifecycle_states=stage_lifecycle_states,
        stage_task_buckets=stage_task_buckets,
        terminals=terminals,
        diagnostics=diagnostics,
        reconciliation_state=reconciliation_state,
        cleanup_state=cleanup_state,
        release_state=release_state,
        semantic_profile_version=semantic_profile_version,
        route=route,
        capabilities=capabilities,
        aggregates=aggregates,
        rejection_category=rejection_category,
        builder_working_bytes=4_096,
        reasons_per_diagnostic=1,
        bucket_value=None,
        byte_value=None,
        payload_witness=None,
        canonical_limit_delta=None,
        unsafe_value=False,
    )


def _plan_case(
    name: str,
    *,
    subject_kind: str,
    lifecycle_state: str,
    profile: str,
    route: str,
    capabilities: tuple[str, ...],
    aggregates: tuple[tuple[str, str], ...],
    rejection_category: str | None,
) -> ReferenceCase:
    return _owner_case(
        name,
        operation="explain",
        subject_kind=subject_kind,
        capture_boundary="admission_terminal",
        lifecycle_state=lifecycle_state,
        stages=(),
        stage_lifecycle_states=(),
        stage_task_buckets=(),
        terminals=(),
        diagnostics=(),
        reconciliation_state="reconciled",
        cleanup_state="not_entered",
        release_state="not_entered",
        semantic_profile_version=profile,
        route=route,
        capabilities=capabilities,
        aggregates=aggregates,
        rejection_category=rejection_category,
    )


def _admission_diagnosis_case(name: str, *, profile: str, reason: str) -> ReferenceCase:
    return _owner_case(
        name,
        operation="diagnose",
        subject_kind="admission_rejection_receipt",
        capture_boundary="admission_terminal",
        lifecycle_state="rejected",
        stages=(),
        stage_lifecycle_states=(),
        stage_task_buckets=(),
        terminals=(),
        diagnostics=(),
        reconciliation_state="reconciled",
        cleanup_state="not_entered",
        release_state="not_entered",
        semantic_profile_version=profile,
        route="mixed",
        capabilities=("grouped_rewrite",),
        aggregates=(("datums", "1"),),
        rejection_category=reason,
    )


def _capture_case(
    name: str,
    *,
    operation: str,
    subject_kind: str,
    capture_boundary: str,
    lifecycle_state: str,
    stages: tuple[str, ...],
    stage_lifecycle_states: tuple[str, ...],
    stage_task_buckets: tuple[str, ...],
    terminals: tuple[tuple[str, str, str], ...],
    diagnostics: tuple[ReferenceDiagnostic, ...],
    reconciliation_state: str,
    cleanup_state: str,
    release_state: str,
    profile: str,
) -> ReferenceCase:
    return _owner_case(
        name,
        operation=operation,
        subject_kind=subject_kind,
        capture_boundary=capture_boundary,
        lifecycle_state=lifecycle_state,
        stages=stages,
        stage_lifecycle_states=stage_lifecycle_states,
        stage_task_buckets=stage_task_buckets,
        terminals=terminals,
        diagnostics=diagnostics,
        reconciliation_state=reconciliation_state,
        cleanup_state=cleanup_state,
        release_state=release_state,
        semantic_profile_version=profile,
        route="mixed",
        capabilities=("grouped_rewrite",),
        aggregates=(("datums", "1"),),
        rejection_category=None,
    )


_ACCOUNTING_DIAGNOSTIC = ReferenceDiagnostic(
    "terminal_evidence_accepted",
    "transform",
    "failed",
    "backend_failed",
    "1",
    "reconciled",
    "not_entered",
)

_REAL_OWNER_EXPECTATIONS = {
    "accounting-plan:explain": _plan_case(
        "owner-accounting-plan-explain",
        subject_kind="admitted_plan",
        lifecycle_state="terminal",
        profile="target-context-v1",
        route="ndd",
        capabilities=("terminal_accounting",),
        aggregates=(("datums", "1"), ("relationships", "0"), ("stages", "1"), ("tasks", "1"), ("groups", "1")),
        rejection_category=None,
    ),
    "context-plan:explain": _plan_case(
        "owner-context-plan-explain",
        subject_kind="admitted_plan",
        lifecycle_state="terminal",
        profile="target-context-v1",
        route="ndd",
        capabilities=("terminal_accounting", "bounded_context"),
        aggregates=(
            ("datums", "2-4"),
            ("relationships", "2-4"),
            ("stages", "1"),
            ("tasks", "2-4"),
            ("groups", "2-4"),
        ),
        rejection_category=None,
    ),
    "phase6-plan:explain": _plan_case(
        "owner-phase6-plan-explain",
        subject_kind="admitted_plan",
        lifecycle_state="terminal",
        profile="phase6-redact-graph/v1",
        route="mixed",
        capabilities=("terminal_accounting", "bounded_context", "anchored_mentions"),
        aggregates=(
            ("datums", "2-4"),
            ("relationships", "5-16"),
            ("stages", "5-16"),
            ("tasks", "5-16"),
            ("groups", "2-4"),
        ),
        rejection_category=None,
    ),
    "phase7-plan:explain": _plan_case(
        "owner-phase7-plan-explain",
        subject_kind="admitted_plan",
        lifecycle_state="terminal",
        profile="anonymizer-phase7-stable-substitute/v1",
        route="mixed",
        capabilities=("terminal_accounting", "bounded_context", "anchored_mentions", "stable_substitute"),
        aggregates=(
            ("datums", "2-4"),
            ("relationships", "2-4"),
            ("stages", "5-16"),
            ("tasks", "5-16"),
            ("groups", "2-4"),
        ),
        rejection_category=None,
    ),
    "phase8-plan:explain": _plan_case(
        "owner-phase8-plan-explain",
        subject_kind="admitted_plan",
        lifecycle_state="terminal",
        profile="anonymizer-phase8-grouped-rewrite/v1",
        route="ndd",
        capabilities=(
            "terminal_accounting",
            "bounded_context",
            "anchored_mentions",
            "stable_substitute",
            "grouped_rewrite",
        ),
        aggregates=(("datums", "1"), ("groups", "1"), ("operations", "2-4"), ("repairs", "0")),
        rejection_category=None,
    ),
    "accounting-rejection:explain": _plan_case(
        "owner-accounting-rejection-explain",
        subject_kind="admission_rejection_receipt",
        lifecycle_state="rejected",
        profile="target-context-v1",
        route="rejected",
        capabilities=(),
        aggregates=(),
        rejection_category="limit_exceeded",
    ),
    "accounting-rejection:diagnose": _admission_diagnosis_case(
        "owner-accounting-rejection-diagnose",
        profile="target-context-v1",
        reason="limit_exceeded",
    ),
    "context-rejection:explain": _plan_case(
        "owner-context-rejection-explain",
        subject_kind="admission_rejection_receipt",
        lifecycle_state="rejected",
        profile="target-context-v1",
        route="rejected",
        capabilities=(),
        aggregates=(),
        rejection_category="admission_rejected",
    ),
    "context-rejection:diagnose": _admission_diagnosis_case(
        "owner-context-rejection-diagnose",
        profile="target-context-v1",
        reason="admission_rejected",
    ),
    "phase6-rejection:explain": _plan_case(
        "owner-phase6-rejection-explain",
        subject_kind="admission_rejection_receipt",
        lifecycle_state="rejected",
        profile="phase6-redact-graph/v1",
        route="rejected",
        capabilities=(),
        aggregates=(),
        rejection_category="capability_mismatch",
    ),
    "phase6-rejection:diagnose": _admission_diagnosis_case(
        "owner-phase6-rejection-diagnose",
        profile="phase6-redact-graph/v1",
        reason="capability_mismatch",
    ),
    "phase7-rejection:explain": _plan_case(
        "owner-phase7-rejection-explain",
        subject_kind="admission_rejection_receipt",
        lifecycle_state="rejected",
        profile="anonymizer-phase7-stable-substitute/v1",
        route="rejected",
        capabilities=(),
        aggregates=(),
        rejection_category="admission_rejected",
    ),
    "phase7-rejection:diagnose": _admission_diagnosis_case(
        "owner-phase7-rejection-diagnose",
        profile="anonymizer-phase7-stable-substitute/v1",
        reason="admission_rejected",
    ),
    "phase8-rejection:explain": _plan_case(
        "owner-phase8-rejection-explain",
        subject_kind="admission_rejection_receipt",
        lifecycle_state="rejected",
        profile="anonymizer-phase8-grouped-rewrite/v1",
        route="rejected",
        capabilities=(),
        aggregates=(),
        rejection_category="admission_rejected",
    ),
    "phase8-rejection:diagnose": _admission_diagnosis_case(
        "owner-phase8-rejection-diagnose",
        profile="anonymizer-phase8-grouped-rewrite/v1",
        reason="admission_rejected",
    ),
    "accounting-ledger:inspect": _capture_case(
        "owner-accounting-ledger-inspect",
        operation="inspect",
        subject_kind="invocation_snapshot",
        capture_boundary="terminal_evidence_accepted",
        lifecycle_state="terminal",
        stages=("transform",),
        stage_lifecycle_states=("terminal",),
        stage_task_buckets=("1",),
        terminals=(("transform", "failed", "1"),),
        diagnostics=(_ACCOUNTING_DIAGNOSTIC,),
        reconciliation_state="reconciled",
        cleanup_state="not_entered",
        release_state="not_entered",
        profile="target-context-v1",
    ),
    "accounting-ledger:diagnose": _capture_case(
        "owner-accounting-ledger-diagnose",
        operation="diagnose",
        subject_kind="invocation_snapshot",
        capture_boundary="terminal_evidence_accepted",
        lifecycle_state="terminal",
        stages=("transform",),
        stage_lifecycle_states=("terminal",),
        stage_task_buckets=("1",),
        terminals=(("transform", "failed", "1"),),
        diagnostics=(_ACCOUNTING_DIAGNOSTIC,),
        reconciliation_state="reconciled",
        cleanup_state="not_entered",
        release_state="not_entered",
        profile="target-context-v1",
    ),
    "accounting-graph-execution:inspect": _capture_case(
        "owner-accounting-graph-execution-inspect",
        operation="inspect",
        subject_kind="terminal_receipt",
        capture_boundary="invocation_closed",
        lifecycle_state="closed",
        stages=("transform",),
        stage_lifecycle_states=("terminal",),
        stage_task_buckets=("1",),
        terminals=(("transform", "succeeded", "1"),),
        diagnostics=(),
        reconciliation_state="reconciled",
        cleanup_state="not_entered",
        release_state="released",
        profile="target-context-v1",
    ),
    "accounting-graph-execution:diagnose": _capture_case(
        "owner-accounting-graph-execution-diagnose",
        operation="diagnose",
        subject_kind="terminal_receipt",
        capture_boundary="invocation_closed",
        lifecycle_state="closed",
        stages=("transform",),
        stage_lifecycle_states=("terminal",),
        stage_task_buckets=("1",),
        terminals=(("transform", "succeeded", "1"),),
        diagnostics=(),
        reconciliation_state="reconciled",
        cleanup_state="not_entered",
        release_state="released",
        profile="target-context-v1",
    ),
    "phase8-operation-ledger:inspect": _capture_case(
        "owner-phase8-operation-ledger-inspect",
        operation="inspect",
        subject_kind="invocation_snapshot",
        capture_boundary="pre_dispatch",
        lifecycle_state="pre_dispatch",
        stages=("validate", "analyze", "rewrite", "evaluate"),
        stage_lifecycle_states=("pre_dispatch", "pre_dispatch", "pre_dispatch", "pre_dispatch"),
        stage_task_buckets=("1", "1", "1", "1"),
        terminals=(),
        diagnostics=(),
        reconciliation_state="not_entered",
        cleanup_state="not_entered",
        release_state="not_entered",
        profile="anonymizer-phase8-grouped-rewrite/v1",
    ),
    "phase8-operation-ledger:diagnose": _capture_case(
        "owner-phase8-operation-ledger-diagnose",
        operation="diagnose",
        subject_kind="invocation_snapshot",
        capture_boundary="pre_dispatch",
        lifecycle_state="pre_dispatch",
        stages=("validate", "analyze", "rewrite", "evaluate"),
        stage_lifecycle_states=("pre_dispatch", "pre_dispatch", "pre_dispatch", "pre_dispatch"),
        stage_task_buckets=("1", "1", "1", "1"),
        terminals=(),
        diagnostics=(),
        reconciliation_state="not_entered",
        cleanup_state="not_entered",
        release_state="not_entered",
        profile="anonymizer-phase8-grouped-rewrite/v1",
    ),
    "phase8-lifecycle:inspect": _capture_case(
        "owner-phase8-lifecycle-inspect",
        operation="inspect",
        subject_kind="terminal_receipt",
        capture_boundary="invocation_closed",
        lifecycle_state="closed",
        stages=("rewrite", "cleanup"),
        stage_lifecycle_states=("terminal", "cleanup_terminal"),
        stage_task_buckets=("1", "1"),
        terminals=(("rewrite", "succeeded", "1"), ("cleanup", "succeeded", "1")),
        diagnostics=(),
        reconciliation_state="reconciled",
        cleanup_state="verified",
        release_state="released",
        profile="anonymizer-phase8-grouped-rewrite/v1",
    ),
    "phase8-lifecycle:diagnose": _capture_case(
        "owner-phase8-lifecycle-diagnose",
        operation="diagnose",
        subject_kind="terminal_receipt",
        capture_boundary="invocation_closed",
        lifecycle_state="closed",
        stages=("rewrite", "cleanup"),
        stage_lifecycle_states=("terminal", "cleanup_terminal"),
        stage_task_buckets=("1", "1"),
        terminals=(("rewrite", "succeeded", "1"), ("cleanup", "succeeded", "1")),
        diagnostics=(),
        reconciliation_state="reconciled",
        cleanup_state="verified",
        release_state="released",
        profile="anonymizer-phase8-grouped-rewrite/v1",
    ),
    "phase8-cleanup:inspect": _capture_case(
        "owner-phase8-cleanup-inspect",
        operation="inspect",
        subject_kind="cleanup_receipt",
        capture_boundary="post_reduction_cleanup_terminal",
        lifecycle_state="cleanup_terminal",
        stages=("cleanup",),
        stage_lifecycle_states=("cleanup_terminal",),
        stage_task_buckets=("1",),
        terminals=(("cleanup", "succeeded", "1"),),
        diagnostics=(),
        reconciliation_state="reconciled",
        cleanup_state="verified",
        release_state="not_entered",
        profile="anonymizer-phase8-grouped-rewrite/v1",
    ),
    "phase8-cleanup:diagnose": _capture_case(
        "owner-phase8-cleanup-diagnose",
        operation="diagnose",
        subject_kind="cleanup_receipt",
        capture_boundary="post_reduction_cleanup_terminal",
        lifecycle_state="cleanup_terminal",
        stages=("cleanup",),
        stage_lifecycle_states=("cleanup_terminal",),
        stage_task_buckets=("1",),
        terminals=(("cleanup", "succeeded", "1"),),
        diagnostics=(),
        reconciliation_state="reconciled",
        cleanup_state="verified",
        release_state="not_entered",
        profile="anonymizer-phase8-grouped-rewrite/v1",
    ),
}


def _assert_owner_view_matches_reference(expected_case: ReferenceCase, view: object) -> bytes:
    module = _module()
    expected = reduce_reference(expected_case)
    encoded = module._encode_phase10_view(view)

    assert expected.decision == "view"
    assert type(encoded) is module._Phase10CanonicalEncoding
    assert encoded.value.decode("utf-8") == expected.canonical_json
    return encoded.value


def test_real_owner_reference_rejects_valid_but_wrong_production_field() -> None:
    module = _module()
    owner = _Identity()
    subject = _accounting_plan("accounting-private")
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.EXPLAIN)
    view = module._explain_phase10(owner, subject, grant)
    assert type(view) is module._Phase10ExplainView
    wrong_view = module._Phase10ExplainView(
        view.provenance,
        module._Phase10Route.LOCAL,
        view.required_capabilities,
        view.declared_limits,
        view.relationship_buckets,
        view.rejection_category,
    )

    with pytest.raises(AssertionError):
        _assert_owner_view_matches_reference(_REAL_OWNER_EXPECTATIONS["accounting-plan:explain"], wrong_view)


def test_named_real_owner_registry_matches_reference_decisions_provenance_rows_limits_and_bytes() -> None:
    module = _module()
    owner = _Identity()
    witnessed: set[str] = set()

    for name, subject, operations in _real_plan_subject_registry():
        for operation_name in operations:
            operation = module._Phase10Operation(operation_name)
            grant = module._issue_phase10_inspection_grant(owner, subject, operation)
            view = getattr(module, f"_{operation_name}_phase10")(owner, subject, grant)
            assert not isinstance(view, module._Phase10InspectionRejected)
            key = f"{name}:{operation_name}"
            _assert_owner_view_matches_reference(_REAL_OWNER_EXPECTATIONS[key], view)
            if type(view) is module._Phase10ExplainView:
                assert tuple((item.name.value, item.value) for item in view.declared_limits) == LIMITS
            witnessed.add(key)

    for name, capture in _real_capture_registry():
        for operation_name in ("inspect", "diagnose"):
            operation = module._Phase10Operation(operation_name)
            grant = module._issue_phase10_inspection_grant(owner, capture, operation)
            result = getattr(module, f"_{operation_name}_phase10")(owner, capture, grant)
            key = f"{name}:{operation_name}"
            expected_case = _REAL_OWNER_EXPECTATIONS[key]
            expected = reduce_reference(expected_case)
            if type(result) is module._Phase10InspectionRejected:
                assert result.code.value == expected.rejection_code
                assert expected.canonical_json is None
            else:
                _assert_owner_view_matches_reference(expected_case, result)
            witnessed.add(key)

    assert witnessed == set(_REAL_OWNER_EXPECTATIONS)
    assert witnessed == {
        "accounting-plan:explain",
        "context-plan:explain",
        "phase6-plan:explain",
        "phase7-plan:explain",
        "phase8-plan:explain",
        "accounting-rejection:explain",
        "accounting-rejection:diagnose",
        "context-rejection:explain",
        "context-rejection:diagnose",
        "phase6-rejection:explain",
        "phase6-rejection:diagnose",
        "phase7-rejection:explain",
        "phase7-rejection:diagnose",
        "phase8-rejection:explain",
        "phase8-rejection:diagnose",
        "accounting-ledger:inspect",
        "accounting-ledger:diagnose",
        "accounting-graph-execution:inspect",
        "accounting-graph-execution:diagnose",
        "phase8-operation-ledger:inspect",
        "phase8-operation-ledger:diagnose",
        "phase8-lifecycle:inspect",
        "phase8-lifecycle:diagnose",
        "phase8-cleanup:inspect",
        "phase8-cleanup:diagnose",
    }


def test_every_reference_case_is_bound_to_one_production_witness_class() -> None:
    classes = {case.name: _witness_class(case) for case in finite_reference_cases()}

    assert tuple(classes) == tuple(case.name for case in finite_reference_cases())
    assert set(classes.values()) == {
        "authorization",
        "bucket",
        "capture",
        "directed",
        "limit",
        "operation-subject",
        "reason-terminal",
        "subject-boundary",
    }


def _witness_class(case: ReferenceCase) -> str:
    if case.name.startswith("authorization-"):
        return "authorization"
    if case.name.startswith(("count-bucket-", "byte-bucket-")):
        return "bucket"
    if case.name.startswith("capture-"):
        return "capture"
    if case.name.startswith("subject-boundary-"):
        return "subject-boundary"
    if case.name.startswith("operation-subject-"):
        return "operation-subject"
    if case.name.startswith("limit-"):
        return "limit"
    if case.name.startswith(("reason-", "terminal-")):
        return "reason-terminal"
    return "directed"


def test_contract_resource_and_production_domains_match_the_independent_oracle() -> None:
    module = _module()
    contract = _contract()

    assert contract.limits == tuple(sorted(LIMITS))
    assert contract.count_buckets == COUNT_BUCKETS
    assert contract.capture_boundaries == CAPTURE_BOUNDARIES
    assert contract.reason_categories == REASONS
    assert tuple(item.value for item in module._Phase10Operation) == OPERATIONS
    assert tuple(item.value for item in module._Phase10SubjectKind) == SUBJECTS
    assert tuple(item.value for item in module._Phase10CaptureBoundary) == CAPTURE_BOUNDARIES
    assert tuple(item.value for item in module._Phase10LifecycleState) == LIFECYCLE_STATES
    assert tuple(item.value for item in module._Phase10ReasonCategory) == REASONS
    assert tuple(item.value for item in module._Phase10CountBucket) == COUNT_BUCKETS
    assert module._unmapped_phase10_reason_values() == ()


def test_known_owner_reason_mapping_matches_the_oracle() -> None:
    module = _module()
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")

    assert (
        module._map_phase10_reason(outcomes._CauseCode.KNOWN_FAILURE).value
        == case_by_name("reason-backend_failed").diagnostics[0].reason_category
    )


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("subject", SUBJECTS)
def test_production_operation_subject_matrix_matches_the_oracle(operation: str, subject: str) -> None:
    module = _module()
    expected = reduce_reference(case_by_name(f"operation-subject-{operation}-{subject}")).decision == "view"

    assert (
        module._valid_view_subject(module._Phase10ViewKind(operation), module._Phase10SubjectKind(subject)) is expected
    )


@pytest.mark.parametrize("boundary", CAPTURE_BOUNDARIES[1:])
@pytest.mark.parametrize("lifecycle", LIFECYCLE_STATES)
def test_production_capture_matrix_matches_the_oracle(boundary: str, lifecycle: str) -> None:
    module = _module()
    expected = reduce_reference(case_by_name(f"capture-{boundary}-{lifecycle}")).decision == "view"
    capture = _unchecked_capture(module, "invocation_snapshot", boundary, lifecycle)

    assert module._valid_owner_capture(capture) is expected


def _unchecked_capture(module: Any, subject: str, boundary: str, lifecycle: str) -> Any:
    stage = module._Phase10Stage.REWRITE
    terminal = module._Phase10TerminalState.FAILED
    snapshot = module._Phase10Snapshot(
        (module._Phase10StageSummary(stage, module._Phase10LifecycleState(lifecycle), module._Phase10CountBucket.ONE),),
        (module._Phase10TerminalSummary(stage, terminal, module._Phase10CountBucket.ONE),),
        module._Phase10ReconciliationState.RECONCILED,
        module._Phase10CleanupState.NOT_ENTERED,
        module._Phase10ReleaseState.NOT_ENTERED,
    )
    value = object.__new__(module._Phase10OwnerCapture)
    object.__setattr__(value, "subject_kind", module._Phase10SubjectKind(subject))
    object.__setattr__(value, "semantic_profile_version", module._Phase10SemanticProfile.GROUPED_REWRITE_V1)
    object.__setattr__(value, "capture_boundary", module._Phase10CaptureBoundary(boundary))
    object.__setattr__(value, "capture_lifecycle_state", module._Phase10LifecycleState(lifecycle))
    object.__setattr__(value, "snapshot", snapshot)
    object.__setattr__(value, "diagnostics", ())
    return value


@pytest.mark.parametrize("grant_state", ["forged", "reused", "expired", "wrong_operation", "wrong_subject"])
def test_production_authorization_failures_match_cause_free_oracle_denial(grant_state: str) -> None:
    module = _module()
    owner = _Identity()
    subject = _Identity()
    operation = module._Phase10Operation.INSPECT
    grant: object
    if grant_state == "forged":
        grant = object()
    else:
        grant = module._issue_phase10_inspection_grant(owner, subject, operation)
        if grant_state == "reused":
            assert module._consume_phase10_inspection_grant(grant, owner, subject, operation)
        elif grant_state == "expired":
            module._revoke_phase10_inspection_grant(grant)
        elif grant_state == "wrong_operation":
            operation = module._Phase10Operation.EXPLAIN
        elif grant_state == "wrong_subject":
            subject = _Identity()

    observed = module._consume_phase10_inspection_grant(grant, owner, subject, operation)
    expected = reduce_reference(case_by_name(f"authorization-{grant_state}"))

    assert not observed
    assert expected.rejection_code == module._Phase10RejectionCode.DENIED.value
    assert not expected.subject_accessed


@pytest.mark.parametrize("value", [-1, 0, 1, 2, 4, 5, 16, 17, 64, 65, 65_537])
def test_production_count_buckets_match_the_oracle(value: int) -> None:
    module = _module()
    expected = reduce_reference(replace(case_by_name("count-bucket-0"), bucket_value=value)).count_bucket
    observed = module._phase10_count_bucket(value)

    assert (observed.value if observed is not None else None) == expected


def test_closed_reason_order_is_independent_of_task_failure_arrival() -> None:
    ledger_module = importlib.import_module("anonymizer.engine.execution.accounting_ledger")
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")

    def encoded(codes: tuple[Any, Any]) -> bytes:
        module = _module()
        ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("first", "second"))
        ledger.open()
        tasks = ledger.ready_tasks()
        for task, code in zip(tasks, codes, strict=True):
            ledger._states[task] = outcomes._TaskFailed(task, ledger_module._causes(code))
        capture = ledger._phase10_snapshot()
        owner = _Identity()
        grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.DIAGNOSE)
        view = module._diagnose_phase10(owner, capture, grant)
        result = module._encode_phase10_view(view)
        assert type(result) is module._Phase10CanonicalEncoding
        return result.value

    verification = outcomes._CauseCode.VERIFICATION_FAILED
    backend = outcomes._CauseCode.KNOWN_FAILURE

    assert encoded((verification, backend)) == encoded((backend, verification))


def test_reference_and_production_serialization_are_permutation_stable() -> None:
    first = ReferenceDiagnostic("terminal_evidence_accepted", "rewrite", "failed", "verification_failed")
    second = replace(first, reason_category="backend_failed")
    base = replace(case_by_name("reason-backend_failed"), diagnostics=(first, second))
    reverse = replace(base, name="reversed", diagnostics=(second, first))

    assert reduce_reference(base).canonical_json == reduce_reference(reverse).canonical_json
    assert _production_encoding(base) == _production_encoding(reverse)


def _encode_real_subject(subject: object, operation_name: str) -> bytes:
    module = _module()
    owner = _Identity()
    operation = module._Phase10Operation(operation_name)
    grant = module._issue_phase10_inspection_grant(owner, subject, operation)
    view = getattr(module, f"_{operation_name}_phase10")(owner, subject, grant)
    assert type(view) is not module._Phase10InspectionRejected
    encoded = module._encode_phase10_view(view)
    assert type(encoded) is module._Phase10CanonicalEncoding
    return encoded.value


def test_real_owner_declaration_mapping_group_cleanup_and_publication_permutations_are_stable() -> None:
    first_plan = _accounting_plan("first-private", "second-private")
    second_plan = _accounting_plan("second-private", "first-private")
    assert _encode_real_subject(first_plan, "explain") == _encode_real_subject(second_plan, "explain")

    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    operation_plan = phase8._compile_group_operation_plan(1, 3)
    assert operation_plan is not None
    ordered = phase8._Phase8OperationLedger(operation_plan)
    assert ordered.succeed(operation_plan.stages[0])
    assert ordered.fail(operation_plan.stages[1], phase8._Phase8Reason.BACKEND_FAILURE)
    reversed_evidence = phase8._Phase8OperationLedger(operation_plan)
    reversed_evidence._terminals = dict(reversed(tuple(ordered._terminals.items())))
    reversed_evidence._attempts = dict(reversed(tuple(ordered._attempts.items())))
    reversed_evidence._dispatched = set(reversed(tuple(ordered._dispatched)))
    assert _encode_real_subject(ordered._phase10_snapshot(), "inspect") == _encode_real_subject(
        reversed_evidence._phase10_snapshot(),
        "inspect",
    )
    assert _encode_real_subject(ordered._phase10_snapshot(), "diagnose") == _encode_real_subject(
        reversed_evidence._phase10_snapshot(),
        "diagnose",
    )

    cleanup = importlib.import_module("anonymizer.engine.execution.phase8_cleanup")
    cleanup_identity = object()
    pre_cleanup = cleanup._issue_phase8_cleanup_receipt(
        cleanup._Phase8CleanupPhase.PRE_REDUCTION,
        cleanup._Phase8CleanupComponent.RUNTIME,
        cleanup._Phase8CleanupStatus.VERIFIED,
        cleanup_identity,
    )
    post_cleanup = cleanup._issue_phase8_cleanup_receipt(
        cleanup._Phase8CleanupPhase.POST_REDUCTION,
        cleanup._Phase8CleanupComponent.RUNTIME,
        cleanup._Phase8CleanupStatus.VERIFIED,
        cleanup_identity,
        retained_candidate_cell_count=2,
    )

    def lifecycle_capture(released: tuple[tuple[object, str], ...]) -> object:
        return phase8._Phase8LifecycleExecution(
            released,
            ("succeeded", "succeeded"),
            False,
            pre_cleanup,
            post_cleanup,
        )._phase10_snapshot()

    released = ((object(), "first-private"), (object(), "second-private"))
    assert _encode_real_subject(lifecycle_capture(released), "inspect") == _encode_real_subject(
        lifecycle_capture(tuple(reversed(released))),
        "inspect",
    )

    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")

    def publication_failure(order: tuple[int, int]) -> bytes:
        plan = _accounting_plan("first-private", "second-private")
        ledger: _AccountingLedger[str] = _AccountingLedger(plan)
        ledger.open()
        tasks = ledger.ready_tasks()
        for index in order:
            dispatch = ledger.dispatch(tasks[index])
            assert dispatch is not None
            ledger.accept_success(dispatch, f"protected-{index}")
        result = ledger.finish(group_release_predicate=lambda _outputs: False)
        capture = graph_runtime._AccountingGraphExecution(plan, result, ())._phase10_snapshot()
        return _encode_real_subject(capture, "diagnose")

    assert publication_failure((0, 1)) == publication_failure((1, 0))


def _real_owner_observations() -> dict[str, dict[str, object]]:
    module = _module()
    owner = _Identity()
    observations: dict[str, dict[str, object]] = {}
    for name, subject, operations in _real_plan_subject_registry():
        for operation_name in operations:
            operation = module._Phase10Operation(operation_name)
            grant = module._issue_phase10_inspection_grant(owner, subject, operation)
            result = getattr(module, f"_{operation_name}_phase10")(owner, subject, grant)
            key = f"{name}:{operation_name}"
            if type(result) is module._Phase10InspectionRejected:
                observations[key] = {"decision": "rejected", "rejection_code": result.code.value}
            else:
                encoded = module._encode_phase10_view(result)
                assert type(encoded) is module._Phase10CanonicalEncoding
                observations[key] = {
                    "canonical_json": encoded.value.decode("utf-8"),
                    "decision": "view",
                    "rejection_code": None,
                }
    for name, capture in _real_capture_registry():
        for operation_name in ("inspect", "diagnose"):
            operation = module._Phase10Operation(operation_name)
            grant = module._issue_phase10_inspection_grant(owner, capture, operation)
            result = getattr(module, f"_{operation_name}_phase10")(owner, capture, grant)
            key = f"{name}:{operation_name}"
            if type(result) is module._Phase10InspectionRejected:
                observations[key] = {"decision": "rejected", "rejection_code": result.code.value}
            else:
                encoded = module._encode_phase10_view(result)
                assert type(encoded) is module._Phase10CanonicalEncoding
                observations[key] = {
                    "canonical_json": encoded.value.decode("utf-8"),
                    "decision": "view",
                    "rejection_code": None,
                }
    return observations


def hashseed_probe() -> str:
    reference_digest = hashlib.sha256(canonical_corpus_bytes()).hexdigest()
    owner_observations = json.dumps(
        _real_owner_observations(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return json.dumps(
        {
            "manifest": reference_manifest(),
            "owner_observations_sha256": hashlib.sha256(owner_observations).hexdigest(),
            "reference_digest": reference_digest,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_reference_manifest_and_production_bytes_are_hash_seed_independent() -> None:
    repository = os.fspath(importlib.import_module("pathlib").Path(__file__).parents[3])
    command = [
        sys.executable,
        "-c",
        "from tests.engine.execution.test_phase10_reference_conformance import hashseed_probe; print(hashseed_probe())",
    ]
    outputs = []
    for seed in ("0", "1", "42", "4294967295"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        outputs.append(completed.stdout)

    assert len(set(outputs)) == 1


def test_canonical_json_has_exact_schema_types_and_no_trailing_newline() -> None:
    allowed = {
        "explain": {
            "declared_limits",
            "inspection_schema_version",
            "provenance",
            "rejection_category",
            "relationship_buckets",
            "required_capabilities",
            "route",
            "subject_kind",
            "view_kind",
        },
        "inspect": {
            "cleanup_state",
            "inspection_schema_version",
            "provenance",
            "reconciliation_state",
            "release_state",
            "stage_summaries",
            "subject_kind",
            "terminal_summaries",
            "view_kind",
        },
        "diagnose": {"diagnostics", "inspection_schema_version", "provenance", "subject_kind", "view_kind"},
    }
    for operation in OPERATIONS:
        case = next(
            case
            for case in finite_reference_cases()
            if case.operation == operation and reduce_reference(case).decision == "view"
        )
        encoded = _production_encoding(case)
        payload = json.loads(encoded, parse_float=lambda _value: pytest.fail("float admitted"))

        assert set(payload) == allowed[operation]
        assert not encoded.endswith("\n")
        assert "NaN" not in encoded and "Infinity" not in encoded


def test_manifest_accounts_for_every_contract_mutation_class() -> None:
    assert reference_manifest()["mutation_class_count"] == len(MUTATION_CLASSES) == 20
