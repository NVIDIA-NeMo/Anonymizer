# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from dataclasses import replace
from typing import cast

import pytest

from anonymizer.engine.execution.accounting_ledger import _AccountingLedger
from anonymizer.engine.execution.accounting_plan import (
    _AccountingPlan,
    _admit_accounting_plan,
    _AtomicGroupKey,
    _CompiledAtomicGroup,
    _CompiledDependency,
    _DatumTaskSubject,
    _StageId,
    _TaskKey,
)
from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _DatumDependency,
    _DatumId,
    _DatumPurpose,
    _ProtectionGraph,
    _RewriteGroup,
    _TextDatum,
)
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.phase7_application import _AppliedDatum
from anonymizer.engine.execution.phase7_runtime import (
    _Phase7CleanupAttestation,
    _Phase7Execution,
    _Phase7Phase4Evidence,
)
from anonymizer.engine.execution.phase8_admission import (
    _compile_phase8_accounting_plan,
    _compile_phase8_plan,
    _is_admitted_phase8_accounting_plan,
)
from anonymizer.engine.execution.phase8_cleanup import (
    _is_phase8_cleanup_receipt,
    _issue_phase8_cleanup_receipt,
    _Phase8CleanupComponent,
    _Phase8CleanupPhase,
    _Phase8CleanupStatus,
)
from anonymizer.engine.execution.phase8_ndd_backend import _Phase8NddBackend, _Phase8Operation
from anonymizer.engine.execution.phase8_runtime import (
    _Phase8FaultKind,
    _Phase8GroupOutcome,
    _Phase8OperationFault,
    _Phase8Reason,
    _run_group_operation,
)
from anonymizer.engine.execution.phase8_service import (
    _is_sealed_candidate_cell,
    _ManagedGroupOperation,
    _Phase8CleanupRuntime,
    _run_accounted_successor,
    _seal_candidate_cell,
)
from anonymizer.engine.ndd.adapter import NddAdapter


def test_phase8_accounting_extension_conserves_prefix_and_uses_compiler_group_identity() -> None:
    first, second = _DatumId("first"), _DatumId("second")
    phase6_final = _StageId("phase6-final")
    phase7_apply = _StageId("phase7-apply")
    phase7: _AccountingPlan = _admit_accounting_plan(
        (_TextDatum(first, "one", _DatumPurpose.TARGET), _TextDatum(second, "two", _DatumPurpose.TARGET)),
        (phase6_final, phase7_apply),
        (
            _TaskKey(phase6_final, _DatumTaskSubject(first)),
            _TaskKey(phase6_final, _DatumTaskSubject(second)),
            _TaskKey(phase7_apply, _DatumTaskSubject(first)),
            _TaskKey(phase7_apply, _DatumTaskSubject(second)),
        ),
        (),
        (),
        (first, second),
    )
    graph = _ProtectionGraph(
        datums=phase7.datums,
        links=(),
        context_scopes=(),
        coherence_scopes=(),
        atomic_groups=(_AtomicGroup((first, second)),),
        rewrite_groups=(_RewriteGroup((first, second)),),
    )
    grouped = _compile_phase8_plan(graph, max_repairs=0)
    assert not hasattr(grouped, "code")

    composed = _compile_phase8_accounting_plan(phase7, grouped)

    assert _is_admitted_phase8_accounting_plan(composed)
    assert composed.accounting.tasks[: len(phase7.tasks)] == phase7.tasks
    assert composed.group_tasks[0].subject is composed.groups[0].accounting_subject
    assert {
        edge.prerequisite for edge in composed.accounting.task_predecessors if edge.dependent == composed.group_tasks[0]
    } == {
        _TaskKey(phase6_final, _DatumTaskSubject(first)),
        _TaskKey(phase6_final, _DatumTaskSubject(second)),
    }
    assert {
        edge.dependent for edge in composed.accounting.task_predecessors if edge.prerequisite == composed.group_tasks[0]
    } == set(composed.qualification_tasks)


def test_phase8_accounting_extension_rejects_a_reconstructed_group_plan() -> None:
    datum = _DatumId("only")
    phase6_final = _StageId("phase6-final")
    phase7_apply = _StageId("phase7-apply")
    phase7 = _admit_accounting_plan(
        (_TextDatum(datum, "one", _DatumPurpose.TARGET),),
        (phase6_final, phase7_apply),
        (
            _TaskKey(phase6_final, _DatumTaskSubject(datum)),
            _TaskKey(phase7_apply, _DatumTaskSubject(datum)),
        ),
        (),
        (),
        (datum,),
    )
    graph = _ProtectionGraph(
        datums=phase7.datums,
        links=(),
        context_scopes=(),
        coherence_scopes=(),
        atomic_groups=(_AtomicGroup((datum,)),),
        rewrite_groups=(_RewriteGroup((datum,)),),
    )
    grouped = _compile_phase8_plan(graph, max_repairs=0)
    assert not hasattr(grouped, "code")

    with pytest.raises(TypeError, match="admitted Phase 7 accounting"):
        _compile_phase8_accounting_plan(phase7, replace(grouped, groups=()))


def test_phase8_ndd_backend_retirement_is_identity_bound_and_irreversible() -> None:
    backend = _Phase8NddBackend(cast(NddAdapter, object()), cast(_CompiledInvocation, object()))
    identity = object()

    receipt = backend.retire_phase8(identity)

    assert _is_phase8_cleanup_receipt(
        receipt,
        identity=identity,
        phase=_Phase8CleanupPhase.PRE_REDUCTION,
        component=_Phase8CleanupComponent.BACKEND,
    )
    assert receipt is not None and receipt.status is _Phase8CleanupStatus.VERIFIED
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(receipt)
    assert backend._adapter is None
    assert backend._invocation is None
    assert backend._invocation_token is None
    assert backend.retire_phase8(identity) is None
    assert backend.run_operation(_Phase8Operation.ANALYZE, {}).failure_kind == "invocation_inconsistent"


def test_phase8_candidate_cells_reject_reconstruction_or_value_substitution() -> None:
    datum = _DatumId("only")
    cell = _seal_candidate_cell(datum, "candidate")

    assert _is_sealed_candidate_cell(cell, datum)
    assert not _is_sealed_candidate_cell(replace(cell, _value="other"), datum)
    assert not _is_sealed_candidate_cell(type(cell)(datum, "candidate"), datum)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(cell)


class _CleanupBackend:
    def __init__(self, status: _Phase8CleanupStatus | None = _Phase8CleanupStatus.VERIFIED) -> None:
        self.status = status

    def retire_phase8(self, identity: object) -> object:
        if self.status is None:
            return object()
        return _issue_phase8_cleanup_receipt(
            _Phase8CleanupPhase.PRE_REDUCTION,
            _Phase8CleanupComponent.BACKEND,
            self.status,
            identity,
            active_workframe_reference_count=int(self.status is _Phase8CleanupStatus.FAILED),
        )


def test_phase8_pre_cleanup_rejects_duplicate_operation_evidence() -> None:
    cleanup = _Phase8CleanupRuntime()
    operation = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.OPERATION,
        _Phase8CleanupStatus.VERIFIED,
        cleanup.identity,
    )
    backend = _CleanupBackend().retire_phase8(cleanup.identity)

    receipt = cleanup.attest_pre(
        (operation, operation),
        backend,
        expected_operation_count=2,
        retained_candidate_cell_count=0,
    )

    assert receipt.status is _Phase8CleanupStatus.UNCONFIRMED


def test_phase8_cleanup_unconfirmed_precedes_a_separate_trusted_failure() -> None:
    cleanup = _Phase8CleanupRuntime()
    failed = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.OPERATION,
        _Phase8CleanupStatus.FAILED,
        cleanup.identity,
        active_operation_count=1,
    )

    receipt = cleanup.attest_pre(
        (failed, object()),
        _CleanupBackend().retire_phase8(cleanup.identity),
        expected_operation_count=2,
        retained_candidate_cell_count=0,
    )

    assert receipt.status is _Phase8CleanupStatus.UNCONFIRMED


def _accounted_fixture(
    *,
    rewrite_groups: tuple[tuple[_DatumId, ...], ...],
    atomic_groups: tuple[tuple[_DatumId, ...], ...],
    dependencies: tuple[tuple[_DatumId, _DatumId], ...] = (),
):
    members = tuple(dict.fromkeys(member for group in rewrite_groups for member in group))
    datums = tuple(_TextDatum(member, f"baseline-{member.value}", _DatumPurpose.TARGET) for member in members)
    phase6_final = _StageId("phase6-final")
    phase7_apply = _StageId("phase7-apply")
    tasks = tuple(
        _TaskKey(stage, _DatumTaskSubject(member)) for stage in (phase6_final, phase7_apply) for member in members
    )
    phase7_plan = _admit_accounting_plan(
        datums,
        (phase6_final, phase7_apply),
        tasks,
        tuple(_CompiledDependency(*edge) for edge in dependencies),
        tuple(_CompiledAtomicGroup(_AtomicGroupKey(), group) for group in atomic_groups),
        members,
    )
    ledger: _AccountingLedger[object] = _AccountingLedger(phase7_plan)
    ledger.open()
    for task in tasks:
        ledger.mark_task_succeeded(task, object())
    accounting = ledger.finish()
    cleanup = _Phase7CleanupAttestation("phase7-cleanup-attestation/v1", True, 0, 0, True, 0, False)
    released = tuple(_AppliedDatum(datum.id, datum.text, True) for datum in datums)
    phase7 = _Phase7Execution((), cleanup, _Phase7Phase4Evidence((), accounting, cleanup, False), released)
    graph = _ProtectionGraph(
        datums,
        (),
        (),
        (),
        tuple(_AtomicGroup(group) for group in atomic_groups),
        tuple(_DatumDependency(*edge) for edge in dependencies),
        tuple(_RewriteGroup(group) for group in rewrite_groups),
    )
    phase8 = _compile_phase8_plan(graph, max_repairs=0)
    assert not hasattr(phase8, "code")
    return phase7, _compile_phase8_accounting_plan(phase7_plan, phase8), phase8


def _managed_operation(manifest, calls: list[tuple[_DatumId, ...]], fault: _Phase8OperationFault | None = None):
    retained: list[_Phase8GroupOutcome] = []

    def run(members: tuple[object, ...], baselines: dict[object, str]):
        calls.append(cast(tuple[_DatumId, ...], members))

        def analyze() -> tuple[bool, bool]:
            if fault is not None:
                raise fault
            return True, True

        outcome = _run_group_operation(
            members,
            baselines,
            analyze=analyze,
            rewrite=lambda values: values,
            evaluate=lambda _values: pytest.fail("zero route must not evaluate"),
            repair=lambda values, _round: values,
            max_repairs=0,
            operation_plan=manifest.operations,
        )
        retained.append(outcome)
        return outcome

    def discard() -> None:
        if retained:
            outcome = retained.pop()
            if outcome.revisions is not None:
                outcome.revisions.clear()
            outcome.ledger.discard()

    return _ManagedGroupOperation(run, discard)


@pytest.mark.parametrize(
    "fault",
    (
        _Phase8OperationFault(_Phase8FaultKind.FAILED, _Phase8Reason.BACKEND_FAILURE),
        _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.CANDIDATE_RECONCILIATION),
    ),
)
def test_phase8_phase4_releases_a_disconnected_group_after_local_failure(fault: _Phase8OperationFault) -> None:
    first, second = _DatumId("first"), _DatumId("second")
    phase7, composition, phase8 = _accounted_fixture(
        rewrite_groups=((first,), (second,)),
        atomic_groups=((first,), (second,)),
    )
    calls: list[tuple[_DatumId, ...]] = []
    operations = (
        _managed_operation(phase8.groups[0], calls, fault),
        _managed_operation(phase8.groups[1], calls),
    )

    execution = _run_accounted_successor(composition, phase7, operations, _CleanupBackend())

    assert calls == [(first,), (second,)]
    assert execution.terminal_group_states == (
        "failed" if fault.kind is _Phase8FaultKind.FAILED else "inconsistent",
        "succeeded",
    )
    assert execution.released == ((second, "baseline-second"),)
    assert not execution.global_embargo
    assert execution.cleanup_verified


@pytest.mark.parametrize(
    ("fault", "expected"),
    (
        (_Phase8OperationFault(_Phase8FaultKind.CANCELLED, _Phase8Reason.CANCELLATION, trusted_stop=True), "cancelled"),
        (_Phase8OperationFault(_Phase8FaultKind.LOST, _Phase8Reason.TRANSPORT_LOST), "lost"),
        (
            _Phase8OperationFault(
                _Phase8FaultKind.INCONSISTENT,
                _Phase8Reason.INVOCATION_INCONSISTENT,
                invocation_global=True,
            ),
            "inconsistent",
        ),
    ),
)
def test_phase8_global_terminals_stop_later_dispatch_and_embargo_release(
    fault: _Phase8OperationFault, expected: str
) -> None:
    first, second = _DatumId("first"), _DatumId("second")
    phase7, composition, phase8 = _accounted_fixture(
        rewrite_groups=((first,), (second,)),
        atomic_groups=((first,), (second,)),
    )
    calls: list[tuple[_DatumId, ...]] = []
    operations = (
        _managed_operation(phase8.groups[0], calls, fault),
        _managed_operation(phase8.groups[1], calls),
    )

    execution = _run_accounted_successor(composition, phase7, operations, _CleanupBackend())

    assert calls == [(first,)]
    assert execution.terminal_group_states == (expected, "blocked")
    assert execution.released == ()
    assert execution.global_embargo
    assert execution.cleanup_verified


@pytest.mark.parametrize(
    ("backend_status", "expected"),
    (
        (_Phase8CleanupStatus.FAILED, _Phase8CleanupStatus.FAILED),
        (None, _Phase8CleanupStatus.UNCONFIRMED),
    ),
)
def test_phase8_pre_reduction_cleanup_failure_or_unconfirmed_evidence_embargoes_release(
    backend_status: _Phase8CleanupStatus | None,
    expected: _Phase8CleanupStatus,
) -> None:
    datum = _DatumId("only")
    phase7, composition, phase8 = _accounted_fixture(
        rewrite_groups=((datum,),),
        atomic_groups=((datum,),),
    )
    operation = _managed_operation(phase8.groups[0], [])

    execution = _run_accounted_successor(composition, phase7, (operation,), _CleanupBackend(backend_status))

    assert execution.released == ()
    assert execution.global_embargo
    assert not execution.cleanup_verified
    assert execution.pre_reduction_cleanup is not None
    assert execution.pre_reduction_cleanup.status is expected


@pytest.mark.parametrize("status", (_Phase8CleanupStatus.FAILED, _Phase8CleanupStatus.UNCONFIRMED))
def test_phase8_post_reduction_cleanup_failure_clears_release_cells(
    monkeypatch: pytest.MonkeyPatch,
    status: _Phase8CleanupStatus,
) -> None:
    datum = _DatumId("only")
    phase7, composition, phase8 = _accounted_fixture(
        rewrite_groups=((datum,),),
        atomic_groups=((datum,),),
    )

    def attest_post(self: _Phase8CleanupRuntime, **counts: int):
        return _issue_phase8_cleanup_receipt(
            _Phase8CleanupPhase.POST_REDUCTION,
            _Phase8CleanupComponent.RUNTIME,
            status,
            self.identity,
            provisional_revision_reference_count=1,
            retained_candidate_cell_count=counts["released_cell_count"],
        )

    monkeypatch.setattr(_Phase8CleanupRuntime, "attest_post", attest_post)
    execution = _run_accounted_successor(
        composition,
        phase7,
        (_managed_operation(phase8.groups[0], []),),
        _CleanupBackend(),
    )

    assert execution.released == ()
    assert execution.global_embargo
    assert not execution.cleanup_verified
    assert execution.post_reduction_cleanup is not None
    assert execution.post_reduction_cleanup.status is status
