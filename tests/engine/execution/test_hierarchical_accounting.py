# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from inspect import signature
from itertools import count, product
from json import dumps, loads
from pathlib import Path
from threading import Barrier
from typing import Callable, cast

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.models import ModelSelection
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.execution.accounting_admission import (
    _AccountingAdmissionCode,
    _AccountingRejected,
    _compile_accounting_plan,
)
from anonymizer.engine.execution.accounting_evidence import _AttemptId, _Dispatch, _RowToken, _SuccessRecord
from anonymizer.engine.execution.accounting_ledger import (
    _AccountingLedger,
    _EvidenceAcceptance,
    _LedgerClosedError,
    _LedgerStateError,
)
from anonymizer.engine.execution.accounting_outcomes import (
    _AccountingResult,
    _CauseCode,
    _DatumFailed,
    _DatumQualified,
    _DependencySatisfied,
    _GroupReleased,
    _GroupWithheld,
    _InvocationCancelled,
    _InvocationCompleted,
    _InvocationFailed,
    _InvocationInconsistent,
    _InvocationLost,
    _StageFailed,
    _StageSucceeded,
    _TaskBlocked,
    _TaskCancelled,
    _TaskFailed,
    _TaskInconsistent,
    _TaskLost,
    _TaskSucceeded,
)
from anonymizer.engine.execution.accounting_plan import _AccountingLimits, _AccountingPlan
from anonymizer.engine.execution.accounting_release import _qualify_release
from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _CoherenceScope,
    _ContextScope,
    _DatumDependency,
    _DatumId,
    _DatumLink,
    _ProtectionGraph,
    _RelationKind,
    _TextDatum,
    _trivial_graph,
)
from anonymizer.engine.execution.graph_runtime import (
    _AccountingGraphAdmissionError,
    _AccountingGraphExecution,
    _AccountingGraphRuntime,
)
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.pandas_runtime import _PandasExecutionResult
from anonymizer.engine.execution.protection_service import _RedactCandidate, _RedactProtectionService
from anonymizer.engine.ndd.adapter import FailedRecord, _FailedRowEvidence
from anonymizer.engine.private_row_verification import (
    PRIVATE_CORRELATION_COLUMN,
    _InvocationRowVerifier,
    _TerminalOutcome,
)
from tests.engine.execution.phase4_reference_model import (
    ReferenceCancellationRequest,
    ReferenceContradiction,
    ReferenceCorruptEvidence,
    ReferenceCorruption,
    ReferenceDeclaration,
    ReferenceDispatch,
    ReferenceFailure,
    ReferenceHierarchyResult,
    ReferenceInvocationOutcome,
    ReferenceObservation,
    ReferenceResultConstructionFailure,
    ReferenceStopAcknowledgement,
    ReferenceSuccess,
    ReferenceTaskOutcome,
    ReferenceTransportLoss,
    acyclic_dependencies,
    flat_partitions,
    reduce_observations,
    reduce_reference,
    streaming_conformance_cases,
)

_SINGLETON = ReferenceDeclaration(("datum-a",), (), (("datum-a",),))
_LIMITS = _AccountingLimits(max_datums=4, max_datum_bytes=64, max_graph_bytes=128)
_CONFORMANCE_MANIFEST = loads(Path(__file__).with_name("phase4_conformance_manifest.json").read_text())


def _crash_backend_worker(marker: str) -> None:
    """Run the test backend effect in a worker that dies before replying."""
    Path(marker).write_text("started")
    os._exit(17)


def _compiled(graph: _ProtectionGraph) -> _AccountingPlan:
    result = _compile_accounting_plan(graph, limits=_LIMITS)
    assert isinstance(result, _AccountingPlan)
    return result


def test_one_shot_ledger_reconciles_complete_singleton_invocation() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    identities = iter(("invocation", "attempt", "row-token"))
    ledger = _AccountingLedger(plan, identity_factory=lambda: next(identities))

    ledger.open()
    ready = ledger.ready_tasks()
    assert ready == plan.tasks
    dispatch = ledger.dispatch(ready[0])
    ledger.accept_success(dispatch, "protected-a")
    result = ledger.finish()

    assert type(result.invocation).__name__ == "_InvocationCompleted"
    assert tuple(type(outcome).__name__ for outcome in result.tasks) == ("_TaskSucceeded",)
    assert tuple(type(outcome).__name__ for outcome in result.datums) == ("_DatumQualified",)
    assert tuple(type(outcome).__name__ for outcome in result.groups) == ("_GroupReleased",)
    group = result.groups[0]
    assert isinstance(group, _GroupReleased)
    assert group.outputs == ((plan.datums[0].id, "protected-a"),)
    with pytest.raises(_LedgerClosedError):
        ledger.finish()


def test_synthetic_multistage_plan_pipelines_per_datum_without_global_stage_dispatch_barrier() -> None:
    graph = _graph("a", "b")
    graph = replace(graph, dependencies=(_dependency(graph, "a", "b"),))
    plan = _compile_accounting_plan(graph, limits=_LIMITS, stages=("detect", "protect"))
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()

    observed_frontiers: list[tuple[tuple[str, str], ...]] = []
    while ready := ledger.ready_tasks():
        observed_frontiers.append(tuple((task.stage.value, task.datum_id.value) for task in ready))
        for task in ready:
            ledger.accept_success(ledger.dispatch(task), f"{task.stage.value}-{task.datum_id.value}")
    result = ledger.finish()

    assert observed_frontiers == [
        (("detect", "a"),),
        (("protect", "a"),),
        (("detect", "b"),),
        (("protect", "b"),),
    ]
    assert len(result.tasks) == len(plan.tasks) == 4
    assert len({outcome.task for outcome in result.tasks}) == 4
    assert len(result.datums) == len(plan.datums) == 2
    assert len(result.dependencies) == len(plan.dependencies) == 1
    assert len(result.stages) == len(plan.stages) == 2
    assert len(result.groups) == len(plan.atomic_groups) == 2
    assert all(isinstance(stage, _StageSucceeded) for stage in result.stages)


def test_one_shot_ledger_allows_exactly_one_concurrent_open() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)

    ready = Barrier(3)

    def _open() -> str:
        ready.wait()
        try:
            ledger.open()
        except _LedgerStateError:
            return "rejected"
        return "opened"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(_open) for _ordinal in range(2))
        ready.wait()
        outcomes = tuple(future.result() for future in futures)

    assert sorted(outcomes) == ["opened", "rejected"]


def test_one_shot_ledger_allows_exactly_one_concurrent_finish() -> None:
    ledger = _ledger(_compiled(_graph("a")))
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])
    ledger.accept_success(dispatch, "protected")
    ready = Barrier(3)

    def _finish() -> str:
        ready.wait()
        try:
            ledger.finish()
        except _LedgerClosedError:
            return "rejected"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(_finish) for _ordinal in range(2))
        ready.wait()
        outcomes = tuple(future.result() for future in futures)

    assert sorted(outcomes) == ["published", "rejected"]


def test_private_plan_evidence_and_result_are_nonserializable_and_content_safe() -> None:
    canary = "private-canary@example.test"
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])
    ledger.accept_success(dispatch, canary)
    result = ledger.finish()

    for private_value in (plan, dispatch, result):
        with pytest.raises(TypeError, match="not serializable"):
            pickle.dumps(private_value)
        assert canary not in repr(private_value)


def _ledger(plan: _AccountingPlan) -> _AccountingLedger[str]:
    identities = count()
    return _AccountingLedger(plan, identity_factory=lambda: f"private-{next(identities)}")


def test_failure_blocks_dependents_without_dispatch_or_retry_and_isolates_independent_group() -> None:
    graph = _graph("a", "b", "c")
    graph = replace(graph, dependencies=(_dependency(graph, "a", "b"),))
    plan = _compile_accounting_plan(graph, limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    ready = ledger.ready_tasks()
    assert tuple(task.datum_id.value for task in ready) == ("a", "c")
    by_datum = {task.datum_id.value: task for task in ready}
    dispatch_a = ledger.dispatch(by_datum["a"])
    dispatch_c = ledger.dispatch(by_datum["c"])
    ledger.accept_failure(dispatch_a)
    ledger.accept_success(dispatch_c, "protected-c")

    assert ledger.ready_tasks() == ()
    with pytest.raises(_LedgerStateError):
        ledger.dispatch(by_datum["a"])
    result = ledger.finish()

    assert isinstance(result.invocation, _InvocationCompleted)
    assert any(isinstance(outcome, _TaskBlocked) and outcome.task.datum_id.value == "b" for outcome in result.tasks)
    released = tuple(outcome for outcome in result.groups if isinstance(outcome, _GroupReleased))
    assert tuple(output[0].value for outcome in released for output in outcome.outputs) == ("c",)


def test_late_atomic_peer_failure_withholds_already_succeeded_dependent_at_fixed_point() -> None:
    graph = _graph("a", "b", "c", "d", "e")
    graph = replace(
        graph,
        dependencies=(_dependency(graph, "a", "c"),),
        atomic_groups=(
            _group(graph, "a", "b"),
            _group(graph, "c", "d"),
            _group(graph, "e"),
        ),
    )
    plan = _compile_accounting_plan(graph, limits=replace(_LIMITS, max_datums=5, max_graph_bytes=256))
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    initial = {task.datum_id.value: task for task in ledger.ready_tasks()}
    ledger.accept_success(ledger.dispatch(initial["a"]), "protected-a")
    after_prerequisite = {task.datum_id.value: task for task in ledger.ready_tasks()}
    ledger.accept_success(ledger.dispatch(after_prerequisite["c"]), "protected-c")
    ledger.accept_failure(ledger.dispatch(after_prerequisite["b"]))
    ledger.accept_success(ledger.dispatch(after_prerequisite["d"]), "protected-d")
    ledger.accept_success(ledger.dispatch(after_prerequisite["e"]), "protected-e")

    result = ledger.finish()
    reference = reduce_observations(
        ReferenceDeclaration(
            ("a", "b", "c", "d", "e"),
            (("a", "c"),),
            (("a", "b"), ("c", "d"), ("e",)),
        ),
        (
            ReferenceDispatch(("protect", "a")),
            ReferenceSuccess(("protect", "a")),
            ReferenceDispatch(("protect", "c")),
            ReferenceSuccess(("protect", "c")),
            ReferenceDispatch(("protect", "b")),
            ReferenceFailure(("protect", "b")),
            ReferenceDispatch(("protect", "d")),
            ReferenceSuccess(("protect", "d")),
            ReferenceDispatch(("protect", "e")),
            ReferenceSuccess(("protect", "e")),
        ),
    )
    released_ids = tuple(
        datum_id.value
        for group in result.groups
        if isinstance(group, _GroupReleased)
        for datum_id, _candidate in group.outputs
    )

    assert any(isinstance(outcome, _TaskSucceeded) and outcome.task.datum_id.value == "c" for outcome in result.tasks)
    assert released_ids == ("e",)
    assert (
        tuple(
            ((outcome.task.stage.value, outcome.task.datum_id.value), ReferenceTaskOutcome.SUCCEEDED)
            if isinstance(outcome, _TaskSucceeded)
            else ((outcome.task.stage.value, outcome.task.datum_id.value), ReferenceTaskOutcome.FAILED)
            for outcome in result.tasks
            if isinstance(outcome, (_TaskSucceeded, _TaskFailed))
        )
        == reference.tasks
    )
    assert (
        tuple(
            (outcome.datum_id.value, ReferenceTaskOutcome.SUCCEEDED)
            if isinstance(outcome, _DatumQualified)
            else (outcome.datum_id.value, ReferenceTaskOutcome.FAILED)
            for outcome in result.datums
            if isinstance(outcome, (_DatumQualified, _DatumFailed))
        )
        == reference.datums
    )
    assert (
        tuple(
            ((outcome.dependency.prerequisite.value, outcome.dependency.dependent.value), True)
            for outcome in result.dependencies
            if isinstance(outcome, _DependencySatisfied)
        )
        == reference.dependencies
    )
    assert isinstance(result.stages[0], _StageFailed)
    assert reference.stages == (("protect", ReferenceTaskOutcome.FAILED),)
    assert reference.released_groups == frozenset((frozenset(("e",)),))
    assert reference.invocation is ReferenceInvocationOutcome.COMPLETED


def test_post_dispatch_cancellation_without_stop_evidence_is_lost() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])

    ledger.request_cancellation()
    result = ledger.finish()
    reference = reduce_observations(
        ReferenceDeclaration(("a",), (), (("a",),)),
        (
            ReferenceDispatch(("protect", "a")),
            ReferenceCancellationRequest(),
        ),
    )

    assert isinstance(result.invocation, _InvocationLost)
    assert isinstance(result.tasks[0], _TaskLost)
    assert isinstance(result.groups[0], _GroupWithheld)
    assert reference.invocation is ReferenceInvocationOutcome.LOST
    assert reference.tasks[0][1] is ReferenceTaskOutcome.LOST
    assert ledger.acknowledge_stop(dispatch) is _EvidenceAcceptance.REJECTED_STALE


def test_pre_dispatch_cancellation_closes_without_dispatch() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    assert ledger.ready_tasks() == plan.tasks

    ledger.request_cancellation()
    result = ledger.finish()

    assert isinstance(result.invocation, _InvocationCancelled)
    assert isinstance(result.tasks[0], _TaskCancelled)
    assert isinstance(result.groups[0], _GroupWithheld)


def test_cancellation_after_success_but_before_publication_embargoes_output() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])
    ledger.accept_success(dispatch, "protected")

    ledger.request_cancellation()
    result = ledger.finish()

    assert isinstance(result.invocation, _InvocationCancelled)
    assert isinstance(result.tasks[0], _TaskSucceeded)
    assert isinstance(result.groups[0], _GroupWithheld)


def test_terminal_success_precedes_late_stop_and_post_publication_cancellation() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])
    ledger.accept_success(dispatch, "protected")

    assert ledger.acknowledge_stop(dispatch) is _EvidenceAcceptance.REJECTED_STALE
    result = ledger.finish()
    ledger.request_cancellation()

    assert isinstance(result.invocation, _InvocationCompleted)
    assert isinstance(result.groups[0], _GroupReleased)


def test_trusted_stop_acknowledgement_before_late_success_is_cancelled() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])
    ledger.request_cancellation()

    assert ledger.acknowledge_stop(dispatch) is _EvidenceAcceptance.ACCEPTED
    assert ledger.accept_success(dispatch, "late") is _EvidenceAcceptance.REJECTED_STALE
    result = ledger.finish()
    reference = reduce_observations(
        ReferenceDeclaration(("a",), (), (("a",),)),
        (
            ReferenceDispatch(("protect", "a")),
            ReferenceCancellationRequest(),
            ReferenceStopAcknowledgement(("protect", "a")),
            ReferenceSuccess(("protect", "a")),
        ),
    )

    assert isinstance(result.invocation, _InvocationCancelled)
    assert isinstance(result.tasks[0], _TaskCancelled)
    assert isinstance(result.groups[0], _GroupWithheld)
    assert reference.invocation is ReferenceInvocationOutcome.CANCELLED
    assert reference.tasks[0][1] is ReferenceTaskOutcome.CANCELLED


def test_terminal_replay_is_idempotent_but_conflicting_late_evidence_is_stale() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])

    assert ledger.accept_success(dispatch, "protected") is _EvidenceAcceptance.ACCEPTED
    assert ledger.accept_success(dispatch, "protected") is _EvidenceAcceptance.IDEMPOTENT_STALE
    assert ledger.accept_failure(dispatch) is _EvidenceAcceptance.REJECTED_STALE
    result = ledger.finish()

    assert isinstance(result.invocation, _InvocationCompleted)
    assert isinstance(result.groups[0], _GroupReleased)


def test_result_construction_failure_closes_exhaustively_without_group_output() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])
    ledger.accept_success(dispatch, "protected")

    def _fail_release(_datum_id: _DatumId, _candidate: str) -> bool:
        raise RuntimeError("private-result-content")

    result = ledger.finish(datum_release_predicate=_fail_release)

    assert isinstance(result.invocation, _InvocationFailed)
    assert isinstance(result.groups[0], _GroupWithheld)
    assert "private-result-content" not in repr(result)
    with pytest.raises(_LedgerClosedError):
        ledger.finish()


def test_trusted_batch_missing_record_is_local_inconsistency() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])

    ledger.reconcile((dispatch,), (), trusted_run_record=True)
    result = ledger.finish()

    assert isinstance(result.invocation, _InvocationCompleted)
    assert isinstance(result.tasks[0], _TaskInconsistent)
    assert tuple(cause.code for cause in result.tasks[0].causes) == (_CauseCode.MISSING,)
    assert isinstance(result.groups[0], _GroupWithheld)


def test_foreign_row_token_is_invocation_global_inconsistency() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])
    foreign = replace(dispatch, row_token=_RowToken("foreign"))

    ledger.reconcile((dispatch,), (_SuccessRecord(foreign, "protected"),), trusted_run_record=True)
    result = ledger.finish()

    assert isinstance(result.invocation, _InvocationInconsistent)
    assert isinstance(result.tasks[0], _TaskInconsistent)
    assert tuple(cause.code for cause in result.tasks[0].causes) == (_CauseCode.FOREIGN,)
    assert isinstance(result.groups[0], _GroupWithheld)


def test_direct_foreign_evidence_before_terminal_acceptance_is_global_inconsistency() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])
    foreign = replace(dispatch, row_token=_RowToken("foreign"))

    assert ledger.accept_success(foreign, "protected") is _EvidenceAcceptance.REJECTED_STALE
    result = ledger.finish()

    assert isinstance(result.invocation, _InvocationInconsistent)
    assert isinstance(result.tasks[0], _TaskInconsistent)
    assert tuple(cause.code for cause in result.tasks[0].causes) == (_CauseCode.FOREIGN,)


def test_duplicate_expected_dispatch_identity_is_global_inconsistency() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])

    ledger.reconcile((dispatch, dispatch), (_SuccessRecord(dispatch, "protected"),), trusted_run_record=True)
    result = ledger.finish()

    assert isinstance(result.invocation, _InvocationInconsistent)
    assert isinstance(result.tasks[0], _TaskInconsistent)
    assert tuple(cause.code for cause in result.tasks[0].causes) == (_CauseCode.DUPLICATE,)
    assert isinstance(result.groups[0], _GroupWithheld)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda dispatch: replace(dispatch, attempt_id=_AttemptId("unknown"), row_token=_RowToken("unknown")),
            "unknown",
        ),
        (lambda dispatch: replace(dispatch, attempt_id=_AttemptId("stale")), "stale"),
    ],
)
def test_unknown_and_stale_attempt_evidence_keep_exact_causes(
    mutation: Callable[[_Dispatch], _Dispatch],
    expected: str,
) -> None:
    plan = _compiled(_graph("a"))
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])

    ledger.reconcile((dispatch,), (_SuccessRecord(mutation(dispatch), "candidate"),), trusted_run_record=True)
    result = ledger.finish()

    assert isinstance(result.tasks[0], _TaskInconsistent)
    assert tuple(cause.code.value for cause in result.tasks[0].causes) == (expected,)


def test_swapped_valid_dispatch_evidence_keeps_exact_cause() -> None:
    plan = _compiled(_graph("a", "b"))
    ledger = _ledger(plan)
    ledger.open()
    first, second = tuple(ledger.dispatch(task) for task in ledger.ready_tasks())
    swapped = replace(first, task=second.task, row_token=second.row_token)

    ledger.reconcile((first, second), (_SuccessRecord(swapped, "candidate"),), trusted_run_record=True)
    result = ledger.finish()

    assert all(isinstance(task, _TaskInconsistent) for task in result.tasks)
    for task in result.tasks:
        assert isinstance(task, _TaskInconsistent)
        assert tuple(cause.code.value for cause in task.causes) == ("swapped",)


def test_ledger_rejects_opaque_identity_collision_before_dispatch() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger: _AccountingLedger[str] = _AccountingLedger(plan, identity_factory=lambda: "collision")
    ledger.open()

    with pytest.raises(_LedgerStateError):
        ledger.dispatch(ledger.ready_tasks()[0])

    assert ledger.ready_tasks() == plan.tasks


def test_plan_mismatch_is_invocation_global_inconsistency() -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    ledger.dispatch(ledger.ready_tasks()[0])

    ledger.mark_inconsistent(_CauseCode.PLAN_MISMATCH)
    result = ledger.finish()

    assert isinstance(result.invocation, _InvocationInconsistent)
    assert isinstance(result.tasks[0], _TaskInconsistent)
    assert isinstance(result.groups[0], _GroupWithheld)


def test_row_verifier_binds_ledger_owned_tokens_without_regenerating_identity() -> None:
    assert "correlations" in signature(_InvocationRowVerifier).parameters
    frame = pd.DataFrame({COL_TEXT: ["first", "second"]})

    verifier = _InvocationRowVerifier(frame, correlations=("ledger-a", "ledger-b"))
    bound = verifier.bind(frame)

    assert tuple(bound[PRIVATE_CORRELATION_COLUMN]) == ("ledger-a", "ledger-b")


def test_accounting_runtime_executes_ready_frontiers_and_preserves_datum_identity(
    stub_slim_model_selection: ModelSelection,
) -> None:
    frames: list[tuple[str, ...]] = []

    class _Backend:
        def run(
            self,
            dataframe: pd.DataFrame,
            *,
            invocation: _CompiledInvocation,
            data_summary: str | None,
            preview_num_records: int | None,
            verifier: _InvocationRowVerifier,
        ) -> _PandasExecutionResult:
            del invocation, data_summary, preview_num_records
            frames.append(tuple(dataframe[COL_TEXT]))
            detected = dataframe.assign(final_entities=[{"entities": []} for _ in range(len(dataframe))])
            verifier.freeze_accepted_detections(detected)
            final = verifier.finish(detected)
            return _PandasExecutionResult(
                dataframe=final,
                failed_records=[],
                terminal_outcomes=verifier.take_terminal_outcomes(),
                result_row_tokens=verifier.take_result_order(),
            )

    graph = _graph("a", "b", "c")
    graph = replace(graph, dependencies=(_dependency(graph, "a", "b"),))
    plan = _compile_accounting_plan(graph, limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    runtime = _AccountingGraphRuntime(_Backend())

    execution = runtime.run(
        plan,
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=lambda datum, row: (datum.id.value, row[COL_TEXT]),
    )

    assert frames == [("synthetic a", "synthetic c"), ("synthetic b",)]
    assert isinstance(execution.accounting.invocation, _InvocationCompleted)
    assert tuple(
        output[1][0]
        for group in execution.accounting.groups
        if isinstance(group, _GroupReleased)
        for output in group.outputs
    ) == ("a", "b", "c")


def test_accounting_runtime_rejects_direct_or_tampered_plan_before_effects(
    stub_slim_model_selection: ModelSelection,
) -> None:
    calls = 0

    class _Backend:
        def run(self, *_args: object, **_kwargs: object) -> _PandasExecutionResult:
            nonlocal calls
            calls += 1
            raise AssertionError("backend must not run")

    compiled = _compiled(_graph("a"))
    direct = _AccountingPlan(
        compiled.datums,
        compiled.stages,
        compiled.tasks,
        compiled.dependencies,
        compiled.atomic_groups,
        compiled.topological_datums,
    )
    tampered = _compiled(_graph("a"))
    object.__setattr__(tampered, "topological_datums", ())
    nested_tampered = _compiled(_graph("a"))
    object.__setattr__(nested_tampered.datums[0], "text", "nested-tamper-canary")
    runtime = _AccountingGraphRuntime(_Backend())
    invocation = _CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection)

    for plan in (direct, tampered, nested_tampered):
        with pytest.raises(_AccountingGraphAdmissionError):
            runtime.run(
                plan,
                invocation=invocation,
                data_summary=None,
                preview_num_records=None,
                hydrate=lambda datum, row: (datum.id.value, row[COL_TEXT]),
            )

    assert calls == 0


def test_accounting_runtime_classifies_verifier_corruption_as_inconsistent(
    stub_slim_model_selection: ModelSelection,
) -> None:
    class _CorruptingBackend:
        def run(
            self,
            dataframe: pd.DataFrame,
            *,
            invocation: _CompiledInvocation,
            data_summary: str | None,
            preview_num_records: int | None,
            verifier: _InvocationRowVerifier,
        ) -> _PandasExecutionResult:
            del invocation, data_summary, preview_num_records
            corrupted = dataframe.assign(**{PRIVATE_CORRELATION_COLUMN: ["foreign-token"]})
            verifier.freeze_accepted_detections(corrupted.assign(**{COL_FINAL_ENTITIES: [{"entities": []}]}))
            raise AssertionError("verifier must reject foreign correlation")

    execution = _AccountingGraphRuntime(_CorruptingBackend()).run(
        _compiled(_graph("a")),
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=lambda datum, row: (datum.id.value, row[COL_TEXT]),
    )

    assert isinstance(execution.accounting.invocation, _InvocationInconsistent)
    assert isinstance(execution.accounting.tasks[0], _TaskInconsistent)


def test_accounting_runtime_accepts_only_explicit_trusted_stop_evidence(
    stub_slim_model_selection: ModelSelection,
) -> None:
    class _StoppedBackend:
        def run(
            self,
            dataframe: pd.DataFrame,
            *,
            invocation: _CompiledInvocation,
            data_summary: str | None,
            preview_num_records: int | None,
            verifier: _InvocationRowVerifier,
        ) -> _PandasExecutionResult:
            del invocation, data_summary, preview_num_records
            verifier.abort(cancelled=True)
            terminal_outcomes = verifier.take_terminal_outcomes()
            token = terminal_outcomes[0][0]
            return _PandasExecutionResult(
                dataframe=pd.DataFrame(),
                failed_records=[],
                terminal_outcomes=terminal_outcomes,
                trusted_stop_tokens=(token,),
            )

    execution = _AccountingGraphRuntime(_StoppedBackend()).run(
        _compiled(_graph("a")),
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=lambda datum, row: (datum.id.value, row[COL_TEXT]),
    )

    assert isinstance(execution.accounting.invocation, _InvocationCancelled)
    assert isinstance(execution.accounting.tasks[0], _TaskCancelled)
    assert isinstance(execution.accounting.groups[0], _GroupWithheld)


def test_accounting_runtime_attributes_failure_by_opaque_token_and_releases_independent_group(
    stub_slim_model_selection: ModelSelection,
) -> None:
    public_failure = FailedRecord("content-derived-public-id", "replace", "dropped")

    class _PartialBackend:
        def run(
            self,
            dataframe: pd.DataFrame,
            *,
            invocation: _CompiledInvocation,
            data_summary: str | None,
            preview_num_records: int | None,
            verifier: _InvocationRowVerifier,
        ) -> _PandasExecutionResult:
            del invocation, data_summary, preview_num_records
            first = dataframe[PRIVATE_CORRELATION_COLUMN].iloc[0]
            surviving = dataframe.iloc[[1]].assign(**{COL_FINAL_ENTITIES: [{"entities": []}]})
            verifier.freeze_accepted_detections(surviving)
            final = verifier.finish(surviving)
            return _PandasExecutionResult(
                dataframe=final,
                failed_records=[public_failure],
                terminal_outcomes=verifier.take_terminal_outcomes(),
                result_row_tokens=verifier.take_result_order(),
                failed_row_evidence=(_FailedRowEvidence(first, public_failure),),
            )

    execution = _AccountingGraphRuntime(_PartialBackend()).run(
        _compiled(_graph("a", "b")),
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=lambda datum, row: (datum.id.value, row[COL_TEXT]),
    )

    assert isinstance(execution.accounting.tasks[0], _TaskFailed)
    assert isinstance(execution.accounting.tasks[1], _TaskSucceeded)
    released = tuple(
        datum_id.value
        for group in execution.accounting.groups
        if isinstance(group, _GroupReleased)
        for datum_id, _candidate in group.outputs
    )
    assert released == ("b",)
    assert execution.failed_records == (public_failure,)


def test_accounting_runtime_rejects_unbound_failed_record_evidence(
    stub_slim_model_selection: ModelSelection,
) -> None:
    public_failure = FailedRecord("public-a", "replace", "dropped")
    unrelated_failure = FailedRecord("public-b", "replace", "dropped")

    class _MismatchedBackend:
        def run(
            self,
            dataframe: pd.DataFrame,
            *,
            invocation: _CompiledInvocation,
            data_summary: str | None,
            preview_num_records: int | None,
            verifier: _InvocationRowVerifier,
        ) -> _PandasExecutionResult:
            del invocation, data_summary, preview_num_records
            first = dataframe[PRIVATE_CORRELATION_COLUMN].iloc[0]
            surviving = dataframe.iloc[[1]].assign(**{COL_FINAL_ENTITIES: [{"entities": []}]})
            verifier.freeze_accepted_detections(surviving)
            final = verifier.finish(surviving)
            return _PandasExecutionResult(
                dataframe=final,
                failed_records=[public_failure],
                terminal_outcomes=verifier.take_terminal_outcomes(),
                result_row_tokens=verifier.take_result_order(),
                failed_row_evidence=(_FailedRowEvidence(first, unrelated_failure),),
            )

    execution = _AccountingGraphRuntime(_MismatchedBackend()).run(
        _compiled(_graph("a", "b")),
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=lambda datum, row: (datum.id.value, row[COL_TEXT]),
    )

    assert isinstance(execution.accounting.invocation, _InvocationInconsistent)
    assert all(isinstance(group, _GroupWithheld) for group in execution.accounting.groups)


def test_accounting_runtime_blocks_dependent_before_effect_when_prerequisite_fails_release(
    stub_slim_model_selection: ModelSelection,
) -> None:
    frames: list[tuple[str, ...]] = []

    class _Backend:
        def run(
            self,
            dataframe: pd.DataFrame,
            *,
            invocation: _CompiledInvocation,
            data_summary: str | None,
            preview_num_records: int | None,
            verifier: _InvocationRowVerifier,
        ) -> _PandasExecutionResult:
            del invocation, data_summary, preview_num_records
            frames.append(tuple(dataframe[COL_TEXT]))
            detected = dataframe.assign(**{COL_FINAL_ENTITIES: [{"entities": []}] * len(dataframe)})
            verifier.freeze_accepted_detections(detected)
            final = verifier.finish(detected)
            return _PandasExecutionResult(
                dataframe=final,
                failed_records=[],
                terminal_outcomes=verifier.take_terminal_outcomes(),
                result_row_tokens=verifier.take_result_order(),
            )

    graph = _graph("a", "b")
    graph = replace(graph, dependencies=(_dependency(graph, "a", "b"),))
    plan = _compile_accounting_plan(graph, limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    runtime = _AccountingGraphRuntime(_Backend())

    execution = runtime.run(
        plan,
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=lambda datum, row: (datum.id.value, row[COL_TEXT]),
        datum_release_predicate=lambda datum_id, _candidate: datum_id.value != "a",
    )

    assert frames == [("synthetic a",)]
    assert any(isinstance(outcome, _TaskBlocked) for outcome in execution.accounting.tasks)


def test_accounting_admission_rejects_cycle_before_backend_effect(
    stub_slim_model_selection: ModelSelection,
) -> None:
    calls = 0

    class _Backend:
        def run(self, *_args: object, **_kwargs: object) -> _AccountingGraphExecution[_RedactCandidate]:
            nonlocal calls
            calls += 1
            raise AssertionError

    graph = _graph("a", "b")
    graph = replace(
        graph,
        dependencies=(_dependency(graph, "a", "b"), _dependency(graph, "b", "a")),
    )

    result = _RedactProtectionService(_Backend()).admit(graph, limits=_LIMITS)

    assert isinstance(result, _AccountingRejected)
    assert calls == 0


def test_accounting_runtime_classifies_worker_process_death_as_lost(
    tmp_path: Path,
    stub_slim_model_selection: ModelSelection,
) -> None:
    marker = tmp_path / "backend-worker-started"
    calls = 0

    class _CrashableBackend:
        def run(self, *_args: object, **_kwargs: object) -> _PandasExecutionResult:
            nonlocal calls
            calls += 1
            with ProcessPoolExecutor(max_workers=1) as executor:
                executor.submit(_crash_backend_worker, str(marker)).result(timeout=10)
            raise AssertionError("crashable worker must not report success")

    execution = _AccountingGraphRuntime(_CrashableBackend()).run(
        _compiled(_graph("a")),
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=lambda datum, row: (datum.id.value, row[COL_TEXT]),
    )
    reference = reduce_reference(
        _SINGLETON,
        {"datum-a": ReferenceTaskOutcome.LOST},
    )

    assert isinstance(execution.accounting.invocation, _InvocationLost)
    assert not any(isinstance(group, _GroupReleased) for group in execution.accounting.groups)
    assert reference.released_groups == frozenset()
    assert marker.read_text() == "started"
    assert calls == 1


def test_accounting_runtime_rejects_malformed_terminal_evidence_without_raising(
    stub_slim_model_selection: ModelSelection,
) -> None:
    class _MalformedBackend:
        def run(
            self,
            dataframe: pd.DataFrame,
            *,
            invocation: _CompiledInvocation,
            data_summary: str | None,
            preview_num_records: int | None,
            verifier: _InvocationRowVerifier,
        ) -> _PandasExecutionResult:
            del invocation, data_summary, preview_num_records, verifier
            token = dataframe[PRIVATE_CORRELATION_COLUMN].iloc[0]
            return _PandasExecutionResult(
                dataframe=pd.DataFrame(),
                failed_records=[],
                terminal_outcomes=((token, cast(_TerminalOutcome, "success")),),
                result_row_tokens=(),
            )

    execution = _AccountingGraphRuntime(_MalformedBackend()).run(
        _compiled(_graph("a")),
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=lambda datum, row: (datum.id.value, row[COL_TEXT]),
    )

    assert isinstance(execution.accounting.invocation, _InvocationInconsistent)
    assert isinstance(execution.accounting.groups[0], _GroupWithheld)


def test_accounting_runtime_localizes_hydration_failure_and_withholds_output(
    stub_slim_model_selection: ModelSelection,
) -> None:
    class _Backend:
        def run(
            self,
            dataframe: pd.DataFrame,
            *,
            invocation: _CompiledInvocation,
            data_summary: str | None,
            preview_num_records: int | None,
            verifier: _InvocationRowVerifier,
        ) -> _PandasExecutionResult:
            del invocation, data_summary, preview_num_records
            detected = dataframe.assign(**{COL_FINAL_ENTITIES: [{"entities": []}]})
            verifier.freeze_accepted_detections(detected)
            final = verifier.finish(detected)
            return _PandasExecutionResult(
                dataframe=final,
                failed_records=[],
                terminal_outcomes=verifier.take_terminal_outcomes(),
                result_row_tokens=verifier.take_result_order(),
            )

    def _malformed_hydration(_datum: _TextDatum, _row: pd.Series) -> str:
        raise TypeError("private-hydration-canary")

    execution = _AccountingGraphRuntime(_Backend()).run(
        _compiled(_graph("a")),
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        hydrate=_malformed_hydration,
    )

    assert isinstance(execution.accounting.invocation, _InvocationCompleted)
    assert isinstance(execution.accounting.tasks[0], _TaskFailed)
    assert isinstance(execution.accounting.groups[0], _GroupWithheld)
    assert "private-hydration-canary" not in repr(execution)


@pytest.mark.parametrize("raises", [False, True])
def test_group_release_predicate_failure_never_exposes_group_output(raises: bool) -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])
    ledger.accept_success(dispatch, "protected")

    def _group_predicate(_outputs: tuple[tuple[_DatumId, str], ...]) -> bool:
        if raises:
            raise RuntimeError("private-group-canary")
        return False

    result = ledger.finish(group_release_predicate=_group_predicate)

    if raises:
        assert isinstance(result.invocation, _InvocationFailed)
    else:
        assert isinstance(result.invocation, _InvocationCompleted)
    assert isinstance(result.groups[0], _GroupWithheld)


def test_group_release_predicate_requires_an_exact_boolean() -> None:
    ledger = _ledger(_compiled(_graph("a")))
    ledger.open()
    dispatch = ledger.dispatch(ledger.ready_tasks()[0])
    ledger.accept_success(dispatch, "protected")

    result = ledger.finish(group_release_predicate=lambda _outputs: cast(bool, "truthy-not-bool"))

    assert isinstance(result.invocation, _InvocationFailed)
    assert isinstance(result.groups[0], _GroupWithheld)
    assert "private-group-canary" not in repr(result)


def test_group_predicate_failure_propagates_through_explicit_dependencies() -> None:
    graph = _graph("a", "b")
    graph = replace(graph, dependencies=(_dependency(graph, "a", "b"),))
    plan = _compiled(graph)
    ledger = _ledger(plan)
    ledger.open()

    first = ledger.ready_tasks()
    assert tuple(task.datum_id.value for task in first) == ("a",)
    ledger.accept_success(ledger.dispatch(first[0]), "protected-a")
    second = ledger.ready_tasks()
    assert tuple(task.datum_id.value for task in second) == ("b",)
    ledger.accept_success(ledger.dispatch(second[0]), "protected-b")

    result = ledger.finish(
        group_release_predicate=lambda outputs: all(datum_id.value != "a" for datum_id, _output in outputs)
    )

    assert tuple(type(group) for group in result.groups) == (_GroupWithheld, _GroupWithheld)


def test_accounting_admission_compiles_a_detached_singleton_plan() -> None:
    source = _TextDatum(_DatumId("datum-a"), "synthetic input")

    result = _compile_accounting_plan(_trivial_graph((source,)), limits=_LIMITS)

    assert isinstance(result, _AccountingPlan)
    assert result.datums == (source,)
    assert result.datums[0] is not source
    assert tuple(task.datum_id for task in result.tasks) == (source.id,)
    object.__setattr__(source, "text", "mutated after compilation")
    assert result.datums[0].text == "synthetic input"


def test_compiled_plan_is_frozen(stub_slim_model_selection: ModelSelection) -> None:
    plan = _compile_accounting_plan(_graph("a"), limits=_LIMITS)
    assert isinstance(plan, _AccountingPlan)

    with pytest.raises(FrozenInstanceError):
        setattr(plan, "datums", ())


def test_accounting_admission_accepts_exact_byte_limits_and_rejects_limit_plus_one() -> None:
    exact = _trivial_graph((_TextDatum(_DatumId("iiii"), "1234"),))
    limits = _AccountingLimits(max_datums=1, max_datum_bytes=4, max_graph_bytes=4, max_id_bytes=4)

    assert isinstance(_compile_accounting_plan(exact, limits=limits), _AccountingPlan)
    assert _compile_accounting_plan(
        _trivial_graph((_TextDatum(_DatumId("iiiii"), "1234"),)),
        limits=limits,
    ) == _AccountingRejected(_AccountingAdmissionCode.MALFORMED_GRAPH)
    assert _compile_accounting_plan(
        _trivial_graph((_TextDatum(_DatumId("iiii"), "12345"),)),
        limits=limits,
    ) == _AccountingRejected(_AccountingAdmissionCode.DATUM_TOO_LARGE)


@pytest.mark.parametrize(
    "datum",
    [
        _TextDatum(_DatumId("invalid-\ud800-id"), "valid text"),
        _TextDatum(_DatumId("valid-id"), "invalid-\ud800-text"),
    ],
)
def test_accounting_admission_rejects_non_utf8_datum_values(datum: _TextDatum) -> None:
    assert _compile_accounting_plan(_trivial_graph((datum,)), limits=_LIMITS) == _AccountingRejected(
        _AccountingAdmissionCode.MALFORMED_GRAPH
    )


def test_accounting_admission_rejects_empty_and_count_limit_plus_one() -> None:
    assert _compile_accounting_plan(_trivial_graph(()), limits=_LIMITS) == _AccountingRejected(
        _AccountingAdmissionCode.MALFORMED_GRAPH
    )
    assert _compile_accounting_plan(_graph("a", "b"), limits=replace(_LIMITS, max_datums=1)) == _AccountingRejected(
        _AccountingAdmissionCode.TOO_MANY_DATUMS
    )


def test_accounting_admission_accepts_exact_count_limits_and_rejects_plus_one() -> None:
    four = _graph("a", "b", "c", "d")
    assert isinstance(_compile_accounting_plan(four, limits=_LIMITS), _AccountingPlan)
    assert _compile_accounting_plan(_graph("a", "b", "c", "d", "e"), limits=_LIMITS) == _AccountingRejected(
        _AccountingAdmissionCode.TOO_MANY_DATUMS
    )

    one_edge = replace(four, dependencies=(_dependency(four, "a", "b"),))
    assert isinstance(_compile_accounting_plan(one_edge, limits=replace(_LIMITS, max_dependencies=1)), _AccountingPlan)
    two_edges = replace(
        four,
        dependencies=(_dependency(four, "a", "b"), _dependency(four, "a", "c")),
    )
    assert _compile_accounting_plan(two_edges, limits=replace(_LIMITS, max_dependencies=1)) == _AccountingRejected(
        _AccountingAdmissionCode.TOO_MANY_DEPENDENCIES
    )

    assert isinstance(_compile_accounting_plan(four, limits=replace(_LIMITS, max_atomic_groups=4)), _AccountingPlan)
    assert _compile_accounting_plan(four, limits=replace(_LIMITS, max_atomic_groups=3)) == _AccountingRejected(
        _AccountingAdmissionCode.TOO_MANY_ATOMIC_GROUPS
    )


def test_accounting_admission_enforces_aggregate_graph_bytes_and_exact_stage_limit() -> None:
    graph = _trivial_graph((_TextDatum(_DatumId("a"), "12"), _TextDatum(_DatumId("b"), "34")))
    exact = _AccountingLimits(max_datums=2, max_datum_bytes=2, max_graph_bytes=4, max_stages=3)

    assert isinstance(_compile_accounting_plan(graph, limits=exact, stages=("a", "b", "c")), _AccountingPlan)
    assert _compile_accounting_plan(graph, limits=replace(exact, max_graph_bytes=3)) == _AccountingRejected(
        _AccountingAdmissionCode.GRAPH_TOO_LARGE
    )


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda graph: replace(
                graph,
                links=(_DatumLink(graph.datums[0].id, graph.datums[1].id, _RelationKind.RELATED),),
            ),
            _AccountingAdmissionCode.UNSUPPORTED_RELATIONSHIPS,
        ),
        (
            lambda graph: replace(
                graph,
                context_scopes=(_ContextScope(graph.datums[0].id, (graph.datums[1].id,)),),
            ),
            _AccountingAdmissionCode.UNSUPPORTED_CONTEXT,
        ),
        (
            lambda graph: replace(
                graph,
                coherence_scopes=(_CoherenceScope(tuple(datum.id for datum in graph.datums)),),
            ),
            _AccountingAdmissionCode.UNSUPPORTED_COHERENCE,
        ),
    ],
)
def test_accounting_admission_rejects_unsupported_phase4_semantics(
    mutation: Callable[[_ProtectionGraph], _ProtectionGraph],
    code: _AccountingAdmissionCode,
) -> None:
    assert _compile_accounting_plan(mutation(_graph("a", "b")), limits=_LIMITS) == _AccountingRejected(code)


def test_accounting_admission_rejects_duplicate_empty_context_declarations() -> None:
    graph = _graph("a", "b")
    duplicated = replace(graph, context_scopes=(*graph.context_scopes, graph.context_scopes[0]))

    assert _compile_accounting_plan(duplicated, limits=_LIMITS) == _AccountingRejected(
        _AccountingAdmissionCode.UNSUPPORTED_CONTEXT
    )


@pytest.mark.parametrize("stages", [(), ("",), ("same", "same"), ("a", "b", "c", "d")])
def test_accounting_admission_rejects_unsupported_stage_cardinality(stages: tuple[str, ...]) -> None:
    assert _compile_accounting_plan(_graph("a"), limits=_LIMITS, stages=stages) == _AccountingRejected(
        _AccountingAdmissionCode.UNSUPPORTED_TASK_CARDINALITY
    )


def test_streaming_conformance_corpus_matches_ledger_and_frozen_digest() -> None:
    digest = sha256()
    graph_count = 0
    trace_count = 0
    for case in streaming_conformance_cases():
        digest.update(_canonical_case(case.declaration, case.observations))
        trace_count += 1
        if not case.declaration.datum_ids:
            graph_count += 1
            assert _compile_accounting_plan(_graph(), limits=_LIMITS) == _AccountingRejected(
                _AccountingAdmissionCode.MALFORMED_GRAPH
            )
            continue
        if case.graph_witness and case.declaration.stages == ("stage-0",):
            graph_count += 1
        expected = reduce_observations(case.declaration, case.observations)
        actual = _run_ledger_case(case.declaration, case.observations)
        assert _ledger_shape(actual) == _reference_shape(expected)

    assert _CONFORMANCE_MANIFEST["generator_version"] == "phase4-stream-v4"
    assert graph_count == _CONFORMANCE_MANIFEST["graph_count"]
    assert trace_count == _CONFORMANCE_MANIFEST["canonical_trace_count"]
    assert digest.hexdigest() == _CONFORMANCE_MANIFEST["sha256"]


def test_barrier_race_matrix_preserves_one_shot_terminal_conservation() -> None:
    """Barrier gates force each legal terminal ordering without timing assumptions."""

    def ordered(first: Callable[[], object], second: Callable[[], object]) -> tuple[object, object]:
        started = Barrier(3)
        first_finished = Barrier(2)

        def run_first() -> object:
            started.wait()
            outcome = first()
            first_finished.wait()
            return outcome

        def run_second() -> object:
            started.wait()
            first_finished.wait()
            return second()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(run_first)
            second_future = executor.submit(run_second)
            started.wait()
        return first_future.result(), second_future.result()

    def opened() -> tuple[_AccountingPlan, _AccountingLedger[str], _Dispatch]:
        plan = _compiled(_graph("a"))
        ledger = _ledger(plan)
        ledger.open()
        return plan, ledger, ledger.dispatch(ledger.ready_tasks()[0])

    def assert_conserved(
        plan: _AccountingPlan,
        result: _AccountingResult[str],
        *,
        released: tuple[str, ...],
    ) -> None:
        assert len(result.tasks) == len(plan.tasks) == 1
        assert len({outcome.task for outcome in result.tasks}) == 1
        assert len(result.datums) == len(plan.datums) == 1
        assert len(result.dependencies) == len(plan.dependencies) == 0
        assert len(result.stages) == len(plan.stages) == 1
        assert len(result.groups) == len(plan.atomic_groups) == 1
        outputs = tuple(
            datum.value for group in result.groups if isinstance(group, _GroupReleased) for datum, _ in group.outputs
        )
        assert outputs == released
        assert sum(isinstance(group, _GroupReleased) for group in result.groups) == (1 if released else 0)

    # Success publishes before a later cancellation: exactly one successful publication wins.
    plan, ledger, dispatch = opened()
    first, second = ordered(
        lambda: ledger.accept_success(dispatch, "protected"),
        lambda: ledger.finish(),
    )
    assert first is _EvidenceAcceptance.ACCEPTED
    assert isinstance(second, _AccountingResult)
    assert isinstance(second.invocation, _InvocationCompleted)
    assert_conserved(plan, second, released=("a",))
    with pytest.raises(_LedgerClosedError):
        ledger.finish()
    assert ledger.request_cancellation() is None

    # Cancellation is recorded before a late success; publication withholds output.
    plan, ledger, dispatch = opened()
    first, second = ordered(ledger.request_cancellation, lambda: ledger.accept_success(dispatch, "late"))
    assert first is None
    assert second is _EvidenceAcceptance.ACCEPTED
    result = ledger.finish()
    assert isinstance(result.invocation, _InvocationCancelled)
    assert_conserved(plan, result, released=())

    # A trusted stop wins before a late success.
    plan, ledger, dispatch = opened()
    ledger.request_cancellation()
    first, second = ordered(lambda: ledger.acknowledge_stop(dispatch), lambda: ledger.accept_success(dispatch, "late"))
    assert first is _EvidenceAcceptance.ACCEPTED
    assert second is _EvidenceAcceptance.REJECTED_STALE
    result = ledger.finish()
    assert isinstance(result.tasks[0], _TaskCancelled)
    assert_conserved(plan, result, released=())

    # Success wins before a late stop, but cancellation still withholds publication.
    plan, ledger, dispatch = opened()
    first, second = ordered(lambda: ledger.accept_success(dispatch, "protected"), ledger.request_cancellation)
    assert first is _EvidenceAcceptance.ACCEPTED
    assert second is None
    assert ledger.acknowledge_stop(dispatch) is _EvidenceAcceptance.REJECTED_STALE
    result = ledger.finish()
    assert isinstance(result.tasks[0], _TaskSucceeded)
    assert isinstance(result.invocation, _InvocationCancelled)
    assert_conserved(plan, result, released=())

    # Post-dispatch cancellation without a trusted stop closes as loss, never retrying.
    plan, ledger, dispatch = opened()
    first, second = ordered(ledger.request_cancellation, ledger.finish)
    assert first is None
    assert isinstance(second, _AccountingResult)
    assert isinstance(second.tasks[0], _TaskLost)
    assert isinstance(second.invocation, _InvocationLost)
    assert_conserved(plan, second, released=())
    with pytest.raises(_LedgerStateError):
        ledger.dispatch(dispatch.task)
    with pytest.raises(_LedgerClosedError):
        ledger.finish()


def test_concurrent_frontier_terminals_preserve_hierarchical_fixed_point() -> None:
    graph = _graph("a", "b", "c")
    graph = replace(
        graph,
        dependencies=(_dependency(graph, "a", "b"),),
        atomic_groups=(_group(graph, "a", "c"), _group(graph, "b")),
    )
    ledger = _ledger(_compiled(graph))
    ledger.open()
    first, peer = tuple(ledger.dispatch(task) for task in ledger.ready_tasks())
    gate = Barrier(3)

    def terminalize(dispatch: _Dispatch, *, succeeds: bool) -> _EvidenceAcceptance:
        gate.wait()
        return (
            ledger.accept_success(dispatch, f"protected-{dispatch.task.datum_id.value}")
            if succeeds
            else ledger.accept_failure(dispatch)
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        success = executor.submit(terminalize, first, succeeds=True)
        failure = executor.submit(terminalize, peer, succeeds=False)
        gate.wait()
    assert success.result() is _EvidenceAcceptance.ACCEPTED
    assert failure.result() is _EvidenceAcceptance.ACCEPTED

    dependent = ledger.dispatch(ledger.ready_tasks()[0])
    ledger.accept_success(dependent, "protected-b")
    result = ledger.finish()

    assert isinstance(result.invocation, _InvocationCompleted)
    assert all(isinstance(group, _GroupWithheld) for group in result.groups)
    assert not any(isinstance(group, _GroupReleased) for group in result.groups)


def _canonical_case(
    declaration: ReferenceDeclaration,
    observations: tuple[ReferenceObservation, ...],
) -> bytes:
    def event(observation: object) -> tuple[str, tuple[str, str] | None]:
        match observation:
            case ReferenceDispatch(task=task):
                return ("dispatch", task)
            case ReferenceSuccess(task=task):
                return ("success", task)
            case ReferenceFailure(task=task):
                return ("failure", task)
            case ReferenceCancellationRequest():
                return ("cancel", None)
            case ReferenceStopAcknowledgement(task=task):
                return ("stop", task)
            case ReferenceTransportLoss(task=task):
                return ("lost", task)
            case ReferenceContradiction():
                return ("contradiction", None)
            case ReferenceResultConstructionFailure():
                return ("result-construction-failure", None)
            case ReferenceCorruptEvidence(kind=kind):
                return (f"corruption:{kind.value}", None)
            case _:
                raise AssertionError("unknown reference observation")

    return dumps(
        {
            "atomic_groups": declaration.atomic_groups,
            "datum_ids": declaration.datum_ids,
            "dependencies": declaration.dependencies,
            "observations": tuple(event(observation) for observation in observations),
            "stages": declaration.stages,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _run_ledger_case(
    declaration: ReferenceDeclaration,
    observations: tuple[ReferenceObservation, ...],
) -> _AccountingResult[str]:
    graph = _graph(*declaration.datum_ids)
    graph = replace(
        graph,
        dependencies=tuple(_dependency(graph, *dependency) for dependency in declaration.dependencies),
        atomic_groups=tuple(_group(graph, *group) for group in declaration.atomic_groups),
    )
    plan = _compile_accounting_plan(graph, limits=_LIMITS, stages=declaration.stages)
    assert isinstance(plan, _AccountingPlan)
    ledger = _ledger(plan)
    ledger.open()
    result_construction_failure = False
    tasks = {(task.stage.value, task.datum_id.value): task for task in plan.tasks}
    dispatches = {}
    for observation in observations:
        match observation:
            case ReferenceDispatch(task=task):
                dispatches[task] = ledger.dispatch(tasks[task])
            case ReferenceSuccess(task=task):
                ledger.accept_success(dispatches[task], f"candidate-{task[0]}-{task[1]}")
            case ReferenceFailure(task=task):
                ledger.accept_failure(dispatches[task])
            case ReferenceCancellationRequest():
                ledger.request_cancellation()
            case ReferenceStopAcknowledgement(task=task):
                ledger.acknowledge_stop(dispatches[task])
            case ReferenceTransportLoss(task=task):
                ledger.mark_transport_lost(dispatches[task])
            case ReferenceContradiction():
                ledger.mark_inconsistent(_CauseCode.CONTRADICTORY)
            case ReferenceResultConstructionFailure():
                result_construction_failure = True
            case ReferenceCorruptEvidence(kind=kind):
                dispatch = dispatches[next(reversed(dispatches))]
                match kind:
                    case ReferenceCorruption.MISSING:
                        ledger.reconcile((dispatch,), (), trusted_run_record=True)
                    case ReferenceCorruption.DUPLICATE:
                        ledger.reconcile(
                            (dispatch, dispatch),
                            (_SuccessRecord(dispatch, "duplicate"),),
                            trusted_run_record=True,
                        )
                    case ReferenceCorruption.UNKNOWN:
                        unknown = replace(
                            dispatch,
                            attempt_id=_AttemptId("unknown-attempt"),
                            row_token=_RowToken("unknown-row"),
                        )
                        ledger.reconcile((dispatch,), (_SuccessRecord(unknown, "unknown"),), trusted_run_record=True)
                    case ReferenceCorruption.FOREIGN:
                        foreign = replace(dispatch, row_token=_RowToken("foreign-token"))
                        ledger.reconcile((dispatch,), (_SuccessRecord(foreign, "foreign"),), trusted_run_record=True)
                    case ReferenceCorruption.STALE:
                        stale = replace(dispatch, attempt_id=_AttemptId("stale-attempt"))
                        ledger.reconcile((dispatch,), (_SuccessRecord(stale, "stale"),), trusted_run_record=True)
                    case ReferenceCorruption.SWAPPED:
                        swapped_task = next(task for task in plan.tasks if task != dispatch.task)
                        swapped = replace(dispatch, task=swapped_task)
                        ledger.reconcile((dispatch,), (_SuccessRecord(swapped, "swapped"),), trusted_run_record=True)
                    case ReferenceCorruption.PLAN_MISMATCH:
                        incompatible_plan = _compile_accounting_plan(
                            graph,
                            limits=_LIMITS,
                            stages=("incompatible-stage",),
                        )
                        assert isinstance(incompatible_plan, _AccountingPlan)
                        incompatible_ledger = _ledger(incompatible_plan)
                        incompatible_ledger.open()
                        incompatible_dispatch = incompatible_ledger.dispatch(incompatible_ledger.ready_tasks()[0])
                        ledger.reconcile(
                            (dispatch,),
                            (_SuccessRecord(incompatible_dispatch, "plan-mismatch"),),
                            trusted_run_record=True,
                        )
                    case ReferenceCorruption.CONTRADICTORY:
                        ledger.reconcile((), (_SuccessRecord(dispatch, "contradiction"),), trusted_run_record=True)
            case _:
                raise AssertionError("unknown reference observation")
    return ledger.finish(
        group_release_predicate=(
            (lambda _outputs: cast(bool, "invalid-result")) if result_construction_failure else (lambda _outputs: True)
        )
    )


def _ledger_shape(result: _AccountingResult[str]) -> tuple[object, ...]:
    return (
        tuple(((task.task.stage.value, task.task.datum_id.value), _outcome_name(task)) for task in result.tasks),
        tuple(
            (
                (task.task.stage.value, task.task.datum_id.value),
                tuple(cause.code.value for cause in getattr(task, "causes", ())),
            )
            for task in result.tasks
        ),
        tuple((datum.datum_id.value, _outcome_name(datum)) for datum in result.datums),
        tuple(
            (
                (dependency.dependency.prerequisite.value, dependency.dependency.dependent.value),
                isinstance(dependency, _DependencySatisfied),
            )
            for dependency in result.dependencies
        ),
        tuple((stage.stage.value, _outcome_name(stage)) for stage in result.stages),
        tuple(
            tuple(datum_id.value for datum_id, _candidate in group.outputs)
            for group in result.groups
            if isinstance(group, _GroupReleased)
        ),
        _outcome_name(result.invocation),
    )


def _reference_shape(result: ReferenceHierarchyResult) -> tuple[object, ...]:
    return (
        result.tasks,
        result.task_causes,
        result.datums,
        result.dependencies,
        result.stages,
        result.released_group_order,
        result.invocation.value,
    )


def _outcome_name(outcome: object) -> str:
    name = type(outcome).__name__.removeprefix("_").removeprefix("Invocation")
    return {
        "TaskSucceeded": "succeeded",
        "TaskFailed": "failed",
        "TaskCancelled": "cancelled",
        "TaskLost": "lost",
        "TaskBlocked": "blocked",
        "TaskInconsistent": "inconsistent",
        "DatumQualified": "succeeded",
        "DatumFailed": "failed",
        "DatumCancelled": "cancelled",
        "DatumLost": "lost",
        "DatumBlocked": "blocked",
        "DatumInconsistent": "inconsistent",
        "StageSucceeded": "succeeded",
        "StageFailed": "failed",
        "StageCancelled": "cancelled",
        "StageLost": "lost",
        "StageBlocked": "blocked",
        "StageInconsistent": "inconsistent",
        "Completed": "completed",
        "Failed": "failed",
        "Cancelled": "cancelled",
        "Lost": "lost",
        "Inconsistent": "inconsistent",
    }[name]


def _graph(*datum_names: str) -> _ProtectionGraph:
    return _trivial_graph(tuple(_TextDatum(_DatumId(name), f"synthetic {name}") for name in datum_names))


def _dependency(graph: _ProtectionGraph, prerequisite: str, dependent: str) -> _DatumDependency:
    by_id = {datum.id.value: datum.id for datum in graph.datums}
    return _DatumDependency(by_id[prerequisite], by_id[dependent])


def _group(graph: _ProtectionGraph, *members: str) -> _AtomicGroup:
    by_id = {datum.id.value: datum.id for datum in graph.datums}
    return _AtomicGroup(tuple(by_id[member] for member in members))


def test_accounting_admission_compiles_declaration_order_independent_dag_and_partition() -> None:
    source = _graph("d", "c", "b", "a")
    source = replace(
        source,
        dependencies=(
            _dependency(source, "a", "b"),
            _dependency(source, "a", "c"),
            _dependency(source, "b", "d"),
            _dependency(source, "c", "d"),
        ),
        atomic_groups=(_group(source, "a", "b"), _group(source, "c", "d")),
    )

    result = _compile_accounting_plan(source, limits=_LIMITS, stages=("detect", "protect"))

    assert isinstance(result, _AccountingPlan)
    assert tuple(datum_id.value for datum_id in result.topological_datums) == ("a", "c", "b", "d")
    assert tuple(task.datum_id.value for task in result.tasks[:4]) == ("a", "c", "b", "d")
    assert len(result.tasks) == 8
    assert {frozenset(member.value for member in group.members) for group in result.atomic_groups} == {
        frozenset(("a", "b")),
        frozenset(("c", "d")),
    }


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda graph: replace(graph, dependencies=(_DatumDependency(graph.datums[0].id, graph.datums[0].id),)),
            _AccountingAdmissionCode.SELF_DEPENDENCY,
        ),
        (
            lambda graph: replace(
                graph,
                dependencies=(_DatumDependency(graph.datums[0].id, _DatumId("missing")),),
            ),
            _AccountingAdmissionCode.DANGLING_DEPENDENCY,
        ),
        (
            lambda graph: replace(
                graph,
                dependencies=(
                    _DatumDependency(graph.datums[0].id, graph.datums[1].id),
                    _DatumDependency(graph.datums[0].id, graph.datums[1].id),
                ),
            ),
            _AccountingAdmissionCode.DUPLICATE_DEPENDENCY,
        ),
        (
            lambda graph: replace(
                graph,
                dependencies=(
                    _DatumDependency(graph.datums[0].id, graph.datums[1].id),
                    _DatumDependency(graph.datums[1].id, graph.datums[0].id),
                ),
            ),
            _AccountingAdmissionCode.DEPENDENCY_CYCLE,
        ),
        (
            lambda graph: replace(graph, atomic_groups=(_AtomicGroup(()), *graph.atomic_groups)),
            _AccountingAdmissionCode.EMPTY_ATOMIC_GROUP,
        ),
        (
            lambda graph: replace(
                graph,
                atomic_groups=(_AtomicGroup((graph.datums[0].id, graph.datums[0].id)), *graph.atomic_groups[1:]),
            ),
            _AccountingAdmissionCode.DUPLICATE_ATOMIC_MEMBER,
        ),
        (
            lambda graph: replace(
                graph,
                atomic_groups=(_AtomicGroup((_DatumId("missing"),)), *graph.atomic_groups[1:]),
            ),
            _AccountingAdmissionCode.DANGLING_ATOMIC_MEMBER,
        ),
        (
            lambda graph: replace(graph, atomic_groups=(*graph.atomic_groups, graph.atomic_groups[0])),
            _AccountingAdmissionCode.DUPLICATE_ATOMIC_GROUP,
        ),
        (
            lambda graph: replace(graph, atomic_groups=(_AtomicGroup((graph.datums[0].id,)),)),
            _AccountingAdmissionCode.ATOMIC_COVERAGE_GAP,
        ),
        (
            lambda graph: replace(
                graph,
                atomic_groups=(
                    _AtomicGroup((graph.datums[0].id, graph.datums[1].id)),
                    _AtomicGroup((graph.datums[1].id, graph.datums[2].id)),
                ),
            ),
            _AccountingAdmissionCode.ATOMIC_GROUP_OVERLAP,
        ),
        (
            lambda graph: replace(
                graph,
                atomic_groups=(
                    _AtomicGroup((graph.datums[0].id,)),
                    _AtomicGroup((graph.datums[0].id, graph.datums[1].id, graph.datums[2].id)),
                ),
            ),
            _AccountingAdmissionCode.UNSUPPORTED_ATOMIC_NESTING,
        ),
    ],
)
def test_accounting_admission_rejects_invalid_topology(
    mutation: Callable[[_ProtectionGraph], _ProtectionGraph],
    code: _AccountingAdmissionCode,
) -> None:
    graph = mutation(_graph("a", "b", "c"))

    result = _compile_accounting_plan(graph, limits=_LIMITS)

    assert result == _AccountingRejected(code)


def test_cycle_precedes_recognized_but_unsupported_atomic_nesting() -> None:
    graph = _graph("a", "b")
    graph = replace(
        graph,
        dependencies=(_dependency(graph, "a", "b"), _dependency(graph, "b", "a")),
        atomic_groups=(_group(graph, "a"), _group(graph, "a", "b")),
    )

    result = _compile_accounting_plan(graph, limits=_LIMITS)

    assert result == _AccountingRejected(_AccountingAdmissionCode.DEPENDENCY_CYCLE)


def test_release_uses_dependency_group_fixed_point_and_matches_independent_model() -> None:
    graph = _graph("a", "b", "c", "d", "e", "f")
    graph = replace(
        graph,
        dependencies=(_dependency(graph, "a", "c"), _dependency(graph, "d", "e")),
        atomic_groups=(
            _group(graph, "a", "b"),
            _group(graph, "c", "d"),
            _group(graph, "e"),
            _group(graph, "f"),
        ),
    )
    limits = replace(_LIMITS, max_datums=6, max_graph_bytes=256)
    plan = _compile_accounting_plan(graph, limits=limits)
    assert isinstance(plan, _AccountingPlan)
    qualified = frozenset(datum.id for datum in plan.datums if datum.id.value != "b")

    actual = _qualify_release(plan, qualified)
    reference = reduce_reference(
        ReferenceDeclaration(
            tuple(datum.id.value for datum in plan.datums),
            tuple((edge.prerequisite.value, edge.dependent.value) for edge in plan.dependencies),
            tuple(tuple(member.value for member in group.members) for group in plan.atomic_groups),
        ),
        {
            datum.id.value: (ReferenceTaskOutcome.FAILED if datum.id.value == "b" else ReferenceTaskOutcome.SUCCEEDED)
            for datum in plan.datums
        },
    )

    assert (
        frozenset(datum.value for datum in actual.release_eligible) == reference.release_eligible == frozenset(("f",))
    )
    released_members = frozenset(
        frozenset(member.value for member in group.members)
        for group in plan.atomic_groups
        if group.key in actual.released_groups
    )
    assert released_members == reference.released_groups == frozenset((frozenset(("f",)),))


def test_release_matches_independent_model_for_all_graphs_through_three_datums() -> None:
    for datum_count in range(1, 4):
        names = tuple(chr(ord("a") + ordinal) for ordinal in range(datum_count))
        base = _graph(*names)
        by_name = {datum.id.value: datum.id for datum in base.datums}
        for dependencies in acyclic_dependencies(names):
            for partition in flat_partitions(names):
                graph = replace(
                    base,
                    dependencies=tuple(
                        _DatumDependency(by_name[prerequisite], by_name[dependent])
                        for prerequisite, dependent in dependencies
                    ),
                    atomic_groups=tuple(
                        _AtomicGroup(tuple(by_name[member] for member in group)) for group in partition
                    ),
                )
                plan = _compile_accounting_plan(graph, limits=_LIMITS)
                assert isinstance(plan, _AccountingPlan)
                declaration = ReferenceDeclaration(names, dependencies, partition)
                for qualification_bits in product((False, True), repeat=datum_count):
                    qualified_names = frozenset(
                        name for name, qualified in zip(names, qualification_bits, strict=True) if qualified
                    )
                    actual = _qualify_release(
                        plan,
                        frozenset(datum.id for datum in plan.datums if datum.id.value in qualified_names),
                    )
                    reference = reduce_reference(
                        declaration,
                        {
                            name: (
                                ReferenceTaskOutcome.SUCCEEDED
                                if name in qualified_names
                                else ReferenceTaskOutcome.FAILED
                            )
                            for name in names
                        },
                    )
                    assert frozenset(datum.value for datum in actual.release_eligible) == reference.release_eligible
                    assert (
                        frozenset(
                            frozenset(member.value for member in group.members)
                            for group in plan.atomic_groups
                            if group.key in actual.released_groups
                        )
                        == reference.released_groups
                    )


def test_seeded_opaque_id_renaming_and_declaration_permutations_preserve_release_semantics() -> None:
    rng = random.Random(0xA11CE)
    logical = tuple(range(4))
    for case in range(128):
        presentation = list(logical)
        rng.shuffle(presentation)
        names = {ordinal: f"d{case:x}-{ordinal:x}" for ordinal in logical}
        presented_names = tuple(names[ordinal] for ordinal in presentation)
        graph = _graph(*presented_names)
        by_name = {datum.id.value: datum.id for datum in graph.datums}
        dependencies = tuple(
            (left, right) for left in logical for right in logical if left < right and rng.choice((False, True))
        )
        bucket_by_member = {member: rng.randrange(4) for member in logical}
        groups = tuple(
            tuple(member for member in logical if bucket_by_member[member] == bucket)
            for bucket in range(4)
            if bucket in bucket_by_member.values()
        )
        graph = replace(
            graph,
            dependencies=tuple(
                _DatumDependency(by_name[names[left]], by_name[names[right]]) for left, right in dependencies
            ),
            atomic_groups=tuple(_AtomicGroup(tuple(by_name[names[member]] for member in group)) for group in groups),
        )
        plan = _compile_accounting_plan(graph, limits=replace(_LIMITS, max_graph_bytes=512))
        assert isinstance(plan, _AccountingPlan)
        qualified_members = frozenset(member for member in logical if rng.choice((False, True)))
        actual = _qualify_release(
            plan,
            frozenset(
                datum.id
                for datum in plan.datums
                if next(k for k, v in names.items() if v == datum.id.value) in qualified_members
            ),
        )
        reference = reduce_reference(
            ReferenceDeclaration(
                presented_names,
                tuple((names[left], names[right]) for left, right in dependencies),
                tuple(tuple(names[member] for member in group) for group in groups),
            ),
            {
                names[member]: (
                    ReferenceTaskOutcome.SUCCEEDED if member in qualified_members else ReferenceTaskOutcome.FAILED
                )
                for member in logical
            },
        )

        assert frozenset(datum.value for datum in actual.release_eligible) == reference.release_eligible


def test_dispatch_batch_size_does_not_change_terminal_or_release_result() -> None:
    graph = _graph("a", "b", "c", "d")
    graph = replace(
        graph,
        dependencies=(_dependency(graph, "a", "c"),),
        atomic_groups=(_group(graph, "a", "b"), _group(graph, "c", "d")),
    )
    plan = _compiled(graph)

    def execute(batch_size: int) -> _AccountingResult[str]:
        ledger = _ledger(plan)
        ledger.open()
        while ready := ledger.ready_tasks():
            for task in ready[:batch_size]:
                dispatch = ledger.dispatch(task)
                if task.datum_id.value == "b":
                    ledger.accept_failure(dispatch)
                else:
                    ledger.accept_success(dispatch, f"candidate-{task.datum_id.value}")
        return ledger.finish()

    assert _ledger_shape(execute(1)) == _ledger_shape(execute(4))
