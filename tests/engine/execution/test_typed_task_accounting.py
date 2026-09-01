# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from anonymizer.engine.execution.accounting_admission import _AccountingAdmissionCode, _compile_accounting_plan
from anonymizer.engine.execution.accounting_ledger import _AccountingLedger
from anonymizer.engine.execution.accounting_outcomes import (
    _DatumBlocked,
    _DatumFailed,
    _DatumQualified,
    _GroupReleased,
    _GroupWithheld,
    _InvocationCancelled,
    _InvocationCompleted,
    _InvocationLost,
    _StageBlocked,
    _StageFailed,
    _StageSucceeded,
    _TaskBlocked,
    _TaskCancelled,
    _TaskFailed,
    _TaskInconsistent,
    _TaskLost,
    _TaskSucceeded,
)
from anonymizer.engine.execution.accounting_plan import (
    _AccountingLimits,
    _AccountingPlan,
    _DatumTaskSubject,
    _is_admitted_accounting_plan,
    _ScopeTaskSubject,
    _StageId,
    _TaskKey,
    _TaskPredecessor,
)
from anonymizer.engine.execution.graph import _DatumDependency, _DatumId, _TextDatum, _trivial_graph
from anonymizer.engine.execution.graph_runtime import (
    _AccountingGraphAdmissionError,
    _AccountingGraphRuntime,
    _FrameExecutionBackend,
)
from anonymizer.engine.execution.invocation import _CompiledInvocation
from tests.engine.execution.phase4_reference_model import (
    ReferenceMixedDeclaration,
    ReferenceMixedTaskKey,
    ReferenceTaskOutcome,
    reduce_mixed_schedule,
)


def test_typed_task_accounting_test_infrastructure() -> None:
    task = _TaskKey(_StageId("protect"), _DatumTaskSubject(_DatumId("fabricated-datum")))

    assert task.stage.value == "protect"


def test_task_key_rejects_any_subject_outside_the_closed_sum() -> None:
    with pytest.raises(TypeError, match="task subject"):
        _TaskKey(_StageId("protect"), cast(_DatumTaskSubject, None))


def test_accounting_admission_emits_only_datum_owned_tasks() -> None:
    graph = _trivial_graph((_TextDatum(_DatumId("fabricated-datum"), "fabricated text"),))
    result = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=1, max_datum_bytes=64, max_graph_bytes=64),
    )

    assert isinstance(result, _AccountingPlan), "typed datum task admission is missing"
    assert result.tasks == (_TaskKey(_StageId("protect"), _DatumTaskSubject(_DatumId("fabricated-datum"))),)
    assert not hasattr(result.tasks[0], "datum_id")


def test_plan_adds_one_opaque_scope_task_for_each_declared_scope() -> None:
    graph = _trivial_graph((_TextDatum(_DatumId("fabricated-datum"), "fabricated text"),))
    plan = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=1, max_datum_bytes=64, max_graph_bytes=64),
    )
    assert isinstance(plan, _AccountingPlan)
    first_scope = _ScopeTaskSubject()
    zero_mention_scope = _ScopeTaskSubject()
    mixed = plan.with_scope_tasks(_StageId("scope-plan"), (first_scope, zero_mention_scope))

    scope_tasks = tuple(task for task in mixed.tasks if isinstance(task.subject, _ScopeTaskSubject))
    assert scope_tasks == (
        _TaskKey(_StageId("scope-plan"), first_scope),
        _TaskKey(_StageId("scope-plan"), zero_mention_scope),
    )
    assert first_scope is not zero_mention_scope
    assert plan.with_scope_tasks(_StageId("scope-plan"), ()) is plan


def test_scope_task_capability_is_bound_into_the_plan_proof() -> None:
    graph = _trivial_graph((_TextDatum(_DatumId("fabricated-datum"), "fabricated text"),))
    plan = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=1, max_datum_bytes=64, max_graph_bytes=64),
    )
    assert isinstance(plan, _AccountingPlan)
    mixed = plan.with_scope_tasks(_StageId("scope-plan"), (_ScopeTaskSubject(),))
    forged = replace(
        mixed,
        tasks=(*mixed.tasks[:-1], _TaskKey(_StageId("scope-plan"), _ScopeTaskSubject())),
    )

    assert _is_admitted_accounting_plan(mixed)
    assert not _is_admitted_accounting_plan(forged)


def test_scope_task_is_ready_without_implicit_datum_edges_and_stays_out_of_datum_reduction() -> None:
    graph = _trivial_graph((_TextDatum(_DatumId("fabricated-datum"), "fabricated text"),))
    plan = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=1, max_datum_bytes=64, max_graph_bytes=64),
    )
    assert isinstance(plan, _AccountingPlan)
    mixed = plan.with_scope_tasks(_StageId("scope-plan"), (_ScopeTaskSubject(),))
    ledger = _AccountingLedger[str](mixed, identity_factory=iter((f"opaque-{index}" for index in range(9))).__next__)
    ledger.open()
    try:
        ready = ledger.ready_tasks()
    except (AttributeError, KeyError, ValueError):
        ready = ()

    assert ready == mixed.tasks
    datum_task, scope_task = ready
    ledger.accept_success(ledger.dispatch(datum_task), "protected datum")
    ledger.accept_success(ledger.dispatch(scope_task), "scope receipt")
    result = ledger.finish()

    assert tuple(type(outcome) for outcome in result.tasks) == (_TaskSucceeded, _TaskSucceeded)
    assert result.datums == (_DatumQualified(_DatumId("fabricated-datum"), "protected datum"),)
    assert isinstance(result.groups[0], _GroupReleased)
    assert result.groups[0].outputs == ((_DatumId("fabricated-datum"), "protected datum"),)


def test_scope_task_readiness_is_governed_only_by_explicit_predecessors() -> None:
    graph = _trivial_graph((_TextDatum(_DatumId("fabricated-datum"), "fabricated text"),))
    plan = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=1, max_datum_bytes=64, max_graph_bytes=64),
    )
    assert isinstance(plan, _AccountingPlan)
    mixed = plan.with_scope_tasks(_StageId("scope-plan"), (_ScopeTaskSubject(),))
    datum_task, scope_task = mixed.tasks
    mixed = mixed.with_task_predecessors((_TaskPredecessor(scope_task, datum_task),))
    ledger = _AccountingLedger[str](mixed, identity_factory=iter((f"opaque-{index}" for index in range(9))).__next__)
    ledger.open()

    assert ledger.ready_tasks() == (scope_task,)
    ledger.accept_failure(ledger.dispatch(scope_task))
    assert ledger.ready_tasks() == ()
    result = ledger.finish()

    assert isinstance(result.tasks[0], _TaskBlocked)
    assert isinstance(result.tasks[1], _TaskFailed)
    assert isinstance(result.datums[0], _DatumBlocked)
    assert isinstance(result.groups[0], _GroupWithheld)


def test_scope_task_admission_rejects_duplicate_ownership_and_invalid_predecessors() -> None:
    graph = _trivial_graph((_TextDatum(_DatumId("fabricated-datum"), "fabricated text"),))
    plan = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=1, max_datum_bytes=64, max_graph_bytes=64),
    )
    assert isinstance(plan, _AccountingPlan)
    scope = _ScopeTaskSubject()

    with pytest.raises(TypeError, match="scope task subjects"):
        plan.with_scope_tasks(_StageId("scope-plan"), (scope, scope))

    mixed = plan.with_scope_tasks(_StageId("scope-plan"), (scope,))
    with pytest.raises(TypeError, match="scope task subjects"):
        mixed.with_scope_tasks(_StageId("scope-plan"), (scope,))

    datum_task, scope_task = mixed.tasks
    unknown = _TaskKey(_StageId("scope-plan"), _ScopeTaskSubject())
    invalid = (
        (_TaskPredecessor(scope_task, scope_task),),
        (_TaskPredecessor(unknown, datum_task),),
        (_TaskPredecessor(scope_task, datum_task), _TaskPredecessor(scope_task, datum_task)),
        (_TaskPredecessor(scope_task, datum_task), _TaskPredecessor(datum_task, scope_task)),
    )
    for predecessors in invalid:
        with pytest.raises(TypeError, match="task predecessor"):
            mixed.with_task_predecessors(predecessors)


@pytest.mark.parametrize(
    ("terminal", "task_type", "group_type", "invocation_type"),
    [
        pytest.param("failure", _TaskFailed, _GroupReleased, _InvocationCompleted, id="known-failure-is-local"),
        pytest.param("cancellation", _TaskCancelled, _GroupWithheld, _InvocationCancelled, id="cancellation-embargoes"),
        pytest.param("loss", _TaskLost, _GroupWithheld, _InvocationLost, id="loss-embargoes"),
        pytest.param("missing", _TaskInconsistent, _GroupReleased, _InvocationCompleted, id="missing-is-local"),
    ],
)
def test_scope_task_terminal_evidence_is_conserved_without_entering_datum_reduction(
    terminal: str,
    task_type: type[object],
    group_type: type[object],
    invocation_type: type[object],
) -> None:
    graph = _trivial_graph((_TextDatum(_DatumId("fabricated-datum"), "fabricated text"),))
    plan = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=1, max_datum_bytes=64, max_graph_bytes=64),
    )
    assert isinstance(plan, _AccountingPlan)
    mixed = plan.with_scope_tasks(_StageId("scope-plan"), (_ScopeTaskSubject(),))
    datum_task, scope_task = mixed.tasks
    ledger = _AccountingLedger[str](
        mixed,
        identity_factory=iter((f"opaque-{index}" for index in range(12))).__next__,
    )
    ledger.open()
    ledger.accept_success(ledger.dispatch(datum_task), "protected datum")
    if terminal == "failure":
        ledger.accept_failure(ledger.dispatch(scope_task))
    elif terminal == "cancellation":
        ledger.request_cancellation()
    elif terminal == "loss":
        ledger.mark_transport_lost(ledger.dispatch(scope_task))
    else:
        dispatch = ledger.dispatch(scope_task)
        ledger.reconcile((dispatch,), (), trusted_run_record=True)

    result = ledger.finish()

    assert len(result.tasks) == len(mixed.tasks) == 2
    assert isinstance(result.tasks[0], _TaskSucceeded)
    assert isinstance(result.tasks[1], task_type)
    assert result.datums == (_DatumQualified(_DatumId("fabricated-datum"), "protected datum"),)
    assert isinstance(result.groups[0], group_type)
    assert isinstance(result.invocation, invocation_type)


@pytest.mark.parametrize(
    ("predecessor_direction", "scope_outcome"),
    [
        pytest.param("scope-before-datum", ReferenceTaskOutcome.SUCCEEDED, id="scope-predecessor-succeeds"),
        pytest.param("scope-before-datum", ReferenceTaskOutcome.FAILED, id="scope-predecessor-fails"),
        pytest.param("datum-before-scope", ReferenceTaskOutcome.FAILED, id="scope-failure-does-not-block-release"),
    ],
)
def test_mixed_plan_matches_independent_reference_model(
    predecessor_direction: str,
    scope_outcome: ReferenceTaskOutcome,
) -> None:
    graph = _trivial_graph(
        (
            _TextDatum(_DatumId("a"), "fabricated a"),
            _TextDatum(_DatumId("b"), "fabricated b"),
        )
    )
    graph = replace(graph, dependencies=(_DatumDependency(_DatumId("a"), _DatumId("b")),))
    plan = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=2, max_datum_bytes=64, max_graph_bytes=64),
    )
    assert isinstance(plan, _AccountingPlan)
    scope_subject = _ScopeTaskSubject()
    mixed = plan.with_scope_tasks(_StageId("scope-plan"), (scope_subject,))
    datum_a, datum_b, scope = mixed.tasks
    predecessor = (
        _TaskPredecessor(scope, datum_a)
        if predecessor_direction == "scope-before-datum"
        else _TaskPredecessor(datum_a, scope)
    )
    mixed = mixed.with_task_predecessors((predecessor,))
    scope_key: ReferenceMixedTaskKey = ("scope-plan", ("scope", "scope-zero-mentions"))
    datum_a_key: ReferenceMixedTaskKey = ("protect", ("datum", "a"))
    datum_b_key: ReferenceMixedTaskKey = ("protect", ("datum", "b"))
    reference_predecessor = (
        (scope_key, datum_a_key) if predecessor_direction == "scope-before-datum" else (datum_a_key, scope_key)
    )
    schedule = (
        ((scope_key, scope_outcome),)
        if predecessor_direction == "scope-before-datum" and scope_outcome is ReferenceTaskOutcome.FAILED
        else (
            (scope_key, ReferenceTaskOutcome.SUCCEEDED),
            (datum_a_key, ReferenceTaskOutcome.SUCCEEDED),
            (datum_b_key, ReferenceTaskOutcome.SUCCEEDED),
        )
        if predecessor_direction == "scope-before-datum"
        else (
            (datum_a_key, ReferenceTaskOutcome.SUCCEEDED),
            (scope_key, scope_outcome),
            (datum_b_key, ReferenceTaskOutcome.SUCCEEDED),
        )
    )
    reference = reduce_mixed_schedule(
        ReferenceMixedDeclaration(
            datum_ids=("a", "b"),
            datum_stages=("protect",),
            scope_tasks=(("scope-plan", "scope-zero-mentions"),),
            datum_dependencies=(("a", "b"),),
            task_predecessors=(reference_predecessor,),
            atomic_groups=(("a",), ("b",)),
        ),
        schedule,
    )
    production_by_reference = {
        datum_a_key: datum_a,
        datum_b_key: datum_b,
        scope_key: scope,
    }
    ledger = _AccountingLedger[str](
        mixed,
        identity_factory=iter((f"opaque-{index}" for index in range(24))).__next__,
    )
    ledger.open()
    observed_frontiers = []
    for reference_task, outcome in schedule:
        observed_frontiers.append(tuple(_as_reference_key(task, scope_subject) for task in ledger.ready_tasks()))
        dispatch = ledger.dispatch(production_by_reference[reference_task])
        if outcome is ReferenceTaskOutcome.SUCCEEDED:
            ledger.accept_success(dispatch, f"candidate-{len(observed_frontiers)}")
        else:
            ledger.accept_failure(dispatch)
    result = ledger.finish()

    assert tuple(observed_frontiers) == reference.ready_frontiers
    assert (
        tuple((_as_reference_key(outcome.task, scope_subject), _task_outcome(outcome)) for outcome in result.tasks)
        == reference.tasks
    )
    assert tuple((outcome.datum_id.value, _datum_outcome(outcome)) for outcome in result.datums) == reference.datums
    assert tuple((outcome.stage.value, _stage_outcome(outcome)) for outcome in result.stages) == reference.stages
    assert (
        frozenset(
            frozenset(datum_id.value for datum_id, _candidate in group.outputs)
            for group in result.groups
            if isinstance(group, _GroupReleased)
        )
        == reference.released_groups
    )


def test_datum_only_graph_runtime_rejects_mixed_plan_before_backend_effects() -> None:
    graph = _trivial_graph((_TextDatum(_DatumId("fabricated-datum"), "fabricated text"),))
    plan = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=1, max_datum_bytes=64, max_graph_bytes=64),
    )
    assert isinstance(plan, _AccountingPlan)
    mixed = plan.with_scope_tasks(_StageId("scope-plan"), (_ScopeTaskSubject(),))
    effects: list[str] = []

    class _Backend:
        def run(self, *_args: object, **_kwargs: object) -> object:
            effects.append("run")
            raise AssertionError("backend must not run")

    runtime = _AccountingGraphRuntime(cast(_FrameExecutionBackend, _Backend()))
    with pytest.raises(_AccountingGraphAdmissionError) as raised:
        runtime.run(
            mixed,
            invocation=cast(_CompiledInvocation, object()),
            data_summary=None,
            preview_num_records=None,
            hydrate=lambda _datum, _row: "unused",
        )

    assert raised.value.code is _AccountingAdmissionCode.UNSUPPORTED_TASK_CARDINALITY
    assert effects == []


def _as_reference_key(task: _TaskKey, scope_subject: _ScopeTaskSubject) -> ReferenceMixedTaskKey:
    if isinstance(task.subject, _DatumTaskSubject):
        subject = ("datum", task.subject.datum_id.value)
    else:
        assert task.subject is scope_subject
        subject = ("scope", "scope-zero-mentions")
    return task.stage.value, subject


def _task_outcome(outcome: object) -> ReferenceTaskOutcome:
    if isinstance(outcome, _TaskSucceeded):
        return ReferenceTaskOutcome.SUCCEEDED
    if isinstance(outcome, _TaskFailed):
        return ReferenceTaskOutcome.FAILED
    if isinstance(outcome, _TaskBlocked):
        return ReferenceTaskOutcome.BLOCKED
    raise AssertionError("unexpected task outcome in mixed differential")


def _datum_outcome(outcome: object) -> ReferenceTaskOutcome:
    if isinstance(outcome, _DatumQualified):
        return ReferenceTaskOutcome.SUCCEEDED
    if isinstance(outcome, _DatumFailed):
        return ReferenceTaskOutcome.FAILED
    if isinstance(outcome, _DatumBlocked):
        return ReferenceTaskOutcome.BLOCKED
    raise AssertionError("unexpected datum outcome in mixed differential")


def _stage_outcome(outcome: object) -> ReferenceTaskOutcome:
    if isinstance(outcome, _StageSucceeded):
        return ReferenceTaskOutcome.SUCCEEDED
    if isinstance(outcome, _StageFailed):
        return ReferenceTaskOutcome.FAILED
    if isinstance(outcome, _StageBlocked):
        return ReferenceTaskOutcome.BLOCKED
    raise AssertionError("unexpected stage outcome in mixed differential")
