# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot phase-4 invocation ledger and terminal evidence acceptance."""

from __future__ import annotations

import operator
import secrets
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from functools import reduce, wraps
from threading import RLock
from typing import Concatenate, Generic, ParamSpec, TypeAlias, TypeVar, assert_never, final

from anonymizer.engine.execution.accounting_evidence import (
    _AttemptId,
    _Dispatch,
    _FailureRecord,
    _InvocationId,
    _RowToken,
    _SuccessRecord,
    _TerminalRecord,
)
from anonymizer.engine.execution.accounting_outcomes import (
    _AccountingResult,
    _CauseCode,
    _CauseSet,
    _DatumBlocked,
    _DatumCancelled,
    _DatumFailed,
    _DatumInconsistent,
    _DatumLost,
    _DatumOutcome,
    _DatumQualified,
    _DependencySatisfied,
    _DependencyUnsatisfied,
    _GroupOutcome,
    _GroupReleased,
    _GroupWithheld,
    _InvocationCancelled,
    _InvocationCompleted,
    _InvocationFailed,
    _InvocationInconsistent,
    _InvocationLost,
    _InvocationOutcome,
    _StageBlocked,
    _StageCancelled,
    _StageFailed,
    _StageInconsistent,
    _StageLost,
    _StageOutcome,
    _StageSucceeded,
    _TaskBlocked,
    _TaskCancelled,
    _TaskFailed,
    _TaskInconsistent,
    _TaskLost,
    _TaskOutcome,
    _TaskSucceeded,
    _TerminalCause,
)
from anonymizer.engine.execution.accounting_plan import _AccountingPlan, _AtomicGroupKey, _StageId, _TaskKey
from anonymizer.engine.execution.accounting_release import _qualify_release
from anonymizer.engine.execution.graph import _DatumId

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


def _serialized(
    method: Callable[Concatenate[_AccountingLedger[T], P], R],
) -> Callable[Concatenate[_AccountingLedger[T], P], R]:
    @wraps(method)
    def wrapped(ledger: _AccountingLedger[T], /, *args: P.args, **kwargs: P.kwargs) -> R:
        with ledger._lock:
            return method(ledger, *args, **kwargs)

    return wrapped


class _LedgerStateError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("private accounting ledger state violation")

    def __repr__(self) -> str:
        return "<private accounting ledger error>"


class _EvidenceAcceptance(str, Enum):
    ACCEPTED = "accepted"
    IDEMPOTENT_STALE = "idempotent_stale"
    REJECTED_STALE = "rejected_stale"


class _ResultConstructionFailure(Exception):
    pass


@final
@dataclass(frozen=True, slots=True)
class _Planned:
    task: _TaskKey


@final
@dataclass(frozen=True, slots=True)
class _Ready:
    task: _TaskKey


@final
@dataclass(frozen=True, slots=True)
class _Dispatched:
    dispatch: _Dispatch


_TaskState: TypeAlias = _Planned | _Ready | _Dispatched | _TaskOutcome[T]


def _default_identity() -> str:
    return secrets.token_hex(16)


class _AccountingLedger(Generic[T]):
    """Identity-bearing one-shot shell around pure accounting reducers."""

    def __init__(
        self,
        plan: _AccountingPlan,
        *,
        identity_factory: Callable[[], str] = _default_identity,
        datum_release_predicate: Callable[[_DatumId, T], bool] = lambda _datum_id, _candidate: True,
    ) -> None:
        self._plan = plan
        self._lock = RLock()
        self._identity_factory = identity_factory
        self._used_identities: set[str] = set()
        self._datum_release_predicate = datum_release_predicate
        self._datum_qualification: dict[_DatumId, bool] = {}
        self._invocation_id: _InvocationId | None = None
        self._states: dict[_TaskKey, _TaskState[T]] = {task: _Planned(task) for task in plan.tasks}
        self._accepted_records: dict[_AttemptId, _TerminalRecord[T]] = {}
        self._opened = False
        self._closed = False
        self._cancellation_requested = False
        self._global_inconsistent = False
        self._invocation_lost = False
        self._cleanup_failed = False
        self._cleanup_unconfirmed = False

    @_serialized
    def open(self) -> None:
        if self._opened or self._closed:
            raise _LedgerStateError
        self._invocation_id = _InvocationId(self._next_identity())
        self._opened = True

    @_serialized
    def ready_tasks(self) -> tuple[_TaskKey, ...]:
        self._require_active()
        self._advance_planned()
        return tuple(task for task in self._plan.tasks if isinstance(self._states[task], _Ready))

    @_serialized
    def dispatch(self, task: _TaskKey, *, row_token_value: str | None = None) -> _Dispatch:
        self._require_active()
        self._advance_planned()
        if not isinstance(self._states.get(task), _Ready) or self._invocation_id is None:
            raise _LedgerStateError
        row_token = self._next_identity() if row_token_value is None else self._claim_identity(row_token_value)
        dispatch = _Dispatch(
            self._invocation_id,
            task,
            _AttemptId(self._next_identity()),
            _RowToken(row_token),
        )
        self._states[task] = _Dispatched(dispatch)
        return dispatch

    @_serialized
    def dispatch_batch(
        self,
        tasks: tuple[_TaskKey, ...],
        *,
        row_token_values: tuple[str, ...],
    ) -> tuple[_Dispatch, ...]:
        """Atomically commit one context frontier after workframe construction."""
        self._require_active()
        self._advance_planned()
        if (
            self._invocation_id is None
            or len(tasks) != len(row_token_values)
            or len(set(tasks)) != len(tasks)
            or not all(isinstance(self._states.get(task), _Ready) for task in tasks)
        ):
            raise _LedgerStateError
        row_tokens = tuple(self._claim_identity(value) for value in row_token_values)
        dispatches = tuple(
            _Dispatch(self._invocation_id, task, _AttemptId(self._next_identity()), _RowToken(row_token))
            for task, row_token in zip(tasks, row_tokens, strict=True)
        )
        for dispatch in dispatches:
            self._states[dispatch.task] = _Dispatched(dispatch)
        return dispatches

    @_serialized
    def accept_success(self, dispatch: _Dispatch, candidate: T) -> _EvidenceAcceptance:
        return self._accept(_SuccessRecord(dispatch, candidate))

    @_serialized
    def accept_failure(self, dispatch: _Dispatch) -> _EvidenceAcceptance:
        return self._accept(_FailureRecord(dispatch))

    @_serialized
    def reconcile(
        self,
        dispatches: tuple[_Dispatch, ...],
        records: tuple[_TerminalRecord[T], ...],
        *,
        trusted_run_record: bool,
    ) -> None:
        self._require_active()
        if not trusted_run_record:
            self._invocation_lost = True
            for dispatch in dispatches:
                self._close_dispatch(dispatch, _TaskLost(dispatch.task, _causes(_CauseCode.TRANSPORT_LOST)))
            return
        expected = {dispatch.attempt_id: dispatch for dispatch in dispatches}
        observed_attempts = tuple(record.dispatch.attempt_id for record in records)
        if len(expected) != len(dispatches) or len(set(observed_attempts)) != len(observed_attempts):
            self._close_globally_inconsistent(_CauseCode.DUPLICATE)
            return
        fault = next(
            (
                code
                for record in records
                if (
                    code := self._reconciliation_fault(
                        record.dispatch,
                        expected.get(record.dispatch.attempt_id),
                        dispatches,
                    )
                )
                is not None
            ),
            None,
        )
        if fault is not None:
            self._close_globally_inconsistent(fault)
            return
        for record in records:
            self._accept(record)
        for attempt_id in expected.keys() - set(observed_attempts):
            dispatch = expected[attempt_id]
            self._close_dispatch(dispatch, _TaskInconsistent(dispatch.task, _causes(_CauseCode.MISSING)))

    @_serialized
    def request_cancellation(self) -> None:
        if self._closed:
            return
        self._require_opened()
        self._cancellation_requested = True
        cause = _causes(_CauseCode.CANCELLATION)
        self._states = {
            task: _TaskCancelled(task, cause) if isinstance(state, (_Planned, _Ready)) else state
            for task, state in self._states.items()
        }

    @_serialized
    def acknowledge_stop(self, dispatch: _Dispatch) -> _EvidenceAcceptance:
        self._require_opened()
        if self._closed:
            return _EvidenceAcceptance.REJECTED_STALE
        state = self._states.get(dispatch.task)
        if isinstance(state, _Dispatched) and state.dispatch == dispatch:
            self._states[dispatch.task] = _TaskCancelled(
                dispatch.task,
                _causes(_CauseCode.CANCELLATION, _CauseCode.STOP_ACKNOWLEDGED),
            )
            return _EvidenceAcceptance.ACCEPTED
        return _EvidenceAcceptance.REJECTED_STALE

    @_serialized
    def mark_transport_lost(self, dispatch: _Dispatch) -> _EvidenceAcceptance:
        self._require_opened()
        if self._closed:
            return _EvidenceAcceptance.REJECTED_STALE
        state = self._states.get(dispatch.task)
        if isinstance(state, _Dispatched) and state.dispatch == dispatch:
            self._invocation_lost = True
            self._states[dispatch.task] = _TaskLost(dispatch.task, _causes(_CauseCode.TRANSPORT_LOST))
            return _EvidenceAcceptance.ACCEPTED
        return _EvidenceAcceptance.REJECTED_STALE

    @_serialized
    def mark_inconsistent(self, code: _CauseCode) -> None:
        self._require_active()
        if code not in {
            _CauseCode.DUPLICATE,
            _CauseCode.UNKNOWN,
            _CauseCode.FOREIGN,
            _CauseCode.STALE,
            _CauseCode.SWAPPED,
            _CauseCode.CONTRADICTORY,
            _CauseCode.PLAN_MISMATCH,
        }:
            raise _LedgerStateError
        self._close_globally_inconsistent(code)

    @_serialized
    def mark_task_inconsistent(self, task: _TaskKey, code: _CauseCode) -> None:
        """Close one attributable task without widening context-derived dependencies."""
        self._require_active()
        if code not in {_CauseCode.MISSING, _CauseCode.DUPLICATE, _CauseCode.CONTRADICTORY}:
            raise _LedgerStateError
        state = self._states.get(task)
        if state is None:
            self._close_globally_inconsistent(_CauseCode.PLAN_MISMATCH)
        elif not _is_terminal(state):
            self._states[task] = _TaskInconsistent(task, _causes(code))

    @_serialized
    def mark_task_failed(self, task: _TaskKey) -> None:
        """Close one known pre-dispatch construction failure locally."""
        self._require_active()
        state = self._states.get(task)
        if state is None:
            self._close_globally_inconsistent(_CauseCode.PLAN_MISMATCH)
        elif isinstance(state, (_Planned, _Ready)):
            self._states[task] = _TaskFailed(task, _causes(_CauseCode.KNOWN_FAILURE))

    @_serialized
    def mark_cleanup_failed(self) -> None:
        self._require_active()
        self._cleanup_failed = True

    @_serialized
    def mark_cleanup_unconfirmed(self) -> None:
        self._require_active()
        self._cleanup_unconfirmed = True

    @_serialized
    def finish(
        self,
        *,
        datum_release_predicate: Callable[[_DatumId, T], bool] | None = None,
        group_release_predicate: Callable[[tuple[tuple[_DatumId, T], ...]], bool] = lambda _outputs: True,
    ) -> _AccountingResult[T]:
        self._require_active()
        if datum_release_predicate is not None:
            if self._datum_qualification:
                raise _LedgerStateError
            self._datum_release_predicate = datum_release_predicate
        self._advance_planned()
        self._close_unfinished()
        self._advance_planned()
        tasks = tuple(self._terminal_state(task) for task in self._plan.tasks)
        try:
            result = _reduce_result(
                self._plan,
                tasks,
                datum_release_predicate=self._qualifies,
                group_release_predicate=group_release_predicate,
                cancellation_requested=self._cancellation_requested,
                global_inconsistent=self._global_inconsistent,
                invocation_lost=self._invocation_lost,
                cleanup_failed=self._cleanup_failed,
                cleanup_unconfirmed=self._cleanup_unconfirmed,
            )
        except Exception:
            result = _construction_failed_result(self._plan, tasks)
        self._closed = True
        return result

    def _accept(self, record: _TerminalRecord[T]) -> _EvidenceAcceptance:
        self._require_opened()
        if self._closed:
            return _EvidenceAcceptance.REJECTED_STALE
        dispatch = record.dispatch
        state = self._states.get(dispatch.task)
        accepted = self._accepted_records.get(dispatch.attempt_id)
        if accepted is not None:
            try:
                identical = accepted == record
            except Exception:
                identical = False
            return _EvidenceAcceptance.IDEMPOTENT_STALE if identical else _EvidenceAcceptance.REJECTED_STALE
        if not isinstance(state, _Dispatched) or state.dispatch != dispatch:
            if state is None or not _is_terminal(state):
                expected = state.dispatch if isinstance(state, _Dispatched) else None
                fault = self._reconciliation_fault(dispatch, expected, (expected,) if expected is not None else ())
                self._close_globally_inconsistent(fault or _CauseCode.CONTRADICTORY)
            return _EvidenceAcceptance.REJECTED_STALE
        self._accepted_records[dispatch.attempt_id] = record
        match record:
            case _SuccessRecord(candidate=candidate):
                self._states[dispatch.task] = _TaskSucceeded(dispatch.task, candidate)
            case _FailureRecord():
                self._states[dispatch.task] = _TaskFailed(dispatch.task, _causes(_CauseCode.KNOWN_FAILURE))
            case unreachable:
                assert_never(unreachable)
        return _EvidenceAcceptance.ACCEPTED

    def _reconciliation_fault(
        self,
        observed: _Dispatch,
        expected: _Dispatch | None,
        batch: tuple[_Dispatch, ...],
    ) -> _CauseCode | None:
        if observed.task not in self._states:
            return _CauseCode.PLAN_MISMATCH
        if observed.invocation_id != self._invocation_id:
            return _CauseCode.FOREIGN
        if expected is None:
            if any(observed == state.dispatch for state in self._states.values() if isinstance(state, _Dispatched)):
                return _CauseCode.CONTRADICTORY
            if any(observed.task == dispatch.task and observed.row_token == dispatch.row_token for dispatch in batch):
                return _CauseCode.STALE
            return _CauseCode.UNKNOWN
        accepted = self._accepted_records.get(expected.attempt_id)
        state = self._states.get(expected.task)
        if accepted is not None or (state is not None and _is_terminal(state)):
            return None
        if observed == expected and isinstance(state, _Dispatched):
            return None
        if any(
            observed.task == dispatch.task
            and observed.row_token == dispatch.row_token
            and dispatch.attempt_id != expected.attempt_id
            for dispatch in batch
        ):
            return _CauseCode.SWAPPED
        if observed.attempt_id != expected.attempt_id:
            return (
                _CauseCode.STALE
                if observed.task == expected.task and observed.row_token == expected.row_token
                else _CauseCode.UNKNOWN
            )
        if observed.row_token != expected.row_token:
            return _CauseCode.FOREIGN
        return _CauseCode.SWAPPED if observed.task != expected.task else _CauseCode.CONTRADICTORY

    def _close_dispatch(self, dispatch: _Dispatch, outcome: _TaskOutcome[T]) -> None:
        state = self._states.get(dispatch.task)
        if isinstance(state, _Dispatched) and state.dispatch == dispatch:
            self._states[dispatch.task] = outcome
        else:
            self._close_globally_inconsistent(_CauseCode.STALE)

    def _close_globally_inconsistent(self, code: _CauseCode) -> None:
        self._global_inconsistent = True
        causes = _causes(code)
        self._states = {
            task: state if _is_terminal(state) else _TaskInconsistent(task, causes)
            for task, state in self._states.items()
        }

    def _advance_planned(self) -> None:
        changed = True
        while changed:
            changed = False
            for task in self._plan.tasks:
                state = self._states[task]
                if not isinstance(state, _Planned):
                    continue
                guard = self._readiness(task)
                if guard == "ready":
                    self._states[task] = _Ready(task)
                    changed = True
                elif guard == "blocked":
                    self._states[task] = _TaskBlocked(task, _causes(_CauseCode.PREREQUISITE))
                    changed = True

    def _readiness(self, task: _TaskKey) -> str:
        stage_index = self._plan.stages.index(task.stage)
        if stage_index:
            previous = _TaskKey(self._plan.stages[stage_index - 1], task.datum_id)
            previous_state = self._states[previous]
            if isinstance(previous_state, _TaskSucceeded):
                pass
            elif _is_terminal(previous_state):
                return "blocked"
            else:
                return "waiting"
        explicit_states = tuple(
            self._states[predecessor.prerequisite]
            for predecessor in self._plan.task_predecessors
            if predecessor.dependent == task
        )
        if any(_is_terminal(state) and not isinstance(state, _TaskSucceeded) for state in explicit_states):
            return "blocked"
        if any(not isinstance(state, _TaskSucceeded) for state in explicit_states):
            return "waiting"
        prerequisites = tuple(
            dependency.prerequisite for dependency in self._plan.dependencies if dependency.dependent == task.datum_id
        )
        prerequisite_states = tuple(self._datum_execution_state(datum_id) for datum_id in prerequisites)
        if any(state == "unsatisfied" for state in prerequisite_states):
            return "blocked"
        return "ready" if all(state == "satisfied" for state in prerequisite_states) else "waiting"

    def _datum_execution_state(self, datum_id: _DatumId) -> str:
        states = tuple(self._states[_TaskKey(stage, datum_id)] for stage in self._plan.stages)
        if all(isinstance(state, _TaskSucceeded) for state in states):
            final_state = self._states[_TaskKey(self._plan.stages[-1], datum_id)]
            if not isinstance(final_state, _TaskSucceeded):
                raise _LedgerStateError
            try:
                return "satisfied" if self._qualifies(datum_id, final_state.candidate) else "unsatisfied"
            except _ResultConstructionFailure:
                return "unsatisfied"
        if any(_is_terminal(state) and not isinstance(state, _TaskSucceeded) for state in states):
            return "unsatisfied"
        return "waiting"

    def _qualifies(self, datum_id: _DatumId, candidate: T) -> bool:
        if datum_id in self._datum_qualification:
            return self._datum_qualification[datum_id]
        try:
            qualified = self._datum_release_predicate(datum_id, candidate)
        except Exception as cause:
            del cause
            raise _ResultConstructionFailure from None
        if type(qualified) is not bool:
            raise _ResultConstructionFailure
        self._datum_qualification[datum_id] = qualified
        return qualified

    def _close_unfinished(self) -> None:
        for task, state in tuple(self._states.items()):
            if isinstance(state, _Dispatched):
                self._invocation_lost = True
                causes = _causes(
                    *(
                        (_CauseCode.CANCELLATION, _CauseCode.TRANSPORT_LOST)
                        if self._cancellation_requested
                        else (_CauseCode.TRANSPORT_LOST,)
                    )
                )
                self._states[task] = _TaskLost(task, causes)
            elif isinstance(state, (_Planned, _Ready)):
                self._states[task] = _TaskBlocked(task, _causes(_CauseCode.PREREQUISITE))

    def _terminal_state(self, task: _TaskKey) -> _TaskOutcome[T]:
        state = self._states[task]
        match state:
            case (
                _TaskSucceeded() | _TaskFailed() | _TaskCancelled() | _TaskLost() | _TaskBlocked() | _TaskInconsistent()
            ):
                return state
            case _Planned() | _Ready() | _Dispatched():
                raise _LedgerStateError
            case unreachable:
                assert_never(unreachable)

    def _next_identity(self) -> str:
        value = self._identity_factory()
        return self._claim_identity(value)

    def _claim_identity(self, value: object) -> str:
        if not isinstance(value, str) or not value or value in self._used_identities:
            raise _LedgerStateError
        self._used_identities.add(value)
        return value

    def _require_opened(self) -> None:
        if not self._opened:
            raise _LedgerStateError

    def _require_active(self) -> None:
        self._require_opened()
        if self._closed:
            raise _LedgerClosedError


class _LedgerClosedError(_LedgerStateError):
    pass


def _is_terminal(state: _TaskState[T]) -> bool:
    return isinstance(state, (_TaskSucceeded, _TaskFailed, _TaskCancelled, _TaskLost, _TaskBlocked, _TaskInconsistent))


def _causes(*codes: _CauseCode) -> _CauseSet:
    return _CauseSet(tuple(_TerminalCause(code) for code in codes))


def _cause_union(outcomes: Iterable[object]) -> _CauseSet:
    return reduce(operator.or_, map(_causes_of, outcomes), _CauseSet())


def _causes_of(outcome: object) -> _CauseSet:
    match outcome:
        case (
            _TaskFailed(causes=causes)
            | _TaskCancelled(causes=causes)
            | _TaskLost(causes=causes)
            | _TaskBlocked(causes=causes)
            | _TaskInconsistent(causes=causes)
            | _DatumFailed(causes=causes)
            | _DatumCancelled(causes=causes)
            | _DatumLost(causes=causes)
            | _DatumBlocked(causes=causes)
            | _DatumInconsistent(causes=causes)
            | _DependencyUnsatisfied(causes=causes)
            | _StageFailed(causes=causes)
            | _StageCancelled(causes=causes)
            | _StageLost(causes=causes)
            | _StageBlocked(causes=causes)
            | _StageInconsistent(causes=causes)
            | _GroupWithheld(causes=causes)
        ):
            return causes
        case _TaskSucceeded() | _DatumQualified() | _DependencySatisfied() | _StageSucceeded() | _GroupReleased():
            return _CauseSet()
        case _:
            raise _LedgerStateError


def _reduce_result(
    plan: _AccountingPlan,
    tasks: tuple[_TaskOutcome[T], ...],
    *,
    datum_release_predicate: Callable[[_DatumId, T], bool],
    group_release_predicate: Callable[[tuple[tuple[_DatumId, T], ...]], bool],
    cancellation_requested: bool,
    global_inconsistent: bool,
    invocation_lost: bool,
    cleanup_failed: bool,
    cleanup_unconfirmed: bool,
) -> _AccountingResult[T]:
    datums = tuple(_reduce_datum(plan, datum.id, tasks, datum_release_predicate) for datum in plan.datums)
    datum_by_id = {outcome.datum_id: outcome for outcome in datums}
    dependencies = tuple(
        _DependencySatisfied(dependency)
        if isinstance(datum_by_id[dependency.prerequisite], _DatumQualified)
        else _DependencyUnsatisfied(
            dependency,
            _cause_union((datum_by_id[dependency.prerequisite],)) | _causes(_CauseCode.PREREQUISITE),
        )
        for dependency in plan.dependencies
    )
    stages = tuple(_reduce_stage(stage, tasks) for stage in plan.stages)
    groups = _reduce_groups(
        plan,
        datum_by_id,
        group_release_predicate,
        cancellation_requested=cancellation_requested,
        global_inconsistent=global_inconsistent,
        invocation_lost=invocation_lost,
        cleanup_failed=cleanup_failed,
        cleanup_unconfirmed=cleanup_unconfirmed,
    )
    all_causes = _cause_union((*tasks, *datums, *dependencies, *stages, *groups))
    invocation = _reduce_invocation(
        groups,
        all_causes,
        cancellation_requested=cancellation_requested,
        global_inconsistent=global_inconsistent,
        invocation_lost=invocation_lost,
        cleanup_failed=cleanup_failed,
        cleanup_unconfirmed=cleanup_unconfirmed,
    )
    return _AccountingResult(tasks, datums, dependencies, stages, groups, invocation)


def _reduce_groups(
    plan: _AccountingPlan,
    datum_by_id: dict[_DatumId, _DatumOutcome[T]],
    group_release_predicate: Callable[[tuple[tuple[_DatumId, T], ...]], bool],
    *,
    cancellation_requested: bool,
    global_inconsistent: bool,
    invocation_lost: bool,
    cleanup_failed: bool,
    cleanup_unconfirmed: bool,
) -> tuple[_GroupOutcome[T], ...]:
    qualified = frozenset(outcome.datum_id for outcome in datum_by_id.values() if isinstance(outcome, _DatumQualified))
    embargoed = (
        global_inconsistent or invocation_lost or cancellation_requested or cleanup_failed or cleanup_unconfirmed
    )
    predicate_failed_groups: frozenset[_AtomicGroupKey] = frozenset()
    if embargoed:
        qualified = frozenset()
    else:
        predicate_failed_groups = _failed_group_predicates(
            plan,
            datum_by_id,
            qualified,
            group_release_predicate,
        )
        qualified -= frozenset(
            member for group in plan.atomic_groups if group.key in predicate_failed_groups for member in group.members
        )
    decision = _qualify_release(plan, qualified)
    groups = tuple(
        _reduce_group(plan, group.key, datum_by_id, decision.released_groups, predicate_failed_groups)
        for group in plan.atomic_groups
    )
    cleanup_code = (
        _CauseCode.CLEANUP_UNCONFIRMED if cleanup_unconfirmed else _CauseCode.CLEANUP_FAILED if cleanup_failed else None
    )
    if cleanup_code is not None:
        groups = tuple(_GroupWithheld(group.key, _causes(cleanup_code)) for group in plan.atomic_groups)
    return groups


def _reduce_invocation(
    groups: tuple[_GroupOutcome[T], ...],
    all_causes: _CauseSet,
    *,
    cancellation_requested: bool,
    global_inconsistent: bool,
    invocation_lost: bool,
    cleanup_failed: bool,
    cleanup_unconfirmed: bool,
) -> _InvocationOutcome[T]:
    if cleanup_unconfirmed:
        return _InvocationInconsistent(all_causes | _causes(_CauseCode.CLEANUP_UNCONFIRMED))
    if cleanup_failed:
        return _InvocationFailed(all_causes | _causes(_CauseCode.CLEANUP_FAILED))
    if global_inconsistent:
        return _InvocationInconsistent(all_causes | _causes(_CauseCode.CONTRADICTORY))
    if invocation_lost:
        return _InvocationLost(all_causes | _causes(_CauseCode.TRANSPORT_LOST))
    if cancellation_requested:
        return _InvocationCancelled(all_causes | _causes(_CauseCode.CANCELLATION))
    return _InvocationCompleted(groups)


def _construction_failed_result(
    plan: _AccountingPlan,
    tasks: tuple[_TaskOutcome[T], ...],
) -> _AccountingResult[T]:
    datums = tuple(_reduce_datum(plan, datum.id, tasks, lambda _datum_id, _candidate: True) for datum in plan.datums)
    datum_by_id = {outcome.datum_id: outcome for outcome in datums}
    dependencies = tuple(
        _DependencySatisfied(dependency)
        if isinstance(datum_by_id[dependency.prerequisite], _DatumQualified)
        else _DependencyUnsatisfied(
            dependency,
            _cause_union((datum_by_id[dependency.prerequisite],)) | _causes(_CauseCode.PREREQUISITE),
        )
        for dependency in plan.dependencies
    )
    stages = tuple(_reduce_stage(stage, tasks) for stage in plan.stages)
    causes = _causes(_CauseCode.RESULT_CONSTRUCTION_FAILED)
    groups = tuple(_GroupWithheld(group.key, causes) for group in plan.atomic_groups)
    return _AccountingResult(tasks, datums, dependencies, stages, groups, _InvocationFailed(causes))


def _reduce_datum(
    plan: _AccountingPlan,
    datum_id: _DatumId,
    tasks: tuple[_TaskOutcome[T], ...],
    release_predicate: Callable[[_DatumId, T], bool],
) -> _DatumOutcome[T]:
    child_tasks = tuple(outcome for outcome in tasks if outcome.task.datum_id == datum_id)
    if all(isinstance(outcome, _TaskSucceeded) for outcome in child_tasks):
        final_task = next(
            outcome
            for outcome in child_tasks
            if outcome.task.stage == plan.stages[-1] and isinstance(outcome, _TaskSucceeded)
        )
        return (
            _DatumQualified(datum_id, final_task.candidate)
            if release_predicate(datum_id, final_task.candidate)
            else _DatumFailed(datum_id, _causes(_CauseCode.RELEASE_PREDICATE_FAILED))
        )
    causes = _cause_union(child_tasks)
    if any(isinstance(outcome, _TaskInconsistent) for outcome in child_tasks):
        return _DatumInconsistent(datum_id, causes)
    if any(isinstance(outcome, _TaskLost) for outcome in child_tasks):
        return _DatumLost(datum_id, causes)
    if any(isinstance(outcome, _TaskCancelled) for outcome in child_tasks):
        return _DatumCancelled(datum_id, causes)
    if any(isinstance(outcome, _TaskFailed) for outcome in child_tasks):
        return _DatumFailed(datum_id, causes)
    if any(isinstance(outcome, _TaskBlocked) for outcome in child_tasks):
        return _DatumBlocked(datum_id, causes)
    raise _LedgerStateError


def _reduce_stage(stage: _StageId, tasks: tuple[_TaskOutcome[T], ...]) -> _StageOutcome:
    children = tuple(outcome for outcome in tasks if outcome.task.stage == stage)
    if all(isinstance(outcome, _TaskSucceeded) for outcome in children):
        return _StageSucceeded(stage)
    causes = _cause_union(children)
    if any(isinstance(outcome, _TaskInconsistent) for outcome in children):
        return _StageInconsistent(stage, causes)
    if any(isinstance(outcome, _TaskLost) for outcome in children):
        return _StageLost(stage, causes)
    if any(isinstance(outcome, _TaskCancelled) for outcome in children):
        return _StageCancelled(stage, causes)
    if any(isinstance(outcome, _TaskFailed) for outcome in children):
        return _StageFailed(stage, causes)
    if any(isinstance(outcome, _TaskBlocked) for outcome in children):
        return _StageBlocked(stage, causes)
    raise _LedgerStateError


def _reduce_group(
    plan: _AccountingPlan,
    group_key: _AtomicGroupKey,
    datum_by_id: dict[_DatumId, _DatumOutcome[T]],
    released_groups: frozenset[_AtomicGroupKey],
    predicate_failed_groups: frozenset[_AtomicGroupKey],
) -> _GroupOutcome[T]:
    group = next(group for group in plan.atomic_groups if group.key == group_key)
    member_outcomes = tuple(datum_by_id[datum.id] for datum in plan.datums if datum.id in group.members)
    if group.key in released_groups:
        outputs = tuple(
            (outcome.datum_id, outcome.candidate) for outcome in member_outcomes if isinstance(outcome, _DatumQualified)
        )
        if len(outputs) != len(group.members):
            return _GroupWithheld(group.key, _causes(_CauseCode.RELEASE_PREDICATE_FAILED))
        return _GroupReleased(group.key, outputs)
    if group.key in predicate_failed_groups:
        return _GroupWithheld(group.key, _causes(_CauseCode.RELEASE_PREDICATE_FAILED))
    return _GroupWithheld(group.key, _cause_union(member_outcomes) | _causes(_CauseCode.PREREQUISITE))


def _failed_group_predicates(
    plan: _AccountingPlan,
    datum_by_id: dict[_DatumId, _DatumOutcome[T]],
    qualified: frozenset[_DatumId],
    release_predicate: Callable[[tuple[tuple[_DatumId, T], ...]], bool],
) -> frozenset[_AtomicGroupKey]:
    """Evaluate complete groups before dependency propagation can release dependents."""
    failed: set[_AtomicGroupKey] = set()
    for group in plan.atomic_groups:
        if not frozenset(group.members).issubset(qualified):
            continue
        outputs = tuple(
            (datum_id, outcome.candidate)
            for datum_id in group.members
            if isinstance((outcome := datum_by_id[datum_id]), _DatumQualified)
        )
        if len(outputs) != len(group.members):
            raise _ResultConstructionFailure
        passed = release_predicate(outputs)
        if type(passed) is not bool:
            raise _ResultConstructionFailure
        if not passed:
            failed.add(group.key)
    return frozenset(failed)
