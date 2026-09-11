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
from typing import TYPE_CHECKING, Concatenate, Generic, ParamSpec, TypeAlias, TypeVar, assert_never, cast, final

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
from anonymizer.engine.execution.accounting_plan import (
    _AccountingPlan,
    _AtomicGroupKey,
    _DatumTaskSubject,
    _is_admitted_accounting_plan,
    _ScopeTaskSubject,
    _StageId,
    _TaskKey,
)
from anonymizer.engine.execution.accounting_release import _qualify_release
from anonymizer.engine.execution.graph import _DatumId

if TYPE_CHECKING:
    from anonymizer.engine.execution.phase10_inspection import (
        _Phase10InspectionRejected,
        _Phase10OwnerCapture,
        _Phase10ReasonCategory,
        _Phase10SemanticProfile,
        _Phase10Stage,
        _Phase10TerminalState,
    )

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
        self._mutation_sealed = False
        self._cancellation_requested = False
        self._global_inconsistent = False
        self._invocation_lost = False
        self._cleanup_failed = False
        self._cleanup_unconfirmed = False
        self._phase10_release_state = "not_entered"
        self._phase10_release_causes: tuple[_CauseCode, ...] = ()

    def __repr__(self) -> str:
        return "<private accounting ledger>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private accounting ledgers are not serializable")

    def _phase10_snapshot(self) -> _Phase10OwnerCapture | _Phase10InspectionRejected:
        """Fail closed when malformed private ledger state cannot be projected."""
        try:
            with self._lock:
                return self._phase10_snapshot_unchecked()
        except Exception:
            from anonymizer.engine.execution.phase10_inspection import (
                _Phase10InspectionRejected,
                _Phase10RejectionCode,
            )

            return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)

    def _phase10_snapshot_unchecked(self) -> _Phase10OwnerCapture | _Phase10InspectionRejected:
        """Validate and build one private accounting capture."""
        return self._phase10_build_capture()

    def _phase10_build_capture(self) -> _Phase10OwnerCapture | _Phase10InspectionRejected:
        """Issue one detached bounded-inspection snapshot under the ledger lock."""
        from anonymizer.engine.execution.phase10_inspection import (
            _MAX_REASON_CODES_PER_DIAGNOSTIC,
            _map_phase10_reason,
            _phase10_count_bucket,
            _phase10_new_builder_budget,
            _phase10_reserve_builder_row,
            _phase10_stage,
            _Phase10CaptureBoundary,
            _Phase10CleanupState,
            _Phase10CountBucket,
            _Phase10Diagnostic,
            _Phase10InspectionRejected,
            _Phase10LifecycleState,
            _Phase10OwnerCapture,
            _Phase10ReasonCategory,
            _Phase10ReconciliationState,
            _Phase10RejectionCode,
            _Phase10ReleaseState,
            _Phase10Snapshot,
            _Phase10Stage,
            _Phase10StageSummary,
            _Phase10SubjectKind,
            _Phase10TerminalState,
            _Phase10TerminalSummary,
        )

        lifecycle_flags = (
            self._opened,
            self._closed,
            self._mutation_sealed,
            self._cancellation_requested,
            self._global_inconsistent,
            self._invocation_lost,
            self._cleanup_failed,
            self._cleanup_unconfirmed,
        )
        if any(type(flag) is not bool for flag in lifecycle_flags):
            return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)
        if not self._opened:
            return _Phase10InspectionRejected(_Phase10RejectionCode.STATE_UNAVAILABLE)
        valid_release_state = type(self._phase10_release_state) is str and (
            self._phase10_release_state in {"released", "withheld"}
            if self._closed
            else self._phase10_release_state == "not_entered"
        )
        if (
            not _is_admitted_accounting_plan(self._plan)
            or type(self._states) is not dict
            or len(self._states) != len(self._plan.tasks)
            or any(actual is not expected for actual, expected in zip(self._states, self._plan.tasks, strict=True))
            or any(not _phase10_valid_task_state(task, self._states[task]) for task in self._plan.tasks)
            or not valid_release_state
            or type(self._phase10_release_causes) is not tuple
            or any(type(code) is not _CauseCode for code in self._phase10_release_causes)
            or len(self._phase10_release_causes) != len(set(self._phase10_release_causes))
            or tuple(code for code in _CauseCode if code in self._phase10_release_causes)
            != self._phase10_release_causes
            or (not self._closed and self._phase10_release_causes)
        ):
            return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)
        builder_budget = _phase10_new_builder_budget()
        if builder_budget is None:
            return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
        stage_groups: dict[_Phase10Stage, list[_StageId]] = {}
        for declared_stage in self._plan.stages:
            mapped_stage = _phase10_stage(declared_stage.value)
            if mapped_stage is None:
                return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)
            stage_groups.setdefault(mapped_stage, []).append(declared_stage)
        if len(stage_groups) > 8:
            return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)

        stage_summaries: list[_Phase10StageSummary] = []
        terminal_summaries: list[_Phase10TerminalSummary] = []
        diagnostics: list[_Phase10Diagnostic] = []
        any_dispatched = False
        any_terminal = False
        cleanup_state = (
            _Phase10CleanupState.UNCONFIRMED
            if self._cleanup_unconfirmed
            else _Phase10CleanupState.FAILED
            if self._cleanup_failed
            else _Phase10CleanupState.NOT_ENTERED
        )
        reconciliation_state = _Phase10ReconciliationState.NOT_ENTERED
        for mapped_stage, declared_stages in stage_groups.items():
            states = tuple(self._states[task] for task in self._plan.tasks if task.stage in declared_stages)
            any_dispatched = any_dispatched or any(isinstance(state, _Dispatched) for state in states)
            terminal_states = tuple(state for state in states if _is_terminal(state))
            any_terminal = any_terminal or bool(terminal_states)
            lifecycle = (
                _Phase10LifecycleState.TERMINAL
                if len(terminal_states) == len(states)
                else _Phase10LifecycleState.POST_DISPATCH
                if any(isinstance(state, _Dispatched) for state in states)
                else _Phase10LifecycleState.PRE_DISPATCH
                if any(isinstance(state, _Ready) for state in states)
                else _Phase10LifecycleState.OPENED
            )
            task_bucket = _phase10_count_bucket(len(states))
            if mapped_stage is None or task_bucket is None:
                return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)
            if not _phase10_reserve_builder_row(builder_budget):
                return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
            stage_summaries.append(_Phase10StageSummary(mapped_stage, lifecycle, task_bucket))
            for terminal_state in _Phase10TerminalState:
                matching = tuple(state for state in terminal_states if _phase10_task_terminal(state) is terminal_state)
                if not matching:
                    continue
                impact = _phase10_count_bucket(len(matching))
                if impact is None:
                    return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)
                if not _phase10_reserve_builder_row(builder_budget):
                    return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
                terminal_summaries.append(_Phase10TerminalSummary(mapped_stage, terminal_state, impact))
                reason_counts: dict[_Phase10ReasonCategory, int] = {}
                reason_codes: dict[_Phase10ReasonCategory, set[_CauseCode]] = {}
                for state in matching:
                    for code, category in _phase10_task_reason_entries(state):
                        reason_counts[category] = reason_counts.get(category, 0) + 1
                        reason_codes.setdefault(category, set()).add(code)
                for category in _Phase10ReasonCategory:
                    count = reason_counts.get(category, 0)
                    if not count:
                        continue
                    if len(reason_codes.get(category, set())) > _MAX_REASON_CODES_PER_DIAGNOSTIC:
                        return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
                    reason_impact = _phase10_count_bucket(count)
                    if reason_impact is None:
                        return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)
                    if not _phase10_reserve_builder_row(builder_budget):
                        return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
                    diagnostics.append(
                        _Phase10Diagnostic(
                            _Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED,
                            mapped_stage,
                            terminal_state,
                            category,
                            reason_impact,
                            _Phase10ReconciliationState.RECONCILED,
                            cleanup_state,
                        )
                    )
        if len(terminal_summaries) > 48 or len(diagnostics) > 64:
            return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
        if self._global_inconsistent:
            reconciliation_state = _Phase10ReconciliationState.INCONSISTENT
        elif any_dispatched:
            reconciliation_state = _Phase10ReconciliationState.PENDING
        elif any_terminal:
            reconciliation_state = _Phase10ReconciliationState.RECONCILED
        boundary = (
            _Phase10CaptureBoundary.INVOCATION_CLOSED
            if self._closed
            else _Phase10CaptureBoundary.PRE_REDUCTION_CLEANUP_TERMINAL
            if cleanup_state is not _Phase10CleanupState.NOT_ENTERED
            else _Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED
            if any_terminal
            else _Phase10CaptureBoundary.POST_DISPATCH
            if any_dispatched
            else _Phase10CaptureBoundary.PRE_DISPATCH
            if any(isinstance(state, _Ready) for state in self._states.values())
            else _Phase10CaptureBoundary.INVOCATION_OPENED
        )
        lifecycle = (
            _Phase10LifecycleState.CLOSED
            if self._closed
            else _Phase10LifecycleState.CLEANUP_TERMINAL
            if cleanup_state is not _Phase10CleanupState.NOT_ENTERED
            else _Phase10LifecycleState.TERMINAL
            if any_terminal
            else _Phase10LifecycleState.POST_DISPATCH
            if any_dispatched
            else _Phase10LifecycleState.PRE_DISPATCH
            if boundary is _Phase10CaptureBoundary.PRE_DISPATCH
            else _Phase10LifecycleState.OPENED
        )
        if cleanup_state is not _Phase10CleanupState.NOT_ENTERED:
            cleanup_terminal = (
                _Phase10TerminalState.FAILED
                if cleanup_state is _Phase10CleanupState.FAILED
                else _Phase10TerminalState.INCONSISTENT
            )
            if not _phase10_reserve_builder_row(builder_budget) or not _phase10_reserve_builder_row(builder_budget):
                return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
            stage_summaries.append(
                _Phase10StageSummary(
                    _Phase10Stage.CLEANUP,
                    _Phase10LifecycleState.CLEANUP_TERMINAL,
                    _Phase10CountBucket.ONE,
                )
            )
            terminal_summaries.append(
                _Phase10TerminalSummary(_Phase10Stage.CLEANUP, cleanup_terminal, _Phase10CountBucket.ONE)
            )
            if not _phase10_reserve_builder_row(builder_budget):
                return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
            diagnostics.append(
                _Phase10Diagnostic(
                    boundary,
                    _Phase10Stage.CLEANUP,
                    cleanup_terminal,
                    (
                        _Phase10ReasonCategory.CLEANUP_FAILED
                        if cleanup_state is _Phase10CleanupState.FAILED
                        else _Phase10ReasonCategory.CLEANUP_UNCONFIRMED
                    ),
                    _Phase10CountBucket.ONE,
                    reconciliation_state,
                    cleanup_state,
                )
            )
        if self._closed and self._phase10_release_state == "withheld":
            if not _phase10_reserve_builder_row(builder_budget) or not _phase10_reserve_builder_row(builder_budget):
                return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
            stage_summaries.append(
                _Phase10StageSummary(
                    _Phase10Stage.PUBLICATION,
                    _Phase10LifecycleState.RELEASE_TERMINAL,
                    _Phase10CountBucket.ONE,
                )
            )
            terminal_summaries.append(
                _Phase10TerminalSummary(
                    _Phase10Stage.PUBLICATION,
                    _Phase10TerminalState.WITHHELD,
                    _Phase10CountBucket.ONE,
                )
            )
            categories: dict[_Phase10ReasonCategory, set[_CauseCode]] = {}
            release_codes = self._phase10_release_causes
            if _CauseCode.STOP_ACKNOWLEDGED in release_codes:
                release_codes = (_CauseCode.STOP_ACKNOWLEDGED,)
            elif _CauseCode.TRANSPORT_LOST in release_codes:
                release_codes = (_CauseCode.TRANSPORT_LOST,)
            elif _CauseCode.CANCELLATION in release_codes:
                release_codes = (_CauseCode.CANCELLATION,)
            else:
                release_codes = tuple(
                    code
                    for code in release_codes
                    if code
                    in {
                        _CauseCode.RELEASE_PREDICATE_FAILED,
                        _CauseCode.CLEANUP_FAILED,
                        _CauseCode.CLEANUP_UNCONFIRMED,
                    }
                )
            for code in release_codes:
                categories.setdefault(_map_phase10_reason(code), set()).add(code)
            for category in _Phase10ReasonCategory:
                codes = categories.get(category)
                if not codes:
                    continue
                if any(item.reason_category is category for item in diagnostics):
                    continue
                if len(codes) > _MAX_REASON_CODES_PER_DIAGNOSTIC:
                    return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
                if not _phase10_reserve_builder_row(builder_budget):
                    return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
                diagnostics.append(
                    _Phase10Diagnostic(
                        boundary,
                        _Phase10Stage.PUBLICATION,
                        _Phase10TerminalState.WITHHELD,
                        category,
                        _Phase10CountBucket.ONE,
                        reconciliation_state,
                        cleanup_state,
                    )
                )
        if len(stage_summaries) > 8 or len(terminal_summaries) > 48 or len(diagnostics) > 64:
            return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
        snapshot = _Phase10Snapshot(
            tuple(stage_summaries),
            tuple(terminal_summaries),
            reconciliation_state,
            cleanup_state,
            _Phase10ReleaseState(self._phase10_release_state),
        )
        diagnostics = [
            _Phase10Diagnostic(
                boundary,
                item.stage,
                item.terminal_state,
                item.reason_category,
                item.impact_count_bucket,
                reconciliation_state,
                cleanup_state,
            )
            for item in diagnostics
        ]
        return _Phase10OwnerCapture(
            _Phase10SubjectKind.INVOCATION_SNAPSHOT,
            _phase10_accounting_profile(self._plan),
            boundary,
            lifecycle,
            snapshot,
            tuple(diagnostics),
        )

    @_serialized
    def open(self) -> None:
        if self._opened or self._closed:
            raise _LedgerStateError
        self._invocation_id = _InvocationId(self._next_identity())
        self._opened = True

    @_serialized
    def import_terminal_outcomes(self, outcomes: tuple[object, ...]) -> None:
        """Import an exact earlier-phase terminal prefix into an expanded plan.

        The compiler may append a later-phase scope task to an already closed
        plan.  Re-reducing that plan must retain the original terminal records,
        rather than manufacturing new prerequisite failures for them.
        """
        self._require_active()
        if not isinstance(outcomes, tuple) or not all(
            isinstance(
                outcome,
                (_TaskSucceeded, _TaskFailed, _TaskCancelled, _TaskLost, _TaskBlocked, _TaskInconsistent),
            )
            for outcome in outcomes
        ):
            raise _LedgerStateError
        typed_outcomes = tuple(cast(_TaskOutcome[T], outcome) for outcome in outcomes)
        if len({outcome.task for outcome in typed_outcomes}) != len(typed_outcomes) or any(
            outcome.task not in self._states or not isinstance(self._states[outcome.task], _Planned)
            for outcome in typed_outcomes
        ):
            raise _LedgerStateError
        for outcome in typed_outcomes:
            self._states[outcome.task] = outcome

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
    def mark_task_succeeded(self, task: _TaskKey, candidate: T) -> None:
        """Close verified no-work without manufacturing a dispatch attempt."""
        self._require_active()
        state = self._states.get(task)
        if state is None:
            self._close_globally_inconsistent(_CauseCode.PLAN_MISMATCH)
        elif isinstance(state, (_Planned, _Ready)):
            self._states[task] = _TaskSucceeded(task, candidate)

    @_serialized
    def mark_task_blocked(self, task: _TaskKey) -> None:
        """Close a known non-dispatch prerequisite gate without an attempt."""
        self._require_active()
        state = self._states.get(task)
        if state is None:
            self._close_globally_inconsistent(_CauseCode.PLAN_MISMATCH)
        elif isinstance(state, (_Planned, _Ready)):
            self._states[task] = _TaskBlocked(task, _causes(_CauseCode.PREREQUISITE))

    @_serialized
    def mark_cleanup_failed(self) -> None:
        self._require_active()
        self._cleanup_failed = True

    @_serialized
    def mark_cleanup_unconfirmed(self) -> None:
        self._require_active()
        self._cleanup_unconfirmed = True

    @_serialized
    def seal_mutation(self) -> None:
        """Freeze external lifecycle transitions before cleanup attestation."""
        self._require_active()
        self._mutation_sealed = True

    @_serialized
    def record_cleanup_unconfirmed_after_seal(self) -> None:
        """Record failed publication-critical cleanup without reopening tasks."""
        self._require_opened()
        if self._closed or not self._mutation_sealed:
            raise _LedgerStateError
        self._cleanup_unconfirmed = True

    @_serialized
    def finish(
        self,
        *,
        datum_release_predicate: Callable[[_DatumId, T], bool] | None = None,
        group_release_predicate: Callable[[tuple[tuple[_DatumId, T], ...]], bool] = lambda _outputs: True,
    ) -> _AccountingResult[T]:
        self._require_opened()
        if self._closed:
            raise _LedgerClosedError
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
        self._phase10_release_state = (
            "released"
            if isinstance(result.invocation, _InvocationCompleted)
            and all(isinstance(group, _GroupReleased) for group in result.groups)
            else "withheld"
        )
        release_codes = {
            cause.code
            for outcome in (*result.groups, result.invocation)
            for cause in getattr(outcome, "causes", ())
            if type(cause) is _TerminalCause and type(cause.code) is _CauseCode
        }
        self._phase10_release_causes = tuple(code for code in _CauseCode if code in release_codes)
        self._closed = True
        return result

    def _accept(self, record: _TerminalRecord[T]) -> _EvidenceAcceptance:
        self._require_opened()
        if self._closed or self._mutation_sealed:
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
        if isinstance(task.subject, _DatumTaskSubject):
            stage_index = self._plan.stages.index(task.stage)
            if stage_index:
                previous = _TaskKey(self._plan.stages[stage_index - 1], task.subject)
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
        if isinstance(task.subject, _ScopeTaskSubject):
            return "ready"
        prerequisites = tuple(
            dependency.prerequisite
            for dependency in self._plan.dependencies
            if dependency.dependent == task.subject.datum_id
        )
        prerequisite_states = tuple(self._datum_execution_state(datum_id) for datum_id in prerequisites)
        if any(state == "unsatisfied" for state in prerequisite_states):
            return "blocked"
        return "ready" if all(state == "satisfied" for state in prerequisite_states) else "waiting"

    def _datum_execution_state(self, datum_id: _DatumId) -> str:
        subject = _DatumTaskSubject(datum_id)
        states = tuple(self._states[_TaskKey(stage, subject)] for stage in self._plan.stages)
        if all(isinstance(state, _TaskSucceeded) for state in states):
            final_state = self._states[_TaskKey(self._plan.stages[-1], subject)]
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
        if self._closed or self._mutation_sealed:
            raise _LedgerClosedError


def _phase10_task_terminal(state: object) -> _Phase10TerminalState | None:
    from anonymizer.engine.execution.phase10_inspection import _Phase10TerminalState

    if type(state) is _TaskInconsistent:
        return _Phase10TerminalState.INCONSISTENT
    if type(state) is _TaskLost:
        return _Phase10TerminalState.LOST
    if type(state) is _TaskCancelled:
        return _Phase10TerminalState.CANCELLED
    if type(state) is _TaskFailed:
        return _Phase10TerminalState.FAILED
    if type(state) is _TaskBlocked:
        return _Phase10TerminalState.BLOCKED
    if type(state) is _TaskSucceeded:
        return _Phase10TerminalState.SUCCEEDED
    return None


def _phase10_valid_cause_set(value: object, *, max_items: int = len(_CauseCode)) -> bool:
    if type(value) is not _CauseSet or type(value.items) is not tuple or len(value.items) > max_items:
        return False
    codes: list[_CauseCode] = []
    for cause in value.items:
        if type(cause) is not _TerminalCause or type(cause.code) is not _CauseCode:
            return False
        codes.append(cause.code)
    return value == _causes(*codes)


def _phase10_valid_task_outcome(task: object, outcome: object) -> bool:
    if type(task) is not _TaskKey or type(outcome) not in {
        _TaskSucceeded,
        _TaskFailed,
        _TaskCancelled,
        _TaskLost,
        _TaskBlocked,
        _TaskInconsistent,
    }:
        return False
    if getattr(outcome, "task", None) is not task:
        return False
    if type(outcome) is _TaskSucceeded:
        return True
    causes = getattr(outcome, "causes", None)
    if not _phase10_valid_cause_set(causes, max_items=4):
        return False
    codes = {cause.code for cause in cast(_CauseSet, causes)}
    if type(outcome) is _TaskCancelled:
        return _CauseCode.CANCELLATION in codes and codes <= {
            _CauseCode.CANCELLATION,
            _CauseCode.STOP_ACKNOWLEDGED,
        }
    if type(outcome) is _TaskLost:
        return _CauseCode.TRANSPORT_LOST in codes and _CauseCode.STOP_ACKNOWLEDGED not in codes
    if type(outcome) is _TaskBlocked:
        return _CauseCode.PREREQUISITE in codes and not codes.intersection(
            {_CauseCode.CANCELLATION, _CauseCode.STOP_ACKNOWLEDGED, _CauseCode.TRANSPORT_LOST}
        )
    return not codes.intersection({_CauseCode.CANCELLATION, _CauseCode.STOP_ACKNOWLEDGED, _CauseCode.TRANSPORT_LOST})


def _phase10_valid_task_state(task: object, state: object) -> bool:
    if type(task) is not _TaskKey:
        return False
    if type(state) in {_Planned, _Ready}:
        return getattr(state, "task", None) is task
    if type(state) is _Dispatched:
        return type(state.dispatch) is _Dispatch and state.dispatch.task is task
    return _phase10_valid_task_outcome(task, state)


def _phase10_task_reason_entries(state: object) -> tuple[tuple[_CauseCode, _Phase10ReasonCategory], ...]:
    from anonymizer.engine.execution.phase10_inspection import _map_phase10_reason

    causes = getattr(state, "causes", None)
    if not _phase10_valid_cause_set(causes):
        return ()
    codes = tuple(cause.code for cause in cast(_CauseSet, causes))
    acknowledged = _CauseCode.STOP_ACKNOWLEDGED in codes
    lost = _CauseCode.TRANSPORT_LOST in codes
    return tuple(
        (code, _map_phase10_reason(code))
        for code in codes
        if not ((acknowledged or lost) and code is _CauseCode.CANCELLATION)
    )


def _phase10_task_reason_categories(state: object) -> tuple[_Phase10ReasonCategory, ...]:
    return tuple(category for _code, category in _phase10_task_reason_entries(state))


def _phase10_accounting_profile(plan: _AccountingPlan) -> _Phase10SemanticProfile:
    from anonymizer.engine.execution.phase10_inspection import _Phase10SemanticProfile

    stages = tuple(stage.value for stage in plan.stages)
    if any(stage.startswith("phase8-") for stage in stages):
        return _Phase10SemanticProfile.GROUPED_REWRITE_V1
    if any(stage.startswith("phase7-") for stage in stages):
        return _Phase10SemanticProfile.SUBSTITUTE_V1
    if "transform" in stages or "verify" in stages:
        return _Phase10SemanticProfile.REDACT_V1
    return _Phase10SemanticProfile.TARGET_CONTEXT_V1


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
    child_tasks = tuple(
        outcome
        for outcome in tasks
        if isinstance(outcome.task.subject, _DatumTaskSubject) and outcome.task.subject.datum_id == datum_id
    )
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
