# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen, content-free accounting for one Phase 8 grouped operation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from anonymizer.engine.execution.phase8_admission import (
    _compile_group_operation_plan,
    _Phase8GroupOperationPlan,
    _Phase8Stage,
)
from anonymizer.engine.execution.phase8_cleanup import (
    _is_phase8_cleanup_receipt,
    _Phase8CleanupComponent,
    _Phase8CleanupPhase,
    _Phase8CleanupReceipt,
    _Phase8CleanupStatus,
)
from anonymizer.engine.execution.phase8_contract import _load_phase8_contract
from anonymizer.engine.execution.phase8_validation import _Phase8Metric, _validate_complete_revisions


class _Phase8Terminal:
    """Content-free terminal base: only a stage and bounded cause code survive."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"


class _Phase8TerminalKind(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"
    BLOCKED = "blocked"
    INCONSISTENT = "inconsistent"


class _Phase8Reason(str, Enum):
    """Closed content-free reason vocabulary for Phase 8 accounting."""

    ANALYSIS_INVALID = "analysis_invalid"
    ANALYSIS_RECONCILIATION = "analysis_reconciliation"
    ANALYSIS_STATE_MISSING = "analysis_state_missing"
    BACKEND_FAILURE = "backend_failure"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    CANCELLATION = "cancellation"
    CANDIDATE_RECONCILIATION = "candidate_reconciliation"
    EVALUATION_INVALID = "evaluation_invalid"
    EVALUATION_RECONCILIATION = "evaluation_reconciliation"
    GROUP_OPERATION_REUSED = "group_operation_reused"
    INCOMPLETE_GROUP = "incomplete_group"
    INVALID_EVALUATION = "invalid_evaluation"
    INVALID_GROUP_INPUT = "invalid_group_input"
    INVALID_REPAIR_BOUND = "invalid_repair_bound"
    INVOCATION_INCONSISTENT = "invocation_inconsistent"
    MISSING_BASELINE = "missing_baseline"
    NO_REPAIR_NEEDED = "no_repair_needed"
    OPERATION_CORRELATION_MISMATCH = "operation_correlation_mismatch"
    PREREQUISITE = "prerequisite"
    REPAIR_EXHAUSTED = "repair_exhausted"
    REPAIR_MEMBERS = "repair_members"
    REPAIR_RECONCILIATION = "repair_reconciliation"
    RETIRED_CORRELATION_TOKEN = "retired_correlation_token"
    REVISION_INVALID = "revision_invalid"
    REVISION_LIMIT = "revision_limit"
    REWRITE_MEMBERS = "rewrite_members"
    REWRITE_RECONCILIATION = "rewrite_reconciliation"
    ROUTE_NOT_SELECTED = "route_not_selected"
    TRANSPORT_LOST = "transport_lost"
    UNATTRIBUTABLE_PROVIDER_FAILURE = "unattributable_provider_failure"


def _require_reason(code: object) -> None:
    if not isinstance(code, _Phase8Reason):
        raise TypeError("terminal reason must be a closed Phase 8 reason code")


class _Phase8CodedTerminal(_Phase8Terminal):
    __slots__ = ()
    code: _Phase8Reason

    def __post_init__(self) -> None:
        _require_reason(self.code)


@dataclass(frozen=True, slots=True, repr=False)
class _StageSucceeded(_Phase8Terminal):
    stage: _Phase8Stage


@dataclass(frozen=True, slots=True, repr=False)
class _StageFailed(_Phase8CodedTerminal):
    stage: _Phase8Stage
    code: _Phase8Reason


@dataclass(frozen=True, slots=True, repr=False)
class _StageCancelled(_Phase8CodedTerminal):
    stage: _Phase8Stage
    code: _Phase8Reason


@dataclass(frozen=True, slots=True, repr=False)
class _StageLost(_Phase8CodedTerminal):
    stage: _Phase8Stage
    code: _Phase8Reason


@dataclass(frozen=True, slots=True, repr=False)
class _StageBlocked(_Phase8CodedTerminal):
    stage: _Phase8Stage
    code: _Phase8Reason


@dataclass(frozen=True, slots=True, repr=False)
class _StageInconsistent(_Phase8CodedTerminal):
    stage: _Phase8Stage
    code: _Phase8Reason


def _terminal_name(terminal: _Phase8Terminal) -> _Phase8TerminalKind:
    if isinstance(terminal, _StageSucceeded):
        return _Phase8TerminalKind.SUCCEEDED
    if isinstance(terminal, _StageFailed):
        return _Phase8TerminalKind.FAILED
    if isinstance(terminal, _StageCancelled):
        return _Phase8TerminalKind.CANCELLED
    if isinstance(terminal, _StageLost):
        return _Phase8TerminalKind.LOST
    if isinstance(terminal, _StageBlocked):
        return _Phase8TerminalKind.BLOCKED
    return _Phase8TerminalKind.INCONSISTENT


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8AttemptReceipt:
    """Closed content-free receipt for one selected operation attempt."""

    group_id: object
    stage: _Phase8Stage
    terminal: _Phase8TerminalKind


@dataclass(slots=True, repr=False)
class _Phase8OperationLedger:
    """One-shot terminal ledger for a precompiled complete-group route."""

    plan: _Phase8GroupOperationPlan
    _terminals: dict[_Phase8Stage, _Phase8Terminal] = field(default_factory=dict)
    _attempts: dict[_Phase8Stage, _Phase8AttemptReceipt] = field(default_factory=dict)
    _dispatched: set[_Phase8Stage] = field(default_factory=set)
    _retired: bool = False

    @property
    def is_closed(self) -> bool:
        return not self._retired and len(self._terminals) == len(self.plan.stages)

    def discard(self) -> None:
        """Erase retained operation evidence and permanently close the ledger."""
        self._terminals.clear()
        self._attempts.clear()
        self._dispatched.clear()
        self._retired = True

    def terminal(self, stage: _Phase8Stage) -> _Phase8Terminal | None:
        if self._retired:
            return None
        return self._terminals.get(stage)

    def reason(self, stage: _Phase8Stage) -> _Phase8Reason | None:
        terminal = self.terminal(stage)
        return getattr(terminal, "code", None)

    def attempt_count(self, stage: _Phase8Stage) -> int:
        return int(stage in self._dispatched)

    def dispatch(self, stage: _Phase8Stage) -> bool:
        if self._retired or stage not in self.plan.stages or stage in self._terminals or stage in self._dispatched:
            return False
        position = self.plan.stages.index(stage)
        if any(
            not isinstance(self._terminals.get(previous), _StageSucceeded) for previous in self.plan.stages[:position]
        ):
            return False
        self._dispatched.add(stage)
        return True

    def succeed(self, stage: _Phase8Stage) -> bool:
        if stage not in self.plan.stages or stage in self._terminals:
            return False
        if stage not in self._dispatched and not self.dispatch(stage):
            return False
        return self._close(stage, _StageSucceeded(stage))

    def fail(self, stage: _Phase8Stage, code: _Phase8Reason) -> bool:
        return self._terminal_failure(stage, _StageFailed, code)

    def inconsistent(self, stage: _Phase8Stage, code: _Phase8Reason) -> bool:
        return self._terminal_failure(stage, _StageInconsistent, code)

    def lost(self, stage: _Phase8Stage, code: _Phase8Reason) -> bool:
        return self._terminal_failure(stage, _StageLost, code)

    def cancel(self, stage: _Phase8Stage, *, trusted_stop: bool, dispatched: bool) -> bool:
        if stage not in self.plan.stages or stage in self._terminals:
            return False
        if dispatched and stage not in self._dispatched and not self.dispatch(stage):
            return False
        if not dispatched:
            position = self.plan.stages.index(stage)
            if any(
                not isinstance(self._terminals.get(previous), _StageSucceeded)
                for previous in self.plan.stages[:position]
            ):
                return False
        if stage in self._terminals:
            return False
        terminal: _Phase8Terminal = (
            _StageCancelled(stage, _Phase8Reason.CANCELLATION)
            if trusted_stop
            else _StageLost(stage, _Phase8Reason.TRANSPORT_LOST)
        )
        self._close(stage, terminal)
        self._block_descendants(stage, _Phase8Reason.PREREQUISITE)
        return True

    def block(self, stage: _Phase8Stage, code: _Phase8Reason) -> bool:
        if stage not in self.plan.stages or stage in self._terminals or stage in self._dispatched:
            return False
        position = self.plan.stages.index(stage)
        if any(previous not in self._terminals for previous in self.plan.stages[:position]):
            return False
        self._close(stage, _StageBlocked(stage, code))
        self._block_descendants(stage, _Phase8Reason.PREREQUISITE)
        return True

    def close_zero_route(self) -> None:
        self._block_from(_Phase8Stage.rewrite(), _Phase8Reason.ROUTE_NOT_SELECTED)

    def close_pass(self, evaluation: _Phase8Stage) -> None:
        self._block_descendants(evaluation, _Phase8Reason.NO_REPAIR_NEEDED)

    def _terminal_failure(
        self,
        stage: _Phase8Stage,
        terminal_type: Callable[[_Phase8Stage, _Phase8Reason], _Phase8Terminal],
        code: _Phase8Reason,
    ) -> bool:
        if stage not in self.plan.stages or stage in self._terminals:
            return False
        if stage not in self._dispatched and not self.dispatch(stage):
            return False
        self._close(stage, terminal_type(stage, code))
        self._block_descendants(stage, _Phase8Reason.PREREQUISITE)
        return True

    def _close(self, stage: _Phase8Stage, terminal: _Phase8Terminal) -> bool:
        if stage in self._terminals:
            return False
        self._terminals[stage] = terminal
        if stage in self._dispatched:
            self._attempts[stage] = _Phase8AttemptReceipt(self.plan.group_id, stage, _terminal_name(terminal))
        return True

    def _block_from(self, stage: _Phase8Stage, code: _Phase8Reason) -> None:
        index = self.plan.stages.index(stage)
        for descendant in self.plan.stages[index:]:
            self._close(descendant, _StageBlocked(descendant, code))

    def _block_descendants(self, stage: _Phase8Stage, code: _Phase8Reason) -> None:
        index = self.plan.stages.index(stage)
        for descendant in self.plan.stages[index + 1 :]:
            self._close(descendant, _StageBlocked(descendant, code))


class _Phase8GroupTerminal:
    __slots__ = ()


class _Phase8CodedGroupTerminal(_Phase8GroupTerminal):
    __slots__ = ()
    code: _Phase8Reason

    def __post_init__(self) -> None:
        _require_reason(self.code)


@dataclass(frozen=True, slots=True, repr=False)
class _GroupSucceeded(_Phase8GroupTerminal):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class _GroupFailed(_Phase8CodedGroupTerminal):
    code: _Phase8Reason


@dataclass(frozen=True, slots=True, repr=False)
class _GroupCancelled(_Phase8CodedGroupTerminal):
    code: _Phase8Reason


@dataclass(frozen=True, slots=True, repr=False)
class _GroupLost(_Phase8CodedGroupTerminal):
    code: _Phase8Reason


@dataclass(frozen=True, slots=True, repr=False)
class _GroupBlocked(_Phase8CodedGroupTerminal):
    code: _Phase8Reason


@dataclass(frozen=True, slots=True, repr=False)
class _GroupInconsistent(_Phase8CodedGroupTerminal):
    code: _Phase8Reason
    invocation_global: bool = False


class _Phase8FaultKind(str, Enum):
    FAILED = "failed"
    INCONSISTENT = "inconsistent"
    CANCELLED = "cancelled"
    LOST = "lost"


class _Phase8OperationFault(RuntimeError):
    """Content-free classified fault raised by a selected operation callback."""

    __slots__ = ("kind", "code", "invocation_global", "trusted_stop")

    def __init__(
        self,
        kind: _Phase8FaultKind,
        code: _Phase8Reason,
        *,
        invocation_global: bool = False,
        trusted_stop: bool = False,
    ) -> None:
        _require_reason(code)
        super().__init__(code.value)
        self.kind = kind
        self.code = code
        self.invocation_global = invocation_global
        self.trusted_stop = trusted_stop


def _group_precedence(terminal: _Phase8GroupTerminal) -> int:
    if isinstance(terminal, _GroupInconsistent):
        return 4
    if isinstance(terminal, _GroupLost):
        return 3
    if isinstance(terminal, _GroupCancelled):
        return 2
    if isinstance(terminal, _GroupFailed):
        return 1
    return 0


@dataclass(slots=True, repr=False)
class _Phase8InvocationLedger:
    """Scheduling-only reducer: local faults continue; global terminals embargo."""

    group_terminals: list[_Phase8GroupTerminal] = field(default_factory=list)
    global_embargo: bool = False

    def admit(self, terminal: _Phase8GroupTerminal) -> bool:
        """Record once and return whether another baseline-ready group may dispatch."""
        self.group_terminals.append(terminal)
        if isinstance(terminal, (_GroupCancelled, _GroupLost)) or (
            isinstance(terminal, _GroupInconsistent) and terminal.invocation_global
        ):
            self.global_embargo = True
        return not self.global_embargo

    def aggregate(self) -> _Phase8GroupTerminal:
        if not self.group_terminals:
            return _GroupBlocked(_Phase8Reason.PREREQUISITE)
        non_success = [terminal for terminal in self.group_terminals if not isinstance(terminal, _GroupSucceeded)]
        return max(non_success, key=_group_precedence) if non_success else _GroupSucceeded()


@dataclass(slots=True, repr=False)
class _Phase8CandidateStore:
    """Runtime-only candidate-bearing state, deliberately outside the ledger."""

    revisions: dict[object, str]


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8LifecycleExecution:
    """Content-free terminal receipt returned by the existing Phase 4 seam."""

    released: tuple[tuple[object, str], ...]
    terminal_group_states: tuple[str, ...]
    global_embargo: bool
    pre_reduction_cleanup: _Phase8CleanupReceipt | None = None
    post_reduction_cleanup: _Phase8CleanupReceipt | None = None

    @property
    def cleanup_verified(self) -> bool:
        """Require two sealed runtime receipts bound to the same invocation."""
        pre = self.pre_reduction_cleanup
        post = self.post_reduction_cleanup
        if pre is None or post is None or pre.identity is not post.identity:
            return False
        return (
            _is_phase8_cleanup_receipt(
                pre,
                identity=pre.identity,
                phase=_Phase8CleanupPhase.PRE_REDUCTION,
                component=_Phase8CleanupComponent.RUNTIME,
            )
            and _is_phase8_cleanup_receipt(
                post,
                identity=pre.identity,
                phase=_Phase8CleanupPhase.POST_REDUCTION,
                component=_Phase8CleanupComponent.RUNTIME,
            )
            and pre.status is _Phase8CleanupStatus.VERIFIED
            and post.status is _Phase8CleanupStatus.VERIFIED
            and post.retained_candidate_cell_count == len(self.released)
            and post.provisional_revision_reference_count == 0
            and post.withheld_candidate_reference_count == 0
        )


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8GroupOutcome:
    terminal: _Phase8GroupTerminal
    revisions: dict[object, str] | None
    repair_iterations: int
    ledger: _Phase8OperationLedger

    @property
    def state(self) -> str:
        """Compatibility projection; accounting itself uses typed terminals."""
        if isinstance(self.terminal, _GroupSucceeded):
            return "succeeded"
        if isinstance(self.terminal, _GroupFailed):
            return "failed"
        if isinstance(self.terminal, _GroupCancelled):
            return "cancelled"
        if isinstance(self.terminal, _GroupLost):
            return "lost"
        if isinstance(self.terminal, _GroupBlocked):
            return "blocked"
        return "inconsistent"


def _failed(ledger: _Phase8OperationLedger, code: _Phase8Reason, repairs: int) -> _Phase8GroupOutcome:
    return _Phase8GroupOutcome(_GroupFailed(code), None, repairs, ledger)


def _blocked(ledger: _Phase8OperationLedger, code: _Phase8Reason) -> _Phase8GroupOutcome:
    return _Phase8GroupOutcome(_GroupBlocked(code), None, 0, ledger)


def _faulted(
    ledger: _Phase8OperationLedger,
    stage: _Phase8Stage,
    fault: _Phase8OperationFault,
    repairs: int,
) -> _Phase8GroupOutcome:
    if fault.kind is _Phase8FaultKind.FAILED:
        ledger.fail(stage, fault.code)
        terminal: _Phase8GroupTerminal = _GroupFailed(fault.code)
    elif fault.kind is _Phase8FaultKind.INCONSISTENT:
        ledger.inconsistent(stage, fault.code)
        terminal = _GroupInconsistent(fault.code, fault.invocation_global)
    elif fault.kind is _Phase8FaultKind.CANCELLED:
        ledger.cancel(stage, trusted_stop=fault.trusted_stop, dispatched=True)
        terminal = _GroupCancelled(fault.code) if fault.trusted_stop else _GroupLost(_Phase8Reason.TRANSPORT_LOST)
    else:
        ledger.lost(stage, fault.code)
        terminal = _GroupLost(fault.code)
    return _Phase8GroupOutcome(terminal, None, repairs, ledger)


def _run_group_operation(
    members: tuple[object, ...],
    baselines: dict[object, str],
    *,
    analyze: Callable[[], tuple[bool, bool]],
    rewrite: Callable[[dict[object, str]], dict[object, str]],
    evaluate: Callable[[dict[object, str]], _Phase8Metric],
    repair: Callable[[dict[object, str], int], dict[object, str]],
    max_repairs: int,
    operation_plan: _Phase8GroupOperationPlan | None = None,
) -> _Phase8GroupOutcome:
    """Execute the fixed route once; no callback can create a later stage."""
    limits = dict(getattr(_load_phase8_contract(), "limits", ()))
    plan = operation_plan or _compile_group_operation_plan(max_repairs, limits.get("max_repair_iterations", 0))
    if (
        plan is None
        or type(max_repairs) is not int
        or not 0 <= max_repairs <= limits.get("max_repair_iterations", 0)
        or plan.max_repairs != max_repairs
    ):
        plan = _compile_group_operation_plan(0, limits.get("max_repair_iterations", 0))
        assert plan is not None
        ledger = _Phase8OperationLedger(plan)
        ledger.fail(_Phase8Stage.validate_baselines(), _Phase8Reason.INVALID_REPAIR_BOUND)
        return _failed(ledger, _Phase8Reason.INVALID_REPAIR_BOUND, 0)
    ledger = _Phase8OperationLedger(plan)
    validation = _Phase8Stage.validate_baselines()
    if not _validate_complete_revisions(members, baselines):
        ledger.block(validation, _Phase8Reason.MISSING_BASELINE)
        return _blocked(ledger, _Phase8Reason.MISSING_BASELINE)
    ledger.succeed(validation)
    analysis_stage = _Phase8Stage.analyze()
    if not ledger.dispatch(analysis_stage):
        return _failed(ledger, _Phase8Reason.PREREQUISITE, 0)
    try:
        zero_obligations, zero_route_guards = analyze()
    except _Phase8OperationFault as fault:
        return _faulted(ledger, analysis_stage, fault, 0)
    except Exception:
        return _faulted(
            ledger,
            analysis_stage,
            _Phase8OperationFault(_Phase8FaultKind.LOST, _Phase8Reason.TRANSPORT_LOST),
            0,
        )
    ledger._close(analysis_stage, _StageSucceeded(analysis_stage))
    if zero_obligations and zero_route_guards:
        ledger.close_zero_route()
        return _Phase8GroupOutcome(_GroupSucceeded(), dict(baselines), 0, ledger)
    store = _Phase8CandidateStore(dict(baselines))
    rewrite_stage = _Phase8Stage.rewrite()
    if not ledger.dispatch(rewrite_stage):
        return _failed(ledger, _Phase8Reason.PREREQUISITE, 0)
    try:
        revisions = rewrite(dict(store.revisions))
    except _Phase8OperationFault as fault:
        return _faulted(ledger, rewrite_stage, fault, 0)
    except Exception:
        return _faulted(
            ledger,
            rewrite_stage,
            _Phase8OperationFault(_Phase8FaultKind.LOST, _Phase8Reason.TRANSPORT_LOST),
            0,
        )
    if not _validate_complete_revisions(members, revisions):
        fault = _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.INCOMPLETE_GROUP)
        return _faulted(ledger, rewrite_stage, fault, 0)
    store.revisions = revisions
    ledger._close(rewrite_stage, _StageSucceeded(rewrite_stage))
    for round_number in range(max_repairs + 1):
        evaluation_stage = _Phase8Stage.evaluate(round_number)
        if not ledger.dispatch(evaluation_stage):
            return _failed(ledger, _Phase8Reason.PREREQUISITE, round_number)
        try:
            metric = evaluate(dict(store.revisions))
        except _Phase8OperationFault as fault:
            return _faulted(ledger, evaluation_stage, fault, round_number)
        except Exception:
            return _faulted(
                ledger,
                evaluation_stage,
                _Phase8OperationFault(_Phase8FaultKind.LOST, _Phase8Reason.TRANSPORT_LOST),
                round_number,
            )
        if not isinstance(metric, _Phase8Metric):
            fault = _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.INVALID_EVALUATION)
            return _faulted(ledger, evaluation_stage, fault, round_number)
        ledger._close(evaluation_stage, _StageSucceeded(evaluation_stage))
        if not metric.needs_repair:
            ledger.close_pass(evaluation_stage)
            return _Phase8GroupOutcome(_GroupSucceeded(), dict(store.revisions), round_number, ledger)
        if round_number == max_repairs:
            return _failed(ledger, _Phase8Reason.REPAIR_EXHAUSTED, round_number)
        repair_stage = _Phase8Stage.repair(round_number + 1)
        if not ledger.dispatch(repair_stage):
            return _failed(ledger, _Phase8Reason.PREREQUISITE, round_number)
        try:
            revisions = repair(dict(store.revisions), round_number + 1)
        except _Phase8OperationFault as fault:
            return _faulted(ledger, repair_stage, fault, round_number + 1)
        except Exception:
            return _faulted(
                ledger,
                repair_stage,
                _Phase8OperationFault(_Phase8FaultKind.LOST, _Phase8Reason.TRANSPORT_LOST),
                round_number + 1,
            )
        if not _validate_complete_revisions(members, revisions):
            fault = _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, _Phase8Reason.INCOMPLETE_GROUP)
            return _faulted(ledger, repair_stage, fault, round_number + 1)
        store.revisions = revisions
        ledger._close(repair_stage, _StageSucceeded(repair_stage))
    return _failed(ledger, _Phase8Reason.REPAIR_EXHAUSTED, max_repairs)
