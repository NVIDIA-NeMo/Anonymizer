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


@dataclass(frozen=True, slots=True, repr=False)
class _StageSucceeded(_Phase8Terminal):
    stage: _Phase8Stage


@dataclass(frozen=True, slots=True, repr=False)
class _StageFailed(_Phase8Terminal):
    stage: _Phase8Stage
    reason: str


@dataclass(frozen=True, slots=True, repr=False)
class _StageCancelled(_Phase8Terminal):
    stage: _Phase8Stage
    reason: str


@dataclass(frozen=True, slots=True, repr=False)
class _StageLost(_Phase8Terminal):
    stage: _Phase8Stage
    reason: str


@dataclass(frozen=True, slots=True, repr=False)
class _StageBlocked(_Phase8Terminal):
    stage: _Phase8Stage
    reason: str


@dataclass(frozen=True, slots=True, repr=False)
class _StageInconsistent(_Phase8Terminal):
    stage: _Phase8Stage
    reason: str


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

    @property
    def is_closed(self) -> bool:
        return len(self._terminals) == len(self.plan.stages)

    def terminal(self, stage: _Phase8Stage) -> _Phase8Terminal | None:
        return self._terminals.get(stage)

    def reason(self, stage: _Phase8Stage) -> str | None:
        terminal = self.terminal(stage)
        return getattr(terminal, "reason", None)

    def attempt_count(self, stage: _Phase8Stage) -> int:
        return int(stage in self._dispatched)

    def dispatch(self, stage: _Phase8Stage) -> bool:
        if stage not in self.plan.stages or stage in self._terminals or stage in self._dispatched:
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

    def fail(self, stage: _Phase8Stage, reason: str) -> bool:
        return self._terminal_failure(stage, _StageFailed, reason)

    def inconsistent(self, stage: _Phase8Stage, reason: str) -> bool:
        return self._terminal_failure(stage, _StageInconsistent, reason)

    def lost(self, stage: _Phase8Stage, reason: str) -> bool:
        return self._terminal_failure(stage, _StageLost, reason)

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
            _StageCancelled(stage, "cancellation") if trusted_stop else _StageLost(stage, "transport_lost")
        )
        self._close(stage, terminal)
        self._block_descendants(stage, "prerequisite")
        return True

    def block(self, stage: _Phase8Stage, reason: str) -> bool:
        if stage not in self.plan.stages or stage in self._terminals or stage in self._dispatched:
            return False
        position = self.plan.stages.index(stage)
        if any(previous not in self._terminals for previous in self.plan.stages[:position]):
            return False
        self._close(stage, _StageBlocked(stage, reason))
        self._block_descendants(stage, "prerequisite")
        return True

    def close_zero_route(self) -> None:
        self._block_from(_Phase8Stage.rewrite(), "route_not_selected")

    def close_pass(self, evaluation: _Phase8Stage) -> None:
        self._block_descendants(evaluation, "no_repair_needed")

    def _terminal_failure(
        self, stage: _Phase8Stage, terminal_type: Callable[[_Phase8Stage, str], _Phase8Terminal], reason: str
    ) -> bool:
        if stage not in self.plan.stages or stage in self._terminals:
            return False
        if stage not in self._dispatched and not self.dispatch(stage):
            return False
        self._close(stage, terminal_type(stage, reason))
        self._block_descendants(stage, "prerequisite")
        return True

    def _close(self, stage: _Phase8Stage, terminal: _Phase8Terminal) -> bool:
        if stage in self._terminals:
            return False
        self._terminals[stage] = terminal
        if stage in self._dispatched:
            self._attempts[stage] = _Phase8AttemptReceipt(self.plan.group_id, stage, _terminal_name(terminal))
        return True

    def _block_from(self, stage: _Phase8Stage, reason: str) -> None:
        index = self.plan.stages.index(stage)
        for descendant in self.plan.stages[index:]:
            self._close(descendant, _StageBlocked(descendant, reason))

    def _block_descendants(self, stage: _Phase8Stage, reason: str) -> None:
        index = self.plan.stages.index(stage)
        for descendant in self.plan.stages[index + 1 :]:
            self._close(descendant, _StageBlocked(descendant, reason))


class _Phase8GroupTerminal:
    __slots__ = ()


@dataclass(frozen=True, slots=True, repr=False)
class _GroupSucceeded(_Phase8GroupTerminal):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class _GroupFailed(_Phase8GroupTerminal):
    reason: str


@dataclass(frozen=True, slots=True, repr=False)
class _GroupCancelled(_Phase8GroupTerminal):
    reason: str


@dataclass(frozen=True, slots=True, repr=False)
class _GroupLost(_Phase8GroupTerminal):
    reason: str


@dataclass(frozen=True, slots=True, repr=False)
class _GroupBlocked(_Phase8GroupTerminal):
    reason: str


@dataclass(frozen=True, slots=True, repr=False)
class _GroupInconsistent(_Phase8GroupTerminal):
    reason: str
    invocation_global: bool = False


class _Phase8FaultKind(str, Enum):
    FAILED = "failed"
    INCONSISTENT = "inconsistent"
    CANCELLED = "cancelled"
    LOST = "lost"


class _Phase8OperationFault(RuntimeError):
    """Content-free classified fault raised by a selected operation callback."""

    __slots__ = ("kind", "reason", "invocation_global", "trusted_stop")

    def __init__(
        self,
        kind: _Phase8FaultKind,
        reason: str,
        *,
        invocation_global: bool = False,
        trusted_stop: bool = False,
    ) -> None:
        super().__init__(reason)
        self.kind = kind
        self.reason = reason
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
            return _GroupBlocked("prerequisite")
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
    cleanup_verified: bool


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


def _failed(ledger: _Phase8OperationLedger, reason: str, repairs: int) -> _Phase8GroupOutcome:
    return _Phase8GroupOutcome(_GroupFailed(reason), None, repairs, ledger)


def _blocked(ledger: _Phase8OperationLedger, reason: str) -> _Phase8GroupOutcome:
    return _Phase8GroupOutcome(_GroupBlocked(reason), None, 0, ledger)


def _faulted(
    ledger: _Phase8OperationLedger,
    stage: _Phase8Stage,
    fault: _Phase8OperationFault,
    repairs: int,
) -> _Phase8GroupOutcome:
    if fault.kind is _Phase8FaultKind.FAILED:
        ledger.fail(stage, fault.reason)
        terminal: _Phase8GroupTerminal = _GroupFailed(fault.reason)
    elif fault.kind is _Phase8FaultKind.INCONSISTENT:
        ledger.inconsistent(stage, fault.reason)
        terminal = _GroupInconsistent(fault.reason, fault.invocation_global)
    elif fault.kind is _Phase8FaultKind.CANCELLED:
        ledger.cancel(stage, trusted_stop=fault.trusted_stop, dispatched=True)
        terminal = _GroupCancelled(fault.reason) if fault.trusted_stop else _GroupLost("transport_lost")
    else:
        ledger.lost(stage, fault.reason)
        terminal = _GroupLost(fault.reason)
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
        ledger.fail(_Phase8Stage.validate_baselines(), "invalid_repair_bound")
        return _failed(ledger, "invalid_repair_bound", 0)
    ledger = _Phase8OperationLedger(plan)
    validation = _Phase8Stage.validate_baselines()
    if not _validate_complete_revisions(members, baselines):
        ledger.block(validation, "missing_baseline")
        return _blocked(ledger, "missing_baseline")
    ledger.succeed(validation)
    analysis_stage = _Phase8Stage.analyze()
    if not ledger.dispatch(analysis_stage):
        return _failed(ledger, "prerequisite", 0)
    try:
        zero_obligations, zero_route_guards = analyze()
    except _Phase8OperationFault as fault:
        return _faulted(ledger, analysis_stage, fault, 0)
    except Exception:
        return _faulted(
            ledger,
            analysis_stage,
            _Phase8OperationFault(_Phase8FaultKind.LOST, "transport_lost"),
            0,
        )
    ledger._close(analysis_stage, _StageSucceeded(analysis_stage))
    if zero_obligations and zero_route_guards:
        ledger.close_zero_route()
        return _Phase8GroupOutcome(_GroupSucceeded(), dict(baselines), 0, ledger)
    store = _Phase8CandidateStore(dict(baselines))
    rewrite_stage = _Phase8Stage.rewrite()
    if not ledger.dispatch(rewrite_stage):
        return _failed(ledger, "prerequisite", 0)
    try:
        revisions = rewrite(dict(store.revisions))
    except _Phase8OperationFault as fault:
        return _faulted(ledger, rewrite_stage, fault, 0)
    except Exception:
        return _faulted(
            ledger,
            rewrite_stage,
            _Phase8OperationFault(_Phase8FaultKind.LOST, "transport_lost"),
            0,
        )
    if not _validate_complete_revisions(members, revisions):
        fault = _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, "incomplete_group")
        return _faulted(ledger, rewrite_stage, fault, 0)
    store.revisions = revisions
    ledger._close(rewrite_stage, _StageSucceeded(rewrite_stage))
    for round_number in range(max_repairs + 1):
        evaluation_stage = _Phase8Stage.evaluate(round_number)
        if not ledger.dispatch(evaluation_stage):
            return _failed(ledger, "prerequisite", round_number)
        try:
            metric = evaluate(dict(store.revisions))
        except _Phase8OperationFault as fault:
            return _faulted(ledger, evaluation_stage, fault, round_number)
        except Exception:
            return _faulted(
                ledger,
                evaluation_stage,
                _Phase8OperationFault(_Phase8FaultKind.LOST, "transport_lost"),
                round_number,
            )
        if not isinstance(metric, _Phase8Metric):
            fault = _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, "invalid_evaluation")
            return _faulted(ledger, evaluation_stage, fault, round_number)
        ledger._close(evaluation_stage, _StageSucceeded(evaluation_stage))
        if not metric.needs_repair:
            ledger.close_pass(evaluation_stage)
            return _Phase8GroupOutcome(_GroupSucceeded(), dict(store.revisions), round_number, ledger)
        if round_number == max_repairs:
            return _failed(ledger, "repair_exhausted", round_number)
        repair_stage = _Phase8Stage.repair(round_number + 1)
        if not ledger.dispatch(repair_stage):
            return _failed(ledger, "prerequisite", round_number)
        try:
            revisions = repair(dict(store.revisions), round_number + 1)
        except _Phase8OperationFault as fault:
            return _faulted(ledger, repair_stage, fault, round_number + 1)
        except Exception:
            return _faulted(
                ledger,
                repair_stage,
                _Phase8OperationFault(_Phase8FaultKind.LOST, "transport_lost"),
                round_number + 1,
            )
        if not _validate_complete_revisions(members, revisions):
            fault = _Phase8OperationFault(_Phase8FaultKind.INCONSISTENT, "incomplete_group")
            return _faulted(ledger, repair_stage, fault, round_number + 1)
        store.revisions = revisions
        ledger._close(repair_stage, _StageSucceeded(repair_stage))
    return _failed(ledger, "repair_exhausted", max_repairs)
