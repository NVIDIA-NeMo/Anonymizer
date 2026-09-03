# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent pure oracle for phase-4 dependency and atomic release semantics."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Literal, TypeAlias, assert_never, final


class ReferenceTaskOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"
    BLOCKED = "blocked"
    INCONSISTENT = "inconsistent"


class ReferenceInvocationOutcome(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"
    INCONSISTENT = "inconsistent"


class ReferenceCorruption(str, Enum):
    """Closed reconciliation evidence algebra, kept independent of ledger types."""

    MISSING = "missing"
    DUPLICATE = "duplicate"
    UNKNOWN = "unknown"
    FOREIGN = "foreign"
    STALE = "stale"
    SWAPPED = "swapped"
    PLAN_MISMATCH = "plan_mismatch"
    CONTRADICTORY = "contradictory"


@dataclass(frozen=True)
class ReferenceDeclaration:
    datum_ids: tuple[str, ...]
    dependencies: tuple[tuple[str, str], ...]
    atomic_groups: tuple[tuple[str, ...], ...]
    stages: tuple[str, ...] = ("protect",)
    rejected_datums: frozenset[str] = frozenset()
    rejected_groups: frozenset[frozenset[str]] = frozenset()


@dataclass(frozen=True)
class ReferenceResult:
    release_eligible: frozenset[str]
    released_groups: frozenset[frozenset[str]]


ReferenceTaskKey: TypeAlias = tuple[str, str]


@final
@dataclass(frozen=True)
class ReferenceDispatch:
    task: ReferenceTaskKey


@final
@dataclass(frozen=True)
class ReferenceSuccess:
    task: ReferenceTaskKey


@final
@dataclass(frozen=True)
class ReferenceFailure:
    task: ReferenceTaskKey


@final
@dataclass(frozen=True)
class ReferenceCancellationRequest:
    pass


@final
@dataclass(frozen=True)
class ReferenceStopAcknowledgement:
    task: ReferenceTaskKey


@final
@dataclass(frozen=True)
class ReferenceTransportLoss:
    task: ReferenceTaskKey


@final
@dataclass(frozen=True)
class ReferenceContradiction:
    pass


@final
@dataclass(frozen=True)
class ReferenceResultConstructionFailure:
    pass


@final
@dataclass(frozen=True)
class ReferenceCorruptEvidence:
    kind: ReferenceCorruption


ReferenceObservation: TypeAlias = (
    ReferenceDispatch
    | ReferenceSuccess
    | ReferenceFailure
    | ReferenceCancellationRequest
    | ReferenceStopAcknowledgement
    | ReferenceTransportLoss
    | ReferenceContradiction
    | ReferenceResultConstructionFailure
    | ReferenceCorruptEvidence
)


@dataclass(frozen=True)
class ReferenceHierarchyResult:
    tasks: tuple[tuple[ReferenceTaskKey, ReferenceTaskOutcome], ...]
    task_causes: tuple[tuple[ReferenceTaskKey, tuple[str, ...]], ...]
    datums: tuple[tuple[str, ReferenceTaskOutcome], ...]
    dependencies: tuple[tuple[tuple[str, str], bool], ...]
    stages: tuple[tuple[str, ReferenceTaskOutcome], ...]
    released_groups: frozenset[frozenset[str]]
    released_group_order: tuple[tuple[str, ...], ...]
    invocation: ReferenceInvocationOutcome


@dataclass(frozen=True)
class ReferenceCorpusCase:
    """One canonical schedule from the bounded phase-4 conformance envelope."""

    declaration: ReferenceDeclaration
    observations: tuple[ReferenceObservation, ...]
    graph_witness: bool = True


def flat_partitions(values: tuple[str, ...]) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Enumerate set partitions without using production group code."""
    if not values:
        return ((),)
    head, *tail = values
    tail_partitions = flat_partitions(tuple(tail))
    expanded = []
    for partition in tail_partitions:
        expanded.append(((head,), *partition))
        expanded.extend(
            tuple(
                (*partition[:index], (head, *group), *partition[index + 1 :]) for index, group in enumerate(partition)
            )
        )
    return tuple(expanded)


def acyclic_dependencies(values: tuple[str, ...]) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Enumerate labeled DAG edge sets independently of admission."""
    candidates = tuple((left, right) for left in values for right in values if left != right)
    return tuple(
        edges
        for selected in product((False, True), repeat=len(candidates))
        if _is_acyclic(values, edges := tuple(edge for edge, included in zip(candidates, selected) if included))
    )


def streaming_conformance_cases() -> Iterator[ReferenceCorpusCase]:
    """Yield, without materializing, canonical schedules for the finite design envelope.

    Independent ready tasks use a stable, independently derived topological order.
    Every admitted graph and stage topology is crossed with each primary terminal
    and reconciliation-fault class. A separate witness axis places those events at
    every logical task position across one through three stages.
    """
    yield ReferenceCorpusCase(ReferenceDeclaration((), (), ()), ())
    for datum_count in range(1, 5):
        datum_ids = tuple(chr(ord("a") + index) for index in range(datum_count))
        for dependencies in acyclic_dependencies(datum_ids):
            for groups in flat_partitions(datum_ids):
                for stage_count in range(1, 4):
                    declaration = ReferenceDeclaration(
                        datum_ids,
                        dependencies,
                        groups,
                        tuple(f"stage-{index}" for index in range(stage_count)),
                    )
                    successful = _successful_observations(declaration)
                    yield ReferenceCorpusCase(declaration, successful)
                    yield from _topology_event_witnesses(declaration, successful)
    yield from _semantic_transition_witnesses()


def _semantic_transition_witnesses() -> Iterator[ReferenceCorpusCase]:
    """Cover non-commuting terminal and reconciliation transitions separately.

    The graph axis above exhausts all 0--4 datum DAG/partition fixed points and
    crosses each topology with every event class at a canonical dispatch. This
    axis uses a non-singleton dependency plus atomic peer declaration to place the
    same transitions at every logical task position and stage cardinality.
    """
    for stage_count in range(1, 4):
        declaration = ReferenceDeclaration(
            ("a", "b", "c"),
            (("a", "b"),),
            (("a", "c"), ("b",)),
            tuple(f"stage-{index}" for index in range(stage_count)),
        )
        success = _successful_observations(declaration)
        for observations in (
            success,
            (*success, ReferenceCancellationRequest()),
            (*success, ReferenceContradiction()),
            (*success, ReferenceResultConstructionFailure()),
        ):
            yield ReferenceCorpusCase(declaration, observations, graph_witness=False)
        tasks = tuple((stage, datum_id) for stage in declaration.stages for datum_id in declaration.datum_ids)
        for task in tasks:
            dispatch_index = success.index(ReferenceDispatch(task))
            prefix = success[: dispatch_index + 1]
            schedules: tuple[tuple[ReferenceObservation, ...], ...] = (
                (*prefix, ReferenceCancellationRequest(), ReferenceStopAcknowledgement(task)),
                (*prefix, ReferenceTransportLoss(task)),
                (*prefix, ReferenceFailure(task)),
                (*prefix, ReferenceCancellationRequest(), ReferenceSuccess(task)),
                (*prefix, ReferenceSuccess(task), ReferenceStopAcknowledgement(task)),
                *(tuple((*prefix, ReferenceCorruptEvidence(kind))) for kind in ReferenceCorruption),
            )
            for observations in schedules:
                yield ReferenceCorpusCase(declaration, observations, graph_witness=False)


def _topology_event_witnesses(
    declaration: ReferenceDeclaration,
    success: tuple[ReferenceObservation, ...],
) -> Iterator[ReferenceCorpusCase]:
    """Cross every admitted topology with every primary terminal/fault class."""
    first_dispatch = next(event for event in success if isinstance(event, ReferenceDispatch))
    dispatch_index = success.index(first_dispatch)
    prefix = success[: dispatch_index + 1]
    task = first_dispatch.task
    schedules: list[tuple[ReferenceObservation, ...]] = [
        (*prefix, ReferenceFailure(task)),
        (*prefix, ReferenceTransportLoss(task)),
        (*prefix, ReferenceCancellationRequest()),
        (*prefix, ReferenceCancellationRequest(), ReferenceStopAcknowledgement(task)),
        (*prefix, ReferenceCancellationRequest(), ReferenceSuccess(task)),
        (*prefix, ReferenceSuccess(task), ReferenceStopAcknowledgement(task)),
        (*success, ReferenceResultConstructionFailure()),
    ]
    schedules.extend(
        (*prefix, ReferenceCorruptEvidence(kind))
        for kind in ReferenceCorruption
        if kind is not ReferenceCorruption.SWAPPED or len(declaration.datum_ids) * len(declaration.stages) > 1
    )
    yield from (ReferenceCorpusCase(declaration, observations, graph_witness=False) for observations in schedules)


def _successful_observations(declaration: ReferenceDeclaration) -> tuple[ReferenceObservation, ...]:
    """Use the unique declaration-order representative of each commute class."""
    ordered_datums = _topological_datums(declaration)
    return tuple(
        event
        for datum_id in ordered_datums
        for stage in declaration.stages
        for event in (ReferenceDispatch((stage, datum_id)), ReferenceSuccess((stage, datum_id)))
    )


def _topological_datums(declaration: ReferenceDeclaration) -> tuple[str, ...]:
    """Derive a stable topological order independently from production admission."""
    predecessors = {dependent: set() for dependent in declaration.datum_ids}
    for prerequisite, dependent in declaration.dependencies:
        predecessors[dependent].add(prerequisite)
    ordered_datums: list[str] = []
    remaining = set(declaration.datum_ids)
    while remaining:
        ready = next(
            datum_id for datum_id in declaration.datum_ids if datum_id in remaining and not predecessors[datum_id]
        )
        remaining.remove(ready)
        ordered_datums.append(ready)
        for blocked_by in predecessors.values():
            blocked_by.discard(ready)
    return tuple(ordered_datums)


def _is_acyclic(values: tuple[str, ...], edges: tuple[tuple[str, str], ...]) -> bool:
    remaining = set(values)
    while remaining:
        roots = {value for value in remaining if not any(right == value and left in remaining for left, right in edges)}
        if not roots:
            return False
        remaining -= roots
    return True


def reduce_reference(
    declaration: ReferenceDeclaration,
    task_outcomes: dict[str, ReferenceTaskOutcome],
) -> ReferenceResult:
    """Derive the least fixed point without using production reducers."""
    eligible = {
        datum_id for datum_id in declaration.datum_ids if task_outcomes[datum_id] is ReferenceTaskOutcome.SUCCEEDED
    }
    changed = True
    while changed:
        changed = False
        for group in declaration.atomic_groups:
            if not set(group).issubset(eligible):
                previous = len(eligible)
                eligible.difference_update(group)
                changed = changed or len(eligible) != previous
        for prerequisite, dependent in declaration.dependencies:
            if prerequisite not in eligible and dependent in eligible:
                eligible.remove(dependent)
                changed = True
    released = frozenset(frozenset(group) for group in declaration.atomic_groups if set(group).issubset(eligible))
    return ReferenceResult(frozenset(eligible), released)


def reduce_observations(
    declaration: ReferenceDeclaration,
    observations: tuple[ReferenceObservation, ...],
) -> ReferenceHierarchyResult:
    """Derive hierarchy outcomes from exogenous observations only."""
    task_order = tuple(
        (stage, datum_id) for stage in declaration.stages for datum_id in _topological_datums(declaration)
    )
    states: dict[ReferenceTaskKey, str | ReferenceTaskOutcome] = dict.fromkeys(task_order, "planned")
    causes: dict[ReferenceTaskKey, tuple[str, ...]] = dict.fromkeys(task_order, ())
    cancellation_requested = False
    invocation_lost = False
    global_inconsistent = False
    result_construction_failed = False

    def mark_global_inconsistent(cause: str = "contradictory") -> None:
        nonlocal global_inconsistent, states
        global_inconsistent = True
        updated: dict[ReferenceTaskKey, str | ReferenceTaskOutcome] = {
            task: state if isinstance(state, ReferenceTaskOutcome) else ReferenceTaskOutcome.INCONSISTENT
            for task, state in states.items()
        }
        causes.update({task: (cause,) for task, state in updated.items() if state is ReferenceTaskOutcome.INCONSISTENT})
        states = updated

    def datum_state(datum_id: str) -> str:
        children = tuple(states[(stage, datum_id)] for stage in declaration.stages)
        if all(child is ReferenceTaskOutcome.SUCCEEDED for child in children):
            return "unsatisfied" if datum_id in declaration.rejected_datums else "satisfied"
        if any(
            isinstance(child, ReferenceTaskOutcome) and child is not ReferenceTaskOutcome.SUCCEEDED
            for child in children
        ):
            return "unsatisfied"
        return "waiting"

    def readiness(task: ReferenceTaskKey) -> str:
        stage, datum_id = task
        stage_index = declaration.stages.index(stage)
        if stage_index:
            previous = states[(declaration.stages[stage_index - 1], datum_id)]
            if previous is not ReferenceTaskOutcome.SUCCEEDED:
                return "blocked" if isinstance(previous, ReferenceTaskOutcome) else "waiting"
        prerequisites = tuple(
            prerequisite for prerequisite, dependent in declaration.dependencies if dependent == datum_id
        )
        prerequisite_states = tuple(map(datum_state, prerequisites))
        if "unsatisfied" in prerequisite_states:
            return "blocked"
        return "ready" if all(state == "satisfied" for state in prerequisite_states) else "waiting"

    def advance() -> None:
        changed = True
        while changed:
            changed = False
            for task in task_order:
                if states[task] != "planned":
                    continue
                guard = readiness(task)
                if guard == "ready":
                    states[task] = "ready"
                    changed = True
                elif guard == "blocked":
                    states[task] = ReferenceTaskOutcome.BLOCKED
                    changed = True

    for observation in observations:
        advance()
        match observation:
            case ReferenceDispatch(task=task):
                if states.get(task) == "ready":
                    states[task] = "dispatched"
                else:
                    mark_global_inconsistent()
            case ReferenceSuccess(task=task):
                if states.get(task) == "dispatched":
                    states[task] = ReferenceTaskOutcome.SUCCEEDED
                elif not isinstance(states.get(task), ReferenceTaskOutcome):
                    mark_global_inconsistent()
            case ReferenceFailure(task=task):
                if states.get(task) == "dispatched":
                    states[task] = ReferenceTaskOutcome.FAILED
                    causes[task] = ("known_failure",)
                elif not isinstance(states.get(task), ReferenceTaskOutcome):
                    mark_global_inconsistent()
            case ReferenceCancellationRequest():
                cancellation_requested = True
                states = {
                    task: ReferenceTaskOutcome.CANCELLED if state in {"planned", "ready"} else state
                    for task, state in states.items()
                }
                causes.update(
                    {
                        task: ("cancellation",)
                        for task, state in states.items()
                        if state is ReferenceTaskOutcome.CANCELLED
                    }
                )
            case ReferenceStopAcknowledgement(task=task):
                if states.get(task) == "dispatched":
                    states[task] = ReferenceTaskOutcome.CANCELLED
                    causes[task] = ("cancellation", "stop_acknowledged")
            case ReferenceTransportLoss(task=task):
                if states.get(task) == "dispatched":
                    states[task] = ReferenceTaskOutcome.LOST
                    causes[task] = ("transport_lost",)
                    invocation_lost = True
            case ReferenceContradiction():
                mark_global_inconsistent()
            case ReferenceResultConstructionFailure():
                result_construction_failed = True
            case ReferenceCorruptEvidence(kind=ReferenceCorruption.MISSING):
                for task, state in states.items():
                    if state == "dispatched":
                        states[task] = ReferenceTaskOutcome.INCONSISTENT
                        causes[task] = ("missing",)
            case ReferenceCorruptEvidence(kind=kind):
                mark_global_inconsistent(kind.value)
            case unreachable:
                assert_never(unreachable)

    advance()
    invocation_lost = invocation_lost or any(state == "dispatched" for state in states.values())
    states = {
        task: (
            ReferenceTaskOutcome.LOST
            if state == "dispatched"
            else ReferenceTaskOutcome.BLOCKED
            if state in {"planned", "ready"}
            else state
        )
        for task, state in states.items()
    }
    causes.update(
        {
            task: ("cancellation", "transport_lost") if cancellation_requested else ("transport_lost",)
            for task, state in states.items()
            if state is ReferenceTaskOutcome.LOST and not causes[task]
        }
    )
    causes.update(
        {
            task: ("prerequisite",)
            for task, state in states.items()
            if state is ReferenceTaskOutcome.BLOCKED and not causes[task]
        }
    )
    terminal_tasks = tuple((task, _terminal(state)) for task, state in states.items())
    terminal_causes = tuple((task, causes[task]) for task, _state in terminal_tasks)
    task_by_key = dict(terminal_tasks)
    datums = tuple(
        (
            datum_id,
            _reduce_children(
                tuple(task_by_key[(stage, datum_id)] for stage in declaration.stages),
                rejected=datum_id in declaration.rejected_datums,
            ),
        )
        for datum_id in declaration.datum_ids
    )
    datum_by_id = dict(datums)
    dependencies = tuple(
        ((prerequisite, dependent), datum_by_id[prerequisite] is ReferenceTaskOutcome.SUCCEEDED)
        for prerequisite, dependent in declaration.dependencies
    )
    stages = tuple(
        (
            stage,
            _reduce_children(tuple(task_by_key[(stage, datum_id)] for datum_id in declaration.datum_ids)),
        )
        for stage in declaration.stages
    )
    release = reduce_reference(
        declaration,
        {datum_id: outcome for datum_id, outcome in datums},
    )
    released = release.released_groups - declaration.rejected_groups
    if global_inconsistent or invocation_lost or cancellation_requested or result_construction_failed:
        released = frozenset()
    invocation = (
        ReferenceInvocationOutcome.FAILED
        if result_construction_failed
        else ReferenceInvocationOutcome.INCONSISTENT
        if global_inconsistent
        else ReferenceInvocationOutcome.LOST
        if invocation_lost
        else ReferenceInvocationOutcome.CANCELLED
        if cancellation_requested
        else ReferenceInvocationOutcome.COMPLETED
    )
    released_order = tuple(
        sorted(
            (group for group in declaration.atomic_groups if frozenset(group) in released),
            key=lambda group: tuple(sorted(group)),
        )
    )
    return ReferenceHierarchyResult(
        terminal_tasks,
        terminal_causes,
        datums,
        dependencies,
        stages,
        released,
        released_order,
        invocation,
    )


def _terminal(state: str | ReferenceTaskOutcome) -> ReferenceTaskOutcome:
    if isinstance(state, ReferenceTaskOutcome):
        return state
    raise AssertionError("reference model left a nonterminal task")


def _reduce_children(
    children: tuple[ReferenceTaskOutcome, ...],
    *,
    rejected: bool = False,
) -> ReferenceTaskOutcome:
    if all(child is ReferenceTaskOutcome.SUCCEEDED for child in children):
        return ReferenceTaskOutcome.FAILED if rejected else ReferenceTaskOutcome.SUCCEEDED
    return next(
        outcome
        for outcome in (
            ReferenceTaskOutcome.INCONSISTENT,
            ReferenceTaskOutcome.LOST,
            ReferenceTaskOutcome.CANCELLED,
            ReferenceTaskOutcome.FAILED,
            ReferenceTaskOutcome.BLOCKED,
        )
        if outcome in children
    )


ReferenceMixedSubject: TypeAlias = tuple[Literal["datum", "scope"], str]
ReferenceMixedTaskKey: TypeAlias = tuple[str, ReferenceMixedSubject]


@dataclass(frozen=True)
class ReferenceMixedDeclaration:
    """Independent declaration for non-rectangular datum/scope task plans."""

    datum_ids: tuple[str, ...]
    datum_stages: tuple[str, ...]
    scope_tasks: tuple[tuple[str, str], ...]
    datum_dependencies: tuple[tuple[str, str], ...] = ()
    task_predecessors: tuple[tuple[ReferenceMixedTaskKey, ReferenceMixedTaskKey], ...] = ()
    atomic_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class ReferenceMixedResult:
    """Observable scheduling and reduction result for a mixed task plan."""

    ready_frontiers: tuple[tuple[ReferenceMixedTaskKey, ...], ...]
    tasks: tuple[tuple[ReferenceMixedTaskKey, ReferenceTaskOutcome], ...]
    datums: tuple[tuple[str, ReferenceTaskOutcome], ...]
    stages: tuple[tuple[str, ReferenceTaskOutcome], ...]
    released_groups: frozenset[frozenset[str]]


def reduce_mixed_schedule(
    declaration: ReferenceMixedDeclaration,
    schedule: tuple[tuple[ReferenceMixedTaskKey, ReferenceTaskOutcome], ...],
) -> ReferenceMixedResult:
    """Evaluate a mixed schedule without importing production planning or ledger code."""
    ordered_datums = _topological_mixed_datums(declaration)
    task_order = (
        *((stage, ("datum", datum_id)) for stage in declaration.datum_stages for datum_id in ordered_datums),
        *((stage, ("scope", scope_id)) for stage, scope_id in declaration.scope_tasks),
    )
    states: dict[ReferenceMixedTaskKey, str | ReferenceTaskOutcome] = dict.fromkeys(task_order, "planned")

    def datum_state(datum_id: str) -> str:
        children = tuple(states[(stage, ("datum", datum_id))] for stage in declaration.datum_stages)
        if all(child is ReferenceTaskOutcome.SUCCEEDED for child in children):
            return "satisfied"
        if any(isinstance(child, ReferenceTaskOutcome) for child in children):
            return "unsatisfied"
        return "waiting"

    def readiness(task: ReferenceMixedTaskKey) -> str:
        stage, (subject_kind, subject_id) = task
        if subject_kind == "datum":
            stage_index = declaration.datum_stages.index(stage)
            if stage_index:
                previous = states[(declaration.datum_stages[stage_index - 1], (subject_kind, subject_id))]
                if previous is not ReferenceTaskOutcome.SUCCEEDED:
                    return "blocked" if isinstance(previous, ReferenceTaskOutcome) else "waiting"
        explicit_states = tuple(
            states[prerequisite] for prerequisite, dependent in declaration.task_predecessors if dependent == task
        )
        if any(
            isinstance(state, ReferenceTaskOutcome) and state is not ReferenceTaskOutcome.SUCCEEDED
            for state in explicit_states
        ):
            return "blocked"
        if any(state is not ReferenceTaskOutcome.SUCCEEDED for state in explicit_states):
            return "waiting"
        if subject_kind == "scope":
            return "ready"
        prerequisites = tuple(
            prerequisite for prerequisite, dependent in declaration.datum_dependencies if dependent == subject_id
        )
        prerequisite_states = tuple(datum_state(datum_id) for datum_id in prerequisites)
        if "unsatisfied" in prerequisite_states:
            return "blocked"
        return "ready" if all(state == "satisfied" for state in prerequisite_states) else "waiting"

    def advance() -> None:
        changed = True
        while changed:
            changed = False
            for task in task_order:
                if states[task] != "planned":
                    continue
                guard = readiness(task)
                if guard == "ready":
                    states[task] = "ready"
                    changed = True
                elif guard == "blocked":
                    states[task] = ReferenceTaskOutcome.BLOCKED
                    changed = True

    ready_frontiers: list[tuple[ReferenceMixedTaskKey, ...]] = []
    for task, outcome in schedule:
        advance()
        ready = tuple(candidate for candidate in task_order if states[candidate] == "ready")
        ready_frontiers.append(ready)
        if task not in ready or outcome not in {
            ReferenceTaskOutcome.SUCCEEDED,
            ReferenceTaskOutcome.FAILED,
            ReferenceTaskOutcome.CANCELLED,
            ReferenceTaskOutcome.LOST,
            ReferenceTaskOutcome.INCONSISTENT,
        }:
            raise AssertionError("reference mixed schedule is not executable")
        states[task] = outcome

    advance()
    states = {
        task: ReferenceTaskOutcome.BLOCKED if state in {"planned", "ready"} else state for task, state in states.items()
    }
    terminal_tasks = tuple((task, _terminal(state)) for task, state in states.items())
    outcome_by_task = dict(terminal_tasks)
    datums = tuple(
        (
            datum_id,
            _reduce_children(
                tuple(outcome_by_task[(stage, ("datum", datum_id))] for stage in declaration.datum_stages)
            ),
        )
        for datum_id in declaration.datum_ids
    )
    stages = tuple(
        (
            stage,
            _reduce_children(
                tuple(outcome_by_task[(stage, ("datum", datum_id))] for datum_id in declaration.datum_ids)
            ),
        )
        for stage in declaration.datum_stages
    )
    released = reduce_reference(
        ReferenceDeclaration(
            declaration.datum_ids,
            declaration.datum_dependencies,
            declaration.atomic_groups,
            declaration.datum_stages,
        ),
        dict(datums),
    ).released_groups
    return ReferenceMixedResult(tuple(ready_frontiers), terminal_tasks, datums, stages, released)


def _topological_mixed_datums(declaration: ReferenceMixedDeclaration) -> tuple[str, ...]:
    legacy_shape = ReferenceDeclaration(
        declaration.datum_ids,
        declaration.datum_dependencies,
        declaration.atomic_groups,
        declaration.datum_stages,
    )
    return _topological_datums(legacy_shape)
