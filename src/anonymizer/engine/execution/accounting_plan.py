# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable proof produced by phase-4 graph admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias, final

from anonymizer.engine.execution.graph import _DatumId, _TextDatum


class _PrivateAccountingPlanValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private accounting plan values are not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _StageId(_PrivateAccountingPlanValue):
    value: str


@final
@dataclass(frozen=True, slots=True, repr=False)
class _DatumTaskSubject(_PrivateAccountingPlanValue):
    datum_id: _DatumId

    def __post_init__(self) -> None:
        if not isinstance(self.datum_id, _DatumId):
            raise TypeError("private datum task subject is malformed")


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _ScopeTaskSubject(_PrivateAccountingPlanValue):
    """Compiler-issued opaque capability for one scope-owned task."""


_TaskSubject: TypeAlias = _DatumTaskSubject | _ScopeTaskSubject


@dataclass(frozen=True, slots=True, repr=False)
class _TaskKey(_PrivateAccountingPlanValue):
    stage: _StageId
    subject: _TaskSubject

    def __post_init__(self) -> None:
        if not isinstance(self.stage, _StageId) or not isinstance(self.subject, (_DatumTaskSubject, _ScopeTaskSubject)):
            raise TypeError("private task subject is malformed")


@dataclass(frozen=True, slots=True, repr=False)
class _TaskPredecessor(_PrivateAccountingPlanValue):
    prerequisite: _TaskKey
    dependent: _TaskKey


@dataclass(frozen=True, slots=True, repr=False)
class _CompiledDependency(_PrivateAccountingPlanValue):
    prerequisite: _DatumId
    dependent: _DatumId


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _AtomicGroupKey(_PrivateAccountingPlanValue):
    """Compiler-created, graph-scoped capability for one atomic group."""


@dataclass(frozen=True, slots=True, repr=False)
class _CompiledAtomicGroup(_PrivateAccountingPlanValue):
    key: _AtomicGroupKey
    members: tuple[_DatumId, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _AccountingPlanProof(_PrivateAccountingPlanValue):
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


_ADMISSION_SEAL = object()


@dataclass(frozen=True, slots=True, repr=False)
class _AccountingPlan(_PrivateAccountingPlanValue):
    datums: tuple[_TextDatum, ...]
    stages: tuple[_StageId, ...]
    tasks: tuple[_TaskKey, ...]
    dependencies: tuple[_CompiledDependency, ...]
    atomic_groups: tuple[_CompiledAtomicGroup, ...]
    topological_datums: tuple[_DatumId, ...]
    task_predecessors: tuple[_TaskPredecessor, ...] = ()
    _proof: _AccountingPlanProof | None = field(default=None, compare=False)

    def with_task_predecessors(self, predecessors: tuple[_TaskPredecessor, ...]) -> _AccountingPlan:
        """Return a resealed plan with explicit compiler-owned task readiness."""
        if not _is_admitted_accounting_plan(self):
            raise TypeError("private accounting plan is not admitted")
        validated = _validate_task_predecessors(self, predecessors)
        return _admit_accounting_plan(
            self.datums,
            self.stages,
            self.tasks,
            self.dependencies,
            self.atomic_groups,
            self.topological_datums,
            validated,
        )

    def with_scope_tasks(
        self,
        stage: _StageId,
        subjects: tuple[_ScopeTaskSubject, ...],
    ) -> _AccountingPlan:
        """Return a resealed plan with one task for each compiler-issued scope."""
        if not _is_admitted_accounting_plan(self):
            raise TypeError("private accounting plan is not admitted")
        existing_scope_subjects = frozenset(
            task.subject for task in self.tasks if isinstance(task.subject, _ScopeTaskSubject)
        )
        if (
            not isinstance(stage, _StageId)
            or not isinstance(stage.value, str)
            or not stage.value
            or stage in self.stages
            or not isinstance(subjects, tuple)
            or not all(isinstance(subject, _ScopeTaskSubject) for subject in subjects)
            or len(set(subjects)) != len(subjects)
            or any(subject in existing_scope_subjects for subject in subjects)
        ):
            raise TypeError("private scope task subjects are malformed")
        if not subjects:
            return self
        tasks = (*self.tasks, *(_TaskKey(stage, subject) for subject in subjects))
        return _admit_accounting_plan(
            self.datums,
            self.stages,
            tasks,
            self.dependencies,
            self.atomic_groups,
            self.topological_datums,
            self.task_predecessors,
        )


def _admit_accounting_plan(
    datums: tuple[_TextDatum, ...],
    stages: tuple[_StageId, ...],
    tasks: tuple[_TaskKey, ...],
    dependencies: tuple[_CompiledDependency, ...],
    atomic_groups: tuple[_CompiledAtomicGroup, ...],
    topological_datums: tuple[_DatumId, ...],
    task_predecessors: tuple[_TaskPredecessor, ...] = (),
) -> _AccountingPlan:
    values = (datums, stages, tasks, dependencies, atomic_groups, topological_datums, task_predecessors)
    plan = _AccountingPlan(*values)
    snapshot = _plan_snapshot(plan)
    if snapshot is None:
        raise TypeError("private accounting plan admission failed")
    return _AccountingPlan(*values, _AccountingPlanProof(_ADMISSION_SEAL, snapshot))


def _is_admitted_accounting_plan(value: object) -> bool:
    if not isinstance(value, _AccountingPlan) or value._proof is None:
        return False
    return value._proof.seal is _ADMISSION_SEAL and value._proof.snapshot == _plan_snapshot(value)


def _plan_snapshot(plan: _AccountingPlan) -> tuple[object, ...] | None:
    """Detach admission proof from every nested mutable Python object."""
    try:
        return (
            tuple((datum.id.value, datum.text, datum.purpose.value) for datum in plan.datums),
            tuple(stage.value for stage in plan.stages),
            tuple(_task_snapshot(task) for task in plan.tasks),
            tuple((dependency.prerequisite.value, dependency.dependent.value) for dependency in plan.dependencies),
            tuple((group.key, tuple(member.value for member in group.members)) for group in plan.atomic_groups),
            tuple(datum_id.value for datum_id in plan.topological_datums),
            tuple(
                (
                    *_task_snapshot(predecessor.prerequisite),
                    *_task_snapshot(predecessor.dependent),
                )
                for predecessor in plan.task_predecessors
            ),
        )
    except (AttributeError, TypeError):
        return None


def _validate_task_predecessors(
    plan: _AccountingPlan,
    predecessors: object,
) -> tuple[_TaskPredecessor, ...]:
    if not isinstance(predecessors, tuple) or not all(
        isinstance(predecessor, _TaskPredecessor) for predecessor in predecessors
    ):
        raise TypeError("private task predecessors are malformed")
    task_set = frozenset(plan.tasks)
    edges = tuple((predecessor.prerequisite, predecessor.dependent) for predecessor in predecessors)
    implicit = frozenset(
        (
            _TaskKey(plan.stages[stage_index - 1], task.subject),
            task,
        )
        for task in plan.tasks
        if isinstance(task.subject, _DatumTaskSubject) and (stage_index := plan.stages.index(task.stage)) > 0
    )
    if (
        any(prerequisite not in task_set or dependent not in task_set for prerequisite, dependent in edges)
        or any(prerequisite == dependent for prerequisite, dependent in edges)
        or len(set(edges)) != len(edges)
        or any(edge in implicit for edge in edges)
    ):
        raise TypeError("private task predecessors are malformed")
    _validate_task_predecessor_dag(plan.tasks, (*implicit, *edges))
    return tuple(predecessors)


def _task_snapshot(task: _TaskKey) -> tuple[object, object]:
    match task.subject:
        case _DatumTaskSubject(datum_id=datum_id):
            return task.stage.value, datum_id.value
        case _ScopeTaskSubject():
            return task.stage.value, task.subject


def _validate_task_predecessor_dag(
    tasks: tuple[_TaskKey, ...],
    edges: tuple[tuple[_TaskKey, _TaskKey], ...],
) -> None:
    incoming = {task: 0 for task in tasks}
    dependents: dict[_TaskKey, list[_TaskKey]] = {task: [] for task in tasks}
    for prerequisite, dependent in edges:
        incoming[dependent] += 1
        dependents[prerequisite].append(dependent)
    ready = [task for task in tasks if incoming[task] == 0]
    visited = 0
    while ready:
        task = ready.pop(0)
        visited += 1
        for dependent in dependents[task]:
            incoming[dependent] -= 1
            if incoming[dependent] == 0:
                ready.append(dependent)
    if visited != len(tasks):
        raise TypeError("private task predecessors contain a cycle")


@dataclass(frozen=True, slots=True)
class _AccountingLimits:
    max_datums: int
    max_datum_bytes: int
    max_graph_bytes: int
    max_id_bytes: int = 256
    max_dependencies: int = 1_024
    max_atomic_groups: int = 1_024
    max_stages: int = 3
