# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable proof produced by phase-4 graph admission."""

from __future__ import annotations

from dataclasses import dataclass, field

from anonymizer.engine.execution.graph import _DatumId, _TextDatum


class _PrivateAccountingPlanValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private accounting plan values are not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _StageId(_PrivateAccountingPlanValue):
    value: str


@dataclass(frozen=True, slots=True, repr=False)
class _TaskKey(_PrivateAccountingPlanValue):
    stage: _StageId
    datum_id: _DatumId


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
    _proof: _AccountingPlanProof | None = field(default=None, compare=False)


def _admit_accounting_plan(
    datums: tuple[_TextDatum, ...],
    stages: tuple[_StageId, ...],
    tasks: tuple[_TaskKey, ...],
    dependencies: tuple[_CompiledDependency, ...],
    atomic_groups: tuple[_CompiledAtomicGroup, ...],
    topological_datums: tuple[_DatumId, ...],
) -> _AccountingPlan:
    values = (datums, stages, tasks, dependencies, atomic_groups, topological_datums)
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
            tuple((task.stage.value, task.datum_id.value) for task in plan.tasks),
            tuple((dependency.prerequisite.value, dependency.dependent.value) for dependency in plan.dependencies),
            tuple((group.key, tuple(member.value for member in group.members)) for group in plan.atomic_groups),
            tuple(datum_id.value for datum_id in plan.topological_datums),
        )
    except (AttributeError, TypeError):
        return None


@dataclass(frozen=True, slots=True)
class _AccountingLimits:
    max_datums: int
    max_datum_bytes: int
    max_graph_bytes: int
    max_id_bytes: int = 256
    max_dependencies: int = 1_024
    max_atomic_groups: int = 1_024
    max_stages: int = 3
