# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure admission boundary for phase-4 accounting plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, TypeAlias

from anonymizer.engine.execution.accounting_plan import (
    _AccountingLimits,
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
    _CoherenceScope,
    _ContextScope,
    _DatumDependency,
    _DatumId,
    _DatumLink,
    _DatumPurpose,
    _ProtectionGraph,
    _TextDatum,
)


class _AccountingAdmissionCode(str, Enum):
    MALFORMED_GRAPH = "malformed_graph"
    TOO_MANY_DATUMS = "too_many_datums"
    DATUM_TOO_LARGE = "datum_too_large"
    GRAPH_TOO_LARGE = "graph_too_large"
    DUPLICATE_DATUM_ID = "duplicate_datum_id"
    TOO_MANY_DEPENDENCIES = "too_many_dependencies"
    TOO_MANY_ATOMIC_GROUPS = "too_many_atomic_groups"
    MALFORMED_DEPENDENCY = "malformed_dependency"
    DANGLING_DEPENDENCY = "dangling_dependency"
    SELF_DEPENDENCY = "self_dependency"
    DUPLICATE_DEPENDENCY = "duplicate_dependency"
    DEPENDENCY_CYCLE = "dependency_cycle"
    EMPTY_ATOMIC_GROUP = "empty_atomic_group"
    DANGLING_ATOMIC_MEMBER = "dangling_atomic_member"
    DUPLICATE_ATOMIC_MEMBER = "duplicate_atomic_member"
    DUPLICATE_ATOMIC_GROUP = "duplicate_atomic_group"
    ATOMIC_COVERAGE_GAP = "atomic_coverage_gap"
    ATOMIC_GROUP_OVERLAP = "atomic_group_overlap"
    UNSUPPORTED_ATOMIC_NESTING = "unsupported_atomic_nesting"
    UNSUPPORTED_RELATIONSHIPS = "unsupported_relationships"
    UNSUPPORTED_CONTEXT = "unsupported_context"
    UNSUPPORTED_COHERENCE = "unsupported_coherence"
    UNSUPPORTED_TASK_CARDINALITY = "unsupported_task_cardinality"


@dataclass(frozen=True, slots=True, repr=False)
class _AccountingRejected:
    code: _AccountingAdmissionCode

    def __repr__(self) -> str:
        return "<private accounting rejection>"


_AccountingAdmissionResult: TypeAlias = _AccountingPlan | _AccountingRejected


class _AdmissionFailure(Exception):
    def __init__(self, code: _AccountingAdmissionCode) -> None:
        self.code = code


def _compile_accounting_plan(
    graph: object,
    *,
    limits: _AccountingLimits,
    stages: tuple[str, ...] = ("protect",),
) -> _AccountingAdmissionResult:
    try:
        return _compile(graph, limits=limits, stages=stages)
    except _AdmissionFailure as failure:
        return _AccountingRejected(failure.code)


def _compile(graph: object, *, limits: _AccountingLimits, stages: tuple[str, ...]) -> _AccountingPlan:
    if not isinstance(graph, _ProtectionGraph):
        _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
    _check_counts(graph, limits)
    datums, datum_by_value = _compile_datums(graph.datums, limits)
    _check_declaration_shapes(graph)
    dependencies = _compile_dependencies(graph.dependencies, datum_by_value)
    group_members = _compile_group_members(graph.atomic_groups, datum_by_value)
    _check_reference_coverage(group_members, datum_by_value)
    _check_dependency_structure(dependencies)
    _check_partial_overlap(group_members)
    topological_datums = _topological_order(tuple(datum.id for datum in datums), dependencies)
    _check_unsupported_semantics(graph, group_members, datum_by_value, stages, limits)
    compiled_stages = tuple(_StageId(stage) for stage in stages)
    groups = _materialize_groups(group_members, datum_by_value)
    tasks = tuple(
        _TaskKey(stage, _DatumTaskSubject(datum_id)) for stage in compiled_stages for datum_id in topological_datums
    )
    return _admit_accounting_plan(
        datums,
        compiled_stages,
        tasks,
        dependencies,
        groups,
        topological_datums,
    )


def _check_counts(graph: _ProtectionGraph, limits: _AccountingLimits) -> None:
    datums = getattr(graph, "datums", None)
    dependencies = getattr(graph, "dependencies", None)
    groups = getattr(graph, "atomic_groups", None)
    if not isinstance(datums, tuple) or not datums:
        _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
    if len(datums) > limits.max_datums:
        _fail(_AccountingAdmissionCode.TOO_MANY_DATUMS)
    if isinstance(dependencies, tuple) and len(dependencies) > limits.max_dependencies:
        _fail(_AccountingAdmissionCode.TOO_MANY_DEPENDENCIES)
    if isinstance(groups, tuple) and len(groups) > limits.max_atomic_groups:
        _fail(_AccountingAdmissionCode.TOO_MANY_ATOMIC_GROUPS)


def _compile_datums(
    source: tuple[object, ...], limits: _AccountingLimits
) -> tuple[tuple[_TextDatum, ...], dict[str, _DatumId]]:
    datums: list[_TextDatum] = []
    datum_by_value: dict[str, _DatumId] = {}
    total_bytes = 0
    for candidate in source:
        if not isinstance(candidate, _TextDatum):
            _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
        try:
            datum_id = candidate.id
            value = datum_id.value
            text = candidate.text
            purpose = candidate.purpose
        except AttributeError:
            _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
        if (
            not isinstance(datum_id, _DatumId)
            or not isinstance(value, str)
            or not value
            or not isinstance(text, str)
            or purpose is not _DatumPurpose.TARGET
        ):
            _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
        if _utf8_size(value) > limits.max_id_bytes:
            _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
        size = _utf8_size(text)
        if size > limits.max_datum_bytes:
            _fail(_AccountingAdmissionCode.DATUM_TOO_LARGE)
        total_bytes += size
        if value in datum_by_value:
            _fail(_AccountingAdmissionCode.DUPLICATE_DATUM_ID)
        detached_id = _DatumId(value)
        datum_by_value[value] = detached_id
        datums.append(_TextDatum(detached_id, text))
    if total_bytes > limits.max_graph_bytes:
        _fail(_AccountingAdmissionCode.GRAPH_TOO_LARGE)
    return tuple(datums), datum_by_value


def _utf8_size(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)


def _check_declaration_shapes(graph: _ProtectionGraph) -> None:
    if not isinstance(getattr(graph, "dependencies", None), tuple):
        _fail(_AccountingAdmissionCode.MALFORMED_DEPENDENCY)
    if not isinstance(getattr(graph, "atomic_groups", None), tuple):
        _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
    if not isinstance(getattr(graph, "links", None), tuple) or not all(
        isinstance(link, _DatumLink) for link in graph.links
    ):
        _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
    if not isinstance(getattr(graph, "context_scopes", None), tuple) or not all(
        isinstance(scope, _ContextScope) for scope in graph.context_scopes
    ):
        _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
    if not isinstance(getattr(graph, "coherence_scopes", None), tuple) or not all(
        isinstance(scope, _CoherenceScope) for scope in graph.coherence_scopes
    ):
        _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)


def _compile_dependencies(
    source: tuple[object, ...], datum_by_value: dict[str, _DatumId]
) -> tuple[_CompiledDependency, ...]:
    dependencies: list[_CompiledDependency] = []
    for candidate in source:
        if not isinstance(candidate, _DatumDependency):
            _fail(_AccountingAdmissionCode.MALFORMED_DEPENDENCY)
        try:
            prerequisite = candidate.prerequisite.value
            dependent = candidate.dependent.value
        except AttributeError:
            _fail(_AccountingAdmissionCode.MALFORMED_DEPENDENCY)
        if not isinstance(prerequisite, str) or not isinstance(dependent, str):
            _fail(_AccountingAdmissionCode.MALFORMED_DEPENDENCY)
        if prerequisite not in datum_by_value or dependent not in datum_by_value:
            _fail(_AccountingAdmissionCode.DANGLING_DEPENDENCY)
        dependencies.append(_CompiledDependency(datum_by_value[prerequisite], datum_by_value[dependent]))
    return tuple(dependencies)


def _compile_group_members(
    source: tuple[object, ...], datum_by_value: dict[str, _DatumId]
) -> tuple[frozenset[str], ...]:
    groups: list[frozenset[str]] = []
    for candidate in source:
        if not isinstance(candidate, _AtomicGroup):
            _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
        try:
            members = candidate.members
        except AttributeError:
            _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
        if not isinstance(members, tuple):
            _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
        if not members:
            _fail(_AccountingAdmissionCode.EMPTY_ATOMIC_GROUP)
        values: list[str] = []
        for member in members:
            value = getattr(member, "value", None)
            if not isinstance(member, _DatumId) or not isinstance(value, str):
                _fail(_AccountingAdmissionCode.MALFORMED_GRAPH)
            if value not in datum_by_value:
                _fail(_AccountingAdmissionCode.DANGLING_ATOMIC_MEMBER)
            values.append(value)
        if len(set(values)) != len(values):
            _fail(_AccountingAdmissionCode.DUPLICATE_ATOMIC_MEMBER)
        member_set = frozenset(values)
        if member_set in groups:
            _fail(_AccountingAdmissionCode.DUPLICATE_ATOMIC_GROUP)
        groups.append(member_set)
    return tuple(groups)


def _check_reference_coverage(groups: tuple[frozenset[str], ...], datum_by_value: dict[str, _DatumId]) -> None:
    covered = set().union(*groups) if groups else set()
    if covered != set(datum_by_value):
        _fail(_AccountingAdmissionCode.ATOMIC_COVERAGE_GAP)


def _check_dependency_structure(dependencies: tuple[_CompiledDependency, ...]) -> None:
    observed: set[tuple[str, str]] = set()
    for dependency in dependencies:
        edge = (dependency.prerequisite.value, dependency.dependent.value)
        if edge[0] == edge[1]:
            _fail(_AccountingAdmissionCode.SELF_DEPENDENCY)
        if edge in observed:
            _fail(_AccountingAdmissionCode.DUPLICATE_DEPENDENCY)
        observed.add(edge)


def _check_partial_overlap(groups: tuple[frozenset[str], ...]) -> None:
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            if left & right and not (left < right or right < left):
                _fail(_AccountingAdmissionCode.ATOMIC_GROUP_OVERLAP)


def _topological_order(
    datum_ids: tuple[_DatumId, ...], dependencies: tuple[_CompiledDependency, ...]
) -> tuple[_DatumId, ...]:
    position = {datum_id.value: index for index, datum_id in enumerate(datum_ids)}
    incoming = {datum_id.value: 0 for datum_id in datum_ids}
    dependents: dict[str, list[str]] = {datum_id.value: [] for datum_id in datum_ids}
    for dependency in dependencies:
        incoming[dependency.dependent.value] += 1
        dependents[dependency.prerequisite.value].append(dependency.dependent.value)
    ready = [datum_id.value for datum_id in datum_ids if incoming[datum_id.value] == 0]
    ordered: list[_DatumId] = []
    while ready:
        value = ready.pop(0)
        ordered.append(datum_ids[position[value]])
        for dependent in dependents[value]:
            incoming[dependent] -= 1
            if incoming[dependent] == 0:
                ready.append(dependent)
                ready.sort(key=position.__getitem__)
    if len(ordered) != len(datum_ids):
        _fail(_AccountingAdmissionCode.DEPENDENCY_CYCLE)
    return tuple(ordered)


def _check_unsupported_semantics(
    graph: _ProtectionGraph,
    groups: tuple[frozenset[str], ...],
    datum_by_value: dict[str, _DatumId],
    stages: tuple[str, ...],
    limits: _AccountingLimits,
) -> None:
    if any(left < right or right < left for index, left in enumerate(groups) for right in groups[index + 1 :]):
        _fail(_AccountingAdmissionCode.UNSUPPORTED_ATOMIC_NESTING)
    if graph.links:
        _fail(_AccountingAdmissionCode.UNSUPPORTED_RELATIONSHIPS)
    expected = set(datum_by_value)
    context_targets = {getattr(scope.target, "value", None) for scope in graph.context_scopes}
    if (
        len(graph.context_scopes) != len(expected)
        or context_targets != expected
        or any(scope.context for scope in graph.context_scopes)
    ):
        _fail(_AccountingAdmissionCode.UNSUPPORTED_CONTEXT)
    coherence = tuple(frozenset(member.value for member in scope.members) for scope in graph.coherence_scopes)
    if len(coherence) != len(expected) or set(coherence) != {frozenset((value,)) for value in expected}:
        _fail(_AccountingAdmissionCode.UNSUPPORTED_COHERENCE)
    if (
        not isinstance(stages, tuple)
        or not stages
        or len(stages) > limits.max_stages
        or any(not isinstance(stage, str) or not stage for stage in stages)
        or len(set(stages)) != len(stages)
    ):
        _fail(_AccountingAdmissionCode.UNSUPPORTED_TASK_CARDINALITY)


def _materialize_groups(
    groups: tuple[frozenset[str], ...], datum_by_value: dict[str, _DatumId]
) -> tuple[_CompiledAtomicGroup, ...]:
    canonical = sorted(groups, key=lambda members: tuple(sorted(members)))
    return tuple(
        _CompiledAtomicGroup(
            _AtomicGroupKey(),
            tuple(datum_id for value, datum_id in datum_by_value.items() if value in members),
        )
        for members in canonical
    )


def _fail(code: _AccountingAdmissionCode) -> NoReturn:
    raise _AdmissionFailure(code)
