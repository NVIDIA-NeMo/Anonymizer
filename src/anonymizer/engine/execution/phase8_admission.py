# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure exact-partition admission for private Phase 8 rewrite groups."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from anonymizer.engine.execution.graph import _DatumId, _DatumPurpose, _ProtectionGraph, _RewriteGroup
from anonymizer.engine.execution.phase8_contract import _load_phase8_contract


class _Phase8AdmissionCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    EMPTY_GROUP = "empty_group"
    COVERAGE_GAP = "coverage_gap"
    DUPLICATE_GROUP = "duplicate_group"
    DUPLICATE_MEMBER = "duplicate_member"
    OVERLAP = "overlap"
    UNKNOWN_MEMBER = "unknown_or_context_only_member"
    CROSS_ATOMIC = "cross_atomic_rewrite_group"
    LIMIT_EXCEEDED = "limit_exceeded"


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _Phase8GroupId:
    """Compiler-issued opaque identity; no source/content identity is retained."""

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 8 group identities are not serializable")


class _Phase8StageKind(str, Enum):
    VALIDATE_BASELINES = "validate-baselines"
    ANALYZE = "analyze"
    REWRITE = "rewrite"
    EVALUATE = "evaluate"
    REPAIR = "repair"


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8Stage:
    kind: _Phase8StageKind
    round_number: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, _Phase8StageKind):
            valid = False
        elif self.kind in {
            _Phase8StageKind.VALIDATE_BASELINES,
            _Phase8StageKind.ANALYZE,
            _Phase8StageKind.REWRITE,
        }:
            valid = self.round_number is None
        elif self.kind is _Phase8StageKind.EVALUATE:
            valid = type(self.round_number) is int and self.round_number >= 0
        else:
            valid = type(self.round_number) is int and self.round_number >= 1
        if not valid:
            raise ValueError("invalid Phase 8 operation stage")

    @property
    def name(self) -> str:
        return self.kind.value if self.round_number is None else f"{self.kind.value}-{self.round_number}"

    @classmethod
    def validate_baselines(cls) -> _Phase8Stage:
        return cls(_Phase8StageKind.VALIDATE_BASELINES)

    @classmethod
    def analyze(cls) -> _Phase8Stage:
        return cls(_Phase8StageKind.ANALYZE)

    @classmethod
    def rewrite(cls) -> _Phase8Stage:
        return cls(_Phase8StageKind.REWRITE)

    @classmethod
    def evaluate(cls, round_number: int) -> _Phase8Stage:
        return cls(_Phase8StageKind.EVALUATE, round_number)

    @classmethod
    def repair(cls, round_number: int) -> _Phase8Stage:
        return cls(_Phase8StageKind.REPAIR, round_number)


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8GroupOperationPlan:
    """Admission-time route whose stages cannot expand after effects begin."""

    group_id: _Phase8GroupId
    max_repairs: int
    stages: tuple[_Phase8Stage, ...]

    def __post_init__(self) -> None:
        expected = [
            _Phase8Stage.validate_baselines(),
            _Phase8Stage.analyze(),
            _Phase8Stage.rewrite(),
            _Phase8Stage.evaluate(0),
        ]
        valid_bound = type(self.max_repairs) is int and self.max_repairs >= 0
        if valid_bound:
            for round_number in range(1, self.max_repairs + 1):
                expected.extend((_Phase8Stage.repair(round_number), _Phase8Stage.evaluate(round_number)))
        if not valid_bound or not isinstance(self.group_id, _Phase8GroupId) or tuple(expected) != self.stages:
            raise ValueError("invalid Phase 8 group operation plan")


def _compile_group_operation_plan(
    max_repairs: object, limit: int, group_id: _Phase8GroupId | None = None
) -> _Phase8GroupOperationPlan | None:
    if type(max_repairs) is not int or not 0 <= max_repairs <= limit:
        return None
    stages = [
        _Phase8Stage.validate_baselines(),
        _Phase8Stage.analyze(),
        _Phase8Stage.rewrite(),
        _Phase8Stage.evaluate(0),
    ]
    for round_number in range(1, max_repairs + 1):
        stages.extend((_Phase8Stage.repair(round_number), _Phase8Stage.evaluate(round_number)))
    return _Phase8GroupOperationPlan(group_id or _Phase8GroupId(), max_repairs, tuple(stages))


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8GroupManifest:
    id: _Phase8GroupId
    members: tuple[_DatumId, ...]
    operations: _Phase8GroupOperationPlan


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8Plan:
    groups: tuple[_Phase8GroupManifest, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8Rejected:
    code: _Phase8AdmissionCode


def _compile_phase8_plan(graph: object, *, max_repairs: int | None = None) -> _Phase8Plan | _Phase8Rejected:
    if not isinstance(graph, _ProtectionGraph) or not graph.rewrite_groups:
        return _Phase8Rejected(_Phase8AdmissionCode.INVALID_INPUT)
    limits = dict(getattr(_load_phase8_contract(), "limits", ()))
    repair_limit = limits.get("max_repair_iterations", 0)
    requested_repairs = repair_limit if max_repairs is None else max_repairs
    if type(requested_repairs) is not int or not 0 <= requested_repairs <= repair_limit:
        return _Phase8Rejected(_Phase8AdmissionCode.LIMIT_EXCEEDED)
    targets = tuple(datum.id for datum in graph.datums if datum.purpose is _DatumPurpose.TARGET)
    if len(targets) > limits.get("max_datums_per_invocation", 0) or len(graph.rewrite_groups) > limits.get(
        "max_rewrite_groups_per_invocation", 0
    ):
        return _Phase8Rejected(_Phase8AdmissionCode.LIMIT_EXCEEDED)
    target_set = set(targets)
    atomic_sets = tuple(frozenset(group.members) for group in graph.atomic_groups)
    seen: set[_DatumId] = set()
    declarations: set[tuple[_DatumId, ...]] = set()
    compiled: list[tuple[int, tuple[_DatumId, ...]]] = []
    for group in graph.rewrite_groups:
        if not isinstance(group, _RewriteGroup):
            return _Phase8Rejected(_Phase8AdmissionCode.INVALID_INPUT)
        members = group.members
        if not members:
            return _Phase8Rejected(_Phase8AdmissionCode.EMPTY_GROUP)
        if len(members) > limits.get("max_members_per_rewrite_group", 0):
            return _Phase8Rejected(_Phase8AdmissionCode.LIMIT_EXCEEDED)
        if len(set(members)) != len(members):
            return _Phase8Rejected(_Phase8AdmissionCode.DUPLICATE_MEMBER)
        if not set(members) <= target_set:
            return _Phase8Rejected(_Phase8AdmissionCode.UNKNOWN_MEMBER)
        if members in declarations:
            return _Phase8Rejected(_Phase8AdmissionCode.DUPLICATE_GROUP)
        declarations.add(members)
        if seen.intersection(members):
            return _Phase8Rejected(_Phase8AdmissionCode.OVERLAP)
        seen.update(members)
        if sum(set(members) <= atomic for atomic in atomic_sets) != 1:
            return _Phase8Rejected(_Phase8AdmissionCode.CROSS_ATOMIC)
        compiled.append((min(targets.index(member) for member in members), members))
    if seen != target_set:
        return _Phase8Rejected(_Phase8AdmissionCode.COVERAGE_GAP)
    compiled.sort(key=lambda item: (item[0], tuple(targets.index(member) for member in item[1])))
    manifests: list[_Phase8GroupManifest] = []
    for _, members in compiled:
        group_id = _Phase8GroupId()
        operations = _compile_group_operation_plan(requested_repairs, repair_limit, group_id)
        assert operations is not None
        manifests.append(_Phase8GroupManifest(group_id, members, operations))
    return _Phase8Plan(tuple(manifests))
