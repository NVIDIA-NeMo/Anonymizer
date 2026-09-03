# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure exact-partition admission for private Phase 8 rewrite groups."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from anonymizer.engine.execution.graph import _DatumId, _DatumPurpose, _ProtectionGraph, _RewriteGroup


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


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8GroupManifest:
    id: _Phase8GroupId
    members: tuple[_DatumId, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8Plan:
    groups: tuple[_Phase8GroupManifest, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8Rejected:
    code: _Phase8AdmissionCode


def _compile_phase8_plan(graph: object) -> _Phase8Plan | _Phase8Rejected:
    if not isinstance(graph, _ProtectionGraph) or not graph.rewrite_groups:
        return _Phase8Rejected(_Phase8AdmissionCode.INVALID_INPUT)
    targets = tuple(datum.id for datum in graph.datums if datum.purpose is _DatumPurpose.TARGET)
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
        if len(members) > 4:
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
    return _Phase8Plan(tuple(_Phase8GroupManifest(_Phase8GroupId(), members) for _, members in compiled))
