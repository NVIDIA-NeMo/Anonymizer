# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private source-neutral protection graph vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class _PrivateGraphValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private protection graph values are not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _DatumId(_PrivateGraphValue):
    value: str


class _DatumPurpose(str, Enum):
    TARGET = "target"
    CONTEXT_ONLY = "context_only"


class _RelationKind(str, Enum):
    RELATED = "related"


@dataclass(frozen=True, slots=True, repr=False)
class _TextDatum(_PrivateGraphValue):
    id: _DatumId
    text: str
    purpose: _DatumPurpose = _DatumPurpose.TARGET


@dataclass(frozen=True, slots=True, repr=False)
class _DatumLink(_PrivateGraphValue):
    source: _DatumId
    target: _DatumId
    relation: _RelationKind


@dataclass(frozen=True, slots=True, repr=False)
class _DatumDependency(_PrivateGraphValue):
    prerequisite: _DatumId
    dependent: _DatumId


@dataclass(frozen=True, slots=True, repr=False)
class _ContextScope(_PrivateGraphValue):
    target: _DatumId
    context: tuple[_DatumId, ...] = ()


@dataclass(frozen=True, slots=True, repr=False)
class _CoherenceScope(_PrivateGraphValue):
    members: tuple[_DatumId, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _AtomicGroup(_PrivateGraphValue):
    members: tuple[_DatumId, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _RewriteGroup(_PrivateGraphValue):
    """Private author declaration for one complete Phase 8 rewrite unit."""

    members: tuple[_DatumId, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _ProtectionGraph(_PrivateGraphValue):
    datums: tuple[_TextDatum, ...]
    links: tuple[_DatumLink, ...]
    context_scopes: tuple[_ContextScope, ...]
    coherence_scopes: tuple[_CoherenceScope, ...]
    atomic_groups: tuple[_AtomicGroup, ...]
    dependencies: tuple[_DatumDependency, ...] = ()
    rewrite_groups: tuple[_RewriteGroup, ...] = ()


def _trivial_graph(datums: tuple[_TextDatum, ...]) -> _ProtectionGraph:
    """Build explicit independent scopes for a compatibility workload."""
    ids = tuple(datum.id for datum in datums)
    return _ProtectionGraph(
        datums=datums,
        links=(),
        context_scopes=tuple(_ContextScope(datum_id) for datum_id in ids),
        coherence_scopes=tuple(_CoherenceScope((datum_id,)) for datum_id in ids),
        atomic_groups=tuple(_AtomicGroup((datum_id,)) for datum_id in ids),
    )
