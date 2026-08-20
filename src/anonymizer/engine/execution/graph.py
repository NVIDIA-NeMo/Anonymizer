# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private source-neutral protection graph and trivial-graph compiler."""

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
class _ProtectionGraph(_PrivateGraphValue):
    datums: tuple[_TextDatum, ...]
    links: tuple[_DatumLink, ...]
    context_scopes: tuple[_ContextScope, ...]
    coherence_scopes: tuple[_CoherenceScope, ...]
    atomic_groups: tuple[_AtomicGroup, ...]


@dataclass(frozen=True, slots=True)
class _GraphLimits:
    max_datums: int
    max_datum_bytes: int
    max_graph_bytes: int
    max_id_bytes: int = 256


class _GraphValidationCode(str, Enum):
    MALFORMED_GRAPH = "malformed_graph"
    DUPLICATE_DATUM_ID = "duplicate_datum_id"
    DATUM_TOO_LARGE = "datum_too_large"
    GRAPH_TOO_LARGE = "graph_too_large"
    TOO_MANY_DATUMS = "too_many_datums"
    UNSUPPORTED_RELATIONSHIPS = "unsupported_relationships"
    UNSUPPORTED_CONTEXT = "unsupported_context"
    UNSUPPORTED_COHERENCE = "unsupported_coherence"
    UNSUPPORTED_ATOMICITY = "unsupported_atomicity"


class _GraphValidationError(ValueError):
    def __init__(self, code: _GraphValidationCode) -> None:
        self.code = code
        super().__init__("private protection graph rejected")

    def __repr__(self) -> str:
        return "<private protection graph error>"


@dataclass(frozen=True, slots=True, repr=False)
class _CompiledTrivialGraph(_PrivateGraphValue):
    datums: tuple[_TextDatum, ...]


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


def _compile_trivial_graph(graph: object, *, limits: _GraphLimits) -> _CompiledTrivialGraph:
    """Validate the first graph profile without accepting related-record semantics."""
    if not isinstance(graph, _ProtectionGraph):
        raise _GraphValidationError(_GraphValidationCode.MALFORMED_GRAPH)
    datums = getattr(graph, "datums", None)
    links = getattr(graph, "links", None)
    context_scopes = getattr(graph, "context_scopes", None)
    coherence_scopes = getattr(graph, "coherence_scopes", None)
    atomic_groups = getattr(graph, "atomic_groups", None)
    if not isinstance(datums, tuple) or not datums:
        raise _GraphValidationError(_GraphValidationCode.MALFORMED_GRAPH)
    if len(datums) > limits.max_datums:
        raise _GraphValidationError(_GraphValidationCode.TOO_MANY_DATUMS)

    datum_ids: list[_DatumId] = []
    total_bytes = 0
    for datum in datums:
        if not isinstance(datum, _TextDatum):
            raise _GraphValidationError(_GraphValidationCode.MALFORMED_GRAPH)
        datum_id = getattr(datum, "id", None)
        identifier = getattr(datum_id, "value", None)
        text = getattr(datum, "text", None)
        purpose = getattr(datum, "purpose", None)
        if (
            not isinstance(datum_id, _DatumId)
            or not isinstance(identifier, str)
            or not identifier
            or len(identifier.encode("utf-8")) > limits.max_id_bytes
            or not isinstance(text, str)
            or purpose is not _DatumPurpose.TARGET
        ):
            raise _GraphValidationError(_GraphValidationCode.MALFORMED_GRAPH)
        datum_size = len(text.encode("utf-8"))
        if datum_size > limits.max_datum_bytes:
            raise _GraphValidationError(_GraphValidationCode.DATUM_TOO_LARGE)
        datum_ids.append(datum_id)
        total_bytes += datum_size
    if total_bytes > limits.max_graph_bytes:
        raise _GraphValidationError(_GraphValidationCode.GRAPH_TOO_LARGE)
    if len({datum_id.value for datum_id in datum_ids}) != len(datum_ids):
        raise _GraphValidationError(_GraphValidationCode.DUPLICATE_DATUM_ID)

    if not isinstance(links, tuple) or not all(isinstance(link, _DatumLink) for link in links):
        raise _GraphValidationError(_GraphValidationCode.MALFORMED_GRAPH)
    if links:
        raise _GraphValidationError(_GraphValidationCode.UNSUPPORTED_RELATIONSHIPS)
    if not _is_trivial_context(context_scopes, datum_ids):
        raise _GraphValidationError(_GraphValidationCode.UNSUPPORTED_CONTEXT)
    if not _is_singleton_partition(coherence_scopes, datum_ids, _CoherenceScope):
        raise _GraphValidationError(_GraphValidationCode.UNSUPPORTED_COHERENCE)
    if not _is_singleton_partition(atomic_groups, datum_ids, _AtomicGroup):
        raise _GraphValidationError(_GraphValidationCode.UNSUPPORTED_ATOMICITY)
    return _CompiledTrivialGraph(datums)


def _is_trivial_context(scopes: object, datum_ids: list[_DatumId]) -> bool:
    if not isinstance(scopes, tuple) or len(scopes) != len(datum_ids):
        return False
    expected = {datum_id.value for datum_id in datum_ids}
    observed: set[str] = set()
    for scope in scopes:
        if not isinstance(scope, _ContextScope):
            return False
        context = getattr(scope, "context", None)
        if not isinstance(context, tuple) or context:
            return False
        target = getattr(scope, "target", None)
        value = getattr(target, "value", None)
        if not isinstance(target, _DatumId) or not isinstance(value, str):
            return False
        observed.add(value)
    return observed == expected and len(observed) == len(scopes)


def _is_singleton_partition(
    scopes: object,
    datum_ids: list[_DatumId],
    scope_type: type[_CoherenceScope] | type[_AtomicGroup],
) -> bool:
    if not isinstance(scopes, tuple) or len(scopes) != len(datum_ids):
        return False
    expected = {datum_id.value for datum_id in datum_ids}
    observed: set[str] = set()
    for scope in scopes:
        if not isinstance(scope, scope_type):
            return False
        members = getattr(scope, "members", None)
        if not isinstance(members, tuple) or len(members) != 1:
            return False
        member = members[0]
        value = getattr(member, "value", None)
        if not isinstance(member, _DatumId) or not isinstance(value, str):
            return False
        observed.add(value)
    return observed == expected and len(observed) == len(scopes)
