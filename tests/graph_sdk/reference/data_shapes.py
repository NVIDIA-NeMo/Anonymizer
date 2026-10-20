# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed shapes for authenticated historical data fixtures; no interpreter."""

from __future__ import annotations

import json
from typing import Literal, TypeAlias, TypedDict, cast

from tests.graph_sdk.reference.corpora import corpus_bytes

JsonScalar: TypeAlias = str | int | bool | None


JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


JsonObject: TypeAlias = dict[str, JsonValue]


Ref: TypeAlias = tuple[str, int]


ValidationCode: TypeAlias = Literal[
    "invalid_type",
    "invalid_value",
    "limit_exceeded",
    "foreign_owner",
    "duplicate",
    "missing",
    "invalid_range",
    "overlap",
    "cycle",
    "contradictory",
]


Family: TypeAlias = Literal[
    "identity",
    "context",
    "dependency",
    "groups",
    "ranges",
    "relation-separation",
    "ownership-negatives",
    "encoding-bounds",
    "precedence",
    "record",
    "record-precedence",
]


RelationKind: TypeAlias = Literal[
    "source_relations",
    "contexts",
    "dependencies",
    "coherence",
    "atomic",
    "output_regions",
]


RecordBoundary: TypeAlias = Literal[
    "activation",
    "artifact",
    "absence",
    "evidence",
    "membership",
    "terminal",
    "status",
    "canonical",
]


RecordFactKind: TypeAlias = Literal[
    "invocation",
    "occurrence",
    "parent",
    "iteration",
    "key",
    "version",
    "query",
    "scope_revision",
    "artifact",
    "consumed",
    "members",
    "closed",
    "activation",
    "attempt",
    "category",
    "reasons",
    "target",
    "completion",
    "qualification",
    "artifact_available",
    "protection_available",
    "plan",
    "invocation_plan",
    "graph",
    "targets",
    "memberships",
    "terminals",
    "artifacts",
    "evidence",
    "statuses",
]


class DatumEnvelope(TypedDict):
    """Syntactic datum declaration; field values remain semantic inputs."""

    id: JsonValue
    text: JsonValue


class SourceRelationEnvelope(TypedDict):
    derived: JsonValue
    source: JsonValue
    start: JsonValue
    end: JsonValue


class ContextEnvelope(TypedDict):
    target: JsonValue
    source: JsonValue
    start: JsonValue
    end: JsonValue


class DependencyEnvelope(TypedDict):
    prerequisite: JsonValue
    dependent: JsonValue


class GroupEnvelope(TypedDict):
    members: JsonValue


class OutputRegionEnvelope(TypedDict):
    target: JsonValue
    source: JsonValue
    start: JsonValue
    end: JsonValue


RelationEnvelope: TypeAlias = (
    SourceRelationEnvelope | ContextEnvelope | DependencyEnvelope | GroupEnvelope | OutputRegionEnvelope
)


class DatumDeclarationEvent(TypedDict):
    op: Literal["declare_datum(id,text)"]
    value: DatumEnvelope


class TargetSelectionEvent(TypedDict):
    op: Literal["select_target(id)"]
    value: JsonValue


class RelationAdditionEvent(TypedDict):
    op: Literal["add_relation(kind,value)"]
    kind: RelationKind
    value: RelationEnvelope


class RecordFactDeclarationEvent(TypedDict):
    op: Literal["declare_record_fact(kind,value)"]
    kind: RecordFactKind
    value: JsonValue


class CloseDeclarationEvent(TypedDict):
    op: Literal["close_declaration"]


class ValidateEvent(TypedDict):
    op: Literal["validate"]


TraceEvent: TypeAlias = (
    DatumDeclarationEvent
    | TargetSelectionEvent
    | RelationAdditionEvent
    | RecordFactDeclarationEvent
    | CloseDeclarationEvent
    | ValidateEvent
)


class DataDeclaration(TypedDict):
    """Neutral static data-graph declaration fixture."""

    kind: Literal["data"]
    datums: list[DatumEnvelope]
    targets: list[JsonValue]
    source_relations: list[SourceRelationEnvelope]
    contexts: list[ContextEnvelope]
    dependencies: list[DependencyEnvelope]
    coherence: list[GroupEnvelope]
    atomic: list[GroupEnvelope]
    output_regions: list[OutputRegionEnvelope]
    limits: JsonObject


class RecordDeclaration(TypedDict):
    """Neutral canonical-record constructor fixture."""

    kind: Literal["record"]
    boundary: RecordBoundary
    facts: JsonObject


Declaration: TypeAlias = DataDeclaration | RecordDeclaration


class AcceptResult(TypedDict):
    verdict: Literal["accept"]
    normalized: JsonObject


class RejectResult(TypedDict):
    verdict: Literal["reject"]
    code: ValidationCode


ValidationResult: TypeAlias = AcceptResult | RejectResult


class CaseInput(TypedDict):
    declaration: Declaration


class FixtureCase(TypedDict):
    """One fully described finite reference fixture."""

    case_id: str
    family: Family
    declaration: Declaration
    expected: ValidationResult
    trace: list[TraceEvent]


def load_data_cases() -> tuple[FixtureCase, ...]:
    """Decode reviewed, digest-authenticated bytes into their original fixture shapes."""
    return tuple(cast(list[FixtureCase], json.loads(corpus_bytes("data"))))
