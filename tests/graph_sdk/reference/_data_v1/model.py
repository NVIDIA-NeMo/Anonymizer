# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent data_v1 reference: model."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict, cast

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


STATIC_ALPHABET = (
    "declare_datum(id,text)",
    "select_target(id)",
    "add_relation(kind,value)",
    "close_declaration",
    "validate",
)


RECORD_ALPHABET = (
    "declare_record_fact(kind,value)",
    "close_declaration",
    "validate",
)


RELATION_KEYS: tuple[RelationKind, ...] = (
    "source_relations",
    "contexts",
    "dependencies",
    "coherence",
    "atomic",
    "output_regions",
)


LIMIT_KEYS = (
    "max_datums",
    "max_targets",
    "max_text_bytes",
    "max_declarations",
    "max_group_members",
)


RENAME = {0: 7, 1: 5, 2: 9, 3: 11}


FAMILIES: tuple[Family, ...] = (
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
)


VALIDATION_CODES: tuple[ValidationCode, ...] = (
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
)


RECORD_FACT_KEYS: dict[RecordBoundary, tuple[RecordFactKind, ...]] = {
    "activation": ("invocation", "occurrence", "parent", "iteration"),
    "artifact": ("invocation", "key", "version"),
    "absence": ("invocation", "query", "scope_revision"),
    "evidence": ("artifact", "consumed"),
    "membership": ("invocation", "parent", "members", "closed"),
    "terminal": ("activation", "attempt", "category", "reasons"),
    "status": ("target", "completion", "qualification", "artifact_available", "protection_available"),
    "canonical": (
        "plan",
        "invocation",
        "invocation_plan",
        "graph",
        "targets",
        "memberships",
        "terminals",
        "artifacts",
        "evidence",
        "statuses",
    ),
}


RELATION_FIELDS: dict[RelationKind, tuple[str, ...]] = {
    "source_relations": ("derived", "source", "start", "end"),
    "contexts": ("target", "source", "start", "end"),
    "dependencies": ("prerequisite", "dependent"),
    "coherence": ("members",),
    "atomic": ("members",),
    "output_regions": ("target", "source", "start", "end"),
}


class _Reject(Exception):
    def __init__(self, code: ValidationCode) -> None:
        self.code = code


def _as_object(value: object) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _Reject("invalid_type")
    return cast(JsonObject, value)


def _as_list(value: JsonValue | object) -> list[JsonValue]:
    if not isinstance(value, list):
        raise _Reject("invalid_type")
    return value


def _as_int(value: JsonValue | object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _Reject("invalid_type")
    if value < (1 if positive else 0):
        raise _Reject("invalid_value")
    return value


def _ref(value: JsonValue | object) -> Ref:
    values = _as_list(value)
    if len(values) != 2 or values[0] not in ("local", "foreign"):
        raise _Reject("invalid_type")
    label = _as_int(values[1])
    return cast(str, values[0]), label


Activation: TypeAlias = tuple[str, int, "Activation | None", int | None]
