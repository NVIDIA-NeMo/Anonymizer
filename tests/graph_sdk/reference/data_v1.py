# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent finite reference for data-graph and canonical-record semantics."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Literal, NotRequired, TypeAlias, TypedDict, cast

JsonScalar: TypeAlias = str | int | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
Ref: TypeAlias = tuple[str, int]


class TraceEvent(TypedDict):
    """One JSON-compatible declaration-trace event."""

    op: str
    kind: NotRequired[str]
    value: NotRequired[JsonValue]


class DataDeclaration(TypedDict):
    """Neutral static data-graph declaration fixture."""

    kind: Literal["data"]
    datums: list[JsonValue]
    targets: list[JsonValue]
    source_relations: list[JsonValue]
    contexts: list[JsonValue]
    dependencies: list[JsonValue]
    coherence: list[JsonValue]
    atomic: list[JsonValue]
    output_regions: list[JsonValue]
    limits: JsonObject


class RecordDeclaration(TypedDict):
    """Neutral canonical-record constructor fixture."""

    kind: Literal["record"]
    boundary: str
    facts: JsonObject


Declaration: TypeAlias = DataDeclaration | RecordDeclaration


class AcceptResult(TypedDict):
    verdict: Literal["accept"]
    normalized: JsonObject


class RejectResult(TypedDict):
    verdict: Literal["reject"]
    code: str


ValidationResult: TypeAlias = AcceptResult | RejectResult


class CaseInput(TypedDict):
    declaration: Declaration


class FixtureCase(TypedDict):
    """One fully described finite reference fixture."""

    case_id: str
    family: str
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
RELATION_KEYS = (
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


class _Reject(Exception):
    def __init__(self, code: str) -> None:
        self.code = code


def canonical_bytes(cases: tuple[FixtureCase, ...]) -> bytes:
    """Encode cases in the frozen JSON representation."""
    return (
        json.dumps(cases, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def validate_case(case: CaseInput | FixtureCase) -> ValidationResult:
    """Validate one neutral data or record declaration."""
    declaration = _as_object(case.get("declaration"))
    try:
        if declaration.get("kind") == "data":
            normalized = _validate_data(declaration)
        elif declaration.get("kind") == "record":
            normalized = _validate_record(declaration)
        else:
            raise _Reject("invalid_value")
    except _Reject as rejection:
        return {"verdict": "reject", "code": rejection.code}
    return {"verdict": "accept", "normalized": normalized}


def generate_cases() -> tuple[FixtureCase, ...]:
    """Enumerate the complete finite v1 domain in stable order."""
    cases: list[FixtureCase] = []
    for family, declaration, label in _base_cases():
        declarations = _limit_cases(declaration) if declaration.get("kind") == "data" else [(declaration, label)]
        for limited, suffix in declarations:
            variants = _data_variants(limited) if limited.get("kind") == "data" else [limited]
            for variant in variants:
                typed_declaration = _parse_declaration(variant)
                expected = validate_case({"declaration": typed_declaration})
                case_id = f"{family}-{len(cases):06d}"
                if label or suffix:
                    case_id += f"-{label}{suffix}"
                cases.append(
                    {
                        "case_id": case_id,
                        "family": family,
                        "declaration": typed_declaration,
                        "expected": expected,
                        "trace": _make_trace(variant),
                    }
                )
    return tuple(cases)


def _parse_cases(value: object) -> tuple[FixtureCase, ...]:
    """Validate untrusted decoded JSON before exposing typed frozen fixtures."""
    if not isinstance(value, list):
        raise ValueError("reference corpus must be a JSON array")
    cases: list[FixtureCase] = []
    try:
        for raw_case in value:
            _require_json_value(raw_case)
            case = _as_object(raw_case)
            case_id = _string(case.get("case_id"))
            family = _string(case.get("family"))
            declaration = _parse_declaration(case.get("declaration"))
            expected = _parse_validation_result(case.get("expected"))
            trace = _parse_trace(case.get("trace"))
            cases.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "declaration": declaration,
                    "expected": expected,
                    "trace": trace,
                }
            )
    except _Reject as rejection:
        raise ValueError("reference corpus has invalid JSON fixture structure") from rejection
    return tuple(cases)


def _require_json_value(value: object) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _require_json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _require_json_value(item)
        return
    raise _Reject("invalid_type")


def _parse_declaration(value: JsonValue | object) -> Declaration:
    declaration = _as_object(value)
    kind = declaration.get("kind")
    if kind == "data":
        for key in ("datums", "targets", *RELATION_KEYS):
            _as_list(declaration.get(key))
        _as_object(declaration.get("limits"))
        return cast(DataDeclaration, declaration)
    if kind == "record":
        _string(declaration.get("boundary"))
        _as_object(declaration.get("facts"))
        return cast(RecordDeclaration, declaration)
    raise _Reject("invalid_value")


def _case_input(declaration: JsonObject | Declaration) -> CaseInput:
    return {"declaration": _parse_declaration(declaration)}


def _parse_validation_result(value: JsonValue | object) -> ValidationResult:
    result = _as_object(value)
    if result.get("verdict") == "accept":
        _as_object(result.get("normalized"))
        return cast(AcceptResult, result)
    if result.get("verdict") == "reject":
        _string(result.get("code"))
        return cast(RejectResult, result)
    raise _Reject("invalid_value")


def _parse_trace(value: JsonValue | object) -> list[TraceEvent]:
    trace: list[TraceEvent] = []
    for raw_event in _as_list(value):
        event = _as_object(raw_event)
        _string(event.get("op"))
        if "kind" in event:
            _string(event["kind"])
        trace.append(cast(TraceEvent, event))
    return trace


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


def _json_ref(ref: Ref) -> list[JsonValue]:
    return [ref[0], ref[1]]


def _datum_map(declaration: JsonObject) -> tuple[list[tuple[Ref, str]], dict[Ref, str]]:
    datums: list[tuple[Ref, str]] = []
    for raw in _as_list(declaration.get("datums")):
        datum = _as_object(raw)
        identifier = _ref(datum.get("id"))
        text = datum.get("text")
        if not isinstance(text, str):
            raise _Reject("invalid_type")
        datums.append((identifier, text))
    if any(identifier[0] != "local" for identifier, _ in datums):
        raise _Reject("foreign_owner")
    if len({identifier for identifier, _ in datums}) != len(datums):
        raise _Reject("duplicate")
    return datums, dict(datums)


def _raw_counts(declaration: Mapping[str, JsonValue]) -> dict[str, int]:
    datums = _as_list(declaration.get("datums"))
    targets = _as_list(declaration.get("targets"))
    declarations = sum(len(_as_list(declaration.get(key))) for key in RELATION_KEYS)
    group_members = sum(
        len(_as_list(_as_object(group).get("members")))
        for key in ("coherence", "atomic")
        for group in _as_list(declaration.get(key))
    )
    text_bytes = 0
    for raw in datums:
        text = _as_object(raw).get("text")
        if not isinstance(text, str):
            raise _Reject("invalid_type")
        try:
            text_bytes += len(text.encode("utf-8", "strict"))
        except UnicodeEncodeError:
            raise _Reject("invalid_value") from None
    return {
        "max_datums": len(datums),
        "max_targets": len(targets),
        "max_text_bytes": text_bytes,
        "max_declarations": declarations,
        "max_group_members": group_members,
    }


def _validate_data(declaration: JsonObject) -> JsonObject:
    limits = _as_object(declaration.get("limits"))
    parsed_limits: dict[str, int] = {}
    for key in LIMIT_KEYS:
        parsed_limits[key] = _as_int(limits.get(key))
    counts = _raw_counts(declaration)
    if any(counts[key] > parsed_limits[key] for key in LIMIT_KEYS):
        raise _Reject("limit_exceeded")

    datums, texts = _datum_map(declaration)
    raw_targets = [_ref(value) for value in _as_list(declaration.get("targets"))]
    local_ids = set(texts)
    if any(identifier[0] != "local" for identifier in raw_targets):
        raise _Reject("foreign_owner")
    if len(set(raw_targets)) != len(raw_targets):
        raise _Reject("duplicate")
    if any(identifier not in local_ids for identifier in raw_targets):
        raise _Reject("missing")
    targets = set(raw_targets)

    sources = [_parse_source(value) for value in _as_list(declaration.get("source_relations"))]
    contexts = [_parse_context(value) for value in _as_list(declaration.get("contexts"))]
    dependencies = [_parse_dependency(value) for value in _as_list(declaration.get("dependencies"))]
    coherence = [_parse_group(value) for value in _as_list(declaration.get("coherence"))]
    atomic = [_parse_group(value) for value in _as_list(declaration.get("atomic"))]
    regions = [_parse_region(value) for value in _as_list(declaration.get("output_regions"))]

    all_refs = _relation_refs(sources, contexts, dependencies, coherence, atomic, regions)
    if any(identifier[0] != "local" for identifier in all_refs):
        raise _Reject("foreign_owner")
    if any(identifier not in local_ids for identifier in all_refs):
        raise _Reject("missing")

    _check_required_targets(targets, contexts, dependencies, coherence, atomic, regions)
    if _has_duplicate_sources(sources) or _has_duplicate_region_targets(regions):
        raise _Reject("duplicate")
    if any(len(group) != len(set(group)) for group in (*coherence, *atomic)):
        raise _Reject("duplicate")
    _check_ranges(texts, sources, contexts, regions)
    region_targets = {target for target, _, _, _ in regions}
    if any(derived in targets and derived not in region_targets and start == end for derived, _, start, end in sources):
        raise _Reject("invalid_range")
    _check_group_overlap(coherence)
    _check_group_overlap(atomic)
    if _has_cycles([(derived, source) for derived, source, _, _ in sources]):
        raise _Reject("cycle")
    source_by_derived = {derived: (source, start, end) for derived, source, start, end in sources}
    effective = [_effective_range(target, texts, source_by_derived, regions) for target in targets]
    _check_ownership_overlap(effective)
    if _has_cycles(dependencies):
        raise _Reject("cycle")
    _check_slices(texts, sources, regions)
    normalized_coherence = _normalize_groups(targets, coherence)
    normalized_atomic = _normalize_groups(targets, atomic)
    return {
        "datums": [_datum_json(identifier, text) for identifier, text in sorted(datums)],
        "targets": [_json_ref(identifier) for identifier in sorted(targets)],
        "source_relations": [_source_json(value) for value in sorted(set(sources))],
        "contexts": [_context_json(value) for value in sorted(set(contexts))],
        "dependencies": [_dependency_json(value) for value in sorted(set(dependencies))],
        "coherence": _groups_json(normalized_coherence),
        "atomic": _groups_json(normalized_atomic),
        "output_regions": [_region_json(value) for value in sorted(set(regions))],
        "effective_ownership": [_ownership_json(value) for value in sorted(effective)],
        "limits": parsed_limits,
    }


def _parse_source(value: JsonValue) -> tuple[Ref, Ref, int, int]:
    item = _as_object(value)
    return _ref(item.get("derived")), _ref(item.get("source")), _offset(item.get("start")), _offset(item.get("end"))


def _parse_context(value: JsonValue) -> tuple[Ref, Ref, int, int]:
    item = _as_object(value)
    return _ref(item.get("target")), _ref(item.get("source")), _offset(item.get("start")), _offset(item.get("end"))


def _parse_dependency(value: JsonValue) -> tuple[Ref, Ref]:
    item = _as_object(value)
    return _ref(item.get("prerequisite")), _ref(item.get("dependent"))


def _parse_group(value: JsonValue) -> tuple[Ref, ...]:
    item = _as_object(value)
    return tuple(_ref(member) for member in _as_list(item.get("members")))


def _parse_region(value: JsonValue) -> tuple[Ref, Ref, int, int]:
    item = _as_object(value)
    return _ref(item.get("target")), _ref(item.get("source")), _offset(item.get("start")), _offset(item.get("end"))


def _offset(value: JsonValue | object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _Reject("invalid_type")
    return value


def _relation_refs(
    sources: Sequence[tuple[Ref, Ref, int, int]],
    contexts: Sequence[tuple[Ref, Ref, int, int]],
    dependencies: Sequence[tuple[Ref, Ref]],
    coherence: Sequence[tuple[Ref, ...]],
    atomic: Sequence[tuple[Ref, ...]],
    regions: Sequence[tuple[Ref, Ref, int, int]],
) -> list[Ref]:
    refs = [ref for relation in (*sources, *contexts, *regions) for ref in relation[:2]]
    refs.extend(ref for relation in dependencies for ref in relation)
    refs.extend(ref for group in (*coherence, *atomic) for ref in group)
    return refs


def _check_required_targets(
    targets: set[Ref],
    contexts: Sequence[tuple[Ref, Ref, int, int]],
    dependencies: Sequence[tuple[Ref, Ref]],
    coherence: Sequence[tuple[Ref, ...]],
    atomic: Sequence[tuple[Ref, ...]],
    regions: Sequence[tuple[Ref, Ref, int, int]],
) -> None:
    required = [target for target, _, _, _ in contexts]
    required.extend(ref for edge in dependencies for ref in edge)
    required.extend(ref for group in (*coherence, *atomic) for ref in group)
    required.extend(target for target, _, _, _ in regions)
    if any(ref not in targets for ref in required):
        raise _Reject("invalid_value")


def _has_duplicate_sources(sources: Sequence[tuple[Ref, Ref, int, int]]) -> bool:
    by_derived: dict[Ref, tuple[Ref, Ref, int, int]] = {}
    for source in sources:
        prior = by_derived.get(source[0])
        if prior is not None and prior != source:
            return True
        by_derived[source[0]] = source
    return False


def _has_duplicate_region_targets(regions: Sequence[tuple[Ref, Ref, int, int]]) -> bool:
    targets = [region[0] for region in regions]
    return len(targets) != len(set(targets))


def _check_ranges(
    texts: dict[Ref, str],
    sources: Sequence[tuple[Ref, Ref, int, int]],
    contexts: Sequence[tuple[Ref, Ref, int, int]],
    regions: Sequence[tuple[Ref, Ref, int, int]],
) -> None:
    for _, source, start, end in sources:
        if not 0 <= start <= end <= len(texts[source]):
            raise _Reject("invalid_range")
    for _, source, start, end in contexts:
        if not 0 <= start <= end <= len(texts[source]):
            raise _Reject("invalid_range")
    for _, source, start, end in regions:
        if not 0 <= start < end <= len(texts[source]):
            raise _Reject("invalid_range")


def _check_group_overlap(groups: Sequence[tuple[Ref, ...]]) -> None:
    unique = {frozenset(group) for group in groups}
    if frozenset() in unique:
        raise _Reject("invalid_value")
    ordered = list(unique)
    if any(ordered[i] & ordered[j] for i in range(len(ordered)) for j in range(i + 1, len(ordered))):
        raise _Reject("overlap")


def _has_cycles(edges: Sequence[tuple[Ref, Ref]]) -> bool:
    adjacency: dict[Ref, set[Ref]] = {}
    for start, end in edges:
        adjacency.setdefault(start, set()).add(end)
    for origin in adjacency:
        frontier = [(origin, 0)]
        while frontier:
            node, depth = frontier.pop()
            if depth >= 3:
                continue
            for successor in adjacency.get(node, set()):
                if successor == origin:
                    return True
                frontier.append((successor, depth + 1))
    return False


def _check_slices(
    texts: dict[Ref, str],
    sources: Sequence[tuple[Ref, Ref, int, int]],
    regions: Sequence[tuple[Ref, Ref, int, int]],
) -> None:
    source_by_derived = {derived: (source, start, end) for derived, source, start, end in sources}
    for derived, source, start, end in sources:
        if texts[derived] != texts[source][start:end]:
            raise _Reject("contradictory")
    for target, source, start, end in regions:
        if texts[target] != texts[source][start:end]:
            raise _Reject("contradictory")
        relation = source_by_derived.get(target)
        if relation is not None and relation != (source, start, end):
            raise _Reject("contradictory")


def _effective_range(
    target: Ref,
    texts: dict[Ref, str],
    source_by_derived: dict[Ref, tuple[Ref, int, int]],
    regions: Sequence[tuple[Ref, Ref, int, int]],
) -> tuple[Ref, Ref, int, int]:
    explicit = next((region for region in regions if region[0] == target), None)
    if explicit is None:
        source, start, end = source_by_derived.get(target, (target, 0, len(texts[target])))
    else:
        _, source, start, end = explicit
    while source in source_by_derived:
        parent, parent_start, _ = source_by_derived[source]
        start += parent_start
        end += parent_start
        source = parent
    return target, source, start, end


def _check_ownership_overlap(ranges: Sequence[tuple[Ref, Ref, int, int]]) -> None:
    for left, right in itertools.combinations(ranges, 2):
        if left[1] == right[1] and max(left[2], right[2]) < min(left[3], right[3]):
            raise _Reject("overlap")


def _normalize_groups(targets: set[Ref], groups: Sequence[tuple[Ref, ...]]) -> set[frozenset[Ref]]:
    normalized = {frozenset(group) for group in groups}
    mentioned = set().union(*normalized) if normalized else set()
    normalized.update(frozenset({target}) for target in targets - mentioned)
    return normalized


def _datum_json(identifier: Ref, text: str) -> JsonObject:
    return {"id": _json_ref(identifier), "text": text}


def _source_json(value: tuple[Ref, Ref, int, int]) -> JsonObject:
    return {"derived": _json_ref(value[0]), "source": _json_ref(value[1]), "start": value[2], "end": value[3]}


def _context_json(value: tuple[Ref, Ref, int, int]) -> JsonObject:
    return {"target": _json_ref(value[0]), "source": _json_ref(value[1]), "start": value[2], "end": value[3]}


def _dependency_json(value: tuple[Ref, Ref]) -> JsonObject:
    return {"prerequisite": _json_ref(value[0]), "dependent": _json_ref(value[1])}


def _region_json(value: tuple[Ref, Ref, int, int]) -> JsonObject:
    return {"target": _json_ref(value[0]), "source": _json_ref(value[1]), "start": value[2], "end": value[3]}


def _ownership_json(value: tuple[Ref, Ref, int, int]) -> JsonObject:
    return {"target": _json_ref(value[0]), "source": _json_ref(value[1]), "start": value[2], "end": value[3]}


def _groups_json(groups: set[frozenset[Ref]]) -> list[JsonValue]:
    return [
        [_json_ref(member) for member in sorted(group)] for group in sorted(groups, key=lambda group: sorted(group))
    ]


Activation: TypeAlias = tuple[str, int, "Activation | None", int | None]


def _validate_record(declaration: JsonObject) -> JsonObject:
    boundary = declaration.get("boundary")
    facts = _as_object(declaration.get("facts"))
    validators = {
        "activation": _activation_key,
        "artifact": _artifact_key,
        "absence": _absence_key,
        "evidence": _evidence,
        "membership": _membership,
        "terminal": _terminal_key,
        "status": _status_target,
        "canonical": _validate_canonical,
    }
    validator = validators.get(boundary)
    if validator is None:
        raise _Reject("invalid_value")
    validator(facts)
    return deepcopy(facts)


def _string(value: JsonValue | object) -> str:
    if not isinstance(value, str):
        raise _Reject("invalid_type")
    return value


def _boolean(value: JsonValue | object) -> bool:
    if not isinstance(value, bool):
        raise _Reject("invalid_type")
    return value


def _activation_key(value: JsonObject) -> Activation:
    invocation = _string(value.get("invocation"))
    occurrence_value = value.get("occurrence")
    iteration_value = value.get("iteration")
    if isinstance(occurrence_value, bool) or not isinstance(occurrence_value, int):
        raise _Reject("invalid_type")
    if iteration_value is not None and (isinstance(iteration_value, bool) or not isinstance(iteration_value, int)):
        raise _Reject("invalid_type")
    parent_value = value.get("parent")
    parent = None if parent_value is None else _activation_key(_as_object(parent_value))
    if occurrence_value < 0 or (iteration_value is not None and iteration_value < 0):
        raise _Reject("invalid_value")
    if parent is not None and parent[0] != invocation:
        raise _Reject("foreign_owner")
    return invocation, occurrence_value, parent, iteration_value


def _artifact_key(value: JsonObject) -> tuple[str, int, int]:
    invocation = _string(value.get("invocation"))
    key_value = value.get("key")
    version_value = value.get("version")
    if isinstance(key_value, bool) or not isinstance(key_value, int):
        raise _Reject("invalid_type")
    if isinstance(version_value, bool) or not isinstance(version_value, int):
        raise _Reject("invalid_type")
    if key_value < 0 or version_value <= 0:
        raise _Reject("invalid_value")
    return invocation, key_value, version_value


def _absence_key(value: JsonObject) -> tuple[str, int, int]:
    invocation = _string(value.get("invocation"))
    query_value = value.get("query")
    revision_value = value.get("scope_revision")
    if isinstance(query_value, bool) or not isinstance(query_value, int):
        raise _Reject("invalid_type")
    if isinstance(revision_value, bool) or not isinstance(revision_value, int):
        raise _Reject("invalid_type")
    if query_value < 0 or revision_value <= 0:
        raise _Reject("invalid_value")
    return invocation, query_value, revision_value


def _status_target(value: JsonObject) -> Ref:
    target = _ref(value.get("target"))
    completion = _string(value.get("completion"))
    qualification = _string(value.get("qualification"))
    _boolean(value.get("artifact_available"))
    protection = _boolean(value.get("protection_available"))
    if completion not in ("pending", "closed") or qualification not in (
        "not_assessed",
        "met",
        "unmet",
        "unknown",
    ):
        raise _Reject("invalid_value")
    if protection and not (completion == "closed" and qualification == "met"):
        raise _Reject("contradictory")
    return target


def _consumed_ref(value: JsonObject) -> tuple[str, tuple[str, int, int], Ref | None]:
    kind = _string(value.get("kind"))
    if kind == "absence":
        return kind, _absence_key(_as_object(value.get("ref"))), None
    if kind not in ("artifact", "candidate", "decision"):
        raise _Reject("invalid_value")
    target = _ref(value.get("target")) if kind == "candidate" else None
    return kind, _artifact_key(_as_object(value.get("ref"))), target


def _evidence(value: JsonObject) -> tuple[tuple[str, int, int], list[tuple[str, int, int]], list[Ref]]:
    artifact = _artifact_key(_as_object(value.get("artifact")))
    consumed_artifacts: list[tuple[str, int, int]] = []
    candidate_targets: list[Ref] = []
    for raw_consumed in _as_list(value.get("consumed")):
        kind, dependency, target = _consumed_ref(_as_object(raw_consumed))
        if dependency[0] != artifact[0]:
            raise _Reject("foreign_owner")
        if kind != "absence":
            consumed_artifacts.append(dependency)
        if target is not None:
            candidate_targets.append(target)
    return artifact, consumed_artifacts, candidate_targets


def _membership(value: JsonObject) -> tuple[str, Activation | None, frozenset[Activation]]:
    invocation = _string(value.get("invocation"))
    parent_value = value.get("parent")
    parent = None if parent_value is None else _activation_key(_as_object(parent_value))
    _boolean(value.get("closed"))
    members = frozenset(_activation_key(_as_object(raw)) for raw in _as_list(value.get("members")))
    if parent is not None and parent[0] != invocation:
        raise _Reject("foreign_owner")
    if any(member[0] != invocation for member in members):
        raise _Reject("foreign_owner")
    if any(member == parent or member[2] != parent for member in members):
        raise _Reject("contradictory")
    return invocation, parent, members


def _terminal_key(value: JsonObject) -> tuple[str, Activation]:
    activation = _activation_key(_as_object(value.get("activation")))
    attempt = value.get("attempt")
    if attempt is not None:
        attempt_value = _as_object(attempt)
        _string(attempt_value.get("id"))
        if _activation_key(_as_object(attempt_value.get("activation"))) != activation:
            raise _Reject("foreign_owner")
    category = _string(value.get("category"))
    reasons = _as_list(value.get("reasons"))
    if not all(isinstance(reason, str) for reason in reasons):
        raise _Reject("invalid_type")
    allowed_reasons = {
        "execution_failed",
        "cancel_requested",
        "transport_lost",
        "missing",
        "duplicate",
        "foreign",
        "stale",
        "contradictory",
        "prerequisite",
    }
    if any(reason not in allowed_reasons for reason in reasons):
        raise _Reject("invalid_value")
    if category not in ("success", "failure", "cancelled", "lost", "blocked", "inconsistent"):
        raise _Reject("invalid_value")
    if (category == "success" and reasons) or (category != "success" and not reasons):
        raise _Reject("contradictory")
    if attempt is None and category not in ("blocked", "inconsistent"):
        raise _Reject("contradictory")
    return activation[0], activation


def _validate_canonical(facts: JsonObject) -> None:
    plan = _string(facts.get("plan"))
    invocation = _string(facts.get("invocation"))
    invocation_plan = _string(facts.get("invocation_plan"))
    _string(facts.get("graph"))
    targets = [_ref(value) for value in _as_list(facts.get("targets"))]
    memberships = [_as_object(value) for value in _as_list(facts.get("memberships"))]
    terminals = [_as_object(value) for value in _as_list(facts.get("terminals"))]
    artifacts = [_as_object(value) for value in _as_list(facts.get("artifacts"))]
    evidence = [_as_object(value) for value in _as_list(facts.get("evidence"))]
    statuses = [_as_object(value) for value in _as_list(facts.get("statuses"))]

    artifact_keys = [_artifact_key(item) for item in artifacts]
    terminal_keys = [_terminal_key(item) for item in terminals]
    status_targets = [_status_target(item) for item in statuses]
    parsed_memberships = [_membership(item) for item in memberships]
    parents = [parent for _, parent, _ in parsed_memberships]
    members = [member for _, _, manifest in parsed_memberships for member in manifest]
    parsed_evidence = [_evidence(item) for item in evidence]
    evidence_artifacts = [artifact for artifact, _, _ in parsed_evidence]
    consumed_artifacts = [item for _, consumed, _ in parsed_evidence for item in consumed]
    candidate_targets = [target for _, _, candidates in parsed_evidence for target in candidates]
    nested_invocations = [key[0] for key in (*artifact_keys, *terminal_keys, *evidence_artifacts, *consumed_artifacts)]
    nested_invocations.extend(member[0] for member in members)
    nested_invocations.extend(owner for owner, _, _ in parsed_memberships)

    target_set = set(targets)
    if any(target[0] == "local" and target not in target_set for target in (*status_targets, *candidate_targets)):
        raise _Reject("invalid_value")
    if invocation_plan != plan or any(value != invocation for value in nested_invocations):
        raise _Reject("foreign_owner")
    if any(target[0] != "local" for target in (*targets, *status_targets, *candidate_targets)):
        raise _Reject("foreign_owner")
    if len(parents) != len(set(parents)) or len(members) != len(set(members)):
        raise _Reject("duplicate")
    if len(terminal_keys) != len({activation for _, activation in terminal_keys}):
        raise _Reject("duplicate")
    if len(evidence_artifacts) != len(set(evidence_artifacts)) or len(status_targets) != len(set(status_targets)):
        raise _Reject("duplicate")
    member_set = set(members)
    if any(parent is not None and parent not in member_set for parent in parents):
        raise _Reject("missing")
    if any(activation not in member_set for _, activation in terminal_keys):
        raise _Reject("missing")
    artifact_set = set(artifact_keys)
    if any(item not in artifact_set for item in (*evidence_artifacts, *consumed_artifacts)):
        raise _Reject("missing")
    if any(target not in set(status_targets) for target in targets):
        raise _Reject("missing")


def _base_cases() -> Iterator[tuple[str, JsonObject, str]]:
    yield from _identity_cases()
    yield from _context_cases()
    yield from _dependency_cases()
    yield from _group_cases()
    yield from _range_cases()
    yield from _separation_cases()
    yield from _ownership_negative_cases()
    yield from _encoding_cases()
    yield from _precedence_cases()
    yield from _record_cases()
    yield from _record_precedence_cases()


def _data(datums: Sequence[tuple[Ref, str]], targets: Sequence[Ref] = ()) -> JsonObject:
    declaration: JsonObject = {
        "kind": "data",
        "datums": [_datum_json(identifier, text) for identifier, text in datums],
        "targets": [_json_ref(identifier) for identifier in targets],
        "source_relations": [],
        "contexts": [],
        "dependencies": [],
        "coherence": [],
        "atomic": [],
        "output_regions": [],
        "limits": {key: 1_000_000 for key in LIMIT_KEYS},
    }
    return declaration


def _identity_cases() -> Iterator[tuple[str, JsonObject, str]]:
    for count in range(4):
        identifiers = [("local", index) for index in range(count)]
        for texts in itertools.product(("x", "y"), repeat=count):
            datums = list(zip(identifiers, texts, strict=True))
            for mask in range(1 << count):
                targets = [identifier for index, identifier in enumerate(identifiers) if mask & (1 << index)]
                yield "identity", _data(datums, targets), ""
    yield (
        "identity",
        _data([(("local", 0), "x"), (("local", 1), "x")], [("local", 0), ("local", 1)]),
        "equal-text-distinct",
    )


def _directed_pairs(count: int) -> list[tuple[Ref, Ref]]:
    identifiers = [("local", index) for index in range(count)]
    return [(left, right) for left in identifiers for right in identifiers if left != right]


def _context_cases() -> Iterator[tuple[str, JsonObject, str]]:
    for count in range(4):
        datums = [(("local", index), "x") for index in range(count)]
        identifiers = [identifier for identifier, _ in datums]
        pairs = _directed_pairs(count)
        for mask in range(1 << len(pairs)):
            declaration = _data(datums, identifiers)
            declaration["contexts"] = [
                _context_json((target, source, 0, 1))
                for index, (target, source) in enumerate(pairs)
                if mask & (1 << index)
            ]
            yield "context", declaration, ""
    declaration = _data([(("local", 0), "x")], [("local", 0)])
    declaration["contexts"] = [_context_json((("local", 0), ("local", 0), 0, 1))]
    yield "context", declaration, "self"


def _dependency_cases() -> Iterator[tuple[str, JsonObject, str]]:
    for count in range(4):
        datums = [(("local", index), "x") for index in range(count)]
        identifiers = [identifier for identifier, _ in datums]
        pairs = _directed_pairs(count)
        for mask in range(1 << len(pairs)):
            declaration = _data(datums, identifiers)
            declaration["dependencies"] = [
                _dependency_json(pair) for index, pair in enumerate(pairs) if mask & (1 << index)
            ]
            yield "dependency", declaration, ""
    declaration = _data([(("local", 0), "x")], [("local", 0)])
    declaration["dependencies"] = [_dependency_json((("local", 0), ("local", 0)))]
    yield "dependency", declaration, "self"


def _partitions(items: tuple[Ref, ...]) -> tuple[tuple[tuple[Ref, ...], ...], ...]:
    if not items:
        return ((),)
    first, *rest = items
    result: list[tuple[tuple[Ref, ...], ...]] = []
    for partition in _partitions(tuple(rest)):
        result.append(((first,), *partition))
        for index in range(len(partition)):
            merged = tuple(sorted((first, *partition[index])))
            result.append((*partition[:index], merged, *partition[index + 1 :]))
    unique = {tuple(sorted(partition)) for partition in result}
    return tuple(sorted(unique))


def _group_json(members: Iterable[Ref]) -> JsonObject:
    return {"members": [_json_ref(member) for member in members]}


def _group_cases() -> Iterator[tuple[str, JsonObject, str]]:
    for count in range(4):
        datums = [(("local", index), "x") for index in range(count)]
        identifiers = tuple(identifier for identifier, _ in datums)
        for size in range(count + 1):
            for targets in itertools.combinations(identifiers, size):
                for coherence, atomic in itertools.product(_partitions(targets), repeat=2):
                    declaration = _data(datums, targets)
                    declaration["coherence"] = [_group_json(group) for group in coherence]
                    declaration["atomic"] = [_group_json(group) for group in atomic]
                    yield "groups", declaration, ""
    base = _data([(("local", index), "x") for index in range(3)], [("local", index) for index in range(3)])
    specials = (
        ("identical", "coherence", [[("local", 0), ("local", 1)], [("local", 0), ("local", 1)]]),
        ("duplicate-member", "coherence", [[("local", 0), ("local", 0)]]),
        ("empty", "coherence", [[]]),
        ("coherence-partial", "coherence", [[("local", 0), ("local", 1)], [("local", 1), ("local", 2)]]),
        ("coherence-subset", "coherence", [[("local", 0), ("local", 1)], [("local", 0)]]),
        ("atomic-partial", "atomic", [[("local", 0), ("local", 1)], [("local", 1), ("local", 2)]]),
        ("atomic-subset", "atomic", [[("local", 0), ("local", 1)], [("local", 0)]]),
        ("foreign", "coherence", [[("foreign", 0)]]),
        ("missing", "coherence", [[("local", 99)]]),
    )
    for label, key, groups in specials:
        declaration = deepcopy(base)
        declaration[key] = [_group_json(group) for group in groups]
        yield "groups", declaration, label
    declaration = deepcopy(base)
    declaration["targets"] = [_json_ref(("local", 0)), _json_ref(("local", 1))]
    declaration["coherence"] = [_group_json([("local", 2)])]
    yield "groups", declaration, "nontarget"


def _range_cases() -> Iterator[tuple[str, JsonObject, str]]:
    intervals = [(start, end) for start in range(4) for end in range(start + 1, 5)]
    for left, right in itertools.product(intervals, repeat=2):
        source = ("local", 0)
        read = _data([(source, "abcd"), (("local", 1), "x"), (("local", 2), "x")], [("local", 1), ("local", 2)])
        read["contexts"] = [
            _context_json((("local", 1), source, *left)),
            _context_json((("local", 2), source, *right)),
        ]
        yield "ranges", read, "reads"
        owned = _data(
            [(source, "abcd"), (("local", 1), "abcd"[slice(*left)]), (("local", 2), "abcd"[slice(*right)])],
            [("local", 1), ("local", 2)],
        )
        owned["source_relations"] = [
            _source_json((("local", 1), source, *left)),
            _source_json((("local", 2), source, *right)),
        ]
        yield "ranges", owned, "outputs"
    yield from _range_specials()


def _range_specials() -> Iterator[tuple[str, JsonObject, str]]:
    base = _data([(("local", 0), "abcd"), (("local", 1), "a")], [("local", 1)])
    for label, start, end in (("negative", -1, 1), ("past-end", 0, 5), ("reversed", 2, 1)):
        declaration = deepcopy(base)
        declaration["contexts"] = [_context_json((("local", 1), ("local", 0), start, end))]
        yield "ranges", declaration, label
    for label, value in (("noninteger", "0"), ("bool-offset", True)):
        declaration = deepcopy(base)
        declaration["contexts"] = [{"target": ["local", 1], "source": ["local", 0], "start": value, "end": 1}]
        yield "ranges", declaration, label
    declaration = deepcopy(base)
    declaration["output_regions"] = [_region_json((("local", 1), ("local", 0), 0, 0))]
    yield "ranges", declaration, "explicit-zero"
    declaration = deepcopy(base)
    declaration["source_relations"] = [_source_json((("local", 1), ("local", 0), 1, 2))]
    yield "ranges", declaration, "wrong-slice"
    yield "ranges", _data([(("local", 0), "")], [("local", 0)]), "empty-whole"
    declaration = _data([(("local", 0), ""), (("local", 1), "x")], [("local", 1)])
    declaration["contexts"] = [_context_json((("local", 1), ("local", 0), 0, 0))]
    yield "ranges", declaration, "empty-read"
    declaration = _data([(("local", 0), "abcd"), (("local", 1), "ab")], [("local", 0), ("local", 1)])
    declaration["source_relations"] = [_source_json((("local", 1), ("local", 0), 0, 2))]
    yield "ranges", declaration, "whole-plus-subview"
    declaration = _data(
        [(("local", 0), "abcd"), (("local", 1), "abc"), (("local", 2), "bc"), (("local", 3), "bc")],
        [("local", 2), ("local", 3)],
    )
    declaration["source_relations"] = [
        _source_json((("local", 1), ("local", 0), 0, 3)),
        _source_json((("local", 2), ("local", 1), 1, 3)),
        _source_json((("local", 3), ("local", 0), 1, 3)),
    ]
    yield "ranges", declaration, "nested-overlap"
    declaration = _data([(("local", 0), "x"), (("local", 1), "x")], [("local", 0), ("local", 1)])
    declaration["source_relations"] = [
        _source_json((("local", 0), ("local", 1), 0, 1)),
        _source_json((("local", 1), ("local", 0), 0, 1)),
    ]
    yield "ranges", declaration, "source-cycle"
    declaration = _data([(("local", 0), "x")], [("local", 0)])
    declaration["source_relations"] = [_source_json((("local", 0), ("local", 0), 0, 2))]
    _as_object(declaration["limits"]).update(_raw_counts(declaration))
    yield "ranges", declaration, "self-source-invalid-range-before-cycle"
    declaration = deepcopy(base)
    declaration["contexts"] = [_context_json((("local", 1), ("foreign", 0), 0, 1))]
    yield "ranges", declaration, "foreign-source"


def _separation_cases() -> Iterator[tuple[str, JsonObject, str]]:
    datums = [(("local", index), "x") for index in range(3)]
    targets = [("local", index) for index in range(3)]
    singleton = [[target] for target in targets]
    joined = [[targets[0], targets[1]], [targets[2]]]
    for coherence, atomic in itertools.product((singleton, joined), repeat=2):
        declaration = _data(datums, targets)
        declaration["contexts"] = [
            _context_json((targets[0], targets[1], 0, 1)),
            _context_json((targets[1], targets[0], 0, 1)),
        ]
        declaration["coherence"] = [_group_json(group) for group in coherence]
        declaration["atomic"] = [_group_json(group) for group in atomic]
        yield "relation-separation", declaration, ""
    declaration = _data(datums, targets)
    declaration["dependencies"] = [_dependency_json((targets[0], targets[1]))]
    yield "relation-separation", declaration, "dependency"
    declaration = deepcopy(declaration)
    cast(list[JsonValue], declaration["dependencies"]).append(_dependency_json((targets[1], targets[0])))
    yield "relation-separation", declaration, "dependency-cycle"


def _ownership_negative_cases() -> Iterator[tuple[str, JsonObject, str]]:
    base = _data([(("local", 0), "x"), (("local", 1), "x")], [("local", 0)])
    modifications: tuple[tuple[str, str, JsonValue], ...] = (
        ("foreign-target", "targets", [["foreign", 0]]),
        ("missing-target", "targets", [["local", 99]]),
        ("duplicate-target", "targets", [["local", 0], ["local", 0]]),
        ("absent-version", "targets", [["local", 2]]),
        (
            "conflicting-source",
            "source_relations",
            [
                _source_json((("local", 0), ("local", 1), 0, 1)),
                _source_json((("local", 0), ("local", 0), 0, 1)),
            ],
        ),
        (
            "duplicate-region",
            "output_regions",
            [
                _region_json((("local", 0), ("local", 1), 0, 1)),
                _region_json((("local", 0), ("local", 0), 0, 1)),
            ],
        ),
        ("foreign-root", "output_regions", [_region_json((("local", 0), ("foreign", 0), 0, 1))]),
    )
    for label, key, value in modifications:
        declaration = deepcopy(base)
        declaration[key] = value
        yield "ownership-negatives", declaration, label
    declaration = deepcopy(base)
    declaration["datums"] = [*_as_list(declaration["datums"]), _datum_json(("local", 0), "x")]
    yield "ownership-negatives", declaration, "duplicate-registry"


def _encoding_cases() -> Iterator[tuple[str, JsonObject, str]]:
    for label, text in (("empty", ""), ("e-acute", "é"), ("emoji", "😀")):
        yield "encoding-bounds", _data([(("local", 0), text)], [("local", 0)]), label
    yield "encoding-bounds", _data([(("local", 0), "x"), (("local", 1), "x")], [("local", 0)]), "repeated"
    yield "encoding-bounds", _data([(("local", 0), "\ud800")], []), "surrogate"
    for key, value, label in (("max_datums", -1, "negative-limit"), ("max_targets", True, "bool-limit")):
        declaration = _data([], [])
        cast(JsonObject, declaration["limits"])[key] = value
        yield "encoding-bounds", declaration, label
    yield "encoding-bounds", _data([], []), "zero-counts"


def _precedence_cases() -> Iterator[tuple[str, JsonObject, str]]:
    targets = [("local", index) for index in range(3)]
    base = _data([(target, "x") for target in targets], targets)
    cases: list[JsonObject] = []
    case = deepcopy(base)
    cast(JsonObject, case["limits"])["max_datums"] = True
    cast(JsonObject, case["limits"])["max_targets"] = -1
    cases.append(case)
    case = deepcopy(base)
    cast(JsonObject, case["limits"])["max_targets"] = -1
    cast(JsonObject, case["limits"])["max_datums"] = 2
    cases.append(case)
    case = deepcopy(base)
    cast(JsonObject, case["limits"])["max_datums"] = 2
    case["targets"] = [["foreign", 0]]
    cases.append(case)
    case = deepcopy(base)
    case["targets"] = [["local", 0], ["local", 0], ["foreign", 0]]
    cases.append(case)
    case = deepcopy(base)
    case["targets"] = [["local", 0], ["local", 0], ["local", 99]]
    cases.append(case)
    case = deepcopy(base)
    case["targets"] = [["local", 0], ["local", 1], ["local", 99]]
    case["contexts"] = [_context_json((targets[0], targets[1], 0, 2))]
    cases.append(case)
    case = deepcopy(base)
    case["contexts"] = [_context_json((targets[0], targets[1], 0, 2))]
    case["coherence"] = [_group_json(targets[:2]), _group_json(targets[1:])]
    cases.append(case)
    case = deepcopy(base)
    case["coherence"] = [_group_json(targets[:2]), _group_json(targets[1:])]
    case["dependencies"] = [_dependency_json((targets[0], targets[1])), _dependency_json((targets[1], targets[0]))]
    cases.append(case)
    case = _data([(targets[0], "x"), (targets[1], "x"), (targets[2], "y")], targets[:2])
    case["source_relations"] = [_source_json((targets[0], targets[2], 0, 1))]
    case["dependencies"] = [_dependency_json((targets[0], targets[1])), _dependency_json((targets[1], targets[0]))]
    cases.append(case)
    case = deepcopy(base)
    case["dependencies"] = [_dependency_json((targets[0], ("local", 99))), _dependency_json((targets[1], targets[1]))]
    cases.append(case)
    case = deepcopy(base)
    case["contexts"] = [_context_json((targets[0], targets[2], 0, 2))]
    case["dependencies"] = [_dependency_json((targets[0], targets[1])), _dependency_json((targets[1], targets[0]))]
    cases.append(case)
    for index, declaration in enumerate(cases, 1):
        yield "precedence", declaration, f"p{index}"


def _record(boundary: str, **facts: JsonValue) -> JsonObject:
    return {"kind": "record", "boundary": boundary, "facts": facts}


def _activation(
    occurrence: int | bool = 0,
    *,
    invocation: str = "I",
    parent: JsonObject | None = None,
    iteration: int | bool | None = None,
) -> JsonObject:
    return {
        "invocation": invocation,
        "occurrence": occurrence,
        "parent": deepcopy(parent),
        "iteration": iteration,
    }


def _status(
    target: Ref = ("local", 0),
    *,
    completion: str = "closed",
    qualification: str = "met",
    artifact_available: bool = True,
    protection_available: bool = False,
) -> JsonObject:
    return {
        "target": _json_ref(target),
        "completion": completion,
        "qualification": qualification,
        "artifact_available": artifact_available,
        "protection_available": protection_available,
    }


def _terminal(
    *,
    activation: JsonObject | None = None,
    attempt: str | None = "T",
    attempt_activation: JsonObject | None = None,
    category: str = "success",
    reasons: Sequence[str] = (),
) -> JsonObject:
    activation = _activation() if activation is None else activation
    owned_activation = activation if attempt_activation is None else attempt_activation
    return {
        "activation": deepcopy(activation),
        "attempt": None if attempt is None else {"id": attempt, "activation": deepcopy(owned_activation)},
        "category": category,
        "reasons": list(reasons),
    }


def _artifact(*, invocation: str = "I", key: int | bool = 0, version: int | bool = 1) -> JsonObject:
    return {"invocation": invocation, "key": key, "version": version}


def _canonical() -> JsonObject:
    evidence_artifact = _artifact(key=0)
    dependency_artifact = _artifact(key=1)
    activation = _activation()
    return {
        "plan": "P",
        "invocation": "I",
        "invocation_plan": "P",
        "graph": "G",
        "targets": [_json_ref(("local", 0))],
        "memberships": [
            {
                "invocation": "I",
                "parent": None,
                "members": [activation],
                "closed": True,
            }
        ],
        "terminals": [_terminal(activation=activation)],
        "artifacts": [evidence_artifact, dependency_artifact],
        "evidence": [
            {
                "artifact": evidence_artifact,
                "consumed": [{"kind": "artifact", "ref": dependency_artifact}],
            }
        ],
        "statuses": [_status()],
    }


def _record_cases() -> Iterator[tuple[str, JsonObject, str]]:
    statuses = (
        ("closed-unassessed", "closed", "not_assessed", True, False),
        ("closed-unmet", "closed", "unmet", True, False),
        ("pending-unknown", "pending", "unknown", False, False),
        ("closed-met-withheld", "closed", "met", True, False),
        ("closed-met-available", "closed", "met", True, True),
        ("pending-protected", "pending", "not_assessed", True, True),
        ("unmet-protected", "closed", "unmet", True, True),
    )
    for label, completion, qualification, artifact, protection in statuses:
        yield (
            "record",
            _record(
                "status",
                target=_json_ref(("local", 0)),
                completion=completion,
                qualification=qualification,
                artifact_available=artifact,
                protection_available=protection,
            ),
            label,
        )
    terminal_reasons: dict[str, list[str]] = {
        "success": [],
        "failure": ["execution_failed"],
        "cancelled": ["cancel_requested"],
        "lost": ["transport_lost"],
        "blocked": ["prerequisite"],
        "inconsistent": ["contradictory"],
    }
    for category, reasons in terminal_reasons.items():
        attempt = None if category in ("blocked", "inconsistent") else "T"
        facts = _terminal(attempt=attempt, category=category, reasons=reasons)
        yield "record", {"kind": "record", "boundary": "terminal", "facts": facts}, f"terminal-{category}"
    yield (
        "record",
        {"kind": "record", "boundary": "terminal", "facts": _terminal(reasons=["execution_failed"])},
        "success-reason",
    )
    yield (
        "record",
        {"kind": "record", "boundary": "terminal", "facts": _terminal(category="failure")},
        "failure-no-reason",
    )
    yield (
        "record",
        {
            "kind": "record",
            "boundary": "terminal",
            "facts": _terminal(attempt=None, category="failure", reasons=["execution_failed"]),
        },
        "failure-no-attempt",
    )
    yield (
        "record",
        {
            "kind": "record",
            "boundary": "terminal",
            "facts": _terminal(attempt_activation=_activation(1)),
        },
        "attempt-mismatch",
    )
    yield "record", _record("activation", **_activation()), "activation-valid"
    yield "record", _record("activation", **_activation(-1)), "activation-negative-occurrence"
    yield "record", _record("activation", **_activation(True)), "activation-bool-occurrence"
    yield "record", _record("activation", **_activation(iteration=-1)), "activation-negative-iteration"
    yield "record", _record("activation", **_activation(iteration=True)), "activation-bool-iteration"
    yield (
        "record",
        _record("activation", **_activation(1, parent=_activation(invocation="J"))),
        "activation-foreign-parent",
    )
    yield "record", _record("artifact", invocation="I", key=-1, version=1), "artifact-negative-key"
    yield "record", _record("artifact", invocation="I", key=True, version=1), "artifact-bool-key"
    yield "record", _record("absence", invocation="I", query=-1, scope_revision=1), "absence-negative-query"
    yield "record", _record("absence", invocation="I", query=True, scope_revision=1), "absence-bool-query"
    yield "record", _record("status", **_status(completion="unexpected")), "unknown-completion"
    yield "record", _record("status", **_status(qualification="unexpected")), "unknown-qualification"
    yield "record", _record("status", **{**_status(), "artifact_available": 1}), "nonbool-status"
    yield "record", _record("terminal", **_terminal(category="unexpected")), "unknown-terminal-category"
    yield (
        "record",
        _record("terminal", **_terminal(category="failure", reasons=["unexpected"])),
        "unknown-reason",
    )
    yield (
        "record",
        _record("terminal", **{**_terminal(), "category": 1}),
        "terminal-vocabulary-wrong-type",
    )
    parent = _activation(10)
    yield (
        "record",
        _record("membership", invocation="I", parent=parent, members=[parent], closed=True),
        "membership-member-equals-parent",
    )
    yield (
        "record",
        _record(
            "membership",
            invocation="I",
            parent=parent,
            members=[_activation(11, parent=_activation(12))],
            closed=True,
        ),
        "membership-parent-mismatch",
    )
    yield (
        "record",
        _record("membership", invocation="I", parent=None, members=[_activation()], closed=1),
        "membership-closed-wrong-type",
    )
    yield (
        "record",
        _record(
            "evidence",
            artifact=_artifact(invocation="I", key=0),
            consumed=[{"kind": "artifact", "ref": _artifact(invocation="J", key=1)}],
        ),
        "evidence-foreign-dependency",
    )
    yield "record", _record("artifact", invocation=1, key=0, version=1), "identity-wrong-type"
    yield "record", _record("terminal", **{**_terminal(), "reasons": [1]}), "reason-element-wrong-type"
    yield "record", _record("terminal", **{**_terminal(), "reasons": "not-a-list"}), "reasons-wrong-collection"
    yield "record", _record("status", **{**_status(), "target": "not-a-reference"}), "reference-wrong-type"
    canonical_cases: list[tuple[str, JsonObject]] = []
    facts = _canonical()
    _as_list(facts["terminals"]).append(deepcopy(_as_list(facts["terminals"])[0]))
    canonical_cases.append(("duplicate_terminal", facts))
    facts = _canonical()
    terminal = _as_object(_as_list(facts["terminals"])[0])
    replacement = _activation(9)
    terminal["activation"] = replacement
    _as_object(terminal["attempt"])["activation"] = replacement
    canonical_cases.append(("undeclared_terminal", facts))
    facts = _canonical()
    facts["artifacts"] = [_artifact(invocation="J", key=0), _artifact(invocation="J", key=1)]
    evidence = _as_object(_as_list(facts["evidence"])[0])
    _as_object(evidence["artifact"])["invocation"] = "J"
    _as_object(_as_object(_as_list(evidence["consumed"])[0])["ref"])["invocation"] = "J"
    canonical_cases.append(("foreign_invocation", facts))
    facts = _canonical()
    facts["invocation_plan"] = "Q"
    canonical_cases.append(("foreign_plan", facts))
    facts = _canonical()
    _as_list(facts["memberships"]).append(deepcopy(_as_list(facts["memberships"])[0]))
    canonical_cases.append(("duplicate_membership_parent", facts))
    facts = _canonical()
    parent = _activation(10)
    child = _activation(11, parent=parent)
    _as_list(facts["memberships"]).append(
        {
            "invocation": "I",
            "parent": parent,
            "members": [child],
            "closed": True,
        }
    )
    canonical_cases.append(("missing_parent", facts))
    facts = _canonical()
    _as_list(facts["evidence"]).append(deepcopy(_as_list(facts["evidence"])[0]))
    canonical_cases.append(("duplicate_evidence", facts))
    facts = _canonical()
    facts["artifacts"] = [_as_list(facts["artifacts"])[0]]
    canonical_cases.append(("missing_consumed_artifact", facts))
    facts = _canonical()
    facts["artifacts"] = [_as_list(facts["artifacts"])[1]]
    canonical_cases.append(("missing_evidence_artifact", facts))
    facts = _canonical()
    consumed = _as_object(_as_list(_as_object(_as_list(facts["evidence"])[0])["consumed"])[0])
    consumed["kind"] = "candidate"
    consumed["target"] = _json_ref(("foreign", 0))
    canonical_cases.append(("foreign_candidate_target", facts))
    facts = _canonical()
    consumed = _as_object(_as_list(_as_object(_as_list(facts["evidence"])[0])["consumed"])[0])
    consumed["kind"] = "candidate"
    consumed["target"] = _json_ref(("local", 2))
    canonical_cases.append(("candidate-outside-targets", facts))
    facts = _canonical()
    _as_list(facts["statuses"])[0] = _status(("foreign", 0))
    canonical_cases.append(("foreign-status-target", facts))
    facts = _canonical()
    facts["targets"] = [_json_ref(("foreign", 0))]
    facts["statuses"] = [_status(("foreign", 0))]
    canonical_cases.append(("foreign-selected-target", facts))
    for label, facts in canonical_cases:
        yield "record", {"kind": "record", "boundary": "canonical", "facts": facts}, label
    for version in (1, 0, True):
        yield "record", _record("artifact", invocation="I", key=0, version=version), f"artifact-version-{version!s}"
    for revision in (1, 0, True):
        yield (
            "record",
            _record("absence", invocation="I", query=0, scope_revision=revision),
            f"absence-revision-{revision!s}",
        )
    for available, consumed_version, label in (
        ([1], 1, "consume-v1"),
        ([1], 2, "missing-v2"),
        ([1, 2], 2, "consume-v2"),
    ):
        facts = _canonical()
        dependency = _artifact(key=1, version=consumed_version)
        facts["artifacts"] = [_artifact(key=0), *[_artifact(key=1, version=version) for version in available]]
        _as_object(_as_list(_as_object(_as_list(facts["evidence"])[0])["consumed"])[0])["ref"] = dependency
        yield "record", {"kind": "record", "boundary": "canonical", "facts": facts}, label
    facts = _canonical()
    _as_list(facts["memberships"]).append(deepcopy(_as_list(facts["memberships"])[0]))
    yield (
        "record",
        {"kind": "record", "boundary": "canonical", "facts": facts},
        "member-repeated-across-manifests-and-duplicate-parent",
    )
    facts = _canonical()
    _as_list(facts["statuses"]).append(deepcopy(_as_list(facts["statuses"])[0]))
    yield "record", {"kind": "record", "boundary": "canonical", "facts": facts}, "duplicate-status"
    facts = _canonical()
    facts["statuses"] = []
    yield "record", {"kind": "record", "boundary": "canonical", "facts": facts}, "missing-status"
    facts = _canonical()
    _as_list(facts["statuses"]).append(_status(("local", 2)))
    yield "record", {"kind": "record", "boundary": "canonical", "facts": facts}, "status-outside-targets"


def _record_precedence_cases() -> Iterator[tuple[str, JsonObject, str]]:
    rp3 = {
        "kind": "record",
        "boundary": "terminal",
        "facts": _terminal(attempt_activation=_activation(1), reasons=["execution_failed"]),
    }
    rp4 = _canonical()
    rp4["invocation_plan"] = "Q"
    _as_list(rp4["terminals"]).append(deepcopy(_as_list(rp4["terminals"])[0]))
    rp5 = _canonical()
    _as_list(rp5["evidence"]).append(deepcopy(_as_list(rp5["evidence"])[0]))
    rp5["artifacts"] = [_as_list(rp5["artifacts"])[0]]
    rp6 = _canonical()
    _as_list(rp6["statuses"]).append(_status(("local", 2)))
    dependency = _as_object(_as_list(_as_object(_as_list(rp6["evidence"])[0])["consumed"])[0])
    _as_object(dependency["ref"])["invocation"] = "J"
    _as_object(_as_object(_as_list(rp6["evidence"])[0])["artifact"])["invocation"] = "J"
    rp6["artifacts"] = [_artifact(invocation="J", key=0), _artifact(invocation="J", key=1)]
    rp7 = _canonical()
    rp7["statuses"] = []
    parent = _activation(10)
    _as_list(rp7["memberships"]).append(
        {
            "invocation": "I",
            "parent": parent,
            "members": [_activation(11, parent=parent)],
            "closed": True,
        }
    )
    terminal = _as_object(_as_list(rp7["terminals"])[0])
    replacement = _activation(9)
    terminal["activation"] = replacement
    _as_object(terminal["attempt"])["activation"] = replacement
    cases: tuple[JsonObject, ...] = (
        _record("artifact", invocation="I", key=-1, version=True),
        _record("absence", invocation="I", query=-1, scope_revision=True),
        rp3,
        {"kind": "record", "boundary": "canonical", "facts": rp4},
        {"kind": "record", "boundary": "canonical", "facts": rp5},
        {"kind": "record", "boundary": "canonical", "facts": rp6},
        {"kind": "record", "boundary": "canonical", "facts": rp7},
    )
    for index, declaration in enumerate(cases, 1):
        yield "record-precedence", declaration, f"rp{index}"


def _limit_cases(declaration: JsonObject) -> list[tuple[JsonObject, str]]:
    exact = deepcopy(declaration)
    try:
        counts = _raw_counts(exact)
    except _Reject:
        return [(exact, "")]
    expected_with_generous_limits = validate_case(_case_input(exact))
    limits = _as_object(exact["limits"])
    if expected_with_generous_limits.get("verdict") == "accept":
        limits.update(counts)
        variants = [(exact, "")]
        for key, count in counts.items():
            if count:
                limited = deepcopy(exact)
                _as_object(limited["limits"])[key] = count - 1
                variants.append((limited, f"-limit-{key}"))
        return variants
    return [(exact, "")]


def _data_variants(declaration: JsonObject) -> list[JsonObject]:
    datums = _as_list(declaration["datums"])
    datum_orders: Iterable[tuple[JsonValue, ...]]
    if len(datums) <= 3:
        datum_orders = itertools.permutations(datums)
    else:
        datum_orders = (tuple(datums), tuple(reversed(datums)))
    variants: dict[bytes, JsonObject] = {}
    for order in datum_orders:
        for reverse in (False, True):
            for rename in (False, True):
                variant = deepcopy(declaration)
                variant["datums"] = list(order)
                if reverse:
                    for key in RELATION_KEYS:
                        variant[key] = list(reversed(_as_list(variant[key])))
                if rename:
                    variant = _rename_declaration(variant)
                key = json.dumps(variant, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
                variants.setdefault(key, variant)
    return list(variants.values())


def _rename_declaration(declaration: JsonObject) -> JsonObject:
    renamed = deepcopy(declaration)

    def visit(value: JsonValue) -> JsonValue:
        if isinstance(value, list):
            if len(value) == 2 and value[0] in ("local", "foreign") and isinstance(value[1], int):
                if value[0] == "local" and value[1] != 99:
                    return [value[0], RENAME.get(value[1], value[1])]
                return value
            return [visit(item) for item in value]
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        return value

    return cast(JsonObject, visit(renamed))


def _make_trace(declaration: JsonObject | Declaration) -> list[TraceEvent]:
    raw_declaration = _as_object(declaration)
    trace: list[TraceEvent]
    if raw_declaration.get("kind") == "record":
        facts = _as_object(raw_declaration["facts"])
        trace = [
            {"op": "declare_record_fact(kind,value)", "kind": key, "value": deepcopy(value)}
            for key, value in facts.items()
        ]
    else:
        trace = []
        for datum in _as_list(raw_declaration["datums"]):
            trace.append({"op": "declare_datum(id,text)", "value": deepcopy(datum)})
        for target in _as_list(raw_declaration["targets"]):
            trace.append({"op": "select_target(id)", "value": deepcopy(target)})
        for key in RELATION_KEYS:
            for relation in _as_list(raw_declaration[key]):
                trace.append({"op": "add_relation(kind,value)", "kind": key, "value": deepcopy(relation)})
    trace.append({"op": "close_declaration"})
    trace.append({"op": "validate"})
    return trace


def _independent(left: TraceEvent, right: TraceEvent, declaration: DataDeclaration) -> bool:
    """Classify independent static additions from their declared effects."""
    try:
        left_key = _addition_key(left)
        right_key = _addition_key(right)
        if left_key is None or right_key is None or left_key == right_key:
            return False
        limit_keys = tuple(sorted(_addition_limit_keys(left) | _addition_limit_keys(right)))
        if not _limits_permit(declaration, limit_keys):
            return False

        left_created = _created_datum(left)
        right_created = _created_datum(right)
        if left_created in _event_refs(right) or right_created in _event_refs(left):
            return False

        left_target = _selected_target(left)
        right_target = _selected_target(right)
        if left_target in _event_refs(right) or right_target in _event_refs(left):
            return False

        if _groups_overlap(left, right) or _source_keys_conflict(left, right):
            return False
        if _ownership_conflicts(left, right, declaration):
            return False
    except _Reject:
        return False
    return True


def _addition_key(event: TraceEvent) -> tuple[str, object] | None:
    operation = event["op"]
    if operation == "declare_datum(id,text)":
        return operation, _ref(_as_object(event.get("value")).get("id"))
    if operation == "select_target(id)":
        return operation, _ref(event.get("value"))
    if operation == "add_relation(kind,value)":
        kind = event.get("kind")
        if kind not in RELATION_KEYS:
            raise _Reject("invalid_value")
        value = cast(JsonValue, event.get("value"))
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return kind, canonical
    return None


def _addition_limit_keys(event: TraceEvent) -> set[str]:
    if event["op"] == "declare_datum(id,text)":
        return {"max_datums", "max_text_bytes"}
    if event["op"] == "select_target(id)":
        return {"max_targets"}
    if event["op"] == "add_relation(kind,value)":
        keys = {"max_declarations"}
        if event.get("kind") in ("coherence", "atomic"):
            keys.add("max_group_members")
        return keys
    return set()


def _created_datum(event: TraceEvent) -> Ref | None:
    if event["op"] != "declare_datum(id,text)":
        return None
    return _ref(_as_object(event.get("value")).get("id"))


def _selected_target(event: TraceEvent) -> Ref | None:
    if event["op"] != "select_target(id)":
        return None
    return _ref(event.get("value"))


def _event_refs(event: TraceEvent) -> set[Ref]:
    if event["op"] == "select_target(id)":
        return {_ref(event.get("value"))}
    if event["op"] != "add_relation(kind,value)":
        return set()
    value = cast(JsonValue, event.get("value"))
    kind = event.get("kind")
    if kind == "source_relations":
        derived, source, _, _ = _parse_source(value)
        return {derived, source}
    if kind == "contexts":
        target, source, _, _ = _parse_context(value)
        return {target, source}
    if kind == "dependencies":
        return set(_parse_dependency(value))
    if kind in ("coherence", "atomic"):
        return set(_parse_group(value))
    if kind == "output_regions":
        target, source, _, _ = _parse_region(value)
        return {target, source}
    raise _Reject("invalid_value")


def _groups_overlap(left: TraceEvent, right: TraceEvent) -> bool:
    if left.get("kind") not in ("coherence", "atomic") or right.get("kind") not in ("coherence", "atomic"):
        return False
    left_members = set(_parse_group(cast(JsonValue, left.get("value"))))
    right_members = set(_parse_group(cast(JsonValue, right.get("value"))))
    return bool(left_members & right_members)


def _source_keys_conflict(left: TraceEvent, right: TraceEvent) -> bool:
    if left.get("kind") != "source_relations" or right.get("kind") != "source_relations":
        return False
    left_derived, _, _, _ = _parse_source(cast(JsonValue, left.get("value")))
    right_derived, _, _, _ = _parse_source(cast(JsonValue, right.get("value")))
    return left_derived == right_derived


def _ownership_conflicts(left: TraceEvent, right: TraceEvent, declaration: DataDeclaration) -> bool:
    left_root = _ownership_root(left, declaration)
    right_root = _ownership_root(right, declaration)
    return left_root is not None and left_root == right_root


def _ownership_root(event: TraceEvent, declaration: DataDeclaration) -> Ref | None:
    kind = event.get("kind")
    if kind == "source_relations":
        derived, source, _, _ = _parse_source(cast(JsonValue, event.get("value")))
        targets = {_ref(value) for value in declaration["targets"]}
        if derived not in targets:
            return None
    elif kind == "output_regions":
        _, source, _, _ = _parse_region(cast(JsonValue, event.get("value")))
    else:
        return None
    source_by_derived = {
        derived: source
        for value in declaration["source_relations"]
        for derived, source, _, _ in (_parse_source(value),)
    }
    return _root_source(source, source_by_derived)


def _root_source(source: Ref, source_by_derived: Mapping[Ref, Ref]) -> Ref:
    visited: set[Ref] = set()
    while source in source_by_derived:
        if source in visited:
            raise _Reject("cycle")
        visited.add(source)
        source = source_by_derived[source]
    return source


def _limits_permit(declaration: DataDeclaration, keys: tuple[str, ...]) -> bool:
    """Return whether shared finite counters admit both additions."""
    try:
        counts = _raw_counts(cast(JsonObject, declaration))
        limits = _as_object(declaration["limits"])
        return all(counts[key] <= _as_int(limits.get(key)) for key in keys)
    except _Reject:
        return False


def _independence_witnesses(cases: Sequence[FixtureCase]) -> Iterator[tuple[FixtureCase, int]]:
    for case in cases:
        declaration = case["declaration"]
        if declaration["kind"] != "data":
            continue
        trace = case["trace"]
        for index in range(len(trace) - 1):
            if _independent(trace[index], trace[index + 1], declaration):
                yield case, index


def _replay_trace(case: FixtureCase, trace: Sequence[TraceEvent]) -> JsonObject:
    original = _as_object(case["declaration"])
    if original.get("kind") == "record":
        facts: JsonObject = {}
        for raw_event in trace[:-2]:
            event = _as_object(raw_event)
            kind = event.get("kind")
            if not isinstance(kind, str):
                raise _Reject("invalid_type")
            facts[kind] = deepcopy(cast(JsonValue, event.get("value")))
        return {
            "kind": "record",
            "boundary": cast(JsonValue, original["boundary"]),
            "facts": facts,
        }
    replayed = _data([], [])
    replayed["limits"] = deepcopy(cast(JsonValue, original["limits"]))
    for raw_event in trace[:-2]:
        event = _as_object(raw_event)
        operation = event.get("op")
        if operation == "declare_datum(id,text)":
            _as_list(replayed["datums"]).append(deepcopy(cast(JsonValue, event["value"])))
        elif operation == "select_target(id)":
            _as_list(replayed["targets"]).append(deepcopy(cast(JsonValue, event["value"])))
        elif operation == "add_relation(kind,value)":
            kind = event.get("kind")
            if not isinstance(kind, str) or kind not in RELATION_KEYS:
                raise _Reject("invalid_value")
            _as_list(replayed[kind]).append(deepcopy(cast(JsonValue, event["value"])))
        else:
            raise _Reject("invalid_value")
    return replayed


def _raw_variant_count() -> int:
    count = 0
    for _, declaration, label in _base_cases():
        del label
        declarations = _limit_cases(declaration) if declaration.get("kind") == "data" else [(declaration, "")]
        for limited, suffix in declarations:
            del suffix
            if limited.get("kind") == "record":
                count += 1
            else:
                datum_count = len(_as_list(limited["datums"]))
                count += (len(tuple(itertools.permutations(range(datum_count)))) if datum_count <= 3 else 2) * 4
    return count
