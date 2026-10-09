# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent data_v1 reference: validation."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._data_v1.model import (
    FAMILIES,
    LIMIT_KEYS,
    RECORD_FACT_KEYS,
    RELATION_FIELDS,
    RELATION_KEYS,
    VALIDATION_CODES,
    AcceptResult,
    CaseInput,
    ContextEnvelope,
    DataDeclaration,
    DatumDeclarationEvent,
    DatumEnvelope,
    Declaration,
    DependencyEnvelope,
    Family,
    FixtureCase,
    JsonObject,
    JsonValue,
    OutputRegionEnvelope,
    RecordBoundary,
    RecordDeclaration,
    RecordFactDeclarationEvent,
    RecordFactKind,
    Ref,
    RejectResult,
    RelationAdditionEvent,
    RelationEnvelope,
    RelationKind,
    SourceRelationEnvelope,
    TargetSelectionEvent,
    TraceEvent,
    ValidationCode,
    ValidationResult,
    _as_int,
    _as_list,
    _as_object,
    _ref,
    _Reject,
)
from tests.graph_sdk.reference._data_v1.records import (
    _absence_key,
    _activation_key,
    _artifact_key,
    _evidence,
    _membership,
    _status_target,
    _string,
    _terminal_key,
    _validate_canonical,
)


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


def _parse_declaration(value: JsonValue | object) -> Declaration:
    declaration = _as_object(value)
    kind = declaration.get("kind")
    if kind == "data":
        _require_fields(declaration, ("kind", "datums", "targets", *RELATION_KEYS, "limits"))
        for raw_datum in _as_list(declaration.get("datums")):
            _parse_datum_envelope(raw_datum)
        for raw_target in _as_list(declaration.get("targets")):
            _parse_ref_envelope(raw_target)
        for relation_kind in RELATION_KEYS:
            for raw_relation in _as_list(declaration.get(relation_kind)):
                _parse_relation_envelope(relation_kind, raw_relation)
        limits = _as_object(declaration.get("limits"))
        _require_fields(limits, LIMIT_KEYS)
        # The cast is safe after the complete envelope grammar above. Scalar
        # values intentionally remain unvalidated for the semantic oracle.
        return cast(DataDeclaration, declaration)
    if kind == "record":
        _require_fields(declaration, ("kind", "boundary", "facts"))
        boundary = _record_boundary(declaration.get("boundary"))
        facts = _as_object(declaration.get("facts"))
        _require_fields(facts, RECORD_FACT_KEYS[boundary])
        # The boundary tag selects and validates the exact fact envelope.
        return cast(RecordDeclaration, declaration)
    raise _Reject("invalid_value")


def _case_input(declaration: JsonObject | Declaration) -> CaseInput:
    return {"declaration": _parse_declaration(declaration)}


def _parse_validation_result(value: JsonValue | object) -> ValidationResult:
    result = _as_object(value)
    if result.get("verdict") == "accept":
        _require_fields(result, ("verdict", "normalized"))
        _as_object(result.get("normalized"))
        return cast(AcceptResult, result)
    if result.get("verdict") == "reject":
        _require_fields(result, ("verdict", "code"))
        _validation_code(result.get("code"))
        return cast(RejectResult, result)
    raise _Reject("invalid_value")


def _parse_data_event(event: JsonObject, operation: str) -> TraceEvent:
    if operation == "declare_datum(id,text)":
        _require_fields(event, ("op", "value"))
        _parse_datum_envelope(event["value"])
        return cast(DatumDeclarationEvent, event)
    if operation == "select_target(id)":
        _require_fields(event, ("op", "value"))
        _parse_ref_envelope(event["value"])
        return cast(TargetSelectionEvent, event)
    if operation == "add_relation(kind,value)":
        _require_fields(event, ("op", "kind", "value"))
        relation_kind = _relation_kind(event["kind"])
        _parse_relation_envelope(relation_kind, event["value"])
        return cast(RelationAdditionEvent, event)
    raise _Reject("invalid_value")


def _parse_record_event(event: JsonObject, operation: str, boundary: RecordBoundary) -> TraceEvent:
    if operation != "declare_record_fact(kind,value)":
        raise _Reject("invalid_value")
    _require_fields(event, ("op", "kind", "value"))
    fact_kind = _record_fact_kind(event["kind"])
    if fact_kind not in RECORD_FACT_KEYS[boundary]:
        raise _Reject("invalid_value")
    return cast(RecordFactDeclarationEvent, event)


def _parse_datum_envelope(value: JsonValue | object) -> DatumEnvelope:
    datum = _as_object(value)
    _require_fields(datum, ("id", "text"))
    _parse_ref_envelope(datum["id"])
    return cast(DatumEnvelope, datum)


def _parse_relation_envelope(kind: RelationKind, value: JsonValue | object) -> RelationEnvelope:
    relation = _as_object(value)
    _require_fields(relation, RELATION_FIELDS[kind])
    ref_fields = {
        "source_relations": ("derived", "source"),
        "contexts": ("target", "source"),
        "dependencies": ("prerequisite", "dependent"),
        "output_regions": ("target", "source"),
    }
    if kind in ("coherence", "atomic"):
        for member in _as_list(relation["members"]):
            _parse_ref_envelope(member)
    else:
        for field in ref_fields[kind]:
            _parse_ref_envelope(relation[field])
    return cast(RelationEnvelope, relation)


def _parse_ref_envelope(value: JsonValue | object) -> None:
    if len(_as_list(value)) != 2:
        raise _Reject("invalid_value")


def _require_fields(value: JsonObject, required: Sequence[str]) -> None:
    if set(value) != set(required):
        raise _Reject("invalid_value")


def _closed_string(value: JsonValue | object, allowed: tuple[str, ...]) -> str:
    parsed = _string(value)
    if parsed not in allowed:
        raise _Reject("invalid_value")
    return parsed


def _family(value: JsonValue | object) -> Family:
    return cast(Family, _closed_string(value, FAMILIES))


def _relation_kind(value: JsonValue | object) -> RelationKind:
    return cast(RelationKind, _closed_string(value, RELATION_KEYS))


def _record_boundary(value: JsonValue | object) -> RecordBoundary:
    return cast(RecordBoundary, _closed_string(value, tuple(RECORD_FACT_KEYS)))


def _record_fact_kind(value: JsonValue | object) -> RecordFactKind:
    allowed = tuple(dict.fromkeys(key for keys in RECORD_FACT_KEYS.values() for key in keys))
    return cast(RecordFactKind, _closed_string(value, allowed))


def _validation_code(value: JsonValue | object) -> ValidationCode:
    return cast(ValidationCode, _closed_string(value, VALIDATION_CODES))


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


def _parse_source(value: object) -> tuple[Ref, Ref, int, int]:
    item = _as_object(value)
    return _ref(item.get("derived")), _ref(item.get("source")), _offset(item.get("start")), _offset(item.get("end"))


def _parse_context(value: object) -> tuple[Ref, Ref, int, int]:
    item = _as_object(value)
    return _ref(item.get("target")), _ref(item.get("source")), _offset(item.get("start")), _offset(item.get("end"))


def _parse_dependency(value: object) -> tuple[Ref, Ref]:
    item = _as_object(value)
    return _ref(item.get("prerequisite")), _ref(item.get("dependent"))


def _parse_group(value: object) -> tuple[Ref, ...]:
    item = _as_object(value)
    return tuple(_ref(member) for member in _as_list(item.get("members")))


def _parse_region(value: object) -> tuple[Ref, Ref, int, int]:
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


def _datum_json(identifier: Ref, text: str) -> DatumEnvelope:
    return {"id": _json_ref(identifier), "text": text}


def _source_json(value: tuple[Ref, Ref, int, int]) -> SourceRelationEnvelope:
    return {"derived": _json_ref(value[0]), "source": _json_ref(value[1]), "start": value[2], "end": value[3]}


def _context_json(value: tuple[Ref, Ref, int, int]) -> ContextEnvelope:
    return {"target": _json_ref(value[0]), "source": _json_ref(value[1]), "start": value[2], "end": value[3]}


def _dependency_json(value: tuple[Ref, Ref]) -> DependencyEnvelope:
    return {"prerequisite": _json_ref(value[0]), "dependent": _json_ref(value[1])}


def _region_json(value: tuple[Ref, Ref, int, int]) -> OutputRegionEnvelope:
    return {"target": _json_ref(value[0]), "source": _json_ref(value[1]), "start": value[2], "end": value[3]}


def _ownership_json(value: tuple[Ref, Ref, int, int]) -> JsonObject:
    return {"target": _json_ref(value[0]), "source": _json_ref(value[1]), "start": value[2], "end": value[3]}


def _groups_json(groups: set[frozenset[Ref]]) -> list[JsonValue]:
    return [
        [_json_ref(member) for member in sorted(group)] for group in sorted(groups, key=lambda group: sorted(group))
    ]


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
