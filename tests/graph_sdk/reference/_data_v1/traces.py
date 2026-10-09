# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent data_v1 reference: traces."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._data_v1.graph_cases import (
    _base_cases,
    _data,
)
from tests.graph_sdk.reference._data_v1.model import (
    RELATION_KEYS,
    RENAME,
    CloseDeclarationEvent,
    DataDeclaration,
    DatumDeclarationEvent,
    Declaration,
    FixtureCase,
    JsonObject,
    JsonValue,
    RecordFactDeclarationEvent,
    Ref,
    RelationAdditionEvent,
    TargetSelectionEvent,
    TraceEvent,
    ValidateEvent,
    _as_int,
    _as_list,
    _as_object,
    _ref,
    _Reject,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _case_input,
    _parse_context,
    _parse_datum_envelope,
    _parse_dependency,
    _parse_group,
    _parse_region,
    _parse_relation_envelope,
    _parse_source,
    _raw_counts,
    _record_fact_kind,
    validate_case,
)


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
    trace: list[TraceEvent] = []
    if raw_declaration.get("kind") == "record":
        facts = _as_object(raw_declaration["facts"])
        for key, value in facts.items():
            trace.append(
                RecordFactDeclarationEvent(
                    op="declare_record_fact(kind,value)",
                    kind=_record_fact_kind(key),
                    value=deepcopy(value),
                )
            )
    else:
        for datum in _as_list(raw_declaration["datums"]):
            trace.append(
                DatumDeclarationEvent(op="declare_datum(id,text)", value=_parse_datum_envelope(deepcopy(datum)))
            )
        for target in _as_list(raw_declaration["targets"]):
            trace.append(TargetSelectionEvent(op="select_target(id)", value=deepcopy(target)))
        for key in RELATION_KEYS:
            for relation in _as_list(raw_declaration[key]):
                trace.append(
                    RelationAdditionEvent(
                        op="add_relation(kind,value)",
                        kind=key,
                        value=_parse_relation_envelope(key, deepcopy(relation)),
                    )
                )
    trace.append(CloseDeclarationEvent(op="close_declaration"))
    trace.append(ValidateEvent(op="validate"))
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
    return _replay_trace_parts(case["declaration"], trace)


def _replay_trace_parts(declaration: Declaration, trace: Sequence[TraceEvent]) -> JsonObject:
    """Rebuild a declaration from an already grammar-checked trace."""
    original = _as_object(declaration)
    if declaration["kind"] == "record":
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
