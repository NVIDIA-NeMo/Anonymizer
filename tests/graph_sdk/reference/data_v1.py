# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent data_v1 reference: facade."""

from __future__ import annotations

import json
from typing import cast

from tests.graph_sdk.reference._data_v1.graph_cases import (
    _base_cases as _base_cases,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _context_cases as _context_cases,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _data as _data,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _dependency_cases as _dependency_cases,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _directed_pairs as _directed_pairs,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _encoding_cases as _encoding_cases,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _group_cases as _group_cases,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _group_json as _group_json,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _identity_cases as _identity_cases,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _ownership_negative_cases as _ownership_negative_cases,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _partitions as _partitions,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _precedence_cases as _precedence_cases,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _range_cases as _range_cases,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _range_specials as _range_specials,
)
from tests.graph_sdk.reference._data_v1.graph_cases import (
    _separation_cases as _separation_cases,
)
from tests.graph_sdk.reference._data_v1.model import (
    FAMILIES as FAMILIES,
)
from tests.graph_sdk.reference._data_v1.model import (
    LIMIT_KEYS as LIMIT_KEYS,
)
from tests.graph_sdk.reference._data_v1.model import (
    RECORD_ALPHABET as RECORD_ALPHABET,
)
from tests.graph_sdk.reference._data_v1.model import (
    RECORD_FACT_KEYS as RECORD_FACT_KEYS,
)
from tests.graph_sdk.reference._data_v1.model import (
    RELATION_FIELDS as RELATION_FIELDS,
)
from tests.graph_sdk.reference._data_v1.model import (
    RELATION_KEYS as RELATION_KEYS,
)
from tests.graph_sdk.reference._data_v1.model import (
    RENAME as RENAME,
)
from tests.graph_sdk.reference._data_v1.model import (
    STATIC_ALPHABET as STATIC_ALPHABET,
)
from tests.graph_sdk.reference._data_v1.model import (
    VALIDATION_CODES as VALIDATION_CODES,
)
from tests.graph_sdk.reference._data_v1.model import (
    AcceptResult as AcceptResult,
)
from tests.graph_sdk.reference._data_v1.model import (
    Activation as Activation,
)
from tests.graph_sdk.reference._data_v1.model import (
    CaseInput as CaseInput,
)
from tests.graph_sdk.reference._data_v1.model import (
    CloseDeclarationEvent as CloseDeclarationEvent,
)
from tests.graph_sdk.reference._data_v1.model import (
    ContextEnvelope as ContextEnvelope,
)
from tests.graph_sdk.reference._data_v1.model import (
    DataDeclaration as DataDeclaration,
)
from tests.graph_sdk.reference._data_v1.model import (
    DatumDeclarationEvent as DatumDeclarationEvent,
)
from tests.graph_sdk.reference._data_v1.model import (
    DatumEnvelope as DatumEnvelope,
)
from tests.graph_sdk.reference._data_v1.model import (
    Declaration as Declaration,
)
from tests.graph_sdk.reference._data_v1.model import (
    DependencyEnvelope as DependencyEnvelope,
)
from tests.graph_sdk.reference._data_v1.model import (
    Family as Family,
)
from tests.graph_sdk.reference._data_v1.model import (
    FixtureCase as FixtureCase,
)
from tests.graph_sdk.reference._data_v1.model import (
    GroupEnvelope as GroupEnvelope,
)
from tests.graph_sdk.reference._data_v1.model import (
    JsonObject as JsonObject,
)
from tests.graph_sdk.reference._data_v1.model import (
    JsonScalar as JsonScalar,
)
from tests.graph_sdk.reference._data_v1.model import (
    JsonValue as JsonValue,
)
from tests.graph_sdk.reference._data_v1.model import (
    OutputRegionEnvelope as OutputRegionEnvelope,
)
from tests.graph_sdk.reference._data_v1.model import (
    RecordBoundary as RecordBoundary,
)
from tests.graph_sdk.reference._data_v1.model import (
    RecordDeclaration as RecordDeclaration,
)
from tests.graph_sdk.reference._data_v1.model import (
    RecordFactDeclarationEvent as RecordFactDeclarationEvent,
)
from tests.graph_sdk.reference._data_v1.model import (
    RecordFactKind as RecordFactKind,
)
from tests.graph_sdk.reference._data_v1.model import (
    Ref as Ref,
)
from tests.graph_sdk.reference._data_v1.model import (
    RejectResult as RejectResult,
)
from tests.graph_sdk.reference._data_v1.model import (
    RelationAdditionEvent as RelationAdditionEvent,
)
from tests.graph_sdk.reference._data_v1.model import (
    RelationEnvelope as RelationEnvelope,
)
from tests.graph_sdk.reference._data_v1.model import (
    RelationKind as RelationKind,
)
from tests.graph_sdk.reference._data_v1.model import (
    SourceRelationEnvelope as SourceRelationEnvelope,
)
from tests.graph_sdk.reference._data_v1.model import (
    TargetSelectionEvent as TargetSelectionEvent,
)
from tests.graph_sdk.reference._data_v1.model import (
    TraceEvent as TraceEvent,
)
from tests.graph_sdk.reference._data_v1.model import (
    ValidateEvent as ValidateEvent,
)
from tests.graph_sdk.reference._data_v1.model import (
    ValidationCode as ValidationCode,
)
from tests.graph_sdk.reference._data_v1.model import (
    ValidationResult as ValidationResult,
)
from tests.graph_sdk.reference._data_v1.model import (
    _as_int as _as_int,
)
from tests.graph_sdk.reference._data_v1.model import (
    _as_list as _as_list,
)
from tests.graph_sdk.reference._data_v1.model import (
    _as_object as _as_object,
)
from tests.graph_sdk.reference._data_v1.model import (
    _ref as _ref,
)
from tests.graph_sdk.reference._data_v1.model import (
    _Reject as _Reject,
)
from tests.graph_sdk.reference._data_v1.record_cases import (
    _activation as _activation,
)
from tests.graph_sdk.reference._data_v1.record_cases import (
    _artifact as _artifact,
)
from tests.graph_sdk.reference._data_v1.record_cases import (
    _canonical as _canonical,
)
from tests.graph_sdk.reference._data_v1.record_cases import (
    _record as _record,
)
from tests.graph_sdk.reference._data_v1.record_cases import (
    _record_cases as _record_cases,
)
from tests.graph_sdk.reference._data_v1.record_cases import (
    _record_precedence_cases as _record_precedence_cases,
)
from tests.graph_sdk.reference._data_v1.record_cases import (
    _status as _status,
)
from tests.graph_sdk.reference._data_v1.record_cases import (
    _terminal as _terminal,
)
from tests.graph_sdk.reference._data_v1.records import (
    _absence_key as _absence_key,
)
from tests.graph_sdk.reference._data_v1.records import (
    _activation_key as _activation_key,
)
from tests.graph_sdk.reference._data_v1.records import (
    _artifact_key as _artifact_key,
)
from tests.graph_sdk.reference._data_v1.records import (
    _boolean as _boolean,
)
from tests.graph_sdk.reference._data_v1.records import (
    _consumed_ref as _consumed_ref,
)
from tests.graph_sdk.reference._data_v1.records import (
    _evidence as _evidence,
)
from tests.graph_sdk.reference._data_v1.records import (
    _membership as _membership,
)
from tests.graph_sdk.reference._data_v1.records import (
    _status_target as _status_target,
)
from tests.graph_sdk.reference._data_v1.records import (
    _string as _string,
)
from tests.graph_sdk.reference._data_v1.records import (
    _terminal_key as _terminal_key,
)
from tests.graph_sdk.reference._data_v1.records import (
    _validate_canonical as _validate_canonical,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _addition_key as _addition_key,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _addition_limit_keys as _addition_limit_keys,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _created_datum as _created_datum,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _data_variants as _data_variants,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _event_refs as _event_refs,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _groups_overlap as _groups_overlap,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _independence_witnesses as _independence_witnesses,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _independent as _independent,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _limit_cases as _limit_cases,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _limits_permit as _limits_permit,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _make_trace as _make_trace,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _ownership_conflicts as _ownership_conflicts,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _ownership_root as _ownership_root,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _raw_variant_count as _raw_variant_count,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _rename_declaration as _rename_declaration,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _replay_trace as _replay_trace,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _replay_trace_parts as _replay_trace_parts,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _root_source as _root_source,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _selected_target as _selected_target,
)
from tests.graph_sdk.reference._data_v1.traces import (
    _source_keys_conflict as _source_keys_conflict,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _case_input as _case_input,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _check_group_overlap as _check_group_overlap,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _check_ownership_overlap as _check_ownership_overlap,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _check_ranges as _check_ranges,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _check_required_targets as _check_required_targets,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _check_slices as _check_slices,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _closed_string as _closed_string,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _context_json as _context_json,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _datum_json as _datum_json,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _datum_map as _datum_map,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _dependency_json as _dependency_json,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _effective_range as _effective_range,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _family as _family,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _groups_json as _groups_json,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _has_cycles as _has_cycles,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _has_duplicate_region_targets as _has_duplicate_region_targets,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _has_duplicate_sources as _has_duplicate_sources,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _json_ref as _json_ref,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _normalize_groups as _normalize_groups,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _offset as _offset,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _ownership_json as _ownership_json,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _parse_context as _parse_context,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _parse_data_event as _parse_data_event,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _parse_datum_envelope as _parse_datum_envelope,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _parse_declaration as _parse_declaration,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _parse_dependency as _parse_dependency,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _parse_group as _parse_group,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _parse_record_event as _parse_record_event,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _parse_ref_envelope as _parse_ref_envelope,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _parse_region as _parse_region,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _parse_relation_envelope as _parse_relation_envelope,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _parse_source as _parse_source,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _parse_validation_result as _parse_validation_result,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _raw_counts as _raw_counts,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _record_boundary as _record_boundary,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _record_fact_kind as _record_fact_kind,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _region_json as _region_json,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _relation_kind as _relation_kind,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _relation_refs as _relation_refs,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _require_fields as _require_fields,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _source_json as _source_json,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _validate_data as _validate_data,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _validate_record as _validate_record,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _validation_code as _validation_code,
)
from tests.graph_sdk.reference._data_v1.validation import (
    validate_case as validate_case,
)


def canonical_bytes(cases: tuple[FixtureCase, ...]) -> bytes:
    """Encode cases in the frozen JSON representation."""
    return (
        json.dumps(cases, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def generate_cases() -> tuple[FixtureCase, ...]:
    """Enumerate the complete finite v1 domain in stable order."""
    cases: list[FixtureCase] = []
    for family, declaration, label in _base_cases():
        typed_family = _family(family)
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
                        "family": typed_family,
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
            _require_fields(case, ("case_id", "family", "declaration", "expected", "trace"))
            case_id = _string(case.get("case_id"))
            family = _family(case.get("family"))
            declaration = _parse_declaration(case.get("declaration"))
            expected = _parse_validation_result(case.get("expected"))
            trace = _parse_trace(case.get("trace"), declaration)
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


def _parse_trace(value: JsonValue | object, declaration: Declaration) -> list[TraceEvent]:
    raw_trace = _as_list(value)
    if len(raw_trace) < 2:
        raise _Reject("invalid_value")
    trace: list[TraceEvent] = []
    for index, raw_event in enumerate(raw_trace):
        event = _as_object(raw_event)
        operation = _string(event.get("op"))
        if index == len(raw_trace) - 2:
            if operation != "close_declaration":
                raise _Reject("invalid_value")
            _require_fields(event, ("op",))
            trace.append(cast(CloseDeclarationEvent, event))
            continue
        if index == len(raw_trace) - 1:
            if operation != "validate":
                raise _Reject("invalid_value")
            _require_fields(event, ("op",))
            trace.append(cast(ValidateEvent, event))
            continue
        if declaration["kind"] == "data":
            trace.append(_parse_data_event(event, operation))
        else:
            trace.append(_parse_record_event(event, operation, declaration["boundary"]))
    replayed = _replay_trace_parts(declaration, trace)
    if replayed != declaration:
        raise _Reject("invalid_value")
    if declaration["kind"] == "record":
        fact_kinds = [event["kind"] for event in trace[:-2] if event["op"] == "declare_record_fact(kind,value)"]
        if len(fact_kinds) != len(set(fact_kinds)):
            raise _Reject("invalid_value")
    return trace
