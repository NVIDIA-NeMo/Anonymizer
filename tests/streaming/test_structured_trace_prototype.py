# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_TEXT
from tests.streaming.structured_trace_prototype import (
    PROTECTED_TEXT_COLUMN,
    SEGMENT_KEY_COLUMN,
    CodecBounds,
    FailureCode,
    FieldRole,
    ProjectedItem,
    SourceFormat,
    StructuredItemError,
    TraceMapping,
    build_synthetic_anonymizer,
    project_complete_item,
    protect_and_emit,
    reconstruct_complete_item,
    run_projected_segments,
)

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "streaming"
JSON_FIXTURE = FIXTURE_DIR / "complete_trace.json"
JSONL_FIXTURE = FIXTURE_DIR / "complete_trace.jsonl"

SENSITIVE_ENTITIES = {
    "alice@example.test": "email",
    "Alice Example": "full_name",
    "+1-555-0100": "phone_number",
}

TARGET_POINTERS = (
    "/messages/0/content",
    "/messages/1/tool_calls/0/arguments/customer_email",
    "/messages/1/tool_calls/0/arguments/case_note",
    "/messages/2/tool_result/content",
    "/messages/2/tool_result/account_email",
)


@pytest.fixture
def bounds() -> CodecBounds:
    return CodecBounds(max_bytes=8_192, max_depth=12, max_targets=8)


@pytest.fixture
def trace_mapping() -> TraceMapping:
    target = FieldRole.TARGET
    preserve = FieldRole.PRESERVE
    structural = FieldRole.STRUCTURAL
    fields = {
        "/schema_version": structural,
        "/trace_id": structural,
        "/metadata/environment": preserve,
        "/metadata/retention_class": preserve,
        "/metadata/sequence": structural,
        "/messages/0/id": structural,
        "/messages/0/parent_id": structural,
        "/messages/0/role": structural,
        "/messages/0/sequence": structural,
        "/messages/0/content": target,
        "/messages/1/id": structural,
        "/messages/1/parent_id": structural,
        "/messages/1/role": structural,
        "/messages/1/sequence": structural,
        "/messages/1/content": preserve,
        "/messages/1/tool_calls/0/id": structural,
        "/messages/1/tool_calls/0/type": structural,
        "/messages/1/tool_calls/0/name": structural,
        "/messages/1/tool_calls/0/arguments/customer_email": target,
        "/messages/1/tool_calls/0/arguments/case_note": target,
        "/messages/2/id": structural,
        "/messages/2/parent_id": structural,
        "/messages/2/role": structural,
        "/messages/2/sequence": structural,
        "/messages/2/tool_call_id": structural,
        "/messages/2/name": structural,
        "/messages/2/tool_result/status": structural,
        "/messages/2/tool_result/content": target,
        "/messages/2/tool_result/account_email": target,
    }
    return TraceMapping(
        version="synthetic-trace/v1",
        fields=fields,
        source_identity_pointer="/trace_id",
        ordered_identity_pointers=("/messages/0/id", "/messages/1/id", "/messages/2/id"),
    )


def test_complete_json_trace_round_trips_through_redact_after_result_reordering(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    source = JSON_FIXTURE.read_bytes()
    original = json.loads(source)

    projected = project_complete_item(
        source,
        source_format=SourceFormat.JSON,
        mapping=trace_mapping,
        bounds=bounds,
    )

    assert isinstance(projected, ProjectedItem)
    assert tuple(segment.pointer for segment in projected.manifest.segments) == TARGET_POINTERS
    assert tuple(projected.dataframe[SEGMENT_KEY_COLUMN]) == tuple(
        f"synthetic-trace/v1:{pointer}#0" for pointer in TARGET_POINTERS
    )
    assert all(segment.occurrence_index == 0 for segment in projected.manifest.segments)
    assert projected.manifest.source_identity == "trace-2026-0001"
    assert projected.manifest.source_order == ("msg-001", "msg-002", "msg-003")
    assert projected.manifest.fidelity == "semantic-json-v1"
    assert "alice@example.test" not in json.dumps(projected.manifest.template)

    anonymizer = build_synthetic_anonymizer(SENSITIVE_ENTITIES)
    result = run_projected_segments(anonymizer, projected, source_ref=JSON_FIXTURE)
    reordered = result.trace_dataframe.iloc[::-1].reset_index(drop=True)
    protected = reconstruct_complete_item(projected, reordered, failed_records=result.failed_records)
    reconstructed = json.loads(protected)

    assert reconstructed["trace_id"] == original["trace_id"]
    assert reconstructed["schema_version"] == original["schema_version"]
    assert reconstructed["metadata"] == original["metadata"]
    assert [message["id"] for message in reconstructed["messages"]] == [
        message["id"] for message in original["messages"]
    ]
    assert [message["parent_id"] for message in reconstructed["messages"]] == [
        message["parent_id"] for message in original["messages"]
    ]
    assert [message["sequence"] for message in reconstructed["messages"]] == [0, 1, 2]
    assert reconstructed["messages"][1]["content"] == "I will query the synthetic account."
    assert reconstructed["messages"][1]["tool_calls"][0]["id"] == "call-001"
    assert reconstructed["messages"][2]["tool_call_id"] == "call-001"

    for pointer in TARGET_POINTERS:
        protected_value = _resolve_pointer(reconstructed, pointer)
        assert isinstance(protected_value, str)
        assert "[REDACTED_" in protected_value
        assert all(value not in protected_value for value in SENSITIVE_ENTITIES)
    assert set(reordered[SEGMENT_KEY_COLUMN]) == {segment.segment_key for segment in projected.manifest.segments}
    assert len(reordered) == len(projected.manifest.segments)
    assert not reordered[SEGMENT_KEY_COLUMN].duplicated().any()


def test_single_item_jsonl_variant_reconstructs_with_line_boundary(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    source = JSONL_FIXTURE.read_bytes()
    projected = project_complete_item(
        source,
        source_format=SourceFormat.JSONL,
        mapping=trace_mapping,
        bounds=bounds,
    )
    assert isinstance(projected, ProjectedItem)

    anonymizer = build_synthetic_anonymizer(SENSITIVE_ENTITIES)
    result = run_projected_segments(anonymizer, projected, source_ref=JSONL_FIXTURE)
    protected = reconstruct_complete_item(projected, result.trace_dataframe, failed_records=result.failed_records)

    assert protected.endswith(b"\n")
    assert len(protected.splitlines()) == 1
    assert json.loads(protected)["trace_id"] == "trace-2026-0001"


def test_protect_and_emit_emits_complete_item_exactly_once(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    emitted: list[bytes] = []

    protected = protect_and_emit(
        JSON_FIXTURE.read_bytes(),
        source_format=SourceFormat.JSON,
        mapping=trace_mapping,
        bounds=bounds,
        anonymizer=build_synthetic_anonymizer(SENSITIVE_ENTITIES),
        source_ref=JSON_FIXTURE,
        emit=emitted.append,
    )

    assert emitted == [protected]
    assert all(value.encode() not in protected for value in SENSITIVE_ENTITIES)


def test_json_and_jsonl_have_semantically_equivalent_protected_output(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    outputs = []
    for source_format, fixture in ((SourceFormat.JSON, JSON_FIXTURE), (SourceFormat.JSONL, JSONL_FIXTURE)):
        projected = project_complete_item(
            fixture.read_bytes(),
            source_format=source_format,
            mapping=trace_mapping,
            bounds=bounds,
        )
        result = run_projected_segments(build_synthetic_anonymizer(SENSITIVE_ENTITIES), projected, source_ref=fixture)
        outputs.append(
            json.loads(
                reconstruct_complete_item(projected, result.trace_dataframe, failed_records=result.failed_records)
            )
        )

    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("outcome", ["missing", "duplicate", "unknown"])
def test_segment_cardinality_failures_prevent_buffered_emission(
    outcome: str,
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    emitted: list[bytes] = []
    expected_code = {
        "missing": FailureCode.MISSING_SEGMENT,
        "duplicate": FailureCode.DUPLICATE_SEGMENT,
        "unknown": FailureCode.UNKNOWN_SEGMENT,
    }[outcome]
    transform = {
        "missing": _drop_first_result,
        "duplicate": _duplicate_first_result,
        "unknown": _add_unknown_result,
    }[outcome]

    with pytest.raises(StructuredItemError) as exc_info:
        protect_and_emit(
            JSON_FIXTURE.read_bytes(),
            source_format=SourceFormat.JSON,
            mapping=trace_mapping,
            bounds=bounds,
            anonymizer=build_synthetic_anonymizer(SENSITIVE_ENTITIES),
            source_ref=JSON_FIXTURE,
            emit=emitted.append,
            result_transform=transform,
        )

    assert exc_info.value.code is expected_code
    assert emitted == []
    assert all(value not in str(exc_info.value) for value in SENSITIVE_ENTITIES)


def test_dropped_failed_row_prevents_emission_and_sanitizes_failure(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    projected = project_complete_item(
        JSON_FIXTURE.read_bytes(),
        source_format=SourceFormat.JSON,
        mapping=trace_mapping,
        bounds=bounds,
    )
    assert isinstance(projected, ProjectedItem)
    failed_key = projected.manifest.segments[0].segment_key
    emitted: list[bytes] = []

    with pytest.raises(StructuredItemError) as exc_info:
        protect_and_emit(
            JSON_FIXTURE.read_bytes(),
            source_format=SourceFormat.JSON,
            mapping=trace_mapping,
            bounds=bounds,
            anonymizer=build_synthetic_anonymizer(SENSITIVE_ENTITIES, failed_segment_key=failed_key),
            source_ref=JSON_FIXTURE,
            emit=emitted.append,
        )

    assert exc_info.value.code is FailureCode.SEGMENT_PROCESSING_FAILED
    assert emitted == []
    assert "engine-row-private-8675309" not in str(exc_info.value)
    assert all(value not in str(exc_info.value) for value in SENSITIVE_ENTITIES)


def test_raw_target_passthrough_is_not_used_as_fallback_output(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    emitted: list[bytes] = []

    with pytest.raises(StructuredItemError) as exc_info:
        protect_and_emit(
            JSON_FIXTURE.read_bytes(),
            source_format=SourceFormat.JSON,
            mapping=trace_mapping,
            bounds=bounds,
            anonymizer=build_synthetic_anonymizer(SENSITIVE_ENTITIES),
            source_ref=JSON_FIXTURE,
            emit=emitted.append,
            result_transform=_restore_raw_target,
        )

    assert exc_info.value.code is FailureCode.UNPROTECTED_TARGET
    assert emitted == []
    assert all(value not in str(exc_info.value) for value in SENSITIVE_ENTITIES)


def test_changed_but_still_leaky_target_prevents_emission(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    emitted: list[bytes] = []

    with pytest.raises(StructuredItemError) as exc_info:
        protect_and_emit(
            JSON_FIXTURE.read_bytes(),
            source_format=SourceFormat.JSON,
            mapping=trace_mapping,
            bounds=bounds,
            anonymizer=build_synthetic_anonymizer(SENSITIVE_ENTITIES),
            source_ref=JSON_FIXTURE,
            emit=emitted.append,
            result_transform=_restore_changed_leaky_target,
        )

    assert exc_info.value.code is FailureCode.UNPROTECTED_TARGET
    assert emitted == []
    assert all(value not in str(exc_info.value) for value in SENSITIVE_ENTITIES)


def test_unchanged_target_is_allowed_when_detection_found_no_entity(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    parsed = json.loads(JSON_FIXTURE.read_bytes())
    safe_text = "No customer details are present."
    parsed["messages"][0]["content"] = safe_text
    source = json.dumps(parsed).encode()
    emitted: list[bytes] = []

    protected = protect_and_emit(
        source,
        source_format=SourceFormat.JSON,
        mapping=trace_mapping,
        bounds=bounds,
        anonymizer=build_synthetic_anonymizer(SENSITIVE_ENTITIES),
        source_ref=JSON_FIXTURE,
        emit=emitted.append,
    )

    assert emitted == [protected]
    assert json.loads(protected)["messages"][0]["content"] == safe_text


def test_unknown_content_field_is_rejected_until_mapping_classifies_it(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    parsed = json.loads(JSON_FIXTURE.read_bytes())
    parsed["messages"][0]["debug_note"] = "synthetic diagnostic"
    source = json.dumps(parsed).encode()

    with pytest.raises(StructuredItemError) as exc_info:
        project_complete_item(
            source,
            source_format=SourceFormat.JSON,
            mapping=trace_mapping,
            bounds=bounds,
        )
    assert exc_info.value.code is FailureCode.UNKNOWN_FIELD
    assert "synthetic diagnostic" not in str(exc_info.value)

    classified_mapping = replace(
        trace_mapping,
        fields={**trace_mapping.fields, "/messages/0/debug_note": FieldRole.PRESERVE},
    )
    projected = project_complete_item(
        source,
        source_format=SourceFormat.JSON,
        mapping=classified_mapping,
        bounds=bounds,
    )
    assert isinstance(projected, ProjectedItem)
    assert "/messages/0/debug_note" in projected.manifest.preserved_sha256


def test_malformed_source_failure_does_not_expose_raw_input(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    raw_fragment = "alice@example.test"
    malformed = f'{{"content":"{raw_fragment}"'.encode()

    with pytest.raises(StructuredItemError) as exc_info:
        project_complete_item(
            malformed,
            source_format=SourceFormat.JSON,
            mapping=trace_mapping,
            bounds=bounds,
        )

    assert exc_info.value.code is FailureCode.INVALID_SOURCE
    assert raw_fragment not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    ("bounds_override", "expected_code"),
    [
        ({"max_bytes": 32}, FailureCode.ITEM_TOO_LARGE),
        ({"max_depth": 3}, FailureCode.STRUCTURE_TOO_DEEP),
        ({"max_targets": 2}, FailureCode.TOO_MANY_TARGETS),
    ],
)
def test_codec_enforces_explicit_byte_depth_and_target_bounds(
    bounds_override: dict[str, int],
    expected_code: FailureCode,
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    constrained = replace(bounds, **bounds_override)

    with pytest.raises(StructuredItemError) as exc_info:
        project_complete_item(
            JSON_FIXTURE.read_bytes(),
            source_format=SourceFormat.JSON,
            mapping=trace_mapping,
            bounds=constrained,
        )

    assert exc_info.value.code is expected_code
    assert all(value not in str(exc_info.value) for value in SENSITIVE_ENTITIES)


def test_deep_json_parser_failure_is_bounded_and_sanitized(bounds: CodecBounds) -> None:
    source = ('{"id":"source","nested":' + "[" * 1_100 + "0" + "]" * 1_100 + "}").encode()
    nested_pointer = "/nested" + "/0" * 1_100
    mapping = TraceMapping(
        "v1",
        {"/id": FieldRole.STRUCTURAL, nested_pointer: FieldRole.PRESERVE},
        "/id",
        (),
    )

    with pytest.raises(StructuredItemError) as exc_info:
        project_complete_item(
            source,
            source_format=SourceFormat.JSON,
            mapping=mapping,
            bounds=bounds,
        )

    assert exc_info.value.code is FailureCode.STRUCTURE_TOO_DEEP
    assert exc_info.value.__cause__ is None


def test_oversized_numeric_scalar_failure_is_bounded_and_sanitized(bounds: CodecBounds) -> None:
    source = ("{" + '"number":' + "9" * 5_000 + "}").encode()
    mapping = TraceMapping("v1", {"/number": FieldRole.STRUCTURAL}, "/number", ())

    with pytest.raises(StructuredItemError) as exc_info:
        project_complete_item(source, source_format=SourceFormat.JSON, mapping=mapping, bounds=bounds)

    assert exc_info.value.code is FailureCode.ITEM_TOO_LARGE
    assert exc_info.value.__cause__ is None


def test_scalar_count_bound_fails_closed() -> None:
    source = b'{"id":"one","extra":"two"}'
    mapping = TraceMapping(
        "v1",
        {"/id": FieldRole.STRUCTURAL, "/extra": FieldRole.PRESERVE},
        "/id",
        (),
    )
    constrained = CodecBounds(128, 4, 1, 1, 32)

    with pytest.raises(StructuredItemError) as exc_info:
        project_complete_item(source, source_format=SourceFormat.JSON, mapping=mapping, bounds=constrained)

    assert exc_info.value.code is FailureCode.ITEM_TOO_LARGE


def test_parser_event_count_bound_fails_closed() -> None:
    source = b'{"id":"source","items":[0,1]}'
    mapping = TraceMapping(
        "v1",
        {
            "/id": FieldRole.STRUCTURAL,
            "/items/0": FieldRole.PRESERVE,
            "/items/1": FieldRole.PRESERVE,
        },
        "/id",
        (),
    )
    constrained = CodecBounds(128, 4, 1, 4, 32, 4)

    with pytest.raises(StructuredItemError) as exc_info:
        project_complete_item(source, source_format=SourceFormat.JSON, mapping=mapping, bounds=constrained)

    assert exc_info.value.code is FailureCode.ITEM_TOO_LARGE


@pytest.mark.parametrize(
    "source",
    [
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":-Infinity}',
        b'{"value":1e999}',
        b'{"value":1,"value":2}',
    ],
)
def test_strict_json_rejects_nonfinite_numbers_and_duplicate_keys(
    source: bytes,
    bounds: CodecBounds,
) -> None:
    mapping = TraceMapping("v1", {"/value": FieldRole.STRUCTURAL}, "/value", ())

    with pytest.raises(StructuredItemError) as exc_info:
        project_complete_item(source, source_format=SourceFormat.JSON, mapping=mapping, bounds=bounds)

    assert exc_info.value.code is FailureCode.INVALID_SOURCE


@pytest.mark.parametrize(
    "mapping",
    [
        TraceMapping("v1", {"/id": FieldRole.PRESERVE}, "/id", ()),
        TraceMapping(
            "v1",
            {"/id": FieldRole.STRUCTURAL, "/ordered": FieldRole.PRESERVE},
            "/id",
            ("/ordered",),
        ),
    ],
)
def test_identity_pointers_must_be_declared_structural(
    mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    source = b'{"id":"source","ordered":"first"}'

    with pytest.raises(StructuredItemError) as exc_info:
        project_complete_item(source, source_format=SourceFormat.JSON, mapping=mapping, bounds=bounds)

    assert exc_info.value.code is FailureCode.MAPPING_MISMATCH


@pytest.mark.parametrize("pointer", ["id", "/items/~2", "/items/-1", "/items/01", "/items/+1"])
def test_invalid_json_pointer_forms_are_rejected(pointer: str, bounds: CodecBounds) -> None:
    mapping = TraceMapping(
        "v1",
        {"/id": FieldRole.STRUCTURAL, pointer: FieldRole.PRESERVE},
        "/id",
        (),
    )

    with pytest.raises(StructuredItemError) as exc_info:
        project_complete_item(
            b'{"id":"source","items":["zero","one"]}',
            source_format=SourceFormat.JSON,
            mapping=mapping,
            bounds=bounds,
        )

    assert exc_info.value.code is FailureCode.MAPPING_MISMATCH


def test_invalid_bounds_and_source_format_fail_closed(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    with pytest.raises(StructuredItemError) as bounds_error:
        project_complete_item(
            JSON_FIXTURE.read_bytes(),
            source_format=SourceFormat.JSON,
            mapping=trace_mapping,
            bounds=replace(bounds, max_depth=-1),
        )
    assert bounds_error.value.code is FailureCode.MAPPING_MISMATCH

    with pytest.raises(StructuredItemError) as format_error:
        project_complete_item(
            JSON_FIXTURE.read_bytes(),
            source_format=cast(SourceFormat, "yaml"),
            mapping=trace_mapping,
            bounds=bounds,
        )
    assert format_error.value.code is FailureCode.INVALID_SOURCE

    projected = project_complete_item(
        JSON_FIXTURE.read_bytes(),
        source_format=SourceFormat.JSON,
        mapping=trace_mapping,
        bounds=bounds,
    )
    invalid_manifest = replace(projected.manifest, source_format=cast(SourceFormat, "yaml"))
    with pytest.raises(StructuredItemError) as manifest_error:
        reconstruct_complete_item(
            replace(projected, manifest=invalid_manifest),
            pd.DataFrame(),
        )
    assert manifest_error.value.code is FailureCode.MAPPING_MISMATCH


def test_reconstruction_revalidates_source_identity_and_order(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    projected = project_complete_item(
        JSON_FIXTURE.read_bytes(),
        source_format=SourceFormat.JSON,
        mapping=trace_mapping,
        bounds=bounds,
    )
    result = run_projected_segments(build_synthetic_anonymizer(SENSITIVE_ENTITIES), projected, source_ref=JSON_FIXTURE)
    mismatched_manifests = (
        replace(projected.manifest, source_identity="other-source"),
        replace(projected.manifest, source_order=("msg-003", "msg-002", "msg-001")),
    )

    for mismatched_manifest in mismatched_manifests:
        with pytest.raises(StructuredItemError) as exc_info:
            reconstruct_complete_item(
                replace(projected, manifest=mismatched_manifest),
                result.trace_dataframe,
                failed_records=result.failed_records,
            )
        assert exc_info.value.code is FailureCode.MAPPING_MISMATCH


def test_manifest_state_cannot_be_mutated_after_projection(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    projected = project_complete_item(
        JSON_FIXTURE.read_bytes(),
        source_format=SourceFormat.JSON,
        mapping=trace_mapping,
        bounds=bounds,
    )
    template = cast(dict[str, object], projected.manifest.template)
    cast(dict[str, object], template["metadata"])["environment"] = "tampered"
    detached_dataframe = projected.dataframe
    detached_dataframe.loc[detached_dataframe.index[0], COL_TEXT] = "tampered projection"

    result = run_projected_segments(build_synthetic_anonymizer(SENSITIVE_ENTITIES), projected, source_ref=JSON_FIXTURE)
    protected = reconstruct_complete_item(projected, result.trace_dataframe, failed_records=result.failed_records)

    assert json.loads(protected)["metadata"]["environment"] == "test"
    assert "tampered projection" not in projected.dataframe[COL_TEXT].tolist()
    with pytest.raises(TypeError):
        cast(dict[str, str], projected.manifest.preserved_sha256)["/new"] = "digest"


@pytest.mark.parametrize("column", [SEGMENT_KEY_COLUMN, PROTECTED_TEXT_COLUMN])
def test_duplicate_required_dataframe_column_is_sanitized(
    column: str,
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    projected = project_complete_item(
        JSON_FIXTURE.read_bytes(),
        source_format=SourceFormat.JSON,
        mapping=trace_mapping,
        bounds=bounds,
    )
    result = run_projected_segments(build_synthetic_anonymizer(SENSITIVE_ENTITIES), projected, source_ref=JSON_FIXTURE)
    malformed = pd.concat([result.trace_dataframe, result.trace_dataframe[[column]]], axis=1)

    with pytest.raises(StructuredItemError) as exc_info:
        reconstruct_complete_item(projected, malformed, failed_records=result.failed_records)

    assert exc_info.value.code is FailureCode.MAPPING_MISMATCH
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("column", [SEGMENT_KEY_COLUMN, PROTECTED_TEXT_COLUMN])
def test_missing_required_dataframe_column_is_sanitized(
    column: str,
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    projected = project_complete_item(
        JSON_FIXTURE.read_bytes(),
        source_format=SourceFormat.JSON,
        mapping=trace_mapping,
        bounds=bounds,
    )
    result = run_projected_segments(build_synthetic_anonymizer(SENSITIVE_ENTITIES), projected, source_ref=JSON_FIXTURE)

    with pytest.raises(StructuredItemError) as exc_info:
        reconstruct_complete_item(
            projected,
            result.trace_dataframe.drop(columns=column),
            failed_records=result.failed_records,
        )

    assert exc_info.value.code is FailureCode.MAPPING_MISMATCH
    assert exc_info.value.__cause__ is None


def test_result_transform_cannot_erase_grounded_detection_inventory(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    emitted: list[bytes] = []

    with pytest.raises(StructuredItemError) as exc_info:
        protect_and_emit(
            JSON_FIXTURE.read_bytes(),
            source_format=SourceFormat.JSON,
            mapping=trace_mapping,
            bounds=bounds,
            anonymizer=build_synthetic_anonymizer(SENSITIVE_ENTITIES),
            source_ref=JSON_FIXTURE,
            emit=emitted.append,
            result_transform=_erase_inventory_and_restore_leaky_target,
        )

    assert exc_info.value.code is FailureCode.MAPPING_MISMATCH
    assert emitted == []


def test_result_transform_failure_is_sanitized_and_does_not_emit(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    emitted: list[bytes] = []

    with pytest.raises(StructuredItemError) as exc_info:
        protect_and_emit(
            JSON_FIXTURE.read_bytes(),
            source_format=SourceFormat.JSON,
            mapping=trace_mapping,
            bounds=bounds,
            anonymizer=build_synthetic_anonymizer(SENSITIVE_ENTITIES),
            source_ref=JSON_FIXTURE,
            emit=emitted.append,
            result_transform=_raise_private_transform_error,
        )

    assert exc_info.value.code is FailureCode.SEGMENT_PROCESSING_FAILED
    assert exc_info.value.__cause__ is None
    assert emitted == []
    assert "engine-row-private-8675309" not in str(exc_info.value)
    assert all(value not in str(exc_info.value) for value in SENSITIVE_ENTITIES)


def test_generic_caller_columns_remain_outside_target_protection(
    trace_mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    projected = project_complete_item(
        JSON_FIXTURE.read_bytes(),
        source_format=SourceFormat.JSON,
        mapping=trace_mapping,
        bounds=bounds,
    )
    assert isinstance(projected, ProjectedItem)
    caller_dataframe = projected.dataframe
    caller_dataframe["caller_owned_raw_copy"] = caller_dataframe[COL_TEXT]
    projected = replace(projected, _dataframe=caller_dataframe)

    result = run_projected_segments(
        build_synthetic_anonymizer(SENSITIVE_ENTITIES),
        projected,
        source_ref=JSON_FIXTURE,
    )

    assert result.trace_dataframe["caller_owned_raw_copy"].tolist() == projected.dataframe[COL_TEXT].tolist()
    assert "alice@example.test" in " ".join(result.trace_dataframe["caller_owned_raw_copy"])
    assert "alice@example.test" not in " ".join(result.trace_dataframe[PROTECTED_TEXT_COLUMN])


def _drop_first_result(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.iloc[1:].copy()


def _duplicate_first_result(dataframe: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([dataframe, dataframe.iloc[[0]]], ignore_index=True)


def _add_unknown_result(dataframe: pd.DataFrame) -> pd.DataFrame:
    unknown = dataframe.iloc[[0]].copy()
    unknown[SEGMENT_KEY_COLUMN] = "synthetic-trace/v1:/unknown#0"
    return pd.concat([dataframe, unknown], ignore_index=True)


def _restore_raw_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    first_index = dataframe.index[0]
    dataframe.loc[first_index, PROTECTED_TEXT_COLUMN] = dataframe.loc[first_index, "segment_text"]
    return dataframe


def _restore_changed_leaky_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    first_index = dataframe.index[0]
    dataframe.loc[first_index, PROTECTED_TEXT_COLUMN] = "changed alice@example.test"
    return dataframe


def _raise_private_transform_error(dataframe: pd.DataFrame) -> pd.DataFrame:
    raw_text = dataframe.loc[dataframe.index[0], "segment_text"]
    raise RuntimeError(f"engine-row-private-8675309 failed for {raw_text}")


def _erase_inventory_and_restore_leaky_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    first_index = dataframe.index[0]
    dataframe.at[first_index, PROTECTED_TEXT_COLUMN] = "changed alice@example.test"
    dataframe.at[first_index, COL_FINAL_ENTITIES] = {"entities": []}
    return dataframe


def _resolve_pointer(document: object, pointer: str) -> object:
    current = document
    for token in pointer.removeprefix("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = cast(list[object], current)[int(token)]
        else:
            assert isinstance(current, dict)
            current = cast(dict[str, object], current)[token]
    return current
