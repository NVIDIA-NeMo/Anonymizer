# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict, bounded measurement snapshots for W&B publication."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from measurement_tools.validation import (
    MeasurementStrategy,
    NonNegativeFloat,
    NonNegativeInt,
    Probability,
    StrictFrozenModel,
)

DEFAULT_MAX_MEASUREMENT_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_MEASUREMENT_RECORDS = 100_000
DEFAULT_MAX_MEASUREMENT_LINE_BYTES = 1024 * 1024
DEFAULT_MAX_JSON_NESTING = 16

Strategy = MeasurementStrategy
RowIndex = StrictInt | None


class _MeasurementEnvelope(StrictFrozenModel):
    schema_version: Literal[1]
    run_id: StrictStr = Field(min_length=1, max_length=256)
    run_tags: dict[StrictStr, JsonValue]
    timestamp_unix_sec: NonNegativeFloat


class RunMeasurement(_MeasurementEnvelope):
    record_type: Literal["run"]
    mode: Literal["replace", "rewrite"]
    strategy: Strategy
    input_row_count: NonNegativeInt
    preview_num_records: NonNegativeInt | None = None
    source_hash: StrictStr
    input_source: dict[StrictStr, JsonValue]
    input_text_column: StrictStr
    input_has_id_column: StrictBool
    input_has_data_summary: StrictBool
    detect: dict[StrictStr, JsonValue]
    replace: dict[StrictStr, JsonValue] | None = None
    rewrite: dict[StrictStr, JsonValue] | None = None
    models: list[dict[StrictStr, JsonValue]]
    runtime: dict[StrictStr, JsonValue]

    @model_validator(mode="after")
    def validate_active_mode_metadata(self) -> RunMeasurement:
        if self.mode == "replace" and self.replace is None:
            raise ValueError("replace metadata is required for replace mode")
        if self.mode == "rewrite" and self.rewrite is None:
            raise ValueError("rewrite metadata is required for rewrite mode")
        return self


class StageMeasurement(_MeasurementEnvelope):
    record_type: Literal["stage"]
    stage: Literal[
        "Anonymizer._run_internal",
        "EntityDetectionWorkflow.run",
        "ReplacementWorkflow.run",
        "RewriteWorkflow.run",
    ]
    status: Literal["completed", "error"]
    elapsed_sec: NonNegativeFloat
    mode: Literal["replace", "rewrite"] | None = None
    strategy: Strategy | None = None
    input_row_count: NonNegativeInt | None = None
    output_row_count: NonNegativeInt | None = None
    failed_record_count: NonNegativeInt | None = None
    preview_num_records: NonNegativeInt | None = None
    input_rows_per_sec: NonNegativeFloat | None = None
    output_rows_per_sec: NonNegativeFloat | None = None
    tag_latent_entities: StrictBool | None = None
    entity_row_count: NonNegativeInt | None = None
    passthrough_row_count: NonNegativeInt | None = None


class _WorkflowMeasurement(_MeasurementEnvelope):
    workflow_name: StrictStr
    status: Literal["completed", "error"]
    model_aliases: list[StrictStr]
    input_row_count: NonNegativeInt
    seed_row_count: NonNegativeInt | None = None
    output_row_count: NonNegativeInt | None = None
    failed_record_count: NonNegativeInt | None = None
    elapsed_sec: NonNegativeFloat
    preview_num_records: NonNegativeInt | None = None
    column_count: NonNegativeInt | None = None
    column_names: list[StrictStr]
    model_usage: dict[StrictStr, JsonValue]
    observed_input_tokens: NonNegativeInt
    observed_output_tokens: NonNegativeInt
    observed_total_tokens: NonNegativeInt
    observed_reasoning_tokens: NonNegativeInt | None = None
    observed_successful_requests: NonNegativeInt
    observed_failed_requests: NonNegativeInt
    observed_total_requests: NonNegativeInt
    observed_failed_request_rate: Probability | None = None
    input_rows_per_sec: NonNegativeFloat | None = None
    output_rows_per_sec: NonNegativeFloat | None = None
    observed_tokens_per_sec: NonNegativeFloat | None = None
    observed_requests_per_sec: NonNegativeFloat | None = None
    observed_tokens_per_successful_request: NonNegativeFloat | None = None


class NddWorkflowMeasurement(_WorkflowMeasurement):
    record_type: Literal["ndd_workflow"]


class ModelWorkflowMeasurement(_WorkflowMeasurement):
    record_type: Literal["model_workflow"]
    local_fields: dict[StrictStr, JsonValue] = Field(default_factory=dict, exclude=True, repr=False)


class _RowMeasurement(_MeasurementEnvelope):
    mode: Literal["replace", "rewrite"]
    strategy: Strategy
    row_index: RowIndex
    record_hash: StrictStr = Field(pattern=r"^[0-9a-fA-F]{64}$")
    text_length_chars: NonNegativeInt
    text_length_chars_bucket: Literal["0", "1-127", "128-511", "512-2047", "2048-8191", "8192+"]
    text_length_tokens: NonNegativeInt
    text_length_tokens_bucket: Literal["0", "1-127", "128-511", "512-2047", "2048-8191", "8192+"]


class RecordMeasurement(_RowMeasurement):
    record_type: Literal["record"]
    final_entity_count: NonNegativeInt
    final_entity_label_counts: dict[StrictStr, NonNegativeInt]
    ground_truth_entity_count: NonNegativeInt | None = None
    ground_truth_entity_label_counts: dict[StrictStr, NonNegativeInt] | None = None
    entity_true_positive_count: NonNegativeInt | None = None
    entity_false_positive_count: NonNegativeInt | None = None
    entity_false_negative_count: NonNegativeInt | None = None
    entity_precision: Probability | None = None
    entity_recall: Probability | None = None
    entity_f1: Probability | None = None
    entity_relaxed_gt_found_count: NonNegativeInt | None = None
    entity_relaxed_detected_tp_count: NonNegativeInt | None = None
    entity_relaxed_precision: Probability | None = None
    entity_relaxed_recall: Probability | None = None
    entity_relaxed_f1: Probability | None = None
    entity_relaxed_label_compatible_gt_found_count: NonNegativeInt | None = None
    entity_relaxed_label_compatible_detected_tp_count: NonNegativeInt | None = None
    entity_relaxed_label_compatible_precision: Probability | None = None
    entity_relaxed_label_compatible_recall: Probability | None = None
    entity_relaxed_label_compatible_f1: Probability | None = None
    replacement_count: NonNegativeInt | None = None
    replacement_label_counts: dict[StrictStr, NonNegativeInt] | None = None
    replacement_duplicate_value_count: NonNegativeInt | None = None
    replacement_missing_final_entity_count: NonNegativeInt | None = None
    replacement_missing_final_entity_label_counts: dict[StrictStr, NonNegativeInt] | None = None
    replacement_missing_final_value_count: NonNegativeInt | None = None
    replacement_synthetic_original_collision_count: NonNegativeInt | None = None
    replacement_synthetic_original_collision_label_counts: dict[StrictStr, NonNegativeInt] | None = None
    replacement_synthetic_original_collision_value_count: NonNegativeInt | None = None
    original_value_leak_count: NonNegativeInt | None = None
    original_value_leak_label_counts: dict[StrictStr, NonNegativeInt] | None = None
    utility_score: Probability | None = None
    leakage_mass: NonNegativeFloat | None = None
    weighted_leakage_rate: Probability | None = None
    repair_iterations: NonNegativeInt | None = None
    any_high_leaked: StrictBool | None = None
    needs_human_review: StrictBool | None = None
    needs_repair: StrictBool | None = None
    detected_candidate_count: NonNegativeInt | None = None
    validation_chunk_count: NonNegativeInt | None = None
    llm_calls_estimated_total: NonNegativeInt | None = None
    llm_calls_estimated_by_stage: dict[StrictStr, NonNegativeInt | None] | None = None


class EvaluationMeasurement(_RowMeasurement):
    record_type: Literal["evaluation_record"]
    detection_valid: StrictBool | None = None
    type_fidelity_valid: StrictBool | None = None
    relational_consistency_valid: StrictBool | None = None
    attribute_fidelity_valid: StrictBool | None = None
    detection_invalid_entity_count: NonNegativeInt = 0
    type_fidelity_invalid_replacement_count: NonNegativeInt = 0
    relational_consistency_invalid_relation_count: NonNegativeInt = 0
    attribute_fidelity_invalid_entity_count: NonNegativeInt = 0


class TraceCoverageMeasurement(_MeasurementEnvelope):
    record_type: Literal["dd_trace_coverage"]
    workflow_name: StrictStr
    trace_backend: StrictStr
    trace_mode: StrictStr
    native_trace_type: StrictStr
    traced_column_count: NonNegativeInt
    traced_column_names: list[StrictStr]
    native_trace_column_count: NonNegativeInt
    native_trace_column_names: list[StrictStr]
    private_trace_column_count: NonNegativeInt
    private_trace_column_names: list[StrictStr]
    private_trace_backend: StrictStr | None = None
    private_trace_note: StrictStr | None = None
    unsupported_column_count: NonNegativeInt
    unsupported_column_names: list[StrictStr]
    unsupported_column_types: list[StrictStr]


MeasurementRecord = Annotated[
    RunMeasurement
    | StageMeasurement
    | RecordMeasurement
    | EvaluationMeasurement
    | NddWorkflowMeasurement
    | ModelWorkflowMeasurement
    | TraceCoverageMeasurement,
    Field(discriminator="record_type"),
]
_RECORD_ADAPTER = TypeAdapter(MeasurementRecord)


class MeasurementSnapshot(StrictFrozenModel):
    path: Path
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: NonNegativeInt
    records: tuple[MeasurementRecord, ...]

    def terminal_stage(self, *, expected_status: Literal["completed", "error"] = "completed") -> StageMeasurement:
        """Return the single terminal stage after enforcing the shared run-status invariant."""
        run_ids = {record.run_id for record in self.records}
        if len(run_ids) != 1:
            raise ValueError("measurement snapshot must contain exactly one run")
        run_id = next(iter(run_ids))
        records = list(self.records)
        _validate_run_identities(records, expected_statuses={run_id: expected_status})
        return next(
            record
            for record in records
            if isinstance(record, StageMeasurement) and record.stage == "Anonymizer._run_internal"
        )


@dataclass(frozen=True)
class CapturedIngressBytes:
    path: Path
    payload: bytes
    sha256: str


def capture_ingress_bytes(path: Path, *, max_bytes: int) -> CapturedIngressBytes:
    """Capture one bounded, immutable regular-file snapshot for W&B ingress."""
    if max_bytes < 0:
        raise ValueError("input byte limit must be non-negative")
    descriptor = _open_file_no_follow(path)
    try:
        before = os.fstat(descriptor)
        _validate_file_metadata(before, max_bytes=max_bytes)
        payload, sha256 = _read_bounded_snapshot(descriptor, max_bytes=max_bytes)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) > max_bytes:
        raise ValueError(f"input exceeds byte limit {max_bytes}")
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity_before != identity_after or len(payload) != after.st_size:
        raise ValueError("input changed while being read")
    return CapturedIngressBytes(path=path, payload=payload, sha256=sha256)


def read_measurement_snapshot(
    path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_MEASUREMENT_BYTES,
    max_records: int = DEFAULT_MAX_MEASUREMENT_RECORDS,
    max_line_bytes: int = DEFAULT_MAX_MEASUREMENT_LINE_BYTES,
    max_nesting: int = DEFAULT_MAX_JSON_NESTING,
    expected_statuses: dict[str, str] | None = None,
) -> MeasurementSnapshot:
    """Capture one immutable regular-file snapshot and parse strict records."""
    if min(max_bytes, max_records, max_line_bytes, max_nesting) < 0:
        raise ValueError("measurement limits must be non-negative")
    captured = capture_ingress_bytes(path, max_bytes=max_bytes)
    records = _parse_records(
        captured.payload,
        max_records=max_records,
        max_line_bytes=max_line_bytes,
        max_nesting=max_nesting,
    )
    _validate_run_identities(records, expected_statuses=expected_statuses)
    return MeasurementSnapshot(
        path=path,
        sha256=captured.sha256,
        byte_count=len(captured.payload),
        records=tuple(records),
    )


def _read_bounded_snapshot(descriptor: int, *, max_bytes: int) -> tuple[bytes, str]:
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    captured = 0
    while captured <= max_bytes:
        chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - captured))
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
        captured += len(chunk)
    return b"".join(chunks), digest.hexdigest()


def _open_file_no_follow(path: Path) -> int:
    """Open a file through pinned directory descriptors without following links."""
    absolute = path.absolute()
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        parent = os.open(absolute.anchor, directory_flags)
        try:
            for part in absolute.parts[1:-1]:
                child = os.open(part, directory_flags | no_follow, dir_fd=parent)
                os.close(parent)
                parent = child
            return os.open(
                absolute.name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0) | no_follow,
                dir_fd=parent,
            )
        finally:
            os.close(parent)
    except OSError as exc:
        raise ValueError("cannot safely open measurement input without following symlinks") from exc


def _validate_file_metadata(metadata: os.stat_result, *, max_bytes: int) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("measurement input must be a regular file")
    if metadata.st_uid != os.geteuid():
        raise ValueError("measurement input must be owned by the current user")
    if metadata.st_nlink != 1:
        raise ValueError("measurement input must not have hard links")
    if metadata.st_size > max_bytes:
        raise ValueError(f"measurement input exceeds byte limit {max_bytes}")


def _parse_records(
    payload: bytes,
    *,
    max_records: int,
    max_line_bytes: int,
    max_nesting: int,
) -> list[MeasurementRecord]:
    lines = payload.splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("measurement input contains an empty record")
    if len(lines) > max_records:
        raise ValueError(f"measurement input exceeds record limit {max_records}")
    records: list[MeasurementRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if len(line) > max_line_bytes:
            raise ValueError(f"measurement record exceeds line byte limit {max_line_bytes}")
        try:
            _validate_json_nesting_bytes(line, max_nesting=max_nesting)
            raw = json.loads(line, parse_constant=_reject_json_constant)
            _validate_json_shape(raw, max_nesting=max_nesting)
            records.append(_RECORD_ADAPTER.validate_python(raw, strict=True))
        except ValidationError as exc:
            locations = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
            location_summary = ", ".join(location for location in locations[:8] if location) or "record_type"
            raise ValueError(
                f"invalid measurement record on line {line_number}: schema violation at {location_summary}"
            ) from None
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid measurement record on line {line_number}: {exc}") from exc
    return records


def _validate_json_nesting_bytes(payload: bytes, *, max_nesting: int) -> None:
    """Reject excessive container nesting before the recursive JSON parser runs."""
    depth = -1
    in_string = False
    escaped = False
    for byte in payload:
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue
        if byte == ord('"'):
            in_string = True
        elif byte in (ord("{"), ord("[")):
            depth += 1
            if depth > max_nesting:
                raise ValueError(f"measurement JSON exceeds nesting limit {max_nesting}")
        elif byte in (ord("}"), ord("]")):
            depth -= 1


def _validate_json_shape(value: JsonValue, *, max_nesting: int, depth: int = 0) -> None:
    if depth > max_nesting:
        raise ValueError(f"measurement JSON exceeds nesting limit {max_nesting}")
    if isinstance(value, dict):
        for nested in value.values():
            _validate_json_shape(nested, max_nesting=max_nesting, depth=depth + 1)
    elif isinstance(value, list):
        for nested in value:
            _validate_json_shape(nested, max_nesting=max_nesting, depth=depth + 1)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _validate_run_identities(records: list[MeasurementRecord], *, expected_statuses: dict[str, str] | None) -> None:
    run_ids = {record.run_id for record in records}
    if expected_statuses is None:
        if len(run_ids) > 1:
            raise ValueError("measurement input contains mixed run identities")
        return
    if run_ids != set(expected_statuses):
        raise ValueError("measurement run identities do not match benchmark cases")
    observed_errors = {
        record.run_id
        for record in records
        if isinstance(record, StageMeasurement | NddWorkflowMeasurement | ModelWorkflowMeasurement)
        and record.status == "error"
    }
    for run_id, status_value in expected_statuses.items():
        if status_value not in {"completed", "error"}:
            raise ValueError("benchmark case has an unsupported terminal status")
        terminal_records = [
            record
            for record in records
            if isinstance(record, StageMeasurement)
            and record.run_id == run_id
            and record.stage == "Anonymizer._run_internal"
        ]
        if len(terminal_records) != 1:
            raise ValueError("measurement run must contain exactly one terminal stage record")
        if terminal_records[0].status != status_value:
            raise ValueError("measurement terminal status does not match benchmark case")
        if (run_id in observed_errors) != (status_value == "error"):
            raise ValueError("measurement status does not match benchmark case")
