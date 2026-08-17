# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only adopter-owned structured trace prototype."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import cast
from unittest.mock import Mock

import pandas as pd
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.resolved_input import ResolvedInput
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.results import AnonymizerResult

SEGMENT_KEY_COLUMN = "caller_segment_key"
PUBLIC_TEXT_COLUMN = "segment_text"
PROTECTED_TEXT_COLUMN = f"{PUBLIC_TEXT_COLUMN}_replaced"


class SourceFormat(str, Enum):
    JSON = "json"
    JSONL = "jsonl"


class FieldRole(str, Enum):
    TARGET = "target"
    PRESERVE = "preserve"
    STRUCTURAL = "structural"


class FailureCode(str, Enum):
    INVALID_SOURCE = "invalid_source"
    ITEM_TOO_LARGE = "item_too_large"
    STRUCTURE_TOO_DEEP = "structure_too_deep"
    TOO_MANY_TARGETS = "too_many_targets"
    UNKNOWN_FIELD = "unknown_field"
    MAPPING_MISMATCH = "mapping_mismatch"
    SEGMENT_PROCESSING_FAILED = "segment_processing_failed"
    MISSING_SEGMENT = "missing_segment"
    DUPLICATE_SEGMENT = "duplicate_segment"
    UNKNOWN_SEGMENT = "unknown_segment"
    UNPROTECTED_TARGET = "unprotected_target"


class StructuredItemError(RuntimeError):
    """Sanitized test-visible complete-item failure."""

    def __init__(self, code: FailureCode) -> None:
        self.code = code
        super().__init__(f"structured item rejected ({code.value})")


@dataclass(frozen=True)
class CodecBounds:
    max_bytes: int
    max_depth: int
    max_targets: int
    max_scalars: int = 64
    max_scalar_bytes: int = 1_024
    max_events: int = 256


@dataclass(frozen=True)
class TraceMapping:
    version: str
    fields: Mapping[str, FieldRole]
    source_identity_pointer: str
    ordered_identity_pointers: tuple[str, ...]


@dataclass(frozen=True)
class SegmentManifest:
    segment_key: str
    pointer: str
    occurrence_index: int
    source_order: int
    input_sha256: str


@dataclass(frozen=True)
class ReconstructionManifest:
    mapping_version: str
    source_format: SourceFormat
    fidelity: str
    source_identity: str
    source_order: tuple[str, ...]
    source_identity_pointer: str
    ordered_identity_pointers: tuple[str, ...]
    segments: tuple[SegmentManifest, ...]
    preserved_sha256: Mapping[str, str]
    structural_sha256: Mapping[str, str]
    _template_json: bytes

    @property
    def template(self) -> object:
        """Return a detached view of the private reconstruction template."""
        return json.loads(self._template_json)


@dataclass(frozen=True)
class ProjectedItem:
    _dataframe: pd.DataFrame
    manifest: ReconstructionManifest

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return a detached view of the private segment projection."""
        return self._dataframe.copy(deep=True)


ResultTransform = Callable[[pd.DataFrame], pd.DataFrame]
Emitter = Callable[[bytes], None]


def project_complete_item(
    source: bytes,
    *,
    source_format: SourceFormat,
    mapping: TraceMapping,
    bounds: CodecBounds,
) -> ProjectedItem:
    """Parse and project one complete source item under a closed field policy."""
    document, scalars = _validate_source_and_mapping(
        source,
        source_format=source_format,
        mapping=mapping,
        bounds=bounds,
    )
    return _build_projection(document, scalars=scalars, source_format=source_format, mapping=mapping)


def _validate_source_and_mapping(
    source: bytes,
    *,
    source_format: SourceFormat,
    mapping: TraceMapping,
    bounds: CodecBounds,
) -> tuple[object, dict[str, object]]:
    _validate_configuration(source_format=source_format, mapping=mapping, bounds=bounds)
    if len(source) > bounds.max_bytes:
        raise StructuredItemError(FailureCode.ITEM_TOO_LARGE)
    document = _parse_source(source, source_format=source_format, bounds=bounds)
    scalars = _collect_bounded_scalars(document, bounds=bounds)

    for pointer in mapping.fields:
        _resolve_pointer(document, pointer)

    declared_pointers = set(mapping.fields)
    actual_pointers = set(scalars)
    if actual_pointers - declared_pointers:
        raise StructuredItemError(FailureCode.UNKNOWN_FIELD)
    if declared_pointers - actual_pointers:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)

    target_pointers = [pointer for pointer in scalars if mapping.fields[pointer] is FieldRole.TARGET]
    if len(target_pointers) > bounds.max_targets:
        raise StructuredItemError(FailureCode.TOO_MANY_TARGETS)
    return document, scalars


def _validate_configuration(
    *,
    source_format: SourceFormat,
    mapping: TraceMapping,
    bounds: CodecBounds,
) -> None:
    if not isinstance(source_format, SourceFormat):
        raise StructuredItemError(FailureCode.INVALID_SOURCE)
    if (
        not isinstance(bounds, CodecBounds)
        or bounds.max_bytes <= 0
        or bounds.max_depth <= 0
        or bounds.max_targets < 0
        or bounds.max_scalars <= 0
        or bounds.max_scalar_bytes <= 0
        or bounds.max_events <= 0
    ):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    if not isinstance(mapping, TraceMapping) or not mapping.version or not isinstance(mapping.fields, Mapping):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)

    identity_pointers = (mapping.source_identity_pointer, *mapping.ordered_identity_pointers)
    if len(identity_pointers) != len(set(identity_pointers)):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    for pointer, role in mapping.fields.items():
        if not isinstance(pointer, str) or not isinstance(role, FieldRole):
            raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
        _pointer_tokens(pointer)
    if any(mapping.fields.get(pointer) is not FieldRole.STRUCTURAL for pointer in identity_pointers):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)


def _build_projection(
    document: object,
    *,
    scalars: Mapping[str, object],
    source_format: SourceFormat,
    mapping: TraceMapping,
) -> ProjectedItem:
    source_identity = _require_string(_resolve_pointer(document, mapping.source_identity_pointer))
    source_order = tuple(
        _require_string(_resolve_pointer(document, pointer)) for pointer in mapping.ordered_identity_pointers
    )

    template = deepcopy(document)
    rows: list[dict[str, str]] = []
    segments: list[SegmentManifest] = []
    preserved_sha256: dict[str, str] = {}
    structural_sha256: dict[str, str] = {}
    for pointer, value in scalars.items():
        role = mapping.fields[pointer]
        if role is FieldRole.TARGET:
            text = _require_string(value)
            occurrence_index = 0
            segment_key = f"{mapping.version}:{pointer}#{occurrence_index}"
            segments.append(
                SegmentManifest(
                    segment_key=segment_key,
                    pointer=pointer,
                    occurrence_index=occurrence_index,
                    source_order=len(segments),
                    input_sha256=_sha256_value(text),
                )
            )
            rows.append({COL_TEXT: text, SEGMENT_KEY_COLUMN: segment_key})
            _set_pointer(template, pointer, {"__protected_segment_key__": segment_key})
        elif role is FieldRole.PRESERVE:
            preserved_sha256[pointer] = _sha256_value(value)
        else:
            structural_sha256[pointer] = _sha256_value(value)

    fidelity = "semantic-json-v1" if source_format is SourceFormat.JSON else "semantic-jsonl-item-v1"
    manifest = ReconstructionManifest(
        mapping_version=mapping.version,
        source_format=source_format,
        fidelity=fidelity,
        source_identity=source_identity,
        source_order=source_order,
        source_identity_pointer=mapping.source_identity_pointer,
        ordered_identity_pointers=mapping.ordered_identity_pointers,
        segments=tuple(segments),
        preserved_sha256=MappingProxyType(preserved_sha256),
        structural_sha256=MappingProxyType(structural_sha256),
        _template_json=json.dumps(template, ensure_ascii=False, separators=(",", ":")).encode(),
    )
    dataframe = pd.DataFrame(
        {
            COL_TEXT: [row[COL_TEXT] for row in rows],
            SEGMENT_KEY_COLUMN: [row[SEGMENT_KEY_COLUMN] for row in rows],
        }
    )
    return ProjectedItem(_dataframe=dataframe.copy(deep=True), manifest=manifest)


class _SyntheticDetectionWorkflow:
    """Deterministic detector used only to exercise the facade and local Redact."""

    def __init__(
        self,
        sensitive_entities: Mapping[str, str],
        *,
        failed_segment_key: str | None,
    ) -> None:
        self._sensitive_entities = sensitive_entities
        self._failed_segment_key = failed_segment_key

    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
        output = dataframe.copy()
        failed_records: list[FailedRecord] = []
        if self._failed_segment_key is not None:
            failed = output[SEGMENT_KEY_COLUMN] == self._failed_segment_key
            if failed.any():
                raw_text = str(output.loc[failed, COL_TEXT].iloc[0])
                failed_records.append(
                    FailedRecord(
                        record_id="engine-row-private-8675309",
                        step="synthetic-detection",
                        reason=f"synthetic detector failed while processing {raw_text}",
                    )
                )
                output = output.loc[~failed].copy()

        entity_rows = [self._find_entities(str(text)) for text in output[COL_TEXT]]
        output[COL_DETECTED_ENTITIES] = entity_rows
        output[COL_FINAL_ENTITIES] = entity_rows
        return EntityDetectionResult(dataframe=output, failed_records=failed_records)

    def _find_entities(self, text: str) -> dict[str, list[dict[str, str | int]]]:
        entities: list[dict[str, str | int]] = []
        for value, label in self._sensitive_entities.items():
            start = 0
            while True:
                found = text.find(value, start)
                if found < 0:
                    break
                entities.append(
                    {
                        "value": value,
                        "label": label,
                        "start_position": found,
                        "end_position": found + len(value),
                    }
                )
                start = found + len(value)
        entities.sort(key=lambda entity: cast(int, entity["start_position"]))
        return {"entities": entities}


def build_synthetic_anonymizer(
    sensitive_entities: Mapping[str, str],
    *,
    failed_segment_key: str | None = None,
) -> Anonymizer:
    detector = _SyntheticDetectionWorkflow(sensitive_entities, failed_segment_key=failed_segment_key)
    data_designer = cast(DataDesigner, Mock(spec=DataDesigner))
    return Anonymizer(
        data_designer=data_designer,
        detection_workflow=cast(EntityDetectionWorkflow, detector),
        replace_runner=ReplacementWorkflow(),
    )


def run_projected_segments(
    anonymizer: Anonymizer,
    projected: ProjectedItem,
    *,
    source_ref: Path,
) -> AnonymizerResult:
    data = AnonymizerInput(
        source=str(source_ref),
        text_column=PUBLIC_TEXT_COLUMN,
        data_summary="Synthetic structured trace target fields.",
    )
    config = AnonymizerConfig(replace=Redact(), emit_telemetry=False)
    context = ResolvedInput(
        dataframe=projected.dataframe,
        requested_text_column=PUBLIC_TEXT_COLUMN,
        resolved_text_column=PUBLIC_TEXT_COLUMN,
    )
    anonymizer.validate_config(config)
    return anonymizer._run_internal(
        config=config,
        data=data,
        context=context,
        preview_num_records=None,
    )


def reconstruct_complete_item(
    projected: ProjectedItem,
    result_dataframe: pd.DataFrame,
    *,
    failed_records: list[FailedRecord] | None = None,
    detected_values_by_key: Mapping[str, tuple[str, ...]] | None = None,
) -> bytes:
    """Patch exactly one protected result per declared segment into the private template."""
    try:
        if not isinstance(projected.manifest.source_format, SourceFormat):
            raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
        protected_by_key = _collect_protected_results(
            projected,
            result_dataframe,
            failed_records=failed_records,
            detected_values_by_key=detected_values_by_key,
        )
        reconstructed = projected.manifest.template
        for segment in projected.manifest.segments:
            _set_pointer(reconstructed, segment.pointer, protected_by_key[segment.segment_key])
        _verify_unchanged_fields(reconstructed, projected.manifest.preserved_sha256)
        _verify_unchanged_fields(reconstructed, projected.manifest.structural_sha256)
        _verify_identity(reconstructed, projected.manifest)

        protected = json.dumps(reconstructed, ensure_ascii=False, separators=(",", ":")).encode()
        if projected.manifest.source_format is SourceFormat.JSONL:
            protected += b"\n"
        return protected
    except StructuredItemError:
        raise
    except Exception:
        raise StructuredItemError(FailureCode.SEGMENT_PROCESSING_FAILED) from None


def _collect_protected_results(
    projected: ProjectedItem,
    result_dataframe: pd.DataFrame,
    *,
    failed_records: list[FailedRecord] | None,
    detected_values_by_key: Mapping[str, tuple[str, ...]] | None,
) -> dict[str, str]:
    if failed_records:
        raise StructuredItemError(FailureCode.SEGMENT_PROCESSING_FAILED)
    if not isinstance(result_dataframe, pd.DataFrame):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    required_columns = (SEGMENT_KEY_COLUMN, PROTECTED_TEXT_COLUMN, COL_FINAL_ENTITIES)
    labels = result_dataframe.columns.tolist()
    if any(labels.count(column) != 1 for column in required_columns):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)

    keys = result_dataframe[SEGMENT_KEY_COLUMN]
    if keys.duplicated().any():
        raise StructuredItemError(FailureCode.DUPLICATE_SEGMENT)
    if not keys.map(lambda value: isinstance(value, str)).all():
        raise StructuredItemError(FailureCode.UNKNOWN_SEGMENT)

    expected = {segment.segment_key: segment for segment in projected.manifest.segments}
    actual_keys = set(cast(list[str], keys.tolist()))
    if actual_keys - set(expected):
        raise StructuredItemError(FailureCode.UNKNOWN_SEGMENT)
    if set(expected) - actual_keys:
        raise StructuredItemError(FailureCode.MISSING_SEGMENT)

    protected_by_key: dict[str, str] = {}
    for _, row in result_dataframe.iterrows():
        segment_key = cast(str, row[SEGMENT_KEY_COLUMN])
        protected_text = row[PROTECTED_TEXT_COLUMN]
        if not isinstance(protected_text, str):
            raise StructuredItemError(FailureCode.SEGMENT_PROCESSING_FAILED)
        row_detected_values = _detected_entity_values(row[COL_FINAL_ENTITIES])
        detected_values = (
            row_detected_values if detected_values_by_key is None else detected_values_by_key.get(segment_key)
        )
        if detected_values is None or (detected_values_by_key is not None and row_detected_values != detected_values):
            raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
        if detected_values and any(value in protected_text for value in detected_values):
            raise StructuredItemError(FailureCode.UNPROTECTED_TARGET)
        protected_by_key[segment_key] = protected_text
    return protected_by_key


def protect_and_emit(
    source: bytes,
    *,
    source_format: SourceFormat,
    mapping: TraceMapping,
    bounds: CodecBounds,
    anonymizer: Anonymizer,
    source_ref: Path,
    emit: Emitter,
    result_transform: ResultTransform | None = None,
) -> bytes:
    projected = project_complete_item(
        source,
        source_format=source_format,
        mapping=mapping,
        bounds=bounds,
    )
    try:
        result = run_projected_segments(anonymizer, projected, source_ref=source_ref)
        detected_values_by_key = _snapshot_detected_values(result.trace_dataframe)
        result_dataframe = result.trace_dataframe
        if result_transform is not None:
            result_dataframe = result_transform(result_dataframe.copy())
        protected = reconstruct_complete_item(
            projected,
            result_dataframe,
            failed_records=result.failed_records,
            detected_values_by_key=detected_values_by_key,
        )
    except StructuredItemError:
        raise
    except Exception:
        raise StructuredItemError(FailureCode.SEGMENT_PROCESSING_FAILED) from None
    emit(protected)
    return protected


def _parse_source(source: bytes, *, source_format: SourceFormat, bounds: CodecBounds) -> object:
    try:
        if source_format is SourceFormat.JSON:
            encoded_item = source
        elif source_format is SourceFormat.JSONL:
            lines = source.splitlines()
            if len(lines) != 1 or not lines[0].strip():
                raise StructuredItemError(FailureCode.INVALID_SOURCE)
            encoded_item = lines[0]
        else:
            raise StructuredItemError(FailureCode.INVALID_SOURCE)
        return json.loads(
            encoded_item.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_number,
            parse_int=lambda value: _parse_bounded_int(value, bounds.max_scalar_bytes),
            parse_float=lambda value: _parse_bounded_float(value, bounds.max_scalar_bytes),
        )
    except StructuredItemError:
        raise
    except RecursionError:
        raise StructuredItemError(FailureCode.STRUCTURE_TOO_DEEP) from None
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise StructuredItemError(FailureCode.INVALID_SOURCE) from None


def _collect_bounded_scalars(value: object, *, bounds: CodecBounds) -> dict[str, object]:
    scalars: dict[str, object] = {}
    pending: list[tuple[object, str, int]] = [(value, "", 1)]
    events = 0
    while pending:
        events += 1
        if events > bounds.max_events:
            raise StructuredItemError(FailureCode.ITEM_TOO_LARGE)
        item, pointer, depth = pending.pop()
        if depth > bounds.max_depth:
            raise StructuredItemError(FailureCode.STRUCTURE_TOO_DEEP)
        if isinstance(item, dict):
            for key, child in reversed(cast(dict[str, object], item).items()):
                if len(key.encode("utf-8")) > bounds.max_scalar_bytes:
                    raise StructuredItemError(FailureCode.ITEM_TOO_LARGE)
                pending.append((child, f"{pointer}/{_escape_pointer_token(key)}", depth + 1))
        elif isinstance(item, list):
            for index in range(len(item) - 1, -1, -1):
                pending.append((item[index], f"{pointer}/{index}", depth + 1))
        else:
            if len(json.dumps(item, ensure_ascii=False).encode()) > bounds.max_scalar_bytes:
                raise StructuredItemError(FailureCode.ITEM_TOO_LARGE)
            scalars[pointer] = item
            if len(scalars) > bounds.max_scalars:
                raise StructuredItemError(FailureCode.ITEM_TOO_LARGE)
    return scalars


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _pointer_tokens(pointer: str) -> tuple[str, ...]:
    if not pointer.startswith("/"):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    decoded: list[str] = []
    for token in pointer[1:].split("/"):
        index = 0
        while index < len(token):
            if token[index] == "~":
                if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
                    raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
                index += 2
            else:
                index += 1
        decoded.append(token.replace("~1", "/").replace("~0", "~"))
    return tuple(decoded)


def _resolve_pointer(document: object, pointer: str) -> object:
    current = document
    try:
        for token in _pointer_tokens(pointer):
            if isinstance(current, list):
                current = cast(list[object], current)[_array_index(token, len(current))]
            elif isinstance(current, dict):
                current = cast(dict[str, object], current)[token]
            else:
                raise KeyError(token)
    except (KeyError, IndexError, ValueError):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH) from None
    return current


def _set_pointer(document: object, pointer: str, value: object) -> None:
    tokens = _pointer_tokens(pointer)
    if not tokens:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    parent = document
    try:
        for token in tokens[:-1]:
            if isinstance(parent, list):
                parent = cast(list[object], parent)[_array_index(token, len(parent))]
            elif isinstance(parent, dict):
                parent = cast(dict[str, object], parent)[token]
            else:
                raise KeyError(token)
        final = tokens[-1]
        if isinstance(parent, list):
            cast(list[object], parent)[_array_index(final, len(parent))] = value
        elif isinstance(parent, dict):
            cast(dict[str, object], parent)[final] = value
        else:
            raise KeyError(final)
    except (KeyError, IndexError, ValueError):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH) from None


def _require_string(value: object) -> str:
    if not isinstance(value, str):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    return value


def _array_index(token: str, length: int) -> int:
    if token != "0" and (not token.isascii() or not token.isdigit() or token.startswith("0")):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    index = int(token)
    if index >= length:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    return index


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonfinite_number(_: str) -> float:
    raise ValueError("non-finite JSON number")


def _parse_bounded_int(value: str, max_scalar_bytes: int) -> int:
    if len(value.encode()) > max_scalar_bytes:
        raise StructuredItemError(FailureCode.ITEM_TOO_LARGE)
    return int(value)


def _parse_bounded_float(value: str, max_scalar_bytes: int) -> float:
    if len(value.encode()) > max_scalar_bytes:
        raise StructuredItemError(FailureCode.ITEM_TOO_LARGE)
    parsed = float(value)
    if not math.isfinite(parsed):
        raise StructuredItemError(FailureCode.INVALID_SOURCE)
    return parsed


def _detected_entity_values(inventory: object) -> tuple[str, ...]:
    if not isinstance(inventory, dict) or set(inventory) != {"entities"}:
        raise StructuredItemError(FailureCode.SEGMENT_PROCESSING_FAILED)
    entities = cast(dict[str, object], inventory)["entities"]
    if not isinstance(entities, list):
        raise StructuredItemError(FailureCode.SEGMENT_PROCESSING_FAILED)
    values: list[str] = []
    for entity in cast(list[object], entities):
        if not isinstance(entity, dict):
            raise StructuredItemError(FailureCode.SEGMENT_PROCESSING_FAILED)
        entity_value = cast(dict[str, object], entity).get("value")
        if not isinstance(entity_value, str) or not entity_value:
            raise StructuredItemError(FailureCode.SEGMENT_PROCESSING_FAILED)
        values.append(entity_value)
    return tuple(values)


def _snapshot_detected_values(result_dataframe: pd.DataFrame) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(result_dataframe, pd.DataFrame):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    labels = result_dataframe.columns.tolist()
    if labels.count(SEGMENT_KEY_COLUMN) != 1 or labels.count(COL_FINAL_ENTITIES) != 1:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    snapshot: dict[str, tuple[str, ...]] = {}
    for _, row in result_dataframe.iterrows():
        segment_key = row[SEGMENT_KEY_COLUMN]
        if not isinstance(segment_key, str) or segment_key in snapshot:
            raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
        snapshot[segment_key] = _detected_entity_values(row[COL_FINAL_ENTITIES])
    return MappingProxyType(snapshot)


def _sha256_value(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _verify_unchanged_fields(document: object, expected_digests: Mapping[str, str]) -> None:
    for pointer, expected_digest in expected_digests.items():
        if _sha256_value(_resolve_pointer(document, pointer)) != expected_digest:
            raise StructuredItemError(FailureCode.MAPPING_MISMATCH)


def _verify_identity(document: object, manifest: ReconstructionManifest) -> None:
    if _require_string(_resolve_pointer(document, manifest.source_identity_pointer)) != manifest.source_identity:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    source_order = tuple(
        _require_string(_resolve_pointer(document, pointer)) for pointer in manifest.ordered_identity_pointers
    )
    if source_order != manifest.source_order:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
