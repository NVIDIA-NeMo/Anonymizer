#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze detection artifacts for augmentation contribution and label-shape risks.

Usage:
    uv run python tools/measurement/analyze_detection_artifacts.py benchmark/artifacts
    uv run python tools/measurement/analyze_detection_artifacts.py benchmark/artifacts --output detection.jsonl
    uv run python tools/measurement/analyze_detection_artifacts.py benchmark/artifacts --json
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Iterable

import cyclopts
import pandas as pd
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from measurement_tools.tables import ExportFormat
from pydantic import BaseModel, Field

from anonymizer.engine.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_VALIDATION_CANDIDATES,
)
from anonymizer.engine.schemas import (
    AugmentedEntitiesSchema,
    EntitiesSchema,
    EntitySchema,
    ValidationCandidatesSchema,
)

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.detection_artifacts")

API_KEY_PREFIX_RE = re.compile(r"^(sk-|sk_|sk-ant-|sk-proj-|ghp_|pat-|hf_|xox[a-z]-|ya29\.|aiza|akia|bearer\s+)", re.I)


class DetectionArtifactRow(BaseModel):
    workflow_name: str
    batch_file: str
    row_index: int
    seed_entity_count: int
    seed_validation_candidate_count: int
    merged_validation_candidate_count: int
    augmented_entity_count: int
    final_entity_count: int
    augmented_duplicate_seed_value_count: int
    augmented_new_value_count: int
    augmented_new_final_value_count: int
    weak_api_key_shape_count: int
    final_entity_signature_count: int
    final_entity_signature_hashes: list[str] = Field(default_factory=list)
    final_entity_signature_labels: dict[str, str] = Field(default_factory=dict)
    final_entity_signature_details: dict[str, dict[str, object]] = Field(default_factory=dict)
    weak_api_key_shape_label_counts: dict[str, int] = Field(default_factory=dict)
    final_label_counts: dict[str, int] = Field(default_factory=dict)
    final_source_counts: dict[str, int] = Field(default_factory=dict)


class DetectionArtifactAnalysis(BaseModel):
    artifact_path: str
    rows: list[DetectionArtifactRow] = Field(default_factory=list)


def analyze_artifacts(
    artifact_path: Path,
    *,
    parquet_files: Iterable[Path] | None = None,
) -> DetectionArtifactAnalysis:
    if not artifact_path.exists() or not artifact_path.is_dir():
        raise ValueError(f"artifact path is not a directory: {artifact_path}")
    rows: list[DetectionArtifactRow] = []
    for parquet_file in parquet_files if parquet_files is not None else iter_detection_parquet_files(artifact_path):
        rows.extend(_analyze_parquet_file(parquet_file, artifact_root=artifact_path))
    return DetectionArtifactAnalysis(artifact_path=str(artifact_path), rows=rows)


def iter_detection_parquet_files(artifact_path: Path) -> list[Path]:
    files: list[Path] = []
    for workflow_dir in sorted(path for path in artifact_path.iterdir() if path.is_dir()):
        if not workflow_dir.name.startswith("entity-detection"):
            continue
        files.extend(sorted((workflow_dir / "parquet-files").glob("*.parquet")))
    return files


def _analyze_parquet_file(parquet_file: Path, *, artifact_root: Path) -> list[DetectionArtifactRow]:
    dataframe = pd.read_parquet(parquet_file)
    workflow_name = parquet_file.parents[1].name
    batch_file = str(parquet_file.relative_to(artifact_root))
    return [
        _analyze_dataframe_row(row, workflow_name=workflow_name, batch_file=batch_file, row_index=row_index)
        for row_index, row in dataframe.iterrows()
    ]


def _analyze_dataframe_row(
    row: pd.Series,
    *,
    workflow_name: str,
    batch_file: str,
    row_index: int,
) -> DetectionArtifactRow:
    seed_entities = _parse_entities(row.get(COL_SEED_ENTITIES_JSON))
    augmented_entities = _parse_augmented_entities(row.get(COL_AUGMENTED_ENTITIES))
    final_entities = _parse_entities(row.get(COL_DETECTED_ENTITIES))
    return build_detection_artifact_row_from_entities(
        workflow_name=workflow_name,
        batch_file=batch_file,
        row_index=row_index,
        seed_entities=seed_entities,
        seed_validation_candidate_count=_parse_validation_candidate_count(row.get(COL_SEED_VALIDATION_CANDIDATES)),
        merged_validation_candidate_count=_parse_validation_candidate_count(row.get(COL_VALIDATION_CANDIDATES)),
        augmented_entities=augmented_entities,
        final_entities=final_entities,
    )


def build_detection_artifact_row_from_entities(
    *,
    workflow_name: str,
    batch_file: str,
    row_index: int,
    seed_entities: list[EntitySchema],
    seed_validation_candidate_count: int,
    merged_validation_candidate_count: int,
    augmented_entities: list[EntitySchema],
    final_entities: list[EntitySchema],
) -> DetectionArtifactRow:
    seed_values = {_value_key(entity.value) for entity in seed_entities}
    final_values = {_value_key(entity.value) for entity in final_entities}
    augmented_new = [entity for entity in augmented_entities if _value_key(entity.value) not in seed_values]
    weak_counts = _weak_api_key_shape_counts(final_entities)
    final_entity_signatures = _entity_signature_hashes(final_entities, row_index=int(row_index))
    final_entity_signature_labels = _entity_signature_labels(final_entities, row_index=int(row_index))
    final_entity_signature_details = _entity_signature_details(final_entities, row_index=int(row_index))
    return DetectionArtifactRow(
        workflow_name=workflow_name,
        batch_file=batch_file,
        row_index=int(row_index),
        seed_entity_count=len(seed_entities),
        seed_validation_candidate_count=seed_validation_candidate_count,
        merged_validation_candidate_count=merged_validation_candidate_count,
        augmented_entity_count=len(augmented_entities),
        final_entity_count=len(final_entities),
        augmented_duplicate_seed_value_count=len(augmented_entities) - len(augmented_new),
        augmented_new_value_count=len(augmented_new),
        augmented_new_final_value_count=sum(1 for entity in augmented_new if _value_key(entity.value) in final_values),
        weak_api_key_shape_count=sum(weak_counts.values()),
        final_entity_signature_count=len(final_entity_signatures),
        final_entity_signature_hashes=final_entity_signatures,
        final_entity_signature_labels=final_entity_signature_labels,
        final_entity_signature_details=final_entity_signature_details,
        weak_api_key_shape_label_counts=dict(weak_counts),
        final_label_counts=_count_by(final_entities, "label"),
        final_source_counts=_count_by(final_entities, "source"),
    )


def _parse_entities(raw: object) -> list[EntitySchema]:
    values = _extract_payload_list(raw, key="entities")
    parsed = EntitiesSchema.model_validate({"entities": values})
    return [entity for entity in parsed.entities if entity.value and entity.label]


def _parse_augmented_entities(raw: object) -> list[EntitySchema]:
    values = _extract_payload_list(raw, key="entities")
    parsed = AugmentedEntitiesSchema.model_validate({"entities": values})
    return [
        EntitySchema(value=entity.value, label=entity.label, source="augmenter")
        for entity in parsed.entities
        if entity.value and entity.label
    ]


def _parse_validation_candidate_count(raw: object) -> int:
    values = _extract_payload_list(raw, key="candidates")
    parsed = ValidationCandidatesSchema.model_validate({"candidates": values})
    return len(parsed.candidates)


def _extract_payload_list(raw: object, *, key: str) -> list[object]:
    payload = _coerce_payload(raw)
    if isinstance(payload, dict):
        return _coerce_list(payload.get(key))
    return _coerce_list(payload)


def _coerce_payload(raw: object) -> object:
    if _is_missing(raw):
        return {}
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {}


def _coerce_list(value: object) -> list[object]:
    value = _coerce_payload(value)
    if isinstance(value, list):
        return value
    return []


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _value_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _entity_signature_hashes(entities: list[EntitySchema], *, row_index: int) -> list[str]:
    signatures = {_entity_signature_hash(entity, row_index=row_index) for entity in entities}
    return sorted(signatures)


def _entity_signature_labels(entities: list[EntitySchema], *, row_index: int) -> dict[str, str]:
    labels = {_entity_signature_hash(entity, row_index=row_index): entity.label for entity in entities}
    return dict(sorted(labels.items()))


def _entity_signature_details(entities: list[EntitySchema], *, row_index: int) -> dict[str, dict[str, object]]:
    details = {
        _entity_signature_hash(entity, row_index=row_index): {
            "label": entity.label,
            "source": entity.source,
            "row_index": int(row_index),
            "start_position": entity.start_position,
            "end_position": entity.end_position,
            "value_length": len(entity.value),
        }
        for entity in entities
    }
    return dict(sorted(details.items()))


def _entity_signature_hash(entity: EntitySchema, *, row_index: int) -> str:
    payload = json.dumps(
        {
            "row": row_index,
            "label": entity.label,
            "start": entity.start_position,
            "end": entity.end_position,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _weak_api_key_shape_counts(entities: list[EntitySchema]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entity in entities:
        if entity.label == "api_key" and not _looks_like_api_key(entity.value):
            counts[entity.label] += 1
    return counts


def _looks_like_api_key(value: str) -> bool:
    stripped = value.strip()
    if API_KEY_PREFIX_RE.search(stripped):
        return True
    compact = re.sub(r"[\s'\";:,/]+", "", stripped)
    if len(compact) < 20:
        return False
    return bool(re.search(r"[A-Za-z]", compact)) and bool(re.search(r"\d", compact))


def _count_by(entities: list[EntitySchema], field: str) -> dict[str, int]:
    counts = Counter(str(getattr(entity, field)) for entity in entities if getattr(entity, field))
    return dict(sorted(counts.items()))


def write_rows(rows: list[DetectionArtifactRow], output_path: Path, export_format: ExportFormat) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.json_normalize([row.model_dump() for row in rows], sep=".")
    if export_format == ExportFormat.parquet:
        table.to_parquet(output_path, index=False)
    elif export_format == ExportFormat.csv:
        table.to_csv(output_path, index=False)
    else:
        table.to_json(output_path, orient="records", lines=True)


def render_result(result: DetectionArtifactAnalysis, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    total_warnings = sum(row.weak_api_key_shape_count for row in result.rows)
    workflows = Counter(row.workflow_name for row in result.rows)
    lines = [f"Analyzed {len(result.rows)} detection artifact row(s) from {result.artifact_path}"]
    for workflow_name, count in sorted(workflows.items()):
        lines.append(f"- {workflow_name}: {count} row(s)")
    lines.append(f"Weak api_key shape warnings: {total_warnings}")
    return "\n".join(lines)


@app.default
def main(
    artifact_path: Path,
    *,
    output: Annotated[Path | None, cyclopts.Parameter(("--output", "-o"))] = None,
    format: Annotated[ExportFormat, cyclopts.Parameter("--format")] = ExportFormat.jsonl,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        result = analyze_artifacts(artifact_path)
        if output is not None:
            write_rows(result.rows, output, format)
    except ValueError as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")


if __name__ == "__main__":
    app()
