#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract masked context for entity signature differences between benchmark artifacts.

Usage:
    uv run python tools/measurement/extract_signature_deltas.py \
      baseline/detection-artifacts.jsonl candidate/detection-artifacts.jsonl \
      --baseline-artifact-root baseline/artifacts --candidate-artifact-root candidate/artifacts \
      --baseline-config legal-default --candidate-config legal-rules-guardrail \
      --workload legal-r2 --output deltas.csv --format csv
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import cyclopts
import pandas as pd
from analyze_detection_artifacts import _entity_signature_hash
from pydantic import BaseModel, Field, ValidationError

from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_TEXT
from anonymizer.engine.detection.rules import detect_high_confidence_entities
from anonymizer.engine.schemas import EntitiesSchema, EntitySchema

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.signature_deltas")


class ExportFormat(StrEnum):
    parquet = "parquet"
    csv = "csv"
    jsonl = "jsonl"


class LogFormat(StrEnum):
    plain = "plain"
    json = "json"


class DeltaSide(StrEnum):
    baseline_only = "baseline_only"
    candidate_only = "candidate_only"


class ContextResolution(StrEnum):
    parquet = "parquet"
    artifact_details = "artifact_details"
    rule = "rule"
    metadata_only = "metadata_only"


_log_format = LogFormat.plain


class SignatureDeltaRow(BaseModel):
    workload_id: str
    row_index: int
    side: DeltaSide
    config_id: str
    signature_hash: str
    label: str | None = None
    source: str | None = None
    start_position: int | None = None
    end_position: int | None = None
    value_hash: str | None = None
    value_length: int | None = None
    masked_context: str | None = None
    resolution: ContextResolution = ContextResolution.metadata_only
    batch_file: str | None = None


class SignatureDeltaResult(BaseModel):
    baseline_artifacts: str
    candidate_artifacts: str
    workload_id: str | None = None
    baseline_config: str | None = None
    candidate_config: str | None = None
    delta_count: int
    rows: list[SignatureDeltaRow] = Field(default_factory=list)


class _ArtifactSide(BaseModel):
    artifacts_path: str
    artifact_root: str
    config_id: str | None = None


def configure_logging(log_format: LogFormat) -> None:
    global _log_format

    _log_format = log_format
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def log_bad_input(error: str) -> None:
    if _log_format == LogFormat.json:
        payload = {"level": "error", "event": "bad_input", "error": error}
        sys.stderr.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        return
    logger.error("bad_input error=%s", error)


def extract_signature_deltas(
    baseline_artifacts: Path,
    candidate_artifacts: Path,
    *,
    baseline_artifact_root: Path,
    candidate_artifact_root: Path,
    baseline_config: str | None = None,
    candidate_config: str | None = None,
    workload: str | None = None,
    context_window: int = 40,
) -> SignatureDeltaResult:
    baseline = _select_artifact_rows(
        _read_artifact_rows(baseline_artifacts), config_id=baseline_config, workload=workload
    )
    candidate = _select_artifact_rows(
        _read_artifact_rows(candidate_artifacts),
        config_id=candidate_config,
        workload=workload,
    )
    rows = _compare_artifact_rows(
        baseline,
        candidate,
        baseline_side=_artifact_side(baseline_artifacts, baseline_artifact_root, baseline_config),
        candidate_side=_artifact_side(candidate_artifacts, candidate_artifact_root, candidate_config),
        context_window=context_window,
    )
    return SignatureDeltaResult(
        baseline_artifacts=str(baseline_artifacts),
        candidate_artifacts=str(candidate_artifacts),
        workload_id=workload,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        delta_count=len(rows),
        rows=rows,
    )


def _artifact_side(artifacts_path: Path, artifact_root: Path, config_id: str | None) -> _ArtifactSide:
    return _ArtifactSide(
        artifacts_path=str(artifacts_path),
        artifact_root=str(artifact_root),
        config_id=config_id,
    )


def _read_artifact_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.is_dir():
        raise ValueError(f"detection artifact path is not a file: {path}")
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _select_artifact_rows(
    rows: list[dict[str, object]],
    *,
    config_id: str | None,
    workload: str | None,
) -> dict[tuple[str, int], dict[str, object]]:
    selected = [_row for _row in rows if _matches(_row, config_id=config_id, workload=workload)]
    if not selected:
        raise ValueError("artifact selector matched no rows")
    return {_artifact_key(row): row for row in selected}


def _matches(row: dict[str, object], *, config_id: str | None, workload: str | None) -> bool:
    if config_id is not None and str(row.get("config_id")) != config_id:
        return False
    if workload is not None and str(row.get("workload_id")) != workload:
        return False
    return True


def _artifact_key(row: dict[str, object]) -> tuple[str, int]:
    return str(row.get("workload_id")), int(row.get("row_index", 0))


def _compare_artifact_rows(
    baseline: dict[tuple[str, int], dict[str, object]],
    candidate: dict[tuple[str, int], dict[str, object]],
    *,
    baseline_side: _ArtifactSide,
    candidate_side: _ArtifactSide,
    context_window: int,
) -> list[SignatureDeltaRow]:
    rows: list[SignatureDeltaRow] = []
    for key in sorted(set(baseline) & set(candidate)):
        rows.extend(
            _row_signature_deltas(key, baseline[key], candidate[key], baseline_side, candidate_side, context_window)
        )
    return rows


def _row_signature_deltas(
    key: tuple[str, int],
    baseline_row: dict[str, object],
    candidate_row: dict[str, object],
    baseline_side: _ArtifactSide,
    candidate_side: _ArtifactSide,
    context_window: int,
) -> list[SignatureDeltaRow]:
    baseline_signatures = _signature_set(baseline_row)
    candidate_signatures = _signature_set(candidate_row)
    return [
        *_delta_rows(
            key,
            baseline_row,
            baseline_signatures - candidate_signatures,
            DeltaSide.baseline_only,
            baseline_side,
            context_window,
        ),
        *_delta_rows(
            key,
            candidate_row,
            candidate_signatures - baseline_signatures,
            DeltaSide.candidate_only,
            candidate_side,
            context_window,
        ),
    ]


def _signature_set(row: dict[str, object]) -> set[str]:
    raw = row.get("final_entity_signature_hashes", [])
    if isinstance(raw, str):
        raw = json.loads(raw)
    return {str(item) for item in raw} if isinstance(raw, list) else set()


def _delta_rows(
    key: tuple[str, int],
    artifact_row: dict[str, object],
    signatures: set[str],
    side: DeltaSide,
    artifact_side: _ArtifactSide,
    context_window: int,
) -> list[SignatureDeltaRow]:
    labels = _signature_labels(artifact_row)
    return [
        _contextual_delta_row(key, artifact_row, signature, labels.get(signature), side, artifact_side, context_window)
        for signature in sorted(signatures)
    ]


def _signature_labels(row: dict[str, object]) -> dict[str, str]:
    labels = row.get("final_entity_signature_labels", {})
    if isinstance(labels, str):
        labels = json.loads(labels)
    flattened = {
        key.removeprefix("final_entity_signature_labels."): str(value)
        for key, value in row.items()
        if key.startswith("final_entity_signature_labels.")
    }
    return {**(labels if isinstance(labels, dict) else {}), **flattened}


def _contextual_delta_row(
    key: tuple[str, int],
    artifact_row: dict[str, object],
    signature: str,
    label: str | None,
    side: DeltaSide,
    artifact_side: _ArtifactSide,
    context_window: int,
) -> SignatureDeltaRow:
    context = _resolve_signature_context(
        artifact_row, signature, label, Path(artifact_side.artifact_root), context_window
    )
    return SignatureDeltaRow(
        workload_id=key[0],
        row_index=key[1],
        side=side,
        config_id=str(artifact_row.get("config_id") or artifact_side.config_id or ""),
        signature_hash=signature,
        label=label,
        batch_file=_optional_string(artifact_row.get("batch_file")),
        **context,
    )


def _resolve_signature_context(
    artifact_row: dict[str, object],
    signature: str,
    label: str | None,
    artifact_root: Path,
    context_window: int,
) -> dict[str, object]:
    parquet_context = _parquet_entity_context(artifact_row, signature, artifact_root, context_window)
    if parquet_context is not None:
        return parquet_context
    rule_context = _rule_entity_context(artifact_row, signature, label, artifact_root, context_window)
    if rule_context is not None:
        return rule_context
    detail_context = _artifact_detail_context(artifact_row, signature, label, artifact_root, context_window)
    return detail_context or {"resolution": ContextResolution.metadata_only}


def _parquet_entity_context(
    artifact_row: dict[str, object],
    signature: str,
    artifact_root: Path,
    context_window: int,
) -> dict[str, object] | None:
    record = _artifact_record(artifact_row, artifact_root)
    if record is None:
        return None
    text, row_index, row = record
    for entity in EntitiesSchema.from_raw(row.get(COL_DETECTED_ENTITIES)).entities:
        if _entity_signature_hash(entity, row_index=row_index) == signature:
            return _entity_context(entity, text, context_window, ContextResolution.parquet)
    return None


def _rule_entity_context(
    artifact_row: dict[str, object],
    signature: str,
    label: str | None,
    artifact_root: Path,
    context_window: int,
) -> dict[str, object] | None:
    record = _artifact_record(artifact_row, artifact_root)
    if record is None or label is None:
        return None
    text, row_index, _row = record
    for span in detect_high_confidence_entities(text, labels=[label]):
        entity = EntitySchema.model_validate(span.as_dict())
        if _entity_signature_hash(entity, row_index=row_index) == signature:
            return _entity_context(entity, text, context_window, ContextResolution.rule)
    return None


def _artifact_detail_context(
    artifact_row: dict[str, object],
    signature: str,
    label: str | None,
    artifact_root: Path,
    context_window: int,
) -> dict[str, object] | None:
    details = _signature_details(artifact_row).get(signature)
    if details is None:
        return None
    start_position = _optional_int(details.get("start_position"))
    end_position = _optional_int(details.get("end_position"))
    value_hash = _optional_string(details.get("value_hash"))
    resolved_label = _optional_string(details.get("label")) or label
    if start_position is None or end_position is None or value_hash is None or resolved_label is None:
        return _metadata_context_from_details(details)
    record = _artifact_record(artifact_row, artifact_root)
    masked_context = None
    if record is not None:
        text, _row_index, _row = record
        masked_context = _masked_context_from_details(
            text,
            label=resolved_label,
            value_hash=value_hash,
            start_position=start_position,
            end_position=end_position,
            window=context_window,
        )
    return {
        "source": _optional_string(details.get("source")),
        "start_position": start_position,
        "end_position": end_position,
        "value_hash": value_hash,
        "value_length": _optional_int(details.get("value_length")),
        "masked_context": masked_context,
        "resolution": ContextResolution.artifact_details
        if masked_context is not None
        else ContextResolution.metadata_only,
    }


def _signature_details(row: dict[str, object]) -> dict[str, dict[str, object]]:
    details = _coerce_detail_map(row.get("final_entity_signature_details", {}))
    prefix = "final_entity_signature_details."
    for key, value in row.items():
        if not key.startswith(prefix):
            continue
        remainder = key.removeprefix(prefix)
        signature_hash, _, field = remainder.partition(".")
        if not signature_hash or not field:
            continue
        details.setdefault(signature_hash, {})[field] = value
    return details


def _coerce_detail_map(raw: object) -> dict[str, dict[str, object]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if isinstance(value, dict)}


def _metadata_context_from_details(details: dict[str, object]) -> dict[str, object]:
    return {
        "source": _optional_string(details.get("source")),
        "start_position": _optional_int(details.get("start_position")),
        "end_position": _optional_int(details.get("end_position")),
        "value_hash": _optional_string(details.get("value_hash")),
        "value_length": _optional_int(details.get("value_length")),
        "resolution": ContextResolution.metadata_only,
    }


def _artifact_record(artifact_row: dict[str, object], artifact_root: Path) -> tuple[str, int, pd.Series] | None:
    batch_file = _optional_string(artifact_row.get("batch_file"))
    if batch_file is None:
        return None
    parquet_file = artifact_root / batch_file
    if not parquet_file.exists():
        return None
    row_index = int(artifact_row.get("row_index", 0))
    row = pd.read_parquet(parquet_file).iloc[row_index]
    return str(row.get(COL_TEXT, "")), row_index, row


def _entity_context(
    entity: EntitySchema,
    text: str,
    context_window: int,
    resolution: ContextResolution,
) -> dict[str, object]:
    return {
        "source": entity.source,
        "start_position": entity.start_position,
        "end_position": entity.end_position,
        "value_hash": _value_hash(entity.value),
        "value_length": len(entity.value),
        "masked_context": _masked_context(text, entity, context_window),
        "resolution": resolution,
    }


def _masked_context(text: str, entity: EntitySchema, window: int) -> str:
    before = text[max(0, entity.start_position - window) : entity.start_position]
    after = text[entity.end_position : entity.end_position + window]
    placeholder = f"[{entity.label.upper()}:{_value_hash(entity.value)}]"
    return (before + placeholder + after).replace("\n", " ")


def _masked_context_from_details(
    text: str,
    *,
    label: str,
    value_hash: str,
    start_position: int,
    end_position: int,
    window: int,
) -> str:
    before = text[max(0, start_position - window) : start_position]
    after = text[end_position : end_position + window]
    placeholder = f"[{label.upper()}:{value_hash}]"
    return (before + placeholder + after).replace("\n", " ")


def _value_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _optional_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def write_rows(rows: list[SignatureDeltaRow], output_path: Path, export_format: ExportFormat) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.json_normalize([row.model_dump() for row in rows], sep=".")
    if export_format == ExportFormat.parquet:
        table.to_parquet(output_path, index=False)
    elif export_format == ExportFormat.csv:
        table.to_csv(output_path, index=False)
    else:
        table.to_json(output_path, orient="records", lines=True)


def render_result(result: SignatureDeltaResult, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    counts = pd.Series([row.side.value for row in result.rows]).value_counts().to_dict() if result.rows else {}
    return (
        f"Extracted {result.delta_count} signature delta(s)"
        f"; baseline_only={counts.get('baseline_only', 0)}"
        f"; candidate_only={counts.get('candidate_only', 0)}"
    )


@app.default
def main(
    baseline_artifacts: Path,
    candidate_artifacts: Path,
    *,
    baseline_artifact_root: Annotated[Path, cyclopts.Parameter("--baseline-artifact-root")],
    candidate_artifact_root: Annotated[Path, cyclopts.Parameter("--candidate-artifact-root")],
    baseline_config: Annotated[str | None, cyclopts.Parameter("--baseline-config")] = None,
    candidate_config: Annotated[str | None, cyclopts.Parameter("--candidate-config")] = None,
    workload: Annotated[str | None, cyclopts.Parameter("--workload")] = None,
    context_window: Annotated[int, cyclopts.Parameter("--context-window")] = 40,
    output: Annotated[Path | None, cyclopts.Parameter(("--output", "-o"))] = None,
    format: Annotated[ExportFormat, cyclopts.Parameter("--format")] = ExportFormat.jsonl,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        result = extract_signature_deltas(
            baseline_artifacts,
            candidate_artifacts,
            baseline_artifact_root=baseline_artifact_root,
            candidate_artifact_root=candidate_artifact_root,
            baseline_config=baseline_config,
            candidate_config=candidate_config,
            workload=workload,
            context_window=context_window,
        )
    except (ValueError, ValidationError) as exc:
        log_bad_input(str(exc))
        raise SystemExit(125) from exc
    if output is not None:
        write_rows(result.rows, output, format)
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")


if __name__ == "__main__":
    app()
