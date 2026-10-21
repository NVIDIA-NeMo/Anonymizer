# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom column generators for the entity detection workflow.

Each function is a single step in the NDD pipeline defined by
``EntityDetectionWorkflow.detect_and_validate_entities``.  They use the
``@custom_column_generator`` decorator so DataDesigner can execute them
as row-level transforms between LLM calls.
"""

from __future__ import annotations

import json
from typing import Any

from data_designer.config import custom_column_generator

from anonymizer.engine.constants import (
    COL_ADMISSION_RESULTS,
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_MERGED_ENTITIES,
    COL_MERGED_TAGGED_TEXT,
    COL_RAW_DETECTED,
    COL_SEED_DETECTION_RESULT,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_SEED_TAGGED_TEXT,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATED_SEED_ENTITIES,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_INVENTORIES,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.admission import apply_validation_admission
from anonymizer.engine.detection.backend import _GlinerDetectionBackend
from anonymizer.engine.detection.postprocess import (
    EntitySpan,
    apply_augmented_entities,
    build_tagged_text,
    build_validation_candidates,
    expand_entity_occurrences,
    filter_excluded_entity_spans,
    get_tag_notation,
    normalize_label,
    normalize_labels,
    project_detection_result_to_entity_spans,
)
from anonymizer.engine.schemas import (
    AdmissionResults,
    AdmissionStatus,
    DetectionResult,
    EntitiesSchema,
    SourceTextRef,
    SourceView,
    ValidationCandidatesSchema,
    ValidationRequestDecisionSets,
    ValidationRequestInventory,
)


@custom_column_generator(
    required_columns=[RECORD_ID_COLUMN, COL_TEXT, COL_RAW_DETECTED],
    side_effect_columns=[COL_SEED_DETECTION_RESULT, COL_TAG_NOTATION],
)
def parse_detected_entities(
    row: dict[str, Any],
    *,
    detection_result_id: str,
    source_instance: str,
) -> dict[str, Any]:
    """Parse detector payload and produce seed entities."""
    text = str(row.get(COL_TEXT, ""))
    source_text_ref = SourceTextRef(
        record_id=str(row[RECORD_ID_COLUMN]),
        field_id=COL_TEXT,
    )
    detection_result = _GlinerDetectionBackend(source_instance=source_instance).to_detection_result(
        detection_result_id=detection_result_id,
        source_view=SourceView(
            source_text_ref=source_text_ref,
            start_position=0,
            end_position=len(text),
        ),
        canonical_text=text,
        raw_output=row.get(COL_RAW_DETECTED, ""),
    )
    entities = project_detection_result_to_entity_spans(detection_result)
    seed_entities = [entity.as_dict() for entity in entities]
    row[COL_SEED_ENTITIES] = EntitiesSchema(entities=seed_entities).model_dump(mode="json")
    row[COL_SEED_DETECTION_RESULT] = detection_result.model_dump(mode="json")
    row[COL_TAG_NOTATION] = get_tag_notation(text=text)
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_VALIDATED_SEED_ENTITIES, COL_AUGMENTED_ENTITIES],
    side_effect_columns=[COL_MERGED_TAGGED_TEXT, COL_VALIDATION_CANDIDATES],
)
def merge_and_build_candidates(
    row: dict[str, Any],
    *,
    excluded_entity_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Merge validated seed + augmented entities, then build tagged text and validation candidates.

    Contract:
    - ``COL_VALIDATED_SEED_ENTITIES`` and ``COL_MERGED_ENTITIES`` store ``EntitiesSchema`` payloads.
    - ``COL_VALIDATION_CANDIDATES`` stores ``ValidationCandidatesSchema`` payloads.
    """
    text = str(row.get(COL_TEXT, ""))
    seed_spans = _parse_entity_spans(row.get(COL_VALIDATED_SEED_ENTITIES, {}))
    merged = apply_augmented_entities(
        text=text,
        entities=seed_spans,
        augmented_output=row.get(COL_AUGMENTED_ENTITIES, {}),
        excluded_entity_labels=set(excluded_entity_labels or []),
    )
    merged_entities = [entity.as_dict() for entity in merged]
    row[COL_MERGED_ENTITIES] = EntitiesSchema(entities=merged_entities).model_dump(mode="json")
    row[COL_MERGED_TAGGED_TEXT] = build_tagged_text(text=text, entities=merged)
    row[COL_VALIDATION_CANDIDATES] = ValidationCandidatesSchema(
        candidates=build_validation_candidates(text=text, entities=merged)
    ).model_dump(mode="json")
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_SEED_DETECTION_RESULT, COL_ADMISSION_RESULTS],
    side_effect_columns=[COL_INITIAL_TAGGED_TEXT, COL_SEED_ENTITIES_JSON, COL_VALIDATED_SEED_ENTITIES],
)
def apply_validation_to_seed_entities(
    row: dict[str, Any],
    *,
    excluded_entity_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Materialize retained detector observations from terminal admission state."""
    text = str(row.get(COL_TEXT, ""))
    validated_seed = _project_admission_results(
        detection_result=DetectionResult.model_validate(row[COL_SEED_DETECTION_RESULT]),
        admission_results=AdmissionResults.model_validate(row[COL_ADMISSION_RESULTS]),
    )
    validated_seed = filter_excluded_entity_spans(validated_seed, excluded_entity_labels)
    seed_entities = [entity.as_dict() for entity in validated_seed]
    row[COL_VALIDATED_SEED_ENTITIES] = EntitiesSchema(entities=seed_entities).model_dump(mode="json")
    row[COL_SEED_ENTITIES_JSON] = json.dumps(seed_entities)
    row[COL_INITIAL_TAGGED_TEXT] = build_tagged_text(text=text, entities=validated_seed)
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_SEED_ENTITIES],
    side_effect_columns=[COL_SEED_TAGGED_TEXT],
)
def prepare_validation_inputs(row: dict[str, Any]) -> dict[str, Any]:
    """Build validation prompt inputs from detector entities only (pre-augmentation)."""
    text = str(row.get(COL_TEXT, ""))
    seed_spans = _parse_entity_spans(row.get(COL_SEED_ENTITIES, {}))
    row[COL_SEED_TAGGED_TEXT] = build_tagged_text(text=text, entities=seed_spans)
    row[COL_SEED_VALIDATION_CANDIDATES] = ValidationCandidatesSchema(
        candidates=build_validation_candidates(text=text, entities=seed_spans)
    ).model_dump(mode="json")
    return row


@custom_column_generator(
    required_columns=[
        COL_SEED_DETECTION_RESULT,
        COL_VALIDATION_DECISIONS,
        COL_VALIDATION_INVENTORIES,
    ],
)
def admit_validation_decisions(row: dict[str, Any]) -> dict[str, Any]:
    """Consume request-scoped decisions into one terminal result per observation."""
    detection_result = DetectionResult.model_validate(row[COL_SEED_DETECTION_RESULT])
    inventories = tuple(ValidationRequestInventory.model_validate(item) for item in row[COL_VALIDATION_INVENTORIES])
    admission_results = apply_validation_admission(
        detection_result=detection_result,
        inventories=inventories,
        decision_sets=ValidationRequestDecisionSets.model_validate(row[COL_VALIDATION_DECISIONS]),
    )
    row[COL_ADMISSION_RESULTS] = AdmissionResults(results=admission_results).model_dump(mode="json")
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_MERGED_ENTITIES],
    side_effect_columns=[COL_TAGGED_TEXT],
)
def finalize_entities(
    row: dict[str, Any],
    *,
    excluded_entity_labels: list[str] | None = None,
    allowed_entity_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Apply label scope, expand retained values, and produce final outputs."""
    text = str(row.get(COL_TEXT, ""))
    validated = _parse_entity_spans(row.get(COL_MERGED_ENTITIES, {}))
    if allowed_entity_labels is not None:
        allowed = normalize_labels(allowed_entity_labels)
        validated = [entity for entity in validated if normalize_label(entity.label) in allowed]
    validated = filter_excluded_entity_spans(validated, excluded_entity_labels)
    expanded = expand_entity_occurrences(text=text, entities=validated)
    row[COL_DETECTED_ENTITIES] = EntitiesSchema(entities=[entity.as_dict() for entity in expanded]).model_dump(
        mode="json"
    )
    row[COL_TAGGED_TEXT] = build_tagged_text(text=text, entities=expanded)
    return row


def _parse_entity_spans(raw_payload: object) -> list[EntitySpan]:
    parsed = EntitiesSchema.from_raw(raw_payload)
    return [
        EntitySpan(
            entity_id=e.id,
            value=e.value,
            label=e.label,
            start_position=e.start_position,
            end_position=e.end_position,
            score=e.score,
            source=e.source,
        )
        for e in parsed.entities
    ]


def _project_admission_results(
    *,
    detection_result: DetectionResult,
    admission_results: AdmissionResults,
) -> list[EntitySpan]:
    observations_by_ref = {item.ref: item for item in detection_result.observations}
    projected: list[EntitySpan] = []
    for result in admission_results.results:
        if result.status is AdmissionStatus.rejected:
            continue
        observation = observations_by_ref.get(result.observation_ref)
        if observation is None:
            raise ValueError("admission result references an unknown observation")
        label = result.effective_label
        if not label:
            raise ValueError("retained admission result has no effective label")
        projected.append(
            EntitySpan(
                entity_id=f"{label}_{observation.start_position}_{observation.end_position}",
                value=observation.value,
                label=label,
                start_position=observation.start_position,
                end_position=observation.end_position,
                score=observation.native_score or 0.0,
                source="detector",
            )
        )
    return projected
