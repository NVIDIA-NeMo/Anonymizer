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
    COL_ADMITTED_CANDIDATES,
    COL_AUGMENTED_CANDIDATES,
    COL_DETECTED_ENTITIES,
    COL_DETECTOR_SOURCE,
    COL_INITIAL_TAGGED_TEXT,
    COL_MERGED_ENTITIES,
    COL_RAW_DETECTED,
    COL_RECONCILED_CANDIDATES,
    COL_SEED_DETECTION_RESULT,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_INVENTORIES,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.admission import apply_validation_admission
from anonymizer.engine.detection.backend import _GlinerDetectionBackend
from anonymizer.engine.detection.candidates import (
    coalesce_exact_candidates,
    materialize_admitted_candidates,
    reconcile_candidates,
    representative_source,
)
from anonymizer.engine.detection.postprocess import (
    EntitySpan,
    build_conflict_aware_tagged_text,
    build_tagged_text,
    build_validation_candidates,
    get_tag_notation,
    project_detection_result_to_entity_spans,
)
from anonymizer.engine.schemas import (
    AdmissionResults,
    AdmittedCandidate,
    AdmittedCandidates,
    DetectionResult,
    EffectiveLabelPolicy,
    EntitiesSchema,
    SourceTextRef,
    SourceView,
    ValidationCandidatesSchema,
    ValidationRequestDecisionSets,
    ValidationRequestInventory,
)


@custom_column_generator(required_columns=[COL_TEXT])
def prepare_detector_source(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve exact source text through DataDesigner's automatic JSON decoding.

    DataDesigner decodes JSON-valued columns before rendering LLM prompts.
    Encoding an outer object lets it decode that object while leaving the
    original source string, including JSON formatting and escapes, untouched.
    """
    row[COL_DETECTOR_SOURCE] = json.dumps({"text": row[COL_TEXT]}, ensure_ascii=False)
    return row


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
    required_columns=[
        COL_TEXT,
        COL_ADMITTED_CANDIDATES,
        COL_AUGMENTED_CANDIDATES,
    ],
    side_effect_columns=[COL_RECONCILED_CANDIDATES, COL_VALIDATION_CANDIDATES],
)
def merge_and_build_candidates(row: dict[str, Any]) -> dict[str, Any]:
    """Reconcile admitted seed and augmentation candidates, then build legacy projections.

    Contract:
    - Both input candidate columns store ``AdmittedCandidates`` payloads.
    - ``COL_MERGED_ENTITIES`` stores an ``EntitiesSchema`` payload.
    - ``COL_VALIDATION_CANDIDATES`` stores ``ValidationCandidatesSchema`` payloads.
    """
    text = str(row.get(COL_TEXT, ""))
    seed_candidates = AdmittedCandidates.model_validate(row[COL_ADMITTED_CANDIDATES]).candidates
    augmented_candidates = AdmittedCandidates.model_validate(row[COL_AUGMENTED_CANDIDATES]).candidates
    reconciled = reconcile_candidates(coalesce_exact_candidates((*seed_candidates, *augmented_candidates)))
    row[COL_RECONCILED_CANDIDATES] = AdmittedCandidates(candidates=reconciled).model_dump(mode="json")
    merged = _project_admitted_candidates(reconciled)
    merged_entities = [entity.as_dict() for entity in merged]
    row[COL_MERGED_ENTITIES] = EntitiesSchema(entities=merged_entities).model_dump(mode="json")
    row[COL_VALIDATION_CANDIDATES] = ValidationCandidatesSchema(
        candidates=build_validation_candidates(text=text, entities=merged)
    ).model_dump(mode="json")
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_SEED_DETECTION_RESULT, COL_ADMISSION_RESULTS],
    side_effect_columns=[
        COL_ADMITTED_CANDIDATES,
        COL_INITIAL_TAGGED_TEXT,
        COL_SEED_ENTITIES_JSON,
    ],
)
def apply_validation_to_seed_entities(row: dict[str, Any]) -> dict[str, Any]:
    """Materialize retained detector observations from terminal admission state."""
    text = str(row.get(COL_TEXT, ""))
    candidates = coalesce_exact_candidates(
        materialize_admitted_candidates(
            canonical_text=text,
            detection_result=DetectionResult.model_validate(row[COL_SEED_DETECTION_RESULT]),
            admission_results=AdmissionResults.model_validate(row[COL_ADMISSION_RESULTS]),
        )
    )
    row[COL_ADMITTED_CANDIDATES] = AdmittedCandidates(candidates=candidates).model_dump(mode="json")
    validated_seed = _project_admitted_candidates(candidates)
    seed_entities = [entity.as_dict() for entity in validated_seed]
    row[COL_SEED_ENTITIES_JSON] = json.dumps(seed_entities)
    row[COL_INITIAL_TAGGED_TEXT] = build_conflict_aware_tagged_text(text=text, entities=validated_seed)
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_SEED_ENTITIES],
)
def prepare_validation_inputs(row: dict[str, Any]) -> dict[str, Any]:
    """Build seed candidate context for measurement, not validator prompt rendering."""
    text = str(row.get(COL_TEXT, ""))
    seed_spans = _parse_entity_spans(row.get(COL_SEED_ENTITIES, {}))
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
def admit_validation_decisions(
    row: dict[str, Any],
    *,
    allowed_entity_labels: list[str] | None = None,
    excluded_entity_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Admit observations under the configured label policy, checking inventory agreement."""
    detection_result = DetectionResult.model_validate(row[COL_SEED_DETECTION_RESULT])
    inventories = tuple(ValidationRequestInventory.model_validate(item) for item in row[COL_VALIDATION_INVENTORIES])
    admission_results = apply_validation_admission(
        detection_result=detection_result,
        inventories=inventories,
        decision_sets=ValidationRequestDecisionSets.model_validate(row[COL_VALIDATION_DECISIONS]),
        effective_label_policy=EffectiveLabelPolicy.from_labels(
            allowed_labels=allowed_entity_labels,
            excluded_labels=excluded_entity_labels,
        ),
    )
    row[COL_ADMISSION_RESULTS] = AdmissionResults(results=admission_results).model_dump(mode="json")
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_RECONCILED_CANDIDATES],
    side_effect_columns=[COL_TAGGED_TEXT],
)
def finalize_entities(row: dict[str, Any]) -> dict[str, Any]:
    """Publish the occurrence-specific candidates selected by reconciliation."""
    text = str(row.get(COL_TEXT, ""))
    reconciled = AdmittedCandidates.model_validate(row[COL_RECONCILED_CANDIDATES]).candidates
    finalized = _project_admitted_candidates(reconciled)
    row[COL_DETECTED_ENTITIES] = EntitiesSchema(entities=[entity.as_dict() for entity in finalized]).model_dump(
        mode="json"
    )
    row[COL_TAGGED_TEXT] = build_tagged_text(text=text, entities=finalized)
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


def _project_admitted_candidates(candidates: tuple[AdmittedCandidate, ...]) -> list[EntitySpan]:
    """Project provenance-bearing candidates into the legacy entity payload."""
    projected: list[EntitySpan] = []
    for candidate in candidates:
        source = representative_source(candidate)
        projected.append(
            EntitySpan(
                entity_id=f"{candidate.label}_{candidate.start_position}_{candidate.end_position}",
                value=candidate.value,
                label=candidate.label,
                start_position=candidate.start_position,
                end_position=candidate.end_position,
                score=max((score.score for score in candidate.native_scores), default=0.0),
                source="detector" if source == "initial_detector" else source,
            )
        )
    return projected
