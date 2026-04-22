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
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_MERGED_ENTITIES,
    COL_MERGED_TAGGED_TEXT,
    COL_RAW_DETECTED,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_SEED_TAGGED_TEXT,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATED_ENTITIES,
    COL_VALIDATED_SEED_ENTITIES,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_SKELETON,
)
from anonymizer.engine.detection.postprocess import (
    EntitySpan,
    apply_augmented_entities,
    apply_validation_decisions,
    build_tagged_text,
    build_validation_candidates,
    expand_entity_occurrences,
    get_tag_notation,
    parse_raw_entities,
)
from anonymizer.engine.schemas import (
    EntitiesSchema,
    RawValidationDecisionsSchema,
    ValidatedDecisionSchema,
    ValidatedDecisionsSchema,
    ValidationCandidatesSchema,
    ValidationSkeletonDecisionSchema,
    ValidationSkeletonSchema,
)


@custom_column_generator(
    required_columns=[COL_TEXT, COL_RAW_DETECTED],
    side_effect_columns=[COL_TAG_NOTATION],
)
def parse_detected_entities(row: dict[str, Any]) -> dict[str, Any]:
    """Parse detector payload and produce seed entities."""
    text = str(row.get(COL_TEXT, ""))
    entities = parse_raw_entities(
        raw_response=str(row.get(COL_RAW_DETECTED, "")),
        text=text,
    )
    seed_entities = [entity.as_dict() for entity in entities]
    row[COL_SEED_ENTITIES] = EntitiesSchema(entities=seed_entities).model_dump(mode="json")
    row[COL_TAG_NOTATION] = get_tag_notation(text=text)
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_VALIDATED_SEED_ENTITIES, COL_AUGMENTED_ENTITIES],
    side_effect_columns=[COL_MERGED_TAGGED_TEXT, COL_VALIDATION_CANDIDATES],
)
def merge_and_build_candidates(row: dict[str, Any]) -> dict[str, Any]:
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
    )
    merged_entities = [entity.as_dict() for entity in merged]
    row[COL_MERGED_ENTITIES] = EntitiesSchema(entities=merged_entities).model_dump(mode="json")
    row[COL_MERGED_TAGGED_TEXT] = build_tagged_text(text=text, entities=merged)
    row[COL_VALIDATION_CANDIDATES] = ValidationCandidatesSchema(
        candidates=build_validation_candidates(text=text, entities=merged)
    ).model_dump(mode="json")
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_SEED_ENTITIES, COL_VALIDATED_ENTITIES],
    side_effect_columns=[COL_INITIAL_TAGGED_TEXT, COL_SEED_ENTITIES_JSON, COL_VALIDATED_SEED_ENTITIES],
)
def apply_validation_to_seed_entities(row: dict[str, Any]) -> dict[str, Any]:
    """Apply validation decisions to detector entities before augmentation."""
    text = str(row.get(COL_TEXT, ""))
    seed_spans = _parse_entity_spans(row.get(COL_SEED_ENTITIES, {}))
    validated_seed = apply_validation_decisions(
        entities=seed_spans,
        validation_output=row.get(COL_VALIDATED_ENTITIES, {}),
    )
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


@custom_column_generator(required_columns=[COL_SEED_VALIDATION_CANDIDATES])
def build_validation_skeleton(row: dict[str, Any]) -> dict[str, Any]:
    """Pre-populate the decisions template with candidate IDs so the LLM only fills in decision/reason."""
    candidates = ValidationCandidatesSchema.from_raw(row.get(COL_SEED_VALIDATION_CANDIDATES, {}))
    skeleton = ValidationSkeletonSchema(
        decisions=[
            ValidationSkeletonDecisionSchema(id=c.id, value=c.value, label=c.label) for c in candidates.candidates
        ]
    )
    row[COL_VALIDATION_SKELETON] = skeleton.model_dump(mode="json")
    return row


@custom_column_generator(
    required_columns=[COL_VALIDATION_DECISIONS, COL_SEED_VALIDATION_CANDIDATES],
)
def enrich_validation_decisions(row: dict[str, Any]) -> dict[str, Any]:
    """Enrich validation decisions with entity value and filter to known candidate IDs only."""
    raw_decisions = RawValidationDecisionsSchema.from_raw(row.get(COL_VALIDATION_DECISIONS, {}))
    candidates = ValidationCandidatesSchema.from_raw(row.get(COL_SEED_VALIDATION_CANDIDATES, {}))

    candidate_lookup = {c.id: c for c in candidates.candidates}
    valid_ids = set(candidate_lookup)

    enriched = [
        ValidatedDecisionSchema(
            id=d.id,
            decision=d.decision,
            proposed_label=d.proposed_label,
            reason=d.reason,
            value=candidate_lookup[d.id].value,
            label=candidate_lookup[d.id].label,
        )
        for d in raw_decisions.decisions
        if d.id in valid_ids
    ]

    row[COL_VALIDATED_ENTITIES] = ValidatedDecisionsSchema(decisions=enriched).model_dump(mode="json")
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_MERGED_ENTITIES, COL_VALIDATED_ENTITIES],
    side_effect_columns=[COL_TAGGED_TEXT],
)
def apply_validation_and_finalize(row: dict[str, Any]) -> dict[str, Any]:
    """Apply keep/reclass/drop decisions, expand to all occurrences, and produce final outputs."""
    text = str(row.get(COL_TEXT, ""))
    merged = _parse_entity_spans(row.get(COL_MERGED_ENTITIES, {}))
    validated = apply_validation_decisions(
        entities=merged,
        validation_output=row.get(COL_VALIDATED_ENTITIES, {}),
    )
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
