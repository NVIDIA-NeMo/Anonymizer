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

from anonymizer.engine.detection.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_INITIAL_TAGGED_TEXT,
    COL_MERGED_ENTITIES,
    COL_MERGED_TAGGED_TEXT,
    COL_RAW_DETECTED,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATED_ENTITIES,
    COL_VALIDATION_CANDIDATES,
)
from anonymizer.engine.detection.postprocess import (
    EntitySpan,
    apply_augmented_entities,
    apply_validation_decisions,
    build_tagged_text,
    build_validation_candidates,
    expand_entity_occurrences,
    get_tag_notation,
    group_entities_by_value,
    parse_raw_entities,
)


@custom_column_generator(
    required_columns=[COL_TEXT, COL_RAW_DETECTED],
    side_effect_columns=[COL_INITIAL_TAGGED_TEXT, COL_SEED_ENTITIES_JSON, COL_TAG_NOTATION],
)
def parse_detected_entities(row: dict[str, Any]) -> dict[str, Any]:
    """Parse detector payload and emit initial tagged text + seed entities JSON."""
    text = str(row.get(COL_TEXT, ""))
    entities = parse_raw_entities(
        raw_response=str(row.get(COL_RAW_DETECTED, "")),
        text=text,
    )
    row[COL_SEED_ENTITIES] = [entity.as_dict() for entity in entities]
    row[COL_INITIAL_TAGGED_TEXT] = build_tagged_text(text=text, entities=entities)
    row[COL_SEED_ENTITIES_JSON] = json.dumps(row[COL_SEED_ENTITIES])
    row[COL_TAG_NOTATION] = get_tag_notation(text=text)
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_SEED_ENTITIES, COL_AUGMENTED_ENTITIES],
    side_effect_columns=[COL_MERGED_TAGGED_TEXT, COL_VALIDATION_CANDIDATES],
)
def merge_and_build_candidates(row: dict[str, Any]) -> dict[str, Any]:
    """Merge seed + augmented entities, then build tagged text and validation candidates."""
    text = str(row.get(COL_TEXT, ""))
    seed_spans = _from_dicts(entities=row.get(COL_SEED_ENTITIES, []))
    merged = apply_augmented_entities(
        text=text,
        entities=seed_spans,
        augmented_output=row.get(COL_AUGMENTED_ENTITIES, {}),
    )
    row[COL_MERGED_ENTITIES] = [entity.as_dict() for entity in merged]
    row[COL_MERGED_TAGGED_TEXT] = build_tagged_text(text=text, entities=merged)
    row[COL_VALIDATION_CANDIDATES] = build_validation_candidates(text=text, entities=merged)
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_MERGED_ENTITIES, COL_VALIDATED_ENTITIES],
    side_effect_columns=[COL_TAGGED_TEXT, COL_ENTITIES_BY_VALUE],
)
def apply_validation_and_finalize(row: dict[str, Any]) -> dict[str, Any]:
    """Apply keep/reclass/drop decisions, expand to all occurrences, and produce final outputs."""
    text = str(row.get(COL_TEXT, ""))
    merged = _from_dicts(entities=row.get(COL_MERGED_ENTITIES, []))
    validated = apply_validation_decisions(
        entities=merged,
        validation_output=row.get(COL_VALIDATED_ENTITIES, {}),
    )
    expanded = expand_entity_occurrences(text=text, entities=validated)
    row[COL_DETECTED_ENTITIES] = [entity.as_dict() for entity in expanded]
    row[COL_TAGGED_TEXT] = build_tagged_text(text=text, entities=expanded)
    row[COL_ENTITIES_BY_VALUE] = group_entities_by_value(entities=expanded)
    return row


def _from_dicts(entities: list[dict[str, str | int | float]]) -> list[EntitySpan]:
    result: list[EntitySpan] = []
    for entity in entities:
        result.append(
            EntitySpan(
                entity_id=str(entity.get("id", "")),
                value=str(entity.get("value", "")),
                label=str(entity.get("label", "")),
                start_position=int(entity.get("start_position", 0)),
                end_position=int(entity.get("end_position", 0)),
                score=float(entity.get("score", 0.0)),
                source=str(entity.get("source", "detector")),
            )
        )
    return result
