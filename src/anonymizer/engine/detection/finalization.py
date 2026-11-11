# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_FINAL_ENTITIES, COL_TAGGED_TEXT, COL_TEXT
from anonymizer.engine.detection.ignored_entity_values import DEFAULT_IGNORED_ENTITY_VALUES
from anonymizer.engine.detection.postprocess import EntitySpan, build_tagged_text, normalize_label, normalize_labels
from anonymizer.engine.schemas import EntitiesSchema

logger = logging.getLogger("anonymizer.detection")


def build_final_tagged_text(text: str, raw: object) -> str:
    """Render public tags from filtered entities while retaining pre-filter detections."""
    parsed = EntitiesSchema.from_raw(raw)
    return build_tagged_text(
        text=text,
        entities=[
            EntitySpan(
                entity_id=entity.id,
                value=entity.value,
                label=entity.label,
                start_position=entity.start_position,
                end_position=entity.end_position,
                score=entity.score,
                source=entity.source,
            )
            for entity in parsed.entities
        ],
    )


def materialize_final_entity_output(
    row: dict[str, Any],
    *,
    allowed_labels: set[str] | None,
    excluded_entity_labels: set[str] | None,
    ignored_entity_values: frozenset[str],
) -> dict[str, Any]:
    """Publish filtered entities and matching public span tags from pre-filter detections."""
    row[COL_FINAL_ENTITIES] = materialize_final_entities(
        row[COL_DETECTED_ENTITIES],
        allowed_labels=allowed_labels,
        excluded_entity_labels=excluded_entity_labels,
        ignored_entity_values=ignored_entity_values,
    )
    row[COL_TAGGED_TEXT] = build_final_tagged_text(row[COL_TEXT], row[COL_FINAL_ENTITIES])
    return row


def materialize_final_entities(
    raw: object,
    *,
    allowed_labels: set[str] | None,
    excluded_entity_labels: set[str] | None = None,
    ignored_entity_values: frozenset[str] = DEFAULT_IGNORED_ENTITY_VALUES,
) -> dict:
    """Build COL_FINAL_ENTITIES, applying label scope and exact value exclusions."""
    parsed = EntitiesSchema.from_raw(raw)
    allowed = normalize_labels(allowed_labels) if allowed_labels is not None else None
    excluded = normalize_labels(excluded_entity_labels)
    kept = []
    for entity in parsed.entities:
        label = normalize_label(entity.label)
        if (allowed is not None and label not in allowed) or label in excluded:
            continue
        if entity.value in ignored_entity_values:
            logger.info(
                "Excluded entity matching ignored_entity_values: value=%r label=%s",
                entity.value,
                entity.label,
            )
            continue
        kept.append(entity)
    return EntitiesSchema(entities=kept).model_dump()
