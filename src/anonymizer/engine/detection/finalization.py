# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from anonymizer.engine.detection.ignored_entity_values import DEFAULT_IGNORED_ENTITY_VALUES
from anonymizer.engine.detection.postprocess import normalize_label, normalize_labels
from anonymizer.engine.schemas import EntitiesSchema

logger = logging.getLogger("anonymizer.detection")


def _materialize_final_entities(
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
