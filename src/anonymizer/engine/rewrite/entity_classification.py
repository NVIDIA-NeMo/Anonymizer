# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT

from anonymizer.config.models import RewriteModelSelection
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_CLASSIFICATION,
    COL_ENTITY_CLASSIFICATION_LLM,
    COL_ENTITY_CLASSIFICATION_LOCAL,
    COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS,
    DEFAULT_ENTITY_LABEL_CATEGORY,
    _jinja,
)
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.rewrite.parsers import normalize_payload
from anonymizer.engine.schemas import EntitiesByValueSchema, EntityLabelClassificationsSchema

logger = logging.getLogger("anonymizer.rewrite.entity_classification")


def _unique_labels(entities_by_value_raw: object) -> list[str]:
    """Return the sorted set of unique entity labels present in a row's detected entities."""
    parsed = EntitiesByValueSchema.from_raw(entities_by_value_raw)
    labels: set[str] = set()
    for entity in parsed.entities_by_value:
        labels.update(entity.labels)
    return sorted(labels)


def classify_labels_locally(labels: list[str]) -> tuple[dict[str, str], list[str]]:
    """Split *labels* into (resolved category map, unmapped labels).

    Resolved labels come from DEFAULT_ENTITY_LABEL_CATEGORY. Unmapped labels
    are custom entity_labels the caller supplied that aren't in that table
    and must go through the LLM classification fallback.
    """
    resolved: dict[str, str] = {}
    unmapped: list[str] = []
    for label in labels:
        category = DEFAULT_ENTITY_LABEL_CATEGORY.get(label)
        if category is None:
            unmapped.append(label)
        else:
            resolved[label] = category
    return resolved, unmapped


# ---------------------------------------------------------------------------
# Local (no-LLM) classification
# ---------------------------------------------------------------------------


@custom_column_generator(
    required_columns=[COL_ENTITIES_BY_VALUE],
    side_effect_columns=[COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS],
)
def _classify_labels_locally(row: dict[str, Any]) -> dict[str, Any]:
    """Resolve each unique label in the row's detected entities via DEFAULT_ENTITY_LABEL_CATEGORY.

    Labels outside that table (custom user-supplied entity_labels) are collected
    into COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS for the LLM fallback prompt.
    Scoped per-document: only labels actually present in this row are considered.
    """
    labels = _unique_labels(row.get(COL_ENTITIES_BY_VALUE))
    resolved, unmapped = classify_labels_locally(labels)
    row[COL_ENTITY_CLASSIFICATION_LOCAL] = resolved
    row[COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS] = unmapped
    return row


# ---------------------------------------------------------------------------
# LLM fallback classification (unmapped labels only)
# ---------------------------------------------------------------------------


def _get_entity_classification_prompt() -> str:
    prompt = """Classify each entity label below as either a direct identifier or a quasi-identifier.

DIRECT IDENTIFIERS uniquely identify an individual (or their account/device) on their own.
Examples: full name, email, phone number, SSN, exact address, account number, medical record number.

QUASI-IDENTIFIERS are not identifying alone, but narrow identity in combination with other
known facts. Examples: age, city, occupation, employer, gender, nationality, education.

Classify based on what the label represents in general, not the specific value in any one
document — this is a fixed taxonomy decision, not a contextual risk assessment.

<labels_to_classify>
<<LABELS>>
</labels_to_classify>

If <labels_to_classify> is empty, return an empty classifications list."""
    return substitute_placeholders(
        prompt,
        {
            "<<LABELS>>": _jinja(COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS),
        },
    )


# ---------------------------------------------------------------------------
# Merge local + LLM results
# ---------------------------------------------------------------------------


@custom_column_generator(required_columns=[COL_ENTITY_CLASSIFICATION_LOCAL, COL_ENTITY_CLASSIFICATION_LLM])
def _merge_entity_classifications(row: dict[str, Any]) -> dict[str, Any]:
    """Merge the local (default-table) and LLM (unmapped-label) classifications into one map."""
    local = normalize_payload(row.get(COL_ENTITY_CLASSIFICATION_LOCAL)) or {}
    llm_raw = normalize_payload(row.get(COL_ENTITY_CLASSIFICATION_LLM))
    llm_result = (
        EntityLabelClassificationsSchema.model_validate(llm_raw) if llm_raw else EntityLabelClassificationsSchema()
    )
    merged = dict(local)
    for item in llm_result.classifications:
        merged[item.label] = item.category
    row[COL_ENTITY_CLASSIFICATION] = merged
    return row


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class EntityClassificationWorkflow:
    """Classifies each non-latent entity label as direct or quasi-identifier.

    Labels covered by DEFAULT_ENTITY_LABEL_CATEGORY are resolved locally
    (no LLM call). Only labels outside that table — i.e. custom user-supplied
    entity_labels — go through an LLM call. Scoped per-document: only labels
    actually present in a row's detected entities are ever sent to the LLM.

    Latent entities are out of scope — those are always classified as
    ``latent_identifier`` directly in the sensitivity disposition step.
    """

    def columns(self, *, selected_models: RewriteModelSelection) -> list[ColumnConfigT]:
        classifier_alias = resolve_model_alias("entity_classifier", selected_models)
        return [
            CustomColumnConfig(
                name=COL_ENTITY_CLASSIFICATION_LOCAL,
                generator_function=_classify_labels_locally,
            ),
            LLMStructuredColumnConfig(
                name=COL_ENTITY_CLASSIFICATION_LLM,
                prompt=_get_entity_classification_prompt(),
                model_alias=classifier_alias,
                output_format=EntityLabelClassificationsSchema,
            ),
            CustomColumnConfig(
                name=COL_ENTITY_CLASSIFICATION,
                generator_function=_merge_entity_classifications,
            ),
        ]
