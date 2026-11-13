# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from functools import partial
from typing import Any

import pandas as pd
from data_designer.config import SkipConfig, custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from pydantic import BaseModel

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
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.rewrite.parsers import normalize_payload
from anonymizer.engine.schemas import EntitiesByValueSchema, EntityLabelClassificationsSchema
from anonymizer.interface.errors import AnonymizerWorkflowError

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


def _get_entity_classification_prompt(labels: list[str] | None = None) -> str:
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

Return one classification for every requested label. Do not include unrequested labels.

If <labels_to_classify> is empty, return an empty classifications list."""
    return substitute_placeholders(
        prompt,
        {
            "<<LABELS>>": json.dumps(labels, ensure_ascii=False)
            if labels is not None
            else _jinja(COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS),
        },
    )


def _validate_classifications(result: EntityLabelClassificationsSchema, requested: set[str]) -> None:
    categories: dict[str, str] = {}
    conflicts: set[str] = set()
    for item in result.classifications:
        if item.label not in requested:
            continue
        if item.label in categories and categories[item.label] != item.category:
            conflicts.add(item.label)
        categories[item.label] = item.category
    errors = []
    if conflicts:
        errors.append(f"Conflicting classifications for requested custom labels: {', '.join(sorted(conflicts))}.")
    missing = requested - categories.keys()
    if missing:
        errors.append(f"Missing classifications for requested custom labels: {', '.join(sorted(missing))}.")
    if errors:
        raise ValueError(
            " ".join(errors)
            + " Return the complete classifications list with exactly one entry per requested label, "
            + 'using category "direct_identifier" or "quasi_identifier". Do not include unrequested labels.'
        )


def _parse_classifications(response: str, *, requested: set[str]) -> EntityLabelClassificationsSchema:
    recipe = PydanticResponseRecipe(data_type=EntityLabelClassificationsSchema)
    result = EntityLabelClassificationsSchema.model_validate(recipe.parse(response))
    try:
        _validate_classifications(result, requested)
    except ValueError as exc:
        raise ParserException(str(exc), source=response) from exc
    return result


def _make_classification_column(classifier_alias: str) -> Any:
    @custom_column_generator(
        required_columns=[COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS],
        model_aliases=[classifier_alias],
    )
    def _classify(row: dict[str, Any], generator_params: Any, models: dict) -> dict[str, Any]:
        labels = normalize_payload(row[COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS])
        recipe = PydanticResponseRecipe(data_type=EntityLabelClassificationsSchema)
        result, _ = models[classifier_alias].generate(
            prompt=recipe.apply_recipe_to_user_prompt(_get_entity_classification_prompt(labels)),
            parser=partial(_parse_classifications, requested=set(labels)),
            max_correction_steps=1,
            max_conversation_restarts=0,
        )
        row[COL_ENTITY_CLASSIFICATION_LLM] = result.model_dump(mode="json")
        return row

    return _classify


# ---------------------------------------------------------------------------
# Merge local + LLM results
# ---------------------------------------------------------------------------


@custom_column_generator(
    required_columns=[
        COL_ENTITY_CLASSIFICATION_LOCAL,
        COL_ENTITY_CLASSIFICATION_LLM,
        COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS,
    ]
)
def _merge_entity_classifications(row: dict[str, Any]) -> dict[str, Any]:
    """Add only requested unmapped classifications, preserving authoritative local categories."""
    local = normalize_payload(row.get(COL_ENTITY_CLASSIFICATION_LOCAL)) or {}
    llm_raw = normalize_payload(row.get(COL_ENTITY_CLASSIFICATION_LLM))
    llm_result = (
        EntityLabelClassificationsSchema.model_validate(llm_raw) if llm_raw else EntityLabelClassificationsSchema()
    )
    merged = dict(local)
    requested_labels = set(normalize_payload(row.get(COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS)) or [])
    _validate_classifications(llm_result, requested_labels - set(local) - DEFAULT_ENTITY_LABEL_CATEGORY.keys())
    for item in llm_result.classifications:
        if item.label in local or item.label in DEFAULT_ENTITY_LABEL_CATEGORY or item.label not in requested_labels:
            logger.warning("Ignoring classification for unrequested or locally resolved label %r.", item.label)
            continue
        merged[item.label] = item.category
    row[COL_ENTITY_CLASSIFICATION] = merged
    return row


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class LabelClassificationParams(BaseModel):
    categories: dict[str, str]


@custom_column_generator(required_columns=[COL_ENTITIES_BY_VALUE])
def _apply_dataset_classifications(row: dict[str, Any], generator_params: LabelClassificationParams) -> dict[str, Any]:
    row[COL_ENTITY_CLASSIFICATION] = {
        label: generator_params.categories[label] for label in _unique_labels(row.get(COL_ENTITIES_BY_VALUE))
    }
    return row


class EntityClassificationWorkflow:
    """Classifies each non-latent entity label as direct or quasi-identifier.

    Labels covered by DEFAULT_ENTITY_LABEL_CATEGORY are resolved locally
    (no LLM call). Only labels outside that table — i.e. custom user-supplied
    entity_labels — go through a single dataset-level LLM classification.
    Each row then receives the categories for its own detected labels.

    Latent entities are out of scope — those are always classified as
    ``latent_identifier`` directly in the sensitivity disposition step.
    """

    def classify_dataset(
        self,
        dataframe: pd.DataFrame,
        *,
        adapter: NddAdapter,
        model_configs: list[ModelConfig],
        selected_models: RewriteModelSelection,
    ) -> dict[str, str]:
        labels = sorted({label for raw in dataframe[COL_ENTITIES_BY_VALUE] for label in _unique_labels(raw)})
        local, unmapped = classify_labels_locally(labels)
        if not unmapped:
            return local
        seed = pd.DataFrame(
            [{COL_ENTITY_CLASSIFICATION_LOCAL: local, COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS: unmapped}]
        )
        result = adapter.run_workflow(
            seed,
            model_configs=model_configs,
            columns=self.columns(selected_models=selected_models)[1:],
            workflow_name="rewrite-label-classification",
        )
        if result.failed_records or len(result.dataframe) != 1:
            raise AnonymizerWorkflowError(
                "Dataset custom-label classification failed; no shared taxonomy is available."
            )
        categories = normalize_payload(result.dataframe.iloc[0][COL_ENTITY_CLASSIFICATION])
        return {label: categories[label] for label in labels}

    def columns(
        self, *, selected_models: RewriteModelSelection, label_categories: dict[str, str] | None = None
    ) -> list[ColumnConfigT]:
        if label_categories is not None:
            return [
                CustomColumnConfig(
                    name=COL_ENTITY_CLASSIFICATION,
                    generator_function=_apply_dataset_classifications,
                    generator_params=LabelClassificationParams(categories=label_categories),
                )
            ]
        classifier_alias = resolve_model_alias("entity_classifier", selected_models)
        return [
            CustomColumnConfig(
                name=COL_ENTITY_CLASSIFICATION_LOCAL,
                generator_function=_classify_labels_locally,
            ),
            CustomColumnConfig(
                name=COL_ENTITY_CLASSIFICATION_LLM,
                generator_function=_make_classification_column(classifier_alias),
                skip=SkipConfig(
                    when=substitute_placeholders(
                        "{{ not <<UNMAPPED_LABELS>> }}",
                        {"<<UNMAPPED_LABELS>>": COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS},
                    ),
                    value=None,
                ),
            ),
            CustomColumnConfig(
                name=COL_ENTITY_CLASSIFICATION,
                generator_function=_merge_entity_classifications,
                propagate_skip=False,
            ),
        ]
