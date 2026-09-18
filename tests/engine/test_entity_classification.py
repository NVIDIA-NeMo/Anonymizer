# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig

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
from anonymizer.engine.rewrite.entity_classification import (
    EntityClassificationWorkflow,
    _classify_labels_locally,
    _get_entity_classification_prompt,
    _merge_entity_classifications,
    _unique_labels,
    classify_labels_locally,
)


def test_columns_returns_exactly_three_in_order(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    cols = EntityClassificationWorkflow().columns(selected_models=stub_rewrite_model_selection)
    assert len(cols) == 3
    assert isinstance(cols[0], CustomColumnConfig)
    assert isinstance(cols[1], LLMStructuredColumnConfig)
    assert isinstance(cols[2], CustomColumnConfig)
    assert cols[0].name == COL_ENTITY_CLASSIFICATION_LOCAL
    assert cols[1].name == COL_ENTITY_CLASSIFICATION_LLM
    assert cols[2].name == COL_ENTITY_CLASSIFICATION
    assert cols[1].model_alias == stub_rewrite_model_selection.entity_classifier


def test_prompt_references_unmapped_labels_column() -> None:
    prompt = _get_entity_classification_prompt()
    assert _jinja(COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS) in prompt


def test_classify_labels_locally_resolves_default_labels() -> None:
    resolved, unmapped = classify_labels_locally(["first_name", "city"])
    assert resolved == {"first_name": "direct_identifier", "city": "quasi_identifier"}
    assert unmapped == []


def test_classify_labels_locally_collects_unmapped_labels() -> None:
    resolved, unmapped = classify_labels_locally(["first_name", "custom_widget_id"])
    assert resolved == {"first_name": "direct_identifier"}
    assert unmapped == ["custom_widget_id"]


def test_classify_labels_locally_covers_every_default_label() -> None:
    resolved, unmapped = classify_labels_locally(list(DEFAULT_ENTITY_LABEL_CATEGORY.keys()))
    assert unmapped == []
    assert resolved == DEFAULT_ENTITY_LABEL_CATEGORY


def test_unique_labels_deduplicates_across_entities() -> None:
    raw = {
        "entities_by_value": [
            {"value": "Alice", "labels": ["first_name"]},
            {"value": "Bob", "labels": ["first_name"]},
            {"value": "Portland", "labels": ["city"]},
        ]
    }
    assert _unique_labels(raw) == ["city", "first_name"]


def test_unique_labels_handles_empty_input() -> None:
    assert _unique_labels({}) == []
    assert _unique_labels(None) == []


def test_classify_labels_locally_generator_populates_local_and_unmapped_columns() -> None:
    row = {
        COL_ENTITIES_BY_VALUE: {
            "entities_by_value": [
                {"value": "Alice", "labels": ["first_name"]},
                {"value": "XZ99", "labels": ["custom_widget_id"]},
            ]
        }
    }
    result = _classify_labels_locally(row)
    assert result[COL_ENTITY_CLASSIFICATION_LOCAL] == {"first_name": "direct_identifier"}
    assert result[COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS] == ["custom_widget_id"]


def test_classify_labels_locally_generator_handles_missing_entities_column() -> None:
    result = _classify_labels_locally({})
    assert result[COL_ENTITY_CLASSIFICATION_LOCAL] == {}
    assert result[COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS] == []


def test_merge_entity_classifications_combines_local_and_llm_results() -> None:
    row = {
        COL_ENTITY_CLASSIFICATION_LOCAL: {"first_name": "direct_identifier", "city": "quasi_identifier"},
        COL_ENTITY_CLASSIFICATION_LLM: {
            "classifications": [{"label": "custom_widget_id", "category": "direct_identifier"}]
        },
    }
    result = _merge_entity_classifications(row)
    assert result[COL_ENTITY_CLASSIFICATION] == {
        "first_name": "direct_identifier",
        "city": "quasi_identifier",
        "custom_widget_id": "direct_identifier",
    }


def test_merge_entity_classifications_handles_empty_llm_result() -> None:
    row = {
        COL_ENTITY_CLASSIFICATION_LOCAL: {"first_name": "direct_identifier"},
        COL_ENTITY_CLASSIFICATION_LLM: {"classifications": []},
    }
    result = _merge_entity_classifications(row)
    assert result[COL_ENTITY_CLASSIFICATION] == {"first_name": "direct_identifier"}


def test_merge_entity_classifications_handles_missing_llm_column() -> None:
    row = {COL_ENTITY_CLASSIFICATION_LOCAL: {"first_name": "direct_identifier"}, COL_ENTITY_CLASSIFICATION_LLM: None}
    result = _merge_entity_classifications(row)
    assert result[COL_ENTITY_CLASSIFICATION] == {"first_name": "direct_identifier"}
