# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionResponse
from data_designer.engine.models.errors import ModelGenerationValidationFailureError
from data_designer.engine.models.facade import ModelFacade
from data_designer.interface.data_designer import DataDesigner

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
from anonymizer.engine.rewrite.entity_classification import (
    EntityClassificationWorkflow,
    _classify_labels_locally,
    _get_entity_classification_prompt,
    _make_classification_column,
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
    assert isinstance(cols[1], CustomColumnConfig)
    assert isinstance(cols[2], CustomColumnConfig)
    assert cols[0].name == COL_ENTITY_CLASSIFICATION_LOCAL
    assert cols[1].name == COL_ENTITY_CLASSIFICATION_LLM
    assert cols[2].name == COL_ENTITY_CLASSIFICATION


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
        COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS: ["custom_widget_id"],
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


def test_merge_entity_classifications_ignores_unrequested_and_default_labels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    row = {
        COL_ENTITY_CLASSIFICATION_LOCAL: {"email": "direct_identifier"},
        COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS: ["custom_widget_id"],
        COL_ENTITY_CLASSIFICATION_LLM: {
            "classifications": [
                {"label": "email", "category": "quasi_identifier"},
                {"label": "city", "category": "direct_identifier"},
                {"label": "invented_label", "category": "direct_identifier"},
                {"label": "custom_widget_id", "category": "direct_identifier"},
            ]
        },
    }
    result = _merge_entity_classifications(row)
    assert result[COL_ENTITY_CLASSIFICATION] == {
        "email": "direct_identifier",
        "custom_widget_id": "direct_identifier",
    }
    for label in ("email", "city", "invented_label"):
        assert f"Ignoring classification for unrequested or locally resolved label {label!r}" in caplog.text


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


@custom_column_generator(required_columns=[COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS])
def _classify_custom_labels(row: dict[str, Any]) -> dict[str, Any]:
    labels = row[COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS]
    if not labels:
        raise ValueError("Classifier must not run without custom labels")
    row[COL_ENTITY_CLASSIFICATION_LLM] = {
        "classifications": [{"label": label, "category": "direct_identifier"} for label in labels]
    }
    return row


def test_classification_skips_only_rows_without_custom_labels(
    tmp_path: Path, stub_rewrite_model_selection: RewriteModelSelection
) -> None:
    columns = EntityClassificationWorkflow().columns(selected_models=stub_rewrite_model_selection)
    classifier = columns[1]
    columns[1] = CustomColumnConfig(
        name=classifier.name,
        generator_function=_classify_custom_labels,
        skip=classifier.skip,
    )
    adapter = NddAdapter(DataDesigner(artifact_path=tmp_path / "artifacts", auto_configure_logging=False))
    result = adapter.run_workflow(
        pd.DataFrame(
            {
                COL_ENTITIES_BY_VALUE: [
                    {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
                    {
                        "entities_by_value": [
                            {"value": "Alice", "labels": ["first_name"]},
                            {"value": "XZ99", "labels": ["custom_widget_id"]},
                        ]
                    },
                    {"entities_by_value": []},
                ]
            }
        ),
        model_configs=[],
        columns=columns,
        workflow_name="entity-classification",
        preview_num_records=3,
    )
    assert result.failed_records == []
    assert result.dataframe[COL_ENTITY_CLASSIFICATION].tolist() == [
        {"first_name": "direct_identifier"},
        {"first_name": "direct_identifier", "custom_widget_id": "direct_identifier"},
        {},
    ]
    seed_path = tmp_path / "evaluation-seed.parquet"
    result.dataframe.to_parquet(seed_path, index=False)
    restored = pd.read_parquet(seed_path)
    classifications = restored[COL_ENTITY_CLASSIFICATION_LLM]
    assert pd.isna(classifications.iloc[0])
    assert classifications.iloc[1]["classifications"][0] == {
        "label": "custom_widget_id",
        "category": "direct_identifier",
    }
    assert pd.isna(classifications.iloc[2])


@pytest.mark.parametrize(
    "categories", [[], ["direct_identifier", "quasi_identifier"], ["quasi_identifier", "direct_identifier"]]
)
@pytest.mark.parametrize("recover", [True, False])
def test_classification_corrects_once_then_succeeds_or_fails(categories: list[str], recover: bool) -> None:
    invalid = {"classifications": [{"label": "custom_widget_id", "category": category} for category in categories]}
    valid = {"classifications": [{"label": "custom_widget_id", "category": "direct_identifier"}]}
    model = Mock(spec=ModelFacade)
    model.model_name = "test"
    model.model_provider_name = "test"
    model._get_mcp_facade.return_value = None
    model.completion.side_effect = [
        ChatCompletionResponse(message=AssistantMessage(content=json.dumps(payload)))
        for payload in (invalid, valid if recover else invalid)
    ]
    model.generate = partial(ModelFacade.generate, model)
    generate = _make_classification_column("classifier")
    row = {COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS: ["custom_widget_id"]}
    if recover:
        result = generate(row, generator_params=None, models={"classifier": model})
        assert result[COL_ENTITY_CLASSIFICATION_LLM] == valid
    else:
        with pytest.raises(ModelGenerationValidationFailureError, match="custom_widget_id"):
            generate(row, generator_params=None, models={"classifier": model})
    assert model.completion.call_count == 2
    messages = model.completion.call_args.args[0]
    assert "custom_widget_id" in str(messages[2])
    assert "complete classifications list" in str(messages[2])


@pytest.mark.parametrize(
    "categories", [[], ["direct_identifier", "quasi_identifier"], ["quasi_identifier", "direct_identifier"]]
)
def test_merge_rejects_missing_or_conflicting_custom_labels(categories: list[str]) -> None:
    row = {
        COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS: ["custom_widget_id"],
        COL_ENTITY_CLASSIFICATION_LLM: {
            "classifications": [{"label": "custom_widget_id", "category": category} for category in categories]
        },
    }
    with pytest.raises(ValueError, match="custom_widget_id"):
        _merge_entity_classifications(row)
    assert COL_ENTITY_CLASSIFICATION not in row


def test_merge_accepts_identical_duplicates() -> None:
    result = _merge_entity_classifications(
        {
            COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS: ["custom_widget_id"],
            COL_ENTITY_CLASSIFICATION_LLM: {
                "classifications": [
                    {"label": "custom_widget_id", "category": "direct_identifier"},
                    {"label": "custom_widget_id", "category": "direct_identifier"},
                ]
            },
        }
    )
    assert result[COL_ENTITY_CLASSIFICATION] == {"custom_widget_id": "direct_identifier"}
