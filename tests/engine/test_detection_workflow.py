# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.models import ModelConfig

from anonymizer.config.models import DetectionModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_LATENT_ENTITIES,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionWorkflow,
    _format_label_examples,
    _get_augment_prompt,
    _get_latent_prompt,
    _get_validation_prompt,
    _merge_labels,
)
from anonymizer.engine.ndd.adapter import FailedRecord, WorkflowRunResult
from anonymizer.engine.ndd.model_loader import load_default_model_selection, resolve_model_alias


def test_run_with_latent_detection_calls_second_workflow(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.side_effect = [
        WorkflowRunResult(
            dataframe=pd.DataFrame(
                {
                    COL_TEXT: ["Alice works in Seattle"],
                    COL_TAGGED_TEXT: ["<first_name>Alice</first_name> works in <city>Seattle</city>"],
                    COL_DETECTED_ENTITIES: [[{"value": "Alice", "label": "first_name"}]],
                }
            ),
            failed_records=[],
        ),
        WorkflowRunResult(
            dataframe=pd.DataFrame(
                {
                    COL_TEXT: ["Alice works in Seattle"],
                    COL_TAGGED_TEXT: ["<first_name>Alice</first_name> works in <city>Seattle</city>"],
                    COL_DETECTED_ENTITIES: [[{"value": "Alice", "label": "first_name"}]],
                    COL_LATENT_ENTITIES: [[{"value": "Acme Corp", "label": "organization", "sensitivity": "medium"}]],
                }
            ),
            failed_records=[FailedRecord(record_id="1", step="latent-entity-detection", reason="none")],
        ),
    ]

    workflow = EntityDetectionWorkflow(adapter=adapter)
    input_df = pd.DataFrame({COL_TEXT: ["Alice works in Seattle"]})

    result = workflow.run(
        input_df,
        model_configs=stub_detector_model_configs,
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.5,
        tag_latent_entities=True,
        privacy_goal=PrivacyGoal(
            protect="Protect direct and latent identifiers from disclosure.",
            preserve="General utility and semantic meaning of the original text.",
        ),
        data_summary="Employee records",
    )

    assert adapter.run_workflow.call_count == 2
    second_columns = adapter.run_workflow.call_args_list[1].kwargs["columns"]
    assert len(second_columns) == 1
    assert isinstance(second_columns[0], LLMStructuredColumnConfig)
    assert second_columns[0].name == COL_LATENT_ENTITIES
    assert COL_LATENT_ENTITIES in result.dataframe.columns
    assert COL_FINAL_ENTITIES in result.dataframe.columns
    assert len(result.failed_records) == 1


def test_latent_prompt_includes_summary_and_goal() -> None:
    prompt = _get_latent_prompt(
        data_summary="Medical visit notes",
        privacy_goal=PrivacyGoal(
            protect="Protect direct and inferred identities from re-identification.",
            preserve="Clinical utility and semantic meaning of the original text.",
        ),
    )
    assert "Data type summary:\nMedical visit notes" in prompt
    assert "The text will be rewritten according to this privacy goal:" in prompt
    assert "PROTECT: Protect direct and inferred identities from re-identification." in prompt
    assert "PRESERVE: Clinical utility and semantic meaning of the original text." in prompt
    assert "Every latent entity MUST include 1-2 short quotes from the text as evidence." in prompt
    assert COL_TAGGED_TEXT in prompt


def test_run_without_latent_detection_materializes_final_entities(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works in Seattle"],
                COL_DETECTED_ENTITIES: [[{"value": "Alice", "label": "first_name"}]],
            }
        ),
        failed_records=[],
    )
    workflow = EntityDetectionWorkflow(adapter=adapter)

    result = workflow.run(
        pd.DataFrame({COL_TEXT: ["Alice works in Seattle"]}),
        model_configs=stub_detector_model_configs,
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.5,
        tag_latent_entities=False,
        privacy_goal=None,
    )

    assert adapter.run_workflow.call_count == 1
    assert COL_FINAL_ENTITIES in result.dataframe.columns
    assert result.dataframe[COL_FINAL_ENTITIES].tolist() == result.dataframe[COL_DETECTED_ENTITIES].tolist()


def test_run_compute_grouped_entities_false_drops_grouped_column(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works in Seattle"],
                COL_DETECTED_ENTITIES: [[{"value": "Alice", "label": "first_name"}]],
                COL_ENTITIES_BY_VALUE: [[{"value": "Alice", "labels": ["first_name"]}]],
            }
        ),
        failed_records=[],
    )
    workflow = EntityDetectionWorkflow(adapter=adapter)
    result = workflow.run(
        pd.DataFrame({COL_TEXT: ["Alice works in Seattle"]}),
        model_configs=stub_detector_model_configs,
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.5,
        tag_latent_entities=False,
        privacy_goal=None,
        compute_grouped_entities=False,
    )
    assert COL_ENTITIES_BY_VALUE not in result.dataframe.columns


def test_run_preserves_original_text_column_attr(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works in Seattle"],
                COL_DETECTED_ENTITIES: [[{"value": "Alice", "label": "first_name"}]],
            }
        ),
        failed_records=[],
    )
    workflow = EntityDetectionWorkflow(adapter=adapter)
    input_df = pd.DataFrame({COL_TEXT: ["Alice works in Seattle"]})
    input_df.attrs["original_text_column"] = "content"
    result = workflow.run(
        input_df,
        model_configs=stub_detector_model_configs,
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.5,
        tag_latent_entities=False,
        privacy_goal=None,
    )
    assert result.dataframe.attrs["original_text_column"] == "content"


def test_run_with_latent_detection_merges_failures_in_order(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.side_effect = [
        WorkflowRunResult(
            dataframe=pd.DataFrame(
                {
                    COL_TEXT: ["Alice works in Seattle"],
                    COL_DETECTED_ENTITIES: [[{"value": "Alice", "label": "first_name"}]],
                }
            ),
            failed_records=[FailedRecord(record_id="d1", step="entity-detection", reason="detected failure")],
        ),
        WorkflowRunResult(
            dataframe=pd.DataFrame(
                {
                    COL_TEXT: ["Alice works in Seattle"],
                    COL_DETECTED_ENTITIES: [[{"value": "Alice", "label": "first_name"}]],
                    COL_LATENT_ENTITIES: [[{"value": "Acme Corp", "label": "organization", "sensitivity": "medium"}]],
                }
            ),
            failed_records=[FailedRecord(record_id="l1", step="latent-entity-detection", reason="latent failure")],
        ),
    ]
    workflow = EntityDetectionWorkflow(adapter=adapter)
    result = workflow.run(
        pd.DataFrame({COL_TEXT: ["Alice works in Seattle"]}),
        model_configs=stub_detector_model_configs,
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.5,
        tag_latent_entities=True,
        privacy_goal=PrivacyGoal(
            protect="Protect direct and latent identifiers from disclosure.",
            preserve="General utility and semantic meaning of the original text.",
        ),
    )
    assert [item.record_id for item in result.failed_records] == ["d1", "l1"]


def test_detect_and_validate_entities_drops_grouped_when_compute_grouped_false(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works in Seattle"],
                COL_DETECTED_ENTITIES: [[{"value": "Alice", "label": "first_name"}]],
                COL_ENTITIES_BY_VALUE: [[{"value": "Alice", "labels": ["first_name"]}]],
            }
        ),
        failed_records=[],
    )
    workflow = EntityDetectionWorkflow(adapter=adapter)
    result = workflow.detect_and_validate_entities(
        pd.DataFrame({COL_TEXT: ["Alice works in Seattle"]}),
        model_configs=stub_detector_model_configs,
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.5,
        compute_grouped=False,
    )
    assert COL_ENTITIES_BY_VALUE not in result.dataframe.columns


def test_run_requires_privacy_goal_for_latent_path(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    workflow = EntityDetectionWorkflow(adapter=Mock())
    with pytest.raises(ValueError, match="privacy_goal is required"):
        workflow.run(
            pd.DataFrame({COL_TEXT: ["Alice"]}),
            model_configs=stub_detector_model_configs,
            selected_models=stub_detection_model_selection,
            gliner_detection_threshold=0.5,
            tag_latent_entities=True,
            privacy_goal=None,
        )


def test_inject_detector_params_does_not_mutate_input_configs(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    workflow = EntityDetectionWorkflow(adapter=Mock())

    assert stub_detector_model_configs[0].inference_parameters.extra_body is None

    updated = workflow._inject_detector_params(
        model_configs=stub_detector_model_configs,
        selected_models=stub_detection_model_selection,
        labels=["email", "phone_number"],
        gliner_detection_threshold=0.42,
    )

    assert stub_detector_model_configs[0].inference_parameters.extra_body is None
    assert updated[0].inference_parameters.extra_body is not None
    assert updated[0].inference_parameters.extra_body["labels"] == ["email", "phone_number"]
    assert updated[0].inference_parameters.extra_body["threshold"] == 0.42
    assert updated[0].inference_parameters.extra_body["chunk_length"] == 384
    assert updated[0].inference_parameters.extra_body["overlap"] == 128
    assert updated[0].inference_parameters.extra_body["flat_ner"] is False


def test_inject_detector_params_no_matching_alias_leaves_configs_unchanged(
    stub_detector_model_configs: list[ModelConfig],
) -> None:
    workflow = EntityDetectionWorkflow(adapter=Mock())
    defaults = load_default_model_selection().detection
    selected_models = defaults.model_copy(update={"entity_detector": "missing-detector"})
    updated = workflow._inject_detector_params(
        model_configs=stub_detector_model_configs,
        selected_models=selected_models,
        labels=["email"],
        gliner_detection_threshold=0.42,
    )
    assert all(config.inference_parameters.extra_body is None for config in updated)


def test_resolve_model_alias_reads_from_selection_model() -> None:
    defaults = load_default_model_selection().detection
    selection = defaults.model_copy(update={"entity_detector": "custom-model"})
    assert resolve_model_alias("entity_detector", selection) == "custom-model"
    assert resolve_model_alias("entity_validator", selection) == defaults.entity_validator


def test_merge_labels_normalizes_and_deduplicates_custom_labels() -> None:
    merged = _merge_labels(["  custom_label  ", "CUSTOM_LABEL", "custom_label"])
    assert "custom_label" in merged
    assert merged.count("custom_label") == 1


def test_latent_prompt_uses_not_provided_defaults() -> None:
    prompt = _get_latent_prompt(data_summary=None, privacy_goal=None)
    assert "Data type summary:\nNot provided" in prompt
    assert "The text will be rewritten according to this privacy goal: Not provided" in prompt


def test_format_label_examples_includes_known_labels() -> None:
    result = _format_label_examples(["first_name", "city", "ssn"])
    assert "- first_name: Michael, Isabella, Carlos, Wei" in result
    assert "- city: Houston, San Diego, Doha, Lahore" in result
    assert "- ssn: 007-52-4910, 252-96-0016, 523-25-1554, 228-94-9430" in result


def test_format_label_examples_handles_custom_labels_without_examples() -> None:
    result = _format_label_examples(["first_name", "custom_label"])
    assert "- first_name: Michael, Isabella, Carlos, Wei" in result
    assert "- custom_label" in result
    assert "- custom_label:" not in result


def test_validation_prompt_includes_label_examples() -> None:
    prompt = _get_validation_prompt(data_summary=None, labels=["email", "city"])
    assert "Here are all the valid entity classes with examples" in prompt
    assert "- email: derez_lester94@icloud.com" in prompt
    assert "- city: Houston, San Diego, Doha, Lahore" in prompt


def test_validation_prompt_includes_data_summary() -> None:
    prompt = _get_validation_prompt(data_summary="Medical records", labels=["first_name"])
    assert "Data context: Medical records" in prompt


def test_augment_prompt_includes_label_names() -> None:
    prompt = _get_augment_prompt(data_summary=None, labels=["phone_number", "age"])
    assert "Here are the known entity classes" in prompt
    assert "phone_number, age" in prompt
    assert "If no known label fits, create a concise snake_case label" in prompt
