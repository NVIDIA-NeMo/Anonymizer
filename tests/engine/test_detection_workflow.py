# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.models import ModelConfig

from anonymizer.config.models import DetectionModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_LATENT_ENTITIES,
    COL_SEED_ENTITIES,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATION_DECISIONS,
    DEFAULT_ENTITY_LABELS,
)
from anonymizer.engine.detection.chunked_validation import ChunkedValidationParams
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionWorkflow,
    _format_label_examples,
    _get_augment_prompt,
    _get_latent_prompt,
    _get_validation_prompt,
    _pad_empty_latent_column,
    _resolve_detection_labels,
)
from anonymizer.engine.ndd.adapter import FailedRecord, WorkflowRunResult
from anonymizer.engine.ndd.model_loader import (
    load_default_model_selection,
    resolve_model_alias,
    resolve_model_aliases,
)
from anonymizer.engine.schemas import EntitiesSchema


@pytest.fixture
def _detection_with_novel_augmented_label(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> tuple[EntityDetectionWorkflow, pd.DataFrame, list[ModelConfig], DetectionModelSelection]:
    """Workflow whose detector returns an entity with a label (server_name) outside the user's list."""
    input_df = pd.DataFrame({COL_TEXT: ["Connect to srv01.internal on 10.0.0.5"]})
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Connect to srv01.internal on 10.0.0.5"],
                COL_DETECTED_ENTITIES: [
                    {
                        "entities": [
                            {"value": "srv01.internal", "label": "hostname", "start_position": 11, "end_position": 25},
                            {"value": "10.0.0.5", "label": "ipv4", "start_position": 29, "end_position": 37},
                            {"value": "srv01", "label": "server_name", "start_position": 11, "end_position": 16},
                        ]
                    }
                ],
            }
        ),
        failed_records=[],
    )
    return (
        EntityDetectionWorkflow(adapter=adapter),
        input_df,
        stub_detector_model_configs,
        stub_detection_model_selection,
    )


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
                    COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
                }
            ),
            failed_records=[],
        ),
        WorkflowRunResult(
            dataframe=pd.DataFrame(
                {
                    COL_TEXT: ["Alice works in Seattle"],
                    COL_TAGGED_TEXT: ["<first_name>Alice</first_name> works in <city>Seattle</city>"],
                    COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
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
                COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
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
    final = result.dataframe[COL_FINAL_ENTITIES].iloc[0]
    assert isinstance(final, dict)
    assert len(final["entities"]) == 1
    assert final["entities"][0]["value"] == "Alice"
    assert final["entities"][0]["label"] == "first_name"


def test_run_compute_grouped_entities_false_drops_grouped_column(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works in Seattle"],
                COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
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
                COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
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
                    COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
                }
            ),
            failed_records=[FailedRecord(record_id="d1", step="entity-detection", reason="detected failure")],
        ),
        WorkflowRunResult(
            dataframe=pd.DataFrame(
                {
                    COL_TEXT: ["Alice works in Seattle"],
                    COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
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
    assert resolve_model_aliases("entity_validator", selection) == defaults.entity_validator


def test_resolve_model_alias_raises_for_list_valued_role() -> None:
    selection = load_default_model_selection().detection
    with pytest.raises(TypeError, match="list-valued"):
        resolve_model_alias("entity_validator", selection)


def test_resolve_model_aliases_wraps_scalar_roles() -> None:
    selection = load_default_model_selection().detection
    assert resolve_model_aliases("entity_detector", selection) == [selection.entity_detector]


def test_resolve_detection_labels_none_uses_defaults() -> None:
    merged = _resolve_detection_labels(None)
    assert merged == DEFAULT_ENTITY_LABELS


def test_resolve_detection_labels_does_not_append_defaults_when_custom_labels_provided() -> None:
    merged = _resolve_detection_labels(["custom_label"])
    assert merged == ["custom_label"]


def test_resolve_detection_labels_preserves_provided_labels_as_is() -> None:
    # cleaning of whitespace and case normalization occur during config validation
    labels = ["FIRST_NAME", " email "]
    assert _resolve_detection_labels(labels) == labels


def test_latent_prompt_uses_not_provided_defaults() -> None:
    prompt = _get_latent_prompt(data_summary=None, privacy_goal=None)
    assert "Data type summary:\nNot provided" in prompt
    assert "The text will be rewritten according to this privacy goal: Not provided" in prompt


def test_format_label_examples_includes_known_labels() -> None:
    result = _format_label_examples(["first_name", "city", "ssn", "race_ethnicity"])
    assert "- first_name: Michael, Isabella, Carlos, Wei" in result
    assert "- city: Houston, San Diego, Doha, Lahore" in result
    assert "- ssn: 007-52-4910, 252-96-0016, 523-25-1554, 228-94-9430" in result
    assert "- race_ethnicity: white, African-American, Korean, Hispanic" in result


def test_format_label_examples_handles_custom_labels_without_examples() -> None:
    result = _format_label_examples(["first_name", "custom_label"])
    assert "- first_name: Michael, Isabella, Carlos, Wei" in result
    assert "- custom_label" in result
    assert "- custom_label:" not in result


def test_validation_prompt_includes_label_examples() -> None:
    prompt = _get_validation_prompt(data_summary=None, labels=["email", "city", "sexuality", "age", "first_name"])
    assert "Here are all the valid entity classes with examples" in prompt
    assert "- email: derez_lester94@icloud.com" in prompt
    assert "- city: Houston, San Diego, Doha, Lahore" in prompt
    assert "Copy ids exactly as given; never modify entries" in prompt
    assert "You MUST fill in a decision for EVERY entry in the template" in prompt
    assert "Return ONLY the entries from the template" in prompt
    assert 'The word "straight" rarely has the label "sexuality"' in prompt
    assert "PARTIAL-TOKEN RULE (HARD DROP):" in prompt
    assert '"((SENSITIVE:political_view|dem))eanor" → drop, because "dem" is inside "demeanor"' in prompt
    assert 'The entity label "occupation" refers only to a specific paid job title or profession' in prompt
    assert "AGE RULE:" in prompt
    assert "indicate duration, not age" in prompt
    assert "tagged as a first_name but is not followed by a last_name, drop it" in prompt


def test_validation_prompt_includes_data_summary() -> None:
    prompt = _get_validation_prompt(data_summary="Medical records", labels=["first_name"])
    assert "Data context: Medical records" in prompt


def test_augment_prompt_permissive_when_using_defaults() -> None:
    """In practice strict_labels=False only fires with DEFAULT_ENTITY_LABELS (entity_labels=None).
    We pass a small list here to verify the permissive prompt text in isolation."""
    prompt = _get_augment_prompt(data_summary=None, labels=["phone_number", "age"], strict_labels=False)
    assert "Strongly prefer labels from this list when they fit" in prompt
    assert "phone_number, age" in prompt
    assert "If no known label fits, create a concise snake_case label" in prompt
    assert "employment_status" in prompt


def test_augment_prompt_strict_when_custom_labels_provided() -> None:
    prompt = _get_augment_prompt(data_summary=None, labels=["hostname", "ipv4"], strict_labels=True)
    assert "Use ONLY labels from this list" in prompt
    assert "hostname, ipv4" in prompt
    assert "Do not create new labels" in prompt
    assert "Strongly prefer" not in prompt
    assert "create a concise snake_case label" not in prompt
    assert "employment_status is NOT in the allowed list" in prompt
    assert "employment_status" not in prompt.split("Output:")[1]


def test_custom_entity_labels_filters_out_of_scope_augmented_entities(
    _detection_with_novel_augmented_label: tuple[
        EntityDetectionWorkflow, pd.DataFrame, list[ModelConfig], DetectionModelSelection
    ],
) -> None:
    """Augmented entities with labels outside entity_labels must be stripped from final_entities."""
    workflow, input_df, model_configs, selected_models = _detection_with_novel_augmented_label
    result = workflow.run(
        input_df,
        model_configs=model_configs,
        selected_models=selected_models,
        gliner_detection_threshold=0.5,
        entity_labels=["hostname", "ipv4"],
        tag_latent_entities=False,
    )

    final = EntitiesSchema.from_raw(result.dataframe[COL_FINAL_ENTITIES].iloc[0])
    final_labels = {e.label for e in final.entities}
    assert final_labels == {"hostname", "ipv4"}
    assert "server_name" not in final_labels

    ebv = result.dataframe[COL_ENTITIES_BY_VALUE].iloc[0]
    ebv_values = {e["value"] for e in ebv["entities_by_value"]}
    assert "srv01" not in ebv_values

    detected = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0])
    assert "server_name" in {e.label for e in detected.entities}


def test_default_entity_labels_preserves_novel_augmented_entities(
    _detection_with_novel_augmented_label: tuple[
        EntityDetectionWorkflow, pd.DataFrame, list[ModelConfig], DetectionModelSelection
    ],
) -> None:
    """When entity_labels=None, augmented entities with novel labels must be preserved."""
    workflow, input_df, model_configs, selected_models = _detection_with_novel_augmented_label
    result = workflow.run(
        input_df,
        model_configs=model_configs,
        selected_models=selected_models,
        gliner_detection_threshold=0.5,
        tag_latent_entities=False,
    )

    final = EntitiesSchema.from_raw(result.dataframe[COL_FINAL_ENTITIES].iloc[0])
    final_labels = {e.label for e in final.entities}
    assert "server_name" in final_labels
    assert "hostname" in final_labels
    assert "ipv4" in final_labels


# ---------------------------------------------------------------------------
# _pad_empty_latent_column — pyarrow struct-inference workaround
# ---------------------------------------------------------------------------


def test_pad_empty_latent_column_injects_sentinel_into_empty_dict_cell() -> None:
    """When DD's latent workflow returns ``{"latent_entities": []}`` for
    every row, pyarrow can't infer the nested struct schema and the
    parquet write fails. _pad_empty_latent_column injects a default
    LatentEntitySchema so pyarrow sees at least one populated nested
    record."""
    df = pd.DataFrame({COL_LATENT_ENTITIES: [{"latent_entities": []}, {"latent_entities": []}]})
    out = _pad_empty_latent_column(df)
    assert out is not df  # returns a copy
    for cell in out[COL_LATENT_ENTITIES]:
        assert cell["latent_entities"], "expected sentinel injected into empty cell"
        assert isinstance(cell["latent_entities"], list)
        assert isinstance(cell["latent_entities"][0], dict)


def test_pad_empty_latent_column_preserves_populated_cells() -> None:
    """Cells that already have entries are left alone."""
    populated = {"latent_entities": [{"label": "employer", "value": "UW"}]}
    df = pd.DataFrame({COL_LATENT_ENTITIES: [populated]})
    out = _pad_empty_latent_column(df)
    assert out[COL_LATENT_ENTITIES].iloc[0] == populated


def test_pad_empty_latent_column_handles_bare_empty_list_cells() -> None:
    """Some adapter paths emit a bare ``[]`` instead of the wrapped dict.
    The padder swaps it for the sentinel list as well."""
    df = pd.DataFrame({COL_LATENT_ENTITIES: [[], []]})
    out = _pad_empty_latent_column(df)
    for cell in out[COL_LATENT_ENTITIES]:
        assert isinstance(cell, list) and cell, "empty list should be padded"


def test_pad_empty_latent_column_no_op_when_column_missing() -> None:
    """Defensive: returns the df unchanged if the column is absent."""
    df = pd.DataFrame({COL_TEXT: ["hello"]})
    out = _pad_empty_latent_column(df)
    assert COL_LATENT_ENTITIES not in out.columns
    assert out is df  # unchanged identity


# ---------------------------------------------------------------------------
# Chunked validation wiring (Commit 2)
# ---------------------------------------------------------------------------


def _find_column(columns: list, name: str):
    for col in columns:
        if getattr(col, "name", None) == name:
            return col
    raise AssertionError(f"Column {name!r} not found in workflow columns: {[getattr(c, 'name', c) for c in columns]}")


def test_validation_column_is_custom_chunked_generator(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    """COL_VALIDATION_DECISIONS is now a CustomColumnConfig bound to the chunked generator,
    not an LLMStructuredColumnConfig."""
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice"],
                COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
            }
        ),
        failed_records=[],
    )
    workflow = EntityDetectionWorkflow(adapter=adapter)
    workflow.run(
        pd.DataFrame({COL_TEXT: ["Alice"]}),
        model_configs=stub_detector_model_configs,
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.5,
        tag_latent_entities=False,
    )
    columns = adapter.run_workflow.call_args.kwargs["columns"]
    validation_col = _find_column(columns, COL_VALIDATION_DECISIONS)
    assert isinstance(validation_col, CustomColumnConfig)
    # Must NOT be the old structured-output LLM column.
    assert not isinstance(validation_col, LLMStructuredColumnConfig)
    assert validation_col.drop is True
    # generator_params must match the Detect config defaults that flow through.
    assert isinstance(validation_col.generator_params, ChunkedValidationParams)
    assert validation_col.generator_params.pool == stub_detection_model_selection.entity_validator
    assert validation_col.generator_params.max_entities_per_call > 0
    assert validation_col.generator_params.excerpt_window_chars > 0
    # The decorated generator's metadata must expose the pool and the exact
    # set of columns it reads, so DataDesigner resolves facades and DAG ordering.
    metadata = validation_col.generator_function.custom_column_metadata
    assert metadata["model_aliases"] == list(stub_detection_model_selection.entity_validator)
    assert set(metadata["required_columns"]) == {
        COL_TEXT,
        COL_SEED_ENTITIES,
        COL_SEED_VALIDATION_CANDIDATES,
        COL_TAG_NOTATION,
    }


def test_validator_pool_kwargs_thread_through_to_generator_params(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    """Explicit ``validation_max_entities_per_call`` and ``validation_excerpt_window_chars``
    propagate from ``run()`` all the way to ``ChunkedValidationParams``."""
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice"],
                COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
            }
        ),
        failed_records=[],
    )
    workflow = EntityDetectionWorkflow(adapter=adapter)
    workflow.run(
        pd.DataFrame({COL_TEXT: ["Alice"]}),
        model_configs=stub_detector_model_configs,
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.5,
        validation_max_entities_per_call=17,
        validation_excerpt_window_chars=42,
        tag_latent_entities=False,
    )
    columns = adapter.run_workflow.call_args.kwargs["columns"]
    params = _find_column(columns, COL_VALIDATION_DECISIONS).generator_params
    assert params.max_entities_per_call == 17
    assert params.excerpt_window_chars == 42


def test_pool_size_greater_than_one_emits_warning(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operators with multiple validator aliases must be alerted that
    ``max_parallel_requests`` is enforced per alias (pool multiplies in-flight)."""
    selection = stub_detection_model_selection.model_copy(
        update={"entity_validator": [*stub_detection_model_selection.entity_validator, "extra-validator"]}
    )
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice"],
                COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
            }
        ),
        failed_records=[],
    )
    workflow = EntityDetectionWorkflow(adapter=adapter)
    with caplog.at_level("WARNING"):
        workflow.run(
            pd.DataFrame({COL_TEXT: ["Alice"]}),
            model_configs=stub_detector_model_configs,
            selected_models=selection,
            gliner_detection_threshold=0.5,
            tag_latent_entities=False,
        )
    # caplog can attach handlers at both the target logger and root, so the
    # same record may appear twice in ``records``; dedupe by identity.
    pool_warnings = {
        id(r): r
        for r in caplog.records
        if r.name == "anonymizer.detection" and "pool of" in r.getMessage() and "aliases" in r.getMessage()
    }
    assert len(pool_warnings) == 1
    (only,) = pool_warnings.values()
    assert "multiplies total in-flight" in only.getMessage()


def test_pool_size_one_does_not_emit_warning(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Default single-alias configurations must not spam the warning: it's a pool caveat, not advice for everyone."""
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice"],
                COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
            }
        ),
        failed_records=[],
    )
    assert len(stub_detection_model_selection.entity_validator) == 1, (
        "baseline default must be a single validator for this test to be meaningful"
    )
    workflow = EntityDetectionWorkflow(adapter=adapter)
    with caplog.at_level("WARNING"):
        workflow.run(
            pd.DataFrame({COL_TEXT: ["Alice"]}),
            model_configs=stub_detector_model_configs,
            selected_models=stub_detection_model_selection,
            gliner_detection_threshold=0.5,
            tag_latent_entities=False,
        )
    pool_warnings = [
        r
        for r in caplog.records
        if r.name == "anonymizer.detection" and "pool of" in r.getMessage() and "aliases" in r.getMessage()
    ]
    assert pool_warnings == []
