# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed import PartitionBlock, SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.interface.data_designer import DataDesigner
from data_designer.plugins import Plugin

from anonymizer.engine.constants import (
    COL_ADMISSION_RESULTS,
    COL_ADMITTED_CANDIDATES,
    COL_AUGMENTED_CANDIDATES,
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_DETECTED_TAGGED_TEXT,
    COL_FINAL_ENTITIES,
    COL_RECONCILED_CANDIDATES,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATION_DECISIONS,
    DEFAULT_ENTITY_LABELS,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.detection.ignored_entity_values import DEFAULT_IGNORED_ENTITY_VALUES
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import parse_model_configs
from anonymizer.engine.workflow_columns.detection.config import (
    AugmentationConfig,
    ChunkedValidationConfig,
    DetectionTransformConfig,
    DetectionTransformOperation,
)
from anonymizer.engine.workflow_columns.detection.impl import DetectionTransformGenerator
from anonymizer.engine.workflow_columns.detection.plugins import (
    augmentation_plugin,
    chunked_validation_plugin,
    detection_transform_plugin,
)


@pytest.mark.parametrize("plugin", [detection_transform_plugin, chunked_validation_plugin, augmentation_plugin])
def test_detection_plugin_satisfies_data_designer_contract(plugin: Plugin) -> None:
    assert_valid_plugin(plugin)


def test_detection_builder_round_trips_through_native_data_designer_config(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    pd.DataFrame({COL_TEXT: ["Alice", "Bob", "Carol"]}).to_parquet(seed_path, index=False)

    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_builder_for_seed(
        seed_path=seed_path,
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.42,
        validation_max_entities_per_call=7,
        validation_excerpt_window_chars=321,
        entity_labels=["first_name", "email"],
        excluded_entity_labels=["email"],
        entity_label_examples={
            "first_name": ["Alicia"],
            "email": ["configured@example.test"],
        },
        data_summary="Customer support messages",
        job_index=1,
        num_jobs=3,
    )

    direct_augmenter = next(column for column in builder.get_column_configs() if column.name == COL_AUGMENTED_ENTITIES)
    assert isinstance(direct_augmenter, AugmentationConfig)

    payload = builder.get_builder_config().to_json()
    assert payload is not None
    restored = DataDesignerConfigBuilder.from_config(payload)

    assert restored.get_builder_config().to_dict() == builder.get_builder_config().to_dict()

    seed_config = restored.get_seed_config()
    assert seed_config is not None
    assert isinstance(seed_config.source, LocalFileSeedSource)
    assert seed_config.source.path == str(seed_path)
    assert seed_config.sampling_strategy == SamplingStrategy.ORDERED
    assert seed_config.selection_strategy == PartitionBlock(index=1, num_partitions=3)

    columns = restored.get_column_configs()
    assert len(columns) == 11
    assert all(column.column_type != "custom" for column in columns)

    transforms = [column for column in columns if isinstance(column, DetectionTransformConfig)]
    assert {DetectionTransformOperation(column.operation) for column in transforms} == set(DetectionTransformOperation)
    merge_transform = next(
        column
        for column in transforms
        if DetectionTransformOperation(column.operation) == DetectionTransformOperation.MERGE_AND_BUILD_CANDIDATES
    )
    seed_validation_transform = next(
        column
        for column in transforms
        if DetectionTransformOperation(column.operation)
        == DetectionTransformOperation.APPLY_VALIDATION_TO_SEED_ENTITIES
    )
    finalize_transform = next(
        column
        for column in transforms
        if DetectionTransformOperation(column.operation) == DetectionTransformOperation.FINALIZE_ENTITIES
    )
    assert seed_validation_transform.excluded_entity_labels == []
    assert merge_transform.excluded_entity_labels == []
    assert merge_transform.required_columns == [COL_TEXT, COL_ADMITTED_CANDIDATES, COL_AUGMENTED_CANDIDATES]
    assert finalize_transform.excluded_entity_labels == []
    assert finalize_transform.allowed_entity_labels is None
    assert finalize_transform.required_columns == [COL_TEXT, COL_RECONCILED_CANDIDATES]

    validation = next(column for column in columns if column.name == COL_VALIDATION_DECISIONS)
    assert isinstance(validation, ChunkedValidationConfig)
    assert validation.max_entities_per_call == 7
    assert validation.excerpt_window_chars == 321
    admission = next(column for column in columns if column.name == COL_ADMISSION_RESULTS)
    assert isinstance(admission, DetectionTransformConfig)
    assert admission.allowed_entity_labels == validation.allowed_entity_labels == ["first_name"]
    assert admission.excluded_entity_labels == validation.excluded_entity_labels == ["email"]
    assert validation.pool == parsed_models.selected_models.detection.entity_validator
    assert "Customer support messages" in validation.prompt_template
    assert "Alicia" in validation.prompt_template
    assert "configured@example.test" not in validation.prompt_template

    augmenter = next(column for column in columns if column.name == COL_AUGMENTED_ENTITIES)
    assert isinstance(augmenter, AugmentationConfig)
    assert "first_name: Alicia" in augmenter.prompt_template
    assert "Michael, Isabella, Carlos, Wei" not in augmenter.prompt_template
    assert "configured@example.test" not in augmenter.prompt_template

    serialized = json.loads(payload)
    serialized_text = json.dumps(serialized)
    assert "anonymizer-detection-transform" in serialized_text
    assert "anonymizer-chunked-validation" in serialized_text
    assert "anonymizer-augmentation" in serialized_text
    assert "generator_function" not in serialized_text
    assert "generator_params" not in serialized_text


def _get_gliner_labels_from_builder(builder: DataDesignerConfigBuilder) -> list[str]:
    payload = builder.get_builder_config().to_json()
    assert payload is not None
    serialized = json.loads(payload)
    model_configs = serialized["data_designer"]["model_configs"]
    gliner = next(m for m in model_configs if m.get("alias") == "gliner-pii-detector")
    return gliner["inference_parameters"]["extra_body"]["labels"]


def test_build_detection_builder_for_seed_respects_excluded_entity_labels(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    pd.DataFrame({COL_TEXT: ["Alice"]}).to_parquet(seed_path, index=False)

    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_builder_for_seed(
        seed_path=seed_path,
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.3,
        entity_labels=["first_name", "email", "city"],
        excluded_entity_labels=["email"],
    )

    labels = _get_gliner_labels_from_builder(builder)
    assert "email" not in labels
    assert "first_name" in labels
    assert "city" in labels


def test_build_detection_config_respects_excluded_entity_labels(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    input_df = pd.DataFrame({COL_TEXT: ["Alice"]})
    input_df.to_parquet(seed_path, index=False)

    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_config(
        input_df,
        seed_path=seed_path,
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.3,
        entity_labels=["first_name", "email", "city"],
        excluded_entity_labels=["email"],
    )

    augmenter = next(column for column in builder.get_column_configs() if column.name == COL_AUGMENTED_ENTITIES)
    assert isinstance(augmenter, AugmentationConfig)
    labels = _get_gliner_labels_from_builder(builder)
    assert "email" not in labels
    assert "first_name" in labels
    assert "city" in labels


def test_exported_builder_includes_explicit_non_default_example_label(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    pd.DataFrame({COL_TEXT: ["Credential acme_live_abc123"]}).to_parquet(seed_path, index=False)

    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_builder_for_seed(
        seed_path=seed_path,
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.3,
        entity_labels=[*DEFAULT_ENTITY_LABELS, "vendor_api_key"],
        entity_label_examples={"vendor_api_key": ["acme_live_abc123"]},
    )

    assert "vendor_api_key" in _get_gliner_labels_from_builder(builder)
    columns = builder.get_column_configs()
    validation = next(column for column in columns if column.name == COL_VALIDATION_DECISIONS)
    augmenter = next(column for column in columns if column.name == COL_AUGMENTED_ENTITIES)
    assert isinstance(validation, ChunkedValidationConfig)
    assert isinstance(augmenter, AugmentationConfig)
    finalize = next(
        column
        for column in columns
        if isinstance(column, DetectionTransformConfig)
        and column.operation == DetectionTransformOperation.FINALIZE_ENTITIES
    )
    assert "- vendor_api_key: acme_live_abc123" in validation.prompt_template
    assert "vendor_api_key: acme_live_abc123" in augmenter.prompt_template
    assert "Use ONLY labels from this list" in augmenter.prompt_template
    assert finalize.allowed_entity_labels is None
    assert validation.allowed_entity_labels == [*DEFAULT_ENTITY_LABELS, "vendor_api_key"]


def test_fresh_process_discovers_plugins_when_loading_native_config(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    pd.DataFrame({COL_TEXT: ["Alice"]}).to_parquet(seed_path, index=False)

    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_builder_for_seed(
        seed_path=seed_path,
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.3,
    )
    config_path = tmp_path / "detection.json"
    output_path = tmp_path / "restored-columns.json"
    builder.get_builder_config().to_json(config_path)

    script = """
import json
import sys
from pathlib import Path

from data_designer.config.config_builder import DataDesignerConfigBuilder

builder = DataDesignerConfigBuilder.from_config(Path(sys.argv[1]))
columns = [
    {
        "name": column.name,
        "column_type": column.column_type,
        "class_name": type(column).__name__,
    }
    for column in builder.get_column_configs()
]
Path(sys.argv[2]).write_text(json.dumps(columns))
"""
    subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    restored_columns = json.loads(output_path.read_text())
    restored_types = {(column["column_type"], column["class_name"]) for column in restored_columns}
    assert ("anonymizer-detection-transform", "DetectionTransformConfig") in restored_types
    assert ("anonymizer-chunked-validation", "ChunkedValidationConfig") in restored_types
    assert ("anonymizer-augmentation", "AugmentationConfig") in restored_types


@pytest.mark.parametrize("existing_seed", [False, True])
def test_exported_ignore_filter_roundtrips_and_executes(
    tmp_path: Path, existing_seed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_path = tmp_path / "seed.parquet"
    frame = pd.DataFrame({COL_TEXT: ["Example Corp uses Visa"]})
    parsed = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    if existing_seed:
        frame.to_parquet(seed_path, index=False)
        builder = workflow.build_detection_builder_for_seed(
            seed_path=seed_path,
            model_configs=parsed.model_configs,
            selected_models=parsed.selected_models.detection,
            gliner_detection_threshold=0.3,
            ignored_entity_values=["Example Corp"],
        )
    else:
        builder = workflow.build_detection_config(
            frame,
            seed_path=seed_path,
            model_configs=parsed.model_configs,
            selected_models=parsed.selected_models.detection,
            gliner_detection_threshold=0.3,
            ignored_entity_values=["Example Corp"],
        )
    payload = builder.get_builder_config().to_json()
    assert payload is not None
    restored = DataDesignerConfigBuilder.from_config(payload)
    config = next(column for column in restored.get_column_configs() if column.name == COL_FINAL_ENTITIES)
    assert isinstance(config, DetectionTransformConfig)
    assert set(config.ignored_entity_values) == DEFAULT_IGNORED_ENTITY_VALUES | {"Example Corp"}
    assert config.required_columns == [COL_TEXT, COL_DETECTED_ENTITIES]
    # Workers must use the serialized list even if their packaged defaults change.
    monkeypatch.setattr("anonymizer.engine.detection.finalization.DEFAULT_IGNORED_ENTITY_VALUES", frozenset({"Alice"}))
    values = ["Visa", "email", "Example Corp", "EXAMPLE CORP", "credit_card_123", "Alice"]
    raw = {"entities": [{"value": value, "label": "full_name"} for value in values]}
    generator = DetectionTransformGenerator(config, resource_provider=Mock())
    result = generator.generate({COL_TEXT: "example", COL_DETECTED_ENTITIES: raw})
    assert [entity["value"] for entity in result[COL_FINAL_ENTITIES]["entities"]] == values[3:]
    assert result[COL_DETECTED_ENTITIES] == raw
    # A saved list must also remain authoritative when it lacks a worker default.
    saved_config = config.model_copy(update={"ignored_entity_values": ["Example Corp"]})
    saved_generator = DetectionTransformGenerator(saved_config, resource_provider=Mock())
    saved_result = saved_generator.generate({COL_TEXT: "example", COL_DETECTED_ENTITIES: raw})
    assert [entity["value"] for entity in saved_result[COL_FINAL_ENTITIES]["entities"]] == [
        value for value in values if value != "Example Corp"
    ]


@pytest.mark.parametrize("ignore_alice", [False, True])
def test_finalization_renders_tags_from_filtered_entities(ignore_alice: bool) -> None:
    text = "Alice uses Visa and email."
    raw = {
        "entities": [
            {
                "value": value,
                "label": label,
                "start_position": text.index(value),
                "end_position": text.index(value) + len(value),
            }
            for value, label in [("Alice", "first_name"), ("Visa", "credit_debit_card"), ("email", "email")]
        ]
    }
    ignored = ["Visa", "email", *(["Alice"] if ignore_alice else [])]
    config = DetectionTransformConfig(
        name=COL_FINAL_ENTITIES,
        operation=DetectionTransformOperation.MATERIALIZE_FINAL_ENTITIES,
        ignored_entity_values=ignored,
    )
    prefilter_tags = "pre-filter tags retained for debugging"
    row = {COL_TEXT: text, COL_DETECTED_ENTITIES: raw, COL_DETECTED_TAGGED_TEXT: prefilter_tags}
    result = DetectionTransformGenerator(config, resource_provider=Mock()).generate(row)
    assert result[COL_TAGGED_TEXT] == (text if ignore_alice else "<first_name>Alice</first_name> uses Visa and email.")
    assert result[COL_DETECTED_TAGGED_TEXT] == prefilter_tags
    assert result[COL_DETECTED_ENTITIES] == raw
    assert config.side_effect_columns == [COL_TAGGED_TEXT]


def test_finalization_generator_applies_allowed_and_excluded_labels() -> None:
    text = "Alice uses alice@example.com in Boston."
    raw = {
        "entities": [
            {
                "value": value,
                "label": label,
                "start_position": text.index(value),
                "end_position": text.index(value) + len(value),
            }
            for value, label in [("Alice", "first_name"), ("alice@example.com", "email"), ("Boston", "city")]
        ]
    }
    config = DetectionTransformConfig(
        name=COL_FINAL_ENTITIES,
        operation=DetectionTransformOperation.MATERIALIZE_FINAL_ENTITIES,
        allowed_entity_labels=["first_name", "email"],
        excluded_entity_labels=["email"],
    )
    result = DetectionTransformGenerator(config, resource_provider=Mock()).generate(
        {COL_TEXT: text, COL_DETECTED_ENTITIES: raw}
    )
    assert [(entity["value"], entity["label"]) for entity in result[COL_FINAL_ENTITIES]["entities"]] == [
        ("Alice", "first_name")
    ]
    assert result[COL_TAGGED_TEXT] == "<first_name>Alice</first_name> uses alice@example.com in Boston."
    assert result[COL_DETECTED_ENTITIES] == raw


@pytest.mark.parametrize(
    "malformed", ["{not json", {"entities": [{"value": "SecretName", "start_position": "bad"}]}, 42]
)
def test_finalization_generator_rejects_malformed_payloads(malformed: object) -> None:
    config = DetectionTransformConfig(
        name=COL_FINAL_ENTITIES,
        operation=DetectionTransformOperation.MATERIALIZE_FINAL_ENTITIES,
        ignored_entity_values=["SecretName"],
    )
    row = {COL_TEXT: "SecretName", COL_DETECTED_ENTITIES: malformed}
    with pytest.raises(ValueError) as exc:
        DetectionTransformGenerator(config, resource_provider=Mock()).generate(row)
    assert "SecretName" not in str(exc.value)
    assert COL_FINAL_ENTITIES not in row
    assert COL_TAGGED_TEXT not in row
