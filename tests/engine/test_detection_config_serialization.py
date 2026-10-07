# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.seed import PartitionBlock, SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.engine.testing.utils import assert_valid_plugin
from data_designer.interface.data_designer import DataDesigner
from data_designer.plugins import Plugin

from anonymizer.config.models import DetectionModelSelection
from anonymizer.config.regex import RegexRule
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_FINAL_ENTITIES,
    COL_MERGED_ENTITIES,
    COL_RAW_DETECTED,
    COL_REGEX_ACCEPTED_ENTITIES,
    COL_REGEX_VALIDATION_TRACE,
    COL_REPLACED_TEXT,
    COL_SEED_ENTITIES,
    COL_TEXT,
    COL_VALIDATION_DECISIONS,
    DEFAULT_ENTITY_LABELS,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import parse_model_configs
from anonymizer.engine.replace.strategies import apply_local_replace_strategy
from anonymizer.engine.workflow_columns.detection.config import (
    ChunkedValidationConfig,
    DetectionTransformConfig,
    DetectionTransformOperation,
    RegexDetectionConfig,
)
from anonymizer.engine.workflow_columns.detection.plugins import (
    chunked_validation_plugin,
    detection_transform_plugin,
    regex_detection_plugin,
)


@pytest.mark.parametrize("plugin", [detection_transform_plugin, chunked_validation_plugin, regex_detection_plugin])
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
    assert len(columns) == 10
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
        if DetectionTransformOperation(column.operation) == DetectionTransformOperation.APPLY_VALIDATION_AND_FINALIZE
    )
    parse_transform = next(
        column
        for column in transforms
        if DetectionTransformOperation(column.operation) == DetectionTransformOperation.PARSE_DETECTED_ENTITIES
    )
    assert seed_validation_transform.excluded_entity_labels == ["email"]
    assert seed_validation_transform.allowed_entity_labels == ["first_name"]
    assert seed_validation_transform.regex_constrained_entity_labels == []
    assert merge_transform.excluded_entity_labels == ["email"]
    assert merge_transform.allowed_entity_labels == ["first_name"]
    assert merge_transform.regex_constrained_entity_labels == []
    assert finalize_transform.excluded_entity_labels == ["email"]
    assert finalize_transform.allowed_entity_labels == ["first_name"]
    assert finalize_transform.regex_constrained_entity_labels == []
    assert parse_transform.propagate_skip is False
    assert merge_transform.propagate_skip is False

    validation = next(column for column in columns if column.name == COL_VALIDATION_DECISIONS)
    assert isinstance(validation, ChunkedValidationConfig)
    assert validation.max_entities_per_call == 7
    assert validation.excerpt_window_chars == 321
    assert validation.pool == parsed_models.selected_models.detection.entity_validator
    assert "Customer support messages" in validation.prompt_template
    assert "Alicia" in validation.prompt_template
    assert "configured@example.test" not in validation.prompt_template

    augmenter = next(column for column in columns if column.name == COL_AUGMENTED_ENTITIES)
    assert isinstance(augmenter, LLMStructuredColumnConfig)
    assert "first_name: Alicia" in augmenter.prompt
    assert "Michael, Isabella, Carlos, Wei" not in augmenter.prompt
    assert "configured@example.test" not in augmenter.prompt

    regex_detection = next(column for column in columns if isinstance(column, RegexDetectionConfig))
    assert regex_detection.rules == []
    assert regex_detection.side_effect_columns == [COL_REGEX_ACCEPTED_ENTITIES, COL_REGEX_VALIDATION_TRACE]

    serialized = json.loads(payload)
    serialized_text = json.dumps(serialized)
    assert "anonymizer-detection-transform" in serialized_text
    assert "anonymizer-chunked-validation" in serialized_text
    assert "anonymizer-regex-detection" in serialized_text
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
    assert isinstance(augmenter, LLMStructuredColumnConfig)
    finalize = next(
        column
        for column in columns
        if isinstance(column, DetectionTransformConfig)
        and column.operation == DetectionTransformOperation.APPLY_VALIDATION_AND_FINALIZE
    )
    assert "- vendor_api_key: acme_live_abc123" in validation.prompt_template
    assert "vendor_api_key: acme_live_abc123" in augmenter.prompt
    assert "Use ONLY labels from this list" in augmenter.prompt
    assert finalize.allowed_entity_labels == [*DEFAULT_ENTITY_LABELS, "vendor_api_key"]


def test_regex_constrained_detection_config_round_trips_with_model_columns_skipped(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.parquet"
    pd.DataFrame({COL_TEXT: ["TKT-123"]}).to_parquet(seed_path, index=False)

    parsed_models = parse_model_configs(None)
    workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = workflow.build_detection_builder_for_seed(
        seed_path=seed_path,
        model_configs=parsed_models.model_configs,
        selected_models=parsed_models.selected_models.detection,
        gliner_detection_threshold=0.3,
        entity_labels=["ticket"],
        regex_rules=[
            RegexRule(
                label="ticket",
                pattern=r"TKT-\d+",
                validate_matches_with_llm=False,
                detect_additional_matches=False,
            )
        ],
    )

    payload = builder.get_builder_config().to_json()
    assert payload is not None
    restored = DataDesignerConfigBuilder.from_config(payload)
    columns = restored.get_column_configs()

    assert _get_gliner_labels_from_builder(builder) == []
    assert next(column for column in columns if column.name == COL_RAW_DETECTED).skip is not None
    assert next(column for column in columns if column.name == COL_AUGMENTED_ENTITIES).skip is not None
    assert next(column for column in columns if column.name == COL_SEED_ENTITIES).propagate_skip is False
    assert next(column for column in columns if column.name == COL_MERGED_ENTITIES).propagate_skip is False


@pytest.mark.parametrize("preview_num_records", [2, None], ids=["preview", "create"])
def test_regex_constrained_matches_survive_skipped_model_columns_in_data_designer(
    tmp_path: Path,
    preview_num_records: int | None,
) -> None:
    input_df = pd.DataFrame({COL_TEXT: ["TKT-123", "nothing"]})
    model_configs = [
        ModelConfig(
            alias="known",
            model="stub-model",
            provider="stub",
            skip_health_check=True,
        )
    ]
    selected_models = DetectionModelSelection(
        entity_detector="known",
        entity_validator="known",
        entity_augmenter="known",
        latent_detector="known",
    )
    build_workflow = EntityDetectionWorkflow(adapter=NddAdapter(data_designer=cast(DataDesigner, Mock())))
    builder = build_workflow.build_detection_config(
        input_df,
        seed_path=tmp_path / "seed.parquet",
        model_configs=model_configs,
        selected_models=selected_models,
        gliner_detection_threshold=0.3,
        entity_labels=["ticket"],
        builtin_regexes=False,
        regex_rules=[
            RegexRule(
                label="ticket",
                pattern=r"TKT-\d+",
                validate_matches_with_llm=False,
                detect_additional_matches=False,
            )
        ],
    )
    payload = builder.get_builder_config().to_json()
    assert payload is not None
    restored = DataDesignerConfigBuilder.from_config(payload)

    provider = ModelProvider(
        name="stub",
        endpoint="http://127.0.0.1:9/v1",
        provider_type="openai",
        api_key="EMPTY",
    )
    data_designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[provider],
        auto_configure_logging=False,
    )
    adapter = NddAdapter(data_designer=data_designer)
    with (
        patch("httpx.Client.send", side_effect=AssertionError("unexpected synchronous model request")),
        patch("httpx.AsyncClient.send", side_effect=AssertionError("unexpected asynchronous model request")),
    ):
        result = adapter.run_workflow(
            input_df,
            model_configs=restored.model_configs,
            columns=restored.get_column_configs(),
            workflow_name=f"regex-constrained-{'preview' if preview_num_records is not None else 'create'}",
            preview_num_records=preview_num_records,
        )

    assert result.failed_records == []
    detected = result.dataframe[COL_DETECTED_ENTITIES].tolist()
    assert [
        [
            (
                entity["value"],
                entity["label"],
                entity["start_position"],
                entity["end_position"],
            )
            for entity in detection["entities"]
        ]
        for detection in detected
    ] == [[("TKT-123", "ticket", 0, 7)], []]

    replacement_input = result.dataframe[[COL_TEXT]].copy()
    replacement_input[COL_FINAL_ENTITIES] = detected
    replaced = apply_local_replace_strategy(replacement_input, strategy=Redact())
    assert replaced[COL_REPLACED_TEXT].tolist() == ["[REDACTED_TICKET]", "nothing"]


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
    assert ("anonymizer-regex-detection", "RegexDetectionConfig") in restored_types
