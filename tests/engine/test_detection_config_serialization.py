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

from anonymizer.engine.constants import COL_TEXT, COL_VALIDATION_DECISIONS
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import parse_model_configs
from anonymizer.engine.workflow_columns.detection.config import (
    ChunkedValidationConfig,
    DetectionTransformConfig,
    DetectionTransformOperation,
)
from anonymizer.engine.workflow_columns.detection.plugins import (
    chunked_validation_plugin,
    detection_transform_plugin,
)


@pytest.mark.parametrize("plugin", [detection_transform_plugin, chunked_validation_plugin])
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
    assert len(columns) == 9
    assert all(column.column_type != "custom" for column in columns)

    transforms = [column for column in columns if isinstance(column, DetectionTransformConfig)]
    assert {DetectionTransformOperation(column.operation) for column in transforms} == set(DetectionTransformOperation)

    validation = next(column for column in columns if column.name == COL_VALIDATION_DECISIONS)
    assert isinstance(validation, ChunkedValidationConfig)
    assert validation.max_entities_per_call == 7
    assert validation.excerpt_window_chars == 321
    assert validation.pool == parsed_models.selected_models.detection.entity_validator
    assert "Customer support messages" in validation.prompt_template

    serialized = json.loads(payload)
    serialized_text = json.dumps(serialized)
    assert "anonymizer-detection-transform" in serialized_text
    assert "anonymizer-chunked-validation" in serialized_text
    assert "generator_function" not in serialized_text
    assert "generator_params" not in serialized_text


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
