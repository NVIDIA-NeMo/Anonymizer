# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.errors import InvalidFileFormatError
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.utils.io_helpers import read_parquet_dataset
from data_designer.config.utils.storage_encoding import read_storage_plan
from data_designer.config.utils.storage_schema import JSON_STORAGE_RUN_KEY
from data_designer.engine.models.facade import ModelFacade
from data_designer.interface.data_designer import DataDesigner
from data_designer.interface.errors import DataDesignerGenerationError

from anonymizer.config.models import DetectionModelSelection, ReplaceModelSelection
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_LATENT_ENTITIES,
    COL_REPLACEMENT_MAP,
    COL_SEED_ENTITIES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_INVENTORIES,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.storage import prepare_json_storage_column, prepare_json_storage_seed
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.interface.errors import AnonymizerWorkflowError

_USER_JSON_TEXT = "user_json_text"
_PLUGIN_DIAGNOSTIC = "plugin_diagnostic"


def test_lifecycle_inventory_arrow_handoff_preserves_nested_values(tmp_path: Path) -> None:
    values = [[{"request_id": "synthetic", "entries": []}], []]
    table = pa.table({COL_VALIDATION_INVENTORIES: values})
    original = table.to_pandas(types_mapper=pd.ArrowDtype)
    prepared = prepare_json_storage_seed(original)
    source = LocalFileSeedSource.from_dataframe(prepared, str(tmp_path / "inventories.parquet"))
    loaded = read_parquet_dataset(Path(source.path))
    assert loaded[COL_VALIDATION_INVENTORIES].tolist() == values
    assert read_storage_plan(pq.read_schema(source.path)).columns == {COL_VALIDATION_INVENTORIES: [[]]}


@custom_column_generator(required_columns=[COL_TEXT])
def _generate_nested_decisions(row: dict[str, Any]) -> dict[str, Any]:
    row[COL_VALIDATION_DECISIONS] = {"decisions": [], "diagnostics": {"count": 0, "details": {}}}
    return row


@custom_column_generator(required_columns=[COL_VALIDATION_DECISIONS])
def _generate_latent_entities(row: dict[str, Any]) -> dict[str, Any]:
    assert isinstance(row[COL_VALIDATION_DECISIONS], dict)
    row[COL_LATENT_ENTITIES] = {"latent_entities": []}
    return row


@custom_column_generator(required_columns=[COL_TEXT], side_effect_columns=[_PLUGIN_DIAGNOSTIC])
def _generate_side_effect(row: dict[str, Any]) -> dict[str, Any]:
    row[COL_VALIDATION_DECISIONS] = {"decisions": []}
    row[_PLUGIN_DIAGNOSTIC] = [{"request": "synthetic", "empty": {}}]
    return row


@custom_column_generator(required_columns=[COL_TEXT], side_effect_columns=[COL_SEED_ENTITIES])
def _generate_seed_entities_side_effect(row: dict[str, Any]) -> dict[str, Any]:
    span = {"value": "Ada", "label": "first_name", "start_position": 0, "end_position": 3}
    row[COL_VALIDATION_DECISIONS] = {"decisions": []}
    row[COL_SEED_ENTITIES] = {"entities": [span] if row[COL_TEXT].startswith("Ada") else []}
    return row


@custom_column_generator(required_columns=[COL_TEXT])
def _fail_generation(row: dict[str, Any]) -> dict[str, Any]:
    raise ValueError("Synthetic deterministic generation failure")


async def _stub_model_generation(
    self: ModelFacade, *, parser: Callable[[str], Any], purpose: str, **kwargs: Any
) -> tuple[Any, list[Any]]:
    if COL_REPLACEMENT_MAP in purpose:
        return parser('{"replacements":[]}'), []
    if COL_LATENT_ENTITIES in purpose:
        return parser('{"latent_entities":[]}'), []
    raise AssertionError(f"Unexpected model boundary: {purpose}")


@pytest.mark.parametrize(
    "values",
    [
        [{}, {}],
        [[], []],
        [None, None],
        [{"nested": {"empty": {}, "values": [1, "two", None]}}, []],
        [{"decisions": []}, {"decisions": [{"keep": True, "reason": None}]}],
    ],
)
def test_json_handoff_roundtrip_preserves_logical_shapes(tmp_path: Path, values: list[Any]) -> None:
    original = pd.DataFrame({COL_VALIDATION_DECISIONS: values, _USER_JSON_TEXT: ['{"untouched":1}'] * 2})
    prepared = prepare_json_storage_seed(original)
    source = LocalFileSeedSource.from_dataframe(prepared, str(tmp_path / "first.parquet"))
    first = read_parquet_dataset(Path(source.path))
    second_source = LocalFileSeedSource.from_dataframe(first, str(tmp_path / "second.parquet"))
    second = read_parquet_dataset(Path(second_source.path))

    assert first[COL_VALIDATION_DECISIONS].tolist() == values
    assert second[COL_VALIDATION_DECISIONS].tolist() == values
    assert second[_USER_JSON_TEXT].tolist() == original[_USER_JSON_TEXT].tolist()
    assert read_storage_plan(pq.read_schema(second_source.path)).columns == {COL_VALIDATION_DECISIONS: [[]]}
    assert original.attrs == {}
    assert original[COL_VALIDATION_DECISIONS].tolist() == values


def test_arrow_backed_batch_column_is_materialized_without_dtype_reuse(tmp_path: Path) -> None:
    values = [{"decisions": []}, {"decisions": [{"id": "synthetic", "keep": True}]}]
    first_path = tmp_path / "native.parquet"
    pq.write_table(pa.Table.from_pylist([{COL_VALIDATION_DECISIONS: value} for value in values]), first_path)
    batch = read_parquet_dataset(first_path)
    assert isinstance(batch[COL_VALIDATION_DECISIONS].dtype, pd.ArrowDtype)

    prepared = prepare_json_storage_seed(batch)
    second_source = LocalFileSeedSource.from_dataframe(prepared, str(tmp_path / "handoff.parquet"))
    restored = read_parquet_dataset(Path(second_source.path))

    assert restored[COL_VALIDATION_DECISIONS].tolist() == values
    assert isinstance(batch[COL_VALIDATION_DECISIONS].dtype, pd.ArrowDtype)
    assert prepared[COL_VALIDATION_DECISIONS].dtype == object


def test_existing_field_contract_is_preserved() -> None:
    frame = pd.DataFrame({COL_VALIDATION_DECISIONS: [{"freeform": {}}]})
    frame.attrs[JSON_STORAGE_RUN_KEY] = {"version": 1, "columns": {COL_VALIDATION_DECISIONS: [["freeform"]]}}
    prepared = prepare_json_storage_seed(frame)
    assert prepared.attrs[JSON_STORAGE_RUN_KEY] == frame.attrs[JSON_STORAGE_RUN_KEY]
    column = LLMStructuredColumnConfig(
        name=COL_VALIDATION_DECISIONS,
        model_alias="synthetic",
        prompt="Synthetic",
        output_format={
            "type": "object",
            "properties": {"freeform": {"type": "object", "x-data-designer-storage": "json"}},
        },
    )
    assert prepare_json_storage_column(column) is column


def test_real_batch_write_read_write_preserves_nested_outputs(tmp_path: Path) -> None:
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=[ModelProvider(name="unused-synthetic", endpoint="http://127.0.0.1:9/v1")],
        auto_configure_logging=False,
    )
    adapter = NddAdapter(designer)
    first = adapter.run_workflow(
        pd.DataFrame({COL_TEXT: ["Synthetic record"]}),
        model_configs=[],
        columns=[CustomColumnConfig(name=COL_VALIDATION_DECISIONS, generator_function=_generate_nested_decisions)],
        workflow_name="synthetic-detection",
    )
    second = adapter.run_workflow(
        first.dataframe,
        model_configs=[],
        columns=[CustomColumnConfig(name=COL_LATENT_ENTITIES, generator_function=_generate_latent_entities)],
        workflow_name="synthetic-latent",
    )
    assert first.failed_records == second.failed_records == []
    assert second.dataframe[COL_VALIDATION_DECISIONS].tolist() == first.dataframe[COL_VALIDATION_DECISIONS].tolist()
    assert second.dataframe[COL_LATENT_ENTITIES].tolist() == [{"latent_entities": []}]
    assert COL_VALIDATION_DECISIONS in second.dataframe.attrs[JSON_STORAGE_RUN_KEY]["columns"]


def test_registered_side_effect_survives_row_groups_with_different_nested_types(tmp_path: Path) -> None:
    # Empty row groups infer a different nested type than populated ones, so DD reads the side-effect column back
    # as object arrays; the next seed must still encode it as JSON.
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=[ModelProvider(name="unused-synthetic", endpoint="http://127.0.0.1:9/v1")],
        auto_configure_logging=False,
    )
    designer.set_run_config(RunConfig(buffer_size=2))
    adapter = NddAdapter(designer)
    first = adapter.run_workflow(
        pd.DataFrame({COL_TEXT: ["No names", "No names", "Ada one", "Ada two"]}),
        model_configs=[],
        columns=[
            CustomColumnConfig(name=COL_VALIDATION_DECISIONS, generator_function=_generate_seed_entities_side_effect)
        ],
        workflow_name="synthetic-detection",
    )
    second = adapter.run_workflow(
        first.dataframe,
        model_configs=[],
        columns=[CustomColumnConfig(name=COL_LATENT_ENTITIES, generator_function=_generate_latent_entities)],
        workflow_name="synthetic-latent",
    )

    assert first.failed_records == second.failed_records == []
    span = {"value": "Ada", "label": "first_name", "start_position": 0, "end_position": 3}
    assert second.dataframe[COL_SEED_ENTITIES].tolist() == [{"entities": []}] * 2 + [{"entities": [span]}] * 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), None),
        (
            {"entities": np.array([{"start_position": np.int64(0)}], dtype=object)},
            {"entities": [{"start_position": 0}]},
        ),
    ],
    ids=["nan", "numpy"],
)
def test_seed_values_are_plain_python_before_json_encoding(tmp_path: Path, value: Any, expected: Any) -> None:
    prepared = prepare_json_storage_seed(pd.DataFrame({COL_VALIDATION_DECISIONS: [value]}))
    source = LocalFileSeedSource.from_dataframe(prepared, str(tmp_path / "seed.parquet"))

    assert read_parquet_dataset(Path(source.path))[COL_VALIDATION_DECISIONS].tolist() == [expected]


def test_build_config_uses_same_storage_contract(tmp_path: Path) -> None:
    adapter = NddAdapter(Mock(spec=DataDesigner))
    path = tmp_path / "seed.parquet"
    adapter.build_config(
        pd.DataFrame({COL_VALIDATION_DECISIONS: [{"empty": {}}]}),
        model_configs=[],
        columns=[],
        seed_path=path,
    )
    assert read_parquet_dataset(path)[COL_VALIDATION_DECISIONS].tolist() == [{"empty": {}}]
    assert read_storage_plan(pq.read_schema(path)).columns == {COL_VALIDATION_DECISIONS: [[]]}


@pytest.mark.parametrize("mode", ["run", "preview", "export"])
@pytest.mark.parametrize("failure", ["metadata", "encoding"])
def test_seed_storage_errors_use_workflow_error(tmp_path: Path, mode: str, failure: str) -> None:
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=[ModelProvider(name="unused-synthetic", endpoint="http://127.0.0.1:9/v1")],
        auto_configure_logging=False,
    )
    adapter = NddAdapter(designer)
    dataframe = pd.DataFrame({COL_TEXT: ["Synthetic record"], COL_VALIDATION_DECISIONS: [{"score": float("inf")}]})
    if failure == "metadata":
        dataframe.attrs[JSON_STORAGE_RUN_KEY] = {"version": 2, "columns": {COL_VALIDATION_DECISIONS: [[]]}}

    with pytest.raises(AnonymizerWorkflowError) as captured:
        if mode == "export":
            adapter.build_config(dataframe, model_configs=[], columns=[], seed_path=tmp_path / "seed.parquet")
        else:
            adapter.run_workflow(
                dataframe,
                model_configs=[],
                columns=[],
                workflow_name="synthetic-invalid-seed",
                preview_num_records=1 if mode == "preview" else None,
            )

    assert isinstance(captured.value.__cause__, InvalidFileFormatError)


@pytest.mark.parametrize("next_step", ["substitute", "latent"])
def test_actual_workflow_handoff_from_batch_loaded_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, next_step: str
) -> None:
    # Only the remote model and readiness boundaries are replaced. Config building,
    # scheduling, prompt rendering, response parsing, and every Parquet I/O are real.
    monkeypatch.setattr(ModelFacade, "agenerate", _stub_model_generation)
    monkeypatch.setattr("data_designer.engine.dataset_builders.dataset_builder.run_readiness_check", Mock())
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=[ModelProvider(name="synthetic", endpoint="http://127.0.0.1:9/v1")],
        auto_configure_logging=False,
    )
    designer.set_run_config(RunConfig(buffer_size=2))
    adapter = NddAdapter(designer)
    texts = ["No names one", "No names two", "Ada one", "Ada two"]
    entities = {"entities_by_value": [{"value": "Ada", "labels": ["first_name"]}]}
    detected = adapter.run_workflow(
        pd.DataFrame(
            {
                COL_TEXT: texts,
                COL_TAGGED_TEXT: texts,
                COL_TAG_NOTATION: ["sentinel"] * len(texts),
                COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}] * 2 + [entities] * 2,
            }
        ),
        model_configs=[],
        columns=[
            CustomColumnConfig(name=COL_VALIDATION_DECISIONS, generator_function=_generate_seed_entities_side_effect)
        ],
        workflow_name="synthetic-detection-batch",
    )
    configs = [ModelConfig(alias="synthetic", model="synthetic-model", provider="synthetic")]
    if next_step == "substitute":
        result = LlmReplaceWorkflow(adapter).generate_map_only(
            detected.dataframe,
            model_configs=configs,
            selected_models=ReplaceModelSelection(replacement_generator="synthetic"),
        )
        assert result.dataframe[COL_REPLACEMENT_MAP].tolist() == [{"replacements": []}] * len(texts)
    else:
        result = EntityDetectionWorkflow(adapter).identify_latent_entities(
            detected.dataframe,
            model_configs=configs,
            selected_models=DetectionModelSelection(
                entity_detector="synthetic",
                entity_validator=["synthetic"],
                entity_augmenter="synthetic",
                latent_detector="synthetic",
            ),
            gliner_detection_threshold=0.3,
            privacy_goal=None,
        )
        assert result.dataframe[COL_LATENT_ENTITIES].tolist() == [{"latent_entities": []}] * len(texts)
    assert detected.failed_records == []
    assert result.failed_records == []
    assert result.dataframe[COL_TEXT].tolist() == texts
    assert result.dataframe[RECORD_ID_COLUMN].tolist() == detected.dataframe[RECORD_ID_COLUMN].tolist()
    entity_spans = [EntitiesSchema.from_raw(raw).entities for raw in result.dataframe[COL_SEED_ENTITIES]]
    assert [
        [(entity.value, entity.label, entity.start_position, entity.end_position) for entity in entities]
        for entities in entity_spans
    ] == [[], [], [("Ada", "first_name", 0, 3)], [("Ada", "first_name", 0, 3)]]
    assert result.dataframe[COL_VALIDATION_DECISIONS].tolist() == detected.dataframe[COL_VALIDATION_DECISIONS].tolist()


def test_side_effect_storage_is_not_implied_by_producer() -> None:
    producer = CustomColumnConfig(name=COL_VALIDATION_DECISIONS, generator_function=_generate_side_effect)
    configured = prepare_json_storage_column(producer)
    assert configured.storage_encoding == "json"
    assert configured.side_effect_columns == [_PLUGIN_DIAGNOSTIC]
    frame = prepare_json_storage_seed(pd.DataFrame({_PLUGIN_DIAGNOSTIC: [[{"empty": {}}]]}))
    assert JSON_STORAGE_RUN_KEY not in frame.attrs


def test_dd010_all_failed_batch_still_raises_not_failed_records(tmp_path: Path) -> None:
    # #331 tracks returning failed records for all-failed batches; retain the current exception until then.
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=[ModelProvider(name="unused-synthetic", endpoint="http://127.0.0.1:9/v1")],
        auto_configure_logging=False,
    )
    adapter = NddAdapter(designer)
    with pytest.raises(AnonymizerWorkflowError) as captured:
        adapter.run_workflow(
            pd.DataFrame({COL_TEXT: ["Synthetic record"], RECORD_ID_COLUMN: ["synthetic-id"]}),
            model_configs=[],
            columns=[CustomColumnConfig(name=COL_VALIDATION_DECISIONS, generator_function=_fail_generation)],
            workflow_name="synthetic-all-failed",
        )
    assert isinstance(captured.value.__cause__, DataDesignerGenerationError)
