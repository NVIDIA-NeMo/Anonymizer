# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from pydantic import BaseModel

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_CLASSIFICATION,
    COL_FINAL_ENTITIES,
    COL_LATENT_ENTITIES,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA,
    COL_REPLACEMENT_MAP,
    COL_SENSITIVITY_DISPOSITION,
    COL_TEXT,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.ndd.adapter import NddAdapter, WorkflowRunResult
from anonymizer.engine.rewrite.sensitivity_disposition import SensitivityDispositionWorkflow
from anonymizer.engine.rewrite.workflow_utils import derive_seed_columns, run_columns, select_seed_cols


class _StubOutput(BaseModel):
    result: str = ""


def test_derive_excludes_internally_produced_columns() -> None:
    """Columns produced by an earlier config in the list are NOT seed columns."""

    @custom_column_generator(required_columns=["_col_a"])
    def gen_b(row: dict) -> dict:
        return row

    columns = [
        LLMStructuredColumnConfig(
            name="_col_a",
            prompt="Generate something from {{ _external_dep }}",
            model_alias="test",
            output_format=_StubOutput,
        ),
        CustomColumnConfig(name="_col_b", generator_function=gen_b),
    ]

    df = pd.DataFrame({RECORD_ID_COLUMN: ["r1"], "_external_dep": ["val"], "_col_a": ["exists"]})
    seed = derive_seed_columns(columns, df)
    assert "_col_a" not in seed
    assert "_external_dep" in seed
    assert RECORD_ID_COLUMN in seed


def test_derive_includes_external_dependencies() -> None:
    columns = [
        LLMStructuredColumnConfig(
            name="_out",
            prompt="Use {{ input_col }} and {{ other_col }}",
            model_alias="test",
            output_format=_StubOutput,
        ),
    ]
    df = pd.DataFrame({RECORD_ID_COLUMN: ["r1"], "input_col": ["a"], "other_col": ["b"]})
    seed = derive_seed_columns(columns, df)
    assert "input_col" in seed
    assert "other_col" in seed


def test_derive_always_includes_record_id() -> None:
    columns = [
        LLMStructuredColumnConfig(
            name="_out",
            prompt="No column refs",
            model_alias="test",
            output_format=_StubOutput,
        ),
    ]
    df = pd.DataFrame({RECORD_ID_COLUMN: ["r1"]})
    seed = derive_seed_columns(columns, df)
    assert RECORD_ID_COLUMN in seed


def test_derive_ignores_columns_missing_from_df() -> None:
    """Required columns not present in the dataframe are silently skipped."""

    @custom_column_generator(required_columns=["_missing_col"])
    def gen(row: dict) -> dict:
        return row

    columns = [CustomColumnConfig(name="_out", generator_function=gen)]
    df = pd.DataFrame({RECORD_ID_COLUMN: ["r1"]})
    seed = derive_seed_columns(columns, df)
    assert "_missing_col" not in seed
    assert RECORD_ID_COLUMN in seed


def test_derive_handles_side_effect_columns() -> None:
    """Side-effect columns count as produced and should not appear in seed."""

    @custom_column_generator(required_columns=["_input"], side_effect_columns=["_side_effect"])
    def gen(row: dict) -> dict:
        return row

    @custom_column_generator(required_columns=["_side_effect"])
    def gen2(row: dict) -> dict:
        return row

    columns = [
        CustomColumnConfig(name="_main_out", generator_function=gen),
        CustomColumnConfig(name="_second_out", generator_function=gen2),
    ]
    df = pd.DataFrame({RECORD_ID_COLUMN: ["r1"], "_input": ["v"], "_side_effect": ["x"]})
    seed = derive_seed_columns(columns, df)
    assert "_input" in seed
    assert "_side_effect" not in seed


def test_select_seed_cols_json_serializes_dicts() -> None:
    df = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["r1"],
            "text_col": ["hello"],
            "dict_col": [{"key": "value"}],
        }
    )
    seed = select_seed_cols(df, [RECORD_ID_COLUMN, "text_col", "dict_col"])
    assert seed["text_col"].iloc[0] == "hello"
    parsed = json.loads(seed["dict_col"].iloc[0])
    assert parsed == {"key": "value"}


def test_select_seed_cols_skips_already_string_columns() -> None:
    df = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["r1"],
            "str_col": ['{"already": "json"}'],
        }
    )
    seed = select_seed_cols(df, [RECORD_ID_COLUMN, "str_col"])
    assert seed["str_col"].iloc[0] == '{"already": "json"}'


def test_select_seed_cols_filters_to_present() -> None:
    df = pd.DataFrame({RECORD_ID_COLUMN: ["r1"], "a": [1]})
    seed = select_seed_cols(df, [RECORD_ID_COLUMN, "a", "missing_col"])
    assert list(seed.columns) == [RECORD_ID_COLUMN, "a"]


_NESTED_SEED_CELLS = {
    COL_ENTITIES_BY_VALUE: {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
    COL_ENTITY_CLASSIFICATION: {"first_name": "direct_identifier", "city": "quasi_identifier"},
    COL_FINAL_ENTITIES: {
        "entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]
    },
    COL_LATENT_ENTITIES: {"latent_entities": [{"label": "employer", "value": "Acme", "confidence": 0.8}]},
    COL_REPLACEMENT_MAP: {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]},
    COL_PRIVACY_QA: {"items": [{"id": 1, "question": "Who?", "sensitivity": "high", "entity_values": ["Alice"]}]},
    COL_PRIVACY_QA_REANSWER: {"answers": [{"id": 1, "answer": "unknown"}]},
    COL_QUALITY_QA: {"items": [{"id": 1, "question": "What role?", "reference_answer": "engineer"}]},
}


@pytest.mark.parametrize("column", sorted(_NESTED_SEED_CELLS))
def test_select_seed_cols_equal_after_parquet_reload(column: str, tmp_path: Path) -> None:
    """A frame saved to parquet and reloaded seeds exactly like the frame built in process."""
    frame = pd.DataFrame({RECORD_ID_COLUMN: ["r1"], column: [_NESTED_SEED_CELLS[column]]})
    path = tmp_path / "frame.parquet"
    frame.to_parquet(path)
    reloaded = pd.read_parquet(path)
    assert isinstance(reloaded[column].iloc[0], dict)
    pd.testing.assert_frame_equal(
        select_seed_cols(reloaded, [RECORD_ID_COLUMN, column]), select_seed_cols(frame, [RECORD_ID_COLUMN, column])
    )


def test_select_seed_cols_writes_models_and_numpy_values_as_plain_json() -> None:
    df = pd.DataFrame({"model": [_StubOutput(result="x")], "array": [{"ids": np.array([1, 2])}]})
    seed = select_seed_cols(df, ["model", "array"])
    assert json.loads(seed["model"].iloc[0]) == {"result": "x"}
    assert json.loads(seed["array"].iloc[0]) == {"ids": [1, 2]}


def test_run_columns_runs_sensitivity_disposition_columns(stub_rewrite_model_selection: RewriteModelSelection) -> None:
    columns = SensitivityDispositionWorkflow().columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=PrivacyGoal(
            protect="All direct identifiers including names", preserve="Career trajectory and skills overall"
        ),
    )
    frame = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["r1", "r2"],
            COL_TEXT: ["Alice works at Acme.", "Bob lives in Paris."],
            "unrelated": ["a", "b"],
            **{col: [cell, cell] for col, cell in _NESTED_SEED_CELLS.items()},
        }
    )
    adapter = Mock(spec=NddAdapter)
    adapter.run_workflow.side_effect = lambda seed, **_: WorkflowRunResult(
        dataframe=seed.iloc[[1]].assign(**{COL_SENSITIVITY_DISPOSITION: ["{}"]}).reset_index(drop=True),
        failed_records=[],
        source_positions=(1,),
    )

    run, seed_cols = run_columns(adapter, frame, columns, model_configs=[], workflow_name="disposition")

    assert seed_cols == derive_seed_columns(columns, frame)
    assert "unrelated" not in seed_cols
    seed = adapter.run_workflow.call_args.args[0]
    assert list(seed.columns) == seed_cols
    assert adapter.run_workflow.call_args.kwargs["workflow_name"] == "disposition"
    assert run.source_positions == (1,)
    assert run.dataframe[COL_SENSITIVITY_DISPOSITION].tolist() == ["{}"]
