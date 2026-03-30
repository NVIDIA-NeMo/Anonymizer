# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pandas as pd
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from pydantic import BaseModel

from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN
from anonymizer.engine.rewrite.workflow_utils import derive_seed_columns, select_seed_cols


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
