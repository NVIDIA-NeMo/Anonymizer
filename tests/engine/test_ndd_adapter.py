# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
from data_designer.interface.data_designer import DataDesigner

from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter


def test_attach_record_ids_adds_deterministic_ids() -> None:
    adapter = NddAdapter(data_designer=Mock(spec=DataDesigner))
    input_df = pd.DataFrame({"text": ["a", "b"]})

    output_a = adapter._attach_record_ids(input_df)
    output_b = adapter._attach_record_ids(input_df)

    assert RECORD_ID_COLUMN in output_a.columns
    assert output_a[RECORD_ID_COLUMN].tolist() == output_b[RECORD_ID_COLUMN].tolist()


def test_detect_missing_records_returns_missing_ids() -> None:
    adapter = NddAdapter(data_designer=Mock(spec=DataDesigner))
    input_df = adapter._attach_record_ids(pd.DataFrame({"text": ["a", "b", "c"]}))
    output_df = input_df.iloc[[0, 2]].copy()

    failed_records = adapter._detect_missing_records(
        workflow_name="replace-workflow",
        input_df=input_df,
        output_df=output_df,
    )

    assert len(failed_records) == 1
    assert failed_records[0].step == "replace-workflow"


def test_detect_missing_records_for_preview_subset_has_no_false_failures() -> None:
    adapter = NddAdapter(data_designer=Mock(spec=DataDesigner))
    full_input_df = adapter._attach_record_ids(pd.DataFrame({"text": ["a", "b", "c"]}))
    preview_input_df = full_input_df.iloc[:1].copy()
    preview_output_df = preview_input_df.copy()

    failed_records = adapter._detect_missing_records(
        workflow_name="detect-workflow",
        input_df=preview_input_df,
        output_df=preview_output_df,
    )

    assert len(failed_records) == 0
