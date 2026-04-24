# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig
from data_designer.interface.data_designer import DataDesigner

from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter

_FORBIDDEN_BACKEND_STRINGS = ("Data Designer", "data_designer", "DD")


def _assert_no_backend_reference(message: str) -> None:
    for forbidden in _FORBIDDEN_BACKEND_STRINGS:
        assert forbidden not in message, f"log message leaks backend reference {forbidden!r}: {message}"


def _unique_records(
    caplog: pytest.LogCaptureFixture,
    *,
    level: int,
    message_contains: str | None = None,
) -> list[logging.LogRecord]:
    """Return unique records at *level* from the ``anonymizer.ndd`` logger.

    The autouse ``_caplog_for_anonymizer`` fixture causes each record to be
    observed twice (once via the anonymizer logger, once via propagation to root),
    so we dedupe by object identity.
    """
    seen: dict[int, logging.LogRecord] = {}
    for record in caplog.records:
        if record.name != "anonymizer.ndd" or record.levelno != level:
            continue
        if message_contains is not None and message_contains not in record.getMessage():
            continue
        seen[id(record)] = record
    return list(seen.values())


def _make_model_config(alias: str = "test-model-alias") -> ModelConfig:
    return ModelConfig(alias=alias, model="dummy-model-id")


def _make_columns() -> list[ColumnConfigT]:
    return [
        LLMTextColumnConfig(
            name="output",
            prompt="Echo: {{ text }}",
            model_alias="test-model-alias",
        ),
    ]


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


def test_preview_exception_warns_with_count_model_and_type_and_debug_carries_workflow(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="anonymizer.ndd")

    class PreviewExplosion(Exception):
        pass

    mock_dd = Mock(spec=DataDesigner)
    mock_dd.preview.side_effect = PreviewExplosion("endpoint unreachable")

    adapter = NddAdapter(data_designer=mock_dd)
    input_df = pd.DataFrame({"text": ["row-1", "row-2", "row-3"]})

    with pytest.raises(PreviewExplosion):
        adapter.run_workflow(
            input_df,
            model_configs=[_make_model_config()],
            columns=_make_columns(),
            workflow_name="detect-workflow",
            preview_num_records=3,
        )

    warning_records = _unique_records(caplog, level=logging.WARNING, message_contains="preview failed")
    assert len(warning_records) == 1
    warning_msg = warning_records[0].getMessage()
    assert "3" in warning_msg
    assert "test-model-alias" in warning_msg
    assert "PreviewExplosion" in warning_msg
    assert "detect-workflow" not in warning_msg
    _assert_no_backend_reference(warning_msg)

    debug_records = _unique_records(caplog, level=logging.DEBUG, message_contains="preview failure context")
    assert len(debug_records) == 1
    debug_msg = debug_records[0].getMessage()
    assert "detect-workflow" in debug_msg
    assert "output" in debug_msg
    _assert_no_backend_reference(debug_msg)


def test_create_exception_warns_with_count_model_and_type_and_debug_carries_workflow(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="anonymizer.ndd")

    class CreateExplosion(Exception):
        pass

    mock_dd = Mock(spec=DataDesigner)
    mock_dd.create.side_effect = CreateExplosion("quota exceeded")

    adapter = NddAdapter(data_designer=mock_dd)
    input_df = pd.DataFrame({"text": ["row-1", "row-2"]})

    with pytest.raises(CreateExplosion):
        adapter.run_workflow(
            input_df,
            model_configs=[_make_model_config()],
            columns=_make_columns(),
            workflow_name="replace-workflow",
            preview_num_records=None,
        )

    warning_records = _unique_records(caplog, level=logging.WARNING, message_contains="execution failed")
    assert len(warning_records) == 1
    warning_msg = warning_records[0].getMessage()
    assert "2" in warning_msg
    assert "test-model-alias" in warning_msg
    assert "CreateExplosion" in warning_msg
    assert "replace-workflow" not in warning_msg
    _assert_no_backend_reference(warning_msg)

    debug_records = _unique_records(caplog, level=logging.DEBUG, message_contains="execution failure context")
    assert len(debug_records) == 1
    debug_msg = debug_records[0].getMessage()
    assert "replace-workflow" in debug_msg
    assert "output" in debug_msg
    _assert_no_backend_reference(debug_msg)


def test_detect_missing_records_short_circuit_warns_when_input_missing_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = NddAdapter(data_designer=Mock(spec=DataDesigner))
    caplog.set_level(logging.DEBUG, logger="anonymizer.ndd")
    caplog.clear()

    input_df = pd.DataFrame({"text": ["a", "b", "c"], "label": [1, 2, 3]})
    output_df = pd.DataFrame({"text": ["a", "b", "c"]})

    result = adapter._detect_missing_records(
        workflow_name="detect-workflow",
        input_df=input_df,
        output_df=output_df,
    )

    assert result == []

    warning_records = _unique_records(caplog, level=logging.WARNING)
    assert len(warning_records) == 1
    warning_msg = warning_records[0].getMessage()
    assert "3" in warning_msg
    assert "detection skipped" in warning_msg
    assert "invariant violation" in warning_msg
    _assert_no_backend_reference(warning_msg)

    debug_records = _unique_records(caplog, level=logging.DEBUG)
    assert len(debug_records) == 1
    debug_msg = debug_records[0].getMessage()
    assert "detect-workflow" in debug_msg
    assert RECORD_ID_COLUMN in debug_msg
    assert "text" in debug_msg
    assert "label" in debug_msg
    _assert_no_backend_reference(debug_msg)


def test_detect_missing_records_short_circuit_warns_when_output_missing_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = NddAdapter(data_designer=Mock(spec=DataDesigner))
    caplog.set_level(logging.DEBUG, logger="anonymizer.ndd")
    caplog.clear()

    input_df = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["id-1", "id-2", "id-3"],
            "text": ["a", "b", "c"],
            "label": [1, 2, 3],
        }
    )
    output_df = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "rewrite": ["A", "B", "C"],
        }
    )

    result = adapter._detect_missing_records(
        workflow_name="rewrite-workflow",
        input_df=input_df,
        output_df=output_df,
    )

    assert len(result) == 3
    for record in result:
        assert record.step == "rewrite-workflow"
        assert RECORD_ID_COLUMN in record.reason
    assert {r.record_id for r in result} == {"id-1", "id-2", "id-3"}

    warning_records = _unique_records(caplog, level=logging.WARNING)
    assert len(warning_records) == 1
    warning_msg = warning_records[0].getMessage()
    assert "3" in warning_msg
    assert "detection disabled" in warning_msg
    assert "'label'" in warning_msg
    assert "'rewrite'" in warning_msg
    assert RECORD_ID_COLUMN not in warning_msg
    _assert_no_backend_reference(warning_msg)

    debug_records = _unique_records(caplog, level=logging.DEBUG)
    assert len(debug_records) == 1
    debug_msg = debug_records[0].getMessage()
    assert "rewrite-workflow" in debug_msg
    assert RECORD_ID_COLUMN in debug_msg
    assert "'text'" in debug_msg
    assert "'label'" in debug_msg
    assert "'rewrite'" in debug_msg
    _assert_no_backend_reference(debug_msg)
