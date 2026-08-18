# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.interface.data_designer import DataDesigner

from anonymizer.engine.ndd import adapter as ndd_adapter
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter
from anonymizer.interface.errors import AnonymizerWorkflowError
from anonymizer.measurement import MeasurementCollector, measurement_session

_FORBIDDEN_BACKEND_STRINGS = ("Data Designer", "DataDesigner", "data_designer", "DD")


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
    return ModelConfig(alias=alias, model="dummy-model-id", provider="stub")


def _make_columns() -> list[ColumnConfigT]:
    return [
        LLMTextColumnConfig(
            name="output",
            prompt="Echo: {{ text }}",
            model_alias="test-model-alias",
        ),
    ]


def test_as_alias_list_drops_none_items_before_stringifying() -> None:
    assert ndd_adapter._as_alias_list(["validator", None, "", 0]) == ["validator", "0"]


def test_attach_record_ids_adds_deterministic_ids() -> None:
    adapter = NddAdapter(data_designer=Mock(spec=DataDesigner))
    input_df = pd.DataFrame({"text": ["a", "b"]})

    output_a = adapter._attach_record_ids(input_df)
    output_b = adapter._attach_record_ids(input_df)

    assert RECORD_ID_COLUMN in output_a.columns
    assert output_a[RECORD_ID_COLUMN].tolist() == output_b[RECORD_ID_COLUMN].tolist()


def test_total_input_tokens_defaults_to_zero() -> None:
    adapter = NddAdapter(data_designer=Mock(spec=DataDesigner))

    assert adapter.total_input_tokens == 0


def test_consume_input_tokens_returns_and_resets() -> None:
    adapter = NddAdapter(data_designer=Mock(spec=DataDesigner))
    adapter._add_input_tokens({"m": {"token_usage": {"input_tokens": 42}}})

    assert adapter.consume_input_tokens() == 42
    assert adapter.consume_input_tokens() == 0


def test_add_input_tokens_sums_positive_counts_under_counter_lock() -> None:
    class CountingLock:
        def __init__(self) -> None:
            self.enter_count = 0

        def __enter__(self) -> CountingLock:
            self.enter_count += 1
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            pass

    adapter = NddAdapter(data_designer=Mock(spec=DataDesigner))
    lock = CountingLock()
    adapter._input_tokens_lock = lock  # ty: ignore[invalid-assignment]

    adapter._add_input_tokens(
        {
            "detector": {"token_usage": {"input_tokens": 12}},
            "validator": {"token_usage": {"input_tokens": 6}},
            "missing": {"token_usage": {}},
            "negative": {"token_usage": {"input_tokens": -1}},
            "invalid": object(),
        }
    )

    assert adapter.total_input_tokens == 18
    assert lock.enter_count == 2


def test_run_workflow_accumulates_input_tokens_without_measurement_collector() -> None:
    input_df = pd.DataFrame({"text": ["Alice works at Acme"], RECORD_ID_COLUMN: ["record-a"]})

    class UsageStats:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {
                "token_usage": {
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "total_tokens": 16,
                },
            }

    class ModelRegistry:
        def get_model_usage_snapshot(self) -> dict[str, UsageStats]:
            return {"dummy-model": UsageStats()}

    class UsageDataDesigner:
        def _create_resource_provider(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(model_registry=ModelRegistry())

        def preview(self, _config_builder: object, *, num_records: int) -> SimpleNamespace:
            self._create_resource_provider("preview-dataset", _config_builder)
            return SimpleNamespace(dataset=input_df.iloc[:num_records].copy())

    adapter = NddAdapter(data_designer=cast(DataDesigner, UsageDataDesigner()))

    adapter.run_workflow(
        input_df,
        model_configs=[_make_model_config()],
        columns=_make_columns(),
        workflow_name="detect-workflow",
        preview_num_records=1,
    )

    assert adapter.total_input_tokens == 12


def test_run_workflow_accumulates_input_tokens_on_error() -> None:
    input_df = pd.DataFrame({"text": ["Alice works at Acme"], RECORD_ID_COLUMN: ["record-a"]})

    class UsageStats:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {
                "token_usage": {
                    "input_tokens": 7,
                    "output_tokens": 0,
                    "total_tokens": 7,
                },
            }

    class ModelRegistry:
        def get_model_usage_snapshot(self) -> dict[str, UsageStats]:
            return {"dummy-model": UsageStats()}

    class UsageDataDesigner:
        def _create_resource_provider(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(model_registry=ModelRegistry())

        def preview(self, _config_builder: object, *, num_records: int) -> SimpleNamespace:
            _ = num_records
            self._create_resource_provider("preview-dataset", _config_builder)
            raise RuntimeError("boom")

    adapter = NddAdapter(data_designer=cast(DataDesigner, UsageDataDesigner()))

    with pytest.raises(AnonymizerWorkflowError, match="Workflow failed"):
        adapter.run_workflow(
            input_df,
            model_configs=[_make_model_config()],
            columns=_make_columns(),
            workflow_name="detect-workflow",
            preview_num_records=1,
        )

    assert adapter.total_input_tokens == 7


def test_private_execution_uses_ephemeral_artifacts_and_suppresses_active_collector(tmp_path: Path) -> None:
    private_canary = "PRIVATE-CORRELATION-CANARY"
    raw_canary = "RAW-ROW-alice@example.test"
    durable_root = tmp_path / "durable"
    durable_root.mkdir()

    class PrivateDataDesigner:
        _artifact_path = durable_root
        run_config = RunConfig(async_trace=True)

        def set_run_config(self, run_config: RunConfig) -> None:
            self.run_config = run_config

        def create(self, _builder: object, **kwargs: object) -> SimpleNamespace:
            assert self.run_config.async_trace is False
            artifact_root = Path(cast(str, kwargs.get("artifact_path", self._artifact_path)))
            assert artifact_root != self._artifact_path
            artifact = artifact_root / "captured.txt"
            artifact.write_text(f"{private_canary}\n{raw_canary}")
            output = pd.DataFrame(
                {
                    "text": [raw_canary],
                    "__anonymizer_private_row_correlation__": [private_canary],
                    RECORD_ID_COLUMN: ["record-a"],
                    "output": ["protected"],
                }
            )
            return SimpleNamespace(load_dataset=lambda: output, task_traces=[])

    data_designer = PrivateDataDesigner()
    adapter = NddAdapter(data_designer=cast(DataDesigner, data_designer))
    collector = MeasurementCollector()
    with measurement_session(collector), adapter.private_execution():
        result = adapter.run_workflow(
            pd.DataFrame(
                {
                    "text": [raw_canary],
                    "__anonymizer_private_row_correlation__": [private_canary],
                    RECORD_ID_COLUMN: ["record-a"],
                }
            ),
            model_configs=[_make_model_config()],
            columns=_make_columns(),
            workflow_name="private-protection",
        )

    assert result.dataframe.iloc[0]["output"] == "protected"
    assert data_designer.run_config.async_trace is True
    assert not any(durable_root.rglob("*"))
    assert collector.records == []


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


def test_preview_exception_wraps_in_workflow_error_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="anonymizer.ndd")

    class DataDesignerRuntimeError(Exception):
        pass

    mock_dd = Mock(spec=DataDesigner)
    mock_dd.preview.side_effect = DataDesignerRuntimeError("endpoint unreachable")

    adapter = NddAdapter(data_designer=mock_dd)
    input_df = pd.DataFrame({"text": ["row-1", "row-2", "row-3"]})

    with pytest.raises(AnonymizerWorkflowError) as exc_info:
        adapter.run_workflow(
            input_df,
            model_configs=[_make_model_config()],
            columns=_make_columns(),
            workflow_name="detect-workflow",
            preview_num_records=3,
        )

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert str(exc_info.value) == "Workflow failed"

    warning_records = _unique_records(caplog, level=logging.WARNING, message_contains="Workflow failed")
    assert len(warning_records) == 1
    warning_msg = warning_records[0].getMessage()
    assert "3" in warning_msg
    assert "test-model-alias" not in warning_msg
    assert "endpoint unreachable" not in warning_msg
    assert "Check endpoint reachability" not in warning_msg
    assert "detect-workflow" not in warning_msg
    _assert_no_backend_reference(warning_msg)

    assert not _unique_records(caplog, level=logging.DEBUG, message_contains="failure context")


def test_create_exception_wraps_in_workflow_error_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="anonymizer.ndd")

    class DataDesignerRuntimeError(Exception):
        pass

    mock_dd = Mock(spec=DataDesigner)
    mock_dd.create.side_effect = DataDesignerRuntimeError("quota exceeded")

    adapter = NddAdapter(data_designer=mock_dd)
    input_df = pd.DataFrame({"text": ["row-1", "row-2"]})

    with pytest.raises(AnonymizerWorkflowError) as exc_info:
        adapter.run_workflow(
            input_df,
            model_configs=[_make_model_config()],
            columns=_make_columns(),
            workflow_name="replace-workflow",
            preview_num_records=None,
        )

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert str(exc_info.value) == "Workflow failed"

    warning_records = _unique_records(caplog, level=logging.WARNING, message_contains="Workflow failed")
    assert len(warning_records) == 1
    warning_msg = warning_records[0].getMessage()
    assert "2" in warning_msg
    assert "test-model-alias" not in warning_msg
    assert "quota exceeded" not in warning_msg
    assert "Check endpoint reachability" not in warning_msg
    assert "replace-workflow" not in warning_msg
    _assert_no_backend_reference(warning_msg)

    assert not _unique_records(caplog, level=logging.DEBUG, message_contains="failure context")


def test_load_dataset_exception_wraps_in_workflow_error_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="anonymizer.ndd")

    class DataDesignerRuntimeError(Exception):
        pass

    mock_dd = Mock(spec=DataDesigner)
    mock_create_results = Mock()
    mock_create_results.load_dataset.side_effect = DataDesignerRuntimeError("corrupt parquet")
    mock_dd.create.return_value = mock_create_results

    adapter = NddAdapter(data_designer=mock_dd)
    input_df = pd.DataFrame({"text": ["row-1", "row-2"]})

    with pytest.raises(AnonymizerWorkflowError) as exc_info:
        adapter.run_workflow(
            input_df,
            model_configs=[_make_model_config()],
            columns=_make_columns(),
            workflow_name="replace-workflow",
            preview_num_records=None,
        )

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert str(exc_info.value) == "Workflow failed"

    warning_records = _unique_records(caplog, level=logging.WARNING, message_contains="Workflow failed")
    assert len(warning_records) == 1
    warning_msg = warning_records[0].getMessage()
    assert "2" in warning_msg
    assert "test-model-alias" not in warning_msg
    assert "corrupt parquet" not in warning_msg
    assert "Check local storage" not in warning_msg
    assert "Check endpoint reachability" not in warning_msg
    assert "replace-workflow" not in warning_msg
    _assert_no_backend_reference(warning_msg)

    assert not _unique_records(caplog, level=logging.DEBUG, message_contains="failure context")


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
    assert "cannot verify" in warning_msg
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
