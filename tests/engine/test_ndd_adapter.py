# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.engine.errors import DataDesignerRuntimeError
from data_designer.errors import DataDesignerError
from data_designer.interface.data_designer import DataDesigner
from data_designer.interface.errors import (
    DataDesignerEarlyShutdownError,
    DataDesignerGenerationError,
    DataDesignerProfilingError,
    DataDesignerWorkflowError,
)

from anonymizer.engine.constants import RECORD_ID_COLUMN
from anonymizer.engine.ndd import adapter as ndd_adapter
from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_response import _RUN_STOP, RunStop
from anonymizer.interface.errors import AnonymizerWorkflowError

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

        def preview(self, _config_builder: DataDesignerConfigBuilder, *, num_records: int) -> SimpleNamespace:
            self._create_resource_provider("preview-dataset", _config_builder)
            seed = _config_builder.get_seed_config()
            assert seed is not None and isinstance(seed.source, LocalFileSeedSource)
            return SimpleNamespace(dataset=pd.read_parquet(seed.source.path).iloc[:num_records].copy())

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

    with pytest.raises(AnonymizerWorkflowError, match="boom"):
        adapter.run_workflow(
            input_df,
            model_configs=[_make_model_config()],
            columns=_make_columns(),
            workflow_name="detect-workflow",
            preview_num_records=1,
        )

    assert adapter.total_input_tokens == 7


def test_preview_exception_wraps_in_workflow_error_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="anonymizer.ndd")

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

    assert isinstance(exc_info.value.__cause__, DataDesignerRuntimeError)
    assert "endpoint unreachable" in str(exc_info.value)

    warning_records = _unique_records(caplog, level=logging.WARNING, message_contains="Workflow failed")
    assert len(warning_records) == 1
    warning_msg = warning_records[0].getMessage()
    assert "3" in warning_msg
    assert "test-model-alias" not in warning_msg
    assert "endpoint unreachable" not in warning_msg
    assert "Check endpoint reachability" not in warning_msg
    assert "detect-workflow" not in warning_msg
    _assert_no_backend_reference(warning_msg)

    debug_records = _unique_records(caplog, level=logging.DEBUG, message_contains="failure context")
    assert len(debug_records) == 1
    debug_msg = debug_records[0].getMessage()
    assert "detect-workflow" not in debug_msg
    assert "output" not in debug_msg
    _assert_no_backend_reference(debug_msg)


def test_create_exception_wraps_in_workflow_error_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="anonymizer.ndd")

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

    assert isinstance(exc_info.value.__cause__, DataDesignerRuntimeError)
    assert "quota exceeded" in str(exc_info.value)

    warning_records = _unique_records(caplog, level=logging.WARNING, message_contains="Workflow failed")
    assert len(warning_records) == 1
    warning_msg = warning_records[0].getMessage()
    assert "2" in warning_msg
    assert "test-model-alias" not in warning_msg
    assert "quota exceeded" not in warning_msg
    assert "Check endpoint reachability" not in warning_msg
    assert "replace-workflow" not in warning_msg
    _assert_no_backend_reference(warning_msg)

    debug_records = _unique_records(caplog, level=logging.DEBUG, message_contains="failure context")
    assert len(debug_records) == 1
    debug_msg = debug_records[0].getMessage()
    assert "replace-workflow" not in debug_msg
    assert "output" not in debug_msg
    _assert_no_backend_reference(debug_msg)


def test_load_dataset_exception_wraps_in_workflow_error_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="anonymizer.ndd")

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

    assert isinstance(exc_info.value.__cause__, DataDesignerRuntimeError)
    assert "corrupt parquet" in str(exc_info.value)

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

    debug_records = _unique_records(caplog, level=logging.DEBUG, message_contains="failure context")
    assert len(debug_records) == 1
    debug_msg = debug_records[0].getMessage()
    assert "replace-workflow" not in debug_msg
    assert "output" not in debug_msg
    _assert_no_backend_reference(debug_msg)


def test_workflow_failure_warning_names_exception_class_but_not_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="anonymizer.ndd")

    class EndpointUnreachableError(Exception):
        pass

    mock_dd = Mock(spec=DataDesigner)
    mock_dd.create.side_effect = EndpointUnreachableError("SENTINEL-RECORD-TEXT")

    adapter = NddAdapter(data_designer=mock_dd)
    input_df = pd.DataFrame({"text": ["row-1", "row-2"]})

    with pytest.raises(AnonymizerWorkflowError):
        adapter.run_workflow(
            input_df,
            model_configs=[_make_model_config()],
            columns=_make_columns(),
            workflow_name="replace-workflow",
            preview_num_records=None,
        )

    warning_records = _unique_records(caplog, level=logging.WARNING, message_contains="Workflow failed")
    assert len(warning_records) == 1
    warning_msg = warning_records[0].getMessage()
    assert "EndpointUnreachableError" in warning_msg
    assert "SENTINEL-RECORD-TEXT" not in caplog.text


_BACKEND_EXCEPTION_LABELS = [
    (DataDesignerError, "BackendError"),
    (DataDesignerGenerationError, "GenerationError"),
    (DataDesignerRuntimeError, "RuntimeError"),
    (DataDesignerWorkflowError, "WorkflowError"),
    (DataDesignerEarlyShutdownError, "EarlyShutdownError"),
    (DataDesignerProfilingError, "ProfilingError"),
]


@pytest.mark.parametrize(("exc_class", "label"), _BACKEND_EXCEPTION_LABELS)
def test_workflow_failure_warning_shortens_backend_exception_class(
    caplog: pytest.LogCaptureFixture, exc_class: type[Exception], label: str
) -> None:
    caplog.set_level(logging.DEBUG, logger="anonymizer.ndd")
    mock_dd = Mock(spec=DataDesigner)
    mock_dd.create.side_effect = exc_class("SENTINEL-RECORD-TEXT")

    adapter = NddAdapter(data_designer=mock_dd)
    with pytest.raises(AnonymizerWorkflowError):
        adapter.run_workflow(
            pd.DataFrame({"text": ["row-1"]}),
            model_configs=[_make_model_config()],
            columns=_make_columns(),
            workflow_name="replace-workflow",
            preview_num_records=None,
        )

    warning_records = _unique_records(caplog, level=logging.WARNING, message_contains="Workflow failed")
    assert len(warning_records) == 1
    warning_msg = warning_records[0].getMessage()
    assert warning_msg.endswith(f": {label}.")
    _assert_no_backend_reference(warning_msg)
    assert "SENTINEL-RECORD-TEXT" not in caplog.text


def test_run_workflow_on_no_rows_returns_the_declared_columns_without_running() -> None:
    """A stage after one that stopped early before any row finished runs on nothing."""
    mock_dd = Mock(spec=DataDesigner)
    adapter = NddAdapter(data_designer=mock_dd)

    result = adapter.run_workflow(
        pd.DataFrame({"text": pd.Series([], dtype=object)}),
        model_configs=[_make_model_config()],
        columns=_make_columns(),
        workflow_name="rewrite-evaluate-0",
    )

    mock_dd.create.assert_not_called()
    assert result.dataframe.empty
    assert {"text", RECORD_ID_COLUMN, "output"} <= set(result.dataframe.columns)
    assert result.failed_records == []


_BUG = "Custom generator function failed for column '_later': TypeError: bug in a later column"


def _trace(row: int, column: str, error: str, status: str = "error") -> SimpleNamespace:
    return SimpleNamespace(status=status, row_group=0, row_index=row, column=column, error=error)


def _trip() -> RunStop:
    stop = _RUN_STOP.get()
    assert stop is not None
    for _ in range(stop.window):
        stop.record("_domain", True)
    return stop


def _stop_refusing(monkeypatch: pytest.MonkeyPatch, refused: list[int], dropped: dict[int, str] | None = None) -> Mock:
    """A designer whose run stops ``_domain`` and has it refuse the rows at ``refused``; DataDesigner keeps every row
    but those in ``dropped``, whose traces record the given error."""
    frames: list[WorkflowSourceFrame] = []
    capture = WorkflowSourceFrame.capture

    def capturing(dataframe: pd.DataFrame) -> WorkflowSourceFrame:
        frames.append(capture(dataframe))
        return frames[-1]

    monkeypatch.setattr(ndd_adapter.WorkflowSourceFrame, "capture", capturing)
    dropped = dropped or {}

    def create(*args: object, **kwargs: object) -> Mock:
        stop = _trip()
        seed = frames[-1].seed()
        for position in refused:
            assert stop.refuses("_domain", seed.iloc[position].to_dict())
        results = Mock()
        results.load_dataset.return_value = seed.drop(index=list(dropped)).assign(output="out")
        results.task_traces = [_trace(position, "_later", error) for position, error in dropped.items()]
        return results

    mock_dd = Mock(spec=DataDesigner)
    mock_dd.create.side_effect = create
    return mock_dd


def _run_two_rows(mock_dd: Mock) -> ndd_adapter.WorkflowRunResult:
    return NddAdapter(data_designer=mock_dd).run_workflow(
        pd.DataFrame({"text": ["row-1", "row-2"]}),
        model_configs=[_make_model_config()],
        columns=_make_columns(),
        workflow_name="rewrite-pipeline",
    )


def test_run_workflow_returns_the_rows_past_a_stopped_step_and_lists_the_refused_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_two_rows(_stop_refusing(monkeypatch, refused=[1]))

    assert result.dataframe["text"].tolist() == ["row-1"]
    assert result.source_positions == (0,)
    assert [(r.step, r.reason) for r in result.failed_records] == [("rewrite-pipeline", "stopped_early (_domain)")]
    assert _RUN_STOP.get() is None  # the stop belongs to that run only


def test_run_workflow_lists_every_row_when_the_stop_refused_them_all(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_two_rows(_stop_refusing(monkeypatch, refused=[0, 1]))

    assert result.dataframe.empty
    assert "output" in result.dataframe.columns
    assert [r.reason for r in result.failed_records] == ["stopped_early (_domain)"] * 2


def test_run_workflow_raises_when_no_row_is_returned_and_one_failed_for_another_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stop refused one row and a bug dropped the other: an error the stop did not cause still fails the run."""
    with pytest.raises(AnonymizerWorkflowError, match="no row was returned"):
        _run_two_rows(_stop_refusing(monkeypatch, refused=[0], dropped={1: _BUG}))


def test_run_workflow_lists_a_row_dropped_for_another_reason_by_its_own_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_two_rows(_stop_refusing(monkeypatch, refused=[], dropped={1: _BUG}))

    assert result.dataframe["text"].tolist() == ["row-1"]
    assert [r.reason for r in result.failed_records] == [
        "Record missing from workflow output: _later: generation failed"
    ]


def _raise_after_stop(error: Exception) -> object:
    def create(*args: object, **kwargs: object) -> None:
        _trip()
        raise error

    return create


def _load_fails_after_stop(*args: object, **kwargs: object) -> object:
    _trip()
    results = Mock()
    results.load_dataset.side_effect = KeyError("adapter-side bug")
    results.task_traces = []
    return results


@pytest.mark.parametrize(
    "create",
    [
        pytest.param(_raise_after_stop(ValueError("bug in a later column")), id="value-error"),
        pytest.param(_raise_after_stop(DataDesignerProfilingError("profiling failed")), id="profiling-error"),
        pytest.param(_raise_after_stop(DataDesignerEarlyShutdownError("early shutdown")), id="shutdown"),
        pytest.param(_load_fails_after_stop, id="load-dataset-key-error"),
    ],
)
def test_run_workflow_raises_an_error_the_stop_did_not_cause(create: object) -> None:
    mock_dd = Mock(spec=DataDesigner)
    mock_dd.create.side_effect = create

    with pytest.raises(AnonymizerWorkflowError):
        _run_two_rows(mock_dd)
