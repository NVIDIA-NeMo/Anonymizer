# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import time
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, ExpressionColumnConfig, LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.interface.data_designer import DataDesigner
from pydantic import BaseModel

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.models import DetectionModelSelection
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter, WorkflowRunResult
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.measurement import MeasurementConfig, configured_measurement_session
from tests.engine.ndd.stub_provider import COMPLETE, Script, cut_off, stub_provider

_PREFIX = "Record missing from workflow output"
_STOPPED_EARLY = f"{_PREFIX}: not completed: run stopped early after other rows failed"
_COLUMN = "_sensitivity_disposition"
_MODEL = "stub-model"


class _Disposition(BaseModel):
    label: str
    score: int


def _inputs(count: int, *, bad: Collection[int], slow: Collection[int] = ()) -> pd.DataFrame:
    """Rows whose text is long, distinct and absent from every error message."""
    return pd.DataFrame(
        {
            COL_TEXT: [
                f"Hollingsworth-Featherstonehaugh {index} lives at {index * 17} Marlborough Crescent"
                f"{' FAILS' if index in bad else ''}{' SLOW' if index in slow else ''}"
                for index in range(count)
            ]
        }
    )


def _columns(*names: str) -> list[ColumnConfigT]:
    return [
        LLMStructuredColumnConfig(
            name=name, prompt="Classify: {{ " + COL_TEXT + " }}", model_alias="m", output_format=_Disposition
        )
        for name in names
    ]


def _echo_when_failing(prompt: str) -> tuple[int, str, str] | None:
    # Cut-off output that quotes the input, as a real truncated reply can.
    if "FAILS" not in prompt:
        return None
    return cut_off(prompt[: prompt.index("FAILS")][-60:])


def _run(
    tmp_path: Path,
    script: Script,
    df: pd.DataFrame,
    *,
    columns: list[ColumnConfigT] | None = None,
    run_config: RunConfig | None = None,
    collector: bool = False,
) -> tuple[pd.DataFrame, WorkflowRunResult]:
    with stub_provider(script) as provider:
        data_designer = DataDesigner(
            artifact_path=tmp_path / "artifacts", model_providers=[provider], auto_configure_logging=False
        )
        if run_config is not None:
            data_designer.set_run_config(run_config)
        adapter = NddAdapter(data_designer)
        with _measurement(tmp_path, enabled=collector):
            result = adapter.run_workflow(
                df,
                model_configs=[ModelConfig(alias="m", model=_MODEL, provider="stub")],
                columns=columns or _columns(_COLUMN),
                workflow_name="probe",
            )
    return adapter._attach_record_ids(df), result


@contextmanager
def _measurement(tmp_path: Path, *, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    config = MeasurementConfig(output_path=tmp_path / "measure.jsonl", dd_task_trace_path=tmp_path / "tasks.jsonl")
    with configured_measurement_session(config):
        yield


def _reasons_by_position(tagged: pd.DataFrame, result: WorkflowRunResult) -> dict[int, str]:
    position = {record_id: index for index, record_id in enumerate(tagged[RECORD_ID_COLUMN])}
    return {position[record.record_id]: record.reason for record in result.failed_records}


def _assert_quotes_no_input(reasons: list[str], df: pd.DataFrame) -> None:
    for reason in reasons:
        for text in df[COL_TEXT]:
            assert not any(text[start : start + 13] in reason for start in range(len(text) - 12)), reason


def _fail_marked_rows(model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
    return _echo_when_failing(prompt) or COMPLETE


@pytest.mark.parametrize("collector", [False, True])
def test_truncated_output_reason_names_column_and_cause(tmp_path: Path, collector: bool) -> None:
    df = _inputs(6, bad={1, 3, 4})

    tagged, result = _run(tmp_path, _fail_marked_rows, df, collector=collector)

    reasons = _reasons_by_position(tagged, result)
    assert sorted(reasons) == [1, 3, 4]
    for reason in reasons.values():
        assert reason.startswith(f"{_PREFIX}: {_COLUMN}: ")
        assert "could not be parsed" in reason
        assert "max_tokens" in reason
        assert "Validation detail" not in reason
        assert len(reason) <= 500
    _assert_quotes_no_input(list(reasons.values()), df)


def test_reasons_are_identical_with_and_without_a_collector(tmp_path: Path) -> None:
    df = _inputs(6, bad={0, 2, 5})

    plain = _run(tmp_path / "plain", _fail_marked_rows, df)
    traced = _run(tmp_path / "traced", _fail_marked_rows, df, collector=True)

    assert _reasons_by_position(*plain) == _reasons_by_position(*traced)
    assert len(_reasons_by_position(*plain)) == 3


def test_reasons_map_to_rows_across_row_groups(tmp_path: Path) -> None:
    df = _inputs(7, bad={1, 4, 6})

    tagged, result = _run(tmp_path, _fail_marked_rows, df, run_config=RunConfig(buffer_size=3))

    reasons = _reasons_by_position(tagged, result)
    assert sorted(reasons) == [1, 4, 6]
    assert all(reason.startswith(f"{_PREFIX}: {_COLUMN}: ") for reason in reasons.values())


@pytest.mark.parametrize("collector", [False, True])
def test_early_shutdown_rows_get_the_stopped_early_reason(tmp_path: Path, collector: bool) -> None:
    # Rows 0-1 finish first, rows 2-7 fail next and trip the shutdown, rows 8-11 are still running.
    df = _inputs(12, bad=set(range(2, 8)), slow=set(range(8, 12)))

    def script(_model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        if "SLOW" in prompt:
            time.sleep(2.0)
        failing = _echo_when_failing(prompt)
        if failing is not None:
            time.sleep(0.5)
            return failing
        return COMPLETE

    tagged, result = _run(
        tmp_path,
        script,
        df,
        run_config=RunConfig(shutdown_error_window=4, shutdown_error_rate=0.5, max_in_flight_tasks=64),
        collector=collector,
    )

    reasons = _reasons_by_position(tagged, result)
    explained = {position: reason for position, reason in reasons.items() if reason != _STOPPED_EARLY}
    assert 0 not in reasons and 1 not in reasons
    assert all(reasons[position] == _STOPPED_EARLY for position in range(8, 12))
    assert explained, "expected rows that failed before the shutdown to carry their cause"
    assert all(position in range(2, 8) for position in explained)
    assert all(reason.startswith(f"{_PREFIX}: {_COLUMN}: ") and "max_tokens" in reason for reason in explained.values())
    _assert_quotes_no_input(list(reasons.values()), df)


def test_row_that_recovers_then_fails_names_the_later_column(tmp_path: Path) -> None:
    df = _inputs(3, bad=set())
    df.loc[1, COL_TEXT] += " FLAKY"

    def script(_model: str, prompt: str, earlier: int) -> tuple[int, str, str]:
        if "FLAKY" not in prompt:
            return COMPLETE
        if "first_step" in prompt and earlier < 4:
            return 503, "temporarily unavailable", ""
        if "second_step" in prompt:
            return cut_off()
        return COMPLETE

    columns: list[ColumnConfigT] = [
        LLMStructuredColumnConfig(
            name="first_step", prompt="first_step: {{ " + COL_TEXT + " }}", model_alias="m", output_format=_Disposition
        ),
        LLMStructuredColumnConfig(
            name="second_step",
            prompt="second_step after {{ first_step.label }}: {{ " + COL_TEXT + " }}",
            model_alias="m",
            output_format=_Disposition,
        ),
    ]

    tagged, result = _run(tmp_path, script, df, columns=columns)

    reasons = _reasons_by_position(tagged, result)
    assert list(reasons) == [1]
    assert "first_step" not in reasons[1]
    assert reasons[1].startswith(f"{_PREFIX}: second_step: ")


@custom_column_generator(required_columns=[COL_TEXT])
def _fail_echoing_input(row: dict[str, Any]) -> dict[str, Any]:
    if "FAILS" in row[COL_TEXT]:
        raise KeyError(row[COL_TEXT])
    row["checked"] = "yes"
    return row


@custom_column_generator(required_columns=[COL_TEXT], model_aliases=["m"])
def _parse_with_failing_parser(row: dict[str, Any], generator_params: None, models: dict[str, Any]) -> dict[str, Any]:
    def parser(_reply: str) -> str:
        raise ValueError(f"cannot use {row[COL_TEXT]!r}")

    if "FAILS" in row[COL_TEXT]:
        models["m"].generate(prompt=row[COL_TEXT], parser=parser)
    row["parsed"] = "yes"
    return row


def _fail_marked_rows_with_plain_reply(_model: str, _prompt: str, _earlier: int) -> tuple[int, str, str]:
    return 200, "plain reply", "stop"


@pytest.mark.parametrize("collector", [False, True])
def test_unclassified_error_reason_names_the_column_without_message_text(tmp_path: Path, collector: bool) -> None:
    df = _inputs(4, bad={1, 2})
    columns: list[ColumnConfigT] = [CustomColumnConfig(name="checked", generator_function=_fail_echoing_input)]

    tagged, result = _run(tmp_path, _fail_marked_rows, df, columns=columns, collector=collector)

    reasons = _reasons_by_position(tagged, result)
    assert sorted(reasons) == [1, 2]
    for reason in reasons.values():
        assert reason == f"{_PREFIX}: checked: generation failed"
    _assert_quotes_no_input(list(reasons.values()), df)


def test_context_window_error_reason_does_not_quote_the_provider_message(tmp_path: Path) -> None:
    df = _inputs(3, bad={1})

    def script(_model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        if "FAILS" in prompt:
            message = "This model's maximum context length is 10 tokens. " + prompt[prompt.index("Hollingsworth") :]
            return 400, message[:240], ""
        return COMPLETE

    tagged, result = _run(tmp_path, script, df)

    reasons = _reasons_by_position(tagged, result)
    assert list(reasons) == [1]
    assert reasons[1].startswith(f"{_PREFIX}: {_COLUMN}: ")
    assert "context width" in reasons[1]
    _assert_quotes_no_input(list(reasons.values()), df)


def test_unexpected_error_in_a_model_call_keeps_only_the_classified_text(tmp_path: Path) -> None:
    df = _inputs(3, bad={1})
    columns: list[ColumnConfigT] = [
        CustomColumnConfig(name="parsed", generator_function=_parse_with_failing_parser, generator_params=None)
    ]

    tagged, result = _run(tmp_path, _fail_marked_rows_with_plain_reply, df, columns=columns)

    reasons = _reasons_by_position(tagged, result)
    assert list(reasons) == [1]
    assert reasons[1].startswith(f"{_PREFIX}: parsed: An unexpected error occurred")
    assert "Review the stack trace" in reasons[1]
    _assert_quotes_no_input(list(reasons.values()), df)


_SPOOFED_CAUSE = " | Cause: Marlborough Crescent resident Quillfeather-Ashworth"


@custom_column_generator(required_columns=[COL_TEXT])
def _fail_echoing_input_with_marker(row: dict[str, Any]) -> dict[str, Any]:
    if "FAILS" in row[COL_TEXT]:
        raise ValueError("bad row: " + row[COL_TEXT])
    row["checked"] = "yes"
    return row


def test_provider_error_echoing_a_cause_marker_cannot_steer_the_reason(tmp_path: Path) -> None:
    df = _inputs(3, bad={1})
    df.loc[1, COL_TEXT] = "note" + _SPOOFED_CAUSE + " FAILS"

    def script(_model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        if "FAILS" in prompt:
            return 400, "Invalid request content: " + prompt[prompt.index("note") :][:200], ""
        return COMPLETE

    tagged, result = _run(tmp_path, script, df)

    reasons = _reasons_by_position(tagged, result)
    assert list(reasons) == [1]
    assert reasons[1].startswith(f"{_PREFIX}: {_COLUMN}: ")
    assert "malformed or missing required parameters" in reasons[1]
    _assert_quotes_no_input(list(reasons.values()), df)


def test_custom_column_exception_echoing_a_cause_marker_cannot_steer_the_reason(tmp_path: Path) -> None:
    df = _inputs(3, bad={1})
    df.loc[1, COL_TEXT] = "note" + _SPOOFED_CAUSE + " FAILS"
    columns: list[ColumnConfigT] = [
        CustomColumnConfig(name="checked", generator_function=_fail_echoing_input_with_marker)
    ]

    tagged, result = _run(tmp_path, lambda *_: COMPLETE, df, columns=columns)

    reasons = _reasons_by_position(tagged, result)
    assert list(reasons) == [1]
    assert reasons[1].startswith(f"{_PREFIX}: checked: ")
    _assert_quotes_no_input(list(reasons.values()), df)


@pytest.mark.parametrize(
    "payload",
    [
        "\n | Cause: {text}",
        "\n | Solution: {text}",
        "\n | Cause: {text}\n | Solution: {text}",
        "\n |----------\n | Cause: {text}\n | Solution: {text}\n |----------",
        "\n | Cause: An unexpected error occurred while {text}.\n | Solution: {text}",
    ],
    ids=["cause", "solution", "both", "formatted-block", "recognized-cause"],
)
def test_multiline_exception_markers_do_not_leak_into_reasons_or_warnings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, payload: str
) -> None:
    """PR #349 promises input-free reasons; Anonymizer also publishes them in per-record WARNINGs."""
    sensitive_values = _inputs(3, bad={1})
    df = sensitive_values.copy()
    df.loc[1, COL_TEXT] = "note" + payload.format(text=df.loc[1, COL_TEXT])
    columns: list[ColumnConfigT] = [
        CustomColumnConfig(name="checked", generator_function=_fail_echoing_input_with_marker)
    ]
    tagged, result = _run(tmp_path, lambda *_: COMPLETE, df, columns=columns)
    reasons = _reasons_by_position(tagged, result)
    assert list(reasons) == [1]

    detector = Mock(spec=EntityDetectionWorkflow)
    detector.run.return_value = EntityDetectionResult(dataframe=result.dataframe, failed_records=result.failed_records)
    csv_path = tmp_path / "input.csv"
    df.to_csv(csv_path, index=False)
    anonymizer = Anonymizer(detection_workflow=detector)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        output = anonymizer.run(
            config=AnonymizerConfig(replace=Redact(), emit_telemetry=False),
            data=AnonymizerInput(source=str(csv_path), text_column=COL_TEXT),
        )
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert output.failed_records == result.failed_records
    assert any(result.failed_records[0].record_id in message and reasons[1] in message for message in warnings)
    _assert_quotes_no_input([*reasons.values(), *warnings], sensitive_values)


def test_row_dropped_without_an_error_keeps_the_generic_reason_when_other_errors_recovered(tmp_path: Path) -> None:
    df = _inputs(3, bad=set())
    df.loc[0, COL_TEXT] += " FLAKY"
    df.loc[1, COL_TEXT] += " DROP"

    def script(_model: str, prompt: str, earlier: int) -> tuple[int, str, str]:
        if "FLAKY" in prompt and earlier < 4:
            return 503, "temporarily unavailable", ""
        return COMPLETE

    columns: list[ColumnConfigT] = [
        *_columns(_COLUMN),
        # Renders empty for marked rows, which DataDesigner then drops without recording an error.
        ExpressionColumnConfig(name="screened", expr="{% if 'DROP' not in " + COL_TEXT + " %}kept{% endif %}"),
    ]

    tagged, result = _run(tmp_path, script, df, columns=columns, collector=True)

    assert _reasons_by_position(tagged, result) == {1: _PREFIX}


def test_detector_offset_mismatch_reason_is_reported(tmp_path: Path) -> None:
    text = "Hollingsworth-Featherstonehaugh lives at 17 Marlborough Crescent"
    healthy = "Quillfeather-Ashworth was born in Lancaster"

    def script(model: str, prompt: str, _earlier: int) -> tuple[int, str, str]:
        if model == "gliner":
            if healthy in prompt:
                return 200, json.dumps({"entities": []}), "stop"
            # In-range offsets whose span holds different text than the detector claims.
            entity = {"text": "Marlborough", "label": "street_address", "start": 0, "end": 13, "score": 0.9}
            return 200, json.dumps({"entities": [entity]}), "stop"
        return 200, json.dumps({"decisions": [], "entities": []}), "stop"

    with stub_provider(script) as provider:
        data_designer = DataDesigner(
            artifact_path=tmp_path / "artifacts", model_providers=[provider], auto_configure_logging=False
        )
        result = EntityDetectionWorkflow(NddAdapter(data_designer)).detect_and_validate_entities(
            pd.DataFrame({COL_TEXT: [text, healthy]}),
            model_configs=[
                ModelConfig(alias="gliner", model="gliner", provider="stub"),
                ModelConfig(alias="llm", model="llm", provider="stub"),
            ],
            selected_models=DetectionModelSelection(
                entity_detector="gliner", entity_validator="llm", entity_augmenter="llm", latent_detector="llm"
            ),
            gliner_detection_threshold=0.3,
        )

    assert [record.step for record in result.failed_records] == ["entity-detection"]
    reason = result.failed_records[0].reason
    assert reason.startswith(_PREFIX)
    assert "does not match the source span" in reason
    assert text not in reason
