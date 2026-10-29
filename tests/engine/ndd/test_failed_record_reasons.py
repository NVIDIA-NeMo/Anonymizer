# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.interface.data_designer import DataDesigner
from pydantic import BaseModel

from anonymizer.config.models import DetectionModelSelection
from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter, WorkflowRunResult
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
