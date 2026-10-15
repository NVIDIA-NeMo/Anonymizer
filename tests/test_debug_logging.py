# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pandas as pd
import pytest

import anonymizer.logging as anonymizer_logging
from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.logging import LoggingConfig, configure_logging


@pytest.fixture(autouse=True)
def _enable_debug() -> None:
    configure_logging(LoggingConfig.debug())


@pytest.fixture
def _stub_anonymizer() -> Anonymizer:
    entities = [
        [{"value": "Alice", "label": "first_name"}, {"value": "Acme", "label": "organization"}],
        [{"value": "Bob", "label": "first_name"}],
    ]
    det_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme", "Bob likes cats"],
            COL_DETECTED_ENTITIES: entities,
            COL_FINAL_ENTITIES: entities,
        }
    )
    detection_workflow = Mock(spec=EntityDetectionWorkflow)
    detection_workflow.run.return_value = EntityDetectionResult(
        dataframe=det_df,
        failed_records=[
            FailedRecord(record_id="r1", step="detection", reason="timeout"),
        ],
    )
    _replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme", "Bob likes cats"],
            COL_REPLACED_TEXT: ["[REDACTED] works at [REDACTED]", "[REDACTED] likes cats"],
        }
    )
    replace_runner = Mock(spec=ReplacementWorkflow)
    replace_runner.run.return_value = ReplacementResult(
        dataframe=_replace_df,
        failed_records=[FailedRecord(record_id="r2", step="replace", reason="parse error")],
    )
    return Anonymizer(detection_workflow=detection_workflow, replace_runner=replace_runner)


@pytest.fixture
def debug_messages(
    tmp_path: Path,
    _stub_anonymizer: Anonymizer,
    caplog: pytest.LogCaptureFixture,
) -> list[str]:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme", "Bob likes cats"]}).to_csv(csv_path, index=False)
    config = AnonymizerConfig(replace=Redact())
    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        result = _stub_anonymizer.run(config=config, data=AnonymizerInput(source=str(csv_path)))
    assert [(f.record_id, f.reason) for f in result.failed_records] == [("r1", "timeout"), ("r2", "parse error")]
    return [r.message for r in caplog.records if r.levelno == logging.DEBUG]


@pytest.mark.parametrize(
    "expected_substring",
    [
        "r1",
        "r2",
        "input text lengths:",
        "detection config: threshold=",
        "configured_label_count=",
        "effective_label_count=",
        "removed_by_exclusions=",
    ],
)
def test_debug_log_contains(debug_messages: list[str], expected_substring: str) -> None:
    if expected_substring in {"r1", "r2"}:
        # FailedRecord details remain public results, but are excluded from logs.
        assert all(expected_substring not in m for m in debug_messages)
    else:
        assert any(expected_substring in m for m in debug_messages)


@pytest.mark.parametrize("preset", ["default", "debug"])
def test_public_pipeline_returns_failed_record_details_without_logging_them(
    preset: str, _stub_anonymizer: Anonymizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    canary = "SENSITIVE_FACADE_CANARY_d492"
    stream = io.StringIO()
    loggers = [logging.getLogger(name) for name in ("anonymizer", "data_designer")]
    previous = [(logger.handlers[:], logger.level, logger.propagate) for logger in loggers]
    for field in ("_active_config", "_configured", "_anonymizer_handler"):
        monkeypatch.setattr(anonymizer_logging, field, getattr(anonymizer_logging, field))
    monkeypatch.setattr("sys.stderr", stream)
    configure_logging(getattr(LoggingConfig, preset)())
    failure = FailedRecord(record_id=canary, step=canary, reason=canary)
    detection = cast(Mock, _stub_anonymizer._detection_workflow.run).return_value
    detection.failed_records[:] = [failure]
    detection.dataframe[COL_DETECTED_ENTITIES] = [[{"value": "Alice", "label": canary}], []]
    cast(Mock, _stub_anonymizer._replace_runner.run).return_value.failed_records[:] = [failure]
    source = tmp_path / (canary + ".csv")
    pd.DataFrame({canary: ["Alice works at Acme", "Bob likes cats"]}).to_csv(source, index=False)
    try:
        result = _stub_anonymizer.run(
            config=AnonymizerConfig(detect={"entity_labels": [canary]}, replace=Redact()),
            data=AnonymizerInput(source=str(source), text_column=canary),
        )
    finally:
        for logger, (handlers, level, propagate) in zip(loggers, previous, strict=True):
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate
    assert [(f.record_id, f.step, f.reason) for f in result.failed_records] == [(canary, canary, canary)] * 2
    assert result.trace_dataframe[canary].tolist() == ["Alice works at Acme", "Bob likes cats"]
    assert "2 total failures" in stream.getvalue()
    assert canary.lower() not in stream.getvalue().lower()
