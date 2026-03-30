# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from unittest.mock import Mock

import pandas as pd
import pytest

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
    _replace_df.attrs["original_text_column"] = "text"
    replace_runner = Mock(spec=ReplacementWorkflow)
    replace_runner.run.return_value = ReplacementResult(
        dataframe=_replace_df,
        failed_records=[FailedRecord(record_id="r2", step="replace", reason="parse error")],
    )
    return Anonymizer(detection_workflow=detection_workflow, replace_runner=replace_runner)


@pytest.fixture
def debug_messages(
    tmp_path: pytest.TempPathFactory,
    _stub_anonymizer: Anonymizer,
    caplog: pytest.LogCaptureFixture,
) -> list[str]:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme", "Bob likes cats"]}).to_csv(csv_path, index=False)
    config = AnonymizerConfig(replace=Redact())
    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        _stub_anonymizer.run(config=config, data=AnonymizerInput(source=str(csv_path)))
    return [r.message for r in caplog.records if r.levelno == logging.DEBUG]


@pytest.mark.parametrize(
    "expected_substring",
    [
        "r1",
        "r2",
        "input text lengths:",
        "detection config: threshold=",
    ],
)
def test_debug_log_contains(debug_messages: list[str], expected_substring: str) -> None:
    assert any(expected_substring in m for m in debug_messages)
