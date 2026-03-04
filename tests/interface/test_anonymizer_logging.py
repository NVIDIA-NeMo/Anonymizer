# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.interface.anonymizer import Anonymizer


@pytest.fixture
def stub_input(tmp_path: Path) -> AnonymizerInput:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme", "Bob likes cats"]}).to_csv(csv_path, index=False)
    return AnonymizerInput(source=str(csv_path))


def _make_logging_anonymizer(
    *,
    detection_entities: list[list[dict]] | None = None,
    detection_failures: list[FailedRecord] | None = None,
    replace_failures: list[FailedRecord] | None = None,
) -> Anonymizer:
    entities = detection_entities or [
        [{"value": "Alice", "label": "first_name"}, {"value": "Acme", "label": "organization"}],
        [{"value": "Bob", "label": "first_name"}],
    ]
    det_df = pd.DataFrame({
        COL_TEXT: ["Alice works at Acme", "Bob likes cats"],
        COL_DETECTED_ENTITIES: entities,
        COL_FINAL_ENTITIES: entities,
    })
    detection_workflow = Mock(spec=EntityDetectionWorkflow)
    detection_workflow.run.return_value = EntityDetectionResult(
        dataframe=det_df,
        failed_records=detection_failures or [],
    )
    replace_runner = Mock(spec=ReplacementWorkflow)
    replace_runner.run.return_value = ReplacementResult(
        dataframe=pd.DataFrame({
            COL_TEXT: ["Alice works at Acme", "Bob likes cats"],
            COL_REPLACED_TEXT: ["[REDACTED] works at [REDACTED]", "[REDACTED] likes cats"],
        }),
        failed_records=replace_failures or [],
    )
    return Anonymizer(detection_workflow=detection_workflow, replace_runner=replace_runner)


def test_run_logs_pipeline_stages(stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture) -> None:
    anonymizer = _make_logging_anonymizer()
    config = AnonymizerConfig(replace=Redact())

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.run(config=config, data=stub_input)

    messages = caplog.text
    assert "📂 Loaded 2 records from" in messages
    assert "🔍 Running entity detection on 2 records" in messages
    assert "✅ Detection complete" in messages
    assert "3 entities" in messages
    assert "🔄 Running Redact replacement" in messages
    assert "✅ Replacement complete" in messages
    assert "🎉 Pipeline complete" in messages
    assert "2 records processed" in messages


def test_run_logs_failure_counts(stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture) -> None:
    anonymizer = _make_logging_anonymizer(
        detection_failures=[FailedRecord(record_id="r1", step="detection", reason="timeout")],
        replace_failures=[FailedRecord(record_id="r2", step="replace", reason="parse error")],
    )
    config = AnonymizerConfig(replace=Redact())

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.run(config=config, data=stub_input)

    messages = caplog.text
    assert "✅ Detection complete" in messages
    assert "1 failed" in messages
    assert "🎉 Pipeline complete" in messages
    assert "2 total failures" in messages


def test_run_without_replacement_skips_replace_logs(
    stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture
) -> None:
    anonymizer = _make_logging_anonymizer()
    config = AnonymizerConfig(rewrite=Rewrite())

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.run(config=config, data=stub_input)

    messages = caplog.text
    assert "🔍 Running entity detection" in messages
    assert "🔄 Running Redact replacement" not in messages
    assert "✅ Replacement complete" not in messages
    assert "🎉 Pipeline complete" in messages


def test_preview_logs_preview_mode(stub_input: AnonymizerInput, caplog: pytest.LogCaptureFixture) -> None:
    anonymizer = _make_logging_anonymizer()
    config = AnonymizerConfig(replace=Redact())

    with caplog.at_level(logging.INFO, logger="anonymizer"):
        anonymizer.preview(config=config, data=stub_input, num_records=5)

    messages = caplog.text
    assert "👀 Preview mode: processing 5 of 2 records" in messages
