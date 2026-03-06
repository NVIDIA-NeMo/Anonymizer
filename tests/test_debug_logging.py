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


def _make_anonymizer(
    *,
    detection_failures: list[FailedRecord] | None = None,
    replace_failures: list[FailedRecord] | None = None,
) -> Anonymizer:
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
        failed_records=detection_failures or [],
    )
    replace_runner = Mock(spec=ReplacementWorkflow)
    replace_runner.run.return_value = ReplacementResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme", "Bob likes cats"],
                COL_REPLACED_TEXT: ["[REDACTED] works at [REDACTED]", "[REDACTED] likes cats"],
            }
        ),
        failed_records=replace_failures or [],
    )
    return Anonymizer(detection_workflow=detection_workflow, replace_runner=replace_runner)


def test_debug_logs_entity_counts(tmp_path: pytest.TempPathFactory, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme", "Bob likes cats"]}).to_csv(csv_path, index=False)
    anonymizer = _make_anonymizer()
    config = AnonymizerConfig(replace=Redact())

    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        anonymizer.run(config=config, data=AnonymizerInput(source=str(csv_path)))

    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("entities per record" in m for m in debug_messages)
    assert any("record 0:" in m and "first_name=1" in m and "organization=1" in m for m in debug_messages)
    assert any("record 1:" in m and "first_name=1" in m for m in debug_messages)


def test_debug_logs_failed_record_details(tmp_path: pytest.TempPathFactory, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme", "Bob likes cats"]}).to_csv(csv_path, index=False)
    anonymizer = _make_anonymizer(
        detection_failures=[FailedRecord(record_id="r1", step="detection", reason="timeout")],
        replace_failures=[FailedRecord(record_id="r2", step="replace", reason="parse error")],
    )
    config = AnonymizerConfig(replace=Redact())

    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        anonymizer.run(config=config, data=AnonymizerInput(source=str(csv_path)))

    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("failed records:" in m for m in debug_messages)
    assert any("r1" in m and "detection" in m for m in debug_messages)
    assert any("r2" in m and "replace" in m for m in debug_messages)
