# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_REWRITTEN_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
)
from anonymizer.engine.detection.result import DetectorResult
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.interface.anonymizer import Anonymizer
from tests.detector_results import as_detector_result, input_record_ids


@pytest.mark.parametrize("rejected", [[0], [0, 1]])
def test_rewrite_reports_rows_that_fail_detection(
    tmp_path: Path, rejected: list[int], caplog: pytest.LogCaptureFixture
) -> None:
    texts = ["Alice Smith lives here", "Bob Jones is here"]
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": texts}).to_csv(csv_path, index=False)

    def entities(row: int) -> dict:
        name, last, verb = texts[row].split()[:3]
        full, tail = f"{name} {last}", f"{last} {verb}"
        spans = [{"id": "a", "value": full, "label": "full_name", "start_position": 0, "end_position": len(full)}]
        if row in rejected:  # two partially overlapping spans: the graph rejects the row's mentions at detect
            start = len(name) + 1
            spans.append(
                {"id": "b", "value": tail, "label": "other", "start_position": start, "end_position": start + len(tail)}
            )
        return {"entities": spans}

    class Detector:
        def run(self, dataframe: pd.DataFrame, **kwargs: object) -> DetectorResult:
            return as_detector_result(dataframe, self.frame(dataframe))

        def frame(self, dataframe: pd.DataFrame) -> EntityDetectionResult:
            frame = dataframe.copy()
            ents = [entities(row) for row in range(len(frame))]
            frame[COL_FINAL_ENTITIES] = ents
            frame[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value(e) for e in ents]
            return EntityDetectionResult(frame, [], tuple(range(len(frame))))

    seen: list[list[str]] = []

    class StubRewrite(RewriteWorkflow):
        # LLM stages stubbed.
        def _produce(self, rows, **options):
            def produce() -> None:
                seen.append(list(rows.frame[COL_TEXT]))
                rows.frame[COL_REWRITTEN_TEXT] = [f"R<{t}>" for t in rows.frame[COL_TEXT]]
                for col in (COL_UTILITY_SCORE, COL_LEAKAGE_MASS, COL_WEIGHTED_LEAKAGE_RATE):
                    rows.frame[col] = 1.0 if col == COL_UTILITY_SCORE else 0.0
                rows.frame[COL_ANY_HIGH_LEAKED] = False

            return produce

    anon = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, Detector()), rewrite_runner=StubRewrite(Mock()))
    with caplog.at_level(logging.INFO, logger="anonymizer"):
        result = anon.run(config=AnonymizerConfig(rewrite=Rewrite()), data=AnonymizerInput(source=str(csv_path)))

    kept = [t for row, t in enumerate(texts) if row not in rejected]
    assert seen == ([kept] if kept else [])
    assert list(result.dataframe["text_rewritten"]) == [f"R<{t}>" for t in kept]
    assert result.failed_records == [
        FailedRecord(
            input_record_ids(AnonymizerInput(source=str(csv_path)))[row], "detect", "overlapping_output_ownership"
        )
        for row in rejected
    ]
    # A record failed at detect is counted under detection, not rewrite.
    detected = f"across {len(texts) - len(rejected)} records ({len(rejected)} failed)"
    assert detected in caplog.text and "Rewrite complete (0 failed)" in caplog.text
