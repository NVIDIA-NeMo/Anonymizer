# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A row rejected at ``detect`` stays out of Substitute's replacement-map generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import patch

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_REPLACEMENT_MAP, COL_TEXT
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.interface.anonymizer import Anonymizer, AnonymizerResult

REJECTED = "Alice Smith lives in Paris."  # overlapping first_name/full_name spans: fails at detect
VALID = "Bob lives in Paris."  # shares (city, Paris) with the rejected row
SUBSTITUTES = {
    REJECTED: {"Alice": "Carol", "Alice Smith": "Carol Jones", "Paris": "Lyon"},
    VALID: {"Bob": "Dave", "Paris": "Nice"},
}


def _span(text: str, value: str, label: str, span_id: int) -> dict[str, Any]:
    start = text.index(value)
    return {
        "id": str(span_id),
        "value": value,
        "label": label,
        "start_position": start,
        "end_position": start + len(value),
    }


def _entities(text: str) -> dict[str, Any]:
    if text == REJECTED:
        names = [_span(text, "Alice", "first_name", 0), _span(text, "Alice Smith", "full_name", 1)]
    else:
        names = [_span(text, "Bob", "first_name", 0)]
    return {"entities": [*names, _span(text, "Paris", "city", 9)]}


class _OverlappingDetector:
    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
        frame = dataframe.copy()
        entities = [_entities(str(text)) for text in frame[COL_TEXT]]
        frame[COL_FINAL_ENTITIES] = entities
        frame[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value(e) for e in entities]
        return EntityDetectionResult(frame, [], tuple(range(len(frame))))


def _run(tmp_path: Path, consistency: Literal["record", "run"]) -> tuple[list[str], AnonymizerResult]:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [REJECTED, VALID]}).to_csv(source, index=False)
    map_rows: list[str] = []

    def run_workflow(self: NddAdapter, df: pd.DataFrame, *, workflow_name: str, **_: Any) -> WorkflowRunResult:
        assert workflow_name == "replace-map-generation", workflow_name
        out = df.copy().reset_index(drop=True)
        texts = [str(text) for text in out[COL_TEXT]]
        map_rows.extend(texts)
        out[COL_REPLACEMENT_MAP] = [
            {
                "replacements": [
                    {"original": e["value"], "label": e["label"], "synthetic": SUBSTITUTES[text][e["value"]]}
                    for e in _entities(text)["entities"]
                ]
            }
            for text in texts
        ]
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    anonymizer = Anonymizer(detection_workflow=cast(EntityDetectionWorkflow, _OverlappingDetector()))
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = anonymizer.run(
            config=AnonymizerConfig(replace=Substitute(consistency=consistency)),
            data=AnonymizerInput(source=str(source)),
        )
    return map_rows, result


@pytest.mark.parametrize("consistency", ["record", "run"])
def test_row_rejected_at_detect_stays_out_of_map_generation(
    tmp_path: Path, consistency: Literal["record", "run"]
) -> None:
    map_rows, result = _run(tmp_path, consistency)
    assert result.failed_records == [FailedRecord("0", "detect", "overlapping_output_ownership")]
    # The valid row keeps its own substitutes; none was first chosen for the rejected row.
    assert list(result.dataframe["text_replaced"]) == ["Dave lives in Nice."]
    # The map request carries only the rows still ready after detect.
    assert map_rows == [VALID]
