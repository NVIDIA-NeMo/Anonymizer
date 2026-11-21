# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A built-in missed repeat that swallows a kept span must not drop that span's value from the Substitute map request.

From a live run: detection spans "credit card ending in 1065" once and only "1065" at the second occurrence. The repeat
span over the second "credit card ending in 1065" collapses the kept "1065" inside it, and the request then asked only
for the wide value. The map model answered with "1065" (as it does when "1065" is not requested next to the phrase), so
the requested pair was never completed and the record failed at map generation."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd

from anonymizer.config.anonymizer_config import AnonymizerConfig, TextRecord, TextRecordsInput
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_REPLACEMENT_MAP
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
)
from anonymizer.engine.ndd.adapter import NddAdapter, WorkflowRunResult
from anonymizer.engine.schemas import EntitiesByValueSchema
from anonymizer.interface.anonymizer import Anonymizer

WIDE, SHORT = "credit card ending in 1065", "1065"
TEXT = f"Paid with your {WIDE}. Receipt: refund to {WIDE} today."


def _detect(self: object, frame: pd.DataFrame, **_: object) -> EntityDetectionResult:
    first = TEXT.index(WIDE)
    short = TEXT.rindex(SHORT)
    spans = [
        {"value": WIDE, "label": "card_last_four", "start_position": first, "end_position": first + len(WIDE)},
        {"value": SHORT, "label": "card_last_four", "start_position": short, "end_position": short + len(SHORT)},
    ]
    out = frame.copy()
    out[COL_FINAL_ENTITIES] = [{"entities": spans}] * len(out)
    out[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value({"entities": spans})] * len(out)
    return EntityDetectionResult(out, [], tuple(range(len(out))))


def _map_model(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
    """Answers like the live model: the phrase maps to a phrase only when "1065" is requested beside it; asked for the
    phrase alone, it maps "1065" instead."""
    out = df.copy().reset_index(drop=True)
    maps = []
    for cell in out[COL_ENTITIES_BY_VALUE]:
        requested = {item.value for item in EntitiesByValueSchema.from_raw(cell).entities_by_value}
        entries = []
        if SHORT in requested:
            entries.append({"original": SHORT, "label": "card_last_four", "synthetic": "4492"})
            if WIDE in requested:
                entries.append({"original": WIDE, "label": "card_last_four", "synthetic": "credit card ending in 4492"})
        elif WIDE in requested:
            entries.append({"original": SHORT, "label": "card_last_four", "synthetic": "4492"})
        maps.append({"replacements": entries})
    out[COL_REPLACEMENT_MAP] = maps
    return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))


def test_a_repeat_span_that_absorbs_a_kept_span_keeps_its_value_in_the_map_request() -> None:
    data = TextRecordsInput(records=[TextRecord(id="r", text=TEXT)])
    # The default built-in detector, with its run scripted (an explicit detector was "custom" on older trees).
    with patch.object(NddAdapter, "run_workflow", _map_model), patch.object(EntityDetectionWorkflow, "run", _detect):
        result = Anonymizer().run(config=AnonymizerConfig(replace=Substitute()), data=data)
    assert result.failed_records == []
    released = result.dataframe["text_replaced"].iloc[0]
    assert SHORT not in released
    assert released.count("credit card ending in 4492") == 2


def _detect_nested(self: object, frame: pd.DataFrame, **_: object) -> EntityDetectionResult:
    """The phrase spanned only at its second occurrence, with "1065" nested inside it (the detector's own collapse
    drops it, so it is never kept); the first phrase is a missed repeat."""
    second = TEXT.rindex(WIDE)
    short = TEXT.rindex(SHORT)
    spans = [
        {"value": WIDE, "label": "card_last_four", "start_position": second, "end_position": second + len(WIDE)},
        {"value": SHORT, "label": "card_last_four", "start_position": short, "end_position": short + len(SHORT)},
    ]
    out = frame.copy()
    out[COL_FINAL_ENTITIES] = [{"entities": spans}] * len(out)
    out[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value({"entities": spans})] * len(out)
    return EntityDetectionResult(out, [], tuple(range(len(out))))


def test_a_nested_value_seen_only_inside_a_missed_repeat_stays_in_the_map_request() -> None:
    """Before the repeat span, "1065" occurred outside every kept span (inside the first phrase), so the base asked for
    it; the repeat span must not take it out of the request."""
    data = TextRecordsInput(records=[TextRecord(id="r", text=TEXT)])
    with (
        patch.object(NddAdapter, "run_workflow", _map_model),
        patch.object(EntityDetectionWorkflow, "run", _detect_nested),
    ):
        result = Anonymizer().run(config=AnonymizerConfig(replace=Substitute()), data=data)
    assert result.failed_records == []
    released = result.dataframe["text_replaced"].iloc[0]
    assert SHORT not in released
