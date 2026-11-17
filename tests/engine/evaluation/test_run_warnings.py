# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pandas as pd

from anonymizer.engine import constants as cols
from anonymizer.engine.evaluation.run_warnings import collect_evaluation_warnings, collect_run_warnings


def test_grounding_attempts_and_skipped_spans_are_separate_events() -> None:
    detection = pd.DataFrame(
        [
            {
                cols.RECORD_ID_COLUMN: "r1",
                cols.COL_AUGMENTATION_ATTEMPTS: json.dumps(
                    {
                        "attempts": [
                            {
                                "attempt_index": 0,
                                "failures": [
                                    {"category": "non_verbatim_context"},
                                    {"category": "zero_contextual_matches"},
                                ],
                            },
                            {"attempt_index": 1, "failures": [{"category": "non_verbatim_context"}]},
                            {"attempt_index": 2, "failures": []},
                        ]
                    }
                ),
            }
        ]
    )
    output = pd.DataFrame(
        [
            {
                cols.RECORD_ID_COLUMN: "r1",
                cols.COL_REPLACEMENT_APPLICATION: {
                    "targeted_span_count": 7,
                    "applied_span_count": 4,
                    "skipped_span_count": 3,
                    "skipped_span_label_counts": {"age": 3},
                },
            }
        ]
    )
    events = collect_run_warnings(detection, output)
    assert [event["code"] for event in events] == ["augmentation_grounding", "augmentation_grounding", "skipped_spans"]
    assert [event["count"] for event in events] == [1, 1, 3]
    assert [event["unit"] for event in events] == ["attempts", "attempts", "spans"]
    assert "non_verbatim_context: 1" in events[0]["detail"]
    assert "zero_contextual_matches: 1" in events[0]["detail"]


def test_evaluation_unavailable_events_preserve_saved_reasons() -> None:
    frame = pd.DataFrame(
        [
            {
                cols.RECORD_ID_COLUMN: "r1",
                cols.COL_PRIVACY_QA_ERROR: "Privacy answers unavailable.",
                cols.COL_ENTITY_COVERAGE: None,
            }
        ]
    )
    events = collect_evaluation_warnings(frame)
    assert [event["code"] for event in events] == ["privacy_qa_unavailable", "judge_unavailable:Entity coverage"]
    assert events[0]["detail"] == "Privacy answers unavailable."
    assert all(event["record_id"] == "r1" for event in events)


def test_clean_pipeline_records_have_no_warning_events() -> None:
    frame = pd.DataFrame(
        [
            {
                cols.COL_AUGMENTATION_ATTEMPTS: {"attempts": []},
                cols.COL_REPLACEMENT_APPLICATION: {"skipped_span_count": 0},
            }
        ]
    )
    assert collect_run_warnings(frame, frame) == []
    assert collect_evaluation_warnings(frame) == []
