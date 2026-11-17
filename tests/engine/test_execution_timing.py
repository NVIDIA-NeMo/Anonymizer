# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from anonymizer.engine.constants import COL_RAW_DETECTED
from anonymizer.engine.execution_timing import capture_execution, record_elapsed
from anonymizer.engine.ndd.adapter import _record_report_timings


@dataclass
class _Result:
    execution_timings: dict[str, dict[str, float | None]] = field(default_factory=dict)


def test_concurrent_detection_spans_are_not_summed() -> None:
    @capture_execution("run")
    def run() -> _Result:
        _record_report_timings(
            "entity-detection",
            15.0,
            [
                {"column": COL_RAW_DETECTED, "slot_acquired_at": 10.0, "completed_at": 17.0},
                {"column": COL_RAW_DETECTED, "slot_acquired_at": 12.0, "completed_at": 20.0},
                {"column": COL_RAW_DETECTED, "slot_acquired_at": None, "completed_at": 50.0},
            ],
        )
        return _Result()

    timings = run().execution_timings["run"]
    assert timings["Detector (GLiNER)"] == 10.0
    assert timings["LLM validation"] is None
    assert timings["LLM augmentation"] is None
    assert timings["Detection workflow"] == 15.0
    assert timings["Total elapsed"] is not None


def test_refresh_preserves_run_and_context_is_reset_after_failure() -> None:
    result = _Result({"run": {"Total elapsed": 12.0}, "evaluate": {"Old task": 1.0}})

    @capture_execution("evaluate")
    def evaluate() -> _Result:
        record_elapsed("New task", 0.0)
        return result

    evaluate()
    assert result.execution_timings["run"] == {"Total elapsed": 12.0}
    assert result.execution_timings["evaluate"]["New task"] == 0.0
    assert "Old task" not in result.execution_timings["evaluate"]

    @capture_execution("evaluate")
    def fail() -> _Result:
        raise ValueError("failed")

    with pytest.raises(ValueError):
        fail()
    record_elapsed("Outside capture", 30.0)
    assert "Outside capture" not in result.execution_timings["evaluate"]
