# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow results correspond to their sources only through reported positions or record IDs, never by text."""

from __future__ import annotations

import pandas as pd
import pytest

from anonymizer.engine.constants import COL_TEXT, RECORD_ID_COLUMN
from anonymizer.engine.detection.detect_step import resolve_positions
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.row_partitioning import resolve_source_positions
from anonymizer.interface.errors import AnonymizerWorkflowError


def test_swapped_identical_text_is_not_placed() -> None:
    source = pd.DataFrame({COL_TEXT: ["Call Ann", "Call Ann"]})
    # Row 1's detection first, then row 0's: equal text, no reported positions.
    returned = pd.DataFrame({COL_TEXT: ["Call Ann", "Call Ann"], "who": ["row 1", "row 0"]})
    with pytest.raises(AnonymizerWorkflowError):
        resolve_positions(source, EntityDetectionResult(returned, []))
    with pytest.raises(AnonymizerWorkflowError, match="uncorrelated_output"):
        resolve_source_positions(source, returned, None)


def test_same_count_without_positions_is_uncorrelated() -> None:
    source = pd.DataFrame({COL_TEXT: ["a", "b"]})
    with pytest.raises(AnonymizerWorkflowError, match="uncorrelated_output"):
        resolve_source_positions(source, pd.DataFrame({COL_TEXT: ["b", "a"]}), None)


def test_positions_and_record_ids_place_dropped_and_reordered_rows() -> None:
    source = pd.DataFrame({COL_TEXT: ["a", "b", "c"], RECORD_ID_COLUMN: ["r0", "r1", "r2"]})
    returned = pd.DataFrame({COL_TEXT: ["c", "a"], RECORD_ID_COLUMN: ["r2", "r0"]})
    assert resolve_source_positions(source, returned.drop(columns=[RECORD_ID_COLUMN]), (2, 0)) == (2, 0)
    assert resolve_source_positions(source, returned, None) == (2, 0)


@pytest.mark.parametrize("positions", [(0, 2), (0, 0)], ids=["out_of_range", "duplicated"])
def test_invalid_positions_are_rejected(positions: tuple[int, ...]) -> None:
    frame = pd.DataFrame({COL_TEXT: ["a", "b"]})
    with pytest.raises(AnonymizerWorkflowError):
        resolve_source_positions(frame, frame.copy(), positions)
