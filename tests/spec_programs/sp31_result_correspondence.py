# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-31 workflow results correspond to their sources only through positions or record IDs (E31.1-E31.4)."""

from __future__ import annotations

import pandas as pd

from anonymizer.engine.constants import COL_TEXT, RECORD_ID_COLUMN
from anonymizer.engine.detection.detect_step import resolve_positions
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.row_partitioning import resolve_source_positions
from anonymizer.interface.errors import AnonymizerWorkflowError
from tests.spec_programs._builder import Checks


def _resolve(source: pd.DataFrame, returned: pd.DataFrame, positions: tuple[int, ...] | None) -> object:
    try:
        return resolve_source_positions(source, returned, positions)
    except AnonymizerWorkflowError as exc:
        return "uncorrelated_output" if "uncorrelated_output" in str(exc) else "rejected"


def program() -> list[dict[str, str]]:
    c = Checks("SP-31")

    def swapped_identical_text() -> tuple[bool, str]:
        source = pd.DataFrame({COL_TEXT: ["Call Ann", "Call Ann"]})
        # Row 1's detection first, then row 0's: equal text, no reported positions.
        returned = pd.DataFrame({COL_TEXT: ["Call Ann", "Call Ann"], "who": ["row 1", "row 0"]})
        got = resolve_positions(source, EntityDetectionResult(returned, []))
        reason = _resolve(source, returned, None)
        return got is None and reason == "uncorrelated_output", f"positions={got}, reason={reason}"

    c.guard("E31.1", swapped_identical_text)

    def same_count_unreported() -> tuple[bool, str]:
        source = pd.DataFrame({COL_TEXT: ["a", "b"]})
        reason = _resolve(source, pd.DataFrame({COL_TEXT: ["b", "a"]}), None)
        return reason == "uncorrelated_output", f"reason={reason}"

    c.guard("E31.2", same_count_unreported)

    def explicit_with_drops_and_reordering() -> tuple[bool, str]:
        source = pd.DataFrame({COL_TEXT: ["a", "b", "c"], RECORD_ID_COLUMN: ["r0", "r1", "r2"]})
        returned = pd.DataFrame({COL_TEXT: ["c", "a"], RECORD_ID_COLUMN: ["r2", "r0"]})
        by_position = _resolve(source, returned.drop(columns=[RECORD_ID_COLUMN]), (2, 0))
        by_record_id = _resolve(source, returned, None)
        ok = by_position == by_record_id == (2, 0)
        return ok, f"positions={by_position}, record ids={by_record_id}"

    c.guard("E31.3", explicit_with_drops_and_reordering)

    def invalid_positions() -> tuple[bool, str]:
        source = pd.DataFrame({COL_TEXT: ["a", "b"]})
        returned = pd.DataFrame({COL_TEXT: ["a", "b"]})
        got = (_resolve(source, returned, (0, 2)), _resolve(source, returned, (0, 0)))
        return got == ("rejected", "rejected"), f"out of range={got[0]}, duplicated={got[1]}"

    c.guard("E31.4", invalid_positions)
    return c.lines
