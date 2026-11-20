# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-policy behaviour through the public API only (``Anonymizer.run``, the result's dataframe, ``outcome()``
and ``failed_records``): the Substitute placeholder flag and an override runner's unplaceable rows. The tests import
no policy module, so they run against a tree that predates it."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, Rewrite, TextRecord, TextRecordsInput
from anonymizer.config.replace_strategies import Redact, Substitute
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_REWRITTEN_TEXT,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow
from anonymizer.interface.anonymizer import Anonymizer


def builtin_stub(run: Any) -> EntityDetectionWorkflow:
    """The built-in workflow class with its ``run`` replaced (a local copy: older trees lack the shared helper)."""
    detector = EntityDetectionWorkflow(adapter=Mock())
    detector.run = Mock(side_effect=run)
    return detector


# --- the Replace placeholder flag (lipika, #426) ----------------------------------------------------------------------


def _alice(frame: pd.DataFrame, **_: object) -> EntityDetectionResult:
    out = frame.copy()
    entities = [
        {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]}
        for _ in range(len(out))
    ]
    out[COL_FINAL_ENTITIES] = entities
    out[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value(e) for e in entities]
    return EntityDetectionResult(out, [], tuple(range(len(out))))


@pytest.mark.parametrize(("synthetic", "flagged"), [("[SUBSTITUTE_FIRST_NAME_1]", True), ("Carol", False)])
def test_a_substitute_row_keeping_a_placeholder_is_flagged_not_failed(synthetic: str, flagged: bool) -> None:
    def run_workflow(self: NddAdapter, df: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        out = df.copy().reset_index(drop=True)
        out[COL_REPLACEMENT_MAP] = [
            {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": synthetic}]}
        ] * len(out)
        return WorkflowRunResult(dataframe=out, failed_records=[], source_positions=tuple(range(len(out))))

    data = TextRecordsInput(records=[TextRecord(id="r", text="Alice wrote.")])
    with patch.object(NddAdapter, "run_workflow", run_workflow):
        result = Anonymizer(detection_workflow=builtin_stub(_alice)).run(
            config=AnonymizerConfig(replace=Substitute()), data=data
        )
    assert result.failed_records == []
    assert list(result.dataframe["placeholder_in_text"]) == [flagged]
    assert result.outcome(0).qualification == "met"


def test_redact_rows_carry_no_placeholder_flag() -> None:
    data = TextRecordsInput(records=[TextRecord(id="r", text="Alice wrote.")])
    result = Anonymizer(detection_workflow=builtin_stub(_alice)).run(
        config=AnonymizerConfig(replace=Redact()), data=data
    )
    assert "placeholder_in_text" not in result.dataframe.columns


# --- an override runner's output that cannot be placed (unplaced_output) ---------------------------------------------

DATA = TextRecordsInput(records=[TextRecord(id=i, text=f"Alice {i}.") for i in ("a", "b", "c")])


@pytest.mark.parametrize("mode", ["replace", "rewrite"])
def test_an_override_runners_unplaceable_rows_fail_their_records(mode: str) -> None:
    """Rows without an id, with an unknown id, or with a repeated id are not released; each input record no placed
    row answers fails with ``unplaced_output`` at the runner's step. Row ``a`` keeps its id and is released."""

    def output(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy().reset_index(drop=True)
        out[COL_REPLACED_TEXT] = "x"
        out[COL_REWRITTEN_TEXT] = "x"
        out[RECORD_ID_COLUMN] = ["a", "zz", None]
        return pd.concat([out, out.iloc[[0]].assign(**{RECORD_ID_COLUMN: "zz"})], ignore_index=True)

    class Replace(ReplacementWorkflow):
        def run(self, dataframe: pd.DataFrame, **_: object) -> ReplacementResult:
            return ReplacementResult(output(dataframe), [])

    class Rewriter(RewriteWorkflow):
        def run(self, dataframe: pd.DataFrame, **_: object) -> RewriteResult:
            return RewriteResult(output(dataframe), [])

    anonymizer = Anonymizer(
        detection_workflow=builtin_stub(_alice), replace_runner=Replace(), rewrite_runner=Rewriter(Mock())
    )
    config = AnonymizerConfig(replace=Redact()) if mode == "replace" else AnonymizerConfig(rewrite=Rewrite())
    result = anonymizer.run(config=config, data=DATA)
    assert result.failed_records == [FailedRecord(i, mode, "unplaced_output") for i in ("b", "c")]
    assert list(result.dataframe["text"]) == ["Alice a."]
