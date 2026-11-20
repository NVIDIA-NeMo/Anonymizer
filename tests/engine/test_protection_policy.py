# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The release-policy table: one row for every situation, mode and detector kind, and the decisions read from it."""

from __future__ import annotations

from itertools import product
from typing import Any, get_args
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
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, _build_entities_by_value
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.engine.protection_policy import POLICY, Detector, Mode, Situation, assess, fail_reasons
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow
from anonymizer.interface.anonymizer import _GRAPH_FAILURE_REASONS, _KNOWN_FAILURE_REASONS, Anonymizer
from tests.detector_results import builtin_stub


def test_every_situation_has_a_row_for_every_mode_and_detector() -> None:
    """Totality: a situation nobody has ruled on for some mode or detector kind fails here."""
    expected = set(product(get_args(Situation), get_args(Mode), get_args(Detector)))
    assert set(POLICY) == expected


def test_every_failure_verdict_names_its_reason_and_the_reason_set_is_derived() -> None:
    assert all(v.reason for v in POLICY.values() if v.action == "fail")
    assert _KNOWN_FAILURE_REASONS == _GRAPH_FAILURE_REASONS | fail_reasons()
    assert {"unspanned_value", "dropped_output", "unplaced_output", "stopped_early"} <= _KNOWN_FAILURE_REASONS


def test_rewrite_verdicts_do_not_depend_on_the_detector_kind() -> None:
    """Rewrite's runners take no detector kind (J3), so its rows must agree across kinds."""
    for situation in get_args(Situation):
        assert assess(situation, "rewrite", "builtin") == assess(situation, "rewrite", "custom"), situation


def test_the_ruled_rows() -> None:
    assert assess("missed_repeat", "replace", "custom").reason == "unspanned_value"
    assert assess("missed_repeat", "replace", "builtin").action == "protect"
    assert assess("missed_repeat", "rewrite", "builtin").action == "release"
    assert {assess("placeholder_in_text", m, "builtin").action for m in get_args(Mode)} == {"flag"}


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
