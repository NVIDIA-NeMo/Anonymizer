# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A phase-scoped operation with an internal bounded loop is verified only on its final candidate.

The covering verifier sees each target's last pass; the lowered rewrite projects review from the final
metrics, keeps source positions and order, and reports detection and rewrite failures in order.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITTEN_TEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_TEXT,
    COL_UTILITY_SCORE,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.detect_step import resolve_positions
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.graph_core.authoring import DATUM, TEXT, Assessment, Operation, Requests
from anonymizer.engine.graph_core.observe import report_requests
from anonymizer.engine.ndd.failed_record import FailedRecord
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow, _rewrite_lowered, _Rows
from anonymizer.engine.row_partitioning import ROW_ORDER_COL
from anonymizer.interface.errors import AnonymizerWorkflowError
from anonymizer.measurement import measurement_session
from tests.graph.helpers import graph, run

TEXTS = ("alpha", "beta", "gamma")
PASSES = (1, 2, 3)
BAD = {(1, 1), (2, 3)}  # (target index, pass): the candidate carries the marker only at that pass


def _ids(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with the record id column the ``Anonymizer`` seam attaches before the detect step."""
    return frame.assign(**{RECORD_ID_COLUMN: [f"r{i}" for i in range(len(frame))]})


class _Loop:
    """The body: three passes over the targets, each rewriting every value; ``history`` keeps every pass."""

    def __init__(self, *, fail: int | None = None, requests: int = 0) -> None:
        self.history: list[list[str]] = []
        self.fail = fail
        self.requests = requests

    def operation(self) -> Operation:
        def body(text: tuple[str, ...]) -> list[Any]:
            values = list(text)
            for p in PASSES:
                values = [f"{v.split(' ')[0]} p{p}{' BAD' if (i, p) in BAD else ''}" for i, v in enumerate(values)]
                self.history.append(values)
                report_requests(self.requests)
            return [ValueError("x") if i == self.fail else v for i, v in enumerate(values)]

        return Operation.per_invocation("loop", body, requests=Requests.opaque())


def _verifier(seen: list[str]) -> Operation:
    def assess(text: str, target: Any) -> Assessment:
        seen.append(text)
        return Assessment("BAD" not in text, target)

    return Operation.verifier(
        "assess", assess, {"text": TEXT, "target": DATUM}, Requests.none(), ("assess",), ("text",)
    )


_METRICS = {
    COL_UTILITY_SCORE: 0.9,
    COL_LEAKAGE_MASS: 0.0,
    COL_ANY_HIGH_LEAKED: False,
    # A JSON string: frame cells here are assigned positionally, which a list value would spread.
    COL_SENSITIVITY_DISPOSITION: json.dumps(
        {
            "sensitivity_disposition": [
                {
                    "id": 1,
                    "sensitivity": "high",
                    "protection_method_suggestion": "replace",
                    "source": "tagged",
                    "category": "direct_identifier",
                    "entity_label": "first_name",
                    "entity_value": "t0",
                }
            ]
        }
    ),
}


def _lowered(
    rows: list[dict[str, Any]],
    *,
    leak: float | None = None,
    util: float | None = None,
    passthrough: tuple[int, ...] = (),
    produce: Any = None,
    review_dtype: Any = None,
    drop_ready: bool = True,
    scramble: bool = False,
    drop: int | None = None,
) -> Any:
    """Drive the lowered rewrite directly: each dict overrides a row's final metrics; thresholds as given."""
    records = []
    for i, over in enumerate(rows):
        record = {ROW_ORDER_COL: i, COL_TEXT: f"t{i}", COL_REWRITTEN_TEXT: f"r{i}", **_METRICS, **over}
        if drop_ready and "ready" not in over:
            record.pop(COL_REWRITE_REPLACEMENT_READY, None)
        records.append(record)
    frame = pd.DataFrame(records)
    if review_dtype is not None:
        frame[COL_NEEDS_HUMAN_REVIEW] = pd.Series([False] * len(frame), dtype=review_dtype)
    extra = pd.DataFrame({ROW_ORDER_COL: list(passthrough), COL_TEXT: [f"p{i}" for i in passthrough]})
    evaluation: Any = SimpleNamespace(flag_leakage_above=leak, flag_utility_below=util)
    state = _Rows(frame)

    def reorder() -> None:  # partial repair order: the later rows first, then the earlier ones
        state.frame = state.frame.iloc[::-1].reset_index(drop=True)

    def dropper() -> None:  # the backend dropped one row's output
        state.frame = state.frame[state.frame[ROW_ORDER_COL] != drop].reset_index(drop=True)

    stage = reorder if scramble else dropper if drop is not None else produce or (lambda: None)
    return _rewrite_lowered(frame.copy(), extra, state, stage, evaluation)


_DETECTION_FAILURE = FailedRecord("d1", "detect", "dropped")
_REWRITE_FAILURE = FailedRecord("w1", "rewrite", "dropped")


def _detection(
    texts: list[str],
    *,
    entity_rows: tuple[int, ...],
    positions: tuple[int, ...] = (),
    failures: tuple[FailedRecord, ...] = (),
) -> EntityDetectionResult:
    """A detector result over ``texts``; the rows listed in ``entity_rows`` carry one detected entity."""
    has = [
        {"entities_by_value": [{"value": "x"}]} if i in entity_rows else {"entities_by_value": []}
        for i in range(len(texts))
    ]
    frame = pd.DataFrame({COL_TEXT: texts, COL_ENTITIES_BY_VALUE: has, COL_FINAL_ENTITIES: [{}] * len(texts)})
    return EntityDetectionResult(frame, list(failures), positions)


def _stubbed(
    workflow: RewriteWorkflow, *, fail_row: int | None = None, failure: FailedRecord = _REWRITE_FAILURE
) -> Any:
    """Replace the model stages: every row is rewritten, except ``fail_row`` which is dropped and reported."""

    def produce(rows: _Rows, **_options: Any) -> Any:
        def stage() -> None:
            frame = rows.frame.copy()
            frame[COL_REWRITTEN_TEXT] = [f"r{o}" for o in frame[ROW_ORDER_COL]]
            frame.loc[:, list(_METRICS)] = list(_METRICS.values())
            if fail_row is not None:
                frame = frame[frame[ROW_ORDER_COL] != fail_row].reset_index(drop=True)
                rows.failed.append(failure)
            rows.frame = frame

        return stage

    setattr(workflow, "_produce", produce)
    return workflow


_OPTIONS: dict[str, Any] = {
    "evaluation": SimpleNamespace(flag_leakage_above=None, flag_utility_below=None),
    "model_configs": [],
    "selected_models": None,
    "replace_model_selection": None,
    "privacy_goal": None,
}


def _reviews(rows: list[dict[str, Any]], **kw: Any) -> list[bool]:
    out = _lowered(rows, **kw).dataframe
    return [bool(v) for v in out[COL_NEEDS_HUMAN_REVIEW]]


def test_verifier_sees_only_final_candidates() -> None:
    g, (a, b, d) = graph(*TEXTS)
    loop, seen = _Loop(), []
    r = run(g, loop.operation(), _verifier(seen))
    last, earlier = loop.history[-1], {v for pass_ in loop.history[:-1] for v in pass_}
    assert len(loop.history) == 3
    assert seen == last
    assert not set(seen) & earlier
    assert [r.value(t) for t in (a, b, d)] == last
    assert [r.outcome(t).qualification for t in (a, b, d)] == ["met", "met", "not_met"]
    assert set(r.protected) == {a, b}
    assert r.consistent


def test_failed_target_is_unprotected_and_not_verified() -> None:
    g, (a, b, _d) = graph(*TEXTS)
    loop, seen = _Loop(fail=1), []
    r = run(g, loop.operation(), _verifier(seen))
    o = r.outcome(b)
    content = dict(r.content())
    assert (o.terminal, o.reason, o.qualification) == ("failed", "raised", "not_met")
    assert b not in r.protected
    assert content[b] is None
    assert set(r.protected) == {a}
    assert seen == [loop.history[-1][0], loop.history[-1][2]]
    assert r.outcome(a).qualification == "met"


def test_requests_reported_inside_the_loop_are_exact() -> None:
    g, _ = graph(*TEXTS)
    r = run(g, _Loop(requests=2).operation(), _verifier([]))
    observed = {name: str(count) for name, count in r.summary().observed}
    assert observed["requests"] == "exact 6"


def test_leakage_review_rules() -> None:
    assert _reviews([{COL_ANY_HIGH_LEAKED: True, COL_LEAKAGE_MASS: 0.0}], leak=0.5, util=0.1) == [True]
    assert _reviews([{COL_LEAKAGE_MASS: 0.4}, {COL_LEAKAGE_MASS: 0.4001}], leak=0.4, util=0.1) == [False, True]
    assert _reviews([{COL_LEAKAGE_MASS: 0.99}], leak=None, util=0.1) == [False]
    # leakage 0.2 is above the utility floor 0.1 but below the leakage limit 0.5
    assert _reviews([{COL_LEAKAGE_MASS: 0.2, COL_UTILITY_SCORE: 0.9}], leak=0.5, util=0.1) == [False]


def test_utility_review_rules() -> None:
    assert _reviews([{COL_UTILITY_SCORE: 0.3}, {COL_UTILITY_SCORE: 0.2999}], leak=0.9, util=0.3) == [False, True]
    assert _reviews([{COL_UTILITY_SCORE: 0.0}], leak=0.9, util=None) == [False]
    assert _reviews([{COL_UTILITY_SCORE: 0.05, COL_LEAKAGE_MASS: 0.0}], leak=0.9, util=0.3) == [True]


def test_body_error_raises_and_verifier_error_or_unavailable_text_flags_review() -> None:
    def boom() -> None:
        raise KeyError("body")

    with pytest.raises(KeyError):
        _lowered([{}, {}], leak=0.5, util=0.1, produce=boom)
    # leakage is not a number under a limit: the target fails, the call returns
    out = _lowered([{COL_LEAKAGE_MASS: "not a number"}, {}], leak=0.5, util=0.1).dataframe
    assert [bool(v) for v in out[COL_NEEDS_HUMAN_REVIEW]] == [True, False]
    # a rewrite that is not text is unavailable, not an escape
    out = _lowered([{COL_REWRITTEN_TEXT: None}, {}], leak=0.5, util=0.1).dataframe
    assert [bool(v) for v in out[COL_NEEDS_HUMAN_REVIEW]] == [True, False]


def test_review_column_keeps_an_existing_dtype() -> None:
    arrow = _lowered([{}], leak=0.5, util=0.1, review_dtype="bool[pyarrow]").dataframe[COL_NEEDS_HUMAN_REVIEW]
    plain = _lowered([{}], leak=0.5, util=0.1).dataframe[COL_NEEDS_HUMAN_REVIEW]
    assert str(arrow.dtype) == "bool[pyarrow]"
    assert str(plain.dtype) == "bool"


def test_source_positions_include_passthrough_rows() -> None:
    r = _lowered([{}, {}], leak=0.5, util=0.1, passthrough=(2, 3))
    assert r.source_positions == (0, 1, 2, 3)


def test_final_text_decides_review_not_the_readiness_flag() -> None:
    assert _reviews([{COL_SENSITIVITY_DISPOSITION: None}], leak=0.5, util=0.1) == [True]
    assert _reviews([{"ready": False, COL_REWRITE_REPLACEMENT_READY: False}], leak=0.5, util=0.1) == [False]
    assert _reviews([{COL_REWRITTEN_TEXT: "T0 stays"}], leak=0.5, util=0.1) == [True]


def test_reordered_stage_output_fans_back_to_input_order() -> None:
    rows = [{COL_LEAKAGE_MASS: 0.9}, {}, {COL_UTILITY_SCORE: 0.01}]
    out = _lowered(rows, leak=0.5, util=0.3, scramble=True).dataframe
    # Row 0's leakage is over the flag, so its text is withheld; rows 1 and 2 keep their own text.
    assert list(out[COL_REWRITTEN_TEXT]) == [None, "r1", "r2"]
    assert [bool(v) for v in out[COL_NEEDS_HUMAN_REVIEW]] == [True, False, True]
    assert list(out[COL_UTILITY_SCORE]) == [0.9, 0.9, 0.01]


def test_dropped_row_is_absent_from_output() -> None:
    out = _lowered([{}, {}, {}], leak=0.5, util=0.1, drop=1).dataframe
    assert list(out[COL_REWRITTEN_TEXT]) == ["r0", "r2"]
    assert [bool(v) for v in out[COL_NEEDS_HUMAN_REVIEW]] == [False, False]


def test_nan_leakage_under_a_limit_is_flagged_for_review() -> None:
    assert _reviews([{COL_LEAKAGE_MASS: float("nan")}], leak=0.5) == [True]


def test_not_met_without_utility_is_flagged_for_review() -> None:
    assert _reviews([{COL_ANY_HIGH_LEAKED: True, COL_UTILITY_SCORE: None}], leak=0.5, util=0.3) == [True]


def test_detection_failures_precede_rewrite_failures() -> None:
    texts = ["a", "b", "c", "d", "e"]
    detection = _detection(
        ["a", "c", "d", "e"], entity_rows=(0, 1, 2), positions=(0, 2, 3, 4), failures=(_DETECTION_FAILURE,)
    )
    workflow = _stubbed(RewriteWorkflow(Mock()), fail_row=2, failure=_REWRITE_FAILURE)
    result = workflow.run_with_detection(_ids(pd.DataFrame({COL_TEXT: texts})), detect=lambda _: detection, **_OPTIONS)
    out = list(result.dataframe[COL_TEXT])
    assert result.failed_records == [_DETECTION_FAILURE, _REWRITE_FAILURE]
    assert "b" not in out
    assert "d" not in out
    assert out == ["a", "c", "e"]
    # The detection result is the detector's, with each row's input record id carried onto it.
    assert result.detection.failed_records == detection.failed_records
    assert list(result.detection.dataframe[RECORD_ID_COLUMN]) == ["r0", "r2", "r3", "r4"]


def test_unplaceable_detection_result_raises_before_rewrite() -> None:
    detection = _detection(["c", "a", "b"], entity_rows=(0, 1, 2), failures=(_DETECTION_FAILURE,))
    workflow = _stubbed(RewriteWorkflow(Mock()), fail_row=1, failure=_REWRITE_FAILURE)
    frame = pd.DataFrame({COL_TEXT: ["a", "b", "c"]})
    with pytest.raises(AnonymizerWorkflowError):
        workflow.run_with_detection(frame, detect=lambda _: detection, **_OPTIONS)


def _missing_text_detection() -> EntityDetectionResult:
    missing = _detection(["a", "b"], entity_rows=())
    missing.dataframe[COL_TEXT] = pd.Series(["a", np.nan], dtype=object)
    return missing


@pytest.mark.parametrize(
    ("source", "result"),
    [
        (pd.DataFrame({COL_TEXT: ["a", "b"]}), lambda: _detection(["a", "x"], entity_rows=())),
        (pd.DataFrame({COL_TEXT: ["a", "b"]}), lambda: _detection(["a", "b"], entity_rows=())),
        (pd.DataFrame({COL_TEXT: ["a", None]}), _missing_text_detection),
    ],
    ids=["differing_text", "equal_text", "missing_values"],
)
def test_no_row_is_placed_by_its_text(source: pd.DataFrame, result: Any) -> None:
    with pytest.raises(AnonymizerWorkflowError):
        resolve_positions(source, result())


def test_run_source_positions_include_passthrough_rows() -> None:
    detection = _detection(["a", "b", "c"], entity_rows=(0, 2), positions=(0, 1, 2))
    workflow = _stubbed(RewriteWorkflow(Mock()))
    frame = _ids(pd.DataFrame({COL_TEXT: ["a", "b", "c"]}))
    result = workflow.run_with_detection(frame, detect=lambda _: detection, **_OPTIONS)
    assert result.source_positions == (0, 1, 2)


def test_combined_rewrite_records_its_own_stage_name() -> None:
    detection = _detection(["a", "b"], entity_rows=(0, 1), positions=(0, 1))
    workflow = _stubbed(CombinedRewriteWorkflow(Mock()))
    with measurement_session() as collector:
        workflow.run_with_detection(_ids(pd.DataFrame({COL_TEXT: ["a", "b"]})), detect=lambda _: detection, **_OPTIONS)
    stages = [r["stage"] for r in collector.records if not r.get("stage", "graph_").startswith("graph_")]
    assert stages == ["CombinedRewriteWorkflow.run"]
