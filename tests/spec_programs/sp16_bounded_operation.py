# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-16 an invocation-scoped operation with an internal bounded loop and a covering verifier (E16.1-E16.18)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd

from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITTEN_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.detection.detect_step import resolve_positions
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.graph_core.authoring import DATUM, TEXT, Assessment, Operation, Requests
from anonymizer.engine.graph_core.observe import report_requests
from anonymizer.engine.ndd.failed_record import FailedRecord
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow, _rewrite_lowered, _Rows
from anonymizer.engine.row_partitioning import ROW_ORDER_COL
from anonymizer.measurement import measurement_session
from tests.spec_programs._builder import Checks, graph, run

TEXTS = ("alpha", "beta", "gamma")
PASSES = (1, 2, 3)
BAD = {(1, 1), (2, 3)}  # (target index, pass): the candidate carries the marker only at that pass


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

    return Operation.verifier("assess", assess, {"text": TEXT, "target": DATUM}, Requests.none())


_METRICS = {COL_UTILITY_SCORE: 0.9, COL_LEAKAGE_MASS: 0.0, COL_ANY_HIGH_LEAKED: False}


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


def program() -> list[dict[str, str]]:
    c = Checks("SP-16")
    g, (a, b, d) = graph(*TEXTS)

    def final_candidates() -> tuple[bool, str]:
        loop, seen = _Loop(), []
        r = run(g, loop.operation(), _verifier(seen))
        last, earlier = loop.history[-1], {v for pass_ in loop.history[:-1] for v in pass_}
        quals = [r.outcome(t).qualification for t in (a, b, d)]
        return (
            len(loop.history) == 3
            and seen == last
            and not set(seen) & earlier
            and [r.value(t) for t in (a, b, d)] == last
            and quals == ["met", "met", "not_met"]
            and set(r.protected) == {a, b}
            and r.consistent,
            f"passes={len(loop.history)}, verifier calls={len(seen)}, final seen={seen == last}, qualifications={quals}",
        )

    c.guard("E16.1", final_candidates)

    def failed_target() -> tuple[bool, str]:
        loop, seen = _Loop(fail=1), []
        r = run(g, loop.operation(), _verifier(seen))
        o = r.outcome(b)
        content = dict(r.content())
        return (
            (o.terminal, o.reason, o.qualification) == ("failed", "raised", "not_met")
            and b not in r.protected
            and content[b] is None
            and set(r.protected) == {a}
            and seen == [loop.history[-1][0], loop.history[-1][2]]
            and r.outcome(a).qualification == "met",
            f"target 2 terminal={o.terminal}, qualification={o.qualification}, protected={len(r.protected)}, "
            f"content is None={content[b] is None}, verifier calls={len(seen)}",
        )

    c.guard("E16.2", failed_target)

    def exact_requests() -> tuple[bool, str]:
        r = run(g, _Loop(requests=2).operation(), _verifier([]))
        observed = {name: str(count) for name, count in r.summary().observed}
        return observed["requests"] == "exact 6", f"requests={observed['requests']}, total={observed['total']}"

    c.guard("E16.3", exact_requests)

    def verifier_rules() -> tuple[bool, str]:
        high = _reviews([{COL_ANY_HIGH_LEAKED: True, COL_LEAKAGE_MASS: 0.0}], leak=0.5, util=0.1)
        edge = _reviews([{COL_LEAKAGE_MASS: 0.4}, {COL_LEAKAGE_MASS: 0.4001}], leak=0.4, util=0.1)
        none = _reviews([{COL_LEAKAGE_MASS: 0.99}], leak=None, util=0.1)
        # distinct thresholds: leakage 0.2 is above the utility floor 0.1 but below the leakage limit 0.5
        distinct = _reviews([{COL_LEAKAGE_MASS: 0.2, COL_UTILITY_SCORE: 0.9}], leak=0.5, util=0.1)
        return (
            high == [True] and edge == [False, True] and none == [False] and distinct == [False],
            f"high leak only={high}, at/above limit={edge}, no limit={none}, distinct thresholds={distinct}",
        )

    c.guard("E16.4", verifier_rules)

    def review_projection() -> tuple[bool, str]:
        edge = _reviews([{COL_UTILITY_SCORE: 0.3}, {COL_UTILITY_SCORE: 0.2999}], leak=0.9, util=0.3)
        none = _reviews([{COL_UTILITY_SCORE: 0.0}], leak=0.9, util=None)
        low_met = _reviews([{COL_UTILITY_SCORE: 0.05, COL_LEAKAGE_MASS: 0.0}], leak=0.9, util=0.3)
        return (
            edge == [False, True] and none == [False] and low_met == [True],
            f"at/below floor={edge}, no floor={none}, met row below floor={low_met}",
        )

    c.guard("E16.5", review_projection)

    def reraise_rules() -> tuple[bool, str]:
        def boom() -> None:
            raise KeyError("body")

        try:
            _lowered([{}, {}], leak=0.5, util=0.1, produce=boom)
            body = "returned"
        except KeyError:
            body = "raised"
        # verifier exception: leakage is not a number under a limit; the target fails, the call returns
        out = _lowered([{COL_LEAKAGE_MASS: "not a number"}, {}], leak=0.5, util=0.1).dataframe
        verifier = [bool(v) for v in out[COL_NEEDS_HUMAN_REVIEW]]
        # a rewrite that is not text is unavailable, not an escape
        out = _lowered([{COL_REWRITTEN_TEXT: None}, {}], leak=0.5, util=0.1).dataframe
        unavailable = [bool(v) for v in out[COL_NEEDS_HUMAN_REVIEW]]
        return (
            body == "raised" and verifier == [True, False] and unavailable == [True, False],
            f"body error={body}, verifier error review={verifier}, unavailable review={unavailable}",
        )

    c.guard("E16.6", reraise_rules)

    def review_dtype() -> tuple[bool, str]:
        arrow = _lowered([{}], leak=0.5, util=0.1, review_dtype="bool[pyarrow]").dataframe[COL_NEEDS_HUMAN_REVIEW]
        plain = _lowered([{}], leak=0.5, util=0.1).dataframe[COL_NEEDS_HUMAN_REVIEW]
        return (
            str(arrow.dtype) == "bool[pyarrow]" and str(plain.dtype) == "bool",
            f"existing column dtype={arrow.dtype}, absent column dtype={plain.dtype}",
        )

    c.guard("E16.7", review_dtype)

    def source_positions() -> tuple[bool, str]:
        r = _lowered([{}, {}], leak=0.5, util=0.1, passthrough=(2, 3))
        return r.source_positions == (0, 1, 2, 3), f"source positions={r.source_positions}"

    c.guard("E16.8", source_positions)

    def ready_default() -> tuple[bool, str]:
        absent = _reviews([{}], leak=0.5, util=0.1)
        no = _reviews([{"ready": False, COL_REWRITE_REPLACEMENT_READY: False}], leak=0.5, util=0.1)
        return absent == [False] and no == [True], f"ready absent review={absent}, ready false review={no}"

    c.guard("E16.9", ready_default)

    def fan_out_order() -> tuple[bool, str]:
        rows = [{COL_LEAKAGE_MASS: 0.9}, {}, {COL_UTILITY_SCORE: 0.01}]
        out = _lowered(rows, leak=0.5, util=0.3, scramble=True).dataframe
        text = list(out[COL_REWRITTEN_TEXT])
        review = [bool(v) for v in out[COL_NEEDS_HUMAN_REVIEW]]
        util = list(out[COL_UTILITY_SCORE])
        return (
            text == ["r0", "r1", "r2"] and review == [True, False, True] and util == [0.9, 0.9, 0.01],
            f"texts={text}, review={review}, utility={util}",
        )

    c.guard("E16.10", fan_out_order)

    def ready_coercion() -> tuple[bool, str]:
        got = []
        for value in (np.True_, np.False_, 1, 0):
            got.append(_reviews([{"ready": 1, COL_REWRITE_REPLACEMENT_READY: value}], leak=0.5, util=0.1))
        arrow = pd.array([True, False], dtype="bool[pyarrow]")
        got.append(
            _reviews(
                [
                    {"ready": 1, COL_REWRITE_REPLACEMENT_READY: arrow[0]},
                    {"ready": 1, COL_REWRITE_REPLACEMENT_READY: arrow[1]},
                ],
                leak=0.5,
                util=0.1,
            )
        )
        return got == [
            [False],
            [True],
            [False],
            [True],
            [False, True],
        ], f"numpy, integer and arrow readiness review={got}"

    c.guard("E16.11", ready_coercion)

    def dropped_row() -> tuple[bool, str]:
        out = _lowered([{}, {}, {}], leak=0.5, util=0.1, drop=1).dataframe
        text = list(out[COL_REWRITTEN_TEXT])
        review = [bool(v) for v in out[COL_NEEDS_HUMAN_REVIEW]]
        return (
            text == ["r0", "r2"] and review == [False, False],
            f"surviving texts={text}, review={review}",
        )

    c.guard("E16.12", dropped_row)

    def unmet_without_utility() -> tuple[bool, str]:
        review = _reviews([{COL_ANY_HIGH_LEAKED: True, COL_UTILITY_SCORE: None}], leak=0.5, util=0.3)
        return review == [True], f"not met with no utility, floor set: review={review}"

    c.guard("E16.13", unmet_without_utility)

    def failure_order() -> tuple[bool, str]:
        texts = ["a", "b", "c", "d", "e"]
        detection = _detection(
            ["a", "c", "d", "e"], entity_rows=(0, 1, 2), positions=(0, 2, 3, 4), failures=(_DETECTION_FAILURE,)
        )
        workflow = _stubbed(RewriteWorkflow(Mock()), fail_row=2, failure=_REWRITE_FAILURE)
        result = workflow.run_with_detection(pd.DataFrame({COL_TEXT: texts}), detect=lambda _: detection, **_OPTIONS)
        out = list(result.dataframe[COL_TEXT])
        return (
            result.failed_records == [_DETECTION_FAILURE, _REWRITE_FAILURE]
            and "b" not in out
            and "d" not in out
            and out == ["a", "c", "e"]
            and result.detection is detection,
            f"failures={[f.step for f in result.failed_records]}, output texts={out}",
        )

    c.guard("E16.14", failure_order)

    def reroute_failures() -> tuple[bool, str]:
        detection = _detection(["c", "a", "b"], entity_rows=(0, 1, 2), failures=(_DETECTION_FAILURE,))
        workflow = _stubbed(RewriteWorkflow(Mock()), fail_row=1, failure=_REWRITE_FAILURE)
        frame = pd.DataFrame({COL_TEXT: ["a", "b", "c"]})
        result = workflow.run_with_detection(frame, detect=lambda _: detection, **_OPTIONS)
        return (
            result.failed_records == [_DETECTION_FAILURE, _REWRITE_FAILURE] and result.detection is detection,
            f"rerouted failures={[f.step for f in result.failed_records]}",
        )

    c.guard("E16.15", reroute_failures)

    def position_resolution() -> tuple[bool, str]:
        source = pd.DataFrame({COL_TEXT: ["a", "b"]})
        differs = _detection(["a", "x"], entity_rows=())
        same = _detection(["a", "b"], entity_rows=())
        missing_source = pd.DataFrame({COL_TEXT: ["a", None]})
        missing = _detection(["a", "b"], entity_rows=())
        missing.dataframe[COL_TEXT] = pd.Series(["a", np.nan], dtype=object)
        got = (
            resolve_positions(source, differs),
            resolve_positions(source, same),
            resolve_positions(missing_source, missing),
        )
        return got == (None, (0, 1), (0, 1)), f"differing text={got[0]}, equal text={got[1]}, missing values={got[2]}"

    c.guard("E16.16", position_resolution)

    def positions_include_passthrough() -> tuple[bool, str]:
        detection = _detection(["a", "b", "c"], entity_rows=(0, 2), positions=(0, 1, 2))
        workflow = _stubbed(RewriteWorkflow(Mock()))
        frame = pd.DataFrame({COL_TEXT: ["a", "b", "c"]})
        result = workflow.run_with_detection(frame, detect=lambda _: detection, **_OPTIONS)
        return result.source_positions == (0, 1, 2), f"source positions={result.source_positions}"

    c.guard("E16.17", positions_include_passthrough)

    def combined_stage_name() -> tuple[bool, str]:
        detection = _detection(["a", "b"], entity_rows=(0, 1), positions=(0, 1))
        workflow = _stubbed(CombinedRewriteWorkflow(Mock()))
        with measurement_session() as collector:
            workflow.run_with_detection(pd.DataFrame({COL_TEXT: ["a", "b"]}), detect=lambda _: detection, **_OPTIONS)
        stages = [r["stage"] for r in collector.records if "stage" in r]
        return stages == ["CombinedRewriteWorkflow.run"], f"recorded stages={stages}"

    c.guard("E16.18", combined_stage_name)
    return c.lines
