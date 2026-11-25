# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Covering verifiers decide qualification; phase-scoped bodies feed verifiers through their own ports.

Only a verifier whose subject derives from the final text covers a target; evaluation judges report
per-row verdicts without qualifying anything.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from anonymizer.engine.constants import (
    COL_DETECTION_VALID,
    COL_ENTITY_COVERAGE,
    COL_JUDGE_EVALUATION,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.graph_core.authoring import DATUM, TEXT, Assessment, Operation, Requests, Workflow
from anonymizer.engine.graph_core.execution import Reason, prepare
from anonymizer.engine.graph_core.kernel import GraphRejection, register_kind
from anonymizer.engine.ndd.failed_record import FailedRecord
from anonymizer.interface.anonymizer import COVERAGE_METRICS, REWRITE_METRICS, _evaluate_on_graph, _Judge
from tests.graph.helpers import graph, rejected, run

TEXTS = ("alpha", "beta", "gamma")
METRICS = register_kind("sp17-metrics", lambda v: isinstance(v, dict))


def _upper() -> Operation:
    return Operation("upper", lambda text: text.upper())


def _verifier(name: str, verdicts: dict[str, bool | None], reason: str) -> Operation:
    """A covering verifier: the verdict is looked up by the candidate text (upper-cased by ``upper``)."""

    def assess(text: str, target: Any) -> Assessment:
        verdict = verdicts[text.lower()]
        return Assessment(verdict, target, "" if verdict else reason)

    return Operation.verifier(name, assess, {"text": TEXT, "target": DATUM}, checks=(name,), subject=("text",))


def _body(name: str, port: str, tag: str) -> Operation:
    def rows(target: tuple) -> list[dict[str, str]]:
        return [{"v": f"{tag}-{i}"} for i, _ in enumerate(target)]

    return Operation.per_invocation(name, rows, {"target": DATUM}, {port: METRICS}, Requests.opaque())


def _reader(name: str, port: str, seen: list[str]) -> Operation:
    """A non-covering verifier reading ``port``; the callable's parameter is named after the port it binds."""

    def assess(target: Any, **ports: Any) -> Assessment:
        seen.append(ports[port]["v"])
        return Assessment(True, target)

    return Operation.verifier(name, assess, {port: METRICS, "target": DATUM})


def _keep(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[FailedRecord]]:
    return frame, []


def _frame(**columns: list[Any]) -> pd.DataFrame:
    """A frame with candidate text ``t`` and object-typed columns, so ``None`` and NaN stay distinct."""
    size = len(next(iter(columns.values())))
    data = {"t": [f"row{i}" for i in range(size)]}
    data.update({name: pd.Series(values, dtype=object) for name, values in columns.items()})
    return pd.DataFrame(data)


def test_two_covering_verifiers_must_both_be_satisfied() -> None:
    g, (a, b, d) = graph(*TEXTS)
    first = _verifier("first", {"alpha": True, "beta": True, "gamma": False}, "first_failed")
    second = _verifier("second", {"alpha": True, "beta": False, "gamma": True}, "second_failed")
    prepared = prepare(g, Workflow("w", (_upper(), first, second), requires=("first", "second")))
    r = run(g, _upper(), first, second)
    outcomes = [r.outcome(t) for t in (a, b, d)]
    reasons = [[r.state.rows[u].outputs[i]["assessment"].reason for i in (1, 2)] for u in range(3)]
    assert prepared.covering_verifiers == {1, 2}
    assert [o.qualification for o in outcomes] == ["met", "not_met", "not_met"]
    assert [o.reasons for o in outcomes] == [
        (),
        (Reason("failed", "second", "second", "second_failed"),),
        (Reason("failed", "first", "first", "first_failed"),),
    ]
    assert reasons == [["", ""], ["", "second_failed"], ["first_failed", ""]]
    assert set(r.protected) == {a}


def test_covering_verifier_that_could_not_assess_leaves_target_not_met() -> None:
    g, (a, b, d) = graph(*TEXTS)
    unsure = _verifier("unsure", {"alpha": True, "beta": None, "gamma": True}, "")
    r = run(g, _upper(), unsure)
    outcomes = [r.outcome(t) for t in (a, b, d)]
    assert [o.qualification for o in outcomes] == ["met", "not_met", "met"]
    assert outcomes[1].reasons == (Reason("could_not_assess", "unsure", "unsure"),)
    assert outcomes[1].terminal == "success"
    assert r.state.rows[1].outputs[1]["assessment"].satisfied is None


def test_only_final_text_verifiers_cover() -> None:
    g, _ = graph(*TEXTS)
    early = _verifier("early", dict.fromkeys(("alpha", "beta", "gamma"), True), "")
    tail = Operation("tail", lambda text: text + "!")
    no_text = Operation.verifier("no-text", lambda target: Assessment(True, target), {"target": DATUM})
    late = _verifier("late", dict.fromkeys(("alpha!", "beta!", "gamma!"), True), "")
    assert prepare(g, Workflow("w", (_upper(), early, tail, no_text))).covering_verifiers == frozenset()
    assert prepare(g, Workflow("w", (_upper(), early, tail, no_text, late))).covering_verifiers == {4}
    assert rejected(lambda: run(g, _upper(), early, tail, no_text)) == "required_check_not_final"


def test_undeclared_required_check_is_rejected_before_any_callback() -> None:
    g, _ = graph(*TEXTS)
    calls: list[str] = []
    counted = Operation("counted", lambda text: calls.append(text) or text)
    late = _verifier("late", dict.fromkeys(("alpha!", "beta!", "gamma!"), True), "")
    with pytest.raises(GraphRejection) as info:
        prepare(g, Workflow("w", (counted, late), requires=("late", "typo")))
    assert info.value.code == "required_check_not_final"
    assert info.value.detail == "typo"
    assert calls == []


def test_bodies_feed_verifiers_through_their_own_ports() -> None:
    g, _ = graph(*TEXTS)
    own: list[str] = []
    own_second: list[str] = []
    run(
        g,
        _body("one", "one_metrics", "one"),
        _body("two", "two_metrics", "two"),
        _reader("reads-one", "one_metrics", own),
        _reader("reads-two", "two_metrics", own_second),
    )
    shared: list[str] = []
    shared_second: list[str] = []
    run(
        g,
        _body("one", "shared", "one"),
        _body("two", "shared", "two"),
        _reader("reads-shared-a", "shared", shared),
        _reader("reads-shared-b", "shared", shared_second),
    )
    assert own == ["one-0", "one-1", "one-2"]
    assert own_second == ["two-0", "two-1", "two-2"]
    # a shared port name feeds the last body
    assert shared == ["two-0", "two-1", "two-2"]
    assert shared_second == ["two-0", "two-1", "two-2"]


def test_coverage_verdicts_per_row_qualify_nothing() -> None:
    nan = float("nan")
    coverage = _Judge("coverage-judge", "metrics", COVERAGE_METRICS, (COL_ENTITY_COVERAGE,), _keep)
    # Rows carry record ids: a judge's rows are placed only by id.
    frame = _frame(**{COL_ENTITY_COVERAGE: [1.0, 0.5, None, nan], RECORD_ID_COLUMN: ["r0", "r1", "r2", "r3"]})
    _, _, r = _evaluate_on_graph(frame, "t", [coverage])
    verdicts = [(x.satisfied, x.reason) for x in (r.state.rows[i].outputs[1]["assessment"] for i in range(4))]
    assert [r.outcome(t).qualification for t in r.targets] == ["not_assessed"] * 4
    assert verdicts == [
        (True, ""),
        (False, "entities_leaked"),
        (None, "judge_unavailable"),
        (None, "judge_unavailable"),
    ]


def test_non_covering_verdicts_never_change_qualification() -> None:
    wide = _Judge(
        "wide", "metrics", COVERAGE_METRICS, (COL_ENTITY_COVERAGE, COL_DETECTION_VALID, COL_JUDGE_EVALUATION), _keep
    )
    frame = _frame(
        **{
            COL_ENTITY_COVERAGE: [1.0, 1.0, 1.0],
            COL_DETECTION_VALID: [True, 0.5, False],
            COL_JUDGE_EVALUATION: [{"score": 1}, None, {"score": 2}],
            RECORD_ID_COLUMN: ["r0", "r1", "r2"],
        }
    )
    _, _, r = _evaluate_on_graph(frame, "t", [wide])

    def verdicts(port: int) -> list[tuple[bool | None, str]]:
        out = [r.state.rows[i].outputs[port]["assessment"] for i in range(3)]
        return [(x.satisfied, x.reason) for x in out]

    assert [r.outcome(t).qualification for t in r.targets] == ["not_assessed"] * 3
    assert verdicts(2) == [(True, ""), (False, "detection_invalid"), (False, "detection_invalid")]
    assert verdicts(3) == [(True, ""), (None, "judge_unavailable"), (True, "")]


def test_judge_bodies_chain_frames_and_failures() -> None:
    seen: list[list[str]] = []
    fail_a, fail_b = FailedRecord("1", "a", "x"), FailedRecord("2", "b", "y")

    def first(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[FailedRecord]]:
        return frame.assign(added=1), [fail_a]

    def second(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[FailedRecord]]:
        seen.append(list(frame.columns))
        return frame.assign(last=2), [fail_b]

    chain = [
        _Judge("first", "one", REWRITE_METRICS, (), first),
        _Judge("second", "two", REWRITE_METRICS, (), second),
    ]
    out, failed, _ = _evaluate_on_graph(_frame(x=[1, 2]), "t", chain)
    assert seen == [["t", "x", "added"]]
    assert list(out.columns) == ["t", "x", "added", "last"]
    assert failed == [fail_a, fail_b]


def test_judge_body_error_is_raised_unchanged() -> None:
    def boom(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[FailedRecord]]:
        raise ValueError("x")

    with pytest.raises(ValueError) as info:
        _evaluate_on_graph(_frame(x=[1]), "t", [_Judge("boom", "p", REWRITE_METRICS, (), boom)])
    assert str(info.value) == "x"


def test_each_judge_body_runs_once_on_an_empty_frame() -> None:
    sizes: list[int] = []

    def counting(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[FailedRecord]]:
        sizes.append(len(frame))
        return frame, []

    empty = _frame(x=[1]).iloc[0:0]
    judges = [_Judge("c1", "p", REWRITE_METRICS, (), counting), _Judge("c2", "q", REWRITE_METRICS, (), counting)]
    _evaluate_on_graph(empty, "t", judges)
    assert sizes == [0, 0]


def test_row_without_candidate_text_becomes_an_empty_text_target() -> None:
    _, _, r = _evaluate_on_graph(
        pd.DataFrame({"t": ["a", None]}), "t", [_Judge("keep", "p", REWRITE_METRICS, (), _keep)]
    )
    assert [r.state.rows[i].seeds["text"] for i in range(2)] == ["a", ""]
