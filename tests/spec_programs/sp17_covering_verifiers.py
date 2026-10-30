# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-17 covering verifiers and invocation-scoped bodies with their own ports (E17.1-E17.9)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from anonymizer.engine.constants import COL_DETECTION_VALID, COL_ENTITY_COVERAGE, COL_JUDGE_EVALUATION
from anonymizer.engine.graph_core.authoring import DATUM, TEXT, Assessment, Operation, Requests, Workflow
from anonymizer.engine.graph_core.execution import prepare
from anonymizer.engine.graph_core.kernel import register_kind
from anonymizer.engine.ndd.failed_record import FailedRecord
from anonymizer.interface.anonymizer import COVERAGE_METRICS, REWRITE_METRICS, _evaluate_on_graph, _Judge
from tests.spec_programs._builder import Checks, graph, run

TEXTS = ("alpha", "beta", "gamma")
METRICS = register_kind("sp17-metrics", lambda v: isinstance(v, dict))


def _upper() -> Operation:
    return Operation("upper", lambda text: text.upper())


def _verifier(name: str, verdicts: dict[str, bool | None], reason: str) -> Operation:
    """A covering verifier: the verdict is looked up by the candidate text (upper-cased by ``upper``)."""

    def assess(text: str, target: Any) -> Assessment:
        verdict = verdicts[text.lower()]
        return Assessment(verdict, target, "" if verdict else reason)

    return Operation.verifier(name, assess, {"text": TEXT, "target": DATUM})


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


def program() -> list[dict[str, str]]:
    c = Checks("SP-17")
    g, (a, b, d) = graph(*TEXTS)

    # E17.1 two covering verifiers: met only when both are satisfied; each reason stays readable per verifier.
    first = _verifier("first", {"alpha": True, "beta": True, "gamma": False}, "first_failed")
    second = _verifier("second", {"alpha": True, "beta": False, "gamma": True}, "second_failed")
    prepared = prepare(g, Workflow("w", (_upper(), first, second)))
    r = run(g, _upper(), first, second)
    outcomes = [r.outcome(t) for t in (a, b, d)]
    reasons = [[r.state.rows[u].outputs[i]["assessment"].reason for i in (1, 2)] for u in range(3)]
    c.check(
        "E17.1",
        prepared.covering_verifiers == {1, 2}
        and [o.qualification for o in outcomes] == ["met", "not_met", "not_met"]
        and [o.reason for o in outcomes] == ["", "unsatisfied", "unsatisfied"]
        and reasons == [["", ""], ["", "second_failed"], ["first_failed", ""]]
        and set(r.protected) == {a},
        f"covering={sorted(prepared.covering_verifiers)}, qualification={[o.qualification for o in outcomes]}, "
        f"outcome reasons={[o.reason for o in outcomes]}, verifier reasons={reasons}",
    )

    # E17.2 a covering verifier that could not assess leaves the target not_met, never met.
    unsure = _verifier("unsure", {"alpha": True, "beta": None, "gamma": True}, "")
    r2 = run(g, _upper(), unsure)
    o2 = [r2.outcome(t) for t in (a, b, d)]
    c.check(
        "E17.2",
        [o.qualification for o in o2] == ["met", "not_met", "met"]
        and o2[1].reason == "unsatisfied"
        and o2[1].terminal == "success"
        and r2.state.rows[1].outputs[1]["assessment"].satisfied is None,
        f"qualification={[o.qualification for o in o2]}, reason={o2[1].reason}, terminal={o2[1].terminal}",
    )

    # E17.3 only a verifier whose text input binds to the final text producer is covering.
    early = _verifier("early", dict.fromkeys(("alpha", "beta", "gamma"), True), "")
    tail = Operation("tail", lambda text: text + "!")
    no_text = Operation.verifier("no-text", lambda target: Assessment(True, target), {"target": DATUM})
    late = _verifier("late", dict.fromkeys(("alpha!", "beta!", "gamma!"), True), "")
    p3 = prepare(g, Workflow("w", (_upper(), early, tail, no_text)))
    p3_late = prepare(g, Workflow("w", (_upper(), early, tail, no_text, late)))
    r3 = run(g, _upper(), early, tail, no_text)
    c.check(
        "E17.3",
        p3.covering_verifiers == frozenset()
        and p3_late.covering_verifiers == {4}
        and r3.outcome(a).qualification == "not_assessed"
        and r3.state.rows[0].outputs[1]["assessment"].satisfied is True,
        f"covering earlier-bound/no-text={sorted(p3.covering_verifiers)}, with a final-text verifier="
        f"{sorted(p3_late.covering_verifiers)}, qualification={r3.outcome(a).qualification}",
    )

    # E17.4 invocation bodies with different ports feed their own verifiers; a shared port name feeds the last body.
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
    c.check(
        "E17.4",
        own == ["one-0", "one-1", "one-2"]
        and own_second == ["two-0", "two-1", "two-2"]
        and shared == shared_second == ["two-0", "two-1", "two-2"],
        f"own ports: first verifier saw {own}, second saw {own_second}; shared port: both saw {shared}",
    )

    # E17.5 coverage verdicts per row: 1.0 is met, a fraction leaked, None and NaN unavailable (never met).
    nan = float("nan")
    coverage = _Judge("coverage", "metrics", COVERAGE_METRICS, (COL_ENTITY_COVERAGE,), _keep)
    _, _, r5 = _evaluate_on_graph(_frame(**{COL_ENTITY_COVERAGE: [1.0, 0.5, None, nan]}), "t", [coverage])
    quals5 = [r5.outcome(t).qualification for t in r5.targets]
    verdicts5 = [(a5.satisfied, a5.reason) for a5 in (r5.state.rows[i].outputs[1]["assessment"] for i in range(4))]
    c.check(
        "E17.5",
        quals5 == ["met", "not_met", "not_met", "not_met"]
        and verdicts5
        == [(True, ""), (False, "entities_leaked"), (None, "judge_unavailable"), (None, "judge_unavailable")],
        f"qualification={quals5}, coverage assessments={verdicts5}",
    )

    # E17.6 non-covering verdicts never change qualification: fractions grade, the rewrite judge only reports availability.
    wide = _Judge(
        "wide",
        "metrics",
        COVERAGE_METRICS,
        (COL_ENTITY_COVERAGE, COL_DETECTION_VALID, COL_JUDGE_EVALUATION),
        _keep,
    )
    frame6 = _frame(
        **{
            COL_ENTITY_COVERAGE: [1.0, 1.0, 1.0],
            COL_DETECTION_VALID: [True, 0.5, False],
            COL_JUDGE_EVALUATION: [{"score": 1}, None, {"score": 2}],
        }
    )
    _, _, r6 = _evaluate_on_graph(frame6, "t", [wide])
    quals6 = [r6.outcome(t).qualification for t in r6.targets]
    detection6 = [
        (r6.state.rows[i].outputs[2]["assessment"].satisfied, r6.state.rows[i].outputs[2]["assessment"].reason)
        for i in range(3)
    ]
    rewrite6 = [
        (r6.state.rows[i].outputs[3]["assessment"].satisfied, r6.state.rows[i].outputs[3]["assessment"].reason)
        for i in range(3)
    ]
    c.check(
        "E17.6",
        quals6 == ["met", "met", "met"]
        and detection6 == [(True, ""), (False, "detection_invalid"), (False, "detection_invalid")]
        and rewrite6 == [(True, ""), (None, "judge_unavailable"), (True, "")],
        f"qualification={quals6}, detection-validity={detection6}, rewrite-judge={rewrite6}",
    )

    # E17.7 bodies chain: each receives the frame the previous one returned; the last frame and all failures come back.
    seen7: list[list[str]] = []
    fail_a, fail_b = FailedRecord("1", "a", "x"), FailedRecord("2", "b", "y")

    def first7(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[FailedRecord]]:
        return frame.assign(added=1), [fail_a]

    def second7(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[FailedRecord]]:
        seen7.append(list(frame.columns))
        return frame.assign(last=2), [fail_b]

    chain = [
        _Judge("first", "one", REWRITE_METRICS, (), first7),
        _Judge("second", "two", REWRITE_METRICS, (), second7),
    ]
    out7, failed7, _ = _evaluate_on_graph(_frame(x=[1, 2]), "t", chain)
    c.check(
        "E17.7",
        seen7 == [["t", "x", "added"]]
        and list(out7.columns) == ["t", "x", "added", "last"]
        and failed7 == [fail_a, fail_b],
        f"second body saw {seen7}, output columns={list(out7.columns)}, failures={[f.record_id for f in failed7]}",
    )

    # E17.8 a body's own error is raised unchanged; with no rows each body still runs once on the empty frame.
    def boom(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[FailedRecord]]:
        raise ValueError("x")

    kind = REWRITE_METRICS
    raised: list[str] = []
    try:
        _evaluate_on_graph(_frame(x=[1]), "t", [_Judge("boom", "p", kind, (), boom)])
    except ValueError as exc:
        raised.append(str(exc))
    sizes: list[int] = []

    def counting(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[FailedRecord]]:
        sizes.append(len(frame))
        return frame, []

    empty = _frame(x=[1]).iloc[0:0]
    _evaluate_on_graph(empty, "t", [_Judge("c1", "p", kind, (), counting), _Judge("c2", "q", kind, (), counting)])
    c.check(
        "E17.8",
        raised == ["x"] and sizes == [0, 0],
        f"error raised={raised}, empty-frame body calls with sizes={sizes}",
    )

    # E17.9 a row without candidate text becomes a target whose text is the empty string.
    _, _, r9 = _evaluate_on_graph(pd.DataFrame({"t": ["a", None]}), "t", [_Judge("keep", "p", kind, (), _keep)])
    texts9 = [r9.state.rows[i].seeds["text"] for i in range(2)]
    c.check("E17.9", texts9 == ["a", ""], f"target texts={texts9}")
    return c.lines
