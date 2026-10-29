# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-17 covering verifiers and invocation-scoped bodies with their own ports (E17.1-E17.4)."""

from __future__ import annotations

from typing import Any

from anonymizer.engine.graph_core.authoring import DATUM, TEXT, Assessment, Operation, Requests, Workflow
from anonymizer.engine.graph_core.execution import prepare
from anonymizer.engine.graph_core.kernel import register_kind
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

    # E17.3 only a verifier whose text input binds to the final text producer is covering (RFC :712).
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
    return c.lines
