# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release: dependencies and all-or-none groups withhold met targets, the ledger names causes, and the final
candidate and its release cause are reported per target independent of declaration order."""

from __future__ import annotations

from typing import Any

import pytest

from anonymizer.engine.graph_core.authoring import (
    DATUM,
    TEXT,
    Assessment,
    Dependency,
    GraphScope,
    Limits,
    Operation,
    OutputGroup,
    ProtectionGraph,
    Workflow,
)
from anonymizer.engine.graph_core.execution import ProtectionResult, prepare, protect
from anonymizer.engine.graph_core.kernel import CancelToken, GraphRejection
from tests.graph.helpers import Recorder, graph, run

# A target whose text contains "bad" is not met; every other target is met.
CHECK = Operation.verifier(
    "check",
    lambda text, target: Assessment("bad" not in text, target),
    {"text": TEXT, "target": DATUM},
    checks=("check",),
    subject=("text",),
)


def _run(
    texts: tuple[str, ...],
    deps: tuple[tuple[int, int], ...] = (),
    groups: tuple[tuple[int, ...], ...] = (),
    limits: Limits | None = None,
    **kw: Any,
) -> tuple[ProtectionResult, tuple]:
    """Targets from ``texts``; ``deps`` are (dependent, on) and ``groups`` member positions."""
    scope = GraphScope()
    ts = tuple(scope.datum(t) for t in texts)
    g = ProtectionGraph(
        scope,
        ts,
        dependencies=tuple(Dependency(ts[a], ts[b]) for a, b in deps),
        groups=tuple(OutputGroup(tuple(ts[i] for i in grp)) for grp in groups),
    )
    return protect(prepare(g, Workflow("w", (CHECK,), requires=("check",), **kw), limits=limits)), ts


def _view(r: ProtectionResult, ts: tuple) -> list[tuple[str, bool, str]]:
    return [(o.qualification, o.protected, o.reason) for o in map(r.outcome, ts)]


# --- dependencies and groups ---


def test_dependent_on_not_met_target_is_withheld() -> None:
    r, ts = _run(("a", "bad"), deps=((0, 1),))
    assert _view(r, ts)[0] == ("met", False, "withheld")


def test_group_with_not_met_member_withholds_the_rest() -> None:
    r, ts = _run(("a", "bad"), groups=((0, 1),))
    assert _view(r, ts) == [("met", False, "withheld"), ("not_met", False, "failed")]


def test_met_dependency_releases_both() -> None:
    r, ts = _run(("a", "b"), deps=((0, 1),))
    assert _view(r, ts) == [("met", True, "")] * 2


def test_release_is_independent_of_declaration_order() -> None:
    # 50 targets: every fifth is not met; a chain of dependencies over the even positions and groups of three.
    texts = tuple("bad" if i % 5 == 4 else f"t{i}" for i in range(50))
    deps = tuple((i + 2, i) for i in range(0, 48, 2))
    groups = tuple((i, i + 1, i + 3) for i in range(1, 46, 6))
    forward, fts = _run(texts, deps, groups)
    rev = list(reversed(range(50)))
    backward, bts = _run(
        tuple(texts[i] for i in rev),
        tuple((49 - a, 49 - b) for a, b in reversed(deps)),
        tuple(tuple(49 - i for i in reversed(grp)) for grp in reversed(groups)),
    )
    assert _view(forward, fts) == list(reversed(_view(backward, bts)))
    assert forward.protected


def test_annotation_release_protects_nothing() -> None:
    r, ts = _run(("a", "b"), release="annotation")
    assert not r.protected
    assert _view(r, ts) == [("met", False, "annotation")] * 2


def test_group_withholding_propagates_along_dependency_chain() -> None:
    # Group {0, 3} with 3 not met withholds 0, and with it the chain 2 -> 1 -> 0; 4 is independent.
    r, ts = _run(("a", "b", "c", "bad", "e"), deps=((1, 0), (2, 1)), groups=((0, 3),))
    assert [o.protected for o in map(r.outcome, ts)] == [False, False, False, False, True]


def test_long_chain_withholds_every_target_without_recursion() -> None:
    n = 5000
    texts = tuple(f"t{i}" for i in range(n - 1)) + ("bad",)
    chain = tuple((i, i + 1) for i in range(n - 1))
    forward, fts = _run(texts, chain, limits=Limits(max_targets=n))
    backward, bts = _run(texts, tuple(reversed(chain)), limits=Limits(max_targets=n))
    fv, bv = _view(forward, fts), _view(backward, bts)
    assert fv == bv
    assert not forward.protected
    assert fv[0][2] == "withheld"


# --- ledger causes and scale ---


def _chain(size: int, cycle: bool) -> ProtectionGraph:
    scope = GraphScope()
    ts = tuple(scope.datum(f"t{i}") for i in range(size))
    deps = [Dependency(ts[i], ts[i + 1]) for i in range(size - 1)]  # each target depends on the next
    return ProtectionGraph(scope, ts, dependencies=(*deps, *([Dependency(ts[-1], ts[0])] if cycle else ())))


def test_failure_before_cancellation_keeps_its_nodes_blocked() -> None:
    g, (a, b) = graph("fail", "cancel")
    token = CancelToken()

    def first(text: str) -> str:
        if text == "fail":
            raise ValueError("x")
        token.cancel()
        return text

    steps = tuple(Operation(n, lambda text: text) for n in ("second", "third"))
    r = protect(prepare(g, Workflow("w", (Operation("first", first), *steps))), cancel=token)
    ledger = [[s for t, _, s in r.ledger() if t == d] for d in (a, b)]
    assert ledger == [["failed", "blocked", "blocked"], ["succeeded", "cancelled", "cancelled"]]


def test_thousand_target_chain_is_released() -> None:
    r = protect(prepare(_chain(1_000, cycle=False), Workflow("w", (Operation("op", lambda text: text),))))
    assert len(r.outcomes) == 1_000


def test_thousand_target_cycle_is_rejected() -> None:
    with pytest.raises(GraphRejection) as info:
        _chain(1_000, cycle=True)
    assert info.value.code == "dependency_cycle"


# --- final candidate and release cause ---


def _boom(text: str) -> str:
    raise ValueError("b")


def test_later_failure_makes_target_absent_but_earlier_text_stays_readable() -> None:
    g, (t,) = graph("Alice")
    r = run(g, Operation("A", lambda text: "a-text"), Operation("B", _boom))
    o = r.outcome(t)
    assert not o.available
    assert o.reason == "raised"
    assert r.content() == ((t, None),)
    assert r.value(t, "text", producer="A") == "a-text"


def test_consumer_of_failed_port_is_blocked() -> None:
    g, _ = graph("Alice")
    consumer = Operation("C", lambda text: text, {"text": TEXT})
    ledger = dict((op, s) for _, op, s in run(g, Operation("A", str), Operation("B", _boom), consumer).ledger())
    assert ledger["C"] == "blocked"


def test_failed_privacy_check_keeps_final_text_available_not_met() -> None:
    g, (t,) = graph("Alice")
    rec = Recorder()
    r = run(g, rec.transform("A", str), rec.transform("B", lambda text: "b-text"), rec.verifier("check", "b-text"))
    o = r.outcome(t)
    assert o.available
    assert o.qualification == "not_met"
    assert r.content() == ((t, "b-text"),)


def test_group_withheld_member_names_its_cause() -> None:
    r, (a, b) = _run(("a", "bad"), groups=((0, 1),))
    oa, ob = r.outcome(a), r.outcome(b)
    assert oa.qualification == "met"
    assert not oa.protected
    assert oa.withheld_by == (b,)
    assert ob.reason == "failed"


def test_dependent_names_failed_dependency() -> None:
    r, (dep, on) = _run(("c", "bad"), deps=((0, 1),))
    assert r.outcome(dep).withheld_by == (on,)
