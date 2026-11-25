# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A declared Record is released whole by default; ``prepare(..., partial_release=True)`` releases each member alone."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from anonymizer.engine.graph_core.authoring import (
    DATUM,
    TEXT,
    Assessment,
    Dependency,
    GraphScope,
    Operation,
    OutputGroup,
    ProtectionGraph,
    Workflow,
)
from anonymizer.engine.graph_core.execution import ProtectionResult, prepare, protect
from anonymizer.engine.graph_core.kernel import GraphRejection

# A target whose text contains "bad" is not met; every other target is met.
CHECK = Operation.verifier(
    "check",
    lambda text, target: Assessment("bad" not in text, target),
    {"text": TEXT, "target": DATUM},
    checks=("check",),
    subject=("text",),
)
WORKFLOW = Workflow("w", (CHECK,), requires=("check",))


def _run(
    texts: tuple[str, ...],
    records: tuple[tuple[int, ...], ...] = (),
    groups: tuple[tuple[int, ...], ...] = (),
    deps: tuple[tuple[int, int], ...] = (),
    *,
    partial: bool = False,
    **kw: Any,
) -> tuple[ProtectionResult, tuple]:
    scope = GraphScope()
    ts = tuple(scope.datum(t) for t in texts)
    g = ProtectionGraph(
        scope,
        ts,
        dependencies=tuple(Dependency(ts[a], ts[b]) for a, b in deps),
        groups=tuple(OutputGroup(tuple(ts[i] for i in grp)) for grp in groups),
        records=tuple(scope.record(tuple(ts[i] for i in r)) for r in records),
    )
    return protect(prepare(g, kw.pop("workflow", WORKFLOW), partial_release=partial, **kw)), ts


def _released(r: ProtectionResult, ts: tuple) -> list[bool]:
    return [r.outcome(t).protected for t in ts]


def test_a_failing_member_withholds_its_record() -> None:
    r, (a, b) = _run(("ok", "bad"), records=((0, 1),))
    oa, ob = r.outcome(a), r.outcome(b)
    assert (oa.qualification, oa.protected, oa.reason, oa.withheld_by) == ("met", False, "withheld", (b,))
    assert (ob.qualification, ob.protected, ob.reason) == ("not_met", False, "failed")


def test_partial_release_releases_each_member_on_its_own_checks() -> None:
    r, ts = _run(("ok", "bad"), records=((0, 1),), partial=True)
    assert _released(r, ts) == [True, False]
    assert r.outcome(ts[0]).withheld_by == ()


def test_records_with_no_member_in_common_release_independently() -> None:
    # Two tickets of subject and body; ticket 1's body fails, ticket 2 passes.
    r, ts = _run(("subj1", "bad body1", "subj2", "body2"), records=((0, 1), (2, 3)))
    assert _released(r, ts) == [False, False, True, True]
    assert r.outcome(ts[0]).withheld_by == (ts[1],)
    r, ts = _run(("subj1", "bad body1", "subj2", "body2"), records=((0, 1), (2, 3)), partial=True)
    assert _released(r, ts) == [True, False, True, True]


def test_a_member_that_raises_withholds_its_record() -> None:
    def boom(text: str) -> str:
        if "boom" in text:
            raise RuntimeError("no")
        return text

    workflow = Workflow("w", (Operation("transform", boom), CHECK), requires=("check",))
    r, (a, b) = _run(("ok", "boom"), records=((0, 1),), workflow=workflow)
    assert r.outcome(b).terminal == "failed"
    assert (r.outcome(a).protected, r.outcome(a).withheld_by) == (False, (b,))
    r, ts = _run(("ok", "boom"), records=((0, 1),), workflow=workflow, partial=True)
    assert _released(r, ts) == [True, False]


@pytest.mark.parametrize("partial", [False, True])
def test_an_output_group_over_two_records_withholds_both(partial: bool) -> None:
    r, ts = _run(("a1", "a2", "b1", "bad b2"), records=((0, 1), (2, 3)), groups=((0, 1, 2, 3),), partial=partial)
    assert _released(r, ts) == [False, False, False, False]


def test_a_group_that_cuts_a_record_merges_with_it() -> None:
    # Record A = (a1, a2); group (a1, s); s fails.
    texts = ("a1", "a2", "bad s")
    r, ts = _run(texts, records=((0, 1),), groups=((0, 2),))
    assert _released(r, ts) == [False, False, False]
    r, ts = _run(texts, records=((0, 1),), groups=((0, 2),), partial=True)
    assert _released(r, ts) == [False, True, False]


def test_a_dependency_across_records_withholds_both_records() -> None:
    # Record A = (x, x2), Record B = (y, y2); x depends on y; y fails.
    r, (x, x2, y, y2) = _run(("x", "x2", "bad y", "y2"), records=((0, 1), (2, 3)), deps=((0, 2),))
    assert _released(r, (x, x2, y, y2)) == [False, False, False, False]
    assert r.outcome(x).withheld_by == (y,)
    assert r.outcome(y2).withheld_by == (y,)
    assert r.outcome(x2).withheld_by == (x,)
    r, ts = _run(("x", "x2", "bad y", "y2"), records=((0, 1), (2, 3)), deps=((0, 2),), partial=True)
    assert _released(r, ts) == [False, True, False, True]


def test_record_release_is_independent_of_declaration_order() -> None:
    texts = tuple(f"t{i}" if i % 7 else f"bad t{i}" for i in range(30))
    records = tuple((i, i + 1, i + 2) for i in range(0, 30, 3))
    forward, ts = _run(texts, records=records)
    backward, ts2 = _run(texts, records=tuple(reversed(records)))
    assert _released(forward, ts) == _released(backward, ts2)
    assert [len(forward.outcome(t).withheld_by) for t in ts] == [len(backward.outcome(t).withheld_by) for t in ts2]


@pytest.mark.parametrize("records", [(), ((0,), (1,), (2,))], ids=["none", "singletons"])
def test_graphs_without_multi_member_records_release_as_partial_release(records: tuple) -> None:
    texts = ("a", "bad b", "c")
    default, ts = _run(texts, records=records, groups=((0, 1),), deps=((2, 1),))
    partial, ts2 = _run(texts, records=records, groups=((0, 1),), deps=((2, 1),), partial=True)
    assert [dataclasses.astuple(default.outcome(t))[:4] for t in ts] == [
        dataclasses.astuple(partial.outcome(t))[:4] for t in ts2
    ]
    assert [len(default.outcome(t).withheld_by) for t in ts] == [len(partial.outcome(t).withheld_by) for t in ts2]


def test_an_annotation_workflow_releases_nothing() -> None:
    workflow = Workflow("w", (CHECK,), requires=("check",), release="annotation")
    r, ts = _run(("a", "b"), records=((0, 1),), workflow=workflow)
    assert _released(r, ts) == [False, False]


def test_partial_release_is_validated_and_sealed() -> None:
    scope = GraphScope()
    g = ProtectionGraph(scope, (scope.datum("a"),))
    with pytest.raises(GraphRejection) as info:
        prepare(g, WORKFLOW, partial_release="yes")  # ty: ignore[invalid-argument-type]
    assert info.value.code == "invalid_partial_release"
    prepared = prepare(g, WORKFLOW)
    with pytest.raises(GraphRejection) as info:
        protect(dataclasses.replace(prepared, partial_release=True))
    assert info.value.code == "not_prepared"


def test_default_prepare_withholds_a_whole_record_and_a_group_sharing_a_member() -> None:
    scope = GraphScope()
    a, b, c, d = (scope.datum(t) for t in ("ok a", "bad b", "ok c", "ok d"))
    g = ProtectionGraph(scope, (a, b, c, d), groups=(OutputGroup((b, c)),), records=(scope.record((a, b)),))
    r = protect(prepare(g, WORKFLOW))
    assert [r.outcome(t).protected for t in (a, b, c, d)] == [False, False, False, True]
    assert (r.outcome(a).reason, r.outcome(a).withheld_by) == ("withheld", (b,))
