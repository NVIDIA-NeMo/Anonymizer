# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records: caller-declared groups of targets; a target in no declared Record is its own singleton Record."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

import anonymizer.graph as graph
from anonymizer.engine.graph_core.authoring import _RESERVED_SEED_OUTPUTS
from anonymizer.engine.graph_core.execution import SEEDS

SECRET = "Alice Smith lives at 12 Elm Street"


def build(*texts: str) -> tuple[graph.GraphScope, tuple[graph.Datum, ...]]:
    scope = graph.GraphScope()
    return scope, tuple(scope.datum(t) for t in texts)


def rejection(fn: Callable[[], Any]) -> graph.GraphRejection:
    with pytest.raises(graph.GraphRejection) as info:
        fn()
    return info.value


def run(g: graph.ProtectionGraph, *steps: graph.Operation) -> graph.ProtectionResult:
    return graph.protect(graph.prepare(g, graph.Workflow("w", steps)))


def _not_tuple(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return graph.ProtectionGraph(scope, (a, b), records=[scope.record((a,))])  # ty: ignore[invalid-argument-type]


def _not_a_record(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return graph.ProtectionGraph(scope, (a, b), records=((a, b),))  # ty: ignore[invalid-argument-type]


def _members_not_tuple(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return scope.record([a, b])  # ty: ignore[invalid-argument-type]


def _member_not_datum(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return scope.record((a, SECRET))  # ty: ignore[invalid-argument-type]


def _bare_string(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return scope.record(SECRET)  # ty: ignore[invalid-argument-type]


def _built_member_not_datum(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    r = dataclasses.replace(scope.record((a,)), members=(SECRET,))
    return graph.ProtectionGraph(scope, (a, b), records=(r,))


def _empty(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return scope.record(())


def _built_empty(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    r = dataclasses.replace(scope.record((a,)), members=())
    return graph.ProtectionGraph(scope, (a, b), records=(r,))


def _foreign_record(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return graph.ProtectionGraph(scope, (a, b), records=(graph.GraphScope().record((a,)),))


def _foreign_member(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return graph.ProtectionGraph(scope, (a, b), records=(scope.record((a, graph.GraphScope().datum(SECRET))),))


def _context_member(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return graph.ProtectionGraph(scope, (a, b), records=(scope.record((a, ctx)),))


def _in_two_records(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return graph.ProtectionGraph(scope, (a, b), records=(scope.record((a, b)), scope.record((b,))))


def _twice_in_one(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return graph.ProtectionGraph(scope, (a, b), records=(scope.record((a, a)),))


def _listed_twice(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    r = scope.record((a,))
    return graph.ProtectionGraph(scope, (a, b), records=(r, r))


def _record_of_non_target(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return graph.ProtectionGraph(scope, (a, b)).record_of(ctx)


def _record_of_foreign(scope: graph.GraphScope, a: graph.Datum, b: graph.Datum, ctx: graph.Datum) -> Any:
    return graph.ProtectionGraph(scope, (a, b)).record_of(graph.GraphScope().datum(SECRET))


REJECTIONS = {
    "not_tuple": (_not_tuple, "invalid_record", ""),
    "not_a_record": (_not_a_record, "invalid_record", ""),
    "members_not_tuple": (_members_not_tuple, "invalid_record", ""),
    "member_not_datum": (_member_not_datum, "invalid_record", ""),
    "bare_string": (_bare_string, "invalid_record", ""),
    "built_member_not_datum": (_built_member_not_datum, "invalid_record", ""),
    "empty": (_empty, "empty_record", ""),
    "built_empty": (_built_empty, "empty_record", ""),
    "foreign_record": (_foreign_record, "foreign_record", ""),
    "foreign_member": (_foreign_member, "foreign_datum", "record_member"),
    "context_member": (_context_member, "unknown_reference", "record_member"),
    "in_two_records": (_in_two_records, "record_overlap", ""),
    "twice_in_one": (_twice_in_one, "record_overlap", ""),
    "listed_twice": (_listed_twice, "record_overlap", ""),
    "record_of_non_target": (_record_of_non_target, "unknown_reference", "target"),
    "record_of_foreign": (_record_of_foreign, "foreign_datum", "target"),
}


@pytest.mark.parametrize("case", REJECTIONS)
def test_invalid_records_are_rejected_with_a_fixed_code_and_no_content(case: str) -> None:
    fn, code, detail = REJECTIONS[case]
    scope, (a, b, ctx) = build(SECRET, SECRET + " too", SECRET + " again")
    exc = rejection(lambda: fn(scope, a, b, ctx))
    assert (exc.code, exc.detail) == (code, detail)
    assert isinstance(exc, ValueError)
    for secret in (SECRET, scope._token, a._id, b._id, ctx._id):
        assert secret not in str(exc) and secret not in repr(exc)


def test_record_and_graph_reprs_show_no_text_or_ids() -> None:
    scope, (a, b) = build(SECRET, SECRET + " too")
    r = scope.record((a, b))
    g = graph.ProtectionGraph(scope, (a, b), records=(r,))
    assert repr(r) == "Record(<opaque>, 2 members)"
    for secret in (SECRET, scope._token, r._id, a._id, b._id):
        assert secret not in repr(r) and secret not in repr(g)
        assert all(secret not in repr(x) for x in g.all_records)


def test_one_member_record_is_allowed_and_differs_from_the_singleton() -> None:
    scope, (a, b) = build("one", "two")
    r = scope.record((a,))
    g = graph.ProtectionGraph(scope, (a, b), records=(r,))
    assert g.record_of(a) is r
    assert graph.ProtectionGraph(scope, (a, b)).record_of(a) != r


def test_record_keeps_the_callers_member_order() -> None:
    scope, (a, b, c) = build("one", "two", "three")
    r = scope.record((c, a))
    assert graph.ProtectionGraph(scope, (a, b, c), records=(r,)).record_of(a).members == (c, a)


def test_undeclared_targets_each_get_their_own_stable_singleton() -> None:
    scope, (a, b, c) = build("one", "two", "three")
    declared = scope.record((a,))
    g = graph.ProtectionGraph(scope, (a, b, c), records=(declared,))
    rb, rc = g.record_of(b), g.record_of(c)
    assert rb.members == (b,) and rc.members == (c,)
    assert rb != rc and rb is not rc
    assert g.record_of(b) == rb and hash(g.record_of(b)) == hash(rb)
    copy = dataclasses.replace(g)
    assert copy.record_of(b) == rb and copy == g
    assert rb != declared and rc != declared
    # Without declared records, each target's singleton is the same Record whatever the graph.
    assert graph.ProtectionGraph(scope, (b, c)).record_of(b) == rb


def test_all_records_lists_each_record_once_by_its_earliest_target() -> None:
    scope, ts = build("t0", "t1", "t2", "t3", "t4")
    late, early = scope.record((ts[4], ts[1])), scope.record((ts[3], ts[2]))
    g = graph.ProtectionGraph(scope, ts, records=(late, early))
    assert g.all_records == (g.record_of(ts[0]), late, early)
    assert g.all_records == graph.ProtectionGraph(scope, ts, records=(early, late)).all_records
    covered = [d for r in g.all_records for d in r.members]
    assert sorted(covered, key=ts.index) == list(ts) and len(covered) == len(ts)
    plain = graph.ProtectionGraph(scope, ts)
    assert [r.members for r in plain.all_records] == [(d,) for d in ts]


def test_target_scoped_record_seed_is_the_targets_record() -> None:
    scope, (a, b, c) = build("one", "two", "three")
    g = graph.ProtectionGraph(scope, (a, b, c), records=(scope.record((a, c)),))
    seen: dict[graph.Datum, graph.Record] = {}

    def look(text: str, target: graph.Datum, record: graph.Record) -> str:
        seen[target] = record
        return text

    run(g, graph.Operation("look", look, {"text": "text", "target": "datum", "record": "record"}))
    assert seen == {d: g.record_of(d) for d in (a, b, c)}


def test_phase_scoped_record_seed_groups_declared_members() -> None:
    scope, (a, b, c, d) = build("one", "two", "three", "four")
    g = graph.ProtectionGraph(scope, (a, b, c, d), records=(scope.record((d, b)),))
    groups: list[list[int]] = []

    def fold(text: tuple[str, ...], record: tuple[graph.Record, ...]) -> list[str]:
        by: dict[graph.Record, list[int]] = {}
        for i, r in enumerate(record):
            by.setdefault(r, []).append(i)
        groups.extend(by.values())
        return list(text)

    run(g, graph.Operation.phase_scoped("fold", fold, {"text": "text", "record": "record"}))
    assert groups == [[0], [1, 3], [2]]


def test_record_seed_is_about_no_candidate() -> None:
    scope, (t,) = build("one")
    verifier = graph.Operation.verifier(
        "v", lambda record: graph.Assessment(True), {"record": "record"}, checks=("c",), subject=("record",)
    )
    g = graph.ProtectionGraph(scope, (t,))
    exc = rejection(lambda: graph.prepare(g, graph.Workflow("w", (verifier,), requires=("c",))))
    assert exc.code == "verifier_subject_invalid"


class _Subclassed(graph.Operation):
    pass


def test_output_named_record_is_rejected_at_construction() -> None:
    outputs = {"text": "text", "record": "record"}
    builds: list[Callable[[], Any]] = [
        lambda: graph.Operation("forge", lambda text: text, {"text": "text"}, outputs),
        lambda: graph.Operation.phase_scoped("forge", lambda text: list(text), {"text": "text"}, outputs),
        lambda: dataclasses.replace(graph.Operation.verifier("forge", lambda text: None), outputs={"record": "record"}),
        lambda: _Subclassed("forge", lambda text: text, {"text": "text"}, outputs),
    ]
    for fn in builds:
        exc = rejection(fn)
        assert (exc.code, exc.detail) == ("seed_output_reserved", "forge: record")


def test_reserved_seed_outputs_are_seeds() -> None:
    assert "record" in _RESERVED_SEED_OUTPUTS
    assert set(_RESERVED_SEED_OUTPUTS) <= set(SEEDS)


def test_graph_without_records_keeps_its_fields_and_positional_order() -> None:
    scope, (a, b) = build("one", "two")
    g = graph.ProtectionGraph(scope, (a, b), (), (), (), ())
    assert g.records == ()
    assert [f.name for f in dataclasses.fields(g)] == [
        "scope",
        "targets",
        "mentions",
        "context",
        "dependencies",
        "groups",
        "records",
    ]
    with pytest.raises(TypeError):
        graph.ProtectionGraph(scope, (a, b), (), (), (), (), ())  # ty: ignore[too-many-positional-arguments]
