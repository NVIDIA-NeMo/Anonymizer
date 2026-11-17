# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity declarations: a span of a target is one subject's, and later mentions inside it resolve to that subject."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import anonymizer.graph as graph
from anonymizer.engine.graph_core.authoring import _RESERVED_SEED_OUTPUTS, subject_of
from anonymizer.engine.graph_core.execution import SEEDS

TEXT = "Alice and Bob and Carol"
ALICE, BOB = graph.SubjectKey("subject-alice"), graph.SubjectKey("subject-bob")
DOC = Path(__file__).resolve().parents[2] / "docs" / "concepts" / "graph-sdk.md"


def build(*texts: str) -> tuple[graph.GraphScope, tuple[graph.Datum, ...]]:
    scope = graph.GraphScope()
    return scope, tuple(scope.datum(t) for t in texts)


def rejection(fn: Callable[[], Any]) -> str:
    with pytest.raises(graph.GraphRejection) as info:
        fn()
    return info.value.code


def run(g: graph.ProtectionGraph, *steps: graph.Operation) -> graph.ProtectionResult:
    return graph.protect(graph.prepare(g, graph.Workflow("w", steps)))


def _invalid_element(scope: graph.GraphScope, t: graph.Datum, ctx: graph.Datum) -> Any:
    return ("not an identity",)


def _plain_subject(scope: graph.GraphScope, t: graph.Datum, ctx: graph.Datum) -> Any:
    return (graph.Identity(t, 0, 5, "subject-alice"),)  # ty: ignore[invalid-argument-type]


def _bare_identity(scope: graph.GraphScope, t: graph.Datum, ctx: graph.Datum) -> Any:
    return graph.Identity(t, 0, 5, ALICE)


def _foreign(scope: graph.GraphScope, t: graph.Datum, ctx: graph.Datum) -> Any:
    return (graph.Identity(graph.GraphScope().datum(TEXT), 0, 5, ALICE),)


def _context_target(scope: graph.GraphScope, t: graph.Datum, ctx: graph.Datum) -> Any:
    return (graph.Identity(ctx, 0, 5, ALICE),)


def _span(start: Any, end: Any) -> Callable[..., Any]:
    return lambda scope, t, ctx: (graph.Identity(t, start, end, ALICE),)


def _overlap(scope: graph.GraphScope, t: graph.Datum, ctx: graph.Datum) -> Any:
    return (graph.Identity(t, 0, 12, ALICE), graph.Identity(t, 1, 2, ALICE), graph.Identity(t, 10, 13, BOB))


@pytest.mark.parametrize(
    ("declare", "code", "detail"),
    [
        (_invalid_element, "invalid_identity", ""),
        (_plain_subject, "invalid_identity", ""),
        (_bare_identity, "invalid_identity", ""),
        (_foreign, "foreign_datum", "identity"),
        (_context_target, "unknown_reference", "identity"),
        (_span(3, 3), "identity_out_of_range", ""),
        (_span(0, len(TEXT) + 1), "identity_out_of_range", ""),
        (_span(-1, 3), "identity_out_of_range", ""),
        (_span(False, 3), "identity_out_of_range", ""),
        (_span(0, True), "identity_out_of_range", ""),
        (_overlap, "identity_conflict", ""),
    ],
)
def test_identity_construction_rejections(declare: Callable[..., Any], code: str, detail: str) -> None:
    scope, (t, ctx) = build(TEXT, "context")
    context = (graph.ContextScope(t, (ctx,)),)
    with pytest.raises(graph.GraphRejection) as info:
        graph.ProtectionGraph(scope, (t,), context=context, identities=declare(scope, t, ctx))
    assert (info.value.code, info.value.detail) == (code, detail)
    assert "subject-alice" not in str(info.value) and "Alice" not in str(info.value)


def test_same_key_overlap_and_duplicates_are_accepted() -> None:
    scope, (t,) = build(TEXT)
    ids = (graph.Identity(t, 0, 12, ALICE), graph.Identity(t, 5, 15, ALICE), graph.Identity(t, 0, 12, ALICE))
    adjacent = (graph.Identity(t, 0, 10, ALICE), graph.Identity(t, 10, 13, BOB))
    assert graph.ProtectionGraph(scope, (t,), identities=ids).identities == ids
    assert graph.ProtectionGraph(scope, (t,), identities=adjacent).identities == adjacent


def test_whole_target_identity_is_a_full_span() -> None:
    scope, (t, other) = build(TEXT, TEXT)
    whole = graph.Identity(t, 0, len(t.text), ALICE)
    graph.ProtectionGraph(scope, (t, other), identities=(whole,))
    n = len(t.text)
    assert all(whole.covers(graph.Mention(t, s, e, "x")) for s in range(n) for e in range(s + 1, n + 1))
    assert not whole.covers(graph.Mention(other, 0, 5, "x"))


def test_identities_seed_is_per_target_and_sorted() -> None:
    scope, (a, b) = build(TEXT, "Dora")
    late, early, wide = graph.Identity(a, 10, 13, BOB), graph.Identity(a, 0, 5, ALICE), graph.Identity(a, 0, 9, ALICE)
    g = graph.ProtectionGraph(scope, (a, b), identities=(late, wide, early))
    seen: dict[str, Any] = {}

    def per_target(text: str, identities: tuple[graph.Identity, ...]) -> str:
        seen[text] = identities
        return text

    def batch(text: tuple[str, ...], identities: tuple[tuple[graph.Identity, ...], ...]) -> list[str]:
        seen["phase"] = identities
        return list(text)

    run(
        g,
        graph.Operation("one", per_target, {"text": "text", "identities": "identities"}),
        graph.Operation.phase_scoped("all", batch, {"text": "text", "identities": "identities"}),
    )
    assert seen[TEXT] == (early, wide, late)
    assert seen["Dora"] == ()
    assert seen["phase"] == ((early, wide, late), ())


def test_produced_mention_resolves_inside_span_only() -> None:
    scope, (t,) = build(TEXT)
    g = graph.ProtectionGraph(scope, (t,), identities=(graph.Identity(t, 0, 12, ALICE),))
    resolved: list[graph.SubjectKey | None] = []

    def detect(text: str, target: graph.Datum) -> tuple[graph.Mention, ...]:
        # inside, straddling the span's end, outside
        return (
            graph.Mention(target, 0, 5, "a"),
            graph.Mention(target, 10, 13, "b"),
            graph.Mention(target, 18, 23, "c"),
        )

    def fold(text: str, mentions: tuple[graph.Mention, ...], identities: tuple[graph.Identity, ...]) -> str:
        resolved.extend(subject_of(m, identities) for m in mentions)
        return text

    run(
        g,
        graph.Operation("detect", detect, {"text": "text", "target": "datum"}, {"mentions": "mentions"}),
        graph.Operation("fold", fold, {"text": "text", "mentions": "mentions", "identities": "identities"}),
    )
    assert resolved == [ALICE, None, None]


def test_mention_has_no_new_field() -> None:
    assert [f.name for f in dataclasses.fields(graph.Mention)] == ["datum", "start", "end", "label", "text"]


def test_identity_repr_hides_subject_value() -> None:
    scope, (t,) = build(TEXT)
    identity = graph.Identity(t, 0, 5, ALICE)
    g = graph.ProtectionGraph(scope, (t,), identities=(identity,))
    for shown in (repr(identity), repr(g)):
        assert "subject-alice" not in shown and "Alice" not in shown


def test_graph_without_identities_is_unchanged() -> None:
    scope, (t,) = build(TEXT)
    assert graph.ProtectionGraph(scope, (t,)).identities == ()
    positional = graph.ProtectionGraph(scope, (t,), (), (), (), ())
    assert positional == graph.ProtectionGraph(scope, (t,))
    assert [f.name for f in dataclasses.fields(graph.ProtectionGraph)] == [
        "scope",
        "targets",
        "mentions",
        "context",
        "dependencies",
        "groups",
        "identities",
        "records",
    ]


def test_identities_seed_is_about_no_candidate() -> None:
    scope, (t,) = build(TEXT)
    verifier = graph.Operation.verifier(
        "v",
        lambda identities: graph.Assessment(True),
        {"identities": "identities"},
        checks=("c",),
        subject=("identities",),
    )
    g = graph.ProtectionGraph(scope, (t,))
    assert rejection(lambda: graph.prepare(g, graph.Workflow("w", (verifier,), requires=("c",)))) == (
        "verifier_subject_invalid"
    )


class _Subclassed(graph.Operation):
    pass


def _forging_builds(port: str, kind: str) -> list[Callable[[], Any]]:
    """The S2 reviewer's forging layout: an operation whose output replaces a seed for every later operation."""
    outputs = {"text": "text", port: kind}
    return [
        lambda: graph.Operation("forge", lambda text: text, {"text": "text"}, outputs),
        lambda: graph.Operation.phase_scoped("forge", lambda text: list(text), {"text": "text"}, outputs),
        lambda: dataclasses.replace(graph.Operation.verifier("forge", lambda text: None), outputs={port: kind}),
        lambda: _Subclassed("forge", lambda text: text, {"text": "text"}, outputs),
    ]


@pytest.mark.parametrize(("port", "kind"), [("identities", "identities"), ("session", "session-view")])
def test_seed_named_outputs_are_rejected_at_construction(port: str, kind: str) -> None:
    for build in _forging_builds(port, kind):
        with pytest.raises(graph.GraphRejection) as info:
            build()
        assert (info.value.code, info.value.detail) == ("seed_output_reserved", f"forge: {port}")


def test_identities_forging_layout_is_rejected_at_construction() -> None:
    assert rejection(_forging_builds("identities", "identities")[0]) == "seed_output_reserved"


def test_reserved_seed_outputs_are_seeds() -> None:
    assert set(_RESERVED_SEED_OUTPUTS) <= set(SEEDS)


def test_mention_across_two_same_key_declarations_resolves_to_none() -> None:
    # Pinned limit: each declaration is tested alone, so a merge of same-key spans changes this answer.
    scope, (t,) = build(TEXT)
    ids = (graph.Identity(t, 0, 4, ALICE), graph.Identity(t, 1, 5, ALICE))
    graph.ProtectionGraph(scope, (t,), identities=ids)
    assert subject_of(graph.Mention(t, 0, 5, "x"), ids) is None


def test_seed_output_reserved_is_documented() -> None:
    assert 'GraphRejection("seed_output_reserved")' in DOC.read_text()
