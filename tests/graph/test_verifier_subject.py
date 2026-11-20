# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A verifier covers a required check only when every port it declares as its subject derives from the final text."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

import anonymizer.graph as graph

NOTES = graph.register_kind("test-notes", lambda v: isinstance(v, str))
SOURCE = "Alice wrote this."
calls: list[str] = []


def op(name: str, fn: Callable[..., Any], inputs: dict[str, str], outputs: dict[str, str], **kw: Any) -> Any:
    def call(**ports: Any) -> Any:
        calls.append(name)
        return fn(**ports)

    return graph.Operation(name, call, inputs, outputs, **kw)


def notes_of(text: str) -> str:
    return "clean" if "Alice" not in text else "dirty"


draft = op("draft", lambda text: {"text": text.replace("Alice", "[NAME]"), "notes": "clean"}, {"text": "text"},
           {"text": "text", "notes": NOTES})  # fmt: skip
revise = op("revise", lambda text: text.replace("[NAME]", "Alice"), {"text": "text"}, {"text": "text"})
revise_notes = op("revise", lambda text: {"text": text, "notes": notes_of(text)}, {"text": "text"},
                  {"text": "text", "notes": NOTES})  # fmt: skip
score = op("score", notes_of, {"text": "text"}, {"notes": NOTES})
rescore = op("rescore", lambda text, notes: notes_of(text), {"text": "text", "notes": NOTES}, {"notes": NOTES})
make_map = op("make-map", lambda target: "Alice", {"target": "datum"}, {"rmap": NOTES})
apply_map = op("apply", lambda text, rmap: text.replace(rmap, "[NAME]"), {"text": "text", "rmap": NOTES},
               {"text": "text"})  # fmt: skip
constant = op("constant", lambda: "clean", {}, {"notes": NOTES})
odd_text = op("odd", lambda text: "clean", {"text": "text"}, {"text": NOTES})


def verifier(name: str, inputs: dict[str, str], subject: tuple[str, ...], check: str = "c") -> Any:
    """Satisfied when the ``notes`` input says clean, else when the ``text`` input holds no name."""

    def assess(target: graph.Datum, **ports: Any) -> graph.Assessment:
        calls.append(name)
        ok = ports["notes"] == "clean" if "notes" in ports else "Alice" not in str(ports.get("text", ""))
        return graph.Assessment(ok, target)

    return graph.Operation.verifier(name, assess, {**inputs, "target": "datum"}, checks=(check,), subject=subject)


T, N = {"text": "text"}, {"notes": NOTES}
REJECT = "required_check_not_final"
INVALID = "verifier_subject_invalid"
CASES: dict[str, Callable[[], tuple[tuple[Any, ...], tuple[str, ...], tuple[str, ...]]]] = {
    # case: (steps, requires, expected (code, detail) or ("admitted", qualification))
    "A": lambda: ((draft, revise, verifier("v", {**N, **T}, ("notes",))), ("c",), (REJECT, "c")),
    "B": lambda: ((draft, revise_notes, verifier("v", N, ("notes",))), ("c",), ("admitted", "met")),
    "b1": lambda: ((draft, score, verifier("v", N, ("notes",))), ("c",), ("admitted", "met")),
    "b2": lambda: ((draft, score, revise, verifier("v", N, ("notes",))), ("c",), (REJECT, "c")),
    "b3": lambda: ((draft, score, revise, verifier("v", {**N, **T}, ("notes",))), ("c",), (REJECT, "c")),
    "b4": lambda: ((draft, score, revise, rescore, verifier("v", N, ("notes",))), ("c",), ("admitted", "not_met")),
    "c1": lambda: (
        (make_map, apply_map, verifier("v", {**T, "rmap": NOTES, "source": "text"}, ("text",))),
        ("c",),
        ("admitted", "met"),
    ),  # fmt: skip
    "c2": lambda: ((make_map, apply_map, verifier("v", {**T, "rmap": NOTES}, ("text", "rmap"))), ("c",), (REJECT, "c")),
    "f1": lambda: ((draft, verifier("v", T, ())), ("c",), (INVALID, "v: undeclared")),
    "f2": lambda: ((draft, verifier("v", T, ("metrics",))), ("c",), (INVALID, "v.metrics: not an input")),
    "f3": lambda: (
        (draft, verifier("v", {"context": "context"}, ("context",))),
        ("c",),
        (INVALID, "v.context: about no candidate"),
    ),
    "f4": lambda: ((draft, constant, verifier("v", N, ("notes",))), ("c",), (INVALID, "v.notes: about no candidate")),
    "f5": lambda: ((draft, verifier("v", {"source": "text"}, ("source",))), ("c",), (REJECT, "c")),
    "f6": lambda: ((draft, verifier("v", T, ("text",)), revise), ("c",), (REJECT, "c")),
    "f7": lambda: ((op("draft", lambda text: text, T, T, subject=("text",)),), (), (INVALID, "draft: not a verifier")),
    "s1": lambda: (
        (draft, verifier("check-draft", T, ("text",), "d"), revise_notes, verifier("check-final", N, ("notes",))),
        ("c",),
        ("admitted", "met"),
    ),  # fmt: skip
    "s2": lambda: ((verifier("v", T, ("text",)),), ("c",), ("admitted", "not_met")),
    "g1": lambda: ((draft, verifier("v", T, ("text",)), odd_text), ("c",), ("final_text_not_text", "odd")),
    # Known hole: the graph checks the declaration, not what the callback reads. This verifier decides on the
    # draft's notes while declaring the final text as its subject, and the revised text with the name is released.
    "x1": lambda: ((draft, revise, verifier("v", {**N, **T}, ("text",))), ("c",), ("admitted", "met")),
}


@pytest.mark.parametrize("case", CASES)
def test_verifier_subject_decides_coverage(case: str) -> None:
    steps, requires, expected = CASES[case]()
    scope = graph.GraphScope()
    target = scope.datum(SOURCE)
    calls.clear()
    try:
        prepared = graph.prepare(graph.ProtectionGraph(scope, (target,)), graph.Workflow("w", steps, requires=requires))
    except graph.GraphRejection as exc:
        assert (exc.code, exc.detail) == expected
        assert calls == []
        return
    assert ("admitted", graph.protect(prepared).outcome(target).qualification) == expected


def test_verifier_without_checks_or_subject_is_valid_and_not_covering() -> None:
    """s3: the facade's evaluate judges declare neither ``checks`` nor ``subject``."""
    scope = graph.GraphScope()
    target = scope.datum(SOURCE)
    judge = graph.Operation.verifier("judge", lambda text, target: graph.Assessment(True, target),
                                     {"text": "text", "target": "datum"})  # fmt: skip
    prepared = graph.prepare(graph.ProtectionGraph(scope, (target,)), graph.Workflow("w", (judge,)))
    assert prepared.covering_verifiers == frozenset()
    assert graph.protect(prepared).outcome(target).qualification == "not_assessed"


def _graph() -> graph.ProtectionGraph:
    scope = graph.GraphScope()
    return graph.ProtectionGraph(scope, (scope.datum(SOURCE),))


def _assess(target: graph.Datum, text: str) -> graph.Assessment:
    return graph.Assessment(True, target)


@pytest.mark.parametrize(
    "build, field",
    [
        # The ignores pin the annotations: if they widen to accept a str, ty reports the ignore as unused.
        (lambda: graph.Operation.verifier("v", _assess, checks="c", subject=("text",)), "checks"),  # ty: ignore[invalid-argument-type]
        (lambda: graph.Operation.verifier("v", _assess, checks=("c",), subject="text"), "subject"),  # ty: ignore[invalid-argument-type]
        (lambda: graph.Operation("v", _assess, checks="c"), "checks"),  # ty: ignore[invalid-argument-type]
        (lambda: graph.Operation("v", _assess, subject="text"), "subject"),  # ty: ignore[invalid-argument-type]
        (lambda: graph.Workflow("w", (), requires="c"), "requires"),  # ty: ignore[invalid-argument-type]
        (lambda: graph.prepare(_graph(), graph.Workflow("w", ()), capabilities="c"), "capabilities"),  # ty: ignore[invalid-argument-type]
    ],
)
def test_bare_string_name_tuple_is_rejected_at_construction(build: Callable[[], Any], field: str) -> None:
    with pytest.raises(graph.GraphRejection) as info:
        build()
    detail = info.value.detail
    assert info.value.code == "names_not_tuple"
    assert f"{field} must be a tuple" in detail
    value = "'text'" if field == "subject" else "'c'"
    assert f"{field}=({value},)" in detail and f"got the string {value}" in detail


def test_list_of_names_is_still_accepted() -> None:
    v = graph.Operation.verifier("v", _assess, checks=["c"], subject=["text"])  # ty: ignore[invalid-argument-type]
    assert v.checks == ("c",) and v.subject == ("text",)
