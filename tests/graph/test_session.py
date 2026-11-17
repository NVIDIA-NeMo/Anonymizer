# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Session keeps assignments across ``protect`` calls; without one, ``protect`` behaves as before."""

from __future__ import annotations

import dataclasses
import json
import pickle
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest

import anonymizer.graph as graph
from anonymizer.engine.graph_core.session import SessionChanges

ORIGINAL = "Alice"
SLOT = (graph.SubjectKey("subject-alice"), "first_name", "alice")
DONE = graph.register_kind("test-session-done", lambda v: v is True)


def changes(*counts: int) -> SessionChanges:
    return SessionChanges(*counts)


def fold(calls: list[str] | None = None, prefix: str = "Name") -> graph.Operation:
    """A toy phase-scoped fold: mints ``{prefix}{n}`` from a per-call counter unless the Session view has one."""

    def run(text: tuple[str, ...], session: tuple[graph.SessionView, ...]) -> list[dict[str, Any]]:
        if calls is not None:
            calls.append("fold")
        minted = iter(range(1, 1_000))
        out = []
        for t, view in zip(text, session):
            if ORIGINAL not in t:
                out.append({"text": t, "decisions": ()})
                continue
            value = view.lookup(SLOT) or f"{prefix}{next(minted)}"
            out.append({"text": t.replace(ORIGINAL, value), "decisions": (graph.Decision(SLOT, value),)})
        return out

    return graph.Operation.phase_scoped(
        "fold", run, {"text": "text", "session": "session-view"}, {"text": "text", "decisions": "decisions"}
    )


def emit(per_target: Callable[[int], tuple[graph.Decision, ...]], name: str = "emit") -> graph.Operation:
    """A per-target operation emitting ``per_target(index)`` decisions, the index being the target's text."""
    return graph.Operation(
        name,
        lambda text: {"text": text, "decisions": per_target(int(text))},
        {"text": "text"},
        {"text": "text", "decisions": "decisions"},
    )


def check(name: str = "v", satisfied: Callable[[str], bool] = lambda t: True, foreign: bool = False) -> Any:
    def assess(text: str, target: graph.Datum) -> graph.Assessment:
        return graph.Assessment(satisfied(text), None if foreign else target)

    return graph.Operation.verifier(name, assess, {"text": "text", "target": "datum"}, checks=("c",), subject=("text",))


def build(*texts: str) -> tuple[graph.ProtectionGraph, tuple[graph.Datum, ...]]:
    scope = graph.GraphScope()
    targets = tuple(scope.datum(t) for t in texts)
    return graph.ProtectionGraph(scope, targets), targets


def prepared(*steps: Any, texts: tuple[str, ...] = ("Alice wrote this.",), **kw: Any) -> Any:
    g, _ = build(*texts)
    requires = ("c",) if any(s.checks for s in steps) else ()
    return graph.prepare(g, graph.Workflow("w", steps, requires=requires), **kw)


def rejection(fn: Callable[[], Any]) -> str:
    with pytest.raises(graph.GraphRejection) as info:
        fn()
    return info.value.code


def test_session_mode_table() -> None:
    off, stable = graph.Session(), graph.Session(stable_replacements=True)
    assert prepared(fold(), session=off).session_mode == graph.SessionMode()
    assert prepared(fold(), session=stable).session_mode == graph.SessionMode(stable_replacements=True)
    assert rejection(lambda: graph.Session(identity_matching=True)) == "session_mode_invalid"
    assert rejection(lambda: graph.Session(stable_replacements=1)) == "session_mode_invalid"  # ty: ignore[invalid-argument-type]
    on_on = graph.Session(stable_replacements=True, identity_matching=True)
    assert rejection(lambda: prepared(fold(), session=on_on)) == "session_mode_unsupported"
    assert rejection(lambda: prepared(fold(), session="stable")) == "not_a_session"


def test_bare_session_is_off_off() -> None:
    assert graph.Session().mode == graph.SessionMode()


def test_stable_session_reuses_an_assignment_across_protect_calls() -> None:
    session = graph.Session(stable_replacements=True)
    p = prepared(fold(), session=session)
    first = graph.protect(p, session=session)
    assert (first.session_changes, session.revision) == (changes(1, 0, 0, 0), 1)
    second = graph.protect(prepared(fold(), session=session), session=session)
    assert (second.session_changes, session.revision) == (changes(0, 1, 0, 0), 1)
    assert [t for _, t in first.content()] == [t for _, t in second.content()] == ["Name1 wrote this."]


def test_off_off_session_retains_nothing() -> None:
    session = graph.Session()
    p = prepared(fold(prefix="A"), session=session)
    assert graph.protect(p, session=session).session_changes == changes()
    p2 = prepared(fold(prefix="B"), session=session)
    result = graph.protect(p2, session=session)
    assert [t for _, t in result.content()] == ["B1 wrote this."]
    assert (result.session_changes, session.revision) == (changes(), 0)


def test_no_session_equals_an_explicit_off_off_session() -> None:
    seen: list[str | None] = []

    def read(text: str, session: graph.SessionView) -> str:
        seen.append(session.lookup(SLOT))
        return text.replace(ORIGINAL, "[NAME]")

    reader = graph.Operation("read", read, {"text": "text", "session": "session-view"})
    p = prepared(reader, check(satisfied=lambda t: ORIGINAL not in t), texts=("Alice", "Bob"))
    bare, explicit = graph.protect(p), graph.protect(p, session=graph.Session())
    assert bare.content() == explicit.content()
    assert bare.outcomes == explicit.outcomes
    assert bare.ledger() == explicit.ledger()
    assert bare.session_changes == explicit.session_changes == changes()
    assert seen == [None] * 4


def test_stable_session_without_decisions_port_is_rejected_before_callbacks() -> None:
    calls: list[str] = []
    op = graph.Operation("t", lambda text: calls.append("t") or text)
    session = graph.Session(stable_replacements=True)
    assert rejection(lambda: prepared(op, session=session)) == "session_without_resolution"
    assert calls == []


def test_mode_mismatch_is_rejected_before_callbacks() -> None:
    calls: list[str] = []
    stable = graph.Session(stable_replacements=True)
    p_stable = prepared(fold(calls), session=stable)
    assert rejection(lambda: graph.protect(p_stable)) == "session_mismatch"
    assert rejection(lambda: graph.protect(prepared(fold(calls)), session=stable)) == "session_mismatch"
    assert rejection(lambda: graph.protect(p_stable, session=object())) == "not_a_session"  # ty: ignore[invalid-argument-type]
    assert calls == []


def test_prepared_value_holds_no_live_session() -> None:
    session = graph.Session(stable_replacements=True)
    p = prepared(fold(), session=session)
    assert not any(isinstance(getattr(p, f.name), graph.Session) for f in dataclasses.fields(p))
    with pytest.raises(TypeError):
        pickle.dumps(p)


def test_second_protect_and_snapshot_are_busy_during_execution() -> None:
    session = graph.Session(stable_replacements=True)
    other = prepared(fold(), session=session)
    codes: list[str] = []

    def nested(text: str) -> str:
        for attempt in (lambda: graph.protect(other, session=session), session.snapshot):
            try:
                attempt()
            except graph.GraphRejection as exc:
                codes.append(exc.code)
        return text

    outer = prepared(graph.Operation("nested", nested), fold(), session=session)
    assert graph.protect(outer, session=session).outcomes[0].terminal == "success"
    assert codes == ["session_busy", "session_busy"]
    assert graph.protect(other, session=session).session_changes == changes(0, 1, 0, 0)
    assert session.snapshot().revision == 1


_CHILD = """
import sys
import anonymizer.graph as g
s = g.Session.restore(g.SessionSnapshot.from_bytes(sys.stdin.buffer.read()))
v = s._view().lookup((g.SubjectKey("subject-alice"), "first_name", "alice"))
print(s.mode.stable_replacements, s.revision, v)
"""


def test_snapshot_round_trips_through_bytes_in_another_process() -> None:
    session = graph.Session(stable_replacements=True)
    graph.protect(prepared(fold(), session=session), session=session)
    data = session.snapshot().to_bytes()
    out = subprocess.run([sys.executable, "-I", "-c", _CHILD], input=data, capture_output=True, check=True).stdout
    assert out.decode().split() == ["True", "1", "Name1"]
    restored = graph.Session.restore(graph.SessionSnapshot.from_bytes(data))
    graph.protect(prepared(emit(lambda i: (graph.Decision(SLOT, "X", True),)), texts=("0",), session=restored),
                  session=restored)  # fmt: skip
    assert (restored.revision, session.revision) == (2, 1)
    assert graph.SessionSnapshot.from_bytes(data) == session.snapshot()


def test_incompatible_snapshot_is_rejected() -> None:
    session = graph.Session(stable_replacements=True)
    graph.protect(prepared(fold(), session=session), session=session)
    good = json.loads(session.snapshot().to_bytes())
    (entry,) = good["slots"]
    variants = [
        {**good, "version": 2},
        {**good, "format": "other"},
        {**good, "mode": {**good["mode"], "normalization": "unknown-v9"}},
        {**good, "mode": {**good["mode"], "stable_replacements": False, "identity_matching": True}},
        {**good, "mode": {**good["mode"], "normalization": []}},
        {**good, "slots": [{"subject": ""}]},
        {**good, "slots": [{**entry, "revision": 2}]},
    ]
    for doc in variants:
        assert rejection(lambda: graph.SessionSnapshot.from_bytes(json.dumps(doc).encode())) == "snapshot_incompatible"
    for garbage in (b"\xff\x00", b"not json", b"[]", "text", b"[" * 200_000):
        assert rejection(lambda: graph.SessionSnapshot.from_bytes(garbage)) == "snapshot_incompatible"  # ty: ignore[invalid-argument-type]
    assert rejection(lambda: graph.Session.restore(b"{}")) == "snapshot_incompatible"  # ty: ignore[invalid-argument-type]


def test_reprs_and_errors_show_no_values() -> None:
    session = graph.Session(stable_replacements=True)
    graph.protect(prepared(fold(prefix="Zebulon"), session=session), session=session)
    view = session._view()
    rendered = [session, session.snapshot(), view, graph.Decision(SLOT, "Zebulon1"), SLOT[0]]
    for value in rendered:
        for text in (repr(value), str(value)):
            assert "Zebulon" not in text and "alice" not in text.lower() and "subject-alice" not in text
    for value in (session, view):
        with pytest.raises(TypeError):
            pickle.dumps(value)
    for bad in (lambda: graph.SubjectKey(""), lambda: view.lookup(("Alice", "x", "y")),  # ty: ignore[invalid-argument-type]
                lambda: graph.Decision(SLOT, 5)):  # ty: ignore[invalid-argument-type]  # fmt: skip
        with pytest.raises(graph.GraphRejection) as info:
            bad()
        assert "Alice" not in str(info.value) and "5" not in str(info.value)


def test_view_membership_questions_survive_a_snapshot() -> None:
    """``assigned_elsewhere`` and ``is_original`` answer from normalized values, the same after a restore."""
    other = (graph.SubjectKey("subject-paris"), "city", "paris")
    session = graph.Session(stable_replacements=True)
    decisions = (graph.Decision(SLOT, "Zebulon1"), graph.Decision(other, "Lyon"))
    graph.protect(prepared(emit(lambda i: decisions), texts=("0",), session=session), session=session)
    restored = graph.Session.restore(graph.SessionSnapshot.from_bytes(session.snapshot().to_bytes()))
    questions = [
        ("assigned_elsewhere", "ZEBULON1", SLOT),
        ("assigned_elsewhere", " zebulon1 ", other),
        ("assigned_elsewhere", "lyon", SLOT),
        ("assigned_elsewhere", "Lyon", other),
        ("assigned_elsewhere", "Nobody", SLOT),
        ("is_original", "ALICE", None),
        ("is_original", "Paris", None),
        ("is_original", "Zebulon1", None),
    ]

    def answers(view: graph.SessionView) -> list[bool]:
        return [getattr(view, name)(*(a for a in (value, slot) if a is not None)) for name, value, slot in questions]

    expected = [False, True, True, False, False, True, True, False]
    assert answers(session._view()) == answers(restored._view()) == expected
    view = session._view()
    with pytest.raises(TypeError):
        iter(view)  # ty: ignore[no-matching-overload]
    with pytest.raises(graph.GraphRejection):
        view.assigned_elsewhere("Lyon", ("Alice", "x", "y"))  # ty: ignore[invalid-argument-type]
    assert "lyon" not in repr(view).lower() and "paris" not in repr(view).lower()


def test_view_membership_questions_check_their_input_and_normalize_slot_values() -> None:
    unnormalized = (graph.SubjectKey("subject-x"), "city", "  New   YORK ")
    session = graph.Session(stable_replacements=True)
    graph.protect(prepared(emit(lambda i: (graph.Decision(unnormalized, "Lyon"),)), texts=("0",), session=session),
                  session=session)  # fmt: skip
    view = session._view()
    assert view.is_original("new york")
    for bad in (lambda: view.is_original(5), lambda: view.assigned_elsewhere(5, SLOT)):  # ty: ignore[invalid-argument-type]
        assert rejection(bad) == "invalid_value"


def test_reserved_subject_keys_cannot_be_constructed_but_restore_from_a_snapshot() -> None:
    for name in ("by-value", "substitute-fallback"):
        assert rejection(lambda: graph.SubjectKey(name)) == "reserved_subject_key"
    session = graph.Session(stable_replacements=True)
    graph.protect(prepared(fold(), session=session), session=session)
    doc = json.loads(session.snapshot().to_bytes())
    doc["slots"] = [{**doc["slots"][0], "subject": "by-value"}, {**doc["slots"][0], "subject": "substitute-fallback"}]
    data = json.dumps(doc).encode()
    restored = graph.Session.restore(graph.SessionSnapshot.from_bytes(data))
    assert json.loads(restored.snapshot().to_bytes()) == json.loads(data)


def test_acceptance_survives_withholding_and_skips_cancel_and_inconsistency() -> None:
    withheld = graph.Session(stable_replacements=True)
    p = prepared(fold(), check(satisfied=lambda t: False), session=withheld)
    result = graph.protect(p, session=withheld)
    assert not result.outcomes[0].protected
    assert (result.session_changes, withheld.revision) == (changes(1, 0, 0, 0), 1)

    token = graph.CancelToken()
    cancelled = graph.Session(stable_replacements=True)
    stop = graph.Operation("stop", lambda text: token.cancel() or text)
    p = prepared(fold(), stop, texts=("Alice one", "Alice two"), session=cancelled)
    result = graph.protect(p, session=cancelled, cancel=token)
    assert result.outcomes[1].terminal == "cancelled"
    assert (result.session_changes, cancelled.revision) == (changes(), 0)

    foreign = graph.Session(stable_replacements=True)
    result = graph.protect(prepared(fold(), check(foreign=True), session=foreign), session=foreign)
    assert not result.consistent
    assert (result.session_changes, foreign.revision) == (changes(), 0)


def test_cancel_after_every_target_is_released_accepts_nothing() -> None:
    token = graph.CancelToken()
    session = graph.Session(stable_replacements=True)
    stop = graph.Operation.phase_scoped("stop", lambda text: token.cancel() or [True] * len(text), None, {"d": DONE})
    p = prepared(fold(), check(), stop, texts=("Alice one", "Alice two"), session=session)
    result = graph.protect(p, session=session, cancel=token)
    assert result.state.cancelled
    assert all(o.protected for o in result.outcomes)
    assert (result.session_changes, session.revision) == (changes(), 0)


def test_conflicting_decisions_follow_target_order() -> None:
    session = graph.Session(stable_replacements=True)
    first = emit(lambda i: (graph.Decision(SLOT, f"V{i}"),))
    graph.protect(prepared(first, texts=("0", "1"), session=session), session=session)
    revise = emit(lambda i: (graph.Decision(SLOT, "R", supersedes=True),), name="revise")
    result = graph.protect(prepared(revise, texts=("0",), session=session), session=session)
    assert (result.session_changes, session.revision) == (changes(0, 0, 1, 0), 2)
    (entry,) = json.loads(session.snapshot().to_bytes())["slots"]
    assert (entry["assignment"], entry["revision"], entry["by"]) == ("R", 2, "revise")
    assert entry["superseded"] == [{"assignment": "V0", "revision": 1, "by": "emit"}]


def test_an_ignored_decision_is_visible_only_as_a_count() -> None:
    """A differing decision without ``supersedes`` keeps the retained value; the Session reports only a count.

    Neither the result's ``session_changes`` nor the Session or its snapshot names the slot, operation or target
    of an ignored decision; the emitted tuple stays readable only as the operation's own port output.
    """
    session = graph.Session(stable_replacements=True)
    p = prepared(emit(lambda i: (graph.Decision(SLOT, f"V{i}"),)), texts=("0", "1"), session=session)
    result = graph.protect(p, session=session)
    assert result.session_changes == changes(1, 0, 0, 1)
    assert [f.name for f in dataclasses.fields(result.session_changes)] == ["accepted", "reused", "revised", "ignored"]
    assert b"V1" not in session.snapshot().to_bytes() and b"V0" in session.snapshot().to_bytes()
    assert "V1" not in repr(result) + repr(session) + repr(session.snapshot())
    assert result.value(result.targets[1], "decisions") == (graph.Decision(SLOT, "V1"),)


def test_session_seed_is_about_no_candidate() -> None:
    v = graph.Operation.verifier(
        "v", lambda session, target: graph.Assessment(True, target), {"session": "session-view", "target": "datum"},
        checks=("c",), subject=("session",),
    )  # fmt: skip
    g, _ = build("Alice")
    with pytest.raises(graph.GraphRejection) as info:
        graph.prepare(g, graph.Workflow("w", (v,), requires=("c",)))
    assert (info.value.code, info.value.detail) == ("verifier_subject_invalid", "v.session: about no candidate")


def test_a_falsy_session_subclass_is_still_used() -> None:
    class Empty(graph.Session):
        def __len__(self) -> int:
            return 0

    session = Empty(stable_replacements=True)
    assert not session
    result = graph.protect(prepared(fold(), session=session), session=session)
    assert (result.session_changes, session.revision) == (changes(1, 0, 0, 0), 1)


def test_view_is_read_only() -> None:
    views: list[graph.SessionView] = []

    def keep(text: str, session: graph.SessionView) -> str:
        views.append(session)
        return text

    graph.protect(prepared(graph.Operation("keep", keep, {"text": "text", "session": "session-view"})))
    (view,) = views
    with pytest.raises(AttributeError):
        view.mode = graph.SessionMode(stable_replacements=True)  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        view.extra = 1  # ty: ignore[unresolved-attribute]
    with pytest.raises(TypeError):
        view._assignments[SLOT] = "X"  # ty: ignore[invalid-assignment]


def test_lock_is_released_when_an_exception_escapes_protect() -> None:
    class Escape(BaseException):
        pass

    session = graph.Session(stable_replacements=True)

    def boom(text: str) -> str:
        raise Escape

    with pytest.raises(Escape):
        graph.protect(prepared(graph.Operation("boom", boom), fold(), session=session), session=session)
    assert session.snapshot().revision == 0


def test_superseded_history_round_trips_through_bytes() -> None:
    session = graph.Session(stable_replacements=True)
    graph.protect(prepared(emit(lambda i: (graph.Decision(SLOT, "A"),)), texts=("0",), session=session),
                  session=session)  # fmt: skip
    revise = emit(lambda i: (graph.Decision(SLOT, "B", supersedes=True),), name="revise")
    graph.protect(prepared(revise, texts=("0",), session=session), session=session)
    data = session.snapshot().to_bytes()
    restored = graph.SessionSnapshot.from_bytes(data)
    assert restored == session.snapshot() and restored.to_bytes() == data
    (entry,) = json.loads(data)["slots"]
    assert entry["superseded"] == [{"assignment": "A", "revision": 1, "by": "emit"}]


def test_decisions_within_one_target_follow_operation_order() -> None:
    session = graph.Session(stable_replacements=True)
    first = emit(lambda i: (graph.Decision(SLOT, "First"),), name="first")
    second = emit(lambda i: (graph.Decision(SLOT, "Second"),), name="second")
    result = graph.protect(prepared(first, second, texts=("0",), session=session), session=session)
    assert result.session_changes == changes(1, 0, 0, 1)
    (entry,) = json.loads(session.snapshot().to_bytes())["slots"]
    assert (entry["assignment"], entry["by"]) == ("First", "first")


def test_a_supersede_in_the_same_call_takes_a_later_revision() -> None:
    session = graph.Session(stable_replacements=True)
    step = emit(lambda i: (graph.Decision(SLOT, f"V{i}", supersedes=i == 1),))
    result = graph.protect(prepared(step, texts=("0", "1"), session=session), session=session)
    assert (result.session_changes, session.revision) == (changes(1, 0, 1, 0), 2)
    (entry,) = json.loads(session.snapshot().to_bytes())["slots"]
    assert (entry["assignment"], entry["revision"]) == ("V1", 2)
    assert entry["superseded"] == [{"assignment": "V0", "revision": 1, "by": "emit"}]
