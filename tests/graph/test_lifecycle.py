# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run lifecycle: nested synchronous frames, cancellation through a CancelToken, and caller-owned resources."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable
from typing import Any

import pytest

from anonymizer.engine.graph_core.authoring import (
    Assessment,
    ContextScope,
    ContextSource,
    GraphScope,
    Operation,
    ProtectionGraph,
    Workflow,
)
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.engine.graph_core.kernel import CancelToken, GraphRejection
from tests.graph.helpers import NAME, graph, run

VAR: contextvars.ContextVar[str] = contextvars.ContextVar("lifecycle", default="unset")


# --- nested synchronous frames ---


def _nested(
    inner_fn: Callable[[str], str] | None = None,
) -> tuple[list[tuple[str, str, int]], dict[str, Any], Any, Any, Any]:
    """Outer one-operation workflow (plus its check) whose callback runs an inner transform+verifier workflow."""
    calls: list[tuple[str, str, int]] = []
    seen: dict[str, Any] = {}
    inner_g, (ia,) = graph(f"{NAME} met Bob.")

    def inner_transform(text: str) -> str:
        calls.append(("transform", text, threading.get_ident()))
        seen["inner_var"] = VAR.get()
        VAR.set("set-inside-inner")
        return (inner_fn or (lambda t: t.replace(NAME, "[NAME]")))(text)

    def inner_verify(text: str, target: Any) -> Assessment:
        calls.append(("verifier", text, threading.get_ident()))
        seen["verifier_var"] = VAR.get()
        return Assessment(NAME not in text, target)

    verifier = Operation.verifier(
        "verifier", inner_verify, {"text": "text", "target": "datum"}, checks=("name",), subject=("text",)
    )
    inner_ops = (Operation("transform", inner_transform), verifier)
    inner = prepare(inner_g, Workflow("inner", inner_ops, requires=("name",)))

    def outer_op(text: str) -> str:
        r = protect(inner)
        seen["inner_result"] = r
        seen["outer_after_inner"] = VAR.get()
        return r.protected.get(ia, text)

    outer_g, (oa,) = graph("outer text")
    check = Operation.verifier("check", lambda text: Assessment(text != "outer text"))
    VAR.set("caller-value")
    result = protect(prepare(outer_g, Workflow("outer", (Operation("outer", outer_op), check))))
    seen["caller_after"] = VAR.get()
    return calls, seen, result, ia, oa


def test_nested_run_executes_on_the_calling_thread_with_its_own_frame() -> None:
    me = threading.get_ident()
    calls, seen, outer, ia, _ = _nested()
    box: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: box.update(tid=threading.get_ident(), out=_nested()))
    worker.start()
    worker.join()
    calls2 = box["out"][0]
    assert len(calls) == 2
    assert all(t == me for _, _, t in calls)
    assert len(calls2) == 2
    assert all(t == box["tid"] for _, _, t in calls2)

    inner = seen["inner_result"]
    assert [op for op, _, _ in calls] == ["transform", "verifier"]
    assert inner.outcome(ia).qualification == "met"
    assert calls[1][1] == "[NAME] met Bob."

    assert seen["inner_var"] == "caller-value"
    assert seen["verifier_var"] in ("caller-value", "set-inside-inner")  # same frame: either is conforming
    assert seen["outer_after_inner"] == "caller-value"
    assert seen["caller_after"] == "caller-value"

    assert len(inner.attempts) == 2
    assert len(outer.attempts) == 2
    assert {name for _, name, _ in inner.ledger()} == {"transform", "verifier"}
    assert {name for _, name, _ in outer.ledger()} == {"outer", "check"}


def test_nested_inner_failure_stays_inside_the_inner_result() -> None:
    def boom(text: str) -> str:
        raise RuntimeError("inner failure")

    _, seen, outer, ia, oa = _nested(boom)
    assert seen["inner_result"].outcome(ia).terminal == "failed"
    assert outer.outcome(oa).qualification != "met"


# --- cancellation ---


def _setup(cancel_in_first: bool) -> tuple[Any, CancelToken, list[tuple[str, str]], tuple]:
    g, targets = graph("Alice one", "Alice two")
    token = CancelToken()
    calls: list[tuple[str, str]] = []

    def first(text: str) -> str:
        calls.append(("first", text))
        if cancel_in_first:
            token.cancel()
        return text.upper()

    def second(text: str) -> str:
        calls.append(("second", text))
        return text

    p = prepare(g, Workflow("w", (Operation("first", first), Operation("second", second))))
    return p, token, calls, targets


def test_cancel_before_protect_dispatches_nothing() -> None:
    p, token, calls, (t1, t2) = _setup(False)
    token.cancel()
    r = protect(p, cancel=token)
    assert not calls
    assert [r.outcome(t).terminal for t in (t1, t2)] == ["cancelled", "cancelled"]
    assert not r.protected
    assert len(r.ledger()) == 4


def test_cancel_during_callback_stops_further_dispatch() -> None:
    p, token, calls, (t1, _) = _setup(True)
    r = protect(p, cancel=token)
    assert calls == [("first", "Alice one")]
    assert not any(a.operation == "second" for a in r.attempts)
    o1 = r.outcome(t1)
    assert o1.terminal not in ("lost", "failed")
    assert len([a for a in r.attempts if a.target == t1]) == 1
    assert not r.protected
    assert len(r.ledger()) == 4


def test_cancel_after_completion_changes_nothing() -> None:
    p, token, _, (t1, t2) = _setup(False)
    r = protect(p, cancel=token)
    before = ([r.outcome(t) for t in (t1, t2)], dict(r.protected), r.ledger())
    token.cancel()
    after = ([r.outcome(t) for t in (t1, t2)], dict(r.protected), r.ledger())
    assert before == after
    assert len(r.ledger()) == 4


def test_keyboard_interrupt_reaches_the_caller_unchanged() -> None:
    g, _ = graph("Alice one", "Alice two")
    interrupt = KeyboardInterrupt()
    calls: list[str] = []

    def raises(text: str) -> str:
        calls.append(text)
        raise interrupt

    with pytest.raises(KeyboardInterrupt) as info:
        protect(prepare(g, Workflow("w", (Operation("op", raises),))))
    assert info.value is interrupt
    assert len(calls) == 1


def test_cancel_in_callback_cancels_only_later_targets() -> None:
    g, (t1, t2, t3) = graph("Alice one", "Alice two", "Alice three")
    token = CancelToken()
    calls: list[str] = []

    def op(text: str) -> str:
        calls.append(text)
        if "two" in text:
            token.cancel()
        return text

    r = protect(prepare(g, Workflow("w", (Operation("op", op),))), cancel=token)
    ledger = r.ledger()
    assert len(calls) == 2
    assert [r.outcome(t).terminal for t in (t1, t2, t3)] == ["success", "success", "cancelled"]
    assert not any(a.target == t3 for a in r.attempts)
    assert len(ledger) == 3
    assert ledger[2] == (t3, "op", "cancelled")


# --- ownership ---


class Client:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("fail", [False, True])
def test_caller_owned_resource_is_never_closed(fail: bool) -> None:
    client = Client()
    scope = GraphScope()
    t = scope.datum("Alice")

    def op(text: str, context: tuple) -> str:
        if fail:
            raise RuntimeError("callback failure")
        return text if client else ""

    src = ContextSource("client", lambda: [repr(client)], max_items=1)
    run(
        ProtectionGraph(scope, (t,), context=(ContextScope(t, (src,)),)),
        Operation("op", op, {"text": "text", "context": "context"}),
    )
    assert client.closed is False


def test_uncertain_effect_is_invoked_once_and_not_met() -> None:
    effects: list[str] = []

    def uncertain(text: str) -> str:
        effects.append("sent")
        raise ConnectionResetError("after effect")

    g, (a,) = graph("Alice")
    r = run(g, Operation("send", uncertain))
    assert effects == ["sent"]
    assert r.outcome(a).qualification != "met"


def test_verifier_finding_keyed_to_another_target_does_not_qualify() -> None:
    g, (t1, t2) = graph("Alice one", "Bob two")
    swap = {t1: t2, t2: t2}  # the verifier for t1 returns its finding keyed to t2

    def verify(text: str, target: Any) -> Assessment:
        return Assessment(True, swap[target])

    r = run(
        g,
        Operation("transform", lambda text: text),
        Operation.verifier("verify", verify, {"text": "text", "target": "datum"}),
    )
    assert "met" not in [r.outcome(t).qualification for t in (t1, t2)]
    assert not r.protected


def test_failed_first_step_never_dispatches_later_steps() -> None:
    g, _ = graph("a", "b", "c")

    def fail(text: str) -> str:
        raise ValueError("no")

    r = run(g, Operation("first", fail), Operation("never", lambda text: text))
    by_op = [a.operation for a in r.attempts]
    assert by_op.count("first") == 3
    assert by_op.count("never") == 0


def test_duplicate_context_source_name_is_rejected() -> None:
    """One target never receives another target's context."""
    scope = GraphScope()
    t1, t2 = scope.datum("Alice one"), scope.datum("Bob two")
    mine, theirs = ContextSource("kb", lambda: ["for t1"], max_items=1), ContextSource("kb", lambda: ["for t2"], 1)
    with pytest.raises(GraphRejection) as info:
        ProtectionGraph(scope, (t1, t2), context=(ContextScope(t1, (mine,)), ContextScope(t2, (theirs,))))
    assert info.value.code == "duplicate_context_source"
