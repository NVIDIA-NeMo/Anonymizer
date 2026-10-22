# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-10 nested synchronous execution frames (E10.1-E10.5)."""

from __future__ import annotations

import contextvars
import threading
from typing import Any

from anonymizer.engine.graph_core.authoring import Assessment, Operation, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from tests.spec_programs._builder import NAME, Checks, graph

VAR: contextvars.ContextVar[str] = contextvars.ContextVar("sp10", default="unset")


def _nested(inner_fn=None):
    """Outer one-operation workflow (plus its check) whose callback runs an inner transform+verifier workflow."""
    calls: list[tuple[str, str, int]] = []
    seen: dict[str, object] = {}
    inner_g, (ia,) = graph(f"{NAME} met Bob.")

    def inner_transform(text: str) -> str:
        calls.append(("transform", text, threading.get_ident()))
        seen["inner_var"] = VAR.get()
        VAR.set("set-inside-inner")
        return (inner_fn or (lambda t: t.replace(NAME, "[NAME]")))(text)

    def inner_verify(text: str, target) -> Assessment:
        calls.append(("verifier", text, threading.get_ident()))
        seen["verifier_var"] = VAR.get()
        return Assessment(NAME not in text, target)

    verifier = Operation.verifier("verifier", inner_verify, {"text": "text", "target": "datum"})
    inner = prepare(inner_g, Workflow("inner", (Operation("transform", inner_transform), verifier)))

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


def program() -> list[dict[str, str]]:
    c = Checks("SP-10")
    me = threading.get_ident()
    calls, seen, outer, ia, oa = _nested()
    box: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: box.update(tid=threading.get_ident(), out=_nested()))
    worker.start()
    worker.join()
    calls2 = box["out"][0]
    c.check(
        "E10.1",
        len(calls) == 2
        and all(t == me for _, _, t in calls)
        and len(calls2) == 2
        and all(t == box["tid"] for _, _, t in calls2),
        f"calls on main thread={len(calls)}, on caller-created thread={len(calls2)}",
    )  # X.D.L3157.s3, s4
    inner = seen["inner_result"]
    order = [op for op, _, _ in calls]
    c.check(
        "E10.2",
        order == ["transform", "verifier"]
        and inner.outcome(ia).qualification == "met"
        and calls[1][1] == "[NAME] met Bob.",
        f"order={order}, inner qualification={inner.outcome(ia).qualification}",
    )  # D.REQ16.s1, D.REQ15.s3
    c.check(
        "E10.3",
        seen["inner_var"] == "caller-value"
        and seen["verifier_var"] in ("caller-value", "set-inside-inner")  # same frame: either is conforming
        and seen["outer_after_inner"] == "caller-value"
        and seen["caller_after"] == "caller-value",
        f"inner saw={seen['inner_var']}, outer after inner={seen['outer_after_inner']}, caller after={seen['caller_after']}",
    )  # X.D.L3157.s5, s6

    def boom(text: str) -> str:
        raise RuntimeError("inner failure")

    try:
        _, seen4, outer4, ia4, oa4 = _nested(boom)
        raised = None
    except Exception as exc:  # noqa: BLE001
        raised = type(exc).__name__
    ok4 = raised is None and seen4["inner_result"].outcome(ia4).terminal == "failed"
    c.check(
        "E10.4",
        ok4 and outer4.outcome(oa4).qualification != "met",
        f"raised={raised}, outer qualification={outer4.outcome(oa4).qualification if raised is None else None}",
    )  # D.REQ7.s1, D.REQ7.s2
    inner_ops = {name for _, name, _ in inner.ledger()}
    outer_ops = {name for _, name, _ in outer.ledger()}
    c.check(
        "E10.5",
        len(inner.attempts) == 2
        and len(outer.attempts) == 2
        and inner_ops == {"transform", "verifier"}
        and outer_ops == {"outer", "check"},
        f"inner attempts={len(inner.attempts)}, outer attempts={len(outer.attempts)} (outer + its verifier)",
    )  # D.REQ7.s1
    return c.lines
