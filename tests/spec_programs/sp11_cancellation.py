# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-11 lifecycle: synchronous cancellation through a CancelToken (E11.1-E11.5; E11.3 retired)."""

from __future__ import annotations

from anonymizer.engine.graph_core import CancelToken, Operation, Workflow, prepare, protect
from tests.spec_programs._builder import Checks, graph


def _setup(cancel_in_first: bool):
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


def _run(p, token):
    try:
        return protect(p, cancel=token), None
    except Exception as exc:  # noqa: BLE001
        return None, type(exc).__name__


def program() -> list[dict[str, str]]:
    c = Checks("SP-11")
    complete: list[bool] = []
    p, token, calls, (t1, t2) = _setup(False)
    token.cancel()
    r1, raised1 = _run(p, token)
    terms = [r1.outcome(t).terminal for t in (t1, t2)] if r1 else []
    c.check(
        "E11.1",
        not calls and terms == ["cancelled", "cancelled"] and not r1.protected,
        f"callbacks={len(calls)}, terminals={terms}",
    )  # C.failure-cancellation-and-release.p2
    complete.append(r1 is not None and raised1 is None and len(r1.ledger()) == 4)
    p, token, calls, (t1, t2) = _setup(True)
    r2, raised2 = _run(p, token)
    c.check(
        "E11.2",
        r2 is not None and calls == [("first", "Alice one")] and not any(a.operation == "second" for a in r2.attempts),
        f"dispatches={[op for op, _ in calls]}",
    )  # X.D.L3182.s4
    c.retired("E11.3", "retired 2026-10-22 (item 15); superseded by E11.3'")
    o1 = r2.outcome(t1) if r2 else None
    dispatched = [a for a in r2.attempts if a.target == t1] if r2 else []
    c.check(
        "E11.3'",
        o1 is not None and o1.terminal not in ("lost", "failed") and len(dispatched) == 1 and not r2.protected,
        f"terminal={o1 and o1.terminal}, recorded attempt={[a.status for a in dispatched]}",
    )  # RFC :758; C.complete-outcome-accounting.p2
    complete.append(r2 is not None and raised2 is None and len(r2.ledger()) == 4)
    p, token, calls, (t1, t2) = _setup(False)
    r3, raised3 = _run(p, token)
    before = ([r3.outcome(t) for t in (t1, t2)], dict(r3.protected), r3.ledger())
    token.cancel()
    after = ([r3.outcome(t) for t in (t1, t2)], dict(r3.protected), r3.ledger())
    c.check(
        "E11.4", before == after, f"terminals={[o.terminal for o in after[0]]}"
    )  # C.failure-cancellation-and-release.p2
    complete.append(raised3 is None and len(r3.ledger()) == 4)
    c.check(
        "E11.5",
        all(complete),
        f"complete accounting (one terminal per target and operation, nothing raised) in {sum(complete)}/3 cases",
    )  # D.REQ7.s1
    return c.lines
