# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-14 invocation-scoped operations: one call over all ready targets, per-target fan-out (E14.1-E14.8)."""

from __future__ import annotations

import threading

from anonymizer.engine.graph_core.authoring import ASSESSMENT, DATUM, TEXT, Assessment, Operation, Requests, Workflow
from anonymizer.engine.graph_core.execution import prepare
from tests.spec_programs._builder import NAME, Checks, Recorder, graph, rejected, run

TEXTS = (f"{NAME} one", f"{NAME} two", f"{NAME} three")


def _replace(texts: tuple[str, ...]) -> list[str]:
    return [t.replace(NAME, "[NAME]") for t in texts]


def program() -> list[dict[str, str]]:
    c = Checks("SP-14")
    g, (a, b, d) = graph(*TEXTS)

    rec = Recorder()
    r = run(g, rec.invocation_transform())
    content = r.content()
    attempts = [x.target for x in r.attempts if x.operation == "transform"]
    c.check(
        "E14.1",
        rec.count() == 1
        and len(rec.calls[0][1]) == 3
        and attempts == [a, b, d]
        and [t for _, t in content] == _replace(TEXTS)
        and all(tid == threading.get_ident() for _, _, tid in rec.calls),
        f"calls={rec.count()}, tuple size={len(rec.calls[0][1])}, attempts={len(attempts)}, items={len(content)}",
    )

    def fail_second(value: object):
        return lambda ts: [value if i == 1 else t.replace(NAME, "[NAME]") for i, t in enumerate(ts)]

    rec = Recorder()
    r = run(g, rec.invocation_transform(fn=fail_second(ValueError("x"))), rec.verifier())
    o = r.outcome(b)
    seen = [t for op, t, _ in rec.calls if op == "verifier"]
    c.check(
        "E14.2",
        (o.terminal, o.reason) == ("failed", "raised")
        and r.outcome(a).terminal == r.outcome(d).terminal == "success"
        and seen == _replace((TEXTS[0], TEXTS[2]))
        and set(r.protected) == {a, d},
        f"target 2 terminal={o.terminal}, reason={o.reason}, verifier calls={len(seen)}, protected={len(r.protected)}",
    )

    rec = Recorder()
    r = run(g, rec.invocation_transform(fn=lambda ts: [None, *_replace(ts[1:])]))
    o = r.outcome(a)
    c.check(
        "E14.3",
        (o.terminal, o.reason) == ("failed", "missing_output")
        and r.outcome(b).terminal == r.outcome(d).terminal == "success",
        f"target 1 terminal={o.terminal}, reason={o.reason}",
    )

    def boom(ts: tuple[str, ...]) -> list[str]:
        raise RuntimeError("x")

    rec = Recorder()
    r = run(g, rec.invocation_transform(fn=boom))
    outcomes = [r.outcome(t) for t in (a, b, d)]
    c.check(
        "E14.4",
        all((o.terminal, o.reason) == ("failed", "raised") for o in outcomes)
        and len(r.ledger()) == 3
        and all(status == "failed" for _, _, status in r.ledger()),
        f"failed={sum(o.terminal == 'failed' for o in outcomes)}, ledger entries={len(r.ledger())}",
    )

    rec = Recorder()
    r = run(g, rec.invocation_transform(fn=lambda ts: _replace(ts)[:2]))
    reasons = {x.reason for x in r.attempts}
    c.check(
        "E14.5",
        len(r.attempts) == 3 and reasons == {"invalid_output"} and all(x.status == "failed" for x in r.attempts),
        f"attempts={len(r.attempts)}, reasons={sorted(reasons)}",
    )

    def pre_fn(t: str) -> str:
        if "two" in t:
            raise ValueError("x")
        return t.replace(NAME, "[NAME]")

    rec = Recorder()
    r = run(g, rec.transform("pre", pre_fn), rec.invocation_transform("inv"))
    ledger = {(t, op): status for t, op, status in r.ledger()}
    c.check(
        "E14.6",
        rec.count("inv") == 1
        and len(rec.calls[-1][1]) == 2
        and ledger[(b, "inv")] == "blocked"
        and ledger[(a, "inv")] == ledger[(d, "inv")] == "succeeded",
        f"inv calls={rec.count('inv')}, tuple size={len(rec.calls[-1][1])}, target 2 inv={ledger[(b, 'inv')]}",
    )

    rec = Recorder()

    def forecast(*ops: Operation, request_budget: int | None = None):
        return prepare(g, Workflow("w", ops), request_budget=request_budget).summary().planned_requests

    inv = Operation.per_invocation("inv", rec.invocation_transform().fn, requests=Requests.exact(1))
    per_target = Operation("call", rec.transform().fn, requests=Requests.exact(2))
    only, both = forecast(inv), forecast(inv, per_target)
    code = rejected(lambda: forecast(inv, per_target, request_budget=6))
    c.check(
        "E14.7",
        (only.bound, only.value) == ("exact", 1)
        and (both.bound, both.value) == ("exact", 7)
        and code is not None
        and rec.count() == 0,
        f"invocation only={only}, with per-target={both}, budget 6 rejection={code}, callbacks={rec.count()}",
    )

    def verify(swap: bool) -> Operation:
        def call(text: tuple[str, ...], target: tuple) -> list[Assessment]:
            blamed = (*target[1:], target[0]) if swap else target
            return [Assessment(NAME not in t, o) for t, o in zip(text, blamed)]

        return Operation.per_invocation(
            "verify", call, {"text": TEXT, "target": DATUM}, {"assessment": ASSESSMENT}, Requests.none()
        )

    def keep_second(ts: tuple[str, ...]) -> list[str]:
        return [t if "two" in t else t.replace(NAME, "[NAME]") for t in ts]

    rec = Recorder()
    r = run(g, rec.invocation_transform(fn=keep_second), verify(False))
    quals = [r.outcome(t).qualification for t in (a, b, d)]
    r2 = run(g, rec.invocation_transform(fn=keep_second), verify(True))
    c.check(
        "E14.8",
        verify(False).is_verifier
        and quals == ["met", "not_met", "met"]
        and set(r.protected) == {a, d}
        and r.consistent
        and not r2.consistent
        and len(r2.protected) == 0,
        f"qualifications={quals}, protected={len(r.protected)}; "
        f"foreign attribution consistent={r2.consistent}, protected={len(r2.protected)}",
    )
    return c.lines
