# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-14 invocation-scoped operations: one call over all ready targets, per-target fan-out (E14.1-E14.20)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from anonymizer.engine.graph_core.authoring import ASSESSMENT, DATUM, TEXT, Assessment, Operation, Requests, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.engine.graph_core.kernel import CancelToken
from anonymizer.engine.graph_core.observe import report_requests
from anonymizer.engine.ndd.adapter import _report_observed_requests
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
    errors = [e.error for e in r.state.entries if e.status == "failed"]
    rec = Recorder()
    rg = run(
        g,
        rec.invocation_transform(fn=lambda ts: (t for t in _replace(ts))),  # ty: ignore[invalid-argument-type]
    )
    gen_errors = [e.error for e in rg.state.entries if e.status == "failed"]
    c.check(
        "E14.5",
        len(r.attempts) == 3
        and reasons == {"invalid_output"}
        and all(x.status == "failed" for x in r.attempts)
        and len(errors) == 3
        and all(isinstance(e, TypeError) for e in errors)
        and {x.reason for x in rg.attempts} == {"invalid_output"}
        and len(gen_errors) == 3
        and all(isinstance(e, TypeError) for e in gen_errors),
        f"attempts={len(r.attempts)}, reasons={sorted(reasons)}, errors={sorted({type(e).__name__ for e in errors})}, "
        f"generator errors={sorted({type(e).__name__ for e in gen_errors})}",
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
    odd = Operation("odd", rec.transform().fn, requests=Requests.exact(2), scope="Invocation")  # ty: ignore[invalid-argument-type]
    odd_only, odd_both = forecast(odd), forecast(odd, inv)
    code = rejected(lambda: forecast(inv, per_target, request_budget=6))
    c.check(
        "E14.7",
        (only.bound, only.value) == ("exact", 1)
        and (both.bound, both.value) == ("exact", 7)
        and (odd_only.bound, odd_only.value) == ("exact", 6)
        and (odd_both.bound, odd_both.value) == ("exact", 7)
        and code is not None
        and rec.count() == 0,
        f"invocation only={only}, with per-target={both}, budget 6 rejection={code}, callbacks={rec.count()}, "
        f"unknown scope={odd_only}, with invocation={odd_both}",
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

    def interrupt_reaches_caller() -> tuple[bool, str]:
        interrupt = KeyboardInterrupt()
        calls: list[int] = []

        def raises(ts: tuple[str, ...]) -> list[str]:
            calls.append(len(ts))
            raise interrupt

        caught: BaseException | None = None
        try:
            run(g, Recorder().invocation_transform(fn=raises))
        except KeyboardInterrupt as exc:
            caught = exc
        return (
            caught is interrupt and len(calls) == 1,
            f"raised={type(caught).__name__}, same object={caught is interrupt}, callbacks={len(calls)}",
        )

    c.guard("E14.9", interrupt_reaches_caller)

    def cancel_inside_invocation() -> tuple[bool, str]:
        token = CancelToken()
        rec = Recorder()

        def cancel_and_replace(ts: tuple[str, ...]) -> list[str]:
            token.cancel()
            return _replace(ts)

        r = protect(
            prepare(g, Workflow("w", (rec.invocation_transform("inv", cancel_and_replace), rec.transform("per")))),
            cancel=token,
        )
        terms = [r.outcome(t).terminal for t in (a, b, d)]
        per_ledger = [status for _, op, status in r.ledger() if op == "per"]
        return (
            rec.count("inv") == 1
            and rec.count("per") == 0
            and len(r.attempts) == 3
            and all(x.operation == "inv" and x.status == "succeeded" for x in r.attempts)
            and terms == ["cancelled"] * 3
            and per_ledger == ["cancelled"] * 3
            and not r.protected,
            f"inv calls={rec.count('inv')}, per-target calls={rec.count('per')}, attempts={len(r.attempts)}, "
            f"terminals={terms}, per-target ledger={per_ledger}, protected={len(r.protected)}",
        )

    c.guard("E14.10", cancel_inside_invocation)

    def cancel_before_invocation() -> tuple[bool, str]:
        token = CancelToken()
        rec = Recorder()

        def cancel_on_third(text: str) -> str:
            if "three" in text:
                token.cancel()
            return text.replace(NAME, "[NAME]")

        r = protect(
            prepare(g, Workflow("w", (rec.transform("pre", cancel_on_third), rec.invocation_transform("inv")))),
            cancel=token,
        )
        terms = [r.outcome(t).terminal for t in (a, b, d)]
        inv_ledger = [status for _, op, status in r.ledger() if op == "inv"]
        return (
            rec.count("pre") == 3
            and rec.count("inv") == 0
            and len(r.attempts) == 3
            and all(x.operation == "pre" and x.status == "succeeded" for x in r.attempts)
            and terms == ["cancelled"] * 3
            and inv_ledger == ["cancelled"] * 3
            and not r.protected,
            f"pre calls={rec.count('pre')}, inv calls={rec.count('inv')}, attempts={len(r.attempts)}, "
            f"terminals={terms}, inv ledger={inv_ledger}, protected={len(r.protected)}",
        )

    c.guard("E14.11", cancel_before_invocation)

    def observed(r) -> tuple[str, list[int | None]]:
        return str(dict(r.summary().observed)["requests"]), [e.observed_requests for e in r.state.entries]

    def per_target_reports() -> tuple[bool, str]:
        def call(text: str) -> str:
            report_requests(2)
            return text.replace(NAME, "[NAME]")

        r = run(g, Operation("call", call, requests=Requests.exact(2)))
        total, entries = observed(r)
        return total == "exact 6" and entries == [2, 2, 2], f"requests={total}, entries={entries}"

    c.guard("E14.12", per_target_reports)

    def opaque_without_report() -> tuple[bool, str]:
        r = run(g, Operation("call", Recorder().transform().fn, requests=Requests.opaque()))
        total, entries = observed(r)
        return total == "unknown" and entries == [None, None, None], f"requests={total}, entries={entries}"

    c.guard("E14.13", opaque_without_report)

    def invocation_reports_once() -> tuple[bool, str]:
        def call(text: tuple[str, ...]) -> list[str]:
            report_requests(4)
            return _replace(text)

        r = run(g, Operation.per_invocation("inv", call, requests=Requests.exact(4)))
        total, entries = observed(r)
        return total == "exact 4" and entries == [4, None, None], f"requests={total}, entries={entries}"

    c.guard("E14.14", invocation_reports_once)

    def report_outside_run() -> tuple[bool, str]:
        report_requests(5)
        total, entries = observed(run(g, Recorder().transform()))
        return total == "exact 0" and entries == [None, None, None], f"requests={total}, entries={entries}"

    c.guard("E14.15", report_outside_run)

    def partial_reporting() -> tuple[bool, str]:
        def call(text: str) -> str:
            if "three" not in text:
                report_requests(1)
            return text.replace(NAME, "[NAME]")

        r = run(g, Operation("call", call, requests=Requests.exact(1)))
        total, entries = observed(r)
        return total == "unknown" and entries == [1, 1, None], f"requests={total}, entries={entries}"

    c.guard("E14.16", partial_reporting)

    def accumulates_within_call() -> tuple[bool, str]:
        def call(text: str) -> str:
            report_requests(2)
            report_requests(3)
            return text.replace(NAME, "[NAME]")

        r = run(g, Operation("call", call, requests=Requests.opaque()))
        total, entries = observed(r)
        return total == "exact 15" and entries == [5, 5, 5], f"requests={total}, entries={entries}"

    c.guard("E14.17", accumulates_within_call)

    def failed_call_keeps_count() -> tuple[bool, str]:
        def call(text: str) -> str:
            report_requests(2)
            raise RuntimeError("x")

        r = run(g, Operation("call", call, requests=Requests.opaque()))
        total, entries = observed(r)
        statuses = [e.status.value for e in r.state.entries]

        def wrong_kind(text: str) -> int:
            report_requests(2)
            return 7

        r2 = run(g, Operation("call", wrong_kind, requests=Requests.opaque()))
        total2, entries2 = observed(r2)
        reasons = sorted({e.reason for e in r2.state.entries})
        return (
            total == "exact 6"
            and entries == [2, 2, 2]
            and statuses == ["failed"] * 3
            and total2 == "exact 6"
            and entries2 == [2, 2, 2]
            and reasons == ["invalid_output"],
            f"raised: requests={total}, entries={entries}, statuses={statuses}; "
            f"invalid output: requests={total2}, entries={entries2}, reasons={reasons}",
        )

    c.guard("E14.18", failed_call_keeps_count)

    def invocation_failures_keep_count() -> tuple[bool, str]:
        def reporting(shape: Callable[[tuple[str, ...]], list[Any]]) -> Callable[[tuple[str, ...]], list[Any]]:
            def call(text: tuple[str, ...]) -> list[Any]:
                report_requests(4)
                return shape(text)

            return call

        def raises(text: tuple[str, ...]) -> list[Any]:
            raise RuntimeError("x")

        shapes = {
            "wrong length": reporting(lambda ts: _replace(ts)[:2]),
            "raise": reporting(raises),
            "none slot": reporting(lambda ts: [None, *_replace(ts[1:])]),
            "exception slot": reporting(lambda ts: [ValueError("x"), *_replace(ts[1:])]),
        }
        results = {}
        for name, fn in shapes.items():
            total, entries = observed(run(g, Operation.per_invocation("inv", fn, requests=Requests.opaque())))
            results[name] = (total, entries)
        ok = all(v == ("exact 4", [4, None, None]) for v in results.values())
        return ok, "; ".join(f"{k}: requests={t}, entries={e}" for k, (t, e) in results.items())

    c.guard("E14.19", invocation_failures_keep_count)

    def adapter_helper_reports() -> tuple[bool, str]:
        usage = {
            "a": {"request_usage": {"total_requests": 3}},
            "b": {"request_usage": {"successful_requests": 2, "failed_requests": 1}},
            "c": {"token_usage": {}},
        }
        returned: list[int | None] = []

        def call(text: str) -> str:
            returned.append(_report_observed_requests(usage))
            returned.append(_report_observed_requests(None))
            returned.append(_report_observed_requests({}))
            return text.replace(NAME, "[NAME]")

        r = run(g, Operation("call", call, requests=Requests.opaque()))
        total, entries = observed(r)

        def silent(text: str) -> str:
            _report_observed_requests(None)
            _report_observed_requests({"a": {"token_usage": {}}})
            return text.replace(NAME, "[NAME]")

        r2 = run(g, Operation("call", silent, requests=Requests.opaque()))
        total2, entries2 = observed(r2)
        outside = _report_observed_requests(usage)
        total3, entries3 = observed(run(g, Recorder().transform()))
        return (
            total == "exact 18"
            and entries == [6, 6, 6]
            and returned == [6, None, None] * 3
            and total2 == "unknown"
            and entries2 == [None, None, None]
            and outside == 6
            and total3 == "exact 0"
            and entries3 == [None, None, None],
            f"requests={total}, entries={entries}, returned={returned[:3]}; no usage: requests={total2}, "
            f"entries={entries2}; outside run: requests={total3}, entries={entries3}",
        )

    c.guard("E14.20", adapter_helper_reports)
    return c.lines
