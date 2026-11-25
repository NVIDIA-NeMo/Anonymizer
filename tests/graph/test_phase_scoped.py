# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase-scoped operations make one call over all ready targets and fan results out per target; stage sequences
run through ``run_stages`` in order on the caller's thread."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_core import stages
from anonymizer.engine.graph_core.authoring import ASSESSMENT, DATUM, TEXT, Assessment, Operation, Requests, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.engine.graph_core.kernel import CancelToken
from anonymizer.engine.graph_core.observe import report_requests
from anonymizer.engine.ndd.adapter import _report_observed_requests
from tests.graph.helpers import NAME, Recorder, graph, rejected, run

TEXTS = (f"{NAME} one", f"{NAME} two", f"{NAME} three")


def _replace(texts: tuple[str, ...]) -> list[str]:
    return [t.replace(NAME, "[NAME]") for t in texts]


def _observed(r: Any) -> tuple[str, list[int | None]]:
    return str(dict(r.summary().observed)["requests"]), [e.observed_requests for e in r.state.entries]


def test_one_call_over_all_targets_on_caller_thread() -> None:
    g, (a, b, d) = graph(*TEXTS)
    rec = Recorder()
    r = run(g, rec.invocation_transform())
    attempts = [x.target for x in r.attempts if x.operation == "transform"]
    assert rec.count() == 1
    assert len(rec.calls[0][1]) == 3
    assert attempts == [a, b, d]
    assert [t for _, t in r.content()] == _replace(TEXTS)
    assert all(tid == threading.get_ident() for _, _, tid in rec.calls)


def test_exception_slot_fails_only_that_target() -> None:
    g, (a, b, d) = graph(*TEXTS)
    rec = Recorder()

    def fail_second(ts: tuple[str, ...]) -> list[Any]:
        return [ValueError("x") if i == 1 else t.replace(NAME, "[NAME]") for i, t in enumerate(ts)]

    r = run(g, rec.invocation_transform(fn=fail_second), rec.verifier())
    o = r.outcome(b)
    assert (o.terminal, o.reason) == ("failed", "raised")
    assert r.outcome(a).terminal == r.outcome(d).terminal == "success"
    assert [t for op, t, _ in rec.calls if op == "verifier"] == _replace((TEXTS[0], TEXTS[2]))
    assert set(r.protected) == {a, d}


def test_none_slot_is_missing_output_for_that_target() -> None:
    g, (a, b, d) = graph(*TEXTS)
    r = run(g, Recorder().invocation_transform(fn=lambda ts: [None, *_replace(ts[1:])]))
    o = r.outcome(a)
    assert (o.terminal, o.reason) == ("failed", "missing_output")
    assert r.outcome(b).terminal == r.outcome(d).terminal == "success"


def test_raising_call_fails_every_target() -> None:
    g, targets = graph(*TEXTS)

    def boom(ts: tuple[str, ...]) -> list[str]:
        raise RuntimeError("x")

    r = run(g, Recorder().invocation_transform(fn=boom))
    assert all((r.outcome(t).terminal, r.outcome(t).reason) == ("failed", "raised") for t in targets)
    assert len(r.ledger()) == 3
    assert all(status == "failed" for _, _, status in r.ledger())


@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(lambda ts: _replace(ts)[:2], id="wrong-length"),
        pytest.param(lambda ts: (t for t in _replace(ts)), id="generator"),
    ],
)
def test_wrong_shaped_output_is_invalid_for_every_target(fn: Callable[[tuple[str, ...]], Any]) -> None:
    g, _ = graph(*TEXTS)
    r = run(g, Recorder().invocation_transform(fn=fn))
    errors = [e.error for e in r.state.entries if e.status == "failed"]
    assert len(r.attempts) == 3
    assert {x.reason for x in r.attempts} == {"invalid_output"}
    assert all(x.status == "failed" for x in r.attempts)
    assert len(errors) == 3
    assert all(isinstance(e, TypeError) for e in errors)


def test_blocked_target_is_left_out_of_the_call() -> None:
    g, (a, b, d) = graph(*TEXTS)

    def pre_fn(t: str) -> str:
        if "two" in t:
            raise ValueError("x")
        return t.replace(NAME, "[NAME]")

    rec = Recorder()
    r = run(g, rec.transform("pre", pre_fn), rec.invocation_transform("inv"))
    ledger = {(t, op): status for t, op, status in r.ledger()}
    assert rec.count("inv") == 1
    assert len(rec.calls[-1][1]) == 2
    assert ledger[(b, "inv")] == "blocked"
    assert ledger[(a, "inv")] == ledger[(d, "inv")] == "succeeded"


def test_request_forecast_counts_invocation_once() -> None:
    g, _ = graph(*TEXTS)
    rec = Recorder()

    def forecast(*ops: Operation, request_budget: int | None = None) -> Any:
        return prepare(g, Workflow("w", ops), request_budget=request_budget).summary().planned_requests

    inv = Operation.per_invocation("inv", rec.invocation_transform().fn, requests=Requests.exact(1))
    per_target = Operation("call", rec.transform().fn, requests=Requests.exact(2))
    only, both = forecast(inv), forecast(inv, per_target)
    odd = Operation("odd", rec.transform().fn, requests=Requests.exact(2), scope="Invocation")  # ty: ignore[invalid-argument-type]
    odd_only, odd_both = forecast(odd), forecast(odd, inv)
    assert (only.bound, only.value) == ("exact", 1)
    assert (both.bound, both.value) == ("exact", 7)
    assert (odd_only.bound, odd_only.value) == ("exact", 6)
    assert (odd_both.bound, odd_both.value) == ("exact", 7)
    assert rejected(lambda: forecast(inv, per_target, request_budget=6)) == "over_request_budget"
    assert rec.count() == 0


def _verify(swap: bool) -> Operation:
    def call(text: tuple[str, ...], target: tuple) -> list[Assessment]:
        blamed = (*target[1:], target[0]) if swap else target
        return [Assessment(NAME not in t, o) for t, o in zip(text, blamed)]

    op = Operation.per_invocation(
        "verify", call, {"text": TEXT, "target": DATUM}, {"assessment": ASSESSMENT}, Requests.none()
    )
    return replace(op, checks=("verify",), subject=("text",))


def test_phase_scoped_verifier_attributes_assessments_per_target() -> None:
    g, (a, b, d) = graph(*TEXTS)

    def keep_second(ts: tuple[str, ...]) -> list[str]:
        return [t if "two" in t else t.replace(NAME, "[NAME]") for t in ts]

    rec = Recorder()
    r = run(g, rec.invocation_transform(fn=keep_second), _verify(False))
    r2 = run(g, rec.invocation_transform(fn=keep_second), _verify(True))
    assert _verify(False).is_verifier
    assert [r.outcome(t).qualification for t in (a, b, d)] == ["met", "not_met", "met"]
    assert set(r.protected) == {a, d}
    assert r.consistent
    assert not r2.consistent
    assert len(r2.protected) == 0


def test_interrupt_reaches_caller() -> None:
    g, _ = graph(*TEXTS)
    interrupt = KeyboardInterrupt()
    calls: list[int] = []

    def raises(ts: tuple[str, ...]) -> list[str]:
        calls.append(len(ts))
        raise interrupt

    with pytest.raises(KeyboardInterrupt) as info:
        run(g, Recorder().invocation_transform(fn=raises))
    assert info.value is interrupt
    assert len(calls) == 1


def test_cancel_inside_invocation() -> None:
    g, targets = graph(*TEXTS)
    token = CancelToken()
    rec = Recorder()

    def cancel_and_replace(ts: tuple[str, ...]) -> list[str]:
        token.cancel()
        return _replace(ts)

    r = protect(
        prepare(g, Workflow("w", (rec.invocation_transform("inv", cancel_and_replace), rec.transform("per")))),
        cancel=token,
    )
    assert rec.count("inv") == 1
    assert rec.count("per") == 0
    assert len(r.attempts) == 3
    assert all(x.operation == "inv" and x.status == "succeeded" for x in r.attempts)
    assert [r.outcome(t).terminal for t in targets] == ["cancelled"] * 3
    assert [status for _, op, status in r.ledger() if op == "per"] == ["cancelled"] * 3
    assert not r.protected


def test_cancel_before_invocation() -> None:
    g, targets = graph(*TEXTS)
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
    assert rec.count("pre") == 3
    assert rec.count("inv") == 0
    assert len(r.attempts) == 3
    assert all(x.operation == "pre" and x.status == "succeeded" for x in r.attempts)
    assert [r.outcome(t).terminal for t in targets] == ["cancelled"] * 3
    assert [status for _, op, status in r.ledger() if op == "inv"] == ["cancelled"] * 3
    assert not r.protected


def _per_target_reporting(text: str) -> str:
    report_requests(2)
    return text.replace(NAME, "[NAME]")


def _partial_reporting(text: str) -> str:
    if "three" not in text:
        report_requests(1)
    return text.replace(NAME, "[NAME]")


def _accumulating(text: str) -> str:
    report_requests(2)
    report_requests(3)
    return text.replace(NAME, "[NAME]")


def _invocation_reporting(text: tuple[str, ...]) -> list[str]:
    report_requests(4)
    return _replace(text)


@pytest.mark.parametrize(
    ("op", "total", "entries"),
    [
        pytest.param(
            Operation("call", _per_target_reporting, requests=Requests.exact(2)), "exact 6", [2, 2, 2], id="per-target"
        ),
        pytest.param(
            Operation("call", Recorder().transform().fn, requests=Requests.opaque()),
            "unknown",
            [None, None, None],
            id="opaque-without-report",
        ),
        pytest.param(
            Operation.per_invocation("inv", _invocation_reporting, requests=Requests.exact(4)),
            "exact 4",
            [4, None, None],
            id="invocation-once",
        ),
        pytest.param(
            Operation("call", _partial_reporting, requests=Requests.exact(1)), "lower 2", [1, 1, None], id="partial"
        ),
        pytest.param(
            Operation("call", _accumulating, requests=Requests.opaque()), "exact 15", [5, 5, 5], id="accumulates"
        ),
    ],
)
def test_observed_requests(op: Operation, total: str, entries: list[int | None]) -> None:
    g, _ = graph(*TEXTS)
    assert _observed(run(g, op)) == (total, entries)


def test_report_outside_run_is_not_counted() -> None:
    g, _ = graph(*TEXTS)
    report_requests(5)
    assert _observed(run(g, Recorder().transform())) == ("exact 0", [None, None, None])


def test_failed_call_keeps_its_count() -> None:
    g, _ = graph(*TEXTS)

    def call(text: str) -> str:
        report_requests(2)
        raise RuntimeError("x")

    def wrong_kind(text: str) -> int:
        report_requests(2)
        return 7

    r = run(g, Operation("call", call, requests=Requests.opaque()))
    r2 = run(g, Operation("call", wrong_kind, requests=Requests.opaque()))
    assert _observed(r) == ("exact 6", [2, 2, 2])
    assert [e.status.value for e in r.state.entries] == ["failed"] * 3
    assert _observed(r2) == ("exact 6", [2, 2, 2])
    assert sorted({e.reason for e in r2.state.entries}) == ["invalid_output"]


def _raises(text: tuple[str, ...]) -> list[Any]:
    raise RuntimeError("x")


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(lambda ts: _replace(ts)[:2], id="wrong-length"),
        pytest.param(_raises, id="raise"),
        pytest.param(lambda ts: [None, *_replace(ts[1:])], id="none-slot"),
        pytest.param(lambda ts: [ValueError("x"), *_replace(ts[1:])], id="exception-slot"),
    ],
)
def test_invocation_failures_keep_count(shape: Callable[[tuple[str, ...]], list[Any]]) -> None:
    g, _ = graph(*TEXTS)

    def call(text: tuple[str, ...]) -> list[Any]:
        report_requests(4)
        return shape(text)

    assert _observed(run(g, Operation.per_invocation("inv", call, requests=Requests.opaque()))) == (
        "exact 4",
        [4, None, None],
    )


def test_adapter_helper_reports_request_usage() -> None:
    g, _ = graph(*TEXTS)
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

    def silent(text: str) -> str:
        _report_observed_requests(None)
        _report_observed_requests({"a": {"token_usage": {}}})
        return text.replace(NAME, "[NAME]")

    r = run(g, Operation("call", call, requests=Requests.opaque()))
    r2 = run(g, Operation("call", silent, requests=Requests.opaque()))
    outside = _report_observed_requests(usage)
    r3 = run(g, Recorder().transform())
    assert _observed(r) == ("exact 18", [6, 6, 6])
    assert returned == [6, None, None] * 3
    assert _observed(r2) == ("unknown", [None, None, None])
    assert outside == 6
    assert _observed(r3) == ("exact 0", [None, None, None])


def test_phase_scoped_name_and_old_spelling() -> None:
    g, _ = graph(*TEXTS)
    rec = Recorder()
    r = run(g, Operation.phase_scoped("transform", rec.invocation_transform().fn))
    old = Operation("old", rec.invocation_transform().fn, scope="invocation")  # ty: ignore[invalid-argument-type]
    assert rec.count() == 1
    assert len(rec.calls[0][1]) == 3
    assert [t for _, t in r.content()] == _replace(TEXTS)
    assert old.scope == "phase_scoped"


def _stages(order: list[str], tids: list[int], behaviours: dict[str, Callable[[], None]] | None = None) -> list:
    def stage(name: str) -> Callable[[], None]:
        def run_stage() -> None:
            order.append(name)
            tids.append(threading.get_ident())
            if behaviours and name in behaviours:
                behaviours[name]()

        return run_stage

    return [(name, stage(name)) for name in ("first", "second", "third")]


def _raise(exc: BaseException) -> Callable[[], None]:
    def run_stage() -> None:
        raise exc

    return run_stage


def test_stages_run_in_order_as_one_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    tids: list[int] = []
    results: list[Any] = []
    real = stages.protect

    def capture(prepared: Any, **kw: Any) -> Any:
        results.append(real(prepared, **kw))
        return results[-1]

    monkeypatch.setattr(stages, "protect", capture)
    stages.run_stages("w", _stages(order, tids))
    assert order == ["first", "second", "third"]
    assert set(tids) == {threading.get_ident()}
    assert len(results) == 1
    assert len(results[0].targets) == 1
    assert len(results[0].attempts) == 3
    assert all(x.status == "succeeded" for x in results[0].attempts)


@pytest.mark.parametrize("exc", [RuntimeError("x"), KeyboardInterrupt()], ids=["exception", "interrupt"])
def test_stage_failure_reaches_caller_unchanged(exc: BaseException) -> None:
    order: list[str] = []
    with pytest.raises(type(exc)) as info:
        stages.run_stages("w", _stages(order, [], {"second": _raise(exc)}))
    assert info.value is exc
    assert order == ["first", "second"]
