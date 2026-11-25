# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission happens before any effect: limits, capabilities, request budgets and prepared-value integrity."""

from __future__ import annotations

import dataclasses

import pytest

from anonymizer.engine.graph_core.authoring import (
    DATUM,
    TEXT,
    Assessment,
    Limits,
    Operation,
    Requests,
    Workflow,
)
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.engine.graph_core.kernel import GraphRejection, register_kind
from anonymizer.engine.graph_core.observe import report_requests
from tests.graph.helpers import NAME, Recorder, graph, run

METRICS = register_kind("sp19-metrics", lambda v: isinstance(v, dict))


@pytest.mark.parametrize(
    ("kwargs", "verify", "code"),
    [
        ({"limits": Limits(max_operations=1)}, True, "over_limit"),
        ({"capabilities": ()}, False, "capability_not_admitted"),
        ({"limits": Limits(max_text_bytes=4)}, False, "over_byte_ceiling"),
    ],
)
def test_rejection_precedes_callbacks(kwargs: dict, verify: bool, code: str) -> None:
    g, _ = graph(f"{NAME} met Bob.")
    rec = Recorder()
    ops = (rec.transform(), rec.verifier()) if verify else (rec.transform(),)
    with pytest.raises(GraphRejection) as info:
        run(g, *ops, **kwargs)
    assert info.value.code == code
    assert rec.count() == 0


def test_raising_callback_is_recorded_as_failure_not_raised() -> None:
    g, (a,) = graph(f"{NAME} met Bob.")

    def boom(text: str) -> str:
        raise RuntimeError(text)

    r = run(g, Operation("transform", boom))
    o = r.outcome(a)
    failed = [x for x in r.attempts if x.status != "succeeded"]
    assert dict(r.content())[a] is None
    assert o.qualification != "met"
    assert not r.protected
    assert failed
    assert o.terminal == "failed"


def test_hard_budget_needs_visible_requests_and_zero_request_ops_pass_budget_zero() -> None:
    g, _ = graph(f"{NAME} met Bob.")
    rec = Recorder()
    opaque = Operation("transform", rec.transform().fn, requests=Requests.opaque(1))
    with pytest.raises(GraphRejection) as info:
        run(g, opaque, request_budget=5)
    assert info.value.code == "hard_budget_without_visibility"
    silent = prepare(g, Workflow("w", (rec.transform(),)), request_budget=0)
    assert rec.count() == 0
    assert silent.planned_requests.value == 0


def _wf(rec: Recorder, requests: Requests) -> Workflow:
    return Workflow("w", (Operation("call", rec.transform().fn, requests=requests),))


@pytest.mark.parametrize(
    ("requests", "budget", "value", "bound"),
    [(Requests.exact(2), 2, 2, "exact"), (Requests.at_most(3), 3, 3, "upper")],
)
def test_forecast_within_budget_is_admitted(requests: Requests, budget: int, value: int, bound: str) -> None:
    g, _ = graph("Alice met Bob.")
    planned = prepare(g, _wf(Recorder(), requests), request_budget=budget).summary().planned_requests
    assert (planned.value, planned.bound) == (value, bound)


@pytest.mark.parametrize(
    ("requests", "budget", "code"),
    [
        (Requests.exact(2), 1, "over_request_budget"),
        (Requests.unknown(), 10, "hard_budget_without_visibility"),
        (Requests.at_most(3), 2, "over_request_budget"),
    ],
)
def test_forecast_over_or_unknown_under_budget_is_rejected(requests: Requests, budget: int, code: str) -> None:
    g, _ = graph("Alice met Bob.")
    rec = Recorder()
    with pytest.raises(GraphRejection) as info:
        prepare(g, _wf(rec, requests), request_budget=budget)
    assert info.value.code == code
    assert rec.count() == 0


def test_unknown_forecast_without_budget_is_admitted_and_rendered() -> None:
    g, _ = graph("Alice met Bob.")
    p = prepare(g, _wf(Recorder(), Requests.unknown()))
    planned = p.summary().planned_requests
    assert planned.value is None
    assert planned.bound == "unknown"
    assert "unknown" in p.summary().render()


def _reports(n: int, skip: str = "") -> Operation:
    def call(text: str) -> str:
        if not skip or skip not in text:
            report_requests(n)
        return text.replace(NAME, "[NAME]")

    return Operation("call", call, requests=Requests.unknown() if skip else Requests.none())


def _observed(r) -> dict[str, str]:
    return {name: str(count) for name, count in r.summary().observed}


def test_partial_request_reports_give_a_lower_bound() -> None:
    g, _ = graph("Alice one.", "Alice two.", "Alice three.")
    seen = _observed(run(g, _reports(2, skip="three")))
    assert seen["requests"] == "lower 4"
    assert seen["total"] == "lower 4"


def test_full_request_reports_give_an_exact_count() -> None:
    g, _ = graph("Alice met Bob.")
    seen = _observed(run(g, _reports(3)))
    assert seen["requests"] == "exact 3"
    assert seen["total"] == "exact 3"


def test_mutating_port_mapping_after_prepare_does_not_change_admitted_operation() -> None:
    g, (a, _) = graph("Alice one", "Alice two")
    outputs = {"metrics": METRICS}
    prepared = prepare(g, Workflow("w", (Operation("m", lambda text: {"n": 1}, {"text": TEXT}, outputs),)))
    outputs["extra"] = METRICS
    assert protect(prepared).outcome(a).terminal == "success"


def test_prepared_value_with_replaced_graph_is_rejected_before_callbacks() -> None:
    calls: list[str] = []
    limits = Limits(max_targets=1, max_text_bytes=4)
    small, _ = graph("ab")
    big, _ = graph(*["x" * 100] * 5)
    p2 = prepare(small, Workflow("w", (Operation("op", lambda text: calls.append(text) or text),)), limits=limits)
    with pytest.raises(GraphRejection) as info:
        protect(dataclasses.replace(p2, graph=big))
    assert info.value.code == "not_prepared"
    assert not calls


def test_negative_request_count_is_rejected() -> None:
    with pytest.raises(GraphRejection) as info:
        Requests.exact(-50)
    assert info.value.code == "invalid_request_count"


def test_verifier_blaming_other_target_makes_run_inconsistent_with_readable_evidence() -> None:
    g, (a, b) = graph("Alice one", "Alice two")
    swap = {a: b, b: a}
    blame = Operation.verifier("blame", lambda target: Assessment(True, swap[target]), {"target": DATUM})
    check = Operation.verifier("check", lambda text, target: Assessment(True, target), {"text": TEXT, "target": DATUM})
    r = protect(prepare(g, Workflow("w", (Operation("op", lambda text: text.upper()), check, blame))))
    assert not r.consistent
    assert [r.outcome(t).qualification for t in (a, b)] == ["not_met", "not_met"]
    assert r.state.rows[0].outputs[2]["assessment"].target == b


def test_rejection_names_the_limit_it_hit() -> None:
    big, _ = graph(*["x" * 100] * 5)
    with pytest.raises(GraphRejection) as info:
        prepare(
            big,
            Workflow("w", (Operation("op", lambda text: text.upper()),)),
            limits=Limits(max_targets=1, max_text_bytes=4),
        )
    assert getattr(info.value, "detail", "") == "max_targets"
