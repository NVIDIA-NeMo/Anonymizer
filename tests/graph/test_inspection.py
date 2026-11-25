# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspection is bounded and content-free: summaries, renderings and measurement records never carry text."""

from __future__ import annotations

from typing import Any

import pytest

from anonymizer.engine.graph_core.authoring import Operation, Requests, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.engine.graph_core.observe import report_requests
from anonymizer.measurement import MeasurementCollector, measurement_session
from tests.graph.helpers import Recorder, graph, run

CANARY_IN, CANARY_OUT = "CANARYxIN7731", "CANARYxOUT4410"


def _prepared(text: str, transform: Operation | None = None) -> tuple[Any, Any]:
    g, (a,) = graph(text)
    rec = Recorder()
    t = transform or rec.transform(fn=lambda s: s.replace(CANARY_IN, CANARY_OUT))
    return prepare(g, Workflow("w", (t, rec.verifier(needle=CANARY_IN)))), a


def test_summaries_before_and_after_protect_are_content_free() -> None:
    p, a = _prepared(f"Hello {CANARY_IN}.")
    before = p.summary()
    assert before.operations == ("transform", "verifier")
    assert before.terminals is None
    assert p.summary().budget is None

    r = protect(p)
    text = r.summary().render()
    assert dict(r.summary().terminals or ()) == {"success": 1}
    assert CANARY_IN not in text
    assert CANARY_OUT not in text
    assert dict(r.content())[a] == f"Hello {CANARY_OUT}."

    obs = dict(r.summary().observed)
    planned = before.planned_requests
    assert planned.value in (0, None)
    assert all(v.value == 0 or v.value is None for v in obs.values())
    assert set(obs) >= {"binding", "correction", "retry", "total"}
    assert r.summary().planned_requests == planned


def test_failure_summary_names_reason_without_content() -> None:
    def boom(text: str) -> str:
        raise ValueError(f"cannot handle {text}")

    p, _ = _prepared(f"Hello {CANARY_IN}.", Operation("transform", boom))
    s = protect(p).summary()
    text = s.render()
    assert dict(s.terminals or ()) == {"failed": 1}
    assert dict(s.reasons) == {"raised": 1}
    assert CANARY_IN not in text
    assert CANARY_OUT not in text


def test_rendering_size_does_not_grow_with_input() -> None:
    small = protect(_prepared("x" * 10)[0]).summary().render()
    large = protect(_prepared("x" * 100_000)[0]).summary().render()
    assert len(large) - len(small) <= 64


def _budget(first: int, second: int | None) -> tuple[str | None, str]:
    """Planned 10 (two steps declaring exact 5); ``None`` leaves the second step unreported."""

    def step(n: int | None):
        def call(text: str) -> str:
            if n is not None:
                report_requests(n)
            return text

        return call

    g, _ = graph("x")
    ops = (
        Operation("a", step(first), requests=Requests.exact(5)),
        Operation("b", step(second), requests=Requests.exact(5)),
    )
    s = run(g, *ops).summary()
    return s.budget, s.render()


@pytest.mark.parametrize(
    ("first", "second", "verdict"), [(12, None, "exceeded"), (7, None, "unknown"), (7, 0, "within")]
)
def test_budget_verdict_from_reported_requests(first: int, second: int | None, verdict: str) -> None:
    got, text = _budget(first, second)
    assert got == verdict
    assert f"budget: {got}" in text


def test_measurement_records_are_content_free_and_do_not_change_results() -> None:
    with measurement_session(MeasurementCollector(keep_records=True)) as collector:
        measured = protect(_prepared(f"Hello {CANARY_IN}.")[0])
    stages = [(m["stage"], m["status"], m["schema_version"]) for m in collector.records]
    assert sorted(s for s, _, _ in stages) == ["graph_prepare", "graph_protect", "graph_reconcile"]
    assert all(st == "completed" and v is not None for _, st, v in stages)
    assert all(m["terminal_counts"] == {"success": 1} for m in collector.records if m["stage"] != "graph_prepare")
    for m in collector.records:
        assert CANARY_IN not in repr(m)
        assert CANARY_OUT not in repr(m)
        assert "transform" not in repr(m)

    plain = protect(_prepared(f"Hello {CANARY_IN}.")[0])
    assert plain.content()[0][1] == measured.content()[0][1]
    assert plain.summary() == measured.summary()
