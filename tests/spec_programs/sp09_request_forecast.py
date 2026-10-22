# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-9 request forecast, hard budgets and unknown bounds (E9.1-E9.5b)."""

from __future__ import annotations

from anonymizer.engine.graph_core.authoring import Operation, Requests, Workflow
from anonymizer.engine.graph_core.execution import prepare
from tests.spec_programs._builder import Checks, Recorder, graph, rejected


def program() -> list[dict[str, str]]:
    c = Checks("SP-9")
    g, _ = graph("Alice met Bob.")
    rec = Recorder()

    def wf(requests: Requests) -> Workflow:
        return Workflow("w", (Operation("call", rec.transform().fn, requests=requests),))

    p = prepare(g, wf(Requests.exact(2)), request_budget=2)
    planned = p.summary().planned_requests
    c.check("E9.1", (planned.value, planned.bound) == (2, "exact"), f"admitted, planned={planned}")
    code = rejected(lambda: prepare(g, wf(Requests.exact(2)), request_budget=1))
    c.check("E9.2", code is not None and rec.count() == 0, f"rejection={code}, callbacks={rec.count()}")
    p = prepare(g, wf(Requests.unknown()))
    planned = p.summary().planned_requests
    c.check(
        "E9.3",
        planned.value is None and planned.bound == "unknown" and "unknown" in p.summary().render(),
        f"admitted, planned={planned}",
    )
    code = rejected(lambda: prepare(g, wf(Requests.unknown()), request_budget=10))
    c.check("E9.4", code is not None and rec.count() == 0, f"rejection={code}")
    p = prepare(g, wf(Requests.at_most(3)), request_budget=3)
    planned = p.summary().planned_requests
    c.check("E9.5", (planned.value, planned.bound) == (3, "upper"), f"admitted, planned={planned}")
    code = rejected(lambda: prepare(g, wf(Requests.at_most(3)), request_budget=2))
    c.check("E9.5b", code is not None and rec.count() == 0, f"rejection={code}")
    return c.lines
