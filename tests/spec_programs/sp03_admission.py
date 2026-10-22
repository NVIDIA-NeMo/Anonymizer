# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-3 admission before effects, failure accounting and hard budgets (E3.1-E3.4b, E3.5')."""

from __future__ import annotations

from anonymizer.engine.graph_core import Limits, Operation, Requests, Workflow, prepare
from tests.spec_programs._builder import NAME, Checks, Recorder, graph, rejected, run


def program() -> list[dict[str, str]]:
    c = Checks("SP-3")
    g, (a,) = graph(f"{NAME} met Bob.")
    rec = Recorder()
    code = rejected(lambda: run(g, rec.transform(), rec.verifier(), limits=Limits(max_operations=1)))
    c.check("E3.1", code is not None and rec.count() == 0, f"rejection={code}, callbacks={rec.count()}")  # D.REQ6.s1
    code = rejected(lambda: run(g, rec.transform(), capabilities=()))
    c.check("E3.2", code is not None and rec.count() == 0, f"rejection={code}, callbacks={rec.count()}")  # D.REQ15.s3
    code = rejected(lambda: run(g, rec.transform(), limits=Limits(max_text_bytes=4)))
    c.check("E3.3", code == "over_byte_ceiling" and rec.count() == 0, f"rejection={code}")  # D.REQ9.s5

    def boom(text: str) -> str:
        raise RuntimeError(text)

    try:
        r = run(g, Operation("transform", boom))
        raised = None
    except Exception as exc:  # noqa: BLE001
        r, raised = None, type(exc).__name__
    o = r.outcome(a) if r else None
    failed = [x for x in r.attempts if x.status != "succeeded"] if r else []
    c.check(
        "E3.4",
        r is not None
        and o is not None
        and raised is None
        and dict(r.content())[a] is None
        and o.qualification != "met"
        and not r.protected
        and bool(failed),
        f"raised={raised}, qualification={o and o.qualification}, failure entries={len(failed)}",
    )  # D.REQ7.s1, D.REQ16.s1
    c.check("E3.4b", o is not None and o.terminal == "failed", f"terminal={o and o.terminal}")
    c.retired("E3.5", "retired 2026-10-22 (item 19); superseded by E3.5'")
    opaque = Operation("transform", rec.transform().fn, requests=Requests.opaque(1))
    code = rejected(lambda: run(g, opaque, request_budget=5))
    silent = prepare(g, Workflow("w", (rec.transform(),)), request_budget=0)
    c.check(
        "E3.5'",
        code is not None and rec.count() == 0 and silent.planned_requests.value == 0,
        f"opaque+hard budget rejection={code}, zero-request implementation admitted under budget 0, callbacks={rec.count()}",
    )  # D.REQ6.s7, D.REQ16.s6
    return c.lines
