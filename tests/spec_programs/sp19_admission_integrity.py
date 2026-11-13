# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-19 admission and evidence integrity: admitted ports, prepared values, request counts, attribution (E19.1-E19.5)."""

from __future__ import annotations

import dataclasses

from anonymizer.engine.graph_core.authoring import DATUM, TEXT, Assessment, Limits, Operation, Requests, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.engine.graph_core.kernel import GraphRejection, register_kind
from tests.spec_programs._builder import Checks, graph, rejected

METRICS = register_kind("sp19-metrics", lambda v: isinstance(v, dict))


def program() -> list[dict[str, str]]:
    c = Checks("SP-19")
    g, (a, b) = graph("Alice one", "Alice two")

    # E19.1 the caller's port mapping changing after preparation does not change the admitted operation.
    outputs = {"metrics": METRICS}
    prepared = prepare(g, Workflow("w", (Operation("m", lambda text: {"n": 1}, {"text": TEXT}, outputs),)))
    outputs["extra"] = METRICS
    r1 = protect(prepared)
    c.check("E19.1", r1.outcome(a).terminal == "success", f"terminal={r1.outcome(a).terminal}")

    # E19.2 a prepared value with a replaced graph is rejected before any callback.
    calls: list[str] = []
    limits = Limits(max_targets=1, max_text_bytes=4)
    small, _ = graph("ab")
    big, _ = graph(*["x" * 100] * 5)
    p2 = prepare(small, Workflow("w", (Operation("op", lambda text: calls.append(text) or text),)), limits=limits)
    code = rejected(lambda: protect(dataclasses.replace(p2, graph=big)))
    c.check("E19.2", code is not None and not calls, f"rejection={code}, callbacks={len(calls)}")

    # E19.3 a negative request count is rejected.
    code3 = rejected(lambda: Requests.exact(-50))
    c.check("E19.3", code3 == "invalid_request_count", f"rejection={code3}")

    # E19.4 a non-covering verifier blaming the other target makes the invocation inconsistent; evidence stays readable.
    swap = {a: b, b: a}
    blame = Operation.verifier("blame", lambda target: Assessment(True, swap[target]), {"target": DATUM})
    check = Operation.verifier("check", lambda text, target: Assessment(True, target), {"text": TEXT, "target": DATUM})
    r4 = protect(prepare(g, Workflow("w", (Operation("op", lambda text: text.upper()), check, blame))))
    quals = [r4.outcome(t).qualification for t in (a, b)]
    readable = r4.state.rows[0].outputs[2]["assessment"].target == b
    c.check(
        "E19.4",
        not r4.consistent and quals == ["not_met", "not_met"] and readable,
        f"consistent={r4.consistent}, qualification={quals}, assessments readable={readable}",
    )

    # E19.5 a rejection names the limit it hit.
    detail = ""
    try:
        prepare(big, Workflow("w", (Operation("op", lambda text: text.upper()),)), limits=limits)
    except GraphRejection as exc:
        detail = getattr(exc, "detail", "")
    c.check("E19.5", detail == "max_targets", f"detail={detail!r}")
    return c.lines
