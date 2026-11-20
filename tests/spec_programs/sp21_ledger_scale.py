# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-21 ledger causes and long dependency chains (E21.1-E21.3)."""

from __future__ import annotations

from anonymizer.engine.graph_core.authoring import Dependency, GraphScope, Operation, ProtectionGraph, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.engine.graph_core.kernel import CancelToken
from tests.spec_programs._builder import Checks, graph, rejected


def _chain(size: int, cycle: bool) -> ProtectionGraph:
    scope = GraphScope()
    ts = tuple(scope.datum(f"t{i}") for i in range(size))
    deps = [Dependency(ts[i], ts[i + 1]) for i in range(size - 1)]  # each target depends on the next
    return ProtectionGraph(scope, ts, dependencies=(*deps, *([Dependency(ts[-1], ts[0])] if cycle else ())))


def program() -> list[dict[str, str]]:
    c = Checks("SP-21")

    # E21.1 a failure before a later cancellation keeps its target's unrun nodes blocked; other targets are cancelled.
    g, (a, b) = graph("fail", "cancel")
    token = CancelToken()

    def first(text: str) -> str:
        if text == "fail":
            raise ValueError("x")
        token.cancel()
        return text

    steps = tuple(Operation(n, lambda text: text) for n in ("second", "third"))
    r = protect(prepare(g, Workflow("w", (Operation("first", first), *steps))), cancel=token)
    ledger = [[s for t, _, s in r.ledger() if t == d] for d in (a, b)]
    c.check(
        "E21.1",
        ledger == [["failed", "blocked", "blocked"], ["succeeded", "cancelled", "cancelled"]],
        f"ledger={ledger}",
    )

    # E21.2 a 1,000-target dependency chain is admitted and released end to end.
    def chain() -> tuple[bool, str]:
        g2 = _chain(1_000, cycle=False)
        r2 = protect(prepare(g2, Workflow("w", (Operation("op", lambda text: text),))))
        return len(r2.outcomes) == 1_000, f"outcomes={len(r2.outcomes)}"

    c.guard("E21.2", chain)

    # E21.3 a 1,000-target cycle is a typed rejection.
    code = rejected(lambda: _chain(1_000, cycle=True))
    c.check("E21.3", code == "dependency_cycle", f"rejection={code}")
    return c.lines
