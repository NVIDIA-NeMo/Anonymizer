# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-12 lifecycle: caller-owned resources, uncertain effects, attribution (E12.2-E12.5; E12.1 retired)."""

from __future__ import annotations

from anonymizer.engine.graph_core import Assessment, ContextScope, ContextSource, GraphScope, Operation, ProtectionGraph
from tests.spec_programs._builder import Checks, graph, rejected, run


class Client:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _with_resource(fail: bool) -> bool:
    client = Client()
    scope = GraphScope()
    t = scope.datum("Alice")

    def op(text: str, context: tuple) -> str:
        if fail:
            raise RuntimeError("callback failure")
        return text if client else ""

    src = ContextSource("client", lambda: [repr(client)], max_items=1)
    run(
        ProtectionGraph(scope, (t,), context=(ContextScope(t, (src,)),)),
        Operation("op", op, {"text": "text", "context": "context"}),
    )
    return client.closed


def program() -> list[dict[str, str]]:
    c = Checks("SP-12")
    c.retired("E12.1", "retired 2026-10-22 (item 20)")
    closed = [_with_resource(False), _with_resource(True)]
    c.check(
        "E12.2", closed == [False, False], f"closed after success/failure={closed}"
    )  # C.failure-cancellation-and-release.p4
    effects: list[str] = []

    def uncertain(text: str) -> str:
        effects.append("sent")
        raise ConnectionResetError("after effect")

    g, (a,) = graph("Alice")
    r = run(g, Operation("send", uncertain))
    c.check(
        "E12.3",
        effects == ["sent"] and r.outcome(a).qualification != "met",
        f"invocations={len(effects)}, qualification={r.outcome(a).qualification}",
    )  # RFC :2677; C.hierarchical-workflows-and-bounded-cycle.r4
    g, (t1, t2) = graph("Alice one", "Bob two")
    swap = {t1: t2, t2: t2}  # the verifier for t1 returns its finding keyed to t2

    def verify(text: str, target) -> Assessment:
        return Assessment(True, swap[target])

    r = run(
        g,
        Operation("transform", lambda text: text),
        Operation.verifier("verify", verify, {"text": "text", "target": "datum"}),
    )
    quals = [r.outcome(t).qualification for t in (t1, t2)]
    c.check(
        "E12.4",
        "met" not in quals and not r.protected,
        f"verifiers attribute findings by target; qualifications={quals}, protected={len(r.protected)}",
    )  # C.failure-cancellation-and-release.p3, RFC :2692
    g, ts = graph("a", "b", "c")

    def fail(text: str) -> str:
        raise ValueError("no")

    r = run(g, Operation("first", fail), Operation("never", lambda text: text))
    by_op = [a.operation for a in r.attempts]
    c.check(
        "E12.5",
        by_op.count("first") == 3 and by_op.count("never") == 0,
        f"entries first={by_op.count('first')}, never={by_op.count('never')}",
    )  # RFC :3182
    scope = GraphScope()
    t1, t2 = scope.datum("Alice one"), scope.datum("Bob two")
    mine, theirs = ContextSource("kb", lambda: ["for t1"], max_items=1), ContextSource("kb", lambda: ["for t2"], 1)
    code = rejected(
        lambda: ProtectionGraph(scope, (t1, t2), context=(ContextScope(t1, (mine,)), ContextScope(t2, (theirs,))))
    )
    c.check(
        "F2",
        code == "duplicate_context_source",
        f"two sources under one name are rejected at graph construction; code={code}",
    )  # A2 ruling: one target never receives another target's context
    return c.lines
