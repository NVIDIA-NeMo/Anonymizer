# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-16 an invocation-scoped operation with an internal bounded loop and a covering verifier (E16.1-E16.3)."""

from __future__ import annotations

from typing import Any

from anonymizer.engine.graph_core.authoring import DATUM, TEXT, Assessment, Operation, Requests
from anonymizer.engine.graph_core.observe import report_requests
from tests.spec_programs._builder import Checks, graph, run

TEXTS = ("alpha", "beta", "gamma")
PASSES = (1, 2, 3)
BAD = {(1, 1), (2, 3)}  # (target index, pass): the candidate carries the marker only at that pass


class _Loop:
    """The body: three passes over the targets, each rewriting every value; ``history`` keeps every pass."""

    def __init__(self, *, fail: int | None = None, requests: int = 0) -> None:
        self.history: list[list[str]] = []
        self.fail = fail
        self.requests = requests

    def operation(self) -> Operation:
        def body(text: tuple[str, ...]) -> list[Any]:
            values = list(text)
            for p in PASSES:
                values = [f"{v.split(' ')[0]} p{p}{' BAD' if (i, p) in BAD else ''}" for i, v in enumerate(values)]
                self.history.append(values)
                report_requests(self.requests)
            return [ValueError("x") if i == self.fail else v for i, v in enumerate(values)]

        return Operation.per_invocation("loop", body, requests=Requests.opaque())


def _verifier(seen: list[str]) -> Operation:
    def assess(text: str, target: Any) -> Assessment:
        seen.append(text)
        return Assessment("BAD" not in text, target)

    return Operation.verifier("assess", assess, {"text": TEXT, "target": DATUM}, Requests.none())


def program() -> list[dict[str, str]]:
    c = Checks("SP-16")
    g, (a, b, d) = graph(*TEXTS)

    def final_candidates() -> tuple[bool, str]:
        loop, seen = _Loop(), []
        r = run(g, loop.operation(), _verifier(seen))
        last, earlier = loop.history[-1], {v for pass_ in loop.history[:-1] for v in pass_}
        quals = [r.outcome(t).qualification for t in (a, b, d)]
        return (
            len(loop.history) == 3
            and seen == last
            and not set(seen) & earlier
            and [r.value(t) for t in (a, b, d)] == last
            and quals == ["met", "met", "not_met"]
            and set(r.protected) == {a, b}
            and r.consistent,
            f"passes={len(loop.history)}, verifier calls={len(seen)}, final seen={seen == last}, qualifications={quals}",
        )

    c.guard("E16.1", final_candidates)

    def failed_target() -> tuple[bool, str]:
        loop, seen = _Loop(fail=1), []
        r = run(g, loop.operation(), _verifier(seen))
        o = r.outcome(b)
        content = dict(r.content())
        return (
            (o.terminal, o.reason, o.qualification) == ("failed", "raised", "not_met")
            and b not in r.protected
            and content[b] is None
            and set(r.protected) == {a}
            and seen == [loop.history[-1][0], loop.history[-1][2]]
            and r.outcome(a).qualification == "met",
            f"target 2 terminal={o.terminal}, qualification={o.qualification}, protected={len(r.protected)}, "
            f"content is None={content[b] is None}, verifier calls={len(seen)}",
        )

    c.guard("E16.2", failed_target)

    def exact_requests() -> tuple[bool, str]:
        r = run(g, _Loop(requests=2).operation(), _verifier([]))
        observed = {name: str(count) for name, count in r.summary().observed}
        return observed["requests"] == "exact 6", f"requests={observed['requests']}, total={observed['total']}"

    c.guard("E16.3", exact_requests)
    return c.lines
