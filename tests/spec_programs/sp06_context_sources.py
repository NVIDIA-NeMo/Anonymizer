# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-6 context binding: two sources, sizes, counts (E6.1-E6.5)."""

from __future__ import annotations

from anonymizer.engine.graph_core.authoring import ContextScope, ContextSource, GraphScope, Operation, ProtectionGraph
from tests.spec_programs._builder import Checks, run


def _two(a_items, b_provider, a_max=1):
    scope = GraphScope()
    t = scope.datum("Alice met Bob.")
    seen: list[tuple] = []

    def op(text: str, context: tuple) -> str:
        seen.append(context)
        return text

    a = ContextSource("A", lambda: a_items, max_items=a_max)
    b = ContextSource("B", b_provider, max_items=2)
    g = ProtectionGraph(scope, (t,), context=(ContextScope(t, (a, b)),))
    r = run(g, Operation("op", op, {"text": "text", "context": "context"}))
    return r, r.outcome(t), seen


def program() -> list[dict[str, str]]:
    c = Checks("SP-6")
    r, o, seen = _two(["a1"], lambda: ["b1", "b2"])
    sources = [i.source for i in seen[0]] if seen else []
    c.check(
        "E6.1",
        sources == ["A", "B", "B"] and [x.source for x in r.retrievals] == ["A", "B"] and len(r.attempts) == 1,
        f"items by source={sources}, retrievals={len(r.retrievals)}, attempts={len(r.attempts)}",
    )

    def down() -> list[str]:
        raise TimeoutError("B down")

    r, o, seen = _two(["a1"], down)
    status = {x.source: x.status for x in r.retrievals}
    c.check(
        "E6.2",
        status["B"] == "failed" and not seen and o.terminal != "success" and o.qualification != "met",
        f"retrievals={status}, operation ran={bool(seen)}, terminal={o.terminal}",
    )
    big = "x" * (1024 * 1024 + 1)
    r, o, seen = _two([big], list)
    c.check(
        "E6.3",
        o.terminal == "success" and seen and seen[0][0].text == big,
        f"terminal={o.terminal}, delivered bytes={len(seen[0][0].text) if seen else None}",
    )
    r, o, seen = _two(["a1", "a2"], list, a_max=5)
    ra = r.retrievals[0]
    c.check(
        "E6.4",
        (ra.declared_max_items, ra.observed_items) == (5, 2),
        f"declared={ra.declared_max_items}, observed={ra.observed_items}",
    )
    r, o, seen = _two(["Zoë Ågren"], list)
    c.check("E6.5", r.retrievals[0].observed_bytes == 11, f"bytes={r.retrievals[0].observed_bytes}")
    return c.lines
