# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-4 initial context binding from a caller-owned provider (E4.1-E4.4)."""

from __future__ import annotations

from anonymizer.engine.graph_core.authoring import ContextScope, ContextSource, GraphScope, Operation, ProtectionGraph
from tests.spec_programs._builder import Checks, run


def _bound(provider, max_items=2, max_bytes=None):
    scope = GraphScope()
    t = scope.datum("Alice met Bob.")
    seen: list[tuple] = []

    def op(text: str, context: tuple) -> str:
        seen.append(context)
        return text

    src = ContextSource("history", provider, max_items=max_items, max_bytes=max_bytes)
    g = ProtectionGraph(scope, (t,), context=(ContextScope(t, (src,)),))
    r = run(g, Operation("transform", op, {"text": "text", "context": "context"}))
    return r, r.outcome(t), seen


def program() -> list[dict[str, str]]:
    c = Checks("SP-4")
    r, o, seen = _bound(lambda: ["one", "two"])
    c.check(
        "E4.1",
        o.terminal == "success" and len(seen) == 1 and len(seen[0]) == 2 and len(r.retrievals) == 1,
        f"context items={len(seen[0]) if seen else None}, retrieval requests={len(r.retrievals)}",
    )
    r, o, seen = _bound(lambda: ["one", "two", "three"])
    c.check(
        "E4.2",
        not seen and o.terminal == "failed" and r.retrievals[0].status == "overflow",
        f"delivered={len(seen)}, terminal={o.terminal}, retrieval={r.retrievals[0].status}",
    )
    r, o, seen = _bound(lambda: ["x" * 40], max_bytes=10)
    c.check(
        "E4.3",
        not seen and o.terminal == "failed" and r.retrievals[0].status == "over_byte_ceiling",
        f"delivered={len(seen)}, retrieval={r.retrievals[0].status}",
    )

    def broken() -> list[str]:
        raise ConnectionError("provider down")

    r, o, seen = _bound(broken)
    c.check(
        "E4.4",
        not seen and o.terminal == "failed" and r.retrievals[0].status == "failed",
        f"delivered={len(seen)}, terminal={o.terminal}, reason={o.reason}",
    )
    return c.lines
