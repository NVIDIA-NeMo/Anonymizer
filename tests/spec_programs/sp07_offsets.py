# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-7 source chains and exact offsets (E7.1-E7.5)."""

from __future__ import annotations

from anonymizer.engine.graph_core.authoring import (
    ContextScope,
    ContextView,
    GraphScope,
    Mention,
    Operation,
    ProtectionGraph,
)
from tests.spec_programs._builder import Checks, Recorder, rejected, run

TEXT = "Alice met Bob."


def program() -> list[dict[str, str]]:
    c = Checks("SP-7")
    scope = GraphScope()
    t = scope.datum(TEXT)
    seen: list[tuple] = []
    op = Operation("op", lambda text, mentions: seen.append(mentions) or text, {"text": "text", "mentions": "mentions"})
    run(ProtectionGraph(scope, (t,), mentions=(Mention(t, 0, 5, "first_name", "Alice"),)), op)
    m = seen[0][0] if seen and seen[0] else None
    c.check(
        "E7.1",
        m is not None and (m.start, m.end) == (0, 5) and TEXT[m.start : m.end] == "Alice",
        f"mention=[{m and m.start}, {m and m.end})",
    )  # D.REQ10.s1
    rec = Recorder()
    code = rejected(
        lambda: run(ProtectionGraph(scope, (t,), mentions=(Mention(t, 1, 6, "first_name", "Alice"),)), rec.transform())
    )
    c.check("E7.2", code is not None and rec.count() == 0, f"rejection={code}, callbacks={rec.count()}")
    overlap = (Mention(t, 0, 5, "first_name"), Mention(t, 3, 9, "other"))
    code = rejected(lambda: run(ProtectionGraph(scope, (t,), mentions=overlap), rec.transform()))
    c.check("E7.3", code is not None and rec.count() == 0, f"rejection={code}, callbacks={rec.count()}")  # D.REQ10.s3
    views = ContextScope(t, (ContextView(t, t, 0, 9), ContextView(t, t, 6, 14)))
    g4 = ProtectionGraph(scope, (t,), mentions=(Mention(t, 0, 5, "first_name"),), context=(views,))
    r4 = run(g4, rec.transform())
    c.check("E7.4", r4.outcome(t).terminal == "success", f"accepted, terminal={r4.outcome(t).terminal}")  # D.REQ10.s4
    ctx = scope.datum("Bob met Alice.")
    bad = ProtectionGraph  # the output span's endpoints come from the context artifact, not the target
    before = rec.count()
    code = rejected(
        lambda: run(
            bad(scope, (t,), mentions=(Mention(ctx, 8, 13, "first_name"),), context=(ContextScope(t, (ctx,)),)),
            rec.transform(),
        )
    )
    c.check("E7.5", code is not None and rec.count() == before, f"rejection={code}")  # D.REQ10.s2
    return c.lines
