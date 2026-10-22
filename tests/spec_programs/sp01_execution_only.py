# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-1 direct authoring, execution-only (EXPECTED.md E1.1-E1.5)."""

from __future__ import annotations

import threading

from tests.spec_programs._builder import NAME, Checks, Recorder, graph, run


def program() -> list[dict[str, str]]:
    c = Checks("SP-1")
    text = f"{NAME} met Bob."
    g, (a, b) = graph(text, text)
    rec = Recorder()
    result = run(g, rec.transform())
    attempts = [x.target for x in result.attempts]
    c.check(
        "E1.1",
        rec.count() == 2 and attempts.count(a) == 1 and attempts.count(b) == 1,
        f"calls={rec.count()}, attempts per target={[attempts.count(a), attempts.count(b)]}",
    )  # documented example "Transform: 1" per target
    content = result.content()
    c.check(
        "E1.2",
        len(content) == 2 and all(t == "[NAME] met Bob." for _, t in content) and a != b,
        f"items={len(content)}, distinct identities={a != b}",
    )  # D.REQ4.s2, D.REQ5.s1, D.REQ5.s2
    quals = [result.outcome(d).qualification for d in (a, b)]
    c.check(
        "E1.3",
        quals == ["not_assessed"] * 2 and len(result.protected) == 0 and all(NAME not in t for _, t in content),
        f"qualifications={quals}, protected={len(result.protected)}",
    )  # D.REQ7.s3, D.REQ8.s6
    before = rec.count()
    again = (result.content(), result.protected, result.summary().render(), result.ledger())
    c.check("E1.4", rec.count() == before and again[0] == content, f"calls after re-projection={rec.count() - before}")
    me = threading.get_ident()
    c.check("E1.5", all(tid == me for _, _, tid in rec.calls), "callbacks on the calling thread")  # X.D.L3157.s3
    return c.lines
