# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-13 identity and correlation refusals (E13.1-E13.5)."""

from __future__ import annotations

import copy
import dataclasses

from anonymizer.engine.graph_core.authoring import GraphScope, Mention, Operation, ProtectionGraph, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from tests.spec_programs._builder import Checks, Recorder, rejected, run


def program() -> list[dict[str, str]]:
    c = Checks("SP-13")
    scope = GraphScope()
    x = scope.datum("same text")
    try:
        setattr(x, "_id", "forged")
        mutable = True
    except dataclasses.FrozenInstanceError:
        mutable = False
    c.check("E13.1", not mutable and copy.copy(x) == x and copy.deepcopy(x) == x, f"assignable={mutable}")  # D.REQ5.s1
    rec = Recorder()
    other = GraphScope()
    code = rejected(lambda: run(ProtectionGraph(other, (x,)), rec.transform()))
    c.check("E13.2", code == "foreign_datum" and rec.count() == 0, f"rejection={code}, callbacks={rec.count()}")
    y, z = scope.datum("same text"), scope.datum("same text")
    r1 = run(ProtectionGraph(scope, (x, y, z)), rec.transform(fn=str.upper))
    r2 = run(ProtectionGraph(scope, (z, x, y)), rec.transform(fn=str.upper))
    same = all((r1.value(d), r1.outcome(d)) == (r2.value(d), r2.outcome(d)) for d in (x, y, z))
    c.check("E13.3", same, "outputs and terminals equal by identity across declaration orders")  # D.REQ5.s4, D.REQ6.s5
    g = ProtectionGraph(scope, (x,))
    p = prepare(g, Workflow("w", (Operation("t", lambda text: text.upper()),)))
    a, b = protect(p), protect(p)
    refs_a, refs_b = {t.ref for t in a.attempts}, {t.ref for t in b.attempts}
    c.check(
        "E13.4",
        bool(refs_a) and bool(refs_b) and not refs_a & refs_b and a.outcome(x).terminal == "success",
        f"runtime refs shared={len(refs_a & refs_b)}",
    )  # D.REQ5.s2
    s2 = GraphScope()
    d1, d2 = s2.datum("Alice"), s2.datum("Alice")
    m1, m2 = Mention(d1, 0, 5, "first_name"), Mention(d2, 0, 5, "first_name")
    c.check("E13.5", d1 != d2 and m1 != m2, f"distinct identities={d1 != d2}")  # D.REQ5.s4, D.REQ4.s2
    return c.lines
