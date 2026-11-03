# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-23 final-candidate access and release cause (E23.1-E23.5)."""

from __future__ import annotations

from anonymizer.engine.graph_core.authoring import TEXT, Operation
from tests.spec_programs._builder import Checks, Recorder, graph, run
from tests.spec_programs.sp18_release import _run


def program() -> list[dict[str, str]]:
    c = Checks("SP-23")
    g, (t,) = graph("Alice")

    def boom(text: str) -> str:
        raise ValueError("b")

    # E23.1 A then B, B raises: the target is absent with B's reason; A's text is read only as A's.
    r1 = run(g, Operation("A", lambda text: "a-text"), Operation("B", boom))
    o1 = r1.outcome(t)
    ok1 = not o1.available and o1.reason == "raised" and r1.content() == ((t, None),)
    c.check("E23.1", ok1 and r1.value(t, "text", producer="A") == "a-text", f"available={o1.available}")

    # E23.2 a consumer of B's port is blocked when B raises.
    consumer = Operation("C", lambda text: text, {"text": TEXT})
    ledger = dict((op, s) for _, op, s in run(g, Operation("A", str), Operation("B", boom), consumer).ledger())
    c.check("E23.2", ledger["C"] == "blocked", f"C={ledger['C']}")

    # E23.3 B's privacy check fails: B's text stays available, not met.
    rec = Recorder()
    r3 = run(g, rec.transform("A", str), rec.transform("B", lambda text: "b-text"), rec.verifier("check", "b-text"))
    o3 = r3.outcome(t)
    c.check("E23.3", o3.available and o3.qualification == "not_met" and r3.content() == ((t, "b-text"),), f"available={o3.available}, {o3.qualification}")

    # E23.4 group {A, B}, B fails: A stays met, is withheld and names B; B carries its own reason.
    r4, (a, b) = _run(("a", "bad"), groups=((0, 1),))
    oa, ob = r4.outcome(a), r4.outcome(b)
    ok4 = oa.qualification == "met" and not oa.protected and oa.withheld_by == (b,) and ob.reason == "failed"
    c.check("E23.4", ok4, f"A={oa.qualification},{oa.withheld_by == (b,)}; B={ob.reason}")

    # E23.5 a dependent C on a failed B names B.
    r5, (dep, on) = _run(("c", "bad"), deps=((0, 1),))
    c.check("E23.5", r5.outcome(dep).withheld_by == (on,), "dependent names its cause")
    return c.lines
