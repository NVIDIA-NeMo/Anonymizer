# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-2 verification, Protected release and withholding (E2.1-E2.3)."""

from __future__ import annotations

from tests.spec_programs._builder import NAME, Checks, Recorder, graph, run


def program() -> list[dict[str, str]]:
    c = Checks("SP-2")
    g, (a,) = graph(f"{NAME} met Bob.")
    rec = Recorder()
    r = run(g, rec.transform(), rec.verifier())
    o = r.outcome(a)
    c.check(
        "E2.1",
        (rec.count("transform"), rec.count("verifier")) == (1, 1)
        and o.qualification == "met"
        and r.protected == {a: "[NAME] met Bob."},
        f"calls=({rec.count('transform')}, {rec.count('verifier')}), qualification={o.qualification}, "
        f"protected={len(r.protected)}",
    )  # documented example
    rec2 = Recorder()
    r2 = run(g, rec2.transform(fn=lambda t: t), rec2.verifier())
    o2 = r2.outcome(a)
    c.check(
        "E2.2",
        o2.qualification != "met" and len(r2.protected) == 0 and o2.terminal == "success",
        f"qualification={o2.qualification}, protected={len(r2.protected)}, terminal={o2.terminal}",
    )
    order = [op for op, _, _ in rec.calls]
    received = [t for op, t, _ in rec.calls if op == "verifier"]
    c.check(
        "E2.3",
        order == ["transform", "verifier"] and received == ["[NAME] met Bob."],
        f"order={order}, verifier received transform output={received == ['[NAME] met Bob.']}",
    )
    return c.lines
