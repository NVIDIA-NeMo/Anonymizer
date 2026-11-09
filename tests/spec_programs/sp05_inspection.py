# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-5 bounded, content-free inspection (E5.1-E5.6)."""

from __future__ import annotations

from anonymizer.engine.graph_core.authoring import Operation, Requests, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.engine.graph_core.observe import report_requests
from tests.spec_programs._builder import Checks, Recorder, graph, run

CANARY_IN, CANARY_OUT = "CANARYxIN7731", "CANARYxOUT4410"


def _prepared(text: str, transform=None):
    g, (a,) = graph(text)
    rec = Recorder()
    t = transform or rec.transform(fn=lambda s: s.replace(CANARY_IN, CANARY_OUT))
    return prepare(g, Workflow("w", (t, rec.verifier(needle=CANARY_IN)))), a


def program() -> list[dict[str, str]]:
    c = Checks("SP-5")
    p, a = _prepared(f"Hello {CANARY_IN}.")
    before = p.summary()
    c.check(
        "E5.1",
        before.operations == ("transform", "verifier") and before.terminals is None,
        f"operations={list(before.operations)}, terminals={before.terminals}",
    )
    r = protect(p)
    text = r.summary().render()
    c.check(
        "E5.2",
        dict(r.summary().terminals or ()) == {"success": 1}
        and CANARY_IN not in text
        and CANARY_OUT not in text
        and dict(r.content())[a] == f"Hello {CANARY_OUT}.",
        f"terminals={r.summary().terminals}, canary in rendering={CANARY_IN in text or CANARY_OUT in text}",
    )

    def boom(text: str) -> str:
        raise ValueError(f"cannot handle {text}")

    p3, _ = _prepared(f"Hello {CANARY_IN}.", Operation("transform", boom))
    s3 = protect(p3).summary()
    text3 = s3.render()
    c.check(
        "E5.3",
        dict(s3.terminals or ()) == {"failed": 1}
        and dict(s3.reasons) == {"raised": 1}
        and CANARY_IN not in text3
        and CANARY_OUT not in text3,
        f"terminals={s3.terminals}, reasons={s3.reasons}",
    )
    small = protect(_prepared("x" * 10)[0]).summary().render()
    large = protect(_prepared("x" * 100_000)[0]).summary().render()
    c.check("E5.4", len(large) - len(small) <= 64, f"growth={len(large) - len(small)} chars")
    obs = dict(r.summary().observed)
    planned = before.planned_requests
    known_zero = all(v.value == 0 or v.value is None for v in obs.values()) and set(obs) >= {
        "binding",
        "correction",
        "retry",
        "total",
    }
    c.check(
        "E5.5",
        planned.value in (0, None) and known_zero and r.summary().planned_requests == planned,
        f"planned={planned}, observed={ {k: str(v) for k, v in obs.items()} }",
    )

    def budget(first: int, second: int | None) -> tuple[str | None, str]:
        """Planned 10 (two steps declaring exact 5); ``None`` leaves the second step unreported."""

        def step(n: int | None):
            def call(text: str) -> str:
                if n is not None:
                    report_requests(n)
                return text

            return call

        g, _ = graph("x")
        ops = (
            Operation("a", step(first), requests=Requests.exact(5)),
            Operation("b", step(second), requests=Requests.exact(5)),
        )
        s = run(g, *ops).summary()
        return s.budget, s.render()

    verdicts = [budget(12, None), budget(7, None), budget(7, 0)]
    c.check(
        "E5.6",
        [v for v, _ in verdicts] == ["exceeded", "unknown", "within"]
        and all(f"budget: {v}" in text for v, text in verdicts)
        and p.summary().budget is None,
        f"verdicts={[v for v, _ in verdicts]}",
    )
    return c.lines
