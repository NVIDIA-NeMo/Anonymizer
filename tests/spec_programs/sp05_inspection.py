# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-5 bounded, content-free inspection (E5.1-E5.5)."""

from __future__ import annotations

from anonymizer.engine.graph_core.authoring import Operation, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.measurement import MeasurementCollector, measurement_session
from tests.spec_programs._builder import Checks, Recorder, graph

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
    with measurement_session(MeasurementCollector(keep_records=True)) as collector:
        measured = protect(_prepared(f"Hello {CANARY_IN}.")[0])
    stages = [(m["stage"], m["status"], m["schema_version"]) for m in collector.records]
    leaked = any(CANARY_IN in repr(m) or CANARY_OUT in repr(m) or "transform" in repr(m) for m in collector.records)
    c.check(
        "E5.6",
        sorted(s for s, _, _ in stages) == ["graph_prepare", "graph_protect", "graph_reconcile"]
        and all(st == "completed" and v is not None for _, st, v in stages)
        and all(m["terminal_counts"] == {"success": 1} for m in collector.records if m["stage"] != "graph_prepare")
        and not leaked,
        f"stages={stages}, content in records={leaked}",
    )
    plain = protect(_prepared(f"Hello {CANARY_IN}.")[0])
    same = plain.content()[0][1] == measured.content()[0][1] and plain.summary() == measured.summary()
    c.check("E5.7", same, f"measured and unmeasured results equal={same}")
    return c.lines
