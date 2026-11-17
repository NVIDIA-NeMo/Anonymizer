# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-33 records: rejections at construction, the per-target seed, phase-scoped grouping, no records (E33.1-E33.4)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from anonymizer.engine.graph_core.authoring import GraphScope, Operation, ProtectionGraph, Record, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from tests.spec_programs._builder import NAME, Checks, Recorder, rejected


def _targets(*texts: str) -> tuple[GraphScope, tuple]:
    scope = GraphScope()
    return scope, tuple(scope.datum(t) for t in texts)


def program() -> list[dict[str, str]]:
    c = Checks("SP-33")

    # E33.1 every invalid record is rejected while the graph is built, before any callback runs.
    def e1() -> tuple[bool, str]:
        rec = Recorder()
        scope, (a, b) = _targets(f"{NAME} one", "Bob two")
        builds: dict[str, Callable[[], Any]] = {
            "invalid_record": lambda: ProtectionGraph(scope, (a, b), records=((a,),)),  # ty: ignore[invalid-argument-type]
            "empty_record": lambda: scope.record(()),
            "foreign_record": lambda: ProtectionGraph(scope, (a, b), records=(GraphScope().record((a,)),)),
            "foreign_datum": lambda: ProtectionGraph(scope, (a,), records=(scope.record((GraphScope().datum("x"),)),)),
            "unknown_reference": lambda: ProtectionGraph(scope, (a,), records=(scope.record((b,)),)),
            "record_overlap": lambda: ProtectionGraph(
                scope, (a, b), records=(scope.record((a, b)), scope.record((a,)))
            ),
        }
        codes = {code: rejected(build) for code, build in builds.items()}
        ok = all(code == got for code, got in codes.items()) and rec.count() == 0
        return ok, f"codes={sorted(codes.values(), key=str)} calls={rec.count()}"

    c.guard("E33.1", e1)

    # E33.2 a two-target Record plus a singleton: each target's seed is its Record.
    def e2() -> tuple[bool, str]:
        scope, (a, b, s) = _targets(f"{NAME} one", "Bob two", "Carol three")
        g = ProtectionGraph(scope, (a, b, s), records=(scope.record((a, b)),))
        seen: dict[Any, Record] = {}

        def look(text: str, target: Any, record: Record) -> str:
            seen[target] = record
            return text

        protect(
            prepare(
                g, Workflow("w", (Operation("look", look, {"text": "text", "target": "datum", "record": "record"}),))
            )
        )
        ok = seen[a] == seen[b] != seen[s] and seen[s].members == (s,) and len(g.all_records) == 2
        return ok, f"records={len(set(seen.values()))}"

    c.guard("E33.2", e2)

    # E33.3 a phase-scoped operation groups the targets it receives by the record seed.
    def e3() -> tuple[bool, str]:
        scope, (a, b, s) = _targets(f"{NAME} one", "Bob two", f"{NAME} three")
        g = ProtectionGraph(scope, (a, b, s), records=(scope.record((s, a)),))
        sizes: list[int] = []

        def fold(text: tuple[str, ...], record: tuple[Record, ...]) -> list[str]:
            groups: dict[Record, int] = {}
            for r in record:
                groups[r] = groups.get(r, 0) + 1
            sizes.extend(groups.values())
            return [t.replace(NAME, "[NAME]") for t in text]

        r = protect(
            prepare(g, Workflow("w", (Operation.phase_scoped("fold", fold, {"text": "text", "record": "record"}),)))
        )
        ok = sizes == [2, 1] and all(NAME not in (t or NAME) for _, t in r.content())
        return ok, f"sizes={sizes}"

    c.guard("E33.3", e3)

    # E33.4 with no records declared, content and outcomes are unchanged.
    def e4() -> tuple[bool, str]:
        rec = Recorder()
        scope, ts = _targets(f"{NAME} one", "Bob two")
        g = ProtectionGraph(scope, ts)
        w = Workflow("w", (rec.transform(), rec.verifier()), requires=("verifier",))
        r = protect(prepare(g, w))
        content = [t for _, t in r.content()]
        met = [(o.qualification, o.protected) for o in map(r.outcome, ts)]
        ok = content == ["[NAME] one", "Bob two"] and met == [("met", True)] * 2 and rec.count() == 4
        ok = ok and g.records == () and [x.members for x in g.all_records] == [(t,) for t in ts]
        return ok, f"met={met} calls={rec.count()}"

    c.guard("E33.4", e4)
    return c.lines
