# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-34 Session: mode table, reuse across calls, no ``session=`` unchanged, resolution path (E34.1-E34.4)."""

from __future__ import annotations

from typing import Any

from anonymizer.engine.graph_core.authoring import Operation, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.engine.graph_core.session import Decision, Session, SessionChanges, SessionMode, SessionView, SubjectKey
from tests.spec_programs._builder import NAME, Checks, Recorder, graph, rejected

SLOT = (SubjectKey("subject-1"), "first_name", "alice")


def fold(recorder: Recorder) -> Operation:
    """Mints a substitute from a per-call counter unless the Session view already holds one."""

    def run(text: tuple[str, ...], session: tuple[SessionView, ...]) -> list[dict[str, Any]]:
        recorder.calls.append(("fold", text, 0))
        minted = iter(range(1, 1_000))
        out = []
        for t, view in zip(text, session):
            value = view.lookup(SLOT) or f"Name{next(minted)}"
            out.append({"text": t.replace(NAME, value), "decisions": (Decision(SLOT, value),)})
        return out

    return Operation.phase_scoped(
        "fold", run, {"text": "text", "session": "session-view"}, {"text": "text", "decisions": "decisions"}
    )


def program() -> list[dict[str, str]]:
    c = Checks("SP-34")

    # E34.1 mode table: off/off and stable prepare; matching alone fails construction; on/on fails at prepare.
    def e1() -> tuple[bool, str]:
        g, _ = graph(f"{NAME} wrote this.")
        w = Workflow("w", (fold(Recorder()),))
        codes = [
            prepare(g, w, session=Session()).session_mode == SessionMode(),
            prepare(g, w, session=Session(stable_replacements=True)).session_mode.stable_replacements,
            rejected(lambda: Session(identity_matching=True)) == "session_mode_invalid",
            rejected(lambda: prepare(g, w, session=Session(stable_replacements=True, identity_matching=True)))
            == "session_mode_unsupported",
        ]
        return all(codes), f"rows={codes}"

    c.guard("E34.1", e1)

    # E34.2 one stable Session across two protect calls reuses the first call's assignment.
    def e2() -> tuple[bool, str]:
        s = Session(stable_replacements=True)
        texts, counts = [], []
        for _ in range(2):
            g, _ = graph(f"{NAME} wrote this.")
            r = protect(prepare(g, Workflow("w", (fold(Recorder()),)), session=s), session=s)
            texts += [t for _, t in r.content()]
            counts.append(r.session_changes)
        ok = texts[0] == texts[1] and counts == [SessionChanges(1), SessionChanges(0, 1)] and s.revision == 1
        return ok, f"same={texts[0] == texts[1]} revision={s.revision}"

    c.guard("E34.2", e2)

    # E34.3 protect without session= equals protect with an explicit off/off Session.
    def e3() -> tuple[bool, str]:
        rec = Recorder()
        g, _ = graph(f"{NAME} one", "Bob two")
        p = prepare(g, Workflow("w", (rec.transform(), rec.verifier()), requires=("verifier",)))
        a, b = protect(p), protect(p, session=Session())
        ok = a.content() == b.content() and a.outcomes == b.outcomes and a.ledger() == b.ledger()
        return ok and a.session_changes == SessionChanges(), f"equal={ok}"

    c.guard("E34.3", e3)

    # E34.4 a stable Session with no decisions output is rejected at prepare; no callback runs.
    def e4() -> tuple[bool, str]:
        rec = Recorder()
        g, _ = graph(f"{NAME} wrote this.")
        code = rejected(
            lambda: prepare(g, Workflow("w", (rec.transform(),)), session=Session(stable_replacements=True))
        )
        return code == "session_without_resolution" and rec.count() == 0, f"code={code} calls={rec.count()}"

    c.guard("E34.4", e4)
    return c.lines
