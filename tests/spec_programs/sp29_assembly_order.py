# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-29 assembly ordering: composition, reconstruction, conflicts, unavailable inputs."""

from __future__ import annotations

from anonymizer.engine.graph_core.authoring import GraphScope, Operation, ProtectionGraph
from tests.spec_programs._builder import Checks, rejected, run

OPENING, FINDINGS, RECOMMENDATION = "Opening. ", "Findings. ", "Recommendation. "


def _sections(order: tuple[int, ...] = (0, 1, 2), fail: str = ""):
    """Three sections of one document, declared as targets in ``order``; the section ``fail`` fails in a later step."""
    scope = GraphScope()
    parts = [scope.datum(t) for t in (OPENING, FINDINGS, RECOMMENDATION)]

    def op(text: str) -> str:
        return text.upper()

    def late(text: str) -> str:
        if text == fail.upper():
            raise RuntimeError("down")
        return text

    r = run(ProtectionGraph(scope, tuple(parts[i] for i in order)), Operation("op", op), Operation("late", late))
    return r, parts


def program() -> list[dict[str, str]]:
    c = Checks("SP-29")
    r, (opening, findings, recommendation) = _sections()
    source = {opening: 0, findings: 1, recommendation: 2}
    brief = r.compose([recommendation, findings])
    report = r.reconstruct([recommendation, opening, findings], source)
    incomplete = rejected(lambda: r.reconstruct([opening, recommendation], source))
    c.check(
        "E29.1",
        brief == "RECOMMENDATION. FINDINGS. "
        and report == "OPENING. FINDINGS. RECOMMENDATION. "
        and incomplete == "missing_source_part",
        f"brief={brief!r}, report={report!r}, incomplete={incomplete}",
    )
    code = rejected(lambda: r.reconstruct([opening, findings, recommendation], {**source, findings: 0}))
    c.check("E29.2", code == "conflicting_position", f"code={code}")
    failed, (_, f_findings, f_recommendation) = _sections(fail=FINDINGS)
    code = rejected(lambda: failed.compose([f_recommendation, f_findings]))
    c.check("E29.3", code == "unavailable_input", f"code={code}")
    shuffled, (s_opening, s_findings, s_recommendation) = _sections(order=(2, 0, 1))
    same = (
        shuffled.compose([s_recommendation, s_findings]) == brief
        and shuffled.reconstruct(
            [s_findings, s_recommendation, s_opening], {s_recommendation: 2, s_opening: 0, s_findings: 1}
        )
        == report
    )
    c.check("E29.4", same, f"declaration order changed, results equal={same}")
    return c.lines
