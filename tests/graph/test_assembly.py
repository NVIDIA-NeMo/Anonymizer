# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assembly ordering: composition, reconstruction and region assembly follow caller order, reject conflicts and
unavailable inputs, and do not depend on declaration order."""

from __future__ import annotations

from typing import Any

import pytest

from anonymizer.engine.graph_core.authoring import GraphScope, Operation, ProtectionGraph
from anonymizer.engine.graph_core.kernel import GraphRejection
from tests.graph.helpers import run

OPENING, FINDINGS, RECOMMENDATION = "Opening. ", "Findings. ", "Recommendation. "
BRIEF = "RECOMMENDATION. FINDINGS. "
REPORT = "OPENING. FINDINGS. RECOMMENDATION. "


def _sections(order: tuple[int, ...] = (0, 1, 2), fail: str = "") -> tuple[Any, list]:
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


def _code(fn: Any) -> str:
    with pytest.raises(GraphRejection) as info:
        fn()
    return info.value.code


def test_compose_and_reconstruct_follow_requested_and_source_order() -> None:
    r, (opening, findings, recommendation) = _sections()
    source = {opening: 0, findings: 1, recommendation: 2}
    assert r.compose([recommendation, findings]) == BRIEF
    assert r.reconstruct([recommendation, opening, findings], source) == REPORT
    assert _code(lambda: r.reconstruct([opening, recommendation], source)) == "missing_source_part"
    assert (
        _code(lambda: r.reconstruct([opening, findings, recommendation], {**source, findings: 0}))
        == "conflicting_position"
    )


def test_unavailable_section_is_rejected() -> None:
    failed, (_, findings, recommendation) = _sections(fail=FINDINGS)
    assert _code(lambda: failed.compose([recommendation, findings])) == "unavailable_input"
    assert (
        _code(lambda: failed.assemble_regions(OPENING + FINDINGS, [(len(OPENING), len(OPENING) + 10, findings)]))
        == "unavailable_input"
    )


def test_declaration_order_does_not_change_assembly() -> None:
    r, (opening, findings, recommendation) = _sections(order=(2, 0, 1))
    assert r.compose([recommendation, findings]) == BRIEF
    assert r.reconstruct([findings, recommendation, opening], {recommendation: 2, opening: 0, findings: 1}) == REPORT


def test_assemble_regions_replaces_spans_and_rejects_bad_regions() -> None:
    scope = GraphScope()
    head, tail = scope.datum("h" * 100), scope.datum("t" * 100)
    regions = run(
        ProtectionGraph(scope, (head, tail)),
        Operation("op", lambda text: text[:60] if text[0] == "h" else text + "x" * 30),
    )
    doc = "h" * 100 + "t" * 100
    assert regions.assemble_regions(doc, [(100, 200, tail), (0, 100, head)]) == "h" * 60 + "t" * 100 + "x" * 30
    assert regions.assemble_regions("<" + doc + ">", [(101, 201, tail)]) == "<" + "h" * 100 + "t" * 100 + "x" * 30 + ">"
    assert _code(lambda: regions.assemble_regions(doc, [(0, 101, head), (100, 200, tail)])) == "overlapping_region"
    assert _code(lambda: regions.assemble_regions(doc, [(150, 201, tail)])) == "region_out_of_bounds"
