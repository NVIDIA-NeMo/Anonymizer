# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A verifier declares exactly ``{"assessment": ASSESSMENT}``; any other layout is rejected at construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anonymizer.engine.graph_core.authoring import (
    ASSESSMENT,
    TEXT,
    Assessment,
    GraphScope,
    Operation,
    ProtectionGraph,
    Requests,
    Workflow,
)
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.engine.graph_core.kernel import GraphRejection

DOC = Path(__file__).resolve().parents[2] / "docs" / "concepts" / "graph-sdk" / "reference.md"


def _operation(outputs: dict[str, str], fn: Any = lambda text: {}) -> Operation:
    return Operation("odd", fn, {"text": TEXT}, outputs, Requests.none(), checks=("c",), subject=("text",))


@pytest.mark.parametrize(
    "outputs",
    [
        {"evidence": ASSESSMENT, "assessment": TEXT},  # the crash layout: the "assessment" port holds text
        {"assessment": ASSESSMENT, "text": TEXT},  # an extra port beside the assessment
        {"evidence": ASSESSMENT},  # the only assessment port is not named "assessment"
        {"assessment": TEXT},  # named "assessment" but not assessment-kind
    ],
    ids=["evidence-and-text-assessment", "extra-port", "evidence-only", "text-assessment"],
)
def test_nonstandard_verifier_layout_is_rejected_at_construction(outputs: dict[str, str]) -> None:
    with pytest.raises(GraphRejection) as info:
        _operation(outputs)
    assert info.value.code == "verifier_ports"
    assert not isinstance(info.value, TypeError)
    assert "odd" in info.value.detail and all(port in info.value.detail for port in outputs)


@pytest.mark.parametrize("requires", [(), ("c",)], ids=["advisory", "required"])
@pytest.mark.parametrize("build", ["verifier", "hand-built"])
def test_standard_verifier_layout_constructs_and_runs(build: str, requires: tuple[str, ...]) -> None:
    scope = GraphScope()
    a = scope.datum("hello")

    def assess(text: str) -> Assessment:
        return Assessment(True, a)

    if build == "verifier":
        op = Operation.verifier("v", assess, {"text": TEXT}, Requests.none(), checks=("c",), subject=("text",))
    else:
        op = Operation("v", assess, {"text": TEXT}, {"assessment": ASSESSMENT}, Requests.none(), checks=("c",),
                       subject=("text",))  # fmt: skip
    assert op.is_verifier
    result = protect(prepare(ProtectionGraph(scope, (a,)), Workflow("w", (op,), requires=requires)))
    outcome = result.outcome(a)
    assert (outcome.terminal, outcome.qualification) == ("success", "met" if requires else "not_assessed")
    assert result.value(a, "assessment", producer="v") == Assessment(True, a)


def test_rejection_code_is_documented() -> None:
    assert 'GraphRejection("verifier_ports")' in DOC.read_text()
