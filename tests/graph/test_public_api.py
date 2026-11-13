# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import anonymizer.graph as graph

PUBLIC_NAMES = {
    "Assessment",
    "CancelToken",
    "ContextScope",
    "ContextSource",
    "ContextView",
    "Datum",
    "Dependency",
    "GraphRejection",
    "GraphScope",
    "InspectionSummary",
    "Limits",
    "Mention",
    "Operation",
    "OutputGroup",
    "PreparedProtection",
    "ProtectionGraph",
    "ProtectionResult",
    "Reason",
    "Requests",
    "TargetOutcome",
    "Workflow",
    "prepare",
    "protect",
    "register_kind",
}

ALLOWED_MODULES = {
    "anonymizer",
    "anonymizer.engine",
    "anonymizer.engine.constants",
    "anonymizer.engine.graph_core",
    "anonymizer.graph",
}

_PROBE = """
import json, sys
before = set(sys.modules)
import anonymizer.graph
print(json.dumps(sorted(set(sys.modules) - before)))
"""


def test_all_is_exactly_the_public_names() -> None:
    assert set(graph.__all__) == PUBLIC_NAMES
    assert len(graph.__all__) == len(PUBLIC_NAMES)
    assert all(hasattr(graph, name) for name in graph.__all__)


def test_import_loads_only_allowed_modules() -> None:
    out = subprocess.run([sys.executable, "-I", "-c", _PROBE], check=True, capture_output=True, text=True).stdout
    loaded = set(json.loads(out))
    stdlib = set(sys.stdlib_module_names)
    unexpected = sorted(
        name
        for name in loaded
        if name.split(".")[0] not in stdlib
        and name not in ALLOWED_MODULES
        and not name.startswith("anonymizer.engine.graph_core.")
    )
    assert unexpected == []


def test_graph_core_imports_nothing_else_from_anonymizer() -> None:
    package = Path(graph.__file__).parent.parent / "engine" / "graph_core"
    outside = []
    for path in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            outside += [
                f"{path.name}: {name}"
                for name in names
                if name.split(".")[0] == "anonymizer"
                and name != "anonymizer.engine.graph_core"
                and not name.startswith("anonymizer.engine.graph_core.")
            ]
    # The measurement hook is imported lazily, only while a measurement session is loaded.
    assert outside == ["execution.py: anonymizer.measurement.recorders"]


def test_required_check_must_be_answered_by_a_final_text_verifier() -> None:
    calls: list[str] = []

    def op(name: str, old: str) -> graph.Operation:
        return graph.Operation(name, lambda text: calls.append(name) or text.replace(old, "[NAME]"))

    def verifier(name: str, check: str, verdict: bool | None = None) -> graph.Operation:
        def assess(text: str, target: graph.Datum) -> graph.Assessment:
            calls.append(name)
            ok = ("Alice" not in text and "Smith" not in text) if verdict is None else verdict
            return graph.Assessment(ok, target, "" if ok else "name_present")

        return graph.Operation.verifier(name, assess, {"text": "text", "target": "datum"}, checks=(check,))

    def attempt(steps: tuple, requires: tuple[str, ...]) -> tuple[str, str]:
        scope = graph.GraphScope()
        target = scope.datum("Alice Smith wrote this.")
        try:
            prepared = graph.prepare(
                graph.ProtectionGraph(scope, (target,)), graph.Workflow("w", steps, requires=requires)
            )
        except graph.GraphRejection as exc:
            return exc.code, exc.detail
        return "admitted", graph.protect(prepared).outcome(target).qualification

    draft, revise = op("draft", "Alice"), op("revise", "Smith")
    early, final = verifier("check-draft", "draft_clean"), verifier("check-final", "final_clean")
    rejected = {
        "draft only": ((draft, early, revise), ("draft_clean",), "draft_clean"),
        "both, requires both": ((draft, early, revise, final), ("draft_clean", "final_clean"), "draft_clean"),
        "passing draft": (
            (draft, verifier("check-draft", "draft_clean", True), revise, final),
            ("draft_clean", "final_clean"),
            "draft_clean",
        ),
        "undeclared": ((draft, revise, final), ("final_clean", "typo_check"), "typo_check"),
    }
    for label, (steps, requires, unanswered) in rejected.items():
        assert attempt(steps, requires) == ("required_check_not_final", unanswered), label
    assert calls == []

    shared = (draft, verifier("check-draft", "clean"), revise, verifier("check-final", "clean"))
    assert attempt((draft, revise, final), ("final_clean",)) == ("admitted", "met")
    assert attempt((draft, early, revise, final), ("final_clean",)) == ("admitted", "met")
    assert attempt(shared, ("clean",)) == ("admitted", "met")
    assert attempt((draft, early, revise), ())[0] == "admitted"

    def nested(*steps: graph.Operation, requires: tuple[str, ...]) -> graph.Workflow:
        return graph.Workflow("inner", steps, requires=requires)

    assert attempt((draft, nested(revise, final, requires=("final_clean",))), ()) == ("admitted", "met")
    assert attempt((nested(draft, early, requires=("draft_clean",)), revise, final), ("final_clean",)) == (
        "required_check_not_final",
        "draft_clean",
    )
    failing = verifier("check-final", "final_clean", False)
    assert attempt((draft, nested(revise, failing, requires=("final_clean",))), ()) == ("admitted", "not_met")
