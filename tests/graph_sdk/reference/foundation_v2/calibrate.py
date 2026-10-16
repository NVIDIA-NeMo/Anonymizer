# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run known answers, source-semantic mutants and two regeneration passes."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Callable, cast

from tests.graph_sdk.reference.foundation_v2.generate import encode_corpus, generate_cases

MUTATIONS = (
    ("cycle", "order.index(a) < order.index(b)", "order.index(a) != order.index(b)"),
    ("window", "source[start:end] == derived", "True"),
    ("adjacency", ">= min(end_a, end_b)", "> min(end_a, end_b)"),
    ("root_identity", "root_a != root_b or", "False or"),
    ("partition", "a == b or not a & b", "True"),
    ("selection", "if event.occurrence not in selected:", "if False:"),
    ("readiness", "if b == event.occurrence", "if False"),
    ("terminal_once", 'len(own) != 1 or own[0].kind != "start"', 'not own or own[0].kind != "start"'),
    (
        "map_failed",
        'if expansion == "failed":\n        return "blocked"',
        'if expansion == "failed":\n        return "success"',
    ),
    (
        "map_all",
        'all(category == "success" for _, category in terminals)',
        'any(category == "success" for _, category in terminals)',
    ),
    ("loop_final", 'return "success", len(outcomes) - 1', 'return "success", 0'),
    ("budget", "len(dispatched) + len(live) >= budget", "len(dispatched) + len(live) > budget"),
    ("reservation", "if event.request not in live:", "if False:"),
    ("first_terminal", "first = terminal_events[0]", "first = terminal_events[-1]"),
    ("unknown_usage", "if settlements else None", "if settlements else 0"),
    ("first_settlement", "settlements[0].usage", "settlements[-1].usage"),
    ("reissue_bound", "attempts >= bound", "attempts > bound"),
    ("resource_owner", "return owned and not local_in_flight", "return not local_in_flight"),
    ("resource_live", "return owned and not local_in_flight", "return owned"),
)


def run_calibration() -> dict[str, object]:
    """Mutants must fail hand-derived assertions, not import or generation."""
    root = Path(__file__).parent
    source = (root / "predicates.py").read_text()
    tests = (root / "test_predicates.py").read_text()
    original = sys.modules["tests.graph_sdk.reference.foundation_v2.predicates"]
    outcomes: dict[str, object] = {}
    for name, before, after in (("baseline", "", ""), *MUTATIONS):
        if name != "baseline" and before not in source:
            raise ValueError(f"mutation target missing: {name}")
        changed = source if name == "baseline" else source.replace(before, after)
        module = types.ModuleType("tests.graph_sdk.reference.foundation_v2.predicates")
        sys.modules["tests.graph_sdk.reference.foundation_v2.predicates"] = module
        exec(compile(changed, f"predicates-{name}.py", "exec"), module.__dict__)
        namespace: dict[str, object] = {}
        exec(compile(tests, "test_predicates.py", "exec"), namespace)
        failures = []
        count = 0
        for test_name, function in namespace.items():
            if test_name.startswith("test_") and callable(function):
                count += 1
                try:
                    cast(Callable[[], None], function)()
                except AssertionError:
                    failures.append(test_name)
        if (name == "baseline" and failures) or (name != "baseline" and not failures):
            raise ValueError(f"bad calibration: {name}: {failures}")
        outcomes[name] = {
            "tests": count,
            "assertion_failures": failures,
            "source_sha256": hashlib.sha256(changed.encode()).hexdigest(),
        }
    sys.modules["tests.graph_sdk.reference.foundation_v2.predicates"] = original
    first, second = encode_corpus(), encode_corpus()
    if first != second:
        raise ValueError("regeneration differs")
    (root / "cases.json").write_bytes(first)
    return {
        "python": sys.version,
        "method": "plain test function execution; no pytest dependency",
        "mutations": outcomes,
        "regeneration_sha256": [hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest()],
        "family_counts": dict(Counter(row["family"] for row in generate_cases())),
    }


if __name__ == "__main__":
    Path(__file__).with_name("evidence.json").write_text(json.dumps(run_calibration(), indent=2) + "\n")
