# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys

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
